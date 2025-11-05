import os
import sys
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torchvision.utils as tvu

# === NEW: wandb 통합 헬퍼 ===
try:
    import wandb
except Exception:
    wandb = None  # wandb 미설치 환경에서도 안전하게 동작

from dynamics.base import Dynamics
from measurements.base import Measurement
from dataloader import prepare_dataset_and_loader
from SBM import forward_process, noise_to_x0

from train.utils import save_vorticity_pairs_color


# ----------------------------------------------------------------------
# 유틸: seed, 스케줄러, EMA
# ----------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * float(step) / float(self.warmup_steps)
            else:
                t = (step - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + torch.cos(torch.tensor(t * 3.1415926535))).item()
            lrs.append(lr)
        return lrs

class EmaModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = type(model)() if hasattr(model, "_init_weights") else None
        # 간단히 같은 구조의 새 모델을 만들 수 없으면 deepcopy 사용
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            if p_ema.dtype == p.dtype and p_ema.shape == p.shape:
                p_ema.mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ----------------------------------------------------------------------
# wandb/TensorBoard 공통 로깅 헬퍼
# ----------------------------------------------------------------------
def log_scalar(tag: str, value: float, step: int, logger: SummaryWriter = None):
    """TensorBoard와 wandb에 동시에 스칼라 로깅"""
    v = float(value)
    if logger is not None:
        logger.add_scalar(tag, v, step)
    if (wandb is not None) and (getattr(wandb, "run", None) is not None):
        wandb.log({tag: v}, step=step)

def log_image(tag: str, image_tensor: torch.Tensor, step: int, logger: SummaryWriter = None, nrow: int = 2):
    """
    image_tensor: [N,C,H,W] or [C,H,W]
    TensorBoard: make_grid로 기록
    wandb: wandb.Image로 기록
    """
    # TensorBoard
    if logger is not None:
        if image_tensor.dim() == 3:
            logger.add_image(tag, image_tensor, step)
        else:
            logger.add_image(tag, tvu.make_grid(image_tensor, nrow=nrow), step)
    # wandb
    if (wandb is not None) and (getattr(wandb, "run", None) is not None):
        if image_tensor.dim() == 3:
            wandb.log({tag: wandb.Image(image_tensor)}, step=step)
        else:
            grid = tvu.make_grid(image_tensor, nrow=nrow)
            wandb.log({tag: wandb.Image(grid)}, step=step)


# ----------------------------------------------------------------------
# helper: evaluate loop (NO condition dropout)  — AMP & EMA 지원
# ----------------------------------------------------------------------
@torch.no_grad()
def _evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    forward_process: Callable[[Tensor, Tensor], Tensor],
    split_name: str,
    global_step: int,
    logger: SummaryWriter,
    use_amp: bool = True,
    ema_model: Optional[EmaModel] = None,
) -> float:
    model_to_eval = ema_model.ema if (ema_model is not None) else model
    model_to_eval.eval()

    total, count = 0.0, 0
    for batch in loader:
        x_t  = batch["x_t"    ].to(device)                         # [B,C,H,W]
        z_t  = batch.get("z_t", None)
        if z_t is not None: z_t = z_t.to(device)                   # [B,C,H,W] or None

        z_ctx    = batch.get("z_ctx", None)
        if z_ctx is not None: z_ctx = z_ctx.to(device)             # [B,S,C,H,W] or None
        ctx_mask = batch.get("ctx_mask", None)
        if ctx_mask is not None: ctx_mask = ctx_mask.to(device)    # [B,S] (bool/uint8)

        t     = batch["t"].to(device).float()                      # [B]
        t_ctx = batch.get("t_ctx", None)
        if t_ctx is not None: t_ctx = t_ctx.to(device).float()     # [B,S] or None

        B = x_t.shape[0]
        tau = torch.rand(B, device=device) * 0.9946                # [B]
        x_noisy, noise = forward_process(x_t, tau)
        
        #debug
        z_t = None
        z_ctx = None
        ctx_mask = None
        t_ctx = None
        if use_amp:
            with torch.cuda.amp.autocast():
                denoised = model_to_eval(
                    x_noisy, t, tau,
                    z_t=z_t,
                    z_seq=z_ctx, seq_mask=ctx_mask, seq_lens=None,
                    t_ctx=t_ctx
                )
                loss = nn.MSELoss()(denoised, noise)
        else:
            denoised = model_to_eval(
                x_noisy, t, tau,
                z_t=z_t,
                z_seq=z_ctx, seq_mask=ctx_mask, seq_lens=None,
                t_ctx=t_ctx
            )
            loss = nn.MSELoss()(denoised, noise)

        total += float(loss.item()) * B
        count += B

    avg = total / max(count, 1)
    # --- 여기서 TB + wandb 모두 로깅 ---
    log_scalar(f"{split_name}/loss", avg, global_step, logger)
    return avg


# ----------------------------------------------------------------------
# main trainer — 시각화는 그대로 유지
# ----------------------------------------------------------------------
def trainer(
    workdir: str,
    device: torch.device,
    logger: SummaryWriter,
    dynamics,
    measurement,
    model: nn.Module,
    prior: Tensor,
    states: Tensor,
    observations: Tensor,
    batch_size: int,
    n_epoch: int,
    lr: float,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    ema_decay: float = 0.999,
    warmup_steps: int = 5000,

):
    """
    - train/val/test split
    - val/test loss 비교 및 로깅 (TensorBoard + wandb)
    - classifier-free style 조건 드롭
    - AMP + EMA + WarmupCosine 스케줄러
    - 시각화(이미지 저장/로깅) 절대 삭제하지 않음
    """
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "figs"), exist_ok=True)

    # === 데이터 로드 (정규화/denorm 반환)
    kg, dataset_or_ds, full_loader, norm, denorm = prepare_dataset_and_loader(
        dynamics, measurement,
        cfg_steps=50, spinup=50, n_sample=400,
        batch_size=batch_size, context_len=51,
        save_path="./data/kf_gridmask_s3.pt",
        require_full_ctx=False,
        apply_normalize=True,
        norm_type= "zscore")

    if isinstance(full_loader, DataLoader):
        base_dataset = full_loader.dataset
        base_bs      = full_loader.batch_size
        base_num_workers = getattr(full_loader, "num_workers", 0)
        base_pin     = getattr(full_loader, "pin_memory", False)
        base_collate = getattr(full_loader, "collate_fn", None)
    else:
        base_dataset = dataset_or_ds
        base_bs, base_num_workers, base_pin, base_collate = batch_size, 0, False, None

    N = len(base_dataset)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(base_dataset, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(train_set, batch_size=base_bs, shuffle=True,
                              num_workers=base_num_workers, pin_memory=base_pin, collate_fn=base_collate, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=base_bs, shuffle=False,
                              num_workers=base_num_workers, pin_memory=base_pin, collate_fn=base_collate, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=base_bs, shuffle=False,
                              num_workers=base_num_workers, pin_memory=base_pin, collate_fn=base_collate, drop_last=False)

    # (옵션) 시뮬레이션 데이터 생성 (원 코드 유지)
    x0: torch.Tensor = dynamics.prior(n_sample=1).to(device)  # (1, 2, H, W)
    states: torch.Tensor = dynamics.generate(x0=x0, steps=50 + 50)[50:, ...][:, 0]
    observations: torch.Tensor = denorm(measurement.measure(states))
    batch = next(iter(train_loader))
    print('x0',x0.shape, x0.min(),x0.max())
    # Optimizer / Scheduler / AMP / EMA
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)

    # 전체 스텝 수 추정 (rough) — cosine 스케줄러용
    est_total_steps = max(1, n_epoch * (len(train_loader) // max(1, grad_accum_steps)))
    scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps, max_steps=est_total_steps, min_lr=lr * 0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EmaModel(model, decay=ema_decay)

    global_step = 0
    best_val = float("inf")
    best_ckpt_path = os.path.join(workdir, "ckpt", "best.pt")

    fixed_vis_batch = next(iter(val_loader)) if len(val_loader) > 0 else None

    # classifier-free 스타일 조건 드롭 확률
    p_drop_inst = 0.2   # (z_t, t)
    p_drop_seq  = 0.2   # (z_ctx, t_ctx)

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------

    # -------------------------
    # Train loop (정리 버전)
    # -------------------------
    for epoch in tqdm(range(1, n_epoch + 1), mininterval=5.0, maxinterval=50.0,
                    leave=False, desc="epoch", file=sys.stdout):
        model.train()
        epoch_loss = 0.0
        num_seen   = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_no, batch in enumerate(train_loader, start=1):
            # --------- batch to device ---------
            x_t = batch["x_t"].to(device)     
            t = batch["t"].to(device).float()       

            #observation                            
            z_t = batch.get("z_t", None)  # [B,C,H,W]
            if z_t is not None:
                z_t = z_t.to(device)

            #observation sequence
            z_ctx = batch.get("z_ctx", None)
            if z_ctx is not None:
                z_ctx = z_ctx.to(device)

            ctx_mask = batch.get("ctx_mask", None)
            if ctx_mask is not None:
                ctx_mask = ctx_mask.to(device).bool()                # bool로 통일

                             # [B]
            t_ctx = batch.get("t_ctx", None)
            if t_ctx is not None:
                t_ctx = t_ctx.to(device).float()                     # [B,S]

            # --------- noise schedule ---------
            B   = x_t.shape[0]
            tau = torch.rand(B, device=x_t.device) * 0.9946          # [B]

            # --------- condition dropout ---------
            if torch.rand((), device=x_t.device).item() < p_drop_inst:
                z_t = None
            if torch.rand((), device=x_t.device).item() < p_drop_seq:
                z_ctx = None
                t_ctx = None
                ctx_mask = None

            # --------- forward process ---------
            x_noisy, noise = forward_process(x_t, tau)

            # debug
            z_t = None
            z_ctx = None 
            t_ctx = None 
            ctx_mask = None

            # --------- forward + loss ---------
            denoised_noise = model(
                x_noisy, t, tau,
                z_t=z_t,
                z_seq=z_ctx, seq_mask=ctx_mask, seq_lens=None,
                t_ctx=t_ctx
            )

            # --- 시각화(원 코드 유지) ---
            pred_x0 = noise_to_x0(denoised_noise, x_noisy, tau)
            save_vorticity_pairs_color(
                x_t, pred_x0, "gt_vs_pred_vorticity_color.png",
                cmap="turbo", periodic=True, noisy_uv=x_noisy
            )

            # ε-예측: MSE(ε̂, ε)
            loss = nn.MSELoss()(denoised_noise, noise) / max(1, grad_accum_steps)

            # --------- backward / step ---------
            if use_amp:  #Automatic Mixed Precision 
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = (batch_no % grad_accum_steps == 0)
            if do_step:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA & scheduler & step count
                ema.update(model)
                scheduler.step()
                global_step += 1
                log_scalar("train/lr", scheduler.get_last_lr()[0], global_step, logger)

            # --------- stats ---------
            epoch_loss += float(loss.item()) * B * max(1, grad_accum_steps)
            num_seen   += B

            if batch_no % 10 == 0:
                log_scalar("train/loss_iter",
                        float(loss.item()) * max(1, grad_accum_steps),
                        global_step, logger)

        train_avg = epoch_loss / max(num_seen, 1)
        log_scalar("train/loss_epoch", train_avg, epoch, logger)

        # --------------------------- VALID ---------------------------
        val_avg = _evaluate(
            model=model, device=device, loader=val_loader,
            forward_process=forward_process, split_name="val",
            global_step=epoch, logger=logger, use_amp=use_amp, ema_model=ema
        )
        log_scalar("val/loss_epoch", val_avg, epoch, logger)       # ★ 추가
        log_scalar("val/best_loss", best_val, epoch, logger)  

        # 베스트 저장 (EMA 기준)
        if val_avg < best_val:
            best_val = val_avg
            torch.save(ema.state_dict(), best_ckpt_path)  # EMA 가중치 저장
            log_scalar("val/best_loss", best_val, epoch, logger) 

        # -------------------- VISUALIZATION (PNG + 로그) --------------------
        if fixed_vis_batch is not None:
            model.eval()
            with torch.no_grad():
                x_t  = fixed_vis_batch["x_t"].to(device)
                t    = fixed_vis_batch["t"].to(device).float()
                z_t  = fixed_vis_batch.get("z_t", None)
                if z_t is not None: z_t = z_t.to(device)
                z_ctx = fixed_vis_batch.get("z_ctx", None)
                if z_ctx is not None: z_ctx = z_ctx.to(device)
                ctx_mask = fixed_vis_batch.get("ctx_mask", None)
                if ctx_mask is not None: ctx_mask = ctx_mask.to(device)
                t_ctx = fixed_vis_batch.get("t_ctx", None)
                if t_ctx is not None: t_ctx = t_ctx.to(device).float()

                B = x_t.shape[0]
                tau = torch.rand(B, device=device) * 0.9946
                x_noisy,noise = forward_process(x_t, tau)


        print(f"[epoch {epoch:03d}] train={train_avg:.6f}  val={val_avg:.6f}  best_val={best_val:.6f}")

    # --------------------------- TEST ---------------------------
    # EMA 가중치 로드하여 평가
    if os.path.isfile(best_ckpt_path):
        ema_sd = torch.load(best_ckpt_path, map_location=device)
        ema.load_state_dict(ema_sd)

    test_avg = _evaluate(
        model=model, device=device, loader=test_loader,
        forward_process=forward_process, split_name="test",
        global_step=n_epoch, logger=logger, use_amp=use_amp, ema_model=ema
    )
    print(f"[TEST] loss={test_avg:.6f}")

    # 최종 체크포인트 저장 (EMA와 현재 모델 둘 다)
    torch.save(ema.state_dict(), os.path.join(workdir, "ckpt", f"model_last_ema.pt"))
    torch.save(model.state_dict(), os.path.join(workdir, "ckpt", f"model_last.pt"))

    # 아래 반환/주석 블록은 원 코드의 형식 유지
    steps = observations.shape[0]
    n_train, *shape = prior.shape
    assimilated_states = torch.empty((steps, n_train, *shape), device=device)
    return assimilated_states