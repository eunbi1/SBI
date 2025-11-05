import datetime
import json
import os
import click
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter  # <- 제거
import wandb

from dynamics import KolmogorovFlow, Lorenz96
from measurements import AveragePooling, CenterMask, GridMask, Linear, RandomMask
from models import ConditionalUNetModel  # UNet1D를 쓰면 여기에도 추가로 import
from train import trainer
from src.utils import plot_kolmogorov_vorticity, plot_lorenz_trajectory


# --------------------------- W&B logger wrapper ---------------------------
class WandbLogger:
    """
    Trainer가 기대하는 SummaryWriter-like API를 W&B로 매핑:
      - add_scalar(tag, value, step)
      - add_image(tag, image, step)  # torch.Tensor도 받아서 wandb.Image로 변환
      - close()
    """
    def __init__(self, project: str, name: str, workdir: str, config: dict):
        self.run = wandb.init(project=project, name=name, dir=workdir, config=config)

    def add_scalar(self, tag: str, value: float, step: int):
        # W&B는 dict로 로그, step은 통일성 위해 명시
        self.run.log({tag: float(value)}, step=step)

    def add_image(self, tag: str, image, step: int):
        # image가 torch.Tensor(CHW) 또는 HWC/np.ndarray여도 OK
        if torch.is_tensor(image):
            img = image.detach().cpu()
            self.run.log({tag: wandb.Image(img)}, step=step)
        else:
            self.run.log({tag: wandb.Image(image)}, step=step)

    def close(self):
        wandb.finish()


@click.group()
def main():
    pass


@main.command()
@click.option("-c", "--config", type=str, default="configs/lorenz96.json", help="path to json configs")
@click.option("--description", type=str, default="lorenz96", help="prefix of work directory")
@click.option("-s", "--seed", type=int, default=42, help="global random seed")
@click.option("-d", "--device", type=str, default="cuda:0", help="PyTorch device")
def lorenz96(config, description, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    with open(config, "r") as f:
        cfg = json.load(f)

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_name = f"{description}-{timestamp}"
    workdir = os.path.join("lorenz_results", run_name)
    os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=False)
    print(f"Results will be saved in {workdir}...")
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(cfg, f)

    dynamics = Lorenz96(
        dim=cfg["dynamics"]["dim"],
        prior_mean=cfg["dynamics"]["prior_mean"],
        prior_std=cfg["dynamics"]["prior_std"],
        dt=cfg["dynamics"]["dt"],
        forcing=cfg["dynamics"]["forcing"],
        perturb_std=cfg["dynamics"]["perturb_std"],
        solver=cfg["dynamics"]["solver"],
    )

    measurement = Linear(noise_std=cfg["measurement"]["noise_std"])

    # Generate GT/obs
    x0 = cfg["dynamics"]["forcing"] * torch.ones((1, cfg["dynamics"]["dim"]), device=device)  # (1, dim)
    x0[0][0] += 0.01
    states: torch.Tensor = dynamics.generate(
        x0=x0,
        steps=cfg["dynamics"]["steps"],
    )  # (steps+1, dim)
    observations: torch.Tensor = measurement.measure(states)

    # Prior guess
    prior: torch.Tensor = dynamics.prior(n_sample=cfg["train"]["n_train"]).to(device)  # (n_train, dim)

    # --- 네트워크 (원 코드대로 사용) ---
    # from models import UNet1D 가 필요하면 위 import에 추가하세요.
    model: nn.Module = UNet1D(
        in_channels=cfg["network"]["in_channels"],
        out_channels=cfg["network"]["out_channels"],
        channels=cfg["network"]["channels"],
    ).to(device)

    # ---- W&B logger ----
    project = cfg.get("wandb_project", description)  # 없으면 description을 프로젝트명으로
    logger = WandbLogger(project=project, name=run_name, workdir=workdir, config=cfg)

    assimilated_states: torch.Tensor = trainer(
        workdir=workdir,
        device=device,
        logger=logger,  # <- W&B 래퍼 전달
        dynamics=dynamics,
        measurement=measurement,
        model=model,
        prior=prior,
        states=states,
        observations=observations,
        batch_size=cfg["train"]["batch_size"],
        n_epoch=cfg["train"]["n_epoch"],
        lr=cfg["train"]["lr"],
        denoising_sigma=cfg["train"]["denoising_sigma"],
        lmc_steps=cfg["langevin"]["steps"],
        lmc_stepsize=cfg["langevin"]["stepsize"],
        anneal_init=cfg["langevin"]["anneal_init"],
        anneal_decay=cfg["langevin"]["anneal_decay"],
        anneal_steps=cfg["langevin"]["anneal_steps"],
        plot_callback=plot_lorenz_trajectory,
    )  # (steps+1, n_train, dim)

    np.savez(
        os.path.join(workdir, "results.npz"),
        states=states.cpu().numpy(),             # (steps, dim)
        observations=observations.cpu().numpy(), # (steps, dim)
        assimilated_states=assimilated_states.cpu().numpy(),  # (steps, n_train, dim)
    )

    # (선택) 결과 파일을 아티팩트로 남기고 싶다면:
    # wandb.save(os.path.join(workdir, "results.npz"))

    logger.close()

import os, json, datetime, glob
import numpy as np
import click, wandb
import torch

# ... (위쪽 import/정의들은 기존과 동일)

def _latest_ckpt_path(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    return cands[-1] if cands else None

def _load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    # 표준 키 권장: model, optimizer, scheduler, ema, epoch, global_step, rng
    return ckpt

def _try_trainer_with_resume(trainer_fn, resume_state, **kwargs):
    """
    trainer가 resume 인자를 지원하면 전달, 아니면 무시하고 호출.
    반환값은 기존과 동일하게 assimilated_states로 가정.
    """
    try:
        return trainer_fn(resume_state=resume_state, **kwargs)
    except TypeError:
        # resume_state 인자를 모르면 빼고 호출
        return trainer_fn(**kwargs)

def _save_checkpoint(path: str, model, optimizer=None, scheduler=None, ema=None,
                     epoch=None, global_step=None, extra: dict=None):
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if ema is not None:
        # ema가 torch.nn.Module 형태/래퍼 형태 등 다양할 수 있어 예외 처리
        try:
            state["ema"] = ema.state_dict()
        except Exception:
            pass
    if extra:
        state.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)



@main.command()
@click.option("-c", "--config", type=str, default="configs/kolmogorov.json", help="path to json configs")
@click.option("--description", type=str, default="kolmogorov", help="prefix of work directory")
@click.option("-s", "--seed", type=int, default=42, help="global random seed")
@click.option("-d", "--device", type=str, default="cuda:0", help="PyTorch device")
def kolmogorov(config, description, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    with open(config, "r") as f:
        cfg = json.load(f)

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_name = f"{description}-{timestamp}"
    workdir = os.path.join("kolmogorov_results", run_name)
    os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=False)
    print(f"Results will be saved in {workdir}...")
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(cfg, f)

    dynamics = KolmogorovFlow(
        grid_size=cfg["dynamics"]["grid_size"],
        reynolds=cfg["dynamics"]["reynolds"],
        dt=cfg["dynamics"]["dt"],
        seed=seed,
    )

    if cfg["measurement"]["type"] == "AvgPool":
        measurement = AveragePooling(
        
            noise_std=cfg["measurement"]["noise_std"],
            kernel_size=cfg["measurement"]["kernel_size"],
        )

    elif cfg["measurement"]["type"] == "GridMask":
        measurement = GridMask(
            image_size=cfg["dynamics"]["grid_size"],
            noise_std=cfg["measurement"]["noise_std"],
            stride=cfg["measurement"]["stride"],
        )

    elif cfg["measurement"]["type"] == "CenterMask":
        measurement = CenterMask(
            noise_std=cfg["measurement"]["noise_std"],
        )
        
    elif cfg["measurement"]["type"] == "RandomMask":
        measurement = RandomMask(
            noise_std=cfg["measurement"]["noise_std"],
            sparsity=cfg["measurement"]["sparsity"],
        )
    elif cfg["measurement"]["type"] == "Linear":
        measurement = Linear(noise_std=cfg["measurement"]["noise_std"])

    # Generate GT/obs (50-step shift)
    x0: torch.Tensor = dynamics.prior(n_sample=1).to(device)  # (1, 2, H, W)
    states: torch.Tensor = dynamics.generate(
        x0=x0,
        steps=cfg["dynamics"]["steps"] + 50,
    )[50:, ...][:, 0]  # (steps, 2, H, W)
    observations: torch.Tensor = measurement.measure(states)

    # Prior guess
    prior: torch.Tensor = dynamics.prior(n_sample=cfg["train"]["n_train"]).to(device)  # (n_train, 2, H, W)
#parameter tuning how to?
    model = ConditionalUNetModel(
        image_size=cfg["dynamics"]["grid_size"],
        in_channels=2,
        model_channels=128,
        out_channels=2,
        num_res_blocks=2,
        channel_mult=(1, 2, 4,8),#(1, 2, 4, 8),
        attention_resolutions=(8, 4),
        dropout=0.1,
        num_heads=64,
        num_head_channels=-1,

        t_range = [10, 20],
        t_learnable_affine = True, 

        inst_obs_dim=(2, cfg["dynamics"]["grid_size"], cfg["dynamics"]["grid_size"]),
        seq_obs_dim=(2, cfg["dynamics"]["grid_size"], cfg["dynamics"]["grid_size"]),
        seq_model_dim=256,
        seq_nheads=8,
        seq_layers=4,
        add_inst_token=True,

        cross_attention_resolutions=(8, 4),
        cross_attn_heads=8,

        seq_pe_dim=64,
        seq_pool="mean",
        seq_t_abs_range=(10,20),  # 예시 범위(원하면 config로)
        seq_t_rel_range=None,

        zero_out=False,
    ).to(device)

    # ---- W&B logger ----
    project = cfg.get("wandb_project", description)
    
    logger = WandbLogger(project=project, name=run_name, workdir=workdir, config=cfg)
    wandb.config.update({
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_mps_available": torch.backends.mps.is_available(),})
    wandb.log({
        "device_type": 0 if torch.cuda.is_available() else (1 if torch.backends.mps.is_available() else 2)
    })
    print("Using device:", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    assimilated_states: torch.Tensor = trainer(
        workdir=workdir,
        device=device,
        logger=logger,  # <- W&B 래퍼
        dynamics=dynamics,
        measurement=measurement,
        model=model,
        prior=prior,
        states=states,
        observations=observations,
        batch_size=cfg["train"]["batch_size"],
        n_epoch=cfg["train"]["n_epoch"],
        lr=cfg["train"]["lr"]
    )  # (steps, n_train, 2, H, W)

    np.savez(
        os.path.join(workdir, "results.npz"),
        states=states.cpu().numpy(),             # (steps, 2, H, W)
        observations=observations.cpu().numpy(),
        assimilated_states=assimilated_states.cpu().numpy(),  # (steps, n_train, 2, H, W)
    )
    # wandb.save(os.path.join(workdir, "results.npz"))

    logger.close()


if __name__ == "__main__":
    main()

# 이거 resume 옵션있으면 ckpt저장된거 불러와서 이어서 학습하는 코드로 수정하기
# 학습 코드 손보기