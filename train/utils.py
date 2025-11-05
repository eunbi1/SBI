import torch
import torchvision.utils as tvu

# -----------------------------
# 1) 와류(= curl z) 계산
# -----------------------------
@torch.no_grad()
def vorticity_from_uv(x_uv: torch.Tensor, dx: float = 1.0, dy: float = 1.0, periodic: bool = True) -> torch.Tensor:
    """
    x_uv: [B, 2, H, W]  (u,v) = (x방향, y방향 성분)
    반환: w = ∂v/∂x − ∂u/∂y  → [B, H, W]
    """
    assert x_uv.dim() == 4 and x_uv.size(1) == 2, f"Expected [B,2,H,W], got {tuple(x_uv.shape)}"
    u = x_uv[:, 0]  # [B,H,W]
    v = x_uv[:, 1]  # [B,H,W]

    if periodic:
        dv_dx = (torch.roll(v, -1, dims=-1) - torch.roll(v,  1, dims=-1)) / (2.0 * dx)  # ∂v/∂x
        du_dy = (torch.roll(u, -1, dims=-2) - torch.roll(u,  1, dims=-2)) / (2.0 * dy)  # ∂u/∂y
    else:
        # 경계는 전방/후방 차분, 내부는 중앙차분
        B, H, W = u.shape
        dv_dx = torch.empty_like(v)
        du_dy = torch.empty_like(u)
        # dv/dx (W 방향)
        dv_dx[..., 1:-1] = (v[..., 2:] - v[..., :-2]) / (2.0 * dx)
        dv_dx[..., 0]    = (v[..., 1]  - v[..., 0])   / dx
        dv_dx[..., -1]   = (v[..., -1] - v[..., -2])  / dx
        # du/dy (H 방향)
        du_dy[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dy)
        du_dy[..., 0,    :] = (u[..., 1, :] - u[..., 0,   :])  / dy
        du_dy[..., -1,   :] = (u[..., -1, :] - u[..., -2, :])  / dy

    w = dv_dx - du_dy
    return w  # [B,H,W]


# -----------------------------
# 2) 스칼라장 → 컬러(RGB)로 변환
# -----------------------------
@torch.no_grad()
def colorize_scalar(
    w: torch.Tensor,                  # [B,H,W]
    cmap: str = "turbo",
    vmin: float = None, vmax: float = None,
    robust: bool = True, q: float = 0.995
) -> torch.Tensor:
    """
    스칼라장 w를 [B,3,H,W] RGB로 변환 (matplotlib 컬러맵 사용).
    - vmin/vmax를 주면 그 범위로, 아니면 robust 퍼센타일로 자동 스케일링.
    """
    import numpy as np
    import matplotlib.cm as cm

    assert w.dim() == 3, f"Expected [B,H,W], got {tuple(w.shape)}"
    B, H, W = w.shape

    if (vmin is None) or (vmax is None):
        if robust:
            # 양끝값 튀는 것 줄이려고 절대값 q-분위로 대칭 스케일
            a = w.abs().reshape(-1)
            vmax = torch.quantile(a, q).item() if a.numel() > 0 else 1.0
            vmax = max(vmax, 1e-8)
            vmin = -vmax
        else:
            vmin = float(w.min().item())
            vmax = float(w.max().item())
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1e-8

    wn = (w - vmin) / (vmax - vmin + 1e-8)  # [0,1]
    wn = wn.clamp(0, 1)

    # matplotlib로 컬러맵 적용 (CPU numpy로 변환 후 다시 torch)
    cmap_fn = cm.get_cmap(cmap)
    rgb = cmap_fn(wn.detach().cpu().numpy())[..., :3]       # [B,H,W,3], float32
    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
    return rgb


# -----------------------------
# 3) 배치 와류 컬러 PNG 저장
# -----------------------------
@torch.no_grad()
def save_vorticity_batch_color(
    x_uv: torch.Tensor,              # [B,2,H,W]
    png_path: str,
    denorm=None,                     # denorm 함수가 있으면 먼저 적용
    cmap: str = "turbo",
    periodic: bool = True,
    dx: float = 1.0, dy: float = 1.0,
    vmin: float = None, vmax: float = None,
    robust: bool = True, q: float = 0.995,
    nrow: int = None,
):
    # 1) denorm → 2) 와류 → 3) 컬러화 → 4) 저장
    if denorm is not None:
        x_uv = denorm(x_uv)

    w = vorticity_from_uv(x_uv, dx=dx, dy=dy, periodic=periodic)   # [B,H,W]
    w_rgb = colorize_scalar(w, cmap=cmap, vmin=vmin, vmax=vmax, robust=robust, q=q)  # [B,3,H,W]

    if nrow is None:
        nrow = w_rgb.size(0)
    tvu.save_image(w_rgb.cpu(), png_path, nrow=nrow)


# -----------------------------
# 4) GT vs 예측을 두 칸씩 컬러 PNG 저장
# -----------------------------
@torch.no_grad()
def save_vorticity_pairs_color(
    gt_uv: torch.Tensor, pred_uv: torch.Tensor,  # 각 [B,2,H,W]
    png_path: str,
    denorm=None,
    cmap: str = "turbo",
    periodic: bool = True,
    dx: float = 1.0, dy: float = 1.0,
    vmin: float = None, vmax: float = None,
    robust: bool = True, q: float = 0.995,
):
    if denorm is not None:
        print('gt_uv',gt_uv.min(),gt_uv.max(),gt_uv.std(),gt_uv.mean())
        gt_uv   = denorm(gt_uv)
        print('gt_uv',gt_uv.min(),gt_uv.max(),gt_uv.std(),gt_uv.mean())
        pred_uv = denorm(pred_uv)

    w_gt   = vorticity_from_uv(gt_uv,   dx=dx, dy=dy, periodic=periodic)  # [B,H,W]
    w_pred = vorticity_from_uv(pred_uv, dx=dx, dy=dy, periodic=periodic)

    # 같은 스케일로 비교되도록 vmin/vmax를 맞춰줌(둘 합쳐 robust 스케일)
    if (vmin is None) or (vmax is None):
        if robust:
            a = torch.cat([w_gt.abs().reshape(-1), w_pred.abs().reshape(-1)], dim=0)
            vmax = torch.quantile(a, q).item() if a.numel() > 0 else 1.0
            vmax = max(vmax, 1e-8)
            vmin = -vmax
        else:
            vmin = float(min(w_gt.min().item(),  w_pred.min().item()))
            vmax = float(max(w_gt.max().item(),  w_pred.max().item()))
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1e-8

    w_gt_rgb   = colorize_scalar(w_gt,   cmap=cmap, robust=False)  # [B,3,H,W]
    w_pred_rgb = colorize_scalar(w_pred, cmap=cmap, robust=False)  # [B,3,H,W]

    # [B,2,3,H,W] → [2B,3,H,W], nrow=2 로 저장(좌:GT | 우:Pred)
    pairs = torch.stack([w_gt_rgb, w_pred_rgb], dim=1).flatten(0, 1)  # [2B,3,H,W]
    tvu.save_image(pairs.cpu(), png_path, nrow=2)
# x_gt, x_hat: [B,2,128,128]  (u,v),  denorm: 데이터 역정규화 함수

# # 배치 전체 와류를 컬러로 한 장에
# save_vorticity_batch_color(x_t, "pred_vorticity_color.png", denorm, cmap="turbo", periodic=True)

# # GT vs Pred를 2열로 비교 저장
# save_vorticity_pairs_color(x_t, torch.randn_like(x_t), "gt_vs_pred_vorticity_color.png",denorm,
#                             cmap="turbo", periodic=True)


@torch.no_grad()
def save_vorticity_pairs_color(
    gt_uv: torch.Tensor,                 # [B,2,H,W]
    pred_uv: torch.Tensor,               # [B,2,H,W]
    png_path: str,
    denorm=None,
    noisy_uv: torch.Tensor = None,       # [B,2,H,W] (옵션) ← 추가
    cmap: str = "turbo",
    periodic: bool = True,
    dx: float = 1.0, dy: float = 1.0,
    vmin: float = None, vmax: float = None,
    robust: bool = True, q: float = 0.995,
):
    """
    GT | (Noisy) | Pred 순으로 vorticity를 컬러로 저장.
    - noisy_uv가 주어지면 3-열, 아니면 2-열로 저장.
    - denorm이 있으면 u,v에 먼저 역정규화 적용 후 와류 계산.
    """
    # 1) (옵션) 역정규화
    if denorm is not None:
        gt_uv   = denorm(gt_uv)
        pred_uv = denorm(pred_uv)
        if noisy_uv is not None:
            noisy_uv = denorm(noisy_uv)

    # 2) vorticity 계산
    w_gt   = vorticity_from_uv(gt_uv,   dx=dx, dy=dy, periodic=periodic)  # [B,H,W]
    w_pred = vorticity_from_uv(pred_uv, dx=dx, dy=dy, periodic=periodic)  # [B,H,W]
    w_noisy = None
    if noisy_uv is not None:
        w_noisy = vorticity_from_uv(noisy_uv, dx=dx, dy=dy, periodic=periodic)  # [B,H,W]

    # 3) 스케일(선택) — 세 세트 모두를 고려해 robust 스케일 산출
    if (vmin is None) or (vmax is None):
        if robust:
            pools = [w_gt.abs().reshape(-1), w_pred.abs().reshape(-1)]
            if w_noisy is not None:
                pools.append(w_noisy.abs().reshape(-1))
            a = torch.cat(pools, dim=0)
            vmax = torch.quantile(a, q).item() if a.numel() > 0 else 1.0
            vmax = max(vmax, 1e-8)
            vmin = -vmax
        else:
            vals_min = [w_gt.min().item(),  w_pred.min().item()]
            vals_max = [w_gt.max().item(),  w_pred.max().item()]
            if w_noisy is not None:
                vals_min.append(w_noisy.min().item())
                vals_max.append(w_noisy.max().item())
            vmin = float(min(vals_min)); vmax = float(max(vals_max))
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1e-8

    # 4) 컬러맵 적용 (colorize_scalar가 내부 정규화를 한다면 vmin/vmax는 참고용)
    w_gt_rgb   = colorize_scalar(w_gt,   cmap=cmap, robust=False)  # [B,3,H,W]
    w_pred_rgb = colorize_scalar(w_pred, cmap=cmap, robust=False)  # [B,3,H,W]
    w_noisy_rgb = None
    if w_noisy is not None:
        w_noisy_rgb = colorize_scalar(w_noisy, cmap=cmap, robust=False)  # [B,3,H,W]

    # 5) 배치 차원으로 쌓아서 그리드 저장
    #    - 2열: [GT, Pred]  → nrow=2
    #    - 3열: [GT, Noisy, Pred] → nrow=3
    if w_noisy_rgb is None:
        tiles = torch.stack([w_gt_rgb, w_pred_rgb], dim=1).flatten(0, 1)   # [2B,3,H,W]
        nrow = 2
    else:
        tiles = torch.stack([w_gt_rgb, w_noisy_rgb, w_pred_rgb], dim=1).flatten(0, 1)  # [3B,3,H,W]
        nrow = 3

    tvu.save_image(tiles.cpu(), png_path, nrow=nrow)