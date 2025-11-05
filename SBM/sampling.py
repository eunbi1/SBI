import torch
from .diffusions import * 

def _expand_like(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    v: (B,) 같은 배치 1D 텐서
    x: (B, ... ) 임의 차원
    반환: v를 x에 곱할 수 있도록 (B,1,1,...)로 reshape
    """
    return v.view(v.shape[0], *([1] * (x.ndim - 1)))

@torch.no_grad()
def sampler(model,x_t_tau_s,t, tau_s, tau_t, z_t, z_ctx, ctx_mask, t_ctx):
    """
    x_s: (B, *D)
    y  : 모델의 conditioning
    s,t: (B,)
    """
    # 미리 계산(스칼라 벡터)
    dc  = diffusion_coeff(tau_s)                 # (B,)
    ms  = marginal_std(tau_s)                    # (B,)
    # inv_var = torch.pow(1e-5 + ms, -2) 
    inv = torch.pow(1e-5 + ms, -1)       # (B,)

    # 브로드캐스팅 준비
    dc_x      = _expand_like(dc, x_t_tau_s) # (B,C,H,W)
    # inv_var_x = _expand_like(inv_var, x_t_tau_s) # (B,C,H,W)
    inv= _expand_like(inv, x_t_tau_s) 

    # score_s: (B,*D)
    # 모델 호출 1회만
    # score_s = (dc_x * model(x_t_tau_s, t, tau_s, z_t, z_ctx, ctx_mask,t_ctx) - x_t_tau_s) * inv_var_x # (B,C,H,W)
    score_s = (-model(x_t_tau_s, t, tau_s, z_t, z_ctx, ctx_mask,t_ctx) ) * inv

    # time/beta step
    time_step  = tau_s - tau_t                       # (B,)
    beta_s     = beta(tau_s)                     # (B,)
    beta_step  = beta_s * time_step          # (B,)

    x_coeff     = 1.0 + 0.5 * beta_step      # (B,)
    score_coeff = beta_step                  # (B,)
    # 수치안정(음수 루트 방지)
    noise_coeff = torch.sqrt(torch.clamp(beta_step, min=0.0))

    # 노이즈 1회 생성
    e = torch.randn_like(x_t_tau_s)

    # 최종 업데이트 (브로드캐스팅)
    x_t = _expand_like(x_coeff, x_t_tau_s)     * x_t_tau_s \
        + _expand_like(score_coeff, x_t_tau_s) * score_s \
        + _expand_like(noise_coeff, x_t_tau_s) * e
    return x_t


def sample(
    model,
    observations,
    ctx_mask,
    dim=(2,128,128),                               
    batch_size=4,
    nfe=100,
    seq_le=16,
    device='cpu',
    verbose = True):
    """
    result:
        trajectory: [B, T, C, H, W]
        xT_mean:    [T, C, H, W]  (마지막 프레임의 배치 평균)
    """
    model.eval()
    x_shape = (batch_size, *tuple(dim)) # [B,C,T,W]
    T = observations.shape[0]

    tausteps = torch.linspace(0.9946, 1e-4, nfe + 1, device=device)  #(nfe+1,)
    times = torch.linspace(10,20,51,  device=device)
    

    C, H, W = dim
    trajectory = torch.empty(batch_size, T, C, H, W, device=device) # [B,T,C,H,W]

    # 컨텍스트 저장소(이전 예측 x)
    pred_list = []  # list of [B,C,H,W] (이미지 가정)

    for i in range(T):
        # For each t, we want to sample x_t ~ p_t(x_t|z_{1:t})
        # z_t = observation[t] # [1,2,128,128]
        # z_{1:t-1} = observation[:t] [1,t-1,2,128,128]

        x = torch.randn(x_shape, device=device) #[B, C, H, W]
        z_t = observations[i].repeat((batch_size, 1, 1, 1)) #[B, C, H, W]

        z_ctx = torch.zeros((batch_size, seq_le, C, H, W), device=device) #[B, T, C, H, W]
        seq_mask = torch.ones((batch_size, seq_le),device=device) #[B, T]
        t_ctx = torch.zeros((batch_size,seq_le))  #[B, T]

        z_ctx[:,seq_le-i:] = observations[:i].repeat((batch_size, 1, 1, 1, 1))  #[B, T, C, H, W]
        seq_mask[:,seq_le-i:] = torch.zeros((batch_size,i))
        t_ctx[:,seq_le-i:] = times[:i].unsqueeze(0).repeat((batch_size,1))

        t=times[i]*torch.ones(batch_size) #[B,]

        for j in range(nfe):
            tau_s = tausteps[j]*torch.ones(batch_size) #[B,]
            tau_t = tausteps[j + 1]*torch.ones(batch_size) #[B,]
            x = sampler(model,x,t, tau_s, tau_t, z_t, z_ctx, ctx_mask, t_ctx)
            if verbose:
                k_min = float(x.min().item()); k_max = float(x.max().item())
                print(f"{i}th time, {j}th nfe x range=({k_min:.4f}, {k_max:.4f})")

        # Store
        trajectory[:,i] = x 
        pred_list.append(x)

    mean = trajectory.mean(dim=1, keepdim=True)

    return mean