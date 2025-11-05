import torch
import math


def _expand_to_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """t: [B], x: [B,...] → [B,1,1,...] 로 reshape (브로드캐스트용)"""
    shape = (t.shape[0],) + (1,) * (x.ndim - 1)
    return t.view(shape)

def beta(t,cosine_s: float = 0.008):
    beta = math.pi/2*2/(cosine_s+1)*torch.tan( (t+cosine_s)/(1+cosine_s)*math.pi/2 )
    beta = torch.clamp(beta,0,20)
    return beta

def cosine_log_alpha(t: torch.Tensor, cosine_s: float = 0.008):
    """
    log α_t = log( cos( (t+s)/(1+s) * π/2 ) ) - log( cos( s/(1+s) * π/2 ) )
    """
    log_alpha_0 = math.log(math.cos(cosine_s / (1.0 + cosine_s) * math.pi / 2.0))
    v = (t + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0)
    return torch.log(torch.cos(v)) - log_alpha_0


def diffusion_coeff(t: torch.Tensor, cosine_s: float = 0.008):
    return torch.exp(cosine_log_alpha(t, cosine_s))


def marginal_std(t: torch.Tensor, cosine_s: float = 0.008):
    """σ_t = sqrt(1 - α_t^2)"""
    a_t = diffusion_coeff(t, cosine_s)
    return torch.sqrt(torch.clamp(1.0 - a_t ** 2, min=0.0))


def forward_process(
    x: torch.Tensor,
    t: torch.Tensor,
    cosine_s: float = 0.008,
):
    """
      x_t = α_t * x + σ_t * ε,  ε ~ N(0, I)
    x: [B, ...]
    t: [B], 보통 0~T (T≈0.9946)
    """
    assert x.shape[0] == t.shape[0], "Batch size of x and t must match."

    a_t = diffusion_coeff(t, cosine_s)
    s_t = marginal_std(t, cosine_s)

    a_t_bc = _expand_to_x(a_t, x)
    s_t_bc = _expand_to_x(s_t, x)

    noise = torch.randn_like(x)
    x_t = a_t_bc * x + s_t_bc * noise
    return x_t, noise

def noise_to_x0(epsilon, x, t, cosine_s = 0.008):
    a_t = diffusion_coeff(t, cosine_s)
    s_t = marginal_std(t, cosine_s)

    a_t_bc = _expand_to_x(a_t, epsilon)
    s_t_bc = _expand_to_x(s_t, epsilon)
    inv_a = torch.pow(1e-5 + a_t_bc, -1)       # (B,)
    return (-s_t_bc*epsilon + x)*inv_a

# if __name__ == "__main__":
#     B, C, H, W = 4, 3, 32, 32
#     x = torch.randn(B, C, H, W, requires_grad=True)
#     t = torch.rand(B) * 0.9946  # cosine 스케줄 상한 T≈0.9946

#     x_t = forward_add_noise_cosine(x, t)
#     print(x_t.shape)