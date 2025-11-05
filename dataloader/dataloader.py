import os
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Z-Score Normalizer (채널별). 4D([B,C,H,W]) / 5D([B,T,C,H,W]) 자동 지원
# ============================================================
class ZScorePerChannel:
    """
    채널별 z-score 정규화.
    - 초기 data에서 channel_dim 기준으로 μ, σ를 (C,)로 추정해 저장
    - 이후 입력 x가 4D/5D든 상관없이 normalize_auto/denormalize_auto로 자동 채널축 추정
    - 필요시 normalize_on/denormalize_on으로 채널축을 명시 지정 가능
    """
    def __init__(self, data: torch.Tensor, channel_dim: int = 1, eps: float = 1e-6):
        assert isinstance(channel_dim, int)
        self.channel_dim = channel_dim
        self.eps = eps

        reduce_dims = tuple(d for d in range(data.ndim) if d != channel_dim)
        self.mu  = data.mean(dim=reduce_dims)                       # (C,)
        self.std = data.std(dim=reduce_dims).clamp_min(eps)         # (C,)
        self.C = int(self.mu.shape[0])

    # ---------- 내부 유틸 ----------
    def _broadcast_stats(self, x: torch.Tensor, channel_dim: int):
        assert x.shape[channel_dim] == self.C, \
            f"channel size mismatch: x has {x.shape[channel_dim]}, stats have {self.C}"
        shape = [1] * x.ndim
        shape[channel_dim] = self.C
        mu  = self.mu.to(device=x.device, dtype=x.dtype).view(shape)
        std = self.std.to(device=x.device, dtype=x.dtype).view(shape)
        return mu, std

    def _apply_affine(self, x: torch.Tensor, channel_dim: int, op: str):
        mu, std = self._broadcast_stats(x, channel_dim)
        if op == "norm":
            return (x - mu) / std
        elif op == "denorm":
            return x * std + mu
        else:
            raise ValueError(op)

    # ---------- 고정 채널축 버전 ----------
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_affine(x, self.channel_dim, op="norm")

    def denormalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self._apply_affine(x_hat, self.channel_dim, op="denorm")

    # ---------- 임의 채널축 명시 ----------
    def normalize_on(self, x: torch.Tensor, channel_dim: int) -> torch.Tensor:
        return self._apply_affine(x, channel_dim, op="norm")

    def denormalize_on(self, x_hat: torch.Tensor, channel_dim: int) -> torch.Tensor:
        return self._apply_affine(x_hat, channel_dim, op="denorm")

    # ---------- 자동 채널축 추정 (이미지/시퀀스) ----------
    def _guess_channel_dim(self, x: torch.Tensor) -> int:
        # 일반: [B,C,H,W] → 1, [B,T,C,H,W] → 2
        if x.ndim == 4:
            return 1
        if x.ndim == 5:
            return 2
        # fallback: 길이가 C인 축을 찾음
        candidates = [i for i in range(x.ndim) if x.shape[i] == self.C]
        if len(candidates) >= 1:
            return candidates[0]
        raise ValueError(
            f"Cannot infer channel dim. x.shape={tuple(x.shape)}, expected channel size C={self.C}"
        )

    def normalize_auto(self, x: torch.Tensor) -> torch.Tensor:
        cd = self._guess_channel_dim(x)
        return self.normalize_on(x, channel_dim=cd)

    def denormalize_auto(self, x_hat: torch.Tensor) -> torch.Tensor:
        cd = self._guess_channel_dim(x_hat)
        return self.denormalize_on(x_hat, channel_dim=cd)

    # ---------- 직렬화 ----------
    def state_dict(self):
        return {
            "mu": self.mu,
            "std": self.std,
            "C": self.C,
            "channel_dim": self.channel_dim,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.mu = state["mu"]
        obj.std = state["std"]
        obj.C = state["C"]
        obj.channel_dim = state.get("channel_dim", 1)
        obj.eps = state.get("eps", 1e-6)
        return obj


# ============================================================
# Min-Max Normalizer (글로벌). [-1,1] 범위 보장
# ============================================================
class MinMaxSymmetricNormalizer:
    """
    전체 데이터(global)에 대해 [min, max]를 추정하고 [-1,1] 범위로 정규화/복원.
    - 초기 data에서 전체 min, max를 구함 (채널별 아님)
    - normalize_auto / denormalize_auto: 4D([B,C,H,W]) / 5D([B,T,C,H,W]) 자동 지원
    """
    def __init__(self, data: torch.Tensor, eps: float = 1e-6):
        self.min = data.min()
        self.max = data.max()
        self.eps = eps
        self.range = (self.max - self.min).clamp_min(eps)

    # ---------- core ----------
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # [min,max] -> [0,1] -> [-1,1]
        return 2 * ((x - self.min) / self.range) - 1

    def denormalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [0,1] -> [min,max]
        return ((x_hat + 1) / 2) * self.range + self.min

    # ---------- auto wrappers ----------
    def normalize_auto(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def denormalize_auto(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.denormalize(x_hat)

    # ---------- 직렬화 ----------
    def state_dict(self):
        return {
            "min": self.min,
            "max": self.max,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.min = state["min"]
        obj.max = state["max"]
        obj.eps = state.get("eps", 1e-6)
        obj.range = (obj.max - obj.min).clamp_min(obj.eps)
        return obj


# ============================================================
# 시간 유틸
# ============================================================
def make_time_vector(total_steps: int, dt: float, start_index: int = 0) -> torch.Tensor:
    """
    return: (T,) — 실제 물리 시간값. 여기서 T = total_steps + 1 - start_index
    """
    idx = torch.arange(start_index, total_steps + 1, dtype=torch.float32)
    return idx * dt


# ============================================================
# 측정(관측) 배치 적용: 입력·출력 모두 [B,T,C,H,W]
# ============================================================
def batched_measure(measurement, traj_BTCHW: torch.Tensor) -> torch.Tensor:
    """
    measurement.measure(x): x는 [B',C,H, W] 배치를 받는다고 가정
    input:  traj_BTCHW — [B,T,C,H,W]
    output: z_BTCHW    — [B,T,C,H,W]
    """
    assert traj_BTCHW.ndim == 5 and traj_BTCHW.shape[2] >= 1, "expect (B,T,C,H,W)"
    B, T, C, H, W = traj_BTCHW.shape
    x = traj_BTCHW.reshape(B * T, C, H, W)
    z = measurement.measure(x)                     # [B*T, C, H, W]
    z = z.reshape(B, T, C, H, W).contiguous()
    return z


# ============================================================
# 데이터 생성/저장/로드 (저장 형식: [B,T,C,H,W])
# ============================================================
def build_dataset(
    dynamics,
    measurement,
    n_sample: int,
    steps: int,
    spinup: int = 0,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    device: Union[torch.device, str] = "cpu",
):
    """
    KolmogorovFlow 등으로 (B,T,C,H,W) states/observations 생성.
    - dynamics.prior(n_sample=B) -> (B,C,H,W)
    - dynamics.generate(x0=prior, steps=S) -> (S+1, B, C, H, W)  # 시간축이 맨 앞 (가정)
    - spinup>0이면 앞쪽 S=spinup 구간을 drop → T = steps+1
    - 저장/로드는 항상 (B,T,C,H,W)로 정렬
    """
    if isinstance(device, str):
        device = torch.device(device)

    if save_path and load_if_exists and os.path.isfile(save_path):
        pkg = torch.load(save_path, map_location=device)
        return pkg

    # 1) prior (B,C,H,W)
    prior = dynamics.prior(n_sample=n_sample).to(device)  # [B,C,H,W]

    # 2) 전체 스텝 생성 후 spinup 절단
    total_steps = steps + spinup
    # dynamics.generate: (total_steps+1, B, C, H, W) — 시간축 맨 앞 가정
    states_full = dynamics.generate(x0=prior, steps=total_steps)        # [total_steps+1, B, C, H, W]
    print('states_full',states_full.shape)
    states_TBCHW = states_full[spinup:]                                 # [steps+1, B, C, H, W]
    # → [B, T, C, H, W]로 전치
    print('states_TBCHW',states_TBCHW.shape)
    states = states_TBCHW.permute(1, 0, 2, 3, 4).contiguous()           # [B, T, C, H, W]
    B, T, C, H, W = states.shape

    # 3) 관측 생성 (동일 순서 [B,T,C,H,W])
    observations = batched_measure(measurement, states)                 # [B, T, C, H, W]
    print('shapeeee',states.shape, observations.shape)
    # 4) 물리 시간 벡터 (길이 T)
    t = make_time_vector(total_steps=total_steps, dt=dynamics.dt, start_index=spinup)  # (T,)

    pkg = {
        "prior": prior,                   # [B,C,H,W]
        "states": states,                 # [B,T,C,H,W]
        "observations": observations,     # [B,T,C,H,W]
        "t": t,                           # (T,)
        "meta": {
            "grid_size": getattr(dynamics, "grid_size", None),
            "reynolds": getattr(dynamics, "reynolds", None),
            "dt": dynamics.dt,
            "spinup": spinup,
            "steps": steps,
            "n_sample": n_sample,
            "measurement": type(measurement).__name__,
            "noise_std": getattr(measurement, "noise_std", None),
            "mask_stride": getattr(measurement, "stride", None),
        },
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(pkg, save_path)

    return pkg


# ============================================================
# Dataset: (B,T,C,*S) → (x_t, z_t, z_{t-s:t-1}, mask, t, t_ctx)
# ※ 저장 순서를 [B,T,...]로 사용
# ============================================================
class StateObsDataset(Dataset):
    """
    (B,T,C,*S) → (x_t, z_t, z_{t-s:t-1}, ctx_mask, t, t_ctx)
      - context_len = s: x_t 직전 s개 관측만 포함 (t 미포함)
      - require_full_ctx=True: t>=s만 사용(항상 정확히 s개)
      - require_full_ctx=False: t<s 허용, 앞쪽 0-패딩 + ctx_mask로 유효표시(1=PAD, 0=VALID)
      - t_vec: (T,) 실제 물리 시간값 (float)
    """
    def __init__(
        self,
        states_BTCS: torch.Tensor,  # [B,T,C,*S]
        obs_BTCS: torch.Tensor,     # [B,T,C,*S]
        t_vec: torch.Tensor,        # (T,)
        context_len: int = 8,
        require_full_ctx: bool = False,
        pad_time_value: float = 0.0,
    ):
        assert states_BTCS.shape == obs_BTCS.shape, "states/obs shape mismatch"
        assert states_BTCS.ndim >= 4, "expect (B,T,C,*S)"
        self.states = states_BTCS
        self.obs    = obs_BTCS
        self.t_vec  = t_vec.to(dtype=torch.float32, device=states_BTCS.device)  # (T,)

        shape = self.states.shape
        print('states',shape)
        self.B, self.T, self.C = shape[0], shape[1], shape[2]
        self.S = tuple(shape[3:])

        self.s = int(context_len)
        self.require_full_ctx = bool(require_full_ctx)
        self.pad_time_value = float(pad_time_value)

        if self.s < 0:
            raise ValueError("context_len (s) must be >= 0")
        if self.t_vec.shape[0] != self.T:
            raise ValueError(f"t_vec length {self.t_vec.shape[0]} must equal T={self.T}")

        # 사용할 (b, t) 인덱스 구성
        if self.require_full_ctx:
            if self.T <= self.s:
                raise ValueError(
                    f"require_full_ctx=True 인데 T={self.T} <= s={self.s} → 샘플 없음."
                )
            t_start = self.s
        else:
            t_start = 0

        self.index = [(b, t) for b in range(self.B) for t in range(t_start, self.T)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        b, t = self.index[idx]

        x_t   = self.states[b, t]  # (C,*S)
        z_t   = self.obs[b, t]     # (C,*S)
        t_val = self.t_vec[t]      # () scalar

        if self.require_full_ctx:
            start = t - self.s
            z_ctx  = self.obs[b, start:t]               # (s,C,*S)
            t_ctx  = self.t_vec[start:t]                # (s,)
            ctx_mask = torch.zeros(self.s, dtype=torch.uint8, device=z_ctx.device)
        else:
            start   = max(0, t - self.s)
            z_hist  = self.obs[b, start:t]              # (real_len<=s, C,*S)
            t_hist  = self.t_vec[start:t]               # (real_len,)
            real_len = z_hist.shape[0]

            if real_len < self.s:
                pad_len = self.s - real_len
                # 앞쪽 PAD
                pad_shape = (pad_len, self.C, *self.S)
                z_pad  = torch.zeros(pad_shape, dtype=z_hist.dtype, device=z_hist.device)
                z_ctx  = torch.cat([z_pad, z_hist], dim=0)  # (s,C,*S)

                t_pad  = torch.full((pad_len,), self.pad_time_value,
                                    dtype=self.t_vec.dtype, device=self.t_vec.device)
                t_ctx  = torch.cat([t_pad, t_hist], dim=0)  # (s,)

                ctx_mask = torch.cat([
                    torch.ones (pad_len, dtype=torch.uint8, device=z_hist.device),  # 1=PAD
                    torch.zeros(real_len, dtype=torch.uint8, device=z_hist.device), # 0=VALID
                ], dim=0)
            else:
                z_ctx  = z_hist
                t_ctx  = t_hist
                ctx_mask = torch.zeros(self.s, dtype=torch.uint8, device=z_hist.device)

        return {
            "x_t": x_t,                 # (C,*S)
            "z_t": z_t,                 # (C,*S)
            "z_ctx": z_ctx,             # (s,C,*S)
            "ctx_mask": ctx_mask,       # (s,)   0=VALID, 1=PAD
            "t": t_val,                 # ()  실시간값 scalar
            "t_ctx": t_ctx,             # (s,) 컨텍스트 시점들의 실시간값
            "b_idx": torch.tensor(b, dtype=torch.long, device=x_t.device),
        }


# ============================================================
# prepare_dataset_and_loader:
#   - 생성/로드 → [B,T,C,H,W]
#   - Normalization 선택 적용 (norm_type='zscore' 또는 'minmax_symmetric')
#   - Dataset/DataLoader 구성
#   - norm/denorm 함수는 4D/5D 모두 자동 처리
# ============================================================
def prepare_dataset_and_loader(
    dynamics,
    measurement,
    cfg_steps: int,
    spinup: int,
    n_sample: int,
    batch_size: int,
    context_len: int,
    save_path: Optional[str] = None,
    load_if_exists: bool = True,
    device: Union[torch.device, str] = "cpu",
    shuffle: bool = True,
    num_workers: int = 0,
    require_full_ctx: bool = True,
    apply_normalize: bool = True,     # 정규화 on/off
    pad_time_value: float = 0.0,      # t_ctx 패딩 값(마스크로 무시)
    norm_type: str = "zscore",  # 'zscore' or 'minmax_symmetric'
):
    # 1) 원데이터 (저장·로드 모두 [B,T,C,H,W])
    pkg = build_dataset(
        dynamics=dynamics,
        measurement=measurement,
        n_sample=n_sample,
        steps=cfg_steps,
        spinup=spinup,
        save_path=save_path,
        load_if_exists=load_if_exists,
        device=device,
    )
    states       = pkg["states"]        # [B,T,C,H,W]
    observations = pkg["observations"]  # [B,T,C,H,W]
    t_vec        = pkg["t"].to(states.device)  # (T,)

    # 2) 정규화 선택
    if apply_normalize:
        if norm_type.lower() == "zscore":
            znorm = ZScorePerChannel(states, channel_dim=2)  # [B,T,**C**,H,W]
            states_n = znorm.normalize_on(states, channel_dim=2)
            obs_n    = znorm.normalize_on(observations, channel_dim=2)
            print('norm type is zscore')
            pkg["norm"] = {
                "type": "zscore_per_channel",
                "mu": znorm.mu, "std": znorm.std,
                "channel_dim": 2,
            }
            norm_fn   = znorm.normalize_auto
            denorm_fn = znorm.denormalize_auto

        elif norm_type.lower() == "minmax_symmetric":
            print('norm type is minmax')
            mm = MinMaxSymmetricNormalizer(states)  # global min/max
            states_n = mm.normalize_auto(states)
            obs_n    = mm.normalize_auto(observations)
            pkg["norm"] = {
                "type": "minmax_symmetric",
                "min": mm.min, "max": mm.max,
            }
            norm_fn   = mm.normalize_auto
            denorm_fn = mm.denormalize_auto

        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    else:
        states_n, obs_n = states, observations
        norm_fn   = lambda x: x
        denorm_fn = lambda x: x

    # 3) Dataset/DataLoader
    ds = StateObsDataset(
        states_BTCS=states_n,
        obs_BTCS=obs_n,
        t_vec=t_vec,                         # (T,)
        context_len=context_len,
        require_full_ctx=require_full_ctx,
        pad_time_value=pad_time_value,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return pkg, ds, loader, norm_fn, denorm_fn