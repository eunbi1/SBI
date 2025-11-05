import math
import os 

os.environ["JAX_PLATFORMS"] = "cpu"  

import jax
import jax.numpy as jnp
import jax.random as jrn
import jax_cfd.base as cfd
import numpy as np
import torch
from torch import Tensor

from .base import Dynamics




class KolmogorovFlow(Dynamics):
    """Kolmogorov flow dynamics.
    Reference: https://github.com/francois-rozet/sda/

    Args:
        grid_size (int): Size of per edge of the spatial grid.
        reynolds (float): Reynolds number.
        dt (float): Time steps intervals between observations.
        seed (int): RNG seed for jax (to generate initial prior states).
    """

    def __init__(
        self,
        grid_size: int = 128,
        reynolds: float = 1e3,
        dt: float = 0.2,
        seed: int = 42,
    ):
        super().__init__(shape=(2, grid_size, grid_size))
        self.seed = seed
        self.reynolds = reynolds
        self.dt = dt
        self.grid_size = grid_size
        grid = cfd.grids.Grid(
            shape=(grid_size, grid_size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )
        bc = cfd.boundaries.periodic_boundary_conditions(2)
        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type="kolmogorov",
        )
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds,
        )
        steps = 1 if dt_min > dt else math.ceil(dt / dt_min)
        step_fn = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps,
                density=1.0,
                viscosity=1 / reynolds,
            ),
            steps=steps,
        )

        def _prior(key):
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )
            return jnp.stack((u.data, v.data))

        self._prior = jax.jit(jnp.vectorize(_prior, signature="(K)->(C,H,W)"))

        def _transition(uv):
            u, v = cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )
            u, v = step_fn((u, v))
            return jnp.stack((u.data, v.data))

        self._transition = jax.jit(jnp.vectorize(_transition, signature="(C,H,W)->(C,H,W)"))

    def prior(self, n_sample):
        key = jrn.PRNGKey(self.seed)
        keys = jrn.split(key, n_sample)
        x = np.array(self._prior(keys))
        return torch.tensor(x)

    def transition(self, x: Tensor) -> Tensor:
        device = x.device
        x = x.detach().cpu().numpy()
        x = np.array(self._transition(x))
        return torch.tensor(x, device=device)

    def generate(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Generate a trajectory of spatial states (batched).

        Args:
            x0   : (..., *shape) initial spatial state(s).  앞쪽 모든 축은 배치로 간주
            steps: int, number of transition steps.

        Returns:
            states: (steps+1, ..., *shape)
                    첫 축이 시간, 그 다음이 입력 x0의 배치 차원들 그대로 유지됨.
        """
        device, dtype = x0.device, x0.dtype
        shape = tuple(self.shape)                    # e.g. (C,H,W)
        assert tuple(x0.shape[-len(shape):]) == shape, \
            f"x0 trailing dims must be {shape}, but got {tuple(x0.shape)}"

        batch_dims = x0.shape[:-len(shape)]          # may be empty ()
        states = torch.empty((steps + 1, *batch_dims, *shape),
                            device=device, dtype=dtype)
        states[0].copy_(x0)

        for t in range(steps):
            states[t + 1].copy_( self._apply_transition_batched(states[t]) )

        return states

    def _apply_transition_batched(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., *shape)
        1) self.transition가 배치 입력을 지원하면 그대로 사용
        2) 아니면 배치 축을 평탄화해 샘플별로 호출한 뒤 다시 원래 배치로 reshape
        """
        # 1) 벡터화 경로 시도
        try:
            y = self.transition(x)
            if y.shape == x.shape:
                return y
        except Exception:
            pass

        # 2) 안전 경로: 배치 평탄화 후 개별 처리
        shape = tuple(self.shape)
        flat = x.reshape(-1, *shape)
        outs = []
        for i in range(flat.shape[0]):
            outs.append(self.transition(flat[i]))
        y_flat = torch.stack(outs, dim=0)
        return y_flat.reshape(x.shape)