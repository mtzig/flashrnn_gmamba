"""
Generalized Mamba2 with per-head triton kernel for the SSM scan.

Key differences from GeneralizedMamba2Naive:
- ngroups == nheads (each head has independent B, C of size d_state)
- Recurrent weights are per-head (block-diagonal), enabling parallelism
- Forward scan uses a fused triton kernel (single launch for all T steps)
- Backward uses PyTorch with state recomputation

The architecture is otherwise identical: x/B/C/dt at time t depend on y_{t-1}.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashrnn.flashrnn.triton_fused.gmamba_fwbw import gmamba_fwbw


class GeneralizedMamba2(nn.Module):
    """
    Generalized Mamba2 with triton-accelerated SSM scan.

    Uses per-head recurrent weights and ngroups=nheads for efficient
    triton kernel execution. The scan loop runs in a single kernel launch.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        conv_init=None,
        expand: int = 2,
        headdim: int = 64,
        d_ssm: Optional[int] = None,
        A_init_range: Tuple[int, int] = (1, 16),
        D_has_hdim: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        bias: bool = False,
        conv_bias: bool = True,
        device=None,
        dtype=None,
        use_recurrent_xbc: bool = True,
        use_recurrent_dt: bool = True,
        recurrent_bias: bool = False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim

        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        # KEY: ngroups = nheads for per-head triton kernel
        self.nheads = self.d_ssm // self.headdim
        self.ngroups = self.nheads  # each head has its own B, C

        self.D_has_hdim = D_has_hdim
        self.dt_limit = dt_limit
        self.activation = "silu"

        self.use_recurrent_xbc = use_recurrent_xbc
        self.use_recurrent_dt = use_recurrent_dt

        if self.d_ssm % self.headdim != 0:
            raise ValueError("d_ssm must be divisible by headdim")

        d_mlp = self.d_inner - self.d_ssm
        # ngroups = nheads, so B and C are per-head
        # conv_dim per head = headdim + 2*d_state
        # total conv_dim = d_ssm + 2*nheads*d_state
        self.conv_dim_per_head = self.headdim + 2 * self.d_state
        self.conv_dim = self.nheads * self.conv_dim_per_head

        # Split: [d_mlp, d_mlp, d_ssm, conv_dim, nheads]
        # Sum = 2*d_mlp + d_ssm + conv_dim + nheads = 2*d_inner - d_ssm + conv_dim + nheads
        d_in_proj = 2 * self.d_inner - self.d_ssm + self.conv_dim + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.conv_dim,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        # Per-head recurrent weights
        if self.use_recurrent_xbc:
            # R_x: (NH, DH, DH), R_B: (NH, N, DH), R_C: (NH, N, DH)
            self.R_x = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.headdim, **factory_kwargs))
            self.R_B = nn.Parameter(torch.zeros(self.nheads, self.d_state, self.headdim, **factory_kwargs))
            self.R_C = nn.Parameter(torch.zeros(self.nheads, self.d_state, self.headdim, **factory_kwargs))
        else:
            self.register_parameter("R_x", None)
            self.register_parameter("R_B", None)
            self.register_parameter("R_C", None)

        if self.use_recurrent_dt:
            self.R_dt = nn.Parameter(torch.zeros(self.nheads, self.headdim, **factory_kwargs))
        else:
            self.register_parameter("R_dt", None)

        # dt_bias init
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A_log init
        if A_init_range[0] <= 0 or A_init_range[1] < A_init_range[0]:
            raise ValueError("Invalid A_init_range")
        A = torch.empty(self.nheads, device=device, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        # D skip
        self.D = nn.Parameter(torch.ones(self.nheads, self.headdim, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self._d_mlp = d_mlp

    def _causal_depthwise_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) where C == conv_dim"""
        x_bct = x.to(dtype=self.conv1d.weight.dtype).transpose(1, 2)
        y = F.conv1d(
            x_bct,
            self.conv1d.weight,
            bias=self.conv1d.bias,
            padding=self.d_conv - 1,
            groups=x_bct.shape[1],
        )
        y = y[..., : x_bct.shape[-1]]
        return y.transpose(1, 2).to(dtype=x.dtype)

    def _pre_loop(self, u_3d: torch.Tensor):
        """
        Compute input-dependent projections (same as naive, but with ngroups=nheads).

        Returns:
            z0: (B, L, d_mlp) - MLP gating
            x0: (B, L, d_mlp) - MLP input
            z:  (B, L, NH, DH) - SSM gating, reshaped per-head
            xBC_u: (B, L, NH, DH+2*N) - per-head x, B, C after conv
            dt_raw_u: (B, L, NH) - dt pre-activation
        """
        B, L, _ = u_3d.shape
        zxbcdt = self.in_proj(u_3d)  # (B, L, d_in_proj)
        z0, x0, z, xBC_in, dt_raw_u = torch.split(
            zxbcdt,
            [self._d_mlp, self._d_mlp, self.d_ssm, self.conv_dim, self.nheads],
            dim=-1,
        )

        # Causal depthwise conv
        xBC_u_flat = self._causal_depthwise_conv1d(xBC_in).to(dtype=u_3d.dtype)

        # Reshape to per-head: (B, L, conv_dim) -> (B, L, NH, DH+2*N)
        xBC_u = xBC_u_flat.view(B, L, self.nheads, self.conv_dim_per_head)

        # Reshape z to per-head: (B, L, d_ssm) -> (B, L, NH, DH)
        z = z.view(B, L, self.nheads, self.headdim)

        # dt_raw_u is already (B, L, NH)
        return z0, x0, z, xBC_u, dt_raw_u

    def forward(self, u: torch.Tensor, seqlen=None, **kwargs):
        if seqlen is None:
            B, L, _ = u.shape
            u_3d = u
            flatten_out = False
        else:
            BL, D = u.shape
            B = BL // seqlen
            L = seqlen
            u_3d = u.view(B, L, D)
            flatten_out = True

        # Negative A
        A = -torch.exp(self.A_log.float())

        # Pre-compute input-dependent terms
        z0, x0, z, xBC_u, dt_raw_u = self._pre_loop(u_3d)

        # Initial states
        ssm_state = torch.zeros(
            B, self.nheads, self.headdim, self.d_state,
            device=u_3d.device, dtype=u_3d.dtype,
        )
        y_prev = torch.zeros(
            B, self.nheads, self.headdim,
            device=u_3d.device, dtype=u_3d.dtype,
        )

        # Per-head recurrent weights (or dummy zeros)
        R_x = self.R_x if self.R_x is not None else torch.zeros(self.nheads, self.headdim, self.headdim, device=u_3d.device, dtype=u_3d.dtype)
        R_B = self.R_B if self.R_B is not None else torch.zeros(self.nheads, self.d_state, self.headdim, device=u_3d.device, dtype=u_3d.dtype)
        R_C = self.R_C if self.R_C is not None else torch.zeros(self.nheads, self.d_state, self.headdim, device=u_3d.device, dtype=u_3d.dtype)
        R_dt = self.R_dt if self.R_dt is not None else torch.zeros(self.nheads, self.headdim, device=u_3d.device, dtype=u_3d.dtype)

        # Run the triton-accelerated scan
        y_gated = gmamba_fwbw(
            xBC_u=xBC_u,
            dt_raw_u=dt_raw_u,
            z=z,
            R_x=R_x,
            R_B=R_B,
            R_C=R_C,
            R_dt=R_dt,
            dt_bias=self.dt_bias,
            A=A,
            D=self.D,
            ssm_state_init=ssm_state,
            y_prev_init=y_prev,
            use_recurrent_xbc=self.use_recurrent_xbc,
            use_recurrent_dt=self.use_recurrent_dt,
            dt_limit=self.dt_limit,
        )  # (B, L, NH, DH)

        # Reshape back: (B, L, NH, DH) -> (B, L, d_ssm)
        y = y_gated.reshape(B, L, self.d_ssm)

        # Optional gated MLP part
        if self._d_mlp > 0:
            mlp = F.silu(z0).to(dtype=y.dtype) * x0.to(dtype=y.dtype)
            y = torch.cat([mlp, y], dim=-1)  # (B, L, d_inner)

        out = self.out_proj(y)

        if flatten_out:
            out = out.reshape(B * L, self.d_model)
        return out
