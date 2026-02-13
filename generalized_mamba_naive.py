# Naive (for-loop) generalized Mamba2 implementation.
# Extends Mamba2Naive by letting the (x, B, C, dt) projections depend on the previous timestep hidden state.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch._dynamo as _dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint



@_dynamo.disable  # graph break around the Python for-loop scan
def _iterative_scan_forward(
    m: nn.Module,
    B: int,
    L: int,
    u_3d: torch.Tensor,
    A: torch.Tensor,
    z0: torch.Tensor,
    x0: torch.Tensor,
    z: torch.Tensor,
    xBC_u_full: torch.Tensor,
    dt_raw_u: torch.Tensor,
    ssm_state: torch.Tensor,
    y_prev: torch.Tensor,
) -> torch.Tensor:
    if L == 0:
        return u_3d.new_empty(B, 0, m.d_model)

    debug_numerics = os.environ.get("HYBRID_DEBUG_NUMERICS", "").strip() not in ("", "0", "false", "False")

    def _check(name: str, t: int, x: torch.Tensor) -> None:
        if torch.isfinite(x).all():
            return
        maxabs = float(x.detach().abs().max().item()) if x.numel() else float("nan")
        raise FloatingPointError(f"Non-finite tensor in generalized_mamba scan: {name} at t={t} (max|x|={maxabs})")

    x_dtype = u_3d.dtype
    state_dtype = ssm_state.dtype
    split_sizes = (m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state)
    head_to_group = m._head_to_group
    A_row = A.unsqueeze(0)

    # Keep dt bias cast and clamp checks outside the scan loop.
    dt_bias = m.dt_bias.to(dtype=dt_raw_u.dtype)
    dt_lo, dt_hi = m.dt_limit
    clamp_dt = not (dt_lo == 0.0 and dt_hi == float("inf"))

    if m.D_has_hdim:
        D_term = m.D.view(m.nheads, m.headdim).to(dtype=x_dtype)
    else:
        D_term = m.D.view(1, m.nheads, 1).to(dtype=x_dtype)

    out = None
    for t in range(L):
        xBC = xBC_u_full[:, t, :]
        dt_raw_t = dt_raw_u[:, t, :]

        if m.use_recurrent_xbc:
            xBC = F.silu(xBC + m.recurrent_xbc(y_prev).to(dtype=xBC.dtype))
        else:
            xBC = F.silu(xBC)
        if m.use_recurrent_dt:
            dt_raw_t = F.silu(dt_raw_t + m.recurrent_dt(y_prev).to(dtype=dt_raw_t.dtype))

        z0_t = z0[:, t, :]
        x0_t = x0[:, t, :]
        z_t = z[:, t, :]

        x_t, Bvec, Cvec = torch.split(xBC, split_sizes, dim=-1)

        # ----- SSM step (explicit recurrence) -----
        dt = F.softplus(dt_raw_t + dt_bias)
        if clamp_dt:
            dt = dt.clamp(min=dt_lo, max=dt_hi)
        dt_f = dt.to(torch.float32)  # (B, nheads) float32
        dA = torch.exp(dt_f * A_row)  # (B, nheads) float32

        x_hp_in = x_t.view(B, m.nheads, m.headdim)
        x_hp = x_hp_in.to(torch.float32)  # (B, h, p)
        B_gn = Bvec.view(B, m.ngroups, m.d_state).to(torch.float32)  # (B, g, n)
        C_gn = Cvec.view(B, m.ngroups, m.d_state).to(torch.float32)  # (B, g, n)

        B_hn = B_gn[:, head_to_group, :]  # (B, h, n)
        C_hn = C_gn[:, head_to_group, :]  # (B, h, n)

        ssm_state_f = ssm_state.to(torch.float32)
        ssm_state_f = ssm_state_f * dA[:, :, None, None] + (dt_f[:, :, None, None] * x_hp[:, :, :, None] * B_hn[:, :, None, :])
        ssm_state = ssm_state_f.to(state_dtype)
        if debug_numerics:
            _check("ssm_state_f", t, ssm_state_f)

        y_hp = (ssm_state_f * C_hn[:, :, None, :]).sum(dim=-1)  # (B, h, p) float32
        y_hp = y_hp.to(x_dtype)
        if debug_numerics:
            _check("y_hp", t, y_hp)

        # skip connection via D
        if m.D_has_hdim:
            y_hp = y_hp + D_term.unsqueeze(0) * x_hp_in
        else:
            y_hp = y_hp + D_term * x_hp_in

        y = y_hp.reshape(B, m.d_ssm)
        # y_prev = y  # feed hidden output into next timestep's x/B/C/dt computation
        if debug_numerics:
            _check("y_prev", t, y_prev)

        # gating / norm
        if m.rmsnorm:
            y = m.norm(y, z_t.to(dtype=y.dtype))
        else:
            y = y * F.silu(z_t).to(dtype=y.dtype)
        y_prev = y  # feed hidden output into next timestep's x/B/C/dt computation
        if debug_numerics:
            _check("y_after_gate", t, y)

        # optional gated MLP part
        if m._d_mlp > 0:
            mlp = F.silu(z0_t).to(dtype=y.dtype) * x0_t.to(dtype=y.dtype)
            y = torch.cat([mlp, y], dim=-1)  # (B, d_inner)

        out_t = m.out_proj(y)
        if out is None:
            out = torch.empty(B, L, m.d_model, device=out_t.device, dtype=out_t.dtype)
        out[:, t, :] = out_t

    return out


class RMSNormGatedTorch(nn.Module):
    """
    Pure PyTorch replacement for mamba_ssm.ops.triton.layernorm_gated.RMSNorm (gated).

    Parameter compatibility: has .weight and .eps, and forward(y, z) -> tensor
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
        group_size: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size  # if None: standard RMSNorm on last dim
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is None:
            ms = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(ms + self.eps)

        dim = x.shape[-1]
        if dim % self.group_size != 0:
            raise ValueError("dim must be divisible by group_size")
        g = dim // self.group_size
        xg = x.view(*x.shape[:-1], g, self.group_size)
        ms = xg.pow(2).mean(dim=-1, keepdim=True)
        xg = xg * torch.rsqrt(ms + self.eps)
        return xg.view(*x.shape)

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gate = F.silu(z)
        if self.norm_before_gate:
            y = self._rmsnorm(y) * gate
        else:
            y = self._rmsnorm(y * gate)
        return y * self.weight


class GeneralizedMamba2Naive(nn.Module):
    """
    Generalized Mamba2 (naive for-loop) where x/B/C/dt at time t can depend on h_{t-1}.

    Concretely, we compute the usual Mamba2 u-dependent path:
      (z0, x0, z, xBC_u, dt_raw_u) = in_proj(u_t) + causal depthwise conv
    then add recurrent terms derived from the previous SSM hidden output y_{t-1}:
      xBC_t   = xBC_u_t   + R_xBC(y_{t-1})
      dt_raw_t = dt_raw_u_t + R_dt(y_{t-1})   (optional)

    This keeps the SSM recurrence unchanged, but breaks Mamba's parallelizability (intentionally).
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
        ngroups: int = 1,
        A_init_range: Tuple[int, int] = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
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
        use_compiled_scan: bool = True,
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
        self.ngroups = ngroups
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"

        self.use_recurrent_xbc = use_recurrent_xbc
        self.use_recurrent_dt = use_recurrent_dt

        if self.d_ssm % self.headdim != 0:
            raise ValueError("d_ssm must be divisible by headdim")
        self.nheads = self.d_ssm // self.headdim
        if self.d_ssm % self.ngroups != 0:
            raise ValueError("d_ssm must be divisible by ngroups")
        if self.nheads % self.ngroups != 0:
            raise ValueError("nheads must be divisible by ngroups")

        d_mlp = self.d_inner - self.d_ssm
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        self.conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
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

        # recurrent terms derived from previous y (shape: d_ssm)
        if self.use_recurrent_xbc:
            self.recurrent_xbc = nn.Linear(
                self.d_ssm, self.conv_dim, bias=recurrent_bias, **factory_kwargs
            )
            nn.init.zeros_(self.recurrent_xbc.weight)
            if self.recurrent_xbc.bias is not None:
                nn.init.zeros_(self.recurrent_xbc.bias)
        else:
            self.register_parameter("recurrent_xbc", None)

        if self.use_recurrent_dt:
            self.recurrent_dt = nn.Linear(
                self.d_ssm, self.nheads, bias=recurrent_bias, **factory_kwargs
            )
            nn.init.zeros_(self.recurrent_dt.weight)
            if self.recurrent_dt.bias is not None:
                nn.init.zeros_(self.recurrent_dt.bias)
        else:
            self.register_parameter("recurrent_dt", None)

        # dt_bias init (same as original)
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A_log init (same idea as original)
        if A_init_range[0] <= 0 or A_init_range[1] < A_init_range[0]:
            raise ValueError("Invalid A_init_range")
        A = torch.empty(self.nheads, device=device, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        # D skip
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            group_size = self.d_ssm // self.ngroups
            self.norm = RMSNormGatedTorch(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=group_size,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # head -> group mapping (contiguous blocks of heads)
        heads_per_group = self.nheads // self.ngroups
        head_to_group = torch.arange(self.nheads, device=device) // heads_per_group
        self.register_buffer("_head_to_group", head_to_group, persistent=False)

        self._d_mlp = d_mlp
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _pre_loop(self, u_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input projection (u-dependent)
        zxbcdt = self.in_proj(u_3d)  # (B, L, d_in_proj)
        z0, x0, z, xBC_in, dt_raw_u = torch.split(
            zxbcdt,
            [self._d_mlp, self._d_mlp, self.d_ssm, self.conv_dim, self.nheads],
            dim=-1,
        )

        # causal depthwise conv across the sequence for u-dependent path
        xBC_u_full = self._causal_depthwise_conv1d(xBC_in).to(dtype=u_3d.dtype)
        return z0, x0, z, xBC_u_full, dt_raw_u

    def allocate_inference_cache(self, batch_size: int, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.conv_dim, self.d_conv, device=device, dtype=conv_dtype)

        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype)

        y_prev = torch.zeros(batch_size, self.d_ssm, device=device, dtype=ssm_dtype)
        return conv_state, ssm_state, y_prev

    def _dt(self, dt_raw: torch.Tensor) -> torch.Tensor:
        dt = F.softplus(dt_raw + self.dt_bias.to(dtype=dt_raw.dtype))
        lo, hi = self.dt_limit
        if not (lo == 0.0 and hi == float("inf")):
            dt = dt.clamp(min=lo, max=hi)
        return dt

    def _causal_depthwise_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) where C == conv_dim
        x_bct = x.to(dtype=self.conv1d.weight.dtype).transpose(1, 2)  # (B, C, T)
        weight_cw = self.conv1d.weight.squeeze(1)  # (C, W)
        bias = self.conv1d.bias


        y = F.conv1d(
            x_bct,
            self.conv1d.weight,
            bias=bias,
            padding=self.d_conv - 1,
            groups=x_bct.shape[1],
        )
        y = y[..., : x_bct.shape[-1]]
        return y.transpose(1, 2).to(dtype=x.dtype)

    def forward(self, u: torch.Tensor, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
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

        # constants
        A = -torch.exp(self.A_log.float())  # (nheads,) float32

        z0, x0, z, xBC_u_full, dt_raw_u = self._pre_loop(u_3d)

        # SSM state and previous hidden output (y_{t-1}); conv cache is unused in full-sequence mode.
        ssm_state = torch.zeros(
            B,
            self.nheads,
            self.headdim,
            self.d_state,
            device=u_3d.device,
            dtype=self.in_proj.weight.dtype,
        )
        y_prev = torch.zeros(B, self.d_ssm, device=u_3d.device, dtype=u_3d.dtype)

        out = _iterative_scan_forward(
            self,
            B,
            L,
            u_3d,
            A,
            z0,
            x0,
            z,
            xBC_u_full,
            dt_raw_u,
            ssm_state,
            y_prev,
        )
        if flatten_out:
            out = out.reshape(B * L, self.d_model)
        return out


@dataclass
class GeneralizedMamba2NaiveConfig:
    hidden_size: int = 768
    d_state: int = 128
    d_conv: int = 4
    conv_init: Any = None
    expand: int = 2
    head_dim: int = 64
    d_ssm: Optional[int] = None
    ngroups: int = 1
    intermediate_size: int = 3072
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-6
    A_init_range: Tuple[int, int] = (1, 16)
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
    bias: bool = False
    conv_bias: bool = True
    device: Any = None
    dtype: Optional[torch.dtype] = None
    use_recurrent_xbc: bool = True
    use_recurrent_dt: bool = True
    recurrent_bias: bool = False
    use_compiled_scan: bool = True

    @property
    def d_model(self) -> int:
        return self.hidden_size


class GeneralizedMamba2NaiveBlock(nn.Module):
    def __init__(self, config: GeneralizedMamba2NaiveConfig) -> None:
        super().__init__()
        from .transformer import TransformerMLP

        self.config = config
        self.gradient_checkpointing = False
        self.mamba = GeneralizedMamba2Naive(
            d_model=config.hidden_size,
            d_state=config.d_state,
            d_conv=config.d_conv,
            conv_init=config.conv_init,
            expand=config.expand,
            headdim=config.head_dim,
            d_ssm=config.d_ssm,
            ngroups=config.ngroups,
            A_init_range=config.A_init_range,
            D_has_hdim=config.D_has_hdim,
            rmsnorm=config.rmsnorm,
            norm_before_gate=config.norm_before_gate,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            dt_limit=config.dt_limit,
            bias=config.bias,
            conv_bias=config.conv_bias,
            device=config.device,
            dtype=config.dtype,
            use_recurrent_xbc=config.use_recurrent_xbc,
            use_recurrent_dt=config.use_recurrent_dt,
            recurrent_bias=config.recurrent_bias,
            use_compiled_scan=config.use_compiled_scan,
        )
        self.mlp = TransformerMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def gradient_checkpointing_enable(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)
        self.mamba.gradient_checkpointing_enable(enabled=enabled)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        self.mamba.gradient_checkpointing_disable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.mamba(x)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        # Keep checkpointing benefits for the MLP, but avoid checkpointing the scan loop above.
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            x = checkpoint(self.mlp, x, use_reentrant=False)
        else:
            x = self.mlp(x)
        return residual + x
