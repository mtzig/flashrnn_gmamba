"""
Forward triton kernel for the Generalized Mamba SSM scan.

Mirrors the sLSTM kernel API pattern:
- Pre-computed input projections (xBC_u, dt_raw_u, z) passed in
- Per-head recurrent weights (R_x, R_B, R_C, R_dt) applied inside the kernel
- Full T-step sequential scan in a single kernel launch
- Grid: (NH, ceil(B/siz_B))

Key difference from sLSTM: SSM state is a matrix (DH, N) per (batch, head),
so we stream it from/to global memory in an inner loop over N (d_state).

Requirements:
- ngroups == nheads (each head has independent B, C of size d_state)
- DH (headdim) must be a power of 2
- siz_B >= 16 (for tensor core tl.dot)
"""

import torch
import triton
import triton.language as tl
from einops import rearrange
from triton import OutOfResources

from .triton_utils import is_power_of_2, next_multiple_of, torch2triton_dtype


ENABLE_AUTOTUNING = False

if ENABLE_AUTOTUNING:
    configs = [
        triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
        for siz_B in [16, 32, 64]
        for s in [1]
        for w in [1, 2, 4, 8]
    ]
else:
    configs = [
        triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
        for siz_B in [16]
        for s in [1]
        for w in [4]
    ]


@triton.jit
def triton_silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def triton_softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.autotune(configs, key=["siz_B", "T", "B", "NH", "DH"])
@triton.jit
def _gmamba_forward_kernel(
    # Pre-computed inputs (kernel-shaped)
    x_u,          # (NH, T, B, DH) - x part from in_proj + conv1d
    B_u,          # (NH, T, B, N) - B part from in_proj + conv1d
    C_u,          # (NH, T, B, N) - C part from in_proj + conv1d
    dt_raw_u,     # (NH, T, B) - dt pre-activation
    z,            # (NH, T, B, DH) - gating input

    # Per-head recurrent weights
    R_x,          # (NH, DH, DH)
    R_B,          # (NH, N, DH)
    R_C,          # (NH, N, DH)
    R_dt,         # (NH, DH)

    # SSM parameters
    dt_bias,      # (NH,)
    A,            # (NH,)
    D,            # (NH, DH)

    # SSM state (global memory, updated in-place per step)
    ssm_state,    # (NH, B, N, DH)

    # Initial y_prev
    y_prev_init,  # (NH, B, DH)

    # Outputs
    y_out,        # (NH, T, B, DH) - gated output at each step

    # Saved for backward
    x_combined_save,       # (NH, T, B, DH) - pre-silu x
    dt_pre_softplus_save,  # (NH, T, B) - pre-softplus dt
    y_ssm_save,            # (NH, T, B, DH) - SSM output before gating

    # Dimensions
    T: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    DH: tl.constexpr,
    siz_B: tl.constexpr,
    USE_RECURRENT_XBC: tl.constexpr,
    USE_RECURRENT_DT: tl.constexpr,
    CLAMP_DT: tl.constexpr,
    DT_LO: tl.constexpr,
    DT_HI: tl.constexpr,
    SAVE_FOR_BACKWARD: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    idx_b_NH = tl.program_id(0)
    idx_b_B = tl.program_id(1)

    # Strides for x_u, z, y_out: (NH, T, B, DH)
    str_xz_NH = T * B * DH
    str_xz_T = B * DH

    # Strides for B_u, C_u: (NH, T, B, N)
    str_BC_NH = T * B * N
    str_BC_T = B * N

    # Strides for dt: (NH, T, B)
    str_dt_NH = T * B
    str_dt_T = B

    # Strides for ssm_state: (NH, B, N, DH)
    str_state_NH = B * N * DH

    # Offsets
    b_offsets = tl.arange(0, siz_B)
    dh_offsets = tl.arange(0, DH)

    # ---- Load recurrent weights (once, before time loop) ----
    if USE_RECURRENT_XBC:
        R_x_ptr = tl.make_block_ptr(
            base=R_x + idx_b_NH * DH * DH,
            shape=(DH, DH),
            strides=(DH, 1),
            offsets=(0, 0),
            block_shape=(DH, DH),
            order=(0, 1),
        )
        matR_x = tl.load(R_x_ptr)  # (DH, DH)

    if USE_RECURRENT_DT:
        R_dt_vec = tl.load(R_dt + idx_b_NH * DH + dh_offsets)  # (DH,)

    # ---- Load SSM parameters ----
    A_val = tl.load(A + idx_b_NH).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + idx_b_NH).to(tl.float32)
    D_vec = tl.load(D + idx_b_NH * DH + dh_offsets).to(tl.float32)  # (DH,)

    # ---- Load initial y_prev ----
    y_prev_ptr = tl.make_block_ptr(
        base=y_prev_init + idx_b_NH * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    y_prev = tl.load(y_prev_ptr).to(tl.float32)  # (siz_B, DH)

    # ---- Time loop ----
    for idx_t in range(T):
        # Base pointers for this timestep
        base_x = x_u + idx_b_NH * str_xz_NH + idx_t * str_xz_T
        base_B = B_u + idx_b_NH * str_BC_NH + idx_t * str_BC_T
        base_C = C_u + idx_b_NH * str_BC_NH + idx_t * str_BC_T

        # == 1. Load x_u[t]: (siz_B, DH) ==
        x_u_ptr = tl.make_block_ptr(
            base=base_x,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        x_u_t = tl.load(x_u_ptr).to(tl.float32)  # (siz_B, DH)

        # == 2. Load dt_raw_u[t]: (siz_B,) ==
        dt_raw_offsets = (
            dt_raw_u + idx_b_NH * str_dt_NH + idx_t * str_dt_T
            + idx_b_B * siz_B + b_offsets
        )
        dt_raw_t = tl.load(dt_raw_offsets).to(tl.float32)

        # == 3. Load z[t]: (siz_B, DH) ==
        z_ptr = tl.make_block_ptr(
            base=z + idx_b_NH * str_xz_NH + idx_t * str_xz_T,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        z_t = tl.load(z_ptr).to(tl.float32)

        # == 4. Compute recurrent contributions ==
        if USE_RECURRENT_XBC:
            x_recurrent = tl.dot(y_prev.to(DTYPE), matR_x).to(tl.float32)  # (siz_B, DH)
            x_combined = x_u_t + x_recurrent
        else:
            x_combined = x_u_t

        if USE_RECURRENT_DT:
            dt_recurrent = tl.sum(y_prev * R_dt_vec[None, :], axis=1)
            dt_raw_combined = dt_raw_t + dt_recurrent
        else:
            dt_raw_combined = dt_raw_t

        # == 5. Apply activations to dt ==
        if USE_RECURRENT_DT:
            dt_raw_silu = triton_silu(dt_raw_combined)
        else:
            dt_raw_silu = dt_raw_combined

        dt_pre_sp = dt_raw_silu + dt_bias_val
        dt_val = triton_softplus(dt_pre_sp)

        if CLAMP_DT:
            dt_val = tl.minimum(tl.maximum(dt_val, DT_LO), DT_HI)

        # == 6. Compute dA ==
        dA_val = tl.exp(dt_val * A_val)  # (siz_B,)

        # == 7. Apply silu to x ==
        x_act = triton_silu(x_combined)  # (siz_B, DH)

        # == 8. Inner loop over N: state update + y accumulation ==
        y_accum = tl.zeros([siz_B, DH], dtype=tl.float32)

        for n in range(N):
            # Load B_u_n and C_u_n
            B_u_n_offsets = base_B + (idx_b_B * siz_B + b_offsets) * N + n
            B_u_n = tl.load(B_u_n_offsets).to(tl.float32)  # (siz_B,)

            C_u_n_offsets = base_C + (idx_b_B * siz_B + b_offsets) * N + n
            C_u_n = tl.load(C_u_n_offsets).to(tl.float32)  # (siz_B,)

            # Compute B/C recurrent for this n
            if USE_RECURRENT_XBC:
                R_B_row = tl.load(R_B + idx_b_NH * N * DH + n * DH + dh_offsets).to(tl.float32)
                B_rec_n = tl.sum(y_prev * R_B_row[None, :], axis=1)  # (siz_B,)

                R_C_row = tl.load(R_C + idx_b_NH * N * DH + n * DH + dh_offsets).to(tl.float32)
                C_rec_n = tl.sum(y_prev * R_C_row[None, :], axis=1)  # (siz_B,)

                B_combined_n = B_u_n + B_rec_n
                C_combined_n = C_u_n + C_rec_n
            else:
                B_combined_n = B_u_n
                C_combined_n = C_u_n

            B_act_n = triton_silu(B_combined_n)  # (siz_B,)
            C_act_n = triton_silu(C_combined_n)  # (siz_B,)

            # Load state_n: (siz_B, DH) from ssm_state[head, :, n, :]
            state_n_ptr = tl.make_block_ptr(
                base=ssm_state + idx_b_NH * str_state_NH + n * DH,
                shape=(B, DH),
                strides=(N * DH, 1),
                offsets=(idx_b_B * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            state_n = tl.load(state_n_ptr).to(tl.float32)

            # SSM state update
            state_n = dA_val[:, None] * state_n + dt_val[:, None] * x_act * B_act_n[:, None]

            # Accumulate y
            y_accum = y_accum + state_n * C_act_n[:, None]

            # Store updated state
            tl.store(state_n_ptr, state_n.to(DTYPE))

        # == 9. D skip connection ==
        y_ssm = y_accum + D_vec[None, :] * x_act

        # == 10. Gating ==
        y_gated = y_ssm * triton_silu(z_t)

        # == 11. Update y_prev ==
        y_prev = y_gated

        # == 12. Store output ==
        y_out_ptr = tl.make_block_ptr(
            base=y_out + idx_b_NH * str_xz_NH + idx_t * str_xz_T,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(y_out_ptr, y_gated.to(DTYPE))

        # == 13. Save for backward ==
        if SAVE_FOR_BACKWARD:
            x_save_ptr = tl.make_block_ptr(
                base=x_combined_save + idx_b_NH * str_xz_NH + idx_t * str_xz_T,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            tl.store(x_save_ptr, x_combined.to(DTYPE))

            dt_save_offsets = (
                dt_pre_softplus_save + idx_b_NH * str_dt_NH + idx_t * str_dt_T
                + idx_b_B * siz_B + b_offsets
            )
            tl.store(dt_save_offsets, dt_pre_sp.to(DTYPE))

            y_ssm_ptr = tl.make_block_ptr(
                base=y_ssm_save + idx_b_NH * str_xz_NH + idx_t * str_xz_T,
                shape=(B, DH),
                strides=(DH, 1),
                offsets=(idx_b_B * siz_B, 0),
                block_shape=(siz_B, DH),
                order=(0, 1),
            )
            tl.store(y_ssm_ptr, y_ssm.to(DTYPE))


def forward_sequence(
    xBC_u: torch.Tensor,        # (B, T, NH, DH + 2*N) per-head input
    dt_raw_u: torch.Tensor,     # (B, T, NH)
    z: torch.Tensor,            # (B, T, NH, DH)
    R_x: torch.Tensor,          # (NH, DH, DH)
    R_B: torch.Tensor,          # (NH, N, DH)
    R_C: torch.Tensor,          # (NH, N, DH)
    R_dt: torch.Tensor,         # (NH, DH)
    dt_bias: torch.Tensor,      # (NH,)
    A: torch.Tensor,            # (NH,)
    D: torch.Tensor,            # (NH, DH)
    ssm_state_init: torch.Tensor,  # (B, NH, DH, N)
    y_prev_init: torch.Tensor,  # (B, NH, DH)
    use_recurrent_xbc: bool = True,
    use_recurrent_dt: bool = True,
    dt_limit: tuple = (0.0, float("inf")),
    save_for_backward: bool = True,
) -> tuple:
    B_orig, T_val, NH, xBC_dim = xBC_u.shape
    DH = D.shape[1]
    N = (xBC_dim - DH) // 2

    dtype = xBC_u.dtype
    device = xBC_u.device

    assert is_power_of_2(DH), f"DH must be a power of 2, got {DH}"

    MIN_BATCH_SIZE = 16
    effective_B = next_multiple_of(B_orig, MIN_BATCH_SIZE)

    # Split xBC_u into separate x, B, C tensors
    x_u_orig = xBC_u[:, :, :, :DH]           # (B, T, NH, DH)
    B_u_orig = xBC_u[:, :, :, DH:DH+N]       # (B, T, NH, N)
    C_u_orig = xBC_u[:, :, :, DH+N:DH+2*N]   # (B, T, NH, N)

    # Pad batch dimension if needed
    if effective_B != B_orig:
        pad_B = effective_B - B_orig
        x_u_orig = torch.cat([x_u_orig, torch.zeros(pad_B, T_val, NH, DH, device=device, dtype=dtype)], dim=0)
        B_u_orig = torch.cat([B_u_orig, torch.zeros(pad_B, T_val, NH, N, device=device, dtype=dtype)], dim=0)
        C_u_orig = torch.cat([C_u_orig, torch.zeros(pad_B, T_val, NH, N, device=device, dtype=dtype)], dim=0)
        dt_raw_u_pad = torch.cat([dt_raw_u, torch.zeros(pad_B, T_val, NH, device=device, dtype=dtype)], dim=0)
        z_pad = torch.cat([z, torch.zeros(pad_B, T_val, NH, DH, device=device, dtype=dtype)], dim=0)
        y_prev_pad = torch.cat([y_prev_init, torch.zeros(pad_B, NH, DH, device=device, dtype=dtype)], dim=0)
        ssm_state_pad = torch.cat([
            ssm_state_init,
            torch.zeros(pad_B, NH, DH, N, device=device, dtype=dtype),
        ], dim=0)
    else:
        dt_raw_u_pad = dt_raw_u
        z_pad = z
        y_prev_pad = y_prev_init
        ssm_state_pad = ssm_state_init

    # Rearrange to kernel layout
    x_u_k = rearrange(x_u_orig, "b t nh d -> nh t b d").contiguous()
    B_u_k = rearrange(B_u_orig, "b t nh n -> nh t b n").contiguous()
    C_u_k = rearrange(C_u_orig, "b t nh n -> nh t b n").contiguous()
    dt_raw_u_k = rearrange(dt_raw_u_pad, "b t nh -> nh t b").contiguous()
    z_k = rearrange(z_pad, "b t nh d -> nh t b d").contiguous()
    y_prev_k = rearrange(y_prev_pad, "b nh d -> nh b d").contiguous()
    ssm_state_k = rearrange(ssm_state_pad, "b nh d n -> nh b n d").contiguous()

    # Allocate outputs
    y_out_k = torch.empty(NH, T_val, effective_B, DH, device=device, dtype=dtype)

    # Allocate save tensors
    if save_for_backward:
        x_combined_save = torch.empty(NH, T_val, effective_B, DH, device=device, dtype=dtype)
        dt_pre_softplus_save = torch.empty(NH, T_val, effective_B, device=device, dtype=dtype)
        y_ssm_save = torch.empty(NH, T_val, effective_B, DH, device=device, dtype=dtype)
    else:
        x_combined_save = torch.empty(0, device=device, dtype=dtype)
        dt_pre_softplus_save = torch.empty(0, device=device, dtype=dtype)
        y_ssm_save = torch.empty(0, device=device, dtype=dtype)

    dt_lo, dt_hi = dt_limit
    clamp_dt = not (dt_lo == 0.0 and dt_hi == float("inf"))

    def grid(args):
        siz_B = args["siz_B"]
        assert siz_B >= MIN_BATCH_SIZE
        if siz_B > effective_B:
            raise OutOfResources(required=siz_B, limit=effective_B, name="siz_B")
        return (NH, triton.cdiv(effective_B, siz_B))

    _gmamba_forward_kernel[grid](
        x_u=x_u_k,
        B_u=B_u_k,
        C_u=C_u_k,
        dt_raw_u=dt_raw_u_k,
        z=z_k,
        R_x=R_x.contiguous(),
        R_B=R_B.contiguous(),
        R_C=R_C.contiguous(),
        R_dt=R_dt.contiguous(),
        dt_bias=dt_bias,
        A=A,
        D=D,
        ssm_state=ssm_state_k,
        y_prev_init=y_prev_k,
        y_out=y_out_k,
        x_combined_save=x_combined_save,
        dt_pre_softplus_save=dt_pre_softplus_save,
        y_ssm_save=y_ssm_save,
        T=T_val,
        N=N,
        B=effective_B,
        NH=NH,
        DH=DH,
        USE_RECURRENT_XBC=use_recurrent_xbc,
        USE_RECURRENT_DT=use_recurrent_dt,
        CLAMP_DT=clamp_dt,
        DT_LO=float(dt_lo) if clamp_dt else 0.0,
        DT_HI=float(dt_hi) if clamp_dt else 0.0,
        SAVE_FOR_BACKWARD=save_for_backward,
        DTYPE=torch2triton_dtype(dtype),
    )

    # Rearrange outputs back
    y_out = rearrange(y_out_k, "nh t b d -> b t nh d")[:B_orig]
    ssm_state_final = rearrange(ssm_state_k, "nh b n d -> b nh d n")[:B_orig]

    saved = {}
    if save_for_backward:
        saved = {
            "x_combined_save": x_combined_save,
            "dt_pre_softplus_save": dt_pre_softplus_save,
            "y_ssm_save": y_ssm_save,
            "y_out_k": y_out_k,
            "effective_B": effective_B,
            "B_orig": B_orig,
        }

    return y_out, ssm_state_final, saved
