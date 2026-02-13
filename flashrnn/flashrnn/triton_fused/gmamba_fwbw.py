"""
Autograd wrapper for Generalized Mamba SSM triton kernel.

Forward: triton kernel (fast)
Backward: PyTorch implementation with state recomputation (correct, can be optimized later)
"""

import torch
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd

from flashrnn.flashrnn.triton_fused.gmamba_fw import forward_sequence


def _silu_backward(grad_output, x):
    """Backward of silu(x) = x * sigmoid(x)."""
    sig = torch.sigmoid(x)
    return grad_output * sig * (1.0 + x * (1.0 - sig))


def _softplus_backward(grad_output, x):
    """Backward of softplus(x) = log(1 + exp(x))."""
    return grad_output * torch.sigmoid(x)


class GMambaFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        xBC_u,          # (B, T, NH, DH + 2*N) pre-computed input
        dt_raw_u,        # (B, T, NH) dt pre-activation
        z,               # (B, T, NH, DH) gating
        R_x,             # (NH, DH, DH) per-head recurrent x
        R_B,             # (NH, N, DH) per-head recurrent B
        R_C,             # (NH, N, DH) per-head recurrent C
        R_dt,            # (NH, DH) per-head recurrent dt
        dt_bias,         # (NH,)
        A,               # (NH,) negative
        D,               # (NH, DH) skip
        ssm_state_init,  # (B, NH, DH, N)
        y_prev_init,     # (B, NH, DH)
        use_recurrent_xbc,  # bool
        use_recurrent_dt,   # bool
        dt_limit,           # tuple
    ):
        y_out, ssm_state_final, saved = forward_sequence(
            xBC_u=xBC_u,
            dt_raw_u=dt_raw_u,
            z=z,
            R_x=R_x,
            R_B=R_B,
            R_C=R_C,
            R_dt=R_dt,
            dt_bias=dt_bias,
            A=A,
            D=D,
            ssm_state_init=ssm_state_init,
            y_prev_init=y_prev_init,
            use_recurrent_xbc=use_recurrent_xbc,
            use_recurrent_dt=use_recurrent_dt,
            dt_limit=dt_limit,
            save_for_backward=True,
        )

        ctx.save_for_backward(
            xBC_u, dt_raw_u, z,
            R_x, R_B, R_C, R_dt,
            dt_bias, A, D,
            ssm_state_init, y_prev_init,
            # Saved from kernel
            saved["dt_pre_softplus_save"],
            saved["y_ssm_save"],
            saved["y_out_k"],
        )
        ctx.use_recurrent_xbc = use_recurrent_xbc
        ctx.use_recurrent_dt = use_recurrent_dt
        ctx.dt_limit = dt_limit
        ctx.effective_B = saved["effective_B"]
        ctx.B_orig = saved["B_orig"]

        return y_out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_y_out):
        """
        Backward pass implemented in PyTorch with state recomputation.

        Strategy:
        1. Recompute forward states from initial state using saved gate activations
        2. Compute gradients backward through time
        """
        (
            xBC_u, dt_raw_u, z,
            R_x, R_B, R_C, R_dt,
            dt_bias, A, D,
            ssm_state_init, y_prev_init,
            dt_pre_softplus_save, y_ssm_save, y_out_k,
        ) = ctx.saved_tensors

        use_recurrent_xbc = ctx.use_recurrent_xbc
        use_recurrent_dt = ctx.use_recurrent_dt
        dt_limit = ctx.dt_limit

        B, T, NH, DH_plus_2N = xBC_u.shape
        DH = D.shape[1]
        N = (DH_plus_2N - DH) // 2

        device = xBC_u.device

        # ---- Forward recomputation to get states and intermediate values ----
        # We recompute states, activated values, and y_prev_gated at each step

        ssm_state = ssm_state_init.clone().to(torch.float32)  # (B, NH, DH, N)
        y_prev = y_prev_init.clone().to(torch.float32)        # (B, NH, DH)

        # Storage for recomputed intermediates
        all_states = []  # state BEFORE update at each step
        all_x_act = []
        all_B_act = []   # (B, NH, N)
        all_C_act = []
        all_dt = []
        all_dA = []
        all_x_combined = []   # pre-silu x
        all_BC_combined = []  # pre-silu B, C (B, NH, 2*N)
        all_dt_raw_combined = []  # pre-silu dt
        all_y_prev = []  # y_prev BEFORE this step (input to recurrent)

        A_neg = A.float()  # (NH,) negative values

        for t in range(T):
            all_states.append(ssm_state.clone())
            all_y_prev.append(y_prev.clone())

            # Extract per-step inputs
            xBC_u_t = xBC_u[:, t, :, :].float()  # (B, NH, DH+2N)
            dt_raw_t = dt_raw_u[:, t, :].float()  # (B, NH)
            z_t = z[:, t, :, :].float()            # (B, NH, DH)

            # Split xBC_u into x, B, C parts
            x_u = xBC_u_t[:, :, :DH]           # (B, NH, DH)
            B_u = xBC_u_t[:, :, DH:DH+N]       # (B, NH, N)
            C_u = xBC_u_t[:, :, DH+N:DH+2*N]   # (B, NH, N)

            # Compute recurrent contributions
            if use_recurrent_xbc:
                # R_x: (NH, DH, DH), y_prev: (B, NH, DH)
                # x_recurrent[b, h, :] = y_prev[b, h, :] @ R_x[h, :, :]
                x_recurrent = torch.einsum("bhi,hij->bhj", y_prev, R_x.float())
                B_recurrent = torch.einsum("bhi,hni->bhn", y_prev, R_B.float())
                C_recurrent = torch.einsum("bhi,hni->bhn", y_prev, R_C.float())
                x_combined = x_u + x_recurrent
                B_combined = B_u + B_recurrent
                C_combined = C_u + C_recurrent
            else:
                x_combined = x_u
                B_combined = B_u
                C_combined = C_u

            all_x_combined.append(x_combined.clone())
            all_BC_combined.append(torch.cat([B_combined, C_combined], dim=-1).clone())

            x_act = F.silu(x_combined)
            B_act = F.silu(B_combined)
            C_act = F.silu(C_combined)

            all_x_act.append(x_act)
            all_B_act.append(B_act)
            all_C_act.append(C_act)

            # dt computation
            if use_recurrent_dt:
                dt_recurrent = torch.einsum("bhi,hi->bh", y_prev, R_dt.float())
                dt_raw_combined = dt_raw_t + dt_recurrent
                dt_raw_silu = F.silu(dt_raw_combined)
            else:
                dt_raw_combined = dt_raw_t
                dt_raw_silu = dt_raw_combined

            all_dt_raw_combined.append(dt_raw_combined.clone())

            dt_pre_sp = dt_raw_silu + dt_bias.float()
            dt_val = F.softplus(dt_pre_sp)

            dt_lo, dt_hi = dt_limit
            if not (dt_lo == 0.0 and dt_hi == float("inf")):
                dt_val = dt_val.clamp(min=dt_lo, max=dt_hi)

            dA_val = torch.exp(dt_val * A_neg[None, :])  # (B, NH)

            all_dt.append(dt_val)
            all_dA.append(dA_val)

            # State update: state = dA * state + dt * x * B (outer product)
            # state: (B, NH, DH, N), x_act: (B, NH, DH), B_act: (B, NH, N)
            ssm_state = (
                dA_val[:, :, None, None] * ssm_state
                + dt_val[:, :, None, None] * x_act[:, :, :, None] * B_act[:, :, None, :]
            )

            # y = (state * C).sum(N) + D * x
            y_ssm = (ssm_state * C_act[:, :, None, :]).sum(dim=-1) + D.float()[None, :, :] * x_act

            # Gating
            y_gated = y_ssm * F.silu(z_t)
            y_prev = y_gated

        all_states.append(ssm_state.clone())  # final state

        # ---- Backward pass ----
        grad_y_out_f = grad_y_out.float()  # (B, T, NH, DH)

        # Initialize gradients
        grad_xBC_u = torch.zeros_like(xBC_u, dtype=torch.float32)
        grad_dt_raw_u = torch.zeros_like(dt_raw_u, dtype=torch.float32)
        grad_z = torch.zeros_like(z, dtype=torch.float32)
        grad_R_x = torch.zeros_like(R_x, dtype=torch.float32)
        grad_R_B = torch.zeros_like(R_B, dtype=torch.float32)
        grad_R_C = torch.zeros_like(R_C, dtype=torch.float32)
        grad_R_dt = torch.zeros_like(R_dt, dtype=torch.float32)
        grad_dt_bias = torch.zeros_like(dt_bias, dtype=torch.float32)
        grad_A = torch.zeros_like(A, dtype=torch.float32)
        grad_D = torch.zeros_like(D, dtype=torch.float32)

        # Gradient flowing backward through state recurrence
        grad_state = torch.zeros(B, NH, DH, N, device=device, dtype=torch.float32)
        # Gradient flowing backward through y_prev recurrence
        grad_y_prev = torch.zeros(B, NH, DH, device=device, dtype=torch.float32)

        for t in range(T - 1, -1, -1):
            # Load saved values
            state_old = all_states[t]      # state BEFORE update at step t
            state_new = all_states[t + 1]  # state AFTER update at step t
            x_act = all_x_act[t]
            B_act = all_B_act[t]
            C_act = all_C_act[t]
            dt_val = all_dt[t]
            dA_val = all_dA[t]
            x_combined = all_x_combined[t]
            BC_combined = all_BC_combined[t]
            B_combined = BC_combined[:, :, :N]
            C_combined = BC_combined[:, :, N:]
            dt_raw_combined = all_dt_raw_combined[t]
            y_prev_for_step = all_y_prev[t]  # y_prev used as input at step t

            z_t = z[:, t, :, :].float()

            # y_ssm = (state_new * C_act).sum(-1) + D * x_act
            y_ssm = (state_new * C_act[:, :, None, :]).sum(-1) + D.float()[None, :, :] * x_act

            # --- Backprop through gating: y_gated = y_ssm * silu(z_t) ---
            grad_y_gated_t = grad_y_out_f[:, t, :, :] + grad_y_prev  # (B, NH, DH)
            silu_z = F.silu(z_t)
            grad_y_ssm = grad_y_gated_t * silu_z
            # grad_z_t: dsilu(z) = sig(z)(1 + z(1-sig(z)))
            sig_z = torch.sigmoid(z_t)
            grad_z_t = grad_y_gated_t * y_ssm * sig_z * (1.0 + z_t * (1.0 - sig_z))
            grad_z[:, t, :, :] = grad_z_t.to(grad_z.dtype)

            # --- Backprop through y = (state * C).sum(-1) + D * x ---
            # grad_state_new from y: (B, NH, DH, N)
            grad_state_from_y = grad_y_ssm[:, :, :, None] * C_act[:, :, None, :]
            grad_C_act = (grad_y_ssm[:, :, :, None] * state_new).sum(dim=2)  # (B, NH, N)
            grad_x_act = grad_y_ssm * D.float()[None, :, :]
            grad_D += (grad_y_ssm * x_act).sum(dim=0)  # (NH, DH)

            # Add gradient from next step's state propagation
            grad_state_new_total = grad_state_from_y + grad_state

            # --- Backprop through state update ---
            # state_new = dA * state_old + dt * x * B (outer product)
            grad_state_old = dA_val[:, :, None, None] * grad_state_new_total
            grad_dA = (grad_state_new_total * state_old).sum(dim=(2, 3))  # (B, NH)
            grad_dt_from_state = (
                grad_state_new_total * x_act[:, :, :, None] * B_act[:, :, None, :]
            ).sum(dim=(2, 3))  # (B, NH)
            grad_x_act = grad_x_act + (
                grad_state_new_total * dt_val[:, :, None, None] * B_act[:, :, None, :]
            ).sum(dim=-1)  # (B, NH, DH)
            grad_B_act = (
                grad_state_new_total * dt_val[:, :, None, None] * x_act[:, :, :, None]
            ).sum(dim=2)  # (B, NH, N)

            # --- Backprop through dA = exp(dt * A) ---
            # grad_dt from dA: grad_dA * dA * A
            grad_dt_val = grad_dA * dA_val * A_neg[None, :] + grad_dt_from_state
            grad_A += (grad_dA * dA_val * dt_val).sum(dim=0)  # (NH,)

            # --- Backprop through dt clamping (if any) ---
            dt_lo, dt_hi = ctx.dt_limit
            if not (dt_lo == 0.0 and dt_hi == float("inf")):
                dt_pre_sp = all_dt_raw_combined[t]
                if use_recurrent_dt:
                    dt_pre_sp_val = F.silu(dt_pre_sp) + dt_bias.float()
                else:
                    dt_pre_sp_val = dt_pre_sp + dt_bias.float()
                dt_unclamped = F.softplus(dt_pre_sp_val)
                # Gradient is zero where clamped
                grad_dt_val = grad_dt_val * ((dt_unclamped >= dt_lo) & (dt_unclamped <= dt_hi)).float()

            # --- Backprop through softplus ---
            if use_recurrent_dt:
                dt_pre_sp_val = F.silu(dt_raw_combined) + dt_bias.float()
            else:
                dt_pre_sp_val = dt_raw_combined + dt_bias.float()
            grad_dt_pre_sp = grad_dt_val * torch.sigmoid(dt_pre_sp_val)  # (B, NH)
            grad_dt_bias += grad_dt_pre_sp.sum(dim=0)

            # --- Backprop through dt silu (if recurrent_dt) ---
            if use_recurrent_dt:
                sig_dt = torch.sigmoid(dt_raw_combined)
                grad_dt_raw_combined = grad_dt_pre_sp * sig_dt * (1.0 + dt_raw_combined * (1.0 - sig_dt))
            else:
                grad_dt_raw_combined = grad_dt_pre_sp

            # --- Backprop through dt recurrence ---
            grad_dt_raw_u[:, t, :] = grad_dt_raw_combined.to(grad_dt_raw_u.dtype)
            if use_recurrent_dt:
                # dt_recurrent = einsum("bhi,hi->bh", y_prev, R_dt)
                grad_y_prev_from_dt = torch.einsum("bh,hi->bhi", grad_dt_raw_combined, R_dt.float())
                grad_R_dt += torch.einsum("bh,bhi->hi", grad_dt_raw_combined, y_prev_for_step)
            else:
                grad_y_prev_from_dt = torch.zeros_like(y_prev_for_step)

            # --- Backprop through silu activations for x, B, C ---
            sig_x = torch.sigmoid(x_combined)
            grad_x_combined = grad_x_act * sig_x * (1.0 + x_combined * (1.0 - sig_x))

            sig_B = torch.sigmoid(B_combined)
            grad_B_combined = grad_B_act * sig_B * (1.0 + B_combined * (1.0 - sig_B))

            sig_C = torch.sigmoid(C_combined)
            grad_C_combined = grad_C_act * sig_C * (1.0 + C_combined * (1.0 - sig_C))

            # --- Backprop through recurrent xBC ---
            grad_x_u = grad_x_combined
            grad_B_u = grad_B_combined
            grad_C_u = grad_C_combined

            if use_recurrent_xbc:
                # x_recurrent = einsum("bhi,hij->bhj", y_prev, R_x)
                grad_y_prev_from_x = torch.einsum("bhj,hij->bhi", grad_x_combined, R_x.float())
                grad_R_x += torch.einsum("bhj,bhi->hij", grad_x_combined, y_prev_for_step)

                grad_y_prev_from_B = torch.einsum("bhn,hni->bhi", grad_B_combined, R_B.float())
                grad_R_B += torch.einsum("bhn,bhi->hni", grad_B_combined, y_prev_for_step)

                grad_y_prev_from_C = torch.einsum("bhn,hni->bhi", grad_C_combined, R_C.float())
                grad_R_C += torch.einsum("bhn,bhi->hni", grad_C_combined, y_prev_for_step)

                grad_y_prev_from_recurrent = grad_y_prev_from_x + grad_y_prev_from_B + grad_y_prev_from_C + grad_y_prev_from_dt
            else:
                grad_y_prev_from_recurrent = grad_y_prev_from_dt

            # Store input gradients
            grad_xBC_u[:, t, :, :DH] = grad_x_u.to(grad_xBC_u.dtype)
            grad_xBC_u[:, t, :, DH:DH+N] = grad_B_u.to(grad_xBC_u.dtype)
            grad_xBC_u[:, t, :, DH+N:DH+2*N] = grad_C_u.to(grad_xBC_u.dtype)

            # Propagate gradients to previous timestep
            grad_state = grad_state_old
            grad_y_prev = grad_y_prev_from_recurrent

        return (
            grad_xBC_u.to(xBC_u.dtype),
            grad_dt_raw_u.to(dt_raw_u.dtype),
            grad_z.to(z.dtype),
            grad_R_x.to(R_x.dtype),
            grad_R_B.to(R_B.dtype),
            grad_R_C.to(R_C.dtype),
            grad_R_dt.to(R_dt.dtype),
            grad_dt_bias.to(dt_bias.dtype),
            grad_A.to(A.dtype),
            grad_D.to(D.dtype),
            None,  # ssm_state_init (no grad)
            None,  # y_prev_init (no grad)
            None,  # use_recurrent_xbc
            None,  # use_recurrent_dt
            None,  # dt_limit
        )


def gmamba_fwbw(
    xBC_u: torch.Tensor,
    dt_raw_u: torch.Tensor,
    z: torch.Tensor,
    R_x: torch.Tensor,
    R_B: torch.Tensor,
    R_C: torch.Tensor,
    R_dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    ssm_state_init: torch.Tensor,
    y_prev_init: torch.Tensor,
    use_recurrent_xbc: bool = True,
    use_recurrent_dt: bool = True,
    dt_limit: tuple = (0.0, float("inf")),
) -> torch.Tensor:
    """
    Forward + backward for Generalized Mamba SSM scan.

    Returns:
        y_out: (B, T, NH, DH) gated SSM output at each step
    """
    return GMambaFunction.apply(
        xBC_u, dt_raw_u, z,
        R_x, R_B, R_C, R_dt,
        dt_bias, A, D,
        ssm_state_init, y_prev_init,
        use_recurrent_xbc, use_recurrent_dt,
        dt_limit,
    )
