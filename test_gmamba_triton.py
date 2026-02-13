"""
Test the triton generalized mamba kernel against a naive PyTorch reference.

Tests both forward correctness and backward gradient correctness.
"""

import torch
import torch.nn.functional as F


def naive_gmamba_scan(
    xBC_u,       # (B, T, NH, DH + 2*N)
    dt_raw_u,    # (B, T, NH)
    z,           # (B, T, NH, DH)
    R_x,         # (NH, DH, DH)
    R_B,         # (NH, N, DH)
    R_C,         # (NH, N, DH)
    R_dt,        # (NH, DH)
    dt_bias,     # (NH,)
    A,           # (NH,) negative
    D,           # (NH, DH)
    ssm_state,   # (B, NH, DH, N)
    y_prev,      # (B, NH, DH)
    use_recurrent_xbc=True,
    use_recurrent_dt=True,
    dt_limit=(0.0, float("inf")),
):
    """Naive for-loop reference implementation for the per-head SSM scan."""
    B, T, NH, xBC_dim = xBC_u.shape
    DH = D.shape[1]
    N = (xBC_dim - DH) // 2

    ssm_state = ssm_state.clone().float()
    y_prev = y_prev.clone().float()
    A_f = A.float()
    D_f = D.float()
    dt_bias_f = dt_bias.float()
    dt_lo, dt_hi = dt_limit

    outputs = []

    for t in range(T):
        xBC_u_t = xBC_u[:, t, :, :].float()
        dt_raw_t = dt_raw_u[:, t, :].float()
        z_t = z[:, t, :, :].float()

        x_u = xBC_u_t[:, :, :DH]
        B_u = xBC_u_t[:, :, DH:DH+N]
        C_u = xBC_u_t[:, :, DH+N:DH+2*N]

        if use_recurrent_xbc:
            x_rec = torch.einsum("bhi,hij->bhj", y_prev, R_x.float())
            B_rec = torch.einsum("bhi,hni->bhn", y_prev, R_B.float())
            C_rec = torch.einsum("bhi,hni->bhn", y_prev, R_C.float())
            x_combined = x_u + x_rec
            B_combined = B_u + B_rec
            C_combined = C_u + C_rec
        else:
            x_combined = x_u
            B_combined = B_u
            C_combined = C_u

        x_act = F.silu(x_combined)
        B_act = F.silu(B_combined)
        C_act = F.silu(C_combined)

        if use_recurrent_dt:
            dt_rec = torch.einsum("bhi,hi->bh", y_prev, R_dt.float())
            dt_combined = dt_raw_t + dt_rec
            dt_silu = F.silu(dt_combined)
        else:
            dt_combined = dt_raw_t
            dt_silu = dt_combined

        dt_pre_sp = dt_silu + dt_bias_f
        dt_val = F.softplus(dt_pre_sp)
        if not (dt_lo == 0.0 and dt_hi == float("inf")):
            dt_val = dt_val.clamp(min=dt_lo, max=dt_hi)

        dA = torch.exp(dt_val * A_f[None, :])

        ssm_state = (
            dA[:, :, None, None] * ssm_state
            + dt_val[:, :, None, None] * x_act[:, :, :, None] * B_act[:, :, None, :]
        )

        y_ssm = (ssm_state * C_act[:, :, None, :]).sum(-1) + D_f[None, :, :] * x_act
        y_gated = y_ssm * F.silu(z_t)
        y_prev = y_gated

        outputs.append(y_gated)

    return torch.stack(outputs, dim=1)  # (B, T, NH, DH)


def test_forward():
    """Test forward pass correctness."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    B, T, NH, DH, N = 2, 32, 4, 64, 16
    xBC_dim = DH + 2 * N

    xBC_u = torch.randn(B, T, NH, xBC_dim, device=device, dtype=dtype) * 0.1
    dt_raw_u = torch.randn(B, T, NH, device=device, dtype=dtype) * 0.1
    z = torch.randn(B, T, NH, DH, device=device, dtype=dtype) * 0.1
    R_x = torch.randn(NH, DH, DH, device=device, dtype=dtype) * 0.01
    R_B = torch.randn(NH, N, DH, device=device, dtype=dtype) * 0.01
    R_C = torch.randn(NH, N, DH, device=device, dtype=dtype) * 0.01
    R_dt = torch.randn(NH, DH, device=device, dtype=dtype) * 0.01
    dt_bias = torch.randn(NH, device=device, dtype=dtype) * 0.1
    A = -torch.rand(NH, device=device, dtype=dtype) * 5  # negative
    D = torch.ones(NH, DH, device=device, dtype=dtype)
    ssm_state = torch.zeros(B, NH, DH, N, device=device, dtype=dtype)
    y_prev = torch.zeros(B, NH, DH, device=device, dtype=dtype)

    # Naive reference
    y_naive = naive_gmamba_scan(
        xBC_u, dt_raw_u, z, R_x, R_B, R_C, R_dt,
        dt_bias, A, D, ssm_state, y_prev,
    )

    # Triton kernel
    from flashrnn.flashrnn.triton_fused.gmamba_fwbw import gmamba_fwbw

    y_triton = gmamba_fwbw(
        xBC_u, dt_raw_u, z, R_x, R_B, R_C, R_dt,
        dt_bias, A, D, ssm_state.clone(), y_prev.clone(),
    )

    # Compare
    max_err = (y_naive - y_triton).abs().max().item()
    rel_err = ((y_naive - y_triton).abs() / (y_naive.abs() + 1e-8)).max().item()
    print(f"Forward test: max_err={max_err:.6e}, rel_err={rel_err:.6e}")
    print(f"  y_naive range: [{y_naive.min():.4f}, {y_naive.max():.4f}]")
    print(f"  y_triton range: [{y_triton.min():.4f}, {y_triton.max():.4f}]")

    assert max_err < 1e-3, f"Forward max error too large: {max_err}"
    print("PASSED: Forward test")
    return True


def test_backward():
    """Test backward pass correctness via gradient comparison."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    B, T, NH, DH, N = 2, 16, 4, 64, 16
    xBC_dim = DH + 2 * N

    torch.manual_seed(42)

    def make_data():
        xBC_u = (torch.randn(B, T, NH, xBC_dim, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
        dt_raw_u = (torch.randn(B, T, NH, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
        z = (torch.randn(B, T, NH, DH, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
        R_x = (torch.randn(NH, DH, DH, device=device, dtype=dtype) * 0.01).detach().requires_grad_(True)
        R_B = (torch.randn(NH, N, DH, device=device, dtype=dtype) * 0.01).detach().requires_grad_(True)
        R_C = (torch.randn(NH, N, DH, device=device, dtype=dtype) * 0.01).detach().requires_grad_(True)
        R_dt = (torch.randn(NH, DH, device=device, dtype=dtype) * 0.01).detach().requires_grad_(True)
        dt_bias = (torch.randn(NH, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
        A = (-torch.rand(NH, device=device, dtype=dtype) * 5).detach().requires_grad_(True)
        D = torch.ones(NH, DH, device=device, dtype=dtype).detach().requires_grad_(True)
        ssm_state = torch.zeros(B, NH, DH, N, device=device, dtype=dtype)
        y_prev = torch.zeros(B, NH, DH, device=device, dtype=dtype)
        return [xBC_u, dt_raw_u, z, R_x, R_B, R_C, R_dt, dt_bias, A, D, ssm_state, y_prev]

    # Test with naive reference
    inputs_naive = make_data()
    y_naive = naive_gmamba_scan(*inputs_naive)
    loss_naive = y_naive.sum()
    loss_naive.backward()
    grads_naive = [p.grad.clone() for p in inputs_naive[:10]]

    # Test with triton (same data)
    torch.manual_seed(42)
    inputs_triton = make_data()

    from flashrnn.flashrnn.triton_fused.gmamba_fwbw import gmamba_fwbw
    y_triton = gmamba_fwbw(*inputs_triton)
    loss_triton = y_triton.sum()
    loss_triton.backward()
    grads_triton = [p.grad.clone() for p in inputs_triton[:10]]

    names = ["xBC_u", "dt_raw_u", "z", "R_x", "R_B", "R_C", "R_dt", "dt_bias", "A", "D"]
    all_ok = True
    for name, gn, gt in zip(names, grads_naive, grads_triton):
        max_err = (gn - gt).abs().max().item()
        rel_err = ((gn - gt).abs() / (gn.abs() + 1e-8)).max().item()
        ok = max_err < 1e-2
        status = "OK" if ok else "FAIL"
        print(f"  grad_{name}: max_err={max_err:.6e}, rel_err={rel_err:.6e} [{status}]")
        if not ok:
            all_ok = False

    if all_ok:
        print("PASSED: Backward test")
    else:
        print("FAILED: Backward test (some gradients don't match)")
    return all_ok


def test_module():
    """Test the full GeneralizedMamba2 module."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    from generalized_mamba import GeneralizedMamba2

    model = GeneralizedMamba2(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=64,
        use_recurrent_xbc=True,
        use_recurrent_dt=True,
        device=device,
        dtype=dtype,
    )

    B, L = 2, 32
    u = torch.randn(B, L, 128, device=device, dtype=dtype)

    # Forward
    out = model(u)
    print(f"Module output shape: {out.shape}")
    assert out.shape == (B, L, 128), f"Expected (B, L, d_model), got {out.shape}"

    # Backward
    loss = out.sum()
    loss.backward()

    # Check that all parameters have gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            print(f"  {name}: grad={'YES' if has_grad else 'NO'}, shape={p.shape}")

    print("PASSED: Module test")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Forward correctness")
    print("=" * 60)
    test_forward()

    print()
    print("=" * 60)
    print("Test 2: Backward correctness")
    print("=" * 60)
    test_backward()

    print()
    print("=" * 60)
    print("Test 3: Full module")
    print("=" * 60)
    test_module()
