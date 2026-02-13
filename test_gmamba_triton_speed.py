"""
Speed benchmark for generalized mamba implementations.

Compares:
1) Triton implementation (gmamba_fwbw)
2) Non-triton reference (naive_gmamba_scan)

Reports per-step latency over the full input sequence tensor for:
- forward only
- backward only
- combined forward + backward
"""

import argparse
import statistics

import torch

from test_gmamba_triton import naive_gmamba_scan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark triton vs naive gmamba latency."
    )
    parser.add_argument("--batch", type=int, default=2, help="Batch size (B)")
    parser.add_argument("--seq", type=int, default=128, help="Sequence length (T)")
    parser.add_argument("--nheads", type=int, default=4, help="Number of heads (NH)")
    parser.add_argument("--headdim", type=int, default=64, help="Head dim (DH)")
    parser.add_argument("--state", type=int, default=16, help="State size (N)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Input dtype",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (not timed)"
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Timed iterations per metric"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-recurrent-xbc",
        action="store_true",
        help="Disable recurrent x/B/C contributions",
    )
    parser.add_argument(
        "--no-recurrent-dt",
        action="store_true",
        help="Disable recurrent dt contribution",
    )
    parser.add_argument(
        "--disable-triton",
        action="store_true",
        help="Skip triton benchmark and run naive only",
    )
    parser.add_argument(
        "--disable-naive",
        action="store_true",
        help="Skip naive benchmark and run triton only",
    )
    return parser.parse_args()


def _get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def make_inputs(B, T, NH, DH, N, dtype, device):
    xBC_dim = DH + 2 * N
    xBC_u = (
        (torch.randn(B, T, NH, xBC_dim, device=device, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    dt_raw_u = (
        (torch.randn(B, T, NH, device=device, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    z = (
        (torch.randn(B, T, NH, DH, device=device, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    R_x = (
        (torch.randn(NH, DH, DH, device=device, dtype=dtype) * 0.01)
        .detach()
        .requires_grad_(True)
    )
    R_B = (
        (torch.randn(NH, N, DH, device=device, dtype=dtype) * 0.01)
        .detach()
        .requires_grad_(True)
    )
    R_C = (
        (torch.randn(NH, N, DH, device=device, dtype=dtype) * 0.01)
        .detach()
        .requires_grad_(True)
    )
    R_dt = (
        (torch.randn(NH, DH, device=device, dtype=dtype) * 0.01)
        .detach()
        .requires_grad_(True)
    )
    dt_bias = (
        (torch.randn(NH, device=device, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    A = ((-torch.rand(NH, device=device, dtype=dtype) * 5).detach().requires_grad_(True))
    D = torch.ones(NH, DH, device=device, dtype=dtype).detach().requires_grad_(True)
    ssm_state = torch.zeros(B, NH, DH, N, device=device, dtype=dtype)
    y_prev = torch.zeros(B, NH, DH, device=device, dtype=dtype)

    return [xBC_u, dt_raw_u, z, R_x, R_B, R_C, R_dt, dt_bias, A, D, ssm_state, y_prev]


def clone_inputs(base_inputs):
    cloned = []
    for i, t in enumerate(base_inputs):
        tc = t.detach().clone()
        if i < 10:
            tc.requires_grad_(True)
        cloned.append(tc)
    return cloned


def _clear_grads(inputs):
    for p in inputs[:10]:
        p.grad = None


def _run_forward(
    fn, inputs, use_recurrent_xbc: bool, use_recurrent_dt: bool, backend_name: str
):
    if backend_name == "triton":
        return fn(
            *inputs,
            use_recurrent_xbc=use_recurrent_xbc,
            use_recurrent_dt=use_recurrent_dt,
        )
    return fn(
        *inputs,
        use_recurrent_xbc=use_recurrent_xbc,
        use_recurrent_dt=use_recurrent_dt,
    )


def _run_combined_once(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name):
    _clear_grads(inputs)
    y = _run_forward(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
    y.sum().backward()


def benchmark_forward(
    fn, inputs, warmup, iters, use_recurrent_xbc, use_recurrent_dt, backend_name
):
    for _ in range(warmup):
        y = _run_forward(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
        del y
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = _run_forward(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
        del y
    return timings


def benchmark_backward(
    fn, inputs, warmup, iters, use_recurrent_xbc, use_recurrent_dt, backend_name
):
    for _ in range(warmup):
        _clear_grads(inputs)
        y = _run_forward(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
        y.sum().backward()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        _clear_grads(inputs)
        y = _run_forward(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
        loss = y.sum()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return timings


def benchmark_combined(
    fn, inputs, warmup, iters, use_recurrent_xbc, use_recurrent_dt, backend_name
):
    for _ in range(warmup):
        _run_combined_once(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_combined_once(fn, inputs, use_recurrent_xbc, use_recurrent_dt, backend_name)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return timings


def summarize(timings_ms):
    mean_ms = statistics.mean(timings_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "steps_per_sec": 1000.0 / mean_ms,
    }


def format_stats(name, stats):
    return (
        f"{name:<18} "
        f"mean={stats['mean_ms']:.3f} ms  "
        f"median={stats['median_ms']:.3f} ms  "
        f"min={stats['min_ms']:.3f} ms  "
        f"max={stats['max_ms']:.3f} ms  "
        f"throughput={stats['steps_per_sec']:.2f} steps/s"
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    from flashrnn.flashrnn.triton_fused.gmamba_fwbw import gmamba_fwbw

    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = _get_dtype(args.dtype)
    use_recurrent_xbc = not args.no_recurrent_xbc
    use_recurrent_dt = not args.no_recurrent_dt

    base_inputs = make_inputs(
        B=args.batch,
        T=args.seq,
        NH=args.nheads,
        DH=args.headdim,
        N=args.state,
        dtype=dtype,
        device=device,
    )
    inputs_triton = clone_inputs(base_inputs)
    inputs_naive = clone_inputs(base_inputs)

    run_triton = not args.disable_triton
    run_naive = not args.disable_naive
    if not run_triton and not run_naive:
        raise ValueError("Both backends are disabled. Enable at least one backend.")

    backend_names = []
    if run_triton:
        backend_names.append("triton")
    if run_naive:
        backend_names.append("naive")

    print("Benchmark config:")
    print(
        f"  B={args.batch}, T={args.seq}, NH={args.nheads}, "
        f"DH={args.headdim}, N={args.state}"
    )
    print(
        f"  dtype={dtype}, warmup={args.warmup}, iters={args.iters}, "
        f"use_recurrent_xbc={use_recurrent_xbc}, use_recurrent_dt={use_recurrent_dt}"
    )
    print(f"  backends={backend_names}")

    all_results = {}

    if run_triton:
        # First call compiles/autotunes triton; keep out of timing.
        _run_combined_once(
            gmamba_fwbw,
            inputs_triton,
            use_recurrent_xbc,
            use_recurrent_dt,
            "triton",
        )
        torch.cuda.synchronize()

        triton_forward = benchmark_forward(
            gmamba_fwbw,
            inputs_triton,
            args.warmup,
            args.iters,
            use_recurrent_xbc,
            use_recurrent_dt,
            "triton",
        )
        triton_backward = benchmark_backward(
            gmamba_fwbw,
            inputs_triton,
            args.warmup,
            args.iters,
            use_recurrent_xbc,
            use_recurrent_dt,
            "triton",
        )
        triton_combined = benchmark_combined(
            gmamba_fwbw,
            inputs_triton,
            args.warmup,
            args.iters,
            use_recurrent_xbc,
            use_recurrent_dt,
            "triton",
        )
        all_results["triton"] = {
            "forward": summarize(triton_forward),
            "backward": summarize(triton_backward),
            "combined": summarize(triton_combined),
        }

    if run_naive:
        naive_forward = benchmark_forward(
            naive_gmamba_scan,
            inputs_naive,
            args.warmup,
            args.iters,
            use_recurrent_xbc,
            use_recurrent_dt,
            "naive",
        )
        naive_backward = benchmark_backward(
            naive_gmamba_scan,
            inputs_naive,
            args.warmup,
            args.iters,
            use_recurrent_xbc,
            use_recurrent_dt,
            "naive",
        )
        naive_combined = benchmark_combined(
            naive_gmamba_scan,
            inputs_naive,
            args.warmup,
            args.iters,
            use_recurrent_xbc,
            use_recurrent_dt,
            "naive",
        )
        all_results["naive"] = {
            "forward": summarize(naive_forward),
            "backward": summarize(naive_backward),
            "combined": summarize(naive_combined),
        }

    print("\nResults (per full sequence step):")
    for backend in ("triton", "naive"):
        if backend not in all_results:
            continue
        print(f"\n[{backend}]")
        print(format_stats("forward", all_results[backend]["forward"]))
        print(format_stats("backward", all_results[backend]["backward"]))
        print(format_stats("forward+backward", all_results[backend]["combined"]))

    if "triton" in all_results and "naive" in all_results:
        print("\nSpeedup (naive / triton, based on mean latency):")
        for metric in ("forward", "backward", "combined"):
            speedup = (
                all_results["naive"][metric]["mean_ms"]
                / all_results["triton"][metric]["mean_ms"]
            )
            print(f"  {metric:<10}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
