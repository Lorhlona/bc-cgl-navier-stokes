"""
BC-CGL × NS experiment runner — GPU-optimized, server-safe.
Float32 only. Timeout protection. Split JIT compilation.

Usage:
  source ~/.venvs/gpu-cuda/bin/activate
  python -u run_experiment.py --N 64 --nu 0.01 --T 3.0 --exp tg
"""
import jax
import jax.numpy as jnp
from jax import random
import json
import time
import argparse
import os
import signal
import sys

# ── SAFETY: float32 only, no x64 ───────────────────────────
# Do NOT enable x64 — doubles memory, no benefit for DNS

from ns_solver import (make_grid, rk4_step, advance, taylor_green_omega,
                       random_omega, abc_omega, truncate_to_grid)
from bc_diagnostics import compute_all_diagnostics, helical_decompose, compute_basic, DIAG_NAMES


# ── Timeout protection ──────────────────────────────────────

def _timeout_handler(signum, frame):
    print("\n*** TIMEOUT — killing process safely ***", flush=True)
    sys.exit(1)


def run_dns(omega_hat_init, g, nu, T_final, dt, diag_every_steps=20):
    N_steps = int(T_final / dt)

    print(f"DNS: N={g.N}, nu={nu:.4f}, dt={dt:.6f}, "
          f"T={T_final}, steps={N_steps}, diag_every={diag_every_steps}",
          flush=True)

    # JIT warmup — compile each piece separately (lighter)
    print("JIT: advance...", end=" ", flush=True)
    t0 = time.time()
    omega_hat = omega_hat_init
    _test = rk4_step(omega_hat, dt, g, nu)
    jax.block_until_ready(_test)
    print(f"({time.time()-t0:.1f}s)", end="  ", flush=True)

    print("helical...", end=" ", flush=True)
    t1 = time.time()
    _op, _om, _hp, _hm = helical_decompose(omega_hat, g)
    jax.block_until_ready(_op)
    print(f"({time.time()-t1:.1f}s)", end="  ", flush=True)

    print("basic...", end=" ", flush=True)
    t2 = time.time()
    _b = compute_basic(omega_hat, _op, _om, g)
    jax.block_until_ready(_b)
    print(f"({time.time()-t2:.1f}s)", flush=True)

    print("Full diag warmup...", end=" ", flush=True)
    t3 = time.time()
    d0 = compute_all_diagnostics(omega_hat, g)
    print(f"({time.time()-t3:.1f}s)", flush=True)

    # Collect
    diag_times = [0.0]
    diag_data = [d0]
    _print_row(0.0, d0)

    t0 = time.time()
    t_phys = 0.0

    for chunk in range(N_steps // diag_every_steps):
        omega_hat = advance(omega_hat, g, nu, dt, diag_every_steps)
        t_phys += diag_every_steps * dt

        d = compute_all_diagnostics(omega_hat, g)
        diag_times.append(float(t_phys))
        diag_data.append(d)

        elapsed = time.time() - t0
        sps = (chunk + 1) * diag_every_steps / elapsed
        _print_row(t_phys, d, sps)

    # Remainder
    rem = N_steps % diag_every_steps
    if rem > 0:
        omega_hat = advance(omega_hat, g, nu, dt, rem)
        t_phys += rem * dt
        d = compute_all_diagnostics(omega_hat, g)
        diag_times.append(float(t_phys))
        diag_data.append(d)

    total = time.time() - t0
    print(f"Done: {N_steps} steps in {total:.1f}s ({N_steps/total:.0f} steps/s)",
          flush=True)
    return diag_times, diag_data


def _print_row(t, d, sps=0.0):
    s = f" | {sps:.0f} st/s" if sps > 0 else ""
    ga = d['gamma_abs']
    gs = d['gamma_signed']
    eps = gs / (ga + 1e-30) * 100  # cancellation ratio %
    print(f"  t={t:6.3f} | Ω={d['Omega']:.3e} | r={d['r']:.4f} | "
          f"γa={ga:.4f} γs={gs:+.4f} | ε={eps:+.1f}% | "
          f"θ={d['theta_rms']:.3f}{s}", flush=True)


def save_results(diag_times, diag_data, label, g, nu):
    outdir = f"results_{label}"
    os.makedirs(outdir, exist_ok=True)

    with open(f"{outdir}/timeseries.csv", "w") as f:
        f.write("time," + ",".join(DIAG_NAMES) + "\n")
        for t, d in zip(diag_times, diag_data):
            vals = ",".join(f"{d[k]:.8e}" for k in DIAG_NAMES)
            f.write(f"{t:.8e},{vals}\n")

    r_arr = [d['r'] for d in diag_data]
    ga_arr = [d['gamma_abs'] for d in diag_data]
    gs_arr = [d['gamma_signed'] for d in diag_data]
    omega_arr = [d['Omega'] for d in diag_data]
    eps_arr = [gs/(ga+1e-30)*100 for gs, ga in zip(gs_arr, ga_arr)]
    omega_max_idx = omega_arr.index(max(omega_arr))
    n = len(r_arr)
    r_mean = sum(r_arr)/n
    eps_mean = sum(eps_arr)/n
    gs_mean = sum(gs_arr)/n
    summary = {
        'label': label,
        'N': g.N, 'nu': nu,
        'Re_approx': int(round(1.0/nu)),
        'n_points': n,
        'Omega_max': max(omega_arr),
        'Omega_max_t': diag_times[omega_max_idx],
        'r_mean': r_mean,
        'r_std': (sum((x-r_mean)**2 for x in r_arr)/n)**0.5,
        'gamma_abs_min': min(ga_arr),
        'gamma_abs_max': max(ga_arr),
        'gamma_signed_mean': gs_mean,
        'gamma_signed_std': (sum((x-gs_mean)**2 for x in gs_arr)/n)**0.5,
        'epsilon_mean': eps_mean,
        'epsilon_std': (sum((x-eps_mean)**2 for x in eps_arr)/n)**0.5,
        'epsilon_max': max(eps_arr),
        'epsilon_min': min(eps_arr),
    }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Run Summary ===", flush=True)
    print(f"N={summary['N']}, ν={nu:.6f}, Re≈{summary['Re_approx']}", flush=True)
    print(f"Ω_max={summary['Omega_max']:.1f} at t={summary['Omega_max_t']:.2f}", flush=True)
    print(f"mean(ε)={summary['epsilon_mean']:.2f}% ± {summary['epsilon_std']:.2f}%", flush=True)
    print(f"ε range: [{summary['epsilon_min']:.2f}%, {summary['epsilon_max']:.2f}%]", flush=True)
    print(f"mean(r)={summary['r_mean']:.4f} ± {summary['r_std']:.4f}", flush=True)
    print(f"mean(γ_signed)={summary['gamma_signed_mean']:.4f} ± {summary['gamma_signed_std']:.4f}", flush=True)
    print(f"Saved → {outdir}/", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.0)
    parser.add_argument("--exp", default="tg",
                        choices=["tg","random","highRe","abc","convergence","all"])
    parser.add_argument("--diag-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Max runtime in seconds (safety kill)")
    args = parser.parse_args()

    # Safety timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)
    print(f"Safety timeout: {args.timeout}s", flush=True)

    N = args.N
    g = make_grid(N)
    dx = float(g.L) / N
    dt = args.dt if args.dt > 0 else min(0.3*dx, 0.3*dx**2/args.nu)

    print("=" * 60, flush=True)
    print(f"BC-CGL × NS DNS  |  {jax.devices()}", flush=True)
    print(f"N={N}  nu={args.nu}  dt={dt:.6f}  T={args.T}  f32", flush=True)
    print("=" * 60, flush=True)

    exps = []  # list of (label, omega_hat, nu, grid)

    if args.exp == "convergence":
        # Resolution convergence: generate IC at N=512, truncate
        print("Generating IC at N=512 for truncation...", flush=True)
        g512 = make_grid(512)
        w0_512 = taylor_green_omega(g512)
        for Nc in [128, 256, 512]:
            gc = make_grid(Nc)
            dxc = float(gc.L) / Nc
            dtc = min(0.3*dxc, 0.3*dxc**2/args.nu)
            w0c = truncate_to_grid(w0_512, g512, gc) if Nc < 512 else w0_512
            exps.append((f"conv_N{Nc}", w0c, args.nu, gc, dtc))
    else:
        if args.exp in ("tg", "all"):
            exps.append(("taylor_green", taylor_green_omega(g), args.nu, g, dt))
        if args.exp in ("random", "all"):
            exps.append(("random_iso",
                          random_omega(g, random.PRNGKey(args.seed)), args.nu, g, dt))
        if args.exp in ("highRe", "all"):
            exps.append(("highRe",
                          random_omega(g, random.PRNGKey(args.seed+1), amplitude=5.0),
                          args.nu / 5.0, g, dt))
        if args.exp in ("abc", "all"):
            exps.append(("abc_flow", abc_omega(g), args.nu, g, dt))

    summaries = {}
    for item in exps:
        label, w0, nu, gi, dti = item
        print(f"\n{'='*60}\n{label}  N={gi.N} nu={nu:.6f}\n{'='*60}", flush=True)
        times, data = run_dns(w0, gi, nu, args.T, dti, args.diag_every)
        summaries[label] = save_results(times, data, label, gi, nu)

    if len(summaries) > 1:
        print(f"\n{'='*60}\n=== Cancellation Summary ===\n{'='*60}", flush=True)
        print(f"  {'Run':20s} | {'Re':>5s} | {'max(Ω)':>8s} | {'mean(ε)':>8s} | {'range(ε)':>16s} | {'mean(r)':>7s}", flush=True)
        for lb, s in summaries.items():
            print(f"  {lb:20s} | {s['Re_approx']:5d} | {s['Omega_max']:8.0f} | "
                  f"{s['epsilon_mean']:+7.2f}% | "
                  f"[{s['epsilon_min']:+.1f}%, {s['epsilon_max']:+.1f}%] | "
                  f"{s['r_mean']:.4f}", flush=True)

    signal.alarm(0)
    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    main()
