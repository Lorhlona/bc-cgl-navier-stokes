"""
Enstrophy Budget Closure Verification + BC Helical Decomposition
All JAX, GPU-only. Float32.

Verifies: dΩ/dt = -νD + Π_total  (residual < 1%)
Decomposes: Π_total = Π_hetero + Π_homo
Measures: ε_CZ = |Π_hetero_signed| / Π_hetero_abs
"""
import jax
import jax.numpy as jnp
from jax import jit
import time, os, sys, signal

from ns_solver import (make_grid, advance, taylor_green_omega,
                       omega_to_vel_hat, ifftn_vec)
from bc_diagnostics import helical_decompose, compute_all_diagnostics


def _timeout_handler(signum, frame):
    print("\n*** TIMEOUT ***", flush=True)
    sys.exit(1)


@jit
def compute_budget(omega_hat, g, nu):
    """All enstrophy budget quantities. Returns 8-element array."""
    N = g.N
    dV = (g.L / N) ** 3

    # Vorticity physical
    w = ifftn_vec(omega_hat)
    omega_mag_sq = w[0]**2 + w[1]**2 + w[2]**2
    Omega = 0.5 * jnp.sum(omega_mag_sq) * dV

    # Palinstrophy via Parseval: ν∫|∇ω|² dx
    omega_sq_k = jnp.sum(jnp.abs(omega_hat)**2, axis=0)
    nuD = nu * jnp.sum(g.k2 * omega_sq_k) * dV / N**3

    # Strain tensor S_ij
    v_hat = omega_to_vel_hat(omega_hat, g)
    S00 = jnp.fft.ifftn(1j * g.kx * v_hat[0]).real
    S11 = jnp.fft.ifftn(1j * g.ky * v_hat[1]).real
    S22 = jnp.fft.ifftn(1j * g.kz * v_hat[2]).real
    S01 = 0.5 * (jnp.fft.ifftn(1j * g.kx * v_hat[1]).real
                + jnp.fft.ifftn(1j * g.ky * v_hat[0]).real)
    S02 = 0.5 * (jnp.fft.ifftn(1j * g.kx * v_hat[2]).real
                + jnp.fft.ifftn(1j * g.kz * v_hat[0]).real)
    S12 = 0.5 * (jnp.fft.ifftn(1j * g.ky * v_hat[2]).real
                + jnp.fft.ifftn(1j * g.kz * v_hat[1]).real)

    def bilin(a, b):
        return (a[0]*b[0]*S00 + a[1]*b[1]*S11 + a[2]*b[2]*S22
                + (a[0]*b[1] + a[1]*b[0])*S01
                + (a[0]*b[2] + a[2]*b[0])*S02
                + (a[1]*b[2] + a[2]*b[1])*S12)

    Pi_total = jnp.sum(bilin(w, w)) * dV

    # Helical decomposition
    op_hat, om_hat, hp, hm = helical_decompose(omega_hat, g)
    wp = jax.vmap(jnp.fft.ifftn)(
        jnp.moveaxis(op_hat[..., None] * hp, -1, 0))
    wm = jax.vmap(jnp.fft.ifftn)(
        jnp.moveaxis(om_hat[..., None] * hm, -1, 0))

    Pi_het_field = 2.0 * bilin(wp, wm).real
    Pi_hom_field = (bilin(wp, wp) + bilin(wm, wm)).real

    Pi_het_signed = jnp.sum(Pi_het_field) * dV
    Pi_het_abs = jnp.sum(jnp.abs(Pi_het_field)) * dV
    Pi_homo = jnp.sum(Pi_hom_field) * dV

    decomp_err = jnp.abs(Pi_total - Pi_het_signed - Pi_homo) / (jnp.abs(Pi_total) + 1e-30)
    eps_CZ = jnp.abs(Pi_het_signed) / (Pi_het_abs + 1e-30)

    return jnp.array([Omega, nuD, Pi_total, Pi_het_signed, Pi_het_abs,
                       Pi_homo, decomp_err, eps_CZ])


def main():
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(3600)

    N = 256
    nu = 0.0005
    T = 15.0
    diag_every = 7

    g = make_grid(N)
    dx = float(g.L) / N
    dt = min(0.3 * dx, 0.3 * dx**2 / nu)
    N_steps = int(T / dt)
    dt_diag = diag_every * dt

    print("=" * 80, flush=True)
    print(f"Enstrophy Budget Closure  |  {jax.devices()}", flush=True)
    print(f"N={N}  nu={nu}  Re={int(1/nu)}  dt={dt:.6f}  T={T}  f32", flush=True)
    print(f"diag_every={diag_every} ({dt_diag:.4f}s)  n_diags~{N_steps//diag_every}", flush=True)
    print("=" * 80, flush=True)

    omega_hat = taylor_green_omega(g)

    # JIT warmup
    print("JIT: budget...", end=" ", flush=True)
    t0 = time.time()
    _b = compute_budget(omega_hat, g, nu)
    jax.block_until_ready(_b)
    print(f"({time.time()-t0:.1f}s)", end="  ", flush=True)

    print("bc_diags...", end=" ", flush=True)
    t0 = time.time()
    _d = compute_all_diagnostics(omega_hat, g)
    print(f"({time.time()-t0:.1f}s)", flush=True)

    times, rows = [], []

    def record(t_phys, wh):
        b = compute_budget(wh, g, nu)
        bc = compute_all_diagnostics(wh, g)
        row = dict(t=t_phys,
                   Omega=float(b[0]), nuD=float(b[1]), Pi_total=float(b[2]),
                   Pi_het_s=float(b[3]), Pi_het_a=float(b[4]), Pi_homo=float(b[5]),
                   decomp_err=float(b[6]), eps_CZ=float(b[7]),
                   r=bc['r'], gamma_abs=bc['gamma_abs'], gamma_signed=bc['gamma_signed'])
        times.append(t_phys)
        rows.append(row)
        return row

    def prow(r, wall=0.0):
        ws = f" {wall:.0f}s" if wall > 0 else ""
        print(f"  t={r['t']:6.2f} | O={r['Omega']:8.1f} | -vD={-r['nuD']:+9.1f} | "
              f"Pi={r['Pi_total']:+9.1f} | het_s={r['Pi_het_s']:+9.2f} | "
              f"het_a={r['Pi_het_a']:9.2f} | eCZ={r['eps_CZ']:.4f} | "
              f"dc={r['decomp_err']:.1e} | r={r['r']:.4f} | "
              f"ga={r['gamma_abs']:.4f} gs={r['gamma_signed']:+.4f}{ws}", flush=True)

    r0 = record(0.0, omega_hat)
    prow(r0)

    t_phys = 0.0
    t_wall = time.time()
    n_chunks = N_steps // diag_every

    for chunk in range(n_chunks):
        omega_hat = advance(omega_hat, g, nu, dt, diag_every)
        t_phys += diag_every * dt
        row = record(t_phys, omega_hat)
        if chunk % 10 == 0 or chunk == n_chunks - 1:
            prow(row, time.time() - t_wall)

    rem = N_steps % diag_every
    if rem > 0:
        omega_hat = advance(omega_hat, g, nu, dt, rem)
        t_phys += rem * dt
        row = record(t_phys, omega_hat)
        prow(row, time.time() - t_wall)

    wall = time.time() - t_wall
    print(f"\nDNS done: {N_steps} steps, {len(rows)} diags, {wall:.0f}s\n", flush=True)

    # ── Post: dΩ/dt via central differences, residual ──
    print("=" * 80)
    print("BUDGET RESIDUAL (central diff)")
    print("=" * 80)
    for i in range(1, len(rows) - 1):
        dt_d = times[i+1] - times[i-1]
        dOdt = (rows[i+1]['Omega'] - rows[i-1]['Omega']) / dt_d
        R = dOdt + rows[i]['nuD'] - rows[i]['Pi_total']
        rel = abs(R) / (abs(dOdt) + 1e-30)
        rows[i]['dOdt'] = dOdt
        rows[i]['R'] = R
        rows[i]['rel_R'] = rel
        if i % 30 == 0:
            print(f"  t={times[i]:5.2f} dO/dt={dOdt:+10.2f} -vD={-rows[i]['nuD']:+10.2f} "
                  f"Pi={rows[i]['Pi_total']:+10.2f} R={R:+.4f} "
                  f"|R|/|dO/dt|={rel:.3e} eCZ={rows[i]['eps_CZ']:.4f}", flush=True)

    turb = [rows[i] for i in range(1, len(rows)-1) if times[i] >= 4.0 and 'rel_R' in rows[i]]
    if turb:
        mr = sum(r['rel_R'] for r in turb) / len(turb)
        xr = max(r['rel_R'] for r in turb)
        me = sum(r['eps_CZ'] for r in turb) / len(turb)
        md = max(r['decomp_err'] for r in turb)
        print(f"\n{'='*80}")
        print(f"RESULTS  (t>=4, {len(turb)} pts)")
        print(f"{'='*80}")
        print(f"Budget:  mean|R|/|dO/dt| = {mr:.4e}  max = {xr:.4e}")
        print(f"Decomp:  max err = {md:.2e}  (float32 ~1e-7 expected)")
        print(f"ε_CZ:    mean = {me:.4f} ({me*100:.2f}%)")
        print(f"{'PASS' if mr < 0.01 else 'FAIL'}: budget {'<' if mr < 0.01 else '>='} 1%")
        print(f"{'PASS' if md < 1e-4 else 'WARN'}: decomp integrity")

    # ── Save ──
    outdir = "results_enstrophy_budget"
    os.makedirs(outdir, exist_ok=True)
    cols = ['t','Omega','dOdt','nuD','Pi_total','R','rel_R',
            'Pi_het_s','Pi_het_a','Pi_homo','eps_CZ',
            'decomp_err','r','gamma_abs','gamma_signed']
    with open(f"{outdir}/timeseries.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(f"{row.get(c,0.0):.10e}" for c in cols) + "\n")
    print(f"\nSaved -> {outdir}/timeseries.csv")

    signal.alarm(0)
    print("ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
