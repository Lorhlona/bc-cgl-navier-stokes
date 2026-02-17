"""
BC-Fisher diagnostics — split into small JIT units to avoid compilation blowup.
Each function is independently @jit'd for fast compilation and low memory.
"""
import jax
import jax.numpy as jnp
from jax import jit
from ns_solver import omega_to_vel_hat, fftn_vec, ifftn_vec


# ── Step 1: Helical decomposition (separate JIT) ───────────

@jit
def helical_decompose(omega_hat, g):
    """ω_hat → (ω+_hat, ω-_hat, h+, h-)"""
    khat = jnp.stack([g.kx, g.ky, g.kz], axis=-1) / g.kmag[..., None]

    ref_z = jnp.array([0.0, 0.0, 1.0])
    e1 = jnp.cross(khat, ref_z)
    e1_norm = jnp.linalg.norm(e1, axis=-1, keepdims=True)
    ref_x = jnp.array([1.0, 0.0, 0.0])
    e1_alt = jnp.cross(khat, ref_x)
    e1_alt_norm = jnp.linalg.norm(e1_alt, axis=-1, keepdims=True)
    use_alt = (e1_norm[..., 0] < 1e-6)
    e1 = jnp.where(use_alt[..., None], e1_alt, e1)
    e1_norm = jnp.where(use_alt[..., None], e1_alt_norm, e1_norm)
    e1 = e1 / jnp.maximum(e1_norm, 1e-30)
    e2 = jnp.cross(khat, e1)

    hp = (e1 + 1j * e2) / jnp.sqrt(2.0)
    hm = (e1 - 1j * e2) / jnp.sqrt(2.0)

    oh = jnp.stack([omega_hat[0], omega_hat[1], omega_hat[2]], axis=-1)
    op_hat = jnp.sum(jnp.conj(hp) * oh, axis=-1)
    om_hat = jnp.sum(jnp.conj(hm) * oh, axis=-1)
    return op_hat, om_hat, hp, hm


# ── Step 2: Basic scalars (Ω, F_θ, r) — separate JIT ──────

@jit
def compute_basic(omega_hat, op_hat, om_hat, g):
    """Compute Ω, F_θ, r, ||ω||_{L3}², θ_rms. Light computation."""
    N = g.N
    dV = (g.L / N) ** 3

    w = ifftn_vec(omega_hat)
    omega_mag_sq = w[0]**2 + w[1]**2 + w[2]**2
    omega_mag = jnp.sqrt(omega_mag_sq + 1e-30)

    op_mag = jnp.abs(jnp.fft.ifftn(op_hat)) + 1e-30
    om_mag = jnp.abs(jnp.fft.ifftn(om_hat)) + 1e-30

    sum_sq = op_mag**2 + om_mag**2
    sech2 = 4.0 * op_mag**2 * om_mag**2 / (sum_sq**2 + 1e-30)

    Omega = 0.5 * jnp.sum(omega_mag_sq) * dV
    F_theta = jnp.sum(omega_mag_sq * sech2) * dV
    r = F_theta / (2.0 * Omega + 1e-30)
    omega_L3_sq = (jnp.sum(omega_mag**3) * dV) ** (2.0 / 3.0)
    theta_rms = jnp.sqrt(jnp.mean(jnp.log(op_mag / om_mag) ** 2))

    return jnp.array([Omega, F_theta, r, omega_L3_sq, theta_rms])


# ── Step 3: γ_eff (heaviest part — separate JIT) ───────────

@jit
def compute_gamma_eff(omega_hat, op_hat, om_hat, hp, hm, g):
    """Compute γ_eff = ∫|S_{+-}| sech²θ |ω| dx / F_θ.
    This is the expensive part: Biot-Savart + strain + projection.
    """
    N = g.N
    dV = (g.L / N) ** 3

    # ω physical
    w = ifftn_vec(omega_hat)
    omega_mag = jnp.sqrt(w[0]**2 + w[1]**2 + w[2]**2 + 1e-30)

    # sech²θ
    op_mag = jnp.abs(jnp.fft.ifftn(op_hat)) + 1e-30
    om_mag = jnp.abs(jnp.fft.ifftn(om_hat)) + 1e-30
    sum_sq = op_mag**2 + om_mag**2
    sech2 = 4.0 * op_mag**2 * om_mag**2 / (sum_sq**2 + 1e-30)
    F_theta = jnp.sum(omega_mag**2 * sech2) * dV

    # ω_- vector → velocity → strain (only 6 unique components, symmetric)
    om_vec = om_hat[..., None] * hm                         # (N,N,N,3)
    om3 = jnp.moveaxis(om_vec, -1, 0)                       # (3,N,N,N)
    vm_hat = omega_to_vel_hat(om3, g)                        # (3,N,N,N)

    # Strain S^(-)_ij: only need the contraction with ω_+ direction,
    # so compute S·d directly without materializing full (3,3,N,N,N).
    # S_ij = (i/2)(k_i v_j + k_j v_i) in Fourier
    # (S·d)_i = (i/2)(k_i (v·d) + v_i (k·d))  for unit vector d
    # This avoids the 9-component tensor entirely.

    # ω_+ direction in physical space
    op_vec = op_hat[..., None] * hp
    op3 = jnp.moveaxis(op_vec, -1, 0)                       # (3,N,N,N) complex
    op_phys = jax.vmap(jnp.fft.ifftn)(op3)                  # (3,N,N,N) complex
    op_dir_mag = jnp.sqrt(jnp.sum(jnp.abs(op_phys)**2, axis=0) + 1e-30)
    d = op_phys / (op_dir_mag[None] + 1e-30)                # unit direction

    # d in Fourier
    d_hat = jax.vmap(jnp.fft.fftn)(d)                       # (3,N,N,N) complex

    # (v·d) and (k·d) in Fourier — these are convolutions, need physical space
    # Simpler: compute S·d entirely in physical space using finite differences
    # Actually: S_ij = (∂_i v_j + ∂_j v_i)/2, so
    # (S·d)_i = (1/2)(∂_i(v·d) + Σ_j d_j ∂_j v_i)  ... still needs derivatives.

    # Most memory-efficient: compute S_{+-} = d_i S_ij d_j
    # = (1/2) d_i (∂_i v_j + ∂_j v_i) d_j
    # = (1/2)(d_i ∂_i v_j d_j + d_i ∂_j v_i d_j)
    # = Re[d_i (∂_i v_j) d_j]  (by symmetry of the two terms)
    #
    # In Fourier: ∂_i v_j → i k_i v_j_hat
    # So S_{+-} in physical space = Re[ ifftn(i k_i v_j_hat) * conj(d_i) * d_j ]
    # Summed over i,j.
    #
    # Compute per-component to avoid (3,3,N,N,N): loop over i, accumulate.
    # But loops kill JIT... Use: S_{+-} = Re[(d·∇)(v·d)]
    # (d·∇)(v·d) = Σ_i d_i ∂_i (Σ_j v_j d_j)
    # This needs v and d in physical space.

    vm_phys = ifftn_vec(vm_hat)  # (3,N,N,N) real

    # v·d in physical space (complex scalar field)
    vd = jnp.sum(vm_phys * jnp.conj(d), axis=0)            # (N,N,N) complex

    # ∂_i(v·d) in Fourier then back
    vd_hat = jnp.fft.fftn(vd)                               # (N,N,N)
    k_arr = jnp.stack([g.kx, g.ky, g.kz])                   # (3,N,N,N)
    grad_vd_hat = 1j * k_arr * vd_hat[None]                 # (3,N,N,N)
    grad_vd = jax.vmap(jnp.fft.ifftn)(grad_vd_hat)          # (3,N,N,N)

    # (d·∇)(v·d) = Σ_i d_i * (∂_i(v·d))
    S_pm = jnp.real(jnp.sum(jnp.conj(d) * grad_vd, axis=0))  # (N,N,N)

    weight = sech2 * omega_mag * dV
    gamma_abs = jnp.sum(jnp.abs(S_pm) * weight) / (F_theta + 1e-30)
    gamma_signed = jnp.sum(S_pm * weight) / (F_theta + 1e-30)

    # coherence
    op2 = jnp.abs(op_hat)**2
    om2 = jnp.abs(om_hat)**2
    sigma = jnp.sum(op2 * om2) / (jnp.sum(op2 + om2)**2 + 1e-30)

    return jnp.array([gamma_abs, gamma_signed, sigma])


# ── Public API ──────────────────────────────────────────────

DIAG_NAMES = ['Omega', 'F_theta', 'r', 'omega_L3_sq',
              'theta_rms', 'gamma_abs', 'gamma_signed', 'sigma']

def compute_all_diagnostics(omega_hat, g):
    """Run all diagnostics (3 separate JIT calls). Returns dict."""
    op_hat, om_hat, hp, hm = helical_decompose(omega_hat, g)
    basic = compute_basic(omega_hat, op_hat, om_hat, g)
    gamma = compute_gamma_eff(omega_hat, op_hat, om_hat, hp, hm, g)
    vals = list(basic) + list(gamma)  # 5 + 3 = 8
    return {k: float(vals[i]) for i, k in enumerate(DIAG_NAMES)}
