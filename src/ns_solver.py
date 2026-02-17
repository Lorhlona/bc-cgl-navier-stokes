"""
Pseudospectral 3D Navier-Stokes solver on T^3 — fully JIT-compiled JAX.
Vorticity formulation, 2/3 dealiasing, RK4 time stepping.
All operations vectorized (no Python for-loops in hot paths).
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import NamedTuple


class Grid(NamedTuple):
    """Immutable grid data — JAX-traceable as pytree."""
    kx: jnp.ndarray       # (N,N,N)
    ky: jnp.ndarray
    kz: jnp.ndarray
    k2: jnp.ndarray        # |k|^2
    k2_safe: jnp.ndarray   # |k|^2 with k=0 replaced by 1
    kmag: jnp.ndarray      # |k|
    dealias: jnp.ndarray   # bool mask
    N: int
    L: float


def make_grid(N, L=2*jnp.pi):
    dx = L / N
    k1d = jnp.fft.fftfreq(N, d=dx) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_safe = jnp.where(k2 == 0, 1.0, k2)
    kmag = jnp.sqrt(k2_safe)
    kmax = (2.0 / 3.0) * (N // 2) * (2 * jnp.pi / L)
    dealias = (jnp.abs(kx) < kmax) & (jnp.abs(ky) < kmax) & (jnp.abs(kz) < kmax)
    return Grid(kx=kx, ky=ky, kz=kz, k2=k2, k2_safe=k2_safe,
                kmag=kmag, dealias=dealias, N=N, L=L)


# ── FFT helpers (batched over component axis) ───────────────

def fftn_vec(v):
    """FFT of vector field v: (3,N,N,N) real → (3,N,N,N) complex."""
    return jax.vmap(jnp.fft.fftn)(v)

def ifftn_vec(v_hat):
    """Inverse FFT of vector field: (3,N,N,N) complex → (3,N,N,N) real."""
    return jax.vmap(lambda x: jnp.fft.ifftn(x).real)(v_hat)


# ── Biot-Savart: ω_hat → v_hat (vectorized) ────────────────

@jit
def omega_to_vel_hat(omega_hat, g):
    """v_hat = i (k × ω_hat) / |k|^2. Fully vectorized."""
    # omega_hat: (3,N,N,N), g: Grid
    inv_k2 = 1.0 / g.k2_safe
    vx = 1j * (g.ky * omega_hat[2] - g.kz * omega_hat[1]) * inv_k2
    vy = 1j * (g.kz * omega_hat[0] - g.kx * omega_hat[2]) * inv_k2
    vz = 1j * (g.kx * omega_hat[1] - g.ky * omega_hat[0]) * inv_k2
    v = jnp.stack([vx, vy, vz])
    return v.at[:, 0, 0, 0].set(0.0)


# ── NS vorticity RHS (single JIT block) ────────────────────

@jit
def rhs_vorticity(omega_hat, g, nu):
    """∂ω/∂t = curl(v×ω) + ν Δω. Single fused kernel."""
    vel_hat = omega_to_vel_hat(omega_hat, g)
    v = ifftn_vec(vel_hat)     # (3,N,N,N) real
    w = ifftn_vec(omega_hat)   # (3,N,N,N) real

    # Cross product v×ω in physical space (vectorized)
    lamb = jnp.stack([
        v[1]*w[2] - v[2]*w[1],
        v[2]*w[0] - v[0]*w[2],
        v[0]*w[1] - v[1]*w[0],
    ])
    lamb_hat = fftn_vec(lamb)

    # curl in Fourier: i k × lamb_hat
    curl = jnp.stack([
        1j*(g.ky*lamb_hat[2] - g.kz*lamb_hat[1]),
        1j*(g.kz*lamb_hat[0] - g.kx*lamb_hat[2]),
        1j*(g.kx*lamb_hat[1] - g.ky*lamb_hat[0]),
    ])
    # Dealias + viscous
    return curl * g.dealias[None] - nu * g.k2[None] * omega_hat


# ── RK4 (single JIT) ───────────────────────────────────────

@jit
def rk4_step(omega_hat, dt, g, nu):
    k1 = rhs_vorticity(omega_hat, g, nu)
    k2 = rhs_vorticity(omega_hat + 0.5*dt*k1, g, nu)
    k3 = rhs_vorticity(omega_hat + 0.5*dt*k2, g, nu)
    k4 = rhs_vorticity(omega_hat + dt*k3, g, nu)
    return omega_hat + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ── Multi-step (lax.fori_loop — no Python overhead) ────────

def advance(omega_hat, g, nu, dt, n_steps):
    """Advance n_steps without Python loop overhead."""
    def body(_, state):
        return rk4_step(state, dt, g, nu)
    return jax.lax.fori_loop(0, n_steps, body, omega_hat)


# ── Initial conditions ──────────────────────────────────────

def taylor_green_omega(g):
    N, L = g.N, g.L
    dx = L / N
    x = jnp.arange(N) * dx
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    omega = jnp.stack([
        jnp.cos(X)*jnp.sin(Y)*jnp.sin(Z),
        jnp.sin(X)*jnp.cos(Y)*jnp.sin(Z),
        -2.0*jnp.sin(X)*jnp.sin(Y)*jnp.cos(Z),
    ])
    return fftn_vec(omega)


def random_omega(g, key, k_peak=4.0, amplitude=1.0):
    N = g.N
    E_k = g.kmag**4 * jnp.exp(-2.0*(g.kmag/k_peak)**2)
    amp = jnp.sqrt(E_k / g.k2_safe)

    keys = jax.random.split(key, 6)
    omega_hat = jnp.stack([
        amp * (jax.random.normal(keys[2*i], (N,N,N)) +
               1j*jax.random.normal(keys[2*i+1], (N,N,N)))
        for i in range(3)
    ])
    # Divergence-free projection: ω -= k(k·ω)/|k|²
    kdotw = g.kx*omega_hat[0] + g.ky*omega_hat[1] + g.kz*omega_hat[2]
    proj = jnp.stack([g.kx, g.ky, g.kz]) * (kdotw / g.k2_safe)[None]
    omega_hat = (omega_hat - proj) * g.dealias[None]
    # Normalize
    norm = jnp.sqrt(jnp.sum(jnp.abs(omega_hat)**2) / N**3)
    return omega_hat * (amplitude / jnp.maximum(norm, 1e-30))


def abc_omega(g, A=1.0, B=1.0, C=1.0):
    """ABC flow initial vorticity. Beltrami: ω = k v, so ω = v for k=1."""
    N, L = g.N, g.L
    dx = L / N
    x = jnp.arange(N) * dx
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    # v = (A sin z + C cos y, B sin x + A cos z, C sin y + B cos x)
    # For Beltrami flow with k=1: ω = curl v = v (eigenvalue +1)
    omega = jnp.stack([
        A*jnp.sin(Z) + C*jnp.cos(Y),
        B*jnp.sin(X) + A*jnp.cos(Z),
        C*jnp.sin(Y) + B*jnp.cos(X),
    ])
    return fftn_vec(omega)


def truncate_to_grid(omega_hat_src, g_src, g_dst):
    """Fourier-truncate IC from high-res grid to low-res grid.
    Keeps modes with |k_i| < N_dst/2 for each component.
    """
    Ns, Nd = g_src.N, g_dst.N
    if Nd >= Ns:
        return omega_hat_src  # no truncation needed
    half_d = Nd // 2
    # Index slicing: keep [0:half_d] and [-half_d:] in each dimension
    def _trunc_3d(arr):
        # arr: (Ns, Ns, Ns) complex
        out = jnp.zeros((Nd, Nd, Nd), dtype=arr.dtype)
        # low-freq block [0:half_d, 0:half_d, 0:half_d]
        out = out.at[:half_d, :half_d, :half_d].set(
            arr[:half_d, :half_d, :half_d])
        # mixed blocks (8 corners of the cube)
        out = out.at[-half_d:, :half_d, :half_d].set(
            arr[-half_d:, :half_d, :half_d])
        out = out.at[:half_d, -half_d:, :half_d].set(
            arr[:half_d, -half_d:, :half_d])
        out = out.at[:half_d, :half_d, -half_d:].set(
            arr[:half_d, :half_d, -half_d:])
        out = out.at[-half_d:, -half_d:, :half_d].set(
            arr[-half_d:, -half_d:, :half_d])
        out = out.at[-half_d:, :half_d, -half_d:].set(
            arr[-half_d:, :half_d, -half_d:])
        out = out.at[:half_d, -half_d:, -half_d:].set(
            arr[:half_d, -half_d:, -half_d:])
        out = out.at[-half_d:, -half_d:, -half_d:].set(
            arr[-half_d:, -half_d:, -half_d:])
        # Scale: FFT normalization factor Nd³/Ns³
        return out * (Nd / Ns) ** 3
    return jnp.stack([_trunc_3d(omega_hat_src[i]) for i in range(3)])
