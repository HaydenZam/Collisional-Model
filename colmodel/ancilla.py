"""Two-qubit ancilla states for the sender (S) and receiver (R) ancillas.

All states are 4x4 complex numpy arrays on H_S (x) H_R, basis |00>,|01>,|10>,|11>
(conventions in colmodel.operators: |0> = ground, sigma^z = +1 on |1>).

Two families
------------
* Zero local coherence: <sigma^+-> = 0 on both marginals, so all the structure
  is in the correlations. These need no rescaling in the collision model.
  -> representative("classical" | "discord" | "entangled"), bell_diagonal,
     thermal_correlated, random_state.
* With local coherence: the first-order (drive) term of the master equation
  is finite only if the coherences scale as sqrt(tau). Use
  collision_scaled_product() or coherent_product() with sqrt(tau)-scaled
  coherences before passing such a state to the dynamics.
  -> representative("product"), coherent_product.

Coherence parameters are always c = <sigma^-> = rho[1, 0], the amplitude that
multiplies sigma^+ of the chain in the drive Hamiltonian.
"""

import numpy as np

from .operators import I2, KET0, KET1, PAULI, SX, SY, SZ, kron, marginals, proj

KETP = (KET0 + KET1) / np.sqrt(2)          # |+>

REPRESENTATIVES = ("product", "classical", "discord", "entangled")


# ---- building blocks ------------------------------------------------------
def bell_diagonal(c1, c2, c3):
    """rho = 1/4 (I(x)I + c1 sx(x)sx + c2 sy(x)sy + c3 sz(x)sz).

    Both marginals are maximally mixed (zero local coherence). Moving (c1,c2,c3)
    sweeps the correlation classes:
      * at most 1 nonzero component      -> classical (zero discord)
      * >= 2 nonzero, inside tetrahedron  -> discordant but separable
      * outside the separability octahedron -> entangled
    The state is physical inside the tetrahedron with vertices
    (1,1,-1), (1,-1,1), (-1,1,1), (-1,-1,-1)."""
    return 0.25 * (kron(I2, I2) + c1 * kron(SX, SX) + c2 * kron(SY, SY) + c3 * kron(SZ, SZ))


def thermal_correlated(p_s, p_r, t1=0.0, t2=0.0, t3=0.0):
    """Diagonal (thermal) marginals with excited populations p_s, p_r and a
    diagonal correlation matrix diag(t1, t2, t3); zero local coherence.

        rho = 1/4 [ I(x)I + a sz(x)I + b I(x)sz + t1 sx(x)sx + t2 sy(x)sy + t3 sz(x)sz ],
        a = 2 p_s - 1,  b = 2 p_r - 1.

    p = 1/(1 + exp(beta omega)): p -> 0 cold, p -> 1/2 hot. (t1, t2) are the
    transverse correlations that feed the non-local Kossakowski block D_LR;
    t3 only moves the correlation class. Reduces to bell_diagonal at
    p_s = p_r = 1/2. Positivity is NOT guaranteed: check with
    colmodel.operators.is_state."""
    a, b = 2 * p_s - 1, 2 * p_r - 1
    rho = 0.25 * (kron(I2, I2) + a * kron(SZ, I2) + b * kron(I2, SZ)
                  + t1 * kron(SX, SX) + t2 * kron(SY, SY) + t3 * kron(SZ, SZ))
    return 0.5 * (rho + rho.conj().T)


def coherent_qubit(p, c):
    """Single-qubit state with excited population p and <sigma^-> = c."""
    if abs(c) ** 2 > p * (1 - p) + 1e-15:
        raise ValueError(f"not a state: |c|^2 = {abs(c)**2:.3g} > p(1-p) = {p*(1-p):.3g}")
    return np.array([[1 - p, np.conj(c)], [c, p]], dtype=complex)


def coherent_product(p_s, c_s, p_r, c_r):
    """Uncorrelated S-R state; each ancilla has excited population p and <sigma^-> = c.
    For the collision model pass sqrt(tau)-scaled coherences."""
    return np.kron(coherent_qubit(p_s, c_s), coherent_qubit(p_r, c_r))


def collision_scaled_product(rho, tau):
    """Product of the two marginals of rho, with each local coherence <sigma^+->
    multiplied by sqrt(tau) and the populations kept.

    This is the form an ancilla with local coherence must take for the drive
    term of the master equation to stay finite as tau -> 0. Correlations of rho
    are discarded."""
    out = []
    for m in marginals(rho):
        m = m.copy()
        m[0, 1] *= np.sqrt(tau)
        m[1, 0] *= np.sqrt(tau)
        out.append(m)
    return np.kron(out[0], out[1])


# ---- fixed representatives ------------------------------------------------
def _product_rep():
    """|+>_S (x) |+>_R: uncorrelated. The one representative with local coherence."""
    return proj(kron(KETP, KETP))


def _classical_rep():
    """Bell-diagonal (1,0,0) = 1/2 (|++><++| + |--><--|): classical correlation,
    zero coherence, zero discord. The correlation lies along x, so the
    sigma^+- correlators are nonzero (<s+ (x) s+> = <s+ (x) s-> = 1/4); along z it
    would only be seen by sz(x)sz."""
    return bell_diagonal(1.0, 0.0, 0.0)


def _discord_rep():
    """Bell-diagonal (0.5, 0.5, 0): separable, nonzero discord, zero coherence."""
    return bell_diagonal(0.5, 0.5, 0.0)


def _entangled_rep():
    """Bell state |Phi+> = (|00> + |11>)/sqrt(2) = Bell-diagonal (1, -1, 1)."""
    return proj((kron(KET0, KET0) + kron(KET1, KET1)) / np.sqrt(2))


_REPS = {
    "product": _product_rep,
    "classical": _classical_rep,
    "discord": _discord_rep,
    "entangled": _entangled_rep,
}


def representative(name):
    """One of REPRESENTATIVES ("product", "classical", "discord", "entangled")."""
    try:
        return _REPS[name]()
    except KeyError:
        raise ValueError(f"unknown representative {name!r}; expected one of {REPRESENTATIVES}") from None


def mixture():
    """Equal mixture of the four representatives (product unscaled)."""
    return sum(f() for f in _REPS.values()) / len(_REPS)


# ---- random zero-local-coherence state ------------------------------------
def random_state(seed=None, strength=None, rng=None):
    """Random two-qubit state with z-diagonal marginals (zero local coherence).

    Builds M = a_z sz(x)I + b_z I(x)sz + sum_ij T_ij s_i(x)s_j with a_z, b_z uniform
    in [-1, 1] and T a standard-normal 3x3 matrix, then
    rho = 1/4 (I(x)I + u t_max M), where t_max is the largest scale keeping rho
    PSD and u = `strength` (uniform in [0, 1) if omitted). The correlation
    matrix is unconstrained, so the state can land in any correlation class.

    Pass either `seed` or a numpy Generator `rng`."""
    if rng is None:
        rng = np.random.default_rng(seed)
    az = rng.uniform(-1, 1)
    bz = rng.uniform(-1, 1)
    T = rng.standard_normal((3, 3))
    M = az * kron(SZ, I2) + bz * kron(I2, SZ)
    for i in range(3):
        for j in range(3):
            M = M + T[i, j] * kron(PAULI[i], PAULI[j])
    lam_min = np.linalg.eigvalsh(M).min().real          # rho >= 0  <=>  1 + t lam_min >= 0
    t_max = 1.0 / max(1e-12, -lam_min)
    u = rng.uniform(0.0, 1.0) if strength is None else float(strength)
    rho = 0.25 * (kron(I2, I2) + (u * t_max) * M)
    return 0.5 * (rho + rho.conj().T)


def named_state(name, *, seed=None, strength=None):
    """Dispatcher: a representative name, "mixture", or "random"."""
    if name == "random":
        return random_state(seed=seed, strength=strength)
    if name == "mixture":
        return mixture()
    return representative(name)
