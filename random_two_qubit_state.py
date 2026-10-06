"""
Random / representative two-qubit state generator (zero-local-coherence variant).

Basis ordering: |00>, |01>, |10>, |11>, with qubit A the most significant
(i.e. rho acts on H_A ⊗ H_B). All states returned as 4x4 complex numpy arrays.

Design choice
-------------
Except for the explicit "product" representative, EVERY state produced here has
**zero local coherence**: <sigma^+ ⊗ I> = <I ⊗ sigma^+> = 0 (and likewise
sigma^-). Both reduced states are diagonal in the z-basis, so all the structure
lives in the correlations, not in local Bloch coherences. This isolates genuine
correlations (classical / discord / entanglement) from local-coherence
bookkeeping.

    product      -> uncorrelated, DOES carry local coherence (the exception)
    classical    -> classically correlated, zero coherence, zero discord
    discord      -> separable but discordant, zero coherence
    entangled    -> Bell state, zero coherence
    random       -> random state with z-diagonal marginals (zero coherence),
                    arbitrary correlation matrix (any correlation class)
    mixture      -> equal convex combination of the four representatives
"""

import numpy as np

from qutip import basis , qeye , tensor , Qobj , mesolve , expect

# ---- single-qubit building blocks -----------------------------------------
_ket0 = np.array([1, 0], dtype=complex)
_ket1 = np.array([0, 1], dtype=complex)
_ketp = (_ket0 + _ket1) / np.sqrt(2)          # |+>

_I = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = (_sx, _sy, _sz)


def _proj(psi):
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


_P0 = _proj(_ket0)     # |0><0|
_P1 = _proj(_ket1)     # |1><1|
_Pp = _proj(_ketp)     # |+><+|
_kron = np.kron


# ---- Bell-diagonal helper --------------------------------------------------
def bell_diagonal(c1, c2, c3):
    """rho = 1/4 (I⊗I + c1 σx⊗σx + c2 σy⊗σy + c3 σz⊗σz).

    Both marginals are maximally mixed, so local coherence is identically zero.
    Moving (c1,c2,c3) sweeps the correlation classes:
      * <=1 nonzero component          -> classical (zero discord)
      * >=2 nonzero, inside tetrahedron-> discordant but separable
      * outside separability tetrahedron-> entangled
    Tetrahedron vertices: (1,1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1)."""
    return 0.25 * (_kron(_I, _I) + c1 * _kron(_sx, _sx)
                   + c2 * _kron(_sy, _sy) + c3 * _kron(_sz, _sz))


# ---- fixed representatives -------------------------------------------------
def _product_rep():
    """|+>_A ⊗ |0>_B — uncorrelated. The ONE case that carries local coherence."""
    return _kron(_Pp, _Pp)


def _classical_rep():
    """Bell-diagonal (1,0,0) = 1/2 (|++><++| + |--><--|) — classical correlation,
    zero coherence, zero discord, correlated in the x-y plane so that sigma^+/-
    correlators are nonzero (<s+ x s+> = <s+ x s-> = 1/4). A single nonzero
    correlation component keeps discord zero; putting it along x rather than z
    makes it visible to sigma^+/- rather than only to sigma^z sigma^z."""
    return bell_diagonal(1.0, 0.0, 0.0)


def _discord_rep():
    """Bell-diagonal (0.5, 0.5, 0) — separable, nonzero discord, zero coherence.
    Two nonzero correlation components => no local basis diagonalizes it
    (discord > 0), but the point stays inside the separability tetrahedron."""
    return bell_diagonal(0.5, 0.5, 0.0)


def _entangled_rep():
    """Bell state |Phi+> = (|00> + |11>)/sqrt(2) — maximally entangled, zero coherence.
    Equivalent to Bell-diagonal (1, -1, 1)."""
    phi = (_kron(_ket0, _ket0) + _kron(_ket1, _ket1)) / np.sqrt(2)
    return _proj(phi)


_REPS = {
    "product": _product_rep,
    "classical": _classical_rep,
    "discord": _discord_rep,
    "entangled": _entangled_rep,
}


# ---- random zero-local-coherence state ------------------------------------
def _random_state(rng, strength=None):
    """Random 2-qubit state with z-diagonal marginals (zero local coherence).

    Construction: build the traceless part
        M = a_z σz⊗I + b_z I⊗σz + Σ_ij T_ij σ_i⊗σ_j
    with random z-marginals a_z,b_z and a random 3x3 correlation matrix T (so
    only σz survives in each marginal => zero σ± coherence by construction).
    Then rho = 1/4 (I⊗I + t·M), with t scaled to the largest value keeping
    rho PSD, times a random fraction `strength` in (0,1] controlling how
    strongly correlated / how pure the state is. The correlation matrix is
    unconstrained, so the output can land in any correlation class."""
    az = rng.uniform(-1, 1)
    bz = rng.uniform(-1, 1)
    T = rng.standard_normal((3, 3))
    M = az * _kron(_sz, _I) + bz * _kron(_I, _sz)
    for i in range(3):
        for j in range(3):
            M += T[i, j] * _kron(_PAULI[i], _PAULI[j])
    lam_min = np.linalg.eigvalsh(M).min().real          # rho >= 0  <=>  1 + t*lam_min >= 0
    t_max = 1.0 / max(1e-12, -lam_min)
    u = rng.uniform(0.0, 1.0) if strength is None else float(strength)
    rho = 0.25 * (_kron(_I, _I) + (u * t_max) * M)
    return (rho + rho.conj().T) / 2                      # symmetrize away round-off


# ---- public API ------------------------------------------------------------
def random_two_qubit_state(mode=None, *, seed=None, strength=None):
    """Generate a two-qubit density matrix (4x4 complex numpy array).

    Parameters
    ----------
    mode : None | "random" | "product" | "classical" | "discord" | "entangled" | "mixture"
        None / "random" (default): random state with zero local coherence and an
        arbitrary correlation matrix (may be classical, discordant, or entangled).
        "product": uncorrelated state that DOES carry local coherence (exception).
        "classical" / "discord" / "entangled": fixed zero-coherence representatives.
        "mixture": equally weighted convex combination of the four representatives.
    seed : int, optional
        Seed for the random generator (only affects the random mode).
    strength : float in (0,1], optional
        For the random mode only: fraction of the maximal PSD-preserving scale.
        Larger => more strongly correlated / closer to pure. Random if omitted.
    """
    if mode in (None, "random"):
        return _random_state(np.random.default_rng(seed), strength=strength)
    if mode in _REPS:
        return _REPS[mode]()
    if mode == "mixture":
        return sum(f() for f in _REPS.values()) / len(_REPS)
    raise ValueError(
        f"unknown mode {mode!r}; expected one of None, 'random', "
        f"'product', 'classical', 'discord', 'entangled', 'mixture'"
    )

ket0 = basis(2, 0)
ket1 = basis(2, 1)

I = qeye(2)

sp = ket1 * ket0.dag()   # sigma^+ = |1><0|
sm = ket0 * ket1.dag()   # sigma^- = |0><1|

n_op = ket1 * ket1.dag()
sz = ket1 * ket1.dag() - ket0 * ket0.dag()

def collision_product_state(tau, joint_state):
    """
    Returns the state of the ancilla which is correctly accounts for the requirement of a finite coherant drive
    i.e., that remains valid for any tau
    """
    rho_s = joint_state.ptrace(0)
    rho_r = joint_state.ptrace(1)
    
    p_s = expect(n_op, rho_s).real
    p_r = expect(n_op, rho_r).real
    
    # single-ancilla coherences (drive):  chain sigma^+ carries <sigma^->, chain sigma^- carries <sigma^+>
    c_s = expect(sm, rho_s)   # <sigma_S^->
    c_r = expect(sm, rho_r)   # <sigma_S^+>
    
    rho_s_col = Qobj([[1 - p_s, c_s*np.sqrt(tau)],
              [np.conjugate(c_s*np.sqrt(tau)), p_s]])
    
    rho_r_col = Qobj([[1 - p_r, c_r*np.sqrt(tau)],
              [np.conjugate(c_r*np.sqrt(tau)), p_r]])
    
    return tensor(rho_s_col, rho_r_col)


# ---- diagnostics -----------------------------------------------------------
def local_coherence(rho):
    """Max |<sigma^+/->| over both marginals; 0  <=>  z-diagonal marginals."""
    vals = []
    for P in (_sx, _sy):
        vals.append(abs(np.trace(rho @ _kron(P, _I))))
        vals.append(abs(np.trace(rho @ _kron(_I, P))))
    return float(max(vals))


def negativity(rho):
    """Entanglement negativity via the partial transpose on B.

    N(rho) = sum of |negative eigenvalues of rho^{T_B}|.
    For two qubits, N > 0  <=>  entangled (Peres–Horodecki is necessary and
    sufficient in 2x2), N = 0  <=>  separable."""
    r = rho.reshape(2, 2, 2, 2)
    rt = r.transpose(0, 3, 2, 1).reshape(4, 4)   # transpose subsystem B
    ev = np.linalg.eigvalsh(rt)
    return float(np.sum(np.abs(ev[ev < 0])))


def zz_covariance(rho):
    """Connected correlator C(sigma_z, sigma_z) = <ZZ> - <Z_A><Z_B>."""
    ZZ = _kron(_sz, _sz)
    ZA = _kron(_sz, _I)
    IB = _kron(_I, _sz)
    ev = lambda O: np.trace(rho @ O).real
    return ev(ZZ) - ev(ZA) * ev(IB)


def _vn_entropy(rho):
    """von Neumann entropy in bits."""
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


# Quantum discord
def _ptrace_B(M):
    return np.einsum('abcb->ac', M.reshape(2, 2, 2, 2))


def _ptrace_A(M):
    return np.einsum('abac->bc', M.reshape(2, 2, 2, 2))

def discord(rho, n_grid=48):
    """Quantum discord D_B(rho), measuring subsystem B (in bits).

    D_B = I(rho) - J_B(rho), where I = S(A)+S(B)-S(AB) is the total (mutual-
    information) correlation and J_B = S(A) - min_{measurements on B} sum_k p_k
    S(rho_{A|k}) is the part extractable by a local projective measurement on B.
    The minimisation runs over projectors |n><n| on B via a grid over the Bloch
    sphere. D_B = 0  <=>  zero discord (classical w.r.t. B); D_B > 0 signals
    genuinely quantum (measurement-disturbing) correlations even when separable.
    Note discord is asymmetric; measure_on='A' would give D_A."""
    S_AB = _vn_entropy(rho)
    S_A = _vn_entropy(_ptrace_B(rho))
    S_B = _vn_entropy(_ptrace_A(rho))
    I_mut = S_A + S_B - S_AB

    def conditional(angles):
        th, ph = angles
        nx = np.sin(th) * np.cos(ph)
        ny = np.sin(th) * np.sin(ph)
        nz = np.cos(th)
        cond = 0.0
        for s in (1, -1):
            Pi = 0.5 * (_I + s * (nx * _sx + ny * _sy + nz * _sz))
            M = _kron(_I, Pi) @ rho @ _kron(_I, Pi)
            pk = np.trace(M).real
            if pk > 1e-12:
                cond += pk * _vn_entropy(_ptrace_B(M) / pk)
        return cond

    best, best_ang = np.inf, (0.0, 0.0)
    for th in np.linspace(0, np.pi, n_grid):
        for ph in np.linspace(0, 2 * np.pi, 2 * n_grid, endpoint=False):
            c = conditional((th, ph))
            if c < best:
                best, best_ang = c, (th, ph)
    span_th, span_ph = np.pi / n_grid, 2 * np.pi / (2 * n_grid)
    for _ in range(8):                      # coarse-to-fine zoom around the optimum
        th0, ph0 = best_ang
        for th in np.linspace(th0 - span_th, th0 + span_th, 7):
            for ph in np.linspace(ph0 - span_ph, ph0 + span_ph, 7):
                c = conditional((th, ph))
                if c < best:
                    best, best_ang = c, (th, ph)
        span_th *= 0.4
        span_ph *= 0.4
    return float(max(0.0, I_mut - (S_A - best)))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print(f"{'mode':>10s}  {'coherence':>9s}  {'negativity':>10s}  {'C(Z,Z)':>8s}")
    for m in ["product", "classical", "discord", "entangled", "mixture"]:
        rho = random_two_qubit_state(m)
        print(f"{m:>10s}  {local_coherence(rho):9.4f}  "
              f"{negativity(rho):10.4f}  {zz_covariance(rho):+8.4f}")
    print("  --- random draws (all zero coherence) ---")
    for k in range(4):
        rho = random_two_qubit_state(seed=k)
        cls = "entangled" if negativity(rho) > 1e-9 else "separable"
        print(f"{'random':>10s}  {local_coherence(rho):9.4f}  "
              f"{negativity(rho):10.4f}  {zz_covariance(rho):+8.4f}   {cls}")
