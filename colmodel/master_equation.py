"""Master equation for the collision model, and generic GKSL tools.

qutip is used internally to build superoperators and for mesolve; inputs and
outputs are numpy arrays. Superoperators act on column-stacked vectors,
vec(rho) = rho.reshape(-1, order="F") (qutip's convention); use vec/unvec.

Chain master equation (tau -> 0 limit of colmodel.collision)
-----------------------------------------------------------
Jump basis on the chain: A = (s+_1, s-_1, s+_N, s-_N). In the collision, A_a is
paired with the ancilla operator b_a = (s-_S, s+_S, s-_R, s+_R), with coupling
g_a = (gL, gL, gR, gR).

    d rho/dt = -i[H_C + H_1, rho] + sum_ab D_ab (A_a rho A_b^dag - 1/2 {A_b^dag A_a, rho})

* first order (drive):   H_1 = sum_a (g_a/sqrt(tau)) <b_a> A_a
  (finite only if the ancilla coherences scale as sqrt(tau)).
* second order (Kossakowski matrix, 4x4):  D_ab = g_a g_b <b_b^dag b_a>,
  i.e. raw second moments J = <ab>. In blocks,
      D_LL = gL^2 diag(p_s, 1 - p_s),   D_RR = gR^2 diag(p_r, 1 - p_r),
      D_LR = gL gR [[J_mp, J_mm], [J_pp, J_pm]],
  with J_xy = <s^x_S s^y_R> (m = -, p = +).
  connected=True subtracts v v^dag, v_a = g_a <b_a>, i.e. uses
  <ab> - <a><b>. The difference is the dissipator of the (unscaled) drive,
  D[v.A] = -1/2 [v.A, [v.A, rho]].
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm

from .model import chain_hamiltonian, ground_chain
from .operators import I2, N_OP, SM, SP, embed, kron, to_array

# ---------------------------------------------------------------------------
# generic GKSL tools
# ---------------------------------------------------------------------------


def vec(rho):
    return np.asarray(rho).reshape(-1, order="F")


def unvec(v, d=None):
    v = np.asarray(v)
    d = int(round(np.sqrt(v.size))) if d is None else d
    return v.reshape(d, d, order="F")


def _gksl_qobj(H, F, c, dims):
    """-i[H, .] + sum_jk c_jk (F_j . F_k^dag - 1/2 {F_k^dag F_j, .}) as a qutip superoperator."""
    q = lambda a: qt.Qobj(np.asarray(a, dtype=complex), dims=[dims, dims])
    Hq = q(H)
    L = -1j * (qt.spre(Hq) - qt.spost(Hq))
    Fq = [q(f) for f in F]
    for j in range(len(F)):
        for k in range(len(F)):
            if c[j, k] == 0:
                continue
            Fkd = Fq[k].dag()
            anti = Fkd * Fq[j]
            L = L + c[j, k] * (qt.spre(Fq[j]) * qt.spost(Fkd)
                               - 0.5 * qt.spre(anti) - 0.5 * qt.spost(anti))
    return L


def superoperator(H, F, c):
    """Dense GKSL superoperator (numpy, column-stacking vec) for Hamiltonian H,
    operator basis F and coefficient (Kossakowski) matrix c."""
    d = np.asarray(H).shape[0]
    return _gksl_qobj(H, F, np.asarray(c), [d]).full()


def evolve(L, rho0, t):
    """exp(L t) applied to rho0 (dense)."""
    d = np.asarray(rho0).shape[0]
    return unvec(expm(L * t) @ vec(rho0), d)


# ---------------------------------------------------------------------------
# chain master equation
# ---------------------------------------------------------------------------
JUMP_LABELS = ("s+_1", "s-_1", "s+_N", "s-_N")
BATHS = {"L": (0, 1), "R": (2, 3)}        # channel indices of each bath in the jump basis


def jump_operators(N):
    """A = (s+_1, s-_1, s+_N, s-_N) on the 2^N chain space."""
    return [embed(SP, 0, N), embed(SM, 0, N), embed(SP, N - 1, N), embed(SM, N - 1, N)]


def ancilla_moments(rho_SR):
    """Populations, means and second moments of the S-R ancilla state.

    Returns dict with p_s, p_r; means b = (<s-_S>, <s+_S>, <s-_R>, <s+_R>);
    J = {"mm": <s-_S s-_R>, "mp": <s-_S s+_R>, "pm": <s+_S s-_R>, "pp": <s+_S s+_R>}
    and the connected versions C = J - <.><.>."""
    rho = to_array(rho_SR)
    ev = lambda O: np.trace(rho @ O)
    ops1 = {"m": SM, "p": SP}
    mS = {k: ev(kron(o, I2)) for k, o in ops1.items()}
    mR = {k: ev(kron(I2, o)) for k, o in ops1.items()}
    J = {x + y: ev(kron(ops1[x], ops1[y])) for x in "mp" for y in "mp"}
    C = {x + y: J[x + y] - mS[x] * mR[y] for x in "mp" for y in "mp"}
    return {
        "p_s": ev(kron(N_OP, I2)).real,
        "p_r": ev(kron(I2, N_OP)).real,
        "means": np.array([mS["m"], mS["p"], mR["m"], mR["p"]]),
        "J": J,
        "C": C,
    }


def kossakowski(rho_SR, gL=0.2, gR=0.2, connected=False):
    """4x4 Kossakowski matrix D in the jump basis (s+_1, s-_1, s+_N, s-_N).

    gL, gR are the effective couplings (Params.gL, Params.gR). Returns
    (D, blocks) with blocks = dict(D_LL, D_RR, D_LR, p_s, p_r)."""
    m = ancilla_moments(rho_SR)
    p_s, p_r, J = m["p_s"], m["p_r"], m["J"]
    D_LL = gL ** 2 * np.diag([p_s, 1.0 - p_s]).astype(complex)
    D_RR = gR ** 2 * np.diag([p_r, 1.0 - p_r]).astype(complex)
    D_LR = gL * gR * np.array([[J["mp"], J["mm"]],
                               [J["pp"], J["pm"]]], dtype=complex)
    D = np.block([[D_LL, D_LR], [D_LR.conj().T, D_RR]])
    D = 0.5 * (D + D.conj().T)                       # remove round-off asymmetry
    if connected:
        v = np.array([gL, gL, gR, gR]) * m["means"]
        D = D - np.outer(v, v.conj())
        D_LL, D_RR, D_LR = D[:2, :2], D[2:, 2:], D[:2, 2:]
    return D, {"D_LL": D_LL, "D_RR": D_RR, "D_LR": D_LR, "p_s": p_s, "p_r": p_r}


def drive_hamiltonian(params, rho_SR):
    """H_1 = gL/sqrt(tau) (<s-_S> s+_1 + <s+_S> s-_1) + gR/sqrt(tau) (<s-_R> s+_N + <s+_R> s-_N)."""
    b = ancilla_moments(rho_SR)["means"]
    g = np.array([params.gL_col, params.gL_col, params.gR_col, params.gR_col])
    return sum(g[a] * b[a] * A for a, A in enumerate(jump_operators(params.N)))


def _chain_generator(params, rho_SR, nonlocal_terms=True, connected=False):
    H = chain_hamiltonian(params) + drive_hamiltonian(params, rho_SR)
    D, _ = kossakowski(rho_SR, params.gL, params.gR, connected=connected)
    if not nonlocal_terms:
        D[:2, 2:] = 0
        D[2:, :2] = 0
    return H, jump_operators(params.N), D


def liouvillian(params, rho_SR, nonlocal_terms=True, connected=False):
    """Dense chain Liouvillian (numpy, column-stacking vec).

    nonlocal_terms=False drops the D_LR block (the "no D_LR" comparison)."""
    H, F, D = _chain_generator(params, rho_SR, nonlocal_terms, connected)
    return _gksl_qobj(H, F, D, [2] * params.N).full()


def solve(params, rho_SR, times, rho_C0=None, nonlocal_terms=True, connected=False,
          method="mesolve", options=None):
    """Chain states at `times` (array (n_t, 2^N, 2^N)).

    method="mesolve": qutip.mesolve (sparse; `options` passed through).
    method="expm":    exact propagation exp(L dt) on a uniform time grid (dense,
                      N up to about 6)."""
    N = params.N
    rho0 = ground_chain(N) if rho_C0 is None else np.asarray(rho_C0, dtype=complex)
    H, F, D = _chain_generator(params, rho_SR, nonlocal_terms, connected)
    L = _gksl_qobj(H, F, D, [2] * N)
    times = np.asarray(times, dtype=float)
    if method == "mesolve":
        res = qt.mesolve(L, qt.Qobj(rho0, dims=[[2] * N, [2] * N]), times, options=options)
        return np.array([s.full() for s in res.states])
    if method == "expm":
        dt = np.diff(times)
        if len(dt) and not np.allclose(dt, dt[0], rtol=1e-9, atol=0):
            raise ValueError("method='expm' needs a uniform time grid")
        out = np.empty((len(times), 2 ** N, 2 ** N), dtype=complex)
        v = vec(evolve(L.full(), rho0, times[0])) if times[0] != 0 else vec(rho0)
        out[0] = unvec(v, 2 ** N)
        if len(dt):
            P = expm(L.full() * dt[0])
            for i in range(1, len(times)):
                v = P @ v
                out[i] = unvec(v, 2 ** N)
        return out
    raise ValueError(f"unknown method {method!r}")
