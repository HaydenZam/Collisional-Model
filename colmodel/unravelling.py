"""Unravelling analysis of a Kossakowski matrix whose channels are grouped into baths.

Baths are a dict  name -> channel indices, e.g. the chain's
colmodel.master_equation.BATHS = {"L": (0, 1), "R": (2, 3)} or the toy model's
colmodel.toy.BATHS = {"A": (0,), "B": (1,)}.

Questions answered
------------------
* gksl_ok:              is D >= 0, i.e. a Lindblad generator (some unravelling exists)?
* strict_unravellable:  can every bath be monitored completely? <=> D is block
                        diagonal over the baths (no inter-bath blocks such as D_LR).
* monitorable_fraction: the largest t in [0, 1] such that D - t * (blocks of the
                        monitored baths) >= 0, i.e. how much of those baths'
                        dissipation can be peeled off as detector clicks while
                        the remaining (no-click) generator stays completely positive.
                        t = 1 when the baths are independent.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from .master_equation import BATHS
from .operators import hermitian_part, is_psd, min_eig

TOL = 1e-9


def gksl_ok(D, tol=TOL):
    return min_eig(D) >= -tol


def block_part(D, groups):
    """D with only the diagonal blocks D[g, g] of the given channel groups kept."""
    D = np.asarray(D)
    out = np.zeros_like(D)
    for g in groups:
        g = list(g)
        out[np.ix_(g, g)] = D[np.ix_(g, g)]
    return out


def _inter_bath_norm(D, baths):
    """sqrt( sum_{i<j} ||D[bath_i, bath_j]||_F^2 )."""
    D = np.asarray(D)
    gs = [list(g) for g in baths.values()]
    s = 0.0
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            s += np.linalg.norm(D[np.ix_(gs[i], gs[j])]) ** 2
    return float(np.sqrt(s))


def strict_unravellable(D, baths=BATHS, tol=TOL):
    """Every bath fully monitorable <=> all inter-bath blocks vanish."""
    return _inter_bath_norm(D, baths) <= tol


def monitorable_fraction(D, monitored="L", baths=BATHS, method="bisection"):
    """Largest t in [0, 1] with D - t * P >= 0, P = blocks of the monitored baths.

    monitored: an iterable of bath names, e.g. "L", "R", "LR" (both at once).
    method: "bisection" (robust, tolerance TOL on the eigenvalue) or "eig"
    (generalised eigenvalue problem P x = lam D x, t = 1/lam_max, with a root
    find when D is singular)."""
    P = block_part(D, [baths[k] for k in monitored])
    if method == "eig":
        return _fraction_eig(np.asarray(D), P)
    if method != "bisection":
        raise ValueError(f"unknown method {method!r}")
    if min_eig(D) < -TOL:                    # not even GKSL -> nothing to peel
        return 0.0
    if min_eig(D - P) >= -TOL:               # monitored blocks fully removable
        return 1.0
    lo, hi = 0.0, 1.0                        # min_eig(D - tP) decreases with t
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if min_eig(D - mid * P) >= -TOL:
            lo = mid
        else:
            hi = mid
    return lo


def _fraction_eig(D, P):
    if is_psd(D - P):
        return 1.0
    try:
        lam = eigh(hermitian_part(P), hermitian_part(D), eigvals_only=True)
        if lam.max() > 0:
            return float(min(1.0, 1.0 / lam.max()))
    except np.linalg.LinAlgError:
        pass
    f = lambda t: np.linalg.eigvalsh(hermitian_part(D - t * P)).min()
    return float(brentq(f, 0.0, 1.0, xtol=1e-12)) if f(0.0) > 0 else 0.0


# ---- closed forms for two baths (cross-checks) ------------------------------
def _two_blocks(D, baths, side):
    names = list(baths)
    if len(names) != 2:
        raise ValueError("closed forms need exactly two baths")
    a, c = (names[0], names[1]) if side == names[0] else (names[1], names[0])
    ia, ic = list(baths[a]), list(baths[c])
    D = np.asarray(D)
    return D[np.ix_(ia, ia)], D[np.ix_(ia, ic)], D[np.ix_(ic, ic)]


def fraction_schur(D, side="L", baths=BATHS):
    """t_max = lambda_min(A^{-1/2} S A^{-1/2}), S = A - B C^+ B^dag, with A the
    monitored block, C the other, B the cross block. None if A is singular."""
    A, B, C = _two_blocks(D, baths, side)
    S = A - B @ np.linalg.pinv(C) @ B.conj().T
    w, V = np.linalg.eigh(hermitian_part(A))
    if w.min() < 1e-12:
        return None
    A_is = V @ np.diag(w ** -0.5) @ V.conj().T
    return float(np.clip(min_eig(A_is @ S @ A_is), 0.0, 1.0))


def _inv_sqrt(X):
    w, V = np.linalg.eigh(hermitian_part(X))
    w = np.clip(w, 1e-15, None)
    return V @ np.diag(w ** -0.5) @ V.conj().T


def coupling_singular_values(D, baths=BATHS):
    """Singular values of M = A^{-1/2} B C^{-1/2} (normalised inter-bath coupling).
    Requires both bath blocks to be nonsingular."""
    A, B, C = _two_blocks(D, baths, list(baths)[0])
    return np.linalg.svd(_inv_sqrt(A) @ B @ _inv_sqrt(C), compute_uv=False)


def fraction_closed_form(D, baths=BATHS):
    """frac_L = frac_R = 1 - sigma_max(M)^2 (symmetric in the two baths).
    GKSL <=> sigma_max(M) <= 1."""
    return float(np.clip(1.0 - coupling_singular_values(D, baths).max() ** 2, 0.0, 1.0))


# ---- canonical (diagonal) channels ------------------------------------------
def canonical_channels(D, baths=BATHS, tol=1e-9):
    """Eigen-decomposition of D into canonical jump channels.

    Each channel L_k = sum_i v_i A_i has rate = eigenvalue and a participation
    in each bath (sum of |v_i|^2 over that bath's channels). A channel is
    delocalised (shared between baths) if more than one bath participates
    by more than 5 %. Returns (channels, sharedness), sharedness =
    ||inter-bath blocks||_F / ||D||_F in [0, 1]."""
    w, V = np.linalg.eigh(hermitian_part(D))
    chans = []
    for k in range(len(w)):
        if w[k] > tol:
            v = V[:, k]
            part = {name: float(np.abs(v[list(g)]) @ np.abs(v[list(g)])) for name, g in baths.items()}
            chans.append({"rate": float(w[k]), "participation": part,
                          "delocalized": sum(x > 0.05 for x in part.values()) > 1, "vec": v})
    shared = _inter_bath_norm(D, baths) / (np.linalg.norm(D) + 1e-30)
    return chans, float(shared)


def canonical_jump_operators(F, c):
    """Diagonalising jump operators: c = U d U^dag -> L_mu = sqrt(d_mu) sum_j U_j,mu F_j.
    Non-positive eigenvalues are dropped, so this is faithful only when c >= 0."""
    d, U = eigh(hermitian_part(c))
    return [np.sqrt(d[mu]) * sum(U[j, mu] * F[j] for j in range(len(F)))
            for mu in range(len(d)) if d[mu] > 1e-12]
