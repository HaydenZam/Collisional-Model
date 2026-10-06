"""
unravelling_test.py
===================

Single-bath unravelling test for the two-bath collisional model.

Focus: option (b) -- the graded "maximum monitorable fraction" of one bath's
dissipation such that the complementary (no-click) generator stays completely
positive (CP). Option (a) (strict: remove the full bath block) is reported as
the trivial limit, since it holds iff there are no non-local terms at all:

        strict (a) holds   <=>   D_LR == 0   <=>   fraction (b) == 1.

Physical reduction
------------------
The jump operators live only on the two end sites of the chain, so the
Kossakowski matrix in the physical (non-diagonal) basis

        (sigma^+_L, sigma^-_L, sigma^+_R, sigma^-_R)

is 4x4 regardless of chain length N or the internal hopping J:

        D = [[ D_LL , D_LR ],
             [ D_LR*, D_RR ]]

        D_LL = gL^2 diag(p_s, 1 - p_s)          left  emission / absorption rates
        D_RR = gR^2 diag(p_r, 1 - p_r)          right emission / absorption rates
        D_LR = gL gR [[<s-_S s+_R>, <s-_S s-_R>],
                      [<s+_S s+_R>, <s+_S s-_R>]]   ancilla two-point correlators

with p_s = <n_S>, p_r = <n_R> the ancilla excited populations and
s+- = sigma^+- the ladder operators. Only *transverse* (x-y) ancilla
correlations feed D_LR; a pure z-z (classical) correlation gives D_LR = 0.

Because the CP-ness of the (b) split is a property of this 4x4 D alone, the
whole unravelling question is independent of N, of the chain Hamiltonian and
of the hopping -- they only ever enter the coherent part, which any CP
generator is free to carry.

Tests implemented
-----------------
    gksl_ok(D)              min eig D >= 0   (diagonal-basis unravelling exists)
    strict_unravellable     option (a); True iff D_LR == 0
    monitorable_fraction    option (b); max t in [0,1] with D - t*embed(D_side) PSD
                            (bisection, cross-checked against a Schur closed form)

Author: prepared for the Collisional Model project.
"""

import os
import numpy as np

from random_two_qubit_state import (
    random_two_qubit_state, bell_diagonal, negativity, discord,
)

# --------------------------------------------------------------------------
# single-qubit operators (convention from Collision_Model.py: |0>=ground)
# --------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_sp = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma^+ = |1><0|  (raising)
_sm = np.array([[0, 1], [0, 0]], dtype=complex)   # sigma^- = |0><1|  (lowering)
_n = np.array([[0, 0], [0, 1]], dtype=complex)    # |1><1|
_kron = np.kron

TOL = 1e-9


def _to_ndarray(rho):
    if hasattr(rho, "full"):
        rho = rho.full()
    return np.asarray(rho, dtype=complex)


# ==========================================================================
# 1.  Kossakowski matrix
# ==========================================================================
def kossakowski(rho_SR, gL=0.2, gR=0.2):
    """4x4 Kossakowski matrix in basis (s+_L, s-_L, s+_R, s-_R).

    rho_SR : 2-qubit ancilla state (S = sender/left, R = receiver/right),
             ordering H_S (x) H_R, basis |00>,|01>,|10>,|11>.
    Returns (D, blocks) with blocks = dict(D_LL, D_RR, D_LR).
    """
    rho = _to_ndarray(rho_SR)

    p_s = np.trace(rho @ _kron(_n, _I)).real
    p_r = np.trace(rho @ _kron(_I, _n)).real

    # ancilla ladder correlators (match the J_.. labels in Collision_Model.py)
    J_mm = np.trace(rho @ _kron(_sm, _sm))   # <s-_S s-_R>
    J_mp = np.trace(rho @ _kron(_sm, _sp))   # <s-_S s+_R>
    J_pm = np.trace(rho @ _kron(_sp, _sm))   # <s+_S s-_R>
    J_pp = np.trace(rho @ _kron(_sp, _sp))   # <s+_S s+_R>

    D_LL = gL ** 2 * np.diag([p_s, 1.0 - p_s]).astype(complex)
    D_RR = gR ** 2 * np.diag([p_r, 1.0 - p_r]).astype(complex)
    D_LR = gL * gR * np.array([[J_mp, J_mm],
                               [J_pp, J_pm]], dtype=complex)

    D = np.block([[D_LL, D_LR], [D_LR.conj().T, D_RR]])
    D = 0.5 * (D + D.conj().T)                      # kill round-off asymmetry
    return D, {"D_LL": D_LL, "D_RR": D_RR, "D_LR": D_LR, "p_s": p_s, "p_r": p_r}


# ==========================================================================
# 2.  Tests
# ==========================================================================
def _min_eig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.conj().T)).min().real)


def gksl_ok(D):
    """Does the full generator stay Lindblad? (diagonal-basis unravelling)."""
    return _min_eig(D) >= -TOL


def strict_unravellable(D):
    """Option (a): remove the whole monitored block, demand CP complement.
    Provably holds iff the off-diagonal (non-local) block vanishes."""
    D_LR = D[:2, 2:]
    return float(np.linalg.norm(D_LR)) <= TOL


def _embed_block(D, side):
    """Projector-weighted block: the physical rate matrix of one bath,
    embedded in the 4x4 space, to be peeled off as detected clicks."""
    P = np.zeros((4, 4), dtype=complex)
    if side == "L":
        P[:2, :2] = D[:2, :2]
    else:
        P[2:, 2:] = D[2:, 2:]
    return P


def monitorable_fraction(D, side="L"):
    """Option (b): largest t in [0,1] with  D - t * embed(D_side)  PSD.

    t = 1 means the full bath can be monitored (equivalent to strict (a));
    t = 0 means not even an infinitesimal amount of that bath is monitorable
    while keeping the no-click evolution CP. Robust to singular bath blocks.
    """
    if _min_eig(D) < -TOL:                 # not even GKSL -> nothing to peel
        return 0.0
    P = _embed_block(D, side)
    if _min_eig(D - P) >= -TOL:            # full block removable
        return 1.0
    lo, hi = 0.0, 1.0                      # min_eig(D - tP) is monotone decreasing in t
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _min_eig(D - mid * P) >= -TOL:
            lo = mid
        else:
            hi = mid
    return lo


def _fraction_schur(D, side="L"):
    """Closed-form cross-check of monitorable_fraction.

    t_max = lambda_min( A^{-1/2} S A^{-1/2} ),  S = A - B C^{-1} B^dag,
    where A = monitored-bath block, C = other-bath block, B = cross block.
    Valid when A, C are well conditioned; used only as a self-test.
    """
    if side == "L":
        A, B, C = D[:2, :2], D[:2, 2:], D[2:, 2:]
    else:
        A, B, C = D[2:, 2:], D[2:, :2], D[:2, :2]
    S = A - B @ np.linalg.pinv(C) @ B.conj().T
    w, V = np.linalg.eigh(0.5 * (A + A.conj().T))
    if w.min() < 1e-12:
        return None                        # A singular -> skip closed form
    A_inv_sqrt = V @ np.diag(w ** -0.5) @ V.conj().T
    M = A_inv_sqrt @ S @ A_inv_sqrt
    return float(np.clip(np.linalg.eigvalsh(0.5 * (M + M.conj().T)).min().real, 0.0, 1.0))


def _inv_sqrt(X):
    w, V = np.linalg.eigh(0.5 * (X + X.conj().T))
    w = np.clip(w, 1e-15, None)
    return V @ np.diag(w ** -0.5) @ V.conj().T


def coupling_singular_values(D):
    """Singular values of the normalized inter-bath coupling
    M = D_LL^{-1/2} D_LR D_RR^{-1/2}. Requires p_s, p_r in (0,1)."""
    A, B, C = D[:2, :2], D[:2, 2:], D[2:, 2:]
    M = _inv_sqrt(A) @ B @ _inv_sqrt(C)
    return np.linalg.svd(M, compute_uv=False)


def fraction_closed_form(D):
    """frac_L = frac_R = 1 - sigma_max(M)^2  (identity, symmetric in the baths).
    Also: GKSL positivity <=> sigma_max(M) <= 1 <=> fraction >= 0."""
    return float(np.clip(1.0 - coupling_singular_values(D).max() ** 2, 0.0, 1.0))


# ==========================================================================
# 3.  Correlation-class label
# ==========================================================================
def classify(rho, disc_grid=20):
    """Coarse correlation class of a 2-qubit state."""
    rho = _to_ndarray(rho)
    if negativity(rho) > 1e-7:
        return "entangled"
    if discord(rho, n_grid=disc_grid) > 1e-4:
        return "discord"
    # separable & zero-discord: correlated (classical) or product?
    rA = np.einsum("abcb->ac", rho.reshape(2, 2, 2, 2))
    rB = np.einsum("abac->bc", rho.reshape(2, 2, 2, 2))
    if np.linalg.norm(rho - _kron(rA, rB)) > 1e-7:
        return "classical"
    return "product"


# ==========================================================================
# 3b. Thermal-marginal correlated ancilla family (no local coherence)
# ==========================================================================
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)


def thermal_correlated_state(p_s, p_r, t1, t2, t3):
    """Two-qubit ancilla with diagonal (thermal) marginals and no local coherence.

        rho = 1/4 [ I(x)I + a sz(x)I + b I(x)sz + t1 sx(x)sx + t2 sy(x)sy + t3 sz(x)sz ]
        a = 2 p_s - 1,  b = 2 p_r - 1

    p_s, p_r are the excited-state populations of the sender / receiver ancilla
    (p = 1/(1+exp(beta*omega)); p->1/2 hot, p->0 cold). Because only sz appears
    at single-body order, both marginals are z-diagonal => zero local coherence.
    The correlation matrix is diag(t1,t2,t3); (t1,t2) are the transverse
    correlations that feed D_LR, t3 is the z-z correlation that only moves the
    correlation class. Reduces to Bell-diagonal when p_s=p_r=1/2.

    Returns the 4x4 state; PSD is not guaranteed -- use physical_thermal() /
    check eigenvalues before trusting it.
    """
    a, b = 2 * p_s - 1, 2 * p_r - 1
    rho = 0.25 * (_kron(_I, _I) + a * _kron(_sz, _I) + b * _kron(_I, _sz)
                  + t1 * _kron(_sx, _sx) + t2 * _kron(_sy, _sy) + t3 * _kron(_sz, _sz))
    return 0.5 * (rho + rho.conj().T)


def is_physical(rho, tol=1e-12):
    return np.linalg.eigvalsh(_to_ndarray(rho)).min() >= -tol


# ==========================================================================
# 3c. One-bath-vs-two diagnostic: canonical (delocalized) jump channels
# ==========================================================================
def canonical_channels(D, tol=1e-9):
    """Diagonalize the Kossakowski matrix into canonical jump operators.

    Each channel is an eigen-jump-operator L_k = sum_i v_i A_i with
    A = (s+_L, s-_L, s+_R, s-_R), rate = eigenvalue. Left/right participation
    pL = |v0|^2+|v1|^2, pR = |v2|^2+|v3|^2 tells whether the channel is local
    (pL or pR ~ 1) or delocalized/shared (both ~ 1/2). Returns a list of dicts
    plus a global 'sharedness' = ||D_LR||_F / ||D||_F in [0,1].
    """
    w, V = np.linalg.eigh(0.5 * (D + D.conj().T))
    chans = []
    for k in range(4):
        if w[k] > tol:
            v = V[:, k]
            pL = float(np.abs(v[:2]) @ np.abs(v[:2]))
            pR = float(np.abs(v[2:]) @ np.abs(v[2:]))
            chans.append({"rate": float(w[k]), "pL": pL, "pR": pR,
                          "delocalized": min(pL, pR) > 0.05, "vec": v})
    shared = float(np.linalg.norm(D[:2, 2:]) / (np.linalg.norm(D) + 1e-30))
    return chans, shared


# ==========================================================================
# 4.  Verification: analytic D  ==  exact dissipative superoperator (N=2)
# ==========================================================================
def _superop(action, dim):
    """Matrix of a linear map rho -> action(rho) by acting on basis matrices."""
    n = dim * dim
    M = np.zeros((n, n), dtype=complex)
    for k in range(n):
        E = np.zeros((dim, dim), dtype=complex)
        E.flat[k] = 1.0
        M[:, k] = action(E).flatten()
    return M


def verify_against_master_equation(rho_SR, gL=0.2, gR=0.2):
    """Rebuild the dissipative superoperator two ways and assert equality:
      (i)  directly from the local + non-local dissipators (as in Collision_Model.py),
      (ii) from the canonical form using the analytic 4x4 Kossakowski matrix.
    """
    rho = _to_ndarray(rho_SR)
    p_s = np.trace(rho @ _kron(_n, _I)).real
    p_r = np.trace(rho @ _kron(_I, _n)).real
    J_mm = np.trace(rho @ _kron(_sm, _sm)); J_mp = np.trace(rho @ _kron(_sm, _sp))
    J_pm = np.trace(rho @ _kron(_sp, _sm)); J_pp = np.trace(rho @ _kron(_sp, _sp))

    # N = 2: the two end sites ARE the two chain qubits.
    sp1, sm1 = _kron(_sp, _I), _kron(_sm, _I)
    spN, smN = _kron(_I, _sp), _kron(_I, _sm)

    def diss(A, r):
        return A @ r @ A.conj().T - 0.5 * (A.conj().T @ A @ r + r @ A.conj().T @ A)

    def nldiss(A, B, r):
        return A @ r @ B - 0.5 * (B @ A @ r + r @ B @ A)

    GLp, GLm = gL ** 2 * p_s, gL ** 2 * (1 - p_s)
    GRp, GRm = gR ** 2 * p_r, gR ** 2 * (1 - p_r)

    def direct(r):
        out = GLp * diss(sp1, r) + GLm * diss(sm1, r) \
            + GRp * diss(spN, r) + GRm * diss(smN, r)
        out += gL * gR * J_mm * (nldiss(sp1, spN, r) + nldiss(spN, sp1, r))
        out += gL * gR * J_mp * (nldiss(sp1, smN, r) + nldiss(smN, sp1, r))
        out += gL * gR * J_pm * (nldiss(sm1, spN, r) + nldiss(spN, sm1, r))
        out += gL * gR * J_pp * (nldiss(sm1, smN, r) + nldiss(smN, sm1, r))
        return out

    D, _ = kossakowski(rho, gL, gR)
    ops = [sp1, sm1, spN, smN]

    def canonical(r):
        out = np.zeros((4, 4), dtype=complex)
        for a in range(4):
            for b in range(4):
                A, Bd = ops[a], ops[b].conj().T
                out += D[a, b] * (A @ r @ Bd - 0.5 * (Bd @ A @ r + r @ Bd @ A))
        return out

    return np.allclose(_superop(direct, 4), _superop(canonical, 4), atol=1e-10)


# ==========================================================================
# 5.  Reporting helpers
# ==========================================================================
def report_state(name, rho, gL=0.2, gR=0.2, do_class=True):
    D, blk = kossakowski(rho, gL, gR)
    row = {
        "name": name,
        "class": classify(rho) if do_class else "",
        "negativity": negativity(rho),
        "discord": discord(rho, n_grid=24) if do_class else np.nan,
        "||D_LR||": float(np.linalg.norm(blk["D_LR"])),
        "min_eig_D": _min_eig(D),
        "GKSL": gksl_ok(D),
        "strict_a": strict_unravellable(D),
        "frac_L": monitorable_fraction(D, "L"),
        "frac_R": monitorable_fraction(D, "R"),
    }
    return row


def _print_table(rows):
    cols = ["name", "class", "negativity", "discord", "||D_LR||",
            "min_eig_D", "GKSL", "strict_a", "frac_L", "frac_R"]
    hdr = f"{'name':>10} {'class':>10} {'neg':>7} {'disc':>7} {'||D_LR||':>9} " \
          f"{'minE(D)':>9} {'GKSL':>5} {'(a)':>5} {'frac_L':>7} {'frac_R':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:>10} {r['class']:>10} {r['negativity']:7.3f} "
              f"{r['discord']:7.3f} {r['||D_LR||']:9.4f} {r['min_eig_D']:9.4f} "
              f"{str(r['GKSL']):>5} {str(r['strict_a']):>5} "
              f"{r['frac_L']:7.3f} {r['frac_R']:7.3f}")


# ==========================================================================
# 6.  Main: self-test, named reps, sweeps, plots
# ==========================================================================
if __name__ == "__main__":
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unravelling_out")
    os.makedirs(OUT, exist_ok=True)
    gL = gR = 0.2

    # ---- 6.0  self-tests --------------------------------------------------
    print("=== self-tests ===")
    ok_all = True
    for m in ["product", "classical", "discord", "entangled"]:
        rho = random_two_qubit_state(m)
        ok = verify_against_master_equation(rho, gL, gR)
        ok_all &= ok
        D, _ = kossakowski(rho, gL, gR)
        fL = monitorable_fraction(D, "L")
        fR = monitorable_fraction(D, "R")
        fs = _fraction_schur(D, "L")
        agree = (fs is None) or abs(fL - fs) < 1e-6
        sym = abs(fL - fR) < 1e-6                    # frac_L == frac_R identity
        ok_all &= agree and sym
        print(f"  {m:>10}: D matches master eq = {ok} ; "
              f"frac_L={fL:.4f} frac_R={fR:.4f} (L==R:{sym}) "
              f"schur={'n/a' if fs is None else f'{fs:.4f}'} agree={agree}")
    print(f"  ALL SELF-TESTS PASS = {ok_all}\n")

    # ---- 6.1  named representatives ---------------------------------------
    print("=== named representatives (gL=gR=0.2) ===")
    rows = [report_state(m, random_two_qubit_state(m), gL, gR)
            for m in ["product", "classical", "discord", "entangled"]]
    _print_table(rows)
    with open(os.path.join(OUT, "named_reps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print()

    # ---- 6.2  Bell-diagonal (c1,c2) landscape of frac_L -------------------
    ng = 81
    cs = np.linspace(-1, 1, ng)
    frac_grid = np.full((ng, ng), np.nan)
    neg_grid = np.full((ng, ng), np.nan)
    for i, c2 in enumerate(cs):
        for j, c1 in enumerate(cs):
            rho = bell_diagonal(c1, c2, 0.0)
            if np.linalg.eigvalsh(rho).min() < -1e-12:   # outside tetrahedron slice
                continue
            D, _ = kossakowski(rho, gL, gR)
            frac_grid[i, j] = monitorable_fraction(D, "L")
            neg_grid[i, j] = negativity(rho)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(frac_grid, origin="lower", extent=[-1, 1, -1, 1],
                   vmin=0, vmax=1, cmap="viridis", aspect="equal")
    cs_sep = ax.contour(cs, cs, np.nan_to_num(neg_grid, nan=-1),
                        levels=[1e-6], colors="white", linewidths=1.5)
    ax.plot([-1, 1], [0, 0], "w:", lw=1)      # zero-discord axes (c1=0 or c2=0)
    ax.plot([0, 0], [-1, 1], "w:", lw=1)
    ax.set_xlabel("c1  (<XX>)"); ax.set_ylabel("c2  (<YY>)")
    ax.set_title("Max monitorable fraction of LEFT bath\n"
                 "Bell-diagonal ancilla, c3 = 0")
    fig.colorbar(im, ax=ax, label="frac_L")
    ax.text(0.02, 0.02, "white line: separability boundary (negativity=0)\n"
                         "dotted: zero-discord axes",
            transform=ax.transAxes, fontsize=7, color="w", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fraction_landscape.png"), dpi=130)
    plt.close(fig)

    # ---- 6.3  c3-invariance: frac independent of correlation class --------
    # For Bell-diagonal states D depends only on (c1,c2); c3 moves the class.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    demo_lines = [(0.5, 0.0), (0.6, -0.4), (0.3, 0.3)]
    cmap = {"classical": "tab:green", "discord": "tab:orange",
            "entangled": "tab:red", "product": "tab:gray"}
    for (c1, c2) in demo_lines:
        c3s, fracs, klass = [], [], []
        for c3 in np.linspace(-1, 1, 81):
            rho = bell_diagonal(c1, c2, c3)
            if np.linalg.eigvalsh(rho).min() < -1e-12:
                continue
            D, _ = kossakowski(rho, gL, gR)
            c3s.append(c3)
            fracs.append(monitorable_fraction(D, "L"))
            klass.append(classify(rho, disc_grid=10))
        c3s, fracs = np.array(c3s), np.array(fracs)
        ax.plot(c3s, fracs, "-", lw=1, color="k", alpha=0.4)
        for k in set(klass):
            mask = np.array([kk == k for kk in klass])
            ax.scatter(c3s[mask], fracs[mask], s=16, color=cmap[k],
                       label=k, zorder=3)
        ax.annotate(f"(c1,c2)=({c1},{c2})", (c3s[-1], fracs[-1]),
                    fontsize=7, va="center")
    # de-duplicate legend
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), fontsize=8, title="class")
    ax.set_xlabel("c3  (<ZZ>)  -- moves the correlation class")
    ax.set_ylabel("frac_L")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Monitorable fraction is FLAT in c3:\n"
                 "it tracks transverse (c1,c2) correlation, not the class")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_invariance.png"), dpi=130)
    plt.close(fig)

    # ---- 6.4  scatter: frac_L vs negativity / discord over random states --
    rng = np.random.default_rng(0)
    N_rand = 120
    negs, discs, fr, klass = [], [], [], []
    for k in range(N_rand):
        rho = random_two_qubit_state(seed=int(rng.integers(1 << 31)))
        D, _ = kossakowski(rho, gL, gR)
        if not gksl_ok(D):
            continue
        ng_i = negativity(rho)
        dc_i = discord(rho, n_grid=10)          # compute once, reuse for class
        negs.append(ng_i); discs.append(dc_i)
        fr.append(monitorable_fraction(D, "L"))
        if ng_i > 1e-7:
            klass.append("entangled")
        elif dc_i > 1e-4:
            klass.append("discord")
        else:
            rA = np.einsum("abcb->ac", _to_ndarray(rho).reshape(2, 2, 2, 2))
            rB = np.einsum("abac->bc", _to_ndarray(rho).reshape(2, 2, 2, 2))
            klass.append("classical" if np.linalg.norm(_to_ndarray(rho) - _kron(rA, rB)) > 1e-7
                         else "product")
    negs, discs, fr = map(np.array, (negs, discs, fr))
    colors = [cmap[k] for k in klass]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    axes[0].scatter(negs, fr, s=14, c=colors, alpha=0.8)
    axes[0].set_xlabel("negativity (entanglement)"); axes[0].set_ylabel("frac_L")
    axes[0].set_title("frac_L vs negativity")
    axes[1].scatter(discs, fr, s=14, c=colors, alpha=0.8)
    axes[1].set_xlabel("discord"); axes[1].set_ylabel("frac_L")
    axes[1].set_title("frac_L vs discord")
    for a in axes:
        a.set_ylim(-0.05, 1.05)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap[k], label=k)
               for k in ["product", "classical", "discord", "entangled"]]
    axes[1].legend(handles=handles, fontsize=8, title="class")
    fig.suptitle("No functional link: a given class spans a range of fractions "
                 "(fraction is set by the Kossakowski data, not the class label)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "frac_vs_correlation.png"), dpi=130)
    plt.close(fig)

    # ---- 6.5  THERMAL marginals: t3-invariance persists; frac is symmetric --
    print("=== thermal-marginal correlated ancillas ===")
    print("D depends only on (p_s, p_r, t1, t2); t3 (the z-z / class knob) drops out.")
    print("frac_L == frac_R = 1 - sigma_max(M)^2 exactly, even for p_s != p_r.\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # (left) thermal marginals, fixed transverse corr, sweep t3 across classes
    p_s0, p_r0, t1_0, t2_0 = 0.45, 0.35, 0.35, 0.0
    t3s, fLs, fRs, kl = [], [], [], []
    for t3 in np.linspace(-1, 1, 121):
        rho = thermal_correlated_state(p_s0, p_r0, t1_0, t2_0, t3)
        if not is_physical(rho):
            continue
        D, _ = kossakowski(rho, gL, gR)
        t3s.append(t3); fLs.append(monitorable_fraction(D, "L"))
        fRs.append(monitorable_fraction(D, "R")); kl.append(classify(rho, disc_grid=10))
    t3s = np.array(t3s)
    axes[0].plot(t3s, fLs, "-", lw=6, color="tab:blue", alpha=0.35, label="frac_L")
    axes[0].plot(t3s, fRs, "--", lw=1.5, color="tab:purple", label="frac_R (= frac_L)")
    for k in set(kl):
        m = np.array([kk == k for kk in kl])
        axes[0].scatter(t3s[m], np.full(m.sum(), -0.03), s=12, color=cmap[k], label=k)
    axes[0].set_xlabel("t3  (<ZZ>, correlation-class knob)"); axes[0].set_ylabel("fraction")
    axes[0].set_ylim(-0.08, 1.05)
    axes[0].set_title(f"Thermal marginals p_s={p_s0}, p_r={p_r0}\n"
                      f"transverse (t1,t2)=({t1_0},{t2_0}): FLAT in t3, frac_L=frac_R")
    h, l = axes[0].get_legend_handles_labels(); seen = dict(zip(l, h))
    axes[0].legend(seen.values(), seen.keys(), fontsize=7, ncol=2)

    # (right) shared fraction vs temperature bias at fixed transverse coupling;
    # colder marginals => same correlation is "larger" vs the rates => more shared
    p_r1, t1_1 = 0.30, 0.30
    ps_phys, frac_t, ps_unphys = [], [], []
    for p_s in np.linspace(0.02, 0.98, 97):
        rho = thermal_correlated_state(p_s, p_r1, t1_1, 0.0, 0.0)
        if not is_physical(rho):
            ps_unphys.append(p_s); continue
        D, _ = kossakowski(rho, gL, gR)
        ps_phys.append(p_s); frac_t.append(monitorable_fraction(D, "L"))
    axes[1].plot(ps_phys, frac_t, "-", color="tab:blue", lw=2,
                 label="shared fraction (=frac_L=frac_R)")
    if ps_unphys:
        axes[1].axvspan(min(ps_unphys), max(ps_unphys), color="red", alpha=0.08)
        axes[1].text(np.mean(ps_unphys), 0.5, "unphysical\n(state not PSD)",
                     ha="center", fontsize=7, color="firebrick")
    axes[1].axvline(p_r1, color="gray", ls=":", lw=1)
    axes[1].text(p_r1, 1.0, f" p_r={p_r1}", fontsize=7, color="gray")
    axes[1].set_xlabel("p_s  (left-ancilla excited population; small = cold)")
    axes[1].set_ylabel("monitorable fraction"); axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(f"Temperature sets the shared fraction\n(t1={t1_1}): colder => more shared")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "thermal_marginals.png"), dpi=130)
    plt.close(fig)

    # ---- 6.6  one bath vs two: canonical channel structure ----------------
    print("=== bath structure: canonical jump channels ===")
    print(f"{'state':>22} {'shared':>7}  channels (rate | pL,pR | delocalized?)")
    print("-" * 78)
    demo = [
        ("product", random_two_qubit_state("product")),
        ("classical (1,0,0)", bell_diagonal(1, 0, 0)),
        ("entangled Bell", random_two_qubit_state("entangled")),
        ("thermal corr (.3,.15)", thermal_correlated_state(0.30, 0.15, 0.5, -0.2, 0.2)),
    ]
    for nm, rho in demo:
        D, _ = kossakowski(rho, gL, gR)
        chans, shared = canonical_channels(D)
        desc = "  ".join(f"[{c['rate']:.3f}|{c['pL']:.2f},{c['pR']:.2f}|"
                         f"{'YES' if c['delocalized'] else 'no'}]" for c in chans)
        print(f"{nm:>22} {shared:7.3f}  {desc}")
    print("\n  pL,pR = left/right participation of each canonical jump operator.")
    print("  delocalized channels (both pL,pR > 0) = shared-bath (nonlocal) jumps.\n")

    print(f"figures + CSV written to: {OUT}")
