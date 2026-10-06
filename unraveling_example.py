"""
When does an unraveling exist?  A minimal, self-contained numerical experiment.
================================================================================

Setting: two qubits, each nominally coupled to "its own" bath.  The dissipator is

    D(rho) = sum_{jk} c_jk ( F_j rho F_k^dag - 1/2 {F_k^dag F_j, rho} ),
    F_1 = sigma_- (x) I ,   F_2 = I (x) sigma_- ,

with Kossakowski matrix

    c = gamma * [[1, m], [m, 1]],     m = channel overlap, real, |m| <= 1.

m = 0  -> two genuinely independent reservoirs, one per qubit.
m != 0 -> the two channels are NOT independent reservoirs.  See the section
"IS m != 0 PHYSICAL?" below: m is the overlap of the two couplings into a
shared vacuum (Dicke superradiance at m = 1), NOT a correlation of bath states.
The labels A and B then name system operators, not reservoirs -- which is the
whole reason a "detector in bath A" turns out not to exist.

Three questions, kept strictly separate:

  Q1  Is this a legitimate Lindblad generator?                <-> c >= 0
  Q2  Does SOME unraveling exist?                             <-> c >= 0  (same test)
  Q3  Does the PER-BATH unraveling exist -- detectors on       <-> c - P_blocks(c) >= 0
      bath A and bath B, each reporting only its own clicks?       i.e. c block diagonal

Q3 is the one with teeth, and it fails for any m != 0.  Everything below is a
numerical demonstration of that, ending with a concrete task that becomes
impossible: balancing the per-bath energy ledger.

The bath partition is set by BATHS (a list of channel-index groups), so the same
code handles more baths or several channels per bath without modification.

Requires numpy + scipy.
"""

import numpy as np
from scipy.linalg import expm, eigvalsh, eigh, block_diag
from scipy.optimize import brentq

rng = np.random.default_rng(20260728)

# ----------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
sm = np.array([[0, 1], [0, 0]], dtype=complex)      # |g><e|,  index 0 = g, 1 = e
sp = sm.conj().T
n_op = sp @ sm                                       # excitation number, one qubit

kron = np.kron
F1 = kron(sm, I2)          # "bath A" channel
F2 = kron(I2, sm)          # "bath B" channel
F  = [F1, F2]
BATHS = [[0], [1]]         # channel indices belonging to each bath
DIM = 4
N_TOT = kron(n_op, I2) + kron(I2, n_op)              # total excitation number
H = np.zeros((DIM, DIM), dtype=complex)              # no Hamiltonian: isolate dissipation


def kossakowski(gamma=1.0, m=0.0):
    return gamma * np.array([[1.0, m], [m, 1.0]], dtype=complex)


def herm(M):
    M = np.asarray(M)
    return (M + M.conj().T) / 2


def is_psd(M, tol=1e-10):
    return bool(np.all(eigvalsh(herm(M)) >= -tol))


def block_part(c, baths=BATHS):
    """Project c onto the block-diagonal part defined by the bath partition:
    what a set of per-bath detectors is able to subtract."""
    blocks = [np.asarray(c)[np.ix_(b, b)] for b in baths]
    perm = [i for b in baths for i in b]
    out = np.zeros_like(np.asarray(c))
    out[np.ix_(perm, perm)] = block_diag(*blocks)
    return out


def liouvillian(c, H=H, F=F):
    """Superoperator matrix acting on vec(rho) with C-order (row-major) flattening.
    Uses  vec(A rho B) = kron(A, B.T) vec(rho)."""
    d = H.shape[0]
    Id = np.eye(d, dtype=complex)
    L = -1j * (kron(H, Id) - kron(Id, H.T))
    for j in range(len(F)):
        for k in range(len(F)):
            if c[j, k] == 0:
                continue
            anti = F[k].conj().T @ F[j]
            L = L + c[j, k] * (kron(F[j], F[k].conj())
                               - 0.5 * kron(anti, Id)
                               - 0.5 * kron(Id, anti.T))
    return L


def evolve_exact(c, rho0, t):
    return (expm(liouvillian(c) * t) @ rho0.reshape(-1)).reshape(DIM, DIM)


def jump_ops_from_c(c, F=F):
    """Canonical (diagonalizing) jump operators: c = U d U^dag  ->  L_mu = sqrt(d_mu) sum_j U_jmu F_j.
    Discards non-positive eigenvalues, so it is only faithful when c >= 0."""
    d, U = eigh(herm(c))
    return [np.sqrt(d[mu]) * sum(U[j, mu] * F[j] for j in range(len(F)))
            for mu in range(len(d)) if d[mu] > 1e-12]


# ============================================================================
# Monte Carlo trajectories -- THE READABLE VERSION
# ============================================================================
# This is the reference implementation: one run of the experiment, one time step
# at a time, plain Python loops, nothing clever.  Read this one to understand
# what a jump unraveling IS.  The vectorised version further down does exactly
# the same thing for every trajectory at once, and is cross-checked against this
# one at the bottom of the file.
#
# The algorithm, in words:
#
#   repeat for each time step dt:
#       for each detector k, compute the probability p_k = dt <psi|L_k^dag L_k|psi>
#           that it clicks during this step
#       draw one uniform random number
#       IF it says a click happened:
#           draw again to decide WHICH detector clicked (weights p_k)
#           apply that jump operator:      |psi> -> L_k |psi>
#       ELSE (no click -- this is still an observation!):
#           apply the non-Hermitian no-jump evolution: |psi> -> (1 - i H_eff dt)|psi>
#       renormalise |psi>
#
# Averaging |psi><psi| over many runs reproduces the master equation -- PROVIDED
# the p_k really are probabilities.  That proviso is the entire subject of this
# script.

def one_trajectory(jumps, psi0, T, dt, rng=rng):
    """Simulate ONE run of the experiment.

    Returns (final state, clicks per detector, click record).
    The click record is the list of (time, detector) pairs -- i.e. literally
    what the experimenter would have written down."""
    n_steps = int(round(T / dt))
    n_det = len(jumps)

    Gamma = sum(Lk.conj().T @ Lk for Lk in jumps)   # total rate operator
    Heff = H - 0.5j * Gamma                          # no-jump generator

    psi = psi0.astype(complex).copy()
    counts = np.zeros(n_det, dtype=int)
    record = []

    for s in range(n_steps):

        # --- 1. click probability of each detector during this step ---------
        p = np.zeros(n_det)
        post_jump = []
        for k in range(n_det):
            phi = jumps[k] @ psi                     # unnormalised post-jump state
            post_jump.append(phi)
            p[k] = dt * float(np.real(np.vdot(phi, phi)))   # dt <psi|L^dag L|psi>
        p_total = p.sum()

        # --- 2. did ANY detector click? -------------------------------------
        if rng.random() < p_total:

            # --- 3a. yes: which one?  (inverse-CDF sampling, written out) ----
            r = rng.random() * p_total
            k = 0
            while r > p[k] and k < n_det - 1:
                r -= p[k]
                k += 1
            psi = post_jump[k]
            counts[k] += 1
            record.append((s * dt, k))

        else:
            # --- 3b. no click.  The state STILL changes: seeing nothing is
            #         information, and it is applied by the non-Hermitian H_eff.
            psi = psi - 1j * dt * (Heff @ psi)

        # --- 4. renormalise: both branches leave ||psi|| != 1 ----------------
        psi = psi / np.linalg.norm(psi)

    return psi, counts, record


def trajectories_explicit(jumps, psi0, T, dt, n_traj, rng=rng):
    """Run one_trajectory() n_traj times and average.  Same signature and same
    return values as trajectories(), just slow and obvious."""
    jumps = list(jumps)
    rho = np.zeros((DIM, DIM), dtype=complex)
    counts = np.zeros(len(jumps))
    n_end = 0.0

    for _ in range(n_traj):
        psi, cnt, _rec = one_trajectory(jumps, psi0, T, dt, rng)
        rho += np.outer(psi, psi.conj())             # accumulate |psi><psi|
        counts += cnt
        n_end += float(np.real(np.vdot(psi, N_TOT @ psi)))

    n0 = float(np.real(np.vdot(psi0, N_TOT @ psi0)))
    return rho / n_traj, counts / n_traj, n0 - n_end / n_traj


# ============================================================================
# Monte Carlo trajectories -- THE FAST VERSION (same algorithm, vectorised)
# ============================================================================
# Line-by-line correspondence with one_trajectory():
#     cand   <-> post_jump   (all detectors, all trajectories, at once)
#     w      <-> p
#     u      <-> the two rng.random() calls, fused into one (see comment below)
#     step   <-> (1 - i H_eff dt)
# Everything else is bookkeeping to keep n_traj trajectories in one array.

def trajectories(jumps, psi0, T, dt, n_traj):
    """Standard jump unraveling, vectorised over the whole ensemble.
    Returns (mean density matrix at T, mean click count per channel,
    mean excitations lost per trajectory)."""
    jumps = list(jumps)
    Gamma = sum(Lk.conj().T @ Lk for Lk in jumps)
    Heff = H - 0.5j * Gamma
    step = np.eye(DIM, dtype=complex) - 1j * Heff * dt      # first order in dt
    n_steps = int(round(T / dt))

    psi = np.tile(psi0.astype(complex), (n_traj, 1))        # (N, DIM)
    counts = np.zeros(len(jumps))

    for _s in range(n_steps):
        # candidate post-jump states and their weights, for every trajectory
        cand = np.stack([psi @ Lk.T for Lk in jumps])       # (nj, N, DIM)
        w = dt * np.sum(np.abs(cand) ** 2, axis=2)          # (nj, N)  jump probabilities
        u = rng.random(n_traj)
        jumped = u < w.sum(axis=0)                          # "did anything click?"
        # "which detector?"  one_trajectory() draws a SECOND random number here;
        # we reuse u instead.  Conditioned on u < p_total, u is uniform on
        # [0, p_total), which is exactly the variable inverse-CDF sampling needs.
        # argmax finds the first cumulative weight exceeding it.  (Trajectories
        # that did not jump get pick = 0, harmless: they are masked out below.)
        pick = np.argmax(np.cumsum(w, axis=0) > u[None, :], axis=0)

        new = psi @ step.T                                   # no-jump branch
        if jumped.any():
            idx = np.where(jumped)[0]
            new[idx] = cand[pick[idx], idx]
            np.add.at(counts, pick[idx], 1.0)
        psi = new
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)

    # NB order matters: psi.T @ psi.conj() gives sum_i |psi_i><psi_i|;
    # psi.conj().T @ psi would give its transpose (invisible for real rho, wrong otherwise).
    rho = (psi.T @ psi.conj()) / n_traj                      # sum_i |psi_i><psi_i| / N
    n0 = float(np.real(psi0.conj() @ (N_TOT @ psi0)))
    n_end = float(np.real(np.einsum("ni,ij,nj->", psi.conj(), N_TOT, psi)) / n_traj)
    return rho, counts / n_traj, n0 - n_end


def check_instrument(jumps, c, dt=1e-4, n_rho=5):
    """The checklist, applied directly.  Given jump Kraus operators M_k = sqrt(dt)*L_k
    and M_0 = 1 - i H_eff dt, verify:
      (1) each piece CP                   -- automatic for M rho M^dag, reported for honesty
      (2) completeness  sum_i M_i^dag M_i = I + O(dt^2)
      (3) sum_i M_i rho M_i^dag = rho + L(rho) dt + O(dt^2)   for the TARGET c
    Failure of (3) is what non-existence looks like in Kraus language."""
    Gamma = sum(Lk.conj().T @ Lk for Lk in jumps)
    Heff = H - 0.5j * Gamma
    Ms = [np.eye(DIM, dtype=complex) - 1j * Heff * dt] + [np.sqrt(dt) * Lk for Lk in jumps]

    completeness = np.max(np.abs(sum(M.conj().T @ M for M in Ms) - np.eye(DIM)))
    Lsup = liouvillian(c)

    worst = 0.0
    for _ in range(n_rho):
        A = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        lhs = sum(M @ rho @ M.conj().T for M in Ms)
        rhs = rho + (Lsup @ rho.reshape(-1)).reshape(DIM, DIM) * dt
        worst = max(worst, np.max(np.abs(lhs - rhs)))

    print(f"    (1) every piece CP (single-Kraus)            : True  [vacuous by construction]")
    print(f"    (2) ||sum M^dag M - I||        = {completeness:.2e}   (should be O(dt^2)={dt**2:.0e})")
    print(f"    (3) ||sum M rho M^dag - rho - L(rho)dt||/dt = {worst/dt:.2e}"
          f"   {'<- reproduces L' if worst/dt < 1e-3 else '<- FAILS: unravels a different L'}")


def tracedist(a, b):
    return 0.5 * np.sum(np.abs(eigvalsh(herm(a - b))))


# ----------------------------------------------------------------------------
# how much per-bath monitoring survives?  (the graded version)
# ----------------------------------------------------------------------------

def max_efficiency(c, monitored=(0, 1)):
    """Largest eta with  c - eta*S >= 0,  S = the monitored diagonal blocks.

    Solved exactly as a generalised eigenvalue problem when c is nonsingular:
    c - eta S >= 0  <=>  eta * lambda_max(S, c) <= 1.  Falls back to a root find
    on lambda_min(c - eta S) when c is singular (m = 1)."""
    S = np.diag([np.real(c[i, i]) if i in monitored else 0.0 for i in range(c.shape[0])])
    if is_psd(c - 1.0 * S):
        return 1.0
    try:
        lam = eigh(herm(S), herm(c), eigvals_only=True)      # generalised, S x = lam c x
        lam_max = lam.max()
        if lam_max > 0:
            return float(min(1.0, 1.0 / lam_max))
    except np.linalg.LinAlgError:
        pass
    f = lambda eta: eigvalsh(herm(c - eta * S)).min()
    return float(brentq(f, 0.0, 1.0, xtol=1e-12)) if f(0.0) > 0 else 0.0



# ----------------------------------------------------------------------------
# IS m != 0 PHYSICAL?  Microscopic collision model with VACUUM ancillas.
# ----------------------------------------------------------------------------
# Worry: c is off-diagonal but the only jumps are emissions (sigma_-), i.e. a
# zero-temperature bath.  Correlated ancilla STATES need mixed marginals, so
# vacuum ancillas cannot be correlated -- does that make m != 0 unphysical?
#
# No.  Off-diagonal c does not require a correlated bath state.  It requires
# overlapping COUPLINGS.  Let each system qubit couple to its own combination
# of a shared set of vacuum modes,
#
#     K = sqrt(gamma) sum_j ( sigma_-^j (x) B_j^dag + h.c. ),   B_j = sum_mu G_j,mu b_mu ,
#
# with all modes in vacuum.  Then <vac| B_j B_k^dag |vac> = (G G^dag)_jk and
#
#     c = gamma * G G^dag                       <- a GRAM MATRIX
#
# which is PSD automatically (this is the c = W^dag W factorisation, realised
# microscopically) and off-diagonal exactly when the coupling vectors overlap.
# m = <g_1, g_2> is the coupling overlap, not a bath correlation.  m = 1 is two
# emitters within a wavelength radiating into one common vacuum: ordinary Dicke
# superradiance, entirely standard and entirely at T = 0.
#
# Moral: when m != 0 there are not two baths.  There is one bath, and A/B label
# SYSTEM operators, not reservoirs.  That is precisely why per-bath monitoring
# fails -- there is no "bath A" to put a detector in.

def microscopic_check(m, gamma=1.0, dt=1e-6):
    """Build the collision unitary explicitly with vacuum ancillas, trace them
    out, and confirm the resulting generator is the c = gamma G G^dag one."""
    DA = 4                                        # two ancilla modes, hard-core truncated
    b = [kron(sm, I2), kron(I2, sm)]              # mode annihilation operators
    G = np.array([[1.0, 0.0], [m, np.sqrt(max(0.0, 1 - m ** 2))]], dtype=complex)
    Bd = [sum(np.conj(G[j, mu]) * b[mu].conj().T for mu in range(2)) for j in range(2)]
    K = np.sqrt(gamma) * sum(kron(F[j], Bd[j]) + kron(F[j].conj().T, Bd[j].conj().T)
                             for j in range(2))
    U = expm(-1j * np.sqrt(dt) * K)

    vac = np.zeros(DA, dtype=complex); vac[0] = 1.0
    A = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM))
    rho = A @ A.conj().T; rho /= np.trace(rho)

    tot = U @ kron(rho, np.outer(vac, vac.conj())) @ U.conj().T
    rho2 = np.einsum("iaja->ij", tot.reshape(DIM, DA, DIM, DA))

    c = gamma * (G @ G.conj().T)
    drho_th = (liouvillian(c) @ rho.reshape(-1)).reshape(DIM, DIM)
    return c, float(np.max(np.abs((rho2 - rho) / dt - drho_th)))


# ----------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------

GAMMA, T, DT, NTRAJ = 1.0, 1.5, 2.5e-3, 6000

# bright state (|eg> + |ge>)/sqrt(2): both channels click, and for m > 0 its decay
# is collectively enhanced to gamma(1+m) -- a visible signature of correlation.
psi0 = np.zeros(DIM, dtype=complex); psi0[1] = psi0[2] = 1 / np.sqrt(2)
rho0 = np.outer(psi0, psi0.conj())


def banner(s):
    print("\n" + "=" * 76 + "\n" + s + "\n" + "=" * 76)


def report(m):
    c = kossakowski(GAMMA, m)
    banner(f"m = {m}      c =\n{np.real(c)}")

    # ---- Q1/Q2: does any unraveling exist? -------------------------------
    print(f"  eig(c)                     = {np.round(eigvalsh(herm(c)), 6)}")
    print(f"  Q2  some unraveling exists : {is_psd(c)}")

    # ---- Q3: does the per-bath unraveling exist? -------------------------
    c_rest = c - block_part(c)                # subtract each bath's own block
    print(f"  residual after per-bath detectors, eig = {np.round(eigvalsh(herm(c_rest)), 6)}")
    ok_local = is_psd(c_rest)
    print(f"  Q3  per-bath unraveling exists : {ok_local}")
    if not ok_local:
        bad = eigvalsh(herm(c_rest)).min()
        print(f"      -> residual channel with rate {bad:+.4f}: jump probability "
              f"{bad:+.3e}*dt < 0.  Not a probability, so no such instrument exists.")
        print("      -> (a PSD matrix with zero diagonal must vanish entirely;")
        print("          hence per-bath monitoring works iff c is block diagonal.)")

    # ---- numerically confirm with trajectories ---------------------------
    rho_exact = evolve_exact(c, rho0, T)

    # (a) the canonical / collective unraveling: always valid when c >= 0
    rho_c, cnt_c, _ = trajectories(jump_ops_from_c(c), psi0, T, DT, NTRAJ)
    print(f"\n  canonical unraveling   : trace distance to exact = {tracedist(rho_c, rho_exact):.4f}")

    # (b) forced per-bath detectors: L_A = sqrt(c_AA) F1, L_B = sqrt(c_BB) F2
    local = [np.sqrt(np.real(c[j, j])) * F[j] for j in range(len(F))]
    rho_l, cnt_l, _ = trajectories(local, psi0, T, DT, NTRAJ)
    print(f"  forced per-bath ensemble: trace distance to exact = {tracedist(rho_l, rho_exact):.4f}")

    # statistical or systematic?  quadruple the sample and look again.
    rho_c4, _, _ = trajectories(jump_ops_from_c(c), psi0, T, DT, 4 * NTRAJ)
    rho_l4, _, _ = trajectories(local, psi0, T, DT, 4 * NTRAJ)
    d_c4, d_l4 = tracedist(rho_c4, rho_exact), tracedist(rho_l4, rho_exact)
    verdict = lambda d: ("consistent with 0 -> purely statistical" if d < 0.02
                         else "stays finite -> SYSTEMATIC, not sampling noise")
    print(f"    with 4x the trajectories: canonical {d_c4:.4f}  ({verdict(d_c4)})")
    print(f"                             per-bath  {d_l4:.4f}  ({verdict(d_l4)})")

    # what the forced-local ensemble IS a valid unraveling of:
    if abs(m) > 1e-12:
        rho_diag = evolve_exact(block_part(c), rho0, T)
        print(f"    distance to the *block-diagonal-c* master equation = "
              f"{tracedist(rho_l, rho_diag):.4f}  <- it faithfully unravels the WRONG generator")

    # ---- the same verdict, algebraically, via the Kraus checklist ---------
    print("\n  Kraus checklist, canonical (delocalised) jump operators:")
    check_instrument(jump_ops_from_c(c), c)
    print("  Kraus checklist, forced per-bath jump operators:")
    check_instrument(local, c)

    # ---- the consequence: per-bath energy ledger -------------------------
    n_lost_exact = float(np.real(np.trace(N_TOT @ (rho0 - rho_exact))))
    ok = lambda x: 'OK' if abs(x - n_lost_exact) < 0.05 else 'MISMATCH'
    print(f"\n  CONSEQUENCE  (per-bath energy ledger)")
    print(f"    excitations actually lost by the system (exact ME) : {n_lost_exact:.4f}")
    print(f"    clicks summed over canonical detectors             : {cnt_c.sum():.4f}   {ok(cnt_c.sum())}")
    print(f"      per canonical channel: {np.round(cnt_c, 4)}"
          f"   (these are the +/- collective modes, not baths)")
    print(f"    clicks summed over per-bath detectors  (N_A + N_B) : {cnt_l.sum():.4f}   {ok(cnt_l.sum())}")
    print(f"      N_A = {cnt_l[0]:.4f},  N_B = {cnt_l[1]:.4f}")
    if ok_local:
        print("    -> the ledger balances: 'a quantum went into bath A' is a bona fide"
              "\n       stochastic variable, and heat currents per bath are well defined.")
    else:
        print("    -> the ledger does NOT balance.  The per-bath counts are records of a"
              "\n       different dynamics, so N_A is not a property of the true evolution."
              "\n       Per-bath counting statistics / fluctuation theorems are therefore"
              "\n       undefined, even though TOTAL emission is perfectly well defined via"
              "\n       the canonical (delocalised) detectors -- which carry no bath label.")
    return c


if __name__ == "__main__":
    report(0.0)      # independent baths  -> per-bath unraveling exists
    report(0.8)      # correlated baths   -> it does not

    banner("What a single run of the experiment actually looks like")
    # independent baths (m = 0, so per-bath detectors are legitimate) and both
    # qubits excited, so a run can contain two clicks in either order.
    psi_ee = np.zeros(DIM, dtype=complex); psi_ee[3] = 1.0
    det = [np.sqrt(GAMMA) * F[0], np.sqrt(GAMMA) * F[1]]
    for i in range(6):
        _psi, cnt, rec = one_trajectory(det, psi_ee, T, DT)
        clicks = "  ".join(f"t={t:.3f}: bath {'AB'[k]}" for t, k in rec) or "(no clicks)"
        print(f"  run {i+1}:  {clicks}")
    print("\n  Each run is a different history -- a different list of what the detectors")
    print("  recorded and when.  Averaging |psi><psi| over runs reproduces the master")
    print("  equation, PROVIDED the click probabilities really are probabilities.")

    banner("Cross-check: explicit loop version vs vectorised version")
    for m in [0.0, 0.8]:
        c = kossakowski(GAMMA, m)
        J, ex = jump_ops_from_c(c), evolve_exact(c, rho0, T)
        r_slow, k_slow, _ = trajectories_explicit(J, psi0, T, DT, 1200)
        r_fast, k_fast, _ = trajectories(J, psi0, T, DT, 1200)
        print(f"  m={m}:  explicit vs exact = {tracedist(r_slow, ex):.4f},"
              f"   vectorised vs exact = {tracedist(r_fast, ex):.4f},"
              f"   explicit vs vectorised = {tracedist(r_slow, r_fast):.4f}")
        print(f"          mean clicks per detector: explicit {np.round(k_slow,4)}, "
              f"vectorised {np.round(k_fast,4)}")
    print("\n  All differences are at the 1/sqrt(N) sampling level -- the two")
    print("  implementations are the same algorithm.")

    banner("Is m != 0 physical?  Exact collision model, ancillas in the VACUUM")
    print("  emission-only coupling, all ancilla modes in |vac>, couplings overlapping by m")
    print(f"\n{'m':>5}   {'c = gamma G G^dag':>28}   {'max|drho_exact - drho_Lindblad|':>32}")
    for m in [0.0, 0.3, 0.8, 1.0]:
        c, err = microscopic_check(m)
        print(f"{m:>5.1f}   {str(np.real(c).round(3).tolist()):>28}   {err:>32.2e}")
    print("\n  -> off-diagonal c arises from OVERLAPPING COUPLINGS into a shared vacuum,")
    print("     not from correlated bath states.  m = 1 is Dicke superradiance at T = 0.")
    print("     So when m != 0 there is ONE bath; A and B label system operators, not")
    print("     reservoirs -- which is exactly why there is no 'bath A' to monitor.")

    banner("How much per-bath monitoring is still allowed?  (partial efficiency)")
    print(f"{'m':>6} {'eta_max (bath A only)':>24} {'1-m^2':>10} "
          f"{'eta_max (both baths)':>24} {'1-m':>8}")
    for m in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        c = kossakowski(GAMMA, m)
        print(f"{m:>6.2f} {max_efficiency(c, (0,)):>24.6f} {1-m**2:>10.6f} "
              f"{max_efficiency(c, (0, 1)):>24.6f} {1-m:>8.6f}")
    print("\nBoth closed forms are reproduced exactly.  Monitoring both baths at once is")
    print("strictly more restrictive than monitoring one -- you are demanding which-bath")
    print("resolution on both sides simultaneously.")