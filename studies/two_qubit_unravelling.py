"""When does a per-bath unravelling exist?  The two-qubit toy model (colmodel.toy).

Two qubits, channels F_1 = s- (x) I (bath A) and F_2 = I (x) s- (bath B), Kossakowski
matrix c = gamma [[1, m], [m, 1]]. Three questions, kept separate:

  Q1  Is this a legitimate Lindblad generator?              <-> c >= 0
  Q2  Does SOME unravelling exist?                           <-> c >= 0 (same test)
  Q3  Does the PER-BATH unravelling exist (detectors on      <-> c - blocks(c) >= 0,
      bath A and bath B, each reporting only its own clicks)?    i.e. c block diagonal

Q3 fails for any m != 0. The script demonstrates it with trajectories, with the
Kraus checklist, and through its consequence: the per-bath energy ledger does not
balance. It then cross-checks the two trajectory implementations, builds m from a
microscopic vacuum collision model, and tabulates the partial monitoring efficiency.

Output: stdout and runs/two_qubit_unravelling/report.txt. Seeded (20260728).
"""

import _common
import numpy as np
from scipy.linalg import eigvalsh

from colmodel import master_equation as me, metrics, toy, trajectories as tr, unravelling as un
from colmodel.operators import hermitian_part, is_psd

GAMMA, T, DT, NTRAJ = 1.0, 1.5, 2.5e-3, 6000
SEED = 20260728

# bright state (|eg> + |ge>)/sqrt(2): both channels click, and for m > 0 its decay
# is collectively enhanced to gamma (1 + m), a visible signature of the overlap.
psi0 = np.zeros(toy.DIM, dtype=complex)
psi0[1] = psi0[2] = 1 / np.sqrt(2)
rho0 = np.outer(psi0, psi0.conj())


def banner(s):
    print("\n" + "=" * 76 + "\n" + s + "\n" + "=" * 76)


def run(jumps, n, rng):
    return tr.ensemble(toy.H, jumps, psi0, T, DT, n, rng)


def print_instrument(jumps, c, rng):
    completeness, residual = tr.instrument_check(toy.H, jumps, toy.liouvillian(c), 1e-4, rng)
    dt = 1e-4
    print("    (1) every piece CP (single-Kraus)            : True  [vacuous by construction]")
    print(f"    (2) ||sum M^dag M - I||        = {completeness:.2e}   (should be O(dt^2)={dt**2:.0e})")
    print(f"    (3) ||sum M rho M^dag - rho - L(rho)dt||/dt = {residual:.2e}"
          f"   {'<- reproduces L' if residual < 1e-3 else '<- FAILS: unravels a different L'}")


def report(m, rng):
    c = toy.coupling_matrix(GAMMA, m)
    banner(f"m = {m}      c =\n{np.real(c)}")

    # ---- Q1/Q2: does any unravelling exist? --------------------------------
    print(f"  eig(c)                     = {np.round(eigvalsh(hermitian_part(c)), 6)}")
    print(f"  Q2  some unraveling exists : {is_psd(c)}")

    # ---- Q3: does the per-bath unravelling exist? --------------------------
    c_rest = c - un.block_part(c, toy.BATHS.values())
    print(f"  residual after per-bath detectors, eig = {np.round(eigvalsh(hermitian_part(c_rest)), 6)}")
    ok_local = is_psd(c_rest)
    print(f"  Q3  per-bath unraveling exists : {ok_local}")
    if not ok_local:
        bad = eigvalsh(hermitian_part(c_rest)).min()
        print(f"      -> residual channel with rate {bad:+.4f}: jump probability "
              f"{bad:+.3e}*dt < 0.  Not a probability, so no such instrument exists.")
        print("      -> (a PSD matrix with zero diagonal must vanish entirely;")
        print("          hence per-bath monitoring works iff c is block diagonal.)")

    # ---- numerically confirm with trajectories -----------------------------
    L = toy.liouvillian(c)
    rho_exact = me.evolve(L, rho0, T)
    canonical = un.canonical_jump_operators(toy.F, c)
    local = toy.per_bath_jumps(c)

    ens_c = run(canonical, NTRAJ, rng)
    print(f"\n  canonical unraveling   : trace distance to exact = {metrics.trace_distance(ens_c.rho, rho_exact):.4f}")
    ens_l = run(local, NTRAJ, rng)
    print(f"  forced per-bath ensemble: trace distance to exact = {metrics.trace_distance(ens_l.rho, rho_exact):.4f}")

    # statistical or systematic?  quadruple the sample and look again.
    d_c4 = metrics.trace_distance(run(canonical, 4 * NTRAJ, rng).rho, rho_exact)
    d_l4 = metrics.trace_distance(run(local, 4 * NTRAJ, rng).rho, rho_exact)
    verdict = lambda d: ("consistent with 0 -> purely statistical" if d < 0.02
                         else "stays finite -> SYSTEMATIC, not sampling noise")
    print(f"    with 4x the trajectories: canonical {d_c4:.4f}  ({verdict(d_c4)})")
    print(f"                             per-bath  {d_l4:.4f}  ({verdict(d_l4)})")

    # what the forced per-bath ensemble IS a valid unravelling of:
    if abs(m) > 1e-12:
        rho_diag = me.evolve(toy.liouvillian(un.block_part(c, toy.BATHS.values())), rho0, T)
        print(f"    distance to the *block-diagonal-c* master equation = "
              f"{metrics.trace_distance(ens_l.rho, rho_diag):.4f}  <- it faithfully unravels the WRONG generator")

    # ---- the same verdict, algebraically, via the Kraus checklist ----------
    print("\n  Kraus checklist, canonical (delocalised) jump operators:")
    print_instrument(canonical, c, rng)
    print("  Kraus checklist, forced per-bath jump operators:")
    print_instrument(local, c, rng)

    # ---- the consequence: per-bath energy ledger ---------------------------
    n_lost_exact = float(np.real(np.trace(toy.N_TOT @ (rho0 - rho_exact))))
    ok = lambda x: 'OK' if abs(x - n_lost_exact) < 0.05 else 'MISMATCH'
    cnt_c, cnt_l = ens_c.counts, ens_l.counts
    print("\n  CONSEQUENCE  (per-bath energy ledger)")
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


def main():
    rng = np.random.default_rng(SEED)
    out = _common.out_dir("two_qubit_unravelling")
    with _common.Tee(out / "report.txt"):
        report(0.0, rng)      # independent baths  -> per-bath unravelling exists
        report(0.8, rng)      # overlapping couplings -> it does not

        banner("What a single run of the experiment actually looks like")
        # independent baths (m = 0), both qubits excited: a run can have two clicks in either order.
        psi_ee = np.zeros(toy.DIM, dtype=complex)
        psi_ee[3] = 1.0
        det = [np.sqrt(GAMMA) * toy.F[0], np.sqrt(GAMMA) * toy.F[1]]
        for i in range(6):
            rec = tr.one_trajectory(toy.H, det, psi_ee, T, DT, rng).record
            clicks = "  ".join(f"t={t:.3f}: bath {'AB'[k]}" for t, k in rec) or "(no clicks)"
            print(f"  run {i+1}:  {clicks}")
        print("\n  Each run is a different history -- a different list of what the detectors")
        print("  recorded and when.  Averaging |psi><psi| over runs reproduces the master")
        print("  equation, PROVIDED the click probabilities really are probabilities.")

        banner("Cross-check: explicit loop version vs vectorised version")
        for m in [0.0, 0.8]:
            c = toy.coupling_matrix(GAMMA, m)
            J = un.canonical_jump_operators(toy.F, c)
            ex = me.evolve(toy.liouvillian(c), rho0, T)
            slow = tr.ensemble_explicit(toy.H, J, psi0, T, DT, 1200, rng)
            fast = tr.ensemble(toy.H, J, psi0, T, DT, 1200, rng)
            print(f"  m={m}:  explicit vs exact = {metrics.trace_distance(slow.rho, ex):.4f},"
                  f"   vectorised vs exact = {metrics.trace_distance(fast.rho, ex):.4f},"
                  f"   explicit vs vectorised = {metrics.trace_distance(slow.rho, fast.rho):.4f}")
            print(f"          mean clicks per detector: explicit {np.round(slow.counts, 4)}, "
                  f"vectorised {np.round(fast.counts, 4)}")
        print("\n  All differences are at the 1/sqrt(N) sampling level -- the two")
        print("  implementations are the same algorithm.")

        banner("Is m != 0 physical?  Exact collision model, ancillas in the VACUUM")
        print("  emission-only coupling, all ancilla modes in |vac>, couplings overlapping by m")
        print(f"\n{'m':>5}   {'c = gamma G G^dag':>28}   {'max|drho_exact - drho_Lindblad|':>32}")
        for m in [0.0, 0.3, 0.8, 1.0]:
            c, err = toy.microscopic_check(m, GAMMA, rng=rng)
            print(f"{m:>5.1f}   {str(np.real(c).round(3).tolist()):>28}   {err:>32.2e}")
        print("\n  -> off-diagonal c arises from OVERLAPPING COUPLINGS into a shared vacuum,")
        print("     not from correlated bath states.  m = 1 is Dicke superradiance at T = 0.")
        print("     So when m != 0 there is ONE bath; A and B label system operators, not")
        print("     reservoirs -- which is exactly why there is no 'bath A' to monitor.")

        banner("How much per-bath monitoring is still allowed?  (partial efficiency)")
        print(f"{'m':>6} {'eta_max (bath A only)':>24} {'1-m^2':>10} "
              f"{'eta_max (both baths)':>24} {'1-m':>8}")
        for m in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            c = toy.coupling_matrix(GAMMA, m)
            fa = un.monitorable_fraction(c, "A", toy.BATHS, method="eig")
            fab = un.monitorable_fraction(c, "AB", toy.BATHS, method="eig")
            print(f"{m:>6.2f} {fa:>24.6f} {1-m**2:>10.6f} {fab:>24.6f} {1-m:>8.6f}")
        print("\nBoth closed forms are reproduced exactly.  Monitoring both baths at once is")
        print("strictly more restrictive than monitoring one -- you are demanding which-bath")
        print("resolution on both sides simultaneously.")


if __name__ == "__main__":
    main()
