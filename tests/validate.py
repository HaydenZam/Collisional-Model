"""Checks of colmodel against independent calculations.

Run:  python tests/validate.py            (all checks, about half a minute)
      python tests/validate.py ancilla    (only checks whose name contains "ancilla")

Each check prints PASS/FAIL with the number it was judged on; the exit code is 1
if any check fails. Everything is seeded.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import qutip as qt
from scipy.linalg import expm

from colmodel import (Params, ancilla, collision, correlations as cr, master_equation as me,
                      metrics, toy, trajectories as tr, unravelling as un)
from colmodel.model import collision_hamiltonian, ground_chain
from colmodel.operators import (I2, KET0, KET1, N_OP, SM, SP, SX, SY, SZ, embed, is_state, kron,
                                marginals, ptrace)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def rand_density(d, rng):
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def physical_states(rng):
    """A spread of valid S-R states, with and without local coherence."""
    out = {n: ancilla.representative(n) for n in ancilla.REPRESENTATIVES}
    out["mixture"] = ancilla.mixture()
    out["thermal"] = ancilla.thermal_correlated(0.3, 0.2, 0.3, -0.1, 0.2)
    out["coherent"] = ancilla.coherent_product(0.3, 0.2 + 0.1j, 0.6, -0.15 + 0.3j)
    for k in range(3):
        out[f"random{k}"] = ancilla.random_state(seed=k)
    out["general"] = rand_density(4, rng)
    return out


# ============================================================================
# operators
# ============================================================================
@check
def operators_conventions():
    errs = [
        np.abs(SP @ KET0 - KET1).max(), np.abs(SM @ KET1 - KET0).max(),
        np.abs(SZ @ KET1 - KET1).max(), np.abs(SZ @ KET0 + KET0).max(),       # excited = +1
        np.abs(SP - 0.5 * (SX + 1j * SY)).max(), np.abs(SM - 0.5 * (SX - 1j * SY)).max(),
        np.abs(SX @ SY - SY @ SX - 2j * SZ).max(), np.abs(SY @ SZ - SZ @ SY - 2j * SX).max(),
        np.abs(N_OP - 0.5 * (I2 + SZ)).max(),
    ]
    return max(errs) < 1e-15, f"max error {max(errs):.1e}"


@check
def operators_ptrace_vs_qutip():
    rng = np.random.default_rng(1)
    rho = rand_density(8, rng)
    q = qt.Qobj(rho, dims=[[2, 2, 2], [2, 2, 2]])
    err = max(np.abs(ptrace(rho, keep, 3) - q.ptrace(keep).full()).max()
              for keep in ([0], [1], [2], [0, 2], [1, 2], [0, 1, 2]))
    return err < 1e-14, f"max error {err:.1e}"


# ============================================================================
# ancilla states
# ============================================================================
@check
def ancilla_all_constructors_give_states():
    rng = np.random.default_rng(2)
    states = list(physical_states(rng).values())
    states += [ancilla.random_state(seed=k) for k in range(50)]
    states += [ancilla.bell_diagonal(*c) for c in [(0.2, -0.3, 0.1), (1, 1, -1), (0.5, 0.5, 0), (-1, -1, -1)]]
    bad = sum(not is_state(r) for r in states)
    return bad == 0, f"{len(states)} states, {bad} invalid"


@check
def ancilla_zero_local_coherence_where_claimed():
    states = [ancilla.representative(n) for n in ("classical", "discord", "entangled")]
    states += [ancilla.random_state(seed=k) for k in range(50)]
    states += [ancilla.thermal_correlated(0.2, 0.4, 0.3, 0.1, -0.2)]
    worst = max(cr.local_coherence(r) for r in states)
    prod = cr.local_coherence(ancilla.representative("product"))
    return worst < 1e-14 and abs(prod - 1) < 1e-14, f"max coherence {worst:.1e}; product {prod:.3f}"


@check
def ancilla_representatives_have_their_class():
    got = {n: cr.classify(ancilla.representative(n)) for n in ancilla.REPRESENTATIVES}
    return all(k == v for k, v in got.items()), str(got)


@check
def ancilla_product_is_plus_plus():
    plus = 0.5 * np.ones((2, 2))
    rA, rB = marginals(ancilla.representative("product"))
    err = max(np.abs(rA - plus).max(), np.abs(rB - plus).max())
    return err < 1e-15, f"marginals vs |+><+|: {err:.1e}"


@check
def ancilla_thermal_populations_are_excited_populations():
    worst = 0.0
    for p_s in np.linspace(0.05, 0.95, 7):
        for p_r in np.linspace(0.05, 0.95, 7):
            m = me.ancilla_moments(ancilla.thermal_correlated(p_s, p_r, 0.1, 0.0, 0.0))
            worst = max(worst, abs(m["p_s"] - p_s), abs(m["p_r"] - p_r))
    return worst < 1e-14, f"max |p read back - p| = {worst:.1e}"


@check
def ancilla_coherence_scaling_keeps_phase():
    tau = 1e-3
    rho = 0.5 * ancilla.coherent_product(0.3, 0.2 + 0.1j, 0.6, -0.15 + 0.3j) + 0.5 * ancilla.representative("entangled")
    m_in = me.ancilla_moments(rho)
    m_out = me.ancilla_moments(ancilla.collision_scaled_product(rho, tau))
    err_c = np.abs(m_out["means"] - np.sqrt(tau) * m_in["means"]).max()
    err_p = max(abs(m_out["p_s"] - m_in["p_s"]), abs(m_out["p_r"] - m_in["p_r"]))
    q = ancilla.coherent_qubit(0.3, 0.2 + 0.1j)
    err_q = abs(np.trace(SM @ q) - (0.2 + 0.1j)) + abs(np.trace(N_OP @ q) - 0.3)
    ok = max(err_c, err_p, err_q) < 1e-15
    return ok, f"<s->: {err_c:.1e}, populations: {err_p:.1e}, coherent_qubit: {err_q:.1e}"


# ============================================================================
# correlations
# ============================================================================
@check
def correlations_concurrence_vs_qutip():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        rho = rand_density(4, rng)
        worst = max(worst, abs(cr.concurrence(rho) - qt.concurrence(qt.Qobj(rho, dims=[[2, 2], [2, 2]]))))
    bell = ancilla.representative("entangled")
    ok = worst < 1e-10 and abs(cr.concurrence(bell) - 1) < 1e-12 and abs(cr.negativity(bell) - 0.5) < 1e-12
    return ok, f"max |C - C_qutip| = {worst:.1e}; Bell: C = {cr.concurrence(bell):.6f}, N = {cr.negativity(bell):.6f}"


def _luo_discord(c1, c2, c3):
    """Closed-form discord of a Bell-diagonal state (Luo, PRA 77, 042303 (2008))."""
    lam = 0.25 * np.array([1 - c1 - c2 - c3, 1 - c1 + c2 + c3, 1 + c1 - c2 + c3, 1 + c1 + c2 - c3])
    lam = lam[lam > 1e-15]
    I = 2 + np.sum(lam * np.log2(lam))
    c = max(abs(c1), abs(c2), abs(c3))
    f = lambda x: x * np.log2(x) if x > 0 else 0.0
    C = 0.5 * f(1 - c) + 0.5 * f(1 + c)
    return I - C


@check
def correlations_discord_vs_closed_form():
    cases = [(0.5, 0.5, 0.0), (1, -1, 1), (0.3, -0.2, 0.4), (0.6, 0.1, -0.2), (1, 0, 0), (-0.4, -0.4, 0.3)]
    worst = max(abs(cr.discord(ancilla.bell_diagonal(*c)) - _luo_discord(*c)) for c in cases)
    return worst < 1e-6, f"max |grid - Luo| = {worst:.1e} over {len(cases)} Bell-diagonal states"


# ============================================================================
# master equation
# ============================================================================
def _old_style_liouvillian(p, rho_SR):
    """The master equation exactly as the original Collision_Model.py assembled it."""
    N = p.N
    q = lambda a: qt.Qobj(a, dims=[[2] * N, [2] * N])
    m = me.ancilla_moments(rho_SR)
    b, J = m["means"], m["J"]
    sp1, sm1, spN, smN = (q(embed(o, s, N)) for o, s in ((SP, 0), (SM, 0), (SP, N - 1), (SM, N - 1)))
    H = q(me.chain_hamiltonian(p))
    H += p.gL_col * (b[0] * sp1 + b[1] * sm1) + p.gR_col * (b[2] * spN + b[3] * smN)
    L = -1j * (qt.spre(H) - qt.spost(H))
    loc = lambda o: qt.spre(o) * qt.spost(o.dag()) - 0.5 * (qt.spre(o.dag() * o) + qt.spost(o.dag() * o))
    nl = lambda o1, o2: qt.spre(o1) * qt.spost(o2) - 0.5 * (qt.spre(o2 * o1) + qt.spost(o2 * o1))
    L += p.gL ** 2 * (m["p_s"] * loc(sp1) + (1 - m["p_s"]) * loc(sm1))
    L += p.gR ** 2 * (m["p_r"] * loc(spN) + (1 - m["p_r"]) * loc(smN))
    gg = p.gL * p.gR
    L += gg * J["mm"] * (nl(sp1, spN) + nl(spN, sp1)) + gg * J["mp"] * (nl(sp1, smN) + nl(smN, sp1))
    L += gg * J["pm"] * (nl(sm1, spN) + nl(spN, sm1)) + gg * J["pp"] * (nl(sm1, smN) + nl(smN, sm1))
    return L.full()


@check
def master_equation_D_matches_generic_formula():
    rng = np.random.default_rng(4)
    worst, min_eig = 0.0, np.inf
    for rho in physical_states(rng).values():
        D, _ = me.kossakowski(rho, 0.2, 0.3)
        anc = [kron(SM, I2), kron(SP, I2), kron(I2, SM), kron(I2, SP)]       # partner of each A_a
        g = [0.2, 0.2, 0.3, 0.3]
        ref = np.array([[g[a] * g[b] * np.trace(rho @ anc[b].conj().T @ anc[a]) for b in range(4)] for a in range(4)])
        worst = max(worst, np.abs(D - ref).max())
        min_eig = min(min_eig, np.linalg.eigvalsh(D).min())
    return worst < 1e-15 and min_eig > -1e-15, f"max |D - g g <b^dag b>| = {worst:.1e}; min eig {min_eig:.1e}"


@check
def master_equation_liouvillian_matches_original_assembly():
    rng = np.random.default_rng(5)
    states = physical_states(rng)
    tau = 1e-3
    states["coherent"] = ancilla.coherent_product(0.3, np.sqrt(tau) * (0.2 + 0.1j), 0.6, np.sqrt(tau) * (-0.15 + 0.3j))
    states["product"] = ancilla.collision_scaled_product(ancilla.representative("product"), tau)
    worst = 0.0
    for N in (2, 3, 4):
        p = Params(N=N, J=0.3, omega=0.1, gL=0.2, gR=0.25, tau=tau)
        for name in ("entangled", "random1", "coherent", "product", "thermal"):
            worst = max(worst, np.abs(me.liouvillian(p, states[name]) - _old_style_liouvillian(p, states[name])).max())
    return worst < 1e-12, f"max difference {worst:.1e} (N = 2, 3, 4; 5 ancillas)"


@check
def master_equation_liouvillian_is_trace_and_hermiticity_preserving():
    rng = np.random.default_rng(6)
    p = Params(N=3, tau=1e-3)
    rho_SR = 0.5 * ancilla.random_state(seed=2) + 0.5 * ancilla.collision_scaled_product(ancilla.representative("product"), 1e-3)
    L = me.liouvillian(p, rho_SR)
    d = 2 ** p.N
    tp = np.abs(me.vec(np.eye(d)).conj() @ L).max()
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    hp = np.abs(me.unvec(L @ me.vec(X.conj().T), d) - me.unvec(L @ me.vec(X), d).conj().T).max()
    return max(tp, hp) < 1e-12, f"trace: {tp:.1e}, hermiticity: {hp:.1e}"


def _permute_SRC_to_SCR(M, N):
    """Reorder an operator on S (x) R (x) C_1..C_N into S (x) C_1..C_N (x) R."""
    n = N + 2
    t = M.reshape([2] * (2 * n))
    order = [0] + list(range(2, n)) + [1]
    return t.transpose(order + [o + n for o in order]).reshape(2 ** n, 2 ** n)


@check
def master_equation_drive_is_first_order_term():
    """H_1 = tr_SR[V (rho_SR (x) 1_C)], with V the coupling part of the collision Hamiltonian."""
    worst = 0.0
    for N in (2, 3):
        p = Params(N=N, tau=1e-3, gL=0.2, gR=0.35)
        V = collision_hamiltonian(p) - collision_hamiltonian(p.with_(gL=0.0, gR=0.0))
        rho = ancilla.coherent_product(0.3, 0.2 + 0.1j, 0.6, -0.15 + 0.3j)
        Omega = _permute_SRC_to_SCR(np.kron(rho, np.eye(2 ** N)), N)
        H1 = ptrace(V @ Omega, range(1, N + 1), N + 2)
        worst = max(worst, np.abs(H1 - me.drive_hamiltonian(p, rho)).max())
    return worst < 1e-12, f"max difference {worst:.1e}"


@check
def master_equation_connected_minus_raw_is_drive_dissipator():
    """D_raw - D_connected = v v^dag, and its dissipator is -1/2 [H, [H, .]] with H = v.A."""
    rho = ancilla.coherent_product(0.3, 0.2 + 0.1j, 0.6, -0.15 + 0.3j)
    gL, gR, N = 0.2, 0.3, 2
    D_raw, _ = me.kossakowski(rho, gL, gR)
    D_con, _ = me.kossakowski(rho, gL, gR, connected=True)
    v = np.array([gL, gL, gR, gR]) * me.ancilla_moments(rho)["means"]
    err1 = np.abs(D_raw - D_con - np.outer(v, v.conj())).max()
    A = me.jump_operators(N)
    Hv = sum(v[a] * A[a] for a in range(4))
    Ldiff = me.superoperator(np.zeros((4, 4)), A, D_raw - D_con)
    K = me.superoperator(Hv, [], np.zeros((0, 0)))            # K(rho) = -i[Hv, rho]
    err2 = np.abs(Ldiff - 0.5 * K @ K).max()                   # 1/2 K K = -1/2 [Hv, [Hv, .]]
    con_min = np.linalg.eigvalsh(D_con).min()
    return max(err1, err2) < 1e-14 and con_min > -1e-15, f"{err1:.1e}, {err2:.1e}; min eig D_connected {con_min:.1e}"


@check
def master_equation_mesolve_agrees_with_expm():
    p = Params(N=2)
    rho = ancilla.representative("entangled")
    times = np.linspace(0, 2, 201)
    a = me.solve(p, rho, times, method="mesolve", options={"atol": 1e-12, "rtol": 1e-10})
    b = me.solve(p, rho, times, method="expm")
    err = metrics.trace_distance(a, b).max()
    return err < 1e-8, f"max trace distance {err:.1e}"


# ============================================================================
# collision model
# ============================================================================
@check
def collision_matches_independent_numpy_implementation():
    N, n_steps = 3, 25
    p = Params(N=N, J=(0.3, 0.5), omega=(0.1, 0.0, -0.2), omega_S=0.05, omega_R=-0.1, gL=0.2, gR=0.3, tau=1e-2)
    rho_SR = 0.5 * ancilla.coherent_product(0.3, 0.1 + 0.05j, 0.6, -0.05 + 0.1j) + 0.5 * ancilla.random_state(seed=7)
    res = collision.run_collisions(p, rho_SR, n_steps)
    U = expm(-1j * p.tau * collision_hamiltonian(p))
    rho_C = ground_chain(N)
    worst = 0.0
    for s in range(1, n_steps + 1):
        tot = _permute_SRC_to_SCR(np.kron(rho_SR, rho_C), N)
        rho_C = ptrace(U @ tot @ U.conj().T, range(1, N + 1), N + 2)
        worst = max(worst, np.abs(rho_C - res.states[s]).max())
    valid = all(is_state(r) for r in res.states)
    return worst < 1e-12 and valid, f"max difference {worst:.1e} over {n_steps} steps; all states valid: {valid}"


@check
def collision_converges_to_master_equation():
    """Error O(tau) for Bell-diagonal ancillas; only O(sqrt(tau)) with mixed <sz sx> correlations."""
    T = 0.5
    zx = 0.25 * (kron(I2, I2) + 0.5 * kron(SZ, SX))

    def err(rho, tau):
        p = Params(tau=tau)
        n = int(round(T / tau))
        res = collision.run_collisions(p, rho, n, store_every=n)
        return metrics.trace_distance(me.solve(p, rho, res.times, method="expm")[-1], res.states[-1])

    r_bell = err(ancilla.representative("entangled"), 2e-3) / err(ancilla.representative("entangled"), 1e-3)
    r_zx = err(zx, 2e-3) / err(zx, 1e-3)
    ok = abs(r_bell - 2) < 0.05 and abs(r_zx - np.sqrt(2)) < 0.05
    return ok, f"error ratio for tau -> tau/2: Bell {r_bell:.3f} (expect 2), <sz sx> state {r_zx:.3f} (expect 1.414)"


# ============================================================================
# unravelling
# ============================================================================
@check
def unravelling_fraction_methods_agree():
    worst, worst_lr, n = 0.0, 0.0, 0
    rng = np.random.default_rng(8)
    for _ in range(40):
        rho = ancilla.random_state(rng=rng)
        D, _ = me.kossakowski(rho, 0.2, 0.2)
        if not un.gksl_ok(D):
            continue
        fb = un.monitorable_fraction(D, "L")
        fe = un.monitorable_fraction(D, "L", method="eig")
        fs = un.fraction_schur(D, "L")
        fc = un.fraction_closed_form(D)
        fr = un.monitorable_fraction(D, "R")
        worst = max(worst, abs(fb - fe), abs(fb - fc), 0 if fs is None else abs(fb - fs))
        worst_lr = max(worst_lr, abs(fb - fr))
        n += 1
    return worst < 1e-6 and worst_lr < 1e-6, f"{n} states: max spread of bisection/eig/Schur/closed form {worst:.1e}; |frac_L - frac_R| {worst_lr:.1e}"


@check
def unravelling_strict_iff_no_inter_bath_block():
    D_ind, _ = me.kossakowski(ancilla.thermal_correlated(0.3, 0.4), 0.2, 0.2)
    D_bell, _ = me.kossakowski(ancilla.representative("entangled"), 0.2, 0.2)
    ok = (un.strict_unravellable(D_ind) and un.monitorable_fraction(D_ind) == 1.0
          and not un.strict_unravellable(D_bell))
    return ok, (f"uncorrelated thermal: strict {un.strict_unravellable(D_ind)}, frac {un.monitorable_fraction(D_ind)}; "
                f"Bell: strict {un.strict_unravellable(D_bell)}")


@check
def unravelling_canonical_channels_rebuild_D():
    D, _ = me.kossakowski(ancilla.thermal_correlated(0.3, 0.15, 0.5, -0.2, 0.2), 0.2, 0.2)
    chans, _ = un.canonical_channels(D)
    rebuilt = sum(c["rate"] * np.outer(c["vec"], c["vec"].conj()) for c in chans)
    err = np.abs(rebuilt - D).max()
    return err < 1e-15, f"max |sum rate v v^dag - D| = {err:.1e}"


@check
def toy_partial_monitoring_closed_forms():
    worst = 0.0
    for m in np.linspace(0, 0.95, 20):
        c = toy.coupling_matrix(1.0, m)
        for method, tol in (("bisection", 1e-8), ("eig", 1e-12)):
            ea = un.monitorable_fraction(c, "A", toy.BATHS, method=method)
            eab = un.monitorable_fraction(c, "AB", toy.BATHS, method=method)
            worst = max(worst, abs(ea - (1 - m ** 2)) / tol, abs(eab - (1 - m)) / tol)
    return worst < 1, f"one bath 1 - m^2, both baths 1 - m: max error {worst:.2f} x tolerance"


@check
def toy_vacuum_collision_model_gives_overlap_matrix():
    rng = np.random.default_rng(9)
    worst = max(toy.microscopic_check(m, rng=rng)[1] for m in (0.0, 0.3, 0.8, 1.0))
    return worst < 1e-5, f"max |drho_exact/dt - L(rho)| = {worst:.1e} (dt = 1e-6)"


# ============================================================================
# trajectories
# ============================================================================
@check
def trajectories_instrument_check():
    rng = np.random.default_rng(10)
    c = toy.coupling_matrix(1.0, 0.8)
    L = toy.liouvillian(c)
    comp_c, res_c = tr.instrument_check(toy.H, un.canonical_jump_operators(toy.F, c), L, 1e-4, rng)
    comp_l, res_l = tr.instrument_check(toy.H, toy.per_bath_jumps(c), L, 1e-4, rng)
    ok = comp_c < 1e-7 and res_c < 1e-3 and res_l > 0.1
    return ok, f"canonical: completeness {comp_c:.1e}, residual {res_c:.1e}; per-bath residual {res_l:.2f}"


@check
def trajectories_converge_to_exact_for_valid_unravelling_only():
    rng = np.random.default_rng(11)
    c = toy.coupling_matrix(1.0, 0.8)
    psi0 = np.zeros(4, dtype=complex)
    psi0[1] = psi0[2] = 1 / np.sqrt(2)
    T, dt = 1.5, 2.5e-3
    exact = me.evolve(toy.liouvillian(c), np.outer(psi0, psi0.conj()), T)
    canon = tr.ensemble(toy.H, un.canonical_jump_operators(toy.F, c), psi0, T, dt, 4000, rng)
    local = tr.ensemble(toy.H, toy.per_bath_jumps(c), psi0, T, dt, 4000, rng)
    slow = tr.ensemble_explicit(toy.H, un.canonical_jump_operators(toy.F, c), psi0, T, dt, 300, rng,
                                observable=toy.N_TOT)
    d_c, d_l, d_s = (metrics.trace_distance(e.rho, exact) for e in (canon, local, slow))
    n_ok = abs(slow.history[0] - 1.0) < 1e-15
    ok = d_c < 0.03 and d_l > 0.1 and d_s < 0.1 and n_ok
    return ok, f"trace distance to exact: canonical {d_c:.4f}, per-bath {d_l:.4f}, explicit loop (300 runs) {d_s:.4f}"


# ============================================================================
def main(pattern=""):
    selected = [f for f in CHECKS if pattern in f.__name__]
    n_fail = 0
    t_all = time.time()
    for fn in selected:
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:                      # report and carry on
            ok, detail = False, f"raised {type(e).__name__}: {e}"
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {fn.__name__:<62} {time.time()-t0:5.1f}s  {detail}")
    print(f"\n{len(selected) - n_fail}/{len(selected)} checks passed in {time.time()-t_all:.0f}s")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else "") else 1)
