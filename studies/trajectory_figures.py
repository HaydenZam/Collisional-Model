"""Pictures of the stochastic trajectories behind two_qubit_unravelling.py (toy model).

trajectories_anatomy.png   (m = 0, independent baths: the VALID case)
    A  six individual runs: staircase decays with detector clicks marked,
       against the smooth master-equation curve they average to
    B  ensemble means for N = 1, 10, 100, 1000, 20000 converging to it
    C  click raster: every click of 80 runs, coloured by which bath fired

valid_vs_invalid.png       (m = 0.8, one shared bath: the INVALID case)
    A  <N>(t) for the canonical unravelling (tracks the exact solution) and for
       the forced per-bath unravelling (tracks a DIFFERENT solution)
    B  |ensemble mean - exact| vs time, with a 4x-sample overlay showing the
       per-bath error is systematic while the canonical error is 1/sqrt(N)

Output: runs/trajectory_figures/. Seeded (7).
"""

import _common
import numpy as np
from scipy.linalg import expm

from colmodel import master_equation as me, toy, trajectories as tr, unravelling as un

SEED = 7
GAMMA = 1.0
T, DT = 3.0, 2e-3
N_STEPS = int(round(T / DT))
TIMES = np.arange(N_STEPS + 1) * DT

# both qubits excited: trajectories are clean two-step staircases
PSI0 = np.zeros(toy.DIM, dtype=complex)
PSI0[3] = 1.0
RHO0 = np.outer(PSI0, PSI0.conj())


def exact_history(c):
    """<N>(t) from the master equation, by repeated application of exp(L dt)."""
    P = expm(toy.liouvillian(c) * DT)
    v = me.vec(RHO0)
    out = np.empty(N_STEPS + 1)
    for s in range(N_STEPS + 1):
        out[s] = np.real(np.trace(toy.N_TOT @ me.unvec(v, toy.DIM)))
        v = P @ v
    return out


def single_run(jumps, rng):
    run = tr.one_trajectory(toy.H, jumps, PSI0, T, DT, rng, observable=toy.N_TOT)
    return run.history, run.record


def ensemble_history(jumps, n_traj, rng):
    return tr.ensemble(toy.H, jumps, PSI0, T, DT, n_traj, rng, observable=toy.N_TOT).history


def figure_anatomy(plt, path, rng):
    c = toy.coupling_matrix(GAMMA, 0.0)
    jumps = [np.sqrt(GAMMA) * toy.F[0], np.sqrt(GAMMA) * toy.F[1]]   # per-bath detectors
    exact = exact_history(c)
    col = ["#d1495b", "#00798c"]                                     # bath A, bath B

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]})

    # A: individual runs
    for i in range(6):
        n_of_t, clicks = single_run(jumps, rng)
        ax[0].step(TIMES, n_of_t, where="post", lw=1.1, alpha=0.75, color=plt.cm.viridis(i / 6))
        for t, k in clicks:
            ax[0].plot(t, n_of_t[int(round(t / DT))], "v", ms=7, color=col[k], mec="k", mew=0.4, zorder=5)
    ax[0].plot(TIMES, exact, "k--", lw=2.2, label="master equation  Tr[N rho(t)]")
    ax[0].plot([], [], "v", color=col[0], mec="k", label="click, bath A")
    ax[0].plot([], [], "v", color=col[1], mec="k", label="click, bath B")
    ax[0].set_title("A.  six individual runs.  Each is a staircase; the smooth curve is "
                    "what they average to", fontsize=11, loc="left")
    ax[0].set_ylabel(r"excitations  $\langle N\rangle$")
    ax[0].set_ylim(-0.15, 2.15)
    ax[0].legend(fontsize=9, loc="upper right")

    # B: convergence
    for n, cm in zip([1, 10, 100, 1000, 20000], np.linspace(0.85, 0.15, 5)):
        ax[1].plot(TIMES, ensemble_history(jumps, n, rng), lw=1.4, color=plt.cm.plasma(cm), label=f"N = {n}")
    ax[1].plot(TIMES, exact, "k--", lw=2.2, label="exact")
    ax[1].set_title("B.  ensemble mean converges to the master equation as 1/sqrt(N)", fontsize=11, loc="left")
    ax[1].set_ylabel(r"$\langle N\rangle$")
    ax[1].legend(fontsize=9, ncol=2)

    # C: click raster
    for i in range(80):
        _n, clicks = single_run(jumps, rng)
        for t, k in clicks:
            ax[2].plot(t, i, "|", ms=6, mew=1.5, color=col[k])
    ax[2].set_title("C.  the raw data: every click of 80 runs, coloured by which bath fired",
                    fontsize=11, loc="left")
    ax[2].set_ylabel("run index")
    ax[2].set_xlabel("time  " + r"($\gamma t$)")

    for a in ax:
        a.set_xlim(0, T)
        a.grid(alpha=0.25)
    fig.suptitle("m = 0:  two independent baths.  The per-bath unraveling is valid, "
                 "so these records exist.", fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def figure_valid_vs_invalid(plt, path, rng, n_traj=4000):
    m = 0.8
    c = toy.coupling_matrix(GAMMA, m)
    canonical = un.canonical_jump_operators(toy.F, c)          # valid
    per_bath = toy.per_bath_jumps(c)                           # invalid
    exact = exact_history(c)
    exact_block = exact_history(un.block_part(c, toy.BATHS.values()))

    can1 = ensemble_history(canonical, n_traj, rng)
    loc1 = ensemble_history(per_bath, n_traj, rng)
    can4 = ensemble_history(canonical, 4 * n_traj, rng)
    loc4 = ensemble_history(per_bath, 4 * n_traj, rng)

    fig, ax = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1.3, 1.0]})
    ax[0].plot(TIMES, exact, "k-", lw=2.6, label="exact master equation")
    ax[0].plot(TIMES, can1, color="#2a9d8f", lw=1.8, label=f"canonical unraveling (N={n_traj})")
    ax[0].plot(TIMES, loc1, color="#e76f51", lw=1.8, label=f"forced per-bath unraveling (N={n_traj})")
    ax[0].plot(TIMES, exact_block, color="#e76f51", ls=":", lw=2.2,
               label="master equation with c -> block-diagonal(c)")
    ax[0].fill_between(TIMES, exact, loc1, color="#e76f51", alpha=0.15)
    ax[0].set_title("A.  the per-bath ensemble converges -- to the WRONG generator.\n"
                    "     It lands on the dotted curve, not the black one.", fontsize=11, loc="left")
    ax[0].set_ylabel(r"$\langle N\rangle$")
    ax[0].legend(fontsize=9)

    ax[1].semilogy(TIMES[1:], np.abs(can1 - exact)[1:], color="#2a9d8f", lw=1.3, label=f"canonical, N={n_traj}")
    ax[1].semilogy(TIMES[1:], np.abs(can4 - exact)[1:], color="#2a9d8f", lw=1.3, ls="--",
                   label=f"canonical, N={4*n_traj}")
    ax[1].semilogy(TIMES[1:], np.abs(loc1 - exact)[1:], color="#e76f51", lw=1.6, label=f"per-bath, N={n_traj}")
    ax[1].semilogy(TIMES[1:], np.abs(loc4 - exact)[1:], color="#e76f51", lw=1.6, ls="--",
                   label=f"per-bath, N={4*n_traj}")
    ax[1].set_title("B.  |ensemble mean - exact|.  Quadrupling N pushes the green curves down\n"
                    "     and leaves the orange ones put: statistical vs systematic.", fontsize=11, loc="left")
    ax[1].set_ylabel("absolute error")
    ax[1].set_xlabel("time  " + r"($\gamma t$)")
    ax[1].legend(fontsize=9, ncol=2)

    for a in ax:
        a.set_xlim(0, T)
        a.grid(alpha=0.25)
    fig.suptitle("m = 0.8:  one shared bath.  There is no 'detector in bath A', "
                 "and the simulation shows it.", fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _common.out_dir("trajectory_figures")
    rng = np.random.default_rng(SEED)
    figure_anatomy(plt, out / "trajectories_anatomy.png", rng)
    figure_valid_vs_invalid(plt, out / "valid_vs_invalid.png", rng)


if __name__ == "__main__":
    main()
