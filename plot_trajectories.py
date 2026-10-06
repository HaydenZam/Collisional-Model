"""
Pictures of the stochastic trajectories behind unraveling_demo.py
================================================================================

Produces two figures:

  trajectories_anatomy.png   (m = 0, independent baths -- the VALID case)
      A  six individual runs: staircase decays with detector clicks marked,
         against the smooth master-equation curve they average to
      B  ensemble means for N = 1, 10, 100, 1000, 20000 converging to it
      C  click raster: every click of 80 runs, coloured by which bath fired

  valid_vs_invalid.png       (m = 0.8, one shared bath -- the INVALID case)
      A  <N>(t) for the canonical unraveling (tracks the exact solution) and
         for the forced per-bath unraveling (tracks a DIFFERENT solution)
      B  |ensemble mean - exact| vs time, with a 4x-sample overlay showing the
         per-bath error is systematic while the canonical error is 1/sqrt(N)

Run after unraveling_demo.py is in the same directory:  python3 plot_trajectories.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

from unraveling_example import (DIM, N_TOT, H, F, GAMMA, kossakowski, jump_ops_from_c,
                             liouvillian, block_part)

rng = np.random.default_rng(7)

T, DT = 3.0, 2e-3
N_STEPS = int(round(T / DT))
TIMES = np.arange(N_STEPS + 1) * DT

# start with both qubits excited: trajectories are clean two-step staircases
PSI0 = np.zeros(DIM, dtype=complex); PSI0[3] = 1.0
RHO0 = np.outer(PSI0, PSI0.conj())


# ----------------------------------------------------------------------------
# the exact master-equation curve
# ----------------------------------------------------------------------------

def exact_history(c):
    """<N>(t) from the master equation, by repeated application of exp(L dt)."""
    P = expm(liouvillian(c) * DT)
    rho = RHO0.reshape(-1).copy()
    out = np.empty(N_STEPS + 1)
    for s in range(N_STEPS + 1):
        out[s] = np.real(np.trace(N_TOT @ rho.reshape(DIM, DIM)))
        rho = P @ rho
    return out


# ----------------------------------------------------------------------------
# trajectories, with the full time series recorded
# ----------------------------------------------------------------------------

def single_run(jumps):
    """One trajectory.  Returns <N>(t) along the run and the list of (t, detector)."""
    Gamma = sum(Lk.conj().T @ Lk for Lk in jumps)
    Heff = H - 0.5j * Gamma
    psi = PSI0.copy()
    n_of_t = np.empty(N_STEPS + 1)
    clicks = []

    for s in range(N_STEPS + 1):
        n_of_t[s] = np.real(np.vdot(psi, N_TOT @ psi))
        if s == N_STEPS:
            break
        p = np.array([DT * np.real(np.vdot(Lk @ psi, Lk @ psi)) for Lk in jumps])
        if rng.random() < p.sum():
            r = rng.random() * p.sum()
            k = 0
            while r > p[k] and k < len(jumps) - 1:
                r -= p[k]; k += 1
            psi = jumps[k] @ psi
            clicks.append((s * DT, k))
        else:
            psi = psi - 1j * DT * (Heff @ psi)
        psi /= np.linalg.norm(psi)

    return n_of_t, clicks


def ensemble_history(jumps, n_traj):
    """Mean <N>(t) over n_traj trajectories, vectorised (same algorithm as above)."""
    jumps = list(jumps)
    Gamma = sum(Lk.conj().T @ Lk for Lk in jumps)
    step = np.eye(DIM, dtype=complex) - 1j * (H - 0.5j * Gamma) * DT

    psi = np.tile(PSI0, (n_traj, 1))
    out = np.empty(N_STEPS + 1)

    for s in range(N_STEPS + 1):
        out[s] = np.real(np.einsum("ni,ij,nj->", psi.conj(), N_TOT, psi)) / n_traj
        if s == N_STEPS:
            break
        cand = np.stack([psi @ Lk.T for Lk in jumps])
        w = DT * np.sum(np.abs(cand) ** 2, axis=2)
        u = rng.random(n_traj)
        jumped = u < w.sum(axis=0)
        pick = np.argmax(np.cumsum(w, axis=0) > u[None, :], axis=0)
        new = psi @ step.T
        if jumped.any():
            idx = np.where(jumped)[0]
            new[idx] = cand[pick[idx], idx]
        psi = new
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)

    return out


# ============================================================================
# FIGURE 1 -- anatomy of the unraveling, m = 0
# ============================================================================

def figure_anatomy(fname="trajectories_anatomy.png"):
    c = kossakowski(GAMMA, 0.0)
    jumps = [np.sqrt(GAMMA) * F[0], np.sqrt(GAMMA) * F[1]]   # per-bath detectors
    exact = exact_history(c)
    col = ["#d1495b", "#00798c"]                              # bath A, bath B

    fig, ax = plt.subplots(3, 1, figsize=(9, 11),
                           gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]})

    # --- A: individual runs -------------------------------------------------
    for i in range(6):
        n_of_t, clicks = single_run(jumps)
        ax[0].step(TIMES, n_of_t, where="post", lw=1.1, alpha=0.75,
                   color=plt.cm.viridis(i / 6))
        for t, k in clicks:
            ax[0].plot(t, n_of_t[int(round(t / DT))], "v", ms=7,
                       color=col[k], mec="k", mew=0.4, zorder=5)
    ax[0].plot(TIMES, exact, "k--", lw=2.2, label="master equation  Tr[N rho(t)]")
    ax[0].plot([], [], "v", color=col[0], mec="k", label="click, bath A")
    ax[0].plot([], [], "v", color=col[1], mec="k", label="click, bath B")
    ax[0].set_title("A.  six individual runs.  Each is a staircase; the smooth curve is "
                    "what they average to", fontsize=11, loc="left")
    ax[0].set_ylabel(r"excitations  $\langle N\rangle$")
    ax[0].set_ylim(-0.15, 2.15); ax[0].legend(fontsize=9, loc="upper right")

    # --- B: convergence -----------------------------------------------------
    for n, cmap in zip([1, 10, 100, 1000, 20000], np.linspace(0.85, 0.15, 5)):
        ax[1].plot(TIMES, ensemble_history(jumps, n), lw=1.4,
                   color=plt.cm.plasma(cmap), label=f"N = {n}")
    ax[1].plot(TIMES, exact, "k--", lw=2.2, label="exact")
    ax[1].set_title("B.  ensemble mean converges to the master equation as 1/sqrt(N)",
                    fontsize=11, loc="left")
    ax[1].set_ylabel(r"$\langle N\rangle$"); ax[1].legend(fontsize=9, ncol=2)

    # --- C: click raster ----------------------------------------------------
    for i in range(80):
        _n, clicks = single_run(jumps)
        for t, k in clicks:
            ax[2].plot(t, i, "|", ms=6, mew=1.5, color=col[k])
    ax[2].set_title("C.  the raw data: every click of 80 runs, coloured by which bath "
                    "fired", fontsize=11, loc="left")
    ax[2].set_ylabel("run index"); ax[2].set_xlabel("time  " + r"($\gamma t$)")

    for a in ax:
        a.set_xlim(0, T); a.grid(alpha=0.25)
    fig.suptitle("m = 0:  two independent baths.  The per-bath unraveling is valid, "
                 "so these records exist.", fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


# ============================================================================
# FIGURE 2 -- valid vs invalid unraveling, m = 0.8
# ============================================================================

def figure_valid_vs_invalid(fname="valid_vs_invalid.png", n_traj=4000):
    m = 0.8
    c = kossakowski(GAMMA, m)
    canonical = jump_ops_from_c(c)                              # valid
    per_bath = [np.sqrt(np.real(c[j, j])) * F[j] for j in range(2)]   # invalid
    exact = exact_history(c)
    exact_block = exact_history(block_part(c))

    can1 = ensemble_history(canonical, n_traj)
    loc1 = ensemble_history(per_bath, n_traj)
    can4 = ensemble_history(canonical, 4 * n_traj)
    loc4 = ensemble_history(per_bath, 4 * n_traj)

    fig, ax = plt.subplots(2, 1, figsize=(9, 8),
                           gridspec_kw={"height_ratios": [1.3, 1.0]})

    ax[0].plot(TIMES, exact, "k-", lw=2.6, label="exact master equation")
    ax[0].plot(TIMES, can1, color="#2a9d8f", lw=1.8,
               label=f"canonical unraveling (N={n_traj})")
    ax[0].plot(TIMES, loc1, color="#e76f51", lw=1.8,
               label=f"forced per-bath unraveling (N={n_traj})")
    ax[0].plot(TIMES, exact_block, color="#e76f51", ls=":", lw=2.2,
               label="master equation with c -> block-diagonal(c)")
    ax[0].fill_between(TIMES, exact, loc1, color="#e76f51", alpha=0.15)
    ax[0].set_title("A.  the per-bath ensemble converges -- to the WRONG generator.\n"
                    "     It lands on the dotted curve, not the black one.",
                    fontsize=11, loc="left")
    ax[0].set_ylabel(r"$\langle N\rangle$"); ax[0].legend(fontsize=9)

    ax[1].semilogy(TIMES[1:], np.abs(can1 - exact)[1:], color="#2a9d8f", lw=1.3,
                   label=f"canonical, N={n_traj}")
    ax[1].semilogy(TIMES[1:], np.abs(can4 - exact)[1:], color="#2a9d8f", lw=1.3,
                   ls="--", label=f"canonical, N={4*n_traj}")
    ax[1].semilogy(TIMES[1:], np.abs(loc1 - exact)[1:], color="#e76f51", lw=1.6,
                   label=f"per-bath, N={n_traj}")
    ax[1].semilogy(TIMES[1:], np.abs(loc4 - exact)[1:], color="#e76f51", lw=1.6,
                   ls="--", label=f"per-bath, N={4*n_traj}")
    ax[1].set_title("B.  |ensemble mean - exact|.  Quadrupling N pushes the green "
                    "curves down\n     and leaves the orange ones put: statistical vs "
                    "systematic.", fontsize=11, loc="left")
    ax[1].set_ylabel("absolute error"); ax[1].set_xlabel("time  " + r"($\gamma t$)")
    ax[1].legend(fontsize=9, ncol=2)

    for a in ax:
        a.set_xlim(0, T); a.grid(alpha=0.25)
    fig.suptitle("m = 0.8:  one shared bath.  There is no 'detector in bath A', "
                 "and the simulation shows it.", fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    print("wrote", fname)


if __name__ == "__main__":
    figure_anatomy()
    figure_valid_vs_invalid()
