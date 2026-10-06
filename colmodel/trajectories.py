"""Quantum-jump (Monte Carlo wave function) unravelling of a GKSL generator.

Given a Hamiltonian H, a list of jump operators L_k (one detector each) and an
initial pure state, each time step dt:
    p_k = dt <psi|L_k^dag L_k|psi>     click probability of detector k
    with probability sum_k p_k a detector clicks; which one is drawn with
    weights p_k, and |psi> -> L_k |psi>;
    otherwise (no click, still an observation) |psi> -> (1 - i H_eff dt)|psi>,
    H_eff = H - (i/2) sum_k L_k^dag L_k;
    renormalise.
Averaging |psi><psi| reproduces the master equation, provided the jump
operators come from a valid decomposition of its Kossakowski matrix.

one_trajectory() is the readable reference; ensemble() is the same algorithm
vectorised over trajectories; ensemble_explicit() loops one_trajectory() and
exists to cross-check ensemble(). All randomness comes from the Generator `rng`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Run:
    psi: np.ndarray                 # final state
    counts: np.ndarray              # clicks per detector
    record: list                    # [(time, detector), ...]
    history: np.ndarray | None      # <O>(t) at the n_steps + 1 grid times, if an observable was given


@dataclass
class Ensemble:
    rho: np.ndarray                 # mean |psi><psi| at T
    counts: np.ndarray              # mean clicks per detector
    history: np.ndarray | None      # mean <O>(t), if an observable was given


def _heff(H, jumps):
    return H - 0.5j * sum(Lk.conj().T @ Lk for Lk in jumps)


def one_trajectory(H, jumps, psi0, T, dt, rng, observable=None):
    """One run of the experiment: a single click record."""
    n_steps = int(round(T / dt))
    jumps = list(jumps)
    n_det = len(jumps)
    Heff = _heff(H, jumps)

    psi = np.asarray(psi0, dtype=complex).copy()
    counts = np.zeros(n_det, dtype=int)
    record = []
    hist = np.empty(n_steps + 1) if observable is not None else None

    for s in range(n_steps):
        if hist is not None:
            hist[s] = np.real(np.vdot(psi, observable @ psi))
        # 1. click probability of each detector during this step
        p = np.zeros(n_det)
        post_jump = []
        for k in range(n_det):
            phi = jumps[k] @ psi
            post_jump.append(phi)
            p[k] = dt * float(np.real(np.vdot(phi, phi)))
        p_total = p.sum()
        # 2. did any detector click?
        if rng.random() < p_total:
            # 3a. yes: which one (inverse-CDF sampling)
            r = rng.random() * p_total
            k = 0
            while r > p[k] and k < n_det - 1:
                r -= p[k]
                k += 1
            psi = post_jump[k]
            counts[k] += 1
            record.append((s * dt, k))
        else:
            # 3b. no click: non-Hermitian no-jump evolution
            psi = psi - 1j * dt * (Heff @ psi)
        # 4. renormalise
        psi = psi / np.linalg.norm(psi)

    if hist is not None:
        hist[n_steps] = np.real(np.vdot(psi, observable @ psi))
    return Run(psi, counts, record, hist)


def ensemble_explicit(H, jumps, psi0, T, dt, n_traj, rng, observable=None):
    """n_traj calls of one_trajectory(), averaged. Slow; for cross-checks."""
    jumps = list(jumps)
    d = len(psi0)
    rho = np.zeros((d, d), dtype=complex)
    counts = np.zeros(len(jumps))
    hist = None
    for _ in range(n_traj):
        run = one_trajectory(H, jumps, psi0, T, dt, rng, observable)
        rho += np.outer(run.psi, run.psi.conj())
        counts += run.counts
        if observable is not None:
            hist = run.history if hist is None else hist + run.history
    return Ensemble(rho / n_traj, counts / n_traj, None if hist is None else hist / n_traj)


def ensemble(H, jumps, psi0, T, dt, n_traj, rng, observable=None):
    """Same algorithm as one_trajectory(), vectorised over n_traj trajectories.

    The two uniform numbers one_trajectory() draws per click are fused into
    one: conditioned on u < p_total, u is uniform on [0, p_total), which is the
    variable inverse-CDF sampling needs."""
    jumps = list(jumps)
    step = np.eye(len(psi0), dtype=complex) - 1j * _heff(H, jumps) * dt   # first order in dt
    n_steps = int(round(T / dt))

    psi = np.tile(np.asarray(psi0, dtype=complex), (n_traj, 1))         # (n_traj, d)
    counts = np.zeros(len(jumps))
    hist = np.empty(n_steps + 1) if observable is not None else None
    mean_obs = lambda: np.real(np.einsum("ni,ij,nj->", psi.conj(), observable, psi)) / n_traj

    for s in range(n_steps):
        if hist is not None:
            hist[s] = mean_obs()
        cand = np.stack([psi @ Lk.T for Lk in jumps])        # (n_jumps, n_traj, d)
        w = dt * np.sum(np.abs(cand) ** 2, axis=2)           # (n_jumps, n_traj)
        u = rng.random(n_traj)
        jumped = u < w.sum(axis=0)
        pick = np.argmax(np.cumsum(w, axis=0) > u[None, :], axis=0)
        new = psi @ step.T
        if jumped.any():
            idx = np.where(jumped)[0]
            new[idx] = cand[pick[idx], idx]
            np.add.at(counts, pick[idx], 1.0)
        psi = new
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)

    if hist is not None:
        hist[n_steps] = mean_obs()
    # psi.T @ psi.conj() = sum_i |psi_i><psi_i|  (psi.conj().T @ psi would be its transpose)
    rho = (psi.T @ psi.conj()) / n_traj
    return Ensemble(rho, counts / n_traj, hist)


def instrument_check(H, jumps, L_target, dt, rng, n_rho=5):
    """Kraus check of the one-step instrument M_0 = 1 - i H_eff dt, M_k = sqrt(dt) L_k:

    returns (completeness, residual):
      completeness = max |sum_i M_i^dag M_i - I|        (should be O(dt^2))
      residual     = max |sum_i M_i rho M_i^dag - rho - L(rho) dt| / dt over random rho,
                     with L_target the target superoperator (column-stacking vec).
    A residual that does not vanish as dt -> 0 means the jumps unravel a different
    generator."""
    jumps = list(jumps)
    d = H.shape[0]
    Ms = [np.eye(d, dtype=complex) - 1j * _heff(H, jumps) * dt] + [np.sqrt(dt) * Lk for Lk in jumps]
    completeness = float(np.max(np.abs(sum(M.conj().T @ M for M in Ms) - np.eye(d))))
    worst = 0.0
    for _ in range(n_rho):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        lhs = sum(M @ rho @ M.conj().T for M in Ms)
        rhs = rho + (L_target @ rho.reshape(-1, order="F")).reshape(d, d, order="F") * dt
        worst = max(worst, float(np.max(np.abs(lhs - rhs))))
    return completeness, worst / dt
