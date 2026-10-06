"""Exact repeated-collision dynamics (qutip used internally; numpy in and out).

Each step: rho_C -> tr_{S,R}[ U (rho_SR (x) rho_C) U^dag ],  U = exp(-i H tau),
with H = colmodel.model.collision_hamiltonian and a fresh S-R pair every step.
"""

from dataclasses import dataclass

import numpy as np
import qutip as qt

from .model import collision_hamiltonian, ground_chain


@dataclass
class CollisionResult:
    times: np.ndarray        # (n_stored,)
    states: np.ndarray       # (n_stored, 2^N, 2^N) chain states


def collision_unitary(params):
    """U = exp(-i H tau) on S (x) C_1..C_N (x) R, as a numpy array."""
    return _unitary(params).full()


def _unitary(params):
    n = params.N + 2
    H = qt.Qobj(collision_hamiltonian(params), dims=[[2] * n, [2] * n])
    return (-1j * H * params.tau).expm()


def run_collisions(params, rho_SR, n_steps, rho_C0=None, store_every=1):
    """Apply n_steps collisions with the S-R state rho_SR (4x4) to the chain.

    rho_C0 defaults to the all-ground chain. States are stored at step 0 and
    every `store_every` steps (and always at the last step)."""
    N = params.N
    U = _unitary(params)
    Ud = U.dag()
    rho_SR = qt.Qobj(np.asarray(rho_SR, dtype=complex), dims=[[2, 2], [2, 2]])
    rho_C = qt.Qobj(ground_chain(N) if rho_C0 is None else np.asarray(rho_C0, dtype=complex),
                    dims=[[2] * N, [2] * N])
    perm = [0] + list(range(2, N + 2)) + [1]          # [S, R, C..] -> [S, C.., R]
    keep = list(range(1, N + 1))

    times, states = [0.0], [rho_C.full()]
    for step in range(1, n_steps + 1):
        rho_tot = qt.tensor(rho_SR, rho_C).permute(perm)
        rho_C = (U * rho_tot * Ud).ptrace(keep)
        if step % store_every == 0 or step == n_steps:
            times.append(step * params.tau)
            states.append(rho_C.full())
    return CollisionResult(np.array(times), np.array(states))
