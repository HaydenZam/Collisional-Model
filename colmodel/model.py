"""Model parameters and Hamiltonians (numpy).

The channel is an N-site XX chain C_1..C_N. A sender ancilla S collides with
site 1 and a receiver ancilla R with site N, simultaneously, for a time tau,
after which the S-R pair is replaced by a fresh copy (in the same joint state).

Collision-space ordering: S (x) C_1 (x) ... (x) C_N (x) R  (indices 0, 1..N, N+1).
Chain-space ordering:     C_1 (x) ... (x) C_N            (indices 0..N-1).

Weak-collision scaling: the couplings during a collision are g/sqrt(tau), so
second-order rates are g^2 and stay finite as tau -> 0 (see master_equation).
"""

from dataclasses import dataclass, replace

import numpy as np

from .operators import N_OP, SM, SP, embed


@dataclass(frozen=True)
class Params:
    """N: chain length.  J: hopping (scalar or N-1 values).  omega: site fields
    (scalar or N values).  omega_S, omega_R: ancilla fields.  gL, gR: effective
    couplings (finite as tau -> 0).  tau: collision time."""
    N: int = 2
    J: float | tuple = 0.3
    omega: float | tuple = 0.0
    omega_S: float = 0.0
    omega_R: float = 0.0
    gL: float = 0.2
    gR: float = 0.2
    tau: float = 1e-3

    def __post_init__(self):
        if self.N < 1:
            raise ValueError("N must be >= 1")
        for name, n in (("J", self.N - 1), ("omega", self.N)):
            v = getattr(self, name)
            if np.ndim(v) and len(v) != n:
                raise ValueError(f"{name} needs {n} values, got {len(v)}")
            if np.ndim(v):
                object.__setattr__(self, name, tuple(float(x) for x in v))

    @property
    def Js(self):
        return np.broadcast_to(np.asarray(self.J, dtype=float), (self.N - 1,)).copy()

    @property
    def omegas(self):
        return np.broadcast_to(np.asarray(self.omega, dtype=float), (self.N,)).copy()

    @property
    def gL_col(self):
        """Coupling during a collision, gL / sqrt(tau)."""
        return self.gL / np.sqrt(self.tau)

    @property
    def gR_col(self):
        return self.gR / np.sqrt(self.tau)

    def with_(self, **changes):
        """Copy with some fields changed, e.g. params.with_(tau=5e-4)."""
        return replace(self, **changes)


def _chain_terms(params, op, offset, n_sites):
    """Local fields and XX hopping on sites offset..offset+N-1 of an n_sites register."""
    N = params.N
    H = np.zeros((2 ** n_sites, 2 ** n_sites), dtype=complex)
    for j, w in enumerate(params.omegas):
        H += w * op(N_OP, offset + j)
    for j, J in enumerate(params.Js):
        H += J * (op(SP, offset + j) @ op(SM, offset + j + 1)
                  + op(SM, offset + j) @ op(SP, offset + j + 1))
    return H


def chain_hamiltonian(params):
    """H_C = sum_j omega_j n_j + sum_j J_j (s+_j s-_{j+1} + h.c.) on the chain (2^N)."""
    N = params.N
    return _chain_terms(params, lambda o, s: embed(o, s, N), 0, N)


def collision_hamiltonian(params):
    """Full Hamiltonian during one collision on S (x) C_1..C_N (x) R (2^(N+2)):

        H = H_C + omega_S n_S + omega_R n_R
            + gL/sqrt(tau) (s-_S s+_1 + s+_S s-_1) + gR/sqrt(tau) (s-_N s+_R + s+_N s-_R)."""
    N = params.N
    n = N + 2
    op = lambda o, s: embed(o, s, n)
    S, R = 0, N + 1
    H = _chain_terms(params, op, 1, n)
    H += params.omega_S * op(N_OP, S) + params.omega_R * op(N_OP, R)
    H += params.gL_col * (op(SM, S) @ op(SP, 1) + op(SP, S) @ op(SM, 1))
    H += params.gR_col * (op(SM, N) @ op(SP, R) + op(SP, N) @ op(SM, R))
    return H


def ground_chain(N):
    """All chain sites in |0>."""
    rho = np.zeros((2 ** N, 2 ** N), dtype=complex)
    rho[0, 0] = 1.0
    return rho
