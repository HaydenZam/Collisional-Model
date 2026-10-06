"""Minimal two-qubit model for the per-bath unravelling question.

Two qubits, each nominally coupled to its own bath through one channel,
F_1 = s- (x) I ("bath A") and F_2 = I (x) s- ("bath B"), no Hamiltonian, and

    c = gamma [[1, m], [m, 1]],   m = overlap of the two couplings, |m| <= 1.

m = 0: two independent reservoirs, and per-bath detectors are a valid
unravelling. m != 0: the off-diagonal part is not a correlation of two bath
states but an overlap of couplings into one shared vacuum (m = 1 is Dicke
superradiance at T = 0); microscopic_check() builds this explicitly. There is
then one bath, A and B label system operators, and per-bath monitoring fails.
"""

import numpy as np
from scipy.linalg import expm

from .master_equation import superoperator, unvec, vec
from .operators import I2, N_OP, SM, kron

DIM = 4
F = [kron(SM, I2), kron(I2, SM)]
BATHS = {"A": (0,), "B": (1,)}
N_TOT = kron(N_OP, I2) + kron(I2, N_OP)          # total excitation number
H = np.zeros((DIM, DIM), dtype=complex)           # no Hamiltonian: isolate dissipation


def coupling_matrix(gamma=1.0, m=0.0):
    """c = gamma [[1, m], [m, 1]]."""
    return gamma * np.array([[1.0, m], [m, 1.0]], dtype=complex)


def liouvillian(c):
    return superoperator(H, F, c)


def per_bath_jumps(c):
    """Forced per-bath detectors L_A = sqrt(c_AA) F_1, L_B = sqrt(c_BB) F_2."""
    return [np.sqrt(np.real(c[j, j])) * F[j] for j in range(len(F))]


def microscopic_check(m, gamma=1.0, dt=1e-6, rng=None):
    """Collision model with VACUUM ancillas whose couplings overlap by m.

    Each system qubit j couples to its own combination B_j = sum_mu G_j,mu b_mu of
    two shared ancilla modes (hard-core, in the vacuum):
        K = sqrt(gamma) sum_j ( F_j (x) B_j^dag + h.c. ),  U = exp(-i sqrt(dt) K).
    Then <vac| B_j B_k^dag |vac> = (G G^dag)_jk, so the generator has
    c = gamma G G^dag, a Gram matrix (automatically >= 0) with overlap m.
    Returns (c, max |(rho' - rho)/dt - L(rho)|) for a random rho."""
    rng = np.random.default_rng() if rng is None else rng
    DA = 4                                                   # two ancilla modes
    b = [kron(SM, I2), kron(I2, SM)]                         # mode annihilation operators
    G = np.array([[1.0, 0.0], [m, np.sqrt(max(0.0, 1 - m ** 2))]], dtype=complex)
    Bd = [sum(np.conj(G[j, mu]) * b[mu].conj().T for mu in range(2)) for j in range(2)]
    K = np.sqrt(gamma) * sum(kron(F[j], Bd[j]) + kron(F[j].conj().T, Bd[j].conj().T)
                             for j in range(2))
    U = expm(-1j * np.sqrt(dt) * K)

    vac = np.zeros(DA, dtype=complex)
    vac[0] = 1.0
    A = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM))
    rho = A @ A.conj().T
    rho /= np.trace(rho)

    tot = U @ kron(rho, np.outer(vac, vac.conj())) @ U.conj().T
    rho2 = np.einsum("iaja->ij", tot.reshape(DIM, DA, DIM, DA))

    c = gamma * (G @ G.conj().T)
    drho = unvec(liouvillian(c) @ vec(rho), DIM)
    return c, float(np.max(np.abs((rho2 - rho) / dt - drho)))
