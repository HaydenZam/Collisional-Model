"""Single-qubit operators, conventions and small linear-algebra helpers.

Conventions (used everywhere in colmodel)
-----------------------------------------
* Basis: |0> = ground, |1> = excited.
* sigma^+ = |1><0| (raising), sigma^- = |0><1| (lowering), n = |1><1|.
* sigma^z = |1><1| - |0><0| = 2n - 1, so the excited state has sigma^z = +1.
* sigma^x = sigma^+ + sigma^-, sigma^y = i(sigma^- - sigma^+), so that
  sigma^+- = (sigma^x +- i sigma^y)/2 and [sigma^x, sigma^y] = 2i sigma^z.
  With these choices sigma^x is the usual Pauli X and sigma^y, sigma^z are
  minus the usual Pauli Y, Z.
* Multi-qubit registers are ordered left to right with np.kron: site 0 is the
  most significant (leftmost) factor.

All operators are plain numpy arrays (complex128).
"""

from functools import reduce

import numpy as np

KET0 = np.array([1, 0], dtype=complex)   # ground
KET1 = np.array([0, 1], dtype=complex)   # excited

I2 = np.eye(2, dtype=complex)
SP = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma^+ = |1><0|
SM = SP.conj().T.copy()                          # sigma^- = |0><1|
N_OP = SP @ SM                                   # n = |1><1|
SX = SP + SM
SY = 1j * (SM - SP)
SZ = SP @ SM - SM @ SP                           # |1><1| - |0><0|
PAULI = (SX, SY, SZ)


def kron(*ops):
    """Kronecker product of any number of operators (or vectors), left to right."""
    return reduce(np.kron, ops)


def embed(op, site, n_sites):
    """Single-qubit operator `op` acting on `site` of an n_sites-qubit register."""
    factors = [I2] * n_sites
    factors[site] = op
    return kron(*factors)


def proj(psi):
    """|psi><psi|."""
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


def dag(a):
    return np.asarray(a).conj().T


def expect(op, rho):
    """tr(op rho)  (complex in general)."""
    return np.trace(np.asarray(op) @ np.asarray(rho))


def ptrace(rho, keep, n_sites):
    """Partial trace of an n_sites-qubit density matrix, keeping the sites in `keep`
    (in their original order)."""
    keep = set(keep)
    t = np.asarray(rho).reshape([2] * (2 * n_sites))
    m = n_sites
    for site in reversed(range(n_sites)):
        if site not in keep:
            t = np.trace(t, axis1=site, axis2=site + m)
            m -= 1
    d = 2 ** len(keep)
    return t.reshape(d, d)


def marginals(rho):
    """(rho_A, rho_B) of a two-qubit state."""
    return ptrace(rho, [0], 2), ptrace(rho, [1], 2)


def to_array(rho):
    """Accept a numpy array or anything with .full() (e.g. a qutip Qobj)."""
    if hasattr(rho, "full"):
        rho = rho.full()
    return np.asarray(rho, dtype=complex)


def hermitian_part(a):
    a = np.asarray(a)
    return 0.5 * (a + a.conj().T)


def min_eig(a):
    """Smallest eigenvalue of the Hermitian part of a."""
    return float(np.linalg.eigvalsh(hermitian_part(a)).min())


def is_psd(a, tol=1e-10):
    return min_eig(a) >= -tol


def is_state(rho, tol=1e-10):
    """Hermitian, unit trace and positive semidefinite."""
    rho = to_array(rho)
    return (np.allclose(rho, rho.conj().T, atol=tol)
            and abs(np.trace(rho) - 1) < tol
            and is_psd(rho, tol))
