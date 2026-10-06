"""Correlation measures for two-qubit states (numpy arrays, basis of colmodel.operators)."""

import numpy as np

from .operators import I2, PAULI, SX, SY, SZ, kron, marginals, ptrace, to_array


def von_neumann_entropy(rho):
    """S(rho) in bits."""
    ev = np.linalg.eigvalsh(to_array(rho)).real
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


def mutual_information(rho):
    """I(A:B) = S(A) + S(B) - S(AB), in bits."""
    rho = to_array(rho)
    rA, rB = marginals(rho)
    return von_neumann_entropy(rA) + von_neumann_entropy(rB) - von_neumann_entropy(rho)


def negativity(rho):
    """Sum of |negative eigenvalues| of the partial transpose on B.
    For two qubits N > 0 <=> entangled (Peres-Horodecki); Bell state: 1/2."""
    r = to_array(rho).reshape(2, 2, 2, 2)
    rt = r.transpose(0, 3, 2, 1).reshape(4, 4)
    ev = np.linalg.eigvalsh(rt)
    return float(np.sum(np.abs(ev[ev < 0])))


def concurrence(rho):
    """Wootters concurrence; Bell state: 1, separable: 0."""
    rho = to_array(rho)
    yy = kron(SY, SY)
    rho_tilde = yy @ rho.conj() @ yy
    lam = np.sqrt(np.clip(np.sort(np.linalg.eigvals(rho @ rho_tilde).real)[::-1], 0, None))
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def local_coherence(rho):
    """max |<sx>|, |<sy>| over both marginals; 0 <=> both marginals z-diagonal."""
    rho = to_array(rho)
    vals = []
    for P in (SX, SY):
        vals.append(abs(np.trace(rho @ kron(P, I2))))
        vals.append(abs(np.trace(rho @ kron(I2, P))))
    return float(max(vals))


def zz_covariance(rho):
    """Connected correlator <sz sz> - <sz_A><sz_B>."""
    rho = to_array(rho)
    ev = lambda O: np.trace(rho @ O).real
    return ev(kron(SZ, SZ)) - ev(kron(SZ, I2)) * ev(kron(I2, SZ))


def discord(rho, n_grid=48):
    """Quantum discord D_B(rho) in bits, measuring subsystem B.

    D_B = I(rho) - J_B(rho), with J_B = S(A) - min over projective measurements
    {|n><n|} on B of sum_k p_k S(rho_{A|k}). The minimum is found by a grid over
    the Bloch sphere followed by 8 rounds of local refinement. D_B = 0 <=> zero
    discord with respect to B. Discord is asymmetric."""
    rho = to_array(rho)
    S_AB = von_neumann_entropy(rho)
    rA, rB = marginals(rho)
    S_A = von_neumann_entropy(rA)
    I_mut = S_A + von_neumann_entropy(rB) - S_AB

    def conditional(th, ph):
        n = (np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))
        cond = 0.0
        for s in (1, -1):
            Pi = 0.5 * (I2 + s * (n[0] * PAULI[0] + n[1] * PAULI[1] + n[2] * PAULI[2]))
            M = kron(I2, Pi) @ rho @ kron(I2, Pi)
            pk = np.trace(M).real
            if pk > 1e-12:
                cond += pk * von_neumann_entropy(ptrace(M, [0], 2) / pk)
        return cond

    best, best_ang = np.inf, (0.0, 0.0)
    for th in np.linspace(0, np.pi, n_grid):
        for ph in np.linspace(0, 2 * np.pi, 2 * n_grid, endpoint=False):
            c = conditional(th, ph)
            if c < best:
                best, best_ang = c, (th, ph)
    span_th, span_ph = np.pi / n_grid, 2 * np.pi / (2 * n_grid)
    for _ in range(8):
        th0, ph0 = best_ang
        for th in np.linspace(th0 - span_th, th0 + span_th, 7):
            for ph in np.linspace(ph0 - span_ph, ph0 + span_ph, 7):
                c = conditional(th, ph)
                if c < best:
                    best, best_ang = c, (th, ph)
        span_th *= 0.4
        span_ph *= 0.4
    return float(max(0.0, I_mut - (S_A - best)))


def classify(rho, disc_grid=20, discord_value=None):
    """Coarse correlation class: "entangled", "discord", "classical" or "product".

    Pass a precomputed `discord_value` to skip the discord computation."""
    rho = to_array(rho)
    if negativity(rho) > 1e-7:
        return "entangled"
    d = discord(rho, n_grid=disc_grid) if discord_value is None else discord_value
    if d > 1e-4:
        return "discord"
    rA, rB = marginals(rho)
    if np.linalg.norm(rho - np.kron(rA, rB)) > 1e-7:
        return "classical"
    return "product"
