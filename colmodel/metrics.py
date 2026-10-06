"""Comparing states and time series."""

import numpy as np

from .operators import N_OP, SZ, embed


def trace_distance(a, b):
    """1/2 ||a - b||_1 for Hermitian a, b (numpy arrays, or stacks of them)."""
    diff = np.asarray(a) - np.asarray(b)
    ev = np.linalg.eigvalsh(0.5 * (diff + np.swapaxes(diff.conj(), -1, -2)))
    return 0.5 * np.sum(np.abs(ev), axis=-1)


def site_expectations(states, op, N):
    """<op_j>(t) for every site j: array (N, n_t), real part."""
    states = np.asarray(states)
    return np.array([np.einsum("ij,tji->t", embed(op, j, N), states).real for j in range(N)])


def populations(states, N):
    """<n_j>(t), shape (N, n_t)."""
    return site_expectations(states, N_OP, N)


def sz_values(states, N):
    """<sigma^z_j>(t), shape (N, n_t)  (excited = +1)."""
    return site_expectations(states, SZ, N)


def error_report(exact, approx, name="", verbose=True):
    """MSE, RMSE, max error and both relative to the range of `exact`."""
    exact, approx = np.asarray(exact), np.asarray(approx)
    diff = exact - approx
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(np.abs(diff)))
    span = float(np.max(exact) - np.min(exact))
    rel = (100 * rmse / span, 100 * max_err / span) if span > 1e-12 else (np.nan, np.nan)
    out = {"mse": mse, "rmse": rmse, "max_err": max_err,
           "rmse_percent": rel[0], "max_percent": rel[1]}
    if verbose:
        print(f"{name}: RMSE = {rmse:.3e} ({rel[0]:.3g} % of range), "
              f"max = {max_err:.3e} ({rel[1]:.3g} %)")
    return out
