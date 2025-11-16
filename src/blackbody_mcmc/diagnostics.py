"""Convergence diagnostics for MCMC chains"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def autocorrelation(chain: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Estimates autocorrelation function for a 1D chain
    Parameters
    ----------
    chain : ndarray
        1D array of samples
    max_lag : int, optional
        Maximum lag to compute
        Defaults to len(chain) // 2
    Returns
    -------
    ndarray
        Autocorrelation for lags 0..max_lag-1
    """
    x = np.asarray(chain, dtype=float)
    x = x - x.mean()
    n = x.size

    if max_lag is None:
        max_lag = n // 2

    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[: max_lag]
    acf /= acf[0]
    return acf


def gelman_rubin(chains: Iterable[np.ndarray]) -> np.ndarray:
    """
    Computes Gelman-Rubin R-hat statistic for multiple chains
    Parameters
    ----------
    chains : iterable of ndarray
        Each chain is (n_samples, n_params)
    Returns
    -------
    ndarray
        R-hat values for each parameter
    """
    chains = [np.asarray(c) for c in chains]
    m = len(chains)
    if m < 2:
        raise ValueError("Need at least two chains for Gelman-Rubin.")

    n, n_params = chains[0].shape
    for c in chains:
        if c.shape != (n, n_params):
            raise ValueError("All chains must have same shape")

    chain_means = np.array([c.mean(axis=0) for c in chains])  # (m, p)
    chain_vars = np.array([c.var(axis=0, ddof=1) for c in chains])  # (m, p)

    B = n * chain_means.var(axis=0, ddof=1)
    W = chain_vars.mean(axis=0)

    var_hat = (n - 1) / n * W + B / n
    R_hat = np.sqrt(var_hat / W)
    return R_hat




