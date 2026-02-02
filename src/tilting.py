"""Exponential tilting and CDF-shift utilities for risk adjustment."""

from __future__ import annotations

import numpy as np


def compute_mean(bin_probs: np.ndarray, bin_midpoints: np.ndarray) -> float:
    return float(np.average(bin_midpoints, weights=bin_probs))


def compute_variance(bin_probs: np.ndarray, bin_midpoints: np.ndarray) -> float:
    mu = compute_mean(bin_probs, bin_midpoints)
    return float(np.average((bin_midpoints - mu) ** 2, weights=bin_probs))


def tilt_exponential(
    bin_probs: np.ndarray,
    bin_midpoints: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Apply exponential tilt: q_lambda(u) proportional to q(u)*exp(lambda*(u - mu_0)).

    Parameters
    ----------
    bin_probs : array of shape (n,)
        Baseline PMF (sums to 1).
    bin_midpoints : array of shape (n,)
        Representative unemployment-rate value for each bin.
    lam : float
        Tilt parameter (per percentage point).  Negative shifts left.

    Returns
    -------
    Tilted PMF (sums to 1).
    """
    mu_0 = compute_mean(bin_probs, bin_midpoints)
    log_w = lam * (bin_midpoints - mu_0)
    log_w -= log_w.max()  # numerical stability
    w = np.exp(log_w)
    tilted = bin_probs * w
    return tilted / tilted.sum()


def shift_cdf(
    bin_probs: np.ndarray,
    thresholds: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Shift distribution left by *delta* percentage points via CDF interpolation.

    Constructs a piecewise-uniform CDF from the fine-grid PMF, evaluates
    F(u + delta) at original thresholds, and differences to recover shifted
    bin masses.

    Parameters
    ----------
    bin_probs : array of shape (n_bins,)
        Fine-grid PMF.  n_bins = len(thresholds) + 1.
    thresholds : array of shape (n_thresh,)
        Sorted strike thresholds.
    delta : float
        Shift in percentage points.  Positive delta shifts left (lower U).

    Returns
    -------
    Shifted PMF on the same bin grid (sums to 1).
    """
    step = float(thresholds[1] - thresholds[0]) if len(thresholds) > 1 else 0.1
    n = len(thresholds)

    # Build CDF breakpoints: each bin is uniform over its interval
    # Bin 0: [t0-step, t0], Bin i: [t_{i-1}, t_i], Bin n: [t_{n-1}, t_{n-1}+step]
    edges = np.empty(n + 2)
    edges[0] = thresholds[0] - step
    edges[1 : n + 1] = thresholds
    edges[n + 1] = thresholds[-1] + step

    # CDF at edges: cumulative sum of bin_probs
    cdf_edges = np.zeros(n + 2)
    for i in range(len(bin_probs)):
        cdf_edges[i + 1] = cdf_edges[i] + bin_probs[i]
    cdf_edges[-1] = 1.0  # ensure exact

    # Evaluate shifted CDF: F_delta(u) = F(u + delta)
    shifted_edges = edges + delta

    # Interpolate original CDF at shifted edge positions
    cdf_shifted = np.interp(edges, shifted_edges, cdf_edges, left=0.0, right=1.0)

    # Recover bin masses by differencing
    shifted_probs = np.diff(cdf_shifted)
    shifted_probs = np.maximum(shifted_probs, 0.0)
    total = shifted_probs.sum()
    if total > 0:
        shifted_probs /= total
    return shifted_probs


def implied_lambda(
    bin_probs: np.ndarray,
    bin_midpoints: np.ndarray,
    target_mean: float,
) -> float:
    """Moment-based tilt parameter: lambda = (mu_target - mu_0) / sigma_0^2."""
    mu_0 = compute_mean(bin_probs, bin_midpoints)
    var_0 = compute_variance(bin_probs, bin_midpoints)
    if var_0 < 1e-12:
        return 0.0
    return (target_mean - mu_0) / var_0
