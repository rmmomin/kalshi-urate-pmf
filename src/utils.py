"""Shared utilities for unemployment-rate distribution analysis."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASELINE = 4.4
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

CHANGE_LABELS = [
    "\u2264 \u20130.3 pp",
    "\u20130.2 pp",
    "\u20130.1 pp",
    "No change",
    "+0.1 pp",
    "+0.2 pp",
    "\u2265 +0.3 pp",
]


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------
def enforce_monotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Clip upward jumps so exceedance probabilities are non-increasing."""
    out = values.copy()
    for i in range(len(out) - 1):
        if out[i + 1] > out[i]:
            out[i + 1] = out[i]
    return out


# ---------------------------------------------------------------------------
# Category aggregation
# ---------------------------------------------------------------------------
def aggregate_to_categories(
    thresholds: np.ndarray,
    bin_probs: np.ndarray,
    baseline: float = DEFAULT_BASELINE,
) -> tuple[list[str], np.ndarray]:
    """Aggregate fine-grid bins into 7 display categories centred on *baseline*.

    Thresholds are sorted ascending.  Bins are:
      bin 0  : U <= t_0
      bin i  : (t_{i-1}, t_i]   for i = 1..n-1
      bin n  : U > t_{n-1}

    Categories (7):
      <=baseline-0.3, baseline-0.2, baseline-0.1, baseline,
      baseline+0.1, baseline+0.2, >=baseline+0.3

    Returns (category_labels, category_probs).
    """
    step = 0.1
    # Boundaries that define the 7 categories
    #   cat 0  : U <= baseline - 0.3  (i.e. <= baseline - 3*step)
    #   cat 1  : (baseline - 0.3, baseline - 0.2]
    #   ...
    #   cat 6  : U > baseline + 0.2   (i.e. >= baseline + 0.3)
    cat_boundaries = [round(baseline + (k - 3) * step, 1) for k in range(7)]
    # cat_boundaries = [bl-0.3, bl-0.2, bl-0.1, bl, bl+0.1, bl+0.2, bl+0.3]

    labels = [
        f"\u2264{cat_boundaries[0]:.1f}%",
        f"{cat_boundaries[1]:.1f}%",
        f"{cat_boundaries[2]:.1f}%",
        f"{cat_boundaries[3]:.1f}%",
        f"{cat_boundaries[4]:.1f}%",
        f"{cat_boundaries[5]:.1f}%",
        f"\u2265{cat_boundaries[6]:.1f}%",
    ]

    # Map each fine-grid bin to one of the 7 categories.
    # Bin upper boundaries:
    #   bin 0  upper = t_0
    #   bin i  upper = t_i   for i=1..n-1
    #   bin n  upper = +inf
    n = len(thresholds)
    bin_uppers = list(thresholds) + [np.inf]
    # bin_lowers:
    #   bin 0  lower = -inf
    #   bin i  lower = t_{i-1}
    bin_lowers = [-np.inf] + list(thresholds)

    cat_probs = np.zeros(7)
    for b_idx in range(len(bin_probs)):
        upper = bin_uppers[b_idx]
        # Which category does this bin belong to?
        if upper <= cat_boundaries[0] + 1e-9:
            cat_probs[0] += bin_probs[b_idx]
        elif upper <= cat_boundaries[1] + 1e-9:
            cat_probs[1] += bin_probs[b_idx]
        elif upper <= cat_boundaries[2] + 1e-9:
            cat_probs[2] += bin_probs[b_idx]
        elif upper <= cat_boundaries[3] + 1e-9:
            cat_probs[3] += bin_probs[b_idx]
        elif upper <= cat_boundaries[4] + 1e-9:
            cat_probs[4] += bin_probs[b_idx]
        elif upper <= cat_boundaries[5] + 1e-9:
            cat_probs[5] += bin_probs[b_idx]
        else:
            cat_probs[6] += bin_probs[b_idx]

    return labels, cat_probs


# ---------------------------------------------------------------------------
# Dual x-axis labels
# ---------------------------------------------------------------------------
def build_dual_labels(categories: list[str], baseline: float = DEFAULT_BASELINE) -> list[str]:
    """Build x-axis labels with both change (pp) and level (%).

    E.g. '\u2264 \u20130.3 pp\\n(4.1%)' for the first category when baseline=4.4.
    """
    return [f"{ch}\n({cat})" for ch, cat in zip(CHANGE_LABELS, categories)]


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------
def compute_moments(bin_probs: np.ndarray, bin_midpoints: np.ndarray) -> dict:
    """Return mean, variance, std of a discrete distribution."""
    mean = float(np.average(bin_midpoints, weights=bin_probs))
    var = float(np.average((bin_midpoints - mean) ** 2, weights=bin_probs))
    return {"mean": mean, "variance": var, "std": var ** 0.5}


def bin_midpoints_from_thresholds(thresholds: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Compute representative midpoint for each bin.

    Bins:
      bin 0  : (-inf, t_0]     → midpoint = t_0 - step/2
      bin i  : (t_{i-1}, t_i]  → midpoint = (t_{i-1}+t_i)/2
      bin n  : (t_{n-1}, inf)  → midpoint = t_{n-1} + step/2
    """
    mids = [thresholds[0] - step / 2]
    for i in range(1, len(thresholds)):
        mids.append((thresholds[i - 1] + thresholds[i]) / 2)
    mids.append(thresholds[-1] + step / 2)
    return np.array(mids)
