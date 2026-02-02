"""Plotting utilities for unemployment-rate distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.utils import CHANGE_LABELS, DEFAULT_BASELINE, build_dual_labels


# ------------------------------------------------------------------
# Chicago Fed loader
# ------------------------------------------------------------------
def load_chicago_fed_probs(xlsx_path: str, release: str = "advance") -> np.ndarray:
    """Load the latest 1-month forecast probabilities from Chicago Fed Excel.

    Returns a 7-element array (in percentage points, e.g. [8.3, 14.2, ...]).
    Reads Sheet "4. Real-Time UR Probs", filters to *release* type,
    picks the latest row with non-null 1-month data.
    """
    df = pd.read_excel(xlsx_path, sheet_name="4. Real-Time UR Probs", header=1)
    # Column names: date, release, <= -0.3 pp, -0.2 pp, ..., >= +0.3 pp, ...
    prob_cols = [c for c in df.columns if "pp" in str(c) and c in df.columns[:9]]
    if len(prob_cols) < 7:
        # Try positional: columns 2..8 are the 1-month forecast probs
        prob_cols = df.columns[2:9].tolist()

    # Filter to chosen release type
    if "release" in df.columns:
        mask = df["release"].astype(str).str.strip().str.lower() == release.lower()
        sub = df[mask].copy()
    else:
        sub = df.copy()

    # Drop rows where all 7 prob columns are NaN
    sub = sub.dropna(subset=prob_cols, how="all")
    if sub.empty:
        raise ValueError(f"No {release!r} rows with 1-month probs in {xlsx_path}")

    latest = sub.iloc[-1]
    probs = np.array([float(latest[c]) for c in prob_cols])
    return probs


# ------------------------------------------------------------------
# Simple bar chart
# ------------------------------------------------------------------
def plot_kalshi_only(
    probs: np.ndarray,
    categories: list[str],
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Simple bar chart of category probabilities."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    ax.bar(x, probs * 100, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Unemployment rate category")
    ax.set_ylabel("Probability (%)")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Comparison plot (matches example screenshot style)
# ------------------------------------------------------------------
def plot_comparison(
    bar_probs: np.ndarray,
    bar_label: str,
    line_probs: np.ndarray,
    line_label: str,
    categories: list[str],
    baseline: float = DEFAULT_BASELINE,
    title: str = "BLS Unemployment Rate Probabilities",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar + dashed-line comparison plot with relative-odds annotation.

    Style matches the example screenshot: blue bars with percentage labels,
    dashed black line with diamond markers, dual x-axis labels, and a
    'Relative Odds' annotation box.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))

    # Bars
    bars = ax.bar(x, bar_probs * 100, color="steelblue", zorder=2, label=bar_label)

    # Percentage labels on bars
    for bar, prob in zip(bars, bar_probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{prob * 100:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    # Dashed line
    ax.plot(x, line_probs * 100, "k--D", markersize=6, zorder=3, label=line_label)

    # Dual x-axis labels
    dual = build_dual_labels(categories, baseline)
    ax.set_xticks(x)
    ax.set_xticklabels(dual, fontsize=8)
    ax.set_ylabel("Probability (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower center", fontsize=9, framealpha=0.9)

    # Relative Odds annotation box (using bar_probs)
    decrease = float(bar_probs[:3].sum() * 100)
    increase = float(bar_probs[4:].sum() * 100)
    net = decrease - increase
    box_text = "Relative Odds"
    box_lines = [
        ("Increase", f"{increase:.0f}%", "red"),
        ("Decrease", f"{decrease:.0f}%", "green"),
        ("Net", f"{net:+.0f}", "black"),
    ]

    # Build annotation with colored text
    y_top = ax.get_ylim()[1]
    x_right = len(categories) - 0.5
    box_y = y_top * 0.98
    fontsize = 9
    line_height = 0.065 * y_top

    # Background box
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9)
    ax.text(x_right, box_y, box_text, ha="right", va="top",
            fontsize=fontsize, fontweight="bold",
            bbox=bbox_props, transform=ax.transData)

    for i, (label, val, color) in enumerate(box_lines):
        y_pos = box_y - (i + 1.3) * line_height
        ax.text(x_right - 0.6, y_pos, label, ha="left", va="top",
                fontsize=fontsize - 1, color=color)
        ax.text(x_right, y_pos, val, ha="right", va="top",
                fontsize=fontsize - 1, fontweight="bold", color=color)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Tilted comparison plot
# ------------------------------------------------------------------
def plot_tilted_comparison(
    original_probs: np.ndarray,
    tilted_probs: np.ndarray,
    model_probs: np.ndarray,
    categories: list[str],
    lam: float,
    baseline: float = DEFAULT_BASELINE,
    save_path: str | None = None,
) -> plt.Figure:
    """Three-way comparison: original Kalshi, tilted Kalshi, and model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, original_probs * 100, width, color="steelblue",
           label="Kalshi (Q)", zorder=2)
    ax.bar(x, tilted_probs * 100, width, color="darkorange",
           label=f"Kalshi tilted (\u03bb={lam:.2f})", zorder=2)
    ax.bar(x + width, model_probs * 100, width, color="seagreen",
           label="Chicago Fed model", zorder=2)

    dual = build_dual_labels(categories, baseline)
    ax.set_xticks(x)
    ax.set_xticklabels(dual, fontsize=8)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Kalshi vs Tilted vs Chicago Fed", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Combined lines plot (Kalshi vs ChiFed + CDF-shift + exp-tilt)
# ------------------------------------------------------------------
def plot_combined_lines(
    kalshi_probs: np.ndarray,
    model_probs: np.ndarray,
    categories: list[str],
    baseline: float = DEFAULT_BASELINE,
    tilted_probs: np.ndarray | None = None,
    tilt_lambda: float | None = None,
    shifted_probs: np.ndarray | None = None,
    cdf_shift_delta: float | None = None,
    implied_lam: float | None = None,
    asof: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """All-lines comparison: Kalshi, Chicago Fed model, plus optional
    CDF-shift and exponential-tilt variants.

    Style matches example/1769722079392.png:
    - Solid red line + circle markers = Kalshi
    - Solid blue line + circle markers = Chicago Fed model
    - Dashed red line + circle markers = Kalshi CDF-shifted
    - Dotted red line + circle markers = Kalshi exp-tilted
    - Dual x-axis labels, footnote, two-line title, legend upper-right
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(categories))
    dual = build_dual_labels(categories, baseline)

    # --- Kalshi (solid red) ---
    ax.plot(x, kalshi_probs * 100, "r-o", markersize=7, linewidth=2,
            label="Kalshi", zorder=4)

    # --- Chicago Fed model (solid blue) ---
    ax.plot(x, model_probs * 100, "b-o", markersize=7, linewidth=2,
            label="Chicago Fed model", zorder=4)

    # --- CDF-shifted Kalshi (dashed red) ---
    if shifted_probs is not None:
        shift_label = f"Kalshi (CDF shift {cdf_shift_delta:+.2f}pp"
        if implied_lam is not None:
            shift_label += f", \u03bb={implied_lam:.2f}"
        shift_label += ")"
        ax.plot(x, shifted_probs * 100, "r--o", markersize=5, linewidth=1.5,
                label=shift_label, zorder=3)

    # --- Exp-tilted Kalshi (dotted red) ---
    if tilted_probs is not None:
        tilt_label = f"Kalshi (exp tilt \u03bb={tilt_lambda:.2f})"
        ax.plot(x, tilted_probs * 100, "r:o", markersize=5, linewidth=1.5,
                label=tilt_label, zorder=3)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(dual, fontsize=9)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Two-line title
    title_l1 = "Unemployment Rate: Kalshi vs Chicago Fed model (lines)"
    title_l2 = f"Kalshi from full strike surface (as of {asof})" if asof else ""
    ax.set_title(f"{title_l1}\n{title_l2}", fontsize=12, fontweight="bold")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # --- Footnote ---
    footnote_lines = ["Footnote:"]
    if shifted_probs is not None and cdf_shift_delta is not None:
        delta_str = f"{abs(cdf_shift_delta):.2f}"
        fn = (f"\u2022 CDF-shift line: F\u03b4(u) = F(u + \u03b4) with"
              f" \u03b4 = {cdf_shift_delta:.2f}pp.")
        if implied_lam is not None:
            fn += (f" Implied \u03bb = (\u03bc\u03b4 \u2212 \u03bc\u2080)"
                   f"/\u03c3\u2080\u00b2 using full-strike Kalshi moments.")
        footnote_lines.append(fn)
    if tilted_probs is not None:
        footnote_lines.append(
            "\u2022 Exp-tilt line: p\u03bb(u) \u221d q(u)exp {\u03bb (u \u2212 \u03bc\u2080)}"
            " on fine-grid bins (then re-aggregated to the 7 categories)."
        )
    footnote_lines.append(
        "\u2022 \u03bc\u2080 = E_q[U], \u03c3\u2080\u00b2 = Var_q(U);"
        " units: \u03bb per percentage point."
    )
    footnote = "\n".join(footnote_lines)

    fig.subplots_adjust(bottom=0.28)
    fig.text(0.04, 0.01, footnote, fontsize=8, family="sans-serif",
             va="bottom", ha="left", linespacing=1.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Temporal comparison plot (two Kalshi snapshots)
# ------------------------------------------------------------------
def plot_temporal_comparison(
    res_old,
    res_new,
    baseline: float = DEFAULT_BASELINE,
    save_path: str | None = None,
) -> plt.Figure:
    """Compare two Kalshi distribution snapshots over time.

    Parameters
    ----------
    res_old, res_new : DistributionResult
        Earlier and later snapshots to compare.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(res_old.categories))
    dual = build_dual_labels(res_old.categories, baseline)

    date_old = res_old.asof[:10]
    date_new = res_new.asof[:10]

    ax.plot(x, res_old.category_probs * 100, "b-o", markersize=7, linewidth=2,
            label=f"Kalshi ({date_old})")
    ax.plot(x, res_new.category_probs * 100, "r-o", markersize=7, linewidth=2,
            label=f"Kalshi ({date_new})")

    # Percentage labels
    for xi, (p_old, p_new) in enumerate(
        zip(res_old.category_probs, res_new.category_probs)
    ):
        offset = 0.8
        ax.text(xi, p_old * 100 + offset, f"{p_old*100:.1f}%",
                ha="center", va="bottom", fontsize=8, color="blue")
        ax.text(xi, p_new * 100 + offset, f"{p_new*100:.1f}%",
                ha="center", va="bottom", fontsize=8, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(dual, fontsize=9)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_title(
        "Kalshi-Implied Unemployment Distribution Over Time\n"
        f"{date_old} vs {date_new}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # Summary stats box
    delta_mean_bps = (res_new.mean - res_old.mean) * 100
    box_lines = [
        f"{date_old}:  mean={res_old.mean:.3f}%"
        f"  dec={res_old.prob_decrease*100:.1f}%"
        f"  inc={res_old.prob_increase*100:.1f}%",
        f"{date_new}:  mean={res_new.mean:.3f}%"
        f"  dec={res_new.prob_decrease*100:.1f}%"
        f"  inc={res_new.prob_increase*100:.1f}%",
        f"\u0394 mean: {delta_mean_bps:+.1f} bps",
    ]
    bbox_props = dict(
        boxstyle="round,pad=0.5", facecolor="lightyellow",
        edgecolor="gray", alpha=0.9,
    )
    ax.text(0.02, 0.97, "\n".join(box_lines), transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", bbox=bbox_props, family="monospace")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig
