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
def load_chicago_fed_probs(xlsx_path: str, release: str = "final") -> np.ndarray:
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
# Temporal comparison plot (multiple Kalshi snapshots)
# ------------------------------------------------------------------
def plot_temporal_comparison(
    results: list,
    baseline: float = DEFAULT_BASELINE,
    save_path: str | None = None,
) -> plt.Figure:
    """Compare multiple Kalshi distribution snapshots over time.

    Parameters
    ----------
    results : list of DistributionResult
        Snapshots to compare (oldest to newest).
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 results for temporal comparison")

    # Color palette for multiple series
    colors = ["blue", "red", "green", "purple", "orange", "brown"]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(results[0].categories))
    dual = build_dual_labels(results[0].categories, baseline)

    dates = [res.asof[:10] for res in results]

    # Plot each series
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        ax.plot(x, res.category_probs * 100, f"-o", color=color,
                markersize=7, linewidth=2, label=f"Kalshi ({dates[i]})")

    # Percentage labels for each series
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        for xi, prob in enumerate(res.category_probs):
            offset = 0.8 + i * 0.6  # Stagger labels vertically
            ax.text(xi, prob * 100 + offset, f"{prob*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(dual, fontsize=9)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Build title from dates
    date_str = " vs ".join(dates)
    ax.set_title(
        "Kalshi-Implied Unemployment Distribution Over Time\n"
        f"{date_str}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Bloomberg consensus histogram
# ------------------------------------------------------------------
def load_bbg_consensus(xlsx_path: str) -> pd.DataFrame:
    """Load Bloomberg consensus forecast data from Excel.

    Returns DataFrame with columns: forecast, count, share
    """
    return pd.read_excel(xlsx_path, sheet_name="Sheet1")


def plot_bbg_consensus_histogram(
    xlsx_path: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of Bloomberg analyst unemployment rate forecasts.

    Parameters
    ----------
    xlsx_path : str
        Path to the Bloomberg consensus forecast Excel file.

    Returns
    -------
    plt.Figure
    """
    df = load_bbg_consensus(xlsx_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    bars = ax.bar(x, df["count"], color="steelblue", edgecolor="navy", linewidth=1.2)

    # Count labels on bars
    for bar, count in zip(bars, df["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(count)),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # X-axis labels
    labels = [f"{rate:.1f}%" for rate in df["forecast"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_xlabel("Unemployment Rate Forecast", fontsize=11)
    ax.set_ylabel("Number of Forecasters", fontsize=11)
    ax.set_title("Bloomberg Consensus Unemployment Rate Forecasts", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# BBG consensus to 7-category converter
# ------------------------------------------------------------------
def bbg_to_7cat(bbg_df: pd.DataFrame, categories: list[str]) -> np.ndarray:
    """Convert BBG consensus (3 rates) to 7-category probability array.

    Maps BBG shares to matching category indices, fills others with 0.

    Parameters
    ----------
    bbg_df : DataFrame with columns 'forecast' and 'share'
    categories : list of 7 category labels (e.g., ['≤4.1%', '4.2%', ...])

    Returns
    -------
    np.ndarray of shape (7,) with BBG shares at matching positions
    """
    result = np.zeros(len(categories))

    for _, row in bbg_df.iterrows():
        rate = row["forecast"]
        share = row["share"]
        # Find matching category by rate value
        target_label = f"{rate:.1f}%"
        for i, cat in enumerate(categories):
            if target_label in cat:
                result[i] = share
                break

    return result


# ------------------------------------------------------------------
# Combined Kalshi + Chicago Fed + BBG consensus chart
# ------------------------------------------------------------------
def plot_kalshi_chifed_bbg(
    kalshi_probs: np.ndarray,
    model_probs: np.ndarray,
    bbg_probs: np.ndarray,
    categories: list[str],
    baseline: float = DEFAULT_BASELINE,
    asof: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Combined chart: Kalshi and Chicago Fed lines with BBG consensus bars.

    Parameters
    ----------
    kalshi_probs : 7-element array of Kalshi probabilities
    model_probs : 7-element array of Chicago Fed model probabilities
    bbg_probs : 7-element array of BBG consensus probabilities (0s where missing)
    categories : list of category labels
    baseline : reference unemployment rate
    asof : timestamp string for title
    save_path : optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(categories))
    dual = build_dual_labels(categories, baseline)

    # --- BBG consensus bars (semi-transparent, behind lines) ---
    # Only plot bars where BBG has data (non-zero)
    bar_mask = bbg_probs > 0
    ax.bar(x[bar_mask], bbg_probs[bar_mask] * 100, color="green", alpha=0.4,
           width=0.6, label="BBG Consensus", zorder=2, edgecolor="darkgreen")

    # --- Kalshi (solid red line) ---
    ax.plot(x, kalshi_probs * 100, "r-o", markersize=8, linewidth=2.5,
            label="Kalshi", zorder=4)

    # --- Chicago Fed model (solid blue line) ---
    ax.plot(x, model_probs * 100, "b-o", markersize=8, linewidth=2.5,
            label="Chicago Fed model", zorder=4)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(dual, fontsize=10)
    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Title
    date_str = asof[:10] if asof else ""
    ax.set_title(
        f"January 2026 Unemployment Rate: Kalshi vs Chicago Fed model (lines)\n"
        f"with Bloomberg Consensus forecasts (bars) — as of {date_str}",
        fontsize=13, fontweight="bold",
    )

    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig
