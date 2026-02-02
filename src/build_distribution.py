"""Build a Kalshi-implied unemployment-rate distribution from strike-level prices.

Implements the model-free approach from Diercks, Katz & Wright (NBER WP 34702):
  strike prices  →  exceedance probs  →  monotonicity  →  bin PMF  →  categories

Can ingest data from the live Kalshi API (--market-id) or a cached CSV (--input).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    DEFAULT_BASELINE,
    aggregate_to_categories,
    bin_midpoints_from_thresholds,
    compute_moments,
    enforce_monotonic_nonincreasing,
)


@dataclass
class DistributionResult:
    asof: str
    thresholds: np.ndarray
    p_above: np.ndarray
    bin_labels: list[str]
    bin_probs: np.ndarray
    bin_midpoints: np.ndarray
    categories: list[str]
    category_probs: np.ndarray
    mean: float
    variance: float
    prob_increase: float
    prob_decrease: float
    prob_no_change: float


# ------------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------------
def build_distribution(
    strikes: np.ndarray,
    prices_cents: np.ndarray,
    baseline: float = DEFAULT_BASELINE,
    timestamp: str | None = None,
) -> DistributionResult:
    """Convert strike prices (in cents) to a full distribution result.

    Parameters
    ----------
    strikes : sorted ascending array of strike thresholds (e.g. [3.8, 3.9, ..., 4.7])
    prices_cents : corresponding YES prices in cents (0-100)
    baseline : reference unemployment rate for category labels (default 4.4)
    timestamp : optional ISO-8601 timestamp string
    """
    # Exceedance probabilities
    p_above = np.array(prices_cents, dtype=float) / 100.0
    p_above = enforce_monotonic_nonincreasing(p_above)

    # Fine-grid bin probs
    thresholds = np.array(strikes, dtype=float)
    bin_labels = [f"\u2264{thresholds[0]:.1f}%"]
    bin_probs = [max(0.0, 1.0 - p_above[0])]
    for i in range(1, len(thresholds)):
        bin_labels.append(f"({thresholds[i-1]:.1f}%, {thresholds[i]:.1f}%]")
        bin_probs.append(max(0.0, p_above[i - 1] - p_above[i]))
    bin_labels.append(f">{thresholds[-1]:.1f}%")
    bin_probs.append(max(0.0, p_above[-1]))
    bin_probs = np.array(bin_probs)

    # Midpoints
    bin_mids = bin_midpoints_from_thresholds(thresholds)

    # Moments
    moments = compute_moments(bin_probs, bin_mids)

    # 7-category aggregation
    categories, category_probs = aggregate_to_categories(thresholds, bin_probs, baseline)

    # Directional odds: first 3 = decrease, middle = no-change, last 3 = increase
    prob_decrease = float(category_probs[:3].sum())
    prob_no_change = float(category_probs[3])
    prob_increase = float(category_probs[4:].sum())

    asof = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return DistributionResult(
        asof=asof,
        thresholds=thresholds,
        p_above=p_above,
        bin_labels=bin_labels,
        bin_probs=bin_probs,
        bin_midpoints=bin_mids,
        categories=categories,
        category_probs=category_probs,
        mean=moments["mean"],
        variance=moments["variance"],
        prob_increase=prob_increase,
        prob_decrease=prob_decrease,
        prob_no_change=prob_no_change,
    )


# ------------------------------------------------------------------
# Input loaders
# ------------------------------------------------------------------
def load_from_csv(path: str) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load a Kalshi strike-history CSV (wide format with 'Above X%' columns).

    Returns (strikes, prices_cents_latest_row, timestamp).
    """
    df = pd.read_csv(path).ffill()
    strike_cols = [c for c in df.columns if c.lower().startswith("above")]
    if not strike_cols:
        raise ValueError("No 'Above X%' columns found in CSV.")

    thresholds = np.array([float(c.split()[1].rstrip("%")) for c in strike_cols])
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    strike_cols = [strike_cols[i] for i in order]

    latest = df.iloc[-1]
    prices = np.array([float(latest[c]) for c in strike_cols])
    ts = str(latest.get("timestamp", "")) or None

    return thresholds, prices, ts


def load_from_api(event_ticker: str, raw_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Pull live strike surface from Kalshi API.

    Returns (strikes, prices_cents, asof_timestamp).
    """
    from src.kalshi_api import KalshiClient

    client = KalshiClient()
    asof_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_path = None
    if raw_dir:
        save_path = str(Path(raw_dir) / f"kalshi_strikes_{asof_tag}.csv")

    df = client.get_strike_surface(event_ticker, save_path=save_path)
    strikes = df["strike"].values
    prices = df["yes_price"].values  # already in cents
    asof = df["asof_timestamp"].iloc[0]
    return strikes, prices, asof


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------
def export_results(res: DistributionResult, outdir: str) -> dict[str, str]:
    """Write fine-grid and category CSVs. Returns dict of paths written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    asof_tag = res.asof.replace(":", "").replace("-", "")[:15]
    paths = {}

    # Strike-level exceedance
    strike_df = pd.DataFrame({
        "asof_timestamp_utc": res.asof,
        "strike": res.thresholds,
        "kalshi_yes_price": (res.p_above * 100).round(4),
        "p_outcome_above_strike": res.p_above,
    })
    p = outdir / f"kalshi_finegrid_strikes_{asof_tag}.csv"
    strike_df.to_csv(p, index=False)
    paths["strikes"] = str(p)

    # Fine-grid bins
    bins_df = pd.DataFrame({
        "asof_timestamp_utc": res.asof,
        "bin": res.bin_labels,
        "probability": res.bin_probs,
        "probability_pct": (res.bin_probs * 100).round(4),
    })
    p = outdir / f"kalshi_finegrid_bins_{asof_tag}.csv"
    bins_df.to_csv(p, index=False)
    paths["bins"] = str(p)

    # Categories
    cat_df = pd.DataFrame({
        "asof_timestamp_utc": res.asof,
        "category": res.categories,
        "probability": res.category_probs,
        "probability_pct": (res.category_probs * 100).round(4),
    })
    p = outdir / f"kalshi_categories_{asof_tag}.csv"
    cat_df.to_csv(p, index=False)
    paths["categories"] = str(p)

    return paths


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build Kalshi-implied unemployment distribution.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", help="Path to Kalshi strike-level price history CSV")
    grp.add_argument("--market-id", help="Kalshi event ticker for live API pull (e.g. UNRATE-26JAN)")
    ap.add_argument("--baseline", type=float, default=DEFAULT_BASELINE,
                     help="Reference unemployment rate (default 4.4)")
    ap.add_argument("--chicago-fed", help="Path to Chicago Fed labor-market-indicators Excel file")
    ap.add_argument("--tilt-lambda", type=float, default=None,
                     help="Exponential tilt parameter (negative = shift left)")
    ap.add_argument("--cdf-shift", type=float, default=None,
                     help="CDF shift in percentage points (positive = shift left)")
    ap.add_argument("--compare-csv", help="Path to an earlier Kalshi CSV to plot a temporal comparison")
    ap.add_argument("--outdir", default="data/processed", help="Output directory")
    ap.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = ap.parse_args()

    # Load strikes
    if args.input:
        strikes, prices, ts = load_from_csv(args.input)
    else:
        strikes, prices, ts = load_from_api(args.market_id, raw_dir="data/raw")

    # Build distribution
    res = build_distribution(strikes, prices, baseline=args.baseline, timestamp=ts)

    # Export CSVs
    paths = export_results(res, args.outdir)
    print(f"As of: {res.asof}")
    print(f"Mean: {res.mean:.3f}%")
    print(f"Decrease: {res.prob_decrease * 100:.1f}%")
    print(f"No change: {res.prob_no_change * 100:.1f}%")
    print(f"Increase: {res.prob_increase * 100:.1f}%")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    # Optional tilting
    tilted_cat_probs = None
    if args.tilt_lambda is not None:
        from src.tilting import tilt_exponential
        tilted_bins = tilt_exponential(res.bin_probs, res.bin_midpoints, args.tilt_lambda)
        _, tilted_cat_probs = aggregate_to_categories(res.thresholds, tilted_bins, args.baseline)
        # Export tilted categories
        lam_tag = f"{args.tilt_lambda:.2f}".replace(".", "p").replace("-", "neg")
        asof_tag = res.asof.replace(":", "").replace("-", "")[:15]
        tilt_df = pd.DataFrame({
            "asof_timestamp_utc": res.asof,
            "category": res.categories,
            "probability": tilted_cat_probs,
            "probability_pct": (tilted_cat_probs * 100).round(4),
        })
        tilt_path = Path(args.outdir) / f"kalshi_categories_tilt_{lam_tag}_{asof_tag}.csv"
        tilt_df.to_csv(tilt_path, index=False)
        print(f"  tilted: {tilt_path}")

    shifted_cat_probs = None
    if args.cdf_shift is not None:
        from src.tilting import shift_cdf
        shifted_bins = shift_cdf(res.bin_probs, res.thresholds, args.cdf_shift)
        _, shifted_cat_probs = aggregate_to_categories(res.thresholds, shifted_bins, args.baseline)

    # Plotting
    if not args.no_plot:
        from src.plotting import plot_comparison, plot_kalshi_only

        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        asof_tag = res.asof.replace(":", "").replace("-", "")[:15]

        # Simple Kalshi-only bar chart
        plot_kalshi_only(
            res.category_probs,
            res.categories,
            title=f"Kalshi-implied distribution (as of {res.asof[:10]})",
            save_path=str(fig_dir / f"kalshi_distribution_{asof_tag}.png"),
        )

        # Comparison with Chicago Fed if provided
        if args.chicago_fed:
            from src.plotting import load_chicago_fed_probs

            chifed = load_chicago_fed_probs(args.chicago_fed)
            plot_comparison(
                bar_probs=chifed / 100.0,
                bar_label="Model-Implied Probability of Next Change",
                line_probs=res.category_probs,
                line_label="Kalshi-Implied Probabilities",
                categories=res.categories,
                baseline=args.baseline,
                title=f"BLS Unemployment Rate Probabilities (as of {res.asof[:10]})",
                save_path=str(fig_dir / f"kalshi_vs_chifed_{asof_tag}.png"),
            )

        # Tilted comparison (bars)
        if tilted_cat_probs is not None and args.chicago_fed:
            from src.plotting import plot_tilted_comparison

            plot_tilted_comparison(
                original_probs=res.category_probs,
                tilted_probs=tilted_cat_probs,
                model_probs=chifed / 100.0,
                categories=res.categories,
                lam=args.tilt_lambda,
                baseline=args.baseline,
                save_path=str(fig_dir / f"kalshi_tilted_{asof_tag}.png"),
            )

        # Combined lines plot (Kalshi vs ChiFed + optional tilt/shift)
        if args.chicago_fed:
            from src.plotting import plot_combined_lines
            from src.tilting import implied_lambda as compute_implied_lam

            impl_lam = None
            if shifted_cat_probs is not None:
                from src.tilting import compute_mean, compute_variance
                shifted_bins_for_mean = shift_cdf(res.bin_probs, res.thresholds, args.cdf_shift) if args.cdf_shift else None
                if shifted_bins_for_mean is not None:
                    shifted_mean = float(np.average(res.bin_midpoints, weights=shifted_bins_for_mean))
                    impl_lam = compute_implied_lam(res.bin_probs, res.bin_midpoints, shifted_mean)

            plot_combined_lines(
                kalshi_probs=res.category_probs,
                model_probs=chifed / 100.0,
                categories=res.categories,
                baseline=args.baseline,
                tilted_probs=tilted_cat_probs,
                tilt_lambda=args.tilt_lambda,
                shifted_probs=shifted_cat_probs,
                cdf_shift_delta=args.cdf_shift,
                implied_lam=impl_lam,
                asof=res.asof,
                save_path=str(fig_dir / f"kalshi_combined_lines_{asof_tag}.png"),
            )

    # Temporal comparison
    if not args.no_plot and args.compare_csv:
        from src.plotting import plot_temporal_comparison

        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        asof_tag = res.asof.replace(":", "").replace("-", "")[:15]

        strikes_old, prices_old, ts_old = load_from_csv(args.compare_csv)
        res_old = build_distribution(strikes_old, prices_old, baseline=args.baseline, timestamp=ts_old)

        plot_temporal_comparison(
            res_old=res_old,
            res_new=res,
            baseline=args.baseline,
            save_path=str(fig_dir / f"kalshi_temporal_{asof_tag}.png"),
        )
        print(f"  temporal: figures/kalshi_temporal_{asof_tag}.png")

    print("Done.")


if __name__ == "__main__":
    main()
