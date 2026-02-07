#!/usr/bin/env python3
"""Run the full Kalshi unemployment-rate distribution workflow.

Usage
-----
# Live API pull with all plots (default):
    python main.py

# Offline from cached CSV:
    python main.py --offline

# Custom tilt/shift parameters:
    python main.py --tilt-lambda -1.0 --cdf-shift -0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.build_distribution import (
    DistributionResult,
    build_distribution,
    export_results,
    load_from_api,
    load_from_csv,
    load_from_strike_csv,
)
from src.plotting import (
    bbg_to_7cat,
    load_bbg_consensus,
    load_chicago_fed_probs,
    plot_bbg_consensus_histogram,
    plot_combined_lines,
    plot_comparison,
    plot_kalshi_chifed_bbg,
    plot_kalshi_only,
    plot_temporal_comparison,
    plot_tilted_comparison,
)
from src.tilting import (
    compute_mean,
    implied_lambda,
    shift_cdf,
    tilt_exponential,
)
from src.utils import DEFAULT_BASELINE, aggregate_to_categories


# -----------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------
EVENT_TICKER = "KXU3-26JAN"
BASELINE = DEFAULT_BASELINE
CHICAGO_FED_PATH = "data/raw/chi-labor-market-indicators.xlsx"
BBG_CONSENSUS_PATH = "data/raw/bbg-consensus-forecast.xlsx"
HISTORY_CSV = "data/raw/kalshi-price-history-kxu3-26jan-day.csv"
FEB2_SNAPSHOT = "data/raw/kalshi_strikes_20260202T011445Z.csv"
OUTDIR = "data/processed"
FIG_DIR = "figures"
TILT_LAMBDA = -0.50
CDF_SHIFT = -0.05


def run(
    offline: bool = False,
    baseline: float = BASELINE,
    tilt_lambda: float = TILT_LAMBDA,
    cdf_shift: float = CDF_SHIFT,
) -> None:
    fig_dir = Path(FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load strikes ------------------------------------------------
    if offline:
        print(f"Loading from CSV: {HISTORY_CSV}")
        strikes, prices, ts = load_from_csv(HISTORY_CSV)
    else:
        print(f"Pulling live data from Kalshi API (event: {EVENT_TICKER})...")
        strikes, prices, ts = load_from_api(EVENT_TICKER, raw_dir="data/raw")

    # ---- 2. Build distribution -------------------------------------------
    res = build_distribution(strikes, prices, baseline=baseline, timestamp=ts)
    paths = export_results(res, OUTDIR)
    asof_tag = res.asof.replace(":", "").replace("-", "")[:15]

    print(f"\nAs of: {res.asof}")
    print(f"Mean:      {res.mean:.3f}%")
    print(f"Decrease:  {res.prob_decrease * 100:.1f}%")
    print(f"No change: {res.prob_no_change * 100:.1f}%")
    print(f"Increase:  {res.prob_increase * 100:.1f}%")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    # ---- 3. Exponential tilt ---------------------------------------------
    tilted_bins = tilt_exponential(res.bin_probs, res.bin_midpoints, tilt_lambda)
    _, tilted_cat = aggregate_to_categories(res.thresholds, tilted_bins, baseline)

    # ---- 4. CDF shift ----------------------------------------------------
    shifted_bins = shift_cdf(res.bin_probs, res.thresholds, cdf_shift)
    _, shifted_cat = aggregate_to_categories(res.thresholds, shifted_bins, baseline)
    shifted_mean = float(np.average(res.bin_midpoints, weights=shifted_bins))
    impl_lam = implied_lambda(res.bin_probs, res.bin_midpoints, shifted_mean)

    # ---- 5. Chicago Fed model --------------------------------------------
    chifed = load_chicago_fed_probs(CHICAGO_FED_PATH) / 100.0

    # ---- 6. Plots --------------------------------------------------------
    # (a) Kalshi-only bar chart
    plot_kalshi_only(
        res.category_probs, res.categories,
        title=f"Kalshi-implied distribution (as of {res.asof[:10]})",
        save_path=str(fig_dir / f"kalshi_distribution_{asof_tag}.png"),
    )

    # (b) Kalshi vs Chicago Fed (bars + dashed line)
    plot_comparison(
        bar_probs=chifed,
        bar_label="Model-Implied Probability of Next Change",
        line_probs=res.category_probs,
        line_label="Kalshi-Implied Probabilities",
        categories=res.categories, baseline=baseline,
        title=f"BLS Unemployment Rate Probabilities (as of {res.asof[:10]})",
        save_path=str(fig_dir / f"kalshi_vs_chifed_{asof_tag}.png"),
    )

    # (c) Three-way bars (Kalshi vs tilted vs ChiFed)
    plot_tilted_comparison(
        original_probs=res.category_probs,
        tilted_probs=tilted_cat,
        model_probs=chifed,
        categories=res.categories, lam=tilt_lambda, baseline=baseline,
        save_path=str(fig_dir / f"kalshi_tilted_{asof_tag}.png"),
    )

    # (d) Combined lines (Kalshi + ChiFed + CDF-shift + exp-tilt)
    plot_combined_lines(
        kalshi_probs=res.category_probs,
        model_probs=chifed,
        categories=res.categories, baseline=baseline,
        tilted_probs=tilted_cat, tilt_lambda=tilt_lambda,
        shifted_probs=shifted_cat, cdf_shift_delta=cdf_shift,
        implied_lam=impl_lam, asof=res.asof,
        save_path=str(fig_dir / f"kalshi_combined_lines_{asof_tag}.png"),
    )

    # (e) Temporal comparison (Jan 30, Feb 2, current) — only when live
    if not offline:
        # Jan 30 snapshot (wide format CSV)
        strikes_jan30, prices_jan30, ts_jan30 = load_from_csv(HISTORY_CSV)
        res_jan30 = build_distribution(strikes_jan30, prices_jan30, baseline=baseline, timestamp=ts_jan30)

        # Feb 2 snapshot (long format CSV)
        strikes_feb2, prices_feb2, ts_feb2 = load_from_strike_csv(FEB2_SNAPSHOT)
        res_feb2 = build_distribution(strikes_feb2, prices_feb2, baseline=baseline, timestamp=ts_feb2)

        plot_temporal_comparison(
            results=[res_jan30, res_feb2, res],
            baseline=baseline,
            save_path=str(fig_dir / f"kalshi_temporal_{asof_tag}.png"),
        )

    # (f) Bloomberg consensus histogram
    plot_bbg_consensus_histogram(
        xlsx_path=BBG_CONSENSUS_PATH,
        save_path=str(fig_dir / f"bbg_consensus_{asof_tag}.png"),
    )

    # (g) Combined Kalshi + Chicago Fed + BBG consensus chart
    bbg_df = load_bbg_consensus(BBG_CONSENSUS_PATH)
    bbg_7cat = bbg_to_7cat(bbg_df, res.categories)
    plot_kalshi_chifed_bbg(
        kalshi_probs=res.category_probs,
        model_probs=chifed,
        bbg_probs=bbg_7cat,
        categories=res.categories,
        baseline=baseline,
        asof=res.asof,
        save_path=str(fig_dir / f"kalshi_chifed_bbg_{asof_tag}.png"),
    )

    print(f"\nFigures saved to {fig_dir}/")
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full Kalshi unemployment distribution workflow.")
    ap.add_argument("--offline", action="store_true",
                    help="Use cached CSV instead of live API")
    ap.add_argument("--baseline", type=float, default=BASELINE,
                    help=f"Reference unemployment rate (default {BASELINE})")
    ap.add_argument("--tilt-lambda", type=float, default=TILT_LAMBDA,
                    help=f"Exponential tilt parameter (default {TILT_LAMBDA})")
    ap.add_argument("--cdf-shift", type=float, default=CDF_SHIFT,
                    help=f"CDF shift in pp (default {CDF_SHIFT})")
    args = ap.parse_args()

    run(
        offline=args.offline,
        baseline=args.baseline,
        tilt_lambda=args.tilt_lambda,
        cdf_shift=args.cdf_shift,
    )


if __name__ == "__main__":
    main()
