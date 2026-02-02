# Kalshi-Implied Unemployment Rate Distributions

Extract risk-neutral probability distributions over the U.S. unemployment rate from [Kalshi](https://kalshi.com) prediction-market prices, compare them to the [Chicago Fed Real-Time Unemployment Rate](https://www.chicagofed.org/research/data/real-time-unemployment-rate) model, and apply exponential tilting / CDF-shift sensitivity analysis.

Distributions are constructed using the methodology described in [Diercks, Katz, and Wright (2026), "Kalshi and the Rise of Macro Markets," NBER Working Paper 34702](https://www.nber.org/papers/w34702). Contract prices from the full strike surface are treated as risk-neutral exceedance probabilities, monotonicity is enforced, and the exceedance curve is differenced to recover a discrete PMF over unemployment outcomes.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

This pulls live prices from the Kalshi API, builds the distribution, applies default tilt/shift parameters, and saves all figures and CSVs.

### Offline mode (from cached CSV)

```bash
python main.py --offline
```

### Custom parameters

```bash
python main.py --tilt-lambda -1.0 --cdf-shift -0.10 --baseline 4.4
```

## Repository Structure

```
.
├── main.py                     # Entry point — runs the full workflow
├── src/
│   ├── kalshi_api.py           # Kalshi API client (RSA-PSS auth)
│   ├── build_distribution.py   # Strike surface → PMF → 7-category distribution
│   ├── tilting.py              # Exponential tilting and CDF-shift utilities
│   ├── plotting.py             # All plot types + Chicago Fed data loader
│   └── utils.py                # Shared constants, monotonicity, aggregation
├── data/
│   ├── raw/                    # Source data: Kalshi CSVs, Chicago Fed Excel
│   └── processed/              # Output CSVs (fine-grid bins, categories)
├── figures/                    # Generated plots
├── keys/                       # RSA private key (gitignored)
├── .env                        # API credentials (gitignored)
├── requirements.txt
└── README.md
```

## Methodology

### Strike surface → distribution

1. **Exceedance probabilities**: Each Kalshi contract "Above X%" pays $1 if unemployment exceeds X. The YES price (in cents / 100) is interpreted as q(U > X).
2. **Monotonicity**: Exceedance probabilities are clipped to be non-increasing in the strike — violations from microstructure noise would produce negative bin masses.
3. **Bin PMF**: Adjacent exceedance probabilities are differenced to recover probability mass for each interval: q(t_{i-1} < U ≤ t_i) = q(U > t_{i-1}) − q(U > t_i).
4. **Category aggregation**: Fine-grid bins are aggregated into 7 display categories centred on the baseline rate (default 4.4%): ≤4.1%, 4.2%, 4.3%, 4.4%, 4.5%, 4.6%, ≥4.7%.

### Risk adjustments

- **Exponential tilting**: q_λ(u) ∝ q(u) · exp{λ(u − μ₀)}. Negative λ shifts weight toward lower unemployment. This is a sensitivity reweighting of the pricing measure, not a uniquely identified Q→P conversion.
- **CDF shift**: F_δ(u) = F(u + δ). A simple parallel shift of the entire distribution by δ percentage points.
- **Implied λ**: For a CDF-shifted distribution with mean μ_δ, the moment-based tilt parameter is λ = (μ_δ − μ₀) / σ₀².

## Outputs

### CSV files (in `data/processed/`)

| File | Contents |
|------|----------|
| `kalshi_finegrid_strikes_*.csv` | Strike-level exceedance probabilities |
| `kalshi_finegrid_bins_*.csv` | Fine-grid bin PMF (11 bins) |
| `kalshi_categories_*.csv` | 7-category aggregated distribution |
| `kalshi_categories_tilt_*.csv` | Tilted distribution (when `--tilt-lambda` is set) |

### Figures (in `figures/`)

| Figure | Description |
|--------|-------------|
| `kalshi_distribution_*.png` | Simple bar chart of Kalshi-implied categories |
| `kalshi_vs_chifed_*.png` | Bars (Chicago Fed) + dashed line (Kalshi) with relative-odds box |
| `kalshi_tilted_*.png` | Grouped bars: Kalshi vs tilted vs Chicago Fed |
| `kalshi_combined_lines_*.png` | All-lines comparison: Kalshi, Chicago Fed, CDF-shifted, exp-tilted |
| `kalshi_temporal_*.png` | Two Kalshi snapshots over time (live vs cached baseline) |

## API Setup

The Kalshi API uses RSA-PSS per-request signing. To configure:

1. Place your RSA private key at `keys/kalshi_rsa_private_key.pem`
2. Create a `.env` file:
   ```
   KALSHI_KEY_ID=your-key-id-here
   KALSHI_PRIVATE_KEY_PATH=keys/kalshi_rsa_private_key.pem
   ```

Both `keys/` and `.env` are gitignored.

## Advanced CLI (`src/build_distribution.py`)

The `src/build_distribution.py` module can also be invoked directly for more control:

```bash
# From CSV with specific options
python -m src.build_distribution \
    --input data/raw/kalshi-price-history-kxu3-26jan-day.csv \
    --chicago-fed data/raw/chi-labor-market-indicators.xlsx \
    --tilt-lambda -0.50 \
    --cdf-shift -0.05 \
    --compare-csv data/raw/kalshi-price-history-kxu3-26jan-day.csv \
    --outdir data/processed

# From live API
python -m src.build_distribution \
    --market-id KXU3-26JAN \
    --chicago-fed data/raw/chi-labor-market-indicators.xlsx \
    --outdir data/processed
```

## References

- Diercks, A., Katz, J., and Wright, J. (2026). "Kalshi and the Rise of Macro Markets." NBER Working Paper 34702. https://www.nber.org/papers/w34702
