# Predicting the Impact of RAM Supply Shortages on Gaming Hardware Prices and Consumer Demand

Early evidence and a data-limited machine learning framework. Since 2025, AI data centers have absorbed much of the
world's DRAM output, and gaming hardware has become more expensive. This project (1) analyses the public data that
exist, and (2) provides a ready-to-run comparison of 15 classifiers for predicting next-month price and demand bands.

## Project status (read this first)

- **Done:** descriptive analysis of public data; a DRAM contract-price index; console price pass-through ratios;
  an August 2026 demand comparison; the 15-classifier program; the paper (Springer LNCS style).
- **Not done:** the classifier comparison has **not been run**, because the monthly retail and demand table it needs is
  paid or login-only and could not be collected. `data/dram_gaming_panel.csv` is a header-only template.
  The program stops with a message if the panel has fewer than 100 rows. **No accuracy figures are reported.**
- **Synthetic data:** `data/synthetic/` holds an invented dataset used only to check the code runs. Never report
  results from it.

## Findings from the data collected

| Finding | Value | Caveat |
| --- | --- | --- |
| DRAM contract index, 4Q26 (4Q25 = 100) | about 408 (range 379 to 438) | Only 1Q26 is a survey result; the rest are forecasts |
| Console list-price rises, 2026 | 11% to 25% | News reports; tariffs and exchange rates also moved |
| Pass-through ratio (price rise / index rise) | 0.04 to 0.10 | Descriptive, not causal |
| Aug 2026 US hardware: price paid / units | +13.7% / -15% | One month, one country, mix-affected |

## Repository layout

```
data/        public datasets (CSV, each row has a source link), SOURCES.md, panel template, synthetic test data
src/         trendforce_index.py (DRAM index), dram_framework.py (15 classifiers + charts), make_synthetic_data.py
tests/       unit tests for the index
paper/       the paper (.docx) and the script that regenerates its figures
results/     program output (tables and charts); git-ignored
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pytest                                                  # checks the DRAM index
python src/trendforce_index.py                          # prints the index (works on the supplied data)
python src/dram_framework.py                            # prints the index, then stops: panel is empty
```

## Running the classifier comparison

1. Fill `data/dram_gaming_panel.csv`: one row per product category per month, at least 30 months, with columns
   `month` (YYYY-MM), `category`, `retail_price`, `msrp`, `price_to_msrp`, `dram_contract_price`, `dram_spot_price`,
   `inventory_weeks`, `hbm_share_pct`, `search_interest`, `demand_index`, `tariff_flag`.
   Suggested sources: DRAMeXchange (memory prices), a retail price tracker, Circana (unit sales), Google Trends.
2. Run `python src/dram_framework.py`. It writes `results/results_price_band.csv`, `results/results_demand_band.csv`
   and three charts (`fig_per_split_price_band.png`, `fig_final_all_executions.png`, `fig_comparison_best_papers.png`).

Method in brief: each product-month is labelled with next-month price band (stable below 2%, moderate 2 to 8%,
severe above 8%) and demand band (declining below -5%, flat, growing above +5%). The 15 classifiers are tested on
60:40, 70:30, 80:20 random splits and on a chronological hold-out, scored by accuracy and macro-F1. The scaler and
imputer are fitted on training data only to avoid leakage. The 2%/8% and 5% thresholds are assumptions.

## Data notes

- `trendforce_dram_public_figures.csv`: quarter-over-quarter contract-price ranges; mostly forecasts.
- `console_price_events_2026.csv`: US console list-price changes with a confidence column.
- `us_hardware_demand_circana_2026.csv`: US hardware spending, units and average price paid (Aug 2026 is the cleanest month).
- The data were compiled from third-party web sources (see `data/SOURCES.md`). Check each source's terms before reuse.

## Paper

`paper/RAM_Shortage_Gaming_Hardware_Springer_LNCS.docx` (single-column Springer LNCS style). Highlighted items in the
paper are placeholders for results that need real data. The program listing in the paper's appendix is a single-file
version of `src/`; the repository splits the DRAM index into `trendforce_index.py` and uses relative data paths.
Regenerate figures with `cd paper/figures && python make_figures.py`.

## License

Code: MIT (see `LICENSE`). The data files are compiled from third-party sources and are not covered by this license.
