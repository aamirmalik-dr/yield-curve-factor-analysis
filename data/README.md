# Data

Daily US Treasury par yield curve rates, public domain US government data.

## Committed sample

`treasury_sample.csv` is committed: 400 trading days, 2024-12-03 to 2026-07-10, 12 maturities from 1 month to 30 years. It is a carved subset of a live download from the US Treasury endpoint (not synthetic), and it is what `scripts/analyze.py` and `scripts/compare.py` use by default, so the whole analysis runs with no network.

## Full dataset

Everything else in this directory is gitignored. To fetch a full multi-year panel, run from the repository root:

```
python scripts/download_data.py
```

That writes `data/treasury_yields.csv` from the Treasury "Daily Treasury Par Yield Curve Rates" CSV download, one file per calendar year:

```
https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv
```

If the endpoint is unreachable, the loader falls back to any previously cached CSV in this directory, then to a synthetic Nelson-Siegel generator (`python scripts/download_data.py --synthetic` writes synthetic data explicitly). The loader prints which source was used, and synthetic data never enters reported results.
