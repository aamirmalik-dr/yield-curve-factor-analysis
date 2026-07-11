# Data

This directory is gitignored except for this file. To populate it, run from the
repository root:

```
python scripts/download_data.py
```

That fetches daily par yield curve rates from the US Treasury website (public
domain US government data) and writes them to `data/treasury_yields.csv`. The
source endpoint is the Treasury "Daily Treasury Par Yield Curve Rates" CSV
download, one file per calendar year:

```
https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv
```

If the endpoint is unreachable, the loader falls back to any previously cached
CSV in this directory, and finally to a synthetic Nelson-Siegel generator so
everything still runs offline (`python scripts/download_data.py --synthetic`
writes synthetic data explicitly). The loader prints which source was used.
