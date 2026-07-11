# yield-curve-factor-analysis

Dimensionality reduction of the US Treasury yield curve. PCA on a panel of daily par yields recovers the classic level, slope, and curvature factors, and an NMF variant provides a nonnegative, parts-based alternative. The two methods are compared by reconstruction error at matched component counts.

## What it does

- Downloads daily Treasury par yield curve rates from the US Treasury website (public domain US government data), covering 12 maturities from 1 month to 30 years.
- Falls back to a locally cached CSV, then to a synthetic Nelson-Siegel generator, so every script, test, and the demo notebook runs fully offline. The loader always prints which source it used.
- Fits PCA on the mean-centered panel, returning explained variance ratios, sign-normalized loadings, factor scores, and a rank-k reconstruction.
- Fits NMF on the raw nonnegative panel, returning nonnegative basis curves, weights, and the reconstruction error.
- Compares PCA and NMF reconstruction RMSE across component counts and writes three figures: factor loadings, scree plot, and the reconstruction comparison.

## What it does not do

- No yield forecasting, no trading signals, no bootstrapping of zero curves. This is a descriptive factor decomposition of par yields.
- No intraday or non-US data. The loader targets the US Treasury daily par yield CSV endpoint only.

## Install

```
python -m venv .venv
.venv\Scripts\activate      # Windows, or: source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11 or newer.

## Run

```
python scripts/download_data.py           # fetch data to data/treasury_yields.csv
python scripts/analyze.py                 # full analysis, figures in results/
python scripts/analyze.py --synthetic     # guaranteed-offline demo
pytest -q                                 # tests, fully offline
```

The demo notebook is `notebooks/demo.ipynb`, executed with saved outputs.

## Results

Produced by `python scripts/analyze.py` on live Treasury data downloaded at run time: 880 trading days, 2023-01-03 to 2026-07-10, 12 maturities.

PCA explained variance ratio (mean-centered yields, no scaling):

| Component | Interpretation | Explained variance |
| --- | --- | --- |
| PC1 | Level | 0.7741 |
| PC2 | Slope | 0.1738 |
| PC3 | Curvature | 0.0406 |
| First three cumulative | | 0.9884 |

The first component loads with the same sign on every maturity (level), the second changes sign between short and long tenors (slope), and the third is hump-shaped (curvature), matching the standard term-structure factor literature.

Reconstruction RMSE in yield points at matched component counts:

| Components | PCA | NMF |
| --- | --- | --- |
| 1 | 0.2459 | 0.3703 |
| 2 | 0.1181 | 0.1297 |
| 3 | 0.0556 | 0.0681 |
| 4 | 0.0365 | 0.0531 |
| 5 | 0.0208 | 0.0586 |

PCA reconstructs more accurately at every matched count, as expected: PCA with its mean term is the optimal linear rank-k approximation, while NMF trades accuracy for nonnegative, more interpretable basis curves. The small NMF uptick from 4 to 5 components reflects a local optimum of the nonconvex NMF objective, reported as observed.

These numbers came from live Treasury data because the endpoint was reachable during the run. Offline, the loader reports that it used the cache or the synthetic generator, and synthetic results will differ.

## Package layout

```
src/yieldcurve/     library code (data loading, PCA and NMF decompositions)
scripts/            download_data.py, analyze.py
notebooks/          demo.ipynb with executed outputs
tests/              pytest suite, runs fully offline
data/               gitignored, see data/README.md
```

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/dr-aamirmalik

## License

MIT, see LICENSE.
