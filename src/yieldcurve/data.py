"""Data loading for US Treasury par yield curve rates.

The loader tries three sources in order and reports which one it used:

1. The live US Treasury endpoint for daily par yield curve rates
   (public-domain US government data).
2. A locally cached CSV written by a previous successful download.
3. A synthetic Nelson-Siegel yield curve generator, so demos and tests
   always run offline.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TREASURY_URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)

# Treasury CSV column names mapped to maturity in years. Short tenors that the
# Treasury only started publishing recently (1.5 Month, 4 Month) are excluded so
# that multi-year panels have no structurally missing columns.
MATURITY_YEARS: dict[str, float] = {
    "1 Mo": 1 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "6 Mo": 6 / 12,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

DEFAULT_CACHE = Path("data") / "treasury_yields.csv"


def download_treasury_yields(years: list[int], timeout: float = 30.0) -> pd.DataFrame:
    """Download daily par yield curve rates from the US Treasury website.

    Args:
        years: Calendar years to fetch, for example ``[2023, 2024, 2025]``.
        timeout: Per-request timeout in seconds.

    Returns:
        DataFrame indexed by date (ascending) with one column per maturity in
        ``MATURITY_YEARS``, values in percent. Rows with any missing tenor are
        dropped.

    Raises:
        requests.RequestException: If any request fails.
        ValueError: If a response does not parse into the expected columns.
    """
    frames: list[pd.DataFrame] = []
    for year in years:
        url = TREASURY_URL_TEMPLATE.format(year=year)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if "Date" not in frame.columns:
            raise ValueError(f"Unexpected Treasury CSV format for {year}: no Date column")
        missing = [col for col in MATURITY_YEARS if col not in frame.columns]
        if missing:
            raise ValueError(f"Treasury CSV for {year} is missing columns: {missing}")
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date")[list(MATURITY_YEARS)]
        frames.append(frame)
    combined = pd.concat(frames).sort_index()
    combined = combined.apply(pd.to_numeric, errors="coerce").dropna()
    if combined.empty:
        raise ValueError("Treasury download produced an empty panel")
    return combined


def generate_synthetic_yields(
    n_days: int = 750,
    seed: int = 0,
    maturities: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate a synthetic daily yield curve panel from a Nelson-Siegel model.

    Level, slope, and curvature factors follow independent AR(1) processes, so
    the panel has the same low-dimensional structure as real Treasury data and
    PCA recovers the classic three factors.

    Args:
        n_days: Number of business days to simulate.
        seed: Seed for the random generator.
        maturities: Mapping of column name to maturity in years. Defaults to
            ``MATURITY_YEARS``.

    Returns:
        DataFrame indexed by business dates with one column per maturity,
        values in percent and strictly positive.
    """
    maturities = maturities or MATURITY_YEARS
    rng = np.random.default_rng(seed)
    tau = np.array(list(maturities.values()))

    # Nelson-Siegel basis functions with fixed decay.
    lam = 0.6
    basis_level = np.ones_like(tau)
    basis_slope = (1 - np.exp(-lam * tau)) / (lam * tau)
    basis_curv = basis_slope - np.exp(-lam * tau)

    def ar1(mean: float, phi: float, sigma: float) -> np.ndarray:
        x = np.empty(n_days)
        x[0] = mean
        shocks = rng.normal(0.0, sigma, n_days)
        for t in range(1, n_days):
            x[t] = mean + phi * (x[t - 1] - mean) + shocks[t]
        return x

    level = ar1(mean=4.0, phi=0.999, sigma=0.03)
    slope = ar1(mean=-1.0, phi=0.995, sigma=0.05)
    curv = ar1(mean=0.5, phi=0.99, sigma=0.08)
    noise = rng.normal(0.0, 0.01, (n_days, len(tau)))

    yields = (
        level[:, None] * basis_level
        + slope[:, None] * basis_slope
        + curv[:, None] * basis_curv
        + noise
    )
    yields = np.clip(yields, 0.01, None)
    dates = pd.bdate_range(end="2025-12-31", periods=n_days)
    return pd.DataFrame(yields, index=dates, columns=list(maturities))


def load_yield_curve(
    years: list[int] | None = None,
    cache_path: str | Path = DEFAULT_CACHE,
    allow_synthetic: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load a yield curve panel, falling back gracefully when offline.

    Tries the live Treasury endpoint first, then a cached CSV, then the
    synthetic generator. A successful download refreshes the cache.

    Args:
        years: Calendar years to request from the Treasury. Defaults to the
            three most recent full years plus the current year.
        cache_path: Location of the cached CSV.
        allow_synthetic: If False, raise instead of falling back to synthetic
            data.
        verbose: If True, print which source was used.

    Returns:
        Tuple of (panel, source) where source is one of ``"treasury"``,
        ``"cache"``, or ``"synthetic"``.

    Raises:
        RuntimeError: If all real sources fail and ``allow_synthetic`` is False.
    """
    if years is None:
        current = pd.Timestamp.now().year
        years = list(range(current - 3, current + 1))
    cache_path = Path(cache_path)

    try:
        panel = download_treasury_yields(years)
    except (requests.RequestException, ValueError) as exc:
        download_error = exc
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(cache_path)
        if verbose:
            print(
                f"Data source: live US Treasury download ({len(panel)} days), cached to {cache_path}"
            )
        return panel, "treasury"

    if cache_path.exists():
        panel = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if verbose:
            print(
                f"Data source: cached CSV at {cache_path} ({len(panel)} days); live download failed: {download_error}"
            )
        return panel, "cache"

    if not allow_synthetic:
        raise RuntimeError(
            f"Treasury download failed ({download_error}) and no cache at {cache_path}"
        )
    panel = generate_synthetic_yields()
    if verbose:
        print(
            f"Data source: synthetic Nelson-Siegel generator ({len(panel)} days); live download failed: {download_error}"
        )
    return panel, "synthetic"
