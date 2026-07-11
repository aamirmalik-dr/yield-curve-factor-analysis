"""Download US Treasury par yield curve rates to data/treasury_yields.csv.

Falls back to a cached CSV, then to the synthetic generator, and prints which
source was used. Run from the repository root:

    python scripts/download_data.py --years 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yieldcurve.data import DEFAULT_CACHE, generate_synthetic_yields, load_yield_curve


def main() -> None:
    """Fetch yield data and write it to the cache path."""
    parser = argparse.ArgumentParser(description="Download Treasury par yield curve rates")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Calendar years to fetch (default: last three full years plus current)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Output CSV path (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip the download and write synthetic data instead",
    )
    args = parser.parse_args()

    if args.synthetic:
        panel = generate_synthetic_yields()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.out)
        print(f"Data source: synthetic Nelson-Siegel generator ({len(panel)} days)")
    else:
        panel, _source = load_yield_curve(years=args.years, cache_path=args.out)
        if args.out != DEFAULT_CACHE or not args.out.exists():
            args.out.parent.mkdir(parents=True, exist_ok=True)
            panel.to_csv(args.out)
    print(f"Wrote {len(panel)} rows x {panel.shape[1]} maturities to {args.out}")


if __name__ == "__main__":
    main()
