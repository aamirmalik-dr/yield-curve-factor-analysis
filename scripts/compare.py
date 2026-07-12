"""PCA vs NMF reconstruction-error sweep over component counts.

Fits both decompositions at k = 1..6 (configurable), prints the RMSE table,
writes ``results/reconstruction.png``, and merges the sweep into
``results/metrics.json``. Runs fully offline on the committed sample by default.

    python scripts/compare.py
    python scripts/compare.py --max-components 8 --data data/treasury_yields.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from yieldcurve.data import SAMPLE_PATH, read_panel_csv  # noqa: E402
from yieldcurve.factors import compare_reconstruction  # noqa: E402
from yieldcurve.reporting import merge_metrics  # noqa: E402


def plot_reconstruction(comparison: pd.DataFrame, out_path: Path) -> None:
    """Plot PCA vs NMF reconstruction RMSE at matched component counts.

    Args:
        comparison: DataFrame indexed by component count with columns
            ``pca_rmse`` and ``nmf_rmse``.
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(comparison.index, comparison["pca_rmse"], marker="o", label="PCA")
    ax.plot(comparison.index, comparison["nmf_rmse"], marker="s", label="NMF")
    ax.set_yscale("log")
    ax.set_xticks(list(comparison.index))
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Reconstruction RMSE (yield points, log scale)")
    ax.set_title("PCA vs NMF reconstruction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the sweep, write the figure, and merge metrics."""
    parser = argparse.ArgumentParser(description="PCA vs NMF reconstruction sweep")
    parser.add_argument(
        "--data",
        type=Path,
        default=SAMPLE_PATH,
        help=f"Local panel CSV to analyze (default: {SAMPLE_PATH})",
    )
    parser.add_argument(
        "--max-components", type=int, default=6, help="Sweep component counts 1..max"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Output directory")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = read_panel_csv(args.data)
    print(f"Data source: {args.data}")
    print(
        f"Panel: {panel.shape[0]} days x {panel.shape[1]} maturities, "
        f"{panel.index.min().date()} to {panel.index.max().date()}"
    )

    counts = list(range(1, args.max_components + 1))
    comparison = compare_reconstruction(panel, component_counts=counts)
    print("\nReconstruction RMSE by component count (yield points):")
    print(comparison.round(4).to_string())

    plot_reconstruction(comparison, args.out_dir / "reconstruction.png")
    merge_metrics(
        args.out_dir / "metrics.json",
        {
            "reconstruction_sweep": {
                str(k): {
                    "pca_rmse": round(float(row["pca_rmse"]), 6),
                    "nmf_rmse": round(float(row["nmf_rmse"]), 6),
                }
                for k, row in comparison.iterrows()
            }
        },
    )
    print(f"\nWrote reconstruction.png and updated metrics.json in {args.out_dir}/")


if __name__ == "__main__":
    main()
