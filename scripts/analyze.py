"""Run the full yield curve factor analysis and write figures to results/.

Loads Treasury data (live, cached, or synthetic fallback), fits PCA and NMF,
prints explained variance and reconstruction errors, and saves three figures:
factor loadings, scree plot, and a reconstruction comparison.

    python scripts/analyze.py --components 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from yieldcurve.data import MATURITY_YEARS, load_yield_curve
from yieldcurve.factors import compare_reconstruction, fit_nmf, fit_pca

matplotlib.use("Agg")

FACTOR_NAMES = ["Level", "Slope", "Curvature"]


def plot_loadings(loadings: pd.DataFrame, out_path: Path) -> None:
    """Plot PCA loadings against maturity on a log axis."""
    tau = [MATURITY_YEARS[m] for m in loadings.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, col in enumerate(loadings.columns):
        label = f"{col} ({FACTOR_NAMES[i]})" if i < len(FACTOR_NAMES) else col
        ax.plot(tau, loadings[col], marker="o", label=label)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(tau)
    ax.set_xticklabels(loadings.index, rotation=45, fontsize=8)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Loading")
    ax.set_title("PCA factor loadings of the Treasury yield curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scree(evr: pd.Series, out_path: Path) -> None:
    """Plot explained variance ratio per component with a cumulative line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(1, len(evr) + 1)
    ax.bar(x, evr, color="#4878CF", label="Per component")
    ax.plot(x, evr.cumsum(), marker="o", color="#D65F5F", label="Cumulative")
    ax.set_xticks(list(x))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reconstruction(comparison: pd.DataFrame, out_path: Path) -> None:
    """Plot PCA vs NMF reconstruction RMSE at matched component counts."""
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
    """Run the decomposition and write figures."""
    parser = argparse.ArgumentParser(description="Yield curve factor analysis")
    parser.add_argument("--components", type=int, default=3, help="Components for PCA and NMF")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Figure directory")
    parser.add_argument(
        "--synthetic", action="store_true", help="Force the synthetic generator (offline demo)"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from yieldcurve.data import generate_synthetic_yields

        panel = generate_synthetic_yields()
        print(f"Data source: synthetic Nelson-Siegel generator ({len(panel)} days)")
    else:
        panel, _source = load_yield_curve()
    print(
        f"Panel: {panel.shape[0]} days x {panel.shape[1]} maturities, "
        f"{panel.index.min().date()} to {panel.index.max().date()}"
    )

    pca = fit_pca(panel, n_components=args.components)
    print("\nPCA explained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio, start=1):
        name = FACTOR_NAMES[i - 1] if i <= len(FACTOR_NAMES) else ""
        print(f"  PC{i} {name:<10} {ratio:.4f}")
    print(f"  Cumulative (first {args.components}): {pca.explained_variance_ratio.sum():.4f}")
    print(f"PCA reconstruction RMSE ({args.components} comps): {pca.rmse:.4f} yield points")

    nmf = fit_nmf(panel, n_components=args.components)
    print(f"NMF reconstruction RMSE ({args.components} comps): {nmf.rmse:.4f} yield points")

    comparison = compare_reconstruction(panel)
    print("\nReconstruction RMSE by component count:")
    print(comparison.round(4).to_string())

    plot_loadings(pca.loadings, args.out_dir / "factor_loadings.png")
    plot_scree(pd.Series(pca.explained_variance_ratio), args.out_dir / "scree.png")
    plot_reconstruction(comparison, args.out_dir / "reconstruction.png")
    print(f"\nFigures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
