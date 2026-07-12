"""Run the yield curve factor analysis and write figures plus metrics.

By default the analysis runs fully offline on the committed sample panel at
``data/treasury_sample.csv``. Outputs go to ``results/``: the factor-loading
heatmap, the scree plot, and ``metrics.json``.

    python scripts/analyze.py                                  # offline, committed sample
    python scripts/analyze.py --data data/treasury_yields.csv  # any local panel CSV
    python scripts/analyze.py --live                           # live Treasury download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from yieldcurve.data import SAMPLE_PATH, load_yield_curve, read_panel_csv  # noqa: E402
from yieldcurve.factors import fit_nmf, fit_pca  # noqa: E402
from yieldcurve.reporting import merge_metrics  # noqa: E402

FACTOR_NAMES = ["Level", "Slope", "Curvature"]


def load_panel(data: Path | None, live: bool, synthetic: bool) -> tuple[pd.DataFrame, str]:
    """Load the analysis panel from a CSV, the live endpoint, or the generator.

    Args:
        data: Path to a local panel CSV, or None.
        live: If True, use the resilient live loader (network, cache, synthetic).
        synthetic: If True, use the synthetic Nelson-Siegel generator.

    Returns:
        Tuple of (panel, source description).
    """
    if synthetic:
        from yieldcurve.data import generate_synthetic_yields

        panel = generate_synthetic_yields()
        return panel, "synthetic Nelson-Siegel generator"
    if live:
        panel, source = load_yield_curve()
        return panel, source
    path = Path(data or SAMPLE_PATH)
    return read_panel_csv(path), path.as_posix()


def plot_loading_heatmap(loadings: pd.DataFrame, out_path: Path) -> None:
    """Plot PCA factor loadings as an annotated maturities-by-components heatmap.

    Args:
        loadings: DataFrame (maturities x components) of PCA loadings.
        out_path: Destination PNG path.
    """
    values = loadings.to_numpy().T  # (components, maturities)
    vmax = abs(values).max()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(loadings.index)))
    ax.set_xticklabels(loadings.index, rotation=45, ha="right", fontsize=9)
    labels = [
        f"{col}\n({FACTOR_NAMES[i]})" if i < len(FACTOR_NAMES) else col
        for i, col in enumerate(loadings.columns)
    ]
    ax.set_yticks(range(len(loadings.columns)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(values[i, j]) > 0.6 * vmax else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("Maturity")
    ax.set_title("PCA factor loadings of the Treasury yield curve")
    fig.colorbar(im, ax=ax, label="Loading", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scree(evr: pd.Series, out_path: Path) -> None:
    """Plot explained variance ratio per component with a cumulative line.

    Args:
        evr: Explained variance ratio per component, in order.
        out_path: Destination PNG path.
    """
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


def main() -> None:
    """Run the decomposition, write figures and metrics."""
    parser = argparse.ArgumentParser(description="Yield curve factor analysis")
    parser.add_argument("--components", type=int, default=3, help="Components for PCA and NMF")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"Local panel CSV to analyze (default: {SAMPLE_PATH})",
    )
    parser.add_argument(
        "--live", action="store_true", help="Download live Treasury data instead of a local CSV"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Force the synthetic generator (offline demo)"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel, source = load_panel(args.data, args.live, args.synthetic)
    print(f"Data source: {source}")
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

    plot_loading_heatmap(pca.loadings, args.out_dir / "factor_loadings.png")
    plot_scree(pd.Series(pca.explained_variance_ratio), args.out_dir / "scree.png")
    merge_metrics(
        args.out_dir / "metrics.json",
        {
            "data_source": source,
            "panel_days": int(panel.shape[0]),
            "panel_maturities": int(panel.shape[1]),
            "panel_start": str(panel.index.min().date()),
            "panel_end": str(panel.index.max().date()),
            "n_components": args.components,
            "pca_explained_variance_ratio": [
                round(float(r), 6) for r in pca.explained_variance_ratio
            ],
            "pca_cumulative_explained": round(float(pca.explained_variance_ratio.sum()), 6),
            "pca_rmse": round(pca.rmse, 6),
            "nmf_rmse": round(nmf.rmse, 6),
        },
    )
    print(f"\nWrote factor_loadings.png, scree.png, metrics.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
