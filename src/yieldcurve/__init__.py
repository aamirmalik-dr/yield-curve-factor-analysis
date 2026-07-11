"""Factor analysis of the US Treasury yield curve with PCA and NMF."""

from yieldcurve.data import (
    MATURITY_YEARS,
    download_treasury_yields,
    generate_synthetic_yields,
    load_yield_curve,
)
from yieldcurve.factors import (
    NMFResult,
    PCAResult,
    compare_reconstruction,
    fit_nmf,
    fit_pca,
)

__version__ = "0.1.0"

__all__ = [
    "MATURITY_YEARS",
    "NMFResult",
    "PCAResult",
    "compare_reconstruction",
    "download_treasury_yields",
    "fit_nmf",
    "fit_pca",
    "generate_synthetic_yields",
    "load_yield_curve",
]
