"""Factor decompositions of a yield curve panel: PCA and NMF.

PCA on the mean-centered panel recovers the classic level, slope, and
curvature factors of the term structure. NMF gives a parts-based alternative
with nonnegative loadings. Both results expose a reconstruction in the
original units so the two methods can be compared at matched component counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA


@dataclass
class PCAResult:
    """Result of a PCA decomposition of a yield panel.

    Attributes:
        explained_variance_ratio: Fraction of variance per component.
        loadings: DataFrame (maturities x components) of component loadings.
        scores: DataFrame (dates x components) of factor scores.
        reconstruction: Rank-k reconstruction of the panel in original units.
        rmse: Root-mean-square reconstruction error in yield points.
    """

    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame
    scores: pd.DataFrame
    reconstruction: pd.DataFrame
    rmse: float


@dataclass
class NMFResult:
    """Result of an NMF decomposition of a yield panel.

    Attributes:
        components: DataFrame (maturities x components) of nonnegative basis curves.
        weights: DataFrame (dates x components) of nonnegative weights.
        reconstruction: Rank-k reconstruction of the panel.
        rmse: Root-mean-square reconstruction error in yield points.
    """

    components: pd.DataFrame
    weights: pd.DataFrame
    reconstruction: pd.DataFrame
    rmse: float


def _rmse(panel: pd.DataFrame, reconstruction: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((panel.to_numpy() - reconstruction.to_numpy()) ** 2)))


def _orient_loadings(loadings: np.ndarray) -> np.ndarray:
    """Flip component signs so each loading vector has a positive mean.

    PCA signs are arbitrary; this convention makes the level factor uniformly
    positive and plots reproducible across runs.
    """
    signs = np.sign(loadings.mean(axis=0))
    signs[signs == 0] = 1.0
    return loadings * signs


def fit_pca(panel: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """Fit PCA to a yield curve panel.

    The panel is mean-centered per maturity (no scaling, so loadings stay in
    yield units). Component signs are normalized so each loading vector has a
    positive mean.

    Args:
        panel: DataFrame (dates x maturities) of yields in percent.
        n_components: Number of principal components to keep.

    Returns:
        PCAResult with explained variance ratios, loadings, scores, the rank-k
        reconstruction, and its RMSE.
    """
    model = PCA(n_components=n_components)
    scores = model.fit_transform(panel.to_numpy())
    loadings = model.components_.T  # (maturities, components)

    oriented = _orient_loadings(loadings)
    flip = np.sign((oriented * loadings).sum(axis=0))  # +1 kept, -1 flipped
    scores = scores * flip

    component_names = [f"PC{i + 1}" for i in range(n_components)]
    recon_values = scores @ oriented.T + model.mean_
    reconstruction = pd.DataFrame(recon_values, index=panel.index, columns=panel.columns)
    return PCAResult(
        explained_variance_ratio=model.explained_variance_ratio_,
        loadings=pd.DataFrame(oriented, index=panel.columns, columns=component_names),
        scores=pd.DataFrame(scores, index=panel.index, columns=component_names),
        reconstruction=reconstruction,
        rmse=_rmse(panel, reconstruction),
    )


def fit_nmf(
    panel: pd.DataFrame,
    n_components: int = 3,
    max_iter: int = 1000,
    seed: int = 0,
) -> NMFResult:
    """Fit NMF to a yield curve panel.

    Yields are nonnegative, so NMF applies directly to the raw panel. Unlike
    PCA there is no mean-centering; the components are nonnegative basis
    curves and every observed curve is a nonnegative mix of them.

    Args:
        panel: DataFrame (dates x maturities) of nonnegative yields in percent.
        n_components: Number of NMF components.
        max_iter: Maximum NMF iterations.
        seed: Random seed for the NNDSVD-based initialization.

    Returns:
        NMFResult with basis curves, weights, the reconstruction, and its RMSE.

    Raises:
        ValueError: If the panel contains negative values.
    """
    values = panel.to_numpy()
    if (values < 0).any():
        raise ValueError("NMF requires a nonnegative panel")
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=max_iter,
        random_state=seed,
    )
    weights = model.fit_transform(values)
    components = model.components_.T  # (maturities, components)
    component_names = [f"NMF{i + 1}" for i in range(n_components)]
    reconstruction = pd.DataFrame(weights @ components.T, index=panel.index, columns=panel.columns)
    return NMFResult(
        components=pd.DataFrame(components, index=panel.columns, columns=component_names),
        weights=pd.DataFrame(weights, index=panel.index, columns=component_names),
        reconstruction=reconstruction,
        rmse=_rmse(panel, reconstruction),
    )


def compare_reconstruction(
    panel: pd.DataFrame,
    component_counts: list[int] | None = None,
) -> pd.DataFrame:
    """Compare PCA and NMF reconstruction RMSE at matched component counts.

    Args:
        panel: DataFrame (dates x maturities) of yields in percent.
        component_counts: Component counts to evaluate. Defaults to [1, 2, 3, 4, 5, 6].

    Returns:
        DataFrame indexed by component count with columns ``pca_rmse`` and
        ``nmf_rmse`` in yield points.
    """
    component_counts = component_counts or [1, 2, 3, 4, 5, 6]
    rows = []
    for k in component_counts:
        rows.append(
            {
                "n_components": k,
                "pca_rmse": fit_pca(panel, n_components=k).rmse,
                "nmf_rmse": fit_nmf(panel, n_components=k).rmse,
            }
        )
    return pd.DataFrame(rows).set_index("n_components")
