"""Tests for the factor decomposition module. All tests run fully offline."""

from __future__ import annotations

import numpy as np
import pytest

from yieldcurve.data import generate_synthetic_yields
from yieldcurve.factors import compare_reconstruction, fit_nmf, fit_pca


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_yields(n_days=600, seed=42)


class TestPCA:
    def test_explained_variance_valid(self, panel) -> None:
        result = fit_pca(panel, n_components=3)
        evr = result.explained_variance_ratio
        assert len(evr) == 3
        assert (evr > 0).all()
        assert np.all(np.diff(evr) <= 0)  # sorted descending
        assert evr.sum() <= 1.0 + 1e-9

    def test_first_factor_dominates(self, panel) -> None:
        result = fit_pca(panel, n_components=3)
        assert result.explained_variance_ratio[0] > 0.5

    def test_level_factor_has_uniform_sign(self, panel) -> None:
        result = fit_pca(panel, n_components=3)
        pc1 = result.loadings["PC1"].to_numpy()
        assert (pc1 > 0).all()

    def test_slope_factor_changes_sign(self, panel) -> None:
        result = fit_pca(panel, n_components=3)
        pc2 = result.loadings["PC2"].to_numpy()
        assert pc2.min() < 0 < pc2.max()

    def test_loadings_orthonormal(self, panel) -> None:
        result = fit_pca(panel, n_components=3)
        gram = result.loadings.to_numpy().T @ result.loadings.to_numpy()
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)

    def test_reconstruction_shapes_and_error(self, panel) -> None:
        result = fit_pca(panel, n_components=3)
        assert result.reconstruction.shape == panel.shape
        assert result.rmse >= 0
        assert result.rmse < 0.1  # 3 factors explain a Nelson-Siegel panel well

    def test_full_rank_reconstruction_exact(self, panel) -> None:
        k = panel.shape[1]
        result = fit_pca(panel, n_components=k)
        np.testing.assert_allclose(result.reconstruction.to_numpy(), panel.to_numpy(), atol=1e-8)


class TestNMF:
    def test_components_nonnegative(self, panel) -> None:
        result = fit_nmf(panel, n_components=3)
        assert (result.components.to_numpy() >= 0).all()
        assert (result.weights.to_numpy() >= 0).all()

    def test_reconstruction_error_finite_and_small(self, panel) -> None:
        result = fit_nmf(panel, n_components=3)
        assert np.isfinite(result.rmse)
        assert result.rmse < 0.5

    def test_error_decreases_with_components(self, panel) -> None:
        errors = [fit_nmf(panel, n_components=k).rmse for k in (1, 2, 3)]
        assert errors[0] > errors[1] > errors[2]

    def test_rejects_negative_panel(self, panel) -> None:
        bad = panel.copy()
        bad.iloc[0, 0] = -1.0
        with pytest.raises(ValueError, match="nonnegative"):
            fit_nmf(bad)


class TestComparison:
    def test_table_shape(self, panel) -> None:
        table = compare_reconstruction(panel, component_counts=[1, 2, 3])
        assert list(table.index) == [1, 2, 3]
        assert list(table.columns) == ["pca_rmse", "nmf_rmse"]

    def test_pca_beats_nmf_at_matched_counts(self, panel) -> None:
        # PCA (with the mean term) gives the optimal linear rank-k fit, so it
        # should not lose to NMF at the same component count.
        table = compare_reconstruction(panel, component_counts=[2, 3])
        assert (table["pca_rmse"] <= table["nmf_rmse"] + 1e-6).all()
