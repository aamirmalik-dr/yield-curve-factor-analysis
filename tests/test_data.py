"""Tests for the data module. All tests run fully offline."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from yieldcurve.data import MATURITY_YEARS, generate_synthetic_yields, load_yield_curve


class TestSyntheticGenerator:
    def test_shape_and_columns(self) -> None:
        panel = generate_synthetic_yields(n_days=100, seed=1)
        assert panel.shape == (100, len(MATURITY_YEARS))
        assert list(panel.columns) == list(MATURITY_YEARS)

    def test_values_positive(self) -> None:
        panel = generate_synthetic_yields(n_days=500, seed=2)
        assert (panel.to_numpy() > 0).all()

    def test_reproducible(self) -> None:
        a = generate_synthetic_yields(n_days=50, seed=3)
        b = generate_synthetic_yields(n_days=50, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = generate_synthetic_yields(n_days=50, seed=4)
        b = generate_synthetic_yields(n_days=50, seed=5)
        assert not a.equals(b)

    def test_index_is_business_days(self) -> None:
        panel = generate_synthetic_yields(n_days=30)
        assert isinstance(panel.index, pd.DatetimeIndex)
        assert (panel.index.dayofweek < 5).all()


class TestLoaderFallback:
    def test_falls_back_to_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        cached = generate_synthetic_yields(n_days=40, seed=6)
        cache_path = tmp_path / "cache.csv"
        cached.to_csv(cache_path)

        def fail(*args, **kwargs):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr("yieldcurve.data.download_treasury_yields", fail)
        panel, source = load_yield_curve(cache_path=cache_path, verbose=False)
        assert source == "cache"
        assert len(panel) == 40

    def test_falls_back_to_synthetic(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        def fail(*args, **kwargs):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr("yieldcurve.data.download_treasury_yields", fail)
        panel, source = load_yield_curve(cache_path=tmp_path / "missing.csv", verbose=False)
        assert source == "synthetic"
        assert not panel.empty

    def test_raises_when_synthetic_disallowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        def fail(*args, **kwargs):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr("yieldcurve.data.download_treasury_yields", fail)
        with pytest.raises(RuntimeError):
            load_yield_curve(
                cache_path=tmp_path / "missing.csv", allow_synthetic=False, verbose=False
            )

    def test_successful_download_writes_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = generate_synthetic_yields(n_days=20, seed=7)

        monkeypatch.setattr("yieldcurve.data.download_treasury_yields", lambda *a, **k: fake)
        cache_path = tmp_path / "sub" / "cache.csv"
        panel, source = load_yield_curve(cache_path=cache_path, verbose=False)
        assert source == "treasury"
        assert cache_path.exists()
        assert len(panel) == 20
