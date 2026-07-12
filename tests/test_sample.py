"""Tests for the committed sample panel and the CSV reader. Fully offline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yieldcurve.data import MATURITY_YEARS, generate_synthetic_yields, read_panel_csv
from yieldcurve.reporting import merge_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "treasury_sample.csv"


class TestCommittedSample:
    def test_sample_exists(self) -> None:
        assert SAMPLE.exists(), "committed sample data/treasury_sample.csv is missing"

    def test_sample_loads_clean(self) -> None:
        panel = read_panel_csv(SAMPLE)
        assert list(panel.columns) == list(MATURITY_YEARS)
        assert len(panel) >= 300
        assert not panel.isna().any().any()
        assert (panel.to_numpy() > 0).all()

    def test_sample_index_sorted_dates(self) -> None:
        panel = read_panel_csv(SAMPLE)
        assert isinstance(panel.index, pd.DatetimeIndex)
        assert panel.index.is_monotonic_increasing


class TestReadPanelCsv:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_panel_csv(tmp_path / "nope.csv")

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Date": ["2024-01-02"], "1 Mo": [5.0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            read_panel_csv(path)

    def test_incomplete_rows_dropped(self, tmp_path: Path) -> None:
        panel = generate_synthetic_yields(n_days=10, seed=0)
        panel.iloc[0, 0] = float("nan")
        path = tmp_path / "gappy.csv"
        panel.to_csv(path)
        loaded = read_panel_csv(path)
        assert len(loaded) == 9

    def test_roundtrip_matches(self, tmp_path: Path) -> None:
        panel = generate_synthetic_yields(n_days=25, seed=1)
        path = tmp_path / "roundtrip.csv"
        panel.to_csv(path)
        loaded = read_panel_csv(path)
        pd.testing.assert_frame_equal(loaded, panel, check_freq=False)


class TestMergeMetrics:
    def test_creates_and_merges(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        merge_metrics(path, {"a": 1})
        merged = merge_metrics(path, {"b": 2})
        assert merged == {"a": 1, "b": 2}

    def test_overwrites_existing_key(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        merge_metrics(path, {"a": 1})
        merged = merge_metrics(path, {"a": 3})
        assert merged["a"] == 3
