"""Tests for geometry sweep plot functions."""

from importlib.util import find_spec

import pytest

if not find_spec("matplotlib"):
    pytest.skip("requires phasesweep[viz] (matplotlib not installed)", allow_module_level=True)

import numpy as np

from phasesweep.vis.plots import plot_sweep_1d, plot_sweep_2d


class TestPlotSweep1D:

    def test_single_metric(self, tmp_path):
        x = [1.0, 2.0, 3.0]
        metrics = {"fundamental": [0.5, 0.6, 0.7]}
        out = plot_sweep_1d(x, metrics, "r_outer (m)", output_dir=tmp_path)
        assert (tmp_path / "sweep_1d.png").exists()
        assert out == str(tmp_path / "sweep_1d.png")

    def test_multiple_metrics(self, tmp_path):
        x = np.linspace(0.04, 0.06, 5)
        metrics = {
            "fundamental": np.linspace(0.4, 0.8, 5),
            "thd_pct": np.linspace(5.0, 3.0, 5),
        }
        out = plot_sweep_1d(
            x, metrics, "r_outer (m)", "B_r metrics",
            title="Test sweep", output_dir=tmp_path, filename="multi.png",
        )
        assert (tmp_path / "multi.png").exists()

    def test_custom_filename(self, tmp_path):
        plot_sweep_1d(
            [1, 2], {"a": [3, 4]}, "x",
            output_dir=tmp_path, filename="custom.png",
        )
        assert (tmp_path / "custom.png").exists()


class TestPlotSweep2D:

    def test_basic_heatmap(self, tmp_path):
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0]
        grid = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        out = plot_sweep_2d(
            x, y, grid, "x_param", "y_param", "metric",
            output_dir=tmp_path,
        )
        assert (tmp_path / "sweep_2d.png").exists()

    def test_no_annotation(self, tmp_path):
        x = [1.0, 2.0]
        y = [1.0, 2.0]
        grid = np.array([[0.1, 0.2], [0.3, 0.4]])
        plot_sweep_2d(
            x, y, grid, "x", "y", "z",
            annotate=False, output_dir=tmp_path,
        )
        assert (tmp_path / "sweep_2d.png").exists()

    def test_with_nan(self, tmp_path):
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        grid = np.array([[0.1, np.nan, 0.3], [0.4, 0.5, 0.6]])
        plot_sweep_2d(
            x, y, grid, "x", "y", "z",
            output_dir=tmp_path, filename="nan_test.png",
        )
        assert (tmp_path / "nan_test.png").exists()
