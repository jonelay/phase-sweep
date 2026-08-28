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


# ---------------------------------------------------------------------------
# Content assertions — capture the figure through _save
# so the tests see artists, not just a PNG on disk.
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_fig(monkeypatch):
    import phasesweep.vis.plots as plots
    real = plots._save
    box = {}

    def spy(fig, output_dir, filename):
        box["fig"] = fig
        return real(fig, output_dir, filename)

    monkeypatch.setattr(plots, "_save", spy)
    return box


class TestPlotSweep1DContent:

    def test_line_data_and_no_legend_for_single_metric(
            self, tmp_path, captured_fig):
        x = [1.0, 2.0, 3.0]
        vals = [0.5, 0.6, 0.7]
        plot_sweep_1d(x, vals_dict := {"fundamental": vals}, "x",
                      output_dir=tmp_path)
        ax = captured_fig["fig"].axes[0]
        assert len(ax.lines) == len(vals_dict)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), vals)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), x)
        assert ax.get_legend() is None
        assert ax.get_ylabel() == "fundamental"

    def test_multi_metric_gets_legend_and_all_lines(
            self, tmp_path, captured_fig):
        x = np.linspace(0.0, 1.0, 5)
        metrics = {"a": np.linspace(0, 1, 5), "b": np.linspace(1, 0, 5)}
        plot_sweep_1d(x, metrics, "x", output_dir=tmp_path)
        ax = captured_fig["fig"].axes[0]
        assert len(ax.lines) == 2
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


class TestPlotSweep2DContent:

    def test_annotations_match_finite_cells(self, tmp_path, captured_fig):
        grid = np.array([[0.1, np.nan, 0.3], [0.4, 0.5, 0.6]])
        plot_sweep_2d([1.0, 2.0, 3.0], [1.0, 2.0], grid, "x", "y", "z",
                      output_dir=tmp_path)
        ax = captured_fig["fig"].axes[0]
        # one text per finite cell — the NaN cell must be masked, not "nan"
        assert len(ax.texts) == np.isfinite(grid).sum() == 5
        assert not any("nan" in t.get_text() for t in ax.texts)

    def test_annotate_false_draws_no_cell_text(self, tmp_path, captured_fig):
        grid = np.array([[0.1, 0.2], [0.3, 0.4]])
        plot_sweep_2d([1.0, 2.0], [1.0, 2.0], grid, "x", "y", "z",
                      annotate=False, output_dir=tmp_path)
        ax = captured_fig["fig"].axes[0]
        assert len(ax.texts) == 0

    def test_heatmap_mesh_and_colorbar_axes(self, tmp_path, captured_fig):
        from matplotlib.collections import QuadMesh
        grid = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        plot_sweep_2d([1.0, 2.0, 3.0], [10.0, 20.0], grid, "x", "y", "z",
                      output_dir=tmp_path)
        fig = captured_fig["fig"]
        assert len(fig.axes) == 2  # heatmap + colorbar
        meshes = [c for c in fig.axes[0].collections
                  if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        np.testing.assert_allclose(
            np.asarray(meshes[0].get_array()).reshape(grid.shape), grid)


class TestPlotGeometry:
    """Smoke tests for the cross-section renderer: region paint order per
    topology, magnet-arc and slot wedge counts (this is
    the only place the arc/slot rendering math lives)."""

    STATOR, ROTOR = "#4488cc", "#888888"
    MAGNET, AIRGAP = "#cc4444", "#ffffff"

    @staticmethod
    def _patch_colors(ax, kind):
        return [p.get_facecolor() for p in ax.patches
                if type(p).__name__ == kind]

    def test_inrunner_region_order_and_counts(self, tmp_path, captured_fig):
        from matplotlib.colors import to_rgba

        from phasesweep.machines.geometry import default_inrunner
        from phasesweep.vis.plots import plot_geometry

        out = plot_geometry(default_inrunner(), n_p=4, alpha_p=0.75,
                            output_dir=tmp_path)
        assert (tmp_path / "geometry.png").exists()
        assert out == str(tmp_path / "geometry.png")
        ax = captured_fig["fig"].axes[0]
        circles = self._patch_colors(ax, "Circle")
        # painter's order outside-in: stator, airgap, magnets, rotor,
        # then the unfilled r_ag marker circle
        expected = [self.STATOR, self.AIRGAP, self.MAGNET, self.ROTOR]
        assert len(circles) == len(expected) + 1
        for got, want in zip(circles, expected):
            assert got == to_rgba(want)
        # 2*n_p interpole-gap wedges for alpha_p < 1, no slots drawn
        wedges = [p for p in ax.patches if type(p).__name__ == "Wedge"]
        assert len(wedges) == 8

    def test_outrunner_region_order(self, tmp_path, captured_fig):
        from matplotlib.colors import to_rgba

        from phasesweep.machines.geometry import outrunner
        from phasesweep.vis.plots import plot_geometry

        geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                        r_stator=0.04, r_inner=0.02)
        plot_geometry(geo, n_p=4, output_dir=tmp_path)
        ax = captured_fig["fig"].axes[0]
        circles = self._patch_colors(ax, "Circle")
        # outside-in: rotor iron first, stator fourth, shaft fifth
        assert circles[0] == to_rgba(self.ROTOR)
        assert circles[1] == to_rgba(self.MAGNET)
        assert circles[2] == to_rgba(self.AIRGAP)
        assert circles[3] == to_rgba(self.STATOR)
        assert len(circles) == 6  # 5 regions + r_ag marker
        # alpha_p = 1.0 -> no gap wedges
        assert not [p for p in ax.patches if type(p).__name__ == "Wedge"]

    def test_slotted_inrunner_draws_slot_wedges(self, tmp_path, captured_fig):
        from phasesweep.machines.geometry import inrunner
        from phasesweep.vis.plots import plot_geometry

        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64,
                       r_rotor=0.30, r_inner=0.0, n_slots=12,
                       slot_depth=0.05)
        plot_geometry(geo, n_p=4, output_dir=tmp_path)
        ax = captured_fig["fig"].axes[0]
        wedges = [p for p in ax.patches if type(p).__name__ == "Wedge"]
        # equal body/opening ratio -> one un-stepped wedge per slot
        assert len(wedges) == geo.n_slots
