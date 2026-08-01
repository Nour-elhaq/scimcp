"""Tests for trajectory visualization tools."""

import json
import os

import pytest

from scimcp.tools.analysis.visualization import (
    plot_time_series,
    plot_histogram,
    plot_scatter,
    plot_phonon_dos,
    plot_thermo_dashboard,
)


class TestPlotTimeSeries:
    def test_basic_plot(self):
        data = {"x": [1, 2, 3, 4], "y": [10, 20, 15, 25]}
        result = json.loads(plot_time_series(json.dumps(data)))
        assert result["shape"] == "time_series"
        assert result["n_points"] == 4

    def test_dict_of_arrays(self):
        data = {"Temp": [300, 310, 305], "Press": [1, 1.1, 0.9]}
        result = json.loads(plot_time_series(json.dumps(data)))
        assert result["shape"] == "time_series"

    def test_save_file(self, tmp_path):
        out = tmp_path / "plot.png"
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        result = json.loads(plot_time_series(json.dumps(data), output_file=str(out)))
        assert out.exists()
        assert result["file"] == str(out)

    def test_base64(self):
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        result = json.loads(plot_time_series(json.dumps(data), return_base64=True))
        assert "base64" in result
        assert len(result["base64"]) > 100


class TestPlotHistogram:
    def test_basic_histogram(self):
        values = list(range(100))
        result = json.loads(plot_histogram(json.dumps(values)))
        assert result["shape"] == "histogram"
        assert "stats" in result
        assert result["stats"]["mean"] == pytest.approx(49.5, abs=0.1)

    def test_custom_bins(self):
        values = [1, 2, 3, 4, 5]
        result = json.loads(plot_histogram(json.dumps(values), n_bins=10))
        assert result["n_bins"] == 10

    def test_save_file(self, tmp_path):
        out = tmp_path / "hist.png"
        result = json.loads(plot_histogram(json.dumps([1, 2, 3]), output_file=str(out)))
        assert out.exists()

    def test_base64(self):
        result = json.loads(plot_histogram(json.dumps([1, 2, 3]), return_base64=True))
        assert "base64" in result


class TestPlotScatter:
    def test_basic_scatter(self):
        result = json.loads(plot_scatter(
            json.dumps([1, 2, 3]),
            json.dumps([4, 5, 6]),
        ))
        assert result["shape"] == "scatter"
        assert result["n_points"] == 3

    def test_with_color(self):
        result = json.loads(plot_scatter(
            json.dumps([1, 2, 3]),
            json.dumps([4, 5, 6]),
            color_json=json.dumps([0.1, 0.5, 0.9]),
        ))
        assert result["shape"] == "scatter"

    def test_save_file(self, tmp_path):
        out = tmp_path / "scatter.png"
        result = json.loads(plot_scatter(
            json.dumps([1, 2, 3]),
            json.dumps([4, 5, 6]),
            output_file=str(out),
        ))
        assert out.exists()


class TestPlotPhononDos:
    def test_basic_dos_plot(self):
        freqs = list(range(-5, 50))
        dos = [max(0, 10 - abs(f - 20)) for f in freqs]
        result = json.loads(plot_phonon_dos(json.dumps(freqs), json.dumps(dos)))
        assert result["shape"] == "phonon_dos"

    def test_save_file(self, tmp_path):
        out = tmp_path / "dos.png"
        result = json.loads(plot_phonon_dos(
            json.dumps([1, 2, 3]),
            json.dumps([0.5, 1.0, 0.3]),
            output_file=str(out),
        ))
        assert out.exists()


class TestPlotThermoDashboard:
    def test_basic_dashboard(self):
        data = {
            "Step": [1, 2, 3, 4],
            "Temp": [300, 310, 305, 308],
            "PotEng": [-100, -101, -100.5, -100.8],
            "Press": [1.0, 1.1, 0.9, 1.0],
        }
        result = json.loads(plot_thermo_dashboard(json.dumps(data)))
        assert result["shape"] == "thermo_dashboard"
        assert len(result["panels"]) == 3

    def test_missing_columns(self):
        data = {"Step": [1, 2]}
        result = json.loads(plot_thermo_dashboard(json.dumps(data)))
        assert "error" in result

    def test_save_file(self, tmp_path):
        out = tmp_path / "dashboard.png"
        data = {"Step": [1, 2], "Temp": [300, 310]}
        result = json.loads(plot_thermo_dashboard(json.dumps(data), output_file=str(out)))
        assert out.exists()
