"""Tests for shear rate sweep tools."""

import json
import pytest
import numpy as np

from scimcp.tools.lammps.shear import (
    generate_sweep_scripts,
    write_sweep_scripts,
    estimate_viscosity,
    SweepConfig,
)


class TestGenerateSweepScripts:
    def test_creates_one_per_rate(self):
        config = SweepConfig(shear_rates=[0.001, 0.01, 0.1])
        scripts = generate_sweep_scripts(config)
        assert len(scripts) == 3

    def test_each_script_has_shear_fix(self):
        config = SweepConfig(shear_rates=[0.001, 0.01])
        scripts = generate_sweep_scripts(config)
        for rate_str, script in scripts.items():
            assert "deform" in script
            assert "erate" in script

    def test_rate_appears_in_script(self):
        config = SweepConfig(shear_rates=[0.0042])
        scripts = generate_sweep_scripts(config)
        assert "0.0042" in scripts["0.0042"]


class TestWriteSweepScripts:
    def test_writes_files(self, tmp_path):
        config = SweepConfig(
            shear_rates=[0.001, 0.01],
            output_dir=str(tmp_path / "sweep"),
        )
        file_map = write_sweep_scripts(config)
        assert len(file_map) >= 2

    def test_writes_config_json(self, tmp_path):
        config = SweepConfig(
            shear_rates=[0.001],
            output_dir=str(tmp_path / "sweep2"),
        )
        file_map = write_sweep_scripts(config)
        assert "config" in file_map
        with open(file_map["config"]) as f:
            meta = json.load(f)
        assert "shear_rates" in meta


class TestEstimateViscosity:
    def test_newtonian(self):
        """Linear stress-rate → n ≈ 1 (Newtonian)."""
        rates = np.array([0.001, 0.01, 0.1])
        stresses = rates * 10.0  # η = 10
        result = estimate_viscosity(rates, stresses)
        assert abs(result["power_law_n"] - 1.0) < 0.1
        assert abs(result["power_law_K"] - 10.0) < 1.0

    def test_shear_thinning(self):
        """Stress grows slower than rate → n < 1."""
        rates = np.array([0.001, 0.01, 0.1])
        stresses = rates ** 0.7
        result = estimate_viscosity(rates, stresses)
        assert result["is_shear_thinning"] is True
        assert result["power_law_n"] < 1.0

    def test_shear_thickening(self):
        """Stress grows faster than rate → n > 1."""
        rates = np.array([0.001, 0.01, 0.1])
        stresses = rates ** 1.5
        result = estimate_viscosity(rates, stresses)
        assert result["is_shear_thickening"] is True
        assert result["power_law_n"] > 1.0
