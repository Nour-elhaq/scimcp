"""Tests for phonon analysis tools."""

import json

import numpy as np
import pytest

from scimcp.tools.analysis.phonon import (
    compute_phonon_dos,
    compute_thermodynamic_properties,
    generate_phonon_band_path,
    estimate_phonon_frequencies,
)


class TestComputePhononDos:
    def test_single_frequency(self):
        result = json.loads(compute_phonon_dos(json.dumps([10.0])))
        assert "frequency_THz" in result
        assert "dos" in result
        assert result["n_modes"] == 1
        assert result["n_imaginary"] == 0

    def test_multiple_frequencies(self):
        freqs = [5.0, 10.0, 15.0, 20.0]
        result = json.loads(compute_phonon_dos(json.dumps(freqs)))
        assert result["n_modes"] == 4
        assert result["min_freq_THz"] == 5.0
        assert result["max_freq_THz"] == 20.0

    def test_imaginary_frequencies(self):
        freqs = [-2.0, 5.0, 10.0]
        result = json.loads(compute_phonon_dos(json.dumps(freqs)))
        assert result["n_imaginary"] == 1

    def test_dos_shape(self):
        result = json.loads(compute_phonon_dos(json.dumps([10.0]), n_points=100))
        assert len(result["frequency_THz"]) == 100
        assert len(result["dos"]) == 100

    def test_dos_positive(self):
        result = json.loads(compute_phonon_dos(json.dumps([10.0, 15.0, 20.0])))
        assert all(d >= 0 for d in result["dos"])


class TestComputeThermodynamicProperties:
    def test_at_zero_temp(self):
        result = json.loads(compute_thermodynamic_properties(
            json.dumps([10.0, 15.0, 20.0]),
            temperature_K=0.0,
        ))
        assert result["temperature_K"] == 0.0
        assert result["ZPE_eV"] == 0.0

    def test_at_room_temp(self):
        result = json.loads(compute_thermodynamic_properties(
            json.dumps([10.0, 15.0, 20.0]),
            temperature_K=300.0,
            n_atoms=1,
        ))
        assert result["temperature_K"] == 300.0
        assert result["ZPE_eV"] > 0
        assert result["Helmholtz_eV"] != 0
        assert result["entropy_J_mol_K"] >= 0
        assert result["Cv_J_mol_K"] >= 0

    def test_imaginary_frequencies_excluded(self):
        result = json.loads(compute_thermodynamic_properties(
            json.dumps([-5.0, 10.0, 15.0]),
            temperature_K=300.0,
        ))
        assert result["n_imaginary"] == 1
        assert result["n_modes_positive"] == 2

    def test_high_temp_limit(self):
        result_300 = json.loads(compute_thermodynamic_properties(
            json.dumps([10.0]), temperature_K=300.0,
        ))
        result_1000 = json.loads(compute_thermodynamic_properties(
            json.dumps([10.0]), temperature_K=1000.0,
        ))
        # At higher T, entropy should be higher
        assert result_1000["entropy_J_mol_K"] >= result_300["entropy_J_mol_K"]


class TestGeneratePhononBandPath:
    def test_cubic_path(self):
        result = json.loads(generate_phonon_band_path({"a": 5.43}))
        assert result["crystal_system"] == "cubic"
        assert "Gamma" in result["high_symmetry_points"]
        assert "X" in result["high_symmetry_points"]
        assert result["n_points"] > 0

    def test_hexagonal_path(self):
        result = json.loads(generate_phonon_band_path(
            {"a": 3.0, "c": 5.0}, crystal_system="hexagonal",
        ))
        assert result["crystal_system"] == "hexagonal"
        assert "K" in result["high_symmetry_points"]

    def test_tetragonal_path(self):
        result = json.loads(generate_phonon_band_path(
            {"a": 4.59, "c": 2.96}, crystal_system="tetragonal",
        ))
        assert result["crystal_system"] == "tetragonal"
        assert "Z" in result["high_symmetry_points"]

    def test_n_points(self):
        result = json.loads(generate_phonon_band_path({"a": 5.0}, n_points=30))
        assert result["n_points"] > 30

    def test_labels_match_kpoints(self):
        result = json.loads(generate_phonon_band_path({"a": 5.0}))
        assert len(result["k_points"]) == len(result["labels"])


class TestEstimatePhononFrequencies:
    def test_si(self):
        result = json.loads(estimate_phonon_frequencies("Si"))
        assert result["composition"] == "Si"
        assert result["n_atoms_unit_cell"] == 1
        assert result["n_acoustic_modes"] == 3
        assert result["n_optical_modes"] == 0
        assert result["estimated_max_freq_THz"] > 0

    def test_gaas(self):
        result = json.loads(estimate_phonon_frequencies("GaAs"))
        assert result["n_atoms_unit_cell"] == 2
        assert result["n_optical_modes"] == 3

    def test_tio2(self):
        result = json.loads(estimate_phonon_frequencies("TiO2"))
        assert result["n_atoms_unit_cell"] == 3
        assert result["estimated_max_freq_THz"] > 0

    def test_unknown_element(self):
        result = json.loads(estimate_phonon_frequencies("Xx"))
        assert "error" in result
