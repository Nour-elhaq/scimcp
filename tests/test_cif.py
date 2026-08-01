"""Tests for CIF parsing and generation tools."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from scimcp.tools.dft.cif import (
    parse_cif,
    generate_cif,
    get_cif_summary,
    cif_to_ase,
    _parse_cif_text,
    _extract_value,
    _lattice_to_vectors,
)


SIMPLE_CIF = """\
data_generated
_symmetry_space_group_name_H-M   'P m -3 m'

_cell_length_a       5.430000
_cell_length_b       5.430000
_cell_length_c       5.430000
_cell_angle_alpha    90.000000
_cell_angle_beta     90.000000
_cell_angle_gamma    90.000000

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
  Si1  Si  0.000000  0.000000  0.000000
  Si2  Si  0.250000  0.250000  0.250000
"""

SIO2_CIF = """\
data_quartz
_symmetry_space_group_name_H-M   'P 31 2 1'

_cell_length_a       4.913000
_cell_length_b       4.913000
_cell_length_c       5.405000
_cell_angle_alpha    90.000000
_cell_angle_beta     90.000000
_cell_angle_gamma    120.000000

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
  Si1  Si  0.469700  0.000000  0.000000
  O1   O   0.413500  0.266900  0.119100
"""


class TestParseCifText:
    def test_parse_simple_cif(self):
        result = _parse_cif_text(SIMPLE_CIF)
        assert "lattice" in result
        assert result["lattice"]["a"] == pytest.approx(5.43, abs=0.01)
        assert result["lattice"]["b"] == pytest.approx(5.43, abs=0.01)
        assert result["lattice"]["c"] == pytest.approx(5.43, abs=0.01)

    def test_parse_atoms(self):
        result = _parse_cif_text(SIMPLE_CIF)
        assert len(result["atoms"]) == 2
        assert result["atoms"][0]["element"] == "Si"
        assert result["atoms"][1]["element"] == "Si"

    def test_parse_symmetry(self):
        result = _parse_cif_text(SIMPLE_CIF)
        assert len(result["symmetry"]) > 0

    def test_parse_sio2(self):
        result = _parse_cif_text(SIO2_CIF)
        assert len(result["atoms"]) == 2
        elements = [a["element"] for a in result["atoms"]]
        assert "Si" in elements
        assert "O" in elements

    def test_parse_noncubic_lattice(self):
        result = _parse_cif_text(SIO2_CIF)
        assert result["lattice"]["a"] == pytest.approx(4.913, abs=0.01)
        assert result["lattice"]["c"] == pytest.approx(5.405, abs=0.01)
        assert result["lattice"]["gamma"] == pytest.approx(120.0, abs=0.1)

    def test_empty_content(self):
        result = _parse_cif_text("")
        assert result["lattice"] == {}
        assert result["atoms"] == []


class TestExtractValue:
    def test_simple_value(self):
        assert _extract_value("_cell_length_a 5.43") == pytest.approx(5.43)

    def test_value_with_uncertainty(self):
        assert _extract_value("_cell_length_a 5.4300(3)") == pytest.approx(5.43)

    def test_missing_value(self):
        assert _extract_value("_cell_length_a") == 0.0


class TestLatticeToVectors:
    def test_cubic(self):
        lattice = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 90}
        vectors = _lattice_to_vectors(lattice)
        assert vectors.shape == (3, 3)
        assert vectors[0, 0] == pytest.approx(5.0, abs=0.01)
        assert vectors[1, 1] == pytest.approx(5.0, abs=0.01)
        assert vectors[2, 2] == pytest.approx(5.0, abs=0.01)

    def test_hexagonal(self):
        lattice = {"a": 3.0, "b": 3.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 120}
        vectors = _lattice_to_vectors(lattice)
        assert vectors.shape == (3, 3)
        assert vectors[0, 0] == pytest.approx(3.0, abs=0.01)


class TestParseCifFile:
    def test_parse_cif_file(self, tmp_path):
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(SIMPLE_CIF)
        result = parse_cif(str(cif_file))
        assert "lattice" in result
        assert result["lattice"]["a"] == pytest.approx(5.43, abs=0.01)
        assert len(result["atoms"]) == 2

    def test_parse_cif_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_cif("nonexistent_file.cif")


class TestGenerateCif:
    def test_generate_basic_cif(self):
        result = generate_cif(
            elements=["Si", "Si"],
            positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        )
        assert "data_generated" in result
        assert "_cell_length_a" in result
        assert "Si" in result

    def test_generate_cif_with_params(self):
        result = generate_cif(
            elements=["Ti", "C"],
            positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            lattice_params={"a": 3.0, "b": 3.0, "c": 3.0, "alpha": 90, "beta": 90, "gamma": 90},
            space_group="P m -3 m",
            label="TiC",
        )
        assert "data_TiC" in result
        assert "P m -3 m" in result
        assert "3.000000" in result

    def test_generate_cif_with_output_file(self, tmp_path):
        output_file = tmp_path / "generated.cif"
        result = generate_cif(
            elements=["Fe"],
            positions=[[0.0, 0.0, 0.0]],
            output_file=str(output_file),
        )
        assert output_file.exists()
        assert output_file.read_text() == result


class TestGetCifSummary:
    def test_summary_simple_cif(self, tmp_path):
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(SIMPLE_CIF)
        summary = get_cif_summary(str(cif_file))
        assert summary["n_atoms"] == 2
        assert "Si" in summary["elements"]
        assert summary["formula"] == "Si2"

    def test_summary_sio2(self, tmp_path):
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(SIO2_CIF)
        summary = get_cif_summary(str(cif_file))
        assert summary["n_atoms"] == 2
        assert set(summary["elements"]) == {"Si", "O"}


class TestCifToAse:
    def test_cif_to_ase_fallback(self, tmp_path):
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(SIMPLE_CIF)
        result = cif_to_ase(str(cif_file))
        assert "symbols" in result
        assert "positions" in result
        assert "cell" in result
        assert result["n_atoms"] >= 2
