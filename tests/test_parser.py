"""Tests for LAMMPS output parsers."""

import pytest
import numpy as np

from scimcp.tools.lammps.parser import (
    parse_thermo_data,
    parse_dump_file,
    get_summary,
)


@pytest.fixture
def thermo_file(tmp_path):
    """Create a sample LAMMPS thermo output file."""
    content = """\
LAMMPS output
Step Temp PotEng TotEng Press
     0    300.0   -5.234   -4.892   12.345
   100    298.5   -5.241   -4.898   11.987
   200    301.2   -5.228   -4.885   12.102
   300    299.8   -5.235   -4.893   12.056
"""
    filepath = tmp_path / "thermo.dat"
    filepath.write_text(content)
    return str(filepath)


@pytest.fixture
def dump_file(tmp_path):
    """Create a sample LAMMPS dump file."""
    content = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 15.78
0 15.78
0 15.78
ITEM: ATOMS id type x y z
1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
3 1 7.0 8.0 9.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 15.78
0 15.78
0 15.78
ITEM: ATOMS id type x y z
1 1 1.1 2.1 3.1
2 1 4.1 5.1 6.1
3 1 7.1 8.1 9.1
"""
    filepath = tmp_path / "dump.lammpstrj"
    filepath.write_text(content)
    return str(filepath)


class TestParseThermoData:
    def test_reads_headers(self, thermo_file):
        result = parse_thermo_data(thermo_file)
        assert "Step" in result["headers"]
        assert "Temp" in result["headers"]
        assert "PotEng" in result["headers"]

    def test_reads_data(self, thermo_file):
        result = parse_thermo_data(thermo_file)
        assert result["n_steps"] == 4

    def test_columns_dict(self, thermo_file):
        result = parse_thermo_data(thermo_file)
        assert "Temp" in result["columns"]
        assert len(result["columns"]["Temp"]) == 4

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_thermo_data("/nonexistent/file.dat")


class TestParseDumpFile:
    def test_reads_frames(self, dump_file):
        result = parse_dump_file(dump_file)
        assert result["n_frames"] == 2

    def test_reads_n_atoms(self, dump_file):
        result = parse_dump_file(dump_file)
        assert result["n_atoms"] == 3

    def test_frame_has_columns(self, dump_file):
        result = parse_dump_file(dump_file)
        frame = result["frames"][0]
        assert "id" in frame["columns"]
        assert "x" in frame["columns"]

    def test_frame_has_box(self, dump_file):
        result = parse_dump_file(dump_file)
        frame = result["frames"][0]
        assert len(frame["box"]) == 3

    def test_max_frames(self, dump_file):
        result = parse_dump_file(dump_file, max_frames=1)
        assert result["n_frames"] == 1

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_dump_file("/nonexistent/file.lammpstrj")


class TestGetSummary:
    def test_thermo_summary(self, thermo_file):
        summary = get_summary(thermo_file)
        assert summary["is_thermo"] is True
        assert summary["n_steps"] == 4

    def test_dump_summary(self, dump_file):
        summary = get_summary(dump_file)
        assert summary["is_dump"] is True
        assert summary["n_frames"] == 2
        assert summary["n_atoms"] == 3

    def test_file_size(self, thermo_file):
        summary = get_summary(thermo_file)
        assert summary["file_size_bytes"] > 0
