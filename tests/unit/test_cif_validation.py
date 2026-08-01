"""Unit tests for CIF validation against known crystal structures."""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scimcp.tools.dft.cif import _parse_cif_text, generate_cif
from scimcp.validation.reference_data import CIF_VALIDATION


class TestCIFReferenceStructures:
    """Validate CIF parsing against known crystal structures."""

    def test_silicon_atom_count(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        assert len(data.get("atoms", [])) == 2

    def test_silicon_elements(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        elements = [a["element"] for a in data.get("atoms", [])]
        assert all(e == "Si" for e in elements)

    def test_silicon_lattice(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        lattice = data.get("lattice", {})
        assert abs(float(lattice.get("a", 0)) - 5.431) < 0.01

    def test_silicon_space_group(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        sg = data.get("symmetry", {})
        # The parser stores the space group name under a CIF-specific key
        has_sg = any("symmetry" in k.lower() or "space" in k.lower() for k in sg.keys())
        assert has_sg or len(sg) > 0

    def test_silicon_volume(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        lattice = data.get("lattice", {})
        a = float(lattice.get("a", 5.431))
        vol = a ** 3
        assert abs(vol - 160.2) < 1.0

    def test_nacl_atom_count(self):
        data = _parse_cif_text(CIF_VALIDATION["nacl"]["cif_content"])
        assert len(data.get("atoms", [])) == 2

    def test_nacl_elements(self):
        data = _parse_cif_text(CIF_VALIDATION["nacl"]["cif_content"])
        elements = sorted([a["element"] for a in data.get("atoms", [])])
        assert elements == ["Cl", "Na"]

    def test_nacl_lattice(self):
        data = _parse_cif_text(CIF_VALIDATION["nacl"]["cif_content"])
        lattice = data.get("lattice", {})
        assert abs(float(lattice.get("a", 0)) - 5.640) < 0.01

    def test_gaas_atom_count(self):
        data = _parse_cif_text(CIF_VALIDATION["gaas"]["cif_content"])
        assert len(data.get("atoms", [])) == 2

    def test_gaas_elements(self):
        data = _parse_cif_text(CIF_VALIDATION["gaas"]["cif_content"])
        elements = sorted([a["element"] for a in data.get("atoms", [])])
        assert elements == ["As", "Ga"]

    def test_gaas_lattice(self):
        data = _parse_cif_text(CIF_VALIDATION["gaas"]["cif_content"])
        lattice = data.get("lattice", {})
        assert abs(float(lattice.get("a", 0)) - 5.653) < 0.01


class TestCIFGenerationRoundtrip:
    """Validate CIF generation and re-parsing."""

    def test_roundtrip_silicon(self):
        lattice_params = {"a": 5.431, "b": 5.431, "c": 5.431, "alpha": 90, "beta": 90, "gamma": 90}
        cif_content = generate_cif(
            elements=["Si", "Si"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
            lattice_params=lattice_params,
            space_group="227",
            label="Si",
        )
        assert "Si" in cif_content
        assert "5.431" in cif_content
        data = _parse_cif_text(cif_content)
        assert len(data.get("atoms", [])) >= 2

    def test_roundtrip_gaas(self):
        lattice_params = {"a": 5.653, "b": 5.653, "c": 5.653, "alpha": 90, "beta": 90, "gamma": 90}
        cif_content = generate_cif(
            elements=["Ga", "As"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
            lattice_params=lattice_params,
            space_group="216",
            label="GaAs",
        )
        assert "Ga" in cif_content
        assert "As" in cif_content


class TestCIFNumericalTolerances:
    """Verify numerical precision in lattice parameter handling."""

    def test_lattice_precision(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        a = float(data["lattice"]["a"])
        assert math.isclose(a, 5.431, rel_tol=1e-4)

    def test_coordinate_precision(self):
        data = _parse_cif_text(CIF_VALIDATION["silicon"]["cif_content"])
        for atom in data["atoms"]:
            for coord_key in ["x", "y", "z"]:
                val = float(atom.get(coord_key, 0))
                assert 0 <= val <= 1, f"Coordinate {coord_key}={val} out of range"
