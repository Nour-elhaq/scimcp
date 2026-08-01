"""Tests for Materials Project integration tools."""

import json

import pytest

from scimcp.tools.materials.materials_project import (
    query_materials_project,
    get_material_details,
    search_stable_materials,
    BUILTIN_MATERIALS,
)


class TestQueryMaterialsProject:
    def test_query_all(self):
        results = query_materials_project()
        assert len(results) >= 10

    def test_query_by_formula(self):
        results = query_materials_project(formula="Si")
        assert len(results) >= 1
        assert any(r["formula"] == "Si" for r in results)

    def test_query_by_material_id(self):
        results = query_materials_project(material_id="mp-149")
        assert len(results) == 1
        assert results[0]["formula"] == "Si"

    def test_query_by_elements(self):
        results = query_materials_project(elements="Ti,O")
        assert len(results) >= 1
        assert all("Ti" in r["elements"] and "O" in r["elements"] for r in results)

    def test_query_metallic_only(self):
        results = query_materials_project(metallic_only=True)
        assert len(results) >= 1
        assert all(r["is_metallic"] for r in results)

    def test_query_band_gap_range(self):
        results = query_materials_project(band_gap_range="1,3")
        assert len(results) >= 1
        assert all(1.0 <= r["band_gap_eV"] <= 3.0 for r in results)

    def test_query_combined(self):
        results = query_materials_project(elements="O", metallic_only=False)
        assert len(results) >= 1

    def test_query_no_results(self):
        results = query_materials_project(material_id="mp-nonexistent")
        assert len(results) == 0


class TestGetMaterialDetails:
    def test_get_si(self):
        result = get_material_details("mp-149")
        assert result is not None
        assert result["formula"] == "Si"
        assert "band_gap_eV" in result

    def test_get_nonexistent(self):
        result = get_material_details("mp-nonexistent")
        assert result is None


class TestSearchStableMaterials:
    def test_search_stable(self):
        results = search_stable_materials(max_e_above_hull=0.0)
        assert len(results) >= 1

    def test_search_with_formula(self):
        results = search_stable_materials(formula="Si", max_e_above_hull=0.0)
        assert len(results) >= 1
