"""Integration tests for all Phase 2 MCP tools."""

import json

import pytest

from scimcp.server import mcp


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


def _call_tool(name: str, args: dict) -> dict:
    """Call an MCP tool and return parsed JSON result."""
    tools = mcp._tool_manager._tools
    assert name in tools, f"Tool {name} not found"
    tool = tools[name]
    result = tool.fn(**args)
    return json.loads(result)


class TestDftParseCif:
    def test_parse_cif(self):
        result = _call_tool("dft_parse_cif", {"cif_content": SIMPLE_CIF})
        assert "lattice" in result
        assert "n_atoms" in result
        assert result["n_atoms"] == 2

    def test_parse_cif_has_space_group(self):
        result = _call_tool("dft_parse_cif", {"cif_content": SIMPLE_CIF})
        assert "space_group" in result


class TestDftGenerateCif:
    def test_generate_cif(self):
        result = _call_tool("dft_generate_cif", {
            "space_group": 225,
            "lattice_params": '{"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90}',
            "atom_types": '["Si", "Si"]',
            "positions": '[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]',
            "formula": "Si2",
        })
        assert "cif_content" in result
        assert "Si" in result["cif_content"]


class TestDftCifSummary:
    def test_cif_summary(self):
        result = _call_tool("dft_cif_summary", {"cif_content": SIMPLE_CIF})
        assert "lattice" in result
        assert "n_atoms" in result
        assert result["n_atoms"] == 2
        assert "Si" in result["elements"]


class TestMaterialsMxeneList:
    def test_list_mxenes(self):
        result = _call_tool("materials_mxene_list", {})
        assert isinstance(result, list)
        assert len(result) >= 10
        assert "Ti3C2" in result


class TestMaterialsMxeneQuery:
    def test_query_all(self):
        result = _call_tool("materials_mxene_query", {})
        assert isinstance(result, list)
        assert len(result) >= 10

    def test_query_by_formula(self):
        result = _call_tool("materials_mxene_query", {"formula": "Ti3C2"})
        assert len(result) >= 1

    def test_query_by_m_element(self):
        result = _call_tool("materials_mxene_query", {"m_element": "V"})
        assert len(result) >= 1

    def test_query_metallic_only(self):
        result = _call_tool("materials_mxene_query", {"metallic_only": True})
        assert len(result) >= 1


class TestMaterialsMxeneProperties:
    def test_get_properties(self):
        result = _call_tool("materials_mxene_properties", {"formula": "Ti3C2"})
        assert "formula" in result
        assert result["formula"] == "Ti3C2"


class TestMaterialsMxeneCompare:
    def test_compare(self):
        result = _call_tool("materials_mxene_compare", {
            "formulas": '["Ti3C2", "V2C"]'
        })
        assert "Ti3C2" in result
        assert "V2C" in result


class TestMaterialsMxeneSearch:
    def test_search_band_gap(self):
        result = _call_tool("materials_mxene_search", {
            "property_name": "band_gap_eV",
            "min_value": 0.0,
            "max_value": 0.0,
        })
        assert isinstance(result, list)
        assert len(result) >= 1


class TestMaterialsPredictBandGap:
    def test_predict_si(self):
        result = _call_tool("materials_predict_band_gap", {"composition": "Si"})
        assert "predicted_band_gap_eV" in result
        assert result["formula"] == "Si"

    def test_predict_gaas(self):
        result = _call_tool("materials_predict_band_gap", {"composition": "GaAs"})
        assert "predicted_band_gap_eV" in result


class TestMaterialsPredictDensity:
    def test_predict_density(self):
        result = _call_tool("materials_predict_density", {"composition": "Si"})
        assert "estimated_density_g_cm3" in result
        assert result["estimated_density_g_cm3"] > 0


class TestMaterialsPredictMeltingPoint:
    def test_predict_melting(self):
        result = _call_tool("materials_predict_melting_point", {"composition": "Si"})
        assert "estimated_melting_point_K" in result
        assert result["estimated_melting_point_K"] > 0


class TestMaterialsElementInfo:
    def test_element_info(self):
        result = _call_tool("materials_element_info", {"element": "Si"})
        assert result is not None
        assert result["Z"] == 14

    def test_unknown_element(self):
        result = _call_tool("materials_element_info", {"element": "Xx"})
        assert result is None


class TestMaterialsPredictAll:
    def test_predict_all(self):
        result = _call_tool("materials_predict_all", {"composition": "Si"})
        assert "band_gap" in result
        assert "density" in result
        assert "melting_point" in result
        assert "composition_features" in result


class TestMaterialsCompositionFeatures:
    def test_features(self):
        result = _call_tool("materials_composition_features", {"composition": "GaAs"})
        assert "features" in result
        assert "mean_en" in result["features"]


class TestLiteratureSearch:
    @pytest.mark.network
    def test_arxiv_search(self):
        result = _call_tool("literature_search_arxiv", {
            "query": "MXene electronic structure",
            "max_results": 3,
        })
        assert "papers" in result

    @pytest.mark.network
    def test_materials_search(self):
        result = _call_tool("literature_search_materials", {
            "topic": "MXene",
            "max_results": 3,
        })
        assert "papers" in result

    @pytest.mark.network
    def test_author_search(self):
        result = _call_tool("literature_search_by_author", {
            "author_name": "Naguib",
            "max_results": 3,
        })
        assert "papers" in result
