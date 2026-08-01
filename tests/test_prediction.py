"""Tests for ML property prediction tools."""

import pytest

from scimcp.tools.materials.prediction import (
    predict_band_gap,
    predict_density,
    predict_melting_point,
    get_element_info,
    predict_all,
    compute_composition_features,
    _parse_composition,
    _get_composition_features,
    ELEMENTAL_PROPERTIES,
)


class TestParseComposition:
    def test_simple_element(self):
        assert _parse_composition("Si") == {"Si": 1}

    def test_with_count(self):
        assert _parse_composition("Ti3C2") == {"Ti": 3, "C": 2}

    def test_single_atom_count(self):
        assert _parse_composition("Fe2O3") == {"Fe": 2, "O": 3}

    def test_complex_formula(self):
        result = _parse_composition("Ca2Fe2O5")
        assert result == {"Ca": 2, "Fe": 2, "O": 5}

    def test_gaas(self):
        result = _parse_composition("GaAs")
        assert result == {"Ga": 1, "As": 1}


class TestGetCompositionFeatures:
    def test_features_keys(self):
        features = _get_composition_features({"Si": 1})
        expected_keys = {"mean_en", "mean_radius", "mean_mass", "mean_Tm", "en_diff", "n_elements", "mixing_entropy"}
        assert set(features.keys()) == expected_keys

    def test_single_element_features(self):
        features = _get_composition_features({"Si": 1})
        assert features["n_elements"] == 1
        assert features["en_diff"] == 0.0
        assert features["mean_en"] == pytest.approx(ELEMENTAL_PROPERTIES["Si"]["en"], abs=0.01)

    def test_binary_features(self):
        features = _get_composition_features({"Ga": 1, "As": 1})
        assert features["n_elements"] == 2
        assert features["en_diff"] > 0

    def test_unknown_element(self):
        features = _get_composition_features({"Xx": 1})
        assert features["mean_en"] == 0.0
        assert features["mean_mass"] == 0.0


class TestGetElementInfo:
    def test_si(self):
        info = get_element_info("Si")
        assert info is not None
        assert info["Z"] == 14
        assert info["mass"] == pytest.approx(28.086, abs=0.01)

    def test_fe(self):
        info = get_element_info("Fe")
        assert info is not None
        assert info["Z"] == 26

    def test_unknown(self):
        info = get_element_info("Xx")
        assert info is None


class TestPredictBandGap:
    def test_metal_prediction(self):
        result = predict_band_gap("Cu")
        assert result["is_metallic"] is True
        assert result["predicted_band_gap_eV"] < 0.1

    def test_semiconductor_prediction(self):
        result = predict_band_gap("GaAs")
        assert "predicted_band_gap_eV" in result
        assert result["formula"] == "GaAs"
        assert "features" in result

    def test_wide_gap_prediction(self):
        result = predict_band_gap("NaCl")
        assert result["predicted_band_gap_eV"] > 0

    def test_result_structure(self):
        result = predict_band_gap("Si")
        assert "formula" in result
        assert "predicted_band_gap_eV" in result
        assert "is_metallic" in result
        assert "classification" in result
        assert "features" in result
        assert "confidence" in result

    def test_classification_values(self):
        result = predict_band_gap("Si")
        assert result["classification"] in [
            "metal", "narrow-gap semiconductor", "semiconductor",
            "wide-gap semiconductor", "insulator"
        ]


class TestPredictDensity:
    def test_si_density(self):
        result = predict_density("Si")
        assert "estimated_density_g_cm3" in result
        assert result["estimated_density_g_cm3"] > 0
        assert result["formula"] == "Si"

    def test_fe_density(self):
        result = predict_density("Fe")
        assert result["estimated_density_g_cm3"] > 5.0

    def test_result_structure(self):
        result = predict_density("GaAs")
        assert "formula" in result
        assert "estimated_density_g_cm3" in result
        assert "n_atoms" in result


class TestPredictMeltingPoint:
    def test_si_melting(self):
        result = predict_melting_point("Si")
        assert "estimated_melting_point_K" in result
        assert result["estimated_melting_point_K"] > 0

    def test_high_melting_metal(self):
        result = predict_melting_point("W")
        assert result["estimated_melting_point_K"] > 3000

    def test_result_structure(self):
        result = predict_melting_point("Fe")
        assert "formula" in result
        assert "estimated_melting_point_K" in result


class TestComputeCompositionFeatures:
    def test_features_for_si(self):
        result = compute_composition_features("Si")
        assert result["formula"] == "Si"
        assert result["composition"] == {"Si": 1}
        assert "features" in result
        assert "mean_en" in result["features"]

    def test_features_for_binary(self):
        result = compute_composition_features("GaAs")
        assert result["composition"] == {"Ga": 1, "As": 1}
        assert result["features"]["n_elements"] == 2


class TestPredictAll:
    def test_all_properties_si(self):
        result = predict_all("Si")
        assert result["formula"] == "Si"
        assert "band_gap" in result
        assert "density" in result
        assert "melting_point" in result
        assert "composition_features" in result

    def test_all_properties_gaas(self):
        result = predict_all("GaAs")
        assert result["band_gap"]["formula"] == "GaAs"
        assert result["density"]["formula"] == "GaAs"
        assert result["melting_point"]["formula"] == "GaAs"

    def test_all_properties_consistent(self):
        result = predict_all("TiO2")
        assert result["band_gap"]["formula"] == "TiO2"
        assert result["density"]["n_atoms"] == 3
