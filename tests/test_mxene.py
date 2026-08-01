"""Tests for MXene database and query tools."""

import json

import pytest

from scimcp.tools.materials.mxene import (
    query_mxene,
    get_mxene_list,
    get_mxene_properties,
    compare_mxenes,
    add_mxene_to_database,
    search_mxene_by_property,
    MXENE_DATABASE,
)


class TestGetMxeneList:
    def test_list_returns_all(self):
        mxenes = get_mxene_list()
        assert len(mxenes) >= 10
        assert "Ti3C2" in mxenes
        assert "V2C" in mxenes
        assert "Nb2C" in mxenes

    def test_list_types(self):
        mxenes = get_mxene_list()
        for m in mxenes:
            assert isinstance(m, str)


class TestQueryMxene:
    def test_query_all(self):
        results = query_mxene()
        assert len(results) >= 10

    def test_query_by_formula(self):
        results = query_mxene(formula="Ti3C2")
        assert len(results) >= 1
        assert all("Ti3C2" in r["name"] for r in results)

    def test_query_by_m_element(self):
        results = query_mxene(M_element="V")
        assert len(results) >= 1
        assert all(r["M"] == "V" for r in results)

    def test_query_by_x_element(self):
        results = query_mxene(X_element="C")
        assert len(results) >= 1

    def test_query_by_termination(self):
        results = query_mxene(termination="O")
        assert len(results) >= 1
        assert all(r["termination"] == "O" for r in results)

    def test_query_metallic_only(self):
        results = query_mxene(metallic_only=True)
        assert len(results) >= 1
        assert all(r["is_metallic"] is True for r in results)

    def test_query_combined(self):
        results = query_mxene(M_element="Ti", metallic_only=True)
        assert len(results) >= 1
        assert all(r["M"] == "Ti" and r["is_metallic"] for r in results)

    def test_query_no_results(self):
        results = query_mxene(formula="Nonexistent")
        assert len(results) == 0


class TestGetMxeneProperties:
    def test_get_ti3c2(self):
        props = get_mxene_properties("Ti3C2")
        assert props is not None
        assert props["formula"] == "Ti3C2"
        assert props["M"] == "Ti"
        assert props["X"] == "C"
        assert "lattice" in props

    def test_get_v2c(self):
        props = get_mxene_properties("V2C")
        assert props is not None
        assert props["M"] == "V"

    def test_get_nonexistent(self):
        props = get_mxene_properties("Nonexistent")
        assert props is None

    def test_get_partial_match(self):
        props = get_mxene_properties("Ti3")
        assert props is not None


class TestCompareMxenes:
    def test_compare_two(self):
        result = compare_mxenes(["Ti3C2", "V2C"])
        assert "Ti3C2" in result
        assert "V2C" in result
        assert result["Ti3C2"]["M"] == "Ti"
        assert result["V2C"]["M"] == "V"

    def test_compare_with_nonexistent(self):
        result = compare_mxenes(["Ti3C2", "Nonexistent"])
        assert "Ti3C2" in result
        assert "Nonexistent" in result
        assert "error" in result["Nonexistent"]

    def test_compare_empty(self):
        result = compare_mxenes([])
        assert result == {}


class TestAddMxeneToDatabase:
    def test_add_new_mxene(self):
        add_mxene_to_database(
            name="Hf2C",
            formula="Hf2C",
            M="Hf",
            X="C",
            termination="O",
            band_gap_eV=0.0,
            is_metallic=True,
        )
        assert "Hf2C" in MXENE_DATABASE
        props = get_mxene_properties("Hf2C")
        assert props is not None
        assert props["M"] == "Hf"
        # Clean up
        del MXENE_DATABASE["Hf2C"]

    def test_add_with_custom_lattice(self):
        custom_lattice = {"a": 3.5, "b": 3.5, "c": 20.0, "alpha": 90, "beta": 90, "gamma": 90}
        add_mxene_to_database(
            name="Ta2C",
            formula="Ta2C",
            M="Ta",
            X="C",
            lattice=custom_lattice,
        )
        props = get_mxene_properties("Ta2C")
        assert props is not None
        assert props["lattice"]["a"] == 3.5
        # Clean up
        del MXENE_DATABASE["Ta2C"]


class TestSearchMxeneByProperty:
    def test_search_band_gap(self):
        results = search_mxene_by_property("band_gap_eV", 0.0, 0.0)
        assert len(results) >= 1
        assert all(r["band_gap_eV"] == 0.0 for r in results)

    def test_search_formation_energy(self):
        results = search_mxene_by_property("formation_energy_eV_per_atom", -2.0, -1.5)
        assert len(results) >= 1

    def test_search_no_results(self):
        results = search_mxene_by_property("band_gap_eV", 10.0, 20.0)
        assert len(results) == 0

    def test_search_elastic_modulus(self):
        results = search_mxene_by_property("elastic_modulus_GPa", 300, 400)
        assert len(results) >= 1
