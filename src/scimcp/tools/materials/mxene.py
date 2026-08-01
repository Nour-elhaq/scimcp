"""MXene database and properties.

Built-in database of common MXene structures with lattice parameters,
electronic properties, and formation energies. Extensible via JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# Built-in MXene database
MXENE_DATABASE: dict[str, dict[str, Any]] = {
    "Ti3C2": {
        "formula": "Ti3C2",
        "formula_terminated": "Ti3C2O2",
        "family": "MXene",
        "M": "Ti",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 3.07, "b": 3.07, "c": 19.5, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.8,
        "magnetic_moment_mu_B": 0.0,
        "elastic_modulus_GPa": 330,
        " Applications": ["energy storage", "EMI shielding", "catalysis", "sensors"],
        "references": ["Naguib et al., Adv. Mater. 2011", "Mashtalir et al., Nat. Commun. 2013"],
    },
    "Ti3C2O2": {
        "formula": "Ti3C2",
        "formula_terminated": "Ti3C2O2",
        "family": "MXene",
        "M": "Ti",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 3.07, "b": 3.07, "c": 19.5, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.8,
        "Applications": ["battery electrodes", "supercapacitors"],
    },
    "Ti3C2F2": {
        "formula": "Ti3C2",
        "formula_terminated": "Ti3C2F2",
        "family": "MXene",
        "M": "Ti",
        "X": "C",
        "termination": "F",
        "lattice": {"a": 3.07, "b": 3.07, "c": 20.0, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.6,
        "Applications": ["Li-ion batteries", "water purification"],
    },
    "Ti3C2OH2": {
        "formula": "Ti3C2",
        "formula_terminated": "Ti3C2(OH)2",
        "family": "MXene",
        "M": "Ti",
        "X": "C",
        "termination": "OH",
        "lattice": {"a": 3.07, "b": 3.07, "c": 21.0, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.2,
        "is_metallic": False,
        "formation_energy_eV_per_atom": -1.4,
        "Applications": ["supercapacitors", "electrocatalysis"],
    },
    "Ti2C": {
        "formula": "Ti2C",
        "formula_terminated": "Ti2CO2",
        "family": "MXene",
        "M": "Ti",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 3.07, "b": 3.07, "c": 15.0, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.5,
        "Applications": ["hydrogen evolution", "sensors"],
    },
    "V2C": {
        "formula": "V2C",
        "formula_terminated": "V2CO2",
        "family": "MXene",
        "M": "V",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 2.88, "b": 2.88, "c": 18.5, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.7,
        "Applications": ["batteries", "catalysis"],
    },
    "Nb2C": {
        "formula": "Nb2C",
        "formula_terminated": "Nb2CO2",
        "family": "MXene",
        "M": "Nb",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 3.16, "b": 3.16, "c": 19.0, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.6,
        "Applications": ["batteries", "EMI shielding"],
    },
    "Mo2C": {
        "formula": "Mo2C",
        "formula_terminated": "Mo2CO2",
        "family": "MXene",
        "M": "Mo",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 2.99, "b": 2.99, "c": 19.2, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.1,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.3,
        "Applications": ["electrocatalysis", "batteries"],
    },
    "Ti3C2Se2": {
        "formula": "Ti3C2",
        "formula_terminated": "Ti3C2Se2",
        "family": "MXene",
        "M": "Ti",
        "X": "C",
        "termination": "Se",
        "lattice": {"a": 3.30, "b": 3.30, "c": 20.5, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.2,
        "Applications": ["optoelectronics", "photocatalysis"],
    },
    "Cr3C2": {
        "formula": "Cr3C2",
        "formula_terminated": "Cr3C2O2",
        "family": "MXene",
        "M": "Cr",
        "X": "C",
        "termination": "O",
        "lattice": {"a": 2.93, "b": 2.93, "c": 18.8, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "formation_energy_eV_per_atom": -1.9,
        "Applications": ["magnetic applications", "catalysis"],
    },
}


def query_mxene(
    formula: str = "",
    M_element: str = "",
    X_element: str = "",
    termination: str = "",
    metallic_only: bool = False,
) -> list[dict[str, Any]]:
    """Query the MXene database.

    Search by formula, M element, X element, termination, or metallic character.

    Args:
        formula: Partial formula match (e.g. 'Ti3C2').
        M_element: Transition metal element (e.g. 'Ti', 'V', 'Nb').
        X_element: Light element (e.g. 'C', 'N').
        termination: Surface termination (e.g. 'O', 'F', 'OH').
        metallic_only: If True, return only metallic MXenes.

    Returns:
        List of matching MXene entries.
    """
    results = []
    for key, entry in MXENE_DATABASE.items():
        if formula and formula.lower() not in key.lower():
            continue
        if M_element and entry.get("M", "") != M_element:
            continue
        if X_element and entry.get("X", "") != X_element:
            continue
        if termination and entry.get("termination", "") != termination:
            continue
        if metallic_only and not entry.get("is_metallic", False):
            continue
        results.append({"name": key, **entry})

    return results


def get_mxene_list() -> list[str]:
    """List all available MXene compositions in the database.

    Returns:
        List of MXene names.
    """
    return list(MXENE_DATABASE.keys())


def get_mxene_properties(formula: str) -> dict[str, Any] | None:
    """Get detailed properties of a specific MXene.

    Args:
        formula: MXene formula (e.g. 'Ti3C2', 'V2C', 'Nb2C').

    Returns:
        Property dict or None if not found.
    """
    if formula in MXENE_DATABASE:
        return {"name": formula, **MXENE_DATABASE[formula]}

    # Try partial match
    for key, entry in MXENE_DATABASE.items():
        if formula.lower() in key.lower():
            return {"name": key, **entry}

    return None


def compare_mxenes(formulas: list[str]) -> dict[str, Any]:
    """Compare properties of multiple MXenes side by side.

    Args:
        formulas: List of MXene formulas to compare.

    Returns:
        Dictionary with comparison data for each MXene.
    """
    comparison = {}
    for formula in formulas:
        props = get_mxene_properties(formula)
        if props:
            comparison[formula] = props
        else:
            comparison[formula] = {"error": f"MXene '{formula}' not found in database"}

    return comparison


def add_mxene_to_database(
    name: str,
    formula: str,
    M: str,
    X: str,
    termination: str = "O",
    lattice: dict[str, float] | None = None,
    band_gap_eV: float = 0.0,
    is_metallic: bool = True,
    **kwargs: Any,
) -> None:
    """Add a new MXene to the database at runtime.

    Args:
        name: MXene name (e.g. 'Hf2C').
        formula: Base formula.
        M: Transition metal.
        X: Light element.
        termination: Surface termination.
        lattice: Lattice parameters.
        band_gap_eV: Band gap in eV.
        is_metallic: Whether it's metallic.
        **kwargs: Additional properties.
    """
    if lattice is None:
        lattice = {"a": 3.0, "b": 3.0, "c": 19.0, "alpha": 90, "beta": 90, "gamma": 90}

    MXENE_DATABASE[name] = {
        "formula": formula,
        "formula_terminated": f"{formula}{termination}2",
        "family": "MXene",
        "M": M,
        "X": X,
        "termination": termination,
        "lattice": lattice,
        "band_gap_eV": band_gap_eV,
        "is_metallic": is_metallic,
        **kwargs,
    }


def search_mxene_by_property(
    property_name: str,
    min_value: float = 0.0,
    max_value: float = float("inf"),
) -> list[dict[str, Any]]:
    """Search MXenes by a specific property range.

    Args:
        property_name: Property to search (e.g. 'band_gap_eV', 'formation_energy_eV_per_atom').
        min_value: Minimum value.
        max_value: Maximum value.

    Returns:
        List of MXenes within the specified range.
    """
    results = []
    for key, entry in MXENE_DATABASE.items():
        val = entry.get(property_name)
        if val is not None and min_value <= val <= max_value:
            results.append({"name": key, property_name: val, **entry})

    return results
