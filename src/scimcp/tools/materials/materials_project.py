"""Materials Project API integration.

Provides access to the Materials Project database for querying
crystal structures, electronic properties, and thermodynamic data.
Falls back to built-in data when no API key is available.
"""

from __future__ import annotations

import os
from typing import Any


# Built-in dataset of common materials (fallback when no API key)
BUILTIN_MATERIALS: dict[str, dict[str, Any]] = {
    "mp-149": {
        "material_id": "mp-149",
        "formula": "Si",
        "space_group": "Fd-3m",
        "crystal_system": "cubic",
        "lattice": {"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 1.11,
        "is_metallic": False,
        "density_g_cm3": 2.329,
        "formation_energy_eV_per_atom": 0.0,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 2,
        "volume_A3": 40.0,
        "elements": ["Si"],
        "symmetry": "diamond",
        "tags": ["semiconductor", "elemental", "diamond"],
    },
    "mp-66": {
        "material_id": "mp-66",
        "formula": "C",
        "space_group": "Fd-3m",
        "crystal_system": "cubic",
        "lattice": {"a": 3.57, "b": 3.57, "c": 3.57, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 5.46,
        "is_metallic": False,
        "density_g_cm3": 3.515,
        "formation_energy_eV_per_atom": 0.0,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 2,
        "elements": ["C"],
        "symmetry": "diamond",
        "tags": ["semiconductor", "elemental", "diamond", "hard"],
    },
    "mp-13": {
        "material_id": "mp-13",
        "formula": "GaAs",
        "space_group": "F-43m",
        "crystal_system": "cubic",
        "lattice": {"a": 5.65, "b": 5.65, "c": 5.65, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 1.42,
        "is_metallic": False,
        "density_g_cm3": 5.317,
        "formation_energy_eV_per_atom": -0.37,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 2,
        "elements": ["As", "Ga"],
        "symmetry": "zincblende",
        "tags": ["semiconductor", "III-V", "optoelectronics"],
    },
    "mp-2657": {
        "material_id": "mp-2657",
        "formula": "TiO2",
        "space_group": "P42/mnm",
        "crystal_system": "tetragonal",
        "lattice": {"a": 4.59, "b": 4.59, "c": 2.96, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 3.03,
        "is_metallic": False,
        "density_g_cm3": 4.147,
        "formation_energy_eV_per_atom": -3.46,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 6,
        "elements": ["O", "Ti"],
        "symmetry": "rutile",
        "tags": ["semiconductor", "oxide", "photocatalyst"],
    },
    "mp-5229": {
        "material_id": "mp-5229",
        "formula": "Al2O3",
        "space_group": "R-3c",
        "crystal_system": "trigonal",
        "lattice": {"a": 4.76, "b": 4.76, "c": 12.99, "alpha": 90, "beta": 90, "gamma": 120},
        "band_gap_eV": 8.80,
        "is_metallic": False,
        "density_g_cm3": 3.987,
        "formation_energy_eV_per_atom": -3.66,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 10,
        "elements": ["Al", "O"],
        "symmetry": "corundum",
        "tags": ["insulator", "oxide", "ceramic"],
    },
    "mp-135": {
        "material_id": "mp-135",
        "formula": "Fe",
        "space_group": "Im-3m",
        "crystal_system": "cubic",
        "lattice": {"a": 2.87, "b": 2.87, "c": 2.87, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "density_g_cm3": 7.874,
        "formation_energy_eV_per_atom": 0.0,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 1,
        "elements": ["Fe"],
        "symmetry": "bcc",
        "tags": ["metal", "elemental", "magnetic"],
    },
    "mp-142": {
        "material_id": "mp-142",
        "formula": "Cu",
        "space_group": "Fm-3m",
        "crystal_system": "cubic",
        "lattice": {"a": 3.64, "b": 3.64, "c": 3.64, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "density_g_cm3": 8.96,
        "formation_energy_eV_per_atom": 0.0,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 1,
        "elements": ["Cu"],
        "symmetry": "fcc",
        "tags": ["metal", "elemental", "conductor"],
    },
    "mp-1960": {
        "material_id": "mp-1960",
        "formula": "CsPbI3",
        "space_group": "Pnma",
        "crystal_system": "orthorhombic",
        "lattice": {"a": 8.84, "b": 8.84, "c": 12.36, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 1.73,
        "is_metallic": False,
        "density_g_cm3": 4.56,
        "formation_energy_eV_per_atom": -1.08,
        "energy_above_hull_eV_per_atom": 0.01,
        "nsites": 20,
        "elements": ["Cs", "I", "Pb"],
        "symmetry": "perovskite",
        "tags": ["perovskite", "solar cell", "halide"],
    },
    "mp-1014380": {
        "material_id": "mp-1014380",
        "formula": "MAPbI3",
        "space_group": "I4/mcm",
        "crystal_system": "tetragonal",
        "lattice": {"a": 8.86, "b": 8.86, "c": 12.66, "alpha": 90, "beta": 90, "gamma": 90},
        "band_gap_eV": 1.55,
        "is_metallic": False,
        "density_g_cm3": 4.16,
        "formation_energy_eV_per_atom": -0.31,
        "energy_above_hull_eV_per_atom": 0.05,
        "nsites": 24,
        "elements": ["C", "H", "I", "N", "Pb"],
        "symmetry": "perovskite",
        "tags": ["perovskite", "solar cell", "halide", "organic-inorganic"],
    },
    "mp-22591": {
        "material_id": "mp-22591",
        "formula": "LiCoO2",
        "space_group": "R-3m",
        "crystal_system": "trigonal",
        "lattice": {"a": 2.82, "b": 2.82, "c": 14.05, "alpha": 90, "beta": 90, "gamma": 120},
        "band_gap_eV": 0.0,
        "is_metallic": True,
        "density_g_cm3": 5.05,
        "formation_energy_eV_per_atom": -2.30,
        "energy_above_hull_eV_per_atom": 0.0,
        "nsites": 12,
        "elements": ["Co", "Li", "O"],
        "symmetry": "layered",
        "tags": ["battery", "cathode", "layered oxide"],
    },
}


def query_materials_project(
    formula: str = "",
    material_id: str = "",
    elements: str = "",
    band_gap_range: str = "",
    metallic_only: bool = False,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Query the Materials Project database.

    Uses the MP API if an API key is provided, otherwise uses built-in data.

    Args:
        formula: Chemical formula (substring match).
        material_id: Specific MP material ID (e.g. 'mp-149').
        elements: Comma-separated element filter (e.g. 'Ti,O').
        band_gap_range: Comma-separated min,max band gap in eV (e.g. '0,2').
        metallic_only: If True, return only metallic materials.
        api_key: Materials Project API key (or set MP_API_KEY env var).
    """
    key = api_key or os.environ.get("MP_API_KEY", "")

    if key:
        return _query_mp_api(
            formula=formula, material_id=material_id, elements=elements,
            band_gap_range=band_gap_range, metallic_only=metallic_only, api_key=key,
        )
    return _query_builtin(
        formula=formula, material_id=material_id, elements=elements,
        band_gap_range=band_gap_range, metallic_only=metallic_only,
    )


def _query_builtin(
    formula: str = "",
    material_id: str = "",
    elements: str = "",
    band_gap_range: str = "",
    metallic_only: bool = False,
) -> list[dict[str, Any]]:
    """Query built-in materials dataset."""
    results = []
    for mid, entry in BUILTIN_MATERIALS.items():
        if material_id and mid != material_id:
            continue
        if formula and formula.lower() not in entry["formula"].lower():
            continue
        if elements:
            required = [e.strip() for e in elements.split(",")]
            if not all(e in entry["elements"] for e in required):
                continue
        if metallic_only and not entry["is_metallic"]:
            continue
        if band_gap_range:
            parts = [float(x) for x in band_gap_range.split(",")]
            if len(parts) == 2:
                if not (parts[0] <= entry["band_gap_eV"] <= parts[1]):
                    continue
        results.append(entry)

    return results


def _query_mp_api(
    formula: str = "",
    material_id: str = "",
    elements: str = "",
    band_gap_range: str = "",
    metallic_only: bool = False,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Query Materials Project via REST API."""
    import json as _json
    import urllib.parse
    import urllib.request

    base_url = "https://api.materialsproject.org/materials/summary"
    criteria: dict[str, Any] = {}

    if material_id:
        criteria["material_id"] = material_id
    if formula:
        criteria["formula"] = formula
    if elements:
        criteria["elements"] = {"$all": [e.strip() for e in elements.split(",")]}
    if metallic_only:
        criteria["is_metallic"] = True
    if band_gap_range:
        parts = [float(x) for x in band_gap_range.split(",")]
        if len(parts) == 2:
            criteria["band_gap"] = {"$gte": parts[0], "$lte": parts[1]}

    params = {"criteria": _json.dumps(criteria), "fields": "formula,band_gap,energy_above_hull,nsites,structure"}
    url = f"{base_url}?{_json.dumps(params)}" if params else base_url

    headers = {"X-API-KEY": api_key}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = _json.loads(response.read().decode("utf-8"))
    except Exception as e:
        return [{"error": str(e)}]

    results = []
    for doc in data.get("data", []):
        results.append({
            "material_id": doc.get("material_id", ""),
            "formula": doc.get("formula", ""),
            "band_gap_eV": doc.get("band_gap", 0),
            "energy_above_hull_eV_per_atom": doc.get("energy_above_hull", 0),
            "nsites": doc.get("nsites", 0),
        })
    return results


def get_material_details(material_id: str, api_key: str = "") -> dict[str, Any] | None:
    """Get detailed properties for a specific material.

    Args:
        material_id: MP material ID (e.g. 'mp-149').
        api_key: Materials Project API key.
    """
    results = query_materials_project(material_id=material_id, api_key=api_key)
    return results[0] if results else None


def search_stable_materials(
    formula: str = "",
    max_e_above_hull: float = 0.01,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Search for thermodynamically stable materials (near hull).

    Args:
        formula: Chemical formula filter.
        max_e_above_hull: Maximum energy above hull in eV/atom.
        api_key: Materials Project API key.
    """
    results = query_materials_project(formula=formula, api_key=api_key)
    return [r for r in results if r.get("energy_above_hull_eV_per_atom", 999) <= max_e_above_hull]
