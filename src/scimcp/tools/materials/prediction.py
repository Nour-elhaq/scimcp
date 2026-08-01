"""ML-accelerated material property prediction.

Uses pre-trained models and composition-based features to predict
material properties without running expensive DFT calculations.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


# Elemental properties database (simplified)
ELEMENTAL_PROPERTIES: dict[str, dict[str, float]] = {
    "H":  {"Z": 1,  "mass": 1.008,   "en": 2.20, "radius": 0.25, "density": 0.00009, "Tm": 14.01},
    "He": {"Z": 2,  "mass": 4.003,   "en": 0.00, "radius": 0.31, "density": 0.00018, "Tm": 0.95},
    "Li": {"Z": 3,  "mass": 6.941,   "en": 0.98, "radius": 1.28, "density": 0.534,  "Tm": 453.69},
    "C":  {"Z": 6,  "mass": 12.011,  "en": 2.55, "radius": 0.77, "density": 2.267,  "Tm": 3823.0},
    "N":  {"Z": 7,  "mass": 14.007,  "en": 3.04, "radius": 0.75, "density": 0.0012, "Tm": 63.15},
    "O":  {"Z": 8,  "mass": 15.999,  "en": 3.44, "radius": 0.73, "density": 0.0014, "Tm": 54.36},
    "F":  {"Z": 9,  "mass": 18.998,  "en": 3.98, "radius": 0.72, "density": 0.0017, "Tm": 53.53},
    "Na": {"Z": 11, "mass": 22.990,  "en": 0.93, "radius": 1.66, "density": 0.971,  "Tm": 370.87},
    "Mg": {"Z": 12, "mass": 24.305,  "en": 1.31, "radius": 1.41, "density": 1.738,  "Tm": 923.0},
    "Al": {"Z": 13, "mass": 26.982,  "en": 1.61, "radius": 1.18, "density": 2.698,  "Tm": 933.47},
    "Si": {"Z": 14, "mass": 28.086,  "en": 1.90, "radius": 1.11, "density": 2.329,  "Tm": 1687.0},
    "P":  {"Z": 15, "mass": 30.974,  "en": 2.19, "radius": 1.06, "density": 1.82,   "Tm": 317.3},
    "S":  {"Z": 16, "mass": 32.065,  "en": 2.58, "radius": 1.02, "density": 2.067,  "Tm": 388.36},
    "Cl": {"Z": 17, "mass": 35.453,  "en": 3.16, "radius": 0.99, "density": 0.0032, "Tm": 171.6},
    "K":  {"Z": 19, "mass": 39.098,  "en": 0.82, "radius": 2.03, "density": 0.862,  "Tm": 336.53},
    "Ca": {"Z": 20, "mass": 40.078,  "en": 1.00, "radius": 1.74, "density": 1.55,   "Tm": 1115.0},
    "Ti": {"Z": 22, "mass": 47.867,  "en": 1.54, "radius": 1.32, "density": 4.54,   "Tm": 1941.0},
    "V":  {"Z": 23, "mass": 50.942,  "en": 1.63, "radius": 1.22, "density": 6.11,   "Tm": 2183.0},
    "Cr": {"Z": 24, "mass": 51.996,  "en": 1.66, "radius": 1.18, "density": 7.15,   "Tm": 2180.0},
    "Mn": {"Z": 25, "mass": 54.938,  "en": 1.55, "radius": 1.17, "density": 7.44,   "Tm": 1519.0},
    "Fe": {"Z": 26, "mass": 55.845,  "en": 1.83, "radius": 1.17, "density": 7.874,  "Tm": 1811.0},
    "Co": {"Z": 27, "mass": 58.933,  "en": 1.88, "radius": 1.16, "density": 8.9,    "Tm": 1768.0},
    "Ni": {"Z": 28, "mass": 58.693,  "en": 1.91, "radius": 1.15, "density": 8.912,  "Tm": 1728.0},
    "Cu": {"Z": 29, "mass": 63.546,  "en": 1.90, "radius": 1.17, "density": 8.96,   "Tm": 1357.77},
    "Zn": {"Z": 30, "mass": 65.38,   "en": 1.65, "radius": 1.25, "density": 7.134,  "Tm": 692.68},
    "Ga": {"Z": 31, "mass": 69.723,  "en": 1.81, "radius": 1.26, "density": 5.907,  "Tm": 302.91},
    "Ge": {"Z": 32, "mass": 72.64,   "en": 2.01, "radius": 1.22, "density": 5.323,  "Tm": 1211.4},
    "As": {"Z": 33, "mass": 74.922,  "en": 2.18, "radius": 1.19, "density": 5.776,  "Tm": 1090.0},
    "Se": {"Z": 34, "mass": 78.96,   "en": 2.55, "radius": 1.16, "density": 4.809,  "Tm": 493.65},
    "Br": {"Z": 35, "mass": 79.904,  "en": 2.96, "radius": 1.14, "density": 3.122,  "Tm": 265.8},
    "Nb": {"Z": 41, "mass": 92.906,  "en": 1.60, "radius": 1.34, "density": 8.57,   "Tm": 2750.0},
    "Mo": {"Z": 42, "mass": 95.94,   "en": 2.16, "radius": 1.30, "density": 10.22,  "Tm": 2896.0},
    "Ag": {"Z": 47, "mass": 107.868, "en": 1.93, "radius": 1.45, "density": 10.501, "Tm": 1234.93},
    "Sn": {"Z": 50, "mass": 118.710, "en": 1.96, "radius": 1.45, "density": 7.287,  "Tm": 505.08},
    "W":  {"Z": 74, "mass": 183.84,  "en": 2.36, "radius": 1.37, "density": 19.25,  "Tm": 3695.0},
    "Pt": {"Z": 78, "mass": 195.084, "en": 2.28, "radius": 1.36, "density": 21.46,  "Tm": 2041.4},
    "Au": {"Z": 79, "mass": 196.967, "en": 2.54, "radius": 1.36, "density": 19.282, "Tm": 1337.33},
    "Pb": {"Z": 82, "mass": 207.2,   "en": 2.33, "radius": 1.46, "density": 11.342, "Tm": 600.61},
}


def _parse_composition(formula: str) -> dict[str, int]:
    """Parse a chemical formula into element counts.

    Args:
        formula: Chemical formula (e.g. 'Ti3C2', 'Fe2O3').

    Returns:
        Dict mapping element symbol to count.
    """
    import re
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)
    composition = {}
    for element, count in matches:
        if element:
            composition[element] = int(count) if count else 1
    return composition


def _get_composition_features(composition: dict[str, int]) -> dict[str, float]:
    """Compute composition-based features for ML prediction.

    Args:
        composition: Dict mapping element to count.

    Returns:
        Feature vector as dictionary.
    """
    total_atoms = sum(composition.values())
    features = {}

    # Elemental fractions
    fractions = {el: count / total_atoms for el, count in composition.items()}

    # Weighted average electronegativity
    en_values = []
    for el, frac in fractions.items():
        if el in ELEMENTAL_PROPERTIES:
            en_values.append(frac * ELEMENTAL_PROPERTIES[el]["en"])
    features["mean_en"] = sum(en_values) if en_values else 0.0

    # Weighted average atomic radius
    radius_values = []
    for el, frac in fractions.items():
        if el in ELEMENTAL_PROPERTIES:
            radius_values.append(frac * ELEMENTAL_PROPERTIES[el]["radius"])
    features["mean_radius"] = sum(radius_values) if radius_values else 0.0

    # Weighted average atomic mass
    mass_values = []
    for el, frac in fractions.items():
        if el in ELEMENTAL_PROPERTIES:
            mass_values.append(frac * ELEMENTAL_PROPERTIES[el]["mass"])
    features["mean_mass"] = sum(mass_values) if mass_values else 0.0

    # Weighted average melting point
    tm_values = []
    for el, frac in fractions.items():
        if el in ELEMENTAL_PROPERTIES:
            tm_values.append(frac * ELEMENTAL_PROPERTIES[el]["Tm"])
    features["mean_Tm"] = sum(tm_values) if tm_values else 0.0

    # Electronegativity difference (proxy for ionic character)
    en_list = [ELEMENTAL_PROPERTIES[el]["en"] for el in composition if el in ELEMENTAL_PROPERTIES]
    features["en_diff"] = max(en_list) - min(en_list) if len(en_list) > 1 else 0.0

    # Number of elements
    features["n_elements"] = len(composition)

    # Entropy of mixing (negated, for ordering tendency)
    fracs = np.array(list(fractions.values()))
    fracs = fracs[fracs > 0]
    features["mixing_entropy"] = float(-np.sum(fracs * np.log(fracs + 1e-10)))

    return features


def predict_band_gap(formula: str) -> dict[str, Any]:
    """Predict band gap from composition using a simple ML model.

    Uses electronegativity difference and atomic size mismatch as features
    with a pre-trained linear model (simplified for demonstration).

    Args:
        formula: Chemical formula (e.g. 'GaAs', 'TiO2').

    Returns:
        Dictionary with predicted band gap and confidence.
    """
    composition = _parse_composition(formula)
    features = _get_composition_features(composition)

    # Simple heuristic model (for demonstration — replace with real ML model)
    # Higher en_diff and ionic character -> larger band gap
    en_diff = features["en_diff"]
    n_elements = features["n_elements"]

    # Heuristic: metal if all same EN range, semiconductor if moderate difference, insulator if large
    if en_diff < 0.3:
        predicted_gap = 0.0  # Metallic
    elif en_diff < 1.0:
        predicted_gap = 0.5 + en_diff * 1.5  # Narrow gap semiconductor
    elif en_diff < 2.0:
        predicted_gap = 1.0 + en_diff * 0.8  # Wide gap semiconductor
    else:
        predicted_gap = 2.0 + en_diff * 0.5  # Insulator

    predicted_gap = max(0.0, min(predicted_gap, 15.0))

    return {
        "formula": formula,
        "predicted_band_gap_eV": round(predicted_gap, 3),
        "is_metallic": predicted_gap < 0.1,
        "classification": (
            "metal" if predicted_gap < 0.1
            else "narrow-gap semiconductor" if predicted_gap < 1.0
            else "semiconductor" if predicted_gap < 3.0
            else "wide-gap semiconductor" if predicted_gap < 5.0
            else "insulator"
        ),
        "features": features,
        "confidence": "heuristic (replace with trained model for production)",
    }


def predict_density(formula: str) -> dict[str, Any]:
    """Estimate material density from composition.

    Args:
        formula: Chemical formula.

    Returns:
        Dictionary with estimated density.
    """
    composition = _parse_composition(formula)
    total_atoms = sum(composition.values())

    # Weighted average density (simplified)
    densities = []
    for el, count in composition.items():
        if el in ELEMENTAL_PROPERTIES:
            densities.append(count * ELEMENTAL_PROPERTIES[el]["density"])

    avg_density = sum(densities) / total_atoms if densities else 0.0

    return {
        "formula": formula,
        "estimated_density_g_cm3": round(avg_density, 3),
        "n_atoms": total_atoms,
    }


def predict_melting_point(formula: str) -> dict[str, Any]:
    """Estimate melting point from composition.

    Uses a weighted average of elemental melting points.

    Args:
        formula: Chemical formula.

    Returns:
        Dictionary with estimated melting point.
    """
    composition = _parse_composition(formula)
    features = _get_composition_features(composition)

    return {
        "formula": formula,
        "estimated_melting_point_K": round(features["mean_Tm"], 1),
    }


def get_element_info(element: str) -> dict[str, Any] | None:
    """Get properties of a single element.

    Args:
        element: Element symbol.

    Returns:
        Element properties or None if not found.
    """
    return ELEMENTAL_PROPERTIES.get(element)


def compute_composition_features(formula: str) -> dict[str, Any]:
    """Compute all compositional features for a formula.

    Args:
        formula: Chemical formula.

    Returns:
        Dictionary with composition features.
    """
    composition = _parse_composition(formula)
    features = _get_composition_features(composition)

    return {
        "formula": formula,
        "composition": composition,
        "features": features,
    }


def predict_all(formula: str) -> dict[str, Any]:
    """Run all prediction models on a formula.

    Args:
        formula: Chemical formula.

    Returns:
        Combined prediction results.
    """
    band_gap = predict_band_gap(formula)
    density = predict_density(formula)
    melting = predict_melting_point(formula)
    features = compute_composition_features(formula)

    return {
        "formula": formula,
        "band_gap": band_gap,
        "density": density,
        "melting_point": melting,
        "composition_features": features,
    }
