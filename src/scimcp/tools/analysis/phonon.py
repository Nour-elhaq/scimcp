"""Phonon analysis and band structure tools.

Computes phonon frequencies, density of states (DOS),
and thermodynamic properties from force constants.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def compute_phonon_dos(
    frequencies_json: str,
    sigma: float = 0.5,
    n_points: int = 200,
) -> str:
    """Compute phonon density of states from frequency list.

    Args:
        frequencies_json: JSON array of phonon frequencies in THz.
        sigma: Gaussian broadening width (THz).
        n_points: Number of points in the DOS grid.

    Returns:
        JSON string with frequency grid and DOS values.
    """
    freqs = np.array(json.loads(frequencies_json))
    freq_min = min(freqs.min(), -5.0)
    freq_max = max(freqs.max(), 5.0)
    freq_grid = np.linspace(freq_min, freq_max, n_points)

    dos = np.zeros(n_points)
    for f in freqs:
        dos += np.exp(-0.5 * ((freq_grid - f) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    return json.dumps({
        "frequency_THz": freq_grid.tolist(),
        "dos": dos.tolist(),
        "n_modes": len(freqs),
        "n_imaginary": int((freqs < 0).sum()),
        "min_freq_THz": float(freqs.min()),
        "max_freq_THz": float(freqs.max()),
    }, indent=2)


def compute_thermodynamic_properties(
    frequencies_json: str,
    temperature_K: float = 300.0,
    n_atoms: int = 1,
) -> str:
    """Compute thermodynamic properties from phonon frequencies.

    Uses the harmonic approximation to compute Helmholtz free energy,
    entropy, and heat capacity.

    Args:
        frequencies_json: JSON array of phonon frequencies in THz.
        temperature_K: Temperature in Kelvin.
        n_atoms: Number of atoms in the unit cell.

    Returns:
        JSON string with thermodynamic properties.
    """
    freqs = np.array(json.loads(frequencies_json))

    # Filter out imaginary (negative) frequencies
    pos_freqs = freqs[freqs > 0]

    kB = 1.380649e-23  # J/K
    h = 6.62607015e-34  # J·s
    THz_to_Hz = 1e12

    T = temperature_K
    properties = {
        "temperature_K": T,
        "n_modes_total": len(freqs),
        "n_modes_positive": len(pos_freqs),
        "n_imaginary": int((freqs < 0).sum()),
    }

    if len(pos_freqs) == 0 or T == 0:
        properties.update({
            "ZPE_eV": 0.0,
            "Helmholtz_eV": 0.0,
            "entropy_J_mol_K": 0.0,
            "Cv_J_mol_K": 0.0,
        })
        return json.dumps(properties, indent=2)

    # Zero-point energy
    ZPE = 0.5 * h * THz_to_Hz * pos_freqs.sum()  # per unit cell in Joules
    properties["ZPE_eV"] = float(ZPE / 1.602176634e-19 / n_atoms)

    # Thermodynamic integrals
    x = h * THz_to_Hz * pos_freqs / (kB * T)
    x = x[x < 500]  # avoid overflow

    # Helmholtz free energy (per atom)
    F = kB * T * np.sum(np.log(2 * np.sinh(x / 2))) / n_atoms
    properties["Helmholtz_eV"] = float(F / 1.602176634e-19)

    # Entropy (per mol)
    S = kB * np.sum(x / np.tanh(x / 2) - np.log(2 * np.sinh(x / 2))) * 6.022e23
    properties["entropy_J_mol_K"] = float(S)

    # Heat capacity at constant volume (per mol)
    Cv = kB * np.sum((x / np.sinh(x / 2)) ** 2) * 6.022e23
    properties["Cv_J_mol_K"] = float(Cv)

    return json.dumps(properties, indent=2)


def generate_phonon_band_path(
    lattice_params: dict[str, float],
    crystal_system: str = "cubic",
    n_points: int = 50,
) -> str:
    """Generate a high-symmetry k-path for phonon band structure.

    Args:
        lattice_params: Lattice parameters (a, b, c, alpha, beta, gamma).
        crystal_system: Crystal system (cubic, hexagonal, tetragonal, orthorhombic).
        n_points: Number of points between high-symmetry points.

    Returns:
        JSON string with k-path labels and coordinates.
    """
    paths: dict[str, list[tuple[str, list[float]]]] = {
        "cubic": [
            ("Gamma", [0, 0, 0]),
            ("X", [0.5, 0, 0]),
            ("M", [0.5, 0.5, 0]),
            ("Gamma", [0, 0, 0]),
            ("R", [0.5, 0.5, 0.5]),
            ("X", [0.5, 0, 0]),
        ],
        "hexagonal": [
            ("Gamma", [0, 0, 0]),
            ("M", [0.5, 0, 0]),
            ("K", [1/3, 1/3, 0]),
            ("Gamma", [0, 0, 0]),
            ("A", [0, 0, 0.5]),
            ("L", [0.5, 0, 0.5]),
            ("M", [0.5, 0, 0]),
        ],
        "tetragonal": [
            ("Gamma", [0, 0, 0]),
            ("X", [0.5, 0, 0]),
            ("M", [0.5, 0.5, 0]),
            ("Gamma", [0, 0, 0]),
            ("Z", [0, 0, 0.5]),
            ("R", [0.5, 0, 0.5]),
            ("A", [0.5, 0.5, 0.5]),
        ],
        "orthorhombic": [
            ("Gamma", [0, 0, 0]),
            ("X", [0.5, 0, 0]),
            ("S", [0.5, 0.5, 0]),
            ("Y", [0, 0.5, 0]),
            ("Gamma", [0, 0, 0]),
            ("Z", [0, 0, 0.5]),
            ("U", [0.5, 0, 0.5]),
            ("R", [0.5, 0.5, 0.5]),
            ("T", [0, 0.5, 0.5]),
        ],
    }

    system = crystal_system.lower()
    if system not in paths:
        system = "cubic"

    kpath = paths[system]

    # Interpolate between high-symmetry points
    all_k = []
    all_labels = []
    for i in range(len(kpath) - 1):
        label_start = kpath[i][0]
        k_start = np.array(kpath[i][1])
        k_end = np.array(kpath[i + 1][1])

        for j in range(n_points):
            t = j / n_points
            k = k_start + t * (k_end - k_start)
            all_k.append(k.tolist())
            if j == 0:
                all_labels.append(label_start)
            else:
                all_labels.append("")

    # Add final point
    all_k.append(kpath[-1][1])
    all_labels.append(kpath[-1][0])

    return json.dumps({
        "crystal_system": system,
        "lattice_params": lattice_params,
        "n_points": len(all_k),
        "k_points": all_k,
        "labels": all_labels,
        "high_symmetry_points": [p[0] for p in kpath],
    }, indent=2)


def estimate_phonon_frequencies(
    composition: str,
    crystal_system: str = "cubic",
) -> str:
    """Estimate phonon frequency range from elemental data.

    Uses a simple mass-frequency scaling relation.

    Args:
        composition: Chemical formula (e.g., 'Si', 'GaAs').
        crystal_system: Crystal system.

    Returns:
        JSON string with estimated frequency range.
    """
    from ..materials.prediction import _parse_composition, ELEMENTAL_PROPERTIES

    comp = _parse_composition(composition)
    total_atoms = sum(comp.values())

    # Simple mass-based estimate: omega ~ 1/sqrt(mass)
    masses = []
    for el, count in comp.items():
        if el in ELEMENTAL_PROPERTIES:
            masses.extend([ELEMENTAL_PROPERTIES[el]["mass"]] * count)

    if not masses:
        return json.dumps({"error": f"Unknown elements in {composition}"}, indent=2)

    avg_mass = np.mean(masses)

    # Rough estimate: optical modes ~ sqrt(K/m), acoustic ~ 0
    # Using typical force constant ~ 10 N/m = 10 J/m^2
    K_typical = 10.0  # N/m
    amu_to_kg = 1.66054e-27

    omega = np.sqrt(K_typical / (avg_mass * amu_to_kg)) / (2 * np.pi * 1e12)

    # Acoustic modes: 3 per unit cell (2 transverse, 1 longitudinal)
    n_acoustic = 3
    n_optical = 3 * total_atoms - n_acoustic

    return json.dumps({
        "composition": composition,
        "crystal_system": crystal_system,
        "n_atoms_unit_cell": total_atoms,
        "n_acoustic_modes": n_acoustic,
        "n_optical_modes": n_optical,
        "estimated_max_freq_THz": round(float(omega), 2),
        "estimated_max_freq_cm1": round(float(omega * 33.356), 1),
        "note": "Rough estimate using mass-force constant model. Use DFT for accurate values.",
    }, indent=2)
