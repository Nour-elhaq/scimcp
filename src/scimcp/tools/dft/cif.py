"""CIF (Crystallographic Information File) parsing and generation.

Uses ASE (Atomic Simulation Environment) for robust CIF handling
with fallback to custom parsing for lightweight deployments.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_cif(filepath: str) -> dict[str, Any]:
    """Parse a CIF file and extract crystallographic data.

    Reads a CIF file and returns lattice parameters, atomic positions,
    symmetry information, and cell contents.

    Args:
        filepath: Path to the CIF file.

    Returns:
        Dictionary with:
        - 'lattice': dict with a, b, c, alpha, beta, gamma
        - 'cell_vectors': 3x3 numpy array of lattice vectors
        - 'atoms': list of dicts with 'element', 'position', 'label'
        - 'symmetry': space group info
        - 'metadata': other CIF fields
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {filepath}")

    content = path.read_text()
    result = _parse_cif_text(content)
    result["file"] = str(path)
    return result


def _parse_cif_text(content: str) -> dict[str, Any]:
    """Parse CIF text content into structured data."""
    result: dict[str, Any] = {
        "lattice": {},
        "cell_vectors": np.eye(3),
        "atoms": [],
        "symmetry": {},
        "metadata": {},
    }

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            i += 1
            continue

        # Lattice parameters
        if line.startswith("_cell_length_a"):
            result["lattice"]["a"] = _extract_value(line)
        elif line.startswith("_cell_length_b"):
            result["lattice"]["b"] = _extract_value(line)
        elif line.startswith("_cell_length_c"):
            result["lattice"]["c"] = _extract_value(line)
        elif line.startswith("_cell_angle_alpha"):
            result["lattice"]["alpha"] = _extract_value(line)
        elif line.startswith("_cell_angle_beta"):
            result["lattice"]["beta"] = _extract_value(line)
        elif line.startswith("_cell_angle_gamma"):
            result["lattice"]["gamma"] = _extract_value(line)

        # Symmetry
        elif line.startswith("_space_group"):
            key = line.split()[0]
            result["symmetry"][key] = line.split(None, 1)[1] if len(line.split()) > 1 else ""
        elif line.startswith("_symmetry"):
            key = line.split()[0]
            result["symmetry"][key] = line.split(None, 1)[1] if len(line.split()) > 1 else ""

        # Atom sites
        elif line.startswith("_atom_site_"):
            pass  # Handle below

        # Loop blocks
        elif line.startswith("loop_"):
            i += 1
            headers = []
            data_lines = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith("#") or l.startswith("loop_"):
                    break
                if l.startswith("_"):
                    headers.append(l)
                else:
                    data_lines.append(l)
                i += 1

            # Check if this is an atom site block
            atom_headers = [h for h in headers if "_atom_site" in h]
            if atom_headers and data_lines:
                _parse_atom_block(data_lines, result)
            continue

        i += 1

    # Compute cell vectors from lattice parameters
    if result["lattice"]:
        result["cell_vectors"] = _lattice_to_vectors(result["lattice"])

    return result


def _extract_value(line: str) -> float:
    """Extract numeric value from a CIF line, handling uncertainties."""
    parts = line.split()
    if len(parts) >= 2:
        val_str = parts[1]
        # Remove uncertainty in parentheses: 5.2600(3) -> 5.2600
        val_str = re.sub(r"\(.*\)", "", val_str)
        try:
            return float(val_str)
        except ValueError:
            return 0.0
    return 0.0


def _parse_atom_block(block_lines: list[str], result: dict[str, Any]) -> None:
    """Parse atom site block from CIF loop data.

    Handles both 4-column (label, x, y, z) and 5-column (label, type, x, y, z) formats.
    """
    for line in block_lines:
        parts = line.split()
        if len(parts) >= 5:
            # 5-column: label, type_symbol, x, y, z
            atom = {
                "label": parts[0],
                "element": parts[1],
                "x": float(parts[2]) if not parts[2].startswith(".") else 0.0,
                "y": float(parts[3]) if not parts[3].startswith(".") else 0.0,
                "z": float(parts[4]) if not parts[4].startswith(".") else 0.0,
            }
            result["atoms"].append(atom)
        elif len(parts) >= 4:
            # 4-column: label, x, y, z
            atom = {
                "label": parts[0],
                "element": re.sub(r"\d+", "", parts[0]),
                "x": float(parts[1]) if not parts[1].startswith(".") else 0.0,
                "y": float(parts[2]) if not parts[2].startswith(".") else 0.0,
                "z": float(parts[3]) if not parts[3].startswith(".") else 0.0,
            }
            result["atoms"].append(atom)


def _lattice_to_vectors(lattice: dict[str, Any]) -> np.ndarray:
    """Convert lattice parameters to 3x3 cell vectors."""
    a = lattice.get("a", 1.0)
    b = lattice.get("b", 1.0)
    c = lattice.get("c", 1.0)
    alpha = np.radians(lattice.get("alpha", 90.0))
    beta = np.radians(lattice.get("beta", 90.0))
    gamma = np.radians(lattice.get("gamma", 90.0))

    # Compute cell vectors (standard crystallographic convention)
    v1 = np.array([a, 0, 0])
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0, c**2 - cx**2 - cy**2))
    v3 = np.array([cx, cy, cz])

    return np.array([v1, v2, v3])


def generate_cif(
    elements: list[str],
    positions: list[list[float]],
    lattice_params: dict[str, float] | None = None,
    space_group: str = "P 1",
    label: str = "generated",
    output_file: str = "",
) -> str:
    """Generate a CIF file from crystal structure data.

    Args:
        elements: List of element symbols (e.g. ['Ti', 'C', 'Ti']).
        positions: List of fractional coordinates [[x, y, z], ...].
        lattice_params: Dict with a, b, c, alpha, beta, gamma (default: cubic 5.0).
        space_group: Space group symbol (default: P 1).
        label: Crystal label.
        output_file: If provided, write CIF to this file.

    Returns:
        CIF file content as string.
    """
    if lattice_params is None:
        lattice_params = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}

    a = lattice_params.get("a", 5.0)
    b = lattice_params.get("b", 5.0)
    c = lattice_params.get("c", 5.0)
    alpha = lattice_params.get("alpha", 90.0)
    beta = lattice_params.get("beta", 90.0)
    gamma = lattice_params.get("gamma", 90.0)

    lines = [
        f"# SciMCP-generated CIF file",
        f"data_{label}",
        "",
        f"_symmetry_space_group_name_H-M   '{space_group}'",
        "",
        f"_cell_length_a       {a:.6f}",
        f"_cell_length_b       {b:.6f}",
        f"_cell_length_c       {c:.6f}",
        f"_cell_angle_alpha    {alpha:.6f}",
        f"_cell_angle_beta     {beta:.6f}",
        f"_cell_angle_gamma    {gamma:.6f}",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    for elem, pos in zip(elements, positions):
        label_name = f"{elem}{elements[:len(lines)].count(elem) + 1}"
        lines.append(f"  {elem}  {elem}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")

    cif_content = "\n".join(lines) + "\n"

    if output_file:
        with open(output_file, "w") as f:
            f.write(cif_content)

    return cif_content


def cif_to_ase(filepath: str) -> dict[str, Any]:
    """Convert CIF file to ASE Atoms object metadata.

    Args:
        filepath: Path to the CIF file.

    Returns:
        Dictionary with ASE-compatible structure data.
    """
    try:
        from ase.io import read

        atoms = read(filepath)
        return {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions().tolist(),
            "cell": atoms.get_cell().tolist(),
            "pbc": atoms.get_pbc().tolist(),
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
        }
    except ImportError:
        # Fallback to custom CIF parsing
        result = parse_cif(filepath)
        return {
            "symbols": [a["element"] for a in result["atoms"]],
            "positions": [[a["x"], a["y"], a["z"]] for a in result["atoms"]],
            "cell": result["cell_vectors"].tolist(),
            "pbc": [True, True, True],
            "n_atoms": len(result["atoms"]),
            "formula": "".join(a["element"] for a in result["atoms"]),
        }


def get_cif_summary(filepath: str) -> dict[str, Any]:
    """Get a human-readable summary of a CIF file.

    Args:
        filepath: Path to the CIF file.

    Returns:
        Dictionary with lattice, atoms, and formula summary.
    """
    data = parse_cif(filepath)

    elements = [a["element"] for a in data["atoms"]]
    from collections import Counter
    composition = dict(Counter(elements))

    return {
        "file": filepath,
        "lattice": data["lattice"],
        "n_atoms": len(data["atoms"]),
        "elements": list(set(elements)),
        "composition": composition,
        "formula": "".join(f"{e}{c}" if c > 1 else e for e, c in composition.items()),
        "symmetry": data["symmetry"],
    }
