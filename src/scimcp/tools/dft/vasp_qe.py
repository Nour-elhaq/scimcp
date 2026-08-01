"""VASP and Quantum ESPRESSO input file generation.

Generates INCAR, POSCAR, KPOINTS for VASP and pw.x input files
for Quantum ESPRESSO from standard parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_vasp_incar(
    encut: float = 520,
    ediff: float = 1e-6,
    isif: int = 3,
    ibrion: int = -1,
    nsw: int = 0,
    potim: float = 0.5,
    ispin: int = 1,
    lorbit: int = 11,
    lwave: bool = True,
    lcharg: bool = True,
    kspacing: float = 0.5,
    sigma: float = 0.1,
    ismear: int = 1,
    nelm: int = 60,
    ediffg: float = -0.01,
    tebreak: float = 0.0,
    output_file: str = "",
) -> str:
    """Generate a VASP INCAR file.

    Args:
        encut: Plane wave cutoff energy (eV).
        ediff: Electronic convergence criterion.
        isif: Ion relaxation mode (2=fixed, 3=relax cell, 4=fix cell relax ions).
        ibrion: Ionic relaxation algorithm (-1=MD, 0=none, 1=CG, 2=QH).
        nsw: Number of ionic steps (0=static).
        potim: Time step for MD (fs).
        ispin: Spin polarization (1=non-magnetic, 2=magnetic).
        lorbit: DOS projection (0=no, 11=PAW, 12=LDA+U).
        lwave: Write WAVECAR.
        lcharg: Write CHGCAR.
        kspacing: k-point spacing (Angstrom^-1).
        sigma: Smearing width (eV).
        ismear: Smearing method (-5=Tetrahedron, 0=Fermi, 1=Gaussian).
        nelm: Max electronic steps.
        ediffg: Force convergence criterion (eV/Angstrom).
        tebreak: Target breaking symmetry energy.
        output_file: If provided, write INCAR to this file.
    """
    lines = [
        "# SciMCP-generated VASP INCAR",
        "",
        "SYSTEM = scimcp_generated",
        "",
        "# Electronic minimization",
        f"ENCUT  = {encut}",
        f"EDIFF  = {ediff:.1e}",
        f"NELM   = {nelm}",
        f"ISMEAR = {ismear}",
        f"SIGMA  = {sigma}",
        "",
        "# k-point sampling",
        f"KSPACING = {kspacing}",
        "",
        "# Write output",
        f"LWAVE  = {'.TRUE.' if lwave else '.FALSE.'}",
        f"LCHARG = {'.TRUE.' if lcharg else '.FALSE.'}",
        "",
    ]

    if ispin > 1:
        lines += [
            "# Spin polarization",
            f"ISPIN  = {ispin}",
            f"LORBIT = {lorbit}",
            "",
        ]

    if nsw > 0:
        lines += [
            "# Ionic relaxation",
            f"IBRION = {ibrion}",
            f"NSW    = {nsw}",
            f"POTIM  = {potim}",
            f"ISIF   = {isif}",
            f"EDIFFG = {ediffg}",
            "",
        ]
    else:
        lines += [
            "# Static calculation",
            f"IBRION = -1",
            f"NSW    = 0",
            "",
        ]

    if tebreak > 0:
        lines += [f"TEBREAK = {tebreak}", ""]

    incar_content = "\n".join(lines)

    if output_file:
        Path(output_file).write_text(incar_content)

    return incar_content


def generate_vasp_poscar(
    elements: list[str],
    positions: list[list[float]],
    lattice_params: dict[str, float] | None = None,
    selective_dynamics: bool = False,
    selective_mask: list[list[bool]] | None = None,
    comment: str = "SciMCP generated",
    output_file: str = "",
) -> str:
    """Generate a VASP POSCAR file.

    Args:
        elements: List of element symbols in order.
        positions: List of fractional coordinates [[x, y, z], ...].
        lattice_params: Dict with a, b, c, alpha, beta, gamma.
        selective_dynamics: Use selective dynamics.
        selective_mask: T/F mask for each atom [T,T,T] or [F,F,F].
        comment: Comment line.
        output_file: If provided, write POSCAR to this file.
    """
    if lattice_params is None:
        lattice_params = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 90}

    import numpy as np

    a = lattice_params.get("a", 5.0)
    b = lattice_params.get("b", 5.0)
    c = lattice_params.get("c", 5.0)
    alpha = np.radians(lattice_params.get("alpha", 90))
    beta = np.radians(lattice_params.get("beta", 90))
    gamma = np.radians(lattice_params.get("gamma", 90))

    v1 = np.array([a, 0, 0])
    v2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0, c**2 - cx**2 - cy**2))
    v3 = np.array([cx, cy, cz])

    # Count atoms per element
    from collections import Counter
    elem_counts = Counter(elements)

    lines = [
        comment,
        "1.0",
        f"  {v1[0]:.6f}  {v1[1]:.6f}  {v1[2]:.6f}",
        f"  {v2[0]:.6f}  {v2[1]:.6f}  {v2[2]:.6f}",
        f"  {v3[0]:.6f}  {v3[1]:.6f}  {v3[2]:.6f}",
    ]

    # Element symbols (sorted unique)
    unique_elems = sorted(set(elements))
    lines.append("  " + "  ".join(unique_elems))

    # Atom counts per element (matching symbol order)
    counts = [elem_counts[e] for e in unique_elems]
    lines.append("  " + "  ".join(str(c) for c in counts))

    if selective_dynamics:
        lines.append("Selective dynamics")

    lines.append("Direct")

    for i, pos in enumerate(positions):
        line = f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}"
        if selective_dynamics and selective_mask and i < len(selective_mask):
            mask = selective_mask[i]
            line += "  " + "  ".join("T" if m else "F" for m in mask)
        lines.append(line)

    poscar_content = "\n".join(lines) + "\n"

    if output_file:
        Path(output_file).write_text(poscar_content)

    return poscar_content


def generate_vasp_kpoints(
    kx: int = 8,
    ky: int = 8,
    kz: int = 8,
    shift: list[float] | None = None,
    output_file: str = "",
) -> str:
    """Generate a VASP KPOINTS file.

    Args:
        kx: Number of k-points along x.
        ky: Number of k-points along y.
        kz: Number of k-points along z.
        shift: Monkhorst-Pack shift [sx, sy, sz].
        output_file: If provided, write KPOINTS to this file.
    """
    if shift is None:
        shift = [0.0, 0.0, 0.0]

    lines = [
        "SciMCP generated KPOINTS",
        "0",
        "Monkhorst-Pack",
        f"{kx}  {ky}  {kz}",
        f"{shift[0]}  {shift[1]}  {shift[2]}",
    ]

    kpoints_content = "\n".join(lines) + "\n"

    if output_file:
        Path(output_file).write_text(kpoints_content)

    return kpoints_content


def generate_qe_pw_input(
    calculation: str = "scf",
    pseudo_dir: str = "",
    prefix: str = "scimcp",
    atom_types: list[str] | None = None,
    atom_positions: list[list[float]] | None = None,
    cell_angles_deg: list[float] | None = None,
    cell_dimensions: list[float] | None = None,
    ecutwfc: float = 30.0,
    ecutrho: float = 240.0,
    k_points: list[int] | None = None,
    conv_thr: float = 1e-6,
    nstep: int = 100,
    occupations: str = "smearing",
    degauss: float = 0.01,
    smearing: str = "gaussian",
    mixing_beta: float = 0.7,
    electron_maxstep: int = 100,
    ion_maxstep: int = 50,
    cell_factor: float = 2.0,
    output_file: str = "",
) -> str:
    """Generate a Quantum ESPRESSO pw.x input file.

    Args:
        calculation: Type of calculation (scf, nscf, relax, vc-relax, md).
        pseudo_dir: Path to pseudopotential directory.
        prefix: Calculation prefix.
        atom_positions: List of fractional coordinates [[x, y, z], ...].
        cell_dimensions: Cell vectors [[ax,ay,az],[bx,by,bz],[cx,cy,cz]].
        ecutwfc: Plane wave cutoff (Ry).
        ecutrho: Charge density cutoff (Ry).
        k_points: k-point grid [nx, ny, nz].
        conv_thr: Convergence threshold.
        nstep: Number of steps.
        occupations: Occupation method (smearing, tetrahedra, fixed).
        degauss: Smearing width (Ry).
        smearing: Smearing type (gaussian, methfessel-paxton, etc.).
        mixing_beta: Mixing parameter for charge density.
        electron_maxstep: Max electronic steps.
        ion_maxstep: Max ionic steps (for relax).
        cell_factor: Cell factor for variable cell.
        output_file: If provided, write input to this file.
    """
    if atom_positions is None:
        atom_positions = [[0.0, 0.0, 0.0]]
    if cell_dimensions is None:
        cell_dimensions = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    if k_points is None:
        k_points = [4, 4, 4]

    calculation = calculation.lower()
    relax_types = {"relax", "vc-relax", "md"}

    lines = [
        "&CONTROL",
        f"  calculation = '{calculation}',",
        f"  prefix = '{prefix}',",
        f"  pseudo_dir = '{pseudo_dir}'," if pseudo_dir else "  pseudo_dir = './',",
        f"  nstep = {nstep},",
        "  disk_io = 'low',",
        "/",
        "",
        "&SYSTEM",
        f"  ibrav = 0,",
        f"  nat = {len(atom_positions)},",
        f"  ntyp = 1,",
        f"  ecutwfc = {ecutwfc},",
        f"  ecutrho = {ecutrho},",
        f"  occupations = '{occupations}',",
        f"  degauss = {degauss},",
        f"  smearing = '{smearing}',",
        "/",
        "",
        "&ELECTRONS",
        f"  conv_thr = {conv_thr:.1e},",
        f"  mixing_beta = {mixing_beta},",
        f"  electron_maxstep = {electron_maxstep},",
        "/",
    ]

    if calculation in relax_types:
        lines += [
            "",
            "&IONS",
            f"  ion_maxstep = {ion_maxstep},",
            f"  upscale = 100,",
            "/",
        ]

    if calculation == "vc-relax":
        lines += [
            "",
            "&CELL",
            f"  cell_factor = {cell_factor},",
            f"  press_conv_thr = 0.1,",
            "/",
        ]

    lines += [
        "",
        "ATOMIC_SPECIES",
        "  Si  28.086  Si.pbe-n-kjpaw_psl.1.0.0.UPF",
        "",
        "ATOMIC_POSITIONS crystal",
    ]

    for i, pos in enumerate(atom_positions):
        lines.append(f"  Si  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")

    lines += [
        "",
        "CELL_PARAMETERS angstrom",
    ]
    for vec in cell_dimensions:
        lines.append(f"  {vec[0]:.6f}  {vec[1]:.6f}  {vec[2]:.6f}")

    lines += [
        "",
        f"K_POINTS automatic",
        f"  {k_points[0]}  {k_points[1]}  {k_points[2]}  1  1  1",
    ]

    pw_content = "\n".join(lines) + "\n"

    if output_file:
        Path(output_file).write_text(pw_content)

    return pw_content
