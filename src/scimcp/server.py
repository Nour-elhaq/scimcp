"""SciMCP — Scientific Computing MCP Server for Computational Materials Science.

This server exposes tools for LAMMPS molecular dynamics, DFT/CIF analysis,
and materials property computation to AI coding agents via the Model Context Protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from .tools.lammps.generator import generate_lammps_input, generate_workflow
from .tools.lammps.parser import parse_thermo_data, parse_dump_file, get_summary
from .tools.lammps.nematic import (
    compute_nematic_order,
    compute_nematic_alignment_vs_time,
    compute_nematic_alignment_vs_z,
    compute_q_tensor_components,
)
from .tools.lammps.nonaffine import (
    compute_nonaffine_displacement,
    compute_d2min_vs_strain,
    identify_plastic_events,
)
from .tools.lammps.shear import (
    generate_sweep_scripts,
    write_sweep_scripts,
    estimate_viscosity,
)
from .tools.dft.cif import parse_cif, generate_cif, get_cif_summary, cif_to_ase
from .tools.materials.mxene import (
    query_mxene,
    get_mxene_list,
    get_mxene_properties,
    compare_mxenes,
    search_mxene_by_property,
)
from .tools.materials.prediction import (
    predict_band_gap,
    predict_density,
    predict_melting_point,
    get_element_info,
    predict_all,
    compute_composition_features,
)
from .tools.materials.literature import (
    search_arxiv,
    search_materials_science,
    search_by_author,
)

mcp = MCPServer(
    "SciMCP",
    instructions=(
        "Scientific computing MCP server for computational materials science. "
        "Provides tools for LAMMPS molecular dynamics simulations, output parsing, "
        "nematic alignment analysis, non-affine displacement computation, "
        "automated shear-rate sweeps, DFT/CIF analysis, MXene database queries, "
        "ML property prediction, and arXiv literature search."
    ),
)


# ── LAMMPS Input Generation ──────────────────────────────────────────────


@mcp.tool()
def lammps_generate_input(
    atoms_per_side: int = 10,
    lattice_spacing: float = 5.26,
    potential: str = "lj",
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float = 2.5,
    ensemble: str = "nvt",
    temperature: float = 300.0,
    pressure: float = 0.0,
    timestep: float = 0.002,
    n_steps: int = 100000,
    dump_freq: int = 1000,
    minimize: bool = True,
    apply_shear: bool = False,
    shear_rate: float = 0.001,
    shear_direction: str = "xy",
    boundary: str = "p p p",
    output_file: str = "",
) -> str:
    """Generate a complete LAMMPS input script for molecular dynamics simulation.

    Creates a ready-to-run LAMMPS input file with support for LJ, EAM, Tersoff,
    Buckingham, and Coulomb potentials. Ensembles: NVT, NPT, NVE, NPH.
    Optional energy minimization and shear deformation.

    Args:
        atoms_per_side: Number of unit cells per side of the cubic box.
        lattice_spacing: Lattice constant in Angstroms.
        potential: Interatomic potential type (lj, eam, tersoff, buckingham, coulomb).
        epsilon: LJ energy parameter.
        sigma: LJ length parameter.
        cutoff: Pair potential cutoff distance.
        ensemble: Thermodynamic ensemble (nvt, npt, nve, nph).
        temperature: Target temperature.
        pressure: Target pressure (for NPT/NPH).
        timestep: Integration timestep.
        n_steps: Total number of MD steps.
        dump_freq: Frequency of trajectory dumps.
        minimize: Run energy minimization before dynamics.
        apply_shear: Apply shear deformation during dynamics.
        shear_rate: Shear strain rate.
        shear_direction: Shear plane (xy, xz, yz).
        boundary: Boundary conditions.
        output_file: If provided, write script to this file.
    """
    return generate_lammps_input(
        atoms_per_side=atoms_per_side,
        lattice_spacing=lattice_spacing,
        potential=potential,
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        ensemble=ensemble,
        temperature=temperature,
        pressure=pressure,
        timestep=timestep,
        n_steps=n_steps,
        dump_freq=dump_freq,
        minimize=minimize,
        apply_shear=apply_shear,
        shear_rate=shear_rate,
        shear_direction=shear_direction,
        boundary=boundary,
        output_file=output_file,
    )


@mcp.tool()
def lammps_generate_workflow(
    atoms_per_side: int = 10,
    lattice_spacing: float = 5.26,
    potential: str = "lj",
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float = 2.5,
    timestep: float = 0.002,
    n_minimize_steps: int = 0,
    n_equilibrate_steps: int = 10000,
    equilibrate_temp: float = 300.0,
    equilibrate_ensemble: str = "nvt",
    n_production_steps: int = 100000,
    production_temp: float = 300.0,
    production_ensemble: str = "nvt",
    production_pressure: float = 0.0,
    apply_shear: bool = False,
    shear_rate: float = 0.001,
    shear_direction: str = "xy",
    dump_freq: int = 100,
    thermo_freq: int = 10,
    output_file: str = "",
) -> str:
    """Generate a multi-step LAMMPS workflow (minimize → equilibrate → production).

    Creates a complete input script with three sequential phases:
    1. Energy minimization (optional)
    2. Equilibration (NVT or NPT)
    3. Production run (with optional shear)

    Args:
        atoms_per_side: Number of unit cells per side.
        lattice_spacing: Lattice constant.
        potential: Interatomic potential type.
        epsilon: LJ energy parameter.
        sigma: LJ length parameter.
        cutoff: Pair potential cutoff.
        timestep: Integration timestep.
        n_minimize_steps: Number of minimization steps (0 to skip).
        n_equilibrate_steps: Number of equilibration steps.
        equilibrate_temp: Temperature for equilibration.
        equilibrate_ensemble: Ensemble for equilibration.
        n_production_steps: Number of production steps.
        production_temp: Temperature for production.
        production_ensemble: Ensemble for production.
        production_pressure: Pressure for production (if NPT).
        apply_shear: Apply shear during production.
        shear_rate: Shear strain rate.
        shear_direction: Shear plane (xy, xz, yz).
        dump_freq: Dump frequency during production.
        thermo_freq: Thermodynamic output frequency.
        output_file: Optional file path to write the script.
    """
    return generate_workflow(
        atoms_per_side=atoms_per_side,
        lattice_spacing=lattice_spacing,
        potential=potential,
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        timestep=timestep,
        n_minimize_steps=n_minimize_steps,
        n_equilibrate_steps=n_equilibrate_steps,
        equilibrate_temp=equilibrate_temp,
        equilibrate_ensemble=equilibrate_ensemble,
        n_production_steps=n_production_steps,
        production_temp=production_temp,
        production_ensemble=production_ensemble,
        production_pressure=production_pressure,
        apply_shear=apply_shear,
        shear_rate=shear_rate,
        shear_direction=shear_direction,
        dump_freq=dump_freq,
        thermo_freq=thermo_freq,
        output_file=output_file,
    )


# ── LAMMPS Output Parsing ────────────────────────────────────────────────


@mcp.tool()
def lammps_parse_thermo(filepath: str) -> str:
    """Parse LAMMPS thermodynamic output and return time-series data.

    Reads a LAMMPS log file or thermo.dat file and extracts columns
    like Step, Temp, PotEng, TotEng, Press, etc.

    Args:
        filepath: Path to the LAMMPS thermo output file.
    """
    data = parse_thermo_data(filepath)
    result = {
        "n_steps": data["n_steps"],
        "columns": data["headers"],
        "summary": {},
    }
    for col_name, col_data in data["columns"].items():
        result["summary"][col_name] = {
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "last": float(col_data[-1]) if len(col_data) > 0 else None,
        }
    return json.dumps(result, indent=2)


@mcp.tool()
def lammps_parse_dump(filepath: str, max_frames: int = 10) -> str:
    """Parse a LAMMPS dump file and return trajectory frame summaries.

    Reads lammpstrj files and extracts atom positions, box dimensions,
    and metadata for each frame.

    Args:
        filepath: Path to the LAMMPS dump file.
        max_frames: Maximum number of frames to parse (0 = all).
    """
    data = parse_dump_file(filepath, max_frames=max_frames)
    result = {
        "n_frames": data["n_frames"],
        "n_atoms": data["n_atoms"],
        "frames": [],
    }
    for frame in data["frames"][:max_frames]:
        frame_info = {
            "step": frame.get("step", -1),
            "n_atoms": frame.get("n_atoms", 0),
            "columns": frame.get("columns", []),
            "box": frame.get("box", []),
        }
        if "data" in frame and hasattr(frame["data"], "shape"):
            frame_info["shape"] = list(frame["data"].shape)
        result["frames"].append(frame_info)
    return json.dumps(result, indent=2)


@mcp.tool()
def lammps_file_summary(filepath: str) -> str:
    """Get a summary of a LAMMPS output file (dump or thermo).

    Identifies the file type and returns key statistics: number of frames,
    atoms, columns, box dimensions, or thermo time-series stats.

    Args:
        filepath: Path to the LAMMPS output file.
    """
    summary = get_summary(filepath)
    return json.dumps(summary, indent=2)


# ── Nematic Alignment ─────────────────────────────────────────────────────


@mcp.tool()
def lammps_nematic_order(quaternions_json: str) -> str:
    """Compute the scalar nematic order parameter S from quaternion data.

    S measures alignment of anisotropic particles. S=1 is perfect alignment,
    S=0 is isotropic, S=-0.5 is perpendicular alignment.

    Input is a JSON array of quaternions [w, x, y, z] for each particle.

    Args:
        quaternions_json: JSON string with shape (N, 4) quaternion data.
            Example: "[[1,0,0,0], [0.9,0.1,0,0], [0.8,0.2,0.1,0]]"
    """
    import numpy as np
    quaternions = np.array(json.loads(quaternions_json))
    S = compute_nematic_order(quaternions)
    q_components = compute_q_tensor_components(quaternions)
    return json.dumps({
        "nematic_order_S": S,
        "q_tensor": q_components,
        "interpretation": (
            "perfect alignment" if S > 0.9
            else "strong alignment" if S > 0.6
            else "moderate alignment" if S > 0.3
            else "weak alignment" if S > 0.1
            else "isotropic"
        ),
    }, indent=2)


@mcp.tool()
def lammps_nematic_vs_z(
    quaternions_json: str,
    z_positions_json: str,
    n_bins: int = 20,
) -> str:
    """Compute nematic alignment profile S(z) along the z-axis.

    Bins particles by z-coordinate and computes S in each bin,
    useful for studying alignment near surfaces or interfaces.

    Args:
        quaternions_json: JSON string with shape (N, 4) quaternion data.
        z_positions_json: JSON string with shape (N,) z-coordinates.
        n_bins: Number of bins along z.
    """
    import numpy as np
    quaternions = np.array(json.loads(quaternions_json))
    z_pos = np.array(json.loads(z_positions_json))
    result = compute_nematic_alignment_vs_z(quaternions, z_pos, n_bins)
    return json.dumps({
        "z_centers": result["z_centers"].tolist(),
        "S": result["S"].tolist(),
        "counts": result["counts"].tolist(),
    }, indent=2)


@mcp.tool()
def lammps_nematic_vs_time(
    quaternions_frames_json: str,
) -> str:
    """Compute nematic alignment S(t) over a trajectory.

    Each frame contains quaternions for all particles at that timestep.

    Args:
        quaternions_frames_json: JSON string — list of (N, 4) arrays, one per frame.
            Example: "[[[1,0,0,0],[0.9,0.1,0,0]], [[0.8,0.2,0.1,0],[0.7,0.3,0,0]]]"
    """
    import numpy as np
    frames = [np.array(f) for f in json.loads(quaternions_frames_json)]
    result = compute_nematic_alignment_vs_time(frames)
    return json.dumps({
        "t": result["t"].tolist(),
        "S": result["S"].tolist(),
        "mean_S": result["mean_S"],
        "std_S": result["std_S"],
    }, indent=2)


# ── Non-Affine Displacement ──────────────────────────────────────────────


@mcp.tool()
def lammps_d2min(
    positions_t0_json: str,
    positions_t1_json: str,
    r_cut: float = 3.0,
) -> str:
    """Compute non-affine displacement D²min for each particle between two frames.

    D²min measures how much a particle's displacement deviates from the
    best-fit affine deformation of its neighbors. High D²min = plastic event.

    Args:
        positions_t0_json: JSON string, shape (N, 3) positions at time t0.
        positions_t1_json: JSON string, shape (N, 3) positions at time t1.
        r_cut: Cutoff radius for neighbor finding.
    """
    import numpy as np
    pos0 = np.array(json.loads(positions_t0_json))
    pos1 = np.array(json.loads(positions_t1_json))
    d2min = compute_nonaffine_displacement(pos0, pos1, r_cut=r_cut)
    return json.dumps({
        "d2min": d2min.tolist(),
        "mean_d2min": float(d2min.mean()),
        "max_d2min": float(d2min.max()),
        "n_plastic": int((d2min > 0.1).sum()),
        "fraction_plastic": float((d2min > 0.1).mean()),
    }, indent=2)


@mcp.tool()
def lammps_identify_plastic_events(
    d2min_json: str,
    threshold: float = 0.1,
) -> str:
    """Identify particles that underwent plastic rearrangement.

    Args:
        d2min_json: JSON string with D²min values for each particle.
        threshold: D²min threshold for plastic events.
    """
    import numpy as np
    d2min = np.array(json.loads(d2min_json))
    result = identify_plastic_events(d2min, threshold)
    return json.dumps({
        "plastic_particles": result["plastic_particles"].tolist(),
        "n_plastic": result["n_plastic"],
        "fraction_plastic": result["fraction_plastic"],
        "mean_d2min_plastic": result["mean_d2min_plastic"],
    }, indent=2)


# ── Shear Rate Sweep ──────────────────────────────────────────────────────


@mcp.tool()
def lammps_shear_sweep(
    shear_rates: str = "[0.0001, 0.0005, 0.001, 0.005, 0.01]",
    atoms_per_side: int = 10,
    lattice_spacing: float = 5.26,
    potential: str = "lj",
    temperature: float = 300.0,
    n_equilibrate: int = 10000,
    n_production: int = 100000,
    shear_direction: str = "xy",
    output_dir: str = "shear_sweep",
) -> str:
    """Generate LAMMPS input scripts for a shear-rate sweep.

    Creates one input script per shear rate to study rheological behavior,
    shear thinning/thickening, and viscosity as a function of shear rate.

    Args:
        shear_rates: JSON array of shear rates to simulate.
        atoms_per_side: Number of unit cells per side.
        lattice_spacing: Lattice constant.
        potential: Interatomic potential type.
        temperature: Simulation temperature.
        n_equilibrate: Number of equilibration steps.
        n_production: Number of production steps.
        shear_direction: Shear plane (xy, xz, yz).
        output_dir: Directory to write scripts.
    """
    from .tools.lammps.shear import SweepConfig

    rates = json.loads(shear_rates)
    config = SweepConfig(
        shear_rates=rates,
        atoms_per_side=atoms_per_side,
        lattice_spacing=lattice_spacing,
        potential=potential,
        temperature=temperature,
        n_equilibrate=n_equilibrate,
        n_production=n_production,
        shear_direction=shear_direction,
        output_dir=output_dir,
    )
    file_map = write_sweep_scripts(config)
    return json.dumps({
        "n_scripts": len(rates),
        "shear_rates": rates,
        "output_dir": output_dir,
        "files": file_map,
        "next_steps": [
            f"Run each script with: lmp -in {fp}" for fp in file_map.values() if fp.endswith(".json") is False
        ],
    }, indent=2)


@mcp.tool()
def lammps_estimate_viscosity(
    shear_rates_json: str,
    shear_stresses_json: str,
) -> str:
    """Estimate viscosity from shear rate vs shear stress data.

    Fits a power-law model: stress = K * rate^n, where n=1 is Newtonian,
    n<1 is shear-thinning, n>1 is shear-thickening.

    Args:
        shear_rates_json: JSON array of shear rates.
        shear_stresses_json: JSON array of corresponding shear stresses.
    """
    import numpy as np
    rates = np.array(json.loads(shear_rates_json))
    stresses = np.array(json.loads(shear_stresses_json))
    result = estimate_viscosity(rates, stresses)
    return json.dumps(result, indent=2)


# ── DFT / CIF Analysis ────────────────────────────────────────────────────


@mcp.tool()
def dft_parse_cif(cif_content: str) -> str:
    """Parse a CIF (Crystallographic Information File) and extract structure data.

    Returns space group, lattice parameters, atomic positions, and metadata.

    Args:
        cif_content: Full CIF file content as a string.
    """
    from .tools.dft.cif import _parse_cif_text
    result = _parse_cif_text(cif_content)
    if "error" in result:
        return json.dumps(result, indent=2)
    return json.dumps({
        "space_group": result.get("symmetry", {}),
        "lattice": result.get("lattice", {}),
        "n_atoms": len(result.get("atoms", [])),
        "formula": "".join(
            f"{a['element']}" for a in result.get("atoms", [])[:4]
        ),
        "atom_sites": result.get("atoms", [])[:10],
        "metadata_keys": list(result.get("metadata", {}).keys()),
    }, indent=2)


@mcp.tool()
def dft_generate_cif(
    space_group: int = 225,
    lattice_params: str = '{"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90}',
    atom_types: str = '["Si"]',
    positions: str = '[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]',
    formula: str = "",
    output_file: str = "",
) -> str:
    """Generate a CIF file from crystallographic parameters.

    Args:
        space_group: International space group number (1-230).
        lattice_params: JSON object with a, b, c, alpha, beta, gamma.
        atom_types: JSON array of element symbols.
        positions: JSON array of fractional coordinates [x, y, z].
        formula: Chemical formula (optional, auto-generated if empty).
        output_file: Optional file path to write the CIF.
    """
    import json as _json
    lattice = _json.loads(lattice_params)
    atoms = _json.loads(atom_types)
    coords = _json.loads(positions)
    result = generate_cif(
        elements=atoms,
        positions=coords,
        lattice_params=lattice,
        space_group=str(space_group),
        label=formula or "generated",
        output_file=output_file or "",
    )
    return json.dumps({"cif_content": result, "formula": formula}, indent=2)


@mcp.tool()
def dft_cif_summary(cif_content: str) -> str:
    """Get a quick summary of a CIF file: formula, space group, lattice.

    Args:
        cif_content: Full CIF file content as a string.
    """
    from .tools.dft.cif import _parse_cif_text
    from collections import Counter
    data = _parse_cif_text(cif_content)
    elements = [a["element"] for a in data.get("atoms", [])]
    composition = dict(Counter(elements))
    return json.dumps({
        "lattice": data.get("lattice", {}),
        "n_atoms": len(data.get("atoms", [])),
        "elements": list(set(elements)),
        "composition": composition,
        "formula": "".join(f"{e}{c}" if c > 1 else e for e, c in composition.items()),
        "symmetry": data.get("symmetry", {}),
    }, indent=2)


# ── MXene Database ────────────────────────────────────────────────────────


@mcp.tool()
def materials_mxene_list() -> str:
    """List all MXenes in the built-in database with key properties."""
    mxenes = get_mxene_list()
    return json.dumps(mxenes, indent=2)


@mcp.tool()
def materials_mxene_query(
    formula: str = "",
    m_element: str = "",
    x_element: str = "",
    termination: str = "",
    metallic_only: bool = False,
) -> str:
    """Query MXene database by formula, elements, termination, or metallic character.

    Returns matching MXene entries with all their properties.

    Args:
        formula: Filter by chemical formula (substring match).
        m_element: Transition metal element (e.g., 'Ti', 'V', 'Nb').
        x_element: Light element (e.g., 'C', 'N').
        termination: Surface termination (e.g., 'O', 'F', 'OH').
        metallic_only: If True, return only metallic MXenes.
    """
    result = query_mxene(
        formula=formula,
        M_element=m_element,
        X_element=x_element,
        termination=termination,
        metallic_only=metallic_only,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_mxene_properties(formula: str) -> str:
    """Get detailed properties for a specific MXene by formula.

    Args:
        formula: MXene formula (e.g., 'Ti3C2', 'Ti2C', 'V2C').
    """
    result = get_mxene_properties(formula)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_mxene_compare(formulas: str) -> str:
    """Compare properties of multiple MXenes side by side.

    Args:
        formulas: JSON array of MXene formulas, e.g., '["Ti3C2", "V2C", "Nb2C"]'.
    """
    import json as _json
    formula_list = _json.loads(formulas)
    result = compare_mxenes(formula_list)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_mxene_search(
    property_name: str,
    min_value: float,
    max_value: float,
) -> str:
    """Search MXenes within a range for a specific property.

    Args:
        property_name: Property to search (conductivity, band_gap, mass, density, etc.).
        min_value: Minimum value.
        max_value: Maximum value.
    """
    result = search_mxene_by_property(property_name, min_value, max_value)
    return json.dumps(result, indent=2)


# ── ML Property Prediction ────────────────────────────────────────────────


@mcp.tool()
def materials_predict_band_gap(composition: str) -> str:
    """Predict band gap for a material composition using ML.

    Args:
        composition: Chemical formula (e.g., 'GaAs', 'Si', 'InP').
    """
    result = predict_band_gap(composition)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_predict_density(composition: str) -> str:
    """Predict density for a material composition using ML.

    Args:
        composition: Chemical formula (e.g., 'Si', 'GaAs', 'Al2O3').
    """
    result = predict_density(composition)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_predict_melting_point(composition: str) -> str:
    """Predict melting point for a material composition using ML.

    Args:
        composition: Chemical formula (e.g., 'Si', 'Fe', 'Cu').
    """
    result = predict_melting_point(composition)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_element_info(element: str) -> str:
    """Get element properties useful for ML feature computation.

    Args:
        element: Element symbol (e.g., 'Si', 'Ga', 'As').
    """
    result = get_element_info(element)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_predict_all(composition: str) -> str:
    """Predict all available properties (band gap, density, melting point) for a composition.

    Args:
        composition: Chemical formula (e.g., 'GaAs', 'Si', 'InP').
    """
    result = predict_all(composition)
    return json.dumps(result, indent=2)


@mcp.tool()
def materials_composition_features(composition: str) -> str:
    """Compute ML feature vector for a composition (elemental properties, statistics).

    Args:
        composition: Chemical formula (e.g., 'GaAs').
    """
    result = compute_composition_features(composition)
    return json.dumps(result, indent=2)


# ── Literature Search ──────────────────────────────────────────────────────


@mcp.tool()
def literature_search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> str:
    """Search arXiv for materials science papers.

    Args:
        query: Search query (e.g., 'MXene electronic structure', 'LAMMPS shear simulation').
        max_results: Maximum number of results (1-20).
        sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'.
    """
    result = search_arxiv(query, max_results=max_results, sort_by=sort_by)
    return json.dumps(result, indent=2)


@mcp.tool()
def literature_search_materials(
    topic: str,
    max_results: int = 5,
) -> str:
    """Search arXiv for materials science papers with domain-specific queries.

    Adds 'materials science' context and common abbreviations to improve results.

    Args:
        topic: Materials science topic (e.g., 'MXene', 'perovskite solar cells', 'battery electrolyte').
        max_results: Maximum number of results (1-20).
    """
    result = search_materials_science(topic, max_results=max_results)
    return json.dumps(result, indent=2)


@mcp.tool()
def literature_search_by_author(
    author_name: str,
    max_results: int = 5,
) -> str:
    """Search arXiv for papers by a specific author.

    Args:
        author_name: Author name (e.g., 'John Smith').
        max_results: Maximum number of results (1-20).
    """
    result = search_by_author(author_name, max_results=max_results)
    return json.dumps(result, indent=2)


# ── Prompts ───────────────────────────────────────────────────────────────


@mcp.prompt()
def lammps_shearing_workflow() -> str:
    """Step-by-step guide for setting up a shear simulation in LAMMPS."""
    return """Guide the user through setting up a shear simulation in LAMMPS:

1. Use `lammps_generate_input` with `apply_shear=True` and the desired shear rate
2. Run the simulation with `lmp -in in.shear`
3. Parse results with `lammps_parse_thermo` for stress/pressure data
4. Parse trajectory with `lammps_parse_dump` for particle positions
5. Compute D²min with `lammps_d2min` to identify plastic events
6. For multiple shear rates, use `lammps_shear_sweep` and `lammps_estimate_viscosity`"""


@mcp.prompt()
def lammps_analysis_workflow() -> str:
    """Step-by-step guide for analyzing LAMMPS simulation output."""
    return """Guide the user through analyzing LAMMPS output:

1. Start with `lammps_file_summary` to understand the output file structure
2. For thermo data: `lammps_parse_thermo` — shows energy, temperature, pressure evolution
3. For trajectories: `lammps_parse_dump` — shows atom positions over time
4. For nematic systems: `lammps_nematic_order` — compute alignment parameter S
5. For shear/deformation: `lammps_d2min` — identify non-affine/plastic events
6. For rheology: `lammps_shear_sweep` + `lammps_estimate_viscosity` — viscosity vs shear rate"""


@mcp.prompt()
def dft_workflow() -> str:
    """Step-by-step guide for DFT structure analysis with CIF files."""
    return """Guide the user through DFT structure analysis:

1. Parse existing CIF: `dft_parse_cif` — extract space group, lattice, positions
2. Get quick summary: `dft_cif_summary` — formula, space group, lattice overview
3. Generate new CIF: `dft_generate_cif` — create CIF from crystallographic parameters
4. Export to ASE: use the underlying `cif_to_ase` function for VASP/QE input
5. Use `materials_predict_band_gap` to estimate electronic properties"""


@mcp.prompt()
def mxene_discovery_workflow() -> str:
    """Step-by-step guide for MXene materials discovery."""
    return """Guide the user through MXene discovery:

1. List all MXenes: `materials_mxene_list` — see available MXene compositions
2. Query by properties: `materials_mxene_query` — filter by conductivity, band gap, mass
3. Compare MXenes: `materials_mxene_compare` — side-by-side property comparison
4. Search by range: `materials_mxene_search` — find MXenes within property ranges
5. Get details: `materials_mxene_properties` — full property data for a specific MXene
6. Predict properties: `materials_predict_band_gap` — ML prediction for custom compositions"""


@mcp.prompt()
def ml_prediction_workflow() -> str:
    """Step-by-step guide for ML property prediction."""
    return """Guide the user through ML property prediction:

1. Get element info: `materials_element_info` — elemental properties and features
2. Compute features: `materials_composition_features` — ML feature vector for a composition
3. Predict band gap: `materials_predict_band_gap` — electronic band gap estimate
4. Predict density: `materials_predict_density` — mass density estimate
5. Predict melting point: `materials_predict_melting_point` — thermal property estimate
6. Predict all: `materials_predict_all` — all properties at once"""


@mcp.prompt()
def literature_review_workflow() -> str:
    """Step-by-step guide for literature search and review."""
    return """Guide the user through literature search:

1. Topic search: `literature_search_arxiv` — general arXiv search
2. Materials-specific: `literature_search_materials` — optimized for materials science
3. Author search: `literature_search_by_author` — find papers by specific authors
4. Refine queries: combine keywords like 'MXene', 'DFT', 'LAMMPS', 'perovskite'
5. Check dates: sort by 'submittedDate' for latest work
6. Export citations: results include arXiv IDs for easy citation"""


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    """Entry point for the SciMCP server."""
    import asyncio
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
