"""Modular MCP tool registration for SciMCP."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server import MCPServer


def register_lammps_tools(server: MCPServer) -> None:
    """Register all LAMMPS-related MCP tools."""
    from ..tools.lammps.generator import generate_lammps_input, generate_workflow
    from ..tools.lammps.parser import parse_thermo_data, parse_dump_file, get_summary
    from ..tools.lammps.nematic import (
        compute_nematic_order,
        compute_nematic_alignment_vs_time,
        compute_nematic_alignment_vs_z,
        compute_q_tensor_components,
    )
    from ..tools.lammps.nonaffine import (
        compute_nonaffine_displacement,
        identify_plastic_events,
    )
    from ..tools.lammps.shear import (
        generate_sweep_scripts,
        write_sweep_scripts,
        estimate_viscosity,
        SweepConfig,
    )
    import json
    import numpy as np

    @server.tool()
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
        """Generate a LAMMPS input script for MD simulation.

        Supports LJ, EAM, Tersoff, Buckingham, Coulomb potentials.
        Ensembles: NVT, NPT, NVE, NPH. Optional minimization and shear.

        Args:
            atoms_per_side: Number of unit cells per side.
            lattice_spacing: Lattice constant in Angstroms.
            potential: Interatomic potential type.
            epsilon: LJ energy parameter.
            sigma: LJ length parameter.
            cutoff: Pair potential cutoff.
            ensemble: Thermodynamic ensemble.
            temperature: Target temperature (K).
            pressure: Target pressure (atm).
            timestep: Integration timestep (fs).
            n_steps: Total MD steps.
            dump_freq: Trajectory dump frequency.
            minimize: Run energy minimization first.
            apply_shear: Apply shear deformation.
            shear_rate: Shear strain rate.
            shear_direction: Shear plane (xy, xz, yz).
            boundary: Boundary conditions.
            output_file: If provided, write script to this file.
        """
        return generate_lammps_input(
            atoms_per_side=atoms_per_side, lattice_spacing=lattice_spacing,
            potential=potential, epsilon=epsilon, sigma=sigma, cutoff=cutoff,
            ensemble=ensemble, temperature=temperature, pressure=pressure,
            timestep=timestep, n_steps=n_steps, dump_freq=dump_freq,
            minimize=minimize, apply_shear=apply_shear, shear_rate=shear_rate,
            shear_direction=shear_direction, boundary=boundary,
            output_file=output_file,
        )

    @server.tool()
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
        """Generate multi-step LAMMPS workflow: minimize → equilibrate → production.

        Args:
            atoms_per_side: Number of unit cells per side.
            lattice_spacing: Lattice constant.
            potential: Interatomic potential type.
            epsilon: LJ energy parameter.
            sigma: LJ length parameter.
            cutoff: Pair potential cutoff.
            timestep: Integration timestep.
            n_minimize_steps: Minimization steps (0 to skip).
            n_equilibrate_steps: Equilibration steps.
            equilibrate_temp: Equilibration temperature.
            equilibrate_ensemble: Equilibration ensemble.
            n_production_steps: Production steps.
            production_temp: Production temperature.
            production_ensemble: Production ensemble.
            production_pressure: Production pressure.
            apply_shear: Apply shear during production.
            shear_rate: Shear strain rate.
            shear_direction: Shear plane.
            dump_freq: Dump frequency during production.
            thermo_freq: Thermodynamic output frequency.
            output_file: Optional file path.
        """
        return generate_workflow(
            atoms_per_side=atoms_per_side, lattice_spacing=lattice_spacing,
            potential=potential, epsilon=epsilon, sigma=sigma, cutoff=cutoff,
            timestep=timestep, n_minimize_steps=n_minimize_steps,
            n_equilibrate_steps=n_equilibrate_steps,
            equilibrate_temp=equilibrate_temp,
            equilibrate_ensemble=equilibrate_ensemble,
            n_production_steps=n_production_steps,
            production_temp=production_temp,
            production_ensemble=production_ensemble,
            production_pressure=production_pressure,
            apply_shear=apply_shear, shear_rate=shear_rate,
            shear_direction=shear_direction, dump_freq=dump_freq,
            thermo_freq=thermo_freq, output_file=output_file,
        )

    @server.tool()
    def lammps_parse_thermo(filepath: str) -> str:
        """Parse LAMMPS thermo output and return time-series statistics.

        Args:
            filepath: Path to the LAMMPS thermo output file.
        """
        data = parse_thermo_data(filepath)
        result = {"n_steps": data["n_steps"], "columns": data["headers"], "summary": {}}
        for col_name, col_data in data["columns"].items():
            result["summary"][col_name] = {
                "min": float(col_data.min()), "max": float(col_data.max()),
                "mean": float(col_data.mean()), "std": float(col_data.std()),
                "last": float(col_data[-1]) if len(col_data) > 0 else None,
            }
        return json.dumps(result, indent=2)

    @server.tool()
    def lammps_parse_dump(filepath: str, max_frames: int = 10) -> str:
        """Parse LAMMPS dump file and return trajectory summaries.

        Args:
            filepath: Path to the LAMMPS dump file.
            max_frames: Maximum frames to parse (0 = all).
        """
        data = parse_dump_file(filepath, max_frames=max_frames)
        result = {"n_frames": data["n_frames"], "n_atoms": data["n_atoms"], "frames": []}
        for frame in data["frames"][:max_frames]:
            frame_info = {
                "step": frame.get("step", -1), "n_atoms": frame.get("n_atoms", 0),
                "columns": frame.get("columns", []), "box": frame.get("box", []),
            }
            if "data" in frame and hasattr(frame["data"], "shape"):
                frame_info["shape"] = list(frame["data"].shape)
            result["frames"].append(frame_info)
        return json.dumps(result, indent=2)

    @server.tool()
    def lammps_file_summary(filepath: str) -> str:
        """Get summary of a LAMMPS output file (dump or thermo).

        Args:
            filepath: Path to the LAMMPS output file.
        """
        return json.dumps(get_summary(filepath), indent=2)

    @server.tool()
    def lammps_nematic_order(quaternions_json: str) -> str:
        """Compute scalar nematic order parameter S from quaternion data.

        S=1: perfect alignment, S=0: isotropic, S=-0.5: perpendicular.

        Args:
            quaternions_json: JSON array of [w,x,y,z] quaternions, shape (N,4).
        """
        quaternions = np.array(json.loads(quaternions_json))
        S = compute_nematic_order(quaternions)
        q_components = compute_q_tensor_components(quaternions)
        return json.dumps({
            "nematic_order_S": S, "q_tensor": q_components,
            "interpretation": (
                "perfect alignment" if S > 0.9 else "strong alignment" if S > 0.6
                else "moderate alignment" if S > 0.3 else "weak alignment" if S > 0.1
                else "isotropic"
            ),
        }, indent=2)

    @server.tool()
    def lammps_nematic_vs_z(quaternions_json: str, z_positions_json: str, n_bins: int = 20) -> str:
        """Compute nematic alignment profile S(z) along the z-axis.

        Args:
            quaternions_json: JSON array of quaternions, shape (N,4).
            z_positions_json: JSON array of z-coordinates, shape (N,).
            n_bins: Number of bins along z.
        """
        quaternions = np.array(json.loads(quaternions_json))
        z_pos = np.array(json.loads(z_positions_json))
        result = compute_nematic_alignment_vs_z(quaternions, z_pos, n_bins)
        return json.dumps({
            "z_centers": result["z_centers"].tolist(),
            "S": result["S"].tolist(),
            "counts": result["counts"].tolist(),
        }, indent=2)

    @server.tool()
    def lammps_nematic_vs_time(quaternions_frames_json: str) -> str:
        """Compute nematic alignment S(t) over a trajectory.

        Args:
            quaternions_frames_json: List of (N,4) arrays, one per frame.
        """
        frames = [np.array(f) for f in json.loads(quaternions_frames_json)]
        result = compute_nematic_alignment_vs_time(frames)
        return json.dumps({
            "t": result["t"].tolist(), "S": result["S"].tolist(),
            "mean_S": result["mean_S"], "std_S": result["std_S"],
        }, indent=2)

    @server.tool()
    def lammps_d2min(positions_t0_json: str, positions_t1_json: str, r_cut: float = 3.0) -> str:
        """Compute non-affine displacement D²min between two frames.

        High D²min indicates plastic rearrangement events.

        Args:
            positions_t0_json: JSON array of positions at t0, shape (N,3).
            positions_t1_json: JSON array of positions at t1, shape (N,3).
            r_cut: Cutoff radius for neighbor finding.
        """
        pos0 = np.array(json.loads(positions_t0_json))
        pos1 = np.array(json.loads(positions_t1_json))
        d2min = compute_nonaffine_displacement(pos0, pos1, r_cut=r_cut)
        return json.dumps({
            "d2min": d2min.tolist(), "mean_d2min": float(d2min.mean()),
            "max_d2min": float(d2min.max()),
            "n_plastic": int((d2min > 0.1).sum()),
            "fraction_plastic": float((d2min > 0.1).mean()),
        }, indent=2)

    @server.tool()
    def lammps_identify_plastic_events(d2min_json: str, threshold: float = 0.1) -> str:
        """Identify particles that underwent plastic rearrangement.

        Args:
            d2min_json: JSON array of D²min values.
            threshold: D²min threshold for plastic events.
        """
        d2min = np.array(json.loads(d2min_json))
        result = identify_plastic_events(d2min, threshold)
        return json.dumps({
            "plastic_particles": result["plastic_particles"].tolist(),
            "n_plastic": result["n_plastic"],
            "fraction_plastic": result["fraction_plastic"],
            "mean_d2min_plastic": result["mean_d2min_plastic"],
        }, indent=2)

    @server.tool()
    def lammps_shear_sweep(
        shear_rates: str = "[0.0001, 0.0005, 0.001, 0.005, 0.01]",
        atoms_per_side: int = 10, lattice_spacing: float = 5.26,
        potential: str = "lj", temperature: float = 300.0,
        n_equilibrate: int = 10000, n_production: int = 100000,
        shear_direction: str = "xy", output_dir: str = "shear_sweep",
    ) -> str:
        """Generate LAMMPS scripts for a shear-rate sweep.

        Args:
            shear_rates: JSON array of shear rates.
            atoms_per_side: Unit cells per side.
            lattice_spacing: Lattice constant.
            potential: Interatomic potential type.
            temperature: Simulation temperature.
            n_equilibrate: Equilibration steps.
            n_production: Production steps.
            shear_direction: Shear plane.
            output_dir: Output directory.
        """
        rates = json.loads(shear_rates)
        config = SweepConfig(
            shear_rates=rates, atoms_per_side=atoms_per_side,
            lattice_spacing=lattice_spacing, potential=potential,
            temperature=temperature, n_equilibrate=n_equilibrate,
            n_production=n_production, shear_direction=shear_direction,
            output_dir=output_dir,
        )
        file_map = write_sweep_scripts(config)
        return json.dumps({
            "n_scripts": len(rates), "shear_rates": rates,
            "output_dir": output_dir, "files": file_map,
            "next_steps": [f"Run each script with: lmp -in {fp}" for fp in file_map.values() if not fp.endswith(".json")],
        }, indent=2)

    @server.tool()
    def lammps_estimate_viscosity(shear_rates_json: str, shear_stresses_json: str) -> str:
        """Estimate viscosity from shear rate vs stress data.

        Fits power-law: stress = K * rate^n.
        n=1: Newtonian, n<1: shear-thinning, n>1: shear-thickening.

        Args:
            shear_rates_json: JSON array of shear rates.
            shear_stresses_json: JSON array of shear stresses.
        """
        rates = np.array(json.loads(shear_rates_json))
        stresses = np.array(json.loads(shear_stresses_json))
        return json.dumps(estimate_viscosity(rates, stresses), indent=2)


def register_dft_tools(server: MCPServer) -> None:
    """Register all DFT/CIF-related MCP tools."""
    from ..tools.dft.cif import _parse_cif_text, generate_cif
    from ..tools.dft.vasp_qe import (
        generate_vasp_incar, generate_vasp_poscar,
        generate_vasp_kpoints, generate_qe_pw_input,
    )
    import json

    @server.tool()
    def dft_parse_cif(cif_content: str) -> str:
        """Parse CIF file and extract space group, lattice, atoms.

        Args:
            cif_content: Full CIF file content as string.
        """
        result = _parse_cif_text(cif_content)
        if "error" in result:
            return json.dumps(result, indent=2)
        return json.dumps({
            "space_group": result.get("symmetry", {}),
            "lattice": result.get("lattice", {}),
            "n_atoms": len(result.get("atoms", [])),
            "formula": "".join(f"{a['element']}" for a in result.get("atoms", [])[:4]),
            "atom_sites": result.get("atoms", [])[:10],
            "metadata_keys": list(result.get("metadata", {}).keys()),
        }, indent=2)

    @server.tool()
    def dft_generate_cif(
        space_group: int = 225,
        lattice_params: str = '{"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90}',
        atom_types: str = '["Si"]',
        positions: str = '[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]',
        formula: str = "", output_file: str = "",
    ) -> str:
        """Generate CIF from crystallographic parameters.

        Args:
            space_group: International space group number (1-230).
            lattice_params: JSON object with a, b, c, alpha, beta, gamma.
            atom_types: JSON array of element symbols.
            positions: JSON array of fractional coordinates [x, y, z].
            formula: Chemical formula (auto-generated if empty).
            output_file: Optional file path to write CIF.
        """
        lattice = json.loads(lattice_params)
        atoms = json.loads(atom_types)
        coords = json.loads(positions)
        result = generate_cif(
            elements=atoms, positions=coords, lattice_params=lattice,
            space_group=str(space_group), label=formula or "generated",
            output_file=output_file or "",
        )
        return json.dumps({"cif_content": result, "formula": formula}, indent=2)

    @server.tool()
    def dft_cif_summary(cif_content: str) -> str:
        """Get quick summary of a CIF file: formula, space group, lattice.

        Args:
            cif_content: Full CIF file content as string.
        """
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

    @server.tool()
    def dft_vasp_incar(
        encut: float = 520, ediff: float = 1e-6, isif: int = 3,
        ibrion: int = -1, nsw: int = 0, potim: float = 0.5,
        ispin: int = 1, lorbit: int = 11, lwave: bool = True,
        lcharg: bool = True, kspacing: float = 0.5, sigma: float = 0.1,
        ismear: int = 1, nelm: int = 60, ediffg: float = -0.01,
        output_file: str = "",
    ) -> str:
        """Generate VASP INCAR for electronic structure calculations.

        Args:
            encut: Plane-wave cutoff (eV).
            ediff: Electronic convergence criterion.
            isif: Ion relaxation mode.
            ibrion: Ionic relaxation algorithm.
            nsw: Ionic steps (0=static).
            potim: MD timestep (fs).
            ispin: Spin polarization (1=non-mag, 2=mag).
            lorbit: DOS projection.
            lwave: Write WAVECAR.
            lcharg: Write CHGCAR.
            kspacing: k-point spacing.
            sigma: Smearing width (eV).
            ismear: Smearing method.
            nelm: Max electronic steps.
            ediffg: Force convergence (eV/A).
            output_file: Optional file path.
        """
        result = generate_vasp_incar(
            encut=encut, ediff=ediff, isif=isif, ibrion=ibrion, nsw=nsw,
            potim=potim, ispin=ispin, lorbit=lorbit, lwave=lwave, lcharg=lcharg,
            kspacing=kspacing, sigma=sigma, ismear=ismear, nelm=nelm,
            ediffg=ediffg, output_file=output_file or "",
        )
        return json.dumps({"incar_content": result, "output_file": output_file}, indent=2)

    @server.tool()
    def dft_vasp_poscar(
        elements: str, positions: str, lattice_params: str = "",
        selective_dynamics: bool = False, comment: str = "SciMCP generated",
        output_file: str = "",
    ) -> str:
        """Generate VASP POSCAR for crystal structure input.

        Args:
            elements: JSON array of element symbols.
            positions: JSON array of fractional coordinates.
            lattice_params: JSON object with a, b, c, alpha, beta, gamma.
            selective_dynamics: Use selective dynamics.
            comment: Comment line.
            output_file: Optional file path.
        """
        elems = json.loads(elements)
        coords = json.loads(positions)
        lattice = json.loads(lattice_params) if lattice_params else None
        result = generate_vasp_poscar(
            elements=elems, positions=coords, lattice_params=lattice,
            selective_dynamics=selective_dynamics, comment=comment,
            output_file=output_file or "",
        )
        return json.dumps({"poscar_content": result, "output_file": output_file}, indent=2)

    @server.tool()
    def dft_vasp_kpoints(
        kx: int = 8, ky: int = 8, kz: int = 8,
        shift: str = "[0,0,0]", output_file: str = "",
    ) -> str:
        """Generate VASP KPOINTS for k-point sampling.

        Args:
            kx: k-points along x.
            ky: k-points along y.
            kz: k-points along z.
            shift: Monkhorst-Pack shift [sx, sy, sz].
            output_file: Optional file path.
        """
        shift_list = json.loads(shift)
        result = generate_vasp_kpoints(kx=kx, ky=ky, kz=kz, shift=shift_list, output_file=output_file or "")
        return json.dumps({"kpoints_content": result, "output_file": output_file}, indent=2)

    @server.tool()
    def dft_qe_pw_input(
        calculation: str = "scf", pseudo_dir: str = "", prefix: str = "scimcp",
        atom_positions: str = "[[0,0,0]]",
        cell_dimensions: str = "[[5,0,0],[0,5,0],[0,0,5]]",
        ecutwfc: float = 30.0, ecutrho: float = 240.0,
        k_points: str = "[4,4,4]", conv_thr: float = 1e-6,
        nstep: int = 100, output_file: str = "",
    ) -> str:
        """Generate Quantum ESPRESSO pw.x input file.

        Args:
            calculation: Type (scf, nscf, relax, vc-relax, md).
            pseudo_dir: Pseudopotential directory.
            prefix: Calculation prefix.
            atom_positions: JSON array of fractional coordinates.
            cell_dimensions: JSON array of cell vectors.
            ecutwfc: Plane-wave cutoff (Ry).
            ecutrho: Charge density cutoff (Ry).
            k_points: JSON array [nx, ny, nz].
            conv_thr: Convergence threshold.
            nstep: Number of steps.
            output_file: Optional file path.
        """
        positions = json.loads(atom_positions)
        cell = json.loads(cell_dimensions)
        kpts = json.loads(k_points)
        result = generate_qe_pw_input(
            calculation=calculation, pseudo_dir=pseudo_dir, prefix=prefix,
            atom_positions=positions, cell_dimensions=cell, ecutwfc=ecutwfc,
            ecutrho=ecutrho, k_points=kpts, conv_thr=conv_thr, nstep=nstep,
            output_file=output_file or "",
        )
        return json.dumps({"pw_content": result, "output_file": output_file}, indent=2)


def register_materials_tools(server: MCPServer) -> None:
    """Register materials database and ML prediction MCP tools."""
    from ..tools.materials.mxene import (
        query_mxene, get_mxene_list, get_mxene_properties,
        compare_mxenes, search_mxene_by_property,
    )
    from ..tools.materials.prediction import (
        predict_band_gap, predict_density, predict_melting_point,
        get_element_info, predict_all, compute_composition_features,
    )
    from ..tools.materials.materials_project import (
        query_materials_project, get_material_details, search_stable_materials,
    )
    import json

    @server.tool()
    def materials_mxene_list() -> str:
        """List all MXenes in the built-in database."""
        return json.dumps(get_mxene_list(), indent=2)

    @server.tool()
    def materials_mxene_query(
        formula: str = "", m_element: str = "", x_element: str = "",
        termination: str = "", metallic_only: bool = False,
    ) -> str:
        """Query MXene database by formula, elements, termination, or metallic character.

        Args:
            formula: Filter by chemical formula (substring match).
            m_element: Transition metal (e.g., 'Ti', 'V').
            x_element: Light element (e.g., 'C', 'N').
            termination: Surface termination (e.g., 'O', 'F', 'OH').
            metallic_only: Return only metallic MXenes.
        """
        return json.dumps(query_mxene(
            formula=formula, M_element=m_element, X_element=x_element,
            termination=termination, metallic_only=metallic_only,
        ), indent=2)

    @server.tool()
    def materials_mxene_properties(formula: str) -> str:
        """Get detailed properties for a specific MXene.

        Args:
            formula: MXene formula (e.g., 'Ti3C2', 'V2C').
        """
        return json.dumps(get_mxene_properties(formula), indent=2)

    @server.tool()
    def materials_mxene_compare(formulas: str) -> str:
        """Compare properties of multiple MXenes side by side.

        Args:
            formulas: JSON array of formulas, e.g., '["Ti3C2", "V2C"]'.
        """
        return json.dumps(compare_mxenes(json.loads(formulas)), indent=2)

    @server.tool()
    def materials_mxene_search(property_name: str, min_value: float, max_value: float) -> str:
        """Search MXenes within a property range.

        Args:
            property_name: Property (conductivity, band_gap, mass, density).
            min_value: Minimum value.
            max_value: Maximum value.
        """
        return json.dumps(search_mxene_by_property(property_name, min_value, max_value), indent=2)

    @server.tool()
    def materials_predict_band_gap(composition: str) -> str:
        """Predict band gap using ML.

        Args:
            composition: Chemical formula (e.g., 'GaAs', 'Si').
        """
        return json.dumps(predict_band_gap(composition), indent=2)

    @server.tool()
    def materials_predict_density(composition: str) -> str:
        """Predict density using ML.

        Args:
            composition: Chemical formula.
        """
        return json.dumps(predict_density(composition), indent=2)

    @server.tool()
    def materials_predict_melting_point(composition: str) -> str:
        """Predict melting point using ML.

        Args:
            composition: Chemical formula.
        """
        return json.dumps(predict_melting_point(composition), indent=2)

    @server.tool()
    def materials_element_info(element: str) -> str:
        """Get element properties for ML feature computation.

        Args:
            element: Element symbol.
        """
        return json.dumps(get_element_info(element), indent=2)

    @server.tool()
    def materials_predict_all(composition: str) -> str:
        """Predict all properties (band gap, density, melting point).

        Args:
            composition: Chemical formula.
        """
        return json.dumps(predict_all(composition), indent=2)

    @server.tool()
    def materials_composition_features(composition: str) -> str:
        """Compute ML feature vector for a composition.

        Args:
            composition: Chemical formula.
        """
        return json.dumps(compute_composition_features(composition), indent=2)

    @server.tool()
    def materials_project_query(
        formula: str = "", material_id: str = "", elements: str = "",
        band_gap_range: str = "", metallic_only: bool = False, api_key: str = "",
    ) -> str:
        """Query Materials Project for crystal structures and properties.

        Args:
            formula: Chemical formula (substring match).
            material_id: MP material ID (e.g., 'mp-149').
            elements: Comma-separated element filter.
            band_gap_range: Comma-separated min,max band gap (eV).
            metallic_only: Return only metallic materials.
            api_key: Materials Project API key.
        """
        return json.dumps(query_materials_project(
            formula=formula, material_id=material_id, elements=elements,
            band_gap_range=band_gap_range, metallic_only=metallic_only, api_key=api_key,
        ), indent=2)

    @server.tool()
    def materials_project_details(material_id: str, api_key: str = "") -> str:
        """Get detailed properties for a Materials Project material.

        Args:
            material_id: MP material ID (e.g., 'mp-149').
            api_key: Materials Project API key.
        """
        return json.dumps(get_material_details(material_id, api_key=api_key), indent=2)

    @server.tool()
    def materials_project_stable(
        formula: str = "", max_e_above_hull: float = 0.01, api_key: str = "",
    ) -> str:
        """Search for thermodynamically stable materials near the hull.

        Args:
            formula: Chemical formula filter.
            max_e_above_hull: Max energy above hull (eV/atom).
            api_key: Materials Project API key.
        """
        return json.dumps(search_stable_materials(
            formula=formula, max_e_above_hull=max_e_above_hull, api_key=api_key,
        ), indent=2)


def register_analysis_tools(server: MCPServer) -> None:
    """Register analysis and visualization MCP tools."""
    from ..tools.analysis.phonon import (
        compute_phonon_dos, compute_thermodynamic_properties,
        generate_phonon_band_path, estimate_phonon_frequencies,
    )
    from ..tools.analysis.visualization import (
        plot_time_series, plot_histogram, plot_scatter,
        plot_phonon_dos, plot_thermo_dashboard,
    )

    @server.tool()
    def analysis_phonon_dos(frequencies_json: str, sigma: float = 0.5, n_points: int = 200) -> str:
        """Compute phonon density of states from frequency list.

        Args:
            frequencies_json: JSON array of frequencies (THz).
            sigma: Gaussian broadening (THz).
            n_points: Grid points in DOS.
        """
        return compute_phonon_dos(frequencies_json, sigma=sigma, n_points=n_points)

    @server.tool()
    def analysis_thermodynamic_properties(
        frequencies_json: str, temperature_K: float = 300.0, n_atoms: int = 1,
    ) -> str:
        """Compute thermodynamic properties from phonon frequencies.

        Args:
            frequencies_json: JSON array of frequencies (THz).
            temperature_K: Temperature (K).
            n_atoms: Atoms in unit cell.
        """
        return compute_thermodynamic_properties(frequencies_json, temperature_K=temperature_K, n_atoms=n_atoms)

    @server.tool()
    def analysis_phonon_band_path(
        lattice_params: str = '{"a":5.43,"b":5.43,"c":5.43,"alpha":90,"beta":90,"gamma":90}',
        crystal_system: str = "cubic", n_points: int = 50,
    ) -> str:
        """Generate high-symmetry k-path for phonon band structure.

        Args:
            lattice_params: JSON object with a, b, c, alpha, beta, gamma.
            crystal_system: Crystal system.
            n_points: Points between high-symmetry points.
        """
        import json as _json
        return generate_phonon_band_path(_json.loads(lattice_params), crystal_system=crystal_system, n_points=n_points)

    @server.tool()
    def analysis_phonon_estimate(composition: str, crystal_system: str = "cubic") -> str:
        """Estimate phonon frequency range from composition.

        Args:
            composition: Chemical formula.
            crystal_system: Crystal system.
        """
        return estimate_phonon_frequencies(composition, crystal_system=crystal_system)

    @server.tool()
    def viz_time_series(data_json: str, x_label: str = "Step", y_label: str = "Value", title: str = "Time Series", output_file: str = "") -> str:
        """Plot time series data.

        Args:
            data_json: JSON object with 'x' and 'y' arrays.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Plot title.
            output_file: Optional PNG output path.
        """
        return plot_time_series(data_json, x_label=x_label, y_label=y_label, title=title, output_file=output_file)

    @server.tool()
    def viz_histogram(data_json: str, n_bins: int = 50, x_label: str = "Value", y_label: str = "Count", title: str = "Distribution", output_file: str = "") -> str:
        """Plot histogram.

        Args:
            data_json: JSON array of values.
            n_bins: Number of bins.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Plot title.
            output_file: Optional PNG output path.
        """
        return plot_histogram(data_json, n_bins=n_bins, x_label=x_label, y_label=y_label, title=title, output_file=output_file)

    @server.tool()
    def viz_scatter(x_json: str, y_json: str, color_json: str = "", x_label: str = "X", y_label: str = "Y", title: str = "Scatter Plot", output_file: str = "") -> str:
        """Plot scatter, optionally colored by third variable.

        Args:
            x_json: JSON array of x values.
            y_json: JSON array of y values.
            color_json: Optional JSON array for color.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Plot title.
            output_file: Optional PNG output path.
        """
        return plot_scatter(x_json, y_json, color_json=color_json, x_label=x_label, y_label=y_label, title=title, output_file=output_file)

    @server.tool()
    def viz_phonon_dos(frequencies_json: str, dos_json: str, title: str = "Phonon DOS", output_file: str = "") -> str:
        """Plot phonon density of states.

        Args:
            frequencies_json: JSON array of frequencies (THz).
            dos_json: JSON array of DOS values.
            title: Plot title.
            output_file: Optional PNG output path.
        """
        return plot_phonon_dos(frequencies_json, dos_json, title=title, output_file=output_file)

    @server.tool()
    def viz_thermo_dashboard(thermo_data_json: str, output_file: str = "") -> str:
        """Generate 4-panel LAMMPS thermo dashboard.

        Args:
            thermo_data_json: JSON object with columns as keys.
            output_file: Optional PNG output path.
        """
        return plot_thermo_dashboard(thermo_data_json, output_file=output_file)


def register_literature_tools(server: MCPServer) -> None:
    """Register literature search MCP tools."""
    from ..tools.materials.literature import (
        search_arxiv, search_materials_science, search_by_author,
    )
    import json

    @server.tool()
    def literature_search_arxiv(query: str, max_results: int = 5, sort_by: str = "relevance") -> str:
        """Search arXiv for materials science papers.

        Args:
            query: Search query.
            max_results: Max results (1-20).
            sort_by: 'relevance', 'lastUpdatedDate', or 'submittedDate'.
        """
        return json.dumps(search_arxiv(query, max_results=max_results, sort_by=sort_by), indent=2)

    @server.tool()
    def literature_search_materials(topic: str, max_results: int = 5) -> str:
        """Search arXiv with materials-science domain optimization.

        Args:
            topic: Materials science topic.
            max_results: Max results (1-20).
        """
        return json.dumps(search_materials_science(topic, max_results=max_results), indent=2)

    @server.tool()
    def literature_search_by_author(author_name: str, max_results: int = 5) -> str:
        """Search arXiv for papers by a specific author.

        Args:
            author_name: Author name.
            max_results: Max results (1-20).
        """
        return json.dumps(search_by_author(author_name, max_results=max_results), indent=2)
