"""SciMCP — Modular MCP Server for Computational Materials Science.

Refactored server with modular tool registration, typed models, and
standardized error handling.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import MCPServer

from .mcp_tools import (
    register_lammps_tools,
    register_dft_tools,
    register_materials_tools,
    register_analysis_tools,
    register_literature_tools,
)

__version__ = "0.3.0"


def create_server() -> MCPServer:
    """Create and configure the SciMCP MCP server."""
    server = MCPServer(
        "SciMCP",
        instructions=(
            "Modular MCP server for reproducible computational materials-science workflows. "
            "Integrates simulation-input generation (LAMMPS, VASP, Quantum ESPRESSO), "
            "materials-data retrieval (MXene database, Materials Project, arXiv), "
            "machine-learning-assisted property prediction, scientific analysis "
            "(phonon DOS, thermodynamics, non-affine displacement, nematic order), "
            "and visualization through a unified natural-language interface. "
            "45 tools across 9 categories. All core databases operate locally "
            "without API keys. Scientific results should be validated independently."
        ),
    )

    register_lammps_tools(server)
    register_dft_tools(server)
    register_materials_tools(server)
    register_analysis_tools(server)
    register_literature_tools(server)

    @server.prompt()
    def lammps_shearing_workflow() -> str:
        """Step-by-step guide for LAMMPS shear simulation."""
        return (
            "Guide the user through setting up a shear simulation in LAMMPS:\n"
            "1. Use `lammps_generate_input` with `apply_shear=True`\n"
            "2. Run: `lmp -in in.shear`\n"
            "3. Parse: `lammps_parse_thermo` for stress data\n"
            "4. Trajectory: `lammps_parse_dump` for positions\n"
            "5. Plasticity: `lammps_d2min` for non-affine events\n"
            "6. Sweep: `lammps_shear_sweep` + `lammps_estimate_viscosity`"
        )

    @server.prompt()
    def lammps_analysis_workflow() -> str:
        """Step-by-step guide for LAMMPS output analysis."""
        return (
            "Guide the user through analyzing LAMMPS output:\n"
            "1. `lammps_file_summary` — understand file structure\n"
            "2. `lammps_parse_thermo` — energy/temperature/pressure evolution\n"
            "3. `lammps_parse_dump` — atom positions over time\n"
            "4. `lammps_nematic_order` — alignment parameter S\n"
            "5. `lammps_d2min` — non-affine/plastic events\n"
            "6. `lammps_shear_sweep` + `lammps_estimate_viscosity` — rheology"
        )

    @server.prompt()
    def dft_workflow() -> str:
        """Step-by-step guide for DFT structure analysis."""
        return (
            "Guide the user through DFT structure analysis:\n"
            "1. `dft_parse_cif` — extract space group, lattice, atoms\n"
            "2. `dft_cif_summary` — formula, space group overview\n"
            "3. `dft_generate_cif` — create CIF from parameters\n"
            "4. `dft_vasp_incar` + `dft_vasp_poscar` + `dft_vasp_kpoints` — VASP input\n"
            "5. `dft_qe_pw_input` — Quantum ESPRESSO input"
        )

    @server.prompt()
    def mxene_discovery_workflow() -> str:
        """Step-by-step guide for MXene materials discovery."""
        return (
            "Guide the user through MXene discovery:\n"
            "1. `materials_mxene_list` — see available MXene compositions\n"
            "2. `materials_mxene_query` — filter by properties\n"
            "3. `materials_mxene_compare` — side-by-side comparison\n"
            "4. `materials_mxene_search` — range search by property\n"
            "5. `materials_predict_band_gap` — ML prediction for custom compositions"
        )

    @server.prompt()
    def ml_prediction_workflow() -> str:
        """Step-by-step guide for ML property prediction."""
        return (
            "Guide the user through ML property prediction:\n"
            "1. `materials_element_info` — elemental properties\n"
            "2. `materials_composition_features` — ML feature vector\n"
            "3. `materials_predict_band_gap` — electronic band gap\n"
            "4. `materials_predict_density` — mass density\n"
            "5. `materials_predict_melting_point` — thermal properties\n"
            "6. `materials_predict_all` — all properties at once"
        )

    @server.prompt()
    def literature_review_workflow() -> str:
        """Step-by-step guide for literature search."""
        return (
            "Guide the user through literature search:\n"
            "1. `literature_search_arxiv` — general search\n"
            "2. `literature_search_materials` — materials-science optimized\n"
            "3. `literature_search_by_author` — find by author\n"
            "4. Refine: combine keywords like 'MXene', 'DFT', 'LAMMPS'\n"
            "5. Sort by 'submittedDate' for latest work"
        )

    return server


mcp = create_server()


def main():
    """Entry point for the SciMCP server."""
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
