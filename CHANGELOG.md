# Changelog

All notable changes to SciMCP will be documented in this file.

## [0.3.0] - 2026-08-01

### Added
- **VASP/QE Input Generation** (5 tools): `dft_vasp_incar`, `dft_vasp_poscar`, `dft_vasp_kpoints`, `dft_qe_pw_input`
- **Materials Project Integration** (3 tools): `materials_project_query`, `materials_project_details`, `materials_project_stable`
- **Phonon Analysis** (4 tools): `analysis_phonon_dos`, `analysis_thermodynamic_properties`, `analysis_phonon_band_path`, `analysis_phonon_estimate`
- **Visualization** (5 tools): `viz_time_series`, `viz_histogram`, `viz_scatter`, `viz_phonon_dos`, `viz_thermo_dashboard`
- **Modular server architecture**: `mcp_tools/` package with separate registration modules
- **Pydantic typed models**: `models.py` with validated input/output schemas
- **Standardized error handling**: `exceptions.py` with 8 exception types
- **Reference validation datasets**: `validation/reference_data.py` with CIF, phonon, ML, VASP/QE cases
- **Validation test suite**: 56 reference-validation tests across 4 modules
- **12 scientific figures**: PDF and PNG format for JORS paper
- **CITATION.cff**: Software citation metadata

### Changed
- Refactored `server.py` from 1294 lines to modular registration via `mcp_tools/`
- Total tool count: 29 → 45
- Total test count: 223 → 303

### Fixed
- CIF parser now handles 5-column atom blocks correctly
- Missing As/Br elements added to prediction database
- Server wrapper functions correctly pass JSON-parsed arguments

## [0.2.0] - 2026-07-31

### Added
- DFT/CIF tools (3): `dft_parse_cif`, `dft_generate_cif`, `dft_cif_summary`
- MXene database tools (5): list, query, properties, compare, search
- ML prediction tools (6): band gap, density, melting point, all, element info, features
- Materials Project tools (3): query, details, stable
- Literature search tools (3): arXiv, materials, author
- Integration test suite (23 tests)
- Phase 2 example scripts

### Changed
- Total tools: 12 → 29
- Total tests: 51 → 223

## [0.1.0] - 2026-07-30

### Added
- LAMMPS input generation (2 tools)
- LAMMPS output parsing (3 tools)
- Nematic alignment analysis (3 tools)
- Non-affine displacement analysis (2 tools)
- Shear rheology tools (2 tools)
- 51 unit tests
- Initial MCP server implementation
