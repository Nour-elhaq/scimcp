<div align="center">

# SciMCP

### Scientific Computing MCP Server for Computational Materials Science

Give your AI coding agent superpowers for molecular dynamics, DFT, crystallography, MXene discovery, ML property prediction, and literature search.

[![PyPI](https://img.shields.io/pypi/v/scimcp?color=0076D6&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/scimcp/)
[![Python](https://img.shields.io/pypi/pyversions/scimcp?color=0076D6&logo=python&logoColor=white)](https://pypi.org/project/scimcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00db00.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-2.0-FF6B35.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-211%20passed-brightgreen)](#testing)

**Works with** [Claude Desktop](https://claude.ai/download) · [Claude Code](https://docs.anthropic.com/en/docs/claude-code) · [Cursor](https://cursor.sh) · [Windsurf](https://codeium.com/windsurf) · Any MCP client

---

```bash
uvx scimcp    # Run instantly, no install needed
```

</div>

---

## What is SciMCP?

SciMCP is an [MCP server](https://modelcontextprotocol.io) that connects AI assistants to computational materials science workflows. It provides **45 tools** across 9 categories: LAMMPS molecular dynamics, DFT/CIF analysis, VASP/QE input generation, MXene materials discovery, ML property prediction, Materials Project queries, phonon analysis, trajectory visualization, and arXiv literature search.

**No more copy-pasting LAMMPS scripts.** Just tell Claude what you need.

---

## Quick Start

### 1. Install

```bash
# No install needed (recommended)
uvx scimcp

# Or install with pip
pip install scimcp

# Or with uv
uv add scimcp
```

### 2. Connect to your AI client

<details>
<summary><b>Claude Desktop</b></summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "scimcp": {
      "command": "uvx",
      "args": ["scimcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Claude Code</b></summary>

```bash
claude mcp add scimcp -- uvx scimcp
```

</details>

<details>
<summary><b>Cursor / Windsurf</b></summary>

Add to your MCP config:

```json
{
  "mcpServers": {
    "scimcp": {
      "command": "uvx",
      "args": ["scimcp"]
    }
  }
}
```

</details>

### 3. Start asking

> Generate a LAMMPS input for 10^3 LJ atoms under NVT at 300K with shear

> Parse this dump file and show me the temperature evolution

> Compute the nematic order for these quaternion data

> Run a shear-rate sweep from 0.0001 to 0.05

> Parse this CIF file and tell me the space group and lattice parameters

> Generate a VASP INCAR for a relaxation calculation

> What MXenes are metallic with high conductivity?

> Predict the band gap of GaAs

> Find recent arXiv papers on MXene battery applications

> Search Materials Project for stable oxides

> Compute phonon DOS from these frequencies

> Plot the thermo dashboard for this simulation

---

## 45 Tools — 9 Categories

### LAMMPS Input Generation (2 tools)

| Tool | What it does |
|------|-------------|
| `lammps_generate_input` | Generate complete LAMMPS scripts. Supports LJ, EAM, Tersoff, Buckingham, Coulomb potentials. NVT/NPT/NVE/NPH ensembles. Optional shear deformation. |
| `lammps_generate_workflow` | Multi-phase workflow: minimize → equilibrate → production. Configurable per-phase ensemble, temperature, and shear. |

### Output Parsing (3 tools)

| Tool | What it does |
|------|-------------|
| `lammps_parse_thermo` | Parse thermo logs into structured data with statistics (min/max/mean/std per column). |
| `lammps_parse_dump` | Parse lammpstrj trajectories. Extract atom positions, velocities, box dimensions per frame. |
| `lammps_file_summary` | Auto-detect file type (dump vs thermo) and return key metrics. |

### Nematic Alignment Analysis (3 tools)

| Tool | What it does |
|------|-------------|
| `lammps_nematic_order` | Scalar order parameter S from quaternion data. S=1 (aligned), S=0 (isotropic). Returns full Q-tensor. |
| `lammps_nematic_vs_z` | Spatial profile S(z) — alignment as a function of position along z-axis. |
| `lammps_nematic_vs_time` | Time evolution S(t) — alignment over a trajectory. |

### Non-Affine Displacement / Plasticity (2 tools)

| Tool | What it does |
|------|-------------|
| `lammps_d2min` | Falk-Langer D²min metric. Measures how much particle motion deviates from affine deformation of neighbors. |
| `lammps_identify_plastic_events` | Threshold-based detection of particles that underwent plastic rearrangement. |

### Shear Rheology (2 tools)

| Tool | What it does |
|------|-------------|
| `lammps_shear_sweep` | Generate LAMMPS scripts for multiple shear rates. Outputs per-rate input files + config JSON. |
| `lammps_estimate_viscosity` | Fit power-law model to stress vs rate data. Returns consistency K, flow index n, shear-thinning/thickening classification. |

### DFT / CIF Analysis (3 tools)

| Tool | What it does |
|------|-------------|
| `dft_parse_cif` | Parse CIF content and extract space group, lattice parameters, atomic positions, and symmetry data. |
| `dft_generate_cif` | Generate CIF files from crystallographic parameters (space group, lattice, atom types, fractional coordinates). |
| `dft_cif_summary` | Quick summary of a CIF file: formula, space group, lattice, element composition. |

### VASP / Quantum ESPRESSO (5 tools)

| Tool | What it does |
|------|-------------|
| `dft_vasp_incar` | Generate VASP INCAR files with configurable encut, k-spacing, smearing, spin polarization, and relaxation settings. |
| `dft_vasp_poscar` | Generate VASP POSCAR files from element lists, fractional coordinates, and lattice parameters. Supports selective dynamics. |
| `dft_vasp_kpoints` | Generate VASP KPOINTS files with Monkhorst-Pack grids and custom shifts. |
| `dft_qe_pw_input` | Generate Quantum ESPRESSO pw.x input files for SCF, relaxation, and variable-cell calculations. |

### MXene Database (5 tools)

| Tool | What it does |
|------|-------------|
| `materials_mxene_list` | List all 10+ MXenes in the built-in database with key properties. |
| `materials_mxene_query` | Query by formula, M element, X element, termination, or metallic character. |
| `materials_mxene_properties` | Get detailed properties for a specific MXene (lattice, band gap, formation energy, applications). |
| `materials_mxene_compare` | Side-by-side comparison of multiple MXenes. |
| `materials_mxene_search` | Search MXenes within a property range (band gap, formation energy, elastic modulus). |

### ML Property Prediction (6 tools)

| Tool | What it does |
|------|-------------|
| `materials_predict_band_gap` | Predict band gap from composition. Classifies as metal/semiconductor/insulator. |
| `materials_predict_density` | Estimate material density from composition. |
| `materials_predict_melting_point` | Estimate melting point from composition. |
| `materials_predict_all` | Run all prediction models at once. |
| `materials_element_info` | Get elemental properties (electronegativity, radius, mass, density, melting point). |
| `materials_composition_features` | Compute ML feature vector (weighted avg properties, EN difference, mixing entropy). |

### Materials Project (3 tools)

| Tool | What it does |
|------|-------------|
| `materials_project_query` | Query Materials Project for crystal structures and properties. Uses built-in data when no API key. |
| `materials_project_details` | Get detailed properties for a specific MP material (mp-id). |
| `materials_project_stable` | Search for thermodynamically stable materials near the hull. |

### Phonon Analysis (4 tools)

| Tool | What it does |
|------|-------------|
| `analysis_phonon_dos` | Compute phonon density of states from frequency list with Gaussian broadening. |
| `analysis_thermodynamic_properties` | Compute ZPE, Helmholtz free energy, entropy, and heat capacity from phonon frequencies. |
| `analysis_phonon_band_path` | Generate high-symmetry k-path for phonon band structures (cubic, hexagonal, tetragonal, orthorhombic). |
| `analysis_phonon_estimate` | Estimate phonon frequency range from composition using mass-force constant model. |

### Visualization (5 tools)

| Tool | What it does |
|------|-------------|
| `viz_time_series` | Plot time series data from LAMMPS thermo output. |
| `viz_histogram` | Plot histograms with statistics (mean, std, median). |
| `viz_scatter` | Scatter plots, optionally colored by a third variable (e.g., D²min). |
| `viz_phonon_dos` | Plot phonon density of states with zero-frequency line. |
| `viz_thermo_dashboard` | Generate 4-panel dashboard (Temperature, Energy, Pressure vs Step). |

### Literature Search (3 tools)

| Tool | What it does |
|------|-------------|
| `literature_search_arxiv` | Search arXiv for scientific papers by keyword. |
| `literature_search_materials` | Materials-science-optimized arXiv search with category filtering. |
| `literature_search_by_author` | Find papers by a specific author on arXiv. |

---

## Example: Full Workflow

```python
from scimcp.tools.lammps.generator import generate_workflow
from scimcp.tools.lammps.shear import write_sweep_scripts, SweepConfig

# Generate a 3-phase simulation with shear
script = generate_workflow(
    atoms_per_side=12,
    potential="lj",
    temperature=300,
    n_production_steps=100000,
    apply_shear=True,
    shear_rate=0.001,
)

# Run a shear-rate sweep
config = SweepConfig(
    shear_rates=[0.0001, 0.001, 0.01, 0.05],
    atoms_per_side=10,
    temperature=300,
)
files = write_sweep_scripts(config)
# Creates: in.shear_0p000100, in.shear_0p001000, in.shear_0p010000, in.shear_0p050000
```

### Sample Output

```
STEP 5: Nematic Order Parameter S
  Aligned particles:  S = 1.0000
  Random particles:   S = 0.3880
  Partial alignment:  S = 0.4049

  Nematic Profile S(z):
      z_center       S    count
          0.50  1.0000       10   ← aligned
          5.50  0.4483       10   ← transition
          9.50  0.5590       10   ← random

STEP 8: Viscosity Estimation
  Power-law n = 0.80  →  Shear thinning confirmed
  rate=0.0001  →  η=3.15
  rate=0.0100  →  η=1.26
  rate=0.0500  →  η=0.91
```

---

## Supported Potentials

| Potential | LAMMPS Style | Typical Use |
|-----------|-------------|-------------|
| Lennard-Jones | `lj/cut` | Noble gases, generic liquids |
| EAM | `eam/alloy` | Metals (Cu, Al, Ni, Fe, ...) |
| Tersoff | `tersoff` | Semiconductors (Si, C, Ge) |
| Buckingham | `buck/coul/long` | Ionic materials |
| Coulomb | `coul/long` | Charged systems |

---

## Architecture

```
scimcp/
├── src/scimcp/
│   ├── server.py                  # MCP server entry point (45 tools, 7 prompts)
│   └── tools/
│       ├── lammps/
│       │   ├── generator.py       # LAMMPS input generation + workflows
│       │   ├── parser.py          # Dump & thermo file parsing
│       │   ├── nematic.py         # Q-tensor, S(t), S(z) computation
│       │   ├── nonaffine.py       # D²min Falk-Langer metric
│       │   ├── shear.py           # Shear-rate sweep + viscosity fitting
│       │   └── templates/         # Reference LAMMPS templates
│       ├── dft/
│       │   ├── cif.py             # CIF parsing, generation, ASE conversion
│       │   └── vasp_qe.py         # VASP INCAR/POSCAR/KPOINTS + QE pw.x input
│       ├── materials/
│       │   ├── mxene.py           # MXene database (10+ entries)
│       │   ├── prediction.py      # ML property prediction (band gap, density, Tm)
│       │   ├── literature.py      # arXiv literature search
│       │   └── materials_project.py  # Materials Project API + built-in data
│       └── analysis/
│           ├── phonon.py          # Phonon DOS, thermodynamics, band paths
│           └── visualization.py   # Time series, histograms, scatter, dashboards
├── tests/                         # 211 tests (unit + integration)
├── examples/                      # Demo scripts
├── pyproject.toml
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/test_nematic.py -v

# Run Phase 3 tests only
pytest tests/test_vasp_qe.py tests/test_phonon.py tests/test_visualization.py tests/test_materials_project.py -v

# Run integration tests (excludes network-dependent literature tests)
pytest tests/test_phase2_integration.py -v -k "not literature"
```

**211 tests** covering all tools: LAMMPS generation/parsing/analysis, CIF handling, VASP/QE input, MXene queries, ML prediction, Materials Project, phonon analysis, visualization, and MCP server integration.

---

## Roadmap

- [x] DFT/CIF file parsing and analysis
- [x] MXene database queries
- [x] ML-accelerated property prediction
- [x] ASE (Atomic Simulation Environment) integration
- [x] arXiv literature search
- [x] Materials Project API integration
- [x] Trajectory visualization tools
- [x] VASP/QE input generation
- [x] Phonon band structure tools

---

## Contributing

Contributions welcome! Whether it's bug fixes, new tools, or documentation.

```bash
git clone https://github.com/Nour-elhaq/scimcp.git
cd scimcp
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the computational materials science community**

[Report Bug](https://github.com/Nour-elhaq/scimcp/issues) · [Request Feature](https://github.com/Nour-elhaq/scimcp/issues)

</div>
