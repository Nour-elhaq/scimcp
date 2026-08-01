<div align="center">

# SciMCP

### Scientific Computing MCP Server for Computational Materials Science

Give your AI coding agent superpowers for molecular dynamics, DFT, crystallography, MXene discovery, ML property prediction, and literature search.

[![PyPI](https://img.shields.io/pypi/v/scimcp?color=0076D6&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/scimcp/)
[![Python](https://img.shields.io/pypi/pyversions/scimcp?color=0076D6&logo=python&logoColor=white)](https://pypi.org/project/scimcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00db00.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-2.0-FF6B35.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-141%20passed-brightgreen)](#testing)

**Works with** [Claude Desktop](https://claude.ai/download) · [Claude Code](https://docs.anthropic.com/en/docs/claude-code) · [Cursor](https://cursor.sh) · [Windsurf](https://codeium.com/windsurf) · Any MCP client

---

```bash
uvx scimcp    # Run instantly, no install needed
```

</div>

---

## What is SciMCP?

SciMCP is an [MCP server](https://modelcontextprotocol.io) that connects AI assistants to computational materials science workflows. It provides **29 tools** across 6 categories: LAMMPS molecular dynamics, DFT/CIF analysis, MXene materials discovery, ML property prediction, and arXiv literature search.

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

> What MXenes are metallic with high conductivity?

> Predict the band gap of GaAs

> Find recent arXiv papers on MXene battery applications

---

## 29 Tools — 6 Categories

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
│   ├── server.py                  # MCP server entry point (29 tools, 7 prompts)
│   └── tools/
│       ├── lammps/
│       │   ├── generator.py       # LAMMPS input generation + workflows
│       │   ├── parser.py          # Dump & thermo file parsing
│       │   ├── nematic.py         # Q-tensor, S(t), S(z) computation
│       │   ├── nonaffine.py       # D²min Falk-Langer metric
│       │   ├── shear.py           # Shear-rate sweep + viscosity fitting
│       │   └── templates/         # Reference LAMMPS templates
│       ├── dft/
│       │   └── cif.py             # CIF parsing, generation, ASE conversion
│       └── materials/
│           ├── mxene.py           # MXene database (10+ entries)
│           ├── prediction.py      # ML property prediction (band gap, density, Tm)
│           └── literature.py      # arXiv literature search
├── tests/                         # 141 tests (unit + integration)
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

# Run Phase 2 tests only
pytest tests/test_cif.py tests/test_mxene.py tests/test_prediction.py -v

# Run integration tests (excludes network-dependent literature tests)
pytest tests/test_phase2_integration.py -v -k "not literature"
```

**141 tests** covering all tools: LAMMPS generation/parsing/analysis, CIF handling, MXene queries, ML prediction, and MCP server integration.

---

## Roadmap

- [x] DFT/CIF file parsing and analysis
- [x] MXene database queries
- [x] ML-accelerated property prediction
- [x] ASE (Atomic Simulation Environment) integration
- [x] arXiv literature search
- [ ] Materials Project API integration
- [ ] Trajectory visualization tools
- [ ] VASP/QE input generation
- [ ] Phonon band structure tools

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
