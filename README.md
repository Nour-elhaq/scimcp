<![CDATA[<div align="center">

# SciMCP

### Scientific Computing MCP Server for Computational Materials Science

Give your AI coding agent superpowers for molecular dynamics, crystallography, and materials analysis.

[![PyPI](https://img.shields.io/pypi/v/scimcp?color=0076D6&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/scimcp/)
[![Python](https://img.shields.io/pypi/pyversions/scimcp?color=0076D6&logo=python&logoColor=white)](https://pypi.org/project/scimcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00db00.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-2.0-FF6B35.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen)](#testing)

**Works with** [Claude Desktop](https://claude.ai/download) · [Claude Code](https://docs.anthropic.com/en/docs/claude-code) · [Cursor](https://cursor.sh) · [Windsurf](https://codeium.com/windsurf) · Any MCP client

---

```bash
uvx scimcp    # Run instantly, no install needed
```

</div>

---

## What is SciMCP?

SciMCP is an [MCP server](https://modelcontextprotocol.io) that connects AI assistants to computational materials science workflows. It wraps LAMMPS input generation, trajectory parsing, nematic alignment analysis, non-affine displacement computation, and shear-rate sweeps into **12 tools** that any MCP-compatible agent can call.

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

---

## 12 Tools — 5 Categories

### LAMMPS Input Generation

| Tool | What it does |
|------|-------------|
| `lammps_generate_input` | Generate complete LAMMPS scripts. Supports LJ, EAM, Tersoff, Buckingham, Coulomb potentials. NVT/NPT/NVE/NPH ensembles. Optional shear deformation. |
| `lammps_generate_workflow` | Multi-phase workflow: minimize → equilibrate → production. Configurable per-phase ensemble, temperature, and shear. |

### Output Parsing

| Tool | What it does |
|------|-------------|
| `lammps_parse_thermo` | Parse thermo logs into structured data with statistics (min/max/mean/std per column). |
| `lammps_parse_dump` | Parse lammpstrj trajectories. Extract atom positions, velocities, box dimensions per frame. |
| `lammps_file_summary` | Auto-detect file type (dump vs thermo) and return key metrics. |

### Nematic Alignment Analysis

| Tool | What it does |
|------|-------------|
| `lammps_nematic_order` | Scalar order parameter S from quaternion data. S=1 (aligned), S=0 (isotropic). Returns full Q-tensor. |
| `lammps_nematic_vs_z` | Spatial profile S(z) — alignment as a function of position along z-axis. |
| `lammps_nematic_vs_time` | Time evolution S(t) — alignment over a trajectory. |

### Non-Affine Displacement (Plasticity)

| Tool | What it does |
|------|-------------|
| `lammps_d2min` | Falk-Langer D²min metric. Measures how much particle motion deviates from affine deformation of neighbors. |
| `lammps_identify_plastic_events` | Threshold-based detection of particles that underwent plastic rearrangement. |

### Shear Rheology

| Tool | What it does |
|------|-------------|
| `lammps_shear_sweep` | Generate LAMMPS scripts for multiple shear rates. Outputs per-rate input files + config JSON. |
| `lammps_estimate_viscosity` | Fit power-law model to stress vs rate data. Returns consistency K, flow index n, shear-thinning/thickening classification. |

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
│   ├── server.py                  # MCP server entry point (12 tools, 2 prompts)
│   └── tools/
│       └── lammps/
│           ├── generator.py       # LAMMPS input generation + workflows
│           ├── parser.py          # Dump & thermo file parsing
│           ├── nematic.py         # Q-tensor, S(t), S(z) computation
│           ├── nonaffine.py       # D²min Falk-Langer metric
│           ├── shear.py           # Shear-rate sweep + viscosity fitting
│           └── templates/         # Reference LAMMPS templates
├── tests/                         # 51 unit tests
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
```

**51 tests** covering all tools: input generation, output parsing, nematic analysis, non-affine displacement, and shear rheology.

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

## Roadmap

- [ ] DFT/CIF file parsing and analysis
- [ ] Materials Project API integration
- [ ] MXene database queries
- [ ] ML-accelerated property prediction
- [ ] ASE (Atomic Simulation Environment) integration
- [ ] Trajectory visualization tools

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the computational materials science community**

[Report Bug](https://github.com/Nour-elhaq/scimcp/issues) · [Request Feature](https://github.com/Nour-elhaq/scimcp/issues)

</div>
]]>