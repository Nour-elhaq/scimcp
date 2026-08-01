"""End-to-end test of all Phase 2 tools with real examples."""

import json
import sys

sys.path.insert(0, "src")

from scimcp.server import mcp

tools = mcp._tool_manager._tools


def run(name, args):
    result = tools[name].fn(**args)
    return json.loads(result)


# ═══════════════════════════════════════════════════════════════
# 1. DFT / CIF
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. DFT / CIF ANALYSIS")
print("=" * 60)

SIO2_CIF = """\
data_quartz
_symmetry_space_group_name_H-M   P 31 2 1

_cell_length_a       4.913000
_cell_length_b       4.913000
_cell_length_c       5.405000
_cell_angle_alpha    90.000000
_cell_angle_beta     90.000000
_cell_angle_gamma    120.000000

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
  Si1  Si  0.469700  0.000000  0.000000
  O1   O   0.413500  0.266900  0.119100
"""

r = run("dft_parse_cif", {"cif_content": SIO2_CIF})
print(f"\n[Parsed SiO2 CIF]")
print(f"  Atoms: {r['n_atoms']}")
print(f"  Lattice: a={r['lattice']['a']:.3f}, c={r['lattice']['c']:.3f}")
print(f"  Space group: {r['space_group']}")

r = run("dft_cif_summary", {"cif_content": SIO2_CIF})
print(f"\n[CIF Summary]")
print(f"  Formula: {r['formula']}")
print(f"  Elements: {r['elements']}")
print(f"  Composition: {r['composition']}")

r = run("dft_generate_cif", {
    "space_group": 225,
    "lattice_params": json.dumps({"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90}),
    "atom_types": json.dumps(["Si", "Si"]),
    "positions": json.dumps([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    "formula": "Si2",
})
print(f"\n[Generated CIF]")
print(f"  Length: {len(r['cif_content'])} chars")
print(f"  Contains 'Si': {'Si' in r['cif_content']}")


# ═══════════════════════════════════════════════════════════════
# 2. MXene Database
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. MXene DATABASE")
print("=" * 60)

r = run("materials_mxene_list", {})
print(f"\n[MXene List] ({len(r)} entries)")
for mx in r[:5]:
    print(f"  - {mx}")
print(f"  ... and {len(r) - 5} more")

r = run("materials_mxene_query", {"m_element": "Ti", "metallic_only": True})
print(f"\n[Ti-based metallic MXenes] ({len(r)} results)")
for mx in r:
    print(f"  - {mx['name']}: band_gap={mx.get('band_gap_eV', '?')} eV, formation={mx.get('formation_energy_eV_per_atom', '?')} eV/atom")

r = run("materials_mxene_properties", {"formula": "Ti3C2"})
print(f"\n[Ti3C2 Properties]")
print(f"  Formula: {r['formula']}")
print(f"  Lattice: a={r['lattice']['a']}, c={r['lattice']['c']}")
print(f"  Metallic: {r['is_metallic']}")
print(f"  Band gap: {r['band_gap_eV']} eV")
print(f"  Applications: {r.get(' Applications', r.get('Applications', []))}")

r = run("materials_mxene_compare", {"formulas": json.dumps(["Ti3C2", "V2C", "Nb2C"])})
print(f"\n[MXene Comparison]")
for name, props in r.items():
    if "error" not in props:
        print(f"  {name}: M={props['M']}, gap={props.get('band_gap_eV', '?')} eV, E_form={props.get('formation_energy_eV_per_atom', '?')} eV/atom")

r = run("materials_mxene_search", {"property_name": "band_gap_eV", "min_value": 0.0, "max_value": 0.0})
print(f"\n[Zero band-gap MXenes] ({len(r)} results)")
for mx in r[:3]:
    print(f"  - {mx['name']}")


# ═══════════════════════════════════════════════════════════════
# 3. ML Property Prediction
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ML PROPERTY PREDICTION")
print("=" * 60)

for formula in ["Si", "GaAs", "TiO2", "Cu", "NaCl"]:
    r = run("materials_predict_all", {"composition": formula})
    bg = r["band_gap"]
    den = r["density"]
    mp = r["melting_point"]
    print(f"\n[{formula}]")
    print(f"  Band gap:   {bg['predicted_band_gap_eV']:.3f} eV  ({bg['classification']})")
    print(f"  Density:    {den['estimated_density_g_cm3']:.3f} g/cm³")
    print(f"  Melting pt: {mp['estimated_melting_point_K']:.1f} K")

r = run("materials_element_info", {"element": "Ti"})
print(f"\n[Ti Element Info]")
print(f"  Z={r['Z']}, mass={r['mass']}, EN={r['en']}, radius={r['radius']}, density={r['density']} g/cm³, Tm={r['Tm']} K")

r = run("materials_composition_features", {"composition": "GaAs"})
print(f"\n[GaAs ML Features]")
for k, v in r["features"].items():
    print(f"  {k}: {v:.4f}")


# ═══════════════════════════════════════════════════════════════
# 4. Literature Search (network-dependent)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. LITERATURE SEARCH")
print("=" * 60)

try:
    r = run("literature_search_arxiv", {"query": "MXene DFT electronic structure", "max_results": 3})
    if r.get("papers"):
        print(f"\n[arXiv Search: 'MXene DFT electronic structure'] ({r['total_results']} results)")
        for p in r["papers"][:3]:
            print(f"  - {p['title'][:70]}...")
            print(f"    Authors: {', '.join(p['authors'][:2])}")
            print(f"    Published: {p['published'][:10]}")
    else:
        print(f"\n[arXiv Search] No results or network unavailable: {r.get('error', '')}")
except Exception as e:
    print(f"\n[arXiv Search] Network error (expected in offline env): {e}")


print("\n" + "=" * 60)
print("ALL PHASE 2 EXAMPLES COMPLETE")
print("=" * 60)
