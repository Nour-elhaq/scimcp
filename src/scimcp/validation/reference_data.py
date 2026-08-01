"""Reference validation datasets for SciMCP scientific tools.

Each module provides known-correct values for regression testing.
"""

from __future__ import annotations

# ── LAMMPS Validation Cases ────────────────────────────────────────────────

LAMMPS_VALIDATION = {
    "lennard_jones_crystal": {
        "description": "FCC Lennard-Jones crystal at T*=0.5, rho*=0.8442",
        "parameters": {
            "atoms_per_side": 5,
            "lattice_spacing": 5.26,
            "potential": "lj",
            "epsilon": 1.0,
            "sigma": 1.0,
            "cutoff": 2.5,
        },
        "expected_properties": {
            "n_atoms_approx": 500,
            "expected_potential_energy_per_atom": -5.5,  # eV, approximate for LJ FCC
            "expected_rdf_peak_1": 1.12,  # sigma * 2^(1/6)
        },
    },
    "affine_deformation_d2min_zero": {
        "description": "Pure affine deformation should give D²min ≈ 0",
        "n_particles": 100,
        "deformation_type": "affine",
        "expected_d2min_max": 1e-10,
    },
    "shear_newtonian": {
        "description": "Low shear rate → Newtonian regime",
        "shear_rates": [0.0001, 0.0002, 0.0005],
        "expected_flow_index_range": [0.9, 1.1],
    },
}


# ── CIF Validation Cases ──────────────────────────────────────────────────

CIF_VALIDATION = {
    "silicon": {
        "description": "Silicon diamond cubic (Fd-3m, #227)",
        "cif_content": """data_Si
_symmetry_space_group_name_H-M   'F d -3 m'
_cell_length_a    5.431
_cell_length_b    5.431
_cell_length_c    5.431
_cell_angle_alpha 90.000
_cell_angle_beta  90.000
_cell_angle_gamma 90.000
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.00000 0.00000 0.00000
Si2 Si 0.25000 0.25000 0.25000
""",
        "expected": {
            "n_atoms": 2,
            "elements": ["Si"],
            "lattice_a": 5.431,
            "volume_approx": 160.2,  # Angstrom³
        },
    },
    "nacl": {
        "description": "Sodium chloride (Fm-3m, #225)",
        "cif_content": """data_NaCl
_symmetry_space_group_name_H-M   'F m -3 m'
_cell_length_a    5.640
_cell_length_b    5.640
_cell_length_c    5.640
_cell_angle_alpha 90.000
_cell_angle_beta  90.000
_cell_angle_gamma 90.000
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.00000 0.00000 0.00000
Cl1 Cl 0.50000 0.50000 0.50000
""",
        "expected": {
            "n_atoms": 2,
            "elements": ["Na", "Cl"],
            "lattice_a": 5.640,
        },
    },
    "gaas": {
        "description": "Gallium arsenide zincblende (F-43m, #216)",
        "cif_content": """data_GaAs
_symmetry_space_group_name_H-M   'F -4 3 m'
_cell_length_a    5.653
_cell_length_b    5.653
_cell_length_c    5.653
_cell_angle_alpha 90.000
_cell_angle_beta  90.000
_cell_angle_gamma 90.000
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ga1 Ga 0.00000 0.00000 0.00000
As1 As 0.25000 0.25000 0.25000
""",
        "expected": {
            "n_atoms": 2,
            "elements": ["Ga", "As"],
            "lattice_a": 5.653,
        },
    },
}


# ── Phonon Validation Cases ───────────────────────────────────────────────

PHONON_VALIDATION = {
    "simple_monatomic": {
        "description": "Monatomic chain with 3 frequencies for ZPE/entropy validation",
        "frequencies_THz": [3.0, 6.0, 9.0],
        "n_atoms": 1,
        "temperature_K": 300.0,
        "expected_ZPE_eV_approx": 0.0124,  # Sum(h*nu/2) for these frequencies
    },
    "dual_atom_cell": {
        "description": "Diatomic cell with 6 modes (3 acoustic + 3 optical)",
        "frequencies_THz": [2.0, 2.5, 3.0, 8.0, 9.0, 10.0],
        "n_atoms": 2,
        "temperature_K": 300.0,
    },
    "normalization_check": {
        "description": "DOS should integrate to total number of modes",
        "frequencies_THz": [1.0, 2.0, 3.0, 4.0, 5.0],
        "sigma_THz": 0.3,
        "n_points": 500,
        "expected_integral": 5.0,  # Number of frequencies
    },
}


# ── ML Prediction Validation Cases ────────────────────────────────────────

ML_VALIDATION = {
    "band_gap_metals": {
        "description": "Known metals should have band_gap ≈ 0",
        "test_cases": [
            {"composition": "Cu", "expected_band_gap_approx": 0.0},
            {"composition": "Fe", "expected_band_gap_approx": 0.0},
            {"composition": "Al", "expected_band_gap_approx": 0.0},
        ],
    },
    "band_gap_semiconductors": {
        "description": "Known semiconductor band gaps",
        "test_cases": [
            {"composition": "Si", "expected_band_gap_approx": 1.1},
            {"composition": "GaAs", "expected_band_gap_approx": 1.4},
            {"composition": "SiC", "expected_band_gap_approx": 2.4},
        ],
    },
    "density_known": {
        "description": "Known material densities",
        "test_cases": [
            {"composition": "Si", "expected_density_approx": 2.33},
            {"composition": "Fe", "expected_density_approx": 7.87},
            {"composition": "Cu", "expected_density_approx": 8.96},
        ],
    },
    "feature_vector": {
        "description": "Feature vector for Si",
        "composition": "Si",
        "expected_features": {
            "n_elements": 1,
            "has_metal": False,
        },
    },
}


# ── VASP/QE Validation ───────────────────────────────────────────────────

VASP_VALIDATION = {
    "static_calculation": {
        "description": "Standard static VASP INCAR",
        "parameters": {
            "encut": 520, "ismear": -5, "sigma": 0.05,
            "nsw": 0, "ibrion": -1, "ispin": 1,
        },
        "expected_tags": ["ENCUT = 520", "ISMEAR = -5", "NSW = 0"],
    },
    "relaxation_calculation": {
        "description": "Geometry relaxation INCAR",
        "parameters": {
            "encut": 400, "ismear": 0, "sigma": 0.1,
            "nsw": 100, "ibrion": 2, "ediffg": -0.01,
        },
        "expected_tags": ["NSW = 100", "IBRION = 2", "EDIFFG = -0.01"],
    },
}

QE_VALIDATION = {
    "scf_calculation": {
        "description": "Standard QE SCF input",
        "parameters": {
            "calculation": "scf", "ecutwfc": 30.0,
            "k_points": [4, 4, 4],
        },
        "expected_tags": ["calculation = 'scf'", "ecutwfc = 30.0"],
    },
}
