"""Tests for LAMMPS input generator."""

import pytest

from scimcp.tools.lammps.generator import (
    generate_lammps_input,
    generate_workflow,
)


class TestGenerateLAMMPSInput:
    """Tests for the lammps_generate_input function."""

    def test_default_generates_valid_script(self):
        script = generate_lammps_input()
        assert "units           lj" in script
        assert "atom_style      atomic" in script
        assert "pair_style      lj/cut" in script
        assert "create_box      1 box" in script
        assert "run             100000" in script

    def test_custom_atoms_per_side(self):
        script = generate_lammps_input(atoms_per_side=20)
        assert "region          box block 0 20 0 20 0 20" in script

    def test_npt_ensemble(self):
        script = generate_lammps_input(ensemble="npt", pressure=1.0)
        assert "npt temp" in script
        assert "iso 1.0 1.0" in script

    def test_nve_ensemble(self):
        script = generate_lammps_input(ensemble="nve")
        assert "fix             1 all nve" in script

    def test_shear_enabled(self):
        script = generate_lammps_input(apply_shear=True, shear_rate=0.005)
        assert "deform 1 xy erate 0.005" in script

    def test_shear_xz(self):
        script = generate_lammps_input(apply_shear=True, shear_direction="xz")
        assert "deform 1 xz erate" in script

    def test_no_minimize(self):
        script = generate_lammps_input(minimize=False)
        assert "min_style" not in script

    def test_eam_potential(self):
        script = generate_lammps_input(potential="eam")
        assert "pair_style      eam/alloy" in script

    def test_output_file(self, tmp_path):
        out = tmp_path / "test.in"
        generate_lammps_input(output_file=str(out))
        assert out.exists()
        content = out.read_text()
        assert "units           lj" in content


class TestGenerateWorkflow:
    """Tests for the lammps_generate_workflow function."""

    def test_default_workflow(self):
        script = generate_workflow()
        assert "Phase 1" in script or "PHASE 1" in script
        assert "Phase 2" in script or "PHASE 2" in script
        assert "Phase 3" in script or "PHASE 3" in script

    def test_minimize_phase(self):
        script = generate_workflow(n_minimize_steps=5000)
        assert "minimize" in script.lower()

    def test_equilibrate_steps(self):
        script = generate_workflow(n_equilibrate_steps=5000)
        assert "5000" in script

    def test_production_steps(self):
        script = generate_workflow(n_production_steps=200000)
        assert "200000" in script

    def test_shear_in_production(self):
        script = generate_workflow(apply_shear=True)
        assert "deform" in script

    def test_output_file(self, tmp_path):
        out = tmp_path / "workflow.in"
        generate_workflow(output_file=str(out))
        assert out.exists()
