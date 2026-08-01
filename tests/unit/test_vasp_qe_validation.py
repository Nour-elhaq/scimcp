"""Unit tests for VASP/QE input generation validation."""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scimcp.tools.dft.vasp_qe import (
    generate_vasp_incar,
    generate_vasp_poscar,
    generate_vasp_kpoints,
    generate_qe_pw_input,
)
from scimcp.validation.reference_data import VASP_VALIDATION, QE_VALIDATION


class TestVASPIncarStatic:
    """Validate VASP INCAR for static calculations."""

    def test_static_has_encut(self):
        result = generate_vasp_incar(encut=520, ismear=-5, sigma=0.05, nsw=0)
        assert "ENCUT" in result and "520" in result

    def test_static_nsw_zero(self):
        result = generate_vasp_incar(nsw=0)
        assert "NSW" in result and "= 0" in result

    def test_static_ibrion(self):
        result = generate_vasp_incar(ibrion=-1, nsw=0)
        assert "IBRION" in result

    def test_tetrahedron_method(self):
        result = generate_vasp_incar(ismear=-5, sigma=0.05)
        assert "ISMEAR" in result and "= -5" in result

    def test_gaussian_smearing(self):
        result = generate_vasp_incar(ismear=0, sigma=0.1)
        assert "ISMEAR" in result and "= 0" in result

    def test_spin_polarized(self):
        result = generate_vasp_incar(ispin=2)
        assert "ISPIN" in result and "= 2" in result

    def test_output_file_written(self, tmp_path):
        outfile = tmp_path / "INCAR"
        generate_vasp_incar(output_file=str(outfile))
        assert outfile.exists()
        content = outfile.read_text()
        assert "ENCUT" in content


class TestVASPIncarRelaxation:
    """Validate VASP INCAR for relaxation calculations."""

    def test_relaxation_has_nsw(self):
        result = generate_vasp_incar(nsw=100, ibrion=2)
        assert "NSW" in result and "= 100" in result
        assert "IBRION" in result and "= 2" in result

    def test_relaxation_ediffg(self):
        result = generate_vasp_incar(ediffg=-0.01, nsw=100)
        assert "EDIFFG" in result

    def test_relaxation_cg(self):
        result = generate_vasp_incar(ibrion=1, nsw=200)
        assert "IBRION" in result and "= 1" in result


class TestVASPPOSCAR:
    """Validate VASP POSCAR generation."""

    def test_poscar_contains_elements(self):
        result = generate_vasp_poscar(
            elements=["Si", "Si"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        assert "Si" in result

    def test_poscar_contains_positions(self):
        result = generate_vasp_poscar(
            elements=["Si", "Si"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        assert "0.00000" in result or "0.25000" in result

    def test_poscar_selective_dynamics(self):
        result = generate_vasp_poscar(
            elements=["Si"],
            positions=[[0, 0, 0]],
            selective_dynamics=True,
        )
        assert "Selective" in result or "T T T" in result

    def test_poscar_with_lattice(self):
        lattice = {"a": 5.43, "b": 5.43, "c": 5.43, "alpha": 90, "beta": 90, "gamma": 90}
        result = generate_vasp_poscar(
            elements=["Si", "Si"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
            lattice_params=lattice,
        )
        assert "5.43" in result

    def test_poscar_comment(self):
        result = generate_vasp_poscar(
            elements=["Si"], positions=[[0, 0, 0]],
            comment="Test POSCAR",
        )
        assert "Test POSCAR" in result


class TestVASPKPOINTS:
    """Validate VASP KPOINTS generation."""

    def test_kpoints_monkhorst_pack(self):
        result = generate_vasp_kpoints(kx=8, ky=8, kz=8)
        assert "8" in result

    def test_kpoints_gamma(self):
        result = generate_vasp_kpoints(kx=4, ky=4, kz=4, shift=[0.5, 0.5, 0.5])
        assert "4" in result

    def test_kpoints_output_file(self, tmp_path):
        outfile = tmp_path / "KPOINTS"
        generate_vasp_kpoints(output_file=str(outfile))
        assert outfile.exists()


class TestQEPWInput:
    """Validate Quantum ESPRESSO pw.x input generation."""

    def test_scf_calculation(self):
        result = generate_qe_pw_input(calculation="scf")
        assert "scf" in result

    def test_relax_calculation(self):
        result = generate_qe_pw_input(calculation="relax")
        assert "relax" in result

    def test_ecutwfc(self):
        result = generate_qe_pw_input(ecutwfc=30.0)
        assert "30.0" in result

    def test_ecutrho(self):
        result = generate_qe_pw_input(ecutrho=240.0)
        assert "240.0" in result

    def test_k_points(self):
        result = generate_qe_pw_input(k_points=[6, 6, 6])
        assert "6" in result

    def test_conv_thr(self):
        result = generate_qe_pw_input(conv_thr=1e-8)
        assert "conv_thr" in result
        assert "1.0e-08" in result

    def test_output_file(self, tmp_path):
        outfile = tmp_path / "pw.in"
        generate_qe_pw_input(output_file=str(outfile))
        assert outfile.exists()
