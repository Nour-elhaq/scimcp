"""Tests for VASP and Quantum ESPRESSO input generation."""

import json

import pytest

from scimcp.tools.dft.vasp_qe import (
    generate_vasp_incar,
    generate_vasp_poscar,
    generate_vasp_kpoints,
    generate_qe_pw_input,
)


class TestVaspIncar:
    def test_default_incar(self):
        result = generate_vasp_incar()
        assert "ENCUT" in result
        assert "EDIFF" in result
        assert "NSW" in result
        assert "IBRION" in result

    def test_static_calculation(self):
        result = generate_vasp_incar(nsw=0)
        assert "NSW    = 0" in result
        assert "IBRION = -1" in result

    def test_relaxation(self):
        result = generate_vasp_incar(nsw=100, ibrion=2)
        assert "NSW    = 100" in result
        assert "IBRION = 2" in result

    def test_spin_polarized(self):
        result = generate_vasp_incar(ispin=2)
        assert "ISPIN  = 2" in result
        assert "LORBIT" in result

    def test_custom_encut(self):
        result = generate_vasp_incar(encut=600)
        assert "ENCUT  = 600" in result

    def test_output_file(self, tmp_path):
        out = tmp_path / "INCAR"
        result = generate_vasp_incar(output_file=str(out))
        assert out.exists()
        assert "ENCUT" in out.read_text()

    def test_lwave_false(self):
        result = generate_vasp_incar(lwave=False)
        assert "LWAVE  = .FALSE." in result

    def test_tebreak(self):
        result = generate_vasp_incar(tebreak=0.01)
        assert "TEBREAK" in result


class TestVaspPoscar:
    def test_simple_poscar(self):
        result = generate_vasp_poscar(
            elements=["Si", "Si"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        assert "Direct" in result
        assert "Si" in result

    def test_with_lattice(self):
        result = generate_vasp_poscar(
            elements=["Fe"],
            positions=[[0, 0, 0]],
            lattice_params={"a": 2.87, "b": 2.87, "c": 2.87, "alpha": 90, "beta": 90, "gamma": 90},
        )
        assert "Direct" in result
        assert "2.87" in result

    def test_selective_dynamics(self):
        result = generate_vasp_poscar(
            elements=["Si", "Si"],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
            selective_dynamics=True,
            selective_mask=[[True, True, True], [False, False, False]],
        )
        assert "Selective dynamics" in result
        assert "T" in result
        assert "F" in result

    def test_output_file(self, tmp_path):
        out = tmp_path / "POSCAR"
        result = generate_vasp_poscar(
            elements=["Si"],
            positions=[[0, 0, 0]],
            output_file=str(out),
        )
        assert out.exists()

    def test_comment(self):
        result = generate_vasp_poscar(
            elements=["Si"],
            positions=[[0, 0, 0]],
            comment="My crystal",
        )
        assert result.startswith("My crystal")


class TestVaspKpoints:
    def test_default_kpoints(self):
        result = generate_vasp_kpoints()
        assert "Monkhorst-Pack" in result
        assert "8  8  8" in result

    def test_custom_kpoints(self):
        result = generate_vasp_kpoints(kx=12, ky=12, kz=12)
        assert "12  12  12" in result

    def test_with_shift(self):
        result = generate_vasp_kpoints(shift=[0.5, 0.5, 0.5])
        assert "0.5  0.5  0.5" in result

    def test_output_file(self, tmp_path):
        out = tmp_path / "KPOINTS"
        result = generate_vasp_kpoints(output_file=str(out))
        assert out.exists()


class TestQePwInput:
    def test_default_scf(self):
        result = generate_qe_pw_input()
        assert "calculation = 'scf'" in result
        assert "ATOMIC_SPECIES" in result
        assert "K_POINTS automatic" in result

    def test_relax(self):
        result = generate_qe_pw_input(calculation="relax")
        assert "calculation = 'relax'" in result
        assert "&IONS" in result

    def test_vc_relax(self):
        result = generate_qe_pw_input(calculation="vc-relax")
        assert "calculation = 'vc-relax'" in result
        assert "&CELL" in result

    def test_custom_ecut(self):
        result = generate_qe_pw_input(ecutwfc=50.0)
        assert "ecutwfc = 50.0" in result

    def test_with_positions(self):
        result = generate_qe_pw_input(
            atom_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        assert "nat = 2" in result

    def test_output_file(self, tmp_path):
        out = tmp_path / "pw.in"
        result = generate_qe_pw_input(output_file=str(out))
        assert out.exists()

    def test_k_points(self):
        result = generate_qe_pw_input(k_points=[6, 6, 6])
        assert "6  6  6" in result
