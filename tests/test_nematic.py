"""Tests for nematic alignment computation."""

import numpy as np
import pytest

from scimcp.tools.lammps.nematic import (
    compute_nematic_order,
    compute_nematic_alignment_vs_time,
    compute_nematic_alignment_vs_z,
    compute_q_tensor_components,
)


class TestNematicOrder:
    def test_perfect_alignment(self):
        """All quaternions pointing same direction → S ≈ 1."""
        quats = np.array([[1, 0, 0, 0]] * 50)
        S = compute_nematic_order(quats)
        assert S > 0.9

    def test_isotropic(self):
        """Random quaternions -> S close to 0."""
        rng = np.random.default_rng(42)
        quats = rng.random((500, 4))
        S = compute_nematic_order(quats)
        assert abs(S) < 0.4

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            compute_nematic_order(np.array([1, 2, 3]))

    def test_invalid_columns(self):
        with pytest.raises(ValueError):
            compute_nematic_order(np.array([[1, 2, 3]] * 10))


class TestNematicVsTime:
    def test_returns_arrays(self):
        frames = [np.array([[1, 0, 0, 0]] * 10) for _ in range(5)]
        result = compute_nematic_alignment_vs_time(frames)
        assert len(result["t"]) == 5
        assert len(result["S"]) == 5

    def test_constant_alignment(self):
        frames = [np.array([[1, 0, 0, 0]] * 10) for _ in range(5)]
        result = compute_nematic_alignment_vs_time(frames)
        assert result["std_S"] < 0.01


class TestNematicVsZ:
    def test_returns_bins(self):
        quats = np.array([[1, 0, 0, 0]] * 100)
        z = np.linspace(0, 10, 100)
        result = compute_nematic_alignment_vs_z(quats, z, n_bins=5)
        assert len(result["z_centers"]) == 5
        assert len(result["S"]) == 5

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_nematic_alignment_vs_z(
                np.array([[1, 0, 0, 0]] * 10),
                np.array([0.0] * 5),
            )


class TestQTensorComponents:
    def test_returns_six_components(self):
        quats = np.array([[1, 0, 0, 0]] * 10)
        result = compute_q_tensor_components(quats)
        assert set(result.keys()) == {"Qxx", "Qyy", "Qzz", "Qxy", "Qxz", "Qyz"}
