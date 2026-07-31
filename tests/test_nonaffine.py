"""Tests for non-affine displacement computation."""

import numpy as np
import pytest

from scimcp.tools.lammps.nonaffine import (
    compute_nonaffine_displacement,
    identify_plastic_events,
)


class TestD2Min:
    def test_affine_deformation(self):
        """Affine deformation should give low D²min."""
        N = 20
        rng = np.random.default_rng(42)
        pos0 = rng.random((N, 3)) * 10

        # Apply uniform affine deformation (translation)
        pos1 = pos0 + np.array([0.1, 0.0, 0.0])

        d2min = compute_nonaffine_displacement(pos0, pos1, r_cut=5.0)
        # Pure translation should give near-zero D²min
        assert d2min.max() < 0.01

    def test_plastic_event(self):
        """Neighbors moving non-affinely should produce high D²min."""
        # Create a small cluster where neighbors move in different directions
        pos0 = np.array([
            [0, 0, 0],    # particle 0 (center)
            [1, 0, 0],    # particle 1 (neighbor)
            [-1, 0, 0],   # particle 2 (neighbor)
            [0, 1, 0],    # particle 3 (neighbor)
            [0, -1, 0],   # particle 4 (neighbor)
            [0, 0, 1],    # particle 5 (neighbor)
            [0, 0, -1],   # particle 6 (neighbor)
            [2, 0, 0],    # particle 7 (far)
            [-2, 0, 0],   # particle 8 (far)
        ], dtype=float)

        pos1 = pos0.copy()
        # Move neighbors in different directions (non-affine: not a uniform deformation)
        pos1[1] += np.array([1.0, 0.0, 0.0])   # neighbor moves right
        pos1[2] += np.array([-1.0, 0.0, 0.0])  # neighbor moves left
        pos1[3] += np.array([0.0, 2.0, 0.0])   # neighbor moves up more

        d2min = compute_nonaffine_displacement(pos0, pos1, r_cut=3.0)
        # At least some particles should have non-zero D²min
        assert d2min.max() > 0.0

    def test_output_shape(self):
        N = 10
        pos = np.random.random((N, 3)) * 10
        d2min = compute_nonaffine_displacement(pos, pos, r_cut=5.0)
        assert d2min.shape == (N,)


class TestPlasticEvents:
    def test_identifies_plastic(self):
        d2min = np.array([0.0, 0.0, 0.5, 0.0, 0.3, 0.0])
        result = identify_plastic_events(d2min, threshold=0.1)
        assert result["n_plastic"] == 2
        assert 2 in result["plastic_particles"]
        assert 4 in result["plastic_particles"]

    def test_no_plastic(self):
        d2min = np.array([0.01, 0.02, 0.03])
        result = identify_plastic_events(d2min, threshold=0.1)
        assert result["n_plastic"] == 0
        assert result["fraction_plastic"] == 0.0

    def test_all_plastic(self):
        d2min = np.array([1.0, 2.0, 3.0])
        result = identify_plastic_events(d2min, threshold=0.1)
        assert result["n_plastic"] == 3
        assert result["fraction_plastic"] == 1.0
