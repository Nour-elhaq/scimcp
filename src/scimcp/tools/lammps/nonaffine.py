"""Compute non-affine displacement D²min from MD trajectory data.

Implements the Falk-Langer metric for measuring non-affine plastic
deformation in amorphous solids and granular materials under shear.
"""

from __future__ import annotations

import numpy as np


def compute_nonaffine_displacement(
    positions_t0: np.ndarray,
    positions_t1: np.ndarray,
    neighbors_t0: list[list[int]] | None = None,
    box: np.ndarray | None = None,
    r_cut: float = 3.0,
) -> np.ndarray:
    """Compute non-affine displacement D²min for each particle.

    D²min measures how much a particle's displacement deviates from the
    best-fit affine deformation of its neighbors.

    Args:
        positions_t0: Positions at time t0, shape (N, 3).
        positions_t1: Positions at time t1, shape (N, 3).
        neighbors_t0: List of neighbor lists for each particle at t0.
            If None, uses all particles within r_cut.
        box: Simulation box vectors [[xlo,xhi],[ylo,yhi],[zlo,zhi]].
            If None, assumes no periodic boundary.
        r_cut: Cutoff radius for neighbor finding (used if neighbors_t0 is None).

    Returns:
        D²min values for each particle, shape (N,).
    """
    N = len(positions_t0)
    d2min = np.zeros(N)

    # Find neighbors if not provided
    if neighbors_t0 is None:
        neighbors_t0 = _find_neighbors(positions_t0, r_cut, box)

    for i in range(N):
        neighbors = neighbors_t0[i]
        if len(neighbors) < 4:
            d2min[i] = 0.0
            continue

        # Displacement vectors
        delta_r = positions_t1[neighbors] - positions_t0[neighbors]
        delta_R = positions_t0[neighbors] - positions_t0[i]

        # Apply periodic boundary corrections if box is provided
        if box is not None:
            delta_r = _apply_pbc(delta_r, box)
            delta_R = _apply_pbc(delta_R, box)

        # Solve for best-fit affine deformation: delta_r ≈ E · delta_R
        # Using least squares: E = (delta_R^T delta_R)^{-1} delta_R^T delta_r
        try:
            # E is the 3x3 deformation gradient
            # Use pinv for robustness with rank-deficient systems
            A = np.linalg.lstsq(delta_R, delta_r, rcond=None)[0]

            # Non-affine part
            non_affine_disp = delta_r - delta_R @ A

            d2min[i] = np.mean(np.sum(non_affine_disp ** 2, axis=1))
        except (np.linalg.LinAlgError, ValueError):
            d2min[i] = 0.0

    return d2min


def compute_d2min_vs_strain(
    trajectory: np.ndarray,
    strains: np.ndarray,
    box: np.ndarray | None = None,
    r_cut: float = 3.0,
    window: int = 1,
) -> dict[str, np.ndarray]:
    """Compute D²min as a function of strain during shear.

    Args:
        trajectory: Positions at each strain step, shape (n_steps, N, 3).
        strains: Strain values at each step, shape (n_steps,).
        box: Simulation box at each step (or single box for all steps).
        r_cut: Cutoff radius for neighbor finding.
        window: Number of steps back to compare for D²min calculation.

    Returns:
        Dictionary with:
        - 'strains': Strain values (trimmed by window)
        - 'mean_d2min': Mean D²min at each strain
        - 'std_d2min': Standard deviation of D²min
        - 'max_d2min': Maximum D²min (largest plastic event)
        - 'd2min_per_particle': Full array (n_strains, N)
    """
    n_steps = len(trajectory)
    if window >= n_steps:
        window = 1

    n_compute = n_steps - window
    N = trajectory.shape[1]
    d2min_history = np.zeros((n_compute, N))

    for i in range(n_compute):
        t0 = i
        t1 = i + window

        box_t = box[t0] if box is not None and box.ndim == 3 else box
        d2min_history[i] = compute_nonaffine_displacement(
            trajectory[t0], trajectory[t1], box=box_t, r_cut=r_cut
        )

    return {
        "strains": strains[window:],
        "mean_d2min": np.mean(d2min_history, axis=1),
        "std_d2min": np.std(d2min_history, axis=1),
        "max_d2min": np.max(d2min_history, axis=1),
        "d2min_per_particle": d2min_history,
    }


def identify_plastic_events(
    d2min: np.ndarray,
    threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """Identify particles that underwent plastic rearrangement.

    Args:
        d2min: Non-affine displacement values, shape (N,).
        threshold: D²min threshold above which a particle is considered
            to have undergone a plastic event.

    Returns:
        Dictionary with:
        - 'plastic_particles': Indices of plastic particles
        - 'n_plastic': Count of plastic particles
        - 'fraction_plastic': Fraction of particles that are plastic
        - 'mean_d2min_plastic': Mean D²min of plastic particles
    """
    plastic_mask = d2min > threshold
    plastic_indices = np.where(plastic_mask)[0]

    return {
        "plastic_particles": plastic_indices,
        "n_plastic": int(len(plastic_indices)),
        "fraction_plastic": float(len(plastic_indices) / len(d2min)),
        "mean_d2min_plastic": float(np.mean(d2min[plastic_indices])) if len(plastic_indices) > 0 else 0.0,
    }


def _find_neighbors(
    positions: np.ndarray,
    r_cut: float,
    box: np.ndarray | None = None,
) -> list[list[int]]:
    """Find neighbors within r_cut for each particle.

    Args:
        positions: Particle positions, shape (N, 3).
        r_cut: Cutoff radius.
        box: Simulation box [[xlo,xhi],[ylo,yhi],[zlo,zhi]].

    Returns:
        List of neighbor indices for each particle.
    """
    N = len(positions)
    neighbors: list[list[int]] = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            dr = positions[j] - positions[i]
            if box is not None:
                dr = _apply_pbc_single(dr, box)
            dist = np.linalg.norm(dr)
            if dist < r_cut:
                neighbors[i].append(j)
                neighbors[j].append(i)

    return neighbors


def _apply_pbc(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply minimum image convention to displacement vectors."""
    result = dr.copy()
    for dim in range(3):
        lo, hi = box[dim]
        L = hi - lo
        result[:, dim] -= L * np.round(result[:, dim] / L)
    return result


def _apply_pbc_single(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply minimum image convention to a single displacement vector."""
    result = dr.copy()
    for dim in range(3):
        lo, hi = box[dim]
        L = hi - lo
        result[dim] -= L * np.round(result[dim] / L)
    return result
