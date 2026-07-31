"""Compute nematic alignment parameter S(t)/S(z) from quaternion data.

Implements the Q-tensor approach for computing nematic order in
liquid crystal and anisotropic particle simulations.
"""

from __future__ import annotations

import numpy as np


def quaternion_to_q_tensor(quaternions: np.ndarray) -> np.ndarray:
    """Convert particle quaternions (w, x, y, z) to the alignment tensor Q.

    The Q-tensor is defined as:
        Q_ab = (1/N) * sum_i [ (3/2) * u_a * u_b - (1/2) * delta_ab ]

    where u is the orientation unit vector derived from the quaternion.

    Args:
        quaternions: Array of shape (N, 4) with columns [w, x, y, z].

    Returns:
        Q-tensor of shape (3, 3).
    """
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) quaternion array, got shape {quaternions.shape}")

    # Normalize quaternions
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    q = quaternions / norms

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Compute orientation vectors from quaternions
    # The director (long axis) of the particle in the lab frame
    u_x = 2 * (x * z + w * y)
    u_y = 2 * (y * z - w * x)
    u_z = 1 - 2 * (x * x + y * y)

    N = len(quaternions)
    Q = np.zeros((3, 3))

    for a, u_a in enumerate([u_x, u_y, u_z]):
        for b, u_b in enumerate([u_x, u_y, u_z]):
            Q[a, b] = (3.0 / 2.0) * np.mean(u_a * u_b) - (0.5 if a == b else 0.0)

    return Q


def compute_nematic_order(quaternions: np.ndarray) -> float:
    """Compute the scalar nematic order parameter S.

    S is the largest eigenvalue of the Q-tensor. S=1 means perfect alignment,
    S=0 means isotropic, S=-0.5 means perpendicular alignment.

    Args:
        quaternions: Array of shape (N, 4) with columns [w, x, y, z].

    Returns:
        Scalar nematic order parameter S.
    """
    Q = quaternion_to_q_tensor(quaternions)
    eigenvalues = np.linalg.eigvalsh(Q)
    return float(np.max(eigenvalues))


def compute_nematic_alignment_vs_time(
    quaternions_frames: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute nematic alignment S(t) over a trajectory.

    Args:
        quaternions_frames: List of quaternion arrays, one per timestep.
            Each array has shape (N, 4) with columns [w, x, y, z].

    Returns:
        Dictionary with:
        - 't': Step indices (0, 1, 2, ...)
        - 'S': Nematic order parameter at each step
        - 'mean_S': Time-averaged S
        - 'std_S': Standard deviation of S
    """
    n_frames = len(quaternions_frames)
    S_values = np.zeros(n_frames)

    for i, quaternions in enumerate(quaternions_frames):
        S_values[i] = compute_nematic_order(quaternions)

    t = np.arange(n_frames)

    return {
        "t": t,
        "S": S_values,
        "mean_S": float(np.mean(S_values)),
        "std_S": float(np.std(S_values)),
    }


def compute_nematic_alignment_vs_z(
    quaternions: np.ndarray,
    z_positions: np.ndarray,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    """Compute nematic alignment profile S(z) along the z-axis.

    Bins particles by their z-coordinate and computes S in each bin.

    Args:
        quaternions: Quaternion array of shape (N, 4) with columns [w, x, y, z].
        z_positions: Z-coordinates of shape (N,).
        n_bins: Number of bins along z.

    Returns:
        Dictionary with:
        - 'z_centers': Bin center positions
        - 'S': Nematic order in each bin
        - 'counts': Number of particles per bin
    """
    if len(quaternions) != len(z_positions):
        raise ValueError("quaternions and z_positions must have the same length")

    z_min, z_max = z_positions.min(), z_positions.max()
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    S_values = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (z_positions >= bin_edges[i]) & (z_positions < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= z_positions == bin_edges[i + 1]

        counts[i] = np.sum(mask)
        if counts[i] >= 3:
            S_values[i] = compute_nematic_order(quaternions[mask])
        else:
            S_values[i] = 0.0

    return {
        "z_centers": z_centers,
        "S": S_values,
        "counts": counts,
    }


def compute_q_tensor_components(quaternions: np.ndarray) -> dict[str, float]:
    """Compute all 6 unique components of the Q-tensor.

    Args:
        quaternions: Array of shape (N, 4) with columns [w, x, y, z].

    Returns:
        Dictionary with Qxx, Qyy, Qzz, Qxy, Qxz, Qyz.
    """
    Q = quaternion_to_q_tensor(quaternions)
    return {
        "Qxx": float(Q[0, 0]),
        "Qyy": float(Q[1, 1]),
        "Qzz": float(Q[2, 2]),
        "Qxy": float(Q[0, 1]),
        "Qxz": float(Q[0, 2]),
        "Qyz": float(Q[1, 2]),
    }
