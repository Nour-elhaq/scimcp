"""LAMMPS output file parsers — dump trajectories and thermo logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_thermo_data(filepath: str) -> dict[str, Any]:
    """Parse LAMMPS thermodynamic output (thermo_style custom).

    Reads a LAMMPS log file or thermo.dat and extracts the time-series data
    from the thermo_style custom output block.

    Args:
        filepath: Path to the LAMMPS log or thermo output file.

    Returns:
        Dictionary with keys: 'headers' (list of column names), 'data' (numpy array),
        'n_steps' (int), and 'columns' (dict of column_name -> numpy array).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Thermo file not found: {filepath}")

    content = path.read_text()
    headers: list[str] = []
    data_rows: list[list[float]] = []

    in_thermo_block = False
    found_header = False

    for line in content.splitlines():
        stripped = line.strip()

        # Detect start of thermo output: header line with known column names
        if not found_header and "Step" in stripped and ("Temp" in stripped or "PotEng" in stripped or "TotEng" in stripped):
            headers = stripped.split()
            found_header = True
            in_thermo_block = True
            continue

        if in_thermo_block:
            # Skip separator lines
            if re.match(r"^[-=]+$", stripped):
                if data_rows:
                    break  # End of this thermo block
                continue

            if not stripped:
                if data_rows:
                    break  # Empty line ends block
                continue

            # Parse data line
            try:
                values = [float(v) for v in stripped.split()]
                if len(values) == len(headers):
                    data_rows.append(values)
            except ValueError:
                if data_rows:
                    break  # Non-numeric line ends block
                continue

    if not headers or not data_rows:
        return {
            "headers": [],
            "data": np.array([]),
            "n_steps": 0,
            "columns": {},
        }

    data = np.array(data_rows)
    columns = {headers[i]: data[:, i] for i in range(len(headers))}

    return {
        "headers": headers,
        "data": data,
        "n_steps": len(data_rows),
        "columns": columns,
    }


def parse_dump_file(
    filepath: str, max_frames: int = 0
) -> dict[str, Any]:
    """Parse a LAMMPS custom dump file (lammpstrj format).

    Reads trajectory frames from a LAMMPS dump file. Each frame contains
    atom positions, velocities, and box information.

    Args:
        filepath: Path to the LAMMPS dump file.
        max_frames: Maximum number of frames to read (0 = all).

    Returns:
        Dictionary with:
        - 'n_frames': Total number of frames read
        - 'n_atoms': Number of atoms per frame
        - 'frames': List of dicts, each containing:
            - 'step': Timestep number
            - 'box': [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
            - 'columns': List of column names (id, type, x, y, z, ...)
            - 'data': numpy array of shape (n_atoms, n_columns)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dump file not found: {filepath}")

    frames: list[dict[str, Any]] = []
    n_atoms = 0
    columns: list[str] = []
    current_frame: dict[str, Any] = {}
    state = "seeking_header"

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            if state == "seeking_header":
                if stripped.startswith("ITEM: TIMESTEP"):
                    state = "reading_timestep"
                    current_frame = {}
                elif stripped.startswith("ITEM: NUMBER OF ATOMS"):
                    state = "reading_natoms"
                elif stripped.startswith("ITEM: BOX BOUNDS"):
                    state = "reading_box"
                    current_frame["box"] = []
                    bounds = stripped.replace("ITEM: BOX BOUNDS", "").strip().split()
                    current_frame["boundary_types"] = bounds
                elif stripped.startswith("ITEM: ATOMS"):
                    columns = stripped.replace("ITEM: ATOMS", "").strip().split()
                    state = "reading_atoms"
                    current_frame["data_lines"] = []

            elif state == "reading_timestep":
                current_frame["step"] = int(stripped)
                state = "seeking_header"

            elif state == "reading_natoms":
                n_atoms = int(stripped)
                current_frame["n_atoms"] = n_atoms
                state = "seeking_header"

            elif state == "reading_box":
                parts = stripped.split()
                current_frame["box"].append([float(parts[0]), float(parts[1])])
                if len(current_frame["box"]) == 3:
                    state = "seeking_header"

            elif state == "reading_atoms":
                if stripped.startswith("ITEM:"):
                    # End of atoms, save frame
                    data_lines = current_frame.pop("data_lines", [])
                    if data_lines and columns:
                        data = np.array(
                            [line.split() for line in data_lines]
                        )
                        try:
                            data = data.astype(float)
                        except ValueError:
                            pass
                        current_frame["columns"] = list(columns)
                        current_frame["data"] = data
                        frames.append(current_frame)

                        if max_frames > 0 and len(frames) >= max_frames:
                            break

                    # Process the new ITEM line
                    current_frame = {}
                    if stripped.startswith("ITEM: TIMESTEP"):
                        state = "reading_timestep"
                    elif stripped.startswith("ITEM: NUMBER OF ATOMS"):
                        state = "reading_natoms"
                    elif stripped.startswith("ITEM: BOX BOUNDS"):
                        state = "reading_box"
                        current_frame["box"] = []
                    elif stripped.startswith("ITEM: ATOMS"):
                        columns = stripped.replace("ITEM: ATOMS", "").strip().split()
                        state = "reading_atoms"
                        current_frame["data_lines"] = []
                else:
                    current_frame["data_lines"].append(stripped)

    # Handle last frame
    if state == "reading_atoms" and current_frame.get("data_lines"):
        data_lines = current_frame.pop("data_lines", [])
        if data_lines and columns:
            data = np.array([line.split() for line in data_lines])
            try:
                data = data.astype(float)
            except ValueError:
                pass
            current_frame["columns"] = list(columns)
            current_frame["data"] = data
            frames.append(current_frame)

    return {
        "n_frames": len(frames),
        "n_atoms": n_atoms,
        "frames": frames,
    }


def extract_trajectory(
    dump_data: dict[str, Any],
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> np.ndarray:
    """Extract position coordinates from parsed dump data into a trajectory array.

    Args:
        dump_data: Output from parse_dump_file().
        x_col: Column name for x coordinate.
        y_col: Column name for y coordinate.
        z_col: Column name for z coordinate.

    Returns:
        numpy array of shape (n_frames, n_atoms, 3).
    """
    frames = dump_data["frames"]
    if not frames:
        return np.array([])

    first_frame = frames[0]
    cols = first_frame["columns"]

    x_idx = cols.index(x_col) if x_col in cols else None
    y_idx = cols.index(y_col) if y_col in cols else None
    z_idx = cols.index(z_col) if z_col in cols else None

    if x_idx is None or y_idx is None or z_idx is None:
        raise ValueError(f"Columns {x_col}, {y_col}, {z_col} not found in dump file")

    n_atoms = first_frame["data"].shape[0]
    n_frames = len(frames)
    trajectory = np.zeros((n_frames, n_atoms, 3))

    for i, frame in enumerate(frames):
        data = frame["data"]
        trajectory[i, :, 0] = data[:, x_idx]
        trajectory[i, :, 1] = data[:, y_idx]
        trajectory[i, :, 2] = data[:, z_idx]

    return trajectory


def get_summary(filepath: str) -> dict[str, Any]:
    """Get a quick summary of a LAMMPS output file.

    Identifies the file type (dump or thermo) and returns key statistics.

    Args:
        filepath: Path to the LAMMPS output file.

    Returns:
        Dictionary with file type, size, and key statistics.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    content = path.read_text()
    file_size = path.stat().st_size

    # Detect file type
    is_dump = "ITEM: ATOMS" in content
    is_thermo = "Step" in content and ("Temp" in content or "PotEng" in content)

    summary: dict[str, Any] = {
        "file": str(path),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "is_dump": is_dump,
        "is_thermo": is_thermo,
    }

    if is_dump:
        dump_data = parse_dump_file(filepath)
        summary["n_frames"] = dump_data["n_frames"]
        summary["n_atoms"] = dump_data["n_atoms"]
        if dump_data["frames"]:
            first_frame = dump_data["frames"][0]
            summary["columns"] = first_frame["columns"]
            box = first_frame.get("box", [])
            if box:
                summary["box"] = {
                    "x": [box[0][0], box[0][1]],
                    "y": [box[1][0], box[1][1]],
                    "z": [box[2][0], box[2][1]],
                }

    elif is_thermo:
        thermo_data = parse_thermo_data(filepath)
        summary["n_steps"] = thermo_data["n_steps"]
        summary["columns"] = thermo_data["headers"]
        for col_name, col_data in thermo_data["columns"].items():
            summary[f"{col_name}_min"] = float(np.min(col_data))
            summary[f"{col_name}_max"] = float(np.max(col_data))
            summary[f"{col_name}_mean"] = float(np.mean(col_data))

    return summary
