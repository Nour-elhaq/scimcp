"""Trajectory visualization tools.

Generates plots from LAMMPS simulation data: time series,
histograms, scatter plots, and trajectory snapshots.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def plot_time_series(
    data_json: str,
    x_label: str = "Step",
    y_label: str = "Value",
    title: str = "Time Series",
    output_file: str = "",
    return_base64: bool = False,
) -> str:
    """Plot time series data from LAMMPS thermo output.

    Args:
        data_json: JSON object with 'x' and 'y' arrays, or dict of y-arrays.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Plot title.
        output_file: If provided, save plot as PNG.
        return_base64: If True, return base64-encoded PNG.

    Returns:
        JSON string with plot metadata, or base64 PNG if requested.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(data_json)

    fig, ax = plt.subplots(figsize=(10, 6))

    if "x" in data and "y" in data:
        ax.plot(data["x"], data["y"], linewidth=1.5)
    elif isinstance(data, dict):
        for key, values in data.items():
            ax.plot(values, label=key, linewidth=1.5)
        if len(data) > 1:
            ax.legend()
    else:
        ax.plot(data)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    result = {"shape": "time_series", "n_points": len(data.get("x", data.get(list(data.keys())[0], [])))}

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        result["file"] = output_file

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result["base64"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)
    return json.dumps(result, indent=2)


def plot_histogram(
    data_json: str,
    n_bins: int = 50,
    x_label: str = "Value",
    y_label: str = "Count",
    title: str = "Distribution",
    output_file: str = "",
    return_base64: bool = False,
) -> str:
    """Plot a histogram of data values.

    Args:
        data_json: JSON array of values.
        n_bins: Number of bins.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Plot title.
        output_file: If provided, save plot as PNG.
        return_base64: If True, return base64-encoded PNG.

    Returns:
        JSON string with plot metadata.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.array(json.loads(data_json))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=n_bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    stats = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }

    result = {"shape": "histogram", "stats": stats, "n_bins": n_bins}

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        result["file"] = output_file

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result["base64"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)
    return json.dumps(result, indent=2)


def plot_scatter(
    x_json: str,
    y_json: str,
    color_json: str = "",
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Scatter Plot",
    output_file: str = "",
    return_base64: bool = False,
) -> str:
    """Plot a scatter plot, optionally colored by a third variable.

    Args:
        x_json: JSON array of x values.
        y_json: JSON array of y values.
        color_json: Optional JSON array for color mapping (e.g., D²min values).
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Plot title.
        output_file: If provided, save plot as PNG.
        return_base64: If True, return base64-encoded PNG.

    Returns:
        JSON string with plot metadata.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(json.loads(x_json))
    y = np.array(json.loads(y_json))

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_json:
        colors = np.array(json.loads(color_json))
        scatter = ax.scatter(x, y, c=colors, cmap="viridis", s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Color")
    else:
        ax.scatter(x, y, s=20, alpha=0.7, color="steelblue")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    result = {"shape": "scatter", "n_points": len(x)}

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        result["file"] = output_file

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result["base64"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)
    return json.dumps(result, indent=2)


def plot_phonon_dos(
    frequencies_json: str,
    dos_json: str,
    title: str = "Phonon Density of States",
    output_file: str = "",
    return_base64: bool = False,
) -> str:
    """Plot phonon density of states.

    Args:
        frequencies_json: JSON array of frequency values (THz).
        dos_json: JSON array of DOS values.
        title: Plot title.
        output_file: If provided, save plot as PNG.
        return_base64: If True, return base64-encoded PNG.

    Returns:
        JSON string with plot metadata.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.array(json.loads(frequencies_json))
    dos = np.array(json.loads(dos_json))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_betweenx(dos, freqs, alpha=0.7, color="steelblue")
    ax.plot(freqs, dos, color="navy", linewidth=1.5)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Frequency = 0")
    ax.set_xlabel("Frequency (THz)", fontsize=12)
    ax.set_ylabel("DOS", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    result = {"shape": "phonon_dos", "n_points": len(freqs)}

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        result["file"] = output_file

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result["base64"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)
    return json.dumps(result, indent=2)


def plot_thermo_dashboard(
    thermo_data_json: str,
    output_file: str = "",
    return_base64: bool = False,
) -> str:
    """Generate a 4-panel dashboard from LAMMPS thermo data.

    Panels: Temperature, Energy, Pressure, Volume vs Step.

    Args:
        thermo_data_json: JSON object with columns as keys and arrays as values.
            Example: {"Step": [...], "Temp": [...], "PotEng": [...], ...}
        output_file: If provided, save plot as PNG.
        return_base64: If True, return base64-encoded PNG.

    Returns:
        JSON string with plot metadata.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(thermo_data_json)

    # Find common columns
    step_key = next((k for k in data if "step" in k.lower()), None)
    temp_key = next((k for k in data if "temp" in k.lower()), None)
    pe_key = next((k for k in data if "pot" in k.lower() or "pe" in k.lower()), None)
    press_key = next((k for k in data if "press" in k.lower()), None)

    n_panels = sum(1 for k in [temp_key, pe_key, press_key] if k)
    if n_panels == 0:
        return json.dumps({"error": "No recognizable thermo columns found"}, indent=2)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    steps = data.get(step_key, list(range(len(data[temp_key or pe_key or press_key]))))
    panel_idx = 0

    if temp_key:
        axes[panel_idx].plot(steps[:len(data[temp_key])], data[temp_key], linewidth=1, color="red")
        axes[panel_idx].set_ylabel("Temperature (K)", fontsize=11)
        axes[panel_idx].grid(True, alpha=0.3)
        panel_idx += 1

    if pe_key:
        axes[panel_idx].plot(steps[:len(data[pe_key])], data[pe_key], linewidth=1, color="blue")
        axes[panel_idx].set_ylabel("Potential Energy (eV)", fontsize=11)
        axes[panel_idx].grid(True, alpha=0.3)
        panel_idx += 1

    if press_key:
        axes[panel_idx].plot(steps[:len(data[press_key])], data[press_key], linewidth=1, color="green")
        axes[panel_idx].set_ylabel("Pressure (atm)", fontsize=11)
        axes[panel_idx].set_xlabel("Step", fontsize=11)
        axes[panel_idx].grid(True, alpha=0.3)
        panel_idx += 1

    fig.suptitle("LAMMPS Thermo Dashboard", fontsize=14, y=1.02)
    fig.tight_layout()

    result = {
        "shape": "thermo_dashboard",
        "panels": [k for k in [temp_key, pe_key, press_key] if k],
    }

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        result["file"] = output_file

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result["base64"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)
    return json.dumps(result, indent=2)
