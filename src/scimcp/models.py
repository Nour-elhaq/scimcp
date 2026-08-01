"""Typed Pydantic models for SciMCP inputs and outputs."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


# ── Crystallography ────────────────────────────────────────────────────────

class LatticeParameters(BaseModel):
    a: float = Field(..., gt=0, description="Lattice constant a (Angstrom)")
    b: float = Field(..., gt=0, description="Lattice constant b (Angstrom)")
    c: float = Field(..., gt=0, description="Lattice constant c (Angstrom)")
    alpha: float = Field(90.0, ge=0, le=180, description="Angle alpha (degrees)")
    beta: float = Field(90.0, ge=0, le=180, description="Angle beta (degrees)")
    gamma: float = Field(90.0, ge=0, le=180, description="Angle gamma (degrees)")

    def volume(self) -> float:
        import math
        ca = math.cos(math.radians(self.alpha))
        cb = math.cos(math.radians(self.beta))
        cg = math.cos(math.radians(self.gamma))
        sg = math.sin(math.radians(self.gamma))
        return self.a * self.b * self.c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)


class CrystalStructure(BaseModel):
    lattice: LatticeParameters
    species: list[str] = Field(..., min_length=1, description="Element symbols")
    fractional_coordinates: list[list[float]] = Field(..., min_length=1)
    space_group: int | None = Field(None, ge=1, le=230)

    def n_atoms(self) -> int:
        return len(self.species)

    def formula(self) -> str:
        from collections import Counter
        counts = Counter(self.species)
        return "".join(f"{e}{c}" if c > 1 else e for e, c in counts.items())

    def density(self, atomic_masses: dict[str, float]) -> float:
        """Compute density in g/cm³ from lattice and atomic masses."""
        V = self.lattice.volume() * 1e-24  # Angstrom³ → cm³
        mass = sum(atomic_masses.get(s, 0) for s in self.species) / 6.022e23
        return mass / V


class CIFInput(BaseModel):
    cif_content: str = Field(..., min_length=10, description="Full CIF file content")


class CIFOutput(BaseModel):
    space_group: dict | None = None
    lattice: dict | None = None
    n_atoms: int
    formula: str
    atom_sites: list[dict]
    metadata_keys: list[str]


# ── LAMMPS ─────────────────────────────────────────────────────────────────

class LAMMPSInputParams(BaseModel):
    atoms_per_side: int = Field(10, ge=2, le=100)
    lattice_spacing: float = Field(5.26, gt=0)
    potential: Literal["lj", "eam", "tersoff", "buckingham", "coulomb"] = "lj"
    epsilon: float = Field(1.0, gt=0)
    sigma: float = Field(1.0, gt=0)
    cutoff: float = Field(2.5, gt=0)
    ensemble: Literal["nvt", "npt", "nve", "nph"] = "nvt"
    temperature: float = Field(300.0, gt=0)
    pressure: float = Field(0.0)
    timestep: float = Field(0.002, gt=0)
    n_steps: int = Field(100000, gt=0)
    dump_freq: int = Field(1000, gt=0)
    minimize: bool = True
    apply_shear: bool = False
    shear_rate: float = Field(0.001, gt=0)
    shear_direction: Literal["xy", "xz", "yz"] = "xy"
    boundary: str = "p p p"
    output_file: str = ""


class Quaternion(BaseModel):
    w: float
    x: float
    y: float
    z: float

    def as_array(self) -> list[float]:
        return [self.w, self.x, self.y, self.z]


class NematicOrderInput(BaseModel):
    quaternions: list[Quaternion] = Field(..., min_length=1)


class NematicOrderResult(BaseModel):
    order_parameter: float = Field(..., description="Scalar nematic order S ∈ [-0.5, 1]")
    director: list[float]
    particle_count: int
    interpretation: str


# ── D²min / Non-affine ────────────────────────────────────────────────────

class D2minInput(BaseModel):
    positions_t0: list[list[float]] = Field(..., min_length=2)
    positions_t1: list[list[float]] = Field(..., min_length=2)
    r_cut: float = Field(3.0, gt=0)


class D2minResult(BaseModel):
    d2min: list[float]
    mean_d2min: float
    max_d2min: float
    n_plastic: int
    fraction_plastic: float


# ── Shear Rheology ────────────────────────────────────────────────────────

class ShearSweepResult(BaseModel):
    K: float = Field(..., description="Consistency index")
    n: float = Field(..., description="Flow index")
    viscosity_model: str = Field(..., description="newtonian / shear_thinning / shear_thickening")
    R_squared: float = Field(..., ge=0, le=1)


# ── ML Prediction ─────────────────────────────────────────────────────────

class PropertyPredictionResult(BaseModel):
    property_name: str
    predicted_value: float
    unit: str
    uncertainty: float | None = None
    model_name: str
    model_version: str
    applicability: str = "in_domain"
    warning: str = ""


# ── Phonon / Thermodynamics ───────────────────────────────────────────────

class ThermodynamicProperties(BaseModel):
    temperature_K: float
    ZPE_eV: float = Field(..., ge=0, description="Zero-point energy")
    entropy_J_mol_K: float = Field(..., ge=0)
    Cv_J_mol_K: float = Field(..., ge=0)
    Helmholtz_eV: float
    n_atoms: int


class PhononDOSResult(BaseModel):
    frequency_THz: list[float]
    dos: list[float]
    n_frequencies: int
    sigma_THz: float


# ── VASP / QE ─────────────────────────────────────────────────────────────

class VaspIncarParams(BaseModel):
    calculation: Literal["static", "relaxation", "cell_optimization", "band_structure", "dos"] = "static"
    encut: float = Field(520, gt=0, description="Plane-wave cutoff (eV)")
    ediff: float = Field(1e-6, gt=0)
    isif: int = Field(3, ge=1, le=7)
    ibrion: int = Field(-1, ge=-1, le=3)
    nsw: int = Field(0, ge=0)
    ispin: int = Field(1, ge=1, le=2)
    lorbit: int = Field(11, ge=0, le=12)
    sigma: float = Field(0.1, gt=0)
    ismear: int = Field(1, ge=-5, le=2)

    def validate_for_calculation(self) -> list[str]:
        warnings = []
        if self.calculation == "static" and self.nsw > 0:
            warnings.append("Static calculation should have NSW=0.")
        if self.calculation in ("relaxation", "cell_optimization") and self.nsw == 0:
            warnings.append(f"{self.calculation} requires NSW>0.")
        if self.ismear == -5 and self.calculation == "dos":
            warnings.append("Tetrahedron method (ISMEAR=-5) not recommended for DOS with small k-mesh.")
        return warnings


class QEInputParams(BaseModel):
    calculation: Literal["scf", "nscf", "relax", "vc-relax", "md"] = "scf"
    prefix: str = "scimcp"
    pseudo_dir: str = ""
    ecutwfc: float = Field(30.0, gt=0, description="Cutoff (Ry)")
    ecutrho: float = Field(240.0, gt=0, description="Charge density cutoff (Ry)")
    conv_thr: float = Field(1e-6, gt=0)
    k_points: list[int] = Field(default_factory=lambda: [4, 4, 4], min_length=3)


# ── Visualization ──────────────────────────────────────────────────────────

class TimeSeriesPlotInput(BaseModel):
    x: list[float]
    y: list[float]
    x_label: str = "Step"
    y_label: str = "Value"
    title: str = "Time Series"


class HistogramPlotInput(BaseModel):
    data: list[float]
    n_bins: int = Field(50, gt=0)
    x_label: str = "Value"
    y_label: str = "Count"
    title: str = "Distribution"
