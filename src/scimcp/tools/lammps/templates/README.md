# LAMMPS Input Templates for SciMCP
# These are reference templates used by the generator tool.

## Template: Simple LJ NVT
```
units           lj
dimension       3
boundary        p p p
atom_style      atomic

lattice         fcc 5.26
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

timestep        0.002
thermo          100
thermo_style    custom step temp pe ke etotal press vol

min_style       cg
minimize        1.0e-4 1.0e-6 10000 100000

fix             1 all nvt temp 300.0 300.0 0.1
dump            1 all custom 1000 dump.lammpstrj id type x y z vx vy vz
dump_modify     1 sort id
run             100000
write_restart   restart.lammps
```

## Template: NPT Equilibration
```
units           lj
dimension       3
boundary        p p p
atom_style      atomic

lattice         fcc 5.26
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

timestep        0.002
thermo          100
thermo_style    custom step temp pe ke etotal press vol lx ly lz

min_style       cg
minimize        1.0e-4 1.0e-6 10000 100000

fix             1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
dump            1 all custom 1000 dump.npt.lammpstrj id type x y z vx vy vz
dump_modify     1 sort id
run             100000
write_restart   restart.npt.lammps
```

## Template: Shear Deformation
```
units           lj
dimension       3
boundary        p p p
atom_style      atomic

lattice         fcc 5.26
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

timestep        0.002
thermo          100
thermo_style    custom step temp pe ke etotal press vol lx ly lz xy

min_style       cg
minimize        1.0e-4 1.0e-6 10000 100000

fix             1 all nvt temp 300.0 300.0 0.1
fix             2 all deform 1 xy erate 0.001 units box remap x
dump            1 all custom 1000 dump.shear.lammpstrj id type x y z vx vy vz
dump_modify     1 sort id
run             100000
write_restart   restart.shear.lammps
```

## Template: Melting Point (Two-Phase)
```
units           metal
dimension       3
boundary        p p p
atom_style      atomic

# Create two-phase system (solid + liquid)
lattice         fcc 3.615
region          box block 0 10 0 10 0 20
create_box      1 box
create_atoms    1 box
mass            1 63.546

# Define regions for solid and liquid
region          solid block INF INF INF INF INF 10
region          liquid block INF INF INF INF 10 INF
set             region solid type 1
set             region liquid type 2

pair_style      eam/alloy
pair_coeff      1 2 Cu.eam.alloy Cu
pair_coeff      2 1 Cu.eam.alloy Cu
pair_coeff      1 1 Cu.eam.alloy Cu
pair_coeff      2 2 Cu.eam.alloy Cu

timestep        0.001
thermo          100
thermo_style    custom step temp pe ke etotal press

fix             1 all nvt temp 2000.0 2000.0 0.1
dump            1 all custom 1000 dump.melting.lammpstrj id type x y z
dump_modify     1 sort id
run             500000
write_restart   restart.melting.lammps
```
