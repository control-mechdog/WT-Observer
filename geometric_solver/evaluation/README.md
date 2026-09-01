# Geometric solver evaluation

This directory contains the scripts used for the numerical experiments in the
paper section **Evaluation of Geometric Pose Solver**. The scripts use the solver
from the parent `geometric_solver` directory and can be executed either directly
or as Python modules.

## Included experiments

- `evaluate_unit_centerline_solver_errors.py` reproduces the 2-D centerline-noise,
  camera-pitch-noise, and camera-roll-noise sensitivity tables.
- `evaluate_residual_blade_deformation.py` reproduces the equivalent 3-D blade
  deformation experiment and combines it with the 2-D centerline results.
- `plot_side_view_sensitivity.py` reproduces the near-side-view Monte Carlo study
  and generates its CSV summary and PDF/PNG figure.
- `test_residual_blade_deformation.py` checks the deformed-blade projection model.
- `results/` contains the summary CSV files used to prepare the reported tables and
  figure. Large per-trial CSV files are intentionally excluded because every trial
  can be regenerated deterministically from the scripts and recorded seeds.

## Installation

From the repository root:

```bash
python -m pip install -r geometric_solver/evaluation/requirements.txt
```

Python 3.10 or later is recommended.

## Reproduce the paper settings

Run the following commands from the repository root:

```bash
python geometric_solver/evaluation/evaluate_unit_centerline_solver_errors.py
python geometric_solver/evaluation/evaluate_residual_blade_deformation.py
python geometric_solver/evaluation/plot_side_view_sensitivity.py
```

The default output directory is `geometric_solver/evaluation/results/`. The first
two commands write both a compact summary CSV and a larger per-trial CSV. The
side-view command writes a summary CSV and PDF/PNG figures. Existing files with the
same names are replaced.

The experiments use fixed random seeds. The unit-centerline experiment must be run
before the deformation experiment because the latter reads
`unit_centerline_solver_error_summary.csv` when constructing the combined summary.

To redraw the side-view figure from the included summary without rerunning the
Monte Carlo experiment:

```bash
python geometric_solver/evaluation/plot_side_view_sensitivity.py --reuse-summary
```

## Quick verification

```bash
python -m unittest geometric_solver.test_geometric_solver -v
python -m unittest geometric_solver.evaluation.test_residual_blade_deformation -v
node geometric_solver/test_web_model.js
```

All centerline vectors follow the directed hub-to-tip convention. The aggregate
sensitivity tables exclude the exact-side-view neighborhood
`|relative_yaw - 90 degrees| <= 5 degrees`; the side-view script evaluates that
region separately.
