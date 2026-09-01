# WT-Observer geometric pose solver

This directory contains the reference implementation of the geometric stage used in
the revised WT-Observer paper, together with a browser-based blade-centerline
direction generator.

## Contents

- `wt_pose_from_unit_centerlines.py`: estimates blade rotation `alpha` and relative
  nacelle yaw `gamma` from two or three directed image-plane centerlines.
- `wt_pose_to_blade_centerlines.py`: generates ideal centerline directions from known
  pose and camera attitude and can add independent angular perturbations.
- `wind_turbine_vector_ui.html`: standalone interactive version of the forward model.
- `test_geometric_solver.py`: closed-loop tests for three blades, two visible blades,
  independent centerline lengths, camera pitch/roll, and exact side-view degeneracy.
- `wt_pose_with_orientation.py`: compatibility imports for scripts using the earlier
  filename.

## Model conventions

The camera frame uses `X` to the right, `Y` along the optical axis, and `Z` upward.
Each input pair `(x, z)` is directed from the hub toward the blade tip. Blade IDs
`1`, `2`, and `3` correspond to fixed offsets of `0`, `120`, and `240` degrees.

The forward rotation chain is

```text
R(alpha, gamma, pitch, roll)
    = Ry(-roll) Rx(-pitch) Rz(gamma) Ry(alpha).
```

The solver independently normalizes every visible centerline, compensates the known
camera roll, obtains finite relative-yaw candidates from pairwise quartic equations in
`tan(gamma/2)`, recovers the blade angle analytically, and applies bounded local
Gauss--Newton refinement. The coarse orientation class selects the physically
consistent yaw branch. No angular grid search is used.

The reported blade angle is periodic modulo `120` degrees. At an exact left or right
side view, the continuous blade angle is unobservable and the solver raises
`PoseDegeneracyError`.

## Installation and tests

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest test_geometric_solver.py -v
node test_web_model.js
```

## Minimal example

```python
from wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
from wt_pose_to_blade_centerlines import project_blade_centerlines

truth = project_blade_centerlines(
    blade_rotation_deg=20.0,
    relative_yaw_deg=135.0,
    camera_pitch_deg=15.0,
    camera_roll_deg=-8.0,
)

result = solve_pose_from_unit_centerlines(
    truth["unit_pairs"],
    truth["orientation_class"],
    camera_pitch_deg=truth["camera_pitch_deg"],
    camera_roll_deg=truth["camera_roll_deg"],
)

print(result["best"]["alpha_deg_120"])
print(result["best"]["relative_yaw_deg_360"])
```

Open `wind_turbine_vector_ui.html` directly in a modern browser to generate matching
unit centerline directions without running a server.

