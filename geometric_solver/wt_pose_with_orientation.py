"""Backward-compatible imports for the current WT-Observer pose solver.

New code should import :func:`solve_pose_from_unit_centerlines` from
``wt_pose_from_unit_centerlines`` directly. This module is kept so that scripts
written against the earlier public filename continue to run.
"""

try:
    from .wt_pose_from_unit_centerlines import (
        PoseDegeneracyError,
        solve_pose_from_unit_centerlines,
    )
except ImportError:
    from wt_pose_from_unit_centerlines import (
        PoseDegeneracyError,
        solve_pose_from_unit_centerlines,
    )


solve_pose_with_orientation = solve_pose_from_unit_centerlines

__all__ = [
    "PoseDegeneracyError",
    "solve_pose_from_unit_centerlines",
    "solve_pose_with_orientation",
]

