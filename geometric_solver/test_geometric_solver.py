# -*- coding: utf-8 -*-
import unittest

try:
    from .wt_pose_from_unit_centerlines import (
        PoseDegeneracyError,
        solve_pose_from_unit_centerlines,
    )
    from .wt_pose_to_blade_centerlines import project_blade_centerlines
except ImportError:
    from wt_pose_from_unit_centerlines import (
        PoseDegeneracyError,
        solve_pose_from_unit_centerlines,
    )
    from wt_pose_to_blade_centerlines import project_blade_centerlines


def periodic_error(estimate: float, truth: float, period: float) -> float:
    return (estimate - truth + period / 2.0) % period - period / 2.0


class UnitCenterlineSolverTest(unittest.TestCase):
    def assert_pose_close(self, result, alpha_deg, yaw_deg, places=6):
        best = result["best"]
        self.assertAlmostEqual(
            periodic_error(best["alpha_deg_120"], alpha_deg, 120.0),
            0.0,
            places=places,
        )
        self.assertAlmostEqual(
            periodic_error(best["relative_yaw_deg_360"], yaw_deg, 360.0),
            0.0,
            places=places,
        )
        self.assertGreater(best["min_directed_scale"], 0.0)

    def test_three_blade_pitch_roll_closed_loop(self):
        forward = project_blade_centerlines(
            blade_rotation_deg=20.0,
            relative_yaw_deg=135.0,
            camera_pitch_deg=15.0,
            camera_roll_deg=-8.0,
        )
        result = solve_pose_from_unit_centerlines(
            forward["unit_pairs"],
            "front-right",
            camera_pitch_deg=15.0,
            camera_roll_deg=-8.0,
        )
        self.assert_pose_close(result, 20.0, 135.0)
        self.assertLess(result["best"]["direction_sse"], 1e-20)

    def test_independent_blade_lengths_do_not_change_solution(self):
        forward = project_blade_centerlines(
            blade_rotation_deg=37.0,
            relative_yaw_deg=222.0,
            camera_pitch_deg=-31.0,
            camera_roll_deg=17.0,
        )
        scaled = {
            blade_id: (pair[0] * scale, pair[1] * scale)
            for (blade_id, pair), scale in zip(
                forward["unit_pairs"].items(),
                (0.4, 2.3, 19.0),
            )
        }
        result = solve_pose_from_unit_centerlines(
            scaled,
            forward["orientation_class"],
            camera_pitch_deg=-31.0,
            camera_roll_deg=17.0,
        )
        self.assert_pose_close(result, 37.0, 222.0)

    def test_two_visible_blades_are_supported(self):
        forward = project_blade_centerlines(
            blade_rotation_deg=20.0,
            relative_yaw_deg=135.0,
            camera_pitch_deg=15.0,
            camera_roll_deg=-8.0,
            blade_ids=(1, 2),
        )
        result = solve_pose_from_unit_centerlines(
            forward["unit_pairs"],
            forward["orientation_class"],
            camera_pitch_deg=15.0,
            camera_roll_deg=-8.0,
        )
        self.assert_pose_close(result, 20.0, 135.0)

    def test_exact_side_view_is_reported_as_degenerate(self):
        forward = project_blade_centerlines(
            blade_rotation_deg=20.0,
            relative_yaw_deg=90.0,
            camera_pitch_deg=15.0,
            camera_roll_deg=-8.0,
        )
        with self.assertRaises(PoseDegeneracyError):
            solve_pose_from_unit_centerlines(
                forward["unit_pairs"],
                "right",
                camera_pitch_deg=15.0,
                camera_roll_deg=-8.0,
            )


if __name__ == "__main__":
    unittest.main()

