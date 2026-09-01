import math
import unittest

try:
    from .evaluate_residual_blade_deformation import (
        project_deformed_blade_centerlines,
        verify_zero_deformation_projection,
    )
except ImportError:
    from evaluate_residual_blade_deformation import (
        project_deformed_blade_centerlines,
        verify_zero_deformation_projection,
    )


class ResidualBladeDeformationProjectionTest(unittest.TestCase):
    def test_zero_deformation_matches_existing_projection(self) -> None:
        verify_zero_deformation_projection()

    def test_projected_directions_are_unit_length(self) -> None:
        result = project_deformed_blade_centerlines(
            25.0,
            40.0,
            15.0,
            -8.0,
            {1: 1.0, 2: -2.0, 3: 4.0},
        )
        for pair in result["unit_pairs"].values():
            self.assertTrue(math.isclose(math.hypot(*pair), 1.0, abs_tol=1e-12))

    def test_nonzero_deformation_changes_projection(self) -> None:
        zero = project_deformed_blade_centerlines(
            30.0, 60.0, 0.0, 0.0, {1: 0.0, 2: 0.0, 3: 0.0}
        )
        deformed = project_deformed_blade_centerlines(
            30.0, 60.0, 0.0, 0.0, {1: 2.0, 2: -1.0, 3: 3.0}
        )
        differences = []
        for blade_id in (1, 2, 3):
            differences.append(
                max(
                    abs(a - b)
                    for a, b in zip(
                        zero["unit_pairs"][blade_id],
                        deformed["unit_pairs"][blade_id],
                    )
                )
            )
        self.assertGreater(max(differences), 1e-6)

    def test_common_flapwise_tilt_changes_projection(self) -> None:
        zero = project_deformed_blade_centerlines(
            30.0, 60.0, 0.0, 0.0, {1: 0.0, 2: 0.0, 3: 0.0}
        )
        deformed = project_deformed_blade_centerlines(
            30.0, 60.0, 0.0, 0.0, {1: 2.0, 2: 2.0, 3: 2.0}
        )
        differences = [
            max(
                abs(a - b)
                for a, b in zip(
                    zero["unit_pairs"][blade_id],
                    deformed["unit_pairs"][blade_id],
                )
            )
            for blade_id in (1, 2, 3)
        ]
        self.assertGreater(max(differences), 1e-6)


if __name__ == "__main__":
    unittest.main()

