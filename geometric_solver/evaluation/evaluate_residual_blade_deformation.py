# -*- coding: utf-8 -*-
"""Evaluate rigid-solver sensitivity to residual blade deformation.

The inverse solver is not modified.  Instead, each ideal blade direction is
tilted along the rotor axis before projection,

    b_i_tilde = cos(delta_i) * b_i + sin(delta_i) * a,

where ``a`` is the rotor-axis direction and ``delta_i`` is an equivalent
root-to-mid-span flapwise direction deviation.  The deformed 3-D directions
are projected to independent unit image-plane directions and passed to the
unchanged rigid solver.

The default pose grid, exact-side-view exclusion, perturbation levels, and
master seed match ``evaluate_unit_centerline_solver_errors.py``.  Unlike the
stochastic 2-D extraction error, each deterministic 3-D deflection case is
evaluated once per pose.  All individual trials and summaries are retained in
CSV files.  A third CSV
combines these results with the existing 2-D network-centerline perturbation
experiment for the compact comparison reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from ..wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
    from ..wt_pose_to_blade_centerlines import (
        BETAS_DEG,
        orientation_class_from_relative_yaw,
        project_blade_centerlines,
    )
except ImportError:
    import sys

    SOLVER_DIR = Path(__file__).resolve().parents[1]
    if str(SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(SOLVER_DIR))
    from wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
    from wt_pose_to_blade_centerlines import (
        BETAS_DEG,
        orientation_class_from_relative_yaw,
        project_blade_centerlines,
    )


SCHEMA_VERSION = "1.0"
SOLVER_VERSION = "unit_centerline_quartic_v1"
EXPERIMENT = "residual_deformation_3d"
PERTURBATION_LEVELS_DEG = (0.0, 1.0, 2.0, 4.0)
TRUE_PITCH_VALUES_DEG = (-45.0, 0.0, 45.0)
TRUE_ROLL_DEG = 0.0
EPS = 1e-12

RAW_FIELDS = (
    "schema_version",
    "solver_version",
    "experiment",
    "perturbation_domain",
    "deformation_model",
    "deformation_magnitude_deg",
    "deformation_sign",
    "group_seed",
    "case_index",
    "repeat_id",
    "blade_rotation_true_deg",
    "relative_yaw_true_deg",
    "orientation_class",
    "camera_pitch_true_deg",
    "camera_roll_true_deg",
    "deformation_blade1_deg",
    "deformation_blade2_deg",
    "deformation_blade3_deg",
    "min_projection_norm",
    "success",
    "blade_rotation_est_deg",
    "relative_yaw_est_deg",
    "direction_sse",
    "blade_signed_error_deg",
    "blade_abs_error_deg",
    "yaw_signed_error_deg",
    "yaw_abs_error_deg",
    "error_message",
)

SUMMARY_FIELDS = (
    "schema_version",
    "solver_version",
    "experiment",
    "perturbation_domain",
    "deformation_model",
    "deformation_magnitude_deg",
    "deformation_sign",
    "camera_pitch_true_deg",
    "camera_roll_true_deg",
    "n_total",
    "n_success",
    "n_failure",
    "failure_rate",
    "yaw_mae_deg",
    "blade_mae_deg",
    "alpha_step_deg",
    "yaw_step_deg",
    "side_exclusion_deg",
    "base_seed",
    "group_seed",
    "configured_repeats",
    "effective_repeats",
    "refinement_enabled",
    "max_refine_seeds",
)

COMBINED_FIELDS = (
    "source",
    "source_label",
    "perturbation_domain",
    "perturbation_model",
    "level_definition",
    "angular_level_deg",
    "n_total",
    "n_success",
    "n_failure",
    "overall_failure_rate",
    "max_pitch_failure_rate",
    "yaw_mae_max_over_pitch_deg",
    "blade_mae_max_over_pitch_deg",
    "pitch_at_yaw_max_deg",
    "pitch_at_blade_max_deg",
    "mae_aggregation",
    "pitch_values_deg",
    "alpha_step_deg",
    "yaw_step_deg",
    "side_exclusion_deg",
)


def float_grid(
    start: float,
    stop: float,
    step: float,
    *,
    include_stop: bool,
) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    values: List[float] = []
    index = 0
    tolerance = 1e-10 * max(1.0, abs(start), abs(stop))
    while True:
        value = start + index * step
        if include_stop:
            if value > stop + tolerance:
                break
        elif value >= stop - tolerance:
            break
        values.append(round(value, 12))
        index += 1
    return values


def signed_periodic_error(
    estimate_deg: float,
    truth_deg: float,
    period_deg: float,
) -> float:
    return (estimate_deg - truth_deg + period_deg / 2.0) % period_deg - period_deg / 2.0


def project_deformed_blade_centerlines(
    blade_rotation_deg: float,
    relative_yaw_deg: float,
    camera_pitch_deg: float,
    camera_roll_deg: float,
    deformation_deg_by_blade: Mapping[int, float],
) -> Dict[str, object]:
    """Project flapwise-tilted 3-D blade directions to unit 2-D directions.

    In the rotor frame the rigid blade is ``(sin(q_i), 0, cos(q_i))`` and
    the rotor axis is ``(0, 1, 0)``.  Positive and negative ``delta_i``
    represent an equivalent signed deviation of the visible proximal chord
    along that axis.  Subsequent yaw, pitch, and roll transforms exactly
    follow ``project_blade_centerlines``.
    """

    required_ids = (1, 2, 3)
    if set(deformation_deg_by_blade) != set(required_ids):
        raise ValueError("deformation_deg_by_blade must contain blades 1, 2, and 3")

    alpha = math.radians(blade_rotation_deg)
    gamma = math.radians(relative_yaw_deg)
    pitch = math.radians(camera_pitch_deg)
    roll = math.radians(camera_roll_deg)
    cg, sg = math.cos(gamma), math.sin(gamma)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    camera_vectors_3d: Dict[int, Tuple[float, float, float]] = {}
    unit_pairs: Dict[int, Tuple[float, float]] = {}
    projection_norms: Dict[int, float] = {}

    for blade_id in required_ids:
        q_value = alpha + math.radians(BETAS_DEG[blade_id])
        delta = math.radians(float(deformation_deg_by_blade[blade_id]))

        # Equivalent proximal blade direction in the rotor frame.
        local_x = math.cos(delta) * math.sin(q_value)
        local_y = math.sin(delta)
        local_z = math.cos(delta) * math.cos(q_value)

        # Rz(gamma), Rx(-pitch), and Ry(-roll), matching the paper model.
        yaw_x = local_x * cg - local_y * sg
        yaw_y = local_x * sg + local_y * cg
        yaw_z = local_z

        pitch_x = yaw_x
        pitch_y = yaw_y * cp + yaw_z * sp
        pitch_z = -yaw_y * sp + yaw_z * cp

        camera_x = pitch_x * cr - pitch_z * sr
        camera_y = pitch_y
        camera_z = pitch_x * sr + pitch_z * cr

        vector_norm = math.sqrt(
            camera_x * camera_x + camera_y * camera_y + camera_z * camera_z
        )
        if abs(vector_norm - 1.0) > 1e-10:
            raise RuntimeError(
                f"deformed blade {blade_id} is not unit length: {vector_norm}"
            )

        projection_norm = math.hypot(camera_x, camera_z)
        if projection_norm <= EPS:
            raise ValueError(f"deformed blade {blade_id} has a degenerate projection")

        camera_vectors_3d[blade_id] = (camera_x, camera_y, camera_z)
        projection_norms[blade_id] = projection_norm
        unit_pairs[blade_id] = (
            camera_x / projection_norm,
            camera_z / projection_norm,
        )

    return {
        "camera_vectors_3d": camera_vectors_3d,
        "unit_pairs": unit_pairs,
        "projection_norms": projection_norms,
        "min_projection_norm": min(projection_norms.values()),
    }


def verify_zero_deformation_projection() -> None:
    """Guard against a coordinate/sign mismatch with the existing generator."""

    checks = (
        (0.0, 0.0, 0.0, 0.0),
        (20.0, 35.0, 15.0, -8.0),
        (75.0, 145.0, -45.0, 6.0),
    )
    zero = {1: 0.0, 2: 0.0, 3: 0.0}
    for alpha, yaw, pitch, roll in checks:
        expected = project_blade_centerlines(
            blade_rotation_deg=alpha,
            relative_yaw_deg=yaw,
            camera_pitch_deg=pitch,
            camera_roll_deg=roll,
        )["unit_pairs"]
        actual = project_deformed_blade_centerlines(
            alpha,
            yaw,
            pitch,
            roll,
            zero,
        )["unit_pairs"]
        for blade_id in (1, 2, 3):
            for expected_value, actual_value in zip(
                expected[blade_id], actual[blade_id]
            ):
                if not math.isclose(
                    float(expected_value),
                    float(actual_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        "zero-deformation projection does not match the existing generator"
                    )


def build_pose_grid(
    alpha_step_deg: float,
    yaw_step_deg: float,
    side_exclusion_deg: float,
) -> List[Dict[str, float]]:
    alpha_values = float_grid(0.0, 120.0, alpha_step_deg, include_stop=False)
    full_yaws = float_grid(0.0, 180.0, yaw_step_deg, include_stop=True)
    yaw_values = [
        yaw
        for yaw in full_yaws
        if abs(yaw - 90.0) > side_exclusion_deg + 1e-12
    ]
    if not alpha_values or not yaw_values:
        raise ValueError("the configured pose grid is empty")
    return [
        {"alpha_deg": alpha, "yaw_deg": yaw}
        for alpha in alpha_values
        for yaw in yaw_values
    ]


def run_trial(
    pose: Mapping[str, float],
    pitch_deg: float,
    deformation_magnitude_deg: float,
    group_seed: int,
    case_index: int,
    repeat_id: int,
    *,
    refine: bool,
    max_refine_seeds: int,
) -> Dict[str, object]:
    true_alpha = float(pose["alpha_deg"])
    true_yaw = float(pose["yaw_deg"])
    # A common positive rotor-axis tilt represents a coherent downwind
    # equivalent deflection of the three visible proximal blade chords.
    deformation = {
        blade_id: deformation_magnitude_deg for blade_id in (1, 2, 3)
    }

    row: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "experiment": EXPERIMENT,
        "perturbation_domain": "3D rotor-axis tilt before projection",
        "deformation_model": "deterministic common rotor-axis tilt",
        "deformation_magnitude_deg": deformation_magnitude_deg,
        "deformation_sign": "positive rotor-axis direction",
        "group_seed": group_seed,
        "case_index": case_index,
        "repeat_id": repeat_id,
        "blade_rotation_true_deg": true_alpha,
        "relative_yaw_true_deg": true_yaw,
        "orientation_class": orientation_class_from_relative_yaw(true_yaw),
        "camera_pitch_true_deg": pitch_deg,
        "camera_roll_true_deg": TRUE_ROLL_DEG,
        "deformation_blade1_deg": deformation[1],
        "deformation_blade2_deg": deformation[2],
        "deformation_blade3_deg": deformation[3],
        "min_projection_norm": "",
        "success": False,
        "blade_rotation_est_deg": "",
        "relative_yaw_est_deg": "",
        "direction_sse": "",
        "blade_signed_error_deg": "",
        "blade_abs_error_deg": "",
        "yaw_signed_error_deg": "",
        "yaw_abs_error_deg": "",
        "error_message": "",
    }

    try:
        forward = project_deformed_blade_centerlines(
            true_alpha,
            true_yaw,
            pitch_deg,
            TRUE_ROLL_DEG,
            deformation,
        )
        row["min_projection_norm"] = forward["min_projection_norm"]
        inverse = solve_pose_from_unit_centerlines(
            forward["unit_pairs"],
            row["orientation_class"],
            camera_pitch_deg=pitch_deg,
            camera_roll_deg=TRUE_ROLL_DEG,
            refine=refine,
            max_refine_seeds=max_refine_seeds,
        )
        best = inverse["best"]
        alpha_est = float(best["alpha_deg_120"])
        yaw_est = float(best["relative_yaw_deg_360"])
        direction_sse = float(best["direction_sse"])
        if not all(math.isfinite(v) for v in (alpha_est, yaw_est, direction_sse)):
            raise RuntimeError("the solver returned a non-finite value")

        blade_signed = signed_periodic_error(alpha_est, true_alpha, 120.0)
        yaw_signed = signed_periodic_error(yaw_est, true_yaw, 360.0)
        row.update(
            {
                "success": True,
                "blade_rotation_est_deg": alpha_est,
                "relative_yaw_est_deg": yaw_est,
                "direction_sse": direction_sse,
                "blade_signed_error_deg": blade_signed,
                "blade_abs_error_deg": abs(blade_signed),
                "yaw_signed_error_deg": yaw_signed,
                "yaw_abs_error_deg": abs(yaw_signed),
            }
        )
    except Exception as exc:
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    deformation_magnitude_deg: float,
    pitch_deg: float,
    alpha_step_deg: float,
    yaw_step_deg: float,
    side_exclusion_deg: float,
    base_seed: int,
    group_seed: int,
    configured_repeats: int,
    effective_repeats: int,
    refine: bool,
    max_refine_seeds: int,
) -> Dict[str, object]:
    successes = [row for row in rows if row["success"]]
    n_total = len(rows)
    n_success = len(successes)
    if successes:
        yaw_mae = statistics.fmean(
            float(row["yaw_abs_error_deg"]) for row in successes
        )
        blade_mae = statistics.fmean(
            float(row["blade_abs_error_deg"]) for row in successes
        )
    else:
        yaw_mae = math.nan
        blade_mae = math.nan
    return {
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "experiment": EXPERIMENT,
        "perturbation_domain": "3D rotor-axis tilt before projection",
        "deformation_model": "deterministic common rotor-axis tilt",
        "deformation_magnitude_deg": deformation_magnitude_deg,
        "deformation_sign": "positive rotor-axis direction",
        "camera_pitch_true_deg": pitch_deg,
        "camera_roll_true_deg": TRUE_ROLL_DEG,
        "n_total": n_total,
        "n_success": n_success,
        "n_failure": n_total - n_success,
        "failure_rate": (n_total - n_success) / n_total if n_total else math.nan,
        "yaw_mae_deg": yaw_mae,
        "blade_mae_deg": blade_mae,
        "alpha_step_deg": alpha_step_deg,
        "yaw_step_deg": yaw_step_deg,
        "side_exclusion_deg": side_exclusion_deg,
        "base_seed": base_seed,
        "group_seed": group_seed,
        "configured_repeats": configured_repeats,
        "effective_repeats": effective_repeats,
        "refinement_enabled": refine,
        "max_refine_seeds": max_refine_seeds,
    }


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def aggregate_pitch_summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    source: str,
    source_label: str,
    perturbation_domain: str,
    perturbation_model: str,
    level_definition: str,
    level_key: str,
    level: float,
    pitch_values_deg: Sequence[float],
    alpha_step_deg: float,
    yaw_step_deg: float,
    side_exclusion_deg: float,
) -> Dict[str, object]:
    selected = [row for row in rows if float(row[level_key]) == level]
    if len(selected) != len(pitch_values_deg):
        raise RuntimeError(
            f"expected {len(pitch_values_deg)} pitch summaries for {source} at {level}, "
            f"found {len(selected)}"
        )
    expected_pitches = sorted(float(value) for value in pitch_values_deg)
    actual_pitches = sorted(float(row["camera_pitch_true_deg"]) for row in selected)
    if actual_pitches != expected_pitches:
        raise RuntimeError(
            f"unexpected pitch summaries for {source} at {level}: "
            f"expected {expected_pitches}, found {actual_pitches}"
        )
    expected_grid = {
        "alpha_step_deg": alpha_step_deg,
        "yaw_step_deg": yaw_step_deg,
        "side_exclusion_deg": side_exclusion_deg,
    }
    for row in selected:
        for field, expected in expected_grid.items():
            actual = float(row[field])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"incompatible {field} for {source} at {level}: "
                    f"expected {expected}, found {actual}"
                )
    yaw_max_row = max(selected, key=lambda row: float(row["yaw_mae_deg"]))
    blade_max_row = max(selected, key=lambda row: float(row["blade_mae_deg"]))
    n_total = sum(int(row["n_total"]) for row in selected)
    n_success = sum(int(row["n_success"]) for row in selected)
    n_failure = sum(int(row["n_failure"]) for row in selected)
    max_pitch_failure_rate = max(float(row["failure_rate"]) for row in selected)
    return {
        "source": source,
        "source_label": source_label,
        "perturbation_domain": perturbation_domain,
        "perturbation_model": perturbation_model,
        "level_definition": level_definition,
        "angular_level_deg": level,
        "n_total": n_total,
        "n_success": n_success,
        "n_failure": n_failure,
        "overall_failure_rate": n_failure / n_total if n_total else math.nan,
        "max_pitch_failure_rate": max_pitch_failure_rate,
        "yaw_mae_max_over_pitch_deg": float(yaw_max_row["yaw_mae_deg"]),
        "blade_mae_max_over_pitch_deg": float(blade_max_row["blade_mae_deg"]),
        "pitch_at_yaw_max_deg": float(yaw_max_row["camera_pitch_true_deg"]),
        "pitch_at_blade_max_deg": float(blade_max_row["camera_pitch_true_deg"]),
        "mae_aggregation": "maximum of pitch-wise MAEs; each MAE uses successful estimates",
        "pitch_values_deg": "/".join(f"{value:g}" for value in pitch_values_deg),
        "alpha_step_deg": alpha_step_deg,
        "yaw_step_deg": yaw_step_deg,
        "side_exclusion_deg": side_exclusion_deg,
    }


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-step-deg", type=float, default=5.0)
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)
    parser.add_argument("--side-exclusion-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--max-refine-seeds", type=int, default=4)
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=default_dir / "residual_blade_deformation_trials.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=default_dir / "residual_blade_deformation_summary.csv",
    )
    parser.add_argument(
        "--network-summary-csv",
        type=Path,
        default=default_dir / "unit_centerline_solver_error_summary.csv",
    )
    parser.add_argument(
        "--combined-summary-csv",
        type=Path,
        default=default_dir / "centerline_deviation_source_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.side_exclusion_deg < 0:
        raise ValueError("side_exclusion_deg must be non-negative")
    if args.max_refine_seeds <= 0:
        raise ValueError("max_refine_seeds must be positive")
    if not args.network_summary_csv.exists():
        raise FileNotFoundError(args.network_summary_csv)

    verify_zero_deformation_projection()
    refine = not args.no_refine
    poses = build_pose_grid(
        args.alpha_step_deg,
        args.yaw_step_deg,
        args.side_exclusion_deg,
    )
    print(
        f"poses/group={len(poses)}, pitches={len(TRUE_PITCH_VALUES_DEG)}, "
        f"refine={refine}",
        flush=True,
    )

    all_trials: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for pitch_index, pitch_deg in enumerate(TRUE_PITCH_VALUES_DEG):
        for level_index, deformation_magnitude in enumerate(PERTURBATION_LEVELS_DEG):
            group_seed = args.seed + 4_000_000 + pitch_index * 10_000 + level_index * 100
            effective_repeats = 1
            group_rows: List[Dict[str, object]] = []
            for repeat_id in range(effective_repeats):
                for case_index, pose in enumerate(poses):
                    group_rows.append(
                        run_trial(
                            pose,
                            pitch_deg,
                            deformation_magnitude,
                            group_seed,
                            case_index,
                            repeat_id,
                            refine=refine,
                            max_refine_seeds=args.max_refine_seeds,
                        )
                    )
            all_trials.extend(group_rows)
            summary = summarize_rows(
                group_rows,
                deformation_magnitude_deg=deformation_magnitude,
                pitch_deg=pitch_deg,
                alpha_step_deg=args.alpha_step_deg,
                yaw_step_deg=args.yaw_step_deg,
                side_exclusion_deg=args.side_exclusion_deg,
                base_seed=args.seed,
                group_seed=group_seed,
                configured_repeats=1,
                effective_repeats=effective_repeats,
                refine=refine,
                max_refine_seeds=args.max_refine_seeds,
            )
            summaries.append(summary)
            print(
                f"deformation pitch={pitch_deg:>5.0f} delta={deformation_magnitude:>3.0f} "
                f"yaw_MAE={summary['yaw_mae_deg']:.4f} "
                f"blade_MAE={summary['blade_mae_deg']:.4f} "
                f"fail={summary['failure_rate']:.2%}",
                flush=True,
            )

    write_csv(args.trials_csv, all_trials, RAW_FIELDS)
    write_csv(args.summary_csv, summaries, SUMMARY_FIELDS)

    network_summaries = [
        row
        for row in read_csv(args.network_summary_csv)
        if row["experiment"] == "centerline_error"
    ]
    combined: List[Dict[str, object]] = []
    for level in PERTURBATION_LEVELS_DEG:
        combined.append(
            aggregate_pitch_summaries(
                network_summaries,
                source="network_extraction_2d",
                source_label="Network centerline extraction",
                perturbation_domain="2D angular deviation after projection",
                perturbation_model="independent zero-mean Gaussian 2D angular noise",
                level_definition="standard deviation",
                level_key="noise_std_deg",
                level=level,
                pitch_values_deg=TRUE_PITCH_VALUES_DEG,
                alpha_step_deg=args.alpha_step_deg,
                yaw_step_deg=args.yaw_step_deg,
                side_exclusion_deg=args.side_exclusion_deg,
            )
        )
    for level in PERTURBATION_LEVELS_DEG:
        combined.append(
            aggregate_pitch_summaries(
                summaries,
                source="residual_deformation_3d",
                source_label="Residual proximal-blade deformation",
                perturbation_domain="3D rotor-axis tilt before projection",
                perturbation_model="deterministic common 3D rotor-axis tilt",
                level_definition="equivalent deflection magnitude",
                level_key="deformation_magnitude_deg",
                level=level,
                pitch_values_deg=TRUE_PITCH_VALUES_DEG,
                alpha_step_deg=args.alpha_step_deg,
                yaw_step_deg=args.yaw_step_deg,
                side_exclusion_deg=args.side_exclusion_deg,
            )
        )
    write_csv(args.combined_summary_csv, combined, COMBINED_FIELDS)

    print(f"trials_csv={args.trials_csv}", flush=True)
    print(f"summary_csv={args.summary_csv}", flush=True)
    print(f"combined_summary_csv={args.combined_summary_csv}", flush=True)


if __name__ == "__main__":
    main()

