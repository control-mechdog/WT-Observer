# -*- coding: utf-8 -*-
"""Reproduce the geometric-solver sensitivity tables reported in the paper.

The script evaluates one uncertainty source at a time: independent 2-D blade
centerline angular noise, camera-pitch measurement noise, and camera-roll
measurement noise. The default pose grid matches the revised manuscript:
blade rotation in [0, 120) degrees, relative yaw in [0, 180] degrees, camera
pitch in {-45, 0, 45} degrees, and noise standard deviations in {0, 1, 2, 4}
degrees. The exact-side-view neighborhood |yaw - 90| <= 5 degrees is excluded
from these aggregate tables and evaluated separately.

Every blade centerline is an independently normalized, directed hub-to-tip
image-plane vector; no relative projected blade lengths are used.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from ..wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
    from ..wt_pose_to_blade_centerlines import project_blade_centerlines
except ImportError:
    import sys

    SOLVER_DIR = Path(__file__).resolve().parents[1]
    if str(SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(SOLVER_DIR))
    from wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
    from wt_pose_to_blade_centerlines import project_blade_centerlines


SCHEMA_VERSION = "2.0"
SOLVER_VERSION = "unit_centerline_quartic_v1"
OBSERVATION_MODEL = "independent_unit_directions"
OBJECTIVE = "paper_cross_product_sse"

EXPERIMENTS = (
    "centerline_error",
    "camera_pitch_error",
    "camera_roll_error",
)
TRUE_PITCH_VALUES = (-45.0, 0.0, 45.0)
NOISE_STD_VALUES = (0.0, 1.0, 2.0, 4.0)
TRUE_ROLL_DEG = 0.0

EXPERIMENT_LABELS = {
    "centerline_error": "Blade centerline angular noise",
    "camera_pitch_error": "Camera pitch measurement noise",
    "camera_roll_error": "Camera roll measurement noise",
}

RAW_FIELDS = (
    "schema_version",
    "solver_version",
    "observation_model",
    "objective",
    "centerline_direction_convention",
    "experiment",
    "experiment_label",
    "noise_std_deg",
    "group_seed",
    "case_index",
    "repeat_id",
    "blade_rotation_true_deg",
    "relative_yaw_true_deg",
    "orientation_class",
    "camera_pitch_true_deg",
    "camera_roll_true_deg",
    "camera_pitch_used_deg",
    "camera_roll_used_deg",
    "pitch_noise_deg",
    "roll_noise_deg",
    "centerline_noise_blade1_deg",
    "centerline_noise_blade2_deg",
    "centerline_noise_blade3_deg",
    "min_clean_projection_norm",
    "success",
    "blade_rotation_est_deg",
    "relative_yaw_est_deg",
    "direction_sse",
    "max_abs_collinearity_residual",
    "min_directed_scale",
    "n_candidates",
    "solver_source",
    "blade_signed_error_deg",
    "blade_abs_error_deg",
    "yaw_signed_error_deg",
    "yaw_abs_error_deg",
    "error_message",
)

SUMMARY_FIELDS = (
    "schema_version",
    "solver_version",
    "observation_model",
    "objective",
    "centerline_direction_convention",
    "experiment",
    "experiment_label",
    "noise_std_deg",
    "camera_pitch_true_deg",
    "camera_roll_true_deg",
    "n_total",
    "n_success",
    "n_failure",
    "failure_rate",
    "yaw_mae_deg",
    "yaw_rmse_deg",
    "yaw_bias_deg",
    "yaw_error_std_deg",
    "yaw_median_abs_error_deg",
    "yaw_p95_abs_error_deg",
    "yaw_max_abs_error_deg",
    "blade_mae_deg",
    "blade_rmse_deg",
    "blade_bias_deg",
    "blade_error_std_deg",
    "blade_median_abs_error_deg",
    "blade_p95_abs_error_deg",
    "blade_max_abs_error_deg",
    "mean_direction_sse",
    "mean_max_abs_collinearity_residual",
    "mean_min_directed_scale",
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


def float_grid(start: float, stop: float, step: float, include_stop: bool) -> List[float]:
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


def signed_periodic_error(estimate_deg: float, truth_deg: float, period_deg: float) -> float:
    return (estimate_deg - truth_deg + period_deg / 2.0) % period_deg - period_deg / 2.0


def rotate_2d_direction(
    pair: Tuple[float, float],
    noise_deg: float,
) -> Tuple[float, float]:
    x_value, z_value = pair
    norm = math.hypot(x_value, z_value)
    if norm <= 1e-12:
        raise ValueError("the 2-D centerline direction is degenerate")
    angle = math.atan2(x_value, z_value) + math.radians(noise_deg)
    return math.sin(angle), math.cos(angle)


def build_base_samples(
    alpha_values: Sequence[float],
    yaw_values: Sequence[float],
    pitch_values: Sequence[float],
) -> Dict[float, List[Dict[str, object]]]:
    samples_by_pitch: Dict[float, List[Dict[str, object]]] = {}
    for pitch_deg in pitch_values:
        samples: List[Dict[str, object]] = []
        for alpha_deg in alpha_values:
            for yaw_deg in yaw_values:
                forward = project_blade_centerlines(
                    blade_rotation_deg=alpha_deg,
                    relative_yaw_deg=yaw_deg,
                    camera_pitch_deg=pitch_deg,
                    camera_roll_deg=TRUE_ROLL_DEG,
                    blade_ids=(1, 2, 3),
                )
                unit_pairs = dict(forward["unit_pairs"])
                if len(unit_pairs) != 3:
                    raise RuntimeError(
                        f"pose alpha={alpha_deg}, yaw={yaw_deg}, pitch={pitch_deg} "
                        "contains a blade projection that cannot be normalized"
                    )
                samples.append(
                    {
                        "alpha_deg": alpha_deg,
                        "yaw_deg": yaw_deg,
                        "pitch_deg": pitch_deg,
                        "roll_deg": TRUE_ROLL_DEG,
                        "orientation_class": forward["orientation_class"],
                        "unit_pairs": unit_pairs,
                        "min_projection_norm": min(forward["projection_norms"].values()),
                    }
                )
        samples_by_pitch[pitch_deg] = samples
    return samples_by_pitch


def run_one_trial(
    sample: Dict[str, object],
    experiment: str,
    noise_std_deg: float,
    rng: random.Random,
    group_seed: int,
    case_index: int,
    repeat_id: int,
    *,
    refine: bool,
    max_refine_seeds: int,
) -> Dict[str, object]:
    true_alpha = float(sample["alpha_deg"])
    true_yaw = float(sample["yaw_deg"])
    true_pitch = float(sample["pitch_deg"])
    true_roll = float(sample["roll_deg"])
    orientation_class = str(sample["orientation_class"])
    clean_pairs = dict(sample["unit_pairs"])

    pitch_noise = 0.0
    roll_noise = 0.0
    centerline_noise = {1: 0.0, 2: 0.0, 3: 0.0}
    measured_pitch = true_pitch
    measured_roll = true_roll
    observed_pairs = clean_pairs

    if experiment == "centerline_error":
        observed_pairs = {}
        for blade_id, pair in clean_pairs.items():
            noise = rng.gauss(0.0, noise_std_deg)
            centerline_noise[blade_id] = noise
            observed_pairs[blade_id] = rotate_2d_direction(pair, noise)
    elif experiment == "camera_pitch_error":
        pitch_noise = rng.gauss(0.0, noise_std_deg)
        measured_pitch = true_pitch + pitch_noise
    elif experiment == "camera_roll_error":
        roll_noise = rng.gauss(0.0, noise_std_deg)
        measured_roll = true_roll + roll_noise
    else:
        raise ValueError(f"unknown experiment: {experiment}")

    row: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "observation_model": OBSERVATION_MODEL,
        "objective": OBJECTIVE,
        "centerline_direction_convention": "hub_to_tip",
        "experiment": experiment,
        "experiment_label": EXPERIMENT_LABELS[experiment],
        "noise_std_deg": noise_std_deg,
        "group_seed": group_seed,
        "case_index": case_index,
        "repeat_id": repeat_id,
        "blade_rotation_true_deg": true_alpha,
        "relative_yaw_true_deg": true_yaw,
        "orientation_class": orientation_class,
        "camera_pitch_true_deg": true_pitch,
        "camera_roll_true_deg": true_roll,
        "camera_pitch_used_deg": measured_pitch,
        "camera_roll_used_deg": measured_roll,
        "pitch_noise_deg": pitch_noise,
        "roll_noise_deg": roll_noise,
        "centerline_noise_blade1_deg": centerline_noise[1],
        "centerline_noise_blade2_deg": centerline_noise[2],
        "centerline_noise_blade3_deg": centerline_noise[3],
        "min_clean_projection_norm": sample["min_projection_norm"],
        "success": False,
        "blade_rotation_est_deg": "",
        "relative_yaw_est_deg": "",
        "direction_sse": "",
        "max_abs_collinearity_residual": "",
        "min_directed_scale": "",
        "n_candidates": "",
        "solver_source": "",
        "blade_signed_error_deg": "",
        "blade_abs_error_deg": "",
        "yaw_signed_error_deg": "",
        "yaw_abs_error_deg": "",
        "error_message": "",
    }

    try:
        inverse = solve_pose_from_unit_centerlines(
            observed_pairs,
            orientation_class,
            camera_pitch_deg=measured_pitch,
            camera_roll_deg=measured_roll,
            refine=refine,
            max_refine_seeds=max_refine_seeds,
        )
        candidates = inverse["candidates"]
        if not candidates:
            raise RuntimeError("the solver returned no candidate")
        best = inverse["best"]
        alpha_est = float(best["alpha_deg_120"])
        yaw_est = float(best["relative_yaw_deg_360"])
        direction_sse = float(best["direction_sse"])
        max_residual = float(best["max_abs_residual"])
        min_directed_scale = float(best["min_directed_scale"])
        finite_values = (
            alpha_est,
            yaw_est,
            direction_sse,
            max_residual,
            min_directed_scale,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise RuntimeError("the solver returned a non-finite result")
        if min_directed_scale <= 0:
            raise RuntimeError("the result violates the directed hub-to-tip constraint")

        blade_signed_error = signed_periodic_error(alpha_est, true_alpha, 120.0)
        yaw_signed_error = signed_periodic_error(yaw_est, true_yaw, 360.0)
        row.update(
            {
                "success": True,
                "blade_rotation_est_deg": alpha_est,
                "relative_yaw_est_deg": yaw_est,
                "direction_sse": direction_sse,
                "max_abs_collinearity_residual": max_residual,
                "min_directed_scale": min_directed_scale,
                "n_candidates": len(candidates),
                "solver_source": best["source"],
                "blade_signed_error_deg": blade_signed_error,
                "blade_abs_error_deg": abs(blade_signed_error),
                "yaw_signed_error_deg": yaw_signed_error,
                "yaw_abs_error_deg": abs(yaw_signed_error),
            }
        )
    except Exception as exc:  # Retain every failed trial in the CSV.
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def root_mean_square(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_group(
    rows: Sequence[Dict[str, object]],
    *,
    experiment: str,
    noise_std_deg: float,
    true_pitch_deg: float,
    group_seed: int,
    alpha_step_deg: float,
    yaw_step_deg: float,
    side_exclusion_deg: float,
    base_seed: int,
    configured_repeats: int,
    effective_repeats: int,
    refine: bool,
    max_refine_seeds: int,
) -> Dict[str, object]:
    successes = [row for row in rows if row["success"]]
    n_total = len(rows)
    n_success = len(successes)
    n_failure = n_total - n_success
    summary: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "observation_model": OBSERVATION_MODEL,
        "objective": OBJECTIVE,
        "centerline_direction_convention": "hub_to_tip",
        "experiment": experiment,
        "experiment_label": EXPERIMENT_LABELS[experiment],
        "noise_std_deg": noise_std_deg,
        "camera_pitch_true_deg": true_pitch_deg,
        "camera_roll_true_deg": TRUE_ROLL_DEG,
        "n_total": n_total,
        "n_success": n_success,
        "n_failure": n_failure,
        "failure_rate": n_failure / n_total if n_total else math.nan,
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

    metric_names = (
        "yaw_mae_deg",
        "yaw_rmse_deg",
        "yaw_bias_deg",
        "yaw_error_std_deg",
        "yaw_median_abs_error_deg",
        "yaw_p95_abs_error_deg",
        "yaw_max_abs_error_deg",
        "blade_mae_deg",
        "blade_rmse_deg",
        "blade_bias_deg",
        "blade_error_std_deg",
        "blade_median_abs_error_deg",
        "blade_p95_abs_error_deg",
        "blade_max_abs_error_deg",
        "mean_direction_sse",
        "mean_max_abs_collinearity_residual",
        "mean_min_directed_scale",
    )
    if not successes:
        summary.update({name: math.nan for name in metric_names})
        return summary

    yaw_signed = [float(row["yaw_signed_error_deg"]) for row in successes]
    yaw_abs = [abs(value) for value in yaw_signed]
    blade_signed = [float(row["blade_signed_error_deg"]) for row in successes]
    blade_abs = [abs(value) for value in blade_signed]
    sse_values = [float(row["direction_sse"]) for row in successes]
    max_residuals = [
        float(row["max_abs_collinearity_residual"]) for row in successes
    ]
    min_scales = [float(row["min_directed_scale"]) for row in successes]
    summary.update(
        {
            "yaw_mae_deg": statistics.fmean(yaw_abs),
            "yaw_rmse_deg": root_mean_square(yaw_signed),
            "yaw_bias_deg": statistics.fmean(yaw_signed),
            "yaw_error_std_deg": statistics.pstdev(yaw_signed),
            "yaw_median_abs_error_deg": statistics.median(yaw_abs),
            "yaw_p95_abs_error_deg": percentile(yaw_abs, 0.95),
            "yaw_max_abs_error_deg": max(yaw_abs),
            "blade_mae_deg": statistics.fmean(blade_abs),
            "blade_rmse_deg": root_mean_square(blade_signed),
            "blade_bias_deg": statistics.fmean(blade_signed),
            "blade_error_std_deg": statistics.pstdev(blade_signed),
            "blade_median_abs_error_deg": statistics.median(blade_abs),
            "blade_p95_abs_error_deg": percentile(blade_abs, 0.95),
            "blade_max_abs_error_deg": max(blade_abs),
            "mean_direction_sse": statistics.fmean(sse_values),
            "mean_max_abs_collinearity_residual": statistics.fmean(max_residuals),
            "mean_min_directed_scale": statistics.fmean(min_scales),
        }
    )
    return summary


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-step-deg", type=float, default=5.0)
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)
    parser.add_argument("--side-exclusion-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="independent repetitions for each nonzero-noise pose; zero noise runs once",
    )
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--max-refine-seeds", type=int, default=4)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=default_dir / "unit_centerline_solver_error_summary.csv",
    )
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=default_dir / "unit_centerline_solver_error_trials.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.side_exclusion_deg < 0:
        raise ValueError("side_exclusion_deg must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.max_refine_seeds <= 0:
        raise ValueError("max_refine_seeds must be positive")
    refine = not args.no_refine

    alpha_values = float_grid(0.0, 120.0, args.alpha_step_deg, include_stop=False)
    full_yaw_values = float_grid(0.0, 180.0, args.yaw_step_deg, include_stop=True)
    yaw_values = [
        yaw
        for yaw in full_yaw_values
        if abs(yaw - 90.0) > args.side_exclusion_deg + 1e-12
    ]
    if not alpha_values or not yaw_values:
        raise ValueError("the pose grid is empty; check the step sizes and side-view exclusion")

    samples_by_pitch = build_base_samples(alpha_values, yaw_values, TRUE_PITCH_VALUES)
    print(
        f"poses/group={len(alpha_values) * len(yaw_values)}, "
        f"alpha={len(alpha_values)}, yaw={len(yaw_values)}, refine={refine}",
        flush=True,
    )

    all_trials: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    experiment_codes = {name: index + 1 for index, name in enumerate(EXPERIMENTS)}

    for experiment in EXPERIMENTS:
        for pitch_index, true_pitch in enumerate(TRUE_PITCH_VALUES):
            samples = samples_by_pitch[true_pitch]
            for noise_index, noise_std in enumerate(NOISE_STD_VALUES):
                group_seed = (
                    args.seed
                    + experiment_codes[experiment] * 1_000_000
                    + pitch_index * 10_000
                    + noise_index * 100
                )
                rng = random.Random(group_seed)
                effective_repeats = 1 if noise_std == 0.0 else args.repeats
                group_rows: List[Dict[str, object]] = []
                for repeat_id in range(effective_repeats):
                    for case_index, sample in enumerate(samples):
                        group_rows.append(
                            run_one_trial(
                                sample,
                                experiment,
                                noise_std,
                                rng,
                                group_seed,
                                case_index,
                                repeat_id,
                                refine=refine,
                                max_refine_seeds=args.max_refine_seeds,
                            )
                        )
                all_trials.extend(group_rows)
                summary = summarize_group(
                    group_rows,
                    experiment=experiment,
                    noise_std_deg=noise_std,
                    true_pitch_deg=true_pitch,
                    group_seed=group_seed,
                    alpha_step_deg=args.alpha_step_deg,
                    yaw_step_deg=args.yaw_step_deg,
                    side_exclusion_deg=args.side_exclusion_deg,
                    base_seed=args.seed,
                    configured_repeats=args.repeats,
                    effective_repeats=effective_repeats,
                    refine=refine,
                    max_refine_seeds=args.max_refine_seeds,
                )
                summaries.append(summary)
                print(
                    f"{experiment:20s} pitch={true_pitch:>5.0f} "
                    f"sigma={noise_std:>3.0f} "
                    f"yaw_MAE={summary['yaw_mae_deg']:.4f} "
                    f"blade_MAE={summary['blade_mae_deg']:.4f} "
                    f"fail={summary['failure_rate']:.2%}",
                    flush=True,
                )

    write_csv(args.trials_csv, all_trials, RAW_FIELDS)
    write_csv(args.summary_csv, summaries, SUMMARY_FIELDS)
    print(f"summary_csv={args.summary_csv}", flush=True)
    print(f"trials_csv={args.trials_csv}", flush=True)


if __name__ == "__main__":
    main()

