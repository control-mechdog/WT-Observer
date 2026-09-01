# -*- coding: utf-8 -*-
"""Generate the near-side-view sensitivity figure used by main.tex.

The experiment follows the independently normalized, hub-to-tip centerline
model implemented by ``geometric_solver/wt_pose_from_unit_centerlines.py``. It adds
independent Gaussian angular noise to the three 2-D blade directions and
evaluates pose accuracy as a function of

    d_side = |relative_yaw - 90 deg|.

Both sides of 90 deg are sampled.  Common random numbers are reused across
all distances and the two branches so that changes along a curve are caused
primarily by geometric conditioning rather than Monte-Carlo fluctuations.

Outputs (in ``evaluation/results`` by default):

* side-view-sensitivity-summary.csv
* side-view-sensitivity.pdf
* side-view-sensitivity.png

The exact side view (d_side=0) is not assigned an arbitrary alpha error: the
unit-centerline model is rank deficient there and alpha is unobservable.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


EVALUATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALUATION_DIR / "results"

try:
    from ..wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
    from ..wt_pose_to_blade_centerlines import project_blade_centerlines
except ImportError:
    SOLVER_DIR = EVALUATION_DIR.parent
    if str(SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(SOLVER_DIR))
    from wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
    from wt_pose_to_blade_centerlines import project_blade_centerlines

DISTANCES_DEG = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
TRUE_PITCH_DEG = (-45.0, 0.0, 45.0)
TRUE_ALPHA_DEG = tuple(float(value) for value in range(0, 120, 10))
TRUE_ROLL_DEG = 0.0
BLADE_IDS = (1, 2, 3)

SUMMARY_FIELDS = (
    "distance_to_side_deg",
    "camera_pitch_true_deg",
    "centerline_noise_std_deg",
    "repeats",
    "seed",
    "n_total",
    "n_success",
    "n_failure",
    "failure_rate",
    "yaw_mae_deg_successful",
    "yaw_rmse_deg_successful",
    "yaw_p95_abs_error_deg_successful",
    "alpha_mae_deg_successful",
    "alpha_rmse_deg_successful",
    "alpha_p95_abs_error_deg_successful",
    "alpha_catastrophic_rate_overall",
    "min_clean_projection_norm",
)


def periodic_error(estimate_deg: float, truth_deg: float, period_deg: float) -> float:
    return (estimate_deg - truth_deg + period_deg / 2.0) % period_deg - period_deg / 2.0


def rotate_direction(
    pair: Tuple[float, float],
    noise_deg: float,
) -> Tuple[float, float]:
    x_value, z_value = pair
    angle = math.atan2(x_value, z_value) + math.radians(noise_deg)
    return math.sin(angle), math.cos(angle)


def root_mean_square(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_common_noise(
    repeats: int,
    seed: int,
) -> Dict[Tuple[float, float, int], Dict[int, float]]:
    """Standard-normal samples shared by every distance and yaw branch."""
    rng = random.Random(seed)
    noise: Dict[Tuple[float, float, int], Dict[int, float]] = {}
    for pitch_deg in TRUE_PITCH_DEG:
        for alpha_deg in TRUE_ALPHA_DEG:
            for repeat_id in range(repeats):
                noise[(pitch_deg, alpha_deg, repeat_id)] = {
                    blade_id: rng.gauss(0.0, 1.0) for blade_id in BLADE_IDS
                }
    return noise


def run_experiment(
    *,
    repeats: int,
    noise_std_deg: float,
    seed: int,
) -> List[Dict[str, object]]:
    common_noise = build_common_noise(repeats, seed)
    summaries: List[Dict[str, object]] = []

    for pitch_deg in TRUE_PITCH_DEG:
        for distance_deg in DISTANCES_DEG:
            yaw_errors: List[float] = []
            alpha_errors: List[float] = []
            n_total = 0
            n_failure = 0
            n_catastrophic = 0
            minimum_projection_norm = math.inf

            for side_sign in (-1.0, 1.0):
                yaw_deg = 90.0 + side_sign * distance_deg
                for alpha_deg in TRUE_ALPHA_DEG:
                    forward = project_blade_centerlines(
                        blade_rotation_deg=alpha_deg,
                        relative_yaw_deg=yaw_deg,
                        camera_pitch_deg=pitch_deg,
                        camera_roll_deg=TRUE_ROLL_DEG,
                        blade_ids=BLADE_IDS,
                    )
                    minimum_projection_norm = min(
                        minimum_projection_norm,
                        min(forward["projection_norms"].values()),
                    )
                    clean_pairs = dict(forward["unit_pairs"])

                    for repeat_id in range(repeats):
                        n_total += 1
                        standard_noise = common_noise[(pitch_deg, alpha_deg, repeat_id)]
                        observed_pairs = {
                            blade_id: rotate_direction(
                                clean_pairs[blade_id],
                                noise_std_deg * standard_noise[blade_id],
                            )
                            for blade_id in BLADE_IDS
                        }
                        try:
                            result = solve_pose_from_unit_centerlines(
                                observed_pairs,
                                str(forward["orientation_class"]),
                                camera_pitch_deg=pitch_deg,
                                camera_roll_deg=TRUE_ROLL_DEG,
                                refine=True,
                                max_refine_seeds=4,
                            )
                            best = result["best"]
                            yaw_error = abs(
                                periodic_error(
                                    float(best["relative_yaw_deg_360"]),
                                    yaw_deg,
                                    360.0,
                                )
                            )
                            alpha_error = abs(
                                periodic_error(
                                    float(best["alpha_deg_120"]),
                                    alpha_deg,
                                    120.0,
                                )
                            )
                            if not (
                                math.isfinite(yaw_error)
                                and math.isfinite(alpha_error)
                            ):
                                raise RuntimeError("non-finite pose error")
                            yaw_errors.append(yaw_error)
                            alpha_errors.append(alpha_error)
                            if alpha_error > 10.0:
                                n_catastrophic += 1
                        except Exception:
                            n_failure += 1

            n_success = n_total - n_failure
            summaries.append(
                {
                    "distance_to_side_deg": distance_deg,
                    "camera_pitch_true_deg": pitch_deg,
                    "centerline_noise_std_deg": noise_std_deg,
                    "repeats": repeats,
                    "seed": seed,
                    "n_total": n_total,
                    "n_success": n_success,
                    "n_failure": n_failure,
                    "failure_rate": n_failure / n_total,
                    "yaw_mae_deg_successful": (
                        sum(yaw_errors) / len(yaw_errors) if yaw_errors else math.nan
                    ),
                    "yaw_rmse_deg_successful": (
                        root_mean_square(yaw_errors) if yaw_errors else math.nan
                    ),
                    "yaw_p95_abs_error_deg_successful": percentile(yaw_errors, 0.95),
                    "alpha_mae_deg_successful": (
                        sum(alpha_errors) / len(alpha_errors)
                        if alpha_errors
                        else math.nan
                    ),
                    "alpha_rmse_deg_successful": (
                        root_mean_square(alpha_errors) if alpha_errors else math.nan
                    ),
                    "alpha_p95_abs_error_deg_successful": percentile(alpha_errors, 0.95),
                    "alpha_catastrophic_rate_overall": (
                        (n_catastrophic + n_failure) / n_total
                    ),
                    "min_clean_projection_norm": minimum_projection_norm,
                }
            )
            latest = summaries[-1]
            print(
                f"pitch={pitch_deg:>5.0f}, d_side={distance_deg:>5.2f}, "
                f"yaw_MAE={latest['yaw_mae_deg_successful']:.3f}, "
                f"alpha_MAE={latest['alpha_mae_deg_successful']:.3f}, "
                f"failure={latest['failure_rate']:.2%}",
                flush=True,
            )
    return summaries


def write_summary_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_summary_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for source in csv.DictReader(csv_file):
            row: Dict[str, object] = dict(source)
            for key in SUMMARY_FIELDS:
                if key in {
                    "repeats",
                    "seed",
                    "n_total",
                    "n_success",
                    "n_failure",
                }:
                    row[key] = int(float(source[key]))
                else:
                    row[key] = float(source[key])
            rows.append(row)
    return rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "black",
            "lines.linewidth": 1.1,
            "lines.markersize": 4.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    configure_plot_style()
    styles = {
        -45.0: ("#0000FF", "o", r"$\theta=-45^\circ$"),
        0.0: ("#D55E00", "s", r"$\theta=0^\circ$"),
        45.0: ("#009E73", "^", r"$\theta=45^\circ$"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.35), constrained_layout=True)
    metric_specs = (
        ("yaw_mae_deg_successful", "Relative yaw MAE", "(a) Relative yaw"),
        ("alpha_mae_deg_successful", "Blade rotation MAE", "(b) Blade rotation"),
        ("failure_rate", "Solver failure rate", "(c) Solver failure"),
    )

    for axis, (metric, ylabel, title) in zip(axes, metric_specs):
        for pitch_deg in TRUE_PITCH_DEG:
            subset = sorted(
                (
                    row
                    for row in rows
                    if float(row["camera_pitch_true_deg"]) == pitch_deg
                ),
                key=lambda row: float(row["distance_to_side_deg"]),
            )
            x_values = [float(row["distance_to_side_deg"]) for row in subset]
            scale = 100.0 if metric == "failure_rate" else 1.0
            y_values = [scale * float(row[metric]) for row in subset]
            color, marker, label = styles[pitch_deg]
            axis.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.7,
                label=label,
            )

        axis.set_xscale("log")
        axis.set_xlim(0.22, 33.0)
        axis.set_xticks((0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0))
        axis.get_xaxis().set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}°")
        )
        if metric == "failure_rate":
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}%")
            )
        else:
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}°")
            )
        axis.set_xlabel(r"Distance to side view $d_{\mathrm{side}}$")
        axis.set_ylabel(ylabel)
        axis.text(
            0.5,
            -0.28,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=3.0,
            width=0.6,
            top=True,
            right=True,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            direction="in",
            length=1.8,
            width=0.45,
            top=True,
            right=True,
        )
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.75)

    axes[0].legend(loc="best", frameon=False, handlelength=1.7)
    axes[2].set_ylim(bottom=0.0)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--noise-std-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="Skip Monte Carlo evaluation and redraw from an existing summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.noise_std_deg < 0:
        raise ValueError("noise-std-deg cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "side-view-sensitivity-summary.csv"
    pdf_path = args.output_dir / "side-view-sensitivity.pdf"
    png_path = args.output_dir / "side-view-sensitivity.png"

    if args.reuse_summary:
        rows = read_summary_csv(summary_path)
    else:
        rows = run_experiment(
            repeats=args.repeats,
            noise_std_deg=args.noise_std_deg,
            seed=args.seed,
        )
        write_summary_csv(summary_path, rows)
    plot_summary(rows, pdf_path=pdf_path, png_path=png_path)
    print(f"summary_csv={summary_path}")
    print(f"figure_pdf={pdf_path}")
    print(f"figure_png={png_path}")


if __name__ == "__main__":
    main()
