# -*- coding: utf-8 -*-
"""
鏍规嵁閫愬彾鐙珛褰掍竴鍖栫殑鏈夊悜涓績绾挎眰瑙ｉ鏈哄Э鎬併€?
鏈枃浠跺搴?瑙傛祴妯″瀷銆傝緭鍏?``pairs`` 鐨勬瘡涓?``(x, z)`` 閮借〃绀轰竴鏉′粠杞瘋鎸囧悜鍙跺皷鐨勫浘鍍忓钩闈腑蹇冪嚎銆傚嚱鏁颁細瀵规瘡鐗囧彾鐗?鍒嗗埆褰掍竴鍖栵紝鍥犳涓夌墖鍙剁墖涔嬮棿涓嶉渶瑕佸叿鏈夊噯纭殑鐩稿闀垮害銆?
浠?``q_i = alpha + beta_i``銆傚弽鏃嬪凡鐭ョ浉鏈?roll 鍚庯紝璁烘枃妯″瀷涓?
    A_i = sin(q_i) cos(gamma)
    B_i = -sin(q_i) sin(gamma) sin(pitch) + cos(q_i) cos(pitch)

瑙傛祴鍗曚綅鏂瑰悜璁颁负 ``(X_i, Z_i)``锛屽叡绾挎畫宸负

    E_i = A_i Z_i - B_i X_i.

鏈眰瑙ｅ櫒鍏堢敱浠绘剰涓ょ墖鍙剁墖娑堝幓 ``alpha``锛岄€氳繃
``t = tan(gamma/2)`` 寰楀埌鏈€楂樺洓娆＄殑澶氶」寮忥紝鏋氫妇鍏ㄩ儴瀹炴牴锛涘啀鐢ㄥ叏閮ㄥ彲瑙?鍙剁墖鐨勪氦鍙夌Н SSE 瑙ｆ瀽鎭㈠ alpha锛屽苟瀵规渶浼樹唬鏁板€欓€変綔杩炵画楂樻柉--鐗涢】
淇銆傛湞鍚戠被鍒彧鐢ㄤ簬闄愬畾瀹屾暣鐨?gamma 鍒嗘敮銆?
閲嶈绾﹀畾
--------
1. 杈撳叆鏂瑰悜蹇呴』鏄€滆疆姣?-> 鍙跺皷鈥濄€傝绾﹀畾瀵瑰簲姣忓彾鐙珛灏哄害 ``lambda_i>0``銆?2. 鍙剁墖缂栧彿 1/2/3 瀵瑰簲 ``beta={0,120,240}`` 搴︼紝骞朵繚鎸佸惊鐜『搴忥紱寰幆骞崇Щ
   鍙細浣?alpha 鏀瑰彉 120 搴︼紝涓嶅奖鍝嶆渶缁?``alpha_deg_120``銆?3. 杈撳嚭 gamma 鏄浉鏈哄潗鏍囩郴涓殑鐩稿鍋忚埅瑙掞紝涓嶆槸鐪熷寳涓栫晫鑸悜銆?4. 绮剧‘宸﹀彸渚ц鏃讹紝鍗曚綅涓績绾挎ā鍨嬫棤娉曡鲸璇嗚繛缁殑 alpha锛屽嚱鏁颁細鎶涘嚭
   ``PoseDegeneracyError``銆?"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


RAD = math.pi / 180.0
DEG = 180.0 / math.pi
TWO_PI = 2.0 * math.pi
ALPHA_PERIOD = 120.0 * RAD
EPS = 1e-12

BETAS = {
    1: 0.0,
    2: 120.0 * RAD,
    3: 240.0 * RAD,
}

ORIENTATION_LABELS = {
    "back": "椋庡姏鍙戠數鏈鸿儗闈㈡瀵规棤浜烘満",
    "back_right": "椋庡姏鍙戠數鏈鸿儗瀵规棤浜烘満鍋忓悜鍙充晶",
    "right": "椋庡姏鍙戠數鏈烘湞鍚戝彸渚?,
    "front_right": "椋庡姏鍙戠數鏈烘瀵规棤浜烘満鍋忓悜鍙充晶",
    "front": "椋庡姏鍙戠數鏈烘闈㈡瀵规棤浜烘満",
    "front_left": "椋庡姏鍙戠數鏈烘瀵规棤浜烘満鍋忓悜宸︿晶",
    "left": "椋庡姏鍙戠數鏈烘湞鍚戝乏渚?,
    "back_left": "椋庡姏鍙戠數鏈鸿儗瀵规棤浜烘満鍋忓悜宸︿晶",
}

ORIENTATION_INTERVALS_DEG = {
    "back": (0.0, 0.0),
    "back_right": (0.0, 90.0),
    "right": (90.0, 90.0),
    "front_right": (90.0, 180.0),
    "front": (180.0, 180.0),
    "front_left": (180.0, 270.0),
    "left": (270.0, 270.0),
    "back_left": (270.0, 360.0),
}

ORIENTATION_ALIASES = {
    "back": "back",
    "鑳岄潰": "back",
    "鑳屽": "back",
    "back_right": "back_right",
    "鑳屽鍋忓彸": "back_right",
    "right": "right",
    "鍙充晶": "right",
    "鏈濆彸": "right",
    "front_right": "front_right",
    "姝ｅ鍋忓彸": "front_right",
    "front": "front",
    "姝ｉ潰": "front",
    "姝ｅ": "front",
    "front_left": "front_left",
    "姝ｅ鍋忓乏": "front_left",
    "left": "left",
    "宸︿晶": "left",
    "鏈濆乏": "left",
    "back_left": "back_left",
    "鑳屽鍋忓乏": "back_left",
}


class PoseDegeneracyError(RuntimeError):
    """褰撳墠瑙傛祴鍑犱綍涓嶈兘鍞竴杈ㄨ瘑鎵€璇锋眰鐨勫Э鎬併€?""


def normalize_angle_360(angle_deg: float) -> float:
    return angle_deg % 360.0


def normalize_alpha_120(angle_deg: float) -> float:
    value = angle_deg % 120.0
    return 0.0 if abs(value - 120.0) < 1e-10 else value


def _wrap_radians_2pi(angle: float) -> float:
    value = angle % TWO_PI
    return 0.0 if abs(value - TWO_PI) < 1e-12 else value


def canonical_orientation(orientation_class: str) -> str:
    if not isinstance(orientation_class, str):
        raise TypeError("orientation_class 蹇呴』鏄瓧绗︿覆")
    key = orientation_class.strip().lower().replace("-", "_").replace(" ", "_")
    orientation = ORIENTATION_ALIASES.get(key)
    if orientation is None:
        valid = ", ".join(ORIENTATION_INTERVALS_DEG)
        raise ValueError(
            f"鏈煡椋庢満鏈濆悜鍒嗙被: {orientation_class!r}; 鍙敤鍒嗙被: {valid}"
        )
    return orientation


def _require_finite(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} 蹇呴』鏄湁闄愭暟锛屽綋鍓嶄负 {value!r}")
    return converted


def normalize_centerline_pairs(
    pairs: Mapping[int, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """閫愮墖褰掍竴鍖栦腑蹇冪嚎锛涗换鎰忔闀垮害鍧囧彲锛屾柟鍚戝繀椤讳负杞瘋鎸囧悜鍙跺皷銆?""
    if not isinstance(pairs, Mapping):
        raise TypeError("pairs 蹇呴』鏄?{blade_id: (x, z)} 鏄犲皠")
    if not 2 <= len(pairs) <= 3:
        raise ValueError("鑷冲皯闇€瑕佷袱鐗囥€佽嚦澶氭敮鎸佷笁鐗囧彾鐗囦腑蹇冪嚎")

    normalized: Dict[int, Tuple[float, float]] = {}
    for blade_id, pair in pairs.items():
        if blade_id not in BETAS:
            raise ValueError(f"鍙剁墖缂栧彿鍙兘鏄?1/2/3锛屽綋鍓嶅寘鍚?{blade_id!r}")
        try:
            x_raw, z_raw = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"鍙剁墖 {blade_id} 鐨勪腑蹇冪嚎蹇呴』鏄簩鍏冪粍 (x, z)"
            ) from exc
        x_value = _require_finite(f"pairs[{blade_id}][0]", x_raw)
        z_value = _require_finite(f"pairs[{blade_id}][1]", z_raw)
        norm = math.hypot(x_value, z_value)
        if norm <= EPS:
            raise ValueError(f"鍙剁墖 {blade_id} 鐨勪腑蹇冪嚎闀垮害涓洪浂锛屾棤娉曠‘瀹氭柟鍚?)
        normalized[blade_id] = (x_value / norm, z_value / norm)
    return dict(sorted(normalized.items()))


def undo_camera_roll(
    normalized_pairs: Mapping[int, Tuple[float, float]],
    camera_roll_rad: float,
) -> Dict[int, Tuple[float, float]]:
    """灏嗗浘鍍忔柟鍚戝弽鏃?roll锛屽緱鍒拌鏂囧叕寮忎腑 roll 涔嬪墠鐨?(X_i,Z_i)銆?""
    cr = math.cos(camera_roll_rad)
    sr = math.sin(camera_roll_rad)
    return {
        blade_id: (
            cr * x_value + sr * z_value,
            -sr * x_value + cr * z_value,
        )
        for blade_id, (x_value, z_value) in normalized_pairs.items()
    }


def _orientation_bounds(orientation: str) -> Tuple[float, float, bool]:
    low_deg, high_deg = ORIENTATION_INTERVALS_DEG[orientation]
    return low_deg * RAD, high_deg * RAD, low_deg == high_deg


def _angle_in_interval(
    gamma: float,
    low: float,
    high: float,
    *,
    tolerance: float = 1e-8,
) -> bool:
    gamma = _wrap_radians_2pi(gamma)
    if low == high:
        distance = abs((gamma - low + math.pi) % TWO_PI - math.pi)
        return distance <= tolerance
    return low - tolerance <= gamma <= high + tolerance


def _deduplicate_angles(angles: Iterable[float], tolerance: float = 1e-8) -> List[float]:
    unique: List[float] = []
    for angle in angles:
        wrapped = _wrap_radians_2pi(angle)
        if all(
            abs((wrapped - existing + math.pi) % TWO_PI - math.pi) > tolerance
            for existing in unique
        ):
            unique.append(wrapped)
    return unique


def _pair_polynomial_coefficients(
    blade_i: int,
    blade_j: int,
    roll_compensated_pairs: Mapping[int, Tuple[float, float]],
    camera_pitch_rad: float,
) -> Tuple[float, float, float, float, float]:
    """杩斿洖 t=tan(gamma/2) 鐨勫洓娆″紡闄嶅箓绯绘暟 (p4,...,p0)銆?""
    x_i, z_i = roll_compensated_pairs[blade_i]
    x_j, z_j = roll_compensated_pairs[blade_j]
    delta = BETAS[blade_j] - BETAS[blade_i]
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)
    sp = math.sin(camera_pitch_rad)
    cp = math.cos(camera_pitch_rad)

    a = -sin_delta * z_i * z_j
    b = -sin_delta * sp * (z_i * x_j + x_i * z_j)
    c = -sin_delta * x_i * x_j * sp * sp
    d = cos_delta * cp * (x_j * z_i - x_i * z_j)
    e = -sin_delta * x_i * x_j * cp * cp

    return (
        a - d + e,
        -2.0 * b,
        -2.0 * a + 4.0 * c + 2.0 * e,
        2.0 * b,
        a + d + e,
    )


def _real_polynomial_roots(
    coefficients: Sequence[float],
    *,
    coefficient_tolerance: float = 1e-11,
    imaginary_tolerance: float = 1e-7,
) -> List[float]:
    coeff = np.asarray(coefficients, dtype=float)
    scale = float(np.max(np.abs(coeff)))
    if not math.isfinite(scale) or scale <= EPS:
        return []
    coeff = coeff / scale
    first = 0
    while first < len(coeff) and abs(float(coeff[first])) <= coefficient_tolerance:
        first += 1
    if first >= len(coeff) - 1:
        return []

    roots = np.roots(coeff[first:])
    real_roots: List[float] = []
    for root in roots:
        real = float(root.real)
        imaginary = float(root.imag)
        if abs(imaginary) <= imaginary_tolerance * (1.0 + abs(real)):
            real_roots.append(real)
    return real_roots


def _pair_gamma_roots(
    blade_i: int,
    blade_j: int,
    roll_compensated_pairs: Mapping[int, Tuple[float, float]],
    camera_pitch_rad: float,
) -> Tuple[List[float], Tuple[float, float, float, float, float]]:
    coefficients = _pair_polynomial_coefficients(
        blade_i,
        blade_j,
        roll_compensated_pairs,
        camera_pitch_rad,
    )
    gamma_roots = [
        _wrap_radians_2pi(2.0 * math.atan(root))
        for root in _real_polynomial_roots(coefficients)
    ]

    # t=tan(gamma/2) 涓嶈〃绀?gamma=pi锛涙樉寮忔楠屾棤绌疯繙鏍广€?    scale = max(1.0, *(abs(value) for value in coefficients))
    if abs(coefficients[0]) <= 1e-8 * scale:
        gamma_roots.append(math.pi)
    return _deduplicate_angles(gamma_roots), coefficients


def _alpha_line_solution(
    gamma: float,
    roll_compensated_pairs: Mapping[int, Tuple[float, float]],
    camera_pitch_rad: float,
) -> Tuple[float, float, float]:
    """鍥哄畾 gamma 鍚庤В鏋愭渶灏忓寲鍏ㄩ儴鍙剁墖 SSE锛沘lpha 鏆傛寜妯?pi 杩斿洖銆?""
    cg = math.cos(gamma)
    sg = math.sin(gamma)
    sp = math.sin(camera_pitch_rad)
    cp = math.cos(camera_pitch_rad)

    m_ss = 0.0
    m_sc = 0.0
    m_cc = 0.0
    for blade_id, (x_value, z_value) in roll_compensated_pairs.items():
        beta = BETAS[blade_id]
        h_value = z_value * cg + x_value * sp * sg
        c_value = x_value * cp
        p_value = h_value * math.cos(beta) + c_value * math.sin(beta)
        q_value = h_value * math.sin(beta) - c_value * math.cos(beta)
        m_ss += p_value * p_value
        m_sc += p_value * q_value
        m_cc += q_value * q_value

    harmonic_cos = 0.5 * (m_cc - m_ss)
    harmonic_sin = m_sc
    anisotropy = math.hypot(harmonic_cos, harmonic_sin)
    alpha = 0.5 * (math.atan2(harmonic_sin, harmonic_cos) + math.pi)
    trace_half = 0.5 * (m_ss + m_cc)
    minimum_sse = max(0.0, trace_half - anisotropy)
    observability = anisotropy / max(trace_half, EPS)
    return alpha % math.pi, minimum_sse, observability


def _residual_jacobian(
    alpha: float,
    gamma: float,
    roll_compensated_pairs: Mapping[int, Tuple[float, float]],
    camera_pitch_rad: float,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    cg = math.cos(gamma)
    sg = math.sin(gamma)
    sp = math.sin(camera_pitch_rad)
    cp = math.cos(camera_pitch_rad)
    residuals: List[float] = []
    jacobian: List[Tuple[float, float]] = []

    for blade_id, (x_value, z_value) in roll_compensated_pairs.items():
        q_value = alpha + BETAS[blade_id]
        sq = math.sin(q_value)
        cq = math.cos(q_value)
        h_value = z_value * cg + x_value * sp * sg
        h_gamma = -z_value * sg + x_value * sp * cg
        c_value = x_value * cp
        residual = h_value * sq - c_value * cq
        derivative_alpha = h_value * cq + c_value * sq
        derivative_gamma = h_gamma * sq
        residuals.append(residual)
        jacobian.append((derivative_alpha, derivative_gamma))
    return residuals, jacobian


def _refine_pose_gauss_newton(
    alpha: float,
    gamma: float,
    roll_compensated_pairs: Mapping[int, Tuple[float, float]],
    camera_pitch_rad: float,
    gamma_low: float,
    gamma_high: float,
    *,
    max_iterations: int = 15,
) -> Tuple[float, float]:
    """浠庝唬鏁板€欓€夎繛缁慨姝?SSE锛涜繖鏄眬閮ㄦ柟绋嬫眰瑙ｏ紝涓嶆槸瑙掑害缃戞牸鎼滅储銆?""
    alpha %= math.pi
    gamma = min(max(gamma, gamma_low), gamma_high)
    damping = 1e-10

    for _ in range(max_iterations):
        residuals, jacobian = _residual_jacobian(
            alpha,
            gamma,
            roll_compensated_pairs,
            camera_pitch_rad,
        )
        old_sse = sum(value * value for value in residuals)
        j11 = sum(row[0] * row[0] for row in jacobian)
        j12 = sum(row[0] * row[1] for row in jacobian)
        j22 = sum(row[1] * row[1] for row in jacobian)
        g1 = sum(row[0] * value for row, value in zip(jacobian, residuals))
        g2 = sum(row[1] * value for row, value in zip(jacobian, residuals))
        if math.hypot(g1, g2) <= 1e-13 * (1.0 + old_sse):
            break

        local_damping = damping * max(1.0, j11 + j22)
        a11 = j11 + local_damping
        a22 = j22 + local_damping
        determinant = a11 * a22 - j12 * j12
        if abs(determinant) <= EPS * max(1.0, a11 * a22):
            damping = max(1e-8, damping * 100.0)
            continue

        step_alpha = (-a22 * g1 + j12 * g2) / determinant
        step_gamma = (j12 * g1 - a11 * g2) / determinant
        if math.hypot(step_alpha, step_gamma) <= 1e-12:
            break

        accepted = False
        step_scale = 1.0
        for _ in range(10):
            trial_alpha = (alpha + step_scale * step_alpha) % math.pi
            trial_gamma = min(
                max(gamma + step_scale * step_gamma, gamma_low),
                gamma_high,
            )
            trial_residuals, _ = _residual_jacobian(
                trial_alpha,
                trial_gamma,
                roll_compensated_pairs,
                camera_pitch_rad,
            )
            trial_sse = sum(value * value for value in trial_residuals)
            if trial_sse <= old_sse + 1e-16:
                alpha, gamma = trial_alpha, trial_gamma
                damping = max(1e-12, damping * 0.3)
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            damping = min(1e6, max(1e-8, damping * 30.0))
            if damping >= 1e5:
                break

    # 鍥哄畾鏈€缁?gamma 鍚庡啀娆¤В鏋愭渶灏忓寲 alpha锛岄伩鍏嶈凯浠ｆ畫鐣欒宸€?    alpha, _, _ = _alpha_line_solution(
        gamma,
        roll_compensated_pairs,
        camera_pitch_rad,
    )
    return alpha, gamma


def _evaluate_directed_candidate(
    alpha_line: float,
    gamma: float,
    normalized_pairs: Mapping[int, Tuple[float, float]],
    camera_pitch_rad: float,
    camera_roll_rad: float,
    *,
    direction_tolerance: float = 1e-10,
) -> Dict[str, object] | None:
    sp = math.sin(camera_pitch_rad)
    cp = math.cos(camera_pitch_rad)
    sr = math.sin(camera_roll_rad)
    cr = math.cos(camera_roll_rad)
    cg = math.cos(gamma)
    sg = math.sin(gamma)

    best_candidate = None
    for alpha in (alpha_line % TWO_PI, (alpha_line + math.pi) % TWO_PI):
        residuals: Dict[int, float] = {}
        directed_scales: Dict[int, float] = {}
        projection_norms: Dict[int, float] = {}
        valid_direction = True

        for blade_id, (x_obs, z_obs) in normalized_pairs.items():
            q_value = alpha + BETAS[blade_id]
            sq = math.sin(q_value)
            cq = math.cos(q_value)
            a_value = sq * cg
            b_value = -sq * sg * sp + cq * cp
            u_x = a_value * cr - b_value * sr
            u_z = a_value * sr + b_value * cr
            residual = u_x * z_obs - u_z * x_obs
            directed_scale = u_x * x_obs + u_z * z_obs
            projection_norm = math.hypot(u_x, u_z)
            residuals[blade_id] = residual
            directed_scales[blade_id] = directed_scale
            projection_norms[blade_id] = projection_norm
            if directed_scale <= direction_tolerance or projection_norm <= EPS:
                valid_direction = False

        if not valid_direction:
            continue

        sse = sum(value * value for value in residuals.values())
        candidate: Dict[str, object] = {
            "alpha_rad": alpha,
            "alpha_deg_360": normalize_angle_360(alpha * DEG),
            "alpha_deg_120": normalize_alpha_120(alpha * DEG),
            # 鍏煎鏃ц皟鐢ㄦ柟鐨勫悕绉帮紱鍏剁墿鐞嗗惈涔変粛鏄?blade alpha銆?            "theta_rad": alpha,
            "theta_deg_mod120": normalize_alpha_120(alpha * DEG),
            "relative_yaw_rad": gamma,
            "relative_yaw_deg_360": normalize_angle_360(gamma * DEG),
            "gamma_rad": gamma,
            "gamma_deg_model": normalize_angle_360(gamma * DEG),
            "yaw_deg_360": normalize_angle_360(gamma * DEG),
            "sse": sse,
            "direction_sse": sse,
            "residuals": residuals,
            "directed_scales": directed_scales,
            "projection_norms": projection_norms,
            "max_abs_residual": max(abs(value) for value in residuals.values()),
            "min_directed_scale": min(directed_scales.values()),
            "min_projection_norm": min(projection_norms.values()),
        }
        if best_candidate is None or candidate["sse"] < best_candidate["sse"]:
            best_candidate = candidate
    return best_candidate


def _candidate_distance(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    alpha_first = float(first["alpha_rad"]) % math.pi
    alpha_second = float(second["alpha_rad"]) % math.pi
    alpha_distance = abs((alpha_first - alpha_second + math.pi / 2.0) % math.pi - math.pi / 2.0)
    gamma_distance = abs(
        (float(first["relative_yaw_rad"]) - float(second["relative_yaw_rad"]) + math.pi)
        % TWO_PI
        - math.pi
    )
    return max(alpha_distance, gamma_distance)


def solve_pose_from_unit_centerlines(
    pairs: Mapping[int, Tuple[float, float]],
    orientation_class: str,
    camera_pitch_deg: float = 0.0,
    camera_roll_deg: float = 0.0,
    *,
    refine: bool = True,
    max_refine_seeds: int = 4,
) -> Dict[str, object]:
    """
    鐢变袱鐗囨垨涓夌墖鏈夊悜涓績绾挎眰瑙?``alpha`` 涓庣浉鏈虹浉瀵?``gamma``銆?
    ``pairs`` 涓悇鍚戦噺浼氳鐙珛鍗曚綅鍖栵紝鎵€浠ュ厑璁镐娇鐢ㄤ换鎰忓儚绱犻暱搴︺€備緥濡傝嫢
    宸茬煡杞瘋鐐?``(h_x,h_z)`` 鍜屽彾灏栫偣 ``(t_x,t_z)``锛屼紶鍏?    ``(t_x-h_x, t_z-h_z)`` 鍗冲彲銆?
    杩斿洖鐨?``best`` 涓富鐗╃悊閲忎负 ``alpha_deg_120`` 鍜?    ``relative_yaw_deg_360``銆傚崟浣嶆柟鍚戞ā鍨嬩笉瀛樺湪鍏变韩 ``k``锛屽洜姝よ繑鍥炲€?    涓湁鎰忎笉鍖呭惈 ``k``銆?    """
    if max_refine_seeds <= 0:
        raise ValueError("max_refine_seeds 蹇呴』澶т簬 0")
    pitch_deg = _require_finite("camera_pitch_deg", camera_pitch_deg)
    roll_deg = _require_finite("camera_roll_deg", camera_roll_deg)
    pitch = pitch_deg * RAD
    roll = roll_deg * RAD
    orientation = canonical_orientation(orientation_class)
    gamma_low, gamma_high, is_canonical = _orientation_bounds(orientation)

    if orientation in {"left", "right"}:
        raise PoseDegeneracyError(
            "绮剧‘宸﹀彸渚ц鏃讹紝閫愬彾鍗曚綅涓績绾垮叏閮ㄥ叡绾匡紝鍙剁墖鏃嬭浆瑙?alpha 涓嶅彲瑙傦紱"
            "璇蜂娇鐢ㄦ枩瑙嗗浘鍍忔垨閲嶆柊鎷嶆憚"
        )
    if abs(math.cos(pitch)) <= 1e-8:
        raise PoseDegeneracyError(
            "camera pitch 鎺ヨ繎 卤90 搴︼紝鍗曚綅涓績绾挎ā鍨嬩腑鐨?alpha 杩炵画淇℃伅閫€鍖?
        )

    normalized_pairs = normalize_centerline_pairs(pairs)
    roll_compensated_pairs = undo_camera_roll(normalized_pairs, roll)

    pair_diagnostics: List[Dict[str, object]] = []
    gamma_seeds: List[float] = []
    if is_canonical:
        gamma_seeds.append(gamma_low)
    else:
        for blade_i, blade_j in combinations(normalized_pairs, 2):
            roots, coefficients = _pair_gamma_roots(
                blade_i,
                blade_j,
                roll_compensated_pairs,
                pitch,
            )
            accepted_roots = [
                gamma
                for gamma in roots
                if _angle_in_interval(gamma, gamma_low, gamma_high)
            ]
            gamma_seeds.extend(accepted_roots)
            pair_diagnostics.append(
                {
                    "blade_pair": (blade_i, blade_j),
                    "quartic_coefficients": coefficients,
                    "all_real_gamma_roots_deg": [
                        normalize_angle_360(value * DEG) for value in roots
                    ],
                    "accepted_gamma_roots_deg": [
                        normalize_angle_360(value * DEG) for value in accepted_roots
                    ],
                }
            )

        # 杩欎簺鏄尯闂寸害鏉熺殑鏈夐檺閿氱偣锛屼笉鏋勬垚瑙掑害缃戞牸锛涘湪寮哄櫔澹板鑷存煇鍙剁墖瀵?        # 娌℃湁鍖洪棿鍐呭疄鏍规椂锛屼粛缁欒繛缁慨姝ｆ彁渚涘彲琛屽垵鍊笺€?        margin = 1e-9
        gamma_seeds.extend(
            (
                gamma_low + margin,
                0.5 * (gamma_low + gamma_high),
                gamma_high - margin,
            )
        )

    gamma_seeds = [
        gamma
        for gamma in _deduplicate_angles(gamma_seeds)
        if _angle_in_interval(gamma, gamma_low, gamma_high)
    ]
    if not gamma_seeds:
        raise RuntimeError("鏈濆悜绫诲埆鍖洪棿鍐呮病鏈夊彲鐢ㄧ殑浠ｆ暟 yaw 鍊欓€?)

    ranked_seeds = []
    for gamma in gamma_seeds:
        alpha_line, seed_sse, observability = _alpha_line_solution(
            gamma,
            roll_compensated_pairs,
            pitch,
        )
        ranked_seeds.append((seed_sse, gamma, alpha_line, observability))
    ranked_seeds.sort(key=lambda item: item[0])

    pose_seeds: List[Tuple[float, float, float, str]] = [
        (alpha, gamma, observability, "algebraic")
        for _, gamma, alpha, observability in ranked_seeds
    ]
    if refine and not is_canonical:
        for _, gamma, alpha, _ in ranked_seeds[:max_refine_seeds]:
            refined_alpha, refined_gamma = _refine_pose_gauss_newton(
                alpha,
                gamma,
                roll_compensated_pairs,
                pitch,
                gamma_low,
                gamma_high,
            )
            _, _, refined_observability = _alpha_line_solution(
                refined_gamma,
                roll_compensated_pairs,
                pitch,
            )
            pose_seeds.append(
                (refined_alpha, refined_gamma, refined_observability, "refined")
            )

    candidates: List[Dict[str, object]] = []
    for alpha_line, gamma, observability, source in pose_seeds:
        candidate = _evaluate_directed_candidate(
            alpha_line,
            gamma,
            normalized_pairs,
            pitch,
            roll,
        )
        if candidate is None:
            continue
        candidate.update(
            {
                "source": source,
                "alpha_observability": observability,
                "is_near_side_view": abs(math.cos(gamma)) < math.sin(5.0 * RAD),
            }
        )
        if all(_candidate_distance(candidate, old) > 1e-7 for old in candidates):
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "娌℃湁鍊欓€夊悓鏃舵弧瓒虫湞鍚戝尯闂村拰杞瘋->鍙跺皷鐨勬鏂瑰悜绾︽潫锛?
            "璇锋鏌ヤ腑蹇冪嚎鏂瑰悜銆佸彾鐗囧惊鐜紪鍙锋垨鐩告満濮挎€?
        )

    candidates.sort(key=lambda item: float(item["sse"]))
    best = dict(candidates[0])
    best.update(
        {
            "orientation": orientation,
            "orientation_label": ORIENTATION_LABELS[orientation],
            "camera_pitch_deg": pitch_deg,
            "camera_roll_deg": roll_deg,
            "needs_side_rephotograph": best["is_near_side_view"],
            "alpha_observable": True,
            "observation_model": "independent_unit_directions",
            "objective": "paper_cross_product_sse",
            "solver_method": "pairwise_quartic_plus_gauss_newton",
        }
    )

    return {
        "best": best,
        "candidates": candidates,
        "normalized_pairs": normalized_pairs,
        "roll_compensated_pairs": roll_compensated_pairs,
        "diagnostics": {
            "orientation_interval_deg": ORIENTATION_INTERVALS_DEG[orientation],
            "n_visible_blades": len(normalized_pairs),
            "n_gamma_seeds": len(gamma_seeds),
            "gamma_seed_deg": [normalize_angle_360(value * DEG) for value in gamma_seeds],
            "pair_polynomials": pair_diagnostics,
            "refinement_enabled": refine,
            "max_refine_seeds": max_refine_seeds,
        },
    }


# 鏄惧紡鍒悕锛屼究浜庝粠鏃хず渚嬭縼绉伙紱杈撳叆璇箟浠嶇敱鏂囦欢鍚嶅拰 docstring 鏄庣‘鍖哄垎銆?solve_pose_with_orientation = solve_pose_from_unit_centerlines


if __name__ == "__main__":
    try:
        from .wt_pose_to_blade_centerlines import project_blade_centerlines
    except ImportError:
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
    print("alpha_deg_120          =", result["best"]["alpha_deg_120"])
    print("relative_yaw_deg_360   =", result["best"]["relative_yaw_deg_360"])
    print("direction_sse          =", result["best"]["direction_sse"])

