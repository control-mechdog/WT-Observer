# -*- coding: utf-8 -*-
"""
鐢卞凡鐭ラ鏈哄Э鎬佷笌鐩告満濮挎€佹鍚戠敓鎴愬彾鐗囦腑蹇冪嚎銆?
鏈枃浠跺疄鐜?draft_revise/main.tex 涓殑姝ｅ悜鏃嬭浆閾撅細

    n_i^c = Ry(-roll) Rx(-pitch) Rz(relative_yaw)
            Ry(alpha + beta_i) b0

鍏朵腑 b0=(0,0,1)锛宐eta_i={0,120,240} 搴︺€傝緭鍑哄寘鎷細

1. camera_vectors_3d锛氱浉鏈哄潗鏍囩郴涓殑涓夌淮鍙剁墖鍗曚綅鏂瑰悜锛?2. projected_pairs锛氭姇褰卞埌褰掍竴鍖栧浘鍍忓钩闈㈢殑鍘熷 (x,z) 鍒嗛噺锛?3. unit_pairs锛氭瘡鐗囧垎鍒崟浣嶅寲鍚庣殑涓績绾挎柟鍚戯紝绗﹀悎璁烘枃鐨勫叡绾挎畫宸緭鍏ワ紱
4. solver_pairs锛氫繚鎸佸叡鍚屾姇褰卞昂搴︾殑鏈夊悜鍚戦噺锛屽彲鐩存帴杈撳叆褰撳墠鍏变韩-k瑙ｆ瀽姹傝В鍣ㄣ€?
鍧愭爣绾﹀畾锛氱浉鏈?X 鍚戝浘鍍忓彸渚э紝Y 娌垮厜杞村悜鍓嶏紝Z 鍚戝浘鍍忎笂鏂癸紱鎵€鏈夎緭鍏?瑙掑潎涓哄害銆俽elative_yaw_deg 鏄鏂囨姇褰辨柟绋嬩腑鐨勭浉鏈虹浉瀵瑰亸鑸銆傝嫢鏀逛负
杈撳叆 turbine_yaw_deg 鍜?camera_yaw_deg锛屾湰鏂囦欢鎸夎鏂囨枃瀛椾腑鐨勫悓涓€鏍囬噺
瑙掑害绾﹀畾璁＄畻 relative_yaw = turbine_yaw - camera_yaw銆?

"""

import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

RAD = math.pi / 180.0
DEG = 180.0 / math.pi
EPS = 1e-12

BETAS_DEG = {
    1: 0.0,
    2: 120.0,
    3: 240.0,
}


def normalize_angle_360(angle_deg: float) -> float:
    return angle_deg % 360.0


def normalize_blade_angle_120(angle_deg: float) -> float:
    return angle_deg % 120.0


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} 蹇呴』鏄湁闄愭暟锛屽綋鍓嶄负 {value!r}")


def _resolve_relative_yaw(
    relative_yaw_deg: Optional[float],
    turbine_yaw_deg: Optional[float],
    camera_yaw_deg: Optional[float],
) -> Tuple[float, Optional[float], Optional[float]]:
    direct_mode = relative_yaw_deg is not None
    world_mode = turbine_yaw_deg is not None or camera_yaw_deg is not None

    if direct_mode and world_mode:
        raise ValueError(
            "relative_yaw_deg 涓?turbine_yaw_deg/camera_yaw_deg 鍙兘閫夋嫨涓€绉嶈緭鍏ユ柟寮?
        )

    if direct_mode:
        _require_finite("relative_yaw_deg", relative_yaw_deg)
        return normalize_angle_360(relative_yaw_deg), None, None

    if turbine_yaw_deg is None or camera_yaw_deg is None:
        raise ValueError(
            "璇锋彁渚?relative_yaw_deg锛屾垨鑰呭悓鏃舵彁渚?turbine_yaw_deg 鍜?camera_yaw_deg"
        )

    _require_finite("turbine_yaw_deg", turbine_yaw_deg)
    _require_finite("camera_yaw_deg", camera_yaw_deg)
    turbine_yaw = normalize_angle_360(turbine_yaw_deg)
    camera_yaw = normalize_angle_360(camera_yaw_deg)
    relative_yaw = normalize_angle_360(turbine_yaw - camera_yaw)
    return relative_yaw, turbine_yaw, camera_yaw


def orientation_class_from_relative_yaw(
    relative_yaw_deg: float,
    canonical_tolerance_deg: float = 1e-9,
) -> str:
    """鎸夌収璁烘枃鐨勫叓涓浉鏈虹浉瀵规湞鍚戝尯闂磋繑鍥炰笅鍒掔嚎褰㈠紡鐨勭被鍒悕銆?""
    _require_finite("relative_yaw_deg", relative_yaw_deg)
    if canonical_tolerance_deg < 0:
        raise ValueError("canonical_tolerance_deg 涓嶈兘涓鸿礋鏁?)

    yaw = normalize_angle_360(relative_yaw_deg)
    canonical = (
        (0.0, "back"),
        (90.0, "right"),
        (180.0, "front"),
        (270.0, "left"),
    )
    for angle, label in canonical:
        distance = abs((yaw - angle + 180.0) % 360.0 - 180.0)
        if distance <= canonical_tolerance_deg:
            return label

    if 0.0 < yaw < 90.0:
        return "back_right"
    if 90.0 < yaw < 180.0:
        return "front_right"
    if 180.0 < yaw < 270.0:
        return "front_left"
    return "back_left"


def project_blade_centerlines(
    blade_rotation_deg: float,
    *,
    relative_yaw_deg: Optional[float] = None,
    turbine_yaw_deg: Optional[float] = None,
    camera_yaw_deg: Optional[float] = None,
    camera_pitch_deg: float = 0.0,
    camera_roll_deg: float = 0.0,
    projection_scale: float = 1.0,
    blade_ids: Iterable[int] = (1, 2, 3),
) -> Dict[str, object]:
    """
    姝ｅ悜鎶曞奖涓夌墖鎴栨寚瀹氬彾鐗囩殑鐞嗚涓績绾裤€?
    鍙傛暟
    ----
    blade_rotation_deg:
        鍙惰疆鏃嬭浆瑙?alpha锛屽厑璁镐换鎰忚搴︼紝缁撴灉鍚屾椂杩斿洖 mod 120 搴﹀舰寮忋€?    relative_yaw_deg:
        璁烘枃鎶曞奖妯″瀷涓殑鐩告満鐩稿鍋忚埅瑙?gamma銆傛帹鑽愮敤浜庡悎鎴愬疄楠屻€?    turbine_yaw_deg, camera_yaw_deg:
        鍙€変笘鐣岃埅鍚戣緭鍏ワ紱涓よ€呭繀椤诲悓鏃剁粰鍑猴紝涓斾笉鑳戒笌 relative_yaw_deg 鍚屾椂缁欏嚭銆?    camera_pitch_deg, camera_roll_deg:
        璁烘枃涓殑鐩告満 pitch 鍜?roll锛屽崟浣嶄负搴︺€?    projection_scale:
        projected_pairs 鐨勭粺涓€涔樻暟銆傚綋鍓嶉€嗘眰瑙ｅ櫒浣跨敤 u=k*x锛屽洜姝よ妯℃嫙
        k=0.2 鏃跺簲璁剧疆 projection_scale=1/k=5銆傝涔樻暟瀵?unit_pairs 鏃犲奖鍝嶃€?    blade_ids:
        瑕佺敓鎴愮殑鍙剁墖缂栧彿锛屽繀椤绘槸 1/2/3 涓殑涓や釜鎴栦笁涓紱榛樿鐢熸垚鍏ㄩ儴涓夌墖銆?
    杩斿洖
    ----
    unit_pairs:
        姣忕墖鐙珛鍗曚綅鍖栫殑浜岀淮鏂瑰悜锛岄€傚悎璁烘枃鐨?cross-product/鍏辩嚎娈嬪樊銆?    solver_pairs:
        projection_scale * (u_ix,u_iz)锛屽彲鐩存帴杈撳叆鍏变韩-k瑙ｆ瀽閫嗘眰瑙ｅ櫒銆?    """
    _require_finite("blade_rotation_deg", blade_rotation_deg)
    _require_finite("camera_pitch_deg", camera_pitch_deg)
    _require_finite("camera_roll_deg", camera_roll_deg)
    _require_finite("projection_scale", projection_scale)
    if projection_scale <= 0:
        raise ValueError("projection_scale 蹇呴』澶т簬 0")

    ids = tuple(blade_ids)
    if not (2 <= len(ids) <= 3):
        raise ValueError("blade_ids 蹇呴』鍖呭惈涓ょ墖鎴栦笁鐗囧彾鐗?)
    if len(set(ids)) != len(ids) or any(i not in BETAS_DEG for i in ids):
        raise ValueError("blade_ids 鍙兘鏄笉閲嶅鐨?1銆?銆?")

    relative_yaw, turbine_yaw, camera_yaw = _resolve_relative_yaw(
        relative_yaw_deg,
        turbine_yaw_deg,
        camera_yaw_deg,
    )

    alpha = blade_rotation_deg * RAD
    gamma = relative_yaw * RAD
    pitch = camera_pitch_deg * RAD
    roll = camera_roll_deg * RAD

    cg = math.cos(gamma)
    sg = math.sin(gamma)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    camera_vectors_3d: Dict[int, Tuple[float, float, float]] = {}
    projected_pairs: Dict[int, Tuple[float, float]] = {}
    solver_pairs: Dict[int, Tuple[float, float]] = {}
    unit_pairs: Dict[int, Tuple[float, float]] = {}
    image_angles_deg: Dict[int, float] = {}
    projection_norms: Dict[int, float] = {}
    degenerate_blades: List[int] = []

    for blade_id in ids:
        q = alpha + BETAS_DEG[blade_id] * RAD
        sq = math.sin(q)
        cq = math.cos(q)

        # Rz(gamma) 涔嬪悗鍐嶇敤 Rx(-pitch) 鍜?Ry(-roll) 琛ュ伩鐩告満濮挎€併€?        a_i = sq * cg
        b_i = -sq * sg * sp + cq * cp
        y_i = sq * sg * cp + cq * sp

        x_i = a_i * cr - b_i * sr
        z_i = a_i * sr + b_i * cr

        vector_norm = math.sqrt(x_i * x_i + y_i * y_i + z_i * z_i)
        if abs(vector_norm - 1.0) > 1e-10:
            raise RuntimeError(
                f"鍙剁墖 {blade_id} 鐨勪笁缁存棆杞粨鏋滀笉鏄崟浣嶅悜閲? norm={vector_norm}"
            )

        camera_vectors_3d[blade_id] = (x_i, y_i, z_i)
        projected_pairs[blade_id] = (x_i, z_i)
        solver_pairs[blade_id] = (
            projection_scale * x_i,
            projection_scale * z_i,
        )

        projection_norm = math.hypot(x_i, z_i)
        projection_norms[blade_id] = projection_norm
        if projection_norm <= EPS:
            degenerate_blades.append(blade_id)
            continue

        unit_x = x_i / projection_norm
        unit_z = z_i / projection_norm
        unit_pairs[blade_id] = (unit_x, unit_z)
        image_angles_deg[blade_id] = normalize_angle_360(
            math.atan2(unit_x, unit_z) * DEG
        )

    return {
        "blade_rotation_deg": blade_rotation_deg,
        "blade_rotation_deg_mod120": normalize_blade_angle_120(blade_rotation_deg),
        "relative_yaw_deg": relative_yaw,
        "orientation_class": orientation_class_from_relative_yaw(relative_yaw),
        "turbine_yaw_deg": turbine_yaw,
        "camera_yaw_deg": camera_yaw,
        "camera_pitch_deg": camera_pitch_deg,
        "camera_roll_deg": camera_roll_deg,
        "projection_scale": projection_scale,
        "shared_scale_k": 1.0 / projection_scale,
        "side_view_factor": abs(cg),
        "is_exact_side_view": abs(cg) <= 1e-10,
        "blade_ids": ids,
        "camera_vectors_3d": camera_vectors_3d,
        "projected_pairs": projected_pairs,
        "solver_pairs": solver_pairs,
        "unit_pairs": unit_pairs,
        "image_angles_deg": image_angles_deg,
        "projection_norms": projection_norms,
        "degenerate_blades": degenerate_blades,
    }


def add_independent_angular_noise(
    pairs: Dict[int, Tuple[float, float]],
    noise_std_deg: float,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    缁欐瘡鐗囦簩缁翠腑蹇冪嚎鍔犲叆鐙珛闆跺潎鍊奸珮鏂搴﹀櫔澹帮紝骞朵繚鎸佸悇鍚戦噺鍘熼暱搴︺€?
    杩欏搴?main.tex 涓?centerline noise 鏍囧噯宸负 1/2/4 搴︾殑鍚堟垚瀹為獙銆?    """
    _require_finite("noise_std_deg", noise_std_deg)
    if noise_std_deg < 0:
        raise ValueError("noise_std_deg 涓嶈兘涓鸿礋鏁?)

    rng = random.Random(seed)
    noisy_pairs: Dict[int, Tuple[float, float]] = {}
    sampled_noise_deg: Dict[int, float] = {}

    for blade_id, (x_i, z_i) in pairs.items():
        radius = math.hypot(x_i, z_i)
        if radius <= EPS:
            raise ValueError(f"鍙剁墖 {blade_id} 鐨勪簩缁存柟鍚戦€€鍖栵紝鏃犳硶娣诲姞瑙掑害鍣０")

        noise_deg = rng.gauss(0.0, noise_std_deg)
        angle = math.atan2(x_i, z_i) + noise_deg * RAD
        noisy_pairs[blade_id] = (
            radius * math.sin(angle),
            radius * math.cos(angle),
        )
        sampled_noise_deg[blade_id] = noise_deg

    return {
        "pairs": noisy_pairs,
        "noise_deg": sampled_noise_deg,
        "noise_std_deg": noise_std_deg,
        "seed": seed,
    }


if __name__ == "__main__":
    forward = project_blade_centerlines(
        blade_rotation_deg=20.0,
        relative_yaw_deg=135.0,
        camera_pitch_deg=15.0,
        camera_roll_deg=-8.0,
    )

    print("orientation_class =", forward["orientation_class"])
    print("unit_pairs        =", forward["unit_pairs"])
    print("solver_pairs      =", forward["solver_pairs"])
    print("image_angles_deg  =", forward["image_angles_deg"])
    print("degenerate_blades =", forward["degenerate_blades"])

    try:
        try:
            from .wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines
        except ImportError:
            from wt_pose_from_unit_centerlines import solve_pose_from_unit_centerlines

        inverse = solve_pose_from_unit_centerlines(
            forward["unit_pairs"],
            forward["orientation_class"],
            camera_pitch_deg=forward["camera_pitch_deg"],
            camera_roll_deg=forward["camera_roll_deg"],
        )
        best = inverse["best"]
        print("recovered_alpha   =", best["alpha_deg_120"])
        print("recovered_yaw     =", best["relative_yaw_deg_360"])
        print("inverse_sse       =", best["sse"])
    except ImportError:
        # 浣滀负鍖呭鍏ユ椂锛屼笉寮哄埗渚濊禆閫嗘眰瑙ｆ枃浠讹紱姝ｅ悜鎶曞奖缁撴灉浠嶇劧鏈夋晥銆?        pass

