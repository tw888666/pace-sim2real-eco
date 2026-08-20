#!/usr/bin/env python3
"""纯 CPU 审计全部冻结 Terrain20sWide 宽场地形，不启动 Isaac Sim 或 GPU。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from pace_eco_lab.multi_terrain_protocol import (
    MAX_AUDITED_BACKWARD_M,
    MAX_AUDITED_FORWARD_M,
    MAX_AUDITED_LATERAL_M,
    PROTOCOL_VERSION,
    TERRAIN_ACTIVE_END_X_M,
    TERRAIN_LENGTH_M,
    TERRAIN_ORIGIN_X_M,
    TERRAIN_WIDTH_M,
    terrain_seed,
)
from pace_eco_lab.terrain_geometry import LongTerrainAuditCfg, geometry_audit
from pace_eco_lab.terrain_boundary import FOOT_BOUNDARY_AUDIT_VERSION


parser = argparse.ArgumentParser(description="审计 Terrain20sWide 宽场地形的边界、出生点和有限几何。")
parser.add_argument("--output", required=True)
args = parser.parse_args()


def _close(left: float, right: float, tolerance: float = 1.0e-8) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖几何审计：{output}")
    instances: list[dict[str, object]] = []
    roles = ("stage1_budget_train", "stage1_calibration", "stage1_holdout")
    categories = {
        "flat": ("level",),
        "rough": ("level",),
        "stairs": ("up", "down"),
        "boxes": ("level",),
        "slope": ("up", "down"),
    }
    for role in roles:
        for category, directions in categories.items():
            seed = terrain_seed(role, category, 0 if role.endswith("train") else None)
            for direction in directions:
                for difficulty in (0.10, 0.50, 0.90):
                    audit = geometry_audit(
                        LongTerrainAuditCfg(category=category, direction=direction, seed=seed),
                        difficulty,
                    )
                    bounds_min = audit["bounds_min"]
                    bounds_max = audit["bounds_max"]
                    origin = audit["origin"]
                    checks = {
                        "覆盖完整长地形块x": _close(float(bounds_min[0]), 0.0)
                        and _close(float(bounds_max[0]), TERRAIN_LENGTH_M),
                        "覆盖完整长地形块y": _close(float(bounds_min[1]), 0.0)
                        and _close(float(bounds_max[1]), TERRAIN_WIDTH_M),
                        "origin位于冻结出生中心线": _close(float(origin[0]), TERRAIN_ORIGIN_X_M)
                        and _close(float(origin[1]), TERRAIN_WIDTH_M / 2.0),
                        "前向30m机身预警线仍在块内且余量5m": _close(
                            TERRAIN_LENGTH_M - (TERRAIN_ORIGIN_X_M + MAX_AUDITED_FORWARD_M), 5.0
                        ),
                        "后向2m机身预警线仍在块内且余量28m": _close(
                            TERRAIN_ORIGIN_X_M - MAX_AUDITED_BACKWARD_M, 28.0
                        ),
                        "侧向3m机身预警线仍有27m余量": _close(
                            TERRAIN_WIDTH_M / 2.0 - MAX_AUDITED_LATERAL_M, 27.0
                        ),
                        "有效几何末端留6m尾部": _close(
                            TERRAIN_LENGTH_M - TERRAIN_ACTIVE_END_X_M, 6.0
                        ),
                        "存在碰撞三角面": int(audit["face_count"]) > 0,
                    }
                    params = audit["parameters"]
                    if category in ("stairs", "slope"):
                        elevation = float(params["total_elevation_change_m"])
                        checks["上下行符号正确"] = elevation > 0.0 if direction == "up" else elevation < 0.0
                    if category == "slope":
                        checks["斜坡有限且不超过8度"] = 0.0 < float(params["slope_degrees"]) <= 8.0
                    if category == "boxes":
                        boxes = params["instances"]
                        checks["箱块不封死长地形宽度"] = all(
                            float(item["lateral_size_m"]) <= 2.4 for item in boxes
                        )
                        checks["箱块位于有效几何段"] = all(
                            TERRAIN_ORIGIN_X_M < float(item["x_center_m"]) < TERRAIN_ACTIVE_END_X_M
                            for item in boxes
                        )
                    if not all(checks.values()):
                        failed = [name for name, passed in checks.items() if not passed]
                        raise RuntimeError(f"几何审计失败 {role}/{category}/{direction}/{difficulty}：{failed}")
                    instances.append(
                        {
                            "角色": role,
                            "地形": category,
                            "方向": direction,
                            "难度连续值": difficulty,
                            "地形seed": seed,
                            "检查": checks,
                            "几何": audit,
                        }
                    )
    payload = {
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "审计状态": "全部通过",
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "是否启动Isaac_Sim": False,
        "是否使用GPU": False,
        "实例数": len(instances),
        "跨块结论": (
            "机身前30m、后2m、侧向3m均为块内路线预警线；真实块边缘相对出生origin为"
            "后-30m、前35m、左右±30m；任一回合的接触足端中心越过真实边缘，"
            "整份评估均无效。"
        ),
        "出生点结论": "origin 位于局部 x=30m、宽度中心；所有类别均保留平整出生平台。",
        "有限斜坡楼梯结论": "斜坡和楼梯在长地形块内结束并接有限平台，不会无限延伸。",
        "逐实例": instances,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
