#!/usr/bin/env python3
"""v1.4 禁用固定权重搜索；未来调优必须另建协议。"""

from __future__ import annotations

import argparse

from pace_eco_lab.multi_terrain_protocol import PROTOCOL_VERSION


parser = argparse.ArgumentParser(description="已禁用的多地形固定权重选择入口。")
parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
parser.add_argument("--calibration_root", required=True)
parser.add_argument("--energy_reference_json", required=True)
parser.add_argument("--output", required=True)
parser.parse_args()

raise RuntimeError(
    f"{PROTOCOL_VERSION} 已冻结使用 PACE 论文 W100=-0.00016，不允许网格选择；"
    "后续调优必须建立新协议版本、新训练与地形 seed、新 calibration/holdout 和新输出目录。"
)
