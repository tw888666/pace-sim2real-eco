#!/usr/bin/env python3
"""从逐回合 CSV 生成独立 JSON/CSV/Markdown 多地形统计。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pace_eco_lab.multi_terrain_statistics import (
    load_episode_csv,
    summarize_rows,
    validate_formal_rows,
)


parser = argparse.ArgumentParser(description="汇总冻结多地形逐回合结果。")
parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()


def main() -> None:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖统计目录：{output}")
    rows = load_episode_csv(args.inputs)
    validate_formal_rows(rows, args.stage)
    output.mkdir(parents=True)
    summary = summarize_rows(rows)
    (output / "gpt-多地形最终统计.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    seed_rows = summary["逐seed逐地形"]
    with (output / "gpt-多地形逐种子汇总.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    csv_tables = {
        "gpt-多地形逐地形汇总.csv": summary["逐地形seed等权"],
        "gpt-多地形方向难度汇总.csv": summary["逐方向逐难度seed等权"],
        "gpt-多地形相对任务型节能汇总.csv": summary["逐地形相对任务型节能汇总"],
    }
    for filename, table in csv_tables.items():
        if not table:
            continue
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ["# 多地形最终宏平均摘要", "", "| 方法 | 宏平均成功率 | 宏平均联合合格率 | 最差地形 | 最差联合合格率 |", "|---|---:|---:|---|---:|"]
    for method, values in summary["跨地形宏平均与最差地形"].items():
        lines.append(
            f"| {method} | {values['地形等权宏平均成功率']:.4f} | "
            f"{values['地形等权宏平均联合合格率']:.4f} | {values['最差地形']} | "
            f"{values['最差地形联合合格率']:.4f} |"
        )
    (output / "gpt-多地形最终统计报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
