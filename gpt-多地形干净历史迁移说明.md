# 多地形实验干净历史迁移说明

## 结论

多地形实验已迁移到以 `origin/codex/pace-eco-ppolag` 为基线的新分支
`exp/multi-terrain-comparison-v2-clean`。该分支与旧多地形分支没有共同祖先，不包含旧
机器人项目的目录、资产、包配置或提交历史。

## 保留内容

- `source/pace_sim2real/`：干净 PACE Sim2Real 基线；
- `pace_eco_lab/`：现有 PACE-ECO/ANYmal D 环境与多地形实现的精确源码快照；
- `scripts/pace_eco/`：训练、评估、审计和汇总入口；
- `tests/pace_eco_lab/`：已有 PACE-ECO 与多地形定向测试；
- `gpt-多地形*`：冻结协议、审计证据和执行文档。

保留 `pace_eco_lab` 的精确实现，是为了确保已有检查点对应的机器人动力学、观测、奖励、
终止、随机种子和地形语义不发生变化。迁移不修改已有模型权重或评估结果。

## 明确排除内容

- 与当前 ANYmal D 实验无关的机器人代码和资产；
- 旧工程的根 `setup.py`、测试入口和上游说明文档；
- 旧分支提交祖先；
- `logs/`、`results/`、模型权重和其他运行产物。

## 数据与结果边界

ANYmal D 的 PACE 标定数组继续作为外部数据管理，不写入代码仓库。GPU（图形处理器）入口
只读取显式 `PACE_ECO_DATA_ROOT`；未提供时仅检查当前干净工作树内的 `pace_data`，不会
静默回退到另一工作树。旧工作树中的训练日志、检查点和评估结果保持原位，不移动、不覆盖。

## 远端替换原则

在新分支通过非 GPU 测试和人工审阅前，不重写已经发布的
`exp/multi-terrain-comparison-v1`。确认后应先 push（推送）干净分支作为备份，再使用带
租约保护的强制推送替换旧分支；不能普通 merge（合并），因为合并会重新引入旧提交祖先。
