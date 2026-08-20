# PACE-ECO 多地形 Terrain20s GPU 冒烟验证报告

> 已归档：本报告只验证v1.3的40×10 m、5×20网格，不能放行v1.4 Terrain20sWide。

协议版本：`gpt-multi-terrain-v1.3`

执行时间：2026-08-06 20:15–20:19 CST

## 1. 结论与边界

经 nullptr 明确授权，使用 GPU0 串行执行阶段一 flat、rough、stairs、boxes、slope 五类
Terrain20s 任务型 PPO（近端策略优化）冒烟。五类均以退出码 0 完成 2/2 次更新，未检出
Traceback（异常堆栈）、CUDA error（并行计算平台错误）、out of memory（显存不足）、
segmentation fault（段错误）或 assertion error（断言错误）。

每类使用 16 个环境、PPO seed 900 和独立地形 seed，共完成 768 个环境步，并分别生成
`model_0.pt` 与 `model_1.pt`。共生成 10 个约 5.24 MB 的冒烟权重，全部位于隔离的
`multi_terrain_v1_3/smoke` 根目录，不得用于 B_ref、calibration（校准）、holdout
（留出集）或正式统计。

本次没有启动 3000 次正式训练、B_ref 标定或正式评估，也没有生成正式评估 JSON/CSV。

## 2. 五类结果

| 地形 | 地形 seed | 成功运行目录时间戳 | 更新 | 最终检查点 | 两次更新时间 |
|---|---:|---|---:|---|---|
| flat | 910900 | `2026-08-06_20-16-01` | 2/2 | `model_1.pt` | 2.67 s、1.70 s |
| rough | 911900 | `2026-08-06_20-16-43` | 2/2 | `model_1.pt` | 1.95 s、1.61 s |
| stairs | 912900 | `2026-08-06_20-17-28` | 2/2 | `model_1.pt` | 2.17 s、1.65 s |
| boxes | 913900 | `2026-08-06_20-18-29` | 2/2 | `model_1.pt` | 2.58 s、1.95 s |
| slope | 914900 | `2026-08-06_20-18-56` | 2/2 | `model_1.pt` | 2.26 s、1.77 s |

成功运行根目录：

`logs/supplementary/multi_terrain_v1_3/smoke/rsl_rl/`

启动日志根目录：

`logs/supplementary/multi_terrain_v1_3/smoke/launch/`

## 3. 运行时不变量

五类任务实际构造结果一致：

- 固定 20 s 协议对应的历史奖励项保持为 velocity 0.2、collision -1.0、
  foot_touchdown -0.1；任务型 energy 权重为 0；
- 终止项恰为 `time_out` 和 `base_contact`；
- Curriculum Manager（课程管理器）为 0 个性能驱动课程项；
- actor（策略网络）输入为 48 维纯本体观测；
- critic（价值网络）输入为 353 维，含 294 维特权高度扫描；
- 环境步长 0.02 s，物理步长 0.0025 s；
- 五类任务 ID 均使用独立 `Terrain20s-Anymal-D-v0` 后缀。

这些证据确认训练入口、环境构造、地形生成、观测、奖励、终止、日志和 PPO 更新链路可用。
2 次更新不代表策略已学会通过地形，也不能代替 B_ref 或正式效果结论。

## 4. GPU 与警告

执行前 GPU0 约占用 41171/81920 MiB，计算利用率为 0；其上已有 ComfyUI（图形工作流）
进程未被停止或修改。五类串行完成后 GPU0 回到约 41172 MiB、利用率 0，未残留本工作树
的训练或评估进程。GPU5 未使用。

十次 Isaac Sim（机器人仿真平台）启动，包括五次冒烟和五次状态审计，均打印
`Skipping NVIDIA GPU due CUDA being in bad state` 设备枚举警告。与此同时设备表明确将
所选 A100 标为 `Active Yes: 0`，环境实际使用 `cuda:0`，所有命令均正常完成。因此当前
将其记录为非致命的 CUDA/Vulkan（并行计算/图形接口）设备枚举警告；后续每次正式启动
仍必须核对 active 设备、退出码和日志，不能无条件忽略。

## 5. 停止点

Terrain20s 五类 GPU 链路冒烟门已通过，文档同步后的56项非GPU回归测试及静态检查也
已通过。正式训练仍未授权，且当前工作树未提交；停止并等待 nullptr 审阅。
