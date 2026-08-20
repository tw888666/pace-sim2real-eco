# PACE-ECO Terrain20sWide GPU冒烟与状态审计报告

协议版本：`gpt-multi-terrain-v1.4`

执行时间：2026-08-10 13:11–13:31 CST

## 1. 结论与执行边界

在 nullptr 明确授权后，GPU0上依次完成阶段一五类任务型PPO（近端策略优化）的16环境×2次
更新链路冒烟、五类实际碰撞状态审计，以及最重rough（粗糙地形）的4096环境×2次更新容量
冒烟。五类链路均为2/2更新并生成隔离的`model_1.pt`；五类状态审计均为17/17项通过；
rough容量冒烟累计196608步并生成`model_1.pt`。

本次未启动3000次正式训练、calibration（校准）或holdout（留出集）评估，未生成正式
JSON/CSV。所有冒烟权重位于`multi_terrain_v1_4/smoke`或`capacity_smoke`，禁止用于正式
评估和论文统计。GPU0上其他用户的ComfyUI（图形工作流）进程未被停止或修改。

## 2. 五类16环境GPU链路冒烟

共同设置为PPO seed 900、每类独立冻结地形seed、16环境、2次更新。实际网络为actor
（策略网络）48维纯本体观测，critic（价值网络）353维观测，其中含294维特权高度扫描；
奖励、终止和课程分别保持历史四项、`time_out/base_contact`两项和零性能课程。

| 地形 | 地形seed | 运行目录时间戳 | 地形生成/场景创建/仿真启动 | 两次更新时间 | `model_1.pt` SHA256 |
|---|---:|---|---|---|---|
| flat | 930900 | `2026-08-10_13-11-58` | 0.02/2.20/1.17 s | 2.56/1.53 s | `3bcbe74225b9a2a567e1eb9024c92120fa94f03794314fa4fca9adbf039a5e33` |
| rough | 931900 | `2026-08-10_13-12-28` | 7.45/117.93/56.75 s | 1.83/1.52 s | `47c5963c847e5d58e61c588de2d43dc1476eedf3b43ec8f205419989e8c1b451` |
| stairs | 932900 | `2026-08-10_13-15-48` | 0.39/2.69/3.33 s | 1.98/1.68 s | `94e400d136920fd2389957eec6c2ca09071f19eb2f5a214256c50e53435d3ea4` |
| boxes | 933900 | `2026-08-10_13-16-13` | 0.12/2.23/0.85 s | 1.78/1.33 s | `9e8783e6aec9d1e7692e963caed18dbc1a85de0503ad519ebecf433ebb99bc08` |
| slope | 934900 | `2026-08-10_13-16-31` | 0.02/2.06/0.93 s | 1.83/1.52 s | `c9ad99a8f0bddff436c2c7da3e7affc1a3a53e034811df715ec8c3e3885273ef` |

rough的启动开销远高于其他类别，来自高分辨率随机高度场和碰撞网格构建；正式排程不能用
flat的启动时间代表rough。启动日志根为
`logs/supplementary/multi_terrain_v1_4/smoke/launch/`，权重根为同级`smoke/rsl_rl/`。

## 3. 五类实际碰撞状态审计

每类审计实际生成5行×10列共50张碰撞地形，并实例化10个环境覆盖全部地形列；使用
199950条射线审计表面。每类以下17项全部为真：首步张量有限、origin网格、列元数据、
接触传感器与物理视图的足端坐标一致、运动学边界、边缘预筛、出生时接触足不越界、出生
高度、表面有限性、五级几何难度、零性能课程，以及只有两个历史终止项。stairs和slope
均精确为5列上行、5列下行，没有无限坡、无限楼梯或提前进入相邻地形类型。

| 地形 | 审计JSON | SHA256 |
|---|---|---|
| flat | `gpt-flat-仿真状态审计-20260810_131646.json` | `9e025f34a1736161c07bfbddd6af502fcba5f150bcdedee5376b22f76df02826` |
| rough | `gpt-rough-仿真状态审计-20260810_131705.json` | `6849ee4f1bf7ee975c1509a3c706838baae3121f7c5fcba83d41050bdbf147a0` |
| stairs | `gpt-stairs-仿真状态审计-20260810_132024.json` | `10044dd05df8c5e9dfb24b70b6a09ddd4e153be1abef8ebbfc3a493db736d79c` |
| boxes | `gpt-boxes-仿真状态审计-20260810_132038.json` | `7306b862109ae7fed48c7a23f53c7ce80f742870a5572e42d49355f71e170c98` |
| slope | `gpt-slope-仿真状态审计-20260810_132051.json` | `ff2de3fae2eec30f5fe0210af71ba0c0c9735923e83ba98cc49862cbe650c5df` |

审计结果根为`logs/supplementary/multi_terrain_v1_4/state_audit/results/`。

## 4. rough 4096环境容量冒烟

第一次容量命令复用了冻结为16环境的`stage1_smoke_train`角色，训练入口在创建环境前正确
拒绝4096环境；但`SimulationApp.close()`清理路径使外层命令显示退出码0，且没有运行目录
或权重。该次不能记为通过，保留日志
`gpt_GPU0_stage1_rough_capacity_smoke_20260810_132247.log`作为失败证据。

修复措施如下：

- 新增独立`stage1_capacity_smoke_train`/`stage2_capacity_smoke_train`角色，冻结为4096环境、
  2次更新；只复用普通冒烟的PPO seed和地形生成序列，不新增或污染数据拆分；
- 异常时在关闭仿真器前明确打印traceback（异常调用栈）；
- Shell脚本要求本次运行必须新增且只新增一个`model_1.pt`，无产物即返回失败。

修复后GPU0可用显存约22492 MiB，rough容量冒烟完成：地形生成7.58秒、场景创建136.38秒、
仿真启动48.20秒；第0/1次更新为6.03/4.14秒，采样98304/196608步，边界预警和接触足
越界比例均为0。最终权重为：

`logs/supplementary/multi_terrain_v1_4/capacity_smoke/rsl_rl/pace_task_only_rough_terrain20s_wide_anymal_d/2026-08-10_13-27-00_gpt_multi_terrain_v1_4_stage1_smoke_capacity_rough_ppo_seed900_terrain_seed931900/model_1.pt`

文件大小5237967 bytes，SHA256为
`027b7601109b3d2bac6e9327b46665a815f688ed1dd863046b811314ecf9fa4c`。进程退出后GPU0恢复为
执行前约58547 MiB占用，未残留本项目训练进程。这里只证明当前批量可运行；未连续采集
峰值显存，因此不能把22492 MiB解释成精确显存需求。

## 5. 警告、测试和时间估算

Isaac Sim仍打印`Skipping NVIDIA GPU due CUDA being in bad state`以及
`CUDA_VISIBLE_DEVICES`重映射警告。但设备表明确把物理GPU0标为active，环境设备为
`cuda:0`，GPU PhysX正常启动且所有更新完成；所以本轮为非致命警告，后续每次仍需同时
核对设备表、训练迭代和权重产物，不能单独依据警告文字判断成功或失败。

修复后`tests/pace_eco_lab`共68项通过，Shell语法和`git diff --check`通过。该次冒烟测试
只运行本实验的定向测试集；这属于当时的测试范围记录，不应解释为全仓测试结论。

用仅有的两次更新线性外推，rough任务型单个3000更新约3.5–5.1 GPU·小时，另加约3.2分钟
启动。该样本过短，不能代表长训练稳定速度、PACE-ECO开销、其他地形或正式评估；总体仍
保守按每次3–8 GPU·小时、96次训练288–768 GPU·小时规划，待首个seed0长训练后更新。

## 6. 停止点

v1.4正式训练前GPU门已经关闭。下一门是阶段一五类任务型PPO seed0的3000次B_ref训练；
本报告完成后不自动启动，等待nullptr确认GPU排程。
