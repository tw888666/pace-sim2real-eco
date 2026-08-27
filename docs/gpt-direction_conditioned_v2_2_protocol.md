# Direction-Conditioned v2.2 实验协议

> 协议版本：`gpt-direction-conditioned-v2.2`  
> 冻结时间：2026-08-24 19:40:07 +08:00  
> 状态：主矩阵、迁移规则、固定权重和调度许可已冻结

## 1. 修改原因

v2.1 原计划对 E1/E2、task-only（仅任务奖励）/ECO、五种地形和五个 PPO seed（随机种子）做全因子组合，共 100 个正式模型。当前 flat、rough 已完成全部 40 个模型，stairs 部分完成，但该矩阵存在两个问题：

1. 缺少 Fixed-weight PPO（固定权重近端策略优化）这一常规能效基线，导致 task-only 与 PPO-Lagrangian（拉格朗日近端策略优化）之间缺少固定标量化对照；
2. 继续补齐 E1/E2 全部五地形五 seed 的边际论文价值低于其 GPU 成本，尤其 boxes、slope 尚未开始。

v2.2 因此停止补齐 v2.1 的 100 模型矩阵，保留已有结果，将主比较收敛为三种方法，并把新增资源优先投入 stairs、boxes、slope 三种复杂地形。

## 2. v2.1 与 v2.2 对比

| 项目 | v2.1 | v2.2 |
| --- | --- | --- |
| 主比较变体 | E1 observation_control、E2 directional | 仅 E2 directional |
| 主比较方法 | E1/E2 × task-only/ECO | Task-only、Fixed-weight、PACE-ECO |
| 正式矩阵 | 4 方法组合 × 5 地形 × 5 seed = 100 | 按地形分配 seed，共 53 个模型 |
| 已完成结果 | 44/100，另有 1 个 E1 任务运行中 | 复用其中 22 个 E2 模型 |
| 新训练 | 原计划仍需 55 个，另有 1 个运行中 | 精确新增 31 个 |
| flat/rough | 四组合各 seed1--5 | Task-only/ECO 复用 seed1--5；Fixed 新训 seed1--3 |
| 复杂地形 | 原计划四组合各 seed1--5 | 三方法统一 seed1--3 |
| E1 结果 | 主矩阵 | 保留为架构消融，不再补训 |
| B_ref/B80 | E2 task-only seed0 标定 | 直接沿用已冻结 v2.1 标定，不重新标定 |
| 新训练输出 | `direction_conditioned_v2_1` | `direction_conditioned_v2_2` |

v2.2 当前只冻结 stage1（第一阶段）五种单一地形。v2.1 原计划的 stage2 mixed（第二阶段混合地形）不自动迁移，也不进入当前调度队列；若论文后续确需 mixed，应另行预注册。

## 3. 方法定义

### A. Task-only PPO

- 使用 E2 `directional/task_only`；
- 使用方向观测、世界方向速度奖励和现有普通 PPO；
- 不加入能耗奖励，不使用 B_ref 训练预算；
- flat/rough 的 seed1--5 与 stairs 的 seed1 直接复用 v2.1。

### B. Fixed-weight PPO

- 使用与 E2 ECO 相同的方向条件观测、地形、任务奖励、能耗计算函数和普通 PPO 网络配置；
- 不运行 PPO-Lagrangian，不维护或更新自适应拉格朗日乘子；
- 沿用项目 v1/v1.4 已冻结的 PACE 论文 ANYmal 权重 `W100`；
- 固定正幅值记为 $\lambda_{\mathrm{fixed}}=1.6\times10^{-4}$，在奖励最大化实现中的能耗项系数为 $-1.6\times10^{-4}$；
- 不接受命令行覆盖，不进行网格搜索，也不根据 holdout（留出集）结果回调权重。

需要明确：v2.1 的 calibration（标定）冻结的是 B_ref/B80，而不是固定 $\lambda$。因此 v2.2 的固定权重数值来源是已冻结的 v1/v1.4 W100；v2.1 B_ref 只用于统一评估尺度和 ECO 预算，不伪称为固定权重搜索结果。

### C. PACE-ECO PPO-Lagrangian

- 使用 E2 `directional/eco`；
- 保留现有 PPO-Lagrangian 自适应乘子；
- 每个地形继续使用 v2.1 E2 task-only seed0 calibration 得到的 B80；
- flat/rough 的 seed1--5 与 stairs 的 seed1 直接复用 v2.1。

本次没有修改 reward（奖励）函数、环境定义、E2 Task-only/ECO 算法配置或 v1 baseline（基线）代码。Fixed-weight 仅通过独立配置包装，把已有能耗奖励项设为冻结的 W100，并选择已有普通 PPO runner（训练执行器）。这会有意改变新增 Fixed-weight 模型的优化目标，但不会改变任何已有 Task-only/ECO 权重。

## 4. 冻结实验矩阵

| 地形 | Task-only | Fixed-weight | ECO | 主矩阵数量 | v2.1 复用 | v2.2 新训 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| flat | seed1--5 | seed1--3 | seed1--5 | 13 | 10 | 3 |
| rough | seed1--5 | seed1--3 | seed1--5 | 13 | 10 | 3 |
| stairs | seed1--3 | seed1--3 | seed1--3 | 9 | 2 | 7 |
| boxes | seed1--3 | seed1--3 | seed1--3 | 9 | 0 | 9 |
| slope | seed1--3 | seed1--3 | seed1--3 | 9 | 0 | 9 |
| **合计** | **19** | **15** | **19** | **53** | **22** | **31** |

新增 31 个任务按方法划分为：Task-only 8 个、Fixed-weight 15 个、ECO 8 个；按地形划分为：stairs 7 个、boxes 9 个、slope 9 个、flat 3 个、rough 3 个。

## 5. Seed 策略

- flat、rough：保留已完成 Task-only/ECO seed1--5，以维持这两类地形已有统计强度；Fixed-weight 只训练 seed1--3。
- stairs、boxes、slope：三方法统一使用 seed1--3，形成严格配对比较。
- 同一地形和 PPO seed 的三方法复用 v2.1 的地形 seed 映射：flat 为 `530000 + seed`，rough 为 `531000 + seed`，stairs 为 `532000 + seed`，boxes 为 `533000 + seed`，slope 为 `534000 + seed`。
- PPO seed 与地形 seed 分离；不创建重复运行用于事后挑选。
- flat/rough 的三方法主显著性比较使用共同存在的 seed1--3；Task-only/ECO 的 seed4--5只用于增强这两种方法自身估计，不用于伪造与 Fixed-weight 的五 seed 配对。

## 6. 评估指标

每个模型继续使用 v2.1 已冻结的 4 批 × 50 回合，即 200 个 holdout 回合。三方法使用同一评估地形 seed、初始状态集、方向条件和 B_ref/B80。

主要指标：

- 方向穿越成功率；
- B80 联合合格率，即方向穿越成功且回合能耗不超过逐地形 B80；
- 回合能耗与单位方向进度能耗；
- 超 B80 率。

次要与诊断指标：

- 生存成功率、方向净进度、最大横轨偏离；
- 路径效率、世界速度跟踪 RMSE（均方根误差）；
- 航向误差，只作诊断；
- 电气、机械、势能分量及四足接触协调性指标；
- 真实地形边界审计结果。

论文主比较按“地形 × 方法”报告逐 seed 结果、跨 seed 均值与样本标准差。复杂地形只有 3 个 seed，应同时展示全部 seed 点，不把小样本置信区间表述为强证据。

## 7. 迁移与归档规则

1. v2.1 原 100 模型矩阵标记为 `archived/frozen`（已归档/冻结），禁止调度器启动任何新的 v2.1 补全任务。
2. 协议切换时已经运行的 v2.1 任务可以自然结束，但不会触发下一项 v2.1 任务；其结果按原 v2.1 身份保存。
3. 已完成 checkpoint（检查点）不删除、不移动、不复制、不改名，原路径永久保留。
4. v2.2 通过原任务 ID、方法、地形、PPO seed、唯一非空 `model_2999.pt` 和复现配置引用 22 个 v2.1 E2 模型。
5. 已完成的 E1 模型继续保留，可用于 appendix（附录）中的方向观测架构消融，但不进入 v2.2 三方法主表。
6. 所有新增模型写入 `logs/supplementary/direction_conditioned_v2_2/`；禁止把新增模型写回 v2.1 根目录。
7. v2.2 Fixed-weight/ECO 不重新搜索 W100 或 B80；holdout 结果不得用于回头调参。

## 8. 调度规则与待训练列表

事件驱动调度器只读取 `/home/xy.chen/tw/experiment-manager/gpt-方向条件v2.2训练计划.json` 中的 31 个任务。启动顺序为 stairs、boxes、slope、flat Fixed、rough Fixed。启动器和 run name（运行名）都带 `v2_2`，调度器还会拒绝任何不以 `gpt_direction_v2_2_` 开头的候选任务。

待训练汇总：

- stairs：Task-only seed2--3，Fixed-weight seed1--3，ECO seed2--3，共 7 个；
- boxes：三方法 seed1--3，共 9 个；
- slope：三方法 seed1--3，共 9 个；
- flat：Fixed-weight seed1--3，共 3 个；
- rough：Fixed-weight seed1--3，共 3 个。

## 9. GPU 预计需求

新增训练总量为 31 个完整的 4096 环境、3000 次更新任务，即 93,000 次单任务迭代总量。

根据 49 个可审计 v2.1 完成任务从运行目录时间到 `model_2999.pt` 的经验统计，单任务中位墙钟时间约 4.0 小时；stairs 中位约 6.3 小时。boxes/slope 尚无正式完成样本，因此只能给区间估计：

- 预计总需求约 125--195 GPU 小时；
- 5 张 GPU 理想并行约 25--39 小时；
- 6 张 GPU 理想并行约 21--33 小时。

实际时间会受 A100/V100 混用、主机内存与 I/O 竞争影响，上述不是完成时间承诺。长训练继续使用 tmux（终端复用会话）。

## 10. 与论文实验设计的对应关系

| 论文问题 | 对应实验 |
| --- | --- |
| 不考虑能耗时，任务性能上限如何 | Task-only PPO |
| 固定标量化是否已足够 | Fixed-weight PPO，W100 不自适应 |
| 约束优化是否比固定权重更稳定地满足预算 | PACE-ECO PPO-Lagrangian |
| 三方法在复杂地形是否仍成立 | stairs、boxes、slope 三方法配对 seed1--3 |
| 方向条件 E2 是否必要 | 已有 E1/E2 结果作为独立架构消融，不继续补全 |
| 结果是否来自同一任务和评估协议 | 三方法共享 E2 环境、地形 seed、初始状态、指标和 B_ref/B80 评估尺度 |

主论文使用 v2.2 三方法矩阵；v2.1 E1 完成结果只能作为补充消融。不得把 v2.1 未完成的 100 模型矩阵描述为完成实验。

## 11. 冻结文件与入口

- 机器可读协议：`gpt-方向条件v2.2冻结配置.json`
- v2.1 调度归档：`/home/xy.chen/tw/experiment-manager/gpt-方向条件v2.1矩阵归档.json`
- v2.2 调度计划：`/home/xy.chen/tw/experiment-manager/gpt-方向条件v2.2训练计划.json`
- v2.2 训练入口：`scripts/pace_eco/gpt-运行方向条件v2.2训练.sh`
- v2.2 评估入口：`scripts/pace_eco/gpt-运行方向条件v2.2评估.sh`
- v2.2 holdout 授权入口：`scripts/pace_eco/freeze_direction_conditioned_v2_2_holdout.py`

文档文件名按仓库 `AGENTS.md` 约定添加了 `gpt-` 前缀；其主体名称对应请求的 `direction_conditioned_v2_2_protocol.md`。
