# 方向条件能效运动 v2.1 冻结实施方案

## 1. 研究边界

项目名称冻结为 **Direction-Conditioned Energy-Efficient Locomotion on Complex Terrains（方向条件约束的复杂地形能效四足运动）**。

研究问题是：在保持 PACE-ECO 能耗约束的情况下，使机器人沿指定世界方向穿越地形。本阶段不声称解决完整导航、任意方向泛化或目标点到达问题。

v1 分支 `exp/multi-terrain-comparison-v1` 是历史基线，reward（奖励）、evaluator（评估器）、B80 和 checkpoint（检查点）均保持冻结。v2 使用独立分支 `exp/direction-conditioned-locomotion-v2`、任务 ID、训练目录、评估目录、地形 seed（随机种子）和评估协议。

## 2. DirectionCommand（方向指令）

v2.1 冻结世界目标方向和速度为：

$$
\mathbf d_w=(1,0), \qquad s^*=1\ \mathrm{m/s}.
$$

策略观测使用机器人坐标系中的方向：

$$
\mathbf d_b=R_{bw}\mathbf d_w.
$$

command 输出冻结为 `[direction_b_x, direction_b_y, target_speed]`。它替换原策略观测中的三维速度 command 槽位，因此 actor（策略网络）和 critic（价值网络）的输入维数仍为 48 和 353。但语义已改变，v1 checkpoint 不得恢复训练为 v2。

保留原 `base_velocity` command 作为 E1 的隐藏奖励参考，它不输入 v2 策略观测。

## 3. 奖励冻结

E2 只替换线速度奖励的参考坐标系。世界水平面目标速度为：

$$
\mathbf v_w^*=s^*\mathbf d_w.
$$

方向速度奖励为：

$$
r_{\mathrm{dir}}=
\exp\!\left(-\frac{\lVert\mathbf v_{w,xy}-\mathbf v_w^*\rVert^2}{\sigma^2}\right),
\qquad \sigma=0.5.
$$

只跟踪世界水平速度 `root_lin_vel_w[:, :2]`，不惩罚上下坡必需的竖直速度。原 PACE 组合项拆为等尺度的两项：

$$
R_{任务}=0.2r_{\mathrm{dir}}+0.2r_{\omega}.
$$

在完美跟踪时仍为 $0.4$，与 v1 的 $0.2(1+1)=0.4$ 一致。FTD（足端触地）、碰撞、yaw rate（偏航角速度）和 energy cost（能耗代价）的定义不变。v2.1 不加 heading reward（航向奖励）。

## 4. 实验矩阵

| 编号 | 方向观测 | 世界方向速度奖励 | 航向奖励 | 用途 |
| --- | --- | --- | --- | --- |
| E0 | 否 | 否 | 否 | 冻结 v1 历史基线，仅用 v2 指标交叉分析 |
| E1 | 是 | 否 | 否 | 观测/网络架构对照，必须重新训练 |
| E2 | 是 | 是 | 否 | v2.1 主要方法，必须重新训练 |
| E3 | 是 | 是 | 是 | 预注册后续消融，航向权重未冻结前不实现 |

v2.1 只实现 E1 和 E2，每个实验包含 `task_only` 和 `eco`。不加入固定权重方法，不将 Motion Prior residual（运动先验残差）列为当前实验。

E1 和 E2 共享网络结构、PPO seed、地形 seed、初始状态集和训练更新数，以形成配对对照。

## 5. 方向穿越评估

所有位移都相对每回合实际起点计算。方向进度为：

$$
p_{\parallel}=(\mathbf p_T-\mathbf p_0)\mathbin{\cdot}\mathbf d_w.
$$

横轨偏离为轨迹在与 $\mathbf d_w$ 垂直方向上的最大绝对投影。方向穿越成功冻结为：

$$
\mathrm{DirectionalSuccess}=
\mathrm{Timeout}_{20s}
\land \neg\mathrm{IllegalTermination}
\land p_{\parallel}\ge 16\ \mathrm m
\land e_{\perp,\max}\le 3\ \mathrm m.
$$

正式评估同时报告：

- v1 历史机身速度成功，仅作兼容诊断；
- 生存成功与方向穿越成功；
- 方向净进度、最大横轨偏离、水平路径长和路径效率；
- 世界目标速度跟踪 RMSE（均方根误差）；
- 机身 heading error（航向误差），只作诊断；
- 回合能耗、单位方向进度能耗、能耗均值/标准差/置信区间与超预算率；
- 方向穿越成功且能耗不超 B80 的联合合格率；
- v1 已冻结的四足边界审计与协调性诊断指标。

v2.1 不预先承诺航向误差下降。主结论只能是轨迹漂移是否下降，以及能耗可行性是否保持。

## 6. B_ref 和 B80

v2 不沿用 v1 的 B_ref/B80。每个阶段使用 E2 `directional/task_only` seed0 的 calibration（标定）结果，只对方向穿越成功回合求平均能耗：

$$
B_{80}=0.8B_{\mathrm{ref}}.
$$

资格门槛为每个地形的方向穿越成功率不低于 95%。同一阶段的 E1-ECO 和 E2-ECO 共享该 E2 任务型基准导出的 B80，避免为每个方法单独改变约束。

E0 继续使用它训练时的 v1 预算。若在 v2 B80 下报告 E0，必须明确标为交叉协议诊断，不得冒充 E0 的训练预算。

## 7. 冻结实施顺序

1. 实现独立 DirectionCommand、E1/E2 环境配置和等尺度奖励。
2. 注册独立任务 ID，冻结训练角色、PPO seed 与地形 seed。
3. 实现独立 v2 评估器，使用自动 reset 前状态快照计算真实终点。
4. 实现 v2 B_ref/B80 冻结与 holdout（留出集）授权。
5. 增加 Shell（命令行脚本）守卫、纯 CPU 测试和配置静态审计。
6. 实现通过后由 nullptr 在 tmux 中依次执行容量冒烟、E2 任务型 seed0、calibration、B80 冻结、E1/E2 正式训练和 holdout 评估。

## 8. 与 Motion Priors Reimagined 的关系

论文表述冻结为：

> Inspired by the goal-conditioned task formulation of Motion Priors Reimagined, we introduce a direction-conditioned locomotion objective without adopting its hierarchical motion-prior architecture.

本工作没有冻结低层运动先验、高层残差策略或分层架构，因此不使用 “Motion Prior Guided” 的方法名称。
