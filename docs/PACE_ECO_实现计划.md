# PACE + ECO 纯仿真实现计划

## 研究边界

目标是在辨识后的 ANYmal-D 仿真中，将 PACE 的固定能耗奖励替换为单约束 PPO-Lagrangian（近端策略优化—拉格朗日），比较它是否比固定权重方法更稳定地满足 PACE 模型能量预算。没有真实机器人，因此不主张真实电池能耗约束或真实 Sim-to-Real（仿真到现实）保证。

固定软件基线：

- Isaac Lab commit（提交）`b4c321024792976150ca55fddb26fa34480d974e`。
- `rsl-rl-lib==5.0.1`。
- PACE 本地源码基线 commit `0703812`；远端网络抓取失败，所以它是已有官方源码快照，不宣称完整等同 GitHub 当前 `617e4e2`。
- 官方 `fitting.npy` SHA-256 为 `4436941fa5e9a5e8e1ef93d55956fcffdb4c4c4526b8ef3e145e6ea619fbe1c8`。

## 完整数据流

1. 从官方 `fitting.npy` 读取 49 维辨识参数 (p)。
2. 按 12 个 armature（等效转子惯量）、12 个 viscous friction（粘性摩擦）、12 个 static/dynamic friction（静/动摩擦）、12 个 encoder bias（编码器偏置）和 1 个 delay（延迟）冻结 ANYmal-D 仿真。
3. 使用 400 Hz 物理仿真、50 Hz 策略；每个策略步累计全部 8 个物理子步能量。
4. reward（奖励）只保留速度跟踪、碰撞和 FTD（足端触地速度）惩罚。
5. cost（代价）使用 PACE 物理能量；摔倒回合只在训练代价中补到 1.25 屏障，屏障不计入模型焦耳或回合能量违反比例。
6. PPO-Lagrangian 同时训练 reward critic（奖励价值网络）和 cost critic（代价价值网络）；乘子由独立 256 回合 dual-eval（对偶评估）更新。

功率按论文公式实现：

\[
P_{el}=\sum_j 0.0192\tau_j^2,\qquad
P_{mech}=\begin{cases}\tau^T\dot q,&\tau^T\dot q\ge0\\0,&\tau^T\dot q<0\end{cases},\qquad
P_{pot}=\sum_bm_bgv_{b,z}.
\]

控制步能量为：

\[
E_k=\sum_{s=1}^{8}0.0025(P_{el,k,s}+P_{mech,k,s}+P_{pot,k,s}).
\]

## 实现阶段和门槛

- [x] 建立独立 branch（分支）`codex/pace-eco-ppolag` 和本地源码基线。
- [x] 冻结官方 49 维参数，并明确 `q_enc=q_sim-b`；reset 使用 `q_sim=q0+b`。
- [x] 实现 400 Hz 能量累计、物理代价和失败屏障。
- [x] 适配 RSL-RL 5.0.1：独立 cost critic、cost GAE（广义优势估计）、`1+lambda` actor loss（策略损失）归一化。
- [x] 实现不可变 checkpoint（检查点）、cycle ID（周期标识）、原子 JSON 和幂等乘子更新。
- [ ] 用户执行 32 环境 GPU smoke test（冒烟测试），核对 8 个子步、task 注册和张量形状。
- [ ] 先训练/评估无能耗约束基线，标定 100% 模型能量预算，再生成 90% 和 80% 两档。
- [ ] cost critic 连续两次满足：初态均值偏差不超过 0.05、explained variance（解释方差）不低于 0.5；失败则共享地把 rollout（采样段）从 24 增至 48，再到 64。
- [ ] 验证连续 200 次更新与两段 100 次更新没有显著差异。
- [ ] 冻结开发超参数后，正式运行种子 0、1、2；最终 checkpoint 使用相同的 200 个随机初态做一次配对测试。

## 本地 CPU 检查

```bash
conda activate env_isaaclab
cd /home/xy.chen/tw/isaaclab_ws/pace-sim2real
PYTHONPATH=$PWD/source/pace_sim2real python -m pytest -q tests
```

这些测试不启动 Isaac Sim，也不使用 GPU。

## 用户执行的 GPU 冒烟测试

以下命令使用物理编号 3 的 V100；进程内仍写 `cuda:0`：

```bash
conda activate env_isaaclab
cd /home/xy.chen/tw/isaaclab_ws/pace-sim2real
CUDA_VISIBLE_DEVICES=3 python scripts/rsl_rl/train.py \
  --task Isaac-Pace-Eco-Anymal-D-Flat-v0 \
  --num_envs 32 \
  --max_iterations 2 \
  --headless \
  --device cuda:0
```

冒烟测试只检查代码通路，生成的权重不能作为实验结果。

## 正式分段训练与对偶更新

主配置每 100 次 PPO update（更新）做一次 dual-eval；开发诊断可用 50 次。每轮顺序是：

1. 训练 100 次更新并保存 checkpoint。
2. 冻结 checkpoint：

```bash
python scripts/pace/freeze_dual_checkpoint.py \
  --checkpoint <model.pt> \
  --output_dir <run>/dual \
  --cycle_id 0 \
  --budget_j <B_E>
```

3. 用户在空闲 GPU 上运行独立 256 回合评估：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/pace/dual_eval.py \
  --request <run>/dual/cycle_0000/request.json \
  --num_envs 256 \
  --headless \
  --device cuda:0
```

4. CPU 进程幂等更新一次乘子：

```bash
python scripts/pace/apply_dual_eval.py \
  --state <run>/dual/state.json \
  --result <run>/dual/cycle_0000/result.json \
  --budget_j <B_E> \
  --lambda_lr 0.05
```

5. 下一段训练同时传入 `--resume --checkpoint <model.pt> --dual_state <run>/dual/state.json --max_iterations 100`。评估进程只读不可变 checkpoint，不修改训练 checkpoint。

正式实验前用下面命令检查连续两次 cost critic 门槛：

```bash
python scripts/pace/check_cost_critic_gate.py \
  <cycle_n/result.json> <cycle_n_plus_1/result.json>
```
