# PACE-ECO 实验元数据模板

## 目的

每次新训练使用一个不可覆盖的实验目录，自动固定代码、配置、seed、物理 GPU 和输出位置，避免再次出现代码、权重和报告无法对应的问题。

创建脚本：

`scripts/pace/create_experiment.py`

默认实验资产根目录：

`/home/xy.chen/tw/experiment-data/PACE-ECO`

## 标准目录

```text
<实验名称>/
├── README.md
├── metadata.yaml
├── config.yaml
├── commit.txt
├── git-status.txt
├── checkpoints/
├── evaluation/
├── logs/
├── videos/
└── figures/
```

其中：

- `README.md`：适合人工阅读的摘要；
- `metadata.yaml`：机器可读的完整元数据；
- `config.yaml`：创建时配置文件的逐字节快照；JSON 配置也是有效的 YAML 1.2；
- `commit.txt`：完整 Git commit（提交）哈希；
- `git-status.txt`：创建时工作树状态；
- 五个子目录：分别保存权重、评估、日志、视频和图表。

## 平地示例

下面只创建实验目录并查询 GPU 信息，不会启动训练：

`python scripts/pace/create_experiment.py --name flat-v2-ppo-lagrangian-seed3 --repo /home/xy.chen/tw/PACE-ECO --config /绝对路径/flat-v2.yaml --algorithm PPO-Lagrangian --train-seed 3 --terrain flat --gpu 0 --iterations 3000`

本机物理 GPU0 和 GPU1 是 A100 80GB，GPU2–GPU5 是 V100 32GB；脚本仍会在创建时使用 `nvidia-smi` 自动查询并记录实际型号、显存、UUID 和驱动版本。

## 多地形示例

`python /home/xy.chen/tw/PACE-ECO/scripts/pace/create_experiment.py --name multi-terrain-v1_4-rough-ppo-lagrangian-seed1 --repo /home/xy.chen/tw/PACE-ECO-multi-terrain --config /home/xy.chen/tw/PACE-ECO-multi-terrain/gpt-多地形补充实验冻结配置.json --algorithm PPO-Lagrangian --train-seed 1 --terrain-seed 331001 --terrain rough --gpu 0 --iterations 3000 --budget-j 17015.6609`

## 安全规则

- 默认拒绝从有未提交修改的工作树创建正式实验；开发实验必须显式使用 `--allow-dirty`，状态会完整写入 `git-status.txt`；
- 已存在的实验目录永不覆盖；需要新实验时使用新的名称；
- `--gpu` 表示物理 GPU 编号，生成的 README 会明确写出“GPU 几”和实际型号；
- `metadata.yaml` 中的显存占用和利用率只是创建瞬间快照，正式启动训练前仍必须检查占用、ECC 和 Xid；
- 脚本不运行训练，也不设置全局环境变量；正式命令仍使用 `CUDA_VISIBLE_DEVICES=<物理编号>` 前缀；
- 超过 1000 次迭代的正式训练仍必须由 nullptr 在 tmux 或 GNU Screen 中启动。
