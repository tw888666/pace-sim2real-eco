# PACE-ECO Terrain20sWide 下一阶段单行命令

协议：`gpt-multi-terrain-v1.4`。工作树：
`/home/xy.chen/tw/PACE-ECO-multi-terrain`。以下均为可直接复制的单行命令；GPU0–5均须在启动前检查占用，GPU5启用证据见[归档记录](../archive/PACE-ECO-multi-terrain-legacy/gpt-GPU5重启后恢复验证与启用记录.md)。
当前只执行第1–3条非GPU关口，执行第4条以后必须先得到nullptr确认并重新检查GPU占用。

## 当前非 GPU 关口

1. `cd /home/xy.chen/tw/PACE-ECO-multi-terrain && CUDA_VISIBLE_DEVICES='' PACE_ECO_DATA_ROOT=/home/xy.chen/tw/PACE-ECO/pace_data PYTHONPATH=/home/xy.chen/tw/PACE-ECO-multi-terrain /home/xy.chen/miniconda3/bin/conda run -n env_isaaclab_v100 --no-capture-output python scripts/pace_eco/audit_multi_terrain_geometry.py --output gpt-多地形Terrain20sWide几何静态审计.json`
   - 使用GPU：否；生成/修改模型权重：否；生成评估结果：否，只生成CPU几何审计JSON；可能影响现有实验：否，但同名文件存在时会拒绝覆盖。

2. `cd /home/xy.chen/tw/PACE-ECO-multi-terrain && CUDA_VISIBLE_DEVICES='' PACE_ECO_DATA_ROOT=/home/xy.chen/tw/PACE-ECO/pace_data PYTHONPATH=/home/xy.chen/tw/PACE-ECO-multi-terrain /home/xy.chen/miniconda3/bin/conda run -n env_isaaclab_v100 --no-capture-output python -m pytest -q tests/pace_eco_lab && python -m compileall -q pace_eco_lab scripts/pace_eco && bash -n scripts/pace_eco/_多地形公共.sh scripts/pace_eco/运行多地形训练.sh scripts/pace_eco/运行多地形冒烟.sh scripts/pace_eco/运行多地形评估.sh scripts/pace_eco/运行多地形状态审计.sh && python -m json.tool gpt-多地形补充实验冻结配置.json >/dev/null && git diff --check`
   - 使用GPU：否，显式屏蔽；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：否，只产生测试缓存和终端输出。

3. `cd /home/xy.chen/tw/PACE-ECO-multi-terrain && git status --short --branch && nvidia-smi && (tmux list-sessions || true) && ps -eo pid,user,etimes,cmd --sort=pid | rg 'scripts/pace_eco/(train|eval|multi_terrain_eval|audit_multi_terrain_sim_state)\.py' || true`
   - 使用GPU：否，仅查询；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：否。

完成审阅后由nullptr决定是否Git commit（提交）；不自动commit或push（推送）。3000次脚本会
拒绝脏工作树。

## 确认后：v1.4 五类 GPU 冒烟

4. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形冒烟.sh 0 stage1 flat`
   - 使用GPU：是，示例GPU0；生成/修改模型权重：生成2次更新的隔离smoke权重；生成评估结果：否；可能影响现有实验：占用所选GPU，只写`multi_terrain_v1_4/smoke`。

5. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形冒烟.sh 0 stage1 rough`
   - 使用GPU：是；生成/修改模型权重：生成smoke权重；生成评估结果：否；可能影响现有实验：同第4条。

6. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形冒烟.sh 0 stage1 stairs`
   - 使用GPU：是；生成/修改模型权重：生成smoke权重；生成评估结果：否；可能影响现有实验：同第4条。

7. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形冒烟.sh 0 stage1 boxes`
   - 使用GPU：是；生成/修改模型权重：生成smoke权重；生成评估结果：否；可能影响现有实验：同第4条。

8. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形冒烟.sh 0 stage1 slope`
   - 使用GPU：是；生成/修改模型权重：生成smoke权重；生成评估结果：否；可能影响现有实验：同第4条。

每条前都用`nvidia-smi`重新选择空闲GPU0–5；命令中的0只是示例。同一卡不并行本项目任务。

## 冒烟通过后：实际碰撞网格状态审计

9. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形状态审计.sh 0 flat`
   - 使用GPU：是；生成/修改模型权重：否；生成评估结果：否，只生成审计JSON；可能影响现有实验：占用所选GPU并写v1.4隔离审计目录。

10. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形状态审计.sh 0 rough`
    - 使用GPU：是；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：同第9条。

11. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形状态审计.sh 0 stairs`
    - 使用GPU：是；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：同第9条。

12. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形状态审计.sh 0 boxes`
    - 使用GPU：是；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：同第9条。

13. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形状态审计.sh 0 slope`
    - 使用GPU：是；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：同第9条。

## 状态审计通过后：阶段一 seed0

14. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形容量冒烟.sh 0 stage1 rough`
    - 使用GPU：是；生成/修改模型权重：只生成4096环境×2次更新的隔离容量冒烟权重，禁止评估；生成评估结果：否；可能影响现有实验：短时占用所选GPU，只写`multi_terrain_v1_4/capacity_smoke`。

15. `tmux new-session -s pace_mt_v14_bref`
    - 使用GPU：否；生成/修改模型权重：否；生成评估结果：否；可能影响现有实验：只创建可恢复终端会话。

16. `bash /home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/运行多地形训练.sh 0 stage1 budget_train flat task_only 0 -`
    - 使用GPU：是；生成/修改模型权重：是，从头生成3000次训练和`model_2999.pt`；生成评估结果：否；可能影响现有实验：占用所选GPU，只写`multi_terrain_v1_4`，不覆盖历史结果。

rough、stairs、boxes、slope的其余4个seed0命令仅替换第16条的地形参数。必须先根据冒烟和
状态审计的实测显存/速度确定并行安排；不得提前排队75个正式模型。seed0训练全部完成后，
再用`运行多地形评估.sh`执行每类4批calibration；脚本内部自动串行4批，单次公开命令不变。
