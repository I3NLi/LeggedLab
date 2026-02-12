# 2026-02-04_11-47-12
根据readme默认启动
# 2026-02-04_13-38-56
使用beyondMimic中的参数，似乎顺序不对
# 2026-02-04_14-31-48
重新使用原始参数训练
# 2026-02-04_14-56-54
使用BeyondMimic kp kd 继续训练
history——length=10  错误 ->修复回1
# 2026-02-04_17-53-52
history=1,使用beyondimic 参数。从头训练

# 2026-02-05_课程迁移与调试
目标：
将“基于 Mean episode length 提升前向速度”的课程逻辑放到 LeggedLab 训练端，并把课程状态输出到 LeggedLab 终端。

主要改动（LeggedLab）：
- `legged_lab/envs/base/base_config.py`：新增 `EpisodeLengthCurriculumCfg`。
- `legged_lab/envs/base/base_env_config.py`：新增 `episode_length_curriculum` 配置入口。
- `legged_lab/envs/base/base_env.py`：在 `reset()` 中新增 `_update_episode_length_curriculum()`，按已结束 episode 统计轮次并提速，同时打印课程状态。
- `legged_lab/envs/g1/g1_config.py`：`g1_flat` 默认启用该课程参数。

当前课程参数（g1_flat）：
- `round_episode_count = 1024`
- `episode_length_ratio = 1.0`（要求均值接近满 episode）
- `required_streak_rounds = 1`
- `speed_increment = 0.25`
- `max_forward_speed = -1.0`（无上限）
- `print_status = True`

终端输出样例：
`[CURRICULUM][LeggedLab] step=... round=... mean_ep_len=.../... streak=.../... lin_vel_x=(..., ...) cap=inf updates=...`

回退说明（whole_body_tracking）：
- 已撤销 `source/whole_body_tracking/...` 下课程相关改动，课程逻辑仅保留在 LeggedLab。

训练报错与排查（OOM）：
- 现象：`torch.OutOfMemoryError`，8096 env + resume 时显存不足。
- 直接原因：GPU 上有其它进程占用显存，且当前进程总占用过高。
- 建议流程：
1. 先 `nvidia-smi` 清理残留进程。
2. 先用 `--num_envs=4096` 恢复训练稳定性。
3. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低碎片风险。

建议启动命令（稳态）：
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python legged_lab/scripts/train.py --task=g1_flat --logger=tensorboard --num_envs=4096 --headless --resume=True --load_run=2026-02-05_18-19-22 --checkpoint=model_49300.pt`

# 2026-02-06_训练现象
根据终端历史：同一批训练中，部分环境学会了跑动，另有一部分倾向于原地踏步。

# 2026-02-06_00-18-44（模型扩大 & 曲线分析）
模型扩大：actor/critic hidden dims = [1024, 512, 256, 128]（见 `logs/g1_flat/2026-02-06_00-18-44/params/agent.yaml`）。
TensorBoard 观察：
- mean_reward 峰值约 16.67（step≈3875），对应 checkpoint `model_3800.pt` 为全程最佳。
- 课程速度上限提升到 3.25（speed_updates=9），此后最佳 checkpoint 为 `model_14200.pt`。
- mean_episode_length 约 987~1000；out_of_bounds_teleports 约 8~11。

课程逻辑调整：
- 仅统计 “timeout 存活” 的 episode；并要求 mean_reward ≥ 22 才升级课程。

# 2026-02-11_激活函数选择
尝试使用 SiLU，收敛明显变慢；继续使用 ELU 作为默认激活函数。

# 2026-02-11_g1_flat 训练日志补充
日志目录：`/home/hiyio/LeggedLab/logs/g1_flat/2026-02-10_19-15-07`
最高 `Train/mean_reward` ≈ 22.10（step≈21567，对应 checkpoint 约 `model_21500.pt`）。

# 2026-02-11_g1_flat 训练分析补充
- `speed_updates=0`，课程未提速；主要原因是 `min_mean_reward=30`，当前最高均值约 22.10。

# 2026-02-11_课程分级
- 课程改为分级配置，每级可覆盖 `speed_increment`、`max_forward_speed`、`min_mean_reward` 等参数，最后一档为无限升级（无上限）。

# 2026-02-11_课程细化与自适应权重
- g1_flat 分级课程加入侧向/转向范围控制，并在恢复阶段限制前向/侧向范围。
- 支持在无限提速阶段，当奖励长期达标失败时，缓慢提升 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 权重（带上限）。
- 课程阶段现可覆盖 `lin_vel_x/lin_vel_y/ang_vel_z` 与接触终止延时。

# 2026-02-12_数据准备流程规范化
- 统一在 `whole_body_tracking` 中使用 `scripts/video_to_motion_bundle.sh` 生成训练用 motion npz（视频→SMPL→SMPLX→GMR→CSV→NPZ）。
- 产物按 `motions/<时间>-<视频名称>/` 管理，便于追溯与批量训练。
