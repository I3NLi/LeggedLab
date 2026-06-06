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

# 2026-02-12_10-48-42
- 放弃 SiLU，恢复默认激活函数（ELU）。
- 训练不移动的主要原因：课程把 forward_speed_max 提到极高（无上限 + 速度增量），命令分布严重失真；且阶段覆盖导致终止延时一直停留在 2s。
- 计划：在课程中加入 [2,3] 速度区间的专项阶段，并在最终阶段放开变速但加速度上限（避免继续无限增大）。

# 2026-02-12_10-57-05
- 需求：保留 rel_standing_envs，不关闭；增加多级课程，逐步提速到 6 m/s；在 2~3 m/s 区间重点训练后再放开变速。
- 计划：重做 g1_flat 课程阶段，细化速度区间（含 2~3 m/s 多阶段），最终阶段放开到 [-2, 6] 并加硬上限 6。

# 2026-02-12_15-13-42
- play.py 中 `--play_lin_vel_x` 被课程阶段覆盖：g1_flat 启用课程后，reset 会将 `lin_vel_x` 重置为当前 stage 的范围（例如 Stage0 的 -0.6~0.8），导致随机速度很低。
- 另外 `rel_standing_envs=0.2` 会让 20% 环境强制站立，单看一个 env 容易误判“不动”。
- 若想在 play 中按参数采样速度，建议禁用课程（`env_cfg.episode_length_curriculum.enable=False`），或固定速度区间。

# 2026-02-12_15-24-07
- 按用户要求保留当前 g1_flat 课程的本地修改（包含新增阶段与参数调整），并提交。

# 2026-02-12_15-25-17
- play.py 禁用课程：`env_cfg.episode_length_curriculum.enable = False`，避免 play 参数被阶段覆盖。

# 2026-02-12_21-39-47
- 分析近两次训练日志（2026-02-12_19-55-06 / 2026-02-12_15-28-09）：当课程进入 3.5~4.0 m/s（stage 8）后，`Episode_Reward/track_lin_vel_xy_exp` 均值显著下降（约 0.41），同时 energy/action_rate 负项幅度变小，说明机器人趋向“少动/不动”。
- 结论：高于 ~3.5 m/s 时跟踪奖励在当前 std=0.5 的指数形式下快速饱和，难以提供梯度；再加上能量/动作惩罚，最优策略变为原地站立。
- 另有干扰：课程中含混合区间（如 -1~3、-1~4）和 `rel_standing_envs=0.2`，导致观察到的速度分布包含大量低速/零速。

# 2026-02-12_22-03-48
- 采用修复方案 B（仅 g1）：提高跟踪奖励 std（lin_xy 1.0, ang_z 0.8），高速度阶段提高跟踪权重，并降低 energy/action_rate 惩罚约 50%。

# 2026-02-13_09-02-16
- 课程更新：针对 >4.5 m/s 不迈步问题，细化 4.5~6.0 区间为更窄速度段并延长停留；收紧高速侧向/转向范围。
- 3 m/s 以上放松胳膊与能量惩罚：新增 stage 覆盖 joint_deviation_arms 权重，并进一步减小 energy 惩罚。

# 2026-02-26_g1_flat_6m_手臂乱飞调试
- 参考日志目录：`/home/hiyio/LeggedLab/logs/g1_flat/2026-02-25_22-42-48`
- 现象：
  - 课程可推进到高速阶段，策略能够跑到约 6 m/s（并进入更高速度开放区间）。
  - 但出现明显“手乱飞/上肢摆动过大”。
- 观测结论：
  - 当时训练中 `Episode_Reward/joint_deviation_arms` 约束不足（接近无约束），高速阶段更容易用手臂补偿姿态与速度误差。
- 本次调整：
  - 在课程与奖励配置中加入/恢复手部控制约束，重点对 `joint_deviation_arms` 进行限制后重新开跑验证。
  - 当前采用默认约束强度：`joint_deviation_arms.weight = -0.2`（与仓库默认对齐）。
- 下一步验证指标：
  - `Episode_Reward/joint_deviation_arms` 是否从“接近 0 约束”转为可观测惩罚。
  - 高速段 `Train/mean_reward`、`Train/mean_episode_length` 是否保持稳定。
  - `Env/out_of_bounds_teleports` 是否下降。

# 2026-02-26_g1_flat_新一轮测试开始（按当前 g1_config）
- 配置来源：`/home/hiyio/LeggedLab/legged_lab/envs/g1/g1_config.py`
- 本轮核心配置：
  - `joint_deviation_arms.weight = -0.2`（恢复手部约束默认强度）。
  - 课程启用，`round_episode_count = 6000`。
  - 分阶段课程 `max_updates=1`（每完成一轮统计即推进下一 stage），最终阶段 `max_updates=-1`。
  - 课程中加入髋关节约束调度：`joint_deviation_hip_weight` 从 `-0.2` 逐步放松到 `-0.02`。
  - 高速阶段跟踪权重提升，最终阶段 `track_lin_vel_xy_exp_weight=4.0`、`track_ang_vel_z_exp_weight=4.0`。
- 测试目标：
  - 保持 6 m/s 附近速度能力。
  - 抑制“手乱飞”现象，减少上肢不必要摆动。
- 本轮重点观测指标：
  - `Episode_Reward/joint_deviation_arms`
  - `Train/mean_reward`、`Train/mean_episode_length`
  - `Curriculum/stage_index`、`Curriculum/forward_speed_max`
  - `Env/out_of_bounds_teleports`
- 启动命令（记录）：
  - `python legged_lab/scripts/train.py --task=g1_flat --num_envs=4096 --headless --logger=tensorboard`

# 2026-02-26_g1_flat_最终日志分析（2026-02-26_21-30-20）
- 日志目录：`/home/hiyio/LeggedLab/logs/g1_flat/2026-02-26_21-30-20`
- 备注：`2026-02-26_23-06-00` 仅有 3 个数据点（step 0~2），属于短启动记录，不作为主分析对象。

- 收敛情况：
  - `Train/mean_reward`：最终约 `46.26`（峰值约 `50.02`）。
  - `Train/mean_episode_length`：最终约 `993.14`，已接近满时长 `1000`。
  - 当前训练总体稳定，未出现明显发散。

- 课程推进：
  - `Curriculum/round_index = 80`
  - `Curriculum/stage_index = 16`（最终无限阶段）
  - `Curriculum/stage_progress = 55`
  - `Curriculum/forward_speed_max = 8.0`
  - 解释：当前 `max_updates` 配置在前置阶段的累计门槛约为 25 轮，因此 round 到 80 时已在最终阶段持续训练。

- 手部控制相关指标：
  - `Episode_Reward/joint_deviation_arms` 最终约 `-0.049`（全程最小约 `-0.254`）。
  - `Episode_Reward/joint_deviation_hip` 最终约 `-0.232`。
  - 说明：手臂偏离惩罚已经生效，不再是接近 0 的无约束状态。

- 风险与后续：
  - 本轮课程推进依然较快（约 step 1210 即进入 stage 16），后续若希望在中高速阶段训练更久，建议提高中段 `max_updates` 或进一步增大 `round_episode_count`。

# 2026-06-06_env_isaacsim51_对齐与提交策略
- 当前整合分支：`magicbot-z1-support`。
- 已提交代码快照：`0bcafb8 Align G1 training with Isaac Sim 5.1`。
- 目标：让 LeggedLab 的 `g1_flat` 对齐正在运行的 `env_isaacsim51`，并保持后续修改可快速回退。

环境链路：
- Python：`/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python`
- Isaac Sim：5.1
- IsaacLab：3.3
- rsl_rl：新版 TensorDict observation API
- 推荐运行时环境变量：
  - `OMNI_KIT_ACCEPT_EULA=YES`
  - `PYTHONNOUSERSITE=1`
  - `PYTHONPATH=/home/hiyio/LeggedLab`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

验证命令：
```bash
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -m py_compile \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/base/base_env.py \
  legged_lab/envs/base/base_env_config.py \
  legged_lab/envs/g1/g1_config.py \
  legged_lab/scripts/train.py

git diff --cached --check
```

已跑通的 smoke train：
```bash
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task=g1_flat \
  --logger=tensorboard \
  --num_envs=64 \
  --device=cuda:0 \
  --kit_args=--portable
```

运行结果：
- 非 headless GUI 正常启动。
- GPU PhysX 未复现 Isaac Sim 4.5 下的 kernel error。
- 已进入 learning iteration，确认运行到 iteration 132。
- 训练进程记录：`PID 2473639`，后续已退出。
- 日志目录：`/home/hiyio/LeggedLab/logs/g1_flat/2026-06-06_10-49-41`。
- 退出原因：
  - `ReferenceError: weakly-referenced object no longer exists`
  - 触发点在 IsaacLab command manager 更新 velocity metrics 时读取 `robot.data.root_lin_vel_b`。
  - 结论：`env_isaacsim51` 已经跑通训练入口和 rollout，但长稳训练还需要单独修复 IsaacLab 5.1/PhysX articulation data 生命周期问题。

后续提交纪律：
- 每完成一个可验证修改就提交一次，避免大杂烩 commit。
- 功能探索走 `feat/*` 分支，窄修复走 `fix/*` 分支。
- 运行记录和对话日志默认不混入功能提交；文档更新单独提交。

建议分支：
- `feat/z1-deploy-native-sdk`：Z1 native SDK 部署代码。
- `feat/z1-assets-leggedlab`：Z1 asset 与 LeggedLab 适配。
- `feat/z1-whole-body-tracking`：Z1 whole_body_tracking 链路。
- `feat/g1-flat-plane-curriculum`：G1 真平面走路/跑步课程。
- `feat/g1-gravel-terrain`：G1 gravel/generator 小地形训练。
- `feat/isaacsim51-compat`：Isaac Sim 5.1 兼容层。
