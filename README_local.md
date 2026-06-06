# LeggedLab 本地使用指南（Local）

**适用范围**
本指南面向单机本地训练/测试（非多机、多 GPU 分布式）。

**准备环境**
1. 安装 Isaac Lab（推荐按官方文档使用 conda 方式）。
2. 使用同一 Python 环境安装本仓库：

```bash
cd /home/hiyio/LeggedLab
pip install -e .
```

**本地训练**
当前推荐环境为 `env_isaacsim51`，对应 Isaac Sim 5.1 / IsaacLab 3.3。运行前保持 `PYTHONPATH` 指向本仓库，避免其它 Python 包路径污染：

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

**MagicBot Z1 Locomotion**

Z1 任务名：

- `magicbot_z1_flat`: 平地 locomotion，日志目录 `logs/magicbot_z1_flat/`。
- `magicbot_z1_rough`: rough terrain locomotion，日志目录 `logs/magicbot_z1_rough/`。

安装/更新本仓：

```bash
cd /home/hiyio/LeggedLab
PYTHONNOUSERSITE=1 /home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -m pip install -e .
```

Z1 小规模训练 smoke test：

```bash
cd /home/hiyio/LeggedLab

OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task=magicbot_z1_flat \
  --logger=tensorboard \
  --num_envs=64 \
  --headless \
  --device=cuda:0 \
  --kit_args=--portable
```

Z1 恢复训练示例：

```bash
cd /home/hiyio/LeggedLab

OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task=magicbot_z1_flat \
  --num_envs=4096 \
  --headless \
  --resume=True \
  --load_run=latest \
  --checkpoint=latest \
  --logger=tensorboard \
  --device=cuda:0 \
  --kit_args=--portable
```

Z1 回放并导出策略文件（本机已验证）：

```bash
cd /home/hiyio/LeggedLab

OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/play.py \
  --task=magicbot_z1_flat \
  --headless \
  --export_only \
  --num_envs=1 \
  --load_run=2026-06-06_12-13-53_z1_flat_8192_rebuild_20260606 \
  --checkpoint=model_7500.pt \
  --device=cuda:0 \
  --kit_args=--portable
```

导出结果位置：

```text
/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-06_12-13-53_z1_flat_8192_rebuild_20260606/exported/
```

包含 `policy.pt`、`policy.onnx` 和 `policy.onnx.data`。如果需要可视化回放，去掉 `--headless --export_only`，并按显存情况调小 `--num_envs`。

常规 headless 训练命令（G1 平地）：

```bash
python legged_lab/scripts/train.py \
  --task=g1_flat \
  --num_envs=4096 \
  --headless \
  --logger=tensorboard
```

恢复训练示例：

```bash
python legged_lab/scripts/train.py \
  --task=g1_flat \
  --num_envs=4096 \
  --headless \
  --resume=True \
  --load_run=latest \
  --checkpoint=latest
```

**本地测试/回放**

```bash
python legged_lab/scripts/play.py \
  --task=g1_flat \
  --headless \
  --play_lin_vel_x 1.0
```

**日志位置**
训练日志默认在 `logs/<experiment_name>/`，支持 TensorBoard：

```bash
tensorboard --logdir /home/hiyio/LeggedLab/logs
```

**常见问题**
- 显存不足：减少 `--num_envs`（例如 2048 或 1024）。
- 训练速度慢：确认 GPU 空闲，关闭桌面占用，保持 `--headless`。

**推荐参数**
- `--num_envs=4096` 是常见平衡点。
- `--headless` 训练更快更稳。
- `--logger=tensorboard` 便于本地查看曲线。

**提交纪律**
后续所有修改都按“小步可回退”原则提交：

- 每完成一个可验证修改就提交一次，不把多条实验路线混在同一个 commit。
- 提交前至少运行轻量检查：`python -m py_compile` 覆盖改动过的 Python 文件，必要时跑一次小规模 `--num_envs=64` smoke test。
- 运行日志、临时对话记录、缓存文件不混入功能提交，除非这次提交的目标就是更新文档或实验记录。
- commit message 用动词开头，说明对象和目的，例如 `Align G1 training with Isaac Sim 5.1`。
- 长训练进程不用为了提交而中断；提交代码状态，日志目录在文档或 commit 说明里记录。

**分支策略**
主线建议保留为可运行整合线，具体实验走 `feat/*` 或 `fix/*`：

- `magicbot-z1-support`：当前整合分支，承载 Z1/G1 共同可运行链路。
- `feat/z1-deploy-native-sdk`：基于 `engineai_robotics_native_sdk` 的 Z1 部署代码。
- `feat/z1-assets-leggedlab`：Z1 asset、关节映射、actuator、contact body 与 LeggedLab 适配。
- `feat/z1-whole-body-tracking`：Z1 在 `whole_body_tracking` 侧的 motion/retarget/tracking 训练链路。
- `feat/g1-flat-plane-curriculum`：G1 真平面走路/跑步课程与 reward 调参。
- `feat/g1-gravel-terrain`：G1 小地形、gravel/generator 训练。
- `feat/isaacsim51-compat`：Isaac Sim 5.1 / IsaacLab 3.3 / rsl_rl API 兼容层。
- `fix/physx-resume-restoffset-*`：PhysX 或 resume 的窄修复，只放最小补丁。

推荐从整合线开新实验分支：

```bash
git switch magicbot-z1-support
git switch -c feat/z1-deploy-native-sdk
```

实验稳定后再合回整合线：

```bash
git switch magicbot-z1-support
git merge --no-ff feat/z1-deploy-native-sdk
```
