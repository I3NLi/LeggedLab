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
常用命令（G1 平地）：

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
