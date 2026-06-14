# Z1 Sprint AMP Progress

Date: 2026-06-14

## Goal

Train MagicBot Z1 toward stable, natural, fast running, using
`sprint1_subject2_magicbot_z1.npz` as the sprint reference motion. The long-term
target is 8 m/s, reached in stages rather than forced in the first run.

## Current Baseline

Stable checkpoint to preserve:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958/model_23000.pt`

Important baseline settings kept:

- `speed_tracking_duration_s = 2.5`
- command x range remains `(-2.5, 5.0)` for the first bridge run
- actor/critic observation dimensions remain `82 / 87`
- existing `magicbot_z1_flat` task is not converted to sprint imitation

## AMP Status

LeggedLab's existing `--amp` flag is autocast mixed precision, not
Adversarial Motion Prior.

This change does not claim full adversarial AMP yet. It adds:

- a name-aware reference motion loader
- policy AMP observation extraction from the current robot state
- expert AMP observation sampling from `sprint1_subject2`
- a separate `magicbot_z1_flat_sprint_bridge` task with small gated sprint-style rewards

Full AMP still needs a discriminator/replay-buffer training runner wired into
LeggedLab's RSL-RL path.

## Sprint Motion Facts

Motion file:

`/home/hiyio/whole_body_tracking/motions/magicbot_z1/collected/sprint1_subject2_magicbot_z1.npz`

Verified data:

- 24 joints, all match MagicBot Z1 joint names
- 25 bodies, required sprint bodies all present
- `fps = 50`
- frames: `13657`
- `torso_link` horizontal speed max: about `5.25 m/s`
- `torso_link` speed p95: about `4.07 m/s`
- frames in `[2.0, 5.8] m/s`: `3364`

## New Task

Task name:

`magicbot_z1_flat_sprint_bridge`

Experiment folder:

`logs/magicbot_z1_flat`

Default run name:

`z1_sprint_bridge_sprint1_subject2`

The task enables reference motion but keeps policy observation dimensions
unchanged. Sprint rewards are gated by `command_speed >= 2.0`.

Sprint bridge rewards:

- `sprint_reference_joint_pos`, weight `0.03`
- `sprint_reference_joint_vel`, weight `0.01`
- `sprint_reference_body_relative_pos`, weight `0.06`

## Verification

Static compile:

```bash
python -m py_compile legged_lab/utils/reference_motion.py \
  legged_lab/envs/base/base_config.py \
  legged_lab/envs/base/base_env_config.py \
  legged_lab/envs/base/base_env.py \
  legged_lab/mdp/rewards.py \
  legged_lab/envs/magicbot_z1/z1_config.py \
  legged_lab/envs/__init__.py
```

Runtime smoke checks passed:

- `4 env`, `max_iterations=0`: task registration, env init, reference motion load, runner init
- `4 env`, `max_iterations=1`: reward compute path
- AMP interface:
  - policy AMP shape: `(4, 73)`
  - expert AMP shape: `(8, 2, 73)`

Smoke log examples:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-17-47_z1_sprint_bridge_smoke_4env_0iter`
- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-18-29_z1_sprint_bridge_smoke_4env_1iter`

## First Real Training Command

Recommended first bridge run from the stable baseline:

```bash
cd /home/hiyio/LeggedLab
run_name=z1_sprint_bridge_sprint1_subject2_from23000_env20000_20260614
log_file=/home/hiyio/LeggedLab/logs/magicbot_z1_flat/${run_name}.out

setsid -f env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/hiyio/LeggedLab \
  OMNI_KIT_ACCEPT_EULA=YES \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
    --task=magicbot_z1_flat_sprint_bridge \
    --num_envs=20000 \
    --headless \
    --logger=tensorboard \
    --resume=True \
    --load_run=2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958 \
    --checkpoint=model_23000.pt \
    --run_name=${run_name} \
    > ${log_file} 2>&1
```

## Stage-1 Success Criteria

For the first sprint bridge run, do not judge it by 8 m/s. Check whether it:

- preserves long episode length near the baseline
- keeps timeout ratio high
- does not increase head/shoulder or body-contact resets
- reduces high-speed start hesitation
- improves visual leg swing and arm posture at 3-5 m/s
- does not degrade MuJoCo/deploy consistency after export

## First Checkpoint: model_23100.pt

Run:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109`

Checkpoint:

`model_23100.pt`

Log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109.out`

Window `23090-23103`:

- Mean reward: `3.5136`
- Mean episode length: `886.44`
- timeout_ratio: `0.7901`
- head/shoulder contact ratio: `0.0340`
- body contact ratio: `0.0000`
- speed tracking failure ratio: `0.1760`
- sprint joint pos reward: `0.0047`
- sprint joint vel reward: `0.0011`
- sprint body relative pos reward: `0.0130`

Interpretation:

- The initial short-episode dip after resume recovered by around `23045`.
- `model_23100.pt` is not obviously degraded against `model_23000.pt`.
- The run should continue to `model_23200.pt` before visual/export testing, unless speed failure rises sharply.

## Second Checkpoint: model_23200.pt

Checkpoint:

`model_23200.pt`

Window `23100-23204`:

- Mean reward: `2.8237`
- Mean episode length: `883.7469`
- timeout_ratio: `0.7734`
- head/shoulder contact ratio: `0.0403`
- body contact ratio: `0.0001`
- speed tracking failure ratio: `0.1862`
- sprint joint pos reward: `0.0046`
- sprint joint vel reward: `0.0011`
- sprint body relative pos reward: `0.0128`

Window `23190-23204`:

- Mean reward: `2.1880`
- Mean episode length: `887.0220`
- timeout_ratio: `0.7810`
- head/shoulder contact ratio: `0.0375`
- body contact ratio: `0.0000`
- speed tracking failure ratio: `0.1815`

Interpretation:

- `model_23200.pt` remains usable and is not a collapse.
- `model_23100.pt` currently has the cleaner short-window score.
- Continue to `model_23300.pt`; keep `model_23100.pt` and `model_23200.pt` as protected candidates for play/export checks.

## Third Checkpoint: model_23300.pt

Checkpoint:

`model_23300.pt`

Window `23250-23305`:

- Mean reward: `1.9518`
- Mean episode length: `895.1350`
- timeout_ratio: `0.7939`
- head/shoulder contact ratio: `0.0451`
- body contact ratio: `0.0001`
- speed tracking failure ratio: `0.1610`
- sprint joint pos reward: `0.0047`
- sprint joint vel reward: `0.0011`
- sprint body relative pos reward: `0.0129`

Window `23290-23305`:

- Mean reward: `2.2037`
- Mean episode length: `899.6519`
- timeout_ratio: `0.8044`
- head/shoulder contact ratio: `0.0461`
- body contact ratio: `0.0000`
- speed tracking failure ratio: `0.1495`

Interpretation:

- `model_23300.pt` has the best speed-tracking failure ratio so far.
- Reward is lower than `model_23100.pt`, but survival and speed failure are stronger.
- Stop the long train here and use `model_23300.pt` as the first visual play candidate.

Play command launched:

```bash
python -u legged_lab/scripts/play.py \
  --task=magicbot_z1_flat_sprint_bridge \
  --num_envs=50 \
  --resume=True \
  --load_run=2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109 \
  --checkpoint=model_23300.pt \
  --play_lin_vel_x_min=-2.5 \
  --play_lin_vel_x=5.0 \
  --play_lin_vel_y_min=-0.5 \
  --play_lin_vel_y=0.5 \
  --play_ang_vel_z_min=-1.57 \
  --play_ang_vel_z=1.57 \
  --velocity_debug_vis
```

Play log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_bridge_play_23300_cmdx-2p5_5_20260614_135823.out`

Exported artifacts:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109/exported/policy.pt`
- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109/exported/policy.onnx`

Deploy YAML snapshots:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109/deploy/LocoMode.yaml`
- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109/deploy/LocoMode_lowKp.yaml`

## Fixed-Speed Evaluation: baseline_23000 vs sprint_bridge_23300

Date: `2026-06-14`

Evaluation setup:

- `num_envs`: `32`
- fixed commands: `-2.0`, `2.5`, `4.0`, `5.0` m/s
- duration per speed: `8.0s`
- warmup excluded from metrics: first `3.0s`
- forced command source: `vel_command_b` and `_command_buf`
- disabled: observation noise, push randomization, episode length curriculum
- actor/critic dimensions observed in both evals: `82 / 87`, action dim `24`

Logs:

- baseline eval: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_baseline_23000_20260614_140729.out`
- sprint bridge eval: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_bridge_23300_20260614_141146.out`

Baseline checkpoint:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958/model_23000.pt`

Sprint bridge checkpoint:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_13-22-02_z1_sprint_bridge_sprint1_subject2_from23000_env10000_20260614_132109/model_23300.pt`

| checkpoint | cmd vx | mean vx | mean abs vx error | reset count | reset/env | min height |
|---|---:|---:|---:|---:|---:|---:|
| baseline_23000 | -2.0 | -1.9586 | 0.0638 | 1 | 0.0313 | 0.6355 |
| sprint_bridge_23300 | -2.0 | -1.9903 | 0.0569 | 1 | 0.0313 | 0.6399 |
| baseline_23000 | 2.5 | 1.0812 | 1.4316 | 21 | 0.6563 | 0.6190 |
| sprint_bridge_23300 | 2.5 | 1.0398 | 1.4635 | 23 | 0.7188 | 0.6623 |
| baseline_23000 | 4.0 | -0.0007 | 4.0007 | 32 | 1.0000 | 0.5905 |
| sprint_bridge_23300 | 4.0 | -0.0185 | 4.0185 | 33 | 1.0313 | 0.6207 |
| baseline_23000 | 5.0 | -0.0125 | 5.0125 | 35 | 1.0938 | 0.4447 |
| sprint_bridge_23300 | 5.0 | -0.0675 | 5.0675 | 33 | 1.0313 | 0.5888 |

Interpretation:

- Backward command is not the core policy weakness. Both checkpoints track `-2.0 m/s` well.
- The stable baseline already struggles at `+2.5 m/s`: some envs can reach the target, but the mean velocity is only about `1.08 m/s` and resets are high.
- Both checkpoints fail at `+4.0` and `+5.0 m/s`: mean forward velocity is effectively zero.
- The current sprint bridge reward did not improve high-speed startup or speed tracking. `model_23300.pt` should not be used as the next high-speed training foundation without changing the learning signal.
- The next useful step is not more blind training on this bridge. Either implement true AMP with discriminator reward, or first rebuild the velocity curriculum/reward so commands above `2.5 m/s` have a learnable path instead of immediately becoming a failure mode.

## True AMP Training Path: initial implementation

Date: `2026-06-14`

Implemented a repo-local adversarial motion prior path under:

- `legged_lab/amp/normalizer.py`
- `legged_lab/amp/discriminator.py`
- `legged_lab/amp/replay_buffer.py`
- `legged_lab/amp/ppo.py`
- `legged_lab/amp/runner.py`

Training entrypoint behavior:

- `legged_lab/scripts/train.py` keeps normal `OnPolicyRunner` for existing tasks.
- It switches to `AMPOnPolicyRunner` only when `agent_cfg.motion_prior.enable == True`.
- Play/export can still use normal `OnPolicyRunner` because the exported policy network remains the same `82 -> 24` actor.

New task:

`magicbot_z1_flat_sprint_amp`

Design:

- Uses the same Z1 flat base reward as the stable baseline.
- Enables `sprint1_subject2` reference motion for AMP only.
- Does not include the earlier bridge imitation rewards.
- AMP observation dim: `73`.
- AMP window: `2` frames.
- AMP reward is gated to command speed magnitude `>= 2.0 m/s` so low-speed walking/backward behavior is not pulled toward sprint data.
- Initial AMP reward coefficient: `0.15`.
- Discriminator hidden dims: `[256, 128]`.
- Discriminator LR: `1e-4`.
- Grad penalty coefficient: `5.0`.

Smoke verification:

Command:

```bash
python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp \
  --num_envs=4 \
  --max_iterations=1 \
  --run_name=z1_sprint_amp_smoke \
  --logger=tensorboard \
  --headless \
  --skip_deploy_yaml
```

Log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_smoke_20260614_142507.out`

Run directory:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-25-20_z1_sprint_amp_smoke`

Observed smoke output:

- `Motion prior AMP runner: True`
- policy/critic dims: `82 / 87`
- action dim: `24`
- `Config/reference_motion_enabled: 1.0`
- `Config/reference_motion_amp_obs_dim: 73.0`
- `amp_discriminator: 0.8381`
- `amp_grad_penalty: 0.0909`
- `amp_policy_logit: -0.0249`
- `amp_expert_logit: 0.1782`

Saved checkpoint:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-25-20_z1_sprint_amp_smoke/model_0.pt`

Checkpoint keys verified:

- `model_state_dict`
- `optimizer_state_dict`
- `amp_discriminator_state_dict`
- `amp_discriminator_optimizer_state_dict`
- `amp_normalizer_state_dict`

Interpretation:

- This is now a real AMP training path: it has policy AMP samples, expert motion samples, replay buffer, discriminator loss, gradient penalty, AMP reward, and checkpointed discriminator/normalizer state.
- It is not yet a quality result. The smoke run only proves the training path executes.
- Next training should start from the protected stable baseline `model_23000.pt`, not from the degraded bridge checkpoint.

Resume smoke from stable baseline:

Command:

```bash
python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp \
  --num_envs=4 \
  --max_iterations=1 \
  --run_name=z1_sprint_amp_resume23000_smoke \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958 \
  --checkpoint=model_23000.pt \
  --headless \
  --skip_deploy_yaml
```

Log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_resume23000_smoke_20260614_142647.out`

Run directory:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-26-59_z1_sprint_amp_resume23000_smoke`

Observed:

- Loaded baseline checkpoint successfully.
- Training iteration started at `23000/23001`.
- Mean action noise std loaded as `0.68`.
- AMP metrics were produced:
  - `amp_discriminator: 0.8500`
  - `amp_grad_penalty: 0.0902`
  - `amp_policy_logit: -0.0054`
  - `amp_expert_logit: 0.1863`

Conclusion:

- `magicbot_z1_flat_sprint_amp` can resume from the protected stable baseline `model_23000.pt`.
- A first real AMP pilot can start from this checkpoint. Keep the pilot short and checkpointed; do not resume from the degraded sprint bridge run.

Policy-only load smoke:

- First attempt failed because the ad-hoc test script did not set `env_cfg.scene.seed`; this was a test harness issue, not a policy/checkpoint issue.
- Second attempt with `env_cfg.scene.seed = agent_cfg.seed` succeeded.

Successful log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_policy_load_smoke_20260614_142829.out`

Output:

`POLICY_LOAD_OK /home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-26-59_z1_sprint_amp_resume23000_smoke/model_23000.pt (4, 24)`

Conclusion:

- Normal `OnPolicyRunner` can load the policy part of an AMP checkpoint with extra discriminator/normalizer keys.
- Play/export/deploy path should remain compatible because actor obs/action dimensions are unchanged: `82 -> 24`.

## First AMP Pilot Training

Date: `2026-06-14`

Purpose:

- Start a conservative true AMP pilot from the protected stable baseline.
- Do not use the degraded sprint bridge checkpoint.
- Keep env count at `10000` because another DogUrdf17 training job is using the same GPU.

Command:

```bash
python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp \
  --num_envs=10000 \
  --max_iterations=600 \
  --run_name=z1_sprint_amp_sprint1_subject2_from23000_amp0p15_env10000_20260614_143049 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958 \
  --checkpoint=model_23000.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot
```

Process:

- PID: `2867329`
- stdout log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_sprint1_subject2_from23000_amp0p15_env10000_20260614_143049.out`
- run directory: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-31-43_z1_sprint_amp_sprint1_subject2_from23000_amp0p15_env10000_20260614_143049`

Early status:

- Scene initialized without OOM.
- GPU memory while co-running DogUrdf17: about `25.9 / 32.6 GB`.
- Throughput around iteration `23006-23007`: `33k-37k steps/s`.
- Early discriminator separation is active:
  - `amp_policy_logit` around `-0.92`
  - `amp_expert_logit` around `0.85`
- Early mean reward and episode length are lower than the stable baseline; this is expected immediately after adding a new reward prior and should not be judged before at least `model_23100.pt`.

Next gate:

- Inspect at `model_23100.pt`.
- Compare against baseline fixed-speed metrics, especially `2.5 m/s` and `4.0 m/s`.
- Stop early if speed tracking failure ratio or head/shoulder contact ratio rises sharply.

## First AMP Pilot Gate: model_23100.pt

Checkpoint:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-31-43_z1_sprint_amp_sprint1_subject2_from23000_amp0p15_env10000_20260614_143049/model_23100.pt`

The pilot was stopped after `model_23100.pt` was saved to preserve a clean first AMP gate.

Training window `23090-23100`:

- Mean reward: `3.8927`
- Mean episode length: `896.50`
- timeout_ratio: `0.8179`
- head/shoulder contact ratio: `0.0511`
- body contact ratio: `0.0000`
- speed tracking failure ratio: `0.1310`
- amp discriminator loss: `0.0202`
- amp policy logit: `-0.9056`
- amp expert logit: `0.9160`

Interpretation from training metrics:

- Survival recovered after the initial AMP transition and is close to the protected baseline.
- Speed tracking failure ratio improved relative to the baseline training window.
- Head/shoulder contact ratio is higher than desired and must be watched.
- Discriminator separates policy/expert strongly, so AMP reward can become sparse unless the policy gets closer to the expert distribution.

Fixed-speed eval log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_23100_20260614_144616.out`

| checkpoint | cmd vx | mean vx | mean abs vx error | reset count | reset/env | min height |
|---|---:|---:|---:|---:|---:|---:|
| baseline_23000 | -2.0 | -1.9586 | 0.0638 | 1 | 0.0313 | 0.6355 |
| sprint_amp_23100 | -2.0 | -1.9871 | 0.0696 | 1 | 0.0313 | 0.6523 |
| baseline_23000 | 2.5 | 1.0812 | 1.4316 | 21 | 0.6563 | 0.6190 |
| sprint_amp_23100 | 2.5 | 1.5124 | 1.0084 | 19 | 0.5938 | 0.4940 |
| baseline_23000 | 4.0 | -0.0007 | 4.0007 | 32 | 1.0000 | 0.5905 |
| sprint_amp_23100 | 4.0 | 0.0395 | 3.9605 | 31 | 0.9688 | 0.6121 |
| baseline_23000 | 5.0 | -0.0125 | 5.0125 | 35 | 1.0938 | 0.4447 |
| sprint_amp_23100 | 5.0 | -0.4795 | 5.4795 | 35 | 1.0938 | 0.6029 |

Interpretation:

- `-2.0 m/s` backward tracking is preserved.
- `2.5 m/s` improves meaningfully: mean velocity rises from `1.08` to `1.51 m/s`, and resets drop from `21` to `19`.
- `4.0 m/s` is still effectively not learned.
- `5.0 m/s` is worse than baseline in mean forward velocity, even though min height is higher.
- The first AMP pilot is useful but not sufficient. It should not be pushed directly toward 5-8 m/s yet.

Next decision:

- Treat this as partial Stage 1 progress.
- Add/try a Stage 1 training range focused around `2.5-3.5 m/s` instead of continuing to expose the policy equally to `5.0 m/s`.
- Before launching the next run, evaluate `model_23100.pt` at `3.0` and `3.5 m/s` to set the next command range.

Stage 1 boundary eval:

Log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_23100_stage1_20260614_145132.out`

| checkpoint | cmd vx | mean vx | mean abs vx error | reset count | reset/env | min height |
|---|---:|---:|---:|---:|---:|---:|
| sprint_amp_23100 | 3.0 | 1.3612 | 1.6412 | 19 | 0.5938 | 0.6148 |
| sprint_amp_23100 | 3.5 | 0.7891 | 2.7111 | 26 | 0.8125 | 0.4373 |

Motion data speed coverage:

- `2.0-3.0 m/s`: `1162` frames
- `2.0-3.6 m/s`: `2017` frames
- `2.0-4.0 m/s`: `2584` frames

Decision:

- `3.0 m/s` is partially available but not stable.
- `3.5 m/s` is not stable enough to make it the next primary target.
- Next run should be Stage 1A:
  - command `lin_vel_x` max around `3.0`
  - expert AMP max reference speed around `3.6`
  - resume from `sprint_amp_23100`
  - evaluate at `2.5`, `3.0`, `3.5` before expanding again

## Stage 1A Task

New task:

`magicbot_z1_flat_sprint_amp_stage1a`

Config:

- command `lin_vel_x`: `(-2.5, 3.0)`
- reference motion min speed: `2.0`
- reference motion max speed: `3.6`
- AMP reward coefficient: `0.15`
- start checkpoint: `sprint_amp_23100`

Smoke:

Command:

```bash
python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage1a \
  --num_envs=4 \
  --max_iterations=1 \
  --run_name=z1_sprint_amp_stage1a_resume23100_smoke \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_14-31-43_z1_sprint_amp_sprint1_subject2_from23000_amp0p15_env10000_20260614_143049 \
  --checkpoint=model_23100.pt \
  --headless \
  --skip_deploy_yaml
```

Log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1a_resume23100_smoke_20260614_145457.out`

Run dir:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-55-11_z1_sprint_amp_stage1a_resume23100_smoke`

Verified in saved config:

- `params/env.yaml`: `lin_vel_x: (-2.5, 3.0)`
- `params/env.yaml`: `max_reference_speed: 3.6`
- `params/agent.yaml`: `motion_prior.reward_coef: 0.15`

Smoke result:

- Loaded `model_23100.pt` successfully.
- `Motion prior AMP runner: True`.
- AMP metrics were produced.

## Stage 1A Training

Date: `2026-06-14`

Command:

```bash
python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage1a \
  --num_envs=10000 \
  --max_iterations=101 \
  --run_name=z1_sprint_amp_stage1a_from23100_cmdx-2p5_3p0_ref2p0_3p6_env10000_20260614_145619 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_14-31-43_z1_sprint_amp_sprint1_subject2_from23000_amp0p15_env10000_20260614_143049 \
  --checkpoint=model_23100.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot
```

Process:

- PID: `3190050`
- stdout log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1a_from23100_cmdx-2p5_3p0_ref2p0_3p6_env10000_20260614_145619.out`
- run directory: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-57-20_z1_sprint_amp_stage1a_from23100_cmdx-2p5_3p0_ref2p0_3p6_env10000_20260614_145619`

Early status around `23109-23110`:

- Throughput: about `34k-36k steps/s`
- Mean episode length: `233-256`
- speed tracking failure ratio: `0.0038-0.0073`
- head/shoulder contact ratio: `0.017-0.027`
- AMP policy/expert logits remain separated around `-0.89 / 0.90`

Interpretation:

- Early low speed-failure is expected after narrowing command max to `3.0`; it is not proof of Stage 1 success yet.
- Need inspect near `model_23200.pt`, then fixed-speed eval at `2.5`, `3.0`, and `3.5`.

Stage 1A finished naturally at `model_23200.pt`.

Checkpoint:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_14-57-20_z1_sprint_amp_stage1a_from23100_cmdx-2p5_3p0_ref2p0_3p6_env10000_20260614_145619/model_23200.pt`

Final window `23190-23200` average:

- mean reward: `13.2709`
- mean episode length: `966.7336`
- timeout ratio: `0.9500`
- head/shoulder contact ratio: `0.0484`
- body contact ratio: `0.0000`
- speed tracking failure ratio: `0.0016`
- AMP discriminator loss: `0.0437`
- AMP policy logit: `-0.8622`
- AMP expert logit: `0.8630`

Fixed-speed eval utility:

`legged_lab/scripts/eval_fixed_speed.py`

Notes:

- Reads speed from `extras["observations"]["critic"]` instead of directly touching articulation root velocity. Direct root velocity access in this headless eval path caused Isaac to exit before results were printed.
- Uses body-frame `root_lin_vel_b[:, 0]`, matching the velocity source already packed into critic observations by `BaseEnv.compute_current_observations`.

Same-script comparison logs:

- `model_23100.pt`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_23100_same_script_20260614_152700.out`
- `model_23200.pt`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_stage1a_23200_20260614_152320.out`

| checkpoint | cmd vx | mean vx | abs vx error | p50 vx | p90 abs vx | reset count | reset/env |
|---|---:|---:|---:|---:|---:|---:|---:|
| sprint_amp_23100 | 2.5 | 2.0073 | 0.5201 | 2.4134 | 2.5591 | 13 | 0.4063 |
| sprint_amp_stage1a_23200 | 2.5 | 2.3477 | 0.1791 | 2.4152 | 2.5564 | 4 | 0.1250 |
| sprint_amp_23100 | 3.0 | 1.0887 | 1.9163 | 0.1462 | 2.9205 | 41 | 1.2813 |
| sprint_amp_stage1a_23200 | 3.0 | 2.1901 | 0.8178 | 2.7653 | 2.9782 | 15 | 0.4688 |
| sprint_amp_23100 | 3.5 | 0.5246 | 2.9754 | -0.1026 | 3.2471 | 52 | 1.6250 |
| sprint_amp_stage1a_23200 | 3.5 | 2.5617 | 0.9383 | 3.0629 | 3.3170 | 12 | 0.3750 |

Interpretation:

- Stage 1A is clearly beneficial compared with `model_23100.pt`.
- `2.5 m/s` is close to usable: mean speed improved, error dropped by about `65%`, and reset count dropped from `13` to `4`.
- `3.0 m/s` is no longer mostly stuck: median speed moved from `0.1462` to `2.7653`, but reset count is still high enough that it is not stable.
- `3.5 m/s` is now reachable in bursts: median speed is `3.0629`, but mean speed `2.5617` and reset count `12` show Stage 1 is not complete yet.

Decision:

- Keep `model_23200.pt` as the current useful Stage 1 checkpoint.
- Do not expand to `5.0 m/s` yet.
- Continue Stage 1A or a slightly expanded Stage 1B only after preserving `model_23200.pt`.
- Recommended next run: resume from `model_23200.pt`, keep command max near `3.0` or at most `3.5`, and focus on reducing resets at `3.0-3.5` before moving toward `4.0-5.0`.

## Stage 1A Stabilization Continue From 23200

Date: `2026-06-14`

Resource check before launch:

- GPU memory: about `17.0 / 32.6 GB` used.
- Existing GPU training: DogUrdf17 rough, PID `1909165`, about `14.8 GB`.
- Decision: keep Z1 at `10000 env`; do not use `20000 env` while DogUrdf17 is active.

Purpose:

- Preserve `model_23200.pt`.
- Continue the same Stage 1A range instead of expanding to `5.0 m/s`.
- Main target is fewer resets at `3.0-3.5 m/s`.

Command:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage1a \
  --num_envs=10000 \
  --max_iterations=101 \
  --run_name=z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_14-57-20_z1_sprint_amp_stage1a_from23100_cmdx-2p5_3p0_ref2p0_3p6_env10000_20260614_145619 \
  --checkpoint=model_23200.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot
```

Result:

- PID: `3669417`
- stdout log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534.out`
- run directory: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534`
- checkpoint: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534/model_23300.pt`

Final window `23290-23300` average:

- mean reward: `12.4473`
- mean episode length: `963.1245`
- timeout ratio: `0.9474`
- head/shoulder contact ratio: `0.0460`
- body contact ratio: `0.0003`
- speed tracking failure ratio: `0.0063`
- AMP discriminator loss: `0.0396`
- AMP policy logit: `-0.8756`
- AMP expert logit: `0.8755`

Same-script fixed-speed eval log:

`/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_stage1a_23300_20260614_154800.out`

| checkpoint | cmd vx | mean vx | abs vx error | p50 vx | p90 abs vx | reset count | reset/env |
|---|---:|---:|---:|---:|---:|---:|---:|
| sprint_amp_stage1a_23200 | 2.5 | 2.3477 | 0.1791 | 2.4152 | 2.5564 | 4 | 0.1250 |
| sprint_amp_stage1a_23300 | 2.5 | 2.3075 | 0.1954 | 2.3119 | 2.4591 | 1 | 0.0313 |
| sprint_amp_stage1a_23200 | 3.0 | 2.1901 | 0.8178 | 2.7653 | 2.9782 | 15 | 0.4688 |
| sprint_amp_stage1a_23300 | 3.0 | 2.5032 | 0.4972 | 2.6921 | 2.8620 | 4 | 0.1250 |
| sprint_amp_stage1a_23200 | 3.5 | 2.5617 | 0.9383 | 3.0629 | 3.3170 | 12 | 0.3750 |
| sprint_amp_stage1a_23300 | 3.5 | 2.6732 | 0.8270 | 3.0377 | 3.2565 | 9 | 0.2813 |

Interpretation:

- `model_23300.pt` is the best current Stage 1 checkpoint.
- `2.5 m/s` is now stable in the fixed-speed test.
- `3.0 m/s` improved substantially versus `23200`, especially resets.
- `3.5 m/s` improved, but is still not stable enough to call Stage 1 complete.

Decision:

- Preserve `model_23300.pt`.
- Add Stage 1B task for a careful expansion:
  - command `lin_vel_x`: `(-2.5, 3.5)`
  - reference motion max speed: `4.0`
  - AMP/reward/termination/domain randomization unchanged
- Do not expand to `5.0 m/s` yet.

## Stage 1B Task

Task:

`magicbot_z1_flat_sprint_amp_stage1b`

Config:

- command `lin_vel_x`: `(-2.5, 3.5)`
- reference motion min speed: `2.0`
- reference motion max speed: `4.0`
- AMP reward coefficient: `0.15`
- no reward/termination/domain randomization changes from Stage 1A

Smoke:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage1b \
  --num_envs=4 \
  --max_iterations=1 \
  --run_name=z1_sprint_amp_stage1b_resume23300_smoke_20260614_1554 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534 \
  --checkpoint=model_23300.pt \
  --headless \
  --skip_deploy_yaml
```

Smoke result:

- log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1b_resume23300_smoke_20260614_1554.out`
- run directory: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-54-09_z1_sprint_amp_stage1b_resume23300_smoke_20260614_1554`
- `Motion prior AMP runner: True`
- loaded `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534/model_23300.pt`
- saved config verified:
  - `lin_vel_x: (-2.5, 3.5)`
  - `max_reference_speed: 4.0`
  - `motion_prior.reward_coef: 0.15`

Stage 1B training command:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage1b \
  --num_envs=10000 \
  --max_iterations=101 \
  --run_name=z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534 \
  --checkpoint=model_23300.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot
```

Result:

- PID: `3942103`
- stdout log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556.out`
- run directory: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556`
- checkpoint: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556/model_23400.pt`

Final window `23390-23400` average:

- mean reward: `-0.6627` because iteration `23396` had a large negative outlier; excluding that outlier, the window is around `10-12`.
- mean episode length: `959.3155`
- timeout ratio: `0.9350`
- head/shoulder contact ratio: `0.0473`
- body contact ratio: `0.0003`
- speed tracking failure ratio: `0.0175`
- AMP discriminator loss: `0.0334`
- AMP policy logit: `-0.8796`
- AMP expert logit: `0.8800`

Same-script fixed-speed eval logs:

- `model_23400.pt`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_stage1b_23400_20260614_161100.out`
- `model_23300.pt @ 4.0`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_speed_eval_sprint_amp_stage1a_23300_4p0_20260614_161800.out`

| checkpoint | cmd vx | mean vx | abs vx error | p50 vx | p90 abs vx | reset count | reset/env |
|---|---:|---:|---:|---:|---:|---:|---:|
| sprint_amp_stage1a_23300 | 2.5 | 2.3075 | 0.1954 | 2.3119 | 2.4591 | 1 | 0.0313 |
| sprint_amp_stage1b_23400 | 2.5 | 2.1763 | 0.3237 | 2.1822 | 2.3410 | 3 | 0.0938 |
| sprint_amp_stage1a_23300 | 3.0 | 2.5032 | 0.4972 | 2.6921 | 2.8620 | 4 | 0.1250 |
| sprint_amp_stage1b_23400 | 3.0 | 2.5172 | 0.4828 | 2.5853 | 2.7647 | 1 | 0.0313 |
| sprint_amp_stage1a_23300 | 3.5 | 2.6732 | 0.8270 | 3.0377 | 3.2565 | 9 | 0.2813 |
| sprint_amp_stage1b_23400 | 3.5 | 2.6344 | 0.8656 | 2.9455 | 3.1753 | 5 | 0.1563 |
| sprint_amp_stage1a_23300 | 4.0 | 2.1946 | 1.8054 | 3.1704 | 3.5474 | 22 | 0.6875 |
| sprint_amp_stage1b_23400 | 4.0 | 2.5608 | 1.4392 | 3.1487 | 3.4826 | 12 | 0.3750 |

Interpretation:

- `model_23400.pt` improves reset counts at `3.0`, `3.5`, and `4.0`.
- `model_23400.pt` regresses `2.5 m/s` tracking versus `model_23300.pt`.
- `4.0 m/s` is reachable in bursts but not stable enough to call Stage 2 complete.

Decision:

- Current stable Stage 1 best: `model_23300.pt`.
- Current high-speed exploratory checkpoint: `model_23400.pt`.
- Do not delete either.
- Do not expand beyond `4.0 m/s` yet.
- Next useful step is visual play for `23300` and `23400`, then decide whether to continue Stage1B or add a low-speed retention/stability term before continuing.

## Play Check: Stable Stage 1 Best 23300

Date: `2026-06-14`

Command:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/play.py \
  --task=magicbot_z1_flat_sprint_amp_stage1a \
  --num_envs=50 \
  --resume=True \
  --load_run=2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534 \
  --checkpoint=model_23300.pt \
  --play_lin_vel_x_min=-2.5 \
  --play_lin_vel_x=3.5 \
  --play_lin_vel_y=0.0 \
  --play_ang_vel_z=0.0 \
  --velocity_debug_vis
```

Status:

- PID: `56730`
- log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage1_best_23300_cmdx-2p5_3p5_20260614_1623.out`
- command range confirmed: `lin_vel_x=(-2.5, 3.5)`
- actor input dim confirmed: `82`
- exported artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534/exported/policy.onnx`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534/exported/policy.onnx.data`

Manual visual checklist:

- whether `2.5-3.0 m/s` starts cleanly without standing still;
- whether `3.5 m/s` shows real stepping instead of sliding;
- whether arms are less front-raised/stiff;
- whether torso pitch looks like controlled sprint lean instead of falling;
- whether backward commands remain preserved.

To stop this play:

```bash
kill 56730
```

Prepared switch command for high-speed exploratory `model_23400.pt` after stopping PID `56730`:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/play.py \
  --task=magicbot_z1_flat_sprint_amp_stage1b \
  --num_envs=50 \
  --resume=True \
  --load_run=2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556 \
  --checkpoint=model_23400.pt \
  --play_lin_vel_x_min=-2.5 \
  --play_lin_vel_x=4.0 \
  --play_lin_vel_y=0.0 \
  --play_ang_vel_z=0.0 \
  --velocity_debug_vis
```

## Gait Quality Eval: 23300 vs 23400

Date: `2026-06-14`

Utility:

`legged_lab/scripts/eval_gait_quality.py`

Purpose:

- Add objective gait metrics beyond velocity tracking.
- Metrics include body tilt proxy, foot height, swing-foot height, stance/flight ratios, contact transition rate, and arm/shoulder joint deviation.
- Speed is read from critic observations to avoid direct root velocity access in Isaac headless mode.

Logs:

- `23300`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_gait_quality_stage1a_23300_20260614_1628.out`
- `23400`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_gait_quality_stage1b_23400_20260614_1636.out`

| checkpoint | cmd vx | mean vx | abs err | resets | tilt xy | p90 swing foot z | single stance | flight | arm abs | shoulder pitch abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 23300 | 2.5 | 2.3114 | 0.1916 | 0 | 0.0543 | 0.2818 | 0.9111 | 0.0889 | 0.3241 | 0.2187 |
| 23400 | 2.5 | 2.1816 | 0.3185 | 1 | 0.0529 | 0.2803 | 0.8941 | 0.1059 | 0.2718 | 0.1910 |
| 23300 | 3.0 | 2.6817 | 0.3186 | 0 | 0.0586 | 0.2960 | 0.8756 | 0.1242 | 0.3797 | 0.2755 |
| 23400 | 3.0 | 2.1822 | 0.8178 | 3 | 0.0608 | 0.2949 | 0.8506 | 0.1148 | 0.3070 | 0.2352 |
| 23300 | 3.5 | 2.4868 | 1.0132 | 8 | 0.0759 | 0.3006 | 0.8270 | 0.1316 | 0.3831 | 0.3097 |
| 23400 | 3.5 | 2.7382 | 0.7618 | 1 | 0.0577 | 0.3093 | 0.8422 | 0.1453 | 0.3493 | 0.2768 |
| 23300 | 4.0 | 2.0046 | 1.9954 | 13 | 0.0845 | 0.2993 | 0.7984 | 0.1138 | 0.3697 | 0.3211 |
| 23400 | 4.0 | 2.4744 | 1.5256 | 7 | 0.0664 | 0.3134 | 0.8161 | 0.1395 | 0.3588 | 0.3028 |

Interpretation:

- `23300` remains better at `2.5-3.0 m/s`, with zero resets in this gait eval.
- `23400` is clearly better at `3.5-4.0 m/s`, with fewer resets, higher mean speed, higher swing-foot height, and lower tilt than `23300`.
- `23400` also has lower arm/shoulder pitch deviation across these tests, which may help the "arms not front-raised/stiff" issue, but this still needs visual confirmation.
- The transition from `23300` to `23400` trades low/mid speed retention for higher-speed stability.

Decision:

- Keep both checkpoints.
- Use `23300` as stable Stage 1 reference.
- Use `23400` as Stage 1B/high-speed exploration reference.
- Next training should not simply continue Stage1B blindly; it should preserve the `23300` low/mid-speed behavior while retaining the `23400` high-speed gains.

## Play Check: High-Speed Exploration 23400

Date: `2026-06-14`

Command:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/play.py \
  --task=magicbot_z1_flat_sprint_amp_stage1b \
  --num_envs=50 \
  --resume=True \
  --load_run=2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556 \
  --checkpoint=model_23400.pt \
  --play_lin_vel_x_min=-2.5 \
  --play_lin_vel_x=4.0 \
  --play_lin_vel_y=0.0 \
  --play_ang_vel_z=0.0 \
  --velocity_debug_vis
```

Status:

- PID: `278512`
- log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage1b_highspeed_23400_cmdx-2p5_4p0_20260614_1642.out`
- command range confirmed: `lin_vel_x=(-2.5, 4.0)`
- actor input dim confirmed: `82`
- exported artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556/exported/policy.onnx`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-56-19_z1_sprint_amp_stage1b_from23300_cmdx-2p5_3p5_ref2p0_4p0_env10000_20260614_1556/exported/policy.onnx.data`

To stop this play:

```bash
kill 278512
```

## Stage1C Retention Plan

Date: `2026-06-14`

Reason:

- `model_23300.pt` is still the stable Stage 1 reference at `2.5-3.0 m/s`.
- `model_23400.pt` improves `3.5-4.0 m/s`, but regresses low/mid-speed tracking and resets.
- The next run should therefore start from `23300`, not from `23400`, and use a conservative expansion instead of blindly continuing Stage1B.

Task:

- `magicbot_z1_flat_sprint_amp_stage1c`

Start checkpoint:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534/model_23300.pt`

Config changes from Stage1A:

- command `lin_vel_x`: `(-2.5, 3.75)`
- reference motion `max_reference_speed`: `4.2`
- AMP reward coefficient: `0.10`
- velocity tracking reward weight: `1.25`
- learning rate: `5.0e-4`
- save interval: `25`

Unchanged safeguards:

- protected baseline `model_23000.pt` is not overwritten or deleted.
- `speed_tracking_duration_s` stays at `2.5`.
- deployment yaml generation remains enabled for the real run.
- reset joint randomization, termination recovery behavior, and domain randomization are not changed in this stage.

Smoke validation:

- log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1c_resume23300_smoke_20260614_1645.out`
- smoke run dir: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-43-31_z1_sprint_amp_stage1c_resume23300_smoke_20260614_1645`
- loaded checkpoint: `model_23300.pt`
- AMP runner enabled: `True`
- actor observation dim: `82`
- critic observation dim: `87`
- saved config confirmed:
  - `lin_vel_x: (-2.5, 3.75)`
  - `max_reference_speed: 4.2`
  - `track_lin_vel_xy_exp.weight: 1.25`
  - `motion_prior.reward_coef: 0.1`
  - `learning_rate: 0.0005`
  - `save_interval: 25`

Planned evaluation:

- evaluate every saved checkpoint at `2.5`, `3.0`, `3.5`, and `4.0 m/s`.
- accept a Stage1C checkpoint only if it preserves `23300` low/mid-speed stability while improving or matching `23400` high-speed behavior.

## Stage1C Training Run

Date: `2026-06-14`

Run name:

- `z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801`

Run dir:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801`

Launch log:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801.out`

Launch command summary:

- task: `magicbot_z1_flat_sprint_amp_stage1c`
- envs: `10000`
- max iterations: `101`
- resume run: `2026-06-14_15-35-05_z1_sprint_amp_stage1a_continue23200_cmdx-2p5_3p0_env10000_20260614_1534`
- checkpoint: `model_23300.pt`
- deploy yaml root: `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot`

Startup validation:

- PID: `399577`
- AMP runner enabled: `True`
- loaded checkpoint: `model_23300.pt`
- generated run-local deploy yaml:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/deploy/LocoMode_lowKp.yaml`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/deploy/LocoMode.yaml`
- synchronized deploy yaml:
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/config/LocoMode_lowKp.yaml`
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/config/LocoMode.yaml`
- first observed GPU state with DogUrdf17 training still running:
  - total GPU memory used: about `26.2 / 32.6 GB`
  - Z1 Stage1C process memory: about `9.1 GB`
  - GPU utilization: about `96%`

Early training note:

- first full iterations were running without OOM.
- `speed_tracking_failure_ratio` was `0.0` in the first observed iterations.
- head/shoulder contact resets were still present and should be monitored before judging the run.

First checkpoint:

- checkpoint: `model_23325.pt`
- saved at: `2026-06-14 16:52:05`
- iteration `23325` snapshot:
  - Mean reward: `-1.21`
  - Mean episode length: `591.62`
  - timeout ratio: `0.8993`
  - head/shoulder contact ratio: `0.0955`
  - speed tracking failure ratio: `0.0052`
- iteration `23326` immediately after checkpoint:
  - Mean reward: `4.66`
  - Mean episode length: `610.20`
  - timeout ratio: `0.8410`
  - head/shoulder contact ratio: `0.0838`
  - speed tracking failure ratio: `0.0752`
- note: first checkpoint is useful to evaluate, but speed-tracking failures started to appear right after it; continue monitoring before treating it as better than `23300`.

Final online training state:

- final checkpoint: `model_23400.pt`
- saved at: `2026-06-14 17:01:12`
- iteration `23400` snapshot:
  - Mean reward: `9.60`
  - Mean episode length: `945.67`
  - timeout ratio: `0.9170`
  - head/shoulder contact ratio: `0.0684`
  - body contact ratio: `0.0000`
  - speed tracking failure ratio: `0.0147`
- online conclusion: the run did not collapse. Online metrics are much healthier than the earlier failed long-duration speed-tracking run, but checkpoint selection must be based on fixed-speed eval.

## Stage1C Fixed-Speed Eval

Date: `2026-06-14`

Task:

- `magicbot_z1_flat_sprint_amp_stage1c`

Eval setup:

- envs: `32`
- warmup: `3.0 s`
- measured duration: `8.0 s`
- speeds: `2.5`, `3.0`, `3.5`, `4.0 m/s`
- log prefix: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_eval_stage1c_model_*_20260614_1703.out`

| checkpoint | cmd vx | mean vx | abs err | resets |
|---|---:|---:|---:|---:|
| 23325 | 2.5 | 2.4487 | 0.1002 | 0 |
| 23325 | 3.0 | 2.6434 | 0.3832 | 8 |
| 23325 | 3.5 | 2.7357 | 0.7813 | 8 |
| 23325 | 4.0 | 2.3766 | 1.6269 | 22 |
| 23350 | 2.5 | 2.4131 | 0.1189 | 0 |
| 23350 | 3.0 | 2.7120 | 0.3027 | 6 |
| 23350 | 3.5 | 2.9746 | 0.5349 | 8 |
| 23350 | 4.0 | 2.3281 | 1.6735 | 21 |
| 23375 | 2.5 | 2.3751 | 0.1403 | 0 |
| 23375 | 3.0 | 2.5855 | 0.4195 | 4 |
| 23375 | 3.5 | 2.7000 | 0.8025 | 11 |
| 23375 | 4.0 | 2.2344 | 1.7670 | 15 |
| 23400 | 2.5 | 2.3885 | 0.1329 | 0 |
| 23400 | 3.0 | 2.6157 | 0.3958 | 3 |
| 23400 | 3.5 | 3.0790 | 0.4294 | 4 |
| 23400 | 4.0 | 2.6634 | 1.3390 | 14 |

Fixed-speed conclusion:

- `model_23400.pt` is the best Stage1C checkpoint overall.
- It preserves low-speed stability well enough at `2.5 m/s` with zero resets.
- It is the best Stage1C checkpoint at `3.5 m/s`.
- It improves `4.0 m/s` over Stage1A `23300` and Stage1B `23400`, but `4.0 m/s` is still not stable enough to call Stage 2 solved.
- `model_23350.pt` is a useful backup if visual play shows `23400` has worse posture, but metrics favor `23400`.

## Stage1C Gait Quality Eval

Date: `2026-06-14`

Checkpoint:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/model_23400.pt`

Eval log:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_gait_quality_stage1c_23400_20260614_1715.out`

| cmd vx | mean vx | abs err | resets | tilt xy | p90 swing foot z | single stance | flight | arm abs | shoulder pitch abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 2.3908 | 0.1298 | 0 | 0.0443 | 0.2691 | 0.9100 | 0.0900 | 0.2439 | 0.1493 |
| 3.0 | 2.7905 | 0.2193 | 0 | 0.0460 | 0.2817 | 0.8728 | 0.1237 | 0.3054 | 0.2030 |
| 3.5 | 2.9861 | 0.5175 | 2 | 0.0579 | 0.2883 | 0.8377 | 0.1412 | 0.3424 | 0.2539 |
| 4.0 | 2.4722 | 1.5283 | 10 | 0.0669 | 0.2893 | 0.8120 | 0.1283 | 0.3393 | 0.2809 |

Gait conclusion:

- `23400` keeps clean `2.5-3.0 m/s` gait in this gait eval, with zero resets.
- Arm and shoulder-pitch offsets are lower than Stage1A `23300` in the previous gait-quality eval, which is a good sign for the "arms front-raised/stiff" issue.
- `3.5 m/s` is now a plausible next-stage base, though still under-commanded.
- `4.0 m/s` remains exploratory; it should not be used as the next stable baseline without play/MuJoCo/deploy checks.

Current Stage1 selection:

- best Stage1C candidate: `model_23400.pt`
- keep protected stable baseline: `model_23000.pt`
- keep stable Stage1A reference: `model_23300.pt`
- keep Stage1C backup: `model_23350.pt`
- next recommended action: play `model_23400.pt` with command range `-2.5..4.0`, then export/update deploy artifacts only if visual gait is acceptable.

## Play Check: Stage1C 23400

Date: `2026-06-14`

Command summary:

- task: `magicbot_z1_flat_sprint_amp_stage1c`
- checkpoint: `model_23400.pt`
- envs: `50`
- command range: `lin_vel_x=(-2.5, 4.0)`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
- velocity debug visualization: enabled

Status:

- PID: `789481`
- log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage1c_23400_cmdx-2p5_4p0_20260614_1720.out`
- command range confirmed in log.
- actor observation dim confirmed: `82`
- critic observation dim confirmed: `87`
- exported artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/exported/policy.onnx`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/exported/policy.onnx.data`

To stop this play:

```bash
kill 789481
```

## Deploy Artifact Sync: Stage1C 23400

Date: `2026-06-14`

Reason:

- Stage1C play exported the correct `model_23400.pt` policy artifacts, but `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/model` still contained older files from `2026-06-14 01:11`.
- The deploy YAML files were already synchronized by the Stage1C training run and matched the run-local deploy YAML exactly.

Synced model artifacts:

- source:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/exported/policy.onnx`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/exported/policy.onnx.data`
- destination:
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/model/policy.pt`
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/model/policy.onnx`
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/model/policy.onnx.data`

Backup files created in deploy repo:

- `policy.pt.bak_20260614_1724_before_stage1c23400`
- `policy.onnx.bak_20260614_1724_before_stage1c23400`
- `policy.onnx.data.bak_20260614_1724_before_stage1c23400`

Hash validation after sync:

- `policy.pt`: `1b1601cb0594b67fac5de6d8c5abb0989921620a3012ef21642a4f90de80217a`
- `policy.onnx`: `aa464ee9d79561721c1bee59a64c7908c9eac319a338a37a0cb8876ab25f9cf4`
- `policy.onnx.data`: `9c11d9f2cf789fdba9b3ca81a3a8921290b7ab6bdf6eda80f1bbf4b81620db3b`

Deploy YAML validation:

- run-local and deploy YAML hashes match exactly:
  - `LocoMode.yaml`: `c77307444621c178f1cc26a4fb469b6942709edb438a43f21e6e3e176ad7a12e`
  - `LocoMode_lowKp.yaml`: `c77307444621c178f1cc26a4fb469b6942709edb438a43f21e6e3e176ad7a12e`
- parsed deploy config:
  - `policy_path: policy.onnx`
  - `command_dim: 4`
  - `num_obs: 82`
  - `num_actions: 24`
  - `policy_dt: 0.02`
  - `cmd_range.lin_vel_x: [-2.5, 3.75]`
  - `cmd_slew_rate: [2.0, 1.0, 2.0]`
  - `root_height_command: 0.69`
  - `obs_clip: 100.0`
  - `action_scale: 0.25`
  - unique `kps`: `[59.336062, 113.02671]`
  - unique `kds`: `[3.777451, 7.195504]`
- ONNX graph check:
  - input: `obs [1, 82]`
  - output: `actions [1, 24]`

Note:

- The deploy YAML keeps max command at `3.75 m/s`, matching Stage1C training. The `4.0 m/s` eval/play is exploratory and should not be treated as a stable deploy command yet.
- Deploy repo local commit:
  - repo: `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot`
  - branch: `refactor/controller-core-adapters`
  - commit: `60cc7b9 Update Z1 loco policy to Stage1C sprint checkpoint`
  - scope: only the two loco YAML files and three policy artifact files were committed.
  - push status: not pushed, because the current branch has no matching remote branch.

## Stage2A Prepared Config

Date: `2026-06-14`

Reason:

- Stage1C `model_23400.pt` is the current best candidate for `2.5-3.5 m/s`.
- `4.0 m/s` improved but is still exploratory, so the next speed expansion should be smaller than a full `4-5 m/s` jump.
- Stage2A is prepared as a conservative entry into Stage 2, but should not be launched until Stage1C play posture is visually accepted.

Task:

- `magicbot_z1_flat_sprint_amp_stage2a`

Prepared config:

- command `lin_vel_x`: `(-2.5, 4.25)`
- reference motion `min_command_speed`: `2.0`
- reference motion `max_reference_speed`: `4.8`
- velocity tracking reward weight: `1.35`
- `speed_tracking_duration_s`: `2.5`
- AMP reward coefficient: `0.08`
- learning rate: `3.0e-4`
- save interval: `25`
- run name base: `z1_sprint_amp_stage2a_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- AppLauncher registry check passed:
  - task registered: `magicbot_z1_flat_sprint_amp_stage2a`
  - `lin_vel_x=(-2.5, 4.25)`
  - `max_reference_speed=4.8`
  - `motion_prior_enable=True`
  - `motion_prior_reward_coef=0.08`
  - `learning_rate=0.0003`

Suggested launch if Stage1C play is visually acceptable:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage2a \
  --num_envs=10000 \
  --max_iterations=101 \
  --run_name=z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_YYYYMMDD_HHMMSS \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801 \
  --checkpoint=model_23400.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot
```

## Stage1C Play Visual Snapshot

Date: `2026-06-14`

Window capture:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage1c_23400_window_20260614_1732.png`
- xwd source:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage1c_23400_window_20260614_1732.xwd`

Visual note:

- No obvious falling robot in the captured frame.
- Legs appear to be stepping rather than fully stuck.
- Arms do not look like the previous clearly bad front-raised posture in this single-frame check.
- This is only a sanity snapshot, not a full visual pass over start/accelerate/decelerate behavior.

Stage1C play process was stopped after the snapshot to free GPU memory:

```bash
kill 789481
```

## Stage2A Smoke

Date: `2026-06-14`

Command summary:

- task: `magicbot_z1_flat_sprint_amp_stage2a`
- envs: `4`
- max iterations: `1`
- resume checkpoint: Stage1C `model_23400.pt`
- deploy yaml: skipped to avoid touching current deploy artifacts.

Log:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2a_resume23400_smoke_20260614_1735.out`

Run dir:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-34-53_z1_sprint_amp_stage2a_resume23400_smoke_20260614_1735`

Validation:

- exit status: `0`
- AMP runner enabled: `True`
- loaded checkpoint: `model_23400.pt`
- actor observation dim: `82`
- critic observation dim: `87`
- action dim: `24`
- saved config confirmed:
  - `lin_vel_x: (-2.5, 4.25)`
  - `max_reference_speed: 4.8`
  - `track_lin_vel_xy_exp.weight: 1.35`
  - `speed_tracking_duration_s: 2.5`
  - `learning_rate: 0.0003`
  - `motion_prior.reward_coef: 0.08`
  - `save_interval: 25`

## Stage2A Training Run

Date: `2026-06-14`

Run name:

- `z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614`

Launch log:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614.out`

Run dir:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614`

Start checkpoint:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_16-48-51_z1_sprint_amp_stage1c_from23300_cmdx-2p5_3p75_ref2p0_4p2_amp0p10_lr5e-4_save25_env10000_20260614_164801/model_23400.pt`

Launch command summary:

- task: `magicbot_z1_flat_sprint_amp_stage2a`
- envs: `10000`
- max iterations: `101`
- save interval: `25`
- command `lin_vel_x`: `(-2.5, 4.25)`
- AMP reward coefficient: `0.08`
- learning rate: `3.0e-4`

Deploy yaml handling:

- To avoid overwriting the current usable Stage1C deploy config, the real deploy root was not used.
- Deploy yaml root for this training run was redirected to:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614`
- The run-local deploy directory should still be generated under the Stage2A log dir.

Startup status:

- PID: `990099`
- checkpoint load confirmed: Stage1C `model_23400.pt`
- AMP runner enabled: `True`
- generated run-local deploy yaml:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/deploy/LocoMode_lowKp.yaml`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/deploy/LocoMode.yaml`
- generated snapshot deploy yaml:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/policies/loco_mode/config/LocoMode_lowKp.yaml`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/policies/loco_mode/config/LocoMode.yaml`
- global deploy YAML was not overwritten:
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/config/LocoMode.yaml` hash stayed `c77307444621c178f1cc26a4fb469b6942709edb438a43f21e6e3e176ad7a12e`
  - `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/config/LocoMode_lowKp.yaml` hash stayed `c77307444621c178f1cc26a4fb469b6942709edb438a43f21e6e3e176ad7a12e`
- first observed GPU state with DogUrdf17 play still running:
  - total GPU memory used: about `20.9 / 32.6 GB`
  - Z1 Stage2A process memory: about `9.1 GB`
  - GPU utilization: about `85%`
- early online note:
  - iteration `23405-23406` had high speed tracking failure ratio around `0.16-0.18`.
  - by iteration `23407`, speed tracking failure ratio dropped to `0.0240`, timeout ratio rose to `0.9555`, and mean reward recovered to `0.27`.
  - continue monitoring until `model_23425.pt` before deciding whether the early failures are transient.

Final online training state:

- final checkpoint: `model_23500.pt`
- saved at: `2026-06-14 17:44:09`
- iteration `23500` snapshot:
  - Mean reward: `7.97`
  - Mean episode length: `977.32`
  - timeout ratio: `0.9227`
  - head/shoulder contact ratio: `0.0606`
  - body contact ratio: `0.0000`
  - speed tracking failure ratio: `0.0167`
- checkpoints saved:
  - `model_23425.pt`
  - `model_23450.pt`
  - `model_23475.pt`
  - `model_23500.pt`

## Stage2A Fixed-Speed Eval

Date: `2026-06-14`

Task:

- `magicbot_z1_flat_sprint_amp_stage2a`

Eval setup:

- envs: `32`
- warmup: `3.0 s`
- measured duration: `8.0 s`
- speeds: `2.5`, `3.0`, `3.5`, `4.0`, `4.25 m/s`
- log prefix: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_fixed_eval_stage2a_model_*_20260614_1745.out`

| checkpoint | cmd vx | mean vx | abs err | resets |
|---|---:|---:|---:|---:|
| 23425 | 2.5 | 2.4886 | 0.0917 | 0 |
| 23425 | 3.0 | 2.8170 | 0.2534 | 3 |
| 23425 | 3.5 | 3.0369 | 0.5110 | 7 |
| 23425 | 4.0 | 3.2050 | 0.8140 | 8 |
| 23425 | 4.25 | 2.2012 | 2.0565 | 24 |
| 23450 | 2.5 | 2.4285 | 0.1093 | 0 |
| 23450 | 3.0 | 2.7889 | 0.2500 | 7 |
| 23450 | 3.5 | 3.1383 | 0.3968 | 5 |
| 23450 | 4.0 | 2.7067 | 1.3103 | 18 |
| 23450 | 4.25 | 2.0614 | 2.1937 | 25 |
| 23475 | 2.5 | 2.4305 | 0.1049 | 1 |
| 23475 | 3.0 | 2.8943 | 0.1485 | 1 |
| 23475 | 3.5 | 2.8400 | 0.6843 | 12 |
| 23475 | 4.0 | 2.6137 | 1.3944 | 15 |
| 23475 | 4.25 | 2.1953 | 2.0553 | 23 |
| 23500 | 2.5 | 2.4376 | 0.1114 | 2 |
| 23500 | 3.0 | 2.9009 | 0.1737 | 1 |
| 23500 | 3.5 | 3.2785 | 0.2809 | 4 |
| 23500 | 4.0 | 2.9576 | 1.0577 | 13 |
| 23500 | 4.25 | 1.7778 | 2.4772 | 38 |

Fixed-speed conclusion:

- `model_23425.pt` is the best high-speed Stage2A candidate.
- Compared with Stage1C `23400`, `23425` substantially improves `4.0 m/s` tracking.
- `model_23500.pt` is better at `3.5 m/s` and preserves `3.0 m/s`, but it loses high-speed stability at `4.0-4.25 m/s`.
- `4.25 m/s` is still not stable in any Stage2A checkpoint and should not yet be treated as solved.

## Stage2A Gait Quality Eval

Date: `2026-06-14`

Eval logs:

- `23425`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_gait_quality_stage2a_model_23425_20260614_1755.out`
- `23500`: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_gait_quality_stage2a_model_23500_20260614_1755.out`

| checkpoint | cmd vx | mean vx | abs err | resets | tilt xy | p90 swing foot z | single stance | flight | arm abs | shoulder pitch abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 23425 | 2.5 | 2.4883 | 0.0886 | 0 | 0.0398 | 0.2679 | 0.9119 | 0.0867 | 0.2346 | 0.1445 |
| 23425 | 3.0 | 2.8079 | 0.2546 | 1 | 0.0469 | 0.2786 | 0.8762 | 0.1084 | 0.2915 | 0.1952 |
| 23425 | 3.5 | 2.9217 | 0.6112 | 2 | 0.0565 | 0.2839 | 0.8352 | 0.1233 | 0.3250 | 0.2383 |
| 23425 | 4.0 | 3.1991 | 0.8093 | 2 | 0.0685 | 0.2932 | 0.8228 | 0.1436 | 0.3622 | 0.2819 |
| 23425 | 4.25 | 2.8470 | 1.4055 | 6 | 0.0725 | 0.2893 | 0.8013 | 0.1453 | 0.3536 | 0.2900 |
| 23500 | 2.5 | 2.4528 | 0.0971 | 0 | 0.0558 | 0.2609 | 0.9058 | 0.0942 | 0.1933 | 0.1321 |
| 23500 | 3.0 | 2.9086 | 0.1477 | 0 | 0.0493 | 0.2760 | 0.8606 | 0.1384 | 0.2648 | 0.1882 |
| 23500 | 3.5 | 3.0589 | 0.4775 | 3 | 0.0496 | 0.2817 | 0.8297 | 0.1520 | 0.3059 | 0.2427 |
| 23500 | 4.0 | 2.8909 | 1.1152 | 7 | 0.0557 | 0.2815 | 0.8200 | 0.1502 | 0.3237 | 0.2777 |
| 23500 | 4.25 | 2.1769 | 2.0733 | 10 | 0.0659 | 0.2779 | 0.8053 | 0.1305 | 0.3107 | 0.2771 |

Gait conclusion:

- `23425` is better for the Stage 2 high-speed objective, especially at `4.0 m/s`.
- `23500` has lower arm/shoulder offsets at low/mid speeds, but gives up too much high-speed tracking.
- Current Stage2A selection:
  - high-speed candidate: `model_23425.pt`
  - smoother mid-speed backup: `model_23500.pt`
- Next action: play `model_23425.pt` with command range `-2.5..4.25` before considering it deployable.

## Play Check: Stage2A 23425

Date: `2026-06-14`

Command summary:

- task: `magicbot_z1_flat_sprint_amp_stage2a`
- checkpoint: `model_23425.pt`
- envs: `50`
- command range: `lin_vel_x=(-2.5, 4.25)`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
- velocity debug visualization: enabled

Status:

- PID: `1364645`
- log: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_cmdx-2p5_4p25_20260614_1806.out`
- command range confirmed in log.
- actor observation dim confirmed: `82`
- critic observation dim confirmed: `87`
- exported artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/exported/policy.onnx`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/exported/policy.onnx.data`

Window capture:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_window_20260614_1808.png`
- xwd source:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_window_20260614_1808.xwd`
- clearer still frame:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_window_clear_20260614_1812.png`
- short video:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_window_20260614_1812.mp4`
- extracted frames:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_frames_20260614_1812/`
- frame montage:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_play_stage2a_23425_frames_20260614_1812_montage.png`

Visual note:

- The play window is live and no obvious fallen robot is visible in the captured frame.
- The captured camera view is too close and partly blocked by the simulation settings panel, so this should be treated only as a weak sanity check.
- The 6-second frame montage shows robots moving with visible stepping and arm swing; no continuous full-body collapse is obvious.
- The view is still not a full visual acceptance test because the camera framing is poor and the simulation settings panel remains visible.
- Do not sync Stage2A `23425` into the deploy repo until a better visual pass confirms start, acceleration, hold, and deceleration behavior.

To stop this play:

```bash
kill 1364645
```

Play process note:

- The Stage2A play process was stopped before the native deploy/MuJoCo smoke checks below.

## Native Deploy/MuJoCo Smoke: Stage2A 23425

Date: `2026-06-14`

Snapshot config:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/policies/loco_mode/config/LocoMode.yaml`
- command range: `lin_vel_x=(-2.5, 4.25)`, `lin_vel_y=(-0.5, 0.5)`, `ang_vel_z=(-1.57, 1.57)`
- dimensions: `command_dim=4`, `num_obs=82`, `num_actions=24`
- PD: `kp=[113.0267, 59.3361]`, `kd=[7.1955, 3.7775]`

Snapshot model files copied from the Stage2A export into the snapshot-local `policies/loco_mode/model/` directory:

- `policy.onnx`: `0c7c326e72f76d7da4093bb199f2b94582224b2c9e9e759073212912b02e3a70`
- `policy.onnx.data`: `ed7e7fb12a0bddebce67869ed1d7fd37f5893d0ab9df72d7d6362afd5793d925`

Global deploy repo was not overwritten:

- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/config/LocoMode.yaml`: `c77307444621c178f1cc26a4fb469b6942709edb438a43f21e6e3e176ad7a12e`
- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/model/policy.onnx`: `aa464ee9d79561721c1bee59a64c7908c9eac319a338a37a0cb8876ab25f9cf4`
- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/policies/loco_mode/model/policy.onnx.data`: `9c11d9f2cf789fdba9b3ca81a3a8921290b7ab6bdf6eda80f1bbf4b81620db3b`

ONNX shape and native latency:

- ONNX graph: `obs [1,82] -> actions [1,24]`
- benchmark command: `scripts/run_onnx_benchmark_native.sh --model <snapshot>/policy.onnx --iters 2000 --warmup 100 --threads 1 --obs-dim 82`
- result: `mean_ms=0.015460`, `p95_ms=0.018791`, `p99_ms=0.020972`, `throughput_hz=63954.81`

C++ consistency fix applied in the deploy repo working tree:

- `controller_cpp/include/mujoco_sim_adapter.h`: default `zero_head_target=false`
- `controller_cpp/src/dual_inference_rate.cpp`: local PD no longer forces the head target to zero
- Reason: current policy is 24-action and training controls `head_joint`; MuJoCo/deploy smoke should use the policy head target unless a caller explicitly overrides it.
- `git diff --check` passed for these two files.

Pure-sim smoke results after the head-target fix:

| normalized vx | approx physical vx | min base height | max gravity xy | max root xy drift | max tau | max dq | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 0.000 | 0.6879 | 0.0932 | 0.0461 | 53.70 | 8.60 | stand healthy |
| 0.1 | 0.425 | 0.6887 | 0.1092 | 1.1536 | 50.00 | 10.43 | stable |
| 0.2 | 0.850 | 0.6863 | 0.1490 | 3.0635 | 66.24 | 12.56 | stable |
| 0.3 | 1.275 | 0.6792 | 0.1660 | 4.6595 | 77.78 | 14.59 | stable |
| 0.5 | 2.125 | 0.0867 | 1.0000 | 1.6114 | 120.00 | 28.24 | falls in MuJoCo |

Pre-fix reference checks:

- `vx=0.0`: healthy, `min_base_height=0.6877`, `max_gravity_xy=0.0990`
- `vx=0.5`: fell, `min_base_height=0.0826`, `max_gravity_xy=1.0000`
- `vx=1.0`: fell, `min_base_height=0.0747`, `max_gravity_xy=1.0000`

Conclusion:

- Native ONNX loading, dimensions, and latency are good.
- C++ pure-sim can stand and move at low normalized commands.
- The Stage2A `23425` policy still fails in native MuJoCo around normalized `vx=0.5` (about `2.125 m/s`) even though Isaac fixed-speed eval is healthy at `2.5 m/s`.
- Head target zeroing was a real deploy/MuJoCo inconsistency and has been fixed locally, but it is not the main high-speed MuJoCo failure.
- Next investigation should prove the action/joint order against Isaac's runtime joint order, then compare MuJoCo XML/contact/inertia with the Isaac training asset.

## Action Order Check: Stage2A Z1

Date: `2026-06-14`

Query:

- created a 1-env headless Isaac task: `magicbot_z1_flat_sprint_amp_stage2a`
- output file: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2a_isaac_joint_order_names_20260614_1852.txt`

Isaac runtime joint/action order:

```text
0: left_hip_pitch_joint
1: right_hip_pitch_joint
2: waist_yaw_joint
3: left_hip_roll_joint
4: right_hip_roll_joint
5: head_joint
6: left_shoulder_pitch_joint
7: right_shoulder_pitch_joint
8: left_hip_yaw_joint
9: right_hip_yaw_joint
10: left_shoulder_roll_joint
11: right_shoulder_roll_joint
12: left_knee_joint
13: right_knee_joint
14: left_shoulder_yaw_joint
15: right_shoulder_yaw_joint
16: left_ankle_pitch_joint
17: right_ankle_pitch_joint
18: left_elbow_joint
19: right_elbow_joint
20: left_ankle_roll_joint
21: right_ankle_roll_joint
22: left_wrist_yaw_joint
23: right_wrist_yaw_joint
```

Conclusion:

- Isaac runtime order matches `legged_lab/utils/magicbot_deploy_yaml.py::LAB_JOINT_NAMES`.
- Deploy YAML `joint2motor_idx` maps this Isaac/policy order into MuJoCo actuator order.
- Action/joint order is therefore not the likely cause of the Stage2A native MuJoCo high-speed fall.
- Next likely mismatch class: MuJoCo XML/contact/inertia/solver behavior versus the Isaac training asset, especially feet/contact and shoulder axis reorientation warnings.

## Native Deploy Fix: Target Rate Limit

Date: `2026-06-14`

Finding:

- C++ deploy applied extra target limiting after ONNX inference:
  - `torque_limited_target(...)`
  - `clamp_and_rate_limit(..., max_target_rate=4 rad/s, policy_dt=0.02)`
- This allowed only `0.08 rad` target movement per policy step.
- Isaac training directly applies `action * action_scale + default_joint_pos` to the implicit actuator target every step; it does not impose this additional 4 rad/s command-target slew limit.
- This mismatch explains the native MuJoCo symptom where low speeds worked but the robot could not open its stride around normalized `vx=0.5`.

Deploy working tree changes:

- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/controller_cpp/include/controller_core.h`
  - default `ControllerCoreOptions::max_target_rate`: `4.0 -> 25.0`
- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/controller_cpp/include/mujoco_sim_adapter.h`
  - default `zero_head_target`: `true -> false`
- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/controller_cpp/src/dual_inference_rate.cpp`
  - added `--max-target-rate`
  - default `25`
  - pure-sim core uses the parsed value
  - local real-state-sim PD no longer zeros the head target
- `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot/controller_cpp/src/magicbot_z1_loco_onnx.cpp`
  - default `--max-target-rate`: `25`
  - usage text documents the default

Validation:

- `git diff --check` passed for the touched deploy C++ files.
- `scripts/run_magicbot_loco_native.sh --config <snapshot LocoMode.yaml> --dry-run --skip-network-check`
  - builds `magicbot_z1_loco_onnx`
  - ONNX input/output: `82 -> 24`
  - dry-run target sample range: `[-0.327..0.801]`
- `scripts/run_mujoco_loco_viewer_native.sh --help`
  - builds `mujoco_loco_viewer`
- Native dual smoke, Stage2A `23425`, default `max_target_rate=25`:
  - command: normalized `vx=0.5` (about `2.125 m/s`)
  - `min_base_height=0.688170`
  - `max_gravity_xy=0.134253`
  - `max_root_xy_drift=6.829144`
  - `max_policy_target_jump=0.905066`
  - result: stable over 5s pure-sim
- Native dual smoke, Stage2A `23425`, explicit `max_target_rate=25`:
  - normalized `vx=0.75` (about `3.1875 m/s`): `min_base_height=0.670252`, `max_gravity_xy=0.168753`
  - normalized `vx=1.0` (about `4.25 m/s`): `min_base_height=0.661294`, `max_gravity_xy=0.149003`
  - result: both remain upright over 5s pure-sim

Comparison to old default:

- old `max_target_rate=4`, normalized `vx=0.5`:
  - `min_base_height=0.086692`
  - `max_gravity_xy=1.000000`
  - result: falls in MuJoCo

Conclusion:

- The native MuJoCo high-speed failure was primarily caused by deploy-side target slew limiting, not action order.
- The head-target fix is still correct for 24-action policy consistency, but it was not the main fall cause.
- Keep `max_target_rate=25` as the deployment default for Stage2A sprint policies unless real-robot safety testing requires a lower value.
- Deploy repo still has a dirty working tree with unrelated changes, but the target-rate/head-target fix was isolated into a minimal commit and pushed:
  - repo: `/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot`
  - branch: `refactor/controller-core-adapters`
  - commit: `c49cce5 Fix Z1 sprint deploy target limiting`
  - remote: `origin/refactor/controller-core-adapters`
- Post-commit default-path smoke:
  - command: Stage2A `23425`, normalized `vx=0.5`, default `max_target_rate=25`, `duration=3`
  - summary json: `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2a_23425_dual_puresim_vx0p5_rate25default_3s_20260614_goal_continue.json`
  - `min_base_height=0.688170`, `max_gravity_xy=0.134253`, `max_policy_target_jump=0.905066`
  - result: stable over 3s pure-sim

## Stage2B Sprint AMP Launch

Date: `2026-06-14`

Purpose:

- Continue from the best Stage2A high-speed checkpoint without touching the stable baseline.
- Expand the positive speed command one conservative step beyond Stage2A, aiming to make `4.0-4.5 m/s` more reliable before attempting the later `5-6 m/s` stage.
- Keep the change small enough that regressions can be attributed to speed-range pressure rather than a large reward or AMP change.

Protected baseline:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958/model_23000.pt`

Start checkpoint:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/model_23425.pt`

Config changes:

- New task: `magicbot_z1_flat_sprint_amp_stage2b`
- `lin_vel_x`: `(-2.5, 4.75)`
- `reference_motion.max_reference_speed`: `5.2`
- `track_lin_vel_xy_exp.weight`: `1.4`
- `speed_tracking_duration_s`: unchanged at `2.5`
- AMP reward coefficient: unchanged at `0.08`
- learning rate: unchanged at `3.0e-4`
- save interval: unchanged at `25`
- reset joint randomization, fall recovery, push, friction/mass/domain randomization: unchanged from current Z1 sprint config.

Validation before launch:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- AppLauncher registry check passed:
  - task registered: `magicbot_z1_flat_sprint_amp_stage2b`
  - `lin_vel_x=(-2.5, 4.75)`
  - `max_reference_speed=5.2`
  - `track_lin_vel_xy_exp.weight=1.4`
  - `motion_prior_enable=True`
  - `motion_prior_reward_coef=0.08`
  - `learning_rate=0.0003`
  - `save_interval=25`
- Smoke training passed:
  - task: `magicbot_z1_flat_sprint_amp_stage2b`
  - envs: `64`
  - max iterations: `1`
  - loaded checkpoint: Stage2A `model_23425.pt`
  - actor shape: `82 -> 24`
  - critic shape: `87 -> 1`
  - AMP runner: enabled

Resource note:

- A separate DogUrdf17 Isaac play process was already using about `9.6 GB` GPU memory.
- Initial ordinary-background launch attempts did not stay alive under the current shell/tool session, so the formal run is launched with `setsid`.
- Formal env count was reduced to `4096` to avoid interfering with the existing Isaac play process.

Formal run:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage2b \
  --num_envs=4096 \
  --max_iterations=126 \
  --run_name=z1_sprint_amp_stage2b_from23425_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4_save25_env4096_20260614_185305 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614 \
  --checkpoint=model_23425.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2b_from23425_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4_save25_env4096_20260614_185305 \
  --device=cuda:0 \
  --kit_args=--portable
```

Run artifacts:

- PID at launch: `1928745`
- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2b_from23425_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4_save25_env4096_20260614_185305.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_18-53-36_z1_sprint_amp_stage2b_from23425_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4_save25_env4096_20260614_185305`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2b_from23425_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4_save25_env4096_20260614_185305/policies/loco_mode/config/LocoMode.yaml`

Initial online indicators:

- Training loaded Stage2A `model_23425.pt`.
- Generated MagicBot Z1 deploy YAML files in both the run directory and deploy snapshot.
- `Learning iteration 23425/23551` started successfully.
- First observed iterations:
  - iteration `23425`: mean reward `-4.33`, mean episode length `18.43`, timeout ratio `0.7553`, head/shoulder ratio `0.2447`, speed failure ratio `0.0`
  - iteration `23426`: mean reward `-22.68`, mean episode length `37.51`, timeout ratio `0.6845`, head/shoulder ratio `0.3155`, speed failure ratio `0.0`
  - iteration `23427`: mean reward `-4.55`, mean episode length `60.09`, timeout ratio `0.8175`, head/shoulder ratio `0.1721`, speed failure ratio `0.0`

Next checks:

- Watch whether episode length recovers toward Stage2A levels after the first high-speed adaptation window.
- First useful checkpoint is expected at `model_23450.pt`.
- Evaluate fixed speeds after at least `model_23450.pt` and compare against Stage2A `model_23425.pt` at `3.5`, `4.0`, `4.25`, and `4.5 m/s`.

## Stage2B Sprint AMP Evaluation

Date: `2026-06-14`

Formal run:

- `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_18-53-36_z1_sprint_amp_stage2b_from23425_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4_save25_env4096_20260614_185305`

Final online training indicators at `model_23550.pt`:

- mean reward: `7.92`
- mean episode length: `973.95`
- timeout ratio: `0.9193`
- head/shoulder contact ratio: `0.0122`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0685`

Important finding:

- Online episode length and timeout ratio recovered, but fixed-speed evaluation shows high-speed command following degraded after further Stage2B training.
- Therefore the final checkpoint `model_23550.pt` is not a good sprint continuation point.

Fixed-speed eval settings:

- num envs: `64`
- warmup: `2 s`
- measured duration: `6 s`
- noise disabled, push disabled, heading/y velocity/yaw fixed to zero.

Reference Stage2A baseline, original task/checkpoint:

- checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/model_23425.pt`
- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2a_23425_fixed_speed_eval_sameparams_20260614_1914.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 2.50 | 2.4701 | 0.1101 | 0 |
| 3.00 | 2.8227 | 0.2499 | 7 |
| 3.50 | 2.4723 | 1.0666 | 10 |
| 4.00 | 2.5841 | 1.4287 | 17 |
| 4.25 | 2.1426 | 2.1110 | 23 |
| 4.50 | 1.2854 | 3.2149 | 40 |
| 4.75 | 0.4092 | 4.3409 | 56 |

Stage2B `model_23450.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2b_23450_fixed_speed_eval_20260614_1906.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 2.50 | 2.4943 | 0.1339 | 2 |
| 3.00 | 2.8504 | 0.2728 | 2 |
| 3.50 | 3.0137 | 0.5578 | 5 |
| 4.00 | 2.2022 | 1.8038 | 23 |
| 4.25 | 2.0116 | 2.2415 | 35 |
| 4.50 | 1.1680 | 3.3324 | 46 |
| 4.75 | 0.8478 | 3.9022 | 52 |

Stage2B `model_23475.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2b_23475_fixed_speed_eval_20260614_1917.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 3.50 | 2.7487 | 0.7782 | 7 |
| 4.00 | 1.9361 | 2.0689 | 22 |
| 4.25 | 1.6189 | 2.6357 | 34 |
| 4.50 | 1.3112 | 3.1893 | 42 |

Stage2B `model_23550.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2b_23550_fixed_speed_eval_20260614_1901.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 2.50 | 2.4284 | 0.1332 | 2 |
| 3.00 | 2.6992 | 0.3789 | 6 |
| 3.50 | 2.0735 | 1.4608 | 24 |
| 4.00 | 1.2971 | 2.7059 | 39 |
| 4.25 | 0.6809 | 3.5700 | 47 |
| 4.50 | 0.1777 | 4.3223 | 66 |
| 4.75 | -0.2419 | 4.9919 | 62 |

Conclusion:

- `model_23550.pt` is degraded and should not be used for deploy or continuation.
- `model_23475.pt` also loses high-speed tracking relative to the Stage2A baseline.
- `model_23450.pt` has a local improvement around `3.5 m/s` but is worse than Stage2A `23425` at `4.0-4.25 m/s`.
- Recommended continuation checkpoint remains Stage2A `model_23425.pt`, not Stage2B.
- Next training attempt should reduce the jump from Stage2A:
  - avoid immediate `4.75` command max;
  - likely use `4.5` or keep `4.25` while changing sampling/curriculum;
  - avoid increasing velocity tracking pressure further until the model can start and hold `4.0+` in fixed-speed eval.

## Stage2C Forward-Focus Sprint AMP

Date: `2026-06-14`

Reason:

- Stage2B showed that directly expanding full command range to `4.75 m/s` can preserve online timeout ratio while degrading fixed-speed high-speed tracking.
- Stage2C-forward keeps the Stage2A speed ceiling but increases the density of forward sprint commands and reduces lateral/yaw distractions.
- This is an intermediate sprint-skill absorption stage, not the final all-direction deploy command distribution.

Config:

- New task: `magicbot_z1_flat_sprint_amp_stage2c_forward`
- start checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614/model_23425.pt`
- `lin_vel_x`: `(0.0, 4.25)`
- `lin_vel_y`: `(-0.2, 0.2)`
- `ang_vel_z`: `(-0.4, 0.4)`
- heading command: `False`
- standing env ratio: `0.05`
- heading env ratio: `0.0`
- `reference_motion.max_reference_speed`: `4.8`
- `track_lin_vel_xy_exp.weight`: `1.35`
- AMP reward coefficient: `0.08`
- learning rate: `3.0e-4`
- save interval: `25`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- AppLauncher registry check passed:
  - task registered: `magicbot_z1_flat_sprint_amp_stage2c_forward`
  - `lin_vel_x=(0.0, 4.25)`
  - `lin_vel_y=(-0.2, 0.2)`
  - `ang_vel_z=(-0.4, 0.4)`
  - `heading_command=False`
  - `rel_standing_envs=0.05`
  - `rel_heading_envs=0.0`
  - `motion_prior_enable=True`
  - `motion_prior_reward_coef=0.08`
- Smoke training passed:
  - envs: `64`
  - max iterations: `1`
  - loaded Stage2A `model_23425.pt`
  - actor shape: `82 -> 24`
  - critic shape: `87 -> 1`
  - AMP runner: enabled

Formal run:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage2c_forward \
  --num_envs=4096 \
  --max_iterations=51 \
  --run_name=z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_17-37-04_z1_sprint_amp_stage2a_from23400_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env10000_20260614_173614 \
  --checkpoint=model_23425.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309 \
  --device=cuda:0 \
  --kit_args=--portable
```

Run artifacts:

- PID at launch: `2271604`
- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_19-23-39_z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23425.pt`
  - `model_23450.pt`
  - `model_23475.pt`

Final online indicators at `model_23475.pt`:

- mean reward: `1.99`
- mean episode length: `960.27`
- timeout ratio: `0.9396`
- head/shoulder contact ratio: `0.0604`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0`

Fixed-speed eval settings:

- num envs: `64`
- warmup: `2 s`
- measured duration: `6 s`
- noise disabled, push disabled, heading/y velocity/yaw fixed to zero.

Stage2C-forward `model_23450.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2c_forward_23450_fixed_speed_eval_20260614_1931.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 2.50 | 2.4594 | 0.1195 | 1 |
| 3.00 | 2.7206 | 0.3558 | 2 |
| 3.50 | 2.6384 | 0.9139 | 8 |
| 4.00 | 2.0925 | 1.9213 | 16 |
| 4.25 | 2.0727 | 2.1822 | 17 |
| 4.50 | 1.1140 | 3.3870 | 38 |

Stage2C-forward `model_23475.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2c_forward_23475_fixed_speed_eval_20260614_1928.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 2.50 | 2.4716 | 0.1108 | 0 |
| 3.00 | 2.7857 | 0.3179 | 2 |
| 3.50 | 3.0088 | 0.5859 | 6 |
| 4.00 | 2.6205 | 1.4136 | 15 |
| 4.25 | 1.8930 | 2.3707 | 35 |
| 4.50 | 0.9488 | 3.5520 | 52 |

Gait eval for Stage2C-forward `model_23475.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2c_forward_23475_gait_eval_20260614_1933.out`

| target | mean vx | abs err | resets | tilt xy | p90 swing foot z | single stance | double stance | flight | contact transitions/env/s | arm abs offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.50 | 2.9235 | 0.6502 | 4 | 0.0616 | 0.2840 | 0.8588 | 0.0244 | 0.1169 | 7.4531 | 0.3139 |
| 4.00 | 2.3005 | 1.7156 | 10 | 0.0752 | 0.2805 | 0.8166 | 0.0736 | 0.1098 | 7.1198 | 0.3109 |

Conclusion:

- Stage2C-forward is more useful than Stage2B:
  - `model_23475.pt` improves `3.5 m/s` and slightly improves `4.0 m/s` fixed-speed tracking relative to the Stage2A `23425` same-parameter eval.
  - gait quality at `3.5 m/s` shows active swing (`p90_swing_foot_z=0.2840`) and mostly single-stance running/walking rhythm.
- It is not yet a real `4.25+ m/s` solution:
  - `model_23475.pt` degrades at `4.25 m/s`;
  - `model_23450.pt` has fewer `4.25 m/s` resets but does not improve mean speed.
- Current best continuation choices:
  - for `3.5-4.0 m/s` forward sprint style: Stage2C-forward `model_23475.pt`
  - for broader command retention and original deploy behavior: Stage2A `model_23425.pt`
- Next attempt should combine the two:
  - continue from Stage2C-forward `23475` for a very short run only if validating `4.0` style;
  - or restart from Stage2A `23425` with forward-focus but add a gentler high-speed curriculum instead of immediately evaluating/forcing `4.25+`.

## Stage2D High-Reference Sprint AMP

Date: `2026-06-14`

Reason:

- Stage2B and Stage2C showed that broadening command range alone is not enough for `4.25 m/s`.
- `sprint1_subject2` speed distribution is heavily skewed to low speed:
  - total frames: `13657`
  - duration: `273.14 s`
  - mean speed: `1.133 m/s`
  - max speed: `5.249 m/s`
  - frames `3.5-4.0`: `715` (`5.24%`)
  - frames `4.0-4.25`: `306` (`2.24%`)
  - frames `4.25-4.5`: `244` (`1.79%`)
  - frames `4.5-4.75`: `150` (`1.10%`)
  - frames `4.75-5.0`: `57` (`0.42%`)
- Existing AMP expert sampler draws uniformly from eligible frames, so the high-speed sprint prior is sparse unless the eligible range is narrowed.

Config:

- New task: `magicbot_z1_flat_sprint_amp_stage2d_highref`
- start checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_19-23-39_z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309/model_23475.pt`
- `lin_vel_x`: `(2.5, 4.25)`
- `lin_vel_y`: `(-0.15, 0.15)`
- `ang_vel_z`: `(-0.3, 0.3)`
- standing env ratio: `0.0`
- heading command: `False`
- `reference_motion.min_command_speed`: `3.0`
- `reference_motion.max_reference_speed`: `4.8`
- `reference_motion.speed_match_tolerance`: `0.5`
- `motion_prior.reward_min_command_speed`: `3.0`
- AMP reward coefficient: `0.08`
- learning rate: `2.0e-4`
- save interval: `25`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
  - `legged_lab/scripts/eval_fixed_speed.py`
- AppLauncher registry check passed:
  - task registered: `magicbot_z1_flat_sprint_amp_stage2d_highref`
  - `lin_vel_x=(2.5, 4.25)`
  - `reference_min_command_speed=3.0`
  - `motion_prior_reward_min_command_speed=3.0`
  - `learning_rate=0.0002`
- Smoke training passed:
  - envs: `64`
  - max iterations: `1`
  - loaded Stage2C-forward `model_23475.pt`
  - actor shape: `82 -> 24`
  - critic shape: `87 -> 1`
  - AMP runner: enabled
  - log confirmed `Config/reference_motion_min_speed=3.0000`

Eval tool update:

- `legged_lab/scripts/eval_fixed_speed.py` now reports reset reasons per fixed-speed target:
  - `timeout_resets`
  - `head_shoulder_resets`
  - `body_contact_resets`
  - `speed_tracking_resets`
  - `other_resets`
- Implementation note:
  - The evaluator wraps `env._log_reset_reasons()` and accumulates counts at the moment the environment records reset reasons.
  - Directly reading reset buffers after `env.step()` was not reliable because reset can clear those buffers.

Formal run:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/LeggedLab \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -u legged_lab/scripts/train.py \
  --task=magicbot_z1_flat_sprint_amp_stage2d_highref \
  --num_envs=4096 \
  --max_iterations=26 \
  --run_name=z1_sprint_amp_stage2d_highref_fromstage2c23475_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4_save25_env4096_20260614_194056 \
  --logger=tensorboard \
  --resume=True \
  --load_run=2026-06-14_19-23-39_z1_sprint_amp_stage2c_forward_from23425_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4_save25_env4096_20260614_192309 \
  --checkpoint=model_23475.pt \
  --headless \
  --deploy_yaml_root=/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2d_highref_fromstage2c23475_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4_save25_env4096_20260614_194056 \
  --device=cuda:0 \
  --kit_args=--portable
```

Run artifacts:

- PID at launch: `2471240`
- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2d_highref_fromstage2c23475_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4_save25_env4096_20260614_194056.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_19-41-25_z1_sprint_amp_stage2d_highref_fromstage2c23475_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4_save25_env4096_20260614_194056`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2d_highref_fromstage2c23475_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4_save25_env4096_20260614_194056/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23475.pt`
  - `model_23500.pt`

Final online indicators at `model_23500.pt`:

- mean reward: `-7.94`
- mean episode length: `602.51`
- timeout ratio: `0.9692`
- head/shoulder contact ratio: `0.0308`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0`

Fixed-speed eval, Stage2D-highref `model_23500.pt`:

- eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2d_highref_23500_fixed_speed_eval_20260614_1944.out`

| target | mean vx | abs err | resets |
| --- | ---: | ---: | ---: |
| 2.50 | 2.5410 | 0.1017 | 1 |
| 3.00 | 2.9356 | 0.1916 | 1 |
| 3.50 | 3.1217 | 0.4688 | 5 |
| 4.00 | 2.5571 | 1.4752 | 20 |
| 4.25 | 1.8823 | 2.3767 | 41 |
| 4.50 | 1.5003 | 3.0011 | 48 |

High-speed reset-reason eval, Stage2D-highref `model_23500.pt`:

- reason eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2d_highref_23500_fixed_speed_reasons_v3_20260614_1953.out`
- repeat eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2d_highref_23500_fixed_speed_repeat_20260614_1957.out`

| target | mean vx | resets | head/shoulder | speed tracking | body contact | timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.00 | 2.9462 | 6 | 1 | 5 | 0 | 0 |
| 4.25 | 2.2757 | 25 | 5 | 20 | 0 | 0 |
| 4.00 repeat | 2.6345 | 18 | 2 | 16 | 0 | 0 |
| 4.25 repeat | 2.3298 | 24 | 5 | 19 | 0 | 0 |

Stage2C-forward `model_23475.pt` high-speed reason comparison:

- reason eval log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_stage2c_forward_23475_fixed_speed_reasons_20260614_1955.out`

| target | mean vx | resets | head/shoulder | speed tracking | body contact | timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.00 | 2.4701 | 21 | 3 | 18 | 0 | 0 |
| 4.25 | 2.0378 | 33 | 4 | 29 | 0 | 0 |

Conclusion:

- Stage2D-highref is useful for `3.0-3.5 m/s`:
  - `3.0 m/s` is close to target with low reset count.
  - `3.5 m/s` improves over Stage2C-forward.
- Stage2D-highref gives a noisy but real improvement at `4.0 m/s` when evaluated from high-speed starts.
- `4.25 m/s` is still not solved.
- The dominant failure mode at `4.0-4.25 m/s` is speed tracking failure, not body contact or timeout:
  - this means the robot usually remains upright enough, but cannot keep commanded speed for the required window.
- Next direction should target acceleration/stride opening and high-speed tracking directly, not fall-contact tuning:
  - keep `speed_tracking_duration_s=2.5`;
  - do not continue Stage2D for many more iterations without checkpoint-by-checkpoint eval;
  - consider a short Stage2E with high-speed command distribution plus either stronger forward-speed reward shaping or command-slew/eval alignment, while preserving Stage2A `23425` and Stage2C/Stage2D candidates.

## Stage2E Forward-Progress Sprint AMP

Date: `2026-06-14`

Reason:

- Stage2D proved that the robot can stay upright at high-speed commands, but `4.0-4.25 m/s` failures were still dominated by speed-tracking termination.
- The exponential velocity reward becomes weak when commanded speed is far above actual speed.
- Stage2E adds a small non-saturating forward-progress term for high forward commands so the policy still gets a useful gradient while learning to open stride and accelerate.

Code/config changes:

- New reward function:
  - `legged_lab/mdp/rewards.py::forward_speed_progress`
  - reward is `yaw_frame_vx / command_x`, clamped to `[0, 1]`, active only for `command_x >= 3.0`.
- New task: `magicbot_z1_flat_sprint_amp_stage2e_progress`
- `lin_vel_x`: `(3.0, 4.25)`
- `lin_vel_y`: `(-0.1, 0.1)`
- `ang_vel_z`: `(-0.25, 0.25)`
- standing env ratio: `0.0`
- heading command: `False`
- `reference_motion.min_command_speed`: `3.0`
- `reference_motion.max_reference_speed`: `4.8`
- `reference_motion.speed_match_tolerance`: `0.5`
- `track_lin_vel_xy_exp.weight`: `1.35`
- `track_lin_vel_xy_exp.std`: `0.75`
- `forward_speed_progress.weight`: `0.35`
- AMP reward coefficient: `0.08`
- `motion_prior.reward_min_command_speed`: `3.0`
- learning rate: `1.0e-4`
- save interval: `25`

Validation:

- AppLauncher registry/config check passed:
  - task registered: `magicbot_z1_flat_sprint_amp_stage2e_progress`
  - `lin_vel_x=(3.0, 4.25)`
  - `reference_motion=(min=3.0, max=4.8, tolerance=0.5)`
  - `forward_speed_progress.weight=0.35`
  - `motion_prior.enable=True`
  - `motion_prior.reward_coef=0.08`
  - `motion_prior.reward_min_command_speed=3.0`
  - `learning_rate=0.0001`
- Smoke training passed:
  - envs: `64`
  - max iterations: `1`
  - start checkpoint: Stage2D `model_23500.pt`
  - `env.yaml` contains `forward_speed_progress`
  - actor shape remains `82 -> 24`
  - critic shape remains `87 -> 1`

Formal run:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2e_progress \
  --num_envs 4096 \
  --headless \
  --resume True \
  --load_run 2026-06-14_19-41-25_z1_sprint_amp_stage2d_highref_fromstage2c23475_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4_save25_env4096_20260614_194056 \
  --checkpoint model_23500.pt \
  --max_iterations 26 \
  --run_name z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940
```

Run artifacts:

- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-10-09_z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23500.pt`
  - `model_23525.pt`
- fixed-speed eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-10-09_z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940/eval_fixed_speed_23525_3p0_4p5.txt`
- Stage2D same-condition comparison eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-10-09_z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940/eval_fixed_speed_stage2d23500_same_conditions.txt`
- gait quality eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-10-09_z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940/eval_gait_quality_23525_3p5_4p25.txt`

Final online indicators at `model_23525.pt`:

- mean reward: `-0.22`
- mean episode length: `602.29`
- timeout ratio: `0.9260`
- head/shoulder contact ratio: `0.0740`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0`
- forward speed progress reward contribution: `0.1673`
- track linear velocity reward contribution: `0.5576`

Fixed-speed eval, same conditions (`num_envs=64`, `duration=6`, `warmup=2`):

| checkpoint | target | mean vx | abs err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2D 23500 | 3.00 | 2.9366 | 0.1840 | 2 | 1 | 1 |
| Stage2E 23525 | 3.00 | 3.0239 | 0.1491 | 1 | 1 | 0 |
| Stage2D 23500 | 3.50 | 3.1153 | 0.4768 | 4 | 1 | 3 |
| Stage2E 23525 | 3.50 | 3.3725 | 0.2369 | 0 | 0 | 0 |
| Stage2D 23500 | 4.00 | 2.6430 | 1.3853 | 16 | 1 | 15 |
| Stage2E 23525 | 4.00 | 3.1852 | 0.8536 | 11 | 6 | 5 |
| Stage2D 23500 | 4.25 | 1.9078 | 2.3492 | 37 | 10 | 27 |
| Stage2E 23525 | 4.25 | 2.7182 | 1.5385 | 23 | 7 | 16 |
| Stage2D 23500 | 4.50 | 1.1788 | 3.3219 | 51 | 6 | 45 |
| Stage2E 23525 | 4.50 | 2.1016 | 2.3999 | 31 | 8 | 23 |

Gait quality eval, Stage2E `model_23525.pt`:

| target | mean vx | abs err | resets | tilt xy | p90 swing foot z | single stance | double stance | flight | contact transitions/env/s | arm abs offset | shoulder pitch abs offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.50 | 3.2107 | 0.3782 | 3 | 0.0500 | 0.2662 | 0.8645 | 0.0161 | 0.1194 | 7.6510 | 0.2660 | 0.2206 |
| 4.00 | 3.1472 | 0.8722 | 6 | 0.0601 | 0.2669 | 0.8461 | 0.0232 | 0.1306 | 7.5990 | 0.2917 | 0.2588 |
| 4.25 | 2.9352 | 1.3221 | 7 | 0.0682 | 0.2657 | 0.8329 | 0.0282 | 0.1389 | 7.5000 | 0.2955 | 0.2702 |

Conclusion:

- Stage2E `model_23525.pt` is a useful checkpoint.
- The forward-progress reward improved actual high-speed tracking under the same evaluator:
  - `4.00 m/s`: mean vx improved from `2.6430` to `3.1852`.
  - `4.25 m/s`: mean vx improved from `1.9078` to `2.7182`.
  - `4.50 m/s`: mean vx improved from `1.1788` to `2.1016`.
- It also reduced speed-tracking resets:
  - `4.00 m/s`: `15 -> 5`
  - `4.25 m/s`: `27 -> 16`
  - `4.50 m/s`: `45 -> 23`
- Tradeoff:
  - head/shoulder resets increased at `4.0 m/s` (`1 -> 6`), which means the policy is now more willing to accelerate but occasionally pitches into unsafe posture.
  - `4.25+ m/s` is still not stable enough for deployment.
- Next direction:
  - preserve Stage2E `model_23525.pt` as the current high-speed candidate;
  - do not keep pushing speed range wider yet;
  - try a short Stage2F that keeps the Stage2E progress reward but adds posture/landing control at high commands, or slightly lowers progress weight while adding a high-speed torso pitch/height guard;
  - continue using `speed_tracking_duration_s=2.5`.

## Stage2F Posture-Guard Sprint AMP

Date: `2026-06-14`

Reason:

- Stage2E improved high-speed tracking, but introduced more head/shoulder contact at `4.0 m/s`.
- Stage2F keeps the Stage2E high-speed command/reference setup and tests a small posture guard:
  - reduce forward-progress reward slightly so it does not over-encourage diving forward;
  - increase torso/flat orientation and pitch/roll angular velocity penalties modestly;
  - increase the extra head/shoulder contact termination penalty.
- This is a short verification run only. The speed range is not widened.

Code/config changes:

- New task: `magicbot_z1_flat_sprint_amp_stage2f_posture`
- inherits Stage2E command/reference settings:
  - `lin_vel_x=(3.0, 4.25)`
  - `lin_vel_y=(-0.1, 0.1)`
  - `ang_vel_z=(-0.25, 0.25)`
  - `reference_motion.min_command_speed=3.0`
  - `reference_motion.max_reference_speed=4.8`
  - `reference_motion.speed_match_tolerance=0.5`
- changed from Stage2E:
  - `forward_speed_progress.weight`: `0.35 -> 0.30`
  - `ang_vel_xy_l2.weight`: `-0.05 -> -0.08`
  - `body_orientation_l2.weight`: `-2.0 -> -2.5`
  - `flat_orientation_l2.weight`: `-1.0 -> -1.2`
  - `head_shoulder_contact_termination_penalty.weight`: `-180.0 -> -240.0`
  - learning rate: `7.5e-5`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- AppLauncher registry/config check passed:
  - task registered: `magicbot_z1_flat_sprint_amp_stage2f_posture`
  - `lin_vel_x=(3.0, 4.25)`
  - `forward_speed_progress.weight=0.30`
  - `ang_vel_xy_l2.weight=-0.08`
  - `body_orientation_l2.weight=-2.5`
  - `flat_orientation_l2.weight=-1.2`
  - `head_shoulder_contact_termination_penalty.weight=-240.0`
  - `motion_prior.enable=True`
  - `motion_prior.reward_coef=0.08`
  - `learning_rate=0.000075`
- Smoke training passed:
  - envs: `64`
  - max iterations: `1`
  - start checkpoint: Stage2E `model_23525.pt`

Formal run:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2f_posture \
  --num_envs 4096 \
  --headless \
  --resume True \
  --load_run 2026-06-14_20-10-09_z1_sprint_amp_stage2e_progress_fromstage2d23500_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4_save25_env4096_20260614_200940 \
  --checkpoint model_23525.pt \
  --max_iterations 26 \
  --run_name z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431
```

Run artifacts:

- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23525.pt`
  - `model_23550.pt`
- fixed-speed eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/eval_fixed_speed_23550_3p0_4p5.txt`
- gait quality eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/eval_gait_quality_23550_3p5_4p25.txt`

Final online indicators at `model_23550.pt`:

- mean reward: `0.80`
- mean episode length: `596.49`
- timeout ratio: `0.9416`
- head/shoulder contact ratio: `0.0584`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0`
- forward speed progress reward contribution: `0.1448`
- track linear velocity reward contribution: `0.5852`

Fixed-speed comparison, same conditions (`num_envs=64`, `duration=6`, `warmup=2`):

| checkpoint | target | mean vx | abs err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2E 23525 | 3.00 | 3.0239 | 0.1491 | 1 | 1 | 0 |
| Stage2F 23550 | 3.00 | 3.0559 | 0.1349 | 0 | 0 | 0 |
| Stage2E 23525 | 3.50 | 3.3725 | 0.2369 | 0 | 0 | 0 |
| Stage2F 23550 | 3.50 | 3.3688 | 0.2232 | 4 | 4 | 0 |
| Stage2E 23525 | 4.00 | 3.1852 | 0.8536 | 11 | 6 | 5 |
| Stage2F 23550 | 4.00 | 3.1606 | 0.8573 | 9 | 5 | 4 |
| Stage2E 23525 | 4.25 | 2.7182 | 1.5385 | 23 | 7 | 16 |
| Stage2F 23550 | 4.25 | 2.9303 | 1.3238 | 10 | 7 | 3 |
| Stage2E 23525 | 4.50 | 2.1016 | 2.3999 | 31 | 8 | 23 |
| Stage2F 23550 | 4.50 | 2.4904 | 2.0101 | 16 | 1 | 15 |

Gait quality eval, Stage2F `model_23550.pt`:

| target | mean vx | abs err | resets | tilt xy | p90 swing foot z | single stance | double stance | flight | contact transitions/env/s | arm abs offset | shoulder pitch abs offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.50 | 3.2905 | 0.2830 | 2 | 0.0421 | 0.2621 | 0.8722 | 0.0060 | 0.1218 | 7.7240 | 0.2630 | 0.2158 |
| 4.00 | 3.3989 | 0.6129 | 1 | 0.0514 | 0.2655 | 0.8464 | 0.0097 | 0.1440 | 7.9010 | 0.2931 | 0.2527 |
| 4.25 | 3.3165 | 0.9359 | 3 | 0.0568 | 0.2663 | 0.8422 | 0.0092 | 0.1486 | 7.8177 | 0.3018 | 0.2687 |

Conclusion:

- Stage2F `model_23550.pt` is now the best high-speed candidate in fixed-speed eval.
- Compared with Stage2E, it improves the important high-speed points:
  - `4.25 m/s`: mean vx `2.7182 -> 2.9303`, resets `23 -> 10`, speed-tracking resets `16 -> 3`.
  - `4.50 m/s`: mean vx `2.1016 -> 2.4904`, resets `31 -> 16`, head/shoulder resets `8 -> 1`.
- It keeps `4.0 m/s` roughly flat:
  - mean vx `3.1852 -> 3.1606`
  - resets `11 -> 9`
- Tradeoff:
  - `3.5 m/s` has new head/shoulder resets in fixed-speed eval (`0 -> 4`), even though gait eval remains good.
  - Because of this, do not widen command range yet.
- Next direction:
  - preserve both Stage2E `model_23525.pt` and Stage2F `model_23550.pt`;
  - run Isaac play/video before deploying Stage2F;
  - for Stage2G, keep speed range at `3.0-4.25` and focus on reducing the new `3.5 m/s` head/shoulder contacts while retaining Stage2F's `4.25/4.5` gains;
  - likely try an even smaller posture adjustment or a command-conditioned guard instead of increasing global orientation penalties further.

Stage2F export/play validation:

- Export-only command succeeded with Stage2F `model_23550.pt`.
- Export artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/exported/policy.onnx`
- ONNX shape check:
  - input: `obs [1, 82]`
  - output: `actions [1, 24]`
- Export log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/play_export_only_23550.txt`
- Isaac visual play launched:
  - PID at launch: `3148080`
  - task: `magicbot_z1_flat_sprint_amp_stage2f_posture`
  - checkpoint: `model_23550.pt`
  - envs: `16`
  - command range: `lin_vel_x=(3.0, 4.25)`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
  - velocity debug visualization: enabled
  - stdout log:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2f_play_23550_cmdx3p0_4p25_env16_20260614_204622.out`
- Visual acceptance is still pending human observation:
  - specifically check whether the robot starts cleanly, avoids head/shoulder diving near `3.5 m/s`, and preserves the more aggressive `4.25 m/s` stride seen in fixed-speed/gait eval.

Stage2F command-profile rollout validation:

- Added tool:
  - `legged_lab/scripts/eval_command_profile.py`
  - Purpose: evaluate start/accelerate/hold/decelerate behavior under a staged forward-speed profile, while reporting per-segment velocity tracking, tilt and reset reasons.
  - Implementation note: velocity is read from critic observations and tilt is read from actor observations. Direct ArticulationData root/tilt reads were avoided because they can be brittle while other Isaac GUI play processes are running.
- Smoke profile passed:
  - log:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/eval_command_profile_smoke_23550_env4_v2.txt`
  - profile: `3.0 -> 4.25 -> 3.0`, `4 envs`, short `1s` segments
  - result: all segments completed with `0` resets.
- Main profile:
  - log:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/eval_command_profile_23550_3p0_4p25_3p0_env16_v2.txt`
  - profile: warmup `1s @ 3.0`, then `3s @ 3.0`, `4s @ 4.25`, `3s @ 3.0`
  - envs: `16`

| segment | target | duration | mean vx | first 1s vx | last 1s vx | abs err | p90 abs vx | mean tilt xy | p90 tilt xy | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.00 | 3.0 | 2.7114 | 2.2088 | 3.0451 | 0.3825 | 3.1678 | 0.0570 | 0.1291 | 0 | 0 | 0 |
| 1 | 4.25 | 4.0 | 3.7424 | 3.3346 | 3.9703 | 0.5085 | 4.1058 | 0.0439 | 0.0732 | 0 | 0 | 0 |
| 2 | 3.00 | 3.0 | 3.2323 | 3.5066 | 3.0762 | 0.2575 | 3.6836 | 0.0360 | 0.0590 | 0 | 0 | 0 |

Profile conclusion:

- Stage2F accelerates cleanly from `3.0` toward `4.25` without reset in this profile.
- It does not fully hit `4.25` in the 4-second segment, but the last-second mean reaches `3.9703 m/s`, which is better than the fixed-speed average and consistent with the gait eval improvement.
- Deceleration back to `3.0` is controlled: last-second mean is `3.0762 m/s`, with no speed-tracking or head/shoulder reset.
- This supports keeping Stage2F `model_23550.pt` as the current high-speed candidate.
- Stage2G should still wait for visual feedback from the running Isaac play, because fixed-speed eval showed `3.5 m/s` head/shoulder resets that may be visually obvious even when the profile rollout is clean.

Human visual play feedback:

- User observed the running Isaac play and reported that the current Stage2F policy "runs very well".
- This upgrades Stage2F `model_23550.pt` from a metrics-only candidate to the current visually accepted high-speed candidate.
- Keep this checkpoint protected for the next stage:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/model_23550.pt`
- Next training should build from Stage2F, still without widening beyond `3.0-4.25` until the `3.5 m/s` fixed-speed head/shoulder reset risk is reduced or confirmed harmless in repeated visual/profile tests.

Stage2G hold experiment:

- Task:
  `magicbot_z1_flat_sprint_amp_stage2g_hold`
- Purpose:
  continue from the visually accepted Stage2F policy with a smaller learning rate, holding the same command/reference/reward shape to see whether the high-speed tracking gain consolidates without changing behavior.
- Inherits Stage2F environment unchanged:
  - `lin_vel_x=(3.0, 4.25)`
  - `forward_speed_progress.weight=0.30`
  - `ang_vel_xy_l2.weight=-0.08`
  - `body_orientation_l2.weight=-2.5`
  - `flat_orientation_l2.weight=-1.2`
  - `head_shoulder_contact_termination_penalty.weight=-240.0`
  - `motion_prior.enable=True`
  - `motion_prior.reward_coef=0.08`
  - `motion_prior.reward_min_command_speed=3.0`
  - `learning_rate=0.00005`
  - `save_interval=25`
- Config validation passed:
  - task registered
  - actor obs: `82`
  - critic obs: `87`
  - command, reference-motion, posture penalty and AMP settings matched the intended Stage2F hold setup
- Smoke training passed:
  - envs: `64`
  - max iterations: `1`
  - start checkpoint: Stage2F `model_23550.pt`

Formal run:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2g_hold \
  --num_envs 4096 \
  --headless \
  --resume True \
  --load_run 2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431 \
  --checkpoint model_23550.pt \
  --max_iterations 26 \
  --run_name z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426
```

Run artifacts:

- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-24-51_z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23550.pt`
  - `model_23575.pt`
- fixed-speed eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-24-51_z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426/eval_fixed_speed_23575_env128_3p5_4p25.txt`
- command-profile eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-24-51_z1_sprint_amp_stage2g_hold_fromstage2f23550_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5_save25_env4096_20260614_212426/eval_command_profile_23575_3p0_4p25_3p0_env16.txt`

Final online indicators at `model_23575.pt`:

- mean reward: `1.43`
- mean episode length: `593.24`
- timeout ratio: `0.9395`
- head/shoulder contact ratio: `0.0605`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0`
- forward speed progress reward contribution: `0.1485`
- track linear velocity reward contribution: `0.6150`

Fixed-speed comparison, strict repeat conditions (`num_envs=128`, `duration=8`, `warmup=3`):

| checkpoint | target | mean vx | abs err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 3.50 | 3.4686 | 0.1545 | 3 | 2 | 1 |
| Stage2G 23575 | 3.50 | 3.4689 | 0.1596 | 5 | 4 | 1 |
| Stage2F 23550 | 4.00 | 3.6600 | 0.3730 | 10 | 4 | 6 |
| Stage2G 23575 | 4.00 | 3.6434 | 0.3845 | 11 | 4 | 7 |
| Stage2F 23550 | 4.25 | 3.4479 | 0.8131 | 29 | 8 | 21 |
| Stage2G 23575 | 4.25 | 3.4979 | 0.7596 | 26 | 6 | 20 |

Command-profile comparison (`3.0 -> 4.25 -> 3.0`, `16 envs`, warmup `1s @ 3.0`):

| checkpoint | segment target | mean vx | first 1s vx | last 1s vx | abs err | p90 abs vx | mean tilt xy | p90 tilt xy | resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 3.00 | 2.7114 | 2.2088 | 3.0451 | 0.3825 | 3.1678 | 0.0570 | 0.1291 | 0 |
| Stage2G 23575 | 3.00 | 2.8265 | 2.3780 | 3.1007 | 0.3188 | 3.2167 | 0.0586 | 0.1397 | 0 |
| Stage2F 23550 | 4.25 | 3.7424 | 3.3346 | 3.9703 | 0.5085 | 4.1058 | 0.0439 | 0.0732 | 0 |
| Stage2G 23575 | 4.25 | 3.7217 | 3.3533 | 3.9076 | 0.5293 | 4.0805 | 0.0486 | 0.0797 | 0 |
| Stage2F 23550 | 3.00 | 3.2323 | 3.5066 | 3.0762 | 0.2575 | 3.6836 | 0.0360 | 0.0590 | 0 |
| Stage2G 23575 | 3.00 | 3.2575 | 3.5103 | 3.1134 | 0.2735 | 3.6497 | 0.0352 | 0.0567 | 0 |

Stage2G conclusion:

- Stage2G is stable in the command-profile rollout and still has `0` resets across `3.0 -> 4.25 -> 3.0`.
- It slightly improves the strict `4.25 m/s` fixed-speed point:
  - mean vx `3.4479 -> 3.4979`
  - resets `29 -> 26`
  - head/shoulder resets `8 -> 6`
  - speed-tracking resets `21 -> 20`
- It is slightly worse at `3.5` and `4.0`, and its command-profile `4.25` last-second mean is lower than Stage2F:
  - Stage2F `3.9703 m/s`
  - Stage2G `3.9076 m/s`
- Therefore Stage2G `model_23575.pt` is a secondary high-speed candidate, not a replacement for Stage2F yet.
- Current primary candidate remains the visually accepted Stage2F checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/model_23550.pt`
- Do not continue widening speed range from Stage2G until a visual play check confirms it preserves the good Stage2F gait.

Stage2H speed-extend design:

- Task:
  `magicbot_z1_flat_sprint_amp_stage2h_speedextend`
- Purpose:
  take the visually accepted Stage2F policy and make only a small speed-range extension toward Stage 2's `4-5 m/s` goal.
- Training start:
  Stage2F `model_23550.pt`
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/model_23550.pt`
- Protected primary candidate:
  keep Stage2F `model_23550.pt` as the current human-accepted baseline until Stage2H has both metrics and visual confirmation.
- Changes versus Stage2F:
  - `lin_vel_x=(3.0, 4.5)` from `(3.0, 4.25)`
  - `reference_motion.max_reference_speed=5.0` from `4.8`
  - `reference_motion.speed_match_tolerance=0.6` from `0.5`
  - `track_lin_vel_xy_exp.std=0.8` from `0.75`
  - `forward_speed_progress.weight=0.32` from `0.30`
  - `learning_rate=0.00005`
- Unchanged safety/behavior constraints:
  - `speed_tracking_duration_s=2.5`
  - `head_shoulder_contact_termination_penalty.weight=-240.0`
  - `body_orientation_l2.weight=-2.5`
  - `flat_orientation_l2.weight=-1.2`
  - `motion_prior.reward_coef=0.08`
  - `motion_prior.reward_min_command_speed=3.0`
- Rationale:
  - Stage2F already looks good in play, so this stage deliberately avoids a large jump to `5+ m/s`.
  - The velocity reward std/tolerance is slightly relaxed so the policy can learn into `4.5 m/s` without being immediately dominated by high-speed tracking failure.
  - The run should be short and checkpointed before deciding whether to continue.
- Config validation:
  - `py_compile` passed for `z1_config.py` and task registry.
  - AppLauncher registry check passed:
    - registered: `True`
    - `lin_vel_x=(3.0, 4.5)`
    - `lin_vel_y=(-0.1, 0.1)`
    - `ang_vel_z=(-0.25, 0.25)`
    - `heading_command=False`
    - `speed_tracking_duration_s=2.5`
    - reference min/max/tolerance: `3.0 / 5.0 / 0.6`
    - track velocity reward: weight `1.35`, std `0.8`
    - progress reward: `0.32`
    - head/shoulder penalty: `-240.0`
    - AMP enabled: `True`, coef `0.08`, min speed `3.0`
    - learning rate: `5e-05`
    - save interval: `25`
- Smoke startup:
  - envs: `32`
  - max iterations: `1`
  - start checkpoint: Stage2F `model_23550.pt`
  - stdout log:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2h_speedextend_smoke_fromstage2f23550_env32_20260614_213931.out`
  - smoke run directory:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-39-43_z1_sprint_amp_stage2h_speedextend_smoke_fromstage2f23550_env32_20260614_213931`
  - startup produced params, tensorboard event file and loaded checkpoint copy; treat this as config/startup smoke only, not a meaningful training result.

Stage2H formal run:

- Initial 4096-env attempt:
  - run name:
    `z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env4096_20260614_214154`
  - run directory:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-42-18_z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env4096_20260614_214154`
  - stdout:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env4096_20260614_214154.out`
  - result:
    failed with CUDA OOM during the first PPO update.
  - cause:
    another rough dog training process was using about `24.9 GB`; Z1 had about `5.0 GB` allocated and only about `54 MB` free at the OOM point.
  - action:
    do not use this failed run as a policy result.
- Successful constrained-resource run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2h_speedextend \
  --num_envs 2048 \
  --headless \
  --resume True \
  --load_run 2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431 \
  --checkpoint model_23550.pt \
  --max_iterations 26 \
  --run_name z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310
```

Run artifacts:

- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-43-28_z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23550.pt`
  - `model_23575.pt`
- fixed-speed eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-43-28_z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310/eval_fixed_speed_23575_env128_3p5_4p5.txt`
- command-profile eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-43-28_z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310/eval_command_profile_23575_3p0_4p5_3p0_env16.txt`

Final online indicators at `model_23575.pt`:

- mean reward: `1.43`
- mean episode length: `577.42`
- timeout ratio: `0.9396`
- head/shoulder contact ratio: `0.0604`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0`
- track linear velocity reward contribution: `0.6171`
- forward speed progress reward contribution: `0.1558`

Fixed-speed eval, strict conditions (`num_envs=128`, `duration=8`, `warmup=3`):

| checkpoint | target | mean vx | abs err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 3.50 | 3.4686 | 0.1545 | 3 | 2 | 1 |
| Stage2H 23575 | 3.50 | 3.5110 | 0.1454 | 5 | 4 | 1 |
| Stage2F 23550 | 4.00 | 3.6600 | 0.3730 | 10 | 4 | 6 |
| Stage2H 23575 | 4.00 | 3.6093 | 0.4329 | 14 | 4 | 10 |
| Stage2F 23550 | 4.25 | 3.4479 | 0.8131 | 29 | 8 | 21 |
| Stage2H 23575 | 4.25 | 3.5673 | 0.7008 | 25 | 8 | 17 |
| Stage2H 23575 | 4.50 | 2.8996 | 1.6029 | 78 | 18 | 60 |

Command-profile eval (`3.0 -> 4.5 -> 3.0`, `16 envs`, warmup `1s @ 3.0`):

| segment | target | duration | mean vx | first 1s vx | last 1s vx | abs err | p90 abs vx | mean tilt xy | p90 tilt xy | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.00 | 3.0 | 2.7715 | 2.2846 | 3.0976 | 0.3780 | 3.2249 | 0.0564 | 0.1414 | 0 | 0 | 0 |
| 1 | 4.50 | 4.0 | 3.8389 | 3.3752 | 4.1076 | 0.6612 | 4.2552 | 0.0477 | 0.0786 | 0 | 0 | 0 |
| 2 | 3.00 | 3.0 | 3.3162 | 3.6297 | 3.1301 | 0.3264 | 3.8030 | 0.0360 | 0.0606 | 0 | 0 | 0 |

Stage2H conclusion:

- Stage2H is not a new primary candidate yet.
- Positive:
  - It improves strict `4.25 m/s` tracking versus Stage2F:
    - mean vx `3.4479 -> 3.5673`
    - abs err `0.8131 -> 0.7008`
    - resets `29 -> 25`
    - speed-tracking resets `21 -> 17`
  - It completes the `3.0 -> 4.5 -> 3.0` command profile with `0` resets and reaches `4.1076 m/s` in the last second of the `4.5` segment.
- Negative:
  - Fixed `4.5 m/s` is not stable: `78/128` resets, mostly speed tracking.
  - `4.0 m/s` strict eval regresses versus Stage2F.
  - `3.5 m/s` speed is slightly better but reset/head contact count is worse.
- Current ranking:
  - primary human-accepted candidate: Stage2F `model_23550.pt`
  - secondary high-speed experiment: Stage2H `model_23575.pt`
- Next step:
  - do not deploy Stage2H before visual play.
  - if continuing training from Stage2H, do not widen speed further; instead stabilize `4.0-4.5` and reduce speed-tracking resets.
  - if choosing a deployment/play candidate right now, prefer Stage2F because it is visually accepted.

Stage2H extra validation against Stage2F:

- Purpose:
  check whether Stage2H's `3.0 -> 4.5 -> 3.0` profile result is truly better than the visually accepted Stage2F baseline under the same command profile, and compare fixed `4.5 m/s` directly.
- Stage2F command-profile eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/eval_command_profile_23550_3p0_4p5_3p0_env16.txt`
- Stage2F fixed `4.5 m/s` eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/eval_fixed_speed_23550_env128_4p5.txt`

Command-profile comparison (`3.0 -> 4.5 -> 3.0`, `16 envs`, warmup `1s @ 3.0`):

| checkpoint | segment target | mean vx | first 1s vx | last 1s vx | abs err | p90 abs vx | mean tilt xy | p90 tilt xy | resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 3.00 | 2.7114 | 2.2088 | 3.0451 | 0.3825 | 3.1678 | 0.0570 | 0.1291 | 0 |
| Stage2H 23575 | 3.00 | 2.7715 | 2.2846 | 3.0976 | 0.3780 | 3.2249 | 0.0564 | 0.1414 | 0 |
| Stage2F 23550 | 4.50 | 3.7443 | 3.3098 | 4.0103 | 0.7557 | 4.1564 | 0.0467 | 0.0773 | 0 |
| Stage2H 23575 | 4.50 | 3.8389 | 3.3752 | 4.1076 | 0.6612 | 4.2552 | 0.0477 | 0.0786 | 0 |
| Stage2F 23550 | 3.00 | 3.2504 | 3.5502 | 3.0847 | 0.2721 | 3.7472 | 0.0381 | 0.0627 | 0 |
| Stage2H 23575 | 3.00 | 3.3162 | 3.6297 | 3.1301 | 0.3264 | 3.8030 | 0.0360 | 0.0606 | 0 |

Fixed `4.5 m/s` direct comparison (`num_envs=128`, `duration=8`, `warmup=3`):

| checkpoint | target | mean vx | abs err | p50 vx | p90 abs vx | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 4.50 | 2.9308 | 1.5704 | 3.7212 | 4.2650 | 56 | 9 | 47 |
| Stage2H 23575 | 4.50 | 2.8996 | 1.6029 | 3.7953 | 4.3374 | 78 | 18 | 60 |

Updated Stage2H interpretation:

- Stage2H gives a real but small dynamic-profile speed improvement at the `4.5 m/s` segment:
  - mean vx `3.7443 -> 3.8389`
  - last-second mean vx `4.0103 -> 4.1076`
  - abs err `0.7557 -> 0.6612`
  - resets remain `0`
- Stage2H is worse in fixed `4.5 m/s`:
  - resets `56 -> 78`
  - speed-tracking resets `47 -> 60`
  - head/shoulder resets `9 -> 18`
- Decision:
  - keep Stage2F `model_23550.pt` as the primary candidate.
  - keep Stage2H `model_23575.pt` only as an experimental reference showing a small dynamic `4.5 m/s` profile gain.
  - do not continue widening speed; the next training should stabilize fixed `4.0-4.5 m/s` and reduce speed-tracking resets.

Stage2H play/export validation:

- A short foreground play smoke reached Isaac App ready, built the env, exported policy artifacts, and was stopped by a deliberate `timeout 25s`.
- Constant GUI play then started successfully:
  - PID: `3940677`
  - task: `magicbot_z1_flat_sprint_amp_stage2h_speedextend`
  - checkpoint: `model_23575.pt`
  - envs: `16`
  - command range: `lin_vel_x=(3.0, 4.5)`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
  - velocity debug visualization: enabled
  - stdout log:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2h_play_23575_cmdx3p0_4p5_env16_20260614_215833.out`
- Exported policy artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-43-28_z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_21-43-28_z1_sprint_amp_stage2h_speedextend_fromstage2f23550_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5_save25_env2048_20260614_214310/exported/policy.onnx`
- ONNX shape:
  - input: `obs [1, 82]`
  - output: `actions [1, 24]`
- Human visual feedback is still pending; do not deploy Stage2H until the running play is inspected.

Stage2I high-track stabilization design:

- Task:
  `magicbot_z1_flat_sprint_amp_stage2i_hightrack`
- Purpose:
  stabilize fixed `4.0-4.5 m/s` behavior and reduce speed-tracking resets, without widening beyond Stage2H.
- Start checkpoint for future formal training:
  Stage2F `model_23550.pt`
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431/model_23550.pt`
- Rationale for starting from Stage2F:
  - Stage2F is still the primary human-accepted candidate.
  - Stage2H only improves dynamic `4.5 m/s` slightly, but fixed `4.5 m/s` reset count is worse.
  - Stage2I therefore uses the Stage2F posture/safety base and focuses the command distribution on higher speeds.
- Changes versus Stage2F:
  - `lin_vel_x=(3.5, 4.5)`
  - `lin_vel_y=(-0.08, 0.08)`
  - `ang_vel_z=(-0.18, 0.18)`
  - `reference_motion.min_command_speed=3.5`
  - `reference_motion.max_reference_speed=5.1`
  - `reference_motion.speed_match_tolerance=0.65`
  - `track_lin_vel_xy_exp.weight=1.70`
  - `track_lin_vel_xy_exp.std=1.0`
  - `forward_speed_progress.weight=0.24`
  - `forward_speed_progress.min_command_x=3.5`
  - `energy.weight=-0.0006`
  - `action_rate_l2.weight=-0.0075`
  - `learning_rate=0.00004`
  - `motion_prior.reward_min_command_speed=3.5`
- Unchanged safety constraints:
  - `speed_tracking_duration_s=2.5`
  - `speed_tracking_abs_error_threshold=0.5`
  - `speed_tracking_rel_error_threshold=0.35`
  - `head_shoulder_contact_termination_penalty.weight=-240.0`
  - `body_orientation_l2.weight=-2.5`
  - `flat_orientation_l2.weight=-1.2`
  - `motion_prior.reward_coef=0.08`
- Reasoning:
  - At fixed `4.5 m/s`, the speed-tracking reset threshold is about `1.575 m/s`; policies averaging around `2.9 m/s` will reset after `2.5s`.
  - The reward changes increase the usable tracking gradient at high error, reduce the pure progress shortcut, and reduce energy/action penalties enough to let the policy spend more effort chasing speed.
  - The termination rule is intentionally not relaxed, because Stage2I should learn to satisfy the same deploy-relevant speed criterion.

Stage2I config validation:

- `py_compile` passed:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- AppLauncher registry check passed:
  - registered: `True`
  - run name:
    `z1_sprint_amp_stage2i_hightrack_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_energy6e-4_lr4e-5`
  - `lin_vel_x=(3.5, 4.5)`
  - `lin_vel_y=(-0.08, 0.08)`
  - `ang_vel_z=(-0.18, 0.18)`
  - speed tracking thresholds: duration `2.5`, abs `0.5`, rel `0.35`
  - reference min/max/tolerance: `3.5 / 5.1 / 0.65`
  - track velocity reward: weight `1.7`, std `1.0`
  - progress reward: weight `0.24`, min command x `3.5`
  - energy/action-rate weights: `-0.0006 / -0.0075`
  - AMP enabled: `True`, coef `0.08`, min speed `3.5`
  - learning rate: `4e-05`
  - save interval: `25`

Stage2I smoke startup:

- Final smoke command:

```bash
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2i_hightrack \
  --num_envs 32 \
  --headless \
  --resume True \
  --load_run 2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431 \
  --checkpoint model_23550.pt \
  --max_iterations 1 \
  --run_name z1_sprint_amp_stage2i_hightrack_smoke_fromstage2f23550_env32_20260614_220632 \
  --logger tensorboard \
  --skip_deploy_yaml
```

- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2i_hightrack_smoke_fromstage2f23550_env32_20260614_220632.out`
- smoke directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-06-45_z1_sprint_amp_stage2i_hightrack_smoke_fromstage2f23550_env32_20260614_220632`
- smoke artifacts:
  - `params/env.yaml`
  - `params/agent.yaml`
  - tensorboard event file
  - loaded start checkpoint copy `model_23550.pt`
- Smoke params confirmed:
  - `lin_vel_x=(3.5, 4.5)`
  - `lin_vel_y=(-0.08, 0.08)`
  - `ang_vel_z=(-0.18, 0.18)`
  - `track_lin_vel_xy_exp.weight=1.7`
  - `track_lin_vel_xy_exp.std=1.0`
  - `forward_speed_progress.weight=0.24`
  - `forward_speed_progress.min_command_x=3.5`
- Formal Stage2I training has not started yet because Stage2H visual play is still running for inspection.
- Recommended formal run when ready:
  - start from Stage2F `model_23550.pt`
  - `num_envs=2048` if other training/play processes are active, or `4096` if GPU is free
  - `max_iterations=26` to get only the next checkpoint before evaluation
  - evaluate fixed speeds `3.5 4.0 4.25 4.5` and profile `3.0 -> 4.5 -> 3.0`

Stage2I formal run and validation:

- Context:
  - Stage2H GUI play and a rough dog training run were active, so the formal Stage2I run used `1024 envs` to avoid OOM and avoid disrupting the play session too much.
  - Stage2F `model_23550.pt` remains protected and unchanged.
- Formal command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2i_hightrack \
  --num_envs 1024 \
  --headless \
  --resume True \
  --load_run 2026-06-14_20-34-59_z1_sprint_amp_stage2f_posture_fromstage2e23525_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5_save25_env4096_20260614_203431 \
  --checkpoint model_23550.pt \
  --max_iterations 26 \
  --run_name z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010
```

Run artifacts:

- stdout log:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010.out`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010`
- deploy snapshot:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/policies/loco_mode/config/LocoMode.yaml`
- checkpoints:
  - `model_23550.pt`
  - `model_23575.pt`
- fixed-speed eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/eval_fixed_speed_23575_env128_3p5_4p5.txt`
- command-profile eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/eval_command_profile_23575_3p0_4p5_3p0_env16.txt`

Final online indicators at `model_23575.pt`:

- mean reward: `6.16`
- mean episode length: `562.28`
- timeout ratio: `1.0000`
- head/shoulder contact ratio: `0.0000`
- body contact ratio: `0.0`
- speed tracking failure ratio: `0.0000`
- track linear velocity reward contribution: `0.8284`
- forward speed progress reward contribution: `0.1161`

Fixed-speed eval, strict conditions (`num_envs=128`, `duration=8`, `warmup=3`):

| checkpoint | target | mean vx | abs err | p50 vx | p90 abs vx | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 3.50 | 3.4686 | 0.1545 | - | - | 3 | 2 | 1 |
| Stage2H 23575 | 3.50 | 3.5110 | 0.1454 | 3.5280 | 3.7305 | 5 | 4 | 1 |
| Stage2I 23575 | 3.50 | 3.5640 | 0.2056 | 3.6099 | 3.8235 | 4 | 3 | 1 |
| Stage2F 23550 | 4.00 | 3.6600 | 0.3730 | - | - | 10 | 4 | 6 |
| Stage2H 23575 | 4.00 | 3.6093 | 0.4329 | 3.8296 | 4.0894 | 14 | 4 | 10 |
| Stage2I 23575 | 4.00 | 3.7580 | 0.3533 | 3.9364 | 4.2044 | 9 | 4 | 5 |
| Stage2F 23550 | 4.25 | 3.4479 | 0.8131 | - | - | 29 | 8 | 21 |
| Stage2H 23575 | 4.25 | 3.5673 | 0.7008 | 3.9452 | 4.2525 | 25 | 8 | 17 |
| Stage2I 23575 | 4.25 | 3.6549 | 0.6404 | 4.0186 | 4.3445 | 15 | 2 | 13 |
| Stage2F 23550 | 4.50 | 2.9308 | 1.5704 | 3.7212 | 4.2650 | 56 | 9 | 47 |
| Stage2H 23575 | 4.50 | 2.8996 | 1.6029 | 3.7953 | 4.3374 | 78 | 18 | 60 |
| Stage2I 23575 | 4.50 | 3.0259 | 1.4851 | 3.9272 | 4.4296 | 35 | 5 | 30 |

Command-profile eval (`3.0 -> 4.5 -> 3.0`, `16 envs`, warmup `1s @ 3.0`):

| checkpoint | segment target | mean vx | first 1s vx | last 1s vx | abs err | p90 abs vx | mean tilt xy | p90 tilt xy | resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2F 23550 | 4.50 | 3.7443 | 3.3098 | 4.0103 | 0.7557 | 4.1564 | 0.0467 | 0.0773 | 0 |
| Stage2H 23575 | 4.50 | 3.8389 | 3.3752 | 4.1076 | 0.6612 | 4.2552 | 0.0477 | 0.0786 | 0 |
| Stage2I 23575 | 4.50 | 3.8774 | 3.4299 | 4.1484 | 0.6231 | 4.2958 | 0.0525 | 0.0857 | 0 |

Stage2I profile details:

| segment | target | duration | mean vx | first 1s vx | last 1s vx | abs err | p90 abs vx | mean tilt xy | p90 tilt xy | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.00 | 3.0 | 2.7702 | 2.2517 | 3.1487 | 0.4311 | 3.2801 | 0.0586 | 0.1274 | 0 | 0 | 0 |
| 1 | 4.50 | 4.0 | 3.8774 | 3.4299 | 4.1484 | 0.6231 | 4.2958 | 0.0525 | 0.0857 | 0 | 0 | 0 |
| 2 | 3.00 | 3.0 | 3.3652 | 3.6694 | 3.1936 | 0.3688 | 3.8750 | 0.0370 | 0.0623 | 0 | 0 | 0 |

Stage2I interpretation:

- Stage2I is a useful high-speed stabilization checkpoint.
- It is the best metrics checkpoint so far for fixed `4.0-4.5 m/s`:
  - fixed `4.0`: resets `10/14 -> 9`, speed resets `6/10 -> 5`
  - fixed `4.25`: resets `29/25 -> 15`, speed resets `21/17 -> 13`
  - fixed `4.5`: resets `56/78 -> 35`, speed resets `47/60 -> 30`
- It also improves the dynamic `4.5` profile segment:
  - Stage2F last-second vx `4.0103`
  - Stage2H last-second vx `4.1076`
  - Stage2I last-second vx `4.1484`
- Remaining limitation:
  - fixed `4.5 m/s` is still not stable enough: `35/128` resets, mostly speed-tracking.
  - 3.5 m/s tracking overshoots slightly and has slightly worse abs error than Stage2F.
- Current ranking:
  - primary human-accepted candidate for visual/deploy safety: Stage2F `model_23550.pt`
  - best metric candidate for fixed high-speed stabilization: Stage2I `model_23575.pt`
  - Stage2H is superseded by Stage2I on both fixed `4.5` reset count and dynamic `4.5` segment speed.

Stage2I play/export validation:

- Stage2H play was stopped to avoid two GUI simulations consuming resources.
- Stage2I GUI play started successfully:
  - PID: `2913`
  - task: `magicbot_z1_flat_sprint_amp_stage2i_hightrack`
  - checkpoint: `model_23575.pt`
  - envs: `16`
  - command range: `lin_vel_x=(3.5, 4.5)`, `lin_vel_y=0.0`, `ang_vel_z=0.0`
  - velocity debug visualization: enabled
  - stdout log:
    `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2i_play_23575_cmdx3p5_4p5_env16_20260614_222245.out`
- Exported policy artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/exported/policy.onnx`
- ONNX shape:
  - input: `obs [1, 82]`
  - output: `actions [1, 24]`
- Human visual feedback on the Stage2I GUI play was positive: the user reported it is running very well.
- Promote Stage2I to the current visually accepted high-speed candidate, while keeping Stage2F `model_23550.pt` protected as the conservative fallback.

Stage2I gait-quality eval (`num_envs=64`, `duration=6`, `warmup=2`):

- artifact:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/eval_gait_quality_23575_3p5_4p5.txt`

| target | mean vx | abs err | p50 vx | p90 abs vx | resets | mean tilt xy | p90 swing foot z | single stance | double stance | flight | contact transitions/env/s | arm abs offset | shoulder pitch offset |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.50 | 3.4289 | 0.2851 | 3.5524 | 3.8016 | 2 | 0.0423 | 0.2618 | 0.8676 | 0.0100 | 0.1224 | 7.9271 | 0.2588 | 0.2218 |
| 4.00 | 3.2580 | 0.8054 | 3.7784 | 4.1279 | 7 | 0.0624 | 0.2644 | 0.8416 | 0.0253 | 0.1332 | 7.8984 | 0.2876 | 0.2691 |
| 4.25 | 3.3281 | 0.9527 | 3.8720 | 4.2805 | 4 | 0.0646 | 0.2653 | 0.8293 | 0.0243 | 0.1464 | 7.9714 | 0.3008 | 0.2823 |
| 4.50 | 2.6576 | 1.8456 | 3.2878 | 4.2917 | 15 | 0.0785 | 0.2581 | 0.8294 | 0.0441 | 0.1265 | 7.4609 | 0.2885 | 0.2858 |

Gait-quality interpretation:

- The user-visible Stage2I play is good, but short fixed-command evals still show high-speed saturation above `4.25 m/s`.
- The gait remains running-like at `3.5-4.25` with high single-stance ratio and non-zero flight ratio.
- At fixed `4.5`, mean speed drops and tilt rises; this remains the next high-speed target after command-direction robustness.

Stage2I turning baseline (`vy=0.20`, `wz=0.35`, `num_envs=32`, `duration=4`, `warmup=2`):

- artifact:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/eval_fixed_command_23575_vy0p20_wz0p35_env32.txt`

| target vx | target vy | target wz | mean vx | mean vy | mean wz | vx abs err | vy abs err | wz abs err | xy abs err | p90 xy err | resets | head/shoulder | speed tracking |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.50 | 0.20 | 0.35 | 3.3465 | 0.2599 | 0.2875 | 0.2690 | 0.1474 | 0.4614 | 0.3413 | 0.5925 | 0 | 0 | 0 |
| 4.00 | 0.20 | 0.35 | 3.0252 | 0.2103 | 0.2981 | 0.9888 | 0.1955 | 0.5189 | 1.0354 | 2.7442 | 3 | 3 | 0 |
| 4.25 | 0.20 | 0.35 | 2.4904 | 0.1883 | 0.3345 | 1.7627 | 0.2243 | 0.5297 | 1.7964 | 3.9550 | 10 | 3 | 7 |

Turning baseline interpretation:

- Stage2I was optimized for straight sprinting (`lin_vel_y=(-0.08, 0.08)`, `ang_vel_z=(-0.18, 0.18)`), so this failure mode is expected.
- It tracks `vy` direction roughly, but high-speed turning causes forward velocity collapse first; at `4.25 + vy0.20 + wz0.35`, most failures are speed tracking.
- Next stage should strengthen lateral/yaw command tracking without immediately increasing the straight-line x-speed target.

## Stage2J: turning and lateral-command robustness

Purpose:

- Build from the visually accepted Stage2I checkpoint.
- Keep the fast-running posture, but train non-zero `lin_vel_y` and `ang_vel_z` so the policy can change direction while preserving forward speed.

Config changes:

- task: `magicbot_z1_flat_sprint_amp_stage2j_turnrobust`
- base: `MagicBotZ1FlatSprintAMPStage2IHighTrackEnvCfg`
- command range:
  - `lin_vel_x=(3.25, 4.5)`
  - `lin_vel_y=(-0.25, 0.25)`
  - `ang_vel_z=(-0.45, 0.45)`
- reference motion:
  - `min_command_speed=3.25`
  - `max_reference_speed=5.1`
  - `speed_match_tolerance=0.75`
- reward tuning:
  - `track_lin_vel_xy_exp.weight=1.80`
  - `track_lin_vel_xy_exp.std=1.05`
  - `track_ang_vel_z_exp.weight=1.45`
  - `track_ang_vel_z_exp.std=0.55`
  - `forward_speed_progress.weight=0.20`
  - `forward_speed_progress.min_command_x=3.25`
- agent:
  - `learning_rate=4e-5`
  - `motion_prior.reward_coef=0.08`
  - `motion_prior.reward_min_command_speed=3.25`
  - `save_interval=25`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
  - `legged_lab/scripts/eval_fixed_speed.py`
- registry check passed:
  - task resolves as `magicbot_z1_flat_sprint_amp_stage2j_turnrobust`
  - `speed_tracking_duration_s` remains `2.5`
  - command and reward values match the intended Stage2J settings.
- `eval_fixed_speed.py` now supports fixed `--lin_vel_y` and `--ang_vel_z`, so future evals can measure turning and lateral tracking instead of only straight-line speed.

Formal Stage2J run:

- unit:
  `z1_stage2j_turnrobust_20260614_234947.service`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947`
- stdout:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947.out`
- deploy yaml snapshot root:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947`
- start checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010/model_23575.pt`
- command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2j_turnrobust \
  --num_envs 1024 \
  --headless \
  --resume True \
  --load_run 2026-06-14_22-10-27_z1_sprint_amp_stage2i_hightrack_fromstage2f23550_cmdx3p5_4p5_ref3p5_5p1_track1p7_std1p0_prog0p24_lr4e-5_save25_env1024_20260614_221010 \
  --checkpoint model_23575.pt \
  --max_iterations 51 \
  --run_name z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947
```

Early online indicators:

| iteration | mean reward | mean episode length | track xy | track yaw | timeout ratio | head/shoulder ratio | speed failure ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23588 | 4.83 | 280.03 | 0.4965 | 0.1798 | 1.0000 | 0.0000 | 0.0000 |
| 23591 | 6.89 | 350.68 | 0.5627 | 0.2103 | 0.9792 | 0.0208 | 0.0000 |
| 23597 | 8.81 | 497.77 | 0.8207 | 0.2983 | 0.9722 | 0.0278 | 0.0000 |

Early interpretation:

- Stage2J starts cleanly from Stage2I and is not showing the previous collapse pattern.
- `track_lin_vel_xy_exp` and `track_ang_vel_z_exp` both increase during the first ten iterations, which is the intended effect.
- Speed-tracking failure remains `0.0` so far despite the wider `vy/yaw` command ranges.

Stage2J completion:

- service status: inactive after completing the scheduled run.
- final checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/model_23625.pt`
- saved checkpoints:
  - `model_23575.pt`
  - `model_23600.pt`
  - `model_23625.pt`

Final online indicators:

- mean reward: `13.76`
- mean episode length: `930.53`
- `Episode_Reward/track_lin_vel_xy_exp`: `1.4531`
- `Episode_Reward/track_ang_vel_z_exp`: `0.5251`
- timeout ratio: `0.9375`
- head/shoulder contact ratio: `0.0625`
- speed tracking failure ratio: `0.0000`

Stage2J eval artifacts:

- `model_23625.pt` turning eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/eval_fixed_command_23625_vy0p20_wz0p35_env32.txt`
- `model_23625.pt` straight eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/eval_fixed_speed_23625_env64_3p5_4p25.txt`
- `model_23600.pt` turning eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/eval_fixed_command_23600_vy0p20_wz0p35_env32.txt`
- `model_23600.pt` straight eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/eval_fixed_speed_23600_env64_3p5_4p25.txt`

Turning eval comparison (`vy=0.20`, `wz=0.35`, `num_envs=32`, `duration=4`, `warmup=2`):

| checkpoint | target vx | mean vx | mean vy | mean wz | vx abs err | vy abs err | wz abs err | xy abs err | p90 xy err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2I 23575 | 3.50 | 3.3465 | 0.2599 | 0.2875 | 0.2690 | 0.1474 | 0.4614 | 0.3413 | 0.5925 | 0 | 0 | 0 |
| Stage2J 23600 | 3.50 | 3.4185 | 0.2240 | 0.2807 | 0.2076 | 0.1195 | 0.4305 | 0.2644 | 0.4433 | 0 | 0 | 0 |
| Stage2J 23625 | 3.50 | 3.3805 | 0.1927 | 0.2585 | 0.2086 | 0.1234 | 0.3965 | 0.2646 | 0.4487 | 0 | 0 | 0 |
| Stage2I 23575 | 4.00 | 3.0252 | 0.2103 | 0.2981 | 0.9888 | 0.1955 | 0.5189 | 1.0354 | 2.7442 | 3 | 3 | 0 |
| Stage2J 23600 | 4.00 | 3.3441 | 0.2223 | 0.2764 | 0.6721 | 0.1527 | 0.4696 | 0.7153 | 1.9400 | 1 | 1 | 0 |
| Stage2J 23625 | 4.00 | 3.3288 | 0.1923 | 0.2573 | 0.6782 | 0.1610 | 0.4447 | 0.7176 | 1.7663 | 2 | 2 | 0 |
| Stage2I 23575 | 4.25 | 2.4904 | 0.1883 | 0.3345 | 1.7627 | 0.2243 | 0.5297 | 1.7964 | 3.9550 | 10 | 3 | 7 |
| Stage2J 23600 | 4.25 | 2.8297 | 0.1908 | 0.2985 | 1.4222 | 0.1973 | 0.5360 | 1.4511 | 3.5907 | 3 | 1 | 2 |
| Stage2J 23625 | 4.25 | 3.0793 | 0.1375 | 0.2419 | 1.1710 | 0.1933 | 0.4856 | 1.2017 | 2.8706 | 3 | 1 | 2 |

Straight eval comparison within Stage2J (`vy=0.0`, `wz=0.0`, `num_envs=64`, `duration=4`, `warmup=2`):

| checkpoint | target vx | mean vx | vx abs err | xy abs err | p90 xy err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2J 23600 | 3.50 | 3.3638 | 0.3211 | 0.3694 | 0.5636 | 0 | 0 | 0 |
| Stage2J 23625 | 3.50 | 3.3586 | 0.2982 | 0.3504 | 0.5045 | 2 | 2 | 0 |
| Stage2J 23600 | 4.00 | 3.2467 | 0.7961 | 0.8362 | 2.8202 | 7 | 3 | 4 |
| Stage2J 23625 | 4.00 | 3.4656 | 0.5647 | 0.6051 | 1.4863 | 1 | 1 | 0 |
| Stage2J 23600 | 4.25 | 3.0896 | 1.1703 | 1.1972 | 3.2714 | 1 | 1 | 0 |
| Stage2J 23625 | 4.25 | 3.1747 | 1.0831 | 1.1130 | 3.2038 | 2 | 1 | 1 |

Stage2J checkpoint choice:

- `model_23625.pt` is better than `model_23600.pt` for the intended turning/lateral objective.
- `model_23625.pt` also has better short straight-line `4.0/4.25` metrics than `model_23600.pt`.
- Stage2J improves the Stage2I turning failure mode substantially, especially at `4.25 + vy0.20 + wz0.35`:
  - resets `10 -> 3`
  - speed-tracking resets `7 -> 2`
  - `xy_abs_err 1.7964 -> 1.2017`
- Remaining limitation:
  - Stage2J still does not solve sustained `4.25-4.5 m/s` straight sprinting.
  - Next stage should mix straight high-speed retention with non-zero `vy/yaw`, rather than only widening the turning commands further.

Stage2J play/export validation:

- stopped the previous Stage2I GUI play (`PID 2913`) and started Stage2J GUI play.
- play unit:
  `z1_stage2j_play_20260615_000341.service`
- play stdout:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2j_play_23625_cmdx3p25_4p5_cmdy0p25_yaw0p45_env16_20260615_000341.out`
- checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/model_23625.pt`
- play command range:
  - `lin_vel_x=(3.25, 4.5)`
  - `lin_vel_y=(-0.25, 0.25)`
  - `ang_vel_z=(-0.45, 0.45)`
- exported artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/exported/policy.onnx`

## Stage2K: mixed straight-speed retention and turning robustness

Purpose:

- Build from Stage2J `model_23625.pt`.
- Preserve the turning/lateral robustness that Stage2J gained.
- Recover more straight-line high-speed retention, because Stage2J still under-tracks sustained `4.25-4.5 m/s`.
- Do not jump toward `8 m/s`; this remains a Stage 2 consolidation step around `4-5 m/s`.

Config changes:

- task: `magicbot_z1_flat_sprint_amp_stage2k_mixedretention`
- base: `MagicBotZ1FlatSprintAMPStage2JTurnRobustEnvCfg`
- command range:
  - `lin_vel_x=(3.5, 4.65)`
  - `lin_vel_y=(-0.22, 0.22)`
  - `ang_vel_z=(-0.40, 0.40)`
- reference motion:
  - `min_command_speed=3.5`
  - `max_reference_speed=5.3`
  - `speed_match_tolerance=0.70`
- reward tuning:
  - `track_lin_vel_xy_exp.weight=1.95`
  - `track_lin_vel_xy_exp.std=0.95`
  - `track_ang_vel_z_exp.weight=1.30`
  - `track_ang_vel_z_exp.std=0.55`
  - `forward_speed_progress.weight=0.30`
  - `forward_speed_progress.min_command_x=3.5`
- retained safety/regularization:
  - `speed_tracking_duration_s=2.5`
  - `energy.weight=-6e-4`
  - `action_rate_l2.weight=-7.5e-3`
  - head/shoulder contact termination and penalty unchanged from Stage2F+
- agent:
  - `learning_rate=3e-5`
  - `motion_prior.reward_coef=0.08`
  - `motion_prior.reward_min_command_speed=3.5`
  - `save_interval=25`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- registry check passed:
  - task resolves as `magicbot_z1_flat_sprint_amp_stage2k_mixedretention`
  - command ranges, reward weights, AMP gate, and `speed_tracking_duration_s=2.5` match the intended Stage2K settings.

Formal Stage2K run:

- unit:
  `z1_stage2k_mixedretention_20260615_000941.service`
- run directory:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941`
- stdout:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941.out`
- deploy yaml snapshot root:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941`
- start checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947/model_23625.pt`
- command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/hiyio/LeggedLab \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python legged_lab/scripts/train.py \
  --task magicbot_z1_flat_sprint_amp_stage2k_mixedretention \
  --num_envs 1024 \
  --headless \
  --resume True \
  --load_run 2026-06-14_23-50-05_z1_sprint_amp_stage2j_turnrobust_fromstage2i23575_cmdx3p25_4p5_cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_env1024_20260614_234947 \
  --checkpoint model_23625.pt \
  --max_iterations 51 \
  --run_name z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941 \
  --logger tensorboard \
  --deploy_yaml_root /home/hiyio/LeggedLab/logs/magicbot_z1_flat/deploy_snapshots/z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941
```

Early online indicators:

| iteration | mean reward | mean episode length | track xy | progress | track yaw | timeout ratio | head/shoulder ratio | speed failure ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23626 | -3.16 | 24.93 | 0.0561 | 0.0000 | 0.0168 | 0.8042 | 0.1958 | 0.0000 |
| 23629 | -0.42 | 73.55 | 0.1546 | 0.0048 | 0.0534 | 0.9583 | 0.0417 | 0.0000 |
| 23632 | -3.13 | 133.07 | 0.2340 | 0.0228 | 0.0912 | 0.9722 | 0.0278 | 0.0000 |
| 23635 | 3.62 | 220.54 | 0.3526 | 0.0439 | 0.1275 | 0.9931 | 0.0069 | 0.0000 |

Early interpretation:

- Stage2K caused a large initial distribution shock because it raised the x-speed floor and tightened x/y tracking compared with Stage2J.
- The run is recovering by `23635`, but the first checkpoint must be evaluated before continuing blindly.

Stage2K first checkpoint:

- training was stopped at the first new checkpoint to avoid blindly continuing after the initial distribution shock.
- checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941/model_23650.pt`
- final observed online line before stopping:
  - iteration: `23650/23676`
  - mean reward: `9.79`
  - mean episode length: `563.00`
  - `track_lin_vel_xy_exp`: `0.9747`
  - `forward_speed_progress`: `0.1483`
  - `track_ang_vel_z_exp`: `0.3146`
  - timeout ratio: `1.0000`
  - head/shoulder contact ratio: `0.0000`
  - speed tracking failure ratio: `0.0000`
- saved checkpoints:
  - `model_23625.pt`
  - `model_23650.pt`

Stage2K eval artifacts:

- straight eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941/eval_fixed_speed_23650_env64_3p5_4p5.txt`
- turning eval:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941/eval_fixed_command_23650_vy0p20_wz0p35_env32.txt`

Straight eval comparison (`vy=0.0`, `wz=0.0`, `duration=4`, `warmup=2`):

| checkpoint | envs | target vx | mean vx | vx abs err | xy abs err | p90 xy err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2J 23625 | 64 | 3.50 | 3.3586 | 0.2982 | 0.3504 | 0.5045 | 2 | 2 | 0 |
| Stage2K 23650 | 64 | 3.50 | 3.5350 | 0.2608 | 0.3132 | 0.4425 | 0 | 0 | 0 |
| Stage2J 23625 | 64 | 4.00 | 3.4656 | 0.5647 | 0.6051 | 1.4863 | 1 | 1 | 0 |
| Stage2K 23650 | 64 | 4.00 | 3.3896 | 0.6785 | 0.7267 | 2.1301 | 5 | 4 | 1 |
| Stage2J 23625 | 64 | 4.25 | 3.1747 | 1.0831 | 1.1130 | 3.2038 | 2 | 1 | 1 |
| Stage2K 23650 | 64 | 4.25 | 3.3884 | 0.8895 | 0.9309 | 2.7273 | 3 | 2 | 1 |
| Stage2K 23650 | 64 | 4.50 | 2.9410 | 1.5630 | 1.5931 | 4.2028 | 4 | 3 | 1 |

Turning eval comparison (`vy=0.20`, `wz=0.35`, `num_envs=32`, `duration=4`, `warmup=2`):

| checkpoint | target vx | mean vx | mean vy | mean wz | vx abs err | vy abs err | wz abs err | xy abs err | p90 xy err | resets | head/shoulder | speed tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2J 23625 | 3.50 | 3.3805 | 0.1927 | 0.2585 | 0.2086 | 0.1234 | 0.3965 | 0.2646 | 0.4487 | 0 | 0 | 0 |
| Stage2K 23650 | 3.50 | 3.5059 | 0.1896 | 0.2649 | 0.2043 | 0.1254 | 0.3834 | 0.2654 | 0.3995 | 1 | 1 | 0 |
| Stage2J 23625 | 4.00 | 3.3288 | 0.1923 | 0.2573 | 0.6782 | 0.1610 | 0.4447 | 0.7176 | 1.7663 | 2 | 2 | 0 |
| Stage2K 23650 | 4.00 | 3.5103 | 0.2029 | 0.2270 | 0.5273 | 0.1448 | 0.4309 | 0.5752 | 1.3413 | 2 | 2 | 0 |
| Stage2J 23625 | 4.25 | 3.0793 | 0.1375 | 0.2419 | 1.1710 | 0.1933 | 0.4856 | 1.2017 | 2.8706 | 3 | 1 | 2 |
| Stage2K 23650 | 4.25 | 3.3491 | 0.1725 | 0.2361 | 0.9067 | 0.1882 | 0.4946 | 0.9476 | 2.4999 | 2 | 2 | 0 |

Stage2K checkpoint choice:

- `model_23650.pt` is useful and should be retained.
- It improves Stage2J's high-speed turning objective:
  - `4.25 + vy0.20 + wz0.35`: mean vx `3.0793 -> 3.3491`
  - speed-tracking resets `2 -> 0`
  - `xy_abs_err 1.2017 -> 0.9476`
- It partially improves straight high-speed retention at `3.5` and `4.25`, but not uniformly:
  - straight `3.5`: better than Stage2J
  - straight `4.0`: worse than Stage2J in this short eval, with more resets
  - straight `4.25`: better than Stage2J
  - straight `4.5`: still too weak
- Current recommendation:
  - keep Stage2K `model_23650.pt` as the best mixed turning/speed checkpoint so far.
  - do not continue this exact Stage2K run blindly; next stage should either soften the `4.0` instability or use a curriculum/mixture that protects the straight `4.0` pocket while pushing `4.25+`.

Stage2K play/export validation:

- stopped the previous Stage2J GUI play and started Stage2K GUI play.
- play unit:
  `z1_stage2k_play_20260615_002008.service`
- play stdout:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/z1_sprint_amp_stage2k_play_23650_cmdx3p5_4p65_cmdy0p22_yaw0p40_env16_20260615_002008.out`
- checkpoint:
  `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941/model_23650.pt`
- play command range:
  - `lin_vel_x=(3.5, 4.65)`
  - `lin_vel_y=(-0.22, 0.22)`
  - `ang_vel_z=(-0.40, 0.40)`
- exported artifacts:
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941/exported/policy.pt`
  - `/home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-15_00-09-58_z1_sprint_amp_stage2k_mixedretention_fromstage2j23625_cmdx3p5_4p65_cmdy0p22_yaw0p40_trackxy1p95_prog0p30_env1024_20260615_000941/exported/policy.onnx`

## Stage2L: stability anchor for the 4.0 m/s pocket

Purpose:

- Build from Stage2K `model_23650.pt`.
- Keep the useful Stage2K gains at `4.25 m/s` and high-speed turning.
- Repair Stage2K's weaker straight `4.0 m/s` eval, where short fixed-speed eval had more resets than Stage2J.
- Do not widen the speed target yet; this is a stabilization stage before pushing toward `4.5-5.0 m/s`.

Config changes:

- task: `magicbot_z1_flat_sprint_amp_stage2l_stabilityanchor`
- base: `MagicBotZ1FlatSprintAMPStage2KMixedRetentionEnvCfg`
- command range:
  - `lin_vel_x=(3.4, 4.55)`
  - `lin_vel_y=(-0.18, 0.18)`
  - `ang_vel_z=(-0.32, 0.32)`
- reference motion:
  - `min_command_speed=3.4`
  - `max_reference_speed=5.2`
  - `speed_match_tolerance=0.75`
- reward tuning:
  - `track_lin_vel_xy_exp.weight=1.85`
  - `track_lin_vel_xy_exp.std=1.05`
  - `track_ang_vel_z_exp.weight=1.20`
  - `track_ang_vel_z_exp.std=0.60`
  - `forward_speed_progress.weight=0.26`
  - `forward_speed_progress.min_command_x=3.4`
- retained safety/regularization:
  - `speed_tracking_duration_s=2.5`
  - `energy.weight=-6e-4`
  - `action_rate_l2.weight=-7.5e-3`
  - head/shoulder contact termination and penalty unchanged from Stage2F+
- agent:
  - `learning_rate=2e-5`
  - `motion_prior.reward_coef=0.08`
  - `motion_prior.reward_min_command_speed=3.4`
  - `save_interval=25`

Validation:

- `py_compile` passed for:
  - `legged_lab/envs/magicbot_z1/z1_config.py`
  - `legged_lab/envs/__init__.py`
- registry check passed:
  - task resolves as `magicbot_z1_flat_sprint_amp_stage2l_stabilityanchor`
  - command ranges, reward weights, AMP gate, and `speed_tracking_duration_s=2.5` match the intended Stage2L settings.
