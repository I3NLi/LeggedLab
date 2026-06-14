# MagicBot Z1 Goal Mode Prompt

Use this prompt when resuming MagicBot Z1 locomotion work in goal mode.

```text
Objective: Improve MagicBot Z1 flat locomotion while preserving the current stable baseline.

Workspace:
- LeggedLab repo: /home/hiyio/LeggedLab
- Deploy repo: /home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot
- Current baseline checkpoint:
  /home/hiyio/LeggedLab/logs/magicbot_z1_flat/2026-06-13_22-11-32_z1_flat_cmdslew2_1_2_alive0p02_speeddur2p5_cmdx-2p5_5_resume21600_env20000_20260613_220958/model_23000.pt

Known baseline metrics:
- Mean reward: 3.33
- Mean episode length: 888.78
- Timeout ratio: 0.7646
- Head/shoulder contact ratio: 0.0200
- Speed tracking failure ratio: 0.2153

Hard constraints:
- Do not use or recreate the deleted post-23000 speed_tracking_duration_s=5.0 continuation as a baseline.
- Keep speed_tracking_duration_s at 2.5 unless there is a controlled experiment with lower PPO learning rate and a clear rollback plan.
- Do not kill unrelated DogUrdf17/T800/other Isaac processes unless explicitly asked.
- Before starting train/play, report current GPU processes and memory.
- Every training run must write a short run description and update RUN_INDEX.md with purpose, source checkpoint, key config, metrics, and keep/delete decision.
- If a run degrades, stop it early, keep only useful checkpoints, delete bad exports, and record why.
- Keep MagicBot deploy YAML generation enabled:
  --deploy_yaml_root=/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot

Preferred experiment flow:
1. Start from model_23000.pt.
2. Change only one risk factor at a time.
3. For 10000 env fine-tuning, consider lowering learning_rate from 1e-3 to 3e-4 or 5e-4 before changing termination duration.
4. Watch action noise std, value loss, action_rate_l2, head/shoulder contact ratio, and speed_tracking_failure_ratio.
5. Candidate checkpoints must beat or match model_23000.pt on stability before deployment testing.

Verification:
- Run play with explicit command ranges, for example:
  --play_lin_vel_x_min=-2.5 --play_lin_vel_x=5.0 --play_lin_vel_y=0.0 --play_ang_vel_z=0.0
- If IsaacSim GUI play crashes with root_view/getRootTransforms invalidation, note it as a GUI runtime issue, not a policy-load failure, and use export/headless or deployment-side simulation for policy validation.
```
