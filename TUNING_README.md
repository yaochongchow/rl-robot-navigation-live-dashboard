# RL Tuning Log and Next-Execution Plan

Last updated: 2026-04-13 (America/Los_Angeles)

## 1) What You Have Done So Far

This section logs the actual tuning executions found in the workspace artifacts.

## Run Timeline (Chronological)

| Time (PDT) | Artifact | What changed vs previous run | Core setup |
|---|---|---|---|
| 2026-04-13 00:36:29 | `sim/models/ppo_robot_nav.zip` | Initial baseline run in this workspace | `recurrent_ppo`, `n_envs=2`, `n_steps=128`, `batch_size=128`, `ent_coef=2e-4`, loaded model timesteps=`4096` |
| 2026-04-13 01:46:17 | `sim/models/seed_11_robot_nav.zip` | Switched to long multi-seed setup | `recurrent_ppo`, `n_envs=8`, `timesteps=1,200,000`, curriculum `8,12,16,20,22`, `n_steps=512`, `batch_size=256`, `ent_coef=2e-4` |
| 2026-04-13 03:28:31 | `sim/models/seed_22_robot_nav.zip` | Changed random seed only (`11 -> 22`) | Same hyperparameters as seed_11 run |
| 2026-04-13 04:49:48 | `sim/models/seed_33_robot_nav.zip` | Changed random seed only (`22 -> 33`) | Same hyperparameters as seed_11 run |
| 2026-04-13 06:15:20 | `sim/models/seed_44_robot_nav.zip` | Changed random seed only (`33 -> 44`) | Same hyperparameters as seed_11 run |
| 2026-04-13 07:42:18 | `sim/models/best_model/best_model.zip` | Auto-saved best checkpoint from the latest seed training job | Same training configuration as active seed run |
| 2026-04-13 07:43:13 | `sim/models/seed_55_robot_nav.zip` | Changed random seed only (`44 -> 55`) | Same hyperparameters as seed_11 run |

## What Was Different Per Tuning

- Baseline tuning (`ppo_robot_nav`) used a much smaller rollout setup (`n_envs=2`, `n_steps=128`, `batch=128`) and is effectively a weak/early policy.
- Multi-seed tunings (`seed_11..55`) used the same long-horizon recipe; the only tuning difference between those runs was random seed.
- During the multi-seed phase, evaluation artifacts were reused in shared paths (`sim/models/eval_logs`, `sim/models/best_model`), so later seeds overwrote earlier seed eval logs/checkpoints.

## Measured Outcomes (Same Evaluation Protocol)

Protocol used here for direct comparison:
- Environment: obstacle count `22`
- Episodes: `500`
- Deterministic policy
- Evaluation seed stream: `777`

| Model | Success Rate | 95% CI | Collision Rate | Avg Steps |
|---|---:|---:|---:|---:|
| `ppo_robot_nav` | 5.40% | +/- 1.98% | 9.80% | 103.52 |
| `seed_11` | 91.80% | +/- 2.40% | 0.60% | 21.59 |
| `seed_22` | 81.20% | +/- 3.42% | 7.00% | 25.62 |
| `seed_33` | 92.20% | +/- 2.35% | 3.80% | 17.54 |
| `seed_44` | 94.40% | +/- 2.02% | 0.80% | 18.54 |
| `seed_55` | 94.20% | +/- 2.05% | 4.60% | 14.11 |
| `best_model` | 94.00% | +/- 2.08% | 4.00% | 15.12 |

## Latest Training-Curve Snapshot (from `sim/models/eval_logs/evaluations.npz`)

- Eval points: `101`
- Timesteps range: `12,000 -> 1,212,000`
- Mean eval reward: `-151.01 -> 111.91`
- Best mean eval reward: `115.07` at timestep `1,200,000`
- End-of-run reward std across eval episodes: `7.43`

## 2) What Will Be Different In The Next Execution (Pre-Patch Plan)

This section is the **before-execution plan**. These are intended changes for the next run, not yet applied.

## Planned Differences

1. Per-seed artifact isolation (no overwrite)
- Next execution will write each seed's best checkpoint and eval logs to unique paths (example: `sim/models/runs/<run_id>/seed_44/best_model/` and `.../eval_logs/`).
- Reason: preserves each seed's own `best_model` and `evaluations.npz`.

2. Safer default evaluation target
- Next execution will evaluate the chosen target explicitly (prefer best checkpoint path), instead of relying on an ambiguous default model file.
- Reason: avoids accidentally evaluating a stale baseline model.

3. Stronger periodic validation
- Increase online eval episodes per checkpoint (for example from `30` to `100`+).
- Reason: lower variance during model selection.

4. Robustness-first reporting
- Next execution summary will include multi-seed or multi-obstacle evaluation in one report, not only a single obstacle/seed snapshot.
- Reason: better estimate of real generalization.

5. Environment guardrail
- Add a hard cap check for `obstacle_count` against free cells before generation starts.
- Reason: prevents potential infinite loops in obstacle placement.

## Proposed Next Execution Contract

Before launching the next training pass, the run should clearly state:
- exact run id/name
- model output directory
- eval log directory
- best checkpoint directory
- eval protocol (episodes, obstacle levels, deterministic/stochastic mode, seed policy)

When this patch is executed, this document can be extended with:
- "Applied patch details"
- "First execution after patch"
- "Result deltas vs pre-patch baseline"

## 3) Applied Patch Details (Implemented)

Implemented on: 2026-04-13

1. Per-seed artifact isolation
- `train.py` now accepts `--best-model-dir` and `--eval-log-dir`.
- default artifact directories are now model-specific (`<model_stem>_best_model`, `<model_stem>_eval_logs`), reducing overwrite risk.
- `train_multi_seed.py` now writes runs under `models/runs/<run_id>/seed_<seed>/...`.

2. Safer evaluation defaults
- `evaluate.py` no longer hardcodes a single potentially stale model.
- default resolution order is now:
  - `models/ppo_robot_nav_best_model/best_model.zip`
  - `models/best_model/best_model.zip` (legacy)
  - `models/ppo_robot_nav.zip`

3. Stronger periodic validation
- `train.py` default `--n-eval-episodes` increased from `30` to `100`.

4. Robustness-first reporting
- `train_multi_seed.py` now supports multi-obstacle evaluation via `--eval-obstacle-counts`.
- summary is ranked by average success across obstacle counts, not single-point score only.

5. Environment guardrail
- `RobotNavEnv` now validates obstacle capacity and raises clear `ValueError` if impossible.
- obstacle generation now samples from available cells without replacement, avoiding unbounded loops.

## 4) First Execution After Patch (Completed)

Run:
- run id: `sweep_apr13`
- seed: `11`
- eval target: best checkpoint
- checkpoint: `sim/models/runs/sweep_apr13/seed_11/best_model/best_model.zip`

Robustness validation protocol:
- 500 episodes per condition
- eval seeds: `101,202,303,404,505`
- obstacle counts: `16,20,22,26`
- deterministic + stochastic

Aggregate results across eval seeds:

| Obstacle | Deterministic Success | Deterministic Collision | Stochastic Success | Stochastic Collision |
|---|---:|---:|---:|---:|
| 16 | 96.40% | 2.64% | 96.96% | 2.84% |
| 20 | 95.76% | 3.04% | 96.24% | 3.44% |
| 22 | 95.28% | 3.28% | 95.68% | 3.96% |
| 26 | 93.24% | 4.84% | 94.28% | 5.44% |

Overall:
- deterministic: success `95.17%`, collision `3.45%`
- stochastic: success `95.79%`, collision `3.92%`

Conclusion:
- baseline quality is strong enough to ship as sim `v1`.

## 5) Separate Hardening Run Context

Purpose:
- push performance on denser obstacle maps and reduce residual collision rate.

What should be different from baseline run:
1. Use harder training curriculum
- baseline used `8,12,16,20,22`
- hardening should use something like `12,18,22,26,30`

2. Evaluate on harder obstacle set
- baseline focus included `16,20,22,26`
- hardening should report at least `20,22,26,30`

3. Keep robust model selection
- keep `--eval-target best`
- keep higher eval episodes (`--n-eval-episodes 150` recommended)

4. Maintain multi-seed robustness
- run multiple seeds and rank by average success across hard obstacle counts.

Recommended hardening command:

```bash
cd /Users/ycchow/Desktop/RL/sim
source venv/bin/activate
python train_multi_seed.py \
  --seeds 11,33,44,55,77 \
  --timesteps 1200000 \
  --num-envs 8 \
  --obstacle-curriculum 12,18,22,26,30 \
  --n-eval-episodes 150 \
  --episodes 500 \
  --eval-obstacle-counts 20,22,26,30 \
  --eval-target best \
  --algo recurrent_ppo \
  --run-id hardening_v1
```

Acceptance gates for hardening:
- obstacle 26: success `>=95%`, collision `<=5%`
- obstacle 30: success `>=90%`, collision `<=8%`
- stochastic gap vs deterministic success: `<=3%`
