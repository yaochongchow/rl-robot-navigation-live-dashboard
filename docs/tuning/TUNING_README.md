# RL Tuning README

Last updated: 2026-04-14 (America/Los_Angeles)

All updates below are collapsed by default.

<details>
<summary><strong>Update 1: Workflow Patch Applied (2026-04-13)</strong></summary>

- Added per-run/per-seed artifact isolation.
- Added safer evaluation model resolution defaults.
- Increased training periodic eval episodes default from `30` to `100`.
- Added multi-obstacle reporting support in multi-seed sweeps.
- Added obstacle-capacity guardrails in environment generation.

Key behavior changes:
- `train.py` supports `--run-id`, `--best-model-dir`, `--eval-log-dir`.
- `train_multi_seed.py` supports `--eval-obstacle-counts`, `--eval-target`, run-scoped outputs.
- `evaluate.py` auto-resolves best known checkpoint paths when `--model-path` is omitted.

</details>

<details>
<summary><strong>Update 2: Baseline v1 Validation (sweep_apr13 / seed_11 best)</strong></summary>

Validated checkpoint:
- `sim/models/runs/sweep_apr13/seed_11/best_model/best_model.zip`

Protocol:
- 500 episodes per condition
- eval seeds: `101,202,303,404,505`
- obstacle counts: `16,20,22,26`
- deterministic + stochastic

Aggregate results:

| Obstacle | Deterministic Success | Deterministic Collision | Stochastic Success | Stochastic Collision |
|---|---:|---:|---:|---:|
| 16 | 96.40% | 2.64% | 96.96% | 2.84% |
| 20 | 95.76% | 3.04% | 96.24% | 3.44% |
| 22 | 95.28% | 3.28% | 95.68% | 3.96% |
| 26 | 93.24% | 4.84% | 94.28% | 5.44% |

Overall:
- deterministic: success `95.17%`, collision `3.45%`
- stochastic: success `95.79%`, collision `3.92%`

Decision at that time:
- Good enough to ship as sim baseline `v1`.

</details>

<details>
<summary><strong>Update 3: Hardening_v1 Run Completed (2026-04-14)</strong></summary>

Run:
- `sim/models/runs/hardening_v1/`
- seeds: `11,33,44,55,77`
- curriculum: `12,18,22,26,30`

Quick deterministic ranking on hard set (`20,22,26,30`, 300 eps each):

| Seed | Avg Success | Avg Collision | Avg Steps |
|---|---:|---:|---:|
| 55 | 97.17% | 1.42% | 15.08 |
| 44 | 96.67% | 2.33% | 14.69 |
| 33 | 96.17% | 1.75% | 15.80 |
| 11 | 95.42% | 3.08% | 15.05 |
| 77 | 93.00% | 3.00% | 18.01 |

Winner selected:
- `sim/models/runs/hardening_v1/seed_55/best_model/best_model.zip`

</details>

<details>
<summary><strong>Update 4: Hardening_v1 Winner Robustness Evaluation (seed_55 best)</strong></summary>

Protocol:
- 500 episodes per condition
- eval seeds: `101,202,303,404,505`
- obstacle counts: `20,22,26,30`
- deterministic + stochastic

Aggregate results:

| Obstacle | Deterministic Success | Deterministic Collision | Stochastic Success | Stochastic Collision |
|---|---:|---:|---:|---:|
| 20 | 98.12% | 0.84% | 98.52% | 1.00% |
| 22 | 97.24% | 1.40% | 98.04% | 1.56% |
| 26 | 96.76% | 1.80% | 97.60% | 1.96% |
| 30 | 95.64% | 2.16% | 96.92% | 2.44% |

Overall:
- deterministic: success `96.94%`, collision `1.55%`
- stochastic: success `97.77%`, collision `1.74%`

Gate check:
- obstacle 26 gate (`>=95% success`, `<=5% collision`): pass
- obstacle 30 gate (`>=90% success`, `<=8% collision`): pass
- stochastic gap gate (`<=3%`): pass

Decision:
- Promote this model as sim `v2` candidate.

</details>

<details>
<summary><strong>Update 5: Actions To Take Next</strong></summary>

1. Freeze and label the winner as production candidate (`v2`).
- Keep immutable copy path and record model checksum.
- Suggested path: `sim/models/production/v2_seed55_hardening.zip`.

2. Run holdout robustness test before deployment handoff.
- Add unseen eval seeds (for example: `606,707,808,909,1001`).
- Keep obstacles `20,22,26,30` and 500 episodes per condition.

3. Add noise/domain randomization evaluation.
- Test sensor noise, obstacle jitter, and start-position perturbations.
- Confirm success/collision remain within an acceptable drop band.

4. Run long-duration stress evaluation.
- Execute >=10,000 episodes at obstacle `30` and monitor rare failure modes.
- Track wall collisions, tape collisions, and timeout rates separately.

5. Prepare sim-to-real handoff packet.
- Export selected checkpoint, eval tables, termination breakdowns, and config.
- Include acceptance criteria and fallback model (`v1`).

6. Start `hardening_v2` only if needed.
- Trigger condition: holdout/noise/stress gates fail.
- Otherwise, avoid unnecessary retraining and proceed to integration.

</details>
