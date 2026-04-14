# RL Robot Navigation + Live Dashboard

A full reinforcement learning project where a PPO agent learns 2D robot navigation and streams live training telemetry to a web dashboard.

## Demo GIF

![Robot Safety Run Demo](dashboard/public/demo_robot_safety_run.gif)

Generated from live telemetry with:

```bash
cd sim
source venv/bin/activate
python generate_demo_gif.py --output ../dashboard/public/demo_robot_safety_run.gif
```

## Quick start (3 terminals)

```bash
# Terminal 1
cd server && npm install && npm start

# Terminal 2
cd dashboard && npm install && npm start

# Terminal 3
cd sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py --timesteps 100000
```

## Frontend navigation

- `http://localhost:3100/` -> Home / Control Center
- `http://localhost:3100/monitor.html` -> Live training dashboard
- Home page includes the main entry link to the monitor page and quick-start commands

## What is built

- Custom Gymnasium navigation environment (`sim/env/robot_nav_env.py`)
- PPO training pipeline via Stable-Baselines3 (`sim/train.py`)
- Episode-level telemetry emitter (`sim/utils/training_callback.py`)
- Real-time metrics backend with Socket.IO (`server/index.js`)
- Live dashboard with charts + trajectory viewer (`dashboard/public/*`)
- Frontend navigation with landing page + dedicated monitor route
- Evaluation and rollout scripts (`sim/evaluate.py`, `sim/play.py`)

## Project layout

```text
RL/
├── sim/
│   ├── train.py
│   ├── evaluate.py
│   ├── play.py
│   ├── generate_demo_gif.py
│   ├── train_multi_seed.py
│   ├── requirements.txt
│   ├── env/robot_nav_env.py
│   ├── agents/ppo_agent.py
│   └── utils/
│       ├── metrics_emitter.py
│       └── training_callback.py
├── server/
│   ├── package.json
│   └── index.js
└── dashboard/
    ├── package.json
    ├── server.js
    └── public/
        ├── index.html
        ├── monitor.html
        ├── styles.css
        └── app.js
```

## Run locally

### 1) Start metrics backend (default: `http://localhost:4100`)

```bash
cd server
npm install
npm start
```

### 2) Start dashboard (default: `http://localhost:3100`)

```bash
cd dashboard
npm install
npm start
```

Frontend routes:
- `http://localhost:3100/` -> landing page (easy navigation)
- `http://localhost:3100/monitor.html` -> live metrics dashboard

### 3) Train PPO and stream live metrics

```bash
cd sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py --timesteps 100000 --algo recurrent_ppo
```

Recommended higher-quality run:

```bash
python train.py \
  --timesteps 1200000 \
  --algo recurrent_ppo \
  --num-envs 8 \
  --obstacle-curriculum 8,12,16,20,22 \
  --n-steps 512 \
  --batch-size 1024 \
  --learning-rate 2e-4 \
  --ent-coef 2e-4 \
  --revisit-penalty 0.25 \
  --near-hazard-penalty 0.35 \
  --near-hazard-threshold 0.20 \
  --eval-freq 24000 \
  --n-eval-episodes 100
```

This now includes:
- obstacle-sensor observations (ray distances + immediate hazard flags)
- multi-env rollout collection
- curriculum learning by obstacle count
- recurrent PPO option (`--algo recurrent_ppo`)
- anti-loop reward shaping (revisit + near-hazard penalties)
- automatic periodic evaluation + best-checkpoint saving with model-specific paths
  (for default model path: `sim/models/ppo_robot_nav_best_model/best_model.zip`)
- explicit run contract printout (`run_id`, model path, best checkpoint dir, eval log dir, eval protocol)

Obstacle note:
- Default training now uses `--obstacle-curriculum 22` (obstacles always on).
- If you want a curriculum, use values that still keep obstacles present (for example `10,15,22`).

### 4) Evaluate trained model

```bash
python evaluate.py --episodes 100 --algo recurrent_ppo
```

You can also control difficulty and evaluation mode:

```bash
python evaluate.py --model-path models/ppo_robot_nav.zip --episodes 300 --obstacle-count 22 --algo recurrent_ppo
python evaluate.py --model-path models/ppo_robot_nav_best_model/best_model.zip --episodes 300 --obstacle-count 22 --stochastic --algo recurrent_ppo
```

Evaluation default model resolution:
- if `--model-path` is omitted, `evaluate.py` tries:
  - `models/ppo_robot_nav_best_model/best_model.zip` (new default training layout)
  - `models/best_model/best_model.zip` (legacy layout)
  - `models/ppo_robot_nav.zip` (final fallback)

### 4.1) Current validated baseline (2026-04-13)

Validated checkpoint:
- `sim/models/runs/sweep_apr13/seed_11/best_model/best_model.zip`

Validation protocol:
- 500 episodes per condition
- eval seeds: `101,202,303,404,505`
- obstacle counts: `16,20,22,26`
- deterministic + stochastic policy evaluation

Aggregate results across eval seeds:

| Obstacle | Deterministic Success | Deterministic Collision | Stochastic Success | Stochastic Collision |
|---|---:|---:|---:|---:|
| 16 | 96.40% | 2.64% | 96.96% | 2.84% |
| 20 | 95.76% | 3.04% | 96.24% | 3.44% |
| 22 | 95.28% | 3.28% | 95.68% | 3.96% |
| 26 | 93.24% | 4.84% | 94.28% | 5.44% |

Overall aggregate:
- deterministic: success `95.17%`, collision `3.45%`
- stochastic: success `95.79%`, collision `3.92%`

Status:
- good enough to ship as a baseline (`v1`) for sim deployment.

### 4.2) Separate hardening run (next)

Goal:
- improve high-density robustness and keep collisions low under harder maps.

Recommended changes for hardening:
- keep per-seed run isolation (`--run-id`, `models/runs/...`)
- train on a harder curriculum (example: `12,18,22,26,30`)
- evaluate on harder obstacles (example: `20,22,26,30`)
- use `--eval-target best` and retain `--n-eval-episodes 100+`

Example hardening sweep:

```bash
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

Suggested acceptance bar for hardening:
- obstacle 26: success `>=95%`, collision `<=5%`
- obstacle 30: success `>=90%`, collision `<=8%`
- stochastic success drop vs deterministic: `<=3%`

Multi-seed search (recommended for >90% target):

```bash
python train_multi_seed.py \
  --seeds 11,22,33,44,55 \
  --timesteps 1200000 \
  --episodes 500 \
  --num-envs 8 \
  --obstacle-curriculum 8,12,16,20,22 \
  --algo recurrent_ppo \
  --run-id sweep_apr13 \
  --eval-target best \
  --eval-obstacle-counts 16,20,22,26
```

This now creates isolated artifacts per seed under:
- `sim/models/runs/<run_id>/seed_<seed>/...`

### 5) Run one rollout in terminal

```bash
python play.py --model-path models/ppo_robot_nav.zip
```

## Live metrics sent to dashboard

- `reward`
- `avg_reward_100`
- `success_rate`
- `collision_rate`
- `steps`
- `trajectory`, `goal`, `obstacles`, `grid_size`

## Notes

- Ports are configurable via env vars:
  - backend: `PORT` (default `4100`)
  - dashboard: `PORT` (default `3100`)
- Training emits metrics via HTTP to backend endpoint `/metrics`, then backend broadcasts over WebSockets.

## Future Goals (Arduino Deployment After >90% Success)

- Validate policy robustness in sim first: test across randomized starts, obstacles, and sensor noise (target: maintain >90% success, low wall/tape collisions).
- Convert policy to embedded-friendly controller: distill PPO policy into a smaller MLP or rule-assisted policy suitable for microcontrollers.
- Export model weights for firmware inference: generate fixed-point/int8 weights and run forward pass in C/C++ on-device.
- Select hardware that can handle inference reliably: prefer ESP32/Teensy-class boards (typical Arduino Uno RAM/flash is often too limited for NN inference).
- Map sim observations to real sensors: fuse wheel odometry + IMU + distance sensors to match `(x, y, goal, distance)`-style inputs.
- Implement motor control bridge: translate policy action output to motor driver commands (`up/down/left/right` equivalent differential drive motions).
- Add hard safety layer in firmware: emergency stop, wall/tape override, timeout recovery, and manual kill-switch independent of RL output.
- Run sim-to-real calibration loop: tune reward/model based on real-world drift, wheel slip, and sensor latency.
- Deploy telemetry back to dashboard: stream live robot metrics (success rate, collisions, path trace) from Arduino/edge bridge to current backend.

## License

MIT
