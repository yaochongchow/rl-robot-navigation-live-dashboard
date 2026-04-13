# RL Navigation Simulator with Real-Time Dashboard

This implementation includes a complete simulation + telemetry + dashboard stack for monitoring PPO training behavior in real time.

## Quick start

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
python train.py --timesteps 100000 --server-url http://localhost:4100
```

Then open:
- `http://localhost:3100/` for Home / navigation
- `http://localhost:3100/monitor.html` for live training metrics

## Architecture

```text
[RobotNavEnv + PPO] -> [Metrics Callback] -> POST /metrics -> [Node Backend + Socket.IO] -> [Live Dashboard]
```

## Implemented features

- Custom 2D environment with randomized start/goal and obstacles
- Reward design:
  - `+100` goal
  - `-100` collision / out-of-bounds
  - `-1` step
  - distance-improvement shaping
- PPO training using Stable-Baselines3
- Real-time metric streaming each episode
- Dashboard visualizations:
  - reward curve
  - success/collision trend
  - latest episode trajectory on grid

## Start services

### Backend server

```bash
cd server
npm install
npm start
```

Runs on `http://localhost:4100` by default.

### Dashboard

```bash
cd dashboard
npm install
npm start
```

Runs on `http://localhost:3100` by default.

Frontend routes:
- `http://localhost:3100/` for the navigation home page
- `http://localhost:3100/monitor.html` for the live training dashboard

### Simulator training

```bash
cd sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py --timesteps 100000 --server-url http://localhost:4100
```

## Extra scripts

```bash
python evaluate.py --model-path models/ppo_robot_nav.zip --episodes 100
python play.py --model-path models/ppo_robot_nav.zip
```

## API surface (backend)

- `GET /health` health + status
- `GET /history` historical metrics for dashboard bootstrap
- `POST /metrics` ingest episode telemetry and broadcast live
- `POST /status` ingest training status (`training_started`, `training_finished`, etc.)

## Config

- Backend port: `PORT` (default `4100`)
- Dashboard port: `PORT` (default `3100`)
- Backend history length: `MAX_HISTORY` (default `1000`)
