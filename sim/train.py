from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.ppo_agent import build_ppo_agent
from env.robot_nav_env import RobotNavEnv
from utils.metrics_emitter import MetricsEmitter
from utils.training_callback import TrainingMetricsCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on RobotNavEnv")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_nav")
    parser.add_argument("--server-url", type=str, default="http://localhost:4100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def make_env(seed: int):
    def _init() -> Monitor:
        env = RobotNavEnv(seed=seed)
        return Monitor(env)

    return _init


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env(args.seed)])
    agent = build_ppo_agent(vec_env)

    emitter = MetricsEmitter(base_url=args.server_url)
    emitter.send_status({"status": "training_started"})

    callback = TrainingMetricsCallback(emitter=emitter)
    agent.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
    )

    agent.save(str(model_path))
    emitter.send_status(
        {
            "status": "training_finished",
            "episodes": callback.episode_count,
            "success_rate": callback.successes / callback.episode_count if callback.episode_count else 0.0,
        }
    )
    print(f"Model saved to {model_path}.zip")


if __name__ == "__main__":
    main()
