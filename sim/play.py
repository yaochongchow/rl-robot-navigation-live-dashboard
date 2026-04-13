from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from env.robot_nav_env import RobotNavEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play one episode with trained agent")
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_nav.zip")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--obstacle-count", type=int, default=22)
    parser.add_argument("--algo", type=str, default="auto", choices=["auto", "ppo", "recurrent_ppo"])
    return parser.parse_args()


def load_model(model_path: str, algo: str):
    if algo == "ppo":
        return PPO.load(model_path), False

    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(model_path), True

    try:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(model_path), True
    except Exception:  # noqa: BLE001
        return PPO.load(model_path), False


def main() -> None:
    args = parse_args()
    env = RobotNavEnv(seed=args.seed, max_steps=args.max_steps, obstacle_count=args.obstacle_count)
    model, is_recurrent = load_model(args.model_path, args.algo)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    state = None
    episode_start = np.array([True], dtype=bool)

    print("Running one rollout...\n")
    while not done:
        env.render()
        if is_recurrent:
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
            episode_start[...] = False
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward

    env.render()
    print(f"Episode reward: {total_reward:.2f}")
    print(f"Algo used: {'recurrent_ppo' if is_recurrent else 'ppo'}")
    print(f"Success: {info.get('success', False)}")
    print(f"Collision: {info.get('collision', False)}")
    print(f"Termination reason: {info.get('termination_reason', 'unknown')}")
    print(f"Steps: {info.get('steps', 0)}")


if __name__ == "__main__":
    main()
