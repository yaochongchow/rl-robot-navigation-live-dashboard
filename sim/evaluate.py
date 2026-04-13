from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from env.robot_nav_env import RobotNavEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_nav.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = RobotNavEnv(seed=args.seed)
    model = PPO.load(args.model_path)

    rewards: list[float] = []
    successes = 0
    collisions = 0
    steps: list[int] = []

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            if done:
                successes += int(info.get("success", False))
                collisions += int(info.get("collision", False))

        rewards.append(total_reward)
        steps.append(step_count)

    print("Evaluation complete")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {successes / args.episodes:.2%}")
    print(f"Collision rate: {collisions / args.episodes:.2%}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average steps: {np.mean(steps):.2f}")


if __name__ == "__main__":
    main()
