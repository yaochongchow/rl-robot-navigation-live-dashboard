from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from env.robot_nav_env import RobotNavEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play one episode with trained agent")
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_nav.zip")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-steps", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = RobotNavEnv(seed=args.seed, max_steps=args.max_steps)
    model = PPO.load(args.model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    print("Running one rollout...\n")
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward

    env.render()
    print(f"Episode reward: {total_reward:.2f}")
    print(f"Success: {info.get('success', False)}")
    print(f"Collision: {info.get('collision', False)}")
    print(f"Steps: {info.get('steps', 0)}")


if __name__ == "__main__":
    main()
