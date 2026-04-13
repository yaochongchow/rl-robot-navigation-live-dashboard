from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from env.robot_nav_env import RobotNavEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO/RecurrentPPO agent")
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_nav.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--obstacle-count", type=int, default=22)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--algo", type=str, default="auto", choices=["auto", "ppo", "recurrent_ppo"])
    return parser.parse_args()


def load_model(model_path: str, algo: str):
    if algo == "ppo":
        return PPO.load(model_path), False

    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(model_path), True

    # auto mode: try recurrent first, then fallback to PPO
    try:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(model_path), True
    except Exception:  # noqa: BLE001
        return PPO.load(model_path), False


def main() -> None:
    args = parse_args()
    env = RobotNavEnv(seed=args.seed, obstacle_count=args.obstacle_count)
    model, is_recurrent = load_model(args.model_path, args.algo)

    rewards: list[float] = []
    successes = 0
    collisions = 0
    steps: list[int] = []
    reasons: Counter[str] = Counter()

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        state = None
        episode_start = np.array([True], dtype=bool)

        while not done:
            if is_recurrent:
                action, state = model.predict(
                    obs,
                    state=state,
                    episode_start=episode_start,
                    deterministic=not args.stochastic,
                )
                episode_start[...] = False
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            if done:
                successes += int(info.get("success", False))
                collisions += int(info.get("collision", False))
                reasons[str(info.get("termination_reason", "unknown"))] += 1

        rewards.append(total_reward)
        steps.append(step_count)

    print("Evaluation complete")
    print(f"Episodes: {args.episodes}")
    print(f"Obstacle count: {args.obstacle_count}")
    print(f"Deterministic policy: {not args.stochastic}")
    print(f"Algo used: {'recurrent_ppo' if is_recurrent else 'ppo'}")
    print(f"Success rate: {successes / args.episodes:.2%}")
    print(f"Collision rate: {collisions / args.episodes:.2%}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average steps: {np.mean(steps):.2f}")
    print(f"Termination reasons: {dict(reasons)}")


if __name__ == "__main__":
    main()
