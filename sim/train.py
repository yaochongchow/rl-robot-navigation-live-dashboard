from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from agents.ppo_agent import build_agent
from env.robot_nav_env import RobotNavEnv
from utils.metrics_emitter import MetricsEmitter
from utils.training_callback import TrainingMetricsCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on RobotNavEnv")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-path", type=str, default="models/ppo_robot_nav")
    parser.add_argument("--server-url", type=str, default="http://localhost:4100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo", type=str, default="recurrent_ppo", choices=["ppo", "recurrent_ppo"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--obstacle-count", type=int, default=22)
    parser.add_argument(
        "--obstacle-curriculum",
        type=str,
        default="22",
        help="Comma-separated obstacle counts for staged curriculum learning (use 22 to always keep obstacles on).",
    )
    parser.add_argument("--eval-freq", type=int, default=12_000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=2e-4)
    parser.add_argument("--revisit-penalty", type=float, default=0.25)
    parser.add_argument("--near-hazard-penalty", type=float, default=0.35)
    parser.add_argument(
        "--near-hazard-threshold",
        type=float,
        default=0.20,
        help="Normalized ray-distance threshold below which hazard proximity penalty applies.",
    )
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def make_env(
    seed: int,
    obstacle_count: int,
    revisit_penalty: float,
    near_hazard_penalty: float,
    near_hazard_threshold: float,
):
    def _init() -> Monitor:
        env = RobotNavEnv(
            seed=seed,
            obstacle_count=obstacle_count,
            revisit_penalty=revisit_penalty,
            near_hazard_penalty=near_hazard_penalty,
            near_hazard_threshold=near_hazard_threshold,
        )
        return Monitor(env)

    return _init


def build_vec_env(
    seed: int,
    num_envs: int,
    obstacle_count: int,
    revisit_penalty: float,
    near_hazard_penalty: float,
    near_hazard_threshold: float,
) -> VecEnv:
    env_fns = [
        make_env(seed + idx, obstacle_count, revisit_penalty, near_hazard_penalty, near_hazard_threshold)
        for idx in range(num_envs)
    ]
    if num_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def parse_curriculum(obstacle_curriculum: str, fallback_obstacle_count: int) -> list[int]:
    items = [s.strip() for s in obstacle_curriculum.split(",") if s.strip()]
    if not items:
        return [fallback_obstacle_count]

    curriculum: list[int] = []
    for item in items:
        value = int(item)
        if value < 0:
            raise ValueError("Obstacle counts in curriculum must be >= 0")
        curriculum.append(value)
    return curriculum


def split_timesteps(total_timesteps: int, phase_count: int) -> list[int]:
    # Allocate more budget to harder curriculum phases later in training.
    weights = [idx + 1 for idx in range(phase_count)]
    weight_sum = sum(weights)
    plan = [int(total_timesteps * w / weight_sum) for w in weights]
    assigned = sum(plan)
    for idx in range(total_timesteps - assigned):
        plan[-(idx % phase_count) - 1] += 1
    return plan


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    curriculum = parse_curriculum(args.obstacle_curriculum, args.obstacle_count)
    timesteps_plan = split_timesteps(args.timesteps, len(curriculum))

    vec_env = build_vec_env(
        args.seed,
        args.num_envs,
        curriculum[0],
        args.revisit_penalty,
        args.near_hazard_penalty,
        args.near_hazard_threshold,
    )
    agent = build_agent(
        vec_env,
        algo=args.algo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
    )

    if args.num_envs == 1:
        eval_env = DummyVecEnv(
            [
                make_env(
                    args.seed + 10_000,
                    curriculum[-1],
                    args.revisit_penalty,
                    args.near_hazard_penalty,
                    args.near_hazard_threshold,
                )
            ]
        )
    else:
        eval_env = SubprocVecEnv(
            [
                make_env(
                    args.seed + 10_000,
                    curriculum[-1],
                    args.revisit_penalty,
                    args.near_hazard_penalty,
                    args.near_hazard_threshold,
                )
            ]
        )
    best_model_dir = model_path.parent / "best_model"
    eval_log_dir = model_path.parent / "eval_logs"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    emitter = MetricsEmitter(base_url=args.server_url)
    emitter.send_status({"status": "training_started"})

    metrics_callback = TrainingMetricsCallback(emitter=emitter)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(args.eval_freq // max(args.num_envs, 1), 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callback = CallbackList([metrics_callback, eval_callback])

    for phase_idx, obstacle_count in enumerate(curriculum):
        vec_env.set_attr("obstacle_count", obstacle_count)
        phase_timesteps = timesteps_plan[phase_idx]
        if phase_timesteps <= 0:
            continue

        emitter.send_status(
            {
                "status": "training_phase",
                "phase": phase_idx + 1,
                "phase_total": len(curriculum),
                "obstacle_count": obstacle_count,
                "phase_timesteps": phase_timesteps,
            }
        )

        agent.learn(
            total_timesteps=phase_timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
            reset_num_timesteps=(phase_idx == 0),
        )

    agent.save(str(model_path))
    emitter.send_status(
        {
            "status": "training_finished",
            "episodes": metrics_callback.episode_count,
            "success_rate": (
                metrics_callback.successes / metrics_callback.episode_count
                if metrics_callback.episode_count
                else 0.0
            ),
            "algo": args.algo,
            "curriculum": curriculum,
            "best_model_path": str(best_model_dir),
        }
    )
    print(f"Model saved to {model_path}.zip")
    print(f"Best eval model saved under {best_model_dir}")

    eval_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
