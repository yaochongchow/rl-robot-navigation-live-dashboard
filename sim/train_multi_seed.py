from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training/evaluation sweep")
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--obstacle-curriculum", type=str, default="8,12,16,20,22")
    parser.add_argument("--server-url", type=str, default="http://localhost:4100")
    parser.add_argument("--algo", type=str, default="recurrent_ppo", choices=["ppo", "recurrent_ppo"])
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-root", type=str, default="models/runs")
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--eval-obstacle-counts", type=str, default="16,20,22,26")
    parser.add_argument("--eval-target", type=str, default="best", choices=["best", "final"])
    parser.add_argument("--stochastic-eval", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> str:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    print(proc.stdout)
    return proc.stdout


def parse_success_rate(stdout: str) -> float:
    match = re.search(r"Success rate:\s*([0-9.]+)%", stdout)
    if not match:
        raise RuntimeError("Could not parse success rate from evaluator output")
    return float(match.group(1))


def parse_collision_rate(stdout: str) -> float:
    match = re.search(r"Collision rate:\s*([0-9.]+)%", stdout)
    if not match:
        raise RuntimeError("Could not parse collision rate from evaluator output")
    return float(match.group(1))


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def main() -> None:
    args = parse_args()
    seeds = parse_int_csv(args.seeds)
    eval_obstacle_counts = parse_int_csv(args.eval_obstacle_counts)
    if not seeds:
        raise ValueError("No valid seeds provided")

    root = Path(__file__).resolve().parent
    run_id = args.run_id or datetime.now().strftime("multi_seed_%Y%m%d_%H%M%S")
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = root / run_root
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Multi-seed Execution Contract ===")
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"seeds={seeds}")
    print(f"model_algo={args.algo}")
    print(f"training_curriculum={args.obstacle_curriculum}")
    print(f"eval_target={args.eval_target}")
    print(f"eval_episodes_per_obstacle={args.episodes}")
    print(f"eval_obstacle_counts={eval_obstacle_counts}")
    print(f"eval_stochastic={args.stochastic_eval}")

    results: list[dict[str, Any]] = []

    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_base = seed_dir / f"seed_{seed}_robot_nav"
        best_model_dir = seed_dir / "best_model"
        eval_log_dir = seed_dir / "eval_logs"

        train_cmd = [
            sys.executable,
            "train.py",
            "--seed",
            str(seed),
            "--timesteps",
            str(args.timesteps),
            "--num-envs",
            str(args.num_envs),
            "--obstacle-curriculum",
            args.obstacle_curriculum,
            "--server-url",
            args.server_url,
            "--algo",
            args.algo,
            "--model-path",
            str(model_base),
            "--best-model-dir",
            str(best_model_dir),
            "--eval-log-dir",
            str(eval_log_dir),
            "--n-eval-episodes",
            str(args.n_eval_episodes),
            "--run-id",
            f"{run_id}_seed_{seed}",
        ]
        run_cmd(train_cmd, root)

        final_model_path = model_base.with_suffix(".zip")
        eval_model_path = (
            best_model_dir / "best_model.zip"
            if args.eval_target == "best"
            else final_model_path
        )
        if not eval_model_path.exists():
            raise FileNotFoundError(f"Expected eval model does not exist: {eval_model_path}")

        seed_evals: list[tuple[int, float, float]] = []
        for obstacle_count in eval_obstacle_counts:
            eval_cmd = [
                sys.executable,
                "evaluate.py",
                "--model-path",
                str(eval_model_path),
                "--episodes",
                str(args.episodes),
                "--obstacle-count",
                str(obstacle_count),
                "--algo",
                args.algo,
            ]
            if args.stochastic_eval:
                eval_cmd.append("--stochastic")
            eval_stdout = run_cmd(eval_cmd, root)
            success_rate = parse_success_rate(eval_stdout)
            collision_rate = parse_collision_rate(eval_stdout)
            seed_evals.append((obstacle_count, success_rate, collision_rate))

        avg_success = sum(item[1] for item in seed_evals) / len(seed_evals)
        avg_collision = sum(item[2] for item in seed_evals) / len(seed_evals)
        results.append(
            {
                "seed": seed,
                "avg_success": avg_success,
                "avg_collision": avg_collision,
                "eval_model_path": str(eval_model_path),
                "final_model_path": str(final_model_path),
                "best_model_path": str(best_model_dir / "best_model.zip"),
                "evals": seed_evals,
            }
        )

    results.sort(key=lambda item: float(item["avg_success"]), reverse=True)

    print("\n=== Multi-seed Summary ===")
    for item in results:
        seed = int(item["seed"])
        avg_success = float(item["avg_success"])
        avg_collision = float(item["avg_collision"])
        model_path = str(item["eval_model_path"])
        print(
            f"seed={seed} avg_success={avg_success:.2f}% avg_collision={avg_collision:.2f}% "
            f"eval_model={model_path}"
        )
        per_obstacle = ", ".join(
            [
                f"obs{obstacle}:success={success:.2f}% collision={collision:.2f}%"
                for obstacle, success, collision in item["evals"]
            ]
        )
        print(f"  {per_obstacle}")

    best = results[0]
    print("\nBest run")
    print(
        f"seed={int(best['seed'])} avg_success={float(best['avg_success']):.2f}% "
        f"avg_collision={float(best['avg_collision']):.2f}% "
        f"eval_model={best['eval_model_path']}"
    )


if __name__ == "__main__":
    main()
