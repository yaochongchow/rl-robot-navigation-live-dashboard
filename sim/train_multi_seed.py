from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training/evaluation sweep")
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--obstacle-curriculum", type=str, default="8,12,16,20,22")
    parser.add_argument("--server-url", type=str, default="http://localhost:4100")
    parser.add_argument("--algo", type=str, default="recurrent_ppo", choices=["ppo", "recurrent_ppo"])
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


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("No valid seeds provided")

    root = Path(__file__).resolve().parent
    results: list[tuple[int, float, str]] = []

    for seed in seeds:
        model_base = f"models/seed_{seed}_robot_nav"
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
            model_base,
        ]
        run_cmd(train_cmd, root)

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--model-path",
            f"{model_base}.zip",
            "--episodes",
            str(args.episodes),
            "--obstacle-count",
            "22",
            "--algo",
            args.algo,
        ]
        eval_stdout = run_cmd(eval_cmd, root)
        success_rate = parse_success_rate(eval_stdout)
        results.append((seed, success_rate, f"{model_base}.zip"))

    results.sort(key=lambda item: item[1], reverse=True)

    print("\n=== Multi-seed Summary ===")
    for seed, success_rate, model_path in results:
        print(f"seed={seed} success={success_rate:.2f}% model={model_path}")

    best_seed, best_success, best_model = results[0]
    print("\nBest run")
    print(f"seed={best_seed} success={best_success:.2f}% model={best_model}")


if __name__ == "__main__":
    main()
