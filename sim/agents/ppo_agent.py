from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def build_agent(
    env: VecEnv,
    algo: str = "recurrent_ppo",
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 256,
    ent_coef: float = 5e-4,
) -> Any:
    common_kwargs = {
        "env": env,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
    }

    if algo == "ppo":
        return PPO(
            policy="MlpPolicy",
            policy_kwargs={"net_arch": [256, 256]},
            **common_kwargs,
        )

    if algo == "recurrent_ppo":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:
            raise ImportError(
                "RecurrentPPO requires sb3-contrib. Install dependencies with: pip install -r requirements.txt"
            ) from exc

        return RecurrentPPO(
            policy="MlpLstmPolicy",
            policy_kwargs={"net_arch": [256, 256], "lstm_hidden_size": 128},
            **common_kwargs,
        )

    raise ValueError(f"Unsupported algo: {algo}")
