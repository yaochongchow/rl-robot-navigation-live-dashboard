from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def build_ppo_agent(env: VecEnv) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )
