from __future__ import annotations

from collections import deque
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from utils.metrics_emitter import MetricsEmitter


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, emitter: MetricsEmitter, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.emitter = emitter
        self.episode_count = 0
        self.successes = 0
        self.collisions = 0
        self.recent_rewards: deque[float] = deque(maxlen=100)

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, done in enumerate(dones):
            if not done:
                continue

            info = infos[idx] if idx < len(infos) else {}
            episode_info = info.get("episode", {})
            reward = float(episode_info.get("r", 0.0))
            steps = int(episode_info.get("l", 0))
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            collision_type = info.get("collision_type")
            collision_point = info.get("collision_point")
            termination_reason = info.get("termination_reason")

            self.episode_count += 1
            self.successes += int(success)
            self.collisions += int(collision)
            self.recent_rewards.append(reward)

            moving_avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            payload = {
                "episode": self.episode_count,
                "reward": reward,
                "steps": steps,
                "success": success,
                "collision": collision,
                "collision_type": collision_type,
                "collision_point": collision_point,
                "termination_reason": termination_reason,
                "success_rate": self.successes / self.episode_count,
                "collision_rate": self.collisions / self.episode_count,
                "avg_reward_100": moving_avg_reward,
                "trajectory": info.get("trajectory", []),
                "goal": info.get("goal", []),
                "obstacles": info.get("obstacles", []),
                "tape_zones": info.get("tape_zones", info.get("obstacles", [])),
                "grid_size": info.get("grid_size", 20),
            }
            self.emitter.send_metric(payload)

        return True
