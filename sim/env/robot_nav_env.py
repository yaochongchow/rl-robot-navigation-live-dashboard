from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    collision: bool


class RobotNavEnv(gym.Env[np.ndarray, int]):
    """Grid world navigation environment with obstacles and shaped rewards."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 20,
        max_steps: int = 120,
        obstacle_count: int = 22,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.rng = np.random.default_rng(seed)

        # Observation: agent(x,y), goal(x,y), normalized distance.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()

        self.step_count = 0
        self.episode_path: list[list[int]] = []
        self.prev_distance = 0.0

    def _sample_free_cell(self, forbidden: set[tuple[int, int]]) -> np.ndarray:
        while True:
            pos = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)
            key = (int(pos[0]), int(pos[1]))
            if key not in forbidden:
                return pos

    def _generate_obstacles(self, forbidden: set[tuple[int, int]]) -> set[tuple[int, int]]:
        obstacles: set[tuple[int, int]] = set()
        while len(obstacles) < self.obstacle_count:
            pos = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)
            key = (int(pos[0]), int(pos[1]))
            if key in forbidden:
                continue
            obstacles.add(key)
        return obstacles

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_pos - self.agent_pos))

    def _get_obs(self) -> np.ndarray:
        max_distance = np.sqrt(2.0 * (self.grid_size - 1) ** 2)
        return np.array(
            [
                self.agent_pos[0] / (self.grid_size - 1),
                self.agent_pos[1] / (self.grid_size - 1),
                self.goal_pos[0] / (self.grid_size - 1),
                self.goal_pos[1] / (self.grid_size - 1),
                self._distance_to_goal() / max_distance,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        forbidden: set[tuple[int, int]] = set()

        self.agent_pos = self._sample_free_cell(forbidden)
        forbidden.add((int(self.agent_pos[0]), int(self.agent_pos[1])))

        self.goal_pos = self._sample_free_cell(forbidden)
        forbidden.add((int(self.goal_pos[0]), int(self.goal_pos[1])))

        self.obstacles = self._generate_obstacles(forbidden)
        self.episode_path = [[int(self.agent_pos[0]), int(self.agent_pos[1])]]
        self.prev_distance = self._distance_to_goal()

        return self._get_obs(), {}

    def _apply_action(self, action: int) -> np.ndarray:
        move_map = {
            0: np.array([0, -1], dtype=np.int32),  # up
            1: np.array([0, 1], dtype=np.int32),   # down
            2: np.array([-1, 0], dtype=np.int32),  # left
            3: np.array([1, 0], dtype=np.int32),   # right
        }
        return self.agent_pos + move_map[action]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        proposed = self._apply_action(action)

        out_of_bounds = bool(
            proposed[0] < 0
            or proposed[0] >= self.grid_size
            or proposed[1] < 0
            or proposed[1] >= self.grid_size
        )

        collision = False
        success = False
        collision_type: str | None = None
        collision_point: list[int] | None = None
        reward = -1.0

        if out_of_bounds:
            collision = True
            collision_type = "wall"
            collision_point = [
                int(np.clip(proposed[0], 0, self.grid_size - 1)),
                int(np.clip(proposed[1], 0, self.grid_size - 1)),
            ]
            reward = -100.0
        else:
            proposed_key = (int(proposed[0]), int(proposed[1]))
            if proposed_key in self.obstacles:
                collision = True
                collision_type = "tape"
                collision_point = [int(proposed[0]), int(proposed[1])]
                reward = -100.0
            else:
                self.agent_pos = proposed
                self.episode_path.append([int(self.agent_pos[0]), int(self.agent_pos[1])])

                current_distance = self._distance_to_goal()
                distance_delta = self.prev_distance - current_distance
                reward += distance_delta * 3.0
                self.prev_distance = current_distance

                if np.array_equal(self.agent_pos, self.goal_pos):
                    success = True
                    reward = 100.0

        terminated = success or collision
        truncated = self.step_count >= self.max_steps and not terminated

        info: dict[str, Any] = {}
        if terminated or truncated:
            if success:
                termination_reason = "goal_reached"
            elif collision_type == "wall":
                termination_reason = "wall_collision"
            elif collision_type == "tape":
                termination_reason = "tape_collision"
            else:
                termination_reason = "max_steps"

            info = {
                "success": success,
                "collision": collision,
                "collision_type": collision_type,
                "collision_point": collision_point,
                "termination_reason": termination_reason,
                "steps": self.step_count,
                "trajectory": self.episode_path,
                "goal": [int(self.goal_pos[0]), int(self.goal_pos[1])],
                "obstacles": [[x, y] for x, y in sorted(self.obstacles)],
                "tape_zones": [[x, y] for x, y in sorted(self.obstacles)],
                "grid_size": self.grid_size,
            }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U1")
        for x, y in self.obstacles:
            grid[y, x] = "#"
        grid[self.goal_pos[1], self.goal_pos[0]] = "G"
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self) -> None:
        return
