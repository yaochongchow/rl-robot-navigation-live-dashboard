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
        revisit_penalty: float = 0.25,
        near_hazard_penalty: float = 0.35,
        near_hazard_threshold: float = 0.20,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.revisit_penalty = revisit_penalty
        self.near_hazard_penalty = near_hazard_penalty
        self.near_hazard_threshold = near_hazard_threshold
        self.rng = np.random.default_rng(seed)

        # Observation:
        #  - agent(x,y), goal(x,y), normalized distance
        #  - 8 directional hazard ray distances
        #  - immediate hazard flags for [up, down, left, right]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()

        self.step_count = 0
        self.episode_path: list[list[int]] = []
        self.visit_counts: dict[tuple[int, int], int] = {}
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

    def _is_out_of_bounds(self, pos: np.ndarray) -> bool:
        return bool(
            pos[0] < 0
            or pos[0] >= self.grid_size
            or pos[1] < 0
            or pos[1] >= self.grid_size
        )

    def _is_hazard(self, pos: np.ndarray) -> bool:
        if self._is_out_of_bounds(pos):
            return True
        return (int(pos[0]), int(pos[1])) in self.obstacles

    def _ray_safe_distance(self, dx: int, dy: int) -> float:
        max_steps = self.grid_size - 1
        x = int(self.agent_pos[0])
        y = int(self.agent_pos[1])
        safe_steps = 0
        for step in range(1, max_steps + 1):
            nx = x + dx * step
            ny = y + dy * step
            pos = np.array([nx, ny], dtype=np.int32)
            if self._is_hazard(pos):
                break
            safe_steps += 1
        return safe_steps / max_steps

    def _immediate_hazard_flags(self) -> list[float]:
        moves = [
            np.array([0, -1], dtype=np.int32),  # up
            np.array([0, 1], dtype=np.int32),   # down
            np.array([-1, 0], dtype=np.int32),  # left
            np.array([1, 0], dtype=np.int32),   # right
        ]
        flags: list[float] = []
        for move in moves:
            proposed = self.agent_pos + move
            flags.append(1.0 if self._is_hazard(proposed) else 0.0)
        return flags

    def _cardinal_min_safe_distance(self) -> float:
        rays = [
            self._ray_safe_distance(0, -1),   # N
            self._ray_safe_distance(1, 0),    # E
            self._ray_safe_distance(0, 1),    # S
            self._ray_safe_distance(-1, 0),   # W
        ]
        return float(min(rays))

    def _get_obs(self) -> np.ndarray:
        max_distance = np.sqrt(2.0 * (self.grid_size - 1) ** 2)
        ray_dirs = [
            (0, -1),   # N
            (1, -1),   # NE
            (1, 0),    # E
            (1, 1),    # SE
            (0, 1),    # S
            (-1, 1),   # SW
            (-1, 0),   # W
            (-1, -1),  # NW
        ]
        rays = [self._ray_safe_distance(dx, dy) for dx, dy in ray_dirs]
        immediate_hazards = self._immediate_hazard_flags()

        return np.array(
            [
                self.agent_pos[0] / (self.grid_size - 1),
                self.agent_pos[1] / (self.grid_size - 1),
                self.goal_pos[0] / (self.grid_size - 1),
                self.goal_pos[1] / (self.grid_size - 1),
                self._distance_to_goal() / max_distance,
                *rays,
                *immediate_hazards,
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
        self.visit_counts = {(int(self.agent_pos[0]), int(self.agent_pos[1])): 1}
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

        out_of_bounds = self._is_out_of_bounds(proposed)

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

                if self.revisit_penalty > 0:
                    visits = self.visit_counts.get(proposed_key, 0)
                    if visits > 0:
                        reward -= self.revisit_penalty
                    self.visit_counts[proposed_key] = visits + 1

                if self.near_hazard_penalty > 0 and self.near_hazard_threshold > 0:
                    min_safe = self._cardinal_min_safe_distance()
                    if min_safe < self.near_hazard_threshold:
                        penalty_scale = (self.near_hazard_threshold - min_safe) / self.near_hazard_threshold
                        reward -= self.near_hazard_penalty * penalty_scale

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
