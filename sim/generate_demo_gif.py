from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate robotics safety demo GIF from telemetry history")
    parser.add_argument("--history-url", default="http://localhost:4100/history")
    parser.add_argument("--history-file", default="")
    parser.add_argument("--output", default="../dashboard/public/demo_robot_safety_run.gif")
    parser.add_argument("--min-trajectory", type=int, default=12)
    return parser.parse_args()


def load_history(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.history_file:
        with open(args.history_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        response = requests.get(args.history_url, timeout=5)
        response.raise_for_status()
        payload = response.json()
    return payload.get("metrics", [])


def pick_episode(metrics: list[dict[str, Any]], min_trajectory: int) -> dict[str, Any]:
    for item in reversed(metrics):
        if item.get("success") and len(item.get("trajectory", [])) >= min_trajectory:
            return item
    if metrics:
        return max(metrics, key=lambda x: len(x.get("trajectory", [])))
    raise RuntimeError("No metrics available to generate demo gif")


def draw_tape_cell(draw: ImageDraw.ImageDraw, x: float, y: float, cell: float) -> None:
    draw.rectangle([x, y, x + cell, y + cell], fill="#f2c84f")
    step = max(4, int(cell * 0.28))
    for off in range(-int(cell), int(cell * 2), step):
        draw.line([(x + off, y), (x + off + cell, y + cell)], fill="#1f1f1f", width=max(1, int(cell * 0.08)))


def draw_robot(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, heading: float) -> None:
    import math

    left = heading + (math.pi * 3 / 4)
    right = heading - (math.pi * 3 / 4)
    p1 = (cx + math.cos(heading) * r, cy + math.sin(heading) * r)
    p2 = (cx + math.cos(left) * r, cy + math.sin(left) * r)
    p3 = (cx + math.cos(right) * r, cy + math.sin(right) * r)
    draw.polygon([p1, p2, p3], fill="#1c3d5a")


def build_frames(episode: dict[str, Any]) -> tuple[list[Image.Image], list[int]]:
    size = int(episode.get("grid_size", 20))
    path: list[list[int]] = episode.get("trajectory", [])
    tape = episode.get("tape_zones", episode.get("obstacles", []))
    goal = episode.get("goal", [])
    reason = episode.get("termination_reason", "unknown")
    ep_num = episode.get("episode", "?")

    width = 760
    height = 820
    top = 90
    margin = 40
    grid_px = width - margin * 2
    cell = grid_px / size

    base_font = ImageFont.load_default()

    frames: list[Image.Image] = []
    durations: list[int] = []

    for idx in range(1, max(2, len(path) + 1)):
        img = Image.new("RGB", (width, height), "#f5f0e4")
        draw = ImageDraw.Draw(img)

        title = f"Robot Safety Run Demo  |  Episode {ep_num}"
        subtitle = f"Outcome: {reason.replace('_', ' ')}"
        draw.text((margin, 24), title, fill="#1f2d24", font=base_font)
        draw.text((margin, 48), subtitle, fill="#5f6b62", font=base_font)

        gx0 = margin
        gy0 = top
        gx1 = margin + grid_px
        gy1 = top + grid_px

        draw.rectangle([gx0, gy0, gx1, gy1], fill="#fdf7e8")

        for i in range(size + 1):
            p = gx0 + i * cell
            draw.line([(p, gy0), (p, gy1)], fill="#c9bfa8", width=1)
            q = gy0 + i * cell
            draw.line([(gx0, q), (gx1, q)], fill="#c9bfa8", width=1)

        border = max(6, int(cell * 0.36))
        draw.rectangle([gx0, gy0, gx1, gy1], outline="#4d4d56", width=border)

        for tx, ty in tape:
            draw_tape_cell(draw, gx0 + tx * cell, gy0 + ty * cell, cell)

        if len(goal) == 2:
            x, y = goal
            draw.rectangle(
                [gx0 + x * cell, gy0 + y * cell, gx0 + (x + 1) * cell, gy0 + (y + 1) * cell],
                fill="#2c7a7b",
            )

        partial = path[:idx]
        if partial:
            points = [(gx0 + x * cell + cell / 2, gy0 + y * cell + cell / 2) for x, y in partial]
            if len(points) > 1:
                draw.line(points, fill="#2b73c5", width=max(2, int(cell * 0.18)))

            sx, sy = points[0]
            draw.ellipse([sx - cell * 0.18, sy - cell * 0.18, sx + cell * 0.18, sy + cell * 0.18], fill="#205e4c")

            ex, ey = points[-1]
            heading = -1.57
            if len(points) >= 2:
                import math

                px, py = points[-2]
                heading = math.atan2(ey - py, ex - px)
            draw_robot(draw, ex, ey, cell * 0.38, heading)

        if idx >= len(path) and episode.get("collision"):
            collision_point = episode.get("collision_point")
            if isinstance(collision_point, list) and len(collision_point) == 2:
                cx = gx0 + collision_point[0] * cell + cell / 2
                cy = gy0 + collision_point[1] * cell + cell / 2
                r = cell * 0.28
                draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill="#b23a48", width=max(2, int(cell * 0.1)))
                draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill="#b23a48", width=max(2, int(cell * 0.1)))

        legend_y = gy1 + 20
        draw.rectangle([margin, legend_y, margin + 14, legend_y + 14], fill="#4d4d56")
        draw.text((margin + 20, legend_y), "arena walls", fill="#5f6b62", font=base_font)
        draw.rectangle([margin + 130, legend_y, margin + 144, legend_y + 14], fill="#f2c84f")
        draw.text((margin + 150, legend_y), "floor tape hazard", fill="#5f6b62", font=base_font)
        draw.rectangle([margin + 320, legend_y, margin + 334, legend_y + 14], fill="#2c7a7b")
        draw.text((margin + 340, legend_y), "docking goal", fill="#5f6b62", font=base_font)

        frames.append(img)
        durations.append(90)

    if frames:
        frames.extend([frames[-1].copy(), frames[-1].copy()])
        durations.extend([700, 700])

    return frames, durations


def main() -> None:
    args = parse_args()
    metrics = load_history(args)
    episode = pick_episode(metrics, args.min_trajectory)
    frames, durations = build_frames(episode)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Saved demo gif to {output_path}")
    print(f"Episode {episode.get('episode')} | reason={episode.get('termination_reason')} | steps={episode.get('steps')}")


if __name__ == "__main__":
    main()
