import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch


class FBETraceLogger:
    def __init__(self, enabled: bool = False, log_dir: str = "data/debug_fbe"):
        self.enabled = bool(enabled)
        self.log_dir = log_dir
        self.episode_idx = None
        self.trace_path = os.path.join(log_dir, "fbe_trace.jsonl")
        self.random_path = os.path.join(log_dir, "random_fallbacks.jsonl")
        self.overlay_dir = os.path.join(log_dir, "overlays")
        if self.enabled:
            os.makedirs(self.overlay_dir, exist_ok=True)

    def start_episode(self, episode_idx: int):
        self.episode_idx = int(episode_idx)

    def log_decision(
        self,
        sample: Dict[str, Any],
        *,
        occupancy_map=None,
        free_map=None,
        agent_rc=None,
        selected_frontier_rc=None,
        candidate_goal_rc=None,
        frontier_locations_valid_rc=None,
        frontier_locations_all_rc=None,
    ):
        if not self.enabled:
            return
        os.makedirs(self.overlay_dir, exist_ok=True)
        item = self._to_jsonable(sample)
        item.setdefault("episode_idx", self.episode_idx)
        overlay_path = self._write_overlay(
            item,
            occupancy_map=occupancy_map,
            free_map=free_map,
            agent_rc=agent_rc,
            selected_frontier_rc=selected_frontier_rc,
            candidate_goal_rc=candidate_goal_rc,
            frontier_locations_valid_rc=frontier_locations_valid_rc,
            frontier_locations_all_rc=frontier_locations_all_rc,
        )
        if overlay_path is not None:
            item["overlay_png"] = overlay_path
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def log_random_fallback(self, sample: Dict[str, Any]):
        if not self.enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        item = self._to_jsonable(sample)
        item.setdefault("episode_idx", self.episode_idx)
        with open(self.random_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _write_overlay(
        self,
        sample,
        *,
        occupancy_map=None,
        free_map=None,
        agent_rc=None,
        selected_frontier_rc=None,
        candidate_goal_rc=None,
        frontier_locations_valid_rc=None,
        frontier_locations_all_rc=None,
    ) -> Optional[str]:
        if occupancy_map is None and free_map is None:
            return None
        base = occupancy_map if occupancy_map is not None else free_map
        base = self._as_numpy(base)
        if base is None or base.ndim != 2 or base.size == 0:
            return None

        image = np.full((*base.shape, 3), 255, dtype=np.uint8)
        if free_map is not None:
            free = self._as_numpy(free_map)
            if free is not None and free.shape == base.shape:
                image[free > 0.5] = (225, 225, 225)
        image[base > 0.5] = (90, 90, 90)

        self._draw_points(image, frontier_locations_all_rc, (180, 210, 255), radius=1)
        self._draw_points(image, frontier_locations_valid_rc, (255, 170, 70), radius=1)
        self._draw_point(image, selected_frontier_rc, (40, 40, 230), radius=4)
        self._draw_point(image, candidate_goal_rc, (210, 40, 210), radius=4)
        self._draw_point(image, agent_rc, (40, 180, 40), radius=4)

        episode = sample.get("episode_idx", self.episode_idx)
        step = sample.get("step", "unknown")
        filename = f"episode_{episode}_step_{step}.png"
        path = os.path.join(self.overlay_dir, filename)
        os.makedirs(self.overlay_dir, exist_ok=True)
        cv2.imwrite(path, image)
        return path

    def _draw_points(self, image, points, color, radius=1):
        if points is None:
            return
        points = self._as_numpy(points)
        if points is None or points.size == 0:
            return
        points = points.reshape(-1, 2)
        for row, col in points:
            self._draw_point(image, (row, col), color, radius=radius)

    def _draw_point(self, image, point, color, radius=3):
        if point is None:
            return
        point = self._as_numpy(point)
        if point is None or point.size < 2:
            return
        row = int(round(float(point[0])))
        col = int(round(float(point[1])))
        if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
            cv2.circle(image, (col, row), radius, color, -1)

    def _as_numpy(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _to_jsonable(self, value):
        if torch.is_tensor(value):
            return self._to_jsonable(value.detach().cpu().numpy())
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value
