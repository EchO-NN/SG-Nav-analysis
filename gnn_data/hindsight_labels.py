from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from gnn_data.raw_schema import make_soft_frontier_label


def as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def frontier_centers_rc(sample: dict) -> np.ndarray:
    frontier = sample.get("frontier", {})
    centers = frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", None))
    if centers is None:
        graph = sample.get("graph", {})
        if isinstance(graph, dict):
            centers = graph.get("frontier_centers_rc")
    if centers is None:
        return np.zeros((0, 2), dtype=np.float32)
    return as_numpy(centers).astype(np.float32).reshape(-1, 2)


def path_costs(sample: dict) -> np.ndarray:
    frontier = sample.get("frontier", {})
    distances = frontier.get("distances_valid", frontier.get("mean_path_dist", None))
    if distances is not None:
        return as_numpy(distances).astype(np.float32).reshape(-1)
    inv = frontier.get("distance_inverse_valid", frontier.get("distance_inverse", None))
    if inv is not None:
        return -as_numpy(inv).astype(np.float32).reshape(-1)
    return np.zeros((len(frontier_centers_rc(sample)),), dtype=np.float32)


def _first_rc_from_dict(payload: dict, keys: Iterable[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in payload and payload[key] is not None:
            arr = as_numpy(payload[key]).astype(np.float32).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                return arr[:2]
    return None


def goal_rc_from_sample(sample: dict) -> Optional[np.ndarray]:
    goal = sample.get("goal", {})
    metadata = sample.get("metadata", {})
    agent = sample.get("agent", {})
    for payload in [goal, metadata, agent]:
        out = _first_rc_from_dict(
            payload,
            [
                "goal_rc",
                "goal_center_rc",
                "true_goal_rc",
                "found_goal_rc",
                "target_rc",
                "position_rc",
                "center_rc",
            ],
        )
        if out is not None:
            return out
    return None


def _map_2d(value):
    if value is None:
        return None
    arr = as_numpy(value).astype(np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _final_free_map(sample: dict):
    maps = sample.get("maps", {})
    for key in ["final_free_map", "free_map", "full_map"]:
        out = _map_2d(maps.get(key))
        if out is not None and out.size:
            return out
    return None


def _fmm_distance_to_goal(sample: dict, goal_rc: np.ndarray):
    free_map = _final_free_map(sample)
    if free_map is None:
        return None
    try:
        from utils.utils_fmm.fmm_planner import FMMPlanner
    except Exception:
        return None

    traversible = (free_map > 0.5).astype(np.float32)
    if traversible.sum() == 0:
        return None
    goal_rc = np.asarray(goal_rc, dtype=np.float32).reshape(-1)[:2]
    r = int(np.clip(round(float(goal_rc[0])), 0, traversible.shape[0] - 1))
    c = int(np.clip(round(float(goal_rc[1])), 0, traversible.shape[1] - 1))
    try:
        planner = FMMPlanner(traversible, None)
        planner.set_goal([r, c])
        return np.asarray(planner.fmm_dist, dtype=np.float32)
    except Exception:
        return None


def frontier_to_goal_cost(
    sample: dict,
    goal_rc,
    lambda_goal: float = 1.0,
    prefer_fmm: bool = True,
    allow_approx_fallback: bool = True,
) -> np.ndarray:
    centers = frontier_centers_rc(sample)
    path = path_costs(sample)
    if len(path) != len(centers):
        path = np.resize(path, (len(centers),)).astype(np.float32)
    if len(centers) == 0:
        return np.zeros((0,), dtype=np.float32)

    goal_rc = np.asarray(goal_rc, dtype=np.float32).reshape(-1)[:2]
    map_resolution_cm = float(sample.get("metadata", {}).get("map_resolution_cm", 5.0))
    goal_dist = None
    if prefer_fmm:
        dist_map = _fmm_distance_to_goal(sample, goal_rc)
        if dist_map is not None:
            vals = []
            for center in centers:
                r = int(np.clip(round(float(center[0])), 0, dist_map.shape[0] - 1))
                c = int(np.clip(round(float(center[1])), 0, dist_map.shape[1] - 1))
                vals.append(float(dist_map[r, c]) * map_resolution_cm / 100.0)
            goal_dist = np.asarray(vals, dtype=np.float32)
    if goal_dist is None and not allow_approx_fallback:
        raise ValueError("skipped_missing_final_map_fmm")
    if goal_dist is None:
        goal_dist = np.linalg.norm(centers - goal_rc.reshape(1, 2), axis=1) * map_resolution_cm / 100.0
    return path.astype(np.float32) + float(lambda_goal) * goal_dist.astype(np.float32)


def make_label_payload(costs, tau: float, label_type: str, extra: Optional[Dict] = None) -> Dict:
    costs = np.asarray(costs, dtype=np.float32).reshape(-1)
    y = make_soft_frontier_label(costs, tau=tau)
    if len(costs) == 0:
        best_idx = -1
    else:
        best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
        if not np.isfinite(costs[best_idx]):
            best_idx = int(np.argmax(y))
    payload = {
        "frontier_y_soft": torch.tensor(y, dtype=torch.float32),
        "frontier_cost": torch.tensor(costs, dtype=torch.float32),
        "frontier_best_idx": int(best_idx),
        "label_type": label_type,
    }
    if extra:
        payload.update(extra)
    return payload


def label_with_goal_rc(
    sample: dict,
    goal_rc,
    tau: float,
    lambda_goal: float,
    label_type: str,
    label_source: str = "sample_goal_rc",
    allow_approx_fallback: bool = True,
) -> dict:
    out = copy.deepcopy(sample)
    costs = frontier_to_goal_cost(
        out,
        goal_rc=goal_rc,
        lambda_goal=lambda_goal,
        prefer_fmm=True,
        allow_approx_fallback=allow_approx_fallback,
    )
    out["labels"] = make_label_payload(
        costs,
        tau=tau,
        label_type=label_type,
        extra={
            "hindsight_goal_rc": torch.tensor(goal_rc, dtype=torch.float32),
            "label_source": label_source,
        },
    )
    return out


def _first_numeric(obj: dict, keys: Sequence[str], default=0):
    for key in keys:
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except Exception:
                pass
    return default


def _valid_category(category: str, exclude_unknown: bool) -> bool:
    if not category:
        return False
    if not exclude_unknown:
        return True
    bad = {"object", "unknown", "misc", "background", "none", "null", "__unknown__"}
    return category.strip().lower() not in bad


def pseudo_goal_objects(
    sample: dict,
    min_confidence: float = 0.5,
    min_observed_count: int = 1,
    min_lifetime_steps: int = 0,
    max_per_category: int = 0,
    exclude_unknown: bool = False,
    exclude_rejected_candidates: bool = False,
    balance_categories: bool = False,
    require_stable: bool = False,
) -> List[dict]:
    out = []
    per_category = defaultdict(int)
    for obj in sample.get("scenegraph", {}).get("objects", []):
        center = obj.get("center_rc")
        if center is None:
            continue
        category = str(obj.get("category", obj.get("caption", "object")))
        if not _valid_category(category, exclude_unknown=exclude_unknown):
            continue
        if exclude_rejected_candidates and bool(obj.get("rejected_candidate", False)):
            continue
        if require_stable and "stable" in obj and not bool(obj.get("stable", False)):
            continue
        confidence = float(obj.get("confidence", 1.0))
        if confidence < min_confidence:
            continue
        observed_count = int(_first_numeric(obj, ["observed_count", "num_detections", "count"], default=1))
        if observed_count < int(min_observed_count):
            continue
        first_seen = _first_numeric(obj, ["first_seen_step"], default=obj.get("last_seen_step", 0) or 0)
        last_seen = _first_numeric(obj, ["last_seen_step"], default=first_seen)
        lifetime = int(max(0, last_seen - first_seen))
        if lifetime < int(min_lifetime_steps):
            continue
        if max_per_category and per_category[category] >= int(max_per_category):
            continue
        center = as_numpy(center).astype(np.float32).reshape(-1)
        if center.size < 2 or not np.isfinite(center[:2]).all():
            continue
        map_size = int(sample.get("metadata", {}).get("map_size", 0))
        if map_size > 0 and not (0 <= center[0] < map_size and 0 <= center[1] < map_size):
            continue
        per_category[category] += 1
        out.append(
            {
                "goal_text": category,
                "goal_rc": center[:2].astype(np.float32),
                "source_node_id": obj.get("node_id"),
                "confidence": confidence,
                "observed_count": observed_count,
                "first_seen_step": int(first_seen),
                "last_seen_step": int(last_seen),
                "lifetime_steps": int(lifetime),
                "rejected_candidate": bool(obj.get("rejected_candidate", False)),
            }
        )
    if balance_categories:
        out = sorted(out, key=lambda item: (item["goal_text"], -item["confidence"], -item["observed_count"]))
    return out
