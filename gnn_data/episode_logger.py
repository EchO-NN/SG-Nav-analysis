from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gnn_data.extract_sgnav_state import extract_scenegraph_summary
from gnn_data.raw_schema import atomic_torch_save, sanitize_filename, to_cpu


EPISODE_SUMMARY_VERSION = "episode_summary_v1"


def _np(value, dtype=np.float32):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _map_array(value, dtype=np.float16):
    arr = _np(value)
    if arr is None:
        return None
    return arr.astype(dtype)


def _goal_xy_to_rc(goal_xy):
    arr = _np(goal_xy, dtype=np.float32)
    if arr is None or arr.size < 2 or not np.isfinite(arr.reshape(-1)[:2]).all():
        return None
    arr = arr.reshape(-1)[:2]
    return np.asarray([arr[1], arr[0]], dtype=np.float32)


def _found_goal_payload(agent, metrics: Dict[str, Any]) -> Dict[str, Any]:
    found_goal_rc = None
    found_goal_world = None
    source = "none"
    confidence = 0.0
    found_step = None

    goal_gps = getattr(agent, "goal_gps", None)
    if bool(getattr(agent, "found_goal", False)) and goal_gps is not None:
        goal_xy = agent.goal_gps_to_map_xy(goal_gps) if hasattr(agent, "goal_gps_to_map_xy") else None
        found_goal_rc = _goal_xy_to_rc(goal_xy)
        found_goal_world = np.asarray([goal_gps[0], 0.0, goal_gps[1]], dtype=np.float32)
        source = "confirmed_candidate"
        confidence = float(getattr(agent, "found_goal_times", 1.0))
        found_step = int(getattr(agent, "total_steps", 0))

    if found_goal_rc is None and float(metrics.get("success", 0.0)) >= 1.0:
        # Simulation debug source only. We keep the source explicit so downstream
        # scripts can exclude it from autonomous-learning experiments.
        episode = getattr(getattr(agent, "simulator", None), "_env", None)
        episode = getattr(episode, "current_episode", None)
        goals = getattr(episode, "goals", []) if episode is not None else []
        for goal in goals:
            pos = getattr(goal, "position", None)
            if pos is not None:
                found_goal_world = np.asarray(pos, dtype=np.float32)
                source = "sim_gt_debug"
                confidence = 1.0
                break

    return {
        "goal_text": str(getattr(agent, "obj_goal_sg", getattr(agent, "obj_goal", ""))),
        "found_goal_rc": None if found_goal_rc is None else found_goal_rc.astype(np.float32),
        "found_goal_world": None if found_goal_world is None else found_goal_world.astype(np.float32),
        "found_step": found_step,
        "source": source,
        "confidence": float(confidence),
    }


def _trajectory(agent):
    out = []
    for step_id, pose in enumerate(getattr(agent, "history_pose", []) or []):
        out.append(
            {
                "step_id": int(step_id),
                "agent_pose": to_cpu(pose),
                "selected_frontier_rc": None,
                "action": None,
            }
        )
    return out


class EpisodeSummaryLogger:
    def __init__(self, log_dir: str, enabled: bool = False, data_tag: str = "episode"):
        self.log_dir = str(log_dir)
        self.enabled = bool(enabled)
        self.data_tag = str(data_tag)
        self.saved_count = 0
        if self.enabled:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def build_summary(self, agent, metrics: Dict[str, Any]) -> Dict[str, Any]:
        env = getattr(getattr(agent, "simulator", None), "_env", None)
        episode = getattr(env, "current_episode", None)
        metadata = {
            "dataset": "mp3d",
            "split": str(getattr(getattr(agent, "config", None), "DATASET", {}).get("SPLIT", "unknown"))
            if isinstance(getattr(getattr(agent, "config", None), "DATASET", None), dict)
            else str(getattr(getattr(getattr(agent, "config", None), "DATASET", None), "SPLIT", "unknown")),
            "scene_id": getattr(episode, "scene_id", "unknown_scene"),
            "episode_id": str(getattr(episode, "episode_id", "unknown_episode")),
            "goal_text": str(getattr(agent, "obj_goal_sg", getattr(agent, "obj_goal", ""))),
            "success": bool(float(metrics.get("success", 0.0)) >= 1.0),
            "spl": float(metrics.get("spl", 0.0)),
            "softspl": float(metrics.get("softspl", metrics.get("soft_spl", 0.0))),
            "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
            "num_steps": int(getattr(agent, "total_steps", 0)),
            "stop_reason": str(getattr(agent, "stop_reason", "")),
            "episode_idx": int(getattr(agent, "count_episodes", -1)),
        }

        scenegraph = extract_scenegraph_summary(
            agent,
            save_edges=bool(getattr(getattr(agent, "args", None), "gnn_save_scenegraph_edges", False)),
        )
        discovered_objects = []
        rejected = getattr(agent, "rejected_goal_candidates", []) or []
        for obj in scenegraph.get("objects", []):
            item = dict(obj)
            last_seen = item.get("last_seen_step")
            first_seen = item.get("first_seen_step", last_seen if last_seen is not None else 0)
            item["first_seen_step"] = int(first_seen if first_seen is not None else 0)
            item["last_seen_step"] = int(last_seen if last_seen is not None else item["first_seen_step"])
            item["stable"] = int(item.get("observed_count", 1)) >= 3
            item["rejected_candidate"] = False
            item["source_node_id"] = item.get("node_id")
            discovered_objects.append(item)

        return {
            "version": EPISODE_SUMMARY_VERSION,
            "metadata": metadata,
            "target_goal": _found_goal_payload(agent, metrics),
            "final_maps": {
                "full_map": _map_array(getattr(agent, "full_map", None)),
                "free_map": _map_array(getattr(agent, "fbe_free_map", None)),
                "room_map": _map_array(getattr(agent, "room_map", None)),
            },
            "trajectory": _trajectory(agent),
            "discovered_objects": discovered_objects,
            "fallback": {
                "num_fallback_calls": 0,
                "fallback_records": [],
            },
            "debug": {
                "rejected_goal_candidates": rejected,
                "reperception_history": getattr(agent, "reperception_history", [])[-20:],
            },
        }

    def save_episode(self, agent, metrics: Dict[str, Any]) -> Optional[str]:
        if not self.enabled:
            return None
        summary = self.build_summary(agent, metrics)
        metadata = summary.get("metadata", {})
        filename = "{}_{}_{}_{}.pt".format(
            sanitize_filename(self.data_tag),
            sanitize_filename(metadata.get("scene_id", "unknown_scene")),
            sanitize_filename(metadata.get("episode_id", "unknown_episode")),
            int(metadata.get("episode_idx", self.saved_count)),
        )
        path = os.path.join(self.log_dir, filename)
        atomic_torch_save(summary, path)
        self.saved_count += 1
        return path
