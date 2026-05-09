from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from gnn_data.raw_schema import RAW_VERSION, as_float_array, as_numpy, make_soft_frontier_label, to_cpu
from gnn_nav.graph_utils import pose_to_map_rc
from gnn_nav.sparse_graph_builder import (
    DEFAULT_ROOM_NAMES,
    safe_get_caption,
    safe_get_center_rc,
    safe_get_confidence,
    safe_get_observed_count,
)


def _safe_scalar(value, default=None):
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value).reshape(-1)
        if arr.size:
            return arr[0].item()
    except Exception:
        pass
    return default


def _episode(agent):
    try:
        return agent.simulator._env.current_episode
    except Exception:
        return None


def _sim(agent):
    for path in [
        ("simulator", "_env", "sim"),
        ("simulator", "sim"),
        ("_env", "sim"),
    ]:
        obj = agent
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    return None


def extract_goal(agent) -> Dict[str, Any]:
    episode = _episode(agent)
    raw_goal = getattr(episode, "object_category", getattr(agent, "obj_goal", ""))
    goal_positions = []
    goals = getattr(episode, "goals", []) if episode is not None else []
    for goal in goals:
        pos = getattr(goal, "position", None)
        if pos is not None:
            goal_positions.append(np.asarray(pos, dtype=np.float32))
            continue
        for view in getattr(goal, "view_points", []) or []:
            state = getattr(view, "agent_state", None)
            pos = getattr(state, "position", None)
            if pos is not None:
                goal_positions.append(np.asarray(pos, dtype=np.float32))
    if len(goal_positions) == 0:
        positions = np.zeros((0, 3), dtype=np.float32)
    else:
        positions = np.stack(goal_positions, axis=0).astype(np.float32)

    return {
        "object_category_raw": raw_goal,
        "object_category_sg": getattr(agent, "obj_goal_sg", raw_goal),
        "episode_goal_category": raw_goal,
        "goal_positions_world": positions,
    }


def extract_agent(agent) -> Dict[str, Any]:
    gps = None
    compass = None
    observations = getattr(agent.scenegraph, "observations", None) if hasattr(agent, "scenegraph") else None
    if isinstance(observations, dict):
        gps = observations.get("gps")
        compass = observations.get("compass")
    return {
        "full_pose": to_cpu(getattr(agent, "full_pose", None)),
        "gps": to_cpu(gps),
        "compass": to_cpu(compass),
        "found_goal": bool(getattr(agent, "found_goal", False)),
        "found_possible_goal": bool(getattr(agent, "found_possible_goal", False)),
        "using_random_goal": bool(getattr(agent, "using_random_goal", False)),
    }


def _iter_scenegraph_nodes(scenegraph) -> Iterable[Any]:
    if scenegraph is None:
        return []
    if hasattr(scenegraph, "get_nodes"):
        try:
            return scenegraph.get_nodes()
        except Exception:
            pass
    return getattr(scenegraph, "nodes", []) or []


def extract_scenegraph_summary(agent, save_edges: bool = False) -> Dict[str, Any]:
    scenegraph = getattr(agent, "scenegraph", None)
    map_size = int(getattr(agent, "map_size", 800))
    map_resolution_cm = float(getattr(agent, "map_resolution", 5.0))

    objects = []
    for node_id, node in enumerate(_iter_scenegraph_nodes(scenegraph)):
        center_rc = safe_get_center_rc(node, map_size=map_size, map_resolution_cm=map_resolution_cm)
        if center_rc is None:
            continue
        room_node = getattr(node, "room_node", None)
        room_id = None
        if room_node is not None and hasattr(scenegraph, "room_nodes"):
            try:
                room_id = list(scenegraph.room_nodes).index(room_node)
            except ValueError:
                room_id = None
        obj = getattr(node, "object", None)
        center_world = None
        if isinstance(obj, dict):
            for key in ["center_world", "world_center", "position"]:
                if key in obj and obj[key] is not None:
                    center_world = as_numpy(obj[key])
                    break
        objects.append(
            {
                "node_id": int(node_id),
                "category": safe_get_caption(node),
                "caption": safe_get_caption(node),
                "center_rc": np.asarray(center_rc, dtype=np.float32).reshape(-1)[:2],
                "center_world": None if center_world is None else np.asarray(center_world, dtype=np.float32),
                "confidence": float(safe_get_confidence(node)),
                "observed_count": int(max(1, round(float(safe_get_observed_count(node))))),
                "last_seen_step": _safe_scalar(getattr(node, "last_seen_step", None), None),
                "is_goal_node": bool(getattr(node, "is_goal_node", False)),
                "room_id": room_id,
            }
        )

    rooms = []
    room_nodes = getattr(scenegraph, "room_nodes", None)
    if room_nodes is None:
        room_nodes = [type("Room", (), {"caption": name}) for name in DEFAULT_ROOM_NAMES]
    for room_id, room_node in enumerate(room_nodes):
        rooms.append({"room_id": int(room_id), "room_name": str(getattr(room_node, "caption", room_node))})

    edges = []
    if save_edges and scenegraph is not None and hasattr(scenegraph, "get_edges"):
        try:
            for edge in scenegraph.get_edges():
                relation = getattr(edge, "relation", None)
                if not relation:
                    continue
                edges.append(
                    {
                        "node1": safe_get_caption(getattr(edge, "node1", None)),
                        "node2": safe_get_caption(getattr(edge, "node2", None)),
                        "relation": str(relation),
                    }
                )
        except Exception:
            edges = []

    return {
        "objects": objects,
        "rooms": rooms,
        "groups": [],
        "edges": edges,
        "text_node": str(getattr(agent, "text_node", "")),
        "text_edge": str(getattr(agent, "text_edge", "")),
    }


def extract_metadata(agent, data_tag: str = "sgnav_teacher") -> Dict[str, Any]:
    episode = _episode(agent)
    return {
        "dataset": "mp3d",
        "split": "unknown",
        "scene_id": getattr(episode, "scene_id", "unknown_scene"),
        "episode_id": getattr(episode, "episode_id", "unknown_episode"),
        "step_id": int(getattr(agent, "total_steps", 0)),
        "navigate_steps": int(getattr(agent, "navigate_steps", 0)),
        "total_steps": int(getattr(agent, "total_steps", 0)),
        "data_tag": str(data_tag),
        "map_size": int(getattr(agent, "map_size", 0)),
        "map_size_cm": int(getattr(agent, "map_size_cm", 0)),
        "map_resolution_cm": float(getattr(agent, "map_resolution", 5.0)),
        "timestamp": float(time.time()),
    }


def map_rc_to_world_best_effort(agent, center_rc) -> Optional[np.ndarray]:
    sim = _sim(agent)
    if sim is None or not hasattr(sim, "get_agent_state"):
        return None
    try:
        state = sim.get_agent_state()
        agent_world = np.asarray(state.position, dtype=np.float32)
    except Exception:
        return None

    map_size = int(getattr(agent, "map_size", 800))
    map_resolution_cm = float(getattr(agent, "map_resolution", 5.0))
    agent_rc = pose_to_map_rc(getattr(agent, "full_pose", [map_size / 2, map_size / 2, 0]), map_size, map_resolution_cm)
    center_rc = np.asarray(center_rc, dtype=np.float32).reshape(-1)[:2]
    delta_rc = center_rc - agent_rc
    delta_m = delta_rc * (map_resolution_cm / 100.0)
    world = agent_world.copy()
    world[0] += float(delta_m[1])
    world[2] += float(delta_m[0])
    return world.astype(np.float32)


def _geodesic_distance(sim, start, goals) -> float:
    if sim is None or goals is None or len(goals) == 0:
        return float("inf")
    start = np.asarray(start, dtype=np.float32)
    best = float("inf")
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is not None and hasattr(pathfinder, "snap_point"):
        try:
            start = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
        except Exception:
            pass
    for goal in np.asarray(goals, dtype=np.float32).reshape(-1, 3):
        try:
            if hasattr(sim, "geodesic_distance"):
                dist = sim.geodesic_distance(start, goal)
            elif pathfinder is not None and hasattr(pathfinder, "geodesic_distance"):
                dist = pathfinder.geodesic_distance(start, goal)
            else:
                dist = np.linalg.norm(start - goal)
            best = min(best, float(dist))
        except Exception:
            continue
    return best


def compute_online_oracle(agent, frontier_locations_valid_rc, distances_valid, tau: float = 2.0, lambda_goal: float = 1.0):
    goal = extract_goal(agent)
    goal_positions = np.asarray(goal["goal_positions_world"], dtype=np.float32)
    frontier_rc = np.asarray(frontier_locations_valid_rc, dtype=np.float32).reshape(-1, 2)
    path_cost = np.asarray(distances_valid, dtype=np.float32).reshape(-1)
    if len(frontier_rc) == 0:
        return {
            "frontier_cost": np.zeros((0,), dtype=np.float32),
            "frontier_y_soft": np.zeros((0,), dtype=np.float32),
            "frontier_best_idx": -1,
            "frontier_goal_geodesic": np.zeros((0,), dtype=np.float32),
            "frontier_world_positions": np.zeros((0, 3), dtype=np.float32),
            "label_type": "online_geodesic_empty",
        }

    sim = _sim(agent)
    world_positions = []
    geo = []
    for center in frontier_rc:
        world = map_rc_to_world_best_effort(agent, center)
        if world is None:
            world = np.full((3,), np.nan, dtype=np.float32)
            dist = float("inf")
        else:
            dist = _geodesic_distance(sim, world, goal_positions)
        world_positions.append(world)
        geo.append(dist)
    geo = np.asarray(geo, dtype=np.float32)
    costs = path_cost + float(lambda_goal) * geo
    y_soft = make_soft_frontier_label(costs, tau=tau)
    if np.isfinite(costs).any():
        best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
    else:
        best_idx = int(np.argmax(y_soft)) if len(y_soft) else -1
    return {
        "frontier_cost": costs.astype(np.float32),
        "frontier_y_soft": y_soft.astype(np.float32),
        "frontier_best_idx": best_idx,
        "frontier_goal_geodesic": geo.astype(np.float32),
        "frontier_world_positions": np.stack(world_positions, axis=0).astype(np.float32),
        "label_type": "online_geodesic_unvalidated",
        "coordinate_warning": "Best-effort SG-Nav map-to-world transform; validate with visualization before training.",
    }


def build_raw_sgnav_step_sample(
    agent,
    frontier_map,
    fmm_dist,
    frontier_locations_all_rc,
    frontier_locations_valid_rc,
    valid_indices_in_all,
    distances_valid,
    distance_inverse_valid,
    scenegraph_scores,
    distance_bias,
    total_scores,
    selected_valid_idx,
    selected_all_idx,
    selected_goal_rc,
    data_tag: str = "sgnav_teacher",
    save_maps: bool = False,
    save_scenegraph_edges: bool = False,
    compute_oracle_online: bool = False,
) -> Dict[str, Any]:
    frontier_locations_all_rc = as_float_array(frontier_locations_all_rc).reshape(-1, 2)
    frontier_locations_valid_rc = as_float_array(frontier_locations_valid_rc).reshape(-1, 2)
    distances_valid = as_float_array(distances_valid).reshape(-1)
    distance_inverse_valid = as_float_array(distance_inverse_valid).reshape(-1)
    scenegraph_scores = as_float_array(scenegraph_scores).reshape(-1)
    distance_bias = as_float_array(distance_bias).reshape(-1)
    total_scores = as_float_array(total_scores).reshape(-1)

    sample = {
        "version": RAW_VERSION,
        "metadata": extract_metadata(agent, data_tag=data_tag),
        "goal": extract_goal(agent),
        "agent": extract_agent(agent),
        "frontier": {
            "frontier_locations_all_rc": frontier_locations_all_rc.astype(np.float32),
            "frontier_locations_valid_rc": frontier_locations_valid_rc.astype(np.float32),
            "valid_indices_in_all": np.asarray(valid_indices_in_all, dtype=np.int64).reshape(-1),
            "distances_valid": distances_valid.astype(np.float32),
            "distance_inverse_valid": distance_inverse_valid.astype(np.float32),
            "selected_valid_idx": int(selected_valid_idx),
            "selected_all_idx": int(selected_all_idx),
            "selected_goal_rc": as_float_array(selected_goal_rc).reshape(-1)[:2].astype(np.float32),
            "frontier_map": as_numpy(frontier_map).astype(bool),
            "fmm_dist": as_float_array(fmm_dist).astype(np.float32),
        },
        "teacher": {
            "scenegraph_scores": scenegraph_scores.astype(np.float32),
            "distance_bias": distance_bias.astype(np.float32),
            "total_scores": total_scores.astype(np.float32),
            "selected_valid_idx": int(selected_valid_idx),
        },
        "scenegraph": extract_scenegraph_summary(agent, save_edges=save_scenegraph_edges),
        "maps": {},
        "oracle": {},
        "debug": {
            "score_shapes": {
                "frontier_locations_valid_rc": tuple(frontier_locations_valid_rc.shape),
                "scenegraph_scores": tuple(scenegraph_scores.shape),
                "total_scores": tuple(total_scores.shape),
            }
        },
    }

    if save_maps:
        sample["maps"] = {
            "full_map": as_float_array(getattr(agent, "full_map", None)).astype(np.float16),
            "free_map": as_float_array(getattr(agent, "fbe_free_map", None)).astype(np.float16),
            "room_map": as_float_array(getattr(agent, "room_map", None)).astype(np.float16),
        }

    if compute_oracle_online:
        try:
            sample["oracle"] = compute_online_oracle(agent, frontier_locations_valid_rc, distances_valid)
        except Exception as exc:
            sample["oracle"] = {
                "label_type": "online_geodesic_failed",
                "error": str(exc),
            }

    return sample
