import math
from typing import Sequence, Tuple

import numpy as np
import torch


def empty_edge(attr_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, attr_dim), dtype=torch.float32)
    return edge_index, edge_attr


def make_edge(src, dst, attr, attr_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(src) == 0:
        return empty_edge(attr_dim)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.view(-1, attr_dim)
    return edge_index, edge_attr


def reverse_edge(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        return edge_index.clone(), edge_attr.clone()
    return edge_index.flip(0), edge_attr.clone()


def to_numpy_map(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x


def to_numpy_room_map(room_map):
    if hasattr(room_map, "detach"):
        room_map = room_map.detach().cpu().numpy()
    room_map = np.asarray(room_map)
    if room_map.ndim == 4:
        room_map = room_map[0]
    if room_map.ndim == 2:
        room_map = room_map[None, :, :]
    return room_map


def sample_room_prob(room_map, center_rc):
    room_np = to_numpy_room_map(room_map)
    room_count, height, width = room_np.shape
    r = int(np.clip(center_rc[0], 0, height - 1))
    c = int(np.clip(center_rc[1], 0, width - 1))
    prob = room_np[:, r, c].astype(np.float32)
    total = float(prob.sum())
    if total > 1e-6:
        prob = prob / total
    return prob.astype(np.float32)


def edge_type_to_str(edge_type):
    return "__".join(edge_type)


def pose_to_map_rc(agent_pose, map_size: int, map_resolution_cm: float) -> np.ndarray:
    """Convert SG-Nav full_pose [x_m, y_m, yaw_deg] to [row, col] map cells."""
    if hasattr(agent_pose, "detach"):
        pose = agent_pose.detach().cpu().numpy()
    else:
        pose = np.asarray(agent_pose)
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.size < 2:
        return np.array([map_size / 2.0, map_size / 2.0], dtype=np.float32)

    scale = 100.0 / float(map_resolution_cm)
    # SG_Nav.get_traversible flips the y coordinate before converting to row.
    row = map_size - pose[1] * scale
    col = pose[0] * scale
    if not np.isfinite(row) or not np.isfinite(col):
        row = map_size / 2.0
        col = map_size / 2.0
    return np.array(
        [
            float(np.clip(row, 0, map_size - 1)),
            float(np.clip(col, 0, map_size - 1)),
        ],
        dtype=np.float32,
    )


def normalize_rc(center_rc: Sequence[float], map_size: int) -> np.ndarray:
    center = np.asarray(center_rc, dtype=np.float32).reshape(-1)[:2]
    if center.size < 2:
        center = np.zeros((2,), dtype=np.float32)
    return np.clip(center / float(map_size), 0.0, 1.0).astype(np.float32)


def compute_unknown_score(full_map, free_map, center_rc, window: int = 20) -> float:
    full = to_numpy_map(full_map)
    free = to_numpy_map(free_map)
    r, c = int(center_rc[0]), int(center_rc[1])
    r0 = max(0, r - window)
    r1 = min(full.shape[0], r + window + 1)
    c0 = max(0, c - window)
    c1 = min(full.shape[1], c + window + 1)
    if r0 >= r1 or c0 >= c1:
        return 0.0
    local_full = full[r0:r1, c0:c1]
    local_free = free[r0:r1, c0:c1]
    unknown = (local_full < 0.5) & (local_free < 0.5)
    return float(unknown.mean())


def compute_heading_feature(agent_pose, frontier_center_rc, map_size, map_resolution_cm):
    agent_rc = pose_to_map_rc(agent_pose, map_size, map_resolution_cm)
    center = np.asarray(frontier_center_rc, dtype=np.float32)
    delta = center - agent_rc
    angle = math.atan2(float(delta[0]), float(delta[1]) + 1e-6)
    return float(math.sin(angle)), float(math.cos(angle))
