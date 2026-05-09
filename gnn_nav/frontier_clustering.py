from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from scipy import ndimage
except Exception:
    ndimage = None


@dataclass
class FrontierCluster:
    cluster_id: int
    center_rc: np.ndarray
    member_rcs: np.ndarray
    size: int
    min_path_dist: float
    mean_path_dist: float
    max_path_dist: float
    distance_inverse: float
    unknown_score: float = 0.0
    reachable: bool = True


def _label_connected_components(valid: np.ndarray):
    if ndimage is not None:
        return ndimage.label(valid)

    labels = np.zeros(valid.shape, dtype=np.int32)
    current = 0
    height, width = valid.shape
    for r in range(height):
        for c in range(width):
            if not valid[r, c] or labels[r, c] != 0:
                continue
            current += 1
            stack = [(r, c)]
            labels[r, c] = current
            while stack:
                rr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if nr < 0 or nr >= height or nc < 0 or nc >= width:
                            continue
                        if valid[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            stack.append((nr, nc))
    return labels, current


def cluster_frontiers(
    frontier_map: np.ndarray,
    fmm_dist: np.ndarray,
    min_path_dist: float = 1.6,
    max_frontiers: int = 32,
    min_cluster_size: int = 3,
) -> List[FrontierCluster]:
    frontier = np.asarray(frontier_map).astype(bool)
    dist = np.asarray(fmm_dist, dtype=np.float32)
    if frontier.shape != dist.shape:
        raise ValueError(f"frontier_map shape {frontier.shape} != fmm_dist shape {dist.shape}")

    valid = frontier.copy()
    valid &= np.isfinite(dist)
    valid &= dist >= float(min_path_dist)

    labeled, num = _label_connected_components(valid)
    clusters = []

    for cid in range(1, num + 1):
        rcs = np.argwhere(labeled == cid)
        if len(rcs) < min_cluster_size:
            continue

        dists = dist[rcs[:, 0], rcs[:, 1]]
        dists = dists[np.isfinite(dists)]
        if len(dists) == 0:
            continue

        center = np.round(rcs.mean(axis=0)).astype(np.int64)
        mean_dist = float(np.mean(dists))
        min_dist = float(np.min(dists))
        max_dist = float(np.max(dists))
        dist_inv = 1.0 - (np.clip(mean_dist, 1.6, 11.6) - 1.6) / 10.0

        clusters.append(
            FrontierCluster(
                cluster_id=len(clusters),
                center_rc=center,
                member_rcs=rcs.astype(np.int64),
                size=int(len(rcs)),
                min_path_dist=min_dist,
                mean_path_dist=mean_dist,
                max_path_dist=max_dist,
                distance_inverse=float(dist_inv),
            )
        )

    clusters.sort(key=lambda c: (-c.size, c.mean_path_dist))
    clusters = clusters[: int(max_frontiers)]
    for idx, cluster in enumerate(clusters):
        cluster.cluster_id = idx
    return clusters


if __name__ == "__main__":
    fmap = np.zeros((12, 12), dtype=bool)
    fmap[2:5, 3:5] = True
    fmap[8:10, 8:11] = True
    dmap = np.ones((12, 12), dtype=np.float32) * 3.0
    out = cluster_frontiers(fmap, dmap, max_frontiers=8)
    print("clusters", len(out))
    print("centers", [c.center_rc.tolist() for c in out])
