import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from gnn_nav.dataset import safe_torch_load


def _np(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _base_map(sample):
    maps = sample.get("maps", {})
    for key in ["free_map", "full_map"]:
        value = maps.get(key)
        if value is not None:
            arr = _np(value).astype(np.float32)
            while arr.ndim > 2:
                arr = arr[0]
            return arr
    fmap = sample.get("frontier", {}).get("frontier_map")
    if fmap is not None:
        return np.zeros_like(_np(fmap), dtype=np.float32)
    return np.zeros((800, 800), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    sample = safe_torch_load(args.path, map_location="cpu")
    frontier = sample.get("frontier", {})
    teacher = sample.get("teacher", {})
    oracle = sample.get("oracle", {})
    labels = sample.get("labels", {})
    scenegraph = sample.get("scenegraph", {})

    base = _base_map(sample)
    valid = _np(frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", [])))
    valid = np.asarray(valid, dtype=np.float32).reshape(-1, 2) if valid is not None else np.zeros((0, 2))
    all_frontiers = _np(frontier.get("frontier_locations_all_rc", []))
    all_frontiers = (
        np.asarray(all_frontiers, dtype=np.float32).reshape(-1, 2)
        if all_frontiers is not None
        else np.zeros((0, 2))
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(base, cmap="gray")
    if len(all_frontiers):
        plt.scatter(all_frontiers[:, 1], all_frontiers[:, 0], s=2, c="tab:cyan", label="all frontier")
    if len(valid):
        plt.scatter(valid[:, 1], valid[:, 0], s=10, c="tab:blue", label="valid frontier")

    teacher_idx = teacher.get("selected_valid_idx", labels.get("teacher_best_idx", None))
    if teacher_idx is not None and len(valid) > int(teacher_idx) >= 0:
        p = valid[int(teacher_idx)]
        plt.scatter([p[1]], [p[0]], s=90, marker="x", c="tab:orange", label="teacher best")

    oracle_idx = oracle.get("frontier_best_idx", labels.get("frontier_best_idx", None))
    if oracle_idx is not None and len(valid) > int(oracle_idx) >= 0:
        p = valid[int(oracle_idx)]
        plt.scatter([p[1]], [p[0]], s=120, marker="*", c="tab:red", label="oracle/label best")

    for obj in scenegraph.get("objects", []):
        center = obj.get("center_rc")
        if center is None:
            continue
        center = np.asarray(center, dtype=np.float32).reshape(-1)
        if center.size >= 2:
            plt.scatter([center[1]], [center[0]], s=18, c="tab:green")

    plt.title(
        f"{sample.get('metadata', {}).get('scene_id', '')} "
        f"step={sample.get('metadata', {}).get('step_id', '')} "
        f"goal={sample.get('goal', {}).get('object_category_sg', '')}"
    )
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print(f"saved visualization: {args.output}")


if __name__ == "__main__":
    main()
