import argparse
import glob
import os
import random
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


def _map_2d(value):
    arr = _np(value)
    if arr is None:
        return None
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32)


def _frontiers(sample):
    frontier = sample.get("frontier", {})
    centers = frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc"))
    centers = _np(centers)
    if centers is None:
        return np.zeros((0, 2), dtype=np.float32)
    return centers.astype(np.float32).reshape(-1, 2)


def _draw_points(ax, points, **kwargs):
    if points is None or len(points) == 0:
        return
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ax.scatter(points[:, 1], points[:, 0], **kwargs)


def _annotate_idx(ax, frontiers, idx, label, color, marker):
    if idx is None:
        return
    try:
        idx = int(idx)
    except Exception:
        return
    if 0 <= idx < len(frontiers):
        point = frontiers[idx]
        ax.scatter([point[1]], [point[0]], s=120, c=color, marker=marker, label=label)


def visualize(path, output):
    sample = safe_torch_load(path, map_location="cpu")
    maps = sample.get("maps", {})
    labels = sample.get("labels", {})
    frontiers = _frontiers(sample)

    partial = _map_2d(maps.get("free_map", maps.get("full_map")))
    final = _map_2d(sample.get("final_maps", {}).get("free_map"))
    if final is None:
        final = partial
    if partial is None:
        partial = np.zeros((800, 800), dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, base, title in [
        (axes[0], partial, "partial map"),
        (axes[1], final, "final/hindsight map"),
    ]:
        ax.imshow(base, cmap="gray")
        _draw_points(ax, frontiers, s=8, c="tab:blue", label="frontiers")
        _annotate_idx(ax, frontiers, sample.get("frontier", {}).get("selected_valid_idx"), "selected", "tab:orange", "x")
        _annotate_idx(ax, frontiers, labels.get("teacher_best_idx"), "teacher best", "tab:purple", "^")
        _annotate_idx(ax, frontiers, labels.get("frontier_best_idx"), "hindsight best", "tab:red", "*")
        goal_rc = labels.get("found_goal_rc", labels.get("hindsight_goal_rc"))
        goal_rc = _np(goal_rc)
        if goal_rc is not None and goal_rc.size >= 2:
            p = goal_rc.reshape(-1)[:2]
            ax.scatter([p[1]], [p[0]], s=140, c="tab:green", marker="P", label="found goal")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)

    metadata = sample.get("metadata", {})
    fig.suptitle(
        f"{metadata.get('scene_id', '')} ep={metadata.get('episode_id', '')} "
        f"step={metadata.get('step_id', '')} label={labels.get('label_type', '')}"
    )
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--labeled_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = []
    if args.path:
        paths = [args.path]
    elif args.labeled_dir:
        paths = sorted(glob.glob(os.path.join(args.labeled_dir, "**", "*.pt"), recursive=True))
        rng = random.Random(args.seed)
        rng.shuffle(paths)
        paths = paths[: args.num_samples]
    else:
        raise ValueError("Provide --path or --labeled_dir")

    for idx, path in enumerate(paths):
        out = Path(args.save_dir) / f"{idx:04d}_{Path(path).stem}.png"
        visualize(path, out)
        print(f"saved visualization: {out}")


if __name__ == "__main__":
    main()
