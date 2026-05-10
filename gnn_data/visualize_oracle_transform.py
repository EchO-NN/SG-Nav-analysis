import argparse
import json
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


def _draw_idx(ax, centers, idx, label, color, marker):
    if idx is None:
        return
    try:
        idx = int(idx)
    except Exception:
        return
    if 0 <= idx < len(centers):
        p = centers[idx]
        ax.scatter([p[1]], [p[0]], s=140, c=color, marker=marker, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_sample", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    sample = safe_torch_load(args.raw_sample, map_location="cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    maps = sample.get("maps", {})
    base = _map_2d(maps.get("free_map", maps.get("full_map")))
    if base is None:
        base = _map_2d(sample.get("frontier", {}).get("frontier_map"))
    if base is None:
        base = np.zeros((800, 800), dtype=np.float32)

    centers = _frontiers(sample)
    teacher = sample.get("teacher", {})
    oracle = sample.get("oracle", {})
    labels = sample.get("labels", {})

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(base, cmap="gray")
    if len(centers):
        ax.scatter(centers[:, 1], centers[:, 0], s=8, c="tab:blue", label="frontiers")
    _draw_idx(ax, centers, teacher.get("selected_valid_idx", labels.get("teacher_best_idx")), "teacher", "tab:orange", "x")
    _draw_idx(ax, centers, oracle.get("frontier_best_idx", labels.get("frontier_best_idx")), "oracle/geodesic", "tab:red", "*")
    _draw_idx(ax, centers, sample.get("frontier", {}).get("selected_valid_idx"), "selected", "tab:purple", "^")

    metadata = sample.get("metadata", {})
    ax.set_title(
        f"{metadata.get('scene_id', '')} ep={metadata.get('episode_id', '')} "
        f"step={metadata.get('step_id', '')} oracle={oracle.get('label_type', 'none')}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    png_path = save_dir / (Path(args.raw_sample).stem + "_oracle_transform.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    geodesic = _np(oracle.get("frontier_goal_geodesic"))
    invalid_ratio = None
    if geodesic is not None and geodesic.size:
        invalid_ratio = float(1.0 - np.isfinite(geodesic).sum() / geodesic.size)
    report = {
        "raw_sample": args.raw_sample,
        "label_type": oracle.get("label_type"),
        "coordinate_warning": oracle.get("coordinate_warning"),
        "num_frontiers": int(len(centers)),
        "invalid_geodesic_ratio": invalid_ratio,
        "teacher_best_idx": teacher.get("selected_valid_idx", labels.get("teacher_best_idx")),
        "oracle_best_idx": oracle.get("frontier_best_idx", labels.get("frontier_best_idx")),
        "figure": str(png_path),
    }
    report_path = save_dir / (Path(args.raw_sample).stem + "_oracle_transform.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
