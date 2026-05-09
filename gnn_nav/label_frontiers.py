import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from gnn_nav.dataset import safe_torch_load


def make_soft_frontier_label(costs, tau=2.0):
    costs = np.asarray(costs, dtype=np.float32)
    valid = np.isfinite(costs)
    y = np.zeros_like(costs, dtype=np.float32)
    if len(costs) == 0:
        return y
    if valid.sum() == 0:
        y[:] = 1.0 / len(y)
        return y
    logits = -costs[valid] / float(tau)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    y[valid] = probs
    return y


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _get_goal_rc(metadata):
    for key in ["goal_rc", "goal_center_rc", "true_goal_rc"]:
        if key in metadata:
            arr = _as_numpy(metadata[key]).astype(np.float32).reshape(-1)
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                return arr[:2]
    return None


def _frontier_arrays(sample):
    frontier = sample.get("frontier", {})
    centers = _as_numpy(frontier.get("centers_rc"))
    if centers is None:
        graph_payload = sample.get("graph", {})
        if isinstance(graph_payload, dict):
            centers = _as_numpy(graph_payload.get("frontier_centers_rc"))
        else:
            centers = _as_numpy(getattr(graph_payload, "frontier_centers_rc", None))
    centers = np.asarray(centers, dtype=np.float32)
    mean_path_dist = _as_numpy(frontier.get("mean_path_dist"))
    distance_inverse = _as_numpy(frontier.get("distance_inverse"))
    if mean_path_dist is None:
        mean_path_dist = np.zeros((len(centers),), dtype=np.float32)
    if distance_inverse is None:
        distance_inverse = np.zeros((len(centers),), dtype=np.float32)
    return centers, np.asarray(mean_path_dist, dtype=np.float32), np.asarray(distance_inverse, dtype=np.float32)


def compute_costs(sample, label_mode: str, lambda_goal: float):
    centers, mean_path_dist, distance_inverse = _frontier_arrays(sample)
    if len(centers) == 0:
        return np.zeros((0,), dtype=np.float32)

    teacher_scores = sample.get("teacher", {}).get("sgnav_scores", None)
    teacher_scores = _as_numpy(teacher_scores)

    if label_mode == "teacher" and teacher_scores is not None and len(teacher_scores) == len(centers):
        return -np.asarray(teacher_scores, dtype=np.float32)

    if label_mode == "habitat_geodesic":
        print("[label_frontiers] habitat_geodesic is not implemented yet; falling back to approximate.")

    goal_rc = _get_goal_rc(sample.get("metadata", {}))
    if goal_rc is not None:
        map_resolution_cm = float(sample.get("metadata", {}).get("map_resolution_cm", 5.0))
        euclidean_m = np.linalg.norm(centers - goal_rc.reshape(1, 2), axis=1) * map_resolution_cm / 100.0
        return mean_path_dist + float(lambda_goal) * euclidean_m.astype(np.float32)

    if teacher_scores is not None and len(teacher_scores) == len(centers):
        return -np.asarray(teacher_scores, dtype=np.float32)

    return -np.asarray(distance_inverse, dtype=np.float32)


def label_sample(sample, label_mode: str, tau: float, lambda_goal: float):
    costs = compute_costs(sample, label_mode, lambda_goal)
    y_soft = make_soft_frontier_label(costs, tau=tau)
    if len(costs) == 0:
        best_idx = -1
    else:
        valid = np.isfinite(costs)
        best_idx = int(np.argmin(np.where(valid, costs, np.inf)))
        if not np.isfinite(costs[best_idx]):
            best_idx = int(np.argmax(y_soft))
    sample["labels"] = {
        "frontier_cost": torch.tensor(costs, dtype=torch.float32),
        "frontier_y_soft": torch.tensor(y_soft, dtype=torch.float32),
        "frontier_best_idx": int(best_idx),
    }
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--label_mode",
        type=str,
        default="approximate",
        choices=["approximate", "teacher", "habitat_geodesic"],
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lambda_goal", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    for path in paths:
        sample = safe_torch_load(path, map_location="cpu")
        sample = label_sample(sample, args.label_mode, args.tau, args.lambda_goal)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        tmp_path = out_path + ".tmp"
        torch.save(sample, tmp_path)
        os.replace(tmp_path, out_path)
        count += 1

    print(f"labeled {count} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
