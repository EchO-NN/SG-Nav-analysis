import argparse
import copy
import glob
import os
from pathlib import Path

import numpy as np
import torch

from gnn_data.hindsight_labels import (
    goal_rc_from_sample,
    label_with_goal_rc,
    pseudo_goal_objects,
)
from gnn_data.raw_schema import LABEL_VERSION, make_soft_frontier_label, softmax_scores
from gnn_nav.dataset import safe_torch_load


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _teacher_scores(sample):
    teacher = sample.get("teacher", {})
    for key in ["total_scores", "sgnav_scores", "teacher_scores"]:
        if key in teacher and teacher[key] is not None:
            return _as_numpy(teacher[key]).astype(np.float32).reshape(-1)
    return None


def _frontier_count(sample):
    frontier = sample.get("frontier", {})
    for key in ["frontier_locations_valid_rc", "centers_rc"]:
        if key in frontier and frontier[key] is not None:
            return len(_as_numpy(frontier[key]).reshape(-1, 2))
    graph = sample.get("graph", {})
    if isinstance(graph, dict) and graph.get("frontier_centers_rc") is not None:
        return len(_as_numpy(graph["frontier_centers_rc"]).reshape(-1, 2))
    return 0


def _distance_cost(sample):
    frontier = sample.get("frontier", {})
    distances = frontier.get("distances_valid", frontier.get("mean_path_dist", None))
    if distances is not None:
        return _as_numpy(distances).astype(np.float32).reshape(-1)
    inv = frontier.get("distance_inverse", frontier.get("distance_inverse_valid", None))
    if inv is not None:
        return -_as_numpy(inv).astype(np.float32).reshape(-1)
    return np.zeros((_frontier_count(sample),), dtype=np.float32)


def _oracle_saved(sample):
    oracle = sample.get("oracle", {})
    if not isinstance(oracle, dict):
        return None
    y = oracle.get("frontier_y_soft")
    costs = oracle.get("frontier_cost")
    if y is None or costs is None:
        return None
    costs = _as_numpy(costs).astype(np.float32).reshape(-1)
    y = _as_numpy(y).astype(np.float32).reshape(-1)
    if len(costs) != len(y) or len(costs) == 0:
        return None
    best_idx = int(oracle.get("frontier_best_idx", int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))))
    return costs, y, best_idx, str(oracle.get("label_type", "oracle_online_saved"))


def _approx_map_cost(sample, lambda_goal: float):
    frontier = sample.get("frontier", {})
    centers = frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", None))
    centers = _as_numpy(centers)
    if centers is None:
        return _distance_cost(sample)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    costs = _distance_cost(sample)
    metadata = sample.get("metadata", {})
    goal_rc = None
    for key in ["goal_rc", "goal_center_rc", "true_goal_rc"]:
        if key in metadata:
            goal_rc = _as_numpy(metadata[key]).astype(np.float32).reshape(-1)[:2]
            break
    if goal_rc is None or len(costs) != len(centers):
        return costs
    map_resolution_cm = float(metadata.get("map_resolution_cm", 5.0))
    goal_dist = np.linalg.norm(centers - goal_rc.reshape(1, 2), axis=1) * map_resolution_cm / 100.0
    return costs + float(lambda_goal) * goal_dist.astype(np.float32)


def label_sample(sample, label_mode: str, tau: float, teacher_temperature: float, lambda_goal: float):
    n_frontiers = _frontier_count(sample)
    teacher_scores = _teacher_scores(sample)
    teacher_y = None
    teacher_best = -1
    if teacher_scores is not None and len(teacher_scores) == n_frontiers:
        teacher_y = softmax_scores(teacher_scores, teacher_temperature)
        teacher_best = int(np.argmax(teacher_scores)) if len(teacher_scores) else -1

    oracle = _oracle_saved(sample)
    extra_label_fields = {}
    if label_mode == "teacher":
        if teacher_scores is None or len(teacher_scores) != n_frontiers:
            raise ValueError("teacher mode requires teacher.total_scores matching frontier count")
        costs = -teacher_scores
        y_soft = teacher_y
        best_idx = teacher_best
        label_type = "teacher_debug"
    elif label_mode in ["oracle_online_saved", "simulator_geodesic"]:
        if oracle is None:
            raise ValueError(f"{label_mode} mode requires saved sample['oracle'] frontier labels")
        costs, y_soft, best_idx, label_type = oracle
        if label_mode == "simulator_geodesic" and "geodesic" not in label_type:
            label_type = f"simulator_geodesic_from_{label_type}"
    elif label_mode in ["final_map_hindsight", "hindsight_goal"]:
        goal_rc = goal_rc_from_sample(sample)
        if goal_rc is None:
            raise ValueError(f"{label_mode} mode requires a goal rc field in sample['goal'], metadata, or agent")
        out = label_with_goal_rc(sample, goal_rc, tau=tau, lambda_goal=lambda_goal, label_type=label_mode)
        sample.clear()
        sample.update(out)
        costs = _as_numpy(sample["labels"]["frontier_cost"]).astype(np.float32)
        y_soft = _as_numpy(sample["labels"]["frontier_y_soft"]).astype(np.float32)
        best_idx = int(sample["labels"]["frontier_best_idx"])
        label_type = str(sample["labels"]["label_type"])
        extra_label_fields["hindsight_goal_rc"] = sample["labels"].get("hindsight_goal_rc")
    elif label_mode == "hybrid":
        if oracle is not None:
            costs, y_soft, best_idx, label_type = oracle
            label_type = f"{label_type}+teacher" if teacher_y is not None else label_type
        elif goal_rc_from_sample(sample) is not None:
            out = label_with_goal_rc(
                sample,
                goal_rc_from_sample(sample),
                tau=tau,
                lambda_goal=lambda_goal,
                label_type="hindsight_goal+teacher" if teacher_y is not None else "hindsight_goal",
            )
            sample.clear()
            sample.update(out)
            costs = _as_numpy(sample["labels"]["frontier_cost"]).astype(np.float32)
            y_soft = _as_numpy(sample["labels"]["frontier_y_soft"]).astype(np.float32)
            best_idx = int(sample["labels"]["frontier_best_idx"])
            label_type = str(sample["labels"]["label_type"])
            extra_label_fields["hindsight_goal_rc"] = sample["labels"].get("hindsight_goal_rc")
        elif teacher_scores is not None and len(teacher_scores) == n_frontiers:
            costs = -teacher_scores
            y_soft = teacher_y
            best_idx = teacher_best
            label_type = "teacher_debug_fallback"
        else:
            costs = _approx_map_cost(sample, lambda_goal)
            y_soft = make_soft_frontier_label(costs, tau=tau)
            best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf))) if len(costs) else -1
            label_type = "oracle_approx_map_fallback"
    elif label_mode == "oracle_approx_map":
        costs = _approx_map_cost(sample, lambda_goal)
        y_soft = make_soft_frontier_label(costs, tau=tau)
        best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf))) if len(costs) else -1
        label_type = "oracle_approx_map"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    sample["version"] = LABEL_VERSION
    sample["labels"] = {
        "frontier_y_soft": torch.tensor(y_soft, dtype=torch.float32),
        "frontier_cost": torch.tensor(costs, dtype=torch.float32),
        "frontier_best_idx": int(best_idx),
        "label_type": label_type,
    }
    sample["labels"].update({k: v for k, v in extra_label_fields.items() if v is not None})
    if teacher_y is not None:
        sample["labels"].update(
            {
                "teacher_y_soft": torch.tensor(teacher_y, dtype=torch.float32),
                "teacher_scores": torch.tensor(teacher_scores, dtype=torch.float32),
                "teacher_best_idx": int(teacher_best),
            }
        )
    return sample


def label_samples(sample, label_mode: str, tau: float, teacher_temperature: float, lambda_goal: float, pseudo_goal_min_confidence: float):
    if label_mode != "hindsight_all_objects":
        return [label_sample(sample, label_mode, tau, teacher_temperature, lambda_goal)]

    pseudo_goals = pseudo_goal_objects(sample, min_confidence=pseudo_goal_min_confidence)
    if not pseudo_goals:
        raise ValueError("hindsight_all_objects mode found no confident objects with center_rc")
    out = []
    for idx, pseudo in enumerate(pseudo_goals):
        relabeled = copy.deepcopy(sample)
        relabeled.setdefault("goal", {})
        relabeled["goal"] = dict(relabeled["goal"])
        relabeled["goal"]["object_category_raw"] = pseudo["goal_text"]
        relabeled["goal"]["object_category_sg"] = pseudo["goal_text"]
        relabeled["goal"]["episode_goal_category"] = pseudo["goal_text"]
        relabeled["goal"]["hindsight_pseudo_goal"] = True
        relabeled["goal"]["source_node_id"] = pseudo.get("source_node_id")
        relabeled["goal"]["goal_rc"] = pseudo["goal_rc"]
        relabeled = label_with_goal_rc(
            relabeled,
            pseudo["goal_rc"],
            tau=tau,
            lambda_goal=lambda_goal,
            label_type="hindsight_all_objects",
        )
        relabeled["version"] = LABEL_VERSION
        relabeled["labels"]["pseudo_goal_text"] = pseudo["goal_text"]
        relabeled["labels"]["pseudo_goal_index"] = int(idx)
        relabeled["labels"]["pseudo_goal_confidence"] = float(pseudo["confidence"])
        out.append(relabeled)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--label_mode",
        type=str,
        required=True,
        choices=[
            "teacher",
            "oracle_online_saved",
            "simulator_geodesic",
            "final_map_hindsight",
            "hindsight_goal",
            "hindsight_all_objects",
            "hybrid",
            "oracle_approx_map",
        ],
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_goal", type=float, default=1.0)
    parser.add_argument("--pseudo_goal_min_confidence", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pt"), recursive=True))
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    for path in paths:
        try:
            sample = safe_torch_load(path, map_location="cpu")
            labeled_samples = label_samples(
                sample,
                args.label_mode,
                args.tau,
                args.teacher_temperature,
                args.lambda_goal,
                args.pseudo_goal_min_confidence,
            )
        except Exception:
            if args.skip_errors:
                skipped += 1
                continue
            raise
        rel = os.path.relpath(path, args.input_dir)
        stem, ext = os.path.splitext(rel)
        for sample_idx, labeled_sample in enumerate(labeled_samples):
            if len(labeled_samples) == 1:
                out_rel = rel
            else:
                suffix = str(labeled_sample.get("labels", {}).get("pseudo_goal_index", sample_idx))
                goal_text = str(labeled_sample.get("labels", {}).get("pseudo_goal_text", "goal"))
                goal_text = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in goal_text)[:48]
                out_rel = f"{stem}__pseudo_{suffix}_{goal_text}{ext}"
            out_path = os.path.join(args.output_dir, out_rel)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path + ".tmp"
            torch.save(labeled_sample, tmp_path)
            os.replace(tmp_path, out_path)
            count += 1

    print(f"labeled {count} samples -> {args.output_dir}")
    if skipped:
        print(f"skipped {skipped} samples")


if __name__ == "__main__":
    main()
