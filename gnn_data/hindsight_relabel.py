from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from gnn_data.raw_schema import LABEL_VERSION, make_soft_frontier_label
from gnn_nav.dataset import safe_torch_load


def _np(value, dtype=np.float32):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _map_2d(value):
    arr = _np(value)
    if arr is None:
        return None
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32)


def _scene_key(scene_id):
    text = str(scene_id).replace("\\", "/")
    parts = text.split("/")
    for i, part in enumerate(parts):
        if part == "mp3d" and i + 1 < len(parts):
            return parts[i + 1]
    return Path(text).stem or text


def _episode_key(metadata):
    return (_scene_key(metadata.get("scene_id", "")), str(metadata.get("episode_id", "")))


def _frontiers(sample):
    frontier = sample.get("frontier", {})
    centers = frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc"))
    centers = _np(centers)
    if centers is None:
        return np.zeros((0, 2), dtype=np.float32)
    return centers.astype(np.float32).reshape(-1, 2)


def _path_distances(sample):
    frontier = sample.get("frontier", {})
    distances = frontier.get("distances_valid", frontier.get("mean_path_dist"))
    if distances is None:
        return None
    return _np(distances).astype(np.float32).reshape(-1)


def _final_traversible(summary):
    final_maps = summary.get("final_maps", {})
    free_map = _map_2d(final_maps.get("free_map"))
    if free_map is not None and free_map.size:
        return (free_map > 0.5).astype(np.float32)
    full_map = _map_2d(final_maps.get("full_map"))
    if full_map is not None and full_map.size:
        return (full_map > 0.5).astype(np.float32)
    return None


def _fmm_distance_map(traversible, goal_rc):
    try:
        from utils.utils_fmm.fmm_planner import FMMPlanner
    except Exception as exc:
        raise RuntimeError(f"FMMPlanner import failed: {exc}") from exc

    if traversible is None or traversible.sum() == 0:
        raise ValueError("missing_final_map")
    goal_rc = np.asarray(goal_rc, dtype=np.float32).reshape(-1)[:2]
    if goal_rc.size < 2 or not np.isfinite(goal_rc).all():
        raise ValueError("missing_goal_rc")
    r = int(np.clip(round(float(goal_rc[0])), 0, traversible.shape[0] - 1))
    c = int(np.clip(round(float(goal_rc[1])), 0, traversible.shape[1] - 1))
    planner = FMMPlanner(traversible.astype(np.float32), None)
    planner.set_goal([r, c])
    return np.asarray(planner.fmm_dist, dtype=np.float32)


def _teacher_fields(sample):
    out = {}
    teacher = sample.get("teacher", {})
    scores = teacher.get("total_scores", teacher.get("sgnav_scores", None))
    if scores is not None:
        scores = _np(scores).astype(np.float32).reshape(-1)
        out["teacher_scores"] = torch.tensor(scores, dtype=torch.float32)
        out["teacher_best_idx"] = int(np.argmax(scores)) if len(scores) else -1
    return out


def label_step_with_final_map(
    step_sample,
    episode_summary,
    tau=2.0,
    lambda_goal=1.0,
    min_label_frontiers=2,
    output_label_type="hindsight_goal_final_map",
    label_source="episode_summary_final_map",
    episode_summary_path=None,
):
    metadata = step_sample.get("metadata", {})
    frontiers = _frontiers(step_sample)
    if len(frontiers) < int(min_label_frontiers):
        raise ValueError("skipped_invalid_frontiers")

    d_path = _path_distances(step_sample)
    if d_path is None or len(d_path) != len(frontiers):
        raise ValueError("skipped_missing_path_distance")

    target_goal = episode_summary.get("target_goal", {})
    goal_rc = _np(target_goal.get("found_goal_rc"))
    if goal_rc is None:
        raise ValueError("skipped_no_found_goal")
    goal_rc = goal_rc.astype(np.float32).reshape(-1)[:2]

    traversible = _final_traversible(episode_summary)
    if traversible is None:
        raise ValueError("skipped_missing_final_map")
    dist_map = _fmm_distance_map(traversible, goal_rc)
    map_resolution_cm = float(metadata.get("map_resolution_cm", 5.0))

    d_goal = []
    for rc in frontiers:
        r = int(np.clip(round(float(rc[0])), 0, dist_map.shape[0] - 1))
        c = int(np.clip(round(float(rc[1])), 0, dist_map.shape[1] - 1))
        d_goal.append(float(dist_map[r, c]) * map_resolution_cm / 100.0)
    d_goal = np.asarray(d_goal, dtype=np.float32)
    costs = d_path.astype(np.float32) + float(lambda_goal) * d_goal
    finite = np.isfinite(costs)
    if finite.sum() == 0:
        raise ValueError("skipped_all_inf_cost")

    y_soft = make_soft_frontier_label(costs, tau=tau)
    if finite.sum() <= 1:
        raise ValueError("skipped_label_uniform")
    best_idx = int(np.argmin(np.where(finite, costs, np.inf)))

    labels = {
        "frontier_cost": torch.tensor(costs, dtype=torch.float32),
        "frontier_y_soft": torch.tensor(y_soft, dtype=torch.float32),
        "frontier_best_idx": int(best_idx),
        "label_type": output_label_type,
        "label_source": label_source,
        "hindsight_goal_rc": torch.tensor(goal_rc, dtype=torch.float32),
        "found_goal_rc": torch.tensor(goal_rc, dtype=torch.float32),
        "frontier_goal_final_map_dist": torch.tensor(d_goal, dtype=torch.float32),
        "episode_summary_path": "" if episode_summary_path is None else str(episode_summary_path),
        "target_goal_source": str(target_goal.get("source", "unknown")),
    }
    labels.update(_teacher_fields(step_sample))
    out = dict(step_sample)
    out["version"] = LABEL_VERSION
    out["labels"] = labels
    out["episode_summary_metadata"] = episode_summary.get("metadata", {})
    out["final_maps"] = episode_summary.get("final_maps", {})
    return out


def load_summaries(summary_dir):
    summaries = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(summary_dir, "**", "*.pt"), recursive=True)):
        summary = safe_torch_load(path, map_location="cpu")
        if summary.get("version") != "episode_summary_v1":
            continue
        metadata = summary.get("metadata", {})
        summaries[_episode_key(metadata)].append((path, summary))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_step_dir", type=str, required=True)
    parser.add_argument("--episode_summary_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_mode", choices=["hindsight_goal_strict"], default="hindsight_goal_strict")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lambda_goal", type=float, default=1.0)
    parser.add_argument("--strict_labels", action="store_true")
    parser.add_argument("--min_label_frontiers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_unlabeled", action="store_true")
    parser.add_argument("--write_label_report", nargs="?", const="hindsight_label_report.json", default=None)
    args = parser.parse_args()

    summaries = load_summaries(args.episode_summary_dir)
    paths = sorted(glob.glob(os.path.join(args.raw_step_dir, "**", "*.pt"), recursive=True))
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    label_type_counts = Counter()
    skip_reason_counts = Counter()
    episode_counts = Counter()
    count = 0
    for path in paths:
        sample = safe_torch_load(path, map_location="cpu")
        key = _episode_key(sample.get("metadata", {}))
        episode_counts[key] += 1
        candidates = summaries.get(key, [])
        if not candidates:
            skip_reason_counts["skipped_missing_episode_summary"] += 1
            if not args.skip_unlabeled and args.strict_labels:
                raise RuntimeError(f"missing episode summary for {key}")
            continue
        _, summary = candidates[-1]
        if not bool(summary.get("metadata", {}).get("success", False)):
            skip_reason_counts["skipped_no_success"] += 1
            continue
        try:
            labeled = label_step_with_final_map(
                sample,
                summary,
                tau=args.tau,
                lambda_goal=args.lambda_goal,
                min_label_frontiers=args.min_label_frontiers,
                output_label_type="hindsight_goal_final_map",
                label_source=f"episode_summary_{summary.get('target_goal', {}).get('source', 'found_goal')}",
                episode_summary_path=candidates[-1][0],
            )
        except Exception as exc:
            reason = str(exc) if str(exc).startswith("skipped_") else f"skipped_{str(exc)}"
            skip_reason_counts[reason] += 1
            if args.strict_labels and not args.skip_unlabeled:
                raise
            continue

        rel = os.path.relpath(path, args.raw_step_dir)
        out_path = os.path.join(args.output_dir, rel)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path + ".tmp"
        torch.save(labeled, tmp_path)
        os.replace(tmp_path, out_path)
        count += 1
        label_type_counts[str(labeled["labels"]["label_type"])] += 1

    report = {
        "raw_step_dir": args.raw_step_dir,
        "episode_summary_dir": args.episode_summary_dir,
        "output_dir": args.output_dir,
        "num_raw_steps": len(paths),
        "num_labeled_samples": count,
        "label_type_counts": dict(label_type_counts),
        "skip_reason_counts": dict(skip_reason_counts),
        "episode_step_counts": {f"{k[0]}::{k[1]}": v for k, v in episode_counts.items()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write_label_report:
        report_path = Path(args.write_label_report)
        if not report_path.is_absolute():
            report_path = Path(args.output_dir) / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote label report: {report_path}")


if __name__ == "__main__":
    main()
