import argparse
import copy
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from gnn_data.hindsight_labels import (
    goal_rc_from_sample,
    label_with_goal_rc,
    pseudo_goal_objects,
)
from gnn_data.hindsight_relabel import _episode_key, label_step_with_final_map, load_summaries
from gnn_data.raw_schema import LABEL_VERSION, make_soft_frontier_label, softmax_scores
from gnn_nav.dataset import safe_torch_load


FINAL_LABEL_MODES = {
    "hindsight_goal",
    "hindsight_goal_strict",
    "hindsight_all_objects",
    "hindsight_all_objects_strict",
    "final_map_hindsight",
    "final_map_hindsight_strict",
    "hybrid_hindsight_first_strict",
}


class SkipSample(Exception):
    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


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


def _validate_frontiers(sample, min_label_frontiers: int):
    n_frontiers = _frontier_count(sample)
    if n_frontiers < int(min_label_frontiers):
        raise SkipSample("skipped_invalid_frontiers", f"need >= {min_label_frontiers} frontiers, got {n_frontiers}")
    return n_frontiers


def _distance_cost(sample):
    frontier = sample.get("frontier", {})
    distances = frontier.get("distances_valid", frontier.get("mean_path_dist", None))
    if distances is not None:
        return _as_numpy(distances).astype(np.float32).reshape(-1)
    inv = frontier.get("distance_inverse", frontier.get("distance_inverse_valid", None))
    if inv is not None:
        return -_as_numpy(inv).astype(np.float32).reshape(-1)
    return np.zeros((_frontier_count(sample),), dtype=np.float32)


def _oracle_saved(sample, allow_unvalidated_geodesic: bool = False):
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
    label_type = str(oracle.get("label_type", "oracle_online_saved"))
    if label_type == "online_geodesic_unvalidated" and not allow_unvalidated_geodesic:
        raise SkipSample(
            "skipped_unvalidated_geodesic",
            "online_geodesic_unvalidated requires --allow_unvalidated_geodesic",
        )
    best_idx = int(oracle.get("frontier_best_idx", int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))))
    return costs, y, best_idx, label_type


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


def _canonical_mode(label_mode: str) -> str:
    aliases = {
        "teacher_debug": "teacher",
        "oracle_geodesic_diagnostic": "simulator_geodesic",
        "oracle_approx_debug": "oracle_approx_map",
    }
    return aliases.get(label_mode, label_mode)


def _set_label_source(sample: dict, source: str):
    sample.setdefault("labels", {})
    sample["labels"]["label_source"] = source


def label_sample(
    sample,
    label_mode: str,
    tau: float,
    teacher_temperature: float,
    lambda_goal: float,
    strict_labels: bool = False,
    forbid_teacher_fallback: bool = False,
    forbid_approx_fallback: bool = False,
    require_hindsight_or_oracle: bool = False,
    min_label_frontiers: int = 1,
    allow_unvalidated_geodesic: bool = False,
    episode_summary: dict = None,
    episode_summary_path: str = None,
    allow_failed_hindsight_debug: bool = False,
):
    original_label_mode = label_mode
    label_mode = _canonical_mode(label_mode)
    if original_label_mode.endswith("_strict") or label_mode.endswith("_strict") or strict_labels:
        strict_labels = True
        final_mode = original_label_mode in FINAL_LABEL_MODES or label_mode in FINAL_LABEL_MODES
        forbid_teacher_fallback = True if final_mode else forbid_teacher_fallback
        forbid_approx_fallback = True if final_mode else forbid_approx_fallback
        require_hindsight_or_oracle = True if final_mode else require_hindsight_or_oracle
    if label_mode == "hindsight_goal_strict":
        label_mode = "hindsight_goal"
    if label_mode == "hybrid_hindsight_first_strict":
        label_mode = "hybrid_hindsight_first"

    _validate_frontiers(sample, min_label_frontiers)
    n_frontiers = _frontier_count(sample)
    teacher_scores = _teacher_scores(sample)
    teacher_y = None
    teacher_best = -1
    if teacher_scores is not None and len(teacher_scores) == n_frontiers:
        teacher_y = softmax_scores(teacher_scores, teacher_temperature)
        teacher_best = int(np.argmax(teacher_scores)) if len(teacher_scores) else -1

    oracle = None
    extra_label_fields = {}
    if label_mode == "teacher":
        if teacher_scores is None or len(teacher_scores) != n_frontiers:
            raise ValueError("teacher mode requires teacher.total_scores matching frontier count")
        costs = -teacher_scores
        y_soft = teacher_y
        best_idx = teacher_best
        label_type = "teacher_debug"
    elif label_mode in ["oracle_online_saved", "simulator_geodesic"]:
        oracle = _oracle_saved(sample, allow_unvalidated_geodesic=allow_unvalidated_geodesic)
        if oracle is None:
            raise ValueError(f"{label_mode} mode requires saved sample['oracle'] frontier labels")
        costs, y_soft, best_idx, label_type = oracle
        if label_mode == "simulator_geodesic" and "geodesic" not in label_type:
            label_type = f"simulator_geodesic_from_{label_type}"
    elif label_mode in ["final_map_hindsight", "hindsight_goal"]:
        goal_rc = goal_rc_from_sample(sample)
        if goal_rc is None:
            raise ValueError(f"{label_mode} mode requires a goal rc field in sample['goal'], metadata, or agent")
        try:
            out = label_with_goal_rc(
                sample,
                goal_rc,
                tau=tau,
                lambda_goal=lambda_goal,
                label_type=label_mode,
                label_source="sample_goal_rc",
                allow_approx_fallback=not (strict_labels or forbid_approx_fallback),
            )
        except Exception as exc:
            reason = str(exc) if str(exc).startswith("skipped_") else f"skipped_{str(exc)}"
            raise SkipSample(reason) from exc
        sample.clear()
        sample.update(out)
        costs = _as_numpy(sample["labels"]["frontier_cost"]).astype(np.float32)
        y_soft = _as_numpy(sample["labels"]["frontier_y_soft"]).astype(np.float32)
        best_idx = int(sample["labels"]["frontier_best_idx"])
        label_type = str(sample["labels"]["label_type"])
        for key in [
            "label_source",
            "hindsight_goal_rc",
            "found_goal_rc",
            "frontier_goal_final_map_dist",
            "episode_summary_path",
            "target_goal_source",
        ]:
            extra_label_fields[key] = sample["labels"].get(key)
        extra_label_fields["hindsight_goal_rc"] = sample["labels"].get("hindsight_goal_rc")
    elif label_mode == "final_map_hindsight_strict":
        if episode_summary is None:
            raise SkipSample("skipped_missing_episode_summary")
        if not bool(episode_summary.get("metadata", {}).get("success", False)) and not allow_failed_hindsight_debug:
            raise SkipSample("skipped_no_success")
        try:
            out = label_step_with_final_map(
                sample,
                episode_summary,
                tau=tau,
                lambda_goal=lambda_goal,
                min_label_frontiers=min_label_frontiers,
                output_label_type="final_map_hindsight_strict",
                label_source="episode_summary_final_map",
                episode_summary_path=episode_summary_path,
            )
        except Exception as exc:
            reason = str(exc) if str(exc).startswith("skipped_") else f"skipped_{str(exc)}"
            raise SkipSample(reason) from exc
        sample.clear()
        sample.update(out)
        costs = _as_numpy(sample["labels"]["frontier_cost"]).astype(np.float32)
        y_soft = _as_numpy(sample["labels"]["frontier_y_soft"]).astype(np.float32)
        best_idx = int(sample["labels"]["frontier_best_idx"])
        label_type = str(sample["labels"]["label_type"])
        for key in [
            "label_source",
            "hindsight_goal_rc",
            "found_goal_rc",
            "frontier_goal_final_map_dist",
            "episode_summary_path",
            "target_goal_source",
        ]:
            extra_label_fields[key] = sample["labels"].get(key)
    elif label_mode == "hybrid":
        oracle = _oracle_saved(sample, allow_unvalidated_geodesic=allow_unvalidated_geodesic)
        if oracle is not None:
            costs, y_soft, best_idx, label_type = oracle
            label_type = f"{label_type}+teacher" if teacher_y is not None else label_type
        elif goal_rc_from_sample(sample) is not None:
            try:
                out = label_with_goal_rc(
                    sample,
                    goal_rc_from_sample(sample),
                    tau=tau,
                    lambda_goal=lambda_goal,
                    label_type="hindsight_goal+teacher" if teacher_y is not None else "hindsight_goal",
                    label_source="sample_goal_rc",
                    allow_approx_fallback=not (strict_labels or forbid_approx_fallback),
                )
            except Exception as exc:
                if strict_labels or forbid_approx_fallback or require_hindsight_or_oracle:
                    reason = str(exc) if str(exc).startswith("skipped_") else f"skipped_{str(exc)}"
                    raise SkipSample(reason) from exc
                raise
            sample.clear()
            sample.update(out)
            costs = _as_numpy(sample["labels"]["frontier_cost"]).astype(np.float32)
            y_soft = _as_numpy(sample["labels"]["frontier_y_soft"]).astype(np.float32)
            best_idx = int(sample["labels"]["frontier_best_idx"])
            label_type = str(sample["labels"]["label_type"])
            extra_label_fields["hindsight_goal_rc"] = sample["labels"].get("hindsight_goal_rc")
        elif teacher_scores is not None and len(teacher_scores) == n_frontiers:
            if strict_labels or forbid_teacher_fallback or require_hindsight_or_oracle:
                raise SkipSample("skipped_no_hindsight_or_oracle")
            costs = -teacher_scores
            y_soft = teacher_y
            best_idx = teacher_best
            label_type = "teacher_debug_fallback"
        else:
            if strict_labels or forbid_approx_fallback or require_hindsight_or_oracle:
                raise SkipSample("skipped_no_hindsight_or_oracle")
            costs = _approx_map_cost(sample, lambda_goal)
            y_soft = make_soft_frontier_label(costs, tau=tau)
            best_idx = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf))) if len(costs) else -1
            label_type = "oracle_approx_map_fallback"
    elif label_mode == "hybrid_hindsight_first":
        goal_rc = goal_rc_from_sample(sample)
        if goal_rc is not None:
            try:
                out = label_with_goal_rc(
                    sample,
                    goal_rc,
                    tau=tau,
                    lambda_goal=lambda_goal,
                    label_type="hindsight_goal",
                    label_source="sample_goal_rc",
                    allow_approx_fallback=not (strict_labels or forbid_approx_fallback),
                )
            except Exception as exc:
                reason = str(exc) if str(exc).startswith("skipped_") else f"skipped_{str(exc)}"
                raise SkipSample(reason) from exc
            sample.clear()
            sample.update(out)
            costs = _as_numpy(sample["labels"]["frontier_cost"]).astype(np.float32)
            y_soft = _as_numpy(sample["labels"]["frontier_y_soft"]).astype(np.float32)
            best_idx = int(sample["labels"]["frontier_best_idx"])
            label_type = str(sample["labels"]["label_type"])
            extra_label_fields["hindsight_goal_rc"] = sample["labels"].get("hindsight_goal_rc")
            extra_label_fields["label_source"] = "sample_goal_rc"
        else:
            oracle = _oracle_saved(sample, allow_unvalidated_geodesic=allow_unvalidated_geodesic)
            if oracle is None:
                raise SkipSample("skipped_no_hindsight_or_oracle")
            costs, y_soft, best_idx, label_type = oracle
            extra_label_fields["label_source"] = "sample_oracle"
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
        "label_source": extra_label_fields.pop("label_source", label_type),
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


def label_samples(
    sample,
    label_mode: str,
    tau: float,
    teacher_temperature: float,
    lambda_goal: float,
    pseudo_goal_min_confidence: float,
    pseudo_goal_min_observed_count: int = 1,
    pseudo_goal_min_lifetime_steps: int = 0,
    pseudo_goal_max_per_category: int = 0,
    pseudo_goal_exclude_unknown: bool = False,
    pseudo_goal_exclude_rejected_candidates: bool = False,
    pseudo_goal_balance_categories: bool = False,
    **label_kwargs,
):
    strict_all_objects = label_mode == "hindsight_all_objects_strict"
    if label_mode not in ["hindsight_all_objects", "hindsight_all_objects_strict"]:
        return [label_sample(sample, label_mode, tau, teacher_temperature, lambda_goal, **label_kwargs)]

    pseudo_source_sample = sample
    episode_summary = label_kwargs.get("episode_summary")
    episode_summary_path = label_kwargs.get("episode_summary_path")
    if strict_all_objects and episode_summary is None:
        raise SkipSample("skipped_missing_episode_summary")
    if strict_all_objects and episode_summary is not None:
        pseudo_source_sample = copy.deepcopy(sample)
        pseudo_source_sample.setdefault("scenegraph", {})
        pseudo_source_sample["scenegraph"] = dict(pseudo_source_sample["scenegraph"])
        pseudo_source_sample["scenegraph"]["objects"] = list(episode_summary.get("discovered_objects", []))
        pseudo_source_sample.setdefault("maps", {})
        pseudo_source_sample["maps"] = dict(pseudo_source_sample["maps"])
        final_maps = episode_summary.get("final_maps", {})
        if final_maps.get("free_map") is not None:
            pseudo_source_sample["maps"]["final_free_map"] = final_maps.get("free_map")
        if final_maps.get("full_map") is not None:
            pseudo_source_sample["maps"]["final_full_map"] = final_maps.get("full_map")

    pseudo_goals = pseudo_goal_objects(
        pseudo_source_sample,
        min_confidence=pseudo_goal_min_confidence,
        min_observed_count=pseudo_goal_min_observed_count,
        min_lifetime_steps=pseudo_goal_min_lifetime_steps,
        max_per_category=pseudo_goal_max_per_category,
        exclude_unknown=pseudo_goal_exclude_unknown or strict_all_objects,
        exclude_rejected_candidates=pseudo_goal_exclude_rejected_candidates or strict_all_objects,
        balance_categories=pseudo_goal_balance_categories,
        require_stable=strict_all_objects,
    )
    if not pseudo_goals:
        raise SkipSample("skipped_no_valid_pseudo_goal")
    out = []
    pseudo_skip_reasons = Counter()
    for idx, pseudo in enumerate(pseudo_goals):
        relabeled = copy.deepcopy(pseudo_source_sample)
        relabeled.setdefault("goal", {})
        relabeled["goal"] = dict(relabeled["goal"])
        relabeled["goal"]["object_category_raw"] = pseudo["goal_text"]
        relabeled["goal"]["object_category_sg"] = pseudo["goal_text"]
        relabeled["goal"]["episode_goal_category"] = pseudo["goal_text"]
        relabeled["goal"]["hindsight_pseudo_goal"] = True
        relabeled["goal"]["source_node_id"] = pseudo.get("source_node_id")
        relabeled["goal"]["goal_rc"] = pseudo["goal_rc"]
        if strict_all_objects:
            pseudo_summary = copy.deepcopy(episode_summary)
            pseudo_summary.setdefault("target_goal", {})
            pseudo_summary["target_goal"] = dict(pseudo_summary["target_goal"])
            pseudo_summary["target_goal"]["goal_text"] = pseudo["goal_text"]
            pseudo_summary["target_goal"]["found_goal_rc"] = pseudo["goal_rc"]
            pseudo_summary["target_goal"]["source"] = "episode_summary_discovered_object"
            pseudo_summary["target_goal"]["confidence"] = float(pseudo["confidence"])
            try:
                relabeled = label_step_with_final_map(
                    relabeled,
                    pseudo_summary,
                    tau=tau,
                    lambda_goal=lambda_goal,
                    min_label_frontiers=int(label_kwargs.get("min_label_frontiers", 1)),
                    output_label_type="hindsight_all_objects_strict",
                    label_source="episode_summary_discovered_object",
                    episode_summary_path=episode_summary_path,
                )
            except Exception as exc:
                reason = str(exc) if str(exc).startswith("skipped_") else f"skipped_{str(exc)}"
                pseudo_skip_reasons[reason] += 1
                continue
        else:
            relabeled = label_with_goal_rc(
                relabeled,
                pseudo["goal_rc"],
                tau=tau,
                lambda_goal=lambda_goal,
                label_type="hindsight_all_objects",
                label_source="scenegraph_object_pseudo_goal",
            )
        relabeled["version"] = LABEL_VERSION
        relabeled["labels"]["pseudo_goal_text"] = pseudo["goal_text"]
        relabeled["labels"]["pseudo_goal_index"] = int(idx)
        relabeled["labels"]["pseudo_goal_confidence"] = float(pseudo["confidence"])
        relabeled["labels"]["pseudo_goal_observed_count"] = int(pseudo["observed_count"])
        relabeled["labels"]["pseudo_goal_first_seen_step"] = int(pseudo["first_seen_step"])
        relabeled["labels"]["pseudo_goal_last_seen_step"] = int(pseudo["last_seen_step"])
        relabeled["labels"]["pseudo_goal_source_node_id"] = (
            -1 if pseudo.get("source_node_id") is None else int(pseudo.get("source_node_id"))
        )
        relabeled["labels"]["episode_summary_path"] = "" if episode_summary_path is None else str(episode_summary_path)
        out.append(relabeled)
    if not out:
        if pseudo_skip_reasons:
            reason = pseudo_skip_reasons.most_common(1)[0][0]
            raise SkipSample(reason)
        raise SkipSample("skipped_no_valid_pseudo_goal")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--episode_summary_dir", type=str, default=None)
    parser.add_argument(
        "--label_mode",
        type=str,
        required=True,
        choices=[
            "teacher",
            "teacher_debug",
            "oracle_online_saved",
            "simulator_geodesic",
            "oracle_geodesic_diagnostic",
            "final_map_hindsight",
            "final_map_hindsight_strict",
            "hindsight_goal",
            "hindsight_goal_strict",
            "hindsight_all_objects",
            "hindsight_all_objects_strict",
            "hybrid",
            "hybrid_hindsight_first_strict",
            "oracle_approx_map",
            "oracle_approx_debug",
        ],
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_goal", type=float, default=1.0)
    parser.add_argument("--pseudo_goal_min_confidence", type=float, default=0.5)
    parser.add_argument("--pseudo_goal_min_observed_count", type=int, default=1)
    parser.add_argument("--pseudo_goal_min_lifetime_steps", type=int, default=0)
    parser.add_argument("--pseudo_goal_max_per_category", type=int, default=0)
    parser.add_argument("--pseudo_goal_exclude_unknown", action="store_true")
    parser.add_argument("--pseudo_goal_exclude_rejected_candidates", action="store_true")
    parser.add_argument("--pseudo_goal_balance_categories", action="store_true")
    parser.add_argument("--strict_labels", action="store_true")
    parser.add_argument("--forbid_teacher_fallback", action="store_true")
    parser.add_argument("--forbid_approx_fallback", action="store_true")
    parser.add_argument("--require_hindsight_or_oracle", action="store_true")
    parser.add_argument("--min_label_frontiers", type=int, default=1)
    parser.add_argument("--skip_unlabeled", action="store_true")
    parser.add_argument("--write_label_report", nargs="?", const="label_report.json", default=None)
    parser.add_argument("--allow_unvalidated_geodesic", action="store_true")
    parser.add_argument("--allow_failed_hindsight_debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_errors", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pt"), recursive=True))
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(args.episode_summary_dir) if args.episode_summary_dir else {}

    count = 0
    skipped = 0
    label_type_counts = Counter()
    skip_reason_counts = Counter()
    pseudo_goal_category_counts = Counter()
    output_dir = Path(args.output_dir)
    for path in paths:
        try:
            sample = safe_torch_load(path, map_location="cpu")
            episode_summary = None
            episode_summary_path = None
            if args.episode_summary_dir:
                key = _episode_key(sample.get("metadata", {}))
                candidates = summaries.get(key, [])
                if candidates:
                    episode_summary_path, episode_summary = candidates[-1]
                elif args.label_mode in ["final_map_hindsight_strict", "hindsight_all_objects_strict"]:
                    raise SkipSample("skipped_missing_episode_summary")
            labeled_samples = label_samples(
                sample,
                args.label_mode,
                args.tau,
                args.teacher_temperature,
                args.lambda_goal,
                args.pseudo_goal_min_confidence,
                pseudo_goal_min_observed_count=args.pseudo_goal_min_observed_count,
                pseudo_goal_min_lifetime_steps=args.pseudo_goal_min_lifetime_steps,
                pseudo_goal_max_per_category=args.pseudo_goal_max_per_category,
                pseudo_goal_exclude_unknown=args.pseudo_goal_exclude_unknown,
                pseudo_goal_exclude_rejected_candidates=args.pseudo_goal_exclude_rejected_candidates,
                pseudo_goal_balance_categories=args.pseudo_goal_balance_categories,
                strict_labels=args.strict_labels,
                forbid_teacher_fallback=args.forbid_teacher_fallback,
                forbid_approx_fallback=args.forbid_approx_fallback,
                require_hindsight_or_oracle=args.require_hindsight_or_oracle,
                min_label_frontiers=args.min_label_frontiers,
                allow_unvalidated_geodesic=args.allow_unvalidated_geodesic,
                episode_summary=episode_summary,
                episode_summary_path=episode_summary_path,
                allow_failed_hindsight_debug=args.allow_failed_hindsight_debug,
            )
        except SkipSample as exc:
            if args.skip_unlabeled or args.skip_errors or args.strict_labels:
                skipped += 1
                skip_reason_counts[exc.reason] += 1
                continue
            raise
        except Exception:
            if args.skip_errors:
                skipped += 1
                skip_reason_counts["skipped_error"] += 1
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
            label_type = str(labeled_sample.get("labels", {}).get("label_type", "missing"))
            label_type_counts[label_type] += 1
            pseudo_goal_text = labeled_sample.get("labels", {}).get("pseudo_goal_text")
            if pseudo_goal_text is not None:
                pseudo_goal_category_counts[str(pseudo_goal_text)] += 1

    report = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "episode_summary_dir": args.episode_summary_dir,
        "label_mode": args.label_mode,
        "num_input_files": len(paths),
        "num_labeled_samples": count,
        "num_skipped_samples": skipped,
        "label_type_counts": dict(label_type_counts),
        "skip_reason_counts": dict(skip_reason_counts),
        "pseudo_goal_category_counts": dict(pseudo_goal_category_counts),
        "strict_labels": bool(args.strict_labels),
        "forbid_teacher_fallback": bool(args.forbid_teacher_fallback),
        "forbid_approx_fallback": bool(args.forbid_approx_fallback),
        "require_hindsight_or_oracle": bool(args.require_hindsight_or_oracle),
    }
    if (args.strict_labels or args.forbid_teacher_fallback) and label_type_counts.get("teacher_debug_fallback", 0) > 0:
        raise RuntimeError("strict labeling produced teacher_debug_fallback labels")
    if (args.strict_labels or args.forbid_approx_fallback) and label_type_counts.get("oracle_approx_map_fallback", 0) > 0:
        raise RuntimeError("strict labeling produced oracle_approx_map_fallback labels")

    print(f"labeled {count} samples -> {args.output_dir}")
    print("label_type_counts:", dict(label_type_counts))
    print("skip_reason_counts:", dict(skip_reason_counts))
    if skipped:
        print(f"skipped {skipped} samples")
    if args.write_label_report:
        report_path = Path(args.write_label_report)
        if not report_path.is_absolute():
            report_path = output_dir / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote label report: {report_path}")


if __name__ == "__main__":
    main()
