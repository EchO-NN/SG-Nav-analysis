import argparse
from pathlib import Path

import numpy as np

from gnn_nav.dataset import safe_torch_load


def _np(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _shape(value):
    arr = _np(value)
    if arr is None:
        return None
    return tuple(arr.shape)


def _is_present(value):
    arr = _np(value)
    return arr is not None and arr.size > 0


def validate_summary(summary, strict=False):
    errors = []
    if summary.get("version") != "episode_summary_v1":
        errors.append(f"bad version: {summary.get('version')}")
    metadata = summary.get("metadata", {})
    target_goal = summary.get("target_goal", {})
    final_maps = summary.get("final_maps", {})
    for key in ["dataset", "split", "scene_id", "episode_id", "goal_text", "success", "num_steps"]:
        if key not in metadata:
            errors.append(f"missing metadata.{key}")
    for key in ["goal_text", "found_goal_rc", "found_goal_world", "found_step", "source", "confidence"]:
        if key not in target_goal:
            errors.append(f"missing target_goal.{key}")
    for key in ["full_map", "free_map", "room_map"]:
        if key not in final_maps:
            errors.append(f"missing final_maps.{key}")
    if "trajectory" not in summary:
        errors.append("missing trajectory")
    if "discovered_objects" not in summary:
        errors.append("missing discovered_objects")
    if "fallback" not in summary:
        errors.append("missing fallback")
    if strict:
        if metadata.get("success") and target_goal.get("source") == "confirmed_candidate":
            if not _is_present(target_goal.get("found_goal_rc")):
                errors.append("successful confirmed_candidate episode has no found_goal_rc")
        if final_maps.get("free_map") is None:
            errors.append("missing final_maps.free_map")
        for idx, obj in enumerate(summary.get("discovered_objects", [])):
            for key in [
                "center_rc",
                "confidence",
                "observed_count",
                "first_seen_step",
                "last_seen_step",
                "stable",
                "rejected_candidate",
            ]:
                if key not in obj:
                    errors.append(f"discovered_objects[{idx}] missing {key}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summary = safe_torch_load(args.path, map_location="cpu")
    metadata = summary.get("metadata", {})
    target_goal = summary.get("target_goal", {})
    final_maps = summary.get("final_maps", {})
    objects = summary.get("discovered_objects", [])
    fallback = summary.get("fallback", {})
    debug = summary.get("debug", {})
    rejected = debug.get("rejected_goal_candidates", [])
    stable_objects = [obj for obj in objects if obj.get("stable")]

    print("path:", args.path)
    print("version:", summary.get("version"))
    print("scene_id:", metadata.get("scene_id"))
    print("episode_id:", metadata.get("episode_id"))
    print("goal_text:", metadata.get("goal_text"))
    print("success:", metadata.get("success"))
    print("spl:", metadata.get("spl"))
    print("softspl:", metadata.get("softspl"))
    print("num_steps:", metadata.get("num_steps"))
    print("found_goal_source:", target_goal.get("source"))
    print("has_found_goal_rc:", _is_present(target_goal.get("found_goal_rc")))
    print("found_goal_rc_shape:", _shape(target_goal.get("found_goal_rc")))
    print("found_goal_world_shape:", _shape(target_goal.get("found_goal_world")))
    print("final_full_map_shape:", _shape(final_maps.get("full_map")))
    print("final_free_map_shape:", _shape(final_maps.get("free_map")))
    print("final_room_map_shape:", _shape(final_maps.get("room_map")))
    print("num_trajectory_steps:", len(summary.get("trajectory", [])))
    print("num_discovered_objects:", len(objects))
    print("num_stable_objects:", len(stable_objects))
    print("num_rejected_candidates:", len(rejected))
    print("num_fallback_calls:", fallback.get("num_fallback_calls", 0))
    print(
        "flat_aliases_present:",
        all(
            key in summary
            for key in [
                "episode_id",
                "scene_id",
                "success",
                "target_goal_text",
                "found_goal_rc",
                "found_goal_world",
                "final_full_map",
                "final_free_map",
                "trajectory",
                "discovered_objects",
                "rejected_candidates",
                "fallback_queries",
            ]
        ),
    )

    errors = validate_summary(summary, strict=args.strict)
    if errors:
        print("validation_errors:")
        for err in errors:
            print(" -", err)
        raise SystemExit(1)
    print("validation: ok")


if __name__ == "__main__":
    main()
