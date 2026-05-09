import argparse

import numpy as np

from gnn_nav.dataset import safe_torch_load


def _shape(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return tuple(np.asarray(value).shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    sample = safe_torch_load(args.path, map_location="cpu")
    metadata = sample.get("metadata", {})
    goal = sample.get("goal", {})
    frontier = sample.get("frontier", {})
    teacher = sample.get("teacher", {})
    scenegraph = sample.get("scenegraph", {})
    maps = sample.get("maps", {})

    valid_frontiers = frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", []))
    teacher_scores = teacher.get("total_scores", teacher.get("sgnav_scores", None))
    n_frontiers = 0 if valid_frontiers is None else len(np.asarray(valid_frontiers).reshape(-1, 2))
    n_teacher = None if teacher_scores is None else len(np.asarray(teacher_scores).reshape(-1))

    print("path:", args.path)
    print("version:", sample.get("version"))
    print("scene_id:", metadata.get("scene_id"))
    print("episode_id:", metadata.get("episode_id"))
    print("step_id:", metadata.get("step_id"))
    print("goal:", goal.get("object_category_sg", goal.get("object_category_raw", metadata.get("goal_text"))))
    print("valid_frontiers:", n_frontiers)
    print("selected_valid_idx:", frontier.get("selected_valid_idx", teacher.get("selected_valid_idx")))
    print("objects:", len(scenegraph.get("objects", [])))
    print("rooms:", len(scenegraph.get("rooms", [])))
    print("teacher_score_shape:", _shape(teacher_scores))
    print("frontier_shape:", _shape(valid_frontiers))
    print("maps_available:", sorted(k for k, v in maps.items() if v is not None))
    print("oracle_label_type:", sample.get("oracle", {}).get("label_type"))
    print("labels_available:", sorted(sample.get("labels", {}).keys()))
    if n_teacher is not None and n_teacher != n_frontiers:
        raise ValueError(f"teacher score count {n_teacher} != frontier count {n_frontiers}")


if __name__ == "__main__":
    main()

