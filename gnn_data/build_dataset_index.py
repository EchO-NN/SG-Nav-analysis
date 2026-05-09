import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

from gnn_nav.dataset import safe_torch_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pt"), recursive=True))
    if args.max_samples is not None:
        paths = paths[: args.max_samples]

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "version",
                "scene_id",
                "episode_id",
                "step_id",
                "goal",
                "num_frontiers",
                "num_objects",
                "label_type",
                "label_entropy",
            ],
        )
        writer.writeheader()
        for path in paths:
            sample = safe_torch_load(path, map_location="cpu")
            metadata = sample.get("metadata", {})
            frontier = sample.get("frontier", {})
            centers = frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", []))
            labels = sample.get("labels", {})
            y = labels.get("frontier_y_soft", None)
            if y is not None:
                if hasattr(y, "detach"):
                    y = y.detach().cpu().numpy()
                y = np.asarray(y, dtype=np.float32)
                entropy = float(-(y * np.log(y + 1e-8)).sum())
            else:
                entropy = ""
            writer.writerow(
                {
                    "path": path,
                    "version": sample.get("version", ""),
                    "scene_id": metadata.get("scene_id", ""),
                    "episode_id": metadata.get("episode_id", ""),
                    "step_id": metadata.get("step_id", ""),
                    "goal": sample.get("goal", {}).get("object_category_sg", metadata.get("goal_text", "")),
                    "num_frontiers": len(np.asarray(centers).reshape(-1, 2)) if len(centers) else 0,
                    "num_objects": len(sample.get("scenegraph", {}).get("objects", [])),
                    "label_type": labels.get("label_type", sample.get("oracle", {}).get("label_type", "")),
                    "label_entropy": entropy,
                }
            )
    print(f"wrote index: {args.output_csv}")


if __name__ == "__main__":
    main()

