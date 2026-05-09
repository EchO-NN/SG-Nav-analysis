import os
import re

import numpy as np
import torch


def sanitize(x):
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_\-]", "_", x)
    return x[:120]


class GNNReplayLogger:
    def __init__(self, log_dir: str, enabled: bool = False):
        self.log_dir = log_dir
        self.enabled = bool(enabled)
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)

    def save_step(self, graph, frontier_clusters, goal_text, metadata, teacher_scores=None, selected_idx=None):
        if not self.enabled or len(frontier_clusters) == 0:
            return None

        os.makedirs(self.log_dir, exist_ok=True)
        scene_id = sanitize(metadata.get("scene_id", "unknown_scene"))
        episode_id = sanitize(metadata.get("episode_id", "unknown_episode"))
        step_id = int(metadata.get("step_id", 0))
        path = os.path.join(self.log_dir, f"{scene_id}_{episode_id}_{step_id:06d}.pt")

        sample = {
            "graph": {
                "node_features": graph.node_features,
                "edge_index": graph.edge_index,
                "edge_attr": graph.edge_attr,
                "node_texts": graph.node_texts,
                "frontier_centers_rc": graph.frontier_centers_rc,
            },
            "frontier": {
                "centers_rc": np.stack([c.center_rc for c in frontier_clusters]),
                "sizes": np.asarray([c.size for c in frontier_clusters], dtype=np.int64),
                "mean_path_dist": np.asarray([c.mean_path_dist for c in frontier_clusters], dtype=np.float32),
                "distance_inverse": np.asarray([c.distance_inverse for c in frontier_clusters], dtype=np.float32),
                "unknown_score": np.asarray([c.unknown_score for c in frontier_clusters], dtype=np.float32),
            },
            "metadata": dict(metadata),
            "teacher": {
                "sgnav_scores": teacher_scores,
                "selected_frontier_idx": selected_idx,
            },
        }

        tmp_path = path + ".tmp"
        torch.save(sample, tmp_path)
        os.replace(tmp_path, path)
        return path


if __name__ == "__main__":
    print("sanitize", sanitize("scene/abc 123"))
