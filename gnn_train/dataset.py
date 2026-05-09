import glob
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from gnn_nav.dataset import safe_torch_load
from gnn_train.graph_schema import SparseDecisionGraph


def graph_from_payload(sample):
    graph_payload = sample["graph"]
    if isinstance(graph_payload, SparseDecisionGraph):
        return graph_payload
    return SparseDecisionGraph(
        node_features=graph_payload["node_features"],
        edge_index=graph_payload["edge_index"],
        edge_attr=graph_payload["edge_attr"],
        node_texts=graph_payload.get("node_texts", {}),
        frontier_centers_rc=graph_payload.get("frontier_centers_rc", None),
        metadata=sample.get("metadata", {}),
    )


class FrontierGraphDataset(Dataset):
    def __init__(self, graph_dir, max_samples: Optional[int] = None, require_labels: bool = True):
        self.paths = sorted(glob.glob(os.path.join(graph_dir, "**", "*.pt"), recursive=True))
        if max_samples is not None:
            self.paths = self.paths[: int(max_samples)]
        self.require_labels = bool(require_labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = safe_torch_load(self.paths[idx], map_location="cpu")
        if self.require_labels and "labels" not in sample:
            raise ValueError(f"Missing labels: {self.paths[idx]}")
        graph = graph_from_payload(sample)
        labels = sample.get("labels", {})
        for key in ["frontier_y_soft", "frontier_cost", "teacher_y_soft", "teacher_scores"]:
            if key in labels and not torch.is_tensor(labels[key]):
                labels[key] = torch.tensor(labels[key], dtype=torch.float32)
        return {
            "graph": graph,
            "labels": labels,
            "metadata": sample.get("metadata", {}),
            "goal": sample.get("goal", {}),
            "frontier": sample.get("frontier", {}),
            "path": self.paths[idx],
        }


def collate_single(batch):
    assert len(batch) == 1
    return batch[0]

