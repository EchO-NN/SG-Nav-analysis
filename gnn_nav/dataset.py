import glob
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from gnn_nav.graph_schema import SparseDecisionGraph


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class GNNStepDataset(Dataset):
    def __init__(self, data_dir: str, require_labels: bool = True, max_samples: Optional[int] = None):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if max_samples is not None:
            self.paths = self.paths[: int(max_samples)]
        self.require_labels = bool(require_labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = safe_torch_load(self.paths[idx], map_location="cpu")
        graph_payload = sample["graph"]
        if isinstance(graph_payload, SparseDecisionGraph):
            graph = graph_payload
        else:
            graph = SparseDecisionGraph(
                node_features=graph_payload["node_features"],
                edge_index=graph_payload["edge_index"],
                edge_attr=graph_payload["edge_attr"],
                node_texts=graph_payload.get("node_texts", {}),
                frontier_centers_rc=graph_payload.get("frontier_centers_rc", None),
                metadata=sample.get("metadata", {}),
            )

        labels = sample.get("labels", {})
        if self.require_labels and "frontier_y_soft" not in labels:
            raise ValueError(f"Missing frontier labels: {self.paths[idx]}")

        return {
            "graph": graph,
            "labels": labels,
            "metadata": sample.get("metadata", {}),
            "teacher": sample.get("teacher", {}),
            "path": self.paths[idx],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--no_labels", action="store_true")
    args = parser.parse_args()
    dataset = GNNStepDataset(args.data_dir, require_labels=not args.no_labels)
    print("samples", len(dataset))
    if len(dataset) > 0:
        item = dataset[0]
        print("node_shapes", {k: tuple(v.shape) for k, v in item["graph"].node_features.items()})
