from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_nav.graph_schema import NODE_FRONTIER, NODE_GOAL, NODE_OBJECT, NODE_ROOM
from gnn_nav.graph_utils import edge_type_to_str


def infer_model_cfg_from_graph(graph, hidden_dim=256, num_layers=3, dropout=0.1):
    return {
        "object_dim": int(graph.node_features[NODE_OBJECT].shape[1]),
        "room_dim": int(graph.node_features[NODE_ROOM].shape[1]),
        "frontier_dim": int(graph.node_features[NODE_FRONTIER].shape[1]),
        "goal_dim": int(graph.node_features[NODE_GOAL].shape[1]),
        "edge_attr_dims": {etype: int(attr.shape[1]) for etype, attr in graph.edge_attr.items()},
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }


def _restore_edge_type(key):
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        parts = key.split("__")
        if len(parts) == 3:
            return tuple(parts)
    return key


class HeteroGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_attr_dims, node_types):
        super().__init__()
        self.message_mlps = nn.ModuleDict()
        self.edge_keys = {}
        for etype, edge_dim in edge_attr_dims.items():
            etype = _restore_edge_type(etype)
            key = edge_type_to_str(etype)
            self.edge_keys[etype] = key
            self.message_mlps[key] = nn.Sequential(
                nn.Linear(hidden_dim + int(edge_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.update_mlps = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for ntype in node_types:
            self.update_mlps[ntype] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norms[ntype] = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        agg = {ntype: torch.zeros_like(h[ntype]) for ntype in h}
        deg = {
            ntype: torch.zeros((h[ntype].shape[0], 1), dtype=h[ntype].dtype, device=h[ntype].device)
            for ntype in h
        }

        for etype_raw, eidx in edge_index.items():
            etype = _restore_edge_type(etype_raw)
            if not isinstance(etype, tuple) or len(etype) != 3:
                continue
            src_type, _, dst_type = etype
            if eidx.numel() == 0 or src_type not in h or dst_type not in h:
                continue
            if h[src_type].shape[0] == 0 or h[dst_type].shape[0] == 0:
                continue

            key = edge_type_to_str(etype)
            if key not in self.message_mlps:
                continue

            device = h[src_type].device
            eidx = eidx.to(device)
            src_idx = eidx[0]
            dst_idx = eidx[1]
            src_h = h[src_type][src_idx]
            eattr = edge_attr[etype_raw].to(device)
            msg_input = torch.cat([src_h, eattr], dim=-1)
            msg = self.message_mlps[key](msg_input)
            agg[dst_type].index_add_(0, dst_idx, msg)
            ones = torch.ones((dst_idx.shape[0], 1), dtype=msg.dtype, device=device)
            deg[dst_type].index_add_(0, dst_idx, ones)

        h_new = {}
        for ntype, h_old in h.items():
            if h_old.shape[0] == 0:
                h_new[ntype] = h_old
                continue
            agg_norm = agg[ntype] / deg[ntype].clamp_min(1.0)
            update_input = torch.cat([h_old, agg_norm], dim=-1)
            delta = self.update_mlps[ntype](update_input)
            h_new[ntype] = self.norms[ntype](h_old + delta)
        return h_new


class GoalConditionedGraphNet(nn.Module):
    def __init__(
        self,
        object_dim: int,
        room_dim: int,
        frontier_dim: int,
        goal_dim: int,
        edge_attr_dims: Dict,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_types = [NODE_OBJECT, NODE_ROOM, NODE_FRONTIER, NODE_GOAL]
        self.node_proj = nn.ModuleDict(
            {
                NODE_OBJECT: nn.Linear(object_dim, hidden_dim),
                NODE_ROOM: nn.Linear(room_dim, hidden_dim),
                NODE_FRONTIER: nn.Linear(frontier_dim, hidden_dim),
                NODE_GOAL: nn.Linear(goal_dim, hidden_dim),
            }
        )
        self.layers = nn.ModuleList(
            [
                HeteroGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_attr_dims=edge_attr_dims,
                    node_types=self.node_types,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.frontier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph):
        h = {}
        for ntype in self.node_types:
            if ntype not in graph.node_features:
                continue
            h[ntype] = self.node_proj[ntype](graph.node_features[ntype])

        if NODE_FRONTIER not in h:
            raise ValueError("SparseDecisionGraph is missing frontier node features")

        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_attr)
            h = {key: self.dropout(value) for key, value in h.items()}

        frontier_logits = self.frontier_head(h[NODE_FRONTIER]).squeeze(-1)
        return {
            "frontier_logits": frontier_logits,
            "node_embeddings": h,
        }


def soft_cross_entropy(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=0)
    return -(target_probs * log_probs).sum()


def distillation_kl(student_logits, teacher_scores, temperature=2.0):
    teacher_probs = torch.softmax(teacher_scores / temperature, dim=0)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=0)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * temperature * temperature


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    from gnn_nav.frontier_clustering import FrontierCluster
    from gnn_nav.sparse_graph_builder import DEFAULT_ROOM_NAMES, SparseDecisionGraphBuilder
    from gnn_nav.text_encoder import TextEmbeddingCache

    scenegraph = SimpleNamespace(nodes=[SimpleNamespace(caption="chair", center=[20, 30])])
    clusters = [
        FrontierCluster(0, np.array([35, 25]), np.zeros((3, 2)), 3, 2.0, 2.5, 3.0, 0.91),
        FrontierCluster(1, np.array([70, 75]), np.zeros((4, 2)), 4, 4.0, 5.0, 6.0, 0.66),
    ]
    room_map = torch.zeros((1, len(DEFAULT_ROOM_NAMES), 100, 100), dtype=torch.float32)
    room_map[0, 0] = 1.0
    builder = SparseDecisionGraphBuilder(
        TextEmbeddingCache(cache_path="", dim=16, backend="fallback"),
        map_resolution_cm=5,
        map_size=100,
        room_names=DEFAULT_ROOM_NAMES,
    )
    graph = builder.build(
        scenegraph,
        clusters,
        "chair",
        torch.tensor([2.5, 2.5, 0.0]),
        torch.zeros((1, 1, 100, 100)),
        torch.zeros((1, 1, 100, 100)),
        room_map,
    )
    cfg = infer_model_cfg_from_graph(graph, hidden_dim=32, num_layers=2)
    model = GoalConditionedGraphNet(**cfg)
    out = model(graph)
    print("frontier_logits_shape", tuple(out["frontier_logits"].shape))
