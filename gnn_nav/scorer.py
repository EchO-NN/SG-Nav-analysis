import os
import time

import numpy as np
import torch

from gnn_nav.model import GoalConditionedGraphNet


class GNNFrontierScorer:
    def __init__(
        self,
        checkpoint_path,
        builder,
        device="cuda",
        fallback_to_distance=True,
    ):
        self.builder = builder
        self.device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        self.fallback_to_distance = bool(fallback_to_distance)
        self.model = None
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None and str(checkpoint_path) not in ["", "none", "None"]:
            if os.path.exists(str(checkpoint_path)):
                ckpt = torch.load(str(checkpoint_path), map_location=self.device)
                model_cfg = ckpt["model_cfg"]
                self.model = GoalConditionedGraphNet(**model_cfg).to(self.device)
                self.model.load_state_dict(ckpt["model"])
                self.model.eval()
            else:
                print(f"[GNN] checkpoint not found, using fallback: {checkpoint_path}")

    @torch.no_grad()
    def score(
        self,
        scenegraph,
        frontier_clusters,
        goal_text,
        agent_pose,
        full_map,
        free_map,
        room_map,
        traversible=None,
        cur_start=None,
        current_step: int = 0,
        candidate_goals=None,
        return_graph_info: bool = False,
    ):
        if len(frontier_clusters) == 0:
            scores = np.zeros((0,), dtype=np.float32)
            if return_graph_info:
                return scores, {"num_object_nodes": 0, "num_frontiers": 0}
            return scores

        build_start = time.perf_counter()
        graph = self.builder.build(
            scenegraph=scenegraph,
            frontier_clusters=frontier_clusters,
            goal_text=goal_text,
            agent_pose=agent_pose,
            full_map=full_map,
            free_map=free_map,
            room_map=room_map,
            traversible=traversible,
            cur_start=cur_start,
            current_step=current_step,
            candidate_goals=candidate_goals,
        )
        sparse_graph_build_time = time.perf_counter() - build_start
        graph_info = dict(getattr(graph, "metadata", {}) or {})
        graph_info["sparse_graph_build_time"] = float(sparse_graph_build_time)

        if self.model is None:
            if self.fallback_to_distance:
                scores = np.asarray([c.distance_inverse for c in frontier_clusters], dtype=np.float32)
            else:
                scores = np.zeros((len(frontier_clusters),), dtype=np.float32)
            graph_info["gnn_forward_time"] = 0.0
            if return_graph_info:
                return scores, graph_info
            return scores

        graph = graph.to(self.device)
        forward_start = time.perf_counter()
        out = self.model(graph)
        graph_info["gnn_forward_time"] = float(time.perf_counter() - forward_start)
        logits = out["frontier_logits"].detach().cpu().numpy()
        scores = logits.astype(np.float32)
        if return_graph_info:
            return scores, graph_info
        return scores


if __name__ == "__main__":
    from types import SimpleNamespace

    from gnn_nav.frontier_clustering import FrontierCluster
    from gnn_nav.sparse_graph_builder import DEFAULT_ROOM_NAMES, SparseDecisionGraphBuilder
    from gnn_nav.text_encoder import TextEmbeddingCache

    scenegraph = SimpleNamespace(nodes=[])
    clusters = [
        FrontierCluster(0, np.array([10, 20]), np.zeros((3, 2)), 3, 2.0, 2.5, 3.0, 0.91),
        FrontierCluster(1, np.array([50, 60]), np.zeros((4, 2)), 4, 4.0, 5.0, 6.0, 0.66),
    ]
    room_map = torch.zeros((1, len(DEFAULT_ROOM_NAMES), 80, 80), dtype=torch.float32)
    builder = SparseDecisionGraphBuilder(
        TextEmbeddingCache(cache_path="", dim=16, backend="fallback"),
        map_resolution_cm=5,
        map_size=80,
        room_names=DEFAULT_ROOM_NAMES,
    )
    scorer = GNNFrontierScorer(checkpoint_path="none", builder=builder, device="cpu")
    scores = scorer.score(
        scenegraph,
        clusters,
        "chair",
        torch.tensor([2.0, 2.0, 0.0]),
        torch.zeros((1, 1, 80, 80)),
        torch.zeros((1, 1, 80, 80)),
        room_map,
    )
    print("fallback_scores", scores.tolist())
