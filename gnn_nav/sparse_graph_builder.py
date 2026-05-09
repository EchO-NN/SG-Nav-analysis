from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch

from gnn_nav.graph_schema import (
    EDGE_FRONTIER_GOAL,
    EDGE_FRONTIER_OBJECT,
    EDGE_FRONTIER_ROOM,
    EDGE_GOAL_FRONTIER,
    EDGE_GOAL_OBJECT,
    EDGE_GOAL_ROOM,
    EDGE_OBJECT_FRONTIER,
    EDGE_OBJECT_GOAL,
    EDGE_OBJECT_OBJECT,
    EDGE_OBJECT_ROOM,
    EDGE_ROOM_FRONTIER,
    EDGE_ROOM_GOAL,
    EDGE_ROOM_OBJECT,
    NODE_FRONTIER,
    NODE_GOAL,
    NODE_OBJECT,
    NODE_ROOM,
    SparseDecisionGraph,
)
from gnn_nav.graph_utils import (
    compute_heading_feature,
    compute_unknown_score,
    empty_edge,
    make_edge,
    normalize_rc,
    pose_to_map_rc,
    reverse_edge,
    sample_room_prob,
    to_numpy_room_map,
)


DEFAULT_ROOM_NAMES = [
    "bedroom",
    "living room",
    "bathroom",
    "kitchen",
    "dining room",
    "office room",
    "gym",
    "lounge",
    "laundry room",
]


def _as_flat_array(value) -> Optional[np.ndarray]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return arr
    except Exception:
        return None
    return None


def _first_float(value) -> Optional[float]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            return float(np.max(arr))
    except Exception:
        return None
    return None


def _dict_get(obj, keys):
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def safe_get_caption(node) -> str:
    if isinstance(node, dict):
        value = _dict_get(node, ["caption", "category", "class_name", "label"])
        if value is not None:
            return str(value)
        captions = node.get("captions")
        if captions:
            try:
                return str(captions[0])
            except Exception:
                pass

    for attr in ["caption", "category", "class_name", "label"]:
        if hasattr(node, attr):
            val = getattr(node, attr)
            if val is not None:
                return str(val)

    obj = getattr(node, "object", None)
    if isinstance(obj, dict):
        value = _dict_get(obj, ["caption", "category", "class_name", "label"])
        if value is not None:
            return str(value)
        captions = obj.get("captions")
        if captions:
            try:
                return str(captions[0])
            except Exception:
                pass

    return "object"


def _center_from_object_dict(obj, map_resolution_cm: float, map_size: int) -> Optional[np.ndarray]:
    value = _dict_get(obj, ["center_rc", "map_center"])
    arr = _as_flat_array(value)
    if arr is not None:
        return arr[:2]

    value = _dict_get(obj, ["center", "xy"])
    arr = _as_flat_array(value)
    if arr is not None:
        return np.array([arr[1], arr[0]], dtype=np.float32)

    pcd = obj.get("pcd") if isinstance(obj, dict) else None
    if pcd is not None and hasattr(pcd, "points"):
        try:
            points = np.asarray(pcd.points, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 2:
                center = points[:, :2].mean(axis=0)
                col = center[0] * 100.0 / float(map_resolution_cm)
                row = map_size - 1 - center[1] * 100.0 / float(map_resolution_cm)
                return np.array([row, col], dtype=np.float32)
        except Exception:
            return None
    return None


def safe_get_center_rc(node, map_size: int, map_resolution_cm: float = 5.0) -> Optional[np.ndarray]:
    if isinstance(node, dict):
        center = _center_from_object_dict(node, map_resolution_cm, map_size)
        if center is not None:
            return center

    for attr in ["center_rc", "map_center"]:
        if hasattr(node, attr):
            arr = _as_flat_array(getattr(node, attr))
            if arr is not None:
                return arr[:2]

    for attr in ["center", "xy"]:
        if hasattr(node, attr):
            arr = _as_flat_array(getattr(node, attr))
            if arr is not None:
                # SG-Nav ObjectNode.center is stored as [x, y].
                return np.array([arr[1], arr[0]], dtype=np.float32)

    obj = getattr(node, "object", None)
    if isinstance(obj, dict):
        return _center_from_object_dict(obj, map_resolution_cm, map_size)

    return None


def safe_get_confidence(node) -> float:
    if isinstance(node, dict):
        value = _dict_get(node, ["conf", "confidence", "score"])
        out = _first_float(value)
        if out is not None:
            return out

    obj = getattr(node, "object", None)
    if isinstance(obj, dict):
        value = _dict_get(obj, ["conf", "confidence", "score"])
        out = _first_float(value)
        if out is not None:
            return out

    for attr in ["confidence", "score"]:
        if hasattr(node, attr):
            out = _first_float(getattr(node, attr))
            if out is not None:
                return out
    return 1.0


def safe_get_observed_count(node) -> float:
    obj = node if isinstance(node, dict) else getattr(node, "object", None)
    if isinstance(obj, dict):
        for key in ["num_detections", "observed_count", "count"]:
            if key in obj:
                out = _first_float(obj[key])
                if out is not None:
                    return out
        if "image_idx" in obj and obj["image_idx"] is not None:
            try:
                return float(len(obj["image_idx"]))
            except Exception:
                pass
    return 1.0


class SparseDecisionGraphBuilder:
    def __init__(
        self,
        text_encoder,
        map_resolution_cm: float,
        map_size: int,
        room_names: Optional[List[str]] = None,
        max_objects: int = 100,
        object_knn_k: int = 6,
        frontier_knn_k: int = 8,
        object_radius_m: float = 2.5,
        frontier_radius_m: float = 4.0,
        device: str = "cpu",
    ):
        self.text_encoder = text_encoder
        self.map_resolution_cm = float(map_resolution_cm)
        self.map_size = int(map_size)
        self.room_names = list(room_names or DEFAULT_ROOM_NAMES)
        self.max_objects = int(max_objects)
        self.object_knn_k = int(object_knn_k)
        self.frontier_knn_k = int(frontier_knn_k)
        self.object_radius_m = float(object_radius_m)
        self.frontier_radius_m = float(frontier_radius_m)
        self.device = device
        self.text_dim = int(getattr(text_encoder, "dim", 384))
        self.room_count = len(self.room_names)
        self.object_dim = self.text_dim + self.room_count + 6
        self.room_dim = self.text_dim + 4
        self.frontier_dim = self.room_count + 8
        self.goal_dim = self.text_dim

    def _iter_raw_object_nodes(self, scenegraph) -> Iterable[Any]:
        for name in ["nodes", "object_nodes", "nodes_list"]:
            raw = getattr(scenegraph, name, None)
            if raw is not None:
                try:
                    if len(raw) > 0:
                        return raw
                except Exception:
                    pass

        raw = getattr(scenegraph, "objects", None)
        if raw is not None:
            return raw
        return []

    def build_object_nodes(self, scenegraph, goal_text, room_map, agent_pose):
        object_feats = []
        object_texts = []
        object_centers = []
        object_confidences = []
        goal_emb = self.text_encoder.encode(goal_text)
        agent_rc = pose_to_map_rc(agent_pose, self.map_size, self.map_resolution_cm)

        for node in self._iter_raw_object_nodes(scenegraph):
            caption = safe_get_caption(node)
            center_rc = safe_get_center_rc(node, self.map_size, self.map_resolution_cm)
            if center_rc is None:
                continue

            center_rc = np.asarray(center_rc, dtype=np.float32).reshape(-1)[:2]
            if center_rc.size < 2 or not np.all(np.isfinite(center_rc)):
                continue
            if center_rc[0] < 0 or center_rc[1] < 0:
                continue
            if center_rc[0] >= self.map_size or center_rc[1] >= self.map_size:
                continue

            cat_emb = self.text_encoder.encode(caption)
            goal_sim = float(torch.dot(cat_emb, goal_emb).item())
            conf = float(safe_get_confidence(node))
            obs_count = float(safe_get_observed_count(node))
            room_prob = sample_room_prob(room_map, center_rc)
            if room_prob.shape[0] != self.room_count:
                room_prob = self._fit_room_prob(room_prob)
            dist_norm = float(np.linalg.norm(center_rc - agent_rc) / max(self.map_size, 1))

            feat = torch.cat(
                [
                    cat_emb,
                    torch.tensor([goal_sim], dtype=torch.float32),
                    torch.tensor(normalize_rc(center_rc, self.map_size), dtype=torch.float32),
                    torch.tensor([conf], dtype=torch.float32),
                    torch.tensor([np.log1p(max(obs_count, 0.0))], dtype=torch.float32),
                    torch.tensor(room_prob, dtype=torch.float32),
                    torch.tensor([dist_norm], dtype=torch.float32),
                ]
            )
            object_feats.append(feat)
            object_texts.append(caption)
            object_centers.append(center_rc)
            object_confidences.append(conf)

            if len(object_feats) >= self.max_objects:
                break

        if len(object_feats) == 0:
            x_obj = torch.zeros((0, self.object_dim), dtype=torch.float32)
            centers = np.zeros((0, 2), dtype=np.float32)
            confidences = np.zeros((0,), dtype=np.float32)
        else:
            x_obj = torch.stack(object_feats, dim=0)
            centers = np.stack(object_centers, axis=0).astype(np.float32)
            confidences = np.asarray(object_confidences, dtype=np.float32)
        return x_obj, object_texts, centers, confidences

    def _fit_room_prob(self, room_prob):
        out = np.zeros((self.room_count,), dtype=np.float32)
        n = min(len(room_prob), self.room_count)
        if n > 0:
            out[:n] = room_prob[:n]
        total = float(out.sum())
        if total > 1e-6:
            out = out / total
        return out

    def build_room_nodes(self, room_map, goal_text):
        goal_emb = self.text_encoder.encode(goal_text)
        room_np = to_numpy_room_map(room_map)
        room_feats = []
        for rid, room_name in enumerate(self.room_names):
            room_emb = self.text_encoder.encode(room_name)
            sim = float(torch.dot(room_emb, goal_emb).item())
            if rid < room_np.shape[0]:
                prob_map = room_np[rid]
                area = float((prob_map > 0.2).sum()) / float(prob_map.size)
                max_prob = float(prob_map.max()) if prob_map.size else 0.0
                mean_prob = float(prob_map.mean()) if prob_map.size else 0.0
            else:
                area = 0.0
                max_prob = 0.0
                mean_prob = 0.0
            room_feats.append(
                torch.cat(
                    [
                        room_emb,
                        torch.tensor([sim, area, max_prob, mean_prob], dtype=torch.float32),
                    ]
                )
            )
        if len(room_feats) == 0:
            return torch.zeros((0, self.room_dim), dtype=torch.float32)
        return torch.stack(room_feats, dim=0)

    def build_frontier_nodes(self, frontier_clusters, room_map, full_map, free_map, agent_pose):
        feats = []
        centers = []
        for cluster in frontier_clusters:
            center = np.asarray(cluster.center_rc, dtype=np.float32).reshape(-1)[:2]
            room_prob = sample_room_prob(room_map, center)
            if room_prob.shape[0] != self.room_count:
                room_prob = self._fit_room_prob(room_prob)
            unknown = compute_unknown_score(full_map, free_map, center)
            cluster.unknown_score = unknown
            heading_sin, heading_cos = compute_heading_feature(
                agent_pose, center, self.map_size, self.map_resolution_cm
            )
            size_norm = float(np.log1p(max(cluster.size, 0)) / 10.0)
            path_norm = float(cluster.mean_path_dist / 20.0)
            feat = torch.cat(
                [
                    torch.tensor(normalize_rc(center, self.map_size), dtype=torch.float32),
                    torch.tensor(
                        [size_norm, path_norm, cluster.distance_inverse, unknown],
                        dtype=torch.float32,
                    ),
                    torch.tensor(room_prob, dtype=torch.float32),
                    torch.tensor([heading_sin, heading_cos], dtype=torch.float32),
                ]
            )
            feats.append(feat)
            centers.append(center)

        if len(feats) == 0:
            return torch.zeros((0, self.frontier_dim), dtype=torch.float32), np.zeros((0, 2), dtype=np.float32)
        return torch.stack(feats, dim=0), np.stack(centers, axis=0).astype(np.float32)

    def build_goal_node(self, goal_text):
        return self.text_encoder.encode(goal_text).unsqueeze(0)

    def build_object_object_edges(self, object_centers, object_texts, room_map):
        count = len(object_centers)
        if count <= 1:
            return empty_edge(attr_dim=5)
        src, dst, attr = [], [], []
        text_embs = self.text_encoder.encode_many(object_texts)
        for i in range(count):
            dists = np.linalg.norm(object_centers - object_centers[i], axis=1)
            order = np.argsort(dists)
            added = 0
            for j in order:
                if i == j:
                    continue
                dist_m = float(dists[j] * self.map_resolution_cm / 100.0)
                if dist_m > self.object_radius_m:
                    continue
                dxdy = (object_centers[j] - object_centers[i]) / float(self.map_size)
                room_i = self._fit_room_prob(sample_room_prob(room_map, object_centers[i]))
                room_j = self._fit_room_prob(sample_room_prob(room_map, object_centers[j]))
                same_room = float(np.dot(room_i, room_j))
                sim = float(torch.dot(text_embs[i], text_embs[j]).item())
                src.append(i)
                dst.append(int(j))
                attr.append([float(dxdy[0]), float(dxdy[1]), dist_m / 10.0, same_room, sim])
                added += 1
                if added >= self.object_knn_k:
                    break
        return make_edge(src, dst, attr, attr_dim=5)

    def build_object_room_edges(self, object_centers, room_map, topk=2):
        src, dst, attr = [], [], []
        for oid, center in enumerate(object_centers):
            room_prob = self._fit_room_prob(sample_room_prob(room_map, center))
            top_rooms = np.argsort(-room_prob)[:topk]
            for rid in top_rooms:
                prob = float(room_prob[rid])
                if prob <= 0:
                    continue
                src.append(int(oid))
                dst.append(int(rid))
                attr.append([prob])
        return make_edge(src, dst, attr, attr_dim=1)

    def build_frontier_object_edges(self, frontier_centers, object_centers, room_map, object_confidences):
        frontier_count = len(frontier_centers)
        object_count = len(object_centers)
        if frontier_count == 0 or object_count == 0:
            return empty_edge(attr_dim=5)
        src, dst, attr = [], [], []
        for fid in range(frontier_count):
            dists = np.linalg.norm(object_centers - frontier_centers[fid], axis=1)
            order = np.argsort(dists)
            added = 0
            for oid in order:
                dist_m = float(dists[oid] * self.map_resolution_cm / 100.0)
                if dist_m > self.frontier_radius_m:
                    continue
                dxdy = (object_centers[oid] - frontier_centers[fid]) / float(self.map_size)
                room_f = self._fit_room_prob(sample_room_prob(room_map, frontier_centers[fid]))
                room_o = self._fit_room_prob(sample_room_prob(room_map, object_centers[oid]))
                same_room = float(np.dot(room_f, room_o))
                obj_conf = float(object_confidences[oid]) if len(object_confidences) > oid else 1.0
                src.append(fid)
                dst.append(int(oid))
                attr.append([float(dxdy[0]), float(dxdy[1]), dist_m / 10.0, same_room, obj_conf])
                added += 1
                if added >= self.frontier_knn_k:
                    break
        return make_edge(src, dst, attr, attr_dim=5)

    def build_frontier_room_edges(self, frontier_centers, room_map, topk=2):
        src, dst, attr = [], [], []
        for fid, center in enumerate(frontier_centers):
            room_prob = self._fit_room_prob(sample_room_prob(room_map, center))
            top_rooms = np.argsort(-room_prob)[:topk]
            for rid in top_rooms:
                prob = float(room_prob[rid])
                if prob <= 0:
                    continue
                src.append(int(fid))
                dst.append(int(rid))
                attr.append([prob])
        return make_edge(src, dst, attr, attr_dim=1)

    def build_goal_object_edges(self, goal_text, object_texts):
        if len(object_texts) == 0:
            return empty_edge(attr_dim=1)
        goal_emb = self.text_encoder.encode(goal_text)
        obj_embs = self.text_encoder.encode_many(object_texts)
        src, dst, attr = [], [], []
        for oid in range(len(object_texts)):
            sim = float(torch.dot(goal_emb, obj_embs[oid]).item())
            src.append(0)
            dst.append(int(oid))
            attr.append([sim])
        return make_edge(src, dst, attr, attr_dim=1)

    def build_goal_room_edges(self, goal_text):
        goal_emb = self.text_encoder.encode(goal_text)
        src, dst, attr = [], [], []
        for rid, room_name in enumerate(self.room_names):
            room_emb = self.text_encoder.encode(room_name)
            sim = float(torch.dot(goal_emb, room_emb).item())
            src.append(0)
            dst.append(int(rid))
            attr.append([sim])
        return make_edge(src, dst, attr, attr_dim=1)

    def build_goal_frontier_edges(self, frontier_clusters):
        src, dst, attr = [], [], []
        for fid, cluster in enumerate(frontier_clusters):
            src.append(0)
            dst.append(int(fid))
            attr.append([float(cluster.distance_inverse), float(cluster.unknown_score)])
        return make_edge(src, dst, attr, attr_dim=2)

    def build(
        self,
        scenegraph,
        frontier_clusters,
        goal_text: str,
        agent_pose,
        full_map,
        free_map,
        room_map,
        traversible=None,
        cur_start=None,
        current_step: int = 0,
        candidate_goals=None,
    ) -> SparseDecisionGraph:
        x_object, object_texts, object_centers, object_confidences = self.build_object_nodes(
            scenegraph, goal_text, room_map, agent_pose
        )
        x_room = self.build_room_nodes(room_map, goal_text)
        x_frontier, frontier_centers = self.build_frontier_nodes(
            frontier_clusters, room_map, full_map, free_map, agent_pose
        )
        x_goal = self.build_goal_node(goal_text)

        edge_index = {}
        edge_attr = {}

        edge_oo, attr_oo = self.build_object_object_edges(object_centers, object_texts, room_map)
        edge_index[EDGE_OBJECT_OBJECT] = edge_oo
        edge_attr[EDGE_OBJECT_OBJECT] = attr_oo

        edge_or, attr_or = self.build_object_room_edges(object_centers, room_map)
        edge_ro, attr_ro = reverse_edge(edge_or, attr_or)
        edge_index[EDGE_OBJECT_ROOM] = edge_or
        edge_attr[EDGE_OBJECT_ROOM] = attr_or
        edge_index[EDGE_ROOM_OBJECT] = edge_ro
        edge_attr[EDGE_ROOM_OBJECT] = attr_ro

        edge_fo, attr_fo = self.build_frontier_object_edges(
            frontier_centers, object_centers, room_map, object_confidences
        )
        edge_of, attr_of = reverse_edge(edge_fo, attr_fo)
        edge_index[EDGE_FRONTIER_OBJECT] = edge_fo
        edge_attr[EDGE_FRONTIER_OBJECT] = attr_fo
        edge_index[EDGE_OBJECT_FRONTIER] = edge_of
        edge_attr[EDGE_OBJECT_FRONTIER] = attr_of

        edge_fr, attr_fr = self.build_frontier_room_edges(frontier_centers, room_map)
        edge_rf, attr_rf = reverse_edge(edge_fr, attr_fr)
        edge_index[EDGE_FRONTIER_ROOM] = edge_fr
        edge_attr[EDGE_FRONTIER_ROOM] = attr_fr
        edge_index[EDGE_ROOM_FRONTIER] = edge_rf
        edge_attr[EDGE_ROOM_FRONTIER] = attr_rf

        edge_go, attr_go = self.build_goal_object_edges(goal_text, object_texts)
        edge_og, attr_og = reverse_edge(edge_go, attr_go)
        edge_index[EDGE_GOAL_OBJECT] = edge_go
        edge_attr[EDGE_GOAL_OBJECT] = attr_go
        edge_index[EDGE_OBJECT_GOAL] = edge_og
        edge_attr[EDGE_OBJECT_GOAL] = attr_og

        edge_gr, attr_gr = self.build_goal_room_edges(goal_text)
        edge_rg, attr_rg = reverse_edge(edge_gr, attr_gr)
        edge_index[EDGE_GOAL_ROOM] = edge_gr
        edge_attr[EDGE_GOAL_ROOM] = attr_gr
        edge_index[EDGE_ROOM_GOAL] = edge_rg
        edge_attr[EDGE_ROOM_GOAL] = attr_rg

        edge_gf, attr_gf = self.build_goal_frontier_edges(frontier_clusters)
        edge_fg, attr_fg = reverse_edge(edge_gf, attr_gf)
        edge_index[EDGE_GOAL_FRONTIER] = edge_gf
        edge_attr[EDGE_GOAL_FRONTIER] = attr_gf
        edge_index[EDGE_FRONTIER_GOAL] = edge_fg
        edge_attr[EDGE_FRONTIER_GOAL] = attr_fg

        return SparseDecisionGraph(
            node_features={
                NODE_OBJECT: x_object.to(self.device),
                NODE_ROOM: x_room.to(self.device),
                NODE_FRONTIER: x_frontier.to(self.device),
                NODE_GOAL: x_goal.to(self.device),
            },
            edge_index={key: value.to(self.device) for key, value in edge_index.items()},
            edge_attr={key: value.to(self.device) for key, value in edge_attr.items()},
            node_texts={
                NODE_OBJECT: object_texts,
                NODE_ROOM: list(self.room_names),
                NODE_GOAL: [goal_text],
            },
            frontier_centers_rc=torch.tensor(frontier_centers, dtype=torch.float32).to(self.device),
            metadata={
                "goal_text": goal_text,
                "current_step": int(current_step),
                "num_objects": int(x_object.shape[0]),
                "num_frontiers": int(x_frontier.shape[0]),
            },
        )


if __name__ == "__main__":
    from types import SimpleNamespace

    from gnn_nav.frontier_clustering import FrontierCluster
    from gnn_nav.text_encoder import TextEmbeddingCache

    nodes = [
        SimpleNamespace(caption="chair", center=[20, 30], score=0.8, object={"image_idx": [1, 2]}),
        SimpleNamespace(caption="table", center=[24, 32], score=0.7, object={"image_idx": [1]}),
    ]
    scenegraph = SimpleNamespace(nodes=nodes, rooms=DEFAULT_ROOM_NAMES)
    clusters = [
        FrontierCluster(0, np.array([35, 25]), np.zeros((3, 2)), 3, 2.0, 2.5, 3.0, 0.91),
        FrontierCluster(1, np.array([70, 75]), np.zeros((4, 2)), 4, 4.0, 5.0, 6.0, 0.66),
    ]
    room_map = torch.zeros((1, len(DEFAULT_ROOM_NAMES), 100, 100), dtype=torch.float32)
    room_map[0, 0, :, :] = 1.0
    builder = SparseDecisionGraphBuilder(
        text_encoder=TextEmbeddingCache(cache_path="", dim=16, backend="fallback"),
        map_resolution_cm=5,
        map_size=100,
        room_names=DEFAULT_ROOM_NAMES,
    )
    graph = builder.build(
        scenegraph=scenegraph,
        frontier_clusters=clusters,
        goal_text="chair",
        agent_pose=torch.tensor([2.5, 2.5, 0.0]),
        full_map=torch.zeros((1, 1, 100, 100)),
        free_map=torch.zeros((1, 1, 100, 100)),
        room_map=room_map,
    )
    print("node_shapes", {k: tuple(v.shape) for k, v in graph.node_features.items()})
    print("edge_count", sum(v.shape[1] for v in graph.edge_index.values()))
