from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gnn_data.raw_schema import make_soft_frontier_label, softmax_scores
from gnn_nav.frontier_clustering import FrontierCluster, cluster_frontiers
from gnn_nav.sparse_graph_builder import DEFAULT_ROOM_NAMES, SparseDecisionGraphBuilder


def _np(value, dtype=np.float32):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _map_tensor(sample: dict, key: str, map_size: int, channels: int = 1):
    maps = sample.get("maps", {})
    value = maps.get(key)
    if value is None:
        if key == "room_map":
            return torch.zeros((1, len(DEFAULT_ROOM_NAMES), map_size, map_size), dtype=torch.float32)
        return torch.zeros((1, 1, map_size, map_size), dtype=torch.float32)
    arr = _np(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]
    return torch.tensor(arr, dtype=torch.float32)


def _clusters_from_points(sample: dict, max_frontiers: int) -> List[FrontierCluster]:
    frontier = sample.get("frontier", {})
    centers = _np(frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", None)))
    if centers is None:
        return []
    centers = centers.reshape(-1, 2)
    distances = _np(frontier.get("distances_valid", frontier.get("mean_path_dist", None)))
    if distances is None:
        distances = np.zeros((len(centers),), dtype=np.float32)
    inv = _np(frontier.get("distance_inverse_valid", frontier.get("distance_inverse", None)))
    if inv is None:
        inv = 1.0 - (np.clip(distances, 1.6, 11.6) - 1.6) / 10.0
    clusters = []
    for idx, center in enumerate(centers[:max_frontiers]):
        d = float(distances[idx]) if idx < len(distances) else 0.0
        di = float(inv[idx]) if idx < len(inv) else 0.0
        member = np.asarray(center, dtype=np.int64).reshape(1, 2)
        clusters.append(
            FrontierCluster(
                cluster_id=idx,
                center_rc=np.asarray(center, dtype=np.float32),
                member_rcs=member,
                size=1,
                min_path_dist=d,
                mean_path_dist=d,
                max_path_dist=d,
                distance_inverse=di,
            )
        )
    return clusters


def extract_frontier_clusters(
    sample: dict,
    max_frontiers: int = 32,
    min_cluster_size: int = 3,
    strict_frontier_clusters: bool = False,
    allow_point_frontiers_debug: bool = True,
    report: Optional[Dict] = None,
):
    frontier = sample.get("frontier", {})
    frontier_map = frontier.get("frontier_map")
    fmm_dist = frontier.get("fmm_dist")
    if frontier_map is not None and fmm_dist is not None:
        try:
            clusters = cluster_frontiers(
                np.asarray(frontier_map).astype(bool),
                _np(fmm_dist, dtype=np.float32),
                min_path_dist=1.6,
                max_frontiers=max_frontiers,
                min_cluster_size=min_cluster_size,
            )
            if len(clusters) > 0:
                return clusters
            if report is not None:
                report["cluster_label_missing_count"] = report.get("cluster_label_missing_count", 0) + 1
        except Exception:
            if report is not None:
                report["cluster_extraction_error_count"] = report.get("cluster_extraction_error_count", 0) + 1
            if strict_frontier_clusters and not allow_point_frontiers_debug:
                raise
    elif report is not None:
        report["cluster_missing_frontier_map_count"] = report.get("cluster_missing_frontier_map_count", 0) + 1

    if strict_frontier_clusters and not allow_point_frontiers_debug:
        raise ValueError("strict frontier clustering failed and point fallback is disabled")
    if report is not None:
        report["cluster_fallback_to_points_count"] = report.get("cluster_fallback_to_points_count", 0) + 1
    return _clusters_from_points(sample, max_frontiers=max_frontiers)


def _pixel_map(values, centers, shape, fill=np.nan):
    out = np.full(shape, fill, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    for value, center in zip(values, centers):
        r, c = np.round(center).astype(np.int64)[:2]
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            out[r, c] = float(value)
    return out


def aggregate_labels_to_clusters(
    sample: dict,
    clusters: List[FrontierCluster],
    tau: float = 2.0,
    teacher_temperature: float = 1.0,
    strict_label_aggregation: bool = False,
    forbid_distance_label_fallback: bool = False,
    report: Optional[Dict] = None,
) -> Dict:
    labels = sample.get("labels", {})
    frontier = sample.get("frontier", {})
    centers = _np(frontier.get("frontier_locations_valid_rc", frontier.get("centers_rc", None)))
    costs = labels.get("frontier_cost")
    teacher_scores = labels.get("teacher_scores")
    if teacher_scores is None:
        teacher_scores = sample.get("teacher", {}).get("total_scores", sample.get("teacher", {}).get("sgnav_scores"))

    if len(clusters) == 0:
        return {
            "frontier_y_soft": torch.zeros((0,), dtype=torch.float32),
            "frontier_cost": torch.zeros((0,), dtype=torch.float32),
            "frontier_best_idx": -1,
            "label_type": labels.get("label_type", "missing"),
        }

    if costs is not None and len(torch.as_tensor(costs).reshape(-1)) == len(clusters):
        cluster_costs = torch.as_tensor(costs, dtype=torch.float32).reshape(-1).numpy()
    elif costs is not None and centers is not None:
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        raw_costs = torch.as_tensor(costs, dtype=torch.float32).reshape(-1).numpy()
        max_r = max([int(np.max(c.member_rcs[:, 0])) for c in clusters] + [int(np.max(centers[:, 0]))]) + 1
        max_c = max([int(np.max(c.member_rcs[:, 1])) for c in clusters] + [int(np.max(centers[:, 1]))]) + 1
        cost_map = _pixel_map(raw_costs, centers, (max_r, max_c))
        cluster_costs = []
        for cluster in clusters:
            vals = cost_map[cluster.member_rcs[:, 0], cluster.member_rcs[:, 1]]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                if strict_label_aggregation:
                    if report is not None:
                        report["cluster_label_missing_count"] = report.get("cluster_label_missing_count", 0) + 1
                    raise ValueError("missing raw labels for frontier cluster")
                idx = int(np.argmin(np.linalg.norm(centers - cluster.center_rc.reshape(1, 2), axis=1)))
                vals = np.asarray([raw_costs[idx]], dtype=np.float32)
            cluster_costs.append(float(np.min(vals)))
        cluster_costs = np.asarray(cluster_costs, dtype=np.float32)
    else:
        if strict_label_aggregation or forbid_distance_label_fallback:
            if report is not None:
                report["distance_only_label_count"] = report.get("distance_only_label_count", 0) + 1
            raise ValueError("missing frontier labels; distance-only label fallback is forbidden")
        if report is not None:
            report["distance_only_label_count"] = report.get("distance_only_label_count", 0) + 1
        cluster_costs = np.asarray([c.mean_path_dist for c in clusters], dtype=np.float32)

    y_soft = make_soft_frontier_label(cluster_costs, tau=tau)
    best_idx = int(np.argmin(np.where(np.isfinite(cluster_costs), cluster_costs, np.inf))) if len(cluster_costs) else -1
    out = {
        "frontier_y_soft": torch.tensor(y_soft, dtype=torch.float32),
        "frontier_cost": torch.tensor(cluster_costs, dtype=torch.float32),
        "frontier_best_idx": int(best_idx),
        "label_type": labels.get("label_type", "converted"),
    }

    if teacher_scores is not None and centers is not None:
        raw_scores = torch.as_tensor(teacher_scores, dtype=torch.float32).reshape(-1).numpy()
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        if len(raw_scores) == len(clusters):
            cluster_teacher = raw_scores
        elif len(raw_scores) == len(centers):
            max_r = max([int(np.max(c.member_rcs[:, 0])) for c in clusters] + [int(np.max(centers[:, 0]))]) + 1
            max_c = max([int(np.max(c.member_rcs[:, 1])) for c in clusters] + [int(np.max(centers[:, 1]))]) + 1
            score_map = _pixel_map(raw_scores, centers, (max_r, max_c))
            cluster_teacher = []
            for cluster in clusters:
                vals = score_map[cluster.member_rcs[:, 0], cluster.member_rcs[:, 1]]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    idx = int(np.argmin(np.linalg.norm(centers - cluster.center_rc.reshape(1, 2), axis=1)))
                    vals = np.asarray([raw_scores[idx]], dtype=np.float32)
                cluster_teacher.append(float(np.max(vals)))
            cluster_teacher = np.asarray(cluster_teacher, dtype=np.float32)
        else:
            cluster_teacher = None
        if cluster_teacher is not None:
            out["teacher_scores"] = torch.tensor(cluster_teacher, dtype=torch.float32)
            out["teacher_y_soft"] = torch.tensor(softmax_scores(cluster_teacher, teacher_temperature), dtype=torch.float32)
            out["teacher_best_idx"] = int(np.argmax(cluster_teacher)) if len(cluster_teacher) else -1
    return out


class RawSampleGraphConverter:
    def __init__(
        self,
        text_encoder,
        room_names: Optional[List[str]] = None,
        max_frontier_clusters: int = 32,
        min_cluster_size: int = 3,
        strict_frontier_clusters: bool = False,
        strict_label_aggregation: bool = False,
        forbid_distance_label_fallback: bool = False,
        allow_point_frontiers_debug: bool = True,
        device: str = "cpu",
    ):
        self.text_encoder = text_encoder
        self.room_names = list(room_names or DEFAULT_ROOM_NAMES)
        self.max_frontier_clusters = int(max_frontier_clusters)
        self.min_cluster_size = int(min_cluster_size)
        self.strict_frontier_clusters = bool(strict_frontier_clusters)
        self.strict_label_aggregation = bool(strict_label_aggregation)
        self.forbid_distance_label_fallback = bool(forbid_distance_label_fallback)
        self.allow_point_frontiers_debug = bool(allow_point_frontiers_debug)
        self.device = device
        self.last_report = {}

    def build_from_raw_sample(self, sample: dict, tau: float = 2.0, teacher_temperature: float = 1.0) -> dict:
        metadata = sample.get("metadata", {})
        map_size = int(metadata.get("map_size", 800))
        map_resolution_cm = float(metadata.get("map_resolution_cm", 5.0))
        room_names = [r.get("room_name", "") for r in sample.get("scenegraph", {}).get("rooms", [])]
        room_names = [r for r in room_names if r] or self.room_names

        builder = SparseDecisionGraphBuilder(
            text_encoder=self.text_encoder,
            map_resolution_cm=map_resolution_cm,
            map_size=map_size,
            room_names=room_names,
            max_objects=100,
            object_knn_k=6,
            frontier_knn_k=8,
            object_radius_m=2.5,
            frontier_radius_m=4.0,
            device=self.device,
        )
        report = {}
        clusters = extract_frontier_clusters(
            sample,
            max_frontiers=self.max_frontier_clusters,
            min_cluster_size=self.min_cluster_size,
            strict_frontier_clusters=self.strict_frontier_clusters,
            allow_point_frontiers_debug=self.allow_point_frontiers_debug,
            report=report,
        )
        scenegraph = SimpleNamespace(
            nodes=sample.get("scenegraph", {}).get("objects", []),
            rooms=room_names,
        )
        full_map = _map_tensor(sample, "full_map", map_size)
        free_map = _map_tensor(sample, "free_map", map_size)
        room_map = _map_tensor(sample, "room_map", map_size)
        agent_pose = sample.get("agent", {}).get("full_pose", metadata.get("agent_pose", [map_size / 2, map_size / 2, 0]))
        goal_text = sample.get("goal", {}).get("object_category_sg", metadata.get("goal_text", "object"))

        graph = builder.build(
            scenegraph=scenegraph,
            frontier_clusters=clusters,
            goal_text=goal_text,
            agent_pose=agent_pose,
            full_map=full_map,
            free_map=free_map,
            room_map=room_map,
            current_step=int(metadata.get("step_id", 0)),
        )
        labels = aggregate_labels_to_clusters(
            sample,
            clusters,
            tau=tau,
            teacher_temperature=teacher_temperature,
            strict_label_aggregation=self.strict_label_aggregation,
            forbid_distance_label_fallback=self.forbid_distance_label_fallback,
            report=report,
        )
        report["raw_frontier_count"] = int(len(_np(sample.get("frontier", {}).get("frontier_locations_valid_rc", []))))
        report["cluster_frontier_count"] = int(len(clusters))
        self.last_report = report
        return {
            "version": "gnn_graph_step_v1",
            "graph": {
                "node_features": graph.node_features,
                "edge_index": graph.edge_index,
                "edge_attr": graph.edge_attr,
                "node_texts": graph.node_texts,
                "frontier_centers_rc": graph.frontier_centers_rc,
            },
            "labels": labels,
            "frontier": {
                "centers_rc": graph.frontier_centers_rc,
                "sizes": torch.tensor([c.size for c in clusters], dtype=torch.long),
                "mean_path_dist": torch.tensor([c.mean_path_dist for c in clusters], dtype=torch.float32),
                "distance_inverse": torch.tensor([c.distance_inverse for c in clusters], dtype=torch.float32),
            },
            "metadata": metadata,
            "goal": sample.get("goal", {}),
            "source_version": sample.get("version", ""),
            "conversion_report": report,
        }
