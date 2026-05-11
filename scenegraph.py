import base64
import hashlib
import json
import math
import os
import re
from collections import Counter
from io import BytesIO
from pathlib import Path, PosixPath
from types import SimpleNamespace
import cv2
import numpy as np
import omegaconf
import requests
import supervision as sv
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.cluster import DBSCAN

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from GroundingDINO.groundingdino.datasets import transforms as T

from utils.utils_scenegraph.mapping import compute_spatial_similarities, merge_detections_to_objects
from utils.utils_scenegraph.slam_classes import MapObjectList
from utils.utils_scenegraph.utils import (
    filter_objects,
    gobs_to_detection_list,
    merge_obj2_into_obj1,
)
from utils.utils_scenegraph.grounded_sam_demo import get_grounding_output, load_image, load_model
from utils.llm_parsing import (
    canonicalize_relation,
    extract_json,
    parse_distance_m,
    parse_probability_01,
    parse_relation_lines,
    parse_room_name,
    parse_yes_no,
    strip_thinking,
)
from utils.sgnav_debug import SGNavDebugStats


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class VLLMChatClient:
    def __init__(self):
        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        self.timeout = float(os.environ.get("VLLM_TIMEOUT", "120"))
        self.default_max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "256"))
        self.default_temperature = float(os.environ.get("VLLM_TEMPERATURE", "0"))
        self.default_top_p = float(os.environ.get("VLLM_TOP_P", "1.0"))
        self.seed = int(os.environ.get("VLLM_SEED", "0"))
        self.disable_thinking = os.environ.get("VLLM_DISABLE_THINKING", "1") not in [
            "0",
            "false",
            "False",
        ]

    def chat(
        self,
        model,
        messages,
        *,
        max_tokens=None,
        temperature=None,
        top_p=None,
        request_type="unknown",
        response_format=None,
        extra_body=None,
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "top_p": self.default_top_p if top_p is None else top_p,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            "seed": self.seed,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            if response_format is None:
                raise
            payload.pop("response_format", None)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        payload_out = response.json()
        return payload_out["choices"][0]["message"].get("content", "")


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []
        self.center = None
        self.map_mask = None
        self.map_score = 0.0
        self.active = False


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.room_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)
        if self.room_node is not None:
            group_text = " and ".join(node.caption for node in self.nodes)
            self.caption += (
                f" Parent room node: {self.room_node.caption}. "
                f"Affiliation edge: ({group_text}, belongs to, {self.room_node.caption})."
            )

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."


class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.is_goal_node = False
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.room_membership_score = 0.0
        self.room_containment_ratio = 0.0
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text


class SceneGraph():
    def __init__(self, map_resolution, map_size_cm, map_size, camera_matrix, is_navigation=True, agent=None) -> None:
        self.map_resolution = map_resolution
        self.map_size_cm = map_size_cm
        self.map_size = map_size
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.camera_matrix = camera_matrix
        self.SAM_ENCODER_VERSION = "vit_h"
        self.sam_variant = 'groundedsam'
        self.force_cpu = os.environ.get("SGNAV_FORCE_CPU", "0") not in [
            "0",
            "false",
            "False",
        ]
        self.device = 'cpu' if self.force_cpu else 'cuda'
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_text = ''
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.reason_visualization = ''
        self.is_navigation = is_navigation
        self.vllm_client = VLLMChatClient()
        self.debug_enabled = False
        self.debug_stats = SGNavDebugStats(enabled=False)
        self.last_debug_print_step = None
        self.llm_name = os.environ.get('VLLM_LLM_MODEL', os.environ.get('VLLM_MODEL', 'qwen3-vl-8b-instruct'))
        self.vlm_name = os.environ.get('VLLM_VLM_MODEL', os.environ.get('VLLM_MODEL', self.llm_name))
        self.seg_xyxy = None
        self.seg_caption = None
        self._subgraph_score_cache_key = None
        self._subgraph_score_cache = []
        self._subgraph_score_cache_by_key = {}
        self._wall_orientation_cache = {}
        self._llm_cache = {}
        self._vlm_cache = {}
        self.last_score_debug = {}
        self._room_graph_version = 0
        
        self.groundingdino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 2, 'bed': 7, 'cabinet': 3, 'chair': 5, 'chest_of_drawers': 5, 'clothes': 9, 'counter': 4, 'cushion': 7, 'fireplace': 4, 'gym_equipment': 7, 'picture': 9, 'plant': 3, 'seating': 2, 'shower': 2, 'sink': 3, 'sofa': 9, 'stool': 5, 'table': 8, 'toilet': 3, 'towel': 4, 'tv_monitor': 2, 'treadmill. fitness equipment.': 0}
        self.small_objects = ['bathtub', 'chest_of_drawers', 'cushion', 'plant', 'seating', 'shower', 'toilet', 'tv_monitor']
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.edge_proposal_batch_size = int(os.environ.get("SGNAV_EDGE_PROPOSAL_BATCH_SIZE", "12"))
        self.node_space = 'bathtub. bed. cabinet. chair. drawers. clothes. counter. cushion. fireplace. gym. picture. plant. seating. shower. sink. sofa. stool. table. toilet. towel. tv. treadmill. fitness equipment.'
        self.related_category_pairs = self.build_related_category_pairs()
        self.prompt_edge_proposal = '''You are an indoor spatial relationship classifier.
For each object pair, output one short spatial relation.
Return a JSON object with a "relationships" array containing exactly one string per input pair.
Do not output markdown or explanation.
Allowed examples: next to, near, above, on top of, opposite to, below, inside, behind, in front of.

Input pairs:
'''
        self.prompt_relation = 'What is the spatial relationship between the {} and the {} in the image? You can only answer a word or phrase that describes a spatial relationship.'
        self.prompt_discriminate_relation = '''Look at the image.
Question: Do the {} and {} satisfy the relationship "{}"?
Return exactly one word: yes or no.'''
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Return a JSON object like {{"room": "bedroom"}} and no explanation.'
        self.prompt_graph_corr_0 = '''Return a JSON object like {{"distance": 2.0, "reason": "short reason"}}.
Question: What is the most likely distance in meters between A and B in an indoor scene?
A: [{}]
B: [{}]
Answer:'''
        self.prompt_graph_corr_1 = 'Ask one short question about the spatial relationship between the object and the goal for predicting their distance. Object: [{}]. Goal: [{}]. Output only the question.'
        self.prompt_graph_corr_2 = 'Given this subgraph with nodes and edges: [{}] Answer the following question with a short sentence based only on the subgraph. Question: {}'
        self.prompt_graph_corr_3 = '''Return a JSON object like {{"distance": 2.0, "reason": "short reason"}}.
Initial object-goal distance estimate in meters: {}
Dialog: [{}]
A: [{}]
B: [{}]
Final most likely distance in meters between this subgraph and the goal:'''
        self.prompt_frontier_explanation = '''Return a JSON object like {{"explanation": "short reason"}}.
Goal: [{}]
Selected frontier evidence from the nearest 3 subgraphs:
{}
Summarize why this frontier is promising or not promising for finding the goal in one concise sentence.'''
        self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
        self.set_cfg()
        self.set_agent(agent)

    def reset(self):
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_text = ''
        self.edge_list = []
        self.reason_visualization = ''
        self.last_debug_print_step = None
        self._subgraph_score_cache_key = None
        self._subgraph_score_cache = []
        self._subgraph_score_cache_by_key = {}
        self._wall_orientation_cache = {}
        self.last_score_debug = {}
        self._room_graph_version = 0

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': self.device, 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def get_arg(self, name, default=None):
        args = getattr(getattr(self, "agent", None), "args", None)
        return getattr(args, name, default)

    def build_related_category_pairs(self):
        pairs = [
            ("bed", "nightstand"),
            ("wardrobe", "dresser"),
            ("bookshelf", "chair"),
            ("counter", "stove"),
            ("table", "chair"),
            ("bathroom sink", "mirror"),
            ("shower", "bathtub"),
            ("refrigerator", "freezer"),
            ("oven", "microwave"),
            ("washing machine", "dryer"),
            ("sofa", "table"),
            ("desk", "office chair"),
            ("computer", "monitor"),
            ("piano", "bench"),
            ("fireplace", "mantel"),
            ("table", "mirror"),
            ("window", "curtains"),
            ("closet", "hangers"),
            ("bathroom cabinet", "toiletries"),
            ("living room rug", "coffee table"),
            ("kitchen cabinet", "dishes"),
            ("dining room chandelier", "dining table"),
            ("clock", "wall"),
            ("floor lamp", "reading chair"),
            ("couch", "throw pillows"),
            ("bookcase", "books"),
            ("tv", "tv cabinet"),
        ]
        return {tuple(sorted(pair)) for pair in pairs}

    def normalize_category(self, caption):
        text = str(caption or "").lower().replace("_", " ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        aliases = {
            "chest of drawers": "dresser",
            "drawers": "dresser",
            "bookcase": "bookshelf",
            "couch": "sofa",
            "television": "tv",
            "tv monitor": "tv",
            "monitor": "tv",
            "office chair": "chair",
            "dining table": "table",
            "coffee table": "table",
            "reading chair": "chair",
            "bathroom sink": "sink",
            "tv cabinet": "cabinet",
        }
        return aliases.get(text, text)

    def category_matches_pattern(self, caption, pattern):
        caption = self.normalize_category(caption)
        pattern = self.normalize_category(pattern)
        if caption == pattern:
            return True
        if pattern in caption:
            return True
        if caption in pattern and len(caption) > 2:
            return True
        return False

    def categories_related_for_group(self, caption_a, caption_b):
        for pattern_a, pattern_b in self.related_category_pairs:
            if (
                self.category_matches_pattern(caption_a, pattern_a)
                and self.category_matches_pattern(caption_b, pattern_b)
            ) or (
                self.category_matches_pattern(caption_a, pattern_b)
                and self.category_matches_pattern(caption_b, pattern_a)
            ):
                return True
        return False

    def score_mode(self):
        return self.get_arg("sgnav_score_mode", "group")

    def score_refresh_bucket(self):
        every_k = max(1, int(self.get_arg("score_refresh_every_k", 5)))
        return int(getattr(self, "navigate_steps", -1)) // every_k

    def edge_update_every_k(self):
        return max(1, int(self.get_arg("edge_update_every_k", 5)))

    def max_edge_proposal_per_step(self):
        return int(self.get_arg("max_edge_proposal_per_step", 128))

    def room_map_active_threshold(self):
        return float(self.get_arg("paper_room_map_active_threshold", 0.05))

    def room_membership_min_score(self):
        return float(self.get_arg("paper_room_min_membership_score", 0.01))

    def room_point_sample_limit(self):
        return max(1, int(self.get_arg("paper_room_point_sample_limit", 512)))

    def duplicate_object_merge_enabled(self):
        return bool(int(self.get_arg("object_duplicate_merge_enabled", 1)))

    def duplicate_merge_center_m(self):
        return float(self.get_arg("object_duplicate_merge_center_m", 0.80))

    def duplicate_merge_strong_center_m(self):
        return float(self.get_arg("object_duplicate_merge_strong_center_m", 0.35))

    def duplicate_merge_point_distance_m(self):
        return float(self.get_arg("object_duplicate_merge_point_distance_m", 0.08))

    def duplicate_merge_point_overlap(self):
        return float(self.get_arg("object_duplicate_merge_point_overlap", 0.12))

    def duplicate_merge_bbox_iou(self):
        return float(self.get_arg("object_duplicate_merge_bbox_iou", 0.05))

    def duplicate_merge_bbox_containment(self):
        return float(self.get_arg("object_duplicate_merge_bbox_containment", 0.55))

    def duplicate_merge_max_passes(self):
        return max(1, int(self.get_arg("object_duplicate_merge_max_passes", 3)))

    def disable_vlm_short_edge_check(self):
        return bool(self.get_arg("disable_vlm_short_edge_check", False))

    def disable_llm_edges(self):
        return bool(self.get_arg("disable_llm_edges", False))

    def set_debug(self, enabled=False, log_dir="data/debug_sgnav"):
        self.debug_enabled = enabled
        self.debug_stats = SGNavDebugStats(log_dir=log_dir, enabled=enabled)

    def set_obj_goal(self, obj_goal, obj_goal_sg):
        self.obj_goal = obj_goal
        self.obj_goal_sg = obj_goal_sg
        if self.obj_goal in self.threshold_list:
            self.cfg.obj_min_detections = self.threshold_list[self.obj_goal]

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def room_map_to_numpy(self):
        if not hasattr(self, "room_map"):
            return None
        room_map = self.room_map
        if torch.is_tensor(room_map):
            room_map = room_map.detach().cpu().numpy()
        else:
            room_map = np.asarray(room_map)
        if room_map.ndim == 4:
            room_map = room_map[0]
        if room_map.ndim != 3 or room_map.shape[0] == 0:
            return None
        return room_map

    def refresh_room_nodes_from_room_map(self):
        self._room_graph_version += 1
        room_map = self.room_map_to_numpy()
        active_threshold = self.room_map_active_threshold()
        for room_node in self.room_nodes:
            room_node.nodes.clear()
            room_node.group_nodes = []
            room_node.center = None
            room_node.map_mask = None
            room_node.map_score = 0.0
            room_node.active = False
        if room_map is None:
            self.debug_stats.inc("paper_room_map_missing")
            return

        for idx, room_node in enumerate(self.room_nodes):
            if idx >= room_map.shape[0]:
                continue
            channel = np.asarray(room_map[idx], dtype=np.float32)
            mask = channel > active_threshold
            room_node.map_mask = mask
            if np.any(mask):
                ys, xs = np.where(mask)
                weights = channel[ys, xs]
                weight_sum = float(np.sum(weights))
                if weight_sum > 1e-6:
                    cx = float(np.sum(xs * weights) / weight_sum)
                    cy = float(np.sum(ys * weights) / weight_sum)
                else:
                    cx = float(np.mean(xs))
                    cy = float(np.mean(ys))
                room_node.center = [cx, cy]
                room_node.map_score = float(np.max(channel))
                room_node.active = True
        active_rooms = sum(1 for room_node in self.room_nodes if room_node.active)
        self.debug_stats.inc("paper_room_map_refresh")
        self.debug_stats.inc("paper_room_active_count", active_rooms)

    def object_points_to_map_xy(self, points):
        if points is None or len(points) == 0:
            return None, None
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 2:
            return None, None
        limit = self.room_point_sample_limit()
        if len(points) > limit:
            stride = max(1, len(points) // limit)
            points = points[::stride][:limit]
        xs = np.rint(points[:, 0] * 100.0 / self.map_resolution).astype(np.int32)
        ys = np.rint(points[:, 1] * 100.0 / self.map_resolution).astype(np.int32)
        ys = self.map_size - 1 - ys
        valid = (
            (xs >= 0)
            & (xs < self.map_size)
            & (ys >= 0)
            & (ys < self.map_size)
        )
        if not np.any(valid):
            return None, None
        return xs[valid], ys[valid]

    def assign_room_node_by_containment(self, node, points):
        room_map = self.room_map_to_numpy()
        xs, ys = self.object_points_to_map_xy(points)
        if room_map is None or xs is None or ys is None:
            return None, 0.0, 0.0

        room_samples = room_map[:, ys, xs]
        if room_samples.size == 0:
            return None, 0.0, 0.0
        mean_scores = np.mean(room_samples, axis=1)
        coverage = np.mean(
            room_samples > self.room_map_active_threshold(),
            axis=1,
        )
        combined = mean_scores + coverage
        best_idx = int(np.argmax(combined))
        best_score = float(mean_scores[best_idx])
        best_coverage = float(coverage[best_idx])
        if combined[best_idx] < self.room_membership_min_score():
            return None, best_score, best_coverage
        if best_idx >= len(self.room_nodes):
            return None, best_score, best_coverage
        return self.room_nodes[best_idx], best_score, best_coverage

    def set_node_room(self, node, room_node, membership_score=0.0, containment_ratio=0.0):
        if node.room_node is not room_node:
            if node.room_node is not None:
                node.room_node.nodes.discard(node)
            node.room_node = room_node
            if room_node is not None:
                room_node.nodes.add(node)
        elif room_node is not None:
            room_node.nodes.add(node)
        node.room_membership_score = float(membership_score)
        node.room_containment_ratio = float(containment_ratio)

    def get_sam_mask_generator(self, variant:str, device) -> SamAutomaticMaskGenerator:
        if variant == "sam":
            sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.sam_checkpoint)
            sam.to(device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=12,
                points_per_batch=144,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                min_mask_region_area=100,
            )
            return mask_generator
        elif variant == "fastsam":
            raise NotImplementedError
            # from ultralytics import YOLO
            # from FastSAM.tools import *
            # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
            # model = YOLO(args.model_path)
            # return model
        elif variant == "groundedsam":
            model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, None, device=device)
            predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
            return model, predictor
        else:
            raise NotImplementedError
    
    def get_sam_segmentation_dense(
        self, variant:str, model, image: np.ndarray
    ) -> tuple:
        '''
        The SAM based on automatic mask generation, without bbox prompting
        
        Args:
            model: The mask generator or the YOLO model
            image: )H, W, 3), in RGB color space, in range [0, 255]
            
        Returns:
            mask: (N, H, W)
            xyxy: (N, 4)
            conf: (N,)
        '''
        if variant == "sam":
            results = model.generate(image)  # type(results) == list
            mask = []
            xyxy = []
            conf = []
            for r in results:  # type(r) == dict
                mask.append(r["segmentation"])  # type(r["segmentation"]) == np.ndarray, r["segmentation"] == [480, 640]
                r_xyxy = r["bbox"].copy()  # type(r["bbox"]) == list, [x, y, h, w]
                # Convert from xyhw format to xyxy format
                r_xyxy[2] += r_xyxy[0]
                r_xyxy[3] += r_xyxy[1]
                xyxy.append(r_xyxy)
                conf.append(r["predicted_iou"])  # type(r["predicted_iou"]) == float
            mask = np.array(mask)
            xyxy = np.array(xyxy)
            conf = np.array(conf)
            return mask, xyxy, conf
        elif variant == "fastsam":
            # The arguments are directly copied from the GSA repo
            results = model(
                image,
                imgsz=1024,
                device="cuda",
                retina_masks=True,
                iou=0.9,
                conf=0.4,
                max_det=100,
            )
            raise NotImplementedError
        elif variant == "groundedsam":
            groundingdino = model[0]
            sam_predictor = model[1]
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
            boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
            if len(caption) == 0:
                return None, None, None, None
            sam_predictor.set_image(image)

            # size = image_pil.size
            H, W = image.shape[0], image.shape[1]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            mask, conf, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
            mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
            return mask, xyxy, conf, caption
        else:
            raise NotImplementedError

    def compute_clip_features(self, image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
        backup_image = image.copy()
        
        image = Image.fromarray(image)
        
        # padding = args.clip_padding  # Adjust the padding amount as needed
        padding = 20  # Adjust the padding amount as needed
        
        image_crops = []
        image_feats = []
        text_feats = []

        
        for idx in range(len(detections.xyxy)):
            # Get the crop of the mask with padding
            x_min, y_min, x_max, y_max = detections.xyxy[idx]

            # Check and adjust padding to avoid going beyond the image borders
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Get the preprocessed image for clip from the crop 
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to(device)

            crop_feat = clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            
            class_id = detections.class_id[idx]
            tokenized_text = clip_tokenizer([classes[class_id]]).to(device)
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            
            crop_feat = crop_feat.cpu().numpy()
            text_feat = text_feat.cpu().numpy()

            image_crops.append(cropped_image)
            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)

        return image_crops, image_feats, text_feats

    def process_cfg(self, cfg: DictConfig):
        cfg.dataset_root = Path(cfg.dataset_root)
        cfg.dataset_config = Path(cfg.dataset_config)
        
        if cfg.dataset_config.name != "multiscan.yaml":
            # For datasets whose depth and RGB have the same resolution
            # Set the desired image heights and width from the dataset config
            dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
            if cfg.image_height is None:
                cfg.image_height = dataset_cfg.camera_params.image_height
            if cfg.image_width is None:
                cfg.image_width = dataset_cfg.camera_params.image_width
            print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        else:
            # For dataset whose depth and RGB have different resolutions
            assert cfg.image_height is not None and cfg.image_width is not None, \
                "For multiscan dataset, image height and width must be specified"

        return cfg

    def crop_image_and_mask(self, image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
        """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
        
        image = np.array(image)
        # Verify initial dimensions
        if image.shape[:2] != mask.shape:
            print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
            return None, None

        # Define the cropping coordinates
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        # round the coordinates to integers
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        # Crop the image and the mask
        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Verify cropped dimensions
        if image_crop.shape[:2] != mask_crop.shape:
            print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
            return None, None
        
        # convert the image back to a pil image
        image_crop = Image.fromarray(image_crop)

        return image_crop, mask_crop
    
    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        if self.sam_variant == 'sam' or self.sam_variant == 'groundedsam':
            with torch.no_grad():
                mask, xyxy, conf, caption = self.get_sam_segmentation_dense(self.sam_variant, self.mask_generator, self.image_rgb)
                self.seg_xyxy = xyxy
                self.seg_caption = caption
            if caption is None:
                return
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            # with torch.no_grad():
            #     image_crops, image_feats, text_feats = self.compute_clip_features(image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.classes, self.device)
            # image_appear_efficiency = [''] * len(image_crops)
            image_appear_efficiency = [''] * len(mask)
            self.segment2d_results.append({
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                # "image_crops": image_crops,
                # "image_feats": image_feats,
                # "text_feats": text_feats,
                "image_appear_efficiency": image_appear_efficiency,
                "image_rgb": self.image_rgb,
                "caption": caption,
            })

    def mapping3d(self):
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]
        gobs = self.segment2d_results[-1]
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = self.pose_matrix,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
            # color_path = color_path,
        )
        
        if len(fg_detection_list) == 0:
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            return
                
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        # visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
        # agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        # agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        
        # self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        self.objects_post = filter_objects(self.cfg, self.objects)
            
    def get_caption(self):
        if self.sam_variant == 'groundedsam':
            for idx, object in enumerate(self.objects_post):
                caption_list = []
                for idx_det in range(len(object["image_idx"])):
                    caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                    caption_list = caption_list + caption.split(' ')
                caption = self.find_modes(caption_list)[0]
                object['captions'] = [caption]

    def update_node(self):
        self.refresh_room_nodes_from_room_map()
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        for new_object in new_objects:
            new_node = ObjectNode()
            caption = new_object['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            self.nodes.append(new_node)
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            map_x = int(center[0] * 100 / self.map_resolution)
            map_y = int(center[1] * 100 / self.map_resolution)
            map_y = self.map_size - 1 - map_y
            node.set_center([map_x, map_y])
            room_node, membership_score, containment_ratio = self.assign_room_node_by_containment(
                node,
                points,
            )
            self.set_node_room(
                node,
                room_node,
                membership_score=membership_score,
                containment_ratio=containment_ratio,
            )
            if node.caption in self.obj_goal_sg:
                node.is_goal_node = True

    def update_edge(self):
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            return
        # create the edge between new_node and old_node
        new_edges = []
        created_count = 0
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                new_edges.append(new_edge)
                created_count += 1
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                new_edges.append(new_edge)
                created_count += 1
        max_edges = self.max_edge_proposal_per_step()
        if max_edges > 0 and len(new_edges) > max_edges:
            for edge in new_edges[max_edges:]:
                edge.delete()
            self.debug_stats.inc("edge_proposal_throttled", len(new_edges) - max_edges)
            new_edges = new_edges[:max_edges]
        self.debug_stats.inc("edges_created", created_count)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        all_new_edges = list(new_edges)
        for new_edge in new_edges:
            image = self.get_joint_image(new_edge.node1, new_edge.node2)
            if image is not None and not self.disable_vlm_short_edge_check():
                prompt = self.prompt_relation.format(new_edge.node1.caption, new_edge.node2.caption)
                response = self.get_vlm_response(
                    prompt=prompt,
                    image=image,
                    request_type="relation_from_image",
                    max_tokens=16,
                )
                new_edge.set_relation(canonicalize_relation(response))
            elif image is not None:
                self.debug_stats.inc("relation_from_image_skipped")
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get relation proposals for long-range edges that do not share an RGB-D frame
        if len(new_edges) > 0:
            relations = self.propose_edge_relations(new_edges)
            if relations is not None:
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)

        # Validate both short-range VLM relations and long-range LLM proposals.
        self.free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
        none_count = sum(edge.relation is None for edge in all_new_edges)
        if none_count > 0:
            self.debug_stats.inc("edge_none_relation", none_count)
        for new_edge in all_new_edges:
            if new_edge.relation is None:
                self.debug_stats.inc("edges_deleted")
                new_edge.delete()
            elif not self.discriminate_relation(new_edge):
                self.debug_stats.inc("edges_deleted")
                new_edge.delete()
            else:
                self.debug_stats.inc("edges_kept")

    def propose_edge_relations(self, new_edges):
        relations = []
        batch_size = max(1, self.edge_proposal_batch_size)
        for start in range(0, len(new_edges), batch_size):
            batch = new_edges[start:start + batch_size]
            pairs = [
                {"object1": edge.node1.caption, "object2": edge.node2.caption}
                for edge in batch
            ]
            prompt = self.prompt_edge_proposal + json.dumps(pairs, ensure_ascii=False)
            self.debug_stats.inc("edge_proposal_total")
            raw = self.get_llm_response(
                prompt=prompt,
                request_type="edge_proposal",
                max_tokens=max(96, 24 * len(batch)),
                response_format={"type": "json_object"},
            )
            batch_relations = parse_relation_lines(raw, expected_n=len(batch))
            if batch_relations is None:
                self.debug_stats.inc("edge_proposal_parse_fail")
                self.debug_stats.inc("edge_relation_mismatch")
                self.debug_stats.log_response(
                    request_type="edge_proposal_parse_fail",
                    prompt=prompt,
                    response=raw,
                    meta={"expected_n": len(batch), "batch_start": start},
                )
                relations.extend([None] * len(batch))
            else:
                relations.extend(batch_relations)
        if len(relations) != len(new_edges):
            self.debug_stats.inc("edge_relation_mismatch")
            return None
        return relations

    def update_group(self):
        total_groups = 0
        total_group_edges = 0
        total_group_room_edges = 0
        for room_node in self.room_nodes:
            room_node.group_nodes = []
        self.group_nodes = []
        object_nodes = [
            node
            for node in self.nodes
            if getattr(node, "caption", None) and getattr(node, "center", None) is not None
        ]
        if len(object_nodes) == 0:
            self.debug_stats.inc("paper_group_update")
            return

        adjacency = {node: set() for node in object_nodes}
        object_node_set = set(object_nodes)
        for node in object_nodes:
            for edge in node.edges:
                if not edge.relation:
                    continue
                other = edge.node2 if edge.node1 is node else edge.node1
                if other not in object_node_set:
                    continue
                if not self.categories_related_for_group(node.caption, other.caption):
                    continue
                adjacency[node].add(other)
                adjacency[other].add(node)
                total_group_edges += 1

        visited = set()
        for node in object_nodes:
            if node in visited or len(adjacency[node]) == 0:
                continue
            stack = [node]
            component = []
            visited.add(node)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if len(component) < 2:
                continue

            group_node = GroupNode()
            group_node.nodes.extend(component)
            group_node.get_graph()
            component_room_list = [getattr(member, "room_node", None) for member in component]
            component_rooms = {room for room in component_room_list if room is not None}
            if len(component_rooms) == 1 and all(room is not None for room in component_room_list):
                group_node.room_node = next(iter(component_rooms))
                group_node.room_node.group_nodes.append(group_node)
                total_group_room_edges += 1
                group_node.get_graph()
            self.group_nodes.append(group_node)
            total_groups += 1
        self.debug_stats.inc("paper_group_update")
        self.debug_stats.inc("paper_group_count", total_groups)
        self.debug_stats.inc("paper_group_related_edge_count", total_group_edges // 2)
        self.debug_stats.inc("paper_group_room_edge_count", total_group_room_edges)

    def insert_goal(self, goal=None):
        self.debug_stats.inc("insert_goal_total")
        if goal is None:
            goal = self.obj_goal_sg
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            self.debug_stats.inc("insert_goal_none")
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        self.debug_stats.inc("room_predict_total")
        response = self.get_llm_response(
            prompt=prompt,
            request_type="room_predict",
            max_tokens=16,
            response_format={"type": "json_object"},
        )
        room_name = parse_room_name(response, [room_node.caption for room_node in self.room_nodes])
        predict_room_node = None
        if room_name is not None:
            for room_node in self.room_nodes:
                if room_node.caption == room_name and len(room_node.group_nodes) > 0:
                    predict_room_node = room_node
                    break
        if predict_room_node is None:
            self.debug_stats.inc("room_predict_parse_fail")
            self.debug_stats.log_response(
                request_type="room_predict_parse_fail",
                prompt=prompt,
                response=response,
                meta={"available_rooms": room_node_text},
            )
            predict_room_node = self.fallback_room_by_cooccurrence()
            if predict_room_node is None:
                self.debug_stats.inc("insert_goal_none")
                return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        self.debug_stats.inc("insert_goal_success")
        return self.mid_term_goal

    def get_scored_subgraphs_for_goal(self, goal, force_refresh=False):
        mode = self.score_mode()
        if mode == "paper_object":
            return self.get_object_centered_subgraphs_for_goal(goal, force_refresh=force_refresh)
        if mode == "hybrid":
            return (
                self._get_group_scored_subgraphs_for_goal(goal, force_refresh=force_refresh)
                + self.get_object_centered_subgraphs_for_goal(goal, force_refresh=force_refresh)
            )
        return self._get_group_scored_subgraphs_for_goal(goal, force_refresh=force_refresh)

    def _get_group_scored_subgraphs_for_goal(self, goal, force_refresh=False):
        """Return the legacy GroupNode-based subgraph scores."""
        cache_key = (
            "group",
            goal,
            self.score_refresh_bucket(),
            len(self.nodes),
            len(self.get_edges()),
            self._room_graph_version,
        )
        if not force_refresh and cache_key in self._subgraph_score_cache_by_key:
            return self._subgraph_score_cache_by_key[cache_key]

        self.update_group()
        items = []
        for group_node in self.group_nodes:
            if len(group_node.nodes) == 0:
                continue
            group_node.get_graph()
            if group_node.center is None or group_node.center_node is None:
                continue
            p_sub = float(self.graph_corr(goal, group_node))
            p_sub = float(np.clip(p_sub, 0.0, 1.0))
            room_node = getattr(group_node, "room_node", None)
            items.append({
                "score": p_sub,
                "center_xy": np.array(group_node.center, dtype=np.float32),
                "center_node": group_node.center_node,
                "group_node": group_node,
                "room": getattr(room_node, "caption", ""),
                "caption": group_node.caption,
                "pred_distance_m": getattr(group_node, "pred_distance_m", None),
                "reason": getattr(group_node, "reason", ""),
                "question": getattr(group_node, "question", ""),
                "answer": getattr(group_node, "answer", ""),
                "initial_distance_m": getattr(group_node, "initial_distance_m", None),
                "initial_reason": getattr(group_node, "initial_reason", ""),
            })

        self._subgraph_score_cache_key = cache_key
        self._subgraph_score_cache = items
        self._subgraph_score_cache_by_key[cache_key] = items
        self.debug_stats.inc("subgraph_score_refresh")
        return items

    def get_object_centered_subgraphs_for_goal(self, goal, force_refresh=False):
        """Return paper-aligned object-centered subgraphs.

        Each subgraph is centered on one object node and includes its parent room,
        containing group if available, directly connected object nodes, directly
        connected edges, and lightweight observation metadata.
        """
        cache_key = (
            "paper_object",
            goal,
            self.score_refresh_bucket(),
            len(self.nodes),
            len(self.get_edges()),
            self._room_graph_version,
        )
        if not force_refresh and cache_key in self._subgraph_score_cache_by_key:
            return self._subgraph_score_cache_by_key[cache_key]

        self.update_group()
        group_by_node = {}
        for group_node in self.group_nodes:
            for grouped_node in group_node.nodes:
                group_by_node[grouped_node] = group_node

        items = []
        for node in self.nodes:
            if getattr(node, "center", None) is None or getattr(node, "caption", None) is None:
                continue
            group_node = group_by_node.get(node)
            edges = [edge for edge in node.edges if edge.relation]
            neighbor_nodes = []
            for edge in edges:
                neighbor = edge.node2 if edge.node1 is node else edge.node1
                if getattr(neighbor, "caption", None):
                    neighbor_nodes.append(neighbor)
            caption = self.object_subgraph_to_text(node, group_node, neighbor_nodes, edges)
            graph = SimpleNamespace(center_node=node, caption=caption)
            p_sub = float(self.graph_corr(goal, graph))
            p_sub = float(np.clip(p_sub, 0.0, 1.0))
            room_node = getattr(node, "room_node", None)
            parent_edges_text = self.affiliation_edges_to_text(node, group_node, room_node)
            graph_nodes_text = self.subgraph_nodes_to_text(node, group_node, room_node, neighbor_nodes)
            graph_edges_text = ", ".join(
                [text for text in [parent_edges_text, ", ".join(edge.text() for edge in edges)] if text]
            )
            items.append({
                "score": p_sub,
                "center_xy": np.array(node.center, dtype=np.float32),
                "center_node": node,
                "group_node": group_node,
                "room": getattr(room_node, "caption", ""),
                "nodes_text": graph_nodes_text,
                "edges_text": graph_edges_text,
                "caption": caption,
                "object_confidence": self.node_confidence(node),
                "observation_count": self.node_observation_count(node),
                "room_membership_score": float(getattr(node, "room_membership_score", 0.0)),
                "room_containment_ratio": float(getattr(node, "room_containment_ratio", 0.0)),
                "pred_distance_m": getattr(graph, "pred_distance_m", None),
                "reason": getattr(graph, "reason", ""),
                "question": getattr(graph, "question", ""),
                "answer": getattr(graph, "answer", ""),
                "initial_distance_m": getattr(graph, "initial_distance_m", None),
                "initial_reason": getattr(graph, "initial_reason", ""),
            })

        self._subgraph_score_cache_key = cache_key
        self._subgraph_score_cache = items
        self._subgraph_score_cache_by_key[cache_key] = items
        self.debug_stats.inc("paper_subgraph_score_refresh")
        self.debug_stats.inc("paper_subgraph_count", len(items))
        return items

    def subgraph_nodes_to_text(self, node, group_node, room_node, neighbor_nodes):
        graph_nodes = [f"object:{node.caption}"]
        if room_node is not None:
            graph_nodes.append(f"room:{room_node.caption}")
        if group_node is not None:
            group_nodes = ", ".join(member.caption for member in group_node.nodes)
            graph_nodes.append(f"group:({group_nodes})")
        graph_nodes.extend(f"object:{neighbor.caption}" for neighbor in neighbor_nodes)
        return ", ".join(graph_nodes)

    def affiliation_edges_to_text(self, node, group_node, room_node):
        edges = []
        if room_node is not None:
            edges.append(f"({node.caption}, belongs to, {room_node.caption})")
        if group_node is not None and room_node is not None and group_node.room_node is room_node:
            group_text = " and ".join(member.caption for member in group_node.nodes)
            edges.append(f"({group_text}, belongs to, {room_node.caption})")
        return ", ".join(edges)

    def object_subgraph_to_text(self, node, group_node, neighbor_nodes, edges):
        room_node = getattr(node, "room_node", None)
        room_name = getattr(room_node, "caption", "unknown room")
        group_text = getattr(group_node, "caption", "") if group_node is not None else ""
        neighbors_text = ", ".join(n.caption for n in neighbor_nodes) or "none"
        object_edges_text = ", ".join(edge.text() for edge in edges)
        parent_edges_text = self.affiliation_edges_to_text(node, group_node, room_node)
        edges_text = ", ".join(
            [text for text in [parent_edges_text, object_edges_text] if text]
        ) or "none"
        confidence = self.node_confidence(node)
        observation_count = self.node_observation_count(node)
        return (
            f"Center object: {node.caption}. "
            f"Parent room node: {room_name}. "
            f"Group context: {group_text or 'none'}. "
            f"Connected objects: {neighbors_text}. "
            f"Edges: {edges_text}. "
            f"Room membership score: {getattr(node, 'room_membership_score', 0.0):.3f}. "
            f"Room containment ratio: {getattr(node, 'room_containment_ratio', 0.0):.3f}. "
            f"Object confidence: {confidence:.3f}. "
            f"Observation count: {observation_count}."
        )

    def node_confidence(self, node):
        obj = getattr(node, "object", None) or {}
        conf = obj.get("conf", obj.get("confidence", []))
        try:
            if torch.is_tensor(conf):
                conf = conf.detach().cpu().numpy()
            arr = np.asarray(conf, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return 0.0
            return float(np.nanmax(arr))
        except Exception:
            return 0.0

    def node_observation_count(self, node):
        obj = getattr(node, "object", None) or {}
        image_idx = obj.get("image_idx", [])
        try:
            return int(len(image_idx))
        except Exception:
            return 0

    def node_merge_priority(self, node):
        point_count = len(self.pcd_points(node))
        return (
            self.node_observation_count(node),
            self.node_confidence(node),
            point_count,
        )

    def pcd_points(self, node):
        obj = getattr(node, "object", None) or {}
        pcd = obj.get("pcd")
        if pcd is None:
            return np.zeros((0, 3), dtype=np.float32)
        try:
            points = np.asarray(pcd.points, dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.zeros((0, 3), dtype=np.float32)
        return points

    def node_xy_bounds_m(self, node):
        points = self.pcd_points(node)
        if len(points) == 0:
            return None
        xy = points[:, :2]
        return np.min(xy, axis=0), np.max(xy, axis=0)

    def bounds_iou_and_containment(self, bounds_a, bounds_b):
        if bounds_a is None or bounds_b is None:
            return 0.0, 0.0
        min_a, max_a = bounds_a
        min_b, max_b = bounds_b
        inter_min = np.maximum(min_a, min_b)
        inter_max = np.minimum(max_a, max_b)
        inter_size = np.maximum(inter_max - inter_min, 0.0)
        inter_area = float(inter_size[0] * inter_size[1])
        area_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
        area_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
        if area_a <= 1e-8 or area_b <= 1e-8:
            return 0.0, 0.0
        iou = inter_area / max(area_a + area_b - inter_area, 1e-8)
        containment = inter_area / max(min(area_a, area_b), 1e-8)
        return float(iou), float(containment)

    def deterministic_sample_points(self, points, max_points=128):
        if len(points) <= max_points:
            return points
        stride = max(1, len(points) // max_points)
        return points[::stride][:max_points]

    def point_overlap_ratio(self, node_a, node_b):
        points_a = self.deterministic_sample_points(self.pcd_points(node_a), 128)
        points_b = self.deterministic_sample_points(self.pcd_points(node_b), 128)
        if len(points_a) == 0 or len(points_b) == 0:
            return 0.0
        xy_a = points_a[:, :2]
        xy_b = points_b[:, :2]
        diff = xy_a[:, None, :] - xy_b[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        threshold2 = self.duplicate_merge_point_distance_m() ** 2
        overlap_a = float(np.mean(np.min(dist2, axis=1) <= threshold2))
        overlap_b = float(np.mean(np.min(dist2, axis=0) <= threshold2))
        return max(overlap_a, overlap_b)

    def room_compatible_for_duplicate_merge(self, node_a, node_b):
        room_a = getattr(node_a, "room_node", None)
        room_b = getattr(node_b, "room_node", None)
        return room_a is None or room_b is None or room_a is room_b

    def node_map_distance_m(self, node_a, node_b):
        if getattr(node_a, "center", None) is None or getattr(node_b, "center", None) is None:
            return float("inf")
        return float(
            np.linalg.norm(
                np.asarray(node_a.center, dtype=np.float32)
                - np.asarray(node_b.center, dtype=np.float32)
            )
            * self.map_resolution
            / 100.0
        )

    def should_merge_duplicate_nodes(self, node_a, node_b):
        label_a = self.normalize_category(getattr(node_a, "caption", ""))
        label_b = self.normalize_category(getattr(node_b, "caption", ""))
        if not label_a or not label_b or label_a != label_b:
            return False, {}
        if not self.room_compatible_for_duplicate_merge(node_a, node_b):
            return False, {"reason": "different_rooms"}

        center_m = self.node_map_distance_m(node_a, node_b)
        bounds_iou, bounds_containment = self.bounds_iou_and_containment(
            self.node_xy_bounds_m(node_a),
            self.node_xy_bounds_m(node_b),
        )
        point_overlap = self.point_overlap_ratio(node_a, node_b)
        strong_center = center_m <= self.duplicate_merge_strong_center_m()
        close_enough = center_m <= self.duplicate_merge_center_m()
        geometry_overlap = (
            point_overlap >= self.duplicate_merge_point_overlap()
            or bounds_iou >= self.duplicate_merge_bbox_iou()
            or bounds_containment >= self.duplicate_merge_bbox_containment()
        )
        accepted = bool(strong_center or (close_enough and geometry_overlap))
        meta = {
            "label": label_a,
            "center_m": center_m,
            "point_overlap": point_overlap,
            "bbox_iou": bounds_iou,
            "bbox_containment": bounds_containment,
            "strong_center": bool(strong_center),
            "close_enough": bool(close_enough),
            "geometry_overlap": bool(geometry_overlap),
        }
        return accepted, meta

    def find_edge_between(self, node_a, node_b):
        for edge in node_a.edges:
            if (edge.node1 is node_a and edge.node2 is node_b) or (
                edge.node1 is node_b and edge.node2 is node_a
            ):
                return edge
        return None

    def remove_object_from_lists(self, obj):
        if obj is None:
            return
        self.objects = MapObjectList([item for item in self.objects if item is not obj])
        self.objects_post = MapObjectList(
            [item for item in self.objects_post if item is not obj]
        )

    def merge_duplicate_node_pair(self, keep_node, duplicate_node, meta):
        keep_obj = getattr(keep_node, "object", None)
        duplicate_obj = getattr(duplicate_node, "object", None)
        if keep_obj is None or duplicate_obj is None:
            return
        objects_merged = keep_obj is duplicate_obj
        if keep_obj is not None and duplicate_obj is not None and keep_obj is not duplicate_obj:
            try:
                merge_obj2_into_obj1(self.cfg, keep_obj, duplicate_obj, run_dbscan=False)
                keep_obj["node"] = keep_node
                keep_node.object = keep_obj
                objects_merged = True
            except Exception as exc:
                self.debug_stats.log_response(
                    "object_duplicate_merge_error",
                    prompt=str(meta),
                    response=repr(exc),
                )
                return

        if keep_node.room_node is None and duplicate_node.room_node is not None:
            self.set_node_room(
                keep_node,
                duplicate_node.room_node,
                membership_score=getattr(duplicate_node, "room_membership_score", 0.0),
                containment_ratio=getattr(duplicate_node, "room_containment_ratio", 0.0),
            )
        keep_node.is_goal_node = bool(keep_node.is_goal_node or duplicate_node.is_goal_node)

        for edge in list(duplicate_node.edges):
            other = edge.node2 if edge.node1 is duplicate_node else edge.node1
            relation = edge.relation
            edge.delete()
            if other is keep_node:
                continue
            existing_edge = self.find_edge_between(keep_node, other)
            if existing_edge is not None:
                if existing_edge.relation is None and relation is not None:
                    existing_edge.set_relation(relation)
                continue
            new_edge = Edge(keep_node, other)
            new_edge.set_relation(relation)

        if duplicate_node.room_node is not None:
            duplicate_node.room_node.nodes.discard(duplicate_node)
        if duplicate_node in self.nodes:
            self.nodes.remove(duplicate_node)
        if objects_merged and duplicate_obj is not keep_obj:
            self.remove_object_from_lists(duplicate_obj)
        self.debug_stats.inc("object_duplicate_merged")
        self.debug_stats.log_response(
            "object_duplicate_merged",
            prompt=f"{duplicate_node.caption} -> {keep_node.caption}",
            response=json.dumps(meta, ensure_ascii=False),
        )

    def merge_duplicate_object_nodes(self):
        if not self.duplicate_object_merge_enabled() or len(self.nodes) < 2:
            return
        total_merged = 0
        for _ in range(self.duplicate_merge_max_passes()):
            best_pair = None
            best_meta = None
            best_score = -1.0
            for idx, node_a in enumerate(list(self.nodes)):
                for node_b in list(self.nodes)[idx + 1:]:
                    accepted, meta = self.should_merge_duplicate_nodes(node_a, node_b)
                    self.debug_stats.inc("object_duplicate_merge_checks")
                    if not accepted:
                        continue
                    score = (
                        float(meta.get("point_overlap", 0.0))
                        + float(meta.get("bbox_iou", 0.0))
                        + float(meta.get("bbox_containment", 0.0))
                    )
                    if meta.get("strong_center", False):
                        score += 1.0
                    if score > best_score:
                        best_score = score
                        best_pair = (node_a, node_b)
                        best_meta = meta
            if best_pair is None:
                break
            node_a, node_b = best_pair
            if self.node_merge_priority(node_b) > self.node_merge_priority(node_a):
                keep_node, duplicate_node = node_b, node_a
            else:
                keep_node, duplicate_node = node_a, node_b
            self.merge_duplicate_node_pair(keep_node, duplicate_node, best_meta or {})
            total_merged += 1
        if total_merged > 0:
            self._subgraph_score_cache_by_key.clear()
            self._subgraph_score_cache = []
            self._subgraph_score_cache_key = None
            self.debug_stats.inc("object_duplicate_merge_passes")
            self.debug_stats.inc("object_duplicate_merge_total", total_merged)

    def score_frontiers_by_subgraphs(self, frontier_locations, goal):
        subgraphs = self.get_object_centered_subgraphs_for_goal(goal)
        agent_args = getattr(getattr(self, "agent", None), "args", None)
        score_norm = getattr(agent_args, "frontier_score_norm", "paper_sum")
        if len(subgraphs) == 0:
            self.last_score_debug = {
                "mode": self.score_mode(),
                "frontier_score_norm": score_norm,
                "num_subgraphs": 0,
                "room_nodes_active": int(sum(1 for room_node in self.room_nodes if room_node.active)),
                "room_nodes_with_objects": int(sum(1 for room_node in self.room_nodes if len(room_node.nodes) > 0)),
                "room_nodes_with_groups": int(sum(1 for room_node in self.room_nodes if len(room_node.group_nodes) > 0)),
                "group_nodes_total": int(len(self.group_nodes)),
                "frontier_score_min": 0.0,
                "frontier_score_max": 0.0,
                "frontier_score_mean": 0.0,
            }
            return np.zeros(len(frontier_locations), dtype=np.float32)

        scores = []
        for frontier in frontier_locations:
            score_sum = 0.0
            weight_sum = 0.0
            for subgraph in subgraphs:
                d_pix = np.linalg.norm(
                    np.asarray(frontier, dtype=np.float32)
                    - np.asarray(subgraph["center_xy"], dtype=np.float32)
                )
                d_m = max(float(d_pix) * self.map_resolution / 100.0, 0.25)
                weight = 1.0 / d_m
                score_sum += float(subgraph["score"]) * weight
                weight_sum += weight
            if score_norm == "weighted_mean":
                scores.append(score_sum / max(weight_sum, 1e-6))
            else:
                scores.append(score_sum)
        scores = np.asarray(scores, dtype=np.float32)
        self.last_score_debug = {
            "mode": self.score_mode(),
            "frontier_score_norm": score_norm,
            "num_subgraphs": int(len(subgraphs)),
            "room_nodes_active": int(sum(1 for room_node in self.room_nodes if room_node.active)),
            "room_nodes_with_objects": int(sum(1 for room_node in self.room_nodes if len(room_node.nodes) > 0)),
            "room_nodes_with_groups": int(sum(1 for room_node in self.room_nodes if len(room_node.group_nodes) > 0)),
            "group_nodes_total": int(len(self.group_nodes)),
            "frontier_score_min": float(np.min(scores)) if len(scores) else 0.0,
            "frontier_score_max": float(np.max(scores)) if len(scores) else 0.0,
            "frontier_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        }
        self.debug_stats.log_response(
            "subgraph_score_distribution",
            prompt=f"goal={goal}",
            response=json.dumps(self.last_score_debug),
        )
        return scores

    def explain_frontier_selection(self, frontier_location, goal, top_k=3):
        subgraphs = self.get_object_centered_subgraphs_for_goal(goal)
        if len(subgraphs) == 0 or frontier_location is None:
            explanation = {
                "selected_frontier_rc": None,
                "nearest_subgraphs": [],
                "explanation": "No scored subgraphs are available for explaining this frontier.",
            }
            self.reason_visualization = explanation["explanation"]
            return explanation

        frontier = np.asarray(frontier_location, dtype=np.float32)
        nearest = []
        for subgraph in subgraphs:
            center_xy = np.asarray(subgraph["center_xy"], dtype=np.float32)
            dist_m = float(np.linalg.norm(frontier - center_xy) * self.map_resolution / 100.0)
            center_node = subgraph.get("center_node")
            nearest.append({
                "center_object": getattr(center_node, "caption", ""),
                "room": subgraph.get("room", ""),
                "room_membership_score": subgraph.get("room_membership_score", 0.0),
                "room_containment_ratio": subgraph.get("room_containment_ratio", 0.0),
                "frontier_distance_m": dist_m,
                "score": float(subgraph.get("score", 0.0)),
                "predicted_goal_distance_m": subgraph.get("pred_distance_m"),
                "question": subgraph.get("question", ""),
                "answer": subgraph.get("answer", ""),
                "reason": subgraph.get("reason", ""),
                "nodes": subgraph.get("nodes_text", ""),
                "edges": subgraph.get("edges_text", ""),
            })
        nearest = sorted(nearest, key=lambda item: item["frontier_distance_m"])[:top_k]
        prompt_payload = json.dumps(nearest, ensure_ascii=False)
        prompt = self.prompt_frontier_explanation.format(goal, prompt_payload)
        self.debug_stats.inc("frontier_explanation_total")
        response = self.get_llm_response(
            prompt=prompt,
            request_type="frontier_explanation",
            max_tokens=96,
            response_format={"type": "json_object"},
        )
        data = extract_json(response)
        if isinstance(data, dict) and data.get("explanation"):
            explanation_text = strip_thinking(str(data["explanation"]))
        else:
            explanation_text = strip_thinking(response)
        if not explanation_text:
            fragments = []
            for item in nearest:
                fragments.append(
                    f"{item['center_object']} in {item['room']} has score "
                    f"{item['score']:.3f} near the selected frontier"
                )
            explanation_text = "; ".join(fragments)
        explanation = {
            "selected_frontier_rc": frontier.astype(int).tolist(),
            "nearest_subgraphs": nearest,
            "explanation": explanation_text,
        }
        self.reason_visualization = explanation_text
        self.debug_stats.log_response(
            "frontier_explanation_selected",
            prompt=prompt,
            response=json.dumps(explanation, ensure_ascii=False),
        )
        return explanation

    def fallback_room_by_cooccurrence(self):
        if not hasattr(self.agent, "prob_array_room"):
            return None
        best_node = None
        best_score = -1e9
        for idx, room_node in enumerate(self.room_nodes):
            if len(room_node.group_nodes) == 0:
                continue
            score = 0.0
            if idx < len(self.agent.prob_array_room):
                score = float(self.agent.prob_array_room[idx])
            if score > best_score:
                best_score = score
                best_node = room_node
        return best_node
    
    def update_scenegraph(self):
        print(f'Navigate Step: {self.navigate_steps}', end='\r')
        self.segment2d()
        if len(self.segment2d_results) > 0:
            self.mapping3d()
            self.get_caption()
            self.update_node()
            self.merge_duplicate_object_nodes()
            if self.disable_llm_edges():
                for node in self.nodes:
                    node.is_new_node = False
                self.debug_stats.inc("edge_update_disabled")
            elif self.navigate_steps % self.edge_update_every_k() == 0:
                self.update_edge()
            else:
                self.debug_stats.inc("edge_update_skipped")
        if (
            self.debug_enabled
            and self.navigate_steps % 20 == 0
            and self.last_debug_print_step != self.navigate_steps
        ):
            print("[SGNAV_DEBUG]", self.debug_stats.summary())
            self.last_debug_print_step = self.navigate_steps
    
    def get_llm_response(
        self,
        prompt,
        request_type="llm",
        max_tokens=None,
        response_format=None,
        extra_body=None,
    ):
        cache_payload = json.dumps(
            {
                "request_type": request_type,
                "model": self.llm_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "extra_body": extra_body,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        cache_key = hashlib.sha1(cache_payload.encode("utf-8")).hexdigest()
        if cache_key in self._llm_cache:
            self.debug_stats.inc("llm_cache_hit")
            return self._llm_cache[cache_key]

        system = {
            "role": "system",
            "content": (
                "You are a strict parser-friendly assistant for a robotics "
                "navigation system. Follow the requested output format exactly. "
                "Do not output markdown, explanation, or hidden reasoning."
            ),
        }
        response = self.vllm_client.chat(
            model=self.llm_name,
            messages=[
                system,
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            request_type=request_type,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            response_format=response_format,
            extra_body=extra_body,
        )
        self.debug_stats.inc("llm_calls_total")
        self._llm_cache[cache_key] = response
        self.debug_stats.log_response(request_type, prompt, response)
        return response
    
    def get_vlm_response(self, prompt, image, request_type="vlm", max_tokens=None):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        raw_image_bytes = buffered.getvalue()
        image_hash = hashlib.sha1(raw_image_bytes).hexdigest()
        cache_payload = json.dumps(
            {
                "request_type": request_type,
                "model": self.vlm_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "image_hash": image_hash,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        cache_key = hashlib.sha1(cache_payload.encode("utf-8")).hexdigest()
        if cache_key in self._vlm_cache:
            self.debug_stats.inc("vlm_cache_hit")
            return self._vlm_cache[cache_key]

        image_bytes = base64.b64encode(raw_image_bytes)
        image_str = str(image_bytes, 'utf-8')
        system = {
            "role": "system",
            "content": "You are a strict visual classifier. Return only the requested short answer.",
        }
        response = self.vllm_client.chat(
            model=self.vlm_name,
            messages=[
                system,
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{image_str}'
                            },
                        },
                        {'type': 'text', 'text': prompt},
                    ],
                },
            ],
            request_type=request_type,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        self.debug_stats.inc("vlm_calls_total")
        self._vlm_cache[cache_key] = response
        self.debug_stats.log_response(
            request_type,
            prompt,
            response,
            meta={"has_image": True},
        )
        return response
        
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def score(self, frontier_locations_16, num_16_frontiers):
        mode = self.score_mode()
        if mode == "paper_object":
            return np.nan_to_num(
                self.score_frontiers_by_subgraphs(frontier_locations_16, self.obj_goal_sg),
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )

        scores = np.zeros((num_16_frontiers))
        for i, loc in enumerate(frontier_locations_16):
            sub_room_map = self.agent.room_map[0,:,max(0,loc[0]-12):min(self.agent.map_size-1,loc[0]+13), max(0,loc[1]-12):min(self.agent.map_size-1,loc[1]+13)].cpu().numpy() # sub_room_map.shape = [9, 25, 25], select the room map around the frontier
            whether_near_room = np.max(np.max(sub_room_map, 1),1)
            score_1 = np.clip(1-(1-self.agent.prob_array_room)-(1-whether_near_room), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_room+(1-whether_near_room), -10,1)
            scores[i] = np.sum(score_1) - np.sum(score_2)
        for i in range(21):
            num_obj = len(self.agent.obj_locations[i])
            if num_obj <= 0:
                continue
            frontier_location_mtx = np.tile(frontier_locations_16, (num_obj,1,1))
            obj_location_mtx = np.array(self.agent.obj_locations[i])[:,1:]
            obj_confidence_mtx = np.tile(np.array(self.agent.obj_locations[i])[:,0],(num_16_frontiers,1)).transpose(1,0)
            obj_location_mtx = np.tile(obj_location_mtx, (num_16_frontiers,1,1)).transpose(1,0,2)
            dist_frontier_obj = np.square(frontier_location_mtx - obj_location_mtx)
            dist_frontier_obj = np.sqrt(np.sum(dist_frontier_obj, axis=2)) / 20
            near_frontier_obj = dist_frontier_obj < 1.6
            obj_confidence_mtx[near_frontier_obj==False] = 0
            obj_confidence_max = np.max(obj_confidence_mtx, axis=0)
            score_1 = np.clip(1-(1-self.agent.prob_array_obj[i])-(1-obj_confidence_max), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_obj[i]+(1-obj_confidence_max), -10,1)
            scores += score_1 - score_2

        predict_goal_xy = self.insert_goal()
        if predict_goal_xy is not None:
            predict_goal_xy = np.array(predict_goal_xy).reshape(1, 2)
            distance = np.linalg.norm(predict_goal_xy - frontier_locations_16, axis=1)
            score = np.ones((num_16_frontiers,), dtype=np.float32)
            score[distance > 32] = 0
            score = score / np.maximum(distance, 1.0)
            scores += score
        if mode == "hybrid":
            scores += self.score_frontiers_by_subgraphs(frontier_locations_16, self.obj_goal_sg)
        else:
            self.last_score_debug = {
                "mode": mode,
                "num_subgraphs": None,
                "frontier_score_min": float(np.min(scores)) if len(scores) else 0.0,
                "frontier_score_max": float(np.max(scores)) if len(scores) else 0.0,
                "frontier_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            }
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    def discriminate_relation(self, edge):
        self.debug_stats.inc("discriminate_total")
        image = self.get_joint_image(edge.node1, edge.node2)
        if image is not None:
            if self.disable_vlm_short_edge_check():
                self.debug_stats.inc("edge_short_vlm_check_skipped")
                self.debug_stats.inc("discriminate_yes")
                return True
            return self.validate_short_edge(edge, image)
        return self.validate_long_edge(edge)

    def validate_short_edge(self, edge, image):
        response = self.get_vlm_response(
            self.prompt_discriminate_relation.format(
                edge.node1.caption,
                edge.node2.caption,
                edge.relation,
            ),
            image,
            request_type="discriminate_short_relation",
            max_tokens=8,
        )
        accepted = parse_yes_no(response, default=False)
        self.debug_stats.inc("edge_short_keep" if accepted else "edge_short_drop")
        self.debug_stats.inc("discriminate_yes" if accepted else "discriminate_no")
        return accepted

    def validate_long_edge(self, edge):
        if not self.nodes_in_same_room(edge.node1, edge.node2):
            self.debug_stats.inc("edge_long_drop_room")
            self.debug_stats.inc("discriminate_no")
            return False

        p1 = np.array(edge.node1.center, dtype=np.float32)
        p2 = np.array(edge.node2.center, dtype=np.float32)

        max_long_edge_dist_m = float(os.environ.get("SGNAV_MAX_LONG_EDGE_DIST_M", "0"))
        if max_long_edge_dist_m > 0:
            distance_m = float(np.linalg.norm(p2 - p1) * self.map_resolution / 100.0)
            if distance_m > max_long_edge_dist_m:
                self.debug_stats.inc("edge_long_drop_distance")
                self.debug_stats.inc("discriminate_no")
                return False

        if not self.line_unobstructed(p1, p2):
            self.debug_stats.inc("edge_long_drop_obstructed")
            self.debug_stats.inc("discriminate_no")
            return False

        if not self.line_parallel_to_room_wall(p1, p2, edge.node1.room_node):
            self.debug_stats.inc("edge_long_drop_not_parallel")
            self.debug_stats.inc("discriminate_no")
            return False

        self.debug_stats.inc("edge_long_keep")
        self.debug_stats.inc("discriminate_yes")
        return True

    def nodes_in_same_room(self, node1, node2):
        return (
            getattr(node1, "room_node", None) is not None
            and getattr(node2, "room_node", None) is not None
            and node1.room_node is node2.room_node
        )

    def sample_line_pixels(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        steps = max(steps, 2)
        xs = np.linspace(x1, x2, steps).round().astype(np.int32)
        ys = np.linspace(y1, y2, steps).round().astype(np.int32)
        xs = np.clip(xs, 0, self.map_size - 1)
        ys = np.clip(ys, 0, self.map_size - 1)
        return xs, ys

    def line_unobstructed(self, p1, p2, min_free_ratio=0.90, ignore_endpoint_px=3):
        if not hasattr(self, "free_map") or self.free_map is None:
            if not hasattr(self, "fbe_free_map"):
                return False
            fbe_free_map = self.fbe_free_map
            if torch.is_tensor(fbe_free_map):
                free_array = fbe_free_map.detach().cpu().numpy()
            else:
                free_array = np.asarray(fbe_free_map)
            if free_array.ndim == 4:
                free_array = free_array[0, 0]
            self.free_map = free_array[::-1].copy() > 0.5

        xs, ys = self.sample_line_pixels(p1, p2)
        if len(xs) > 2 * ignore_endpoint_px:
            xs = xs[ignore_endpoint_px:-ignore_endpoint_px]
            ys = ys[ignore_endpoint_px:-ignore_endpoint_px]
        if len(xs) == 0:
            return True
        free_values = self.free_map[ys, xs]
        return float(np.mean(free_values)) >= min_free_ratio

    def angle_diff_mod_pi(self, a, b):
        return abs((a - b + np.pi / 2) % np.pi - np.pi / 2)

    def estimate_wall_orientations(self, room_node=None, min_line_length_px=20):
        cache_key = (id(room_node), getattr(self, "navigate_steps", -1), min_line_length_px)
        if cache_key in self._wall_orientation_cache:
            return self._wall_orientation_cache[cache_key]
        if not hasattr(self, "full_map"):
            return []

        full_map = self.full_map
        if torch.is_tensor(full_map):
            map_array = full_map.detach().cpu().numpy()
        else:
            map_array = np.asarray(full_map)
        if map_array.ndim == 4:
            map_array = map_array[0, 0]
        obstacle = map_array[::-1] > 0.5
        img = obstacle.astype(np.uint8) * 255

        crop = img
        if room_node is not None and len(getattr(room_node, "nodes", [])) > 0:
            pts = np.array([
                node.center
                for node in room_node.nodes
                if getattr(node, "center", None) is not None
            ])
            if len(pts) > 0:
                x0 = max(int(np.min(pts[:, 0])) - 80, 0)
                y0 = max(int(np.min(pts[:, 1])) - 80, 0)
                x1 = min(int(np.max(pts[:, 0])) + 80, self.map_size - 1)
                y1 = min(int(np.max(pts[:, 1])) + 80, self.map_size - 1)
                crop = img[y0:y1 + 1, x0:x1 + 1]

        if crop.size == 0:
            self._wall_orientation_cache[cache_key] = []
            return []

        edges = cv2.Canny(crop, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=min_line_length_px,
            maxLineGap=5,
        )
        if lines is None:
            self._wall_orientation_cache[cache_key] = []
            return []

        angles = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            if x1 == x2 and y1 == y2:
                continue
            angles.append(math.atan2(y2 - y1, x2 - x1) % np.pi)

        if len(angles) == 0:
            self._wall_orientation_cache[cache_key] = []
            return []

        hist, bins = np.histogram(angles, bins=18, range=(0, np.pi))
        top_bins = np.argsort(hist)[-3:]
        dominant = []
        for idx in top_bins:
            if hist[idx] > 0:
                dominant.append(float((bins[idx] + bins[idx + 1]) / 2.0))
        self._wall_orientation_cache[cache_key] = dominant
        return dominant

    def line_parallel_to_room_wall(self, p1, p2, room_node=None, tol_deg=10.0):
        edge_angle = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])) % np.pi
        wall_angles = self.estimate_wall_orientations(room_node)
        if len(wall_angles) == 0:
            self.debug_stats.inc("edge_long_wall_orientation_fallback")
            if os.environ.get("SGNAV_LONG_EDGE_AXIS_FALLBACK", "0") in ["1", "true", "True"]:
                wall_angles = [0.0, np.pi / 2]
            else:
                return False

        tol = math.radians(tol_deg)
        return min(self.angle_diff_mod_pi(edge_angle, wall_angle) for wall_angle in wall_angles) <= tol
        
    def perception(self):
        if not self.agent.found_goal:
            self.agent.detect_objects(self.observations)
            if self.agent.total_steps % 2 == 0:
                room_detection_result = self.agent.glip_demo.inference(self.observations["rgb"][:,:,[2,1,0]], self.agent.rooms_captions)
                self.agent.update_room_map(self.observations, room_detection_result)

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.get_llm_response(
            prompt=prompt,
            request_type="graph_corr_object_distance",
            max_tokens=96,
            response_format={"type": "json_object"},
        )
        initial_distance = self.parse_distance_response(
            response_0,
            prompt,
            "graph_corr_object_distance",
        )
        initial_reason = self.parse_reason_response(response_0)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.get_llm_response(
            prompt=prompt,
            request_type="graph_corr_question",
            max_tokens=64,
        )
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.get_llm_response(
            prompt=prompt,
            request_type="graph_corr_answer",
            max_tokens=96,
        )
        prompt = self.prompt_graph_corr_3.format(
            f"{initial_distance:.3f}",
            response_1 + response_2,
            graph.center_node.caption,
            goal,
        )
        response_3 = self.get_llm_response(
            prompt=prompt,
            request_type="graph_corr_subgraph_distance",
            max_tokens=96,
            response_format={"type": "json_object"},
        )
        subgraph_distance = self.parse_distance_response(
            response_3,
            prompt,
            "graph_corr_subgraph_distance",
        )
        final_reason = self.parse_reason_response(response_3)
        corr_score = float(np.clip(1.0 / max(subgraph_distance, 0.25), 0.0, 1.0))
        graph.initial_distance_m = float(initial_distance)
        graph.initial_reason = initial_reason
        graph.question = strip_thinking(response_1)
        graph.answer = strip_thinking(response_2)
        graph.pred_distance_m = float(subgraph_distance)
        graph.reason = final_reason
        graph.corr_score = corr_score
        return corr_score

    def parse_distance_response(self, response, prompt, request_type):
        self.debug_stats.inc("distance_parse_total")
        value = parse_distance_m(response, default=10.0)
        if self.distance_parse_failed(response):
            self.debug_stats.inc("distance_parse_fail")
            self.debug_stats.log_response(
                request_type=request_type + "_parse_fail",
                prompt=prompt,
                response=response,
            )
        return value

    def distance_parse_failed(self, response):
        data = extract_json(response)
        if isinstance(data, dict):
            for key in ["distance", "distance_m", "meters", "metres", "value", "answer"]:
                if key in data:
                    return self.distance_parse_failed(str(data[key]))
        elif isinstance(data, (int, float)):
            return False
        return re.search(r"[-+]?\d*\.\d+|[-+]?\d+", strip_thinking(response)) is None

    def parse_reason_response(self, response):
        data = extract_json(response)
        if isinstance(data, dict):
            for key in ["reason", "explanation", "analysis", "answer"]:
                if key in data:
                    return strip_thinking(str(data[key]))
        return strip_thinking(response)

    def parse_probability_response(self, response, prompt, request_type):
        self.debug_stats.inc("probability_parse_total")
        value = parse_probability_01(response, default=0.0)
        if self.probability_parse_failed(response):
            self.debug_stats.inc("probability_parse_fail")
            self.debug_stats.log_response(
                request_type=f"{request_type}_parse_fail",
                prompt=prompt,
                response=response,
            )
        return value

    def probability_parse_failed(self, response):
        text = strip_thinking(response)
        data = extract_json(text)
        if isinstance(data, (int, float)):
            return False
        if isinstance(data, dict):
            for key in ["probability", "score", "value", "p", "answer"]:
                if key in data:
                    return self.probability_parse_failed(str(data[key]))
            return True
        return re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace("%", "")) is None
