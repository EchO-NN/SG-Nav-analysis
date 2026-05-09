import argparse
import copy
import math
import os
from matplotlib import colors
import cv2
import numpy as np
import pandas
import skimage
import torch
import habitat

from maskrcnn_benchmark.config import cfg as glip_cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from scenegraph import SceneGraph
from utils.episode_logger import EpisodeLogger

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner    
from utils.utils_fmm.mapping import Semantic_Mapping
from utils.utils_glip import *
from utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    crop_around_point,
    draw_agent,
    draw_goal,
    line_list
)


class SG_Nav_Agent():
    def __init__(self, task_config, args=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.distance_threshold = 5
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.last_segment_num = 0
        self.goal_merge_threshold = 0.8
        self.reperception_max_steps = int(getattr(self.args, "reperception_max_steps", 10))
        self.reperception_threshold = float(getattr(self.args, "reperception_threshold", 0.8))
        self.reperception_min_observations = max(
            1, int(getattr(self.args, "reperception_min_observations", 3))
        )
        self.reperception_min_dist_m = float(getattr(self.args, "reperception_min_dist_m", 0.25))
        self.reperception_same_goal_radius_m = float(
            getattr(self.args, "reperception_same_goal_radius_m", self.goal_merge_threshold)
        )
        self.rejected_goal_radius_m = float(
            getattr(self.args, "rejected_goal_radius_m", self.goal_merge_threshold)
        )
        self.rejected_goal_ttl = int(getattr(self.args, "rejected_goal_ttl", 80))
        self.found_goal_stop_distance_m = max(
            0.05, float(getattr(self.args, "found_goal_stop_distance_m", 0.35))
        )
        self.reset_reperception_state()
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        ### ------ init glip model ------ ###
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )

        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix
        
        self.goal_idx = {}
        for key in projection:
            self.goal_idx[projection[key]] = categories_21.index(projection[key])
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self)
        self.debug_sgnav = bool(getattr(self.args, "debug_sgnav", False))
        self.debug_sgnav_dir = getattr(self.args, "debug_sgnav_dir", "data/debug_sgnav")
        self.scenegraph.set_debug(self.debug_sgnav, self.debug_sgnav_dir)
        self.episode_logger = EpisodeLogger(
            log_dir=self.debug_sgnav_dir,
            enabled=self.debug_sgnav,
        )
        self.episode_logged = False
        self.gnn_graph_builder = None
        self.gnn_scorer = None
        self.gnn_logger = None
        self.gnn_raw_logger = None
        if (
            bool(getattr(self.args, "use_gnn_nav", False))
            or bool(getattr(self.args, "gnn_log", False))
            or bool(getattr(self.args, "collect_gnn_data", False))
        ):
            self.init_gnn_nav_modules()

        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        self.visualization_dir = f'data/visualization/{self.experiment_name}/'

        print('scene graph module init finish!!!')

    def add_predicates(self, model):
        predicate = Predicate('IsNearObj', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('ObjCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('IsNearRoom', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('RoomCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))

    def init_gnn_nav_modules(self):
        from gnn_data.raw_logger import GNNRawLogger

        self.gnn_raw_logger = GNNRawLogger(
            log_dir=getattr(self.args, "gnn_raw_log_dir", "data/gnn_raw/mp3d/train"),
            enabled=bool(getattr(self.args, "collect_gnn_data", False)),
            collect_every_k_fbe=getattr(self.args, "gnn_collect_every_k_fbe", 1),
            data_tag=getattr(self.args, "gnn_data_tag", "sgnav_teacher"),
        )

        if bool(getattr(self.args, "use_gnn_nav", False)) or bool(getattr(self.args, "gnn_log", False)):
            from gnn_nav.replay_logger import GNNReplayLogger
            from gnn_nav.scorer import GNNFrontierScorer
            from gnn_nav.sparse_graph_builder import DEFAULT_ROOM_NAMES, SparseDecisionGraphBuilder
            from gnn_nav.text_encoder import TextEmbeddingCache

            self.gnn_text_encoder = TextEmbeddingCache(
                cache_path="data/gnn/text_embeddings.pt",
                dim=int(getattr(self.args, "gnn_text_dim", 384)),
                device="cpu",
            )
            room_names = getattr(self.scenegraph, "rooms", DEFAULT_ROOM_NAMES)
            self.gnn_graph_builder = SparseDecisionGraphBuilder(
                text_encoder=self.gnn_text_encoder,
                map_resolution_cm=self.map_resolution,
                map_size=self.map_size,
                room_names=room_names,
                max_objects=100,
                object_knn_k=6,
                frontier_knn_k=8,
                object_radius_m=2.5,
                frontier_radius_m=4.0,
                device="cpu",
            )
            self.gnn_scorer = GNNFrontierScorer(
                checkpoint_path=getattr(self.args, "gnn_ckpt", None),
                builder=self.gnn_graph_builder,
                device=str(self.device),
                fallback_to_distance=True,
            )
            self.gnn_logger = GNNReplayLogger(
                log_dir=getattr(self.args, "gnn_log_dir", "data/gnn_replay/mp3d/train"),
                enabled=bool(getattr(self.args, "gnn_log", False)),
            )
    
    def reset(self):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = True
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_possible_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.last_segment_num = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        self.obj_goal = self.simulator._env.current_episode.object_category
        self.obj_goal_sg = self.simulator._env.current_episode.object_category
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(21)]
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self.stop_reason = ''
        self.episode_logged = False
        self.reset_reperception_state()

        self.scenegraph.reset()

    def reset_reperception_state(self):
        self.reperception_active = False
        self.reperception_goal_gps = None
        self.reperception_goal_map_xy = None
        self.reperception_source = ""
        self.reperception_score_sum = 0.0
        self.reperception_steps = 0
        self.reperception_observation_count = 0
        self.reperception_last_step = -1
        self.reperception_history = []
        self.rejected_goal_candidates = []

    def goal_gps_to_map_xy(self, goal_gps):
        """Return map pixel [x, y] in the same convention as SceneGraph node.center."""
        goal_gps = np.asarray(goal_gps, dtype=np.float32)
        x = int(self.map_size_cm / 10 + goal_gps[0] * 100 / self.resolution)
        y = int(self.map_size_cm / 10 + goal_gps[1] * 100 / self.resolution)
        x = min(max(x, 0), self.map_size - 1)
        y = min(max(y, 0), self.map_size - 1)
        return np.array([x, y], dtype=np.float32)

    def confidence_to_float(self, confidence, default=0.5):
        if confidence is None:
            self.scenegraph.debug_stats.inc("reperception_confidence_fallback")
            return float(default)
        try:
            if torch.is_tensor(confidence):
                return float(confidence.detach().cpu().item())
            value = np.asarray(confidence).reshape(-1)[0]
            return float(value)
        except Exception:
            self.scenegraph.debug_stats.inc("reperception_confidence_fallback")
            return float(default)

    def mark_goal_candidate_map(self, goal_gps):
        goal_xy = self.goal_gps_to_map_xy(goal_gps)
        x = int(goal_xy[0])
        y = int(goal_xy[1])
        thres = int(self.goal_merge_threshold * 100 / self.map_resolution)
        x0 = max(x - thres, 0)
        x1 = min(x + thres + 1, self.map_size)
        y0 = max(y - thres, 0)
        y1 = min(y + thres + 1, self.map_size)
        local = self.goal_gps_map[y0:y1, x0:x1]
        if local.size == 0:
            return
        if local.max() > 0:
            max_idx = np.unravel_index(np.argmax(local), local.shape)
            local[max_idx] += 1
        else:
            self.goal_gps_map[y, x] = 1

    def compute_reperception_score_k(self, goal_gps, confidence):
        goal_xy = self.goal_gps_to_map_xy(goal_gps)
        subgraphs = self.scenegraph.get_scored_subgraphs_for_goal(self.obj_goal_sg)
        score_graph = 0.0
        contributions = []
        for subgraph in subgraphs:
            center_xy = np.asarray(subgraph["center_xy"], dtype=np.float32)
            dist_pix = float(np.linalg.norm(center_xy - goal_xy))
            dist_m = max(dist_pix * self.map_resolution / 100.0, self.reperception_min_dist_m)
            p_sub = float(np.clip(subgraph["score"], 0.0, 1.0))
            term = p_sub / dist_m
            score_graph += term
            center_node = subgraph.get("center_node")
            contributions.append({
                "center": center_xy.tolist(),
                "center_caption": getattr(center_node, "caption", ""),
                "room": subgraph.get("room", ""),
                "p_sub": p_sub,
                "dist_m": dist_m,
                "term": term,
            })
        score_k = float(confidence) * float(score_graph)
        return score_k, contributions

    def is_rejected_goal_candidate(self, goal_gps):
        goal_gps = np.asarray(goal_gps, dtype=np.float32)
        kept = []
        rejected = False
        for item in self.rejected_goal_candidates:
            if self.total_steps - item["step"] <= self.rejected_goal_ttl:
                kept.append(item)
                if np.linalg.norm(goal_gps - item["gps"]) <= self.rejected_goal_radius_m:
                    rejected = True
        self.rejected_goal_candidates = kept
        return rejected

    def start_or_update_reperception_candidate(self, goal_gps, confidence, source):
        goal_gps = np.asarray(goal_gps, dtype=np.float32)
        confidence = float(confidence)
        if self.is_rejected_goal_candidate(goal_gps):
            self.scenegraph.debug_stats.inc("reperception_rejected_blacklist")
            return "rejected_blacklist"

        same_candidate_same_step = (
            self.reperception_active
            and self.reperception_last_step == self.total_steps
            and self.reperception_goal_gps is not None
            and np.linalg.norm(goal_gps - self.reperception_goal_gps) <= self.reperception_same_goal_radius_m
        )
        if (
            not self.reperception_active
            or self.reperception_goal_gps is None
            or np.linalg.norm(goal_gps - self.reperception_goal_gps) > self.reperception_same_goal_radius_m
        ):
            self.reperception_active = True
            self.reperception_goal_gps = goal_gps.copy()
            self.reperception_goal_map_xy = self.goal_gps_to_map_xy(goal_gps)
            self.reperception_source = source
            self.reperception_score_sum = 0.0
            self.reperception_steps = 0
            self.reperception_observation_count = 0
            self.reperception_history = []
            self.scenegraph.debug_stats.inc("reperception_candidates_started")

        self.reperception_goal_gps = goal_gps.copy()
        self.reperception_goal_map_xy = self.goal_gps_to_map_xy(goal_gps)
        self.reperception_source = source
        if same_candidate_same_step:
            self.found_goal = False
            self.found_possible_goal = True
            self.possible_goal_temp_gps = self.reperception_goal_gps.copy()
            self.scenegraph.debug_stats.inc("reperception_duplicate_observation")
            return "pending"

        score_k, contributions = self.compute_reperception_score_k(goal_gps, confidence)
        self.reperception_score_sum += score_k
        self.reperception_steps += 1
        self.reperception_observation_count += 1
        self.reperception_last_step = self.total_steps
        top_contributions = sorted(contributions, key=lambda x: x["term"], reverse=True)[:3]
        history_item = {
            "step": int(self.total_steps),
            "source": source,
            "confidence": confidence,
            "score_k": float(score_k),
            "score_sum": float(self.reperception_score_sum),
            "observation_count": int(self.reperception_observation_count),
            "num_subgraphs": len(contributions),
            "top_contributions": top_contributions,
            "status": "pending",
        }
        self.reperception_history.append(history_item)
        self.scenegraph.debug_stats.inc("reperception_observations")

        score_ready = self.reperception_score_sum >= self.reperception_threshold
        enough_observations = self.reperception_observation_count >= self.reperception_min_observations
        if score_ready and enough_observations and self.reperception_steps < self.reperception_max_steps:
            history_item["status"] = "confirmed"
            self.confirm_reperception_goal()
            return "confirmed"

        if self.reperception_steps >= self.reperception_max_steps:
            history_item["status"] = "rejected"
            self.reject_reperception_goal(reason="credibility_below_threshold")
            return "rejected"

        if score_ready and not enough_observations:
            history_item["status"] = "pending_min_observations"
            self.scenegraph.debug_stats.inc("reperception_wait_min_observations")

        self.found_goal = False
        self.found_possible_goal = True
        self.possible_goal_temp_gps = self.reperception_goal_gps.copy()
        self.found_goal_times = self.reperception_score_sum
        return "pending"

    def tick_reperception_without_observation(self, source):
        if (
            not self.reperception_active
            or self.reperception_goal_gps is None
            or self.reperception_last_step == self.total_steps
        ):
            return

        self.reperception_steps += 1
        self.reperception_last_step = self.total_steps
        history_item = {
            "step": int(self.total_steps),
            "source": source,
            "confidence": 0.0,
            "score_k": 0.0,
            "score_sum": float(self.reperception_score_sum),
            "observation_count": int(self.reperception_observation_count),
            "num_subgraphs": 0,
            "top_contributions": [],
            "status": "pending",
        }
        self.reperception_history.append(history_item)
        self.scenegraph.debug_stats.inc("reperception_missed_observations")

        if self.reperception_steps >= self.reperception_max_steps:
            history_item["status"] = "rejected"
            self.reject_reperception_goal(reason="candidate_not_reconfirmed")
            return

        self.found_goal = False
        self.found_possible_goal = True
        self.possible_goal_temp_gps = self.reperception_goal_gps.copy()
        self.found_goal_times = self.reperception_score_sum

    def confirm_reperception_goal(self):
        if self.reperception_goal_gps is None:
            return
        self.goal_gps = self.reperception_goal_gps.copy()
        self.found_goal = True
        self.found_possible_goal = False
        self.found_goal_times = self.reperception_score_sum
        self.reperception_active = False
        self.scenegraph.debug_stats.inc("reperception_confirmed")

    def reject_reperception_goal(self, reason):
        if self.reperception_goal_gps is not None:
            self.rejected_goal_candidates.append({
                "gps": self.reperception_goal_gps.copy(),
                "step": int(self.total_steps),
                "reason": reason,
            })
        self.found_goal = False
        self.found_possible_goal = False
        self.found_goal_times = 0
        self.goal_gps_map.fill(0)
        self.reperception_active = False
        self.reperception_goal_gps = None
        self.reperception_goal_map_xy = None
        self.reperception_source = ""
        self.reperception_score_sum = 0.0
        self.reperception_steps = 0
        self.reperception_observation_count = 0
        self.scenegraph.debug_stats.inc("reperception_rejected")
        
    def detect_objects(self, observations):
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # GLIP object detection, time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)

        
        shortest_distance = 120
        shortest_distance_angle = 0
        obj_labels = self.current_obj_predictions.get_field("labels")
        obj_scores = self.current_obj_predictions.get_field("scores")
        goal_detections = []
        for j, label in enumerate(obj_labels):
            score = self.confidence_to_float(obj_scores[j])
            if self.obj_goal in label:
                goal_detections.append({
                    "bbox": self.current_obj_predictions.bbox[j],
                    "score": score,
                })
            elif self.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_detections.append({
                    "bbox": self.current_obj_predictions.bbox[j],
                    "score": score,
                })
        
        for j, label in enumerate(obj_labels):
            if label in categories_21_origin:
                confidence = self.confidence_to_float(self.current_obj_predictions.get_field("scores")[j])
                bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.depth[center_point[1],center_point[0],0]
                if temp_distance >= self.distance_threshold:
                    continue
                obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
                self.obj_locations[categories_21_origin.index(label)].append([confidence, x, y])
        
        if self.scenegraph.obj_goal in self.scenegraph.small_objects:
            self.segment_num = len(self.scenegraph.segment2d_results)
            goal_masks = []
            if self.segment_num > self.last_segment_num:
                self.last_segment_num = self.segment_num
                segment2d_result = self.scenegraph.segment2d_results[-1]
                for index, element in enumerate(segment2d_result['caption']):
                    if self.obj_goal_sg in element.split(' '):
                        for node in self.scenegraph.nodes:
                            if node.is_goal_node and node.object['image_idx'][-1] == len(self.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                                confidence = None
                                if "confidence" in segment2d_result:
                                    confidence = segment2d_result["confidence"][index]
                                elif "conf" in node.object and len(node.object["conf"]) > 0:
                                    confidence = node.object["conf"][-1]
                                goal_masks.append({
                                    "mask": segment2d_result['mask'][index],
                                    "confidence": self.confidence_to_float(confidence),
                                })
                                break
            if len(goal_masks) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                for item in goal_masks:
                    mask = item["mask"]
                    confidence = item["confidence"]
                    center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
                    center_point = torch.tensor([center_point[1], center_point[0]])
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    k = 0
                    pos_neg = 1
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])

                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    if self.is_rejected_goal_candidate(goal_gps):
                        continue
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    else:
                        status = self.start_or_update_reperception_candidate(
                            goal_gps=goal_gps,
                            confidence=confidence,
                            source="groundedsam_mask",
                        )
                        if status == "confirmed":
                            break
                    
                    ## select the closest goal
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                if (
                    not self.found_goal
                    and not possible_goal_detected_before
                    and self.found_possible_goal
                    and not self.reperception_active
                ):
                    # if detected a long goal before, then don't change it until see a goal within 5 meters
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            else:
                if self.found_goal:
                    self.found_goal = False
                    self.found_goal_times = 0
            self.tick_reperception_without_observation("groundedsam_mask_missing")
            return
        else:
            if len(goal_detections) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                for detection in goal_detections:
                    box = detection["bbox"].to(torch.int64)
                    confidence = detection["score"]
                    center_point = (box[:2] + box[2:]) // 2
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    k = 0
                    pos_neg = 1
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])

                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    if self.is_rejected_goal_candidate(goal_gps):
                        continue
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    else:
                        self.mark_goal_candidate_map(goal_gps)
                        status = self.start_or_update_reperception_candidate(
                            goal_gps=goal_gps,
                            confidence=confidence,
                            source="glip_bbox",
                        )
                        if status == "confirmed":
                            break
                    
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                if (
                    not self.found_goal
                    and not possible_goal_detected_before
                    and self.found_possible_goal
                    and not self.reperception_active
                ):
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            self.tick_reperception_without_observation("glip_bbox_missing")
            return
                        
    def act(self, observations):
        if self.total_steps >= 500:
            self.stop_reason = 'max_episode_steps'
            return {"action": 0}
        
        self.total_steps += 1
        if self.navigate_steps == 0:
            self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self.obj_goal]]
            self.prob_array_obj = self.co_occur_mtx[self.goal_idx[self.obj_goal]]

        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        self.rgb_visualization = observations["rgb"]

        self.scenegraph.set_agent(self)
        self.scenegraph.set_navigate_steps(self.navigate_steps)
        self.scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg)
        self.scenegraph.set_room_map(self.room_map)
        self.scenegraph.set_fbe_free_map(self.fbe_free_map)
        self.scenegraph.set_observations(observations)
        self.scenegraph.set_full_map(self.full_map)
        self.scenegraph.set_full_pose(self.full_pose)
        self.scenegraph.update_scenegraph()
        
        self.update_map(observations)
        self.update_free_map(observations)
        
        if self.total_steps == 1:
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            return {"action": 5}
        elif self.total_steps <= 7:
            return {"action": 6}
        elif self.total_steps == 8:
            self.sem_map_module.set_view_angles(60)
            self.free_map_module.set_view_angles(60)
            return {"action": 5}
        elif self.total_steps <= 14:
            return {"action": 6}
        elif self.total_steps <= 15:
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            return {"action": 4}
        elif self.total_steps <= 16:
            self.sem_map_module.set_view_angles(0)
            self.free_map_module.set_view_angles(0)
            return {"action": 4}
        if self.total_steps <= 22 and not self.found_goal:
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            self.detect_objects(observations)
            room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
            self.update_room_map(observations, room_detection_result)
            if not self.found_goal: # if found a goal, directly go to it
                return {"action": 6}
                    
        if np.linalg.norm(observations["gps"] - self.last_gps) >= 0.05:
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            
        self.last_gps = observations["gps"]
        
        self.scenegraph.perception()
          
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        if self.found_goal: 
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_possible_goal: 
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif self.first_fbe:
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = False
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
        
        # local policy
        stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        if self.found_possible_goal and number_action == 0:
            self.found_possible_goal = False
        
        # reach long-term goal and fbe
        if (not self.found_goal and not self.found_possible_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20): 
            if (self.using_random_goal and self.move_since_random > 20):
                goal_x, goal_y = np.where(self.goal_map == 1)
                x_0 = max(goal_x[0] - 8, 0)
                y_0 = max(goal_y[0] - 8, 0)
                x_1 = min(goal_x[0] + 8, self.map_size)
                y_1 = min(goal_y[0] + 8, self.map_size)
                self.fbe_free_map[x_0:x_1, y_0:y_1] = 0
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        self.loop_time = 0
        while (not self.found_goal and number_action == 0) or self.not_move_steps >= 7:
            if self.not_move_steps >= 7:
                self.found_goal = False
                self.found_possible_goal = False
            self.loop_time += 1
            self.random_this_ex += 1
            if self.loop_time > 20:
                self.stop_reason = 'no_valid_plan_after_random_retries'
                return {"action": 0}
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        if number_action == 0:
            if self.found_goal:
                self.stop_reason = 'planner_stop_after_found_goal'
            else:
                self.stop_reason = 'planner_stop_without_confirmed_goal'
        else:
            self.stop_reason = 'running'
        if self.args.visualize:
            self.update_visualization_text(number_action)
            self.visualize(traversible, observations, number_action)

        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        self.navigate_steps += 1
        torch.cuda.empty_cache()
        
        return {"action": number_action}
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels

    def get_gnn_frontier_clusters(self, frontier_map, fmm_dist_m):
        from gnn_nav.frontier_clustering import cluster_frontiers

        if hasattr(frontier_map, "detach"):
            frontier_np = frontier_map.detach().cpu().numpy()
        else:
            frontier_np = np.asarray(frontier_map)
        fmm_dist_np = np.asarray(fmm_dist_m, dtype=np.float32)
        if fmm_dist_np.shape != frontier_np.shape:
            raise ValueError(
                f"GNN frontier shape mismatch: frontier={frontier_np.shape}, fmm_dist={fmm_dist_np.shape}"
            )
        return cluster_frontiers(
            frontier_map=frontier_np,
            fmm_dist=fmm_dist_np,
            min_path_dist=1.6,
            max_frontiers=int(getattr(self.args, "gnn_max_frontiers", 32)),
        )

    def get_gnn_episode_metadata(self):
        episode = None
        try:
            episode = self.simulator._env.current_episode
        except Exception:
            episode = None
        return {
            "scene_id": getattr(episode, "scene_id", "unknown_scene"),
            "episode_id": getattr(episode, "episode_id", "unknown_episode"),
            "step_id": int(getattr(self, "total_steps", 0)),
            "goal_text": getattr(self, "obj_goal_sg", ""),
            "agent_pose": self.full_pose.detach().cpu() if hasattr(self.full_pose, "detach") else self.full_pose,
            "map_size": int(self.map_size),
            "map_resolution_cm": float(self.map_resolution),
        }

    def log_gnn_raw_sample(
        self,
        frontier_map,
        fmm_dist,
        frontier_locations_all_rc,
        frontier_locations_valid_rc,
        valid_indices_in_all,
        distances_valid,
        distance_inverse_valid,
        scenegraph_scores,
        distance_bias,
        total_scores,
        selected_valid_idx,
        selected_all_idx,
        selected_goal_rc,
    ):
        if self.gnn_raw_logger is None or not bool(getattr(self.args, "collect_gnn_data", False)):
            return
        try:
            from gnn_data.extract_sgnav_state import build_raw_sgnav_step_sample

            sample = build_raw_sgnav_step_sample(
                agent=self,
                frontier_map=frontier_map,
                fmm_dist=fmm_dist,
                frontier_locations_all_rc=frontier_locations_all_rc,
                frontier_locations_valid_rc=frontier_locations_valid_rc,
                valid_indices_in_all=valid_indices_in_all,
                distances_valid=distances_valid,
                distance_inverse_valid=distance_inverse_valid,
                scenegraph_scores=scenegraph_scores,
                distance_bias=distance_bias,
                total_scores=total_scores,
                selected_valid_idx=selected_valid_idx,
                selected_all_idx=selected_all_idx,
                selected_goal_rc=selected_goal_rc,
                data_tag=getattr(self.args, "gnn_data_tag", "sgnav_teacher"),
                save_maps=bool(getattr(self.args, "gnn_save_maps", False)),
                save_scenegraph_edges=bool(getattr(self.args, "gnn_save_scenegraph_edges", False)),
                compute_oracle_online=bool(getattr(self.args, "gnn_compute_oracle_online", False)),
            )
            self.gnn_raw_logger.save_step(sample)
        except Exception as exc:
            if bool(getattr(self.args, "debug_gnn", False)):
                print("[GNN] raw logging failed:", exc)

    def log_gnn_frontier_sample(self, frontier_clusters, teacher_scores=None, selected_idx=None):
        if self.gnn_logger is None or not bool(getattr(self.args, "gnn_log", False)):
            return
        if self.gnn_graph_builder is None or len(frontier_clusters) == 0:
            return

        graph = self.gnn_graph_builder.build(
            scenegraph=self.scenegraph,
            frontier_clusters=frontier_clusters,
            goal_text=self.obj_goal_sg,
            agent_pose=self.full_pose,
            full_map=self.full_map,
            free_map=self.fbe_free_map,
            room_map=self.room_map,
            current_step=getattr(self, "total_steps", 0),
        )
        self.gnn_logger.save_step(
            graph=graph,
            frontier_clusters=frontier_clusters,
            goal_text=self.obj_goal_sg,
            metadata=self.get_gnn_episode_metadata(),
            teacher_scores=teacher_scores,
            selected_idx=selected_idx,
        )

    def fbe_gnn(self, traversible, start, frontier_map, fmm_dist_m):
        if self.gnn_scorer is None:
            return None

        frontier_clusters = self.get_gnn_frontier_clusters(frontier_map, fmm_dist_m)
        if len(frontier_clusters) == 0:
            return None

        scores = self.gnn_scorer.score(
            scenegraph=self.scenegraph,
            frontier_clusters=frontier_clusters,
            goal_text=self.obj_goal_sg,
            agent_pose=self.full_pose,
            full_map=self.full_map,
            free_map=self.fbe_free_map,
            room_map=self.room_map,
            traversible=traversible,
            cur_start=start,
            current_step=getattr(self, "total_steps", 0),
        )

        if bool(getattr(self.args, "gnn_add_distance_bias", False)):
            dist_bias = np.asarray([c.distance_inverse for c in frontier_clusters], dtype=np.float32)
            scores = scores + float(getattr(self.args, "gnn_distance_weight", 1.0)) * dist_bias

        data_policy = getattr(self.args, "gnn_data_policy", "sgnav")
        if data_policy == "random":
            best_idx = int(np.random.randint(len(frontier_clusters)))
        elif data_policy == "distance":
            best_idx = int(np.argmax([c.distance_inverse for c in frontier_clusters]))
        else:
            best_idx = int(np.argmax(scores))

        if bool(getattr(self.args, "debug_gnn", False)):
            print("[GNN] goal:", self.obj_goal_sg)
            print("[GNN] num_frontier_clusters:", len(frontier_clusters))
            print("[GNN] scores:", np.asarray(scores, dtype=np.float32).tolist())
            print("[GNN] selected:", best_idx, frontier_clusters[best_idx].center_rc.tolist())

        teacher_scores = None
        if bool(getattr(self.args, "gnn_log", False)):
            try:
                cluster_centers_shifted = np.stack(
                    [cluster.center_rc for cluster in frontier_clusters],
                    axis=0,
                ) + 1
                teacher_scores = self.scenegraph.score(cluster_centers_shifted, len(frontier_clusters))
                teacher_scores = teacher_scores + 2.0 * np.asarray(
                    [cluster.distance_inverse for cluster in frontier_clusters],
                    dtype=np.float32,
                )
            except Exception as exc:
                if bool(getattr(self.args, "debug_gnn", False)):
                    print("[GNN] teacher score logging failed:", exc)

        self.log_gnn_frontier_sample(
            frontier_clusters=frontier_clusters,
            teacher_scores=teacher_scores,
            selected_idx=best_idx,
        )
        return frontier_clusters[best_idx].center_rc.astype(np.int64)
    
    def fbe(self, traversible, start):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        fmm_dist_m = fmm_dist[1:-1, 1:-1] / 20.0

        if bool(getattr(self.args, "use_gnn_nav", False)):
            gnn_goal = self.fbe_gnn(
                traversible=traversible,
                start=start,
                frontier_map=frontier_map,
                fmm_dist_m=fmm_dist_m,
            )
            if gnn_goal is not None:
                return gnn_goal

        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        scenegraph_scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
        distance_bias = 2 * distances_16_inverse
        scores = scenegraph_scores + distance_bias
        selected_valid_idx = int(np.argmax(scores))
        idx_16_max = idx_16[0][selected_valid_idx]
        goal = frontier_locations[idx_16_max] - 1
        if bool(getattr(self.args, "collect_gnn_data", False)):
            self.log_gnn_raw_sample(
                frontier_map=frontier_map,
                fmm_dist=fmm_dist_m,
                frontier_locations_all_rc=frontier_locations - 1,
                frontier_locations_valid_rc=frontier_locations_16 - 1,
                valid_indices_in_all=idx_16[0],
                distances_valid=distances_16,
                distance_inverse_valid=distances_16_inverse,
                scenegraph_scores=scenegraph_scores,
                distance_bias=distance_bias,
                total_scores=scores,
                selected_valid_idx=selected_valid_idx,
                selected_all_idx=idx_16_max,
                selected_goal_rc=goal,
            )
        if bool(getattr(self.args, "gnn_log", False)) and self.gnn_graph_builder is not None:
            try:
                frontier_clusters = self.get_gnn_frontier_clusters(frontier_map, fmm_dist_m)
                if len(frontier_clusters) > 0:
                    cluster_centers_shifted = np.stack(
                        [cluster.center_rc for cluster in frontier_clusters],
                        axis=0,
                    ) + 1
                    teacher_scores = self.scenegraph.score(cluster_centers_shifted, len(frontier_clusters))
                    teacher_scores = teacher_scores + 2.0 * np.asarray(
                        [cluster.distance_inverse for cluster in frontier_clusters],
                        dtype=np.float32,
                    )
                    selected_idx = int(
                        np.argmin(
                            np.linalg.norm(
                                np.stack([cluster.center_rc for cluster in frontier_clusters], axis=0)
                                - goal.reshape(1, 2),
                                axis=1,
                            )
                        )
                    )
                    self.log_gnn_frontier_sample(
                        frontier_clusters=frontier_clusters,
                        teacher_scores=teacher_scores,
                        selected_idx=selected_idx,
                    )
            except Exception as exc:
                if bool(getattr(self.args, "debug_gnn", False)):
                    print("[GNN] replay logging failed:", exc)
        self.scores = scores
        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def update_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
    
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
    
    def get_traversible(self, map_pred, pose_pred):
        grid = np.rint(map_pred)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    self.selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        traversible = 1 - traversible
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem)
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                for i in range(length):
                    wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1)))
                    wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1)))
                    r, c = wy, wx
                    r = int(round(r * 100 / self.map_resolution))
                    c = int(round(c * 100 / self.map_resolution))
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        if goal_found:
            decrease_stop_cond = max(
                decrease_stop_cond,
                self.planner.stop_cond - self.found_goal_stop_distance_m,
            )
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop
    
    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        self.metrics['success'] = metrics.get('success', 0)
        if self.args.visualize:
            if self.simulator._env.episode_over or self.total_steps == 500:
                self.save_video()
        if self.simulator._env.episode_over or self.total_steps == 500:
            self.log_episode_result(metrics)

    def log_episode_result(self, metrics):
        if getattr(self, "episode_logged", False):
            return
        if not getattr(self.episode_logger, "enabled", False):
            return
        self.episode_logged = True
        episode = self.simulator._env.current_episode
        edges = [edge for edge in self.scenegraph.get_edges() if edge.relation]
        row = {
            "episode_idx": self.count_episodes,
            "episode_id": getattr(episode, "episode_id", ""),
            "scene_id": getattr(episode, "scene_id", ""),
            "goal": self.obj_goal,
            "success": int(metrics.get("success", 0)),
            "spl": float(metrics.get("spl", 0.0)),
            "softspl": float(metrics.get("softspl", 0.0)),
            "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
            "total_steps": int(self.total_steps),
            "stop_reason": self.stop_reason,
            "frontier_calls": int(self.fronter_this_ex),
            "random_goal_count": int(self.random_this_ex),
            "nodes_final": len(self.scenegraph.get_nodes()),
            "edges_final": len(edges),
            "room_nodes_with_groups": sum(
                1 for room_node in self.scenegraph.room_nodes if len(room_node.group_nodes) > 0
            ),
            "goal_detection_count": float(self.found_goal_times),
            "found_goal": bool(self.found_goal),
            "found_possible_goal": bool(self.found_possible_goal),
            "reperception_active": bool(self.reperception_active),
            "reperception_steps": int(self.reperception_steps),
            "reperception_observation_count": int(self.reperception_observation_count),
            "reperception_min_observations": int(self.reperception_min_observations),
            "reperception_threshold": float(self.reperception_threshold),
            "reperception_score_sum": float(self.reperception_score_sum),
            "found_goal_stop_distance_m": float(self.found_goal_stop_distance_m),
            "reperception_rejected_count": len(self.rejected_goal_candidates),
            "reperception_history": self.reperception_history[-10:],
            "llm_parse_failures": self.scenegraph.debug_stats.summary(),
        }
        self.episode_logger.log(row)

    def update_visualization_text(self, number_action):
        nodes = self.scenegraph.get_nodes()
        edges = [edge for edge in self.scenegraph.get_edges() if edge.relation]
        self.text_node = "\n".join(
            f"{idx + 1}. {node.caption}"
            for idx, node in enumerate(nodes)
            if node.caption
        )
        self.text_edge = "\n".join(edge.text() for edge in edges)
        goal_state = "found" if self.found_goal else "possible" if self.found_possible_goal else "exploring"
        nav_target = "random" if self.using_random_goal else "frontier"
        self.explanation = (
            f"Step {self.total_steps}, action {number_action}, goal {self.obj_goal}: "
            f"{goal_state}. Current navigation target: {nav_target}. "
            f"Stop reason: {self.stop_reason}. "
            f"Distance to goal: {self.metrics['distance_to_goal']:.2f}, "
            f"SPL: {self.metrics['spl']:.2f}, SoftSPL: {self.metrics['softspl']:.2f}."
        )

    def visualize(self, traversible, observations, number_action):
        if self.args.visualize:
            save_map = copy.deepcopy(torch.from_numpy(traversible))
            gray_map = torch.stack((save_map, save_map, save_map))
            paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
            paper_map = torch.zeros_like(paper_obstacle_map)
            paper_map_trans = paper_map.permute(1,2,0)
            unknown_rgb = colors.to_rgb('#FFFFFF')
            paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
            free_rgb = colors.to_rgb('#E7E7E7')
            paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
            obstacle_rgb = colors.to_rgb('#A2A2A2')
            paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()
            paper_map_trans = paper_map_trans.permute(2,0,1)
            self.visualize_agent_and_goal(paper_map_trans)
            agent_coordinate = (int(self.history_pose[-1][0]*100/self.resolution), int((self.map_size_cm/100-self.history_pose[-1][1])*100/self.resolution))
            occupancy_map = crop_around_point((paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8), agent_coordinate, (150, 200))
            visualize_image = np.full((450, 800, 3), 255, dtype=np.uint8)
            visualize_image = add_resized_image(visualize_image, self.rgb_visualization, (10, 60), (320, 240))
            visualize_image = add_resized_image(visualize_image, occupancy_map, (340, 60), (180, 240))
            visualize_image = add_rectangle(visualize_image, (10, 60), (330, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (340, 60), (520, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 60), (790, 165), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 195), (790, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (10, 350), (790, 400), (128, 128, 128), thickness=1)
            visualize_image = add_text(visualize_image, "Observation (Goal: {})".format(self.obj_goal), (70, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Occupancy Map", (370, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Nodes", (580, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Edges", (580, 185), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "LLM Explanation", (330, 340), font_scale=0.5, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_node, 40), (550, 80), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_edge, 40), (550, 215), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.explanation, 150), (20, 370), font_scale=0.3, thickness=1)
            visualize_image = visualize_image[:, :, ::-1]
            self.visualize_image_list.append(visualize_image)

    def save_video(self):
        if len(self.visualize_image_list) == 0:
            return
        save_video_dir = os.path.join(self.visualization_dir, 'video')
        save_video_path = f'{save_video_dir}/vid_{self.count_episodes:06d}.mp4'
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        height, width, layers = self.visualize_image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video writer: {save_video_path}")
        for visualize_image in self.visualize_image_list:  
            video.write(visualize_image)
        video.release()

    def visualize_agent_and_goal(self, map):
        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self, map=map, pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
        draw_goal(agent=self, map=map, goal_size=2, color_index=1)
        return map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    parser.add_argument(
        "--num_episodes", default=None, type=int
    )
    parser.add_argument(
        "--episodes_per_scene", default=-1, type=int
    )
    parser.add_argument(
        "--shuffle_scenes", action="store_true"
    )
    parser.add_argument(
        "--config",
        default="configs/challenge_objectnav2021.local.rgbd.yaml",
        type=str,
    )
    parser.add_argument(
        "--debug_sgnav", action='store_true'
    )
    parser.add_argument(
        "--debug_sgnav_dir", default="data/debug_sgnav", type=str
    )
    parser.add_argument(
        "--reperception_min_observations", default=3, type=int
    )
    parser.add_argument(
        "--reperception_threshold", default=0.8, type=float
    )
    parser.add_argument(
        "--reperception_max_steps", default=10, type=int
    )
    parser.add_argument(
        "--reperception_min_dist_m", default=0.25, type=float
    )
    parser.add_argument(
        "--reperception_same_goal_radius_m", default=0.8, type=float
    )
    parser.add_argument(
        "--rejected_goal_radius_m", default=0.8, type=float
    )
    parser.add_argument(
        "--rejected_goal_ttl", default=80, type=int
    )
    parser.add_argument(
        "--found_goal_stop_distance_m", default=0.35, type=float
    )
    parser.add_argument(
        "--use_gnn_nav", action="store_true"
    )
    parser.add_argument(
        "--collect_gnn_data", action="store_true"
    )
    parser.add_argument(
        "--gnn_raw_log_dir", default="data/gnn_raw/mp3d/train", type=str
    )
    parser.add_argument(
        "--gnn_collect_every_k_fbe", default=1, type=int
    )
    parser.add_argument(
        "--gnn_save_maps", action="store_true"
    )
    parser.add_argument(
        "--gnn_save_scenegraph_edges", action="store_true"
    )
    parser.add_argument(
        "--gnn_compute_oracle_online", action="store_true"
    )
    parser.add_argument(
        "--gnn_data_tag", default="sgnav_teacher", type=str
    )
    parser.add_argument(
        "--gnn_ckpt", default=None, type=str
    )
    parser.add_argument(
        "--gnn_log", action="store_true"
    )
    parser.add_argument(
        "--gnn_log_dir", default="data/gnn_replay/mp3d/train", type=str
    )
    parser.add_argument(
        "--gnn_max_frontiers", default=32, type=int
    )
    parser.add_argument(
        "--gnn_text_dim", default=384, type=int
    )
    parser.add_argument(
        "--gnn_add_distance_bias", action="store_true"
    )
    parser.add_argument(
        "--gnn_distance_weight", default=1.0, type=float
    )
    parser.add_argument(
        "--gnn_data_policy", default="sgnav", choices=["sgnav", "distance", "random", "gnn"], type=str
    )
    parser.add_argument(
        "--debug_gnn", action="store_true"
    )
    args = parser.parse_args()
    os.environ["CHALLENGE_CONFIG_FILE"] = args.config
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = SG_Nav_Agent(task_config=config, args=args)

    challenge_kwargs = {
        "eval_remote": False,
        "split_l": args.split_l,
        "split_r": args.split_r,
    }
    if args.episodes_per_scene > 0:
        challenge_kwargs["max_scene_repeat_episodes"] = args.episodes_per_scene
    if args.shuffle_scenes:
        challenge_kwargs["iterator_shuffle"] = True

    try:
        challenge = habitat.Challenge(**challenge_kwargs)
    except TypeError as exc:
        if "max_scene_repeat_episodes" not in str(exc) and "iterator_shuffle" not in str(exc):
            raise
        challenge = habitat.Challenge(
            eval_remote=False,
            split_l=args.split_l,
            split_r=args.split_r,
        )
        env = getattr(challenge, "_env", None)
        config_env = getattr(env, "_config", None)
        if env is not None and config_env is not None:
            try:
                config_env.defrost()
                if args.episodes_per_scene > 0:
                    config_env.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = int(
                        args.episodes_per_scene
                    )
                    config_env.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
                if args.shuffle_scenes:
                    config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
                config_env.freeze()
                env._setup_episode_iterator()
                print(
                    "[GNN] configured episode iterator after Challenge init: "
                    f"episodes_per_scene={args.episodes_per_scene}, "
                    f"shuffle_scenes={bool(args.shuffle_scenes)}"
                )
            except Exception as iterator_exc:
                print("[GNN] failed to configure episode iterator:", iterator_exc)

    challenge.submit(agent, num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
