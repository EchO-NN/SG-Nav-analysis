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
from utils.fbe_trace_logger import FBETraceLogger

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


def startup_log(message):
    print(f"[SGNAV_STARTUP] {message}", flush=True)


class SG_Nav_Agent():
    def __init__(self, task_config, args=None):
        startup_log("agent init begin")
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.force_cpu = os.environ.get("SGNAV_FORCE_CPU", "0") not in [
            "0",
            "false",
            "False",
        ]
        self.device = (
            torch.device("cpu")
            if self.force_cpu
            else (
                torch.device("cuda:{}".format(0))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        startup_log(f"torch device={self.device}")
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
        self.max_episode_steps = 500
        self.paper_reperception_mode = bool(
            getattr(self.args, "paper_reperception_mode", False)
        )
        self.disable_extra_stop_verification = bool(
            getattr(self.args, "disable_extra_stop_verification", False)
        )
        self.stop_liveness_last_steps = max(
            0, int(getattr(self.args, "stop_liveness_last_steps", 5))
        )
        self.reperception_max_steps = int(getattr(self.args, "reperception_max_steps", 10))
        self.reperception_threshold = float(getattr(self.args, "reperception_threshold", 0.8))
        if self.paper_reperception_mode:
            self.reperception_max_steps = 10
            self.reperception_threshold = 0.8
        self.reperception_min_observations = max(
            1, int(getattr(self.args, "reperception_min_observations", 3))
        )
        self.reperception_score_norm = getattr(
            self.args, "reperception_score_norm", "weighted_mean"
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
            0.05, float(getattr(self.args, "found_goal_stop_distance_m", 0.30))
        )
        self.stop_verification_steps = max(
            0, int(getattr(self.args, "stop_verification_steps", 4))
        )
        self.stop_verification_min_hits = max(
            1, int(getattr(self.args, "stop_verification_min_hits", 2))
        )
        self.stop_verification_same_goal_radius_m = float(
            getattr(self.args, "stop_verification_same_goal_radius_m", 1.0)
        )
        self.stop_verification_max_detection_distance_m = float(
            getattr(self.args, "stop_verification_max_detection_distance_m", 2.5)
        )
        self.stop_verification_goal_node_radius_m = float(
            getattr(self.args, "stop_verification_goal_node_radius_m", 2.0)
        )
        self.stop_verification_turn_action = int(
            getattr(self.args, "stop_verification_turn_action", 3)
        )
        self.direct_goal_approach_enabled = bool(
            int(getattr(self.args, "direct_goal_approach_enabled", 1))
        )
        self.direct_goal_approach_min_distance_m = float(
            getattr(self.args, "direct_goal_approach_min_distance_m", 0.35)
        )
        self.direct_goal_approach_turn_threshold_deg = float(
            getattr(self.args, "direct_goal_approach_turn_threshold_deg", 20.0)
        )
        self.direct_goal_approach_max_detection_distance_m = float(
            getattr(self.args, "direct_goal_approach_max_detection_distance_m", 4.5)
        )
        self.direct_goal_approach_max_steps = max(
            0, int(getattr(self.args, "direct_goal_approach_max_steps", 20))
        )
        self.reset_reperception_state()
        self.reset_stop_verification_state()
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        ### ------ init glip model ------ ###
        startup_log("GLIP init begin")
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cpu" if self.force_cpu else "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )
        startup_log("GLIP init done")

        startup_log("mapping init begin")
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
        startup_log("mapping init done")
        
        startup_log("co-occurrence matrices load begin")
        self.goal_idx = {}
        for key in projection:
            self.goal_idx[projection[key]] = categories_21.index(projection[key])
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        startup_log("co-occurrence matrices load done")
        
        startup_log("scene graph init begin")
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self)
        self.debug_sgnav = bool(getattr(self.args, "debug_sgnav", False))
        self.debug_sgnav_dir = getattr(self.args, "debug_sgnav_dir", "data/debug_sgnav")
        self.scenegraph.set_debug(self.debug_sgnav, self.debug_sgnav_dir)
        self.episode_logger = EpisodeLogger(
            log_dir=self.debug_sgnav_dir,
            enabled=self.debug_sgnav,
        )
        self.debug_fbe_trace = bool(getattr(self.args, "debug_fbe_trace", False))
        self.fbe_trace_logger = FBETraceLogger(
            enabled=self.debug_fbe_trace,
            log_dir=getattr(self.args, "fbe_trace_dir", "data/debug_fbe"),
        )
        self.random_goal_reasons = []
        self.fbe_frontier_count_valid_history = []
        self.episode_logged = False

        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        self.visualization_dir = f'data/visualization/{self.experiment_name}/'

        startup_log("scene graph module init finish")

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
        self.fbe_trace_logger.start_episode(self.count_episodes)
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
        self.random_goal_reasons = []
        self.fbe_frontier_count_valid_history = []
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self.stop_reason = ''
        self.episode_logged = False
        self.reset_reperception_state()
        self.reset_stop_verification_state()

        self.scenegraph.reset()

    def reset_stop_verification_state(self, clear_history=True):
        self.stop_verification_active = False
        self.stop_verification_target_gps = None
        self.stop_verification_steps_taken = 0
        self.stop_verification_hits = 0
        self.stop_verification_consecutive_failures = 0
        self.direct_goal_approach_steps = 0
        if clear_history:
            self.stop_verification_history = []

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
        self.last_reperception_score_graph = 0.0
        self.last_stop_liveness_decision = {}

    def goal_gps_to_map_xy(self, goal_gps):
        """Return map pixel [x, y] in the same convention as SceneGraph node.center."""
        goal_gps = np.asarray(goal_gps, dtype=np.float32)
        x = int(self.map_size_cm / 10 + goal_gps[0] * 100 / self.resolution)
        y = int(self.map_size_cm / 10 + goal_gps[1] * 100 / self.resolution)
        x = min(max(x, 0), self.map_size - 1)
        y = min(max(y, 0), self.map_size - 1)
        return np.array([x, y], dtype=np.float32)

    def goal_gps_to_map_rc(self, goal_gps):
        """Return map pixel [row, col] for arrays indexed as map[row, col]."""
        xy = self.goal_gps_to_map_xy(goal_gps)
        return np.array([xy[1], xy[0]], dtype=np.float32)

    def distance_to_gps(self, candidate_gps, observations=None):
        candidate_gps = np.asarray(candidate_gps, dtype=np.float32)
        if observations is None:
            observations = getattr(self, "current_observations", None)
        if observations is not None and "gps" in observations:
            agent_gps = np.asarray(observations["gps"], dtype=np.float32)
        elif getattr(self, "last_gps", None) is not None and np.all(np.isfinite(self.last_gps)):
            agent_gps = np.asarray(self.last_gps, dtype=np.float32)
        else:
            full_pose = self.full_pose.detach().cpu().numpy() if torch.is_tensor(self.full_pose) else np.asarray(self.full_pose)
            center_m = self.map_size_cm / 100.0 / 2.0
            agent_gps = np.array([full_pose[0] - center_m, center_m - full_pose[1]], dtype=np.float32)
        return float(np.linalg.norm(agent_gps - candidate_gps))

    def get_stop_liveness_candidate_gps(self):
        if getattr(self, "found_goal", False) and getattr(self, "goal_gps", None) is not None:
            return np.asarray(self.goal_gps, dtype=np.float32), "confirmed_goal"
        if (
            getattr(self, "reperception_active", False)
            and getattr(self, "reperception_goal_gps", None) is not None
        ):
            return np.asarray(self.reperception_goal_gps, dtype=np.float32), "active_reperception"
        if (
            getattr(self, "found_possible_goal", False)
            and getattr(self, "possible_goal_temp_gps", None) is not None
        ):
            return np.asarray(self.possible_goal_temp_gps, dtype=np.float32), "possible_goal"
        return None, ""

    def should_stop_near_candidate_goal(self):
        """Liveness guard for credible nearby goal candidates."""
        candidate_gps, source = self.get_stop_liveness_candidate_gps()
        if candidate_gps is None or not np.all(np.isfinite(candidate_gps)):
            self.last_stop_liveness_decision = {"should_stop": False, "reason": "no_candidate"}
            return False

        dist_m = self.distance_to_gps(candidate_gps)
        if dist_m > self.found_goal_stop_distance_m:
            self.last_stop_liveness_decision = {
                "should_stop": False,
                "reason": "candidate_not_close",
                "source": source,
                "distance_m": dist_m,
            }
            return False

        score_ready = self.reperception_score_sum >= self.reperception_threshold
        node_support = self.get_goal_node_support(candidate_gps)
        node_supported = bool(node_support.get("supported", False))
        verification_hits_supported = (
            getattr(self, "stop_verification_hits", 0)
            >= max(1, self.stop_verification_min_hits - 1)
        )
        detector_supported = (
            verification_hits_supported
            or (
                source == "active_reperception"
                and self.reperception_observation_count > 0
                and score_ready
            )
            or (
                source == "possible_goal"
                and self.found_goal_times >= self.reperception_threshold
            )
        )
        last_step_override = (
            self.stop_liveness_last_steps > 0
            and self.total_steps >= self.max_episode_steps - self.stop_liveness_last_steps
            and (score_ready or detector_supported or node_supported)
        )
        credible = bool(
            self.found_goal
            or (score_ready and (node_supported or detector_supported))
            or verification_hits_supported
            or last_step_override
        )
        self.last_stop_liveness_decision = {
            "should_stop": credible,
            "reason": "near_credible_goal_candidate" if credible else "candidate_not_credible",
            "source": source,
            "candidate_gps": candidate_gps.tolist(),
            "distance_m": dist_m,
            "score_ready": bool(score_ready),
            "score_sum": float(self.reperception_score_sum),
            "node_support": node_support,
            "detector_supported": bool(detector_supported),
            "verification_hits": int(getattr(self, "stop_verification_hits", 0)),
            "last_step_override": bool(last_step_override),
        }
        if credible:
            self.scenegraph.debug_stats.inc("stop_liveness_guard")
            self.scenegraph.debug_stats.inc("near_credible_goal_candidate")
        return credible

    def apply_stop_liveness_guard(self):
        if self.disable_extra_stop_verification:
            return False
        if not self.should_stop_near_candidate_goal():
            return False
        candidate_gps = np.asarray(
            self.last_stop_liveness_decision.get("candidate_gps"),
            dtype=np.float32,
        )
        if candidate_gps.size == 2 and np.all(np.isfinite(candidate_gps)):
            self.goal_gps = candidate_gps.copy()
        self.found_goal = True
        self.found_possible_goal = False
        self.reperception_active = False
        self.stop_reason = "near_credible_goal_candidate"
        return True

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
        weight_sum = 0.0
        contributions = []
        for subgraph in subgraphs:
            center_xy = np.asarray(subgraph["center_xy"], dtype=np.float32)
            dist_pix = float(np.linalg.norm(center_xy - goal_xy))
            dist_m = max(dist_pix * self.map_resolution / 100.0, self.reperception_min_dist_m)
            p_sub = float(np.clip(subgraph["score"], 0.0, 1.0))
            weight = 1.0 / dist_m
            term = p_sub * weight
            score_graph += term
            weight_sum += weight
            center_node = subgraph.get("center_node")
            contributions.append({
                "center": center_xy.tolist(),
                "center_caption": getattr(center_node, "caption", ""),
                "room": subgraph.get("room", ""),
                "p_sub": p_sub,
                "dist_m": dist_m,
                "weight": weight,
                "term": term,
            })
        if self.reperception_score_norm == "weighted_mean":
            score_graph = score_graph / max(weight_sum, 1e-6)
        self.last_reperception_score_graph = float(score_graph)
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
            "score_graph": float(self.last_reperception_score_graph),
            "score_norm": self.reperception_score_norm,
            "observation_count": int(self.reperception_observation_count),
            "num_subgraphs": len(contributions),
            "top_contributions": top_contributions,
            "status": "pending",
        }
        self.reperception_history.append(history_item)
        self.scenegraph.debug_stats.inc("reperception_observations")

        score_ready = self.reperception_score_sum >= self.reperception_threshold
        enough_observations = (
            self.paper_reperception_mode
            or self.reperception_observation_count >= self.reperception_min_observations
        )
        if score_ready and enough_observations:
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
        if self.paper_reperception_mode:
            self.scenegraph.debug_stats.inc("paper_reperception_confirmed")

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

    def goal_label_matches(self, label):
        label = str(label).lower().replace(" ", "_")
        goal = str(self.obj_goal).lower().replace(" ", "_")
        if goal == "gym_equipment":
            return label in ["gym_equipment", "treadmill", "exercise_machine"]
        if goal == "chest_of_drawers":
            return label in ["chest_of_drawers", "drawers"] or "drawers" in label
        if goal == "tv_monitor":
            return label in ["tv_monitor", "tv"] or label == "television"
        return goal == label or goal in label

    def estimate_goal_from_bbox(self, observations, bbox):
        box = bbox.to(torch.int64)
        center_point = (box[:2] + box[2:]) // 2
        width = self.config.SIMULATOR.RGB_SENSOR.WIDTH
        height = self.config.SIMULATOR.RGB_SENSOR.HEIGHT
        x = int(np.clip(center_point[0].item(), 0, width - 1))
        y = int(np.clip(center_point[1].item(), 0, height - 1))
        temp_distance = float(self.depth[y, x, 0])
        k = 0
        pos_neg = 1
        while (
            temp_distance >= 100
            and 0 < y + int(pos_neg * k) < height - 1
            and 0 < x + int(pos_neg * k) < width - 1
        ):
            pos_neg *= -1
            k += 0.5
            temp_distance = max(
                float(self.depth[y + int(pos_neg * k), x, 0]),
                float(self.depth[y, x + int(pos_neg * k), 0]),
            )
        if not np.isfinite(temp_distance) or temp_distance >= 100:
            return None
        hfov = self.config.SIMULATOR.RGB_SENSOR.HFOV
        temp_direction = (x - width / 2) * hfov / width
        goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
        return goal_gps, temp_distance, temp_direction

    def estimate_bbox_direction(self, bbox):
        box = bbox.to(torch.int64)
        center_point = (box[:2] + box[2:]) // 2
        width = self.config.SIMULATOR.RGB_SENSOR.WIDTH
        x = int(np.clip(center_point[0].item(), 0, width - 1))
        hfov = self.config.SIMULATOR.RGB_SENSOR.HFOV
        return float((x - width / 2) * hfov / width)

    def get_goal_node_support(self, goal_gps):
        if self.stop_verification_goal_node_radius_m <= 0:
            return {"supported": True, "best": None}
        target_xy = self.goal_gps_to_map_xy(goal_gps)
        best = None
        best_dist_m = float("inf")
        for node in self.scenegraph.nodes:
            if getattr(node, "center", None) is None:
                continue
            if not (
                getattr(node, "is_goal_node", False)
                or self.goal_label_matches(getattr(node, "caption", ""))
            ):
                continue
            center_xy = np.asarray(node.center, dtype=np.float32)
            dist_m = float(np.linalg.norm(center_xy - target_xy) * self.map_resolution / 100.0)
            if dist_m < best_dist_m:
                best_dist_m = dist_m
                best = {
                    "caption": getattr(node, "caption", ""),
                    "distance": dist_m,
                    "center": center_xy.tolist(),
                }
        return {
            "supported": best is not None and best_dist_m <= self.stop_verification_goal_node_radius_m,
            "best": best,
        }

    def observe_stop_verification(self, observations):
        prediction = self.glip_demo.inference(
            observations["rgb"][:, :, [2, 1, 0]],
            object_captions,
        )
        labels = self.get_glip_real_label(prediction)
        scores = prediction.get_field("scores")
        target_gps = self.stop_verification_target_gps
        best = None
        best_label_only = None
        best_delta = float("inf")
        for idx, label in enumerate(labels):
            if not self.goal_label_matches(label):
                continue
            score = self.confidence_to_float(scores[idx])
            label_only = {
                "label": str(label),
                "score": score,
                "distance": None,
                "delta": None,
                "direction": self.estimate_bbox_direction(prediction.bbox[idx]),
                "gps": np.asarray(target_gps, dtype=np.float32).tolist(),
                "depth_valid": False,
            }
            if best_label_only is None or score > best_label_only["score"]:
                best_label_only = label_only
            estimate = self.estimate_goal_from_bbox(observations, prediction.bbox[idx])
            if estimate is None:
                continue
            goal_gps, distance, direction = estimate
            delta = float(np.linalg.norm(goal_gps - target_gps))
            if delta < best_delta:
                best_delta = delta
                best = {
                    "label": str(label),
                    "score": score,
                    "distance": float(distance),
                    "delta": delta,
                    "direction": float(direction),
                    "gps": np.asarray(goal_gps, dtype=np.float32).tolist(),
                    "depth_valid": True,
                }

        if best is None:
            best = best_label_only
        node_support = self.get_goal_node_support(target_gps)
        hit = (
            best is not None
            and best["depth_valid"]
            and best["delta"] <= self.stop_verification_same_goal_radius_m
            and best["distance"] <= self.stop_verification_max_detection_distance_m
            and node_support["supported"]
        )
        if hit:
            self.stop_verification_hits += 1
            self.scenegraph.debug_stats.inc("stop_verification_hit")
        else:
            self.scenegraph.debug_stats.inc("stop_verification_miss")

        self.stop_verification_history.append({
            "step": int(self.total_steps),
            "hit": bool(hit),
            "hits": int(self.stop_verification_hits),
            "steps_taken": int(self.stop_verification_steps_taken + 1),
            "best_detection": best,
            "goal_node_support": node_support,
        })
        return hit, best, node_support

    def get_distance_to_stop_target(self, observations):
        if self.stop_verification_target_gps is None:
            return float("inf")
        agent_gps = np.asarray(observations["gps"], dtype=np.float32)
        target_gps = np.asarray(self.stop_verification_target_gps, dtype=np.float32)
        return float(np.linalg.norm(agent_gps - target_gps))

    def get_direct_goal_approach_action(self, best_detection, node_support, target_distance):
        if not self.direct_goal_approach_enabled:
            return None
        if best_detection is None or not node_support["supported"]:
            return None
        if not best_detection.get("depth_valid", False):
            return None
        if best_detection["delta"] > self.stop_verification_same_goal_radius_m:
            return None
        if best_detection["distance"] > self.direct_goal_approach_max_detection_distance_m:
            return None
        if target_distance <= self.direct_goal_approach_min_distance_m:
            return None

        direction = best_detection["direction"]
        if direction > self.direct_goal_approach_turn_threshold_deg:
            return 3
        if direction < -self.direct_goal_approach_turn_threshold_deg:
            return 2
        return 1

    def reject_confirmed_goal(self, reason):
        if getattr(self, "goal_gps", None) is not None:
            self.rejected_goal_candidates.append({
                "gps": np.asarray(self.goal_gps, dtype=np.float32).copy(),
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
        self.scenegraph.debug_stats.inc("stop_verification_rejected_goal")

    def handle_stop_verification(self, observations):
        if self.disable_extra_stop_verification:
            self.scenegraph.debug_stats.inc("extra_stop_verification_disabled")
            candidate_gps, source = self.get_stop_liveness_candidate_gps()
            if candidate_gps is None:
                self.last_stop_liveness_decision = {
                    "should_stop": False,
                    "reason": "no_candidate",
                }
                self.stop_reason = "stop_without_candidate_rejected"
                self.reject_confirmed_goal(reason=self.stop_reason)
                self.reset_stop_verification_state(clear_history=False)
                return False, self.stop_verification_turn_action

            self.stop_verification_target_gps = np.asarray(
                candidate_gps, dtype=np.float32
            ).copy()
            self.stop_verification_steps_taken = 0
            self.stop_verification_hits = 0
            hit, best_detection, node_support = self.observe_stop_verification(observations)

            if hit and self.should_stop_near_candidate_goal():
                return True, 0
            self.last_stop_liveness_decision = {
                "should_stop": False,
                "reason": "stop_without_current_detector_support",
                "source": source,
                "candidate_gps": np.asarray(candidate_gps, dtype=np.float32).tolist(),
                "current_detector_hit": bool(hit),
                "best_detection": best_detection,
                "node_support": node_support,
            }
            self.scenegraph.debug_stats.inc("stop_without_current_detector_rejected")
            self.stop_reason = self.last_stop_liveness_decision.get(
                "reason",
                "stop_without_verification_rejected",
            )
            self.reject_confirmed_goal(reason=self.stop_reason)
            self.reset_stop_verification_state(clear_history=False)
            return False, self.stop_verification_turn_action
        if self.stop_verification_steps == 0:
            return True, 0
        if not self.stop_verification_active:
            self.stop_verification_active = True
            self.stop_verification_target_gps = np.asarray(
                self.goal_gps, dtype=np.float32
            ).copy()
            self.stop_verification_steps_taken = 0
            self.stop_verification_hits = 0
            self.stop_verification_consecutive_failures = 0
            self.scenegraph.debug_stats.inc("stop_verification_started")

        hit, best_detection, node_support = self.observe_stop_verification(observations)
        if hit and best_detection is not None:
            self.stop_verification_consecutive_failures = 0
            self.stop_verification_target_gps = np.asarray(
                best_detection["gps"],
                dtype=np.float32,
            )
            self.goal_gps = self.stop_verification_target_gps.copy()
            node_support = self.get_goal_node_support(self.stop_verification_target_gps)
        target_distance = self.get_distance_to_stop_target(observations)
        if self.stop_verification_history:
            self.stop_verification_history[-1]["target_distance"] = target_distance
            self.stop_verification_history[-1]["updated_target_gps"] = (
                np.asarray(self.stop_verification_target_gps, dtype=np.float32).tolist()
            )
            self.stop_verification_history[-1]["goal_node_support"] = node_support

        approach_action = self.get_direct_goal_approach_action(
            best_detection,
            node_support,
            target_distance,
        )
        if approach_action is not None:
            if self.direct_goal_approach_steps >= self.direct_goal_approach_max_steps:
                self.stop_reason = "direct_goal_approach_rejected"
                self.reject_confirmed_goal(reason="direct_goal_approach_timeout")
                self.reset_stop_verification_state(clear_history=False)
                return False, self.stop_verification_turn_action
            self.direct_goal_approach_steps += 1
            self.stop_verification_consecutive_failures = 0
            self.stop_reason = "direct_goal_approach"
            self.scenegraph.debug_stats.inc("direct_goal_approach")
            return False, approach_action

        self.stop_verification_steps_taken += 1

        final_detection_seen = (
            best_detection is not None
            and node_support["supported"]
            and (
                not best_detection.get("depth_valid", False)
                or best_detection["delta"] <= self.stop_verification_same_goal_radius_m
            )
        )
        close_to_target = target_distance <= self.direct_goal_approach_min_distance_m
        enough_hits = self.stop_verification_hits >= self.stop_verification_min_hits
        if (
            self.direct_goal_approach_enabled
            and close_to_target
            and final_detection_seen
            and self.stop_verification_hits >= max(0, self.stop_verification_min_hits - 1)
        ):
            enough_hits = True

        if enough_hits:
            close_enough = (
                not self.direct_goal_approach_enabled
                or (final_detection_seen and close_to_target)
            )
            if close_enough:
                self.stop_reason = "stop_verification_confirmed"
                self.scenegraph.debug_stats.inc("stop_verification_confirmed")
                self.reset_stop_verification_state(clear_history=False)
                return True, 0

        if final_detection_seen:
            self.stop_verification_consecutive_failures = 0
        else:
            self.stop_verification_consecutive_failures += 1

        if self.stop_verification_consecutive_failures >= self.stop_verification_steps:
            self.stop_reason = "stop_verification_rejected"
            self.reject_confirmed_goal(reason="stop_verification_failed")
            self.reset_stop_verification_state(clear_history=False)
            return False, self.stop_verification_turn_action

        self.stop_reason = "stop_verification_scanning"
        return False, self.stop_verification_turn_action
        
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
            if self.goal_label_matches(label):
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
                obj_row = int(self.map_size_cm / 10 - obj_gps[1] * 100 / self.resolution)
                obj_col = int(self.map_size_cm / 10 + obj_gps[0] * 100 / self.resolution)
                self.obj_locations[categories_21_origin.index(label)].append(
                    [confidence, obj_row, obj_col]
                )
        
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
                if self.found_goal and not self.stop_verification_active:
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
        self.current_observations = observations
        if self.total_steps >= self.max_episode_steps:
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
            if self.apply_stop_liveness_guard():
                return {"action": 0}
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
                self.record_random_goal("fbe_no_valid_goal_initial")
                self.goal_map = self.set_random_goal(reason="fbe_no_valid_goal_initial")
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
                self.record_random_goal("fbe_no_valid_goal_replan")
                self.goal_map = self.set_random_goal(reason="fbe_no_valid_goal_replan")
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
                self.reset_stop_verification_state(clear_history=False)
            self.loop_time += 1
            self.random_this_ex += 1
            random_reason = (
                "agent_stuck_not_move_steps"
                if self.not_move_steps >= 7
                else "planner_stop_without_goal"
            )
            self.record_random_goal(random_reason)
            if self.loop_time > 20:
                self.stop_reason = 'no_valid_plan_after_random_retries'
                return {"action": 0}
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal(reason=random_reason)
            self.using_random_goal = True
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        verification_ran = False
        liveness_stop = self.apply_stop_liveness_guard()
        if liveness_stop:
            number_action = 0

        verification_should_run = (
            not liveness_stop
            and (
                (self.stop_verification_active and self.stop_verification_target_gps is not None)
                or (number_action == 0 and self.found_goal)
            )
        )
        if verification_should_run:
            verification_ran = True
            verified_stop, verification_action = self.handle_stop_verification(observations)
            if not verified_stop:
                number_action = verification_action
                if verification_action in [2, 3]:
                    self.not_move_steps = 0
                if (
                    self.stop_verification_active
                    and self.stop_verification_target_gps is not None
                ):
                    self.found_goal = True
                    self.found_possible_goal = False
                    self.goal_gps = np.asarray(
                        self.stop_verification_target_gps,
                        dtype=np.float32,
                    ).copy()

        if number_action == 0:
            if self.stop_reason == "near_credible_goal_candidate":
                pass
            elif self.found_goal:
                if not self.stop_reason.startswith("stop_verification_confirmed"):
                    self.stop_reason = 'planner_stop_after_found_goal'
            else:
                self.stop_reason = 'planner_stop_without_confirmed_goal'
        else:
            if not verification_ran:
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

    def record_random_goal(self, reason):
        item = {
            "step": int(self.total_steps),
            "navigate_step": int(self.navigate_steps),
            "reason": str(reason),
            "using_random_goal": bool(self.using_random_goal),
        }
        self.random_goal_reasons.append(item)
        counter_key = "random_goal_" + "".join(
            ch if ch.isalnum() else "_" for ch in str(reason).lower()
        )
        self.scenegraph.debug_stats.inc(counter_key)
        self.fbe_trace_logger.log_random_fallback(item)
        
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
            self.fbe_frontier_count_valid_history.append(0)
            self.log_fbe_trace({
                "step": int(self.total_steps),
                "navigate_step": int(self.navigate_steps),
                "frontier_count_all": 0,
                "frontier_count_valid": 0,
                "distances_16": [],
                "distance_inverse": [],
                "scenegraph_scores": [],
                "total_scores": [],
                "selected_valid_idx": None,
                "selected_goal_rc": None,
                "goal_map_rc": None,
                "fmm_dist_selected": None,
                "used_random_goal": True,
                "reason": "no_frontiers",
            }, traversible=traversible, start=start)
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
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
            self.fbe_frontier_count_valid_history.append(0)
            self.log_fbe_trace({
                "step": int(self.total_steps),
                "navigate_step": int(self.navigate_steps),
                "frontier_count_all": int(num_frontiers),
                "frontier_count_valid": 0,
                "distances_16": [],
                "distance_inverse": [],
                "scenegraph_scores": [],
                "total_scores": [],
                "selected_valid_idx": None,
                "selected_goal_rc": None,
                "goal_map_rc": None,
                "fmm_dist_selected": None,
                "used_random_goal": True,
                "reason": "no_frontiers_after_distance_filter",
            }, traversible=traversible, start=start, frontier_locations_all_rc=frontier_locations - 1)
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        scenegraph_scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
        scores = scenegraph_scores + 2 * distances_16_inverse
        selected_valid_idx = int(np.argmax(scores))
        idx_16_max = idx_16[0][selected_valid_idx]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        self.fbe_frontier_count_valid_history.append(int(num_16_frontiers))
        self.log_fbe_trace({
            "step": int(self.total_steps),
            "navigate_step": int(self.navigate_steps),
            "frontier_count_all": int(num_frontiers),
            "frontier_count_valid": int(num_16_frontiers),
            "distances_16": distances_16,
            "distance_inverse": distances_16_inverse,
            "scenegraph_scores": scenegraph_scores,
            "total_scores": scores,
            "selected_valid_idx": selected_valid_idx,
            "selected_goal_rc": goal,
            "goal_map_rc": goal,
            "fmm_dist_selected": float(fmm_dist[frontier_locations[idx_16_max][0], frontier_locations[idx_16_max][1]]),
            "used_random_goal": False,
            "reason": "selected_frontier",
            "score_mode": getattr(self.args, "sgnav_score_mode", "group"),
        }, traversible=traversible, start=start,
            selected_frontier_rc=goal,
            frontier_locations_valid_rc=frontier_locations_16 - 1,
            frontier_locations_all_rc=frontier_locations - 1)
        return goal

    def log_fbe_trace(
        self,
        sample,
        *,
        traversible=None,
        start=None,
        selected_frontier_rc=None,
        frontier_locations_valid_rc=None,
        frontier_locations_all_rc=None,
    ):
        if not self.debug_fbe_trace:
            return
        occupancy_map = self.full_map.detach().cpu().numpy()[0, 0, ::-1]
        free_map = self.fbe_free_map.detach().cpu().numpy()[0, 0, ::-1]
        candidate_goal_rc = None
        candidate_gps, _ = self.get_stop_liveness_candidate_gps()
        if candidate_gps is not None:
            candidate_goal_rc = self.goal_gps_to_map_rc(candidate_gps)
        self.fbe_trace_logger.log_decision(
            sample,
            occupancy_map=occupancy_map,
            free_map=free_map,
            agent_rc=start,
            selected_frontier_rc=selected_frontier_rc,
            candidate_goal_rc=candidate_goal_rc,
            frontier_locations_valid_rc=frontier_locations_valid_rc,
            frontier_locations_all_rc=frontier_locations_all_rc,
        )
        
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
                self.random_this_ex += 1
                self.record_random_goal("block_goal_exception")
                goal = self.set_random_goal(goal, reason="block_goal_exception")

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
    
    def set_random_goal(self, base_goal=None, reason="unspecified"):
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
        if self.simulator._env.episode_over or self.total_steps == self.max_episode_steps:
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
            "random_goal_reasons": self.random_goal_reasons[-50:],
            "fbe_frontier_count_valid_history": self.fbe_frontier_count_valid_history,
            "fbe_frontier_count_valid_summary": self.summarize_fbe_valid_counts(),
            "sgnav_score_mode": getattr(self.args, "sgnav_score_mode", "group"),
            "paper_reperception_mode": bool(self.paper_reperception_mode),
            "disable_extra_stop_verification": bool(self.disable_extra_stop_verification),
            "reperception_score_norm": self.reperception_score_norm,
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
            "stop_verification_steps": int(self.stop_verification_steps),
            "stop_verification_min_hits": int(self.stop_verification_min_hits),
            "stop_verification_same_goal_radius_m": float(
                self.stop_verification_same_goal_radius_m
            ),
            "stop_verification_max_detection_distance_m": float(
                self.stop_verification_max_detection_distance_m
            ),
            "stop_verification_goal_node_radius_m": float(
                self.stop_verification_goal_node_radius_m
            ),
            "direct_goal_approach_enabled": bool(self.direct_goal_approach_enabled),
            "direct_goal_approach_min_distance_m": float(
                self.direct_goal_approach_min_distance_m
            ),
            "direct_goal_approach_steps": int(self.direct_goal_approach_steps),
            "direct_goal_approach_max_steps": int(self.direct_goal_approach_max_steps),
            "stop_verification_consecutive_failures": int(
                self.stop_verification_consecutive_failures
            ),
            "stop_verification_history": self.stop_verification_history[-10:],
            "stop_liveness_decision": self.last_stop_liveness_decision,
            "reperception_rejected_count": len(self.rejected_goal_candidates),
            "reperception_history": self.reperception_history[-10:],
            "scenegraph_score_debug": getattr(self.scenegraph, "last_score_debug", {}),
            "llm_parse_failures": self.scenegraph.debug_stats.summary(),
        }
        self.episode_logger.log(row)

    def summarize_fbe_valid_counts(self):
        counts = list(self.fbe_frontier_count_valid_history)
        if len(counts) == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        arr = np.asarray(counts, dtype=np.float32)
        return {
            "count": int(len(counts)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

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
        "--debug_fbe_trace", action="store_true"
    )
    parser.add_argument(
        "--fbe_trace_dir", default="data/debug_fbe", type=str
    )
    parser.add_argument(
        "--paper_reperception_mode", action="store_true"
    )
    parser.add_argument(
        "--disable_extra_stop_verification", action="store_true"
    )
    parser.add_argument(
        "--stop_liveness_last_steps", default=5, type=int
    )
    parser.add_argument(
        "--sgnav_score_mode",
        default="group",
        choices=["group", "paper_object", "hybrid"],
    )
    parser.add_argument(
        "--reperception_score_norm",
        default="weighted_mean",
        choices=["paper_sum", "weighted_mean"],
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
        "--found_goal_stop_distance_m", default=0.30, type=float
    )
    parser.add_argument(
        "--stop_verification_steps", default=4, type=int
    )
    parser.add_argument(
        "--stop_verification_min_hits", default=2, type=int
    )
    parser.add_argument(
        "--stop_verification_same_goal_radius_m", default=1.0, type=float
    )
    parser.add_argument(
        "--stop_verification_max_detection_distance_m", default=2.5, type=float
    )
    parser.add_argument(
        "--stop_verification_goal_node_radius_m", default=2.0, type=float
    )
    parser.add_argument(
        "--stop_verification_turn_action", default=3, type=int
    )
    parser.add_argument(
        "--direct_goal_approach_enabled", default=1, type=int
    )
    parser.add_argument(
        "--direct_goal_approach_min_distance_m", default=0.35, type=float
    )
    parser.add_argument(
        "--direct_goal_approach_turn_threshold_deg", default=20.0, type=float
    )
    parser.add_argument(
        "--direct_goal_approach_max_detection_distance_m", default=4.5, type=float
    )
    parser.add_argument(
        "--direct_goal_approach_max_steps", default=20, type=int
    )
    parser.add_argument(
        "--edge_update_every_k", default=5, type=int
    )
    parser.add_argument(
        "--score_refresh_every_k", default=5, type=int
    )
    parser.add_argument(
        "--max_edge_proposal_per_step", default=128, type=int
    )
    parser.add_argument(
        "--disable_vlm_short_edge_check", action="store_true"
    )
    parser.add_argument(
        "--disable_llm_edges", action="store_true"
    )
    args = parser.parse_args()
    startup_log(f"args parsed split=[{args.split_l}:{args.split_r}] num_episodes={args.num_episodes}")
    os.environ["CHALLENGE_CONFIG_FILE"] = args.config
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    startup_log(f"habitat config load begin config={config_paths}")
    config = habitat.get_config(config_paths)
    startup_log("habitat config load done")
    agent = SG_Nav_Agent(task_config=config, args=args)

    startup_log("habitat challenge init begin")
    challenge = habitat.Challenge(eval_remote=False, split_l=args.split_l, split_r=args.split_r)
    startup_log("habitat challenge init done")

    startup_log("challenge submit begin")
    challenge.submit(agent, num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
