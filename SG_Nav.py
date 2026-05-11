import argparse
import copy
import math
import os
import subprocess
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

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
from utils.candidate_trace_logger import CandidateTraceLogger
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


def get_git_commit_hash_or_unknown():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


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
        self.realtime_monitor = bool(getattr(self.args, "realtime_monitor", False))
        self.realtime_monitor_every = max(
            1, int(getattr(self.args, "realtime_monitor_every", 1))
        )
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
        self.frontier_score_norm = getattr(
            self.args, "frontier_score_norm", "paper_sum"
        )
        requested_frontier_distance_weight = getattr(
            self.args, "frontier_distance_weight", None
        )
        if requested_frontier_distance_weight is None:
            requested_frontier_distance_weight = (
                0.0
                if getattr(self.args, "sgnav_score_mode", "group") == "paper_object"
                else 2.0
            )
        self.frontier_distance_weight = float(requested_frontier_distance_weight)
        self.frontier_distance_tiebreaker = max(
            0.0, float(getattr(self.args, "frontier_distance_tiebreaker", 1e-6))
        )
        self.reperception_min_dist_m = float(getattr(self.args, "reperception_min_dist_m", 0.25))
        self.reperception_same_goal_radius_m = float(
            getattr(self.args, "reperception_same_goal_radius_m", self.goal_merge_threshold)
        )
        self.rejected_goal_radius_m = float(
            getattr(self.args, "rejected_goal_radius_m", self.goal_merge_threshold)
        )
        self.rejected_goal_ttl = int(getattr(self.args, "rejected_goal_ttl", 80))
        self.rejected_goal_visible_blacklist_max_distance_m = max(
            0.1,
            float(
                getattr(
                    self.args,
                    "rejected_goal_visible_blacklist_max_distance_m",
                    2.0,
                )
            ),
        )
        self.stuck_position_blacklist_steps = max(
            1, int(getattr(self.args, "stuck_position_blacklist_steps", 50))
        )
        self.stuck_position_blacklist_radius_m = max(
            0.1, float(getattr(self.args, "stuck_position_blacklist_radius_m", 1.0))
        )
        self.candidate_stuck_blacklist_steps = max(
            1, int(getattr(self.args, "candidate_stuck_blacklist_steps", 7))
        )
        self.candidate_no_progress_blacklist_steps = max(
            1, int(getattr(self.args, "candidate_no_progress_blacklist_steps", 20))
        )
        self.candidate_progress_min_delta_m = max(
            0.0, float(getattr(self.args, "candidate_progress_min_delta_m", 0.10))
        )
        self.invalid_episode_stuck_steps = max(
            0, int(getattr(self.args, "invalid_episode_stuck_steps", 200))
        )
        self.invalid_episode_stationary_radius_m = max(
            0.05,
            float(getattr(self.args, "invalid_episode_stationary_radius_m", 0.50)),
        )
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
        self.stop_verification_force_stop_confidence = float(
            getattr(self.args, "stop_verification_force_stop_confidence", 0.85)
        )
        self.stop_require_near_visual_hit = bool(
            int(getattr(self.args, "stop_require_near_visual_hit", 1))
        )
        # Deprecated: final STOP no longer requires the hit to happen inside
        # the verification state; a near visual hit collected while approaching
        # is valid evidence too.
        self.stop_verification_require_verification_hit = bool(
            int(getattr(self.args, "stop_verification_require_verification_hit", 0))
        )
        self.stop_verification_required_hit_max_distance_m = max(
            0.05,
            float(
                getattr(
                    self.args,
                    "stop_verification_required_hit_max_distance_m",
                    1.5,
                )
            ),
        )
        self.stop_verification_anchor_radius_m = float(
            getattr(
                self.args,
                "stop_verification_anchor_radius_m",
                self.reperception_same_goal_radius_m,
            )
        )
        self.stop_verification_turn_action = int(
            getattr(self.args, "stop_verification_turn_action", 3)
        )
        self.planned_goal_arrival_scan_steps = max(
            0, int(getattr(self.args, "planned_goal_arrival_scan_steps", 8))
        )
        self.planned_goal_verification_max_observations = max(
            1,
            int(
                getattr(
                    self.args,
                    "planned_goal_verification_max_observations",
                    24,
                )
            ),
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
        self.goal_detection_min_confidence = float(
            getattr(self.args, "goal_detection_min_confidence", 0.60)
        )
        self.candidate_start_min_confidence = float(
            getattr(self.args, "candidate_start_min_confidence", 0.60)
        )
        self.planned_goal_approach_enabled = bool(
            int(getattr(self.args, "planned_goal_approach_enabled", 1))
        )
        self.planned_goal_stop_distance_m = max(
            0.05, float(getattr(self.args, "planned_goal_stop_distance_m", 0.70))
        )
        self.planned_goal_approach_max_steps = max(
            0, int(getattr(self.args, "planned_goal_approach_max_steps", 40))
        )
        planned_min_radius = getattr(
            self.args, "planned_goal_approach_min_radius_m", None
        )
        if planned_min_radius is None:
            planned_min_radius = 0.30
        self.planned_goal_approach_min_radius_m = max(
            0.05, float(planned_min_radius)
        )
        planned_max_radius = getattr(
            self.args, "planned_goal_approach_max_radius_m", None
        )
        if planned_max_radius is None:
            planned_max_radius = 0.70
        self.planned_goal_approach_max_radius_m = max(
            self.planned_goal_approach_min_radius_m, float(planned_max_radius)
        )
        self.planned_goal_approach_station_stop_distance_m = max(
            0.05,
            float(
                getattr(
                    self.args,
                    "planned_goal_approach_station_stop_distance_m",
                    0.10,
                )
            ),
        )
        self.planned_goal_approach_radius_cost = max(
            0.0,
            float(getattr(self.args, "planned_goal_approach_radius_cost", 2.0)),
        )
        self.planned_goal_retreat_enabled = bool(
            int(getattr(self.args, "planned_goal_retreat_enabled", 1))
        )
        self.planned_goal_retreat_min_radius_m = max(
            0.05,
            float(getattr(self.args, "planned_goal_retreat_min_radius_m", 1.20)),
        )
        self.planned_goal_retreat_max_radius_m = max(
            self.planned_goal_retreat_min_radius_m,
            float(getattr(self.args, "planned_goal_retreat_max_radius_m", 1.50)),
        )
        self.planned_goal_retreat_require_line_of_sight = bool(
            int(getattr(self.args, "planned_goal_retreat_require_line_of_sight", 1))
        )
        self.planned_goal_retreat_los_endpoint_skip_cells = max(
            0,
            int(getattr(self.args, "planned_goal_retreat_los_endpoint_skip_cells", 2)),
        )
        self.planned_goal_retreat_station_stop_distance_m = max(
            0.05,
            float(getattr(self.args, "planned_goal_retreat_station_stop_distance_m", 0.25)),
        )
        self.planned_goal_retreat_scan_steps = max(
            0, int(getattr(self.args, "planned_goal_retreat_scan_steps", 6))
        )
        self.planned_goal_retreat_max_steps = max(
            0, int(getattr(self.args, "planned_goal_retreat_max_steps", 15))
        )
        self.planned_goal_viewpoint_max_attempts = max(
            1, int(getattr(self.args, "planned_goal_viewpoint_max_attempts", 2))
        )
        self.planned_goal_retreat_viewpoint_attempts = max(
            1, int(getattr(self.args, "planned_goal_retreat_viewpoint_attempts", 2))
        )
        self.planned_goal_viewpoint_min_separation_m = max(
            0.0,
            float(getattr(self.args, "planned_goal_viewpoint_min_separation_m", 0.45)),
        )
        self.candidate_min_detector_hits = max(
            1, int(getattr(self.args, "candidate_min_detector_hits", 2))
        )
        candidate_strong_evidence_min_hits = getattr(
            self.args, "candidate_strong_evidence_min_hits", None
        )
        if candidate_strong_evidence_min_hits is None:
            candidate_strong_evidence_min_hits = getattr(
                self.args,
                "candidate_strong_evidence_min_consecutive_hits",
                6,
            )
        self.candidate_strong_evidence_min_hits = max(
            1, int(candidate_strong_evidence_min_hits)
        )
        # Backward-compatible alias for old configs/debug readers. The evidence is
        # cumulative now; misses do not reset this qualification.
        self.candidate_strong_evidence_min_consecutive_hits = (
            self.candidate_strong_evidence_min_hits
        )
        self.candidate_min_distinct_views = max(
            1, int(getattr(self.args, "candidate_min_distinct_views", 1))
        )
        self.candidate_min_hit_ratio = float(
            getattr(self.args, "candidate_min_hit_ratio", 0.0)
        )
        self.candidate_max_misses = max(
            1, int(getattr(self.args, "candidate_max_misses", 6))
        )
        self.candidate_miss_penalty = max(
            0.0, float(getattr(self.args, "candidate_miss_penalty", 0.20))
        )
        self.candidate_score_decay = float(
            np.clip(getattr(self.args, "candidate_score_decay", 0.85), 0.0, 1.0)
        )
        self.candidate_context_cap = float(
            np.clip(getattr(self.args, "candidate_context_cap", 0.65), 0.0, 1.0)
        )
        self.candidate_direct_match_bonus = max(
            0.0, float(getattr(self.args, "candidate_direct_match_bonus", 0.25))
        )
        self.candidate_require_detector_for_stop = bool(
            getattr(self.args, "candidate_require_detector_for_stop", True)
        )
        self.candidate_require_direct_goal_for_confirm = bool(
            int(getattr(self.args, "candidate_require_direct_goal_for_confirm", 1))
        )
        self.candidate_view_scan_max_steps = max(
            0, int(getattr(self.args, "candidate_view_scan_max_steps", 8))
        )
        self.candidate_single_hit_search_enabled = bool(
            int(getattr(self.args, "candidate_single_hit_search_enabled", 1))
        )
        self.candidate_multiview_enabled = bool(
            int(getattr(self.args, "candidate_multiview_enabled", 1))
        )
        self.candidate_multiview_min_radius_m = max(
            0.05, float(getattr(self.args, "candidate_multiview_min_radius_m", 0.70))
        )
        self.candidate_multiview_max_radius_m = max(
            self.candidate_multiview_min_radius_m,
            float(getattr(self.args, "candidate_multiview_max_radius_m", 1.10)),
        )
        self.candidate_multiview_station_stop_distance_m = max(
            0.05,
            float(getattr(self.args, "candidate_multiview_station_stop_distance_m", 0.15)),
        )
        self.candidate_multiview_min_start_distance_m = max(
            0.0,
            float(getattr(self.args, "candidate_multiview_min_start_distance_m", 0.35)),
        )
        self.current_candidate = None
        self.candidate_summaries = []
        self.candidate_sequence = 0
        self.reset_reperception_state()
        self.reset_stop_verification_state()
        self.last_planned_goal_approach_station = None
        self.last_planned_goal_retreat_station = None
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
        self.agent_footprint_clearance_cells = max(
            1, int(getattr(self.args, "agent_footprint_clearance_cells", 2))
        )
        self.traversible_start_corrections = 0
        self.explanation = ''
        self.last_frontier_explanation = {}
        
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
        self.candidate_trace_logger = CandidateTraceLogger(
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
        self.realtime_monitor_dir = getattr(
            self.args,
            "realtime_monitor_dir",
            os.path.join(self.visualization_dir, "realtime"),
        ) or os.path.join(self.visualization_dir, "realtime")
        self.realtime_monitor_latest_path = os.path.join(
            self.realtime_monitor_dir, "latest.jpg"
        )

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
        self.candidate_trace_logger.start_episode(self.count_episodes)
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
        self.current_candidate = None
        self.candidate_summaries = []
        self.candidate_sequence = 0
        self.traversible_start_corrections = 0
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.invalid_episode_stationary_steps = 0
        self.invalid_episode_stationary_anchor_gps = None
        self.stuck_position_blacklist = []
        self.explanation = ''
        self.last_frontier_explanation = {}
        self.text_node = ''
        self.text_edge = ''
        self.stop_reason = ''
        self.invalid_episode = False
        self.invalid_episode_reason = ""
        self.episode_logged = False
        self.reset_reperception_state()
        self.reset_stop_verification_state()

        self.scenegraph.reset()

    def reset_stop_verification_state(self, clear_history=True):
        self.stop_verification_active = False
        self.stop_verification_target_gps = None
        self.stop_verification_anchor_gps = None
        self.stop_verification_steps_taken = 0
        self.stop_verification_hits = 0
        self.stop_verification_near_hits = 0
        self.stop_verification_observation_count = 0
        self.stop_verification_consecutive_failures = 0
        self.direct_goal_approach_steps = 0
        self.planned_goal_approach_steps = 0
        self.planned_goal_approach_blocked_steps = 0
        self.planned_goal_arrival_scan_steps_taken = 0
        self.planned_goal_retreat_active = False
        self.planned_goal_retreat_steps = 0
        self.planned_goal_retreat_blocked_steps = 0
        self.planned_goal_retreat_scan_steps_taken = 0
        self.planned_goal_retreat_confirmed = False
        self.planned_goal_failed_viewpoints = []
        self.last_planned_goal_approach_station = None
        self.last_planned_goal_retreat_station = None
        self.last_planned_goal_approach_selection_debug = {}
        self.last_planned_goal_retreat_selection_debug = {}
        self.stop_verification_blacklist_visible_seen = False
        self.stop_verification_blacklist_visibility = None
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
        self.current_candidate = None
        self.candidate_progress_candidate_id = None
        self.candidate_progress_best_distance_m = float("inf")
        self.candidate_progress_last_distance_m = float("inf")
        self.candidate_progress_no_improve_steps = 0

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

    def goal_gps_to_goal_map(self, goal_gps):
        goal_map = np.zeros(self.full_map.shape[-2:])
        rc = self.goal_gps_to_map_rc(goal_gps).astype(np.int64)
        goal_map[
            max(0, min(self.map_size - 1, int(rc[0]))),
            max(0, min(self.map_size - 1, int(rc[1]))),
        ] = 1
        return goal_map

    def candidate_stop_distance_m(self):
        if self.planned_goal_approach_enabled:
            return self.planned_goal_stop_distance_m
        return self.found_goal_stop_distance_m

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
            self.log_candidate_event(
                "stop_guard",
                decision="wait",
                reason="no_candidate",
                source=source,
                stop_liveness_decision=self.last_stop_liveness_decision,
            )
            return False

        dist_m = self.distance_to_gps(candidate_gps)
        stop_distance_m = self.candidate_stop_distance_m()
        if dist_m > stop_distance_m:
            self.last_stop_liveness_decision = {
                "should_stop": False,
                "reason": "candidate_not_close",
                "source": source,
                "distance_m": dist_m,
                "stop_distance_m": float(stop_distance_m),
            }
            self.log_candidate_event(
                "stop_guard",
                candidate_gps=candidate_gps,
                decision="wait",
                reason="candidate_not_close",
                source=source,
                agent_distance_m=dist_m,
                stop_liveness_decision=self.last_stop_liveness_decision,
            )
            return False

        score_ready = self.reperception_score_sum >= self.reperception_threshold
        node_support = self.get_goal_node_support(candidate_gps)
        node_supported = bool(node_support.get("supported", False))
        candidate_support = self.current_candidate_detector_support_for(candidate_gps)
        verification_support = self.stop_verification_hit_support_for(candidate_gps)
        verification_hits_supported = verification_support["supported"]
        candidate_detector_supported = candidate_support["supported"]
        detector_supported = verification_hits_supported or candidate_detector_supported
        found_goal_verified = bool(self.found_goal and source == "confirmed_goal")
        graph_only_candidate = bool(
            (not self.candidate_require_detector_for_stop)
            and score_ready
            and node_supported
        )
        verified_candidate = found_goal_verified or detector_supported
        last_step_override = (
            self.stop_liveness_last_steps > 0
            and self.total_steps >= self.max_episode_steps - self.stop_liveness_last_steps
            and detector_supported
        )
        credible = bool(verified_candidate or last_step_override)
        self.last_stop_liveness_decision = {
            "should_stop": credible,
            "reason": "near_credible_goal_candidate" if credible else "candidate_not_credible",
            "source": source,
            "candidate_gps": candidate_gps.tolist(),
            "distance_m": dist_m,
            "found_goal_stop_distance_m": float(self.found_goal_stop_distance_m),
            "stop_distance_m": float(stop_distance_m),
            "planned_goal_stop_distance_m": float(self.planned_goal_stop_distance_m),
            "score_ready": bool(score_ready),
            "score_sum": float(self.reperception_score_sum),
            "node_support": node_support,
            "node_supported": bool(node_supported),
            "detector_supported": bool(detector_supported),
            "candidate_detector_supported": bool(candidate_detector_supported),
            "candidate_evidence_matches": bool(candidate_support["matches"]),
            "candidate_evidence_distance_m": float(candidate_support["distance_m"]),
            "candidate_evidence_binding_radius_m": float(
                candidate_support["binding_radius_m"]
            ),
            "current_candidate_id": candidate_support["candidate_id"],
            "candidate_hit_count": int(candidate_support["hit_count"]),
            "candidate_consecutive_hit_count": int(
                candidate_support["consecutive_hit_count"]
            ),
            "candidate_max_consecutive_hit_count": int(
                candidate_support["max_consecutive_hit_count"]
            ),
            "candidate_strong_consecutive_evidence": bool(
                candidate_support["strong_consecutive_evidence"]
            ),
            "candidate_strong_historical_evidence": bool(
                candidate_support["strong_historical_evidence"]
            ),
            "candidate_strong_evidence_min_hits": int(
                self.candidate_strong_evidence_min_hits
            ),
            "candidate_strong_evidence_min_consecutive_hits": int(
                self.candidate_strong_evidence_min_consecutive_hits
            ),
            "candidate_near_hit_count": int(candidate_support["near_hit_count"]),
            "candidate_miss_count": int(candidate_support["miss_count"]),
            "candidate_distinct_view_count": int(
                candidate_support["distinct_view_count"]
            ),
            "candidate_hit_ratio": float(candidate_support["hit_ratio"]),
            "verification_hits_supported": bool(verification_hits_supported),
            "verification_target_matches": bool(verification_support["matches"]),
            "verification_target_distance_m": float(
                verification_support["distance_m"]
            ),
            "verification_target_binding_radius_m": float(
                verification_support["binding_radius_m"]
            ),
            "verification_hits": int(getattr(self, "stop_verification_hits", 0)),
            "found_goal_verified": bool(found_goal_verified),
            "stop_verification_active": bool(getattr(self, "stop_verification_active", False)),
            "candidate_require_detector_for_stop": bool(self.candidate_require_detector_for_stop),
            "graph_only_candidate_ignored": bool(
                graph_only_candidate and not detector_supported
            ),
            "last_step_override": bool(last_step_override),
        }
        self.log_candidate_event(
            "stop_guard",
            candidate_gps=candidate_gps,
            decision="stop" if credible else "wait",
            reason=self.last_stop_liveness_decision["reason"],
            source=source,
            same_candidate_distance_m=candidate_support["distance_m"],
            agent_distance_m=dist_m,
            node_support=node_support,
            stop_liveness_decision=self.last_stop_liveness_decision,
        )
        if credible:
            self.scenegraph.debug_stats.inc("stop_liveness_guard")
            self.scenegraph.debug_stats.inc("near_credible_goal_candidate")
        elif graph_only_candidate and not detector_supported:
            self.scenegraph.debug_stats.inc("graph_only_stop_guard_ignored")
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
        restart_verification = True
        if (
            self.stop_verification_active
            and self.stop_verification_target_gps is not None
        ):
            active_target = np.asarray(
                self.stop_verification_target_gps, dtype=np.float32
            )
            restart_verification = (
                float(np.linalg.norm(active_target - candidate_gps))
                > self.stop_verification_same_goal_radius_m
            )
        if restart_verification:
            self.stop_verification_active = True
            self.stop_verification_target_gps = candidate_gps.copy()
            self.stop_verification_anchor_gps = candidate_gps.copy()
            self.stop_verification_steps_taken = 0
            self.stop_verification_hits = 0
            self.stop_verification_near_hits = 0
            self.stop_verification_observation_count = 0
            self.stop_verification_consecutive_failures = 0
            self.planned_goal_approach_steps = 0
            self.planned_goal_approach_blocked_steps = 0
            self.planned_goal_arrival_scan_steps_taken = 0
            self.planned_goal_retreat_active = False
            self.planned_goal_retreat_steps = 0
            self.planned_goal_retreat_blocked_steps = 0
            self.planned_goal_retreat_scan_steps_taken = 0
            self.planned_goal_retreat_confirmed = False
            self.planned_goal_failed_viewpoints = []
            self.last_planned_goal_approach_station = None
            self.last_planned_goal_retreat_station = None
            self.scenegraph.debug_stats.inc("near_credible_goal_verify_started")
            self.scenegraph.debug_stats.inc("stop_verification_started")
        self.stop_reason = "near_credible_goal_candidate_verify"
        return False

    def start_candidate_state(self, goal_gps, source):
        self.candidate_sequence += 1
        goal_gps = np.asarray(goal_gps, dtype=np.float32)
        candidate = {
            "candidate_id": (
                f"ep{self.count_episodes}_cand{self.candidate_sequence}_"
                f"step{self.total_steps}"
            ),
            "gps": goal_gps.copy(),
            "started_step": int(self.total_steps),
            "last_seen_step": -1,
            "source": source,
            "hit_count": 0,
            "consecutive_hit_count": 0,
            "max_consecutive_hit_count": 0,
            "near_hit_count": 0,
            "miss_count": 0,
            "view_angles": [],
            "score_sum": 0.0,
            "score_graph": 0.0,
            "score_k": 0.0,
            "top_contributions": [],
            "view_scan_steps": 0,
            "decision": "pending",
            "reason": "",
            "finalized": False,
        }
        self.current_candidate = candidate
        self.candidate_progress_candidate_id = candidate["candidate_id"]
        self.candidate_progress_best_distance_m = float("inf")
        self.candidate_progress_last_distance_m = float("inf")
        self.candidate_progress_no_improve_steps = 0
        self.scenegraph.debug_stats.inc("candidate_started")
        self.log_candidate_event(
            "start",
            candidate=candidate,
            candidate_gps=goal_gps,
            source=source,
            decision="pending",
            reason="new_candidate",
        )
        return candidate

    def current_candidate_hit_count(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0
        return int(candidate.get("hit_count", 0))

    def current_candidate_max_consecutive_hit_count(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0
        return int(candidate.get("max_consecutive_hit_count", 0))

    def current_candidate_has_strong_consecutive_evidence(self):
        return (
            self.current_candidate_hit_count()
            >= self.candidate_strong_evidence_min_hits
        )

    def current_candidate_has_strong_historical_evidence(self):
        return self.current_candidate_has_strong_consecutive_evidence()

    def current_candidate_near_hit_count(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0
        return int(candidate.get("near_hit_count", 0))

    def active_near_visual_hit_count(self):
        return int(self.current_candidate_near_hit_count()) + int(
            getattr(self, "stop_verification_near_hits", 0)
        )

    def current_candidate_miss_count(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0
        return int(candidate.get("miss_count", 0))

    def current_candidate_distinct_view_count(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0
        return int(len(candidate.get("view_angles", [])))

    def current_candidate_hit_ratio(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0.0
        hits = int(candidate.get("hit_count", 0))
        misses = int(candidate.get("miss_count", 0))
        return float(hits / max(1, hits + misses))

    def current_candidate_direct_goal_contribution_count(self):
        candidate = getattr(self, "current_candidate", None)
        if not candidate:
            return 0
        top_contributions = candidate.get("top_contributions", [])
        return int(
            sum(1 for item in top_contributions if item.get("is_direct_match"))
        )

    def current_candidate_distance_to_gps(self, goal_gps):
        candidate = getattr(self, "current_candidate", None)
        if candidate is None or goal_gps is None:
            return float("inf")
        candidate_gps = candidate.get("gps")
        if candidate_gps is None:
            return float("inf")
        try:
            candidate_gps = np.asarray(candidate_gps, dtype=np.float32).reshape(-1)[:2]
            goal_gps = np.asarray(goal_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return float("inf")
        if candidate_gps.size != 2 or goal_gps.size != 2:
            return float("inf")
        if not (np.all(np.isfinite(candidate_gps)) and np.all(np.isfinite(goal_gps))):
            return float("inf")
        return float(np.linalg.norm(candidate_gps - goal_gps))

    def current_candidate_detector_support_for(self, goal_gps):
        candidate = getattr(self, "current_candidate", None)
        distance_m = self.current_candidate_distance_to_gps(goal_gps)
        binding_radius_m = float(self.reperception_same_goal_radius_m)
        matches = bool(np.isfinite(distance_m) and distance_m <= binding_radius_m)
        hit_count = self.current_candidate_hit_count()
        consecutive_hit_count = (
            int(candidate.get("consecutive_hit_count", 0)) if candidate else 0
        )
        max_consecutive_hit_count = self.current_candidate_max_consecutive_hit_count()
        near_hit_count = self.current_candidate_near_hit_count()
        miss_count = self.current_candidate_miss_count()
        distinct_view_count = self.current_candidate_distinct_view_count()
        hit_ratio = self.current_candidate_hit_ratio()
        supported = bool(
            matches
            and hit_count >= self.candidate_min_detector_hits
        )
        return {
            "supported": supported,
            "matches": matches,
            "distance_m": distance_m,
            "binding_radius_m": binding_radius_m,
            "candidate_id": candidate.get("candidate_id", "") if candidate else "",
            "hit_count": int(hit_count),
            "consecutive_hit_count": int(consecutive_hit_count),
            "max_consecutive_hit_count": int(max_consecutive_hit_count),
            "strong_consecutive_evidence": bool(
                hit_count >= self.candidate_strong_evidence_min_hits
            ),
            "strong_historical_evidence": bool(
                hit_count >= self.candidate_strong_evidence_min_hits
            ),
            "strong_evidence_min_hits": int(
                self.candidate_strong_evidence_min_hits
            ),
            "strong_consecutive_evidence_min_hits": int(
                self.candidate_strong_evidence_min_consecutive_hits
            ),
            "near_hit_count": int(near_hit_count),
            "miss_count": int(miss_count),
            "distinct_view_count": int(distinct_view_count),
            "hit_ratio": float(hit_ratio),
        }

    def current_candidate_needs_distinct_view(self):
        if not getattr(self, "reperception_active", False):
            return False
        if getattr(self, "reperception_goal_gps", None) is None:
            return False
        if self.current_candidate_distinct_view_count() >= self.candidate_min_distinct_views:
            return False
        hit_count = self.current_candidate_hit_count()
        if hit_count >= self.candidate_min_detector_hits:
            return True
        if not self.candidate_single_hit_search_enabled or hit_count < 1:
            return False
        candidate = getattr(self, "current_candidate", None)
        score_graph = float(candidate.get("score_graph", 0.0)) if candidate else 0.0
        direct_goal_count = self.current_candidate_direct_goal_contribution_count()
        node_support = self.get_goal_node_support(self.reperception_goal_gps)
        return bool(
            direct_goal_count > 0
            and (
                node_support.get("supported", False)
                or score_graph >= self.reperception_threshold * 0.75
            )
        )

    def current_reperception_step_budget(self):
        budget = int(self.reperception_max_steps)
        if self.current_candidate_needs_distinct_view():
            budget += int(self.candidate_view_scan_max_steps)
        return max(1, budget)

    def candidate_approach_status(self):
        if getattr(self, "stop_verification_active", False):
            return {"in_progress": False, "reason": "stop_verification_active"}
        if not (
            getattr(self, "found_possible_goal", False)
            or getattr(self, "reperception_active", False)
        ):
            return {"in_progress": False, "reason": "no_active_candidate"}

        target_gps = getattr(self, "reperception_goal_gps", None)
        if target_gps is None:
            target_gps = getattr(self, "possible_goal_temp_gps", None)
        candidate = getattr(self, "current_candidate", None)
        if target_gps is None and candidate is not None:
            target_gps = candidate.get("gps")
        if target_gps is None:
            return {"in_progress": False, "reason": "candidate_target_missing"}

        try:
            target_gps = np.asarray(target_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return {"in_progress": False, "reason": "candidate_target_invalid"}
        if target_gps.size != 2 or not np.all(np.isfinite(target_gps)):
            return {"in_progress": False, "reason": "candidate_target_invalid"}

        distance_m = self.distance_to_gps(target_gps)
        stop_distance_m = self.candidate_stop_distance_m()
        return {
            "in_progress": bool(distance_m > stop_distance_m),
            "distance_m": float(distance_m),
            "stop_distance_m": float(stop_distance_m),
            "target_gps": target_gps.tolist(),
            "reason": "candidate_approach_in_progress"
            if distance_m > stop_distance_m
            else "candidate_approach_arrived",
        }

    def active_candidate_target_gps(self):
        target_gps = getattr(self, "reperception_goal_gps", None)
        if target_gps is None and getattr(self, "found_possible_goal", False):
            target_gps = getattr(self, "possible_goal_temp_gps", None)
        candidate = getattr(self, "current_candidate", None)
        if target_gps is None and candidate is not None:
            target_gps = candidate.get("gps")
        if target_gps is None:
            return None
        try:
            target_gps = np.asarray(target_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return None
        if target_gps.size != 2 or not np.all(np.isfinite(target_gps)):
            return None
        return target_gps

    def active_candidate_progress_id(self, target_gps):
        candidate = getattr(self, "current_candidate", None)
        if candidate is not None and candidate.get("candidate_id"):
            return str(candidate.get("candidate_id"))
        if target_gps is None:
            return "none"
        rc = self.goal_gps_to_map_rc(target_gps).astype(np.int64)
        return f"possible_goal_{int(rc[0])}_{int(rc[1])}"

    def reset_candidate_progress_tracking(self):
        self.candidate_progress_candidate_id = None
        self.candidate_progress_best_distance_m = float("inf")
        self.candidate_progress_last_distance_m = float("inf")
        self.candidate_progress_no_improve_steps = 0

    def protect_near_hit_candidate_from_progress_reject(
        self, reason, target_gps, distance_m
    ):
        near_visual_hits = self.active_near_visual_hit_count()
        if near_visual_hits <= 0 or target_gps is None:
            return False
        try:
            target_gps = np.asarray(target_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return False
        if target_gps.size != 2 or not np.all(np.isfinite(target_gps)):
            return False

        already_verifying_target = False
        if getattr(self, "stop_verification_active", False):
            verification_target = getattr(self, "stop_verification_target_gps", None)
            if verification_target is not None:
                try:
                    verification_target = np.asarray(
                        verification_target, dtype=np.float32
                    ).reshape(-1)[:2]
                    already_verifying_target = bool(
                        verification_target.size == 2
                        and np.all(np.isfinite(verification_target))
                        and np.linalg.norm(verification_target - target_gps)
                        <= self.stop_verification_same_goal_radius_m
                    )
                except Exception:
                    already_verifying_target = False

        self.goal_gps = target_gps.copy()
        self.found_goal = True
        self.found_possible_goal = False
        self.reperception_active = False
        self.reperception_goal_gps = None
        self.reperception_goal_map_xy = None
        self.reperception_source = ""
        self.reperception_steps = 0
        self.reperception_observation_count = 0
        if not already_verifying_target:
            self.stop_verification_active = True
            self.stop_verification_target_gps = target_gps.copy()
            self.stop_verification_anchor_gps = target_gps.copy()
            self.stop_verification_steps_taken = 0
            self.stop_verification_hits = 0
            self.stop_verification_near_hits = 0
            self.stop_verification_observation_count = 0
            self.stop_verification_consecutive_failures = 0
            self.direct_goal_approach_steps = 0
            self.planned_goal_approach_steps = 0
            self.planned_goal_approach_blocked_steps = 0
            self.planned_goal_arrival_scan_steps_taken = 0
            self.planned_goal_retreat_active = False
            self.planned_goal_retreat_steps = 0
            self.planned_goal_retreat_blocked_steps = 0
            self.planned_goal_retreat_scan_steps_taken = 0
            self.planned_goal_failed_viewpoints = []
            self.last_planned_goal_approach_station = None
            self.last_planned_goal_retreat_station = None

        best_distance_m = float(self.candidate_progress_best_distance_m)
        last_distance_m = float(self.candidate_progress_last_distance_m)
        no_improve_steps = int(self.candidate_progress_no_improve_steps)
        stuck_steps = int(self.not_move_steps)
        self.reset_candidate_progress_tracking()
        self.not_move_steps = 0
        self.stop_reason = "near_hit_candidate_progress_protected"
        self.scenegraph.debug_stats.inc("near_hit_candidate_progress_protected")
        self.scenegraph.debug_stats.inc(f"{reason}_protected")
        self.log_candidate_event(
            "candidate_progress_protected",
            candidate=getattr(self, "current_candidate", None),
            candidate_gps=target_gps,
            decision="verify",
            reason=reason,
            agent_distance_m=float(distance_m)
            if distance_m is not None and np.isfinite(distance_m)
            else None,
            near_visual_hits=int(near_visual_hits),
            already_verifying_target=bool(already_verifying_target),
            candidate_progress_best_distance_m=best_distance_m,
            candidate_progress_last_distance_m=last_distance_m,
            candidate_progress_no_improve_steps=no_improve_steps,
            candidate_stuck_steps=stuck_steps,
        )
        return True

    def reject_active_candidate_for_progress(self, reason, target_gps, distance_m):
        if target_gps is None:
            return False
        if reason in {"blacklist_candidate_stuck", "blacklist_candidate_no_progress"}:
            if self.protect_near_hit_candidate_from_progress_reject(
                reason, target_gps, distance_m
            ):
                return True
        best_distance_m = float(self.candidate_progress_best_distance_m)
        last_distance_m = float(self.candidate_progress_last_distance_m)
        no_improve_steps = int(self.candidate_progress_no_improve_steps)
        stuck_steps = int(self.not_move_steps)
        self.finalize_candidate("reject", reason)
        self.add_rejected_goal_candidate(target_gps, reason)
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
        self.current_candidate = None
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.first_fbe = True
        self.reset_stop_verification_state(clear_history=False)
        self.reset_candidate_progress_tracking()
        self.not_use_random_goal()
        self.not_move_steps = 0
        self.stop_reason = reason
        self.scenegraph.debug_stats.inc(reason)
        self.log_candidate_event(
            "candidate_progress_reject",
            candidate_gps=target_gps,
            decision="blacklist",
            reason=reason,
            agent_distance_m=float(distance_m),
            candidate_progress_best_distance_m=best_distance_m,
            candidate_progress_last_distance_m=last_distance_m,
            candidate_progress_no_improve_steps=no_improve_steps,
            candidate_stuck_steps=stuck_steps,
        )
        return True

    def maybe_reject_active_candidate_for_stuck_or_no_progress(self, observations):
        if not (
            getattr(self, "found_possible_goal", False)
            or getattr(self, "reperception_active", False)
        ):
            self.reset_candidate_progress_tracking()
            return False
        target_gps = self.active_candidate_target_gps()
        if target_gps is None:
            return False
        distance_m = self.distance_to_gps(target_gps, observations)
        if not np.isfinite(distance_m):
            return False

        stop_distance_m = self.candidate_stop_distance_m()
        if distance_m <= stop_distance_m:
            self.candidate_progress_best_distance_m = min(
                float(self.candidate_progress_best_distance_m),
                float(distance_m),
            )
            self.candidate_progress_last_distance_m = float(distance_m)
            self.candidate_progress_no_improve_steps = 0
            return False

        progress_id = self.active_candidate_progress_id(target_gps)
        if progress_id != getattr(self, "candidate_progress_candidate_id", None):
            self.candidate_progress_candidate_id = progress_id
            self.candidate_progress_best_distance_m = float(distance_m)
            self.candidate_progress_last_distance_m = float(distance_m)
            self.candidate_progress_no_improve_steps = 0
            return False

        improved = (
            float(distance_m)
            < float(self.candidate_progress_best_distance_m)
            - self.candidate_progress_min_delta_m
        )
        if improved:
            self.candidate_progress_best_distance_m = float(distance_m)
            self.candidate_progress_no_improve_steps = 0
        else:
            self.candidate_progress_no_improve_steps += 1
        self.candidate_progress_last_distance_m = float(distance_m)

        if self.not_move_steps >= self.candidate_stuck_blacklist_steps:
            return self.reject_active_candidate_for_progress(
                "blacklist_candidate_stuck",
                target_gps,
                distance_m,
            )
        if (
            self.candidate_progress_no_improve_steps
            >= self.candidate_no_progress_blacklist_steps
        ):
            return self.reject_active_candidate_for_progress(
                "blacklist_candidate_no_progress",
                target_gps,
                distance_m,
            )
        return False

    def get_candidate_view_scan_action(self, traversible=None, cur_start=None, cur_start_o=None):
        candidate = getattr(self, "current_candidate", None)
        if candidate is None or not self.current_candidate_needs_distinct_view():
            return None
        if int(candidate.get("view_scan_steps", 0)) >= self.candidate_view_scan_max_steps:
            return None
        candidate["view_scan_steps"] = int(candidate.get("view_scan_steps", 0)) + 1
        action = None
        reason = "need_distinct_view"
        station = None
        if (
            self.candidate_multiview_enabled
            and traversible is not None
            and cur_start is not None
            and cur_start_o is not None
            and getattr(self, "reperception_goal_gps", None) is not None
        ):
            target_gps = np.asarray(self.reperception_goal_gps, dtype=np.float32)
            desired_radius_m = (
                self.candidate_multiview_min_radius_m
                + self.candidate_multiview_max_radius_m
            ) / 2.0
            goal_map = self.select_planned_goal_approach_station(
                traversible,
                cur_start,
                target_gps,
                min_radius_m=self.candidate_multiview_min_radius_m,
                max_radius_m=self.candidate_multiview_max_radius_m,
                desired_radius_m=desired_radius_m,
                station_kind="approach",
                min_start_distance_m=self.candidate_multiview_min_start_distance_m,
            )
            station = self.last_planned_goal_approach_station
            if goal_map is not None:
                planned_action = self.get_planned_goal_approach_action(
                    traversible,
                    cur_start,
                    cur_start_o,
                    target_gps,
                    approach_goal_map=goal_map,
                    station_stop_distance_m=self.candidate_multiview_station_stop_distance_m,
                    min_radius_m=self.candidate_multiview_min_radius_m,
                    max_radius_m=self.candidate_multiview_max_radius_m,
                    desired_radius_m=desired_radius_m,
                    station_kind="approach",
                )
                if planned_action is not None and planned_action != 0:
                    action = int(planned_action)
                    reason = "candidate_multiview_move"
                elif planned_action == 0:
                    reason = "candidate_multiview_arrived_turn"
                else:
                    reason = "candidate_multiview_plan_failed_turn"
            else:
                reason = "candidate_multiview_station_missing_turn"
        if action is None:
            action = self.stop_verification_turn_action
        self.scenegraph.debug_stats.inc("candidate_view_scan")
        if reason.startswith("candidate_multiview"):
            self.scenegraph.debug_stats.inc(reason)
        self.log_candidate_event(
            "view_scan",
            candidate=candidate,
            source=getattr(self, "reperception_source", ""),
            decision="scan",
            reason=reason,
            candidate_multiview_station=station,
        )
        return action

    def stop_verification_hit_support_for(self, goal_gps):
        target_gps = getattr(self, "stop_verification_target_gps", None)
        hits = int(getattr(self, "stop_verification_hits", 0))
        if target_gps is None or goal_gps is None:
            return {
                "supported": False,
                "matches": False,
                "distance_m": float("inf"),
                "binding_radius_m": float(self.stop_verification_same_goal_radius_m),
                "hits": hits,
            }
        try:
            target_gps = np.asarray(target_gps, dtype=np.float32).reshape(-1)[:2]
            goal_gps = np.asarray(goal_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return {
                "supported": False,
                "matches": False,
                "distance_m": float("inf"),
                "binding_radius_m": float(self.stop_verification_same_goal_radius_m),
                "hits": hits,
            }
        distance_m = float("inf")
        if target_gps.size == 2 and goal_gps.size == 2:
            if np.all(np.isfinite(target_gps)) and np.all(np.isfinite(goal_gps)):
                distance_m = float(np.linalg.norm(target_gps - goal_gps))
        binding_radius_m = float(self.stop_verification_same_goal_radius_m)
        matches = bool(np.isfinite(distance_m) and distance_m <= binding_radius_m)
        return {
            "supported": bool(
                matches and hits >= self.stop_verification_min_hits
            ),
            "matches": matches,
            "distance_m": distance_m,
            "binding_radius_m": binding_radius_m,
            "hits": hits,
        }

    def update_distinct_view_count(self, candidate, observations=None):
        if candidate is None:
            return
        angle = None
        if observations is None:
            observations = getattr(self, "current_observations", None)
        if observations is not None and "compass" in observations:
            angle = float(np.asarray(observations["compass"]).reshape(-1)[0])
        else:
            pose = (
                self.full_pose.detach().cpu().numpy()
                if torch.is_tensor(self.full_pose)
                else np.asarray(self.full_pose)
            )
            if pose.size >= 3:
                angle = float(np.deg2rad(pose[2]))
        if angle is None or not np.isfinite(angle):
            return
        min_sep = np.deg2rad(25.0)
        views = candidate.setdefault("view_angles", [])
        if not any(abs(np.arctan2(np.sin(angle - a), np.cos(angle - a))) < min_sep for a in views):
            views.append(angle)

    def candidate_visibility_for_miss(
        self,
        observations=None,
        candidate_gps=None,
        max_distance_m=None,
        fov_margin_deg=10.0,
    ):
        if observations is None:
            observations = getattr(self, "current_observations", None)
        if candidate_gps is None:
            candidate = getattr(self, "current_candidate", None)
            candidate_gps = getattr(self, "reperception_goal_gps", None)
            if candidate_gps is None and candidate is not None:
                candidate_gps = candidate.get("gps")
        visibility = {
            "expected_visible": True,
            "reason": "visibility_unknown",
            "bearing_deg": None,
            "distance_m": None,
            "half_fov_with_margin_deg": None,
            "projected_x": None,
            "max_distance_m": max_distance_m,
        }
        if observations is None or candidate_gps is None:
            return visibility
        if "gps" not in observations or "compass" not in observations:
            return visibility
        try:
            agent_gps = np.asarray(observations["gps"], dtype=np.float32).reshape(-1)[:2]
            candidate_gps = np.asarray(candidate_gps, dtype=np.float32).reshape(-1)[:2]
            agent_compass = float(np.asarray(observations["compass"]).reshape(-1)[0])
        except Exception:
            return visibility
        if (
            agent_gps.size != 2
            or candidate_gps.size != 2
            or not np.all(np.isfinite(agent_gps))
            or not np.all(np.isfinite(candidate_gps))
            or not np.isfinite(agent_compass)
        ):
            return visibility

        direction_vector = candidate_gps - agent_gps
        distance_m = float(np.linalg.norm(direction_vector))
        if distance_m < 1e-5:
            visibility.update({
                "expected_visible": False,
                "reason": "candidate_too_close_for_detector",
                "bearing_deg": 0.0,
                "distance_m": distance_m,
            })
            return visibility

        goal_direction = float(
            np.arctan2(-direction_vector[1], direction_vector[0])
        )
        bearing_rad = float(
            np.arctan2(
                np.sin(agent_compass - goal_direction),
                np.cos(agent_compass - goal_direction),
            )
        )
        try:
            hfov_deg = float(self.config.SIMULATOR.RGB_SENSOR.HFOV)
            width = float(self.config.SIMULATOR.RGB_SENSOR.WIDTH)
        except Exception:
            hfov_deg = 79.0
            width = 640.0
        margin_deg = float(fov_margin_deg)
        half_fov_with_margin_deg = hfov_deg / 2.0 + margin_deg
        projected_x = width / 2.0 + np.degrees(bearing_rad) * width / hfov_deg
        visibility.update({
            "bearing_deg": float(np.degrees(bearing_rad)),
            "distance_m": distance_m,
            "half_fov_with_margin_deg": float(half_fov_with_margin_deg),
            "projected_x": float(projected_x),
        })
        if max_distance_m is not None and distance_m > float(max_distance_m):
            visibility.update({
                "expected_visible": False,
                "reason": "candidate_beyond_visible_blacklist_range",
            })
            return visibility
        if distance_m > float(self.distance_threshold):
            visibility.update({
                "expected_visible": False,
                "reason": "candidate_beyond_detector_range",
            })
            return visibility
        if abs(np.degrees(bearing_rad)) > half_fov_with_margin_deg:
            visibility.update({
                "expected_visible": False,
                "reason": "candidate_out_of_view",
            })
            return visibility

        visibility.update({
            "expected_visible": True,
            "reason": "candidate_expected_visible",
        })
        return visibility

    def update_stop_blacklist_visibility_evidence(self, observations, target_gps, hit):
        if hit or target_gps is None:
            return None
        visibility = self.candidate_visibility_for_miss(
            observations,
            candidate_gps=target_gps,
            max_distance_m=self.rejected_goal_visible_blacklist_max_distance_m,
            fov_margin_deg=0.0,
        )
        self.stop_verification_blacklist_visibility = visibility
        if visibility.get("expected_visible", False):
            self.stop_verification_blacklist_visible_seen = True
            self.scenegraph.debug_stats.inc("blacklist_visible_candidate_seen")
        return visibility

    def get_spatial_turn_action_towards_gps(
        self,
        observations,
        target_gps,
        *,
        threshold_deg=None,
    ):
        if observations is None or target_gps is None:
            return None, {}
        if "gps" not in observations or "compass" not in observations:
            return None, {}
        try:
            agent_gps = np.asarray(observations["gps"], dtype=np.float32).reshape(-1)[:2]
            target_gps = np.asarray(target_gps, dtype=np.float32).reshape(-1)[:2]
            agent_compass = float(np.asarray(observations["compass"]).reshape(-1)[0])
        except Exception:
            return None, {}
        if (
            agent_gps.size != 2
            or target_gps.size != 2
            or not np.all(np.isfinite(agent_gps))
            or not np.all(np.isfinite(target_gps))
            or not np.isfinite(agent_compass)
        ):
            return None, {}

        direction_vector = target_gps - agent_gps
        distance_m = float(np.linalg.norm(direction_vector))
        if distance_m < 1e-5:
            return None, {
                "distance_m": distance_m,
                "bearing_deg": 0.0,
                "reason": "target_too_close",
            }
        goal_direction = float(
            np.arctan2(-direction_vector[1], direction_vector[0])
        )
        bearing_rad = float(
            np.arctan2(
                np.sin(agent_compass - goal_direction),
                np.cos(agent_compass - goal_direction),
            )
        )
        bearing_deg = float(np.degrees(bearing_rad))
        if threshold_deg is None:
            threshold_deg = self.direct_goal_approach_turn_threshold_deg
        threshold_deg = float(threshold_deg)
        info = {
            "distance_m": distance_m,
            "bearing_deg": bearing_deg,
            "threshold_deg": threshold_deg,
            "reason": "spatial_target_turn",
        }
        if bearing_deg > threshold_deg:
            return 3, info
        if bearing_deg < -threshold_deg:
            return 2, info
        info["reason"] = "spatial_target_aligned"
        return None, info

    def register_candidate_hit(
        self,
        *,
        goal_gps,
        source,
        confidence,
        score_k,
        score_graph,
        top_contributions,
        det_distance_m=None,
        detected_label=None,
        same_candidate_distance_m=0.0,
    ):
        candidate = self.current_candidate
        if candidate is None:
            candidate = self.start_candidate_state(goal_gps, source)
        candidate["gps"] = np.asarray(goal_gps, dtype=np.float32).copy()
        candidate["source"] = source
        candidate["hit_count"] = int(candidate.get("hit_count", 0)) + 1
        candidate["consecutive_hit_count"] = (
            int(candidate.get("consecutive_hit_count", 0)) + 1
        )
        candidate["max_consecutive_hit_count"] = max(
            int(candidate.get("max_consecutive_hit_count", 0)),
            int(candidate.get("consecutive_hit_count", 0)),
        )
        near_hit = False
        try:
            near_hit = (
                det_distance_m is not None
                and np.isfinite(float(det_distance_m))
                and float(det_distance_m)
                <= self.stop_verification_required_hit_max_distance_m
            )
        except Exception:
            near_hit = False
        if near_hit:
            candidate["near_hit_count"] = int(candidate.get("near_hit_count", 0)) + 1
            self.scenegraph.debug_stats.inc("candidate_near_hit")
        candidate["last_seen_step"] = int(self.total_steps)
        candidate["score_sum"] = float(self.reperception_score_sum)
        candidate["score_graph"] = float(score_graph)
        candidate["score_k"] = float(score_k)
        candidate["top_contributions"] = top_contributions
        self.update_distinct_view_count(candidate)
        self.log_candidate_event(
            "hit",
            candidate=candidate,
            candidate_gps=goal_gps,
            source=source,
            detected_label=detected_label or self.obj_goal,
            det_confidence=confidence,
            det_distance_m=det_distance_m,
            near_hit=bool(near_hit),
            consecutive_hit_count=int(candidate.get("consecutive_hit_count", 0)),
            max_consecutive_hit_count=int(
                candidate.get("max_consecutive_hit_count", 0)
            ),
            strong_consecutive_evidence=bool(
                int(candidate.get("hit_count", 0))
                >= self.candidate_strong_evidence_min_hits
            ),
            strong_historical_evidence=bool(
                int(candidate.get("hit_count", 0))
                >= self.candidate_strong_evidence_min_hits
            ),
            near_hit_count=int(candidate.get("near_hit_count", 0)),
            near_hit_max_distance_m=float(
                self.stop_verification_required_hit_max_distance_m
            ),
            score_k=score_k,
            score_graph=score_graph,
            top_contributions=top_contributions,
            same_candidate_distance_m=same_candidate_distance_m,
            decision=candidate.get("decision", "pending"),
            reason="detector_observation",
        )

    def register_candidate_miss(self, source, reason, visibility=None):
        candidate = getattr(self, "current_candidate", None)
        if candidate is None:
            return
        candidate["miss_count"] = int(candidate.get("miss_count", 0)) + 1
        candidate["consecutive_hit_count"] = 0
        candidate["score_sum"] = float(self.reperception_score_sum)
        self.log_candidate_event(
            "miss",
            candidate=candidate,
            source=source,
            decision="pending",
            reason=reason,
            candidate_visibility=visibility,
        )

    def finalize_candidate(self, decision, reason):
        candidate = getattr(self, "current_candidate", None)
        if candidate is None or candidate.get("finalized", False):
            return
        candidate["decision"] = decision
        candidate["reason"] = reason
        candidate["finalized"] = True
        summary = self.candidate_snapshot(candidate)
        self.candidate_summaries.append(summary)
        counter_key = {
            "confirm": "candidate_confirmed",
            "reject": "candidate_rejected",
            "blacklist": "candidate_blacklisted",
        }.get(decision, "candidate_finalized")
        self.scenegraph.debug_stats.inc(counter_key)
        if decision == "confirm" and summary["hit_count"] < self.candidate_min_detector_hits:
            self.scenegraph.debug_stats.inc("graph_only_confirmations")
            self.scenegraph.debug_stats.inc("confirmations_without_detector_hits")
        self.log_candidate_event(
            decision,
            candidate=candidate,
            decision=decision,
            reason=reason,
        )

    def candidate_snapshot(self, candidate=None):
        candidate = candidate or getattr(self, "current_candidate", None)
        if candidate is None:
            return {}
        gps = np.asarray(candidate.get("gps", [np.nan, np.nan]), dtype=np.float32)
        top_contributions = candidate.get("top_contributions", [])
        direct_count = sum(1 for item in top_contributions if item.get("is_direct_match"))
        non_goal_count = sum(1 for item in top_contributions if not item.get("is_direct_match"))
        hits = int(candidate.get("hit_count", 0))
        consecutive_hits = int(candidate.get("consecutive_hit_count", 0))
        max_consecutive_hits = int(candidate.get("max_consecutive_hit_count", 0))
        near_hits = int(candidate.get("near_hit_count", 0))
        misses = int(candidate.get("miss_count", 0))
        return {
            "candidate_id": candidate.get("candidate_id", ""),
            "candidate_gps": gps.tolist(),
            "candidate_rc": self.goal_gps_to_map_rc(gps).tolist()
            if np.all(np.isfinite(gps))
            else None,
            "source": candidate.get("source", ""),
            "started_step": int(candidate.get("started_step", -1)),
            "last_seen_step": int(candidate.get("last_seen_step", -1)),
            "hit_count": hits,
            "consecutive_hit_count": consecutive_hits,
            "max_consecutive_hit_count": max_consecutive_hits,
            "strong_consecutive_evidence": bool(
                hits >= self.candidate_strong_evidence_min_hits
            ),
            "strong_historical_evidence": bool(
                hits >= self.candidate_strong_evidence_min_hits
            ),
            "strong_evidence_min_hits": int(self.candidate_strong_evidence_min_hits),
            "near_hit_count": near_hits,
            "miss_count": misses,
            "distinct_view_count": len(candidate.get("view_angles", [])),
            "hit_ratio": float(hits / max(1, hits + misses)),
            "score_sum": float(candidate.get("score_sum", 0.0)),
            "score_graph": float(candidate.get("score_graph", 0.0)),
            "score_k": float(candidate.get("score_k", 0.0)),
            "top_contributions": top_contributions,
            "view_scan_steps": int(candidate.get("view_scan_steps", 0)),
            "top_contribution_direct_goal_count": int(direct_count),
            "top_contribution_non_goal_count": int(non_goal_count),
            "decision": candidate.get("decision", "pending"),
            "reason": candidate.get("reason", ""),
            "graph_only_confirmation": bool(
                candidate.get("decision") == "confirm"
                and hits < self.candidate_min_detector_hits
            ),
        }

    def log_candidate_event(self, event, **kwargs):
        candidate = kwargs.pop("candidate", None) or getattr(self, "current_candidate", None)
        candidate_gps = kwargs.pop("candidate_gps", None)
        if candidate_gps is None and candidate is not None:
            candidate_gps = candidate.get("gps")
        if candidate_gps is not None:
            candidate_gps = np.asarray(candidate_gps, dtype=np.float32)
            candidate_rc = self.goal_gps_to_map_rc(candidate_gps)
        else:
            candidate_rc = None
        hits = int(candidate.get("hit_count", 0)) if candidate else 0
        consecutive_hits = int(candidate.get("consecutive_hit_count", 0)) if candidate else 0
        max_consecutive_hits = int(candidate.get("max_consecutive_hit_count", 0)) if candidate else 0
        near_hits = int(candidate.get("near_hit_count", 0)) if candidate else 0
        misses = int(candidate.get("miss_count", 0)) if candidate else 0
        row = {
            "episode_idx": int(getattr(self, "count_episodes", -1)),
            "step": int(getattr(self, "total_steps", -1)),
            "goal": getattr(self, "obj_goal", ""),
            "event": event,
            "candidate_id": candidate.get("candidate_id", "") if candidate else "",
            "candidate_gps": candidate_gps.tolist() if candidate_gps is not None else None,
            "candidate_rc": candidate_rc.tolist() if candidate_rc is not None else None,
            "source": kwargs.pop("source", candidate.get("source", "") if candidate else ""),
            "detected_label": kwargs.pop("detected_label", getattr(self, "obj_goal", "")),
            "det_confidence": kwargs.pop("det_confidence", None),
            "det_distance_m": kwargs.pop("det_distance_m", None),
            "same_candidate_distance_m": kwargs.pop("same_candidate_distance_m", None),
            "hit_count": hits,
            "consecutive_hit_count": kwargs.pop(
                "consecutive_hit_count", consecutive_hits
            ),
            "max_consecutive_hit_count": kwargs.pop(
                "max_consecutive_hit_count", max_consecutive_hits
            ),
            "strong_consecutive_evidence": kwargs.pop(
                "strong_consecutive_evidence",
                bool(hits >= self.candidate_strong_evidence_min_hits),
            ),
            "strong_historical_evidence": kwargs.pop(
                "strong_historical_evidence",
                bool(hits >= self.candidate_strong_evidence_min_hits),
            ),
            "strong_evidence_min_hits": int(self.candidate_strong_evidence_min_hits),
            "strong_consecutive_evidence_min_hits": int(
                self.candidate_strong_evidence_min_consecutive_hits
            ),
            "near_hit_count": near_hits,
            "miss_count": misses,
            "distinct_view_count": len(candidate.get("view_angles", [])) if candidate else 0,
            "view_scan_steps": int(candidate.get("view_scan_steps", 0)) if candidate else 0,
            "hit_ratio": float(hits / max(1, hits + misses)),
            "score_graph": kwargs.pop("score_graph", candidate.get("score_graph", None) if candidate else None),
            "score_k": kwargs.pop("score_k", candidate.get("score_k", None) if candidate else None),
            "score_sum": kwargs.pop("score_sum", float(self.reperception_score_sum)),
            "score_norm": self.reperception_score_norm,
            "score_ready": bool(self.reperception_score_sum >= self.reperception_threshold),
            "node_support": kwargs.pop("node_support", None),
            "stop_liveness_decision": kwargs.pop("stop_liveness_decision", None),
            "top_contributions": kwargs.pop(
                "top_contributions",
                candidate.get("top_contributions", []) if candidate else [],
            ),
            "decision": kwargs.pop("decision", ""),
            "reason": kwargs.pop("reason", ""),
        }
        row.update(kwargs)
        self.candidate_trace_logger.log(row)

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

    def bind_depth_capped_detection_to_active_candidate(self, goal_gps, det_distance_m):
        if det_distance_m is None:
            return goal_gps, False
        try:
            depth_capped = float(det_distance_m) >= float(self.distance_threshold)
        except Exception:
            depth_capped = False
        if not depth_capped:
            return goal_gps, False
        if (
            getattr(self, "reperception_active", False)
            and getattr(self, "reperception_goal_gps", None) is not None
        ):
            return np.asarray(self.reperception_goal_gps, dtype=np.float32).copy(), True
        if (
            getattr(self, "found_possible_goal", False)
            and getattr(self, "possible_goal_temp_gps", None) is not None
            and getattr(self, "current_candidate", None) is not None
        ):
            return np.asarray(self.possible_goal_temp_gps, dtype=np.float32).copy(), True
        return goal_gps, False

    def set_possible_goal_from_visual_evidence(
        self,
        goal_gps,
        source,
        *,
        confidence=None,
        det_distance_m=None,
        require_existing_visual_evidence=False,
    ):
        if require_existing_visual_evidence:
            has_candidate_hit = self.current_candidate_hit_count() > 0
            has_verification_hit = int(getattr(self, "stop_verification_hits", 0)) > 0
            if not (has_candidate_hit or has_verification_hit):
                self.scenegraph.debug_stats.inc("possible_goal_without_visual_evidence_blocked")
                return False
        try:
            goal_gps = np.asarray(goal_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            self.scenegraph.debug_stats.inc("possible_goal_invalid_gps")
            return False
        if goal_gps.size != 2 or not np.all(np.isfinite(goal_gps)):
            self.scenegraph.debug_stats.inc("possible_goal_invalid_gps")
            return False
        self.found_goal = False
        self.found_possible_goal = True
        self.possible_goal_temp_gps = goal_gps.copy()
        self.found_goal_times = self.reperception_score_sum
        self.scenegraph.debug_stats.inc("possible_goal_from_visual_evidence")
        self.log_candidate_event(
            "possible_goal",
            candidate_gps=goal_gps,
            source=source,
            det_confidence=confidence,
            det_distance_m=det_distance_m,
            decision="possible",
            reason="visual_evidence",
        )
        return True

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
            center_node = subgraph.get("center_node")
            center_caption = str(getattr(center_node, "caption", "")).lower()
            is_direct_match = self.goal_label_matches(center_caption)
            p_sub_raw = float(np.clip(subgraph["score"], 0.0, 1.0))
            if is_direct_match:
                p_sub = min(1.0, p_sub_raw + self.candidate_direct_match_bonus)
            else:
                p_sub = min(p_sub_raw, self.candidate_context_cap)
            weight = 1.0 / dist_m
            term = p_sub * weight
            score_graph += term
            weight_sum += weight
            contributions.append({
                "center": center_xy.tolist(),
                "center_caption": center_caption,
                "is_direct_match": bool(is_direct_match),
                "room": subgraph.get("room", ""),
                "p_sub_raw": p_sub_raw,
                "p_sub_used": p_sub,
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
            ttl = int(item.get("ttl", self.rejected_goal_ttl))
            radius_m = float(item.get("radius_m", self.rejected_goal_radius_m))
            if self.total_steps - item["step"] <= ttl:
                kept.append(item)
                if np.linalg.norm(goal_gps - item["gps"]) <= radius_m:
                    rejected = True
        self.rejected_goal_candidates = kept
        return rejected

    def should_blacklist_rejected_goal(self, reason):
        reason = str(reason or "")
        soft_reasons = {
            "too_many_misses",
            "credibility_below_threshold",
            "candidate_not_reconfirmed",
            "stop_verification_failed",
            "stop_without_candidate_rejected",
            "stop_without_current_detector_support",
            "stop_without_verification_rejected",
            "planned_goal_retreat_failed",
            "planned_goal_retreat_aligned_no_detection",
            "planned_goal_retreat_timeout",
            "planned_goal_retreat_blocked",
            "planned_goal_verification_timeout",
            "planned_goal_approach_timeout",
            "planned_goal_approach_blocked",
            "direct_goal_approach_timeout",
        }
        if reason in soft_reasons:
            return False
        if (
            "timeout" in reason
            or "blocked" in reason
            or "miss" in reason
            or "not_reconfirmed" in reason
            or "verification_failed" in reason
        ):
            return False
        return reason.startswith("hard_false_positive") or reason.startswith(
            "blacklist"
        )

    def add_rejected_goal_candidate(self, goal_gps, reason):
        if goal_gps is None:
            return
        if not self.should_blacklist_rejected_goal(reason):
            self.scenegraph.debug_stats.inc("rejected_goal_blacklist_skipped")
            self.scenegraph.debug_stats.inc(
                "rejected_goal_blacklist_skipped_" + str(reason or "unknown")
            )
            return
        reason_text = str(reason)
        needs_visible_blacklist_evidence = (
            reason_text.startswith("hard_false_positive")
            and "station_missing" not in reason_text
        )
        if needs_visible_blacklist_evidence and not getattr(
            self, "stop_verification_blacklist_visible_seen", False
        ):
            self.scenegraph.debug_stats.inc("rejected_goal_blacklist_skipped")
            self.scenegraph.debug_stats.inc(
                "rejected_goal_blacklist_skipped_not_visible_within_2m"
            )
            self.log_candidate_event(
                "blacklist_skipped",
                candidate_gps=goal_gps,
                decision="skip",
                reason="not_visible_within_blacklist_range",
                blacklist_visibility=getattr(
                    self, "stop_verification_blacklist_visibility", None
                ),
                visible_blacklist_max_distance_m=float(
                    self.rejected_goal_visible_blacklist_max_distance_m
                ),
            )
            return
        self.rejected_goal_candidates.append({
            "gps": np.asarray(goal_gps, dtype=np.float32).copy(),
            "step": int(self.total_steps),
            "reason": reason,
            "ttl": int(self.rejected_goal_ttl),
            "radius_m": float(self.rejected_goal_radius_m),
        })
        self.scenegraph.debug_stats.inc("rejected_goal_blacklist_added")
        self.first_fbe = True
        self.not_use_random_goal()
        self.not_move_steps = 0
        self.scenegraph.debug_stats.inc("blacklist_force_frontier")

    def start_or_update_reperception_candidate(
        self,
        goal_gps,
        confidence,
        source,
        det_distance_m=None,
        detected_label=None,
    ):
        goal_gps = np.asarray(goal_gps, dtype=np.float32)
        confidence = float(confidence)
        if confidence < self.candidate_start_min_confidence:
            self.scenegraph.debug_stats.inc("candidate_start_low_confidence_skipped")
            self.scenegraph.debug_stats.inc(
                "candidate_start_low_confidence_skipped_" + str(source)
            )
            self.log_candidate_event(
                "low_confidence_skip",
                candidate_gps=goal_gps,
                source=source,
                detected_label=detected_label or self.obj_goal,
                det_confidence=confidence,
                det_distance_m=det_distance_m,
                decision="skip",
                reason="candidate_start_low_confidence",
                candidate_start_min_confidence=float(
                    self.candidate_start_min_confidence
                ),
            )
            return "low_confidence"
        if self.is_rejected_goal_candidate(goal_gps):
            self.scenegraph.debug_stats.inc("reperception_rejected_blacklist")
            self.scenegraph.debug_stats.inc("candidate_blacklisted")
            self.log_candidate_event(
                "blacklist",
                candidate_gps=goal_gps,
                source=source,
                detected_label=detected_label or self.obj_goal,
                det_confidence=confidence,
                det_distance_m=det_distance_m,
                decision="blacklist",
                reason="rejected_candidate_match",
            )
            return "rejected_blacklist"

        same_candidate_distance_m = float("inf")
        if self.reperception_goal_gps is not None:
            same_candidate_distance_m = float(
                np.linalg.norm(goal_gps - self.reperception_goal_gps)
            )
        same_candidate_same_step = (
            self.reperception_active
            and self.reperception_last_step == self.total_steps
            and self.reperception_goal_gps is not None
            and same_candidate_distance_m <= self.reperception_same_goal_radius_m
        )
        if (
            not self.reperception_active
            or self.reperception_goal_gps is None
            or same_candidate_distance_m > self.reperception_same_goal_radius_m
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
            self.start_candidate_state(goal_gps, source)

        self.reperception_goal_gps = goal_gps.copy()
        self.reperception_goal_map_xy = self.goal_gps_to_map_xy(goal_gps)
        self.reperception_source = source
        if same_candidate_same_step:
            self.set_possible_goal_from_visual_evidence(
                self.reperception_goal_gps,
                source,
                confidence=confidence,
                det_distance_m=det_distance_m,
            )
            self.scenegraph.debug_stats.inc("reperception_duplicate_observation")
            self.log_candidate_event(
                "hit",
                candidate_gps=goal_gps,
                source=source,
                detected_label=detected_label or self.obj_goal,
                det_confidence=confidence,
                det_distance_m=det_distance_m,
                same_candidate_distance_m=same_candidate_distance_m,
                decision="pending",
                reason="duplicate_same_step",
            )
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
        self.register_candidate_hit(
            goal_gps=goal_gps,
            source=source,
            confidence=confidence,
            score_k=score_k,
            score_graph=self.last_reperception_score_graph,
            top_contributions=top_contributions,
            det_distance_m=det_distance_m,
            detected_label=detected_label,
            same_candidate_distance_m=same_candidate_distance_m,
        )

        score_ready = self.reperception_score_sum >= self.reperception_threshold
        hit_ready = self.current_candidate_hit_count() >= self.candidate_min_detector_hits
        view_ready = True
        hit_ratio_ready = True
        direct_goal_ready = self.current_candidate_direct_goal_contribution_count() > 0
        direct_goal_confirm_ready = bool(
            direct_goal_ready or not self.candidate_require_direct_goal_for_confirm
        )
        evidence_ready = hit_ready
        history_item.update({
            "hit_count": int(self.current_candidate_hit_count()),
            "miss_count": int(self.current_candidate_miss_count()),
            "distinct_view_count": int(self.current_candidate_distinct_view_count()),
            "hit_ratio": float(self.current_candidate_hit_ratio()),
            "direct_goal_ready": bool(direct_goal_ready),
            "direct_goal_confirm_ready": bool(direct_goal_confirm_ready),
            "hit_ready": bool(hit_ready),
            "view_ready": bool(view_ready),
            "hit_ratio_ready": bool(hit_ratio_ready),
            "evidence_ready": bool(evidence_ready),
        })
        if evidence_ready:
            history_item["status"] = "approach_confirmed"
            self.confirm_candidate_for_approach()
            return "approach_confirmed"

        if self.current_candidate_miss_count() >= self.candidate_max_misses:
            history_item["status"] = "rejected"
            self.reject_reperception_goal(reason="too_many_misses")
            return "rejected"

        if self.reperception_steps >= self.current_reperception_step_budget():
            history_item["status"] = "rejected"
            self.reject_reperception_goal(reason="credibility_below_threshold")
            return "rejected"

        if score_ready and not evidence_ready:
            if self.current_candidate_needs_distinct_view():
                history_item["status"] = "pending_distinct_view"
                self.scenegraph.debug_stats.inc("reperception_wait_distinct_view")
            else:
                history_item["status"] = "pending_detector_evidence"
                self.scenegraph.debug_stats.inc("reperception_wait_detector_evidence")
        self.set_possible_goal_from_visual_evidence(
            self.reperception_goal_gps,
            source,
            confidence=confidence,
            det_distance_m=det_distance_m,
        )
        return "pending"

    def tick_reperception_without_observation(self, source):
        if (
            not self.reperception_active
            or self.reperception_goal_gps is None
            or self.reperception_last_step == self.total_steps
        ):
            return

        candidate = getattr(self, "current_candidate", None)
        approach_status = self.candidate_approach_status()
        if approach_status.get("in_progress", False):
            self.reperception_last_step = self.total_steps
            history_item = {
                "step": int(self.total_steps),
                "source": source,
                "confidence": 0.0,
                "score_k": 0.0,
                "score_sum": float(self.reperception_score_sum),
                "observation_count": int(self.reperception_observation_count),
                "hit_count": int(self.current_candidate_hit_count()),
                "miss_count": int(self.current_candidate_miss_count()),
                "distinct_view_count": int(self.current_candidate_distinct_view_count()),
                "hit_ratio": float(self.current_candidate_hit_ratio()),
                "num_subgraphs": 0,
                "top_contributions": [],
                "status": "approach_in_progress",
                "candidate_approach": approach_status,
            }
            self.reperception_history.append(history_item)
            self.scenegraph.debug_stats.inc("reperception_miss_skipped_approach")
            self.log_candidate_event(
                "miss_skipped",
                source=source,
                decision="pending",
                reason="candidate_approach_in_progress",
                candidate_approach=approach_status,
            )
            self.set_possible_goal_from_visual_evidence(
                approach_status["target_gps"],
                source,
                require_existing_visual_evidence=True,
            )
            return

        view_scan_steps = int(candidate.get("view_scan_steps", 0)) if candidate else 0
        waiting_for_distinct_view = bool(
            self.current_candidate_needs_distinct_view()
            and view_scan_steps < self.candidate_view_scan_max_steps
        )
        if waiting_for_distinct_view:
            self.reperception_last_step = self.total_steps
            history_item = {
                "step": int(self.total_steps),
                "source": source,
                "confidence": 0.0,
                "score_k": 0.0,
                "score_sum": float(self.reperception_score_sum),
                "observation_count": int(self.reperception_observation_count),
                "hit_count": int(self.current_candidate_hit_count()),
                "miss_count": int(self.current_candidate_miss_count()),
                "distinct_view_count": int(self.current_candidate_distinct_view_count()),
                "hit_ratio": float(self.current_candidate_hit_ratio()),
                "num_subgraphs": 0,
                "top_contributions": [],
                "status": "waiting_distinct_view",
                "candidate_view_scan_steps": int(view_scan_steps),
                "candidate_view_scan_max_steps": int(self.candidate_view_scan_max_steps),
            }
            self.reperception_history.append(history_item)
            self.scenegraph.debug_stats.inc("reperception_miss_skipped_distinct_view")
            self.log_candidate_event(
                "miss_skipped",
                source=source,
                decision="pending",
                reason="waiting_distinct_view",
                candidate_view_scan_steps=view_scan_steps,
                candidate_view_scan_max_steps=self.candidate_view_scan_max_steps,
            )
            self.set_possible_goal_from_visual_evidence(
                self.reperception_goal_gps,
                source,
                require_existing_visual_evidence=True,
            )
            return

        visibility = self.candidate_visibility_for_miss()
        if not visibility.get("expected_visible", True):
            self.reperception_last_step = self.total_steps
            history_item = {
                "step": int(self.total_steps),
                "source": source,
                "confidence": 0.0,
                "score_k": 0.0,
                "score_sum": float(self.reperception_score_sum),
                "observation_count": int(self.reperception_observation_count),
                "hit_count": int(self.current_candidate_hit_count()),
                "miss_count": int(self.current_candidate_miss_count()),
                "distinct_view_count": int(self.current_candidate_distinct_view_count()),
                "hit_ratio": float(self.current_candidate_hit_ratio()),
                "num_subgraphs": 0,
                "top_contributions": [],
                "status": "visibility_skipped",
                "candidate_visibility": visibility,
            }
            self.reperception_history.append(history_item)
            self.scenegraph.debug_stats.inc("reperception_miss_skipped_not_visible")
            self.scenegraph.debug_stats.inc(
                "reperception_miss_skipped_" + str(visibility.get("reason", "unknown"))
            )
            self.log_candidate_event(
                "miss_skipped",
                source=source,
                decision="pending",
                reason=visibility.get("reason", "candidate_not_expected_visible"),
                candidate_visibility=visibility,
            )
            self.set_possible_goal_from_visual_evidence(
                self.reperception_goal_gps,
                source,
                require_existing_visual_evidence=True,
            )
            return

        self.reperception_steps += 1
        self.reperception_last_step = self.total_steps
        self.reperception_score_sum *= self.candidate_score_decay
        self.reperception_score_sum = max(
            0.0,
            self.reperception_score_sum - self.candidate_miss_penalty,
        )
        self.register_candidate_miss(
            source,
            reason="candidate_not_observed",
            visibility=visibility,
        )
        history_item = {
            "step": int(self.total_steps),
            "source": source,
            "confidence": 0.0,
            "score_k": 0.0,
            "score_sum": float(self.reperception_score_sum),
            "observation_count": int(self.reperception_observation_count),
            "hit_count": int(self.current_candidate_hit_count()),
            "miss_count": int(self.current_candidate_miss_count()),
            "distinct_view_count": int(self.current_candidate_distinct_view_count()),
            "hit_ratio": float(self.current_candidate_hit_ratio()),
            "num_subgraphs": 0,
            "top_contributions": [],
            "status": "pending",
            "candidate_visibility": visibility,
        }
        self.reperception_history.append(history_item)
        self.scenegraph.debug_stats.inc("reperception_missed_observations")

        if (
            self.current_candidate_has_strong_historical_evidence()
            and not approach_status.get("in_progress", False)
        ):
            history_item["status"] = "strong_historical_evidence_verify"
            self.promote_candidate_to_stop_verification(
                reason="strong_historical_evidence_arrived",
                target_gps=self.reperception_goal_gps,
                distance_m=approach_status.get("distance_m"),
            )
            return

        if self.current_candidate_miss_count() >= self.candidate_max_misses:
            if not approach_status.get("in_progress", False):
                history_item["status"] = "viewpoint_search_after_misses"
                if self.promote_candidate_to_stop_verification(
                    reason="candidate_too_many_misses_viewpoint_search",
                    target_gps=self.reperception_goal_gps,
                    distance_m=approach_status.get("distance_m"),
                    event_name="candidate_misses_viewpoint_search",
                    debug_counter="candidate_misses_viewpoint_search",
                ):
                    return
            history_item["status"] = "rejected"
            self.reject_reperception_goal(reason="too_many_misses")
            return

        if self.reperception_steps >= self.current_reperception_step_budget():
            if not approach_status.get("in_progress", False):
                history_item["status"] = "viewpoint_search_after_step_budget"
                if self.promote_candidate_to_stop_verification(
                    reason="candidate_not_reconfirmed_viewpoint_search",
                    target_gps=self.reperception_goal_gps,
                    distance_m=approach_status.get("distance_m"),
                    event_name="candidate_step_budget_viewpoint_search",
                    debug_counter="candidate_step_budget_viewpoint_search",
                ):
                    return
            history_item["status"] = "rejected"
            self.reject_reperception_goal(reason="candidate_not_reconfirmed")
            return

        self.set_possible_goal_from_visual_evidence(
            self.reperception_goal_gps,
            source,
            require_existing_visual_evidence=True,
        )

    def confirm_candidate_for_approach(self):
        if self.reperception_goal_gps is None:
            return
        candidate = getattr(self, "current_candidate", None)
        already_confirmed = bool(
            candidate is not None
            and candidate.get("decision") == "approach_confirmed"
        )
        if candidate is not None:
            candidate["decision"] = "approach_confirmed"
            candidate["reason"] = "detector_evidence_ready"
            candidate["score_sum"] = float(self.reperception_score_sum)
            candidate["gps"] = np.asarray(
                self.reperception_goal_gps, dtype=np.float32
            ).copy()

        self.goal_gps = self.reperception_goal_gps.copy()
        self.set_possible_goal_from_visual_evidence(
            self.reperception_goal_gps,
            "candidate_approach_confirmed",
            require_existing_visual_evidence=True,
        )
        self.reperception_active = True
        if not already_confirmed:
            self.scenegraph.debug_stats.inc("candidate_approach_confirmed")
            self.log_candidate_event(
                "approach_confirmed",
                candidate=candidate,
                candidate_gps=self.reperception_goal_gps,
                source=getattr(self, "reperception_source", ""),
                decision="approach_confirmed",
                reason="detector_evidence_ready",
            )

    def promote_candidate_to_stop_verification(
        self,
        *,
        reason,
        target_gps=None,
        distance_m=None,
        event_name=None,
        debug_counter=None,
    ):
        candidate = getattr(self, "current_candidate", None)
        if target_gps is None and self.reperception_goal_gps is not None:
            target_gps = self.reperception_goal_gps
        if target_gps is None and candidate is not None:
            target_gps = candidate.get("gps")
        if target_gps is None:
            return False
        try:
            target_gps = np.asarray(target_gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return False
        if target_gps.size != 2 or not np.all(np.isfinite(target_gps)):
            return False

        self.goal_gps = target_gps.copy()
        self.found_goal = True
        self.found_possible_goal = False
        self.found_goal_times = float(self.reperception_score_sum)
        self.reperception_active = False
        self.reperception_goal_gps = target_gps.copy()
        self.reperception_goal_map_xy = self.goal_gps_to_map_xy(target_gps)
        self.stop_verification_active = True
        self.stop_verification_target_gps = target_gps.copy()
        self.stop_verification_anchor_gps = target_gps.copy()
        self.stop_verification_steps_taken = 0
        self.stop_verification_hits = 0
        self.stop_verification_near_hits = 0
        self.stop_verification_observation_count = 0
        self.stop_verification_consecutive_failures = 0
        self.direct_goal_approach_steps = 0
        self.planned_goal_approach_steps = 0
        self.planned_goal_approach_blocked_steps = 0
        self.planned_goal_arrival_scan_steps_taken = 0
        self.planned_goal_retreat_active = False
        self.planned_goal_retreat_steps = 0
        self.planned_goal_retreat_blocked_steps = 0
        self.planned_goal_retreat_scan_steps_taken = 0
        self.planned_goal_retreat_confirmed = False
        self.planned_goal_failed_viewpoints = []
        self.last_planned_goal_approach_station = None
        self.last_planned_goal_retreat_station = None
        self.stop_verification_blacklist_visible_seen = False
        self.stop_verification_blacklist_visibility = None
        self.stop_reason = reason
        if event_name is None:
            event_name = (
                "strong_historical_evidence_verify"
                if "strong_historical" in str(reason)
                else "candidate_arrived_viewpoint_search"
            )
        if debug_counter is None:
            debug_counter = event_name
        self.scenegraph.debug_stats.inc(debug_counter)
        self.scenegraph.debug_stats.inc("stop_verification_started")
        self.log_candidate_event(
            event_name,
            candidate=candidate,
            candidate_gps=target_gps,
            source=getattr(self, "reperception_source", ""),
            decision="verify",
            reason=reason,
            agent_distance_m=float(distance_m)
            if distance_m is not None and np.isfinite(distance_m)
            else None,
            max_consecutive_hit_count=int(
                candidate.get("max_consecutive_hit_count", 0)
            )
            if candidate
            else 0,
        )
        return True

    def confirm_reperception_goal(self):
        if self.reperception_goal_gps is None:
            return
        self.goal_gps = self.reperception_goal_gps.copy()
        self.found_goal = True
        self.found_possible_goal = False
        self.found_goal_times = self.reperception_score_sum
        self.reperception_active = False
        self.finalize_candidate("confirm", "score_and_detector_evidence_ready")
        self.scenegraph.debug_stats.inc("reperception_confirmed")
        if self.paper_reperception_mode:
            self.scenegraph.debug_stats.inc("paper_reperception_confirmed")

    def reject_reperception_goal(self, reason):
        self.finalize_candidate("reject", reason)
        self.add_rejected_goal_candidate(self.reperception_goal_gps, reason)
        self.found_goal = False
        self.found_possible_goal = False
        self.found_goal_times = 0
        self.goal_gps_map.fill(0)
        self.first_fbe = True
        self.reperception_active = False
        self.reperception_goal_gps = None
        self.reperception_goal_map_xy = None
        self.reperception_source = ""
        self.reperception_score_sum = 0.0
        self.reperception_steps = 0
        self.reperception_observation_count = 0
        self.current_candidate = None
        self.reset_candidate_progress_tracking()
        self.not_use_random_goal()
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
            if score < self.goal_detection_min_confidence:
                self.scenegraph.debug_stats.inc(
                    "stop_verification_low_confidence_skipped"
                )
                continue
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
        anchor_gps = getattr(self, "stop_verification_anchor_gps", None)
        anchor_delta = None
        anchor_drift = False
        if (
            best is not None
            and best.get("depth_valid", False)
            and anchor_gps is not None
        ):
            try:
                anchor_delta = float(
                    np.linalg.norm(
                        np.asarray(best["gps"], dtype=np.float32)
                        - np.asarray(anchor_gps, dtype=np.float32)
                    )
                )
                anchor_drift = (
                    anchor_delta > self.stop_verification_anchor_radius_m
                )
            except Exception:
                anchor_delta = None
                anchor_drift = False
        hit = (
            best is not None
            and best["depth_valid"]
            and best["delta"] <= self.stop_verification_same_goal_radius_m
            and best["distance"] <= self.stop_verification_max_detection_distance_m
            and node_support["supported"]
            and not anchor_drift
        )
        if anchor_drift:
            self.stop_verification_hits = 0
            self.stop_verification_near_hits = 0
            self.scenegraph.debug_stats.inc("stop_verification_anchor_drift_reset")
        near_hit = bool(
            hit
            and best is not None
            and best.get("depth_valid", False)
            and best.get("distance", float("inf"))
            <= self.stop_verification_required_hit_max_distance_m
        )
        if hit:
            self.stop_verification_hits += 1
            self.scenegraph.debug_stats.inc("stop_verification_hit")
            if near_hit:
                self.stop_verification_near_hits += 1
                self.scenegraph.debug_stats.inc("stop_verification_near_hit")
            else:
                self.scenegraph.debug_stats.inc("stop_verification_far_hit")
        else:
            self.scenegraph.debug_stats.inc("stop_verification_miss")

        self.stop_verification_observation_count += 1
        self.stop_verification_history.append({
            "step": int(self.total_steps),
            "hit": bool(hit),
            "near_hit": bool(near_hit),
            "hits": int(self.stop_verification_hits),
            "near_hits": int(self.stop_verification_near_hits),
            "near_hit_max_distance_m": float(
                self.stop_verification_required_hit_max_distance_m
            ),
            "steps_taken": int(self.stop_verification_observation_count),
            "observation_count": int(self.stop_verification_observation_count),
            "best_detection": best,
            "goal_node_support": node_support,
            "anchor_gps": np.asarray(anchor_gps, dtype=np.float32).tolist()
            if anchor_gps is not None
            else None,
            "anchor_delta": anchor_delta,
            "anchor_radius_m": float(self.stop_verification_anchor_radius_m),
            "anchor_drift_reset": bool(anchor_drift),
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

    def map_rc_to_goal_gps(self, rc):
        rc = np.asarray(rc, dtype=np.float32).reshape(-1)[:2]
        return np.array(
            [
                (rc[1] - self.map_size_cm / 10.0) * self.resolution / 100.0,
                (rc[0] - self.map_size_cm / 10.0) * self.resolution / 100.0,
            ],
            dtype=np.float32,
        )

    def traversible_patch_clear(self, traversible, rc, radius_cells):
        r, c = int(rc[0]), int(rc[1])
        radius_cells = max(0, int(radius_cells))
        r0 = max(0, r - radius_cells)
        r1 = min(traversible.shape[0], r + radius_cells + 1)
        c0 = max(0, c - radius_cells)
        c1 = min(traversible.shape[1], c + radius_cells + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        return bool(np.all(traversible[r0:r1, c0:c1] > 0.5))

    def traversible_line_of_sight_clear(
        self,
        traversible,
        start_rc,
        target_rc,
        *,
        endpoint_skip_cells=2,
    ):
        start_rc = np.asarray(start_rc, dtype=np.float32).reshape(-1)[:2]
        target_rc = np.asarray(target_rc, dtype=np.float32).reshape(-1)[:2]
        if start_rc.size != 2 or target_rc.size != 2:
            return False
        if not (np.all(np.isfinite(start_rc)) and np.all(np.isfinite(target_rc))):
            return False
        delta = target_rc - start_rc
        steps = int(max(abs(delta[0]), abs(delta[1])))
        if steps <= 0:
            return True
        endpoint_skip_cells = max(0, int(endpoint_skip_cells))
        last_checked = max(1, steps - endpoint_skip_cells)
        samples = np.linspace(start_rc, target_rc, steps + 1)
        for sample in samples[1:last_checked + 1]:
            r = int(round(float(sample[0])))
            c = int(round(float(sample[1])))
            if (
                r < 0
                or r >= traversible.shape[0]
                or c < 0
                or c >= traversible.shape[1]
                or traversible[r, c] <= 0.5
            ):
                return False
        return True

    def make_single_cell_goal_map(self, shape, rc):
        goal_map = np.zeros(shape, dtype=np.float32)
        r = int(np.clip(rc[0], 0, shape[0] - 1))
        c = int(np.clip(rc[1], 0, shape[1] - 1))
        goal_map[r, c] = 1
        return goal_map

    def select_planned_goal_approach_station(
        self,
        traversible,
        cur_start,
        target_gps,
        *,
        min_radius_m=None,
        max_radius_m=None,
        desired_radius_m=None,
        station_kind="approach",
        min_start_distance_m=0.0,
        exclude_stations=None,
        min_station_separation_m=0.0,
    ):
        station_kind = "retreat" if station_kind == "retreat" else "approach"
        if station_kind == "retreat":
            self.last_planned_goal_retreat_station = None
            self.last_planned_goal_retreat_selection_debug = {}
        else:
            self.last_planned_goal_approach_station = None
            self.last_planned_goal_approach_selection_debug = {}
        if traversible is None or cur_start is None:
            return None
        if target_gps is None or not np.all(np.isfinite(target_gps)):
            return None

        map_shape = self.full_map.shape[-2:]
        target_rc = np.rint(self.goal_gps_to_map_rc(target_gps)).astype(np.int64)
        if (
            target_rc[0] < 0
            or target_rc[0] >= map_shape[0]
            or target_rc[1] < 0
            or target_rc[1] >= map_shape[1]
        ):
            return None

        start_rc = np.asarray(cur_start, dtype=np.int64).reshape(-1)[:2]
        start_rc[0] = int(np.clip(start_rc[0], 0, map_shape[0] - 1))
        start_rc[1] = int(np.clip(start_rc[1], 0, map_shape[1] - 1))
        border_offset = (
            1
            if (
                traversible.shape[0] == map_shape[0] + 2
                and traversible.shape[1] == map_shape[1] + 2
            )
            else 0
        )
        start_trav_rc = start_rc + border_offset

        try:
            start_goal = self.make_single_cell_goal_map(
                traversible.shape, start_trav_rc
            )
            planner = FMMPlanner(traversible, None)
            planner.set_multi_goal(start_goal, start_trav_rc.tolist())
            path_dist = planner.fmm_dist
        except Exception:
            self.scenegraph.debug_stats.inc(
                f"planned_goal_{station_kind}_station_fmm_failed"
            )
            return None

        resolution_m = self.map_resolution / 100.0
        if min_radius_m is None:
            min_radius_m = self.planned_goal_approach_min_radius_m
        if max_radius_m is None:
            max_radius_m = self.planned_goal_approach_max_radius_m
        if desired_radius_m is None:
            desired_radius_m = self.planned_goal_stop_distance_m
        min_radius_m = float(min_radius_m)
        max_radius_m = max(min_radius_m, float(max_radius_m))
        desired_radius_m = float(np.clip(desired_radius_m, min_radius_m, max_radius_m))
        max_radius_cells = int(math.ceil(max_radius_m / resolution_m))
        footprint_radius = max(0, int(self.agent_footprint_clearance_cells))
        require_line_of_sight = bool(
            station_kind == "retreat"
            and self.planned_goal_retreat_require_line_of_sight
        )
        los_endpoint_skip_cells = self.planned_goal_retreat_los_endpoint_skip_cells
        target_trav_rc = target_rc + border_offset
        max_path_value = float(np.nanmax(path_dist)) if path_dist.size else float("inf")
        exclude_station_rcs = []
        for item in exclude_stations or []:
            station_rc = item.get("station_rc") if isinstance(item, dict) else item
            if station_rc is None:
                continue
            try:
                station_rc = np.asarray(station_rc, dtype=np.float32).reshape(-1)[:2]
            except Exception:
                continue
            if station_rc.size == 2 and np.all(np.isfinite(station_rc)):
                exclude_station_rcs.append(station_rc)
        min_sep_cells = (
            float(min_station_separation_m) / max(resolution_m, 1e-6)
            if min_station_separation_m
            else 0.0
        )

        best = None
        best_cost = float("inf")
        r0 = max(0, int(target_rc[0]) - max_radius_cells)
        r1 = min(map_shape[0], int(target_rc[0]) + max_radius_cells + 1)
        c0 = max(0, int(target_rc[1]) - max_radius_cells)
        c1 = min(map_shape[1], int(target_rc[1]) + max_radius_cells + 1)
        selection_debug = {
            "station_kind": station_kind,
            "target_rc": target_rc.astype(int).tolist(),
            "target_gps": np.asarray(target_gps, dtype=np.float32).tolist(),
            "start_rc": start_rc.astype(int).tolist(),
            "min_radius_m": float(min_radius_m),
            "max_radius_m": float(max_radius_m),
            "desired_radius_m": float(desired_radius_m),
            "require_line_of_sight": bool(require_line_of_sight),
            "los_endpoint_skip_cells": int(los_endpoint_skip_cells),
            "footprint_radius_cells": int(footprint_radius),
            "window_cells": int(max(0, r1 - r0) * max(0, c1 - c0)),
            "blocked_cells": 0,
            "radius_reject": 0,
            "separation_reject": 0,
            "start_distance_reject": 0,
            "footprint_reject": 0,
            "line_of_sight_reject": 0,
            "unreachable_reject": 0,
            "candidate_cells": 0,
        }

        for r in range(r0, r1):
            for c in range(c0, c1):
                trav_r = r + border_offset
                trav_c = c + border_offset
                if (
                    trav_r < 0
                    or trav_r >= traversible.shape[0]
                    or trav_c < 0
                    or trav_c >= traversible.shape[1]
                    or traversible[trav_r, trav_c] <= 0.5
                ):
                    selection_debug["blocked_cells"] += 1
                    continue
                object_dist_m = (
                    float(np.linalg.norm(np.asarray([r, c]) - target_rc))
                    * resolution_m
                )
                if object_dist_m < min_radius_m or object_dist_m > max_radius_m:
                    selection_debug["radius_reject"] += 1
                    continue
                if min_sep_cells > 0.0 and any(
                    float(np.linalg.norm(np.asarray([r, c], dtype=np.float32) - excluded_rc))
                    < min_sep_cells
                    for excluded_rc in exclude_station_rcs
                ):
                    selection_debug["separation_reject"] += 1
                    continue
                if min_start_distance_m and min_start_distance_m > 0:
                    start_dist_m = (
                        float(np.linalg.norm(np.asarray([r, c]) - start_rc))
                        * resolution_m
                    )
                    if start_dist_m < min_start_distance_m:
                        selection_debug["start_distance_reject"] += 1
                        continue
                if not self.traversible_patch_clear(
                    traversible, [trav_r, trav_c], footprint_radius
                ):
                    selection_debug["footprint_reject"] += 1
                    continue
                if require_line_of_sight and not self.traversible_line_of_sight_clear(
                    traversible,
                    [trav_r, trav_c],
                    target_trav_rc,
                    endpoint_skip_cells=los_endpoint_skip_cells,
                ):
                    selection_debug["line_of_sight_reject"] += 1
                    continue
                path_cells = float(path_dist[trav_r, trav_c])
                if (
                    not np.isfinite(path_cells)
                    or path_cells >= max_path_value
                ):
                    selection_debug["unreachable_reject"] += 1
                    continue
                selection_debug["candidate_cells"] += 1
                path_m = path_cells * resolution_m
                radius_cost = (
                    self.planned_goal_approach_radius_cost
                    * abs(object_dist_m - desired_radius_m)
                )
                cost = path_m + radius_cost
                if cost < best_cost:
                    best_cost = cost
                    best = {
                        "station_rc": [int(r), int(c)],
                        "station_gps": self.map_rc_to_goal_gps([r, c]).tolist(),
                        "target_rc": target_rc.astype(int).tolist(),
                        "target_gps": np.asarray(target_gps, dtype=np.float32).tolist(),
                        "object_distance_m": float(object_dist_m),
                        "path_distance_m": float(path_m),
                        "cost": float(cost),
                        "min_radius_m": float(min_radius_m),
                        "max_radius_m": float(max_radius_m),
                        "desired_radius_m": float(desired_radius_m),
                        "station_kind": station_kind,
                        "line_of_sight_clear": True if require_line_of_sight else None,
                        "line_of_sight_required": bool(require_line_of_sight),
                    }

        if best is None:
            selection_debug["selected"] = False
            if station_kind == "retreat":
                self.last_planned_goal_retreat_selection_debug = selection_debug
            else:
                self.last_planned_goal_approach_selection_debug = selection_debug
            self.scenegraph.debug_stats.inc(
                f"planned_goal_{station_kind}_station_missing"
            )
            return None

        selection_debug["selected"] = True
        selection_debug["best_station"] = best
        self.scenegraph.debug_stats.inc(f"planned_goal_{station_kind}_station_selected")
        if station_kind == "retreat":
            self.last_planned_goal_retreat_station = best
            self.last_planned_goal_retreat_selection_debug = selection_debug
        else:
            self.last_planned_goal_approach_station = best
            self.last_planned_goal_approach_selection_debug = selection_debug
        return self.make_single_cell_goal_map(map_shape, best["station_rc"])

    def get_planned_goal_approach_action(
        self,
        traversible,
        cur_start,
        cur_start_o,
        target_gps,
        approach_goal_map=None,
        station_stop_distance_m=None,
        min_radius_m=None,
        max_radius_m=None,
        desired_radius_m=None,
        station_kind="approach",
    ):
        if not self.planned_goal_approach_enabled:
            return None
        if traversible is None or cur_start is None or cur_start_o is None:
            return None
        if target_gps is None or not np.all(np.isfinite(target_gps)):
            return None
        goal_map = approach_goal_map
        if station_stop_distance_m is None:
            station_stop_distance_m = self.planned_goal_approach_station_stop_distance_m
        if goal_map is None:
            goal_map = self.select_planned_goal_approach_station(
                traversible,
                cur_start,
                target_gps,
                min_radius_m=min_radius_m,
                max_radius_m=max_radius_m,
                desired_radius_m=desired_radius_m,
                station_kind=station_kind,
            )
        if goal_map is None:
            if station_kind == "retreat":
                return None
            goal_map = self.goal_gps_to_goal_map(target_gps)
            station_stop_distance_m = self.planned_goal_stop_distance_m
        _, _, _, action = self._plan(
            traversible,
            goal_map,
            self.full_pose,
            cur_start,
            cur_start_o,
            True,
            stop_distance_m=station_stop_distance_m,
        )
        return int(action)

    def record_failed_goal_viewpoint(self, station, reason):
        station_missing = station is None
        if station_missing:
            target_gps = getattr(self, "stop_verification_target_gps", None)
            if target_gps is not None:
                station_rc = self.goal_gps_to_map_rc(target_gps).tolist()
            else:
                station_rc = [
                    -1.0,
                    float(len(getattr(self, "planned_goal_failed_viewpoints", []))),
                ]
            station = {
                "station_kind": "retreat",
                "station_rc": station_rc,
                "station_gps": None,
                "station_missing": True,
                "missing_attempt": int(
                    len(getattr(self, "planned_goal_failed_viewpoints", [])) + 1
                ),
            }
        if station is None:
            return
        station_rc = station.get("station_rc") if isinstance(station, dict) else None
        if station_rc is None:
            return
        try:
            station_rc_np = np.asarray(station_rc, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return
        if station_rc_np.size != 2 or not np.all(np.isfinite(station_rc_np)):
            return
        if not station_missing:
            for item in getattr(self, "planned_goal_failed_viewpoints", []):
                old_rc = np.asarray(item.get("station_rc", []), dtype=np.float32).reshape(-1)[:2]
                if old_rc.size == 2 and float(np.linalg.norm(old_rc - station_rc_np)) < 1.0:
                    return
        entry = dict(station)
        entry["failed_step"] = int(self.total_steps)
        entry["failed_reason"] = str(reason)
        self.planned_goal_failed_viewpoints.append(entry)
        self.scenegraph.debug_stats.inc("planned_goal_viewpoint_failed")
        self.log_candidate_event(
            "viewpoint_failed",
            source=getattr(self, "reperception_source", ""),
            decision="pending",
            reason=reason,
            failed_viewpoint=entry,
            failed_viewpoint_count=len(self.planned_goal_failed_viewpoints),
            planned_goal_viewpoint_max_attempts=int(
                self.planned_goal_viewpoint_max_attempts
            ),
        )

    def planned_goal_retreat_failure_count(self):
        count = 0
        for item in getattr(self, "planned_goal_failed_viewpoints", []):
            failed_reason = str(item.get("failed_reason", ""))
            if item.get("station_kind") == "retreat" or failed_reason.startswith(
                "planned_goal_retreat"
            ):
                count += 1
        return int(count)

    def planned_goal_retreat_attempts_exhausted(self):
        return (
            self.planned_goal_retreat_failure_count()
            >= self.planned_goal_retreat_viewpoint_attempts
        )

    def hard_reject_goal_after_viewpoints(self, reason):
        hard_reason = "hard_false_positive_" + str(reason)
        self.register_candidate_miss(
            "planned_goal_viewpoint_search",
            reason=hard_reason,
            visibility={
                "expected_visible": True,
                "reason": hard_reason,
                "failed_viewpoints": getattr(
                    self, "planned_goal_failed_viewpoints", []
                ),
                "blacklist_visibility": getattr(
                    self, "stop_verification_blacklist_visibility", None
                ),
                "blacklist_visible_seen": bool(
                    getattr(self, "stop_verification_blacklist_visible_seen", False)
                ),
                "visible_blacklist_max_distance_m": float(
                    self.rejected_goal_visible_blacklist_max_distance_m
                ),
            },
        )
        self.scenegraph.debug_stats.inc("planned_goal_viewpoint_hard_reject")
        self.stop_reason = hard_reason
        self.reject_confirmed_goal(reason=hard_reason)
        self.reset_stop_verification_state(clear_history=False)

    def reject_confirmed_goal(self, reason):
        self.finalize_candidate("reject", reason)
        reject_gps = getattr(self, "goal_gps", None)
        if reject_gps is None:
            reject_gps = getattr(self, "stop_verification_target_gps", None)
        if reject_gps is None:
            reject_gps = getattr(self, "reperception_goal_gps", None)
        if reject_gps is None:
            reject_gps = getattr(self, "possible_goal_temp_gps", None)
        self.add_rejected_goal_candidate(reject_gps, reason)
        self.found_goal = False
        self.found_possible_goal = False
        self.found_goal_times = 0
        self.goal_gps_map.fill(0)
        self.first_fbe = True
        self.reperception_active = False
        self.reperception_goal_gps = None
        self.reperception_goal_map_xy = None
        self.reperception_source = ""
        self.reperception_score_sum = 0.0
        self.reperception_steps = 0
        self.reperception_observation_count = 0
        self.current_candidate = None
        self.reset_candidate_progress_tracking()
        self.not_use_random_goal()
        self.scenegraph.debug_stats.inc("stop_verification_rejected_goal")

    def handle_stop_verification(
        self,
        observations,
        traversible=None,
        cur_start=None,
        cur_start_o=None,
    ):
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
            self.stop_verification_anchor_gps = self.stop_verification_target_gps.copy()
            self.stop_verification_steps_taken = 0
            self.stop_verification_hits = 0
            self.stop_verification_near_hits = 0
            hit, best_detection, node_support = self.observe_stop_verification(observations)

            if (
                hit
                and getattr(self, "stop_verification_near_hits", 0) > 0
                and self.should_stop_near_candidate_goal()
            ):
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
            candidate_gps, _ = self.get_stop_liveness_candidate_gps()
            if candidate_gps is None:
                self.stop_reason = "stop_verification_no_candidate"
                return False, self.stop_verification_turn_action
            self.stop_verification_active = True
            self.stop_verification_target_gps = np.asarray(
                candidate_gps, dtype=np.float32
            ).copy()
            self.stop_verification_anchor_gps = self.stop_verification_target_gps.copy()
            self.stop_verification_steps_taken = 0
            self.stop_verification_hits = 0
            self.stop_verification_near_hits = 0
            self.stop_verification_observation_count = 0
            self.stop_verification_consecutive_failures = 0
            self.planned_goal_arrival_scan_steps_taken = 0
            self.planned_goal_retreat_active = False
            self.planned_goal_retreat_steps = 0
            self.planned_goal_retreat_blocked_steps = 0
            self.planned_goal_retreat_scan_steps_taken = 0
            self.planned_goal_failed_viewpoints = []
            self.last_planned_goal_retreat_station = None
            self.stop_verification_blacklist_visible_seen = False
            self.stop_verification_blacklist_visibility = None
            self.scenegraph.debug_stats.inc("stop_verification_started")

        hit, best_detection, node_support = self.observe_stop_verification(observations)
        if hit and best_detection is not None:
            self.stop_verification_consecutive_failures = 0
            self.planned_goal_arrival_scan_steps_taken = 0
            self.stop_verification_target_gps = np.asarray(
                best_detection["gps"],
                dtype=np.float32,
            )
            self.goal_gps = self.stop_verification_target_gps.copy()
            node_support = self.get_goal_node_support(self.stop_verification_target_gps)
        elif (
            self.stop_verification_history
            and self.stop_verification_history[-1].get("anchor_drift_reset", False)
        ):
            self.stop_verification_consecutive_failures = 0
            self.planned_goal_arrival_scan_steps_taken = 0
            self.planned_goal_retreat_scan_steps_taken = 0
            if getattr(self, "stop_verification_anchor_gps", None) is not None:
                self.stop_verification_target_gps = np.asarray(
                    self.stop_verification_anchor_gps, dtype=np.float32
                ).copy()
                self.goal_gps = self.stop_verification_target_gps.copy()
        target_distance = self.get_distance_to_stop_target(observations)
        blacklist_visibility = self.update_stop_blacklist_visibility_evidence(
            observations,
            self.stop_verification_target_gps,
            hit,
        )
        approach_goal_map = None
        approach_station = None
        station_distance = float("inf")
        distance_to_stop_point = target_distance
        stop_point_kind = "target"
        planned_station_kind = (
            "retreat"
            if getattr(self, "planned_goal_retreat_active", False)
            else "approach"
        )
        verification_stop_distance_m = (
            self.planned_goal_stop_distance_m
            if self.planned_goal_approach_enabled
            else self.direct_goal_approach_min_distance_m
        )
        if self.planned_goal_approach_enabled:
            station_min_radius_m = self.planned_goal_approach_min_radius_m
            station_max_radius_m = self.planned_goal_approach_max_radius_m
            station_desired_radius_m = self.planned_goal_stop_distance_m
            station_stop_distance_m = self.planned_goal_approach_station_stop_distance_m
            if planned_station_kind == "retreat":
                station_min_radius_m = self.planned_goal_retreat_min_radius_m
                station_max_radius_m = self.planned_goal_retreat_max_radius_m
                station_desired_radius_m = (
                    self.planned_goal_retreat_min_radius_m
                    + self.planned_goal_retreat_max_radius_m
                ) / 2.0
                station_stop_distance_m = (
                    self.planned_goal_retreat_station_stop_distance_m
                )
            approach_goal_map = self.select_planned_goal_approach_station(
                traversible,
                cur_start,
                self.stop_verification_target_gps,
                min_radius_m=station_min_radius_m,
                max_radius_m=station_max_radius_m,
                desired_radius_m=station_desired_radius_m,
                station_kind=planned_station_kind,
                exclude_stations=(
                    self.planned_goal_failed_viewpoints
                    if planned_station_kind == "retreat"
                    else None
                ),
                min_station_separation_m=(
                    self.planned_goal_viewpoint_min_separation_m
                    if planned_station_kind == "retreat"
                    else 0.0
                ),
            )
            approach_station = (
                self.last_planned_goal_retreat_station
                if planned_station_kind == "retreat"
                else self.last_planned_goal_approach_station
            )
            retreat_station_missing = bool(
                planned_station_kind == "retreat" and approach_station is None
            )
            if approach_station is not None:
                station_gps = np.asarray(
                    approach_station.get("station_gps", [np.nan, np.nan]),
                    dtype=np.float32,
                )
                if station_gps.size == 2 and np.all(np.isfinite(station_gps)):
                    station_distance = self.distance_to_gps(station_gps, observations)
                    distance_to_stop_point = station_distance
                    stop_point_kind = f"planned_{planned_station_kind}_station"
                    verification_stop_distance_m = station_stop_distance_m
            elif planned_station_kind == "retreat":
                stop_point_kind = "planned_retreat_station_missing"
                distance_to_stop_point = float("inf")
                verification_stop_distance_m = station_stop_distance_m
        else:
            retreat_station_missing = False
        if self.stop_verification_history:
            self.stop_verification_history[-1]["target_distance"] = target_distance
            self.stop_verification_history[-1]["stop_point_distance"] = (
                float(distance_to_stop_point)
            )
            self.stop_verification_history[-1]["stop_point_kind"] = stop_point_kind
            self.stop_verification_history[-1]["station_distance"] = (
                float(station_distance)
            )
            self.stop_verification_history[-1][
                "planned_goal_approach_station"
            ] = approach_station
            self.stop_verification_history[-1]["updated_target_gps"] = (
                np.asarray(self.stop_verification_target_gps, dtype=np.float32).tolist()
            )
            self.stop_verification_history[-1]["goal_node_support"] = node_support
            self.stop_verification_history[-1]["planned_station_kind"] = (
                planned_station_kind
            )
            self.stop_verification_history[-1]["planned_goal_retreat_active"] = (
                bool(getattr(self, "planned_goal_retreat_active", False))
            )
            self.stop_verification_history[-1]["planned_goal_failed_viewpoints"] = (
                list(getattr(self, "planned_goal_failed_viewpoints", []))
            )
            self.stop_verification_history[-1][
                "planned_goal_retreat_station_missing"
            ] = bool(retreat_station_missing)
            self.stop_verification_history[-1][
                "planned_goal_station_selection_debug"
            ] = (
                self.last_planned_goal_retreat_selection_debug
                if planned_station_kind == "retreat"
                else self.last_planned_goal_approach_selection_debug
            )
            self.stop_verification_history[-1]["blacklist_visibility"] = (
                blacklist_visibility
            )
            self.stop_verification_history[-1][
                "blacklist_visible_seen"
            ] = bool(getattr(self, "stop_verification_blacklist_visible_seen", False))
            self.stop_verification_history[-1][
                "visible_blacklist_max_distance_m"
            ] = float(self.rejected_goal_visible_blacklist_max_distance_m)

        final_detection_seen = (
            best_detection is not None
            and node_support["supported"]
            and (
                not best_detection.get("depth_valid", False)
                or best_detection["delta"] <= self.stop_verification_same_goal_radius_m
            )
        )
        stop_point_close = distance_to_stop_point <= verification_stop_distance_m
        target_stop_distance_m = (
            self.planned_goal_stop_distance_m
            if self.planned_goal_approach_enabled
            else verification_stop_distance_m
        )
        target_distance_close = target_distance <= target_stop_distance_m
        require_station_close = bool(
            self.planned_goal_approach_enabled
            and planned_station_kind == "retreat"
        )
        close_to_target = bool(
            stop_point_close
            if require_station_close
            else (stop_point_close or target_distance_close)
        )
        candidate_support = self.current_candidate_detector_support_for(
            self.stop_verification_target_gps
        )
        candidate_detector_supported = candidate_support["supported"]
        candidate_evidence_hits = (
            int(candidate_support["hit_count"])
            if candidate_support["matches"]
            else 0
        )
        candidate_max_consecutive_hits = (
            int(candidate_support["max_consecutive_hit_count"])
            if candidate_support["matches"]
            else 0
        )
        strong_historical_evidence = bool(
            candidate_support["matches"]
            and candidate_evidence_hits >= self.candidate_strong_evidence_min_hits
        )
        strong_consecutive_evidence = bool(
            strong_historical_evidence
        )
        candidate_near_hits = (
            int(candidate_support["near_hit_count"])
            if candidate_support["matches"]
            else 0
        )
        verification_evidence_hits = int(getattr(self, "stop_verification_hits", 0))
        verification_near_hits = int(
            getattr(self, "stop_verification_near_hits", 0)
        )
        near_visual_hits = candidate_near_hits + verification_near_hits
        cumulative_evidence_hits = (
            candidate_evidence_hits + verification_evidence_hits
        )
        candidate_direct_goal_count = (
            self.current_candidate_direct_goal_contribution_count()
        )
        candidate_has_direct_goal_support = candidate_direct_goal_count > 0
        try:
            best_detection_score = (
                float(best_detection.get("score", 0.0) or 0.0)
                if best_detection is not None
                else 0.0
            )
        except Exception:
            best_detection_score = 0.0
        high_confidence_current_hit = bool(
            close_to_target
            and hit
            and near_visual_hits > 0
            and best_detection_score >= self.stop_verification_force_stop_confidence
        )
        historical_candidate_supported = bool(
            self.planned_goal_approach_enabled
            and self.found_goal
            and candidate_detector_supported
            and node_support["supported"]
            and candidate_has_direct_goal_support
        )
        weak_historical_stop = bool(
            historical_candidate_supported
            and cumulative_evidence_hits >= self.stop_verification_min_hits
            and self.stop_verification_observation_count >= max(1, self.stop_verification_steps)
        )
        enough_hits = cumulative_evidence_hits >= self.stop_verification_min_hits
        retreat_confirmed_evidence = bool(
            getattr(self, "planned_goal_retreat_confirmed", False)
        )
        raw_enough_evidence = bool(
            enough_hits
            or high_confidence_current_hit
            or strong_consecutive_evidence
            or retreat_confirmed_evidence
            or (weak_historical_stop and close_to_target)
        )
        near_visual_hit_ready = bool(
            (not self.stop_require_near_visual_hit)
            or near_visual_hits > 0
            or strong_consecutive_evidence
            or high_confidence_current_hit
            or retreat_confirmed_evidence
        )
        enough_evidence = bool(raw_enough_evidence and near_visual_hit_ready)
        if historical_candidate_supported and not enough_hits:
            self.scenegraph.debug_stats.inc("stop_verification_historical_support")
        if weak_historical_stop:
            self.scenegraph.debug_stats.inc("stop_verification_weak_historical_ready")
        if raw_enough_evidence and not near_visual_hit_ready:
            self.scenegraph.debug_stats.inc(
                "stop_verification_waiting_for_verification_hit"
            )
            self.scenegraph.debug_stats.inc(
                "stop_verification_waiting_for_near_hit"
            )
        if high_confidence_current_hit:
            self.scenegraph.debug_stats.inc("stop_verification_high_confidence_ready")
        if strong_consecutive_evidence:
            self.scenegraph.debug_stats.inc(
                "stop_verification_strong_historical_evidence_ready"
            )

        if self.stop_verification_history:
            self.stop_verification_history[-1]["verification_stop_distance_m"] = (
                float(verification_stop_distance_m)
            )
            self.stop_verification_history[-1]["target_stop_distance_m"] = (
                float(target_stop_distance_m)
            )
            self.stop_verification_history[-1]["stop_point_close"] = (
                bool(stop_point_close)
            )
            self.stop_verification_history[-1]["target_distance_close"] = (
                bool(target_distance_close)
            )
            self.stop_verification_history[-1]["require_station_close"] = (
                bool(require_station_close)
            )
            self.stop_verification_history[-1]["close_to_target"] = (
                bool(close_to_target)
            )
            self.stop_verification_history[-1]["planned_goal_approach_enabled"] = (
                bool(self.planned_goal_approach_enabled)
            )
            self.stop_verification_history[-1]["candidate_detector_supported"] = (
                bool(candidate_detector_supported)
            )
            self.stop_verification_history[-1]["candidate_evidence_matches"] = (
                bool(candidate_support["matches"])
            )
            self.stop_verification_history[-1]["candidate_evidence_distance_m"] = (
                float(candidate_support["distance_m"])
            )
            self.stop_verification_history[-1][
                "candidate_evidence_binding_radius_m"
            ] = float(candidate_support["binding_radius_m"])
            self.stop_verification_history[-1]["current_candidate_id"] = (
                candidate_support["candidate_id"]
            )
            self.stop_verification_history[-1]["candidate_hit_count"] = (
                int(candidate_support["hit_count"])
            )
            self.stop_verification_history[-1]["candidate_consecutive_hit_count"] = (
                int(candidate_support["consecutive_hit_count"])
            )
            self.stop_verification_history[-1][
                "candidate_max_consecutive_hit_count"
            ] = int(candidate_max_consecutive_hits)
            self.stop_verification_history[-1][
                "candidate_strong_consecutive_evidence"
            ] = bool(strong_consecutive_evidence)
            self.stop_verification_history[-1][
                "candidate_strong_historical_evidence"
            ] = bool(strong_historical_evidence)
            self.stop_verification_history[-1][
                "candidate_strong_evidence_min_hits"
            ] = int(self.candidate_strong_evidence_min_hits)
            self.stop_verification_history[-1][
                "candidate_strong_evidence_min_consecutive_hits"
            ] = int(self.candidate_strong_evidence_min_consecutive_hits)
            self.stop_verification_history[-1]["candidate_evidence_hits"] = (
                int(candidate_evidence_hits)
            )
            self.stop_verification_history[-1]["candidate_near_hits"] = (
                int(candidate_near_hits)
            )
            self.stop_verification_history[-1]["verification_evidence_hits"] = (
                int(verification_evidence_hits)
            )
            self.stop_verification_history[-1]["verification_near_hits"] = (
                int(verification_near_hits)
            )
            self.stop_verification_history[-1]["near_visual_hits"] = (
                int(near_visual_hits)
            )
            self.stop_verification_history[-1]["cumulative_evidence_hits"] = (
                int(cumulative_evidence_hits)
            )
            self.stop_verification_history[-1]["candidate_distinct_view_count"] = (
                int(candidate_support["distinct_view_count"])
            )
            self.stop_verification_history[-1]["candidate_hit_ratio"] = (
                float(candidate_support["hit_ratio"])
            )
            self.stop_verification_history[-1][
                "candidate_direct_goal_count"
            ] = int(candidate_direct_goal_count)
            self.stop_verification_history[-1][
                "candidate_has_direct_goal_support"
            ] = bool(candidate_has_direct_goal_support)
            self.stop_verification_history[-1]["enough_evidence"] = (
                bool(enough_evidence)
            )
            self.stop_verification_history[-1]["enough_hits"] = bool(enough_hits)
            self.stop_verification_history[-1][
                "raw_enough_evidence"
            ] = bool(raw_enough_evidence)
            self.stop_verification_history[-1][
                "verification_hit_required"
            ] = bool(self.stop_require_near_visual_hit)
            self.stop_verification_history[-1][
                "verification_required_hit_max_distance_m"
            ] = float(self.stop_verification_required_hit_max_distance_m)
            self.stop_verification_history[-1][
                "verification_hit_ready"
            ] = bool(near_visual_hit_ready)
            self.stop_verification_history[-1][
                "verification_near_hit_ready"
            ] = bool(near_visual_hit_ready)
            self.stop_verification_history[-1][
                "near_visual_hit_ready"
            ] = bool(near_visual_hit_ready)
            self.stop_verification_history[-1][
                "high_confidence_current_hit"
            ] = bool(high_confidence_current_hit)
            self.stop_verification_history[-1][
                "best_detection_score"
            ] = float(best_detection_score)
            self.stop_verification_history[-1][
                "force_stop_confidence"
            ] = float(self.stop_verification_force_stop_confidence)
            self.stop_verification_history[-1][
                "historical_candidate_supported"
            ] = bool(historical_candidate_supported)
            self.stop_verification_history[-1][
                "weak_historical_stop"
            ] = bool(weak_historical_stop)
            self.stop_verification_history[-1][
                "planned_goal_retreat_confirmed"
            ] = bool(retreat_confirmed_evidence)
            self.stop_verification_history[-1][
                "verification_observation_count"
            ] = int(self.stop_verification_observation_count)
            self.stop_verification_history[-1][
                "planned_goal_verification_max_observations"
            ] = int(self.planned_goal_verification_max_observations)

        if self.planned_goal_approach_enabled:
            planned_action = None
            if not close_to_target:
                planned_action = self.get_planned_goal_approach_action(
                    traversible,
                    cur_start,
                    cur_start_o,
                    self.stop_verification_target_gps,
                    approach_goal_map=approach_goal_map,
                    station_stop_distance_m=verification_stop_distance_m,
                    min_radius_m=station_min_radius_m,
                    max_radius_m=station_max_radius_m,
                    desired_radius_m=station_desired_radius_m,
                    station_kind=planned_station_kind,
                )
                if self.stop_verification_history:
                    self.stop_verification_history[-1][
                        "planned_goal_approach_action"
                    ] = int(planned_action) if planned_action is not None else None
                    self.stop_verification_history[-1][
                        "planned_goal_approach_steps"
                    ] = int(self.planned_goal_approach_steps)
                    self.stop_verification_history[-1][
                        "planned_goal_approach_station"
                    ] = self.last_planned_goal_approach_station
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_action"
                    ] = int(planned_action) if planned_action is not None else None
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_steps"
                    ] = int(self.planned_goal_retreat_steps)
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_station"
                    ] = self.last_planned_goal_retreat_station
            planned_arrived = close_to_target or planned_action == 0
            if self.stop_verification_history:
                self.stop_verification_history[-1]["planned_goal_arrived"] = (
                    bool(planned_arrived)
                )
                self.stop_verification_history[-1][
                    "planned_goal_arrival_scan_steps"
                ] = int(self.planned_goal_arrival_scan_steps_taken)
                self.stop_verification_history[-1][
                    "planned_goal_retreat_scan_steps"
                ] = int(self.planned_goal_retreat_scan_steps_taken)

            retreat_viewpoint_hit = bool(planned_station_kind == "retreat" and hit)
            retreat_viewpoint_hit_at_station = bool(
                retreat_viewpoint_hit and planned_arrived
            )
            if self.stop_verification_history:
                self.stop_verification_history[-1][
                    "planned_goal_retreat_viewpoint_hit"
                ] = bool(retreat_viewpoint_hit)
                self.stop_verification_history[-1][
                    "planned_goal_retreat_viewpoint_hit_at_station"
                ] = bool(retreat_viewpoint_hit_at_station)

            retreat_ready_for_reapproach = bool(
                planned_station_kind == "retreat"
                and planned_arrived
                and (
                    retreat_viewpoint_hit
                    or strong_consecutive_evidence
                )
            )
            if self.stop_verification_history:
                self.stop_verification_history[-1][
                    "planned_goal_retreat_ready_for_reapproach"
                ] = bool(retreat_ready_for_reapproach)

            if retreat_ready_for_reapproach:
                self.planned_goal_retreat_active = False
                self.planned_goal_retreat_steps = 0
                self.planned_goal_retreat_blocked_steps = 0
                self.planned_goal_retreat_scan_steps_taken = 0
                self.last_planned_goal_retreat_station = None
                self.planned_goal_approach_steps = 0
                self.planned_goal_approach_blocked_steps = 0
                self.planned_goal_arrival_scan_steps_taken = 0
                self.stop_verification_consecutive_failures = 0
                self.stop_reason = (
                    "planned_goal_retreat_hit_reapproach"
                    if retreat_viewpoint_hit_at_station
                    else "planned_goal_retreat_strong_evidence_reapproach"
                )
                self.scenegraph.debug_stats.inc(
                    self.stop_reason
                )
                self.log_candidate_event(
                    "retreat_viewpoint_confirmed",
                    candidate=getattr(self, "current_candidate", None),
                    candidate_gps=self.stop_verification_target_gps,
                    source=getattr(self, "reperception_source", ""),
                    decision="approach",
                    reason=self.stop_reason,
                    planned_goal_retreat_station=approach_station,
                    retreat_viewpoint_hit=bool(retreat_viewpoint_hit),
                    retreat_viewpoint_hit_at_station=bool(
                        retreat_viewpoint_hit_at_station
                    ),
                    strong_consecutive_evidence=bool(strong_consecutive_evidence),
                    near_visual_hits=int(near_visual_hits),
                    cumulative_evidence_hits=int(cumulative_evidence_hits),
                    target_distance_m=float(target_distance),
                    station_distance_m=float(station_distance),
                )
                if self.stop_verification_history:
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_confirmed_reapproach"
                    ] = True
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_reapproach_reason"
                    ] = self.stop_reason
                return False, self.stop_verification_turn_action

            if planned_station_kind == "retreat" and retreat_station_missing:
                self.record_failed_goal_viewpoint(
                    None,
                    "planned_goal_retreat_station_missing",
                )
                self.scenegraph.debug_stats.inc(
                    "planned_goal_retreat_station_missing_failed"
                )
                if self.stop_verification_history:
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_station_missing_failed"
                    ] = True
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_failure_count"
                    ] = int(self.planned_goal_retreat_failure_count())
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_viewpoint_attempts"
                    ] = int(self.planned_goal_retreat_viewpoint_attempts)
                self.hard_reject_goal_after_viewpoints(
                    "planned_goal_retreat_station_missing"
                )
                return False, self.stop_verification_turn_action

            if enough_evidence and close_to_target:
                confirm_reason = (
                    "stop_verification_high_confidence"
                    if high_confidence_current_hit
                    else "stop_verification_confirmed"
                )
                self.finalize_candidate("confirm", confirm_reason)
                self.stop_reason = confirm_reason
                self.scenegraph.debug_stats.inc("stop_verification_confirmed")
                if high_confidence_current_hit:
                    self.scenegraph.debug_stats.inc(
                        "stop_verification_high_confidence"
                    )
                self.reset_stop_verification_state(clear_history=False)
                return True, 0

            planned_verification_timed_out = bool(
                close_to_target
                and not enough_evidence
                and self.stop_verification_observation_count
                >= self.planned_goal_verification_max_observations
                and self.planned_goal_retreat_attempts_exhausted()
            )
            if self.stop_verification_history:
                self.stop_verification_history[-1][
                    "planned_goal_verification_timed_out"
                ] = bool(planned_verification_timed_out)
                self.stop_verification_history[-1][
                    "planned_goal_retreat_failure_count"
                ] = int(self.planned_goal_retreat_failure_count())
                self.stop_verification_history[-1][
                    "planned_goal_retreat_viewpoint_attempts"
                ] = int(self.planned_goal_retreat_viewpoint_attempts)
            if planned_verification_timed_out:
                self.stop_reason = "planned_goal_verification_timeout"
                self.reject_confirmed_goal(reason="planned_goal_verification_timeout")
                self.reset_stop_verification_state(clear_history=False)
                return False, self.stop_verification_turn_action

            if (
                close_to_target
                and not enough_evidence
                and planned_station_kind == "retreat"
                and self.planned_goal_retreat_scan_steps_taken
                < self.planned_goal_retreat_scan_steps
            ):
                retreat_turn_action, retreat_turn_info = (
                    self.get_spatial_turn_action_towards_gps(
                        observations,
                        self.stop_verification_target_gps,
                    )
                )
                if (
                    retreat_turn_action is None
                    and retreat_turn_info.get("reason")
                    in {"spatial_target_aligned", "target_too_close"}
                ):
                    self.record_failed_goal_viewpoint(
                        approach_station,
                        "planned_goal_retreat_aligned_no_detection",
                    )
                    self.scenegraph.debug_stats.inc(
                        "planned_goal_retreat_aligned_no_detection"
                    )
                    if self.stop_verification_history:
                        self.stop_verification_history[-1][
                            "planned_goal_retreat_scan_steps"
                        ] = int(self.planned_goal_retreat_scan_steps_taken)
                        self.stop_verification_history[-1][
                            "planned_goal_retreat_spatial_turn"
                        ] = retreat_turn_info
                        self.stop_verification_history[-1][
                            "planned_goal_retreat_scan_action"
                        ] = None
                    if self.planned_goal_retreat_attempts_exhausted():
                        self.hard_reject_goal_after_viewpoints(
                            "planned_goal_viewpoints_failed"
                        )
                        return False, self.stop_verification_turn_action

                    self.planned_goal_retreat_steps = 0
                    self.planned_goal_retreat_blocked_steps = 0
                    self.planned_goal_retreat_scan_steps_taken = 0
                    self.last_planned_goal_retreat_station = None
                    self.stop_reason = "planned_goal_next_viewpoint"
                    self.scenegraph.debug_stats.inc("planned_goal_next_viewpoint")
                    return False, self.stop_verification_turn_action

                self.planned_goal_retreat_scan_steps_taken += 1
                self.stop_verification_consecutive_failures = 0
                self.stop_reason = "planned_goal_retreat_spatial_turning"
                self.scenegraph.debug_stats.inc("planned_goal_retreat_scan")
                scan_action = (
                    retreat_turn_action
                    if retreat_turn_action is not None
                    else self.stop_verification_turn_action
                )
                if self.stop_verification_history:
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_scan_steps"
                    ] = int(self.planned_goal_retreat_scan_steps_taken)
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_spatial_turn"
                    ] = retreat_turn_info
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_scan_action"
                    ] = int(scan_action)
                if retreat_turn_action is not None:
                    self.scenegraph.debug_stats.inc(
                        "planned_goal_retreat_spatial_turn"
                    )
                return False, scan_action

            if (
                close_to_target
                and not enough_evidence
                and planned_station_kind == "retreat"
                and self.planned_goal_retreat_scan_steps_taken
                >= self.planned_goal_retreat_scan_steps
            ):
                self.record_failed_goal_viewpoint(
                    approach_station,
                    "planned_goal_retreat_scan_failed",
                )
                if self.planned_goal_retreat_attempts_exhausted():
                    self.hard_reject_goal_after_viewpoints(
                        "planned_goal_viewpoints_failed"
                    )
                    return False, self.stop_verification_turn_action

                self.planned_goal_retreat_steps = 0
                self.planned_goal_retreat_blocked_steps = 0
                self.planned_goal_retreat_scan_steps_taken = 0
                self.last_planned_goal_retreat_station = None
                self.stop_reason = "planned_goal_next_viewpoint"
                self.scenegraph.debug_stats.inc("planned_goal_next_viewpoint")
                return False, self.stop_verification_turn_action

            if (
                close_to_target
                and not enough_evidence
                and planned_station_kind == "approach"
                and self.planned_goal_arrival_scan_steps_taken
                < self.planned_goal_arrival_scan_steps
            ):
                self.planned_goal_arrival_scan_steps_taken += 1
                self.stop_verification_consecutive_failures = 0
                self.stop_reason = "planned_goal_arrival_scanning"
                self.scenegraph.debug_stats.inc("planned_goal_arrival_scan")
                if self.stop_verification_history:
                    self.stop_verification_history[-1][
                        "planned_goal_arrival_scan_steps"
                    ] = int(self.planned_goal_arrival_scan_steps_taken)
                return False, self.stop_verification_turn_action

            if (
                close_to_target
                and not enough_evidence
                and planned_station_kind == "approach"
                and self.planned_goal_retreat_enabled
                and self.planned_goal_retreat_max_steps > 0
            ):
                self.record_failed_goal_viewpoint(
                    approach_station,
                    "planned_goal_arrival_scan_failed",
                )
                self.planned_goal_retreat_active = True
                self.planned_goal_retreat_steps = 0
                self.planned_goal_retreat_blocked_steps = 0
                self.planned_goal_retreat_scan_steps_taken = 0
                self.last_planned_goal_retreat_station = None
                self.stop_reason = "planned_goal_retreat_started"
                self.scenegraph.debug_stats.inc("planned_goal_retreat_started")
                if self.stop_verification_history:
                    self.stop_verification_history[-1][
                        "planned_goal_retreat_started"
                    ] = True
                return False, self.stop_verification_turn_action

            if not close_to_target:
                self.planned_goal_arrival_scan_steps_taken = 0
                if planned_station_kind == "retreat":
                    if (
                        self.planned_goal_retreat_steps
                        >= self.planned_goal_retreat_max_steps
                    ):
                        self.stop_reason = "planned_goal_retreat_rejected"
                        self.reject_confirmed_goal(reason="planned_goal_retreat_timeout")
                        self.reset_stop_verification_state(clear_history=False)
                        return False, self.stop_verification_turn_action

                    if planned_action is not None and planned_action != 0:
                        self.planned_goal_retreat_steps += 1
                        self.planned_goal_retreat_blocked_steps = 0
                        self.stop_verification_consecutive_failures = 0
                        self.stop_reason = "planned_goal_retreat"
                        self.scenegraph.debug_stats.inc("planned_goal_retreat")
                        return False, planned_action

                    self.planned_goal_retreat_blocked_steps += 1
                    self.scenegraph.debug_stats.inc("planned_goal_retreat_blocked")
                    if (
                        self.planned_goal_retreat_blocked_steps
                        >= max(1, self.stop_verification_steps)
                    ):
                        self.stop_reason = "planned_goal_retreat_blocked"
                        self.reject_confirmed_goal(
                            reason="planned_goal_retreat_blocked"
                        )
                        self.reset_stop_verification_state(clear_history=False)
                        return False, self.stop_verification_turn_action
                    return False, self.stop_verification_turn_action

                if (
                    self.planned_goal_approach_steps
                    >= self.planned_goal_approach_max_steps
                ):
                    self.stop_reason = "planned_goal_approach_rejected"
                    self.reject_confirmed_goal(reason="planned_goal_approach_timeout")
                    self.reset_stop_verification_state(clear_history=False)
                    return False, self.stop_verification_turn_action

                if planned_action is not None and planned_action != 0:
                    self.planned_goal_approach_steps += 1
                    self.planned_goal_approach_blocked_steps = 0
                    self.stop_verification_consecutive_failures = 0
                    self.stop_reason = "planned_goal_approach"
                    self.scenegraph.debug_stats.inc("planned_goal_approach")
                    return False, planned_action

                if planned_action is None:
                    self.planned_goal_approach_blocked_steps += 1
                    self.scenegraph.debug_stats.inc("planned_goal_approach_blocked")
                    if (
                        self.planned_goal_approach_blocked_steps
                        >= max(1, self.stop_verification_steps)
                    ):
                        self.stop_reason = "planned_goal_approach_blocked"
                        self.reject_confirmed_goal(
                            reason="planned_goal_approach_blocked"
                        )
                        self.reset_stop_verification_state(clear_history=False)
                        return False, self.stop_verification_turn_action

        approach_action = None
        if not self.planned_goal_approach_enabled:
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

        if (
            self.direct_goal_approach_enabled
            and not self.planned_goal_approach_enabled
            and close_to_target
            and final_detection_seen
            and self.stop_verification_hits >= max(0, self.stop_verification_min_hits - 1)
            and (
                not self.stop_verification_require_verification_hit
                or (
                    self.current_candidate_near_hit_count()
                    + getattr(self, "stop_verification_near_hits", 0)
                )
                > 0
            )
        ):
            enough_evidence = True

        if enough_evidence:
            close_enough = (
                (
                    self.planned_goal_approach_enabled
                    and close_to_target
                )
                or (
                    not self.planned_goal_approach_enabled
                    and (
                        not self.direct_goal_approach_enabled
                        or (final_detection_seen and close_to_target)
                    )
                )
            )
            if close_enough:
                confirm_reason = (
                    "stop_verification_high_confidence"
                    if high_confidence_current_hit
                    else "stop_verification_confirmed"
                )
                self.finalize_candidate("confirm", confirm_reason)
                self.stop_reason = confirm_reason
                self.scenegraph.debug_stats.inc("stop_verification_confirmed")
                if high_confidence_current_hit:
                    self.scenegraph.debug_stats.inc(
                        "stop_verification_high_confidence"
                    )
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
                if score < self.candidate_start_min_confidence:
                    self.scenegraph.debug_stats.inc(
                        "candidate_start_low_confidence_skipped"
                    )
                    self.scenegraph.debug_stats.inc(
                        "candidate_start_low_confidence_skipped_glip_bbox"
                    )
                    continue
                goal_detections.append({
                    "bbox": self.current_obj_predictions.bbox[j],
                    "score": score,
                    "label": str(label),
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
                    if confidence < self.candidate_start_min_confidence:
                        self.scenegraph.debug_stats.inc(
                            "candidate_start_low_confidence_skipped"
                        )
                        self.scenegraph.debug_stats.inc(
                            "candidate_start_low_confidence_skipped_groundedsam_mask"
                        )
                        continue
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
                    candidate_goal_gps, depth_capped_bound = (
                        self.bind_depth_capped_detection_to_active_candidate(
                            goal_gps,
                            temp_distance,
                        )
                    )
                    if self.is_rejected_goal_candidate(candidate_goal_gps):
                        continue
                    status = self.start_or_update_reperception_candidate(
                        goal_gps=candidate_goal_gps,
                        confidence=confidence,
                        source=(
                            "groundedsam_mask_depth_capped"
                            if depth_capped_bound
                            else "groundedsam_mask"
                        ),
                        det_distance_m=float(temp_distance),
                        detected_label=self.obj_goal,
                    )
                    if status in ("confirmed", "approach_confirmed"):
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
                    self.set_possible_goal_from_visual_evidence(
                        self.get_goal_gps(
                            observations, shortest_distance_angle, shortest_distance
                        ),
                        "groundedsam_mask_far_closest",
                        det_distance_m=float(shortest_distance),
                    )
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
                    candidate_goal_gps, depth_capped_bound = (
                        self.bind_depth_capped_detection_to_active_candidate(
                            goal_gps,
                            temp_distance,
                        )
                    )
                    if self.is_rejected_goal_candidate(candidate_goal_gps):
                        continue
                    self.mark_goal_candidate_map(candidate_goal_gps)
                    status = self.start_or_update_reperception_candidate(
                        goal_gps=candidate_goal_gps,
                        confidence=confidence,
                        source=(
                            "glip_bbox_depth_capped"
                            if depth_capped_bound
                            else "glip_bbox"
                        ),
                        det_distance_m=float(temp_distance),
                        detected_label=detection.get("label", self.obj_goal),
                    )
                    if status in ("confirmed", "approach_confirmed"):
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
                    self.set_possible_goal_from_visual_evidence(
                        self.get_goal_gps(
                            observations, shortest_distance_angle, shortest_distance
                        ),
                        "glip_bbox_far_closest",
                        det_distance_m=float(shortest_distance),
                    )
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
                    
        current_gps = np.asarray(observations["gps"], dtype=np.float32).reshape(-1)[:2]
        if np.linalg.norm(current_gps - self.last_gps) >= 0.05:
            self.move_steps += 1
            self.not_move_steps = 0
            self.current_stuck_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            self.current_stuck_steps += 1
            self.total_stuck_steps += 1
            
        if (
            self.invalid_episode_stationary_anchor_gps is None
            or not np.all(np.isfinite(self.invalid_episode_stationary_anchor_gps))
        ):
            self.invalid_episode_stationary_anchor_gps = current_gps.copy()
            self.invalid_episode_stationary_steps = 0
        else:
            stationary_anchor_dist_m = float(
                np.linalg.norm(
                    current_gps
                    - np.asarray(
                        self.invalid_episode_stationary_anchor_gps,
                        dtype=np.float32,
                    )
                )
            )
            if stationary_anchor_dist_m <= self.invalid_episode_stationary_radius_m:
                self.invalid_episode_stationary_steps += 1
            else:
                self.invalid_episode_stationary_anchor_gps = current_gps.copy()
                self.invalid_episode_stationary_steps = 0

        self.last_gps = current_gps.copy()
        consecutive_stuck_invalid = bool(
            self.invalid_episode_stuck_steps > 0
            and self.current_stuck_steps >= self.invalid_episode_stuck_steps
        )
        if consecutive_stuck_invalid:
            self.invalid_episode = True
            self.invalid_episode_reason = (
                "invalid_episode_consecutive_stuck_"
                f"{self.invalid_episode_stuck_steps}_steps"
            )
            self.stop_reason = self.invalid_episode_reason
            self.scenegraph.debug_stats.inc("invalid_episode_stuck")
            self.log_candidate_event(
                "invalid_episode",
                decision="discard",
                reason=self.invalid_episode_reason,
                current_stuck_steps=int(self.current_stuck_steps),
                total_stuck_steps=int(self.total_stuck_steps),
                invalid_episode_consecutive_stuck_steps=int(
                    self.current_stuck_steps
                ),
                invalid_episode_stationary_steps=int(
                    self.invalid_episode_stationary_steps
                ),
                invalid_episode_stationary_radius_m=float(
                    self.invalid_episode_stationary_radius_m
                ),
                invalid_episode_stationary_anchor_gps=np.asarray(
                    self.invalid_episode_stationary_anchor_gps,
                    dtype=np.float32,
                ).tolist(),
                invalid_episode_stuck_steps=int(self.invalid_episode_stuck_steps),
            )
            return {"action": 0}
        self.maybe_blacklist_stationary_position(observations)
        self.maybe_reject_active_candidate_for_stuck_or_no_progress(observations)
        
        self.scenegraph.perception()
          
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)

        candidate_view_scan_action = self.get_candidate_view_scan_action(
            traversible,
            cur_start,
            cur_start_o,
        )
        if candidate_view_scan_action is not None:
            self.not_use_random_goal()
            self.not_move_steps = 0
            if getattr(self, "reperception_goal_gps", None) is not None:
                if self.set_possible_goal_from_visual_evidence(
                    self.reperception_goal_gps,
                    "candidate_view_scan",
                    require_existing_visual_evidence=True,
                ):
                    self.goal_map = self.goal_gps_to_goal_map(
                        self.possible_goal_temp_gps
                    )
            self.stop_reason = "candidate_view_scan"
            if self.args.visualize or self.realtime_monitor:
                self.update_visualization_text(candidate_view_scan_action)
                self.visualize(traversible, observations, candidate_view_scan_action)
            observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(
                observations
            )
            self.last_loc = copy.deepcopy(self.full_pose)
            self.prev_action = candidate_view_scan_action
            self.navigate_steps += 1
            torch.cuda.empty_cache()
            return {"action": candidate_view_scan_action}
        
        if self.found_goal: 
            self.not_use_random_goal()
            self.goal_map = self.goal_gps_to_goal_map(self.goal_gps)
        elif self.found_possible_goal: 
            self.not_use_random_goal()
            self.goal_map = self.goal_gps_to_goal_map(self.possible_goal_temp_gps)
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
                self.use_frontier_goal(self.goal_loc, reason="fbe_initial")

        if (
            self.using_random_goal
            and not self.found_goal
            and not self.found_possible_goal
            and not self.reperception_active
        ):
            self.goal_loc = self.fbe(traversible, cur_start)
            if self.goal_loc is not None:
                self.use_frontier_goal(
                    self.goal_loc,
                    reason="random_preempted_by_frontier",
                )
                self.scenegraph.debug_stats.inc("random_preempted_by_frontier")
        
        goal_stop_distance_m = (
            self.candidate_stop_distance_m()
            if (self.found_goal or self.found_possible_goal)
            else None
        )
        planning_goal_found = bool(self.found_goal or self.found_possible_goal)

        # local policy
        stg_y, stg_x, replan, number_action = self._plan(
            traversible,
            self.goal_map,
            self.full_pose,
            cur_start,
            cur_start_o,
            planning_goal_found,
            stop_distance_m=goal_stop_distance_m,
        )
        if self.found_possible_goal and number_action == 0:
            self.scenegraph.debug_stats.inc("pending_candidate_arrival")
        
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
                self.use_frontier_goal(self.goal_loc, reason="fbe_replan")
            stg_y, stg_x, replan, number_action = self._plan(
                traversible,
                self.goal_map,
                self.full_pose,
                cur_start,
                cur_start_o,
                self.found_goal,
            )
        
        self.loop_time = 0
        while (
            (
                not self.found_goal
                and not self.found_possible_goal
                and not self.reperception_active
                and number_action == 0
            )
            or (
                self.not_move_steps >= 7
                and not (
                    self.found_possible_goal
                    or self.reperception_active
                    or self.stop_verification_active
                )
            )
        ):
            if self.not_move_steps >= 7:
                self.found_goal = False
                self.found_possible_goal = False
                self.reset_stop_verification_state(clear_history=False)
            self.loop_time += 1
            random_reason = (
                "agent_stuck_not_move_steps"
                if self.not_move_steps >= 7
                else "planner_stop_without_goal"
            )
            self.goal_loc = self.fbe(traversible, cur_start)
            if self.goal_loc is not None:
                self.use_frontier_goal(
                    self.goal_loc,
                    reason=f"{random_reason}_frontier_retry",
                )
                stg_y, stg_x, replan, number_action = self._plan(
                    traversible,
                    self.goal_map,
                    self.full_pose,
                    cur_start,
                    cur_start_o,
                    False,
                )
                if number_action != 0:
                    self.scenegraph.debug_stats.inc("random_fallback_avoided_by_frontier")
                    break
                self.clear_fbe_free_region(self.goal_loc, radius_cells=4)
                self.scenegraph.debug_stats.inc("frontier_planner_stop_rejected")
                if self.loop_time > 20:
                    self.stop_reason = "no_valid_plan_after_frontier_retries"
                    return {"action": 0}
                continue
            self.random_this_ex += 1
            self.record_random_goal(random_reason)
            if self.loop_time > 20:
                self.stop_reason = 'no_valid_plan_after_random_retries'
                return {"action": 0}
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal(reason=random_reason)
            self.using_random_goal = True
            stg_y, stg_x, replan, number_action = self._plan(
                traversible,
                self.goal_map,
                self.full_pose,
                cur_start,
                cur_start_o,
                self.found_goal,
            )
        
        verification_ran = False
        liveness_stop = self.apply_stop_liveness_guard()
        if liveness_stop:
            number_action = 0

        verification_should_run = (
            not liveness_stop
            and (
                (self.stop_verification_active and self.stop_verification_target_gps is not None)
                or (
                    number_action == 0
                    and (
                        self.found_goal
                        or self.found_possible_goal
                        or self.reperception_active
                    )
                )
            )
        )
        if verification_should_run:
            verification_ran = True
            verified_stop, verification_action = self.handle_stop_verification(
                observations,
                traversible=traversible,
                cur_start=cur_start,
                cur_start_o=cur_start_o,
            )
            if verified_stop:
                number_action = verification_action
            else:
                number_action = verification_action
                if verification_action in [2, 3]:
                    self.not_move_steps = 0
                if (
                    self.stop_verification_active
                    and self.stop_verification_target_gps is not None
                ):
                    target_gps = np.asarray(
                        self.stop_verification_target_gps, dtype=np.float32
                    ).copy()
                    if self.found_goal:
                        self.goal_gps = target_gps
                        self.found_possible_goal = False
                    else:
                        self.set_possible_goal_from_visual_evidence(
                            target_gps,
                            "stop_verification_target",
                            require_existing_visual_evidence=True,
                        )

        if number_action == 0:
            if self.stop_reason == "near_credible_goal_candidate":
                pass
            elif self.found_goal:
                if not (
                    self.stop_reason.startswith("stop_verification_confirmed")
                    or self.stop_reason.startswith(
                        "stop_verification_high_confidence"
                    )
                ):
                    self.stop_reason = 'planner_stop_after_found_goal'
            else:
                self.stop_reason = 'planner_stop_without_confirmed_goal'
        else:
            if not verification_ran:
                self.stop_reason = 'running'
        if self.args.visualize or self.realtime_monitor:
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

    def use_frontier_goal(self, goal_loc, reason="frontier"):
        if goal_loc is None:
            return False
        self.not_use_random_goal()
        self.fronter_this_ex += 1
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        r = int(np.clip(goal_loc[0], 0, self.map_size - 1))
        c = int(np.clip(goal_loc[1], 0, self.map_size - 1))
        self.goal_map[r, c] = 1
        self.goal_map = self.goal_map[::-1]
        self.scenegraph.debug_stats.inc("frontier_goal_selected")
        self.scenegraph.debug_stats.inc(
            "frontier_goal_selected_" + "".join(
                ch if ch.isalnum() or ch == "_" else "_"
                for ch in str(reason).lower()
            )
        )
        return True

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

    def _map_radius_cells(self, radius_m):
        return max(1, int(math.ceil(float(radius_m) * 100.0 / self.map_resolution)))

    def _region_bounds_from_rc(self, rc, radius_cells, shape, reverse_rows=False):
        rc = np.asarray(rc, dtype=np.float32).reshape(-1)[:2]
        if rc.size != 2 or not np.all(np.isfinite(rc)):
            return None
        row = int(round(float(rc[0])))
        col = int(round(float(rc[1])))
        height, width = int(shape[0]), int(shape[1])
        if reverse_rows:
            row = height - 1 - row
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        r0 = max(row - radius_cells, 0)
        r1 = min(row + radius_cells + 1, height)
        c0 = max(col - radius_cells, 0)
        c1 = min(col + radius_cells + 1, width)
        return r0, r1, c0, c1

    def clear_fbe_free_region(self, rc, radius_cells=4):
        if not hasattr(self, "fbe_free_map"):
            return False
        shape = self.fbe_free_map.shape[-2:]
        bounds = self._region_bounds_from_rc(
            rc,
            int(radius_cells),
            shape,
            reverse_rows=False,
        )
        if bounds is None:
            return False
        r0, r1, c0, c1 = bounds
        if torch.is_tensor(self.fbe_free_map) and self.fbe_free_map.dim() >= 4:
            self.fbe_free_map[0, 0, r0:r1, c0:c1] = 0
        else:
            self.fbe_free_map[r0:r1, c0:c1] = 0
        return True

    def apply_stuck_position_blacklist_to_fbe_free_map(self):
        if not getattr(self, "stuck_position_blacklist", []):
            return
        radius_cells = self._map_radius_cells(self.stuck_position_blacklist_radius_m)
        if torch.is_tensor(self.fbe_free_map):
            if self.fbe_free_map.dim() >= 4:
                shape = self.fbe_free_map.shape[-2:]
                target = self.fbe_free_map[0, 0]
            elif self.fbe_free_map.dim() == 2:
                shape = self.fbe_free_map.shape
                target = self.fbe_free_map
            else:
                return
            for item in self.stuck_position_blacklist:
                bounds = self._region_bounds_from_rc(
                    item.get("rc"), radius_cells, shape, reverse_rows=False
                )
                if bounds is None:
                    continue
                r0, r1, c0, c1 = bounds
                target[r0:r1, c0:c1] = 0
            return

        shape = self.fbe_free_map.shape[-2:]
        for item in self.stuck_position_blacklist:
            bounds = self._region_bounds_from_rc(
                item.get("rc"), radius_cells, shape, reverse_rows=False
            )
            if bounds is None:
                continue
            r0, r1, c0, c1 = bounds
            self.fbe_free_map[r0:r1, c0:c1] = 0

    def apply_stuck_position_blacklist_to_mask(self, mask, reverse_rows=False):
        if not getattr(self, "stuck_position_blacklist", []):
            return mask
        radius_cells = self._map_radius_cells(self.stuck_position_blacklist_radius_m)
        for item in self.stuck_position_blacklist:
            bounds = self._region_bounds_from_rc(
                item.get("rc"), radius_cells, mask.shape, reverse_rows=reverse_rows
            )
            if bounds is None:
                continue
            r0, r1, c0, c1 = bounds
            mask[r0:r1, c0:c1] = False
        return mask

    def add_stuck_position_blacklist(self, gps, reason):
        try:
            gps = np.asarray(gps, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            return False
        if gps.size != 2 or not np.all(np.isfinite(gps)):
            return False
        for item in getattr(self, "stuck_position_blacklist", []):
            item_gps = np.asarray(item.get("gps", []), dtype=np.float32).reshape(-1)[:2]
            if item_gps.size == 2 and np.all(np.isfinite(item_gps)):
                if float(np.linalg.norm(gps - item_gps)) <= self.stuck_position_blacklist_radius_m:
                    self.apply_stuck_position_blacklist_to_fbe_free_map()
                    return False

        rc = self.goal_gps_to_map_rc(gps)
        item = {
            "gps": gps.copy(),
            "rc": rc.astype(np.float32),
            "step": int(self.total_steps),
            "reason": str(reason),
            "radius_m": float(self.stuck_position_blacklist_radius_m),
        }
        self.stuck_position_blacklist.append(item)
        self.apply_stuck_position_blacklist_to_fbe_free_map()
        self.scenegraph.debug_stats.inc("stuck_position_blacklist_added")
        self.log_candidate_event(
            "stuck_position_blacklist",
            candidate_gps=gps,
            decision="blacklist",
            reason=reason,
            stuck_steps=int(self.current_stuck_steps),
            stuck_blacklist_radius_m=float(self.stuck_position_blacklist_radius_m),
            stuck_blacklist_rc=rc.tolist(),
        )
        return True

    def maybe_blacklist_stationary_position(self, observations):
        if self.current_stuck_steps < self.stuck_position_blacklist_steps:
            return False
        gps = observations.get("gps") if isinstance(observations, dict) else None
        if self.active_near_visual_hit_count() > 0:
            target_gps = self.active_candidate_target_gps()
            if target_gps is None:
                target_gps = getattr(self, "stop_verification_target_gps", None)
            if target_gps is not None and self.protect_near_hit_candidate_from_progress_reject(
                f"agent_stationary_{self.stuck_position_blacklist_steps}_steps",
                target_gps,
                self.distance_to_gps(target_gps, observations),
            ):
                self.current_stuck_steps = 0
                self.scenegraph.debug_stats.inc(
                    "stuck_position_blacklist_near_hit_protected"
                )
                return True
        added = self.add_stuck_position_blacklist(
            gps,
            f"agent_stationary_{self.stuck_position_blacklist_steps}_steps",
        )
        self.current_stuck_steps = 0
        self.not_move_steps = 0
        if added:
            self.found_goal = False
            self.found_possible_goal = False
            self.reperception_active = False
            self.goal_gps_map.fill(0)
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.first_fbe = True
            self.reset_stop_verification_state(clear_history=False)
            self.not_use_random_goal()
            self.scenegraph.debug_stats.inc("agent_stationary_position_blacklisted")
            self.scenegraph.debug_stats.inc("blacklist_force_frontier")
        return added
        
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
                "agent_rc": self.fbe_trace_logger._to_jsonable(start),
                "num_frontiers_all": 0,
                "num_frontiers_valid": 0,
                "frontier_count_all": 0,
                "frontier_count_valid": 0,
                "distances_16": [],
                "distance_inverse": [],
                "scenegraph_scores": [],
                "total_scores": [],
                "selected_valid_idx": None,
                "selected_frontier_rc": None,
                "selected_score": None,
                "top5_frontiers": [],
                "selected_goal_rc": None,
                "goal_map_rc": None,
                "fmm_dist_selected": None,
                "fmm_finite_ratio": None,
                "traversible_connected_component_size": None,
                "used_random_goal": True,
                "reason": "no_frontiers",
                "fallback_reason": "no_frontiers",
            }, traversible=traversible, start=start)
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        fmm_finite = np.isfinite(fmm_dist)
        fmm_finite_count = int(np.count_nonzero(fmm_finite))
        fmm_finite_ratio = float(fmm_finite_count / max(1, fmm_dist.size))
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        # Keep all reachable frontiers. Close frontiers should be allowed to win
        # when the scene graph score says they are promising.
        idx_16 = np.where(np.isfinite(distances))
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,1.6,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            self.fbe_frontier_count_valid_history.append(0)
            self.log_fbe_trace({
                "step": int(self.total_steps),
                "navigate_step": int(self.navigate_steps),
                "agent_rc": self.fbe_trace_logger._to_jsonable(start),
                "num_frontiers_all": int(num_frontiers),
                "num_frontiers_valid": 0,
                "frontier_count_all": int(num_frontiers),
                "frontier_count_valid": 0,
                "distances_16": [],
                "distance_inverse": [],
                "scenegraph_scores": [],
                "total_scores": [],
                "selected_valid_idx": None,
                "selected_frontier_rc": None,
                "selected_score": None,
                "top5_frontiers": [],
                "selected_goal_rc": None,
                "goal_map_rc": None,
                "fmm_dist_selected": None,
                "fmm_finite_ratio": fmm_finite_ratio,
                "traversible_connected_component_size": fmm_finite_count,
                "used_random_goal": True,
                "reason": "no_reachable_frontiers",
                "fallback_reason": "no_reachable_frontiers",
            }, traversible=traversible, start=start, frontier_locations_all_rc=frontier_locations - 1)
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        scenegraph_scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
        distance_tiebreaker = (
            self.frontier_distance_tiebreaker
            if self.frontier_distance_weight == 0.0
            else 0.0
        )
        distance_term = (
            self.frontier_distance_weight + distance_tiebreaker
        ) * distances_16_inverse
        scores = scenegraph_scores + distance_term
        selected_valid_idx = int(np.argmax(scores))
        distance_only_valid_idx = int(np.argmax(distances_16_inverse))
        scenegraph_only_valid_idx = int(np.argmax(scenegraph_scores))
        scenegraph_changed_selection = selected_valid_idx != distance_only_valid_idx
        idx_16_max = idx_16[0][selected_valid_idx]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        top5_frontiers = []
        for valid_idx in np.argsort(scores)[::-1][:5]:
            frontier_rc = frontier_locations_16[valid_idx] - 1
            fmm_rc = frontier_locations_16[valid_idx]
            top5_frontiers.append({
                "valid_idx": int(valid_idx),
                "frontier_rc": frontier_rc.tolist(),
                "score": float(scores[valid_idx]),
                "scenegraph_score": float(scenegraph_scores[valid_idx]),
                "distance_inverse": float(distances_16_inverse[valid_idx]),
                "distance_term": float(distance_term[valid_idx]),
                "fmm_dist": float(fmm_dist[fmm_rc[0], fmm_rc[1]]),
            })
        frontier_explanation = self.scenegraph.explain_frontier_selection(
            goal,
            self.obj_goal_sg,
            top_k=3,
        )
        self.last_frontier_explanation = frontier_explanation
        self.fbe_frontier_count_valid_history.append(int(num_16_frontiers))
        self.log_fbe_trace({
            "step": int(self.total_steps),
            "navigate_step": int(self.navigate_steps),
            "agent_rc": self.fbe_trace_logger._to_jsonable(start),
            "num_frontiers_all": int(num_frontiers),
            "num_frontiers_valid": int(num_16_frontiers),
            "frontier_count_all": int(num_frontiers),
            "frontier_count_valid": int(num_16_frontiers),
            "distances_16": distances_16,
            "distance_inverse": distances_16_inverse,
            "scenegraph_scores": scenegraph_scores,
            "distance_term": distance_term,
            "total_scores": scores,
            "selected_valid_idx": selected_valid_idx,
            "distance_only_valid_idx": distance_only_valid_idx,
            "scenegraph_only_valid_idx": scenegraph_only_valid_idx,
            "scenegraph_changed_selection": bool(scenegraph_changed_selection),
            "selected_frontier_rc": goal,
            "selected_score": float(scores[selected_valid_idx]),
            "selected_scenegraph_score": float(scenegraph_scores[selected_valid_idx]),
            "selected_distance_inverse": float(distances_16_inverse[selected_valid_idx]),
            "distance_only_frontier_rc": (
                frontier_locations_16[distance_only_valid_idx] - 1
            ).tolist(),
            "distance_only_score": float(scores[distance_only_valid_idx]),
            "distance_only_scenegraph_score": float(
                scenegraph_scores[distance_only_valid_idx]
            ),
            "distance_only_distance_inverse": float(
                distances_16_inverse[distance_only_valid_idx]
            ),
            "scenegraph_only_frontier_rc": (
                frontier_locations_16[scenegraph_only_valid_idx] - 1
            ).tolist(),
            "scenegraph_only_score": float(scores[scenegraph_only_valid_idx]),
            "scenegraph_only_scenegraph_score": float(
                scenegraph_scores[scenegraph_only_valid_idx]
            ),
            "scenegraph_only_distance_inverse": float(
                distances_16_inverse[scenegraph_only_valid_idx]
            ),
            "top5_frontiers": top5_frontiers,
            "frontier_explanation": frontier_explanation,
            "selected_goal_rc": goal,
            "goal_map_rc": goal,
            "fmm_dist_selected": float(fmm_dist[frontier_locations[idx_16_max][0], frontier_locations[idx_16_max][1]]),
            "fmm_finite_ratio": fmm_finite_ratio,
            "traversible_connected_component_size": fmm_finite_count,
            "used_random_goal": False,
            "reason": "selected_frontier",
            "fallback_reason": None,
            "score_mode": getattr(self.args, "sgnav_score_mode", "group"),
            "frontier_score_norm": self.frontier_score_norm,
            "frontier_distance_weight": float(self.frontier_distance_weight),
            "frontier_distance_tiebreaker": float(self.frontier_distance_tiebreaker),
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
        self.apply_stuck_position_blacklist_to_fbe_free_map()
    
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
        grid = np.rint(map_pred).astype(np.float32, copy=True)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)
        clearance = int(self.agent_footprint_clearance_cells)

        def set_local_patch(mat, center, radius, value):
            center_r, center_c = int(center[0]), int(center[1])
            r0 = max(0, center_r - radius)
            r1 = min(mat.shape[0], center_r + radius + 1)
            c0 = max(0, center_c - radius)
            c1 = min(mat.shape[1], center_c + radius + 1)
            if r0 < r1 and c0 < c1:
                mat[r0:r1, c0:c1] = value

        visited_window = self.visited[gy1:gy2, gx1:gx2]
        collision_window = self.collision_map[gy1:gy2, gx1:gx2]
        start_was_blocked = bool(grid[start[0], start[1]] != 0)
        collision_on_footprint = bool(
            np.any(
                collision_window[
                    max(0, start[0] - clearance):min(collision_window.shape[0], start[0] + clearance + 1),
                    max(0, start[1] - clearance):min(collision_window.shape[1], start[1] + clearance + 1),
                ]
                == 1
            )
        )
        set_local_patch(visited_window, start, clearance, 1)
        set_local_patch(grid, start, clearance, 0)
        set_local_patch(collision_window, start, clearance, 0)
        if start_was_blocked or collision_on_footprint:
            self.traversible_start_corrections += 1
            if hasattr(self, "scenegraph"):
                self.scenegraph.debug_stats.inc("agent_footprint_cleared")
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

        traversible = 1 - traversible
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem)
        traversible[collision_window[y1:y2, x1:x2] == 1] = 1
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        set_local_patch(
            traversible,
            [int(start[0] - y1), int(start[1] - x1)],
            clearance,
            1,
        )
        traversible = traversible * 1.
        
        traversible[visited_window[y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(
        self,
        traversible,
        goal_map,
        agent_pose,
        start,
        start_o,
        goal_found,
        stop_distance_m=None,
    ):
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

        stg, replan, stop, = self._get_stg(
            traversible,
            start,
            np.copy(goal_map),
            goal_found,
            stop_distance_m=stop_distance_m,
        )

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
    
    def _get_stg(
        self,
        traversible,
        start,
        goal,
        goal_found,
        stop_distance_m=None,
    ):
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
            target_stop_distance_m = (
                self.found_goal_stop_distance_m
                if stop_distance_m is None
                else max(0.05, float(stop_distance_m))
            )
            decrease_stop_cond = max(
                decrease_stop_cond,
                self.planner.stop_cond - target_stop_distance_m,
            )
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop
    
    def set_random_goal(self, base_goal=None, reason="unspecified"):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        available_goal_mask = obstacle_map < 1
        available_goal_mask = self.apply_stuck_position_blacklist_to_mask(
            available_goal_mask,
            reverse_rows=True,
        )
        goal_index = np.where(available_goal_mask)
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
            "git_commit": get_git_commit_hash_or_unknown(),
            "argv": list(sys.argv),
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
            "invalid_episode": bool(getattr(self, "invalid_episode", False)),
            "invalid_episode_reason": str(
                getattr(self, "invalid_episode_reason", "")
            ),
            "invalid_episode_stuck_steps": int(self.invalid_episode_stuck_steps),
            "invalid_episode_stationary_steps": int(
                self.invalid_episode_stationary_steps
            ),
            "invalid_episode_stationary_radius_m": float(
                self.invalid_episode_stationary_radius_m
            ),
            "invalid_episode_stationary_anchor_gps": (
                np.asarray(
                    self.invalid_episode_stationary_anchor_gps,
                    dtype=np.float32,
                ).tolist()
                if self.invalid_episode_stationary_anchor_gps is not None
                else None
            ),
            "frontier_calls": int(self.fronter_this_ex),
            "random_goal_count": int(self.random_this_ex),
            "random_goal_reasons": self.random_goal_reasons[-50:],
            "current_stuck_steps": int(self.current_stuck_steps),
            "total_stuck_steps": int(self.total_stuck_steps),
            "stuck_position_blacklist_steps": int(
                self.stuck_position_blacklist_steps
            ),
            "stuck_position_blacklist_radius_m": float(
                self.stuck_position_blacklist_radius_m
            ),
            "candidate_stuck_blacklist_steps": int(
                self.candidate_stuck_blacklist_steps
            ),
            "candidate_strong_evidence_min_hits": int(
                self.candidate_strong_evidence_min_hits
            ),
            "candidate_strong_evidence_min_consecutive_hits": int(
                self.candidate_strong_evidence_min_consecutive_hits
            ),
            "candidate_no_progress_blacklist_steps": int(
                self.candidate_no_progress_blacklist_steps
            ),
            "candidate_progress_min_delta_m": float(
                self.candidate_progress_min_delta_m
            ),
            "candidate_progress_best_distance_m": float(
                self.candidate_progress_best_distance_m
            ),
            "candidate_progress_last_distance_m": float(
                self.candidate_progress_last_distance_m
            ),
            "candidate_progress_no_improve_steps": int(
                self.candidate_progress_no_improve_steps
            ),
            "stuck_position_blacklist": [
                {
                    **item,
                    "gps": np.asarray(item.get("gps", [])).tolist(),
                    "rc": np.asarray(item.get("rc", [])).tolist(),
                }
                for item in getattr(self, "stuck_position_blacklist", [])
            ],
            "fbe_frontier_count_valid_history": self.fbe_frontier_count_valid_history,
            "fbe_frontier_count_valid_summary": self.summarize_fbe_valid_counts(),
            "last_frontier_explanation": self.last_frontier_explanation,
            "traversible_start_corrections": int(
                self.traversible_start_corrections
            ),
            "agent_footprint_clearance_cells": int(
                self.agent_footprint_clearance_cells
            ),
            "sgnav_score_mode": getattr(self.args, "sgnav_score_mode", "group"),
            "frontier_score_norm": self.frontier_score_norm,
            "frontier_distance_weight": float(self.frontier_distance_weight),
            "frontier_distance_tiebreaker": float(self.frontier_distance_tiebreaker),
            "paper_reperception_mode": bool(self.paper_reperception_mode),
            "disable_extra_stop_verification": bool(self.disable_extra_stop_verification),
            "reperception_score_norm": self.reperception_score_norm,
            "goal_detection_min_confidence": float(
                self.goal_detection_min_confidence
            ),
            "candidate_start_min_confidence": float(
                self.candidate_start_min_confidence
            ),
            "vllm_model": os.environ.get("VLLM_MODEL"),
            "vllm_llm_model": os.environ.get("VLLM_LLM_MODEL"),
            "vllm_vlm_model": os.environ.get("VLLM_VLM_MODEL"),
            "nodes_final": len(self.scenegraph.get_nodes()),
            "edges_final": len(edges),
            "room_nodes_active": sum(
                1 for room_node in self.scenegraph.room_nodes if getattr(room_node, "active", False)
            ),
            "room_nodes_with_objects": sum(
                1 for room_node in self.scenegraph.room_nodes if len(room_node.nodes) > 0
            ),
            "room_nodes_with_groups": sum(
                1 for room_node in self.scenegraph.room_nodes if len(room_node.group_nodes) > 0
            ),
            "group_nodes_total": len(getattr(self.scenegraph, "group_nodes", [])),
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
            "stop_verification_force_stop_confidence": float(
                self.stop_verification_force_stop_confidence
            ),
            "stop_verification_require_verification_hit": bool(
                self.stop_verification_require_verification_hit
            ),
            "stop_require_near_visual_hit": bool(
                self.stop_require_near_visual_hit
            ),
            "stop_verification_required_hit_max_distance_m": float(
                self.stop_verification_required_hit_max_distance_m
            ),
            "stop_verification_anchor_radius_m": float(
                self.stop_verification_anchor_radius_m
            ),
            "planned_goal_arrival_scan_steps": int(
                self.planned_goal_arrival_scan_steps
            ),
            "planned_goal_verification_max_observations": int(
                self.planned_goal_verification_max_observations
            ),
            "direct_goal_approach_enabled": bool(self.direct_goal_approach_enabled),
            "direct_goal_approach_min_distance_m": float(
                self.direct_goal_approach_min_distance_m
            ),
            "direct_goal_approach_steps": int(self.direct_goal_approach_steps),
            "direct_goal_approach_max_steps": int(self.direct_goal_approach_max_steps),
            "planned_goal_approach_enabled": bool(self.planned_goal_approach_enabled),
            "planned_goal_stop_distance_m": float(self.planned_goal_stop_distance_m),
            "planned_goal_approach_min_radius_m": float(
                self.planned_goal_approach_min_radius_m
            ),
            "planned_goal_approach_max_radius_m": float(
                self.planned_goal_approach_max_radius_m
            ),
            "planned_goal_approach_station_stop_distance_m": float(
                self.planned_goal_approach_station_stop_distance_m
            ),
            "planned_goal_approach_radius_cost": float(
                self.planned_goal_approach_radius_cost
            ),
            "planned_goal_retreat_enabled": bool(self.planned_goal_retreat_enabled),
            "planned_goal_retreat_min_radius_m": float(
                self.planned_goal_retreat_min_radius_m
            ),
            "planned_goal_retreat_max_radius_m": float(
                self.planned_goal_retreat_max_radius_m
            ),
            "planned_goal_retreat_require_line_of_sight": bool(
                self.planned_goal_retreat_require_line_of_sight
            ),
            "planned_goal_retreat_los_endpoint_skip_cells": int(
                self.planned_goal_retreat_los_endpoint_skip_cells
            ),
            "planned_goal_retreat_station_stop_distance_m": float(
                self.planned_goal_retreat_station_stop_distance_m
            ),
            "planned_goal_retreat_scan_steps": int(
                self.planned_goal_retreat_scan_steps
            ),
            "planned_goal_retreat_scan_steps_taken": int(
                self.planned_goal_retreat_scan_steps_taken
            ),
            "planned_goal_retreat_steps": int(self.planned_goal_retreat_steps),
            "planned_goal_retreat_blocked_steps": int(
                self.planned_goal_retreat_blocked_steps
            ),
            "planned_goal_retreat_max_steps": int(
                self.planned_goal_retreat_max_steps
            ),
            "planned_goal_viewpoint_max_attempts": int(
                self.planned_goal_viewpoint_max_attempts
            ),
            "planned_goal_retreat_viewpoint_attempts": int(
                self.planned_goal_retreat_viewpoint_attempts
            ),
            "planned_goal_retreat_failure_count": int(
                self.planned_goal_retreat_failure_count()
            ),
            "planned_goal_viewpoint_min_separation_m": float(
                self.planned_goal_viewpoint_min_separation_m
            ),
            "planned_goal_failed_viewpoints": list(
                getattr(self, "planned_goal_failed_viewpoints", [])
            ),
            "planned_goal_retreat_active": bool(self.planned_goal_retreat_active),
            "planned_goal_approach_steps": int(self.planned_goal_approach_steps),
            "planned_goal_approach_max_steps": int(
                self.planned_goal_approach_max_steps
            ),
            "planned_goal_approach_blocked_steps": int(
                self.planned_goal_approach_blocked_steps
            ),
            "last_planned_goal_approach_station": self.last_planned_goal_approach_station,
            "last_planned_goal_retreat_station": self.last_planned_goal_retreat_station,
            "stop_verification_consecutive_failures": int(
                self.stop_verification_consecutive_failures
            ),
            "stop_verification_history": self.stop_verification_history[-10:],
            "stop_liveness_decision": self.last_stop_liveness_decision,
            "reperception_rejected_count": len(self.rejected_goal_candidates),
            "reperception_history": self.reperception_history[-10:],
            "candidate_summary": self.summarize_candidate_metrics(),
            "candidate_view_scan_max_steps": int(self.candidate_view_scan_max_steps),
            "candidate_require_direct_goal_for_confirm": bool(
                self.candidate_require_direct_goal_for_confirm
            ),
            "candidate_single_hit_search_enabled": bool(
                self.candidate_single_hit_search_enabled
            ),
            "candidate_multiview_enabled": bool(self.candidate_multiview_enabled),
            "candidate_multiview_min_radius_m": float(
                self.candidate_multiview_min_radius_m
            ),
            "candidate_multiview_max_radius_m": float(
                self.candidate_multiview_max_radius_m
            ),
            "candidate_multiview_station_stop_distance_m": float(
                self.candidate_multiview_station_stop_distance_m
            ),
            "candidate_multiview_min_start_distance_m": float(
                self.candidate_multiview_min_start_distance_m
            ),
            "candidate_current": self.candidate_snapshot(),
            "candidate_history": self.candidate_summaries[-10:],
            "scenegraph_score_debug": getattr(self.scenegraph, "last_score_debug", {}),
            "llm_parse_failures": self.scenegraph.debug_stats.summary(),
        }
        row.update(row["candidate_summary"])
        self.episode_logger.log(row)

    def summarize_candidate_metrics(self):
        summaries = list(self.candidate_summaries)
        current = self.candidate_snapshot()
        if current and current.get("decision") == "pending":
            summaries.append(current)
        if not summaries:
            return {
                "candidate_started": 0,
                "candidate_confirmed": 0,
                "candidate_rejected": 0,
                "candidate_blacklisted": int(
                    self.scenegraph.debug_stats.summary().get("candidate_blacklisted", 0)
                ),
                "candidate_false_positive_like": 0,
                "mean_candidate_hit_ratio": 0.0,
                "mean_candidate_miss_count": 0.0,
                "graph_only_confirmations": 0,
                "confirmations_without_detector_hits": 0,
                "top_contribution_non_goal_count": 0,
                "top_contribution_direct_goal_count": 0,
            }
        confirmed = [item for item in summaries if item.get("decision") == "confirm"]
        rejected = [item for item in summaries if item.get("decision") == "reject"]
        graph_only = [
            item for item in confirmed
            if item.get("hit_count", 0) < self.candidate_min_detector_hits
        ]
        false_positive_like = [
            item for item in confirmed
            if (
                item.get("hit_count", 0) < self.candidate_min_detector_hits
            )
        ]
        return {
            "candidate_started": int(
                self.scenegraph.debug_stats.summary().get("candidate_started", len(summaries))
            ),
            "candidate_confirmed": int(len(confirmed)),
            "candidate_rejected": int(len(rejected)),
            "candidate_blacklisted": int(
                self.scenegraph.debug_stats.summary().get("candidate_blacklisted", 0)
            ),
            "candidate_false_positive_like": int(len(false_positive_like)),
            "mean_candidate_hit_ratio": float(
                np.mean([item.get("hit_ratio", 0.0) for item in summaries])
            ),
            "mean_candidate_miss_count": float(
                np.mean([item.get("miss_count", 0) for item in summaries])
            ),
            "graph_only_confirmations": int(len(graph_only)),
            "confirmations_without_detector_hits": int(len(graph_only)),
            "top_contribution_non_goal_count": int(
                sum(item.get("top_contribution_non_goal_count", 0) for item in summaries)
            ),
            "top_contribution_direct_goal_count": int(
                sum(item.get("top_contribution_direct_goal_count", 0) for item in summaries)
            ),
        }

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
        frontier_explanation = self.last_frontier_explanation.get("explanation", "")
        if frontier_explanation:
            self.explanation += f" Frontier reasoning: {frontier_explanation}"

    def visualize(self, traversible, observations, number_action):
        if self.args.visualize or self.realtime_monitor:
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
            if self.args.visualize:
                self.visualize_image_list.append(visualize_image)
            if (
                self.realtime_monitor
                and self.total_steps % self.realtime_monitor_every == 0
            ):
                self.write_realtime_monitor_frame(visualize_image)

    def write_realtime_monitor_frame(self, visualize_image):
        os.makedirs(self.realtime_monitor_dir, exist_ok=True)
        episode_path = os.path.join(
            self.realtime_monitor_dir,
            f"latest_ep_{self.count_episodes:06d}.jpg",
        )
        tmp_latest_path = self.realtime_monitor_latest_path + ".tmp.jpg"
        tmp_episode_path = episode_path + ".tmp.jpg"
        cv2.imwrite(tmp_latest_path, visualize_image)
        cv2.imwrite(tmp_episode_path, visualize_image)
        os.replace(tmp_latest_path, self.realtime_monitor_latest_path)
        os.replace(tmp_episode_path, episode_path)

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
        "--realtime_monitor", action="store_true"
    )
    parser.add_argument(
        "--realtime_monitor_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--realtime_monitor_every",
        default=1,
        type=int,
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
        "--frontier_score_norm",
        default="paper_sum",
        choices=["paper_sum", "weighted_mean"],
    )
    parser.add_argument(
        "--frontier_distance_weight", default=None, type=float
    )
    parser.add_argument(
        "--frontier_distance_tiebreaker", default=1e-6, type=float
    )
    parser.add_argument(
        "--paper_room_map_active_threshold", default=0.05, type=float
    )
    parser.add_argument(
        "--paper_room_min_membership_score", default=0.01, type=float
    )
    parser.add_argument(
        "--paper_room_point_sample_limit", default=512, type=int
    )
    parser.add_argument(
        "--object_duplicate_merge_enabled", default=1, type=int
    )
    parser.add_argument(
        "--object_duplicate_merge_center_m", default=0.80, type=float
    )
    parser.add_argument(
        "--object_duplicate_merge_strong_center_m", default=0.35, type=float
    )
    parser.add_argument(
        "--object_duplicate_merge_point_distance_m", default=0.08, type=float
    )
    parser.add_argument(
        "--object_duplicate_merge_point_overlap", default=0.12, type=float
    )
    parser.add_argument(
        "--object_duplicate_merge_bbox_iou", default=0.05, type=float
    )
    parser.add_argument(
        "--object_duplicate_merge_bbox_containment", default=0.55, type=float
    )
    parser.add_argument(
        "--object_duplicate_merge_max_passes", default=3, type=int
    )
    parser.add_argument(
        "--reperception_score_norm",
        default="weighted_mean",
        choices=["paper_sum", "sum", "weighted_mean"],
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
        "--rejected_goal_visible_blacklist_max_distance_m", default=2.0, type=float
    )
    parser.add_argument(
        "--stuck_position_blacklist_steps", default=50, type=int
    )
    parser.add_argument(
        "--stuck_position_blacklist_radius_m", default=1.0, type=float
    )
    parser.add_argument(
        "--candidate_stuck_blacklist_steps", default=7, type=int
    )
    parser.add_argument(
        "--candidate_no_progress_blacklist_steps", default=20, type=int
    )
    parser.add_argument(
        "--candidate_progress_min_delta_m", default=0.10, type=float
    )
    parser.add_argument(
        "--invalid_episode_stuck_steps", default=200, type=int
    )
    parser.add_argument(
        "--invalid_episode_stationary_radius_m", default=0.50, type=float
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
        "--stop_verification_force_stop_confidence", default=0.85, type=float
    )
    parser.add_argument(
        "--stop_require_near_visual_hit", default=1, type=int
    )
    parser.add_argument(
        "--stop_verification_require_verification_hit", default=0, type=int
    )
    parser.add_argument(
        "--stop_verification_required_hit_max_distance_m", default=1.5, type=float
    )
    parser.add_argument(
        "--stop_verification_anchor_radius_m", default=0.8, type=float
    )
    parser.add_argument(
        "--stop_verification_turn_action", default=3, type=int
    )
    parser.add_argument(
        "--planned_goal_arrival_scan_steps", default=8, type=int
    )
    parser.add_argument(
        "--planned_goal_verification_max_observations", default=24, type=int
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
        "--goal_detection_min_confidence", default=0.60, type=float
    )
    parser.add_argument(
        "--candidate_start_min_confidence", default=0.60, type=float
    )
    parser.add_argument(
        "--planned_goal_approach_enabled", default=1, type=int
    )
    parser.add_argument(
        "--planned_goal_stop_distance_m", default=0.70, type=float
    )
    parser.add_argument(
        "--planned_goal_approach_max_steps", default=40, type=int
    )
    parser.add_argument(
        "--planned_goal_approach_min_radius_m", default=None, type=float
    )
    parser.add_argument(
        "--planned_goal_approach_max_radius_m", default=None, type=float
    )
    parser.add_argument(
        "--planned_goal_approach_station_stop_distance_m", default=0.10, type=float
    )
    parser.add_argument(
        "--planned_goal_approach_radius_cost", default=2.0, type=float
    )
    parser.add_argument(
        "--planned_goal_retreat_enabled", default=1, type=int
    )
    parser.add_argument(
        "--planned_goal_retreat_min_radius_m", default=1.20, type=float
    )
    parser.add_argument(
        "--planned_goal_retreat_max_radius_m", default=1.50, type=float
    )
    parser.add_argument(
        "--planned_goal_retreat_require_line_of_sight", default=1, type=int
    )
    parser.add_argument(
        "--planned_goal_retreat_los_endpoint_skip_cells", default=2, type=int
    )
    parser.add_argument(
        "--planned_goal_retreat_station_stop_distance_m", default=0.25, type=float
    )
    parser.add_argument(
        "--planned_goal_retreat_scan_steps", default=6, type=int
    )
    parser.add_argument(
        "--planned_goal_retreat_max_steps", default=15, type=int
    )
    parser.add_argument(
        "--planned_goal_viewpoint_max_attempts", default=2, type=int
    )
    parser.add_argument(
        "--planned_goal_retreat_viewpoint_attempts", default=2, type=int
    )
    parser.add_argument(
        "--planned_goal_viewpoint_min_separation_m", default=0.45, type=float
    )
    parser.add_argument(
        "--agent_footprint_clearance_cells", default=2, type=int
    )
    parser.add_argument(
        "--candidate_min_detector_hits", type=int, default=2
    )
    parser.add_argument(
        "--candidate_strong_evidence_min_hits", type=int, default=None
    )
    parser.add_argument(
        "--candidate_strong_evidence_min_consecutive_hits", type=int, default=6
    )
    parser.add_argument(
        "--candidate_min_distinct_views", type=int, default=1
    )
    parser.add_argument(
        "--candidate_min_hit_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_max_misses", type=int, default=6
    )
    parser.add_argument(
        "--candidate_miss_penalty", type=float, default=0.20
    )
    parser.add_argument(
        "--candidate_score_decay", type=float, default=0.85
    )
    parser.add_argument(
        "--candidate_context_cap", type=float, default=0.65
    )
    parser.add_argument(
        "--candidate_direct_match_bonus", type=float, default=0.25
    )
    parser.add_argument(
        "--candidate_require_detector_for_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate_require_direct_goal_for_confirm", type=int, default=1
    )
    parser.add_argument(
        "--candidate_view_scan_max_steps", type=int, default=8
    )
    parser.add_argument(
        "--candidate_single_hit_search_enabled", type=int, default=1
    )
    parser.add_argument(
        "--candidate_multiview_enabled", type=int, default=1
    )
    parser.add_argument(
        "--candidate_multiview_min_radius_m", type=float, default=0.70
    )
    parser.add_argument(
        "--candidate_multiview_max_radius_m", type=float, default=1.10
    )
    parser.add_argument(
        "--candidate_multiview_station_stop_distance_m", type=float, default=0.15
    )
    parser.add_argument(
        "--candidate_multiview_min_start_distance_m", type=float, default=0.35
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
