from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


NODE_OBJECT = "object"
NODE_ROOM = "room"
NODE_FRONTIER = "frontier"
NODE_GOAL = "goal"
NODE_CANDIDATE = "candidate_goal"

EdgeType = Tuple[str, str, str]

EDGE_OBJECT_OBJECT = (NODE_OBJECT, "spatial", NODE_OBJECT)
EDGE_OBJECT_ROOM = (NODE_OBJECT, "in", NODE_ROOM)
EDGE_ROOM_OBJECT = (NODE_ROOM, "contains", NODE_OBJECT)
EDGE_FRONTIER_OBJECT = (NODE_FRONTIER, "near", NODE_OBJECT)
EDGE_OBJECT_FRONTIER = (NODE_OBJECT, "near_by", NODE_FRONTIER)
EDGE_FRONTIER_ROOM = (NODE_FRONTIER, "near", NODE_ROOM)
EDGE_ROOM_FRONTIER = (NODE_ROOM, "near_by", NODE_FRONTIER)
EDGE_GOAL_OBJECT = (NODE_GOAL, "query_object", NODE_OBJECT)
EDGE_OBJECT_GOAL = (NODE_OBJECT, "queried_by", NODE_GOAL)
EDGE_GOAL_ROOM = (NODE_GOAL, "query_room", NODE_ROOM)
EDGE_ROOM_GOAL = (NODE_ROOM, "queried_by", NODE_GOAL)
EDGE_GOAL_FRONTIER = (NODE_GOAL, "query_frontier", NODE_FRONTIER)
EDGE_FRONTIER_GOAL = (NODE_FRONTIER, "queried_by", NODE_GOAL)

EDGE_CANDIDATE_OBJECT = (NODE_CANDIDATE, "context", NODE_OBJECT)
EDGE_OBJECT_CANDIDATE = (NODE_OBJECT, "context_of", NODE_CANDIDATE)
EDGE_CANDIDATE_ROOM = (NODE_CANDIDATE, "in", NODE_ROOM)
EDGE_ROOM_CANDIDATE = (NODE_ROOM, "contains_candidate", NODE_CANDIDATE)
EDGE_CANDIDATE_GOAL = (NODE_CANDIDATE, "matches", NODE_GOAL)
EDGE_GOAL_CANDIDATE = (NODE_GOAL, "query_candidate", NODE_CANDIDATE)


@dataclass
class SparseDecisionGraph:
    node_features: Dict[str, torch.Tensor]
    edge_index: Dict[EdgeType, torch.Tensor]
    edge_attr: Dict[EdgeType, torch.Tensor]
    node_texts: Dict[str, List[str]] = field(default_factory=dict)
    frontier_centers_rc: Optional[torch.Tensor] = None
    candidate_metadata: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device):
        self.node_features = {k: v.to(device) for k, v in self.node_features.items()}
        self.edge_index = {k: v.to(device) for k, v in self.edge_index.items()}
        self.edge_attr = {k: v.to(device) for k, v in self.edge_attr.items()}
        if self.frontier_centers_rc is not None:
            self.frontier_centers_rc = self.frontier_centers_rc.to(device)
        return self
