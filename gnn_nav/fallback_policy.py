from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FallbackDecision:
    use_fallback: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FallbackPolicyConfig:
    max_prob_threshold: float = 0.45
    margin_threshold: float = 0.10
    entropy_threshold: float = 1.50
    min_object_nodes: int = 1
    stuck_steps_threshold: int = 8
    rare_goal_categories: Sequence[str] = field(default_factory=list)


def frontier_uncertainty(scores) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return {"max_prob": 0.0, "margin": 0.0, "entropy": 0.0, "num_frontiers": 0.0}
    logits = scores - float(np.max(scores))
    probs = np.exp(logits)
    probs = probs / max(float(probs.sum()), 1e-8)
    order = np.argsort(-probs)
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    entropy = float(-(probs * np.log(probs + 1e-8)).sum())
    return {
        "max_prob": top1,
        "margin": top1 - top2,
        "entropy": entropy,
        "num_frontiers": float(scores.size),
    }


def zscore(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    return (values - float(values.mean())) / (float(values.std()) + 1e-6)


def combine_scores(gnn_scores, fallback_scores, alpha: float = 1.0) -> np.ndarray:
    gnn_scores = np.asarray(gnn_scores, dtype=np.float32).reshape(-1)
    fallback_scores = np.asarray(fallback_scores, dtype=np.float32).reshape(-1)
    if gnn_scores.shape != fallback_scores.shape:
        raise ValueError(f"score shape mismatch: gnn={gnn_scores.shape}, fallback={fallback_scores.shape}")
    return zscore(gnn_scores) + float(alpha) * zscore(fallback_scores)


class GNNFallbackPolicy:
    """Uncertainty rules for future VLM/LLM fallback integration.

    This module only decides whether fallback should be considered. The actual
    fallback query and action selection remain outside this class so original
    SG-Nav behavior stays unchanged until explicit integration.
    """

    def __init__(self, config: Optional[FallbackPolicyConfig] = None):
        self.config = config or FallbackPolicyConfig()

    def evaluate(
        self,
        scores,
        *,
        num_object_nodes: int = 0,
        goal_text: str = "",
        stuck_steps: int = 0,
        candidate_credibility: Optional[float] = None,
    ) -> FallbackDecision:
        metrics = frontier_uncertainty(scores)
        reasons = []
        if metrics["num_frontiers"] == 0:
            reasons.append("no_frontiers")
        if metrics["max_prob"] < self.config.max_prob_threshold:
            reasons.append("low_max_probability")
        if metrics["margin"] < self.config.margin_threshold and metrics["num_frontiers"] > 1:
            reasons.append("low_top1_top2_margin")
        if metrics["entropy"] > self.config.entropy_threshold:
            reasons.append("high_entropy")
        if int(num_object_nodes) < self.config.min_object_nodes:
            reasons.append("sparse_object_context")
        if int(stuck_steps) >= self.config.stuck_steps_threshold:
            reasons.append("agent_stuck")
        if str(goal_text).lower().strip() in {str(x).lower().strip() for x in self.config.rare_goal_categories}:
            reasons.append("rare_goal_category")
        if candidate_credibility is not None:
            metrics["candidate_credibility"] = float(candidate_credibility)
            if float(candidate_credibility) < 0.5:
                reasons.append("low_candidate_credibility")
        return FallbackDecision(use_fallback=bool(reasons), reasons=reasons, metrics=metrics)
