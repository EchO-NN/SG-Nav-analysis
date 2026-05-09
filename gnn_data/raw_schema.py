import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


RAW_VERSION = "raw_sgnav_step_v1"
LABEL_VERSION = "labeled_sgnav_step_v1"


def sanitize_filename(value: Any, max_len: int = 120) -> str:
    text = str(value)
    text = re.sub(r"[^a-zA-Z0-9_.-]", "_", text)
    return text[:max_len] or "unknown"


def as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_cpu(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_cpu(v) for v in value]
    return value


def as_float_array(value, default_shape=None) -> np.ndarray:
    if value is None:
        if default_shape is None:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros(default_shape, dtype=np.float32)
    arr = as_numpy(value)
    return np.asarray(arr, dtype=np.float32)


def make_soft_frontier_label(costs, tau: float = 2.0) -> np.ndarray:
    costs = np.asarray(costs, dtype=np.float32)
    y = np.zeros_like(costs, dtype=np.float32)
    if costs.size == 0:
        return y

    valid = np.isfinite(costs)
    if valid.sum() == 0:
        y[:] = 1.0 / float(len(y))
        return y

    logits = -costs[valid] / float(tau)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / max(float(probs.sum()), 1e-8)
    y[valid] = probs
    return y


def softmax_scores(scores, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    out = np.zeros_like(scores, dtype=np.float32)
    if scores.size == 0:
        return out

    valid = np.isfinite(scores)
    if valid.sum() == 0:
        out[:] = 1.0 / float(len(out))
        return out

    logits = scores[valid] / max(float(temperature), 1e-6)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / max(float(probs.sum()), 1e-8)
    out[valid] = probs
    return out


def atomic_torch_save(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

