import json
import os


class CandidateTraceLogger:
    def __init__(self, enabled=False, log_dir="data/debug_sgnav"):
        self.enabled = bool(enabled)
        self.log_dir = log_dir
        self.path = os.path.join(log_dir, "candidate_events.jsonl")
        self.episode_idx = None
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)

    def start_episode(self, episode_idx):
        self.episode_idx = int(episode_idx)

    def log(self, row):
        if not self.enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        item = self._to_jsonable(row)
        item.setdefault("episode_idx", self.episode_idx)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _to_jsonable(self, value):
        if self._is_torch_tensor(value):
            return self._to_jsonable(value.detach().cpu().numpy())
        if self._is_numpy_array(value):
            return value.tolist()
        if self._is_numpy_scalar(value):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value

    def _is_torch_tensor(self, value):
        return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")

    def _is_numpy_array(self, value):
        return value.__class__.__module__.startswith("numpy") and hasattr(value, "tolist")

    def _is_numpy_scalar(self, value):
        return value.__class__.__module__.startswith("numpy") and hasattr(value, "item")
