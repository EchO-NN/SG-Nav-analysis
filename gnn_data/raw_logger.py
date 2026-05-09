import os
from pathlib import Path
from typing import Optional

from gnn_data.raw_schema import RAW_VERSION, atomic_torch_save, sanitize_filename


class GNNRawLogger:
    """Atomic writer for raw SG-Nav frontier-decision snapshots."""

    def __init__(
        self,
        log_dir: str,
        enabled: bool = False,
        collect_every_k_fbe: int = 1,
        data_tag: str = "sgnav_teacher",
    ):
        self.log_dir = str(log_dir)
        self.enabled = bool(enabled)
        self.collect_every_k_fbe = max(1, int(collect_every_k_fbe))
        self.data_tag = str(data_tag)
        self.saved_count = 0
        if self.enabled:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def should_log(self, step_id: int) -> bool:
        if not self.enabled:
            return False
        return int(step_id) % self.collect_every_k_fbe == 0

    def save_step(self, sample: dict) -> Optional[str]:
        if not self.enabled:
            return None
        if sample.get("version") != RAW_VERSION:
            raise ValueError(f"Expected {RAW_VERSION}, got {sample.get('version')}")

        metadata = sample.get("metadata", {})
        step_id = int(metadata.get("step_id", 0))
        if not self.should_log(step_id):
            return None

        scene_id = sanitize_filename(metadata.get("scene_id", "unknown_scene"))
        episode_id = sanitize_filename(metadata.get("episode_id", "unknown_episode"))
        data_tag = sanitize_filename(metadata.get("data_tag", self.data_tag))
        filename = f"{data_tag}_{scene_id}_{episode_id}_{step_id:06d}.pt"
        path = os.path.join(self.log_dir, filename)
        atomic_torch_save(sample, path)
        self.saved_count += 1
        return path

