import csv
import os
import time
from pathlib import Path
from typing import Dict, Iterable


class ReplayBufferManifest:
    """Small CSV manifest for autonomous replay bookkeeping."""

    fieldnames = [
        "path",
        "scene_id",
        "episode_id",
        "step_id",
        "goal",
        "split",
        "label_type",
        "source_policy",
        "timestamp",
    ]

    def __init__(self, manifest_path: str):
        self.manifest_path = str(manifest_path)
        Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def append(self, row: Dict):
        payload = {key: row.get(key, "") for key in self.fieldnames}
        payload["timestamp"] = payload["timestamp"] or time.time()
        with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(payload)

    def extend(self, rows: Iterable[Dict]):
        for row in rows:
            self.append(row)

