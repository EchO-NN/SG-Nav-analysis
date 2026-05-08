import json
import os
from collections import defaultdict


class EpisodeLogger:
    def __init__(self, log_dir="data/debug_sgnav", enabled=False):
        self.enabled = enabled
        self.log_dir = log_dir
        self.path = os.path.join(log_dir, "episodes.jsonl")
        self.rows = []
        if enabled:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, row):
        self.rows.append(row)
        if self.enabled:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.print_summary(row)

    def print_summary(self, row):
        count = len(self.rows)
        cumulative_sr = sum(float(r.get("success", 0)) for r in self.rows) / max(1, count)
        rolling_50 = self._rolling_sr(50)
        rolling_100 = self._rolling_sr(100)
        scene_sr = self._group_sr("scene_id", row.get("scene_id", "unknown"))
        goal_sr = self._group_sr("goal", row.get("goal", "unknown"))
        print(
            "[SGNAV_EPISODE] "
            f"episode={count} success={row.get('success', 0)} "
            f"cum_sr={cumulative_sr:.3f} roll50={rolling_50:.3f} "
            f"roll100={rolling_100:.3f} scene_sr={scene_sr:.3f} "
            f"goal_sr={goal_sr:.3f} goal={row.get('goal', 'unknown')} "
            f"scene={row.get('scene_id', 'unknown')}"
        )

    def _rolling_sr(self, window):
        rows = self.rows[-window:]
        return sum(float(r.get("success", 0)) for r in rows) / max(1, len(rows))

    def _group_sr(self, key, value):
        rows = [r for r in self.rows if r.get(key, "unknown") == value]
        return sum(float(r.get("success", 0)) for r in rows) / max(1, len(rows))

    def grouped_summary(self, key):
        groups = defaultdict(list)
        for row in self.rows:
            groups[row.get(key, "unknown")].append(row)
        summary = {}
        for name, rows in groups.items():
            summary[name] = {
                "count": len(rows),
                "sr": sum(float(r.get("success", 0)) for r in rows) / max(1, len(rows)),
            }
        return summary
