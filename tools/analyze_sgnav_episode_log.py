import json
import os
import sys
from collections import defaultdict


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(rows, key):
    return sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))


def latest_run_rows(rows):
    if not rows:
        return rows, False

    start = 0
    prev_idx = rows[0].get("episode_idx")
    for idx, row in enumerate(rows[1:], start=1):
        cur_idx = row.get("episode_idx")
        if isinstance(prev_idx, int) and isinstance(cur_idx, int) and cur_idx <= prev_idx:
            start = idx
        prev_idx = cur_idx
    return rows[start:], start > 0


def print_group(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "unknown")].append(row)
    print(f"\n== {key} ==")
    for name, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        sr = mean(items, "success")
        spl = mean(items, "spl")
        softspl = mean(items, "softspl")
        print(f"{name} count={len(items)} sr={sr:.3f} spl={spl:.3f} softspl={softspl:.3f}")


def main():
    if len(sys.argv) != 2:
        print("usage: python tools/analyze_sgnav_episode_log.py path/to/episodes.jsonl")
        raise SystemExit(2)
    if not os.path.exists(sys.argv[1]):
        print(f"episode log not found: {sys.argv[1]}")
        raise SystemExit(1)

    all_rows = load_rows(sys.argv[1])
    rows, trimmed = latest_run_rows(all_rows)
    if trimmed:
        print(
            f"warning: detected appended logs from multiple runs; "
            f"analyzing latest run only ({len(rows)}/{len(all_rows)} rows)"
        )
    if not rows:
        print("no rows")
        return

    print(f"episodes={len(rows)}")
    print(f"cumulative_sr={mean(rows, 'success'):.3f}")
    print(f"cumulative_spl={mean(rows, 'spl'):.3f}")
    print(f"cumulative_softspl={mean(rows, 'softspl'):.3f}")
    print(f"avg_nodes={mean(rows, 'nodes_final'):.2f}")
    print(f"avg_edges={mean(rows, 'edges_final'):.2f}")

    for window in [50, 100, 200]:
        recent = rows[-window:]
        print(f"rolling_sr_{window}={mean(recent, 'success'):.3f} n={len(recent)}")

    latest_counters = rows[-1].get("llm_parse_failures", {})
    if latest_counters:
        def rate(fail_key, total_key):
            return float(latest_counters.get(fail_key, 0)) / max(
                1.0,
                float(latest_counters.get(total_key, 0)),
            )

        print("\n== parser counters ==")
        print(json.dumps(latest_counters, indent=2, sort_keys=True))
        print(f"probability_parse_fail_rate={rate('probability_parse_fail', 'probability_parse_total'):.3f}")
        print(f"edge_proposal_parse_fail_rate={rate('edge_proposal_parse_fail', 'edge_proposal_total'):.3f}")
        print(f"room_predict_fail_rate={rate('room_predict_parse_fail', 'room_predict_total'):.3f}")

    print_group(rows, "scene_id")
    print_group(rows, "goal")


if __name__ == "__main__":
    main()
