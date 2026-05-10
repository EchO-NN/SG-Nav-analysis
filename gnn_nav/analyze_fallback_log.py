import argparse
import glob
import os
from collections import Counter

from gnn_nav.dataset import safe_torch_load


def _records_from_payload(payload):
    fallback = payload.get("fallback")
    if isinstance(fallback, dict):
        return list(fallback.get("fallback_records", []) or [])
    debug = payload.get("debug")
    if isinstance(debug, dict) and debug.get("gnn_fallback"):
        return [debug["gnn_fallback"]]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--strict_nonzero", action="store_true")
    args = parser.parse_args()

    paths = []
    if args.path:
        paths = [args.path]
    if args.dir:
        paths.extend(sorted(glob.glob(os.path.join(args.dir, "**", "*.pt"), recursive=True)))
    if not paths:
        raise ValueError("provide --path or --dir")

    total_records = 0
    used_records = 0
    reasons = Counter()
    modes = Counter()
    episodes = set()
    for path in paths:
        payload = safe_torch_load(path, map_location="cpu")
        for record in _records_from_payload(payload):
            total_records += 1
            if bool(record.get("use_fallback", False)):
                used_records += 1
            modes[str(record.get("fallback_mode", "unknown"))] += 1
            episodes.add(str(record.get("episode_id", "")))
            for reason in record.get("fallback_reasons", []) or []:
                reasons[str(reason)] += 1

    print("num_files:", len(paths))
    print("num_episodes_with_records:", len([x for x in episodes if x]))
    print("num_fallback_records:", total_records)
    print("num_used_fallback:", used_records)
    print("fallback_rate:", float(used_records) / max(float(total_records), 1.0))
    print("fallback_mode_counts:", dict(modes))
    print("fallback_reason_counts:", dict(reasons))
    if args.strict_nonzero and total_records == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
