import argparse
import json
from pathlib import Path


FORBIDDEN_FINAL_COUNTS = [
    "cluster_fallback_to_points_count",
    "distance_only_label_count",
    "cluster_label_missing_count",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = json.loads(Path(args.path).read_text(encoding="utf-8"))
    counts = report.get("report_counts", {})
    print("path:", args.path)
    print("num_input_files:", report.get("num_input_files", 0))
    print("num_converted_samples:", report.get("num_converted_samples", 0))
    print("num_skipped_samples:", report.get("num_skipped_samples", 0))
    print("strict_frontier_clusters:", report.get("strict_frontier_clusters", False))
    print("strict_label_aggregation:", report.get("strict_label_aggregation", False))
    print("forbid_distance_label_fallback:", report.get("forbid_distance_label_fallback", False))
    print("allow_point_frontiers_debug:", report.get("allow_point_frontiers_debug", False))
    print("skip_reason_counts:", report.get("skip_reason_counts", {}))
    print("report_counts:", counts)
    print("report_means:", report.get("report_means", {}))

    bad = {key: int(counts.get(key, 0)) for key in FORBIDDEN_FINAL_COUNTS if int(counts.get(key, 0)) > 0}
    if bad:
        print("forbidden_final_counts:", bad)
        if args.strict:
            raise SystemExit(1)
    print("validation:", "ok" if not bad else "has_debug_fallbacks")


if __name__ == "__main__":
    main()
