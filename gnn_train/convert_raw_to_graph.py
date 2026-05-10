import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

from gnn_nav.dataset import safe_torch_load
from gnn_train.sparse_graph_builder import RawSampleGraphConverter
from gnn_train.text_encoder import TextEmbeddingCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frontier_clusters", type=int, default=32)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--text_cache", type=str, default="data/gnn/text_embeddings.pt")
    parser.add_argument("--text_dim", type=int, default=384)
    parser.add_argument("--text_backend", type=str, default="auto")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--drop_empty_frontiers", action="store_true")
    parser.add_argument("--skip_errors", action="store_true")
    parser.add_argument("--strict_frontier_clusters", action="store_true")
    parser.add_argument("--strict_label_aggregation", action="store_true")
    parser.add_argument("--forbid_distance_label_fallback", action="store_true")
    parser.add_argument("--allow_point_frontiers_debug", action="store_true")
    parser.add_argument("--allow_debug_fallbacks", action="store_true")
    parser.add_argument("--write_conversion_report", nargs="?", const="conversion_report.json", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pt"), recursive=True))
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    encoder = TextEmbeddingCache(
        cache_path=args.text_cache,
        dim=args.text_dim,
        backend=args.text_backend,
        device="cpu",
    )
    converter = RawSampleGraphConverter(
        text_encoder=encoder,
        max_frontier_clusters=args.max_frontier_clusters,
        min_cluster_size=args.min_cluster_size,
        strict_frontier_clusters=args.strict_frontier_clusters,
        strict_label_aggregation=args.strict_label_aggregation,
        forbid_distance_label_fallback=args.forbid_distance_label_fallback,
        allow_point_frontiers_debug=args.allow_point_frontiers_debug or not args.strict_frontier_clusters,
        device="cpu",
    )

    count = 0
    skipped = 0
    skip_reason_counts = Counter()
    report_counts = Counter()
    report_values = defaultdict(list)
    for path in paths:
        try:
            sample = safe_torch_load(path, map_location="cpu")
            graph_sample = converter.build_from_raw_sample(
                sample,
                tau=args.tau,
                teacher_temperature=args.teacher_temperature,
            )
            num_frontiers = int(graph_sample["graph"]["node_features"]["frontier"].shape[0])
            if args.drop_empty_frontiers and num_frontiers == 0:
                skipped += 1
                skip_reason_counts["samples_skipped_empty_frontiers"] += 1
                continue
        except Exception as exc:
            if args.skip_errors:
                skipped += 1
                reason = str(exc) or type(exc).__name__
                reason = reason.splitlines()[0][:160]
                skip_reason_counts[f"samples_skipped_{reason}"] += 1
                continue
            raise
        conversion_report = graph_sample.get("conversion_report", {})
        for key, value in conversion_report.items():
            if key in {"raw_frontier_count", "cluster_frontier_count"}:
                report_values[key].append(float(value))
                continue
            if isinstance(value, (int, bool)):
                report_counts[key] += int(value)
            elif isinstance(value, float):
                report_values[key].append(float(value))
        rel = os.path.relpath(path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path + ".tmp"
        torch.save(graph_sample, tmp_path)
        os.replace(tmp_path, out_path)
        count += 1

    encoder.save()
    report = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "num_input_files": len(paths),
        "num_converted_samples": count,
        "num_skipped_samples": skipped,
        "skip_reason_counts": dict(skip_reason_counts),
        "report_counts": dict(report_counts),
        "report_means": {k: float(sum(v) / max(1, len(v))) for k, v in report_values.items()},
        "strict_frontier_clusters": bool(args.strict_frontier_clusters),
        "strict_label_aggregation": bool(args.strict_label_aggregation),
        "forbid_distance_label_fallback": bool(args.forbid_distance_label_fallback),
        "allow_point_frontiers_debug": bool(args.allow_point_frontiers_debug),
    }
    for key in [
        "cluster_extraction_error_count",
        "cluster_fallback_to_points_count",
        "cluster_label_missing_count",
        "distance_only_label_count",
    ]:
        report["report_counts"].setdefault(key, 0)

    print(f"converted {count} samples -> {args.output_dir}")
    if skipped:
        print(f"skipped {skipped} samples")
    print("conversion_report:", json.dumps(report, indent=2, ensure_ascii=False))
    if args.write_conversion_report:
        report_path = Path(args.write_conversion_report)
        if not report_path.is_absolute():
            report_path = Path(args.output_dir) / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote conversion report: {report_path}")

    if not args.allow_debug_fallbacks and (
        args.strict_frontier_clusters or args.strict_label_aggregation or args.forbid_distance_label_fallback
    ):
        bad_counts = {
            key: int(report["report_counts"].get(key, 0))
            for key in [
                "cluster_fallback_to_points_count",
                "distance_only_label_count",
                "cluster_label_missing_count",
            ]
        }
        bad_counts = {key: value for key, value in bad_counts.items() if value > 0}
        if bad_counts:
            raise RuntimeError(f"strict conversion produced forbidden fallbacks: {bad_counts}")


if __name__ == "__main__":
    main()
