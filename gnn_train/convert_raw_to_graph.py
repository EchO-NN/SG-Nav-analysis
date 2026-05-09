import argparse
import glob
import os
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
        device="cpu",
    )

    count = 0
    skipped = 0
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
                continue
        except Exception:
            if args.skip_errors:
                skipped += 1
                continue
            raise
        rel = os.path.relpath(path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path + ".tmp"
        torch.save(graph_sample, tmp_path)
        os.replace(tmp_path, out_path)
        count += 1

    encoder.save()
    print(f"converted {count} samples -> {args.output_dir}")
    if skipped:
        print(f"skipped {skipped} samples")


if __name__ == "__main__":
    main()

