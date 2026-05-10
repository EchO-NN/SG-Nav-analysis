import argparse
import json
from pathlib import Path


def _ratio(metrics, key):
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    args = parser.parse_args()

    records = []
    for line in Path(args.path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no metrics found: {args.path}")

    split = args.split
    usable = [record for record in records if record.get(split)]
    if not usable and split == "val":
        split = "train"
        usable = [record for record in records if record.get(split)]
    if not usable:
        raise SystemExit(f"no {args.split} metrics found: {args.path}")

    best = min(
        usable,
        key=lambda record: _ratio(record[split], "pred_cost_ratio") or _ratio(record[split], "loss"),
    )
    last = usable[-1]
    for name, record in [("best", best), ("last", last)]:
        metrics = record[split]
        print(f"{name}_epoch:", record.get("epoch"))
        for key in [
            "loss",
            "frontier_loss",
            "rank_loss",
            "teacher_loss",
            "pred_label_top1",
            "pred_label_top3",
            "cost_ratio",
            "random_cost_ratio",
            "distance_cost_ratio",
            "teacher_cost_ratio",
            "pred_teacher_agreement",
            "teacher_label_agreement",
            "num_frontiers_mean",
            "num_objects_mean",
            "count",
        ]:
            print(f"{name}_{key}:", metrics.get(key, 0))
        print(f"{name}_label_type_counts:", metrics.get("label_type_counts", {}))


if __name__ == "__main__":
    main()
