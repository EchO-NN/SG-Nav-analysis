import argparse
import os

import numpy as np
import torch

from gnn_nav.dataset import safe_torch_load
from gnn_train.dataset import FrontierGraphDataset, collate_single
from gnn_train.model import GoalConditionedGraphNet


def _distance_scores(sample):
    frontier = sample.get("frontier", {})
    inv = frontier.get("distance_inverse")
    if inv is not None:
        if hasattr(inv, "detach"):
            inv = inv.detach().cpu().numpy()
        return np.asarray(inv, dtype=np.float32).reshape(-1)
    graph = sample["graph"]
    x = graph.node_features["frontier"].detach().cpu().numpy()
    if x.shape[1] > 4:
        return x[:, 4]
    return np.zeros((x.shape[0],), dtype=np.float32)


def _metrics(records):
    if len(records) == 0:
        return {"top1": 0.0, "top3": 0.0, "chosen_cost": 0.0, "best_cost": 0.0, "cost_ratio": 0.0, "count": 0}
    chosen = np.asarray([r["chosen"] for r in records], dtype=np.float32)
    best = np.asarray([r["best"] for r in records], dtype=np.float32)
    valid = np.isfinite(chosen) & np.isfinite(best) & (np.abs(best) > 1e-6)
    ratio = float(np.mean(chosen[valid] / (best[valid] + 1e-6))) if valid.sum() else 0.0
    return {
        "top1": float(np.mean([r["top1"] for r in records])),
        "top3": float(np.mean([r["top3"] for r in records])),
        "chosen_cost": float(np.mean(chosen)),
        "best_cost": float(np.mean(best)),
        "cost_ratio": ratio,
        "count": len(records),
    }


def _record(costs, scores, best_idx):
    if len(scores) == 0 or best_idx < 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    pred = int(order[0])
    finite_costs = np.where(np.isfinite(costs), costs, np.inf)
    return {
        "top1": float(pred == best_idx),
        "top3": float(best_idx in order[: min(3, len(order))]),
        "chosen": float(finite_costs[pred]),
        "best": float(finite_costs[best_idx]),
    }


@torch.no_grad()
def evaluate_dataset(dataset, ckpt=None, device="cpu", seed=0):
    model = None
    if ckpt is not None:
        payload = safe_torch_load(ckpt, map_location=device)
        model = GoalConditionedGraphNet(**payload["model_cfg"]).to(device)
        model.load_state_dict(payload["model"])
        model.eval()

    rng = np.random.default_rng(seed)
    records = {"random": [], "distance": [], "teacher": [], "gnn": []}
    for sample in dataset:
        labels = sample["labels"]
        costs = labels["frontier_cost"]
        if hasattr(costs, "detach"):
            costs = costs.detach().cpu().numpy()
        costs = np.asarray(costs, dtype=np.float32).reshape(-1)
        best_idx = int(labels.get("frontier_best_idx", -1))
        if len(costs) == 0 or best_idx < 0:
            continue

        random_scores = rng.random(len(costs)).astype(np.float32)
        rec = _record(costs, random_scores, best_idx)
        if rec:
            records["random"].append(rec)

        rec = _record(costs, _distance_scores(sample), best_idx)
        if rec:
            records["distance"].append(rec)

        teacher_scores = labels.get("teacher_scores")
        if teacher_scores is not None:
            if hasattr(teacher_scores, "detach"):
                teacher_scores = teacher_scores.detach().cpu().numpy()
            rec = _record(costs, np.asarray(teacher_scores, dtype=np.float32), best_idx)
            if rec:
                records["teacher"].append(rec)

        if model is not None:
            graph = sample["graph"].to(device)
            logits = model(graph)["frontier_logits"].detach().cpu().numpy()
            rec = _record(costs, logits, best_idx)
            if rec:
                records["gnn"].append(rec)

    return {name: _metrics(vals) for name, vals in records.items() if vals or name != "gnn" or model is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.ckpt and not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    dataset = FrontierGraphDataset(args.data_dir, max_samples=args.max_samples, require_labels=True)
    metrics = evaluate_dataset(dataset, ckpt=args.ckpt, device=device, seed=args.seed)
    for name, vals in metrics.items():
        print(name, vals)


if __name__ == "__main__":
    main()

