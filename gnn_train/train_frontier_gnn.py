import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_train.dataset import FrontierGraphDataset, collate_single
from gnn_train.losses import distillation_kl, pairwise_ranking_loss, soft_cross_entropy
from gnn_train.model import GoalConditionedGraphNet, infer_model_cfg_from_graph


def make_loader(dataset, shuffle=False, batch_size=1):
    if int(batch_size) != 1:
        raise ValueError("SparseGNN training currently supports batch_size=1")
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_single)


def _safe_ratio(chosen, best):
    chosen = np.asarray(chosen, dtype=np.float32)
    best = np.asarray(best, dtype=np.float32)
    valid = np.isfinite(chosen) & np.isfinite(best) & (np.abs(best) > 1e-6)
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(chosen[valid] / (best[valid] + 1e-6)))


def _metric_dict(records):
    if len(records) == 0:
        return {
            "loss": 0.0,
            "frontier_loss": 0.0,
            "teacher_loss": 0.0,
            "rank_loss": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "chosen_cost": 0.0,
            "best_cost": 0.0,
            "cost_ratio": 0.0,
            "teacher_agreement": 0.0,
            "num_frontiers_mean": 0.0,
            "num_objects_mean": 0.0,
            "count": 0,
        }
    chosen = [r["chosen_cost"] for r in records]
    best = [r["best_cost"] for r in records]
    return {
        "loss": float(np.mean([r["loss"] for r in records])),
        "frontier_loss": float(np.mean([r["frontier_loss"] for r in records])),
        "teacher_loss": float(np.mean([r["teacher_loss"] for r in records])),
        "rank_loss": float(np.mean([r["rank_loss"] for r in records])),
        "top1": float(np.mean([r["top1"] for r in records])),
        "top3": float(np.mean([r["top3"] for r in records])),
        "chosen_cost": float(np.mean(chosen)),
        "best_cost": float(np.mean(best)),
        "cost_ratio": _safe_ratio(chosen, best),
        "teacher_agreement": float(np.mean([r["teacher_agreement"] for r in records if r["teacher_agreement"] >= 0]))
        if any(r["teacher_agreement"] >= 0 for r in records)
        else 0.0,
        "num_frontiers_mean": float(np.mean([r["num_frontiers"] for r in records])),
        "num_objects_mean": float(np.mean([r["num_objects"] for r in records])),
        "count": len(records),
    }


def _record_from_logits(sample, logits, loss, frontier_loss, teacher_loss, rank_loss):
    labels = sample["labels"]
    y_soft = labels["frontier_y_soft"].to(logits.device)
    costs = labels["frontier_cost"].to(logits.device)
    best_idx = int(labels.get("frontier_best_idx", int(torch.argmax(y_soft).item())))
    if logits.numel() == 0 or best_idx < 0:
        return None
    order = torch.argsort(logits, descending=True)
    pred = int(order[0].item())
    top3 = bool(best_idx in order[: min(3, len(order))].detach().cpu().tolist())
    finite_costs = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, float("inf")))
    chosen_cost = float(finite_costs[pred].detach().cpu().item())
    best_cost = float(finite_costs[best_idx].detach().cpu().item())
    teacher_best = int(labels.get("teacher_best_idx", -1))
    return {
        "loss": float(loss.detach().cpu().item()),
        "frontier_loss": float(frontier_loss.detach().cpu().item()),
        "teacher_loss": float(teacher_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "top1": float(pred == best_idx),
        "top3": float(top3),
        "chosen_cost": chosen_cost,
        "best_cost": best_cost,
        "teacher_agreement": float(teacher_best == best_idx) if teacher_best >= 0 else -1.0,
        "num_frontiers": int(logits.numel()),
        "num_objects": int(sample["graph"].node_features.get("object").shape[0]),
    }


def _losses_from_sample(model, sample, device, args):
    graph = sample["graph"].to(device)
    labels = sample["labels"]
    y_soft = labels["frontier_y_soft"].to(device)
    costs = labels["frontier_cost"].to(device)
    out = model(graph)
    logits = out["frontier_logits"]
    frontier_loss = soft_cross_entropy(logits, y_soft)
    teacher_loss = logits.sum() * 0.0
    rank_loss = logits.sum() * 0.0
    if args.lambda_teacher > 0 and "teacher_scores" in labels:
        teacher_scores = labels["teacher_scores"].to(device)
        if teacher_scores.numel() == logits.numel():
            teacher_loss = distillation_kl(logits, teacher_scores, args.teacher_temperature)
    if args.lambda_rank > 0:
        rank_loss = pairwise_ranking_loss(logits, costs, margin=args.rank_margin, max_pairs=args.rank_max_pairs)
    loss = frontier_loss + float(args.lambda_teacher) * teacher_loss + float(args.lambda_rank) * rank_loss
    return logits, loss, frontier_loss, teacher_loss, rank_loss


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    records = []
    for sample in loader:
        if sample["labels"].get("frontier_y_soft", torch.zeros(0)).numel() == 0:
            continue
        logits, loss, frontier_loss, teacher_loss, rank_loss = _losses_from_sample(model, sample, device, args)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        record = _record_from_logits(sample, logits, loss, frontier_loss, teacher_loss, rank_loss)
        if record is not None:
            records.append(record)
    return _metric_dict(records)


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    records = []
    for sample in loader:
        if sample["labels"].get("frontier_y_soft", torch.zeros(0)).numel() == 0:
            continue
        logits, loss, frontier_loss, teacher_loss, rank_loss = _losses_from_sample(model, sample, device, args)
        record = _record_from_logits(sample, logits, loss, frontier_loss, teacher_loss, rank_loss)
        if record is not None:
            records.append(record)
    return _metric_dict(records)


def save_checkpoint(model, model_cfg, args, output_dir, best_metrics):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "model_cfg": model_cfg,
        "train_args": vars(args),
        "best_val_metrics": best_metrics,
    }
    path = os.path.join(output_dir, "frontier_gnn.pt")
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    compat_path = os.path.join(output_dir, "gnn_scorer.pt")
    tmp_compat = compat_path + ".tmp"
    torch.save(ckpt, tmp_compat)
    os.replace(tmp_compat, compat_path)
    return path


def _format_metrics(prefix, metrics):
    keys = [
        "loss",
        "frontier_loss",
        "teacher_loss",
        "rank_loss",
        "top1",
        "top3",
        "cost_ratio",
        "teacher_agreement",
        "num_frontiers_mean",
        "num_objects_mean",
        "count",
    ]
    return " ".join(f"{prefix}_{key}={metrics[key]:.4f}" if key != "count" else f"{prefix}_count={metrics[key]}" for key in keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_teacher", type=float, default=0.0)
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--teacher_temperature", type=float, default=2.0)
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--rank_max_pairs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    train_dataset = FrontierGraphDataset(args.train_dir, max_samples=args.max_train_samples, require_labels=True)
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {args.train_dir}")
    train_loader = make_loader(train_dataset, shuffle=True, batch_size=args.batch_size)

    val_loader = None
    if args.val_dir and os.path.isdir(args.val_dir):
        val_dataset = FrontierGraphDataset(args.val_dir, max_samples=args.max_val_samples, require_labels=True)
        if len(val_dataset) > 0:
            val_loader = make_loader(val_dataset, shuffle=False, batch_size=1)

    graph0 = train_dataset[0]["graph"]
    model_cfg = infer_model_cfg_from_graph(
        graph0,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = GoalConditionedGraphNet(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("inf")
    best_metrics = {}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args)
        msg = f"epoch={epoch:03d} " + _format_metrics("train", train_metrics)
        monitor = train_metrics["cost_ratio"] if train_metrics["cost_ratio"] > 0 else train_metrics["loss"]
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args)
            msg += " " + _format_metrics("val", val_metrics)
            monitor = val_metrics["cost_ratio"] if val_metrics["cost_ratio"] > 0 else val_metrics["loss"]
            if not best_metrics or monitor <= best_score:
                best_score = monitor
                best_metrics = val_metrics
                save_checkpoint(model, model_cfg, args, args.output_dir, best_metrics)
        else:
            if not best_metrics or monitor <= best_score:
                best_score = monitor
                best_metrics = train_metrics
                save_checkpoint(model, model_cfg, args, args.output_dir, best_metrics)
        print(msg)

    path = save_checkpoint(model, model_cfg, args, args.output_dir, best_metrics)
    print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()

