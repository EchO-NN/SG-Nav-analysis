import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_nav.dataset import GNNStepDataset
from gnn_nav.model import (
    GoalConditionedGraphNet,
    distillation_kl,
    infer_model_cfg_from_graph,
    soft_cross_entropy,
)


def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=lambda xs: xs[0])


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_count = 0

    for sample in loader:
        graph = sample["graph"].to(device)
        labels = sample["labels"]
        y_soft = labels["frontier_y_soft"].to(device)
        best_idx = int(labels["frontier_best_idx"])
        if best_idx < 0 or y_soft.numel() == 0:
            continue

        out = model(graph)
        logits = out["frontier_logits"]
        loss = soft_cross_entropy(logits, y_soft)

        if args.lambda_distill > 0:
            teacher_scores = sample.get("teacher", {}).get("sgnav_scores", None)
            if teacher_scores is not None:
                teacher_scores = torch.as_tensor(teacher_scores, dtype=torch.float32, device=device)
                if teacher_scores.numel() == logits.numel():
                    loss = loss + args.lambda_distill * distillation_kl(
                        logits,
                        teacher_scores,
                        args.distill_temperature,
                    )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        pred_idx = int(torch.argmax(logits).item())
        total_top1 += float(pred_idx == best_idx)
        total_loss += float(loss.item())
        total_count += 1

    return {
        "loss": total_loss / max(total_count, 1),
        "top1": total_top1 / max(total_count, 1),
        "count": total_count,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    cost_pred = []
    cost_best = []

    for sample in loader:
        graph = sample["graph"].to(device)
        labels = sample["labels"]
        best_idx = int(labels["frontier_best_idx"])
        if best_idx < 0:
            continue

        costs = labels["frontier_cost"].cpu().numpy()
        logits = model(graph)["frontier_logits"]
        if logits.numel() == 0:
            continue
        order = torch.argsort(logits, descending=True).cpu().numpy()
        pred = int(order[0])
        top1 += int(pred == best_idx)
        top3 += int(best_idx in order[: min(3, len(order))])
        total += 1
        cost_pred.append(float(costs[pred]))
        cost_best.append(float(costs[best_idx]))

    if total == 0:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "cost_pred": 0.0,
            "cost_best": 0.0,
            "cost_ratio": 0.0,
            "count": 0,
        }

    cost_pred_np = np.asarray(cost_pred, dtype=np.float32)
    cost_best_np = np.asarray(cost_best, dtype=np.float32)
    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "cost_pred": float(np.mean(cost_pred_np)),
        "cost_best": float(np.mean(cost_best_np)),
        "cost_ratio": float(np.mean(cost_pred_np / (cost_best_np + 1e-6))),
        "count": total,
    }


def save_checkpoint(model, model_cfg, args, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "model_cfg": model_cfg,
        "args": vars(args),
    }
    path = os.path.join(output_dir, "gnn_scorer.pt")
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    return path


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
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--lambda_distill", type=float, default=0.0)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    train_dataset = GNNStepDataset(args.train_dir, require_labels=True, max_samples=args.max_train_samples)
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {args.train_dir}")
    train_loader = make_loader(train_dataset, shuffle=True)

    val_loader = None
    if args.val_dir and os.path.isdir(args.val_dir):
        val_dataset = GNNStepDataset(args.val_dir, require_labels=True, max_samples=args.max_val_samples)
        if len(val_dataset) > 0:
            val_loader = make_loader(val_dataset, shuffle=False)

    graph0 = train_dataset[0]["graph"]
    model_cfg = infer_model_cfg_from_graph(
        graph0,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = GoalConditionedGraphNet(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args)
        msg = (
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1']:.3f} "
            f"train_count={train_metrics['count']}"
        )
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            msg += (
                f" val_top1={val_metrics['top1']:.3f}"
                f" val_top3={val_metrics['top3']:.3f}"
                f" val_cost_ratio={val_metrics['cost_ratio']:.3f}"
            )
        print(msg)

    path = save_checkpoint(model, model_cfg, args, args.output_dir)
    print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()
