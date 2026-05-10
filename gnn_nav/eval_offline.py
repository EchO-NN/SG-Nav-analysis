import argparse
import os

import numpy as np
import torch

from gnn_nav.dataset import GNNStepDataset, safe_torch_load
from gnn_nav.model import GoalConditionedGraphNet
from gnn_nav.train_gnn import evaluate, make_loader


def teacher_label_agreement(dataset):
    total = 0
    agree = 0
    for sample in dataset:
        labels = sample["labels"]
        best_idx = int(labels.get("frontier_best_idx", -1))
        teacher_scores = sample.get("teacher", {}).get("sgnav_scores", None)
        if best_idx < 0 or teacher_scores is None:
            continue
        teacher_scores = np.asarray(teacher_scores, dtype=np.float32)
        if len(teacher_scores) == 0:
            continue
        agree += int(int(np.argmax(teacher_scores)) == best_idx)
        total += 1
    return agree / max(total, 1), total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    dataset = GNNStepDataset(args.data_dir, require_labels=True, max_samples=args.max_samples)
    ckpt = safe_torch_load(args.ckpt, map_location=device)
    model = GoalConditionedGraphNet(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate(model, make_loader(dataset, shuffle=False), device)
    agreement, agreement_count = teacher_label_agreement(dataset)
    print("metrics", metrics)
    print("teacher_label_agreement", {"top1": agreement, "count": agreement_count})


if __name__ == "__main__":
    main()
