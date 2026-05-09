import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_nav.dataset import safe_torch_load
from gnn_train.dataset import graph_from_payload
from gnn_train.model import GoalConditionedGraphNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sample = safe_torch_load(args.path, map_location="cpu")
    graph = graph_from_payload(sample)
    centers = graph.frontier_centers_rc
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    labels = sample.get("labels", {})
    best_idx = int(labels.get("frontier_best_idx", -1))
    teacher_idx = int(labels.get("teacher_best_idx", -1))
    pred_idx = -1

    if args.ckpt:
        device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
        ckpt = safe_torch_load(args.ckpt, map_location=device)
        model = GoalConditionedGraphNet(**ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            logits = model(graph.to(device))["frontier_logits"]
            if logits.numel() > 0:
                pred_idx = int(torch.argmax(logits).item())

    map_size = int(sample.get("metadata", {}).get("map_size", 800))
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros((map_size, map_size), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
    if len(centers):
        plt.scatter(centers[:, 1], centers[:, 0], s=18, c="tab:blue", label="frontier cluster")
    if 0 <= best_idx < len(centers):
        p = centers[best_idx]
        plt.scatter([p[1]], [p[0]], s=160, marker="*", c="tab:red", label="label best")
    if 0 <= teacher_idx < len(centers):
        p = centers[teacher_idx]
        plt.scatter([p[1]], [p[0]], s=100, marker="x", c="tab:orange", label="teacher best")
    if 0 <= pred_idx < len(centers):
        p = centers[pred_idx]
        plt.scatter([p[1]], [p[0]], s=120, marker="P", c="tab:green", label="gnn pred")

    obj_centers = graph.node_features.get("object")
    if obj_centers is not None and obj_centers.shape[0] > 0:
        # Object center is stored after text embedding and goal similarity.
        text_dim_guess = obj_centers.shape[1] - len(graph.node_texts.get("room", [])) - 6
        idx0 = max(0, text_dim_guess + 1)
        obj_rc = obj_centers[:, idx0 : idx0 + 2].detach().cpu().numpy() * float(map_size)
        plt.scatter(obj_rc[:, 1], obj_rc[:, 0], s=12, c="tab:purple", label="objects")

    plt.title(
        f"{sample.get('metadata', {}).get('scene_id', '')} "
        f"step={sample.get('metadata', {}).get('step_id', '')}"
    )
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print(f"saved visualization: {args.output}")


if __name__ == "__main__":
    main()
