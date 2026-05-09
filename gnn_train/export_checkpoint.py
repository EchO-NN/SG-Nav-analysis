import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    required = {"model", "model_cfg"}
    missing = sorted(required - set(ckpt.keys()))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {missing}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, args.output)
    print(f"exported checkpoint: {args.output}")


if __name__ == "__main__":
    main()

