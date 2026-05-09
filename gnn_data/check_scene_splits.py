import argparse
import glob
import os

from gnn_nav.dataset import safe_torch_load


def scenes_from_dir(path):
    scenes = set()
    for sample_path in sorted(glob.glob(os.path.join(path, "**", "*.pt"), recursive=True)):
        sample = safe_torch_load(sample_path, map_location="cpu")
        scene_id = sample.get("metadata", {}).get("scene_id")
        if scene_id:
            scenes.add(str(scene_id))
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    args = parser.parse_args()

    splits = {"train": scenes_from_dir(args.train_dir)}
    if args.val_dir:
        splits["val"] = scenes_from_dir(args.val_dir)
    if args.test_dir:
        splits["test"] = scenes_from_dir(args.test_dir)

    for name, scenes in splits.items():
        print(f"{name}: {len(scenes)} scenes")

    names = list(splits)
    failed = False
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(splits[left] & splits[right])
            if overlap:
                failed = True
                print(f"OVERLAP {left}/{right}: {overlap}")
    if failed:
        raise SystemExit(1)
    print("scene split check passed")


if __name__ == "__main__":
    main()

