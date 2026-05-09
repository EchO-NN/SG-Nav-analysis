#!/usr/bin/env python3
import argparse
import collections
import gzip
import json
import random
import re
import shutil
from pathlib import Path


def log(message):
    print(f"[sample_objectnav] {message}", flush=True)


def load_json_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def save_json_gz(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    with gzip.open(tmp_path, "wt") as f:
        json.dump(payload, f)
    tmp_path.replace(path)


def read_scene_ids(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def scene_id_from_content_path(path):
    name = Path(path).name
    if not name.endswith(".json.gz"):
        raise ValueError(f"not a content json.gz: {path}")
    return name[: -len(".json.gz")]


def episode_category(episode):
    if episode.get("object_category"):
        return str(episode["object_category"])
    goals = episode.get("goals") or []
    if goals and isinstance(goals[0], dict):
        for key in ("object_category", "object_category_name", "category"):
            if goals[0].get(key):
                return str(goals[0][key])
    return "__unknown__"


def sample_random(episodes, limit, rng):
    sampled = list(episodes)
    rng.shuffle(sampled)
    return sampled[:limit]


def sample_balanced_categories(episodes, limit, rng):
    groups = collections.defaultdict(list)
    for episode in episodes:
        groups[episode_category(episode)].append(episode)

    for group in groups.values():
        rng.shuffle(group)

    categories = sorted(groups)
    rng.shuffle(categories)
    sampled = []
    while len(sampled) < limit:
        progressed = False
        round_categories = list(categories)
        rng.shuffle(round_categories)
        for category in round_categories:
            if not groups[category]:
                continue
            sampled.append(groups[category].pop())
            progressed = True
            if len(sampled) >= limit:
                break
        if not progressed:
            break
    return sampled


def sample_episodes(episodes, limit, strategy, rng):
    if limit <= 0:
        raise ValueError("--episodes_per_scene must be positive")
    if len(episodes) <= limit:
        sampled = list(episodes)
        rng.shuffle(sampled)
        return sampled
    if strategy == "random":
        return sample_random(episodes, limit, rng)
    if strategy == "balanced_categories":
        return sample_balanced_categories(episodes, limit, rng)
    raise ValueError(f"unknown strategy: {strategy}")


def available_content_paths(source_split_dir):
    content_dir = Path(source_split_dir) / "content"
    if not content_dir.is_dir():
        raise FileNotFoundError(f"missing content dir: {content_dir}")
    return sorted(content_dir.glob("*.json.gz"))


def select_content_paths(source_split_dir, scene_ids, scene_l, scene_r):
    all_paths = available_content_paths(source_split_dir)
    by_scene = {scene_id_from_content_path(path): path for path in all_paths}
    if scene_ids:
        missing = [scene_id for scene_id in scene_ids if scene_id not in by_scene]
        if missing:
            raise FileNotFoundError(f"missing scenes in source split: {missing[:8]}")
        selected = [by_scene[scene_id] for scene_id in scene_ids]
    else:
        selected = all_paths

    left = 0 if scene_l is None else scene_l
    right = len(selected) if scene_r is None else scene_r
    return selected[left:right]


def write_config(base_config, output_config, output_split, seed):
    base_config = Path(base_config)
    output_config = Path(output_config)
    text = base_config.read_text()
    text = re.sub(r"(?m)^(\s*SPLIT:\s*).*$", rf"\1{output_split}", text, count=1)
    if re.search(r"(?m)^SEED:\s*", text):
        text = re.sub(r"(?m)^SEED:\s*.*$", f"SEED: {seed}", text, count=1)
    else:
        text = f"SEED: {seed}\n" + text
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(text)


def format_counter(counter, limit=8):
    items = counter.most_common(limit)
    return ", ".join(f"{key}:{value}" for key, value in items)


def main():
    parser = argparse.ArgumentParser(
        description="Create a small sampled Habitat ObjectNav split from an existing split."
    )
    parser.add_argument(
        "--objectnav_root",
        type=str,
        default="data/MatterPort3D/objectnav/mp3d/v1",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Defaults to --objectnav_root. Useful for testing into /tmp.",
    )
    parser.add_argument("--source_split", type=str, default="gnn_unseen50")
    parser.add_argument("--output_split", type=str, required=True)
    parser.add_argument("--episodes_per_scene", type=int, default=2)
    parser.add_argument(
        "--strategy",
        choices=["balanced_categories", "random"],
        default="balanced_categories",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene_l", type=int, default=None)
    parser.add_argument("--scene_r", type=int, default=None)
    parser.add_argument("--scene_list", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/challenge_objectnav2021.local.rgbd.gnn_unseen50.yaml",
    )
    parser.add_argument("--write_config", type=str, default=None)
    args = parser.parse_args()

    objectnav_root = Path(args.objectnav_root)
    output_root = Path(args.output_root) if args.output_root else objectnav_root
    source_split_dir = objectnav_root / args.source_split
    output_split_dir = output_root / args.output_split
    if not source_split_dir.is_dir():
        raise FileNotFoundError(f"missing source split dir: {source_split_dir}")
    if output_split_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_split_dir} already exists; use --overwrite or choose a new --output_split"
            )
        shutil.rmtree(output_split_dir)

    scene_ids = read_scene_ids(args.scene_list) if args.scene_list else None
    content_paths = select_content_paths(
        source_split_dir=source_split_dir,
        scene_ids=scene_ids,
        scene_l=args.scene_l,
        scene_r=args.scene_r,
    )
    if not content_paths:
        raise RuntimeError("no scenes selected")

    rng = random.Random(args.seed)
    main_payload = load_json_gz(source_split_dir / f"{args.source_split}.json.gz")
    main_payload["episodes"] = []
    save_json_gz(main_payload, output_split_dir / f"{args.output_split}.json.gz")

    total_before = 0
    total_after = 0
    category_counter = collections.Counter()
    selected_scene_ids = []
    output_content_dir = output_split_dir / "content"

    for content_path in content_paths:
        scene_id = scene_id_from_content_path(content_path)
        payload = load_json_gz(content_path)
        episodes = payload.get("episodes", [])
        sampled = sample_episodes(
            episodes=episodes,
            limit=args.episodes_per_scene,
            strategy=args.strategy,
            rng=rng,
        )
        payload["episodes"] = sampled
        save_json_gz(payload, output_content_dir / f"{scene_id}.json.gz")

        total_before += len(episodes)
        total_after += len(sampled)
        selected_scene_ids.append(scene_id)
        category_counter.update(episode_category(episode) for episode in sampled)
        log(f"{scene_id}: {len(episodes)} -> {len(sampled)}")

    (output_split_dir / "scenes.txt").write_text("\n".join(selected_scene_ids) + "\n")

    if args.write_config:
        write_config(
            base_config=args.base_config,
            output_config=args.write_config,
            output_split=args.output_split,
            seed=args.seed,
        )
        log(f"wrote config: {args.write_config}")

    log(f"source split: {args.source_split}")
    log(f"output split: {args.output_split}")
    log(f"scenes: {len(selected_scene_ids)}")
    log(f"episodes: {total_before} -> {total_after}")
    log(f"categories: {format_counter(category_counter)}")
    log(f"split dir: {output_split_dir}")


if __name__ == "__main__":
    main()
