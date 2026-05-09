#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


OBJECTNAV_MP3D_URL = (
    "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/"
    "objectnav_mp3d_v1.zip"
)
MP3D_BASE_URL = "http://kaldir.vc.in.tum.de/matterport/v1/scans"
MP3D_HABITAT_URL = "https://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip"


def log(message):
    print(f"[prepare_mp3d] {message}", flush=True)


def download_file(url, path, force=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        log(f"skip existing download: {path}")
        return path
    tmp_path = Path(str(path) + ".tmp")
    log(f"download: {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(path)
    return path


def existing_scene_ids(mp3d_dir, objectnav_root, exclude_objectnav_splits):
    ids = set()
    mp3d_dir = Path(mp3d_dir)
    if mp3d_dir.exists():
        for glb in mp3d_dir.glob("*/*.glb"):
            if glb.stem == glb.parent.name:
                ids.add(glb.parent.name)

    objectnav_root = Path(objectnav_root)
    for split in exclude_objectnav_splits:
        content_dir = objectnav_root / split / "content"
        if not content_dir.is_dir():
            continue
        for item in content_dir.glob("*.json.gz"):
            ids.add(item.name[: -len(".json.gz")])
    return ids


def safe_extract_zip(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        zf.extractall(out_dir)


def find_objectnav_root(root):
    root = Path(root)
    candidates = []
    for path in root.rglob("train.json.gz"):
        if path.parent.name == "train" and (path.parent / "content").is_dir():
            candidates.append(path.parent.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def ensure_objectnav_source(args):
    source_root = Path(args.objectnav_source_root)
    if (source_root / "train" / "content").is_dir():
        return source_root

    if not args.download_objectnav:
        raise FileNotFoundError(
            f"ObjectNav train content not found at {source_root}. "
            "Run with --download_objectnav or point --objectnav_source_root to an existing dataset."
        )

    zip_path = Path(args.download_dir) / "objectnav_mp3d_v1.zip"
    download_file(OBJECTNAV_MP3D_URL, zip_path, force=args.force)

    extract_root = Path(args.download_dir) / "objectnav_mp3d_v1_extracted"
    if args.force and extract_root.exists():
        shutil.rmtree(extract_root)
    if not extract_root.exists():
        log(f"extract objectnav dataset: {zip_path}")
        safe_extract_zip(zip_path, extract_root)

    found_root = find_objectnav_root(extract_root)
    if found_root is None:
        raise RuntimeError(f"Could not find objectnav train/content after extracting {zip_path}")

    source_root.parent.mkdir(parents=True, exist_ok=True)
    if source_root.exists() and args.force:
        shutil.rmtree(source_root)
    if not source_root.exists():
        log(f"copy objectnav source root: {found_root} -> {source_root}")
        shutil.copytree(found_root, source_root)
    return source_root


def objectnav_scene_ids(source_root, preferred_splits):
    scenes = []
    for split in preferred_splits:
        content_dir = Path(source_root) / split / "content"
        if not content_dir.is_dir():
            continue
        for item in sorted(content_dir.glob("*.json.gz")):
            scene_id = item.name[: -len(".json.gz")]
            if scene_id not in scenes:
                scenes.append(scene_id)
    return scenes


def select_unseen_scenes(source_root, existing_ids, count, seed, preferred_splits):
    candidates = [sid for sid in objectnav_scene_ids(source_root, preferred_splits) if sid not in existing_ids]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = sorted(candidates[:count])
    if len(selected) < count:
        raise RuntimeError(f"Need {count} unseen scenes, found only {len(selected)}")
    return selected


def find_glbs_in_zip(zip_path):
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".glb"):
                out.append(member)
    return out


def extract_scan_glb(zip_path, scene_id, mp3d_dir):
    target_dir = Path(mp3d_dir) / scene_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_glb = target_dir / f"{scene_id}.glb"
    if target_glb.exists() and target_glb.stat().st_size > 0:
        return target_glb

    glb_members = find_glbs_in_zip(zip_path)
    if not glb_members:
        raise RuntimeError(f"No .glb found inside {zip_path}")

    preferred = [m for m in glb_members if Path(m).name == f"{scene_id}.glb"]
    member = preferred[0] if preferred else glb_members[0]
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp_dir:
        extracted = zf.extract(member, tmp_dir)
        shutil.move(extracted, target_glb)
    return target_glb


def download_scene_mesh(scene_id, args):
    target_glb = Path(args.mp3d_dir) / scene_id / f"{scene_id}.glb"
    if target_glb.exists() and target_glb.stat().st_size > 0:
        log(f"mesh exists: {target_glb}")
        return target_glb

    if not args.download_meshes:
        log(f"missing mesh, dry mode: {target_glb}")
        return None

    zip_path = Path(args.download_dir) / "mp3d_scans" / scene_id / f"{args.mesh_type}.zip"
    url = f"{MP3D_BASE_URL}/{scene_id}/{args.mesh_type}.zip"
    download_file(url, zip_path, force=args.force)
    glb = extract_scan_glb(zip_path, scene_id, args.mp3d_dir)
    log(f"prepared mesh: {glb}")
    return glb


def ensure_habitat_archive(args):
    archive_path = Path(args.habitat_archive_path)
    if archive_path.exists() and archive_path.stat().st_size > 0 and not args.force:
        return archive_path
    if not args.download_meshes:
        return archive_path
    return download_file(args.habitat_archive_url, archive_path, force=args.force)


def extract_scene_from_habitat_archive(scene_id, archive_path, mp3d_dir):
    target_dir = Path(mp3d_dir) / scene_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_glb = target_dir / f"{scene_id}.glb"
    if target_glb.exists() and target_glb.stat().st_size > 0:
        log(f"mesh exists: {target_glb}")
        return target_glb

    with zipfile.ZipFile(archive_path) as zf:
        members = zf.namelist()
        preferred_names = [
            f"mp3d/{scene_id}/{scene_id}.glb",
            f"{scene_id}/{scene_id}.glb",
        ]
        member = None
        for name in preferred_names:
            if name in members:
                member = name
                break
        if member is None:
            suffix = f"/{scene_id}/{scene_id}.glb"
            matches = [name for name in members if name.endswith(suffix)]
            if matches:
                member = matches[0]
        if member is None:
            glb_matches = [
                name
                for name in members
                if name.lower().endswith(".glb") and f"/{scene_id}/" in name
            ]
            if glb_matches:
                member = glb_matches[0]
        if member is None:
            raise FileNotFoundError(f"No GLB for scene {scene_id} in {archive_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            extracted = zf.extract(member, tmp_dir)
            shutil.move(extracted, target_glb)
    log(f"prepared mesh: {target_glb}")
    return target_glb


def extract_selected_from_habitat_archive(selected, args):
    archive_path = ensure_habitat_archive(args)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Habitat archive missing: {archive_path}. Run with --download_meshes "
            "or provide --habitat_archive_path."
        )
    for scene_id in selected:
        extract_scene_from_habitat_archive(scene_id, archive_path, args.mp3d_dir)


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


def prepare_objectnav_split(selected, source_root, dest_root, split_name, preferred_splits):
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    split_dir = dest_root / split_name
    content_dir = split_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)

    main_payload = None
    for split in preferred_splits:
        candidate = source_root / split / f"{split}.json.gz"
        if candidate.exists():
            main_payload = load_json_gz(candidate)
            break
    if main_payload is None:
        raise FileNotFoundError(f"Could not find source split metadata in {source_root}")
    main_payload["episodes"] = []
    save_json_gz(main_payload, split_dir / f"{split_name}.json.gz")

    copied = 0
    for scene_id in selected:
        src = None
        for split in preferred_splits:
            candidate = source_root / split / "content" / f"{scene_id}.json.gz"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            raise FileNotFoundError(f"Missing ObjectNav content for {scene_id}")
        dst = content_dir / f"{scene_id}.json.gz"
        shutil.copy2(src, dst)
        copied += 1

    list_path = split_dir / "scenes.txt"
    list_path.write_text("\n".join(selected) + "\n")
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_name", type=str, default="gnn_unseen50")
    parser.add_argument("--mp3d_dir", type=str, default="data/MatterPort3D/mp3d")
    parser.add_argument("--objectnav_root", type=str, default="data/MatterPort3D/objectnav/mp3d/v1")
    parser.add_argument(
        "--objectnav_source_root",
        type=str,
        default="data/MatterPort3D/objectnav/mp3d/v1_full",
    )
    parser.add_argument("--download_dir", type=str, default="data/downloads")
    parser.add_argument("--preferred_splits", nargs="+", default=["train", "val"])
    parser.add_argument("--exclude_objectnav_splits", nargs="+", default=["val"])
    parser.add_argument("--scene_list", type=str, default=None)
    parser.add_argument("--mesh_type", type=str, default="matterport_mesh")
    parser.add_argument("--mesh_source", choices=["habitat_archive", "scan_mesh"], default="habitat_archive")
    parser.add_argument("--habitat_archive_url", type=str, default=MP3D_HABITAT_URL)
    parser.add_argument("--habitat_archive_path", type=str, default="data/downloads/mp3d_habitat.zip")
    parser.add_argument("--download_objectnav", action="store_true")
    parser.add_argument("--download_meshes", action="store_true")
    parser.add_argument("--prepare_split", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_root = ensure_objectnav_source(args)
    existing = existing_scene_ids(args.mp3d_dir, args.objectnav_root, args.exclude_objectnav_splits)
    if args.scene_list:
        selected = [
            line.strip()
            for line in Path(args.scene_list).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        selected = select_unseen_scenes(
            source_root=source_root,
            existing_ids=existing,
            count=args.num_scenes,
            seed=args.seed,
            preferred_splits=args.preferred_splits,
        )

    log(f"existing scenes: {len(existing)}")
    log(f"selected unseen scenes ({len(selected)}):")
    for scene_id in selected:
        print(scene_id)

    if args.download_meshes:
        if args.mesh_source == "habitat_archive":
            extract_selected_from_habitat_archive(selected, args)
        else:
            for scene_id in selected:
                download_scene_mesh(scene_id, args)

    if args.prepare_split:
        copied = prepare_objectnav_split(
            selected=selected,
            source_root=source_root,
            dest_root=args.objectnav_root,
            split_name=args.split_name,
            preferred_splits=args.preferred_splits,
        )
        log(f"prepared ObjectNav split {args.split_name}: {copied} content files")
        log(f"split dir: {Path(args.objectnav_root) / args.split_name}")


if __name__ == "__main__":
    main()
