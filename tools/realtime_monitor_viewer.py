#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def make_placeholder(path, message, width=900, height=560):
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "SG-Nav realtime monitor",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    lines = [
        message,
        f"path: {path}",
        "Press q or Esc to close.",
    ]
    y = 95
    for line in lines:
        cv2.putText(
            image,
            line[:130],
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        y += 32
    return image


def resize_to_fit(image, max_width, max_height):
    if max_width <= 0 and max_height <= 0:
        return image
    height, width = image.shape[:2]
    scale = 1.0
    if max_width > 0:
        scale = min(scale, max_width / max(width, 1))
    if max_height > 0:
        scale = min(scale, max_height / max(height, 1))
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def add_overlay(image, path, mtime_ns):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    timestamp = "waiting"
    if mtime_ns is not None:
        timestamp = time.strftime(
            "%H:%M:%S",
            time.localtime(mtime_ns / 1_000_000_000),
        )
    text = f"{Path(path).name} | updated {timestamp} | q/Esc closes"
    cv2.putText(
        image,
        text[:150],
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qt/OpenCV auto-refresh viewer for SG-Nav realtime_monitor images."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/realtime_sgnav/paper_frontier_planned_approach_v8/latest.jpg",
        help="Path to latest.jpg written by --realtime_monitor.",
    )
    parser.add_argument("--interval_ms", type=int, default=200)
    parser.add_argument("--title", default="SG-Nav Realtime Monitor")
    parser.add_argument("--max_width", type=int, default=1200)
    parser.add_argument("--max_height", type=int, default=800)
    parser.add_argument("--no_overlay", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.title, min(args.max_width, 1200), min(args.max_height, 800))

    last_mtime_ns = None
    last_frame = None
    while True:
        current_mtime_ns = None
        if image_path.exists():
            try:
                current_mtime_ns = image_path.stat().st_mtime_ns
            except OSError:
                current_mtime_ns = None

        if current_mtime_ns != last_mtime_ns or last_frame is None:
            if current_mtime_ns is None:
                frame = make_placeholder(
                    image_path,
                    "Waiting for realtime image. Start SG-Nav with --realtime_monitor.",
                )
            else:
                frame = read_image(image_path)
                if frame is None:
                    frame = make_placeholder(
                        image_path,
                        "Image exists but is not readable yet. Waiting...",
                    )
                elif not args.no_overlay:
                    frame = add_overlay(frame, image_path, current_mtime_ns)
            frame = resize_to_fit(frame, args.max_width, args.max_height)
            last_frame = frame
            last_mtime_ns = current_mtime_ns

        cv2.imshow(args.title, last_frame)
        key = cv2.waitKey(max(1, int(args.interval_ms))) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
