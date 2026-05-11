import argparse
import os

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None


def clamp_pixel(value, map_size):
    return int(min(max(int(float(value)), 0), map_size - 1))


def goal_gps_to_map_xy(goal_gps, map_size_cm=4000, resolution=5, map_size=800):
    goal_gps = [float(goal_gps[0]), float(goal_gps[1])]
    x = clamp_pixel(map_size_cm / 10 + goal_gps[0] * 100 / resolution, map_size)
    y = clamp_pixel(map_size_cm / 10 + goal_gps[1] * 100 / resolution, map_size)
    return make_pair([x, y])


def goal_gps_to_map_rc(goal_gps, map_size_cm=4000, resolution=5, map_size=800):
    xy = goal_gps_to_map_xy(goal_gps, map_size_cm, resolution, map_size)
    return make_pair([xy[1], xy[0]])


def object_detection_gps_to_frontier_rc(obj_gps, map_size_cm=4000, resolution=5):
    obj_gps = [float(obj_gps[0]), float(obj_gps[1])]
    row = int(map_size_cm / 10 - obj_gps[1] * 100 / resolution)
    col = int(map_size_cm / 10 + obj_gps[0] * 100 / resolution)
    return make_pair([row, col])


def scenegraph_center_world_to_xy(center_xy_m, map_resolution=5, map_size=800):
    center_xy_m = [float(center_xy_m[0]), float(center_xy_m[1])]
    x = int(center_xy_m[0] * 100 / map_resolution)
    y = int(center_xy_m[1] * 100 / map_resolution)
    y = map_size - 1 - y
    return make_pair([x, y])


def make_pair(values):
    if np is not None:
        return np.array(values, dtype=np.int32)
    return [int(values[0]), int(values[1])]


def pair_to_list(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return [int(values[0]), int(values[1])]


def draw_overlay(path, map_size, agent_rc, selected_frontier_rc, candidate_goal_rc):
    if np is None:
        return draw_overlay_ppm(path, map_size, agent_rc, selected_frontier_rc, candidate_goal_rc)

    image = np.full((map_size, map_size, 3), 245, dtype=np.uint8)
    for row in range(0, map_size, 50):
        image[row : row + 1, :, :] = (220, 220, 220)
    for col in range(0, map_size, 50):
        image[:, col : col + 1, :] = (220, 220, 220)

    points = [
        (agent_rc, (40, 170, 40), "agent_rc"),
        (selected_frontier_rc, (255, 140, 30), "selected_frontier_rc"),
        (candidate_goal_rc, (210, 40, 210), "candidate_goal_rc"),
    ]
    for point, color, label in points:
        row, col = int(point[0]), int(point[1])
        if 0 <= row < map_size and 0 <= col < map_size:
            rr0 = max(row - 6, 0)
            rr1 = min(row + 7, map_size)
            cc0 = max(col - 6, 0)
            cc1 = min(col + 7, map_size)
            image[rr0:rr1, cc0:cc1, :] = color
            if cv2 is not None:
                cv2.putText(
                    image,
                    label,
                    (min(col + 8, map_size - 160), max(row - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(path, image)
        return path
    fallback_path = os.path.splitext(path)[0] + ".ppm"
    with open(fallback_path, "wb") as handle:
        handle.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        handle.write(image.astype(np.uint8).tobytes())
    return fallback_path


def draw_overlay_ppm(path, map_size, agent_rc, selected_frontier_rc, candidate_goal_rc):
    fallback_path = os.path.splitext(path)[0] + ".ppm"
    directory = os.path.dirname(fallback_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pixels = bytearray([245]) * (map_size * map_size * 3)

    def set_pixel(row, col, color):
        if 0 <= row < map_size and 0 <= col < map_size:
            idx = (row * map_size + col) * 3
            pixels[idx : idx + 3] = bytes(color)

    for row in range(0, map_size, 50):
        for col in range(map_size):
            set_pixel(row, col, (220, 220, 220))
    for col in range(0, map_size, 50):
        for row in range(map_size):
            set_pixel(row, col, (220, 220, 220))

    for point, color in [
        (agent_rc, (40, 170, 40)),
        (selected_frontier_rc, (255, 140, 30)),
        (candidate_goal_rc, (210, 40, 210)),
    ]:
        row, col = int(point[0]), int(point[1])
        for rr in range(max(row - 6, 0), min(row + 7, map_size)):
            for cc in range(max(col - 6, 0), min(col + 7, map_size)):
                set_pixel(rr, cc, color)

    with open(fallback_path, "wb") as handle:
        handle.write(f"P6\n{map_size} {map_size}\n255\n".encode("ascii"))
        handle.write(pixels)
    return fallback_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_size_cm", type=int, default=4000)
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--goal_gps", nargs=2, type=float, default=[0.25, -0.15])
    parser.add_argument("--agent_gps", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--selected_frontier_rc", nargs=2, type=int, default=[410, 420])
    parser.add_argument(
        "--output_png",
        type=str,
        default="data/debug_fbe/coordinate_consistency.png",
    )
    args = parser.parse_args()

    map_size = args.map_size_cm // args.resolution
    goal_xy = goal_gps_to_map_xy(args.goal_gps, args.map_size_cm, args.resolution, map_size)
    goal_rc = goal_gps_to_map_rc(args.goal_gps, args.map_size_cm, args.resolution, map_size)
    agent_rc = goal_gps_to_map_rc(args.agent_gps, args.map_size_cm, args.resolution, map_size)
    selected_frontier_rc = make_pair(args.selected_frontier_rc)

    assert int(goal_rc[0]) == int(goal_xy[1])
    assert int(goal_rc[1]) == int(goal_xy[0])

    obj_rc = object_detection_gps_to_frontier_rc(
        args.goal_gps,
        args.map_size_cm,
        args.resolution,
    )
    sg_xy = scenegraph_center_world_to_xy(
        [
            args.map_size_cm / 100.0 / 2.0 + args.goal_gps[0],
            args.map_size_cm / 100.0 / 2.0 - args.goal_gps[1],
        ],
        args.resolution,
        map_size,
    )

    overlay_path = draw_overlay(args.output_png, map_size, agent_rc, selected_frontier_rc, goal_rc)
    print("[OK] coordinate consistency checks passed")
    print(f"goal_xy={pair_to_list(goal_xy)} goal_rc={pair_to_list(goal_rc)}")
    print(
        "object_detection_gps_to_frontier_rc="
        f"{pair_to_list(obj_rc)} scenegraph_center_world_to_xy={pair_to_list(sg_xy)}"
    )
    print(f"overlay_path={overlay_path}")


if __name__ == "__main__":
    main()
