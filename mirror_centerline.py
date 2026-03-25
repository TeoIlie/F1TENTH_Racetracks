#!/usr/bin/env python3
"""
Generate a mirrored centerline CSV by transforming the original map's centerline.

Instead of re-extracting waypoints from a mirrored image (which can produce
different/noisier contours), this script mathematically transforms the original
centerline to match a horizontally-flipped (imagemagick -flop) image.

Transformation for horizontal flip:
  - Reverse waypoint order (track traversal direction flips)
  - Mirror x-coordinates: x_mirror = 2*origin_x + (width-1)*resolution - x_original
  - y-coordinates unchanged
  - Track widths unchanged (symmetric from distance transform)

Usage:
    python3 mirror_centerline.py --map Drift_large
    python3 mirror_centerline.py --map Drift2
    python3 mirror_centerline.py --map Drift2 --smooth 11
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import savgol_filter


def main():
    parser = argparse.ArgumentParser(
        description="Generate mirrored centerline from original map's centerline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--map", type=str, required=True, help="Original map name (e.g. Drift_large)")
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Savitzky-Golay filter window length (odd integer, 0=off). "
        "Try 7-15 for light-heavy smoothing. Uses two-pass technique for seamless loop closure.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    map_name = args.map
    mirror_name = f"{map_name}_mirror"

    # Paths
    original_dir = script_dir / map_name
    mirror_dir = script_dir / mirror_name
    original_csv = original_dir / f"{map_name}_centerline.csv"
    mirror_csv = mirror_dir / f"{mirror_name}_centerline.csv"
    original_yaml = original_dir / f"{map_name}_map.yaml"
    original_img = original_dir / f"{map_name}.png"

    # Validate paths
    for path, desc in [
        (original_dir, "Original map directory"),
        (mirror_dir, "Mirror map directory"),
        (original_csv, "Original centerline CSV"),
        (original_yaml, "Original YAML"),
        (original_img, "Original image"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    # Parse resolution and origin from YAML
    resolution = None
    origin = None
    with open(original_yaml, "r") as f:
        for line in f:
            if line.startswith("resolution:"):
                resolution = float(line.split(":")[1].strip())
            elif line.startswith("origin:"):
                origin_str = line.split(":", 1)[1].strip()
                origin = eval(origin_str)

    if resolution is None or origin is None:
        raise ValueError("Could not parse resolution and origin from YAML")

    # Get image width
    img = Image.open(original_img)
    width = img.size[0]

    print(f"Original map: {map_name}")
    print(f"Resolution: {resolution} m/px")
    print(f"Origin: {origin}")
    print(f"Image width: {width} px")

    # Load original centerline
    waypoints = []
    with open(original_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) >= 4:
                waypoints.append([float(v) for v in row])

    waypoints = np.array(waypoints)
    print(f"Original waypoints: {len(waypoints)}")

    # Transform: mirror x-coordinates
    # x_mirror = 2*origin_x + (width-1)*resolution - x_original
    mirror_constant = 2 * origin[0] + (width - 1) * resolution
    waypoints_mirror = waypoints.copy()
    waypoints_mirror[:, 0] = mirror_constant - waypoints[:, 0]

    # Reverse waypoint order (track traversal direction flips)
    waypoints_mirror = waypoints_mirror[::-1]

    print(f"Mirror constant (2*ox + (w-1)*res): {mirror_constant:.6f}")
    print(f"Original x range: [{waypoints[:, 0].min():.3f}, {waypoints[:, 0].max():.3f}]")
    print(f"Mirror x range:   [{waypoints_mirror[:, 0].min():.3f}, {waypoints_mirror[:, 0].max():.3f}]")

    # Optional smoothing (two-pass savgol for seamless loop closure)
    if args.smooth > 0:
        window = args.smooth
        if window % 2 == 0:
            window += 1  # must be odd
        if window >= len(waypoints_mirror):
            window = len(waypoints_mirror) // 2 * 2 - 1  # largest valid odd window
        polyorder = min(3, window - 1)

        xy = waypoints_mirror[:, :2]
        n = len(xy)

        # Pass 1: smooth as-is
        xy_smooth = savgol_filter(xy, window, polyorder, axis=0)

        # Pass 2: shift by half so the original seam is in the middle, smooth again
        half = n // 2
        xy_shifted = np.roll(xy, -half, axis=0)
        xy_shifted_smooth = savgol_filter(xy_shifted, window, polyorder, axis=0)

        # Stitch: take boundary regions from pass 2 (smooth at original seam)
        xy_smooth[:window] = np.roll(xy_shifted_smooth, half, axis=0)[:window]
        xy_smooth[-window:] = np.roll(xy_shifted_smooth, half, axis=0)[-window:]

        waypoints_mirror[:, :2] = xy_smooth

        # Report smoothing effect
        diffs = np.diff(xy_smooth, axis=0)
        spacings = np.sqrt(np.sum(diffs**2, axis=1))
        print(f"\nSmoothing applied (window={window}, polyorder={polyorder})")
        print(f"  Post-smooth spacing: mean={np.mean(spacings):.4f} std={np.std(spacings):.4f}")

    # Write mirrored centerline CSV
    with open(mirror_csv, "w", newline="") as f:
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        writer = csv.writer(f)
        for row in waypoints_mirror:
            writer.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.1f}", f"{row[3]:.1f}"])
        writer.writerow([])

    print(f"\nSaved mirrored centerline to: {mirror_csv}")


if __name__ == "__main__":
    exit(main())
