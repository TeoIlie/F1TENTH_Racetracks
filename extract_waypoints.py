#!/usr/bin/env python3
"""
Extract centerline waypoints from a track map image.

This script implements steps 2.1-2.4 of the Drift map extraction plan:
1. Load and prepare the track image
2. Extract skeleton (medial axis) as centerline
3. Order skeleton pixels into sequential path
4. Measure path length and calculate required waypoints

Usage:
    python3 extract_waypoints.py --map Drift [--visualize] [--spacing SPACING_VALUE]
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

# Default flag values
DEFAULT_MAP = "Drift"
DEFAULT_SPACING = 1.0


def load_and_prepare_image(map_path, map_name):
    """
    Step 2.1: Load and prepare track image.

    Args:
        map_path: Path to map directory

    Returns:
        track_mask: Binary numpy array (255=track, 0=walls)
        img_array: Original grayscale image array
    """
    print("Step 2.1: Loading and preparing image...")

    img_path = map_path / f"{map_name}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Map image not found: {img_path}")

    # Load image
    img = Image.open(img_path)
    img_gray = img.convert("L")  # Convert to grayscale
    img_array = np.array(img_gray)

    print(f"  Image size: {img_array.shape[1]} × {img_array.shape[0]} pixels")

    # Binarize: track=255 (white), walls=0 (black)
    track_mask = np.where(img_array > 127, 255, 0).astype(np.uint8)

    track_pixels = np.sum(track_mask > 0)
    total_pixels = track_mask.size
    track_percent = 100.0 * track_pixels / total_pixels

    print(f"  Track pixels: {track_pixels:,} ({track_percent:.1f}%)")

    # Morphological opening to remove noise, thin peninsulas, and ragged edges
    # before skeletonization. Matches the race stack's filter_map_occupancy_grid().
    kernel = np.ones((9, 9), np.uint8)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    filtered_pixels = np.sum(track_mask > 0)
    removed = track_pixels - filtered_pixels
    print(f"  After morphological opening (9x9, 2 iter): {filtered_pixels:,} pixels ({removed:,} removed)")

    return track_mask, img_array


def extract_skeleton(track_mask):
    """
    Step 2.2: Extract skeleton (centerline) using morphological skeletonization.

    Args:
        track_mask: Binary numpy array (255=track, 0=walls)

    Returns:
        skeleton: uint8 numpy array (255=centerline pixels, 0=background)
    """
    print("\nStep 2.2: Extracting skeleton...")

    # skeletonize expects a boolean or 0/1 array
    skeleton_bool = skeletonize(track_mask > 0)
    skeleton = (skeleton_bool * 255).astype(np.uint8)

    skeleton_pixels = np.sum(skeleton > 0)
    print(f"  Skeleton pixels: {skeleton_pixels:,}")

    return skeleton


def extract_centerline_contour(skeleton, map_resolution, expected_length_m=0.0):
    """
    Step 2.3: Extract centerline as an ordered closed contour from the skeleton.

    Uses OpenCV contour finding on the skeleton image to directly obtain closed,
    sequentially-ordered loops. Selects the contour whose arc length best matches
    the expected track length (within ±15%). This replaces the previous greedy
    neighbor-walking SkeletonTracer, which could fail at junctions and did not
    guarantee a closed path.

    Based on the ForzaETH race stack's extract_centerline() in global_planner_utils.py.

    Args:
        skeleton: uint8 numpy array (255=skeleton, 0=background)
        map_resolution: Meters per pixel
        expected_length_m: Expected centerline length in meters (0 = accept any)

    Returns:
        ordered_path: Numpy array of shape (N, 2) with (y, x) coordinates
    """
    print("\nStep 2.3: Extracting centerline via contour detection...")

    # Find all contours with hierarchy info
    contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None or len(contours) == 0:
        raise IOError("No contours found in skeleton image")

    # Keep only closed contours (those that have parent or child in the hierarchy)
    closed_contours = []
    for i, cont in enumerate(contours):
        opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
        if not opened:
            closed_contours.append(cont)

    print(f"  Total contours: {len(contours)}, closed: {len(closed_contours)}")

    if len(closed_contours) == 0:
        # Fallback: if no contours are detected as closed by hierarchy,
        # use all contours (some maps produce only top-level contours)
        print("  WARNING: No closed contours found via hierarchy, using all contours")
        closed_contours = list(contours)

    # Calculate arc length of each contour and match against expected length
    contour_lengths_m = []
    for cont in closed_contours:
        length_px = cv2.arcLength(cont, closed=True)
        contour_lengths_m.append(length_px * map_resolution)

    print(f"  Contour lengths (m): {[f'{l:.1f}' for l in contour_lengths_m]}")

    # Select the best contour
    valid_indices = []
    if expected_length_m > 0:
        # Filter to contours within ±15% of expected length
        for i, length_m in enumerate(contour_lengths_m):
            if abs(expected_length_m / length_m - 1.0) < 0.15:
                valid_indices.append(i)

    if len(valid_indices) == 0:
        # No length filter or nothing matched: accept all
        valid_indices = list(range(len(closed_contours)))

    # Take the shortest valid contour (innermost loop = centerline)
    best_idx = min(valid_indices, key=lambda i: contour_lengths_m[i])
    best_contour = closed_contours[best_idx]
    best_length_m = contour_lengths_m[best_idx]

    print(f"  Selected contour: {len(best_contour)} points, {best_length_m:.1f} m")

    # OpenCV contours are shape (N, 1, 2) with (x, y) — convert to (N, 2) with (y, x)
    contour_flat = best_contour.reshape(-1, 2)  # (N, 2) as (x, y)
    ordered_path = contour_flat[:, ::-1]  # flip to (y, x) to match rest of pipeline

    return ordered_path


def smooth_centerline(centerline):
    """
    Smooth the centerline with a two-pass Savitzky-Golay filter.

    The savgol filter doesn't ensure a smooth transition at the end and beginning
    of the centerline. That's why we apply the filter a second time with start and
    end points on the other half of the track, then stitch the boundary regions from
    the second pass into the first to get an overall smooth closed centerline.

    Based on the ForzaETH race stack implementation in global_planner_utils.py.

    Args:
        centerline: Numpy array of shape (N, 2) with (y, x) coordinates

    Returns:
        centerline_smooth: Smoothed centerline, same shape
    """
    print("\nSmoothing centerline (two-pass Savitzky-Golay)...")

    centerline_length = len(centerline)

    # Adaptive filter length based on number of points
    if centerline_length > 2000:
        filter_length = int(centerline_length / 200) * 10 + 1
    elif centerline_length > 1000:
        filter_length = 81
    elif centerline_length > 500:
        filter_length = 41
    else:
        filter_length = 21

    print(f"  Centerline points: {centerline_length}")
    print(f"  Filter length: {filter_length} (polyorder=3)")

    # Pass 1: smooth the centerline as-is
    centerline_smooth = savgol_filter(centerline, filter_length, 3, axis=0)

    # Pass 2: shift centerline by half its length so the original start/end seam
    # is now in the middle (far from filter boundaries), then smooth again
    cen_len = centerline_length // 2
    centerline_shifted = np.append(centerline[cen_len:], centerline[:cen_len], axis=0)
    centerline_smooth_shifted = savgol_filter(centerline_shifted, filter_length, 3, axis=0)

    # Stitch: take the boundary regions (first and last filter_length points)
    # from the second pass, which are smooth at the original seam location
    centerline_smooth[:filter_length] = centerline_smooth_shifted[cen_len : (cen_len + filter_length)]
    centerline_smooth[-filter_length:] = centerline_smooth_shifted[(cen_len - filter_length) : cen_len]

    print("  Two-pass smoothing complete (seam region stitched)")

    return centerline_smooth


def measure_path_and_calculate_waypoints(ordered_path, resolution, target_spacing):
    """
    Step 2.4: Measure path length and calculate required number of waypoints.

    Args:
        ordered_path: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        resolution: Meters per pixel
        target_spacing: Target spacing between waypoints in meters

    Returns:
        Dictionary with:
            - total_length_px: Total path length in pixels
            - total_length_m: Total path length in meters
            - cumulative_distance_px: Cumulative distance array
            - num_waypoints: Recommended number of waypoints
            - waypoint_spacing_m: Actual spacing that will be used
    """
    print("\nStep 2.4: Measuring path length and calculating waypoint count...")

    # Calculate distances between consecutive points
    # ordered_path is (y, x), so we need Euclidean distance
    deltas = np.diff(ordered_path, axis=0)  # Shape: (N-1, 2)
    distances = np.sqrt(np.sum(deltas**2, axis=1))  # Shape: (N-1,)

    # Cumulative distance
    cumulative_distance_px = np.concatenate([[0], np.cumsum(distances)])
    total_length_px = cumulative_distance_px[-1]

    # Convert to meters
    total_length_m = total_length_px * resolution

    print(f"  Total path length: {total_length_px:.1f} pixels = {total_length_m:.2f} meters")

    # Calculate number of waypoints
    num_waypoints = int(np.round(total_length_m / target_spacing))

    # Ensure at least minimum number of waypoints
    if num_waypoints < 10:
        print(f"  WARNING: Only {num_waypoints} waypoints calculated.")
        print("           Using minimum of 10 waypoints.")
        num_waypoints = 10

    # Calculate actual spacing that will be achieved
    actual_spacing = total_length_m / num_waypoints

    print(f"  Target waypoint spacing: {target_spacing:.3f} meters")
    print(f"  Number of waypoints: {num_waypoints}")
    print(f"  Actual waypoint spacing: {actual_spacing:.3f} meters")

    # Check loop closure (distance from last point to first)
    loop_closure_px = np.linalg.norm(ordered_path[-1] - ordered_path[0])
    loop_closure_m = loop_closure_px * resolution

    print(f"  Loop closure gap: {loop_closure_px:.1f} pixels = {loop_closure_m:.3f} meters")

    if loop_closure_m > 2 * target_spacing:
        print("  WARNING: Large loop closure gap detected!")
        print("           Track may not be a closed loop.")

    return {
        "total_length_px": total_length_px,
        "total_length_m": total_length_m,
        "cumulative_distance_px": cumulative_distance_px,
        "num_waypoints": num_waypoints,
        "waypoint_spacing_m": actual_spacing,
        "loop_closure_m": loop_closure_m,
    }


def interpolate_centerline(ordered_path, target_spacing_px):
    """
    Step 2.5a: Interpolate ordered path at uniform arc-length spacing using
    periodic cubic splines (C2 continuous at closure).

    Replaces the previous every-Nth-point subsampling, which produced uneven
    spacing (diagonal skeleton pixels are ~1.41x farther apart). A periodic
    CubicSpline ensures C2 smoothness everywhere, including at the lap
    closure point.

    Args:
        ordered_path: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        target_spacing_px: Desired spacing between waypoints in pixels

    Returns:
        interpolated_path: Numpy array of shape (M, 2) with uniformly-spaced (y, x)
    """
    print("\nStep 2.5a: Arc-length interpolation (periodic CubicSpline)...")

    # Compute cumulative arc length along the path
    deltas = np.diff(ordered_path, axis=0)
    segment_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    s = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = s[-1]

    print(f"  Total arc length: {total_length:.1f} pixels")
    print(f"  Target spacing: {target_spacing_px:.2f} pixels")

    # Close the loop: append the first point at the end so the spline wraps
    path_closed = np.append(ordered_path, [ordered_path[0]], axis=0)
    closure_gap = np.linalg.norm(ordered_path[-1] - ordered_path[0])
    s_closed = np.append(s, total_length + closure_gap)

    # Build periodic cubic splines for y and x
    cs_y = CubicSpline(s_closed, path_closed[:, 0], bc_type="periodic")
    cs_x = CubicSpline(s_closed, path_closed[:, 1], bc_type="periodic")

    # Sample at uniform arc-length intervals (exclude last point = duplicate of first)
    total_closed_length = s_closed[-1]
    num_waypoints = max(10, int(np.round(total_closed_length / target_spacing_px)))
    s_uniform = np.linspace(0, total_closed_length, num_waypoints, endpoint=False)

    interpolated_y = cs_y(s_uniform)
    interpolated_x = cs_x(s_uniform)
    interpolated_path = np.column_stack([interpolated_y, interpolated_x])

    # Verify spacing uniformity
    check_deltas = np.diff(interpolated_path, axis=0)
    check_dists = np.sqrt(np.sum(check_deltas**2, axis=1))
    actual_spacing = np.mean(check_dists)

    print(f"  Interpolated to {num_waypoints} waypoints")
    print(f"  Actual mean spacing: {actual_spacing:.2f} pixels")
    print(f"  Spacing std: {np.std(check_dists):.4f} pixels")

    return interpolated_path


def calculate_track_widths(waypoints_px, track_mask, resolution):
    """
    Step 2.5b: Calculate track width at each waypoint using distance transform.

    Uses distance transform for efficient width calculation. The distance transform
    gives the distance from each track pixel to the nearest boundary.

    Args:
        waypoints_px: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        track_mask: Binary track mask (1=track, 0=walls)
        resolution: Meters per pixel

    Returns:
        Tuple of (w_tr_right, w_tr_left) as numpy arrays in meters
    """
    print("\nStep 2.5b: Calculating track widths using distance transform...")

    # Compute distance transform: value = distance to nearest boundary (in pixels)
    distance_map = distance_transform_edt(track_mask > 0)

    w_tr_right = []
    w_tr_left = []

    for i, (y_px, x_px) in enumerate(waypoints_px):
        # Sample distance map value at waypoint
        # This gives distance to nearest boundary in pixels
        center_dist_px = distance_map[int(y_px), int(x_px)]

        # Convert to meters
        # Using symmetric width (simplified approach)
        width_m = center_dist_px * resolution

        w_tr_right.append(width_m)
        w_tr_left.append(width_m)

    w_tr_right = np.array(w_tr_right)
    w_tr_left = np.array(w_tr_left)

    avg_width = np.mean(w_tr_right) + np.mean(w_tr_left)
    print(f"  Average track width: {avg_width:.3f} meters")
    print(f"  Width range: {2 * np.min(w_tr_right):.3f} - {2 * np.max(w_tr_right):.3f} meters")

    return w_tr_right, w_tr_left


def convert_to_world_coordinates(waypoints_px, image_shape, resolution, origin):
    """
    Step 2.6: Convert pixel coordinates to world coordinates.

    Transforms from image pixel coordinates to simulation world coordinates.
    Critical: y-axis flip is required because image y=0 is top,
    but world y increases upward.

    Args:
        waypoints_px: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        image_shape: Tuple of (height, width) of the image
        resolution: Meters per pixel
        origin: Origin from YAML file [x, y, z]

    Returns:
        waypoints_world: Numpy array of shape (N, 2) with (x, y) in meters
    """
    print("\nStep 2.6: Converting to world coordinates...")

    image_height, image_width = image_shape

    waypoints_world = []

    for y_px, x_px in waypoints_px:
        # Image coordinates: origin top-left, y-axis down
        # World coordinates: origin at map center, y-axis up

        # Flip y-axis
        y_flipped = (image_height - 1) - y_px

        # Convert to world coordinates
        x_m = origin[0] + x_px * resolution
        y_m = origin[1] + y_flipped * resolution

        waypoints_world.append([x_m, y_m])

    waypoints_world = np.array(waypoints_world)

    print("  World coordinate range:")
    print(f"    X: [{np.min(waypoints_world[:, 0]):.3f}, {np.max(waypoints_world[:, 0]):.3f}] meters")
    print(f"    Y: [{np.min(waypoints_world[:, 1]):.3f}, {np.max(waypoints_world[:, 1]):.3f}] meters")

    return waypoints_world


def write_centerline_csv(waypoints_world, w_tr_right, w_tr_left, output_path):
    """
    Step 2.7: Generate centerline CSV file.

    Writes waypoints in the format expected by the F1TENTH Gym simulator.

    Args:
        waypoints_world: Numpy array of shape (N, 2) with (x, y) in meters
        w_tr_right: Right track widths in meters
        w_tr_left: Left track widths in meters
        output_path: Path to output CSV file
    """
    print("\nStep 2.7: Writing centerline CSV...")

    with open(output_path, "w", newline="") as f:
        # Write header comment line directly (not using csv.writer to avoid quotes)
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")

        writer = csv.writer(f)

        # Write waypoints
        for i in range(len(waypoints_world)):
            x_m, y_m = waypoints_world[i]
            w_right = w_tr_right[i]
            w_left = w_tr_left[i]

            writer.writerow([f"{x_m:.4f}", f"{y_m:.4f}", f"{w_right:.1f}", f"{w_left:.1f}"])

        # Write empty line at end (to match format of other tracks)
        writer.writerow([])

    print(f"  Saved centerline to: {output_path}")
    print(f"  Total waypoints: {len(waypoints_world)}")


def validate_centerline(waypoints_world):
    """
    Step 3.2: Verify track properties.

    Args:
        waypoints_world: Numpy array of shape (N, 2) with (x, y) in meters

    Returns:
        Dictionary with validation metrics
    """
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check waypoint spacing
    spacings = np.sqrt(np.sum(np.diff(waypoints_world, axis=0) ** 2, axis=1))
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    variation_pct = 100 * std_spacing / mean_spacing

    print("Waypoint spacing:")
    print(f"  Mean: {mean_spacing:.4f} m")
    print(f"  Std:  {std_spacing:.4f} m")
    print(f"  Variation: {variation_pct:.1f}%")
    print(f"  Range: [{np.min(spacings):.4f}, {np.max(spacings):.4f}] m")

    # Check loop closure
    loop_gap = np.linalg.norm(waypoints_world[-1] - waypoints_world[0])
    print(f"\nLoop closure gap: {loop_gap:.4f} m")

    if loop_gap > 0.1:
        print("  WARNING: Loop closure gap is large!")
    else:
        print("  ✓ Excellent loop closure")

    # Check total length
    total_length = np.sum(spacings) + loop_gap
    print(f"\nTotal track length: {total_length:.2f} m")

    print("=" * 70)

    return {
        "mean_spacing": mean_spacing,
        "std_spacing": std_spacing,
        "variation_pct": variation_pct,
        "loop_gap": loop_gap,
        "total_length": total_length,
    }


def visualize_centerline(img_array, waypoints_px, waypoints_world, origin, resolution, output_path):
    """
    Step 3.1: Visualize waypoints on image.

    Creates visualization showing the final centerline overlaid on the track image.

    Args:
        img_array: Original grayscale image
        waypoints_px: Waypoint coordinates in pixels (y, x)
        waypoints_world: Waypoint coordinates in world frame (x, y)
        origin: Origin from YAML
        resolution: Meters per pixel
        output_path: Path to save visualization
    """
    print("\nCreating centerline visualization...")

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img_array, cmap="gray")

    # Plot the centerline
    ax.plot(waypoints_px[:, 1], waypoints_px[:, 0], "r-", linewidth=2, label="Centerline")

    # Mark start point (green)
    ax.scatter(
        waypoints_px[0, 1],
        waypoints_px[0, 0],
        c="green",
        s=100,
        marker="o",
        zorder=10,
        label="Start",
    )

    # Mark end point (blue)
    ax.scatter(
        waypoints_px[-1, 1],
        waypoints_px[-1, 0],
        c="blue",
        s=100,
        marker="s",
        zorder=10,
        label="End",
    )

    # Mark every 10th waypoint for density visualization
    every_n = max(1, len(waypoints_px) // 30)
    ax.scatter(
        waypoints_px[::every_n, 1],
        waypoints_px[::every_n, 0],
        c="yellow",
        s=20,
        marker="o",
        alpha=0.6,
        zorder=5,
    )

    ax.set_title(f"Drift Track Centerline ({len(waypoints_px)} waypoints)")
    ax.legend()
    ax.axis("equal")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved visualization to: {output_path}")


def visualize_skeleton(img_array, skeleton, ordered_path, output_path):
    """
    Visualize the skeleton and ordered path overlaid on the original image.

    Args:
        img_array: Original grayscale image
        skeleton: Binary skeleton array
        ordered_path: Ordered path array (y, x)
        output_path: Path to save visualization
    """
    print("\nCreating skeleton visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Original Track Image")
    axes[0].axis("equal")

    # Skeleton overlay
    axes[1].imshow(img_array, cmap="gray")
    skeleton_bool = skeleton > 0
    skeleton_overlay = np.zeros((*skeleton.shape, 3))
    skeleton_overlay[skeleton_bool, 0] = 1  # Red for skeleton
    axes[1].imshow(skeleton_overlay, alpha=0.6)
    axes[1].set_title(f"Skeleton ({np.sum(skeleton_bool)} pixels)")
    axes[1].axis("equal")

    # Ordered path
    axes[2].imshow(img_array, cmap="gray")
    # Plot path as line
    axes[2].plot(ordered_path[:, 1], ordered_path[:, 0], "r-", linewidth=2, label="Ordered Path")
    # Mark start point
    axes[2].scatter(
        ordered_path[0, 1],
        ordered_path[0, 0],
        c="green",
        s=100,
        marker="o",
        zorder=10,
        label="Start",
    )
    # Mark end point
    axes[2].scatter(
        ordered_path[-1, 1],
        ordered_path[-1, 0],
        c="blue",
        s=100,
        marker="s",
        zorder=10,
        label="End",
    )
    axes[2].set_title(f"Ordered Path ({len(ordered_path)} points)")
    axes[2].legend()
    axes[2].axis("equal")

    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved visualization to: {output_path}")

    # Show plot if running interactively
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Extract centerline waypoints from track map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP,
        help=f"Map name (default: {DEFAULT_MAP})",
    )
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help=f"Target waypoint spacing in meters (default: {DEFAULT_SPACING})",
    )

    args = parser.parse_args()

    # Paths
    map_name = args.map
    script_dir = Path(__file__).parent
    map_path = script_dir / map_name

    if not map_path.exists():
        print(f"ERROR: Map directory not found: {map_path}")
        return 1

    print("=" * 70)
    print(f"EXTRACTING WAYPOINTS FOR {map_name.upper()} TRACK")
    print("=" * 70)

    # Load map config to get resolution
    yaml_path = map_path / f"{map_name}_map.yaml"
    if not yaml_path.exists():
        print(f"ERROR: Map YAML not found: {yaml_path}")
        return 1

    # Parse YAML to get resolution and origin
    resolution = None
    origin = None
    with open(yaml_path, "r") as f:
        for line in f:
            if line.startswith("resolution:"):
                resolution = float(line.split(":")[1].strip())
            elif line.startswith("origin:"):
                # Parse origin list [x, y, z]
                origin_str = line.split(":", 1)[1].strip()
                origin = eval(origin_str)  # Parse Python list literal

    if resolution is None:
        print("ERROR: Resolution not found in the yaml file")
        return 1

    if origin is None:
        print("ERROR: Origin not found in the yaml file")
        return 1

    print(f"\nMap: {map_name}")
    print(f"Resolution: {resolution:.6f} m/px")
    print(f"Origin: {origin}")
    print(f"Target spacing: {args.spacing} m")

    # Step 2.1: Load and prepare image
    track_mask, img_array = load_and_prepare_image(map_path, map_name)

    # Step 2.2: Extract skeleton
    skeleton = extract_skeleton(track_mask)

    # Step 2.3: Extract centerline as closed contour
    ordered_path = extract_centerline_contour(skeleton, resolution)

    # Step 2.3b: Smooth the ordered path (two-pass Savitzky-Golay)
    ordered_path = smooth_centerline(ordered_path)

    # Step 2.4: Measure path and report diagnostics
    _ = measure_path_and_calculate_waypoints(ordered_path, resolution, args.spacing)

    # Step 2.5: Arc-length interpolation and calculate track widths
    target_spacing_px = args.spacing / resolution  # convert meters to pixels
    waypoints_px = interpolate_centerline(ordered_path, target_spacing_px)
    w_tr_right, w_tr_left = calculate_track_widths(waypoints_px, track_mask, resolution)

    # Step 2.6: Convert to world coordinates
    waypoints_world = convert_to_world_coordinates(waypoints_px, img_array.shape, resolution, origin)

    # Step 2.7: Write centerline CSV
    csv_path = map_path / f"{map_name}_centerline.csv"
    write_centerline_csv(waypoints_world, w_tr_right, w_tr_left, csv_path)

    # Step 3.2: Validation
    validation_metrics = validate_centerline(waypoints_world)

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Track length:        {validation_metrics['total_length']:.2f} meters")
    print(f"Waypoints created:   {len(waypoints_world)}")
    print(
        f"Waypoint spacing:    {validation_metrics['mean_spacing']:.4f} ± {validation_metrics['std_spacing']:.4f} meters"
    )
    print(f"Loop closure gap:    {validation_metrics['loop_gap']:.4f} meters")
    print(f"\nOutput file:         {csv_path}")
    print("=" * 70)

    # Visualizations
    if args.visualize:
        # Create generation subfolder
        generation_path = map_path / "generation"
        generation_path.mkdir(exist_ok=True)

        # Skeleton extraction visualization
        skeleton_vis_path = generation_path / "skeleton_extraction.png"
        visualize_skeleton(img_array, skeleton, ordered_path, skeleton_vis_path)

        # Final centerline visualization
        centerline_vis_path = generation_path / "centerline_final.png"
        visualize_centerline(
            img_array,
            waypoints_px,
            waypoints_world,
            origin,
            resolution,
            centerline_vis_path,
        )

    print("\nAll steps (2.1-2.7) complete!")
    print(f"Centerline saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    exit(main())
