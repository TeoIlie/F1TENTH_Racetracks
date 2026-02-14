"""
Extract global track normalization bounds across all available maps.

This script scans the maps/ directory for all track folders and computes
global normalization bounds for observation normalization.

Note "Drift" map has very different proportions than other tracks, and
is therefore ignored.

Usage:
    python maps/extract_global_track_norm_bounds.py

After running, update the constants in f1tenth_gym/envs/utils.py:
    GLOBAL_MAX_CURVATURE
    GLOBAL_MIN_WIDTH
    GLOBAL_MAX_WIDTH
"""

import sys
from pathlib import Path

# Add parent directory to path to import training_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train_utils import compute_global_track_bounds


def get_all_track_names() -> list[str]:
    """Extract all track names from subdirectories in maps/ folder."""
    maps_dir = Path(__file__).parent
    track_names = []

    for subdir in maps_dir.iterdir():
        track_name = subdir.name
        if subdir.is_dir() and not track_name.startswith(".") and track_name != "Drift":
            track_names.append(track_name)

    return sorted(track_names)


if __name__ == "__main__":
    print("Scanning maps/ directory for available tracks...")
    track_names = get_all_track_names()
    print(f"Found {len(track_names)} tracks: {', '.join(track_names)}\n")

    # Compute global bounds across all tracks
    bounds = compute_global_track_bounds(track_names)

    print("Done! Copy the values above to f1tenth_gym/envs/utils.py")
