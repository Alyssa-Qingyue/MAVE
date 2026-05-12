"""
Three-stage cleaning:
- Stage 1 (PPT): iterate full-frame screenshots (frames/minute_xx/t_*.jpg). If a timestamp is identified as PPT,
  all grids at that timestamp (all grid_yy under images) are excluded.
- Stage 2 (ICON): iterate remaining grid images (images/minute_xx/grid_yy/t_*.jpg, excluding Stage-1 timestamps).
  If identified as icon-like, exclude only that single frame (minute+grid+frame), not the entire timestamp.
- Stage 3 (BLACK): iterate remaining grid images again (also skipping Stage-1 timestamps).
  If identified as all-black, exclude only that single frame.
Outputs: ppt_excluded_times.json, icon_excluded_frames.json, black_excluded_frames.json.
Uses PIL + numpy only (no OpenCV).
"""
import os
import json

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise ImportError("Need PIL (Pillow) and numpy. pip install Pillow numpy")

# Stage 1: full-frame screenshot root (output from cut_video.py)
FRAMES_ROOT = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/frames_10s"

# Stage 2/3: grid image root (output from extract_faces.py, same format as preprocessor input)
IMAGE_ROOT = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/images_out" 

# Output root + video subfolder (cleaning results are written here)
OUTPUT_BASE = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class"
OUTPUT_VIDEO_SUBDIR = "output"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, OUTPUT_VIDEO_SUBDIR)
# Stage 1 output: exclude all grids for this timestamp
PPT_EXCLUDED_JSON = os.path.join(OUTPUT_DIR, "ppt_excluded_times.json")
# Stage 2 output: exclude only (minute, grid, frame), format {minute_id: {grid_id: [frame_id, ...]}, ...}
ICON_EXCLUDED_JSON = os.path.join(OUTPUT_DIR, "icon_excluded_frames.json")
# Stage 3 output: exclude all-black frames, same format as icon output
BLACK_EXCLUDED_JSON = os.path.join(OUTPUT_DIR, "black_excluded_frames.json")

# Detection parameters (tunable)
CENTER_CROP_RATIO = 0.6   # Center area ratio (for large uniform-color blocks)
STD_THRESHOLD = 32        # Region grayscale std below this is considered highly uniform (0-255)
STD_THRESHOLD_STRICT = 22 # Stricter uniformity threshold (for dark PPT backgrounds)
MEAN_WHITE_LOW = 170      # Background tends to be white/gray
# Top-bottom split: if both top bar and bottom area are uniform (e.g. dark top + light bottom), classify as PPT
TOP_RATIO = 0.40          # Top region height ratio
BOTTOM_RATIO = 0.45       # Bottom region height ratio (keep middle gap to reduce boundary effects)
# Whole-image "large monochrome/narrow-band": if most pixels fall in a narrow grayscale band, classify as PPT/slide
GLOBAL_STD_PPT = 55       # Potential PPT when global std is below this and ratio conditions are met
PIXEL_NEAR_MEDIAN_RATIO = 0.58  # Pixel ratio within [median-25, median+25]
PIXEL_NEAR_MEDIAN_HALF = 25     # Grayscale distance from median

# Example icon detection (placeholder avatar, logo, etc.): excluded like PPT
BORDER_RATIO = 0.04       # Outer border ratio (thick dark border)
BORDER_DARK_MEAN = 90     # Border grayscale mean below this implies dark border
INNER_LIGHT_MEAN = 170    # If border is dark and inner area is bright, likely white-background icon
WHITE_PIXEL_RATIO = 0.42  # High white-pixel (>=230) ratio + moderate black-pixel (<=100) ratio -> avatar placeholder
BLACK_PIXEL_RATIO_MIN = 0.04
BLACK_PIXEL_RATIO_MAX = 0.35
ICON_STD_LOW = 30        # Overall std range for avatar-like icons
ICON_STD_HIGH = 135
# Bimodal/logo: histogram is dominated by dark + bright regions
BIMODAL_DARK_BIN = 32     # Upper bound for dark region
BIMODAL_BRIGHT_BIN = 223  # Lower bound for bright region
BIMODAL_MIN_RATIO = 0.70  # If dark+bright ratio exceeds this, treat as bimodal


def load_grayscale_numpy(path: str) -> np.ndarray:
    """Read image with PIL and convert to grayscale numpy array; return None on failure."""
    try:
        with Image.open(path) as im:
            im = im.convert("L")  # Grayscale
        return np.array(im, dtype=np.float64)
    except Exception:
        return None


def _region_mean_std(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> tuple:
    """Return (mean, std) for specified rectangular region (y0:y1, x0:x1)."""
    region = arr[max(0, y0):min(arr.shape[0], y1), max(0, x0):min(arr.shape[1], x1)]
    if region.size == 0:
        return 0.0, 255.0
    return float(np.mean(region)), float(np.std(region))


def center_region_std(arr: np.ndarray, crop_ratio: float = CENTER_CROP_RATIO) -> tuple:
    """Take center region with `crop_ratio` and return (mean, std)."""
    h, w = arr.shape
    c = crop_ratio
    y0 = int(h * (1 - c) / 2)
    y1 = int(h * (1 + c) / 2)
    x0 = int(w * (1 - c) / 2)
    x1 = int(w * (1 + c) / 2)
    return _region_mean_std(arr, y0, y1, x0, x1)


def is_frame_likely_ppt(path: str) -> bool:
    """
    Determine whether a single frame is clearly PPT-like (frame-only, no temporal context).
    - Large uniform-color block: center region is uniform and white/gray or extremely uniform.
    - Two-tone top/bottom blocks: both top bar and bottom area are uniform (e.g. dark top + light bottom).
    - Narrow-band whole image: global std is low and most pixels are near median grayscale
      (typical slide where background dominates despite some text).
    Uses PIL + numpy only, without OpenCV.
    """
    arr = load_grayscale_numpy(path)
    if arr is None or arr.size == 0:
        return False
    h, w = arr.shape
    flat = arr.flatten()

    # 1) Large center monochrome block: center is highly uniform
    mean_c, std_c = center_region_std(arr, CENTER_CROP_RATIO)
    if std_c <= STD_THRESHOLD and MEAN_WHITE_LOW <= mean_c <= 255:
        return True
    if std_c <= STD_THRESHOLD_STRICT:
        return True

    # 2) Uniform top and bottom regions (e.g. title bar + blank content area)
    y_top = int(h * TOP_RATIO)
    y_bottom_start = int(h * (1 - BOTTOM_RATIO))
    _, std_top = _region_mean_std(arr, 0, y_top, 0, w)
    _, std_bottom = _region_mean_std(arr, y_bottom_start, h, 0, w)
    if std_top <= STD_THRESHOLD and std_bottom <= STD_THRESHOLD:
        return True

    # 3) Whole-image large monochrome / narrow band:
    # most pixels near median and low overall contrast (typical slide)
    global_std = float(np.std(flat))
    if global_std <= GLOBAL_STD_PPT:
        median_val = float(np.median(flat))
        low, high = median_val - PIXEL_NEAR_MEDIAN_HALF, median_val + PIXEL_NEAR_MEDIAN_HALF
        near_ratio = np.sum((flat >= low) & (flat <= high)) / flat.size
        if near_ratio >= PIXEL_NEAR_MEDIAN_RATIO:
            return True

    return False


def is_frame_all_black(path: str) -> bool:
    arr = load_grayscale_numpy(path)
    if arr is None or arr.size == 0:
        return False

    flat = arr.flatten()

    mean_val = np.mean(flat)
    std_val = np.std(flat)
    dark_ratio = np.sum(flat <= 50) / flat.size

    # Core idea:
    # majority of pixels are dark + overall structure is weak
    if dark_ratio >= 0.85 and std_val <= 30 and mean_val <= 50:
        return True

    return False

def is_frame_likely_icon(path: str) -> bool:
    """
    Determine whether a single frame looks like an icon example:
    placeholder avatar (white background + gray shape + dark border), logo on dark background, etc.
    Uses PIL + numpy only: dark-border/bright-inner pattern, white/black pixel ratios, bimodal histogram, etc.
    """
    arr = load_grayscale_numpy(path)
    if arr is None or arr.size == 0:
        return False
    h, w = arr.shape
    flat = arr.flatten()
    mean_all = float(np.mean(flat))
    std_all = float(np.std(flat))

    # 1) Thick dark border + bright inner area: similar to boxed placeholder/icon
    br = max(1, int(min(h, w) * BORDER_RATIO))
    top_edge = arr[:br, :].flatten()
    bottom_edge = arr[-br:, :].flatten()
    left_edge = arr[:, :br].flatten()
    right_edge = arr[:, -br:].flatten()
    border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    inner = arr[br:-br, br:-br] if h > 2 * br and w > 2 * br else arr
    mean_border = float(np.mean(border_pixels))
    mean_inner = float(np.mean(inner))
    if mean_border <= BORDER_DARK_MEAN and mean_inner >= INNER_LIGHT_MEAN:
        return True

    # 2) White background + moderate black-pixel ratio (border/outline) + medium std: avatar-like placeholder
    white_ratio = np.sum(flat >= 230) / flat.size
    black_ratio = np.sum(flat <= 100) / flat.size
    if white_ratio >= WHITE_PIXEL_RATIO and BLACK_PIXEL_RATIO_MIN <= black_ratio <= BLACK_PIXEL_RATIO_MAX:
        if ICON_STD_LOW <= std_all <= ICON_STD_HIGH:
            return True

    # 3) Bimodal histogram (dark + bright dominant): logo/icon on dark or light background
    dark_ratio = np.sum(flat <= BIMODAL_DARK_BIN) / flat.size
    bright_ratio = np.sum(flat >= BIMODAL_BRIGHT_BIN) / flat.size
    if dark_ratio + bright_ratio >= BIMODAL_MIN_RATIO and dark_ratio >= 0.15 and bright_ratio >= 0.15:
        return True

    return False


def _frame_id_from_path(path: str) -> str:
    """Extract frame_id from image path, e.g. .../t_3090s.jpg -> t_3090s."""
    return os.path.splitext(os.path.basename(path))[0]


def stage1_collect_ppt_excluded_times() -> dict:
    """
    Stage 1: iterate full-frame screenshots (frames/minute_xx/t_*.jpg).
    - If a timestamp (minute_id + t_xxs) is detected as PPT, exclude all grids at that timestamp.
    Return {minute_id: [frame_id, ...], ...}.
    """
    excluded_times = {}
    if not os.path.isdir(FRAMES_ROOT):
        return excluded_times
    for minute_id in sorted(os.listdir(FRAMES_ROOT)):
        minute_dir = os.path.join(FRAMES_ROOT, minute_id)
        if not os.path.isdir(minute_dir):
            continue
        frames = sorted(
            [f for f in os.listdir(minute_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        for f in frames:
            path = os.path.join(minute_dir, f)
            frame_id = _frame_id_from_path(path)
            if is_frame_likely_ppt(path):
                excluded_times.setdefault(minute_id, []).append(frame_id)
                print(f"[PPT] {minute_id} {frame_id} -> exclude this time for all grids")
    for k in excluded_times:
        excluded_times[k] = sorted(set(excluded_times[k]))
    return excluded_times


def stage2_collect_icon_excluded_frames(ppt_excluded_times: dict) -> dict:
    """
    Stage 2 (ICON): iterate all remaining images (all grids/all frames; skip timestamps excluded by PPT stage).
    If detected as icon-like, exclude only that single frame.
    Return {minute_id: {grid_id: [frame_id, ...]}, ...}.
    Uses hand-crafted logic without OpenCV.
    """
    ppt_set = {}  # minute_id -> set(frame_id)
    for mid, flist in ppt_excluded_times.items():
        ppt_set[mid] = set(flist)
    icon_excluded = {}
    if not os.path.isdir(IMAGE_ROOT):
        return icon_excluded
    for minute_id in sorted(os.listdir(IMAGE_ROOT)):
        video_dir = os.path.join(IMAGE_ROOT, minute_id)
        if not os.path.isdir(video_dir):
            continue
        for grid_id in sorted(os.listdir(video_dir)):
            grid_dir = os.path.join(video_dir, grid_id)
            if not os.path.isdir(grid_dir):
                continue
            files = sorted(
                [f for f in os.listdir(grid_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            )
            for f in files:
                path = os.path.join(grid_dir, f)
                frame_id = _frame_id_from_path(path)
                if frame_id in ppt_set.get(minute_id, set()):
                    continue
                if is_frame_likely_icon(path):
                    icon_excluded.setdefault(minute_id, {}).setdefault(grid_id, []).append(frame_id)
                    print(f"[ICON] {minute_id} {grid_id} {frame_id} -> exclude this frame only")
    for mid in icon_excluded:
        for gid in icon_excluded[mid]:
            icon_excluded[mid][gid] = sorted(set(icon_excluded[mid][gid]))
    return icon_excluded


def stage3_collect_black_excluded_frames(ppt_excluded_times: dict) -> dict:
    """
    Stage 3 (BLACK): iterate all remaining images (all grids/all frames; skip PPT-excluded timestamps).
    If detected as all-black, exclude only that single frame.
    Return {minute_id: {grid_id: [frame_id, ...]}, ...}.
    """
    ppt_set = {}  # minute_id -> set(frame_id)
    for mid, flist in ppt_excluded_times.items():
        ppt_set[mid] = set(flist)
    black_excluded = {}
    if not os.path.isdir(IMAGE_ROOT):
        return black_excluded
    for minute_id in sorted(os.listdir(IMAGE_ROOT)):
        video_dir = os.path.join(IMAGE_ROOT, minute_id)
        if not os.path.isdir(video_dir):
            continue
        for grid_id in sorted(os.listdir(video_dir)):
            grid_dir = os.path.join(video_dir, grid_id)
            if not os.path.isdir(grid_dir):
                continue
            files = sorted(
                [f for f in os.listdir(grid_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            )
            for f in files:
                path = os.path.join(grid_dir, f)
                frame_id = _frame_id_from_path(path)
                if frame_id in ppt_set.get(minute_id, set()):
                    continue
                if is_frame_all_black(path):
                    black_excluded.setdefault(minute_id, {}).setdefault(grid_id, []).append(frame_id)
                    print(f"[BLACK] {minute_id} {grid_id} {frame_id} -> exclude this frame only")
    for mid in black_excluded:
        for gid in black_excluded[mid]:
            black_excluded[mid][gid] = sorted(set(black_excluded[mid][gid]))
    return black_excluded


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ppt_excluded_times = stage1_collect_ppt_excluded_times()
    with open(PPT_EXCLUDED_JSON, "w", encoding="utf-8") as f:
        json.dump(ppt_excluded_times, f, indent=2)
    n_ppt = sum(len(v) for v in ppt_excluded_times.values())
    print(f"[SAVED] {PPT_EXCLUDED_JSON} (stage1: {n_ppt} time slots excluded for all grids)")

    icon_excluded_frames = stage2_collect_icon_excluded_frames(ppt_excluded_times)
    with open(ICON_EXCLUDED_JSON, "w", encoding="utf-8") as f:
        json.dump(icon_excluded_frames, f, indent=2)
    n_icon = sum(len(v) for g in icon_excluded_frames.values() for v in g.values())
    print(f"[SAVED] {ICON_EXCLUDED_JSON} (stage2: {n_icon} single icon frames excluded)")

    black_excluded_frames = stage3_collect_black_excluded_frames(ppt_excluded_times)
    with open(BLACK_EXCLUDED_JSON, "w", encoding="utf-8") as f:
        json.dump(black_excluded_frames, f, indent=2)
    n_black = sum(len(v) for g in black_excluded_frames.values() for v in g.values())
    print(f"[SAVED] {BLACK_EXCLUDED_JSON} (stage3: {n_black} single black frames excluded)")


if __name__ == "__main__":
    main()
