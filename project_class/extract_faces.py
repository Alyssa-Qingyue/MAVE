import cv2
import numpy as np
import os
import sys
import traceback

print("USING PYTHON:", sys.executable)


def read_image_unicode_safe(path):
    """Read image bytes in Python then decode via OpenCV to avoid Unicode path issues in OpenCV C++ layer."""
    try:
        with open(path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def write_image_unicode_safe(path, img):
    """Encode with OpenCV and write bytes in Python to avoid cv2.imwrite failures on Unicode paths."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False

# Upstream input: output directory from cut_video.py.
# It should contain minute_00, minute_01, ... and files like t_0s.jpg, t_10s.jpg.
# If no output appears, verify:
# 1) FRAMES_ROOT exists and points to the intended video
# 2) FRAMES_ROOT has minute_* folders with .jpg files
FRAMES_ROOT = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\frames"
OUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\images"
GRID_ROWS = 5
GRID_COLS = 5

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(FRAMES_ROOT):
    raise RuntimeError(f"FRAMES_ROOT not found: {FRAMES_ROOT}")

# One folder per minute: minute_00, minute_01, ...
minute_dirs = sorted([
    d for d in os.listdir(FRAMES_ROOT)
    if os.path.isdir(os.path.join(FRAMES_ROOT, d)) and d.startswith("minute_")
])

print(f"FRAMES_ROOT: {FRAMES_ROOT}")
print(f"Found {len(minute_dirs)} minute_* folders: {minute_dirs[:5]}{'...' if len(minute_dirs) > 5 else ''}")

if not minute_dirs:
    raise RuntimeError(f"No minute_* folders found in {FRAMES_ROOT}")

total_written = 0
for minute_id in minute_dirs:
    minute_path = os.path.join(FRAMES_ROOT, minute_id)
    image_files = sorted([
        f for f in os.listdir(minute_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"[SKIP] {minute_id}: no images in {minute_path}")
        continue

    print(f"\nProcessing {minute_id}: {len(image_files)} images")

    # Output root for this minute: images_out/minute_00/, minute_01/, ...
    minute_out_base = os.path.join(OUT_DIR, minute_id)
    os.makedirs(minute_out_base, exist_ok=True)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gid = r * GRID_COLS + c
            os.makedirs(os.path.join(minute_out_base, f"grid_{gid:02d}"), exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(minute_path, img_name)
        frame = read_image_unicode_safe(img_path)

        if frame is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        if frame.ndim != 3:
            continue

        h, w, _ = frame.shape
        if h == 0 or w == 0:
            continue

        # Use filename as time tag (e.g. t_0s.jpg -> t_0s)
        time_tag = os.path.splitext(img_name)[0]

        cell_h = h // GRID_ROWS
        cell_w = w // GRID_COLS

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y1 = r * cell_h
                y2 = h if r == GRID_ROWS - 1 else (r + 1) * cell_h
                x1 = c * cell_w
                x2 = w if c == GRID_COLS - 1 else (c + 1) * cell_w

                if y2 <= y1 or x2 <= x1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                gid = r * GRID_COLS + c
                grid_dir = os.path.join(minute_out_base, f"grid_{gid:02d}")
                out_path = os.path.join(grid_dir, f"{time_tag}.jpg")
                if write_image_unicode_safe(out_path, crop):
                    total_written += 1

print(f"\nFinished. Total images written: {total_written}")
print(f"Output directory: {OUT_DIR}")
