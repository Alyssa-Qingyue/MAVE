import subprocess
import os

PROJECT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021"
VIDEO_FILENAME = "Zoom Class Meeting Downing Soc 100 4⧸19⧸2021.mp4"
input_video = os.path.join(PROJECT_DIR, VIDEO_FILENAME)

output_root = os.path.join(PROJECT_DIR, "frames")
os.makedirs(output_root, exist_ok=True)

#extract vedio frames
FFMPEG = r"D:\Desktop\llm_as_a_judge\data\llm\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

TOTAL_MINUTES = 60
INTERVAL_SEC = 10

if not os.path.isfile(input_video):
    raise FileNotFoundError(f"Video not found: {input_video}")

for minute in range(TOTAL_MINUTES):
    minute_start = minute * 60
    minute_dir = os.path.join(output_root, f"minute_{minute:02d}")
    os.makedirs(minute_dir, exist_ok=True)

    for offset in range(0, 60, INTERVAL_SEC):
        t = minute_start + offset
        output_path = os.path.join(minute_dir, f"t_{t}s.jpg")

        cmd = [
            FFMPEG,
            "-y",
            "-ss", str(t),
            "-i", input_video,
            "-vframes", "1",
            "-q:v", "2",
            "-pix_fmt", "yuvj420p",
            output_path,
        ]

        subprocess.run(cmd, check=True)
