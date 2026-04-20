# -*- coding: utf-8 -*-
"""

ffmpeg -y -i "D:\Desktop\llm_as_a_judge\data\llm\your_filename.mkv" -c copy "D:\Desktop\llm_as_a_judge\data\llm\your_filename.mp4"

Current video inventory:
Creator: sociology Jason
Demo videos
https://www.youtube.com/watch?v=i-w2sxYTQbc
\Zoom Class Meeting Downing Intro Soc 220 1⧸19⧸2021.mp4
https://www.youtube.com/watch?v=WxmGD4PwAPk
\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021.mp4
https://www.youtube.com/watch?v=yuCaZtANtjU


HIS 101 (4) - ZOOM Class from Fri. Aug 21st - Syllabus Questions and Announcements for next week.mp4
https://www.youtube.com/watch?v=yi4LNeipqHk


Town Hall Video for MSA 601 January 12, 2026
https://www.youtube.com/watch?v=s0ZZcqNFIWs


"""

import os
import subprocess
import sys

DEFAULT_OUTPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm"

# Default video URL to download
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=yuCaZtANtjU"

# Path to yt-dlp executable (set to None if already in PATH; script will call "yt-dlp" directly)
YT_DLP_PATH = r"C:\tools\yt-dlp\yt-dlp.exe"
# If already in PATH, you can use: YT_DLP_PATH = "yt-dlp"

# Output filename template: %(title)s is video title, %(ext)s is extension (e.g., mp4, webm)
OUTPUT_TEMPLATE = "%(title)s.%(ext)s"


def get_yt_dlp_cmd():
    """Return yt-dlp command as a list (convenient for subprocess)."""
    if YT_DLP_PATH and os.path.isfile(YT_DLP_PATH):
        return [YT_DLP_PATH]
    return ["yt-dlp"]


def download_youtube_video(
    video_url: str,
    output_dir: str | None = None,
    output_template: str | None = None,
) -> bool:
    """
    Download the specified YouTube video to the target folder using yt-dlp.

    Parameters:
        video_url: Video page URL, e.g. https://www.youtube.com/watch?v=xxxxx
        output_dir: Save directory, defaults to DEFAULT_OUTPUT_DIR; auto-created if missing.
        output_template: Filename template, default "%(title)s.%(ext)s".

    Returns:
        True means success (or already exists), False means failure.
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_template = output_template or OUTPUT_TEMPLATE

    os.makedirs(output_dir, exist_ok=True)

    cmd = get_yt_dlp_cmd() + [
        "-P", output_dir,
        "-o", output_template,
        video_url,
    ]

    print(f"[yt-dlp] Output directory: {output_dir}")
    print(f"[yt-dlp] Video URL: {video_url}")
    print(f"[yt-dlp] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if result.returncode == 0:
            print("[yt-dlp] Download completed.")
            return True
        print(f"[yt-dlp] Exit code: {result.returncode}")
        return False
    except FileNotFoundError:
        print("[ERROR] yt-dlp not found. Put yt-dlp.exe in C:\\tools\\yt-dlp\\ or add it to PATH.")
        return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Download timed out.")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def check_yt_dlp_and_ffmpeg() -> bool:
    """
    Check whether yt-dlp and ffmpeg are available.
    Return True only if both are available.
    """
    ok = True
    # Check yt-dlp
    try:
        subprocess.run(
            get_yt_dlp_cmd() + ["--version"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        print("yt-dlp: installed")
    except FileNotFoundError:
        print("yt-dlp: not found. Install it and add to PATH, or configure YT_DLP_PATH.")
        ok = False
    except Exception as e:
        print(f"yt-dlp: check failed - {e}")
        ok = False

    # Check ffmpeg (pip install ffmpeg is not the binary itself; install ffmpeg.exe separately)
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        print("ffmpeg: installed")
    except FileNotFoundError:
        print("ffmpeg: not found. Install it and add the bin directory to PATH.")
        ok = False
    except Exception as e:
        print(f"ffmpeg: check failed - {e}")
        ok = False

    return ok


if __name__ == "__main__":
    # Optional: validate environment first
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        if check_yt_dlp_and_ffmpeg():
            print("\nEnvironment check passed. Ready to download.")
        else:
            print("\nPlease install yt-dlp and FFmpeg as noted at the top of this script, then retry.")
        sys.exit(0 if check_yt_dlp_and_ffmpeg() else 1)

    # CLI usage: python get_vedio.py [video_url] [output_dir]
    video_url = DEFAULT_VIDEO_URL
    output_dir = DEFAULT_OUTPUT_DIR
    if len(sys.argv) >= 2:
        video_url = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]

    if "xxxxx" in video_url:
        print("Please update DEFAULT_VIDEO_URL in this script, or run:")
        print('  python get_vedio.py "https://www.youtube.com/watch?v=actual_id" [output_dir]')
        sys.exit(1)

    success = download_youtube_video(video_url, output_dir)
    sys.exit(0 if success else 1)
