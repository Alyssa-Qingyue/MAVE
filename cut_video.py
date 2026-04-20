import subprocess
import os

input_video = r"D:\Desktop\llm_as_a_judge\data\llm\Zoom Class Meeting Downing Soc 220 2⧸18⧸2021 [i-w2sxYTQbc]_001.mp4"
output_dir = r"D:\Desktop\llm_as_a_judge\data\llm\cut_10s"

os.makedirs(output_dir, exist_ok=True)

# 3 minutes = 180 seconds, 10 seconds per segment, 18 segments total
for i in range(18):
    start = i * 10
    end = start + 10
    output = f"{output_dir}/segment_{start}s_{end}s.mp4"

    cmd = [
        r"D:\Desktop\llm_as_a_judge\data\llm\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        "-y",
        "-ss", str(start),
        "-t", "10",
        "-i", input_video,
        "-c", "copy",
        output
    ]

    subprocess.run(cmd, check=True)
