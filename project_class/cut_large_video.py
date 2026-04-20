import os
from moviepy import VideoFileClip

input_video_path = r"D:\Desktop\llm_as_a_judge\data\appendix\Zoom Class Meeting Downing Soc 220 2⧸18⧸2021 [i-w2sxYTQbc].mp4"

# extract 2nd hour vedio
EXTRACT_START_SEC = 3600       
EXTRACT_DURATION_SEC = 3600    
base_dir = r"D:\Desktop\llm_as_a_judge\data\llm\project_class"


def extract_one_hour(input_path):
    video = VideoFileClip(input_path)
    duration = video.duration
    start = min(EXTRACT_START_SEC, duration)
    end = min(start + EXTRACT_DURATION_SEC, duration)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{basename}_hour2.mp4"
    output_path = os.path.join(base_dir, output_filename)

    print(f"Extracting {start}s → {end}s (second hour or until video end)")
    clip = video.subclipped(start, end)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
    video.close()
    print("Finished.")


if __name__ == "__main__":
    extract_one_hour(input_video_path)
