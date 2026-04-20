# Whisper transcription script designed for academic lecture/course videos:
# standardized audio -> automatic language detection -> context-continuous decoding
# -> prompt constraints to maximize verbatim, professional, and complete transcription quality.

import whisper
import os
import subprocess

VIDEO_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\Zoom Class Meeting Downing Soc 220 2⧸18⧸2021 [i-w2sxYTQbc]_001.mp4"
OUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\whisper_out"
LANGUAGE = "en"     
MODEL_SIZE = "medium"  # tiny / base / small / medium / large


os.makedirs(OUT_DIR, exist_ok=True)

audio_path = os.path.join(OUT_DIR, "audio.wav")

# Extract audio
FFMPEG_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

subprocess.run([
    FFMPEG_PATH, "-y",
    "-i", VIDEO_PATH,
    "-ar", "16000",
    "-ac", "1",
    "-vn",
    audio_path
], check=True)

# Load Whisper
model = whisper.load_model(MODEL_SIZE)

# # 3) Transcription
# result = model.transcribe(
#     audio_path,
#     language=LANGUAGE,
#     fp16=False  # Must be False on CPU; optional on GPU
# )
result = model.transcribe(
    audio_path,
    fp16=False,
    language=None,
    condition_on_previous_text=True,
    initial_prompt=(
        "This is a university-level lecture. "
        "Transcribe verbatim, do not summarize, "
        "keep all technical terms and full sentences."
    )
)

# Save plain text output
txt_path = os.path.join(OUT_DIR, "transcript_2.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription completed")
print("Output:", txt_path)
