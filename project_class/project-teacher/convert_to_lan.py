# Whisper transcription script designed for lecture/class videos:
# split a 1-hour video into 6-minute chunks, extract audio per chunk, and transcribe sequentially.

import whisper
import os
import subprocess

VIDEO_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\video\HIS 101 (4) - ZOOM Class from Fri. Aug 21st - Syllabus Questions and Announcements for next week.mp4"
OUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\HIS 101 (4) - ZOOM Class from Fri. Aug 21st - Syllabus Questions and Announcements for next week\whisper_out"
LANGUAGE = "en"
MODEL_SIZE = "medium"  # tiny / base / small / medium / large

# Segment length in minutes. A 1-hour video yields 10 segments: 0-6, 6-12, ..., 54-60.
SEGMENT_MINUTES = 6
TOTAL_MINUTES = 60

FFMPEG_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

INITIAL_PROMPT = (
    "This is a university-level lecture. "
    "Transcribe verbatim, do not summarize, "
    "keep all technical terms and full sentences."
)

os.makedirs(OUT_DIR, exist_ok=True)


def extract_audio_segment(start_min: int, duration_min: int, out_wav_path: str) -> bool:
    """Use ffmpeg to extract audio from [start_min, start_min+duration_min) and save as 16kHz mono wav."""
    start_sec = start_min * 60
    duration_sec = duration_min * 60
    ret = subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", VIDEO_PATH,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        out_wav_path
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if ret.returncode != 0:
        print(f"[WARN] ffmpeg segment {start_min}m–{start_min + duration_min}m: {ret.stderr[:200]}")
        return False
    return True


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return
    if not os.path.isfile(FFMPEG_PATH):
        print(f"[ERROR] ffmpeg not found: {FFMPEG_PATH}")
        return

    print("Loading Whisper model:", MODEL_SIZE)
    model = whisper.load_model(MODEL_SIZE)

    all_segments_text = []
    n = (TOTAL_MINUTES + SEGMENT_MINUTES - 1) // SEGMENT_MINUTES  # 10 segments

    for i in range(n):
        start_min = i * SEGMENT_MINUTES
        end_min = min(start_min + SEGMENT_MINUTES, TOTAL_MINUTES)
        duration_min = end_min - start_min
        label = f"{start_min}m_{end_min}m"

        print(f"[{i+1}/{n}] Extracting audio {label} ...")
        seg_wav = os.path.join(OUT_DIR, f"audio_{label}.wav")
        if not extract_audio_segment(start_min, duration_min, seg_wav):
            all_segments_text.append(f"[{label}]\n(audio extraction failed)\n")
            continue

        if not os.path.isfile(seg_wav) or os.path.getsize(seg_wav) == 0:
            all_segments_text.append(f"[{label}]\n(no valid audio)\n")
            continue

        print(f"        Transcribing {label} ...")
        try:
            result = model.transcribe(
                seg_wav,
                fp16=False,
                language=None,
                condition_on_previous_text=True,
                initial_prompt=INITIAL_PROMPT,
            )
            text = (result.get("text") or "").strip()
        except Exception as e:
            text = f"(transcription exception: {e})"
            print(f"        [WARN] {e}")

        # Save this segment separately
        txt_path = os.path.join(OUT_DIR, f"transcript_{label}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"        Saved: {txt_path}")

        all_segments_text.append(f"[{label}]\n{text}\n\n")

    # Merge into one full transcript
    full_path = os.path.join(OUT_DIR, "transcript_full.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        f.writelines(all_segments_text)
    print("Transcription completed")
    print("Segment output directory:", OUT_DIR)
    print("Merged transcript:", full_path)


if __name__ == "__main__":
    main()
