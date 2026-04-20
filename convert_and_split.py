# WhisperX script for academic lecture/course videos:
# 1) Use whisperx for verbatim transcription (transcribe + alignment)
# 2) Use diarization for speaker separation
# 3) Automatically map speakers to teacher/student and output corresponding transcripts

import os
import re
import json
import subprocess
from typing import Dict, List

import torch


VIDEO_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\Zoom Class Meeting Downing Soc 220 2⧸18⧸2021 [i-w2sxYTQbc]_001.mp4"
OUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\whisper_out"

# WhisperX model (recommended: large-v2 / medium / small; depends on speed and VRAM)
WHISPERX_MODEL_NAME = os.getenv("WHISPERX_MODEL_NAME", "medium")

# Language: leave empty if unknown (let whisperx auto-detect)
LANGUAGE = os.getenv("WHISPERX_LANGUAGE", "en").strip()

# diarization (pyannote) requires a token
PYANNOTE_TOKEN = os.getenv("PYANNOTE_TOKEN", "").strip()

# If you want to manually specify which speaker is the teacher (e.g. SPEAKER_00),
# set this env var; otherwise the speaker with the longest total duration is treated as teacher.
TEACHER_SPEAKER_ID = os.getenv("TEACHER_SPEAKER_ID", "").strip()

# Output files: keep transcript_2.txt from your original pipeline
FULL_TRANSCRIPT_PATH = os.path.join(OUT_DIR, "transcript_2.txt")
TEACHER_TRANSCRIPT_PATH = os.path.join(OUT_DIR, "transcript_teacher.txt")
STUDENT_TRANSCRIPT_PATH = os.path.join(OUT_DIR, "transcript_student.txt")
UTTERANCES_JSON_PATH = os.path.join(OUT_DIR, "utterances_teacher_student.json")

# Extract audio (keep your ffmpeg path unchanged)
FFMPEG_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
SAMPLE_RATE = 16000
CHANNELS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


def _clean_joined_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return text


def extract_audio_ffmpeg(video_path: str, audio_path: str) -> None:
    subprocess.run(
        [
            FFMPEG_PATH,
            "-y",
            "-i",
            video_path,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-vn",
            audio_path,
        ],
        check=True,
    )


def load_whisperx():
    try:
        import whisperx  # type: ignore
    except ImportError as e:
        raise ImportError(
            "whisperx is not installed. Please install it first: pip install whisperx pyannote.audio\n"
            "and configure PYANNOTE_TOKEN."
        ) from e
    return whisperx


def map_speakers_to_roles(diarize_segments: List[Dict], teacher_speaker_id: str) -> Dict[str, str]:
    """
    diarization outputs speakers like SPEAKER_00 / SPEAKER_01 ...
    Role mapping: teacher / student
    Default strategy: speaker with the longest total duration -> teacher
    """
    durations: Dict[str, float] = {}
    for s in diarize_segments:
        sp = str(s.get("speaker", "UNKNOWN"))
        st = float(s.get("start", 0.0))
        en = float(s.get("end", 0.0))
        durations[sp] = durations.get(sp, 0.0) + max(0.0, en - st)

    if teacher_speaker_id:
        teacher = teacher_speaker_id
    elif durations:
        teacher = max(durations.items(), key=lambda kv: kv[1])[0]
    else:
        teacher = "SPEAKER_00"

    speaker_role = {sp: ("teacher" if sp == teacher else "student") for sp in durations.keys()}
    speaker_role.setdefault(teacher, "teacher")
    return speaker_role


def build_utterances_from_word_speakers(
    whisperx_result: Dict,
    speaker_role: Dict[str, str],
    gap_threshold_sec: float = 0.8,
) -> List[Dict]:
    """
    After whisperx.assign_word_speakers, segments->words includes speaker/start/end/word.
    Merge consecutive words from the same speaker into utterances.
    """
    words = []
    for seg in whisperx_result.get("segments", []):
        for w in seg.get("words", []) or []:
            if not isinstance(w, dict):
                continue
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append(
                {
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": str(w.get("speaker", "UNKNOWN")),
                    "word": str(w.get("word", "")),
                }
            )
    words.sort(key=lambda x: x["start"])

    utterances: List[Dict] = []
    cur = None
    last_end = None

    for w in words:
        sp = w["speaker"]
        role = speaker_role.get(sp, "student")
        if cur is None:
            cur = {
                "start": w["start"],
                "end": w["end"],
                "speaker": sp,
                "role": role,
                "word_pieces": [w["word"]],
            }
            last_end = w["end"]
            continue

        if sp != cur["speaker"] or (last_end is not None and (w["start"] - last_end) > gap_threshold_sec):
            cur["text"] = _clean_joined_text("".join(cur["word_pieces"]))
            utterances.append(cur)
            cur = {
                "start": w["start"],
                "end": w["end"],
                "speaker": sp,
                "role": role,
                "word_pieces": [w["word"]],
            }
        else:
            cur["end"] = w["end"]
            cur["word_pieces"].append(w["word"])
        last_end = w["end"]

    if cur is not None and cur.get("word_pieces"):
        cur["text"] = _clean_joined_text("".join(cur["word_pieces"]))
        utterances.append(cur)

    for u in utterances:
        u.pop("word_pieces", None)
    return utterances


def main():
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"VIDEO_PATH not found: {VIDEO_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
    audio_path = os.path.join(OUT_DIR, "audio.wav")

    print(f"[1/5] Extracting audio: {audio_path}")
    extract_audio_ffmpeg(VIDEO_PATH, audio_path)

    whisperx = load_whisperx()

    print(f"[2/5] Loading whisperx model: {WHISPERX_MODEL_NAME} device={DEVICE} compute_type={COMPUTE_TYPE}")
    model = whisperx.load_model(WHISPERX_MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)

    print("[3/5] Transcribing")
    transcribe_kwargs = {"batch_size": 16}
    if LANGUAGE:
        transcribe_kwargs["language"] = LANGUAGE
    result = model.transcribe(audio_path, **transcribe_kwargs)

    language_code = result.get("language", LANGUAGE) or LANGUAGE
    print(f"detected language: {language_code}")

    print("[4/5] alignment")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=DEVICE)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_path, DEVICE)

    if not PYANNOTE_TOKEN:
        raise ValueError(
            "PYANNOTE_TOKEN is missing. Set environment variable PYANNOTE_TOKEN before running.\n"
            "For example: $env:PYANNOTE_TOKEN='your_token'"
        )

    print("[5/5] diarization + assign_word_speakers")
    diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=PYANNOTE_TOKEN, device=DEVICE)
    diarize_segments = diarize_pipeline(audio_path)
    speaker_role = map_speakers_to_roles(diarize_segments, TEACHER_SPEAKER_ID)

    final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    utterances = build_utterances_from_word_speakers(final_result, speaker_role)

    full_text = _clean_joined_text("\n".join([u.get("text", "") for u in utterances if str(u.get("text", "")).strip()]))
    teacher_text = _clean_joined_text("\n".join([u.get("text", "") for u in utterances if u.get("role") == "teacher" and str(u.get("text", "")).strip()]))
    student_text = _clean_joined_text("\n".join([u.get("text", "") for u in utterances if u.get("role") == "student" and str(u.get("text", "")).strip()]))

    with open(FULL_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)
    with open(TEACHER_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(teacher_text)
    with open(STUDENT_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(student_text)

    with open(UTTERANCES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": VIDEO_PATH,
                "audio_path": audio_path,
                "whisperx_model_name": WHISPERX_MODEL_NAME,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "detected_language": language_code,
                "speaker_role_mapping": speaker_role,
                "utterances": utterances,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== Done ===")
    print("Full transcript:", FULL_TRANSCRIPT_PATH)
    print("Teacher transcript:", TEACHER_TRANSCRIPT_PATH)
    print("Student transcript:", STUDENT_TRANSCRIPT_PATH)
    print("utterances:", UTTERANCES_JSON_PATH)


if __name__ == "__main__":
    main()
