import glob
import json
import os
import re
from typing import Dict, List, Tuple

from openai import OpenAI


# API config (consistent with other project scripts: prefer OpenRouter, fallback to official OpenAI)
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    MODEL = "openai/gpt-4o-mini"
    client = OpenAI(
        api_key=_openrouter_key.strip(),
        base_url="https://openrouter.ai/api/v1",
    )
elif _openai_key and _openai_key.strip():
    MODEL = "gpt-4o-mini"
    client = OpenAI(api_key=_openai_key.strip())
else:
    raise ValueError(
        "API key is not set. Please configure one of the following:\n"
        "  - set OPENROUTER_API_KEY=your_key   (OpenRouter)\n"
        "  - set OPENAI_API_KEY=your_key       (Official OpenAI API)"
    )


# Path configuration
TRANSCRIPT_DIR = (
    r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Soc 100 4⧸19⧸2021/whisper_out"
)
OUTPUT_DIR = (
    r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/prompt/Zoom Class Meeting Downing Soc 100 4.19.2021"
)
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DIR, "teaching_suggestions_prompt_only.jsonl")

# Inference parameters
TIMEOUT = int(os.getenv("PROMPT_ONLY_TIMEOUT", "120"))
TEMPERATURE = float(os.getenv("PROMPT_ONLY_TEMPERATURE", "0.2"))

# Token statistics
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0

TRANSCRIPT_PATTERN = re.compile(r"transcript_(\d+)m_(\d+)m\.txt$")

# Maximum suggestions kept per transcript (extra paragraphs are truncated)
MAX_SUGGESTIONS = 3

SYSTEM_PROMPT = """
Provide practical teaching suggestions based only on this transcript. At most 3.
"""


def parse_suggestions_from_text(text: str) -> Dict:
    if not text or not text.strip():
        raise ValueError("Empty model output.")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not parts:
        raise ValueError("No suggestions parsed from model output.")
    return {"teaching_suggestions": parts[:MAX_SUGGESTIONS]}


def call_gpt(system_prompt: str, user_prompt: str) -> str:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
        TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
    REQUEST_COUNT += 1
    return response.choices[0].message.content


def list_transcript_windows(folder: str) -> List[Tuple[int, int, str]]:
    items: List[Tuple[int, int, str]] = []
    for path in glob.glob(os.path.join(folder, "transcript_*m_*m.txt")):
        m = TRANSCRIPT_PATTERN.search(os.path.basename(path))
        if not m:
            continue
        start_m = int(m.group(1))
        end_m = int(m.group(2))
        items.append((start_m, end_m, path))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def build_user_prompt(window_label: str, transcript_text: str) -> str:
    _ = window_label
    return transcript_text


def main() -> None:
    windows = list_transcript_windows(TRANSCRIPT_DIR)
    if not windows:
        raise FileNotFoundError(f"No transcript windows found in: {TRANSCRIPT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records: List[Dict] = []

    for start_m, end_m, transcript_path in windows:
        window_label = f"{start_m}m_{end_m}m"
        print(f"\n[RUN] {window_label}")

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()

        if not transcript_text:
            print("  [SKIP] empty transcript")
            continue

        user_prompt = build_user_prompt(window_label, transcript_text)
        raw = call_gpt(SYSTEM_PROMPT, user_prompt)
        result = parse_suggestions_from_text(raw)

        record = {
            "window": window_label,
            "transcript_path": transcript_path,
            "method": "prompt_only",
            "model_name": MODEL,
            **result,
        }
        all_records.append(record)

        single_out = os.path.join(OUTPUT_DIR, f"teaching_suggestions_{window_label}.json")
        with open(single_out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {single_out}")

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DONE] jsonl saved to: {OUTPUT_JSONL_PATH}")
    if REQUEST_COUNT > 0:
        total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
        print(
            f"[TOKEN] prompt={TOTAL_PROMPT_TOKENS}, completion={TOTAL_COMPLETION_TOKENS}, "
            f"total={total_tokens}, requests={REQUEST_COUNT}"
        )


if __name__ == "__main__":
    main()
