"""
Simple engagement scoring ablation:
- Uses only a minimal prompt with 0-3 engagement definitions.
- Directly asks the LLM for a final score from preprocessed text evidence.
"""

import os
import json
from openai import OpenAI

# ---------- LLM CONFIG ----------
TIMEOUT = 60

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

# Input/Output paths
PREPROCESSOR_OUTPUT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Intro Soc 220 1⧸19⧸2021/preprocessor_output"
OUTPUT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/ablation/focus_ouput/Zoom Class Meeting Downing Intro Soc 220 1⧸19⧸2021"
WINDOW_MINUTES = 3


def get_minute_files():
    """Find all preprocessor minute files and sort by minute index."""
    if not os.path.isdir(PREPROCESSOR_OUTPUT_DIR):
        return []
    out = []
    for f in os.listdir(PREPROCESSOR_OUTPUT_DIR):
        if not f.endswith(".json") or not f.startswith("minute_"):
            continue
        try:
            num = int(f.replace("minute_", "").replace(".json", ""))
            out.append((num, f))
        except ValueError:
            continue
    return [fname for _, fname in sorted(out, key=lambda x: x[0])]


def minute_index_from_fname(fname: str) -> int:
    try:
        return int(fname.replace("minute_", "").replace(".json", ""))
    except Exception:
        return 0


def group_minutes_by_window(minute_fnames, window_minutes=3):
    """Group minute files into fixed windows."""
    if not minute_fnames:
        return []
    with_idx = [(minute_index_from_fname(f), f) for f in minute_fnames]
    with_idx.sort(key=lambda x: x[0])
    minute_fnames = [f for _, f in with_idx]
    groups = []
    i = 0
    while i < len(minute_fnames):
        window_files = minute_fnames[i : i + window_minutes]
        if not window_files:
            break
        start_min = minute_index_from_fname(window_files[0])
        end_min = minute_index_from_fname(window_files[-1]) + 1
        window_label = f"{start_min}m_{end_min}m"
        groups.append((window_label, window_files))
        i += window_minutes
    return groups


SIMPLE_PROMPT = """
You are a behavioral engagement evaluation model.
You must be strict, but you must also use the full 0-3 scale.

You are given observational evidence (scene descriptions from visual frames) describing a person's
visible behavior across a 3-minute window (three 1-minute segments, one paragraph per minute).

The evidence is standardized narrative from preprocessor output (one description per minute per grid).

Your task is to assess the engagement level of the person
based strictly on the observable evidence provided.

Engagement is defined on FOUR ordered levels:

0 - Not Engaged:
- frequent looking away or down
- obvious inattentive behavior

1 - Barely Engaged:
- weak or unstable attention
- inconsistent orientation or posture

2 - Engaged:
- generally oriented forward
- stable posture most of the time
- attentive with minor lapses

3 - Highly Engaged:
- this is the DEFAULT level when evidence shows
  stable forward orientation and posture
- no dominant or sustained distraction
- brief lapses are acceptable

You must make a holistic judgment based on the OVERALL tendency
across all segments.

Return STRICT JSON ONLY:
{{
  "engagement": <integer 0-3>,
  "reasoning": "<concise explanation grounded in evidence>"
}}

## ATTENTION_EVIDENCE
{attention_text}
"""


def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        timeout=TIMEOUT,
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def clamp_score(v) -> int:
    try:
        x = int(v)
    except Exception:
        return 1
    return max(0, min(3, x))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    minute_fnames = get_minute_files()
    if not minute_fnames:
        raise FileNotFoundError(f"No minute_*.json found in {PREPROCESSOR_OUTPUT_DIR}")

    windows = group_minutes_by_window(minute_fnames, WINDOW_MINUTES)
    if not windows:
        print("No 3-minute windows to process.")
        return

    for window_label, window_fnames in windows:
        print(f"\n[WINDOW] {window_label} from {window_fnames}")

        minute_data = []
        for fname in window_fnames:
            path = os.path.join(PREPROCESSOR_OUTPUT_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                minute_data.append(json.load(f))

        grid_ids = sorted(set().union(*(set(d.keys()) for d in minute_data)))
        results = {}

        for grid_id in grid_ids:
            blocks = []
            for i, data in enumerate(minute_data):
                if grid_id not in data:
                    continue
                text = data[grid_id]
                if isinstance(text, str) and text.strip():
                    blocks.append(f"[MINUTE {i+1}]\n{text.strip()}")

            if not blocks:
                continue

            evidence_text = "\n\n".join(blocks)
            prompt = SIMPLE_PROMPT.format(attention_text=evidence_text)

            try:
                result = call_llm(prompt)
                engagement = clamp_score(result.get("engagement"))
                reasoning = result.get("reasoning", "")
            except Exception as e:
                engagement = None
                reasoning = str(e)

            results[grid_id] = {
                "engagement": engagement,
                "reasoning": reasoning,
                "num_segments": len(blocks),
                "method": "simple_direct_scoring",
            }
            print(f"[DONE] {window_label} {grid_id} -> {engagement}")

        out_path = os.path.join(OUTPUT_DIR, f"engagement_{window_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
