"""
Input: minute-level preprocessor output from project_class (one JSON per 1-minute window).
Goal: aggregate every 3 minutes and produce structured attention descriptions for
each grid, only for the current 3-minute block.

Example input layout (from preprocessor.py):
  preprocessor_output/
    minute_00.json
    minute_01.json
    minute_02.json
    minute_03.json
    ...

Each minute_XX.json:
{
  "grid_00": "narrative text for this minute and grid",
  "grid_01": "...",
  ...
}

This script groups every 3 consecutive minutes into a 3-minute segment
and aggregates the narratives for each grid, then calls the LLM once per
grid+segment to extract attention-related structure.

It also tracks token usage and approximate API cost.
[USAGE] requests=500, total_prompt_tokens=431274, total_completion_tokens=167966, 
avg_prompt_tokens_per_req=862.5, avg_completion_tokens_per_req=335.9
[COST] input_tokens_k=431.27, output_tokens_k=167.97, 
estimated_input_cost=$0.0647, estimated_output_cost=$0.1008, estimated_total_cost=$0.1655
"""

import os
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"
TIMEOUT = 60

# Input: minute-level preprocessor output from project_class
INPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project_class\output\preprocessor_output"
# Output: one structured_output file per 3-minute window
OUTPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project_class\output\structured_output_3min"

# Token and cost statistics (prices can be adjusted as needed)
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0

# Assumed gpt-4o-mini pricing (example; adjust to actual pricing if needed)
PRICE_INPUT_PER_1K = 0.00015   # USD per 1K prompt tokens
PRICE_OUTPUT_PER_1K = 0.00060  # USD per 1K completion tokens


# PROMPT: ATTENTION EXTRACTION (same logic as original narrative, with 3-minute aggregation clarified)

ATTENTION_PROMPT = """
You are an expert visual behavior analyst.

You are given multiple factual visual scene descriptions
corresponding to the same 3-minute window, aggregated from several
shorter observations (for example, minute-level narratives).

Your task is to extract ALL observable information that may be
relevant to attentional focus.

## INPUT_DESCRIPTION
{description}

## Instructions:
- List as many observable details as possible.
- Focus on head orientation, gaze-related cues, body posture,
  hand movements, and interactions with objects.
- Include small or subtle movements if mentioned.
- Do NOT infer mental states, intentions, or emotions.
- Do NOT summarize or judge yet.
- Use plain descriptive language.

## Output Format:
Write your answer in clearly separated sections:

HEAD AND GAZE:
- ...

BODY POSTURE:
- ...

HAND ACTIVITY:
- ...

OBJECT INTERACTION:
- ...

TEMPORAL CHANGES (if any):
- ...

NOTES / UNCERTAINTIES:
- ...
"""


def extract_attention(narrative: str) -> str:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    prompt = ATTENTION_PROMPT.format(description=narrative)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=TIMEOUT,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")

    try:
        text = response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Malformed OpenAI response: {e}")

    usage = getattr(response, "usage", None)
    if usage is not None:
        TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
        TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
    REQUEST_COUNT += 1

    return text


def group_minute_files_by_3min():
    """
    Group minute_XX.json files by chronological order in chunks of 3.
    Return: [(segment_label, [fname1, fname2, fname3]), ...]
    segment_label is used for output naming, e.g. 'segment_00m_03m'.
    """
    all_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".json") and f.startswith("minute_")
    ]
    if not all_files:
        return []

    def minute_index(fname: str) -> int:
        # minute_00.json -> 0
        try:
            stem = os.path.splitext(fname)[0]  # minute_00
            return int(stem.split("_")[1])
        except Exception:
            return 0

    all_files_sorted = sorted(all_files, key=minute_index)

    groups = []
    n = len(all_files_sorted)
    i = 0
    while i < n:
        window_files = all_files_sorted[i:i + 3]
        if not window_files:
            break

        start_min = minute_index(window_files[0])
        end_min_excl = start_min + len(window_files)  # [start, end)
        segment_label = f"segment_{start_min:02d}m_{end_min_excl:02d}m"
        groups.append((segment_label, window_files))
        i += 3

    return groups


# MAIN PIPELINE

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = group_minute_files_by_3min()
    if not groups:
        print("No minute_*.json files found to aggregate.")
        return

    for segment_label, fnames in groups:
        print(f"\n[SEGMENT] {segment_label} from files: {fnames}")

        # Read all minute files in this 3-minute window
        minute_dicts = []
        for fname in fnames:
            in_path = os.path.join(INPUT_DIR, fname)
            with open(in_path, "r", encoding="utf-8") as f:
                minute_dicts.append(json.load(f))

        # Collect all grid IDs (union)
        grid_ids = set()
        for d in minute_dicts:
            grid_ids.update(d.keys())
        grid_ids = sorted(grid_ids)

        new_data = {}

        for grid_id in grid_ids:
            parts = []
            for fname, d in zip(fnames, minute_dicts):
                if grid_id not in d:
                    continue
                content = d[grid_id]
                # preprocessor output format: grid_id -> str
                if isinstance(content, str):
                    minute_tag = os.path.splitext(fname)[0]  # e.g. minute_00
                    parts.append(f"[{minute_tag}]\n{content}")
                else:
                    # Unsupported structure; skip this minute
                    continue

            if not parts:
                continue

            narrative = "\n\n".join(parts)

            try:
                attention = extract_attention(narrative)
            except Exception as e:
                print(f"[FAIL] {segment_label} {grid_id}: {e}")
                continue

            new_data[grid_id] = {
                "narrative": narrative,
                "attention": attention,
            }

            print(f"[OK] {segment_label} {grid_id}")

        if new_data:
            out_path = os.path.join(OUTPUT_DIR, f"{segment_label}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {out_path}")

    # Usage statistics and cost estimation
    if REQUEST_COUNT > 0:
        total_prompt_k = TOTAL_PROMPT_TOKENS / 1000.0
        total_completion_k = TOTAL_COMPLETION_TOKENS / 1000.0
        cost_input = total_prompt_k * PRICE_INPUT_PER_1K
        cost_output = total_completion_k * PRICE_OUTPUT_PER_1K
        total_cost = cost_input + cost_output

        avg_prompt = TOTAL_PROMPT_TOKENS / REQUEST_COUNT
        avg_completion = TOTAL_COMPLETION_TOKENS / REQUEST_COUNT

        print(
            f"\n[USAGE] requests={REQUEST_COUNT}, "
            f"total_prompt_tokens={TOTAL_PROMPT_TOKENS}, "
            f"total_completion_tokens={TOTAL_COMPLETION_TOKENS}, "
            f"avg_prompt_tokens_per_req={avg_prompt:.1f}, "
            f"avg_completion_tokens_per_req={avg_completion:.1f}"
        )
        print(
            f"[COST] input_tokens_k={total_prompt_k:.2f}, "
            f"output_tokens_k={total_completion_k:.2f}, "
            f"estimated_input_cost=${cost_input:.4f}, "
            f"estimated_output_cost=${cost_output:.4f}, "
            f"estimated_total_cost=${total_cost:.4f}"
        )


if __name__ == "__main__":
    main()
