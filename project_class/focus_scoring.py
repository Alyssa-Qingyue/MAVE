'''
Engagement scoring: Pre-Decision Binary Check (Structured reasoning for evaluation).

Pipeline:
  The LLM first performs two binary checks (extreme inattention / extreme engagement),
  then chooses middle level 1 or 2 for non-extreme cases.
  Decision: extreme_inattention -> 0; extreme_engagement -> 3; otherwise anchor in {1, 2}.
  Keep the 4-level engagement definition without ordinal reconstruction or B3/MAE.
  
[USAGE] requests=500, total_prompt_tokens=633274, 
total_completion_tokens=43210, avg_prompt_tokens_per_req=1266.5, avg_completion_tokens_per_req=86.4
[COST] input_tokens_k=633.27, output_tokens_k=43.21, 
estimated_input_cost=$0.0950, estimated_output_cost=$0.0259, estimated_total_cost=$0.1209
'''

import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
MODEL = "gpt-4o-mini"
TIMEOUT = 60

# Input: minute-level, grid-level output from preprocessor.py
# (each minute_XX.json has format {grid_id: "description text"})
PREPROCESSOR_OUTPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project_class\output\preprocessor_output"
# Output one score file every 3 minutes:
# engagement_0m_3m.json, engagement_3m_6m.json, ...
OUTPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project_class\output\focus_output"
WINDOW_MINUTES = 3  # One window per 3 minutes (3 x 1-minute segments merged into one scoring pass)

# Token and cost statistics (prices can be updated to latest OpenAI pricing)
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
PRICE_INPUT_PER_1K = 0.00015   # USD per 1K prompt tokens (gpt-4o-mini example)
PRICE_OUTPUT_PER_1K = 0.00060  # USD per 1K completion tokens



def get_minute_files():
    """Find all minute files from preprocessor output and sort by minute index."""
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
    """minute_00.json -> 0, minute_03.json -> 3"""
    try:
        return int(fname.replace("minute_", "").replace(".json", ""))
    except Exception:
        return 0


def group_minutes_by_window(minute_fnames, window_minutes=3):
    """
    Build fixed windows every `window_minutes` minutes with consecutive 1-minute files.
    Return [(window_label, [fname1, fname2, ...]), ...]
    Example for window_minutes=3: (0m_3m, [minute_00.json, minute_01.json, minute_02.json]), (3m_6m, [minute_03, 04, 05]), ...
    """
    if not minute_fnames:
        return []
    # Sort by minute index
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

# ENGAGEMENT PROMPT: Pre-Decision Binary Check (binary extreme check first, then middle level)
ENGAGEMENT_PROMPT = """
You are a behavioral engagement evaluation model. You MUST use the full scale 0–3.
Do not avoid 0 or 1 when the evidence supports them; many segments are legitimately low engagement.

You are given observational evidence (scene descriptions from visual frames) describing a person's
visible behavior across a 3-minute window (three 1-minute segments, one paragraph per minute).

The evidence is standardized narrative from preprocessor output (one description per minute per grid).

Your task is to assess the engagement level (0–3) based strictly on the observable evidence.

---

### FOUR LEVELS

0 - Not Engaged (use when inattention is clear or dominant):
- frequent looking away or down, obvious inattentive behavior
- distraction is the main pattern (e.g. repeatedly looking at phone, away from screen, head down for long periods)

1 - Barely Engaged (use when attention is weak or unstable):
- weak or unstable attention, inconsistent orientation or posture
- mixed engagement: some attention but often not forward, or flickering focus

2 - Engaged:
- generally oriented forward, stable posture most of the time, attentive with minor lapses

3 - Highly Engaged:
- stable forward orientation and posture, no dominant or sustained distraction, brief lapses acceptable

---

### PRE-DECISION BINARY CHECK (do this BEFORE choosing a level)

Answer these two questions honestly. Many segments will correctly get 0 or 3.

1) Does the evidence show **sustained or repeated inattention**?
   Examples: repeated looking away/down, obvious distraction, phone use, head down for much of the window.
   If YES → set "extreme_inattention": true (final level = 0). Do not hesitate when the evidence is there.

2) Does the evidence show **consistently strong engagement** with no sustained distraction?
   Examples: stable forward orientation, attentive throughout, at most brief lapses.
   If YES → set "extreme_engagement": true (final level = 3).

**Only if BOTH are false** (no clear 0 and no clear 3), then set "anchor" to 1 or 2:
- Prefer 1 when attention is weak, inconsistent, or often not forward.
- Prefer 2 when generally engaged with minor lapses.

- At most one of extreme_inattention and extreme_engagement may be true.
- When neither extreme applies, "anchor" must be 1 or 2.
- Calibration: in real data, 0 and 1 are common; use them whenever the evidence supports low engagement.
---

Return STRICT JSON ONLY:

{{
  "extreme_inattention": bool,
  "extreme_engagement": bool,
  "anchor": 1 or 2 (only used when both extreme flags are false),
  "reasoning": "Concise explanation grounded in the evidence"
}}

## ATTENTION_EVIDENCE
{attention_text}

"""


def engagement_from_binary_check(result: dict) -> int:
    """
    Pre-Decision Binary Check decision:
    extreme_inattention -> 0; extreme_engagement -> 3;
    otherwise anchor in {1, 2} (clipped if out of range).
    """
    if result.get("extreme_inattention"):
        return 0
    if result.get("extreme_engagement"):
        return 3
    anchor = result.get("anchor", 1)
    return min(2, max(1, int(anchor)))


def call_llm(prompt: str) -> dict:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        timeout=TIMEOUT,
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
        TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
    REQUEST_COUNT += 1

    content = response.choices[0].message.content.strip()

    # Support responses wrapped in ```json fences
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)



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
            prompt = ENGAGEMENT_PROMPT.format(attention_text=evidence_text)

            try:
                result = call_llm(prompt)
            except Exception as e:
                results[grid_id] = {
                    "engagement": None,
                    "reasoning": str(e)
                }
                continue

            engagement = engagement_from_binary_check(result)

            results[grid_id] = {
                "engagement": engagement,
                "extreme_inattention": result.get("extreme_inattention", False),
                "extreme_engagement": result.get("extreme_engagement", False),
                "anchor": result.get("anchor"),
                "reasoning": result.get("reasoning", ""),
                "num_segments": len(blocks)
            }
            print(f"[DONE] {window_label} {grid_id} -> {engagement}")

        out_path = os.path.join(OUTPUT_DIR, f"engagement_{window_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # Token usage, API cost, and efficiency
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
