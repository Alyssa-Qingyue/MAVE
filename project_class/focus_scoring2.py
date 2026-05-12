'''
Engagement scoring (fine): read minute-level and grid-level preprocessor output, score every 3-minute window.
LLM → anchor + acceptable_deviation → reconstruct_confidence → expected MAE → engagement.

调参重点：alpha，alpha越小，confidence越偏向于anchor，alpha越大，confidence越偏向于0和3。
'''

# client = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"]
# )
# MODEL = "gpt-4o-mini"
# TIMEOUT = 60

import os
import json
from openai import OpenAI
import math

# ---------- LLM CONFIG ----------
# Supports two modes: prefer OpenRouter; fallback to official OpenAI API
TIMEOUT = 60

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    # Use OpenRouter
    MODEL = "openai/gpt-4o-mini"
    client = OpenAI(
        api_key=_openrouter_key.strip(),
        base_url="https://openrouter.ai/api/v1"
    )
elif _openai_key and _openai_key.strip():
    # Use official OpenAI API
    MODEL = "gpt-4o-mini"
    client = OpenAI(api_key=_openai_key.strip())
else:
    raise ValueError(
        "API key is not set. Please configure one of the following:\n"
        "  - set OPENROUTER_API_KEY=your_key   (OpenRouter)\n"
        "  - set OPENAI_API_KEY=your_key       (Official OpenAI API)"
    )

# Input: minute-level and grid-level output from preprocessor.py
PREPROCESSOR_OUTPUT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/preprocessor_output_now"
# Output one scoring file every 3 minutes:
# engagement_0m_3m.json, engagement_3m_6m.json, ...
OUTPUT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/focus_output_now"
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
    """minute_00.json -> 0, minute_03.json -> 3"""
    try:
        return int(fname.replace("minute_", "").replace(".json", ""))
    except Exception:
        return 0


def group_minutes_by_window(minute_fnames, window_minutes=3):
    """
    Build windows every `window_minutes` minutes.
    Return [(window_label, [fname1, fname2, ...]), ...]
    Example: (0m_3m, [minute_00, minute_01, minute_02]), (3m_6m, [minute_03, 04, 05]), ...
    """
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

# ENGAGEMENT PROMPT (TEXT-BASED)
# ENGAGEMENT PROMPT (TEXT-BASED WITH EXPERT SCORING LOGIC)
# ENGAGEMENT PROMPT (TEXT-BASED)
ENGAGEMENT_PROMPT = """
You are a behavioral engagement evaluation model.

You are given observational evidence (scene descriptions from visual frames) describing a person's
visible behavior across a 3-minute window (three 1-minute segments, one paragraph per minute).

### EXPERT SCORING LOGIC (Reference for scoring)
High Score (Focus Indicators): Nodding, Moving closer to screen, Active hand movements, Speaking.
Mid Score: Smiling, Hand on mouth (thinking), Consistent pose, Gesture+Speaking.
Low Score (Distraction Indicators): Yawning, Slouching, Hand at back of head, Playing hands.

Engagement is defined on FOUR ordered levels:
0 - Not Engaged:
- frequent looking away or down
- obvious inattentive behavior（(Yawning, Slouching) ）

1 - Barely Engaged:
- weak or unstable attention
- inconsistent orientation or posture
- mostly low-score or inconsistent behaviors.

2 - Engaged:
- generally oriented forward
- stable posture most of the time
- mid-score behaviors (Smile, Thinking), minor lapses.

3 - Highly Engaged:
- this is the DEFAULT level when evidence shows high-score indicators (Nodding, Leaning in) or stable forward orientation and posture
- no dominant or sustained distraction
- brief lapses are acceptable

You must make a holistic judgment based on the OVERALL tendency across all segments.

Strict calibration policy:
- Do not avoid low scores. Use 0 and 1 whenever evidence supports them.
- Do not avoid high scores. Use 3 whenever strong forward/stable engagement is evident.
- Do NOT use Level 2 as a safe middle ground. If the person shows recurring distractions or "passive" looking, you can consider drop to Level 1.
- If deciding between 2 and 3, lean toward 3 when there is no strong evidence of sustained distraction.
- If deciding between 1 and 2, use 1 when attention is unstable or often not forward.


---

### ORDINAL JUDGMENT INSTRUCTIONS (IMPORTANT)

Based on the definitions above, first choose the SINGLE engagement
level (0–3) that best represents the overall behavior.
This will be referred to as the ANCHOR level.

Then, assess whether neighboring engagement levels could still be
reasonable interpretations of the same evidence, considering the
ORDER of the levels.

Specifically, judge the following ordinal deviations from the anchor:

- Is a ONE-LEVEL LOWER interpretation (anchor - 1) still plausible?
- Is a ONE-LEVEL HIGHER interpretation (anchor + 1) still plausible?
- Is a TWO-LEVEL LOWER interpretation (anchor - 2) plausible?
- Is a TWO-LEVEL HIGHER interpretation (anchor + 2) plausible?

These judgments must respect ordinal distance:
- A two-level deviation should NOT be considered plausible
  if the corresponding one-level deviation is not plausible.
- Larger deviations should be treated as less plausible
  than smaller deviations.

Do NOT assign probabilities or confidence scores directly.
Only make binary plausibility judgments for each deviation.

---

Return STRICT JSON ONLY:

{{
  "anchor": int,
  "acceptable_deviation": {{
    "-1": bool,
    "+1": bool,
    "-2": bool,
    "+2": bool
  }},
  "reasoning": "Concise explanation grounded in the evidence"
}}

## ATTENTION_EVIDENCE
{attention_text}

"""
def reconstruct_confidence_from_ordinal(anchor, acceptable_deviation, alpha=1.0):
    """
    Sakai-style ordinal-consistent confidence reconstruction.
    anchor: int (0-3)
    acceptable_deviation: dict with keys "-1","+1","-2","+2"
    """
    MAX_DISTANCE = 3
    distances = {}

    for label in range(4):
        d = abs(label - anchor)

        if d == 0:
            distances[label] = 0
        elif d == 1:
            key = "+1" if label > anchor else "-1"
            distances[label] = 1 if acceptable_deviation.get(key, False) else MAX_DISTANCE
        elif d == 2:
            key = "+2" if label > anchor else "-2"
            distances[label] = 2 if acceptable_deviation.get(key, False) else MAX_DISTANCE
        else:
            distances[label] = MAX_DISTANCE

        # enforce monotonic ordinal consistency
        distances[label] = max(distances[label], d)

    # convert distances -> weights (exponential decay)
    weights = {k: math.exp(-alpha * distances[k]) for k in distances}
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Invalid ordinal weights")

    confidence = {str(k): weights[k] / total for k in weights}
    return confidence

def choose_by_expected_mae(confidence_dict):
    probs = [confidence_dict[str(i)] for i in range(4)]
    best_k = None
    best_loss = float("inf")

    for k in range(4):
        loss = sum(abs(k - i) * probs[i] for i in range(4))
        if loss < best_loss:
            best_loss = loss
            best_k = k

    return best_k

def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

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

        # If all minute_*.json files in this 3-minute window have no non-empty text,
        # treat it as a PPT lecture (no usable student video evidence) and skip scoring output.
        has_any_evidence = False
        for data in minute_data:
            if not isinstance(data, dict):
                continue
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    has_any_evidence = True
                    break
            if has_any_evidence:
                break

        if not has_any_evidence:
            print(f"[SKIP] {window_label}: no preprocessor output (likely PPT lecture), skip engagement scoring.")
            continue

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

            anchor = result["anchor"]
            acceptable_deviation = result["acceptable_deviation"]

            confidence = reconstruct_confidence_from_ordinal(
                anchor=anchor,
                acceptable_deviation=acceptable_deviation,
                alpha=0.7
            )

            engagement = choose_by_expected_mae(confidence)

            results[grid_id] = {
                "engagement": engagement,
                "confidence": confidence,
                "anchor": anchor,
                "acceptable_deviation": acceptable_deviation,
                "reasoning": result["reasoning"],
                "num_segments": len(blocks)
            }
            print(f"[DONE] {window_label} {grid_id} -> {engagement}")

        out_path = os.path.join(OUTPUT_DIR, f"engagement_{window_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
