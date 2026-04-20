'''
[USAGE] requests=500, total_prompt_tokens=431274, total_completion_tokens=167966, 
avg_prompt_tokens_per_req=862.5, avg_completion_tokens_per_req=335.9
[COST] input_tokens_k=431.27, output_tokens_k=167.97, 
estimated_input_cost=$0.0647, estimated_output_cost=$0.1008, estimated_total_cost=$0.1655
'''
import os
import json
import glob
import re
from collections import defaultdict
from statistics import mean
from openai import OpenAI

# Basic configuration
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

ENGAGEMENT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\focus_output"
TRANSCRIPT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\whisper_out"
OUTPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\teaching_report"

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
PRICE_INPUT_PER_1K = 0.00015   
PRICE_OUTPUT_PER_1K = 0.00060  


def call_gpt(system_prompt: str, user_prompt: str) -> str:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
        TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
    REQUEST_COUNT += 1
    return response.choices[0].message.content




# 1) Convert 3-minute engagement files into 6-minute windows

ENGAGEMENT_PATTERN = re.compile(r"engagement_(\d+)m_(\d+)m\.json")
TRANSCRIPT_PATTERN = re.compile(r"transcript_(\d+)m_(\d+)m\.txt")

def list_3min_engagement_files(folder):
    """List engagement_*m_*m.json files sorted by start minute."""
    out = []
    for path in glob.glob(os.path.join(folder, "engagement_*.json")):
        fname = os.path.basename(path)
        m = ENGAGEMENT_PATTERN.match(fname)
        if not m:
            continue
        start_min, end_min = int(m.group(1)), int(m.group(2))
        out.append((start_min, end_min, path))
    out.sort(key=lambda x: x[0])
    return out


def group_into_6min_windows(three_min_list):
    """Merge every two entries into a 6-minute window. Return [(window_label, [path1, path2]), ...]."""
    windows = []
    i = 0
    while i + 1 < len(three_min_list):
        s1, e1, p1 = three_min_list[i]
        s2, e2, p2 = three_min_list[i + 1]
        if e1 != s2:
            i += 1
            continue
        label = f"{s1}m_{e2}m"
        windows.append((label, [p1, p2]))
        i += 2
    return windows


def aggregate_engagement_for_6min(paths):
    """
    Aggregate two 3-minute engagement files within one 6-minute window.
    Order:
    1) Aggregate all raw engagement observations to build score distribution.
    2) Compute average per student (grid).
    Return (student_data, distribution_from_raw).
    """
    all_grids = set()
    engagements_by_grid = defaultdict(list)
    reasonings_by_grid = defaultdict(list)
    # 1) Collect all raw observations first (for distribution; no averaging yet)
    all_raw_scores = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for grid_id, g in data.items():
            if not isinstance(g, dict) or "engagement" not in g:
                continue
            eng = g["engagement"]
            if eng is None:
                continue
            try:
                e = int(eng)
            except (TypeError, ValueError):
                continue
            if e not in (0, 1, 2, 3):
                continue
            all_grids.add(grid_id)
            engagements_by_grid[grid_id].append(e)
            reasonings_by_grid[grid_id].append(g.get("reasoning", ""))
            all_raw_scores.append(e)

    n_obs = len(all_raw_scores)
    counts = {str(i): all_raw_scores.count(i) for i in range(4)}
    proportions = (
        {str(i): round(counts[str(i)] / n_obs, 4) for i in range(4)} if n_obs else {str(i): 0.0 for i in range(4)}
    )
    distribution_from_raw = {
        "total_observations": n_obs,
        "counts_by_level": counts,
        "proportion_by_level": proportions,
    }

    # 2) Compute each student's average score in this 6-minute window
    # (average across two 3-minute chunks, then round)
    student_data = {}
    for grid_id in sorted(all_grids):
        eng_list = engagements_by_grid[grid_id]
        reasoning_list = reasonings_by_grid[grid_id]
        avg_eng = round(mean(eng_list)) if eng_list else 0
        reasoning = " | ".join(r for r in reasoning_list if r)
        student_data[grid_id] = {"engagement": avg_eng, "reasoning": reasoning}

    return student_data, distribution_from_raw


def list_6min_transcript_windows(folder):
    """
    List transcript_{start}m_{end}m.txt (typically one per 6 minutes), sorted by start_min.
    Return [(start_min, end_min, path), ...]
    """
    out = []
    for path in glob.glob(os.path.join(folder, "transcript_*.txt")):
        fname = os.path.basename(path)
        m = TRANSCRIPT_PATTERN.match(fname)
        if not m:
            continue
        start_min, end_min = int(m.group(1)), int(m.group(2))
        out.append((start_min, end_min, path))
    out.sort(key=lambda x: x[0])
    return out


def build_engagement_index(three_min_list):
    """(start_min, end_min) -> path"""
    idx = {}
    for s, e, p in three_min_list:
        idx[(s, e)] = p
    return idx


def get_continuous_3min_segments_for_window(start_min: int, end_min: int, step_min: int = 3):
    """
    Split [start_min, end_min) into continuous `step_min` segments.
    Example: 0-6 -> [(0,3), (3,6)].
    """
    segments = []
    cur = start_min
    while cur + step_min <= end_min:
        segments.append((cur, cur + step_min))
        cur += step_min
    return segments


def summarize_students(student_data, distribution_from_raw):
    """
    Build per-student summary and overall_engagement from raw-observation distribution.
    In overall section, keep distribution first, then overall mean of student averages.
    """
    student_summary = {}
    per_student_averages = []
    for grid_id, info in student_data.items():
        engagement = info["engagement"]
        reasoning = info.get("reasoning", "")
        student_summary[grid_id] = {"engagement": engagement, "reasoning": reasoning}
        per_student_averages.append(engagement)

    mean_per_student = (
        round(mean(per_student_averages), 2) if per_student_averages else 0.0
    )
    overall = {
        "score_distribution_raw_observations": distribution_from_raw,
        "num_students": len(per_student_averages),
        "mean_engagement_per_student": mean_per_student,
        "distribution": distribution_from_raw["proportion_by_level"],
        "mean_engagement": mean_per_student,
    }
    return student_summary, overall


def build_engagement_evidence_for_prompt(student_summary, overall_engagement):
    """
    Compress engagement data into a high-decision-value summary so suggestions become more concrete and actionable.
    """
    low_students = []
    high_students = []
    for grid_id, info in student_summary.items():
        score = info.get("engagement", 0)
        reasoning = (info.get("reasoning", "") or "").strip()
        item = {
            "grid_id": grid_id,
            "engagement": score,
            "reasoning": reasoning[:240],
        }
        if score <= 1:
            low_students.append(item)
        elif score >= 3:
            high_students.append(item)

    low_students.sort(key=lambda x: x["engagement"])
    high_students.sort(key=lambda x: -x["engagement"])

    dist = (overall_engagement or {}).get("score_distribution_raw_observations", {})
    proportions = dist.get("proportion_by_level", {}) if isinstance(dist, dict) else {}

    evidence = {
        "num_students": (overall_engagement or {}).get("num_students", len(student_summary)),
        "mean_engagement_per_student": (overall_engagement or {}).get("mean_engagement_per_student", 0.0),
        "distribution_by_level": proportions,
        "low_engagement_students_top3": low_students[:3],
        "high_engagement_students_top2": high_students[:2],
        "primary_risk_signal": (
            "low_engagement_cluster"
            if len(low_students) >= max(1, len(student_summary) // 3)
            else "mixed_or_stable"
        ),
    }
    return evidence



# 3) Teaching transcript analysis (LLM)

def analyze_transcript_with_llm(transcript_text):
    system_prompt = (
        "You are an expert in classroom teaching analysis. "
        "Analyze the transcript to extract teaching style and structure."
    )

    user_prompt = f"""
Analyze the following teaching transcript and summarize key characteristics.

Return STRICT JSON with:
- teaching_style
- interaction_level
- pacing_comment

Transcript:
\"\"\"
{transcript_text}
\"\"\"
"""
    raw = call_gpt(system_prompt, user_prompt)

    try:
        return safe_json_loads(raw)
    except json.JSONDecodeError:
        print("[WARN] Failed to parse JSON from LLM. Raw response below:")
        print("-------- LLM RAW RESPONSE --------")
        print(raw)
        print("---------------------------------")

        # Fallback to a safe structure so the main pipeline does not crash
        return {
            "teaching_style": "unknown",
            "explanation_ratio": 0.0,
            "interaction_ratio": 0.0,
            "example_ratio": 0.0,
            "pacing_comment": "LLM response could not be parsed."
        }



# 4) Teaching suggestion generation (LLM)

def generate_teaching_suggestions(overall_engagement, student_summary, teaching_analysis, transcript_text=""):
    system_prompt = (
        "You are an experienced educational coach. "
        "Produce rigorous, evidence-grounded, and implementation-ready teaching suggestions. "
        "Your output must be internally consistent, specific, and practically executable in the next class session. "
        "Never fabricate evidence."
    )
    transcript_block = ""
    if transcript_text and transcript_text.strip():
        transcript_block = f"""

Transcript for this 6-minute window (use to give concrete suggestions on wording, pacing, or content):
\"\"\"
{transcript_text[:12000]}
\"\"\"
"""


    engagement_evidence = build_engagement_evidence_for_prompt(student_summary, overall_engagement)

    user_prompt = f"""
You are given empirical data and analysis from a real classroom session.

Use ONLY the information provided below.
Do NOT assume any facts, behaviors, or classroom conditions that are not explicitly supported.

Overall student engagement (this window):
{json.dumps(overall_engagement, ensure_ascii=False, indent=2)}

Per-student engagement summary:
{json.dumps(student_summary, ensure_ascii=False, indent=2)}

Teaching analysis (from transcript):
{json.dumps(teaching_analysis, ensure_ascii=False, indent=2)}
{transcript_block}

Compact engagement evidence for intervention design:
{json.dumps(engagement_evidence, ensure_ascii=False, indent=2)}

Task:
Write a small set of teaching suggestions for this real classroom window that will help improve student engagement and support the teacher’s professional growth.

Important grounding & consistency rules (follow strictly, without mentioning these rules in your output):
- Use ONLY the evidence given above (engagement numbers, per-student summary, teaching analysis, and the optional transcript block). Do NOT invent facts.
- Do NOT make contradictory recommendations across suggestions. If two actions would conflict, keep only the clearer one and adjust phrasing.
- Keep the wording logically coherent: avoid “because X happened” when X is not explicitly supported by the inputs.
- When the transcript is provided, anchor at least 1–2 suggestions to specific transcript moments/content to make the advice concrete.

Quality target (optimize for high-evidence, high-actionability feedback):
- Suggestions must read like real coaching notes, not generic advice.
- Use explicit evidence anchors (engagement level/distribution, low-engagement student patterns, or specific transcript moments).
- Ensure recommendation logic is coherent across the full set (no hidden conflicts).

Required structure for each suggestion (continuous paragraph, 2-4 sentences):
1) Evidence anchor: cite one concrete observed signal from the provided data.
2) Diagnosis: explain what that signal likely means for teaching/learning in this window.
3) Action: provide one next-class action with timing/trigger (when to do it).
4) Success check: provide one observable indicator to verify improvement.

Intervention quality constraints:
- At least 2 suggestions must directly target low-engagement signals or low-engagement students listed above.
- At least 1 suggestion must preserve/scale an existing strength (not only fix weaknesses).
- Do not repeat the same action in different wording.
- Prioritize high-leverage, near-term actions for the next class session.
- Avoid vague verbs like "improve", "enhance", "consider" unless followed by specific operational steps.

Output format:
Return STRICT JSON in the following format (no extra keys, no extra text):

{{
  "teaching_suggestions": [
    "<continuous readable suggestion 1 (1-3 sentences)>",
    "<continuous readable suggestion 2 (1-3 sentences)>",
    "<continuous readable suggestion 3 (1-3 sentences)>",
    "<optional suggestion 4>",
    "<optional suggestion 5>"
  ]
}}
"""

    raw = call_gpt(system_prompt, user_prompt)
    return safe_json_loads(raw)
# Normalize JSON output format
def safe_json_loads(text: str):
    """
    Robustly extract and parse JSON from LLM output.
    Handles markdown ```json fences and extra text.
    """
    if not text:
        raise ValueError("Empty response")

    # Remove markdown code fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]

    # Extract the first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")

    json_str = text[start:end + 1]
    return json.loads(json_str)


# 5) Main pipeline: aggregate engagement + transcript per 6-minute window, then output one teaching suggestion report

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    three_min_list = list_3min_engagement_files(ENGAGEMENT_DIR)
    engagement_idx = build_engagement_index(three_min_list)

    transcript_windows = list_6min_transcript_windows(TRANSCRIPT_DIR)
    if not transcript_windows:
        print(f"[ERROR] No transcript_{'{start}'}m_{'{end}'}m.txt found in {TRANSCRIPT_DIR}")
        return

    for t_start, t_end, transcript_path in transcript_windows:
        transcript_window_label = f"{t_start}m_{t_end}m"
        print(f"\n[WINDOW] transcript {transcript_window_label}")

        # 1) Use continuous 3-minute segments inside this transcript window
        segments = get_continuous_3min_segments_for_window(t_start, t_end, step_min=3)

        # 2) Match engagement outputs by time (some segments may be missing)
        candidate_paths = []
        contributed_segments = []
        for s, e in segments:
            p = engagement_idx.get((s, e))
            if p and os.path.isfile(p):
                candidate_paths.append(p)
                contributed_segments.append((s, e))

        # 3) Aggregate engagement using only existing 3-minute files
        student_data, distribution_from_raw = aggregate_engagement_for_6min(candidate_paths)

        # If this transcript window has no engagement output, treat it as PPT lecture
        # (or no student video evidence), and skip report generation
        if not student_data:
            print(f"  [SKIP] {transcript_window_label}: no upstream engagement output, skip teaching suggestions.")
            continue

        # 4) If some 3-minute segments are missing within this 6-minute window,
        #    rename report by the actual covered time range.
        #    Example: transcript 0-6 but only 0-3 has engagement -> teaching_suggestions_0m_3m.json
        actual_start = min(s for s, _ in contributed_segments) if contributed_segments else t_start
        actual_end = max(e for _, e in contributed_segments) if contributed_segments else t_end
        window_label = f"{actual_start}m_{actual_end}m"

        student_summary, overall_engagement = summarize_students(student_data, distribution_from_raw)

        # 5) Transcript text for this window (strictly aligned by output transcript filename)
        transcript_text = ""
        if os.path.isfile(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            # Should not happen in theory: transcript_windows are enumerated from actual files
            print(f"  [WARN] No transcript file: {transcript_path}")

        # 3) Teaching transcript analysis (LLM)
        teaching_analysis = analyze_transcript_with_llm(transcript_text or "(No transcript for this window.)")

        # 4) Teaching suggestions (combine engagement + transcript for concrete recommendations)
        try:
            suggestions = generate_teaching_suggestions(
                overall_engagement,
                student_summary,
                teaching_analysis,
                transcript_text=transcript_text,
            )
            suggestion_list = suggestions.get("teaching_suggestions", [])
        except Exception as e:
            print(f"  [WARN] LLM suggestions failed: {e}")
            suggestion_list = []

        # 5) Write report for this window
        report = {
            # window: used for output filename (may be narrowed to the actual engagement-covered range)
            "window": window_label,
            # transcript_window: transcript time window linked to this report (fixed from output transcript filename)
            "transcript_window": transcript_window_label,
            "engagement_segments_expected": [{"start_min": s, "end_min": e} for s, e in segments],
            "engagement_segments_found": [{"start_min": s, "end_min": e} for s, e in contributed_segments],
            "overall_engagement": overall_engagement,
            "student_engagement_summary": student_summary,
            "teaching_analysis": teaching_analysis,
            "teaching_suggestions": suggestion_list,
        }
        out_path = os.path.join(OUTPUT_DIR, f"teaching_suggestions_{window_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  [SAVED] {out_path}")

    # Token usage, efficiency, and cost
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

    print(f"\n[DONE] All reports saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
