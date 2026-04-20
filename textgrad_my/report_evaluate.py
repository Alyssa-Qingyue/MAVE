'''
Evaluate teaching reports (6-minute teaching suggestions) together with transcript text (whisper_out).
Flow: load suggestions + transcript by window -> multi-dimension evaluators (with transcript context) -> Dean aggregation -> safety gates -> output.

[TOKEN] total_prompt_tokens=334317, total_completion_tokens=17317, total_tokens=351634, requests=170
[EFFICIENCY] avg_prompt_tokens_per_req=1966.6, avg_completion_tokens_per_req=101.9, avg_total_tokens_per_req=2068.4
[COST] input_tokens_k=334.32, output_tokens_k=17.32, estimated_input_cost=$0.0501, estimated_output_cost=$0.0104, estimated_total_cost=$0.0605
'''
import os
import json
import glob
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

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
TIMEOUT = 60
TEMPERATURE = 0  # Evaluation must use temperature 0

# Input: TextGrad training results
# (including per-window step_records and text_gradient)
INPUT_JSONL_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\textgrad_my\Zoom Class Meeting Downing Soc 100 4.19.2021\optimized_transcripts_with_gradients.jsonl"
# Output: text_gradient evaluation results
OUTPUT_JSON_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\textgrad_my\Zoom Class Meeting Downing Soc 100 4.19.2021\evaluation_text_gradients.json"

# Original transcript directory:
# if input_transcript_path in jsonl is invalid, fallback to transcript_{window}.txt from this directory
TRANSCRIPT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project_class\project-teacher\whisper_out"
# Maximum transcript characters injected into evaluator prompts
# (default aligned with project-teacher)
MAX_TRANSCRIPT_CHARS = int(os.getenv("TEXTGRAD_EVAL_MAX_TRANSCRIPT_CHARS", "8000"))

# Token and cost statistics (prices can be updated to latest OpenAI pricing)
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
PRICE_INPUT_PER_1K = 0.00015   # USD per 1K prompt tokens (gpt-4o-mini)
PRICE_OUTPUT_PER_1K = 0.00060  # USD per 1K completion tokens

# Evaluation dimensions (must exactly match keys in DIMENSION_DEFINITIONS)
EVALUATION_DIMENSIONS = [
    "Alignment",
    "Comprehensiveness",
    "Directionality",
    "Consistency",
    "Actionability",
    "Tone",
]

# Consistency is a critical dimension: if <= 2, trigger a strong penalty
CONSISTENCY_DIMENSION = "Consistency"
CONSISTENCY_SEVERE_THRESHOLD = 2



# Evaluator prompt (includes transcript + student engagement evidence to avoid false "missing evidence" penalties)

EVALUATOR_PROMPT_TEMPLATE = """
You are an expert evaluator specializing in educational feedback and teaching improvement assessment.

Your task is to evaluate teaching suggestions for one 6-minute window according to ONE specific criterion.

Important: The evaluated text is a TextGrad `text_gradient` (instructional improvement feedback). Treat it like teaching feedback to a teacher.

Grounding / hallucination (especially for "Consistency"): When the "Teaching transcript" section appears below, treat it as the **authoritative factual reference** for what actually occurred or was said in this time window. Penalize the feedback if it asserts specific facts, direct quotes, events, roles, or classroom details that **contradict** the transcript or are **fabricated as if observed** when the transcript gives no support. Generic pedagogical advice that neither contradicts nor falsely claims transcript-specific evidence should not be scored as hallucination. If the transcript section is absent, judge Consistency mainly on **internal** coherence of the feedback and flag only clear self-contradictions.

Student engagement evidence may be empty. Do NOT penalize for "lack of engagement evidence" when that block is empty—evaluate against transcript + feedback text.

Context provided:
1. Student engagement evidence (optional; may be empty).
2. Teaching feedback to evaluate (`text_gradient`, shown as teaching suggestions below).
3. Teaching transcript (when present): primary reference for alignment and hallucination checks.

Be strict and objective on suggestion quality. Do not lower scores solely because engagement numbers are not repeated in the suggestion text.

Evaluation Criterion:
{criterion_name}

Definition:
{criterion_definition}

{engagement_evidence_block}

Teaching Suggestion (to evaluate):
{feedback}
{transcript_block}

Scoring Rubric (1–5):
1 = Very poor / fails this criterion
3 = Acceptable but with clear weaknesses
5 = Excellent and fully satisfies this criterion

For hallucination-type criteria, use the definition's scoring guidance. Be conservative: prefer lower scores when in doubt.

Output your evaluation strictly in JSON format:
{{
  "dimension": "{criterion_name}",
  "score": <integer 1–5>,
  "justification": "<concise explanation>"
}}
"""



# Definitions for each dimension

DIMENSION_DEFINITIONS = {

    # 1) Alignment
    "Alignment": (
        "Measures the degree to which the feedback corresponds to the pre-defined teaching goals, "
        "curriculum standards, and the specific instructional context provided in the transcripts. "
        "The feedback should stay focused on the intended learning outcomes and the teacher's "
        "actual performance, avoiding tangential or irrelevant pedagogical advice."
    ),

    # 2) Comprehensiveness
    "Comprehensiveness": (
        "Evaluates the balanced identification of both strengths and weaknesses. "
        "A high score requires the model to explicitly recognize effective practices (Strength) "
        "while clearly and accurately diagnosing areas for improvement (Weakness). "
        "It should provide a holistic view of the teaching session rather than focusing "
        "disproportionately on only one side."
    ),

    # 3) Directionality
    "Directionality": (
        "Assesses the structural scaffolding of the feedback across three temporal layers: "
        "1) Feed back: Does it accurately analyze the current state/past actions? "
        "2) Feed up: Does it clarify or reinforce the overarching instructional goals? "
        "3) Feed forward: Does it provide a clear, forward-looking growth path? "
        "High-quality feedback must connect where the teacher is, where they are going, and how to get there."
    ),

    # 4) Consistency
    "Consistency": (
        "Ensures internal logical coherence and the absence of transcript-inconsistent fabrication. "
        "Use the provided teaching transcript as the primary factual anchor when present: "
        "the feedback must not assert transcript-specific claims that contradict the transcript (Input-conflicting) "
        "or invent classroom events/quotes/behaviors not supported by it; "
        "it must not be self-contradictory (Context-conflicting); "
        "general pedagogy that does not over-claim transcript facts is acceptable. "
        "Scoring guidance: 5 = No material hallucinations relative to transcript / internally coherent; "
        "1 = Severe contradictions or invented transcript-specific facts."
    ),

    # 5) Actionability
    "Actionability": (
        "Focuses on the specificity and practicality of the advice. "
        "Suggestions must be concrete, specific, and detailed enough for a teacher to "
        "implement realistically in future sessions. Vague, generic, or abstract advice "
        "(e.g., 'be more engaging') without clear 'how-to' steps should be scored lower."
    ),

    # 6) Tone
    "Tone": (
        "Evaluates the professional and motivational quality of the language. "
        "The feedback should be encouraging, respectful, and supportive, fostering a "
        "growth mindset. It should avoid judgmental, blaming, or overly critical language "
        "that might undermine the recipient's confidence or professional agency."
    ),

}

# Dean prompt template
'''
Dean role:
Integrate dimension scores;
identify critical failures (for example, average may be high,
but hallucination-related consistency is weak, so penalty must be applied);
output the final score.

'''
DEAN_PROMPT_TEMPLATE = """
You are the Dean of LLM Tutors.

You act as a centralized screening and arbitration authority.
Your role is NOT to average scores, but to make a principled,
safety-prioritized, and non-linear final decision.

You are given structured evaluations from expert judges.
Each judge evaluates one single dimension.

Your responsibilities:

1. Consistency First:
   - "Consistency" is a critical reliability dimension.
   - If Consistency score ≤ 2, this must significantly constrain the overall score.
   - Severe consistency failures (hallucination/contradiction risk) should be treated as veto-like issues.

2. Identify Critical Failures:
   - A dimension scored ≤ 2 is considered a severe weakness.
   - Multiple severe weaknesses across core dimensions
     should substantially lower the overall assessment.

3. Non-linear Integration:
   - DO NOT compute or approximate a simple average.
   - Prioritize correctness, safety, and internal consistency
     over stylistic quality or tone.
   - High scores in secondary dimensions (e.g., tone)
     must NOT compensate for failures in safety or correctness.

4. Holistic Judgment:
   - Consider patterns across dimensions.
   - Distinguish between minor weaknesses and structural flaws.
   - Produce a defensible final decision.

5. Calibration (important):
   - If ALL dimension scores are ≥ 4, the overall_score MUST be at least 4 (prefer 4 or 5). Do NOT give 3 when every dimension is 4 or 5.
   - To assign overall_score = 5, require that the majority of dimensions are extremely strong:
     * With the current setup (N = 6 dimensions), give overall_score = 5 when at least 4 dimensions are scored 5
       AND the produced critical_issues list is empty.
   - If no dimension is ≤ 2, there is no severe weakness; overall_score should reflect the actual pattern (typically 4 or 5 when most scores are 4+).
   - Only use overall_score = 3 when there are clearly multiple dimensions at 3 or below, or at least one critical weakness. Do not default to 3 when the evidence supports 4 or 5.

Here are the evaluations:
{evaluations_json}

Produce your final evaluation STRICTLY in JSON format:

{{
  "overall_score": <integer 1–5>,
  "critical_issues": [<list of dimension names that represent severe or structural problems>],
  "decision_rationale": "<brief explanation of the primary factors determining the overall score>",
  "summary": "<concise holistic assessment>"
}}

Scoring Guidance:

5 = Excellent overall: at least 4/6 dimensions are 5 and there are no critical_issues
4 = Strong overall quality, only minor weaknesses (use when most dimensions are 4+ but the “4/6 are 5” condition is not met)
3 = Acceptable but with clear weaknesses (do NOT use when all dimensions are ≥ 4)
2 = Major weaknesses or safety concerns
1 = Severe structural failure or critical hallucination
"""



def safe_json_loads(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty LLM output")

    # Extract the first JSON object (automatically strips ```json ... ``` wrappers)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")

    json_str = match.group(0)
    return json.loads(json_str)

# LLM call function

def call_llm(prompt: str) -> str:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
        TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
    REQUEST_COUNT += 1
    return response.choices[0].message.content


def list_teaching_report_windows(reports_dir: str) -> List[str]:
    """List window labels from teaching_suggestions_*m_*m.json, sorted by time. Example: ['0m_6m', '6m_12m', ...]."""
    pattern = re.compile(r"teaching_suggestions_(\d+)m_(\d+)m\.json")
    windows = []
    for path in glob.glob(os.path.join(reports_dir, "teaching_suggestions_*.json")):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if m:
            windows.append(f"{m.group(1)}m_{m.group(2)}m")
    windows.sort(key=lambda w: (int(w.split("m_")[0]), int(w.split("m_")[1].replace("m", ""))))
    return windows


def load_report_and_transcript(window_label: str) -> tuple:
    """
    Load suggestions, transcript, and engagement evidence for one window.
    Return (feedback_text, transcript_text, engagement_evidence_str).
    engagement_evidence_str helps evaluators understand suggestions were generated from evidence,
    avoiding unfair penalties for "missing engagement evidence".
    """
    report_path = os.path.join(TEACHING_REPORTS_DIR, f"teaching_suggestions_{window_label}.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    suggestions = data.get("teaching_suggestions", [])
    if isinstance(suggestions, list):
        feedback_text = "\n\n".join(str(s) for s in suggestions)
    else:
        feedback_text = str(suggestions)

    transcript_path = os.path.join(TRANSCRIPT_DIR, f"transcript_{window_label}.txt")
    transcript_text = ""
    if os.path.isfile(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()

    # Extract engagement evidence from report for evaluator reference
    engagement_evidence_str = ""
    overall = data.get("overall_engagement")
    student_summary = data.get("student_engagement_summary")
    if overall or student_summary:
        parts = []
        if overall:
            parts.append("Overall engagement (this window): " + json.dumps(overall, ensure_ascii=False))
        if student_summary and isinstance(student_summary, dict):
            # Brief summary: engagement distribution across grids to avoid excessive length
            counts = {}
            for g, info in student_summary.items():
                if isinstance(info, dict) and "engagement" in info:
                    e = info["engagement"]
                    counts[e] = counts.get(e, 0) + 1
            parts.append("Per-student engagement counts (0–3): " + json.dumps(counts, ensure_ascii=False))
        engagement_evidence_str = "\n".join(parts)
    return feedback_text, transcript_text, engagement_evidence_str


def save_evaluation_result(output_path: str, result: Dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
'''
Goal: even if Dean LLM misjudges, deterministic core-dimension constraints still apply
(do not rely only on prompt behavior).
This version implements one key rule:
if Consistency <= 2, aggressively lower the Dean result.
No extra rule penalties are applied to other dimensions; use Dean's holistic judgment directly.
'''
def apply_safety_and_core_gates(dean_result: Dict, evaluator_results: List[Dict]) -> Dict:
    """
    Deterministic penalty applied AFTER dean_result generated by LLM.
    - If Consistency score <= threshold => aggressively lower dean overall score.
    - Otherwise keep dean score unchanged (except calibration floor).
    Returns possibly-modified dean_result with explanation appended to decision_rationale.
    """
    # collect scores map
    score_map = {r["dimension"]: r["score"] for r in evaluator_results}

    rationale_additions = []

    # Key rule: severe Consistency loss triggers a strong penalty
    consistency_score = score_map.get(CONSISTENCY_DIMENSION)
    if isinstance(consistency_score, (int, float)) and consistency_score <= CONSISTENCY_SEVERE_THRESHOLD:
        old = dean_result.get("overall_score")
        # "Aggressive reduction" strategy: cap directly at 2; keep lower value if already lower
        new_overall = min(int(old), 2) if isinstance(old, (int, float)) else 2
        dean_result["overall_score"] = new_overall
        rationale_additions.append(
            f"Consistency severe-loss penalty applied: {CONSISTENCY_DIMENSION}={consistency_score} "
            f"(<= {CONSISTENCY_SEVERE_THRESHOLD}), dean overall capped to {new_overall}."
        )

    # attach explicit critical_issues additions if not already present
    critical = set(dean_result.get("critical_issues", []))
    if isinstance(consistency_score, (int, float)) and consistency_score <= CONSISTENCY_SEVERE_THRESHOLD:
        critical.add(CONSISTENCY_DIMENSION)
    dean_result["critical_issues"] = list(critical)

    # Calibration floor: if all dimensions score >= 4, overall_score must be at least 4
    # (avoid overly conservative Dean score of 3)
    scores = [r.get("score") for r in evaluator_results if isinstance(r.get("score"), (int, float))]
    if len(scores) == len(evaluator_results) and all(s >= 4 for s in scores):
        current = dean_result.get("overall_score")
        if isinstance(current, (int, float)) and current < 4:
            dean_result["overall_score"] = 4
            rationale_additions.append("Calibration: all dimension scores ≥ 4, so overall_score set to at least 4.")

    # For other dimensions: no extra rule penalties; use Dean result directly

    # append to decision_rationale
    prev_rationale = dean_result.get("decision_rationale", "")
    append = " ".join(rationale_additions)
    dean_result["decision_rationale"] = (prev_rationale + " " + append).strip()

    return dean_result


# Single-dimension evaluator (transcript can be passed as context)

def run_single_evaluator(
    dimension: str,
    feedback: str,
    transcript: str = "",
    engagement_evidence: str = "",
) -> Dict:
    transcript_block = ""
    if transcript and transcript.strip():
        transcript_block = (
            "\n\nTeaching transcript for this window (authoritative reference for hallucination / factual alignment):\n\"\"\"\n"
            + transcript[:MAX_TRANSCRIPT_CHARS]
            + "\n\"\"\""
        )
    engagement_evidence_block = ""
    if engagement_evidence and engagement_evidence.strip():
        engagement_evidence_block = (
            "Student engagement evidence available at report generation (do NOT penalize for lack of evidence in suggestion text):\n"
            "```\n" + engagement_evidence.strip()[:2000] + "\n```\n\n"
        )
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        criterion_name=dimension,
        criterion_definition=DIMENSION_DEFINITIONS[dimension],
        feedback=feedback,
        transcript_block=transcript_block,
        engagement_evidence_block=engagement_evidence_block,
    )
    raw_output = call_llm(prompt)
    return safe_json_loads(raw_output)


# Run all evaluators

def run_all_evaluators(feedback: str, transcript: str = "", engagement_evidence: str = "") -> List[Dict]:
    results = []
    for dimension in EVALUATION_DIMENSIONS:
        result = run_single_evaluator(
            dimension=dimension,
            feedback=feedback,
            transcript=transcript,
            engagement_evidence=engagement_evidence,
        )
        results.append(result)
    return results


# Dean aggregation

def run_dean(evaluator_results: List[Dict]) -> Dict:
    evaluations_json = json.dumps(evaluator_results, indent=2)

    dean_prompt = DEAN_PROMPT_TEMPLATE.format(
        evaluations_json=evaluations_json
    )

    raw_output = call_llm(dean_prompt)
    return safe_json_loads(raw_output)



# Main flow

def evaluate_feedback(feedback: str, transcript: str = "", engagement_evidence: str = "") -> Dict:
    evaluator_results = run_all_evaluators(
        feedback, transcript=transcript, engagement_evidence=engagement_evidence
    )
    dean_result = run_dean(evaluator_results)
    dean_result = apply_safety_and_core_gates(dean_result, evaluator_results)
    return {
        "dimension_results": evaluator_results,
        "dean_result": dean_result,
    }


def resolve_transcript_for_window(rec: Dict, window_label: str) -> Tuple[str, str]:
    """
    (transcript_text, path_used). Prefer input_transcript_path in jsonl;
    on failure, fallback to transcript_{window}.txt under
    TEXTGRAD_EVAL_TRANSCRIPT_DIR or TRANSCRIPT_DIR.
    """
    declared = str(rec.get("input_transcript_path", "")).strip()
    if declared and os.path.isfile(declared):
        try:
            with open(declared, "r", encoding="utf-8") as tf:
                return tf.read().strip(), declared
        except OSError:
            pass

    env_dir = os.getenv("TEXTGRAD_EVAL_TRANSCRIPT_DIR", "").strip()
    for base in (env_dir, TRANSCRIPT_DIR):
        if not base:
            continue
        p = os.path.join(base, f"transcript_{window_label}.txt")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as tf:
                    return tf.read().strip(), p
            except OSError:
                continue
    return "", declared if declared else ""


# Entry point

if __name__ == "__main__":
    if not os.path.isfile(INPUT_JSONL_PATH):
        raise FileNotFoundError(f"[ERROR] INPUT_JSONL_PATH not found: {INPUT_JSONL_PATH}")

    def _window_sort_key(label: str):
        m = re.match(r"(\d+)m_(\d+)m", str(label))
        if not m:
            return (10**9, 10**9)
        return (int(m.group(1)), int(m.group(2)))

    all_results = {}
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                all_results[f"line_{line_idx}"] = {"error": f"invalid json: {e}"}
                continue

            window_label = str(rec.get("window", f"unknown_window_{line_idx}"))
            print(f"\n[EVAL] {window_label}")

            # Evaluate text_gradient signal only
            text_gradient = ""
            step_idx = None
            step_records = rec.get("step_records", [])
            if isinstance(step_records, list):
                # Prefer step=0; if missing, use the smallest step
                valid_steps = [s for s in step_records if isinstance(s, dict)]
                if valid_steps:
                    step0 = [s for s in valid_steps if s.get("step") == 0]
                    selected = step0[0] if step0 else min(
                        valid_steps, key=lambda x: int(x.get("step", 10**9))
                    )
                    step_idx = selected.get("step")
                    text_gradient = str(selected.get("text_gradient", "")).strip()

            # Fallback: if top-level text_gradient exists, evaluate it as well
            if not text_gradient:
                text_gradient = str(rec.get("text_gradient", "")).strip()

            if not text_gradient:
                print("  [SKIP] empty text_gradient")
                all_results[window_label] = {"error": "Empty text_gradient"}
                continue

            declared_tp = str(rec.get("input_transcript_path", "")).strip()
            transcript_text, transcript_path_used = resolve_transcript_for_window(rec, window_label)
            if not transcript_text.strip():
                print(
                    "  [WARN] Teaching transcript not loaded; hallucination evaluation will mainly rely on self-consistency of feedback."
                    f" Declared path: {declared_tp or '(none)'}; set TEXTGRAD_EVAL_TRANSCRIPT_DIR or check TRANSCRIPT_DIR."
                )

            try:
                result = evaluate_feedback(
                    text_gradient,
                    transcript=transcript_text,
                    engagement_evidence="",
                )
                all_results[window_label] = {
                    "signal_type": "text_gradient",
                    "step": step_idx,
                    "text_gradient": text_gradient,
                    "input_transcript_path": declared_tp,
                    "transcript_path_used": transcript_path_used,
                    **result,
                }
                score = result.get("dean_result", {}).get("overall_score", "?")
                print(f"  [DONE] overall_score={score}")
            except Exception as e:
                print(f"  [FAIL] {e}")
                all_results[window_label] = {"error": str(e)}

    # Sort output by window time for easier alignment: 0m_6m -> ... -> 42m_48m
    ordered_results = {
        k: all_results[k] for k in sorted(all_results.keys(), key=_window_sort_key)
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2, ensure_ascii=False)

    # Token usage, efficiency, and cost
    if REQUEST_COUNT > 0:
        total_prompt_k = TOTAL_PROMPT_TOKENS / 1000.0
        total_completion_k = TOTAL_COMPLETION_TOKENS / 1000.0
        cost_input = total_prompt_k * PRICE_INPUT_PER_1K
        cost_output = total_completion_k * PRICE_OUTPUT_PER_1K
        total_cost = cost_input + cost_output
        avg_prompt = TOTAL_PROMPT_TOKENS / REQUEST_COUNT
        avg_completion = TOTAL_COMPLETION_TOKENS / REQUEST_COUNT
        total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
        avg_total_per_req = total_tokens / REQUEST_COUNT
        print(
            f"\n[TOKEN] total_prompt_tokens={TOTAL_PROMPT_TOKENS}, "
            f"total_completion_tokens={TOTAL_COMPLETION_TOKENS}, "
            f"total_tokens={total_tokens}, "
            f"requests={REQUEST_COUNT}"
        )
        print(
            f"[EFFICIENCY] avg_prompt_tokens_per_req={avg_prompt:.1f}, "
            f"avg_completion_tokens_per_req={avg_completion:.1f}, "
            f"avg_total_tokens_per_req={avg_total_per_req:.1f}"
        )
        print(
            f"[COST] input_tokens_k={total_prompt_k:.2f}, "
            f"output_tokens_k={total_completion_k:.2f}, "
            f"estimated_input_cost=${cost_input:.4f}, "
            f"estimated_output_cost=${cost_output:.4f}, "
            f"estimated_total_cost=${total_cost:.4f}"
        )

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")





