import os
import re
import sys
import json
import glob
from typing import Dict, List, Tuple

# Use the local TextGrad repository code (not the installed site-packages version).
TEXTGRAD_REPO_ROOT = r"D:\Desktop\llm_as_a_judge\data\llm\textgrad"
if TEXTGRAD_REPO_ROOT not in sys.path:
    sys.path.insert(0, TEXTGRAD_REPO_ROOT)

import textgrad as tg
from textgrad import Variable
from textgrad.model import BlackboxLLM
from textgrad.engine import get_engine


INPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\whisper_out"
OUTPUT_JSONL_PATH = r"D:\Desktop\llm_as_a_judge\data\llm\textgrad_my\Zoom Class Meeting Downing Soc 100 4.19.2021\optimized_transcripts_with_gradients.jsonl"

# By default, do one TextGrad step to limit cost. You can override with env var.
STEPS = int(os.getenv("TEXTGRAD_STEPS", "1"))

# To avoid runaway prompt size, you can cap transcript length (characters).
MAX_CHARS = int(os.getenv("TEXTGRAD_MAX_CHARS", "12000"))


TRANSCRIPT_RE = re.compile(r"transcript_(\d+)m_(\d+)m\.txt$")


def resolve_engine_and_model() -> Tuple[str, object]:
    """
    Resolve model routing to match focus_scoring_fine.py:
    - Prefer OpenRouter (OPENROUTER_API_KEY + base_url=https://openrouter.ai/api/v1)
    - Else fallback to OpenAI official (OPENAI_API_KEY)
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key and openrouter_key.strip():
        model_name = "openai/gpt-4o-mini"
        base_url = "https://openrouter.ai/api/v1"
        api_key = openrouter_key.strip()
    elif openai_key and openai_key.strip():
        model_name = "gpt-4o-mini"
        base_url = None
        api_key = openai_key.strip()
    else:
        raise ValueError(
            "Missing API key. Please set either OPENROUTER_API_KEY or OPENAI_API_KEY."
        )

    engine = get_engine(model_name, base_url=base_url, api_key=api_key)
    return model_name, engine


def list_transcript_files(folder: str) -> List[Tuple[str, str, int, int]]:
    """
    Returns a list of (window_label, transcript_path, start_min, end_min), sorted by start_min.
    """
    out: List[Tuple[str, str, int, int]] = []
    for path in glob.glob(os.path.join(folder, "transcript_*.txt")):
        fname = os.path.basename(path)
        m = TRANSCRIPT_RE.match(fname)
        if not m:
            continue
        start_min = int(m.group(1))
        end_min = int(m.group(2))
        window_label = f"{start_min}m_{end_min}m"
        out.append((window_label, path, start_min, end_min))
    out.sort(key=lambda x: (x[2], x[3]))
    return out


def build_evaluation_instruction() -> str:
    """
    TextGrad loss system instruction (must be English).
    The model output will be used as feedback to compute the textual gradient.
    """
    return (
        "You are an expert evaluator for classroom teaching transcripts. "
        "The transcript you receive is a spoken teaching script for one 6-minute window. "
        "Your feedback will be used by a text-optimization algorithm to rewrite the transcript. "
        "Therefore, you MUST NOT rewrite the transcript yourself. "
        "Instead, provide a short, critical ENGLISH critique describing concrete edits to improve: "
        "(1) clarity and structure, (2) pacing, (3) transitions/signposting, and "
        "(4) opportunities to engage learners (e.g., questions, participation prompts, checks for understanding). "
        "Be strict and specific: point out where the transcript is unclear, overly long, repetitive, "
        "missing transitions, or missing engagement cues. "
        "CRUCIAL: Do not invent any new facts, events, dates, names, or numeric details. "
        "When suggesting edits, instruct changes that preserve the original teaching meaning and factual content. "
        "Output ONLY English feedback (no JSON, no headings)."
    )


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR}")

    transcripts = list_transcript_files(INPUT_DIR)
    if not transcripts:
        raise RuntimeError(f"No transcript_{{start}}m_{{end}}m.txt files found in: {INPUT_DIR}")

    model_name, engine = resolve_engine_and_model()
    tg.set_backward_engine(engine, override=True)

    # BlackboxLLM is not strictly required here, but keeping it ensures
    # TextGrad can always make an LLM call if needed by internal components.
    _model = BlackboxLLM(engine)

    evaluation_instruction = build_evaluation_instruction()
    loss_fn = tg.TextLoss(evaluation_instruction)

    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out_f:
        for window_label, transcript_path, _, _ in transcripts:
            print(f"\n[WINDOW] {window_label}  source={transcript_path}")

            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read().strip()

            if not transcript_text:
                print(f"  [SKIP] empty transcript.")
                continue

            if MAX_CHARS > 0:
                transcript_for_opt = transcript_text[:MAX_CHARS]
            else:
                transcript_for_opt = transcript_text

            # The transcript itself is the variable we optimize.
            transcript_var = Variable(
                transcript_for_opt,
                requires_grad=True,
                role_description="classroom teaching transcript",
            )

            constraints = [
                "Preserve the original factual meaning: do not invent new events, names, dates, deadlines, course requirements, or numeric details.",
                "Do not change the core teaching intent; rewrite only for clarity, structure, pacing, and engagement cues.",
                "Maintain the original topic sequence and keep a coherent timeline within the 6-minute window.",
                "Keep the instructor's tone and approximate speaking style; avoid adding unrelated content.",
            ]

            optimizer = tg.TGD(parameters=[transcript_var], constraints=constraints)

            step_records: List[Dict] = []
            for step in range(STEPS):
                # Forward: compute evaluation loss/feedback text
                loss = loss_fn(transcript_var)
                loss_text = str(loss.value)

                # Backward: compute textual gradients on transcript_var
                transcript_var.reset_gradients()
                loss.backward()
                text_gradient = transcript_var.get_gradient_text()

                # Update transcript using gradients
                optimizer.step()

                step_records.append(
                    {
                        "step": step,
                        "loss_text": loss_text,
                        "text_gradient": text_gradient,
                    }
                )

                print(f"  [STEP {step}] loss_text_len={len(loss_text)} grad_len={len(text_gradient)}")

            optimized_transcript = transcript_var.value

            out_item = {
                "window": window_label,
                "input_transcript_path": transcript_path,
                "model_name": model_name,
                "steps": STEPS,
                "step_records": step_records,
                "optimized_transcript": optimized_transcript,
            }

            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print(f"\n[SAVED] {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    main()

