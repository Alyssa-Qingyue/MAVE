"""
Teaching transcript optimization + Reflexion components from the original hotpotqa_runs.

Directly reuses assets from reflexion/hotpotqa_runs:
  - prompts.reflect_prompt (reflection prompt template from the paper)
  - fewshots.REFLECTIONS (few-shot reflection examples)
  - prompts.REFLECTION_HEADER / LAST_TRIAL_HEADER (injection format aligned with ReactReflectAgent)

This script does not directly call ReactReflectAgent / ReAct because those classes
strongly depend on Wikipedia, QA questions, and action parsing. Transcript optimization
is treated as a standalone task while preserving the Reflexion protocol:
"generate reflection + inject into next iteration".
"""
import os
import re
import sys
import json
import glob
import time
import enum
from typing import Dict, List, Tuple

from openai import OpenAI

# ---------- LLM CONFIG ----------
# Consistent with project_class/focus_scoring_fine.py: prefer OpenRouter, fallback to official OpenAI
TIMEOUT = 60

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    # Use OpenRouter
    MODEL = "openai/gpt-4o-mini"
    client = OpenAI(
        api_key=_openrouter_key.strip(),
        base_url="https://openrouter.ai/api/v1",
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


def _api_provider_label() -> str:
    if _openrouter_key and _openrouter_key.strip():
        return "openrouter"
    return "openai"


# --- Aligned with original reflexion/hotpotqa_runs ---
_HOTPOT_ROOT = r"D:\Desktop\llm_as_a_judge\data\llm\reflexion\hotpotqa_runs"
if _HOTPOT_ROOT not in sys.path:
    sys.path.insert(0, _HOTPOT_ROOT)

# fewshots has no third-party dependency; prompts depends on langchain.
# If unavailable, fallback to equivalent plain-string templates (consistent with prompts.py).
try:
    from fewshots import REFLECTIONS  # type: ignore  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"Failed to import hotpotqa_runs.fewshots (does path exist: {_HOTPOT_ROOT}?)"
    ) from e

_REFLECT_PROMPT_SOURCE = "embedded_fallback_matching_prompts_py"

try:
    from prompts import reflect_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER  # type: ignore  # noqa: E402

    _REFLECT_PROMPT_SOURCE = "reflexion.hotpotqa_runs.prompts"

    def build_reflect_user_prompt(
        examples: str, question: str, scratchpad: str
    ) -> str:
        return reflect_prompt.format(
            examples=examples, question=question, scratchpad=scratchpad
        )

except ImportError:
    # Keep wording consistent with REFLECT_INSTRUCTION / HEADER from prompts.py (avoid langchain dependency)
    REFLECTION_HEADER = (
        "You have attempted to answer following question before and failed. "
        "The following reflection(s) give a plan to avoid failing to answer the question in "
        "the same way you did previously. Use them to improve your strategy of correctly "
        "answering the given question.\n"
    )
    LAST_TRIAL_HEADER = (
        "You have attempted to answer the following question before and failed. "
        "Below is the last trial you attempted to answer the question.\n"
    )
    _REFLECT_INSTRUCTION_FALLBACK = (
        "You are an advanced reasoning agent that can improve based on self refection. You will be given "
        "a previous reasoning trial in which you were given access to an Docstore API environment and "
        "a question to answer. You were unsuccessful in answering the question either because you guessed "
        "the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a "
        "few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan "
        "that aims to mitigate the same failure. Use complete sentences.  \n"
        "Here are some examples:\n"
        "{examples}\n"
        "\n"
        "Previous trial:\n"
        "Question: {question}{scratchpad}\n"
        "\n"
        "Reflection:"
    )

    def build_reflect_user_prompt(
        examples: str, question: str, scratchpad: str
    ) -> str:
        return _REFLECT_INSTRUCTION_FALLBACK.format(
            examples=examples, question=question, scratchpad=scratchpad
        )


TRANSCRIPT_INPUT_DIR = os.getenv(
    "REFLEXION_TRANSCRIPT_INPUT_DIR",
    r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\whisper_out",
)
OUTPUT_DIR = os.getenv(
    "REFLEXION_OUTPUT_DIR",
    r"D:\Desktop\llm_as_a_judge\data\llm\reflexion_my\Zoom Class Meeting Downing Soc 100 4.19.2021",
)
OUTPUT_JSONL_PATH = os.getenv(
    "REFLEXION_REPO_OUTPUT_JSONL",
    os.path.join(OUTPUT_DIR, "reflexion_transcript.jsonl"),
)
OUTPUT_REFLECTION_JSONL_PATH = os.getenv(
    "REFLEXION_REPO_SIGNAL_JSONL",
    os.path.join(OUTPUT_DIR, "reflection_signals.jsonl"),
)

MAX_ITERS = int(os.getenv("REFLEXION_MAX_ITERS", "3"))
MAX_CHARS = int(os.getenv("REFLEXION_MAX_CHARS", "12000"))
REFLECT_SCRATCHPAD_MAX = int(os.getenv("REFLEXION_REFLECT_SCRATCHPAD_MAX", "8000"))
TEMPERATURE = float(os.getenv("REFLEXION_TEMPERATURE", "0.2"))
RETRY_TIMES = int(os.getenv("REFLEXION_RETRY_TIMES", "3"))

# reflexion: accumulate verbal reflections
# (aligned with ReactReflectAgent + ReflexionStrategy.REFLEXION)
# last_trial: inject only the last trial trace
# (aligned with ReflexionStrategy.LAST_ATTEMPT, without separate verbal reflection)
_REF_STRAT = (os.getenv("REFLEXION_STRATEGY") or "reflexion").strip().lower()


class TranscriptReflexionStrategy(str, enum.Enum):
    REFLEXION = "reflexion"
    LAST_TRIAL = "last_trial"


TRANSCRIPT_TASK_QUESTION = (
    "Improve this classroom teaching transcript for clarity, pacing, and student engagement "
    "while preserving factual content. The transcript may be Chinese or English; "
    "if rewriting, prefer clear, classroom-ready English unless the source is already another target language."
)

TRANSCRIPT_PATTERN = re.compile(r"transcript_(\d+)m_(\d+)m\.txt$", re.I)

# If few-shot examples are too long, disable with: set REFLEXION_USE_HOTPOT_FEWSHOT=0
USE_HOTPOT_FEWSHOT = (os.getenv("REFLEXION_USE_HOTPOT_FEWSHOT") or "1").strip() not in (
    "0",
    "false",
    "False",
)


def list_transcripts(folder: str) -> List[Tuple[int, int, str]]:
    paths = sorted(glob.glob(os.path.join(folder, "transcript_*m_*m.txt")))
    windows: List[Tuple[int, int, str]] = []
    for p in paths:
        m = TRANSCRIPT_PATTERN.match(os.path.basename(p))
        if not m:
            continue
        windows.append((int(m.group(1)), int(m.group(2)), p))
    windows.sort(key=lambda x: x[0])
    return windows


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chat_once(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    last_error = None
    for attempt in range(RETRY_TIMES):
        try:
            response = client.chat.completions.create(
                model=model,
                timeout=TIMEOUT,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt < RETRY_TIMES - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {RETRY_TIMES} retries: {last_error}")


# --- String helpers aligned with agents.py (avoid importing agents and pulling Wikipedia / langchain.docstore) ---
def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def format_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ""
    return (
        REFLECTION_HEADER
        + "Reflections:\n- "
        + "\n- ".join([r.strip() for r in reflections])
    )


def format_last_attempt(question: str, scratchpad: str) -> str:
    return (
        LAST_TRIAL_HEADER
        + f"Question: {question}\n"
        + scratchpad.strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


def truncate_block(text: str, limit: int) -> str:
    t = text.strip()
    if len(t) <= limit:
        return t
    head = limit // 2
    tail = limit - head
    return t[:head] + "\n...[truncated for max token safety]...\n" + t[-tail:]


def build_hotpot_style_scratchpad(feedback: str, candidate: str) -> str:
    """Pack evaluator feedback + current draft into a trial record consumable by reflect_prompt."""
    fb = truncate_block(feedback, REFLECT_SCRATCHPAD_MAX // 3)
    cand = truncate_block(candidate, 2 * REFLECT_SCRATCHPAD_MAX // 3)
    return (
        "\nThought 1: I should review evaluator feedback on my current transcript draft.\n"
        "Action 1: Submit[draft]\n"
        f"Observation 1:\n{fb}\n\n"
        "Current draft transcript:\n"
        f"{cand}\n"
    )


def reflection_fewshot_block() -> str:
    if USE_HOTPOT_FEWSHOT:
        return REFLECTIONS
    return "(Task adapted from HotPotQA-style Reflexion; few-shot examples omitted to save tokens.)\n"


def run_hotpot_reflect_llm(
    client: OpenAI,
    model: str,
    feedback: str,
    candidate: str,
) -> str:
    """Use reflect_prompt + REFLECTIONS from the repo to generate verbal reflection."""
    scratchpad = truncate_block(
        build_hotpot_style_scratchpad(feedback, candidate),
        REFLECT_SCRATCHPAD_MAX,
    )
    user_prompt = build_reflect_user_prompt(
        examples=reflection_fewshot_block(),
        question=TRANSCRIPT_TASK_QUESTION,
        scratchpad=scratchpad,
    )
    system_prompt = (
        "Follow the user instructions exactly. Output only the text that belongs after 'Reflection:' "
        "(a short diagnosis plus a concise new plan), without repeating the examples block."
    )
    return format_step(chat_once(client, model, system_prompt, user_prompt))


def evaluator_feedback(
    client: OpenAI,
    model: str,
    original: str,
    candidate: str,
) -> str:
    user = f"""
Evaluate the candidate teaching transcript against the original for teaching quality.

Return in English with sections: Strengths / Weaknesses / Top 3 priorities.
Do not rewrite the transcript.

[Original]
{truncate_block(original, MAX_CHARS)}

[Candidate]
{truncate_block(candidate, MAX_CHARS)}
""".strip()
    return chat_once(
        client,
        model,
        "You are a strict instructional coach.",
        user,
    ).strip()


def revise_with_reflexion_memory(
    client: OpenAI,
    model: str,
    original: str,
    candidate: str,
    reflections_section: str,
) -> str:
    user = f"""
{reflections_section}

Task question:
{TRANSCRIPT_TASK_QUESTION}

[Original transcript — factual grounding]
{truncate_block(original, MAX_CHARS)}

[Current draft to revise]
{truncate_block(candidate, MAX_CHARS)}

Output ONLY the full improved transcript text.
""".strip()
    return chat_once(
        client,
        model,
        "You are a precise teaching-transcript editor.",
        user,
    ).strip()


def run_single_transcript(
    client: OpenAI,
    model: str,
    transcript_text: str,
    strategy: TranscriptReflexionStrategy,
) -> Dict:
    original = transcript_text[:MAX_CHARS].strip()
    candidate = original
    step_records: List[Dict] = []
    verbal_reflections: List[str] = []

    for step in range(MAX_ITERS):
        fb = evaluator_feedback(client, model, original, candidate)
        injected: str
        verbal: str

        if strategy == TranscriptReflexionStrategy.LAST_TRIAL:
            trial_pad = truncate_block(
                build_hotpot_style_scratchpad(fb, candidate),
                REFLECT_SCRATCHPAD_MAX,
            )
            injected = format_last_attempt(TRANSCRIPT_TASK_QUESTION, trial_pad)
            verbal = trial_pad
        else:
            verbal = run_hotpot_reflect_llm(client, model, fb, candidate)
            verbal_reflections.append(verbal)
            injected = format_reflections(verbal_reflections)

        improved = revise_with_reflexion_memory(
            client, model, original, candidate, injected
        ).strip()
        if not improved:
            improved = candidate

        step_records.append(
            {
                "step": step,
                "strategy": strategy.value,
                "evaluator_feedback": fb,
                "reflection_signal": verbal,
                "reflexion_injected_block": injected,
                "candidate_before": candidate,
                "candidate_after": improved,
                "accumulated_verbal_reflections": list(verbal_reflections),
            }
        )
        candidate = improved

    return {
        "reflexion_strategy": strategy.value,
        "max_iters": MAX_ITERS,
        "used_hotpot_fewshot": USE_HOTPOT_FEWSHOT,
        "original_transcript": original,
        "optimized_transcript": candidate,
        "step_records": step_records,
    }


def main() -> None:
    if not os.path.isdir(TRANSCRIPT_INPUT_DIR):
        raise FileNotFoundError(f"Input directory does not exist: {TRANSCRIPT_INPUT_DIR}")
    if not os.path.isdir(_HOTPOT_ROOT):
        raise FileNotFoundError(
            f"hotpotqa_runs directory not found, please check: {_HOTPOT_ROOT}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        strategy = TranscriptReflexionStrategy(_REF_STRAT)
    except ValueError:
        raise ValueError(
            f"Invalid REFLEXION_STRATEGY={_REF_STRAT!r}; use reflexion or last_trial"
        )

    windows = list_transcripts(TRANSCRIPT_INPUT_DIR)
    if not windows:
        raise ValueError(f"No transcript_*m_*m.txt found in: {TRANSCRIPT_INPUT_DIR}")

    provider = _api_provider_label()

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out_all, open(
        OUTPUT_REFLECTION_JSONL_PATH, "w", encoding="utf-8"
    ) as out_sig:
        for start_m, end_m, path in windows:
            text = read_text(path)
            label = f"{start_m}m_{end_m}m"
            if not text.strip():
                print(f"[SKIP] {label} empty")
                continue

            result = run_single_transcript(client, MODEL, text, strategy)
            item = {
                "window": label,
                "transcript_path": path,
                "provider": provider,
                "model": MODEL,
                "reflect_prompt_resolved_from": _REFLECT_PROMPT_SOURCE,
                "hotpotqa_fewshots": "reflexion.hotpotqa_runs.fewshots.REFLECTIONS"
                if USE_HOTPOT_FEWSHOT
                else "(omitted)",
                **result,
            }
            out_all.write(json.dumps(item, ensure_ascii=False) + "\n")

            for rec in result["step_records"]:
                out_sig.write(
                    json.dumps(
                        {
                            "window": label,
                            "transcript_path": path,
                            "step": rec["step"],
                            "strategy": rec["strategy"],
                            "reflection_signal": rec["reflection_signal"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"[DONE] {label} steps={len(result['step_records'])}")

    print(
        f"\nDone.\nFull output: {OUTPUT_JSONL_PATH}\nReflection signals: {OUTPUT_REFLECTION_JSONL_PATH}"
    )


if __name__ == "__main__":
    main()
