import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


INPUT_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/preprocessor_output3")
OUTPUT_JSON = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/writing_behavior3.json")
MODEL = "gpt-4o-mini"
TIMEOUT = 60


def build_client() -> OpenAI:
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    if openai_key:
        return OpenAI(api_key=openai_key, timeout=TIMEOUT)

    raise ValueError(
        "API key is missing. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."
    )


def parse_minute(file_name: str) -> int:
    return int(file_name.replace("minute_", "").replace(".json", ""))


def normalize_json_text(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    return content


def build_prompt(description: str) -> str:
    return f"""
You are a strict behavior classifier.
Determine whether this classroom visual description contains POSITIVE evidence of writing behavior.

Writing behavior includes:
- actively writing notes/answers
- holding pen/pencil/stylus and writing
- note-taking/drawing for class task

If description says "no writing", "no writing detected", "not clearly visible", or uncertain mention only,
then has_writing_behavior must be false.

Return STRICT JSON only:
{{
  "has_writing_behavior": true/false,
  "reasoning": "one concise sentence in English"
}}

Description:
{description}
"""


def classify_writing_behavior(client: OpenAI, description: str) -> Tuple[bool, str]:
    if not isinstance(description, str) or not description.strip():
        return False, "Empty description."

    prompt = build_prompt(description.strip())
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(normalize_json_text(content))
    return bool(payload.get("has_writing_behavior", False)), str(payload.get("reasoning", "")).strip()


def scan_all_minutes(input_dir: Path, client: OpenAI) -> List[Dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    minute_files = sorted(input_dir.glob("minute_*.json"), key=lambda p: parse_minute(p.name))
    hits: List[Dict] = []

    for minute_file in minute_files:
        minute = parse_minute(minute_file.name)
        with open(minute_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for grid_id, desc in data.items():
            try:
                ok, reason = classify_writing_behavior(client, desc if isinstance(desc, str) else "")
            except Exception as e:
                ok, reason = False, f"LLM error: {e}"

            if ok:
                hits.append(
                    {
                        "minute": minute,
                        "grid": grid_id,
                        "reasoning": reason,
                        "file": minute_file.name,
                    }
                )
                print(f"[HIT] minute={minute}, grid={grid_id}, reasoning={reason}")

    return hits


def main() -> None:
    client = build_client()
    hits = scan_all_minutes(INPUT_DIR, client)
    summary = {
        "model": MODEL,
        "input_dir": str(INPUT_DIR),
        "total_hits": len(hits),
        "hits": hits,
    }

    print(f"Total writing behavior hits: {len(hits)}")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
