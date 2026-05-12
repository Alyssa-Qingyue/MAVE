import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any


TARGET_REASONING = "Strong rule applied: writing behavior detected in evidence, directly assign engagement=3."
INPUT_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/focus_output_now")
OUTPUT_JSON = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/writing_rule_summary.json")


def parse_time_window(file_name: str) -> str:
    """
    engagement_0m_3m.json -> 0m_3m
    """
    return file_name.replace("engagement_", "").replace(".json", "")


def collect_hits() -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    json_files = sorted(INPUT_DIR.glob("engagement_*.json"))

    if not json_files:
        raise FileNotFoundError(f"No engagement_*.json found in: {INPUT_DIR}")

    for file_path in json_files:
        time_window = parse_time_window(file_path.name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for grid_id, info in data.items():
            if not isinstance(info, dict):
                continue
            reasoning = info.get("reasoning", "")
            if reasoning == TARGET_REASONING:
                hits.append(
                    {
                        "time_window": time_window,
                        "file_name": file_path.name,
                        "grid_id": grid_id,
                        "engagement": info.get("engagement"),
                    }
                )
    return hits


def build_summary(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_time = Counter(item["time_window"] for item in hits)
    by_grid = Counter(item["grid_id"] for item in hits)

    summary = {
        "target_reasoning": TARGET_REASONING,
        "input_dir": str(INPUT_DIR),
        "total_count": len(hits),
        "by_time_window": dict(sorted(by_time.items(), key=lambda x: x[0])),
        "by_grid_id": dict(sorted(by_grid.items(), key=lambda x: x[0])),
        "details": sorted(hits, key=lambda x: (x["time_window"], x["grid_id"])),
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Writing Rule Count Summary ===")
    print(f"Target reasoning: {summary['target_reasoning']}")
    print(f"Total matched count: {summary['total_count']}")

    print("\n[By Time Window]")
    if summary["by_time_window"]:
        for time_window, count in summary["by_time_window"].items():
            print(f"- {time_window}: {count}")
    else:
        print("- No match")

    print("\n[By Grid ID]")
    if summary["by_grid_id"]:
        for grid_id, count in summary["by_grid_id"].items():
            print(f"- {grid_id}: {count}")
    else:
        print("- No match")

    print("\n[Detail List: time_window | grid_id | file_name]")
    if summary["details"]:
        for item in summary["details"]:
            print(f"- {item['time_window']} | {item['grid_id']} | {item['file_name']}")
    else:
        print("- No match")


def main() -> None:
    hits = collect_hits()
    summary = build_summary(hits)
    print_summary(summary)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
