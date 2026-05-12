import json
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


INPUT_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/focus_output_now")
OUTPUT_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/vis_result/focuas_output_black")
SUMMARY_CSV = OUTPUT_DIR / "focus_output3_summary.csv"


def parse_time_window(file_name: str) -> Tuple[int, int]:
    # engagement_9m_12m.json -> (9, 12)
    body = file_name.replace("engagement_", "").replace(".json", "")
    start_str, end_str = body.split("_")
    return int(start_str.replace("m", "")), int(end_str.replace("m", ""))


def list_engagement_files(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.glob("engagement_*.json") if p.is_file()]
    return sorted(files, key=lambda p: parse_time_window(p.name))


def collect_scores(one_file: Path) -> List[int]:
    with open(one_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []

    scores: List[int] = []
    for grid_info in data.values():
        if not isinstance(grid_info, dict):
            continue
        engagement = grid_info.get("engagement")
        if isinstance(engagement, int) and 0 <= engagement <= 3:
            scores.append(engagement)
    return scores


def summarize(input_files: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for p in input_files:
        start_m, end_m = parse_time_window(p.name)
        scores = collect_scores(p)
        if not scores:
            continue

        n = len(scores)
        counts = {k: 0 for k in range(4)}
        for s in scores:
            counts[s] += 1

        row = {
            "file_name": p.name,
            "window_label": f"{start_m}m-{end_m}m",
            "start_minute": start_m,
            "end_minute": end_m,
            "n_grids": n,
            "mean_score": mean(scores),
            "median_score": median(scores),
            "std_score": pstdev(scores) if n > 1 else 0.0,
            "count_0": counts[0],
            "count_1": counts[1],
            "count_2": counts[2],
            "count_3": counts[3],
            "ratio_0": counts[0] / n,
            "ratio_1": counts[1] / n,
            "ratio_2": counts[2] / n,
            "ratio_3": counts[3] / n,
            "scores": scores,
        }
        rows.append(row)
    return rows


def save_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    header = [
        "file_name",
        "window_label",
        "start_minute",
        "end_minute",
        "n_grids",
        "mean_score",
        "median_score",
        "std_score",
        "count_0",
        "count_1",
        "count_2",
        "count_3",
        "ratio_0",
        "ratio_1",
        "ratio_2",
        "ratio_3",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [r[h] for h in header]
            f.write(",".join(str(v) for v in vals) + "\n")


def plot_mean_trend(rows: List[Dict], out_path: Path) -> None:
    x = [r["window_label"] for r in rows]
    y = [r["mean_score"] for r in rows]

    plt.figure(figsize=(14, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.title("Engagement Mean Trend by Time Window")
    plt.xlabel("Time Window")
    plt.ylabel("Mean Engagement")
    plt.ylim(0, 3)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_stacked_ratio(rows: List[Dict], out_path: Path) -> None:
    x = [r["window_label"] for r in rows]
    ratio_0 = [r["ratio_0"] for r in rows]
    ratio_1 = [r["ratio_1"] for r in rows]
    ratio_2 = [r["ratio_2"] for r in rows]
    ratio_3 = [r["ratio_3"] for r in rows]

    plt.figure(figsize=(14, 6))
    plt.bar(x, ratio_0, label="Score 0")
    plt.bar(x, ratio_1, bottom=ratio_0, label="Score 1")
    bottom_2 = [a + b for a, b in zip(ratio_0, ratio_1)]
    plt.bar(x, ratio_2, bottom=bottom_2, label="Score 2")
    bottom_3 = [a + b + c for a, b, c in zip(ratio_0, ratio_1, ratio_2)]
    plt.bar(x, ratio_3, bottom=bottom_3, label="Score 3")

    plt.title("Engagement Score Ratio by Time Window (Stacked)")
    plt.xlabel("Time Window")
    plt.ylabel("Ratio")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_overall_hist(rows: List[Dict], out_path: Path) -> None:
    all_scores: List[int] = []
    for r in rows:
        all_scores.extend(r["scores"])

    plt.figure(figsize=(8, 5))
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
    plt.hist(all_scores, bins=bins, rwidth=0.8)
    plt.title("Overall Engagement Distribution")
    plt.xlabel("Engagement Score")
    plt.ylabel("Count")
    plt.xticks([0, 1, 2, 3])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list_engagement_files(INPUT_DIR)
    if not files:
        raise FileNotFoundError(f"No engagement_*.json found in: {INPUT_DIR}")

    rows = summarize(files)
    if not rows:
        raise ValueError("No valid engagement data found in input files.")

    save_summary_csv(rows, SUMMARY_CSV)
    plot_mean_trend(rows, OUTPUT_DIR / "mean_trend.png")
    plot_stacked_ratio(rows, OUTPUT_DIR / "stacked_ratio.png")
    plot_overall_hist(rows, OUTPUT_DIR / "overall_histogram.png")

    print(f"Input files: {len(files)}")
    print(f"Valid windows: {len(rows)}")
    print(f"Saved CSV: {SUMMARY_CSV}")
    print(f"Saved chart dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
