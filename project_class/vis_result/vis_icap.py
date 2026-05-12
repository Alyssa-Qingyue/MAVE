import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


WRITING_JSON = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/writing_behavior3.json")
ICAP_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/i_c_output")
OUT_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/vis_result/icap_align_charts")


def window_label(start_min: int, end_min: int) -> str:
    return f"{start_min}m-{end_min}m"


def parse_transcript_window(file_name: str) -> Tuple[int, int]:
    # transcript_12m_18m_classification.json -> (12, 18)
    m = re.match(r"^transcript_(\d+)m_(\d+)m_classification\.json$", file_name)
    if not m:
        raise ValueError(f"Invalid transcript window file name: {file_name}")
    return int(m.group(1)), int(m.group(2))


def load_a_counts_per_6m(writing_json: Path) -> Counter:
    with open(writing_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    hits = data.get("hits", [])
    counter = Counter()

    if not isinstance(hits, list):
        return counter

    for item in hits:
        if not isinstance(item, dict):
            continue
        minute = item.get("minute")
        if not isinstance(minute, int):
            continue
        start = (minute // 6) * 6
        end = start + 6
        counter[(start, end)] += 1
    return counter


def load_c_i_student_counts(icap_dir: Path) -> Tuple[Counter, Counter, Counter]:
    # c_counter/window -> student_speaker_count (when final_ci_level == c)
    # i_counter/window -> student_speaker_count (when final_ci_level == i)
    # student_total/window -> student_speaker_count (regardless of c/i)
    c_counter = Counter()
    i_counter = Counter()
    student_total = Counter()

    files = sorted(icap_dir.glob("transcript_*m_*m_classification.json"))
    for p in files:
        try:
            start, end = parse_transcript_window(p.name)
        except ValueError:
            continue

        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)

        ci_level = payload.get("final_ci_level")
        student_speaker_count = payload.get("student_speaker_count", 0)
        if not isinstance(student_speaker_count, int):
            student_speaker_count = 0

        student_total[(start, end)] += student_speaker_count
        if ci_level == "c":
            c_counter[(start, end)] += student_speaker_count
        elif ci_level == "i":
            i_counter[(start, end)] += student_speaker_count

    return c_counter, i_counter, student_total


def save_align_csv(
    windows: List[Tuple[int, int]],
    a_counter: Counter,
    c_counter: Counter,
    i_counter: Counter,
    student_total: Counter,
    out_csv: Path,
) -> None:
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("window,start_min,end_min,a_count,c_student_count,i_student_count,student_speaker_count_total\n")
        for s, e in windows:
            f.write(
                f"{window_label(s,e)},{s},{e},{a_counter[(s,e)]},{c_counter[(s,e)]},{i_counter[(s,e)]},{student_total[(s,e)]}\n"
            )


def plot_aligned_series(
    windows: List[Tuple[int, int]],
    a_counter: Counter,
    c_counter: Counter,
    i_counter: Counter,
    out_png: Path,
) -> None:
    labels = [window_label(s, e) for s, e in windows]
    x = list(range(len(labels)))
    a_vals = [a_counter[(s, e)] for s, e in windows]
    c_vals = [c_counter[(s, e)] for s, e in windows]
    i_vals = [i_counter[(s, e)] for s, e in windows]

    plt.figure(figsize=(14, 6))

    # A 用柱状，C/I 用折线，便于一张图对比
    bars = plt.bar(x, a_vals, alpha=0.35, label="A count (writing hits, 6m aligned)")
    plt.plot(x, c_vals, marker="o", linewidth=2, label="C count (student_speaker_count)")
    plt.plot(x, i_vals, marker="o", linewidth=2, label="I count (student_speaker_count)")

    # 在图上标注每个类别的人数
    for idx, b in enumerate(bars):
        height = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            height + 0.15,
            f"A:{int(a_vals[idx])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for xi, yi in zip(x, c_vals):
        plt.text(
            xi,
            yi + 0.18,
            f"C:{int(yi)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for xi, yi in zip(x, i_vals):
        plt.text(
            xi,
            yi - 0.18,
            f"I:{int(yi)}",
            ha="center",
            va="top",
            fontsize=8,
        )

    plt.title("Time-aligned A / C / I Counts (6-minute windows)")
    plt.xlabel("Time Window")
    plt.ylabel("Count")
    max_y = max(a_vals + c_vals + i_vals) if (a_vals or c_vals or i_vals) else 1
    plt.ylim(0, max_y + 1.2)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a_counter = load_a_counts_per_6m(WRITING_JSON)
    c_counter, i_counter, student_total = load_c_i_student_counts(ICAP_DIR)

    # 用 transcript 时间窗作为主轴，保证和 i_c_output 时间严格对齐
    windows = sorted(set(c_counter.keys()) | set(i_counter.keys()) | set(student_total.keys()))
    if not windows:
        raise ValueError("No aligned transcript windows found in i_c_output.")

    out_csv = OUT_DIR / "a_c_i_aligned_summary.csv"
    out_png = OUT_DIR / "a_c_i_aligned_plot.png"

    save_align_csv(windows, a_counter, c_counter, i_counter, student_total, out_csv)
    plot_aligned_series(windows, a_counter, c_counter, i_counter, out_png)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved Plot: {out_png}")
    print(f"Windows: {len(windows)}")


if __name__ == "__main__":
    main()
