import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


INPUT_JSON = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/writing_behavior3.json")
OUTPUT_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/vis_result/writing_behavior_charts")


def load_hits(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hits = data.get("hits", [])
    if not isinstance(hits, list):
        return []
    return [h for h in hits if isinstance(h, dict)]


def save_counter_csv(counter: Counter, out_csv: Path, k1: str, k2: str) -> None:
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"{k1},{k2}\n")
        for key, value in sorted(counter.items(), key=lambda x: x[0]):
            f.write(f"{key},{value}\n")


def plot_hits_by_minute(minute_counter: Counter, out_png: Path) -> None:
    minutes = sorted(minute_counter.keys())
    counts = [minute_counter[m] for m in minutes]

    plt.figure(figsize=(14, 5))
    plt.plot(minutes, counts, marker="o", linewidth=2)
    plt.title("Writing Behavior Hits by Minute")
    plt.xlabel("Minute")
    plt.ylabel("Hit Count")
    if minutes:
        plt.xlim(min(minutes), max(minutes))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_top_grids(grid_counter: Counter, out_png: Path, top_n: int = 15) -> None:
    top_items = grid_counter.most_common(top_n)
    labels = [x[0] for x in top_items]
    values = [x[1] for x in top_items]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(f"Top {top_n} Grids by Writing Behavior Hits")
    plt.xlabel("Grid")
    plt.ylabel("Hit Count")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_minute_grid_heatmap(hits: List[Dict], out_png: Path) -> None:
    minute_set = sorted({int(h["minute"]) for h in hits if "minute" in h})
    grid_set = sorted({str(h["grid"]) for h in hits if "grid" in h})
    if not minute_set or not grid_set:
        return

    minute_to_idx = {m: i for i, m in enumerate(minute_set)}
    grid_to_idx = {g: j for j, g in enumerate(grid_set)}

    matrix = [[0 for _ in grid_set] for _ in minute_set]
    for h in hits:
        try:
            m = int(h["minute"])
            g = str(h["grid"])
            matrix[minute_to_idx[m]][grid_to_idx[g]] += 1
        except Exception:
            continue

    plt.figure(figsize=(16, 8))
    plt.imshow(matrix, aspect="auto")
    plt.title("Writing Behavior Heatmap (Minute x Grid)")
    plt.xlabel("Grid")
    plt.ylabel("Minute")
    plt.xticks(range(len(grid_set)), grid_set, rotation=90)
    plt.yticks(range(len(minute_set)), minute_set)
    cbar = plt.colorbar()
    cbar.set_label("Hit Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_detail_csv(hits: List[Dict], out_csv: Path) -> None:
    header = ["minute", "grid", "file", "reasoning"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for h in sorted(hits, key=lambda x: (int(x.get("minute", 0)), str(x.get("grid", "")))):
            minute = int(h.get("minute", 0))
            grid = str(h.get("grid", ""))
            file_name = str(h.get("file", ""))
            reasoning = str(h.get("reasoning", "")).replace('"', '""')
            f.write(f'{minute},{grid},{file_name},"{reasoning}"\n')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hits = load_hits(INPUT_JSON)
    if not hits:
        raise ValueError(f"No valid hits found in: {INPUT_JSON}")

    minute_counter = Counter()
    grid_counter = Counter()
    for h in hits:
        try:
            minute_counter[int(h.get("minute"))] += 1
            grid_counter[str(h.get("grid"))] += 1
        except Exception:
            continue

    # CSV outputs
    save_counter_csv(minute_counter, OUTPUT_DIR / "hits_by_minute.csv", "minute", "hit_count")
    save_counter_csv(grid_counter, OUTPUT_DIR / "hits_by_grid.csv", "grid", "hit_count")
    save_detail_csv(hits, OUTPUT_DIR / "hits_detail.csv")

    # Chart outputs
    plot_hits_by_minute(minute_counter, OUTPUT_DIR / "hits_by_minute_trend.png")
    plot_top_grids(grid_counter, OUTPUT_DIR / "top_grids_bar.png", top_n=15)
    plot_minute_grid_heatmap(hits, OUTPUT_DIR / "minute_grid_heatmap.png")

    print(f"Total hits: {len(hits)}")
    print(f"Unique minutes: {len(minute_counter)}")
    print(f"Unique grids: {len(grid_counter)}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
