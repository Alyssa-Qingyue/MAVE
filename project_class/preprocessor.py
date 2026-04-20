'''
[USAGE] requests=1500, total_prompt_tokens=76879500, 
total_completion_tokens=322774, avg_prompt_tokens_per_req=51253.0, 
avg_completion_tokens_per_req=215.2
≈ $11.72 USD

[USAGE] requests=1353, total_prompt_tokens=66032212, total_completion_tokens=296806, 
avg_prompt_tokens_per_req=48804.3, avg_completion_tokens_per_req=219.4
Total estimated cost ≈ 9.90 + 0.18 ≈ $10.08 USD
'''
import os
import base64
import json
from openai import OpenAI


# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
MODEL = "gpt-4o-mini"
TIMEOUT = 60

IMAGE_ROOT = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\images"
OUTPUT_DIR = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\preprocesssor_ouput"
PPT_EXCLUDED_TIMES_JSON = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\ppt_excluded_times.json"
ICON_EXCLUDED_FRAMES_JSON = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\icon_excluded_frames.json"
BLACK_EXCLUDED_FRAMES_JSON = r"D:\Desktop\llm_as_a_judge\data\llm\project\Zoom Class Meeting Downing Soc 100 4⧸19⧸2021\black_excluded_frames.json"
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0


# PROMPT

PREPROCESSOR_PROMPT = """
You are a visual scene narrator.

You will be shown one or more image frames presented together.
Your task is to describe, in natural language, what is visibly shown in the scene.

Write a detailed third-person description that covers the scene from multiple perspectives, for example:
- an overall view of the scene,
- the visible posture and positioning of the person in the frame,
- the arrangement and appearance of visible objects,
- any visible changes or consistency across the frames.

Guidelines:
- Describe only what can be directly seen.
- Use natural, continuous prose rather than lists.
- It is acceptable to restate or elaborate on details if it helps clarity.
- Do NOT infer mental states, intentions, emotions, or reasons.
- Do NOT evaluate or judge the person.
- If little movement is visible, describe the stillness in detail.

Style:
- Neutral, descriptive narration.
- Third-person.
- Aim for 1–3 short paragraphs.

"""


# UTILS

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# CORE FUNCTION (process multiple frames for one grid within one minute at once)

def run_minute_frames_to_text(minute_id: str, grid_id: str, image_paths):
    """
    Input: 1–6 frame paths from the same 1-minute window and grid
    (after cleaning there may be fewer than 6 frames; logic is compatible).
    Output: one detailed natural language description for the WHOLE minute (string).
    """
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    if not image_paths:
        return None, "no images"

    # There may be fewer than 6 frames after cleaning; use available frames (up to 6).
    image_paths = image_paths[:6]
    n_frames = len(image_paths)

    intro_text = (
        PREPROCESSOR_PROMPT
        + "\n\nYou are given "
        + (f"{n_frames} frame(s) " if n_frames < 6 else "up to six frames ")
        + "from the same 1-minute window "
        + f"(folder: {minute_id}, grid: {grid_id}). "
        + "Write ONE coherent description for the entire minute, taking into account "
        + "what is common and what changes across these frames."
    )

    user_content = [{"type": "text", "text": intro_text}]
    for p in image_paths:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"},
            }
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            timeout=TIMEOUT,
        )
        text = (response.choices[0].message.content or "").strip()

        usage = getattr(response, "usage", None)
        if usage is not None:
            TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
            TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
        REQUEST_COUNT += 1
    except Exception as e:
        return None, f"Request failed: {e}"

    return text, None


# MAIN PIPELINE

def _load_ppt_excluded_times() -> dict:
    """If ppt_excluded_times.json exists, return {minute_id: set(frame_id)}; otherwise return {}."""
    if not os.path.isfile(PPT_EXCLUDED_TIMES_JSON):
        return {}
    try:
        with open(PPT_EXCLUDED_TIMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_icon_excluded_frames() -> dict:
    """If icon_excluded_frames.json exists, return {minute_id: {grid_id: set(frame_id)}}; otherwise return {}."""
    if not os.path.isfile(ICON_EXCLUDED_FRAMES_JSON):
        return {}
    try:
        with open(ICON_EXCLUDED_FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {mid: {gid: set(flist) for gid, flist in grids.items()} for mid, grids in data.items()}
    except Exception:
        return {}


def _load_black_excluded_frames() -> dict:
    """If black_excluded_frames.json exists, return {minute_id: {grid_id: set(frame_id)}}; otherwise return {}."""
    if not os.path.isfile(BLACK_EXCLUDED_FRAMES_JSON):
        return {}
    try:
        with open(BLACK_EXCLUDED_FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {mid: {gid: set(flist) for gid, flist in grids.items()} for mid, grids in data.items()}
    except Exception:
        return {}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ppt_excluded_times = _load_ppt_excluded_times()
    icon_excluded_frames = _load_icon_excluded_frames()
    black_excluded_frames = _load_black_excluded_frames()
    if ppt_excluded_times:
        n = sum(len(v) for v in ppt_excluded_times.values())
        print(f"[PPT] Excluding {n} time slots (grid_00=PPT) from all grids")
    if icon_excluded_frames:
        n = sum(len(v) for g in icon_excluded_frames.values() for v in g.values())
        print(f"[ICON] Excluding {n} single frames (icon per grid)")
    if black_excluded_frames:
        n = sum(len(v) for g in black_excluded_frames.values() for v in g.values())
        print(f"[BLACK] Excluding {n} single frames (black per grid)")

    for minute_id in sorted(os.listdir(IMAGE_ROOT)):
        video_dir = os.path.join(IMAGE_ROOT, minute_id)
        if not os.path.isdir(video_dir):
            continue

        print(f"\n[MINUTE] {minute_id}")
        video_outputs = {}
        ppt_frames = ppt_excluded_times.get(minute_id, set())
        icon_grid_frames = icon_excluded_frames.get(minute_id, {})
        black_grid_frames = black_excluded_frames.get(minute_id, {})

        for grid_id in sorted(os.listdir(video_dir)):
            grid_dir = os.path.join(video_dir, grid_id)
            if not os.path.isdir(grid_dir):
                continue

            all_paths = sorted(
                [
                    os.path.join(grid_dir, f)
                    for f in os.listdir(grid_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            icon_frames = icon_grid_frames.get(grid_id, set())
            black_frames = black_grid_frames.get(grid_id, set())
            # Exclude: frames removed by PPT at this timestamp (all grids)
            #        + icon-filtered frames for this grid
            #        + black-frame-filtered frames for this grid
            image_paths = [
                p
                for p in all_paths
                if (frame_id := os.path.splitext(os.path.basename(p))[0]) not in ppt_frames
                and frame_id not in icon_frames
                and frame_id not in black_frames
            ]

            if not image_paths:
                continue

            text, err = run_minute_frames_to_text(minute_id, grid_id, image_paths)
            if err or not text:
                continue

            lower = text.lower()
            if lower.startswith(("i'm sorry", "i’m sorry", "i cannot", "i can't")):
                continue

            # Output only one minute-level narrative per grid
            video_outputs[grid_id] = text
            print(f"[OK] {grid_id}: 1 minute-level description")

        if video_outputs:
            out_path = os.path.join(OUTPUT_DIR, f"{minute_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(video_outputs, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {out_path}")

    # Print basic token usage and efficiency stats
    if REQUEST_COUNT > 0:
        avg_prompt = TOTAL_PROMPT_TOKENS / REQUEST_COUNT
        avg_completion = TOTAL_COMPLETION_TOKENS / REQUEST_COUNT
        print(
            f"\n[USAGE] requests={REQUEST_COUNT}, "
            f"total_prompt_tokens={TOTAL_PROMPT_TOKENS}, "
            f"total_completion_tokens={TOTAL_COMPLETION_TOKENS}, "
            f"avg_prompt_tokens_per_req={avg_prompt:.1f}, "
            f"avg_completion_tokens_per_req={avg_completion:.1f}"
        )


if __name__ == "__main__":
    main()
