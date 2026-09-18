"""Record six actual model turns in a persistent simulated desktop.

Usage from the repository root::

    python experiments/record_computer_demo.py \
        --checkpoint runs/computer/checkpoint.pt --output runs/computer/demo

The output directory receives demo.json and demo.png. This script performs
inference only: it does not train, invoke an oracle, or supply expected answers.
Recorded prompts and environment seed are fixed; generated replies and actions
come exclusively from the supplied checkpoint. Timings vary between runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import torch


# Allow the documented direct script invocation in an uninstalled checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain_in_computer import __version__
from brain_in_computer.computer_use.environment import ACTION_NAMES, POSITIONS, MiniDesktop
from brain_in_computer.computer_use.training import load_computer_checkpoint, run_turn


PROMPTS = (
    "hello",
    "click red",
    "what did you click",
    "what color is the top left button",
    "type hi",
    "type ok",
)


def visible_state(env: MiniDesktop) -> dict:
    """Describe human-visible state for the transcript, never for model input."""
    return {
        "buttons": dict(zip(POSITIONS, env.button_colors)),
        "text": env.text,
        "last_clicked": env.last_clicked,
        "cursor": list(env.cursor),
        "field_focused": env.focused,
    }


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, typeface, width: int) -> list[str]:
    """Wrap by rendered pixel width, including unexpectedly long output words."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            proposal = f"{current} {word}" if current else word
            if draw.textlength(proposal, font=typeface) <= width:
                current = proposal
                continue
            if current:
                lines.append(current)
                current = ""
            for character in word:
                if current and draw.textlength(current + character, font=typeface) > width:
                    lines.append(current)
                    current = ""
                current += character
        lines.append(current)
    return lines


def draw_wrapped(draw, text, xy, typeface, width, fill, line_height):
    x, y = xy
    for line in wrap_lines(draw, text, typeface, width):
        draw.text((x, y), line, font=typeface, fill=fill)
        y += line_height
    return y


def contact_sheet(turns: list[dict], screenshots: list[Image.Image], checkpoint_hash: str) -> Image.Image:
    """Show the actual final screen and unedited generated reply for each turn."""
    canvas = Image.new("RGB", (1440, 1260), "#0c1017")
    draw = ImageDraw.Draw(canvas)
    title, normal, small = font(34), font(21), font(16)
    draw.text((28, 24), f"BiC v{__version__} | Recorded model session", font=title, fill="#edf4fb")
    draw.text((30, 73), "Restricted English / simulated desktop / generated replies",
              font=normal, fill="#a7b6c7")
    draw.text((30, 104), "Six consecutive turns; each screenshot is the resulting screen. Seed 7; CPU; one thread.",
              font=small, fill="#8397ad")
    for index, (turn, screenshot) in enumerate(zip(turns, screenshots)):
        x, y = 28 + (index % 2) * 702, 142 + (index // 2) * 352
        draw.rounded_rectangle((x, y, x + 682, y + 334), radius=14,
                               fill="#151e29", outline="#344658", width=1)
        draw_wrapped(draw, f"{index + 1}. You: {turn['prompt']}", (x + 18, y + 17),
                     normal, 645, "#bacaff", 25)
        # Scale the actual 96x96 rendering by an integer factor, without edits.
        canvas.paste(screenshot.resize((192, 192), Image.Resampling.NEAREST), (x + 18, y + 91))
        draw.text((x + 234, y + 85), "BiC generated:", font=small, fill="#95e4c4")
        next_y = draw_wrapped(draw, turn["reply"] or "(empty reply)", (x + 234, y + 108),
                              normal, 425, "#f1f6fb", 25)
        action_y = max(y + 157, next_y + 10)
        draw.text((x + 234, action_y), "Actual actions:", font=small, fill="#9aacbf")
        draw_wrapped(draw, ", ".join(turn["actions"]) or "(none)", (x + 234, action_y + 22),
                     small, 425, "#d3deeb", 21)
        status = "stopped" if turn["stop_requested"] else "action limit reached"
        draw.text((x + 18, y + 301), f"{turn['inference_seconds']:.3f}s | {status}",
                  font=small, fill="#8298ae")
    draw.text((30, 1214), f"Checkpoint SHA-256: {checkpoint_hash}", font=font(14), fill="#7d91a7")
    return canvas


def record(checkpoint: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    checkpoint = checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = load_computer_checkpoint(checkpoint, device="cpu")
    env = MiniDesktop(seed=7)
    # Constructor defaults provide an empty field and no previous click; retain
    # its seeded button arrangement, and preserve this desktop across all turns.
    if env.text or env.last_clicked is not None:
        raise RuntimeError("The recorded session requires an initially empty desktop")
    initial_state = visible_state(env)
    turns, screenshots = [], []
    for index, prompt in enumerate(PROMPTS, start=1):
        before = visible_state(env)
        started = time.perf_counter()
        result = run_turn(model, env, prompt)
        elapsed = time.perf_counter() - started
        if result.get("used_teacher") is not False:
            raise RuntimeError("The controller must explicitly report that no teacher was used")
        action_ids = list(result["actions"])
        turns.append({
            "turn": index,
            "prompt": prompt,
            "reply": result["reply"],
            "action_ids": action_ids,
            "actions": [ACTION_NAMES[action] for action in action_ids],
            "action_events": list(env.action_history),
            "visible_before": before,
            "visible_after": visible_state(env),
            "terminated": bool(result["terminated"]),
            "stop_requested": bool(env.stop_requested),
            "inference_seconds": elapsed,
            "used_teacher": False,
        })
        screenshots.append(env.render().copy())

    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during the recording; no demo was written")
    report = {
        "version": __version__,
        "kind": "recorded_model_session",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "seed": 7,
        "device": "cpu",
        "threads": 1,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "initial_visible_state": initial_state,
        "teacher_used": False,
        "training_performed": False,
        "real_os_access": False,
        "turns": turns,
    }
    sheet = contact_sheet(turns, screenshots, checkpoint_hash)
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / "demo.png")
    (output / "demo.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                     encoding="utf-8")
    for turn in turns:
        print(f"You: {turn['prompt']}\nBiC: {turn['reply']}\nActions: {', '.join(turn['actions'])}\n")
    print(f"Saved {output / 'demo.json'} and {output / 'demo.png'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/computer/checkpoint.pt"))
    parser.add_argument("--output", "--outdir", type=Path, default=Path("runs/computer/demo"),
                        help="Directory receiving demo.json and demo.png")
    args = parser.parse_args(argv)
    record(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
