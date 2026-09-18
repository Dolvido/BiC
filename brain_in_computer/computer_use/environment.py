"""A closed miniature desktop for grounded language experiments.

This simulator never controls the real OS, browser, keyboard, files, or network.
The agent sees only ``pixels`` (float32 [3,32,32], RGB in [0,1]) and ``body``
(float32 [4]: cursor x/95, cursor y/95, field focus, turn steps/10).
Goals, teacher actions, button identities, and reply labels are not observations.
The screen itself visibly presents four colored buttons, editable text, and a
last-click color swatch. A recall question therefore tests reading visible state,
not hidden episodic or hippocampal memory.

PROMPT_TEMPLATES declares every exact paraphrase family. Train/validation/test
families are disjoint, but colors, positions, words, and task meanings overlap.
They are limited synthetic paraphrase tests, not general English evaluation.
Episode generation uniformly samples tasks and relevant color/position/word
values independently; finite draws are approximately, not exactly, balanced.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


COLORS = ("red", "blue", "green", "yellow")
POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
ACTION_NAMES = (
    "click_top_left", "click_top_right", "click_bottom_left", "click_bottom_right",
    "click_field", "key_h", "key_i", "key_o", "key_k", "backspace", "stop",
)
TASKS = ("click_color", "click_position", "describe_position", "type_word", "recall", "greet", "clarify")
WIDTH = HEIGHT = 96
MAX_STEPS = 10
BUTTON_CENTERS = ((25, 20), (70, 20), (25, 51), (70, 51))
FIELD_CENTER = (32, 83)
PALETTE = {"red": (228, 57, 57), "blue": (48, 104, 229),
           "green": (42, 173, 82), "yellow": (239, 207, 47)}

PROMPT_TEMPLATES = {
    "train": {
        "click_color": ("click the {color} button", "click {color}", "press the {color} button"),
        "click_position": ("click the {position} button", "click {position}", "press the button at {position}"),
        "describe_position": ("what color is the {position} button", "what color is at {position}", "tell me the {position} color"),
        "type_word": ("type {word}", "enter {word}", "write {word} in the field"),
        "recall": ("what did you click", "what was the last color clicked", "show the last clicked color"),
        "greet": ("hello", "hi", "hey"),
        "clarify": ("open the moon", "make me a sandwich", "sing a song"),
    },
    "validation": {
        "click_color": ("please click {color}", "choose the {color} button"),
        "click_position": ("please click {position}", "choose the button at {position}"),
        "describe_position": ("name the color at {position}", "which color is the {position} button"),
        "type_word": ("please type {word}", "put {word} in the field"),
        "recall": ("which color did you click last", "tell me your last clicked color"),
        "greet": ("hello there", "good morning"),
        "clarify": ("cook some pasta", "find a spaceship"),
    },
    "test": {
        "click_color": ("could you press {color}", "select the {color} button"),
        "click_position": ("could you press the {position} button", "select the button at {position}"),
        "describe_position": ("tell me what color is at {position}", "identify the color of the {position} button"),
        "type_word": ("could you enter {word}", "fill the field with {word}"),
        "recall": ("name the color you last clicked", "what color was selected last"),
        "greet": ("greetings", "good evening"),
        "clarify": ("fly to mars", "dance with a dragon"),
    },
}


@dataclass(frozen=True)
class Goal:
    """Teacher/evaluation metadata; never pass this object to the agent model."""

    task: str
    prompt: str
    target: str
    expected_reply: str


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


class MiniDesktop:
    """A deterministic raster desktop with an explicit, bounded event loop."""

    def __init__(self, seed: int = 0, shift: bool = False):
        if type(seed) is not int or type(shift) is not bool:
            raise ValueError("seed must be an integer and shift a boolean")
        self.seed, self.shift = seed, shift
        self.generator = random.Random(seed)
        self.reset()

    def reset(self, button_colors: Sequence[str] | None = None,
              last_clicked: str | None = None, text: str = "") -> None:
        colors = list(COLORS) if button_colors is None else list(button_colors)
        if len(colors) != 4 or set(colors) != set(COLORS):
            raise ValueError("button_colors must be a permutation of the four colors")
        if button_colors is None:
            self.generator.shuffle(colors)
        if last_clicked is not None and last_clicked not in COLORS:
            raise ValueError("last_clicked must be None or a known color")
        if not isinstance(text, str) or len(text) > 2 or any(char not in "hiok" for char in text):
            raise ValueError("text must contain at most two characters from h, i, o, k")
        self.button_colors = colors
        self.last_clicked, self.text = last_clicked, text
        self.cursor = (47.0, 47.0)
        self.focused = False
        self.button_rects = []
        for x, y in BUTTON_CENTERS:
            dx = self.generator.randint(-2, 2) if self.shift else 0
            dy = self.generator.randint(-2, 2) if self.shift else 0
            self.button_rects.append((x - 17 + dx, y - 11 + dy, x + 17 + dx, y + 11 + dy))
        self.field_rect = (7, 74, 58, 92)
        self.begin_turn()

    def begin_turn(self) -> None:
        """Start another instruction while retaining the visible desktop state."""
        self.steps = 0
        self.terminated = False
        self.stop_requested = False
        self.action_history: list[dict] = []

    def _ensure_active(self) -> None:
        if self.terminated:
            raise RuntimeError("This turn has ended; call begin_turn before another event")

    def _record(self, event: dict) -> dict:
        self.action_history.append(event)
        self.steps += 1
        self.terminated = self.stop_requested or self.steps >= MAX_STEPS
        return {"terminated": self.terminated}

    @staticmethod
    def _contains(rect, x, y) -> bool:
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def click(self, x: float, y: float) -> dict:
        """Hit-test a real screen coordinate; invalid off-screen clicks fail."""
        self._ensure_active()
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("click coordinates must be finite real numbers")
        if not 0 <= x <= WIDTH - 1 or not 0 <= y <= HEIGHT - 1:
            raise ValueError("click coordinates must be within the 96x96 screen")
        self.cursor = (float(x), float(y))
        event = {"kind": "click", "x": float(x), "y": float(y), "hit": "background"}
        self.focused = self._contains(self.field_rect, x, y)
        if self.focused:
            event["hit"] = "field"
        else:
            for index, rect in enumerate(self.button_rects):
                if self._contains(rect, x, y):
                    self.last_clicked = self.button_colors[index]
                    event.update(hit="button", position=POSITIONS[index], color=self.last_clicked)
                    break
        return self._record(event)

    def key(self, text: str) -> dict:
        """Deliver one key event; only the small declared alphabet is supported."""
        self._ensure_active()
        if text not in ("h", "i", "o", "k", "backspace"):
            raise ValueError("key must be h, i, o, k, or backspace")
        if self.focused:
            if text == "backspace":
                self.text = self.text[:-1]
            elif len(self.text) < 2:
                self.text += text
        return self._record({"kind": "key", "key": text})

    def step(self, action_id: int) -> dict:
        self._ensure_active()
        if type(action_id) is not int or not 0 <= action_id < len(ACTION_NAMES):
            raise ValueError("action_id must be an integer in [0, 11)")
        if action_id < 4:
            return self.click(*BUTTON_CENTERS[action_id])
        if action_id == 4:
            return self.click(*FIELD_CENTER)
        if action_id < 9:
            return self.key(ACTION_NAMES[action_id][-1])
        if action_id == 9:
            return self.key("backspace")
        self.stop_requested = True
        return self._record({"kind": "stop"})

    def render(self) -> Image.Image:
        background = (230, 234, 241) if self.shift else (243, 245, 248)
        image = Image.new("RGB", (WIDTH, HEIGHT), background)
        draw = ImageDraw.Draw(image)
        for color, rect in zip(self.button_colors, self.button_rects):
            rgb = PALETTE[color]
            if self.shift:
                rgb = tuple(round(value * 0.94 + 5) for value in rgb)
            draw.rounded_rectangle(rect, radius=3, fill=rgb, outline=(35, 39, 47), width=1)
        draw.rectangle(self.field_rect, fill=(255, 255, 255),
                       outline=(29, 98, 220) if self.focused else (75, 79, 86), width=2 if self.focused else 1)
        try:
            font = ImageFont.load_default(size=14)
        except TypeError:  # Pillow before scalable built-in fonts.
            font = ImageFont.load_default()
        draw.text((11, 75), self.text, fill=(15, 18, 24), font=font)
        draw.text((67, 67), "last", fill=(50, 55, 64), font=ImageFont.load_default())
        swatch = (68, 79, 88, 92)
        draw.rectangle(swatch, fill=PALETTE[self.last_clicked] if self.last_clicked else (255, 255, 255),
                       outline=(35, 39, 47), width=1)
        if self.last_clicked is None:
            draw.line((71, 82, 85, 89), fill=(90, 90, 90), width=1)
            draw.line((71, 89, 85, 82), fill=(90, 90, 90), width=1)
        x, y = self.cursor
        draw.line((x - 2, y, x + 2, y), fill=(10, 10, 10), width=1)
        draw.line((x, y - 2, x, y + 2), fill=(10, 10, 10), width=1)
        return image

    def observe(self) -> dict[str, torch.Tensor]:
        image = self.render().resize((32, 32), resample=Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        return {"pixels": pixels,
                "body": torch.tensor([self.cursor[0] / 95, self.cursor[1] / 95,
                                       float(self.focused), self.steps / MAX_STEPS], dtype=torch.float32)}

    def state_dict(self) -> dict:
        """JSON-friendly state, including RNG for deterministic future resets."""
        return deepcopy({"schema_version": 1, "seed": self.seed, "shift": self.shift,
                         "button_colors": self.button_colors, "last_clicked": self.last_clicked,
                         "text": self.text, "cursor": list(self.cursor), "focused": self.focused,
                         "steps": self.steps, "terminated": self.terminated,
                         "stop_requested": self.stop_requested, "action_history": self.action_history,
                         "button_rects": [list(rect) for rect in self.button_rects],
                         "field_rect": list(self.field_rect), "rng_state": self.generator.getstate()})

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported desktop state schema")
        restored = MiniDesktop(state["seed"], state["shift"])
        restored.reset(state["button_colors"], state["last_clicked"], state["text"])
        cursor = state["cursor"]
        if len(cursor) != 2 or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                                    or not 0 <= v <= WIDTH - 1 for v in cursor):
            raise ValueError("Invalid cursor in saved state")
        if type(state["steps"]) is not int or not 0 <= state["steps"] <= MAX_STEPS:
            raise ValueError("Invalid step count in saved state")
        for key in ("focused", "terminated", "stop_requested"):
            if type(state[key]) is not bool:
                raise ValueError(f"Invalid {key} in saved state")
        if state["terminated"] != (state["stop_requested"] or state["steps"] >= MAX_STEPS):
            raise ValueError("Inconsistent terminal state")
        if not isinstance(state["action_history"], list) or len(state["action_history"]) != state["steps"]:
            raise ValueError("Action history must match the saved step count")
        rects = [*state["button_rects"], state["field_rect"]]
        if len(rects) != 5 or any(len(rect) != 4 or any(type(v) is not int or not 0 <= v < WIDTH for v in rect)
                                 or rect[0] > rect[2] or rect[1] > rect[3] for rect in rects):
            raise ValueError("Invalid saved screen geometry")
        restored.cursor = tuple(float(v) for v in cursor)
        for key in ("focused", "steps", "terminated", "stop_requested", "action_history"):
            setattr(restored, key, deepcopy(state[key]))
        restored.button_rects = [tuple(rect) for rect in state["button_rects"]]
        restored.field_rect = tuple(state["field_rect"])
        restored.generator.setstate(_tuples(state["rng_state"]))
        self.__dict__.update(restored.__dict__)

    def clone(self) -> MiniDesktop:
        result = MiniDesktop()
        result.load_state_dict(self.state_dict())
        return result


def make_episode(generator: random.Random, task: str | None = None, split: str = "train",
                 shift: bool = False) -> tuple[MiniDesktop, Goal]:
    if not isinstance(generator, random.Random):
        raise ValueError("generator must be a random.Random instance")
    if split not in PROMPT_TEMPLATES:
        raise ValueError("split must be train, validation, or test")
    task = generator.choice(TASKS) if task is None else task
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    env = MiniDesktop(generator.randrange(2**32), shift=shift)
    # Visible pre-existing state varies independently of the requested task.
    env.last_clicked = generator.choice((None, *COLORS))
    env.text = generator.choice(("", "hi", "ok", "h", "o"))
    target = ""
    values = {}
    if task == "click_color":
        target = generator.choice(COLORS)
        values["color"] = target
        reply = f"clicked {target}."
    elif task in ("click_position", "describe_position"):
        target = generator.choice(POSITIONS)
        values["position"] = target.replace("_", " ")
        color = env.button_colors[POSITIONS.index(target)]
        reply = f"clicked {color}." if task == "click_position" else f"it is {color}."
    elif task == "type_word":
        target = generator.choice(("hi", "ok"))
        values["word"] = target
        reply = f"typed {target}."
    elif task == "recall":
        target = env.last_clicked or "nothing"
        reply = f"last was {target}." if env.last_clicked else "nothing yet."
    elif task == "greet":
        reply = "hello."
    else:
        reply = "please rephrase."
    prompt = generator.choice(PROMPT_TEMPLATES[split][task]).format(**values)
    return env, Goal(task=task, prompt=prompt, target=target, expected_reply=reply)


def oracle_action(env: MiniDesktop, goal: Goal) -> int:
    """Teacher-only policy; never give its Goal or result as model input."""
    if env.terminated:
        raise RuntimeError("Oracle requires an active turn")
    if goal.task in ("click_color", "click_position"):
        if env.steps:
            return ACTION_NAMES.index("stop")
        return env.button_colors.index(goal.target) if goal.task == "click_color" else POSITIONS.index(goal.target)
    if goal.task == "type_word":
        if env.text == goal.target:
            return ACTION_NAMES.index("stop")
        if not env.focused:
            return ACTION_NAMES.index("click_field")
        if not goal.target.startswith(env.text):
            return ACTION_NAMES.index("backspace")
        return ACTION_NAMES.index(f"key_{goal.target[len(env.text)]}")
    if goal.task in ("describe_position", "recall", "greet", "clarify"):
        return ACTION_NAMES.index("stop")
    raise ValueError(f"Unknown task: {goal.task}")


def goal_success(env: MiniDesktop, goal: Goal) -> bool:
    """Evaluate stopped behavior; timeout alone never counts as success."""
    if not env.terminated or not env.stop_requested or not env.action_history or env.action_history[-1]["kind"] != "stop":
        return False
    if goal.task in ("click_color", "click_position"):
        desired = goal.target if goal.task == "click_color" else env.button_colors[POSITIONS.index(goal.target)]
        clicks = [event for event in env.action_history if event["kind"] != "stop"]
        # Do not count a lucky final correction after clicking a wrong button.
        return env.last_clicked == desired and bool(clicks) and all(
            event["kind"] == "click" and event["hit"] == "button" and event["color"] == desired for event in clicks)
    if goal.task == "type_word":
        return env.text == goal.target
    if goal.task in ("describe_position", "recall", "greet", "clarify"):
        return all(event["kind"] == "stop" for event in env.action_history)
    return False
