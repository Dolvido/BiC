"""Run the published research learner on visible English, entirely on CPU.

Run from the repository root: python -m examples.english_demo [--interactive]
The default examples are fixed illustrations, not a benchmark or success filter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import _generate_reply_tokens, checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments.composition_data import pack_observations
from experiments.sequence_student import SequenceConfig
from experiments.shared_state_student import ARCHITECTURE, build_shared_state_student
from examples.export_demo_weights import SCHEMA, sha256_file


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "bic-lower-2592.pt"
ACTION_NAMES = ("DENY", "ALLOW", "ASK", "ACK")

# Chosen before running the exported model. These are observations only: no
# labels, expected replies, parsed world state, or task IDs enter inference.
SCENARIOS = (
    ("Remember and revise colors", (
        "dax is red.",
        "wug is blue.",
        "Is dax red?",
        "dax is now green.",
        "Is dax red?",
        "Is dax green?",
    )),
    ("Update a count", (
        "dax starts with a count of 3.",
        "Increase the count of dax by 2.",
        "Is the count of dax 5?",
        "Decrease the count of dax by 1.",
        "Is the count of dax 5?",
        "Is the count of dax 4?",
    )),
    ("Use a defined copy operation", (
        "Define jaskel for color: copy source to destination.",
        "Set the color of dax to red.",
        "Set the color of wug to blue.",
        "Apply jaskel to color from dax to wug.",
        "Is the color of wug red?",
        "Is the color of wug blue?",
        "Is the color of dax red?",
    )),
    ("Try a composed operation", (
        "Define jaskel for color: advance source once; copy source to destination.",
        "Define klunvo for color: copy source to destination; advance source once.",
        "Set the color of dax to red.",
        "Set the color of wug to blue.",
        "Apply jaskel to color from dax to wug.",
        "Is the color of dax green?",
        "Is the color of wug green?",
        "Is the color of wug red?",
    )),
)


def load_model(weights_path: Path = DEFAULT_WEIGHTS):
    """Verify the bundled artifact and load exact trained tensors on CPU."""
    weights_path = Path(weights_path)
    manifest = json.loads(weights_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256_file(weights_path) != manifest["asset_sha256"]:
        raise ValueError("inference asset SHA-256 does not match its manifest")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported public inference artifact")
    model = build_shared_state_student(seed=0, config=SequenceConfig(**payload["config"]))
    model.load_state_dict(payload["weights"], strict=True)
    if checkpoint_digest(model) != payload["weights_sha256"] or payload["weights_sha256"] != manifest["weights_sha256"]:
        raise ValueError("native parameter bytes do not match their declared digest")
    model.eval()
    model.requires_grad_(False)
    return model, manifest


class EnglishSession:
    """Explicit text history; rejected input never changes the session."""

    def __init__(self, model):
        self.model = model
        self.history: list[str] = []

    def reset(self) -> None:
        self.history.clear()

    @torch.inference_mode()
    def observe(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("enter a nonempty English utterance")
        c = self.model.config
        if len(self.history) >= c.max_turns:
            raise ValueError(f"session limit is {c.max_turns} turns; use /reset to start a new world")
        proposed = [*self.history, text]
        inputs = pack_observations([proposed], device="cpu", max_turns=c.max_turns,
            max_input_bytes=c.max_input_bytes, max_context_tokens=c.max_positions)
        # BOS is the entire decoder prefix. No reference response is supplied.
        bos = torch.full((1, len(proposed), 1), ByteCodec.BOS, dtype=torch.long)
        output = self.model(**inputs, decoder_input_ids=bos)
        logits = output["logits"][0, -1]
        if not torch.isfinite(logits).all():
            raise ValueError("model produced non-finite action logits")
        generated = _generate_reply_tokens(self.model, output["production_context"][:, -1])[0]
        tokens = generated.tolist()
        try:
            reply = self.model.codec.decode(tokens)
            valid_utf8 = True
        except UnicodeDecodeError:
            reply = self.model.codec.decode(tokens, errors="replace")
            valid_utf8 = False
        self.history.append(text)
        return {
            "turn": len(self.history), "text": text,
            "action": ACTION_NAMES[int(logits.argmax().item())],
            "reply": reply, "reply_tokens": tokens,
            "reply_terminated": ByteCodec.EOS in tokens,
            "valid_utf8": valid_utf8,
            "context_tokens": int(inputs["lengths"][0]),
        }


def run_examples(model) -> list[dict]:
    session = EnglishSession(model)
    results = []
    for title, observations in SCENARIOS:
        session.reset()
        results.append({"scenario": title, "turns": [session.observe(text) for text in observations]})
    return results


def _display(turn: dict) -> None:
    print(f"  {turn['turn']}. {turn['text']}")
    print(f"     {turn['action']} | {json.dumps(turn['reply'], ensure_ascii=True)}")
    if not turn["reply_terminated"] or not turn["valid_utf8"]:
        print("     [decoder output was unterminated or contained invalid UTF-8]")


def interactive(model) -> None:
    session = EnglishSession(model)
    print("Type one sentence at a time. /reset clears the world; /quit exits.")
    print("Replies are generated independently of the action label and may be wrong.")
    while True:
        try:
            text = input(f"English [{len(session.history)}/{model.config.max_turns}]> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip() == "/quit":
            return
        if text.strip() == "/reset":
            session.reset()
            print("New empty world.")
            continue
        try:
            _display(session.observe(text))
        except ValueError as exc:
            print(f"Input rejected; history unchanged: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="type your own bounded English history")
    parser.add_argument("--json", action="store_true", help="emit all fixed-example predictions as JSON")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads, default 2")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.interactive and args.json:
        parser.error("--interactive and --json cannot be combined")
    torch.set_num_threads(args.threads)
    model, manifest = load_model(args.weights)
    if args.json:
        print(json.dumps({"weights_sha256": manifest["weights_sha256"],
            "device": "cpu", "illustrations_not_benchmark": True,
            "scenarios": run_examples(model)}, indent=2))
        return
    print("BiC: local English research demonstration (CPU, no tutor, no training)")
    print("Finite grammar; max 12 turns, 128 UTF-8 bytes per turn, 1,024 context tokens.")
    print("Research candidate: not a promoted home-learning checkpoint or a general chatbot.")
    if args.interactive:
        interactive(model)
    else:
        for result in run_examples(model):
            print(f"\n{result['scenario']}")
            for turn in result["turns"]:
                _display(turn)


if __name__ == "__main__":
    main()
