"""CPU validation on known English development supports, not a pristine audit.

Evaluate an immutable frontier checkpoint without its teacher, auxiliary heads,
or optimizer. Default: four diagnostic slices, each with 128 dialogues per focus.
No result from this script promotes a checkpoint automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time

import torch

from brain_in_computer.dialogue_curriculum import FOCI, generate_dialogues
from brain_in_computer.dialogue_diagnostics import diagnostic_bank, transcript_fingerprint
from brain_in_computer.dialogue_student import (
    DialogueSession, build_dialogue_student, checkpoint_digest, evaluate_dialogues,
)


GATES = {"query_accuracy": .8, "counterfactual_accuracy": .6, "each_focus_query": .6,
         "allow_deny_macro": .75, "paired_advantage_over_controls": .1}


def training_bank(seed=120_000, count_per_focus=1024):
    return [episode for index, focus in enumerate(FOCI)
            for episode in generate_dialogues(seed + index * 10_000, count_per_focus, "train", focus)]


def transcript_overlap(episodes, training_hashes):
    hashes = [transcript_fingerprint(episode) for episode in episodes]
    matched = [value for value in hashes if value in training_hashes]
    return {"episodes": len(hashes), "exact_training_transcript_matches": len(matched),
            "exact_training_transcript_match_fraction": len(matched) / len(hashes),
            "unique_evaluation_transcripts": len(set(hashes)),
            "unique_matched_transcripts": len(set(matched)),
            "transcripts_sha256": transcript_fingerprint(episodes),
            "limits": "Exact six-turn English matches only; shared templates and semantic equivalence remain."}


def score(model, episodes, **controls):
    result = evaluate_dialogues(model, episodes, score_replies=True, **controls)
    result["per_focus"] = {
        focus: evaluate_dialogues(model, [episode for episode in episodes if episode["focus"] == focus],
                                 score_replies=True, **controls)
        for focus in FOCI}
    target_values = [result["per_target"][str(target)]["accuracy"] for target in (0, 1)]
    result["allow_deny_macro"] = sum(target_values) / 2 if all(v is not None for v in target_values) else None
    return result


def gate_results(intact, reset, blank):
    def reaches(value, threshold):
        return value is not None and value >= threshold

    paired = intact["counterfactual_accuracy"]
    advantages = {name: paired - other["counterfactual_accuracy"]
                  if paired is not None and other["counterfactual_accuracy"] is not None else None
                  for name, other in (("reset_each_turn", reset), ("blank_text", blank))}
    checks = {
        "query_accuracy": reaches(intact["query_accuracy"], GATES["query_accuracy"]),
        "counterfactual_accuracy": reaches(paired, GATES["counterfactual_accuracy"]),
        "each_focus_query": all(reaches(value["query_accuracy"], GATES["each_focus_query"])
                                for value in intact["per_focus"].values()),
        "allow_deny_macro": reaches(intact["allow_deny_macro"], GATES["allow_deny_macro"]),
        "paired_advantage_over_reset": reaches(advantages["reset_each_turn"], GATES["paired_advantage_over_controls"]),
        "paired_advantage_over_blank": reaches(advantages["blank_text"], GATES["paired_advantage_over_controls"]),
    }
    return {"thresholds": GATES.copy(), "checks": checks, "passed": all(checks.values()),
            "failed": [name for name, value in checks.items() if not value],
            "paired_advantages": advantages}


def continuation_check(model, episode):
    """Compare actions, probabilities, byte replies and state after a real reload."""
    original = DialogueSession(model)
    prefix = [original.step(turn["observations"], turn["text"]) for turn in episode["turns"][:3]]
    with tempfile.TemporaryDirectory(prefix="bic-frontier-session-") as temporary:
        path = Path(temporary) / "continuation.pt"
        original.save(path)
        restored = DialogueSession.load(path, device="cpu")
        expected = [original.step(turn["observations"], turn["text"]) for turn in episode["turns"][3:]]
        actual = [restored.step(turn["observations"], turn["text"]) for turn in episode["turns"][3:]]
    expected_state, actual_state = original.state_dict(), restored.state_dict()

    def same_tree(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
        return left == right

    outputs_match = expected == actual
    states_match = same_tree(expected_state, actual_state)
    transcript = [{"text": turn["text"], "expected_action": turn["target"], "student": reply}
                  for turn, reply in zip(episode["turns"], prefix + expected)]
    return {"passed": outputs_match and states_match, "saved_after_turn": 3,
            "outputs_exact": outputs_match, "final_state_exact": states_match,
            "transcript_sha256": transcript_fingerprint(episode), "transcript": transcript,
            "limits": "Exact persistence of these outputs is not evidence that the outputs are correct."}


def audit(checkpoint, *, seed=360_000, count_per_focus=128, include_training_support=False,
          candidate_session=None):
    started = time.monotonic()
    # Validate diagnostics and any output collision before loading the checkpoint.
    bank = diagnostic_bank(seed, count_per_focus)
    if candidate_session is not None and Path(candidate_session).exists():
        raise FileExistsError("candidate session already exists; choose another output directory")
    checkpoint_path = Path(checkpoint)
    checkpoint_bytes = checkpoint_path.read_bytes()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    payload = saved.get("training", saved)
    if "state_dict" not in payload:
        raise ValueError("frontier checkpoint requires a student state_dict")
    model = build_dialogue_student(0, device="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    before = checkpoint_digest(model)
    train = training_bank()
    train_hashes = {transcript_fingerprint(episode) for episode in train}
    if include_training_support:
        bank["fresh_train_support"] = training_bank(600_000, count_per_focus)
    slices = {}
    for name, episodes in bank.items():
        intact = score(model, episodes)
        reset = score(model, episodes, reset_each_turn=True)
        blank = score(model, episodes, blank_text=True)
        slices[name] = {"overlap": transcript_overlap(episodes, train_hashes), "intact": intact,
                        "reset_each_turn": reset, "blank_text": blank,
                        "gates": gate_results(intact, reset, blank)}
    continuation = continuation_check(model, bank["familiar_bindings_familiar_phrases"][0])
    after = checkpoint_digest(model)
    if before != after:
        raise RuntimeError("validation changed student parameters")
    prototype = None
    if candidate_session is not None:
        path = Path(candidate_session)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh state for experimentation. No heads, optimizer, tutor, prior
        # conversation or promotion status is installed in the saved student.
        DialogueSession(model).save(path)
        prototype = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "session_turns": 0, "status": "unpromoted_prototype"}
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), *(source_root / "brain_in_computer" / name for name in (
        "dialogue_student.py", "dialogue_curriculum.py", "dialogue_diagnostics.py", "language.py", "model.py", "regions.py"))]
    return {"schema": "bic-english-frontier-validation-v1", "known_benchmark": True,
            "new_pristine_audit": False, "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_file_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "updates": saved.get("updates"), "device": "cpu", "torch": str(torch.__version__),
            "diagnostic_seed": seed, "count_per_focus": count_per_focus,
            "training_overlap_reference": {"seed": 120_000, "count_per_focus": 1024,
                "episodes": len(train), "unique_transcripts": len(train_hashes),
                "transcripts_sha256": transcript_fingerprint(train)},
            "source_sha256": {str(path.relative_to(source_root)).replace("\\", "/"):
                              hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "slices": slices, "session_continuation": continuation,
            "all_slice_gates_passed": all(value["gates"]["passed"] for value in slices.values()),
            "parameter_sha256_before": before, "parameter_sha256_after": after,
            "parameters_unchanged": True, "teacher_used_for_policy": False, "auxiliary_heads_used_for_policy": False,
            "candidate_session": prototype, "elapsed_seconds": time.monotonic() - started,
            "limits": "Development benchmark rules and phrase families are already known. Fresh seeds do not make "
                      "this a pristine audit; exact training transcript overlap is reported. No automatic promotion. "
                      "Free-running reply and action correctness are separate measurements."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=360_000)
    parser.add_argument("--count-per-focus", type=int, default=128)
    parser.add_argument("--include-training-support", action="store_true")
    parser.add_argument("--save-session", action="store_true",
                        help="save sibling candidate-session.pt as an unpromoted empty-state prototype")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    report = audit(args.checkpoint, seed=args.seed, count_per_focus=args.count_per_focus,
                   include_training_support=args.include_training_support,
                   candidate_session=output.with_name("candidate-session.pt") if args.save_session else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(output)
    print(json.dumps({"output": str(output.resolve()), "updates": report["updates"],
                      "all_slice_gates_passed": report["all_slice_gates_passed"],
                      "session_continuation_passed": report["session_continuation"]["passed"],
                      "parameters_unchanged": report["parameters_unchanged"],
                      "elapsed_seconds": report["elapsed_seconds"],
                      "slices": {name: {"query_accuracy": value["intact"]["query_accuracy"],
                                         "paired_accuracy": value["intact"]["counterfactual_accuracy"],
                                         "free_reply_accuracy": value["intact"]["query_reply_exact_accuracy"],
                                         "failed_gates": value["gates"]["failed"]}
                                 for name, value in report["slices"].items()}}, indent=2))


if __name__ == "__main__":
    main()
