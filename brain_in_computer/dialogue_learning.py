"""Continuous, English-input BiC learning in a small verified dialogue world.

The teacher is external. State, actions, and responses belong to the learned
language/brain network. Every episode presents English teaching and correction,
with no hidden rule or task identifier in its numerical observations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import torch

from .dialogue_curriculum import generate_dialogues
from .dialogue_student import build_dialogue_student, evaluate_dialogues, train_dialogue_candidate
from .dialogue_tutor import DialogueTutor
from .learning_loop import atomic_json, run_lock


FOCI = ("grounding", "revision", "uncertainty")
SCHEMA = "bic-continuing-english-v1"


def fingerprint():
    digest = hashlib.sha256()
    for name in ("dialogue_learning.py", "dialogue_student.py", "dialogue_curriculum.py",
                 "language.py", "model.py", "regions.py", "learning_loop.py", "learning_student.py",
                 "dialogue_tutor.py", "curriculum_tutor.py", "curriculum.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DialogueConfig:
    seed: int = 401
    steps: int = 50
    batch_size: int = 32
    lessons: int = 128
    replay_per_focus: int = 32
    dev_per_focus: int = 48
    candidates: int = 2
    learning_rate: float = .003
    retention_tolerance: float = .08
    device: str = "cpu"
    tutor_model: str | None = None

    def __post_init__(self):
        for name in ("seed", "steps", "batch_size", "lessons", "replay_per_focus", "dev_per_focus", "candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        if self.candidates > 2 or self.device not in ("cpu", "cuda"):
            raise ValueError("at most two local branches on cpu/cuda")
        for name in ("learning_rate", "retention_tolerance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"invalid {name}")
        if self.tutor_model is not None and (not isinstance(self.tutor_model, str) or not self.tutor_model):
            raise ValueError("invalid local tutor model")


def assess(model, seed, count, split="dev", **ablations):
    ablations.setdefault("score_replies", split == "audit")
    results = {focus: evaluate_dialogues(model, generate_dialogues(seed + index * 10_000, count,
                                                                  split=split, focus=focus), **ablations)
               for index, focus in enumerate(FOCI)}
    if not all(math.isfinite(v[k]) for v in results.values() for k in ("query_accuracy", "accuracy", "loss")):
        raise ValueError("nonfinite dialogue metrics")
    return {"query_accuracy": sum(v["query_accuracy"] for v in results.values()) / len(results),
            "accuracy": sum(v["accuracy"] for v in results.values()) / len(results),
            "loss": sum(v["loss"] for v in results.values()) / len(results), "per_focus": results}


class DialogueLoop:
    def __init__(self, directory, config=None, *, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.pt"
        self.source = fingerprint()
        if resume:
            checkpoint_bytes = self.path.read_bytes()
            payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
            if payload["schema"] != SCHEMA or payload["source"] != self.source:
                raise ValueError("dialogue source changed; use a new experiment")
            self.config = DialogueConfig(**payload["config"])
            if config is not None and config != self.config:
                raise ValueError("resume configuration is frozen")
            self.weights, self.optimizer, self.explorer = payload["weights"], payload["optimizer"], payload["explorer"]
            self.state = payload["state"]
            self.checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        else:
            if any(self.directory.iterdir()):
                raise FileExistsError("new dialogue runs need an empty directory")
            self.config = config or DialogueConfig()
            with run_lock(self.directory):
                if self.path.exists():
                    raise FileExistsError("dialogue run already exists")
                model = build_dialogue_student(self.config.seed, device=self.config.device)
                self.weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.optimizer = self.explorer = None
                development = assess(model, self.config.seed + 20_000_000, self.config.dev_per_focus)
                self.state = {"cycle": 0, "promotions": 0, "updates": 0, "wall_seconds": 0.,
                              "development": development, "anchors": {},
                              "replay": {f: [] for f in FOCI}, "last": {f: -1 for f in FOCI},
                              "history": [], "status": "ready"}
                self.save()
                torch.save(self.snapshot(), self.directory / "initial.pt")
                atomic_json(self.directory / "protocol.json", {
                    "schema": SCHEMA, "config": asdict(self.config), "source": self.source,
                    "student": "existing byte-language modules connected to persistent 13-region Brain; no pretrained LLM",
                    "inputs": "current English utterance and public numeric observations only; caller-owned state",
                    "selection": "development-only query accuracy, same-cohort paired benefit and fixed retention anchors",
                    "audit": "separate composition/phrase split, no teacher, reset-state and blank-English controls",
                    "evolution": "two learning-rate branches; no oracle/objective/source mutation",
                    "scope": "inert tool borrowing simulation, no real-world actions or suffering objectives",
                    "budget": "wall hours; completes atomic updates/evaluations before saving"})
                source_directory = self.directory / "source"
                source_directory.mkdir()
                for name in ("dialogue_learning.py", "dialogue_student.py", "dialogue_curriculum.py",
                             "dialogue_tutor.py", "language.py", "model.py", "regions.py",
                             "learning_loop.py", "learning_student.py", "curriculum_tutor.py", "curriculum.py"):
                    (source_directory / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        self.tutor = DialogueTutor(self.config.tutor_model)

    def snapshot(self):
        return {"schema": SCHEMA, "source": self.source, "config": asdict(self.config),
                "weights": self.weights, "optimizer": self.optimizer, "explorer": self.explorer, "state": self.state}

    def save(self):
        temporary = self.path.with_suffix(".pt.tmp")
        torch.save(self.snapshot(), temporary)
        os.replace(temporary, self.path)
        self.checkpoint_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        atomic_json(self.directory / "status.json", self.state)

    def model(self):
        model = build_dialogue_student(self.config.seed, device=self.config.device)
        model.load_state_dict(self.weights)
        return model

    def _check(self):
        if fingerprint() != self.source:
            raise ValueError("dialogue source changed during run")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.checkpoint_hash:
            raise RuntimeError("dialogue checkpoint advanced; reload it")

    def cycle(self, deadline):
        self._check()
        c, state = self.config, copy.deepcopy(self.state)
        number = state["cycle"]
        eligible = ["grounding"] if number < 4 else list(FOCI)
        focus = max(eligible, key=lambda f: 1 - state["development"]["per_focus"][f]["query_accuracy"] +
                    .025 * min(30, number - state["last"][f]))
        seed = c.seed + number * 100_000
        episodes = generate_dialogues(seed, c.lessons, split="train", focus=focus)
        question = {
            "grounding": "Which tool does this new word refer to, and may I borrow it?",
            "revision": "How does the corrected rule change my answer?",
            "uncertainty": "Do I know enough to answer, or should I ask for clarification?"}[focus]
        incumbent = self.model()
        attempt = evaluate_dialogues(incumbent, episodes[:2], score_replies=False)
        tutor = {"source": "none", "question": question, "focus": focus, "rejection_reason": None}
        if attempt["query_accuracy"] < 1.:
            tutor = self.tutor.teach(focus=focus, seed=seed + 50_000, question=question, deadline=deadline)
            episodes.append(tutor["episode"])
        for old_focus in FOCI:
            if old_focus != focus:
                episodes.extend(generate_dialogues(s, 1, split="train", focus=old_focus)[0]
                                for s in state["replay"][old_focus])
        dev_seed = c.seed + 100_000_000 + number * 100_000
        before = assess(incumbent, dev_seed, c.dev_per_focus)
        if time.monotonic() >= deadline:
            return None
        def branch(index):
            prior = self.explorer if index else None
            trained = train_dialogue_candidate(self.weights if prior is None else prior["weights"], episodes,
                        seed=seed + index * 13, steps=c.steps, batch_size=c.batch_size,
                        learning_rate=c.learning_rate * (1. if index else .7),
                        optimizer_state=self.optimizer if prior is None else prior["optimizer"],
                        deadline=deadline, device=c.device)
            if not trained["updates"]:
                return {"index": index, "trained": trained, "accepted": False, "reason": "no_updates",
                        "score": 0., "development": before, "retention": before}
            model = build_dialogue_student(c.seed, device=c.device)
            model.load_state_dict(trained["state_dict"])
            development = assess(model, dev_seed, c.dev_per_focus)
            retention = assess(model, c.seed + 20_000_000, c.dev_per_focus)
            for metrics in (before, development, retention):
                values = [metrics[k] for k in ("query_accuracy", "accuracy", "loss")]
                values.extend(v[k] for v in metrics["per_focus"].values() for k in ("query_accuracy", "accuracy", "loss"))
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("nonfinite dialogue candidate metrics")
            score = development["query_accuracy"] - before["query_accuracy"] + .02 * (before["loss"] - development["loss"])
            reason = "benefit"
            if not all(math.isfinite(v) for v in (score, development["loss"], development["query_accuracy"])):
                reason, score = "nonfinite", 0.
            elif development["query_accuracy"] < before["query_accuracy"] - .01 or score < .002:
                reason = "insufficient_benefit"
            for topic, best in state["anchors"].items():
                if not math.isfinite(best):
                    raise ValueError("nonfinite dialogue retention anchor")
                if (retention["per_focus"][topic]["query_accuracy"] < best - c.retention_tolerance or
                    development["per_focus"][topic]["query_accuracy"] < before["per_focus"][topic]["query_accuracy"] - c.retention_tolerance):
                    reason = "retention:" + topic
            return {"index": index, "trained": trained, "accepted": reason == "benefit",
                    "reason": reason, "score": score, "development": development, "retention": retention}
        with ThreadPoolExecutor(max_workers=c.candidates if c.device == "cpu" else 1) as pool:
            results = list(pool.map(branch, range(c.candidates)))
        if not any(r["trained"]["updates"] for r in results):
            return None
        winner = max((r for r in results if r["accepted"]), key=lambda r: (r["score"], -r["index"]), default=None)
        state["development"] = winner["development"] if winner else before
        if winner:
            state["promotions"] += 1
            for topic in FOCI:
                accuracy = winner["retention"]["per_focus"][topic]["query_accuracy"]
                if accuracy >= .6 or topic in state["anchors"]:
                    state["anchors"][topic] = max(state["anchors"].get(topic, 0.), accuracy)
        state["cycle"] += 1
        state["updates"] += sum(r["trained"]["updates"] for r in results)
        state["last"][focus] = number
        descriptors = list(dict.fromkeys(state["replay"][focus] + [e["seed"] for e in episodes if e["focus"] == focus]))
        state["replay"][focus] = descriptors[-c.replay_per_focus:]
        state["status"] = "learning"
        event = {"cycle": number, "focus": focus, "question": question,
            "attempt_query_accuracy": attempt["query_accuracy"],
            "tutor": {k: v for k, v in tutor.items() if k != "episode"},
            "winner": None if winner is None else winner["index"],
            "candidates": [{"index": r["index"], "reason": r["reason"], "score": r["score"],
                            "updates": r["trained"]["updates"], "query_accuracy": r["development"]["query_accuracy"]} for r in results]}
        state["history"] = (state["history"] + [event])[-64:]
        weights, optimizer, explorer = self.weights, self.optimizer, self.explorer
        if len(results) > 1 and results[1]["trained"]["updates"]:
            explorer = {"weights": results[1]["trained"]["state_dict"], "optimizer": results[1]["trained"]["optimizer_state"]}
        if winner:
            weights, optimizer = winner["trained"]["state_dict"], winner["trained"]["optimizer_state"]
            if winner["index"] == 0:
                explorer = None
        self.weights, self.optimizer, self.explorer, self.state = weights, optimizer, explorer, state
        return event

    def run(self, hours, *, max_cycles=None):
        if type(hours) not in (int, float) or not math.isfinite(hours) or hours <= 0:
            raise ValueError("hours must be finite and positive")
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise ValueError("max_cycles must be positive")
        deadline, last = time.monotonic() + hours * 3600, time.monotonic()
        count = 0
        with run_lock(self.directory):
            self._check()
            try:
                while time.monotonic() < deadline and (max_cycles is None or count < max_cycles):
                    committed = self.weights, self.optimizer, self.explorer, self.state
                    try:
                        event = self.cycle(deadline)
                    except BaseException:
                        self.weights, self.optimizer, self.explorer, self.state = committed
                        raise
                    if event is None:
                        break
                    count += 1
                    now = time.monotonic()
                    self.state["wall_seconds"] += now - last
                    self.save()
                    last = now
                    print(json.dumps({"cycle": self.state["cycle"], "focus": event["focus"],
                                      "query_accuracy": self.state["development"]["query_accuracy"],
                                      "promoted": event["winner"] is not None}), flush=True)
            except KeyboardInterrupt:
                self.state["status"] = "interrupted"
            finally:
                self.state["wall_seconds"] += time.monotonic() - last
                self.save()
        return self.state

    def audit(self, seed=900_000_001, count=128):
        if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
            raise ValueError("invalid audit seed/count")
        with run_lock(self.directory):
            self._check()
            destination = self.directory / f"audit-{self.checkpoint_hash[:12]}-{seed}.json"
            if destination.exists():
                raise FileExistsError("audit already exists")
            model = self.model()
            result = assess(model, seed, count, split="audit")
            reset = assess(model, seed, count, split="audit", reset_each_turn=True)
            blank = assess(model, seed, count, split="audit", blank_text=True)
            initial = torch.load(self.directory / "initial.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(initial["weights"])
            baseline = assess(model, seed, count, split="audit")
            report = {"schema": SCHEMA, "source": self.source, "checkpoint_sha256": self.checkpoint_hash,
                      "seed": seed, "count_per_focus": count, "cycle": self.state["cycle"],
                      "student": result, "reset_each_turn": reset, "blank_english": blank, "initial": baseline,
                      "checkpoint_unchanged": self.checkpoint_hash == hashlib.sha256(self.path.read_bytes()).hexdigest(),
                      "limits": "Restricted simulated English, new alias/color bindings and phrase templates; "
                                "not adult comprehension. Ablations use the same learned weights."}
            atomic_json(destination, report)
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "resume", "audit", "status", "chat"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tutor-model", help="optional installed local Ollama tutor")
    parser.add_argument("--session", help="optional saved continuing dialogue activity")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "status":
        print((Path(args.output) / "status.json").read_text(encoding="utf-8"))
        return 0
    config = DialogueConfig(seed=args.seed, steps=args.steps, device=args.device, tutor_model=args.tutor_model) if args.command == "run" else None
    loop = DialogueLoop(args.output, config, resume=args.command != "run")
    if args.command == "chat":
        from .dialogue_student import DialogueSession
        from .dialogue_curriculum import TARGET_NAMES
        session = DialogueSession(loop.model())
        session_path = Path(args.session) if args.session else Path(args.output) / "conversation.pt"
        if session_path.exists():
            restored = DialogueSession.load(session_path, device=loop.config.device)
            if restored.checkpoint_sha256 != session.checkpoint_sha256:
                raise ValueError("saved conversation belongs to different weights; choose another --session path")
            session = restored
        observation = generate_dialogues(0, 1)[0]["turns"][0]["observations"]
        print("Experimental English world. Example: A dax is the red tool. Then: Only red tools may be borrowed.")
        print("Ask: May I borrow the dax? Use /reset for a new conversation, /quit to save and stop.")
        while True:
            try:
                utterance = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if utterance.strip() == "/quit":
                break
            if utterance.strip() == "/reset":
                session = DialogueSession(loop.model())
                continue
            response = session.step(observation, utterance)
            print(f'BiC: {response["reply"]}  [decision: {TARGET_NAMES[response["action"]]}]')
            session.save(session_path)
        session.save(session_path)
    elif args.command == "audit":
        print(json.dumps(loop.audit(), indent=2))
    else:
        loop.run(args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
