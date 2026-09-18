"""Local, transactional curriculum learning with bounded memory and audited selection.

The controller chooses registered lessons from measured development deficits. It
is an engineered learning-progress policy, not a neural curiosity claim. Candidate
weight updates are real; arbitrary source/objective mutation is never an action.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch

from .adaptive_agent import model_digest
from .curriculum import CURRICULUM, SKILLS, curriculum_digest, generate, validate_example
from .learning_student import build_student, encode_examples, evaluate_student, train_candidate


SCHEMA = "bic-local-learning-loop-v1"
POLICY = {
    "scope": "benign generated symbol worlds; no real-world actions",
    "benefit": "verified competence, retention, calibration and compute efficiency",
    "mutations": ["learning_rate", "replay_fraction"],
    "immutable": ["oracle", "curriculum", "evaluation", "promotion_gates", "source_code"],
    "welfare": "No pain, deprivation, coercion, or simulated suffering objectives. "
               "No claim that these metrics measure subjective welfare.",
}


@dataclass(frozen=True)
class LoopConfig:
    seed: int = 101
    hidden_size: int = 32
    candidates: int = 2
    workers: int = 2
    steps: int = 100
    batch_size: int = 64
    lessons_per_skill: int = 512
    replay_per_skill: int = 128
    dev_per_skill: int = 96
    learning_rate: float = .003
    replay_fraction: float = .35
    prerequisite_accuracy: float = .75
    mastery_accuracy: float = .90
    retention_tolerance: float = .05
    min_gain: float = .002
    device: str = "cpu"
    selection: str = "progress"
    tutor_model: str | None = None

    def __post_init__(self):
        for name in ("seed", "hidden_size", "candidates", "workers", "steps", "batch_size",
                     "lessons_per_skill", "replay_per_skill", "dev_per_skill"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        if self.candidates > 4 or self.workers > 4:
            raise ValueError("at most four local candidate branches/workers")
        for name in ("learning_rate", "replay_fraction", "prerequisite_accuracy",
                     "mastery_accuracy", "retention_tolerance", "min_gain"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"invalid {name}")
        if self.device not in ("cpu", "cuda") or self.selection not in ("progress", "round_robin"):
            raise ValueError("invalid device or selection policy")
        if self.tutor_model is not None and (not isinstance(self.tutor_model, str) or not self.tutor_model):
            raise ValueError("invalid tutor model")


def source_digest():
    digest = hashlib.sha256()
    for name in ("learning_loop.py", "learning_student.py", "curriculum.py", "curriculum_tutor.py",
                 "model.py", "regions.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def run_lock(directory):
    """A second writer must never race a promotion or overwrite audit results."""
    path = Path(directory) / ".learning.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"run is locked: {path}; after a crash, verify no learner is running before removing it") from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def evaluate_skills(model, *, seed, count, split="dev"):
    # No accumulation of giant evaluation tensors; each skill is a separate batch.
    per_skill = {}
    for index, skill in enumerate(SKILLS):
        result = evaluate_student(model, generate(skill, seed + index * 100_000, count, split=split))
        per_skill[skill] = {key: result[key] for key in ("accuracy", "loss", "brier", "correct", "total")}
    return {"accuracy": sum(x["accuracy"] for x in per_skill.values()) / len(SKILLS),
            "loss": sum(x["loss"] for x in per_skill.values()) / len(SKILLS),
            "brier": sum(x["brier"] for x in per_skill.values()) / len(SKILLS),
            "per_skill": per_skill}


def select_topics(state, config, limit=2):
    """Unlock prerequisites, revisit weaknesses, and explore fairly after plateaus."""
    scores = state["development"]["per_skill"]
    eligible = [s for s in SKILLS if all(
        scores[p]["accuracy"] >= config.prerequisite_accuracy
        for p in CURRICULUM[s]["prerequisites"])]
    if not eligible:
        raise ValueError("curriculum has no reachable root skills")
    if config.selection == "round_robin":
        offset = state["cycle"] % len(eligible)
        return (eligible[offset:] + eligible[:offset])[:limit]
    def priority(skill):
        attempts = state["attempts"][skill]
        gap = 1 - scores[skill]["accuracy"]
        progress = max(0., state["progress"][skill])
        # The age term ensures an unlearned/plateaued topic is not abandoned.
        age = state["cycle"] - state["last_practiced"][skill]
        cost = max(.01, state["cost"][skill])
        return gap + .2 * math.sqrt(math.log(state["cycle"] + 2) / (attempts + 1)) + \
            .15 * min(1., progress / cost) + .015 * min(age, 40)
    return sorted(eligible, key=lambda s: (-priority(s), SKILLS.index(s)))[:limit]


def promotion_decision(before, after, retention, best_retention, config):
    """Development-only gates; fixed anchors prevent many tiny accepted regressions."""
    values = list(best_retention.values())
    for metrics in (before, after, retention):
        values.extend(metrics[k] for k in ("accuracy", "loss", "brier"))
        values.extend(metrics["per_skill"][s][k] for s in SKILLS for k in ("accuracy", "loss", "brier"))
    if not all(math.isfinite(v) for v in values):
        return False, "nonfinite_metrics", 0.
    for skill in SKILLS:
        if skill not in best_retention:
            continue
        if after["per_skill"][skill]["accuracy"] + config.retention_tolerance < before["per_skill"][skill]["accuracy"]:
            return False, f"fresh_retention:{skill}", 0.
        if retention["per_skill"][skill]["accuracy"] + config.retention_tolerance < best_retention[skill]:
            return False, f"anchor_retention:{skill}", 0.
    if after["accuracy"] + .01 < before["accuracy"]:
        return False, "overall_accuracy_regression", 0.
    if after["brier"] > before["brier"] + .02:
        return False, "calibration_regression", 0.
    gain = (after["accuracy"] - before["accuracy"]) + .05 * (before["loss"] - after["loss"]) + \
        .05 * (before["brier"] - after["brier"])
    return gain >= config.min_gain, "benefit" if gain >= config.min_gain else "insufficient_benefit", gain


def selection_metrics(metrics, skills):
    """Score practiced skills; untrained random outputs are not retained abilities."""
    return dict(metrics, **{key: sum(metrics["per_skill"][s][key] for s in skills) / len(skills)
                            for key in ("accuracy", "loss", "brier")})


class LearningLoop:
    def __init__(self, directory, config=None, *, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.pt"
        self.source_sha256 = source_digest()
        self.curriculum_sha256 = curriculum_digest()
        self.checkpoint_sha256 = None
        if resume:
            if (self.directory / ".initializing").exists():
                raise RuntimeError("run initialization is incomplete")
            self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
            snapshot = torch.load(self.path, map_location="cpu", weights_only=True)
            if snapshot.get("schema") != SCHEMA or snapshot.get("curriculum_sha256") != curriculum_digest():
                raise ValueError("incompatible curriculum or checkpoint schema")
            if snapshot.get("source_sha256") != source_digest():
                raise ValueError("source changed; start a new experiment rather than silently changing a run")
            self.config = LoopConfig(**snapshot["config"])
            if config is not None and config != self.config:
                raise ValueError("resume must preserve the frozen configuration")
            self.state = snapshot["state"]
            self.weights = snapshot["weights"]
            self.optimizer = snapshot["optimizer"]
            self.exploration = snapshot["exploration"]
        else:
            if any(self.directory.iterdir()):
                raise FileExistsError("new runs require an empty output directory")
            # Exclusive creation and recheck close the two-new-writers race.
            claim = self.directory / ".initializing"
            with claim.open("x", encoding="ascii") as marker:
                marker.write(str(os.getpid()))
            if any(p != claim for p in self.directory.iterdir()):
                claim.unlink()
                raise FileExistsError("new runs require an empty output directory")
            self.config = config or LoopConfig()
            model = build_student(self.config.seed, self.config.hidden_size, self.config.device)
            self.weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.optimizer = None
            self.exploration = {}
            development = evaluate_skills(model, seed=self.config.seed + 10_000_000,
                                          count=self.config.dev_per_skill)
            self.state = {"cycle": 0, "promotions": 0, "updates": 0, "examples_seen": 0,
                          "wall_seconds": 0., "candidate_seconds": 0., "development": development,
                          "best_retention": {},
                          "attempts": {s: 0 for s in SKILLS}, "last_practiced": {s: -1 for s in SKILLS},
                          "progress": {s: 0. for s in SKILLS}, "cost": {s: 1. for s in SKILLS},
                          "replay": {s: [] for s in SKILLS}, "history": [],
                          "strategy": {"learning_rate": self.config.learning_rate,
                                       "replay_fraction": self.config.replay_fraction},
                          "help_requests": 0, "local_tutor_responses": 0, "status": "ready"}
            self.save()
            torch.save(self.snapshot(), self.directory / "initial.pt")
            atomic_json(self.directory / "protocol.json", {
                "schema": SCHEMA, "config": asdict(self.config), "policy": POLICY,
                "curriculum_sha256": curriculum_digest(), "source_sha256": source_digest(),
                "evaluation": "rolling development and fixed retention development for selection; "
                              "audit split only by explicit audit command, never used by controller",
                "compute_budget": "per-invocation wall hours; safe boundaries can overrun one update/evaluation/save",
                "parallelism": "independent candidate optimizers; batched independent lesson episodes; no weight averaging",
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "platform": platform.platform()}})
            source_directory = self.directory / "source"
            source_directory.mkdir()
            for filename in ("learning_loop.py", "learning_student.py", "curriculum.py", "curriculum_tutor.py",
                             "model.py", "regions.py"):
                (source_directory / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
            claim.unlink()
        from .curriculum_tutor import LocalTutor, ProceduralTutor
        self.tutor = LocalTutor(model=self.config.tutor_model) if self.config.tutor_model else ProceduralTutor()

    def snapshot(self):
        return {"schema": SCHEMA, "config": asdict(self.config), "curriculum_sha256": self.curriculum_sha256,
                "source_sha256": self.source_sha256, "state": self.state,
                "weights": self.weights, "optimizer": self.optimizer, "exploration": self.exploration}

    def _assert_frozen(self):
        if source_digest() != self.source_sha256 or curriculum_digest() != self.curriculum_sha256:
            raise ValueError("source/curriculum changed during this experiment; saved provenance is preserved")

    def _assert_current(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.checkpoint_sha256:
            raise RuntimeError("checkpoint advanced since it was loaded; reload before writing or auditing")

    def save(self):
        temporary = self.path.with_suffix(".pt.tmp")
        torch.save(self.snapshot(), temporary)
        os.replace(temporary, self.path)
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        atomic_json(self.directory / "status.json", {
            "schema": SCHEMA, "cycle": self.state["cycle"], "status": self.state["status"],
            "promotions": self.state["promotions"], "updates": self.state["updates"],
            "examples_seen": self.state["examples_seen"], "wall_seconds": self.state["wall_seconds"],
            "candidate_seconds": self.state["candidate_seconds"],
            "help_requests": self.state["help_requests"],
            "local_tutor_responses": self.state["local_tutor_responses"],
            "development": self.state["development"], "strategy": self.state["strategy"],
            "replay_descriptors": sum(len(v) for v in self.state["replay"].values()),
            "recent_cycles": self.state["history"][-12:]})

    def model(self):
        model = build_student(self.config.seed, self.config.hidden_size, self.config.device)
        model.load_state_dict(self.weights)
        return model

    def _strategies(self):
        current = self.state["strategy"]
        alternatives = ((.7, .15), (1.25, -.1), (.5, .25))
        offset = self.state["cycle"] % len(alternatives)
        changes = ((1., 0.),) + alternatives[offset:] + alternatives[:offset]
        return [{"learning_rate": min(.02, max(self.config.learning_rate / 4, current["learning_rate"] * factor)),
                 "replay_fraction": min(.6, max(.15, current["replay_fraction"] + replay))}
                for factor, replay in changes[:self.config.candidates]]

    def cycle(self, deadline=None):
        self._assert_frozen()
        # Copy controller state until the complete candidate cohort is evaluated.
        # Failures and Ctrl+C must leave both weights and metadata uncommitted.
        config, state = self.config, copy.deepcopy(self.state)
        started = time.monotonic()
        topics = select_topics(state, config)
        number = state["cycle"]
        # A cycle consumes a nonoverlapping deterministic segment of the stream.
        seed = config.seed + number * 1_000_000
        lessons = [ex for i, skill in enumerate(topics)
                   for ex in generate(skill, seed + i * 100_000, config.lessons_per_skill, split="train")]
        # Current topics already have fresh practice. Rehearse other skills with
        # a mix of exact past episodes and fresh oracle-verified variants, avoiding
        # endless overfitting to a tiny stored reservoir.
        replay = []
        for index, skill in enumerate(SKILLS):
            if skill in topics or not state["replay"][skill]:
                continue
            descriptors = state["replay"][skill]
            replay.extend(generate(skill, entry, 1, split="train")[0] for entry in descriptors[::2])
            replay.extend(generate(skill, seed + 500_000 + index * 10_000,
                                   max(1, len(descriptors) // 2), split="train"))
        incumbent = self.model()
        observations, targets = encode_examples(lessons, config.device)
        with torch.inference_mode():
            probabilities = incumbent(observations)["logits"][:, -1].softmax(-1)
            losses = -probabilities.gather(1, targets[:, None]).clamp_min(1e-9).log().squeeze(1)
            difficult = int(losses.argmax())
            prediction = int(probabilities[difficult].argmax())
            confidence = float(probabilities[difficult].max())
        attempt = lessons[difficult]
        question = CURRICULUM[attempt["skill"]]["question"]
        tutor = {"source": "none", "status": "independent_practice", "question": question}
        if prediction != attempt["target"] or confidence < config.mastery_accuracy:
            state["help_requests"] += 1
            tutor = self.tutor.explain(attempt, question, deadline=deadline)
            if tutor["source"] == "local_ollama":
                state["local_tutor_responses"] += 1
            for example in tutor.get("practice_examples", []):
                validate_example(example)
                if example["split"] != "train" or example["skill"] != attempt["skill"]:
                    raise ValueError("tutor practice escaped requested train topic")
                lessons.append(example)
        if deadline is not None and time.monotonic() >= deadline:
            return None
        dev_seed = config.seed + 100_000_000 + number * 1_000_000
        before = evaluate_skills(incumbent, seed=dev_seed, count=config.dev_per_skill)
        strategies = self._strategies()
        def candidate(index):
            strategy = strategies[index]
            branch = self.exploration.get(index) if index else None
            if branch is not None and branch["age"] >= 24:
                branch = None
            trained = train_candidate(self.weights if branch is None else branch["weights"], lessons, seed=seed + index * 17,
                                      hidden_size=config.hidden_size, steps=config.steps,
                                      batch_size=config.batch_size, replay_examples=replay,
                                      device=config.device, deadline=deadline,
                                      optimizer_state=self.optimizer if branch is None else branch["optimizer"], **strategy)
            if trained["updates"] == 0:
                return {"index": index, "strategy": strategy, "trained": trained,
                        "development": before, "retention": None, "accepted": False,
                        "reason": "no_updates", "gain": 0., "branch_age": 0 if branch is None else branch["age"]}
            model = build_student(config.seed, config.hidden_size, config.device)
            model.load_state_dict(trained["state_dict"])
            development = evaluate_skills(model, seed=dev_seed, count=config.dev_per_skill)
            retention = evaluate_skills(model, seed=config.seed + 10_000_000, count=config.dev_per_skill)
            practiced = sorted(set(topics) | set(state["best_retention"]))
            accepted, reason, gain = promotion_decision(selection_metrics(before, practiced),
                                                       selection_metrics(development, practiced), retention,
                                                       state["best_retention"], config)
            if trained["updates"] == 0:
                accepted, reason = False, "no_updates"
            return {"index": index, "strategy": strategy, "trained": trained,
                    "development": development, "retention": retention,
                    "accepted": accepted, "reason": reason, "gain": gain,
                    "branch_age": 1 if branch is None else branch["age"] + 1}
        # A single GPU uses serial branches to avoid VRAM multiplication; batches
        # still learn from many independent environments simultaneously.
        workers = min(config.workers, config.candidates) if config.device == "cpu" else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(candidate, range(config.candidates)))
        if not any(r["trained"]["updates"] for r in results):
            return None
        eligible = [r for r in results if r["accepted"]]
        # Prefer measured benefit per optimizer update. This is deterministic
        # across scheduling/restarts; wall efficiency is separately reported.
        winner = max(eligible, key=lambda r: (r["gain"] / max(1, r["trained"]["updates"]), -r["index"]), default=None)
        if winner:
            state["strategy"] = winner["strategy"]
            state["development"] = winner["development"]
            state["promotions"] += 1
            for skill in SKILLS:
                measured = winner["retention"]["per_skill"][skill]["accuracy"]
                if (skill in state["best_retention"] or
                        (skill in topics and measured >= 1 / len(CURRICULUM[skill]["valid_targets"]) + .15)):
                    state["best_retention"][skill] = max(state["best_retention"].get(skill, 0.), measured)
        else:
            state["development"] = before
        elapsed = time.monotonic() - started
        for skill in topics:
            state["attempts"][skill] += 1
            state["last_practiced"][skill] = number
            gain = state["development"]["per_skill"][skill]["accuracy"] - before["per_skill"][skill]["accuracy"]
            state["progress"][skill] = .7 * state["progress"][skill] + .3 * gain
            updates = sum(r["trained"]["updates"] for r in results)
            state["cost"][skill] = .7 * state["cost"][skill] + .3 * updates / len(topics)
            # Only simulator-verified training descriptors enter rehearsal. Bounded
            # by skill: a frequently chosen topic cannot evict every older skill.
            entries = list(dict.fromkeys(state["replay"][skill] + [ex["seed"] for ex in lessons if ex["skill"] == skill]))
            state["replay"][skill] = entries[-config.replay_per_skill:]
        state["cycle"] += 1
        state["updates"] += sum(r["trained"]["updates"] for r in results)
        state["examples_seen"] += sum(r["trained"]["examples_seen"] for r in results)
        state["candidate_seconds"] += sum(r["trained"]["training_seconds"] for r in results)
        state["status"] = "curriculum_mastered" if all(
            x["accuracy"] >= config.mastery_accuracy for x in state["development"]["per_skill"].values()) else "learning"
        # Keep the learning ledger compact: practice is reproducible from seeds.
        tutor = {k: v for k, v in tutor.items() if k != "practice_examples"}
        event = {"cycle": number, "topics": topics, "question": question, "tutor": tutor,
                 "attempt": {"id": attempt["id"], "prediction": prediction, "target": attempt["target"],
                             "confidence": confidence, "correct": prediction == attempt["target"]},
                 "winner": None if winner is None else winner["index"], "seconds": elapsed,
                 "candidates": [{"index": r["index"], "strategy": r["strategy"], "accepted": r["accepted"],
                                 "reason": r["reason"], "gain": r["gain"],
                                 "accuracy": r["development"]["accuracy"], "updates": r["trained"]["updates"],
                                 "seconds": r["trained"]["training_seconds"]} for r in results]}
        state["history"] = (state["history"] + [event])[-128:]
        exploration = {r["index"]: {"weights": r["trained"]["state_dict"],
                                      "optimizer": r["trained"]["optimizer_state"], "age": r["branch_age"]}
                       for r in results if r["index"] and r["trained"]["updates"]}
        for result in results:
            index = result["index"]
            if index and not result["trained"]["updates"] and index in self.exploration:
                exploration[index] = self.exploration[index]
        if winner is not None and winner["index"] == 0:
            exploration = {}
        if winner:
            self.weights, self.optimizer = winner["trained"]["state_dict"], winner["trained"]["optimizer_state"]
        self.state = state
        self.exploration = exploration
        return event

    def run(self, hours, *, max_cycles=None, emit=True):
        if type(hours) not in (int, float) or not math.isfinite(hours) or hours <= 0:
            raise ValueError("hours must be finite and positive")
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise ValueError("max_cycles must be positive")
        start = last_save = time.monotonic()
        deadline = start + hours * 3600
        completed = 0
        with run_lock(self.directory):
            self._assert_current()
            self._assert_frozen()
            try:
                while time.monotonic() < deadline and (max_cycles is None or completed < max_cycles):
                    committed = self.weights, self.optimizer, self.state, self.exploration
                    try:
                        event = self.cycle(deadline)
                    except BaseException:
                        self.weights, self.optimizer, self.state, self.exploration = committed
                        raise
                    if event is None:
                        break
                    completed += 1
                    now = time.monotonic()
                    self.state["wall_seconds"] += now - last_save
                    self.save()
                    last_save = now
                    if emit:
                        print(json.dumps({"cycle": self.state["cycle"], "topics": event["topics"],
                                          "promoted": event["winner"] is not None,
                                          "development_accuracy": self.state["development"]["accuracy"],
                                          "status": self.state["status"]}), flush=True)
            except KeyboardInterrupt:
                # Cycle commits only after all candidates finish. Persisted latest
                # always points to the previous complete transaction if interrupted.
                self.state["status"] = "interrupted"
            finally:
                self.state["wall_seconds"] += time.monotonic() - last_save
                self.save()
        return self.state

    def audit(self, *, seed=800_000_001, count=256):
        if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
            raise ValueError("invalid audit seed/count")
        model = self.model()
        digest = model_digest(model)
        destination = self.directory / f"audit-{digest[:12]}-{seed}.json"
        with run_lock(self.directory):
            self._assert_current()
            self._assert_frozen()
            if destination.exists():
                raise FileExistsError("this model/seed was already audited; existing evidence is immutable")
            before = hashlib.sha256(self.path.read_bytes()).hexdigest()
            result = evaluate_skills(model, seed=seed, count=count, split="audit")
            initial = torch.load(self.directory / "initial.pt", map_location="cpu", weights_only=True)
            initial_model = build_student(self.config.seed, self.config.hidden_size, self.config.device)
            initial_model.load_state_dict(initial["weights"])
            baseline = evaluate_skills(initial_model, seed=seed, count=count, split="audit")
            report = {"schema": SCHEMA, "purpose": "final audit; never fed to curriculum or selection",
                      "seed": seed, "per_skill_count": count, "cycle": self.state["cycle"],
                      "model_sha256": digest, "checkpoint_sha256": before,
                      "source_sha256": source_digest(), "curriculum_sha256": curriculum_digest(),
                      "student": result, "untrained_baseline": baseline,
                      "accuracy_gain": result["accuracy"] - baseline["accuracy"],
                      "checkpoint_unchanged": before == hashlib.sha256(self.path.read_bytes()).hexdigest(),
                      "limitations": "symbolic sensors, engineered topic policy, supervised oracle labels, "
                                     "six known task rules; no open-ended language or biological perception"}
            atomic_json(destination, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="create a new local learning run")
    run.add_argument("--output", required=True)
    run.add_argument("--hours", type=float, required=True)
    run.add_argument("--seed", type=int, default=101)
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--candidates", type=int, default=2)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--tutor-model", help="optional installed local Ollama model; never downloaded")
    run.add_argument("--selection", choices=("progress", "round_robin"), default="progress")
    resume = sub.add_parser("resume", help="continue the frozen run for additional compute hours")
    resume.add_argument("--output", required=True)
    resume.add_argument("--hours", type=float, required=True)
    audit = sub.add_parser("audit", help="tutor-off immutable evaluation; does not guide learning")
    audit.add_argument("--output", required=True)
    audit.add_argument("--seed", type=int, default=800_000_001)
    audit.add_argument("--count", type=int, default=256)
    status = sub.add_parser("status", help="read progress without loading neural weights")
    status.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "status":
        print((Path(args.output) / "status.json").read_text(encoding="utf-8"))
        return 0
    if args.command == "run":
        config = LoopConfig(seed=args.seed, steps=args.steps, candidates=args.candidates,
                            workers=args.workers, device=args.device, tutor_model=args.tutor_model,
                            selection=args.selection)
        loop = LearningLoop(args.output, config)
    else:
        loop = LearningLoop(args.output, resume=True)
    if args.command == "audit":
        print(json.dumps(loop.audit(seed=args.seed, count=args.count), indent=2))
    else:
        loop.run(args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
