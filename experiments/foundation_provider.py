"""In-memory prescribed foundation practice; the outer controller owns disk.

This adapter never chooses or recycles lessons. Historical admission protection
and the larger trainer/evaluation protection are separate bound identities.
No files are written, and no existing study checkpoint is adopted implicitly.
"""
from __future__ import annotations

import copy
import hashlib
import math
import platform
import threading
import time
from pathlib import Path

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments import foundation_plan as planning
from experiments.foundation_admission import authenticate_admission
from experiments.foundation_curriculum import FAMILIES, VERSION
from experiments.foundation_evaluation import FoundationBank
from experiments.foundation_evidence import json_digest, prepare_training, transcript_set
from experiments.foundation_metrics import _canonical, _cell, _validate_row
from experiments.foundation_training import FoundationTrainer, source_hashes as trainer_sources
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig
from experiments.train_cognitive import _check_finite_tree


SCHEMA = "bic-foundation-practice-provider-v1"
EVALUATION_SCHEMA = "bic-foundation-provider-evaluation-v1"
ROLES = ("development", "retention")
DEFAULT_CHUNK_UPDATES = 16
MAX_CHUNK_UPDATES = 4096


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("foundation_provider", "foundation_admission", "foundation_evidence",
             "foundation_evaluation", "foundation_metrics", "execution_profile")
    return {**trainer_sources(), **{f"experiments/{name}.py": hashlib.sha256(
        (root / "experiments" / f"{name}.py").read_bytes()).hexdigest() for name in names}}


def runtime_identity(device):
    device = torch.device(device)
    if device.type == "cuda":
        from experiments.execution_profile import runtime_profile
        return runtime_profile(str(device))  # Caller must opt in before CUDA initialization.
    if device.type != "cpu":
        raise ValueError("foundation provider supports explicit CPU or strict CUDA")
    return {"device": "cpu", "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision()}


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError(name + " must be an integer within bounds")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")


def _same(left, right):
    return json_digest(left) == json_digest(right)


def _progress(row):
    known = row["known_total"]
    unsupported = row["ask_predicted"] - row["ask_correct"]
    return {"paired_action": {"correct": row["final_pairs"]["correct"],
                "total": row["final_pairs"]["total"], "rate": row["final_pair_accuracy"]},
        "paired_reply": {"correct": row["final_reply_pair_correct"],
                "total": row["episodes"]//2, "rate": row["final_reply_pair_accuracy"]},
        "known": {"correct": row["known_correct"], "total": known, "rate": row["known_accuracy"]},
        "unsupported_ask": {"count": unsupported, "total": known,
                            "rate": unsupported/known if known else None}}


class FoundationPracticeProvider:
    def __init__(self, plan, order="curriculum", *, seed, evaluation_banks,
                 admission_protected_transcripts, protected_transcripts, admission_receipt,
                 config=None, learning_rate=.001, device="cpu", payload=None):
        started = time.monotonic()
        self._lock = threading.RLock()
        self._sources = source_hashes()
        self._device = torch.device(device)
        self._runtime = runtime_identity(self._device)
        self._plan, self._receipt = copy.deepcopy(plan), copy.deepcopy(admission_receipt)
        history = transcript_set(admission_protected_transcripts)
        protection = transcript_set(protected_transcripts)
        if not history <= protection:
            raise ValueError("original admission protection must be a subset of trainer protection")
        base = planning.build_plan(**self._plan["config"])
        authenticate_admission(base, self._plan, history, receipt=self._receipt)
        self._protected = tuple(sorted(protection))
        if config is not None and type(config) is not SequenceConfig:
            raise ValueError("explicit SequenceConfig required")
        self._config = config or SequenceConfig(max_turns=12)
        self._order, self._seed, self._learning_rate = order, seed, learning_rate
        if type(evaluation_banks) is not dict or set(evaluation_banks) != set(ROLES):
            raise ValueError("exact development and retention bank roles required; no audits")
        self._banks, self._canonical_metrics, self._specs = {}, {}, {}
        for role in ROLES:
            rows_by_name = evaluation_banks[role]
            if type(rows_by_name) is not dict or not rows_by_name:
                raise ValueError("each evaluation role requires nonempty named banks")
            self._banks[role], self._canonical_metrics[role], self._specs[role] = {}, {}, {}
            for name in sorted(rows_by_name):
                rows = copy.deepcopy(rows_by_name[name])
                identity, counts, opposite = _canonical(rows, name, self._config, "dev")
                if any(transcript_digest(row) not in protection for row in rows):
                    raise ValueError("all evaluation transcripts must be in trainer protection")
                bank = FoundationBank(rows, role="dev", config=self._config)
                if not _same(identity, bank.identity):
                    raise ValueError("foundation preparation and metric identities differ")
                _, family, depth, mechanism, turns = _cell(name)
                self._banks[role][name] = bank
                self._canonical_metrics[role][name] = (identity, counts, opposite)
                self._specs[role][name] = {"identity": identity,
                    "cell": {"family": family, "depth": int(depth[1:]),
                             "mechanism": mechanism, "turns": int(turns)}}
            if {value["cell"]["family"] for value in self._specs[role].values()} != set(FAMILIES):
                raise ValueError("each evaluation role must cover all three declared families")
        # The historical admission receipt does not establish exclusion from
        # evaluation banks reserved later. Authenticate the ENTIRE finite plan
        # against the expanded protection before constructing any learner.
        manifest, transcripts = prepare_training(self._plan,
            protected_transcripts=self._protected, anchor_limit=1)
        training_identity = {"manifest_sha256": json_digest(manifest),
            "transcripts_sha256": manifest["transcript_sha256"],
            "unique_transcripts": len(transcripts), "totals": manifest["totals"]}
        self._trainer = self._new_trainer()
        self._identity = {"schema": SCHEMA, "curriculum_version": VERSION,
            "source_sha256": self._sources, "runtime": self._runtime,
            "plan_sha256": json_digest(self._plan), "order": order,
            "admission_receipt_sha256": json_digest(self._receipt),
            "admission_protection_sha256": json_digest(sorted(history)),
            "admission_protection_count": len(history),
            "trainer_protection_sha256": json_digest(self._protected),
            "trainer_protection_count": len(protection), "evaluation_specs": self._specs,
            "training_identity": training_identity,
            "trainer_recipe": self._trainer.recipe,
            "default_chunk_updates": DEFAULT_CHUNK_UPDATES, "maximum_chunk_updates": MAX_CHUNK_UPDATES}
        self._identity_hash = json_digest(self._identity)
        self._pending, self._failed, self.last_report = None, False, None
        if payload is not None:
            self.restore(payload)
        self.setup_seconds = time.monotonic()-started

    def _new_trainer(self, payload=None):
        return FoundationTrainer(self._plan, self._order, seed=self._seed, config=self._config,
            learning_rate=self._learning_rate, device=self._device,
            protected_transcripts=self._protected, payload=payload)

    def _check(self, *, allow_failed=False):
        if self._failed and not allow_failed:
            raise RuntimeError("failed provider requires explicit transactional restoration")
        if source_hashes() != self._sources or not _same(runtime_identity(self._device), self._runtime):
            raise RuntimeError("foundation provider source or runtime identity changed")

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    @property
    def identity_sha256(self):
        return self._identity_hash

    def specs(self):
        return copy.deepcopy(self._specs)

    def status(self):
        with self._lock:
            self._check()
            saved = self._trainer.snapshot()
            cursor = saved["cursor"]
            cells = {}
            for bundle_id in self._plan["schedules"][self._order][:cursor]:
                bundle = self._plan["bundles"][str(bundle_id)]
                for family in FAMILIES:
                    key = f"{family}/d{bundle['depth']}/t{bundle['turns']}"
                    cells[key] = cells.get(key, 0)+1
            return {"curriculum_version": VERSION, "provider_sha256": self._identity_hash,
                "updates": cursor, "cursor": cursor,
                "total_updates": len(self._plan["bundles"]),
                "remaining_updates": len(self._plan["bundles"])-cursor,
                "exhausted": cursor == len(self._plan["bundles"]),
                "family_depth_turn_microbatches": cells,
                "pending": copy.deepcopy(self._pending),
                **{key: copy.deepcopy(saved["evidence"][key]) for key in
                   ("family_microbatches", "bucket_microbatches", "depth_microbatches", "exposures")}}

    def _request(self, request, *, cursor=None):
        fields = {"kind", "provider_sha256", "plan_sha256", "order", "start_cursor", "stop_cursor"}
        if (type(request) is not dict or set(request) != fields
                or request["kind"] != "prescribed_prefix"
                or request["provider_sha256"] != self._identity_hash
                or request["plan_sha256"] != self._identity["plan_sha256"]
                or request["order"] != self._order):
            raise ValueError("prescribed prefix identity or fields differ")
        start, stop = request["start_cursor"], request["stop_cursor"]
        _integer(start, "prefix start", maximum=len(self._plan["bundles"]))
        _integer(stop, "prefix stop", minimum=start+1, maximum=len(self._plan["bundles"]))
        if stop-start > MAX_CHUNK_UPDATES or cursor is not None and start != cursor:
            raise ValueError("prefix is stale or exceeds its finite budget")
        return copy.deepcopy(request)

    def _validate_pending(self, pending, cursor):
        _integer(cursor, "learner cursor", maximum=len(self._plan["bundles"]))
        if pending is None:
            return
        if type(pending) is not dict or set(pending) != {"request", "request_sha256", "consumed_updates"}:
            raise ValueError("pending request fields differ")
        request = self._request(pending["request"])
        _integer(pending["consumed_updates"], "consumed prefix updates")
        if (pending["request_sha256"] != json_digest(request)
                or not request["start_cursor"] <= cursor < request["stop_cursor"]
                or pending["consumed_updates"] != cursor-request["start_cursor"]):
            raise ValueError("pending request disagrees with learner cursor")

    def snapshot(self):
        with self._lock:
            self._check()
            learner = self._trainer.snapshot()
            self._validate_pending(self._pending, learner["cursor"])
            return {"schema": SCHEMA, "identity": self.identity, "learner": learner,
                    "pending": copy.deepcopy(self._pending)}

    def restore(self, payload):
        with self._lock:
            self._check(allow_failed=True)
            if (type(payload) is not dict or set(payload) != {"schema", "identity", "learner", "pending"}
                    or payload["schema"] != SCHEMA or not _same(payload["identity"], self._identity)
                    or type(payload["learner"]) is not dict or "cursor" not in payload["learner"]):
                raise ValueError("provider checkpoint identity or fields differ")
            _check_finite_tree(payload, "provider checkpoint")
            self._validate_pending(payload["pending"], payload["learner"]["cursor"])
            candidate = self._new_trainer(copy.deepcopy(payload["learner"]))
            pending = copy.deepcopy(payload["pending"])
            self._failed = True
            self._trainer, self._pending = candidate, pending
            self.last_report = None
            self._failed = False
            return self

    def validate_evaluation(self, response, *, expected_weights_sha256=None, expected_updates=None):
        """Validate current evidence or an explicitly bound historical producer.

        Historical expectations must come from the controller's authenticated
        producing snapshot, never be copied from the response under validation.
        Numeric consistency does not independently reproduce model predictions.
        """
        with self._lock:
            self._check()
            if (expected_weights_sha256 is None) != (expected_updates is None):
                raise ValueError("historical evidence requires both producing weights and updates")
            if expected_weights_sha256 is None:
                expected_weights_sha256 = checkpoint_digest(self._trainer.model)
                expected_updates = self._trainer.cursor
            if (type(expected_weights_sha256) is not str or len(expected_weights_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in expected_weights_sha256)):
                raise ValueError("expected producing weight digest differs")
            _integer(expected_updates, "expected producing updates", maximum=len(self._plan["bundles"]))
            fields = {"schema", "provider_sha256", "role", "weights_sha256", "updates", "control",
                      "per_bank", "progress", "wall_seconds", "automatic_promotion"}
            if (type(response) is not dict or set(response) != fields
                    or response["schema"] != EVALUATION_SCHEMA
                    or response["provider_sha256"] != self._identity_hash
                    or response["role"] not in ROLES or response["control"] not in ("normal", "blank", "reset")
                    or response["automatic_promotion"] is not False):
                raise ValueError("foundation evaluation response identity differs")
            _integer(response["updates"], "evaluation updates")
            _finite(response["wall_seconds"], "evaluation wall")
            if (response["updates"] != expected_updates
                    or response["weights_sha256"] != expected_weights_sha256):
                raise ValueError("evaluation belongs to another learner state")
            role, per_bank = response["role"], response["per_bank"]
            if (type(per_bank) is not dict or not per_bank or not set(per_bank) <= set(self._banks[role])
                    or type(response["progress"]) is not dict or set(response["progress"]) != set(per_bank)):
                raise ValueError("evaluation names differ from prepared banks")
            try:
                json_digest(response)
                for name, row in per_bank.items():
                    _validate_row(row, *self._canonical_metrics[role][name], response["control"])
                    if not _same(response["progress"][name], _progress(row)):
                        raise ValueError("evaluation progress differs from canonical counts")
            except (KeyError, TypeError, AttributeError, OverflowError) as error:
                raise ValueError("malformed foundation evaluation") from error
            return True

    def evaluate(self, role, *, names=None, batch_size=32, control="normal"):
        started = time.monotonic()
        with self._lock:
            self._check()
            if role not in ROLES or control not in ("normal", "blank", "reset"):
                raise ValueError("development/retention role and explicit control required")
            _integer(batch_size, "evaluation batch size", minimum=1)
            if names is not None and type(names) not in (list, tuple):
                raise ValueError("bank names must be a finite list or tuple")
            chosen = sorted(self._banks[role]) if names is None else list(names)
            if (not chosen or any(type(name) is not str for name in chosen)
                    or len(set(chosen)) != len(chosen) or not set(chosen) <= set(self._banks[role])):
                raise ValueError("evaluation requires distinct prepared bank names")
            weights = checkpoint_digest(self._trainer.model)
            cursor = self._trainer.cursor
            scoring_failed = False
            try:
                per_bank = {name: self._banks[role][name].score(self._trainer.model,
                    batch_size=batch_size, score_replies=True, control=control) for name in chosen}
            except BaseException:
                scoring_failed = True
                raise
            finally:
                # Scorers can fail after partially executing. Preserve their
                # original exception, but never permit changed/uncertain state
                # to proceed merely because the normal post-score path skipped.
                try:
                    changed = (checkpoint_digest(self._trainer.model) != weights
                               or self._trainer.cursor != cursor)
                except BaseException:
                    self._failed = True
                    if not scoring_failed:
                        raise
                else:
                    if changed:
                        self._failed = True
                        if not scoring_failed:
                            raise RuntimeError("evaluation unexpectedly changed the learner")
            result = {"schema": EVALUATION_SCHEMA, "provider_sha256": self._identity_hash,
                "role": role, "weights_sha256": weights, "updates": cursor, "control": control,
                "per_bank": per_bank, "progress": {name: _progress(row) for name, row in per_bank.items()},
                "wall_seconds": time.monotonic()-started, "automatic_promotion": False}
            self.validate_evaluation(result)
            result["wall_seconds"] = time.monotonic()-started
            return result

    def practice(self, request=None, *, max_updates=DEFAULT_CHUNK_UPDATES, deadline=None):
        started = time.monotonic()
        with self._lock:
            report = {"status": "preflight", "starting_updates": self._trainer.cursor,
                "completed_updates": 0, "physical_optimizer_updates": 0,
                "unknown_optimizer_attempts": 0, "failed_step_attempts": 0,
                "drawn_episode_exposures": 0, "neural_attempted_episode_exposures": 0,
                "completed_microbatch_episode_exposures": 0, "step_reports": [],
                "accounting_uncertain": False,
                "in_memory_retained_updates": 0, "durable_retained_updates": None,
                "wall_seconds": 0., "step_seconds": 0., "materialization_seconds": 0.,
                "scope": "In-memory provider work only; outer controller owns durable commits and rollback. Step/materialization are included subsets of wall time. Failed or uncertain work is never inferred from cursor alone."}
            self.last_report = report
            attempted, in_flight_accounting = False, False
            try:
                self._check()
                _integer(max_updates, "chunk update limit", minimum=1, maximum=MAX_CHUNK_UPDATES)
                if deadline is not None:
                    _finite(deadline, "monotonic deadline")
                if request is not None:
                    if self._pending is not None:
                        raise ValueError("a pending prefix cannot be replaced; resume with request=None")
                    request = self._request(request, cursor=self._trainer.cursor)
                    self._pending = {"request": request, "request_sha256": json_digest(request),
                                     "consumed_updates": 0}
                if self._pending is None:
                    if self._trainer.cursor == len(self._plan["bundles"]):
                        report["status"] = "exhausted"
                        return report
                    raise ValueError("an explicit prescribed prefix is required")
                self._validate_pending(self._pending, self._trainer.cursor)
                report["status"] = "running"
                while self._pending is not None and report["completed_updates"] < max_updates:
                    if deadline is not None and time.monotonic() >= deadline:
                        report["status"] = "deadline"
                        break
                    attempted = True
                    in_flight_accounting = True
                    self._failed = True
                    self._trainer.last_report = None
                    try:
                        step = self._trainer.step()
                    except BaseException:
                        report["failed_step_attempts"] += 1
                        step = self._trainer.last_report
                        if step is not None:
                            self._account(report, step)
                        else:
                            report["unknown_optimizer_attempts"] += 1
                            report["physical_optimizer_updates"] = None
                            for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures",
                                        "completed_microbatch_episode_exposures"):
                                report[key] = None
                        in_flight_accounting = False
                        raise
                    self._account(report, step)
                    in_flight_accounting = False
                    report["completed_updates"] += 1
                    self._pending["consumed_updates"] += 1
                    if self._trainer.cursor == self._pending["request"]["stop_cursor"]:
                        self._pending = None
                    self._failed = False
                if report["status"] == "running":
                    report["status"] = ("exhausted" if self._trainer.cursor == len(self._plan["bundles"])
                        else "prefix_complete" if self._pending is None else "update_bound")
                return report
            except BaseException as error:
                report.update(status="failed", error=repr(error))
                if attempted:
                    self._failed = True
                if in_flight_accounting:
                    # A step may have finished before report construction was
                    # interrupted. Never present a partial sum as known work.
                    report["accounting_uncertain"] = True
                    for key in ("physical_optimizer_updates", "drawn_episode_exposures",
                                "neural_attempted_episode_exposures", "completed_microbatch_episode_exposures",
                                "step_seconds", "materialization_seconds"):
                        report[key] = None
                raise
            finally:
                report["wall_seconds"] = time.monotonic()-started
                report["ending_updates"] = None if self._failed else self._trainer.cursor
                report["in_memory_retained_updates"] = None if self._failed else report["completed_updates"]
                report["restore_required"] = self._failed
                self.last_report = copy.deepcopy(report)

    @staticmethod
    def _account(report, step):
        report["step_reports"].append(copy.deepcopy(step))
        physical = step["physical_optimizer_updates"]
        if physical is None:
            report["unknown_optimizer_attempts"] += 1
            report["physical_optimizer_updates"] = None
        elif report["physical_optimizer_updates"] is not None:
            report["physical_optimizer_updates"] += physical
        for key in ("drawn_episode_exposures", "neural_attempted_episode_exposures",
                    "completed_microbatch_episode_exposures"):
            if step[key] is None:
                report[key] = None
            elif report[key] is not None:
                report[key] += step[key]
        report["step_seconds"] += step["step_seconds"]
        report["materialization_seconds"] += step["materialization_seconds"]
