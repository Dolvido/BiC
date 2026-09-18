"""Fixed historical diagnostic inputs; no optimizer restore or lesson replay.

The seven exclusion inventories and each selected checkpoint are decoded only
from their single, freshly authenticated immutable byte images. A prior full
archive receipt supplies historical verification provenance; required source and
JSON bytes are reauthenticated here. No 424-file revalidation is claimed.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time

import torch

from brain_in_computer.dialogue_student import checkpoint_digest
from experiments.foundation_historical_archive import _json_bytes, _physical, _relative_name
from experiments.sequence_student import SequenceConfig, build_sequence_student
from experiments.foundation_training import source_hashes as trainer_sources


SCHEMA = "bic-foundation-diagnostic-inputs-v1"
HISTORICAL = "runs/foundation-study-local-v2"
FORMAL = "runs/variant-study-local/data"
SUMMARY_SHA256 = "188b0d026c4debce596f3cca5a1b64f6f5f0e0e3992dd4174904a33b6c3e87c2"
MANIFEST_SHA256 = "d0170189ffa373c68492972df83ed7e39aaa1c40f2cc5c9520b89454f0672be6"
GATE_SHA256 = "6f9a85c2e91f1bd43f13d009ffd3c70108241acaccfdff921a3c598d31e98f10"
ARCHIVE_RECEIPT = "runs/foundation-historical-archive-validation-local/attempt-20260917T063531153774Z/historical-archive-receipt.json"
ARCHIVE_RECEIPT_SHA256 = "b0281a317da505fde1cc924ae4b4ae24713752edd8da84594f3f747b2797bc66"
SOLVER_SHA256 = "bce32f11cc29073610d5422a29d55864cbb15cd2a7c847aa5fabbf18ffa6eead"
CONFIG = dict(width=192, layers=4, heads=4, feedforward=768, max_positions=1024,
              max_turns=12, max_input_bytes=128, max_output_bytes=32)
PROTOCOL_SHA256 = "af2f8c722e85e87e90291ea4f52f65b8e1959a2d88824f89cfd88ff539c8fd8d"
STATES = {
    "initial": dict(arm="curriculum", updates=0,
        file_sha256="053e1f3519be0f46fca90b58b141d810217c45fae0e455320d77af73a7c6b804",
        weights_sha256="49f332666a350146422e4e9827674a26cca86109b7f5ed81dc038764314c7489"),
    "curriculum": dict(arm="curriculum", updates=1536,
        file_sha256="d80d7c4c1bca74514440deaef6568ea5d831248624c6ad4cf44f1dcb432be61c",
        weights_sha256="455ca00c33d02d7ac83bbbde1a8fb0394ebadb7f7be28616cd95369fc1e763cb"),
    "mixed": dict(arm="mixed", updates=1536,
        file_sha256="8dcb38be7e587460730a4025a5de487523888565cc4ff9f3ff6c80e4241f8980",
        weights_sha256="569f65cb9c48eaa61207220ccfe37c5660e5be3c7a32bba405555cbf64ff4486")}
ADDED_SOURCES = {
    "experiments/foundation_historical_archive.py": "b3827b0129b600611f3ed76b12711d859d684b6a941dce63762a77f0071de0e4",
    "experiments/foundation_diagnostic_features.py": "a97665daab85248f3764fef79f7ce5b66212e29e46d45ce7d61c0bb2e08aa905",
    "experiments/foundation_representation_probe.py": "9eb6bb46a321395e41bfe8d36a85a9a3807a4b202fb6cd53bdd24b303cdf399c",
    "experiments/foundation_diagnostic_data.py": "04e1547d7049d05e8ac8aff8d640e6d608c9d14692e47b0daaaad497010936eb"}
_SELF = Path(__file__).resolve()
_IMPORTED_SHA256 = hashlib.sha256(_SELF.read_bytes()).hexdigest()
_KEY = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _root(repo):
    value = _physical(Path(repo).absolute(), directory=True)
    _require(value == _SELF.parents[1], "diagnostic inputs must use their imported repository root")
    return value


def _read_bound(root, name, expected_sha256, *, maximum=512 * 1024 * 1024):
    _relative_name(name)
    _require(type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "independently bound lowercase SHA256 required")
    path = _physical(root / name)
    _require(path.is_relative_to(root) and path.stat().st_size <= maximum, "input exceeds its path/size boundary")
    with path.open("rb") as stream:
        image = stream.read(maximum + 1)
    _require(len(image) <= maximum and hashlib.sha256(image).hexdigest() == expected_sha256,
             "authenticated input byte mismatch: " + name)
    return image


def _runtime():
    _require(str(torch.__version__) == "2.11.0+cu128"
        and torch.version.git_version == "70d99e998b4955e0049d13a98d77ae1b14db1f45"
        and platform.python_version() == "3.12.14", "declared diagnostic Python/PyTorch runtime differs")
    settings = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision())
    _require(settings == dict(threads=1, interop_threads=1, deterministic_algorithms=True,
        deterministic_warn_only=False, cudnn_deterministic=True, cudnn_benchmark=False,
        cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, float32_matmul_precision="highest"),
        "caller must configure the declared deterministic diagnostic runtime")
    solver = _physical(Path(torch.__file__).resolve().parent / "optim/lbfgs.py")
    _require(hashlib.sha256(solver.read_bytes()).hexdigest() == SOLVER_SHA256, "installed LBFGS implementation differs")
    module = sys.modules.get(torch.optim.LBFGS.__module__)
    _require(module is not None and Path(module.__file__).resolve() == solver, "LBFGS import resolution differs")
    return dict(torch_version=str(torch.__version__), torch_git_version=torch.version.git_version,
        python_version=platform.python_version(), platform=platform.platform(), cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), settings=settings,
        solver_path=str(solver), solver_sha256=SOLVER_SHA256)


def _imports(root, names):
    checked = {}
    for package in ("experiments", "brain_in_computer"):
        module = sys.modules.get(package)
        _require(module is not None and {str(Path(p).resolve()) for p in module.__path__} == {str(root / package)},
                 "repository package import path differs: " + package)
    _require(not (root / "experiments/__init__.py").exists(), "namespace package initializer appeared")
    for name in names:
        if not name.endswith(".py"):
            continue
        path = root / name
        module_name = name[:-3].replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[:-9]
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            _require(getattr(loaded, "__file__", None) and Path(loaded.__file__).resolve() == path,
                     "loaded module comes from another source: " + module_name)
            origin = getattr(getattr(loaded, "__spec__", None), "origin", None)
            _require(origin is not None and Path(origin).resolve() == path, "loaded module spec differs")
        else:
            spec = importlib.machinery.PathFinder.find_spec(module_name, [str(path.parent)])
            _require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == path,
                     "module import resolution differs: " + module_name)
        checked[module_name] = str(path)
    return checked


def _guard(root, sources, runtime=None):
    expected_brain = {name for name in sources if name.startswith("brain_in_computer/") and name.count("/") == 1}
    _require({path.relative_to(root).as_posix() for path in (root / "brain_in_computer").glob("*.py")} == expected_brain,
             "historical nonrecursive source membership changed")
    for name, expected in sources.items():
        _read_bound(root, name, expected, maximum=4 * 1024 * 1024)
    resolved = _imports(root, sources)
    current_runtime = _runtime()
    if runtime is not None:
        _require(current_runtime == runtime, "runtime or solver changed after authentication")
    return resolved, current_runtime


def _decode_inventory(image, *, semantic_sha256=None, count=None):
    values = torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
    _require(type(values) is list and values and all(type(value) is str
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in values), "canonical transcript list required")
    _require(all(left < right for left, right in zip(values, values[1:])), "sorted unique transcript list required")
    if count is not None:
        _require(len(values) == count, "transcript inventory count differs")
    digest = _hash(values)
    if semantic_sha256 is not None:
        _require(digest == semantic_sha256, "transcript inventory semantic digest differs")
    return values, digest


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AuthenticatedInputs:
    _repo: Path
    _metadata: bytes
    _exclusions: frozenset
    _reports: list
    _attempted: set
    _lock: object
    _pid: int

    def __init__(self, key, repo, metadata, exclusions):
        _require(key is _KEY and type(self) is AuthenticatedInputs, "use authenticate_inputs")
        for name, value in dict(_repo=repo, _metadata=_encoded(metadata), _exclusions=frozenset(exclusions),
                _reports=[], _attempted=set(), _lock=threading.RLock(), _pid=os.getpid()).items():
            object.__setattr__(self, name, value)

    @property
    def metadata(self):
        return json.loads(self._metadata)

    @property
    def exclusions(self):
        return self._exclusions

    @property
    def load_reports(self):
        with self._lock:
            return copy.deepcopy(self._reports)

    def __reduce__(self):
        raise TypeError("authenticated inputs are process owned; reauthenticate after restart")


def authenticate_inputs(repo):
    """Freshly authenticate required metadata/source bytes and seven inventories."""
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(schema=SCHEMA, operation="authenticate", status="incomplete", inventory_loads_started=0,
                  inventory_loads_completed=0, inventory_bytes=0, model_constructions=0, checkpoint_loads=0)
    try:
        root = _root(repo)
        files = []
        def read_json(name, digest):
            raw = _read_bound(root, name, digest, maximum=40 * 1024 * 1024)
            files.append(dict(path=name, sha256=digest, bytes=len(raw), kind="json"))
            return _json_bytes(raw)
        summary = read_json(HISTORICAL + "/evaluation/summary.json", SUMMARY_SHA256)
        _require(summary["schema"] == "bic-foundation-pilot-summary-v1" and summary["status"] == "completed_descriptive",
                 "historical summary is not the declared completed study")
        inputs, historical_sources = summary["integrity"]["input_file_sha256"], summary["integrity"]["source_sha256"]
        _require(len(inputs) == 424 and len(historical_sources) == 79, "historical input/source count differs")
        def old_json(name):
            relative = HISTORICAL + "/" + name
            return read_json(relative, inputs[str(root / relative)])
        protocol = old_json("protocol.json")
        verification = old_json("verification.json")
        plan = old_json("plan.json")
        receipts = {arm: old_json(arm + "/receipt.json") for arm in ("curriculum", "mixed")}
        _require(inputs[str(root / HISTORICAL / "protocol.json")] == PROTOCOL_SHA256
            and protocol["source_sha256"] == historical_sources and protocol["config"] == CONFIG
            and protocol["contract"] == summary["contract"] and protocol["contract"]["model_seed"] == 6101
            and protocol["contract"]["updates"] == 1536, "historical protocol/configuration differs")
        _require(verification["schema"] == "bic-foundation-results-v1" and verification["status"] == "completed"
            and verification["protocol_sha256"] == PROTOCOL_SHA256 and verification["source_sha256"] == historical_sources,
                 "historical verification record differs")
        for arm, receipt in receipts.items():
            _require(receipt["schema"] == "bic-foundation-order-pilot-v2" and receipt["arm"] == arm
                and receipt["status"] == "completed" and receipt["protocol_sha256"] == PROTOCOL_SHA256
                and receipt["retained_updates"] == 1536 and receipt["physical_work_unknown"] is False
                and verification["arms"][arm]["exact_official_checkpoint_restores"] is True,
                "historical completion receipt differs")
        prior = read_json(ARCHIVE_RECEIPT, ARCHIVE_RECEIPT_SHA256)
        _require(prior["status"] == "verified" and prior["summary_sha256"] == SUMMARY_SHA256
            and prior["input_file_sha256"] == inputs and prior["source_sha256"] == historical_sources,
            "prior full archive receipt does not bind the pinned summary")
        for name, digest in historical_sources.items():
            snapshot = HISTORICAL + "/source/" + name
            _require(inputs[str(root / snapshot)] == digest == inputs[str(root / name)], "historical source bindings differ")
            _read_bound(root, snapshot, digest, maximum=4 * 1024 * 1024)
        gate = read_json("runs/foundation-planning-local/variant-launch-gate.json", GATE_SHA256)
        _require(gate["ready"] is True and len(gate["source_sha256"]) == 88, "formal source gate differs")
        sources = dict(historical_sources)
        for mapping in (gate["source_sha256"], ADDED_SOURCES,
                        {"experiments/foundation_diagnostic_inputs.py": _IMPORTED_SHA256}):
            for name, digest in mapping.items():
                _require(name not in sources or sources[name] == digest, "source contracts disagree")
                sources[name] = digest
        resolutions, runtime = _guard(root, sources)
        formal = read_json(FORMAL + "/manifest.json", MANIFEST_SHA256)
        _require(formal["schema"] == "bic-foundation-variant-data-v1" and formal["automatic_promotion"] is False
            and formal["neural_training_or_inference"] is False and set(formal["stages"]) == {"calibration", "main"},
            "formal data manifest schema differs")
        _require(all(sources.get(name) == digest for name, digest in formal["source_sha256"].items()),
                 "formal data source map differs")
        states = {}
        for state_name, selected in STATES.items():
            arm, step = selected["arm"], selected["updates"]
            relative = HISTORICAL + f"/{arm}/checkpoint-{step:06d}.pt"
            _require(inputs[str(root / relative)] == selected["file_sha256"]
                == receipts[arm]["checkpoints"][Path(relative).name]
                == verification["input_file_sha256"][f"{arm}/checkpoint-{step:06d}.pt"], "selected checkpoint pins disagree")
            _require(verification["arms"][arm]["weights_sha256"][str(step)] == selected["weights_sha256"],
                     "selected weight record differs")
            rows = [row for row in summary["arms"][arm]["development_curve"] if row["updates"] == step]
            _require(len(rows) == 1 and rows[0]["weights_sha256"] == selected["weights_sha256"], "summary weight record differs")
            states[state_name] = dict(selected, path=relative, config=CONFIG, seed=6101, protocol_sha256=PROTOCOL_SHA256)
        initial = STATES["initial"]["weights_sha256"]
        _require(protocol["initial_weights_sha256"] == initial
            and all(verification["arms"][arm]["weights_sha256"]["0"] == initial for arm in ("curriculum", "mixed")),
            "recorded common initialization differs")
        specs = [(HISTORICAL + "/protected.pt", inputs[str(root / HISTORICAL / "protected.pt")], None, protocol["protected_count"]),
                 (HISTORICAL + "/training-transcripts.pt", inputs[str(root / HISTORICAL / "training-transcripts.pt")], None, protocol["training_count"]),
                 (FORMAL + "/history.pt", formal["artifacts_sha256"]["history.pt"], formal["history"]["sha256"], formal["history"]["count"])]
        for stage in ("calibration", "main"):
            for filename, field in (("training-transcripts.pt", "training_transcripts"), ("protected.pt", "protected_transcripts")):
                relative = stage + "/" + filename
                specs.append((FORMAL + "/" + relative, formal["artifacts_sha256"][relative], formal["stages"][stage][field], None))
        union, inventories = set(), []
        old_union = set()
        for index, (name, digest, semantic, count) in enumerate(specs):
            image = _read_bound(root, name, digest)
            report["inventory_bytes"] += len(image)
            report["inventory_loads_started"] += 1
            values, semantic_digest = _decode_inventory(image, semantic_sha256=semantic, count=count)
            report["inventory_loads_completed"] += 1
            record = dict(path=name, sha256=digest, bytes=len(image), count=len(values), semantic_sha256=semantic_digest)
            inventories.append(record)
            if index < 2:
                old_union.update(values)
            if index == 2:
                _require(set(values) == old_union, "formal history is not the authenticated historical union")
            union.update(values)
            del image, values
        recipe_sources = trainer_sources()
        _require(all(historical_sources.get(name) == digest for name, digest in recipe_sources.items()),
                 "trainer recipe source closure differs from the authenticated archive")
        _guard(root, sources, runtime)
        report.update(status="completed", wall_seconds=time.monotonic() - started,
                      cpu_seconds=time.process_time() - cpu_started)
        metadata = dict(schema=SCHEMA, repository_root=str(root), source_sha256=sources, import_resolutions=resolutions,
            runtime=runtime, states=states, inventory_files=inventories, dataset_input_files=files + inventories,
            exclusion_union=dict(count=len(union), sha256=_hash(sorted(union))),
            historical=dict(summary_sha256=SUMMARY_SHA256, protocol_sha256=PROTOCOL_SHA256,
                prior_full_archive_receipt_sha256=ARCHIVE_RECEIPT_SHA256, prior_full_logical_inputs=424,
                prior_full_physical_files=345, fresh_scope="Required JSON, all79 archived/live historical sources, formal88 and diagnostic sources, seven inventory images. Selected checkpoint images checked only at load_state; no full424 revalidation or historical execution replay.",
                execution_profile=protocol["execution_profile"], plan_sha256=_hash(plan),
                protected_sha256=inventories[0]["semantic_sha256"], protected_count=inventories[0]["count"],
                trainer_source_sha256=recipe_sources), authentication_report=report)
        return AuthenticatedInputs(_KEY, root, metadata, union)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=repr(error), wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
        error.input_report = copy.deepcopy(report)
        raise


def _load_weights(model, weights, expected_digest):
    template = model.state_dict()
    _require(type(weights) is dict and set(weights) == set(template), "exact learner weight keys required")
    for name, target in template.items():
        value = weights[name]
        _require(type(value) is torch.Tensor and value.device.type == "cpu" and value.layout == torch.strided
            and value.dtype == target.dtype and value.shape == target.shape and not value.requires_grad
            and bool(torch.isfinite(value).all()), "finite CPU weight shape/dtype differs: " + name)
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        if len(names) > 1:
            reference = weights[names[0]].contiguous().numpy().tobytes()
            _require(all(weights[name].contiguous().numpy().tobytes() == reference for name in names),
                     "tied parameter copies disagree")
    model.load_state_dict(weights, strict=True)
    _require(model.tokens.weight is model.observation_head.weight, "loaded tied byte embedding alias differs")
    actual = checkpoint_digest(model)
    _require(actual == expected_digest, "loaded learner weight digest differs")
    return actual


def _envelope(saved, selected, metadata):
    _require(type(saved) is dict and set(saved) == {"schema", "arm", "protocol_sha256", "learner", "weights_sha256", "execution_profile"},
             "exact historical checkpoint envelope required")
    _require(saved["schema"] == "bic-foundation-order-pilot-v2" and saved["arm"] == selected["arm"]
        and saved["protocol_sha256"] == PROTOCOL_SHA256 and saved["weights_sha256"] == selected["weights_sha256"]
        and saved["execution_profile"] == metadata["historical"]["execution_profile"], "checkpoint identity differs")
    learner = saved["learner"]
    _require(type(learner) is dict and set(learner) == {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}
        and learner["schema"] == "bic-foundation-trainer-v1" and type(learner["cursor"]) is int
        and learner["cursor"] == selected["updates"], "historical learner schema/cursor differs")
    recipe = learner["recipe"]
    fields = {"schema", "curriculum_version", "source_sha256", "plan_sha256", "order", "seed", "config",
        "protected_transcripts_sha256", "protected_transcripts_count", "micro_batch_size", "family_order",
        "learning_rate", "optimizer", "objective", "gradient_clip"}
    _require(type(recipe) is dict and set(recipe) == fields, "historical recipe fields differ")
    wanted = dict(schema="bic-foundation-trainer-v1", curriculum_version="bic-shared-foundation-v1",
        source_sha256=metadata["historical"]["trainer_source_sha256"], plan_sha256=metadata["historical"]["plan_sha256"],
        order=selected["arm"], seed=6101, config=CONFIG,
        protected_transcripts_sha256=metadata["historical"]["protected_sha256"],
        protected_transcripts_count=metadata["historical"]["protected_count"], micro_batch_size=32,
        family_order=["color", "count", "switch"], learning_rate=.001,
        objective="mean of three unchanged sequence_objective microbatches", gradient_clip=1.)
    _require(_encoded({key: recipe[key] for key in wanted}) == _encoded(wanted), "historical recipe identity differs")
    _require(type(recipe["optimizer"]) is dict and set(recipe["optimizer"]) == {"name", "param_groups"}
        and recipe["optimizer"]["name"] == "AdamW", "recorded optimizer identity differs")
    _require(type(learner["evidence"]) is dict and type(learner["evidence"].get("cursor")) is int
        and learner["evidence"]["cursor"] == selected["updates"],
             "recorded evidence cursor differs")
    return learner["weights"]


def load_state(repo, state_name, authenticated):
    """Load each fixed state at most once per authentication; return an exact CPU model."""
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(schema=SCHEMA, operation="load_state", state_name=state_name, status="incomplete",
        checkpoint_byte_images=0, checkpoint_bytes=0, checkpoint_loads_started=0, checkpoint_loads_completed=0,
        model_constructions_started=0, model_constructions_completed=0, neural_forwards=0,
        optimizer_restores=0, optimizer_updates=0, historical_replay=False)
    owner = None
    try:
        root = _root(repo)
        _require(type(authenticated) is AuthenticatedInputs and authenticated._pid == os.getpid()
            and authenticated._repo == root, "same-process authenticated inputs required")
        owner = authenticated
        with owner._lock:
            _require(type(state_name) is str and state_name in STATES, "initial/curriculum/mixed state name required")
            _require(state_name not in owner._attempted, "this state was already attempted; no automatic reload")
            owner._attempted.add(state_name)
            metadata = owner.metadata
            _guard(root, metadata["source_sha256"], metadata["runtime"])
            selected = metadata["states"][state_name]
            report.update(path=selected["path"], file_sha256=selected["file_sha256"], expected_weights_sha256=selected["weights_sha256"])
            image = _read_bound(root, selected["path"], selected["file_sha256"], maximum=64 * 1024 * 1024)
            report.update(checkpoint_byte_images=1, checkpoint_bytes=len(image), checkpoint_loads_started=1)
            saved = torch.load(io.BytesIO(image), map_location="cpu", weights_only=True)
            report["checkpoint_loads_completed"] = 1
            weights = _envelope(saved, selected, metadata)
            report["model_constructions_started"] = 1
            model = build_sequence_student(6101, device="cpu", config=SequenceConfig(**CONFIG))
            report["model_constructions_completed"] = 1
            actual = _load_weights(model, weights, selected["weights_sha256"])
            model.eval()
            _guard(root, metadata["source_sha256"], metadata["runtime"])
            report.update(status="completed", loaded_weights_sha256=actual, configuration=asdict(model.config), device="cpu",
                          wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
            owner._reports.append(copy.deepcopy(report))
            return model
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=repr(error), wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
        if owner is not None:
            with owner._lock:
                owner._reports.append(copy.deepcopy(report))
        error.input_report = copy.deepcopy(report)
        raise
