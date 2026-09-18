"""Explicit, externally pinned restart of an actually captured completed parent.

The caller's trusted export digest carries historical execution provenance.
Structural checks do not reproduce gradients or prove cached predictions. One
current CPU model template is built; no ancestor model or plan index is loaded.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

import torch

from brain_in_computer.learning_student import _check_finite_tree
from experiments import foundation_cycle_training as cycles
from experiments import foundation_loop as loops
from experiments import foundation_provider as providers
from experiments import foundation_variant_provider as variants
from experiments import foundation_variant_training as trainers
from experiments.foundation_curriculum import FAMILIES
from experiments.foundation_evidence import json_digest, transcript_set
from experiments.foundation_metrics import _canonical, _cell, _validate_row
from experiments.realization_banks import transcript_digest
from experiments.sequence_student import SequenceConfig, build_sequence_student


SCHEMA = "bic-foundation-parent-capsule-v1"
MAX_BYTES = 2 * 1024**3
MEMBERS = ("manifest.json", "sources.json", "identity.json", "metadata.json",
           "training-transcripts.json", "protected-transcripts.json", "learner.pt")
SCOPE = ("Caller-pinned export of a process-captured completed parent. Historical execution and "
         "admission provenance are inherited from that trusted pin, not independently replayed. "
         "No gradient replay, inference, promotion, source/runtime migration, or ancestor reconstruction.")


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate capsule JSON member")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite capsule JSON constant: " + value)

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if cycles._encoded(result) != raw:
        raise ValueError("canonical capsule JSON encoding required")
    return result


def _keys(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(label + " fields differ")


def _equal(a, b, label):
    if cycles._encoded(a) != cycles._encoded(b):
        raise ValueError(label + " differs")


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer within bounds")
    return value


def _encode(parts):
    """Container codec, not a completion proof or public token constructor."""
    if set(parts) != set(MEMBERS) - {"manifest.json"} or any(type(value) is not bytes for value in parts.values()):
        raise ValueError("exact immutable capsule parts required")
    manifest = dict(schema=SCHEMA, provenance="process-captured-completed-parent",
                    members_sha256={key: _digest(value) for key, value in parts.items()}, scope=SCOPE)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, cycles._encoded(manifest) if name == "manifest.json" else parts[name])
    return buffer.getvalue()


def _decode(raw, expected_sha256, max_bytes):
    if not _sha(expected_sha256):
        raise ValueError("external caller-pinned capsule SHA-256 required")
    _integer(max_bytes, "capsule byte limit", 1)
    if len(raw) > max_bytes or _digest(raw) != expected_sha256:
        raise ValueError("capsule differs from caller pin or byte limit")
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        entries = archive.infolist()
        if (len(entries) != len(MEMBERS) or [entry.filename for entry in entries] != list(MEMBERS)
                or any(entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1
                       or entry.is_dir() for entry in entries)
                or sum(entry.file_size for entry in entries) > max_bytes):
            raise ValueError("exact uncompressed capsule member set required")
        parts = {name: archive.read(name) for name in MEMBERS}
    manifest = _json(parts.pop("manifest.json"))
    _keys(manifest, ("schema", "provenance", "members_sha256", "scope"), "capsule manifest")
    if (manifest["schema"] != SCHEMA or manifest["provenance"] != "process-captured-completed-parent"
            or manifest["scope"] != SCOPE):
        raise ValueError("capsule schema or provenance declaration differs")
    _equal(manifest["members_sha256"], {key: _digest(value) for key, value in parts.items()}, "capsule member hashes")
    return parts


def _binding(parts, learner):
    sources, identity, metadata = (_json(parts[name]) for name in ("sources.json", "identity.json", "metadata.json"))
    training, protected = (_json(parts[name]) for name in ("training-transcripts.json", "protected-transcripts.json"))
    _equal(sources, cycles.source_hashes(), "capsule current source closure")
    _keys(identity, ("schema", "source_sha256", "provider_sha256", "loop_source_sha256", "loop_options",
        "loop_envelope_sha256", "parent_envelope", "plan_sha256", "index_identity", "base_updates", "cycle",
        "cycle_updates", "inherited_updates", "predecessor_identity_sha256", "predecessor_loop_envelope_sha256",
        "weights_sha256", "optimizer_sha256", "learner_metadata_sha256", "metadata_sha256",
        "excluded_transcripts_count", "excluded_transcripts_sha256"), "parent identity")
    _keys(metadata, ("provider_identity", "evaluation_specs", "final_evaluation", "retention_references",
        "reference_anchor", "trainer_recipe", "parent_work", "parent_timing", "device"), "parent metadata")
    if identity["schema"] != cycles.PARENT_SCHEMA:
        raise ValueError("current completed-parent schema required; no migration")
    _equal(identity["source_sha256"], sources, "parent source identity")
    _equal(identity["loop_source_sha256"], loops.source_hashes(), "parent loop sources")
    for name in ("provider_sha256", "loop_envelope_sha256", "plan_sha256", "weights_sha256", "optimizer_sha256",
                 "learner_metadata_sha256", "metadata_sha256", "excluded_transcripts_sha256"):
        if not _sha(identity[name]):
            raise ValueError("parent digest format differs: " + name)
    _keys(identity["parent_envelope"], ("kind", "file_sha256"), "parent envelope provenance")
    envelope = identity["parent_envelope"]
    if (envelope["kind"] not in ("memory", "durable")
            or (envelope["file_sha256"] is not None if envelope["kind"] == "memory" else not _sha(envelope["file_sha256"]))):
        raise ValueError("parent envelope provenance differs")
    for values in (training, protected):
        if type(values) is not list or values != sorted(transcript_set(values)):
            raise ValueError("sorted unique capsule transcript inventory required")
    if not set(training) <= set(protected):
        raise ValueError("parent training missing from lifetime exclusions")
    _equal(identity["excluded_transcripts_count"], len(protected), "excluded transcript count")
    _equal(identity["excluded_transcripts_sha256"], json_digest(protected), "excluded transcript digest")
    index = identity["index_identity"]
    if (type(index) is not dict or index.get("plan_sha256") != identity["plan_sha256"]
            or index.get("unique_transcripts") != len(training) or index.get("transcript_sha256") != json_digest(training)):
        raise ValueError("parent index/transcript identity differs")
    for name in ("base_updates", "cycle", "cycle_updates"):
        _integer(identity[name], name, 1)
    _integer(identity["inherited_updates"], "inherited_updates")
    if (identity["cycle_updates"] != index.get("bundle_count")
            or identity["base_updates"] != identity["inherited_updates"] + identity["cycle_updates"]):
        raise ValueError("completed parent local/lifetime/index counters differ")
    continued = identity["cycle"] > 1
    fields = {"schema", "recipe", "weights", "optimizer", "cursor", "evidence", "timing"}
    if continued:
        fields |= {"base_updates", "lifetime_updates", "cycle"}
    _keys(learner, fields, "completed learner")
    _check_finite_tree(learner, "capsule learner")
    if learner["schema"] != (cycles.SCHEMA if continued else trainers.SCHEMA):
        raise ValueError("learner generation/schema differs")
    recipe, provider = learner["recipe"], metadata["provider_identity"]
    _equal(metadata["trainer_recipe"], recipe, "parent trainer recipe")
    _equal(provider["trainer_recipe"], recipe, "provider trainer recipe")
    _equal(provider["evaluation_specs"], metadata["evaluation_specs"], "provider evaluation specs")
    _equal(provider["training_identity"], {"authenticated_plan_index": index}, "provider training index")
    _equal(recipe["plan_index_identity"], index, "learner index identity")
    if recipe["plan_sha256"] != identity["plan_sha256"] or provider["plan_sha256"] != identity["plan_sha256"]:
        raise ValueError("provider/learner plan identity differs")
    _equal(learner["cursor"], identity["cycle_updates"], "completed local cursor")
    if continued:
        from experiments.foundation_cycle_provider import SCHEMA as provider_schema, source_hashes as provider_sources
        for key, expected in (("base_updates", identity["inherited_updates"]),
                              ("lifetime_updates", identity["base_updates"]), ("cycle", identity["cycle"])):
            _equal(learner[key], expected, "learner " + key)
        predecessor = recipe["parent_identity"]
        if (recipe["base_updates"] != identity["inherited_updates"] or recipe["cycle"] != identity["cycle"]
                or predecessor["cycle"] + 1 != identity["cycle"]
                or predecessor["base_updates"] != identity["inherited_updates"]
                or json_digest(predecessor) != identity["predecessor_identity_sha256"]
                or predecessor["loop_envelope_sha256"] != identity["predecessor_loop_envelope_sha256"]):
            raise ValueError("compact predecessor lineage differs")
        _equal(provider["parent_identity"], predecessor, "provider predecessor")
        _equal(recipe["source_sha256"], sources, "cycle learner sources")
    else:
        provider_schema, provider_sources = variants.SCHEMA, variants.source_hashes
        if identity["inherited_updates"] != 0 or any(identity[key] is not None for key in (
                "predecessor_identity_sha256", "predecessor_loop_envelope_sha256")):
            raise ValueError("first generation cannot invent prior lineage")
        _equal(recipe["source_sha256"], trainers.source_hashes(), "variant learner sources")
    if provider["schema"] != provider_schema:
        raise ValueError("parent provider schema differs")
    _equal(provider["source_sha256"], provider_sources(), "parent provider sources")
    original_protection = sorted(set(protected) - set(training))
    for obj, prefix in ((provider, "trainer_protection"), (recipe, "protected_transcripts")):
        _equal(obj[prefix + "_count"], len(original_protection), "original protection count")
        _equal(obj[prefix + "_sha256"], json_digest(original_protection), "original protection hash")
    for key, expected in (("provider_sha256", json_digest(provider)), ("metadata_sha256", json_digest(metadata)),
                          ("weights_sha256", cycles._tensor_tree_digest(learner["weights"])),
                          ("optimizer_sha256", cycles._tensor_tree_digest(learner["optimizer"])),
                          ("learner_metadata_sha256", json_digest({k: v for k, v in learner.items() if k not in ("weights", "optimizer")}))):
        if identity[key] != expected:
            raise ValueError("capsule identity digest differs: " + key)
    return sources, identity, metadata, training, protected


def _validate_tensor_state(learner, model, optimizer, lifetime):
    """Validate only current tensors/moments; no forward or optimizer step."""
    _check_finite_tree(learner, "capsule learner")
    weights, expected = learner["weights"], model.state_dict()
    if type(weights) is not dict or set(weights) != set(expected):
        raise ValueError("capsule model parameter set differs")
    for name, tensor in weights.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype or tensor.device.type != "cpu" or tensor.requires_grad
                or tensor.layout != torch.strided):
            raise ValueError("capsule model shape/dtype/CPU/layout boundary differs")
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        first = aliases.setdefault(id(parameter), name)
        if not torch.equal(weights[name], weights[first]):
            raise ValueError("capsule tied weights disagree")
    state = learner["optimizer"]
    _keys(state, ("state", "param_groups"), "capsule optimizer")
    _equal(state["param_groups"], optimizer.state_dict()["param_groups"], "capsule AdamW configuration")
    parameters, moments = list(model.parameters()), state["state"]
    if (type(moments) is not dict or any(type(key) is not int for key in moments)
            or set(moments) != set(range(len(parameters)))):
        raise ValueError("capsule AdamW parameter coverage differs")
    for index, values in moments.items():
        _keys(values, ("step", "exp_avg", "exp_avg_sq"), "capsule AdamW moments")
        step = values["step"]
        if (not isinstance(step, torch.Tensor) or step.numel() != 1 or not step.is_floating_point()
                or step.device.type != "cpu" or step.requires_grad or float(step) != lifetime):
            raise ValueError("capsule AdamW step differs from lifetime")
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                    or value.dtype != parameters[index].dtype or value.device.type != "cpu"
                    or value.requires_grad or value.layout != torch.strided):
                raise ValueError("capsule AdamW moment shape/dtype/CPU/layout differs")
        if bool(values["exp_avg_sq"].lt(0).any()):
            raise ValueError("capsule AdamW second moment must be nonnegative")


def _template(learner, identity):
    recipe = learner["recipe"]
    architecture = recipe["architecture"]
    if (type(architecture) is not dict or set(architecture) != {"name", "schema"}
            or architecture.get("name") not in trainers.ARCHITECTURES
            or architecture["schema"] != trainers.ARCHITECTURES[architecture["name"]]):
        raise ValueError("capsule architecture contract differs")
    if (recipe["optimizer"].get("name") != "AdamW" or recipe["gradient_clip"] != 1.
            or recipe["objective"] != "mean of three unchanged sequence_objective microbatches"
            or recipe["family_order"] != list(FAMILIES)):
        raise ValueError("capsule optimizer/objective contract differs")
    config = SequenceConfig(**recipe["config"])
    builder = build_sequence_student
    if architecture["name"] == "hierarchical":
        from experiments.hierarchical_sequence_student import build_hierarchical_sequence_student
        builder = build_hierarchical_sequence_student
    model = builder(recipe["seed"], device="cpu", config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])
    _equal(recipe["optimizer"]["param_groups"], optimizer.state_dict()["param_groups"], "recipe AdamW groups")
    _validate_tensor_state(learner, model, optimizer, identity["base_updates"])
    return config


def _reference_not_worse(best, observed, field):
    """Count arithmetic for an invariant, not a replacement retention policy."""
    if observed["rate"] is None:
        return True
    if best["rate"] is None:
        return False
    numerator = "count" if field == "unsupported_ask" else "correct"
    left, right = best[numerator] * observed["total"], observed[numerator] * best["total"]
    return left <= right if field == "unsupported_ask" else left >= right


def _evaluations(metadata, identity, learner, banks, protected, config):
    if type(banks) is not dict or set(banks) != set(providers.ROLES):
        raise ValueError("exact caller development/retention banks required")
    canonical, specs = {}, {}
    protection = set(protected)
    for role in providers.ROLES:
        if type(banks[role]) is not dict or not banks[role]:
            raise ValueError("nonempty caller evaluation banks required")
        specs[role] = {}
        for name, rows in sorted(banks[role].items()):
            bank, counts, opposite = _canonical(rows, name, config, "dev")
            if any(transcript_digest(row) not in protection for row in rows):
                raise ValueError("evaluation transcript omitted from exclusions")
            _, family, depth, mechanism, turns = _cell(name)
            specs[role][name] = dict(identity=bank, cell=dict(family=family, depth=int(depth[1:]),
                                                            mechanism=mechanism, turns=int(turns)))
            canonical[role + ":" + name] = (bank, counts, opposite)
        if {value["cell"]["family"] for value in specs[role].values()} != set(FAMILIES):
            raise ValueError("each role must retain all declared families")
    _equal(specs, metadata["evaluation_specs"], "caller evaluation bank identities")
    producer = loops._producer({"learner": learner})
    final = metadata["final_evaluation"]
    _keys(final, ("index", "producer", "responses", "alarms"), "final evaluation")
    _equal(final["producer"], producer, "final actual learner producer")
    options = identity["loop_options"]
    _keys(options, ("window_updates", "chunk_updates", "history_limit"), "parent loop options")
    for name, value in options.items():
        _integer(value, name, 1)
        if value > 4096:
            raise ValueError("loop option exceeds source contract")
    if options["chunk_updates"] > options["window_updates"]:
        raise ValueError("practice chunk exceeds prescribed window")
    evaluations = math.ceil(identity["cycle_updates"] / options["window_updates"]) + 1
    _equal(final["index"], evaluations - 1, "final prescribed evaluation index")
    if type(final["responses"]) is not dict or set(final["responses"]) != set(canonical):
        raise ValueError("complete final bank response set required")

    def response(value, key, expected_producer):
        role, name = key.split(":", 1)
        _keys(value, ("schema", "provider_sha256", "role", "weights_sha256", "updates", "control",
                       "per_bank", "progress", "wall_seconds", "automatic_promotion"), "evaluation response")
        if (value["schema"] != providers.EVALUATION_SCHEMA or value["provider_sha256"] != identity["provider_sha256"]
                or value["role"] != role or value["control"] != "normal" or value["automatic_promotion"] is not False
                or value["updates"] != expected_producer["updates"] or value["weights_sha256"] != expected_producer["weights_sha256"]
                or set(value["per_bank"]) != {name} or set(value["progress"]) != {name}):
            raise ValueError("evaluation source/bank/producer contract differs")
        providers._finite(value["wall_seconds"], "evaluation wall")
        _validate_row(value["per_bank"][name], *canonical[key], "normal")
        _equal(value["progress"][name], providers._progress(value["per_bank"][name]), "canonical evaluation progress")

    for key, value in final["responses"].items():
        response(value, key, producer)
    boundaries = {min(i * options["window_updates"], learner["cursor"]) for i in range(evaluations)}
    known_producers = {learner["cursor"]: producer["weights_sha256"]}
    for kind in ("retention_references", "reference_anchor"):
        references = metadata[kind]
        dropped = max(0, evaluations - options["history_limit"])
        allowed = set(canonical) if kind == "retention_references" or dropped else set()
        allowed_boundaries = boundaries if kind == "retention_references" else {
            min(i * options["window_updates"], learner["cursor"]) for i in range(dropped)}
        if type(references) is not dict or set(references) != allowed:
            raise ValueError("retention reference coverage differs")
        for key, fields in references.items():
            _keys(fields, loops.FIELDS, "retention fields")
            for ref in fields.values():
                _keys(ref, ("producer", "response"), "retention reference")
                p = ref["producer"]
                _keys(p, ("updates", "weights_sha256"), "retention producer")
                if type(p["updates"]) is not int or p["updates"] not in allowed_boundaries or not _sha(p["weights_sha256"]):
                    raise ValueError("retention producer boundary differs")
                previous = known_producers.setdefault(p["updates"], p["weights_sha256"])
                if previous != p["weights_sha256"]:
                    raise ValueError("retention weights conflict at one producing boundary")
                response(ref["response"], key, p)
    final_alarms = []
    for key in sorted(canonical):
        role, name = key.split(":", 1)
        for field in loops.FIELDS:
            reference = metadata["retention_references"][key][field]
            best = reference["response"]["progress"][name][field]
            observed = [final["responses"][key]["progress"][name][field]]
            if key in metadata["reference_anchor"]:
                observed.append(metadata["reference_anchor"][key][field]["response"]["progress"][name][field])
            if any(not _reference_not_worse(best, value, field) for value in observed):
                raise ValueError("retained best reference is worse than final or dropped-history anchor")
            current_rate, best_rate = observed[0]["rate"], best["rate"]
            if current_rate is not None and best_rate is not None and (
                    current_rate > best_rate if field == "unsupported_ask" else current_rate < best_rate):
                final_alarms.append(dict(role=role, bank=name, metric=field, observed=current_rate,
                                         reference=best_rate, reference_producer=reference["producer"]))
    _equal(final["alarms"], final_alarms, "final retention alarms")
    work = metadata["parent_work"]
    _keys(work, (*loops.WORK_FIELDS, "evaluation_banks", "evaluation_episode_exposures", "training_seconds", "evaluation_seconds"), "parent work")
    evidence = learner["evidence"]
    _equal(evidence["cursor"], learner["cursor"], "local evidence cursor")
    exposures = sum(value["episodes"] for value in evidence["exposures"].values())
    for name in loops.WORK_FIELDS:
        _equal(work[name], learner["cursor"] if name == "physical_optimizer_updates" else exposures, "parent committed " + name)
    _equal(work["evaluation_banks"], evaluations * len(canonical), "completed scoring count")
    _equal(work["evaluation_episode_exposures"], evaluations * sum(value[0]["episodes"] for value in canonical.values()), "scored episode count")
    for name in ("training_seconds", "evaluation_seconds"):
        providers._finite(work[name], name)
    _equal(metadata["parent_timing"], learner["timing"], "parent timing")
    _keys(learner["timing"], ("retained_step_seconds", "materialization_seconds_included_in_step"), "learner timing")
    for name, value in learner["timing"].items():
        providers._finite(value, name)
    if learner["timing"]["materialization_seconds_included_in_step"] > learner["timing"]["retained_step_seconds"]:
        raise ValueError("materialization exceeds retained step time")


def export_parent_capsule(parent, path):
    """Export only an actual token; exclusive file creation returns its trust pin."""
    started, cpu = time.monotonic(), time.process_time()
    if type(parent) is not cycles.CompletedParent:
        raise ValueError("exact process-captured CompletedParent required for export")
    identity = parent.identity
    parts = {"sources.json": parent._sources, "identity.json": parent._identity,
             "metadata.json": parent._metadata, "learner.pt": parent._learner_image,
             "training-transcripts.json": cycles._encoded(parent.training_transcripts),
             "protected-transcripts.json": cycles._encoded(parent.protected_transcripts)}
    _binding(parts, parent._learner())
    raw = _encode(parts)
    if len(raw) > MAX_BYTES:
        raise ValueError("capsule exceeds supported byte limit")
    _equal(parent.identity, identity, "parent identity during export")
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sha = _digest(raw)
    if _digest(destination.read_bytes()) != sha:
        raise RuntimeError("capsule changed during publication")
    parent._check()
    return dict(schema=SCHEMA, status="exported", path=str(destination), sha256=sha,
                parent_identity_sha256=json_digest(identity), bytes=len(raw), wall_seconds=time.monotonic()-started,
                cpu_seconds=time.process_time()-cpu, neural_training_or_inference=False, scope=SCOPE)


def load_parent_capsule(path, *, expected_sha256, device="cpu", evaluation_banks, max_bytes=MAX_BYTES):
    """Load the exact trusted export, without replaying any ancestor curriculum.

    Timing is available as ``load_parent_capsule.last_report`` for this call.
    It is observational and is never a trust input to a subsequent load.
    """
    started, cpu = time.monotonic(), time.process_time()
    load_parent_capsule.last_report = None
    _integer(max_bytes, "capsule byte limit", 1)
    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    parts = _decode(raw, expected_sha256, max_bytes)
    sources = _json(parts["sources.json"])
    _equal(sources, cycles.source_hashes(), "capsule current source closure")
    metadata = _json(parts["metadata.json"])
    runtime = providers.runtime_identity(torch.device(device))
    _equal(metadata["provider_identity"]["runtime"], runtime, "capsule runtime (no migration)")
    if str(torch.device(metadata["device"])) != str(torch.device(device)):
        raise ValueError("capsule device differs; migration is not implemented")
    learner = torch.load(io.BytesIO(parts["learner.pt"]), map_location="cpu", weights_only=True)
    sources, identity, metadata, training, protected = _binding(parts, learner)
    tick = time.monotonic()
    config = _template(learner, identity)
    template_seconds = time.monotonic()-tick
    _evaluations(metadata, identity, learner, evaluation_banks, protected, config)
    _equal(sources, cycles.source_hashes(), "sources during capsule load")
    _equal(runtime, providers.runtime_identity(torch.device(device)), "runtime during capsule load")
    parent = object.__new__(cycles.CompletedParent)
    for name, value in (("_sources", parts["sources.json"]), ("_identity", parts["identity.json"]),
                        ("_metadata", parts["metadata.json"]), ("_learner_image", parts["learner.pt"]),
                        ("_training_transcripts", tuple(training)), ("_protected_transcripts", tuple(protected))):
        object.__setattr__(parent, name, value)
    parent._check()
    load_parent_capsule.last_report = dict(schema=SCHEMA, status="loaded", sha256=expected_sha256,
        parent_identity_sha256=json_digest(identity), bytes=len(raw), cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-started, current_cpu_template_seconds=template_seconds,
        current_model_templates=1, ancestor_models_loaded=0, plan_indexes_constructed=0,
        neural_training_or_inference=False, scope=SCOPE)
    return parent
