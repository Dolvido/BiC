"""Read-only local geometry of four unchanged foundation objective components.

The caller supplies authenticated loaded models and fitting banks, pins sources,
and owns the endpoint schedule, durable work journal and wall allowance. No file
reads, checkpoint restore, optimizer, clipping, resampling or parameter update.
This is unpreconditioned preclip geometry, not an AdamW update or causal ablation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import time

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments.cognitive_credit import balanced_query_loss
from experiments.composition_data import pack_composition_episodes
from experiments.foundation_curriculum import FAMILIES, validate_pair
from experiments.foundation_diagnostic_data import CELLS
from experiments.foundation_diagnostic_features import _encoded, _model, _tensor_sha
from experiments.foundation_metrics import _canonical


SCHEMA = "bic-foundation-gradient-diagnostic-v1"
WEIGHTS = dict(query=1., acknowledgement=.25, reply=.1, observation=.1)
COMPONENTS = tuple(WEIGHTS)


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _batch_sha(batch):
    return _sha({group: {name: _tensor_sha(value) for name, value in sorted(values.items())}
                 for group, values in sorted(batch.items())})


def _components(output, batch):
    """Exact original class and turn-index reductions, before external weights."""
    inputs, supervision = batch["inputs"], batch["supervision"]
    labels, logits = supervision["action_targets"], output["logits"]
    query = balanced_query_loss(logits.flatten(0, 1), labels.flatten())
    acknowledgements = labels.eq(3)
    acknowledgement = (F.cross_entropy(logits[acknowledgements], labels[acknowledgements])
                       if bool(acknowledgements.any()) else logits.sum() * 0.)
    targets = supervision["reply_targets"]
    reply_tokens = F.cross_entropy(output["language_logits"].flatten(0, 2), targets.flatten(),
        ignore_index=ByteCodec.PAD, reduction="none").reshape_as(targets)
    reply_counts = targets.ne(ByteCodec.PAD).sum(dim=(0, 2))
    reply = (reply_tokens.sum(dim=(0, 2)) / reply_counts.clamp_min(1)).mean()
    observation_targets = supervision["observation_next_byte_targets"]
    token_loss = F.cross_entropy(output["observation_language_logits"].transpose(1, 2),
        observation_targets, ignore_index=ByteCodec.PAD, reduction="none")
    ends = inputs["eos_positions"]
    begins = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1] + 1), dim=1)
    positions = torch.arange(token_loss.shape[1], device=token_loss.device)[None, :]
    observations, observation_counts = [], []
    for turn in range(ends.shape[1]):
        mask = ((positions >= begins[:, turn:turn + 1]) & (positions < ends[:, turn:turn + 1])
                & observation_targets.ne(ByteCodec.PAD))
        count = mask.sum()
        observations.append((token_loss * mask).sum() / count.clamp_min(1))
        observation_counts.append(int(count))
    observation = torch.stack(observations).mean()
    values = dict(query=query, acknowledgement=acknowledgement, reply=reply, observation=observation)
    if any(not bool(torch.isfinite(value)) for value in values.values()):
        raise FloatingPointError("nonfinite gradient diagnostic objective")
    denominators = dict(query_class_counts={str(target): int(labels.eq(target).sum()) for target in range(3)},
        acknowledgement_count=int(acknowledgements.sum()), reply_tokens_per_turn=reply_counts.tolist(),
        observation_tokens_per_turn=observation_counts,
        reduction="Query: mean present-class CE; ACK: its CE. Reply/observation: mean turn-index pooled-token means within each family. Then mean the three families.")
    return values, denominators


def _partitions(model):
    if model.tokens.weight is not model.observation_head.weight:
        raise ValueError("the original tied token/observation parameter is required")
    parameters, names, partitions, seen = [], [], {}, set()
    for group, entries in (("shared_blocks", model.blocks.named_parameters(prefix="blocks", remove_duplicate=False)),
                           ("tied_token_embedding", (("tokens.weight", model.tokens.weight),))):
        indices = []
        for name, parameter in entries:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            indices.append(len(parameters))
            parameters.append(parameter)
            names.append(name)
        if not indices:
            raise ValueError("nonempty disjoint parameter partitions required")
        partitions[group] = indices
    identity = {group: dict(parameter_names=[names[index] for index in indices],
        parameter_count=len(indices), scalar_count=sum(parameters[index].numel() for index in indices))
        for group, indices in partitions.items()}
    identity["tied_token_embedding"]["aliases"] = ["tokens.weight", "observation_head.weight"]
    return parameters, partitions, identity


def _geometry(gradients, partitions):
    """CPU float64 reductions of detached, already family-mean gradients."""
    result = {}
    for partition, indices in partitions.items():
        vectors = {name: torch.cat([gradients[name][index].detach().reshape(-1).to(device="cpu", dtype=torch.float64)
                                   for index in indices]) for name in COMPONENTS}
        if any(not bool(torch.isfinite(value).all()) for value in vectors.values()):
            raise FloatingPointError("nonfinite component gradient")
        norms = {name: float(torch.linalg.vector_norm(value)) for name, value in vectors.items()}
        if any(not math.isfinite(value) for value in norms.values()):
            raise FloatingPointError("nonfinite gradient norm")

        def cosine(left, right, left_norm, right_norm):
            if left_norm == 0. or right_norm == 0.:
                return None
            value = float(torch.dot(left, right) / (left_norm * right_norm))
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite gradient cosine")
            return max(-1., min(1., value))

        cosines = {left + "/" + right: cosine(vectors[left], vectors[right], norms[left], norms[right])
                   for left, right in itertools.combinations(COMPONENTS, 2)}
        auxiliary = sum(WEIGHTS[name] * vectors[name] for name in COMPONENTS if name != "query")
        auxiliary_norm = float(torch.linalg.vector_norm(auxiliary))
        total_norm = float(torch.linalg.vector_norm(vectors["query"] + auxiliary))
        if not math.isfinite(auxiliary_norm) or not math.isfinite(total_norm):
            raise FloatingPointError("nonfinite weighted gradient sum")
        result[partition] = dict(unweighted_norms=norms,
            weighted_norms={name: WEIGHTS[name] * norms[name] for name in COMPONENTS},
            component_cosines=cosines, weighted_auxiliary_norm=auxiliary_norm,
            query_weighted_auxiliary_cosine=cosine(vectors["query"], auxiliary, norms["query"], auxiliary_norm),
            weighted_total_norm=total_norm)
    return result


def _protected(model, selected, report, operation):
    """Temporarily enable selected partial derivatives; restore all learner state."""
    modes = [(module, module.training) for module in model.modules()]
    parameters = list(model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
    grad_devices = [None if p.grad is None else p.grad.device for p in parameters]
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    layout = {name: (value.device, value.dtype, tuple(value.shape)) for name, value in model.state_dict().items()}
    config, digest = model.config, checkpoint_digest(model)
    selected_ids = set(map(id, selected))
    handles, error, traceback, result = [], None, None, None

    def decoder_count(suffix, arguments):
        embedded = arguments[0]
        report["teacher_forced_decoder_calls_" + suffix] += 1
        report["teacher_forced_decoder_row_steps_" + suffix] += embedded.shape[0] * embedded.shape[1]

    try:
        handles.append(model.inferior_frontal.recurrent.register_forward_pre_hook(
            lambda module, arguments: decoder_count("attempted", arguments)))
        handles.append(model.inferior_frontal.recurrent.register_forward_hook(
            lambda module, arguments, output: decoder_count("completed", arguments)))
        for parameter in parameters:
            parameter.requires_grad_(id(parameter) in selected_ids)
        model.train()  # Historical training mode; this architecture has zero dropout.
        with torch.enable_grad():
            result = operation()
    except BaseException as caught:
        error, traceback = caught, caught.__traceback__
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in modes:
            module.training = mode
        report["modes_restored"] = True
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        report["requires_grad_flags_restored"] = True
        try:
            if next(model.parameters()).device.type == "cuda":
                torch.cuda.synchronize(next(model.parameters()).device)
            report["device_synchronized"] = True
            placement_intact = (list(map(id, model.parameters())) == list(map(id, parameters))
                and {name: (value.device, value.dtype, tuple(value.shape))
                     for name, value in model.state_dict().items()} == layout
                and model.tokens.weight is model.observation_head.weight)
            if not placement_intact:
                report["model_state_unchanged"] = False
                raise RuntimeError("learner identity, placement or alias changed; restoration is not claimed")
            same_grads = all((p.grad is None and old is None) or
                (p.grad is not None and old is not None and p.grad.device == device and _tensor_sha(p.grad) == _tensor_sha(old))
                for p, old, device in zip(parameters, grads, grad_devices))
            report["weights_sha256_after"] = checkpoint_digest(model)
            report["model_state_unchanged"] = (report["weights_sha256_after"] == digest and same_grads and model.config == config)
            if not report["model_state_unchanged"]:
                with torch.no_grad():
                    model.load_state_dict(state, strict=True)
                    for parameter, old in zip(parameters, grads):
                        parameter.grad = None if old is None else old.to(parameter.device).clone()
                    model.config = config
                report["model_state_restored"] = True
                raise RuntimeError("diagnostic operation changed learner state; original values restored")
        except BaseException as cleanup_error:
            if error is None:
                error, traceback = cleanup_error, cleanup_error.__traceback__
            else:
                report["cleanup_error"] = type(cleanup_error).__name__ + ": " + str(cleanup_error)
    if error is not None:
        raise error.with_traceback(traceback)
    return result


def inspect_gradients(model, fit_banks, *, endpoint, cell_inventory=CELLS, pairs_per_cell=4):
    """Inspect one supplied endpoint; production uses 21 groups and 504 episodes.

    A smaller explicit matched inventory/pair count is only a synthetic fixture.
    Return plain detached metadata; on failure preserve the original exception
    type and completed/partial work in its ``gradient_report`` attribute.
    """
    started, cpu_started = time.monotonic(), time.process_time()
    counters = ("groups", "family_forwards", "encoder_episodes", "encoder_actual_positions",
        "encoder_padded_positions", "teacher_forced_decoder_calls", "teacher_forced_decoder_row_steps",
        "component_gradient_evaluations")
    report = dict(schema=SCHEMA, status="incomplete", endpoint=endpoint, component_weights=dict(WEIGHTS),
        **{name + "_" + suffix: 0 for name in counters for suffix in ("attempted", "completed")},
        groups=[], optimizer_calls=0, optimizer_updates=0, clipping_calls=0, checkpoint_reads=0,
        model_state_unchanged=None, model_state_restored=False, modes_restored=False,
        requires_grad_flags_restored=False, device_synchronized=False, input_banks_unchanged=None,
        gradient_scope="Unweighted three-family-mean components; selected blocks and tied token parameter only; no .grad accumulation. CPU float64 geometry; positive weights leave pairwise cosines unchanged.",
        batch_shape="Three separate family batches per matched group; no cross-family padding or class pooling.",
        completion_scope="Attempted counts precede calls; completed counts mean the call returned. Finite checks and final device synchronization can still fail.")
    original_banks = None
    try:
        if torch.is_inference_mode_enabled() or torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("gradient inspection requires ordinary tensors without inference mode or autocast")
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise ValueError("strict deterministic algorithms must be configured by the caller")
        device = _model(model)
        if endpoint not in ("curriculum", "mixed"):
            raise ValueError("explicit curriculum or mixed endpoint required")
        if type(pairs_per_cell) is not int or not 1 <= pairs_per_cell <= 4:
            raise ValueError("one to four first pair slots required")
        if (type(cell_inventory) not in (tuple, list) or not cell_inventory
                or any(type(cell) is not str or cell not in CELLS for cell in cell_inventory)
                or len(set(cell_inventory)) != len(cell_inventory) or type(fit_banks) is not dict
                or set(fit_banks) != set(cell_inventory)):
            raise ValueError("explicit unique canonical fitting cell inventory required")
        inventory = tuple(sorted(cell_inventory))
        suffixes = sorted({cell.split("/", 1)[1] for cell in inventory})
        if set(inventory) != {family + "/" + suffix for suffix in suffixes for family in FAMILIES}:
            raise ValueError("every matched group must include all three families")
        original_banks = _encoded(fit_banks)
        admitted = json.loads(original_banks)
        packed, identities, selected_banks, transcript_images = {}, {}, {}, set()
        for cell in inventory:
            rows = admitted[cell]
            _canonical(rows, "gradient/" + cell, model.config, "train_fit")
            if len(rows) < pairs_per_cell * 2:
                raise ValueError("not enough predeclared fitting pairs")
            for row in rows:
                if row["recipe"]["structure_split"] != ("shared" if row["recipe"]["depth"] < 2 else "train"):
                    raise ValueError("train-admissible fitting ancestry required")
                transcript = _encoded([turn["text"] for turn in row["turns"]])
                if transcript in transcript_images:
                    raise ValueError("duplicate fitting transcript")
                transcript_images.add(transcript)
            selected = rows[:pairs_per_cell * 2]
            selected_banks[cell] = selected
            identities[cell] = dict(rows_sha256=_sha(selected), pair_slots=list(range(pairs_per_cell)),
                pair_sha256=[_sha(selected[index:index+2]) for index in range(0, len(selected), 2)])
            packed[cell] = pack_composition_episodes(selected, device="cpu", training=True, pair_validator=validate_pair,
                max_turns=model.config.max_turns, max_input_bytes=model.config.max_input_bytes,
                max_context_tokens=model.config.max_positions, max_reply_bytes=model.config.max_output_bytes)
        batch_pins = {cell: _batch_sha(batch) for cell, batch in packed.items()}
        parameters, partitions, partition_identity = _partitions(model)
        report.update(production_contract=(inventory == CELLS and pairs_per_cell == 4),
            cell_inventory=list(inventory), matched_groups=suffixes, pairs_per_cell=pairs_per_cell,
            planned_groups=len(suffixes), planned_family_forwards=3 * len(suffixes),
            planned_encoder_episodes=len(inventory) * pairs_per_cell * 2,
            planned_component_gradient_evaluations=4 * len(suffixes),
            configuration=asdict(model.config), device=str(device), weights_sha256_before=checkpoint_digest(model),
            input_banks_sha256=hashlib.sha256(original_banks).hexdigest(), selected_banks_sha256=_sha(selected_banks),
            selected_cells=identities, packed_batch_sha256=batch_pins, parameter_partitions=partition_identity)

        def operation():
            for suffix in suffixes:
                report["groups_attempted"] += 1
                record = dict(group=suffix, status="incomplete", families={}, components={})
                report["groups"].append(record)
                component_values = {name: [] for name in COMPONENTS}
                for family in FAMILIES:
                    cell = family + "/" + suffix
                    batch = {section: {key: value.clone().to(device) for key, value in values.items()}
                             for section, values in packed[cell].items()}
                    inputs = batch["inputs"]
                    work = dict(family_forwards=1, encoder_episodes=inputs["token_ids"].shape[0],
                        encoder_actual_positions=int(inputs["lengths"].sum()), encoder_padded_positions=inputs["token_ids"].numel())
                    for key, value in work.items():
                        report[key + "_attempted"] += value
                    output = model(**inputs, decoder_input_ids=batch["supervision"]["reply_decoder_input_ids"])
                    for key, value in work.items():
                        report[key + "_completed"] += value
                    if _batch_sha(batch) != batch_pins[cell]:
                        raise RuntimeError("forward inputs or supervision changed")
                    values, denominators = _components(output, batch)
                    record["families"][family] = dict(cell=cell, batch_shape=list(inputs["token_ids"].shape),
                        packed_batch_sha256=batch_pins[cell],
                        decoder_input_shape=list(batch["supervision"]["reply_decoder_input_ids"].shape),
                        loss_components={name: float(value.detach()) for name, value in values.items()}, denominators=denominators)
                    for name, value in values.items():
                        component_values[name].append(value)
                means = {name: torch.stack(values).mean() for name, values in component_values.items()}
                if any(not bool(torch.isfinite(value)) for value in means.values()):
                    raise FloatingPointError("nonfinite family-mean objective")
                gradients = {}
                for index, name in enumerate(COMPONENTS):
                    record["components"][name] = dict(status="attempted", unweighted_loss=float(means[name].detach()))
                    report["component_gradient_evaluations_attempted"] += 1
                    gradients[name] = torch.autograd.grad(means[name], parameters,
                        retain_graph=index < len(COMPONENTS) - 1, create_graph=False, allow_unused=False)
                    report["component_gradient_evaluations_completed"] += 1
                    record["components"][name]["status"] = "completed"
                record["geometry"] = _geometry(gradients, partitions)
                record["weighted_loss"] = sum(WEIGHTS[name] * float(means[name].detach()) for name in COMPONENTS)
                record["status"] = "completed"
                report["groups_completed"] += 1
                del output, batch, means, component_values, gradients
            if {cell: _batch_sha(batch) for cell, batch in packed.items()} != batch_pins:
                raise RuntimeError("admitted packed batches changed")
            report["packed_batches_unchanged"] = True

        _protected(model, parameters, report, operation)
        report["input_banks_unchanged"] = _encoded(fit_banks) == original_banks
        if not report["input_banks_unchanged"]:
            raise RuntimeError("caller fitting banks changed during inspection")
        report.update(status="completed", wall_seconds=time.monotonic() - started,
                      cpu_seconds=time.process_time() - cpu_started)
        return copy.deepcopy(report)
    except BaseException as error:
        if original_banks is not None:
            try:
                report["input_banks_unchanged"] = _encoded(fit_banks) == original_banks
            except BaseException as check_error:
                report["input_banks_unchanged"] = False
                report["bank_check_error"] = repr(check_error)
        report.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=type(error).__name__ + ": " + str(error), wall_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - cpu_started)
        error.gradient_report = copy.deepcopy(report)
        raise
