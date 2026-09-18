"""Detached features from explicitly supplied learners and canonical row banks.

No file/checkpoint access, optimizer or backward calls. The caller owns source,
learner and dataset authentication, exclusions, runtime setup and wall limits.
These caches bind supplied bytes; they do not attest historical execution.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import time

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_student import _exact_replies, _generate_reply_tokens, checkpoint_digest
from brain_in_computer.language import ByteCodec
from experiments.composition_data import pack_composition_episodes
from experiments.foundation_metrics import _canonical
from experiments.sequence_student import SequenceConfig, SequenceStudent


SCHEMA = "bic-foundation-diagnostic-features-v1"
REPRESENTATIONS = ("eos", "production")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(_encoded([str(value.dtype), list(value.shape)]))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cache_sha(metadata, tensors):
    return hashlib.sha256(_encoded([json.loads(metadata),
        {name: _tensor_sha(value) for name, value in sorted(tensors.items())}])).hexdigest()


def _report(operation):
    counters = ("encoder_batches", "encoder_episodes", "encoder_actual_positions",
        "encoder_padded_positions", "bos_decoder_recurrent_calls", "bos_decoder_row_steps",
        "generated_decoder_calls", "generated_decoder_recurrent_steps", "generated_decoder_row_steps")
    return dict(schema=SCHEMA, operation=operation, status="incomplete",
        **{name + "_" + suffix: 0 for name in counters for suffix in ("attempted", "completed")},
        learner_optimizer_updates=0, backward_evaluations=0, checkpoint_reads=0,
        model_state_unchanged=None, model_state_restored=False, modes_restored=False,
        state_scope="Exact learned-state bytes, parameter/buffer placement and layout, parameter identities/flags, gradient bytes and configuration; module modes restored individually.",
        timing_scope="Function entry through validation, copies and cleanup; returned object serialization excluded.")


def _attach(error, report, started, cpu_started):
    report.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                  error=type(error).__name__ + ": " + str(error),
                  wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
    error.diagnostic_report = copy.deepcopy(report)


def _model(model):
    if type(model) is not SequenceStudent or type(model.config) is not SequenceConfig:
        raise ValueError("an explicit exact SequenceStudent and SequenceConfig are required")
    parameters = list(model.parameters())
    devices, dtypes = {p.device for p in parameters}, {p.dtype for p in parameters}
    if (len(devices) != 1 or len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64)
            or any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values())):
        raise ValueError("one-device finite float32/float64 learner required")
    return next(iter(devices))


def _batch_size(value, *, pairs=False):
    if type(value) is not int or value < (2 if pairs else 1) or pairs and value % 2:
        raise ValueError("positive batch size required; encoder batches preserve complete pairs")


def _run(model, report, phase, operation):
    """Restore checked values/flags when placement is intact; fail honestly otherwise."""
    modes = [(module, module.training) for module in model.modules()]
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    state_layout = {name: (value.device, value.dtype, tuple(value.shape))
                    for name, value in model.state_dict().items()}
    weight_digest, original_config = checkpoint_digest(model), model.config
    parameters = list(model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
    grad_devices = [None if p.grad is None else p.grad.device for p in parameters]
    handles, error, traceback, result = [], None, None, None

    def count(suffix, inputs):
        embedded = inputs[0]
        if phase == "bos":
            report["bos_decoder_recurrent_calls_" + suffix] += 1
            report["bos_decoder_row_steps_" + suffix] += embedded.shape[0] * embedded.shape[1]
        else:
            report["generated_decoder_recurrent_steps_" + suffix] += embedded.shape[1]
            report["generated_decoder_row_steps_" + suffix] += embedded.shape[0] * embedded.shape[1]

    def recurrent_completed(module, inputs, output):
        count("completed", inputs)
        if any(not bool(torch.isfinite(value).all()) for value in output):
            raise FloatingPointError("nonfinite diagnostic recurrent activity or hidden state")

    def readout_completed(module, inputs, output):
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("nonfinite diagnostic decoder logits before argmax")

    try:
        handles.append(model.inferior_frontal.recurrent.register_forward_pre_hook(
            lambda module, inputs: count("attempted", inputs)))
        handles.append(model.inferior_frontal.recurrent.register_forward_hook(recurrent_completed))
        handles.append(model.inferior_frontal.readout.register_forward_hook(readout_completed))
        model.eval()
        with torch.inference_mode():
            result = operation()
    except BaseException as caught:
        error, traceback = caught, caught.__traceback__
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in modes:
            module.training = mode
        report["modes_restored"] = True
        try:
            current = model.state_dict()
            placement_intact = (list(map(id, model.parameters())) == list(map(id, parameters))
                and {name: (value.device, value.dtype, tuple(value.shape))
                     for name, value in current.items()} == state_layout)
            if not placement_intact:
                report["model_state_unchanged"] = False
                raise RuntimeError("learner identities, dtype, device or layout changed; state restoration is not claimed")
            same_weights = checkpoint_digest(model) == weight_digest
            same_grads = all((p.grad is None and old is None) or
                (p.grad is not None and old is not None and p.grad.device == device
                 and _tensor_sha(p.grad) == _tensor_sha(old))
                for p, old, device in zip(parameters, grads, grad_devices))
            report["model_state_unchanged"] = (same_weights and same_grads
                and [p.requires_grad for p in parameters] == flags and model.config == original_config)
            if not report["model_state_unchanged"]:
                with torch.no_grad():
                    model.load_state_dict(state, strict=True)
                    for parameter, old, flag in zip(parameters, grads, flags):
                        parameter.grad = None if old is None else old.to(parameter.device).clone()
                        parameter.requires_grad_(flag)
                    model.config = original_config
                report["model_state_restored"] = True
                raise RuntimeError("diagnostic operation changed learner state; original state restored")
        except BaseException as cleanup_error:
            if error is None:
                error, traceback = cleanup_error, cleanup_error.__traceback__
            else:
                report["cleanup_error"] = type(cleanup_error).__name__ + ": " + str(cleanup_error)
    if error is not None:
        raise error.with_traceback(traceback)
    return result


@dataclass(frozen=True, slots=True)
class FeatureCache:
    """Detached in-memory data; getters clone tensors and expose no learner."""
    _metadata: bytes
    _tensors: dict
    _sha256: str
    _report: bytes

    @property
    def metadata(self):
        return json.loads(self._metadata)

    @property
    def report(self):
        return json.loads(self._report)

    @property
    def sha256(self):
        return self._sha256

    @property
    def tensors(self):
        return {name: value.clone() for name, value in self._tensors.items()}

    def _check(self):
        if _cache_sha(self._metadata, self._tensors) != self._sha256:
            raise ValueError("cached features or metadata changed")

    def probe_inputs(self, representation):
        if representation not in REPRESENTATIONS:
            raise ValueError("representation must be eos or production")
        self._check()
        metadata = self.metadata
        return dict(features=self._tensors[representation].clone(), labels=self._tensors["labels"].clone(),
            cells=metadata["cells"], kinds=metadata["kinds"], coordinates=metadata["coordinates"],
            cell_inventory=metadata["cell_inventory"])


def extract_features(model, banks, *, role, cell_inventory, batch_size=32):
    """Admit all cells before forwards and cache one BOS-only pass per episode.

    ``banks`` is a plain dict mapping family/depth/operator/length cell names to
    canonical pair lists. ``role`` is fit (train rows) or evaluation (dev rows).
    Explicit smaller inventories support fixtures; no full-63-cell claim is made.
    """
    started, cpu_started = time.monotonic(), time.process_time()
    report = _report("extract")
    try:
        device = _model(model)
        _batch_size(batch_size, pairs=True)
        if role not in ("fit", "evaluation"):
            raise ValueError("explicit fit/evaluation role required")
        if (type(cell_inventory) not in (list, tuple) or not cell_inventory
                or any(type(name) is not str for name in cell_inventory)
                or len(set(cell_inventory)) != len(cell_inventory)
                or type(banks) is not dict or set(banks) != set(cell_inventory)):
            raise ValueError("banks must exactly cover an explicit unique cell inventory")
        inventory = tuple(cell_inventory)
        canonical = _encoded(banks)
        admitted = json.loads(canonical)
        config = model.config
        packed, identities, cells, kinds, coordinates = {}, {}, [], [], []
        transcripts = set()
        for cell in inventory:
            rows = admitted[cell]
            identity, _, _ = _canonical(rows, "diagnostic/" + cell, config,
                                       "train_fit" if role == "fit" else "dev")
            for index, row in enumerate(rows):
                wanted = "shared" if row["recipe"]["depth"] < 2 else "train"
                if row["recipe"]["structure_split"] != wanted:
                    raise ValueError("diagnostic rows must use train-admissible composition ancestry")
                transcript = _encoded([turn["text"] for turn in row["turns"]])
                if transcript in transcripts:
                    raise ValueError("duplicate diagnostic transcript")
                transcripts.add(transcript)
                for turn_index, turn in enumerate(row["turns"]):
                    cells.append(cell)
                    kinds.append(turn["kind"])
                    coordinates.append([index // 2, index % 2, turn_index])
            packed[cell] = pack_composition_episodes(rows, device="cpu", training=False,
                max_turns=config.max_turns, max_input_bytes=config.max_input_bytes,
                max_context_tokens=config.max_positions, max_reply_bytes=config.max_output_bytes)
            identities[cell] = identity
        weights_sha = checkpoint_digest(model)
        storage = {name: [] for name in ("eos", "production", "action_logits", "bos_logits", "labels", "reply_targets")}

        def operation():
            for cell in inventory:
                batch = packed[cell]
                count = batch["inputs"]["token_ids"].shape[0]
                for lo in range(0, count, batch_size):
                    hi = min(count, lo + batch_size)
                    inputs = {name: value[lo:hi].clone().to(device) for name, value in batch["inputs"].items()}
                    width = int(inputs["lengths"].max())
                    for name in ("token_ids", "valid_mask"):
                        inputs[name] = inputs[name][:, :width]
                    shape = inputs["eos_positions"].shape
                    work = {"encoder_batches": 1, "encoder_episodes": hi - lo,
                        "encoder_actual_positions": int(inputs["lengths"].sum()),
                        "encoder_padded_positions": (hi - lo) * width}
                    for name, value in work.items():
                        report[name + "_attempted"] += value
                    output = model(**inputs, decoder_input_ids=torch.full((*shape, 1), ByteCodec.BOS,
                                                                         dtype=torch.long, device=device))
                    for name, value in work.items():
                        report[name + "_completed"] += value
                    values = {
                        "eos": output["context_states"].gather(1, inputs["eos_positions"][:, :, None].expand(-1, -1, config.width)).flatten(0, 1),
                        "production": output["production_context"].flatten(0, 1),
                        "action_logits": output["logits"].flatten(0, 1),
                        "bos_logits": output["language_logits"][:, :, 0].flatten(0, 1),
                        "labels": batch["supervision"]["action_targets"][lo:hi].flatten(),
                        "reply_targets": batch["supervision"]["reply_targets"][lo:hi].flatten(0, 1)}
                    values["reply_targets"] = F.pad(values["reply_targets"],
                        (0, config.max_output_bytes + 1 - values["reply_targets"].shape[1]), value=ByteCodec.PAD)
                    for name, value in values.items():
                        if not bool(torch.isfinite(value).all()):
                            raise FloatingPointError("nonfinite diagnostic features")
                        storage[name].append(value.detach().cpu().clone())
            return {name: torch.cat(values).contiguous() for name, values in storage.items()}

        tensors = _run(model, report, "bos", operation)
        metadata = _encoded(dict(schema=SCHEMA, role=role, cell_inventory=list(inventory),
            cells=cells, kinds=kinds, coordinates=coordinates, config=asdict(config),
            model_weights_sha256=weights_sha, dataset_sha256=hashlib.sha256(canonical).hexdigest(),
            bank_identities=identities, coordinate_order="cell inventory, pair slot, member zero/one, actual turn"))
        digest = _cache_sha(metadata, tensors)
        report.update(status="completed", cache_sha256=digest, turns=len(cells),
                      wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
        return FeatureCache(metadata, tensors, digest, _encoded(report))
    except BaseException as error:
        _attach(error, report, started, cpu_started)
        raise


def decode_cached_replies(model, cache, *, batch_size=64):
    """Decode cached reply contexts with no encoder call; return tokens/exact flags.

    Counts physical GRU row-steps, including padded work for already-finished
    replies until their chunk completes. BOS work in extraction is separate.
    """
    started, cpu_started = time.monotonic(), time.process_time()
    report = _report("decode_cached")
    try:
        device = _model(model)
        _batch_size(batch_size)
        if type(cache) is not FeatureCache:
            raise ValueError("an explicit FeatureCache is required")
        cache._check()
        metadata = cache.metadata
        if (metadata["config"] != asdict(model.config)
                or metadata["model_weights_sha256"] != checkpoint_digest(model)):
            raise ValueError("cached representations require the same learner weights and configuration")
        production = cache._tensors["production"]
        targets = cache._tensors["reply_targets"]
        def operation():
            tokens, correct = [], []
            for lo in range(0, len(production), batch_size):
                contexts = production[lo:lo + batch_size].clone().to(device)
                expected = targets[lo:lo + batch_size].clone().to(device)
                report["generated_decoder_calls_attempted"] += 1
                generated = _generate_reply_tokens(model, contexts)
                report["generated_decoder_calls_completed"] += 1
                exact = _exact_replies(generated, expected)
                tokens.append(F.pad(generated, (0, model.config.max_output_bytes + 2 - generated.shape[1]),
                                    value=ByteCodec.PAD).detach().cpu().clone())
                correct.append(exact.detach().cpu().clone())
            return torch.cat(tokens), torch.cat(correct)
        tokens, correct = _run(model, report, "generated", operation)
        # Clone outside inference_mode so callers receive ordinary detached tensors.
        result = dict(tokens=tokens.clone(), first_tokens=tokens[:, 1].clone(), reply_correct=correct.clone())
        report.update(status="completed", cache_sha256=cache.sha256, replies=len(production),
                      wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
        return {**result, "report": copy.deepcopy(report)}
    except BaseException as error:
        _attach(error, report, started, cpu_started)
        raise
