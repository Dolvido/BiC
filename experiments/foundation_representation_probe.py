"""Fixed CPU probes of detached cached features; never updates a source learner.

Callers authenticate/admit their caches separately and configure one CPU thread,
one inter-op thread and deterministic algorithms before tensor work. No setting
is changed at import or fit time. Evaluation labels are never accepted here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import torch
from torch.nn import functional as F


SCHEMA = "bic-foundation-representation-probe-v1"
SHUFFLE_SCHEMA = "foundation-shared-probe-shuffle-v1"
SHUFFLE_SEED = 916170003
FOUNDATION_CELLS = tuple(sorted(f"{family}/d{depth}/{operator}/t{turns}"
    for family in ("color", "count", "switch") for depth in range(6)
    for operator in (("direct",) if depth == 0 else ("copy", "advance") if depth == 1 else ("composed",))
    for turns in (8, 10, 12)))
STD_FLOOR = 1e-12
RIDGE = 1e-3
MAX_ITERATIONS = 200
MAX_CLOSURES = 999
GRADIENT_TOLERANCE = 1e-7
SOLVER_OPTIONS = dict(lr=1., max_iter=MAX_ITERATIONS, max_eval=MAX_CLOSURES,
    history_size=20, tolerance_grad=GRADIENT_TOLERANCE,
    tolerance_change=1e-12, line_search_fn="strong_wolfe")


class ProbeBudgetExceeded(RuntimeError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


def _runtime(*, for_fit=False):
    if for_fit and torch.is_inference_mode_enabled():
        raise ValueError("fitting is unavailable in ambient inference_mode")
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError("caller must configure CPU threads=1, interop_threads=1 and strict deterministic algorithms before tensor work")
    return dict(torch_version=str(torch.__version__), device="cpu", dtype="float64",
                threads=1, interop_threads=1, deterministic_algorithms=True)


def _features(value, *, width=None):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.float64) or value.layout != torch.strided
            or value.ndim != 2 or min(value.shape) < 1 or width is not None and value.shape[1] != width):
        raise ValueError("nonempty CPU float32/float64 feature matrix with matching width required")
    result = value.detach().to(dtype=torch.float64).clone().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("finite cached features required")
    return result


def _metadata(labels, cells, kinds, coordinates, cell_inventory):
    if (type(cell_inventory) not in (tuple, list) or not cell_inventory
            or any(type(c) is not str or not c for c in cell_inventory)
            or len(set(cell_inventory)) != len(cell_inventory)):
        raise ValueError("explicit distinct nonempty cell inventory required")
    if (type(labels) is not torch.Tensor or labels.device.type != "cpu" or labels.dtype != torch.long
            or labels.layout != torch.strided or labels.ndim != 1 or labels.numel() < 1):
        raise ValueError("nonempty CPU int64 fitting labels required")
    labels = labels.detach().clone().contiguous()
    size = len(labels)
    for value in (cells, kinds, coordinates):
        if type(value) not in (list, tuple) or len(value) != size:
            raise ValueError("one cell, kind and coordinate per fitting turn required")
    if any(type(c) is not str or c not in cell_inventory for c in cells) or set(cells) != set(cell_inventory):
        raise ValueError("fitting cells must exactly cover the explicit inventory")
    if any(type(k) is not str or k not in ("statement", "query") for k in kinds):
        raise ValueError("canonical statement/query kinds required")
    canonical = []
    for coordinate in coordinates:
        if (type(coordinate) not in (list, tuple) or len(coordinate) != 3
                or any(type(v) is not int or v < 0 for v in coordinate)
                or coordinate[1] not in (0, 1)):
            raise ValueError("coordinate must be (nonnegative pair slot, member 0/1, nonnegative turn)")
        canonical.append(tuple(coordinate))
    if len(set(zip(cells, canonical))) != size:
        raise ValueError("duplicate cell/turn coordinates")
    for label, kind in zip(labels.tolist(), kinds):
        if label not in (0, 1, 2, 3) or (label == 3) != (kind == "statement"):
            raise ValueError("labels must agree with canonical query/statement kinds")
    for cell in cell_inventory:
        seen = {kind for c, kind in zip(cells, kinds) if c == cell}
        if seen != {"statement", "query"}:
            raise ValueError("each cell requires query and acknowledgement fitting turns")
    return labels, tuple(cells), tuple(kinds), tuple(canonical), tuple(sorted(cell_inventory))


def _weights(labels, cells, inventory):
    weights = torch.zeros(len(labels), dtype=torch.float64)
    for cell in inventory:
        mask = torch.tensor([value == cell for value in cells], dtype=torch.bool)
        present = [target for target in range(3) if bool((mask & labels.eq(target)).any())]
        for target in present:
            selected = mask & labels.eq(target)
            weights[selected] = 1. / (len(inventory) * len(present) * int(selected.sum()))
        ack = mask & labels.eq(3)
        weights[ack] = .25 / (len(inventory) * int(ack.sum()))
    return weights


def _permutation(cells, kinds, coordinates):
    permutation = list(range(len(cells)))
    bindings = []
    for cell, kind in sorted(set(zip(cells, kinds))):
        positions = sorted((i for i, item in enumerate(zip(cells, kinds)) if item == (cell, kind)),
                           key=lambda i: coordinates[i])
        sources = sorted(positions, key=lambda i: (
            _hash([SHUFFLE_SCHEMA, SHUFFLE_SEED, cell, kind, *coordinates[i]]), coordinates[i]))
        for destination, source in zip(positions, sources):
            permutation[destination] = source
            bindings.append([cell, kind, list(coordinates[destination]), list(coordinates[source])])
    return torch.tensor(permutation, dtype=torch.long), _hash([SHUFFLE_SCHEMA, bindings])


def shuffled_labels(labels, *, cells, kinds, coordinates, cell_inventory=FOUNDATION_CELLS):
    """Detached labels and one fixed coordinate permutation; no canonical row edits."""
    labels, cells, kinds, coordinates, _ = _metadata(labels, cells, kinds, coordinates, cell_inventory)
    permutation, digest = _permutation(cells, kinds, coordinates)
    result = labels[permutation].clone()
    return dict(labels=result, source_indices=permutation.clone(), sha256=digest,
                changed_labels=int(result.ne(labels).sum()))


def _prepare(features, labels, cells, kinds, coordinates, cell_inventory):
    runtime = _runtime(for_fit=True)
    values = _metadata(labels, cells, kinds, coordinates, cell_inventory)
    labels, cells, kinds, coordinates, inventory = values
    matrix = _features(features)
    if matrix.shape[0] != len(labels):
        raise ValueError("feature/label rows differ")
    # Explicit coordinates, rather than caller batch order, own reduction order.
    order = sorted(range(len(labels)), key=lambda i: (inventory.index(cells[i]), coordinates[i]))
    matrix, labels = matrix[order].clone(), labels[order].clone()
    cells, kinds, coordinates = (tuple(value[i] for i in order) for value in (cells, kinds, coordinates))
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, correction=0)
    active = std.gt(STD_FLOOR)
    scale = torch.where(active, std, torch.ones_like(std))
    normalized = ((matrix - mean) / scale).masked_fill(~active, 0.)
    if not all(bool(torch.isfinite(x).all()) for x in (mean, std, normalized)):
        raise ValueError("nonfinite fit-only feature normalization")
    permutation, permutation_hash = _permutation(cells, kinds, coordinates)
    return dict(features=normalized, labels=labels, cells=cells, inventory=inventory,
                mean=mean, scale=scale, active=active, permutation=permutation,
                permutation_sha256=permutation_hash, runtime=runtime)


class ProbeResult:
    """Detached result; interrupted/nonfinite fits have no callable readout."""
    def __init__(self, report, prepared, weight=None, bias=None):
        self._report = copy.deepcopy(report)
        self._mean, self._scale, self._active = (prepared[k].detach().clone() for k in ("mean", "scale", "active"))
        self._weight = None if weight is None else weight.detach().clone()
        self._bias = None if bias is None else bias.detach().clone()

    @property
    def report(self):
        return copy.deepcopy(self._report)

    @property
    def normalization(self):
        return dict(mean=self._mean.clone(), scale=self._scale.clone(), active=self._active.clone())

    @property
    def coefficients(self):
        return None if self._weight is None else dict(weight=self._weight.clone(), bias=self._bias.clone())

    def logits(self, evaluation_features):
        _runtime()
        if self._weight is None:
            raise RuntimeError("incomplete probe has no evaluation predictions")
        matrix = _features(evaluation_features, width=self._mean.numel())
        normalized = ((matrix - self._mean) / self._scale).masked_fill(~self._active, 0.)
        result = normalized @ self._weight.T + self._bias
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite evaluation logits")
        return result.detach().clone()

    def predict(self, evaluation_features):
        return self.logits(evaluation_features).argmax(dim=1)


def _objective(features, labels, weights, weight, bias):
    per_turn = F.cross_entropy(features @ weight.T + bias, labels, reduction="none")
    return (weights * per_turn).sum() + .5 * RIDGE * (weight.square().sum() + bias.square().sum())


def _fit(prepared, shuffle):
    started, cpu_started = time.monotonic(), time.process_time()
    features, original = prepared["features"], prepared["labels"]
    labels = original[prepared["permutation"]].clone() if shuffle else original.clone()
    weights = _weights(labels, prepared["cells"], prepared["inventory"])
    weight = torch.nn.Parameter(torch.zeros((4, features.shape[1]), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([weight, bias], **SOLVER_OPTIONS)
    report = dict(schema=SCHEMA, runtime=copy.deepcopy(prepared["runtime"]),
        cell_inventory=list(prepared["inventory"]), production_inventory=prepared["inventory"] == FOUNDATION_CELLS,
        fitting_turns=len(labels), feature_width=features.shape[1], active_features=int(prepared["active"].sum()),
        shuffled=shuffle, permutation_sha256=prepared["permutation_sha256"],
        changed_labels=int(labels.ne(original).sum()), sample_weight_sum=float(weights.sum()), ridge=RIDGE,
        solver_options=copy.deepcopy(SOLVER_OPTIONS), learner_optimizer_updates=0,
        timing_scope="Single optimizer fit and final residual; input preparation and predictions excluded.",
        probe_optimizer_step_attempts=0, completed_probe_optimizer_steps=0, probe_solver_iterations=0,
        closure_attempts=0, completed_closures=0, objective_evaluations_started=0,
        objective_evaluations_completed=0, backward_evaluations_started=0, backward_evaluations_completed=0,
        final_evaluation_attempts=0, final_evaluations_completed=0, converged=False,
        status="incomplete", final_objective=None, final_gradient_max_abs=None)

    def evaluate():
        report["objective_evaluations_started"] += 1
        with torch.enable_grad():
            optimizer.zero_grad(set_to_none=True)
            loss = _objective(features, labels, weights, weight, bias)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite probe fitting objective")
            report["backward_evaluations_started"] += 1
            loss.backward()
            report["backward_evaluations_completed"] += 1
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in (weight, bias)):
            raise FloatingPointError("nonfinite probe fitting gradient")
        report["objective_evaluations_completed"] += 1
        return loss

    def closure():
        report["closure_attempts"] += 1
        if report["completed_closures"] >= MAX_CLOSURES:
            raise ProbeBudgetExceeded("hard fitting-closure budget exhausted")
        loss = evaluate()
        report["completed_closures"] += 1
        return loss

    valid, interrupted = False, None
    try:
        report["probe_optimizer_step_attempts"] = 1
        optimizer.step(closure)
        report["completed_probe_optimizer_steps"] = 1
        if not all(bool(torch.isfinite(p).all()) for p in (weight, bias)):
            raise FloatingPointError("nonfinite returned probe coefficients")
        report["final_evaluation_attempts"] = 1
        final = evaluate()
        report["final_evaluations_completed"] = 1
        residual = max(float(p.grad.abs().max()) for p in (weight, bias))
        report.update(final_objective=float(final.detach()), final_gradient_max_abs=residual,
                      converged=residual <= GRADIENT_TOLERANCE,
                      status="converged" if residual <= GRADIENT_TOLERANCE else "normal_return_unconverged")
        valid = True
    except ProbeBudgetExceeded as error:
        report.update(status="budget_interrupted", error=str(error))
    except Exception as error:
        report.update(status="nonfinite" if isinstance(error, FloatingPointError) else "solver_error",
                      error=repr(error))
    except BaseException as error:
        interrupted = error
        report.update(status="interrupted", error=repr(error))
    finally:
        state = optimizer.state.get(weight, {})
        report["probe_solver_iterations"] = int(state.get("n_iter", 0))
        report["solver_recorded_function_evaluations"] = int(state.get("func_evals", 0))
        report.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started)
    if interrupted is not None:
        interrupted.probe_report = copy.deepcopy(report)
        raise interrupted
    return ProbeResult(report, prepared, weight if valid else None, bias if valid else None)


def fit_probe(features, labels, *, cells, kinds, coordinates,
              cell_inventory=FOUNDATION_CELLS, shuffle=False):
    """Fit one declared readout; no evaluation inputs/labels or source models."""
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be a boolean")
    prepared = _prepare(features, labels, cells, kinds, coordinates, cell_inventory)
    return _fit(prepared, shuffle)


def fit_true_and_shuffled(features, labels, *, cells, kinds, coordinates,
                          cell_inventory=FOUNDATION_CELLS):
    """Two fixed fits sharing exactly one detached fitting normalization."""
    prepared = _prepare(features, labels, cells, kinds, coordinates, cell_inventory)
    first = _fit(prepared, False)
    try:
        second = _fit(prepared, True)
    except BaseException as error:
        error.completed_probe_reports = [first.report,
            *copy.deepcopy(getattr(error, "completed_probe_reports", []))]
        raise
    return {"true": first, "shuffled": second}
