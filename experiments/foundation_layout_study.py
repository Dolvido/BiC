"""Fixed local layout comparison; exclusive launch, no resume or promotion.

``prepare`` authenticates already-prepared data and freezes counts without any
model construction or scoring. ``run`` is one explicitly pinned invocation.
The caller must review the resulting launch before starting that invocation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-foundation-layout-study-v1"
PROTOCOL = "docs/FOUNDATION_LAYOUT_STUDY_PROTOCOL.md"
CONTEXT_LAUNCH = "runs/foundation-context-diagnostic-local/attempt-001/launch.json"
CONTEXT_LAUNCH_SHA256 = "2511467a862262a0004480838e2465324a2f27ec1818c90c9427b61f69fe02ba"
CONFIG = dict(width=192, layers=4, heads=4, feedforward=768, max_positions=1024,
              max_turns=12, max_input_bytes=128, max_output_bytes=32)
ARMS, FAMILIES = ("original", "varied"), ("color", "count", "switch")
STEPS, SEED, RATE, MICRO, BATCH, SECONDS = (0, 144, 288, 432, 576, 720, 864), 852020799, .003, 32, 32, 5400
VALIDATIONS = {
    "curriculum": ("runs/foundation-layout-curriculum-validation-local/attempt-001/report.json",
        "1f9301e5e7a18f4440fdb8e8b8ad0d1c6f6942cf586ddf61974072423530ad5a", 8),
    "trainer": ("runs/foundation-layout-training-validation-local/attempt-001/report.json",
        "12813a39a6c61856ec974593b19561c7f48c37dec72f13924ed2e599f7866792", 2),
    "evaluator": ("runs/foundation-layout-evaluation-validation-local/attempt-002/report.json",
        "3e947c8f2e3cfab46be682ea8c9249c61f301ac1fa7a2fae77db8b32fe02bd96", 7),
}
FAILED_EVALUATOR = ("runs/foundation-layout-evaluation-validation-local/attempt-001/report.json",
                    "d3472515d34828a5a538274881b3a66e678bcde159ee8be7180d719245476a94")
PAIR_METRICS = ("anchor_pair_action", "anchor_pair_reply")


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def _hash(value):
    return hashlib.sha256(encoded(value).rstrip(b"\n")).hexdigest()


def native(path):
    value = str(Path(path).absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\"+value[2:] if value.startswith("\\\\") else "\\\\?\\"+value
    return Path(value)


def digest(path):
    value = hashlib.sha256()
    with native(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(native(path).read_bytes())


def publish(path, value, *, checkpoint=False):
    target = native(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        if checkpoint:
            import torch
            torch.save(value, stream)
        else:
            stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    return digest(target)


def relative_root(path):
    value = Path(path).resolve()
    if not value.is_relative_to(ROOT) or value == ROOT:
        raise ValueError("a concrete path within the repository is required")
    return value.relative_to(ROOT).as_posix()


def verify_pins(pins):
    for name, expected in pins.items():
        if relative_root(ROOT/name) != name or digest(ROOT/name) != expected:
            raise ValueError("pinned source/input changed: "+name)


def source_pins():
    from experiments import foundation_layout_data as data, foundation_layout_evaluation as evaluation
    names = set(data.source_hashes()) | set(evaluation.source_hashes()) | {
        "experiments/foundation_layout_study.py", "experiments/execution_profile.py", PROTOCOL}
    if native(ROOT/"experiments/__init__.py").exists():
        raise ValueError("experiment namespace initializer must remain absent")
    return {name: digest(ROOT/name) for name in sorted(names)}


def contract():
    return dict(schema=SCHEMA, arms=list(ARMS), family_order=list(FAMILIES), seed=SEED, config=CONFIG,
        learning_rate=RATE, micro_batch_size=MICRO, evaluation_batch_size=BATCH, checkpoints=list(STEPS),
        updates_per_arm=864, max_seconds=SECONDS, device="cuda:0", optimizer="AdamW", gradient_clip=1.,
        objective_id="unchanged-sequence-objective-three-family-mean-v1", cpu_threads=1, cpu_interop_threads=1,
        initialization="Freeze seed/config/source/runtime recipe; record first actual digest and require second fresh construction to match exactly.",
        initial_evaluation="Shared once on normal train_fit/dev, then referenced by both equal initial weights.",
        evaluator_grouping="Concatenate complete cells by role/view/panel/turn count; preserve original cell and episode identities.",
        screen=dict(minimum_pooled_audit_gain_numerator=1, minimum_pooled_audit_gain_denominator=20,
            family_gain="Both pair measures strictly improve in the same panel/layout somewhere for every family; no family regression anywhere.",
            controls="Endpoint normal strictly exceeds blank and reset per dev family/layout/measure.",
            forgetting="max(0,best earlier observed normal minus endpoint), varied minus original <= 1/20, per dev family/layout/measure."),
        automatic_retry=False, optimizer_resume_verified=False, automatic_promotion=False, teacher_calls=0,
        timing_confound="Original always precedes varied; timing is not counterbalanced.")


def evaluation_schedule():
    records = [dict(id="shared-000000-normal", arm="shared", step=0, control="normal", roles=["train_fit", "dev"])]
    for arm in ARMS:
        for step in STEPS[1:]:
            records.append(dict(id=f"{arm}-{step:06d}-normal", arm=arm, step=step, control="normal",
                                roles=["train_fit", "dev", "audit"] if step == 864 else ["dev"]))
            if step == 864:
                for control in ("blank", "reset"):
                    records.append(dict(id=f"{arm}-{step:06d}-{control}", arm=arm, step=step, control=control, roles=["dev"]))
    return records


def denominators(rows):
    queries = [(row, query, row["turns"][query["turn_index"]]["target"])
               for row in rows for query in row["queries"]]
    total, known = len(queries), sum(target < 2 for _, _, target in queries)
    other = sum(target < 2 and query["query_id"] != row["anchor"]["query_id"] for row, query, target in queries)
    result = {name: count for prefix, count in (("query", total), ("known", known), ("other_known", other), ("unknown", total-known))
              for name in (prefix+"_action", prefix+"_reply")}
    result.update(action_reply_agreement=total, reply_parseable=total,
                  known_unsupported_ask=known, known_reply_unsupported_ask=known)
    result.update({"anchor_pair_"+name: len(rows)//2 for name in ("action", "reply", "both")})
    return result


def _encoder_positions(rows):
    counts = {control: dict(actual=0, padded=0) for control in ("normal", "blank", "reset")}
    for offset in range(0, len(rows), BATCH):
        chunk = rows[offset:offset+BATCH]
        lengths = [[len(turn["text"].encode("utf-8"))+2 for turn in row["turns"]] for row in chunk]
        for control, sizes in (("normal", [sum(row) for row in lengths]),
                ("blank", [2*len(row) for row in lengths]), ("reset", [value for row in lengths for value in row])):
            counts[control]["actual"] += sum(sizes)
            counts[control]["padded"] += len(sizes)*max(sizes)
    return counts


def bank_inventory(banks):
    result = {}
    for role, views in sorted(banks.items()):
        if role not in ("train_fit", "dev", "audit") or set(views) != set(ARMS):
            raise ValueError("exact role and two evaluation layouts required")
        for view in ARMS:
            groups = defaultdict(list)
            for name, rows in sorted(views[view].items()):
                panel = name.split("/")[0]
                if panel not in (("fit",) if role == "train_fit" else ("fresh", "composed")):
                    raise ValueError("unexpected bank panel name")
                if type(rows) is not list or len(rows) != 16 or any(row["recipe"]["layout"] != view for row in rows):
                    raise ValueError("each admitted bank cell requires eight complete pairs of its declared view")
                turns = {len(row["turns"]) for row in rows}
                if len(turns) != 1 or next(iter(turns)) not in (8, 10, 12):
                    raise ValueError("uniform declared turn bucket required")
                groups[(panel, next(iter(turns)))].append((name, rows))
            for (panel, turns), cells in sorted(groups.items()):
                rows = [row for _, cell in cells for row in cell]
                identity = f"{role}.{view}.{panel}.t{turns}"
                by_episode = {row["id"]: name for name, cell in cells for row in cell}
                if len(by_episode) != len(rows):
                    raise ValueError("repeated evaluation episode identity")
                result[identity] = dict(id=identity, role=role, view=view, panel=panel, turns=turns,
                    episodes=len(rows), pairs=len(rows)//2, rows_sha256=_hash(rows), cells=[name for name, _ in cells],
                    cell_by_episode=by_episode, denominators=denominators(rows),
                    encoder_positions=_encoder_positions(rows),
                    cell_denominators={name: denominators(cell) for name, cell in cells})
    for role in banks:
        expected = {"fit": 504} if role == "train_fit" else {"fresh": 504, "composed": 216 if role == "dev" else 288}
        for view in ARMS:
            for panel, count in expected.items():
                actual = sum(item["pairs"] for item in result.values()
                             if (item["role"], item["view"], item["panel"]) == (role, view, panel))
                if actual != count:
                    raise ValueError("fixed panel pair denominator differs")
    return result


def expected_work(inventory):
    value = dict(model_constructions=2, optimizer_updates=1728, training_forwards=5184,
        training_backwards=5184, training_episodes=165888, full_snapshots=14,
        scored_episode_records=0, sequence_forward_calls=0, encoder_episode_passes=0,
        encoder_actual_positions=0, encoder_padded_positions=0,
        bos_reply_contexts=0, free_reply_contexts=0, scored_query_turns=0, teacher_calls=0)
    for spec in evaluation_schedule():
        for bank in inventory.values():
            if bank["role"] not in spec["roles"]:
                continue
            rows, turns = bank["episodes"], bank["turns"]
            value["scored_episode_records"] += rows
            value["sequence_forward_calls"] += math.ceil(rows/BATCH)
            value["encoder_episode_passes"] += rows*turns if spec["control"] == "reset" else rows
            value["encoder_actual_positions"] += bank["encoder_positions"][spec["control"]]["actual"]
            value["encoder_padded_positions"] += bank["encoder_positions"][spec["control"]]["padded"]
            value["bos_reply_contexts"] += rows*turns
            value["free_reply_contexts"] += rows*turns
            value["scored_query_turns"] += bank["denominators"]["query_action"]
    value["free_decoder_recurrent_calls_max"] = value["sequence_forward_calls"]*(CONFIG["max_output_bytes"]+1)
    value["free_decoder_row_steps_max"] = value["free_reply_contexts"]*(CONFIG["max_output_bytes"]+1)
    return value


def _validation_evidence():
    pins, reports = {}, {}
    for label, (name, expected, count) in VALIDATIONS.items():
        verify_pins({name: expected})
        report = read(ROOT/name)
        if report.get("status") != "passed" or report.get("tests", report.get("tests_run")) != count:
            raise ValueError("passed reviewed validation required: "+label)
        pins[name] = expected
        if label == "evaluator":
            started = str(Path(name).parent.as_posix())+"/started.json"
            pins[started] = report["artifacts_sha256"]["started.json"]
            verify_pins({started: pins[started]})
            tested = read(ROOT/started)["source_sha256"]
            pins[FAILED_EVALUATOR[0]] = FAILED_EVALUATOR[1]
        else:
            tested = report.get("file_sha256", report["source_sha256"])
        verify_pins(tested)
        pins.update(tested)
        reports[label] = dict(path=name, sha256=expected, wall_seconds=report["wall_seconds"], cpu_seconds=report["cpu_seconds"],
            cumulative_wall_seconds=report.get("cumulative_attempt_wall_seconds", report["wall_seconds"]),
            cumulative_cpu_seconds=report.get("cumulative_attempt_cpu_seconds", report["cpu_seconds"]))
    verify_pins(pins)
    return pins, reports


def _pure_ast_signature(image):
    """Bind the tested arithmetic without a whole-runner self-pin cycle."""
    import ast
    functions = {"summarize", "_pool", "contract", "evaluation_schedule", "expected_work",
                 "denominators", "_encoder_positions", "bank_inventory", "_hash", "encoded"}
    constants = {"SCHEMA", "ARMS", "FAMILIES", "STEPS", "PAIR_METRICS", "CONFIG", "SEED", "RATE", "MICRO", "BATCH", "SECONDS"}
    result, found = {}, set()
    for node in ast.parse(image).body:
        keys = []
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            keys = ["function:"+node.name]
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {name.id for target in node.targets for name in ast.walk(target) if isinstance(name, ast.Name)} & constants
            keys = ["constant:"+name for name in sorted(names)]
            found.update(names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keys = ["import:"+ast.dump(node, include_attributes=False)]
        for key in keys:
            result[key] = hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
    if found != functions | constants:
        raise ValueError("complete pure-screen function and constant closure required")
    return result


def _screen_evidence(path, expected_sha256):
    name = relative_root(path)
    verify_pins({name: expected_sha256})
    report = read(ROOT/name)
    if report.get("status") != "passed" or report.get("tests", report.get("tests_run")) != 8:
        raise ValueError("passed eight-test pure-screen receipt required")
    pins = {name: expected_sha256}
    archived = None
    for tested in ("experiments/foundation_layout_study.py", "tests/test_foundation_layout_study.py"):
        snapshot_name = report["source_snapshots"][tested]
        snapshot = (ROOT/name).parent/snapshot_name
        if not snapshot.resolve().is_relative_to((ROOT/name).parent.resolve()):
            raise ValueError("screen source snapshot escapes receipt directory")
        digest_value = report["source_sha256"][tested]
        pins[relative_root(snapshot)] = digest_value
        verify_pins({relative_root(snapshot): digest_value})
        if tested.startswith("experiments/"):
            archived = native(snapshot).read_bytes()
        else:
            pins[tested] = digest_value
    verify_pins(pins)
    signature = _pure_ast_signature(native(Path(__file__)).read_bytes())
    if signature != _pure_ast_signature(archived):
        raise ValueError("tested screen arithmetic/configuration changed")
    return pins, dict(path=name, sha256=expected_sha256, tested_runner_sha256=report["source_sha256"]["experiments/foundation_layout_study.py"],
        current_pure_ast_sha256=signature, wall_seconds=report["wall_seconds"], cpu_seconds=report["cpu_seconds"],
        scope="Eight synthetic-count tests. Current pure functions, constants and imports match the archived tested source; no model validation is inferred.")


def counted_data_load(call, counters):
    """Synchronous data owner: count actual safe archive attempts and returns."""
    import torch
    original = torch.load
    def tracked(*args, **kwargs):
        if kwargs.get("weights_only") is not True or kwargs.get("map_location") != "cpu":
            raise ValueError("data loading requires explicit CPU weights_only=True")
        counters["weights_only_bank_loads_attempted"] += 1
        value = original(*args, **kwargs)
        counters["weights_only_bank_loads_completed"] += 1
        return value
    torch.load = tracked
    try:
        return call()
    finally:
        torch.load = original


def prepare(output, data_directory, *, expected_data_manifest_sha256, expected_data_preparation_sha256,
            expected_data_process_sha256, screen_validation, expected_screen_validation_sha256):
    """Freeze an executable launch from sealed data; no model or inference."""
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve()
    native(output).mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="running", started_utc=utc(), model_constructions=0, neural_calls=0,
                  weights_only_bank_loads_attempted=0, weights_only_bank_loads_completed=0)
    try:
        from experiments import foundation_layout_data as data
        source = source_pins()
        pins, validations = _validation_evidence()
        screen_pins, screen_evidence = _screen_evidence(screen_validation, expected_screen_validation_sha256)
        pins.update(screen_pins)
        pins[CONTEXT_LAUNCH] = CONTEXT_LAUNCH_SHA256
        data_name = relative_root(data_directory)
        pins[data_name+"/manifest.json"] = expected_data_manifest_sha256
        pins[data_name+"/preparation.json"] = expected_data_preparation_sha256
        process_name = relative_root((ROOT/data_name).parent/"receipt.json")
        pins[process_name] = expected_data_process_sha256
        verify_pins(pins)
        process = read(ROOT/process_name)
        if (process.get("schema") != "bic-foundation-layout-preparation-process-v1" or process.get("status") != "completed"
                or process.get("manifest_sha256") != expected_data_manifest_sha256
                or process.get("preparation_sha256") != expected_data_preparation_sha256
                or process.get("cuda_initialized_before") is not False or process.get("cuda_initialized_after") is not False
                or not process.get("guard_attempts") or any(process["guard_attempts"].values())):
            raise ValueError("completed enclosing CPU-only preparation receipt required")
        pins["experiments/foundation_layout_prepare.py"] = process["source_sha256"]
        verify_pins({"experiments/foundation_layout_prepare.py": process["source_sha256"]})
        sealed = counted_data_load(lambda: data.load(ROOT/data_name, expected_manifest_sha256=expected_data_manifest_sha256,
                           roles=("train_fit", "dev", "audit")), report)
        if report["weights_only_bank_loads_completed"] != 6:
            raise ValueError("launch census must load exactly six bank archives")
        prior_specs = sealed["manifest"]["contract"].get("prior_preparation_receipts")
        if type(prior_specs) is not list or not prior_specs:
            raise ValueError("amended data must bind its prior failed preparation process receipts")
        prior_cost, prior_names = [], set()
        for spec in prior_specs:
            if (type(spec) is not dict or set(spec) != {"path", "sha256"}
                    or spec["path"] in prior_names or spec["path"] == process_name):
                raise ValueError("distinct exact prior-preparation receipt identities required")
            verify_pins({spec["path"]: spec["sha256"]})
            previous = read(ROOT/spec["path"])
            if (previous.get("schema") != "bic-foundation-layout-preparation-process-v1" or previous.get("status") != "failed"
                    or previous.get("cuda_initialized_before") is not False
                    or not previous.get("guard_attempts") or any(previous["guard_attempts"].values())
                    or any(type(previous.get(key)) not in (int, float) or not math.isfinite(previous[key]) or previous[key] < 0
                           for key in ("wall_seconds", "cpu_seconds"))):
                raise ValueError("preserved failed CPU-only preparation and finite costs required")
            prior_names.add(spec["path"])
            pins[spec["path"]] = spec["sha256"]
            prior_cost.append(dict(**spec, status=previous["status"], wall_seconds=previous["wall_seconds"],
                                   cpu_seconds=previous["cpu_seconds"]))
        inventory = bank_inventory(sealed["banks"])
        for name, expected in sealed["manifest"]["artifacts_sha256"].items():
            pins[data_name+"/"+name] = expected
        pins[data_name+"/events.jsonl"] = sealed["preparation"]["journal_sha256"]
        pins.update(sealed["manifest"]["input_file_sha256"])
        runtime = read(ROOT/CONTEXT_LAUNCH)["expected_runtime"]
        if sealed["plan"]["schedules"]["curriculum"] != list(range(864)):
            raise ValueError("both arms require the same contiguous canonical schedule")
        snapshot_paths = {}
        for index, (name, expected) in enumerate(sorted({**source, **pins}.items())):
            # Preserve code and compact receipts; large data archives retain
            # their immutable existing location and explicit byte digest.
            if name in source or name.endswith(("report.json", "receipt.json", "started.json", "manifest.json", "preparation.json")):
                target = output/"source"/(f"s{index:03}"+Path(name).suffix)
                native(target.parent).mkdir(parents=True, exist_ok=True)
                image = native(ROOT/name).read_bytes()
                if hashlib.sha256(image).hexdigest() != expected:
                    raise ValueError("input changed during launch snapshot")
                with native(target).open("xb") as handle:
                    handle.write(image); handle.flush(); os.fsync(handle.fileno())
                snapshot_paths[name] = target.relative_to(output).as_posix()
        launch = dict(schema=SCHEMA, contract=contract(), source_sha256=source, input_sha256=pins,
            source_snapshots=snapshot_paths, data_directory=data_name, data_manifest_sha256=expected_data_manifest_sha256,
            data_preparation_sha256=expected_data_preparation_sha256, expected_runtime=runtime,
            initialization_recipe=dict(seed=SEED, config=CONFIG, source_sha256=source, runtime=runtime),
            bank_inventory=inventory, evaluation_schedule=evaluation_schedule(), expected_work=expected_work(inventory),
            validation_evidence=validations, pure_screen_validation=screen_evidence,
            data_preparation_cost=dict(wall_seconds=sealed["preparation"]["wall_seconds"],
                cpu_seconds=sealed["preparation"]["cpu_seconds"], work=sealed["preparation"]["work"],
                enclosing_process_receipt=dict(path=process_name, sha256=expected_data_process_sha256,
                    wall_seconds=process["wall_seconds"], cpu_seconds=process["cpu_seconds"],
                    scope="Includes imports and the nested data.prepare work; do not add the two times together."),
                prior_failed_process_receipts=prior_cost,
                cumulative_process_wall_seconds=process["wall_seconds"]+sum(row["wall_seconds"] for row in prior_cost),
                cumulative_process_cpu_seconds=process["cpu_seconds"]+sum(row["cpu_seconds"] for row in prior_cost)),
            automatic_retry=False, automatic_promotion=False)
        if source_pins() != source:
            raise ValueError("source closure changed during launch freeze")
        verify_pins(pins)
        report.update(status="completed", launch_sha256=publish(output/"launch.json", launch),
                      expected_work=launch["expected_work"])
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                      partial_bank_load_count_unknown=report["weights_only_bank_loads_attempted"] != report["weights_only_bank_loads_completed"])
        raise
    finally:
        report.update(ended_utc=utc(), wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            timing_scope="Launch preparation through input authentication, six bank archive loads, denominator census and source copies; final receipt publication excluded.")
        publish(output/"preparation.json", report)
    return report


def _pool(summaries):
    keys = set(summaries[0]["counts"])
    if any(set(row["counts"]) != keys for row in summaries):
        raise ValueError("metric sets differ during pooling")
    result = {key: sum(row[key] for row in summaries) for key in ("episodes", "pairs", "query_turns")}
    result["counts"] = {}
    for key in sorted(keys):
        count = sum(row["counts"][key]["count"] for row in summaries)
        total = sum(row["counts"][key]["total"] for row in summaries)
        result["counts"][key] = dict(count=count, total=total, rate=count/total if total else None)
    return result


def summarize(results):
    """The five fixed descriptive screen rules, without checkpoint selection."""
    checks, trajectories = [], {}

    def counts(arm, step, role, view, panel=None, family=None, control="normal"):
        name = "shared-000000-normal" if step == 0 else f"{arm}-{step:06d}-{control}"
        panels = results[name]["by_role"][role][view]
        chosen = [panels[panel]] if panel is not None else list(panels.values())
        return _pool([row["metrics"]["by_family"][family] if family else row["metrics"]["overall"] for row in chosen])["counts"]

    def fraction(row):
        return Fraction(row["count"], row["total"]) if row["total"] else None

    def compare(rule, label, left, right, *, minimum=Fraction(0), strict=False, optional=False):
        if left["total"] != right["total"]:
            raise ValueError("matched screen denominators differ")
        a, b = fraction(left), fraction(right)
        if a is None:
            if not optional:
                raise ValueError("required screen denominator is empty")
            checks.append(dict(rule=rule, condition=label, applicable=False, passed=True, left=left, right=right))
            return None
        delta = a-b
        passed = delta > minimum if strict else delta >= minimum
        checks.append(dict(rule=rule, condition=label, applicable=True, passed=passed,
            left=left, right=right, delta=float(delta), threshold=float(minimum), strict=strict))
        return delta

    improved = {family: [] for family in FAMILIES}
    for view in ARMS:
        for panel in ("fresh", "composed"):
            a, b = (counts(arm, 864, "audit", view, panel) for arm in ("varied", "original"))
            for metric in PAIR_METRICS:
                compare(1, f"{view}/{panel}/{metric}", a[metric], b[metric], minimum=Fraction(1, 20))
            for family in FAMILIES:
                fa, fb = (counts(arm, 864, "audit", view, panel, family) for arm in ("varied", "original"))
                deltas = [compare(2, f"{view}/{panel}/{family}/{metric}", fa[metric], fb[metric]) for metric in PAIR_METRICS]
                if all(delta > 0 for delta in deltas):
                    improved[family].append(f"{view}/{panel}")
            for metric in ("unknown_action", "unknown_reply", "action_reply_agreement", "other_known_action", "other_known_reply"):
                compare(3, f"{view}/{panel}/{metric}", a[metric], b[metric], optional=metric.startswith("other_known"))
    for family, locations in improved.items():
        checks.append(dict(rule=2, condition=f"{family}/joint-positive-condition", passed=bool(locations), conditions=locations))
    for view in ARMS:
        for family in FAMILIES:
            normal = counts("varied", 864, "dev", view, family=family)
            for control in ("blank", "reset"):
                controlled = counts("varied", 864, "dev", view, family=family, control=control)
                for metric in PAIR_METRICS:
                    compare(4, f"{view}/{family}/{control}/{metric}", normal[metric], controlled[metric], strict=True)
            for metric in PAIR_METRICS:
                losses, detail = {}, {}
                for arm in ARMS:
                    points = [dict(step=step, **counts(arm, step, "dev", view, family=family)[metric]) for step in STEPS]
                    if len({p["total"] for p in points}) != 1 or points[0]["total"] == 0:
                        raise ValueError("development trajectory denominator differs")
                    earlier, endpoint = max(fraction(point) for point in points[:-1]), fraction(points[-1])
                    losses[arm] = max(Fraction(0), earlier-endpoint)
                    detail[arm] = dict(trajectory=points, best_earlier_rate=float(earlier), endpoint_rate=float(endpoint),
                                       absolute_loss=float(losses[arm]), signed_earlier_minus_endpoint=float(earlier-endpoint))
                identity = f"{view}/{family}/{metric}"
                trajectories[identity] = detail
                excess = losses["varied"]-losses["original"]
                checks.append(dict(rule=5, condition=identity, passed=excess <= Fraction(1, 20),
                    varied_loss=float(losses["varied"]), original_loss=float(losses["original"]),
                    excess_loss=float(excess), maximum_excess_loss=.05))
    return dict(schema=SCHEMA, screen_passed=all(check["passed"] for check in checks), checks=checks,
        rule_passed={str(rule): all(check["passed"] for check in checks if check["rule"] == rule) for rule in range(1, 6)},
        development_trajectories=trajectories, automatic_promotion=False,
        scope="Prospective descriptive engineering screen for one paired initialization; no general learning, causal-mechanism or statistical-confidence claim.")


class Journal:
    def __init__(self, path):
        self.handle = native(path).open("xb")
        self.sequence, self.chain = 0, _hash([SCHEMA, "events"])

    def event(self, value):
        body = dict(sequence=self.sequence, previous_sha256=self.chain, utc=utc(), **value)
        digest_value = _hash(body)
        self.handle.write(encoded(dict(**body, event_sha256=digest_value)))
        self.handle.flush(); os.fsync(self.handle.fileno())
        self.sequence += 1
        self.chain = digest_value

    def close(self):
        self.handle.flush(); os.fsync(self.handle.fileno()); self.handle.close()


def _metric_denominators(metrics, expected):
    actual = {key: value["total"] for key, value in metrics["overall"]["counts"].items()}
    if actual != expected:
        raise ValueError("scored query/pair denominators differ from prospective inventory")


def run(output, *, expected_launch_sha256):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output).resolve()
    if digest(output/"launch.json") != expected_launch_sha256:
        raise ValueError("explicit frozen launch SHA256 required")
    launch = read(output/"launch.json")
    if (launch.get("schema") != SCHEMA or launch.get("contract") != contract()
            or launch.get("evaluation_schedule") != evaluation_schedule()
            or launch.get("expected_work") != expected_work(launch["bank_inventory"])):
        raise ValueError("complete fixed executable launch contract required")
    attempt = output/"training"
    native(attempt).mkdir(exist_ok=False)
    deadline = started+SECONDS
    receipt = dict(schema=SCHEMA, status="running", started_utc=utc(), process_id=os.getpid(),
        launch_sha256=expected_launch_sha256, model_construction_attempts=0, model_constructions=0,
        snapshot_attempts=0, completed_full_snapshots=0, checkpoint_publication_attempts=0,
        checkpoint_publications=0, weights_only_bank_loads_attempted=0, weights_only_bank_loads_completed=0,
        prepared_bank_attempts=0, prepared_banks=0, completed_evaluations=[], completed_arms=[], arms={},
        teacher_calls=0, automatic_retry=False, automatic_promotion=False, partial_work_unknown=False)
    journal, trainer, data_work, preparation_work = None, None, None, None
    ledgers, prepared, banks, summaries, operations = [], {}, {}, {}, {}
    active_evaluator, active_arm = None, None
    initial_digest, runtime = None, None

    def boundary():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed 5400-second process allowance expired; no automatic continuation")

    def event(value):
        journal.event(value)

    def operation(kind, identity, call, *, timed=True):
        if timed:
            boundary()
        stats = operations.setdefault(kind, dict(attempts=0, completions=0, failures=0, wall_seconds=0., cpu_seconds=0.))
        stats["attempts"] += 1
        receipt["active_operation"] = dict(kind=kind, identity=identity)
        t0, c0 = time.monotonic(), time.process_time()
        try:
            event(dict(event="operation_intent", kind=kind, identity=identity))
            value = call()
            stats["completions"] += 1
            event(dict(event="operation_complete", kind=kind, identity=identity,
                       wall_seconds=time.monotonic()-t0, cpu_seconds=time.process_time()-c0))
            receipt["active_operation"] = None
            return value
        except BaseException:
            stats["failures"] += 1
            raise
        finally:
            stats["wall_seconds"] += time.monotonic()-t0
            stats["cpu_seconds"] += time.process_time()-c0

    def authenticate():
        if source_pins() != launch["source_sha256"]:
            raise ValueError("study source identity changed")
        verify_pins(launch["input_sha256"])
        for name, local in launch["source_snapshots"].items():
            expected = launch["source_sha256"].get(name, launch["input_sha256"].get(name))
            if digest(output/local) != expected:
                raise ValueError("frozen source/receipt copy changed")

    def evaluate(spec):
        nonlocal active_evaluator
        from experiments import foundation_layout_evaluation as evaluation
        from experiments import foundation_layout_data as data
        from experiments.sequence_student import SequenceConfig
        raw_panels, group_artifacts = defaultdict(list), {}
        for group_id, group in sorted(launch["bank_inventory"].items()):
            if group["role"] not in spec["roles"]:
                continue
            role, view, panel = group["role"], group["view"], group["panel"]
            if role not in banks:
                if role != "audit" or spec["step"] != 864:
                    raise ValueError("audit loading is restricted to the fixed endpoint")
                loaded = operation("bank_load", role, lambda: counted_data_load(lambda: data.load_banks(
                    ROOT/launch["data_directory"], role, expected_manifest_sha256=launch["data_manifest_sha256"]), receipt))
                observed = bank_inventory({role: loaded["banks"]})
                expected = {key: value for key, value in launch["bank_inventory"].items() if value["role"] == role}
                if observed != expected:
                    raise ValueError("endpoint bank census differs from frozen launch")
                banks[role] = loaded["banks"]
                del loaded
            if group_id not in prepared:
                rows = [row for name in group["cells"] for row in banks[role][view][name]]
                if _hash(rows) != group["rows_sha256"]:
                    raise ValueError("grouped evaluation rows changed")
                receipt["prepared_bank_attempts"] += 1
                prepared[group_id] = operation("evaluation_preparation", group_id,
                    lambda: evaluation.PreparedLayoutBank(rows, role=role, config=SequenceConfig(**CONFIG), validation_work=preparation_work))
                receipt["prepared_banks"] += 1
                if prepared[group_id].identity["sha256"] != group["rows_sha256"]:
                    raise ValueError("compiled bank identity differs")
            active_evaluator = prepared[group_id]
            ledger = evaluation.EvaluationLedger()
            ledgers.append(dict(evaluation_id=spec["id"], group_id=group_id, ledger=ledger))
            result = operation("evaluation", spec["id"]+"/"+group_id,
                lambda: active_evaluator.score(trainer.model, batch_size=BATCH, control=spec["control"], deadline=deadline,
                    work=ledger, progress=lambda value: event(dict(evaluation_id=spec["id"], group_id=group_id, **value))))
            if result["status"] != "completed" or result["model_state_unchanged"] is not True or result["training_modes_restored"] is not True:
                raise ValueError("evaluation did not complete with learner state intact")
            _metric_denominators(result["metrics"], group["denominators"])
            recount = evaluation.score_records(result["raw_records"], expected_bank=result["bank"])
            if recount != result["metrics"]:
                raise ValueError("raw metric recount differs")
            cell_rows = defaultdict(list)
            for row in result["raw_records"]:
                cell_rows[group["cell_by_episode"][row["episode_id"]]].append(row)
            if set(cell_rows) != set(group["cells"]):
                raise ValueError("scoring lost an original evaluation cell")
            cells = {name: evaluation.score_records(rows) for name, rows in sorted(cell_rows.items())}
            for name, metrics in cells.items():
                _metric_denominators(metrics, group["cell_denominators"][name])
            name = "evaluations/"+spec["id"]+"/"+group_id+".json"
            group_artifacts[group_id] = operation("evaluation_publication", name,
                lambda: publish(attempt/name, dict(spec=spec, group_id=group_id, cell_by_episode=group["cell_by_episode"],
                    cell_metrics=cells, result=result)))
            raw_panels[(role, view, panel)].extend(result["raw_records"])
            active_evaluator = None
        by_role = {}
        for (role, view, panel), rows in sorted(raw_panels.items()):
            by_role.setdefault(role, {}).setdefault(view, {})[panel] = dict(metrics=evaluation.score_records(rows))
        value = dict(spec=spec, by_role=by_role, group_artifacts_sha256=group_artifacts)
        summary_name = "evaluations/"+spec["id"]+".json"
        operation("evaluation_summary_publication", summary_name, lambda: publish(attempt/summary_name, value))
        summaries[spec["id"]] = value
        receipt["completed_evaluations"].append(spec["id"])

    def snapshot(arm, step):
        from brain_in_computer.dialogue_student import checkpoint_digest
        if receipt["snapshot_attempts"] >= 14:
            raise ValueError("snapshot construction budget exceeded")
        receipt["snapshot_attempts"] += 1
        saved = operation("snapshot_cpu_copy", f"{arm}/{step}", trainer.snapshot)
        receipt["completed_full_snapshots"] += 1
        if saved["cursor"] != step or saved["evidence"] != accumulators[arm]:
            raise ValueError("checkpoint committed evidence differs from prescribed prefix")
        weights = operation("parameter_digest", f"{arm}/{step}", lambda: checkpoint_digest(trainer.model))
        if step == 0 and weights != initial_digest:
            raise ValueError("fresh initial snapshot parameter identity differs")
        payload = dict(schema=SCHEMA, arm=arm, step=step, launch_sha256=expected_launch_sha256,
                       learner=saved, weights_sha256=weights, execution_profile=runtime)
        receipt["checkpoint_publication_attempts"] += 1
        name = f"checkpoints/{arm}-{step:06d}.pt"
        archive = operation("checkpoint_publication", name, lambda: publish(attempt/name, payload, checkpoint=True))
        receipt["checkpoint_publications"] += 1
        receipt["arms"][arm]["checkpoints"][str(step)] = dict(path=name, archive_sha256=archive, weights_sha256=weights)
        receipt["arms"][arm]["last_published_checkpoint"] = step
        print(json.dumps(dict(event="checkpoint_complete", arm=arm, step=step, elapsed_seconds=time.monotonic()-started)), flush=True)

    def total_evaluation_work():
        if not ledgers:
            return {}
        totals = {key: sum(item["ledger"].counts[key] for item in ledgers) for key in ledgers[0]["ledger"].counts}
        totals.update(unmatched_sequence_attempts=totals["sequence_forward_attempts"]-totals["sequence_forward_completions"],
            unmatched_decoder_attempts=totals["decoder_recurrent_attempts"]-totals["decoder_recurrent_completions"])
        return totals

    try:
        journal = Journal(attempt/"events.jsonl")
        operation("authentication", "initial", authenticate)
        boundary()
        import torch
        from experiments import execution_profile as execution
        from experiments import foundation_layout_data as data, foundation_layout_training as training
        from experiments.foundation_layout_curriculum import WorkLedger
        from experiments.sequence_student import SequenceConfig
        from brain_in_computer.dialogue_student import checkpoint_digest
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("strict FP32 requires the unmodified float32 default")
        execution.configure_strict_profile()
        runtime = operation("runtime_setup", "cuda:0", lambda: execution.runtime_profile("cuda:0"))
        if runtime != launch["expected_runtime"]:
            raise ValueError("strict local runtime differs from frozen context runtime")
        receipt["execution_profile"] = runtime
        torch.cuda.reset_peak_memory_stats()
        data_work, preparation_work = data.WorkLedger(), WorkLedger()
        data_work.deadline, data_work.phase = deadline, "runtime_materialization"
        sealed = operation("data_load", "plan/dev/train_fit", lambda: counted_data_load(lambda: data.load(
            ROOT/launch["data_directory"], expected_manifest_sha256=launch["data_manifest_sha256"], roles=("dev", "train_fit")), receipt))
        banks.update(sealed["banks"])
        observed = bank_inventory(banks)
        expected = {key: value for key, value in launch["bank_inventory"].items() if value["role"] != "audit"}
        if observed != expected or sealed["plan"]["schedules"]["curriculum"] != list(range(864)):
            raise ValueError("loaded bank census or canonical schedule differs")
        plan, expected_bundles = sealed["plan"], sealed["expected_bundles"]
        del sealed
        accumulators = {arm: training._initial_evidence() for arm in ARMS}
        for arm in ARMS:
            active_arm = arm
            arm_started, arm_cpu = time.monotonic(), time.process_time()
            receipt["arms"][arm] = dict(status="running", checkpoints={}, last_published_checkpoint=None)
            boundary(); execution.assert_strict_profile()
            if receipt["model_construction_attempts"] >= 2:
                raise ValueError("fresh model construction budget exceeded")
            receipt["model_construction_attempts"] += 1
            trainer = operation("model_construction", arm, lambda: training.FoundationLayoutTrainer(
                seed=SEED, config=SequenceConfig(**CONFIG), learning_rate=RATE, micro_batch_size=MICRO,
                layout=arm, objective_id=training.OBJECTIVE_ID, device="cuda:0"))
            receipt["model_constructions"] += 1
            if any(value.dtype != torch.float32 or value.device.type != "cuda" for value in trainer.model.state_dict().values()):
                raise ValueError("fresh production learner must contain only FP32 CUDA weights")
            actual = operation("parameter_digest", arm+"/initial", lambda: checkpoint_digest(trainer.model))
            if initial_digest is None:
                initial_digest = actual
                receipt["initial_weights_sha256"] = actual
            elif actual != initial_digest:
                raise ValueError("paired arms do not share exact seeded initial weights")
            snapshot(arm, 0)
            if arm == ARMS[0]:
                evaluate(evaluation_schedule()[0])
            for bundle_id in range(864):
                boundary(); execution.assert_strict_profile()
                bundle = operation("bundle_materialization", f"{arm}/{bundle_id}",
                    lambda: data.materialize_bundle(plan, bundle_id, arm, work=data_work))
                frozen = expected_bundles["bundles"][arm][str(bundle_id)]
                if data_work.last_bundle_evidence != frozen:
                    raise ValueError("materialized bundle differs from sealed evidence before forward")
                report = operation("trainer_step", f"{arm}/{bundle_id}", lambda: trainer.step(bundle))
                del bundle
                accumulators[arm] = training._accumulate(accumulators[arm], frozen)
                if report["bundle_evidence"] != frozen or trainer._evidence != accumulators[arm] or trainer.cursor != bundle_id+1:
                    raise ValueError("committed trainer evidence differs from sealed bundle prefix")
                event(dict(event="retained_step", arm=arm, step=trainer.cursor, report=report,
                    consumed_bundle_identity_sha256=accumulators[arm]["consumed_bundle_identity_sha256"]))
                if trainer.cursor % 48 == 0:
                    print(json.dumps(dict(event="training_progress", arm=arm, updates=trainer.cursor,
                                          elapsed_seconds=time.monotonic()-started)), flush=True)
                if trainer.cursor in STEPS:
                    snapshot(arm, trainer.cursor)
                    for spec in evaluation_schedule():
                        if spec["arm"] == arm and spec["step"] == trainer.cursor:
                            evaluate(spec)
            if trainer._evidence != expected_bundles["final_arm_evidence"][arm]:
                raise ValueError("complete arm differs from frozen final input evidence")
            receipt["arms"][arm].update(status="completed", evidence=deepcopy(trainer._evidence), accounting=trainer.accounting,
                wall_seconds=time.monotonic()-arm_started, cpu_seconds=time.process_time()-arm_cpu)
            receipt["completed_arms"].append(arm)
            trainer = None
            operation("arm_shutdown", arm, lambda: (gc.collect(), torch.cuda.empty_cache()))
            active_arm = None
        operation("authentication", "final", authenticate)
        execution.assert_strict_profile()
        if runtime != execution.runtime_profile("cuda:0"):
            raise ValueError("runtime drifted during the comparison")
        budget, total = launch["expected_work"], total_evaluation_work()
        exact = dict(sequence_forward_attempts=budget["sequence_forward_calls"], sequence_forward_completions=budget["sequence_forward_calls"],
            batch_intents=budget["sequence_forward_calls"], completed_batches=budget["sequence_forward_calls"],
            encoder_episode_attempts=budget["encoder_episode_passes"], encoder_episode_completions=budget["encoder_episode_passes"],
            encoder_actual_position_attempts=budget["encoder_actual_positions"], encoder_actual_position_completions=budget["encoder_actual_positions"],
            encoder_padded_position_attempts=budget["encoder_padded_positions"], encoder_padded_position_completions=budget["encoder_padded_positions"],
            bos_reply_context_attempts=budget["bos_reply_contexts"], bos_reply_context_completions=budget["bos_reply_contexts"],
            bos_decoder_recurrent_attempts=budget["sequence_forward_calls"], bos_decoder_recurrent_completions=budget["sequence_forward_calls"],
            bos_decoder_row_step_attempts=budget["bos_reply_contexts"], bos_decoder_row_step_completions=budget["bos_reply_contexts"],
            free_reply_context_attempts=budget["free_reply_contexts"], free_reply_context_completions=budget["free_reply_contexts"],
            free_decoder_calls_attempted=budget["sequence_forward_calls"], free_decoder_calls_completed=budget["sequence_forward_calls"],
            scored_episode_records=budget["scored_episode_records"], unmatched_sequence_attempts=0, unmatched_decoder_attempts=0)
        if any(total[key] != value for key, value in exact.items()):
            raise ValueError("evaluation work differs from complete prospective count inventory")
        if (total["decoder_recurrent_attempts"] != total["decoder_recurrent_completions"]
                or total["decoder_row_step_attempts"] != total["decoder_row_step_completions"]
                or not budget["sequence_forward_calls"] <= total["decoder_recurrent_completions"] <= budget["free_decoder_recurrent_calls_max"]
                or not budget["free_reply_contexts"] <= total["decoder_row_step_completions"] <= budget["free_decoder_row_steps_max"]):
            raise ValueError("free decoder work exceeds declared bounds")
        training_totals = {key: sum(receipt["arms"][arm]["accounting"]["work"][key] for arm in ARMS) for key in training.WORK_COUNTS}
        expected_training = dict(step_invocations=1728, attempted_forwards=5184, completed_forwards=5184,
            attempted_backwards=5184, completed_backwards=5184, optimizer_attempts=1728, optimizer_returns=1728,
            synchronized_optimizer_updates=1728, unknown_optimizer_outcomes=0, retained_updates=1728, retained_episodes=165888)
        if (any(training_totals[key] != value for key, value in expected_training.items())
                or receipt["model_constructions"] != 2 or receipt["checkpoint_publications"] != 14
                or receipt["completed_full_snapshots"] != 14 or receipt["completed_arms"] != list(ARMS)
                or receipt["weights_only_bank_loads_attempted"] != 6 or receipt["weights_only_bank_loads_completed"] != 6
                or receipt["completed_evaluations"] != [spec["id"] for spec in evaluation_schedule()]):
            raise ValueError("completed teaching/checkpoint/evaluation coverage differs")
        screen = operation("descriptive_screen", "fixed_endpoint", lambda: summarize(summaries))
        operation("screen_publication", "screen.json", lambda: publish(attempt/"screen.json", screen))
        receipt.update(status="completed", screen_passed=screen["screen_passed"], training_work=training_totals,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        receipt.update(status="incomplete" if isinstance(error, TimeoutError) else "failed", error=repr(error),
            traceback=traceback.format_exc(), partial_work_unknown=True)
        if trainer is not None and active_arm is not None:
            receipt["arms"][active_arm].update(status=receipt["status"], accounting=trainer.accounting,
                evidence=deepcopy(trainer._evidence), last_step_report=deepcopy(trainer.last_report),
                last_snapshot_report=deepcopy(trainer.last_snapshot_report))
        if active_evaluator is not None:
            receipt["last_evaluation_report"] = deepcopy(active_evaluator.last_report)
        if "torch" in locals() and torch.cuda.is_initialized():
            t0 = time.monotonic()
            try:
                torch.cuda.synchronize()
                receipt["failure_device_synchronized"] = True
            except BaseException as sync_error:
                receipt["failure_device_synchronized"] = False
                receipt["failure_synchronization_error"] = repr(sync_error)
            receipt["failure_synchronization_wall_seconds"] = time.monotonic()-t0
        raise
    finally:
        t0, c0 = time.monotonic(), time.process_time()
        try:
            trainer = None
            prepared.clear(); banks.clear()
            gc.collect()
            if "torch" in locals() and torch.cuda.is_initialized():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                receipt.setdefault("peak_cuda_allocated_bytes", torch.cuda.max_memory_allocated())
                receipt.setdefault("peak_cuda_reserved_bytes", torch.cuda.max_memory_reserved())
        except BaseException as cleanup_error:
            receipt.update(status="failed", cleanup_error=repr(cleanup_error), partial_work_unknown=True)
        receipt["shutdown_wall_seconds"] = time.monotonic()-t0
        receipt["shutdown_cpu_seconds"] = time.process_time()-c0
        if journal is not None:
            try:
                journal.close()
            except BaseException as journal_error:
                receipt.update(status="failed", journal_close_error=repr(journal_error), partial_work_unknown=True)
            receipt["journal_events"], receipt["journal_chain_sha256"] = journal.sequence, journal.chain
        receipt.update(ended_utc=utc(), evaluation_work=total_evaluation_work(), operation_costs=operations,
            data_materialization_work=data_work.report() if data_work is not None else None,
            evaluation_preparation_work=preparation_work.report() if preparation_work is not None else None,
            evaluation_invocations=[dict(evaluation_id=item["evaluation_id"], group_id=item["group_id"], work=item["ledger"].report()) for item in ledgers],
            artifacts_sha256={path.relative_to(native(attempt)).as_posix(): digest(path) for path in native(attempt).rglob("*") if path.is_file()},
            timing_scope="Run entry through authentication, loads, runtime setup, construction, materialization, repeated validation, learning, evaluation, checkpoints, screen, shutdown and artifact hashing; final receipt publication excluded.")
        receipt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                       deadline_overrun_seconds=max(0., time.monotonic()-deadline))
        publish(attempt/"receipt.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError("study cleanup/publication failed; preserved receipt is terminal")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--expected-data-manifest-sha256")
    parser.add_argument("--expected-data-preparation-sha256")
    parser.add_argument("--expected-data-process-sha256")
    parser.add_argument("--screen-validation", type=Path)
    parser.add_argument("--expected-screen-validation-sha256")
    parser.add_argument("--expected-launch-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        if (args.data is None or not args.expected_data_manifest_sha256 or not args.expected_data_preparation_sha256
                or not args.expected_data_process_sha256 or args.screen_validation is None or not args.expected_screen_validation_sha256):
            parser.error("prepare requires sealed data, manifest/internal/enclosing receipt SHA256 pins and an explicit pure-screen receipt/pin")
        value = prepare(args.output, args.data, expected_data_manifest_sha256=args.expected_data_manifest_sha256,
                        expected_data_preparation_sha256=args.expected_data_preparation_sha256,
                        expected_data_process_sha256=args.expected_data_process_sha256, screen_validation=args.screen_validation,
                        expected_screen_validation_sha256=args.expected_screen_validation_sha256)
    else:
        if not args.expected_launch_sha256:
            parser.error("run requires the independently reviewed launch SHA256")
        value = run(args.output, expected_launch_sha256=args.expected_launch_sha256)
    print(json.dumps({key: value[key] for key in ("status", "wall_seconds", "cpu_seconds")}), flush=True)


if __name__ == "__main__":
    main()
