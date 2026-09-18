"""Pure JSON evidence fixtures; no curriculum, model, optimizer or GPU work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import summarize_foundation_pilot as summary


CONFIG = {"width": 8, "layers": 1, "heads": 2, "max_turns": 12}


def metric(name, role="dev", control="normal", episodes=2, perfect=True):
    turns = int(name.rsplit("t", 1)[1])
    n, k = episodes, episodes if perfect else 0
    q = n+k
    by_turn = [{"turn": t, "correct": n if t == turns-2 else k if t == turns-1 else 0,
        "total": n if t >= turns-2 else 0, "accuracy": 1. if t == turns-2 else k/n if t == turns-1 else None,
        "opposite_pair_total": n//2 if t == turns-1 else 0,
        "opposite_pair_correct": k//2 if t == turns-1 else 0} for t in range(turns)]
    return {"episodes": n, "turns_per_episode": turns, "query_correct": q, "query_total": 2*n,
        "query_accuracy": q/(2*n), "known_correct": k, "known_total": n, "known_accuracy": k/n,
        "final_correct": k, "final_total": n, "final_accuracy": k/n,
        "final_pairs": {"correct": k//2, "total": n//2}, "final_pair_accuracy": k/n,
        "opposite_pair_correct": k//2, "opposite_pair_total": n//2, "opposite_pair_accuracy": k/n,
        "query_reply_correct": q, "query_reply_accuracy": q/(2*n),
        "final_reply_pair_correct": k//2, "final_reply_pair_accuracy": k/n,
        "ask_true": n, "ask_correct": n, "ask_predicted": n, "ask_recall": 1., "ask_precision": 1.,
        "action_reply_agreement": 1., "reply_parseable_queries": 2*n, "query_loss": .5, "brier_score": .5,
        "confusion_matrix": [[k//2, (n-k)//2, 0, 0], [(n-k)//2, k//2, 0, 0], [0, 0, n, 0], [0, 0, 0, 0]],
        "per_target": {str(t): {"total": n if t == 2 else n//2, "correct": n if t == 2 else k//2,
                                "accuracy": 1. if t == 2 else k/n} for t in range(3)},
        "by_turn": by_turn, "seconds": .001,
        "bank": {"sha256": summary.digest([name, role]), "version": summary.VERSION,
                 "role": role, "config": CONFIG, "episodes": n, "turns": turns},
        "control": control, "free_running_replies": True, "teacher_used_for_policy": False,
        "decoder_prefix": "BOS only"}


def metrics(rows):
    return {"per_bank": rows, **{"macro_"+key: sum(row[key] for row in rows.values())/len(rows) for key in summary.METRICS}}


def score(role, control="normal"):
    rows = {name: metric(name, role, control) for name in sorted(summary.expected_cells(role))}
    episodes = sum(row["episodes"] for row in rows.values())
    turns = sum(row["episodes"]*row["turns_per_episode"] for row in rows.values())
    return {"metrics": metrics(rows), "summary": {"not_used": "derived projection is recomputed"},
        "cost": {"scored_episodes": episodes, "scored_turns": turns, "free_reply_turns": turns,
            "encoder_episode_instances": turns if control == "reset" else episodes, "control": control,
            "wall_seconds": 1., "bank_score_seconds_included_in_wall": len(rows)*.001}}


def fixture(directory):
    """All binary-looking artifacts are inert text, intentionally never loaded."""
    def write(name, value):
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf8")
        return summary.file_hash(path)
    source_hash = write("repo/source.py", {"synthetic_source": True})
    write("source/source.py", {"synthetic_source": True})
    proof_artifact = write("proof/artifact.pt", {"inert": True})
    profile = {"strict": True}
    proof = {"status": "passed", "admission": {"mode": "naming_overrides", "executed_prefix_overridden_pairs": 2},
        "flags": {"exact": True}, "config": CONFIG, "learning_rate": .001,
        "execution_profile": profile, "artifact_sha256": {"artifact.pt": proof_artifact}}
    write("proof/probe.json", proof)
    admission = {role: {"banks": {name: {"sha256": row["bank"]["sha256"], "episodes": row["episodes"],
        "query_target_counts": {"0": 1, "1": 1, "2": 2}, "final_opposite_pair_total": 1}
        for name, row in score(role)["metrics"]["per_bank"].items()}} for role in ("dev", "audit")}
    files = {name: write(name, admission if name == "bank-admission.json" else {"synthetic": name})
        for name in ("plan.json", "lesson-admission.json", "training-manifest.json", "bank-admission.json",
                     "capacity-choice.json", "banks.pt", "protected.pt", "training-transcripts.pt")}
    contract = {"schema": "bic-foundation-order-pilot-v2", "version": summary.VERSION, "arms": list(summary.ARMS),
        "automatic_promotion": False, "evaluation_during_training": False, "checkpoints": [0, 2], "updates": 2,
        "plan_options": {"micro_batch_size": 2}, "pairs_per_evaluation_cell": 1}
    protocol = {"contract": contract, "source_sha256": {"source.py": source_hash}, "files_sha256": files,
        "capacity_choice": {"files_sha256": {}}, "proof_directory": str(directory/"proof"),
        "proof_sha256": summary.digest(proof), "config": CONFIG, "learning_rate": .001,
        "execution_profile": profile, "initial_weights_sha256": "1"*64}
    p_hash = write("protocol.json", protocol)
    prep_hash = write("preparation.json", {"status": "prepared", "protocol_sha256": p_hash,
        "neural_training_or_inference": False, "wall_seconds": 2.})
    inputs = {"protocol.json": p_hash, "preparation.json": prep_hash, **files}
    arms = {}
    for arm in summary.ARMS:
        journal = write(f"{arm}/steps.jsonl", {"inert_verified_journal": arm})
        checkpoints = {f"checkpoint-{step:06d}.pt": write(f"{arm}/checkpoint-{step:06d}.pt", {"inert": [arm, step]}) for step in (0, 2)}
        counts = {key: 2 if key == summary.PHYSICAL[0] else 12 for key in summary.PHYSICAL}
        receipt = {"schema": contract["schema"], "arm": arm, "status": "completed", "retained_updates": 2,
            "automatic_promotion": False, "physical_work_unknown": False, "checkpoints": checkpoints,
            "protocol_sha256": p_hash, "journal_sha256": journal, "execution_profile": profile,
            **counts, "wall_seconds": 1., "setup_seconds": .1, "peak_cuda_allocated_mib": 20., "peak_cuda_reserved_mib": 30.}
        inputs[f"{arm}/receipt.json"] = write(f"{arm}/receipt.json", receipt)
        inputs[f"{arm}/steps.jsonl"] = journal
        inputs.update({f"{arm}/{key}": value for key, value in checkpoints.items()})
        arms[arm] = {"receipt": receipt, "weights_sha256": {"0": "1"*64, "2": ("2" if arm == "curriculum" else "3")*64},
            "exact_official_checkpoint_restores": True,
            "journal": {**counts, "retained_step_seconds": .2,
                        "materialization_seconds_included_in_step": .05, "step_invocation_seconds": .3},
            "endpoint_evidence": {"exposures": {family: {"episodes": 4} for family in ("color", "count", "switch")}}}
    verified = {"schema": summary.RESULTS_SCHEMA, "status": "completed", "protocol_sha256": p_hash,
        "input_file_sha256": inputs, "source_sha256": protocol["source_sha256"], "proof_sha256": protocol["proof_sha256"],
        "neural_training_or_inference": False, "automatic_promotion": False, "arms": arms, "wall_seconds": 3.}
    v_hash = write("verification.json", verified)
    binding = {"schema": summary.RESULTS_SCHEMA, "status": "completed", "protocol_sha256": p_hash,
        "verification_sha256": v_hash, "input_file_sha256": inputs, "execution_profile": profile,
        "automatic_promotion": False}
    outcomes, hashes = {}, {}
    for arm in summary.ARMS:
        curve = [{"updates": step, "weights_sha256": arms[arm]["weights_sha256"][str(step)], **score("dev")} for step in (0, 2)]
        fit, audit = score("train_fit"), score("audit")
        controls = {control: score("audit", control) for control in ("blank", "reset")}
        pieces = [*curve, fit, audit, *controls.values()]
        costs = {key: sum(piece["cost"][key] for piece in pieces) for key in summary.SCORE_COUNTS}
        costs["score_wall_seconds_included_in_arm_wall"] = 6.
        outcomes[arm] = {**binding, "arm": arm, "development_curve": curve, "train_fit": fit, "audit": audit,
                        "controls": controls, "cost": costs, "wall_seconds": 6.5}
        hashes[arm] = write(f"evaluation/{arm}.json", outcomes[arm])
    report = {**binding, "results": outcomes, "base_checkpoints_unchanged": True, "preparation_seconds": 1.,
              "wall_seconds": 15., "arm_files_sha256": hashes}
    write("evaluation/report.json", report)
    return protocol


class FoundationCompletionSummaryTests(unittest.TestCase):
    def test_projection_keeps_macro_and_pooled_counts_distinct_and_preserves_cells(self):
        names = ("fresh/color/d0/direct/t8", "fresh/color/d0/direct/t10", "fresh/count/d1/copy/t12")
        rows = {names[0]: metric(names[0]), names[1]: metric(names[1], episodes=6, perfect=False), names[2]: metric(names[2])}
        before = deepcopy(rows)
        result = summary.project_metrics(metrics(rows), "dev", "normal", CONFIG)
        group = result["by_panel_family_mechanism"]["fresh/color/direct"]
        self.assertEqual(group["macro_equal_cells"]["final_pair_accuracy"], .5)
        self.assertEqual(group["pooled_counts"]["final_action_pairs"], {"correct": 1, "total": 4})
        self.assertEqual(group["cell_names"], sorted(names[:2]))
        self.assertEqual(set(result["cells"]), set(names))
        self.assertEqual(result["cells"][names[1]]["by_turn"], rows[names[1]]["by_turn"])
        self.assertEqual(rows, before)

    def test_actual_role_grid_has_separate_primitives_and_no_phantom_depth_two_dev(self):
        self.assertEqual([len(summary.expected_cells(role)) for role in ("train_fit", "dev", "audit")], [63, 90, 99])
        self.assertNotIn("composed/count/d2/composed/t8", summary.expected_cells("dev"))
        for operator in ("copy", "advance"):
            self.assertIn(f"fresh/switch/d1/{operator}/t12", summary.expected_cells("dev"))

    def test_projection_rejects_corrupt_counts_boundary_and_macro(self):
        name = "fresh/color/d0/direct/t8"
        for change in ({"final_correct": 3}, {"query_correct": True}, {"ask_predicted": 99},
                       {"final_reply_pair_correct": 2}, {"known_total": 1},
                       {"decoder_prefix": "answers"}, {"free_running_replies": False}):
            row = metric(name); row.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.project_metrics(metrics({name: row}), "dev", "normal", CONFIG)
        values = metrics({name: metric(name)}); values["macro_final_pair_accuracy"] = .2
        with self.assertRaisesRegex(ValueError, "macro"):
            summary.project_metrics(values, "dev", "normal", CONFIG)

    def test_complete_json_fixture_projects_every_curve_and_writes_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); fixture(directory)
            with patch.object(summary, "ROOT", directory/"repo"):
                result = summary.write_summary(directory)
                with self.assertRaises(FileExistsError): summary.write_summary(directory)
            self.assertFalse(result["automatic_promotion"])
            self.assertFalse(result["integrity"]["neural_training_or_inference"])
            self.assertFalse(result["integrity"]["canonical_replay_repeated"])
            self.assertEqual(result["cost"]["retained_optimizer_updates"], 4)
            self.assertEqual(result["cost"]["completed_microbatch_episode_exposures"], 24)
            for arm in summary.ARMS:
                value = result["arms"][arm]
                self.assertEqual([point["updates"] for point in value["development_curve"]], [0, 2])
                self.assertEqual(len(value["endpoint_audit"]["cells"]), 99)
                self.assertEqual(set(value["endpoint_controls"]), {"blank", "reset"})
                self.assertEqual(value["training_cost"]["journal"]["materialization_seconds_included_in_step"], .05)

    def test_incomplete_gate_and_modified_bound_artifact_fail_before_projection(self):
        for changed in ("evaluation/report.json", "mixed/checkpoint-000002.pt", "source/source.py", "proof/artifact.pt"):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); fixture(directory)
                path = directory/changed
                if changed == "evaluation/report.json": path.unlink()
                else: path.write_text("changed", encoding="utf8")
                with patch.object(summary, "ROOT", directory/"repo"), \
                     patch.object(summary, "project_metrics", side_effect=AssertionError("must not project incomplete evidence")):
                    with self.subTest(path=changed), self.assertRaises(ValueError): summary.summarize(directory)
                self.assertFalse((directory/"evaluation/summary.json").exists())

    def test_embedded_arm_and_official_curve_bindings_cannot_be_silently_changed(self):
        for mutate in (lambda report: report["results"]["mixed"].update(status="running"),
                       lambda report: report["results"]["mixed"]["development_curve"].pop(),
                       lambda report: report["results"]["mixed"]["development_curve"][0].update(weights_sha256="0"*64)):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); fixture(directory)
                path = directory/"evaluation/report.json"; report = summary.read(path); mutate(report)
                path.write_text(json.dumps(report), encoding="utf8")
                with patch.object(summary, "ROOT", directory/"repo"), self.assertRaises(ValueError):
                    summary.load_complete(directory)

    def test_scoring_exposure_and_nested_cost_corruption_fail(self):
        protocol = {"config": CONFIG, "contract": {"pairs_per_evaluation_cell": 1}}
        for change in ({"scored_episodes": True}, {"encoder_episode_instances": 1}, {"wall_seconds": .0001}):
            piece = score("train_fit"); piece["cost"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary._score_piece(piece, "train_fit", "normal", protocol, {}, {})
        piece = score("train_fit"); piece["metrics"]["per_bank"].pop(next(iter(piece["metrics"]["per_bank"])))
        with self.assertRaisesRegex(ValueError, "omitted"):
            summary._score_piece(piece, "train_fit", "normal", protocol, {}, {})

    def test_nonfinite_json_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"bad.json"; path.write_text('{"cost":NaN}', encoding="utf8")
            with self.assertRaises(ValueError): summary.read(path)
            with self.assertRaises(ValueError): summary.relative(Path(temporary), "../outside")

    def test_helper_source_change_during_projection_prevents_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); fixture(directory)
            actual = summary.file_hash
            helper_reads = []
            def changed(path):
                if Path(path).resolve() == Path(summary.__file__).resolve():
                    helper_reads.append(path)
                    return ("1" if len(helper_reads) == 1 else "2")*64
                return actual(path)
            with patch.object(summary, "ROOT", directory/"repo"), patch.object(summary, "file_hash", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "helper source changed"):
                    summary.write_summary(directory)
            self.assertEqual(len(helper_reads), 2)
            self.assertFalse((directory/"evaluation/summary.json").exists())


if __name__ == "__main__": unittest.main()
