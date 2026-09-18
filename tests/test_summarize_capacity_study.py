"""Synthetic summary arithmetic, corruption and strict completion gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import summarize_capacity_study as summary
from experiments.composition_evaluation import METRICS
from experiments.realization_training import COUNTS, stream_evidence
from tests.test_capacity_evidence import audit_fixture


def scores(value=.25,panels=("name_only","value_only","both","composed")):
    rows={}
    for panel in panels:
        for family in summary.study.FAMILIES:
            for turns in (8,10,12):
                row={**dict.fromkeys(METRICS,value),"query_correct":int(320*value),"query_total":320,
                    "known_correct":int(256*value),"known_total":256,
                    "final_pairs":{"correct":int(64*value),"total":64},
                    "final_reply_pair_correct":int(64*value),"ask_precision":None,"ask_recall":None,
                    "action_reply_agreement":.9,"query_loss":.4,"brier_score":.3,
                    "confusion_matrix":[[64,56,8,0],[56,64,8,0],[0,0,64,0],[0,0,0,0]],
                    "by_turn":[{"opposite_pair_total":64}]}
                rows[f"{panel}/{family}/t{turns}"]=row
    return {"per_bank":rows,**{f"macro_{key}":value for key in METRICS}}


def fixture():
    protocol={"schema":summary.study.SCHEMA}
    stages={}
    for stage in summary.study.TOTALS:
        steps=summary.study.TOTALS[stage]
        exposures={f:dict(zip(COUNTS,(steps*32,steps*320,steps*10240,steps*9600,steps*3200,steps*2800))) for f in summary.study.FAMILIES}
        stream={"exposures":exposures,"canonical_stream_sha256":"same"+stage,
                "occurrences":{"color":{8:[steps]}},"seen_transcript_count":steps*96}
        stages[stage]={}
        for job,spec in summary.study.jobs(stage).items():
            value={96:.25,192:.5,256:.75}[spec["width"]]
            stages[stage][job]={"recipe":dict(spec),"updates":steps,"exposures":copy.deepcopy(exposures),
                "stream":copy.deepcopy(stream),"retained_step_seconds":40.,"weights_sha256":job,
                "generation_timing_subset":{"generation_validation_seconds":3.,"rejected_candidate_seconds":.1},
                "development_curve":[{"updates":step,"episodes_per_family":step*32,"seconds":2.,
                    "weights_sha256":job,"metrics":scores(value)} for step in summary.study.STEPS[stage]],
                "invocation_costs":{"recorded_worker_wall_seconds":60.,"recorded_commit_seconds":5.,
                    "recorded_discarded_step_seconds":0.,"known_discarded_completed_updates":0,
                    "unrecorded_physical_work_possible":False,"recorded_setup_seconds":10.,"recorded_post_setup_wall_seconds":50.}}
    audits={}
    for job,spec in summary.study.jobs("main").items():
        value={96:.25,192:.5,256:.75}[spec["width"]]
        curve=[{"updates":step,"episodes_per_family":step*32,"weights_sha256":job,"metrics":scores(value)} for step in summary.study.STEPS["main"]]
        audits[job]={"curve":curve,"final":curve[-1]["metrics"],"initial_fit":scores(value,panels=("initial",)),
            "latest_observed_fit":scores(value,panels=("latest",)),"latest_observed_metadata":{
                "all_initial_realizations_encountered":True,"absent_recipes":[],"per_pair":[{"occurrence":4}]},
            "controls":{control:scores(0.) for control in ("blank","reset")},"cpu_restart":{"exact":True},"seconds":7.}
    return protocol,{"selected":{}},stages,{"results":audits}


class CapacitySummaryArithmeticTests(unittest.TestCase):
    def test_all_calibration_and_main_costs_without_nested_double_counting(self):
        result=summary.assemble_summary(*fixture())
        costs=result["compute"]
        self.assertEqual(costs["calibration"]["jobs"],9)
        self.assertEqual(costs["main"]["jobs"],3)
        self.assertEqual(costs["total"]["retained_optimizer_updates"],27000)
        self.assertEqual(costs["total"]["sampled_exposures"]["episodes"],2592000)
        self.assertEqual(costs["total"]["retained_worker_step_seconds"],480.)
        self.assertEqual(costs["total"]["generation_validation_seconds_already_in_steps"],36.)
        self.assertEqual(costs["total"]["recorded_development_seconds"],66.)
        self.assertEqual(costs["total"]["recorded_invocation_wall_seconds"],720.)
        self.assertEqual(costs["audit_worker_seconds"],21.)
        self.assertEqual(costs["total"]["recorded_setup_seconds"],120.)
        self.assertEqual(costs["total"]["recorded_post_setup_wall_seconds"],600.)
        self.assertFalse(result["automatic_promotion"])
        self.assertIsNone(result["selected_checkpoint"])

    def test_counts_unknown_nulls_brier_curves_and_absolute_width_differences(self):
        result=summary.assemble_summary(*fixture())
        row=result["audit"]["w192"]["final"]["per_panel_family"]["composed"]["color"]
        self.assertEqual(row["per_length"]["t8"]["final_pairs"],{"correct":32,"total":64})
        self.assertEqual(row["pooled_counts"]["final_reply_pairs"],{"correct":96,"total":192})
        self.assertEqual(row["pooled_counts"]["unsupported_ask_count"],48)
        self.assertEqual(row["per_length"]["t8"]["unsupported_ask_rate"],.0625)
        self.assertIsNone(row["macro"]["ask_recall"])
        self.assertAlmostEqual(row["macro"]["brier_score"],.3)
        self.assertEqual(len(result["audit"]["w192"]["curve"]),5)
        differences=result["width_differences"]
        self.assertEqual(set(differences),{"w192_minus_w96","w256_minus_w96","w256_minus_w192"})
        self.assertEqual(len(differences["w192_minus_w96"]["curve"]),5)
        self.assertEqual(differences["w256_minus_w96"]["endpoint"]["both"]["switch"]["per_length"]["t12"]["final_pair_accuracy"],.5)
        self.assertEqual(differences["w192_minus_w96"]["areas_by_family"]["count"]["composed"]["final_pair_accuracy"],.25)

    def test_fit_coverage_is_not_replaced_with_complete_coverage_claim(self):
        args=fixture();args[3]["results"]["w192"]["latest_observed_metadata"].update(
            all_initial_realizations_encountered=False,absent_recipes=[{"family":"count","turns":12,"recipe_index":3}])
        result=summary.assemble_summary(*args)
        metadata=result["audit"]["w192"]["latest_observed_metadata"]
        self.assertFalse(metadata["all_initial_realizations_encountered"])
        self.assertEqual(len(metadata["absent_recipes"]),1)
        self.assertIn("not an unbiased",result["fit_note"])
        self.assertIn("without significance",result["comparison_note"])

    def test_input_immutability_and_json_bucket_keys(self):
        args=fixture();before=copy.deepcopy(args)
        args[2]["main"]["w192"]["stream"]=json.loads(json.dumps(args[2]["main"]["w192"]["stream"]))
        result=summary.assemble_summary(*args)
        self.assertEqual(args[2]["main"]["w96"],before[2]["main"]["w96"])
        self.assertNotIn("unsupported_ask_count",next(iter(args[3]["results"]["w96"]["final"]["per_bank"].values())))
        self.assertNotIn("occurrences",result["stages"]["main"]["w192"]["stream"])

    def test_stream_change_missing_job_timing_and_curve_corruption_rejected(self):
        for kind in ("stream","job","timing","curve"):
            args=fixture()
            if kind=="stream":args[2]["main"]["w192"]["stream"]["canonical_stream_sha256"]="changed"
            elif kind=="job":del args[2]["calibration"]["w96-r001"]
            elif kind=="timing":args[2]["main"]["w192"]["generation_timing_subset"]["generation_validation_seconds"]=100.
            else:args[3]["results"]["w192"]["curve"].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):summary.assemble_summary(*args)


class CapacitySummaryBoundaryTests(unittest.TestCase):
    def test_endpoint_optimizer_cursor_and_stream_are_bound_not_just_weights(self):
        final={"learner":{"weights":{"w":torch.tensor([1.])},"optimizer":{"moment":torch.tensor([.1])},
            "realization":{"seen_transcripts":[],"samplers":{"color":torch.tensor([1],dtype=torch.uint8)},"canonical_stream_sha256":"a"*64}},
            "curriculum":{"cursor":3},"accounting":{"retained_step_seconds":1.}}
        replay=stream_evidence(final["learner"]["realization"])
        summary._validate_endpoint_binding(final,copy.deepcopy(final),replay)
        for field in ("optimizer","cursor","stream"):
            changed=copy.deepcopy(final)
            if field=="optimizer":changed["learner"]["optimizer"]["moment"]+=1
            elif field=="cursor":changed["curriculum"]["cursor"]+=1
            else:changed["learner"]["realization"]["canonical_stream_sha256"]="b"*64
            with self.subTest(field=field),self.assertRaises(ValueError):
                summary._validate_endpoint_binding(changed,final,replay)

    def test_cached_audit_is_revalidated_against_canonical_pairs(self):
        result,arguments=audit_fixture()
        summary._validated_audit(result,copy.deepcopy(result),**arguments)
        changed=copy.deepcopy(result)
        changed["final"]["macro_final_pair_accuracy"] = .2
        with self.assertRaises(ValueError):summary._validated_audit(result,changed,**arguments)
        changed=copy.deepcopy(result)
        for point in changed["curve"]:
            next(iter(point["metrics"]["per_bank"].values()))["by_turn"][-1]["opposite_pair_total"]=0
        changed["final"]=copy.deepcopy(changed["curve"][-1]["metrics"])
        # Mirroring bad counts into report/cache cannot replace canonical truth.
        with self.assertRaises(ValueError):summary._validated_audit(changed,copy.deepcopy(changed),**arguments)

    def test_every_completion_boundary_precedes_numerical_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for path in summary._required(root):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text("{}")
            missing=("calibration/w256-r003/report.json","main/w96/checkpoint-001800.pt",
                     "verification/calibration.json","selection.json","audit/w256.json","audit/report.json")
            for name in missing:
                path=root/name;path.unlink()
                with patch.object(summary.study,"load_protocol") as protocol,patch.object(summary,"_read") as read,patch.object(summary.torch,"load") as load:
                    for operation in (summary.summarize,summary.verify_completed):
                        with self.assertRaisesRegex(ValueError,"all9 calibration"):
                            operation(root)
                    protocol.assert_not_called();read.assert_not_called();load.assert_not_called()
                path.write_text("{}")
            self.assertFalse((root/"audit/summary.json").exists())

    def test_source_rejection_precedes_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for path in summary._required(root):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text("{}")
            with patch.object(summary.study,"load_protocol",side_effect=ValueError("source changed")),patch.object(summary.torch,"load") as load:
                with self.assertRaisesRegex(ValueError,"source changed"):summary.summarize(root)
                load.assert_not_called()

    def test_nonterminal_dispatcher_cannot_be_called_a_complete_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"execution.json").write_text(json.dumps({"schema":"bic-capacity-dispatch-v1","status":"running","protocol_sha256":"p"}))
            with self.assertRaisesRegex(ValueError,"dispatcher must be terminal"):
                summary._execution(root,{"protocol.json":"p"})
            (root/"execution.json").unlink()
            self.assertFalse(summary._execution(root,{"protocol.json":"p"})["available"])

    def test_invocation_costs_preserve_discarded_unknown_and_no_duplicate_failure_report(self):
        chunk={"completed_updates":16,"committed_updates":16,"discarded_completed_updates":0,"failed_step_attempts":0,
               "step_seconds":2.,"commit_seconds":.5,"rollback_seconds":0.,"discarded_step_seconds":0.}
        failed={"status":"failed","starting_updates":0,"ending_updates":16,"chunks":[chunk],
                "failure_report":copy.deepcopy(chunk),"wall_seconds":4.,"post_setup_wall_seconds":3.,"setup_seconds":1.,"peak_cuda_allocated_mib":10.,"setup_peak_cuda_allocated_mib":5.}
        lost={"status":"running","starting_updates":16,"chunks":[],"setup_seconds":1.}
        final={"status":"completed","starting_updates":16,"ending_updates":32,"chunks":[copy.deepcopy(chunk)],
               "wall_seconds":3.,"post_setup_wall_seconds":2.,"setup_seconds":1.,"peak_cuda_allocated_mib":20.,"setup_peak_cuda_allocated_mib":30.}
        result=summary.invocation_costs({"0001.json":failed,"0002.json":lost,"0003.json":final},total_updates=32)
        self.assertEqual(result["recorded_completed_update_calls"],32)
        self.assertEqual(result["recorded_step_seconds"],4.)
        self.assertEqual(result["recorded_worker_wall_seconds"],7.)
        self.assertTrue(result["unrecorded_physical_work_possible"])
        self.assertEqual(result["peak_cuda_allocated_mib"],30.)
        self.assertEqual(result["post_setup_peak_cuda_allocated_mib"],20.)
        self.assertEqual(result["recorded_setup_seconds"],3.)
        self.assertEqual(result["recorded_post_setup_wall_seconds"],5.)
        bad=copy.deepcopy(final);bad["chunks"][0]["step_seconds"]=-1
        with self.assertRaises(ValueError):summary.invocation_costs({"0001.json":bad},total_updates=32)
        duplicated=copy.deepcopy(final);duplicated["chunks"].append(copy.deepcopy(chunk))
        with self.assertRaisesRegex(ValueError,"committed update ledger"):
            summary.invocation_costs({"0001.json":duplicated},total_updates=32)

    def test_invocation_prefix_gaps_and_overlapping_settled_ranges(self):
        def receipt(start,end):
            return {"status":"completed","starting_updates":start,"ending_updates":end,
                    "setup_seconds":1.,"wall_seconds":3.,"post_setup_wall_seconds":2.,
                    "chunks":[{"completed_updates":end-start,"committed_updates":end-start,
                               "discarded_completed_updates":0,"failed_step_attempts":0}]}
        tail=receipt(16,32)
        missing_prefix=summary.invocation_costs({"0002.json":tail},total_updates=32)
        self.assertTrue(missing_prefix["unrecorded_physical_work_possible"])
        self.assertEqual(missing_prefix["recorded_completed_update_calls"],16)
        gap=summary.invocation_costs({"0001.json":receipt(0,8),"0003.json":tail},total_updates=32)
        self.assertTrue(gap["unrecorded_physical_work_possible"])
        complete=summary.invocation_costs({"0001.json":receipt(0,16),"0002.json":tail},total_updates=32)
        self.assertFalse(complete["unrecorded_physical_work_possible"])
        self.assertEqual(complete["recorded_completed_update_calls"],32)
        with self.assertRaisesRegex(ValueError,"ranges overlap or reverse"):
            summary.invocation_costs({"0001.json":receipt(0,24),"0002.json":tail},total_updates=32)
        reversed_range=receipt(0,16)
        reversed_range.update(starting_updates=24,ending_updates=16)
        with self.assertRaisesRegex(ValueError,"committed update ledger"):
            summary.invocation_costs({"0001.json":reversed_range,"0002.json":tail},total_updates=32)


if __name__=="__main__":unittest.main()
