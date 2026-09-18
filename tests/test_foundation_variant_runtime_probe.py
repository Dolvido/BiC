"""Separate tiny CPU worker fixtures and pure integrity guards; no CUDA work."""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import foundation_variant_runtime_probe as probe
from experiments.sequence_student import SequenceConfig


class FoundationVariantRuntimeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)
        explicit=os.environ.get("BIC_VARIANT_PROBE_TEST_ARTIFACTS")
        cls.temporary=None if explicit else tempfile.TemporaryDirectory()
        cls.root=Path(explicit if explicit else cls.temporary.name)
        cls.root.mkdir(parents=True,exist_ok=True)
        cls.config=SequenceConfig(width=8,layers=4,heads=2,feedforward=16,max_turns=12)
        reuse=os.environ.get("BIC_VARIANT_PROBE_REUSE_FIXTURE")
        cls.directory=Path(reuse).resolve() if reuse else cls.root/"complete-cpu"
        if reuse:
            cls.protocol=probe._inputs(cls.directory)["protocol"]
            receipt=probe.core._read(cls.directory/"probe.json")
            if receipt["artifact_sha256"]!=probe.core._artifact_hashes(cls.directory,("probe.json",)):
                raise AssertionError("reused CPU fixture artifacts changed")
            if cls.protocol["device"]!="cpu" or cls.protocol["config"]!=asdict(cls.config):
                raise AssertionError("only the exact tiny CPU fixture may be reused")
        else:
            cls.protocol=probe.prepare(cls.directory,device="cpu",config=cls.config,
                                       steps=2,midpoint=1,micro_batch_size=2)
            for command,name in [("worker",name) for name in probe.WORKERS]+[("verify",None)]:
                args=[sys.executable,"-m","experiments.foundation_variant_runtime_probe",command,"--output",str(cls.directory)]
                if name: args.extend(("--name",name))
                completed=subprocess.run(args,cwd=probe.ROOT,capture_output=True,text=True,timeout=120)
                label=name or "verify"
                (cls.root/f"{label}-stdout.txt").write_text(completed.stdout,encoding="utf8")
                (cls.root/f"{label}-stderr.txt").write_text(completed.stderr,encoding="utf8")
                if completed.returncode:
                    raise AssertionError(f"{label} exit{completed.returncode}\n{completed.stdout}\n{completed.stderr}")
        cls.receipt=probe.core._read(cls.directory/"probe.json")
        if cls.receipt["status"]!="passed_cpu_engineering_only": raise AssertionError(cls.receipt)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        print("VARIANT_PROBE_CPU_ACCOUNTING="+json.dumps({"fixture_physical":cls.receipt["physical"],
            "reused_authenticated_fixture":bool(os.environ.get("BIC_VARIANT_PROBE_REUSE_FIXTURE")),
            "new_optimizer_updates":0 if os.environ.get("BIC_VARIANT_PROBE_REUSE_FIXTURE") else 36,
            "new_optimizer_episode_exposures":0 if os.environ.get("BIC_VARIANT_PROBE_REUSE_FIXTURE") else 216},sort_keys=True))
        if cls.temporary: cls.temporary.cleanup()

    def clone(self,name):
        path=self.root/("copy-"+hashlib.sha256(name.encode()).hexdigest()[:10])
        shutil.copytree(self.directory,path)
        return path

    def rewrite(self,path,value):
        path.write_text(probe.core._json(value),encoding="utf8")

    def test_six_cases_repeat_resume_outputs_and_exact_physical_counts(self):
        r=self.receipt
        self.assertEqual(r["status"],"passed_cpu_engineering_only")
        self.assertEqual(set(r["cases"]),{c["id"] for c in probe.CASES})
        self.assertTrue(all(all(values.values()) for values in r["cases"].values()))
        self.assertEqual(r["rates"],[.0003,.001,.003])
        self.assertEqual(r["architectures"],probe.ARCHITECTURES)
        self.assertEqual(r["physical_updates"],36)
        self.assertEqual(r["physical_episode_exposures"],216)
        self.assertEqual(r["physical"]["known_encoder_episodes"],252)
        self.assertEqual(r["physical"]["known_free_reply_turns"],2520)
        self.assertEqual(r["physical"]["unreported_started_attempts"],[])
        self.assertEqual(r["physical"]["evaluation_incomplete_or_unknown"],[])
        self.assertTrue(all(c["setup"]["index_reused"] for w in r["workers"].values() for c in w["cases"].values()))
        self.assertEqual(len({w["process"]["instance"] for w in r["workers"].values()}),4)
        self.assertFalse(r["automatic_promotion"])
        self.assertFalse(r["neural_work_in_verification"])

    def test_index_and_override_protection_covers_both_sides_of_split(self):
        p=self.protocol
        self.assertEqual(p["coverage"]["before_split"],1)
        self.assertEqual(p["coverage"]["after_split"],1)
        self.assertEqual(p["coverage"]["overridden_prefix_bundles"],2)
        plan=probe.core._read(self.directory/"plan.json")
        prefix=plan["schedules"]["mixed"][:2]
        self.assertTrue(all(f"{bundle}/color/0" in plan["admission"]["realization_attempts"] for bundle in prefix))
        context=probe._inputs(self.directory)
        self.assertEqual(context["index"].identity,probe.core._read(self.directory/"index-identity.json"))
        self.assertEqual(set(p["evaluation_identities"]),{"color/d0/t8","count/d1/t10","switch/d2/t12"})

    def test_cuda_contract_and_cpu_proof_gate_are_strict(self):
        with self.assertRaisesRegex(ValueError,"strict CUDA proof"):
            probe.load_proof(self.directory,self.config)
        with self.assertRaisesRegex(ValueError,"strict CUDA proof"):
            probe.load_proof(self.directory,probe.CUDA_CONFIG)
        for steps,midpoint,micro in ((7,3,32),(8,4,32),(8,3,16)):
            with self.subTest(steps=steps,midpoint=midpoint,micro=micro),self.assertRaises(ValueError):
                probe._contract("cuda:0",probe.CUDA_CONFIG,steps,midpoint,micro)
        formal=probe._contract("cuda:0",probe.CUDA_CONFIG,8,3,32)
        self.assertEqual(formal["planned_physical_updates"],144)
        self.assertEqual(formal["planned_episode_exposures"],13824)
        self.assertEqual(formal["config"],asdict(probe.CUDA_CONFIG))

    def test_one_shot_directories_cannot_retry_completed_work(self):
        with self.assertRaises(FileExistsError):
            probe.prepare(self.directory,device="cpu",config=self.config,steps=2,midpoint=1,micro_batch_size=2)
        with self.assertRaises(FileExistsError): probe.worker(self.directory,"uninterrupted")
        with self.assertRaises(FileExistsError): probe.verify(self.directory)

    def test_changed_inputs_and_source_snapshot_reject_before_learning(self):
        for filename in ("protected.json","evaluation-banks.json","admission.json",
                         "sources/experiments/hierarchical_sequence_student.py"):
            path=self.clone("tamper-"+filename.replace("/","_").replace(".","_"))
            with (path/filename).open("ab") as stream: stream.write(b" ")
            with self.subTest(filename=filename),self.assertRaisesRegex(ValueError,"changed"):
                probe._inputs(path)

    def test_six_case_contract_cannot_be_reduced_by_rebinding_protocol(self):
        path=self.clone("missing-case"); p=probe.core._read(path/"protocol.json")
        p["cases"].pop(); self.rewrite(path/"protocol.json",p)
        prepared=probe.core._read(path/"prepared.json"); prepared["protocol_sha256"]=probe.core._file_hash(path/"protocol.json")
        self.rewrite(path/"prepared.json",prepared)
        with self.assertRaisesRegex(ValueError,"six-case"):
            probe._inputs(path)

    def test_worker_artifact_mutation_is_detected(self):
        path=self.clone("changed-worker")
        with (path/"uninterrupted"/probe.CASES[0]["id"]/"evaluation-001.pt").open("ab") as stream: stream.write(b"x")
        with self.assertRaisesRegex(ValueError,"artifacts changed"):
            probe._worker_receipt(path,"uninterrupted")

    def test_journal_guards_reject_wrong_rows_counts_and_boolean_cursors(self):
        context=probe._inputs(self.directory)
        records=context["index"].replay("mixed",2,include_bundles=True)["bundles"]
        original=probe._events(self.directory/"uninterrupted"/probe.CASES[0]["id"]/"steps.jsonl")
        self.assertEqual(len(probe._validate_journal(original,records,0,2,2)),2)
        for kind in ("count","rows","cursor","missing","time"):
            events=copy.deepcopy(original)
            if kind=="count": events[1]["report"]["physical_optimizer_updates"]=2
            elif kind=="rows": events[1]["report"]["microbatches"][0]["rows_sha256"]="0"*64
            elif kind=="cursor": events[0]["cursor"]=False
            elif kind=="missing": events.pop()
            else: events[1]["report"]["step_seconds"]=-1
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                probe._validate_journal(events,records,0,2,2)

    def test_failed_and_dangling_work_remains_unknown_in_ledger(self):
        directory=self.root/"partial"; path=directory/"uninterrupted"/probe.CASES[0]["id"]
        path.mkdir(parents=True)
        probe.core._append(path/"steps.jsonl",{"event":"started","cursor":0})
        probe.core._append(path/"evaluation.jsonl",{"event":"eval_started","cursor":0})
        physical=probe._physical(directory)
        self.assertEqual(physical["known_completed_optimizer_updates"],0)
        self.assertTrue(physical["unreported_started_attempts"])
        self.assertTrue(physical["evaluation_incomplete_or_unknown"])
        probe.core._append(path/"steps.jsonl",{"event":"reported","report":None})
        probe.core._append(path/"evaluation.jsonl",{"event":"eval_failed","episodes":2,"turns":16,
            "encoder_completed":True,"generation_completed":False})
        physical=probe._physical(directory)
        self.assertEqual(physical["optimizer_completion_unknown_attempts"],1)
        self.assertEqual(physical["partial_materialization_unknown_attempts"],1)
        self.assertEqual(physical["known_encoder_episodes"],2)
        self.assertEqual(physical["known_free_reply_turns"],0)
        self.assertTrue(physical["evaluation_incomplete_or_unknown"])

    def test_only_declared_checkpoint_timing_is_ignored(self):
        case=probe.CASES[0]["id"]
        a=probe.core._load(self.directory/"uninterrupted"/case/"final.pt")
        b=copy.deepcopy(a); b["timing"]["retained_step_seconds"]+=1
        self.assertTrue(probe.core._difference(a,b)["exact"])
        b["evidence"]["cursor"]+=1
        self.assertFalse(probe.core._difference(a,b)["exact"])
        b=copy.deepcopy(a); b["timing"]["unrecognized"]=1
        with self.assertRaises(ValueError): probe.core._difference(a,b)


class ProofCacheTests(unittest.TestCase):
    """Pure JSON authentication control tests; no model or fabricated GPU proof use."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.directory=Path(self.temp.name)
        self.sources={"source.py":"a"*64}; self.old_cache=probe._PROOF_CACHE; probe._PROOF_CACHE=None
        (self.directory/"artifact.json").write_text("{}",encoding="utf8")
        self.value={"status":"passed","config":asdict(probe.CUDA_CONFIG),"rates":list(probe.RATES),
            "architectures":probe.ARCHITECTURES,"cases":{},"source_sha256":self.sources,
            "input_sha256":{},"protocol_sha256":"b"*64,"execution_profile":{},"coverage":{},
            "physical":{},"physical_updates":144,"physical_episode_exposures":13824,"automatic_promotion":False}
        self.publish()
        self.source_patch=patch.object(probe,"source_hashes",return_value=self.sources)
        self.verify_patch=patch.object(probe,"_verified",side_effect=lambda directory:copy.deepcopy(self.value))
        self.source_patch.start(); self.verifier=self.verify_patch.start()

    def publish(self):
        self.value["artifact_sha256"]=probe.core._artifact_hashes(self.directory,("probe.json",))
        (self.directory/"probe.json").write_text(probe.core._json(self.value),encoding="utf8")

    def tearDown(self):
        self.verify_patch.stop(); self.source_patch.stop(); probe._PROOF_CACHE=self.old_cache; self.temp.cleanup()

    def test_first_load_verifies_once_then_copy_isolated_json_cache(self):
        first=probe.load_proof(self.directory,probe.CUDA_CONFIG); first["cases"]["mutation"]=True
        second=probe.load_proof(self.directory,asdict(probe.CUDA_CONFIG))
        self.assertNotIn("mutation",second["cases"]); self.assertEqual(self.verifier.call_count,1)
        self.assertIsInstance(probe._PROOF_CACHE[1],str)
        self.value["extra_receipt_note"]="changed"; self.publish()
        probe.load_proof(self.directory,probe.CUDA_CONFIG)
        self.assertEqual(self.verifier.call_count,2)

    def test_cache_hit_still_rejects_artifact_or_current_source_mutation(self):
        probe.load_proof(self.directory,probe.CUDA_CONFIG)
        (self.directory/"artifact.json").write_text("[]",encoding="utf8")
        with self.assertRaisesRegex(ValueError,"changed"): probe.load_proof(self.directory,probe.CUDA_CONFIG)
        self.publish(); probe.load_proof(self.directory,probe.CUDA_CONFIG)
        with patch.object(probe,"source_hashes",return_value={"source.py":"c"*64}):
            with self.assertRaisesRegex(ValueError,"changed"): probe.load_proof(self.directory,probe.CUDA_CONFIG)

    def test_drift_during_first_authentication_cannot_publish_cache(self):
        def changed(directory):
            (directory/"artifact.json").write_text("[]",encoding="utf8")
            return copy.deepcopy(self.value)
        self.verifier.side_effect=changed
        with self.assertRaisesRegex(ValueError,"changed"): probe.load_proof(self.directory,probe.CUDA_CONFIG)
        self.assertIsNone(probe._PROOF_CACHE)


if __name__=="__main__":
    unittest.main()
