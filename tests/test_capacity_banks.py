"""Capacity banks: factor isolation, stage exclusion and sealed history."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import capacity_banks as banks
from experiments import composition_curriculum as curriculum
from experiments.realization_training import stream_evidence


SMALL = dict(train_pairs=3, panel_pairs=2, calibration_train_pairs=2, calibration_panel_pairs=1)


class CapacityBanksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = patch.object(banks, "historical_transcript_digests", return_value=["a"*64])
        cls.evidence = patch.object(banks, "historical_evidence", return_value={"unique_transcripts": 1})
        cls.history.start(); cls.evidence.start()
        cls.addClassCleanup(cls.history.stop); cls.addClassCleanup(cls.evidence.stop)
        cls.values, cls.details = banks.prepare_banks(**SMALL, with_diagnostics=True)

    def test_roles_counts_and_canonical_independent_truth(self):
        self.assertEqual(set(self.values), {"main", "calibration"})
        self.assertNotIn("audit", self.values["calibration"])
        self.assertEqual(self.details["boundaries"]["episodes"], {
            "main/train": 54, "main/development": 144, "main/audit": 144,
            "calibration/train": 36, "calibration/development": 72})
        for rows in banks._all_banks(self.values):
            for i in range(0,len(rows),2):
                self.assertTrue(curriculum.validate_pair(rows[i:i+2]))
                for row in rows[i:i+2]:
                    self.assertEqual(curriculum.english_oracle(row), [t["target"] for t in row["turns"]])

    def test_familiar_controls_in_both_stages_preserve_coordinates(self):
        for stage in banks.STAGES:
            for role in banks.ROLES[stage][1:]:
                for family in banks.FAMILIES:
                    for turns in banks.LENGTHS:
                        panels={p:self.values[stage][role][f"{p}/{family}/t{turns}"] for p in banks.PANELS[:3]}
                        source=self.values[stage]["train"][family][turns]
                        for i in range(0,len(panels["both"]),2):
                            banks._check_familiar(source[i:i+2],{p:rows[i:i+2] for p,rows in panels.items()})
                            n,v,b=(panels[p][i] for p in banks.PANELS[:3])
                            self.assertEqual(n["recipe"]["value_seed"],source[i]["recipe"]["value_seed"])
                            self.assertEqual(v["recipe"]["naming_seed"],source[i]["recipe"]["naming_seed"])
                            self.assertEqual(n["recipe"]["naming_seed"],b["recipe"]["naming_seed"])
                            self.assertEqual(v["recipe"]["value_seed"],b["recipe"]["value_seed"])

    def test_global_exact_and_supervised_ancestry_boundaries(self):
        detail=self.details["boundaries"]
        train=set(detail["training_known_composed_query_ids"])
        dev=set(detail["development_known_composed_query_ids"])
        self.assertFalse(train & set(detail["development_final_composed_ids"]))
        self.assertFalse((train|dev)&set(detail["audit_final_composed_ids"]))
        digests=[banks.transcript_digest(r) for rows in banks._all_banks(self.values) for r in rows]
        self.assertEqual(len(digests),len(set(digests)))
        for stage in banks.STAGES:
            for rows in self.values[stage]["development"].values():
                for row in rows:
                    self.assertFalse(any(q["structure_partition"]=="audit" for q in banks.known_composed_queries(row)))

    def test_protection_every_evaluation_other_training_and_planned_stream(self):
        all_eval={banks.transcript_digest(row) for stage in banks.STAGES for role in banks.ROLES[stage][1:]
                  for rows in self.values[stage][role].values() for row in rows}
        original={stage:{banks.transcript_digest(row) for bs in self.values[stage]["train"].values()
                        for rows in bs.values() for row in rows} for stage in banks.STAGES}
        for stage,other in (("main","calibration"),("calibration","main")):
            values=banks.protected_transcripts(self.values,stage=stage)
            self.assertEqual(values,sorted(set(values)))
            self.assertEqual(set(values),all_eval|original[other]|{"a"*64})
            self.assertFalse(set(values)&original[stage])
        enhanced=banks.protected_transcripts(self.values,stage="main",consumed_calibration=["b"*64])
        self.assertIn("b"*64,enhanced)
        self.assertNotIn("b"*64,banks.protected_transcripts(self.values))
        with self.assertRaisesRegex(ValueError,"overlap"):
            banks.protected_transcripts(self.values,consumed_calibration=[next(iter(original["main"]))])
        for kwargs in ({"stage":"bad"},{"stage":"calibration","consumed_calibration":["b"*64]},
                       {"consumed_calibration":["B"*64]},{"consumed_calibration":["b"*64,"b"*64]},
                       {"consumed_calibration":"b"*64}):
            with self.assertRaises(ValueError):banks.protected_transcripts(self.values,**kwargs)

    def test_historical_single_variant_rejects_entire_pair(self):
        first=self.values["main"]["train"]["color"][8][:2]
        forbidden=banks.transcript_digest(first[1])
        with patch.object(banks,"historical_transcript_digests",return_value=[forbidden]):
            changed,details=banks.prepare_banks(**SMALL,with_diagnostics=True)
            self.assertNotEqual(changed["main"]["train"]["color"][8][0]["recipe"]["seed"],first[0]["recipe"]["seed"])
            self.assertNotIn(forbidden,{banks.transcript_digest(row) for rows in banks._all_banks(changed) for row in rows})
            self.assertGreater(details["selection"][0]["rejected_pairs_or_factor_triplets"]["historical_transcript"],0)
            with self.assertRaisesRegex(ValueError,"historical observations"):
                banks.verify_boundaries(self.values)

    def test_manifest_denominators_coverage_and_no_calibration_audit(self):
        m=banks.bank_manifest(self.values)
        self.assertEqual(set(m["calibration"]),{"train","development"})
        for row in m["main"]["audit"].values():
            self.assertEqual(row["episodes"],4)
            self.assertEqual(row["final_opposite_pair_total"],2)
            self.assertEqual(row["opposite_pair_total_by_turn"][-1],2)
            self.assertEqual(row["target_counts_by_turn"][-1],{"0":2,"1":2,"2":0,"3":0})
            self.assertGreater(row["query_target_counts"]["2"],0)
            self.assertLessEqual(row["max_context_tokens"],1024)
            self.assertEqual(len(row["transcript_sha256"]),4)

    def test_canonical_wrong_factor_and_cross_stage_reuse_rejected(self):
        changed=copy.deepcopy(self.values)
        row=changed["main"]["development"]["name_only/color/t8"][0];r=row["recipe"]
        changed["main"]["development"]["name_only/color/t8"][:2]=curriculum.generate_pair(
            "color",r["seed"],split="dev",turns=8,naming_seed=r["naming_seed"],
            value_seed=r["value_seed"]+1,structure_split="train")
        with self.assertRaisesRegex(ValueError,"wrong realization coordinate"):
            banks.verify_boundaries(changed)
        changed=copy.deepcopy(self.values)
        changed["calibration"]["train"]["color"][8][:2]=copy.deepcopy(changed["main"]["train"]["color"][8][:2])
        with self.assertRaisesRegex(ValueError,"exact transcript reused"):
            banks.verify_boundaries(changed)

    def test_missing_role_and_tampered_truth_fail(self):
        for mutation in (
            lambda b:b["calibration"].update(audit={}),
            lambda b:b["main"]["development"].pop("both/count/t8"),
            lambda b:b["main"]["train"]["color"][8][0]["turns"][-1].update(target=2)):
            changed=copy.deepcopy(self.values);mutation(changed)
            with self.assertRaises(ValueError):banks.verify_boundaries(changed)

    def test_determinism_mutation_isolation_and_old_globals_unchanged(self):
        from experiments import realization_banks as old
        before=copy.deepcopy(old.SEEDS)
        result,details=banks.prepare_banks(**SMALL,with_diagnostics=True)
        self.assertEqual(result,self.values);self.assertEqual(details,self.details)
        result["main"]["train"]["color"][8][0]["turns"][0]["text"]="changed"
        self.assertNotEqual(result,self.values)
        self.assertEqual(old.SEEDS,before)
        self.assertEqual(banks.SEEDS["main"]["train"],210_000_000)
        self.assertEqual(banks.SEEDS["calibration"]["train"],240_000_000)

    def test_invalid_sizes_and_bounded_search(self):
        for kw in ({"train_pairs":True},{"panel_pairs":0},{"calibration_train_pairs":False},
                   {"calibration_train_pairs":1,"calibration_panel_pairs":2},{"with_diagnostics":1}):
            with self.assertRaises(ValueError):banks.prepare_banks(**kw)
        with patch.object(banks,"transcript_digest",return_value="collision"):
            with self.assertRaisesRegex(ValueError,"bounded search"):
                banks.prepare_banks(train_pairs=1,panel_pairs=1,calibration_train_pairs=1,calibration_panel_pairs=1)


class HistoricalCapacityTests(unittest.TestCase):
    def _fixture(self,root):
        """A sealed primitive checkpoint fixture; no neural model is constructed."""
        prior=root/"realization-study-local";old=root/"composition-study-local"
        (prior/"audit").mkdir(parents=True);(prior/"verification").mkdir()
        old.mkdir()
        pair=curriculum.generate_pair("color",7654,turns=8)
        previous={"train":{"color":{8:pair}},"development":{},"audit":{}}
        torch.save(previous,old/"banks.pt")
        (old/"protocol.json").write_text(json.dumps({"banks_file_sha256":banks._file_digest(old/"banks.pt")}),encoding="utf8")
        torch.save(previous,prior/"banks.pt")
        protocol={"banks_file_sha256":banks._file_digest(prior/"banks.pt"),"checkpoints":[0,1],"updates":1}
        (prior/"protocol.json").write_text(json.dumps(protocol),encoding="utf8")
        hashes={"banks.pt":banks._file_digest(prior/"banks.pt"),"protocol.json":banks._file_digest(prior/"protocol.json")}
        for job in ("fixed","fresh"):
            (prior/"main"/job).mkdir(parents=True)
            streams={}
            for step in (0,1):
                stream={"seen_transcripts":[] if step==0 else ["c"*64],"samplers":{"color":torch.tensor([1,2],dtype=torch.uint8)},"recipe":{"mode":job,"banks":{"color":{8:"small-bank"}}}}
                payload={"job":job,"protocol":protocol,"training":{"updates":step,"realization":stream}}
                name=f"main/{job}/checkpoint-{step:06d}.pt"
                torch.save(payload,prior/name);hashes[name]=banks._file_digest(prior/name)
                streams[str(step)]=stream_evidence(stream)
            torch.save(payload,prior/"main"/job/"latest.pt")
            hashes[f"main/{job}/latest.pt"]=banks._file_digest(prior/"main"/job/"latest.pt")
            name=f"verification/{job}.json"
            (prior/name).write_text(json.dumps({"protocol_sha256":hashes["protocol.json"],"replayed_streams":streams}),encoding="utf8")
            hashes[name]=banks._file_digest(prior/name)
        summary={"verification":{"input_file_sha256":hashes,"historical_evidence":{
            "protocol_sha256":banks._file_digest(old/"protocol.json"),"banks_sha256":banks._file_digest(old/"banks.pt")}}}
        self._seal(prior,summary)
        return prior,summary,pair

    def _seal(self,root,summary):
        (root/"audit"/"summary.json").write_text(json.dumps(summary),encoding="utf8")
        (root/"decision.json").write_text(json.dumps({"evidence_sha256":{
            "audit/summary.json":banks._file_digest(root/"audit"/"summary.json")}}),encoding="utf8")

    def test_authenticated_saved_consumed_indices_included_and_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            prior,summary,pair=self._fixture(Path(directory))
            with patch.object(banks,"HISTORICAL_DIRECTORY",prior):
                values=banks.historical_transcript_digests()
                self.assertEqual(set(values),{"c"*64}|{banks.transcript_digest(r) for r in pair})
                detail=banks.historical_evidence()
                self.assertEqual(len(detail["checkpoint_streams"]),6)
                self.assertEqual(detail["unique_transcripts"],3)
                (prior/"main"/"fresh"/"latest.pt").write_bytes(b"changed after cache")
                with self.assertRaisesRegex(ValueError,"historical file digest differs"):
                    banks.historical_transcript_digests()

    def test_unsealed_summary_and_consumed_stream_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            prior,summary,_=self._fixture(Path(directory))
            with patch.object(banks,"HISTORICAL_DIRECTORY",prior):
                (prior/"audit"/"summary.json").write_text("{}",encoding="utf8")
                with self.assertRaisesRegex(ValueError,"summary digest"):
                    banks.historical_evidence()
                self._seal(prior,summary)
                rel="main/fresh/checkpoint-000001.pt"
                payload=torch.load(prior/rel,map_location="cpu",weights_only=True)
                payload["training"]["realization"]["seen_transcripts"]=["d"*64]
                torch.save(payload,prior/rel)
                summary["verification"]["input_file_sha256"][rel]=banks._file_digest(prior/rel)
                self._seal(prior,summary)
                with self.assertRaisesRegex(ValueError,"sealed canonical replay"):
                    banks.historical_transcript_digests()


if __name__=="__main__":unittest.main()
