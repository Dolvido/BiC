"""Stdlib-only adapter boundaries using synthetic compiler/author results."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import verified_tutor_cycle_data as data
from experiments import verified_tutor_author_v2 as author

WORK = dict(tests=0, mocked_inventory_reads=0, mocked_author_loads=0,
            mocked_compiler_calls=0, synthetic_images=0, mock_tensor_publications=0,
            tensor_archive_loads=0, lesson_generation=0, models=0, teacher_calls=0)


class CycleDataTests(unittest.TestCase):
    def setUp(self):
        WORK["tests"] += 1
        directory = data.ROOT/"runs"/"verified-tutor-cycle-data-validation-local"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = dict(identity_sha256="a"*64, weights_sha256="b"*64,
                           cycle=7, lifetime_updates=5000)
        self.inventory = dict(schema=data.compiler.INVENTORY_SCHEMA, plans={}, motifs={}, replay=[])
        self.protected = {"c"*64}
        self.inventory_patch = patch.object(data, "_inventory", side_effect=self.mock_inventory)
        self.inventory_patch.start(); self.addCleanup(self.inventory_patch.stop)
        guard = patch.object(data.io, "_cpu_guard", side_effect=lambda work:nullcontext())
        guard.start(); self.addCleanup(guard.stop)
        self.original_write = data.io.write
        writer = patch.object(data.io, "write", side_effect=self.write)
        writer.start(); self.addCleanup(writer.stop)
        loader = patch.object(author, "load_result", side_effect=self.load_author)
        loader.start(); self.addCleanup(loader.stop)
        compiler = patch.object(data.compiler, "compile_curriculum", side_effect=self.compile)
        compiler.start(); self.addCleanup(compiler.stop)
        self.bad_cursor = False

    def mock_inventory(self, run, directory, pin, extras):
        WORK["mocked_inventory_reads"] += 1
        return deepcopy(self.inventory), set(self.protected), []

    def write(self, path, value):
        if Path(path).suffix != ".pt":
            return self.original_write(path, value)
        WORK["mock_tensor_publications"] += 1
        path = data.io.native(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data.io.encoded(value))
        return data.io.digest(path)

    def load_author(self, path, *, expected_sha256, expected_request_sha256):
        WORK["mocked_author_loads"] += 1
        self.assertEqual(data.io.digest(path), expected_sha256)
        self.assertEqual(expected_request_sha256, "d"*64)
        return deepcopy(self.decision)

    def compile(self, recipe, *, admitted_parent_inventory, protected_transcripts, coverage_contract):
        WORK["mocked_compiler_calls"] += 1
        images = []
        for index, slot in enumerate(coverage_contract["slots"]):
            cursor = coverage_contract["start_cursor"]+index
            if self.bad_cursor and index == 0:
                cursor += 1
            text = "synthetic-"+str(index)
            if not slot["replay"]:
                text += recipe["chapters"][slot["chapter"]]["realization"]
            bundle = dict(bundle_id=cursor, families={family:[dict(turns=[dict(
                text=text, observations={}, target=0, reply="synthetic")])]
                for family in data.compiler.FAMILIES})
            evidence = dict(bundle_id=cursor, bundle_rows_sha256=data.compiler._hash(bundle))
            images.append(dict(bundle=bundle, expected_evidence=evidence,
                               expected_evidence_sha256=data.compiler._hash(evidence)))
        WORK["synthetic_images"] += len(images)
        return dict(images=images, provenance=[], receipt=dict(status="mocked", neural_work=0),
            manifest=dict(contract_sha256=data.compiler.contract_sha256(coverage_contract),
                recipe_sha256=data.compiler.recipe_sha256(recipe), exposures={"synthetic":len(images)}))

    def prepare(self, fresh_only=()):
        self.output = self.root/"cycle"
        context = data.prepare_cycle(self.output, inventory_directory=self.root/"inventory",
            inventory_manifest_sha256="e"*64, parent=self.parent, seed=991,
            withdrawal_seed=992, compiler_seconds=0.01, fresh_only_protected=fresh_only)
        self.context_pin = data.io.digest(self.output/"cycle.json")
        contract = data.io.read(data.ROOT/context["teaching_contract"]["path"])
        recipe = data.compiler.procedural_recipe(contract)
        recipe["chapters"][2]["realization"] = "rename"
        author_dir = self.root/"author"; author_dir.mkdir()
        intent_pin = self.original_write(author_dir/"intent.json", {"synthetic":"intent"})
        self.decision = dict(parent=deepcopy(self.parent), contract_sha256=data.compiler.contract_sha256(contract),
            recipe=recipe, recipe_sha256=data.compiler.recipe_sha256(recipe), intent_sha256=intent_pin,
            outcome="local_accepted", teacher_cost=dict(response_file_sha256=None, api_events=[]))
        decision_pin = self.original_write(author_dir/"result.json", self.decision)
        self.decision_record = dict(path=data._relative(author_dir/"result.json"), sha256=decision_pin,
                                    request_sha256="d"*64)
        return context

    def execute(self):
        return data.compile_data(self.output, cycle_sha256=self.context_pin,
                                 teacher_decision=self.decision_record, max_seconds=60)

    def test_generic_contract_and_seed_bounds(self):
        before = deepcopy(self.parent)
        a = data.make_contract("e"*64,"f"*64,7,parent=self.parent,seed=2**63-1)
        b = data.make_contract("e"*64,"f"*64,7,parent=self.parent,seed=42,withdrawal=True)
        self.assertEqual((a["start_cursor"],b["start_cursor"]),(5000,5108))
        self.assertEqual(sum(x["replay"] for x in a["slots"]),36)
        self.assertTrue(all(x["replay"] for x in b["slots"]))
        self.assertEqual(self.parent,before)
        with self.assertRaises(ValueError):
            data.make_contract("e"*64,"f"*64,7,parent=self.parent,seed=2**63)

    def test_fingerprints_are_exact(self):
        self.assertEqual(data._fingerprints(["a"*64,"b"*64]),{"a"*64,"b"*64})
        for bad in (["a"*64,"a"*64],["b"*64,"a"*64],["x"*64],{"a"*64}):
            with self.assertRaises(ValueError): data._fingerprints(bad)

    def test_complete_cycle_and_exclusive_resume(self):
        context = self.prepare()
        self.assertEqual(context["phase_seeds"],dict(teaching=991,withdrawal=992))
        manifest = self.execute()
        directory = self.output/"compiled"
        loaded = data.load_manifest(directory, expected_manifest_sha256=data.io.digest(directory/"manifest.json"))
        self.assertEqual(loaded,manifest)
        self.assertEqual(manifest["phases"]["withdrawal"]["procedural"],manifest["phases"]["withdrawal"]["tutor"])
        self.assertEqual(manifest["treatment"]["different_teaching_bundles"],18)
        self.assertFalse(manifest["treatment"]["zero_treatment_contrast"])
        self.assertEqual(manifest["phases"]["teaching"]["tutor"][0]["cursor"],5000)
        self.assertEqual(manifest["phases"]["withdrawal"]["tutor"][-1]["cursor"],5215)
        fresh = data.io.read(data.ROOT/manifest["fresh_transcripts"]["path"])
        self.assertEqual(len(fresh),90)
        self.assertFalse({data.compiler.transcript_sha256(dict(turns=[dict(text="synthetic-0")]))}&set(fresh))
        before = data.io.digest(directory/"preparation.json")
        with self.assertRaises(FileExistsError): self.execute()
        self.assertEqual(before,data.io.digest(directory/"preparation.json"))
        path = directory/manifest["phases"]["teaching"]["procedural"][0]["path"]
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            data.load_manifest(directory,expected_manifest_sha256=data.io.digest(directory/"manifest.json"))

    def test_saved_parent_mismatch_before_compilation(self):
        self.prepare(); before = WORK["mocked_compiler_calls"]
        self.decision["parent"]["lifetime_updates"] += 1
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(WORK["mocked_compiler_calls"],before)
        self.assertEqual(data.io.read(self.output/"compiled"/"preparation.json")["status"],"failed")

    def test_saved_contract_mismatch_before_compilation(self):
        self.prepare(); before = WORK["mocked_compiler_calls"]
        self.decision["contract_sha256"] = "0"*64
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(WORK["mocked_compiler_calls"],before)

    def test_protection_change_rejected_before_author(self):
        self.prepare(); before = WORK["mocked_author_loads"]
        self.protected.add("9"*64)
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(WORK["mocked_author_loads"],before)

    def test_bad_compiled_cursor_is_preserved_failure(self):
        self.prepare(); self.bad_cursor = True
        with self.assertRaises(ValueError): self.execute()
        receipt = data.io.read(self.output/"compiled"/"preparation.json")
        self.assertEqual((receipt["compiler_attempts"],receipt["compiler_completions"]),(1,1))
        self.assertFalse((self.output/"compiled"/"manifest.json").exists())

    def test_global_deadline_has_no_implicit_extension(self):
        self.prepare(); before = WORK["mocked_compiler_calls"]
        with self.assertRaises(TimeoutError):
            data.compile_data(self.output,cycle_sha256=self.context_pin,
                teacher_decision=self.decision_record,max_seconds=1)
        self.assertEqual(WORK["mocked_compiler_calls"],before)

    def test_fresh_only_exclusion_allows_literal_replay(self):
        def fingerprint(text):
            return data.compiler.transcript_sha256(dict(turns=[dict(text=text)]))
        replay = fingerprint("synthetic-0")
        path = self.root/"prior-training.json"
        pin = self.original_write(path,[replay])
        self.prepare([dict(path=data._relative(path),sha256=pin)])
        manifest = self.execute()
        self.assertEqual(manifest["fresh_only_protection"]["count"],1)
        # The same admitted replay is valid, while a non-replay overlap rejects
        # the complete compiler result without retrying or silently substituting.
        self.root = self.root/"next"; self.root.mkdir()
        path = self.root/"prior-training.json"
        repeated = fingerprint("synthetic-36independent")
        pin = self.original_write(path,sorted([replay,repeated]))
        self.prepare([dict(path=data._relative(path),sha256=pin)])
        before = WORK["mocked_compiler_calls"]
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(WORK["mocked_compiler_calls"],before+1)
        receipt = data.io.read(self.output/"compiled"/"preparation.json")
        self.assertEqual(receipt["fresh_only_rejection"]["overlap"],[repeated])
        self.assertEqual((receipt["compiler_attempts"],receipt["compiler_completions"]),(1,1))
        self.assertFalse((self.output/"compiled"/"manifest.json").exists())


if __name__ == "__main__": unittest.main()
