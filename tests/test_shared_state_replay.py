"""One cached archive; CPU packing only. Invoke through the preserved harness."""
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch


def prove(report):
    import torch
    from experiments import shared_state_replay as replay, foundation_layout_prepared as prepared
    from experiments import foundation_layout_training as training, composition_curriculum as composition
    from experiments.sequence_student import SequenceConfig
    from experiments.foundation_layout_study import CONFIG, ROOT
    owners = []
    report.update(archive_loads=0, tensor_family_comparisons=0, rejected_cases=[], guards={})
    def forbidden(name):
        def call(*args, **kwargs):
            report["guards"][name] = report["guards"].get(name, 0)+1
            raise AssertionError("forbidden proof work: "+name)
        return call
    original_load = torch.load
    def load(*args, **kwargs):
        if report["archive_loads"] or kwargs.get("weights_only") is not True or kwargs.get("map_location") != "cpu":
            raise AssertionError("only one weights-only CPU archive load authorized")
        value = original_load(*args, **kwargs); report["archive_loads"] += 1
        return value
    config = SequenceConfig(**CONFIG)
    def owner():
        value = prepared.PreparedLayoutOwner(config=config, layout="original", micro_batch_size=32)
        owners.append(value); return value
    def pack(value, image, pin=None):
        return value.prepare(image["bundle"], expected_evidence=image["expected_evidence"],
            expected_evidence_sha256=pin or prepared.evidence_sha256(image["expected_evidence"]))
    def reject(name, call):
        try: call()
        except (ValueError, RuntimeError): report["rejected_cases"].append(name)
        else: raise AssertionError("mutation was accepted: "+name)
    reader = None
    with ExitStack() as stack:
        for target, name, label in (
            (torch.nn.Module, "__init__", "model_construction"),
            (torch.nn.Module, "_call_impl", "model_forward"),
            (torch.optim.Optimizer, "__init__", "optimizer_construction"),
            (torch.Tensor, "backward", "backward"), (torch.autograd, "grad", "gradient"),
            (torch.cuda, "_lazy_init", "cuda"),
            (training.curriculum.foundation, "generate_pair", "parent_generation"),
            (training.curriculum, "validate_pair", "canonical_pair_validation"),
            (composition, "english_oracle", "english_oracle"),
            (composition, "abstract_oracle", "abstract_oracle")):
            stack.enter_context(patch.object(target, name, forbidden(label)))
        stack.enter_context(patch.object(torch, "load", load))
        try:
            reader = replay.ReplayReader(ROOT/"runs/entity-retrieval-data-local/attempt-001",
                expected_manifest_sha256=replay.MANIFEST_SHA256)
            source = reader.load(0); original = source.original_image
            image648, provenance648 = source.remap(648)
            image1296, provenance1296 = source.remap(1296)
            report["provenance"] = [provenance648, provenance1296]
            reference = None; chains = []
            for cursor, image in ((0, original), (648, image648), (1296, image1296)):
                unchanged = deepcopy(image)
                unchanged["bundle"]["bundle_id"] = unchanged["expected_evidence"]["bundle_id"] = 0
                unchanged["expected_evidence_sha256"] = original["expected_evidence_sha256"]
                assert unchanged == original
                assert image["expected_evidence_sha256"] == replay._hash(image["expected_evidence"])
                value = owner(); token = pack(value, image)
                batches, evidence = value._consume(token, cursor=cursor, config=config, layout="original", micro_batch_size=32)
                assert evidence == image["expected_evidence"]
                if reference is None: reference = batches
                else:
                    for family in training.FAMILIES:
                        assert batches[family].keys() == reference[family].keys()
                        for section, fields in batches[family].items():
                            assert fields.keys() == reference[family][section].keys()
                            for name, tensor in fields.items():
                                prior = reference[family][section][name]
                                assert (tensor.dtype, tensor.shape, tensor.stride()) == (prior.dtype, prior.shape, prior.stride())
                                assert torch.equal(tensor, prior)
                        report["tensor_family_comparisons"] += 1
                # Synthetic prefix state: checks ID-sensitive chain arithmetic, not historical training.
                previous = training._initial_evidence(); previous["cursor"] = cursor
                after = training._accumulate(previous, evidence)
                assert after["cursor"] == cursor+1
                assert after["consumed_bundle_identity_sha256"] == training._hash([previous["consumed_bundle_identity_sha256"], evidence])
                assert after["consumed_common_parent_identity_sha256"] == training._hash([
                    previous["consumed_common_parent_identity_sha256"], cursor, evidence["common_parents_sha256"]])
                assert after["exposures"] == {f: evidence["families"][f]["exposures"] for f in training.FAMILIES}
                chains.append(after["consumed_bundle_identity_sha256"])
            assert len(set(chains)) == 3
            reject("wrong_source_mapping", lambda: source.remap(649))
            reject("bad_expected_hash", lambda: pack(owner(), image648, "0"*64))
            mutated = source.original_image
            mutated["bundle"]["families"]["color"][0]["turns"][0]["text"] += " changed"
            reject("nested_row_mutation", lambda: pack(owner(), mutated))
            assert source.original_image == original
            wrong = owner(); token = pack(wrong, image1296)
            reject("wrong_live_consumer_cursor", lambda: wrong._consume(token, cursor=1297,
                config=config, layout="original", micro_batch_size=32))
            assert wrong.failed and wrong.report()["consumed_bundles"] == 0
            totals = {name: sum(value.report()[name] for value in owners) for name in (
                "prepare_attempts", "prepared_bundles", "packed_microbatches", "packed_episodes",
                "exact_pair_matches", "canonical_regenerations", "consume_attempts", "consumed_bundles")}
            assert totals == dict(prepare_attempts=6, prepared_bundles=4, packed_microbatches=12,
                packed_episodes=384, exact_pair_matches=192, canonical_regenerations=0,
                consume_attempts=4, consumed_bundles=3)
            assert report["archive_loads"] == 1 and report["tensor_family_comparisons"] == 6
            assert not report["guards"] and not torch.cuda.is_initialized()
            report.update(packing_work=totals, distinct_consumed_chains=chains,
                chain_test_scope="Synthetic prefix states at cursors0/648/1296; no production history claimed.")
        finally:
            report["reader_work"] = reader.report() if reader is not None else None
            report["owners"] = [value.report() for value in owners]
            for value in owners: value.close()
