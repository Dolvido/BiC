"""Pure count/lineage tests with explicit non-neural synthetic frames.

These frames test the fold, not completion or prediction authenticity. Real
CompletedParent integration is exercised by the repeated-cycle trainer suite.
"""
import copy
import json
import pickle
import unittest
from unittest.mock import patch

from experiments import foundation_lifetime_retention as module
from experiments.foundation_evidence import json_digest


SOURCES = {"synthetic-non-neural-test": "a"*64}
BANKS = {"retention:fixture/color/d0/direct/t8": {"identity": {"episodes": 20, "rows_sha256": "b"*64}}}


def frame(cycle=1, *, action=8, reply=7, known=30, ask=3, cycle_best=None):
    digest = json_digest(["synthetic parent", cycle])
    counts = dict(paired_action=action, paired_reply=reply, known=known, unsupported_ask=ask)
    def record(field, number, step):
        total = 40 if field in ("known", "unsupported_ask") else 10
        numerator = "count" if field == "unsupported_ask" else "correct"
        return dict(value={numerator:number, "total":total, "rate":number/total},
            producer=dict(cycle=cycle, local_updates=step, lifetime_updates=(cycle-1)*10+step,
                weights_sha256=json_digest([cycle,step]), provider_sha256="d"*64,
                parent_identity_sha256=digest), response_sha256=json_digest([cycle,step,field,number]))
    endpoint = {key: {field:record(field,number,10) for field,number in counts.items()} for key in BANKS}
    best_counts = counts if cycle_best is None else cycle_best
    best = {key: {field:record(field,number,10 if number==counts[field] else 0)
        for field,number in best_counts.items()} for key in BANKS}
    return dict(cycle=cycle, lifetime_updates=10*cycle, inherited_updates=10*(cycle-1),
        parent_identity_sha256=digest, predecessor_identity_sha256=None if cycle==1 else json_digest(["synthetic parent",cycle-1]),
        bank_specs_sha256=json_digest(BANKS), banks=copy.deepcopy(BANKS), endpoint=endpoint, cycle_best=best)


class LifetimeRetentionTests(unittest.TestCase):
    def test_old_best_survives_two_later_cycles_and_ties_keep_original_producer(self):
        first = module._fold(None, frame(), SOURCES)
        second = module._fold(first, frame(2,action=2,reply=1,known=10,ask=8), SOURCES)
        self.assertEqual(len(second["alarms"]), 4)
        third = module._fold(second, frame(3), SOURCES)
        self.assertEqual(third["alarms"], [])
        for fields in third["best"].values():
            self.assertEqual({item["producer"]["cycle"] for item in fields.values()}, {1})

    def test_one_improvement_cannot_hide_other_field_regressions(self):
        first = module._fold(None, frame(), SOURCES)
        second = module._fold(first, frame(2,action=9,reply=1,known=10,ask=8), SOURCES)
        key = next(iter(BANKS))
        self.assertEqual(second["best"][key]["paired_action"]["producer"]["cycle"], 2)
        self.assertEqual({item["field"] for item in second["alarms"]}, {"paired_reply","known","unsupported_ask"})

    def test_current_endpoint_is_compared_with_earlier_best_inside_same_cycle(self):
        best = dict(paired_action=9,paired_reply=8,known=35,unsupported_ask=1)
        state = module._fold(None, frame(cycle_best=best), SOURCES)
        self.assertEqual(len(state["alarms"]), 4)
        self.assertTrue(all(item["best"]["producer"]["local_updates"] == 0 for item in state["alarms"]))

    def test_skipped_reused_foreign_cycle_and_lifetime_reset_are_rejected(self):
        original = module._fold(None, frame(), SOURCES)
        for kind in ("skip","reuse","predecessor","lifetime","inherited"):
            changed = frame(2)
            if kind=="skip": changed["cycle"]=3
            elif kind=="reuse": changed["cycle"]=1
            elif kind=="predecessor": changed["predecessor_identity_sha256"]="f"*64
            elif kind=="lifetime": changed["lifetime_updates"]=10
            else: changed["inherited_updates"]=0
            with self.subTest(kind=kind), self.assertRaises(ValueError): module._fold(original,changed,SOURCES)
        self.assertEqual(original,module._fold(None,frame(),SOURCES))
        with self.assertRaises(ValueError): module._fold(None,frame(2),SOURCES)

    def test_roles_bank_content_and_denominators_cannot_be_relabelled(self):
        prior = module._fold(None,frame(),SOURCES)
        for kind in ("role","bank","denominator"):
            changed=frame(2); key=next(iter(BANKS))
            if kind=="role": changed["banks"]={"development:renamed":changed["banks"][key]}
            elif kind=="bank": changed["banks"][key]["identity"]["rows_sha256"]="f"*64
            else:
                value=changed["cycle_best"][key]["paired_action"]["value"]
                value.update(total=20,rate=value["correct"]/20)
            with self.subTest(kind=kind),self.assertRaises(ValueError): module._fold(prior,changed,SOURCES)

    def test_undefined_scores_stay_undefined_and_invalid_counts_fail(self):
        value=frame()
        for panel in ("endpoint","cycle_best"):
            for fields in value[panel].values():
                fields["known"]["value"]=dict(correct=0,total=0,rate=None)
        state=module._fold(None,value,SOURCES)
        self.assertEqual(state["alarms"],[])
        for bad in (dict(correct=True,total=2,rate=.5),dict(correct=3,total=2,rate=1.5),
                    dict(correct=0,total=0,rate=0),dict(correct=1,total=2,rate=float("nan"))):
            with self.assertRaises(ValueError): module._metric(bad,"known")

    def test_many_cycles_keep_fixed_reference_inventory_without_nested_history(self):
        state=None
        for cycle in range(1,101):
            state=module._fold(state,frame(cycle),SOURCES)
            if cycle==1: initial_size=len(json.dumps(state))
        self.assertLess(len(json.dumps(state)),initial_size+200)
        self.assertEqual(set(state["best"]),set(BANKS))
        self.assertNotIn("history",state)
        self.assertEqual(state["cycle"],100)

    def test_immutable_public_snapshots_exact_external_pin_and_source_guard(self):
        # Patch only the input adapter: no mocked prediction is called real evidence.
        with patch.object(module,"source_hashes",return_value=SOURCES),patch.object(module,"_frame",return_value=frame()):
            ledger=module.LifetimeRetention.start(object()); saved=ledger.snapshot()
            self.assertEqual(module.LifetimeRetention.restore(saved,expected_sha256=json_digest(saved),parent=object()).snapshot(),saved)
            with self.assertRaises(ValueError): module.LifetimeRetention.restore(saved,expected_sha256="f"*64,parent=object())
            changed=ledger.snapshot(); changed["best"].clear()
            self.assertEqual(ledger.snapshot(),saved)
            with self.assertRaises(TypeError):pickle.dumps(ledger)
        with patch.object(module,"source_hashes",return_value={}):
            with self.assertRaises(RuntimeError):ledger.snapshot()
        with self.assertRaises(ValueError):module._frame({"claim":"completed"})

    def test_pinned_restore_checks_endpoint_counters_best_and_alarm_consistency(self):
        first=module._fold(None,frame(),SOURCES)
        current=frame(2,action=1,reply=1,known=5,ask=8)
        original=module._fold(first,current,SOURCES)
        module._validate_state(original,current,SOURCES)
        for kind in ("endpoint","counter","best","alarms","future","missing"):
            changed=copy.deepcopy(original); key=next(iter(BANKS))
            if kind=="endpoint":changed["endpoint"][key]["known"]["value"]["correct"]+=1
            elif kind=="counter":changed["cycle"]=True
            elif kind=="best":changed["best"][key]["known"]["value"]["rate"]=.99
            elif kind=="alarms":changed["alarms"]=[]
            elif kind=="future":changed["best"][key]["known"]["producer"]["lifetime_updates"]=19
            else:changed["best"][key].pop("paired_reply")
            with self.subTest(kind=kind),self.assertRaises(ValueError):module._validate_state(changed,current,SOURCES)

    def test_restore_authenticates_detached_bytes_if_caller_mutates_during_hash(self):
        first=module._fold(None,frame(),SOURCES)
        current=frame(2,action=1)
        payload=module._fold(first,current,SOURCES)
        expected=copy.deepcopy(payload); pin=json_digest(payload)
        def hash_then_mutate_caller(value):
            digest=json_digest(value)
            key=next(iter(BANKS))
            payload["best"][key]["paired_action"]["value"].update(correct=9,rate=.9)
            payload["alarms"]=module._alarms(payload["endpoint"],payload["best"])
            return digest
        with patch.object(module,"source_hashes",return_value=SOURCES),patch.object(module,"_frame",return_value=current),patch.object(module,"json_digest",side_effect=hash_then_mutate_caller):
            restored=module.LifetimeRetention.restore(payload,expected_sha256=pin,parent=object())
            self.assertEqual(restored.snapshot(),expected)


if __name__=="__main__":unittest.main(verbosity=2)
