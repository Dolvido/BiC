"""Realization factor controls, canonical truth and prospective data boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import composition_curriculum as curriculum
from experiments import realization_banks as banks


class RealizationBanksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = patch.object(banks, 'historical_transcript_digests', return_value=['a' * 64])
        cls.evidence = patch.object(banks, 'historical_evidence', return_value={'episodes': 1, 'unique_transcripts': 1})
        cls.history.start(); cls.evidence.start()
        cls.addClassCleanup(cls.history.stop); cls.addClassCleanup(cls.evidence.stop)
        cls.values, cls.diagnostics = banks.prepare_banks(train_pairs=4, panel_pairs=2, with_diagnostics=True)

    def test_counts_and_independent_canonical_truth(self):
        boundary = self.diagnostics['boundaries']
        self.assertEqual(boundary['train_episodes'], 72)
        self.assertEqual(boundary['development_episodes'], 144)
        self.assertEqual(boundary['audit_episodes'], 144)
        self.assertEqual(len(self.values['development']), 36)
        for rows in banks._all_banks(self.values):
            for i in range(0, len(rows), 2):
                self.assertTrue(curriculum.validate_pair(rows[i:i+2]))
                for row in rows[i:i+2]:
                    self.assertEqual(curriculum.english_oracle(row), [t['target'] for t in row['turns']])

    def test_factor_panels_preserve_procedure_and_match_changed_coordinates(self):
        for role in ('development', 'audit'):
            for family in banks.FAMILIES:
                for turns in banks.LENGTHS:
                    for i in range(4):
                        source = self.values['train'][family][turns][i]
                        n,v,b = [self.values[role][f'{p}/{family}/t{turns}'][i]
                                 for p in ('name_only','value_only','both')]
                        self.assertEqual(n['recipe']['value_seed'], source['recipe']['value_seed'])
                        self.assertEqual(v['recipe']['naming_seed'], source['recipe']['naming_seed'])
                        self.assertEqual(b['recipe']['naming_seed'], n['recipe']['naming_seed'])
                        self.assertEqual(b['recipe']['value_seed'], v['recipe']['value_seed'])
                        self.assertEqual(banks._anonymous_realization(source), banks._anonymous_realization(n))
                        self.assertEqual(banks._anonymous_realization(v), banks._anonymous_realization(b))
                        self.assertNotEqual(banks._map_id(source), banks._map_id(n))
                        for row in (n,v,b):
                            for field in ('structure_id','program_id','query_ancestries'):
                                self.assertEqual(row[field],source[field])

    def test_all_supervised_ancestry_boundaries_and_exact_transcripts(self):
        boundary = self.diagnostics['boundaries']
        train = set(boundary['training_known_composed_query_ids'])
        dev = set(boundary['development_known_composed_query_ids'])
        self.assertFalse(train & set(boundary['development_final_composed_ids']))
        self.assertFalse((train|dev) & set(boundary['audit_final_composed_ids']))
        for rows in self.values['development'].values():
            for row in rows:
                self.assertFalse(any(q['structure_partition']=='audit' for q in banks.known_composed_queries(row)))
        observed = [banks.transcript_digest(row) for rows in banks._all_banks(self.values) for row in rows]
        self.assertEqual(len(observed),len(set(observed)))
        self.assertEqual(len(observed),boundary['unique_exact_transcripts'])

    def test_protected_set_excludes_training_and_includes_retired_evaluation(self):
        actual = set(banks.protected_transcripts(self.values))
        expected = {'a'*64} | {banks.transcript_digest(row) for g in ('development','audit')
                               for rows in self.values[g].values() for row in rows}
        self.assertEqual(actual,expected)
        self.assertFalse(actual & {banks.transcript_digest(row) for fs in self.values['train'].values()
                                  for rows in fs.values() for row in rows})
        self.assertEqual(banks.protected_transcripts(self.values),sorted(actual))
        self.assertEqual(set(banks.protected_transcript_digests(self.values,include_historical=False)),expected-{'a'*64})

    def test_digest_is_exact_text_only_with_unambiguous_boundaries(self):
        row=copy.deepcopy(self.values['train']['color'][8][0])
        expected=hashlib.sha256(json.dumps([t['text'] for t in row['turns']],ensure_ascii=False,
                               separators=(',',':')).encode('utf8')).hexdigest()
        self.assertEqual(banks.transcript_digest(row),expected)
        row['id']='metadata change'; row['turns'][0]['target']=123
        self.assertEqual(banks.transcript_digest(row),expected)
        row['turns'][0]['text']+=' '
        self.assertNotEqual(banks.transcript_digest(row),expected)

    def test_historical_collision_rejects_whole_pair_deterministically(self):
        first=self.values['train']['color'][8][:2]
        forbidden=banks.transcript_digest(first[1])
        with patch.object(banks,'historical_transcript_digests',return_value=[forbidden]):
            result,details=banks.prepare_banks(train_pairs=4,panel_pairs=2,with_diagnostics=True)
            self.assertNotEqual(result['train']['color'][8][0]['recipe']['seed'],first[0]['recipe']['seed'])
            self.assertNotIn(forbidden,{banks.transcript_digest(row) for rows in banks._all_banks(result) for row in rows})
            self.assertGreater(details['selection'][0]['rejected_pairs_or_factor_triplets']['historical_transcript'],0)
            with self.assertRaisesRegex(ValueError,'historical banks'):
                banks.verify_boundaries(self.values)

    def test_manifest_actual_pair_denominators_targets_and_coverage(self):
        manifest=banks.bank_manifest(self.values)
        for rows in manifest['development'].values():
            self.assertEqual(rows['episodes'],4)
            self.assertEqual(rows['final_opposite_pair_total'],2)
            self.assertEqual(rows['opposite_pair_total_by_turn'][-1],2)
            self.assertEqual(rows['opposite_pair_total'],sum(rows['opposite_pair_total_by_turn']))
            self.assertEqual(rows['target_counts_by_turn'][-1],{'0':2,'1':2,'2':0,'3':0})
            self.assertEqual(rows['query_target_counts']['3'],0)
            self.assertGreater(rows['query_target_counts']['2'],0)
            self.assertLessEqual(rows['max_context_tokens'],1024)
            self.assertLessEqual(rows['max_utterance_bytes'],128)
            self.assertEqual(len(rows['transcript_sha256']),4)
            self.assertEqual(rows['distinct_final_structures'],len(rows['final_structure_ids']))

    def test_canonical_but_wrong_factor_change_is_rejected(self):
        changed=copy.deepcopy(self.values)
        old=changed['development']['name_only/color/t8'][0]
        recipe=old['recipe']
        replacement=curriculum.generate_pair('color',recipe['seed'],split='dev',turns=8,
                    naming_seed=recipe['naming_seed'],value_seed=recipe['value_seed']+99999,structure_split='train')
        changed['development']['name_only/color/t8'][:2]=replacement
        with self.assertRaisesRegex(ValueError,'wrong realization coordinate'):
            banks.verify_boundaries(changed)

    def test_tampering_duplicates_and_missing_banks_fail(self):
        for mutate in (
            lambda b:b['audit'].pop('both/color/t8'),
            lambda b:b['train']['count'].pop(12),
            lambda b:b['audit']['composed/switch/t8'][0]['turns'][-1].update(target=2),
            lambda b:b['train']['color'][8].__setitem__(slice(2,4),copy.deepcopy(b['train']['color'][8][:2])),
            lambda b:b['audit'].update({'name_only/color/t8':[]})):
            changed=copy.deepcopy(self.values);mutate(changed)
            with self.assertRaises(ValueError):banks.verify_boundaries(changed)

    def test_determinism_and_mutation_isolation(self):
        values,details=banks.prepare_banks(train_pairs=4,panel_pairs=2,with_diagnostics=True)
        self.assertEqual(values,self.values);self.assertEqual(details,self.diagnostics)
        values['train']['color'][8][0]['turns'][0]['text']='changed'
        self.assertNotEqual(values,self.values)

    def test_invalid_sizes_and_bounded_collision_search(self):
        for kw in ({'train_pairs':True},{'panel_pairs':0},{'train_pairs':2,'panel_pairs':3},{'with_diagnostics':1}):
            with self.assertRaises(ValueError):banks.prepare_banks(**kw)
        with patch.object(banks,'transcript_digest',return_value='collision'):
            with self.assertRaisesRegex(ValueError,'bounded search'):
                banks.prepare_banks(train_pairs=1,panel_pairs=1)


class HistoricalAuthenticationTests(unittest.TestCase):
    def test_changed_retired_bank_bytes_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'banks.pt').write_bytes(b'untrusted or damaged bytes')
            (root/'protocol.json').write_text(json.dumps({'banks_file_sha256':'0'*64}),encoding='utf8')
            with patch.object(banks,'HISTORICAL_DIRECTORY',root):
                with self.assertRaisesRegex(ValueError,'digest differs'):
                    banks.historical_evidence()


if __name__=='__main__':unittest.main()
