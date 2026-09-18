"""Focused canonical CPU proof; no full bank preparation or archive reads.

Twelve requested parent pairs span all families and primitive/short/deep transfer
cells. All nested canonical reconstructions are counted separately, <=200 calls.
"""
from copy import deepcopy
import unittest

from experiments import shared_acquisition_transfer_data as data
from experiments import foundation_curriculum as foundation
from experiments import foundation_layout_curriculum as layout
from experiments import foundation_tutor_loop_data as io

WORK = io.WorkLedger(layout=layout.WorkLedger())
COUNTS = dict(test_calls=0, requested_pairs=0, malformed_variants=0, mock_admission_cases=0)
FIXTURES = {}


class TransferBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounting = data.generation_accounting(WORK, maximum=200)
        cls.accounting.__enter__()
        cls.addClassCleanup(cls.accounting.__exit__, None, None, None)
        owners, blocked = {}, set()
        for transfer, depth, turns in ((False, 0, 8), (False, 1, 10), (True, 2, 8), (True, 5, 12)):
            for family in foundation.FAMILIES:
                rejected = {}
                from collections import Counter
                views, records = data.build_cell(family, depth, turns, transfer=transfer, seed=852990001,
                    pairs=1, blocked=blocked, owners=owners, work=WORK, rejections=Counter())
                COUNTS['requested_pairs'] += 1
                FIXTURES[family, depth, turns] = (views, records)

    def setUp(self):
        COUNTS['test_calls'] += 1

    def test_01_expected_partition_and_shared_primitives(self):
        for (family, depth, turns), (views, records) in FIXTURES.items():
            for name, rows in views.items():
                self.assertEqual(len(rows), 2)
                self.assertEqual({r['structure_partition'] for r in rows}, {'audit' if name.startswith('transfer') else 'train'})
                self.assertEqual({r['family'] for r in rows}, {family})
                self.assertEqual({r['depth'] for r in rows}, {depth})
                self.assertEqual({len(r['turns']) for r in rows}, {turns})
                self.assertEqual({r['primitive_shared'] for r in rows}, {depth < 2})

    def test_02_both_transfer_views_preserve_every_query_and_byte_multiset(self):
        from collections import Counter
        for (_, depth, _), (views, _) in FIXTURES.items():
            if depth < 2: continue
            for first, second in zip(views['transfer_original'], views['transfer_varied']):
                self.assertEqual(Counter(map(layout._json, first['turns'])), Counter(map(layout._json, second['turns'])))
                self.assertEqual({q['query_id']:q['version_signature'] for q in first['queries']},
                                 {q['query_id']:q['version_signature'] for q in second['queries']})
                self.assertEqual(first['anchor']['version_signature_sha256'], second['anchor']['version_signature_sha256'])
                self.assertEqual(first['turns'][first['anchor']['turn_index']]['target'],
                                 second['turns'][second['anchor']['turn_index']]['target'])
                self.assertEqual(second['layout_summary']['changed'], second['recipe']['permutation'] != list(range(len(second['turns']))))

    def test_03_family_independent_parent_and_layout_seeds(self):
        for depth, turns in ((0, 8), (1, 10), (2, 8), (5, 12)):
            records = [FIXTURES[f, depth, turns][1][0] for f in foundation.FAMILIES]
            self.assertEqual(len({r['layout_seed'] for r in records}), 1)
            for key in ('seed', 'naming_seed', 'value_seed'):
                self.assertEqual(len({r['recipe'][key] for r in records}), 1)

    def test_04_atomic_rejection_and_only_matching_view_member_overlap(self):
        views = deepcopy(FIXTURES['color', 2, 8][0]); owners = {}
        blocked = {data.transcript(views['transfer_varied'][1])}
        self.assertEqual(data.admit_views(views, blocked=blocked, owners=owners), (False, 'excluded_transcript'))
        self.assertEqual(owners, {})
        identical = {'transfer_original': views['transfer_original'], 'transfer_varied': deepcopy(views['transfer_original'])}
        self.assertEqual(data.admit_views(identical, blocked=set(), owners=owners), (True, None))
        self.assertEqual(len(owners), 2)
        before = deepcopy(owners)
        self.assertEqual(data.admit_views(identical, blocked=set(), owners=owners), (False, 'different_pair_transcript'))
        self.assertEqual(owners, before)
        COUNTS['mock_admission_cases'] += 3

    def test_05_foreign_pair_same_text_is_rejected(self):
        views = deepcopy(FIXTURES['switch', 2, 8][0]); owners = {}
        views['transfer_varied'] = deepcopy(views['transfer_original'])
        for row in views['transfer_varied']: row['recipe']['base_pair_sha256'] = 'a' * 64
        self.assertEqual(data.admit_views(views, blocked=set(), owners=owners), (False, 'different_pair_transcript'))
        self.assertEqual(owners, {})
        COUNTS['mock_admission_cases'] += 1

    def test_06_corrupted_or_wrong_split_rows_fail_closed(self):
        row = FIXTURES['color', 2, 8][0]['transfer_original'][0]
        recipe = row['recipe']['base_recipe']
        pair = foundation.generate_pair('color', recipe['seed'], depth=recipe['depth'], turns=recipe['turns'],
            split='audit', naming_seed=recipe['naming_seed'], value_seed=recipe['value_seed'], structure_split='audit')
        with self.assertRaisesRegex(ValueError, 'panel partition'):
            data.paired_views(pair, seed=0, transfer=False, work=WORK)
        malformed = deepcopy(pair); malformed[0]['turns'][-1]['target'] = 2
        with self.assertRaises(ValueError): data.paired_views(malformed, seed=0, transfer=True, work=WORK)
        COUNTS['malformed_variants'] += 2

    def test_07_deadline_precedes_generation(self):
        expired = io.WorkLedger(deadline=0, layout=layout.WorkLedger())
        before = WORK.counts['canonical_generate_attempts']
        with self.assertRaises(TimeoutError):
            data.build_cell('color', 2, 8, transfer=True, seed=1, pairs=1, blocked=set(), owners={},
                work=expired, rejections={})
        self.assertEqual(WORK.counts['canonical_generate_attempts'], before)

    def test_08_generator_budget_is_hard(self):
        tiny = io.WorkLedger(layout=layout.WorkLedger())
        before = WORK.counts['canonical_generate_attempts']
        with data.generation_accounting(tiny, maximum=0):
            with self.assertRaisesRegex(ValueError, 'generation allowance'):
                foundation.generate_pair('color', 1, depth=0, turns=8)
        self.assertEqual(WORK.counts['canonical_generate_attempts'], before)


if __name__ == '__main__': unittest.main()
