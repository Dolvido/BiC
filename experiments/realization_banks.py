"""Prospective fixed/fresh realization banks over immutable composition v1.

Name and value controls share the same training procedure and the same changed
coordinates within each panel triplet. Exact individual transcripts are excluded
across all new banks and the retired composition study. Syntactic ancestry IDs
may recur historically; this is not semantic-world or unseen-vocabulary proof.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from experiments import composition_curriculum as curriculum
from experiments.composition_banks import _check_pair, _bank_stats, known_composed_queries
from experiments.train_cognitive import fingerprint_rows


SCHEMA = 'bic-realization-banks-v1'
FAMILIES = curriculum.FAMILIES
LENGTHS = (8, 10, 12)
PANELS = ('name_only', 'value_only', 'both', 'composed')
TRAIN_PAIRS, PANEL_PAIRS = 256, 64
SEEDS = {'train': 170_000_000, 'development': 180_000_000, 'audit': 190_000_000}
NAMING_OFFSET, VALUE_OFFSET, COMPOSED_OFFSET = 100_000_000, 200_000_000, 300_000
HISTORICAL_DIRECTORY = Path(__file__).resolve().parents[1] / 'runs' / 'composition-study-local'
MAX_ATTEMPTS_PER_PAIR = 1_000


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf8')).hexdigest()


def transcript_digest(row):
    """SHA256 of canonical JSON exact observation-text list, excluding metadata."""
    return _digest([turn['text'] for turn in row['turns']])


def _all_banks(banks):
    yield from (rows for buckets in banks['train'].values() for rows in buckets.values())
    yield from banks['development'].values()
    yield from banks['audit'].values()


def _anonymous_realization(row):
    """Parsed event transcript with aliases replaced by actual original roles.

    Retains typed constants and order; this is syntactic, not semantic equivalence.
    """
    inverse = {alias: role for role, alias in curriculum.naming_map(row['recipe']['naming_seed']).items()}
    events = []
    for turn in row['turns']:
        family, event = curriculum.parse_sentence(turn['text'])
        event = dict(event)
        for key in ('name', 'source'):
            if key in event:
                event[key] = inverse[event[key]]
        events.append([family, event])
    return _digest(events)


def _map_id(row):
    return _digest(curriculum.naming_map(row['recipe']['naming_seed']))


@lru_cache(maxsize=4)
def _read_historical(directory, expected_digest):
    import torch
    path = Path(directory) / 'banks.pt'
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError('retired composition bank digest differs from its protocol')
    banks = torch.load(path, map_location='cpu', weights_only=True)
    rows = [row for values in _all_banks(banks) for row in values]
    return frozenset(transcript_digest(row) for row in rows), len(rows)


def historical_evidence():
    """Authenticate the local retired bank file; no models or scores are read."""
    protocol_path = HISTORICAL_DIRECTORY / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf8'))
    expected = protocol['banks_file_sha256']
    # Recheck bytes even when parsed historical transcripts are cached.
    actual = hashlib.sha256((HISTORICAL_DIRECTORY / 'banks.pt').read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('retired composition bank digest differs from its protocol')
    digests, count = _read_historical(str(HISTORICAL_DIRECTORY), expected)
    return {'directory': str(HISTORICAL_DIRECTORY), 'banks_sha256': expected,
            'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'episodes': count, 'unique_transcripts': len(digests)}


def historical_transcript_digests():
    evidence = historical_evidence()
    digests, _ = _read_historical(str(HISTORICAL_DIRECTORY), evidence['banks_sha256'])
    return sorted(digests)


def protected_transcript_digests(banks, *, include_historical=True):
    """Fresh-training exclusions: new dev/audit plus retired rows, never new train."""
    if type(include_historical) is not bool:
        raise ValueError('include_historical must be boolean')
    values = set(historical_transcript_digests()) if include_historical else set()
    values.update(transcript_digest(row) for group in ('development', 'audit')
                  for rows in banks[group].values() for row in rows)
    return sorted(values)


def protected_transcripts(banks):
    """Runner-facing stable alias; includes all retired and new evaluation rows."""
    return protected_transcript_digests(banks)


def _seed(group, family, turns):
    return SEEDS[group] + FAMILIES.index(family) * 1_000_000 + turns * 10_000


def _recipe(pair):
    row = pair[0]
    return {'family': row['family'], 'split': row['split'], **deepcopy(row['recipe']),
            'structure_id': row['structure_id'], 'program_id': row['program_id'],
            'pair_sha256': fingerprint_rows(pair)}


def _stats(rows):
    result = _bank_stats(rows)
    opposite_by_turn = [sum({rows[i]['turns'][t]['target'], rows[i+1]['turns'][t]['target']} == {0, 1}
                            for i in range(0, len(rows), 2)) for t in range(result['turns'])]
    result.update({'final_opposite_pair_total': len(rows) // 2,
                   'opposite_pair_total': sum(opposite_by_turn),
                   'opposite_pair_total_by_turn': opposite_by_turn,
                   'query_target_counts': {str(target): sum(turn['kind'] == 'query' and turn['target'] == target
                       for row in rows for turn in row['turns']) for target in range(4)},
                   'distinct_anonymous_realization_transcripts': len({_anonymous_realization(row) for row in rows}),
                   'transcript_sha256': sorted(transcript_digest(row) for row in rows)})
    return result


def bank_manifest(banks):
    return {'train': {family: {str(turns): _stats(rows) for turns, rows in buckets.items()}
                      for family, buckets in banks['train'].items()},
            **{group: {name: _stats(rows) for name, rows in banks[group].items()}
               for group in ('development', 'audit')}}


def _check_familiar(source, panels):
    name, value, both = (panels[key] for key in ('name_only', 'value_only', 'both'))
    for original, named, valued, combined in zip(source, name, value, both):
        for row in (named, valued, combined):
            if any(row[key] != original[key] for key in ('program_id', 'structure_id', 'query_ancestries')):
                raise ValueError('familiar panel changed procedure or causal ancestry')
            if row['recipe']['seed'] != original['recipe']['seed']:
                raise ValueError('familiar panel changed original procedure seed')
        if (named['recipe']['value_seed'] != original['recipe']['value_seed']
                or valued['recipe']['naming_seed'] != original['recipe']['naming_seed']
                or combined['recipe']['naming_seed'] != named['recipe']['naming_seed']
                or combined['recipe']['value_seed'] != valued['recipe']['value_seed']):
            raise ValueError('familiar panel changed the wrong realization coordinate')
        if _map_id(named) == _map_id(original) or _map_id(combined) != _map_id(named):
            raise ValueError('fresh naming coordinate must change the actual bijection')
        if _anonymous_realization(named) != _anonymous_realization(original):
            raise ValueError('name-only panel changed value events')
        if _anonymous_realization(combined) != _anonymous_realization(valued):
            raise ValueError('both panel differs from its value-only realization')
    if {_anonymous_realization(row) for row in source} == {_anonymous_realization(row) for row in value}:
        raise ValueError('fresh value coordinate did not change actual anonymous realization')


def verify_boundaries(banks):
    """Authenticate rows, factor controls and current/historical transcript exclusion."""
    if not isinstance(banks, dict) or set(banks) != {'train', 'development', 'audit'}:
        raise ValueError('expected train/development/audit realization banks')
    if set(banks['train']) != set(FAMILIES) or any(set(v) != set(LENGTHS) for v in banks['train'].values()):
        raise ValueError('training requires every family and length bucket')
    expected = {f'{panel}/{family}/t{turns}' for panel in PANELS for family in FAMILIES for turns in LENGTHS}
    if any(set(banks[group]) != expected for group in ('development', 'audit')):
        raise ValueError('evaluation panel set differs from declared study')
    if any(not isinstance(rows, (list, tuple)) or not rows or len(rows) % 2 for rows in _all_banks(banks)):
        raise ValueError('every bank requires nonempty adjacent complete pairs')
    historical = set(historical_transcript_digests())
    exact, train_queries, dev_queries = set(), set(), set()
    composed = {'development': set(), 'audit': set()}
    counts = Counter()
    def admit(rows, group, family, turns, partition):
        admission = 'dev' if group == 'development' else group
        for i in range(0, len(rows), 2):
            _check_pair(rows[i:i+2], family=family, turns=turns, admission=admission, structure_split=partition)
        for row in rows:
            digest = transcript_digest(row)
            if digest in exact or digest in historical:
                raise ValueError('exact observation transcript reused in current or historical banks')
            exact.add(digest)
            queries = known_composed_queries(row)
            if group == 'train':
                train_queries.update(q['structure_id'] for q in queries)
            elif group == 'development':
                if any(q['structure_partition'] == 'audit' for q in queries):
                    raise ValueError('development exposes supervised audit ancestry')
                dev_queries.update(q['structure_id'] for q in queries)
            counts[f'{group}_episodes'] += 1
    for family, buckets in banks['train'].items():
        for turns, rows in buckets.items():
            admit(rows, 'train', family, turns, 'train')
    for group in ('development', 'audit'):
        partition = 'dev' if group == 'development' else 'audit'
        for family in FAMILIES:
            for turns in LENGTHS:
                source = banks['train'][family][turns]
                familiar = {p: banks[group][f'{p}/{family}/t{turns}'] for p in PANELS[:3]}
                lengths = {len(rows) for rows in familiar.values()}
                if len(lengths) != 1 or next(iter(lengths)) > len(source):
                    raise ValueError('familiar panels must use matching first training recipes')
                for i in range(0, next(iter(lengths)), 2):
                    _check_familiar(source[i:i+2], {p: rows[i:i+2] for p, rows in familiar.items()})
                for panel in PANELS:
                    rows = banks[group][f'{panel}/{family}/t{turns}']
                    admit(rows, group, family, turns, partition if panel == 'composed' else 'train')
                    if panel == 'composed':
                        composed[group].update(row['structure_id'] for row in rows)
    if composed['development'] & train_queries or composed['audit'] & (train_queries | dev_queries):
        raise ValueError('held final ancestry overlaps an earlier supervised composition')
    return {**dict(counts), 'unique_exact_transcripts': len(exact), 'historical_excluded_transcripts': len(historical),
            'training_known_composed_query_ids': sorted(train_queries),
            'development_known_composed_query_ids': sorted(dev_queries),
            'development_final_composed_ids': sorted(composed['development']),
            'audit_final_composed_ids': sorted(composed['audit']),
            'global_exact_transcript_disjoint': True, 'historical_exact_transcript_disjoint': True,
            'supervised_composed_query_boundary_verified': True, 'familiar_factor_controls_verified': True}


def prepare_banks(*, with_diagnostics=False, train_pairs=TRAIN_PAIRS, panel_pairs=PANEL_PAIRS):
    """Build fresh bank instances without model access, evaluation or file writes."""
    if type(with_diagnostics) is not bool:
        raise ValueError('with_diagnostics must be boolean')
    if type(train_pairs) is not int or type(panel_pairs) is not int or not 1 <= panel_pairs <= train_pairs:
        raise ValueError('positive panel_pairs must not exceed train_pairs')
    historical = set(historical_transcript_digests())
    exact = set(historical)
    banks = {'train': {family: {} for family in FAMILIES}, 'development': {}, 'audit': {}}
    selections = []

    def build(group, family, turns, count, familiar=False):
        admission = 'dev' if group == 'development' else group
        partition = 'train' if group == 'train' or familiar else admission
        panel_names = PANELS[:3] if familiar else ('train' if group == 'train' else 'composed',)
        base = _seed(group, family, turns) + (COMPOSED_OFFSET if panel_names == ('composed',) else 0)
        output = {p: [] for p in panel_names}
        accepted = {p: [] for p in panel_names}
        attempts, rejected = 0, Counter()
        while len(next(iter(output.values()))) < count * 2:
            if attempts >= max(1_000, count * MAX_ATTEMPTS_PER_PAIR):
                raise ValueError('insufficient disjoint realizations within bounded search')
            index = len(next(iter(output.values()))) // 2
            candidate = base + attempts
            attempts += 1
            if familiar:
                source = banks['train'][family][turns][2*index:2*index+2]
                original = source[0]['recipe']
                names, values = candidate + NAMING_OFFSET, candidate + VALUE_OFFSET
                coordinates = {'name_only': (names, original['value_seed']),
                               'value_only': (original['naming_seed'], values), 'both': (names, values)}
                pairs = {p: curriculum.generate_pair(family, original['seed'], split=admission, turns=turns,
                         naming_seed=n, value_seed=v, structure_split='train') for p, (n, v) in coordinates.items()}
                try:
                    _check_familiar(source, pairs)
                except ValueError:
                    rejected['unchanged_realization'] += 1
                    continue
            else:
                pairs = {panel_names[0]: curriculum.generate_pair(family, candidate, split=admission, turns=turns,
                         naming_seed=candidate + NAMING_OFFSET, value_seed=candidate + VALUE_OFFSET,
                         structure_split=partition)}
            for pair in pairs.values():
                _check_pair(pair, family=family, turns=turns, admission=admission, structure_split=partition)
            if group == 'development' and any(q['structure_partition'] == 'audit'
                    for pair in pairs.values() for row in pair for q in known_composed_queries(row)):
                rejected['development_audit_query'] += 1
                continue
            digests = [transcript_digest(row) for pair in pairs.values() for row in pair]
            if len(set(digests)) != len(digests) or any(d in exact for d in digests):
                rejected['historical_transcript' if any(d in historical for d in digests) else 'current_transcript'] += 1
                continue
            exact.update(digests)
            for panel, pair in pairs.items():
                output[panel].extend(deepcopy(pair))
                accepted[panel].append(_recipe(pair))
        selections.append({'group': group, 'family': family, 'turns': turns, 'panels': list(panel_names),
                           'base_seed': base, 'requested_pairs_per_panel': count, 'candidate_attempts': attempts,
                           'rejected_pairs_or_factor_triplets': dict(rejected), 'recipes': accepted})
        return output

    for family in FAMILIES:
        for turns in LENGTHS:
            banks['train'][family][turns] = build('train', family, turns, train_pairs)['train']
    for group in ('development', 'audit'):
        for family in FAMILIES:
            for turns in LENGTHS:
                for familiar in (True, False):
                    values = build(group, family, turns, panel_pairs, familiar=familiar)
                    banks[group].update({f'{p}/{family}/t{turns}': rows for p, rows in values.items()})
    boundary = verify_boundaries(banks)
    details = {'schema': SCHEMA, 'seed_namespaces': dict(SEEDS), 'family_seed_stride': 1_000_000,
               'length_seed_stride': 10_000, 'naming_seed_offset': NAMING_OFFSET, 'value_seed_offset': VALUE_OFFSET,
               'composed_panel_offset': COMPOSED_OFFSET, 'training_pairs_per_bucket': train_pairs,
               'evaluation_pairs_per_panel': panel_pairs, 'selection': selections,
               'historical_evidence': historical_evidence(), 'boundaries': boundary,
               'factor_note': 'Familiar triplets reuse first training procedures. Name-only and both share new naming; value-only and both share new values. Unchanged coordinates are exact original seeds.',
               'scope_note': 'Current held ancestry partitions exclude earlier current supervised compositions. Historical ancestry IDs may recur; only exact individual observation transcripts are excluded historically.',
               'semantic_note': 'Aliases use shared vocabulary; name maps, value seeds, normalized event transcripts and syntactic programs are not distinct semantic algorithms. Values also change increments/query truth and byte lengths.'}
    return (banks, details) if with_diagnostics else banks
