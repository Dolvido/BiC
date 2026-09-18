"""Frozen-rule arithmetic for the prospective foundation architecture comparison.

Standard-library JSON/count analysis only: no model, oracle, selection feedback,
inference or training. Callers MUST first run foundation_metrics.validate_metrics
against the actual canonical banks and bind checkpoint/source identities. The
checks here establish declared cell coverage and numerical consistency, not the
origin of predictions or truth of externally supplied bank hashes.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math

SCHEMA = "bic-foundation-variant-analysis-v1"
FAMILIES = ("color", "count", "switch")
SEEDS = ("8462", "8463", "8464")
RATES = (Decimal("0.0003"), Decimal("0.001"), Decimal("0.003"))
VERSION = "bic-shared-foundation-v1"
METRICS = ("query_accuracy", "known_accuracy", "final_accuracy", "final_pair_accuracy",
           "opposite_pair_accuracy", "query_reply_accuracy", "final_reply_pair_accuracy")
CONFIG = dict(width=192, layers=4, heads=4, feedforward=768, max_positions=1024,
              max_turns=12, max_input_bytes=128, max_output_bytes=32)


def expected_cells(role):
    if role not in ("dev", "audit"): raise ValueError("declared dev or audit role required")
    cells = []
    for family in FAMILIES:
        for depth in range(6):
            operators = ("direct",) if depth == 0 else ("copy", "advance") if depth == 1 else ("composed",)
            for operator in operators:
                for turns in (8, 10, 12): cells.append(f"fresh/{family}/d{depth}/{operator}/t{turns}")
        for depth in range(3 if role == "dev" else 2, 6):
            for turns in (8, 10, 12): cells.append(f"composed/{family}/d{depth}/composed/t{turns}")
    return tuple(sorted(cells))


def _json(value):
    def check(item):
        if type(item) is dict:
            if any(type(k) is not str for k in item): raise ValueError("JSON string keys required")
            for child in item.values(): check(child)
        elif type(item) is list:
            for child in item: check(child)
        elif type(item) is float:
            if not math.isfinite(item): raise ValueError("finite JSON values required")
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError("JSON-only evidence required")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value): return hashlib.sha256(_json(value).encode()).hexdigest()


def _integer(value, name, maximum=None):
    if type(value) is not int or value < 0 or maximum is not None and value > maximum:
        raise ValueError(name+" must be a bounded nonnegative integer")


def _finite(value, name, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or maximum is not None and value > maximum:
        raise ValueError(name+" must be finite and within bounds")


def _ratio(value, correct, total, name):
    _integer(total, name+" denominator"); _integer(correct, name+" numerator", total)
    if total == 0:
        if value is not None: raise ValueError(name+" absent denominator requires null")
    else:
        _finite(value, name, 1.)
        if not math.isclose(value, correct/total, rel_tol=0., abs_tol=1e-7):
            raise ValueError(name+" differs from integer counts")


def _validate_row(row, name, role, pairs):
    _, _, _, _, length = name.split("/"); turns, episodes = int(length[1:]), 2*pairs
    bank = row["bank"]
    if (type(bank) is not dict or set(bank) != {"sha256", "version", "episodes", "turns", "role", "config"}
            or bank["version"] != VERSION or bank["role"] != role
            or bank["config"] != CONFIG or type(bank["config"]) is not dict
            or any(type(v) is not int for v in bank["config"].values())):
        raise ValueError("declared foundation role/version/config differs")
    digest = bank["sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("canonical bank hash required")
    for source, key, value in ((bank,"episodes",episodes),(bank,"turns",turns),
            (row,"episodes",episodes),(row,"turns_per_episode",turns),(row,"final_total",episodes)):
        _integer(source[key], key)
        if source[key] != value: raise ValueError("declared episode/turn count differs")
    if (row["control"] != "normal" or row["free_running_replies"] is not True
            or row["teacher_used_for_policy"] is not False or row["decoder_prefix"] != "BOS only"):
        raise ValueError("teacher-free normal endpoint evidence required")
    final = row["final_pairs"]
    if type(final) is not dict or set(final) != {"correct", "total"}:
        raise ValueError("complete final paired counts required")
    _integer(final["total"], "final pairs")
    if final["total"] != pairs: raise ValueError("declared final pair denominator differs")
    q, k = row["query_total"], row["known_total"]
    _integer(q, "query total", episodes*turns); _integer(k, "known total", q)
    if k < episodes: raise ValueError("known queries cannot omit final opposite pairs")
    for rate, correct, total in (
        ("query_accuracy",row["query_correct"],q),("known_accuracy",row["known_correct"],k),
        ("final_accuracy",row["final_correct"],episodes),("final_pair_accuracy",final["correct"],pairs),
        ("opposite_pair_accuracy",row["opposite_pair_correct"],row["opposite_pair_total"]),
        ("query_reply_accuracy",row["query_reply_correct"],q),
        ("final_reply_pair_accuracy",row["final_reply_pair_correct"],pairs)):
        _ratio(row[rate], correct, total, rate)
    if (not max(0,row["final_correct"]-pairs) <= final["correct"] <= min(pairs,row["final_correct"]//2)
            or row["final_correct"] > row["known_correct"]):
        raise ValueError("final pair/query correctness is inconsistent")
    confusion = row["confusion_matrix"]
    if type(confusion) is not list or len(confusion) != 4 or any(type(v) is not list or len(v)!=4 for v in confusion):
        raise ValueError("four-way confusion counts required")
    for values in confusion:
        for value in values: _integer(value, "confusion")
    totals = [sum(v) for v in confusion]
    if (totals[3] != 0 or sum(totals) != q or sum(totals[:2]) != k
            or min(totals[:2]) < pairs or sum(confusion[i][i] for i in range(3)) != row["query_correct"]
            or confusion[0][0]+confusion[1][1] != row["known_correct"]):
        raise ValueError("query/known/confusion totals differ")
    if type(row["per_target"]) is not dict or set(row["per_target"]) != {"0","1","2"}:
        raise ValueError("all query target counts required")
    for target in range(3):
        score = row["per_target"][str(target)]
        if type(score) is not dict or set(score) != {"correct","total","accuracy"}:
            raise ValueError("target score fields differ")
        _ratio(score["accuracy"],score["correct"],score["total"],"target accuracy")
        if score["total"] != totals[target] or score["correct"] != confusion[target][target]:
            raise ValueError("target scores differ from confusion")
    for key,value in (("ask_true",totals[2]),("ask_predicted",sum(v[2] for v in confusion)),("ask_correct",confusion[2][2])):
        _integer(row[key],key)
        if row[key] != value: raise ValueError("ASK counts differ")
    _ratio(row["ask_precision"],row["ask_correct"],row["ask_predicted"],"ASK precision")
    _ratio(row["ask_recall"],row["ask_correct"],row["ask_true"],"ASK recall")
    by_turn = row["by_turn"]
    if type(by_turn) is not list or len(by_turn)!=turns: raise ValueError("every turn required")
    for index,score in enumerate(by_turn):
        for key in ("turn","correct","total","opposite_pair_correct","opposite_pair_total"): _integer(score[key],key)
        total, paired, correct = score["total"], score["opposite_pair_total"], score["correct"]
        if score["turn"] != index or total > episodes or paired > total//2:
            raise ValueError("turn identity/pair denominator differs")
        _ratio(score["accuracy"],correct,total,"turn accuracy")
        if not max(0,correct-(total-paired)) <= score["opposite_pair_correct"] <= min(paired,correct//2):
            raise ValueError("turn complete pairs contradict correct queries")
    if (sum(r["total"] for r in by_turn)!=q or sum(r["correct"] for r in by_turn)!=row["query_correct"]
            or sum(r["opposite_pair_total"] for r in by_turn)!=row["opposite_pair_total"]
            or sum(r["opposite_pair_correct"] for r in by_turn)!=row["opposite_pair_correct"]
            or by_turn[-1]["total"]!=episodes or by_turn[-1]["correct"]!=row["final_correct"]
            or by_turn[-1]["opposite_pair_total"]!=pairs or by_turn[-1]["opposite_pair_correct"]!=final["correct"]):
        raise ValueError("turn/global/final counts differ")
    _integer(row["reply_parseable_queries"],"parseable replies",q)
    if (row["reply_parseable_queries"] < row["query_reply_correct"]
            or not max(0,row["query_reply_correct"]-(q-pairs)) <= row["final_reply_pair_correct"] <= min(pairs,row["query_reply_correct"]//2)):
        raise ValueError("reply correctness/parseability contradict complete pairs")
    _finite(row["action_reply_agreement"],"action/reply agreement",1.)
    agreement = round(row["action_reply_agreement"]*q)
    if agreement > row["reply_parseable_queries"] or not math.isclose(row["action_reply_agreement"],agreement/q,rel_tol=0.,abs_tol=1e-7):
        raise ValueError("agreement has no integer count")
    _finite(row["query_loss"],"query loss"); _finite(row["brier_score"],"Brier score",2.+1e-6)
    _finite(row["seconds"],"score seconds")


def _validate(metrics, role):
    try:
        _json(metrics)
        if (type(metrics) is not dict or set(metrics) != {"per_bank", *("macro_"+k for k in METRICS)}
                or type(metrics["per_bank"]) is not dict or set(metrics["per_bank"]) != set(expected_cells(role))):
            raise ValueError("exact declared evaluator fields and complete cell set required")
        pairs = 16 if role == "dev" else 32
        identities = {}
        for name,row in metrics["per_bank"].items():
            if type(row) is not dict: raise ValueError("metric rows must be dictionaries")
            _validate_row(row,name,role,pairs); identities[name] = row["bank"]
        if len({row["sha256"] for row in identities.values()}) != len(identities):
            raise ValueError("distinct canonical cells require distinct bank identities")
        for key in METRICS:
            value = metrics["macro_"+key]; _finite(value,"macro "+key,1.)
            expected = sum(row[key] for row in metrics["per_bank"].values())/len(identities)
            if not math.isclose(value,expected,rel_tol=0.,abs_tol=1e-7): raise ValueError("macro differs from full cell set")
        return _digest(identities)
    except (KeyError,TypeError,AttributeError,ZeroDivisionError,OverflowError) as error:
        raise ValueError("malformed foundation variant metric evidence") from error


def _pair(row, modality):
    count = row["final_pairs"]["correct"] if modality == "action" else row["final_reply_pair_correct"]
    return Fraction(count,row["final_pairs"]["total"])


def _balanced(row):
    return sum((Fraction(row["per_target"][str(t)]["correct"],row["per_target"][str(t)]["total"]) for t in (0,1)),Fraction())/2


def _count(correct,total): return dict(correct=correct,total=total,rate=correct/total if total else None)


def _aggregate(rows):
    rows = list(rows)
    if not rows: raise ValueError("an empty group must not disappear from analysis")
    action = sum((_pair(row,"action") for row in rows),Fraction())/len(rows)
    reply = sum((_pair(row,"reply") for row in rows),Fraction())/len(rows)
    known = sum((_balanced(row) for row in rows),Fraction())/len(rows)
    total = sum(row["final_pairs"]["total"] for row in rows)
    return dict(cells=len(rows), components=dict(paired_action=float(action),paired_reply=float(reply),
                target_balanced_known=float(known),composite=float((action+reply+known)/3)),
        counts=dict(action_pairs=_count(sum(r["final_pairs"]["correct"] for r in rows),total),
            reply_pairs=_count(sum(r["final_reply_pair_correct"] for r in rows),total),
            known=_count(sum(r["known_correct"] for r in rows),sum(r["known_total"] for r in rows)),
            per_target={str(t):_count(sum(r["per_target"][str(t)]["correct"] for r in rows),
                                     sum(r["per_target"][str(t)]["total"] for r in rows)) for t in range(3)},
            unknown_ask=_count(sum(r["ask_correct"] for r in rows),sum(r["ask_true"] for r in rows)),
            unsupported_ask=dict(count=sum(r["ask_predicted"]-r["ask_correct"] for r in rows),
                total=sum(r["known_total"] for r in rows),
                rate=sum(r["ask_predicted"]-r["ask_correct"] for r in rows)/sum(r["known_total"] for r in rows))),
        unsupported_ask_macro=sum((r["ask_predicted"]-r["ask_correct"])/r["known_total"] for r in rows)/len(rows))


def calibration_rank(metrics):
    """Validate90 dev cells and return declared rounded rank plus full vectors."""
    bank_digest = _validate(metrics,"dev"); rows = metrics["per_bank"]
    by_domain, fractions = {}, []
    for family in FAMILIES:
        selected = [r for n,r in rows.items() if n.split("/")[1]==family]
        by_domain[family] = _aggregate(selected)
        fractions.append(sum((_pair(r,"action")+_pair(r,"reply")+_balanced(r) for r in selected),Fraction())/(3*len(selected)))
    score = float(sum(fractions,Fraction())/len(FAMILIES))
    return dict(schema=SCHEMA,rank=round(score,12),score_unrounded=score,by_domain=by_domain,
        by_cell={name:_aggregate([row]) for name,row in sorted(rows.items())},bank_identity_sha256=bank_digest,
        cells=90,pairs_per_cell=16,role="dev",automatic_promotion=False,
        scope="Engineered calibration score: equal domain, equal cell, equal action/reply/target-balanced-known components. Upstream canonical validation and weight/source binding remain required.")


def choose_rate(candidates):
    if type(candidates) is not dict or len(candidates)!=3: raise ValueError("all three declared calibration rates required")
    numeric = {}
    for name in candidates:
        if type(name) is not str: raise ValueError("rate keys must be strings")
        try: rate = Decimal(name)
        except InvalidOperation as error: raise ValueError("invalid rate string") from error
        if not rate.is_finite() or rate not in RATES or rate in numeric.values(): raise ValueError("rate set or duplicate numeric alias differs")
        numeric[name] = rate
    ranks = {name:calibration_rank(candidates[name]) for name in sorted(candidates,key=lambda n:numeric[n])}
    if len({r["bank_identity_sha256"] for r in ranks.values()}) != 1: raise ValueError("rate candidates used different development banks")
    selected = max(ranks,key=lambda name:(ranks[name]["rank"],-numeric[name]))
    return dict(schema=SCHEMA,selected_rate=selected,selected_learning_rate=float(numeric[selected]),rate=float(numeric[selected]),
        candidates=ranks,tie_rule="round scalar to12 decimal places, then lower learning rate",automatic_promotion=False)


def _counts_fraction(group,modality):
    count = group["counts"][modality+"_pairs"]
    return Fraction(count["correct"],count["total"])


def comparison_screen(results):
    """Descriptive paired-seed/noncollapse screen; never mastery or promotion."""
    if type(results) is not dict or set(results)!={"flat","hierarchical"}: raise ValueError("both declared architectures required")
    digests = set()
    for architecture in ("flat","hierarchical"):
        if type(results[architecture]) is not dict or set(results[architecture])!=set(SEEDS): raise ValueError("all three exact paired seeds required")
        for seed in SEEDS: digests.add(_validate(results[architecture][seed],"audit"))
    if len(digests)!=1: raise ValueError("architectures/seeds must use the identical sealed audit banks")
    panels = {}
    for panel in ("fresh_primitive","held_composed"):
        names = [name for name in expected_cells("audit") if
                 (name.startswith("fresh/") and name.split("/")[2] in ("d0","d1") if panel=="fresh_primitive" else name.startswith("composed/"))]
        per_seed = {}
        for seed in SEEDS:
            per_seed[seed] = {}
            for architecture in ("flat","hierarchical"):
                rows = results[architecture][seed]["per_bank"]
                per_seed[seed][architecture] = dict(overall=_aggregate(rows[n] for n in names),
                    by_domain={f:_aggregate(rows[n] for n in names if n.split("/")[1]==f) for f in FAMILIES})
            flat, candidate = (per_seed[seed][a] for a in ("flat","hierarchical"))
            delta = sum((_counts_fraction(candidate["overall"],m)-_counts_fraction(flat["overall"],m) for m in ("action","reply")),Fraction())/2
            domains = {}
            for family in FAMILIES:
                domains[family] = {m:float(_counts_fraction(candidate["by_domain"][family],m)-_counts_fraction(flat["by_domain"][family],m)) for m in ("action","reply")}
            nonzero = {f:{m:candidate["by_domain"][f]["counts"][m+"_pairs"]["correct"]>0 for m in ("action","reply")} for f in FAMILIES}
            per_seed[seed].update(paired_composite_delta=float(delta),positive_paired_composite=delta>0,
                                 by_domain_paired_delta=domains,candidate_nonzero=nonzero)
        pooled = {}
        for family in FAMILIES:
            pooled[family] = {}
            for modality in ("action","reply"):
                counts = {}
                for architecture in ("flat","hierarchical"):
                    rows = [per_seed[seed][architecture]["by_domain"][family]["counts"][modality+"_pairs"] for seed in SEEDS]
                    counts[architecture] = _count(sum(r["correct"] for r in rows),sum(r["total"] for r in rows))
                a,b = counts["hierarchical"],counts["flat"]
                delta = Fraction(a["correct"],a["total"])-Fraction(b["correct"],b["total"])
                pooled[family][modality] = dict(**counts,delta=float(delta),nonnegative=delta>=0)
        criteria = dict(every_seed_improves=all(r["positive_paired_composite"] for r in per_seed.values()),
            every_pooled_domain_modality_nonnegative=all(row["nonnegative"] for domain in pooled.values() for row in domain.values()),
            every_candidate_domain_seed_modality_nonzero=all(flag for row in per_seed.values() for domain in row["candidate_nonzero"].values() for flag in domain.values()))
        panels[panel] = dict(cell_names=names,cells=len(names),pairs_per_cell=32,per_seed=per_seed,
                            pooled_by_domain_modality=pooled,criteria=criteria,passes=all(criteria.values()))
    return dict(schema=SCHEMA,panels=panels,promising_comparison=all(p["passes"] for p in panels.values()),
        bank_identity_sha256=next(iter(digests)),seed_order=list(SEEDS),automatic_promotion=False,
        omitted_from_primary_screen=[n for n in expected_cells("audit") if n.startswith("fresh/") and n.split("/")[2] not in ("d0","d1")],
        scope="Prospective descriptive screen on paired initialization seeds and fixed banks. Not statistical significance, mastery or general intelligence. Weak operators, known/ASK tradeoffs and retention remain limits; upstream canonical/weight/source validation is required.")
