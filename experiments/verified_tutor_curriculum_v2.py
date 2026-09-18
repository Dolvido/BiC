"""Versioned finite collision repair; all author choices and admission stay v1.

Candidate zero is byte-identical to v1. Fresh pair retries change only an allowed
realization seed, never parent, motif, family, procedure, split or coverage. The
caller still owns admitted-inventory provenance. No teacher or learner API.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import time
from experiments import verified_tutor_curriculum as original

ROOT = original.ROOT
SCHEMA = "bic-verified-tutor-curriculum-v2"
MAX_CANDIDATES = 64
COLLISION_POLICY = "bounded-per-family-pair-realization-v2"
RECIPE_SCHEMA, CONTRACT_SCHEMA, INVENTORY_SCHEMA = original.RECIPE_SCHEMA, original.CONTRACT_SCHEMA, original.INVENTORY_SCHEMA
FAMILIES, CELLS, REALIZATIONS, COUNTS = original.FAMILIES, original.CELLS, original.REALIZATIONS, original.COUNTS
_json, _hash, _keys, _pin, _id, _integer = original._json, original._hash, original._keys, original._pin, original._id, original._integer
_track, _evidence = original._track, original._evidence
transcript_sha256 = original.transcript_sha256
validate_contract, validate_recipe = original.validate_contract, original.validate_recipe
procedural_recipe, recipe_schema, authoring_catalogue = original.procedural_recipe, original.recipe_schema, original.authoring_catalogue
contract_sha256, recipe_sha256, inventory_sha256 = original.contract_sha256, original.recipe_sha256, original.inventory_sha256


def source_hashes():
    return {**original.source_hashes(), "experiments/verified_tutor_curriculum_v2.py": hashlib.sha256((ROOT/"experiments/verified_tutor_curriculum_v2.py").read_bytes()).hexdigest()}


def _candidate_seeds(base, choice, contract, index, offset, attempt):
    def seed(purpose):
        value = [original.SCHEMA, contract["seed"], index, offset, purpose]
        if attempt: value.extend([COLLISION_POLICY, attempt])
        return int(_hash(value),16) % (2**63)
    mode = choice["realization"]
    names = base["naming_seed"] if mode == "revalue" else seed("names")
    # Independent retries retain candidate-zero values/labels; only names vary.
    values = base["value_seed"] if mode == "rename" else (seed("values") if mode == "revalue" else int(_hash([original.SCHEMA,contract["seed"],index,offset,"values"]),16) % (2**63))
    return names, values


def _fresh_pair(foundation, base, *, family, choice, contract, index, offset,
                protected, fresh_seen, work, rejections, boundary):
    """Return one whole pair without mutating either exclusion set.

    Every candidate and rejected member is counted. Finite revalue spaces may
    exhaust; there is no fallback to renaming, a different parent or replay.
    """
    for attempt in range(MAX_CANDIDATES):
        boundary()
        names, values = _candidate_seeds(base,choice,contract,index,offset,attempt)
        work["fresh_candidate_attempts"] += 1
        pair = foundation.generate_pair(family,base["seed"],depth=base["depth"],turns=base["turns"],split="train",
            naming_seed=names,value_seed=values,structure_split=base["structure_split"])
        work["fresh_candidate_completions"] += 1
        hashes=[transcript_sha256(row) for row in pair]
        rejected=[]
        for member,digest in enumerate(hashes):
            reasons=[]
            if digest in protected: reasons.append("protected")
            if digest in fresh_seen: reasons.append("repeated_fresh")
            if hashes.count(digest)>1: reasons.append("repeated_within_pair")
            if reasons: rejected.append(dict(member=member,transcript_sha256=digest,reasons=reasons))
        actual=dict(replay=False,candidate_attempt=attempt,naming_seed=names,value_seed=values)
        if not rejected: return pair,actual
        work["rejected_fresh_candidates"] += 1
        rejections.append(dict(slot=index,pair_offset=offset,family=family,realization=choice["realization"],
            **actual,rejected_members=rejected))
    raise ValueError(f"fresh realization exhausted {MAX_CANDIDATES} candidates: slot={index}, offset={offset}, family={family}, realization={choice['realization']}")


def compile_curriculum(recipe, *, admitted_parent_inventory, protected_transcripts, coverage_contract):
    """Return detached verified images plus immutable manifest identities/cost.

    Inventory: {schema, plans:{plan_sha:admitted_plan}, motifs:{id:[descriptor]}, replay:[descriptor]}.
    Descriptor: {plan_sha256,bundle_id,pair_index,pair_sha256:{family:pin}}.
    Caller owns provenance of these admitted parent pins, not teacher output.
    """
    started, cpu = time.monotonic(), time.process_time()
    work = dict(canonical_generate_attempts=0, canonical_generate_completions=0, canonical_rows_returned=0,
        parent_pairs_verified=0, materialized_pairs=0, materialized_bundles=0, explicit_typed_oracle_calls=0, explicit_english_oracle_calls=0, fresh_candidate_attempts=0, fresh_candidate_completions=0, rejected_fresh_candidates=0)
    receipt = dict(status="failed", work=work, neural_work=0, teacher_calls=0, collision_rejections=[], max_candidates_per_fresh_pair=MAX_CANDIDATES)
    layout_work = None
    try:
        contract = validate_contract(coverage_contract); recipe = validate_recipe(recipe, contract)
        inventory = json.loads(_json(admitted_parent_inventory))
        protected = set(protected_transcripts)
        if (any(not _pin(x) for x in protected) or len(protected) != contract["protected_count"]
                or _hash(sorted(protected)) != contract["protected_sha256"] or _hash(inventory) != contract["inventory_sha256"]):
            raise ValueError("caller-pinned inventory/protected identity differs")
        sources = source_hashes()
        work["max_generator_calls"] = contract["limits"]["max_generator_calls"]
        deadline = started + contract["limits"]["max_seconds"]
        def boundary():
            if time.monotonic() >= deadline: raise TimeoutError("fixed compiler allowance expired")
        from experiments import foundation_plan as planning, foundation_curriculum as foundation
        from experiments import foundation_layout_curriculum as layout, composition_curriculum as oracle
        _keys(inventory, ("schema","plans","motifs","replay"), "admitted inventory")
        if inventory["schema"] != INVENTORY_SCHEMA or type(inventory["plans"]) is not dict or type(inventory["motifs"]) is not dict:
            raise ValueError("declared admitted parent inventory required")
        for pin, plan in inventory["plans"].items():
            if not _pin(pin) or _hash(plan) != pin or "admission" not in plan: raise ValueError("unchanged admitted parent plan required")
            planning.validate_plan(plan)
        pools = {}
        for identifier, descriptors in [*inventory["motifs"].items(), ("__replay__",inventory["replay"])]:
            if identifier != "__replay__" and not _id(identifier): raise ValueError("invalid motif pool identifier")
            if type(descriptors) is not list: raise ValueError("ordered parent descriptors required")
            by_cell, seen = {}, set()
            for desc in descriptors:
                _keys(desc, ("plan_sha256","bundle_id","pair_index","pair_sha256"), "parent descriptor")
                plan = inventory["plans"].get(desc["plan_sha256"])
                if plan is None: raise ValueError("parent plan absent from pinned inventory")
                _integer(desc["bundle_id"],0,len(plan["bundles"])-1,"parent bundle")
                _integer(desc["pair_index"],0,plan["config"]["micro_batch_size"]//2-1,"parent pair")
                _keys(desc["pair_sha256"],FAMILIES,"all-family parent pins")
                if any(not _pin(pin) for pin in desc["pair_sha256"].values()) or _hash(desc) in seen:
                    raise ValueError("distinct canonical descriptors with all-family pins required")
                seen.add(_hash(desc)); slot=plan["bundles"][str(desc["bundle_id"])]
                by_cell.setdefault((slot["depth"],slot["turns"]),[]).append(desc)
            pools[identifier] = by_cell
        for slot in contract["slots"]:
            ids = ["__replay__"] if slot["replay"] else contract["chapters"][slot["chapter"]]["motifs"]
            if any(len(pools.get(i,{}).get((slot["depth"],slot["turns"]),[])) < contract["micro_batch_size"]//2 for i in ids):
                raise ValueError("unsupported motif: no full all-family parent pool for a fixed slot")
        layout_work = layout.WorkLedger(); images, provenance = [], []
        cache, fresh_seen = {}, set()
        exposures = {family:dict.fromkeys(COUNTS,0) for family in FAMILIES}
        coverage, operators, labels, replay_counts = Counter(), Counter(), Counter(), Counter()
        with _track(foundation, work, boundary):
            for index, slot in enumerate(contract["slots"]):
                boundary(); choice=recipe["chapters"][slot["chapter"]]
                pool = pools["__replay__" if slot["replay"] else choice["motif"]][slot["depth"],slot["turns"]]
                bundle=dict(schema="bic-foundation-layout-bundle-v1",bundle_id=contract["start_cursor"]+index,layout="original",families={f:[] for f in FAMILIES})
                pair_origins, seen_parents, seen_ids = [], set(), set()
                for offset in range(contract["micro_batch_size"]//2):
                    desc=pool[(index*(contract["micro_batch_size"]//2)+offset)%len(pool)]
                    key=_hash(desc); plan=inventory["plans"][desc["plan_sha256"]]
                    if key not in cache:
                        parents={}
                        for family in FAMILIES:
                            attempt=plan["admission"]["realization_attempts"].get(f"{desc['bundle_id']}/{family}/{desc['pair_index']}",0)
                            pair=planning._materialize_validated_pair(plan,desc["bundle_id"],family,desc["pair_index"],attempt)
                            if _hash(pair)!=desc["pair_sha256"][family]: raise ValueError("admitted canonical parent bytes differ")
                            parents[family]=pair;work["parent_pairs_verified"]+=1
                        if len({pair[0]["program_id"] for pair in parents.values()})!=1: raise ValueError("family-independent parent procedure required")
                        cache[key]=parents
                    pair_origins.append(dict(parent=deepcopy(desc),replay=slot["replay"],choice=None if slot["replay"] else deepcopy(choice), realizations={}))
                    for family in FAMILIES:
                        parent=cache[key][family]; base=parent[0]["recipe"]
                        if slot["replay"]:
                            pair=deepcopy(parent)
                            actual=dict(replay=True, candidate_attempt=None, naming_seed=base["naming_seed"], value_seed=base["value_seed"])
                        else:
                            pair, actual = _fresh_pair(foundation, base, family=family, choice=choice,
                                contract=contract, index=index, offset=offset, protected=protected,
                                fresh_seen=fresh_seen, work=work, rejections=receipt["collision_rejections"], boundary=boundary)
                        pair_origins[-1]["realizations"][family]=actual
                        foundation.validate_pair(pair)
                        for row in pair:
                            targets=[turn["target"] for turn in row["turns"]]
                            work["explicit_typed_oracle_calls"]+=1
                            if oracle.abstract_oracle(foundation._program(row),family)!=targets: raise ValueError("typed truth differs")
                            work["explicit_english_oracle_calls"]+=1
                            if oracle.english_oracle(row)!=targets: raise ValueError("English truth differs")
                            if [oracle.REPLIES[t] for t in targets]!=[t["reply"] for t in row["turns"]]: raise ValueError("canonical reply truth differs")
                            digest=transcript_sha256(row)
                            if digest in protected: raise ValueError("protected replay transcript" if slot["replay"] else "protected fresh transcript after admission")
                            if not any(t==2 for t in targets) or not any(t in (0,1) for t in targets): raise ValueError("known and unknown questions required")
                            labels.update((family,str(t)) for t in targets)
                            operators.update((family,event["op"]) for event in foundation._program(row))
                        if not slot["replay"]: fresh_seen.update(transcript_sha256(row) for row in pair)
                        adapted=layout.materialize_pair(pair,layout="original",seed=0,work=layout_work)
                        layout.validate_pair(adapted,work=layout_work)
                        if any(row["family"]!=family or row["split"]!="train" or row["structure_partition"]!="train"
                               or row["depth"]!=slot["depth"] or len(row["turns"])!=slot["turns"] for row in adapted):
                            raise ValueError("actual materialized family/depth/length/admission differs")
                        parent_pin=adapted[0]["recipe"]["base_pair_sha256"]; ids=adapted[0]["recipe"]["base_ids"]
                        if parent_pin in seen_parents or any(x in seen_ids for x in ids): raise ValueError("duplicate parent inside bundle")
                        seen_parents.add(parent_pin);seen_ids.update(ids)
                        bundle["families"][family].extend(adapted);work["materialized_pairs"]+=1
                evidence=_evidence(bundle)
                work["materialized_bundles"]+=1
                for family in FAMILIES:
                    coverage[family,slot["depth"],slot["turns"]]+=1
                    for key,n in evidence["families"][family]["exposures"].items():exposures[family][key]+=n
                replay_counts[slot["depth"],slot["turns"]]+=int(slot["replay"])
                for name,limit in (("episodes","max_episodes"),("observation_bytes","max_observation_bytes"),("reply_target_bytes","max_reply_target_bytes")):
                    if sum(c[name] for c in exposures.values())>contract["limits"][limit]: raise ValueError("actual exposure limit exceeded: "+name)
                images.append(dict(bundle=bundle,expected_evidence=evidence,expected_evidence_sha256=_hash(evidence)))
                provenance.append(dict(global_cursor=bundle["bundle_id"],slot=deepcopy(slot),parents=pair_origins))
        if source_hashes()!=sources: raise ValueError("compiler or canonical source changed")
        boundary()
        receipt.update(status="completed",layout_work=layout_work.report())
        manifest=dict(schema=SCHEMA,collision_policy=COLLISION_POLICY,recipe=recipe,recipe_sha256=_hash(recipe),contract_sha256=_hash(contract),
            inventory_sha256=_hash(inventory),protected_sha256=contract["protected_sha256"],source_sha256=sources,
            images_sha256=_hash(images),provenance_sha256=_hash(provenance),exposures=exposures,
            coverage={f"{f}/d{d}/t{t}":n for (f,d,t),n in sorted(coverage.items())},
            operator_counts={f"{f}/{op}":n for (f,op),n in sorted(operators.items())},
            target_counts={f"{f}/{target}":n for (f,target),n in sorted(labels.items())},
            replay_bundles=sum(replay_counts.values()),fresh_bundles=len(images)-sum(replay_counts.values()),
            unique_transcripts=len({transcript_sha256(row) for image in images for rows in image["bundle"]["families"].values() for row in rows}),
            scope="Verified examples from caller-admitted canonical parents; no explanation prose, learned self-direction or benefit claim.")
        return dict(manifest=manifest,images=images,provenance=provenance,receipt=receipt)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        error.compilation_report=receipt
        raise
    finally:
        if layout_work is not None: receipt["layout_work"]=layout_work.report()
        receipt.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
