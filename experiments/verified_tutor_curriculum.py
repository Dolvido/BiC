"""Compile a pinned whole-curriculum chapter choice into verified English lessons.

Author-facing validation is stdlib-only. Compilation has no lesson-file, teacher,
model, tensor or optimizer API. The caller authenticates parent admission and
persists returned images once; a failed compilation returns no partial lessons.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bic-verified-tutor-curriculum-v1"
RECIPE_SCHEMA = "bic-verified-tutor-curriculum-recipe-v1"
CONTRACT_SCHEMA = "bic-verified-tutor-curriculum-contract-v1"
INVENTORY_SCHEMA = "bic-verified-tutor-parent-inventory-v1"
REALIZATIONS = ("independent", "rename", "revalue")
FAMILIES = ("color", "count", "switch")
CELLS = tuple((d, t) for d in range(6) for t in (8, 10, 12))
COUNTS = ("episodes", "turns", "observation_tokens", "observation_bytes", "reply_target_tokens", "reply_target_bytes")
_ACTIVE = False


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value): return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
def contract_sha256(value): return _hash(value)
def recipe_sha256(value): return _hash(value)
def inventory_sha256(value): return _hash(value)


def transcript_sha256(row):
    raw = json.dumps([turn["text"] for turn in row["turns"]], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _keys(value, names, label):
    if type(value) is not dict or set(value) != set(names): raise ValueError(label + " fields differ")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high: raise ValueError(label + " integer bound differs")


def _pin(value): return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
def _id(value): return type(value) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,47}", value) is not None


def source_hashes():
    names = ("experiments/verified_tutor_curriculum.py", "experiments/foundation_plan.py",
        "experiments/foundation_curriculum.py", "experiments/composition_curriculum.py",
        "experiments/cognitive_curriculum.py", "experiments/foundation_layout_curriculum.py",
        "experiments/foundation_layout_training.py")
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def validate_contract(contract):
    _keys(contract, ("schema", "inventory_sha256", "protected_sha256", "protected_count", "seed",
        "start_cursor", "micro_batch_size", "layout", "chapters", "slots", "rehearsal_floor_by_depth",
        "replay_floor_per_cell", "limits"), "coverage contract")
    value = json.loads(_json(contract))
    if (value["schema"] != CONTRACT_SCHEMA or not _pin(value["inventory_sha256"])
            or not _pin(value["protected_sha256"]) or value["layout"] != "original"):
        raise ValueError("explicit inventory/protection and fixed original layout required")
    for name in ("seed", "start_cursor", "protected_count"):
        _integer(value[name], 0, 2**63-1, name)
    _integer(value["micro_batch_size"], 2, 32, "microbatch")
    if value["micro_batch_size"] % 2: raise ValueError("complete paired microbatch required")
    chapters = value["chapters"]
    if type(chapters) is not list or not 1 <= len(chapters) <= 6: raise ValueError("one to six fixed chapters required")
    for chapter in chapters:
        _keys(chapter, ("motifs", "realizations", "default"), "chapter menu")
        for key in ("motifs", "realizations"):
            choices = chapter[key]
            if type(choices) is not list or not 1 <= len(choices) <= 8 or len(set(choices)) != len(choices):
                raise ValueError("bounded distinct menu choices required")
        if any(not _id(x) for x in chapter["motifs"]) or any(x not in REALIZATIONS for x in chapter["realizations"]):
            raise ValueError("unsupported motif identifier or realization")
        _keys(chapter["default"], ("motif", "realization"), "procedural chapter")
        if chapter["default"]["motif"] not in chapter["motifs"] or chapter["default"]["realization"] not in chapter["realizations"]:
            raise ValueError("procedural chapter must use the same menu")
    slots = value["slots"]
    if type(slots) is not list or not 18 <= len(slots) <= 108: raise ValueError("bounded fixed all-cell schedule required")
    coverage, replay = Counter(), Counter()
    for slot in slots:
        _keys(slot, ("chapter", "depth", "turns", "replay"), "fixed slot")
        _integer(slot["chapter"], 0, len(chapters)-1, "chapter index")
        _integer(slot["depth"], 0, 5, "depth")
        if type(slot["turns"]) is not int or slot["turns"] not in (8, 10, 12) or type(slot["replay"]) is not bool:
            raise ValueError("canonical length and explicit replay required")
        coverage[slot["chapter"], slot["depth"], slot["turns"]] += 1
        replay[slot["depth"], slot["turns"]] += int(slot["replay"])
    if (set(coverage) != {(c,d,t) for c in range(len(chapters)) for d,t in CELLS}
            or len(set(coverage.values())) != 1):
        raise ValueError("each chapter must have identical full18-cell coverage")
    _keys(value["rehearsal_floor_by_depth"], map(str, range(5)), "earlier-depth rehearsal floors")
    for depth, floor in value["rehearsal_floor_by_depth"].items():
        _integer(floor, 0, 108, "rehearsal floor")
        if sum(slot["depth"] == int(depth) for slot in slots) < floor: raise ValueError("earlier-depth rehearsal floor not met")
    _integer(value["replay_floor_per_cell"], 0, 6, "replay floor")
    if any(replay[cell] < value["replay_floor_per_cell"] for cell in CELLS): raise ValueError("literal replay floor not met")
    limits = value["limits"]
    _keys(limits, ("max_episodes", "max_observation_bytes", "max_reply_target_bytes", "max_generator_calls", "max_seconds"), "limits")
    for name in ("max_episodes", "max_observation_bytes", "max_reply_target_bytes", "max_generator_calls"):
        _integer(limits[name], 1, 2**40, name)
    seconds = limits["max_seconds"]
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("at-most600-second compilation allowance required")
    if len(slots)*3*value["micro_batch_size"] > limits["max_episodes"] or value["start_cursor"] + len(slots) >= 2**63:
        raise ValueError("declared update/episode limit exceeded")
    return value


def validate_recipe(recipe, contract):
    contract = validate_contract(contract)
    _keys(recipe, ("schema", "chapters"), "teacher recipe")
    if recipe["schema"] != RECIPE_SCHEMA or type(recipe["chapters"]) is not list or len(recipe["chapters"]) != len(contract["chapters"]):
        raise ValueError("exact fixed chapter recipe required")
    for choice, menu in zip(recipe["chapters"], contract["chapters"]):
        _keys(choice, ("motif", "realization"), "chapter choice")
        if choice["motif"] not in menu["motifs"] or choice["realization"] not in menu["realizations"]:
            raise ValueError("chapter choice is outside caller-frozen menu")
    return json.loads(_json(recipe))


def procedural_recipe(contract):
    contract = validate_contract(contract)
    return dict(schema=RECIPE_SCHEMA, chapters=[deepcopy(c["default"]) for c in contract["chapters"]])


def recipe_schema(contract):
    contract = validate_contract(contract)
    choices = [dict(type="object", additionalProperties=False, required=["motif","realization"],
        properties=dict(motif=dict(type="string",enum=c["motifs"]), realization=dict(type="string",enum=c["realizations"])))
        for c in contract["chapters"]]
    return dict(type="object", additionalProperties=False, required=["schema","chapters"], properties=dict(
        schema=dict(type="string", const=RECIPE_SCHEMA), chapters=dict(type="array", minItems=len(choices), maxItems=len(choices), prefixItems=choices, items=False)))


def authoring_catalogue(contract):
    value = validate_contract(contract)
    return dict(chapters=value["chapters"], realizations=dict(independent="change aliases and values",
        rename="change aliases; retain parent values", revalue="retain aliases; change values"),
        fixed=dict(families=list(FAMILIES), depths=list(range(6)),
        turns=[8,10,12], updates=len(value["slots"]), episodes=len(value["slots"])*3*value["micro_batch_size"],
        replay_bundles=sum(slot["replay"] for slot in value["slots"]),
        policy="Shared chapter choices within fixed slots; no family allocations, labels, code or prose."))


@contextmanager
def _track(foundation, work, boundary):
    global _ACTIVE
    if _ACTIVE: raise RuntimeError("one synchronous compiler owner required")
    original = foundation.generate_pair
    def generated(*args, **kwargs):
        boundary()
        if work["canonical_generate_attempts"] >= work["max_generator_calls"]: raise ValueError("canonical generation budget exceeded")
        work["canonical_generate_attempts"] += 1
        result = original(*args, **kwargs)
        work["canonical_generate_completions"] += 1
        work["canonical_rows_returned"] += len(result)
        return result
    _ACTIVE = True; foundation.generate_pair = generated
    try: yield
    finally: foundation.generate_pair = original; _ACTIVE = False


def _evidence(bundle):
    # Exact evidence encoding consumed by PreparedLayoutOwner. Pair truth and
    # uniqueness are validated before this arithmetic; no model dependency.
    result = dict(bundle_id=bundle["bundle_id"], layout="original", families={})
    common = []
    for family in FAMILIES:
        rows = bundle["families"][family]
        parents = [[family, row["recipe"]["base_pair_sha256"], row["recipe"]["base_ids"]] for row in rows[::2]]
        common.extend(parents)
        counts = dict.fromkeys(COUNTS, 0)
        for row in rows:
            turns = len(row["turns"])
            observation = sum(len(t["text"].encode("utf-8")) for t in row["turns"])
            reply = sum(len(t["reply"].encode("utf-8")) for t in row["turns"])
            for key, n in zip(COUNTS, (1,turns,observation+2*turns,observation,reply+turns,reply)): counts[key] += n
        result["families"][family] = dict(rows_sha256=_hash(rows), recipes_sha256=_hash([r["recipe"] for r in rows[::2]]),
            common_parents_sha256=_hash(parents), exposures=counts)
    row = bundle["families"][FAMILIES[0]][0]
    result.update(depth=row["depth"], turns=len(row["turns"]), common_parents_sha256=_hash(common),
        bundle_rows_sha256=_hash([[family,bundle["families"][family]] for family in FAMILIES]))
    return result


def compile_curriculum(recipe, *, admitted_parent_inventory, protected_transcripts, coverage_contract):
    """Return detached verified images plus immutable manifest identities/cost.

    Inventory: {schema, plans:{plan_sha:admitted_plan}, motifs:{id:[descriptor]}, replay:[descriptor]}.
    Descriptor: {plan_sha256,bundle_id,pair_index,pair_sha256:{family:pin}}.
    Caller owns provenance of these admitted parent pins, not teacher output.
    """
    started, cpu = time.monotonic(), time.process_time()
    work = dict(canonical_generate_attempts=0, canonical_generate_completions=0, canonical_rows_returned=0,
        parent_pairs_verified=0, materialized_pairs=0, materialized_bundles=0, explicit_typed_oracle_calls=0, explicit_english_oracle_calls=0)
    receipt = dict(status="failed", work=work, neural_work=0, teacher_calls=0)
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
                    pair_origins.append(dict(parent=deepcopy(desc),replay=slot["replay"],choice=None if slot["replay"] else deepcopy(choice)))
                    for family in FAMILIES:
                        parent=cache[key][family]; base=parent[0]["recipe"]
                        if slot["replay"]: pair=deepcopy(parent)
                        else:
                            seed=lambda purpose:int(_hash([SCHEMA,contract["seed"],index,offset,purpose]),16)%(2**63)
                            names=base["naming_seed"] if choice["realization"]=="revalue" else seed("names")
                            values=base["value_seed"] if choice["realization"]=="rename" else seed("values")
                            pair=foundation.generate_pair(family,base["seed"],depth=slot["depth"],turns=slot["turns"],split="train",
                                naming_seed=names,value_seed=values,structure_split=base["structure_split"])
                        foundation.validate_pair(pair)
                        for row in pair:
                            targets=[turn["target"] for turn in row["turns"]]
                            work["explicit_typed_oracle_calls"]+=1
                            if oracle.abstract_oracle(foundation._program(row),family)!=targets: raise ValueError("typed truth differs")
                            work["explicit_english_oracle_calls"]+=1
                            if oracle.english_oracle(row)!=targets: raise ValueError("English truth differs")
                            if [oracle.REPLIES[t] for t in targets]!=[t["reply"] for t in row["turns"]]: raise ValueError("canonical reply truth differs")
                            digest=transcript_sha256(row)
                            if digest in protected or (not slot["replay"] and digest in fresh_seen): raise ValueError("protected or repeated fresh transcript")
                            if not slot["replay"]: fresh_seen.add(digest)
                            if not any(t==2 for t in targets) or not any(t in (0,1) for t in targets): raise ValueError("known and unknown questions required")
                            labels.update((family,str(t)) for t in targets)
                            operators.update((family,event["op"]) for event in foundation._program(row))
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
        manifest=dict(schema=SCHEMA,recipe=recipe,recipe_sha256=_hash(recipe),contract_sha256=_hash(contract),
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
