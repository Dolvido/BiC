"""Grounded primitive basis and held-out order, with competing English rules.

This is a new curriculum, not an old foundation-layout canonical row. Its
admission produces the existing immutable bundle/evidence carrier. Only visible
English and zero observations enter policy inputs; rules and labels are not
inference arguments. The typed and English interpreters use separate state and
rule representations. Binding episodes contain two rules with shuffled definition
order; revision checks old state before a new application with swapped roles.
Query rule_version names the TESTED APPLICATION, including unchanged arguments;
it is not the previous version's entity-mutation provenance. tested_rule_word is
verification metadata only. Composition and sequence meanings never enter train.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

from experiments.cognitive_curriculum import ALIASES, COLORS, REPLIES, DENY, ALLOW, ASK, ACK

VERSION = "bic-english-definition-basis-v1"
FAMILIES = ("color", "count", "switch")
PANELS = ("binding", "revision", "composition", "sequence")
SPLITS = ("train", "dev", "audit")
BUNDLE_SCHEMA = "bic-foundation-layout-bundle-v1"
NONCES = {"train": ("bralvek", "crimlo", "dunvek", "flarpo", "gremvo", "hurnex"),
          "dev": ("jaskel", "klunvo", "marnix", "neldro", "pruvik", "quastro"),
          "audit": ("rilvex", "sarnup", "tulvek", "vondri", "weskal", "yulpro")}
MEANINGS = {"copy": ("copy",), "advance_source": ("advance_source",),
            "advance_destination": ("advance_destination",),
            "advance_source_copy": ("advance_source", "copy"),
            "copy_advance_source": ("copy", "advance_source"),
            "advance_source_copy_advance_source": ("advance_source", "copy", "advance_source"),
            "copy_advance_source_twice": ("copy", "advance_source", "advance_source")}
TRAIN_MEANINGS = ("copy", "advance_source", "advance_destination")
BASIS_PAIRS = (("copy", "advance_source"), ("advance_source", "advance_destination"),
               ("copy", "advance_destination"))
COMPOSITION_MEANINGS = ("advance_source_copy", "copy_advance_source")
SEQUENCE_MEANINGS = ("advance_source_copy_advance_source", "copy_advance_source_twice")
_PHRASES = {"copy":"copy source to destination", "advance_source":"advance source once",
            "advance_destination":"advance destination once"}
COUNTS = ("episodes", "turns", "observation_tokens", "observation_bytes", "reply_target_tokens", "reply_target_bytes")
_WORK = dict(generation_attempts=0, generated_pairs=0, materialized_rows=0, validation_reconstructions=0, pair_validations=0,
             typed_interpretations=0, english_interpretations=0, prefix_label_calls=0, bundle_validations=0)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def _hash(value): return hashlib.sha256(_json(value).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ("experiments/definition_basis_curriculum.py", "experiments/cognitive_curriculum.py")}


def work_report(): return deepcopy(_WORK)


def _options(family, seed, split, panel, turns):
    if (family not in FAMILIES or type(seed) is not int or not 0 <= seed < 2**63
            or split not in SPLITS or panel not in PANELS or type(turns) is not int or turns != 12):
        raise ValueError("declared family,63-bit seed,split,panel and exactly12turns required")
    if split == "train" and panel in ("composition","sequence"):
        raise ValueError("composition and sequence meanings are withheld from training")
    vocab = [word for words in NONCES.values() for word in words]
    if len(vocab) != len(set(vocab)) or set(vocab)&set(ALIASES):
        raise ValueError("split-disjoint nonce verbs must differ from every entity alias")


def _label(family,value):
    if value is None: return 0
    if family == "color" and type(value) is str and value in COLORS: return COLORS.index(value)+1
    if family == "count" and type(value) is int and 0 <= value <= 99: return value+5
    if family == "switch" and type(value) is bool: return 106 if value else 105
    raise ValueError("typed state leaves the declared107-class domain")


def _advance(family,value):
    if value is None: return None
    if family == "color": result=COLORS[(COLORS.index(value)+1)%4]
    elif family == "count": result=value+1
    else: result=not value
    _label(family,result)
    return result


def _typed_run(program,family):
    """Typed semantics; never parses English or consults supplied labels."""
    _WORK["typed_interpretations"] += 1
    if family not in FAMILIES: raise ValueError("typed family required")
    state,rules,versions,causes,answers,prefixes,query_versions = {},{},{},{},[],[],[]
    for event in program:
        op=event["op"]; answer=ACK; cause=None
        if op in ("define","revise"):
            word,meaning=event["word"],event["meaning"]
            if meaning not in MEANINGS or word in ALIASES or not re.fullmatch("[a-z]+",word):
                raise ValueError("typed rule outside finite definition language")
            if (op=="define" and word in rules) or (op=="revise" and word not in rules):
                raise ValueError("definition/revision lifecycle differs")
            rules[word]=MEANINGS[meaning];versions[word]=versions.get(word,-1)+1
        elif op=="set":
            if event["value"] is None: raise ValueError("setting unknown is not a visible literal")
            _label(family,event["value"])
            if event["name"] not in ALIASES: raise ValueError("unknown entity alias")
            state[event["name"]]=event["value"];causes[event["name"]]=None
        elif op=="apply":
            word,source,destination=event["word"],event["source"],event["destination"]
            if word not in rules or source not in ALIASES or destination not in ALIASES or source==destination:
                raise ValueError("defined rule and two distinct entity arguments required")
            for instruction in rules[word]:
                if instruction=="copy":
                    state[destination]=state.get(source);causes[destination]=versions[word]
                else:
                    name=source if instruction=="advance_source" else destination
                    state[name]=_advance(family,state.get(name));causes[name]=versions[word]
        elif op=="query":
            if event["name"] not in ALIASES: raise ValueError("unknown entity alias")
            if event["value"] is None: raise ValueError("query requires a visible value literal")
            _label(family,event["value"])
            value=state.get(event["name"])
            answer=ASK if value is None else int(value==event["value"])
            cause=causes.get(event["name"])
        else: raise ValueError("unknown typed instruction")
        answers.append(answer);query_versions.append(cause)
        prefixes.append([_label(family,state.get(alias)) for alias in ALIASES])
    return answers,prefixes,query_versions


_DEFINE = re.compile(r"(Define|Revise) ([a-z]+) for (color|count|switch): (.+)\.")
_APPLY = re.compile(r"Apply ([a-z]+) to (color|count|switch) from ([a-z]+) to ([a-z]+)\.")
_SET = re.compile(r"Set the (color|count|switch) of ([a-z]+) to ([a-z]+|[0-9]+)\.")
_QUERY = re.compile(r"Is the (color|count|switch) of ([a-z]+) ([a-z]+|[0-9]+)\?")


def _english_run(texts):
    """Visible-text interpreter: string values and phrase rules, no typed AST."""
    _WORK["english_interpretations"] += 1
    memory,rules,versions,causes,answers,prefixes,query_versions = {},{},{},{},[],[],[]
    domain=None
    def literal(family,value):
        good=(value in ("red","green","blue","yellow") if family=="color" else
              value in ("off","on") if family=="switch" else
              value.isdecimal() and str(int(value))==value and 0<=int(value)<=99)
        if not good: raise ValueError("English literal outside typed world")
        return value
    def advance(family,value):
        if value is None: return None
        if family=="color": result={"red":"green","green":"blue","blue":"yellow","yellow":"red"}[value]
        elif family=="switch": result={"off":"on","on":"off"}[value]
        else: result=str(int(value)+1)
        return literal(family,result)
    for text in texts:
        if type(text) is not str or not 0 < len(text.encode()) <= 128:
            raise ValueError("nonempty at-most128-byte utterance required")
        answer=ACK;cause=None
        if match:=_DEFINE.fullmatch(text):
            verb,word,family,body=match.groups()
            phrases=body.split("; ")
            if (word in ALIASES or not 1<=len(phrases)<=3
                    or any(p not in ("copy source to destination","advance source once","advance destination once") for p in phrases)
                    or phrases.count("copy source to destination")>1):
                raise ValueError("unsupported visible definition")
            if (verb=="Define" and word in rules) or (verb=="Revise" and word not in rules):
                raise ValueError("visible definition/revision lifecycle differs")
            rules[word]=phrases;versions[word]=versions.get(word,-1)+1
        elif match:=_SET.fullmatch(text):
            family,name,value=match.groups()
            if name not in ALIASES: raise ValueError("English entity outside fixed vocabulary")
            memory[name]=literal(family,value);causes[name]=None
        elif match:=_APPLY.fullmatch(text):
            word,family,source,destination=match.groups()
            if word not in rules or source not in ALIASES or destination not in ALIASES or source==destination:
                raise ValueError("English application requires defined verb and distinct entities")
            for phrase in rules[word]:
                if phrase=="copy source to destination":
                    memory[destination]=memory.get(source);causes[destination]=versions[word]
                else:
                    name=source if phrase=="advance source once" else destination
                    memory[name]=advance(family,memory.get(name));causes[name]=versions[word]
        elif match:=_QUERY.fullmatch(text):
            family,name,value=match.groups()
            if name not in ALIASES: raise ValueError("English query entity outside vocabulary")
            literal(family,value);known=memory.get(name)
            answer=ASK if known is None else ALLOW if known==value else DENY
            cause=causes.get(name)
        else: raise ValueError("unsupported definition English")
        if domain is not None and family!=domain: raise ValueError("mixed family English")
        domain=family
        labels=[]
        for alias in ALIASES:
            value=memory.get(alias)
            if value is None: labels.append(0)
            elif family=="color": labels.append({"red":1,"green":2,"blue":3,"yellow":4}[value])
            elif family=="switch": labels.append({"off":105,"on":106}[value])
            else: labels.append(5+int(value))
        answers.append(answer);prefixes.append(labels);query_versions.append(cause)
    return domain,answers,prefixes,query_versions


def prefix_labels(row):
    """Causal Python[T][12] labels derived only from this row's visible text."""
    _WORK["prefix_label_calls"] += 1
    family,_,labels,_=_english_run([turn["text"] for turn in row["turns"]])
    if family!=row["family"]: raise ValueError("row family differs from English")
    return labels


def _render(event,family):
    op=event["op"]
    if op in ("define","revise"):
        return f"{op.title()} {event['word']} for {family}: "+"; ".join(_PHRASES[x] for x in MEANINGS[event["meaning"]])+"."
    if op=="apply": return f"Apply {event['word']} to {family} from {event['source']} to {event['destination']}."
    value=("on" if event["value"] else "off") if family=="switch" else str(event["value"])
    if op=="set": return f"Set the {family} of {event['name']} to {value}."
    if op=="query": return f"Is the {family} of {event['name']} {value}?"
    raise ValueError("unsupported typed rendering")


def _programs(family,seed,split,panel):
    # Pair class is explicit seed%3; one shared plan covers all three families.
    rng=random.Random(int(_hash([VERSION,seed,split,panel,"structure"]),16))
    source,destination,unknown=rng.sample(ALIASES,3)
    word,other_word=rng.sample(NONCES[split],2)
    reverse=rng.randrange(2);definition_order=rng.randrange(2)
    raw=random.Random(int(_hash([VERSION,seed,split,panel,"values"]),16))
    index=raw.randrange(4)
    count_base=30+raw.randrange(25)
    a=COLORS[index] if family=="color" else bool(index%2) if family=="switch" else count_base
    pair_class=seed%3
    meanings=(COMPOSITION_MEANINGS if panel=="composition" else
              SEQUENCE_MEANINGS if panel=="sequence" else BASIS_PAIRS[pair_class])
    if reverse: meanings=meanings[::-1]
    def event(op,**values): return dict(op=op,**values)
    def query(name,value,kind,version=0,tested_word=word):
        return event("query",name=name,value=value,kind=kind,tested_version=version,tested_word=tested_word)
    def outcome(meaning,s,d):
        # Only chooses useful question literals. Admission truth is independently
        # recomputed by the typed executor and the visible-English interpreter.
        for instruction in MEANINGS[meaning]:
            if instruction=="copy":d=s
            elif instruction=="advance_source":s=_advance(family,s)
            else:d=_advance(family,d)
        return s,d
    def probe(values):
        value=values[rng.randrange(len(values))]
        return _advance(family,value) if rng.randrange(2) else value
    if panel=="revision":
        prior="advance_destination" if pair_class==0 else "copy"
        b=a if prior=="advance_destination" else _advance(family,a)
        old_source,old_destination=outcome(prior,a,b)
        later=[outcome(m,old_destination,old_source) for m in meanings]
        source_probe=probe([v[0] for v in later]);anchor_probe=rng.choice([v[1] for v in later])
        if later[0][1]==later[1][1]:raise ValueError("revision contrast needs an opposite destination anchor")
    else:
        # C/S requires a real copy effect. C/D requires equal initial values to
        # avoid equivalence for binary switches. S/D works under either binding.
        b=(_advance(family,a) if pair_class==0 else
           a if pair_class==2 else _advance(family,a) if rng.randrange(2) else a)
        first=[outcome(m,a,b) for m in meanings]
        other=next(m for m in TRAIN_MEANINGS if m not in meanings) if panel=="binding" else rng.choice(TRAIN_MEANINGS)
        later=[outcome(other,d,s) for s,d in first]
        first_source_probe=probe([v[0] for v in first]);anchor_probe=rng.choice([v[1] for v in first])
        if first[0][1]==first[1][1]:raise ValueError("binding contrast needs an opposite destination anchor")
        later_source_probe=probe([v[0] for v in later]);later_destination_probe=probe([v[1] for v in later])
    result=[]
    for meaning in meanings:
        if panel=="revision":
            program=[event("define",word=word,meaning=prior),event("set",name=source,value=a),
                event("set",name=destination,value=b),event("apply",word=word,source=source,destination=destination),
                query(destination,old_destination,"prior_value"),event("revise",word=word,meaning=meaning),
                query(destination,old_destination,"prior_value"),
                event("apply",word=word,source=destination,destination=source),
                query(destination,source_probe,"source",1),query(unknown,a,"unknown",None,None),
                query(source,anchor_probe,"destination",1),query(destination,_advance(family,source_probe),"source",1)]
        else:
            definitions=[event("define",word=word,meaning=meaning),event("define",word=other_word,meaning=other)]
            if definition_order:definitions.reverse()
            program=definitions+[event("set",name=source,value=a),event("set",name=destination,value=b),
                event("apply",word=word,source=source,destination=destination),
                query(source,first_source_probe,"source"),query(destination,anchor_probe,"destination"),
                event("apply",word=other_word,source=destination,destination=source),
                query(destination,later_source_probe,"source",0,other_word),
                query(source,later_destination_probe,"destination",0,other_word),
                query(unknown,a,"unknown",None,None),
                query(destination,_advance(family,later_source_probe),"source",0,other_word)]
        result.append(program)
    return result,dict(word=word,other_word=None if panel=="revision" else other_word,
        basis_pair_class=pair_class,definition_order=None if panel=="revision" else definition_order,
        meaning_pair=list(meanings),anchor_turn_index=10 if panel=="revision" else 6)


def _observations():
    return {"visual":[[0.]*32],"auditory":[[0.]*4],"body":[[0.]*4],"feedback":[[0.]*2],"tokens":[0]}


def _generate_pair(family,seed,split,panel,turns):
    _options(family,seed,split,panel,turns)
    programs,details=_programs(family,seed,split,panel)
    parent=_hash(dict(version=VERSION,family=family,seed=seed,split=split,panel=panel,programs=programs))
    ids=[_hash([VERSION,parent,v]) for v in (0,1)]
    recipe=dict(layout="original",seed=seed,split=split,panel=panel,turns=turns,**details,
        base_pair_sha256=parent,base_ids=ids)
    rows=[]
    for variant,program in enumerate(programs):
        actions,labels,causes=_typed_run(program,family)
        texts=[_render(e,family) for e in program]
        parsed_family,english_actions,english_labels,english_causes=_english_run(texts)
        if (parsed_family!=family or actions!=english_actions or labels!=english_labels or causes!=english_causes
                or sum(len(t.encode())+2 for t in texts)>1024):
            raise ValueError("independent interpreters or context budget disagree")
        queries=[dict(query_id=f"query-{i:02d}",turn_index=i,kind=e["kind"],rule_version=e["tested_version"],
                      tested_rule_word=e["tested_word"])
                 for i,e in enumerate(program) if e["op"]=="query"]
        rows.append(dict(version=VERSION,id=ids[variant],family=family,split=split,structure_partition=split,
            panel=panel,variant=variant,counterfactual_group=parent,
            depth=max(len(MEANINGS[e["meaning"]]) for e in program if e["op"] in ("define","revise")),recipe=deepcopy(recipe),
            semantic_partition=("heldout_"+panel if panel in ("composition","sequence") else "primitive_basis"),
            depth_semantics="Maximum primitive instructions in one defined operation, not legacy foundation ancestry depth.",
            query_version_semantics="rule_version and tested_rule_word identify the application being tested, including unchanged arguments; prior_value tests the earlier application.",
            anchor=dict(query_id=f"query-{details['anchor_turn_index']:02d}",turn_index=details["anchor_turn_index"]),
            queries=queries,program=deepcopy(program),
            turns=[dict(text=text,observations=_observations(),target=target,reply=REPLIES[target])
                   for text,target in zip(texts,actions)]))
        _WORK["materialized_rows"] += 1
    at=details["anchor_turn_index"]
    if {rows[0]["turns"][at]["target"],rows[1]["turns"][at]["target"]}!={DENY,ALLOW}:
        raise ValueError("definition contrast must produce opposite known anchor answers")
    return rows


def generate_pair(family,seed,*,split="train",panel="binding",turns=12):
    _WORK["generation_attempts"] += 1
    result=_generate_pair(family,seed,split,panel,turns)
    _WORK["generated_pairs"] += 1
    return result


def validate_pair(pair):
    _WORK["pair_validations"] += 1
    if type(pair) not in (list,tuple) or len(pair)!=2 or any(type(r) is not dict for r in pair):
        raise ValueError("complete adjacent definition counterfactual pair required")
    try:
        first=pair[0];recipe=first["recipe"]
        _WORK["validation_reconstructions"] += 1
        expected=_generate_pair(first["family"],recipe["seed"],recipe["split"],recipe["panel"],recipe["turns"])
        if _json(list(pair))!=_json(expected): raise ValueError("rows differ from canonical definition admission")
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError("malformed definition pair") from error
    return True


def bundle_evidence(bundle):
    """New canonical admission, existing public PreparedLayoutOwner carrier."""
    _WORK["bundle_validations"] += 1
    if (type(bundle) is not dict or set(bundle)!={"schema","bundle_id","layout","families"}
            or bundle["schema"]!=BUNDLE_SCHEMA or bundle["layout"]!="original"
            or type(bundle["bundle_id"]) is not int or not 0<=bundle["bundle_id"]<2**63
            or type(bundle["families"]) is not dict or set(bundle["families"])!=set(FAMILIES)):
        raise ValueError("exact original-layout three-family bundle carrier required")
    sizes={len(rows) for rows in bundle["families"].values() if type(rows) is list}
    if len(sizes)!=1 or any(type(rows) is not list for rows in bundle["families"].values()) or not 2<=next(iter(sizes))<=32 or next(iter(sizes))%2:
        raise ValueError("equal even2..32 complete-pair family microbatches required")
    result=dict(bundle_id=bundle["bundle_id"],layout="original",families={})
    common,seen_parents,seen_ids,dimensions=[],set(),set(),set()
    for family in FAMILIES:
        rows=bundle["families"][family];parents=[];counts=dict.fromkeys(COUNTS,0)
        for index in range(0,len(rows),2):
            pair=rows[index:index+2];validate_pair(pair)
            recipe=pair[0]["recipe"];pin=recipe["base_pair_sha256"];ids=recipe["base_ids"]
            if pin in seen_parents or set(ids)&seen_ids: raise ValueError("duplicate/crossed admitted parent")
            seen_parents.add(pin);seen_ids.update(ids)
            parents.append([family,pin,ids]);common.append([family,pin,ids])
            for row in pair:
                if row["family"]!=family or row["split"]!="train" or row["structure_partition"]!="train":
                    raise ValueError("new definition training admission required")
                dimensions.add((row["depth"],len(row["turns"]),row["panel"]))
                observation=sum(len(t["text"].encode()) for t in row["turns"])
                reply=sum(len(t["reply"].encode()) for t in row["turns"])
                for key,n in zip(COUNTS,(1,12,observation+24,observation,reply+12,reply)):counts[key]+=n
        result["families"][family]=dict(rows_sha256=_hash(rows),recipes_sha256=_hash([r["recipe"] for r in rows[::2]]),
            common_parents_sha256=_hash(parents),exposures=counts)
    if len(dimensions)!=1:raise ValueError("one shared panel/depth/length across all families required")
    result.update(depth=next(iter(dimensions))[0],turns=12,common_parents_sha256=_hash(common),
        bundle_rows_sha256=_hash([[family,bundle["families"][family]] for family in FAMILIES]))
    return result
