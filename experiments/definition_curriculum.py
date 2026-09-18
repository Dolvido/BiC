"""Finite English definitions, applications and revisions across three worlds.

This is a new curriculum, not an old foundation-layout canonical row. Its
admission produces the existing immutable bundle/evidence carrier. Only visible
English and zero observations enter policy inputs; rules and labels are not
inference arguments. The typed and English interpreters use separate state and
rule representations. Composition meanings are never admitted to training.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

from experiments.cognitive_curriculum import ALIASES, COLORS, REPLIES, DENY, ALLOW, ASK, ACK

VERSION = "bic-english-definition-curriculum-v1"
FAMILIES = ("color", "count", "switch")
PANELS = ("binding", "revision", "composition")
SPLITS = ("train", "dev", "audit")
BUNDLE_SCHEMA = "bic-foundation-layout-bundle-v1"
NONCES = {"train": ("plin", "trave", "glorp", "smeb", "yorn", "quib"),
          "dev": ("snarp", "blen", "froop", "zindle", "mave", "tulp"),
          "audit": ("vindle", "cren", "julp", "spave", "drel", "noof")}
MEANINGS = {"copy": ("copy",), "copy_advance_destination": ("copy", "advance_destination"),
            "advance_source_copy": ("advance_source", "copy"),
            "copy_advance_source": ("copy", "advance_source")}
TRAIN_MEANINGS = ("copy", "copy_advance_destination")
COMPOSITION_MEANINGS = ("advance_source_copy", "copy_advance_source")
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
            ("experiments/definition_curriculum.py", "experiments/cognitive_curriculum.py")}


def work_report(): return deepcopy(_WORK)


def _options(family, seed, split, panel, turns):
    if (family not in FAMILIES or type(seed) is not int or not 0 <= seed < 2**63
            or split not in SPLITS or panel not in PANELS or type(turns) is not int or turns != 12):
        raise ValueError("declared family,63-bit seed,split,panel and exactly12turns required")
    if split == "train" and panel == "composition":
        raise ValueError("composition meanings are withheld from training")
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
            if (word in ALIASES or not 1<=len(phrases)<=2
                    or any(p not in ("copy source to destination","advance source once","advance destination once") for p in phrases)
                    or phrases.count("copy source to destination")!=1):
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
    # Roles, nonce and meaning order share one structural draw across families.
    rng=random.Random(int(_hash([VERSION,seed,split,panel,"structure"]),16))
    source,first,second,unknown=rng.sample(ALIASES,4)
    word=rng.choice(NONCES[split]);reverse=rng.randrange(2)
    raw=random.Random(int(_hash([VERSION,seed,split,panel,"values"]),16))
    index=raw.randrange(90)
    count_base=30+raw.randrange(25)
    values=((COLORS[index%4],COLORS[(index+2)%4],COLORS[(index+3)%4]) if family=="color" else
            (bool(index%2),not bool(index%2),bool(index%2)) if family=="switch" else (count_base,count_base+3,count_base+7))
    a,b,c=values; source_probe=rng.randrange(2)
    meanings=COMPOSITION_MEANINGS if panel=="composition" else TRAIN_MEANINGS
    if reverse: meanings=meanings[::-1]
    def event(op,**values): return dict(op=op,**values)
    def query(name,value,kind): return event("query",name=name,value=value,kind=kind)
    result=[]
    for meaning in meanings:
        define=event("define",word=word,meaning="copy" if panel=="revision" else meaning)
        prefix=[define,event("set",name=source,value=a),event("apply",word=word,source=source,destination=first)]
        if panel=="revision":
            middle=[query(first,a,"destination"),query(unknown,a,"unknown"),event("revise",word=word,meaning=meaning)]
        else:
            middle=[query(source,_advance(family,a) if source_probe else a,"source"),
                    query(first,a,"destination"),query(unknown,a,"unknown")]
        tail=[event("set",name=source,value=b),event("apply",word=word,source=source,destination=second),
              query(source,_advance(family,b) if source_probe else b,"source"),query(first,a,"prior_value"),
              event("set",name=source,value=c),query(second,b,"destination")]
        result.append(prefix+middle+tail)
    return result,word


def _observations():
    return {"visual":[[0.]*32],"auditory":[[0.]*4],"body":[[0.]*4],"feedback":[[0.]*2],"tokens":[0]}


def _generate_pair(family,seed,split,panel,turns):
    _options(family,seed,split,panel,turns)
    programs,word=_programs(family,seed,split,panel)
    parent=_hash(dict(version=VERSION,family=family,seed=seed,split=split,panel=panel,programs=programs))
    ids=[_hash([VERSION,parent,v]) for v in (0,1)]
    recipe=dict(layout="original",seed=seed,split=split,panel=panel,turns=turns,word=word,
        base_pair_sha256=parent,base_ids=ids)
    rows=[]
    for variant,program in enumerate(programs):
        actions,labels,causes=_typed_run(program,family)
        texts=[_render(e,family) for e in program]
        parsed_family,english_actions,english_labels,english_causes=_english_run(texts)
        if (parsed_family!=family or actions!=english_actions or labels!=english_labels or causes!=english_causes
                or sum(len(t.encode())+2 for t in texts)>1024):
            raise ValueError("independent interpreters or context budget disagree")
        queries=[dict(query_id=f"query-{i:02d}",turn_index=i,kind=e["kind"],rule_version=causes[i])
                 for i,e in enumerate(program) if e["op"]=="query"]
        rows.append(dict(version=VERSION,id=ids[variant],family=family,split=split,structure_partition=split,
            panel=panel,variant=variant,counterfactual_group=parent,depth=2,recipe=deepcopy(recipe),
            semantic_partition="heldout_composition" if panel=="composition" else "binding_revision",
            depth_semantics="Maximum primitive instructions in one defined operation, not legacy foundation ancestry depth.",
            anchor=dict(query_id="query-11",turn_index=11),queries=queries,program=deepcopy(program),
            turns=[dict(text=text,observations=_observations(),target=target,reply=REPLIES[target])
                   for text,target in zip(texts,actions)]))
        _WORK["materialized_rows"] += 1
    if {rows[0]["turns"][11]["target"],rows[1]["turns"][11]["target"]}!={DENY,ALLOW}:
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
    result.update(depth=2,turns=12,common_parents_sha256=_hash(common),
        bundle_rows_sha256=_hash([[family,bundle["families"][family]] for family in FAMILIES]))
    return result
