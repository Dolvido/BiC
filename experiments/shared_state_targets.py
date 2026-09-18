"""Training-only causal English state targets and a shared 107-class loss.

Every prefix labels all twelve fixed aliases. State is derived only from the
visible English prefix; recipes, latent programs and future names are ignored.
The two existing interpreters independently check actions/ACKs after packing.
This module must not be imported by the policy/model inference path.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.cognitive_curriculum import ALIASES, COLORS, ACK, ASK, REPLIES
from experiments.composition_curriculum import parse_sentence, english_oracle, abstract_oracle

SCHEMA = "bic-shared-state-targets-v1"
CLASSES = 107
FAMILIES = ("color", "count", "switch")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("experiments/shared_state_targets.py", "experiments/cognitive_curriculum.py",
             "experiments/composition_curriculum.py")
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def _label(family, value):
    if value is None:
        return 0
    if family == "color" and type(value) is str and value in COLORS:
        return 1+COLORS.index(value)
    if family == "count" and type(value) is int and 0 <= value <= 99:
        return 5+value
    if family == "switch" and type(value) is bool:
        return 106 if value else 105
    raise ValueError("state leaves the declared typed domain")


def pack_state_targets(rows):
    """Return detached CPU long[B,T,12] for one uniform training microbatch.

Labels: 0 unknown; 1..4 COLORS; 5..104 counts 0..99; 105 off; 106 on.
All rows must belong to one family, with equal nonzero turn counts. No oracle
result becomes policy input; this function produces auxiliary supervision only.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("nonempty training rows required")
    if any(type(row) is not dict or row.get("family") not in FAMILIES
           or row.get("split") != "train" or row.get("structure_partition") != "train"
           or type(row.get("turns")) is not list or not row["turns"] for row in rows):
        raise ValueError("typed train-split and train-structure rows required")
    if len({row["family"] for row in rows}) != 1 or len({len(row["turns"]) for row in rows}) != 1:
        raise ValueError("one family and uniform actual turn count required")
    packed = []
    for row in rows:
        family, state, prefixes, program, answers = row["family"], {}, [], [], []
        for turn in row["turns"]:
            if type(turn) is not dict:
                raise ValueError("visible turn mappings required")
            parsed_family, event = parse_sentence(turn.get("text"))
            if parsed_family != family:
                raise ValueError("English mixes typed families")
            program.append(event)
            op, name, answer = event["op"], event["name"], ACK
            if op == "set":
                state[name] = event["value"]
            elif op == "copy":
                if event["source"] in state:
                    state[name] = state[event["source"]]
                else:
                    state.pop(name, None)
            elif op == "advance":
                if name in state:
                    value = state[name]
                    if family == "color": value = COLORS[(COLORS.index(value)+1) % len(COLORS)]
                    elif family == "count": value += event["amount"]
                    else: value = not value
                    _label(family, value)  # Reject overflow, including 99+1 and 0-1.
                    state[name] = value
            elif op == "query":
                # A literal in a question does not assert or overwrite a fact.
                answer = ASK if name not in state else int(state[name] == event["value"])
            else:
                raise ValueError("unsupported state operation")
            if (type(turn.get("target")) is not int or turn["target"] != answer
                    or turn.get("reply") != REPLIES[answer]):
                raise ValueError("supplied query/ACK supervision disagrees with causal English")
            answers.append(answer)
            prefixes.append([_label(family, state.get(alias)) for alias in ALIASES])
        if english_oracle(row) != answers or abstract_oracle(program, family) != answers:
            raise ValueError("independent English and typed oracles disagree")
        packed.append(prefixes)
    return torch.tensor(packed, dtype=torch.long, device="cpu")


def state_loss(state_logits, targets):
    """Equal known/unknown group means per turn, then the B*T mean.

A turn containing only one group uses that group's mean without halving it.
Targets may remain on CPU; their detached copy is moved to the logits device.
    """
    if (type(state_logits) is not torch.Tensor or not state_logits.is_floating_point()
            or state_logits.ndim != 4 or state_logits.shape[-2:] != (len(ALIASES), CLASSES)
            or state_logits.shape[0] < 1 or state_logits.shape[1] < 1
            or type(targets) is not torch.Tensor or targets.dtype != torch.long
            or tuple(targets.shape) != tuple(state_logits.shape[:-1])
            or bool((targets < 0).any()) or bool((targets >= CLASSES).any())):
        raise ValueError("finite logits[B,T,12,107] and long class targets[B,T,12] required")
    if not bool(torch.isfinite(state_logits).all()):
        raise ValueError("nonfinite state logits")
    labels = targets.detach().to(device=state_logits.device)
    losses = F.cross_entropy(state_logits.reshape(-1, CLASSES), labels.reshape(-1), reduction="none").reshape(labels.shape)
    known, unknown = labels.ne(0), labels.eq(0)
    nk, nu = known.sum(-1), unknown.sum(-1)
    known_mean = (losses*known).sum(-1)/nk.clamp_min(1)
    unknown_mean = (losses*unknown).sum(-1)/nu.clamp_min(1)
    groups = nk.gt(0).to(losses.dtype)+nu.gt(0).to(losses.dtype)
    return ((known_mean+unknown_mean)/groups).mean()


auxiliary_loss = state_loss
