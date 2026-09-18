"""Learn byte-level English across continuing, independent dialogue episodes.

English enters the trainable comprehension network, then the thirteen-region
brain.  Caller-owned BrainState carries activity between turns.  Labels, reply
targets, episode kind, and curriculum metadata never enter the policy inputs.
There is no pretrained language model, symbolic planner, or inference oracle.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .language import ByteCodec, LanguageBrain, LanguageConfig
from .learning_student import (_INITIALIZATION_LOCK, _check_finite_tree,
                               _cpu_copy, _finite_number, _integer)
from .model import Brain, BrainConfig


NUM_TURNS = 6
ACK = 3
REPLIES = ("No.", "Yes.", "I need more information.", "Understood.")
_WIDTHS = {"visual": 32, "auditory": 4, "body": 4, "feedback": 2}
SESSION_SCHEMA = "bic-byte-dialogue-session-v1"


def build_dialogue_student(seed: int, device="cpu") -> LanguageBrain:
    """Initialize a small, untrained byte-English brain without changing RNGs."""
    _integer("seed", seed)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    brain_config = BrainConfig(hidden_size=32, visual_dim=32, auditory_dim=4,
                               body_dim=4, vocab_size=1, num_actions=4, memory_slots=8)
    language_config = LanguageConfig(hidden_size=64, embedding_size=24,
                                     max_input_bytes=128, max_output_bytes=32)
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = LanguageBrain(Brain(brain_config), language_config)
    return model.to(device)


def _observations(rows: Sequence[Mapping], device) -> dict:
    result = {name: [] for name in (*_WIDTHS, "tokens")}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(result):
            raise ValueError("observations require exactly visual, auditory, body, feedback and tokens")
        for name, width in _WIDTHS.items():
            try:
                value = torch.as_tensor(row[name], dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("observations must be finite numeric arrays") from error
            if value.shape != (1, width) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must have finite shape (1, {width})")
            result[name].append(value)
        if (not isinstance(row["tokens"], (list, tuple)) or len(row["tokens"]) != 1
                or type(row["tokens"][0]) is not int or row["tokens"][0] != 0):
            raise ValueError("dialogue token input must be the constant [0]")
        result["tokens"].append(torch.zeros(1, dtype=torch.long))
    return {name: torch.stack(values).to(device) for name, values in result.items()}


def encode_dialogues(model: LanguageBrain, episodes: Sequence[Mapping], *,
                     blank_text: bool = False) -> list[dict]:
    """Encode all turns once. Metadata and supervision stay outside observations.

    Evaluation uses only this structural boundary, never a curriculum oracle.
    Training authenticates canonical episodes separately before calling here.
    """
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    device = next(model.parameters()).device
    for episode in episodes:
        if (not isinstance(episode, Mapping) or not isinstance(episode.get("turns"), list)
                or len(episode["turns"]) != NUM_TURNS):
            raise ValueError("each dialogue must contain exactly six turns")
        for turn in episode["turns"]:
            if (not isinstance(turn, Mapping) or type(turn.get("target")) is not int
                    or not 0 <= turn["target"] < 4):
                raise ValueError("each turn requires a target in [0, 3]")
            if not isinstance(turn.get("text"), str) or turn.get("reply") != REPLIES[turn["target"]]:
                raise ValueError("each turn requires English text and its canonical reply")
            if not isinstance(turn.get("kind"), str):
                raise ValueError("each turn requires a kind for evaluation metadata")
    batches = []
    for position in range(NUM_TURNS):
        turns = [episode["turns"][position] for episode in episodes]
        ids, lengths = model.prepare_inputs(["" if blank_text else turn["text"] for turn in turns])
        replies, _ = model.codec.batch_encode([turn["reply"] for turn in turns],
                                              device=device, max_bytes=model.config.max_output_bytes)
        batches.append({
            "observations": _observations([turn["observations"] for turn in turns], device),
            "text_ids": ids, "text_lengths": lengths,
            "decoder_input_ids": replies[:, :-1], "reply_targets": replies[:, 1:],
            "targets": torch.tensor([turn["target"] for turn in turns], dtype=torch.long, device=device),
            "kinds": [turn["kind"] for turn in turns],
        })
    return batches


def _select(batch: dict, indices: torch.Tensor) -> dict:
    return {key: ({name: tensor[indices] for name, tensor in value.items()} if key == "observations"
                 else value[indices]) for key, value in batch.items() if key != "kinds"}


def _run_turn(model: LanguageBrain, batch: dict, state, *, decoder=None):
    return model.forward_with_state(batch["observations"], batch["text_ids"],
                                    batch["text_lengths"],
                                    batch["decoder_input_ids"] if decoder is None else decoder,
                                    state=state)


def _generate_reply_tokens(model: LanguageBrain, context: torch.Tensor) -> torch.Tensor:
    """Greedy next-byte inference; no answer mapping and no teacher-forcing."""
    batch = context.shape[0]
    token = torch.full((batch, 1), ByteCodec.BOS, dtype=torch.long, device=context.device)
    rows = [token]
    hidden = model.inferior_frontal.initial_state(context).tanh().unsqueeze(0)
    done = torch.zeros(batch, dtype=torch.bool, device=context.device)
    for _ in range(model.config.max_output_bytes + 1):
        embedded = torch.cat((model.inferior_frontal.embedding(token), context[:, None, :]), dim=-1)
        activity, hidden = model.inferior_frontal.recurrent(embedded, hidden)
        token = model.inferior_frontal.readout(activity).argmax(dim=-1)
        token = torch.where(done[:, None], torch.full_like(token, ByteCodec.PAD), token)
        rows.append(token)
        done = done | token[:, 0].eq(ByteCodec.EOS)
        if bool(done.all()):
            break
    return torch.cat(rows, dim=1)


def _exact_replies(generated: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    expected = torch.cat((torch.full((targets.shape[0], 1), ByteCodec.BOS,
                                     dtype=torch.long, device=targets.device), targets), dim=1)
    width = max(generated.shape[1], expected.shape[1])
    return (F.pad(generated, (0, width - generated.shape[1]))
            == F.pad(expected, (0, width - expected.shape[1]))).all(dim=1)


def _action_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    total = targets.numel()
    if total == 0:
        return {"accuracy": None, "correct": 0, "total": 0, "loss": None}
    if not torch.isfinite(logits).all():
        raise ValueError("dialogue student produced nonfinite logits")
    correct = int((logits.argmax(dim=-1) == targets).sum().item())
    return {"accuracy": correct / total, "correct": correct, "total": total,
            "loss": float(F.cross_entropy(logits, targets).item())}


def evaluate_dialogues(model: LanguageBrain, episodes: Sequence[Mapping], *,
                       reset_each_turn: bool = False, blank_text: bool = False,
                       score_replies: bool = True) -> dict:
    """Measure action and free-generated reply accuracy without updating weights.

    Each row carries its own BrainState across the six turns. Both ablations
    preserve labels and observations: one removes memory, the other English.
    Reply cross entropy is explicitly teacher-forced; reply_exact_accuracy is
    greedy inference from BOS without any supplied response bytes. Candidate
    selection may set score_replies=False to skip production metrics and use
    only BOS as the decoder input; final audits should score the replies.
    """
    batches = encode_dialogues(model, episodes, blank_text=blank_text)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            logits, labels, kinds, reply_correct, reply_losses = [], [], [], [], []
            state = None
            for batch in batches:
                decoder = None if score_replies else torch.full(
                    (len(episodes), 1), ByteCodec.BOS, dtype=torch.long,
                    device=batch["text_ids"].device)
                output, state = _run_turn(model, batch, None if reset_each_turn else state, decoder=decoder)
                logits.append(output["logits"][:, -1])
                labels.append(batch["targets"])
                kinds.extend(batch["kinds"])
                if score_replies:
                    language_logits = output["language_logits"]
                    reply_loss = F.cross_entropy(language_logits.flatten(0, 1), batch["reply_targets"].flatten(),
                                                 ignore_index=ByteCodec.PAD)
                    if not torch.isfinite(reply_loss):
                        raise ValueError("dialogue student produced nonfinite reply loss")
                    reply_losses.append(reply_loss)
                    reply_correct.append(_exact_replies(_generate_reply_tokens(model, output["production_context"]),
                                                         batch["reply_targets"]))
            logits = torch.cat(logits)
            labels = torch.cat(labels)
            reply_correct = torch.cat(reply_correct) if score_replies else None
            query = labels.ne(ACK)
            metrics = _action_metrics(logits, labels)
            query_metrics = _action_metrics(logits[query], labels[query])
            metrics.update({"query_accuracy": query_metrics["accuracy"],
                            "query_correct": query_metrics["correct"],
                            "query_total": query_metrics["total"],
                            "query_loss": query_metrics["loss"],
                            "reply_exact_accuracy": float(reply_correct.float().mean().item()) if score_replies else None,
                            "query_reply_exact_accuracy": float(reply_correct[query].float().mean().item()) if score_replies and bool(query.any()) else None,
                            "teacher_forced_reply_loss": float(torch.stack(reply_losses).mean().item()) if score_replies else None,
                            "reset_each_turn": reset_each_turn, "blank_text": blank_text,
                            "free_running_replies": score_replies, "teacher_used_for_policy": False})
            metrics["per_target"] = {str(target): _action_metrics(logits[labels == target], labels[labels == target])
                                     for target in range(4)}
            metrics["per_kind"] = {}
            for kind in sorted(set(kinds)):
                mask = torch.tensor([value == kind for value in kinds], device=labels.device) & query
                if bool(mask.any()):
                    metrics["per_kind"][kind] = _action_metrics(logits[mask], labels[mask])
            # Paired episodes share questions and observations while reversing
            # a preceding English rule. Both opposite answers must be correct.
            groups = {}
            for index, episode in enumerate(episodes):
                if isinstance(episode.get("counterfactual_group"), str):
                    groups.setdefault(episode["counterfactual_group"], []).append(index)
            predictions = logits.argmax(dim=-1).reshape(NUM_TURNS, len(episodes))
            target_matrix = labels.reshape(NUM_TURNS, len(episodes))
            paired_total = paired_correct = paired_flips = 0
            for members in groups.values():
                if len(members) != 2:
                    continue
                left, right = members
                eligible = (target_matrix[:, left].ne(ACK) & target_matrix[:, right].ne(ACK)
                            & target_matrix[:, left].ne(target_matrix[:, right]))
                paired_total += int(eligible.sum().item())
                paired_correct += int((eligible & predictions[:, left].eq(target_matrix[:, left])
                                       & predictions[:, right].eq(target_matrix[:, right])).sum().item())
                paired_flips += int((eligible & predictions[:, left].ne(predictions[:, right])).sum().item())
            metrics.update(counterfactual_query_pairs=paired_total,
                           counterfactual_accuracy=paired_correct / paired_total if paired_total else None,
                           counterfactual_prediction_flip_rate=paired_flips / paired_total if paired_total else None)
            return metrics
    finally:
        for module, training in modes:
            module.training = training


def train_dialogue_candidate(initial_state: Mapping, episodes: Sequence[Mapping], *,
                             seed: int, steps: int, batch_size: int, learning_rate: float = .003,
                             optimizer_state: Mapping | None = None, deadline: float | None = None,
                             device="cpu", sampler_state: torch.Tensor | None = None) -> dict:
    """Train isolated weights with six-turn BPTT and resumable AdamW moments.

    Whole independent dialogues are sampled in parallel. Recurrent activity
    resets only between sampled episodes, never between their training turns.
    Action loss is mean query CE plus 0.25 mean acknowledgement CE; production
    contributes 0.1 mean next-byte CE. A whole update finishes atomically if it
    crosses the wall-clock deadline.
    """
    from .dialogue_curriculum import validate_dialogue

    started = time.monotonic()
    _integer("steps", steps)
    _integer("batch_size", batch_size, 1)
    _finite_number("learning_rate", learning_rate)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if deadline is not None:
        _finite_number("deadline", deadline)
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("split") != "train":
            raise ValueError("training accepts only train dialogues; held-out leakage rejected")
        validate_dialogue(episode)
    model = build_dialogue_student(seed, device)
    _check_finite_tree(initial_state, "initial_state")
    model.load_state_dict(initial_state, strict=True)
    batches = encode_dialogues(model, episodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        _check_finite_tree(optimizer_state, "optimizer_state")
        optimizer.load_state_dict(_cpu_copy(optimizer_state))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if sampler_state is not None:
        generator.set_state(sampler_state.detach().cpu().clone())
    updates = 0
    loss_value = action_loss_value = reply_loss_value = None
    model.train()
    for _ in range(steps):
        if deadline is not None and time.monotonic() >= deadline:
            break
        indices = torch.randint(len(episodes), (batch_size,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        action_logits, action_targets, reply_losses = [], [], []
        state = None
        for full_batch in batches:
            batch = _select(full_batch, indices)
            output, state = _run_turn(model, batch, state)
            action_logits.append(output["logits"][:, -1])
            action_targets.append(batch["targets"])
            reply_losses.append(F.cross_entropy(output["language_logits"].flatten(0, 1),
                                                batch["reply_targets"].flatten(), ignore_index=ByteCodec.PAD))
        action_logits = torch.cat(action_logits)
        action_targets = torch.cat(action_targets)
        query = action_targets.ne(ACK)
        action_loss = action_logits.sum() * 0
        if bool(query.any()):
            action_loss = action_loss + F.cross_entropy(action_logits[query], action_targets[query])
        if bool((~query).any()):
            action_loss = action_loss + .25 * F.cross_entropy(action_logits[~query], action_targets[~query])
        reply_loss = torch.stack(reply_losses).mean()
        loss = action_loss + .1 * reply_loss
        if not torch.isfinite(loss):
            raise ValueError("dialogue training produced nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        loss_value = float(loss.detach().item())
        action_loss_value = float(action_loss.detach().item())
        reply_loss_value = float(reply_loss.detach().item())
    _check_finite_tree(model.state_dict(), "trained_state")
    _check_finite_tree(optimizer.state_dict(), "trained_optimizer_state")
    return {"state_dict": _cpu_copy(model.state_dict()), "optimizer_state": _cpu_copy(optimizer.state_dict()),
            "sampler_state": generator.get_state().clone(), "updates": updates,
            "episodes_seen": updates * batch_size, "turns_seen": updates * batch_size * NUM_TURNS,
            "loss": loss_value, "action_loss": action_loss_value, "reply_loss": reply_loss_value,
            "training_seconds": time.monotonic() - started,
            "deadline_reached": deadline is not None and time.monotonic() >= deadline}


def checkpoint_digest(model: LanguageBrain) -> str:
    """Bind resumable activity to the exact learned parameter bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(data.shape), str(data.dtype))).encode("ascii"))
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


class DialogueSession:
    """Explicit continuing inference state with checkpoint-bound persistence."""

    def __init__(self, model: LanguageBrain):
        if not isinstance(model, LanguageBrain):
            raise ValueError("model must be a LanguageBrain")
        self.model = model
        self.state = None
        self.turns = 0
        self.checkpoint_sha256 = checkpoint_digest(model)

    def _check_weights(self):
        if checkpoint_digest(self.model) != self.checkpoint_sha256:
            raise ValueError("session weights changed; start a new session for the new checkpoint")

    def step(self, observation: Mapping, text: str, *, generate_reply: bool = True) -> dict:
        self._check_weights()
        device = next(self.model.parameters()).device
        observations = _observations([observation], device)
        ids, lengths = self.model.prepare_inputs([text])
        decoder = torch.full((1, 1), ByteCodec.BOS, dtype=torch.long, device=device)
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode():
                output, state = self.model.forward_with_state(observations, ids, lengths, decoder, state=self.state)
                logits = output["logits"][0, -1]
                if not torch.isfinite(logits).all():
                    raise ValueError("session produced nonfinite logits")
                result = {"action": int(logits.argmax().item()),
                          "probabilities": logits.softmax(dim=-1).cpu().tolist(),
                          "checkpoint_sha256": self.checkpoint_sha256, "turn": self.turns + 1}
                if generate_reply:
                    tokens = _generate_reply_tokens(self.model, output["production_context"])[0]
                    try:
                        reply = self.model.codec.decode(tokens)
                        valid_utf8 = True
                    except UnicodeDecodeError:
                        reply = self.model.codec.decode(tokens, errors="replace")
                        valid_utf8 = False
                    result.update({"reply": reply, "reply_ids": tokens.cpu().tolist(),
                                   "reply_valid_utf8": valid_utf8,
                                   "reply_complete": bool(tokens.eq(ByteCodec.EOS).any())})
                self.state = state.detached()
                self.turns += 1
                return result
        finally:
            for module, training in modes:
                module.training = training

    def state_dict(self) -> dict:
        self._check_weights()
        state = self.model.brain.initial_state(1) if self.state is None else self.state
        return {"schema": SESSION_SCHEMA, "checkpoint_sha256": self.checkpoint_sha256,
                "turns": self.turns, "brain_state": self.model.brain.state_to_dict(state)}

    def load_state_dict(self, payload: Mapping) -> None:
        self._check_weights()
        if (not isinstance(payload, Mapping)
                or set(payload) != {"schema", "checkpoint_sha256", "turns", "brain_state"}
                or payload["schema"] != SESSION_SCHEMA
                or payload["checkpoint_sha256"] != self.checkpoint_sha256):
            raise ValueError("session schema or checkpoint binding mismatch")
        _integer("turns", payload["turns"])
        state = self.model.brain.state_from_dict(payload["brain_state"])
        if state.previous_prefrontal.shape[0] != 1:
            raise ValueError("a dialogue session must contain exactly one row")
        self.state = state
        self.turns = payload["turns"]

    def save(self, path: str | Path) -> None:
        torch.save({"schema": SESSION_SCHEMA, "model_state": _cpu_copy(self.model.state_dict()),
                    "session_state": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "DialogueSession":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (not isinstance(payload, Mapping) or set(payload) != {"schema", "model_state", "session_state"}
                or payload["schema"] != SESSION_SCHEMA):
            raise ValueError("invalid dialogue checkpoint schema")
        _check_finite_tree(payload["model_state"], "model_state")
        model = build_dialogue_student(0, device)
        model.load_state_dict(payload["model_state"], strict=True)
        session = cls(model)
        session.load_state_dict(payload["session_state"])
        return session
