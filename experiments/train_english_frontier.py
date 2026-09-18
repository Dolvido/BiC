"""Prospective, local-only English learning diagnosis with resumable branches.

This is an experiment, not a champion promotion. Parsers create training
targets only; every evaluated action still comes from bytes and BrainState.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from brain_in_computer.dialogue_curriculum import FOCI, generate_dialogues, validate_dialogue
from brain_in_computer.dialogue_student import (
    ACK, ByteCodec, _run_turn, _select, build_dialogue_student, checkpoint_digest,
    encode_dialogues, evaluate_dialogues, train_dialogue_candidate,
)
from brain_in_computer.learning_student import _cpu_copy


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def training_bank(seed=120_000, count_per_focus=1024):
    return [episode for index, focus in enumerate(FOCI)
            for episode in generate_dialogues(seed + index * 10_000, count_per_focus, "train", focus)]


def train_chunk(payload, episodes, *, seed, steps, batch_size, device, arm, lr=.003):
    if arm == "baseline":
        return train_dialogue_candidate(payload["state_dict"], episodes, seed=seed, steps=steps,
            batch_size=batch_size, device=device, learning_rate=lr,
            optimizer_state=payload.get("optimizer_state"), sampler_state=payload.get("sampler_state"))
    if batch_size % 2 or len(episodes) % 2:
        raise ValueError("paired training requires an even batch and complete pairs")
    for index in range(0, len(episodes), 2):
        if episodes[index]["counterfactual_group"] != episodes[index + 1]["counterfactual_group"]:
            raise ValueError("adjacent episodes must form counterfactual pairs")
    for episode in episodes:
        if episode["split"] != "train":
            raise ValueError("training accepts canonical train dialogues only")
        validate_dialogue(episode)
    start = time.monotonic()
    model = build_dialogue_student(seed, device)
    model.load_state_dict(payload["state_dict"])
    batches = encode_dialogues(model, episodes)
    heads, targets = None, None
    if arm in ("scaffold", "text_scaffold"):
        from brain_in_computer.dialogue_scaffolding import ScaffoldHeads, causal_targets
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 1)
            heads = ScaffoldHeads(include_state=arm == "scaffold").to(device)
        if "heads_state" in payload:
            heads.load_state_dict(payload["heads_state"])
        targets = causal_targets(episodes, device=device)
    parameters = list(model.parameters()) + (list(heads.parameters()) if heads is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = lr
    generator = torch.Generator().manual_seed(seed)
    if payload.get("sampler_state") is not None:
        generator.set_state(payload["sampler_state"])
    last = {}
    model.train()
    for _ in range(steps):
        pair_indices = torch.randint(len(episodes) // 2, (batch_size // 2,), generator=generator)
        indices = (pair_indices[:, None] * 2 + torch.arange(2)[None, :]).flatten().to(device)
        selected_targets = None if targets is None else {
            key: ({name: tensor[:, indices] for name, tensor in value.items()} if key == "masks"
                  else value[:, indices]) for key, value in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        state = None
        all_logits, all_labels, reply_losses, auxiliary_losses = [], [], [], []
        for turn, full_batch in enumerate(batches):
            batch = _select(full_batch, indices)
            output, state = _run_turn(model, batch, state)
            all_logits.append(output["logits"][:, -1])
            all_labels.append(batch["targets"])
            reply_losses.append(F.cross_entropy(output["language_logits"].flatten(0, 1),
                                                batch["reply_targets"].flatten(), ignore_index=ByteCodec.PAD))
            if heads is not None:
                auxiliary_losses.append(heads.loss(output, selected_targets, turn=turn))
        logits, labels = torch.cat(all_logits), torch.cat(all_labels)
        # Equal weight per query answer class prevents ASK frequency hiding
        # failure to distinguish allowed from denied borrowing.
        query_losses = [F.cross_entropy(logits[labels == target], labels[labels == target])
                        for target in range(ACK) if bool((labels == target).any())]
        action_loss = torch.stack(query_losses).mean() + .25 * F.cross_entropy(logits[labels == ACK], labels[labels == ACK])
        reply_loss = torch.stack(reply_losses).mean()
        aux = torch.stack(auxiliary_losses).mean() if auxiliary_losses else action_loss * 0
        loss = action_loss + .1 * reply_loss + .5 * aux
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        last = {"loss": loss.item(), "action_loss": action_loss.item(),
                "reply_loss": reply_loss.item(), "auxiliary_loss": aux.item()}
    result = {"state_dict": _cpu_copy(model.state_dict()), "optimizer_state": _cpu_copy(optimizer.state_dict()),
              "sampler_state": generator.get_state(), "updates": steps,
              "episodes_seen": steps * batch_size, "training_seconds": time.monotonic() - start, **last}
    if heads is not None:
        result["heads_state"] = _cpu_copy(heads.state_dict())
    return result


def source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *sorted((root / "brain_in_computer").glob("*.py"))]
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def run(args):
    from brain_in_computer.dialogue_diagnostics import diagnostic_bank, transcript_fingerprint
    torch.set_num_threads(1)
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "bic-english-frontier-v1", "arm": args.arm, "seed": args.seed,
        "device": args.device, "batch_size": args.batch_size, "planned_updates": args.steps,
        "checkpoint_every": args.checkpoint_every, "learning_rate": args.lr,
        "train_seed": 120000, "train_per_focus": 1024, "diagnostic_seed": 180000,
        "diagnostic_per_focus": 64, "source_sha256": source_fingerprints(),
        "known_benchmark": True, "new_pristine_audit": False, "tutor": "off",
        "selection": "No automatic champion promotion. Examine paired correctness and 2x2 transfer.",
        "gates": {"query_accuracy": .8, "counterfactual_accuracy": .6, "each_focus_query": .6,
                  "allow_deny_macro": .75, "paired_advantage_over_controls": .1},
        "stall_rule": "baseline/paired stop after four checkpoints with train-support pair accuracy zero",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": str(torch.__version__)}
    protocol_path = path / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("run protocol/source changed; start a new isolated output directory")
    else:
        write_json(protocol_path, protocol)
    train = training_bank()
    diagnostics = diagnostic_bank(180000, 64)
    # Learning-support probes are disjoint episode seeds but familiar training
    # support. They are diagnostics, never evidence of unfamiliar wording.
    support = training_bank(240000, 64)
    dataset_manifest = {"train_fingerprint": transcript_fingerprint(train),
        "support_fingerprint": transcript_fingerprint(support),
        "diagnostic_fingerprints": {key: transcript_fingerprint(value) for key, value in diagnostics.items()}}
    write_json(path / "datasets.json", dataset_manifest)
    latest = path / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        payload, updates, history = saved["training"], saved["updates"], saved["history"]
    else:
        payload = {"state_dict": _cpu_copy(build_dialogue_student(args.seed).state_dict())}
        updates, history = 0, []
        save_checkpoint(path / "initial.pt", payload)
    started = time.monotonic()
    stop_reason = "planned_updates"
    while updates < args.steps:
        chunk = min(args.checkpoint_every, args.steps - updates)
        payload = train_chunk(payload, train, seed=args.seed, steps=chunk,
            batch_size=args.batch_size, device=args.device, arm=args.arm, lr=args.lr)
        updates += payload["updates"]
        model = build_dialogue_student(args.seed, args.device)
        model.load_state_dict(payload["state_dict"])
        digest = checkpoint_digest(model)
        row = {key: value for key, value in payload.items() if key not in
               ("state_dict", "optimizer_state", "sampler_state", "heads_state")}
        row.update(updates=updates, cumulative_examples=updates * args.batch_size,
                   train_support=evaluate_dialogues(model, support, score_replies=False),
                   diagnostics={key: evaluate_dialogues(model, value, score_replies=False)
                                for key, value in diagnostics.items()})
        assert digest == checkpoint_digest(model), "evaluation changed weights"
        history.append(row)
        saved = {"schema": protocol["schema"], "training": payload, "updates": updates, "history": history}
        save_checkpoint(latest, saved)
        save_checkpoint(path / f"checkpoint-{updates:06d}.pt", saved)
        write_json(path / "progress.json", {"protocol": protocol, "history": history})
        print(json.dumps({"arm": args.arm, "seed": args.seed, "updates": updates,
            "seconds": row["training_seconds"], "loss": row["loss"],
            "support_query": row["train_support"]["query_accuracy"],
            "support_pairs": row["train_support"]["counterfactual_accuracy"],
            "diagnostics": {key: [value["query_accuracy"], value["counterfactual_accuracy"]]
                            for key, value in row["diagnostics"].items()}}), flush=True)
        if args.arm in ("baseline", "paired") and len(history) >= 4 and all(
                item["train_support"]["counterfactual_accuracy"] == 0 for item in history[-4:]):
            stop_reason = "four_zero_pair_checkpoints"
            break
        if time.monotonic() - started >= args.max_seconds:
            stop_reason = "local_time_budget"
            break
    model = build_dialogue_student(args.seed, args.device)
    model.load_state_dict(payload["state_dict"])
    digest = checkpoint_digest(model)
    final = {"support": evaluate_dialogues(model, support),
             "diagnostics": {key: evaluate_dialogues(model, value) for key, value in diagnostics.items()},
             "reset_support": evaluate_dialogues(model, support, reset_each_turn=True),
             "blank_support": evaluate_dialogues(model, support, blank_text=True)}
    assert digest == checkpoint_digest(model)
    write_json(path / "report.json", {"protocol": protocol, "datasets": dataset_manifest,
        "updates": updates, "examples": updates * args.batch_size, "history": history,
        "final": final, "checkpoint_sha256": digest, "stop_reason": stop_reason,
        "training_seconds": sum(row["training_seconds"] for row in history),
        "this_invocation_seconds": time.monotonic() - started,
        "checkpoint_unchanged_by_evaluation": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("baseline", "paired", "scaffold", "text_scaffold"), required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--lr", type=float, default=.003)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.checkpoint_every, args.max_seconds, args.lr) <= 0:
        parser.error("budgets and learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
