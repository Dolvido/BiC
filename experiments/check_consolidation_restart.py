"""Repeat the fixed restart cases with visibly erased source-memory artifacts.

The original sealed report is preserved. This supplemental check addresses source
files remaining available between execution sessions; it never updates weights.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from brain_in_computer.associative import render_glyph
from brain_in_computer.consolidation import ConsolidatedMemory
from brain_in_computer.regional_memory import load_regional_checkpoint
from brain_in_computer.training import atomic_json
from experiments.consolidate_memory import NAMES, sealed_specs, sha, score_data, teach


@torch.no_grad()
def parity(output, protocol):
    """Verify cached observations equal public inference, without updating weights."""
    torch.set_num_threads(1)
    agent = load_regional_checkpoint(protocol["arguments"]["regional"])
    rows = []
    for session in protocol["sessions"]:
        specs = sealed_specs(session)
        chosen = [next(s for s in specs if s["stage"] == stage and s["category"] == category)
                  for stage in ("old", "new") for category in ("valid", "unknown_label", "absent", "ambiguous", "boundary")]
        pixels = torch.stack([torch.stack([render_glyph(p, v) for p, v in zip(s["objects"], s["object_seeds"])]) for s in chosen])
        embeddings = agent.encoder(pixels.flatten(0, 1)).reshape(len(chosen), 4, -1)
        folder = output / f"seed-{session['seed']}"
        for arm in protocol["sealed_controls"]:
            if arm == "episodic":
                memory = teach(agent.encoder, NAMES, session["taught_patterns"], session["teaching_seed"])
            else:
                memory = ConsolidatedMemory.load(folder / ("with_replay.pt" if arm == "erased" else f"{arm}.pt"), agent.encoder)
                if arm == "erased":
                    for parameter in memory.recall.parameters():
                        parameter.zero_()
            data = score_data(memory, chosen, embeddings)
            direct = [memory.evidence(s["name"], crop) for s, crop in zip(chosen, pixels)]
            maximum = max(float((actual["scores"] - expected).abs().max()) for actual, expected in zip(direct, data["scores"]))
            row = {"seed": session["seed"], "condition": arm, "public_evidence_cases": len(chosen),
                   "max_score_difference": maximum, "scores_match_within_1e-6": maximum <= 1e-6,
                   "public_known_bits_match": all(float(d["known"]) == float(k) for d, k in zip(direct, data["known"].flatten())),
                   "all_original_known_bits_equal_actual_membership": all(s["known"] == (s["name"] in memory.labels) for s in specs),
                   "membership_cases_checked": len(specs)}
            if not all(row[k] for k in ("scores_match_within_1e-6", "public_known_bits_match", "all_original_known_bits_equal_actual_membership")):
                raise AssertionError(row)
            rows.append(row)
    report = {"protocol_sha256": sha(output / "protocol.json"), "sealed_report_unchanged_sha256": sha(output / "final_evaluation.json"),
              "current_runner_sha256": sha(Path(__file__).with_name("consolidate_memory.py")), "conditions": rows,
              "revision": "After the sealed run, cached known bits were changed from generated case flags to actual membership. Every historical case flag matches real membership. Direct evidence parity also checked each category and age group. Checkpoint and dependency hash checks added for future runs; no weights or sealed results changed."}
    atomic_json(output / "evidence_parity.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/consolidation-v06")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    if args.parity_only:
        parity(output, protocol)
        return
    regional = Path(protocol["arguments"]["regional"]).resolve()
    runner = Path(__file__).with_name("consolidate_memory.py").resolve()
    rows = []
    for session in protocol["sessions"]:
        folder = output / f"seed-{session['seed']}"
        source = folder / "temporary-source-memory.json"
        source.write_text('{"schema":"bic-erased-source-memory","entries":{}}\n')
        before = sha(source)
        specs = sealed_specs(session)
        chosen = [next(s for s in specs if s["name"] == name and s["category"] == "valid" and s["relation"] == "find") for name in NAMES]
        checkpoint = folder / "with_replay.pt"
        checkpoint_before = sha(checkpoint)
        with tempfile.TemporaryDirectory(prefix="bic-erased-bank-restart-") as directory:
            inputs, answers = Path(directory) / "inputs.pt", Path(directory) / "outputs.json"
            torch.save({"instructions": [s["prompt"] for s in chosen],
                        "pixels": torch.stack([torch.stack([render_glyph(p, v) for p, v in zip(s["objects"], s["object_seeds"])]) for s in chosen])}, inputs)
            subprocess.run([sys.executable, str(runner), "worker", "--regional", str(regional),
                            "--student", str(checkpoint), "--inputs", str(inputs), "--worker-output", str(answers)],
                           cwd=directory, check=True, timeout=90, capture_output=True, text=True)
            fresh = json.loads(answers.read_text())
        rows.append({"seed": session["seed"], "parent_pid": os.getpid(), "worker_pid": fresh["pid"],
                     "distinct_process": fresh["pid"] != os.getpid(), "episodes": len(chosen),
                     "correct_actions": sum(a["action"] == s["target"] for a, s in zip(fresh["rows"], chosen)),
                     "source_exemplars_erased": json.loads(source.read_text()) == {"schema": "bic-erased-source-memory", "entries": {}},
                     "source_stayed_erased": sha(source) == before,
                     "checkpoint_unchanged": sha(checkpoint) == checkpoint_before,
                     "checkpoint_sha256": checkpoint_before,
                     "episodic_construction_lookup_and_teaching_forbidden": fresh["episodic_access_forbidden"]})
    atomic_json(output / "restart_verification.json", {"protocol_sha256": sha(output / "protocol.json"),
                "original_sealed_report_sha256": sha(output / "final_evaluation.json"), "sessions": rows,
                "reason": "Training recorded successful file deletion, but files were present in later execution sessions. Their contents are now explicitly erased. Original report is preserved, and the same fixed 32 cases per seed were repeated without retraining or case selection."})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
