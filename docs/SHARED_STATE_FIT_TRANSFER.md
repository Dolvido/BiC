# Transfer check after the fixed-buffer diagnostic

This separate, inference-only check is declared after the completed four-arm fit
diagnostic and before any scores from these checkpoints on the selected bank.
It asks whether success on the practiced buffer extends to other realizations
of the same restricted English curriculum. It cannot establish general English,
new-family transfer, causal benefit of LLM teaching or checkpoint readiness.

Compare both joint-objective models at their predetermined final 432-update
checkpoints, joint_fast and joint_slow. Their source run is
runs/shared-state-fit-local/attempt-001, launch SHA256
1b8a920caf81d3c0dbc02a447adc0480505577192632003a913df147003642e1,
and completed summary SHA256
9e4de8297cf917d54d58306fb1337ea407089cfef5ea19eee1f274286b383f5c.
Restore exact full weights and the same strict FP32 runtime; do not construct an
optimizer, update weights, tune thresholds or substitute the midpoint checkpoint.
Both state-only arms are outside this comparison because their native output
heads received no direct training. This selection is by training objective,
not a ranking among development scores.

Use all 720 episodes of the existing dev bank in
runs/shared-state-data-local/attempt-001/banks.pt, SHA256
c42c117d04070cae026e6372615bcb044b493ff41f80603a268dc8aff93e4efb.
This bank was previously inspected in the older shared-state pilot. It is reused
development data, not a pristine audit or independent replication. These two
fresh fit learners trained only on the 108 fit rows, and the selected dev bank
is disjoint from their training transcripts. Verify that disjointness directly.
Shared finite grammar, vocabulary and semantic equivalents remain limitations.

Split only for execution into nine family-by-length groups of 80 episodes;
use native evaluation batch size 32, three forwards per group. This is 27 native
forwards per model, 54 total, and 1,440 evaluation episode passes. Start reply
generation from BOS only and generate freely. The auxiliary state head is never
called, and no parsed state or answers enter policy inputs. Preserve per-episode
raw actions and replies, whole paired outcomes, known/unknown performance,
nonanchor questions and per-family totals. Use the existing validated native
scorer and verify unchanged full model state. Load two final checkpoints, build
two models, make zero training/optimizer/tutor calls. Existing canonical admission
may reconstruct validation records; log that work separately from model work.

Use an exclusive attempt directory and a 180-second wall allowance including
setup, validation, inference and publication. Preserve complete or partial
receipts, source/protocol/input identities, runtime, wall/CPU time and peak GPU
allocation. Do not retry or extend automatically after a failure. Recount saved
predictions without another neural run.

Report both final checkpoints on the full bank, even if either disappoints.
No threshold in this diagnostic promotes a checkpoint. Near-perfect practice
with weak dev performance would indicate that the next learning comparison must
address acquisition across varied examples and transfer; it would not prove
that the successful fitting was purely rote or that the architecture cannot
generalize. A lower-rate advantage here remains single-seed, reused-development
evidence and requires a matched broader learning comparison.
