# Next acquisition study: exact restart and broader rehearsal

Selected after the completed [broad rate comparison](SHARED_STATE_RATE_RESULTS.md).
This is an implementation design, not a frozen launch or a checkpoint promotion.
Both tested rates failed the native screen. At 648 updates, the slower learner
nevertheless improved conditional state-value decoding in all three subjects,
with substantially better color decoding and better value ranks. Its nonanchor
known actions improved from the fast arm's 188/874 to 285/874. These observations
support testing acquisition over more practice before changing architecture.
They do not predict further improvement: slow known development actions declined
from 718 at update 432 to 638 at 648, and final joint pairs remained zero.

## Intended comparison

Resume both predetermined final 648-update SharedStateStudent checkpoints from
`runs/shared-state-rate-pilot-local/attempt-001`. Preserve each complete AdamW
state, original rate (0.003 or 0.0003), objective plus 0.3 state loss, strict FP32
runtime, configuration, evidence and cumulative work. No rate is adopted as a
winner and neither arm restarts from fresh weights. The original completed study
remains closed and unchanged; this requires a separately frozen new experiment.

The proposed additional work is two complete passes over the same authenticated
648-bundle broad cache, in the same order in each branch. Both learners receive
1,296 new updates and 124,416 repeated episode exposures, reaching lifetime 1,944
updates and 186,624 exposures apiece. There are 62,208 distinct cached training
rows, not 186,624 new lessons. Repeating the complete cache tests continued broad
acquisition; the 108-row fit diagnostic is not separately rehearsed.

Keep the immutable archive bytes and original canonical parents/row identities.
The continuation scheduler must bind an original archive SHA, source bundle ID,
pass index and new monotonic consumed bundle ID. Any in-memory ID remapping must
be independently verified against the original rows/evidence and admitted by the
existing preparation boundary. Static inspection found that only the bundle and
expected evidence's top-level `bundle_id` must change: the prepared consumer checks
that ID against the kernel cursor, independently of its exact nested row/parent
hashes. Bind source index as global ID modulo 648 and pass as integer division by
648, with new global IDs restricted to 648 through 1943. Keep replay provenance
outside the strict bundle envelope. Do not call the old generation helpers, which
reject cursor values beyond their original 648-step schedule.

Prove one authenticated cached bundle at consumed IDs 0/648/1296 gives identical
packed inputs and supervision while retaining distinct valid consumption chains.
Check wrong replay mapping, altered nested rows/evidence and wrong consumer cursor
are rejected before a model forward. This is CPU packing/identity work, separate
from the optimizer restoration proof. It must not pretend replayed episodes are
newly generated evidence. Neither adapter is implemented by this design.

Prepare a new development bank with recorded-study transcript exclusions. Score
it first at the exact restored 648 endpoint before any added update, then at fixed
lifetime 1296 and 1944 endpoints. Also retain the prior development bank, actual
training-fit sample and retention bank as explicitly reused measures. Report both
between-arm differences and change from each arm's restored starting point;
aggregate unknown-answer gains cannot hide factual regressions. Preserve every
raw result and both nonzero new full model/AdamW checkpoints per arm. A fixed
final-only training-state diagnostic should again separate conditional values,
knownness and full argmax. No held-out state targets enter training.

Freeze the actual new data seed, complete work counts, run deadline, endpoint
rules and any success screen in a new protocol before starting. A positive
single-seed result would still require replication and fresh capability tests;
a negative result would bound these two rates and this exposure schedule only.
No automatic extension, checkpoint selection or promotion follows this design.

## Shared integration prerequisite

The [verified tutor integration design](VERIFIED_TUTOR_CURRICULUM_DESIGN.md)
identifies the missing SharedStateContinuation adapter. The older tutor loop's
LayoutContinuation reconstructs a different trainer. Implement an explicit loader
for the current snapshot schema, with exact complete weights, tied parameters,
AdamW moments/groups, evidence and accounting preservation. Verify an uninterrupted
next update against a restored next update before using it for acquisition or
tutoring. Source and checkpoint authentication alone do not demonstrate numerical
continuation equivalence.

This same restart adapter will let later verified tutor-authored chapter recipes
continue the learner across processes. Teacher transport and chapter compilation
remain separate from the acquisition comparison. The next acquisition study makes
no LLM calls; richer tutoring must receive its own matched procedural control and
tutor-withdrawal comparison. Broad English and useful self-directed learning remain
open requirements, rather than conclusions from these restricted experiments.
