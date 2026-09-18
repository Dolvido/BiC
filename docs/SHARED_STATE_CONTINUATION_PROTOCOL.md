# Shared acquisition through exact continuation

Prospective local study following the completed broad rate comparison and its
selected continuation design. Both original arms failed the native screen. The
slower arm improved conditional state values and known actions, with a declining
late action trajectory. This study tests whether further broad practice produces
usable gains; it does not assume more training will help or adopt either rate.

## Origin and intervention

Use only both predetermined final 648-update checkpoints from
`runs/shared-state-rate-pilot-local/attempt-001`. Authenticate the completed launch
SHA256 `bcfdfb8f5b369d4591988e7b7db26321256cd6a9db263003979b3d10619fd3f0`
and summary SHA256 `e268e983015ef56eb10b14d864e82566e2a102c0de0e156b3d5d03f314548931`.
Preserve each exact SharedStateStudent, full AdamW state, original rate (fast
0.003, slow 0.0003), source recipe, cumulative evidence and work. Preserve the
original action/reply/observation objective plus 0.3 causal state loss, clipping
1, width 192, four layers/heads, feedforward 768 and all byte/context limits.
Use the same strict local FP32 runtime on the RTX 5080. No fresh initialization,
architecture modification, optimizer reset, warmup or rate schedule is allowed.

The assigned between-arm difference remains their original learning-rate history.
Each receives two additional full passes over the same 648-bundle admitted cache
in the same order. Alternate first-arm execution by global bundle-ID parity.
Continue at global IDs 648 through 1943: source index is ID modulo 648; replay
pass is integer division by 648. Each source bundle has 32 episodes in each of
three families. This adds 1,296 updates and 124,416 episode exposures per arm,
2,592 updates and 248,832 exposures in total. Each learner ends at lifetime
1,944 updates and 186,624 exposures on 62,208 distinct cached training rows.

Authenticate every source byte image before weights-only deserialization. Replay
changes only copied bundle/evidence top-level consumed IDs. Keep nested rows,
canonical recipes/parent IDs and labels intact. Bind original archive/manifest,
source index, pass and consumed ID in a separate provenance record. Compute state
targets once per shared consumed bundle; these remain training labels, never
model-forward inputs. The 108-row fit sample receives no separate rehearsal.

## Data, endpoints and decisions

The fresh 720-episode development bank uses seed 852702001 and excludes the
verified recorded-study union of 1,703,160 exact transcripts. It remains in the
same finite grammar and vocabulary; exclusion does not imply novel semantics or
broad English. Keep the preceding entity-development bank as `previous_dev`
(720 rows), and reuse actual training-fit (108) and retention (720). New data
preparation generates no training bundles. Its own declaration and work ledger
record all delegated archive loads and canonical construction work.

Score all four banks at lifetime 648 immediately after exact restore and before
any added update, then at fixed lifetime 1296 and 1944. The reused banks' restored
648 metrics must equal the original study's 648 metrics exactly before training.
Preserve 24 bank endpoints and all 13,608 raw native evaluation episode records.
Policy evaluation uses native actions and free replies from BOS, with no tutor,
state targets or auxiliary-decoder answers. Save complete continuation snapshots
at 1296 and 1944 for each arm, four saved continuation checkpoints total. Exact
restoration also materializes a kernel snapshot for its equality check; that
work is included in the restoration receipt. Select no earlier endpoint.

At 1944, apply the existing shared_state_screen.compare unchanged to fast versus
slow using new development, fit and retention. Also apply the identical screen
within each arm from restored 648 to 1944. The comparative screen requires at
least 5 percentage points of joint development gain, strict joint gain in every
family, better joint training-fit score, no greater than 5-point loss in any
development known/unknown action/reply family slice, and no greater than 5-point
joint retention loss. Within-arm screens answer continued-acquisition questions;
the between-arm screen compares the rate histories. Report them separately,
including failures. Check every corresponding known/unknown family slice on the
reused previous development bank against its own restored baseline with a 5-point
nonregression margin. No one screen substitutes for another. Passing a screen
supports replication and further validation, not automatic promotion.

Report all native trajectories, factual and nonanchor answers, unsupported
abstention, agreement, retention, changes from each restored baseline and final
between-arm differences. A single initialization, repeated finite lessons and
previous development-guided choices limit inference. Lower auxiliary/token loss
alone does not establish useful native behavior or general intellect.

After the primary pure recount, a separate predeclared final-only state diagnostic
may inspect both predetermined 1944 checkpoints on the same 108 actual training
rows. It must use the existing metrics, no family masks, no held-out state labels,
zero learning and exact work accounting. It cannot select checkpoints or override
the primary screens. Pin its source and protocol before invocation.

## Correctness and cost

Before launch, verify the new restart wrapper on a small CPU case and one bounded
production GPU continuation case. Compare complete weights, full AdamW, losses,
evidence and counters for uninterrupted versus restored next updates. Check wrong
identities, source changes and malformed moments fail without returning a learner.
Verify the replay adapter independently using one cached archive at consumed IDs
0/648/1296: identical packed tensors and unchanged nested evidence, distinct valid
consumption chains, and rejection of altered mapping/evidence/rows or cursor.
Record every proof invocation and physical work, including any failure.

The freeze step authenticates prior launch/summary and deserializes the two pinned
origin checkpoint images on CPU to bind their complete expected metadata. It
constructs no models and decodes no curriculum archives; record both metadata
loads and its enclosing cost. Main restoration decodes each checkpoint once again
through the checked continuation wrapper. It must recover the original lifetime
accounting and keep new restore/step/snapshot work separate. Replayed exposures
are new physical work, not unique lessons. Report inherited versus newly incurred
work and costs explicitly, including replay reads, target preparation, packing,
training, evaluation, checkpoint publication and peak allocated/reserved memory.

The main execution allowance is 3,600 seconds, including restoration, setup,
training, scoring and saving. Check deadlines between operations; an in-flight
operation may finish later. Use an exclusive output directory and preserve
partial work on failure. There is no automatic retry, score-dependent extension,
checkpoint promotion or LLM call. The larger local home-learning objective stays
active; useful external tutoring and broad independent learning remain unproven.
