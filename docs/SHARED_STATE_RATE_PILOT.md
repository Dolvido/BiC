# Broad acquisition at two learning rates

Prospective optimization diagnostic, selected after the completed causal entity
retrieval comparison. The added retrieval/fusion did not pass its native screen;
both final learners solved zero of 360 development pairs and zero of 54 actual
training-fit pairs. Their conditional dense state values remained near simple
value priors. This does not establish that another architecture is required.
Earlier tiny-buffer fitting at a smaller rate showed trainability but weak
transfer. The missing comparison is that smaller rate on broad training.

## Fixed comparison

Use the unchanged SharedStateStudent and SharedStateKernel in both arms, with
the original action/reply/observation objective plus 0.3 prefix-state loss.
Initialize both with seed 852604001 and require exact equality of all weights.
Keep width 192, four layers/heads, feedforward 768, 1024 positions, 12 turns,
128 input bytes and 32 output bytes. Use AdamW defaults except learning rate,
gradient clipping 1 and the existing strict local FP32 runtime on the RTX 5080.
The fast arm uses 0.003; the slow arm uses 0.0003. No schedule, warmup, architecture
change or extra auxiliary term is introduced. Both have identical parameter counts.

Reuse the entire authenticated 648-bundle original-layout training cache through
the entity-retrieval data overlay, manifest SHA256
`7c0d1f208f59a183cf5c052c991c842c2358107a3c40c84f4fc36ba42fa63612`.
Each update contains 32 episodes from each of color/count/switch. Both arms see
the same ordered lessons, with first-arm execution alternating. Compute causal
state targets once per bundle and share them; targets are never forward inputs.
Complete 648 updates and 62,208 episode exposures per arm, 1,296 updates and
124,416 exposures total. This is one pass over the existing cache, which itself
contains three graded curriculum cycles. Do not separately rehearse the 108-row
fit sample. Do not repeat the cache or extend training in response to scores.

Evaluate native actions and free replies from BOS, with no tutor, at updates
0/216/432/648. Reuse the 720 development, 108 actual training-fit and 720 retention
rows from the authenticated overlay. All three banks have previously been
observed by the project: development is untrained for these learners but is NOT
a new pristine validation bank. No new lesson generation or preparation run is
needed. Preserve all 24 bank endpoints and 12,384 raw evaluation episode records.
Save all six nonzero full model and AdamW checkpoints. No earlier checkpoint is
selected for the final comparison.

Use a 3,600-second main execution allowance including setup, scoring and saving.
Check deadlines between operations; an in-flight operation may finish later.
Preserve any failure or partial work. No automatic retry, continuation, promotion
or LLM call occurs. Training and scoring remain entirely local.

## Interpretation and checks

Reuse shared_state_screen.compare unchanged, with fast as baseline and slow as
candidate, at the fixed final endpoint. Its legacy `fresh_joint_gain` key refers
here to the reused development bank, not pristine validation. Require at least
5 percentage points of overall joint-pair improvement, strict improvement in
each family, improved training-fit joint score, no greater than 5-point loss in
any known/unknown action/reply family slice, and no greater than 5-point joint
retention loss. Apply exact-count thresholds. A pass supports further replication
and fresh validation only. A failure says this rate did not help enough at this
exposure budget; it does not rule out lower rates, longer training or other
optimization settings generally. This is a single-seed diagnostic.

Recount every saved prediction using the existing pure scorer. Report separate
known/unknown and nonanchor factual accuracy, action/reply agreement, unsupported
abstention, all endpoint trajectories, actual work, full execution and per-arm
training/preparation/evaluation time, and peak allocated/reserved GPU memory.
State target preparation is shared. Report the reused data preparation cost as
historical, not newly incurred by this comparison. Lower token loss alone is not
evidence of factual comprehension or general intellect.

Declare a separate final-checkpoint state diagnostic before this run starts.
Its only data are the same 108 actual training-fit rows; use the existing
conditional value, knownness, rank, Brier and ordinary argmax metrics without
family masks. It cannot override the native screen. Pin its separate protocol
and source before invoking it, after the main native report.

Both model and kernel are already validated, including the .0003 setting in the
prior fixed-buffer diagnostic. The new runner changes orchestration and the
rate assigned to each arm only. Review the adapted runner and report, validate
syntax and contract without constructing learners, then authenticate the frozen
source/data closure. Runtime checks enforce equal initialization, exact paired
evidence, full work counts and unchanged strict execution. Do not repeat prior
model tests or count this experiment as independent confirmation of those tests.

Keep the rejected retrieval design and all prior evidence unchanged. This run
does not establish broad English, learned self-direction, persistent memory or
beneficial external tutoring. Those remain open parts of the home-learning goal.
