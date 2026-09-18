# Preserving learning across repeated curriculum cycles

Version two passed **20 focused CPU checks on the first run**, including three
actual completed 66-update cycles for each architecture. Both learners continued
through 198 lifetime updates with complete weights and AdamW state, fresh admitted
curricula, cumulative transcript exclusions and retained evaluation history.
The version-two receipt (archive reference: `../runs/foundation-cycle-repeated-validation-local/attempt-001/report.json`)
records exact sources, all physical work, timings and persisted origin artifacts.
This verifies continuity and restart behavior; it does not establish a learning
advantage or increased general intellect. No GPU, formal-study data or formal
study weights were used.

## Version two: repeated capture and bounded ancestry

The trainer, provider and completed-parent schemas explicitly advance to version
two. Version-one cycle checkpoints are rejected; there is no silent migration.
`CompletedParent.from_loop` accepts only an actual completed `FoundationLoop`
with the exact original variant provider/trainer or the current cycle
provider/trainer. Final evaluation, learner state, input evidence, ownership and
source/runtime checks still precede capture.

The token's `cycle` identifies the completed parent's ordinal. Its `base_updates`
is the parent's complete lifetime count, ready to become its successor's base;
`cycle_updates` and `inherited_updates` identify the local and prior portions.
The child exposes its own explicit `cycle`, with local `cursor`/`updates` and
separate `base_updates`/`lifetime_updates`. Checkpoint restore rejects conflicting
ordinals and counters, and every AdamW step must equal the lifetime count.
The second handoff preserved complete moments at step 132; the third completed
cycle had step 198 throughout its optimizer state.

Each parent identity contains only predecessor identity/envelope digests,
current source/index evidence and current completion identities. It does not
recursively embed earlier parent identities or tensor images. Both tested
architectures retained one learner image. From cycle two to three, token identity
size increased by two bytes, the learner image increased by 128 bytes for flat
and 192 bytes for hierarchical, and ordinary metadata changed by -289/+824 bytes.
Those small changes include counters and variable evaluation content; they do
not establish constant size for arbitrary configurations.

The cumulative exact-transcript exclusion inventory necessarily grows: these
fixtures carried 408, 804 and 1,200 digests after successive cycles. All previous
training and protected transcripts remained excluded, and every child used the
same evaluation role/name/bank identities. Fresh exact transcripts do not imply
new algorithms or unrestricted language competence.

## Compact restart and lifetime retention

The separate `foundation_parent_capsule` API exports an actual authenticated
token and returns its file-byte hash. Loading requires that exact trusted caller
pin, matching runtime/sources, and the original bank rows. The loader validates
the current learner against one CPU model/optimizer template and checks current
metadata, metrics and transcript inventories. It inherits historical execution
provenance from the known export; it does not replay past gradients or model
predictions. A hash taken from an untrusted neighboring receipt is insufficient.

All six actual parent capsules round-tripped exactly. Loaded tokens were used
to begin the next cycles. Guards confirmed that compact load constructed no
historical plan index and loaded no historical loop; wrong pins rejected before
tensor loading, and exporting over an existing path failed. The independent
saved-chain reconstruction route also produced the same identities and complete
states, but that route explicitly repeats historical setup work.

Capsule sizes in this tiny fixture were:

| Architecture | Cycle two bytes | Cycle three bytes | Increase |
| --- | ---: | ---: | ---: |
| Flat | 2,691,006 | 2,717,379 | 26,373 |
| Hierarchical | 2,689,612 | 2,717,162 | 27,550 |

Most growth was the additional 396 exclusion digests, not another ancestor
model. The [owner design](FOUNDATION_CYCLE_OWNER_DESIGN.md) details the external
pin and provenance boundaries. Persisted test capsules, exact banks and pinned
retention ledgers are under the receipt's `artifacts/flat` and
`artifacts/hierarchical` directories; they are engineering fixtures, not selected
trained candidates.

The separate [lifetime retention fold](FOUNDATION_LIFETIME_RETENTION.md) was
advanced through all three completed cycles and restored from its pinned
snapshot. It preserved best references across boundaries and used explicit local
and lifetime coordinates. A child's local loop history is still distinct from
this lifetime comparison. Neither component silently promotes weights or
relabels a deterioration as progress.

## Measured validation cost and limits

The 20-check suite took **585.143 seconds** on one CPU thread. It performed 558
actual optimizer calls across the actual cycle fixtures and additional restart,
arithmetic and failure checks. Accounting includes 559 step attempts; 3,354
drawn, 3,352 neural-attempted and 3,350 completed-microbatch episode exposures;
and 138 evaluation banks/276 scored episodes. One deliberate partial-microbatch
failure and one update discarded after failed publication are included. There
were no unknown optimizer reports.

The following intervals come from the completed 66-update cycles themselves.
Practice includes retained steps; retained steps include materialization. These
columns must not be added as if they were disjoint phases.

| Architecture / cycle | Retained steps (s) | Practice including steps (s) | Scoring (s) | Capsule reload (s) |
| --- | ---: | ---: | ---: | ---: |
| Flat / 1 | 5.124 | 7.078 | 1.281 | 0.765 |
| Flat / 2 | 5.031 | 12.094 | 5.829 | 0.782 |
| Flat / 3 | 4.968 | 12.156 | 6.282 | 0.875 |
| Hierarchical / 1 | 3.577 | 5.204 | 1.501 | 0.719 |
| Hierarchical / 2 | 3.764 | 10.812 | 5.610 | 0.937 |
| Hierarchical / 3 | 3.641 | 10.594 | 5.874 | 0.828 |

Materialization took 0.421-0.515 seconds per cycle and was already included in
retained-step time. The remainder of step time includes packing, forward and
backward passes, and optimizer work; it is not a pure gradient measurement.
For cycles two and three, roughly seven seconds of provider practice fell
outside retained steps, exceeding those tiny models' retained-step time.
Repeated validation therefore warrants profiling before claiming efficient
small-cycle execution. Capsule reload took 0.719-0.937 seconds and did not add
inference or optimizer work.

The full suite also deliberately repeats hostile restore, publication, admission
and reconstruction checks. Its wall time is not production throughput. Exact
source-hashing, construction, admission and checkpoint intervals were not all
retained separately, so no complete wall-time decomposition or GPU speedup is
claimed. Measurements were extracted from existing records without rerunning
training. The combined runtime still needs its own GPU proof before a GPU
learning deployment. The [bounded runner](FOUNDATION_CYCLE_RUNNER.md) and durable
owner have separate validation records.

## Historical version-one first-handoff validation

The following describes version one as validated at that time. Its exact sources
are preserved in the version-one archive (archive reference: `../runs/foundation-cycle-repeated-validation-local/v1-archive/manifest.json`).
Its work counts are separate from the version-two run above.

The additive cycle trainer and provider implement one explicit handoff from a
completed foundation variant loop to a fresh finite curriculum. They preserve
the learned weights, all AdamW moments and optimizer step counts. The existing
training objective, model architecture, learning rate and prescribed loop remain
unchanged. This is execution infrastructure; its learning benefit has not been
measured and no formal study checkpoint is adopted.

`CompletedParent.from_loop(completed_loop, training_transcripts=...)` requires
an actual completed loop with its full final evaluation. It checks ownership,
source and runtime identity, validates the learner through the original restore
path, and authenticates the supplied training transcript inventory against the
process-owned plan index. A claim of completion in a sidecar is insufficient.
The token binds the terminal envelope, final scores, persistent retention
references, complete weights and optimizer, and memory-versus-disk provenance.
It retains one learner image and detached metadata, without recursive ancestor
tensor images. It cannot be serialized or used as a study-checkpoint adapter.

`CycleFoundationTrainer` keeps its `cursor` and `updates` local to the child
curriculum. Separate `base_updates` and `lifetime_updates` preserve prior work;
each AdamW step must equal the lifetime count. At child cursor zero, weights and
moments must equal the authenticated parent exactly. Restore admits the full
new-plan evidence and validates all state before replacing the live learner.
Failed steps retain the existing explicit-restoration requirement.

Freshness requires more than another seed. The new plan's original admission
protection must include all parent training and previously protected transcripts.
Expanded protection may add exclusions, never remove them. The unchanged plan
index authenticates the entire admitted child curriculum before model creation.
Exact transcript disjointness is a claim about new realizations, not unseen
algorithms or general English competence.

`CycleFoundationPracticeProvider` connects that learner to the unchanged
`FoundationLoop`. It derives the learner contract from the parent and requires
the exact same development/retention roles, names and bank identities. Prescribed
requests, pending practice, teacher-free scoring, atomic publication and rollback
use the existing loop. Provider status exposes local and lifetime updates
separately; the loop's work counters describe child work only. `parent_evidence`
returns detached earlier results and references. They are not silently converted
into child-cycle retention alarms.

Resume requires reconstructing the authenticated parent from its original
completed loop, admitting the same child plan and banks, then loading the child
loop envelope with a matching cycle provider. The child checkpoint does not
replace those external evidence boundaries. Save a parent before capture if a
durable parent lineage is needed; an in-memory capture cannot claim disk evidence.

The active architecture comparison is independent and retains all 88 frozen
sources. These new modules do not change its data, models, budgets or selection.
First-handoff engineering validation is recorded separately under
`runs/foundation-cycle-training-validation-local/` and
`runs/foundation-cycle-provider-validation-local/`.

On 17 September, **11 trainer/token checks and 8 provider/loop checks passed
across focused CPU runs**. Both architectures used actual completed 66-update
parent fixtures. The checks cover exact weights and complete AdamW continuation,
durable parent reconstruction, child checkpoint restart, partial evaluation and
pending practice, transactional rejection, and rollback after failed publication.
Child baseline scores exactly matched parent final teacher-free scores on the
unchanged banks. This establishes state continuity on these fixtures, not a new
learning advantage. Independent review found no remaining implementation defect.

Two test-helper mistakes failed their intended rejection checks initially: one
selected the wrong parent, and one supplied a malformed bank name. Both were
corrected in tests only; only corrected and added checks were rerun. The original
failures and all repeated fixture work remain in the
trainer receipt (archive reference: `../runs/foundation-cycle-training-validation-local/report.json`)
and provider receipt (archive reference: `../runs/foundation-cycle-provider-validation-local/report.json`).
Together they record **558 completed CPU optimizer calls**, 3,354 drawn episode
exposures, 3,352 neural-attempted and 3,350 completed-microbatch exposures, plus
138 evaluation banks/276 scored episodes. One deliberate partial microbatch
failure and one completed update discarded after failed publication are included;
there were no unknown optimizer reports. No formal data or weights were used.

Repeated-cycle capture, automatic fresh-plan generation, a persistent retention
comparison across cycles, adaptive curriculum choice and a combined GPU proof
remain separate work. A completed handoff does not establish self-direction,
teacher independence, transfer or a benefit from evolutionary changes. Those
claims require prospective learning comparisons and retained capabilities at
measured local cost. See the [cycle design](FOUNDATION_CONTINUAL_CYCLE_DESIGN.md).
