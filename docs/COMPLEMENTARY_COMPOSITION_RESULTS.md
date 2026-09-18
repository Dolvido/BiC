# Complementary composition: completed first-pass comparison

The complete matched comparison finished and its raw answers were independently
recounted. Neither candidate qualifies for retention. This result primarily
shows inadequate acquisition of the added composed lessons, including practiced
examples; it does not establish a ceiling on transfer after successful learning.

Both branches started from the same unadopted research checkpoint at update
9,760 and performed 648 updates with the same model, optimizer, learning rate,
older-definition practice and broad replay. The control continued atomic-basis
practice. The curriculum branch used 36 atomic and 180 complementary composed
teaching batches. Each new composed batch was encountered once.

Complete pairs require correct native actions and generated replies for every
query in both counterfactual episodes. These are stricter than average query
accuracy, and are used consistently across branches.

| Final complete pairs | Shared start | Control | Curriculum |
| --- | ---: | ---: | ---: |
| Atomic binding /96 | 10 | 31 | 3 |
| Atomic revision /96 | 78 | 89 | 55 |
| Held-out composition /96 | 0 | 0 | 0 |
| Held-out sequence /96 | 0 | 0 | 0 |
| Complementary fit binding /60 | 0 | 0 | 0 |
| Complementary fit revision /60 | 2 | 3 | 7 |
| Complementary development binding /120 | 0 | 0 | 0 |
| Complementary development revision /120 | 3 | 2 | 10 |

The primary composition/sequence contrast is zero percentage points. The
curriculum fails all required acquisition/capability screens. Control fails
one retention check against the shared parent and four against retained update
9,112; curriculum fails eight against each. The retained checkpoint is unchanged.

The curriculum's practiced revision known-query joint accuracy reached 474/600
(79%), but includes 240/240 prior-value queries. Its composed destination joint
accuracy was only 63/120 (52.5%). Practiced binding destination joint accuracy
was 122/240 (50.83%). Those aggregates do not show reliable composed execution.
Binding also showed disagreement between the action and generated reply outputs.

The next experiment extends the same schedules with full optimizer continuation,
fixed fit/development panels and retention screens. It does not add another
targeted curriculum patch or infer a particular architectural cause.

The held-out programs are structural holdouts. Equivalent functions exist in
some domains; this is not an all-domain test of unseen mathematical functions.
The learner's earlier history already included some multi-clause definitions.

## Measured work and immutable evidence

- Worker: 1,296 updates, 124,416 episode exposures, 3,888 family forwards and
  backwards, eight full restorations and six snapshots. No unknown optimizer
  outcomes and no teacher calls in this experiment.
- Worker wall time: 568.235 seconds; CPU time: 535.78125 seconds. Peak allocated
  GPU memory: 1,557,689,856 bytes; peak reserved: 4,286,578,688 bytes.
- New data preparation: 16.610 seconds wall and 16.40625 seconds CPU; separately
  accounted from the worker limit.
- Independent recount: 12.391 seconds wall, 12.359375 seconds CPU; 128 bank
  endpoints, 208 raw score files, 35,232 episodes, 182,480 queries and all 1,296
  step receipts. No model execution or checkpoint deserialization.

Protocol: [COMPLEMENTARY_COMPOSITION_PROTOCOL.md](COMPLEMENTARY_COMPOSITION_PROTOCOL.md).

Artifacts relative to the project:

- `runs/complementary-composition-local/attempt-001/launch.json`:
  `a4f4a73d4f45ff23ec7ca1d079ffd5a5ef9bd58b23ad315ce63e91d91038b458`.
- Its `execution/summary.json`:
  `b384bb7aa293dff92d7774c9f6f11552e5d8488a5a2df38a29a123945d2f1e2e`.
- `runs/complementary-composition-analysis-local/attempt-001/independent-analysis.json`:
  `1af8fe62eaa3736ea5bfe8d7fec2f8460f339572f470127ba323ac1b400a9ab7`.
- Data manifest: `f5a71eafc49e0b99f9fa893b0c1f9385df26b618e12c79a99b9dd2f7747de896`.

These development panels have been examined repeatedly. They are not a fresh
generalization audit and do not establish general English comprehension.
