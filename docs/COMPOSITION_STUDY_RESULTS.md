# Shared compositions: completed local comparison

Neither joint practice nor sequential practice with replay learned these shared
compositions reliably. The schedule effect differed by domain: joint practice
did better on withheld color compositions, while sequential practice did better
on count compositions. A separate diagnostic found strong training fitting but
poor generalization for color and switch, alongside poor fitting of count.
**No checkpoint was promoted.** This closes the comparison, not the broad
home-learning objective.

All five training runs and five audits completed locally. The
verified summary (archive reference: `../runs/composition-study-local/audit/summary.json`) retains
every panel, length, denominator, confusion matrix, reply result and comparison.
The [prospective protocol](COMPOSITION_STUDY_PROTOCOL.md) defined no automatic
promotion or numerical winner. Sixty frozen source files, bank boundaries,
checkpoints, optimizer states and exact sample streams were verified; all five
CPU restart checks were exact. The final engineering suite passed 578 tests in
59.221 seconds. Validation (archive reference: `../runs/composition-study-local/validation.json`)

## Shared curriculum and matched schedules

Color, count and switch share one sampled set/copy/advance/query procedure.
Copy captures the current source value; advancing an unknown value leaves it
unknown. Color follows a stated four-color cycle, count uses a signed increment,
and switch toggles. Independent abstract and English interpreters verify every
answer. The learner sees raw observation bytes, without parsed state, family
IDs, target labels or teacher replies as context.

Episodes contain 8, 10 or 12 turns. Each counterfactual pair changes one initial
fact while preserving questions and remaining events. The final question
depends on that fact through both copy and advance and has opposite known
answers. Unknown probes occur earlier; final uncertainty is unmeasured. These
are restricted unary dependency chains. Their syntactic identities do not prove
distinct semantic algorithms, arbitrary relational reasoning or general intelligence.

All runs used the 753,610-parameter sequence learner, initial seed 2801 and
learning rate 0.001. Each update averaged three 32-episode objectives. Joint
practice used one draw per family throughout 3,600 updates. Each sequential run
used 1,800 old-family updates, then 1,800 updates with two late-family draws and
one old-family replay draw. Every main run consumed the **same ordered 115,200
episodes per family**, including replay. Each fresh control used its late-family
stream over 1,200 updates. Fresh controls match late-family exposure, not total
compute, optimizer steps or other-family exposure. Both rotations share one
joint reference and are not independent replications.

## Bank boundaries and actual coverage

There were 4,608 training, 2,304 development and 2,304 audit episodes. All 9,216
full observation transcripts were distinct. Each training bucket had 256 complete
pairs and 256 naming maps/value seeds; each evaluation panel had 64 pairs and 64
maps/value seeds. Value-seed counts are not counts of distinct semantic worlds.
Whole procedures can repeat with fresh values/names, especially for Boolean values.

| Family | Training final ancestry IDs, 8 / 10 / 12 turns | Training full procedures, 8 / 10 / 12 | Audit held ancestry IDs, 8 / 10 / 12 |
|---|---|---|---|
| Color | 29 / 101 / 169 | 236 / 254 / 256 | 6 / 27 / 41 |
| Count | 31 / 99 / 181 | 244 / 256 / 256 | 7 / 29 / 34 |
| Switch | 29 / 102 / 157 | 243 / 256 / 256 | 8 / 24 / 44 |

Across domains and lengths there were 501 distinct supervised composed training
ancestries, 153 development final composed ancestries and 126 audit final composed
ancestries, with the declared boundaries verified. These global counts are not
sums of overlapping buckets. Development rejected 61 candidate pairs because an
earlier supervised query exposed an audit motif. Primitive queries and unqueried
subexpressions are outside the withholding claim. Earlier audit queries can mix
familiarity; final pairs supply the clean held-composition endpoint. The maximum
utterance was 67 bytes and the longest encoded episode 634 tokens, within the
1,024-token model capacity. Frozen manifest (archive reference: `../runs/composition-study-local/protocol.json`)

`seen` panels use familiar sampled structural recipes with fresh values and
consistent name mappings. `composed` panels use withheld final ancestry signatures
and fresh renderings. Both use familiar alias vocabulary. Failure on `seen`
therefore already precedes the new-ancestry challenge.

## Final paired decisions and learning curves

Each cell counts **both opposite final answers correct**, out of 192 pairs per
family/panel: 64 at each length.

| Family and panel | Joint | Color late + replay | Count late + replay | Fresh on this family |
|---|---:|---:|---:|---:|
| Color, seen | 32/192 (16.67%) | 37/192 (19.27%) | 38/192 (19.79%) | 29/192 (15.10%) |
| Color, composed | 47/192 (24.48%) | 33/192 (17.19%) | 46/192 (23.96%) | 31/192 (16.15%) |
| Count, seen | 10/192 (5.21%) | 11/192 (5.73%) | 30/192 (15.63%) | 10/192 (5.21%) |
| Count, composed | 8/192 (4.17%) | 14/192 (7.29%) | 25/192 (13.02%) | 6/192 (3.13%) |
| Switch, seen | 51/192 (26.56%) | 44/192 (22.92%) | 40/192 (20.83%) | Not run |
| Switch, composed | 45/192 (23.44%) | 49/192 (25.52%) | 39/192 (20.31%) | Not run |

For late-family composed panels, correct pairs at 8 / 10 / 12 turns were
15 / 13 / 19 for joint color, 9 / 7 / 17 for sequential color and 9 / 8 / 14 for
fresh color; count was 4 / 3 / 1, 8 / 8 / 9 and 1 / 3 / 2 respectively, each out
of 64. There is no consistent length trend or schedule winner.

Sparse normalized area under the **final-pair learning curve**, indexed by
late-family episode exposure, was:

| Held composition | Joint | Sequential + replay | Fresh | Sequential minus joint |
|---|---:|---:|---:|---:|
| Color | 21.20% | 18.36% | 16.75% | -2.83 percentage points |
| Count | 1.64% | 5.75% | 1.23% | +4.11 percentage points |

Sequential zero exposure already contains old-family learning and its optimizer
state. Areas mix starting competence with later acquisition; they are not pure
learning-speed or meta-learning estimates. One initialization cannot establish
a robust ranking.

Aggregate query accuracy on late-family composed panels was 46.71% / 47.31% /
44.01% for joint/sequential/fresh color, and 49.63% / 51.70% / 47.34% for count.
Unknown responses contribute materially. Sequential color answered 290/808 known
and 349/544 unknown queries correctly; sequential count answered 372/834 known
and 340/544 unknown. Percentages average the three length buckets; these pooled
counts preserve the actual denominators. Aggregate success is not reliable
known-state computation.

Free-running final reply pairs on these sequential composed panels were also
33/192 for color and 25/192 for count. All query replies were parseable, but
parseability is not correctness. High action/reply agreement can mean both outputs
are wrong. Blank and reset-history controls solved zero final pairs in every run.
Resetting history preserved roughly 38% aggregate query accuracy through unknown
responses while known-query accuracy was zero. History matters to behavior; these
controls do not establish mastery of composition.

## Retention and the separate fitting diagnostic

The old-family audit macros below average two old families, both panels and lengths.

| Rotation | Old query accuracy | Old known-query accuracy | Old final-pair accuracy | Old free-reply accuracy |
|---|---|---|---|---|
| Color late | 48.88% -> 50.62% | 43.20% -> 40.88% | 16.67% -> 15.36% | 48.59% -> 50.86% |
| Count late | 48.72% -> 49.93% | 43.13% -> 43.34% | 20.83% -> 21.22% | 48.58% -> 49.98% |

Color-late's higher aggregate query accuracy concealed worse known decisions.
Its familiar-motif count panel lost 5.63 points of known accuracy and 3.65 points
of final-pair accuracy. Panel/family final-pair changes ranged from -3.65 to +4.17
points across both rotations. Preservation was not uniform and initial abilities
were already limited.
Length-specific losses were larger: color-late's seen count and seen switch
8-turn banks each lost six correct pairs out of 64; count-late's seen color
8-turn bank also lost six, and composed switch at 12 turns lost five.

After the prospective comparison closed, a separately labeled
read-only training-fit diagnostic (archive reference: `../runs/composition-study-local/diagnostics/train-fit.json`)
scored every admitted training row at the fixed endpoints. It used no optimizer
updates or checkpoint selection and did not alter the original audit. Its
15.25 seconds included 5.766 seconds of preparation.

| Learner | Training color pairs | Training count pairs | Training switch pairs |
|---|---:|---:|---:|
| Joint | 750/768 (97.66%) | 119/768 (15.49%) | 743/768 (96.74%) |
| Color late + replay | 760/768 (98.96%) | 151/768 (19.66%) | 692/768 (90.10%) |
| Count late + replay | 744/768 (96.88%) | 264/768 (34.38%) | 731/768 (95.18%) |
| Fresh color | 746/768 (97.14%) | Not trained | Not trained |
| Fresh count | Not trained | 66/768 (8.59%) | Not trained |

Color and switch fit repeated examples but transfer poorly even to familiar
motifs with fresh realizations. Count remains poorly fitted. Both limitations
coexist; the data do not identify one universal capacity, curriculum or optimizer
cause. Training and evaluation contents/target mixtures differ, so fitting gaps
are descriptive diagnostics rather than causal estimates.

## Local computation and decision

The retained study contains **13,200 updates, 39,600 microbatches and 1,267,200
repeated episode exposures**, totaling 393,019,622 observed UTF-8 bytes. Durable
worker optimizer intervals sum to 2,345.312 seconds; audit worker intervals sum
to 37.829 seconds. Concurrent worker sums are not elapsed time or dedicated
GPU-hours. Final-invocation peak allocated CUDA memory ranged from 764.11 to
1,431.59 MiB, excluding other processes and driver memory.

Both fresh controls were deliberately paused after combined GPU memory reached
15,865/16,303 MiB; there was no out-of-memory failure. They resumed from durable
update 64 and completed fixed 1,200-update endpoints with restored optimizer and
sampler state. Discarded uncheckpointed work is unknown: **0–192 updates and
0–18,432 episodes per control**, separate from retained counts and durable timers.
A separate disposable execution probe used two updates and 192 episodes; its
weights were discarded. Invocation and allocator measurements cover the final
resumed invocation only. The
execution record (archive reference: `../runs/composition-study-local/execution.json`) preserves the
observed phase envelope and timing limitations. These are execution measurements,
not evidence that home computation or learning efficiency is exhausted.

The next general experiment should compare fixed-bank repetition with continually
fresh, independently verified values and name realizations of shared training
procedures, with identical structural draw streams and declared matched episode
budgets. Freshness also changes count increments and query truth, so this is not
lexical-only or byte/FLOP-matched; record target mixtures, collisions, bytes and
runtime with independently resumable realization randomness. Preserve separate familiar-motif
and held-composition boundaries across domains; measure fitting, known paired
decisions and retention together. This tests reusable-rule generalization rather
than repairing one domain. Capacity or representation comparisons remain useful
follow-ups if fresh realizations do not resolve the limits. Repeat seeds before
capability claims; an adaptive selector first needs a progress signal that tracks
real acquired skills.

The separate 1,309,188-parameter regional adapter remains untrained. No external
teacher participated in policy evaluation, and no released checkpoint changed.
Useful local learning, retention, self-direction, usable integration and declining
teacher dependence remain the open [home-learning objective](HOME_LEARNING_ROADMAP.md).
