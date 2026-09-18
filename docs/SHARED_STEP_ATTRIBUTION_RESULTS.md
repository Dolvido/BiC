# Two-update shared-step attribution

The first and only attempt completed in **5.485 seconds wall / 4.90625 seconds CPU**, including durable receipt publication, within its 60-second allowance. Completion-marker publication was separately reported as 0.015 seconds wall. Receipt, marker, copied harness/declaration and both step-artifact hashes were authenticated when writing this document; profile self-time sums and source-open counts were independently checked using JSON only.

The unchanged control objective restored the pinned update-13,000 research parent and consumed predetermined curriculum positions 0 and 1. It completed **two updates, 192 episode exposures, six family forwards/backwards and six auxiliary readouts/objectives**, reaching diagnostic cursor 13,002. There were no unknown optimizer outcomes, new lesson generation, evaluations, tutor/network calls, saved continuation or promotion. Restoration performed two data-only decodes of the same parent image, one model construction and one optimizer construction; broad replay added one archive decode. Hash-only setup archive reads are additional I/O, not model/archive decodes. Prepared-owner work was six family packs, 192 rows and 96 exact pair matches. Replay targets performed 96 English and 96 typed checks. Peak allocated/reserved GPU memory was 1,197.249/1,314 MiB.

## What was measured

Setup before either update took 3.703 seconds, including imports (1.250), original complete manifest admission (0.609), control transition (0.719), source/input authentication and runtime setup. These are nested setup components, not amounts to add to setup. Final authentication took 0.141 seconds and cleanup 0.063 seconds.

| Profiled operation | First: basis JSON | Second: broad replay |
| --- | ---: | ---: |
| Whole step, including loading and durable report publication | 0.891 s | 0.687 s |
| Actual turns per episode | 12 | 8 |
| Observation-token exposures | 38,892 | 21,525 |
| Kernel's own elapsed-time counter | 0.485 s | 0.140 s |
| Source-file opens observed by audit hook | 1,639 | 1,639 |

The first step is also the first native GPU forward/backward of this process. Its larger contexts and possible first-use CUDA effects are confounded with lesson type and order. The difference between the two steps therefore does not measure JSON-versus-archive efficiency or steady-state GPU cost.

The following cProfile categories contain **exclusive function self time**, in milliseconds. Categories are disjoint, but deliberately broad: the tensor category includes Torch archive decoding, and filesystem/path time includes in-memory stream operations. They are not GPU-device-time or physical-disk-I/O measurements.

| Exclusive category | Basis JSON | Broad replay |
| --- | ---: | ---: |
| Model/optimizer/tensor operations | 383.417 | 174.717 |
| Other Python/builtins | 193.117 | 205.798 |
| Python copying | 143.094 | 124.145 |
| Filesystem/path operations | 108.989 | 120.932 |
| JSON | 30.465 | 17.204 |
| Packing/tensor identity | 24.228 | 18.854 |
| Hashing | 12.612 | 11.739 |
| Prefix targets/oracles | 0 | 6.038 |
| Source-closure dispatch | 1.483 | 1.503 |
| Guard/runtime dispatch | 0.667 | 0.717 |
| Sum of recorded function self time | 898.072 | 681.646 |

The dispatch rows exclude their child reads, hashes and copies; their small self times do not imply cheap source checking. External elapsed counters and profiler sums use different clocks/resolution and do not reconcile exactly. No unprofiled comparator exists, so profiler and audit-hook overhead are **unmeasured**. Hundreds of thousands of recursive copy/builtin calls make profiling perturbation particularly relevant.

## Inclusive paths: do not add to the preceding table

| Function, inclusive across its calls | Basis JSON | Broad replay |
| --- | ---: | ---: |
| Owner `prepare` | 219.873 ms | 251.512 ms |
| Objective bridge `step` | 539.052 ms | 190.545 ms |
| Shared-state kernel `step`, inside the bridge | 489.950 ms | 141.091 ms |
| Owner `_consume`, inside the kernel | 24.090 ms | 25.243 ms |
| Owner `_guard`, three calls | 69.664 ms | 70.128 ms |
| Objective bridge `_guard`, two calls | 49.068 ms | 49.431 ms |
| Shared-state kernel `_guard`, three calls | 17.586 ms | 17.219 ms |
| `deepcopy`, including recursive children | 234.878 ms | 258.775 ms |
| Ordinary step-report `publish` | 2.530 ms | 1.576 ms |

The shared-state guard calls the base kernel guard; the latter's 17.578/17.213 ms is **already inside** the shared-state guard and is not another cost. The three distinct guard paths above sum to approximately **136.318/136.778 ms** per profiled step without that duplicate. Nested `source_hashes()` timings likewise must not be summed across inherited closure layers. JSON loading/admission took 17.337 ms in the first step; the replay weights-only unpickler took 127.693 ms inclusively in the second, with prefix-target construction adding 9.088 ms on a separate path. These observations do not make either loading format generally faster.

Repeated source verification is now observed work, not only a call-graph hypothesis: both steps opened source files exactly 1,639 times. Copying and preparation are also substantial in this profile; report publication was small. The two samples do **not** establish that source I/O dominates the preceding 2,174.875-second sustained run or explain its entire time outside the kernel counter. Prepared-owner and kernel timings overlap there as well as here.

## Smallest useful change to consider

Prototype an additive source-verification boundary that assembles the exact reviewed dependency-path set without recursively hashing inherited closures, then reads/hashes each unique file **once per existing guard invocation**. Preserve the same source membership checks, expected hashes, guard cadence, runtime checks and failure behavior. Do not cache successful verification across updates or use modification times as a substitute for bytes. Frozen existing sources and identities remain unchanged; any new execution boundary needs an explicit new identity.

Structurally, owner checks currently perform 277 reads for 70 unique files, bridge closure checks 292 for 78, and kernel checks already read their 74-file set once. Deduplicating only work *within* each guard could reduce ordinary-step source reads from 1,639 to 590 while retaining every guard. That is a proposed operation-count reduction, **not a measured speedup**. It is a narrower first experiment than changing mutable lesson ownership or removing validation. Before integration, require exact source-map/mutation rejection, unchanged next-update weights/AdamW/evidence, and a separately bounded unprofiled inclusive-cost comparison. No optimization was implemented by this report.

## Immutable evidence

| Artifact | SHA-256 |
| --- | --- |
| `runs/shared-step-attribution-local/attempt-001/receipt.json` | `47bbbe87a2e16c9e8b16528f8edfabbaa5ca1808f0c626faad8dade84013dcf9` |
| `runs/shared-step-attribution-local/attempt-001/completion.json` | `fe5bcc8871f3643f2a37f8e00df37cd6510552d31e382386b97613be9ff0f465` |
| Final harness, including reviewed partial-work failure accounting | `5057d8c27440878a578b43f6361dcec4ce2846f87635ce6d0f0e7cf5371bf3c7` |
| Prospective declaration | `564525c75df10e889cb3f17f91eb12c3f55af8436b8f6a03120a54f1f6a1451d` |
| Original objective-study launch | `f6670f922676306500d6f97e91c83a2dbdb171f2ae3293cebecc8fcf623d804e` |
| Original update-13,000 checkpoint | `4780cd241a17516003e2e162b45d2fb3072d9f533e68586edb4e562de4b412fa` |

Raw per-function profiles, source-open maps, nested restore costs, original lesson records and step-artifact pins remain in the receipt. All lessons, prior studies and frozen sources remain unchanged.

## Separate byte-cache measurement

The first byte-cache comparison subsequently completed in **123.703 seconds wall / 123.078125 seconds CPU**, with zero archive accesses, neural imports or network calls. Its authenticated receipt is `runs/immutable-lesson-bytes-cache-analysis-local/attempt-001/receipt.json`, SHA-256 `d3893067aba92257d5fdfc3d6d798df98247622b6c9adb65c46e6a78ea7f8c04`. This was a separate isolated CPU measurement, not a concurrent phase of the two-update profile above.

| Fixed JSON schedule | Reference inclusive | Cache inclusive | Observed reduction | Reference loading only | Cache loading only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control, four passes | 32.641 s | 26.953 s | 17.43% | 19.163 s | 12.188 s |
| Curriculum, four passes | 33.109 s | 30.766 s | 7.08% | 19.387 s | 14.745 s |

Inclusive path costs retain construction, cold admission, detached returns, all five verification boundaries, exact-content checks and cleanup. Loading-only values are components of those totals. All **5,184 returned images** passed exact image, ordered learning-input/state-target, evidence and original-cursor checks; each reference/cache ordered stream hash agreed. There were no evictions with 324 entries and a 256 MiB serialized-byte cap. Across both cache arms, actual work included **540 admission decodes, 2,592 returned-lesson decodes, 4,752 metadata-record decodes and 2,160 resident-file checks**. Hits still decode JSON. Retained serialized bytes were 137,814,848 for control and 211,493,182 for curriculum; these are not Python RAM measurements.

The fixed order was control/reference, control/cache, curriculum/reference, curriculum/cache. OS file-cache effects and chronology can favor later paths; there was no randomized order, unprofiled training comparator or repeated best-of selection. The measured saving applies to this isolated JSON-loading protocol. Its share of whole-learner cost remains unknown, and these seconds must not be directly extrapolated from the different, nonconcurrent training/profile settings. There is **no end-to-end training-throughput evidence or automatic integration**. The rate comparison retains the original data-loading and preparation path so the scientific comparison stays unchanged.
