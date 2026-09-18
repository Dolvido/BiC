# Reusing verified packed lessons

The additive packed-v2 preparation, kernel and explicit migration adapter are
implemented. The original design below remains the record of their compatibility
and validation requirements; no frozen prior implementation was changed.

The GPU proof (archive reference: `../runs/shared-state-packed-v2-gpu-validation-local/attempt-001/report.json`)
passed exact full weights, AdamW, evidence, cursor and physical-work equivalence.
Its matched timing slice used six images over two passes: 5.033 seconds packed-v2
versus 4.826 seconds reference. This short check shows no speedup and does not
justify adoption. Receipt SHA-256:
`f6033dd6c54a4569372e763390e29cd8279e91897a5e3a509495d27b2c840c26`.

The sustained acquisition run (archive reference: `../runs/sustained-acquisition-local/attempt-001/launch.json`)
is active with a 1,800-second wall budget, complete 648-bundle curriculum replay
and 2,340 evaluation episodes per endpoint. It uses the original reference path
with a bounded resident replay cache, not packed-v2. No new teacher requests are
made; the completed learning report is pending. Launch SHA-256:
`5ab38001fd31c740b47fe4e6e7b26119d39dce019f7cf2b1d3322c46f1053525`.

## Observed cost and opportunity

The completed shared-state continuation attempt001 took 1,225.703 seconds wall
and 1,161.34375 seconds CPU. Its recorded operation times were:

| Operation | Wall seconds | Fraction of total wall |
| --- | ---: | ---: |
| Replay archive loading | 285.212 | 23.27% |
| Replay remapping | 140.205 | 11.44% |
| Training preparation | 294.707 | 24.04% |
| Training, including token consumption | 452.008 | 36.88% |
| State targets | 5.553 | 0.45% |
| Evaluation-bank preparation | 5.625 | 0.46% |

Source: `runs/shared-state-continuation-pilot-local/attempt-001/execution/summary.json`.
Loading/remapping includes source guards, JSON hashing and copying as well as
deserialization; these are not isolated measurements of disk or pickle speed.
The replay reader's 422.043-second internal cost overlaps those operation times.
Owner consumption timing likewise overlaps training. Do not add either again.

The two arms separately prepared the same 1,296 shared bundles: 2,592 preparation
calls. Eliminating one preparation per pair addresses approximately 147.354
seconds, or 12.02% of total wall, before replacement checks and copying. This is
an affected-work ceiling, not a measured speedup. Training preparation already
reports zero canonical regenerations; repeating oracle generation is not its
bottleneck. State-target computation is also a small share.

The verified-tutor runner already shares withdrawal decoding and state targets,
but independently packs each learner's batch. Its 108 common withdrawal updates
and first36 fixed teaching replay updates represent144 duplicate preparations.
At the prior average of0.113699 seconds per preparation, these address about16.37
seconds. Actual current-run cost must be measured before prioritizing the change.
Persisting packed images may later reduce nested raw-row deserialization, but
moving a single preparation earlier does not itself save total work.

## Frozen compatibility blocker

`foundation_layout_prepared.PreparedLayoutOwner.prepare` is the only public
token-producing path and always performs admission matching and all three family
packs. `PreparedLayoutBundle` requires the module-private `_MINT` object.
`_consume` requires the token to be that owner's exact active token, checks its
stored evidence and tensor identity, and clears its tensors after consumption.

Both `foundation_layout_kernel.PreparedLayoutKernel.step` and
`shared_state_training.SharedStateKernel.step` require the exact existing token
and owner classes. A subclass or external owner adapter is rejected. The
continuation bridge delegates to this same strict kernel. Therefore an additive
adapter compatible with the unchanged core would have to forge private tokens,
rewrite owner state, or monkeypatch packing. None is acceptable.

## Proposed public contract

A future explicitly versioned preparation module should expose a verified packed
image producer and `owner.prepare_from_verified_image(image, *, expected_evidence,
expected_evidence_sha256)`. The producer performs the present row admission,
complete-pair matching and CPU packing once. It binds ordered row identity,
configuration, layout, microbatch dimensions, packer sources and exact tensor
dtype/shape/content. State labels, if included, additionally bind target-generation
sources and stay in supervision, never policy inputs.

The image exposes no writable tensor views. Each owner authenticates the image
and caller-pinned evidence, then internally mints its own independent single-use
token. Consumer cursor and evidence remain branch-specific; changing a replay
cursor must not change packed examples. Each owner allows at most one outstanding
token. Consumption retains detached consumer copies, mutation checks, failure
poisoning and exact cursor/configuration checks. Closing one owner must not damage
the other owner's token or the source image. Initially retain only one shared
image through the paired update; no prefetch process or unbounded cache is needed.

The corresponding versioned kernel must explicitly accept the new public types;
it must not weaken admission to arbitrary duck-typed objects. Preserve objective,
family order, state-loss weight, clipping and AdamW exactly. Checkpoints must name
the new implementation and explicitly authenticate any old-to-new state transition;
never silently rewrite a frozen recipe or its source identity.

## Bounded validation before use

First use one existing admitted two-episode fixture per family. Compare one
reference preparation with one new image and two independent token consumptions:
six family packs,12 packed rows, three successful consumptions, no model, oracle
generation or optimizer work. Compare all inputs, supervision and state targets
exactly; verify unchanged evidence apart from an explicitly permitted cursor.
Exercise wrong evidence/config/cursor, mutation, cross-owner use and double-use
rejection without using an already-spent token as the wrong-cursor test.

Then compare one official next update with one new-path update from the same
authenticated full learner/AdamW state under the strict production FP32 profile:
two restored models, two physical updates, six family forwards/backwards and192
episode exposures at microbatch32. Require exact losses, complete weights, AdamW,
targets, evidence chain and exposure counters. This is an equivalence check, not
a throughput benchmark.

Finally measure complete paired-update wall/CPU cost on a fixed ordered slice,
including image creation, hashing, copying, source checks and cleanup. Alternate
which implementation runs first across matched slices; preserve equal examples,
updates and thermal order. Report all setup costs rather than hiding them as
amortized work. Adopt only a measured net benefit. The completed bounded proof
above establishes exactness for its tested slice; its measured timing does not
establish a benefit. Broader speed or adoption claims remain unsupported.
