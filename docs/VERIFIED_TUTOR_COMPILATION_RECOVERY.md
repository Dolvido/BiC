# Deterministic recovery of a colliding lesson

The first complete curriculum packaging attempt stopped before learner training.
It compiled 108 procedural bundles and 84 complete tutor bundles, then rejected
a protected or repeated fresh transcript in the next tutor bundle. The failed
receipt is retained at `runs/verified-tutor-data-local/attempt-001/preparation.json`,
SHA256 `35b26ab660acca727dad4806ed377d8d51b1670da1c5a093e40e78978a2f386a`.
Its 102.031 wall seconds, 101.359375 CPU seconds, 68,532 canonical generation
calls, and 137,064 returned rows remain preparation costs. It performed no
neural updates and made no teacher requests. None of its partial output is
silently adopted as a completed curriculum.

A focused reproduction identified the collision as protected text: both switch
members at zero-based slot84, pair2, depth4 and eight turns. The selected
change-emphasis/rename transformation reused text despite a different naming
seed. A finite alias vocabulary makes that possible. The recorded diagnosis
used 35 canonical pairs, one protected-set read and 2.312 wall seconds; it made
no learner or teacher calls.

The separately versioned compiler preserves the existing author result,
inventory, chapter choices, coverage, canonical procedures, protected transcript
set, accepted parent checkpoint, objective and training comparison. It preserves
the original candidate seeds for attempt zero. Only a fresh candidate that
collides may use a bounded deterministic alternative seed in the realization's
permitted dimension. Rename retains values; revalue retains names; independent
may change both. Replay is unchanged. Both members of a candidate pair must pass
before either is admitted. Every rejection and selected seed is recorded. If the
finite budget cannot produce a legal candidate, compilation fails rather than
weakening exclusions or changing the selected motif or realization.
Retries apply to one family pair at a time and may therefore change its aliases
without changing the other families' aliases. They preserve the shared procedure
and coverage; they do not promise identical rendered aliases across families.

The new data adapter runs compilation in a new attempt directory and pins the
original failed receipt alongside the earlier failed author receipt. It reuses
the accepted author result from attempt002 without any network request. Both
comparison curricula use the corrected compiler; changed examples are measured
from their actual model-visible contents. Their matched counts, original full
optimizer parent, unaided evaluation, common withdrawal and acceptance screens
stay as declared in `VERIFIED_TUTOR_PROTOCOL.md`. This correction is declared
before any tutor-guided learning outcome is available.
