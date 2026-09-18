# Explicit recovery of the first chapter request

The first live author slot, `runs/verified-tutor-author-local/attempt-001`, made
two successful local metadata requests and one chat attempt. The chat returned
HTTP 400 after the local service loaded and warmed the model. No completed reply
was recorded. The original transport discarded the error body, so the exact
cause is unconfirmed; an array-schema compatibility problem is the current
hypothesis. This is not a zero-compute attempt. Its 2.5-second author receipt
retains unknown physical generation work and no claimed token count.

Preserve that intent and uncommitted receipt unchanged. Its receipt SHA256 is
`6284cc0d41cf1dbab7c8eae77fb8e90e286ee645f0bdc484a3bb1896e3a3d150`.
It is closed without adopting any teaching choice. Never replay that slot or
silently convert it into a successful request.

This is a separately declared compatibility correction, made before any tutor
training or comparison outcome. The additive v2 author uses a portable uniform
array-item schema with the permitted choice union and fixed chapter count.
The unchanged compiler still enforces every position's exact menu, so a first
replay chapter cannot select an authored alternative. Preserve the same parent,
aggregate development evidence, chapter coverage, default, realization meanings,
seed, model digest, prompt budget, 180-second allowance and keep_alive=0.
Record bounded HTTP error bodies if a future request is rejected.

One new explicit live request may be made in `attempt-002` after the revised
author's focused mocked checks pass. This is not an automatic retry loop and
does not erase the first attempt. If it succeeds, retain both attempts' costs;
report one successful completion separately from the total two chat attempts.
If it is rejected or uncertain, preserve that outcome and diagnose it rather
than silently making further calls.

The completed inventory and canonical compiler remain unchanged. A separate
data adapter authenticates the v2 author result and compiles through that same
compiler. The matched training design, original full AdamW parent, unaided
evaluation, common withdrawal, acceptance screens and compute bounds remain as
declared in [the protocol](VERIFIED_TUTOR_PROTOCOL.md). Freeze the final execution
sources and both teacher-attempt records before any learner update.
