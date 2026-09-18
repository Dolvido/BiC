# Optional local tutor data pilot

This adds a separate offline data preparation experiment to BiC v0.6. Normal
BiC inference does not import it, contact Ollama, or use a language model to
choose actions. Existing release data and checkpoints are unchanged.

The initial scope is **50 candidate records**, using three explicitly allowed
wording families across the five existing regional relations. The teacher is
asked to copy a chosen approved sentence into a structured record. This tests
the data pipeline, local service, validation, and provenance. It does **not**
measure open-ended paraphrasing, teach BiC, or demonstrate improved learning.
It also provides no evidence that asking an LLM to copy these sentences is more
useful than generating the same sentences procedurally.

## Run from the source directory

Use the project's installed Python environment. On this Windows machine:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_local_tutor_pilot.py -v
.\.venv\Scripts\python.exe -m experiments.local_tutor_pilot prepare --output ..\local-results\tutor-smoke-20260916\protocol
.\.venv\Scripts\python.exe -m experiments.local_tutor_pilot generate --protocol ..\local-results\tutor-smoke-20260916\protocol\protocol.json --output ..\local-results\tutor-smoke-20260916\generation --model ministral-3:3b --max-seconds 600
```

Use new output directories for each run; existing outputs are never overwritten.
Preparation and importing require the ordinary BiC dependencies because the
verifier calls the existing simulator function. They do not require Ollama.
Generation requires an already installed local model and the existing Ollama
service at `127.0.0.1:11434`. It does not install packages, download models, train,
start a service, or execute generated code.

The generator refuses to start while any Ollama model is active. Schedule it
separately from GPU training or evaluation. It unloads only its selected model
afterward and records the final running-model list. A generation failure stops
new HTTP requests; any remaining cases are reported as missing. The total
generation budget is checked before each candidate, each generation request
has at most a 60-second timeout, and cleanup/provenance requests can take
additional time. There are no automatic retries.

The fixed loopback endpoint bypasses environment proxies and rejects HTTP
redirects. Cloud-tagged models and model metadata declaring a remote model or
host are refused. This is a client boundary, not an independent audit of the
local Ollama service's implementation.

Reimport saved candidates without Ollama:

```powershell
.\.venv\Scripts\python.exe -m experiments.local_tutor_pilot import --protocol ..\local-results\tutor-smoke-20260916\protocol\protocol.json --candidates ..\local-results\tutor-smoke-20260916\generation\candidates.jsonl --output ..\local-results\tutor-smoke-20260916\reimport
```

An external-file import marks model provenance as unverified; it never invents
a model revision for manually supplied data. The original generation manifest
remains the source for model provenance.

## Data and validation contract

The teacher may return exactly three string fields:

```json
{
  "lesson_id": "pilot-000",
  "instruction": "choose the item called \"pilot000\".",
  "relation": "find"
}
```

The exact quoted name must match the trusted simulator case. Each instruction
must match exactly one complete sentence in `GRAMMAR`, including its punctuation
and casing. The independent parser's relation must agree with both the proposed
relation and the simulator's chosen relation. Unknown wording, negation,
compound or ambiguous commands, wrong names, multiple quoted names, extra
fields, duplicate JSON keys, invalid Unicode, and nonfinite JSON numbers are
rejected. There is no keyword-based acceptance or LLM grading.

Input instructions are limited to 128 UTF-8 bytes, canonical replies to 64,
and candidate payloads to 2,048. The generator also limits model output to 192
tokens with a 4,096-token context. Limits cause rejection, never truncation.
Import files and HTTP responses have separate size caps.

The verifier derives unknown-label, absent, and ambiguous conditions from the
trusted case. For one known matching object, it calls the existing
`experiments.train_regional_memory.relation_target` to recompute the selected
cell or boundary STOP. The teacher cannot supply targets or replies. Replies
are always `cannot select.` or `selected <cell name>.`.

The 50 cases contain 25 valid selections, 7 unknown labels, and 6 each of absent,
ambiguous, and boundary cases. Every relation has ten candidates. Protocol
fixtures are recomputed from their seed when importing; modified scene facts
fail validation. Code and grammar hashes must match the frozen protocol. A code
change requires preparing a new protocol.

Accepted records include a `spec` field compatible with the existing
`encode_specs` function, with the actual accepted instruction in `spec.prompt`.
Targets, hidden object identities, category, and expected reply are
training/scoring metadata. They must not enter the learner's inference inputs.

## Files and interpretation

- `protocol/protocol.json`: frozen simulator facts, seed, grammar, schema, and
  source hashes.
- `generation/manifest.json`: model digest/revision and quantization metadata,
  runtime version, generation settings, timestamps, elapsed time, cleanup, and
  source/protocol hashes.
- `generation/raw/`: serialized requests and complete responses.
- `generation/candidates.jsonl`: every attempted candidate or budget/error
  record, request/response hashes, and running-model digest.
- `generation/validated/accepted.jsonl`: validated lessons with independently
  derived actions, fixed replies, actual prompts, and provenance hashes.
- `generation/validated/rejected.jsonl`: rejected records and concrete reasons;
  malformed source lines are retained when within size bounds.
- `generation/validated/report.json`: acceptance counts, relation/category/family
  coverage, accepted lessons per minute, hashes, and model provenance.

Duplicate lesson IDs are all rejected, rather than selecting a favorable
duplicate. Missing cases are explicit rejections. An importer can therefore
produce more rejection records than candidates when duplicate and missing IDs
coexist. Acceptance rate uses the fixed denominator of 50 expected cases.

These are reusable pipeline fixtures, **not a training/development/test split**.
Their random glyph identities are not guaranteed disjoint from release data.
Do not use this sample for a held-out accuracy or learning-improvement claim.
Before the larger experiment, freeze independent identities and entire phrasing
families by split, review any new grammar, preserve old-task replay, and compare
matched procedural and teacher curricula across seeds as proposed in
[LOCAL_LLM_TEACHER.md](LOCAL_LLM_TEACHER.md).

The API choices follow Ollama's official [chat endpoint](https://docs.ollama.com/api/chat),
[structured output guide](https://docs.ollama.com/capabilities/structured-outputs),
[running-model endpoint](https://docs.ollama.com/api/ps), and
[model unloading behavior](https://docs.ollama.com/api/generate).

## First local results — 2026-09-16

The original 50 candidates were all rejected: 35 failed the exact wording check
and 15 failed the quoted-name check. After clarifying only the generation prompt
with an exact JSON example and explicit punctuation instructions, a separate
50-candidate run passed 50/50 with the same grammar, validator, settings, model,
and cases. The first failure remains preserved. Total generation time was 14.46
seconds and 13.47 seconds, respectively; these are tiny copying-task measurements.

All 15 new tests passed, and offline reimport reproduced the second run's accepted
file hash. Both runs unloaded the model. See the local
sample report (archive reference: `../../local-results/tutor-smoke-20260916-v2/README.md`) for measured
counts, hashes, source snapshots, and limits. No BiC weights were updated.

The existing v0.6 checkpoint subsequently achieved 42/50 correct actions and
joint action/reply outcomes (84%), with 43/50 exact replies (86%). Valid
selections were 19/25 (76%). Above accounted for seven of the eight action
errors and scored 3/10; this is a useful debugging lead on these fixtures.
Evaluation made no teacher requests and verified unchanged weights. The next
step is a same-scene comparison against existing release phrasing to separate
language variation from visual or action failures. These figures describe an
integration sample, not held-out performance or learning improvement.
