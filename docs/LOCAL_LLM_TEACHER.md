# Could a local LLM teach BiC?

**Yes: a local LLM is a feasible lesson writer for BiC.** The most useful first
experiment is broader, verified language for tasks BiC can already experience.
It could make teaching less repetitive and expose gaps in the curriculum.
Whether BiC learns more from those lessons remains an experimental question.

This is a feasibility study, checked against primary sources on **2026-09-16**.
No teacher model was downloaded or run, and no LLM-generated lessons were used
for this release. Recommendations and resource budgets below are proposals,
not measurements on Luke's RTX 5080.

## What it could teach

| Lesson | Example | Feasibility with the present student |
|---|---|---|
| Alternative wording | “Choose the item immediately left of \"dax\".” | Strong first candidate: an existing relation expressed differently. New phrasing still needs meaning checks. |
| Short grounded replies | A brief acknowledgment of the cell actually selected | Feasible within the byte decoder's limits; accuracy must include the action, not just a plausible reply. |
| Clarification and abstention | An unknown name, two matching objects, or an impossible neighbor | Feasible with independently labeled examples and balanced successful selections. Always refusing must not score well. |
| Curriculum suggestions | Practice “above” after repeated above/below mistakes | Feasible as recommendations based on development results. A scheduler should enforce coverage and old-task replay. |
| Demonstrations and corrections | Propose a sequence that focuses the textbox and types `hi` | Feasible only inside the miniature desktop's existing action space, after the simulator verifies every step. |
| Longer dialogue and multistep plans | “Now select the one next to the object we discussed earlier.” | Requires additional dialogue state, observations, tasks, and evaluation; more text alone does not supply them. |
| General knowledge or unrestricted English | Explain an unfamiliar scientific topic or converse freely | Beyond demonstrated BiC capacity. A useful teacher does not establish that this small student can absorb the teacher's broad competence. |

The current regional-memory task has five relations on four already separated
object crops. New names are exact addresses in an exemplar bank, normalized to
`<name>` before instruction encoding. Teaching a name stores examples; it does
not teach that word to the language weights. These boundaries are described in
[REGIONAL_MEMORY.md](REGIONAL_MEMORY.md).

A text-only teacher cannot see pixels without a separate vision component. For
the first pilot it should receive a simulator description used only for making
lessons. BiC must continue receiving its declared sensory observations and
memory evidence. Correct cell indices and hidden glyph identities stay in the
teacher/verifier, outside the student's inference inputs.

## Why this is plausible, and what the evidence does not establish

[Self-Instruct](https://arxiv.org/abs/2212.10560) demonstrated that generating and
filtering instruction examples can improve an already pretrained language
model. [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
demonstrated learning from a teacher's output sequences in machine translation.
These support the proposed teaching method; neither tests BiC's architecture or
shows that a roughly half-million-parameter student can reproduce an LLM.

BiC uses a **259-symbol UTF-8 byte vocabulary**, while a typical teacher predicts
subword tokens. Their output probability vectors have different meanings and
cannot simply be matched position by position. Start with accepted text
sequences: decode the teacher's final answer, validate it, then encode it using
BiC's existing `ByteCodec`. Use the usual byte prediction loss plus independently
verified action targets. Specialized cross-tokenizer distillation is additional
research, unnecessary for the first pilot.

The desktop model currently allows **128 input bytes and 64 output bytes**;
regional response generation normally uses a shorter budget. A teacher's large
context window does not expand those limits. Reject oversized lessons instead
of truncating their meaning. Do not train on lengthy reasoning traces: the
student needs the instruction, observable experience, correct behavior, and a
short final reply.

Biologically, the useful analogy is a tutor arranging experiences while the
learner forms associations and rehearses skills. The external LLM would remain
the tutor. BiC would still contain the networks that learn and act. Offline
weight updates are slower learning; exemplar storage is rapid association.
Demonstrating transfer into durable weights requires later tests with the
relevant exemplar memory removed. Merely asking the LLM during inference would
measure the combined system and would not demonstrate BiC learning independently.

## A practical local teacher

Start with a **4–8B parameter instruction model quantized to roughly four bits
per weight**. Two concrete, mature candidates are
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) and
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). Their publisher cards list
Apache-2.0 licensing and a non-thinking mode, which suits short lesson records.
Use 4B for the inexpensive pipeline check; compare 8B on accepted lesson quality
before paying for more generation. This is a starting recommendation, not a
claim that these are the newest or best models available.

Pin the chosen model revision, quantization file hash, runtime version, and
generation settings in the lesson manifest. Include the publisher's license
and notices with any redistributed weights. The model-card license alone does
not establish ownership of every generated sentence; keep lessons original and
grounded in our own simulator rather than asking for copied books or datasets.

[Ollama supports native Windows](https://docs.ollama.com/windows), and its
[hardware documentation explicitly lists the RTX 5080](https://docs.ollama.com/gpu).
It offers a straightforward local service for the Windows 11 desktop. Keep the
teacher and lesson-generation client in the same environment initially; adding
Windows/WSL networking is unnecessary for the feasibility test. BiC training
can consume the saved lesson file later in either environment.

[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
accept a JSON schema. An alternative is
[llama.cpp's JSON-schema/grammar support](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
for more explicit deployment control. llama.cpp supports only a subset of JSON
Schema, so independently validate every returned record. Neither tool's
structured format certifies that “left” was preserved in a paraphrase.

### Memory and time budget

The following arithmetic is for **nominal model classes**, not actual GGUF file
sizes or measured VRAM use:

| Nominal parameters | Four-bit weights alone, decimal GB | Same weights, GiB |
|---|---:|---:|
| 4 billion | 2.0 | 1.86 |
| 8 billion | 4.0 | 3.73 |
| 14 billion | 7.0 | 6.52 |

The estimate is `parameters × 4 / 8` bytes. Quantization metadata, some tensors
stored at higher precision, the conversation's key/value cache, temporary
buffers, and the display consume additional memory. Qwen3-8B's published count
is 8.2B, so its idealized four-bit weights would be 4.1 GB before that overhead.
Full-precision training of the teacher is a different, much larger workload;
it is not part of this proposal.

On the 16 GB GPU / 32 GB RAM desktop, a quantized 4–8B teacher is a reasonable
inference candidate. A 14B teacher may also be practical at short context, but
there is no measured benefit that justifies starting there. Use one generation
request at a time, a 4,096-token context, and at most 256 generated tokens per
lesson. Treat **12 GiB GPU memory and 24 GiB system RAM** as initial operating
budgets, leaving headroom; measure and reduce the model/context if needed.

Generate and save lessons first, unload the teacher, then train BiC. This avoids
having two models compete for the same GPU. Measure a 50-lesson sample before
estimating throughput. No tokens-per-second or completion-time claim is made
here. The earlier 175.7M-parameter BiC benchmark used a tiny workload and says
nothing about teacher throughput or the desktop's maximum trainable model.

## How to make trustworthy lessons

1. **Choose the meaning first.** The simulator selects an existing goal and
   computes its answer. Ask the LLM for alternate wording of that meaning.
   Initially keep response targets canonical so any gain measures comprehension.
2. **Validate structure and limits.** Require a lesson ID, instruction, proposed
   relation, and provenance. Check allowed relations, UTF-8 byte limits, and
   exactly one properly quoted name for regional-memory commands.
3. **Check meaning separately.** A deterministic parser can accept approved
   wording families. Human review must approve each genuinely new family before
   it becomes training data. An LLM agreeing with itself is insufficient. Mark
   uncertain phrases as rejected or awaiting review.
4. **Verify behavior in the simulator.** Recompute labels and execute proposed
   trajectories using existing allowed actions. Reject unsupported steps,
   unsuccessful outcomes, or contradictory replies. Trying an action verifies
   its outcome; it does not by itself prove a new sentence meant that action.
5. **Freeze accepted data and splits.** Keep generator settings, decisions,
   rejection reasons, and hashes. Separate phrasing families and identities,
   rather than randomly splitting near-duplicate lines. Keep the final test
   unavailable to the teacher and model-selection process.
6. **Train with replay and test without the teacher.** Start both comparison arms
   from identical BiC weights and preserve older skills in both. Evaluate the
   same independent tasks with the LLM process stopped and no network dependency.

Existing implementation points are
[`TEMPLATES`, `build_specs`, and `relation_target`](../experiments/train_regional_memory.py),
[`normalize_instruction` and `RegionalMemoryAgent.act`](../brain_in_computer/regional_memory.py),
and the desktop's
[`oracle_action` and `goal_success`](../brain_in_computer/computer_use/environment.py).
These are useful interfaces for a future importer; none currently imports LLM
lessons. The LLM should supply data rather than executable Python or repository
edits.

## Proposed first pilot — not run

Aim for **1,000 accepted lessons** across the five existing relations, successful
selections, and the four reasons for stopping. Allocate 800 to training and 200
to development by whole phrasing family. Start with short instructions and fixed
replies. Cap generation at 2,000 candidates or one hour; if the target cannot be
met, report acceptance rate and stop rather than relaxing meaning checks.

Compare an equal-size expanded procedural curriculum with the accepted
LLM-written curriculum. Use the same starting checkpoint, optimizer-update
budget, action distribution, and old-task replay in both, across three training
seeds. Proposed cap: 20 minutes per training run and 30 minutes for evaluation;
these are budget limits, not predicted runtimes. Record processed bytes and wall
time as well as update count, since phrase length changes compute cost.

Reserve at least 500 independent evaluation episodes using newly reviewed
phrasing families and new object identities, plus the unchanged old-task suite.
Report action accuracy, valid selection success, each STOP cause, exact reply
accuracy, old-task retention, and accepted lessons per minute. Score canonical
replies first; semantic grading of varied conversation is a later experiment.

Before running, freeze a promotion rule: seek at least a five-percentage-point
gain on the deliberately harder new-phrasing test, with no more than a
two-point drop in old-task joint accuracy. Report per-seed results and uncertainty;
a small pilot can be inconclusive. Do not expand the teacher model simply
because the student fails: first check mislabeled wording, representational
limits, and interference.

Keep a future integration small: one lesson schema, one optional generator, one
validator/importer, and one experiment report. Teacher weights belong outside
the repository, and the normal BiC interface should launch without Ollama. The
first success criterion is a measurable improvement in BiC after the tutor is
removed.
