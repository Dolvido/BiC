# Learning associations in shared neural weights

BiC v0.6 includes a small, separate neural recall network that learns visual
associations from episodic examples. After training, it can supply the existing
regional controller with visual evidence without using the original examples.
The frozen two-seed experiment achieved **99.34% correct actions** on both the
first 16 names and the next 16 names after rehearsal. These are taught object
identities viewed under new rendering seeds, not unseen object concepts.

This is a research experiment and Python API. The live teaching interface still
uses rapid exemplar memory; pressing Teach does not silently retrain this network.

## What changed

The earlier memory bank stored up to four visual embeddings under each name.
The new network instead reads the UTF-8 bytes of a name, encodes them through a
shared GRU, and predicts a normalized visual embedding through a shared MLP.
Training minimizes cosine distance to the frozen visual embeddings in the bank.
An optimizer changes the same **56,824 parameters** for all names.

At inference, the predicted embedding is compared with the four candidate crops.
All four similarities go into the existing recall adapter and regional network;
motor cortex chooses an action and the existing byte decoder generates the reply.
The name encoder does not receive the desired location, relationship, object ID,
or action. The regional controller and visual encoder remain frozen throughout
this experiment, using the delivered v0.5 checkpoint for comparability.

A set of known names remains as an explicit gate for untaught names. It contains
strings only. There is no visual vector per name, per-name output weight table,
object ID, or exemplar image in the student checkpoint. This demonstrates neural
learning of a small set of associations; it does not demonstrate general English
word learning or open vocabulary recognition.

The biological inspiration is complementary learning: rapid association supplies
experiences for slower parameter updates, and interleaved rehearsal protects old
associations. This transfers knowledge into an additional shared recall network.
It does not transfer associations into the existing cortical-region weights,
simulate sleep, or establish biological equivalence. The resulting evidence still
passes through the existing hippocampal pathway.

## Frozen experiment and results

Two independent seeds each taught 32 arbitrary names, paired with 32 glyphs
excluded from the earlier visual and regional training identities. Each name had
four differently rendered teaching examples. The first 16 associations trained
for 1,000 updates. Two copies then trained for another 1,000 updates on the next
16: one with old-example rehearsal and one without it. The compared copies see
the same new-example sequence; rehearsal adds old examples and extra compute.

All stages used their final update. No development selection or tuning on the
sealed results occurred. Held-out rendering seeds changed appearances while
retaining the familiar glyph-generation families and relationship grammar.
Each seed has 1,280 first-group and 1,280 second-group cases. Half require a valid
selection; half require stopping because of an absent object, an unknown name,
ambiguous duplicates, or a board boundary.

| Recall source | First 16: action accuracy | Next 16: action accuracy |
| --- | ---: | ---: |
| Original four-exemplar bank | 99.18% | 98.98% |
| Shared weights, with rehearsal | **99.34%** | **99.34%** |
| Shared weights, without rehearsal | 57.03% | 99.34% |
| Untrained shared weights | 50.00% | 50.00% |
| Erased shared weights | 50.00% | 50.00% |

These are equal-weight averages across two seeds. The rehearsal condition's
action-plus-exact-reply success is **98.71%** in each group. On cases requiring a
real selection, old-name accuracy is **99.22% with rehearsal versus 15.94% without
it**. Untrained and erased controls select none correctly; their overall 50%
comes from stopping on the half of cases where stopping is correct.

Two fresh OS processes also selected **64/64** fixed named objects after the
source exemplar contents were explicitly erased. Their inputs were rendered
candidate pixels and quoted instructions. Episodic-memory construction, lookup,
and teaching were patched to raise errors in those processes. Student checkpoint
hashes remained unchanged. This is persistence of learned weights, without
reteaching in the fresh processes.

The two-seed optimizer workload took about **15 seconds with one CPU thread** on
the execution host. Rendering, evaluation, process startup, and model loading are
additional costs. This is not an RTX 5080 benchmark or a measure of the maximum
number of associations that fit on a home computer.

## Running it

From the repository root, inspect the delivered evidence:

```bash
python experiments/check_consolidation_restart.py
python experiments/check_consolidation_restart.py --parity-only
```

To repeat training in a new directory, then run its sealed evaluation once:

```bash
python experiments/consolidate_memory.py train --output runs/my-consolidation
python experiments/consolidate_memory.py evaluate --output runs/my-consolidation
```

Training refuses to overwrite a frozen protocol. Evaluation refuses to overwrite
existing sealed results and checks the current runner, module, encoder, regional
checkpoint, and student hashes against the experiment's recorded dependencies.
Default training is deliberately small: one CPU thread, two seeds, 1,000 updates
per stage, and 32 names per seed. This experiment runner currently trains on CPU.

The reusable API is in `brain_in_computer/consolidation.py`:

```python
from brain_in_computer.consolidation import ConsolidatedMemory
from brain_in_computer.regional_memory import load_regional_checkpoint

agent = load_regional_checkpoint("runs/regional-memory-v05/checkpoint.pt")
memory = ConsolidatedMemory.load(
    "runs/consolidation-v06/seed-6101/with_replay.pt", agent.encoder
)

# crops: four RGB tensors of shape [3, 32, 32], values in [0, 1],
# corresponding to top-left, top-right, bottom-left, bottom-right.
# The delivered student recognizes only its particular taught glyphs and names.
result = memory.act(agent, 'click the object left of "dax"', crops)
print(result["action"], result["reply"])
```

To consolidate other objects, acquire them with `AssociativeMemory.teach`, export
training targets with `replay_examples`, optimize a `NeuralRecall`, and construct
a `ConsolidatedMemory` with the taught-name list. Its checkpoint saves the learned
weights, names, configuration, and frozen encoder identity. The experiment's
`fit` function provides the small reference training loop. New labels require
learning updates; adding a string to the known-name set does not teach a concept.

## Size and limits

| Representation of 32 names | Floating-point values | Float32 payload |
| --- | ---: | ---: |
| Four 64-dimensional exemplars per name | 8,192 | 32 KiB |
| Shared neural recall parameters | 56,824 | 222 KiB |

The learned network is larger than the raw exemplar vectors in this small case.
This experiment is about a different way to retain and use experience, not a
compression benefit. Checkpoint containers, names, and the shared visual/controller
models add storage beyond these payloads. The complete combined model has 615,291
parameters when the additional recall network is counted once.

The evidence covers 32 fixed associations, two seeds, four predefined object
crops, five familiar relations, and one action per instruction. Appearance families
are familiar even though exact view seeds are reserved. It does not establish
general object detection, many-thousand-name capacity, general English dialogue,
new-task discovery, or resistance to arbitrary interference. Rehearsal reduces
forgetting here; it does not eliminate it universally.

## Evidence and provenance

- `runs/consolidation-v06/protocol.json`: names, identity partitions, seeds,
  losses, fixed update counts, and original source/checkpoint hashes.
- `training_report.json`: optimization times, losses, and selected weight hashes.
- `final_evaluation.json`: original sealed scores and initial restart checks.
- `restart_verification.json`: separate verification with visibly erased source
  contents and unchanged weights.
- `evidence_parity.json`: cached evidence agrees with the public inference API.
- `frozen_runner.txt`: exact original runner source, matching the protocol hash.

The original training process recorded successful source-file deletion, but those
files were present in a later execution session. The original sealed report
honestly records that discrepancy. A separate check explicitly erased their
contents and repeated the same fixed restart cases, yielding 64/64 without
training or changing case selection. Empty source markers remain as evidence.

An independent review also prompted the cached evaluator to derive known-name
availability directly from memory membership. All 25,600 original case/condition
bits equal that membership. Direct public-API evidence checks cover every case
category and old/new group across the five conditions and both seeds; all 100
checks agree within 1e-6. Stronger hash guards were added for future runs. The
original sealed report and trained weights remain unchanged; source provenance
is retained rather than retroactively replacing the frozen protocol.
