# Persistent visual teaching

This document describes the v0.4 visual matcher mode, still available in v0.5
by omitting `--regional-checkpoint`. The optional v0.5 integration is described
in [REGIONAL_MEMORY.md](REGIONAL_MEMORY.md).

The teaching lab combines a trained image encoder with a bounded exemplar bank. It is **a separate experiment** from the computer-use model: retrieval is not yet routed through BiC's 13-region core, and its names are not learned by the English comprehension network.

## Use the lab

```bash
python -m brain_in_computer memory-serve --checkpoint runs/associative/encoder.pt --memory runs/personal-memory.json --port 8766 --threads 1
```

Open [localhost:8766](http://127.0.0.1:8766). Select one of four glyph objects, enter a name, and press **Teach**. **Change scene** retains identities while changing location, color, size, and alignment. Find the saved name or ask what the selected object is. Teach another view with the same name to add an exemplar.

Stop with Ctrl+C and restart the same command to reload the bank. The companion `.scene.json` file preserves which objects the environment renders, independently of stored associations. **Generate new objects** replaces those objects without deleting names; absent objects should usually produce no match. The × beside a saved name explicitly forgets it.

Command conversation recognizes `remember dax`, `find dax`, and `what is this?`. The command parser and replies are templates. Names are case-sensitive exact labels: `dax` and `Dax` are different addresses. The visual match is neural; language parsing is not learned in this lab. This differs from the computer lab's byte-generated replies.

## What is learned and stored

The encoder was trained on two independently varied views of each synthetic 4 × 4 occupancy glyph, using contrastive learning plus a training-only occupancy target. Inference receives RGB crops, never occupancy masks or object identifiers. Object separation is supplied by the environment; automatic detection in general images is not implemented.

Teaching saves a normalized visual embedding under the selected label. Default capacity is 128 labels with up to four recent exemplars per label. Adding a fifth replaces that label's oldest exemplar; a full label bank rejects new labels until a label is explicitly forgotten. Reported accuracy tests 32-label naming and 48-label retention, not full-capacity behavior.

Find compares candidate embeddings with the named exemplars; naming compares one crop with all saved labels. Acceptance requires a strong best cosine similarity, sufficient separation, and no second above-threshold match. Otherwise the system returns unknown or ambiguous. Cosine similarity is **not a calibrated probability**; thresholds were fitted only on validation glyphs. More exemplars or competing labels can alter error rates.

JSON persistence stores embeddings, thresholds, capacities, and a hash of exact encoder configuration/weights. Loading with another encoder is rejected. Changing the encoder requires an explicit migration/re-encoding design; stale embeddings are not silently reused. No optimizer updates occur during teaching or retrieval.

The release's real HTTP restart trial recovered **4/4 taught names** in a second server process after all four object images and their order changed. No reteaching occurred, and encoder/memory files stayed byte-identical. This complements the larger offline retrieval test; it is not a browser rendering check. [Restart verification](../experiments/teaching_ui_verification.json), recorded scenes (archive reference: `../runs/associative/restart_demo.png`).

## Biological motivation and next integration

Rapid association alongside slower representation learning is inspired by complementary learning systems. Research models examine how memory transfer can help or harm generalization; storing an exemplar alone is not consolidation. [Sun et al., 2023](https://www.nature.com/articles/s41593-023-01382-9)

BiC currently freezes perception to preserve existing embeddings. A future integration should pass retrieved evidence through the regional pathways, test unknown-object handling and interference, and compare replay-based consolidation against frozen retrieval and no-memory controls. [Measured results](V04_RESULTS.md) distinguish these proposed mechanisms from capabilities already tested.
