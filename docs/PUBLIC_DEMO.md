# Run the native English learner

The repository includes a **9.09 MB inference artifact** for the lower-learning-rate
research candidate. It runs on a CPU and needs no tutor, internet connection,
research archive, optimizer state, or GPU. Dependencies must already be installed
to run offline. This is a bounded English experiment, not a general chatbot.

From the repository root, after installing the project's dependencies:

```console
python -m examples.english_demo
python -m examples.english_demo --interactive
python -m examples.english_demo --json
python -m unittest discover -s tests -p test_public_demo.py -v
```

The first command prints four fixed scenarios. Interactive mode accepts one
sentence per turn; `/reset` starts a new empty world and `/quit` exits. Start with
the exact English forms printed by the examples. The trained aliases include
`dax`, `wug`, `fep`, `zot`, `blick`, `toma`, `kiv`, `nup`, `ral`, `sog`, `vex`, and
`pim`. Color advancement follows red → green → blue → yellow → red.

The limits are **12 turns**, **128 UTF-8 bytes per utterance**, and **1,024 total
observation tokens**, including two boundary tokens per turn. Whichever bound
is reached first applies. Oversized input is rejected without changing history;
nothing is silently truncated. History contains only the user's observations,
not previous generated replies. Resetting history does not train or change weights.
Free-form input is accepted within these size limits, but competence outside the
finite training grammar is unestablished.

## What the output means

Each turn shows a native four-way action and a separately generated reply. The
action labels are `DENY`, `ALLOW`, `ASK`, and `ACK`; their conventional replies
are “No.”, “Yes.”, “I need more information.”, and “Understood.” The demo does
**not** turn an action into a canned reply: the recurrent byte decoder generates
its own text, beginning with only a start token. Action and reply can disagree.
Unterminated or invalid UTF-8 outputs are explicitly flagged.

Only UTF-8 observation bytes and their boundaries enter the neural encoder.
No expected answer, parsed world state, curriculum family label, oracle output,
or LLM response is supplied. The training-only auxiliary state head is not called.

## Observed successes and failures

The four scenarios were written before their first model run and were kept
unchanged afterward. These small illustrations are **not a benchmark**, a random
sample, or evidence of general English understanding. All 27 turns, including
generated token IDs, are retained in
[demo-observations.json](../assets/demo-observations.json).

The first CPU run produced the following query results. “Expected” is a human
reading of the displayed English, documented here after inference; these answers
are not input to the demo.

| Scenario | Query | Native reply | Expected |
| --- | --- | --- | --- |
| Colors | Is dax red? (before revision) | Understood. | Yes. |
| Colors | Is dax red? (after revision to green) | No. | No. |
| Colors | Is dax green? | I need more information. | Yes. |
| Count | Is the count of dax 5? (after adding 2 to 3) | I need more information. | Yes. |
| Count | Is the count of dax 5? (after subtracting 1) | No. | No. |
| Count | Is the count of dax 4? | I need more information. | Yes. |
| Defined copy | Is the color of wug red? | Yes. | Yes. |
| Defined copy | Is the color of wug blue? | No. | No. |
| Defined copy | Is the color of dax red? | Yes. | Yes. |
| Composition | Is the color of dax green? | No. | Yes. |
| Composition | Is the color of wug green? | No. | Yes. |
| Composition | Is the color of wug red? | No. | No. |

All statement turns produced `ACK` / “Understood.” Actions agreed with generated
replies on this run. The copy scenario succeeded; the simpler color and count
histories still exposed errors. Familiar vocabulary alone does not ensure
correct performance on a different arrangement of turns. The composition failure
is consistent with the much larger formal study's weak composition result.
Numerical predictions can vary across software versions or hardware; the recorded
run used Python 3.12.14, PyTorch 2.11.0+cu128 on **CPU**, and two CPU threads.

## Artifact and validation

[bic-lower-2592.json](../assets/bic-lower-2592.json) records the artifact size,
hash, architecture, configuration, and original checkpoint provenance. The
2,264,725-parameter model is the final lower-rate branch at 18,184 lifetime
updates, including 2,592 updates in the rate study. It **failed the complete
promotion gate** and is exported for inspection, not adopted as the retained
home-learning checkpoint. See [the study results](SHARED_RATE_RESULTS.md).

| Identity | SHA-256 |
| --- | --- |
| Published inference file | `1cb5d827e16c26c2b67cdb90a9cc7a61b10dfc05e05dbec38fbc31f7688d5986` |
| Native parameter digest | `002a5cdd39c1ad1e4625f2a7e804eb1d1a13ba62598db90ff0bc3354467d758b` |
| Original full research archive | `ad39a15737834e20203209261cd58dc9f70773a849914fc463ab705bcaf70637` |

Loading verifies both the file hash and the parameter digest. The export contains
model tensors and inference configuration only, including the unused auxiliary
head to preserve the original parameter identity. It does not include an
optimizer, full training archive, tutor model, or resume state. The model's
in-context behavior does not update its parameters and cannot resume training
exactly from this asset.

The release's nine demo tests passed locally. They check the original weight
identity, CPU placement, absence of training state, no oracle or auxiliary-head
call, unchanged parameters, repeatable resets, all three input limits, and
rejection of a changed artifact before deserialization. These checks establish
that the demonstration is isolated and usable; they do not establish new learning
or comprehension gains.

Maintainers with the original archive can reproduce the extraction with
`python -m examples.export_demo_weights PATH_TO_LOWER_2592.pt OUTPUT.pt`.
The exporter accepts only the recorded original archive hash. PyTorch container
bytes can depend on the serialization environment; the native parameter digest
must remain identical.
