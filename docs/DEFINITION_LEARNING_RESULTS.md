# First native English-definition acquisition experiment

BiC learned to use unseen verb words for taught meanings and to change their
meaning after an English revision. It did not solve the withheld operation
combinations, and two older retention slices exceeded the allowed loss. The
candidate is preserved for research and was not automatically adopted.

The unchanged 2.26-million-parameter learner continued from update 7,816 to 8,032.
It received 216 updates:108 definition bundles alternating with 108 broad-practice
bundles,20,736 total episode exposures. The run took 115.250 seconds and completed
three exact full-state restores and two snapshots, with no teacher calls or
unknown optimizer outcomes. Both fixed endpoints were evaluated after restart.
The time endpoint, rather than the best intermediate score, is reported below.

Here a complete pair requires correct native actions and freely generated
replies on **every question in both counterfactual episodes**:

| Unseen-word evaluation panel | Before | At 108 updates | At 216 updates |
|---|---:|---:|---:|
| Apply a taught meaning | 0/96 | 94/96 | 95/96 |
| Revise a word's meaning | 2/96 | 93/96 | 96/96 |
| Withheld operation combinations | 0/96 | 0/96 | 0/96 |

The final application results by family were 31/32 color,32/32 count and 32/32
switch. Revision was 32/32 in each family. Combinations were 0/32 in every family.
The less demanding focal-question pair metric improved from 1/96 to 11/96 on
combinations; its intermediate score was 22/96. Neither metric meets the stated
75% composition milestone. The protocol did not uniquely name which paired
metric governed that milestone; both are therefore reported explicitly.

Old native development pairs improved 257/324 to 262/324. Overall retention fell
286/360 to 275/360, while color retention fell 110/120 to 103/120 and count fell
68/120 to 60/120: losses of 5.83 and 6.67 percentage points. Switch improved 108/120
to 112/120. Original-position transfer changed 136/216 to 132/216, and varied
positions changed 125/216 to 134/216. The new skill does not excuse the declared
retention failures.

Training defined only copy and copy-then-advance-destination. Both withheld
combinations advance the source, an effect absent from training. The zero
complete-pair score therefore cannot distinguish failure to learn a new
primitive from failure to order already grounded primitives. Every source-query
pair failed in that panel. The next general curriculum should teach the shared
primitive basis and argument roles before testing held-out combinations; it
should not patch a single subject or train on these evaluation answers.

The curriculum has a new explicit admission schema, separate typed and visible
English interpreters, independently checked causal prefix labels, and disjoint
training/evaluation verb vocabulary. It generated 10,368 definition episodes
containing 10,358 distinct transcripts;10 repeated rows were counted as practice.
There was zero exact train/evaluation transcript overlap. Its preparation took
7.890 seconds, including all generation and validation work. The public packing
check admitted all three subjects through the unchanged learner interface and
verified that only visible text enters neural inputs.

The result remains restricted to twelve turns, twelve aliases, bounded color,
count and switch worlds, a finite definition language, and four reply forms.
Each episode defines only one verb, so this result also permits a strategy
that reads the latest definition while ignoring the verb identity. It demonstrates
conditional use and revision of taught meanings, not discrimination among
multiple simultaneously defined verbs, broad English comprehension, general
program induction or learned curriculum selection. Further training must preserve these skills alongside
older abilities.

The run is `runs/definition-learning-local/attempt-001`. Launch SHA256:
`56a576b13ebc7a3a451e6a7dc7aeda11786054da0589342de5ab4e08e98635f8`.
Terminal summary SHA256:
`d13be1cef5f50609e1079495dcca498e33286c93103b9901b2bf5524427e1b90`.
An independent raw-record recount verified all 24 endpoints, 54 raw score files,
8,748 episodes, 41,646 question predictions and 216 step reports. It confirmed
both milestone interpretations fail and the two retention losses above. The
report is `runs/definition-learning-analysis-local/attempt-001/independent-analysis.json`,
SHA256 `a1cecc851f4881480b7de676d1dce20b0c99cbd7a7ad7ba2335b774ad440f9c8`.
The pure recount took 1.516 seconds, with no model calls or archive decoding.
