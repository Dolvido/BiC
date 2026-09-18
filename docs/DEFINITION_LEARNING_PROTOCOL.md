# Learning an instruction from its English definition

This next experiment tests a reusable learning operation: read the meaning of
an unfamiliar verb, apply it to different entities, and update its meaning after
a revision. The same operation is exercised across color, count and switch
worlds. It does not establish broad English comprehension or learned curriculum
selection. The existing recurring tutor campaign remains a separate experiment.

The student sees only the English utterances. Typed events, state labels,
answers, family names and curriculum metadata are supervision or verification
data. A visible-English interpreter and a separately implemented typed executor
must agree before a lesson is admitted. A definition changes a verb binding;
it does not change any entity's state. A revision changes future applications
only. Missing entity information remains unknown.

Nonce verbs are distinct from entity aliases. Word-to-meaning bindings vary
across lessons, and evaluation uses unseen verb vocabulary and new entity/value
bindings. Training covers copy and copy-then-advance-destination. A composition
panel withholds advance-source-then-copy and copy-then-advance-source from
training. Questions about both source and destination distinguish their effects.
Counterfactual pairs change a definition while retaining the application and
question; at least one known answer must change. Revision transfer uses unseen
words and bindings, rather than claiming that revision itself was untrained.

The first candidate retains the exact current model, optimizer, learning rate,
objective and auxiliary weight. New rows have an explicit definition-curriculum
version and their own admission evidence. They use the public observation
packer and prepared-owner interface, with independently checked causal state
targets in the existing twelve-alias, 107-class representation. No original
checkpoint identity or historical curriculum admission is rewritten.

The planned first run has 216 updates: 108 definition bundles alternating with
108 existing broad-practice bundles. Every update includes 32 episodes from
each of the three families, for 20,736 total training exposures. Binding and
revision lessons are balanced. The starting checkpoint and fixed local runtime
allowance will be pinned before any model execution; no score-dependent
extension or checkpoint selection occurs within this run.

Three separate evaluation panels contain 32 counterfactual pairs per family:
unseen word/binding transfer, definition revision, and withheld compositions.
Together they contain 576 episodes. Existing retention and transfer banks also
remain visible. Baseline and fixed endpoints report native actions, freely
generated replies, both-member pair correctness, source/destination effects,
revision effects and unknown-information responses. A saved and reloaded
endpoint is evaluated with no tutor calls or interpreter outputs in inference.

A useful first milestone is at least 75% paired action-and-free-reply correctness
in every family on composition and revision panels, improvement over the
unchanged parent, and no greater than five percentage points of loss on the
declared retention slices. All results are reported if this milestone fails.
This is a research criterion, not automatic release promotion.

Before the run, focused hand-computed cases must establish definition scope,
revision timing, source/destination order, unknown propagation and counterfactual
truth. A small public-interface compatibility check must establish that the new
admission provider works without changing the existing training implementation.
Compute, unique/repeated lessons and retention costs are recorded. No additional
GPU job starts while the recurring tutor campaign owns the GPU. A terminal,
fully accounted pre-training compilation refusal can supply its unchanged
committed parent while the campaign recovery is prepared; partial or live
learner work cannot. The definition experiment remains a separate research
branch and does not replace the campaign parent automatically.

Current limits remain twelve turns, twelve entity aliases, 128 bytes per
utterance, 1,024 total context tokens, bounded typed values and four reply forms.
The tutor can eventually propose verified definitions and lesson sequences;
this first experiment establishes whether BiC can use those definitions itself.
