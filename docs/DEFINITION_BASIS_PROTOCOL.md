# Reusable operation grounding and composition

Continue the exact selected learner after the currently running local campaign
has terminated successfully with all work accounted for. This is a separate
research branch; the previous definition probe is not its parent. Preserve the
model, full optimizer, learning rate, auxiliary objective and strict runtime.

The fixed training schedule contains 648 updates: 216 basis lessons, 108 prior
definition lessons and 324 broad replay lessons. Every lesson contains 32
episodes from each of color, count and switch. Each six-update group is basis,
broad, prior definition, broad, basis, broad. The 108 new basis bundles are
consumed twice; the prior definition catalogue once; the existing broad replay
catalogue three times. Repeated examples are practice, and are counted as such.

Basis lessons balance copy, advance-source and advance-destination meanings,
two-word selection and revisions. Training contains individual basis operations.
The earlier copy/copy-then-advance-destination lessons remain rehearsal.
Training never contains the reserved source/copy orders or three-instruction
sequences. Test words differ from training words. The curriculum's independent
typed executor and visible-English interpreter must agree before admission.
Only visible observations enter the learner; metadata and state labels are
supervision and measurement, not inference inputs.

Measure the unchanged parent and fixed steps 216, 432 and 648. At each trained
endpoint save and exactly restore the full learner before testing. Test four
basis panels, all three prior definition panels and all five original broad
banks. The primary definition metric requires correct native actions and
freely generated replies for every question in both counterfactual episodes.
An anchor question alone is diagnostic. Report all endpoints; do not choose the
best checkpoint after inspecting scores.

The final composition milestone requires at least 75% complete pairs in each
family on both two-instruction composition and three-instruction sequence
panels, with a positive improvement over the parent in each. Binding and
revision must also reach at least 75% per family. Retention requires no decline
greater than five percentage points in each original bank/family's joint pair,
known-action, known-reply, unknown-action and unknown-reply rates. Report prior
definition retention separately; its complete-pair rates may not decline more
than five percentage points in any panel/family. Evaluate these requirements
together; improved averages cannot offset a failing family.

Use a fixed 900-second local run allowance, including loading and evaluation.
No score-based extension, automatic retry, checkpoint adoption or external
teacher calls occur in this experiment. New CPU preparation and all prior
failed or rejected work retain their actual costs. No new GPU worker starts
until the current campaign has ended. A partially completed run is reported
with its committed work and any uncertainty, never silently restarted.

This tests a narrow reusable instruction-learning mechanism with bounded
worlds, twelve turns and four reply forms. It does not establish conversational
English, general intellect or learned self-direction. The two-instruction
hypothesis was already observed in the earlier probe; new examples are
development transfer, not a pristine new hypothesis. The reserved sequence
panel is additional evidence, not a general-intelligence benchmark.
