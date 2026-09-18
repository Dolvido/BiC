# First complete local external-tutor learning loop

This integration run takes priority over the separate lesson-layout comparison.
Its purpose is to join real local tutor selection, current raw-English learner
updates, unaided evaluation and optimizer continuation across fresh processes.
It does not claim that the tutor improves learning relative to a control, that
BiC has learned its own scheduling policy, or that withdrawal alone establishes
independence.

## Fixed first run

Use one fresh width-192, four-layer, four-head sequence learner with feedforward
width 768, 1,024 context positions, twelve-turn capacity, 128 input bytes and 32
output bytes. Keep the unchanged three-family sequence objective, AdamW learning
rate 0.003 and clipping at 1. Use the existing strict local FP32 RTX 5080 profile
with one CPU intra-op and one inter-op thread. All lessons use the original
causally verified layout and independent truth checks.

Run three fresh curriculum cycles of 216 updates each: six 24-update stages,
earlier-depth rehearsal every fourth update, then a 72-update mixed finish.
Each update uses 32 episodes from each of color, count and switch. Total actual
training is 648 updates and 62,208 episode exposures. Every cycle includes all
six depths and lengths 8/10/12. Plans use seeds 852100001, 852100002, 852100003
and ordering seeds 852101001, 852101002, 852101003.
The learner initialization seed is 852104001. Development seeds are
852102001, 852102002, 852102003; the final audit seed is 852103001.

1. **Initial learning:** use the fixed curriculum order and save a full learner
   and optimizer checkpoint after 216 updates.
2. **Tutor-selected learning:** supply only aggregate development counts from
   the first cycle and the initial reference to local `ministral-3:3b`. Make one
   persisted chat request selecting curriculum or mixed order from the same
   balanced lesson inventory. Continue the same weights and full AdamW state in
   a new process, to lifetime update 432.
3. **Withdrawal:** record a caller-selected procedural decision with zero tutor
   calls. Restart from the second checkpoint and learn the third fresh curriculum
   to lifetime update 648, then evaluate without teacher assistance.

The pinned local teacher digest is
`f04aa1c738f64e13c625b82ae92504fc0260fa6723b509ed1ece0fa188179b1d`,
freshly checked in the installed local inventory on September 17. Verify it
again before and after the actual generation. Request `keep_alive=0`, and keep
teacher and learner GPU phases sequential. A rejected known response may use
the recorded procedural fallback, but that is not a successful live tutor
selection. An uncertain interrupted request must not be silently repeated.

## Evidence

Admit training and evaluation transcripts before learning against the explicit
historical inventory and across all three cycles. Names-only repairs preserve
the planned problems and answers. Store compact plans and expected evidence;
do not store a redundant complete training corpus. Constant development and
each new cycle's development bank have four counterfactual pairs per cell,
including fresh realizations and held compositions. The final audit also uses
four pairs per cell. The constant bank doubles as cycle zero's development bank.

Evaluate the constant development bank before and after initial learning. For
each later cycle, evaluate its new development bank before and after training,
and the constant bank after training. Reuse the preceding constant-bank counts
only after the restored parameter identity is verified. Evaluate the audit once
at the end. All scoring uses native actions and freely generated replies, with
explicit anchors and no tutor calls or answers in the policy inputs.

Report per-family known counterfactual actions and exact replies, other known
questions, unknown answers, unsupported asking, action/reply agreement, changes
on the new-cycle development bank and retention on the constant bank. Preserve
the raw outputs and count denominators. Weak or negative learning results remain
visible. No checkpoint is automatically promoted.

The data-preparation ceiling is 1,800 seconds. The separate complete training
orchestration ceiling is 3,600 seconds, including worker startup, actual tutor
time, preparation of scoring tensors, training, checkpoint publication and
evaluation. Record all completed, rejected and uncertain work. Deadlines are
checked between operations and do not erase the cost of an operation already
started. Prior failed comparison preparation remains a separate recorded cost.

Before launch, bind the actual source versions, seeds, data, runtime and passed
continuation/tutor checks. This document records a prospective integration run;
its existence is not evidence that any learner updates or live tutor calls have
occurred.
