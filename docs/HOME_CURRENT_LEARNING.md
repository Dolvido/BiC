# Run the current retained home learner

For the double-click menu and offline continuation instructions, see
[OFFLINE_HOME_HANDOFF.md](OFFLINE_HOME_HANDOFF.md). The current retained checkpoint
is recorded in the session profile; always read that profile for current counts.

This entrypoint resumes the existing verified local tutor campaign. It does not
create another trainer, replace the released desktop controller, or adopt the
newest research checkpoint. After the completed offline setup validation, the
retained research learner is update 9,328 after nine completed cycles. Its tenth
curriculum is already authored and compiled; a resume reuses those artifacts
before making any later request. Later experimental branches remain separate.

The loop uses the installed local teacher to choose bounded, verified lesson
recipes, then trains and evaluates native actions and generated replies without
the teacher. Full weights and AdamW survive restarts. Selection and retention
rules are engineered. Restricted English performance is measured; useful tutor
advantage, arbitrary English teaching and general intelligence remain unproven.

This MSI already has an initialized default session. Use the status and resume
commands below. The following is the historical one-time initialization command,
included for provenance; it is not a command to repeat on the existing profile:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.home_learning init --campaign-directory runs/continuous-tutor-campaign-local/attempt-003 --summary-sha256 76cf485b6f010c0d0c43f5ad17d7397829f6b05bb7896673a96f2adb5fe8ac8e
```

The default profile is `runs/home-learning-local/session.json`. It pins the
completed campaign, launch, committed state, retained checkpoint, frozen source
files, teacher identity and evaluation manifest. Initialization refuses to
overwrite an existing profile.

Read status without loading weights, importing Torch or contacting a service:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.home_learning status
.\.venv\Scripts\python.exe -B -m experiments.home_learning status --json
```

Resume only when other project learning jobs have finished:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.home_learning resume --training-hours 1
```

`--training-hours` is the existing controller's wall-time allowance, from two
minutes through 24 hours. CPU authentication/freeze/setup occurs separately and
is reported separately, so total command time can exceed this allowance. The
controller starts a stage only when its full declared allowance and final
reserve fit. Short invocations may preserve a pending stage without training.
There is no automatic retry or automatic release promotion.

An exclusive project/session lock records the actual process identity. A fresh
Windows process inventory refuses other project Python learners, inaccessible
Python process metadata and inventory failures. The launcher recognizes its own
Windows virtual-environment bootstrap ancestors. It never kills other jobs or
deletes an unexplained lock. This check cannot prevent a separate, manually
started experimental command from launching afterward; do not start concurrent
learner jobs while a home session runs.

Every invocation saves its intent before controller startup, preserves setup and
execution receipts, and advances the profile only after an authenticated,
completed controller result. A failed or interrupted invocation leaves the old
retained learner and an unresolved intent. Status shows that it needs attention;
resume refuses to repeat it. Inspect its receipts before an explicit recovery.
Avoid interrupting an active stage; an interruption can leave unknown work.

Status distinguishes committed physical updates and teacher calls from retained
lineage progress. Historical discarded work is not reconstructed by subtraction.
The default profile remains a research session, distinct from `config/checkpoints.json`
and the older `learn-loop`, `english-loop` and desktop `launch` commands.

Implementation: [home_learning.py](../experiments/home_learning.py), delegating
to [the existing v3 controller](../experiments/continuous_tutor_campaign_v3.py).
Evidence: [completed campaign results](CONTINUOUS_TUTOR_RESULTS.md).
