# Explicit continuation of the architecture comparison

On 17 September 2026 the original dispatcher completed preparation, independent
data verification, six calibration jobs, checkpoint verification, development
evaluation and selection. Calibration consumed **3,060 optimizer updates /
293,760 training episode exposures** and selected learning rate **0.003** for
both architectures. The original ledger (archive reference: `../runs/variant-study-execution-local/execution.json`)
is preserved byte-for-byte, including its stale running statuses.

At 06:16:55 UTC its execution session was absent. Independent Windows process
checks also found no Python dispatcher or workers. The interrupted main tree
contained exactly two worker records, with no job directories, checkpoints or
step journals. The frozen runner creates a persistent job directory and receipt
before constructing a learner and records intent before each optimizer step.
Under that observed intact filesystem and execution order, **zero formal main
optimizer updates occurred before recovery**. The setup CPU cost and exact
interruption time remain unknown; absence of a final status is not itself proof
of zero work.

The recovery record (archive reference: `../runs/variant-study-recovery-local/attempt-001/recovery.json`)
binds the unchanged protocol, selected rates, original dispatcher, 88 sources
and exact interruption evidence. Original phase logs and the interrupted worker
records are archived beside it. No completed phase was repeated. No lesson,
budget, source, evaluation bank or selection rule changed.

A hidden local dispatcher launched at 06:22:42 UTC and started the two main
workers at 06:22:43 UTC. Its current ledger (archive reference: `../runs/variant-study-recovery-local/attempt-001/execution.json`)
is authoritative for remaining work: six main runs totaling **18,432 updates /
1,769,472 training episode exposures**, followed by verification, evaluation and
summary only after both workers succeed. There is no automatic retry or model
promotion. Main results were pending when this record was written.

The launch record (archive reference: `../runs/variant-study-recovery-launch-local/launcher.json`)
identifies the Windows process. Monitor process creation time and command as
well as its PID; stale JSON alone must not trigger a second launch. The original
execution session 4384 must not be resumed or its dispatcher rerun.
