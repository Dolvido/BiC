# Training-state diagnostic after the fixed learning comparison

Declared while the shared-state learning comparison is running, before its final
scores have been inspected. This does not alter its fixed schedule or adoption
screen. It addresses a separate question: does the auxiliary decoder recover
the trained facts, even if the native policy cannot use them?

After the main run releases the GPU, inspect its two final 648-update checkpoints
using only the 108 previously selected actual training-fit rows. No development
or audit state targets are constructed. Keep both checkpoints unchanged, use
the same strict FP32 profile, and perform no optimization or teacher calls.

Group by family and actual turn count: nine groups of twelve episodes per model.
Each group receives one native forward with BOS-only reply prefixes, followed
by the shared auxiliary decoder. This is 18 native forwards, 18 auxiliary calls,
216 episode exposures, and 25,920 entity-state predictions in total. No freely
generated reply sequence is needed. Derive state targets from the verified
training English, then count exact typed-value accuracy separately for known
and unknown states, overall and by family. Also record the balanced training
loss and verify unchanged model state after inference.

The untrained auxiliary head in the zero-weight control supplies context, not
a competing state-trained model. Good auxiliary accuracy alone cannot establish
comprehension or justify promotion. Poor auxiliary accuracy does not prove the
encoder contains no facts. These are training-fit diagnostics; they do not
measure state-decoding transfer or uniquely identify a causal bottleneck.
