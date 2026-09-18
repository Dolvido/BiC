# Compact cognitive curriculum

The frozen `bic-cognitive-worlds-v2` generator supplies four small procedural
worlds. A lesson is reconstructed from its version, seed, split, family and
level. Six English turns enter the student through its ordinary byte encoder;
numeric observations and task tokens are zero. Labels, family names and recipe
metadata are outside policy inputs. The reviewed parser and state simulator
verify lessons before training and are absent from inference.

| Family | Levels 1 and 2 | Level 3 |
|---|---|---|
| Variable binding | Establish and revise named colors | Copy a value, revise its source, query either value |
| Graph reachability | Follow one- and two-link paths | Follow a three-link path |
| Arithmetic updates | Increase a count; then decrease it | Copy a derived count into another variable and update it |
| Conditional logic | Copy a switch; apply two-input ALL/ANY rules | Apply a three-input rule after an intervention |

The initial study trains the first three families at levels 1 and 2. Conditional
logic is reserved for adaptation. Graph level 3 extends path depth; binding and
arithmetic level 3 introduce previously untrained copying operations and syntax.
Their results must not be described as isolated depth transfer.

Every even/odd seed pair changes exactly one statement, keeps its questions
identical and reverses at least one known answer. In graphs, the changed statement
swaps two edge destinations. It preserves all nodes and their in/out degrees;
the queried terminal edge also remains unchanged at levels 2 and 3. Edge order,
the interrupted connection and early query placement vary. These controls remove
the identified endpoint, node-count and fixed-edge shortcuts; they do not prove
that every possible structural shortcut has been eliminated.

Undefined facts produce ASK. Graphs require an explicit declaration that all
links are listed before a missing path becomes DENY. Binding questions vary
their location and referenced variable. Exactly one quarter of conditional
episodes end by querying a lamp whose rule is undefined; the other three quarters
query the intervention's outcome. Conditional unknown-input reasoning exists in
the oracle tests but is not the generated uncertainty condition.

## Frozen bank inventory

The CPU-only quality report (archive reference: `../runs/general-learning-local/curriculum-quality.json`)
validated every canonical row and complete pair, checked the saved fingerprints,
and confirmed that the four arms share the same training/development banks. It
read no model checkpoints or performance results and changed no frozen source
or bank. Counts describe the fixed banks, not repeated optimizer exposures.

| Bank | Episodes | Distinct transcripts | Episode pairs | Maximum input bytes |
|---|---:|---:|---:|---:|
| Training: three families, levels 1/2 | 3,072 | 3,064 | 1,536 | 55 |
| Development: three families, levels 1/2 | 384 | 384 | 192 | 55 |
| Conditional adaptation support, levels 1/2 | 64 | 64 | 32 | 54 |
| Conditional audit queries, level 2 | 256 | 256 | 128 | 54 |
| Conditional audit queries, level 3 | 256 | 256 | 128 | 59 |
| Retention: three families, levels 1/2 | 384 | 384 | 192 | 55 |
| Trained-family audit transfer, level 3 | 768 | 758 | 384 | 55 |
| **Total** | **5,184** | **5,166** | **2,592** | **59** |

The banks contain 31,104 turns and 3,556 opposite-known-answer query pairs.
An episode pair can contain more than one eligible query pair. Training
arithmetic has 1,016 distinct transcripts among 1,024 rows; advanced audit
arithmetic has 246 among 256. All other listed cells contain distinct transcripts.
No exact six-sentence transcript overlaps between the bank groups. No support
overlap exclusions were needed for the prepared conditional query banks.
Different transcripts or seed identities need not represent independent semantic
worlds. Advanced arithmetic has no ASK targets; uncertainty cannot be assessed
for that cell.

## Split meaning and extension rules

All splits share twelve aliases. For primary alias indices `i,j`, training uses
`(i+j) % 4` residues 0/1, development uses 2 and audit uses 3: respectively
66, 30 and 36 allowed ordered pairs. These primary supports are disjoint;
incidental pairs, words, numerical ranges and operators can overlap. The held-out
conditional family changes grammar and vocabulary as well as its reasoning
mechanism. This finite curriculum measures a limited learning experiment, not
unrestricted language or general intelligence.

Additions require reviewed causal truth rules, hand-checked examples, provenance
and leakage tests, complete counterfactual pairs and appropriate shortcut
controls. A new family, grammar, generator, split or difficulty definition gets
a new version and a new prospective protocol with frozen source/data hashes
before training. Previously inspected audit banks become known benchmarks.
Preserve the current banks and report old and new evidence separately; never
silently expand a running study's curriculum or select its final questions from
model results. The full study contract is in
[GENERAL_LEARNING_PROTOCOL.md](GENERAL_LEARNING_PROTOCOL.md).
