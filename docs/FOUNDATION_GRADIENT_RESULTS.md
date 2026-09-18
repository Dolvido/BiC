# Fixed shared-objective gradient inspection results

Recorded 2026-09-17 09:05 UTC. Completed descriptive diagnostic. The subsequent selected comparison is [shared semantic reply normalization](FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md), recorded before its training. No checkpoint adoption or learning improvement follows automatically from this diagnostic.

The inspected endpoints do not show broad auxiliary-gradient opposition. Query gradients are larger than each individually weighted auxiliary in every one of the 42 matched groups in both measured parameter partitions. Reply gradients consistently align with query gradients but are small at their existing weight. Observation-byte gradients are larger than reply gradients and mostly near orthogonal to query gradients. These findings narrow the local explanation; they do not identify a successful replacement objective.

## Evidence and fixed scope

- Result: attempt-001/result.json (archive reference: `../runs/foundation-gradient-diagnostic-local/attempt-001/result.json`), SHA256 `875b8d6cfbd189f720fa08d00bd33a6b228c4d7f5e0560c75300b450294cbd92`.
- Source manifest SHA256: `003e7548c2789dba78683a1d38c8b523879abe4eddd2b1eb0b5409a3217bc4da`.
- Authenticated fitting-data SHA256: `7aa148e6637c134154394d771d18ddde249f9f8c391f6d2fbed13a38b21a08bd`.
- Endpoint checkpoint and weight identities are preserved in the result and its two load reports. They are the historical flat curriculum and mixed endpoints at 1,536 updates, not current formal-study models.
- Both endpoints used the same predetermined first four canonical pair slots in every fitting cell: 63 cells, 21 matched depth/operator/length groups, three families per group, eight episodes per family batch. No group was selected or resampled from scores.

The component reproduces each family's mean CE over present query classes 0/1/2, separate ACK CE, and reply/observation means over turn positions of token means pooled across that family batch. Four losses are then averaged across the three families. Weights are query 1, ACK 0.25, reply 0.1 and observation 0.1. Four component gradients come from the same three-family forward graphs before clipping.

The partitions are all `blocks.*` parameters and the one deduplicated `tokens.weight == observation_head.weight` parameter. The latter includes both input-embedding and tied observation-readout gradient paths. Positional embeddings, final normalization, action head, reply projection/decoder and observation bias are outside the reported geometry. Reductions of detached gradients use CPU float64; learner differentiation retains its declared float32 runtime.

## Actual work and preservation

The complete run performed **42 matched groups, 126 family forwards, 1,008 episode forwards and 168 component-gradient calls**. Every attempted call completed. It made zero learner optimizer updates and zero clipping calls. Two authenticated checkpoint images were decoded and two CPU model templates constructed; no optimizer state was restored or historical lesson stream replayed. The verified enclosing run receipt reports 26.375 seconds wall and 25.25 seconds CPU; peak CUDA allocated/reserved bytes were 843,961,856 / 889,192,448. Component-only wall times were approximately 4.890 and 4.906 seconds; these exclude enclosing input authentication and publication.

Both endpoint reports confirm unchanged weight digests, model state and input banks, restored individual module modes and requires-grad flags, unchanged packed batches, and completed device synchronization. Each endpoint used 151,467 actual / 155,768 padded encoder positions and 126,000 teacher-forced decoder row-steps. This is inspection work, not fitted-probe work or learner training.

## Aggregate interpretation

All summaries below give each of the 21 fixed matched groups equal weight within an endpoint. They summarize norms and cosines of **family-mean gradients**. They are not the norm/cosine of an average gradient across groups, and the groups or the two related endpoints are not independent statistical replicates. No confidence interval or significance claim is inferred.

| Endpoint | Partition | Median weighted reply/query norm | Median weighted observation/query norm | Mean query/reply cosine | Mean query/observation cosine | Negative query/auxiliary cosines |
|---|---|---:|---:|---:|---:|---:|
| curriculum | Blocks | 0.00694135 | 0.249062 | 0.736194 | 0.0554185 | 3/21 |
| curriculum | Tied embedding | 0.00730874 | 0.153676 | 0.806619 | 0.0388436 | 1/21 |
| mixed | Blocks | 0.00621875 | 0.178688 | 0.724593 | 0.00860374 | 3/21 |
| mixed | Tied embedding | 0.00532612 | 0.0976685 | 0.816899 | 0.00723971 | 1/21 |

Ratios are fractions: 0.00694 means about 0.694%, not 0.00694%. Query/reply cosine is positive in all 42 groups for each partition. Query/observation cosine is negative in 3/21 curriculum and 6/21 mixed block groups, and 3/21 curriculum and 8/21 mixed embedding groups. The combined weighted auxiliary is negative relative to query in 3/21 groups at each endpoint for blocks, and 1/21 for each endpoint's embedding.

The summed weighted gradient remains close to the query direction. Derived directly from the saved norms and query/auxiliary cosine, mean cosine(query,total) is 0.952727 / 0.973979 for curriculum/mixed blocks and 0.980810 / 0.993278 for the tied embedding; respective minima are 0.846753 / 0.878547 and 0.946334 / 0.967108. This calculation requires no further backward pass. It argues against describing the measured endpoint gradient as broadly redirected against the query objective.

Observation gradients remain a substantial secondary component, with block norm ratios spanning approximately 0.192–0.646 in curriculum and 0.0465–0.543 in mixed. Near orthogonality alone does not show harm: the auxiliary could support useful features, consume capacity, or affect AdamW moments in ways this inspection cannot settle. The small weighted reply gradient similarly does not establish that increasing it would improve representation learning.

### Weighted component norm summaries

| Endpoint | Partition | Component | Mean norm | Median norm | Minimum | Maximum |
|---|---|---|---:|---:|---:|---:|
| curriculum | Blocks | query | 0.356734 | 0.381375 | 0.160879 | 0.542648 |
| curriculum | Blocks | acknowledgement | 0.0061235 | 0.000224261 | 4.12162e-05 | 0.038738 |
| curriculum | Blocks | reply | 0.00275333 | 0.0019818 | 0.000778091 | 0.00818261 |
| curriculum | Blocks | observation | 0.0979469 | 0.0967306 | 0.0812945 | 0.114167 |
| curriculum | Tied embedding | query | 0.390422 | 0.381374 | 0.20083 | 0.626742 |
| curriculum | Tied embedding | acknowledgement | 0.00827137 | 0.000236604 | 5.19653e-05 | 0.0554995 |
| curriculum | Tied embedding | reply | 0.00311236 | 0.00195563 | 0.000822573 | 0.00958544 |
| curriculum | Tied embedding | observation | 0.062564 | 0.0613867 | 0.0501361 | 0.0794366 |
| mixed | Blocks | query | 0.332814 | 0.25554 | 0.127319 | 1.61235 |
| mixed | Blocks | acknowledgement | 8.4504e-05 | 8.43214e-05 | 2.65449e-05 | 0.000261699 |
| mixed | Blocks | reply | 0.00246912 | 0.00184231 | 0.000598792 | 0.0166917 |
| mixed | Blocks | observation | 0.0508795 | 0.047029 | 0.0244423 | 0.0976314 |
| mixed | Tied embedding | query | 0.508271 | 0.377045 | 0.201354 | 2.37182 |
| mixed | Tied embedding | acknowledgement | 7.63386e-05 | 7.43484e-05 | 2.35557e-05 | 0.00026096 |
| mixed | Tied embedding | reply | 0.00354448 | 0.00231979 | 0.000910105 | 0.0246598 |
| mixed | Tied embedding | observation | 0.038906 | 0.0367471 | 0.0230207 | 0.0669969 |

### Weighted component/query norm ratios

| Endpoint | Partition | Numerator | Mean ratio | Median ratio | Minimum | Maximum |
|---|---|---|---:|---:|---:|---:|
| curriculum | Blocks | acknowledgement | 0.0180733 | 0.000696576 | 8.52777e-05 | 0.0901099 |
| curriculum | Blocks | reply | 0.00738972 | 0.00694135 | 0.00262854 | 0.0209437 |
| curriculum | Blocks | observation | 0.309889 | 0.249062 | 0.192303 | 0.646239 |
| curriculum | Tied embedding | acknowledgement | 0.0233593 | 0.000683832 | 9.07701e-05 | 0.148451 |
| curriculum | Tied embedding | reply | 0.00766899 | 0.00730874 | 0.00298208 | 0.0216957 |
| curriculum | Tied embedding | observation | 0.182623 | 0.153676 | 0.101384 | 0.345405 |
| mixed | Blocks | acknowledgement | 0.000294084 | 0.000281059 | 0.000116818 | 0.000479781 |
| mixed | Blocks | reply | 0.00689229 | 0.00621875 | 0.00284289 | 0.015931 |
| mixed | Blocks | observation | 0.203897 | 0.178688 | 0.0465175 | 0.543362 |
| mixed | Tied embedding | acknowledgement | 0.000174852 | 0.000172549 | 6.47913e-05 | 0.000306589 |
| mixed | Tied embedding | reply | 0.0064083 | 0.00532612 | 0.00276582 | 0.0147131 |
| mixed | Tied embedding | observation | 0.102563 | 0.0976685 | 0.0268757 | 0.262708 |

### All pairwise cosine summaries

Every row contains 21 defined cosines; no norm is zero and no cosine is null. Negative counts use the exact saved values, not rounded display values. Positive loss weights do not change these component-pair cosines.

| Endpoint | Partition | Pair | Mean | Median | Minimum | Maximum | Negative / 21 |
|---|---|---|---:|---:|---:|---:|---:|
| curriculum | Blocks | acknowledgement/observation | 0.0488028 | 0.0317621 | -0.0117134 | 0.179616 | 2 |
| curriculum | Blocks | acknowledgement/reply | -0.0108206 | -0.00992136 | -0.174521 | 0.0512358 | 12 |
| curriculum | Blocks | query/acknowledgement | 0.00291144 | 0.00501161 | -0.0563615 | 0.05116 | 10 |
| curriculum | Blocks | query/observation | 0.0554185 | 0.085265 | -0.197583 | 0.148718 | 3 |
| curriculum | Blocks | query/reply | 0.736194 | 0.75388 | 0.571884 | 0.917483 | 0 |
| curriculum | Blocks | reply/observation | 0.0353606 | 0.0551898 | -0.235751 | 0.127848 | 5 |
| curriculum | Tied embedding | acknowledgement/observation | 0.0226764 | 0.0170349 | -0.0150468 | 0.101113 | 4 |
| curriculum | Tied embedding | acknowledgement/reply | -0.0282374 | -0.0282902 | -0.405457 | 0.211381 | 12 |
| curriculum | Tied embedding | query/acknowledgement | -0.018775 | -0.0222086 | -0.174581 | 0.17039 | 15 |
| curriculum | Tied embedding | query/observation | 0.0388436 | 0.0434807 | -0.069341 | 0.106426 | 3 |
| curriculum | Tied embedding | query/reply | 0.806619 | 0.829901 | 0.464613 | 0.945299 | 0 |
| curriculum | Tied embedding | reply/observation | 0.0303144 | 0.0500269 | -0.142336 | 0.113451 | 5 |
| mixed | Blocks | acknowledgement/observation | 0.0163741 | 0.0106476 | -0.0345181 | 0.071679 | 6 |
| mixed | Blocks | acknowledgement/reply | -0.00358686 | -0.00127374 | -0.0360523 | 0.0219094 | 12 |
| mixed | Blocks | query/acknowledgement | 0.00744135 | 0.00841256 | -0.0354995 | 0.0563292 | 7 |
| mixed | Blocks | query/observation | 0.00860374 | 0.01026 | -0.0162719 | 0.0378569 | 6 |
| mixed | Blocks | query/reply | 0.724593 | 0.697403 | 0.560112 | 0.990198 | 0 |
| mixed | Blocks | reply/observation | 0.0114839 | 0.0107446 | -0.0263763 | 0.0369305 | 4 |
| mixed | Tied embedding | acknowledgement/observation | -0.0325738 | -0.0281569 | -0.0981079 | -0.00119103 | 21 |
| mixed | Tied embedding | acknowledgement/reply | 0.00437571 | -0.00442599 | -0.105782 | 0.0975937 | 11 |
| mixed | Tied embedding | query/acknowledgement | 0.0291234 | 0.0227299 | -0.119608 | 0.163142 | 9 |
| mixed | Tied embedding | query/observation | 0.00723971 | 0.00580946 | -0.0247923 | 0.0436761 | 8 |
| mixed | Tied embedding | query/reply | 0.816899 | 0.823655 | 0.631995 | 0.989104 | 0 |
| mixed | Tied embedding | reply/observation | 0.0138966 | 0.0136007 | -0.0330998 | 0.0395104 | 3 |

### Query versus summed weighted auxiliary

| Endpoint | Partition | Mean cosine | Median cosine | Minimum | Maximum | Negative / 21 | Mean auxiliary/query norm | Median auxiliary/query norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| curriculum | Blocks | 0.0743953 | 0.0943376 | -0.183257 | 0.190764 | 3 | 0.312997 | 0.264883 |
| curriculum | Tied embedding | 0.0730446 | 0.0627773 | -0.0514173 | 0.225593 | 1 | 0.189137 | 0.16216 |
| mixed | Blocks | 0.046594 | 0.0354172 | -0.00188394 | 0.191937 | 3 | 0.204245 | 0.17924 |
| mixed | Tied embedding | 0.0812071 | 0.0641044 | -0.00581971 | 0.35947 | 1 | 0.103049 | 0.0980981 |

### Loss context

These are unweighted component losses, averaged across all 21 groups; family means below retain each of the three family views. The loss scales have different denominators and are not comparable as learning quality scores. ACK and teacher-forced reply CE are low while balanced query CE remains much larger.

| Endpoint | Family | Query CE | ACK CE | Reply CE | Observation CE |
|---|---|---:|---:|---:|---:|
| curriculum | color | 0.953864 | 0.001139 | 0.0485973 | 0.395727 |
| curriculum | count | 0.808261 | 4.80981e-05 | 0.039429 | 0.456695 |
| curriculum | switch | 0.903589 | 2.31967e-05 | 0.0446201 | 0.347455 |
| curriculum | equal-family mean | 0.888571 | 0.000403432 | 0.0442155 | 0.399959 |
| mixed | color | 0.959707 | 1.11544e-05 | 0.0489751 | 0.222093 |
| mixed | count | 0.840384 | 3.85958e-05 | 0.0412377 | 0.293533 |
| mixed | switch | 0.906158 | 7.41396e-06 | 0.0447003 | 0.185091 |
| mixed | equal-family mean | 0.902083 | 1.90547e-05 | 0.044971 | 0.233572 |

## All 42 matched groups

Each geometry tuple is **(weighted reply/query norm, weighted observation/query norm, query/reply cosine, query/observation cosine, query/summed-weighted-auxiliary cosine)**. Every row includes the same three families; these are not per-family gradient measurements. All tuples are rounded to six significant digits for display; the pinned JSON retains full values.

### Curriculum

| Group | Shared blocks tuple | Tied embedding tuple |
|---|---|---|
| d0/direct/t10 | (0.00839221, 0.298017, 0.823221, 0.111797, 0.134516) | (0.0080786, 0.192503, 0.846059, 0.0355503, 0.0707508) |
| d0/direct/t12 | (0.00457874, 0.487791, 0.679362, 0.122517, 0.128941) | (0.00431627, 0.309209, 0.770047, 0.0434807, 0.054096) |
| d0/direct/t8 | (0.0209437, 0.242709, 0.710771, 0.13198, 0.190764) | (0.0216957, 0.134517, 0.87882, 0.0894156, 0.225593) |
| d1/advance/t10 | (0.00575577, 0.280936, 0.781711, 0.104125, 0.119985) | (0.00621904, 0.162925, 0.797086, 0.0557903, 0.0858701) |
| d1/advance/t12 | (0.00718785, 0.525538, 0.670116, 0.0994613, 0.107969) | (0.00684311, 0.313781, 0.723331, 0.0824389, 0.0959148) |
| d1/advance/t8 | (0.0124648, 0.21135, 0.754162, 0.0934669, 0.137174) | (0.0136714, 0.128531, 0.853824, 0.0834103, 0.171549) |
| d1/copy/t10 | (0.00873053, 0.289806, 0.75684, 0.148718, 0.170882) | (0.008752, 0.161678, 0.917565, 0.0546768, 0.104087) |
| d1/copy/t12 | (0.00671688, 0.646239, 0.571884, 0.0371439, 0.043531) | (0.00730874, 0.345405, 0.670264, 0.0181276, 0.034316) |
| d1/copy/t8 | (0.00940817, 0.193017, 0.729278, -0.0484199, -0.012889) | (0.00992941, 0.115756, 0.79421, 0.0212853, 0.0885508) |
| d2/composed/t10 | (0.00534163, 0.232459, 0.660364, 0.0750445, 0.0919053) | (0.00475617, 0.131462, 0.687579, 0.0356718, 0.0569129) |
| d2/composed/t12 | (0.00694135, 0.454994, 0.670132, 0.0168953, 0.0265164) | (0.00757551, 0.306927, 0.735977, 0.00601591, 0.0327644) |
| d2/composed/t8 | (0.00789367, 0.200121, 0.767435, 0.0102613, 0.040454) | (0.00770939, 0.107227, 0.894452, -0.054648, 0.00964624) |
| d3/composed/t10 | (0.00437824, 0.241241, 0.806778, -0.0282738, -0.0136787) | (0.00426056, 0.153676, 0.799205, 4.96667e-05, 0.0221064) |
| d3/composed/t12 | (0.00469307, 0.398554, 0.665532, 0.0529867, 0.0621597) | (0.00591971, 0.234342, 0.758775, 0.066117, 0.0965022) |
| d3/composed/t8 | (0.00779797, 0.192303, 0.75388, 0.085265, 0.115616) | (0.00803282, 0.101384, 0.893387, 0.106426, 0.175831) |
| d4/composed/t10 | (0.00476508, 0.217968, 0.823256, 0.130153, 0.13092) | (0.00470074, 0.141834, 0.909591, 0.0952527, 0.0627773) |
| d4/composed/t12 | (0.00359555, 0.443292, 0.790048, 0.00547149, 0.0177895) | (0.00328686, 0.255586, 0.829901, 0.00130662, 0.0370373) |
| d4/composed/t8 | (0.00756359, 0.216427, 0.723587, 0.0866458, 0.111662) | (0.00782516, 0.125638, 0.895384, 0.0810951, 0.136069) |
| d5/composed/t10 | (0.0110203, 0.206704, 0.584157, 0.0867846, 0.0943376) | (0.0133211, 0.124982, 0.464613, 0.0774295, 0.0134066) |
| d5/composed/t12 | (0.00438651, 0.249062, 0.917483, -0.197583, -0.183257) | (0.00386444, 0.176749, 0.873628, -0.069341, -0.0514173) |
| d5/composed/t8 | (0.00262854, 0.27915, 0.82007, 0.0393476, 0.0470047) | (0.00298208, 0.110967, 0.945299, -0.0138355, 0.0115741) |

### Mixed

| Group | Shared blocks tuple | Tied embedding tuple |
|---|---|---|
| d0/direct/t10 | (0.012433, 0.223277, 0.641766, -0.00223078, 0.0334237) | (0.0138086, 0.129169, 0.631995, -0.0247923, 0.0426213) |
| d0/direct/t12 | (0.00315507, 0.228328, 0.8216, -0.00233669, 0.00902546) | (0.00311095, 0.116784, 0.823655, -0.0163202, 0.00566655) |
| d0/direct/t8 | (0.0127641, 0.178688, 0.631025, 0.0243711, 0.0692525) | (0.00957558, 0.0983375, 0.789093, 0.0206427, 0.0968274) |
| d1/advance/t10 | (0.00284289, 0.0752095, 0.884235, 0.013873, 0.0471895) | (0.00279704, 0.0372042, 0.929019, 0.0281159, 0.0973503) |
| d1/advance/t12 | (0.00496955, 0.223993, 0.902578, 0.0221722, 0.0421617) | (0.00486395, 0.122631, 0.915674, 0.0292042, 0.0653664) |
| d1/advance/t8 | (0.015931, 0.121403, 0.665385, 0.0378569, 0.12341) | (0.0147131, 0.077949, 0.801313, 0.0436761, 0.190138) |
| d1/copy/t10 | (0.0103362, 0.283435, 0.610378, 0.0280062, 0.0501815) | (0.0085748, 0.135831, 0.678182, 0.0214295, 0.0641044) |
| d1/copy/t12 | (0.00470309, 0.543362, 0.770347, -0.00789769, -0.00122381) | (0.00451992, 0.262708, 0.874535, -0.0209456, -0.00581971) |
| d1/copy/t8 | (0.00628933, 0.08336, 0.769088, 0.01026, 0.0680337) | (0.00532612, 0.0513377, 0.897379, 0.0313151, 0.123436) |
| d2/composed/t10 | (0.00347884, 0.177858, 0.828734, 0.00314759, 0.0193656) | (0.00356953, 0.104541, 0.863405, 0.00326073, 0.0328208) |
| d2/composed/t12 | (0.00425057, 0.312805, 0.758981, -0.012248, -0.00188394) | (0.00446792, 0.156565, 0.819335, -0.0136673, 0.00977374) |
| d2/composed/t8 | (0.00935782, 0.124152, 0.608956, 0.0179023, 0.0633867) | (0.00768167, 0.0791573, 0.850385, 0.0315018, 0.112892) |
| d3/composed/t10 | (0.00311685, 0.166936, 0.770919, -0.0162719, -0.00184769) | (0.00276582, 0.06977, 0.918375, 0.00210568, 0.0387422) |
| d3/composed/t12 | (0.00876939, 0.222792, 0.845213, 0.00219438, 0.0354172) | (0.00854778, 0.0976685, 0.852197, -0.0112184, 0.0630454) |
| d3/composed/t8 | (0.0066452, 0.0465175, 0.602626, 0.0119816, 0.0970048) | (0.00631452, 0.0268757, 0.754377, 0.00580946, 0.177911) |
| d4/composed/t10 | (0.00386972, 0.304072, 0.697403, 0.000604217, 0.00950376) | (0.00334463, 0.130753, 0.87894, -0.00747723, 0.0148439) |
| d4/composed/t12 | (0.00399267, 0.488546, 0.613768, 0.00230839, 0.00731273) | (0.00383452, 0.217522, 0.698229, -0.00600388, 0.00628568) |
| d4/composed/t8 | (0.00679817, 0.0870768, 0.614526, 0.0105633, 0.0583252) | (0.00704413, 0.0579282, 0.796191, 0.00681329, 0.102673) |
| d5/composed/t10 | (0.00446348, 0.234244, 0.628623, 0.0111816, 0.0231892) | (0.0039444, 0.091649, 0.741508, -0.00254742, 0.0294688) |
| d5/composed/t12 | (0.0103524, 0.0605523, 0.990198, 0.0263743, 0.191937) | (0.010397, 0.0273299, 0.989104, 0.00940244, 0.35947) |
| d5/composed/t8 | (0.00621875, 0.095227, 0.560112, -0.00113349, 0.0353093) | (0.0053723, 0.0621179, 0.651997, 0.0217295, 0.0777325) |

## Limits and next decision

- This inspection describes two trained historical flat endpoints on four fitting pairs per cell. It does not measure early-training gradients, held-out gradient behavior, another architecture, or the full historical sampling distribution.
- Three family losses are averaged before differentiation. Component geometry therefore cannot establish separate per-family gradient agreement or diagnose cross-family gradient cancellation. Per-family loss values are descriptive context only.
- The query component combines known and unknown classes. Reply gradients combine the full teacher-forced reply; the experiment does not isolate first-token versus suffix gradients, known versus unknown gradients, or prove that the small reply gradient is caused by a particular reduction.
- Norms/cosines use ordinary, unpreconditioned partial derivatives in two partitions. They omit other parameters, clipping, weight decay, AdamW moments and actual parameter displacements. Endpoint geometry cannot reconstruct the training trajectory or identify the cause of weak transfer.
- All 42 groups are disclosed. Reused samples, related cells and shared endpoints do not provide 42 independent replications. The source diagnostic admission establishes transcript freshness within a known grammar, not novel algorithms or general intelligence.
- The completed representation probes were bounded and unconverged. Their outcomes and these local gradients do not establish absence of useful semantic information or select a unique loss change.

**Next comparison: shared semantic reply normalization.** The [separate prospective protocol](FOUNDATION_OBJECTIVE_STUDY_PROTOCOL.md) changes the reply reduction to equal present semantic classes of per-utterance token means, retaining its 0.1 coefficient and every other objective branch. It preserves paired initial weights, architecture, admitted lessons and update allowances, with equal learning-rate calibration for each arm. It assesses all families and cells, paired final decisions and teacher-free replies, retention and cost. Previously viewed banks are explicitly reused benchmarks. A useful result requires measured improvement; neither a low auxiliary loss nor a favorable gradient cosine is sufficient.

Only pinned JSON was read to produce this report. Aggregation used Python standard-library arithmetic, with no model, tensor, probe, optimizer or additional gradient execution.
