# RPlan corpus adjacency statistics

Generated 2026-11-02. Runtime 21.0s. Files: 80791. Parsed: 80791. Failed/unusable: 0.
Wall threshold: shared boundary >= 0.3 m (normalised coord x 18.0 = metres).
Null model: permutation of room-type labels within each plan, geometry fixed. Sample 10000 plans (parsed 10000), 20 shuffles each, seed 44.
Lift = observed adjacency rate / pooled null adjacency rate. 95% CI by percentile bootstrap (2000 resamples) over plans, resampling the OBSERVED rate only; the null rate is held fixed because it is estimated from a much larger pseudo-sample (plans x shuffles) and its own sampling error is small relative to plan-to-plan variance in the observed rate.
Data problems logged: 1102 (sample of up to 500 in the JSON).

| type A | type B | plans w/ both | room pairs | adjacent | obs rate | null rate | lift | 95% CI | door rate | open rate |
|---|---|---|---|---|---|---|---|---|---|---|
| living | storage | 14955 | 14980 | 14676 | 0.9797 | 0.4612 | 2.12 | [2.12, 2.13] | 1.00 | 0.00 |
| bedroom | living | 78404 | 104750 | 103714 | 0.9901 | 0.4695 | 2.11 | [2.11, 2.11] | 1.00 | 0.00 |
| dining | living | 1310 | 1311 | 1288 | 0.9825 | 0.4661 | 2.11 | [2.09, 2.12] | 0.99 | 0.00 |
| living | master_bedroom | 79532 | 80461 | 79855 | 0.9925 | 0.4794 | 2.07 | [2.07, 2.07] | 1.00 | 0.00 |
| kitchen | living | 77597 | 77737 | 76889 | 0.9891 | 0.4781 | 2.07 | [2.07, 2.07] | 1.00 | 0.00 |
| dining | kitchen | 1234 | 1239 | 1123 | 0.9064 | 0.4636 | 1.96 | [1.92, 1.99] | 0.44 | 0.00 |
| bathroom | living | 80659 | 97029 | 89474 | 0.9221 | 0.4726 | 1.95 | [1.95, 1.95] | 0.98 | 0.00 |
| hallway | living | 4285 | 4382 | 3685 | 0.8409 | 0.4457 | 1.89 | [1.86, 1.91] | 0.96 | 0.00 |
| balcony | living | 67735 | 86808 | 55117 | 0.6349 | 0.4566 | 1.39 | [1.38, 1.40] | 0.89 | 0.00 |
| kitchen | kitchen | 140 | 140 | 88 | 0.6286 | 0.4906 | 1.28 | [1.12, 1.44] | 0.42 | 0.00 |
| bathroom | master_bedroom | 79405 | 96743 | 49561 | 0.5123 | 0.4694 | 1.09 | [1.09, 1.10] | 0.11 | 0.00 |
| dining | storage | 219 | 220 | 104 | 0.4727 | 0.4806 | 0.98 | [0.84, 1.12] | 0.20 | 0.00 |
| bathroom | bedroom | 78279 | 130183 | 54363 | 0.4176 | 0.4620 | 0.90 | [0.90, 0.91] | 0.02 | 0.00 |
| master_bedroom | master_bedroom | 902 | 960 | 391 | 0.4073 | 0.4549 | 0.90 | [0.83, 0.96] | 0.04 | 0.00 |
| hallway | master_bedroom | 4225 | 4387 | 1750 | 0.3989 | 0.4516 | 0.88 | [0.85, 0.91] | 0.14 | 0.00 |
| bedroom | dining | 1180 | 1447 | 562 | 0.3884 | 0.4444 | 0.87 | [0.82, 0.93] | 0.17 | 0.00 |
| bedroom | storage | 14171 | 14738 | 5609 | 0.3806 | 0.4534 | 0.84 | [0.82, 0.86] | 0.03 | 0.00 |
| balcony | master_bedroom | 66665 | 86311 | 31746 | 0.3678 | 0.4557 | 0.81 | [0.80, 0.81] | 0.52 | 0.00 |
| dining | hallway | 74 | 80 | 25 | 0.3125 | 0.3923 | 0.80 | [0.55, 1.06] | 0.28 | 0.00 |
| bedroom | master_bedroom | 77156 | 102524 | 38106 | 0.3717 | 0.4680 | 0.79 | [0.79, 0.80] | 0.01 | 0.00 |
| bedroom | bedroom | 25744 | 26958 | 9104 | 0.3377 | 0.4447 | 0.76 | [0.75, 0.77] | 0.02 | 0.00 |
| bedroom | hallway | 4037 | 5115 | 1699 | 0.3322 | 0.4439 | 0.75 | [0.72, 0.78] | 0.08 | 0.00 |
| bathroom | dining | 1308 | 1450 | 493 | 0.3400 | 0.4675 | 0.73 | [0.67, 0.78] | 0.19 | 0.00 |
| hallway | hallway | 94 | 100 | 29 | 0.2900 | 0.4107 | 0.71 | [0.50, 0.93] | 0.14 | 0.00 |
| master_bedroom | storage | 14783 | 14884 | 4741 | 0.3185 | 0.4639 | 0.69 | [0.67, 0.70] | 0.03 | 0.00 |
| storage | storage | 24 | 26 | 8 | 0.3077 | 0.4500 | 0.68 | [0.34, 1.11] | 0.00 | 0.00 |
| hallway | storage | 807 | 824 | 240 | 0.2913 | 0.4276 | 0.68 | [0.61, 0.76] | 0.17 | 0.00 |
| bathroom | kitchen | 77473 | 92739 | 29332 | 0.3163 | 0.4701 | 0.67 | [0.67, 0.68] | 0.04 | 0.00 |
| kitchen | storage | 14271 | 14314 | 4400 | 0.3074 | 0.4595 | 0.67 | [0.65, 0.69] | 0.01 | 0.00 |
| bathroom | bathroom | 16261 | 16479 | 4596 | 0.2789 | 0.4324 | 0.64 | [0.63, 0.66] | 0.11 | 0.00 |
| bathroom | storage | 14924 | 19130 | 5577 | 0.2915 | 0.4551 | 0.64 | [0.63, 0.66] | 0.04 | 0.00 |
| bedroom | kitchen | 75309 | 100548 | 27680 | 0.2753 | 0.4667 | 0.59 | [0.58, 0.60] | 0.01 | 0.00 |
| bathroom | hallway | 4275 | 4951 | 1243 | 0.2511 | 0.4494 | 0.56 | [0.53, 0.58] | 0.14 | 0.00 |
| hallway | kitchen | 4099 | 4188 | 1001 | 0.2390 | 0.4432 | 0.54 | [0.51, 0.57] | 0.19 | 0.00 |
| balcony | kitchen | 64999 | 82874 | 17579 | 0.2121 | 0.4571 | 0.46 | [0.46, 0.47] | 0.49 | 0.00 |
| balcony | bedroom | 65793 | 110285 | 18409 | 0.1669 | 0.4478 | 0.37 | [0.37, 0.38] | 0.44 | 0.00 |
| balcony | storage | 11891 | 14468 | 1882 | 0.1301 | 0.4432 | 0.29 | [0.28, 0.31] | 0.43 | 0.00 |
| dining | master_bedroom | 1265 | 1296 | 161 | 0.1242 | 0.4556 | 0.27 | [0.23, 0.31] | 0.11 | 0.00 |
| balcony | dining | 791 | 887 | 101 | 0.1139 | 0.4433 | 0.26 | [0.21, 0.30] | 0.57 | 0.00 |
| kitchen | master_bedroom | 76369 | 77407 | 5593 | 0.0723 | 0.4761 | 0.15 | [0.15, 0.16] | 0.01 | 0.00 |
| balcony | bathroom | 67612 | 101159 | 6709 | 0.0663 | 0.4527 | 0.15 | [0.14, 0.15] | 0.32 | 0.00 |
| balcony | balcony | 18420 | 19730 | 1168 | 0.0592 | 0.4312 | 0.14 | [0.13, 0.15] | 0.06 | 0.00 |
| balcony | hallway | 3015 | 3545 | 157 | 0.0443 | 0.4271 | 0.10 | [0.09, 0.12] | 0.20 | 0.00 |
| dining | dining | 1 | 1 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |

## Data problems (sample)

- EDGE_MISSING_ROOM: 562
- ADJACENT_NO_EDGE: 333
- INVALID_POLYGON: 207

First 50 raw problem entries:

- INVALID_POLYGON 10006.json: room r_2
- EDGE_MISSING_ROOM 10006.json: u=r_1 v=r_2
- EDGE_MISSING_ROOM 10006.json: u=r_2 v=r_3
- INVALID_POLYGON 10012.json: room r_2
- EDGE_MISSING_ROOM 10012.json: u=r_1 v=r_2
- EDGE_MISSING_ROOM 10012.json: u=r_2 v=r_4
- INVALID_POLYGON 10054.json: room r_6
- EDGE_MISSING_ROOM 10054.json: u=r_0 v=r_6
- EDGE_MISSING_ROOM 10054.json: u=r_1 v=r_6
- EDGE_MISSING_ROOM 10054.json: u=r_6 v=r_7
- ADJACENT_NO_EDGE 10054: rooms r_0/r_1 types ('master_bedroom', 'storage')
- INVALID_POLYGON 10072.json: room r_6
- EDGE_MISSING_ROOM 10072.json: u=r_5 v=r_6
- EDGE_MISSING_ROOM 10072.json: u=r_6 v=r_7
- ADJACENT_NO_EDGE 10157: rooms r_6/r_7 types ('balcony', 'living')
- ADJACENT_NO_EDGE 10318: rooms r_0/r_2 types ('bathroom', 'kitchen')
- ADJACENT_NO_EDGE 10359: rooms r_2/r_3 types ('balcony', 'bathroom')
- INVALID_POLYGON 10462.json: room r_5
- EDGE_MISSING_ROOM 10462.json: u=r_0 v=r_5
- EDGE_MISSING_ROOM 10462.json: u=r_4 v=r_5
- EDGE_MISSING_ROOM 10462.json: u=r_5 v=r_6
- ADJACENT_NO_EDGE 10462: rooms r_4/r_6 types ('bedroom', 'kitchen')
- INVALID_POLYGON 10490.json: room r_5
- EDGE_MISSING_ROOM 10490.json: u=r_0 v=r_5
- EDGE_MISSING_ROOM 10490.json: u=r_4 v=r_5
- EDGE_MISSING_ROOM 10490.json: u=r_5 v=r_6
- ADJACENT_NO_EDGE 10490: rooms r_4/r_6 types ('bedroom', 'kitchen')
- INVALID_POLYGON 10494.json: room r_5
- EDGE_MISSING_ROOM 10494.json: u=r_0 v=r_5
- EDGE_MISSING_ROOM 10494.json: u=r_4 v=r_5
- EDGE_MISSING_ROOM 10494.json: u=r_5 v=r_6
- ADJACENT_NO_EDGE 10494: rooms r_4/r_6 types ('bedroom', 'kitchen')
- INVALID_POLYGON 10607.json: room r_5
- EDGE_MISSING_ROOM 10607.json: u=r_4 v=r_5
- EDGE_MISSING_ROOM 10607.json: u=r_5 v=r_7
- ADJACENT_NO_EDGE 10693: rooms r_5/r_7 types ('bathroom', 'living')
- INVALID_POLYGON 1099.json: room r_5
- EDGE_MISSING_ROOM 1099.json: u=r_1 v=r_5
- EDGE_MISSING_ROOM 1099.json: u=r_2 v=r_5
- INVALID_POLYGON 10896.json: room r_2
- EDGE_MISSING_ROOM 10896.json: u=r_1 v=r_2
- EDGE_MISSING_ROOM 10896.json: u=r_2 v=r_3
- EDGE_MISSING_ROOM 10896.json: u=r_2 v=r_4
- ADJACENT_NO_EDGE 10896: rooms r_1/r_4 types ('bedroom', 'kitchen')
- INVALID_POLYGON 11482.json: room r_4
- INVALID_POLYGON 11482.json: room r_5
- EDGE_MISSING_ROOM 11482.json: u=r_2 v=r_5
- EDGE_MISSING_ROOM 11482.json: u=r_3 v=r_4
- EDGE_MISSING_ROOM 11482.json: u=r_4 v=r_5
- EDGE_MISSING_ROOM 11482.json: u=r_4 v=r_6
