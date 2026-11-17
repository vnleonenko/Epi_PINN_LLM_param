> **POST HOC sensitivity analysis, not pre-registered.** Parse errors caused only by `primary_metric` = none/any are re-mapped from the saved raw outputs; everything else is identical to the main analysis. See `recovered_rows.csv`.

# Intent Parser pilot test — results

Exact set match of the majority vote (3 repeats) against gold,
with a Wilson 95% confidence interval per level. For group O
(out of set), a prediction of `none` (unsupported) is also
counted correct; the opposite direction is a critical error.
A lenient metric, gold ⊆ predicted, is reported alongside.

## Run

- Model: `unknown`
- Prompt source: `unknown`
- Prompt SHA-256: `unknown`
- Blind labels: 2 item(s) excluded; tables below use 109 of 111.

## Blind label agreement

An item is dropped from the main tables if any labeler disagrees
with corpus gold (exact set match; for group O, `none` also agrees).
Empty labels are treated as not labeled: they are counted and
warned about, but do not exclude the item.

| Labeler | File | Labeled | Empty | Agree with gold | 95% CI (Wilson) |
| --- | --- | ---: | ---: | ---: | --- |
| AK | `blind_labels_AK.csv` | 33/33 | 0 | 32/33 (97.0%) | [0.847, 0.995] |
| VN | `blind_labels_VN.csv` | 33/33 | 0 | 32/33 (97.0%) | [0.847, 0.995] |

Inter-labeler agreement is computed on items both labelers filled,
by exact set match (Cohen's κ on those set-valued categories).

| Pair | n both labeled | Agree with each other | Cohen's κ |
| --- | ---: | ---: | ---: |
| AK vs VN | 33 | 31/33 (93.9%) | 0.932 |

Excluded from the main analysis: **2** items (any labeler disagrees with gold):

- `Q067` (L5; gold=peak_higher, AK=peak_higher (agree=1), VN=none (agree=0)) — disagree:VN
- `Q107` (N; gold=none, AK=growth_slower;peak_lower (agree=0), VN=none (agree=1)) — disagree:AK

Per-level tables, McNemar, the confusion matrix and `errors.csv` below are computed on the remaining **109** of 111 items. Unfiltered tables: `summary_table_all_items.csv`. Item-level comparison: `label_agreement.csv`.

## Accuracy by level

| Level | n | LLM (majority) | LLM 95% CI (Wilson) | Keyword rule | Keyword 95% CI | Template rule |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| L0 | 6 | 6/6 (100.0%) | [0.610, 1.000] | 6/6 (100.0%) | [0.610, 1.000] | 6/6 (100.0%) |
| L1 | 18 | 17/18 (94.4%) | [0.742, 0.990] | 18/18 (100.0%) | [0.824, 1.000] | 0/18 (0.0%) |
| L2 | 18 | 17/18 (94.4%) | [0.742, 0.990] | 15/18 (83.3%) | [0.608, 0.942] | 0/18 (0.0%) |
| L3 | 12 | 12/12 (100.0%) | [0.758, 1.000] | 10/12 (83.3%) | [0.552, 0.953] | 0/12 (0.0%) |
| L4 | 12 | 8/12 (66.7%) | [0.391, 0.862] | 12/12 (100.0%) | [0.758, 1.000] | 0/12 (0.0%) |
| L5 | 11 | 6/11 (54.5%) | [0.280, 0.787] | 0/11 (0.0%) | [0.000, 0.259] | 0/11 (0.0%) |
| RU | 12 | 12/12 (100.0%) | [0.758, 1.000] | 11/12 (91.7%) | [0.646, 0.985] | 0/12 (0.0%) |
| C | 7 | 6/7 (85.7%) | [0.487, 0.974] | 6/7 (85.7%) | [0.487, 0.974] | 0/7 (0.0%) |
| H | 2 | 2/2 (100.0%) | [0.342, 1.000] | 2/2 (100.0%) | [0.342, 1.000] | 0/2 (0.0%) |
| N | 7 | 7/7 (100.0%) | [0.646, 1.000] | 7/7 (100.0%) | [0.646, 1.000] | 7/7 (100.0%) |
| O | 4 | 4/4 (100.0%) | [0.510, 1.000] | 4/4 (100.0%) | [0.510, 1.000] | 4/4 (100.0%) |
| **All items** | 109 | **97/109 (89.0%)** | [0.817, 0.936] | 91/109 (83.5%) | [0.754, 0.893] | 17/109 (15.6%) |
| All calls (no vote) | 327 | 291/327 (89.0%) | [0.851, 0.919] | — | — | — |

Lenient metric (gold ⊆ predicted), all items: 
103/109 (94.5%), Wilson [0.885, 0.975].

## Repeat consistency

109/109 items (100.0%) had all repeats identical (including unanimous parse errors).

## Error types (majority vote, incorrect items)

| Type | Count |
| --- | ---: |
| extra_target | 6 |
| missing_target | 6 |

Total incorrect items: 12 / 109. Full list: `errors.csv`.

## Exact McNemar test (LLM vs keyword rule)

Two-sided exact test on discordant pairs (binomial, p = 1/2).
`b` = LLM correct and keyword wrong; `c` = the reverse.

| Subset | n | LLM | Keyword | b | c | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L5 | 11 | 6/11 | 0/11 | 6 | 0 | 0.031 |
| C | 7 | 6/7 | 6/7 | 1 | 1 | 1.000 |
| overall | 109 | 97/109 | 91/109 | 13 | 7 | 0.263 |

## Confusion matrix (group = single, majority vote)

Rows are gold, columns are predicted. The six paper targets plus
`none`. Multi-label or out-of-set predictions are binned as `other`.

| gold \ pred | peak_higher | peak_lower | peak_earlier | peak_later | growth_faster | mortality_higher | none | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| peak_higher | 12 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| peak_lower | 0 | 13 | 0 | 0 | 0 | 0 | 1 | 1 |
| peak_earlier | 0 | 0 | 13 | 0 | 0 | 0 | 1 | 1 |
| peak_later | 0 | 0 | 0 | 14 | 0 | 0 | 1 | 0 |
| growth_faster | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 2 |
| mortality_higher | 0 | 0 | 0 | 0 | 0 | 13 | 2 | 0 |
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Notes

- Gold codes in the corpus are separated by a single `;`.
- `primary_metric` and `reasoning` are ignored by the mapping.
- Direction `any` on a cared-about axis contributes no code.
- The `correct` column in `intent_results.csv` is a strict exact
  set match per repeat (group-O special rule is applied only here).
- Template rule and keyword rule are the frozen baselines in
  `keyword_baseline.py` / `baseline_results.csv`.

