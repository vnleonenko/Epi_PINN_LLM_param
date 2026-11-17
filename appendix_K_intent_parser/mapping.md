# Mapping: parser output → target code

English translation of `mapping_original_ru.md`, the file that was frozen before the run (SHA-256 `f00c8b40c59937906cc7d444631645c962f5a331e7beb479114e8424e4672ee4`). The code that applies it is `MAPPING` in `intent_mapping.py`.

Fields come from the `ExpertIntent` class of the article's parser (reproduced in `bundled_parser.py`). The fields `primary_metric` and `reasoning` are ignored.

## Rule

A code is set only if `cares_X` is true **and** the direction is specific (not `any` and not empty). If the resulting set is empty, the code is `none`.

| Axis (cares_…) | Direction field | Value | Code |
| --- | --- | --- | --- |
| `cares_about_position` | `position_direction` | `later` | `peak_later` |
| `cares_about_position` | `position_direction` | `earlier` | `peak_earlier` |
| `cares_about_position` | `position_direction` | `any` | *(no code)* |
| `cares_about_height` | `height_direction` | `higher` | `peak_higher` |
| `cares_about_height` | `height_direction` | `lower` | `peak_lower` |
| `cares_about_height` | `height_direction` | `any` | *(no code)* |
| `cares_about_growth` | `growth_direction` | `faster` | `growth_faster` |
| `cares_about_growth` | `growth_direction` | `slower` | `growth_slower` |
| `cares_about_growth` | `growth_direction` | `any` | *(no code)* |
| `cares_about_mortality` | `mortality_direction` | `higher` | `mortality_higher` |
| `cares_about_mortality` | `mortality_direction` | `lower` | `mortality_lower` |
| `cares_about_mortality` | `mortality_direction` | `any` | *(no code)* |
| — | — | empty set of codes | `none` |

If `cares_X` is true and the direction is neither in the table nor `any`, the call counts as `parse_error`.

Several axes can give several codes (group C). Codes are written to the CSV files separated by `;` in alphabetical order, and are compared with gold as sets.

## Group O (outside the six targets)

Gold is `growth_slower` or `mortality_lower`. By the protocol, the correct answer is **this direction or "not supported"**. "Not supported" in this schema is the set `{none}` (the parser set no target). The opposite direction (`growth_faster` / `mortality_higher`) is a critical error. `analyze_intent.py` applies this rule to all three methods (Intent Parser, keyword rule, template rule). The `correct` column of `intent_results.csv` is the strict set match, without this exception.

## Error types

| Type | Meaning |
| --- | --- |
| `opposite_direction` | Opposite direction on the same axis (critical) |
| `wrong_axis` | Another axis *instead of* the gold one (`peak_later` instead of `peak_lower`) |
| `extra_target` | An extra target in addition to the gold ones |
| `missing_target` | A gold target is missing (often group C) |
| `false_positive` | A target was given although gold is `none` (group N) |
| `parse_error` | The reply cannot be parsed, or a direction is outside the table |
| `no_majority` | All three calls differ (analysis stage only) |
