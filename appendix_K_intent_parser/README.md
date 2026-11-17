# Appendix K: test of the Intent Parser on paraphrased requests (Table A19)

The Intent Parser alone is called on 111 free-text requests (99 in English, 12 in Russian), three times each, and its output is compared with a gold label: a set of the six targets of Table 2, `none`, or, for group O, a direction outside the six. A keyword rule and a template rule are scored on the same requests. The Generator, the Critic and the optimizer are not called. The corpus, the gold labels, the mapping and both rules were frozen before the run, with their SHA-256 hashes recorded.

The folder is standalone: it does not import anything from the rest of the repository. The repository's `agents/IntentParserAgent.py` is the earlier peak-only parser without the growth and mortality fields. The test used the article's full parser, which is reproduced here as `bundled_parser.py` (the `ExpertIntent` schema and the reply parser) and `data/system_prompt.txt` (the exact system prompt of the run).

## Contents

- Scripts:
  - `run_intent_test.py` calls the model through LM Studio's OpenAI-compatible server and writes `intent_results.csv`, `prompt_used.txt` and `run_meta.json`. With `--mock` it runs without a model.
  - `analyze_intent.py` scores a run: majority of three calls, exact set match, Wilson 95% intervals, exact McNemar test against the keyword rule, and the blind-label exclusion.
  - `posthoc_primary_metric.py` is the post hoc check (not pre-registered). It re-maps the saved replies that failed only on the `primary_metric` field.
  - `l5_without_examples.py` repeats the L5 comparison without the two requests shown to the labelers.
  - `reproduce.py` runs the analysis and the post hoc check on the archived replies and compares the output with `results/`.
  - `selftest.py` holds unit tests and mock runs.
  - `intent_mapping.py` (the mapping and the statistics), `keyword_baseline.py` (the keyword rule and the template rule) and `bundled_parser.py` are used by the scripts.
- `data/`:
  - `intent_corpus.csv` is the corpus, with id, group, level, language, text and gold.
  - `baseline_results.csv` holds the outputs of the keyword rule and the template rule.
  - `system_prompt.txt` is the system prompt of the run, SHA-256 `2e89915c…f94bfb997e`.
- `labels/`: the blind labels of the two authors (`blind_labels_AK.csv`, `blind_labels_VN.csv`; their comments are in Russian) and `excluded_items.csv` (Q067 and Q107).
- `mapping.md`: the mapping from parser output to target code. `mapping_original_ru.md` is the frozen original.
- `results/` is the archived run behind Table A19:
  - `raw/` holds the 333 model replies (`intent_results.csv`), `prompt_used.txt` and `run_meta.json`. The model was google/gemma-4-e4b (MLX 4-bit, context 8192) on LM Studio 0.4.26+4, at temperature 0, seed 0 and max_tokens 2048. Paths in `run_meta.json` are those of the machine that ran the test.
  - `analysis/` is the primary analysis on 109 requests: `summary.md`, `summary_table.csv`, `errors.csv`, `confusion_matrix.csv` and `label_agreement.csv`. `summary_table_all_items.csv` gives the robustness row on all 111 requests.
  - `posthoc/` holds the post hoc check: `summary.md`, `recovered_rows.csv` and the re-mapped `intent_results_posthoc.csv`.

Main numbers (`results/analysis/summary.md`):

- The Intent Parser read 93 of 109 requests (85.3%, Wilson 95% CI 77.5–90.8%), the keyword rule 91, and the template rule 17 (exact McNemar test, p = 0.839).
- On L5, the parser read 6 of 11 and the keyword rule 0 of 11 (p = 0.031).
- Without the exclusion, on all 111 requests, the parser read 93, the keyword rule 92 and the template rule 18.
- In the post hoc check (`results/posthoc/summary.md`), the parser read 97 of 109 (p = 0.263).

Local absolute paths of the authors' working copy in `results/raw/run_meta.json` and `results/analysis/summary.md` were replaced with `<authors' working copy>` before publication. Hashes, model settings, timestamps and all results are unchanged.

## Reproduce

Run from the repository root:

```bash
pip install -r appendix_K_intent_parser/requirements.txt
python appendix_K_intent_parser/reproduce.py             # Table A19 and the post hoc check from the archived replies, a few seconds
python appendix_K_intent_parser/l5_without_examples.py   # L5 without Q068 and Q077: 5/9 against 0/9, p = 0.063
python appendix_K_intent_parser/selftest.py              # unit tests and mock runs, no model
```

`reproduce.py` exits with code 0 only when every regenerated file is byte-identical to the archived one. It runs the following two commands, which can also be run on their own:

```bash
python appendix_K_intent_parser/analyze_intent.py --labels blind_labels_AK.csv blind_labels_VN.csv --labeler-names AK VN
python appendix_K_intent_parser/posthoc_primary_metric.py --labels blind_labels_AK.csv blind_labels_VN.csv --labeler-names AK VN
```

The analysis, the self-test and the mock runs need only pydantic.

### New model run

1. In LM Studio, load google/gemma-4-e4b, the MLX 4-bit build with a context of 8192, the same build as in the ablation and Table 2.
2. Start the local server on port 1234.
3. Run:

```bash
python appendix_K_intent_parser/run_intent_test.py --model google/gemma-4-e4b --base-url http://127.0.0.1:1234/v1 --lmstudio-version "0.4.26+4" --model-format "MLX 4-bit, context 8192"
python appendix_K_intent_parser/analyze_intent.py --results appendix_K_intent_parser/results_rerun/raw/intent_results.csv --labels blind_labels_AK.csv blind_labels_VN.csv --labeler-names AK VN
```

The run makes 333 calls, which took about two hours in the reported run. It sends `data/system_prompt.txt` unchanged and refuses to start if the prompt's SHA-256 differs from that of the reported run. If the run is interrupted, repeat the same command: pairs (id, repeat) that are already written are skipped. Add `--no-resume` to start again from the beginning.

A dry run without a model:

```bash
python appendix_K_intent_parser/run_intent_test.py --mock gold   # or keyword, mixed
```

`--print-prompt` prints the system prompt and its hash.

### Output

Reruns write to `appendix_K_intent_parser/results_rerun/`:

- `raw/` holds a new model run.
- `analysis/` and `posthoc/` hold the regenerated tables.
- `mock/` holds the mock runs.
- `selftest/` holds the self-test output.

The archived `results/` folder is left untouched, and the scripts refuse to write into it. Set the `INTENT_OUT` environment variable to write somewhere else, or use `--out-dir`.
