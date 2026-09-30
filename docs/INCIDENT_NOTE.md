# Incident note — the ORPO correction attempts

The attempt to correct the suspected recency bias with ORPO (paper, Section 4.3) failed twice.
This note records, in order, the bugs found along the way, how each was fixed, and what each
run showed. The first two incidents were found and fixed before any training.

| When | What happened | Section |
|---|---|---|
| August 2026, first pair collection | The pair distribution came out inverted: prompt drift between scripts | [1](#1-prompt-drift-between-scripts) |
| 24 August, second collection | Rebalancing discarded 303 of 318 type-A pairs | [2](#2-the-rebalancing-cap-discarded-the-central-signal) |
| Run 1 | Every training signal looked healthy; the model had collapsed | [3](#3-run-1--silent-collapse) |
| Run 2 | Rebalanced and retrained; still a clear failure | [4](#4-run-2--refuge-category-and-malformed-output) |
| September 2026, audit | The evaluation reports misstated the malformed-output rate | [5](#5-evaluation-reports-misstated-the-malformed-output-rate) |

---

## 1. Prompt drift between scripts

**Symptom.** The first run of `build_orpo_pairs.py` (3,000 candidates, 2,122 without API
error) produced:

```
before balancing: B_overprediction_fp 280 · A_recency_fn 24 · C_category_confusion 74
```

Type A — a recency false negative, the case the hypothesis is about — was the rarest type,
while type B, meant as a secondary guardrail, dominated by more than ten to one.

**Cause.** `benchmark_temporal_drift.py`, `build_orpo_pairs.py` and `train_orpo_mistral7b.py`
each kept their own copy of the system prompt, and the copies had drifted apart without any
test noticing:

1. the taxonomy text was detailed in the benchmark and condensed, differently, in the two
   ORPO scripts;
2. the benchmark results used for the base policy were measured in 3-shot, while
   `build_orpo_pairs.py` queried the base policy (`open-mistral-7b`) in zero-shot.

Pair collection was therefore not measuring the errors documented in Section 4.1. This was a
failure to reproduce the measurement conditions, not evidence against the hypothesis.

**Fix.** The taxonomy, task instructions and few-shot examples were copied verbatim from the
benchmark into a single module, [`src/evaluation/shared_prompts.py`](../src/evaluation/shared_prompts.py),
which every ORPO script now imports; the base policy is queried with `build_system_prompt(3)`.
[`tests/test_prompts.py`](../tests/test_prompts.py) fails if the benchmark's prompt stops being
byte-identical to the shared one, if an ORPO script redefines a prompt block, or if the policy
is no longer queried in 3-shot. The exact prompts are also published as text in
[`prompts/`](../prompts/).

The same pass removed an API key hard-coded in `build_orpo_pairs.py`: keys are read from
environment variables only; the exposed key has to be treated as compromised and revoked.

## 2. The rebalancing cap discarded the central signal

After the prompt fix, a new collection (24 August, 3-shot) produced the distribution the
hypothesis predicts:

```
A_recency_fn 318 · B_overprediction_fp 8 · C_category_confusion 5
```

`balance_pairs()` then capped every type at `min_count × max_type_ratio`, using the rarest
type as the base: `5 × 3.0 = 15`. Type A went from 318 to 15 pairs — 303 valid examples of
the phenomenon under study thrown away, for no gain on the guardrail types.

**Fix.** The cap is now computed from `max(min_count, min_cap_floor)`, with
`min_cap_floor = 100`. On the same counts it gives `max(5, 100) × 3.0 = 300`, hence
300 / 8 / 5 pairs instead of 15 / 8 / 5; types B and C are unaffected.

## 3. Run 1 — silent collapse

Run 1 was trained on 300 type-A, 10 type-B and 5 type-C pairs (273 for training, 42 for
validation), for 3 epochs at a learning rate of 5 × 10⁻⁵. Its configuration, training log and
evaluation report are archived in [`data/results/orpo_run1/`](../data/results/orpo_run1/).

| Signal | What it suggested | What was actually happening |
|---|---|---|
| Validation loss from 1.03 to 0.147 | Healthy convergence | The loss measures the preference objective, not the classification task |
| Preference accuracy: 95% on validation from the first evaluation, 100% at the end; 100% on training batches from step 15 of 51 | Objective learned | Preferring a CSRD label over `none` satisfies 95% of the pairs |
| FNR = 0 in all four eras | Recency bias corrected | No relevant paragraph was answered `none` any more |
| Accuracy 70.8% → **16.4%** | — | Collapse on the actual task |

The false-negative rate reached its ideal value because the model no longer answered `none`,
which nulls the metric without correcting anything. The paper attributes the collapse to the
imbalance between pair types (300 type A against 15 guardrail pairs) and to the absence of
pairs reinforcing a correct `none`.

Run 1's per-example predictions were not preserved, so its κ, its false-positive rate and its
malformed-output rate cannot be recomputed. Its pair file (`orpo_pairs_20260825_012910.jsonl`)
is not published.

**Fixes before Run 2.**

- `balance_pairs()` also caps type A at `max_ratio_to_guardrail × (B + C)`, 5 by default.
- [`rebalance_existing_pairs.py`](../src/orpo/rebalance_existing_pairs.py) re-applies the
  balancing to an existing pair file without any API call, and oversamples the guardrail pairs
  three times.
- The context budget is measured with the real tokenizer on the longest prompt of the dataset,
  and raised automatically when too small. With the 3-shot system prompt (about 8,200
  characters), an early setting of 768 prompt tokens silently cut most of it, because TRL
  truncates from the start of the prompt.
- Learning rate and epochs lowered to 2 × 10⁻⁵ and 2, as a secondary safety margin.
- `evaluate_orpo.py` reports the false-positive rate per era, the confusion matrix and the
  distribution of predicted categories, and warns when the false-positive rate exceeds 30%.

## 4. Run 2 — refuge category and malformed output

Run 2 was trained on the rebalanced pairs — 225 type A, 30 type B, 15 type C — and is the
adapter published as [CID99/Mistral-7B-ORPO-CSRD](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD).
Its evaluation report and per-example predictions are archived in
[`data/results/orpo_run2/`](../data/results/orpo_run2/), and
[`tests/test_results.py`](../tests/test_results.py) recomputes the report from the predictions.

| | Base policy (3-shot) | Run 2 |
|---|---|---|
| Accuracy | 70.8% | 35.7% |
| Macro-F1 | 18.7% | 35.9% |
| Binary κ [95% CI] | 0.193 [0.051, 0.352] | 0.238 [0.079, 0.387] |
| Malformed JSON | 0% | 30.7% (43 / 140) |
| `none` paragraphs not answered `none` | not measured | 64.9% (63 / 97) |
| … of which given a CSRD label | not measured | 36.1% (35 / 97) |
| … of which malformed | not measured | 28.9% (28 / 97) |

- **Refuge category.** `ESRS2` reaches 5.3% precision (1 of 19 predictions) and 5.9% recall
  (1 of 17 instances), triggered notably by the word *risque* in standard legal clauses.
- **Capability regression.** The base policy always returned valid JSON; the tuned model does
  not in 30.7% of cases.
- **κ.** The gain from 0.193 to 0.238 is within overlapping confidence intervals and is not an
  improvement.

The paper reports the 64.9% as a global false-positive rate. It counts every answer other than
`none`, including the 28 malformed ones; the table above separates the two. Likewise, the
Run 2 false-negative rate (0.0 to 0.043 by era) does not count malformed answers as misses —
with the κ convention, where a malformed answer means no category detected, it is 0.33 / 0.40 /
0.42 / 0.35.

## 5. Evaluation reports misstated the malformed-output rate

Both archived reports give `parse_failure_rate: 0.0`. The script version that wrote them
compared predictions with the string `"PARSE_ERROR"`, while unparseable outputs were stored as
missing values, so the test never matched. The true Run 2 rate, 43 / 140 = 30.7%, can be read
from `orpo_confusion.predicted_category_distribution` in the same report; Run 1's cannot be
recovered.

[`src/orpo/evaluate_orpo.py`](../src/orpo/evaluate_orpo.py) now tests both representations,
reports the false-negative and false-positive rates under both conventions, and can rebuild a
report from an archived predictions file without a GPU. The archived reports are kept as they
were written.

---

## Lessons

- **Prompts are experimental artefacts.** They need a single definition, a test, and a
  published copy.
- **Training telemetry does not measure the task.** ORPO's loss and preference accuracy are
  computed on its own preference objective; neither saw the collapse. An evaluation callback on
  a held-out set, tracking FNR, FPR and the malformed-output rate during training, would have
  caught Run 1 early.
- **A target metric can be satisfied by collapse.** A false-negative rate of zero is only good
  news next to the false-positive rate and the prediction distribution.
- **Failed runs are results.** Both runs are kept and published.

## Checking

```bash
python -m pytest tests/test_prompts.py tests/test_results.py
python src/orpo/evaluate_orpo.py --from-predictions data/results/orpo_run2/eval_predictions.csv \
    --out data/orpo/eval_report_run2_recomputed.json
```
