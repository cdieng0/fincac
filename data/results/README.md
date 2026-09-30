# Run artefacts

Files written by the runs reported in the paper, kept exactly as the scripts produced them.
[`tests/test_results.py`](../../tests/test_results.py) checks them against each other and
against the paper.

## Linear probing — Section 4.2

| File | Contents |
|---|---|
| [`probing/probing_results_20260827_164651.json`](probing/probing_results_20260827_164651.json) | Configuration (Mistral-7B-Instruct-v0.3 in 4-bit NF4, last-token pooling, L2 logistic regression C = 0.1, 5-fold CV, seed 42), per-layer and per-fold accuracy, permutation test |
| [`probing/probing_results_20260827_164651.csv`](probing/probing_results_20260827_164651.csv) | Mean and standard deviation of the accuracy for each of the 33 layers |
| [`probing/figure_probing_20260827_164651.png`](probing/figure_probing_20260827_164651.png) | Figure 2 of the paper |

Best layer 19: 45.0% accuracy against 30.7% for the majority class and 25% for chance;
permutation test p = 1/201 ≈ 0.005, the smallest value 200 permutations can give.

## ORPO Run 1 — total collapse

| File | Contents |
|---|---|
| [`orpo_run1/run_meta.json`](orpo_run1/run_meta.json) | Configuration: Mistral-7B-Instruct-v0.3, QLoRA NF4, LoRA r = 16 / α = 32, β = 0.1, 3 epochs, learning rate 5e-5, 273 training and 42 validation pairs, A100 40 GB, seed 42 |
| [`orpo_run1/training_log.json`](orpo_run1/training_log.json) | Telemetry: validation loss from 1.03 to 0.147; validation preference accuracy 95% at the first evaluation and 100% at the end |
| [`orpo_run1/eval_report.json`](orpo_run1/eval_report.json) | Gold evaluation: accuracy 16.4%, macro-F1 23.1%, FNR = 0 in all four eras |

Run 1's per-example predictions and pairs were not preserved.

## ORPO Run 2 — refuge-category collapse (published adapter)

| File | Contents |
|---|---|
| [`orpo_run2/eval_report.json`](orpo_run2/eval_report.json) | Accuracy 35.7%, macro-F1 35.9%, binary κ 0.238 [0.079, 0.387], FNR and FPR by era, confusion matrix, predicted-category distribution |
| [`orpo_run2/eval_predictions.csv`](orpo_run2/eval_predictions.csv) | One prediction per Gold paragraph; `pred_category` is empty when the output could not be parsed |

Both files are byte-identical to `eval_report.json` and `eval_predictions.csv` on
[CID99/Mistral-7B-ORPO-CSRD](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD/tree/main),
which also holds the pairs (`orpo_pairs.jsonl`), the training log and the adapter. To rebuild
the report from the predictions, on CPU:

```bash
python src/orpo/evaluate_orpo.py --from-predictions data/results/orpo_run2/eval_predictions.csv \
    --out data/orpo/eval_report_run2_recomputed.json
```

## Read before using these reports

- **`parse_failure_rate` is wrong in both reports.** It reads 0.0 because the script version
  that wrote them looked for the string `"PARSE_ERROR"`, while unparseable outputs were stored
  as missing values. Run 2's true rate is 43 / 140 = 30.7%, visible in
  `orpo_confusion.predicted_category_distribution`; Run 1's cannot be recovered. The corrected
  script is [`src/orpo/evaluate_orpo.py`](../../src/orpo/evaluate_orpo.py).
- **The FPR counts malformed answers.** Run 2's 64.9% (63 of 97 `none` paragraphs) is 35 CSRD
  labels plus 28 malformed outputs. The FNR does the opposite and does not count malformed
  answers as misses. The recomputed report gives both conventions.
- The `raw_predictions_csv` path inside the Run 2 report points to the original workstation;
  the file it names is `orpo_run2/eval_predictions.csv`.
