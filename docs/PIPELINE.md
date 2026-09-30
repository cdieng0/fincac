# Pipeline

Every stage, the script that runs it, and the exact command. Run all commands from the
repository root. Scripts write to `data/` unless stated otherwise.

| # | Stage | Script | Needs | Used for the paper |
|---|---|---|---|---|
| 1 | Corpus extraction | `src/extraction/process_amf_extraction.py` | CPU, network, hours | yes |
| 2 | Stratified pool | `src/sampling/build_annotation_sample_rigorous.py` | CPU, minutes | yes |
| 3 | Annotation workbook | `src/annotation/export_gold_for_annotation.py` | CPU | planned hold-out, see [data.md](data.md#gold-standard) |
| 4 | Four-LLM benchmark | `src/evaluation/benchmark_temporal_drift.py` | API keys | yes (Section 4.1) |
| 5 | Metrics | `src/evaluation/compute_metrics.py`, `compute_missing_kappa.py` | CPU | yes |
| 6 | Linear probing | `src/probing/probe_temporal_signal.py` | GPU (T4 in 4-bit is enough) | yes (Section 4.2) |
| 7 | ORPO pairs | `src/orpo/build_orpo_pairs.py`, `rebalance_existing_pairs.py` | Mistral API key | yes (Section 4.3) |
| 8 | ORPO training | `src/orpo/train_orpo_mistral7b.py` | GPU (A100 used) | yes |
| 9 | ORPO evaluation | `src/orpo/evaluate_orpo.py` | GPU, or CPU from archived predictions | yes |
| 10 | Publication | `src/publishing/*.py` | Hugging Face login | yes |

`src/annotation/annotation_mistral_mass.py` and `validation_iaa.py` (LLM annotation of the
pool and Cohen's κ against a human annotator) belong to the planned 500-paragraph hold-out and
were not used for any reported result.

---

## 1. Corpus extraction

Input: the AMF metadata export `flux-amf-new-prod.csv`, downloaded from
[info-financiere.gouv.fr](https://www.info-financiere.gouv.fr) and placed at the repository
root. The script filters issuers, language and period, downloads each PDF, extracts the text
with `pdfplumber` and rebuilds paragraphs. No API key is needed.

```bash
python src/extraction/process_amf_extraction.py              # --workers 3 --batch 30 by default
python src/extraction/process_amf_extraction.py --resume     # continue after an interruption
```

Output: `frafin_raw_{TS}.parquet` and `.csv` (313,898 paragraphs for the release),
`run_meta_{TS}.json` (volumes, SHA-256 of the export), a log file and a checkpoint of the
URLs already processed. Most users should take the corpus from the Hub instead.

## 2. Stratified pool

```bash
python src/sampling/build_annotation_sample_rigorous.py --input data/frafin_raw_{TS}.parquet
```

Accepts the raw extraction or the Hub corpus. Output: `frafin_sample_rigorous_{TS}.parquet`
(14,974 paragraphs for the release), `stratum_table_{TS}.csv`,
`sampling_report_rigorous_{TS}.json`. Method in [data.md](data.md#stratified-pool).

## 3. Annotation workbook

```bash
python src/annotation/export_gold_for_annotation.py --input data/frafin_sample_rigorous_{TS}.parquet \
    --n-gold 500 --n-iaa 200 --seed 42
```

Output: `frafin_gold_500_{TS}.xlsx`, a blind annotation workbook. Protocol in
[annotation_protocol.md](annotation_protocol.md).

To work with the released Gold Standard instead, download it:

```bash
python src/publishing/fetch_gold_from_hf.py   # → data/gold_150_annotated_clean_reformulated_without.xlsx
```

## 4. Four-LLM benchmark

```bash
python src/evaluation/benchmark_temporal_drift.py --n-shot 0
python src/evaluation/benchmark_temporal_drift.py --n-shot 3
python src/evaluation/benchmark_temporal_drift.py --n-shot 3 --model mistral-7b --limit 10   # quick test
```

Models: `open-mistral-7b`, `mistral-large-latest`, `gpt-4o`, `claude-sonnet-4-6`, at
temperature 0 and seed 42. Needs `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
(only for the models you run). Zero-shot evaluates the 140 Gold paragraphs; 3-shot removes the
three few-shot examples and evaluates 137. `--resume` continues from the per-model checkpoint.

Output: `data/results/predictions_shot{0,3}_{TS}.csv` (schema in
[data.md](data.md#files-written-by-the-pipeline)).

## 5. Metrics

```bash
python src/evaluation/compute_metrics.py --csv data/results/predictions_shot0_{TS}.csv
python src/evaluation/compute_missing_kappa.py --csv data/results/predictions_shot3_{TS}.csv --model-key mistral-7b
```

`compute_metrics.py` computes binary F1, Cohen's κ with bootstrap intervals (B = 2000 globally,
500 per era, seed 42), FNR and FPR per era, McNemar tests between models and per-category
scores. Output: `metrics_computed_{TS}.json`, `latex_tables_{TS}.tex`, `fnr_by_epoch_{TS}.csv`,
`confusion_matrices/`. `compute_missing_kappa.py` computes the binary κ of Table 4 from any
predictions file, without API calls.

## 6. Linear probing

```bash
python src/probing/probe_temporal_signal.py --check-data-only          # no GPU
python src/probing/probe_temporal_signal.py --4bit --save-representations
python src/probing/probe_temporal_signal.py --load-representations data/probing/representations_{TS}.npz
```

Extracts the last-token hidden state of Mistral-7B-Instruct-v0.3 at its 33 layers for each Gold
paragraph (raw text, no prompt), then fits one L2 logistic regression per layer (C = 0.1,
stratified 5-fold CV) to predict the era, and runs a 200-permutation test on the best layer.
Output in `data/probing/`: `probing_results_{TS}.json` and `.csv`, `figure_probing_{TS}.png`.
The paper's run is archived in [`data/results/probing/`](../data/results/probing/).

## 7. ORPO pairs

```bash
python src/orpo/build_orpo_pairs.py --dry-run          # sampling plan, no API call
python src/orpo/build_orpo_pairs.py --target-pairs 1000
python src/orpo/build_orpo_pairs.py --resume           # continue; capped by --max-calls (6000)
python src/orpo/rebalance_existing_pairs.py --input data/orpo/orpo_pairs_{TS}.jsonl
```

Rejection sampling on the pool, Gold excluded: `mistral-large-latest` gives a silver label,
`open-mistral-7b` answers in 3-shot with the benchmark's prompt, and each disagreement becomes
a `chosen`/`rejected` pair. `rebalance_existing_pairs.py` caps type A at five times the
guardrail pairs and oversamples those three times — the Run 2 setting — without API calls.
Output: `data/orpo/orpo_pairs_{TS}.jsonl`, `orpo_pairs_rebalanced_{TS}.jsonl`, statistics and a
checkpoint. See [INCIDENT_NOTE.md](INCIDENT_NOTE.md) for why each safeguard exists.

## 8. ORPO training

```bash
python src/orpo/train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_rebalanced_{TS}.jsonl --check-data-only
python src/orpo/train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_rebalanced_{TS}.jsonl \
    --gold data/gold_150_annotated_clean_reformulated_without.xlsx --gold-safety-check
```

QLoRA 4-bit NF4, LoRA r = 16 / α = 32 / dropout 0.05 on all linear projections, ORPO β = 0.1,
effective batch 16, seed 42. Defaults are the Run 2 settings (2 epochs, learning rate 2e-5);
Run 1 used `--epochs 3 --learning-rate 5e-5`. The prompt budget is measured on the data and
raised if needed. Output in `outputs/mistral7b-orpo-csrd/`: checkpoints, `final_adapter/`,
`run_meta_{TS}.json`, `training_log_{TS}.json`.

## 9. ORPO evaluation

```bash
python src/orpo/evaluate_orpo.py --adapter-dir outputs/mistral7b-orpo-csrd/final_adapter \
    --baseline-csv data/results/predictions_shot3_{TS}.csv
# without a GPU, from archived predictions:
python src/orpo/evaluate_orpo.py --from-predictions data/results/orpo_run2/eval_predictions.csv \
    --out data/orpo/eval_report_run2_recomputed.json
```

Reports accuracy, macro-F1, binary κ with bootstrap interval, malformed-output rate, FNR and
FPR per era under both conventions for malformed answers, the confusion matrix and the
distribution of predicted categories. `src/orpo/diagnose_esrs2_bias.py`,
`diagnose_truncation.py` and `compare_base_vs_orpo_colab.py` are the diagnostics used to
analyse the failure.

## 10. Publication

```bash
huggingface-cli login
python src/publishing/publish_dataset_to_hf.py --corpus data/frafin_raw_{TS}.parquet \
    --gold data/gold_150_annotated_clean_reformulated_without.xlsx --repo CID99/FinCAC40 --push
python src/publishing/publish_model_to_hf.py --adapter outputs/mistral7b-orpo-csrd/final_adapter \
    --repo CID99/Mistral-7B-ORPO-CSRD --pairs data/orpo/orpo_pairs_rebalanced_{TS}.jsonl \
    --training-log outputs/mistral7b-orpo-csrd/training_log_{TS}.json \
    --eval-report data/orpo/eval_report_orpo.json \
    --eval-predictions data/orpo/raw_predictions_orpo_eval_report_orpo.csv --push
```

Without `--push`, both scripts only prepare the upload in a local folder. The cards they
upload are [`cards/dataset_card.md`](../cards/dataset_card.md) and
[`cards/model_card.md`](../cards/model_card.md). The demo Space is in [`space/`](../space/).
