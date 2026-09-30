# data/

Scripts read and write their data here. Git ignores this folder except for this file and
[`results/`](results/), which archives the artefacts of the runs reported in the paper.

## Getting the data

The corpus and the Gold Standard are on Hugging Face:
**[CID99/FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40)**.

| Need | Command |
|---|---|
| Gold Standard, in the format the scripts read | `python src/publishing/fetch_gold_from_hf.py` → `data/gold_150_annotated_clean_reformulated_without.xlsx` |
| Full corpus (313,608 paragraphs) | `load_dataset("CID99/FinCAC40", "corpus", split="train")` |
| Rebuild the corpus from the AMF source | put `flux-amf-new-prod.csv` ([info-financiere.gouv.fr](https://www.info-financiere.gouv.fr)) at the repository root, then `python src/extraction/process_amf_extraction.py` |

## Files written by the pipeline

| File | Contents | Script |
|---|---|---|
| `frafin_raw_{TS}.parquet`, `.csv` | Extracted corpus (313,898 paragraphs for the release) | `src/extraction/process_amf_extraction.py` |
| `run_meta_{TS}.json` | Extraction provenance: SHA-256 of the AMF export, volumes | idem |
| `frafin_sample_rigorous_{TS}.parquet` | Stratified pool (14,974 paragraphs) | `src/sampling/build_annotation_sample_rigorous.py` |
| `stratum_table_{TS}.csv`, `sampling_report_rigorous_{TS}.json` | Strata and sampling report | idem |
| `frafin_gold_500_{TS}.xlsx` | Blind annotation workbook | `src/annotation/export_gold_for_annotation.py` |
| `results/predictions_shot{0,3}_{TS}.csv` | Four-LLM predictions on the Gold | `src/evaluation/benchmark_temporal_drift.py` |
| `results/metrics_computed_{TS}.json` | Metrics, bootstrap intervals, FNR by era | `src/evaluation/compute_metrics.py` |
| `probing/` | Probing results and figure | `src/probing/probe_temporal_signal.py` |
| `orpo/orpo_pairs_{TS}.jsonl` | Preference pairs | `src/orpo/build_orpo_pairs.py` |
| `orpo/eval_report_orpo.json`, `orpo/raw_predictions_orpo_*.csv` | Evaluation of an ORPO adapter | `src/orpo/evaluate_orpo.py` |

`{TS}` is the run timestamp (`YYYYMMDD_HHMMSS`). Schemas are in
[`docs/data.md`](../docs/data.md); run order in [`docs/PIPELINE.md`](../docs/PIPELINE.md).
