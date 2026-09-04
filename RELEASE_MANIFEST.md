# Release Manifest — FINCAC40 v1.0 (paper reproduction)

Release correspondant au preprint Dieng (2026).

## Code

| Composant | Fichiers |
|---|---|
| Extraction AMF | `main_amf.py` |
| Sampling | `build_annotation_sample.py`, `export_gold_for_annotation.py`, `export_150qwen.py` |
| Annotation | `frafin_annotator*.py`, `annotation_mistral_mass.py`, `validation_iaa.py` |
| Évaluation LLM | `benchmark_temporal_drift.py`, `metrics.py`, `shared_prompts.py` |
| ORPO | `build_orpo_pairs.py`, `rebalance_existing_pairs.py`, `train_orpo_mistral7b.py`, `06_evaluate_orpo.py` |
| Publication | `publish_to_hf.py`, `publish_model_to_hf.py` |
| Tests | `test_prompt_consistency.py`, `tests/` |

## Corpus

| Artefact | Emplacement | SHA256 source |
|---|---|---|
| Export AMF | `flux-amf-new-prod.csv` | `adad19f6...` |
| Corpus parquet | `data/frafin_raw_20260531_110531.parquet` | — |
| HF corpus | `hf_build/corpus/train.parquet` | — |

## Gold

| Artefact | Emplacement | n |
|---|---|---:|
| Gold Excel | `data/gold_150_annotated_clean_reformulated_without.xlsx` | 140 |
| HF gold | `hf_build/gold/test.parquet` | 140 |

## Prédictions archivées

| Mode | Fichier |
|---|---|
| 0-shot | `data/results/predictions_shot0_20260817_095926.csv` |
| 3-shot | `data/results/predictions_shot3_20260818_103646.csv` |
| Métriques 0-shot | `data/results/metrics_computed_20260817_235709.json` |
| Métriques 3-shot | `data/results/metrics_computed_20260818_115712.json` |

## ORPO

| Run | Dataset | Checkpoints | Eval |
|---|---|---|---|
| Run 1 | `orpo_pairs_20260825_012910.jsonl` | **Non conservés** | Métriques dans `MODEL_CARD.md` |
| Run 2 | `orpo_pairs_rebalanced_20260827_105744.jsonl` | `outputs/mistral7b-orpo-csrd/` | `data/orpo/eval_report_orpo.json` |

## Figures / Tables

| Output | Script source |
|---|---|
| `fnr_by_epoch_*.csv` | `metrics.py` |
| `latex_tables_*.tex` | `metrics.py` |
| `confusion_matrices/*.csv` | `metrics.py` |

## Environment

- Python 3.10+
- Voir `requirements.txt`
- Seeds : 42 partout ; bootstrap 2000/500

## Known limitations

1. Linear probing script not in repository
2. ORPO Run 1 checkpoints deleted
3. Proprietary model versions may drift
4. API keys required for full re-run of LLM evaluation

## Hugging Face

- Dataset : `CID99/FinCAC40`
- Model : `CID99/Mistral-7B-ORPO-CSRD`
- Space : `CID99/FinCAC40-collapse-demo`
