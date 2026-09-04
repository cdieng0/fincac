# FINCAC40 — Reproduction Matrix

Matrice de traçabilité **papier → code → données → résultats**.  
Les numéros de table/figure sont indicatifs — ajuster selon la version finale du preprint.

| Résultat papier | Script | Config / seed | Input | Output | Coût | Tier | Statut |
|---|---|---|---|---|---|---|---|
| Corpus 313 898 paragraphes | `main_amf.py` | `DATE_FROM/TO`, seed implicite | `flux-amf-new-prod.csv` | `data/frafin_raw_*.parquet` | CPU, ~24h | 4 | Reproductible (si source AMF) |
| Échantillon 14 974 | `build_annotation_sample.py` | seed=42 | `frafin_raw_*.parquet` | `frafin_sample_rigorous_*.parquet` | Faible | 1 | Reproductible |
| Gold 500 (annotation) | `export_gold_for_annotation.py` | seed=42 | sample 14.8k | `frafin_gold_500_*.xlsx` | Faible | 1 | Reproductible |
| Gold 140 (évaluation) | `export_150qwen.py` | seed=42 | pool 150 | `gold_150_annotated_clean_*.xlsx` | Faible | 1 | **Archive** (fichier présent) |
| IAA Cohen's κ | `validation_iaa.py` | — | Gold 500 + annotations Mistral | rapport IAA | API Mistral | 3 | Archive partielle |
| Table perf globale (0-shot) | `benchmark_temporal_drift.py --n-shot 0` | T=0, seed=42 | Gold 140 | `predictions_shot0_*.csv` | **API ×4** | 3 | **Archive** (CSV présent) |
| Table perf globale (3-shot) | `benchmark_temporal_drift.py --n-shot 3` | idem | Gold 140 | `predictions_shot3_*.csv` | **API ×4** | 3 | **Archive** (CSV présent) |
| FNR/FPR par époque | `metrics.py` | bootstrap B=2000, seed=42 | predictions CSV | `metrics_computed_*.json` | Faible | 2 | Reproductible depuis archive |
| McNemar inter-modèles | `metrics.py` | — | predictions CSV | idem | Faible | 2 | Reproductible |
| Figure dérive FNR | `metrics.py` | — | predictions CSV | `fnr_by_epoch_*.csv` | Faible | 2 | Reproductible |
| Linear probing Mistral-7B | **Non présent dans le dépôt** | C=0.1, 5-fold, 33 layers | hidden states | — | GPU | 4 | **GAP** |
| Paires ORPO | `build_orpo_pairs.py` | seed=42, 3-shot | sample 14.8k | `orpo_pairs_*.jsonl` | API Mistral | 3 | Archive + partiellement reproductible |
| Rééquilibrage paires | `rebalance_existing_pairs.py` | seed=42 | paires brutes | `orpo_pairs_rebalanced_*.jsonl` | Faible | 1 | Reproductible |
| ORPO Run 1 | `train_orpo_mistral7b.py` (août 2026) | β=0.1, seed=42 | paires déséquilibrées | checkpoints **supprimés** | GPU A100 | 4 | **Archive métriques seulement** |
| ORPO Run 2 | `train_orpo_mistral7b.py` | idem + paires rééquilibrées | 270 paires | `outputs/mistral7b-orpo-csrd/` | GPU A100 | 4 | **Archive complète** |
| Éval ORPO Run 2 | `06_evaluate_orpo.py` | 3-shot, seed=42 | Gold 140 + adapter | `eval_report_orpo.json` | GPU | 2 | Reproductible depuis archive |
| Table ORPO (échec) | `06_evaluate_orpo.py` + `MODEL_CARD.md` | — | Gold 140 | `eval_report_orpo.json` | — | 2 | Archive |
| Dataset HF | `publish_to_hf.py` | — | corpus + gold parquet | `hf_build/` | Faible | 1 | Publié `CID99/FinCAC40` |
| Modèle ORPO HF | `publish_model_to_hf.py` | — | Run 2 artifacts | `hf_model_build/` | Faible | 2 | Publié `CID99/Mistral-7B-ORPO-CSRD` |

## Tiers de reproductibilité

| Tier | Description | Exemples |
|---|---|---|
| **1** | Entièrement reproductible localement | sampling, metrics depuis CSV, rebalance ORPO |
| **2** | Vérifiable depuis artefacts archivés | prédictions LLM, eval ORPO, figures |
| **3** | Nécessite API externe (versions peuvent changer) | benchmark 4 modèles, collecte paires ORPO |
| **4** | Coûteux en calcul (GPU / temps) | extraction AMF complète, entraînement ORPO, probing |

## Commandes de reproduction

Voir `docs/reproducibility.md` pour le parcours complet numéroté 01–10.
