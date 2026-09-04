# Données FINCAC40

Ce répertoire contient les données intermédiaires et finales du pipeline scientifique.

## Fichiers principaux

| Fichier | Type | Rôle |
|---|---|---|
| `frafin_raw_20260531_110531.parquet` | FINAL | Corpus 313 898 paragraphes |
| `frafin_sample_rigorous_20260531_225127.parquet` | DERIVED | Échantillon stratifié 14 974 |
| `gold_150_annotated_clean_reformulated_without.xlsx` | FINAL | Gold Standard 140 (évaluation) |
| `run_meta_20260531_110531.json` | METADATA | Provenance extraction |
| `sampling_report_rigorous_*.json` | METADATA | Protocole sampling |
| `results/` | ARTIFACT | Prédictions et métriques LLM |
| `orpo/` | ARTIFACT | Paires et évaluation ORPO |

## Régénération

Voir [`docs/reproducibility.md`](../docs/reproducibility.md).

## Gros fichiers

Les CSV corpus (>300 MB) sont exclus de Git. Utiliser les fichiers Parquet ou Hugging Face.

Voir [`docs/large_files_report.md`](../docs/large_files_report.md).
