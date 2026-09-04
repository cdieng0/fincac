# Large Files Report — FINCAC40

Rapport généré lors de l'audit de reproductibilité (septembre 2026).  
Seuil de signalement : **> 10 MB**.

| Fichier | Taille (MB) | Type | Rôle | Source / généré | Reproductible ? | GitHub | LFS | Hugging Face |
|---|---:|---|---|---|---|---|---|---|
| `flux-amf-new-prod.csv` | 454 | CSV | Export métadonnées AMF brut | Source officielle | Téléchargement | **Non** | Non | Non |
| `data/frafin_raw_20260531_110531.csv` | 393 | CSV | Corpus 313 898 paragraphes | Généré (`main_amf.py`) | Oui (si source AMF) | **Non** | Non | Via HF |
| `data/frafin_raw_20260531_110531.parquet` | 89 | Parquet | Corpus (format compact) | Généré | Oui | **Non** | Non | Via HF |
| `hf_build/corpus/train.parquet` | 88 | Parquet | Release HF corpus | Copie du corpus | Oui | **Non** | Non | **Oui** |
| `data/frafin_sample_rigorous_20260531_225127.csv` | 28 | CSV | Échantillon stratifié 14 974 | Généré | Oui | **Non** | Non | Non |
| `data/frafin_sample_rigorous_20260531_225127.parquet` | 10 | Parquet | Idem | Généré | Oui | Optionnel | Non | Non |
| `outputs/.../adapter_model.safetensors` (×4) | 160 each | LoRA | Checkpoints ORPO Run 2 | Généré | Oui (GPU) | **Non** | Non | **Oui** |
| `outputs/.../optimizer.pt` (×3) | 82 each | PyTorch | États optimiseur | Généré | Oui (GPU) | **Non** | Non | Non |
| `hf_model_build/adapter_model.safetensors` | 160 | LoRA | Release HF modèle ORPO | Copie final_adapter | Archive | **Non** | Non | **Oui** |

## Fichiers < 10 MB mais scientifiquement critiques

| Fichier | Taille | Rôle | GitHub |
|---|---:|---|---|
| `hf_build/gold/test.parquet` | 96 KB | Gold Standard 140 | **Oui** |
| `data/gold_150_annotated_clean_reformulated_without.xlsx` | 84 KB | Gold source (Excel) | **Oui** |
| `data/orpo/orpo_pairs_rebalanced_*.jsonl` | 496 KB | Paires ORPO finales | **Oui** |
| `data/results/predictions_shot*.csv` | ~850 KB | Prédictions multi-LLM archivées | **Oui** |
| `data/orpo/eval_predictions.csv` | 131 KB | Prédictions ORPO Run 2 | **Oui** |

## Politique recommandée

- **GitHub** : code, configs, prompts, Gold Standard, métadonnées JSON, prédictions archivées, petits datasets.
- **Hugging Face** : corpus parquet, Gold parquet, adapter ORPO Run 2, bundle `hf_model_build/`.
- **Reproduction par téléchargement** : `flux-amf-new-prod.csv` depuis [info-financiere.gouv.fr](https://www.info-financiere.gouv.fr).
- **Reproduction par génération** : corpus, échantillon stratifié, paires ORPO (avec clés API).
- **Exclure de Git** : `.venv/`, `optimizer.pt`, CSV corpus brut, export AMF source.

## SHA-256 (artefacts clés)

| Artefact | SHA-256 | Référence |
|---|---|---|
| Source AMF | `adad19f619981c3e7ae38813b2a8d6523fef6522c99a845c294e9a6d0ccd72f1` | `data/run_meta_20260531_110531.json` |

> Pour les autres artefacts, exécuter `python scripts/compute_hashes.py` (à créer) ou `certutil -hashfile <file> SHA256` sous Windows.

## Risque historique Git

Le dépôt a déjà tracké `flux-amf-new-prod.csv` (454 MB) dans l'historique Git. **Un nettoyage d'historique (`git filter-repo` ou BFG) sera nécessaire avant le push public**, même si le fichier est désormais dans `.gitignore`.
