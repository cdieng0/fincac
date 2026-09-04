# Hugging Face — Ressources FINCAC40

## Séparation GitHub vs Hugging Face

| Plateforme | Contenu |
|---|---|
| **GitHub** | Code, protocoles, documentation, prompts, métadonnées, prédictions archivées, Gold Excel |
| **Hugging Face** | Corpus parquet, Gold parquet, adapter ORPO, bundle modèle complet |

## Dataset — `CID99/FinCAC40`

| Config | Split | Fichier local | Rows |
|---|---|---|---|
| `corpus` | train | `hf_build/corpus/train.parquet` | 313 898 |
| `gold` | test | `hf_build/gold/test.parquet` | 140 |

```python
from datasets import load_dataset
corpus = load_dataset("CID99/FinCAC40", "corpus", split="train")
gold = load_dataset("CID99/FinCAC40", "gold", split="test")
```

**Licence :** Etalab 2.0  
**Dataset card :** `hf_build/README.md` (copie de `README_en.md`)

### Génération locale

```bash
python publish_to_hf.py
```

## Modèle ORPO — `CID99/Mistral-7B-ORPO-CSRD`

Artefact de recherche documentant un **échec d'alignement** (Run 2).

| Fichier local | Rôle |
|---|---|
| `hf_model_build/adapter_model.safetensors` | LoRA adapter |
| `hf_model_build/prompt_template.txt` | Prompt d'évaluation exact |
| `hf_model_build/orpo_pairs.jsonl` | Paires d'entraînement |
| `hf_model_build/training_log.json` | Télémétrie entraînement |
| `hf_model_build/eval_predictions.csv` | Prédictions Gold 140 |
| `hf_model_build/eval_report.json` | Métriques complètes |

**Licence :** Apache 2.0 (adapter)  
**Model card :** `MODEL_CARD.md`

### Génération locale

```bash
python publish_model_to_hf.py
```

## Space démo

`CID99/FinCAC40-collapse-demo` — comparaison base vs ORPO (`app.py`)

## Limitations HF

- Le corpus publié peut contaminer les évaluations futures (cf. avertissement dataset card).
- Le modèle ORPO ne doit **pas** être déployé en production.
