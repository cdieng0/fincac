# Reproductibilité — FINCAC40

Guide pas-à-pas pour un chercheur externe.

## Prérequis

- Python 3.10+ (testé avec 3.14 dans `.venv` local)
- 16+ GB RAM pour le corpus ; GPU NVIDIA pour ORPO/probing
- Clés API pour réévaluations LLM (Mistral, OpenAI, Anthropic)

## 01 — Environnement

```bash
git clone <URL_DU_DEPOT>
cd finsent

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Éditer .env avec vos clés API
```

## 02 — Vérifier l'installation

```bash
python test_prompt_consistency.py   # ne nécessite pas de clé API réelle
python -m pytest tests/ -v          # tests de reproductibilité (counts, schémas)
```

## 03 — Acquérir les données

### Option A — Hugging Face (recommandé pour Gold + corpus)

```python
from datasets import load_dataset
corpus = load_dataset("CID99/FinCAC40", "corpus", split="train")
gold = load_dataset("CID99/FinCAC40", "gold", split="test")
```

### Option B — Reconstruire depuis AMF

1. Télécharger l'export depuis [info-financiere.gouv.fr](https://www.info-financiere.gouv.fr)
2. Placer le fichier à la racine : `flux-amf-new-prod.csv`
3. Vérifier SHA256 contre `data/run_meta_20260531_110531.json`

## 04 — Construire le corpus

```bash
python main_amf.py
# Attendu : 313 898 paragraphes (cf. run_meta JSON généré)
```

**Coût :** CPU intensif, plusieurs heures, téléchargement ~23k PDFs.

## 05 — Vérifier le corpus

```bash
python -m pytest tests/test_corpus_counts.py -v
```

Attendu : `total_paragraphs == 313898`, colonnes requises présentes.

## 06 — Gold Standard

Le Gold 140 est archivé dans :

- `data/gold_150_annotated_clean_reformulated_without.xlsx`
- `hf_build/gold/test.parquet`

```bash
python -m pytest tests/test_gold_standard.py -v
```

## 07 — Évaluation LLM (Tier 3 — API)

**Les prédictions archivées permettent de vérifier sans relancer les APIs.**

```bash
# Relancer (coûteux, ~$50–200 selon tarifs)
python benchmark_temporal_drift.py --n-shot 0
python benchmark_temporal_drift.py --n-shot 3

# Vérifier depuis archive
python metrics.py --csv data/results/predictions_shot0_20260817_095926.csv
python metrics.py --csv data/results/predictions_shot3_20260818_103646.csv
```

## 08 — Linear probing

**GAP connu :** le script de linear probing (Mistral-7B, 33 layers, L2 logistic regression C=0.1, 5-fold CV) n'est pas présent dans ce dépôt. Les résultats du papier ne peuvent pas être reproduits localement sans le script original.

## 09 — ORPO (Tier 4 — GPU)

### Vérifier depuis artefacts archivés (Run 2)

```bash
python 06_evaluate_orpo.py \
  --adapter-dir outputs/mistral7b-orpo-csrd/final_adapter \
  --gold data/gold_150_annotated_clean_reformulated_without.xlsx \
  --baseline-csv data/results/predictions_shot3_20260818_103646.csv \
  --out data/orpo/eval_report_orpo.json
```

### Reconstruire paires + entraîner (GPU A100 ~30 min)

```bash
python build_orpo_pairs.py --resume
python rebalance_existing_pairs.py --input data/orpo/orpo_pairs_20260825_012910.jsonl
python train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_rebalanced_20260827_105744.jsonl
```

## 10 — Figures et tables

```bash
python metrics.py --csv data/results/predictions_shot0_20260817_095926.csv
# → data/results/latex_tables_*.tex, fnr_by_epoch_*.csv
```

## Seeds et paramètres globaux

| Paramètre | Valeur | Fichier |
|---|---|---|
| `random_seed` | 42 | sampling, training, benchmark |
| `bootstrap` (global) | 2000 | `metrics.py` |
| `bootstrap` (par époque) | 500 | `metrics.py` |
| `temperature` | 0.0 | `benchmark_temporal_drift.py` |
| `orpo_beta` | 0.1 | `train_orpo_mistral7b.py` |
| `lora_r` / `lora_alpha` | 16 / 32 | `train_orpo_mistral7b.py` |

## Limitations de reproductibilité

1. **Modèles propriétaires** — GPT-4o, Claude, Mistral-Large peuvent évoluer côté provider.
2. **Run 1 ORPO** — checkpoints non conservés ; métriques dans `MODEL_CARD.md` et `INCIDENT_NOTE.md`.
3. **Linear probing** — script absent.
4. **Annotation humaine** — protocole documenté, mais re-annotation produirait des labels différents.

Voir `docs/reproduction_matrix.md` pour le tableau complet.
