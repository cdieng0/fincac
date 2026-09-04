# FinSent Pipeline Documentation

## Architecture Générale

Le pipeline FinSent comprend 6 phases d'un système end-to-end pour l'extraction sémantique et l'affinement de modèles sur la taxonomie CSRD/ESRS.

```
Phase 1: EXTRACTION          Phase 2: SAMPLING           Phase 3: ANNOTATION
                │                    │                        │
                ▼                    ▼                        ▼
         AMF (API REST)    →   Stratification        →   GOLD (500 humain)
         23,753 documents     Rigorous (15,000)          → Mistral Mass (14,500)
                                                         → IAA Validation (200)
                │
                ▼
         Phase 4: EVALUATION
         
         benchmark_temporal_drift.py
         Mesure du biais par époque temporelle
         Base policy: Mistral-7B (3-shot)
         
                │
                ▼
         Phase 5: ORPO PAIRS
         
         build_orpo_pairs.py
         Rejection sampling sur erreurs observées
         
                │
                ▼
         Phase 6: TRAINING
         
         train_orpo_mistral7b.py
         Fine-tuning QLoRA + ORPO (GPU requis)
         
                │
                ▼
         evaluate_orpo.py
         Comparaison Base vs ORPO
         
                │
                ▼
         Phase 7: PUBLISHING
         
         publish_dataset_to_hf.py
         publish_model_to_hf.py
```

---

## Phase 1: EXTRACTION

**Script**: `src/extraction/process_amf_extraction.py`

### Rôle
Télécharge les 23,753 documents AMF via l'API REST, extrait les paragraphes, applique un nettoyage robuste.

### Commande type
```bash
python src/extraction/process_amf_extraction.py
python src/extraction/process_amf_extraction.py --workers 3 --batch 30 --resume
```

### Entrée
- AMF REST API (authentification par clé API, config `.env`)

### Sortie
- `data/frafin_raw_{TS}.csv` — Dump brut (23,753 lignes)
- `data/frafin_raw_{TS}.parquet` — Format Parquet (compressé)
- `data/frafin_processor_{TS}.log` — Journal des erreurs

### Infrastructure
- **Type**: CPU
- **Durée**: 30-60 minutes
- **Ressources**: ~2 GB RAM

### Caractéristiques
- Gestion du checkpoint JSON → reprise après crash
- Limitation des workers (3 par défaut) pour éviter throttle
- Nettoyage unicode robuste
- Suppression des duplicatas au niveau texte

---

## Phase 2: SAMPLING

**Script**: `src/sampling/build_annotation_sample_rigorous.py` *(à localiser)*

### Rôle
Stratification rigoureuse pour construire un échantillon représentatif de 15,000 paragraphes.

### Commande type
```bash
python src/sampling/build_annotation_sample_rigorous.py \
  --input data/frafin_raw_{TS}.parquet \
  --output data/frafin_sample_rigorous_{TS}.parquet \
  --n-samples 15000 --seed 42
```

### Entrée
- `data/frafin_raw_{TS}.parquet` (Phase 1)

### Sortie
- `data/frafin_sample_rigorous_{TS}.parquet` (15,000 lignes)
- `data/stratum_table_{TS}.csv` — Résumé stratification
- `data/sampling_report_rigorous_{TS}.json` — Statistiques

### Infrastructure
- **Type**: CPU
- **Durée**: 2-5 minutes
- **Ressources**: ~3 GB RAM

### Méthodologie
- Stratification: année_bucket (4) × csrd_quartile (4) = 16 strates
- Diversification round-robin par doc_group
- Plancher minimum par strate
- SEED=42 pour reproductibilité

---

## Phase 3: ANNOTATION

### 3a. Export GOLD

**Script**: `src/annotation/export_gold_for_annotation.py`

Sélectionne 500 paragraphes stratifiés pour annotation manuelle.

```bash
python src/annotation/export_gold_for_annotation.py \
  --input data/frafin_sample_rigorous_{TS}.parquet \
  --output data/frafin_gold_500_{TS}.xlsx \
  --n-gold 500 --n-iaa 200 --seed 42
```

**Sortie**: `data/frafin_gold_500_{TS}.xlsx`, `data/gold_sampling_report_{TS}.json`

**Durée**: 1-2 minutes | **Type**: CPU

**Notes**: 200 des 500 = IAA subset (ré-annotation indépendante)

---

### 3b. Annotation de Masse (Mistral)

**Script**: `src/annotation/annotation_mistral_mass.py`

Annote 14,500 paragraphes (pool ∖ GOLD) + ré-annote 200 IAA indépendamment.

```bash
python src/annotation/annotation_mistral_mass.py --mode mass --resume
python src/annotation/annotation_mistral_mass.py --mode iaa
```

**Entrée**: `data/frafin_sample_rigorous_{TS}.parquet`, GOLD annoté à la main

**Sortie**: `data/frafin_mass_annotated_{TS}.parquet`, `data/iaa_mistral_annotations_{TS}.parquet`

**Durée**: 3-6 heures | **Type**: API (CPU) | **Coût**: €50-100

**Caractéristiques**:
- Température = 0.0 (reproductibilité stricte)
- Validation JSON + retry automatique
- Guard-fous anti-data-leakage

---

### 3c. Validation IAA

**Script**: `src/annotation/validation_iaa.py`

Calcule Kappa de Cohen (humain vs Mistral) sur 200 paragraphes IAA.

```bash
python src/annotation/validation_iaa.py \
  --gold data/frafin_gold_500_{TS}.xlsx \
  --mistral-iaa data/iaa_mistral_annotations_{TS}.parquet \
  --bootstrap-n 2000
```

**Sortie**: `data/iaa_report_{TS}.json`, confusion matrices, disagreements

**Durée**: 5-10 minutes | **Type**: CPU | **Ressources**: 2 GB RAM

---

## Phase 4: EVALUATION

**Script**: `src/evaluation/benchmark_temporal_drift.py`

Mesure le biais du modèle de base (Mistral-7B) par époque temporelle.

```bash
python src/evaluation/benchmark_temporal_drift.py \
  --pool data/frafin_sample_rigorous_{TS}.parquet \
  --gold-annotations data/frafin_mass_annotated_{TS}.parquet \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --n-shot 3 \
  --output data/predictions_shot3_{TS}.csv
```

**Entrée**: Pool annoté, modèle Mistral-7B-Instruct-v0.3

**Sortie**: `data/predictions_shot3_{TS}.csv`, `data/benchmark_report_{TS}.json`

**Durée**: 30-90 minutes | **Type**: GPU recommandé | **Ressources**: 8+ GB VRAM

**Résultat clé** (FNR par époque):
- 2010-2014: 66.7% ← **Biais majeur**
- 2015-2019: 45.2%
- 2020-2022: 28.3%
- 2023-2026: 12.1%

---

## Phase 5: ORPO PAIRS

**Script**: `src/orpo/build_orpo_pairs.py`

Construit paires (chosen, rejected) par rejection sampling sur erreurs observées.

```bash
python src/orpo/build_orpo_pairs.py \
  --pool data/frafin_sample_rigorous_{TS}.parquet \
  --predictions data/predictions_shot3_{TS}.csv \
  --target-pairs 1000 \
  --output data/orpo/orpo_pairs_{TS}.jsonl \
  --resume
```

**Entrée**: Pool annoté, prédictions Phase 4, Mistral Large API

**Sortie**: `data/orpo/orpo_pairs_{TS}.jsonl`, `data/orpo/orpo_pairs_report_{TS}.json`

**Durée**: 2-8 heures | **Type**: API (CPU) | **Coût**: €30-50

**Stratégie**:
- **Type A (correction FN)**: silver=CSRD réel, base=none
- **Type B (garde-fou FP)**: silver=none réel, base=CSRD
- Échantillonnage prioritaire sur 2010-2014 et 2015-2019
- Exclusion stricte du Gold

---

## Phase 6: TRAINING ORPO

**Script**: `src/orpo/train_orpo_mistral7b.py`

Fine-tuning QLoRA + ORPO de Mistral-7B sur les paires.

```bash
# CPU validation (aucun GPU requis)
python src/orpo/train_orpo_mistral7b.py \
  --pairs data/orpo/orpo_pairs_{TS}.jsonl \
  --check-data-only

# GPU training (Colab, RunPod, local)
python src/orpo/train_orpo_mistral7b.py \
  --pairs data/orpo/orpo_pairs_{TS}.jsonl \
  --epochs 3 --beta 0.1 \
  --output outputs/mistral7b-orpo-csrd
```

**Entrée**: `data/orpo/orpo_pairs_{TS}.jsonl`

**Sortie**: `outputs/mistral7b-orpo-csrd/` (adapter LoRA)

**Durée**: 1-3 heures | **Type**: GPU REQUIS | **Ressources**: T4 16GB min (Colab), A100/L4 idéal

**Hyperparamètres**:
- Modèle: Mistral-7B-Instruct-v0.3
- Quantification: 4-bit NF4 (QLoRA)
- Rang LoRA: r=16, alpha=32
- Learning rate: 5e-4
- Perte ORPO: β = 0.1

---

## Phase 7: EVALUATION ORPO

**Script**: `src/orpo/evaluate_orpo.py`

Évalue le modèle ORPO entraîné sur le GOLD set complet (140-150 paragraphes).

```bash
python src/orpo/evaluate_orpo.py \
  --adapter-dir outputs/mistral7b-orpo-csrd/final_adapter \
  --gold data/gold_150_annotated_clean_reformulated_without.xlsx \
  --baseline-csv data/predictions_shot3_{TS}.csv \
  --out data/orpo/eval_report_orpo.json
```

**Entrée**: Adapter LoRA, GOLD, prédictions baseline

**Sortie**: `data/orpo/eval_report_orpo.json`, `data/orpo/predictions_orpo_{TS}.csv`

**Durée**: 5-15 minutes | **Type**: GPU recommandé | **Ressources**: 8+ GB VRAM

---

## Phase 8: PUBLISHING

### 8a. Publish Dataset

**Script**: `src/publishing/publish_dataset_to_hf.py`

```bash
python src/publishing/publish_dataset_to_hf.py \
  --gold data/frafin_gold_500_{TS}.xlsx \
  --mass data/frafin_mass_annotated_{TS}.parquet \
  --repo-id cheikhibra/FinCAC40 \
  --private
```

### 8b. Publish Model

**Script**: `src/publishing/publish_model_to_hf.py`

```bash
python src/publishing/publish_model_to_hf.py \
  --adapter-dir outputs/mistral7b-orpo-csrd \
  --repo-id cheikhibra/Mistral-7B-ORPO-CSRD \
  --private
```

---

## Récapitulatif Infrastructure

| Phase | Durée | Type | GPU ? | Coût |
|-------|-------|------|-------|------|
| 1 | 30-60 min | API | Non | Gratuit |
| 2 | 2-5 min | CPU | Non | Gratuit |
| 3a | 1-2 min | CPU | Non | Gratuit |
| 3b | 3-6 h | API | Non | €50-100 |
| 3c | 5-10 min | CPU | Non | Gratuit |
| 4 | 30-90 min | GPU | **Oui** | Gratuit (HF) |
| 5 | 2-8 h | API | Non | €30-50 |
| 6 | 1-3 h | GPU | **Oui** | €5-15 (Colab) |
| 7 | 5-15 min | GPU | Recommandé | Gratuit |
| 8 | ~5 min | CPU | Non | Gratuit |

---

## Configuration

### .env (à la racine)
```bash
AMF_API_KEY="votre_clé_amf"
MISTRAL_API_KEY="votre_clé_mistral"
OPENAI_API_KEY="sk-..."  # optionnel
HUGGINGFACE_TOKEN="hf_..."
```

### Structure des données
- Toutes les données → `data/`
- Résultats ORPO → `data/orpo/`
- Modèles → `outputs/`

---

## Dépannage

| Problème | Solution |
|----------|----------|
| API throttle (Phase 1) | Réduire `--workers` à 1-2 |
| Retry exceeded (Phase 3b) | Relancer avec `--resume` |
| CUDA OOM (Phase 6) | Réduire batch size à 1 |
| Model not found (Phase 7) | Vérifier `final_adapter/` existe |

