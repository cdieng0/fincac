# ORPO — Documentation des deux runs

Les deux échecs ORPO constituent un **résultat scientifique négatif instructif** (Goodhart's law). Voir `MODEL_CARD.md` et `INCIDENT_NOTE.md`.

## Architecture des runs

```
experiments/orpo/  (organisation cible — artefacts actuels ci-dessous)

Run 1 (août 2026, collapse)
├── Dataset : orpo_pairs_20260825_012910.jsonl (556 KB, ~300 paires déséquilibrées)
├── Training : train_orpo_mistral7b.py
├── Checkpoints : SUPPRIMÉS (non conservés localement)
└── Métriques : accuracy 16.4%, FNR→0 (collapse)

Run 2 (août 2026, échec partiel — release HF)
├── Dataset : orpo_pairs_rebalanced_20260827_105744.jsonl (270 paires)
│   └── 225 Type A / 30 Type B / 15 Type C
├── Training : outputs/mistral7b-orpo-csrd/ (checkpoints 20/25/28 + final_adapter)
├── Config : run_meta_20260827_101159.json
├── Eval : 06_evaluate_orpo.py → eval_report_orpo.json
└── HF bundle : hf_model_build/
```

## Différences Run 1 vs Run 2

| | Run 1 | Run 2 |
|---|---|---|
| Paires | ~300, ratio Type A ~97% | 270, ratio A/B/C rééquilibré |
| Cause échec | Collapse "toujours CSRD" | Sur-prédiction + 30.7% JSON malformé |
| FNR | →0 (mécanique) | →0 (mécanique) |
| Accuracy | 16.4% | 35.7% |
| Checkpoints | Non conservés | `outputs/mistral7b-orpo-csrd/` |
| Prompt | Dérivé (pré-correctif) | Aligné via `shared_prompts.py` |

## Configuration Run 2 (archivée)

| Paramètre | Valeur |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Méthode | ORPO (Hong et al., 2024) |
| Quantization | QLoRA 4-bit NF4 |
| LoRA r / α | 16 / 32 |
| ORPO β | 0.1 |
| Learning rate | 2e-5 |
| Batch effectif | 16 |
| Epochs | 2 |
| Seed | 42 |
| GPU | NVIDIA A100-SXM4-40GB |
| Train / val pairs | 234 / 36 |
| Prompt format | Manual `[INST]` (pas `apply_chat_template`) |
| Policy n-shot | 3 |

## Parsing des réponses ORPO

`06_evaluate_orpo.py` et `eval_report_orpo.json` :

- Catégorie `PARSE_ERROR` si JSON invalide ou catégorie absente
- Convention métrique binaire : parse errors comptés comme `none`
- **Note :** `parse_failure_rate: 0.0` dans le JSON utilise une convention différente du comptage `PARSE_ERROR` (43/140 = 30.7%) — voir distribution dans `predicted_category_distribution`

## Reproduction Run 2

```bash
# 1. Vérifier cohérence prompts
MISTRAL_API_KEY=dummy python test_prompt_consistency.py

# 2. (Optionnel) Reconstruire paires — nécessite API
python build_orpo_pairs.py --resume
python rebalance_existing_pairs.py

# 3. Entraîner — GPU requis (~30 min sur A100)
python train_orpo_mistral7b.py

# 4. Évaluer
python 06_evaluate_orpo.py \
  --adapter-dir outputs/mistral7b-orpo-csrd/final_adapter \
  --gold data/gold_150_annotated_clean_reformulated_without.xlsx \
  --baseline-csv data/results/predictions_shot3_20260818_103646.csv
```

## Artefacts à ne jamais supprimer

- `data/orpo/orpo_pairs_*.jsonl` (toutes versions)
- `data/orpo/checkpoint_orpo_candidates.json`
- `data/orpo/orpo_stats_*.json`
- `outputs/mistral7b-orpo-csrd/` (Run 2)
- `hf_model_build/`
- `INCIDENT_NOTE.md`
