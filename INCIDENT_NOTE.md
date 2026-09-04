# Incident de dérive de prompt — Phase 6 ORPO

**Date de découverte :** août 2026  
**Statut :** corrigé via `shared_prompts.py` + `test_prompt_consistency.py`  
**Impact scientifique :** Run 1 ORPO entraîné sur un signal de paires biaisé

---

## Résumé

Trois scripts (`benchmark_temporal_drift.py`, `build_orpo_pairs.py`, `train_orpo_mistral7b.py`) dupliquaient indépendamment le bloc de prompt (taxonomie + few-shot). Ces copies ont **divergé silencieusement** :

1. **Texte de taxonomie** — version détaillée vs condensée selon le script.
2. **Mode few-shot** — `build_orpo_pairs.py` appelait la politique de base en **0-shot** alors que le FNR=0.667 documenté en Section 4 avait été mesuré en **3-shot**.

Conséquence : la collecte de paires ORPO mesurait un phénomène différent de celui du papier. La distribution de paires s'est inversée (~280 faux positifs vs ~24 faux négatifs de récence), produisant un dataset fortement déséquilibré Type A / garde-fou (~97% Type A).

---

## Run 1 — collapse documenté

| Signal d'entraînement | Interprétation erronée | Réalité |
|---|---|---|
| `eval_loss` décroissant | Convergence saine | — |
| Préférence accuracy → 100% | Objectif appris | — |
| FNR → 0.000 (toutes époques) | Biais corrigé | Collapse : le modèle ne prédit plus `none` |
| Accuracy tâche | — | **70.8% → 16.4%** |

**Artefacts Run 1 :** non conservés localement dans ce dépôt (checkpoints supprimés après diagnostic). Les métriques sont documentées dans `MODEL_CARD.md`. Le dataset de paires initial (`orpo_pairs_20260825_012910.jsonl`, 556 KB) est conservé.

---

## Correctifs appliqués

1. **`shared_prompts.py`** — source unique de vérité pour taxonomie, tâche et few-shot.
2. **`build_orpo_pairs.py`** — `N_SHOT_POLICY = 3` aligné sur Section 4.
3. **`rebalance_existing_pairs.py`** — rééquilibrage 225/30/15 (Type A/B/C) sans nouvel appel API → `orpo_pairs_rebalanced_20260827_105744.jsonl`.
4. **`test_prompt_consistency.py`** — garde-fou byte-for-byte avant toute campagne ORPO.

---

## Run 2 — échec partiel documenté

Entraînement sur paires rééquilibrées (270 paires, seed=42). Artefacts complets :

- `outputs/mistral7b-orpo-csrd/` (checkpoints 20/25/28 + `final_adapter`)
- `data/orpo/eval_report_orpo.json`
- `hf_model_build/` (bundle Hugging Face)

| Métrique | Base (3-shot) | ORPO Run 2 |
|---|---|---|
| Accuracy | 70.8% | 35.7% |
| Malformed JSON | 0% | 30.7% |
| FNR global | élevé (récence) | ~0 (collapse inverse) |
| FPR global | — | 64.9% |

---

## Leçons pour la reproductibilité

- Les prompts expérimentaux sont des **artefacts scientifiques** — ils doivent être versionnés centralement.
- Un signal d'alignement (FNR→0) peut être **Goodhart-compatible** sans amélioration réelle.
- Les échecs négatifs (Run 1 et Run 2) font partie du résultat scientifique et ne doivent pas être supprimés.

---

## Vérification

```bash
MISTRAL_API_KEY=dummy python test_prompt_consistency.py
```
