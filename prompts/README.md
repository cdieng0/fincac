# Prompts expérimentaux FINCAC40

## Source canonique

Les prompts utilisés pour **toutes** les expériences (benchmark Section 4, collecte paires ORPO, entraînement ORPO) proviennent de :

```
shared_prompts.py
```

Ce module exporte :
- `TAXONOMY_BLOCK` — 12 catégories ESRS/CSRD
- `TASK_BLOCK` — instructions + format JSON
- `FEW_SHOT_EXAMPLES` — 3 exemples de référence
- `build_system_prompt(n_shot)` — assemblage 0-shot ou 3-shot

## Versions

| Version | Fichier | Usage |
|---|---|---|
| v1 (canonique) | `shared_prompts.py` | Benchmark, ORPO Run 2 |
| HF release | `hf_model_build/prompt_template.txt` | Évaluation modèle ORPO publié |

## Règle

Ne jamais modifier un prompt expérimental sans :
1. Créer une nouvelle version (`v2`, etc.)
2. Documenter la justification dans `CHANGELOG.md`
3. Relancer `test_prompt_consistency.py`

## Vérification

```bash
MISTRAL_API_KEY=dummy python test_prompt_consistency.py
```

Voir `INCIDENT_NOTE.md` pour l'historique de la dérive de prompt.
