# Provenance et schémas des données

## Chaîne de provenance AMF

```
Export AMF (info-financiere.gouv.fr)
  → flux-amf-new-prod.csv [SHA256: adad19f6...]
  → Filtre CAC 40
  → Filtre langue française
  → Filtre période 2010–2026
  → Téléchargement PDF (23 753 docs soumis)
  → Extraction pdfplumber (97.44% succès)
  → Reconstruction paragraphes + filtre structurel
  → frafin_raw_20260531_110531.parquet (313 898 paragraphes)
```

Volumes intermédiaires (`build_annotation_sample.py`, seed=42) :

| Étape | N |
|---|---:|
| Brut | 313 898 |
| Après filtre mécanique | 251 495 |
| Après déduplication contenu | 97 620 |
| Échantillon stratifié final | 14 974 |

## Corpus (`frafin_raw_*.parquet`)

| Colonne | Type | Description | Nullable |
|---|---|---|---|
| `row_id` | int | Identifiant unique ligne | Non |
| `paragraph_id` | string | SHA256 du contenu (**non unique**, ~18% doublons) | Non |
| `document_id` | string | ID document AMF | Non |
| `content` | string | Texte paragraphe (20–340 mots) | Non |
| `date_envoi` | date | Date certification AMF | Non |
| `emetteur` | string | Société émettrice CAC 40 | Non |
| `type_document` | string | Type réglementaire AMF | Non |
| `epoch` | string | Régime réglementaire dérivé | Non |

### Valeurs `epoch`

`2010-2014`, `2015-2019`, `2020-2022`, `2023-2026`

## Gold Standard (140 paragraphes)

| Colonne | Type | Description |
|---|---|---|
| (colonnes corpus) | — | Identiques au corpus |
| `csrd_category` | string | Catégorie ESRS ou `none` (12 valeurs) |
| `esrs_subcategory` | string | Sous-catégorie ESRS si applicable |
| `chain_of_thought` | string | Justification annotateur (~620 car.) |

### Distribution des labels (Gold 140)

| Catégorie | n |
|---|---:|
| none | 97 |
| E1 | 18 |
| ESRS2 | 17 |
| E5 | 5 |
| E2, S1, S4 | 1 chacun |
| E3, E4, S2, S3, G1 | 0 |

## Échantillonnage Gold — protocole

1. **Univers :** 14 974 paragraphes stratifiés (pas le corpus complet).
2. **Axes :** epoch × csrd_quartile (16 strates).
3. **Taille :** 500 exportés → 140 retenus pour évaluation finale.
4. **Seed :** 42.
5. **Exclusions :** aucun filtre sémantique préalable (quartile CSRD = axe de stratification, pas de filtre).
6. **Blind annotation :** quartile CSRD masqué à l'annotateur humain.

Rapports : `data/gold_sampling_report_20260621_210435.json`, `data/gold_150_report_20260809_181437.json`.

## Paires ORPO (`orpo_pairs_*.jsonl`)

| Champ | Description |
|---|---|
| `pair_id` | Identifiant paire |
| `paragraph_id` | Référence paragraphe source |
| `epoch` | Époque réglementaire |
| `pair_type` | A_recency_fn / B_overprediction_fp / C_category_confusion |
| `prompt` | Tour utilisateur seul |
| `chosen` / `rejected` | Réponses JSON string |
| `chosen_category` / `rejected_category` | Labels |
| `chosen_model` / `rejected_model` | Modèles source |

## Prédictions LLM (`predictions_shot*.csv`)

Colonnes par modèle : `{model}_prediction`, `{model}_raw`, `{model}_parse_ok`, plus métadonnées Gold (`csrd_category`, `epoch`, `content`).

## Alertes d'intégrité (audit 2026-09-04)

| Fichier | Problème | Action |
|---|---|---|
| `data/frafin_raw_20260531_110531.parquet` | Schéma incorrect (colonnes type Les Echos, pas AMF) malgré n=313898 | **Régénérer** via `main_amf.py` ou utiliser `hf_build/corpus/train.parquet` |
| `hf_build/corpus/train.parquet` | n=313608 (≠ 313898 du run_meta) | Vérifier filtre publish ; documenter delta 290 lignes |

- Documents AMF source : Licence Ouverte Etalab 2.0
- Corpus dérivé et Gold : Etalab 2.0 (cf. `README_en.md`)
- Code : à préciser dans `LICENSE` (voir action manuelle restante)
