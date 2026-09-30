# Annotation protocol

This is the protocol for annotating a FinCAC40 paragraph under the CSRD/ESRS taxonomy. It
is the protocol the dataset card points to, and the one to follow to extend the Gold Standard.

- **Taxonomy version:** 1.0 — ESRS Set 1 — Règlement délégué (UE) 2023/2772.
- **Source of truth:** [`src/annotation/csrd_taxonomy.py`](../src/annotation/csrd_taxonomy.py).
  Sections 3 to 7 below are generated from that module, and
  [`tests/test_docs.py`](../tests/test_docs.py) fails if the two drift apart.
- **Language:** the paragraphs are French, and so are the definitions below — they are quoted
  as the annotator read them and as the model prompt states them.

How the 140 released Gold paragraphs were selected is a separate question, covered in
[`data.md`](data.md#gold-standard).

---

## 1. What is annotated

One row per paragraph. Only the first three fields are published in the `gold` configuration
of [CID99/FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40); the others are part of the
protocol and of the annotation workbook, and are not released.

| Field | Values | Rule |
|---|---|---|
| `csrd_category` | 12 codes (section 3) | Mandatory. The main ESRS standard the paragraph discloses on, or `none`. |
| `esrs_subcategory` | code of section 4 | The most likely ESRS disclosure requirement. Empty when `csrd_category` is `none`. |
| `chain_of_thought` | free text | Written justification in three steps (section 8). |
| `materialite_score` | 0–5 (section 5) | 0 if and only if `csrd_category` is `none`. |
| `materialite_financiere` | yes / no | Financial materiality, *outside-in* (section 6). |
| `materialite_impact` | yes / no | Impact materiality, *inside-out* (section 6). Assessed independently of the previous one. |
| `market_surprise` | 4 levels (section 7) | Surprise relative to what the market expected. |
| `horizon_temporel` | 4 horizons (section 7) | When the effect materialises. Empty when `csrd_category` is `none`. |
| `confiance_annotation` | 0.0–1.0 | Optional. The annotator's confidence. |
| `notes_annotateur` | free text | Optional. Ambiguities, hesitations, cases to discuss. |

## 2. The `none` coherence rule

`none` is the majority class (69.3% of the Gold) and must be annotated with the same care as
any other label: never force a CSRD category onto a purely financial or administrative
paragraph. When `csrd_category` is `none`, the other fields are fixed:

| Field | Required value |
|---|---|
| `esrs_subcategory` | empty |
| `materialite_score` | 0 |
| `materialite_financiere`, `materialite_impact` | no |
| `horizon_temporel` | empty |

Conversely, a CSRD category requires a materiality score between 1 and 5. The function
`validate_annotation()` in `csrd_taxonomy.py` enforces these rules, checks that the subcategory
belongs to the chosen category, and rejects a justification shorter than 10 characters.

## 3. Categories

| Code | Pillar | Label | Definition |
|---|---|---|---|
| `none` | — | No CSRD relevance — *Aucune pertinence CSRD* | Le paragraphe ne traite d'aucun enjeu de durabilité environnemental, social ou de gouvernance au sens CSRD. Inclut : opérations financières pures (résultats, dividendes, fusions-acquisitions sans volet ESG), mentions légales, formalités administratives, déclarations de transactions sur titres sans lien avec la durabilité. |
| `ESRS2` | General | General disclosures — *Informations générales (gouvernance, stratégie, IRO)* | Gouvernance des enjeux de durabilité, modèle d'affaires et stratégie, processus d'identification des impacts/risques/opportunités (IRO), double matérialité au niveau de l'entité (hors enjeu sectoriel spécifique). |
| `E1` | Environment | Climate change — *Changement climatique* | Plan de transition climatique, émissions de gaz à effet de serre (Scope 1, 2, 3), consommation et mix énergétique, prix interne du carbone, effets financiers anticipés des risques climatiques physiques et de transition. |
| `E2` | Environment | Pollution — *Pollution* | Pollution de l'air, de l'eau et des sols, substances préoccupantes, microplastiques, effets financiers anticipés liés à la pollution. |
| `E3` | Environment | Water and marine resources — *Eau et ressources marines* | Consommation d'eau, prélèvements en zones de stress hydrique, ressources marines, rejets dans l'eau. |
| `E4` | Environment | Biodiversity and ecosystems — *Biodiversité et écosystèmes* | Plan de transition biodiversité, impacts sur les écosystèmes, sites sensibles, métriques de pression sur la biodiversité. |
| `E5` | Environment | Resource use and circular economy — *Utilisation des ressources et économie circulaire* | Flux entrants/sortants de ressources, déchets, recyclage, conception circulaire des produits. |
| `S1` | Social | Own workforce — *Effectifs propres* | Caractéristiques des effectifs, négociation collective, diversité, rémunération adéquate, santé-sécurité, formation, équilibre vie professionnelle/personnelle, incidents et plaintes. |
| `S2` | Social | Workers in the value chain — *Travailleurs de la chaîne de valeur* | Conditions de travail chez les fournisseurs et sous-traitants, travail des enfants, travail forcé, dialogue avec les travailleurs de la chaîne de valeur. |
| `S3` | Social | Affected communities — *Communautés affectées* | Droits des communautés locales, peuples autochtones, accès aux ressources, déplacement de populations, libertés civiles. |
| `S4` | Social | Consumers and end-users — *Consommateurs et utilisateurs finaux* | Sécurité des produits, protection de la vie privée, accès à l'information, pratiques commerciales responsables, inclusion. |
| `G1` | Governance | Business conduct — *Conduite des affaires* | Culture d'entreprise, protection des lanceurs d'alerte, lutte anti-corruption, relations fournisseurs, pratiques de paiement, influence politique et lobbying. |

## 4. Subcategories

Codes follow the numbering of the ESRS disclosure requirements.

- **ESRS2** — `ESRS2-GOV` Gouvernance des enjeux de durabilité · `ESRS2-SBM` Modèle d'affaires et stratégie · `ESRS2-IRO` Identification impacts/risques/opportunités · `ESRS2-MDR` Exigences de publication minimales
- **E1** — `E1-1` Plan de transition climatique · `E1-2` Politiques liées au climat · `E1-3` Actions et ressources climatiques · `E1-4` Cibles de réduction GES · `E1-5` Consommation et mix énergétique · `E1-6` Émissions GES brutes (Scope 1, 2, 3) · `E1-7` Absorptions et crédits carbone · `E1-8` Prix interne du carbone · `E1-9` Effets financiers anticipés (risques physiques/transition)
- **E2** — `E2-1` Politiques de prévention de la pollution · `E2-2` Actions liées à la pollution · `E2-3` Cibles de réduction de la pollution · `E2-4` Pollution air/eau/sol · `E2-5` Substances préoccupantes · `E2-6` Effets financiers anticipés
- **E3** — `E3-1` Politiques eau et ressources marines · `E3-2` Actions eau et ressources marines · `E3-3` Cibles de consommation d'eau · `E3-4` Consommation d'eau · `E3-5` Effets financiers anticipés
- **E4** — `E4-1` Plan de transition biodiversité · `E4-2` Politiques biodiversité · `E4-3` Actions biodiversité · `E4-4` Cibles biodiversité · `E4-5` Métriques d'impact biodiversité
- **E5** — `E5-1` Politiques ressources et économie circulaire · `E5-2` Actions économie circulaire · `E5-3` Cibles économie circulaire · `E5-4` Flux entrants de ressources · `E5-5` Flux sortants de ressources et déchets · `E5-6` Effets financiers anticipés
- **S1** — `S1-1` Politiques effectifs propres · `S1-2` Dialogue avec les effectifs · `S1-3` Mécanismes de remédiation · `S1-4` Actions effectifs propres · `S1-5` Cibles effectifs propres · `S1-6` Caractéristiques des effectifs · `S1-7` Travailleurs non-salariés · `S1-8` Couverture par négociation collective · `S1-9` Indicateurs de diversité · `S1-10` Rémunération adéquate · `S1-11` Protection sociale · `S1-12` Inclusion du handicap · `S1-13` Formation et développement des compétences · `S1-14` Santé et sécurité · `S1-15` Équilibre vie professionnelle/personnelle · `S1-16` Indicateurs de rémunération · `S1-17` Incidents, plaintes et impacts graves
- **S2** — `S2-1` Politiques travailleurs chaîne de valeur · `S2-2` Dialogue avec la chaîne de valeur · `S2-3` Mécanismes de remédiation · `S2-4` Actions chaîne de valeur · `S2-5` Cibles chaîne de valeur
- **S3** — `S3-1` Politiques communautés affectées · `S3-2` Dialogue avec les communautés · `S3-3` Mécanismes de remédiation · `S3-4` Actions communautés affectées · `S3-5` Cibles communautés affectées
- **S4** — `S4-1` Politiques consommateurs · `S4-2` Dialogue avec les consommateurs · `S4-3` Mécanismes de remédiation · `S4-4` Actions consommateurs · `S4-5` Cibles consommateurs
- **G1** — `G1-1` Culture d'entreprise et conduite des affaires · `G1-2` Relations avec les fournisseurs · `G1-3` Prévention et détection de la corruption · `G1-4` Incidents de corruption · `G1-5` Influence politique et lobbying · `G1-6` Pratiques de paiement

## 5. Materiality score

| Score | Level | Definition |
|---|---|---|
| 0 | Non applicable | Réservé aux paragraphes csrd_category = 'none'. |
| 1 | Négligeable | Mention factuelle ou descriptive sans implication significative sur la performance financière ou les impacts de durabilité. |
| 2 | Mineure | Information pertinente mais d'ampleur limitée, n'affecte pas significativement la trajectoire de l'entreprise. |
| 3 | Modérée | Enjeu identifié comme pertinent dans l'analyse de double matérialité, avec un effet mesurable mais non déterminant. |
| 4 | Significative | Impact direct sur la performance financière OU impact environnemental/social substantiel et documenté. |
| 5 | Critique | Enjeu central pour le modèle d'affaires : risque majeur, controverse significative, ou transformation structurelle de l'activité liée à la durabilité. |

## 6. Double materiality

The CSRD asks for both dimensions, and they are annotated independently: a paragraph can be
material on one, both or neither.

| Field | Definition |
|---|---|
| `materialite_financiere` | L'enjeu a/aura un effet sur la situation financière, la performance ou les flux de trésorerie de l'entreprise (matérialité 'outside-in'). |
| `materialite_impact` | L'entreprise a/aura un effet réel ou potentiel sur les personnes ou l'environnement, positif ou négatif (matérialité 'inside-out'). |

## 7. Market surprise and time horizon

| `market_surprise` | Definition |
|---|---|
| `aucune` — Aucune surprise | Information de routine, pleinement anticipée (ex : publication périodique conforme au calendrier réglementaire). |
| `faible` — Surprise faible | Information globalement attendue, avec des nuances mineures par rapport au consensus ou aux communications précédentes. |
| `partielle` — Surprise partielle | Information partiellement anticipée mais d'ampleur supérieure ou de nature différente de ce qui était attendu. |
| `forte` — Surprise forte | Information non anticipée, rupture avec la communication précédente ou le consensus de marché (ex : profit warning, incident majeur, changement de gouvernance soudain). |

| `horizon_temporel` | Definition |
|---|---|
| `immediat` — Immédiat | Effet déjà constaté ou se matérialisant sous 12 mois. |
| `court_terme` — Court terme | Effet attendu entre 1 et 3 ans. |
| `moyen_terme` — Moyen terme | Effet attendu entre 3 et 10 ans (horizon ESRS standard). |
| `long_terme` — Long terme | Effet attendu au-delà de 10 ans. |

## 8. Written justification (`chain_of_thought`)

Two to three sentences, always in this order:

1. **Nature of the information** — document type and factual content.
2. **Materiality analysis** — does the paragraph fall under the CSRD? If so, which category and
   why; if not, why not.
3. **Market surprise** — was the information anticipated by the market, and to what degree?

The model prompt asks for the same three steps, so human and model reasoning can be compared
step by step.

## 9. Procedure

[`src/annotation/export_gold_for_annotation.py`](../src/annotation/export_gold_for_annotation.py)
writes the annotation workbook from the stratified pool:

```bash
python src/annotation/export_gold_for_annotation.py --n-gold 500 --n-iaa 200 --seed 42
```

- The `Annotation` sheet has one row per paragraph. Yellow columns are filled by the
  annotator, grey columns are read-only metadata, and drop-down lists restrict each field to the
  values above (the subcategory list depends on the chosen category).
- **Blind annotation.** The variables used to draw the sample — keyword score, density
  quartile, stratum — are moved to a hidden sheet, so that they cannot anchor the annotator's
  judgement. Each paragraph is annotated on its content alone.
- **Agreement subset.** 200 of the 500 rows are flagged for an independent second annotation by
  Mistral Large ([`annotation_mistral_mass.py`](../src/annotation/annotation_mistral_mass.py)
  `--mode iaa`), and Cohen's κ is computed by
  [`validation_iaa.py`](../src/annotation/validation_iaa.py).

This 500-row, blind workbook is the hold-out set planned for the next version (paper,
Section 5, limitation 8). It is not the released Gold Standard, which was annotated by a single
expert with no agreement measure; see [`data.md`](data.md#gold-standard).

## 10. Extending the Gold Standard

Contributions are most useful, in this order:

1. **A second annotator** on the released 140 paragraphs, to measure inter-annotator agreement.
2. **The absent categories** — `E3`, `E4`, `S2`, `S3`, `G1`.
3. **The early eras** — 2010–2014 and 2015–2019 hold only 3 and 5 relevant paragraphs.

Annotate with the workbook above, check every row with `validate_annotation()`, and record
the taxonomy version you used.
