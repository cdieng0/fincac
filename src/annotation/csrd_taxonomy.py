"""
csrd_taxonomy.py — Taxonomie ESRS/CSRD Partagée — FraFin-Reasoning
═══════════════════════════════════════════════════════════════════════════════════════
Module fondation importé par :
    - export_gold_for_annotation.py  (génération du fichier Excel des 500 Gold)
    - annotation_mistral_mass.py     (prompt + validation des réponses Mistral)
    - validation_iaa.py              (calcul du Kappa de Cohen par catégorie)

RÉFÉRENCE OFFICIELLE :
    European Sustainability Reporting Standards (ESRS) — Set 1
    Règlement délégué (UE) 2023/2772 de la Commission européenne
    https://eur-lex.europa.eu/eli/reg_del/2023/2772

STRUCTURE DE LA TAXONOMIE :
    ESRS 2   — Informations générales (gouvernance, stratégie, IRO)
    ESRS E1  — Changement climatique
    ESRS E2  — Pollution
    ESRS E3  — Eau et ressources marines
    ESRS E4  — Biodiversité et écosystèmes
    ESRS E5  — Utilisation des ressources et économie circulaire
    ESRS S1  — Effectifs propres (own workforce)
    ESRS S2  — Travailleurs de la chaîne de valeur
    ESRS S3  — Communautés affectées
    ESRS S4  — Consommateurs et utilisateurs finaux
    ESRS G1  — Conduite des affaires
    NONE     — Pas de pertinence CSRD/ESG (classe indispensable, cf. Phase 2 q0)

PRINCIPE DE CONCEPTION :
    La classe "none" est un citoyen de première classe de cette taxonomie, pas une
    exception. Conformément à la stratégie de sampling rigoureuse de la Phase 2
    (stratification sans filtrage sémantique préalable), ~25% du corpus échantillonné
    (quartile q0) n'a pas de pertinence CSRD. Le modèle fine-tuné doit apprendre à
    retourner "none" avec la même confiance qu'il identifie "E1".

VERSIONING :
    CSRD_TAXONOMY_VERSION doit être incrémenté à chaque modification de la taxonomie
    et logué dans chaque ligne annotée (colonne taxonomy_version) pour la
    reproductibilité arXiv — une annotation de juin 2026 doit rester interprétable
    même si la taxonomie évolue en v1.1.

Usage :
    from csrd_taxonomy import (
        CSRD_CATEGORIES, ESRS_SUBCATEGORIES, MATERIALITY_SCALE,
        MARKET_SURPRISE_LEVELS, TIME_HORIZONS,
        validate_annotation, get_category_label, build_taxonomy_prompt_block,
    )

Auto-test :
    python csrd_taxonomy.py
"""

from __future__ import annotations

import json
from typing import Any

# ═════════════════════════════════════════════════════════════════════════════
# VERSIONING — à incrémenter à chaque modification de la taxonomie
# ═════════════════════════════════════════════════════════════════════════════

CSRD_TAXONOMY_VERSION = "1.0"
CSRD_TAXONOMY_REFERENCE = "ESRS Set 1 — Règlement délégué (UE) 2023/2772"

# ═════════════════════════════════════════════════════════════════════════════
# CATÉGORIE "NONE" — classe indispensable (cf. quartile q0, Phase 2)
# ═════════════════════════════════════════════════════════════════════════════

NONE_CATEGORY = "none"

# ═════════════════════════════════════════════════════════════════════════════
# CATÉGORIES PRINCIPALES ESRS
# ═════════════════════════════════════════════════════════════════════════════
# Chaque entrée : code → {label_fr, label_en, pilier, esrs_standard, description}
# "pilier" sert à l'agrégation E / S / G / general / none dans les rapports IAA.

CSRD_CATEGORIES: dict[str, dict[str, str]] = {

    "none": {
        "label_fr":   "Aucune pertinence CSRD",
        "label_en":   "No CSRD relevance",
        "pilier":     "none",
        "esrs_standard": None,
        "description": (
            "Le paragraphe ne traite d'aucun enjeu de durabilité environnemental, "
            "social ou de gouvernance au sens CSRD. Inclut : opérations financières "
            "pures (résultats, dividendes, fusions-acquisitions sans volet ESG), "
            "mentions légales, formalités administratives, déclarations de "
            "transactions sur titres sans lien avec la durabilité."
        ),
    },

    "ESRS2": {
        "label_fr":   "Informations générales (gouvernance, stratégie, IRO)",
        "label_en":   "General disclosures",
        "pilier":     "general",
        "esrs_standard": "ESRS 2",
        "description": (
            "Gouvernance des enjeux de durabilité, modèle d'affaires et stratégie, "
            "processus d'identification des impacts/risques/opportunités (IRO), "
            "double matérialité au niveau de l'entité (hors enjeu sectoriel spécifique)."
        ),
    },

    "E1": {
        "label_fr":   "Changement climatique",
        "label_en":   "Climate change",
        "pilier":     "environment",
        "esrs_standard": "ESRS E1",
        "description": (
            "Plan de transition climatique, émissions de gaz à effet de serre "
            "(Scope 1, 2, 3), consommation et mix énergétique, prix interne du "
            "carbone, effets financiers anticipés des risques climatiques physiques "
            "et de transition."
        ),
    },

    "E2": {
        "label_fr":   "Pollution",
        "label_en":   "Pollution",
        "pilier":     "environment",
        "esrs_standard": "ESRS E2",
        "description": (
            "Pollution de l'air, de l'eau et des sols, substances préoccupantes, "
            "microplastiques, effets financiers anticipés liés à la pollution."
        ),
    },

    "E3": {
        "label_fr":   "Eau et ressources marines",
        "label_en":   "Water and marine resources",
        "pilier":     "environment",
        "esrs_standard": "ESRS E3",
        "description": (
            "Consommation d'eau, prélèvements en zones de stress hydrique, "
            "ressources marines, rejets dans l'eau."
        ),
    },

    "E4": {
        "label_fr":   "Biodiversité et écosystèmes",
        "label_en":   "Biodiversity and ecosystems",
        "pilier":     "environment",
        "esrs_standard": "ESRS E4",
        "description": (
            "Plan de transition biodiversité, impacts sur les écosystèmes, "
            "sites sensibles, métriques de pression sur la biodiversité."
        ),
    },

    "E5": {
        "label_fr":   "Utilisation des ressources et économie circulaire",
        "label_en":   "Resource use and circular economy",
        "pilier":     "environment",
        "esrs_standard": "ESRS E5",
        "description": (
            "Flux entrants/sortants de ressources, déchets, recyclage, "
            "conception circulaire des produits."
        ),
    },

    "S1": {
        "label_fr":   "Effectifs propres",
        "label_en":   "Own workforce",
        "pilier":     "social",
        "esrs_standard": "ESRS S1",
        "description": (
            "Caractéristiques des effectifs, négociation collective, diversité, "
            "rémunération adéquate, santé-sécurité, formation, équilibre vie "
            "professionnelle/personnelle, incidents et plaintes."
        ),
    },

    "S2": {
        "label_fr":   "Travailleurs de la chaîne de valeur",
        "label_en":   "Workers in the value chain",
        "pilier":     "social",
        "esrs_standard": "ESRS S2",
        "description": (
            "Conditions de travail chez les fournisseurs et sous-traitants, "
            "travail des enfants, travail forcé, dialogue avec les travailleurs "
            "de la chaîne de valeur."
        ),
    },

    "S3": {
        "label_fr":   "Communautés affectées",
        "label_en":   "Affected communities",
        "pilier":     "social",
        "esrs_standard": "ESRS S3",
        "description": (
            "Droits des communautés locales, peuples autochtones, accès aux "
            "ressources, déplacement de populations, libertés civiles."
        ),
    },

    "S4": {
        "label_fr":   "Consommateurs et utilisateurs finaux",
        "label_en":   "Consumers and end-users",
        "pilier":     "social",
        "esrs_standard": "ESRS S4",
        "description": (
            "Sécurité des produits, protection de la vie privée, accès à "
            "l'information, pratiques commerciales responsables, inclusion."
        ),
    },

    "G1": {
        "label_fr":   "Conduite des affaires",
        "label_en":   "Business conduct",
        "pilier":     "governance",
        "esrs_standard": "ESRS G1",
        "description": (
            "Culture d'entreprise, protection des lanceurs d'alerte, lutte "
            "anti-corruption, relations fournisseurs, pratiques de paiement, "
            "influence politique et lobbying."
        ),
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# SOUS-CATÉGORIES — Disclosure Requirements (DR) officiels ESRS
# ═════════════════════════════════════════════════════════════════════════════
# Codes officiels du règlement délégué. Permettent une annotation fine pour
# les paragraphes GOLD (annotation humaine) et un signal d'entraînement plus
# riche pour le modèle fine-tuné.

ESRS_SUBCATEGORIES: dict[str, list[dict[str, str]]] = {

    "none": [
        {"code": "none", "label_fr": "Sans objet"},
    ],

    "ESRS2": [
        {"code": "ESRS2-GOV", "label_fr": "Gouvernance des enjeux de durabilité"},
        {"code": "ESRS2-SBM", "label_fr": "Modèle d'affaires et stratégie"},
        {"code": "ESRS2-IRO", "label_fr": "Identification impacts/risques/opportunités"},
        {"code": "ESRS2-MDR", "label_fr": "Exigences de publication minimales"},
    ],

    "E1": [
        {"code": "E1-1", "label_fr": "Plan de transition climatique"},
        {"code": "E1-2", "label_fr": "Politiques liées au climat"},
        {"code": "E1-3", "label_fr": "Actions et ressources climatiques"},
        {"code": "E1-4", "label_fr": "Cibles de réduction GES"},
        {"code": "E1-5", "label_fr": "Consommation et mix énergétique"},
        {"code": "E1-6", "label_fr": "Émissions GES brutes (Scope 1, 2, 3)"},
        {"code": "E1-7", "label_fr": "Absorptions et crédits carbone"},
        {"code": "E1-8", "label_fr": "Prix interne du carbone"},
        {"code": "E1-9", "label_fr": "Effets financiers anticipés (risques physiques/transition)"},
    ],

    "E2": [
        {"code": "E2-1", "label_fr": "Politiques de prévention de la pollution"},
        {"code": "E2-2", "label_fr": "Actions liées à la pollution"},
        {"code": "E2-3", "label_fr": "Cibles de réduction de la pollution"},
        {"code": "E2-4", "label_fr": "Pollution air/eau/sol"},
        {"code": "E2-5", "label_fr": "Substances préoccupantes"},
        {"code": "E2-6", "label_fr": "Effets financiers anticipés"},
    ],

    "E3": [
        {"code": "E3-1", "label_fr": "Politiques eau et ressources marines"},
        {"code": "E3-2", "label_fr": "Actions eau et ressources marines"},
        {"code": "E3-3", "label_fr": "Cibles de consommation d'eau"},
        {"code": "E3-4", "label_fr": "Consommation d'eau"},
        {"code": "E3-5", "label_fr": "Effets financiers anticipés"},
    ],

    "E4": [
        {"code": "E4-1", "label_fr": "Plan de transition biodiversité"},
        {"code": "E4-2", "label_fr": "Politiques biodiversité"},
        {"code": "E4-3", "label_fr": "Actions biodiversité"},
        {"code": "E4-4", "label_fr": "Cibles biodiversité"},
        {"code": "E4-5", "label_fr": "Métriques d'impact biodiversité"},
    ],

    "E5": [
        {"code": "E5-1", "label_fr": "Politiques ressources et économie circulaire"},
        {"code": "E5-2", "label_fr": "Actions économie circulaire"},
        {"code": "E5-3", "label_fr": "Cibles économie circulaire"},
        {"code": "E5-4", "label_fr": "Flux entrants de ressources"},
        {"code": "E5-5", "label_fr": "Flux sortants de ressources et déchets"},
        {"code": "E5-6", "label_fr": "Effets financiers anticipés"},
    ],

    "S1": [
        {"code": "S1-1",  "label_fr": "Politiques effectifs propres"},
        {"code": "S1-2",  "label_fr": "Dialogue avec les effectifs"},
        {"code": "S1-3",  "label_fr": "Mécanismes de remédiation"},
        {"code": "S1-4",  "label_fr": "Actions effectifs propres"},
        {"code": "S1-5",  "label_fr": "Cibles effectifs propres"},
        {"code": "S1-6",  "label_fr": "Caractéristiques des effectifs"},
        {"code": "S1-7",  "label_fr": "Travailleurs non-salariés"},
        {"code": "S1-8",  "label_fr": "Couverture par négociation collective"},
        {"code": "S1-9",  "label_fr": "Indicateurs de diversité"},
        {"code": "S1-10", "label_fr": "Rémunération adéquate"},
        {"code": "S1-11", "label_fr": "Protection sociale"},
        {"code": "S1-12", "label_fr": "Inclusion du handicap"},
        {"code": "S1-13", "label_fr": "Formation et développement des compétences"},
        {"code": "S1-14", "label_fr": "Santé et sécurité"},
        {"code": "S1-15", "label_fr": "Équilibre vie professionnelle/personnelle"},
        {"code": "S1-16", "label_fr": "Indicateurs de rémunération"},
        {"code": "S1-17", "label_fr": "Incidents, plaintes et impacts graves"},
    ],

    "S2": [
        {"code": "S2-1", "label_fr": "Politiques travailleurs chaîne de valeur"},
        {"code": "S2-2", "label_fr": "Dialogue avec la chaîne de valeur"},
        {"code": "S2-3", "label_fr": "Mécanismes de remédiation"},
        {"code": "S2-4", "label_fr": "Actions chaîne de valeur"},
        {"code": "S2-5", "label_fr": "Cibles chaîne de valeur"},
    ],

    "S3": [
        {"code": "S3-1", "label_fr": "Politiques communautés affectées"},
        {"code": "S3-2", "label_fr": "Dialogue avec les communautés"},
        {"code": "S3-3", "label_fr": "Mécanismes de remédiation"},
        {"code": "S3-4", "label_fr": "Actions communautés affectées"},
        {"code": "S3-5", "label_fr": "Cibles communautés affectées"},
    ],

    "S4": [
        {"code": "S4-1", "label_fr": "Politiques consommateurs"},
        {"code": "S4-2", "label_fr": "Dialogue avec les consommateurs"},
        {"code": "S4-3", "label_fr": "Mécanismes de remédiation"},
        {"code": "S4-4", "label_fr": "Actions consommateurs"},
        {"code": "S4-5", "label_fr": "Cibles consommateurs"},
    ],

    "G1": [
        {"code": "G1-1", "label_fr": "Culture d'entreprise et conduite des affaires"},
        {"code": "G1-2", "label_fr": "Relations avec les fournisseurs"},
        {"code": "G1-3", "label_fr": "Prévention et détection de la corruption"},
        {"code": "G1-4", "label_fr": "Incidents de corruption"},
        {"code": "G1-5", "label_fr": "Influence politique et lobbying"},
        {"code": "G1-6", "label_fr": "Pratiques de paiement"},
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# ÉCHELLE DE MATÉRIALITÉ (0–5)
# ═════════════════════════════════════════════════════════════════════════════
# 0 réservé à csrd_category = "none". Pour toute autre catégorie, 1 à 5.

MATERIALITY_SCALE: dict[int, dict[str, str]] = {
    0: {
        "label_fr": "Non applicable",
        "description": "Réservé aux paragraphes csrd_category = 'none'.",
    },
    1: {
        "label_fr": "Négligeable",
        "description": (
            "Mention factuelle ou descriptive sans implication significative "
            "sur la performance financière ou les impacts de durabilité."
        ),
    },
    2: {
        "label_fr": "Mineure",
        "description": (
            "Information pertinente mais d'ampleur limitée, n'affecte pas "
            "significativement la trajectoire de l'entreprise."
        ),
    },
    3: {
        "label_fr": "Modérée",
        "description": (
            "Enjeu identifié comme pertinent dans l'analyse de double "
            "matérialité, avec un effet mesurable mais non déterminant."
        ),
    },
    4: {
        "label_fr": "Significative",
        "description": (
            "Impact direct sur la performance financière OU impact "
            "environnemental/social substantiel et documenté."
        ),
    },
    5: {
        "label_fr": "Critique",
        "description": (
            "Enjeu central pour le modèle d'affaires : risque majeur, "
            "controverse significative, ou transformation structurelle "
            "de l'activité liée à la durabilité."
        ),
    },
}

# ── Dimensions de la double matérialité (booléens indépendants) ──────────────
MATERIALITY_DIMENSIONS = {
    "materialite_financiere": (
        "L'enjeu a/aura un effet sur la situation financière, la performance "
        "ou les flux de trésorerie de l'entreprise (matérialité 'outside-in')."
    ),
    "materialite_impact": (
        "L'entreprise a/aura un effet réel ou potentiel sur les personnes ou "
        "l'environnement, positif ou négatif (matérialité 'inside-out')."
    ),
}

# ═════════════════════════════════════════════════════════════════════════════
# NIVEAUX DE SURPRISE DE MARCHÉ
# ═════════════════════════════════════════════════════════════════════════════

MARKET_SURPRISE_LEVELS: dict[str, dict[str, str]] = {
    "aucune": {
        "label_fr": "Aucune surprise",
        "description": (
            "Information de routine, pleinement anticipée (ex : publication "
            "périodique conforme au calendrier réglementaire)."
        ),
    },
    "faible": {
        "label_fr": "Surprise faible",
        "description": (
            "Information globalement attendue, avec des nuances mineures "
            "par rapport au consensus ou aux communications précédentes."
        ),
    },
    "partielle": {
        "label_fr": "Surprise partielle",
        "description": (
            "Information partiellement anticipée mais d'ampleur supérieure "
            "ou de nature différente de ce qui était attendu."
        ),
    },
    "forte": {
        "label_fr": "Surprise forte",
        "description": (
            "Information non anticipée, rupture avec la communication "
            "précédente ou le consensus de marché (ex : profit warning, "
            "incident majeur, changement de gouvernance soudain)."
        ),
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# HORIZON TEMPOREL
# ═════════════════════════════════════════════════════════════════════════════

TIME_HORIZONS: dict[str, dict[str, str]] = {
    "immediat": {
        "label_fr": "Immédiat",
        "description": "Effet déjà constaté ou se matérialisant sous 12 mois.",
    },
    "court_terme": {
        "label_fr": "Court terme",
        "description": "Effet attendu entre 1 et 3 ans.",
    },
    "moyen_terme": {
        "label_fr": "Moyen terme",
        "description": "Effet attendu entre 3 et 10 ans (horizon ESRS standard).",
    },
    "long_terme": {
        "label_fr": "Long terme",
        "description": "Effet attendu au-delà de 10 ans.",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# SCHÉMA JSON DE LA RÉPONSE FINALE (final_answer_json)
# ═════════════════════════════════════════════════════════════════════════════
# Utilisé pour :
#   - response_format structuré dans les appels API Mistral
#   - validation programmatique de chaque annotation (humaine ou Mistral)

FINAL_ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "csrd_category", "materialite_score",
        "materialite_financiere", "materialite_impact",
        "market_surprise", "horizon_temporel",
    ],
    "properties": {
        "csrd_category": {
            "type": "string",
            "enum": list(CSRD_CATEGORIES.keys()),
            "description": "Catégorie ESRS principale, ou 'none'.",
        },
        "esrs_subcategory": {
            "type": ["string", "null"],
            "description": (
                "Code de disclosure requirement précis (ex: 'E1-6'). "
                "Null si csrd_category == 'none'."
            ),
        },
        "materialite_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "0 si none, 1-5 sinon.",
        },
        "materialite_financiere": {
            "type": "boolean",
        },
        "materialite_impact": {
            "type": "boolean",
        },
        "market_surprise": {
            "type": "string",
            "enum": list(MARKET_SURPRISE_LEVELS.keys()),
        },
        "horizon_temporel": {
            "type": ["string", "null"],
            "enum": list(TIME_HORIZONS.keys()) + [None],
            "description": "Null si csrd_category == 'none'.",
        },
        "chain_of_thought": {
            "type": "string",
            "description": (
                "Raisonnement étape par étape : (1) nature de l'annonce, "
                "(2) analyse de matérialité, (3) évaluation de la surprise."
            ),
        },
        "confiance_annotation": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confiance de l'annotateur (humain ou modèle).",
        },
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_annotation(annotation: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Valide une annotation complète (humaine ou Mistral) contre la taxonomie.
    Retourne (is_valid, liste_erreurs).

    Utilisé par :
      - export_gold_for_annotation.py  : validation post-saisie Excel
      - annotation_mistral_mass.py     : validation de chaque réponse API
                                          (re-prompt si invalide)
      - validation_iaa.py              : filtre les annotations malformées
                                          avant calcul du Kappa
    """
    errors: list[str] = []

    # ── Champ obligatoire : csrd_category ────────────────────────────────────
    category = annotation.get("csrd_category")
    if category not in CSRD_CATEGORIES:
        errors.append(
            f"csrd_category invalide : '{category}' "
            f"(attendu un de {list(CSRD_CATEGORIES.keys())})"
        )
        return False, errors   # erreur bloquante, inutile de continuer

    # ── Cohérence avec la catégorie "none" ───────────────────────────────────
    is_none = (category == NONE_CATEGORY)

    materialite_score = annotation.get("materialite_score")
    if not isinstance(materialite_score, int) or not (0 <= materialite_score <= 5):
        errors.append(f"materialite_score invalide : {materialite_score} (attendu 0-5)")
    elif is_none and materialite_score != 0:
        errors.append(
            f"materialite_score doit être 0 quand csrd_category='none' "
            f"(reçu {materialite_score})"
        )
    elif not is_none and materialite_score == 0:
        errors.append(
            "materialite_score=0 incohérent avec csrd_category != 'none' "
            "(utiliser 1-5 ou repasser csrd_category à 'none')"
        )

    # ── Sous-catégorie ────────────────────────────────────────────────────────
    subcategory = annotation.get("esrs_subcategory")
    valid_subcodes = {s["code"] for s in ESRS_SUBCATEGORIES.get(category, [])}
    if is_none:
        if subcategory not in (None, "none"):
            errors.append("esrs_subcategory doit être null quand csrd_category='none'")
    else:
        if subcategory not in valid_subcodes:
            errors.append(
                f"esrs_subcategory '{subcategory}' invalide pour la catégorie "
                f"'{category}' (attendu un de {sorted(valid_subcodes)})"
            )

    # ── Booléens de double matérialité ───────────────────────────────────────
    for field_name in ("materialite_financiere", "materialite_impact"):
        val = annotation.get(field_name)
        if not isinstance(val, bool):
            errors.append(f"{field_name} doit être un booléen (reçu {val!r})")
        elif is_none and val is True:
            errors.append(f"{field_name} doit être False quand csrd_category='none'")

    # ── Surprise de marché ────────────────────────────────────────────────────
    surprise = annotation.get("market_surprise")
    if surprise not in MARKET_SURPRISE_LEVELS:
        errors.append(
            f"market_surprise invalide : '{surprise}' "
            f"(attendu un de {list(MARKET_SURPRISE_LEVELS.keys())})"
        )

    # ── Horizon temporel ──────────────────────────────────────────────────────
    horizon = annotation.get("horizon_temporel")
    if is_none:
        if horizon is not None:
            errors.append("horizon_temporel doit être null quand csrd_category='none'")
    else:
        if horizon not in TIME_HORIZONS:
            errors.append(
                f"horizon_temporel invalide : '{horizon}' "
                f"(attendu un de {list(TIME_HORIZONS.keys())})"
            )

    # ── Chain of thought non vide (sauf none, où une courte justification suffit) ──
    cot = annotation.get("chain_of_thought", "")
    if not isinstance(cot, str) or len(cot.strip()) < 10:
        errors.append("chain_of_thought trop court ou absent (minimum 10 caractères)")

    # ── Confiance ─────────────────────────────────────────────────────────────
    confiance = annotation.get("confiance_annotation")
    if confiance is not None:
        if not isinstance(confiance, (int, float)) or not (0.0 <= confiance <= 1.0):
            errors.append(f"confiance_annotation invalide : {confiance} (attendu 0.0-1.0)")

    return (len(errors) == 0), errors


def validate_batch(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Valide un lot d'annotations (ex: sortie complète de annotation_mistral_mass.py).
    Retourne un rapport agrégé utile pour le run_meta du papier.
    """
    n_valid, n_invalid = 0, 0
    errors_by_type: dict[str, int] = {}
    invalid_examples: list[dict[str, Any]] = []

    for i, ann in enumerate(annotations):
        is_valid, errors = validate_annotation(ann)
        if is_valid:
            n_valid += 1
        else:
            n_invalid += 1
            for err in errors:
                key = err.split(" :")[0].split(" doit")[0].strip()
                errors_by_type[key] = errors_by_type.get(key, 0) + 1
            if len(invalid_examples) < 10:
                invalid_examples.append({"index": i, "errors": errors})

    return {
        "total":            len(annotations),
        "n_valid":          n_valid,
        "n_invalid":        n_invalid,
        "validity_rate_pct": round(n_valid / max(1, len(annotations)) * 100, 2),
        "errors_by_type":   errors_by_type,
        "invalid_examples": invalid_examples,
    }

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — accès pratique à la taxonomie
# ═════════════════════════════════════════════════════════════════════════════

def get_category_label(code: str, lang: str = "fr") -> str:
    """Retourne le libellé humain d'un code catégorie."""
    entry = CSRD_CATEGORIES.get(code)
    if not entry:
        return f"[Catégorie inconnue : {code}]"
    return entry["label_fr"] if lang == "fr" else entry["label_en"]


def get_subcategories(category_code: str) -> list[dict[str, str]]:
    """Retourne la liste des sous-catégories valides pour une catégorie donnée."""
    return ESRS_SUBCATEGORIES.get(category_code, [])


def is_valid_category(code: str) -> bool:
    return code in CSRD_CATEGORIES


def is_valid_subcategory(category_code: str, subcode: str) -> bool:
    valid = {s["code"] for s in ESRS_SUBCATEGORIES.get(category_code, [])}
    return subcode in valid


def list_all_category_codes() -> list[str]:
    return list(CSRD_CATEGORIES.keys())


def list_all_subcategory_codes() -> list[str]:
    """Tous les codes de sous-catégories, toutes catégories confondues (pour validation)."""
    codes = []
    for subs in ESRS_SUBCATEGORIES.values():
        codes.extend(s["code"] for s in subs)
    return codes


def get_pilier(category_code: str) -> str:
    """Retourne le pilier (environment/social/governance/general/none) d'une catégorie."""
    entry = CSRD_CATEGORIES.get(category_code, {})
    return entry.get("pilier", "unknown")


def empty_annotation_template() -> dict[str, Any]:
    """Gabarit vide conforme au schéma — utile pour initialiser une ligne d'annotation."""
    return {
        "csrd_category":          None,
        "esrs_subcategory":       None,
        "materialite_score":      None,
        "materialite_financiere": None,
        "materialite_impact":     None,
        "market_surprise":        None,
        "horizon_temporel":       None,
        "chain_of_thought":       "",
        "confiance_annotation":   None,
        "annotateur":             None,
        "taxonomy_version":       CSRD_TAXONOMY_VERSION,
    }

# ═════════════════════════════════════════════════════════════════════════════
# BLOC DE PROMPT — injecté dans annotation_mistral_mass.py
# ═════════════════════════════════════════════════════════════════════════════

def build_taxonomy_prompt_block() -> str:
    """
    Génère le bloc de texte décrivant la taxonomie complète, à insérer dans le
    system prompt de Mistral Large. Garantit que le prompt et la validation
    Python utilisent EXACTEMENT la même source de vérité (pas de divergence
    entre ce que le modèle voit et ce qui est validé).
    """
    lines = [
        f"TAXONOMIE CSRD/ESRS (version {CSRD_TAXONOMY_VERSION}, "
        f"réf. {CSRD_TAXONOMY_REFERENCE})",
        "",
        "Catégories disponibles pour csrd_category :",
    ]
    for code, info in CSRD_CATEGORIES.items():
        lines.append(f"  • {code} — {info['label_fr']} : {info['description']}")

    lines.append("")
    lines.append("Échelle de matérialité (materialite_score) :")
    for score, info in MATERIALITY_SCALE.items():
        lines.append(f"  {score} = {info['label_fr']} : {info['description']}")

    lines.append("")
    lines.append("Niveaux de surprise de marché (market_surprise) :")
    for code, info in MARKET_SURPRISE_LEVELS.items():
        lines.append(f"  • {code} : {info['description']}")

    lines.append("")
    lines.append("Horizons temporels (horizon_temporel) :")
    for code, info in TIME_HORIZONS.items():
        lines.append(f"  • {code} : {info['description']}")

    lines.append("")
    lines.append(
        "RÈGLE CRITIQUE : si csrd_category = 'none', alors esrs_subcategory=null, "
        "materialite_score=0, materialite_financiere=false, materialite_impact=false, "
        "horizon_temporel=null. Ne forcez JAMAIS une catégorie CSRD sur un paragraphe "
        "purement financier ou administratif sans dimension de durabilité."
    )

    return "\n".join(lines)


def export_taxonomy_markdown() -> str:
    """
    Génère une table Markdown de la taxonomie complète — utilisable telle quelle
    dans la Dataset Card HuggingFace ou en annexe du papier arXiv.
    """
    lines = [
        f"# Taxonomie CSRD/ESRS — FraFin-Reasoning (v{CSRD_TAXONOMY_VERSION})",
        f"\nRéférence : {CSRD_TAXONOMY_REFERENCE}\n",
        "| Code | Pilier | Libellé (FR) | Sous-catégories |",
        "|------|--------|--------------|------------------|",
    ]
    for code, info in CSRD_CATEGORIES.items():
        n_sub = len(ESRS_SUBCATEGORIES.get(code, []))
        lines.append(
            f"| `{code}` | {info['pilier']} | {info['label_fr']} | {n_sub} |"
        )
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════════════
# AUTO-TEST — validation de cohérence interne au chargement du module
# ═════════════════════════════════════════════════════════════════════════════

def _self_check() -> None:
    """Vérifie la cohérence interne de la taxonomie (exécuté en mode test direct)."""
    # Toute catégorie (sauf none) doit avoir au moins une sous-catégorie
    for code in CSRD_CATEGORIES:
        if code == NONE_CATEGORY:
            continue
        assert code in ESRS_SUBCATEGORIES, f"Catégorie '{code}' sans sous-catégories définies"
        assert len(ESRS_SUBCATEGORIES[code]) > 0, f"Catégorie '{code}' a une liste vide"

    # Le schéma JSON doit référencer des enums cohérents avec les dicts
    schema_categories = set(FINAL_ANSWER_JSON_SCHEMA["properties"]["csrd_category"]["enum"])
    assert schema_categories == set(CSRD_CATEGORIES.keys()), (
        "Désynchronisation entre FINAL_ANSWER_JSON_SCHEMA et CSRD_CATEGORIES"
    )

    # Exemple de validation positive et négative (non-regression basique)
    valid_example = {
        "csrd_category": "E1",
        "esrs_subcategory": "E1-6",
        "materialite_score": 4,
        "materialite_financiere": True,
        "materialite_impact": True,
        "market_surprise": "partielle",
        "horizon_temporel": "moyen_terme",
        "chain_of_thought": "Le paragraphe mentionne une hausse des émissions Scope 1.",
        "confiance_annotation": 0.9,
    }
    ok, errs = validate_annotation(valid_example)
    assert ok, f"L'exemple valide a échoué la validation : {errs}"

    invalid_example = dict(valid_example)
    invalid_example["csrd_category"] = "none"
    # incohérent : materialite_score=4 mais category=none
    ok2, errs2 = validate_annotation(invalid_example)
    assert not ok2, "L'exemple incohérent (none + score 4) aurait dû échouer"

    print("  ✅ Auto-check de cohérence interne : OK")


if __name__ == "__main__":
    print("═" * 65)
    print(f"  csrd_taxonomy.py — v{CSRD_TAXONOMY_VERSION}")
    print(f"  {CSRD_TAXONOMY_REFERENCE}")
    print("═" * 65)

    _self_check()

    print(f"\n  Catégories principales : {len(CSRD_CATEGORIES)}")
    for code, info in CSRD_CATEGORIES.items():
        n_sub = len(ESRS_SUBCATEGORIES.get(code, []))
        print(f"    {code:<8} ({info['pilier']:<11}) {info['label_fr']:<45} "
              f"[{n_sub} sous-cat.]")

    print(f"\n  Total sous-catégories  : {len(list_all_subcategory_codes())}")
    print(f"  Échelle matérialité    : 0–{max(MATERIALITY_SCALE.keys())}")
    print(f"  Niveaux surprise marché: {list(MARKET_SURPRISE_LEVELS.keys())}")
    print(f"  Horizons temporels     : {list(TIME_HORIZONS.keys())}")

    print(f"\n  Exemple — gabarit vide :")
    print(f"  {json.dumps(empty_annotation_template(), indent=2, ensure_ascii=False)}")

    print(f"\n  Aperçu du bloc de prompt généré pour Mistral :")
    print("  " + "─" * 61)
    block = build_taxonomy_prompt_block()
    for line in block.split("\n")[:8]:
        print(f"  {line}")
    print(f"  ... ({len(block.split(chr(10)))} lignes au total)")

    print("\n" + "═" * 65)
    print("  Module prêt à être importé par :")
    print("    export_gold_for_annotation.py")
    print("    annotation_mistral_mass.py")
    print("    validation_iaa.py")
    print("═" * 65)
