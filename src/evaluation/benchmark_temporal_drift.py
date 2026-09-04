"""
benchmark_temporal_drift.py — Évaluation Multi-LLM du Biais de Récence Temporelle
═══════════════════════════════════════════════════════════════════════════════════════
Contribution centrale du papier :
    "Temporal Regulatory Semantic Drift: How LLMs Fail on 16 Years
     of French Financial Disclosures — and How to Fix It"

HYPOTHÈSE TESTÉE (H1) :
    Les LLM sous-classifient systématiquement les documents réglementaires
    antérieurs à 2019 comme 'none' — même lorsque le contenu décrit un enjeu
    de durabilité réel — parce qu'ils ne reconnaissent pas le vocabulaire
    pré-CSRD. La dérive est mesurable via la chute monotone du F1-binaire
    sur les 4 époques (2010-2014 → 2015-2019 → 2020-2022 → 2023-2026).

PROTOCOLE :
    - 137 paragraphes (Gold 140 moins 3 réservés au pool few-shot)
    - 4 modèles : Mistral-7B / Mistral-Large / GPT-4o / Claude-4.6-Sonnet
    - 2 modes : zero-shot (--n-shot 0) et 3-shot (--n-shot 3)
    - température = 0 / random_seed = 42 partout
    - Checkpoint par modèle → résume après crash

MÉTRIQUES (par modèle, par mode, par époque) :
    - F1 binaire (CSRD vs none) → métrique principale de dérive temporelle
    - Kappa de Cohen binaire + bootstrap CI 95%
    - ECE (Expected Calibration Error, confiance auto-rapportée)
    - Accuracy multi-classe
    - Macro-F1 multi-classe (catégories avec n ≥ 5)
    - TD (Temporal Drift) = F1(2023-2026) − F1(2010-2014) par modèle
    - β_drift (pente de régression linéaire F1 ~ epoch_index) par modèle

SORTIES :
    results/raw_predictions_{mode}_{RUN_TS}.csv    ← toutes les prédictions
    results/metrics_{mode}_{RUN_TS}.json           ← métriques complètes
    results/latex_tables_{mode}_{RUN_TS}.tex        ← tables LaTeX copier-coller
    results/checkpoint_{model_id}.json             ← reprise après crash

MODÈLES ET JUSTIFICATION SCIENTIFIQUE :
    Groupe 1 — Contrôle intra-famille (scaling law) :
        mistral-7b-instruct-v0.3  :  7B, français natif, open-weights
        mistral-large-latest      :  ~123B, même famille, même tokenizer
        → isoler l'effet de l'échelle des paramètres SANS changer l'architecture

    Groupe 2 — Frontier cross-architecture :
        gpt-4o                    :  référence anglophone dominante
        claude-sonnet-4-6         :  référence raisonnement structuré

Dépendances :
    pip install pandas numpy scikit-learn scipy openai anthropic mistralai pyarrow openpyxl


Usage :
    python benchmark_temporal_drift.py --n-shot 0           # zero-shot (recommandé en premier)
    python benchmark_temporal_drift.py --n-shot 3           # 3-shot
    python benchmark_temporal_drift.py --n-shot 0 --resume  # reprendre après crash
    python benchmark_temporal_drift.py --n-shot 0 --limit 10 --model mistral-7b  # test rapide
    python benchmark_temporal_drift.py --report-only        # régénère tables depuis CSV existant
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    f1_score, precision_score, recall_score,
)


import warnings
warnings.filterwarnings("ignore")


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Variable d'environnement {name} requise. "
            f"Copiez .env.example vers .env et renseignez vos clés API."
        )
    return value


MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════


GOLD_FILE = "data/gold_150_annotated_clean_reformulated_without.xlsx"

RESULTS_DIR  = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

RUN_TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
TEMPERATURE  = 0.0
RANDOM_SEED  = 42
MAX_TOKENS   = 256
RETRY_WAIT   = 60      # secondes d'attente sur rate-limit
MAX_RETRIES  = 4

EPOCH_ORDER  = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]
EPOCH_INDEX  = {e: i for i, e in enumerate(EPOCH_ORDER)}

# Catégories valides (depuis csrd_taxonomy.py — inline pour éviter la dépendance)
VALID_CATEGORIES = [
    "none", "ESRS2", "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4", "G1",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# MODÈLES ET ROUTING
# ═════════════════════════════════════════════════════════════════════════════

MODELS = {
    "mistral-7b": {
        "provider":   "mistral",
        "model_id":   "open-mistral-7b",
        "group":      "7B (Mistral — français natif)",
        "params_b":   7,
    },
    "mistral-large": {
        "provider":   "mistral",
        "model_id":   "mistral-large-latest",
        "group":      "Large (Mistral — même famille)",
        "params_b":   None,   # MoE, non divulgué
    },
    "gpt-4o": {
        "provider":   "openai",
        "model_id":   "gpt-4o",
        "group":      "Frontier (OpenAI)",
        "params_b":   None,
    },
   "claude-4-6-sonnet": {
    "provider":   "anthropic",
    "model_id":   "claude-sonnet-4-6",   # ← remplace claude-3-5-sonnet-20241022
    "group":      "Frontier (Anthropic)",
    "params_b":   None,
},
}

# ═════════════════════════════════════════════════════════════════════════════
# EXEMPLES FEW-SHOT (tirés du Gold, exclus du test set)
# Ces 3 exemples couvrent les 3 catégories dominantes (none, E1, ESRS2)
# sur 3 époques différentes → maximisent la couverture contextuelle.
# Leurs paragraph_id sont documentés pour reproductibilité.
# ═════════════════════════════════════════════════════════════════════════════

FEW_SHOT_IDS = set()   # rempli lors du chargement

FEW_SHOT_EXAMPLES = [
    # #101 | 2023-2026 | STMICROELECTRONICS NV | none
    {

        "content": (
            "Pour de plus amples informations sur les engagements de ST dans les domaines "
            "de la transition énergétique et du changement climatique, cliquez ici."
        ),
        "answer": {
            "csrd_category": "none",
            "esrs_subcategory": "",
            "reasoning": (
                "Ce paragraphe, extrait d'un document « Informations privilégiées » de STMicroelectronics, "
                "évoque l'engagement du groupe vis-à-vis de l'Accord de Paris ou de la lutte contre le "
                "changement climatique. Le paragraphe ne relève pas de la CSRD/ESG : il s'agit d'une "
                "information administrative ou financière générique, sans mention d'un enjeu environnemental, "
                "social ou de gouvernance durable, et ne mentionne pas de dimension de durabilité. N'ayant pas "
                "de portée ESG, ce paragraphe ne constitue pas une surprise pour les investisseurs au sens de "
                "la matérialité de durabilité."
            ),
        },
    },

    # #23 | 2015-2019 | TOTAL S.A. | E1
    {


        "content": (
            "- L’Accord de Paris : Total reconnaît l'Accord de Paris comme une avancée majeure dans la lutte "
            "contre le réchauffement climatique et soutient les initiatives des États parties prenantes pour "
            "atteindre les objectifs de cet accord."
        ),
        "answer": {
            "csrd_category": "E1",
            "esrs_subcategory": "E1-1",
            "reasoning": (
                "Chez Total, cet extrait (Informations privilégiées) évoque l'engagement du groupe vis-à-vis "
                "de l'Accord de Paris ou de la lutte contre le changement climatique. Le paragraphe relève de "
                "la CSRD/ESG (catégorie E1 — Changement climatique), avec une matérialité notée 3/5 (modérée) "
                "car il traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et "
                "d'impact (conséquences sur l'environnement ou la société) ; il mentionne explicitement la "
                "trajectoire climatique du groupe (réduction d'émissions, transition énergétique). Cette "
                "information ne représente pas une surprise pour les investisseurs : il s'agit d'une information "
                "de routine, pleinement anticipée par le marché."
            ),
        },
    },

    # #57 | 2023-2026 | TOTALENERGIES SE | E2
    {
        "content": (
            "En Inde, le gaz naturel jouera un rôle essentiel dans la transition énergétique. En tant que "
            "carburant alternatif plus propre pour les activités industrielles, la cuisson, et le transport, "
            "il contribue à réduire les émissions de gaz à effet de serre et la pollution, améliorant ainsi "
            "la qualité de l'air."
        ),
        "answer": {
            "csrd_category": "E2",
            "esrs_subcategory": "E2-1",
            "reasoning": (
                "Dans ce document « Informations privilégiées » publié par TotalEnergies, le paragraphe aborde "
                "la réduction de la pollution générée par les activités du groupe. Le paragraphe relève de la "
                "CSRD/ESG (catégorie E2 — Pollution), avec une matérialité notée 4/5 (significative) car il "
                "traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et d'impact "
                "(conséquences sur l'environnement ou la société) ; il mentionne explicitement la réduction de "
                "la pollution générée par les activités du groupe. Du point de vue des investisseurs, la surprise "
                "est faible, car information globalement attendue, avec des nuances mineures par rapport au consensus."
            ),
        },
    },

    # #53 | 2015-2019 | VEOLIA ENVIRONNEMENT | E5
    {
        "content": (
            "Recyclage et Valorisation des Déchets / France Veolia renouvelle auprès du SMEDAR son contrat lié "
            "à l’exploitation de l’unité de valorisation VESTA à Rouen d’une durée de 6 ans et demi pour un "
            "montant de 116 millions d’euros Le Syndicat Mixte d’Elimination des Déchets de l’Arrondissement de "
            "Rouen (SMEDAR) a renouvelé sa confiance à Veolia, après mise en concurrence, pour l’exploitation de "
            "l’Usine de valorisation énergétique VESTA, dans le cadre d’un marché de 6 ans et demi pour un montant "
            "global de 116 M€. Points forts de ce nouveau marché : l’optimisation opérationnelle de l’usine et la "
            "fourniture au SMEDAR d’outils de contrôle lui permettant de maîtriser une unité désormais partie "
            "intégrante du mix énergétique territorial."
        ),
        "answer": {
            "csrd_category": "E5",
            "esrs_subcategory": "E5-2",
            "reasoning": (
                "Dans ce document « Informations privilégiées » publié par Veolia, le paragraphe porte sur la "
                "gestion des déchets ou l'économie circulaire. Le paragraphe relève de la CSRD/ESG (catégorie E5 — "
                "Utilisation des ressources et économie circulaire), avec une matérialité notée 3/5 (modérée) car "
                "il traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et d'impact "
                "(conséquences sur l'environnement ou la société) ; il mentionne explicitement l'économie circulaire "
                "et la gestion des déchets. Pour le marché, il s'agit d'une surprise partielle : information "
                "partiellement anticipée mais d'une ampleur ou d'une nature différente de ce qui était attendu."
            ),
        },
    },

    # #34 | 2023-2026 | SCHNEIDER ELECTRIC SE | S1
    {
        
        "content": (
            "Nous sommes une entreprise humaine rassemblant un écosystème de 150 000 collaborateurs et de plus "
            "d'un million de partenaires dans plus de 100 pays au plus proche de nos clients et de nos parties "
            "prenantes. Nous plaçons la diversité et l'inclusion au cœur de tout ce que nous faisons, guidés par "
            "notre volonté profonde de contribuer à un futur durable pour tous."
        ),
        "answer": {
            "csrd_category": "S1",
            "esrs_subcategory": "S1-3",
            "reasoning": (
                "Il s'agit d'un extrait de Schneider Electric (Acquisition ou cession des actions de l'émetteur) "
                "qui présente les engagements du groupe en matière de diversité, d'inclusion ou de conditions de "
                "travail des collaborateurs (texte dégradé par la reconnaissance OCR). Le paragraphe relève de la "
                "CSRD/ESG (catégorie S1 — Effectifs propres), avec une matérialité notée 2/5 (mineure) car il "
                "traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et d'impact "
                "(conséquences sur l'environnement ou la société) ; il mentionne explicitement les conditions de "
                "travail, la diversité et l'inclusion au sein des effectifs. Du point de vue des investisseurs, "
                "la surprise est faible, car information globalement attendue, avec des nuances mineures par rapport "
                "au consensus."
            ),
        },
    },

    # #63 | 2020-2022 | BOUYGUES | S4
    {
      
        "content": (
            "transitions environnementale, industrielle et numérique, pour œuvrer en faveur d’une croissance plus "
            "durable et énergétiquement plus sobre, et enfin pour accompagner nos clients dans leur transition vers "
            "une activité bas carbone. Nous avons tous les atouts pour transformer cette acquisition en un succès "
            "pour l’ensemble des parties prenantes du Groupe, notamment ses collaborateurs, ses clients et ses "
            "actionnaires. »"
        ),
        "answer": {
            "csrd_category": "S4",
            "esrs_subcategory": "S4-4",
            "reasoning": (
                "Dans ce document « Informations privilégiées » publié par Bouygues, le paragraphe aborde la "
                "protection des consommateurs et des pratiques commerciales responsables. Le paragraphe relève de "
                "la CSRD/ESG (catégorie S4 — Consommateurs et utilisateurs finaux), avec une matérialité notée 1/5 "
                "(négligeable) car il traduit un effet à la fois financier (risques/opportunités pour l'entreprise) "
                "et d'impact (conséquences sur l'environnement ou la société) ; il mentionne explicitement la "
                "protection des consommateurs et des pratiques commerciales responsables. Cette information ne "
                "représente pas une surprise pour les investisseurs : il s'agit d'une information de routine, "
                "pleinement anticipée par le marché."
            ),
        },
    },

    # #38 | 2023-2026 | ENGIE | ESRS2
    {
        "categorie": "ESRS2",
        "content": (
            "- Enfin, la stratégie climatique du Groupe inclut désormais également l’adaptation au changement "
            "climatique afin d’assurer l'intégrité des actifs et des chaînes d'approvisionnement résilientes."
        ),
        "answer": {
            "csrd_category": "ESRS2",
            "esrs_subcategory": "ESRS2-SBM",
            "reasoning": (
                "Ce paragraphe, extrait d'un document « Informations privilégiées » de Engie, aborde la stratégie "
                "ou la gouvernance des enjeux de durabilité de l'entreprise. Le paragraphe relève de la CSRD/ESG "
                "(catégorie ESRS2 — Informations générales (gouvernance, stratégie, IRO)), avec une matérialité "
                "notée 3/5 (modérée) car il traduit un effet à la fois financier (risques/opportunités pour "
                "l'entreprise) et d'impact (conséquences sur l'environnement ou la société) ; il mentionne "
                "explicitement la stratégie ou la gouvernance des enjeux de durabilité de l'entreprise. Pour le "
                "marché, il s'agit d'une surprise partielle : information partiellement anticipée mais d'une ampleur "
                "ou d'une nature différente de ce qui était attendu."
            ),
        },
    },
]

# ═════════════════════════════════════════════════════════════════════════════
# PROMPT SYSTÈME
# ═════════════════════════════════════════════════════════════════════════════


TAXONOMY_BLOCK = """

CATÉGORIES DISPONIBLES (12) :

• none — Aucune pertinence CSRD
  Le paragraphe ne traite d'aucun enjeu de durabilité environnemental,
  social ou de gouvernance au sens CSRD. Inclut : opérations financières
  pures (résultats, dividendes, fusions-acquisitions sans volet ESG),
  mentions légales, formalités administratives, déclarations de
  transactions sur titres sans lien avec la durabilité.
  Sous-catégories : aucune.

• ESRS2 — Informations générales (gouvernance, stratégie, IRO)
  Gouvernance des enjeux de durabilité, modèle d'affaires et stratégie,
  processus d'identification des impacts/risques/opportunités (IRO),
  double matérialité au niveau de l'entité (hors enjeu sectoriel spécifique).
  Sous-catégories : ESRS2-GOV, ESRS2-SBM, ESRS2-IRO, ESRS2-MDR.

• E1 — Changement climatique
  Plan de transition climatique, émissions de gaz à effet de serre
  (Scope 1, 2, 3), consommation et mix énergétique, prix interne du
  carbone, effets financiers anticipés des risques climatiques physiques
  et de transition.
  Sous-catégories : E1-1 à E1-9.

• E2 — Pollution
  Pollution de l'air, de l'eau et des sols, substances préoccupantes,
  microplastiques, effets financiers anticipés liés à la pollution.
  Sous-catégories : E2-1 à E2-6.

• E3 — Eau et ressources marines
  Consommation d'eau, prélèvements en zones de stress hydrique,
  ressources marines, rejets dans l'eau.
  Sous-catégories : E3-1 à E3-5.

• E4 — Biodiversité et écosystèmes
  Plan de transition biodiversité, impacts sur les écosystèmes,
  sites sensibles, métriques de pression sur la biodiversité.
  Sous-catégories : E4-1 à E4-5.

• E5 — Utilisation des ressources et économie circulaire
  Flux entrants/sortants de ressources, déchets, recyclage,
  conception circulaire des produits.
  Sous-catégories : E5-1 à E5-6.

• S1 — Effectifs propres
  Caractéristiques des effectifs, négociation collective, diversité,
  rémunération adéquate, santé-sécurité, formation, équilibre vie
  professionnelle/personnelle, incidents et plaintes.
  Sous-catégories : S1-1 à S1-17.

• S2 — Travailleurs de la chaîne de valeur
  Conditions de travail chez les fournisseurs et sous-traitants,
  travail des enfants, travail forcé, dialogue avec les travailleurs
  de la chaîne de valeur.
  Sous-catégories : S2-1 à S2-5.

• S3 — Communautés affectées
  Droits des communautés locales, peuples autochtones, accès aux
  ressources, déplacement de populations, libertés civiles.
  Sous-catégories : S3-1 à S3-5.

• S4 — Consommateurs et utilisateurs finaux
  Sécurité des produits, protection de la vie privée, accès à
  l'information, pratiques commerciales responsables, inclusion.
  Sous-catégories : S4-1 à S4-5.

• G1 — Conduite des affaires
  Culture d'entreprise, protection des lanceurs d'alerte, lutte
  anti-corruption, relations fournisseurs, pratiques de paiement,
  influence politique et lobbying.
  Sous-catégories : G1-1 à G1-6.


SURPRISE DE MARCHÉ (market_surprise)
aucune   — Information de routine, pleinement anticipée
           (ex : publication périodique conforme au calendrier).
faible   — Information globalement attendue, avec des nuances mineures
           par rapport au consensus ou aux communications précédentes.
partielle — Information partiellement anticipée mais d'ampleur supérieure
           ou de nature différente de ce qui était attendu.
forte    — Information non anticipée, rupture avec la communication
           précédente ou le consensus de marché.

RÈGLE CRITIQUE : si le texte ne mentionne aucun enjeu ESG explicite ou implicite,
répondre csrd_category = "none". Ne pas forcer une catégorie CSRD sur un texte
purement financier ou administratif.
"""


TASK_BLOCK = """
Tu es un auditeur ESG senior, expert en réglementation européenne CSRD
(Corporate Sustainability Reporting Directive) et en normes ESRS,
spécialisé dans l'analyse de documents réglementaires AMF français
(Communiqués, Informations privilégiées, Rapports financiers,
Documents de référence, Déclarations de franchissement de seuil).

Ta mission : classifier l'extrait ci-dessous selon la taxonomie CSRD/ESRS.

CONSIGNES STRICTES :
1. Lis l'extrait en entier avant de répondre.
2. Détermine d'abord si le texte contient un enjeu de durabilité
   (environnemental, social ou de gouvernance au sens CSRD).
3. Si aucun enjeu de durabilité n'est présent → csrd_category = "none"
   et applique la règle de cohérence none.
4. Si un enjeu est présent → identifie la catégorie ESRS principale,
   la sous-catégorie la plus probable, et évalue la matérialité.
5. Le raisonnement (chain_of_thought) doit suivre EXACTEMENT cette
   structure en 3 étapes :
   (1) Nature de l'information : type de document, contenu factuel.
   (2) Analyse de matérialité ESG : le texte relève-t-il de la CSRD/ESG ?
       Si oui, quelle catégorie et pourquoi ? Si non, pourquoi ?
   (3) Surprise pour les investisseurs : cette information est-elle
       anticipée par le marché ? Quel niveau de surprise ?

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant
ni après, sans bloc de code markdown :

{
  "csrd_category": "<none|ESRS2|E1|E2|E3|E4|E5|S1|S2|S3|S4|G1>",
  "esrs_subcategory": "<code sous-catégorie ou chaîne vide si none>",
  "chain_of_thought": "<3 phrases : (1) nature du texte, (2) analyse matérialité ESG, (3) surprise investisseurs>"
}
"""


def build_system_prompt(n_shot: int) -> str:
    """Construit le system prompt zero-shot ou few-shot."""
    blocks = [TAXONOMY_BLOCK.strip(), ""]

    if n_shot > 0:
        blocks.append("EXEMPLES DE RÉFÉRENCE :\n")
        for i, ex in enumerate(FEW_SHOT_EXAMPLES[:n_shot], 1):
            blocks.append(f"Exemple {i} :")
            blocks.append(f"Extrait : «{ex['content']}»")
            blocks.append(f"Réponse : {json.dumps(ex['answer'], ensure_ascii=False)}")
            blocks.append("")

    blocks.append(TASK_BLOCK.strip())
    return "\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU GOLD
# ──────────────────────────────────────────────────────────────────────────────

def load_gold(filepath: str, n_shot: int, limit: int | None) -> pd.DataFrame:
    path = Path(filepath)

    if not path.exists():
        candidates = sorted(Path("data").glob("gold_150*.xlsx"), reverse=True)
        if not candidates:
            candidates = sorted(Path(".").glob("gold_150*.xlsx"), reverse=True)
        if not candidates:
            logger.error(f"Fichier introuvable : {path.resolve()}")
            sys.exit(1)
        path = candidates[0]

    logger.info(f"Gold chargé : {path.name}")
    df = pd.read_excel(path, sheet_name="Annotation")
    logger.info(f"  {len(df)} lignes chargées")

    # Colonnes utiles
    df["content"]     = df["content — Extrait à annoter"].astype(str).str.strip()
    df["gold_label"]  = df["csrd_category"].astype(str).str.strip()
    df["epoch"]       = df["Période"].astype(str).str.strip()
    df["gold_subcat"] = df["esrs_subcategory"].fillna("").astype(str).str.strip()

    # Exclusion du pool few-shot du test set
    few_shot_signatures = {ex["content"][:80] for ex in FEW_SHOT_EXAMPLES[:n_shot]}
    df["_is_few_shot"] = df["content"].str[:80].isin(few_shot_signatures)
    n_excluded = df["_is_few_shot"].sum()
    if n_excluded > 0:
        logger.info(f"  Exclus du test set (pool few-shot) : {n_excluded} exemples")
    df = df[~df["_is_few_shot"]].reset_index(drop=True)

    if limit:
        df = df.head(limit)
        logger.info(f"  Mode test limité à {limit} exemples")

    logger.info(f"  Test set final : {len(df)} exemples")
    logger.info(f"  Distribution époque :")
    for ep, n in df["epoch"].value_counts().sort_index().items():
        logger.info(f"    {ep}: {n}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────

def ckpt_path(model_key: str, n_shot: int) -> Path:
    return RESULTS_DIR / f"checkpoint_{model_key}_shot{n_shot}.json"


def load_checkpoint(model_key: str, n_shot: int) -> dict:
    p = ckpt_path(model_key, n_shot)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"  Checkpoint chargé : {len(data)} prédictions existantes")
        return data
    return {}


def save_checkpoint(model_key: str, n_shot: int, results: dict) -> None:
    p = ckpt_path(model_key, n_shot)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp.replace(p)


# ──────────────────────────────────────────────────────────────────────────────
# PARSING DE LA RÉPONSE JSON
# ──────────────────────────────────────────────────────────────────────────────

def parse_response(text: str) -> tuple[str, str, str]:
    """
    Retourne : (csrd_category, esrs_subcategory, chain_of_thought)
    Fallback sur ('none', '', '') en cas d'échec.
    """
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Aucun objet JSON trouvé")

        data = json.loads(text[start:end])

        # Normaliser les clés (strip + lower)
        data = {str(k).strip().lower(): v for k, v in data.items()}

        # Catégorie
        cat = str(data.get("csrd_category", data.get("category", "none"))).strip()
        if cat not in VALID_CATEGORIES:
            cat = "none"

        # Sous-catégorie
        sub = str(data.get("esrs_subcategory", data.get("subcategory", ""))).strip()
        if cat == "none":
            sub = ""

        # Chain of thought (accepte aussi "reasoning" comme fallback)
        cot = str(
            data.get("chain_of_thought",
            data.get("reasoning",
            data.get("reason", "")))
        ).strip()

        return cat, sub, cot

    except Exception:
        return "none", "", ""


# ──────────────────────────────────────────────────────────────────────────────
# APPELS API
# ──────────────────────────────────────────────────────────────────────────────

def _retry_call(fn, max_retries: int = MAX_RETRIES):
    """Wrapper de retry avec backoff exponentiel sur rate-limit."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if attempt < max_retries and any(
                kw in msg for kw in ("rate", "429", "timeout", "overloaded", "503")
            ):
                wait = RETRY_WAIT * (2 ** attempt)
                logger.warning(f"    Rate-limit / timeout — attente {wait}s (tentative {attempt+1})")
                time.sleep(wait)
            else:
                raise


def call_mistral(model_id: str, system_prompt: str, user_text: str) -> tuple[str, str]:
    from mistralai.client import Mistral  # type: ignore

    api_key = _require_env("MISTRAL_API_KEY")
    
    client = Mistral(api_key=api_key)

    def _call():
        res = client.chat.complete(
            model=model_id,
            temperature=TEMPERATURE,
            random_seed=RANDOM_SEED,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Extrait : «{user_text}»"},
            ],
            max_tokens=MAX_TOKENS,
        )
        content  = res.choices[0].message.content
        resolved = res.model or model_id
        return content, resolved

    return _retry_call(_call)


def call_openai(model_id: str, system_prompt: str, user_text: str) -> tuple[str, str]:
    from openai import OpenAI  # type: ignore

    api_key = _require_env("OPENAI_API_KEY")
    
    client = OpenAI(api_key=api_key)

    def _call():
        res = client.chat.completions.create(
            model=model_id,
            temperature=TEMPERATURE,
            seed=RANDOM_SEED,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Extrait : «{user_text}»"},
            ],
            max_tokens=MAX_TOKENS,
        )
        content  = res.choices[0].message.content
        resolved = res.model or model_id
        return content, resolved

    return _retry_call(_call)


def call_anthropic(model_id: str, system_prompt: str, user_text: str) -> tuple[str, str]:
    from anthropic import Anthropic  # type: ignore

    api_key = _require_env("ANTHROPIC_API_KEY")
    
    client = Anthropic(api_key=api_key)

    def _call():
        res = client.messages.create(
            model=model_id,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Extrait : «{user_text}»"}],
        )
        content  = res.content[0].text
        resolved = res.model or model_id
        return content, resolved

    return _retry_call(_call)


def dispatch(model_key: str, system_prompt: str, user_text: str) -> tuple[str, str, str, str]:
    """Route vers le bon provider et retourne (category, subcategory, chain_of_thought, resolved_model)."""
    cfg      = MODELS[model_key]
    provider = cfg["provider"]
    model_id = cfg["model_id"]

    if provider == "mistral":
        raw, resolved = call_mistral(model_id, system_prompt, user_text)
    elif provider == "anthropic":
        raw, resolved = call_anthropic(model_id, system_prompt, user_text)
    elif provider == "openai":
        raw, resolved = call_openai(model_id, system_prompt, user_text)

    else:
        raise ValueError(f"Provider inconnu : {provider}")

    cat, sub, cot = parse_response(raw)
    return cat, sub, cot, resolved


# ──────────────────────────────────────────────────────────────────────────────
# BOUCLE D'ÉVALUATION — UN MODÈLE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    df: pd.DataFrame,
    system_prompt: str,
    n_shot: int,
    resume: bool,
) -> list[dict]:
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Modèle : {model_key}  ({MODELS[model_key]['group']})")
    logger.info(f"{'─'*60}")

    done = load_checkpoint(model_key, n_shot) if resume else {}
    rows = []

    for i, row in df.iterrows():
        pid = str(row.get("paragraph_id", i))

        if pid in done:
            rows.append(done[pid])
            continue

        pct = (len(done) + len(rows)) / max(len(df), 1) * 100
        logger.info(
            f"  [{i+1:>3}/{len(df)}] {pct:4.1f}%  "
            f"époque={row['epoch']}  gold={row['gold_label']}"
        )

        try:
            cat, sub, cot, resolved = dispatch(
                model_key, system_prompt, row["content"]
            )
            err = None
        except Exception as e:
            logger.error(f"    ✗ Échec : {e}")
            cat, sub, cot, resolved = "none", "", "", MODELS[model_key]["model_id"]
            err = str(e)[:200]

        record = {
            "numero":               row.get("#", i + 1),
            "paragraph_id":         pid,
            "periode":              row["epoch"],
            "societe":              row.get("Société", ""),
            "content":              row["content"],
            "gold_label":           row["gold_label"],
            "gold_subcat":          row.get("gold_subcat", ""),
            "model_key":            model_key,
            "model_id_resolved":    resolved,
            "n_shot":               n_shot,
            "pred_category":        cat,
            "pred_esrs_subcategory": sub,
            "pred_chain_of_thought": cot,
            "error":                err,
        }

        done[pid] = record
        rows.append(record)
        save_checkpoint(model_key, n_shot, done)
        time.sleep(0.3)  # politesse API

    n_ok  = sum(1 for r in rows if not r.get("error"))
    n_err = len(rows) - n_ok
    logger.info(f"  ✅ {n_ok} OK  |  ❌ {n_err} erreurs")

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE DES SORTIES
# ──────────────────────────────────────────────────────────────────────────────

def save_predictions(all_rows: list[dict], n_shot: int) -> Path:
    mode_tag = f"shot{n_shot}"
    csv_path = RESULTS_DIR / f"predictions_{mode_tag}_{RUN_TS}.csv"

    df_out = pd.DataFrame(all_rows)

    # Ordre des colonnes
    preferred_cols = [
        "numero", "paragraph_id", "periode", "societe",
        "content", "gold_label", "gold_subcat",
        "model_key", "model_id_resolved", "n_shot",
        "pred_category", "pred_esrs_subcategory", "pred_chain_of_thought",
        "error",
    ]
    preferred_cols = [c for c in preferred_cols if c in df_out.columns]
    remaining_cols = [c for c in df_out.columns if c not in preferred_cols]

    df_out[preferred_cols + remaining_cols].to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    logger.info(f"\n  ✅ Prédictions : {csv_path}")
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI & MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FraFin — Benchmark Temporal Drift Multi-LLM (version simplifiée)"
    )
    p.add_argument("--gold", default=GOLD_FILE,
                   help="Fichier Excel annoté (défaut : gold_150_annotated_clean_reformulated_without.xlsx)")
    p.add_argument("--n-shot", type=int, default=0,
                   help="0 = zero-shot, 3 = 3-shot (défaut : 0)")
    p.add_argument("--model", default=None,
                   choices=list(MODELS.keys()),
                   help="Tester un seul modèle (défaut : tous)")
    p.add_argument("--resume", action="store_true",
                   help="Reprendre depuis le checkpoint")
    p.add_argument("--limit", type=int, default=None,
                   help="Limiter à N exemples (test rapide)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("═" * 65)
    logger.info("  FraFin — Benchmark Temporal Drift Multi-LLM (simplifié)")
    logger.info(f"  Mode    : {args.n_shot}-shot | Seed : {RANDOM_SEED} | T = {TEMPERATURE}")
    logger.info(f"  Modèles : {args.model or 'tous'}")
    logger.info("═" * 65)

    df = load_gold(args.gold, args.n_shot, args.limit)
    system_prompt = build_system_prompt(args.n_shot)

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    all_rows: list[dict] = []

    for mk in models_to_run:
        rows = evaluate_model(mk, df, system_prompt, args.n_shot, args.resume)
        all_rows.extend(rows)

    csv_path = save_predictions(all_rows, args.n_shot)

    # Résumé rapide
    df_summary = pd.DataFrame(all_rows)
    logger.info(f"\n{'═'*65}")
    logger.info(f"  RÉSUMÉ — {len(all_rows)} prédictions totales")
    logger.info(f"{'═'*65}")
    logger.info(f"\n  Prédictions par catégorie :")
    for cat, n in df_summary["pred_category"].value_counts().items():
        logger.info(f"    {cat:<22} {n:>4}")
    logger.info(f"\n  Erreurs : {df_summary['error'].notna().sum()}")
    logger.info(f"\n  Fichier : {csv_path}")
    logger.info("═" * 65)


if __name__ == "__main__":
    main()