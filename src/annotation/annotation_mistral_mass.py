"""
annotation_mistral_mass.py — Annotation de Masse via Mistral Large API
═══════════════════════════════════════════════════════════════════════════════════════
Phase 3b du pipeline FraFin-Reasoning.

RÔLE DANS LE PIPELINE :
    frafin_sample_rigorous_{TS}.parquet  (15 000, Phase 2)
    frafin_gold_500_{TS}.xlsx            (500 annotés à la main, Phase 3a)
              │
              ▼
    annotation_mistral_mass.py   ← CE SCRIPT, deux passes :
              │
              ├── PASSE "mass" : annote les ~14 500 paragraphes hors GOLD
              │       → frafin_mass_annotated_{TS}.parquet
              │         (ces labels Mistral entrent dans le dataset publié)
              │
              └── PASSE "iaa"  : ré-annote indépendamment les 200 paragraphes
                      IAA (sous-ensemble du GOLD), pour comparaison ultérieure
                      avec les labels humains
                      → iaa_mistral_annotations_{TS}.parquet
                        (PAS fusionné au dataset — sert uniquement au Kappa
                         de Cohen dans validation_iaa.py)
              ▼
    validation_iaa.py  (prochaine étape)

GARDE-FOU MÉTHODOLOGIQUE — ANTI-FUITE DE DONNÉES (data leakage) :
    Les exemples few-shot injectés dans le prompt système sont tirés UNIQUEMENT
    des paragraphes GOLD qui ne font PAS partie du sous-ensemble IAA (200).
    Si un exemple few-shot montrait à Mistral la réponse humaine exacte d'un
    paragraphe qu'on lui demande ensuite d'annoter indépendamment pour mesurer
    l'accord inter-annotateurs, le Kappa de Cohen serait artificiellement
    gonflé — le modèle n'aurait pas "deviné" la réponse, il l'aurait lue.

REPRODUCTIBILITÉ STRICTE (pour le papier arXiv) :
    - temperature = 0.0
    - random_seed = 42 (paramètre natif de l'API Mistral)
    - response_format = {"type": "json_object"}
    - Version exacte du modèle résolu par l'API loguée par appel
      (le champ "model" de la réponse, ex: "mistral-large-2411", peut différer
      de l'alias demandé "mistral-large-latest")
    - Snapshot complet du system prompt sauvegardé sur disque

VALIDATION & CORRECTION AUTOMATIQUE :
    Chaque réponse JSON est validée contre csrd_taxonomy.validate_annotation()
    avant acceptation. En cas d'échec (JSON malformé ou incohérence de schéma,
    ex: csrd_category='none' avec materialite_score=4), un re-prompt correctif
    est envoyé avec le détail des erreurs (jusqu'à MAX_RETRIES_VALIDATION fois)
    avant d'abandonner et de loguer l'échec.

RÉSILIENCE (gros volume, run de plusieurs heures) :
    - Traitement par batches + checkpoint JSON → reprise avec --resume
    - Échecs définitifs isolés dans un fichier séparé → reprise ciblée avec
      --retry-failed sans retraiter ce qui a déjà réussi

Dépendances :
    pip install pandas pyarrow requests openpyxl

Variable d'environnement :
    export MISTRAL_API_KEY="votre_clé"

Usage :
    python annotation_mistral_mass.py
    python annotation_mistral_mass.py --mode mass
    python annotation_mistral_mass.py --mode iaa
    python annotation_mistral_mass.py --resume
    python annotation_mistral_mass.py --retry-failed
    python annotation_mistral_mass.py --limit 50          # test rapide
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_SUPPORT = True
except ImportError:
    PARQUET_SUPPORT = False

try:
    from src.annotation.csrd_taxonomy import (
        CSRD_TAXONOMY_VERSION,
        CSRD_CATEGORIES,
        build_taxonomy_prompt_block,
        validate_annotation,
        validate_batch,
    )
except ImportError:
    print(
        "❌ csrd_taxonomy.py introuvable.\n"
        "   Placez csrd_taxonomy.py dans le même dossier que ce script."
    )
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

API_URL          = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL    = "mistral-large-latest"
TEMPERATURE      = 0.0
RANDOM_SEED      = 42
MAX_TOKENS       = 900
PROMPT_VERSION   = "v1.0"

N_FEW_SHOT       = 5

MAX_WORKERS      = 10
BATCH_SIZE       = 50

HTTP_TIMEOUT         = 60
HTTP_RETRIES         = 5
HTTP_BACKOFF         = 2.0
HTTP_RETRY_CODES     = {429, 500, 502, 503, 504}
MAX_RETRIES_VALIDATION = 2     # re-prompts correctifs après échec de validation

# ═════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ═════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")

CHECKPOINT_MASS = DATA_DIR / "checkpoint_mass_annotation.json"
CHECKPOINT_IAA  = DATA_DIR / "checkpoint_iaa_annotation.json"
FAILED_MASS     = DATA_DIR / "failed_mass_annotation.json"
FAILED_IAA      = DATA_DIR / "failed_iaa_annotation.json"
PROMPT_SNAPSHOT = DATA_DIR / f"system_prompt_snapshot_{RUN_TS}.txt"
REPORT_OUT      = DATA_DIR / f"annotation_mistral_report_{RUN_TS}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# SCHÉMA DE SORTIE
# ═════════════════════════════════════════════════════════════════════════════

OUTPUT_COLUMNS = [
    "paragraph_id", "article_id", "publish_date", "author", "headline", "content",
    "doc_type", "doc_group", "emetteur_lei", "source_url", "language",
    "_year", "year_bucket", "csrd_score_raw", "csrd_quartile", "stratum_id", "_n_words",
    "csrd_category", "esrs_subcategory",
    "materialite_financiere", "materialite_impact", "materialite_score",
    "horizon_temporel", "market_surprise", "chain_of_thought", "final_answer_json",
    "annotateur", "taxonomy_version", "prompt_version",
    "confiance_annotation", "annotation_validated",
    "api_model_resolved", "api_usage_prompt_tokens", "api_usage_completion_tokens",
]

ANNOTATION_FIELDS = [
    "csrd_category", "esrs_subcategory", "materialite_score",
    "materialite_financiere", "materialite_impact",
    "market_surprise", "horizon_temporel",
    "chain_of_thought", "confiance_annotation",
]

# ═════════════════════════════════════════════════════════════════════════════
# LOCKS & COMPTEURS
# ═════════════════════════════════════════════════════════════════════════════

csv_lock = Lock()

# ═════════════════════════════════════════════════════════════════════════════
# EXEMPLES FEW-SHOT PAR DÉFAUT — utilisés UNIQUEMENT si le GOLD n'est pas
# encore (ou pas suffisamment) annoté. Marqués explicitement comme tels dans
# le prompt et dans le rapport, pour la transparence méthodologique.
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_FEW_SHOT = [
    {
        "content": (
            "La société annonce avoir procédé au rachat de 125 000 actions "
            "propres au cours du mois, dans le cadre du programme de rachat "
            "autorisé par l'assemblée générale du 14 mai 2023, à un prix "
            "moyen de 42,17 euros par action."
        ),
        "answer": {
            "csrd_category": "none",
            "esrs_subcategory": None,
            "materialite_score": 0,
            "materialite_financiere": False,
            "materialite_impact": False,
            "market_surprise": "aucune",
            "horizon_temporel": None,
            "chain_of_thought": (
                "Étape 1 : déclaration réglementaire de rachat d'actions, "
                "opération financière courante. Étape 2 : aucune dimension "
                "de durabilité environnementale, sociale ou de gouvernance "
                "n'est mentionnée. Étape 3 : information de routine, "
                "pleinement anticipée par le calendrier du programme."
            ),
            "confiance_annotation": 0.97,
        },
    },
    {
        "content": (
            "Le groupe a réduit ses émissions de gaz à effet de serre Scope 1 "
            "et 2 de 18% sur l'exercice, dépassant l'objectif annuel de 12% "
            "fixé dans son plan de transition climatique, grâce à l'arrêt "
            "anticipé de deux unités de production au charbon."
        ),
        "answer": {
            "csrd_category": "E1",
            "esrs_subcategory": "E1-6",
            "materialite_score": 4,
            "materialite_financiere": True,
            "materialite_impact": True,
            "market_surprise": "partielle",
            "horizon_temporel": "court_terme",
            "chain_of_thought": (
                "Étape 1 : annonce d'une performance climatique sur les "
                "émissions GES Scope 1/2. Étape 2 : matérialité financière "
                "(coûts évités de transition, exposition réduite à la "
                "tarification carbone) et matérialité d'impact (réduction "
                "réelle des émissions) toutes deux engagées. Étape 3 : "
                "résultat supérieur à l'objectif annoncé, donc surprise "
                "partielle positive pour le marché."
            ),
            "confiance_annotation": 0.9,
        },
    },
    {
        "content": (
            "Suite à un accident industriel survenu sur le site de production "
            "principal, l'entreprise déplore un décès et trois blessés graves "
            "parmi son personnel. Une enquête interne a été ouverte et les "
            "autorités compétentes ont été immédiatement informées."
        ),
        "answer": {
            "csrd_category": "S1",
            "esrs_subcategory": "S1-14",
            "materialite_score": 5,
            "materialite_financiere": True,
            "materialite_impact": True,
            "market_surprise": "forte",
            "horizon_temporel": "immediat",
            "chain_of_thought": (
                "Étape 1 : incident grave de santé-sécurité au travail avec "
                "décès. Étape 2 : matérialité d'impact maximale (atteinte à "
                "la vie humaine) et matérialité financière probable "
                "(responsabilité, arrêt de production, réputation). "
                "Étape 3 : événement non anticipé, rupture brutale avec la "
                "communication précédente — surprise forte."
            ),
            "confiance_annotation": 0.95,
        },
    },
]

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU POOL ET DES IDENTIFIANTS GOLD/IAA
# ═════════════════════════════════════════════════════════════════════════════

def load_pool(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists() or filepath == "auto":
        candidates = sorted(DATA_DIR.glob("frafin_sample_rigorous_*.parquet"), reverse=True)
        if not candidates:
            candidates = sorted(DATA_DIR.glob("frafin_sample_rigorous_*.csv"), reverse=True)
        if not candidates:
            logger.error("Aucun frafin_sample_rigorous_*.parquet trouvé. Lancez Phase 2 d'abord.")
            sys.exit(1)
        path = candidates[0]
        logger.info(f"Pool auto-détecté : {path.name}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else \
         pd.read_csv(path, dtype=str, low_memory=False)

    if "annotation_prompt" not in df.columns:
        logger.error(
            "Colonne 'annotation_prompt' absente du pool. "
            "Ce fichier ne vient pas de build_annotation_sample_rigorous.py."
        )
        sys.exit(1)

    logger.info(f"  Pool chargé : {len(df):,} paragraphes")
    return df


def load_gold_ids(filepath: str) -> tuple[set[str], set[str]]:
    path = Path(filepath)
    if not path.exists() or filepath == "auto":
        candidates = sorted(DATA_DIR.glob("frafin_gold_500_*.xlsx"), reverse=True)
        if not candidates:
            logger.error("Aucun frafin_gold_500_*.xlsx trouvé. Lancez Phase 3a d'abord.")
            sys.exit(1)
        path = candidates[0]
        logger.info(f"GOLD auto-détecté : {path.name}")

    gold_sheet = pd.read_excel(path, sheet_name="Annotation")
    gold_ids = set(gold_sheet["paragraph_id"].dropna().astype(str))

    iaa_col = "IAA ?"
    if iaa_col in gold_sheet.columns:
        iaa_ids = set(
            gold_sheet.loc[
                gold_sheet[iaa_col].astype(str).str.strip().str.lower() == "oui",
                "paragraph_id",
            ].dropna().astype(str)
        )
    else:
        iaa_ids = set()

    logger.info(f"  GOLD : {len(gold_ids):,} paragraphes  |  IAA : {len(iaa_ids):,} paragraphes")
    return gold_ids, gold_sheet, iaa_ids

# ═════════════════════════════════════════════════════════════════════════════
# FEW-SHOT — chargé depuis le GOLD humain (hors sous-ensemble IAA)
# ═════════════════════════════════════════════════════════════════════════════

def _excel_value_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("oui", "true", "1", "vrai")


def _excel_value_to_opt_str(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def load_few_shot_from_gold(
    gold_sheet: pd.DataFrame, iaa_ids: set[str], n: int
) -> tuple[list[dict], bool]:
    """
    Construit les exemples few-shot à partir des lignes GOLD déjà annotées
    par l'humain, EN EXCLUANT le sous-ensemble IAA (anti-fuite de données,
    cf. docstring du module). Retourne (exemples, is_default_fallback).
    """
    candidates = gold_sheet[
        gold_sheet["paragraph_id"].notna()
        & (~gold_sheet["paragraph_id"].astype(str).isin(iaa_ids))
        & gold_sheet["csrd_category"].notna()
        & (gold_sheet["csrd_category"].astype(str).str.strip() != "")
    ]

    if candidates.empty:
        logger.warning(
            "  ⚠ Aucune annotation GOLD trouvée (hors IAA) — "
            "utilisation des exemples few-shot par défaut (génériques)."
        )
        return DEFAULT_FEW_SHOT[:n], True

    examples: list[dict] = []
    seen_categories: set[str] = set()
    shuffled = candidates.sample(frac=1, random_state=RANDOM_SEED)

    for _, row in shuffled.iterrows():
        cat = str(row["csrd_category"]).strip()
        if cat in seen_categories:
            continue
        examples.append({
            "content": row.get("content", ""),
            "answer": {
                "csrd_category": cat,
                "esrs_subcategory": _excel_value_to_opt_str(row.get("esrs_subcategory")),
                "materialite_score": (
                    int(row["materialite_score"])
                    if pd.notna(row.get("materialite_score")) else None
                ),
                "materialite_financiere": _excel_value_to_bool(row.get("materialite_financiere")),
                "materialite_impact": _excel_value_to_bool(row.get("materialite_impact")),
                "market_surprise": _excel_value_to_opt_str(row.get("market_surprise")),
                "horizon_temporel": _excel_value_to_opt_str(row.get("horizon_temporel")),
                "chain_of_thought": _excel_value_to_opt_str(row.get("chain_of_thought")) or "",
                "confiance_annotation": (
                    float(row["confiance_annotation"])
                    if pd.notna(row.get("confiance_annotation")) else 0.85
                ),
            },
        })
        seen_categories.add(cat)
        if len(examples) >= n:
            break

    if len(examples) < n:
        logger.warning(
            f"  ⚠ Seulement {len(examples)} exemples GOLD disponibles "
            f"(catégories distinctes) — complément avec exemples par défaut."
        )
        for ex in DEFAULT_FEW_SHOT:
            if len(examples) >= n:
                break
            examples.append(ex)

    logger.info(f"  ✅ {len(examples)} exemples few-shot chargés depuis le GOLD humain "
                f"(catégories : {sorted(seen_categories)})")
    return examples, False

# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

def format_few_shot_block(examples: list[dict]) -> str:
    lines = ["EXEMPLES ANNOTÉS (référence de style et de rigueur attendue) :", ""]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"--- Exemple {i} ---")
        lines.append(f"Extrait : «{ex['content']}»")
        lines.append(f"Réponse attendue :")
        lines.append(json.dumps(ex["answer"], ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(few_shot_examples: list[dict]) -> str:
    taxonomy_block = build_taxonomy_prompt_block()
    few_shot_block = format_few_shot_block(few_shot_examples)

    final_instructions = (
        "INSTRUCTIONS DE SORTIE :\n"
        "Vous êtes un expert financier et auditeur réglementaire CSRD. "
        "Pour chaque extrait fourni, répondez UNIQUEMENT avec un objet JSON valide, "
        "sans texte avant ni après, sans balises markdown.\n\n"
        "Le JSON doit contenir exactement ces champs :\n"
        "csrd_category, esrs_subcategory, materialite_score, "
        "materialite_financiere, materialite_impact, market_surprise, "
        "horizon_temporel, chain_of_thought, confiance_annotation.\n\n"
        "chain_of_thought doit suivre 3 étapes explicites : "
        "(1) nature de l'information, (2) analyse de matérialité, "
        "(3) évaluation de la surprise de marché."
    )

    return "\n\n".join([taxonomy_block, few_shot_block, final_instructions])

# ═════════════════════════════════════════════════════════════════════════════
# APPEL API MISTRAL — seule fonction touchant le réseau (isolée pour les tests)
# ═════════════════════════════════════════════════════════════════════════════

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES, backoff_factor=HTTP_BACKOFF,
        status_forcelist=HTTP_RETRY_CODES, allowed_methods=["POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def call_mistral_raw(
    session: requests.Session, api_key: str,
    system_prompt: str, user_prompt: str,
    model: str, temperature: float, seed: int, max_tokens: int,
) -> tuple[str, str, dict]:
    """
    Unique point de contact réseau. Retourne (content_str, resolved_model, usage_dict).
    Lève une exception en cas d'échec réseau/HTTP (gérée par l'appelant).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "random_seed": seed,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    r = session.post(API_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    resolved_model = data.get("model", model)
    usage = data.get("usage", {}) or {}
    return content, resolved_model, usage

# ═════════════════════════════════════════════════════════════════════════════
# ANNOTATION D'UN PARAGRAPHE — parsing + validation + retries correctifs
# ═════════════════════════════════════════════════════════════════════════════

def annotate_paragraph(
    session: requests.Session, api_key: str,
    system_prompt: str, user_prompt: str,
    model: str, seed: int,
) -> tuple[dict | None, str, dict | None, dict]:
    """
    Retourne (annotation_validée_ou_None, resolved_model, error_info_ou_None, usage_cumulé).
    Effectue jusqu'à MAX_RETRIES_VALIDATION re-prompts correctifs si le JSON
    est malformé ou viole le schéma de csrd_taxonomy.
    """
    last_errors: list[str] = []
    last_raw = ""
    resolved_model = model
    cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    current_user_prompt = user_prompt

    for attempt in range(MAX_RETRIES_VALIDATION + 1):
        try:
            content, resolved_model, usage = call_mistral_raw(
                session, api_key, system_prompt, current_user_prompt,
                model, TEMPERATURE, seed, MAX_TOKENS,
            )
        except Exception as e:
            return None, resolved_model, {
                "error": f"api_error:{type(e).__name__}",
                "detail": str(e)[:300],
            }, cumulative_usage

        for k in cumulative_usage:
            cumulative_usage[k] += usage.get(k, 0)
        last_raw = content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            last_errors = [f"JSON invalide : {e}"]
            current_user_prompt = (
                user_prompt
                + "\n\nVotre réponse précédente n'était pas un JSON valide "
                + f"({last_errors[0]}). Renvoyez UNIQUEMENT un objet JSON valide, "
                + "sans aucun texte ni balise markdown autour."
            )
            continue

        is_valid, errors = validate_annotation(parsed)
        if is_valid:
            return parsed, resolved_model, None, cumulative_usage

        last_errors = errors
        current_user_prompt = (
            user_prompt
            + "\n\nVotre réponse précédente contenait des erreurs de cohérence :\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nCorrigez ces erreurs et renvoyez UNIQUEMENT le JSON corrigé."
        )

    return None, resolved_model, {
        "error": "validation_failed_after_retries",
        "errors": last_errors,
        "last_raw_response": last_raw[:500],
    }, cumulative_usage

# ═════════════════════════════════════════════════════════════════════════════
# WORKER DE THREAD
# ═════════════════════════════════════════════════════════════════════════════

def pipeline_worker(
    row: dict, system_prompt: str, api_key: str, model: str, seed: int,
) -> dict:
    session = make_session()
    user_prompt = row.get("annotation_prompt") or ""

    parsed, resolved_model, error_info, usage = annotate_paragraph(
        session, api_key, system_prompt, user_prompt, model, seed
    )

    return {
        "paragraph_id": row["paragraph_id"],
        "success": parsed is not None,
        "annotation": parsed,
        "resolved_model": resolved_model,
        "usage": usage,
        "error_info": error_info,
    }

# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT & FAILED LOG
# ═════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("done_ids", []))
    return set()


def save_checkpoint(path: Path, done_ids: set[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done_ids": list(done_ids), "saved_at": datetime.now().isoformat()}, f)
    tmp.replace(path)


def load_failed_log(path: Path) -> dict[str, dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_failed_log(path: Path, failed: dict[str, dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def find_existing_output(pattern: str) -> Path | None:
    candidates = sorted(DATA_DIR.glob(pattern), reverse=True)
    return candidates[0] if candidates else None

# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA LIGNE DE SORTIE
# ═════════════════════════════════════════════════════════════════════════════

def build_output_row(result: dict, original_row: dict) -> dict:
    ann = result["annotation"]
    out = {k: original_row.get(k) for k in OUTPUT_COLUMNS if k in original_row}
    usage = result.get("usage") or {}
    out.update({
        "csrd_category":          ann["csrd_category"],
        "esrs_subcategory":       ann.get("esrs_subcategory"),
        "materialite_financiere": ann.get("materialite_financiere"),
        "materialite_impact":     ann.get("materialite_impact"),
        "materialite_score":      ann.get("materialite_score"),
        "horizon_temporel":       ann.get("horizon_temporel"),
        "market_surprise":        ann.get("market_surprise"),
        "chain_of_thought":       ann.get("chain_of_thought"),
        "final_answer_json":      json.dumps(ann, ensure_ascii=False),
        "annotateur":             f"mistral:{result.get('resolved_model', 'unknown')}",
        "taxonomy_version":       CSRD_TAXONOMY_VERSION,
        "prompt_version":         PROMPT_VERSION,
        "confiance_annotation":   ann.get("confiance_annotation"),
        "annotation_validated":   True,
        "api_model_resolved":     result.get("resolved_model"),
        "api_usage_prompt_tokens":     usage.get("prompt_tokens"),
        "api_usage_completion_tokens": usage.get("completion_tokens"),
    })
    return out

# ═════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE D'UNE PASSE (mass OU iaa) — batches + checkpoint
# ═════════════════════════════════════════════════════════════════════════════

def run_annotation_pass(
    pass_name: str,
    df: pd.DataFrame,
    system_prompt: str,
    api_key: str,
    model: str,
    seed: int,
    out_csv_pattern: str,
    checkpoint_path: Path,
    failed_log_path: Path,
    resume: bool,
    retry_failed: bool,
    limit: int | None,
) -> dict:
    logger.info(f"\n{'═'*65}")
    logger.info(f"  PASSE '{pass_name}' — {len(df):,} paragraphes dans le périmètre")
    logger.info(f"{'═'*65}")

    done_ids = load_checkpoint(checkpoint_path) if resume else set()
    failed_entries = load_failed_log(failed_log_path)

    if retry_failed and failed_entries:
        retry_ids = set(failed_entries.keys())
        done_ids -= retry_ids
        logger.info(f"  --retry-failed : {len(retry_ids)} paragraphes relancés")

    pool = df[~df["paragraph_id"].astype(str).isin(done_ids)].copy()
    if limit:
        pool = pool.head(limit)

    total = len(pool)
    logger.info(f"  À traiter : {total:,}  (déjà fait : {len(done_ids):,})")

    if total == 0:
        logger.info("  Rien à faire pour cette passe.")
        return {
            "n_submitted": 0, "n_success": 0, "n_failed": 0,
            "tokens_prompt": 0, "tokens_completion": 0,
            "out_csv": None,
        }

    existing = find_existing_output(out_csv_pattern) if resume else None
    out_csv  = existing if existing else DATA_DIR / out_csv_pattern.replace("*", RUN_TS)
    mode     = "a" if existing else "w"

    rows = pool.to_dict("records")
    row_by_id = {r["paragraph_id"]: r for r in rows}

    stats = {"n_submitted": total, "n_success": 0, "n_failed": 0,
             "tokens_prompt": 0, "tokens_completion": 0}

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_start in range(0, total, BATCH_SIZE):
            batch = rows[batch_start: batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            logger.info(
                f"  Batch {batch_num}/{n_batches} "
                f"[{batch_start+1}-{batch_start+len(batch)}/{total}] "
                f"ok={stats['n_success']} fail={stats['n_failed']}"
            )

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        pipeline_worker, row, system_prompt, api_key, model, seed
                    ): row["paragraph_id"]
                    for row in batch
                }

                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "paragraph_id": pid, "success": False,
                            "error_info": {
                                "error": f"worker_exception:{type(e).__name__}",
                                "detail": str(e)[:300],
                            },
                        }

                    if result["success"]:
                        out_row = build_output_row(result, row_by_id[pid])
                        with csv_lock:
                            writer.writerow(out_row)
                            f.flush()
                        stats["n_success"] += 1
                        usage = result.get("usage") or {}
                        stats["tokens_prompt"]     += usage.get("prompt_tokens", 0)
                        stats["tokens_completion"] += usage.get("completion_tokens", 0)
                        failed_entries.pop(pid, None)
                    else:
                        stats["n_failed"] += 1
                        failed_entries[pid] = result.get("error_info", {})

                    done_ids.add(pid)

            save_checkpoint(checkpoint_path, done_ids)
            save_failed_log(failed_log_path, failed_entries)

    stats["out_csv"] = str(out_csv)
    logger.info(f"\n  Passe '{pass_name}' terminée : "
                f"{stats['n_success']:,} OK, {stats['n_failed']:,} échecs")
    return stats

# ═════════════════════════════════════════════════════════════════════════════
# EXPORT PARQUET
# ═════════════════════════════════════════════════════════════════════════════

def export_parquet(csv_path: str | None) -> str | None:
    if not csv_path or not PARQUET_SUPPORT:
        return None
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        df["publish_date"] = pd.to_datetime(df.get("publish_date"), errors="coerce").dt.date
        df["materialite_score"] = pd.to_numeric(df.get("materialite_score"), errors="coerce")
        df["confiance_annotation"] = pd.to_numeric(df.get("confiance_annotation"), errors="coerce")
        for bcol in ("materialite_financiere", "materialite_impact", "annotation_validated"):
            if bcol in df.columns:
                df[bcol] = df[bcol].map({"True": True, "False": False, "true": True, "false": False})
        if "final_answer_json" in df.columns:
            df["final_answer_json"] = df["final_answer_json"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else None
            )
        parquet_path = csv_path.with_suffix(".parquet")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                        parquet_path, compression="snappy")
        logger.info(f"  ✅ Parquet : {parquet_path}")
        return str(parquet_path)
    except Exception as e:
        logger.error(f"  ✗ Export Parquet échoué : {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# QA FINALE — re-validation de tous les enregistrements écrits
# ═════════════════════════════════════════════════════════════════════════════

def final_qa(csv_path: str | None) -> dict:
    if not csv_path or not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    annotations = []
    for _, row in df.iterrows():
        try:
            ann = json.loads(row["final_answer_json"])
            annotations.append(ann)
        except Exception:
            continue
    return validate_batch(annotations)

# ═════════════════════════════════════════════════════════════════════════════
# RAPPORT GLOBAL
# ═════════════════════════════════════════════════════════════════════════════

def save_global_report(
    model: str, is_default_few_shot: bool, n_few_shot_used: int,
    mass_stats: dict | None, iaa_stats: dict | None,
    mass_qa: dict, iaa_qa: dict,
) -> None:
    report = {
        "run_timestamp": RUN_TS,
        "model_requested": model,
        "temperature": TEMPERATURE,
        "random_seed": RANDOM_SEED,
        "taxonomy_version": CSRD_TAXONOMY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "few_shot": {
            "n_examples": n_few_shot_used,
            "source": "default_generic" if is_default_few_shot else "human_gold_non_iaa",
            "anti_leakage_note": (
                "Few-shot examples were drawn exclusively from GOLD paragraphs "
                "outside the IAA subset to prevent data leakage into the "
                "Cohen's Kappa validation."
            ),
        },
        "mass_pass": mass_stats,
        "mass_pass_qa": mass_qa,
        "iaa_pass": iaa_stats,
        "iaa_pass_qa": iaa_qa,
        "prompt_snapshot_file": str(PROMPT_SNAPSHOT),
        "arxiv_text": (
            f"Mass annotation was performed using {model} with temperature=0, "
            f"a fixed random seed (42), and structured JSON output mode. "
            f"Each response was validated against the ESRS taxonomy schema "
            f"(Section 3.3); responses violating coherence rules (e.g., "
            f"non-zero materiality with csrd_category='none') triggered up to "
            f"{MAX_RETRIES_VALIDATION} corrective re-prompts before being "
            f"logged as failed. The system prompt included "
            f"{n_few_shot_used} few-shot examples drawn from the human-annotated "
            f"Gold Standard, strictly excluding the 200-paragraph IAA subset "
            f"to prevent data leakage into the inter-annotator agreement "
            f"computation (Cohen's Kappa, reported in Section 4)."
        ),
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  ✅ Rapport global : {REPORT_OUT}")

# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FraFin-Reasoning — Annotation Mistral Large (Phase 3b)"
    )
    p.add_argument("--pool",   default="auto", help="frafin_sample_rigorous_*.parquet")
    p.add_argument("--gold",   default="auto", help="frafin_gold_500_*.xlsx")
    p.add_argument("--mode",   choices=["mass", "iaa", "both"], default="both")
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=None, help="Sinon lu depuis MISTRAL_API_KEY")
    p.add_argument("--seed",   type=int, default=RANDOM_SEED)
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--n-few-shot", type=int, default=N_FEW_SHOT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Test rapide sur N lignes")
    return p.parse_args()


def main():
    global MAX_WORKERS, BATCH_SIZE

    args = parse_args()
    MAX_WORKERS = args.workers
    BATCH_SIZE  = args.batch_size

    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        logger.error("Clé API manquante : définissez MISTRAL_API_KEY ou utilisez --api-key.")
        sys.exit(1)

    logger.info("═" * 65)
    logger.info("  FraFin-Reasoning — Annotation Mistral Large (Phase 3b)")
    logger.info(f"  Mode : {args.mode}  |  Modèle : {args.model}  |  Seed : {args.seed}")
    logger.info(f"  Workers : {MAX_WORKERS}  |  Batch : {BATCH_SIZE}")
    logger.info("═" * 65)

    pool_df = load_pool(args.pool)
    gold_ids, gold_sheet, iaa_ids = load_gold_ids(args.gold)

    few_shot_examples, is_default = load_few_shot_from_gold(
        gold_sheet, iaa_ids, args.n_few_shot
    )
    system_prompt = build_system_prompt(few_shot_examples)

    with open(PROMPT_SNAPSHOT, "w", encoding="utf-8") as f:
        f.write(system_prompt)
    logger.info(f"  📄 Snapshot du system prompt : {PROMPT_SNAPSHOT}")

    mass_stats = iaa_stats = None
    mass_qa = iaa_qa = {}

    if args.mode in ("mass", "both"):
        mass_pool = pool_df[~pool_df["paragraph_id"].astype(str).isin(gold_ids)]
        mass_stats = run_annotation_pass(
            "mass", mass_pool, system_prompt, api_key, args.model, args.seed,
            "frafin_mass_annotated_*.csv", CHECKPOINT_MASS, FAILED_MASS,
            args.resume, args.retry_failed, args.limit,
        )
        export_parquet(mass_stats.get("out_csv"))
        mass_qa = final_qa(mass_stats.get("out_csv"))

    if args.mode in ("iaa", "both"):
        iaa_pool = pool_df[pool_df["paragraph_id"].astype(str).isin(iaa_ids)]
        iaa_stats = run_annotation_pass(
            "iaa", iaa_pool, system_prompt, api_key, args.model, args.seed,
            "iaa_mistral_annotations_*.csv", CHECKPOINT_IAA, FAILED_IAA,
            args.resume, args.retry_failed, args.limit,
        )
        export_parquet(iaa_stats.get("out_csv"))
        iaa_qa = final_qa(iaa_stats.get("out_csv"))

    save_global_report(
        args.model, is_default, len(few_shot_examples),
        mass_stats, iaa_stats, mass_qa, iaa_qa,
    )

    logger.info(f"\n{'═'*65}")
    logger.info(f"  TERMINÉ")
    if mass_stats:
        logger.info(f"  Mass : {mass_stats['n_success']:,} OK / "
                    f"{mass_stats['n_submitted']:,}  "
                    f"({mass_stats['n_failed']:,} échecs)")
    if iaa_stats:
        logger.info(f"  IAA  : {iaa_stats['n_success']:,} OK / "
                    f"{iaa_stats['n_submitted']:,}  "
                    f"({iaa_stats['n_failed']:,} échecs)")
    logger.info(f"  ⚙  Prochaine étape : validation_iaa.py")
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
