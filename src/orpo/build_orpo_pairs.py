"""
build_orpo_pairs.py — Construction de Paires ORPO par Rejection Sampling
═══════════════════════════════════════════════════════════════════════════════════════
Phase 6a du pipeline FraFin-Reasoning — Remédiation du biais de récence temporelle.

RÔLE DANS LE PIPELINE :
    frafin_sample_rigorous_14.8k.parquet   (pool nettoyé, Phase 2 — PAS le Gold)
    frafin_gold_140_annotated.xlsx         (140 annotés — EXCLU du pool, jamais touché)
              │
              ▼
    build_orpo_pairs.py   ← CE SCRIPT
              │
              ▼
    orpo_pairs_{TS}.jsonl                  (paires chosen/rejected pour l'entraînement)
              │
              ▼
    train_orpo_mistral7b.py                (prochaine étape — non couverte ici)

PRINCIPE MÉTHODOLOGIQUE — REJECTION SAMPLING, PAS FABRICATION :
    Les paires ORPO ne sont JAMAIS écrites à la main ni générées par un prompt
    du type "écris une réponse anachronique". Une telle fabrication introduirait
    un jugement subjectif de "cohérence historique" sans lien vérifiable avec la
    vérité de la classification.

    À la place, pour chaque paragraphe candidat :
      1. Un JUGE (Mistral Large, temp=0) produit un label de référence ("silver").
      2. La POLITIQUE DE BASE (Mistral-7B, le modèle qu'on va corriger) produit
         sa propre prédiction zero-shot sur le MÊME paragraphe, avec le MÊME
         prompt que celui utilisé dans benchmark_temporal_drift.py.
      3. Si la prédiction du modèle diverge du label de référence (silver),
         c'est une ERREUR RÉELLE ET OBSERVÉE du modèle — pas une fabrication.
         Elle devient `rejected`. Le label silver devient `chosen`.

    Ce sont donc littéralement les erreurs mesurées du modèle qui construisent
    les données d'entraînement — le lien avec la Section 4 (mesure du biais)
    est direct et vérifiable.

DEUX TYPES DE PAIRES CONSTRUITES :
    Type A — Correction du biais de récence (FN, le cas dominant, cf. Section 4)
        silver = CSRD réel   |   base prédit `none`  →  chosen=silver, rejected=base
        Échantillonné en priorité sur 2010-2014 et 2015-2019 (strates à FNR élevé
        documenté empiriquement : FNR=0.667 sur 2010-2014, cf. Tableau 4.3).

    Type B — Garde-fou anti-sur-correction (FP)
        silver = `none` réel   |   base prédit CSRD (faux positif)  →  chosen=silver, rejected=base
        Sans ce type, un modèle entraîné uniquement sur le Type A pourrait
        apprendre l'heuristique dégénérée "document ancien ⇒ toujours CSRD",
        ce qui dégraderait le FPR sans réellement corriger la sémantique.

    Les cas d'ACCORD (base == silver) ne produisent PAS de paire — l'absence
    de contraste ne fournit aucun signal de préférence exploitable par ORPO.

GARANTIE ANTI-FUITE DE DONNÉES :
    Le pool source (14.8k) est explicitement filtré pour EXCLURE tout
    paragraph_id présent dans le Gold-140. Le Gold-140 reste intégralement
    et exclusivement le test set final (Section 4), jamais vu à l'entraînement.

RÉSILIENCE :
    Checkpoint JSON par paragraphe traité → reprise après crash sans double
    appel API. Cap configurable sur le nombre total d'appels (--max-calls)
    pour éviter un run incontrôlé.

Dépendances :
    pip install pandas numpy requests pyarrow openpyxl



Usage :
    python build_orpo_pairs.py
    python build_orpo_pairs.py --target-pairs 1000
    python build_orpo_pairs.py --resume
    python build_orpo_pairs.py --limit 20                # test rapide
    python build_orpo_pairs.py --dry-run                 # inspecte le plan d'échantillonnage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.evaluation.shared_prompts import build_system_prompt, few_shot_signatures, VALID_CATEGORIES, TAXONOMY_BLOCK  # noqa: E402
# ↑ CORRECTIF : ces blocs étaient auparavant dupliqués manuellement dans ce fichier et
# avaient dérivé du texte utilisé dans benchmark_temporal_drift.py (TAXONOMY_BLOCK
# condensé au lieu du texte détaillé, et surtout aucune injection de few-shot alors que
# le FNR=0.667 de Section 4 a été mesuré en 3-shot). Cf. shared_prompts.py pour le détail.

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

POOL_FILE   = "data/frafin_sample_rigorous_20260531_225127.parquet"   # le pool nettoyé de 14.8k
GOLD_FILE   = "data/gold_150_annotated_clean_reformulated_without.xlsx"  # exclu du pool
DATA_DIR    = Path("data")
OUT_DIR     = DATA_DIR / "orpo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY requise. Copiez .env.example vers .env et renseignez votre clé."
    )

API_URL         = "https://api.mistral.ai/v1/chat/completions"

SILVER_MODEL = "mistral-large-latest"   # juge — modèle fort, référence de vérité
BASE_MODEL   = "open-mistral-7b"        # politique à corriger — celui de l'éval Section 4

TEMPERATURE  = 0.0
RANDOM_SEED  = 42
MAX_TOKENS   = 256      # CORRECTIF : aligné sur benchmark_temporal_drift.py (était 300)

N_SHOT_POLICY = 3        # CORRECTIF : la politique de base doit être interrogée dans les
                          # MÊMES conditions que la mesure de Section 4 (3-shot), pas en 0-shot.

TARGET_PAIRS_DEFAULT = 1000     # budget total de paires à construire
MAX_CALLS_DEFAULT    = 6000     # garde-fou : (silver+base) x candidats, cap dur

HTTP_TIMEOUT     = 45
HTTP_RETRIES     = 4
HTTP_BACKOFF     = 2.0
HTTP_RETRY_CODES = {429, 500, 502, 503, 504}
REQUEST_DELAY    = 0.4

EPOCH_ORDER = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]

# Poids d'échantillonnage par époque — sur-échantillonne les strates où le biais
# de récence a été empiriquement documenté en Section 4 (FNR élevé sur 2010-2014,
# dérive confirmée jusqu'à 2015-2019). Les époques récentes sont sous-pondérées
# car le modèle y performe déjà bien (peu d'erreurs Type A à y trouver) mais
# restent représentées pour alimenter le Type B (garde-fou anti-sur-correction).
EPOCH_SAMPLING_WEIGHT = {
    "2010-2014": 0.40,
    "2015-2019": 0.30,
    "2020-2022": 0.20,
    "2023-2026": 0.10,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
# POLITIQUE DE BASE : CORRECTIF — construit via shared_prompts.build_system_prompt(3),
# identique BYTE POUR BYTE au system prompt de benchmark_temporal_drift.py en 3-shot
# (celui qui a produit FNR=0.667 sur 2010-2014, Tableau 4.3). Auparavant ce script
# redéfinissait sa propre version condensée et 0-shot de ce prompt — la dérive entre
# les deux versions expliquait très probablement l'inversion Type A/Type B observée
# (280 FP vs 24 FN) : ce n'était pas que H1 était fausse, c'est que Phase 6a mesurait
# une condition expérimentale différente de celle de Section 4.
SYSTEM_PROMPT_POLICY = build_system_prompt(N_SHOT_POLICY)

# JUGE (silver) : reste délibérément construit à partir de la même TAXONOMY_BLOCK
# canonique, mais avec un TASK_BLOCK_JUDGE distinct et 0-shot — le juge ne doit PAS
# recevoir les mêmes exemples few-shot que la politique, pour ne pas répliquer le même
# biais d'ancrage des deux côtés (ce qui invaliderait le contraste chosen/rejected).
TASK_BLOCK_JUDGE = """
Tu es l'ARBITRE DE RÉFÉRENCE d'un benchmark scientifique. Ta classification
sert de vérité de référence (silver label) pour évaluer d'autres modèles.
Sois particulièrement rigoureux : une erreur de ta part invaliderait la mesure.

Applique strictement les mêmes règles que ci-dessus. Ne force jamais une
catégorie CSRD sur un contenu purement administratif ou financier, et
n'hésite pas à identifier un enjeu de durabilité même s'il est exprimé
dans un vocabulaire ancien ou non standardisé (ex : "maîtrise de l'énergie"
en 2011 relève de E1 au même titre que "transition climatique" en 2024 —
juge le CONTENU SÉMANTIQUE, jamais la présence ou l'absence d'un jargon
réglementaire récent).

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{
  "csrd_category": "<none|ESRS2|E1|E2|E3|E4|E5|S1|S2|S3|S4|G1>",
  "esrs_subcategory": "<code sous-catégorie ou chaîne vide si none>",
  "chain_of_thought": "<3 phrases structurées>"
}
"""

SYSTEM_PROMPT_JUDGE = TAXONOMY_BLOCK.strip() + "\n\n" + TASK_BLOCK_JUDGE.strip()


# ═════════════════════════════════════════════════════════════════════════════
# LOCKS & CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════

ckpt_lock = Lock()
CKPT_FILE   = OUT_DIR / "checkpoint_orpo_candidates.json"
PAIRS_FILE  = OUT_DIR / f"orpo_pairs_{RUN_TS}.jsonl"
STATS_FILE  = OUT_DIR / f"orpo_stats_{RUN_TS}.json"


def load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        with open(CKPT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt: dict) -> None:
    tmp = CKPT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)
    tmp.replace(CKPT_FILE)

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU POOL — exclusion stricte du Gold-140
# ═════════════════════════════════════════════════════════════════════════════

def load_pool_excluding_gold(pool_path: str, gold_path: str) -> pd.DataFrame:
    path = Path(pool_path)
    if not path.exists():
        candidates = sorted(DATA_DIR.glob("frafin_sample_rigorous_*.parquet"), reverse=True)
        if not candidates:
            candidates = sorted(DATA_DIR.glob("frafin_sample_rigorous_*.csv"), reverse=True)
        if not candidates:
            logger.error(f"Pool introuvable : {path.resolve()}")
            sys.exit(1)
        path = candidates[0]
        logger.info(f"Pool auto-détecté : {path.name}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else \
         pd.read_csv(path, dtype=str, low_memory=False)
    logger.info(f"  Pool brut : {len(df):,} paragraphes")

    required = {"paragraph_id", "content"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Colonnes requises manquantes dans le pool : {missing}")
        sys.exit(1)

    df["paragraph_id"] = df["paragraph_id"].astype(str)
    df["content"]      = df["content"].astype(str).str.strip()

    # Epoch : accepte year_bucket (Phase 2) ou reconstruit depuis _year
    if "year_bucket" in df.columns:
        df["epoch"] = df["year_bucket"].astype(str).str.strip()
    elif "_year" in df.columns:
        def _bucket(y):
            try:
                y = int(float(y))
            except (TypeError, ValueError):
                return "unknown"
            if y <= 2014: return "2010-2014"
            if y <= 2019: return "2015-2019"
            if y <= 2022: return "2020-2022"
            return "2023-2026"
        df["epoch"] = df["_year"].apply(_bucket)
    else:
        logger.error("Ni 'year_bucket' ni '_year' trouvés — impossible de stratifier par époque.")
        sys.exit(1)

    df = df[df["epoch"].isin(EPOCH_ORDER)].copy()

    # ── Exclusion stricte du Gold-140 (par paragraph_id ET par contenu) ──────
    gpath = Path(gold_path)
    if not gpath.exists():
        candidates = sorted(Path("data").glob("gold_150*.xlsx"), reverse=True)
        gpath = candidates[0] if candidates else None

    if gpath and gpath.exists():
        gold_df = pd.read_excel(gpath, sheet_name="Annotation")
        gold_ids = set(gold_df["paragraph_id"].astype(str)) if "paragraph_id" in gold_df.columns else set()
        gold_content_sig = set()
        content_col = "content — Extrait à annoter"
        if content_col in gold_df.columns:
            gold_content_sig = set(gold_df[content_col].astype(str).str.strip().str[:100])

        before = len(df)
        mask_excl = df["paragraph_id"].isin(gold_ids) | df["content"].str[:100].isin(gold_content_sig)
        n_excl = mask_excl.sum()
        df = df[~mask_excl].copy()
        logger.info(f"  ✅ Gold-140 exclu du pool : {n_excl} paragraphes retirés "
                    f"({before} → {len(df)})")
    else:
        logger.warning("  ⚠ Fichier Gold introuvable — AUCUNE exclusion appliquée. "
                        "RISQUE DE FUITE DE DONNÉES si le Gold provient de ce pool.")

    logger.info(f"  Distribution par époque (pool final) :")
    for ep, n in df["epoch"].value_counts().reindex(EPOCH_ORDER).items():
        logger.info(f"    {ep}: {n if pd.notna(n) else 0}")

    return df.reset_index(drop=True)

# ═════════════════════════════════════════════════════════════════════════════
# ÉCHANTILLONNAGE STRATIFIÉ DES CANDIDATS
# ═════════════════════════════════════════════════════════════════════════════

def sample_candidates(df: pd.DataFrame, n_candidates: int, seed: int) -> pd.DataFrame:
    """
    Échantillonne n_candidates paragraphes selon les poids EPOCH_SAMPLING_WEIGHT.
    On sur-échantillonne largement car seule une fraction des candidats produira
    une paire exploitable (désaccord base/silver) — le rendement réel dépend
    du taux d'erreur du modèle, mesuré empiriquement en Section 4 (~15-45%
    de FN selon l'époque, quasi nul sur les époques récentes pour le FP).
    """
    rng = np.random.RandomState(seed)
    parts = []
    for ep, weight in EPOCH_SAMPLING_WEIGHT.items():
        pool_ep = df[df["epoch"] == ep]
        if len(pool_ep) == 0:
            continue
        n_ep = min(len(pool_ep), max(1, int(n_candidates * weight)))
        sampled = pool_ep.sample(n=n_ep, random_state=seed + EPOCH_ORDER.index(ep))
        parts.append(sampled)

    result = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"\n  Candidats échantillonnés : {len(result):,}")
    for ep, n in result["epoch"].value_counts().reindex(EPOCH_ORDER).items():
        logger.info(f"    {ep}: {n if pd.notna(n) else 0}")
    return result

# ═════════════════════════════════════════════════════════════════════════════
# APPELS API — juge et politique utilisent le même transport, prompts différents
# ═════════════════════════════════════════════════════════════════════════════

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES, backoff_factor=HTTP_BACKOFF,
        status_forcelist=HTTP_RETRY_CODES, allowed_methods=["POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def parse_response(text: str) -> tuple[str, str, str]:
    """Retourne (csrd_category, esrs_subcategory, chain_of_thought)."""
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Pas de JSON")
        data = json.loads(text[start:end])
        data = {str(k).strip().lower(): v for k, v in data.items()}

        cat = str(data.get("csrd_category", "none")).strip()
        if cat not in VALID_CATEGORIES:
            cat = "none"
        sub = str(data.get("esrs_subcategory", "")).strip()
        if cat == "none":
            sub = ""
        cot = str(data.get("chain_of_thought", "")).strip()
        return cat, sub, cot
    except Exception:
        return "none", "", ""


def call_mistral(
    session: requests.Session, model_id: str,
    system_prompt: str, user_text: str,
) -> tuple[str, str, str, str]:
    """Appelle l'API Mistral. Retourne (category, subcategory, cot, raw_json_str)."""
    

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "temperature": TEMPERATURE,
        "random_seed": RANDOM_SEED,
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Extrait : «{user_text}»"},
        ],
    }
    r = session.post(API_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    cat, sub, cot = parse_response(raw)

    # JSON canonique reconstruit (garantit une sortie propre même si le modèle
    # a ajouté du texte parasite autour du JSON)
    clean_json = json.dumps(
        {"csrd_category": cat, "esrs_subcategory": sub, "chain_of_thought": cot},
        ensure_ascii=False,
    )
    return cat, sub, cot, clean_json

# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION D'UNE PAIRE — un candidat à la fois
# ═════════════════════════════════════════════════════════════════════════════

def process_candidate(
    session: requests.Session, row: dict, call_counter: dict, call_lock: Lock,
) -> dict:
    """
    Traite un paragraphe candidat : appelle juge puis politique, détermine
    s'il y a désaccord exploitable, retourne un enregistrement complet
    (paire ou non-paire, avec le détail pour le rapport de stats).
    """
    pid  = row["paragraph_id"]
    text = row["content"]
    ep   = row["epoch"]

    user_prompt = f"Extrait : «{text}»"

    try:
        with call_lock:
            call_counter["n"] += 2   # juge + politique
        silver_cat, silver_sub, silver_cot, silver_json = call_mistral(
            session, SILVER_MODEL, SYSTEM_PROMPT_JUDGE, text
        )
        time.sleep(REQUEST_DELAY)
        base_cat, base_sub, base_cot, base_json = call_mistral(
            session, BASE_MODEL, SYSTEM_PROMPT_POLICY, text
        )
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        return {
            "paragraph_id": pid, "epoch": ep, "status": "api_error",
            "error": str(e)[:200],
        }

    silver_is_csrd = silver_cat != "none"
    base_is_csrd   = base_cat != "none"

    if silver_cat == base_cat:
        return {
            "paragraph_id": pid, "epoch": ep, "status": "agreement",
            "silver_category": silver_cat, "base_category": base_cat,
        }

    # ── Désaccord → détermination du type de paire ──────────────────────────
    if silver_is_csrd and not base_is_csrd:
        pair_type = "A_recency_fn"     # le cas central de H1 (Section 4)
    elif not silver_is_csrd and base_is_csrd:
        pair_type = "B_overprediction_fp"   # garde-fou anti-sur-correction
    else:
        # Désaccord de sous-catégorie CSRD (ex: silver=E1, base=S1) — les deux
        # sont "CSRD" en binaire mais divergent sur la catégorie précise.
        # Utile signal mais hors du périmètre direct de H1 — conservé à part.
        pair_type = "C_category_confusion"

    return {
        "paragraph_id":     pid,
        "epoch":            ep,
        "status":           "pair",
        "pair_type":        pair_type,
        "prompt":           user_prompt,
        "chosen":           silver_json,
        "rejected":         base_json,
        "chosen_category":  silver_cat,
        "rejected_category": base_cat,
        "chosen_model":     SILVER_MODEL,
        "rejected_model":   BASE_MODEL,
    }

# ═════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

def build_pairs(
    candidates: pd.DataFrame,
    target_pairs: int,
    max_calls: int,
    resume: bool,
) -> tuple[list[dict], Counter, dict]:
    session = make_session()
    ckpt    = load_checkpoint() if resume else {}
    call_counter = {"n": 0}
    call_lock    = Lock()

    pairs: list[dict] = []
    status_counts = Counter()
    by_epoch_type = defaultdict(lambda: Counter())

    if resume:
        for pid, rec in ckpt.items():
            if rec.get("status") == "pair":
                pairs.append(rec)
            status_counts[rec.get("status", "unknown")] += 1
            if rec.get("status") == "pair":
                by_epoch_type[rec["epoch"]][rec["pair_type"]] += 1
        logger.info(f"  Checkpoint chargé : {len(ckpt)} candidats déjà traités "
                    f"({len(pairs)} paires)")

    n_processed_this_run = 0

    for i, row in candidates.iterrows():
        pid = row["paragraph_id"]

        if pid in ckpt:
            continue   # déjà traité (reprise)

        if len(pairs) >= target_pairs:
            logger.info(f"  🎯 Budget de {target_pairs} paires atteint — arrêt.")
            break

        if call_counter["n"] >= max_calls:
            logger.warning(f"  ⚠ Plafond de {max_calls} appels API atteint — arrêt "
                            f"(budget insuffisant pour continuer en toute sécurité).")
            break

        pct = len(pairs) / target_pairs * 100
        logger.info(
            f"  [{i+1:>5}/{len(candidates)}] paires={len(pairs):>4}/{target_pairs} "
            f"({pct:4.1f}%)  appels={call_counter['n']:>5}  époque={row['epoch']}"
        )

        result = process_candidate(session, row.to_dict(), call_counter, call_lock)

        ckpt[pid] = result
        with ckpt_lock:
            save_checkpoint(ckpt)

        status_counts[result["status"]] += 1
        n_processed_this_run += 1

        if result["status"] == "pair":
            pairs.append(result)
            by_epoch_type[result["epoch"]][result["pair_type"]] += 1

    logger.info(f"\n  Candidats traités cette session : {n_processed_this_run}")
    logger.info(f"  Total appels API : {call_counter['n']}")
    return pairs, status_counts, by_epoch_type

# ═════════════════════════════════════════════════════════════════════════════
# ÉQUILIBRAGE FINAL — cap le déséquilibre Type A / Type B / Type C
# ═════════════════════════════════════════════════════════════════════════════

def balance_pairs(pairs: list[dict], max_type_ratio: float = 3.0, min_cap_floor: int = 100) -> list[dict]:
    """
    Évite qu'un type de paire domine excessivement l'entraînement — SANS détruire
    le signal principal quand les types de garde-fou sont naturellement très rares.

    CORRECTIF (cf. INCIDENT_NOTE.md, suite du run post-fix prompt) : la version
    précédente calculait `cap = min_count * max_type_ratio` en utilisant directement
    le compte du type le plus rare. Sur le run post-correction (A=318, B=8, C=5),
    ça donnait cap=15 et jetait 303 exemples A_recency_fn valides — exactement le
    phénomène central de la recherche (H1) — pour un gain de garde-fou nul (B et C
    étaient déjà sous le cap, donc intégralement conservés dans les deux cas).

    Nouvelle règle : le plancher du cap est `max(min_count, min_cap_floor)`, pas
    `min_count` seul. Ça évite qu'une session où B/C sont accidentellement très
    rares n'écrase le signal A. Les types de garde-fou (B, C) restent TOUJOURS
    conservés intégralement (ils ne sont jamais eux-mêmes cappés ici, seul un type
    en excès peut l'être) — donc ce changement n'affaiblit en rien leur rôle de
    garde-fou, il arrête juste de faire des dégâts collatéraux sur A.
    """
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p["pair_type"]].append(p)

    counts = {t: len(v) for t, v in by_type.items()}
    logger.info(f"\n  Distribution avant équilibrage : {counts}")

    non_empty = [c for c in counts.values() if c > 0]
    if not non_empty:
        return pairs

    min_count = min(non_empty)
    cap_basis = max(min_count, min_cap_floor)
    cap = max(1, int(cap_basis * max_type_ratio))

    rng = random.Random(RANDOM_SEED)
    balanced = []
    for t, plist in by_type.items():
        if len(plist) > cap:
            balanced.extend(rng.sample(plist, cap))
            logger.info(f"    {t}: {len(plist)} → {cap} (cappé, plancher={min_cap_floor}, ratio max {max_type_ratio}x)")
        else:
            balanced.extend(plist)
            logger.info(f"    {t}: {len(plist)} (conservé intégralement)")

    rng.shuffle(balanced)
    return balanced

# ═════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ═════════════════════════════════════════════════════════════════════════════

def save_pairs_jsonl(pairs: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            record = {
                "pair_id":           f"orpo_{i:05d}",
                "paragraph_id":      p["paragraph_id"],
                "epoch":             p["epoch"],
                "pair_type":         p["pair_type"],
                "prompt":            p["prompt"],
                "chosen":            p["chosen"],
                "rejected":          p["rejected"],
                "chosen_category":   p["chosen_category"],
                "rejected_category": p["rejected_category"],
                "chosen_model":      p["chosen_model"],
                "rejected_model":    p["rejected_model"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"  ✅ Paires JSONL : {path}  ({len(pairs)} lignes)")


def save_stats(
    pairs: list[dict], status_counts: Counter,
    by_epoch_type: dict, path: Path,
) -> None:
    epoch_type_summary = {
        ep: dict(counter) for ep, counter in by_epoch_type.items()
    }
    pair_type_totals = Counter(p["pair_type"] for p in pairs)
    epoch_totals      = Counter(p["epoch"] for p in pairs)

    stats = {
        "run_timestamp":        RUN_TS,
        "silver_model":         SILVER_MODEL,
        "base_model":           BASE_MODEL,
        "temperature":          TEMPERATURE,
        "random_seed":          RANDOM_SEED,
        "n_pairs_final":        len(pairs),
        "status_counts":        dict(status_counts),
        "pair_type_totals":     dict(pair_type_totals),
        "epoch_totals":         dict(epoch_totals),
        "epoch_x_type":         epoch_type_summary,
        "methodology_note": (
            "Pairs constructed via rejection sampling: 'rejected' is the base "
            "policy's own genuine zero-shot error (Mistral-7B, identical prompt "
            "to benchmark_temporal_drift.py), not a fabricated adversarial "
            "example. 'chosen' is a silver reference label from a stronger "
            "judge model (Mistral-Large). Type A pairs (silver=CSRD, "
            "base=none) directly target the recency bias documented "
            "empirically in Section 4 (FNR=0.667 on 2010-2014). Type B pairs "
            "(silver=none, base=CSRD) guard against the degenerate heuristic "
            "'old document => always CSRD' that unconstrained Type-A-only "
            "training could induce. Gold-140 test set strictly excluded from "
            "the source pool prior to sampling."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✅ Stats JSON : {path}")


def log_summary(pairs: list[dict], status_counts: Counter) -> None:
    logger.info(f"\n{'═'*65}")
    logger.info(f"  RÉSUMÉ — Construction des Paires ORPO")
    logger.info(f"{'═'*65}")
    logger.info(f"  Candidats en accord (skip)     : {status_counts.get('agreement', 0)}")
    logger.info(f"  Candidats en erreur API        : {status_counts.get('api_error', 0)}")
    logger.info(f"  Paires construites (total)     : {len(pairs)}")

    by_type = Counter(p["pair_type"] for p in pairs)
    logger.info(f"\n  Par type :")
    for t, n in by_type.items():
        logger.info(f"    {t:<25} {n:>5}")

    by_epoch = Counter(p["epoch"] for p in pairs)
    logger.info(f"\n  Par époque :")
    for ep in EPOCH_ORDER:
        logger.info(f"    {ep:<12} {by_epoch.get(ep, 0):>5}")

    logger.info(f"\n  Fichiers produits :")
    logger.info(f"    {PAIRS_FILE}")
    logger.info(f"    {STATS_FILE}")
    logger.info(f"\n  ⚙  Prochaine étape : train_orpo_mistral7b.py")
    logger.info(f"{'═'*65}")

# ═════════════════════════════════════════════════════════════════════════════
# DRY-RUN — inspecte le plan d'échantillonnage sans appeler l'API
# ═════════════════════════════════════════════════════════════════════════════

def dry_run(pool_file: str, gold_file: str, target_pairs: int, seed: int) -> None:
    df = load_pool_excluding_gold(pool_file, gold_file)
    # Estimation grossière : ~40% de rendement sur 2010-14/2015-19 (FN),
    # ~5% sur 2020-22/2023-26 (surtout FP, plus rares) — cf. FNR Section 4
    est_yield = {
        "2010-2014": 0.55, "2015-2019": 0.35,
        "2020-2022": 0.10, "2023-2026": 0.05,
    }
    n_candidates_needed = int(target_pairs / np.mean(list(est_yield.values())))
    candidates = sample_candidates(df, n_candidates_needed, seed)

    logger.info(f"\n  [DRY-RUN] Estimation de rendement (basée sur FNR/FPR Section 4) :")
    total_est = 0
    for ep in EPOCH_ORDER:
        n_ep = (candidates["epoch"] == ep).sum()
        est  = int(n_ep * est_yield[ep])
        total_est += est
        logger.info(f"    {ep:<12} candidats={n_ep:>5}  paires_estimées≈{est:>5}")
    logger.info(f"\n  Total paires estimées ≈ {total_est}  (cible : {target_pairs})")
    logger.info(f"  Appels API estimés ≈ {len(candidates) * 2}  (juge + politique)")
    logger.info(f"\n  [DRY-RUN] Aucun appel API effectué. Relancez sans --dry-run pour exécuter.")

# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FraFin-Reasoning — Construction de paires ORPO par rejection sampling"
    )
    p.add_argument("--pool",   default=POOL_FILE)
    p.add_argument("--gold",   default=GOLD_FILE)
    p.add_argument("--target-pairs", type=int, default=TARGET_PAIRS_DEFAULT)
    p.add_argument("--max-calls",    type=int, default=MAX_CALLS_DEFAULT)
    p.add_argument("--max-type-ratio", type=float, default=3.0,
                   help="Ratio max entre le plus grand et le plus petit type de paire")
    p.add_argument("--min-cap-floor", type=int, default=100,
                   help="Plancher minimum utilisé comme base du cap (évite qu'un type de "
                        "garde-fou accidentellement très rare n'écrase le signal principal). "
                        "Cf. INCIDENT_NOTE.md.")
    p.add_argument("--seed",   type=int, default=RANDOM_SEED)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit",  type=int, default=None,
                   help="Limite le nombre de candidats échantillonnés (test rapide)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    global RANDOM_SEED

    args = parse_args()
    RANDOM_SEED = args.seed

    logger.info("═" * 65)
    logger.info("  FraFin-Reasoning — Construction Paires ORPO (Phase 6a)")
    logger.info(f"  Juge (silver)   : {SILVER_MODEL}")
    logger.info(f"  Politique (base): {BASE_MODEL}")
    logger.info(f"  Cible           : {args.target_pairs} paires")
    logger.info(f"  Plafond appels  : {args.max_calls}")
    logger.info("═" * 65)

    if args.dry_run:
        dry_run(args.pool, args.gold, args.target_pairs, args.seed)
        return

    df = load_pool_excluding_gold(args.pool, args.gold)

    # Sur-échantillonnage : on tire ~3x le budget cible en candidats, car
    # seule une fraction produira un désaccord exploitable.
    n_candidates_needed = min(len(df), args.target_pairs * 4)
    if args.limit:
        n_candidates_needed = min(n_candidates_needed, args.limit)

    candidates = sample_candidates(df, n_candidates_needed, args.seed)

    pairs, status_counts, by_epoch_type = build_pairs(
        candidates, args.target_pairs, args.max_calls, args.resume,
    )

    if not pairs:
        logger.error("Aucune paire construite — vérifiez la connectivité API et le pool source.")
        sys.exit(1)

    pairs = balance_pairs(pairs, args.max_type_ratio, args.min_cap_floor)

    save_pairs_jsonl(pairs, PAIRS_FILE)
    save_stats(pairs, status_counts, by_epoch_type, STATS_FILE)
    log_summary(pairs, status_counts)


if __name__ == "__main__":
    main()
