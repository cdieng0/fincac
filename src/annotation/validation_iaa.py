"""
validation_iaa.py — Accord Inter-Annotateurs (Kappa de Cohen) Humain vs Mistral
═══════════════════════════════════════════════════════════════════════════════════════
Phase 3c du pipeline FraFin-Reasoning — dernière étape avant fusion finale du dataset.

RÔLE DANS LE PIPELINE :
    frafin_gold_500_{TS}.xlsx                 (500 annotés à la main, Phase 3a)
    iaa_mistral_annotations_{TS}.parquet       (200 annotés indépendamment, Phase 3b)
              │
              ▼
    validation_iaa.py   ← CE SCRIPT
              │
              ▼
    iaa_report_{TS}.json                       (κ par variable + IC95% + interprétation)
    iaa_confusion_matrix_csrd_category_{TS}.csv
    iaa_disagreements_{TS}.csv                 (cas à revoir manuellement)

CE QUI EST MESURÉ :
    κ de Cohen calculé séparément pour chaque variable annotée, avec la
    pondération adaptée à son échelle de mesure :

    Variable                      Type        Pondération     Échantillon
    ────────────────────────────────────────────────────────────────────
    csrd_category (PRIMAIRE)      nominal     aucune          n = 200
    pilier (E/S/G/general/none)   nominal     aucune          n = 200  (robustesse)
    esrs_subcategory (complet)    nominal     aucune          n = 200  (n/a = classe)
    esrs_subcategory (condit.)    nominal     aucune          n ≤ 200 (les 2 ≠ none)
    materialite_score             ordinal     quadratique     n = 200  (0-5)
    materialite_financiere        binaire     aucune          n = 200
    materialite_impact            binaire     aucune          n = 200
    market_surprise                ordinal     quadratique     n = 200  (4 niveaux)
    horizon_temporel (complet)    nominal     aucune          n = 200  (n/a = classe)
    horizon_temporel (condit.)    ordinal     quadratique     n ≤ 200 (les 2 non-null)

    Chaque κ est accompagné d'un intervalle de confiance à 95% par bootstrap
    (2000 ré-échantillonnages, seed=42) et d'une interprétation selon l'échelle
    de Landis & Koch (1977).

POURQUOI DEUX VERSIONS POUR esrs_subcategory ET horizon_temporel :
    Ces champs sont structurellement non-définis (None) quand csrd_category=
    'none'. La version "complète" traite None comme une classe à part entière
    (n/a) — mesure honnête sur l'intégralité de l'échantillon, mais nominale
    (perd le crédit ordinal pour horizon_temporel). La version "conditionnelle"
    se restreint aux paires où les deux annotateurs ont identifié un enjeu CSRD
    réel — permet une pondération ordinale propre pour horizon_temporel, au
    prix d'un échantillon réduit. Les deux sont rapportées avec leur n exact.

Dépendances :
    pip install pandas numpy scikit-learn pyarrow openpyxl

Usage :
    python validation_iaa.py
    python validation_iaa.py --gold data/frafin_gold_500_XXX.xlsx \\
                              --mistral-iaa data/iaa_mistral_annotations_XXX.parquet
    python validation_iaa.py --bootstrap-n 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    from src.annotation.csrd_taxonomy import (
        CSRD_CATEGORIES,
        MARKET_SURPRISE_LEVELS,
        TIME_HORIZONS,
        list_all_subcategory_codes,
        get_pilier,
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

BOOTSTRAP_N        = 2000
BOOTSTRAP_SEED      = 42
CI_LEVEL            = 0.95
MIN_PAIRS_WARNING   = 30
TARGET_KAPPA_PRIMARY = 0.75   # seuil annoncé dans la note de stratégie initiale

GOLD_RENAME = {
    "Société":            "author",
    "Année":               "_year",
    "Type de document":   "doc_type",
    "Source (lien)":       "source_url",
    "Extrait à annoter":   "content",
    "IAA ?":                "iaa_flag",
}

FIELDS_TO_COMPARE = [
    "paragraph_id", "content", "csrd_category", "esrs_subcategory",
    "materialite_score", "materialite_financiere", "materialite_impact",
    "market_surprise", "horizon_temporel", "chain_of_thought",
]

# ═════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ═════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_REPORT       = DATA_DIR / f"iaa_report_{RUN_TS}.json"
OUT_CONFUSION    = DATA_DIR / f"iaa_confusion_matrix_csrd_category_{RUN_TS}.csv"
OUT_DISAGREEMENTS = DATA_DIR / f"iaa_disagreements_{RUN_TS}.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# COERCION DE TYPES — Excel/CSV → types canoniques comparables
# ═════════════════════════════════════════════════════════════════════════════

def _bool_from_any(v) -> bool | None:
    if pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("oui", "true", "1", "vrai"):
        return True
    if s in ("non", "false", "0", "faux"):
        return False
    return None


def _opt_str(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def _opt_int(v) -> int | None:
    if pd.isna(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT — annotations humaines (sous-ensemble IAA du GOLD)
# ═════════════════════════════════════════════════════════════════════════════

def load_human_iaa(filepath: str) -> tuple[pd.DataFrame, int]:
    path = Path(filepath)
    if not path.exists() or filepath == "auto":
        candidates = sorted(DATA_DIR.glob("frafin_gold_500_*.xlsx"), reverse=True)
        if not candidates:
            logger.error("Aucun frafin_gold_500_*.xlsx trouvé. Lancez Phase 3a d'abord.")
            sys.exit(1)
        path = candidates[0]
        logger.info(f"GOLD auto-détecté : {path.name}")

    df = pd.read_excel(path, sheet_name="Annotation")
    df = df.rename(columns=GOLD_RENAME)

    if "iaa_flag" not in df.columns:
        logger.error("Colonne 'IAA ?' introuvable dans l'onglet Annotation.")
        sys.exit(1)

    df = df[df["iaa_flag"].astype(str).str.strip().str.lower() == "oui"].copy()
    n_total_iaa = len(df)

    df["csrd_category"] = df["csrd_category"].apply(_opt_str)
    annotated = df[df["csrd_category"].notna()].copy()

    annotated["esrs_subcategory"]       = annotated["esrs_subcategory"].apply(_opt_str)
    annotated["materialite_score"]      = annotated["materialite_score"].apply(_opt_int)
    annotated["materialite_financiere"] = annotated["materialite_financiere"].apply(_bool_from_any)
    annotated["materialite_impact"]     = annotated["materialite_impact"].apply(_bool_from_any)
    annotated["market_surprise"]        = annotated["market_surprise"].apply(_opt_str)
    annotated["horizon_temporel"]       = annotated["horizon_temporel"].apply(_opt_str)
    annotated["chain_of_thought"]       = annotated["chain_of_thought"].apply(_opt_str)
    annotated["paragraph_id"]           = annotated["paragraph_id"].astype(str)

    n_annotated = len(annotated)
    if n_annotated < n_total_iaa:
        logger.warning(
            f"  ⚠ {n_total_iaa - n_annotated} / {n_total_iaa} paragraphes IAA pas "
            f"encore annotés par l'humain — calculs limités à {n_annotated}."
        )
    logger.info(f"  Annotations humaines IAA : {n_annotated} / {n_total_iaa} désignés")
    return annotated, n_total_iaa

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT — annotations Mistral (passe IAA indépendante, Phase 3b)
# ═════════════════════════════════════════════════════════════════════════════

def load_mistral_iaa(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists() or filepath == "auto":
        candidates = sorted(DATA_DIR.glob("iaa_mistral_annotations_*.parquet"), reverse=True)
        if not candidates:
            candidates = sorted(DATA_DIR.glob("iaa_mistral_annotations_*.csv"), reverse=True)
        if not candidates:
            logger.error(
                "Aucun iaa_mistral_annotations_*.parquet/csv trouvé. "
                "Lancez annotation_mistral_mass.py --mode iaa d'abord."
            )
            sys.exit(1)
        path = candidates[0]
        logger.info(f"Mistral IAA auto-détecté : {path.name}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else \
         pd.read_csv(path, dtype=str, low_memory=False)

    df["csrd_category"]          = df["csrd_category"].apply(_opt_str)
    df["esrs_subcategory"]       = df["esrs_subcategory"].apply(_opt_str)
    df["materialite_score"]      = df["materialite_score"].apply(_opt_int)
    df["materialite_financiere"] = df["materialite_financiere"].apply(_bool_from_any)
    df["materialite_impact"]     = df["materialite_impact"].apply(_bool_from_any)
    df["market_surprise"]        = df["market_surprise"].apply(_opt_str)
    df["horizon_temporel"]       = df["horizon_temporel"].apply(_opt_str)
    df["chain_of_thought"]       = df["chain_of_thought"].apply(_opt_str)
    df["paragraph_id"]           = df["paragraph_id"].astype(str)

    logger.info(f"  Annotations Mistral IAA : {len(df)}")
    return df

# ═════════════════════════════════════════════════════════════════════════════
# FUSION DES PAIRES (humain, mistral) PAR paragraph_id
# ═════════════════════════════════════════════════════════════════════════════

def merge_pairs(
    human_df: pd.DataFrame, mistral_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    keep_h = [c for c in FIELDS_TO_COMPARE if c in human_df.columns]
    keep_m = [c for c in FIELDS_TO_COMPARE if c in mistral_df.columns]

    h = human_df[keep_h].add_suffix("_human").rename(
        columns={"paragraph_id_human": "paragraph_id"})
    m = mistral_df[keep_m].add_suffix("_mistral").rename(
        columns={"paragraph_id_mistral": "paragraph_id"})

    merged = h.merge(m, on="paragraph_id", how="inner")

    missing_in_mistral = sorted(set(human_df["paragraph_id"]) - set(mistral_df["paragraph_id"]))
    missing_in_human    = sorted(set(mistral_df["paragraph_id"]) - set(human_df["paragraph_id"]))

    return merged, missing_in_mistral, missing_in_human

# ═════════════════════════════════════════════════════════════════════════════
# KAPPA DE COHEN — calcul générique (nominal ou ordinal) + IC95% bootstrap
# ═════════════════════════════════════════════════════════════════════════════

def interpret_kappa(k: float | None) -> str:
    """Échelle de Landis & Koch (1977)."""
    if k is None:
        return "Indéfini"
    if k < 0:
        return "Pas d'accord (< hasard)"
    if k < 0.20:
        return "Accord négligeable"
    if k < 0.40:
        return "Accord faible"
    if k < 0.60:
        return "Accord modéré"
    if k < 0.80:
        return "Accord substantiel"
    return "Accord quasi-parfait"


def interpret_kappa_en(k: float | None) -> str:
    """Landis & Koch (1977) scale, English wording — for the arXiv text export."""
    if k is None:
        return "undefined"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def bootstrap_kappa_ci(
    y1: np.ndarray, y2: np.ndarray, labels: list, weights: str | None,
    n_boot: int = BOOTSTRAP_N, ci: float = CI_LEVEL, seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    rng = np.random.RandomState(seed)
    n = len(y1)
    if n < 2:
        return None, None

    boot_values: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx], labels=labels, weights=weights)
        except Exception:
            continue
        if k is not None and not np.isnan(k):
            boot_values.append(k)

    if len(boot_values) < n_boot * 0.5:
        # Trop de ré-échantillonnages dégénérés (peu de variabilité) pour un IC fiable
        return None, None

    lo = float(np.percentile(boot_values, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_values, (1 + ci) / 2 * 100))
    return round(lo, 4), round(hi, 4)


def kappa_block(
    y1_raw: pd.Series, y2_raw: pd.Series,
    name: str,
    labels: list | None = None,
    ordinal_order: list | None = None,
    weights_override: str | None = None,
) -> dict:
    """
    Calcule le Kappa de Cohen pour une variable, en mode nominal (labels=...)
    ou ordinal (ordinal_order=..., pondération quadratique par défaut).
    Exclut les paires où l'un des deux annotateurs a une valeur manquante
    pour CETTE variable spécifique (cf. docstring du module pour la logique
    "complet" vs "conditionnel" sur esrs_subcategory et horizon_temporel).
    """
    y1_raw = pd.Series(y1_raw).reset_index(drop=True)
    y2_raw = pd.Series(y2_raw).reset_index(drop=True)
    valid = y1_raw.notna() & y2_raw.notna()
    y1 = y1_raw[valid]
    y2 = y2_raw[valid]
    n = len(y1)

    if n < 2:
        return {
            "name": name, "n": n, "kappa": None,
            "ci_95_low": None, "ci_95_high": None,
            "observed_agreement_pct": None,
            "interpretation": "n insuffisant", "label_space_size": None,
        }

    if ordinal_order is not None:
        rank_map = {v: i for i, v in enumerate(ordinal_order)}
        y1n = y1.map(rank_map)
        y2n = y2.map(rank_map)
        if y1n.isna().any() or y2n.isna().any():
            # Valeurs hors de l'échelle ordinale attendue — on les retire proprement
            ok = y1n.notna() & y2n.notna()
            y1n, y2n = y1n[ok], y2n[ok]
            n = len(y1n)
        label_space = list(range(len(ordinal_order)))
        kw = weights_override or "quadratic"
        y1n = y1n.astype(int).values
        y2n = y2n.astype(int).values
    else:
        label_space = labels
        kw = weights_override
        y1n = y1.values
        y2n = y2.values

    try:
        k = cohen_kappa_score(y1n, y2n, labels=label_space, weights=kw)
        if k is not None and np.isnan(k):
            k = None
    except Exception:
        k = None

    obs_agreement = float((np.asarray(y1n) == np.asarray(y2n)).mean()) if n > 0 else None

    ci_low, ci_high = (None, None)
    if k is not None and n >= 2:
        ci_low, ci_high = bootstrap_kappa_ci(np.asarray(y1n), np.asarray(y2n), label_space, kw)

    return {
        "name": name,
        "n": int(n),
        "kappa": round(float(k), 4) if k is not None else None,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "observed_agreement_pct": round(obs_agreement * 100, 2) if obs_agreement is not None else None,
        "interpretation": interpret_kappa(k),
        "label_space_size": len(label_space) if label_space else None,
    }

# ═════════════════════════════════════════════════════════════════════════════
# CALCUL DE TOUS LES KAPPA
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_kappas(merged: pd.DataFrame) -> dict:
    results: dict[str, dict] = {}

    # 1. csrd_category — PRIMAIRE, nominal, 12 classes
    cat_labels = list(CSRD_CATEGORIES.keys())
    results["csrd_category"] = kappa_block(
        merged["csrd_category_human"], merged["csrd_category_mistral"],
        name="csrd_category (PRIMAIRE)", labels=cat_labels,
    )

    # 2. pilier — dérivé, nominal, robustesse
    pilier_labels = sorted({info["pilier"] for info in CSRD_CATEGORIES.values()})
    pilier_h = merged["csrd_category_human"].map(get_pilier)
    pilier_m = merged["csrd_category_mistral"].map(get_pilier)
    results["pilier"] = kappa_block(
        pilier_h, pilier_m, name="pilier (E/S/G/general/none — robustesse)",
        labels=pilier_labels,
    )

    # 3. esrs_subcategory — version complète (n/a = classe à part entière)
    all_subcodes = list_all_subcategory_codes()
    sub_labels_full = list(dict.fromkeys(all_subcodes + ["n/a"]))   # dédoublonné, ordre stable
    sub_h_full = merged["esrs_subcategory_human"].fillna("n/a").replace("", "n/a")
    sub_m_full = merged["esrs_subcategory_mistral"].fillna("n/a").replace("", "n/a")
    results["esrs_subcategory_full"] = kappa_block(
        sub_h_full, sub_m_full,
        name="esrs_subcategory (complet, n/a inclus)", labels=sub_labels_full,
    )

    # 3b. esrs_subcategory — conditionnel (les deux ≠ 'none')
    mask_cond = (
        (merged["csrd_category_human"] != "none") &
        (merged["csrd_category_mistral"] != "none")
    )
    results["esrs_subcategory_conditional"] = kappa_block(
        merged.loc[mask_cond, "esrs_subcategory_human"],
        merged.loc[mask_cond, "esrs_subcategory_mistral"],
        name="esrs_subcategory (conditionnel, les 2 ≠ none)", labels=all_subcodes,
    )

    # 4. materialite_score — ordinal 0-5, pondération quadratique
    results["materialite_score"] = kappa_block(
        merged["materialite_score_human"], merged["materialite_score_mistral"],
        name="materialite_score (ordinal 0-5)", ordinal_order=list(range(6)),
    )

    # 5. materialite_financiere — binaire
    results["materialite_financiere"] = kappa_block(
        merged["materialite_financiere_human"], merged["materialite_financiere_mistral"],
        name="materialite_financiere (binaire)", labels=[False, True],
    )

    # 6. materialite_impact — binaire
    results["materialite_impact"] = kappa_block(
        merged["materialite_impact_human"], merged["materialite_impact_mistral"],
        name="materialite_impact (binaire)", labels=[False, True],
    )

    # 7. market_surprise — ordinal 4 niveaux
    surprise_order = list(MARKET_SURPRISE_LEVELS.keys())
    results["market_surprise"] = kappa_block(
        merged["market_surprise_human"], merged["market_surprise_mistral"],
        name="market_surprise (ordinal)", ordinal_order=surprise_order,
    )

    # 8. horizon_temporel — version complète (n/a = classe, nominal)
    horizon_h_full = merged["horizon_temporel_human"].fillna("n/a")
    horizon_m_full = merged["horizon_temporel_mistral"].fillna("n/a")
    horizon_labels_full = ["n/a"] + list(TIME_HORIZONS.keys())
    results["horizon_temporel_full"] = kappa_block(
        horizon_h_full, horizon_m_full,
        name="horizon_temporel (complet, n/a inclus, nominal)", labels=horizon_labels_full,
    )

    # 8b. horizon_temporel — conditionnel (les deux non-null, ordinal)
    mask_cond_h = (
        merged["horizon_temporel_human"].notna() & merged["horizon_temporel_mistral"].notna()
    )
    results["horizon_temporel_conditional"] = kappa_block(
        merged.loc[mask_cond_h, "horizon_temporel_human"],
        merged.loc[mask_cond_h, "horizon_temporel_mistral"],
        name="horizon_temporel (conditionnel, ordinal, les 2 non-null)",
        ordinal_order=list(TIME_HORIZONS.keys()),
    )

    return results

# ═════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX & DISAGREEMENTS (champ primaire csrd_category)
# ═════════════════════════════════════════════════════════════════════════════

def export_confusion_matrix(merged: pd.DataFrame, path: Path) -> None:
    labels = list(CSRD_CATEGORIES.keys())
    cm = pd.crosstab(
        merged["csrd_category_human"], merged["csrd_category_mistral"],
        rownames=["humain"], colnames=["mistral"],
    )
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    cm.to_csv(path, encoding="utf-8")
    logger.info(f"  ✅ Matrice de confusion : {path}")


def export_disagreements(merged: pd.DataFrame, path: Path) -> int:
    dis = merged[
        merged["csrd_category_human"] != merged["csrd_category_mistral"]
    ].copy()

    dis["content"] = dis["content_human"].fillna(dis.get("content_mistral"))

    cols = [
        "paragraph_id", "content",
        "csrd_category_human", "csrd_category_mistral",
        "esrs_subcategory_human", "esrs_subcategory_mistral",
        "materialite_score_human", "materialite_score_mistral",
        "market_surprise_human", "market_surprise_mistral",
        "horizon_temporel_human", "horizon_temporel_mistral",
        "chain_of_thought_human", "chain_of_thought_mistral",
    ]
    cols = [c for c in cols if c in dis.columns]
    dis[cols].to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  ✅ Désaccords exportés ({len(dis)}) : {path}")
    return len(dis)

# ═════════════════════════════════════════════════════════════════════════════
# RAPPORT — table console + JSON + texte arXiv
# ═════════════════════════════════════════════════════════════════════════════

def log_kappa_table(kappas: dict) -> None:
    logger.info(f"\n{'═'*82}")
    logger.info(f"  RÉSULTATS — ACCORD INTER-ANNOTATEURS (Kappa de Cohen, humain vs Mistral)")
    logger.info(f"{'═'*82}")
    logger.info(f"  {'Variable':<42} {'n':>5} {'κ':>7} {'IC95%':>16}  Interprétation")
    logger.info(f"  {'-'*80}")
    for r in kappas.values():
        kappa_str = f"{r['kappa']:.3f}" if r["kappa"] is not None else "—"
        ci_str = (
            f"[{r['ci_95_low']:.3f},{r['ci_95_high']:.3f}]"
            if r["ci_95_low"] is not None else "—"
        )
        logger.info(
            f"  {r['name']:<42} {r['n']:>5} {kappa_str:>7} {ci_str:>16}  {r['interpretation']}"
        )
    logger.info(f"{'═'*82}")

    primary = kappas.get("csrd_category", {})
    if primary.get("kappa") is not None:
        status = "✅ ATTEINT" if primary["kappa"] >= TARGET_KAPPA_PRIMARY else "❌ NON ATTEINT"
        logger.info(
            f"\n  Seuil cible κ ≥ {TARGET_KAPPA_PRIMARY} (csrd_category) : {status} "
            f"(κ = {primary['kappa']:.3f}, n = {primary['n']})"
        )


def build_arxiv_text(kappas: dict, n_pairs: int, n_total_iaa: int) -> str:
    primary    = kappas["csrd_category"]
    pilier     = kappas["pilier"]
    materiality = kappas["materialite_score"]

    def fmt(r, key):
        v = r.get(key)
        return f"{v:.3f}" if v is not None else "n/a"

    return (
        f"Inter-annotator agreement was assessed on {n_pairs} of {n_total_iaa} "
        f"designated paragraphs, independently annotated by a human domain "
        f"expert and Mistral Large (temperature=0, fixed seed=42, structured "
        f"JSON output). Cohen's Kappa for the primary CSRD category "
        f"classification (12 nominal classes including 'none') was "
        f"κ = {fmt(primary,'kappa')} (95% CI [{fmt(primary,'ci_95_low')}, "
        f"{fmt(primary,'ci_95_high')}]), indicating "
        f"{interpret_kappa_en(primary['kappa'])} agreement (Landis and Koch, 1977). "
        f"The coarser environmental/social/governance pillar classification "
        f"achieved κ = {fmt(pilier,'kappa')}, and quadratic-weighted agreement "
        f"on the materiality score (0-5 ordinal scale) reached "
        f"κ = {fmt(materiality,'kappa')}. Per-variable confusion matrices and "
        f"all disagreeing cases were exported for manual adjudication."
    )


def save_report(
    kappas: dict, n_pairs: int, n_total_iaa: int,
    missing_in_mistral: list[str], missing_in_human: list[str], n_disagree: int,
) -> None:
    primary_kappa = kappas["csrd_category"]["kappa"]
    report = {
        "run_timestamp": RUN_TS,
        "n_iaa_designated": n_total_iaa,
        "n_valid_pairs": n_pairs,
        "n_missing_in_mistral": len(missing_in_mistral),
        "n_missing_in_human": len(missing_in_human),
        "missing_in_mistral_ids": missing_in_mistral,
        "missing_in_human_ids": missing_in_human,
        "n_disagreements_primary": n_disagree,
        "bootstrap_n_resamples": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "kappa_scale_reference": (
            "Landis & Koch (1977): <0 poor, 0.00-0.20 slight, 0.21-0.40 fair, "
            "0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect"
        ),
        "target_threshold_csrd_category": TARGET_KAPPA_PRIMARY,
        "threshold_met": (primary_kappa is not None and primary_kappa >= TARGET_KAPPA_PRIMARY),
        "kappas": kappas,
        "arxiv_text": build_arxiv_text(kappas, n_pairs, n_total_iaa),
        "output_files": {
            "confusion_matrix": str(OUT_CONFUSION),
            "disagreements": str(OUT_DISAGREEMENTS),
        },
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n  ✅ Rapport complet : {OUT_REPORT}")

# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FraFin-Reasoning — Validation IAA (Kappa de Cohen), Phase 3c"
    )
    p.add_argument("--gold",         default="auto", help="frafin_gold_500_*.xlsx")
    p.add_argument("--mistral-iaa",  default="auto", help="iaa_mistral_annotations_*.parquet")
    p.add_argument("--bootstrap-n",  type=int, default=BOOTSTRAP_N)
    p.add_argument("--min-pairs-warning", type=int, default=MIN_PAIRS_WARNING)
    return p.parse_args()


def main():
    global BOOTSTRAP_N, MIN_PAIRS_WARNING

    args = parse_args()
    BOOTSTRAP_N = args.bootstrap_n
    MIN_PAIRS_WARNING = args.min_pairs_warning

    logger.info("═" * 65)
    logger.info("  FraFin-Reasoning — Validation IAA (Phase 3c)")
    logger.info(f"  Bootstrap : {BOOTSTRAP_N} ré-échantillonnages, seed={BOOTSTRAP_SEED}")
    logger.info("═" * 65)

    human_df, n_total_iaa = load_human_iaa(args.gold)
    mistral_df = load_mistral_iaa(args.mistral_iaa)

    merged, missing_in_mistral, missing_in_human = merge_pairs(human_df, mistral_df)
    n_pairs = len(merged)

    logger.info(f"\n  Paires valides (humain ET Mistral) : {n_pairs} / {n_total_iaa} IAA désignés")
    if missing_in_mistral:
        logger.warning(
            f"  ⚠ {len(missing_in_mistral)} annotés par l'humain mais absents côté "
            f"Mistral (échecs API probables) : {missing_in_mistral[:5]}"
            f"{'...' if len(missing_in_mistral) > 5 else ''}"
        )
    if missing_in_human:
        logger.warning(
            f"  ⚠ {len(missing_in_human)} annotés par Mistral mais pas encore par "
            f"l'humain : {missing_in_human[:5]}{'...' if len(missing_in_human) > 5 else ''}"
        )
    if n_pairs < MIN_PAIRS_WARNING:
        logger.warning(
            f"  ⚠ Échantillon faible (n={n_pairs} < {MIN_PAIRS_WARNING}) — "
            f"intervalles de confiance larges, conclusions à confirmer une fois "
            f"l'annotation des 200 IAA complétée des deux côtés."
        )

    if n_pairs == 0:
        logger.error(
            "  Aucune paire valide — vérifiez que les mêmes paragraph_id sont "
            "présents côté humain (GOLD, IAA='Oui') et côté Mistral (passe iaa)."
        )
        sys.exit(1)

    kappas = compute_all_kappas(merged)
    log_kappa_table(kappas)

    export_confusion_matrix(merged, OUT_CONFUSION)
    n_disagree = export_disagreements(merged, OUT_DISAGREEMENTS)

    save_report(kappas, n_pairs, n_total_iaa, missing_in_mistral, missing_in_human, n_disagree)

    logger.info(f"\n{'═'*65}")
    logger.info(f"  TERMINÉ")
    logger.info(f"{'═'*65}")
    logger.info(
        f"  ⚙  Prochaine étape : fusion GOLD (humain, 500) + mass (Mistral, "
        f"~14 500) en frafin_reasoning_final.parquet, puis push_to_hub()."
    )
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
