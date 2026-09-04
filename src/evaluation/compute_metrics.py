"""
compute_metrics_from_csv.py — Métriques Complètes pour le Papier FraFin-Temporal-Drift
════════════════════════════════════════════════════════════════════════════════════════
Calcule l'ensemble des métriques depuis le CSV de prédictions produit par
benchmark_temporal_drift.py. Aucun appel API. Pas de colonne de confiance requise.

MÉTRIQUES CALCULÉES :
─────────────────────────────────────────────────────────────────────────────────────
A. Performance Globale (binaire CSRD vs none)
   • Accuracy, Précision, Rappel, F1 binaire
   • Kappa de Cohen binaire + IC 95% bootstrap (2000 rééchantillonnages)
   • Macro-F1 (catégories avec ≥5 exemples Gold : E1, ESRS2)
   • Kappa multi-classe (sur toutes les catégories présentes)

B. Dérive Temporelle — THE KEY CONTRIBUTION (H1)
   • F1 binaire par époque (2010-2014 → 2015-2019 → 2020-2022 → 2023-2026)
   • FNR par époque : Faux Négatifs (CSRD labellisé none) — direct test de H1
   • FPR par époque : Faux Positifs (none labellisé CSRD)
   • TD (Temporal Drift) = F1(2023-26) − F1(2010-14) par modèle
   • β_drift : pente OLS de F1 ~ epoch_index
   • ρ_Spearman : corrélation de rang entre epoch et F1 (monotonicity test)
   • Kappa binaire par époque + IC 95% bootstrap (500 rééchantillonnages)

C. Comparaison Inter-Modèles
   • Test de McNemar par paire de modèles (avec correction de continuité)
   • Interprétation binaire : différence significative (p < 0.05) ou non
   • Justification : "Is model A significantly better than B?" → réponse statistique

D. Performance par Catégorie CSRD
   • Précision, Rappel, F1 pour E1 et ESRS2 (seules catégories avec n ≥ 5)
   • Montre où la dérive se produit au niveau des catégories

E. Matrices de Confusion
   • Une matrice par modèle → CSV dans results/confusion_matrices/

JUSTIFICATIONS SCIENTIFIQUES DES MÉTRIQUES :
─────────────────────────────────────────────────────────────────────────────────────
• F1 binaire (CSRD vs none) : métrique principale car dataset déséquilibré (69% none)
  → la précision seule serait trompeuse (toujours prédire "none" = 69%)
• FNR par époque : test direct de H1 — si FNR(2010-14) >> FNR(2023-26), les LLM
  ratent systématiquement les enjeux ESG exprimés en vocabulaire pré-CSRD
• Kappa de Cohen : accord corrigé du hasard, plus interprétable que l'accuracy brute
  pour des datasets multi-classes déséquilibrés
• IC 95% bootstrap : plus robuste que les IC paramétriques sur n=140 (petit échantillon)
• McNemar's test : seul test valide pour comparer deux classificateurs sur LE MÊME
  test set (les erreurs sont dépendantes — les mêmes exemples sont partagés)
• Spearman ρ : teste la monotonie de la dégradation sans hypothèse sur la linéarité
• β_drift OLS : donne une mesure quantitative continue de la dérive, citable comme
  "la performance baisse de β points de F1 par période de 5 ans"

TABLES LATEX GÉNÉRÉES :
─────────────────────────────────────────────────────────────────────────────────────
  Table 1 : Performance globale (Acc, F1, κ, Macro-F1)
  Table 2 : F1 binaire par époque + TD + β + ρ_Spearman → Figure narrative du papier
  Table 3 : FNR par époque — Core de H1 (% documents CSRD manqués)
  Table 4 : McNemar p-values (pairwise, triangle inférieur)
  Table 5 : F1 par catégorie CSRD (E1, ESRS2) par modèle

FICHIERS DE SORTIE :
─────────────────────────────────────────────────────────────────────────────────────
  data/results/metrics_computed_{TS}.json              ← toutes les métriques
  data/results/latex_tables_{TS}.tex                   ← tables copier-coller
  data/results/confusion_matrices/confusion_{mk}.csv   ← une par modèle
  data/results/fnr_by_epoch_{TS}.csv                   ← données Figure 2 du papier

Usage :
    python compute_metrics_from_csv.py --csv data/results/predictions_shot0_20260817_095926.csv
    python compute_metrics_from_csv.py --csv data/results/predictions_shot0_20260817_095926.csv --n-boot 5000
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

EPOCH_ORDER    = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]
EPOCH_INDEX    = {e: i for i, e in enumerate(EPOCH_ORDER)}
RANDOM_SEED    = 42
N_BOOT_GLOBAL  = 2000    # rééchantillonnages pour IC globaux (F1, κ)
N_BOOT_EPOCH   = 500     # rééchantillonnages par époque (n plus petit)
CI_LEVEL       = 0.95

# Catégories assez représentées pour des métriques par catégorie fiables
MIN_GOLD_PER_CAT = 5     # n Gold minimum pour inclure dans Macro-F1 et Table 5

# Noms d'affichage pour les tables LaTeX
MODEL_DISPLAY = {
    "mistral-7b":        r"Mistral-7B",
    "mistral-large":     r"Mistral-Large",
    "gpt-4o":            r"GPT-4o",
    "claude-4.6-sonnet": r"Claude-4.6-Sonnet",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT ET VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)

    required = {"model_key", "pred_category", "gold_label"}
    missing  = required - set(df.columns)
    if missing:
        logger.error(f"Colonnes requises manquantes : {missing}")
        sys.exit(1)

    # Nettoyer les valeurs
    df["gold_label"]    = df["gold_label"].astype(str).str.strip()
    df["pred_category"] = df["pred_category"].astype(str).str.strip()
    df["model_key"]     = df["model_key"].astype(str).str.strip()

    # Époque : accepte 'periode' ou 'epoch'
    if "periode" in df.columns:
        df["epoch"] = df["periode"].astype(str).str.strip()
    elif "epoch" in df.columns:
        df["epoch"] = df["epoch"].astype(str).str.strip()
    else:
        logger.warning("Colonne 'periode'/'epoch' absente — époque inconnue pour toutes les lignes")
        df["epoch"] = "unknown"

    # Variables binaires
    df["gold_is_csrd"] = (df["gold_label"] != "none").astype(int)
    df["pred_is_csrd"] = (df["pred_category"] != "none").astype(int)

    # Erreurs API
    if "error" in df.columns:
        df["_has_error"] = df["error"].notna() & (df["error"].str.strip() != "")
    else:
        df["_has_error"] = False

    logger.info(f"  {len(df)} lignes chargées")
    logger.info(f"  Modèles : {sorted(df['model_key'].unique())}")
    logger.info(f"  Époques : {sorted(df['epoch'].unique())}")
    logger.info(f"  Erreurs API : {df['_has_error'].sum()}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# FONCTIONS STATISTIQUES
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_metric_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
    metric_fn, n_boot: int, seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """IC bootstrap pour une métrique quelconque. Retourne (valeur, lo, hi)."""
    try:
        val = float(metric_fn(y_true, y_pred))
    except Exception:
        return float("nan"), float("nan"), float("nan")

    rng  = np.random.RandomState(seed)
    boot = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            boot.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            pass

    if len(boot) < n_boot * 0.5:
        return val, float("nan"), float("nan")

    lo = float(np.percentile(boot, (1 - CI_LEVEL) / 2 * 100))
    hi = float(np.percentile(boot, (1 + CI_LEVEL) / 2 * 100))
    return round(val, 4), round(lo, 4), round(hi, 4)


def kappa_bin_fn(y_true, y_pred):
    return cohen_kappa_score(y_true.astype(int), y_pred.astype(int))

def f1_bin_fn(y_true, y_pred):
    return f1_score(y_true.astype(int), y_pred.astype(int), zero_division=0)


def mcnemar_test(correct_A: np.ndarray, correct_B: np.ndarray) -> tuple[float, float, str]:
    """
    Test de McNemar avec correction de continuité de Yates.
    Valide pour comparer deux classificateurs sur LE MÊME test set.

    correct_A, correct_B : arrays booléens (1=correct, 0=incorrect)

    Retourne (chi2, p_value, interprétation)
    """
    n10 = int(( correct_A & ~correct_B).sum())   # A correct, B faux
    n01 = int((~correct_A &  correct_B).sum())   # A faux, B correct
    discordant = n01 + n10

    if discordant == 0:
        return 0.0, 1.0, "Identiques"
    if discordant < 25:
        # Test exact binomial (plus fiable sur petit n)
        from scipy.stats import binomtest
        result = binomtest(min(n01, n10), n=discordant, p=0.5, alternative="two-sided")
        p_val = result.pvalue
        chi2  = float("nan")
    else:
        # Chi2 avec correction de continuité (Yates)
        chi2  = (abs(n01 - n10) - 1.0) ** 2 / discordant
        from scipy.stats import chi2 as chi2_dist
        p_val = float(chi2_dist.sf(chi2, df=1))
        chi2  = round(chi2, 3)

    interpretation = "✓ signif. (p<0.05)" if p_val < 0.05 else "n.s."
    return chi2, round(p_val, 4), interpretation


# ═════════════════════════════════════════════════════════════════════════════
# CALCUL DES MÉTRIQUES — UN MODÈLE
# ═════════════════════════════════════════════════════════════════════════════

def compute_model_metrics(df_m: pd.DataFrame) -> dict:
    """Calcule toutes les métriques pour un seul modèle."""

    # Exclure les erreurs API pour les métriques de performance
    df = df_m[~df_m["_has_error"]].copy()
    n_err = df_m["_has_error"].sum()

    y_gold_bin  = df["gold_is_csrd"].values
    y_pred_bin  = df["pred_is_csrd"].values
    y_gold_cat  = df["gold_label"].values
    y_pred_cat  = df["pred_category"].values
    correct_bin = (y_gold_bin == y_pred_bin)

    # ── A. Performance globale ─────────────────────────────────────────────

    acc         = float(accuracy_score(y_gold_cat, y_pred_cat))
    prec_bin    = float(precision_score(y_gold_bin, y_pred_bin, zero_division=0))
    rec_bin     = float(recall_score(y_gold_bin, y_pred_bin, zero_division=0))
    f1_bin, f1_lo, f1_hi = bootstrap_metric_ci(
        y_gold_bin, y_pred_bin, f1_bin_fn, N_BOOT_GLOBAL
    )
    kappa, k_lo, k_hi = bootstrap_metric_ci(
        y_gold_bin, y_pred_bin, kappa_bin_fn, N_BOOT_GLOBAL
    )

    # Kappa multi-classe
    try:
        kappa_mc = float(cohen_kappa_score(y_gold_cat, y_pred_cat))
    except Exception:
        kappa_mc = float("nan")

    # Macro-F1 sur catégories avec n ≥ MIN_GOLD_PER_CAT
    cat_counts  = pd.Series(y_gold_cat).value_counts()
    rich_cats   = cat_counts[cat_counts >= MIN_GOLD_PER_CAT].index.tolist()
    mask_rich   = np.isin(y_gold_cat, rich_cats)
    f1_macro_rich = float(
        f1_score(y_gold_cat[mask_rich], y_pred_cat[mask_rich],
                  average="macro", zero_division=0)
    ) if mask_rich.sum() > 0 else float("nan")

    # FNR global : CSRD manqués / total CSRD
    tp_glob = ((y_gold_bin == 1) & (y_pred_bin == 1)).sum()
    fn_glob = ((y_gold_bin == 1) & (y_pred_bin == 0)).sum()
    fnr_glob = fn_glob / (tp_glob + fn_glob) if (tp_glob + fn_glob) > 0 else float("nan")

    # FPR global : none faussement labellisés CSRD
    tn_glob = ((y_gold_bin == 0) & (y_pred_bin == 0)).sum()
    fp_glob = ((y_gold_bin == 0) & (y_pred_bin == 1)).sum()
    fpr_glob = fp_glob / (tn_glob + fp_glob) if (tn_glob + fp_glob) > 0 else float("nan")

    # ── B. Métriques par époque ────────────────────────────────────────────

    by_epoch = {}
    f1_vals  = []
    fnr_vals = []

    for ep in EPOCH_ORDER:
        sub = df[df["epoch"] == ep]
        if len(sub) == 0:
            continue

        g_b = sub["gold_is_csrd"].values
        p_b = sub["pred_is_csrd"].values
        g_c = sub["gold_label"].values
        p_c = sub["pred_category"].values

        n_tot  = len(sub)
        n_csrd = int(g_b.sum())
        n_none = n_tot - n_csrd

        f1_ep   = float(f1_score(g_b, p_b, zero_division=0))
        prec_ep = float(precision_score(g_b, p_b, zero_division=0))
        rec_ep  = float(recall_score(g_b, p_b, zero_division=0))
        acc_ep  = float(accuracy_score(g_c, p_c))

        # FNR : documents CSRD classés 'none' par le modèle
        fn_ep  = int(((g_b == 1) & (p_b == 0)).sum())
        tp_ep  = int(((g_b == 1) & (p_b == 1)).sum())
        fnr_ep = fn_ep / (fn_ep + tp_ep) if (fn_ep + tp_ep) > 0 else float("nan")

        # FPR : documents none classés CSRD par le modèle
        fp_ep  = int(((g_b == 0) & (p_b == 1)).sum())
        tn_ep  = int(((g_b == 0) & (p_b == 0)).sum())
        fpr_ep = fp_ep / (fp_ep + tn_ep) if (fp_ep + tn_ep) > 0 else float("nan")

        kap_ep, kl_ep, kh_ep = bootstrap_metric_ci(
            g_b, p_b, kappa_bin_fn, N_BOOT_EPOCH,
        )

        by_epoch[ep] = {
            "n":              n_tot,
            "n_csrd_gold":    n_csrd,
            "n_none_gold":    n_none,
            "f1_binary":      round(f1_ep,   3),
            "precision":      round(prec_ep, 3),
            "recall":         round(rec_ep,  3),
            "accuracy":       round(acc_ep,  3),
            "fnr":            round(fnr_ep,  3) if not np.isnan(fnr_ep) else None,
            "fpr":            round(fpr_ep,  3) if not np.isnan(fpr_ep) else None,
            "fn_count":       fn_ep,
            "kappa":          round(kap_ep,  3),
            "kappa_ci_lo":    round(kl_ep,   3),
            "kappa_ci_hi":    round(kh_ep,   3),
        }
        f1_vals.append((EPOCH_INDEX[ep], f1_ep))
        fnr_vals.append((EPOCH_INDEX[ep], fnr_ep if not np.isnan(fnr_ep) else None))

    # ── C. Dérive temporelle ───────────────────────────────────────────────

    epochs_with_data = [ep for ep in EPOCH_ORDER if ep in by_epoch]
    f1_series = [by_epoch[ep]["f1_binary"] for ep in epochs_with_data]

    if len(f1_series) >= 3:
        x_idx = [EPOCH_INDEX[ep] for ep in epochs_with_data]

        # Temporal Drift (TD) : simple différence premier/dernier
        td = float(f1_series[-1] - f1_series[0])

        # OLS slope (β_drift)
        slope, _, r_val, p_val, _ = linregress(x_idx, f1_series)
        beta_drift = float(slope)
        r2_drift   = float(r_val ** 2)
        p_ols      = float(p_val)

        # Spearman ρ (test de monotonie, hypothèse-libre)
        rho, p_rho = spearmanr(x_idx, f1_series)
        rho        = float(rho)
        p_rho      = float(p_rho)
    else:
        td = beta_drift = r2_drift = p_ols = rho = p_rho = float("nan")

    # FNR drift : FNR(2010-14) − FNR(2023-26) → positif = H1 confirmée
    fnr_first = by_epoch.get(EPOCH_ORDER[0], {}).get("fnr")
    fnr_last  = by_epoch.get(EPOCH_ORDER[-1], {}).get("fnr")
    fnr_drift = (
        round(float(fnr_first) - float(fnr_last), 3)
        if fnr_first is not None and fnr_last is not None else None
    )

    # ── D. Performance par catégorie (E1, ESRS2) ──────────────────────────

    per_category = {}
    for cat in rich_cats:
        mask_cat = y_gold_cat == cat
        if mask_cat.sum() < MIN_GOLD_PER_CAT:
            continue
        # Traité en "one-vs-rest"
        g_ovr = (y_gold_cat == cat).astype(int)
        p_ovr = (y_pred_cat == cat).astype(int)
        per_category[cat] = {
            "n_gold":   int(mask_cat.sum()),
            "precision": round(float(precision_score(g_ovr, p_ovr, zero_division=0)), 3),
            "recall":    round(float(recall_score(g_ovr, p_ovr, zero_division=0)), 3),
            "f1":        round(float(f1_score(g_ovr, p_ovr, zero_division=0)), 3),
        }

    return {
        "n_evaluated":         len(df),
        "n_errors":            int(n_err),
        # Globaux
        "accuracy":            round(acc,          3),
        "f1_binary":           round(f1_bin,       3),
        "f1_binary_ci_lo":     round(f1_lo,        3),
        "f1_binary_ci_hi":     round(f1_hi,        3),
        "precision_binary":    round(prec_bin,     3),
        "recall_binary":       round(rec_bin,      3),
        "kappa_binary":        round(kappa,        3),
        "kappa_binary_ci_lo":  round(k_lo,         3),
        "kappa_binary_ci_hi":  round(k_hi,         3),
        "kappa_multiclass":    round(kappa_mc,     3) if not np.isnan(kappa_mc) else None,
        "f1_macro_rich":       round(f1_macro_rich, 3) if not np.isnan(f1_macro_rich) else None,
        "rich_categories":     rich_cats,
        "fnr_global":          round(fnr_glob, 3) if not np.isnan(fnr_glob) else None,
        "fpr_global":          round(fpr_glob, 3) if not np.isnan(fpr_glob) else None,
        # Dérive temporelle
        "temporal_drift_td":   round(td,           3) if not np.isnan(td) else None,
        "beta_drift":          round(beta_drift,   4) if not np.isnan(beta_drift) else None,
        "r2_drift":            round(r2_drift,     4) if not np.isnan(r2_drift) else None,
        "p_ols":               round(p_ols,        4) if not np.isnan(p_ols) else None,
        "spearman_rho":        round(rho,          3) if not np.isnan(rho) else None,
        "spearman_p":          round(p_rho,        4) if not np.isnan(p_rho) else None,
        "fnr_drift":           fnr_drift,
        # Par époque
        "by_epoch":            by_epoch,
        # Par catégorie
        "per_category":        per_category,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TEST DE McNEMAR — TOUTES LES PAIRES
# ═════════════════════════════════════════════════════════════════════════════

def compute_mcnemar_matrix(df: pd.DataFrame, model_list: list[str]) -> dict:
    """
    Teste toutes les paires de modèles sur les exemples communs (même paragraph_id).
    Retourne une matrice triangulaire inférieure de p-values.
    """
    # Pivot : une colonne par modèle, une ligne par exemple
    df_clean = df[~df["_has_error"]].copy()
    df_pivot = df_clean.pivot_table(
        index="paragraph_id", columns="model_key",
        values="pred_is_csrd", aggfunc="first",
    )
    gold_pivot = df_clean.pivot_table(
        index="paragraph_id", columns="model_key",
        values="gold_is_csrd", aggfunc="first",
    )
    # Garder les exemples présents dans tous les modèles
    common_ids = df_pivot.dropna().index
    gold_col   = gold_pivot.loc[common_ids, model_list[0]].values.astype(int)

    results = {}
    for i, mk_a in enumerate(model_list):
        for mk_b in model_list[i+1:]:
            if mk_a not in df_pivot.columns or mk_b not in df_pivot.columns:
                continue
            pred_a = df_pivot.loc[common_ids, mk_a].values.astype(int)
            pred_b = df_pivot.loc[common_ids, mk_b].values.astype(int)
            correct_a = (pred_a == gold_col)
            correct_b = (pred_b == gold_col)
            chi2, p_val, interp = mcnemar_test(correct_a, correct_b)
            results[f"{mk_a}_vs_{mk_b}"] = {
                "chi2": chi2, "p_value": p_val,
                "interpretation": interp,
                "n_common": int(len(common_ids)),
            }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# MATRICES DE CONFUSION
# ═════════════════════════════════════════════════════════════════════════════

def export_confusion_matrices(
    df: pd.DataFrame, model_list: list[str], out_dir: Path
) -> None:
    out_dir.mkdir(exist_ok=True)
    cats = sorted(df["gold_label"].unique())
    for mk in model_list:
        sub  = df[(df["model_key"] == mk) & ~df["_has_error"]]
        y_g  = sub["gold_label"].values
        y_p  = sub["pred_category"].values
        cm   = confusion_matrix(y_g, y_p, labels=cats)
        df_cm = pd.DataFrame(cm, index=cats, columns=cats)
        df_cm.index.name   = "gold \\ pred"
        path = out_dir / f"confusion_{mk}.csv"
        df_cm.to_csv(path, encoding="utf-8")
    logger.info(f"  ✅ Matrices de confusion → {out_dir}/")


def export_fnr_csv(all_metrics: dict, model_list: list[str], out_path: Path) -> None:
    """Exporte FNR par époque — utilisé pour Figure 2 du papier."""
    rows = []
    for mk in model_list:
        m = all_metrics[mk]
        for ep in EPOCH_ORDER:
            ep_d = m["by_epoch"].get(ep, {})
            rows.append({
                "model": mk, "epoch": ep,
                "f1_binary":   ep_d.get("f1_binary"),
                "fnr":         ep_d.get("fnr"),
                "fpr":         ep_d.get("fpr"),
                "n":           ep_d.get("n"),
                "n_csrd_gold": ep_d.get("n_csrd_gold"),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"  ✅ FNR par époque → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FORMATAGE LATEX
# ═════════════════════════════════════════════════════════════════════════════

def _ml(mk: str) -> str:
    """Nom affiché dans les tables LaTeX."""
    return MODEL_DISPLAY.get(mk, mk)


def _f(v, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}"


def latex_table1_overall(all_metrics: dict, model_list: list[str]) -> str:
    """Table 1 — Performance globale."""
    lines = [
        r"\begin{table}[htb]", r"\centering",
        r"\caption{Overall zero-shot classification performance (binary: CSRD vs.~\emph{none}). "
        r"$\kappa_\text{bin}$ with 95\% bootstrap CI (2000 resamples, seed=42). "
        r"Macro-F1$^\dagger$ on categories with $\geq 5$ Gold examples (E1, ESRS2). "
        r"$n_\text{err}$ = API call failures excluded from metrics.}",
        r"\label{tab:overall}", r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Model & $n$ & Acc & F$_1^\text{bin}$ [95\%~CI] & "
        r"$\kappa_\text{bin}$ [95\%~CI] & $\kappa_\text{mc}$ & "
        r"Macro-F$_1^\dagger$ & $n_\text{err}$ \\",
        r"\midrule",
    ]
    for mk in model_list:
        m  = all_metrics[mk]
        f1_ci = f"[{_f(m['f1_binary_ci_lo'])}, {_f(m['f1_binary_ci_hi'])}]"
        k_ci  = f"[{_f(m['kappa_binary_ci_lo'])}, {_f(m['kappa_binary_ci_hi'])}]"
        lines.append(
            f"{_ml(mk)} & {m['n_evaluated']} & "
            f"{_f(m['accuracy'])} & "
            f"{_f(m['f1_binary'])} {f1_ci} & "
            f"{_f(m['kappa_binary'])} {k_ci} & "
            f"{_f(m['kappa_multiclass'])} & "
            f"{_f(m['f1_macro_rich'])} & "
            f"{m['n_errors']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_table2_drift(all_metrics: dict, model_list: list[str]) -> str:
    """Table 2 — F1 binaire par époque + indicateurs de dérive."""
    ep_cols = " & ".join(ep.replace("-", "--") for ep in EPOCH_ORDER)
    lines = [
        r"\begin{table}[htb]", r"\centering",
        r"\caption{Binary F$_1$ (CSRD vs.~\emph{none}) per temporal epoch and "
        r"temporal drift indicators. "
        r"TD $= \text{F}_1(2023\text{--}26) - \text{F}_1(2010\text{--}14)$ "
        r"(positive $\Rightarrow$ recency bias). "
        r"$\hat{\beta}$: OLS slope per epoch unit. "
        r"$\rho_S$: Spearman correlation (monotonicity test).}",
        r"\label{tab:temporal_drift}",
        r"\begin{tabular}{l" + "c" * len(EPOCH_ORDER) + "ccc}",
        r"\toprule",
        f"Model & {ep_cols} & TD$\\uparrow$ & $\\hat{{\\beta}}$ & $\\rho_S$ \\\\",
        r"\midrule",
    ]
    for mk in model_list:
        m  = all_metrics[mk]
        f1s = [_f(m["by_epoch"].get(ep, {}).get("f1_binary")) for ep in EPOCH_ORDER]
        lines.append(
            f"{_ml(mk)} & " + " & ".join(f1s) +
            f" & {_f(m['temporal_drift_td'])} "
            f"& {_f(m['beta_drift'], 4)} "
            f"& {_f(m['spearman_rho'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_table3_fnr(all_metrics: dict, model_list: list[str]) -> str:
    """Table 3 — FNR par époque. Core de H1."""
    ep_cols = " & ".join(ep.replace("-", "--") for ep in EPOCH_ORDER)
    lines = [
        r"\begin{table}[htb]", r"\centering",
        r"\caption{False Negative Rate (FNR) per epoch: proportion of real CSRD documents "
        r"incorrectly classified as \emph{none}. "
        r"FNR$_\Delta$ $=$ FNR(2010--14) $-$ FNR(2023--26): "
        r"positive values confirm H1 (recency bias). "
        r"Lower is better.}",
        r"\label{tab:fnr_epoch}",
        r"\begin{tabular}{l" + "c" * len(EPOCH_ORDER) + "c}",
        r"\toprule",
        f"Model & {ep_cols} & FNR$_\\Delta\\uparrow$ \\\\",
        r"\midrule",
    ]
    for mk in model_list:
        m  = all_metrics[mk]
        fnrs = [_f(m["by_epoch"].get(ep, {}).get("fnr")) for ep in EPOCH_ORDER]
        lines.append(
            f"{_ml(mk)} & " + " & ".join(fnrs) +
            f" & {_f(m['fnr_drift'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(len(EPOCH_ORDER) + 2) + r"}{l}{"
        r"\footnotesize FNR $=$ FN\,/\,(FN$+$TP), computed on CSRD-positive examples per epoch.} \\",
        r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


def latex_table4_mcnemar(mcnemar: dict, model_list: list[str]) -> str:
    """Table 4 — McNemar p-values (triangle inférieur)."""
    n = len(model_list)
    col_spec = "l" + "c" * (n - 1)
    lines = [
        r"\begin{table}[htb]", r"\centering",
        r"\caption{McNemar's test p-values (with Yates continuity correction) "
        r"for pairwise comparison of classifiers on the same test set ($n$~shared). "
        r"* $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$, n.s.~not significant.}",
        r"\label{tab:mcnemar}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    # Header
    header = "Model & " + " & ".join(_ml(mk) for mk in model_list[:-1]) + " \\\\"
    lines += [header, r"\midrule"]

    for i, mk_a in enumerate(model_list[1:], 1):
        row = [_ml(mk_a)]
        for mk_b in model_list[:i]:
            key = f"{mk_b}_vs_{mk_a}"
            alt = f"{mk_a}_vs_{mk_b}"
            entry = mcnemar.get(key) or mcnemar.get(alt)
            if entry is None:
                row.append("—")
            else:
                p = entry["p_value"]
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                row.append(f"{_f(p, 4)} {stars}")
        # Remplir les diagonales supérieures avec —
        row += [""] * (n - 1 - i)
        lines.append(" & ".join(row) + " \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_table5_per_category(all_metrics: dict, model_list: list[str]) -> str:
    """Table 5 — F1 par catégorie CSRD (E1, ESRS2)."""
    # Collect all rich categories across models
    all_rich = sorted({
        cat
        for mk in model_list
        for cat in all_metrics[mk].get("per_category", {}).keys()
    })
    if not all_rich:
        return "% Table 5 : aucune catégorie avec n >= 5 — non générée\n"

    col_spec = "ll" + "c" * len(model_list)
    lines = [
        r"\begin{table}[htb]", r"\centering",
        r"\caption{Per-category F$_1$ (one-vs-rest) for categories with "
        r"$\geq 5$ Gold examples. P = Precision, R = Recall, F = F$_1$.}",
        r"\label{tab:per_category}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Category & Metric & " + " & ".join(_ml(mk) for mk in model_list) + " \\\\",
        r"\midrule",
    ]
    for cat in all_rich:
        for metric, label in [("precision", "P"), ("recall", "R"), ("f1", "F")]:
            vals = []
            for mk in model_list:
                v = all_metrics[mk].get("per_category", {}).get(cat, {}).get(metric)
                vals.append(_f(v))
            prefix = f"\\textbf{{{cat}}}" if metric == "precision" else ""
            lines.append(f"{prefix} & {label} & " + " & ".join(vals) + " \\\\")
        lines.append(r"\midrule" if cat != all_rich[-1] else "")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_table_kappa_epoch(all_metrics: dict, model_list: list[str]) -> str:
    """Table bonus — κ binaire par époque avec IC 95%."""
    ep_cols = " & ".join(ep.replace("-", "--") for ep in EPOCH_ORDER)
    lines = [
        r"\begin{table}[htb]", r"\centering",
        r"\caption{Binary Cohen's $\kappa$ per epoch "
        r"(500-resample bootstrap CI). "
        r"Dashes indicate insufficient class diversity.}",
        r"\label{tab:kappa_epoch}",
        r"\begin{tabular}{l" + "c" * len(EPOCH_ORDER) + "}",
        r"\toprule",
        f"Model & {ep_cols} \\\\",
        r"\midrule",
    ]
    for mk in model_list:
        m   = all_metrics[mk]
        vals = []
        for ep in EPOCH_ORDER:
            ep_d = m["by_epoch"].get(ep, {})
            k    = ep_d.get("kappa")
            kl   = ep_d.get("kappa_ci_lo")
            kh   = ep_d.get("kappa_ci_hi")
            if k is None or np.isnan(k):
                vals.append("—")
            else:
                vals.append(f"{_f(k)} [{_f(kl)},{_f(kh)}]")
        lines.append(f"{_ml(mk)} & " + " & ".join(vals) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# RAPPORT CONSOLE
# ═════════════════════════════════════════════════════════════════════════════

def print_console_report(all_metrics: dict, mcnemar: dict, model_list: list[str]) -> None:
    logger.info(f"\n{'═'*76}")
    logger.info("  RÉSULTATS — FraFin Temporal Drift Benchmark")
    logger.info(f"{'═'*76}")

    logger.info(f"\n  TABLE 1 — Performance globale\n")
    hdr = f"  {'Modèle':<22}{'n':>5}{'Acc':>7}{'F1-bin':>8}{'κ-bin':>8}{'κ-mc':>7}{'MacroF1':>9}{'Err':>5}"
    logger.info(hdr)
    logger.info("  " + "─" * 68)
    for mk in model_list:
        m = all_metrics[mk]
        logger.info(
            f"  {mk:<22}{m['n_evaluated']:>5}"
            f"{_f(m['accuracy']):>7}{_f(m['f1_binary']):>8}"
            f"{_f(m['kappa_binary']):>8}{_f(m['kappa_multiclass']):>7}"
            f"{_f(m['f1_macro_rich']):>9}{m['n_errors']:>5}"
        )

    logger.info(f"\n  TABLE 2 — F1 binaire par époque (dérive temporelle)\n")
    hdr2 = f"  {'Modèle':<22}" + "".join(f"{ep:>12}" for ep in EPOCH_ORDER) + f"{'TD':>7}{'β':>8}{'ρ_S':>7}"
    logger.info(hdr2)
    logger.info("  " + "─" * 76)
    for mk in model_list:
        m = all_metrics[mk]
        row = f"  {mk:<22}"
        for ep in EPOCH_ORDER:
            row += f"{_f(m['by_epoch'].get(ep,{}).get('f1_binary')):>12}"
        row += f"{_f(m['temporal_drift_td']):>7}{_f(m['beta_drift'],4):>8}{_f(m['spearman_rho']):>7}"
        logger.info(row)

    logger.info(f"\n  TABLE 3 — FNR par époque (test direct de H1)\n")
    hdr3 = f"  {'Modèle':<22}" + "".join(f"{ep:>12}" for ep in EPOCH_ORDER) + f"{'FNR_Δ':>8}"
    logger.info(hdr3)
    logger.info("  " + "─" * 68)
    for mk in model_list:
        m = all_metrics[mk]
        row = f"  {mk:<22}"
        for ep in EPOCH_ORDER:
            row += f"{_f(m['by_epoch'].get(ep,{}).get('fnr')):>12}"
        row += f"{_f(m['fnr_drift']):>8}"
        logger.info(row)

    logger.info(f"\n  VÉRIFICATION H1 (biais de récence temporelle) :\n")
    for mk in model_list:
        m  = all_metrics[mk]
        td = m.get("temporal_drift_td")
        rho = m.get("spearman_rho")
        fnr_d = m.get("fnr_drift")
        if td is not None:
            drift_status = "✅ confirmé" if td > 0.08 else ("⚠ modéré" if td > 0 else "❌ non détecté")
            logger.info(
                f"  {mk:<22} TD={td:+.3f}  ρ_S={_f(rho)}  FNR_Δ={_f(fnr_d)}  {drift_status}"
            )

    logger.info(f"\n  TABLE 4 — McNemar (significativité des différences) :\n")
    for key, entry in mcnemar.items():
        parts = key.split("_vs_")
        logger.info(
            f"  {parts[0]:<20} vs {parts[1]:<20} "
            f"p={_f(entry['p_value'],4)}  {entry['interpretation']}"
        )
    logger.info(f"\n{'═'*76}")


# ═════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ═════════════════════════════════════════════════════════════════════════════

def save_all_outputs(
    all_metrics: dict, mcnemar: dict, model_list: list[str],
    df: pd.DataFrame, csv_path: Path
) -> None:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = csv_path.parent

    # JSON
    json_path = out_dir / f"metrics_computed_{ts}.json"
    payload   = {
        "source_csv":    csv_path.name,
        "timestamp":     ts,
        "n_boot_global": N_BOOT_GLOBAL,
        "n_boot_epoch":  N_BOOT_EPOCH,
        "ci_level":      CI_LEVEL,
        "random_seed":   RANDOM_SEED,
        "models":        model_list,
        "metrics":       all_metrics,
        "mcnemar":       mcnemar,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  ✅ JSON        : {json_path}")

    # LaTeX
    tex_path = out_dir / f"latex_tables_{ts}.tex"
    tex_content = "\n\n".join([
        f"% FraFin Temporal Drift — Tables LaTeX",
        f"% Source : {csv_path.name}  |  {ts}",
        f"% Modèles : {', '.join(model_list)}",
        f"% Bootstrap : {N_BOOT_GLOBAL} (global), {N_BOOT_EPOCH} (epoch)",
        "",
        latex_table1_overall(all_metrics, model_list),
        "",
        latex_table2_drift(all_metrics, model_list),
        "",
        latex_table3_fnr(all_metrics, model_list),
        "",
        latex_table4_mcnemar(mcnemar, model_list),
        "",
        latex_table5_per_category(all_metrics, model_list),
        "",
        latex_table_kappa_epoch(all_metrics, model_list),
    ])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    logger.info(f"  ✅ LaTeX       : {tex_path}")

    # Matrices de confusion
    cm_dir = out_dir / "confusion_matrices"
    export_confusion_matrices(df, model_list, cm_dir)

    # FNR par époque (pour Figure 2)
    fnr_path = out_dir / f"fnr_by_epoch_{ts}.csv"
    export_fnr_csv(all_metrics, model_list, fnr_path)

    logger.info(f"\n  Tous les fichiers → {out_dir}/")


# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FraFin — Métriques complètes depuis le CSV de prédictions"
    )
    p.add_argument("--csv", required=True,
                   help="data/results/predictions_shot0_20260817_095926.csv")
    p.add_argument("--n-boot", type=int, default=N_BOOT_GLOBAL,
                   help=f"Rééchantillonnages bootstrap globaux (défaut {N_BOOT_GLOBAL})")
    p.add_argument("--models", nargs="*", default=None,
                   help="Sous-ensemble de modèles à analyser (défaut : tous)")
    return p.parse_args()


def main():
    global N_BOOT_GLOBAL

    args = parse_args()
    N_BOOT_GLOBAL = args.n_boot

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Auto-détection du plus récent
        candidates = sorted(
            list(Path("data/results").glob("predictions_shot0_*.csv")), reverse=True
        )
        if not candidates:
            logger.error(f"Fichier introuvable : {csv_path}")
            sys.exit(1)
        csv_path = candidates[0]
        logger.warning(f"CSV non trouvé → auto-détecté : {csv_path.name}")

    logger.info("═" * 65)
    logger.info("  FraFin — Calcul des métriques (sans confiance)")
    logger.info(f"  CSV    : {csv_path.name}")
    logger.info(f"  Bootstrap : {N_BOOT_GLOBAL} (global) / {N_BOOT_EPOCH} (epoch)")
    logger.info("═" * 65)

    df = load_csv(csv_path)

    model_list = args.models or sorted(df["model_key"].unique())
    logger.info(f"  Modèles analysés : {model_list}\n")

    # ── Métriques par modèle ──────────────────────────────────────────────
    all_metrics = {}
    for mk in model_list:
        logger.info(f"  [{mk}] calcul en cours...")
        df_m = df[df["model_key"] == mk]
        all_metrics[mk] = compute_model_metrics(df_m)
        m = all_metrics[mk]
        logger.info(
            f"    n={m['n_evaluated']} err={m['n_errors']} "
            f"F1={_f(m['f1_binary'])} κ={_f(m['kappa_binary'])} "
            f"TD={_f(m['temporal_drift_td'])} ρ_S={_f(m['spearman_rho'])}"
        )

    # ── McNemar pairwise ─────────────────────────────────────────────────
    logger.info("\n  McNemar pairwise...")
    mcnemar = compute_mcnemar_matrix(df, model_list)

    # ── Rapport console ───────────────────────────────────────────────────
    print_console_report(all_metrics, mcnemar, model_list)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    save_all_outputs(all_metrics, mcnemar, model_list, df, csv_path)


if __name__ == "__main__":
    main()