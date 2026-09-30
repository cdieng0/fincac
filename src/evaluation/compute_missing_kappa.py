"""
compute_missing_kappa.py — Récupère les κ manquants du Tableau 4, SANS RÉ-INFÉRENCE
═══════════════════════════════════════════════════════════════════════════════════════
Contexte : le Tableau 4 du papier porte « n.d. » pour le κ de la politique de base et du
Run 1 ORPO. Or les prédictions individuelles de la BASELINE existent déjà dans le CSV
produit par benchmark_temporal_drift.py — aucun appel API ni GPU n'est nécessaire pour
récupérer son κ. Ce script le calcule directement.

PROTOCOLE — identique à celui déclaré dans le papier (Section 3.5.2), pour que les
valeurs soient comparables au κ = 0,238 déjà rapporté pour le Run 2 :
    • κ binaire : CSRD (toute catégorie ≠ none) vs. none
    • IC 95 % par bootstrap non paramétrique, B = 2000, graine = 42
    • Les lignes en échec de parsing sont traitées comme "none" (comportement du
      classifieur en production : une réponse illisible = aucune catégorie détectée).
      L'option --drop-parse-errors permet de les exclure à la place, pour vérifier
      la sensibilité du résultat à ce choix.

CE QUE CE SCRIPT PEUT ET NE PEUT PAS FAIRE :
    ✅ κ de la politique de base  → depuis predictions_shot*.csv (benchmark)  [gratuit]
    ✅ κ du Run 2 ORPO            → depuis raw_predictions_orpo_*.csv         [gratuit]
    ❌ κ du Run 1 ORPO            → ses prédictions individuelles n'ont jamais été
       sauvegardées (le CSV brut a été ajouté au script d'évaluation après ce run).
       Pour l'obtenir, il faut relancer evaluate_orpo.py en pointant --adapter-dir
       vers l'adaptateur du Run 1, s'il existe encore sur disque.

Usage :
    # Baseline (3-shot) depuis le CSV de benchmark
    python src/evaluation/compute_missing_kappa.py --csv data/results/predictions_shot3_XXXX.csv --model-key mistral-7b

    # Zero-shot pour comparaison
    python src/evaluation/compute_missing_kappa.py --csv data/results/predictions_shot0_XXXX.csv --model-key mistral-7b

    # Run 2 ORPO (re-vérification du 0,238 déjà rapporté)
    python src/evaluation/compute_missing_kappa.py --csv data/orpo/raw_predictions_orpo_eval_report_orpo.csv

    # Tous les modèles d'un CSV de benchmark d'un coup
    python src/evaluation/compute_missing_kappa.py --csv data/results/predictions_shot0_XXXX.csv --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

N_BOOT = 2000
SEED = 42
CI = 0.95


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Détecte les colonnes gold et prédiction selon le format du CSV.
      • benchmark_temporal_drift.py  → gold_label     / pred_category
      • evaluate_orpo.py           → csrd_category  / pred_category
    """
    gold_col = next((c for c in ("gold_label", "csrd_category") if c in df.columns), None)
    pred_col = next((c for c in ("pred_category", "prediction") if c in df.columns), None)
    if gold_col is None or pred_col is None:
        print(f"❌ Colonnes introuvables. Colonnes présentes : {list(df.columns)}")
        sys.exit(1)
    return gold_col, pred_col


def kappa_with_ci(y_gold: np.ndarray, y_pred: np.ndarray) -> dict:
    """κ de Cohen + IC bootstrap, protocole exact du papier."""
    if len(y_gold) < 2:
        return {"kappa": None, "ci_low": None, "ci_high": None, "n": int(len(y_gold))}

    kappa = float(cohen_kappa_score(y_gold, y_pred))
    rng = np.random.RandomState(SEED)
    n = len(y_gold)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        try:
            k = cohen_kappa_score(y_gold[idx], y_pred[idx])
            if not np.isnan(k):
                boots.append(float(k))
        except Exception:
            pass

    if len(boots) < N_BOOT * 0.5:
        return {"kappa": round(kappa, 4), "ci_low": None, "ci_high": None, "n": n}

    lo = float(np.percentile(boots, (1 - CI) / 2 * 100))
    hi = float(np.percentile(boots, (1 + CI) / 2 * 100))
    return {
        "kappa": round(kappa, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n": n,
    }


def analyse(df: pd.DataFrame, label: str, drop_parse_errors: bool) -> dict:
    gold_col, pred_col = detect_columns(df)

    d = df.copy()
    d[gold_col] = d[gold_col].astype(str).str.strip()

    n_total = len(d)
    # Les échecs de parsing apparaissent en NaN ou "PARSE_ERROR" selon le script source
    is_parse_error = d[pred_col].isna() | (
        d[pred_col].astype(str).str.strip().isin(["PARSE_ERROR", "nan", ""])
    )
    n_parse_err = int(is_parse_error.sum())

    if drop_parse_errors:
        d = d[~is_parse_error].copy()
        note = f"{n_parse_err} échec(s) de parsing EXCLU(S)"
    else:
        d.loc[is_parse_error, pred_col] = "none"
        note = f"{n_parse_err} échec(s) de parsing traité(s) comme 'none'"

    d[pred_col] = d[pred_col].astype(str).str.strip()

    gold_bin = (d[gold_col] != "none").astype(int).values
    pred_bin = (d[pred_col] != "none").astype(int).values

    res_bin = kappa_with_ci(gold_bin, pred_bin)
    res_multi = kappa_with_ci(d[gold_col].values, d[pred_col].values)

    acc = float(accuracy_score(d[gold_col], d[pred_col]))
    labels = sorted(set(d[gold_col]) | set(d[pred_col]))
    f1m = float(f1_score(d[gold_col], d[pred_col], labels=labels,
                          average="macro", zero_division=0))

    print(f"\n{'═' * 68}")
    print(f"  {label}")
    print(f"{'═' * 68}")
    print(f"  n évalué            : {res_bin['n']} / {n_total}   ({note})")
    print(f"  Accuracy (12 cl.)   : {acc:.3f}")
    print(f"  F1 macro            : {f1m:.3f}")
    print()
    ci_b = (f"[{res_bin['ci_low']:.3f}, {res_bin['ci_high']:.3f}]"
            if res_bin["ci_low"] is not None else "IC non estimable")
    print(f"  κ BINAIRE (CSRD vs none) : {res_bin['kappa']:.3f}  {ci_b}")
    print("      ← c'est la valeur à reporter dans le Tableau 4 du papier")
    km = res_multi["kappa"]
    print(f"  κ multiclasse (12 cat.)  : {km:.3f}" if km is not None else
          "  κ multiclasse            : non estimable")

    return {
        "label": label,
        "n_evaluated": res_bin["n"],
        "n_total": n_total,
        "n_parse_errors": n_parse_err,
        "parse_error_handling": "dropped" if drop_parse_errors else "treated_as_none",
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "kappa_binary": res_bin["kappa"],
        "kappa_binary_ci95": [res_bin["ci_low"], res_bin["ci_high"]],
        "kappa_multiclass": res_multi["kappa"],
        "bootstrap": {"n_resamples": N_BOOT, "seed": SEED},
    }


def main():
    p = argparse.ArgumentParser(
        description="Calcule les κ manquants du Tableau 4 depuis des CSV existants (0 appel API)"
    )
    p.add_argument("--csv", required=True, help="CSV de prédictions (benchmark ou eval ORPO)")
    p.add_argument("--model-key", default=None,
                   help="Filtre sur un modèle (CSV de benchmark multi-modèles), ex. mistral-7b")
    p.add_argument("--all-models", action="store_true",
                   help="Traite chaque model_key du CSV séparément")
    p.add_argument("--drop-parse-errors", action="store_true",
                   help="Exclut les échecs de parsing au lieu de les traiter comme 'none'")
    p.add_argument("--out", default=None, help="Fichier JSON de sortie (optionnel)")
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"❌ Fichier introuvable : {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"CSV chargé : {path.name}  ({len(df)} lignes)")
    print(f"Protocole  : κ binaire, bootstrap B={N_BOOT}, graine={SEED} "
          f"(identique au papier, Section 3.5.2)")

    results = []

    if args.all_models and "model_key" in df.columns:
        for mk in sorted(df["model_key"].dropna().unique()):
            sub = df[df["model_key"] == mk]
            results.append(analyse(sub, f"Modèle : {mk}", args.drop_parse_errors))
    elif args.model_key:
        if "model_key" not in df.columns:
            print("⚠  Ce CSV n'a pas de colonne 'model_key' — --model-key ignoré.")
            results.append(analyse(df, path.stem, args.drop_parse_errors))
        else:
            available = sorted(df["model_key"].dropna().unique())
            if args.model_key not in available:
                print(f"❌ model_key={args.model_key!r} absent. Disponibles : {available}")
                sys.exit(1)
            sub = df[df["model_key"] == args.model_key]
            results.append(analyse(sub, f"Modèle : {args.model_key}", args.drop_parse_errors))
    else:
        results.append(analyse(df, path.stem, args.drop_parse_errors))

    print(f"\n{'═' * 68}")
    print("  À REPORTER DANS LE TABLEAU 4")
    print(f"{'═' * 68}")
    for r in results:
        kb, ci = r["kappa_binary"], r["kappa_binary_ci95"]
        if kb is not None and ci[0] is not None:
            print(f"  {r['label']:<34} κ = {kb:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        elif kb is not None:
            print(f"  {r['label']:<34} κ = {kb:.3f}  (IC non estimable)")
    print(f"{'═' * 68}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"source_csv": str(path), "results": results}, f,
                       indent=2, ensure_ascii=False)
        print(f"\n  ✅ JSON écrit : {args.out}")


if __name__ == "__main__":
    main()
