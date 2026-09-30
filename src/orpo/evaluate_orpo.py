"""
evaluate_orpo.py
=====================
Évalue le modèle ORPO (Phase 6b) sur l'intégralité du gold set (140 paragraphes) —
aucun d'entre eux n'a été vu pendant la collecte de paires ORPO ni pendant
l'entraînement (exclusion stricte assurée par load_pool_excluding_gold dans
build_orpo_pairs.py), donc les 140 constituent un test set entièrement propre
pour ce modèle.

Métrique centrale : le FNR (False Negative Rate — taux de "none" prédit à tort
sur du contenu réellement CSRD-pertinent) PAR ÉPOQUE TEMPORELLE, directement
comparable au FNR de la politique de base par époque (section 4.1, tableau 3 du
papier ; benchmark_temporal_drift.py, open-mistral-7b, 3-shot).

C'est la métrique qui valide (ou infirme) l'objectif réel de cette expérience :
est-ce que la correction ORPO réduit spécifiquement le biais de récence,
epoch par epoch, par rapport à la politique de base ?

Usage:
    python src/orpo/evaluate_orpo.py \
        --adapter-dir outputs/mistral7b-orpo-csrd/final_adapter \
        --gold data/gold_150_annotated_clean_reformulated_without.xlsx \
        --baseline-csv data/results/predictions_shot3_20260818_103646.csv \
        --out data/orpo/eval_report_orpo.json

    --baseline-csv est le CSV écrit par benchmark_temporal_drift.py --n-shot 3. S'il est
    absent, l'évaluation s'exécute quand même et le rapport omet la comparaison.

    Recalcul sans GPU depuis des prédictions archivées (ex. le Run 2) :
    python src/orpo/evaluate_orpo.py \\
        --from-predictions data/results/orpo_run2/eval_predictions.csv \\
        --out data/orpo/eval_report_run2_recomputed.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Rend le paquet `src` importable quand le script est lancé par son chemin
# depuis la racine du dépôt : python src/<module>/<script>.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.shared_prompts import build_system_prompt, VALID_CATEGORIES  # noqa: E402

N_SHOT_POLICY = 3
SYSTEM_PROMPT = build_system_prompt(N_SHOT_POLICY)
EPOCH_ORDER = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def load_orpo_model(adapter_dir: str, base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
    # Imports GPU en lazy — permet de tester load_gold/parse_orpo_output/compute_fnr_by_epoch
    # sans torch/peft/transformers installés (couche données, cf. train_orpo_mistral7b.py
    # qui suit le même principe pour --check-data-only).
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def format_prompt(content: str) -> str:
    """Identique à format_prompt() de train_orpo_mistral7b.py — gabarit Mistral explicite."""
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nExtrait : «{content}» [/INST]"


def generate_prediction(model, tokenizer, content: str, max_new_tokens: int = 220) -> str:
    import torch
    prompt = format_prompt(content)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Parsing (schéma ORPO : JSON brut, pas de balises <reasoning>/<answer>)
# ---------------------------------------------------------------------------

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_orpo_output(text: str) -> dict | None:
    m = JSON_RE.search(text)
    if not m:
        return None
    raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("'", '"')
        cleaned = re.sub(r"\bNone\b", "null", cleaned)
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
    if data.get("csrd_category") not in VALID_CATEGORIES:
        return None
    return data


# ---------------------------------------------------------------------------
# Chargement du gold (classeur écrit par src/publishing/fetch_gold_from_hf.py)
# ---------------------------------------------------------------------------

def load_gold(gold_path: str) -> pd.DataFrame:
    df = pd.read_excel(gold_path, sheet_name="Annotation")
    df = df.rename(columns={
        "content — Extrait à annoter": "content",
        "Période": "epoch",
    })
    return df[["paragraph_id", "content", "epoch", "csrd_category"]].copy()


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def _is_parse_error(pred: pd.Series) -> pd.Series:
    """Sortie non analysable : valeur manquante (écrite par ce script) ou chaîne PARSE_ERROR."""
    return pred.isna() | (pred.astype(str).str.strip() == "PARSE_ERROR")


def compute_fnr_by_epoch(df: pd.DataFrame) -> dict:
    """
    FNR = P(prédit "none" | gold != "none"), calculé par epoch.
    C'est la métrique exacte de la section 4.1 (tableau 3) — reproduite ici pour
    comparaison directe modèle de base (Section 4) vs modèle ORPO (ce script).

    Convention : `fnr` ne compte comme faux négatif que la réponse « none » ; une
    sortie non analysable n'y est pas comptée. `fnr_parse_errors_as_none` applique
    la convention du κ (échec de parsing = aucune catégorie détectée).
    """
    result = {}
    for epoch in EPOCH_ORDER:
        sub = df[(df["epoch"] == epoch) & (df["csrd_category"] != "none")]
        if len(sub) == 0:
            result[epoch] = {"n_relevant": 0, "fnr": None}
            continue
        n_fn = (sub["pred_category"] == "none").sum()
        n_pe = _is_parse_error(sub["pred_category"]).sum()
        result[epoch] = {
            "n_relevant": int(len(sub)),
            "n_false_negative": int(n_fn),
            "fnr": float(n_fn / len(sub)),
            "n_parse_error": int(n_pe),
            "fnr_parse_errors_as_none": float((n_fn + n_pe) / len(sub)),
        }
    return result


def compute_fpr_by_epoch(df: pd.DataFrame) -> dict:
    """
    FPR = P(prédit != "none" | gold == "none"), calculé par époque.
    Métrique manquante dans la version précédente du script -- sans elle,
    un modèle qui prédit systématiquement une catégorie CSRD obtient
    mécaniquement FNR=0.0 partout (succès en trompe-l'œil) sans que rien
    ne signale l'effondrement de la précision sur la classe majoritaire
    'none'. Cf. docs/INCIDENT_NOTE.md, Run 1.

    Convention : `fpr` compte toute réponse autre que « none », sorties non
    analysables comprises — c'est la valeur du tableau 4 du papier (64,9 % pour le
    Run 2). La décomposition sépare les vraies étiquettes CSRD (`n_csrd_label`,
    `fpr_csrd_label_only`) des sorties non analysables (`n_parse_error`).
    """
    result = {}
    for epoch in EPOCH_ORDER:
        sub = df[(df["epoch"] == epoch) & (df["csrd_category"] == "none")]
        if len(sub) == 0:
            result[epoch] = {"n_none_gold": 0, "fpr": None}
            continue
        n_fp = (sub["pred_category"] != "none").sum()
        n_pe = _is_parse_error(sub["pred_category"]).sum()
        result[epoch] = {
            "n_none_gold": int(len(sub)),
            "n_false_positive": int(n_fp),
            "fpr": float(n_fp / len(sub)),
            "n_csrd_label": int(n_fp - n_pe),
            "n_parse_error": int(n_pe),
            "fpr_csrd_label_only": float((n_fp - n_pe) / len(sub)),
        }
    return result


def compute_confusion_summary(df: pd.DataFrame) -> dict:
    """
    Matrice de confusion binaire (CSRD vs none) + distribution des
    catégories prédites -- permet de détecter un collapse vers une seule
    catégorie dominante (ex: le modèle prédit "E1" pour presque tout),
    invisible dans FNR/FPR seuls si le collapse est partiellement correct.
    """
    from sklearn.metrics import confusion_matrix
    gold_bin = (df["csrd_category"] != "none").astype(int)
    pred_bin = (df["pred_category"].fillna("none") != "none").astype(int)
    cm = confusion_matrix(gold_bin, pred_bin, labels=[0, 1]).tolist()
    pred_dist = df["pred_category"].fillna("PARSE_ERROR").value_counts().to_dict()
    return {
        "confusion_matrix_binary": {
            "labels": ["none", "CSRD"],
            "matrix": cm,
            "note": "lignes=gold, colonnes=prédit, ordre [none, CSRD]",
        },
        "predicted_category_distribution": pred_dist,
    }


def _kappa_bootstrap(y_gold, y_pred, n_boot: int = 2000, seed: int = 42, ci: float = 0.95):
    """
    Kappa de Cohen + IC bootstrap -- protocole identique a celui declare
    dans le papier (Section 3.5.2) : B=2000 reechantillonnages, graine 42.
    Necessaire pour remplir les cases "n.d." du Tableau 4.
    """
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    y_gold, y_pred = np.asarray(y_gold), np.asarray(y_pred)
    if len(y_gold) < 2:
        return None, None, None
    try:
        k = float(cohen_kappa_score(y_gold, y_pred))
    except Exception:
        return None, None, None
    rng = np.random.RandomState(seed)
    n = len(y_gold); boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            b = cohen_kappa_score(y_gold[idx], y_pred[idx])
            if not np.isnan(b):
                boots.append(float(b))
        except Exception:
            pass
    if len(boots) < n_boot * 0.5:
        return round(k, 4), None, None
    import numpy as _np
    lo = float(_np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(_np.percentile(boots, (1 + ci) / 2 * 100))
    return round(k, 4), round(lo, 4), round(hi, 4)


def compute_overall_metrics(df: pd.DataFrame) -> dict:
    y_true = df["csrd_category"].tolist()
    y_pred = df["pred_category"].fillna("PARSE_ERROR").tolist()
    labels = sorted(set(y_true) | set(y_pred))

    # Kappa binaire (CSRD vs none), la valeur reportee dans le Tableau 4.
    # ATTENTION A LA CONVENTION : accuracy/f1_macro ci-dessus comptent PARSE_ERROR
    # comme une categorie distincte (jamais correcte). Pour le kappa binaire, un
    # echec de parsing signifie "aucune categorie detectee" = comportement 'none'
    # en production. Les deux conventions sont rapportees pour transparence, afin
    # que le papier n'en melange pas deux sans le dire.
    gold_bin = [1 if str(g).strip() != "none" else 0 for g in y_true]
    pred_bin_as_none = [
        1 if (pd.notna(p) and str(p).strip() not in ("none", "PARSE_ERROR")) else 0
        for p in df["pred_category"]
    ]
    k_none, k_none_lo, k_none_hi = _kappa_bootstrap(gold_bin, pred_bin_as_none)

    return {
        "n_examples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        # Les echecs de parsing sont stockes en NaN dans le CSV, pas
        # sous la chaine "PARSE_ERROR" : l ancienne comparaison renvoyait toujours
        # 0.0 alors que le run 2 presentait 30,7 % d echecs reels. On teste donc
        # les deux representations possibles.
        "parse_failure_rate": (
            df["pred_category"].isna()
            | (df["pred_category"].astype(str).str.strip() == "PARSE_ERROR")
        ).mean(),
        "kappa_binary": k_none,
        "kappa_binary_ci95_low": k_none_lo,
        "kappa_binary_ci95_high": k_none_hi,
        "kappa_convention": "parse_errors_counted_as_none (binary CSRD vs none, bootstrap B=2000, seed=42)",
    }


def compute_baseline_fnr_by_epoch(df_bench: pd.DataFrame, model_key: str, gold_df: pd.DataFrame) -> dict:
    """Recalcule le FNR par époque pour une baseline du CSV (ex. mistral-7b 3-shot = Section 4)."""
    available_keys = df_bench["model_key"].unique().tolist()
    if model_key not in available_keys:
        raise ValueError(
            f"model_key={model_key!r} absent du CSV de baseline. Valeurs disponibles : {available_keys}. "
            "Vérifie --baseline-model-key (ce n'est pas l'identifiant API mais le nom d'affichage du CSV)."
        )
    sub = df_bench[df_bench["model_key"] == model_key].merge(
        gold_df[["paragraph_id", "epoch"]], on="paragraph_id", how="inner"
    )
    result = {}
    for epoch in EPOCH_ORDER:
        e_sub = sub[(sub["epoch"] == epoch) & (sub["gold_label"] != "none")]
        if len(e_sub) == 0:
            result[epoch] = {"n_relevant": 0, "fnr": None}
            continue
        n_fn = (e_sub["pred_category"] == "none").sum()
        result[epoch] = {
            "n_relevant": int(len(e_sub)),
            "n_false_negative": int(n_fn),
            "fnr": float(n_fn / len(e_sub)),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", type=str, default="outputs/mistral7b-orpo-csrd/final_adapter")
    ap.add_argument("--base-model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--gold", type=str, default="data/gold_150_annotated_clean_reformulated_without.xlsx")
    ap.add_argument("--baseline-csv", type=str, default="data/results/predictions_shot3_20260818_103646.csv",
                     help="CSV écrit par benchmark_temporal_drift.py --n-shot 3 (optionnel : "
                          "sans lui, le rapport omet la comparaison à la politique de base)")
    ap.add_argument("--baseline-model-key", type=str, default="mistral-7b",
                     help="Doit correspondre à la valeur model_key du CSV pour la politique de base "
                          "(Section 4) — 'mistral-7b' dans predictions_shot3_*.csv correspond à "
                          "l'identifiant API open-mistral-7b (cf. MODELS dans benchmark_temporal_drift.py), "
                          "PAS la chaîne 'open-mistral-7b' elle-même.")
    ap.add_argument("--out", type=str, default="data/orpo/eval_report_orpo.json")
    ap.add_argument("--check-data-only", action="store_true",
                     help="Valide le chargement du gold + le parsing, sans charger le modèle (pas de GPU requis)")
    ap.add_argument("--from-predictions", default=None, metavar="CSV",
                     help="Recalcule le rapport depuis un CSV de prédictions déjà écrit par ce script "
                          "(ex. data/results/orpo_run2/eval_predictions.csv), sans modèle ni GPU")
    args = ap.parse_args()

    baseline_available = Path(args.baseline_csv).exists()
    if not baseline_available and not args.check_data_only:
        print(f"⚠️  CSV de baseline introuvable ({args.baseline_csv}) : la comparaison à la "
              "politique de base sera omise du rapport. Il est produit par "
              "python src/evaluation/benchmark_temporal_drift.py --n-shot 3.")

    if args.from_predictions:
        gold_df = pd.read_csv(args.from_predictions)
        missing = {"paragraph_id", "epoch", "csrd_category", "pred_category"} - set(gold_df.columns)
        if missing:
            sys.exit(f"❌ Colonnes absentes de {args.from_predictions} : {sorted(missing)}")
        print(f"=== Recalcul depuis {args.from_predictions} "
              f"({len(gold_df)} prédictions, aucun modèle chargé) ===")
        raw_preds_path = args.from_predictions
    else:
        gold_df, raw_preds_path = _predict_on_gold(args)
        if gold_df is None:        # --check-data-only
            return

    _report(args, gold_df, raw_preds_path, baseline_available)


def _predict_on_gold(args):
    """Charge le Gold, l'adaptateur, génère les prédictions et les sauvegarde."""
    print("=== Chargement du gold set complet (140, aucune contamination ORPO) ===")
    gold_df = load_gold(args.gold)
    print(f"{len(gold_df)} paragraphes")
    print(gold_df["epoch"].value_counts().to_dict())

    if args.check_data_only:
        print("\n=== [--check-data-only] Vérification du parser sur des cas synthétiques ===")
        cases = [
            ('{"csrd_category": "E1", "esrs_subcategory": "E1-1", "chain_of_thought": "ok"}', True),
            ("Voici : {'csrd_category': 'none', 'esrs_subcategory': None, 'chain_of_thought': 'x',}", True),
            ("je ne sais pas répondre à ceci", False),
        ]
        for raw, expect_ok in cases:
            parsed = parse_orpo_output(raw)
            status = "✅" if (parsed is not None) == expect_ok else "❌"
            print(f"  {status} {raw[:60]!r} -> {parsed}")
        print("\n✅ Données validées. Relancez sans --check-data-only sur un environnement GPU pour l'inférence.")
        return None, None

    print("=== Chargement du modèle ORPO ===")
    model, tokenizer = load_orpo_model(args.adapter_dir, args.base_model_id)

    print("=== Génération des prédictions (peut prendre plusieurs minutes sur 140 exemples) ===")
    preds = []
    for _, row in gold_df.iterrows():
        raw = generate_prediction(model, tokenizer, row["content"])
        parsed = parse_orpo_output(raw)
        preds.append(parsed["csrd_category"] if parsed else None)
    gold_df["pred_category"] = preds

    # Sauvegarde des prédictions brutes -- indispensable pour tout
    # diagnostic ultérieur (ex: détecter un collapse vers une seule catégorie).
    # L'ancienne version du script ne conservait que les métriques agrégées,
    # rendant tout post-mortem impossible sans relancer l'inférence GPU.
    raw_preds_path = str(Path(args.out).parent / f"raw_predictions_orpo_{Path(args.out).stem}.csv")
    Path(raw_preds_path).parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(raw_preds_path, index=False, encoding="utf-8")
    print(f"  ✅ Prédictions brutes sauvegardées : {raw_preds_path}")
    return gold_df, raw_preds_path


def _report(args, gold_df: pd.DataFrame, raw_preds_path: str, baseline_available: bool) -> None:
    """Calcule les métriques, compare à la politique de base si possible, écrit le rapport."""
    print("=== Métriques globales (modèle ORPO) ===")
    orpo_metrics = compute_overall_metrics(gold_df)
    orpo_fnr = compute_fnr_by_epoch(gold_df)
    orpo_fpr = compute_fpr_by_epoch(gold_df)
    orpo_confusion = compute_confusion_summary(gold_df)
    for k, v in orpo_metrics.items():
        print(f"  {k}: {v}")
    print("  FNR par époque (rate à laquelle un vrai CSRD est manqué) :")
    for epoch, stats in orpo_fnr.items():
        print(f"    {epoch}: {stats}")
    print("  FPR par époque (part des vrais 'none' qui ne reçoivent pas 'none') :")
    for epoch, stats in orpo_fpr.items():
        print(f"    {epoch}: {stats}")
    print("  Distribution des catégories prédites (détection de collapse) :")
    for cat, n in orpo_confusion["predicted_category_distribution"].items():
        print(f"    {cat}: {n}")

    # Garde-fou automatique : alerte explicite si le FPR global dépasse
    # un seuil suspect, plutôt que de laisser un FNR=0.0 flatteur passer inaperçu.
    total_none_fp = sum(v.get("n_false_positive", 0) or 0 for v in orpo_fpr.values())
    total_none_pe = sum(v.get("n_parse_error", 0) or 0 for v in orpo_fpr.values())
    total_none_n  = sum(v.get("n_none_gold", 0) or 0 for v in orpo_fpr.values())
    if total_none_n > 0:
        global_fpr = total_none_fp / total_none_n
        print(f"\n  FPR global (tous les 'none' confondus) : {global_fpr:.3f} "
              f"= {total_none_fp - total_none_pe} étiquettes CSRD + {total_none_pe} sorties "
              f"non analysables, sur {total_none_n}")
        if global_fpr > 0.30:
            print(
                f"  ⚠️  ALERTE : FPR global de {global_fpr:.1%} — le modèle ne répond plus "
                f"« none » sur les paragraphes qui le sont (étiquette CSRD ou sortie non "
                f"analysable). Si le FNR est simultanément proche de 0, ceci indique "
                f"probablement un COLLAPSE, pas une correction réelle du biais de récence. "
                f"Cf. docs/INCIDENT_NOTE.md avant de considérer ce run comme un succès."
            )

    baseline_fnr = None
    if baseline_available:
        print("\n=== Comparaison à la politique de base (section 4.1, mêmes paragraphes) ===")
        df_bench = pd.read_csv(args.baseline_csv)
        baseline_fnr = compute_baseline_fnr_by_epoch(df_bench, args.baseline_model_key, gold_df)
        for epoch, stats in baseline_fnr.items():
            print(f"    {epoch}: {stats}")

    report = {
        "orpo_metrics": orpo_metrics,
        "orpo_fnr_by_epoch": orpo_fnr,
        "orpo_fpr_by_epoch": orpo_fpr,
        "orpo_confusion": orpo_confusion,
        "baseline_fnr_by_epoch": baseline_fnr,
        "baseline_model_key": args.baseline_model_key if baseline_available else None,
        "raw_predictions_csv": raw_preds_path,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== RÉSUMÉ — FNR 2010-2014 (le chiffre central de H1) ===")
    if baseline_fnr is not None:
        print(f"  Politique de base (section 4.1) : {baseline_fnr['2010-2014']['fnr']}")
    print(f"  Modèle ORPO (ce run)           : {orpo_fnr['2010-2014']['fnr']}")
    print(f"\nRapport complet : {args.out}")


if __name__ == "__main__":
    main()
