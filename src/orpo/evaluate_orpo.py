"""
06_evaluate_orpo.py
=====================
Évalue le modèle ORPO (Phase 6b) sur l'intégralité du gold set (140 paragraphes) —
aucun d'entre eux n'a été vu pendant la collecte de paires ORPO ni pendant
l'entraînement (exclusion stricte assurée par load_pool_excluding_gold dans
build_orpo_pairs.py), donc les 140 constituent un test set entièrement propre
pour ce modèle (contrairement au modèle QLoRA-SFT du dossier principal, qui a
utilisé 103/140 en entraînement).

Métrique centrale : le FNR (False Negative Rate — taux de "none" prédit à tort
sur du contenu réellement CSRD-pertinent) PAR ÉPOQUE TEMPORELLE, directement
comparable au chiffre FNR=0.667 sur 2010-2014 documenté en Section 4
(benchmark_temporal_drift.py, modèle de base open-mistral-7b, 3-shot).

C'est la métrique qui valide (ou infirme) l'objectif réel de cette expérience :
est-ce que la correction ORPO réduit spécifiquement le biais de récence,
epoch par epoch, par rapport à la politique de base ?

Usage:
    python 06_evaluate_orpo.py \
        --adapter-dir outputs/mistral7b-orpo-csrd/final_adapter \
        --gold data/gold_150_annotated_clean_reformulated_without.xlsx \
        --baseline-csv data/predictions_shot3_20260818_103646.csv \
        --out data/orpo/eval_report_orpo.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
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
# Chargement du gold (identique à 01_data_preparation.py, colonnes minimales)
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

def compute_fnr_by_epoch(df: pd.DataFrame) -> dict:
    """
    FNR = P(prédit "none" | gold != "none"), calculé par epoch.
    C'est la métrique exacte de Section 4 (Tableau 4.3) — reproduite ici pour
    comparaison directe modèle de base (Section 4) vs modèle ORPO (ce script).
    """
    result = {}
    for epoch in EPOCH_ORDER:
        sub = df[(df["epoch"] == epoch) & (df["csrd_category"] != "none")]
        if len(sub) == 0:
            result[epoch] = {"n_relevant": 0, "fnr": None}
            continue
        n_fn = (sub["pred_category"] == "none").sum()
        result[epoch] = {
            "n_relevant": int(len(sub)),
            "n_false_negative": int(n_fn),
            "fnr": float(n_fn / len(sub)),
        }
    return result


def compute_overall_metrics(df: pd.DataFrame) -> dict:
    y_true = df["csrd_category"].tolist()
    y_pred = df["pred_category"].fillna("PARSE_ERROR").tolist()
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n_examples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "parse_failure_rate": (df["pred_category"] == "PARSE_ERROR").mean(),
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
    ap.add_argument("--baseline-csv", type=str, default="data/predictions_shot3_20260818_103646.csv")
    ap.add_argument("--baseline-model-key", type=str, default="mistral-7b",
                     help="Doit correspondre à la valeur model_key du CSV pour la politique de base "
                          "(Section 4) — 'mistral-7b' dans predictions_shot3_*.csv correspond à "
                          "l'identifiant API open-mistral-7b (cf. MODELS dans benchmark_temporal_drift.py), "
                          "PAS la chaîne 'open-mistral-7b' elle-même.")
    ap.add_argument("--out", type=str, default="data/orpo/eval_report_orpo.json")
    ap.add_argument("--check-data-only", action="store_true",
                     help="Valide le chargement du gold + le parsing, sans charger le modèle (pas de GPU requis)")
    args = ap.parse_args()

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
        return

    print("=== Chargement du modèle ORPO ===")
    model, tokenizer = load_orpo_model(args.adapter_dir, args.base_model_id)

    print("=== Génération des prédictions (peut prendre plusieurs minutes sur 140 exemples) ===")
    preds = []
    for _, row in gold_df.iterrows():
        raw = generate_prediction(model, tokenizer, row["content"])
        parsed = parse_orpo_output(raw)
        preds.append(parsed["csrd_category"] if parsed else None)
    gold_df["pred_category"] = preds

    print("=== Métriques globales (modèle ORPO) ===")
    orpo_metrics = compute_overall_metrics(gold_df)
    orpo_fnr = compute_fnr_by_epoch(gold_df)
    for k, v in orpo_metrics.items():
        print(f"  {k}: {v}")
    print("  FNR par époque:")
    for epoch, stats in orpo_fnr.items():
        print(f"    {epoch}: {stats}")

    print("\n=== Comparaison à la politique de base (Section 4, mêmes paragraphes) ===")
    df_bench = pd.read_csv(args.baseline_csv)
    baseline_fnr = compute_baseline_fnr_by_epoch(df_bench, args.baseline_model_key, gold_df)
    for epoch, stats in baseline_fnr.items():
        print(f"    {epoch}: {stats}")

    report = {
        "orpo_metrics": orpo_metrics,
        "orpo_fnr_by_epoch": orpo_fnr,
        "baseline_fnr_by_epoch": baseline_fnr,
        "baseline_model_key": args.baseline_model_key,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== RÉSUMÉ — FNR 2010-2014 (le chiffre central de H1) ===")
    print(f"  Politique de base (Section 4) : {baseline_fnr['2010-2014']['fnr']}")
    print(f"  Modèle ORPO (ce run)           : {orpo_fnr['2010-2014']['fnr']}")
    print(f"\nRapport complet : {args.out}")


if __name__ == "__main__":
    main()
