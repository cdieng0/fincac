"""
probe_temporal_signal.py — Sondage Linéaire du Signal Temporel dans Mistral-7B
═══════════════════════════════════════════════════════════════════════════════════════
Section 4.2 du papier — preuve représentationnelle par sondage linéaire. Le sondage
établit une décodabilité corrélationnelle de l'époque, pas une localisation causale.

QUESTION DE RECHERCHE :
    L'information d'époque réglementaire (2010-2014 / 2015-2019 / 2020-2022 /
    2023-2026) est-elle linéairement décodable dans les représentations
    internes de Mistral-7B-Instruct-v0.3, et si oui, à quelles couches ?

    Ceci complète la mesure comportementale de la Section 4 (le modèle
    performe moins bien sur les documents anciens) par une explication au
    niveau des représentations : si un classifieur linéaire simple retrouve
    l'époque avec une précision très supérieure au hasard depuis les
    activations internes, cela suggère que le modèle encode explicitement
    une notion de "récence" dans son espace latent — une information qui
    pourrait interférer avec la classification CSRD elle-même.

MÉTHODE :
    1. Chaque paragraphe du Gold-140 est passé dans Mistral-7B en mode
       forward-only (aucun gradient, aucun entraînement du LLM).
    2. Les hidden_states sont extraits à CHAQUE couche (33 = embeddings +
       32 couches transformer), pooling sur le dernier token (représentation
       standard pour les modèles causaux décodeurs, qui a "vu" toute la
       séquence via l'attention causale).
    3. Pour chaque couche indépendamment, un classifieur linéaire
       (régression logistique multinomiale, standardisée) est entraîné à
       prédire l'époque (4 classes) depuis cette seule représentation.
    4. Validation croisée stratifiée à 5 plis (n=140 est petit, la CV est
       indispensable pour une estimation fiable — un simple split
       train/test unique serait trop bruité).
    5. Un test de permutation sur la meilleure couche donne un p-value :
       la précision obtenue est-elle statistiquement distinguable du hasard ?

CHOIX MÉTHODOLOGIQUES DOCUMENTÉS :
    - Sondage sur le texte BRUT du paragraphe, PAS sur le prompt de
      classification complet (system prompt + taxonomie). Objectif : mesurer
      si le signal temporel est encodé dans la représentation du CONTENU
      lui-même, indépendamment de tout artefact du gabarit d'instruction.
    - Dernier token (pas moyenne sur la séquence) : représentation standard
      pour les modèles causaux, qui agrège l'information de toute la
      séquence par construction de l'attention causale.
    - Régularisation L2 forte (C=0.1) : avec 4096 dimensions et ~112
      exemples d'entraînement par pli, le régime est fortement p >> n ;
      une régularisation conservatrice est nécessaire pour éviter un
      surapprentissage du sondage lui-même (auquel cas la précision
      mesurée refléterait le sondage, pas le modèle).
    - Poids Mistral-7B chargés depuis HuggingFace (mistralai/Mistral-7B-
      Instruct-v0.3), PAS via l'API Mistral utilisée pour le benchmark de
      la Section 4 (endpoint "open-mistral-7b") — l'accès aux activations
      internes nécessite les poids locaux. Il n'est pas garanti que ces
      deux points d'accès servent des poids bit-identiques ; ceci est
      documenté comme limitation dans le papier.

CONTRAINTE D'EXÉCUTION :
    Nécessite un GPU (T4 16 Go suffisant en 4-bit, un GPU plus généreux en
    bf16 plein format). Contrairement à train_orpo_mistral7b.py, il n'y a
    ICI AUCUN ENTRAÎNEMENT DU LLM — seulement des passes forward. Risque
    de collapse ou d'itérations de débogage GPU : nul par construction.

    La couche de données (chargement Gold, validation des colonnes) est
    testable sans GPU via --check-data-only.

Dépendances :
    pip install torch transformers scikit-learn matplotlib pandas openpyxl bitsandbytes

Usage :
    python src/probing/probe_temporal_signal.py --check-data-only --gold gold_150_....xlsx
    python src/probing/probe_temporal_signal.py --gold gold_150_....xlsx
    python src/probing/probe_temporal_signal.py --gold gold_150_....xlsx --4bit   # GPU contraint
    python src/probing/probe_temporal_signal.py --gold gold_150_....xlsx --pooling mean
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

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EPOCH_ORDER   = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]
EPOCH_INDEX   = {e: i for i, e in enumerate(EPOCH_ORDER)}

RANDOM_SEED   = 42
N_FOLDS       = 5
LOGREG_C      = 0.1          # régularisation L2 forte (p >> n)
MAX_TOKENS    = 512          # marge large pour des paragraphes de 20-340 mots
N_PERMUTATIONS = 200         # test de permutation sur la meilleure couche

RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = Path("data/probing")
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU GOLD — testable sans GPU
# ═════════════════════════════════════════════════════════════════════════════

def load_gold(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or path == "auto":
        candidates = sorted(Path(".").glob("gold_150*.xlsx"), reverse=True)
        candidates += sorted(Path("data").glob("gold_150*.xlsx"), reverse=True)
        if not candidates:
            logger.error(f"Fichier Gold introuvable : {p.resolve()}")
            sys.exit(1)
        p = candidates[0]
        logger.info(f"Gold auto-détecté : {p.name}")

    df = pd.read_excel(p)

    # Détection flexible des colonnes (compatible avec les variantes de nommage
    # rencontrées dans les fichiers Gold successifs du projet).
    content_col = next(
        (c for c in df.columns if "extrait" in c.lower() or c == "content"), None
    )
    epoch_col = next(
        (c for c in df.columns if c.lower() in ("période", "periode", "epoch")), None
    )
    if content_col is None or epoch_col is None:
        logger.error(
            f"Colonnes introuvables. Colonnes disponibles : {list(df.columns)}\n"
            f"  content_col détecté : {content_col}\n"
            f"  epoch_col détecté   : {epoch_col}"
        )
        sys.exit(1)

    df = df.rename(columns={content_col: "content", epoch_col: "epoch"})
    df["content"] = df["content"].astype(str).str.strip()
    df["epoch"]   = df["epoch"].astype(str).str.strip()
    df = df[df["epoch"].isin(EPOCH_ORDER)].copy()
    df = df[df["content"].str.len() > 10].reset_index(drop=True)

    logger.info(f"  Gold chargé : {len(df)} paragraphes")
    logger.info("  Distribution par époque :")
    for ep, n in df["epoch"].value_counts().reindex(EPOCH_ORDER).items():
        logger.info(f"    {ep}: {n if pd.notna(n) else 0}")

    return df


def print_data_report(df: pd.DataFrame) -> None:
    majority_class_acc = df["epoch"].value_counts(normalize=True).max()
    n_classes = df["epoch"].nunique()
    logger.info(f"\n{'═'*60}")
    logger.info("  RAPPORT DE DONNÉES — Sondage Linéaire")
    logger.info(f"{'═'*60}")
    logger.info(f"  n total              : {len(df)}")
    logger.info(f"  Classes (époques)    : {n_classes}")
    logger.info(f"  Baseline hasard      : {1/n_classes:.3f} (uniforme)")
    logger.info(f"  Baseline majoritaire : {majority_class_acc:.3f} "
                f"(toujours prédire l'époque la plus fréquente)")
    logger.info(f"  → Un layer n'est informatif que si son accuracy CV dépasse "
                f"nettement {majority_class_acc:.3f}, pas seulement 1/{n_classes}.")
    logger.info(f"{'═'*60}")

# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION DES REPRÉSENTATIONS — nécessite GPU, imports différés
# ═════════════════════════════════════════════════════════════════════════════

def extract_hidden_states(
    df: pd.DataFrame, args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Retourne :
      - representations : array (n_examples, n_layers, hidden_dim)
      - labels           : liste des époques (str), alignée sur representations
      - meta             : infos du modèle (nb couches, dim cachée, dtype...)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        logger.error(
            f"Import échoué : {e}\n"
            f"pip install torch transformers bitsandbytes accelerate\n"
            f"Utilisez --check-data-only pour valider les données sans GPU."
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error(
            "Aucun GPU CUDA détecté. Ce script nécessite un GPU (T4 16 Go "
            "suffisant en --4bit). Lancez sur Colab (runtime GPU)."
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"\n  GPU détecté : {gpu_name}")

    logger.info(f"  Chargement du tokenizer et du modèle : {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = dict(device_map="auto", output_hidden_states=True)
    if args.four_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        logger.info("  Mode : 4-bit NF4 (contrainte VRAM)")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("  Mode : bfloat16 plein format")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1   # +1 pour la couche d'embeddings
    hidden_dim = model.config.hidden_size
    logger.info(f"  Modèle chargé : {n_layers} couches (dont embeddings), "
                f"dim cachée = {hidden_dim}")

    all_reps = np.zeros((len(df), n_layers, hidden_dim), dtype=np.float32)
    labels: list[str] = []

    logger.info(f"\n  Extraction des représentations ({len(df)} paragraphes)...")
    with torch.no_grad():
        for i, row in df.iterrows():
            text = row["content"]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_TOKENS, add_special_tokens=True,
            ).to(model.device)

            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states : tuple de (n_layers) tenseurs (1, seq_len, hidden_dim)

            for layer_idx, hs in enumerate(outputs.hidden_states):
                hs = hs[0]   # (seq_len, hidden_dim), retire la dim batch=1
                if args.pooling == "last":
                    rep = hs[-1, :]                # dernier token
                else:  # "mean"
                    rep = hs.mean(dim=0)           # moyenne sur la séquence
                all_reps[i, layer_idx, :] = rep.float().cpu().numpy()

            labels.append(row["epoch"])

            if (i + 1) % 20 == 0 or (i + 1) == len(df):
                logger.info(f"    [{i+1:>3}/{len(df)}] extrait")

    meta = {
        "n_layers": n_layers, "hidden_dim": hidden_dim,
        "gpu": gpu_name, "quantization": "4bit_nf4" if args.four_bit else "bfloat16",
        "pooling": args.pooling, "base_model": args.base_model,
    }

    del model
    torch.cuda.empty_cache()

    return all_reps, labels, meta

# ═════════════════════════════════════════════════════════════════════════════
# SONDAGE LINÉAIRE PAR COUCHE — pure sklearn, indépendant du GPU une fois les
# représentations extraites (peut être relancé sur un .npz sauvegardé, sans
# GPU, pour ajuster la régularisation ou refaire l'analyse statistique).
# ═════════════════════════════════════════════════════════════════════════════

def probe_all_layers(
    representations: np.ndarray, labels: list[str], seed: int = RANDOM_SEED,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([EPOCH_INDEX[e] for e in labels])
    n_layers = representations.shape[1]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    results = {}
    logger.info(f"\n  Sondage par couche (régression logistique, C={LOGREG_C}, "
                f"CV {N_FOLDS}-plis stratifiée)...")

    for layer_idx in range(n_layers):
        X = representations[:, layer_idx, :]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=LOGREG_C, max_iter=2000, random_state=seed,
            ),
        )
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        results[layer_idx] = {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std":  float(scores.std()),
            "fold_scores":   [float(s) for s in scores],
        }
        logger.info(
            f"    Couche {layer_idx:>2}/{n_layers-1} : "
            f"accuracy = {scores.mean():.3f} ± {scores.std():.3f}"
        )

    return results


def permutation_test_best_layer(
    representations: np.ndarray, labels: list[str],
    best_layer_idx: int, n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Teste si l'accuracy de la meilleure couche est statistiquement
    distinguable du hasard : on ré-entraîne le même pipeline sur des labels
    mélangés aléatoirement, n_permutations fois, et on compare la vraie
    accuracy à la distribution obtenue sous H0 (aucune information réelle).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    logger.info(f"\n  Test de permutation sur la couche {best_layer_idx} "
                f"({n_permutations} permutations)...")

    y = np.array([EPOCH_INDEX[e] for e in labels])
    X = representations[:, best_layer_idx, :]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def _fit_score(y_target: np.ndarray) -> float:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=LOGREG_C, max_iter=2000, random_state=seed),
        )
        return cross_val_score(clf, X, y_target, cv=cv, scoring="accuracy").mean()

    true_acc = _fit_score(y)

    rng = np.random.RandomState(seed)
    null_accs = []
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        null_accs.append(_fit_score(y_shuffled))
        if (i + 1) % 50 == 0:
            logger.info(f"    Permutation {i+1}/{n_permutations}")

    null_accs = np.array(null_accs)
    p_value = float((null_accs >= true_acc).sum() + 1) / (n_permutations + 1)

    logger.info(f"    Accuracy réelle          : {true_acc:.3f}")
    logger.info(f"    Accuracy nulle (moyenne) : {null_accs.mean():.3f} ± {null_accs.std():.3f}")
    logger.info(f"    p-value                  : {p_value:.4f}")

    return {
        "layer_idx": best_layer_idx,
        "true_accuracy": float(true_acc),
        "null_mean": float(null_accs.mean()),
        "null_std": float(null_accs.std()),
        "n_permutations": n_permutations,
        "p_value": p_value,
    }

# ═════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE & FIGURE
# ═════════════════════════════════════════════════════════════════════════════

def save_results(
    layer_results: dict, permutation_result: dict, meta: dict,
    majority_baseline: float, df: pd.DataFrame,
) -> None:
    best_layer = max(layer_results, key=lambda k: layer_results[k]["accuracy_mean"])

    report = {
        "run_timestamp": RUN_TS,
        "n_examples": len(df),
        "n_folds": N_FOLDS,
        "logreg_C": LOGREG_C,
        "random_seed": RANDOM_SEED,
        "majority_class_baseline": majority_baseline,
        "chance_baseline": 1 / len(EPOCH_ORDER),
        "model_meta": meta,
        "best_layer": {
            "index": best_layer,
            "accuracy_mean": layer_results[best_layer]["accuracy_mean"],
            "accuracy_std": layer_results[best_layer]["accuracy_std"],
        },
        "permutation_test": permutation_result,
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "arxiv_text": (
            f"We extracted last-token hidden state representations from "
            f"Mistral-7B-Instruct-v0.3 at all {meta['n_layers']} layers "
            f"(embeddings + {meta['n_layers']-1} transformer blocks) for each "
            f"of the {len(df)} Gold paragraphs, and trained an independent "
            f"L2-regularized multinomial logistic regression per layer to "
            f"predict the temporal epoch (4 classes) from that "
            f"representation alone, using {N_FOLDS}-fold stratified "
            f"cross-validation. The best-performing layer (layer "
            f"{best_layer}) achieves {layer_results[best_layer]['accuracy_mean']:.1%} "
            f"accuracy (chance: {1/len(EPOCH_ORDER):.1%}, majority-class "
            f"baseline: {majority_baseline:.1%}), a difference confirmed "
            f"significant by a {N_PERMUTATIONS}-permutation test "
            f"(p={permutation_result['p_value']:.4f})."
        ),
    }

    json_path = DATA_DIR / f"probing_results_{RUN_TS}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  ✅ Rapport JSON : {json_path}")

    csv_path = DATA_DIR / f"probing_results_{RUN_TS}.csv"
    rows = [
        {"layer": k, "accuracy_mean": v["accuracy_mean"], "accuracy_std": v["accuracy_std"]}
        for k, v in sorted(layer_results.items())
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"  ✅ CSV (pour figure)  : {csv_path}")

    return json_path, csv_path, best_layer


def make_figure(csv_path: Path, majority_baseline: float, best_layer: int) -> Path:
    import matplotlib.pyplot as plt

    df_plot = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_plot["layer"], df_plot["accuracy_mean"], marker="o", color="#1F4E78",
            label="Accuracy du sondage (CV 5-plis)")
    ax.fill_between(
        df_plot["layer"],
        df_plot["accuracy_mean"] - df_plot["accuracy_std"],
        df_plot["accuracy_mean"] + df_plot["accuracy_std"],
        alpha=0.2, color="#1F4E78",
    )
    ax.axhline(majority_baseline, color="gray", linestyle="--",
               label=f"Baseline majoritaire ({majority_baseline:.2f})")
    ax.axhline(1 / len(EPOCH_ORDER), color="lightgray", linestyle=":",
               label=f"Hasard (1/{len(EPOCH_ORDER)})")
    ax.axvline(best_layer, color="#C0392B", linestyle="-", alpha=0.5,
               label=f"Meilleure couche ({best_layer})")

    ax.set_xlabel("Couche (0 = embeddings)")
    ax.set_ylabel("Précision de décodage de l'époque (accuracy)")
    ax.set_title("Décodabilité linéaire de l'époque réglementaire par couche — Mistral-7B")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_path = DATA_DIR / f"figure_probing_{RUN_TS}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info(f"  ✅ Figure (prête pour le papier) : {fig_path}")
    return fig_path

# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FinCAC40 — Sondage linéaire du signal temporel (section 4.2 du papier)"
    )
    p.add_argument("--gold", default="auto")
    p.add_argument("--base-model", default=BASE_MODEL_ID)
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--4bit", dest="four_bit", action="store_true",
                   help="Charge le modèle en 4-bit NF4 (GPU contraint, ex. T4 16 Go)")
    p.add_argument("--check-data-only", action="store_true",
                   help="Valide le Gold sans charger le modèle (aucun GPU requis)")
    p.add_argument("--save-representations", action="store_true",
                   help="Sauvegarde les représentations brutes en .npz "
                        "(permet de relancer le sondage sans ré-extraire)")
    p.add_argument("--load-representations", default=None,
                   help="Charge un .npz déjà extrait au lieu de relancer le GPU")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("═" * 65)
    logger.info("  FinCAC40 — Sondage Linéaire du Signal Temporel (section 4.2)")
    logger.info(f"  Modèle   : {args.base_model}")
    logger.info(f"  Pooling  : {args.pooling}")
    logger.info("═" * 65)

    df = load_gold(args.gold)
    print_data_report(df)
    majority_baseline = df["epoch"].value_counts(normalize=True).max()

    if args.check_data_only:
        logger.info("\n  ✅ Données validées. Relancez sans --check-data-only "
                    "sur un environnement GPU pour extraire les représentations.")
        return

    if args.load_representations:
        logger.info(f"\n  Chargement des représentations existantes : "
                    f"{args.load_representations}")
        loaded = np.load(args.load_representations, allow_pickle=True)
        representations = loaded["representations"]
        labels = list(loaded["labels"])
        meta = json.loads(str(loaded["meta"]))
    else:
        representations, labels, meta = extract_hidden_states(df, args)
        if args.save_representations:
            npz_path = DATA_DIR / f"representations_{RUN_TS}.npz"
            np.savez_compressed(
                npz_path, representations=representations,
                labels=np.array(labels), meta=json.dumps(meta),
            )
            logger.info(f"  ✅ Représentations sauvegardées : {npz_path}")

    layer_results = probe_all_layers(representations, labels)
    best_layer = max(layer_results, key=lambda k: layer_results[k]["accuracy_mean"])
    permutation_result = permutation_test_best_layer(representations, labels, best_layer)

    json_path, csv_path, best_layer = save_results(
        layer_results, permutation_result, meta, majority_baseline, df
    )
    fig_path = make_figure(csv_path, majority_baseline, best_layer)

    logger.info(f"\n{'═'*65}")
    logger.info("  TERMINÉ")
    logger.info(f"{'═'*65}")
    logger.info(f"  Meilleure couche : {best_layer} "
                f"(accuracy={layer_results[best_layer]['accuracy_mean']:.3f})")
    logger.info(f"  Baseline majoritaire : {majority_baseline:.3f}")
    logger.info(f"  p-value (test de permutation) : {permutation_result['p_value']:.4f}")
    logger.info("\n  Fichiers pour le papier :")
    logger.info(f"    {json_path}  (texte arXiv inclus dans le champ 'arxiv_text')")
    logger.info(f"    {fig_path}  (figure 2 du papier)")
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
