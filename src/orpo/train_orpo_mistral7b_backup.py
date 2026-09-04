"""
train_orpo_mistral7b.py — Fine-Tuning QLoRA + ORPO de Mistral-7B sur les Paires de Préférence
═══════════════════════════════════════════════════════════════════════════════════════
Phase 6b du pipeline FraFin-Reasoning — Remédiation du biais de récence temporelle.

RÔLE DANS LE PIPELINE :
    orpo_pairs_{TS}.jsonl              (paires chosen/rejected, Phase 6a)
              │
              ▼
    train_orpo_mistral7b.py   ← CE SCRIPT
              │
              ▼
    mistral7b-orpo-csrd/               (adaptateur LoRA entraîné)
              │
              ▼
    (script d'évaluation, non couvert ici — comparaison Base vs ORPO sur Gold-140)

MÉTHODE — ORPO (Odds Ratio Preference Optimization, Hong et al., 2024) :
    Contrairement à un pipeline SFT-puis-DPO en deux étapes, ORPO combine
    l'objectif de vraisemblance de la réponse préférée (terme NLL classique
    du SFT) et un terme de pénalité sur le ratio de cotes (odds ratio) entre
    la réponse préférée et la réponse rejetée, en une seule passe d'entraînement
    et SANS modèle de référence figé. Ceci réduit de moitié l'empreinte mémoire
    GPU par rapport à DPO classique (qui charge une copie gelée du modèle de
    référence en plus du modèle entraîné) — décisif sur un budget de calcul
    modeste (GPU unique, 16-24 Go de VRAM).

    La fonction de perte ORPO s'écrit :
        L_ORPO = L_SFT(chosen) + λ · L_OR
    où L_OR pénalise le ratio de cotes log[odds(chosen)/odds(rejected)],
    λ (paramètre `beta` dans l'implémentation TRL) contrôlant l'intensité
    de la pénalité de préférence relativement au terme de vraisemblance pure.

CONTRAT DE DONNÉES ATTENDU (sortie de build_orpo_pairs.py) :
    Chaque ligne JSONL contient : pair_id, paragraph_id, epoch, pair_type,
    prompt, chosen (JSON string), rejected (JSON string), chosen_category,
    rejected_category, chosen_model, rejected_model.
    Le `prompt` du JSONL ne contient QUE le tour utilisateur ("Extrait : «...»").
    Ce script reconstruit le PROMPT COMPLET (système + utilisateur) à
    l'identique du prompt de politique utilisé dans benchmark_temporal_drift.py
    et build_orpo_pairs.py — condition nécessaire pour que le modèle entraîné
    reste directement comparable au modèle de base évalué en Section 4.

CHOIX MÉTHODOLOGIQUES DOCUMENTÉS :

  (1) Format du prompt — PAS de template de chat automatique (`apply_chat_template`).
      Le tokenizer officiel Mistral-7B-Instruct-v0.3 ne supporte pas de manière
      fiable un rôle "system" séparé selon les versions (certaines lèvent une
      erreur Jinja "Conversation roles must alternate..."). Pour éviter toute
      dépendance à un comportement de template non garanti entre versions,
      nous formatons manuellement le prompt selon le gabarit d'instruction
      Mistral officiel : "<s>[INST] {système}\\n\\n{utilisateur} [/INST]".
      Ce gabarit est identique à celui utilisé implicitement par l'API Mistral
      lors de l'évaluation zero-shot (Section 4), garantissant la cohérence.

  (2) QLoRA — quantification 4-bit NF4 (Dettmers et al., 2023) avec double
      quantification et calcul en bfloat16. Rang r=16, alpha=32 (ratio 2:1,
      valeur standard de la littérature QLoRA), dropout 0.05, appliqué sur
      l'ensemble des projections linéaires (attention + feed-forward) plutôt
      que sur les seules projections d'attention, pour maximiser la capacité
      d'adaptation sur un budget de paramètres entraînables toujours minime
      (<1% des poids du modèle de base).

  (3) Découpage train/validation stratifié par (époque × pair_type) — évite
      qu'une époque ou un type de paire soit totalement absent de la validation,
      ce qui rendrait le monitoring de l'overfitting non fiable sur un si petit
      volume de données (typiquement 300-1500 paires).

  (4) Le Gold-140 n'est JAMAIS chargé par ce script. Une vérification de
      sécurité optionnelle (--gold-safety-check) recharge le fichier Gold et
      confirme qu'aucun paragraph_id des paires d'entraînement n'y figure —
      défense en profondeur, en plus de l'exclusion déjà appliquée en amont
      dans build_orpo_pairs.py.

CONTRAINTE D'EXÉCUTION :
    Ce script nécessite un GPU (T4 16 Go minimum, A100/L4 recommandé pour la
    vitesse) et les librairies torch/transformers/peft/trl/bitsandbytes.
    Il N'EST PAS exécutable dans un environnement CPU-only. La couche de
    chargement et de validation des données (fonctions load_pairs,
    stratified_split, format_prompt, validate_pairs) est en revanche
    testable indépendamment via --check-data-only, sans aucune dépendance
    GPU — utile pour valider le fichier de paires avant de lancer un run
    coûteux sur Colab/RunPod.

Dépendances (environnement GPU) :
    pip install "torch>=2.1" "transformers>=4.44" "trl>=0.11,<0.13" \\
                "peft>=0.12" "bitsandbytes>=0.43" accelerate datasets

Usage :
    # Validation locale des données (aucun GPU requis) :
    python train_orpo_mistral7b.py --check-data-only --pairs data/orpo/orpo_pairs_XXX.jsonl

    # Entraînement (environnement GPU — Colab / RunPod / poste avec GPU) :
    python train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_XXX.jsonl
    python train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_XXX.jsonl --epochs 3 --beta 0.1
    python train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_XXX.jsonl --resume-from-checkpoint outputs/checkpoint-100
    python train_orpo_mistral7b.py --pairs data/orpo/orpo_pairs_XXX.jsonl --merge-and-save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BASE_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR      = Path("outputs/mistral7b-orpo-csrd")
RUN_TS          = datetime.now().strftime("%Y%m%d_%H%M%S")

RANDOM_SEED     = 42
VAL_FRACTION    = 0.12       # ~12% des paires en validation, stratifié
MIN_VAL_PER_STRATUM = 1

EPOCH_ORDER = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]

# ── Hyperparamètres QLoRA (Dettmers et al., 2023) ─────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32          # ratio 2:1, valeur standard de la littérature
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",     # attention
    "gate_proj", "up_proj", "down_proj",        # feed-forward
]

# ── Hyperparamètres ORPO (Hong et al., 2024) ──────────────────────────────────
ORPO_BETA           = 0.1     # λ — poids du terme de ratio de cotes
MAX_PROMPT_LENGTH    = 768
MAX_COMPLETION_LENGTH = 256
MAX_LENGTH           = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH

# ── Hyperparamètres d'entraînement ────────────────────────────────────────────
NUM_EPOCHS               = 3
LEARNING_RATE             = 5e-5
PER_DEVICE_TRAIN_BATCH    = 2
GRADIENT_ACCUMULATION     = 8     # batch effectif = 2 * 8 = 16
LR_SCHEDULER              = "cosine"
WARMUP_RATIO               = 0.05
LOGGING_STEPS              = 10
EVAL_STEPS                 = 50
SAVE_STEPS                 = 50
WEIGHT_DECAY                = 0.01
MAX_GRAD_NORM                = 0.3   # standard QLoRA — évite l'instabilité en 4-bit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# PROMPT SYSTÈME — identique à build_orpo_pairs.py (TASK_BLOCK_POLICY)
# Dupliqué ici intentionnellement (cohérence garantie par duplication littérale
# et documentée, plutôt que par un import qui créerait une dépendance fragile
# entre un script d'entraînement GPU et un script de collecte de données API).
# ═════════════════════════════════════════════════════════════════════════════

TAXONOMY_BLOCK = """
CATÉGORIES DISPONIBLES (12) :

• none — Aucune pertinence CSRD
  Le paragraphe ne traite d'aucun enjeu de durabilité environnemental,
  social ou de gouvernance au sens CSRD. Inclut : opérations financières
  pures, formalités administratives, déclarations sans lien avec la durabilité.
  Sous-catégories : aucune.

• ESRS2 — Informations générales (gouvernance, stratégie, IRO)
  Sous-catégories : ESRS2-GOV, ESRS2-SBM, ESRS2-IRO, ESRS2-MDR.

• E1 — Changement climatique (GES Scope 1/2/3, transition, énergie)
  Sous-catégories : E1-1 à E1-9.

• E2 — Pollution (air, eau, sol, substances préoccupantes)
  Sous-catégories : E2-1 à E2-6.

• E3 — Eau et ressources marines
  Sous-catégories : E3-1 à E3-5.

• E4 — Biodiversité et écosystèmes
  Sous-catégories : E4-1 à E4-5.

• E5 — Ressources et économie circulaire
  Sous-catégories : E5-1 à E5-6.

• S1 — Effectifs propres (santé-sécurité, diversité, rémunération)
  Sous-catégories : S1-1 à S1-17.

• S2 — Travailleurs de la chaîne de valeur
  Sous-catégories : S2-1 à S2-5.

• S3 — Communautés affectées
  Sous-catégories : S3-1 à S3-5.

• S4 — Consommateurs et utilisateurs finaux
  Sous-catégories : S4-1 à S4-5.

• G1 — Conduite des affaires (anti-corruption, lobbying)
  Sous-catégories : G1-1 à G1-6.

RÈGLE CRITIQUE : si le texte ne mentionne aucun enjeu ESG explicite ou implicite,
répondre csrd_category = "none". Ne pas forcer une catégorie CSRD sur un texte
purement financier ou administratif.
"""

TASK_BLOCK_POLICY = """
Tu es un auditeur ESG senior, expert en réglementation européenne CSRD
et en normes ESRS, spécialisé dans l'analyse de documents réglementaires
AMF français.

Ta mission : classifier l'extrait ci-dessous selon la taxonomie CSRD/ESRS.

CONSIGNES STRICTES :
1. Détermine si le texte contient un enjeu de durabilité.
2. Si aucun enjeu → csrd_category = "none".
3. Si un enjeu est présent → identifie la catégorie ESRS principale.
4. Le raisonnement doit suivre 3 étapes : (1) nature de l'information,
   (2) analyse de matérialité ESG, (3) surprise pour les investisseurs.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{
  "csrd_category": "<none|ESRS2|E1|E2|E3|E4|E5|S1|S2|S3|S4|G1>",
  "esrs_subcategory": "<code sous-catégorie ou chaîne vide si none>",
  "chain_of_thought": "<3 phrases structurées>"
}
"""

SYSTEM_PROMPT = TAXONOMY_BLOCK.strip() + "\n\n" + TASK_BLOCK_POLICY.strip()

VALID_CATEGORIES = [
    "none", "ESRS2", "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4", "G1",
]

# ═════════════════════════════════════════════════════════════════════════════
# COUCHE DE DONNÉES — testable sans GPU (aucun import torch/transformers ici)
# ═════════════════════════════════════════════════════════════════════════════

def format_prompt(user_extract_line: str) -> str:
    """
    Formate le prompt complet au gabarit d'instruction Mistral officiel.

    Choix délibéré de NE PAS utiliser tokenizer.apply_chat_template() :
    le template de chat de Mistral-7B-Instruct-v0.3 ne garantit pas un
    support stable du rôle "system" selon les versions de `transformers`
    et de `tokenizer_config.json` du modèle (cf. docstring du module).
    Le gabarit explicite ci-dessous reproduit fidèlement le format
    d'instruction Mistral (`<s>[INST] ... [/INST]`) en injectant le
    system prompt en tête du tour utilisateur — un contournement standard
    et documenté dans la communauté pour ce problème précis.

    `user_extract_line` est la valeur du champ "prompt" du JSONL, de la
    forme littérale : Extrait : «...texte du paragraphe...»
    """
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_extract_line} [/INST]"


def validate_pair_json(raw: str) -> tuple[bool, str]:
    """Valide qu'une chaîne chosen/rejected est un JSON conforme au schéma attendu."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, "JSON invalide"
    cat = data.get("csrd_category")
    if cat not in VALID_CATEGORIES:
        return False, f"csrd_category invalide : {cat!r}"
    if "chain_of_thought" not in data:
        return False, "chain_of_thought manquant"
    return True, ""


def load_pairs(path: Path) -> pd.DataFrame:
    """Charge le JSONL produit par build_orpo_pairs.py et valide chaque ligne."""
    if not path.exists():
        logger.error(f"Fichier de paires introuvable : {path.resolve()}")
        sys.exit(1)

    records = []
    n_bad = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                logger.warning(f"  Ligne {line_no} : JSON de ligne invalide, ignorée")
                continue

            ok_c, err_c = validate_pair_json(rec.get("chosen", ""))
            ok_r, err_r = validate_pair_json(rec.get("rejected", ""))
            if not ok_c or not ok_r:
                n_bad += 1
                logger.warning(
                    f"  Ligne {line_no} (pair_id={rec.get('pair_id')}) : "
                    f"chosen={'OK' if ok_c else err_c}, rejected={'OK' if ok_r else err_r} — ignorée"
                )
                continue

            required = {"prompt", "chosen", "rejected", "epoch", "pair_type", "paragraph_id"}
            if not required.issubset(rec.keys()):
                n_bad += 1
                logger.warning(f"  Ligne {line_no} : champs manquants — ignorée")
                continue

            records.append(rec)

    df = pd.DataFrame(records)
    logger.info(f"  Paires chargées : {len(df)}  (rejetées : {n_bad})")
    if len(df) == 0:
        logger.error("Aucune paire valide chargée — abandon.")
        sys.exit(1)
    return df


def gold_safety_check(pairs_df: pd.DataFrame, gold_path: str | None) -> None:
    """
    Défense en profondeur : re-vérifie qu'aucun paragraph_id des paires
    d'entraînement ne figure dans le Gold-140, indépendamment de
    l'exclusion déjà appliquée dans build_orpo_pairs.py.
    """
    if not gold_path:
        logger.info("  (Vérification de sécurité Gold ignorée — aucun --gold fourni)")
        return

    gpath = Path(gold_path)
    if not gpath.exists():
        logger.warning(f"  ⚠ Fichier Gold introuvable pour vérification : {gpath}")
        return

    gold_df = pd.read_excel(gpath, sheet_name="Annotation")
    if "paragraph_id" not in gold_df.columns:
        logger.warning("  ⚠ Colonne paragraph_id absente du fichier Gold — vérification impossible")
        return

    gold_ids = set(gold_df["paragraph_id"].astype(str))
    train_ids = set(pairs_df["paragraph_id"].astype(str))
    overlap = gold_ids & train_ids

    if overlap:
        logger.error(
            f"  ❌ FUITE DE DONNÉES DÉTECTÉE : {len(overlap)} paragraph_id du "
            f"Gold-140 (test set) sont présents dans les paires d'entraînement !\n"
            f"     IDs concernés : {sorted(overlap)[:10]}\n"
            f"     Abandon de l'entraînement — corrigez build_orpo_pairs.py."
        )
        sys.exit(1)

    logger.info(f"  ✅ Vérification de sécurité Gold : aucun chevauchement "
                f"({len(train_ids)} paires vs {len(gold_ids)} Gold)")


def stratified_split(
    df: pd.DataFrame, val_fraction: float, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Découpage train/validation stratifié par (epoch, pair_type), avec un
    plancher minimum d'exemples de validation par strate quand la strate
    le permet — garantit que chaque combinaison époque×type est représentée
    dans le monitoring de validation, évitant un signal d'eval trompeur sur
    un volume de données restreint.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["_stratum"] = df["epoch"].astype(str) + "__" + df["pair_type"].astype(str)

    val_parts, train_parts = [], []
    for stratum, group in df.groupby("_stratum"):
        n = len(group)
        n_val = max(MIN_VAL_PER_STRATUM, int(round(n * val_fraction))) if n > 1 else 0
        n_val = min(n_val, n - 1) if n > 1 else 0   # garde au moins 1 exemple en train
        shuffled = group.sample(frac=1, random_state=seed)
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])

    val_df   = pd.concat(val_parts, ignore_index=True) if val_parts else df.head(0)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df

    val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"\n  Split train/validation (stratifié epoch × pair_type, seed={seed}) :")
    logger.info(f"    Train      : {len(train_df)}")
    logger.info(f"    Validation : {len(val_df)}")
    return train_df, val_df


def build_hf_dataset_records(df: pd.DataFrame) -> list[dict]:
    """
    Construit les enregistrements au format attendu par ORPOTrainer (TRL) :
    colonnes "prompt", "chosen", "rejected" en TEXTE BRUT (non tokenisé).
    Le prompt est le gabarit d'instruction complet ; chosen/rejected sont
    les complétions JSON telles que produites par le juge (chosen) et par
    la politique de base en zero-shot (rejected) — cf. build_orpo_pairs.py.
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            "prompt":   format_prompt(row["prompt"]),
            "chosen":   row["chosen"],
            "rejected": row["rejected"],
            # Colonnes conservées pour la traçabilité (ignorées par le trainer
            # si remove_unused_columns=True, sinon utiles pour l'inspection) :
            "paragraph_id": row["paragraph_id"],
            "epoch":        row["epoch"],
            "pair_type":    row["pair_type"],
        })
    return records


def print_data_report(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    logger.info(f"\n{'═'*65}")
    logger.info(f"  RAPPORT DE DONNÉES — Paires ORPO")
    logger.info(f"{'═'*65}")

    for name, d in [("TRAIN", train_df), ("VALIDATION", val_df)]:
        logger.info(f"\n  [{name}] n = {len(d)}")
        if len(d) == 0:
            continue
        logger.info(f"    Par époque :")
        for ep, n in d["epoch"].value_counts().reindex(EPOCH_ORDER, fill_value=0).items():
            logger.info(f"      {ep:<12} {n}")
        logger.info(f"    Par type de paire :")
        for t, n in d["pair_type"].value_counts().items():
            logger.info(f"      {t:<25} {n}")

    # Statistiques de longueur (approximation par nombre de mots, sans tokenizer)
    prompt_lens = train_df["prompt"].astype(str).str.split().str.len()
    chosen_lens = train_df["chosen"].astype(str).str.split().str.len()
    logger.info(f"\n  Longueurs approximatives (en mots, hors tokenisation) :")
    logger.info(f"    Prompt utilisateur : moyenne={prompt_lens.mean():.0f}, "
                f"max={prompt_lens.max()}")
    logger.info(f"    Chosen JSON        : moyenne={chosen_lens.mean():.0f}, "
                f"max={chosen_lens.max()}")
    logger.info(f"{'═'*65}")

# ═════════════════════════════════════════════════════════════════════════════
# COUCHE D'ENTRAÎNEMENT — imports lourds différés (nécessite un GPU)
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    train_records: list[dict],
    val_records: list[dict],
    args: argparse.Namespace,
) -> None:
    """
    Construit le modèle quantifié 4-bit, l'adaptateur LoRA, et lance
    l'entraînement ORPO via TRL. Toutes les librairies lourdes (torch,
    transformers, peft, trl, datasets) sont importées ici, PAS au niveau
    module — permet à --check-data-only de fonctionner sans ces dépendances.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        )
        from trl import ORPOConfig, ORPOTrainer
    except ImportError as e:
        logger.error(
            f"Import échoué : {e}\n"
            f"Ce script nécessite un environnement GPU avec les dépendances suivantes :\n"
            f'  pip install "torch>=2.1" "transformers>=4.44" "trl>=0.11,<0.13" '
            f'"peft>=0.12" "bitsandbytes>=0.43" accelerate datasets\n'
            f"Utilisez --check-data-only pour valider les données sans ces dépendances."
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error(
            "Aucun GPU CUDA détecté. L'entraînement QLoRA 4-bit nécessite un GPU "
            "(T4 16 Go minimum). Lancez ce script sur Google Colab (GPU runtime), "
            "RunPod, ou tout poste équipé d'un GPU NVIDIA."
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"\n  GPU détecté : {gpu_name}  ({gpu_mem_gb:.1f} Go)")
    if gpu_mem_gb < 14:
        logger.warning(
            f"  ⚠ VRAM détectée ({gpu_mem_gb:.1f} Go) < 14 Go recommandés pour "
            f"Mistral-7B en 4-bit + LoRA avec la config par défaut. Réduisez "
            f"--per-device-batch et/ou --max-length si vous rencontrez un OOM."
        )

    # ── Quantification 4-bit NF4 (QLoRA, Dettmers et al., 2023) ──────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"\n  Chargement du tokenizer et du modèle : {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False   # requis pour le gradient checkpointing

    # ── Configuration LoRA ────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    logger.info(
        f"  LoRA config : r={args.lora_r}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, modules={LORA_TARGET_MODULES}"
    )

    # ── Datasets HuggingFace ──────────────────────────────────────────────────
    train_ds = Dataset.from_list(train_records)
    val_ds   = Dataset.from_list(val_records) if val_records else None

    # ── Configuration ORPO ────────────────────────────────────────────────────
    output_dir = str(OUTPUT_DIR)

    orpo_kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=RANDOM_SEED,
        run_name=f"mistral7b-orpo-csrd-{RUN_TS}",
    )

    # ── Compatibilité de version : eval_strategy vs evaluation_strategy ───────
    # `transformers` a renommé cet argument entre versions ; on tente les deux
    # noms pour rester compatible avec l'environnement Colab au moment du run.
    eval_kwarg_name = "eval_strategy"
    if val_ds is not None:
        orpo_kwargs[eval_kwarg_name] = "steps"
        orpo_kwargs["eval_steps"] = EVAL_STEPS
    else:
        orpo_kwargs[eval_kwarg_name] = "no"
        logger.warning("  ⚠ Aucun jeu de validation — monitoring d'overfitting désactivé.")

    try:
        training_args = ORPOConfig(**orpo_kwargs)
    except TypeError as e:
        if "eval_strategy" in str(e):
            logger.info("  (Compat) 'eval_strategy' non reconnu — repli sur 'evaluation_strategy'")
            orpo_kwargs["evaluation_strategy"] = orpo_kwargs.pop("eval_strategy")
            training_args = ORPOConfig(**orpo_kwargs)
        else:
            raise

    logger.info(
        f"\n  Config ORPO : beta={args.beta}, epochs={args.epochs}, "
        f"lr={args.learning_rate}, batch_effectif="
        f"{args.per_device_batch * args.grad_accum}, "
        f"max_length={args.max_length} (prompt≤{args.max_prompt_length})"
    )

    # ── Construction du trainer — compat processing_class/tokenizer ──────────
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )
    try:
        trainer = ORPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        logger.info("  (Compat) 'processing_class' non reconnu — repli sur 'tokenizer'")
        trainer = ORPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    # ── Entraînement ───────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*65}")
    logger.info(f"  DÉBUT DE L'ENTRAÎNEMENT ORPO")
    logger.info(f"{'═'*65}")

    resume_ckpt = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Sauvegarde de l'adaptateur LoRA ────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"\n  ✅ Adaptateur LoRA sauvegardé : {final_dir}")

    # ── Log des métriques d'entraînement ───────────────────────────────────────
    metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
    log_history = trainer.state.log_history if hasattr(trainer, "state") else []

    run_meta = {
        "run_timestamp":      RUN_TS,
        "base_model":         args.base_model,
        "method":             "ORPO (Odds Ratio Preference Optimization)",
        "reference":          "Hong et al., 2024 — arXiv:2403.07691",
        "quantization":       "4-bit NF4, double quant, compute dtype bfloat16",
        "lora_config": {
            "r": args.lora_r, "alpha": args.lora_alpha,
            "dropout": args.lora_dropout, "target_modules": LORA_TARGET_MODULES,
        },
        "orpo_beta":          args.beta,
        "num_epochs":         args.epochs,
        "learning_rate":      args.learning_rate,
        "effective_batch_size": args.per_device_batch * args.grad_accum,
        "max_length":         args.max_length,
        "max_prompt_length":  args.max_prompt_length,
        "random_seed":        RANDOM_SEED,
        "gpu":                gpu_name,
        "n_train_pairs":      len(train_records),
        "n_val_pairs":        len(val_records),
        "final_train_metrics": metrics,
        "prompt_format": (
            "Manual Mistral instruction template "
            "'<s>[INST] {system}\\n\\n{user} [/INST]' — NOT tokenizer.apply_chat_template(), "
            "cf. module docstring for rationale."
        ),
        "gold_140_excluded": "Verified via build_orpo_pairs.py exclusion + optional --gold-safety-check",
    }
    meta_path = OUTPUT_DIR / f"run_meta_{RUN_TS}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  ✅ Métadonnées du run : {meta_path}")

    log_path = OUTPUT_DIR / f"training_log_{RUN_TS}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  ✅ Historique d'entraînement : {log_path}")

    # ── Fusion optionnelle LoRA → poids complets ──────────────────────────────
    if args.merge_and_save:
        logger.info(f"\n  Fusion de l'adaptateur LoRA dans les poids de base...")
        merged_model = trainer.model.merge_and_unload()
        merged_dir = OUTPUT_DIR / "merged_model"
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info(f"  ✅ Modèle fusionné sauvegardé : {merged_dir}  "
                    f"(⚠ occupe ~14-15 Go sur disque en bfloat16)")

    logger.info(f"\n{'═'*65}")
    logger.info(f"  ENTRAÎNEMENT TERMINÉ")
    logger.info(f"{'═'*65}")
    logger.info(f"  ⚙  Prochaine étape : évaluation comparative sur Gold-140")
    logger.info(f"     (Base Mistral-7B zero-shot vs Mistral-7B-ORPO-CSRD)")
    logger.info(f"     Réutilisez le prompt SYSTEM_PROMPT de ce script pour")
    logger.info(f"     charger l'adaptateur et générer les prédictions à comparer")
    logger.info(f"     aux résultats de la Section 4 (FNR par époque notamment).")
    logger.info(f"{'═'*65}")

# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FraFin-Reasoning — Fine-tuning QLoRA+ORPO de Mistral-7B (Phase 6b)"
    )
    p.add_argument("--pairs", required=True,
                   help="Fichier JSONL produit par build_orpo_pairs.py")
    p.add_argument("--gold", default=None,
                   help="Chemin du Gold-140 pour vérification de sécurité anti-fuite (optionnel)")
    p.add_argument("--gold-safety-check", action="store_true",
                   help="Active la vérification anti-fuite (nécessite --gold)")
    p.add_argument("--base-model", default=BASE_MODEL_ID)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)

    # LoRA
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)

    # ORPO / entraînement
    p.add_argument("--beta", type=float, default=ORPO_BETA,
                   help="Poids λ du terme de ratio de cotes ORPO (défaut 0.1)")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--per-device-batch", type=int, default=PER_DEVICE_TRAIN_BATCH)
    p.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)

    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--merge-and-save", action="store_true",
                   help="Fusionne l'adaptateur LoRA dans les poids de base après entraînement")
    p.add_argument("--report-to", default="none",
                   help="'none', 'wandb', 'tensorboard'...")

    p.add_argument("--check-data-only", action="store_true",
                   help="Valide et rapporte les données sans lancer l'entraînement "
                        "(aucune dépendance GPU requise)")
    return p.parse_args()


def main():
    global RANDOM_SEED, VAL_FRACTION, OUTPUT_DIR

    args = parse_args()
    RANDOM_SEED  = args.seed
    VAL_FRACTION = args.val_fraction

    logger.info("═" * 65)
    logger.info("  FraFin-Reasoning — Entraînement QLoRA + ORPO (Phase 6b)")
    logger.info(f"  Modèle de base : {args.base_model}")
    logger.info(f"  Mode           : {'VALIDATION DES DONNÉES SEULE' if args.check_data_only else 'ENTRAÎNEMENT COMPLET'}")
    logger.info("═" * 65)

    # ── Chargement et validation des paires (aucune dépendance GPU) ──────────
    pairs_df = load_pairs(Path(args.pairs))

    if args.gold_safety_check or args.gold:
        gold_safety_check(pairs_df, args.gold)

    train_df, val_df = stratified_split(pairs_df, VAL_FRACTION, RANDOM_SEED)
    print_data_report(train_df, val_df)

    train_records = build_hf_dataset_records(train_df)
    val_records   = build_hf_dataset_records(val_df)

    if args.check_data_only:
        logger.info(f"\n  [--check-data-only] Aperçu d'un prompt formaté (premier exemple train) :")
        logger.info(f"  {'─'*61}")
        preview = train_records[0]["prompt"][:600]
        logger.info(f"  {preview}...")
        logger.info(f"  {'─'*61}")
        logger.info(f"\n  Aperçu chosen  : {train_records[0]['chosen'][:200]}")
        logger.info(f"  Aperçu rejected: {train_records[0]['rejected'][:200]}")
        logger.info(f"\n  ✅ Données validées. Relancez sans --check-data-only sur un "
                    f"environnement GPU pour lancer l'entraînement.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_training(train_records, val_records, args)


if __name__ == "__main__":
    main()
