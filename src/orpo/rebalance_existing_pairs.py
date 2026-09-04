"""
rebalance_existing_pairs.py — Corrige un fichier de paires ORPO déjà collecté,
SANS AUCUN APPEL API.
═══════════════════════════════════════════════════════════════════════════════════════
Contexte : le premier entraînement ORPO a collapsé vers "toujours prédire CSRD"
(FNR=0 partout, accuracy=0.164) parce que le fichier de paires utilisé pour
l'entraînement contenait ~97% de paires Type A (silver=CSRD, base=none) contre
seulement ~3% de paires de garde-fou (Type B + Type C). Cf. diagnostic complet
et correctif de build_orpo_pairs.py::balance_pairs() (paramètre
--max-ratio-to-guardrail).

Ce script réapplique EXACTEMENT le même correctif (plafond de A relatif au
volume combiné B+C) directement sur le fichier .jsonl DÉJÀ PRODUIT — que ce
fichier soit :
  (a) le orpo_pairs_*.jsonl final (déjà équilibré par l'ancienne logique
      buguée — contient encore largement assez de paires A pour re-plafonner
      à la baisse), ou
  (b) le checkpoint_orpo_candidates.json brut (contient tous les candidats
      traités, y compris les accords et erreurs — le script filtre
      automatiquement pour ne garder que les paires exploitables).

Aucun appel réseau. Aucune clé API requise. Coût : quelques secondes CPU.

Usage :
    python rebalance_existing_pairs.py --input orpo_pairs_20260823_194630.jsonl
    python rebalance_existing_pairs.py --input checkpoint_orpo_candidates.json
    python rebalance_existing_pairs.py --input orpo_pairs_XXX.jsonl --max-ratio-to-guardrail 4.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RANDOM_SEED = 42
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REQUIRED_PAIR_FIELDS = [
    "paragraph_id", "epoch", "pair_type", "prompt",
    "chosen", "rejected", "chosen_category", "rejected_category",
    "chosen_model", "rejected_model",
]

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT — accepte JSONL de paires OU checkpoint JSON brut
# ═════════════════════════════════════════════════════════════════════════════

def load_pairs_any_format(path: Path) -> list[dict]:
    if not path.exists():
        logger.error(f"Fichier introuvable : {path.resolve()}")
        sys.exit(1)

    raw_text = path.read_text(encoding="utf-8").strip()

    # ── Cas 1 : checkpoint JSON (un seul objet, clé = paragraph_id) ──────────
    if raw_text.startswith("{") and not raw_text.startswith('{"pair_id"'):
        try:
            ckpt = json.loads(raw_text)
            if isinstance(ckpt, dict):
                pairs = [
                    rec for rec in ckpt.values()
                    if isinstance(rec, dict) and rec.get("status") == "pair"
                ]
                if pairs:
                    logger.info(f"  Format détecté : checkpoint JSON brut")
                    logger.info(f"  {len(ckpt)} candidats au total dans le checkpoint, "
                                f"{len(pairs)} sont des paires exploitables")
                    return pairs
        except json.JSONDecodeError:
            pass  # tombe dans le cas JSONL ci-dessous

    # ── Cas 2 : JSONL, une paire par ligne ────────────────────────────────────
    pairs = []
    n_bad = 0
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        if "pair_type" not in rec:
            n_bad += 1
            continue
        pairs.append(rec)

    logger.info(f"  Format détecté : JSONL de paires")
    logger.info(f"  {len(pairs)} paires chargées (lignes ignorées : {n_bad})")
    return pairs


def validate_pairs(pairs: list[dict]) -> list[dict]:
    """Filtre les enregistrements incomplets (champs requis manquants)."""
    valid = []
    for p in pairs:
        if all(k in p for k in ("paragraph_id", "epoch", "pair_type", "chosen", "rejected")):
            valid.append(p)
    if len(valid) < len(pairs):
        logger.warning(f"  ⚠ {len(pairs) - len(valid)} paires incomplètes ignorées")
    return valid

# ═════════════════════════════════════════════════════════════════════════════
# RÉ-ÉQUILIBRAGE — logique identique à build_orpo_pairs.py (corrigée)
# ═════════════════════════════════════════════════════════════════════════════

def rebalance(
    pairs: list[dict],
    max_type_ratio: float,
    min_cap_floor: int,
    max_ratio_to_guardrail: float,
    seed: int,
    guardrail_oversample: int = 1,
) -> list[dict]:
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p["pair_type"]].append(p)

    counts = {t: len(v) for t, v in by_type.items()}
    logger.info(f"\n  Distribution disponible dans le fichier source : {counts}")

    non_empty = [c for c in counts.values() if c > 0]
    if not non_empty:
        logger.error("Aucune paire trouvée — vérifiez le fichier source.")
        sys.exit(1)

    # [AJOUT] Sur-échantillonnage des types de garde-fou (B, C) — pas de nouvel
    # appel API : simple duplication contrôlée. Motivation : avec seulement
    # 8 exemples B et 5 exemples C, même après plafonnement correct de A, un
    # batch effectif de 16 exemples a ~85% de chances de ne contenir AUCUN
    # exemple de garde-fou (13 sur 78 au total). Plusieurs batches consécutifs
    # peuvent donc ne transmettre que du signal "préfère CSRD", laissant le
    # raccourci dégénéré se reformer localement avant qu'un contre-exemple
    # n'apparaisse. Dupliquer B/C x{guardrail_oversample} densifie leur
    # présence par batch, technique standard de correction de déséquilibre de
    # classe (minority oversampling), sans fabriquer de contenu nouveau — les
    # doublons restent des erreurs RÉELLEMENT observées du modèle de base.
    if guardrail_oversample > 1:
        for t in list(by_type.keys()):
            if t != "A_recency_fn":
                original = by_type[t]
                duplicated = []
                for copy_idx in range(guardrail_oversample):
                    for p in original:
                        dup = dict(p)
                        # [FIX] paragraph_id distinct par copie -- sinon le split
                        # stratifié train/val de train_orpo_mistral7b.py pourrait
                        # placer la même paire des deux côtés (fuite locale).
                        dup["paragraph_id"] = f"{p['paragraph_id']}__dup{copy_idx}"
                        duplicated.append(dup)
                by_type[t] = duplicated
        counts = {t: len(v) for t, v in by_type.items()}
        logger.info(
            f"  [Sur-échantillonnage x{guardrail_oversample} sur B/C] "
            f"Distribution après duplication : {counts}"
        )

    min_count = min(c for c in counts.values() if c > 0)
    cap_basis = max(min_count, min_cap_floor)
    cap = max(1, int(cap_basis * max_type_ratio))

    n_guardrail = sum(n for t, n in counts.items() if t != "A_recency_fn")
    if n_guardrail == 0:
        logger.warning(
            "  ⚠ Aucune paire de garde-fou (B/C) dans le fichier source — "
            "le plafond A-vs-garde-fou ne peut pas être appliqué. Le collapse "
            "observé ne sera PAS corrigé par ce script. Une collecte ciblée "
            "de paires B/C est nécessaire (cf. recommandations)."
        )
        guardrail_cap = cap
    else:
        guardrail_cap = max(1, int(max_ratio_to_guardrail * n_guardrail))
        logger.info(
            f"  Plafond combiné A vs garde-fou (B+C={n_guardrail}, "
            f"ratio max {max_ratio_to_guardrail}x) : A <= {guardrail_cap}"
        )

    rng = random.Random(seed)
    balanced = []
    for t, plist in by_type.items():
        effective_cap = min(cap, guardrail_cap) if t == "A_recency_fn" else cap
        if len(plist) > effective_cap:
            balanced.extend(rng.sample(plist, effective_cap))
            logger.info(f"    {t}: {len(plist)} → {effective_cap} (cappé)")
        else:
            balanced.extend(plist)
            logger.info(f"    {t}: {len(plist)} (conservé intégralement — "
                        f"{'⚠ insuffisant, sous le cap visé' if t != 'A_recency_fn' else 'OK'})")

    final_counts = Counter(p["pair_type"] for p in balanced)
    logger.info(f"\n  Distribution finale : {dict(final_counts)}")
    n_a = final_counts.get("A_recency_fn", 0)
    n_g = sum(v for k, v in final_counts.items() if k != "A_recency_fn")
    if n_g > 0:
        logger.info(f"  Ratio A / garde-fou final : {n_a / n_g:.2f}x "
                    f"(cible <= {max_ratio_to_guardrail}x)")
    else:
        logger.warning("  ⚠ Toujours aucun garde-fou dans le résultat final.")

    rng.shuffle(balanced)
    return balanced

# ═════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE — même format que build_orpo_pairs.py::save_pairs_jsonl
# ═════════════════════════════════════════════════════════════════════════════

def save_pairs_jsonl(pairs: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            record = {
                "pair_id":           f"orpo_rebal_{i:05d}",
                "paragraph_id":      p["paragraph_id"],
                "epoch":             p["epoch"],
                "pair_type":         p["pair_type"],
                "prompt":            p.get("prompt", ""),
                "chosen":            p["chosen"],
                "rejected":          p["rejected"],
                "chosen_category":   p.get("chosen_category", ""),
                "rejected_category": p.get("rejected_category", ""),
                "chosen_model":      p.get("chosen_model", ""),
                "rejected_model":    p.get("rejected_model", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"\n  ✅ Fichier rééquilibré : {path}  ({len(pairs)} lignes)")


def save_stats(pairs: list[dict], source_path: Path, path: Path, args) -> None:
    by_type = Counter(p["pair_type"] for p in pairs)
    by_epoch = Counter(p["epoch"] for p in pairs)
    stats = {
        "run_timestamp":            RUN_TS,
        "source_file":              str(source_path),
        "method":                   "rebalance_existing_pairs.py — zéro appel API",
        "max_type_ratio":           args.max_type_ratio,
        "min_cap_floor":            args.min_cap_floor,
        "max_ratio_to_guardrail":   args.max_ratio_to_guardrail,
        "random_seed":              args.seed,
        "n_pairs_final":            len(pairs),
        "pair_type_totals":         dict(by_type),
        "epoch_totals":             dict(by_epoch),
        "incident_reference": (
            "Corrige un dataset ORPO déjà collecté qui présentait un "
            "déséquilibre Type A / garde-fou extrême (ex: 300/8/5, soit 97% "
            "de Type A), causant un collapse du modèle entraîné vers la "
            "prédiction systématique de CSRD (FNR≈0 mais accuracy très basse "
            "et FPR élevé). Voir INCIDENT_NOTE.md pour le diagnostic complet."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✅ Stats : {path}")

# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Réapplique le correctif de balance_pairs sur un fichier ORPO "
                    "déjà collecté, sans aucun appel API."
    )
    p.add_argument("--input", required=True,
                   help="orpo_pairs_*.jsonl OU checkpoint_orpo_candidates.json")
    p.add_argument("--output-dir", default="data/orpo")
    p.add_argument("--max-type-ratio", type=float, default=3.0)
    p.add_argument("--min-cap-floor", type=int, default=100)
    p.add_argument("--max-ratio-to-guardrail", type=float, default=5.0,
                   help="Plafonne A à ce ratio maximum de (B+C). Défaut 5.0 : "
                        "sur un cas B=8,C=5, donne A<=65.")
    p.add_argument("--guardrail-oversample", type=int, default=3,
                   help="Duplique B/C x N avant de calculer le plafond de A. "
                        "Défaut 3 : densifie le signal de garde-fou par batch "
                        "sans nouvel appel API. Mettre 1 pour désactiver.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("═" * 65)
    logger.info("  Réparation d'un fichier ORPO existant — 0 appel API")
    logger.info(f"  Source : {args.input}")
    logger.info("═" * 65)

    input_path = Path(args.input)
    raw_pairs = load_pairs_any_format(input_path)
    raw_pairs = validate_pairs(raw_pairs)

    if not raw_pairs:
        logger.error("Aucune paire valide extraite du fichier source.")
        sys.exit(1)

    balanced = rebalance(
        raw_pairs,
        args.max_type_ratio, args.min_cap_floor,
        args.max_ratio_to_guardrail, args.seed,
        args.guardrail_oversample,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"orpo_pairs_rebalanced_{RUN_TS}.jsonl"
    out_stats = out_dir / f"orpo_stats_rebalanced_{RUN_TS}.json"

    save_pairs_jsonl(balanced, out_jsonl)
    save_stats(balanced, input_path, out_stats, args)

    logger.info(f"\n{'═'*65}")
    logger.info(f"  TERMINÉ — Prochaine étape :")
    logger.info(f"    python train_orpo_mistral7b.py --check-data-only \\")
    logger.info(f"        --pairs {out_jsonl}")
    logger.info(f"  Puis, si le rapport de données est sain :")
    logger.info(f"    python train_orpo_mistral7b.py --pairs {out_jsonl} --epochs 1")
    logger.info(f"  (1 epoch de vérification d'abord — regardez rewards/accuracies")
    logger.info(f"   dans les 10 premiers steps avant de lancer un run complet)")
    logger.info(f"{'═'*65}")


if __name__ == "__main__":
    main()
