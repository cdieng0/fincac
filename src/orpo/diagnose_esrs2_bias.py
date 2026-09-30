"""
diagnose_esrs2_bias.py — Vérifie si le juge silver sur-utilise ESRS2 comme
catégorie "fourre-tout" dans les paires d'entraînement.
════════════════════════════════════════════════════════════════════════════
Diagnostic gratuit, local, sans GPU ni appel API — répond à la question :
le collapse ESRS2 observé sur le Gold-140 (précision=5,3%, rappel=5,9%)
vient-il d'un biais présent DÈS les paires d'entraînement, ou a-t-il été
introduit/amplifié par l'entraînement ORPO lui-même ?

Usage :
    python src/orpo/diagnose_esrs2_bias.py --pairs data/orpo/orpo_pairs_rebalanced_XXX.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--pairs", required=True)
args = p.parse_args()

path = Path(args.pairs)
records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Paires chargées : {len(records)}")

chosen_cats = Counter()
rejected_cats = Counter()
for r in records:
    try:
        chosen = json.loads(r["chosen"])
        rejected = json.loads(r["rejected"])
        chosen_cats[chosen.get("csrd_category", "?")] += 1
        rejected_cats[rejected.get("csrd_category", "?")] += 1
    except Exception:
        continue

print("\n=== Distribution des labels 'chosen' (= verdict du juge silver) ===")
total_chosen = sum(chosen_cats.values())
for cat, n in chosen_cats.most_common():
    print(f"  {cat:<10} {n:>4}  ({n/total_chosen*100:.1f}%)")

print("\n=== Distribution des labels 'rejected' (= erreurs du modèle de base) ===")
total_rejected = sum(rejected_cats.values())
for cat, n in rejected_cats.most_common():
    print(f"  {cat:<10} {n:>4}  ({n/total_rejected*100:.1f}%)")

esrs2_chosen_rate = chosen_cats.get("ESRS2", 0) / max(1, total_chosen) * 100
print(f"\n>>> ESRS2 représente {esrs2_chosen_rate:.1f}% de tous les labels 'chosen'.")
if esrs2_chosen_rate > 20:
    print(">>> ⚠ Taux élevé — cohérent avec l'hypothèse d'un juge silver biaisé vers ESRS2.")
else:
    print(">>> Taux modéré — le biais ESRS2 observé au test provient probablement de "
          "l'entraînement ORPO lui-même (généralisation excessive), pas des labels source.")

# Répartition par type de paire pour affiner
print("\n=== ESRS2 chosen, par pair_type ===")
by_type = Counter()
by_type_total = Counter()
for r in records:
    try:
        chosen = json.loads(r["chosen"])
        by_type_total[r["pair_type"]] += 1
        if chosen.get("csrd_category") == "ESRS2":
            by_type[r["pair_type"]] += 1
    except Exception:
        continue
for t, total in by_type_total.items():
    n = by_type.get(t, 0)
    print(f"  {t:<25} {n}/{total}  ({n/max(1,total)*100:.1f}%)")
