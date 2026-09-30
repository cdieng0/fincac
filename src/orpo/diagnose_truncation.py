"""
diagnose_truncation.py — Vérifie l'hypothèse de troncature JSON dans le checkpoint ORPO
Usage : python src/orpo/diagnose_truncation.py --checkpoint data/orpo/checkpoint_orpo_candidates.json
"""
import argparse, json

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="data/orpo/checkpoint_orpo_candidates.json")
args = p.parse_args()

ckpt = json.load(open(args.checkpoint, encoding="utf-8"))

n_pair, n_agreement, n_error = 0, 0, 0
n_pair_empty_cot = 0
n_agreement_both_none = 0

for pid, rec in ckpt.items():
    status = rec.get("status")
    if status == "pair":
        n_pair += 1
        chosen  = json.loads(rec["chosen"])   if isinstance(rec["chosen"], str)   else rec["chosen"]
        reject  = json.loads(rec["rejected"]) if isinstance(rec["rejected"], str) else rec["rejected"]
        cot_c = chosen.get("chain_of_thought", "")
        cot_r = reject.get("chain_of_thought", "")
        if len(cot_c) < 10 or len(cot_r) < 10:
            n_pair_empty_cot += 1
    elif status == "agreement":
        n_agreement += 1
        if rec.get("silver_category") == "none" and rec.get("base_category") == "none":
            n_agreement_both_none += 1
    elif status == "api_error":
        n_error += 1

print(f"Paires totales           : {n_pair}")
print(f"  dont chain_of_thought vide/très court (signature de troncature) : {n_pair_empty_cot} "
      f"({n_pair_empty_cot/max(1,n_pair)*100:.1f}%)")
print(f"Accords 'both none'      : {n_agreement_both_none} / {n_agreement}")
print(f"Erreurs API              : {n_error}")
