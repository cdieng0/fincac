"""
publish_model_to_hf.py — Publie l'adaptateur ORPO et ses artefacts de recherche
═══════════════════════════════════════════════════════════════════════════════════════
Rassemble l'adaptateur LoRA, la carte de modèle, le prompt système et l'ensemble des
artefacts qui rendent l'échec reproductible (paires d'entraînement, télémétrie, prédictions
d'évaluation), puis pousse le tout sur le Hub.

⚠️  VÉRIFICATION À BLANC par défaut. Rien n'est publié sans --push.

Prérequis :
    pip install huggingface_hub
    huggingface-cli login          # token de type WRITE

Usage :
    # 1. Vérifier ce qui sera publié
    python publish_model_to_hf.py --adapter outputs/mistral7b-orpo-csrd/final_adapter \\
                                   --repo CID99/Mistral-7B-ORPO-CSRD

    # 2. Publier en privé d'abord
    python publish_model_to_hf.py --adapter ... --repo ... --push --private
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Fichiers de l'adaptateur LoRA proprement dit — sans eux, rien n'est chargeable.
ADAPTER_REQUIRED = ["adapter_config.json"]
ADAPTER_WEIGHTS = ["adapter_model.safetensors", "adapter_model.bin"]
ADAPTER_OPTIONAL = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "chat_template.jinja",
]

# Le prompt système EXACT utilisé pour toutes les évaluations rapportées dans la carte.
# Sans lui, les chiffres publiés ne sont pas reproductibles : c'est l'artefact le plus
# souvent oublié dans les publications de modèles, et le plus coûteux à reconstituer.
PROMPT_TEMPLATE = """\
Tu es un auditeur ESG senior, expert en réglementation européenne CSRD
et en normes ESRS, spécialisé dans l'analyse de documents réglementaires
AMF français.

Ta mission : classifier l'extrait ci-dessous selon la taxonomie CSRD/ESRS.

CONSIGNES STRICTES :
1. Détermine si le texte contient un enjeu de durabilité.
2. Si aucun enjeu -> csrd_category = "none".
3. Si un enjeu est présent -> identifie la catégorie ESRS principale.
4. Le raisonnement doit suivre 3 étapes : (1) nature de l'information,
   (2) analyse de matérialité ESG, (3) surprise pour les investisseurs.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{
  "csrd_category": "<none|ESRS2|E1|E2|E3|E4|E5|S1|S2|S3|S4|G1>",
  "esrs_subcategory": "<code sous-catégorie ou chaîne vide si none>",
  "chain_of_thought": "<3 phrases structurées>"
}

Paramètres d'évaluation : température 0, graine 42, mode JSON natif activé.
"""


def collect(adapter_dir: Path, extras: dict[str, Path | None], out: Path) -> tuple[bool, list]:
    """Copie l'adaptateur et les artefacts dans un dossier de publication propre."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ok = True
    manifest = []

    # --- Adaptateur ---
    for f in ADAPTER_REQUIRED:
        src = adapter_dir / f
        if not src.exists():
            print(f"  ❌ MANQUANT (bloquant) : {f}")
            ok = False
        else:
            shutil.copy(src, out / f)
            manifest.append(f)

    weights = [f for f in ADAPTER_WEIGHTS if (adapter_dir / f).exists()]
    if not weights:
        print(f"  ❌ MANQUANT (bloquant) : aucun poids trouvé "
              f"({' ou '.join(ADAPTER_WEIGHTS)})")
        ok = False
    else:
        shutil.copy(adapter_dir / weights[0], out / weights[0])
        manifest.append(weights[0])

    for f in ADAPTER_OPTIONAL:
        src = adapter_dir / f
        if src.exists():
            shutil.copy(src, out / f)
            manifest.append(f)

    # --- Prompt système (toujours écrit, jamais optionnel) ---
    (out / "prompt_template.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    manifest.append("prompt_template.txt")

    # --- Artefacts de recherche ---
    for target, src in extras.items():
        if src is None:
            continue
        if not src.exists():
            print(f"  ⚠  Introuvable, ignoré : {src}")
            continue
        shutil.copy(src, out / target)
        manifest.append(target)

    return ok, manifest


def sanity_check_adapter(out: Path) -> bool:
    """Vérifie que adapter_config.json pointe bien vers le modèle de base attendu."""
    cfg_path = out / "adapter_config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ❌ adapter_config.json illisible : {e}")
        return False

    base = cfg.get("base_model_name_or_path", "")
    r = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    print(f"  Modèle de base déclaré : {base}")
    print(f"  LoRA r={r}, alpha={alpha}")

    if "Mistral-7B-Instruct-v0.3" not in str(base):
        print(f"  ⚠  Le modèle de base ne correspond pas à celui annoncé dans la carte "
              f"(Mistral-7B-Instruct-v0.3). Corrigez la carte ou vérifiez l'adaptateur.")
        return False
    if r != 16 or alpha != 32:
        print(f"  ⚠  r/alpha diffèrent des valeurs annoncées dans la carte (r=16, alpha=32).")
    return True


def main():
    p = argparse.ArgumentParser(description="Publie l'adaptateur ORPO sur le Hub")
    p.add_argument("--adapter", required=True, help="Dossier de l'adaptateur LoRA final")
    p.add_argument("--repo", required=True, help="CID99/Mistral-7B-ORPO-CSRD")
    p.add_argument("--card", default="MODEL_CARD.md", help="Carte de modèle à publier")
    p.add_argument("--pairs", default=None, help="orpo_pairs_*.jsonl")
    p.add_argument("--training-log", default=None, help="training_log_*.json")
    p.add_argument("--eval-report", default=None, help="eval_report_orpo.json")
    p.add_argument("--eval-predictions", default=None, help="raw_predictions_orpo_*.csv")
    p.add_argument("--out", default="hf_model_build")
    p.add_argument("--push", action="store_true")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        print(f"❌ Dossier d'adaptateur introuvable : {adapter_dir}")
        sys.exit(1)

    out = Path(args.out)
    extras = {
        "orpo_pairs.jsonl":    Path(args.pairs) if args.pairs else None,
        "training_log.json":   Path(args.training_log) if args.training_log else None,
        "eval_report.json":    Path(args.eval_report) if args.eval_report else None,
        "eval_predictions.csv": Path(args.eval_predictions) if args.eval_predictions else None,
    }

    print(f"{'═'*64}\n  Préparation\n{'═'*64}")
    ok, manifest = collect(adapter_dir, extras, out)

    print(f"\n{'═'*64}\n  Vérification de l'adaptateur\n{'═'*64}")
    ok = sanity_check_adapter(out) and ok

    # --- Carte de modèle ---
    card = Path(args.card)
    if card.exists():
        shutil.copy(card, out / "README.md")
        manifest.append("README.md")
        txt = card.read_text(encoding="utf-8")
        if "YOUR_USERNAME" in txt:
            print(f"\n  ⚠  « YOUR_USERNAME » subsiste dans {card.name} — remplacez-le par "
                  f"votre identifiant avant publication, sinon les liens seront morts.")
    else:
        print(f"\n  ❌ Carte de modèle introuvable : {card}")
        print(f"     Publier un modèle sans carte le rend inutilisable et peu sérieux.")
        ok = False

    # --- Artefacts manquants : avertir sans bloquer ---
    missing_research = [k for k, v in extras.items() if v is None]
    if missing_research:
        print(f"\n  ⚠  Artefacts de recherche non fournis : {', '.join(missing_research)}")
        print(f"     Ils sont annoncés dans la carte de modèle. Sans eux, l'échec publié")
        print(f"     n'est pas reproductible — c'est pourtant tout l'intérêt de cette")
        print(f"     publication. Fournissez-les via --pairs / --training-log /")
        print(f"     --eval-report / --eval-predictions, ou retirez-les de la carte.")

    print(f"\n{'═'*64}\n  Contenu du dépôt ({len(manifest)} fichiers)\n{'═'*64}")
    total = 0
    for f in sorted(out.rglob("*")):
        if f.is_file():
            mo = f.stat().st_size / 1_048_576
            total += mo
            print(f"    {str(f.relative_to(out)):<32} {mo:>8.1f} Mo")
    print(f"    {'TOTAL':<32} {total:>8.1f} Mo")

    if not args.push:
        print(f"\n{'═'*64}")
        print("  VÉRIFICATION À BLANC — rien n'a été publié.")
        print("  Relancez avec --push --private quand tout est correct.")
        print(f"{'═'*64}")
        return

    if not ok:
        print("\n❌ Problèmes bloquants ci-dessus. Publication annulée.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model")
    print(f"\n✅ Publié : https://huggingface.co/{args.repo}")
    if args.private:
        print("   (dépôt privé — Settings → Change visibility pour le rendre public)")


if __name__ == "__main__":
    main()
