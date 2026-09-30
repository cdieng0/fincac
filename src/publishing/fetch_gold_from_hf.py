"""
fetch_gold_from_hf.py — Reconstitue le Gold Standard local depuis Hugging Face
═══════════════════════════════════════════════════════════════════════════════
Les scripts d'évaluation (benchmark, paires ORPO, entraînement, évaluation ORPO,
probing) lisent le Gold Standard sous la forme du classeur utilisé pendant
l'annotation :

    data/gold_150_annotated_clean_reformulated_without.xlsx   (feuille « Annotation »)

Ce classeur n'est pas versionné dans git. Le Gold est publié sur le Hub
(CID99/FinCAC40, config « gold », split « test », 140 paragraphes). Ce script
télécharge ce split et écrit le classeur au format attendu, sans clé API.

Correspondance des colonnes (Hub → classeur d'annotation) :
    content → « content — Extrait à annoter »
    epoch   → « Période »
Les autres colonnes sont conservées telles quelles. Le classeur ne contient
volontairement pas à la fois `content` et `content — Extrait à annoter` (ni
`epoch` et `Période`) : evaluate_orpo.py renomme ces colonnes, et un doublon y
produirait des colonnes homonymes.

Usage :
    python src/publishing/fetch_gold_from_hf.py
    python src/publishing/fetch_gold_from_hf.py --repo CID99/FinCAC40 --out data/mon_gold.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_REPO = "CID99/FinCAC40"
DEFAULT_OUT = "data/gold_150_annotated_clean_reformulated_without.xlsx"
SHEET_NAME = "Annotation"
EXPECTED_ROWS = 140

HUB_TO_WORKBOOK = {
    "content": "content — Extrait à annoter",
    "epoch": "Période",
}
REQUIRED_HUB_COLUMNS = {"content", "epoch", "csrd_category", "esrs_subcategory"}


def to_annotation_workbook(gold: pd.DataFrame) -> pd.DataFrame:
    """Convertit le split `gold` du Hub au schéma du classeur d'annotation."""
    missing = REQUIRED_HUB_COLUMNS - set(gold.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du split gold : {sorted(missing)}")
    clashes = [new for new in HUB_TO_WORKBOOK.values() if new in gold.columns]
    if clashes:
        raise ValueError(f"Colonnes déjà présentes, renommage ambigu : {clashes}")
    return gold.rename(columns=HUB_TO_WORKBOOK)


def download_gold(repo: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Installez la bibliothèque `datasets` : pip install datasets")
    return load_dataset(repo, "gold", split="test").to_pandas()


def main() -> None:
    p = argparse.ArgumentParser(description="Télécharge le Gold Standard FinCAC40 depuis le Hub")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"dépôt du dataset (défaut : {DEFAULT_REPO})")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"classeur à écrire (défaut : {DEFAULT_OUT})")
    args = p.parse_args()

    gold = download_gold(args.repo)
    workbook = to_annotation_workbook(gold)
    if len(workbook) != EXPECTED_ROWS:
        print(f"⚠  {len(workbook)} lignes au lieu des {EXPECTED_ROWS} attendues.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    workbook.to_excel(out, sheet_name=SHEET_NAME, index=False)
    print(f"Gold Standard écrit : {out} ({len(workbook)} paragraphes, feuille « {SHEET_NAME} »)")


if __name__ == "__main__":
    main()
