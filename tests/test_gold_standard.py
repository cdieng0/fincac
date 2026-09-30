"""Vérifie le Gold Standard archivé (140 paragraphes)."""

from pathlib import Path

import pandas as pd
import pytest

GOLD_XLSX = Path("data/gold_150_annotated_clean_reformulated_without.xlsx")
GOLD_PARQUET = Path("hf_build/gold/test.parquet")  # sortie locale de publish_dataset_to_hf.py

EXPECTED_GOLD_SIZE = 140
VALID_CATEGORIES = {
    "none", "ESRS2", "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4", "G1",
}


def _load_gold():
    if GOLD_XLSX.exists():
        return pd.read_excel(GOLD_XLSX)
    if GOLD_PARQUET.exists():
        return pd.read_parquet(GOLD_PARQUET)
    pytest.skip("Gold Standard absent : lancez python src/publishing/fetch_gold_from_hf.py")


def test_gold_size():
    df = _load_gold()
    assert len(df) == EXPECTED_GOLD_SIZE


def test_gold_categories_valid():
    df = _load_gold()
    col = "csrd_category" if "csrd_category" in df.columns else "category"
    if col not in df.columns:
        pytest.skip("Colonne catégorie absente")
    invalid = set(df[col].dropna()) - VALID_CATEGORIES
    assert not invalid, f"Catégories invalides : {invalid}"


def test_gold_epoch_coverage():
    df = _load_gold()
    col = next((c for c in ("epoch", "Période") if c in df.columns), None)
    if col is None:
        pytest.skip("Colonne d'époque absente")
    assert len(df[col].unique()) == 4
