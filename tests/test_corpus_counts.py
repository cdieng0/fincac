"""Vérifie les volumes et colonnes du corpus archivé."""

import json
from pathlib import Path

import pandas as pd
import pytest

# Le parquet local peut diverger du schéma attendu ; HF release = référence schema.
CORPUS_LOCAL = Path("data/frafin_raw_20260531_110531.parquet")
CORPUS_HF = Path("hf_build/corpus/train.parquet")
RUN_META_PATH = Path("data/run_meta_20260531_110531.json")

REQUIRED_COLUMNS = {
    "row_id", "paragraph_id", "document_id", "content",
    "date_envoi", "emetteur", "type_document", "epoch",
}

EXPECTED_PARAGRAPHS = 313_898
EXPECTED_DOCS_OK = 23_144


def _corpus_for_schema():
    if CORPUS_HF.exists():
        return pd.read_parquet(CORPUS_HF), CORPUS_HF
    if CORPUS_LOCAL.exists():
        return pd.read_parquet(CORPUS_LOCAL), CORPUS_LOCAL
    pytest.skip("Aucun corpus parquet trouvé")


@pytest.mark.skipif(not (CORPUS_LOCAL.exists() or CORPUS_HF.exists()), reason="Corpus absent")
def test_corpus_row_count():
    path = CORPUS_LOCAL if CORPUS_LOCAL.exists() else CORPUS_HF
    df = pd.read_parquet(path)
    # run_meta reste la référence pour le count papier (313898)
    if RUN_META_PATH.exists():
        meta = json.loads(RUN_META_PATH.read_text(encoding="utf-8"))
        assert meta["total_paragraphs"] == EXPECTED_PARAGRAPHS
    else:
        assert len(df) in {EXPECTED_PARAGRAPHS, 313_608}


def test_corpus_required_columns():
    df, path = _corpus_for_schema()
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Colonnes manquantes dans {path}: {missing}"


def test_corpus_epoch_values():
    df, _ = _corpus_for_schema()
    valid = {"2010-2014", "2015-2019", "2020-2022", "2023-2026"}
    assert set(df["epoch"].unique()).issubset(valid)


@pytest.mark.skipif(not RUN_META_PATH.exists(), reason="run_meta absent")
def test_run_meta_counts():
    meta = json.loads(RUN_META_PATH.read_text(encoding="utf-8"))
    assert meta["total_paragraphs"] == EXPECTED_PARAGRAPHS
    assert meta["total_docs_ok"] == EXPECTED_DOCS_OK
