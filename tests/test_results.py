"""Vérifie les artefacts archivés dans data/results/, sans GPU, sans API, sans téléchargement.

Les rapports d'évaluation du Run 2 sont recalculés à partir des prédictions par exemple
archivées, avec le script d'évaluation du dépôt, et comparés aux rapports archivés. Les
chiffres cités dans le papier sont vérifiés au passage.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.orpo.evaluate_orpo import (
    compute_confusion_summary,
    compute_fnr_by_epoch,
    compute_fpr_by_epoch,
    compute_overall_metrics,
)

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"
RUN1 = RESULTS / "orpo_run1"
RUN2 = RESULTS / "orpo_run2"
PROBING = RESULTS / "probing"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run2_predictions():
    return pd.read_csv(RUN2 / "eval_predictions.csv")


@pytest.fixture(scope="module")
def run2_report():
    return _json(RUN2 / "eval_report.json")


def test_run2_predictions_cover_the_gold(run2_predictions):
    assert len(run2_predictions) == 140
    assert run2_predictions["paragraph_id"].is_unique
    assert run2_predictions["epoch"].value_counts().to_dict() == {
        "2010-2014": 33, "2015-2019": 31, "2020-2022": 33, "2023-2026": 43,
    }


def test_run2_report_is_reproduced_from_its_predictions(run2_predictions, run2_report):
    metrics = compute_overall_metrics(run2_predictions)
    for key in ("n_examples", "accuracy", "f1_macro",
                "kappa_binary", "kappa_binary_ci95_low", "kappa_binary_ci95_high"):
        assert metrics[key] == pytest.approx(run2_report["orpo_metrics"][key]), key

    for section, compute in (("orpo_fnr_by_epoch", compute_fnr_by_epoch),
                             ("orpo_fpr_by_epoch", compute_fpr_by_epoch)):
        recomputed = compute(run2_predictions)
        for epoch, archived in run2_report[section].items():
            for field, value in archived.items():
                assert recomputed[epoch][field] == pytest.approx(value), (section, epoch, field)

    assert compute_confusion_summary(run2_predictions) == run2_report["orpo_confusion"]


def test_run2_paper_figures(run2_predictions):
    """Tableau 4 du papier : 35,7 % ; 35,9 % ; κ 0,238 [0,079 ; 0,387] ; 30,7 % ; FPR 64,9 %."""
    m = compute_overall_metrics(run2_predictions)
    assert round(m["accuracy"], 3) == 0.357
    assert round(m["f1_macro"], 3) == 0.359
    assert (m["kappa_binary"], m["kappa_binary_ci95_low"], m["kappa_binary_ci95_high"]) == (
        pytest.approx(0.2378), pytest.approx(0.0787), pytest.approx(0.3866))
    assert m["parse_failure_rate"] == pytest.approx(43 / 140)

    fpr = compute_fpr_by_epoch(run2_predictions)
    assert sum(v["n_none_gold"] for v in fpr.values()) == 97
    assert sum(v["n_false_positive"] for v in fpr.values()) == 63     # 64,9 %
    # Décomposition : 35 étiquettes CSRD, 28 sorties non analysables
    assert sum(v["n_csrd_label"] for v in fpr.values()) == 35
    assert sum(v["n_parse_error"] for v in fpr.values()) == 28


def test_run2_esrs2_refuge_category(run2_predictions):
    """Section 4.3 : précision ESRS2 de 1/19 et rappel de 1/17."""
    pred = run2_predictions["pred_category"] == "ESRS2"
    gold = run2_predictions["csrd_category"] == "ESRS2"
    assert (int((pred & gold).sum()), int(pred.sum()), int(gold.sum())) == (1, 19, 17)


def test_archived_parse_failure_rate_is_the_documented_bug(run2_report):
    """Le rapport archivé porte 0.0 (bug décrit dans data/results/README.md) ; le vrai taux est 43/140."""
    assert run2_report["orpo_metrics"]["parse_failure_rate"] == 0.0
    assert run2_report["orpo_confusion"]["predicted_category_distribution"]["PARSE_ERROR"] == 43


def test_run1_artefacts_match_the_paper():
    report, meta = _json(RUN1 / "eval_report.json"), _json(RUN1 / "run_meta.json")
    assert round(report["orpo_metrics"]["accuracy"], 3) == 0.164
    assert round(report["orpo_metrics"]["f1_macro"], 3) == 0.231
    assert all(v["fnr"] == 0.0 for v in report["orpo_fnr_by_epoch"].values())
    assert (meta["num_epochs"], meta["learning_rate"]) == (3, 5e-05)
    assert (meta["n_train_pairs"], meta["n_val_pairs"]) == (273, 42)


def test_run1_training_telemetry():
    log = _json(RUN1 / "training_log.json")
    evals = [e for e in log if "eval_loss" in e]
    assert round(evals[0]["eval_loss"], 2) == 1.03 and round(evals[-1]["eval_loss"], 3) == 0.147
    assert evals[-1]["eval_rewards/accuracies"] == 1.0


def test_probing_results_match_the_paper():
    """Section 4.2 : couche 19, 45,0 % contre 30,7 % (classe majoritaire), p = 0,005."""
    results = _json(PROBING / "probing_results_20260827_164651.json")
    table = pd.read_csv(PROBING / "probing_results_20260827_164651.csv")
    assert len(table) == 33 == results["model_meta"]["n_layers"]
    assert int(table.loc[table["accuracy_mean"].idxmax(), "layer"]) == 19
    assert results["best_layer"]["index"] == 19
    assert results["best_layer"]["accuracy_mean"] == pytest.approx(0.45)
    assert results["majority_class_baseline"] == pytest.approx(43 / 140)
    assert results["permutation_test"]["p_value"] == pytest.approx(1 / 201)
