"""Garde-fou contre la dérive de prompt (cf. docs/INCIDENT_NOTE.md).

Le benchmark de la Section 4 et toute la chaîne ORPO (collecte des paires,
entraînement, évaluation) doivent interroger les modèles avec exactement le même
prompt système. En août 2026, des copies indépendantes de ce prompt avaient
divergé silencieusement ; ces tests échouent dès qu'une copie diverge à nouveau.

Aucune clé API ni GPU n'est nécessaire.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MISTRAL_API_KEY", "dummy_for_tests")

from src.evaluation import shared_prompts as sp  # noqa: E402

EXPECTED_CATEGORIES = {
    "none", "ESRS2", "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4", "G1",
}
ORPO_SCRIPTS = [
    "src/orpo/build_orpo_pairs.py",
    "src/orpo/train_orpo_mistral7b.py",
    "src/orpo/evaluate_orpo.py",
]
PROMPT_BLOCKS = {"TAXONOMY_BLOCK", "TASK_BLOCK", "FEW_SHOT_EXAMPLES"}


def _load_benchmark():
    path = ROOT / "src" / "evaluation" / "benchmark_temporal_drift.py"
    spec = importlib.util.spec_from_file_location("benchmark_temporal_drift", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(rel):
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _assigned_names(tree):
    return {
        target.id
        for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }


@pytest.mark.parametrize("n_shot", [0, 3])
def test_benchmark_prompt_is_byte_identical_to_shared_prompt(n_shot):
    """Le benchmark garde sa propre copie du prompt : elle doit rester identique."""
    benchmark = _load_benchmark()
    assert benchmark.build_system_prompt(n_shot) == sp.build_system_prompt(n_shot)


def test_shared_prompt_is_deterministic():
    assert sp.build_system_prompt(3) == sp.build_system_prompt(3)


def test_three_shot_prompt_extends_zero_shot_prompt():
    zero, three = sp.build_system_prompt(0), sp.build_system_prompt(3)
    assert sp.TAXONOMY_BLOCK.strip() in zero
    assert sp.TAXONOMY_BLOCK.strip() in three
    assert len(three) > len(zero)


def test_taxonomy_has_the_twelve_esrs_categories():
    assert set(sp.VALID_CATEGORIES) == EXPECTED_CATEGORIES


@pytest.mark.parametrize("rel", ORPO_SCRIPTS)
def test_orpo_script_imports_the_shared_prompt(rel):
    imports = [
        node for node in ast.walk(_tree(rel))
        if isinstance(node, ast.ImportFrom) and node.module == "src.evaluation.shared_prompts"
    ]
    assert imports, f"{rel} doit importer build_system_prompt depuis shared_prompts"
    assert any(alias.name == "build_system_prompt" for node in imports for alias in node.names)


@pytest.mark.parametrize("rel", ORPO_SCRIPTS)
def test_orpo_script_does_not_redefine_prompt_blocks(rel):
    redefined = _assigned_names(_tree(rel)) & PROMPT_BLOCKS
    assert not redefined, f"{rel} redéfinit localement {sorted(redefined)}"


@pytest.mark.parametrize("rel", ORPO_SCRIPTS)
def test_orpo_policy_is_queried_in_three_shot(rel):
    """La politique de base est interrogée en 3-shot, comme en Section 4."""
    values = [
        node.value.value
        for node in ast.walk(_tree(rel))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "N_SHOT_POLICY" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert values == [3], f"{rel} : N_SHOT_POLICY = {values}"


@pytest.mark.parametrize("n_shot", [0, 3])
def test_published_prompt_files_match_the_code(n_shot):
    """prompts/system_prompt_{n}shot.txt est le prompt exact, lisible sans exécuter le code."""
    path = ROOT / "prompts" / f"system_prompt_{n_shot}shot.txt"
    assert path.read_text(encoding="utf-8") == sp.build_system_prompt(n_shot), (
        f"{path.name} a divergé de shared_prompts.build_system_prompt({n_shot})"
    )
