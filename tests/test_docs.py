"""Garde-fous sur la documentation : protocole synchronisé avec la taxonomie, liens internes
valides, aucun gabarit laissé vide."""

import re
from pathlib import Path

import pytest

from src.annotation import csrd_taxonomy as taxonomy

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = sorted(
    p for p in ROOT.rglob("*.md")
    if not {".venv", ".git", "node_modules", ".pytest_cache"} & set(p.parts)
)
PLACEHOLDERS = (
    "[More Information Needed]", "YOUR_USERNAME", "VOTRE_USERNAME", "<URL_DU_DEPOT>",
    "(à créer)", "(à localiser)", "TODO", "TBD",
)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def test_annotation_protocol_lists_every_taxonomy_code():
    protocol = (ROOT / "docs" / "annotation_protocol.md").read_text(encoding="utf-8")
    missing = [code for code in taxonomy.CSRD_CATEGORIES if f"`{code}`" not in protocol]
    missing += [
        sub["code"]
        for code, subs in taxonomy.ESRS_SUBCATEGORIES.items() if code != taxonomy.NONE_CATEGORY
        for sub in subs if f"`{sub['code']}`" not in protocol
    ]
    assert not missing, f"Codes absents de docs/annotation_protocol.md : {missing}"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_placeholder_left(path):
    text = path.read_text(encoding="utf-8")
    found = [p for p in PLACEHOLDERS if p in text]
    assert not found, f"{path.relative_to(ROOT)} contient {found}"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path):
    broken = []
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue                                  # URL externe ou ancre locale
        file_part = target.split("#", 1)[0]
        if file_part and not (path.parent / file_part).exists():
            broken.append(target)
    assert not broken, f"{path.relative_to(ROOT)} : liens cassés {broken}"
