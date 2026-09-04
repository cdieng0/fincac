"""Vérifie la cohérence des prompts expérimentaux."""

import os
import subprocess
import sys


def test_prompt_consistency():
    env = os.environ.copy()
    env.setdefault("MISTRAL_API_KEY", "dummy_for_test")
    result = subprocess.run(
        [sys.executable, "test_prompt_consistency.py"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr
