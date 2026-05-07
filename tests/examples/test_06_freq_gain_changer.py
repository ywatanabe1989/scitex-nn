"""Smoke test for examples/06_freq_gain_changer.ipynb — runs jupyter nbconvert --execute."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parent.parent.parent
    / "examples" / "06_freq_gain_changer.ipynb"
)


def test_06_freq_gain_changer(tmp_path: Path) -> None:
    assert NOTEBOOK.exists(), f"missing example: {NOTEBOOK}"
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter not installed")

    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    r = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--output", target.name, str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, (
        f"06_freq_gain_changer.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
