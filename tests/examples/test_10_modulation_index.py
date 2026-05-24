"""Smoke test for examples/10_modulation_index.ipynb — runs jupyter nbconvert --execute."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parent.parent.parent
    / "examples" / "10_modulation_index.ipynb"
)


@pytest.mark.skipif(shutil.which('jupyter') is None, reason='jupyter not installed')
def test_10_modulation_index_exists(tmp_path: Path) -> None:
    # Arrange
    # Act
    # Assert
    assert NOTEBOOK.exists(), f'missing example: {NOTEBOOK}'
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    r = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--output', target.name, str(target)], cwd=tmp_path, capture_output=True, text=True, timeout=300)

@pytest.mark.skipif(shutil.which('jupyter') is None, reason='jupyter not installed')
def test_10_modulation_index_returncode(tmp_path: Path) -> None:
    # Arrange
    # Act
    # Assert
    pass
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    r = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--output', target.name, str(target)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f'10_modulation_index.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}'
