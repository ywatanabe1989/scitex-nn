"""Smoke test for examples/13_spatial_attention.ipynb — runs jupyter nbconvert --execute."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parent.parent.parent
    / "examples" / "13_spatial_attention.ipynb"
)


@pytest.mark.skipif(shutil.which('jupyter') is None, reason='jupyter not installed')
def test_13_spatial_attention_exists(tmp_path: Path) -> None:
    # Arrange
    # Act
    # Assert
    assert NOTEBOOK.exists(), f'missing example: {NOTEBOOK}'
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    r = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--output', target.name, str(target)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    pass

@pytest.mark.skipif(shutil.which('jupyter') is None, reason='jupyter not installed')
def test_13_spatial_attention_returncode(tmp_path: Path) -> None:
    # Arrange
    # Act
    # Assert
    pass
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    r = subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--output', target.name, str(target)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f'13_spatial_attention.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}'
