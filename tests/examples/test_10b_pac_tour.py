"""Smoke test: examples/10b_pac_tour.ipynb executes every cell."""

import shutil
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "10b_pac_tour.ipynb"


def test_runs(tmp_path):
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter not installed")
    out = tmp_path / "executed.ipynb"
    r = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(out),
            str(EXAMPLE),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, (
        f"10b_pac_tour.ipynb failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
