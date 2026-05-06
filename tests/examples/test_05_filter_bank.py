"""Smoke test: examples/05_filter_bank.py runs to completion."""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "05_filter_bank.py"


def test_runs(tmp_path):
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, (
        f"05_filter_bank.py failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
