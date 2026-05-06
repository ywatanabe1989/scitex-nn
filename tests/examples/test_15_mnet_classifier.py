"""Smoke test: examples/15_mnet_classifier.py runs to completion."""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "15_mnet_classifier.py"


def test_runs(tmp_path):
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, (
        f"15_mnet_classifier.py failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
