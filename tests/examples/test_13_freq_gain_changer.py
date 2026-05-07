"""Smoke test: examples/13_freq_gain_changer.py runs to completion."""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "13_freq_gain_changer.py"


def test_runs(tmp_path):
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, (
        f"13_freq_gain_changer.py failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
