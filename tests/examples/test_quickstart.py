"""Smoke test: examples/quickstart.py runs to completion.

Per-example smoke tests for the numbered examples live in
``tests/examples/test_<NN>_<name>.py``; this one only covers the
unnumbered ``quickstart.py`` so the suite stays a thin parallel of
``examples/`` (audit-project PS303).
"""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "quickstart.py"


def test_runs(tmp_path):
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, (
        f"quickstart.py failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
