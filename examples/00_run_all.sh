#!/usr/bin/env bash
# Re-execute every example notebook in place.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

for nb in 0[1-9]_*.ipynb 1[0-9]_*.ipynb; do
    [ -f "$nb" ] || continue
    echo "==> $nb"
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
