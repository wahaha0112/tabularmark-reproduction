#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -f /.dockerenv ]]; then
  echo "This script must run inside the project Docker container." >&2
  exit 1
fi

if [[ ! -x .conda-env/bin/python ]]; then
  conda create --yes --prefix "$project_root/.conda-env" python=3.10 pip
fi

.conda-env/bin/python -m pip install --upgrade pip
.conda-env/bin/python -m pip install -r requirements.txt

echo "Environment ready: $project_root/.conda-env"
