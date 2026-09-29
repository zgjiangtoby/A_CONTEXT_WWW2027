#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
case "$command" in
  smoke) exec "${PYTHON:-python3}" smoke.py "$@" ;;
  grid) exec "${PYTHON:-python3}" a_capacity_grid.py "$@" ;;
  policy) exec "${PYTHON:-python3}" a_policy_controls.py "$@" ;;
  fine) exec "${PYTHON:-python3}" a_fine_boundary.py "$@" ;;
  natural) exec "${PYTHON:-python3}" a_natural_probe.py "$@" ;;
  *) printf '%s\n' 'Usage: bash run.sh {smoke|grid|policy|fine|natural} [arguments]' ;;
esac
