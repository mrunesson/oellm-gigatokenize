#!/usr/bin/env bash
#SBATCH --job-name=oellm-tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=oellm-tokenize-%j.out
#
# Tokenize INPUT_DIR into OUTPUT_DIR with oellm-gigatokenize. Submit from the
# repository root and give the time limit on the command line:
#
#   sbatch --time=02:00:00 --account=PROJECT scripts/tokenize.sbatch INPUT_DIR OUTPUT_DIR [tool options...]
#
# Options after the directories go to oellm-gigatokenize, e.g. -n 8.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: sbatch --time=... $0 INPUT_DIR OUTPUT_DIR [tool options...]" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-.}"
uv run --frozen oellm-gigatokenize "$@"
