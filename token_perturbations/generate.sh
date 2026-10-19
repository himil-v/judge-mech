#!/usr/bin/env bash
# Regenerate the full perturbation corpus: 2 datasets x 3 seeds x 8 attacks, 200 samples each, k = 1.
# Requires OPENAI_API_KEY (see .env.example). Roughly 4,800 GPT-4o calls per dataset.
# The released files in generated_perturbations/ are the corpus used in the paper; generation
# is sampled at temperature 0.7, so a rerun produces a different (but equivalent) corpus.
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(cnn_dailymail xsum)
SEEDS=(42 23 17)
ATTACKS=(
  read_preposition read_tense read_spelling read_reordering
  adeq_entity_1to1 adeq_polarity_swap adeq_numerical adeq_coreference
)

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
      python generate_perturbations.py \
        --dataset "$dataset" --attack "$attack" \
        --samples 200 --intensity 1 --seed "$seed"
    done
  done
done