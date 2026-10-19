# Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation

Code and data for the EMNLP 2026 paper by **Himil Vasava** and **Ming Jiang** (University of Wisconsin-Madison).

Paper: [arXiv:2609.01604](https://arxiv.org/abs/2609.01604)

We study how LLM judges compute a rating. Using an eight-attack perturbation taxonomy and paired clean/corrupt summaries with token-level modification maps, we apply causal tracing, logit-lens projection and attention-head knockout to Themis (Llama-3-8B) and Prometheus (Mistral-7B), with Llama-3-8B as a base-model control.

## Repository structure

```
judge-mech/
├── token_perturbations/
│   ├── prompts.py                    # GPT-4o few-shot prompts for the eight attacks
│   ├── generate_perturbations.py     # generates clean/corrupt pairs with GPT-4o
│   ├── generate.sh                   # regenerates the full corpus
│   └── generated_perturbations/      # released corpus (48 files)
├── configs/
│   └── aspect_criteria.json          # evaluation criteria definitions (Hu et al., 2024a)
├── causal_tracing/
│   ├── causal_tracing_logit_diff.py       # activation patching, logit-difference statistic
│   └── causal_tracing_expected_rating.py  # activation patching, expected-rating statistic
├── logit_lens/
│   └── logit_lens.py                 # rating-token trajectories across layers
├── head_knockout/
│   └── head_knockout.py              # per-head ablation of all 32 x 32 attention heads
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/himil-v/judge-mech.git
cd judge-mech
pip install -r requirements.txt
```

- `meta-llama/Meta-Llama-3-8B` is a gated model on Hugging Face. Request access and run `huggingface-cli login` before using it.
- An OpenAI API key is needed only to regenerate the perturbation corpus: create a .env file containing OPENAI_API_KEY=your-key, or export the variable.
- Models are loaded in float16 and sharded across available GPUs with `device_map="auto"`.

## Data

`token_perturbations/generated_perturbations/` contains the corpus used in our experiments: 2 datasets (CNN/DailyMail, XSum) x 8 attacks x 3 seeds (17, 23, 42), with 200 samples per file drawn from the train split, at intensity k = 1.

Files are named `{dataset}_{attack}_int1_200_{seed}.json`. Each record contains:

| Field | Description |
|---|---|
| `document` | source article |
| `summary` | clean reference summary |
| `perturbed_summary` | corrupted summary |
| `original_tokens` | tokens replaced in the clean summary |
| `new_tokens` | tokens inserted in the corrupted summary |
| `perturbation_type` | attack key |
| `intensity` | k |

Attack keys and names:

| Key | Attack | Criterion |
|---|---|---|
| `read_preposition` | Preposition Mismatch | Readability |
| `read_tense` | Tense Mismatch | Readability |
| `read_spelling` | Spelling | Readability |
| `read_reordering` | Sequential Reordering | Readability |
| `adeq_entity_1to1` | Entity Swap | Adequacy |
| `adeq_numerical` | Numerical/Date Swap | Adequacy |
| `adeq_coreference` | Coreference Mismatch | Adequacy |
| `adeq_polarity_swap` | Antonym & Negation (also called Polarity) | Adequacy |

The files are the raw generator output. Records whose generation failed have no `perturbation_type` and are dropped by the experiment scripts, along with records whose `new_tokens` is empty. Behavioral filters are applied per evaluator inside each experiment script.

To regenerate the corpus (about 4,800 GPT-4o calls per dataset):

```bash
cd token_perturbations
bash generate.sh
```

Generation uses temperature 0.7, so a rerun produces a different corpus. Use the released files to reproduce our results.

## Evaluators and criteria

| Model | Prompt |
|---|---|
| `PKU-ONELab/Themis` | single-instruction completion prompt |
| `prometheus-eval/prometheus-7b-v2.0` | five-shot direct-rating prompt in the Mistral chat template |
| `meta-llama/Meta-Llama-3-8B` | same prompt as Themis |


`configs/aspect_criteria.json` is taken from Hu et al. (2024a), *Are LLM-Based Evaluators Confusing NLG Quality Criteria?* We use the `Gold` definition of **Readability** for the readability attacks and of **Adequacy** for the adequacy attacks (`--criteria Readability` / `--criteria Adequacy`). The definition text is inserted into the prompt verbatim.

## Experiments

All experiment scripts take `--dataset_path`, `--dataset_name`, `--attack`, `--criteria` and `--seed`, and write results and plots under `results/` and `plots/`. Each run covers one (model, attack, seed) file; loop over the 8 attacks and 3 seeds for the full sweep.

```bash
DATA=token_perturbations/generated_perturbations
```

### Causal tracing

Both scripts patch clean activations (MLP outputs or attention output projections) into the corrupt forward pass and report the normalized effect `(X_patched - X_corrupt) / (X_clean - X_corrupt)`. They differ in the output statistic X:

- `causal_tracing_logit_diff.py`: X is the logit difference between the clean and corrupt argmax tokens. Samples must flip the argmax rating and satisfy `X_clean - X_corrupt >= 0.05`.
- `causal_tracing_expected_rating.py`: X is the expected rating over the five rating tokens. Samples must satisfy `|E_clean - E_corrupt| >= 0.05`; in window mode, length-mismatched pairs are realigned with difflib.

Our experiments use both statistics. A detailed mapping from each figure and table to its script and settings will be added.

```bash
# last-token mode
python causal_tracing/causal_tracing_logit_diff.py \
  --model_name PKU-ONELab/Themis --dataset_name cnn_dailymail \
  --dataset_path $DATA/cnn_dailymail_adeq_entity_1to1_int1_200_42.json \
  --attack adeq_entity_1to1 --criteria Adequacy --seed 42 \
  --target_pos last --module_type mlp

# window mode (6-token window starting at the perturbed token)
python causal_tracing/causal_tracing_expected_rating.py \
  --model_name PKU-ONELab/Themis --dataset_name cnn_dailymail \
  --dataset_path $DATA/cnn_dailymail_read_preposition_int1_200_42.json \
  --attack read_preposition --criteria Readability --seed 42 \
  --target_pos window --module_type attn
```

Use `--module_type mlp|attn` for the two sublayers. `--slice_start` / `--slice_end` split a run across GPUs. The two scripts write files with the same names, so their default output folders differ (`results/logit_diff/`, `results/expected_rating/`).

### Logit lens

Projects the residual stream at the final position through the final norm and unembedding at every layer and records the probabilities of the five rating tokens.

```bash
python logit_lens/logit_lens.py \
  --model_name PKU-ONELab/Themis --dataset_name cnn_dailymail \
  --dataset_path $DATA/cnn_dailymail_adeq_entity_1to1_int1_200_42.json \
  --attack adeq_entity_1to1 --criteria Adequacy --seed 42 \
  --filter logit_diff --threshold 0.1
```

### Attention-head knockout

Zeros each of the 32 x 32 attention heads (its slice of the output-projection input) on the corrupt prompt and reports the normalized logit-difference effect. We use global ablation (all positions).

```bash
python head_knockout/head_knockout.py \
  --model_name PKU-ONELab/Themis --dataset_name cnn_dailymail \
  --dataset_path $DATA/cnn_dailymail_adeq_entity_1to1_int1_200_42.json \
  --attack adeq_entity_1to1 --criteria Adequacy --seed 42 \
  --ablation_mode global --threshold 0.1 --batch_size 4
```

`--sample_start` / `--sample_end` split a run across processes.

## Citation

```bibtex
@article{vasava2026beyond,
  title={Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation},
  author={Vasava, Himil and Jiang, Ming},
  journal={arXiv preprint arXiv:2609.01604},
  year={2026}
}
```

## Acknowledgments

This research is partially supported by the National Science Foundation (IIS-2604291).