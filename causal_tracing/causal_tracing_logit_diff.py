"""
Activation patching (causal tracing) for LLM judges using the logit-difference metric.

Patches clean activations into the corrupt forward pass at (layer, position) sites, for
either MLP outputs (`model.layers.{i}.mlp`) or attention output projections
(`model.layers.{i}.self_attn.o_proj`), and reports the normalized causal effect

    effect = (X_patched - X_corrupt) / (X_clean - X_corrupt)

where X = z[t_clean] - z[t_corrupt] is the logit difference at the final position between
the clean run's argmax token and the corrupt run's argmax token.

Modes (--target_pos):
    last    patch only the final input position at every layer
    window  patch a 6-token window starting at each perturbed (anchor) token, offsets +0..+5.
            Clean and corrupt prompts must tokenize to the same length (no realignment).

Sample filters, applied in this order:
    1. structural: keep records whose perturbation_type matches --attack and whose
       new_tokens is non-empty (drops JSON/API failures and unmodified summaries)
    2. rating flip: the clean and corrupt argmax tokens must differ (the logit difference
       is defined between these two tokens)
    3. non-rating: both argmax tokens must be rating digits
    4. score difference: X_clean - X_corrupt >= 0.05
    5. window mode only: equal sequence lengths, and anchors (positions found by difflib)
       must decode to one of the generator's new_tokens

Supported evaluators: PKU-ONELab/Themis,
prometheus-eval/prometheus-7b-v2.0, meta-llama/Meta-Llama-3-8B (same prompt as Themis).

Outputs (in --results_dir / --plots_dir):
    *_sweep.csv        mean effect per cell (layers x positions)
    *_sweep_logs.csv   one row per (instance, layer, offset)
    *_sweep.png        quick-look heatmap (unsliced runs only)
"""

import os
import argparse
import json
import difflib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CONSTANTS & TEMPLATES
# ==========================================

PROMPT_TEMPLATE = """###Instruction###
Please act as an impartial and helpful evaluator for natural language generation (NLG).
Your task is to evaluate the quality of the Summarization strictly based on the given evaluation criterion.
You MUST keep to the strict boundaries of the evaluation criterion.
**Scoring Rules:**
1. Use a Likert scale from 1 to 5.
2. 5 is the highest quality, 1 is the lowest.
3. Output ONLY the number.
###Evaluation Criterion###
{prompt_criteria}
###Task###
Article:
{article}
Summary:
{summary}
###Rating###
The rating is """


# ---- Prometheus-2 (Mistral-Instruct based) ----
#
# Few-shot direct-rating prompt: 5 in-context examples spanning ratings
# 1-5, then the test sample, with "Rating: " appended after Mistral's
# [/INST] so the next-token argmax should be a rating digit. This avoids
# both force-decoding the feedback (greedy-decoding determinism concerns)
# and prepending a fixed stub feedback (per-sample rating bias concerns).
# The balanced rating coverage in the examples keeps the model from
# anchoring to a single rating value.

PROMETHEUS2_FEW_SHOT_BODY = """###Task Description:
You are an impartial evaluator. Read the article, the response, and the score rubric. Output ONLY a single integer between 1 and 5 representing your rating. Do not write any feedback, explanation, or any other text — just the digit. The examples below illustrate the expected output format.

{examples_block}

###Now evaluate this:

###Article:
{article}

###Response to evaluate:
{summary}

###Score Rubric:
{prompt_criteria}
"""


DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "article": "The Federal Reserve announced a 0.25 percentage point interest rate cut today, the third such reduction this year, citing concerns about slowing economic growth and persistent global trade tensions affecting U.S. manufacturing output.",
        "summary": "The Fed cut interest rates by 0.25 points today due to economic growth concerns.",
        "rubric": "Evaluate the overall quality of the summary based on how accurately and completely it captures the main facts of the article. Rate from 1 (highly inaccurate or off-topic) to 5 (fully accurate and comprehensive).",
        "rating": 4,
    },
    {
        "article": "The European Union announced a new trade agreement with Vietnam yesterday, eliminating tariffs on 99 percent of goods traded between the two regions over the next seven years and establishing new standards for labor and environmental protections.",
        "summary": "Vietnam and the EU had a meeting about trade and economic cooperation last week.",
        "rubric": "Evaluate the overall quality of the summary based on how accurately and completely it captures the main facts of the article. Rate from 1 (highly inaccurate or off-topic) to 5 (fully accurate and comprehensive).",
        "rating": 2,
    },
    {
        "article": "Scientists at Stanford University have developed a new lithium-metal battery technology that can charge a smartphone in 30 seconds and last three times longer than current lithium-ion batteries, with potential applications in electric vehicles and grid storage.",
        "summary": "Stanford researchers developed a lithium-metal battery that charges phones in 30 seconds and lasts three times longer than lithium-ion, with potential uses in electric vehicles and grid storage.",
        "rubric": "Evaluate the overall quality of the summary based on how accurately and completely it captures the main facts of the article. Rate from 1 (highly inaccurate or off-topic) to 5 (fully accurate and comprehensive).",
        "rating": 5,
    },
    {
        "article": "NASA's Perseverance rover has identified organic compounds in rock samples from Mars's Jezero crater, providing the strongest evidence yet of conditions that could have supported ancient microbial life on the planet, according to a study published in Nature.",
        "summary": "Astronauts from China landed on the moon last week to collect lunar samples for future research.",
        "rubric": "Evaluate the overall quality of the summary based on how accurately and completely it captures the main facts of the article. Rate from 1 (highly inaccurate or off-topic) to 5 (fully accurate and comprehensive).",
        "rating": 1,
    },
    {
        "article": "A magnitude 6.2 earthquake struck off the coast of northeastern Japan early Tuesday morning, causing minor structural damage to buildings in three coastal cities and triggering brief tsunami advisories that were lifted within two hours. No casualties have been reported.",
        "summary": "An earthquake hit Japan and caused some damage. There were no injuries.",
        "rubric": "Evaluate the overall quality of the summary based on how accurately and completely it captures the main facts of the article. Rate from 1 (highly inaccurate or off-topic) to 5 (fully accurate and comprehensive).",
        "rating": 3,
    },
]


DIRECT_RATING_SUFFIX = "Rating: "
RATING_TOKEN_STRINGS = {"1", "2", "3", "4", "5"}


# ==========================================
# 2. CACHING & HOOK INFRASTRUCTURE
# ==========================================

class ActivationCache:
    def __init__(self):
        self.clean_cache = {}

    def make_save_hook(self, module_name: str):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.clean_cache[module_name] = tensor.detach().clone().cpu()
        return hook

    def clear(self):
        self.clean_cache.clear()


def make_patch_hook(module_name: str, patch_pos: int, cache: ActivationCache):
    """Write clean's activation at `patch_pos` into the corrupt forward pass at the same
    index. In last-token mode patch_pos = -1, so clean and corrupt may differ in length."""
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        tensor = output[0].clone() if is_tuple else output.clone()
        clean_act = cache.clean_cache[module_name].to(tensor.device)
        tensor[:, patch_pos, :] = clean_act[:, patch_pos, :]
        if is_tuple:
            return (tensor,) + output[1:]
        return tensor
    return hook


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def is_prometheus_model(model_name: str) -> bool:
    """Detects Prometheus-2-family models (Mistral-Instruct based)."""
    return "prometheus" in model_name.lower()


def is_base_llama_model(model_name: str) -> bool:
    """Detects the un-fine-tuned base Llama-3-8B control (not -Instruct, not Themis).
    Used only to tag its output filenames."""
    n = model_name.lower()
    return ("meta-llama-3-8b" in n or "meta-llama/meta-llama-3-8b" in n) \
        and "instruct" not in n and "themis" not in n and "pku" not in n


def _build_examples_block(examples):
    """Render a list of {article, summary, rubric, rating} dicts into a
    formatted few-shot block."""
    parts = []
    for i, ex in enumerate(examples, start=1):
        parts.append(
            f"###Example {i}\n"
            f"Article:\n{ex['article']}\n\n"
            f"Response to evaluate:\n{ex['summary']}\n\n"
            f"Score Rubric: {ex['rubric']}\n\n"
            f"Rating: {ex['rating']}"
        )
    return "\n\n".join(parts)


def is_rating_token(token_id, tokenizer) -> bool:
    """True iff decoding token_id (whitespace-stripped) is a rating digit."""
    return tokenizer.decode(token_id).strip() in RATING_TOKEN_STRINGS


def format_prompt(document: str, summary: str, prompt_criteria: str, model_name: str,
                  tokenizer=None) -> str:
    """Formats the document and summary into the appropriate evaluator
    template based on the model architecture. Themis and the base Llama-3-8B
    control share the same single-instruction completion prompt."""

    if is_prometheus_model(model_name):
        # Prometheus-2: few-shot direct-rating prompt wrapped in Mistral
        # chat template. "Rating: " appended after [/INST] so the model's
        # next-token argmax should be a rating digit (1-5).
        examples_block = _build_examples_block(DEFAULT_FEW_SHOT_EXAMPLES)
        body = PROMETHEUS2_FEW_SHOT_BODY.format(
            examples_block=examples_block,
            article=document,
            summary=summary,
            prompt_criteria=prompt_criteria,
        )
        messages = [{"role": "user", "content": body}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return chat_prompt + DIRECT_RATING_SUFFIX

    # Themis and the base Llama-3-8B control (raw completion).
    return PROMPT_TEMPLATE.format(
        prompt_criteria=prompt_criteria,
        article=document,
        summary=summary,
    )


def calculate_score(logits, target_id, contra_id):
    """Logit difference z[target] - z[contra] at the final position."""
    final_logits = logits[0, -1, :]
    return (final_logits[target_id] - final_logits[contra_id]).item()


def get_all_perturbation_indices(clean_ids, corr_ids):
    """Returns all indices in the corrupt sequence that differ from clean."""
    matcher = difflib.SequenceMatcher(None, clean_ids, corr_ids)
    corrupt_anchor_indices = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            for j in range(j1, j2):
                corrupt_anchor_indices.append(j)
    return corrupt_anchor_indices


# ==========================================
# 4. MAIN EXPERIMENT PIPELINE
# ==========================================

def run_experiment(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.criteria_file, "r") as f:
        criteria_def = json.load(f)
    prompt_criteria = criteria_def[args.criteria]["Gold"]

    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'r') as f:
        raw_dataset = json.load(f)

    dataset = [
        sample for sample in raw_dataset
        if sample.get("perturbation_type") == args.attack
        and len(sample.get("new_tokens", [])) >= 1
    ]
    print(f"Filtered dataset to {len(dataset)} valid samples.")

    # Optional parallel slice [slice_start, slice_end); slice_end is clamped to the
    # filtered dataset size.
    slice_suffix = ""
    if (args.slice_start is None) != (args.slice_end is None):
        raise ValueError("--slice_start and --slice_end must be provided together.")
    if args.slice_start is not None:
        if not (0 <= args.slice_start < len(dataset)):
            raise ValueError(f"slice_start {args.slice_start} is out of bounds for a dataset "
                             f"of size {len(dataset)}.")
        if args.slice_end <= args.slice_start:
            raise ValueError(f"slice_end ({args.slice_end}) must be greater than slice_start "
                             f"({args.slice_start}).")
        effective_end = min(args.slice_end, len(dataset))
        dataset = dataset[args.slice_start:effective_end]
        slice_suffix = f"_slice{args.slice_start}-{effective_end}"
        print(f"Parallel worker: processing slice [{args.slice_start}:{effective_end}] "
              f"({len(dataset)} samples).")

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    if is_prometheus_model(args.model_name):
        print(f"Detected Prometheus-2-family model. Using few-shot direct-rating prompt "
              f"with {len(DEFAULT_FEW_SHOT_EXAMPLES)} in-context examples.")

    num_layers = model.config.num_hidden_layers
    window_size = 1 if args.target_pos == "last" else args.window_size

    dataset_effects = np.zeros((num_layers, window_size))
    valid_perturbations_count = 0
    cache = ActivationCache()

    if args.module_type == "mlp":
        target_modules = {i: f"model.layers.{i}.mlp" for i in range(num_layers)}
        plot_title_prefix = "MLP"
    elif args.module_type == "attn":
        target_modules = {i: f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)}
        plot_title_prefix = "Attention (o_proj)"
    else:
        raise ValueError("module_type must be 'mlp' or 'attn'")

    named_modules = dict(model.named_modules())
    target_module_objs = {i: named_modules[target_modules[i]] for i in range(num_layers)}

    print(f"Starting dataset sweep for {plot_title_prefix} modules (Mode: {args.target_pos})...")

    sample_logs = []
    n_skipped_argmax_match = 0
    n_skipped_non_rating = 0
    n_skipped_score_diff = 0
    n_skipped_seq_len = 0
    n_skipped_no_anchors = 0

    for sample_idx, sample in enumerate(tqdm(dataset)):
        clean_prompt = format_prompt(sample["document"], sample["summary"], prompt_criteria, args.model_name, tokenizer)
        corrupt_prompt = format_prompt(sample["document"], sample["perturbed_summary"], prompt_criteria, args.model_name, tokenizer)

        inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
        inputs_corr = tokenizer(corrupt_prompt, return_tensors="pt").to(model.device)

        # 1. Baseline outputs
        with torch.no_grad():
            out_clean = model(**inputs_clean)
            out_corr = model(**inputs_corr)

        target_id = torch.argmax(out_clean.logits[0, -1, :]).item()
        contra_id = torch.argmax(out_corr.logits[0, -1, :]).item()
        target_token_str = tokenizer.decode(target_id)
        contra_token_str = tokenizer.decode(contra_id)

        # The logit difference is defined between the two argmax tokens, so the
        # perturbation must flip the predicted rating.
        if target_id == contra_id:
            n_skipped_argmax_match += 1
            continue

        # Both next-token predictions must be rating digits.
        if not (is_rating_token(target_id, tokenizer) and is_rating_token(contra_id, tokenizer)):
            n_skipped_non_rating += 1
            continue

        score_clean = calculate_score(out_clean.logits, target_id, contra_id)
        score_corr = calculate_score(out_corr.logits, target_id, contra_id)

        if (score_clean - score_corr) < 0.05:
            n_skipped_score_diff += 1
            continue

        # 2. Anchor indices
        if args.target_pos == "last":
            # -1 indexing patches corrupt's last position with clean's last position,
            # so clean and corrupt do not need matched lengths.
            anchor_indices = [-1]
        else:
            clean_ids = inputs_clean.input_ids[0].cpu().tolist()
            corr_ids = inputs_corr.input_ids[0].cpu().tolist()

            if len(clean_ids) != len(corr_ids):
                n_skipped_seq_len += 1
                continue

            raw_anchor_indices = get_all_perturbation_indices(clean_ids, corr_ids)
            if not raw_anchor_indices:
                n_skipped_no_anchors += 1
                continue

            expected_tokens = [t.strip().lower() for t in sample.get("new_tokens", [])]
            anchor_indices = []
            for idx in raw_anchor_indices:
                raw_token_str = tokenizer.decode(corr_ids[idx])
                clean_token_str = raw_token_str.strip(' \n"\'.?!').lower()
                if clean_token_str in expected_tokens:
                    anchor_indices.append(idx)

        if not anchor_indices:
            n_skipped_no_anchors += 1
            continue

        # 3. Cache clean activations
        cache.clear()
        save_hooks = []
        for layer_idx in range(num_layers):
            mod_name = target_modules[layer_idx]
            mod = target_module_objs[layer_idx]
            save_hooks.append(mod.register_forward_hook(cache.make_save_hook(mod_name)))
        with torch.no_grad():
            model(**inputs_clean)
        for h in save_hooks:
            h.remove()

        # 4. Patching sweep
        for anchor_idx in anchor_indices:
            if anchor_idx + window_size > inputs_corr.input_ids.shape[1]:
                continue

            sample_effects = np.zeros((num_layers, window_size))

            for layer in range(num_layers):
                mod_name = target_modules[layer]
                mod = target_module_objs[layer]

                for offset in range(window_size):
                    patch_pos = anchor_idx + offset
                    hook = mod.register_forward_hook(make_patch_hook(mod_name, patch_pos, cache))
                    try:
                        with torch.no_grad():
                            patched_out = model(**inputs_corr)
                        score_patched = calculate_score(patched_out.logits, target_id, contra_id)
                        effect = (score_patched - score_corr) / (score_clean - score_corr)
                        sample_effects[layer, offset] = effect

                        sample_logs.append({
                            "sample_idx": sample_idx,
                            "clean_token": target_token_str,
                            "corrupt_token": contra_token_str,
                            "module_name": mod_name,
                            "layer": layer,
                            "offset": offset,
                            "clean_score": score_clean,
                            "corrupt_score": score_corr,
                            "patched_score": score_patched,
                            "normalized_effect": effect,
                        })
                    finally:
                        hook.remove()

            dataset_effects += sample_effects
            valid_perturbations_count += 1

    # ==========================================
    # 5. DIAGNOSTICS, VISUALIZATION & SAVING
    # ==========================================

    print("\n=== Sweep diagnostics ===")
    print(f"  Total samples processed:              {len(dataset)}")
    print(f"  Skipped (clean argmax == corrupt):    {n_skipped_argmax_match}")
    print(f"  Skipped (next-token not a rating):    {n_skipped_non_rating}")
    print(f"  Skipped (score diff < 0.05):          {n_skipped_score_diff}")
    if args.target_pos == "window":
        print(f"  Skipped (seq-len mismatch):           {n_skipped_seq_len}")
    print(f"  Skipped (no valid anchor tokens):     {n_skipped_no_anchors}")
    print(f"  Valid perturbation instances:         {valid_perturbations_count}")

    if valid_perturbations_count == 0:
        print("\nNo valid samples passed the criteria/thresholds.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    avg_effects = dataset_effects / valid_perturbations_count
    print(f"\nSweep complete. Averaged over {valid_perturbations_count} perturbation instances.")

    prompt_tag = "_completion" if is_base_llama_model(args.model_name) else ""
    base_filename = (
        f"{args.model_name.replace('/', '_')}_{args.dataset_name}_aggregated_{args.attack}"
        f"_int{args.intensity}_{args.samples}_{args.seed}"
        f"_{args.target_pos}_{args.module_type}{prompt_tag}_sweep{slice_suffix}"
    )

    if args.target_pos == "last":
        x_labels = ["Last Token"]
    else:
        x_labels = [f"Token +{i}" for i in range(window_size)]
        x_labels[0] = "Perturbed Token (+0)"
    y_labels = [f"Layer {i}" for i in range(num_layers)]

    csv_filename = os.path.join(args.results_dir, f"{base_filename}.csv")
    pd.DataFrame(avg_effects, index=y_labels, columns=x_labels).to_csv(csv_filename)
    print(f"Saved raw data to '{csv_filename}'")

    log_csv_filename = os.path.join(args.results_dir, f"{base_filename}_logs.csv")
    pd.DataFrame(sample_logs).to_csv(log_csv_filename, index=False)
    print(f"Saved per-sample log to '{log_csv_filename}'")

    if not slice_suffix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_effects, cmap="RdBu", center=0,
                    xticklabels=x_labels, yticklabels=range(num_layers),
                    annot=True, fmt=".2f", cbar_kws={'label': 'Normalized Causal Effect'})
        plt.title(f"Aggregated {plot_title_prefix} Causal Tracing\n(Task: {args.criteria}, Mode: {args.target_pos}, n={valid_perturbations_count})")
        plt.ylabel("Layer")
        plt.xlabel("Relative Token Position")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        png_filename = os.path.join(args.plots_dir, f"{base_filename}.png")
        plt.savefig(png_filename, dpi=300)
        print(f"Saved heatmap to '{png_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal tracing on LLM evaluators (logit-difference metric).")

    parser.add_argument("--model_name", type=str, default="PKU-ONELab/Themis", help="HuggingFace model ID")
    parser.add_argument("--dataset_name", type=str, required=True, help="cnn_dailymail or xsum (used in file names)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a perturbation JSON file")
    parser.add_argument("--criteria_file", type=str, default="configs/aspect_criteria.json", help="Path to criteria JSON")

    parser.add_argument("--module_type", type=str, choices=["mlp", "attn"], required=True)
    parser.add_argument("--target_pos", type=str, choices=["window", "last"], required=True)
    parser.add_argument("--window_size", type=int, default=6)

    parser.add_argument("--criteria", type=str, required=True,
                        help="Key in --criteria_file whose 'Gold' definition fills {prompt_criteria}")
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack key, e.g. read_preposition or adeq_entity_1to1 (see token_perturbations/prompts.py)")
    parser.add_argument("--intensity", type=str, default="1")
    parser.add_argument("--samples", type=str, default="200")
    parser.add_argument("--seed", type=str, default="42")

    parser.add_argument("--gpu", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES string, e.g. '0,1'. The model is sharded with device_map='auto'.")
    parser.add_argument("--results_dir", type=str, default="results/logit_diff",
                        help="Kept separate from the expected-rating results: both scripts use the same filenames")
    parser.add_argument("--plots_dir", type=str, default="plots/logit_diff")
    parser.add_argument("--slice_start", type=int, default=None,
                        help="Inclusive start index of this worker's slice (use with --slice_end)")
    parser.add_argument("--slice_end", type=int, default=None,
                        help="Exclusive end index of this worker's slice (clamped to the filtered size)")

    args = parser.parse_args()
    run_experiment(args)