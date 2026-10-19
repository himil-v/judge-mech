"""
Logit-lens trajectories of the rating tokens at the final input position.

For each retained clean/corrupt pair, the corrupt prompt is run once with
output_hidden_states=True. At every entry of hidden_states (the embedding output followed
by the output of each of the num_layers decoder layers, i.e. num_layers + 1 points), the
residual stream at the final position is passed through the model's final norm
(model.model.norm) and unembedding (model.lm_head), a softmax is taken over the full
vocabulary, and the probabilities of the five rating tokens "1".."5" are recorded.
Note that the last entry of hidden_states is already normalized by the model; the final
norm is applied to every entry, including that one.

Sample filters, applied in this order:
    1. structural: keep records whose perturbation_type matches --attack and whose
       new_tokens is non-empty
    2. non-rating: both clean and corrupt next-token argmax must be rating tokens
    3. behavioral (--filter, --threshold):
         logit_diff        clean and corrupt argmax must differ, and the logit difference
                           z[t_clean] - z[t_corrupt] must shrink by at least --threshold
         expected_rating   |E_clean - E_corrupt| >= --threshold, where E is the expected
                           rating over the five rating tokens (renormalized)

Supported evaluators:
    PKU-ONELab/Themis                    single-instruction completion prompt
    meta-llama/Meta-Llama-3-8B           base model, same prompt as Themis
    prometheus-eval/prometheus-7b-v2.0   multi-turn few-shot prompt in the Mistral chat
                                         template (five example turns, "Rating: " appended)

Outputs (in --results_dir / --plots_dir):
    logit_lens_1to5_*.csv        mean probability of each rating token per layer
    logit_lens_1to5_*_logs.csv   one row per (sample, layer) with the five probabilities
    logit_lens_1to5_*.png        trajectory plot
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

RATINGS = ["1", "2", "3", "4", "5"]
RATING_LABELS = [f'Rating "{r}"' for r in RATINGS]

DIRECT_RATING_SUFFIX = "Rating: "

# Prometheus few-shot examples: ratings span 1-5 in a non-monotonic order (4, 2, 5, 1, 3).
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "article": (
            "The Federal Reserve lowered interest rates by 0.25 percentage points "
            "on Wednesday, citing easing inflation and a slowing labor market. "
            "Officials signaled at least one additional cut before year-end but "
            "flagged caution due to persistent housing-cost pressures."
        ),
        "summary": (
            "The Federal Reserve cut interest rates by 0.25 percentage points on "
            "Wednesday, signaling further cuts ahead while staying cautious about "
            "housing costs."
        ),
        "rating": "4",
    },
    {
        "article": (
            "The European Union and Vietnam signed a free trade agreement on Friday, "
            "eliminating tariffs on 99 percent of goods over the next decade. The deal "
            "is expected to boost Vietnamese textile and electronics exports while "
            "opening Vietnam to European pharmaceuticals."
        ),
        "summary": (
            "Brazil and Vietnam signed a trade deal that increases tariffs on "
            "agricultural products."
        ),
        "rating": "2",
    },
    {
        "article": (
            "Stanford researchers announced a new lithium-metal battery design that "
            "retains 85 percent of its capacity after 500 charge cycles, doubling "
            "the lifespan of conventional electric-vehicle batteries. The team used "
            "a solid-state electrolyte to prevent dendrite formation."
        ),
        "summary": (
            "Stanford researchers developed a lithium-metal battery retaining 85 "
            "percent capacity after 500 cycles, doubling EV battery lifespan, by "
            "using a solid-state electrolyte that prevents dendrite formation."
        ),
        "rating": "5",
    },
    {
        "article": (
            "NASA's Perseverance rover collected its 24th rock sample on Tuesday "
            "inside Jezero Crater, advancing the search for signs of ancient Martian "
            "life. Mission scientists noted the sample contains evidence of past "
            "hydrothermal activity."
        ),
        "summary": (
            "China's lunar rover discovered liquid water on the moon during a "
            "mission earlier this week."
        ),
        "rating": "1",
    },
    {
        "article": (
            "A magnitude 6.3 earthquake struck off the coast of Hokkaido, Japan, "
            "early Saturday morning. No tsunami warning was issued. Authorities "
            "reported minor structural damage to buildings in several coastal towns "
            "but no immediate casualties."
        ),
        "summary": (
            "An earthquake hit Japan on Saturday. Some damage was reported, but no "
            "major casualties."
        ),
        "rating": "3",
    },
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def is_prometheus_model(model_name: str) -> bool:
    return "prometheus" in model_name.lower()


def is_base_llama_model(model_name: str) -> bool:
    """Un-fine-tuned base Llama-3-8B (not -Instruct, not Themis)."""
    n = model_name.lower()
    return (("meta-llama-3-8b" in n or "meta-llama/meta-llama-3-8b" in n)
            and "instruct" not in n and "themis" not in n and "pku" not in n)


def format_prompt_prometheus_fewshot(document, summary, prompt_criteria, tokenizer,
                                     examples=DEFAULT_FEW_SHOT_EXAMPLES):
    """Multi-turn few-shot prompt in the Mistral chat template.

    - The first user turn holds the instructions, the criterion and the first example
      (Mistral has no system role).
    - Each example is a (user, assistant="Rating: N") turn pair.
    - The sample to rate is the final user turn, and DIRECT_RATING_SUFFIX is appended
      after [/INST].
    """
    system_instructions = (
        "Please act as an impartial and helpful evaluator for natural language generation (NLG).\n"
        "Your task is to evaluate the quality of the Summarization strictly based on the given evaluation criterion.\n"
        "You MUST keep to the strict boundaries of the evaluation criterion.\n"
        "**Scoring Rules:**\n"
        "1. Use a Likert scale from 1 to 5.\n"
        "2. 5 is the highest quality, 1 is the lowest.\n"
        "3. Output ONLY the number."
    )

    first = examples[0]
    messages = [
        {"role": "user", "content": (
            f"{system_instructions}\n\n"
            f"###Evaluation Criterion###\n{prompt_criteria}\n"
            f"###Task###\nArticle:\n{first['article']}\nSummary:\n{first['summary']}"
        )},
        {"role": "assistant", "content": f"Rating: {first['rating']}"},
    ]
    for ex in examples[1:]:
        messages.append({"role": "user",
                         "content": f"###Task###\nArticle:\n{ex['article']}\nSummary:\n{ex['summary']}"})
        messages.append({"role": "assistant", "content": f"Rating: {ex['rating']}"})
    messages.append({"role": "user",
                     "content": f"###Task###\nArticle:\n{document}\nSummary:\n{summary}"})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + DIRECT_RATING_SUFFIX


def format_prompt(document, summary, prompt_criteria, model_name, tokenizer=None):
    if is_prometheus_model(model_name):
        return format_prompt_prometheus_fewshot(document, summary, prompt_criteria, tokenizer)
    # Themis and the base Llama-3-8B model.
    return PROMPT_TEMPLATE.format(prompt_criteria=prompt_criteria, article=document, summary=summary)


def calculate_score(logits, target_id, contra_id):
    """Logit difference z[target] - z[contra] at the final position."""
    final_logits = logits[0, -1, :]
    return (final_logits[target_id] - final_logits[contra_id]).item()


def _first_emitted_token_id(tokenizer, s: str) -> int:
    """First token id the model would emit for `s`. Mistral's SentencePiece tokenizer
    prepends a '▁' marker (e.g. encode("5") -> [28705, 28782]); that marker is skipped
    and the digit token returned. Single-token encodings (Llama-3) are returned as is."""
    ids = tokenizer.encode(s, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer produced no tokens for {s!r}")
    if len(ids) == 1:
        return ids[0]
    for tid in ids:
        if tokenizer.decode([tid]).strip() == s:
            return tid
    return ids[-1]


def expected_rating_from_logits(logits, rating_ids):
    """E[r] = sum_r r * P(r): softmax over the full vocabulary at the last position,
    restricted to the five rating ids and renormalized within that set."""
    final_logits = logits[0, -1, :].float()
    probs = torch.nn.functional.softmax(final_logits, dim=-1)
    rating_probs = probs[rating_ids]
    rating_probs = rating_probs / rating_probs.sum()
    rating_probs_np = rating_probs.detach().cpu().numpy()
    expected = float(np.dot(np.arange(1, 6, dtype=np.float64), rating_probs_np))
    return expected, rating_probs_np

# ==========================================
# 3. MAIN EXPERIMENT PIPELINE
# ==========================================

def run_experiment(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.criteria_file, "r") as f:
        criteria_def = json.load(f)
    prompt_criteria = criteria_def[args.criteria]["Gold"]

    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        raw_dataset = json.load(f)
    dataset = [
        sample for sample in raw_dataset
        if sample.get("perturbation_type") == args.attack
        and len(sample.get("new_tokens", [])) >= 1
    ]
    print(f"Filtered dataset to {len(dataset)} valid samples.")

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    num_layers = model.config.num_hidden_layers

    rating_ids = [_first_emitted_token_id(tokenizer, r) for r in RATINGS]
    rating_id_set = set(rating_ids)
    if len(rating_id_set) != len(RATINGS):
        raise RuntimeError(f"Rating ids collapsed to {len(rating_id_set)} unique ids: {rating_ids}. "
                           f"Check _first_emitted_token_id for this tokenizer.")
    print(f"Rating token ids (1..5): {rating_ids}")

    using_base = is_base_llama_model(args.model_name)
    use_expected_rating_filter = (args.filter == "expected_rating")

    avg_trajectories = np.zeros((num_layers + 1, len(RATINGS)))
    valid_samples_count = 0
    sample_logs = []
    n_skipped_non_rating = 0
    n_skipped_argmax_eq = 0
    n_skipped_score_diff = 0
    n_skipped_expected_diff = 0

    for sample_idx, sample in enumerate(tqdm(dataset)):
        clean_prompt = format_prompt(sample["document"], sample["summary"], prompt_criteria, args.model_name, tokenizer)
        corrupt_prompt = format_prompt(sample["document"], sample["perturbed_summary"], prompt_criteria, args.model_name, tokenizer)

        inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
        inputs_corr = tokenizer(corrupt_prompt, return_tensors="pt").to(model.device)
        corr_len = inputs_corr.input_ids.shape[1]

        with torch.no_grad():
            clean_outputs = model(**inputs_clean)
            clean_rating_id = clean_outputs.logits[0, -1, :].argmax().item()
            outputs = model(**inputs_corr, output_hidden_states=True)
            corrupt_rating_id = outputs.logits[0, -1, :].argmax().item()

        # Both next-token predictions must be rating tokens.
        if not (clean_rating_id in rating_id_set and corrupt_rating_id in rating_id_set):
            n_skipped_non_rating += 1
            continue

        if use_expected_rating_filter:
            score_clean, _ = expected_rating_from_logits(clean_outputs.logits, rating_ids)
            score_corr, _ = expected_rating_from_logits(outputs.logits, rating_ids)
            if abs(score_clean - score_corr) < args.threshold:
                n_skipped_expected_diff += 1
                continue
        else:
            if clean_rating_id == corrupt_rating_id:
                n_skipped_argmax_eq += 1
                continue
            score_clean = calculate_score(clean_outputs.logits, clean_rating_id, corrupt_rating_id)
            score_corr = calculate_score(outputs.logits, clean_rating_id, corrupt_rating_id)
            if (score_clean - score_corr) < args.threshold:
                n_skipped_score_diff += 1
                continue

        anchor_idx = corr_len - 1
        sample_trajectory = np.zeros((num_layers + 1, len(RATINGS)))

        # Final norm and lm_head are called with explicit device routing so this also
        # works when the model is sharded across GPUs with device_map="auto".
        norm_device = model.model.norm.weight.device
        head_device = model.lm_head.weight.device
        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            residual = hidden_state[0, anchor_idx, :]
            with torch.no_grad():
                hidden_normed = model.model.norm(residual.to(norm_device))
                logits = model.lm_head(hidden_normed.to(head_device))
                probs = torch.nn.functional.softmax(logits, dim=-1)

            layer_log = {
                "sample_idx": sample_idx,
                "layer": layer_idx,
                "clean_prediction": tokenizer.decode(clean_rating_id),
                "corrupt_prediction": tokenizer.decode(corrupt_rating_id),
                "score_clean": score_clean,
                "score_corr": score_corr,
            }
            for i, r_id in enumerate(rating_ids):
                p = float(probs[r_id].item())
                sample_trajectory[layer_idx, i] = p
                layer_log[f"prob_{RATING_LABELS[i]}"] = p
            sample_logs.append(layer_log)

        avg_trajectories += sample_trajectory
        valid_samples_count += 1

    # ==========================================
    # 4. SAVING & VISUALIZATION
    # ==========================================
    total_seen = (valid_samples_count + n_skipped_non_rating + n_skipped_argmax_eq
                  + n_skipped_score_diff + n_skipped_expected_diff)
    print("\n=== Skip summary ===")
    print(f"  Samples seen             : {total_seen}")
    print(f"  Passed                   : {valid_samples_count}")
    print(f"  Skipped (non-rating)     : {n_skipped_non_rating}")
    if use_expected_rating_filter:
        print(f"  Skipped (|E_c-E_r| < {args.threshold}) : {n_skipped_expected_diff}")
    else:
        print(f"  Skipped (argmax equal)   : {n_skipped_argmax_eq}")
        print(f"  Skipped (score diff < {args.threshold}) : {n_skipped_score_diff}")

    if valid_samples_count == 0:
        print("\nNo valid samples passed the criteria/thresholds.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    avg_trajectories /= valid_samples_count
    print(f"\nExtraction complete. Averaged over {valid_samples_count} samples.")

    prompt_tag = "_completion" if using_base else ""
    base_filename = (f"logit_lens_1to5_{args.model_name.replace('/', '_')}_{args.dataset_name}"
                     f"_{args.attack}_int{args.intensity}_{args.samples}_{args.seed}{prompt_tag}")

    layers = np.arange(num_layers + 1)
    csv_filename = os.path.join(args.results_dir, f"{base_filename}.csv")
    pd.DataFrame(avg_trajectories, columns=[f"Prob_{label}" for label in RATING_LABELS],
                 index=layers).to_csv(csv_filename)
    print(f"Saved average trajectory to '{csv_filename}'")

    log_csv_filename = os.path.join(args.results_dir, f"{base_filename}_logs.csv")
    pd.DataFrame(sample_logs).to_csv(log_csv_filename, index=False)
    print(f"Saved per-sample log to '{log_csv_filename}'")

    plt.figure(figsize=(12, 6))
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]
    markers = ["o", "v", "s", "P", "*"]
    for i, label in enumerate(RATING_LABELS):
        plt.plot(layers, avg_trajectories[:, i], label=label,
                 color=colors[i], linewidth=2.5, marker=markers[i], alpha=0.8)
    plt.title(f"Rating-token trajectory at the final position (logit lens)\n"
              f"(Task: {args.criteria}, Valid Samples: {valid_samples_count})")
    plt.xlabel("Layer Depth")
    plt.ylabel("Token Probability (logit lens)")
    plt.legend(title="Target Ratings", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    plt.tight_layout()
    plot_filename = os.path.join(args.plots_dir, f"{base_filename}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved trajectory plot to '{plot_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit-lens rating-token trajectories for LLM evaluators.")
    parser.add_argument("--model_name", type=str, default="PKU-ONELab/Themis", help="HuggingFace model ID")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a perturbation JSON file")
    parser.add_argument("--dataset_name", type=str, required=True, help="cnn_dailymail or xsum (used in file names)")
    parser.add_argument("--criteria_file", type=str, default="configs/aspect_criteria.json")
    parser.add_argument("--criteria", type=str, required=True,
                        help="Key in --criteria_file whose 'Gold' definition fills {prompt_criteria}")
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack key, e.g. read_preposition or adeq_entity_1to1 (see token_perturbations/prompts.py)")
    parser.add_argument("--filter", type=str, choices=["logit_diff", "expected_rating"], required=True,
                        help="Behavioral filter applied after the rating-token check")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Minimum score change for the behavioral filter")
    parser.add_argument("--intensity", type=str, default="1")
    parser.add_argument("--samples", type=str, default="200")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES string, e.g. '0,1'. The model is sharded with device_map='auto'.")
    parser.add_argument("--results_dir", type=str, default="results/logit_lens")
    parser.add_argument("--plots_dir", type=str, default="plots/logit_lens")
    args = parser.parse_args()
    run_experiment(args)