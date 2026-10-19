"""
Attention-head knockout for LLM judges.

For every (layer, head) pair, the head's contribution is zeroed in the input of that
layer's attention output projection (o_proj), i.e. the slice
[head * head_dim, (head + 1) * head_dim] of the concatenated head outputs, during the
forward pass on the corrupt prompt. The normalized effect at the final position is

    effect = (X_ablated - X_corrupt) / (X_clean - X_corrupt)

where X = z[t_clean] - z[t_corrupt] is the logit difference between the clean run's
argmax token and the corrupt run's argmax token. Positive effects mean the ablation moves
the output back toward the clean rating; negative effects mean it pushes further toward
the corrupt rating.

Ablation modes (--ablation_mode):
    global    zero the head at every token position
    targeted  zero the head only at the final position

Sample filters, applied in this order (pass 1):
    1. structural: keep records whose perturbation_type matches --attack and whose
       new_tokens is non-empty
    2. rating flip: the clean and corrupt argmax tokens must differ
    3. score difference: X_clean - X_corrupt >= --threshold
    4. non-rating: both argmax tokens must be rating digits

Inputs are batched with left padding, so position -1 is the last real token of every
prompt in a batch.

Supported evaluators:
    PKU-ONELab/Themis                    single-instruction completion prompt
    meta-llama/Meta-Llama-3-8B           base model, same prompt as Themis
    prometheus-eval/prometheus-7b-v2.0   multi-turn few-shot prompt in the Mistral chat
                                         template (five example turns, "Rating: " appended)

Outputs (in --results_dir / --plots_dir):
    head_knockout_<mode>_*.csv        mean effect per (layer, head)
    head_knockout_<mode>_*_logs.csv   one row per (sample, layer, head)
    head_knockout_<mode>_*.png        heatmap (red = positive, blue = negative)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

DIRECT_RATING_SUFFIX = "Rating: "
RATING_TOKEN_STRINGS = {"1", "2", "3", "4", "5"}

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


def get_knockout_hook(target_head: int, head_dim: int, mode: str, target_pos: int = -1):
    """Forward pre-hook on o_proj that zeroes one head's slice of its input
    ([batch, seq, num_heads * head_dim]), at all positions ('global') or only at
    target_pos ('targeted')."""
    def knockout_hook(module, args):
        hidden_states = args[0].clone()
        start_dim = target_head * head_dim
        end_dim = (target_head + 1) * head_dim
        if mode == "global":
            hidden_states[..., start_dim:end_dim] = 0.0
        elif mode == "targeted":
            hidden_states[:, target_pos, start_dim:end_dim] = 0.0
        return (hidden_states,)
    return knockout_hook

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

    # Optional slice [sample_start, sample_end) for running several processes in parallel.
    sample_end = args.sample_end if args.sample_end is not None else len(dataset)
    dataset = dataset[args.sample_start:sample_end]
    print(f"Processing samples [{args.sample_start}, {sample_end}) -> {len(dataset)} samples.")

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    # Left padding keeps position -1 as the last real token for every prompt in a batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    print(f"Model: {num_layers} layers x {num_heads} heads x {head_dim} head_dim.")

    # The last token of each digit's encoding is the digit itself (this skips the
    # leading '▁' marker that Mistral's tokenizer adds).
    rating_id_set = {tokenizer.encode(s, add_special_tokens=False)[-1] for s in RATING_TOKEN_STRINGS}
    print(f"Rating token ids: {sorted(rating_id_set)}")

    oproj_modules = {
        layer: model.get_submodule(f"model.layers.{layer}.self_attn.o_proj")
        for layer in range(num_layers)
    }

    # ==========================================
    # PASS 1: BASELINES AND FILTERING
    # ==========================================
    n_skipped_argmax_eq = 0
    n_skipped_score_diff = 0
    n_skipped_non_rating = 0
    valid_samples = []

    for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc="Baselines"):
        batch = dataset[batch_start:batch_start + args.batch_size]
        clean_prompts = [format_prompt(s["document"], s["summary"], prompt_criteria, args.model_name, tokenizer)
                         for s in batch]
        corrupt_prompts = [format_prompt(s["document"], s["perturbed_summary"], prompt_criteria, args.model_name, tokenizer)
                           for s in batch]

        inputs_clean = tokenizer(clean_prompts, return_tensors="pt", padding=True).to(model.device)
        inputs_corr = tokenizer(corrupt_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out_clean = model(**inputs_clean)
            out_corr = model(**inputs_corr)

        for i in range(len(batch)):
            clean_logits = out_clean.logits[i, -1, :]
            corr_logits = out_corr.logits[i, -1, :]
            target_id = clean_logits.argmax().item()
            contra_id = corr_logits.argmax().item()

            if target_id == contra_id:
                n_skipped_argmax_eq += 1
                continue

            score_clean = (clean_logits[target_id] - clean_logits[contra_id]).item()
            score_corr = (corr_logits[target_id] - corr_logits[contra_id]).item()
            if (score_clean - score_corr) < args.threshold:
                n_skipped_score_diff += 1
                continue

            if not (target_id in rating_id_set and contra_id in rating_id_set):
                n_skipped_non_rating += 1
                continue

            valid_samples.append({
                "global_idx": args.sample_start + batch_start + i,
                "target_id": target_id,
                "contra_id": contra_id,
                "target_token": tokenizer.decode(target_id),
                "contra_token": tokenizer.decode(contra_id),
                "score_clean": score_clean,
                "score_corr": score_corr,
                "corrupt_prompt": corrupt_prompts[i],
            })

    print("\n=== Pass 1 skip summary ===")
    print(f"  Samples seen             : {len(dataset)}")
    print(f"  Passed                   : {len(valid_samples)}")
    print(f"  Skipped (argmax equal)   : {n_skipped_argmax_eq}")
    print(f"  Skipped (score diff < {args.threshold}) : {n_skipped_score_diff}")
    print(f"  Skipped (non-rating)     : {n_skipped_non_rating}")
    if not valid_samples:
        print("No valid samples; nothing to ablate.")
        return

    # Tokenize the retained corrupt prompts once into fixed batches.
    valid_batches = []
    for i in range(0, len(valid_samples), args.batch_size):
        meta = valid_samples[i:i + args.batch_size]
        inputs = tokenizer([m["corrupt_prompt"] for m in meta], return_tensors="pt", padding=True).to(model.device)
        valid_batches.append({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "metadata": meta})

    # ==========================================
    # PASS 2: ABLATION LOOP
    # ==========================================
    sum_heatmap = np.zeros((num_layers, num_heads))
    sample_logs = []
    ablation_pairs = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    print(f"Pass 2: {len(ablation_pairs)} ablations x {len(valid_batches)} batches.")

    for target_layer, target_head in tqdm(ablation_pairs, desc="Ablations"):
        hook_fn = get_knockout_hook(target_head, head_dim, args.ablation_mode, target_pos=-1)
        handle = oproj_modules[target_layer].register_forward_pre_hook(hook_fn)
        try:
            for vb in valid_batches:
                with torch.no_grad():
                    out = model(input_ids=vb["input_ids"], attention_mask=vb["attention_mask"])
                for i, meta in enumerate(vb["metadata"]):
                    logits_i = out.logits[i, -1, :]
                    score_ablated = (logits_i[meta["target_id"]] - logits_i[meta["contra_id"]]).item()
                    denom = meta["score_clean"] - meta["score_corr"] + 1e-8
                    effect = (score_ablated - meta["score_corr"]) / denom
                    sum_heatmap[target_layer, target_head] += effect
                    sample_logs.append({
                        "sample_idx": meta["global_idx"],
                        "layer": target_layer,
                        "head": target_head,
                        "normalized_effect": effect,
                        "score_ablated": score_ablated,
                        "score_corr": meta["score_corr"],
                        "score_clean": meta["score_clean"],
                        "target_token": meta["target_token"],
                        "contra_token": meta["contra_token"],
                    })
        finally:
            handle.remove()

    # ==========================================
    # 4. SAVING & VISUALIZATION
    # ==========================================
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    n_valid = len(valid_samples)
    ablation_heatmap = sum_heatmap / n_valid
    print(f"\nKnockout complete. Averaged over {n_valid} samples.")

    slice_tag = (f"_slice{args.sample_start}-{sample_end}"
                 if (args.sample_start != 0 or args.sample_end is not None) else "")
    base_filename = (f"head_knockout_{args.ablation_mode}_{args.model_name.replace('/', '_')}_{args.dataset_name}"
                     f"_{args.attack}_int{args.intensity}_{args.samples}_{args.seed}{slice_tag}")

    csv_filename = os.path.join(args.results_dir, f"{base_filename}.csv")
    pd.DataFrame(ablation_heatmap, columns=[f"Head_{h}" for h in range(num_heads)],
                 index=range(num_layers)).to_csv(csv_filename)
    print(f"Saved mean effects to '{csv_filename}'")

    log_csv_filename = os.path.join(args.results_dir, f"{base_filename}_logs.csv")
    pd.DataFrame(sample_logs).to_csv(log_csv_filename, index=False)
    print(f"Saved per-sample log to '{log_csv_filename}'")

    vmax = float(np.max(np.abs(ablation_heatmap))) or 1.0
    plt.figure(figsize=(20, 10))
    sns.heatmap(ablation_heatmap, cmap="RdBu_r", center=0, vmax=vmax, vmin=-vmax,
                cbar_kws={"label": "Normalized Effect\n(Red = Disruptive, Blue = Suppressive)"})
    plt.title(f"Attention Head Knockout ({args.ablation_mode.capitalize()} Ablation)\n"
              f"(Task: {args.criteria}, Valid Samples: {n_valid})")
    plt.xlabel("Attention Head Index")
    plt.ylabel("Layer Depth")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_filename = os.path.join(args.plots_dir, f"{base_filename}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved heatmap to '{plot_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention-head knockout on LLM evaluators.")
    parser.add_argument("--model_name", type=str, default="PKU-ONELab/Themis", help="HuggingFace model ID")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a perturbation JSON file")
    parser.add_argument("--dataset_name", type=str, required=True, help="cnn_dailymail or xsum (used in file names)")
    parser.add_argument("--criteria_file", type=str, default="configs/aspect_criteria.json")
    parser.add_argument("--criteria", type=str, required=True,
                        help="Key in --criteria_file whose 'Gold' definition fills {prompt_criteria}")
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack key, e.g. read_preposition or adeq_entity_1to1 (see token_perturbations/prompts.py)")
    parser.add_argument("--ablation_mode", type=str, choices=["global", "targeted"], required=True,
                        help="'global' zeroes the head at all positions, 'targeted' only at the final position")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Minimum logit-difference change for a sample to be kept")
    parser.add_argument("--intensity", type=str, default="1")
    parser.add_argument("--samples", type=str, default="200")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for baseline and ablation forward passes (1 disables batching)")
    parser.add_argument("--sample_start", type=int, default=0,
                        help="Start index into the filtered dataset (for parallel runs)")
    parser.add_argument("--sample_end", type=int, default=None,
                        help="End index (exclusive) into the filtered dataset; default is the end")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES string, e.g. '0,1'. The model is sharded with device_map='auto'.")
    parser.add_argument("--results_dir", type=str, default="results/head_knockout")
    parser.add_argument("--plots_dir", type=str, default="plots/head_knockout")
    args = parser.parse_args()
    run_experiment(args)