"""
Activation patching (causal tracing) for LLM judges using the expected-rating metric.

Patches clean activations into the corrupt forward pass at (layer, position) sites, for
either MLP outputs (`model.layers.{i}.mlp`) or attention output projections
(`model.layers.{i}.self_attn.o_proj`), and reports the normalized causal effect

    effect = (X_patched - X_corrupt) / (X_clean - X_corrupt)

where X is the expected rating E[r] = sum_r r * P(r) over the five rating tokens,
renormalized within the rating set (G-Eval style). The same statistic is used in both modes.

Modes (--target_pos):
    window  patch a 6-token window starting at each perturbed (anchor) token, offsets +0..+5
            Length-mismatched clean/corrupt pairs are aligned with difflib:
            'equal' maps one-to-one, 'replace' is clamped to the last valid clean position,
            'insert' has no clean counterpart and is left as NaN.
    last    patch only the final input position at every layer.

Sample filters, applied in this order:
    1. structural: keep records whose perturbation_type matches --attack and whose
       new_tokens is non-empty (drops JSON/API failures and unmodified summaries)
    2. non-rating: both clean and corrupt next-token argmax must be a rating digit
    3. score difference: |E_clean - E_corrupt| >= 0.05
    4. anchor validity (window mode): perturbed positions found by difflib must decode to
       one of the generator's new_tokens

Supported evaluators:
    PKU-ONELab/Themis                    single-instruction completion prompt
    prometheus-eval/prometheus-7b-v2.0   five-shot direct-rating prompt in the Mistral chat template
    meta-llama/Meta-Llama-3-8B           base model, same prompt as Themis

Outputs (in --results_dir / --plots_dir):
    *_sweep.csv        per-cell mean effect (layers x positions)
    *_sweep_n.csv      per-cell sample counts (sliced runs only)
    *_sweep_logs.csv   one row per (instance, layer, offset) with clean/corrupt/patched
                       rating distributions
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


def make_patch_hook(module_name: str, target_pos: int, cache: ActivationCache,
                    source_pos: int = None):
    """Write clean's activation at `source_pos` into corrupt's activation at
    `target_pos` during the corrupt forward pass.

    When source_pos is None, defaults to target_pos -- same-index patching,
    correct for every length-matched perturbation (Entity / Tense / Spelling /
    Preposition / etc., and Reordering when clean and corrupt tokenize to the
    same length). For length-mismatched cases (Mistral retokenizing around the
    perturbation boundary), positions past the perturbation in corrupt no
    longer correspond to the same content in clean, so source_pos is shifted
    via compute_clean_source_pos.
    """
    if source_pos is None:
        source_pos = target_pos

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        tensor = output[0].clone() if is_tuple else output.clone()
        clean_act = cache.clean_cache[module_name].to(tensor.device)
        tensor[:, target_pos, :] = clean_act[:, source_pos, :]
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


def _first_emitted_token_id(tokenizer, s: str) -> int:
    """Returns the first token id that the model would emit for `s` after the
    prompt. For Mistral's SentencePiece tokenizer that prepends a '▁'
    leading-space marker to every encoded string (e.g. encode("5") -> [28705,
    28782]), we skip that marker and return the digit token. For Llama-3
    where the digit encodes as a single token, that token is returned
    directly."""
    ids = tokenizer.encode(s, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer produced no tokens for {s!r}")
    if len(ids) == 1:
        return ids[0]
    # Multi-token encoding: find the id whose own decode equals the target
    # (this skips the leading '▁' / 28705 marker in SentencePiece tokenizers).
    for tid in ids:
        if tokenizer.decode([tid]).strip() == s:
            return tid
    # Fallback for unexpected tokenizers: assume last id holds the content.
    return ids[-1]


def expected_rating_from_logits(logits, rating_ids):
    """Returns the rating-restricted expected value E[r] = Σ_r r · P(r | rating set).

    Follows the probability-weighted scoring scheme of G-Eval (Liu et al.,
    EMNLP 2023): softmax over the full vocab at the last position, restrict
    to the five rating-digit token IDs, renormalize within that set, and
    take the dot product with [1, 2, 3, 4, 5].

    Returns
    -------
    expected : float
        E[r] in [1.0, 5.0].
    rating_probs : np.ndarray, shape (5,)
        [P(1), P(2), P(3), P(4), P(5)] renormalized within the rating set.
        Returned alongside the scalar so per-sample logs can record the full
        distribution and any alternative score metric is derivable offline.
    """
    final_logits = logits[0, -1, :].float()
    probs = torch.softmax(final_logits, dim=-1)
    rating_probs = probs[rating_ids]
    rating_probs = rating_probs / rating_probs.sum()
    rating_probs_np = rating_probs.detach().cpu().numpy()
    expected = float(np.dot(np.arange(1, 6, dtype=np.float64), rating_probs_np))
    return expected, rating_probs_np


def compute_diff_opcodes(clean_ids, corr_ids):
    """Returns difflib opcodes describing how to transform `clean_ids` into
    `corr_ids`. Each opcode is a tuple (tag, i1, i2, j1, j2) where clean[i1:i2]
    aligns with corr[j1:j2]; tag is one of 'equal', 'replace', 'insert',
    'delete'. Used by compute_clean_source_pos to remap corrupt positions back
    to clean positions when the two sequences have different lengths."""
    matcher = difflib.SequenceMatcher(None, clean_ids, corr_ids)
    return matcher.get_opcodes()


def compute_clean_source_pos(target_pos, opcodes, clean_len):
    """Map a position in the corrupt sequence back to its corresponding
    position in the clean sequence, given the difflib opcodes between the two.

    The four opcode tags handled:
      - 'equal': clean[i1:i2] == corr[j1:j2] and they have the same length, so
        the mapping is direct: source = i1 + (target_pos - j1).
      - 'replace': clean[i1:i2] differs from corr[j1:j2] AND they may have
        different lengths. If corr's block is longer than clean's (the
        common Mistral retokenization case where 'USA' (1 token) becomes
        'Am','erica' (2 tokens)), we clamp the offset to the last valid
        clean position in the block, so all sub-token positions on the
        corrupt side get patched against the single clean source token.
      - 'insert': i1 == i2, no content on the clean side; no meaningful
        source position exists. Returns -1.
      - 'delete': j1 == j2, no content on the corrupt side, so no target_pos
        can fall within it.

    Returns -1 when no meaningful clean counterpart exists; the caller
    should leave the corresponding cell as NaN.
    """
    for tag, i1, i2, j1, j2 in opcodes:
        if j1 <= target_pos < j2:
            offset_in_block = target_pos - j1
            if tag == 'equal':
                clean_pos = i1 + offset_in_block
                # Bounds check (should always succeed for valid opcodes).
                if 0 <= clean_pos < clean_len:
                    return clean_pos
                return -1
            elif tag == 'replace':
                clean_block_len = i2 - i1
                if clean_block_len == 0:
                    # Degenerate replace with empty clean side (shouldn't
                    # occur in well-formed opcodes; insert covers that case).
                    return -1
                clean_pos = i1 + min(offset_in_block, clean_block_len - 1)
                if 0 <= clean_pos < clean_len:
                    return clean_pos
                return -1
            elif tag == 'insert':
                return -1
            # 'delete' can't contain any target_pos because j1 == j2
    return -1  # target_pos outside all opcodes (should not happen)


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

    # Optional parallel slice: each worker processes the half-open range
    # [slice_start, slice_end). Misconfigured arguments raise an error.
    slice_suffix = ""
    if (args.slice_start is None) != (args.slice_end is None):
        raise ValueError("--slice_start and --slice_end must be provided together.")
    if args.slice_start is not None:
        # slice_start must be a valid index; slice_end is CLAMPED to len(dataset)
        # rather than erroring. This lets a caller pass the raw pre-filter size
        # (e.g. --slice_end 200) even though the behavioral filter above may have
        # dropped a few empty-perturbation samples (e.g. 200 -> 198). The tail
        # slice then processes whatever remains instead of raising.
        if not (0 <= args.slice_start < len(dataset)):
            raise ValueError(
                f"slice_start {args.slice_start} is out of bounds for a dataset "
                f"of size {len(dataset)}."
            )
        if args.slice_end <= args.slice_start:
            raise ValueError(
                f"slice_end ({args.slice_end}) must be greater than slice_start "
                f"({args.slice_start})."
            )
        effective_end = min(args.slice_end, len(dataset))
        if effective_end != args.slice_end:
            print(f"NOTE: slice_end {args.slice_end} exceeds filtered dataset size "
                  f"{len(dataset)}; clamping to {effective_end}.")
        dataset = dataset[args.slice_start:effective_end]
        # Suffix reflects the EFFECTIVE (clamped) end so the filename matches the
        # samples actually processed, and so the glob-based merge finds it.
        slice_suffix = f"_slice{args.slice_start}-{effective_end}"
        print(f"Parallel worker: processing slice [{args.slice_start}:{effective_end}] "
              f"({len(dataset)} samples).")

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    is_prometheus = is_prometheus_model(args.model_name)
    if is_prometheus:
        print(f"Detected Prometheus-2-family model. Using few-shot direct-rating prompt "
              f"with {len(DEFAULT_FEW_SHOT_EXAMPLES)} in-context examples.")

    num_layers = model.config.num_hidden_layers

    if args.target_pos == "last":
        window_size = 1
    else:
        window_size = args.window_size

    # Per-cell accumulators (NaN-safe). With shift-correction enabled, some
    # cells in some samples may have no valid clean counterpart (when a
    # corrupt window offset falls inside a difflib 'insert' block) and end up
    # as NaN. We accumulate sum + count separately per cell so the cell-mean
    # only divides by the number of samples that actually contributed.
    dataset_effects_sum = np.zeros((num_layers, window_size))
    dataset_effects_count = np.zeros((num_layers, window_size), dtype=np.int64)
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

    # Cache module objects once (avoids repeated dict(model.named_modules()) calls inside loops)
    named_modules = dict(model.named_modules())
    target_module_objs = {i: named_modules[target_modules[i]] for i in range(num_layers)}

    print(f"Starting dataset sweep for {plot_title_prefix} modules (Mode: {args.target_pos})...")

    # Pre-compute the five rating-token IDs once. The _first_emitted_token_id
    # helper handles Mistral's SentencePiece '▁' marker; on Llama-3 the
    # single-token path triggers and behavior is identical.
    rating_ids = torch.tensor(
        [_first_emitted_token_id(tokenizer, str(r)) for r in range(1, 6)],
        device=model.device,
    )
    if len(set(rating_ids.tolist())) != 5:
        raise RuntimeError(
            f"Rating-token IDs collapsed to fewer than 5 distinct tokens: "
            f"{rating_ids.tolist()}. Tokenizer encoding of digits 1-5 is broken; "
            f"results would be meaningless. Check the tokenizer."
        )
    print(f"Rating token IDs (1..5): {rating_ids.tolist()}")

    sample_logs = []

    # Skip counters (printed at end of sweep). No argmax-flip filter is applied:
    # the expected rating is defined whether or not the argmax changes.
    n_skipped_non_rating = 0
    n_skipped_score_diff = 0
    n_skipped_no_anchors = 0

    for sample_idx, sample in enumerate(tqdm(dataset)):
        clean_prompt = format_prompt(sample["document"], sample["summary"], prompt_criteria, args.model_name, tokenizer)
        corrupt_prompt = format_prompt(sample["document"], sample["perturbed_summary"], prompt_criteria, args.model_name, tokenizer)

        inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
        inputs_corr = tokenizer(corrupt_prompt, return_tensors="pt").to(model.device)

        # 1. Get baseline outputs
        with torch.no_grad():
            out_clean = model(**inputs_clean)
            out_corr = model(**inputs_corr)

        target_id = torch.argmax(out_clean.logits[0, -1, :]).item()
        contra_id = torch.argmax(out_corr.logits[0, -1, :]).item()

        target_token_str = tokenizer.decode(target_id)
        contra_token_str = tokenizer.decode(contra_id)

        # Rating-token safety filter: both next-token predictions must
        # decode to a rating digit. No-op for Themis (which always
        # produces a digit after "The rating is "); important for
        # Prometheus-2 where a malformed prompt could otherwise cause
        # us to trace random tokens.
        if not (is_rating_token(target_id, tokenizer) and is_rating_token(contra_id, tokenizer)):
            n_skipped_non_rating += 1
            continue

        # Expected-rating-based score: E[r] = Σ_r r · P(r) for r ∈ {1..5},
        # probabilities renormalized within the rating-token set. Follows
        # G-Eval (Liu et al., EMNLP 2023) and is sensitive to the full
        # rating distribution rather than the discrete argmax.
        score_clean, rating_probs_clean = expected_rating_from_logits(
            out_clean.logits, rating_ids
        )
        score_corr, rating_probs_corr = expected_rating_from_logits(
            out_corr.logits, rating_ids
        )

        # Skip samples where the perturbation produced essentially no
        # distribution shift -- numerator and denominator of the normalized
        # effect would both be ~0 and the metric is uninformative. Threshold
        # is in rating-point units: a ~5% mass shift between adjacent
        # ratings produces roughly a 0.05 expected-rating change. Absolute
        # value because the perturbation can move E[r] either direction.
        if abs(score_clean - score_corr) < 0.05:
            n_skipped_score_diff += 1
            continue

        # 2. Determine Anchor Indices based on mode

        # diff_opcodes / clean_ids_len carry the shift-correction state for
        # length-mismatched samples. They stay None for last-mode (which uses
        # -1 indexing) and for length-matched window-mode samples (where
        # same-index patching is correct).
        diff_opcodes = None
        clean_ids_len = None

        if args.target_pos == "last":
            # Use -1 indexing — PyTorch's negative indexing resolves to
            # the last element of each tensor's sequence dim independently,
            # so we can patch corrupt's last token with clean's last
            # token without requiring matched lengths. This matters for
            # Prometheus-2 where Mistral's BPE often produces
            # different-length tokenizations even when the perturbation
            # is token-isomorphic under Llama-3's tokenizer.
            anchor_indices = [-1]
        else:
            clean_ids = inputs_clean.input_ids[0].cpu().tolist()
            corr_ids = inputs_corr.input_ids[0].cpu().tolist()

            # Length-mismatched samples are realigned: difflib opcodes are
            # computed once and used in the patching loop to map each corrupt
            # position back to its clean counterpart. For length-matched
            # samples diff_opcodes stays None and same-index patching is used.
            clean_ids_len = len(clean_ids)
            if len(clean_ids) != len(corr_ids):
                diff_opcodes = compute_diff_opcodes(clean_ids, corr_ids)

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

        # 3. Cache Clean Activations
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

        # 4. Patching Sweep
        for anchor_idx in anchor_indices:
            if anchor_idx + window_size > inputs_corr.input_ids.shape[1]:
                continue

            # NaN-initialized so cells without a valid clean source position
            # remain NaN through the accumulator pass instead of polluting
            # the cell-mean with spurious zeros.
            sample_effects = np.full((num_layers, window_size), np.nan)

            for layer in range(num_layers):
                mod_name = target_modules[layer]
                mod = target_module_objs[layer]

                for offset in range(window_size):
                    target_pos = anchor_idx + offset

                    # Determine the source position in the clean sequence:
                    #  - last mode or length-matched window mode: same-index
                    #    patching (source_pos = target_pos).
                    #  - length-mismatched window mode: walk the difflib
                    #    opcodes to find the clean position that
                    #    semantically corresponds to target_pos. -1 means
                    #    no meaningful clean counterpart exists (target_pos
                    #    fell in an 'insert' block), so the cell stays NaN.
                    if diff_opcodes is None:
                        source_pos = target_pos
                    else:
                        source_pos = compute_clean_source_pos(
                            target_pos, diff_opcodes, clean_ids_len
                        )
                        if source_pos < 0:
                            continue  # leave sample_effects[layer, offset] as NaN

                    # Defensive bounds check on source_pos. In window mode
                    # the cached clean activation has length = clean_ids_len;
                    # in last mode -1 indexing handles the bounds implicitly.
                    if (clean_ids_len is not None
                            and not (0 <= source_pos < clean_ids_len)
                            and source_pos != -1):
                        continue

                    hook = mod.register_forward_hook(
                        make_patch_hook(mod_name, target_pos, cache,
                                       source_pos=source_pos)
                    )

                    try:
                        with torch.no_grad():
                            patched_out = model(**inputs_corr)

                        score_patched, rating_probs_patched = expected_rating_from_logits(
                            patched_out.logits, rating_ids
                        )
                        effect = (score_patched - score_corr) / (score_clean - score_corr)
                        sample_effects[layer, offset] = effect

                        sample_logs.append({
                            "sample_idx": sample_idx,
                            "clean_token": target_token_str,
                            "corrupt_token": contra_token_str,
                            "module_name": mod_name,
                            "layer": layer,
                            "offset": offset,
                            # Both positions logged so downstream consumers
                            # can audit shift-correction decisions per cell.
                            "source_pos": source_pos,
                            "target_pos": target_pos,
                            "clean_score": score_clean,
                            "corrupt_score": score_corr,
                            "patched_score": score_patched,
                            "normalized_effect": effect,
                            # Full per-rating probability distributions saved
                            # for offline post-hoc analysis. Any alternative
                            # score metric (logit-diff, KL, argmax-prob, etc.)
                            # can be reconstructed from these columns without
                            # re-running the model.
                            "P_clean_1": float(rating_probs_clean[0]),
                            "P_clean_2": float(rating_probs_clean[1]),
                            "P_clean_3": float(rating_probs_clean[2]),
                            "P_clean_4": float(rating_probs_clean[3]),
                            "P_clean_5": float(rating_probs_clean[4]),
                            "P_corr_1": float(rating_probs_corr[0]),
                            "P_corr_2": float(rating_probs_corr[1]),
                            "P_corr_3": float(rating_probs_corr[2]),
                            "P_corr_4": float(rating_probs_corr[3]),
                            "P_corr_5": float(rating_probs_corr[4]),
                            "P_patched_1": float(rating_probs_patched[0]),
                            "P_patched_2": float(rating_probs_patched[1]),
                            "P_patched_3": float(rating_probs_patched[2]),
                            "P_patched_4": float(rating_probs_patched[3]),
                            "P_patched_5": float(rating_probs_patched[4]),
                        })

                    finally:
                        hook.remove()

            # NaN-safe per-cell accumulation: each cell's sum and count are
            # updated only by samples that produced a non-NaN effect for
            # that cell. Cells where every sample landed in a difflib
            # 'insert' block end up with count=0 and render as NaN.
            valid_cells = ~np.isnan(sample_effects)
            dataset_effects_sum[valid_cells] += sample_effects[valid_cells]
            dataset_effects_count[valid_cells] += 1
            valid_perturbations_count += 1

    # ==========================================
    # 5. DIAGNOSTICS, VISUALIZATION & SAVING
    # ==========================================

    print("\n=== Sweep diagnostics ===")
    print(f"  Total samples processed:           {len(dataset)}")
    print(f"  Skipped (next-token not a rating):       {n_skipped_non_rating}")
    print(f"  Skipped (|E_clean - E_corr| < 0.05):     {n_skipped_score_diff}")
    print(f"  Skipped (no valid anchor tokens):  {n_skipped_no_anchors}")
    print(f"  Valid perturbation instances:      {valid_perturbations_count}")
    # Per-cell coverage diagnostic: with shift-correction enabled, cells
    # corresponding to corrupt-side 'insert' opcode positions may have N=0
    # for some attacks. Print min/max coverage so anomalies are visible.
    if valid_perturbations_count > 0:
        min_n = int(dataset_effects_count.min())
        max_n = int(dataset_effects_count.max())
        zero_cells = int(np.sum(dataset_effects_count == 0))
        print(f"  Per-cell coverage: min N = {min_n}, max N = {max_n}, "
              f"cells with N=0 = {zero_cells}")

    if valid_perturbations_count > 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.plots_dir, exist_ok=True)

        # Per-cell mean via NaN-safe accumulator. Cells with count=0 (no
        # sample produced a valid effect there, e.g. all samples had the
        # offset land in a difflib 'insert' block) become NaN and render
        # as gray in the heatmap, rather than getting silently averaged
        # toward zero.
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_effects = np.where(
                dataset_effects_count > 0,
                dataset_effects_sum / np.maximum(dataset_effects_count, 1),
                np.nan,
            )
        print(f"\nSweep complete. Averaged over {valid_perturbations_count} perturbation instances.")

        safe_model_name = args.model_name.replace("/", "_")
        # The base-model control files carry a "_completion" tag (kept for compatibility
        # with the released result files); Themis / Prometheus filenames have no tag.
        prompt_tag = "_completion" if is_base_llama_model(args.model_name) else ""
        base_filename = (
            f"{safe_model_name}_{args.dataset_name}_aggregated_{args.attack}"
            f"_int{args.intensity}_{args.samples}_{args.seed}"
            f"_{args.target_pos}_{args.module_type}{prompt_tag}_sweep{slice_suffix}"
        )

        if args.target_pos == "last":
            x_labels = ["Last Token"]
        else:
            x_labels = [f"Token +{i}" for i in range(window_size)]
            x_labels[0] = "Perturbed Token (+0)"

        y_labels = [f"Layer {i}" for i in range(num_layers)]

        effects_df = pd.DataFrame(avg_effects, index=y_labels, columns=x_labels)
        csv_filename = os.path.join(args.results_dir, f"{base_filename}.csv")
        effects_df.to_csv(csv_filename)
        print(f"Saved raw data to '{csv_filename}'")

        # For sliced runs, also save the per-cell sample count so the
        # aggregator can do a properly weighted cross-slice average rather
        # than a naive arithmetic mean of per-slice means (which would
        # weight each slice equally regardless of how many samples
        # contributed to each cell).
        if slice_suffix:
            n_df = pd.DataFrame(
                dataset_effects_count, index=y_labels, columns=x_labels
            )
            n_csv_filename = os.path.join(args.results_dir, f"{base_filename}_n.csv")
            n_df.to_csv(n_csv_filename)
            print(f"Saved per-cell N matrix to '{n_csv_filename}'")

        log_df = pd.DataFrame(sample_logs)
        log_csv_filename = os.path.join(args.results_dir, f"{base_filename}_logs.csv")
        log_df.to_csv(log_csv_filename, index=False)
        print(f"Saved per-sample debugging log to '{log_csv_filename}'")

        # Skip per-slice PNG: it would show effects from only a subset of
        # samples and could mislead. Aggregation downstream produces the
        # canonical heatmap from the pooled slice _logs.csv files.
        if not slice_suffix:
            # Mask NaN cells so they render as light gray in the heatmap rather
            # than being annotated as 'nan' or coerced to 0 by the colormap.
            nan_mask = np.isnan(avg_effects)
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_effects, cmap="RdBu", center=0,
                        mask=nan_mask,
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
    else:
        print("\nNo valid samples passed the criteria/thresholds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Causal Tracing on LLM Evaluators.")

    # Model & Data Arguments
    parser.add_argument("--model_name", type=str, default="PKU-ONELab/Themis", help="HuggingFace Model ID")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., xsum, cnn_dailymail) for file naming")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON perturbations dataset")
    parser.add_argument("--criteria_file", type=str, default="configs/aspect_criteria.json", help="Path to criteria JSON")

    # Experiment Configurations
    parser.add_argument("--module_type", type=str, choices=["mlp", "attn"], required=True, help="Target module to patch")
    parser.add_argument("--target_pos", type=str, choices=["window", "last"], required=True, help="Where to trace: perturbed token window or final token")
    parser.add_argument("--window_size", type=int, default=6, help="Window size for 'window' target_pos")

    # Dataset specific trackers (mostly for file naming)
    parser.add_argument("--criteria", type=str, required=True,
                        help="Key in --criteria_file whose 'Gold' definition fills {prompt_criteria}")
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack key, e.g. read_preposition or adeq_entity_1to1 (see token_perturbations/prompts.py)")
    parser.add_argument("--intensity", type=str, default="1")
    parser.add_argument("--samples", type=str, default="200")
    parser.add_argument("--seed", type=str, default="42")

    # Environment
    parser.add_argument("--gpu", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES string, e.g. '0,1'. The model is sharded with device_map='auto'.")
    parser.add_argument("--results_dir", type=str, default="results/expected_rating",
                        help="Kept separate from the logit-difference results: both scripts use the same filenames")
    parser.add_argument("--plots_dir", type=str, default="plots/expected_rating")

    # Parallel execution support (e.g. one job per GPU).
    # Each worker is given a half-open dataset slice [slice_start, slice_end);
    # the output filename gains a "_slice{start}-{end}" suffix so per-worker
    # CSVs don't collide. Aggregation across slices is done post-hoc by
    # concat-and-regroup of the per-sample _logs.csv files (the saved
    # P_clean/P_corr/P_patched columns mean any score metric is recoverable
    # from the slices without re-running anything).
    parser.add_argument("--slice_start", type=int, default=None,
                        help="Inclusive start index of this worker's dataset slice. "
                             "Must be set together with --slice_end. When unset, "
                             "the worker processes the full filtered dataset.")
    parser.add_argument("--slice_end", type=int, default=None,
                        help="Exclusive end index of this worker's dataset slice. "
                             "Must be set together with --slice_start.")
    args = parser.parse_args()
    run_experiment(args)