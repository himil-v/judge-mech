"""
Generate clean/corrupt summary pairs with GPT-4o (Section 3, "Automated Perturbation Generation").

For each sampled summary, GPT-4o is prompted with category-specific few-shot exemplars and
asked to inject exactly k errors of one attack type. The model returns the perturbed summary
together with explicit `original_tokens` / `new_tokens` arrays (the token-level modification
map used to anchor causal tracing). Output is requested in JSON mode.

Output: <output_dir>/<dataset>_<attack>_int<k>_<samples>_<seed>.json, a list of records:
    document            source article
    summary             clean reference summary
    perturbed_summary   corrupted summary ("JSON_ERROR" if the response could not be parsed;
                        the key is absent if the API call failed after all retries)
    original_tokens     tokens replaced in the clean summary
    new_tokens          tokens inserted in the perturbed summary
    perturbation_type   attack key (see prompts.py)
    intensity           k

Structural filtering (JSON failures, intensity violations, empty `new_tokens`) is applied
downstream, before scoring and causal tracing.

Note: generation uses temperature 0.7, so re-running will not reproduce the released corpus
token for token. Use the released files in this directory to reproduce the paper's results.

Example:
    python generate_perturbations.py --dataset cnn_dailymail --attack read_preposition \
        --samples 200 --intensity 1 --seed 42
"""

import argparse
import json
import os
import sys
import time

from datasets import load_dataset
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tqdm import tqdm

from prompts import ATTACKS, PROMPTS


def get_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set. Add it to a .env file (OPENAI_API_KEY=...) or export it.")
    return OpenAI(api_key=api_key)


def load_and_prep_data(dataset_name, seed, sample_size):
    """Load the train split, shuffle with `seed`, take the first `sample_size` items,
    and standardize keys to {document, summary}."""
    print(f"Loading {dataset_name} (seed={seed}, n={sample_size})...")

    if dataset_name == "xsum":
        data = load_dataset("xsum")["train"]
        doc_key, sum_key = "document", "summary"
    elif dataset_name == "cnn_dailymail":
        data = load_dataset("cnn_dailymail", "3.0.0")["train"]
        doc_key, sum_key = "article", "highlights"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    selected = data.shuffle(seed=seed).select(range(sample_size))
    return [{"document": item[doc_key], "summary": item[sum_key]} for item in selected]


def generate_with_retry(client, model, prompt, index, max_retries=3):
    """Call the OpenAI chat API in JSON mode, retrying on rate-limit and transient errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            print(f"Warning: sample {index} returned an empty response (attempt {attempt + 1}).")

        except RateLimitError:
            wait = (attempt + 1) * 5
            print(f"Rate limit hit. Waiting {wait}s...")
            time.sleep(wait)
        except APIConnectionError as e:
            print(f"Connection error: {e}")
            time.sleep(2)
        except APIStatusError as e:
            print(f"Server error {e.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    print(f"Failed to generate sample {index}.")
    return None


def run(dataset_name, attack_name, sample_size, intensity, seed, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{dataset_name}_{attack_name}_int{intensity}_{sample_size}_{seed}.json"
    )
    print(f"--- {attack_name} on {dataset_name} (k={intensity}) -> {output_path}")

    client = get_client()
    samples = load_and_prep_data(dataset_name, seed, sample_size)
    template = PROMPTS[attack_name]

    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        prompt = template.format(right_summary=sample["summary"], intensity=intensity)
        response = generate_with_retry(client, model_name, prompt, i)

        if response:
            try:
                parsed = json.loads(response)
                sample["perturbed_summary"] = parsed.get("perturbed_summary", "")
                sample["original_tokens"] = parsed.get("original_tokens", [])
                sample["new_tokens"] = parsed.get("new_tokens", [])
                sample["perturbation_type"] = attack_name
                sample["intensity"] = intensity
            except json.JSONDecodeError:
                print(f"Could not decode JSON for sample {i}. Raw output: {response}")
                sample["perturbed_summary"] = "JSON_ERROR"

        if (i + 1) % 10 == 0:  # checkpoint
            with open(output_path, "w") as f:
                json.dump(samples, f, indent=4)

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=4)
    print(f"Done: {sample_size} samples saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GPT-4o perturbations of summaries.")
    parser.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default="cnn_dailymail")
    parser.add_argument("--attack", choices=ATTACKS, required=True)
    parser.add_argument("--samples", type=int, default=200, help="Samples per run (paper: 200)")
    parser.add_argument("--intensity", type=int, default=1, help="Errors per summary, k (paper: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (paper: 17, 23, 42)")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output_dir", default="generated_perturbations")
    args = parser.parse_args()

    run(args.dataset, args.attack, args.samples, args.intensity, args.seed, args.model, args.output_dir)