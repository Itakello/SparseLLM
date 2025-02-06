import csv
import gc
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


def clear_gpu_memory():
    """Frees GPU memory (if CUDA is available)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _ensure_results_dir():
    """Ensures that the 'results/' directory exists."""
    if not os.path.exists("results"):
        os.makedirs("results")


def load_model(model_path: str) -> PreTrainedModel:
    """
    Load a model in full precision or auto precision, clearing GPU memory first.
    Adjust to your specific device or config as needed.
    """
    clear_gpu_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", low_cpu_mem_usage=True
    ).to("cuda")
    return model


def extract_model_details(model_path: str):
    """
    Basic string-parsing for:
      model_type   (either 'raw' or 'pruned', guessed from the path)
      model_name   (the 'base' name)
      pruned_on    (a subtask name or 'None')
      sparsity     (e.g. '25%', '50%', etc.)
    Adapt as needed depending on your naming conventions.
    """
    base = os.path.basename(model_path)
    if "--" in base:
        # e.g. meta-llama--AST_25pct
        prefix, details = base.split("--", 1)
        detail_parts = details.split("_")
        model_name = detail_parts[0] if len(detail_parts) > 0 else base
        pruned_on = detail_parts[1] if len(detail_parts) > 1 else "None"
        sparsity = detail_parts[2] if len(detail_parts) > 2 else "0%"
    else:
        model_name = base
        pruned_on = "None"
        sparsity = "0%"
    model_type = (
        "pruned" if "pruned" in base.lower() or "pct" in base.lower() else "raw"
    )
    return model_type, model_name, pruned_on, sparsity


# 1. Layer-wise Weight Distribution
def get_layerwise_weight_distribution(model_paths: list[str]):
    """
    For each model, compute per-layer weight distribution statistics and append each model's results
    immediately to 'results/layerwise_weight_distribution.csv'
    """
    _ensure_results_dir()
    csv_path = os.path.join("results", "layerwise_weight_distribution.csv")
    # Write header once
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "type",
                "name",
                "pruned_on",
                "sparsity",
                "layer",
                "mean",
                "std",
                "min",
                "max",
                "l2_norm",
                "total_params",
                "nonzero_params",
            ]
        )
    for model_path in model_paths:
        try:
            print(f"[LayerWeightDist] Analyzing model from: {model_path}")
            model = load_model(model_path)
            model_type, model_name, pruned_on, sparsity = extract_model_details(
                model_path
            )
            rows = []
            for name, param in model.named_parameters():
                if param.dim() == 0:
                    continue  # skip scalars
                data = param.data.cpu().float().numpy()
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                l2_norm = np.linalg.norm(data)
                total_count = data.size
                nonzero_count = np.count_nonzero(data)
                rows.append(
                    [
                        model_type,
                        model_name,
                        pruned_on,
                        sparsity,
                        name,
                        f"{mean_val:.4f}",
                        f"{std_val:.4f}",
                        f"{min_val:.4f}",
                        f"{max_val:.4f}",
                        f"{l2_norm:.4f}",
                        total_count,
                        nonzero_count,
                    ]
                )
            del model
            clear_gpu_memory()
        except Exception as e:
            print(
                f"Error in layer-wise weight distribution for model at {model_path}: {str(e)}"
            )
            continue
        # Append the results for this model immediately.
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"[LayerWeightDist] Processed {model_path} and appended results.")


# 3. Activation Analysis
def get_activation_statistics(model_paths: list[str]):
    """
    For each model, registers forward hooks to capture activation statistics and
    immediately appends each model's results to 'results/activation_statistics.csv'
    """
    _ensure_results_dir()
    csv_path = os.path.join("results", "activation_statistics.csv")
    # Write header once.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "type",
                "name",
                "pruned_on",
                "sparsity",
                "module",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    for model_path in model_paths:
        try:
            print(f"[ActivationStats] Analyzing model from: {model_path}")
            model = load_model(model_path)
            model.eval()
            model_type, model_name, pruned_on, sparsity = extract_model_details(
                model_path
            )
            activation_stats = {}
            hooks = []

            def get_hook(name):
                def hook(module, input, output):
                    out = output[0] if isinstance(output, (list, tuple)) else output
                    if torch.is_tensor(out):
                        act = out.detach().cpu().float()
                        activation_stats[name] = {
                            "mean": act.mean().item(),
                            "std": act.std().item(),
                            "min": act.min().item(),
                            "max": act.max().item(),
                        }

                return hook

            # Register hooks on leaf modules.
            for name, module in model.named_modules():
                if (
                    any(p.requires_grad for p in module.parameters())
                    and len(list(module.children())) == 0
                ):
                    hooks.append(module.register_forward_hook(get_hook(name)))

            vocab_size = (
                model.config.vocab_size if hasattr(model.config, "vocab_size") else 1000
            )
            dummy_input = torch.randint(0, vocab_size, (1, 10)).to(
                next(model.parameters()).device
            )
            with torch.no_grad():
                model(dummy_input)

            for h in hooks:
                h.remove()

            rows = []
            for module_name, stats in activation_stats.items():
                rows.append(
                    [
                        model_type,
                        model_name,
                        pruned_on,
                        sparsity,
                        module_name,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                        f"{stats['min']:.4f}",
                        f"{stats['max']:.4f}",
                    ]
                )
            del model
            clear_gpu_memory()
        except Exception as e:
            print(f"Error in activation statistics for model at {model_path}: {str(e)}")
            continue
        # Append the results for this model immediately.
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"[ActivationStats] Processed {model_path} and appended results.")


# 4. Sensitivity Testing
def get_sensitivity_analysis(model_paths: list[str]):
    """
    For each model, performs sensitivity analysis and appends each model's results immediately
    to 'results/sensitivity_analysis.csv'
    """
    _ensure_results_dir()
    csv_path = os.path.join("results", "sensitivity_analysis.csv")
    # Write header once.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["type", "name", "pruned_on", "sparsity", "module", "activation_diff"]
        )
    for model_path in model_paths:
        try:
            print(f"[SensitivityAnalysis] Analyzing model from: {model_path}")
            model = load_model(model_path)
            model.eval()
            model_type, model_name, pruned_on, sparsity = extract_model_details(
                model_path
            )
            baseline_acts = {}
            perturbed_acts = {}
            hooks = []

            def get_hook(storage, name):
                def hook(module, input, output):
                    out = output[0] if isinstance(output, (list, tuple)) else output
                    if torch.is_tensor(out):
                        storage[name] = out.detach().cpu().float()

                return hook

            for name, module in model.named_modules():
                if (
                    any(p.requires_grad for p in module.parameters())
                    and len(list(module.children())) == 0
                ):
                    hooks.append(
                        (
                            name,
                            module.register_forward_hook(get_hook(baseline_acts, name)),
                        )
                    )

            vocab_size = (
                model.config.vocab_size if hasattr(model.config, "vocab_size") else 1000
            )
            dummy_input = torch.randint(0, vocab_size, (1, 10)).to(
                next(model.parameters()).device
            )
            with torch.no_grad():
                model(dummy_input)

            for _, h in hooks:
                h.remove()
            hooks = []

            for name, module in model.named_modules():
                if (
                    any(p.requires_grad for p in module.parameters())
                    and len(list(module.children())) == 0
                ):
                    hooks.append(
                        (
                            name,
                            module.register_forward_hook(
                                get_hook(perturbed_acts, name)
                            ),
                        )
                    )

            perturbed_input = dummy_input.clone()
            last_token = perturbed_input[0, -1].item()
            new_token = (last_token + 1) % vocab_size
            perturbed_input[0, -1] = new_token

            with torch.no_grad():
                model(perturbed_input)

            for _, h in hooks:
                h.remove()

            rows = []
            for module_name in baseline_acts.keys():
                if module_name in perturbed_acts:
                    diff = torch.norm(
                        baseline_acts[module_name] - perturbed_acts[module_name]
                    ).item()
                    rows.append(
                        [
                            model_type,
                            model_name,
                            pruned_on,
                            sparsity,
                            module_name,
                            f"{diff:.4f}",
                        ]
                    )
            del model
            clear_gpu_memory()
        except Exception as e:
            print(f"Error in sensitivity analysis for model at {model_path}: {str(e)}")
            continue
        # Append the results for this model immediately.
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"[SensitivityAnalysis] Processed {model_path} and appended results.")


def eval_zero_shot(
    model,
    tokenizer,
    task_list,
):
    """
    Evaluates zero-shot performance on a set of tasks.
    """

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    wrapped_model = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=task_list,
        num_fewshot=0,
        device="cuda",
        limit=1000,
    )
    return results


def run_eval_zero_shot_all(
    model_paths: list,
    tokenizer,
    task_list=[
        "hellaswag",
        "winogrande",
        "arc_easy",
        "boolq",
        "arc_challenge",
        "wikitext",
    ],
) -> None:
    """
    Evaluates zero-shot performance on a set of tasks and appends each model's results immediately
    to 'results/eval_zero_shot_summary.csv'
    """
    # Monkey-patch datasets.load_dataset to include trust_remote_code=True
    import datasets

    _orig_load_dataset = datasets.load_dataset

    def _load_dataset_with_trust(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return _orig_load_dataset(*args, **kwargs)

    datasets.load_dataset = _load_dataset_with_trust

    _ensure_results_dir()
    csv_path = os.path.join("results", "eval_zero_shot_summary.csv")
    header = ["model_type", "model_name", "pruned_on", "sparsity"] + task_list
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    model_paths = [model_path for model_path in model_paths if "3b" in model_path]
    for model_path in model_paths:
        try:
            print(f"[ZeroShot] Loading model from: {model_path}")
            model = load_model(model_path)
            model_type, model_name, pruned_on, sparsity = extract_model_details(
                model_path
            )
            res = eval_zero_shot(model, tokenizer, task_list)
            row = [model_type, model_name, pruned_on, sparsity]
            for task in task_list:
                if task in res["results"]:
                    score = (
                        res["results"][task].get("acc,none", "N/A")
                        if task != "wikitext"
                        else res["results"][task].get("word_perplexity,none", "N/A")
                    )
                else:
                    score = "N/A"
                row.append(score)
            del model
            clear_gpu_memory()
        except Exception as e:
            print(f"Error evaluating model at {model_path}: {str(e)}")
            continue
        # Append the row for this model immediately.
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[ZeroShot] Processed {model_path} and appended results.")


def get_attention_heads_statistics(
    model_paths: list[str],
    seq_len: int = 32,
    results_csv="results/attention_heads_stats.csv",
):
    """
    For each model, captures the attention probabilities for each layer and each head
    on a dummy input (length=seq_len), then computes basic stats: mean, std, min, max
    across the attention matrix. Writes them to a CSV file with columns:
      type, name, pruned_on, sparsity, layer_index, head_index, mean_attn, std_attn, min_attn, max_attn

    Args:
      model_paths: list of model checkpoint directories
      seq_len: length of the dummy input
      results_csv: path to the output CSV
    """
    import csv

    import torch

    _ensure_results_dir()  # ensure 'results/' folder
    # Write CSV header
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "type",
                "name",
                "pruned_on",
                "sparsity",
                "layer_index",
                "head_index",
                "mean_attn",
                "std_attn",
                "min_attn",
                "max_attn",
            ]
        )

    for model_path in model_paths:
        try:
            print(f"[AttentionHeadsStats] Analyzing model from: {model_path}")
            model = load_model(model_path)
            model_type, model_name, pruned_on, sparsity = extract_model_details(
                model_path
            )

            # Some models require the config to explicitly return attentions.
            # We'll do: model.config.output_attentions = True
            model.config.output_attentions = True
            model.eval()

            # Create dummy input
            vocab_size = (
                model.config.vocab_size
                if hasattr(model.config, "vocab_size")
                else 30522
            )
            dummy_input = torch.randint(0, vocab_size, (1, seq_len)).to(
                next(model.parameters()).device
            )

            with torch.no_grad():
                outputs = model(dummy_input, output_attentions=True)
                # outputs.attentions is a tuple of shape (#layers, ), each is (batch_size, num_heads, seq_len, seq_len)

            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                print(f"  -> No attentions returned for model {model_name}. Skipping.")
                del model
                clear_gpu_memory()
                continue

            # We'll collect rows to write after processing all heads
            csv_rows = []

            # Loop over each layer
            for layer_idx, attn_tensor in enumerate(outputs.attentions):
                # attn_tensor shape = (batch_size=1, num_heads, seq_len, seq_len)
                attn_array = (
                    attn_tensor[0].cpu().float().numpy()
                )  # shape = (num_heads, seq_len, seq_len)
                num_heads = attn_array.shape[0]
                for head_idx in range(num_heads):
                    mat = attn_array[head_idx]  # shape = (seq_len, seq_len)
                    mean_val = float(mat.mean())
                    std_val = float(mat.std())
                    min_val = float(mat.min())
                    max_val = float(mat.max())

                    csv_rows.append(
                        [
                            model_type,
                            model_name,
                            pruned_on,
                            sparsity,
                            layer_idx,
                            head_idx,
                            f"{mean_val:.6f}",
                            f"{std_val:.6f}",
                            f"{min_val:.6f}",
                            f"{max_val:.6f}",
                        ]
                    )

            # Append results for this model
            with open(results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)

            del model
            clear_gpu_memory()
            print(f"[AttentionHeadsStats] Processed {model_path} and appended results.")

        except Exception as e:
            print(
                f"Error in attention heads statistics for model at {model_path}: {str(e)}"
            )
            continue
