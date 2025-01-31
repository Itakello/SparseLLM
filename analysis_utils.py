import os
import csv
import torch
import math
from transformers import PreTrainedModel


def _ensure_results_dir():
    """
    Ensures that the './results' directory exists.
    If not, creates it.
    """
    if not os.path.exists("results"):
        os.makedirs("results")


def extract_zero_mask(tensor: torch.Tensor):
    """
    Given a tensor, return a Boolean mask where True means 'weight is nonzero'.
    This helps identify which weights survived pruning.
    """
    return tensor != 0.0


def extract_llama_masks(model: PreTrainedModel):
    """
    Given a pruned LLaMA model, returns a dict of {layer_name: mask (bool Tensor)}
    for all weight matrices (typically nn.Linear in each transformer block).

    Note: This function assumes your model structure includes `model.model.layers[i]`.
          Adjust as needed if your model has a different naming.
    """
    masks = {}
    for i, layer in enumerate(model.model.layers):
        for name, param in layer.named_parameters():
            # e.g. name = 'self_attn.q_proj.weight' or 'mlp.up_proj.weight'
            # We only care about weight parameters, not biases
            if "weight" in name:
                full_name = f"layers.{i}.{name}"
                masks[full_name] = extract_zero_mask(param.data)
    return masks


def get_global_sparsity(model: PreTrainedModel, model_id: str = "model"):
    """
    Computes overall fraction of weights that are pruned across the entire model.
    Saves CSV file: `results/global_sparsity_{model_id}.csv`.

    CSV columns:
      - total_params
      - nonzero_params
      - fraction_pruned  (pruned / total)
      - fraction_remaining (1 - fraction_pruned)
    """
    _ensure_results_dir()
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        if param.dim() == 0:
            continue  # skip scalars
        data = param.data
        total_params += data.numel()
        nonzero_params += torch.count_nonzero(data).item()

    fraction_pruned = (total_params - nonzero_params) / total_params
    fraction_remaining = 1.0 - fraction_pruned

    csv_path = f"results/global_sparsity_{model_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["total_params", "nonzero_params", "fraction_pruned", "fraction_remaining"]
        )
        writer.writerow(
            [total_params, nonzero_params, fraction_pruned, fraction_remaining]
        )


def get_per_layer_sparsity(model: PreTrainedModel, model_id: str = "model"):
    """
    Computes fraction of nonzero weights *per layer* for the model.
    Saves CSV file: `results/per_layer_sparsity_{model_id}.csv`.

    CSV columns:
      - layer_name
      - total_params
      - nonzero_params
      - fraction_pruned
      - fraction_remaining
    """
    _ensure_results_dir()
    rows = []
    for i, layer in enumerate(model.model.layers):
        layer_total = 0
        layer_nonzero = 0

        for name, param in layer.named_parameters():
            if "weight" not in name:
                continue
            data = param.data
            layer_total += data.numel()
            layer_nonzero += torch.count_nonzero(data).item()

        fraction_pruned = (layer_total - layer_nonzero) / layer_total
        fraction_remaining = 1.0 - fraction_pruned
        rows.append(
            [
                f"layer_{i}",
                layer_total,
                layer_nonzero,
                fraction_pruned,
                fraction_remaining,
            ]
        )

    csv_path = f"results/per_layer_sparsity_{model_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_name",
                "total_params",
                "nonzero_params",
                "fraction_pruned",
                "fraction_remaining",
            ]
        )
        for row in rows:
            writer.writerow(row)


def get_attention_mlp_breakdown(model: PreTrainedModel, model_id: str = "model"):
    """
    Computes sparsity breakdown for:
      - The attention projection sub-layers (q_proj, k_proj, v_proj, o_proj)
      - The MLP sub-layers (gate_proj, up_proj, down_proj)
    Saves CSV: `results/attention_mlp_breakdown_{model_id}.csv`.

    CSV columns:
      - sublayer_type (e.g. 'attention_qkv', 'mlp_up_proj', etc.)
      - total_params
      - nonzero_params
      - fraction_pruned
      - fraction_remaining
    """
    _ensure_results_dir()
    # We'll track counts in a dictionary keyed by sublayer type
    stats = {}

    def update_stats(key, tensor):
        tnum = tensor.numel()
        nz = torch.count_nonzero(tensor).item()
        if key not in stats:
            stats[key] = {"total": 0, "nonzero": 0}
        stats[key]["total"] += tnum
        stats[key]["nonzero"] += nz

    for i, layer in enumerate(model.model.layers):
        # Each layer should have 'self_attn' with q_proj, k_proj, v_proj, o_proj
        # and 'mlp' with gate_proj, up_proj, down_proj
        for name, param in layer.named_parameters():
            if "weight" not in name:
                continue
            data = param.data
            # Identify whether it's attention or MLP
            if "self_attn.q_proj" in name:
                update_stats("attention_q_proj", data)
            elif "self_attn.k_proj" in name:
                update_stats("attention_k_proj", data)
            elif "self_attn.v_proj" in name:
                update_stats("attention_v_proj", data)
            elif "self_attn.o_proj" in name:
                update_stats("attention_o_proj", data)
            elif "mlp.gate_proj" in name:
                update_stats("mlp_gate_proj", data)
            elif "mlp.up_proj" in name:
                update_stats("mlp_up_proj", data)
            elif "mlp.down_proj" in name:
                update_stats("mlp_down_proj", data)

    # Convert to CSV
    csv_path = f"results/attention_mlp_breakdown_{model_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sublayer_type",
                "total_params",
                "nonzero_params",
                "fraction_pruned",
                "fraction_remaining",
            ]
        )
        for k, vals in stats.items():
            total = vals["total"]
            nz = vals["nonzero"]
            fraction_pruned = (total - nz) / total
            fraction_remaining = 1.0 - fraction_pruned
            writer.writerow([k, total, nz, fraction_pruned, fraction_remaining])


def get_heads_vs_intermediate_sparsity(model: PreTrainedModel, model_id: str = "model"):
    """
    Example function to measure how many weights remain in each attention head
    vs. the MLP intermediate dimension. This is more detailed and requires
    some knowledge about how LLaMA organizes heads. Here we do a
    simplified approach: we collect Q/K/V by head dimension, and MLP separately.

    Saves CSV: `results/heads_vs_intermediate_{model_id}.csv`.

    CSV columns (one row per layer):
      - layer_index
      - total_qkv_params
      - nonzero_qkv
      - fraction_qkv_pruned
      - total_mlp_params
      - nonzero_mlp
      - fraction_mlp_pruned
      - fraction_remaining_qkv
      - fraction_remaining_mlp
    """
    _ensure_results_dir()
    rows = []

    for i, layer in enumerate(model.model.layers):
        qkv_total = 0
        qkv_nonzero = 0
        mlp_total = 0
        mlp_nonzero = 0

        for name, param in layer.named_parameters():
            if "weight" not in name:
                continue
            data = param.data
            if any(x in name for x in ["q_proj", "k_proj", "v_proj"]):
                qkv_total += data.numel()
                qkv_nonzero += torch.count_nonzero(data).item()
            elif any(
                x in name for x in ["mlp.up_proj", "mlp.down_proj", "mlp.gate_proj"]
            ):
                mlp_total += data.numel()
                mlp_nonzero += torch.count_nonzero(data).item()

        if qkv_total == 0:
            qkv_total = 1
        if mlp_total == 0:
            mlp_total = 1

        qkv_frac_pruned = (qkv_total - qkv_nonzero) / qkv_total
        mlp_frac_pruned = (mlp_total - mlp_nonzero) / mlp_total

        rows.append(
            [
                i,
                qkv_total,
                qkv_nonzero,
                qkv_frac_pruned,
                mlp_total,
                mlp_nonzero,
                mlp_frac_pruned,
                1.0 - qkv_frac_pruned,
                1.0 - mlp_frac_pruned,
            ]
        )

    csv_path = f"results/heads_vs_intermediate_{model_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_index",
                "total_qkv_params",
                "nonzero_qkv",
                "fraction_qkv_pruned",
                "total_mlp_params",
                "nonzero_mlp",
                "fraction_mlp_pruned",
                "fraction_remaining_qkv",
                "fraction_remaining_mlp",
            ]
        )
        for row in rows:
            writer.writerow(row)


def compute_mask_overlap(
    maskA: torch.Tensor, maskB: torch.Tensor, metric: str = "jaccard"
):
    """
    Compute overlap between two Boolean masks of the same shape.
    By default we use Jaccard similarity:
       jaccard = intersection / union
    (where intersection is the count of True in both, union is True in either).

    If you want another metric, you could implement 'cosine', etc.
    """
    assert maskA.shape == maskB.shape, "Masks must have the same shape"
    intersection = torch.logical_and(maskA, maskB).sum().item()
    union = torch.logical_or(maskA, maskB).sum().item()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    if metric == "jaccard":
        return intersection / union
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def get_mask_overlap(
    modelA: PreTrainedModel,
    modelB: PreTrainedModel,
    model_id_a: str = "modelA",
    model_id_b: str = "modelB",
    metric: str = "jaccard",
):
    """
    Computes the (global) mask overlap between two LLaMA models
    (assuming they've been pruned and have zeros for pruned weights).
    We gather all weight parameters, build a single combined mask for each model,
    then compute overlap according to `metric`.

    Saves CSV file: results/mask_overlap_{model_id_a}_vs_{model_id_b}.csv
      columns: [overlap_metric, overlap_score, intersection, union]
    """
    _ensure_results_dir()

    # Create one big flatten mask for each model
    maskA_list = []
    maskB_list = []
    for paramA, paramB in zip(modelA.parameters(), modelB.parameters()):
        if paramA.dim() == 0 or paramB.dim() == 0:
            # skip scalars
            continue
        # Compute boolean masks
        mA = (paramA.data != 0.0).view(-1)
        mB = (paramB.data != 0.0).view(-1)
        # Append
        maskA_list.append(mA)
        maskB_list.append(mB)

    allA = torch.cat(maskA_list, dim=0)
    allB = torch.cat(maskB_list, dim=0)

    intersection = torch.logical_and(allA, allB).sum().item()
    union = torch.logical_or(allA, allB).sum().item()

    if union == 0:
        overlap_score = 1.0 if intersection == 0 else 0.0
    else:
        overlap_score = intersection / union

    csv_path = f"results/mask_overlap_{model_id_a}_vs_{model_id_b}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["overlap_metric", "overlap_score", "intersection", "union"])
        writer.writerow([metric, overlap_score, intersection, union])


def get_layer_by_layer_mask_overlap(
    modelA: PreTrainedModel,
    modelB: PreTrainedModel,
    model_id_a: str = "modelA",
    model_id_b: str = "modelB",
    metric: str = "jaccard",
):
    """
    For each layer i, compute the overlap in that layer's weight masks
    and write them as rows in a CSV.
    CSV: results/layer_by_layer_overlap_{model_id_a}_vs_{model_id_b}.csv

    columns:
      - layer_index
      - sublayer_name (like 'self_attn.q_proj' or 'mlp.up_proj')
      - intersection
      - union
      - overlap (intersection/union)
    """
    _ensure_results_dir()
    # We'll iterate over corresponding layers in both models
    # and compute mask overlap sublayer by sublayer.
    rows = []
    for i, (layerA, layerB) in enumerate(zip(modelA.model.layers, modelB.model.layers)):
        # gather submodules
        for (nameA, paramA), (nameB, paramB) in zip(
            layerA.named_parameters(), layerB.named_parameters()
        ):
            if "weight" not in nameA:
                continue
            assert nameA == nameB, "Mismatch in sublayer structure"
            dataA = paramA.data
            dataB = paramB.data

            # Flatten
            maskA = (dataA != 0.0).view(-1)
            maskB = (dataB != 0.0).view(-1)
            intersection = torch.logical_and(maskA, maskB).sum().item()
            union = torch.logical_or(maskA, maskB).sum().item()

            if union == 0:
                overlap = 1.0 if intersection == 0 else 0.0
            else:
                overlap = intersection / union

            rows.append([i, nameA, intersection, union, overlap])

    # write to CSV
    csv_path = f"results/layer_by_layer_overlap_{model_id_a}_vs_{model_id_b}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer_index", "sublayer_name", "intersection", "union", "overlap"]
        )
        for row in rows:
            writer.writerow(row)
