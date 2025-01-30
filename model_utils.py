import torch
import torch.nn as nn

from submodules.wanda.prune import prune_wanda


@torch.no_grad()
def llama_sparsellm(model, dataloader, dev, sparsity) -> None:
    """
    Replacement of the old 'llama_sparsellm' that now calls Wanda's prune_wanda
    using the calibration data from 'dataloader'.
    """
    print("Starting Wanda-based pruning...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # The LLaMA layers
    layers = model.model.layers

    # We'll gather the hidden input states (inps) for each calibration sample,
    # plus the attention_mask and position_ids (mirroring old logic).
    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
    max_len = max(batch[0].shape[1] for batch in dataloader)  # The largest seq_len
    inps = torch.zeros(
        (nsamples, max_len, model.config.hidden_size), dtype=dtype, device=dev
    )
    attention_masks = [None] * nsamples
    position_embeddings = [None] * nsamples

    # We'll use a forward hook on the first layer to capture the hidden states
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            idx = cache["i"]
            inps[idx, : hidden_states.size(1)] = hidden_states
            attention_masks[idx] = kwargs.get("attention_mask", None)
            position_embeddings[idx] = kwargs.get("position_embeddings", None)
            cache["i"] += 1
            raise ValueError  # early stop

    # Hook the first layer
    layers[0] = Catcher(layers[0]).to(dev)

    # Move embedding + norm to dev so forward will run
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)

    # Step through each calibration sample
    sample_idx = 0
    for batch in dataloader:
        try:
            inp_ids = batch[0].to(dev)
            _ = model(inp_ids, attention_mask=batch[2].to(dev), use_cache=False)
        except ValueError:
            pass
        sample_idx += 1

    # Restore the actual layer
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    # Update the model seqlen if needed
    model.seqlen = max_len

    # Now call Wanda's prune_wanda with these calibration inputs
    # We'll package them as a "calib_data" dict to pass to prune_wanda
    calib_data = {
        "inps": inps,
        "attention_masks": attention_masks,
        "position_embeddings": position_embeddings,
    }

    # Prepare Wanda arguments: prune_n, prune_m, unstructured ratio...
    from types import SimpleNamespace

    wanda_args = SimpleNamespace(
        sparsity_ratio=sparsity,  # Wanda expects e.g. 0.5 for 50%
        # Because we do unstructured by default
        sparsity_type="unstructured",
        prune_method="wanda",  # Hard-coded for Wanda
        use_variant=False,  # Or we can set from args
        nsamples=nsamples,
        seed=0,  # unused here
    )

    prune_wanda(model, calib_data, wanda_args, device=dev)

    model.config.use_cache = use_cache
    print("Wanda-based pruning done!")
