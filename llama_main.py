# /lsc_project/submodules/SparseLLM/llama_main.py

import argparse

import torch
from datautils import get_loaders
from model_utils import get_llama, llama_eval, llama_sparsellm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    # ... other arguments ...
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--sparsity", type=float, default=0.5)
    # ...
    args = parser.parse_args()

    model = get_llama(args)
    model.eval()
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if args.sparsity > 0.0:
        llama_sparsellm(model, None, dataloader, torch.device("cuda"), args)

    # Evaluate
    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        llama_eval(model, testloader, torch.device("cuda"), args, dataset)

    # Optional save
    if args.save:
        model.save_pretrained(args.save)


if __name__ == "__main__":
    main()
