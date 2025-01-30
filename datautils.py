import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    tokenizer = get_tokenizer(model)
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)


def _build_calibration_prompt(record):
    """
    Build a single text block that includes the 'correct' answer,
    so that the pruning procedure can see relevant text+labels.
    For example:

      [SYSTEM]
      <some instruction about picking A/B/C/D>

      [USER]
      <Story + 4 candidate answers in canonical order>

      [ASSISTANT]
      [[CORRECT_ANSWER]]
    """
    # Simple system instruction
    system_prompt = """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
    Note:
    (1) Please only output the most likely answer index in the format: [[Answer Index]];
    (2) You must choose one of A, B, C, D even if the story doesn't have enough info;
    (3) Output only the answer index, nothing else.
    """

    # Distinguish whether we have 4 choices or 2
    optC = record.get("OPTION-C", None)
    if optC is not None:
        # 4-choice
        A = record["OPTION-A"].replace("A. ", "")
        B = record["OPTION-B"].replace("B. ", "")
        C = record["OPTION-C"].replace("C. ", "")
        D = record["OPTION-D"].replace("D. ", "")
        user_part = (
            f"[Story]\n{record['STORY']}\n\n"
            f"[Question]\n{record['QUESTION']}\n\n"
            f"[Candidate Answers]\n"
            f"A. {A}\n"
            f"B. {B}\n"
            f"C. {C}\n"
            f"D. {D}"
        )
    else:
        # 2-choice
        A = record["OPTION-A"].replace("A. ", "")
        B = record["OPTION-B"].replace("B. ", "")
        user_part = (
            f"[Story]\n{record['STORY']}\n\n"
            f"[Question]\n{record['QUESTION']}\n\n"
            f"[Candidate Answers]\n"
            f"A. {A}\n"
            f"B. {B}"
        )

    answer = record.get("ANSWER\nANSWER", "A") or "A"
    assistant_part = f"[[{answer}]]"

    text_block = (
        f"[SYSTEM]\n{system_prompt}\n"
        f"[USER]\n{user_part}\n"
        f"[ASSISTANT]\n{assistant_part}"
    )
    return text_block


def _load_subtask_data(subtask_file):
    """
    Load all records from the specified ToMBench subtask JSONL file.
    """
    path = os.path.join("data", subtask_file)
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records


def get_tom(tokenizer, subtask_file, train_num=32, test_num=5, seed=42):
    """
    Prepare the ToMBench calibration data (trainloader) and test data (list of leftover records).

    1) Loads subtask_file from ToMBench.
    2) Splits into 'train_num' for calibration vs. remainder for test.
    3) Builds a "calibration prompt" for each train record, then tokenizes & pads to the max length across them.
       The returned 'trainloader' is a list of (inp, tar) pairs, each shaped [1, seq_len].
       - We do *not* fix a seqlen; we let them pad to the longest sample.
    4) The test set is limited to `test_num` samples (default 5). For each test record, we keep it raw (a dict).
       We'll handle prompting in the evaluation code.

    Returns:
      trainloader: list of (inp, tar) pairs ready for unstructured pruning with e.g. llama_sparsellm
      test_records: list of leftover records (the test data)

    Example usage:
      trainloader, test_recs = get_tom(tokenizer, "False Belief Task.jsonl", 32, 5)
    """
    random.seed(seed)
    records = _load_subtask_data(subtask_file)
    random.shuffle(records)

    # 1) Split
    train_records = records[:train_num]
    leftover_records = records[train_num:]

    # 2) Build calibration prompts

    train_prompts = [_build_calibration_prompt(r) for r in train_records]

    # 3) Tokenize each prompt (variable length)
    encoded_list = []
    for txt in train_prompts:
        enc = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
        # shape: [1, length]
        encoded_list.append(enc)

    # Find max length
    max_len = 0
    for enc in encoded_list:
        length = enc["input_ids"].shape[1]
        if length > max_len:
            max_len = length

    # 4) Pad each to max_len and build (inp, tar)
    trainloader = []
    for enc in encoded_list:
        length = enc["input_ids"].shape[1]
        pad_needed = max_len - length

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if pad_needed > 0:
            pad_ids = torch.full(
                (1, pad_needed), tokenizer.pad_token_id, dtype=torch.long
            )
            pad_mask = torch.zeros((1, pad_needed), dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        # Now create the targets so that we are effectively doing next-token or LM-style supervision.
        tar = input_ids.clone()
        # Standard trick: mask everything except the “shifted by 1”
        tar[:, :-1] = -100

        trainloader.append((input_ids, tar, attention_mask))

    # 5) Limit test set size to test_num (if positive)
    test_records = leftover_records
    if test_num is not None and test_num > 0:
        test_records = test_records[:test_num]

    return trainloader, test_records
