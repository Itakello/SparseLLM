import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = """Below is a multiple-choice question with a story and serveral answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
Note:
(1) Please only output the most likely answer index in the format: [[Answer Index]], for example, if the most likely answer option is 'A. Handbag', then output '[[A]]';
(2) You must choose one of the given answer options 'A, B, C, D' as the most likely answer, regardless of whether the story provides enough information. If you think there is not enough information in the story to choose an answer, please randomly output one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]";
(3) Please only output the most likely answer index based on the given information, and do not output any other content."""


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


def get_loader_wikitext2(nsamples=128, seed=0, seqlen=2048, model=""):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)


def _shuffle_options(options, letter_map):
    """Helper function to shuffle options while maintaining answer mapping"""
    items = list(zip(options, letter_map.keys()))
    random.shuffle(items)
    shuffled_options, shuffled_keys = zip(*items)
    new_letter_map = {old: new for old, new in zip(letter_map.keys(), shuffled_keys)}
    return shuffled_options, new_letter_map


def _build_user_message(record, shuffle=False):
    # Build the user part for both calibration and testing
    optC = record.get("OPTION-C", None)
    if isinstance(optC, str) and len(optC) > 0:
        # 4-choice
        options = [
            record["OPTION-A"].replace("A. ", ""),
            record["OPTION-B"].replace("B. ", ""),
            record["OPTION-C"].replace("C. ", ""),
            record["OPTION-D"].replace("D. ", ""),
        ]
        letter_map = {"A": "A", "B": "B", "C": "C", "D": "D"}

        if shuffle:
            options, letter_map = _shuffle_options(options, letter_map)

        user_msg = (
            f"[Story]\n{record['STORY']}\n\n"
            f"[Question]\n{record['QUESTION']}\n\n"
            f"[Candidate Answers]\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}"
        )
    else:
        # 2-choice
        options = [
            record["OPTION-A"].replace("A. ", ""),
            record["OPTION-B"].replace("B. ", ""),
        ]
        letter_map = {"A": "A", "B": "B"}

        if shuffle:
            options, letter_map = _shuffle_options(options, letter_map)

        user_msg = (
            f"[Story]\n{record['STORY']}\n\n"
            f"[Question]\n{record['QUESTION']}\n\n"
            f"[Candidate Answers]\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}"
        )
    return user_msg, letter_map


def _build_calibration_prompt(record, tokenizer):

    user_msg, _ = _build_user_message(
        record, shuffle=False
    )  # no shuffle for calibration
    answer = record.get("ANSWER", "A") or "A"
    assistant_msg = f"[[{answer}]]"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text_block = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
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

    train_prompts = [_build_calibration_prompt(r, tokenizer) for r in train_records]

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
