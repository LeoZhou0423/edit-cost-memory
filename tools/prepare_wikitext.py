#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare WikiText-103 token streams for training.

Downloads WikiText-103 (raw) from Hugging Face, tokenizes with the GPT-2 BPE
encoder, filters out tokens >= the reserved probe id (VOCAB - 1), and writes
``int64`` arrays into the data directory:

    <data_dir>/wikitext103_train.npy
    <data_dir>/wikitext103_valid.npy
    <data_dir>/wikitext103_test.npy

The data directory defaults to ``data/`` at the repository root and can be
overridden with the ``DCGR3D_DATA`` environment variable (the same variable the
training/evaluation scripts honour).

Usage:
    python tools/prepare_wikitext.py

Dependencies: ``huggingface_hub``, ``pyarrow`` (or ``datasets``), ``tiktoken``.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

VOCAB = 50257
PROBE = VOCAB - 1       # 50256 reserved for the trigger-gate marker
MAXTOK = PROBE          # keep ids < 50256

# HF source (raw split keeps newlines, matching the paper's pre-processing)
REPO = "wikitext"
SPLIT_FILES = {
    "wikitext103_train": [
        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "wikitext-103-raw-v1/train-00001-of-00002.parquet",
    ],
    "wikitext103_valid": ["wikitext-103-raw-v1/validation.parquet"],
    "wikitext103_test": ["wikitext-103-raw-v1/test.parquet"],
}

enc = tiktoken.get_encoding("gpt2")


def data_dir() -> str:
    d = os.environ.get("DCGR3D_DATA")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def parquet_to_tokens(files, name, cache_dir):
    texts = []
    for f in files:
        p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f,
                            cache_dir=cache_dir)
        print(f"  downloaded {f} -> {p}", flush=True)
        texts.extend(pq.read_table(p).column("text").to_pylist())
    big = "\n\n".join(texts)
    ids = enc.encode(big)
    ids = [x for x in ids if x < MAXTOK]
    a = np.asarray(ids, dtype=np.int64)
    out = os.path.join(data_dir(), f"{name}.npy")
    os.makedirs(data_dir(), exist_ok=True)
    np.save(out, a)
    print(f"{name}: {len(a)} tokens  max={a.max()}  -> {out}", flush=True)


if __name__ == "__main__":
    cache = os.path.join(data_dir(), ".hf_cache")
    os.makedirs(cache, exist_ok=True)
    for name, files in SPLIT_FILES.items():
        parquet_to_tokens(files, name, cache)
    print("done", flush=True)