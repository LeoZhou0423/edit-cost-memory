"""dcgr3d — "Load-One Zero-Interference Associative Recall in a Single Linear
Recurrent Language Model: Frozen Stable Keys, a Trigger Gate, and Decoupled
Training".

A linear recurrent language model that couples a Mamba-2-style selective-SSM
backbone with a trigger-gated, frozen-stable-key memory cell. Training is
decoupled into an MQAR recall phase (memory readout on) and a dense language
modeling phase (memory readout off), so a single checkpoint achieves exact
associative recall and low perplexity with zero interference between the two.

Modules
-------
- :mod:`dcgr3d.data`       — MQAR generation and wiki-token loading
- :mod:`dcgr3d.memory`     — the trigger-gated stable-key memory cell
- :mod:`dcgr3d.ssm`        — GatedScan / SelectiveSSM backbones
- :mod:`dcgr3d.model`      — MemLM12/13/14 model definitions
- :mod:`dcgr3d.eval`       — recall / perplexity evaluation
- :mod:`dcgr3d.baselines`  — parameter-matched SSM / lin-attn / transformer
                             baselines
"""
from dcgr3d.data import (VOCAB, make_mqar, make_mqar_canonical, make_ear,
                         make_wmask, mqar_token_pools, mqar_write_mask,
                         dense_data, load_tokens, wiki_loader)
from dcgr3d.memory import (StableMemCell, DeltaMemCell, Tensor3DMemCell,
                           EditMemCell, GatedEditMemCell, make_cell, CELL_KINDS,
                           orth_init, orth_penalty)
from dcgr3d.ssm import GatedScan, SelectiveSSM
from dcgr3d.model import (Mamba2GatedBlock, MemLM12, MemLM13, MemLM14,
                          count_params, safe_step)
from dcgr3d.eval import eval_mqar, eval_ppl, eval_ppl_both, eval_wiki
from dcgr3d.baselines import (MambaScan, scan_lin, MambaLM, TransformerLM, LinAtt,
                              build, gpt_init)

__all__ = [
    "VOCAB", "make_mqar", "make_mqar_canonical", "make_ear", "make_wmask",
    "mqar_token_pools", "mqar_write_mask", "dense_data", "load_tokens", "wiki_loader",
    "StableMemCell", "DeltaMemCell", "Tensor3DMemCell", "EditMemCell",
    "GatedEditMemCell", "make_cell", "CELL_KINDS", "orth_init", "orth_penalty",
    "GatedScan", "SelectiveSSM",
    "Mamba2GatedBlock", "MemLM12", "MemLM13", "MemLM14", "count_params", "safe_step",
    "eval_mqar", "eval_ppl", "eval_ppl_both", "eval_wiki",
    "MambaScan", "scan_lin", "MambaLM", "TransformerLM", "LinAtt", "build",
    "gpt_init",
]

__version__ = "1.0.0"
