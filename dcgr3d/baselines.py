"""Parameter-matched baseline models (SSM / linear attention / Transformer).

Used by the head-to-head comparison to show that a single compressed state of
matched size cannot simultaneously reach exact associative recall *and* low
dense perplexity — the property the trigger-gated stable-key cell provides.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dcgr3d.data import VOCAB


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def gpt_init(module, std=0.02):
    """GPT-2 style initialisation for every Linear and Embedding module.

    PyTorch's defaults give Embedding weights ``N(0, 1)``.  Combined with a tied
    output head at width 512 that puts the initial logits at scale ~O(dm), so the
    softmax starts saturated: the measured initial retrieval loss is ~300 against
    ``ln(V) ~ 10.8`` for a uniform distribution.  Almost all of the early gradient
    then goes into rescaling the output instead of shaping the retrieval circuit,
    which is not a property of the architecture.

    Both the earlier baselines in this project and standard MQAR harnesses use
    ``std=0.02``, so the PyTorch default here was an artefact of the rewrite.
    ``benchmark_mqar_fair.py --emb-init`` exposes this so the comparison can be
    repeated with the same setting for every family.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, std)


class MambaScan(nn.Module):
    """选择性状态空间参考: 对角状态 h∈R^{d_state}, 输入相关增量 dt 与
    (B,C) 投影, 平行联想扫描 out[i]=a[i]·out[i-1]+b[i]。
    a_t = exp(softplus(A)·dt) ∈ (0,1): 选择性遗忘; 增量随输入变化。"""
    def __init__(self, Dm=64, d_state=16):
        super().__init__()
        self.D = Dm; self.ds = d_state
        self.x_proj = nn.Linear(Dm, Dm)            # 输入混合
        self.dt_proj = nn.Linear(Dm, d_state)      # 输入相关增量
        self.B_proj = nn.Linear(Dm, d_state)
        self.C_proj = nn.Linear(Dm, d_state)
        self.logA = nn.Parameter(torch.log(0.5 * torch.ones(d_state)))
        self.out_proj = nn.Linear(Dm, Dm)
    def forward(self, u):
        B, L, Dm = u.shape
        x = F.silu(self.x_proj(u))                                   # (B,L,D)
        dt = torch.exp(self.dt_proj(x).clamp(min=-8, max=8))         # (B,L,ds)
        A = torch.exp(self.logA)                                     # (ds,)
        alpha = torch.exp(A.unsqueeze(0).unsqueeze(0) * (-dt))       # (B,L,ds)
        beta = torch.einsum('bld,bls->blsd', x, self.B_proj(x))      # (B,L,ds,D)
        h = scan_lin(alpha, beta)                                    # (B,L,ds,D)
        y = torch.einsum('bldv,bld->blv', h, self.C_proj(x))        # (B,L,D)
        return self.out_proj(y)


def scan_lin(alpha, beta):
    """out[i]=alpha[i]*out[i-1]+beta[i], alpha,beta:(B,ds,rest)。并行扫。
    alpha∈(0,1] 为输入相关的选择性保持系数（Mamba 式增量）。"""
    alpha = alpha.contiguous().clone(); beta = beta.contiguous().clone()
    S = alpha.shape[1]
    step = 1
    while step < S:
        a_s = torch.cat([torch.ones_like(alpha[:, :step]), alpha[:, :-step]], 1)
        b_s = torch.cat([torch.zeros_like(beta[:, :step]), beta[:, :-step]], 1)
        ao = alpha; alpha = ao * a_s; beta = ao.unsqueeze(-1) * b_s + beta; step *= 2
    return beta


class MambaLM(nn.Module):
    """Mamba-2 结构参考; alpha 为占位(对应 eval_mqar 设置)保持接口一致.
    逐层 gradient checkpoint 省显存(数学不变)."""
    def __init__(self, V, dm, d_state, nl):
        super().__init__()
        self.alpha = 1.0; self.dm = dm
        self.emb = nn.Embedding(V, dm)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"norm": nn.LayerNorm(dm),
                           "ssm": MambaScan(dm, d_state)}) for _ in range(nl)])
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V, bias=False)
        self.head.weight = self.emb.weight
        self.apply(gpt_init)
    def forward(self, ids, wmask=None):
        x = self.emb(ids)
        for blk in self.layers:
            x = checkpoint(lambda t: t + blk["ssm"](blk["norm"](t)), x,
                           use_reentrant=False)
        return self.head(self.ln(x))


class TFBlock(nn.Module):
    """标准 causal self-attention block (FlashAttention via SDPA). 可选 sliding-window."""
    def __init__(self, dm, nh, window=None):
        super().__init__()
        self.dm = dm; self.nh = nh; self.hd = dm // nh; self.window = window
        self.ln1 = nn.LayerNorm(dm)
        self.qkv = nn.Linear(dm, 3 * dm, bias=False)
        self.out = nn.Linear(dm, dm, bias=False)
        self.ln2 = nn.LayerNorm(dm)
        self.fc1 = nn.Linear(dm, 4 * dm); self.fc2 = nn.Linear(4 * dm, dm)

    def _attn_mask(self, n, dev):
        # sliding window 相对注意力掩码; None => full causal
        if self.window is None:
            return None
        mask = torch.full((n, n), -1e4)
        j = torch.arange(n).unsqueeze(0)
        i = torch.arange(n).unsqueeze(1)
        causal = (i >= j)
        window = (i - j <= self.window)
        return torch.where(causal & window, torch.zeros(()), mask).to(dev)

    def forward(self, x):
        B, T, D = x.shape
        z = self.ln1(x)
        qkv = self.qkv(z).reshape(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = self._attn_mask(T, x.device)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=(mask is None))  # flash
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.out(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TransformerLM(nn.Module):
    """因果 Transformer 基线 (FlashAttention), alpha 占位接口一致. 逐层 checkpoint.

    Learned positional embeddings are required here: without any position
    representation the baseline cannot express the key-followed-by-value
    relation used by MQAR, making it an invalid comparison.
    """
    def __init__(self, V, dm, nh, nl, window=None, max_seq_len=2048):
        super().__init__()
        self.alpha = 1.0; self.dm = dm
        self.emb = nn.Embedding(V, dm)
        self.pos = nn.Embedding(max_seq_len, dm)
        self.blocks = nn.ModuleList([TFBlock(dm, nh, window) for _ in range(nl)])
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V, bias=False)
        self.head.weight = self.emb.weight
        self.apply(gpt_init)
    def forward(self, ids, wmask=None):
        if ids.shape[1] > self.pos.num_embeddings:
            raise ValueError(f"sequence length {ids.shape[1]} exceeds positional capacity "
                             f"{self.pos.num_embeddings}")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.emb(ids) + self.pos(positions)[None, :, :]
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        return self.head(self.ln(x))


class LinAtt(nn.Module):
    """线性注意力 (Based/JRT-RNN 风格). 无限累加 h = Σ k⊗v (无遗忘, 对精确
    回忆理论上最友好) + Torch cumsum 全向量化; alpha 占位. 逐层 checkpoint."""
    def __init__(self, V, dm, dk_state, dv_state, nl):
        super().__init__()
        self.alpha = 1.0; self.dm = dm; self.dk = dk_state
        self.emb = nn.Embedding(V, dm)
        self.layers = nn.ModuleList()
        for _ in range(nl):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dm),
                "q": nn.Linear(dm, dk_state, bias=False),
                "k": nn.Linear(dm, dk_state, bias=False),
                "v": nn.Linear(dm, dv_state, bias=False),
                "out": nn.Linear(dv_state, dm, bias=False),
                "ln2": nn.LayerNorm(dm),
            }))
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V, bias=False)
        self.head.weight = self.emb.weight
        self.apply(gpt_init)
    def forward(self, ids, wmask=None):
        x = self.emb(ids)
        B, T, D = x.shape
        for blk in self.layers:
            def fwd(t):
                z = blk["ln1"](t)                                  # (B,T,D)
                q = blk["q"](z); k = blk["k"](z); v = blk["v"](z)  # (B,T,dk/dv)
                kv = k.unsqueeze(-1) * v.unsqueeze(-2)             # (B,T,dk,dv)
                cumkv = torch.cumsum(kv, dim=1)
                out = torch.einsum('bti,btij->btj', q, cumkv)      # (B,T,dv)
                return t + blk["ln2"](blk["out"](out))
            x = checkpoint(fwd, x, use_reentrant=False)
        return self.head(self.ln(x))


def build(which, dm, seed, emb_init="gpt", depth=None, state_dim=None):
    """Build a baseline of a given family at width ``dm``.

    With ``depth=None`` the depth is chosen to match the full model's
    non-embedding parameter budget (~54M), which at ``dm=512`` gives 17
    Transformer blocks.  That rule is a parameter-count convention, not a
    validated choice, and published MQAR results use two layers; a narrow
    17-block stack with a tied output head may simply be a worse learner at the
    same parameter count.  ``depth`` overrides it so depth can be tuned and
    reported like any other hyperparameter.

    ``state_dim`` overrides the recurrent state width of the fixed-state
    families (``d_state`` for Mamba-2, ``dk_state=dv_state`` for linear
    attention).  That is the axis a state-capacity frontier is plotted against.

    ``emb_init`` is ``gpt`` (GPT-2 style, ``std=0.02``) by default because the
    PyTorch defaults leave the tied output head saturated at initialisation; see
    :func:`gpt_init`.  Pass ``default`` only to reproduce the earlier
    mis-initialised artifacts.
    """
    torch.manual_seed(seed)
    target_non_emb = 54_000_000
    if which == "mamba2":
        # MambaScan(D=1024,d_state=32) 单层约 2.20M
        nl = depth or max(1, int(target_non_emb / 2_200_000))
        model = MambaLM(VOCAB, dm, d_state=state_dim or 32, nl=nl)
    elif which == "linatt":
        # LinAtt 向量化单层(dk=dv=512)约 2.1M
        nl = depth or max(1, int(target_non_emb / 2_100_000))
        dim = state_dim or 512
        model = LinAtt(VOCAB, dm, dk_state=dim, dv_state=dim, nl=nl)
    else:
        # Transformer (FlashAttention). 单块 qkv+out+mlp ~4D²+8D²=12D²(@D=1024≈12.6M)
        nl = depth or max(1, int(target_non_emb / (12 * dm ** 2)))
        model = TransformerLM(VOCAB, dm, nh=8, nl=nl, window=None)
    if emb_init != "gpt":
        model.apply(_torch_default_init)
        # The output head shares its weight tensor with the token embedding, and
        # ``apply`` visits the head last, so the head's Linear initialisation
        # would otherwise overwrite the embedding scale.  The historical runs had
        # the embedding's N(0, 1) scale on the shared tensor, so restore it.
        with torch.no_grad():
            if hasattr(model, "emb"):
                nn.init.normal_(model.emb.weight, 0.0, 1.0)
    return model


def _torch_default_init(module):
    """Restore PyTorch's default Linear/Embedding initialisation."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            bound = 1 / math.sqrt(module.weight.shape[1])
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 1.0)
