"""Model definitions for the "Load-One Zero-Interference Associative Recall"
paper.

The final architecture is :class:`MemLM14`: a Mamba-2-style linear backbone
(``Mamba2GatedBlock`` = ``SelectiveSSM`` + siLU output gate) coupled with the
trigger-gated :class:`~dcgr3d.memory.StableMemCell` readout. Training is
decoupled: phase 1 (``alpha=1``) trains exact MQAR recall, phase 2 (``alpha=0``)
trains pure dense language modeling with the memory readout switched off, so the
same checkpoint keeps both capabilities with zero interference.

:class:`MemLM13` and :class:`MemLM12` are earlier iterations of the same
decoupled design, kept for reproducibility of the ablation baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dcgr3d.ssm import SelectiveSSM, GatedScan
from dcgr3d.memory import StableMemCell, make_cell


class Mamba2GatedBlock(nn.Module):
    """Mamba2 风格带门控块: 选择性扫描 + siLU 双支路 + 输出门.
    保持线性循环(无注意力). 数值扫描复用 SelectiveSSM 的稳定实现."""
    def __init__(self, d, N):
        super().__init__()
        self.d = d; self.N = N
        self.norm = nn.LayerNorm(d)
        self.ssm = SelectiveSSM(d, N)        # 向量态选择扫描 (含 residual+ln)
        self.gate = nn.Linear(d, d)          # 输出门 z
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    def forward(self, x):
        z = torch.sigmoid(self.gate(self.norm(x)))   # (B,S,D) 输出门
        h = self.ssm(x)                              # 扫描输出 (含 residual)
        return z * h + (1 - z) * x                   # 门控融合


class MemLM14(nn.Module):
    def __init__(self, V, dm, dk, dv, nl=6, n_ssm=3, N=32, lam0=0.999,
                 cell_kind="stable"):
        super().__init__()
        self.alpha = 1.0; self.dm = dm
        self.cell_kind = cell_kind
        self.emb = nn.Embedding(V, dm)
        self.ssms = nn.ModuleList([Mamba2GatedBlock(dm, N) for _ in range(n_ssm)])
        self.blocks = nn.ModuleList()
        for _ in range(nl):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dm), "ln2": nn.LayerNorm(dm),
                "fc1": nn.Linear(dm, 4*dm), "fc2": nn.Linear(4*dm, dm),
            }))
        self.hn = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V, bias=False)
        self.head.weight = self.emb.weight
        # Backbone, tied head and additive readout into logit space are held fixed
        # across cell kinds so that a comparison isolates the cell itself.
        self.cell = make_cell(cell_kind, V, dk, dv, lam_init=lam0, trigger=True)

    def hidden(self, ids):
        """Backbone hidden state feeding the memory cell.

        Exposed so analysis code can replay a cell's own gate/readout path on a
        sequence without going through the tied output head.
        """
        x = self.emb(ids)
        for s in self.ssms:
            x = s(x)
        for b in self.blocks:
            x = x + b["fc2"](F.gelu(b["fc1"](b["ln1"](x))))
        return x

    def forward(self, ids, wmask=None, ops=None):
        x = self.hidden(ids)
        dense = self.head(self.hn(x))
        if self.alpha <= 0.0:
            return dense
        # ``ops`` carries the edit stream (0 none / 1 write / 2 erase) for cells
        # that expose editing; the plain cells ignore it and use the positional
        # write mask instead.
        if ops is None:
            return dense + self.alpha * self.cell(ids, x, wmask)
        return dense + self.alpha * self.cell(ids, x, wmask, ops)


class MemLM13(nn.Module):
    def __init__(self, V, dm, dk, dv, nl=4, n_ssm=2, N=16, lam0=0.999):
        super().__init__()
        self.alpha = 1.0; self.dm = dm
        self.emb = nn.Embedding(V, dm)
        self.ssms = nn.ModuleList([SelectiveSSM(dm, N) for _ in range(n_ssm)])
        self.blocks = nn.ModuleList()
        for _ in range(nl):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dm), "ln2": nn.LayerNorm(dm),
                "fc1": nn.Linear(dm, 4*dm), "fc2": nn.Linear(4*dm, dm),
            }))
        self.hn = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V, bias=False)
        self.head.weight = self.emb.weight
        self.cell = StableMemCell(V, dk, dv, lam_init=lam0, trigger=True)

    def forward(self, ids, wmask=None):
        x = self.emb(ids)
        for s in self.ssms:
            x = s(x)
        for b in self.blocks:
            x = x + b["fc2"](F.gelu(b["fc1"](b["ln1"](x))))
        dense = self.head(self.hn(x))
        if self.alpha <= 0.0:
            return dense
        return dense + self.alpha * self.cell(ids, x, wmask)


class MemLM12(nn.Module):
    def __init__(self, V, dm, dk, dv, nl=2, alpha=1.0, lam0=0.999, trigger=True,
                 with_cell=True):
        super().__init__()
        self.alpha = alpha; self.trigger = trigger; self.dm = dm
        self.with_cell = with_cell
        self.emb = nn.Embedding(V, dm)
        self.scan = GatedScan(dm)
        if with_cell:
            self.cell = StableMemCell(V, dk, dv, lam_init=lam0, trigger=trigger)
        self.blocks = nn.ModuleList()
        for _ in range(nl):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dm), "ln2": nn.LayerNorm(dm),
                "fc1": nn.Linear(dm, 4*dm), "fc2": nn.Linear(4*dm, dm),
            }))
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids, wmask=None):
        x = self.emb(ids)
        x = self.scan(x)
        for b in self.blocks:
            x = x + b["fc2"](F.gelu(b["fc1"](b["ln1"](x))))
        dense = self.head(self.ln(x))
        if (not self.with_cell) or self.alpha <= 0.0:
            return dense
        return dense + self.alpha * self.cell(ids, x, wmask)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def safe_step(opt, model, max_norm=1.0):
    """NaN 安全梯度更新: 先把 inf/nan 的梯度置 0, 再全局 clip 后 step.
    避免单个 inf 梯度在被 clip_grad_norm 乘 0 后变成 nan 污染整批权重."""
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad
            if not torch.isfinite(g).all():
                g[~torch.isfinite(g)] = 0.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    opt.step()
