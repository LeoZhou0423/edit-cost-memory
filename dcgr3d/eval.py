"""Evaluation helpers shared across training scripts and ablations."""
import math
import torch
import torch.nn.functional as F

from dcgr3d.data import VOCAB, make_wmask


def eval_mqar(model, gen, m_list, dev, bs=128):
    """Evaluate MQAR top-1 recall at several store sizes ``m``, each reported at
    three query distances (d0, dmid, dfar). Sets ``model.alpha = 1.0`` (memory
    readout on)."""
    model.eval(); res = {}
    model.alpha = 1.0
    with torch.no_grad():
        for m in m_list:
            r = {}
            for tag, d in [("d0", 0), ("dmid", (m-1)//2), ("dfar", m-1)]:
                seq, tgt, _ = gen(bs, m, d, dev)
                wm = make_wmask(tgt, seq.shape[1])
                out = model(seq, wm)
                pred = out[:, 2*m+1].argmax(-1)
                r[tag] = round((pred == tgt[:, 2*m+1]).float().mean().item(), 3)
            res[f"m{m}"] = r
    model.train(); return res


def eval_ppl(model, data, L, dev, n=4, alpha=None):
    model.eval()
    if alpha is not None:
        model.alpha = alpha
    tot = 0.0; mt = 0
    with torch.no_grad():
        for _ in range(n):
            seq = data(8)
            out = model(seq)
            tot += F.cross_entropy(out[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1),
                                   reduction='sum').item()
            mt += 8 * (L - 1)
    model.train()
    if not math.isfinite(tot) or tot / mt > 88 or tot < 0:
        return float('inf')
    return math.exp(tot / mt)


def eval_ppl_both(model, data, L, dev, n=8):
    """同一批序列上做双读数: 记忆开(full) vs 关(lm).
    因触发门控保证 dense(无 PROBE) 记忆恒0, 理论上 ppl_full==ppl_lm;
    共用同一批数据消除采样噪声, 直接证明记忆对 LM 零污染."""
    model.eval()
    tot_full = 0.0; tot_lm = 0.0; mt = 0
    with torch.no_grad():
        for _ in range(n):
            seq = data(8)
            mt += 8 * (L - 1)
            model.alpha = 1.0; of = model(seq)
            model.alpha = 0.0; ol = model(seq)
            tot_full += F.cross_entropy(of[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1),
                                        reduction='sum').item()
            tot_lm += F.cross_entropy(ol[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1),
                                      reduction='sum').item()
    model.train()
    full = math.exp(tot_full / mt) if math.isfinite(tot_full) and tot_full / mt < 88 else float('inf')
    lm = math.exp(tot_lm / mt) if math.isfinite(tot_lm) and tot_lm / mt < 88 else float('inf')
    return full, lm


def eval_wiki(model, tok, L, dev, alpha=None, n_seq=64):
    """Evaluate dense perplexity on a pre-tokenized wiki split (``tok``)."""
    model.eval()
    if alpha is not None:
        model.alpha = alpha
    tot = 0.0; mt = 0
    rs = None
    n = len(tok); ntok = (n // L) * L
    W = tok[:ntok].view(-1, L)
    nseq = W.shape[0]
    # deterministic sampling of a subset of non-overlapping windows
    import numpy as np
    rs = np.random.RandomState(1)
    picks = rs.permutation(nseq)[:min(n_seq, nseq)]
    bs = 64
    with torch.no_grad():
        for s in range(0, len(picks), bs):
            idx = picks[s:s+bs]
            seq = W[idx].to(dev)
            out = model(seq)
            tot += F.cross_entropy(out[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1),
                                   reduction="sum").item()
            mt += seq.shape[0] * (L - 1)
    model.train()
    return math.exp(tot / mt) if math.isfinite(tot) and tot / mt < 88 else float("inf")
