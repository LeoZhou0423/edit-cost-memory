"""The trigger-gated stable-key memory cell.

``StableMemCell`` is the content-addressed readout of the model. It binds each
input token to a *frozen* orthogonal key (``key = token``) and accumulates a
second-order outer-product state; readout is content-addressed by the query key.
A **trigger gate** keeps the readout at zero until a reserved ``PROBE`` token is
seen, so the memory contributes nothing during dense language modeling and
exactly the recalled value once the probe activates it — this is the
"zero-interference" property.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def orth_init(model, KIDX, dk, dev):
    """Initialize the key rows indexed by ``KIDX`` to a QR-orthonormal basis.

    ``KIDX`` selects the reserved MQAR key tokens; every other row keeps the
    default embedding initialization.
    """
    with torch.no_grad():
        n = len(KIDX)
        X = torch.randn(n, dk, device=dev)
        Q, _ = torch.linalg.qr(X)
        model.key.weight.data[KIDX] = Q


def orth_penalty(model, KIDX, lam=1.0):
    """Orthogonality regularizer on the (learnable) key rows indexed by KIDX.

    Pulls the key gram matrix toward identity; only has a gradient when the key
    embedding is trainable (the frozen-key ablation sets ``requires_grad`` off).
    """
    Ksub = model.key(KIDX)
    G = Ksub @ Ksub.t()
    n = Ksub.shape[0]
    return lam * ((G - torch.eye(n, device=G.device)) ** 2).mean()


class StableMemCell(nn.Module):
    """稳定 key 记忆胞: key=token自身(冻结正交), 二阶外积叠加写, 内容寻址读."""
    def __init__(self, V, dk, dv, lam_init=0.999, trigger=True):
        super().__init__()
        self.V = V
        self.trigger = trigger
        self.dk = dk; self.dv = dv
        self.key = nn.Embedding(V, dk)
        self.val = nn.Embedding(V, dv)
        self.loglam = nn.Parameter(torch.tensor(math.log(max(lam_init, 1e-3) /
                                                          (1 - max(lam_init, 1e-3)))))
    def forward(self, ids, x, wmask=None):
        """触发式记忆(trigger-gated): 记忆读出只在进入记忆模式后激活.
        激活信号 = 序列中出现 PROBE 标记 token(V-1). MQAR 有 PROBE =>
        记忆在 probe 位激活召回; dense 文本无 PROBE => 记忆全程输出0,
        完全不污染 LM => 端到端同一 checkpoint 同时保留召回与 LM.
        trigger=False 时退化为每位置激活(消融对照): 记忆污染 LM 应在 dense 出现."""
        B, S = ids.shape
        lam = torch.sigmoid(self.loglam)
        S_t = torch.zeros(B, self.dk, self.dv, device=x.device)
        kp = None; mem = []
        probe = self.V - 1
        active = torch.zeros(B, dtype=torch.bool, device=x.device)
        for t in range(S):
            k = self.key(ids[:, t])
            a = kp if kp is not None else k
            write = True if wmask is None else bool(wmask[:, t].any())
            if write:
                v = self.val(ids[:, t])
                S_t = lam * S_t + torch.einsum('bi,bd->bid', a, v)
            active = active | (ids[:, t] == probe)   # 见到触发标记 => 进入记忆模式
            s = torch.einsum('bi,bid->bd', k, S_t)
            out = s @ self.val.weight.t()       # (B,V) 记忆读出 logits
            mem.append(torch.where(active[:, None], out, torch.zeros_like(out)) if self.trigger else out)
            kp = k
        return torch.stack(mem, 1)


class DeltaMemCell(nn.Module):
    """Delta-rule write into the same fixed state.

    Structurally identical to :class:`StableMemCell` except that a write first
    reads the state at the incoming address and stores only the residual::

        pred  = a^T S_{t-1}
        S_t   = lam * S_{t-1} + beta * a (v - lam * pred)^T

    which is one step of online gradient descent on ``||S^T a - v||^2``.  The
    point of including it is that the read is unchanged, so the comparison
    isolates the write rule at a fixed state shape.
    """
    def __init__(self, V, dk, dv, lam_init=0.999, trigger=True, beta_logit=4.0):
        super().__init__()
        self.V = V
        self.trigger = trigger
        self.dk = dk; self.dv = dv
        self.key = nn.Embedding(V, dk)
        self.val = nn.Embedding(V, dv)
        self.loglam = nn.Parameter(torch.tensor(math.log(max(lam_init, 1e-3) /
                                                          (1 - max(lam_init, 1e-3)))))
        # beta_logit = 4.0 puts the write step at sigmoid(4) = 0.982, close to the
        # exact delta rule (beta = 1) while staying inside the sigmoid's linear
        # range so the step size remains learnable, as in DeltaNet.
        self.logit_beta = nn.Parameter(torch.tensor(float(beta_logit)))

    def forward(self, ids, x, wmask=None):
        B, S = ids.shape
        lam = torch.sigmoid(self.loglam)
        beta = torch.sigmoid(self.logit_beta)
        S_t = torch.zeros(B, self.dk, self.dv, device=x.device)
        kp = None; mem = []
        probe = self.V - 1
        active = torch.zeros(B, dtype=torch.bool, device=x.device)
        for t in range(S):
            k = self.key(ids[:, t])
            a = kp if kp is not None else k
            write = True if wmask is None else bool(wmask[:, t].any())
            if write:
                v = self.val(ids[:, t])
                pred = torch.einsum('bi,bid->bd', a, S_t)
                S_t = lam * S_t + beta * torch.einsum('bi,bd->bid', a, v - lam * pred)
            active = active | (ids[:, t] == probe)
            s = torch.einsum('bi,bid->bd', k, S_t)
            out = s @ self.val.weight.t()
            mem.append(torch.where(active[:, None], out, torch.zeros_like(out)) if self.trigger else out)
            kp = k
        return torch.stack(mem, 1)


class Tensor3DMemCell(nn.Module):
    """Second-order (bilinear) address with a trilinear read.

    The state is a three-tensor ``S in R^{dk x dk x dv}``.  A write binds the
    address to itself and to the value, ``a (x) a (x) v``, and the read contracts
    the query twice::

        out = sum_ij k_i k_j S[i, j, :]

    Every stored pair therefore enters the read weighted by ``(k . a_s)^2``
    instead of ``(k . a_s)``, so cross-talk between addresses is suppressed from
    the coherence ``mu`` to ``mu^2``.  The price is an address block that is
    ``dk`` times larger than the linear cell's for the same address dimension, so
    at a fixed state budget the bilinear cell can only afford ``dk`` smaller.

    Ported from ``archive/repo_legacy/scripts/experiments/arch3d/native3d.py``
    (:class:`Tensor3DState`), with two deliberate simplifications so that the
    comparison isolates the address order: the per-axis cross gate is replaced by
    a single scalar decay matching the other cells, and keys are unit-normalised
    instead of softmax-normalised so that the same orthonormal initialisation
    applies to all cells.
    """
    def __init__(self, V, dk, dv, lam_init=0.999, trigger=True):
        super().__init__()
        self.V = V
        self.trigger = trigger
        self.dk = dk; self.dv = dv
        self.key = nn.Embedding(V, dk)
        self.val = nn.Embedding(V, dv)
        self.loglam = nn.Parameter(torch.tensor(math.log(max(lam_init, 1e-3) /
                                                          (1 - max(lam_init, 1e-3)))))

    def forward(self, ids, x, wmask=None):
        B, S = ids.shape
        lam = torch.sigmoid(self.loglam)
        S_t = torch.zeros(B, self.dk, self.dk, self.dv, device=x.device)
        kp = None; mem = []
        probe = self.V - 1
        active = torch.zeros(B, dtype=torch.bool, device=x.device)
        for t in range(S):
            k = self.key(ids[:, t])
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-8)
            a = kp if kp is not None else k
            write = True if wmask is None else bool(wmask[:, t].any())
            if write:
                v = self.val(ids[:, t])
                S_t = lam * S_t + torch.einsum('bi,bj,bd->bijd', a, a, v)
            active = active | (ids[:, t] == probe)
            s = torch.einsum('bi,bj,bijd->bd', k, k, S_t)
            out = s @ self.val.weight.t()
            mem.append(torch.where(active[:, None], out, torch.zeros_like(out)) if self.trigger else out)
            kp = k
        return torch.stack(mem, 1)


class EditMemCell(nn.Module):
    """A cell that exposes erase/overwrite as first-class operations.

    The state update rules are the ones above; what is new is that the write
    stream may also carry *edits*.  ``ops`` is an integer tensor with

    - ``0`` do nothing,
    - ``1`` write ``key(ids[t-1]) (x) val(ids[t])``,
    - ``2`` erase the binding addressed by ``key(ids[t])``.

    An erase is implemented as *read then subtract*::

        S  <-  S - a (a^T S)^T            a = key(ids[t])

    which is exact and needs no decay accounting.  The read at the erased
    address becomes ``a^T S - ||a||^2 (a^T S)``, i.e. exactly zero for any
    normalised ``a`` and any ``lam``; the read at another address ``p`` changes
    by ``(a_p . a) (a^T S)``, which vanishes if and only if ``a_p . a = 0``.
    That second factor is the whole reason this class exists: the write rule
    (``add`` vs ``delta``) can shrink ``(a^T S)`` but cannot zero ``(a_p . a)``,
    because the latter is a property of the addressing code rather than of the
    write rule.

    ``write_rule='delta'`` keeps the same erase and makes the write store only
    the residual, matching :class:`DeltaMemCell`.  ``bilinear=True`` uses the
    second-order address of :class:`Tensor3DMemCell` for both write and read.
    """

    def __init__(self, V, dk, dv, lam_init=0.999, trigger=True,
                 write_rule="add", bilinear=False, beta_logit=4.0):
        super().__init__()
        if write_rule not in ("add", "delta"):
            raise ValueError(f"unknown write_rule {write_rule!r}")
        self.V = V
        self.trigger = trigger
        self.dk = dk
        self.dv = dv
        self.write_rule = write_rule
        self.bilinear = bilinear
        self.key = nn.Embedding(V, dk)
        self.val = nn.Embedding(V, dv)
        self.loglam = nn.Parameter(torch.tensor(math.log(max(lam_init, 1e-3) /
                                                          (1 - max(lam_init, 1e-3)))))
        self.logit_beta = nn.Parameter(torch.tensor(float(beta_logit)))

    def _unit(self, k):
        """Normalise an address to unit length.

        Applied to every key in both the write and the read, so the stored
        contribution of binding ``j`` is ``a_j (x) v_j`` with ``||a_j|| = 1``.
        That is what makes read-then-subtract exact rather than exact up to a
        factor of ``||a_j||^2``, and it matches what
        :class:`Tensor3DMemCell` already does.  The address *directions* carry
        all the information here; the norms only scale the state.
        """
        return k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-8)

    def _read(self, a, S_t):
        if self.bilinear:
            return torch.einsum('bi,bj,bijd->bd', a, a, S_t)
        return torch.einsum('bi,bid->bd', a, S_t)

    def _add(self, a, v, lam, S_t):
        if self.bilinear:
            return lam * S_t + torch.einsum('bi,bj,bd->bijd', a, a, v)
        return lam * S_t + torch.einsum('bi,bd->bid', a, v)

    def _erase(self, a, S_t):
        """``S <- S - a (a^T S)^T``, with no decay applied.

        The decay is charged at write positions only, so an erase removes
        exactly the binding it addresses: that binding contributes
        ``lam^d a (x) v`` to the state and reads back as ``lam^d v``, so
        subtracting ``a (x) (a^T S)`` removes it in full without ever needing to
        know ``lam`` or how many writes have elapsed.  Keeping the erase free of
        decay also makes the erase and no-erase arms of the ablation differ by
        exactly the removed binding and nothing else.
        """
        removed = self._read(a, S_t)
        if self.bilinear:
            return S_t - torch.einsum('bi,bj,bd->bijd', a, a, removed)
        return S_t - torch.einsum('bi,bd->bid', a, removed)

    def forward(self, ids, x=None, wmask=None, ops=None):
        mem, _ = self._rollout(ids, wmask, ops, want_readout=True)
        return mem

    @torch.no_grad()
    def state_after(self, ids, ops=None, wmask=None, x=None):
        """Return the state after replaying ``ids`` under ``ops``.

        Exposed so the analysis code measures the *same* state the model used,
        including the erase and overwrite operations, rather than re-deriving
        one from the stored pairs.  ``x`` is accepted and ignored so that the
        analysis code can call every cell kind the same way.
        """
        _, state = self._rollout(ids, wmask, ops, want_readout=False)
        return state

    def _rollout(self, ids, wmask=None, ops=None, want_readout=True):
        B, S = ids.shape
        device = ids.device
        if ops is None:
            ops = (wmask > 0).long() if wmask is not None else torch.ones(
                B, S, dtype=torch.long, device=device)
        lam = torch.sigmoid(self.loglam)
        beta = torch.sigmoid(self.logit_beta)
        shape = (B, self.dk, self.dk, self.dv) if self.bilinear else (B, self.dk, self.dv)
        S_t = torch.zeros(*shape, device=device)
        kp = None
        mem = []
        probe = self.V - 1
        active = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(S):
            k = self._unit(self.key(ids[:, t]))
            a = self._unit(kp) if kp is not None else k
            op = ops[:, t]
            if bool((op == 2).any()):
                S_t = self._erase(k, S_t)
            if bool((op == 1).any()):
                v = self.val(ids[:, t])
                if self.write_rule == "delta":
                    pred = self._read(a, S_t)
                    S_t = self._add(a, beta * (v - lam * pred), lam, S_t)
                else:
                    S_t = self._add(a, v, lam, S_t)
            # an edit implies the memory is in addressed mode, so it also arms
            # the readout gate in the same way the probe token does
            active = active | (ids[:, t] == probe) | (op == 2)
            if want_readout:
                out = self._read(k, S_t) @ self.val.weight.t()
                mem.append(torch.where(active[:, None], out, torch.zeros_like(out))
                           if self.trigger else out)
            kp = k
        readout = torch.stack(mem, 1) if want_readout else None
        return readout, S_t


class GatedEditMemCell(nn.Module):
    """Channel-wise decoupled erase/write gates on top of the delta rule.

    This is the update rule of Gated DeltaNet-2 (arXiv:2605.22791) written in this
    project's notation, introduced here as the *strongest available gate-based
    baseline* for editing rather than as a contribution::

        S_t = (I - k_t (b_t * k_t)^T) diag(d_t) S_{t-1} + k_t (w_t * v_t)^T

    where ``b`` is a channel-wise erase gate on the key axis, ``w`` a
    channel-wise write gate on the value axis, and ``d`` a channel-wise decay.
    With ``b = w = beta`` and scalar ``d`` this reduces first to KDA and then to
    Gated DeltaNet, exactly as in that paper.

    The reason it belongs in this study is what the gates *cannot* do.  Erasing
    the binding at address ``k`` writes ``-k (x) (e^T S)`` into the state, with
    ``e = b * k`` the gated erase direction.  The change this causes at another
    address ``a_p`` is therefore

        -(a_p . k) (e^T S)

    and the gate ``b`` appears only in the second factor.  Collateral damage is
    the product of a **geometric** factor ``(a_p . k)``, which is a property of
    the addressing code and which no gate can touch, and an **amplitude** factor
    ``(e^T S)``, which the gate can shrink.  Whether that distinction is real or
    nominal is an empirical question, and it is the one this cell is here to
    answer: if per-channel gates move the locality off the code's coherence
    ``mu``, the field's direction is right and the framing here is wrong.
    """

    def __init__(self, V, dk, dv, lam_init=0.999, trigger=True, beta_logit=4.0,
                 gate_bias=(2.0, 2.0, 6.0)):
        super().__init__()
        self.V = V
        self.trigger = trigger
        self.dk = dk
        self.dv = dv
        self.key = nn.Embedding(V, dk)
        self.val = nn.Embedding(V, dv)
        self.gate = nn.Linear(dk + dv, 2 * dk + dv)
        self._gate_bias = tuple(gate_bias)
        self.init_gates()
        # Intervention hook (analysis only): when set to a float, the erase
        # gate is replaced by that constant everywhere it is used to erase, so
        # the erase direction can be forced without touching the write gate,
        # the decay, the addresses or anything else.  None means "as trained".
        self.erase_override = None
        self.loglam = nn.Parameter(torch.tensor(math.log(max(lam_init, 1e-3) /
                                                          (1 - max(lam_init, 1e-3)))))
        self.logit_beta = nn.Parameter(torch.tensor(float(beta_logit)))

    def init_gates(self):
        """Reset the gate to (almost) pass-through.

        The gates must start near one so the cell *begins* as a plain delta rule
        and any improvement attributable to gating is earned by training rather
        than handed over at initialisation.  Exposed separately because a generic
        weight initialiser applied to the whole model would otherwise overwrite
        it.
        """
        with torch.no_grad():
            nn.init.zeros_(self.gate.weight)
            dk, dv = self.dk, self.dv
            self.gate.bias[:dk] = self._gate_bias[0]
            self.gate.bias[dk:2 * dk] = self._gate_bias[1]
            self.gate.bias[2 * dk:] = self._gate_bias[2]

    def _unit(self, k):
        return k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-8)

    def _gates(self, tok, cur):
        """Gates for one step, driven by the addressed key and the incoming value.

        Returns ``(erase, decay, write)``, laid out as the gate's output is
        split: the erase gate and the decay are per key channel (``dk``) and the
        write gate is per value channel (``dv``).  At initialisation the weights
        are zero so the gates are constant and the cell is a plain delta rule.
        """
        feat = torch.cat([cur, self.val(tok)], dim=-1)
        g = torch.sigmoid(self.gate(feat))
        return g[:, :self.dk], g[:, self.dk:2 * self.dk], g[:, 2 * self.dk:]

    def _rollout(self, ids, x, wmask=None, ops=None, want_readout=True):
        B, S = ids.shape
        device = ids.device
        if ops is None:
            ops = (wmask > 0).long() if wmask is not None else torch.ones(
                B, S, dtype=torch.long, device=device)
        lam = torch.sigmoid(self.loglam)
        beta = torch.sigmoid(self.logit_beta)
        S_t = torch.zeros(B, self.dk, self.dv, device=device)
        kp = None
        mem = []
        probe = self.V - 1
        active = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(S):
            k = self._unit(self.key(ids[:, t]))
            a = self._unit(kp) if kp is not None else k
            v = self.val(ids[:, t])
            op = ops[:, t]
            if bool((op == 2).any()):
                # Erase: the address being removed is the current token, so the
                # erase gate must be computed from that address.
                b, _, _ = self._gates(ids[:, t], k)
                if self.erase_override is not None:
                    b = torch.full_like(b, float(self.erase_override))
                S_t = S_t - torch.einsum('bi,bd->bid', k,
                                         torch.einsum('bi,bid->bd', b * k, S_t))
            if bool((op == 1).any()):
                # Write: in this layout the binding address is the *previous*
                # token, which is where the other cells bind too.  Using the
                # current token here puts the binding on the value token's
                # address while queries arrive on the key token's address, and
                # the memory then reads nothing that was stored.  The gate has
                # to be computed from the same address the write uses.
                b, d, w = self._gates(ids[:, t], a)
                Y = d[:, :, None] * S_t
                coef = torch.einsum('bi,bid->bd', b * a, Y)
                S_t = Y - torch.einsum('bi,bd->bid', a, coef)
                S_t = S_t + torch.einsum('bi,bd->bid', a, w * v)
                _ = beta
            active = active | (ids[:, t] == probe) | (op == 2)
            if want_readout:
                out = torch.einsum('bi,bid->bd', k, S_t) @ self.val.weight.t()
                mem.append(torch.where(active[:, None], out, torch.zeros_like(out))
                           if self.trigger else out)
            kp = k
        readout = torch.stack(mem, 1) if want_readout else None
        return readout, S_t

    def forward(self, ids, x=None, wmask=None, ops=None):
        mem, _ = self._rollout(ids, x, wmask, ops, want_readout=True)
        return mem

    @torch.no_grad()
    def state_after(self, ids, ops=None, wmask=None, x=None):
        if x is None:
            x = self.key(ids)          # rollouts outside training use addressing only
        _, state = self._rollout(ids, x, wmask, ops, want_readout=False)
        return state


CELL_KINDS = {
    "stable": StableMemCell,
    "delta": DeltaMemCell,
    "tensor3d": Tensor3DMemCell,
    "edit_add": lambda V, dk, dv, **kw: EditMemCell(V, dk, dv, write_rule="add", **kw),
    "edit_delta": lambda V, dk, dv, **kw: EditMemCell(V, dk, dv, write_rule="delta", **kw),
    "edit_add_bi": lambda V, dk, dv, **kw: EditMemCell(V, dk, dv, write_rule="add",
                                                        bilinear=True, **kw),
    "edit_gated": GatedEditMemCell,
}


def make_cell(kind, *args, **kwargs):
    """Instantiate a memory cell by name; see :data:`CELL_KINDS`."""
    try:
        cell_cls = CELL_KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown cell kind {kind!r}; "
                         f"expected one of {sorted(CELL_KINDS)}") from None
    return cell_cls(*args, **kwargs)
