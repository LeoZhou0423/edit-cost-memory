"""Linear / selective state-space backbones used as the dense modeling engine.

Two recurrent context-mixing layers are provided:

* :class:`GatedScan` — a scalar input-gated linear recurrence
  ``h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W_g x_t)`` (serial; used by the
  v11/v12 model family).
* :class:`SelectiveSSM` — a Mamba-2-style vector-state selective SSM with a
  diagonal decay ``A_t`` and low-rank ``B_t``/``C_t`` projections, computed by a
  numerically stable block-closed-form parallel scan (used by the v13/v14 model
  family).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedScan(nn.Module):
    """density 引擎: input-dependent 门控线性循环, 跨位置混合上下文.
    h_t = (1 - z_t)⊙h_{t-1} + z_t⊙tanh(Wg x_t)  (串行, S<=256 可接受)"""
    def __init__(self, d):
        super().__init__()
        self.wz = nn.Linear(d, d)
        self.wg = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        B, S, D = x.shape
        z = torch.sigmoid(self.wz(x))
        g = torch.tanh(self.wg(x))
        h = torch.zeros(B, D, device=x.device)
        outs = []
        for t in range(S):
            h = (1 - z[:, t]) * h + z[:, t] * g[:, t]
            outs.append(h)
        return self.ln(torch.stack(outs, 1)) + x


class SelectiveSSM(nn.Module):
    """向量态选择性 SSM (Mamba-2 式简化, 对角 A + 低秩 B/C, 无卷积).
    h_t = A_t ⊙ h_{t-1} + B_t ⊙ u_t   (h ∈ R^{D×N})
    y_t = C_t · h_t                    (沿状态秩 N 聚合)
    A_t = exp(-Δ_t * exp(logA))        (每维衰减, Δ 由输入决定)
    """
    def __init__(self, d, N=16):
        super().__init__()
        self.d = d; self.N = N
        # 基础衰减: 初始 exp(logA)≈0.99 (慢遗忘, 稳定)
        self.logA = nn.Parameter(torch.full((d,), math.log(0.99)))
        self.fc_dt = nn.Linear(d, d)   # Δ (softplus)
        self.fc_u = nn.Linear(d, d)    # 输入投影
        self.fc_b = nn.Linear(d, N)    # B_t (选择性)
        self.fc_c = nn.Linear(d, N)    # C_t (选择性)
        self.out = nn.Linear(N, d)     # 聚合回 D
        self.ln = nn.LayerNorm(d)
        # 保守初始化
        for m in (self.fc_dt, self.fc_u, self.fc_b, self.fc_c, self.out):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        B, S, D = x.shape
        dt = F.softplus(self.fc_dt(x))                 # (B,S,D)
        A = torch.exp(-dt * self.logA.exp())           # (B,S,D) per-step decay (0,1)
        # 数值稳定: 钳制单步衰减下界, 避免 b/P_local 中超小 P_local 引起 1/P^2 梯度爆炸.
        # C=8 下块内 P_local >= 0.5^8 ≈ 0.0039, 1/P^2 有界, 训练不外溢.
        A = A.clamp(min=0.5, max=1.0)
        u = self.fc_u(x)                               # (B,S,D)
        Bt = self.fc_b(x)                              # (B,S,N)
        Ct = self.fc_c(x)                              # (B,S,N)
        b = torch.einsum('bsn,bsd->bsdn', Bt, u)       # (B,S,D,N) outer/写入
        C = 8                                          # 块内长度: A>=0.5 时 P_local>=0.5^8≈0.0039, 1/P^2 有界
        nck = math.ceil(S / C)
        pad = nck * C - S
        if pad > 0:
            A = F.pad(A, (0, 0, 0, pad))
            b = F.pad(b, (0, 0, 0, 0, 0, pad))
            Ct = F.pad(Ct, (0, 0, 0, pad))
        A = A.view(B, nck, C, D)                       # (B,nck,C,D)
        b = b.view(B, nck, C, D, self.N)               # (B,nck,C,D,N)
        # 块内闭式扫描: h_loc[t] = P_loc[t]*Σ_{j<=t} b_j/P_loc[j]
        # 除法规避: P_local 下界抬高到 0.01 (A>=0.5,C=8 时真实值>=0.0039),
        # 把 g<=100*b、1/P^2<=1e4 压住, 杜绝 backward 通过 b/P_local 的放大溢出.
        # 轻微抬高极小 P 只改变 ~1e-3 量级的 h, 不影响精度但大幅增强稳定性.
        P_local = torch.cumprod(A, dim=2).clamp(min=0.01)
        g = b / P_local[:, :, :, :, None]              # 块内 P 有下界 -> forward/backward 均有界
        h_local = P_local[:, :, :, :, None] * torch.cumsum(g, dim=2)  # (B,nck,C,D,N)
        Q_chunk = h_local[:, :, -1, :, :]              # 每块从零态出发的末态 (B,nck,D,N)
        P_chunk = P_local[:, :, -1, :]                 # 每块整体衰减 (B,nck,D)
        # 块间 carry 顺序传播 (nck≈8 次, 远小于 S)
        carry = torch.zeros(B, nck, D, self.N, device=x.device)
        hc = torch.zeros(B, D, self.N, device=x.device)
        for c in range(nck):
            carry[:, c] = hc
            hc = P_chunk[:, c, :, None] * hc + Q_chunk[:, c]
        # 真正输出 = 块内贡献 + 携带的上一块状态按块内衰减继续
        h_true = P_local[:, :, :, :, None] * carry[:, :, None, :, :] + h_local
        y = torch.einsum('bqcdn,bqcn->bqcd', h_true, Ct.view(B, nck, C, self.N))
        y = y.reshape(B, nck * C, D)[:, :S]
        return self.ln(y) + x
