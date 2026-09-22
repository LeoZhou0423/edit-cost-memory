"""Edits in a fixed-state associative memory: the erase trade-off.

Pure linear algebra, no training.  The state is ``S = sum_i b_i (x) v_i`` and a
read at address ``a_p`` returns ``a_p^T S``.  Erasing the binding addressed by
``k`` writes ``-k (x) (e^T S)`` into the state for some erase direction ``e``,
so two numbers describe the operation:

    residual   r_j = || a_j^T S' || / || a_j^T S ||      (did the binding go?)
    locality   l_j = max_{p != j} ||a_p^T S' - a_p^T S|| / ||a_p^T S||
                                                          (what else moved?)

For ``e = c k`` the two are not independent:

    r_j = |1 - c|          l_j = c * mu_j,     mu_j = max_{p != j} |a_p . a_j|

so as the erase is made complete (r -> 0) the collateral damage is forced to
``mu_j``, and the only way to have both small is to make the *code* incoherent,
i.e. to make ``mu`` small.  Every gate in this literature -- Gated DeltaNet-2's
channel-wise erase gate, KDA's channel-wise decay -- enters through ``c`` or
through a channel-wise modulation of ``e``, and therefore moves the arm along
this line without being able to change its slope.  That is the claim this script
checks, and the first table below is the evidence for it.

The other table is the write rule.  A write code ``b_i`` that is biorthogonal to
the addressing code, ``a_p . b_i = delta_pi``, removes the trade-off entirely:
reads are exact and ``l_j = 0``, because the first factor above vanishes.  The
delta rule is the online approximation to that code, and the closed-form dual
``B = (A A^T)^{-1} A`` is the thing it approximates, which exists only while
``m <= dk``.

Run: python experiments/edit_locality.py [--out results.json]
"""
import argparse
import json
import math

import torch

POOL = 256          # reserved addresses (the MQAR key pool)
# The per-step decay is deliberately off in the geometry tables.  It multiplies
# each binding's contribution by lam^(writes after it), which is a positive scalar
# per binding: it rescales the readout, and because a recall decision compares
# token logits it does not couple bindings or introduce cross-talk.  Leaving it on
# would put every write rule on a different scale and hide the thing being
# measured.  Table 4 reports the decay separately so its effect is not silently
# dropped.
LAM = 1.0


def address_code(m, dk, seed=0, unit=True):
    """``m`` addresses drawn from ``POOL`` reserved keys embedded in ``R^dk``.

    Two regimes, and the difference between them is the point of the sweep:

    - ``dk >= POOL``: an orthonormal row basis exists, so the code can be made
      exactly incoherent (``mu = 0``) by QR of the transpose.  Rows are
      orthonormal, not merely orthogonal columns.
    - ``dk < POOL``: at most ``dk`` orthonormal vectors fit in ``R^dk``, so the
      ``POOL`` addresses cannot all be mutually orthogonal.  This is what
      ``orth_init`` produces in the model -- ``torch.linalg.qr`` of a
      ``(POOL, dk)`` matrix, whose *columns* are orthonormal and whose rows are
      not -- and the resulting coherence is bounded below by the Welch bound.
    """
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(POOL, dk, generator=gen)
    if dk >= POOL:
        Q, _ = torch.linalg.qr(X.t())          # (dk, POOL), orthonormal columns
        A = Q.t()[:m]                          # (m, dk), orthonormal rows
    else:
        Q, _ = torch.linalg.qr(X)              # (POOL, dk), orthonormal columns
        A = Q[:m]
    if unit:
        A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return A


def welch(dk, K=POOL):
    """Welch lower bound on the coherence of K unit vectors in R^dk."""
    if dk >= K:
        return 0.0
    return math.sqrt((K - dk) / (dk * (K - 1)))


def write_code(A, rule, m, dk, lam=LAM, beta=0.995):
    if rule == "additive":
        return A.clone()
    if rule == "dual":
        Ad = A.double()
        return torch.linalg.solve(Ad @ Ad.t(), Ad).float()
    if rule == "delta":
        # run the sequential delta rule with V = I so the (dk, m) state equals B^T
        S = torch.zeros(dk, m)
        for i in range(m):
            k = A[i]
            target = torch.zeros(m)
            target[i] = 1.0
            pred = k @ S
            S = lam * S + beta * torch.outer(k, target - lam * pred)
        return S.t()
    raise ValueError(rule)


def evaluate(m, dk, rule, seed=0, gate_scale=1.0, erase_dir="address",
             gate_vec=None):
    """Erase one binding and measure what it cost.

    ``erase_dir`` is the *direction* the removal is written along, and it is the
    variable this whole script is about:

    - ``address``: remove ``a_j (x) (a_j^T S)``.  This is what every recurrent
      delta-rule model does in practice, because ``a_j`` is the only direction the
      cell holds at the time of the edit.
    - ``write``: remove ``b_j (x) (a_j^T S)``, i.e. along the *write* code, with
      the value recovered by reading.
    - ``gated``: remove ``e_j (x) (a_j^T S)`` with ``e_j = gate * a_j`` for a
      channel-wise gate, the erase direction of Gated DeltaNet-2.

    Two numbers describe the result.  With ``g_j = a_j . e_j`` the alignment of the
    erase direction with the address it is aimed at, and ``L1_j`` the locality at
    unit erase scale, they are

        residual_j = |1 - c g_j|            locality_j = c L1_j

    so, eliminating the scale ``c``,

        locality_j = (L1_j / g_j) (1 - residual_j)                (line)

    for ``c g_j <= 1``.  This is the general form; Eq. (1) of the manuscript is the
    special case ``g_j = 1``, which holds for a unit-norm address erase and fails
    for the delta and gated directions.  The slope of the line is ``L1_j / g_j``,
    so an erase direction that is *misaligned* with the address it targets pays
    proportionally more collateral damage per unit of erasure.  Perfect editing
    needs both ``g_j = 1`` and ``L1_j = 0``, i.e. ``a_j . e_j = 1`` and
    ``a_p . e_j = 0`` for every ``p != j``, which is the dual vector of the code.
    """
    A = address_code(m, dk, seed=seed)
    B = write_code(A, rule, m, dk)
    V = torch.randn(m, dk, generator=torch.Generator().manual_seed(seed + 1))
    S = B.t() @ V

    G = A @ A.t()
    off = G - torch.eye(m)
    mu = float(off.abs().max())

    reads = A @ S
    read_err = float((reads - V).norm(dim=-1).mean() / V.norm(dim=-1).mean())

    if gate_vec is not None:
        def direction(j):
            return gate_vec[j] * A[j]
    elif erase_dir == "write":
        def direction(j):
            return B[j]
    else:
        def direction(j):
            return A[j]

    resid, loc, align = [], [], []
    for j in range(m):
        e = gate_scale * direction(j)
        g = float(A[j] @ e)
        Sp = S - torch.outer(e, A[j] @ S)
        r = float((A[j] @ Sp).norm() / (A[j] @ S).norm().clamp_min(1e-9))
        keep = torch.arange(m) != j
        base = reads[keep]
        now = A[keep] @ Sp
        l = float(((now - base).norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-9)).max())
        resid.append(r)
        loc.append(l)
        align.append(g)
    L1 = max(loc)
    g = min(align)
    return {"m": m, "dk": dk, "rule": rule, "seed": seed, "erase_dir": erase_dir,
            "gate_scale": gate_scale,
            "mu": mu, "welch": welch(dk),
            "read_err": read_err,
            "residual": max(resid), "locality": L1,
            "alignment_g": g,
            "line": (L1 / g) * (1.0 - max(resid)) if g > 1e-9 else float("nan"),
            "mean_mu_j": float(off.abs().max(dim=1).values.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    print("=" * 100)
    print("Table 1.  Erase direction and write rule vs address dimension.  m = 64 "
          "bindings, K = 256 addresses.")
    print("locality = collateral damage at an unedited address; it is what a gate "
          "is supposed to reduce.")
    print("'addr' erases along the address (what delta-rule cells do); "
          "'write' erases along the write code.")
    print("Both rows of every pair are reported: biorthogonality of the write code "
          "makes reads exact")
    print("regardless of the erase direction, but zero locality needs the erase to run "
          "along the dual too.")
    print("=" * 100)
    print(f"{'dk':>4}{'rule':>10}{'erase':>7}{'mu':>9}{'Welch':>9}{'read_err':>10}"
          f"{'g':>9}{'residual':>10}{'locality':>10}")
    for dk in (256, 128, 64, 32):
        for rule in ("additive", "delta", "dual"):
            for ed in ("address", "write"):
                r = evaluate(64, dk, rule, erase_dir=ed)
                rows.append(r)
                print(f"{dk:>4}{rule:>10}{ed:>7}{r['mu']:>9.4f}{r['welch']:>9.4f}"
                      f"{r['read_err']:>10.4f}{r['alignment_g']:>9.4f}"
                      f"{r['residual']:>10.4f}{r['locality']:>10.4f}")
        print()

    print("=" * 100)
    print("Table 1b.  Seed dispersion.  Every table above is one draw of the address "
          "code.  This repeats")
    print("Table 1 over five draws and reports the range, so 'of the order of mu' can "
          "be judged against")
    print("how much mu itself moves.  m = 64, erase along the address, additive write.")
    print("=" * 100)
    print(f"{'dk':>4}{'mu (min-max)':>22}{'L1 (min-max)':>22}{'L1/mu':>18}")
    for dk in (128, 64, 32):
        mus, l1s, ratios = [], [], []
        for s in range(5):
            r = evaluate(64, dk, "additive", seed=s, erase_dir="address")
            mus.append(r["mu"])
            l1s.append(r["locality"])
            ratios.append(r["locality"] / r["mu"])
            rows.append({**r, "table": "1b"})
        print(f"{dk:>4}{min(mus):>10.4f}-{max(mus):<11.4f}"
              f"{min(l1s):>10.4f}-{max(l1s):<11.4f}"
              f"{min(ratios):>8.3f}-{max(ratios):<9.3f}")
    print()

    print("=" * 100)
    print("Table 1c.  The delta rule's step size.  At an orthonormal code the "
          "additive write is exact but the")
    print("delta rule is not, because a step size below one does not reach the exact "
          "fixed point in m steps.")
    print("=" * 100)
    print(f"{'dk':>4}{'beta':>7}{'read_err':>11}{'locality':>10}")
    for dk in (256, 128):
        for beta in (1.0, 0.995, 0.98):
            A = address_code(64, dk)
            B = None
            rows_beta = None
            Ad = A
            S = torch.zeros(dk, 64)
            for i in range(64):
                k = Ad[i]
                tgt = torch.zeros(64)
                tgt[i] = 1.0
                pred = k @ S
                S = LAM * S + beta * torch.outer(k, tgt - LAM * pred)
            V = torch.randn(64, dk, generator=torch.Generator().manual_seed(1))
            Bt = S.t()
            St = Bt.t() @ V
            re = float((A @ St - V).norm(dim=-1).mean() / V.norm(dim=-1).mean())
            # locality of an address-direction erase on the delta state
            loc = 0.0
            reads = A @ St
            for j in range(64):
                Sp = St - torch.outer(A[j], A[j] @ St)
                keep = torch.arange(64) != j
                base = reads[keep]
                now = A[keep] @ Sp
                loc = max(loc, float(((now - base).norm(dim=-1)
                                      / base.norm(dim=-1).clamp_min(1e-9)).max()))
            print(f"{dk:>4}{beta:>7.3f}{re:>11.4f}{loc:>10.4f}")
            _ = (B, rows_beta)
        print()

    print("=" * 100)
    print("Table 2.  The erase trade-off.  Two knobs are swept: the erase scale c, and "
          "the erase direction.")
    print("With g_j = a_j . e_j, the alignment of the erase direction with the address "
          "it targets, and")
    print("L1_j the locality at alignment 1, the prediction is residual = |1 - g_j| and "
          "locality =")
    print("at alignment 1, the prediction is residual = |1 - g_j| and locality = "
          "g_j L1_j / g0_j,")
    print("where g0 is the alignment of the unscaled direction.  Eliminating the scale "
          "gives a V, not a")
    print("line: locality = (L1_j/g0_j)(1 - residual) while g_j <= 1, and "
          "(L1_j/g0_j)(1 + residual)")
    print("past the fold.  m = 64, additive write, erase scaled along the address.")
    print("=" * 100)
    print(f"{'dk':>4}{'c':>7}{'g':>9}{'resid':>9}{'|1-g|':>9}{'loc':>9}{'g*L1':>9}"
          f"{'mu':>9}")
    for dk in (128, 64, 32):
        unit = evaluate(64, dk, "additive", gate_scale=1.0)
        L1 = unit["locality"]
        g0 = unit["alignment_g"]
        for c in (1.0, 0.5, 0.25, 1.25, 1.5):
            r = evaluate(64, dk, "additive", gate_scale=c)
            g = r["alignment_g"]
            pred_res = abs(1.0 - g)
            pred_loc = g * L1 / g0
            rows.append({**r, "L1": L1, "g0": g0, "pred_residual": pred_res,
                         "pred_locality": pred_loc})
            print(f"{dk:>4}{c:>7.2f}{g:>9.4f}{r['residual']:>9.4f}{pred_res:>9.4f}"
                  f"{r['locality']:>9.4f}{pred_loc:>9.4f}{r['mu']:>9.4f}")
        print()

    print("=" * 100)
    print("Table 2b.  A channel-wise erase gate, the Gated DeltaNet-2 form.  The erase "
          "direction is")
    print("e_j = b * a_j for a per-channel gate b, so the gate changes the alignment g "
          "as well as the")
    print("amplitude.  Columns report both.  A gate that lowers g buys locality by "
          "leaving residue; a")
    print("gate that keeps g = 1 stays on the same line.  Neither reaches L1 = 0.  "
          "m = 64, dk = 128.")
    print("=" * 100)
    print(f"{'gate':>22}{'g':>9}{'residual':>10}{'locality':>10}{'|1-g|':>9}")
    dk = 128
    for label, gate in (
            ("all ones", torch.ones(64, dk)),
            ("uniform 0.5", 0.5 * torch.ones(64, dk)),
            ("half channels off", (torch.arange(dk) < dk // 2).float().repeat(64, 1)),
            ("random in [0.5, 1]",
             (0.5 + 0.5 * torch.rand(dk, generator=torch.Generator().manual_seed(3)))
             .repeat(64, 1)),
            ("per-row random",
             torch.rand(64, dk, generator=torch.Generator().manual_seed(4))),
    ):
        r = evaluate(64, dk, "additive", erase_dir="gated", gate_vec=gate)
        rows.append({**r, "gate_label": label})
        print(f"{label:>22}{r['alignment_g']:>9.4f}{r['residual']:>10.4f}"
              f"{r['locality']:>10.4f}{abs(1.0 - r['alignment_g']):>9.4f}")
    ref = evaluate(64, dk, "additive")
    print(f"{'(address, no gate)':>22}{ref['alignment_g']:>9.4f}{ref['residual']:>10.4f}"
          f"{ref['locality']:>10.4f}{'-':>9}")
    print()

    print("=" * 100)
    print("Table 3.  The exact dual write above its rank.  (A A^T)^-1 exists only while "
          "m <= dk; beyond it the")
    print("read stops being exact and, unlike the additive write, the failure is not "
          "graceful.  Erase is along the")
    print("write code, which is where the dual is unbounded when A A^T is singular.")
    print("=" * 100)
    print(f"{'dk':>4}{'m':>5}{'rule':>10}{'read_err':>11}{'residual':>10}{'locality':>10}")
    for dk in (64, 32, 16):
        for m in (dk // 2, dk, 2 * dk, 4 * dk):
            for rule in ("additive", "dual"):
                try:
                    r = evaluate(m, dk, rule, erase_dir="write")
                except Exception as exc:                       # noqa: BLE001
                    print(f"{dk:>4}{m:>5}{rule:>10}   failed: {type(exc).__name__}")
                    continue
                rows.append(r)
                print(f"{dk:>4}{m:>5}{rule:>10}{r['read_err']:>11.4f}"
                      f"{r['residual']:>10.4f}{r['locality']:>10.4f}")
        print()

    print("=" * 100)
    print("Table 4.  What the threshold costs at realistic alphabet sizes.  The Welch "
          "bound is the floor on")
    print("coherence for K addresses in R^dk, and coherence is the floor on the price "
          "of an edit.  A real")
    print("tokenizer has K in the tens of thousands, so the orthonormal route "
          "(dk >= K) is out of reach and")
    print("the dual route (m <= dk) is the only one -- with the rank guard that Table 3 "
          "shows is mandatory.")
    print("=" * 100)
    print(f"{'K':>8}" + "".join(f"{'dk=' + str(d):>12}" for d in (128, 256, 512, 1024)))
    for K in (256, 4096, 50257):
        cells = []
        for dk in (128, 256, 512, 1024):
            cells.append("0 (exact)" if dk >= K else f"{welch(dk, K):.4f}")
        print(f"{K:>8}" + "".join(f"{c:>12}" for c in cells))
        rows.append({"table": 4, "K": K,
                     "welch": {str(d): welch(d, K) for d in (128, 256, 512, 1024)}})
    print()

    print("=" * 100)
    print("Table 5.  The decay on its own, at the orthonormal code (dk = 256), where "
          "cross-talk is zero by")
    print("construction.  A per-step decay lam rescales every read by lam^(writes "
          "after it); the column 'scale'")
    print("shows the resulting spread across bindings, which is a gain change and not "
          "an error term.  The delta")
    print("rule's own step size beta is the separate effect in Table 1b.")
    print("=" * 100)
    print(f"{'lam':>8}{'rule':>10}{'beta':>7}{'read_err':>11}{'spread':>10}{'locality':>10}")
    for lam in (1.0, 0.999):
        for rule, beta in (("additive", 1.0), ("delta", 1.0), ("delta", 0.98)):
            A = address_code(64, 256)
            m = 64
            if rule == "additive":
                S = torch.zeros(256, m)
                for i in range(m):
                    S = lam * S + torch.outer(A[i], torch.eye(m)[i])
                Bt = S
            else:
                S = torch.zeros(256, m)
                for i in range(m):
                    k = A[i]
                    tgt = torch.zeros(m)
                    tgt[i] = 1.0
                    pred = k @ S
                    S = lam * S + beta * torch.outer(k, tgt - lam * pred)
                Bt = S
            V = torch.randn(m, 256, generator=torch.Generator().manual_seed(1))
            St = Bt @ V
            reads = A @ St
            re = float((reads - V).norm(dim=-1).mean() / V.norm(dim=-1).mean())
            spread = float((reads.norm(dim=-1) / V.norm(dim=-1)).std()
                           / (reads.norm(dim=-1) / V.norm(dim=-1)).mean())
            loc = 0.0
            for j in range(m):
                Sp = St - torch.outer(A[j], A[j] @ St)
                keep = torch.arange(m) != j
                base = reads[keep]
                now = A[keep] @ Sp
                loc = max(loc, float(((now - base).norm(dim=-1)
                                      / base.norm(dim=-1).clamp_min(1e-9)).max()))
            print(f"{lam:>8.4f}{rule:>10}{beta:>7.3f}{re:>11.4f}{spread:>10.4f}"
                  f"{loc:>10.4f}")
            rows.append({"dk": 256, "m": m, "rule": rule, "lam": lam, "beta": beta,
                         "read_err": re, "scale_spread": spread, "locality": loc})
        print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
