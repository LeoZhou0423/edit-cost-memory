"""Data utilities: multi-query associative recall (MQAR) generation and dense
language-model token loading.

These routines are shared by the training scripts, the ablation studies and the
evaluation harness. The MQAR generator uses the GPT-2 vocabulary (VOCAB=50257),
an alphabet of 256 reserved key/value tokens, and a single query (``PROBE``)
issued at the end of the sequence ("load-one" setting).
"""
import os
import numpy as np
import torch

# GPT-2 vocabulary size.
VOCAB = 50257


def _data_dir() -> str:
    """Root directory holding the pre-tokenized ``.npy`` corpora.

    Override with the ``DCGR3D_DATA`` environment variable; defaults to the
    ``data/`` directory at the repository root.
    """
    return os.environ.get("DCGR3D_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))


def mqar_token_pools(disjoint=False):
    """Return the fixed key/value token pools used by MQAR.

    ``disjoint=False`` preserves the historical protocol.  Revised experiments
    use ``disjoint=True`` to exclude the reserved probe token and prevent a
    token from serving as both a key and a value.
    """
    pool_rs = np.random.RandomState(0)
    if disjoint:
        selected = pool_rs.choice(VOCAB - 1, 512, replace=False)
        return selected[:256], selected[256:]
    keys = pool_rs.choice(VOCAB, 256, replace=False)
    values = pool_rs.choice(VOCAB, 256, replace=False)
    return keys, values


def make_mqar(seed=None, disjoint=False):
    """Return a reproducible MQAR generator.

    When ``seed`` is supplied it controls generated examples, allowing revised
    experiments to use independent train/validation/test streams.  With the
    default ``None``, sampling continues to use NumPy's global RNG for backward
    compatibility with the original experiment scripts.  Token pools retain
    their historical fixed seed in both cases.

    Layout (``queries=1`` reproduces the historical "load-one" sequence)::

        0 2 4 ... 2m-2   keys
        1 3 5 ... 2m-1   values
        2m + 2q          PROBE
        2m + 2q + 1      query key of block q

    The supervised retrieval target of every query block sits at the position of
    its query key, because the readout is content-addressed: the key token is the
    input and the value token is the output at the same index.  ``queries > 1``
    appends further query blocks so a single sequence can supervise many
    bindings, which is the standard dense MQAR objective.
    """
    sample_rs = np.random if seed is None else np.random.RandomState(seed)
    KEYS, VALS = mqar_token_pools(disjoint=disjoint)
    PROBE = VOCAB - 1
    def gen(bs, m, d, dev, queries=1):
        p = m - 1 - d
        ki = np.array([sample_rs.permutation(256)[:m] for _ in range(bs)])
        vi = sample_rs.randint(0, 256, (bs, m))
        keys = KEYS[ki]; vals = VALS[vi]
        L = 2 * m + 2 * queries
        seq = np.zeros((bs, L), dtype=np.int64); tgt = np.full((bs, L), -100, dtype=np.int64)
        seq[:, 0:2*m:2] = keys; seq[:, 1:2*m:2] = vals
        rows = np.arange(bs)
        for q in range(queries):
            # Block 0 keeps the deterministic distance-controlled target p so the
            # historical near/middle/far evaluation stays comparable.  Extra
            # blocks sample a fresh stored key, otherwise they would repeat the
            # same binding and teach nothing new.
            qi = np.full(bs, p) if q == 0 else sample_rs.randint(0, m, bs)
            seq[:, 2*m+2*q] = PROBE
            seq[:, 2*m+2*q+1] = keys[rows, qi]
            tgt[:, 2*m+2*q+1] = vals[rows, qi]
        tgt[:, 1:2*m+1:2] = vals[:, :m]     # 所有 value 写入位(第j个值=vals[j])
        wt = (tgt >= 0).astype(np.float32)
        return (torch.from_numpy(seq).to(dev), torch.from_numpy(tgt).to(dev),
                torch.from_numpy(wt).to(dev))
    return gen


def make_ear(seed=None, disjoint=True):
    """Return a generator for *edited associative recall* (EAR).

    The sequence is the canonical MQAR layout with an edit region inserted
    between the storage prefix and the query region::

        0 .. 2m-1        storage: k1 v1 k2 v2 ... km vm
        2m .. 2m+2e-1    edit region: e blocks of two tokens
        rest             query region: NOISE with m query keys at random slots

    Each edit block targets a stored key.  A *delete* block is ``(k_j, NOISE)``;
    an *overwrite* block is ``(k_j, v')`` and rebinds ``k_j`` to ``v'``.  The
    block that carries the edit is signalled through the ``ops`` tensor rather
    than through a reserved token: ``ops = 2`` at the address being erased and
    ``ops = 1`` at a value being written.  Passing the operation explicitly is an
    interface assumption, the same status as the trigger gate -- the task
    measures whether a fixed state *supports* an edit, not whether a model can
    schedule one.

    Returns ``(seq, target, ops, meta)`` where ``meta`` carries, per row, the
    index of each stored key inside the sequence, its current value, and whether
    it was edited, deleted or untouched.  The targets at query slots are the
    current value of the queried key; a query on a deleted key is not supervised
    and is scored separately as "did the stale value stay suppressed".
    """
    sample_rs = np.random if seed is None else np.random.RandomState(seed)
    KEYS, VALS = mqar_token_pools(disjoint=disjoint)
    NOISE = VOCAB - 1

    def gen(bs, m, seq_len, dev, n_edit, n_query=None):
        n_query = m if n_query is None else n_query
        edit_len = 2 * n_edit
        region_start = 2 * m + edit_len
        region = seq_len - region_start
        if region < n_query:
            raise ValueError("sequence too short for the requested queries")

        seq = np.full((bs, seq_len), NOISE, dtype=np.int64)
        ops = np.zeros((bs, seq_len), dtype=np.int64)
        tgt = np.full((bs, seq_len), -100, dtype=np.int64)
        meta = {"stored_keys": np.zeros((bs, m), dtype=np.int64),
                "current": np.full((bs, m), -1, dtype=np.int64),
                "state": np.zeros((bs, m), dtype=np.int64),
                "query_slot": np.full((bs, n_query), -1, dtype=np.int64),
                "query_pair": np.full((bs, n_query), -1, dtype=np.int64),
                "query_kind": np.zeros((bs, n_query), dtype=np.int64)}

        for b in range(bs):
            ki = sample_rs.permutation(256)[:m]
            vi = sample_rs.randint(0, 256, m)
            keys, vals = KEYS[ki], VALS[vi]
            seq[b, 0:2 * m:2] = keys
            seq[b, 1:2 * m:2] = vals
            ops[b, 1:2 * m:2] = 1
            current = vals.copy()
            state = np.zeros(m, dtype=np.int64)          # 0 clean, 1 updated, 2 deleted

            # the edit region: half deletes, half overwrites, distinct targets
            targets = sample_rs.choice(m, size=min(n_edit, m), replace=False)
            for slot, j in enumerate(targets):
                base = 2 * m + 2 * slot
                seq[b, base] = keys[j]
                ops[b, base] = 2
                if slot % 2 == 0:
                    state[j] = 2                          # delete: second slot stays NOISE
                else:
                    new_val = VALS[sample_rs.randint(0, 256)]
                    seq[b, base + 1] = new_val
                    ops[b, base + 1] = 1
                    current[j] = new_val
                    state[j] = 1

            # the query region: queries drawn to cover all three states.  The live
            # pool can be smaller than n_query once most keys are deleted, so it is
            # tiled rather than sliced.
            live = np.flatnonzero(state != 2)
            deleted = np.flatnonzero(state == 2)
            if len(live):
                reps = int(np.ceil(n_query / len(live)))
                picks = np.tile(sample_rs.permutation(live), reps)[:n_query]
            else:
                picks = np.zeros(n_query, dtype=np.int64)
            if len(deleted):
                # a quarter of the queries land on deleted keys, so the
                # suppression metric is estimated on a usable sample
                k = min(max(1, n_query // 4), n_query, len(deleted))
                picks[:k] = sample_rs.permutation(deleted)[:k]
            slots = region_start + sample_rs.permutation(region)[:n_query]
            seq[b, slots] = keys[picks]
            for q, (slot, j) in enumerate(zip(slots, picks)):
                meta["query_slot"][b, q] = slot
                meta["query_pair"][b, q] = j
                meta["query_kind"][b, q] = state[j]
                if state[j] != 2:
                    tgt[b, slot] = current[j]

            meta["stored_keys"][b] = keys
            meta["current"][b] = current
            meta["state"][b] = state

        wmask = (ops == 1).astype(np.float32)
        return (torch.from_numpy(seq).to(dev), torch.from_numpy(tgt).to(dev),
                torch.from_numpy(wmask).to(dev), torch.from_numpy(ops).to(dev),
                {k: torch.from_numpy(v).to(dev) for k, v in meta.items()})

    return gen


def mqar_write_mask(S, m, device, batch=1):
    """Boolean write gate over the sequence layout.

    Value-storage positions are the odd indices ``1, 3, ..., 2m-1``.  Unlike
    :func:`make_wmask` this is derived from the layout rather than from the
    targets, so it remains correct once extra query blocks also carry targets and
    must *not* be written.
    """
    mask = torch.zeros(batch, S, dtype=torch.bool, device=device)
    mask[:, 1:2 * m:2] = True
    return mask


def make_mqar_canonical(seed=None):
    """Return a generator for canonical multi-query associative recall.

    This is the Zoology layout (Arora et al., 2023): ``n_pairs`` key-value pairs
    occupy a contiguous prefix, and a query region of length
    ``seq_len - 2 * n_pairs`` follows, filled with the reserved marker token as
    noise.  Each stored key reappears exactly once at a uniformly random slot in
    the query region and the target at that slot is its bound value.

    Two properties matter for a fair comparison and are the reason this layout
    exists alongside :func:`make_mqar`:

    - queries are scattered through noise instead of sitting in adjacent blocks,
      so no "copy whatever followed the previous query" shortcut exists;
    - the number of supervised answers per sequence equals ``n_queries``, which
      is the dense objective the published MQAR numbers are trained under.

    Only query slots are supervised.  The write gate covers the storage prefix
    only (``mqar_write_mask(seq_len, n_pairs, ...)``), so the noise region never
    writes into a content-addressed readout.
    """
    sample_rs = np.random if seed is None else np.random.RandomState(seed)
    KEYS, VALS = mqar_token_pools(disjoint=True)
    NOISE = VOCAB - 1
    def gen(bs, n_pairs, seq_len, dev, n_queries=None):
        n_queries = n_pairs if n_queries is None else n_queries
        if seq_len < 2 * n_pairs + n_queries:
            raise ValueError("sequence too short for the requested pairs/queries")
        seq = np.full((bs, seq_len), NOISE, dtype=np.int64)
        tgt = np.full((bs, seq_len), -100, dtype=np.int64)
        for b in range(bs):
            ki = sample_rs.permutation(256)[:n_pairs]
            vi = sample_rs.permutation(256)[:n_pairs]
            keys, vals = KEYS[ki], VALS[vi]
            seq[b, 0:2*n_pairs:2] = keys
            seq[b, 1:2*n_pairs:2] = vals
            region = seq_len - 2 * n_pairs
            slots = 2 * n_pairs + sample_rs.permutation(region)[:n_queries]
            order = sample_rs.permutation(n_pairs)[:n_queries]
            seq[b, slots] = keys[order]
            tgt[b, slots] = vals[order]
        wt = np.zeros((bs, seq_len), dtype=np.float32)
        wt[:, 1:2*n_pairs:2] = 1.0
        return (torch.from_numpy(seq).to(dev), torch.from_numpy(tgt).to(dev),
                torch.from_numpy(wt).to(dev))
    return gen


def make_wmask(tgt, S):
    """Write gate: only value-storage positions (targets >= 0) are written,
    excluding the final query position (the read is content-addressed)."""
    col = torch.arange(S, device=tgt.device)
    return (tgt >= 0) & (col != S - 1)


def dense_data(n_tok, L, dev, seed=123, path=None, offset=0):
    rs = np.random.RandomState(seed)
    if path is None:
        path = os.path.join(_data_dir(), "wiki.npy")
    # 用真实 token 流
    a = np.load(path).astype(np.int64)
    a = a[offset:offset + n_tok]
    def batch(bs):
        s = rs.randint(0, len(a) - L, bs)
        x = np.stack([a[i:i + L] for i in s])
        return torch.from_numpy(x).to(dev)
    return batch


def load_tokens(name):
    """Load a pre-tokenized corpus split (e.g. ``wikitext103_train``) as a long
    1-D int64 tensor. The file is looked up as ``<data_dir>/<name>.npy``."""
    a = np.load(os.path.join(_data_dir(), f"{name}.npy")).astype(np.int64)
    return torch.from_numpy(a)


def wiki_loader(tok, L, dev, seed=0, offset=0, n=None):
    """Build a batched sequence generator over a long token stream by slicing
    ``L``-length windows from uniformly sampled start positions."""
    if n is None:
        n = len(tok) - offset
    seg = tok[offset:offset + n].to(dev)
    rs = np.random.RandomState(seed)
    nbatch = int(n // L)
    positions = rs.permutation(nbatch)
    pos_i = 0
    def gen(bs):
        nonlocal pos_i
        idx = positions[pos_i:pos_i + bs] * L
        pos_i += bs
        if pos_i + bs > nbatch:
            pos_i = 0
        if len(idx) < bs:
            idx = np.concatenate([idx, np.random.randint(0, nbatch, bs - len(idx)) * L])
        idx = torch.from_numpy(idx).long().to(dev)
        rows = seg.view(-1)[idx[:, None] + torch.arange(L, device=dev)[None, :]]
        return rows
    return gen, nbatch
