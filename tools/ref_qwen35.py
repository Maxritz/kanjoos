#!/usr/bin/env python3
"""Reference vectors for the `qwen35` front end (the C20/qwen35 oracle).

The engine's qwen35 path is a *front end only*: metadata, tensor binding,
tokenizer, embeddings, and a layer's input projections. That much has to be
verified against something that is not the engine, because "it loaded" is not a
result -- a wrong stride, a transposed weight or a mis-decoded block all load
happily.

This tool reads the same GGUF, decodes the same blocks with an independent
implementation (numpy, written from the ggml block layouts), computes the same
vectors in float32, and writes them as raw little-endian f32 plus a manifest.
`kanjoos-run --qwen35-ref DIR` then compares ITS full output against these files
element by element and reports max abs error, relative RMSE, zero/unwritten
outputs and the mismatching indices -- the shape of check AGENTS.md section 5
requires. A digest or a strided sample would be consistent with nonsense.

    python tools/ref_qwen35.py <file.gguf> \
        --layer 0 --tokens 1,2,3 --out records/qwen35-probe-2026-10-08/layer0

Writes into --out:
    manifest.json     geometry, the layer kind, the token ids, the vector list,
                      and the dequant cross-check below
    <name>.f32        one raw float32 LE file per vector, row-major

The oracle refuses (nonzero exit) on any block type it cannot decode, and names
it, rather than skipping the tensor.

Before writing anything, the same block decoders are checked against **gguf-py's
own dequantizers** (the implementation tools/ref_qwen3moe.py already uses): the
first eight blocks of one tensor of every quantized type in the file are decoded
both ways and compared. If those disagree the run stops, because a reference that
is wrong is worse than no reference -- the engine would be checked against a bug.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ggufmeta"))
from gguf_meta import R, GGML  # noqa: E402  (the KV reader lives there)

# (block bytes, weights per block) for the types this oracle decodes.
BLOCKS = {
    0: (4, 1),     # F32
    1: (2, 1),     # F16
    30: (2, 1),    # BF16
    2: (18, 32),   # Q4_0
    8: (34, 32),   # Q8_0
    12: (144, 256),  # Q4_K
    13: (176, 256),  # Q5_K
    14: (210, 256),  # Q6_K
}


def align_up(x, a):
    return (x + a - 1) // a * a


def read_container(path):
    """Header, metadata, tensor table, data offset. Offsets are kept this time."""
    with open(path, "rb") as f:
        r = R(f)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise SystemExit("not a GGUF: %r" % magic)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.s()
            kv[k] = r.val(r.u32())
        tensors = {}
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            tt = r.u32()
            off = r.u64()
            tensors[name] = {"dims": dims, "type": tt, "offset": off}
        alignment = int(kv.get("general.alignment", 32))
        data_offset = align_up(f.tell(), alignment)
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
    return {"version": version, "kv": kv, "tensors": tensors,
            "data_offset": data_offset, "file_size": file_size,
            "alignment": alignment}


def dequant_blocks(tt, raw, ne0):
    """Decode raw bytes covering whole columns -> float32 (ncol, ne0).

    `raw` is the concatenated payload of N output columns, each ne0 weights
    stored as ne0/bw consecutive blocks of bb bytes. Every branch therefore
    works on an (nblocks, bb) array and reshapes to (ncol, ne0) at the end.
    """
    bb, bw = BLOCKS[tt]   # (block bytes, weights per block)
    nblocks = raw.size // bb
    ncol = nblocks // (ne0 // bw)
    a = np.ascontiguousarray(raw).reshape(nblocks, bb)
    if tt == 0:      # F32
        out = a.view("<f4")
    elif tt == 1:    # F16
        out = a.view("<f2").astype(np.float32)
    elif tt == 30:   # BF16
        out = (a.view("<u2").astype(np.uint32) << 16).astype("<u4").view("<f4")
    elif tt == 8:    # Q8_0: f16 d + 32 int8
        d = np.ascontiguousarray(a[:, 0:2]).view("<f2").astype(np.float32)
        q = a[:, 2:].view(np.int8).astype(np.float32)
        out = q * d
    elif tt == 2:    # Q4_0: f16 d + 16 bytes of nibbles, -8 bias
        d = np.ascontiguousarray(a[:, 0:2]).view("<f2").astype(np.float32)
        qs = a[:, 2:]
        lo = (qs & 0x0F).astype(np.float32) - np.float32(8.0)
        hi = (qs >> 4).astype(np.float32) - np.float32(8.0)
        out = np.concatenate([lo, hi], axis=1) * d
    elif tt == 12:   # Q4_K: f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes
        d = np.ascontiguousarray(a[:, 0:2]).view("<f2").astype(np.float32)
        dmin = np.ascontiguousarray(a[:, 2:4]).view("<f2").astype(np.float32)
        sc = a[:, 4:16]        # (n, 12)
        qs = a[:, 16:144]      # (n, 128)
        out = np.zeros((nblocks, 256), dtype=np.float32)
        for j in range(0, 256, 64):
            g = j // 64
            is_ = g * 2
            s = np.zeros((2, nblocks), np.float32)
            m = np.zeros((2, nblocks), np.float32)
            for k, idx in enumerate((is_, is_ + 1)):
                if idx < 4:
                    s[k] = (sc[:, idx] & 63).astype(np.float32)
                    m[k] = (sc[:, idx + 4] & 63).astype(np.float32)
                else:
                    s[k] = ((sc[:, idx + 4] & 0x0F) |
                            ((sc[:, idx - 4] >> 6) << 4)).astype(np.float32)
                    m[k] = ((sc[:, idx + 4] >> 4) |
                            ((sc[:, idx] >> 6) << 4)).astype(np.float32)
            blk = qs[:, g * 32:(g + 1) * 32]
            out[:, j:j + 32] = d * s[0][:, None] * (blk & 0x0F).astype(np.float32) \
                - dmin * m[0][:, None]
            out[:, j + 32:j + 64] = d * s[1][:, None] * (blk >> 4).astype(np.float32) \
                - dmin * m[1][:, None]
    elif tt == 13:   # Q5_K: f16 d, f16 dmin, 12 scales, 32 qh bits, 128 nibbles
        d = np.ascontiguousarray(a[:, 0:2]).view("<f2").astype(np.float32)
        dmin = np.ascontiguousarray(a[:, 2:4]).view("<f2").astype(np.float32)
        sc = a[:, 4:16]
        qh = a[:, 16:48]
        qs = a[:, 48:176]
        out = np.zeros((nblocks, 256), dtype=np.float32)
        for sb in range(8):
            if sb < 4:
                s = (sc[:, sb] & 63).astype(np.float32)
                m = (sc[:, sb + 4] & 63).astype(np.float32)
            else:
                s = ((sc[:, sb + 4] & 0x0F) |
                     ((sc[:, sb - 4] >> 6) << 4)).astype(np.float32)
                m = ((sc[:, sb + 4] >> 4) |
                     ((sc[:, sb] >> 6) << 4)).astype(np.float32)
            q = qs[:, (sb // 2) * 32:(sb // 2) * 32 + 32]
            lo = (q >> 4) if (sb & 1) else (q & 0x0F)
            hi = ((qh[:, 0:32] >> sb) & 1) << 4
            out[:, sb * 32:(sb + 1) * 32] = \
                d * s[:, None] * (lo | hi).astype(np.float32) - dmin * m[:, None]
    elif tt == 14:   # Q6_K: 128 ql, 64 qh, 16 int8 scales, f16 d
        ql = a[:, 0:128]
        qh = a[:, 128:192]
        scales = a[:, 192:208].view(np.int8).astype(np.float32)
        d = np.ascontiguousarray(a[:, 208:210]).view("<f2").astype(np.float32)
        out = np.zeros((nblocks, 256), dtype=np.float32)
        for nb in range(2):  # two 128-weight halves per 256-block
            ql_h = ql[:, nb * 64:(nb + 1) * 64]
            qh_h = qh[:, nb * 32:(nb + 1) * 32]
            sc_h = scales[:, nb * 8:(nb + 1) * 8]
            base = nb * 128
            for l in range(32):
                is_ = l // 16
                q1 = (((ql_h[:, l] & 0x0F) |
                       (((qh_h[:, l] >> 0) & 3) << 4)).astype(np.int32) - 32)
                q2 = (((ql_h[:, l + 32] & 0x0F) |
                       (((qh_h[:, l] >> 2) & 3) << 4)).astype(np.int32) - 32)
                q3 = (((ql_h[:, l] >> 4) |
                       (((qh_h[:, l] >> 4) & 3) << 4)).astype(np.int32) - 32)
                q4 = (((ql_h[:, l + 32] >> 4) |
                       (((qh_h[:, l] >> 6) & 3) << 4)).astype(np.int32) - 32)
                out[:, base + l + 0] = d[:, 0] * sc_h[:, is_ + 0] * q1
                out[:, base + l + 32] = d[:, 0] * sc_h[:, is_ + 2] * q2
                out[:, base + l + 64] = d[:, 0] * sc_h[:, is_ + 4] * q3
                out[:, base + l + 96] = d[:, 0] * sc_h[:, is_ + 6] * q4
    else:
        raise SystemExit("oracle: no decoder for ggml type %s (%d)"
                         % (GGML.get(tt, "?"), tt))
    return out.reshape(ncol, ne0)


class Container:
    def __init__(self, path):
        self.path = path
        self.info = read_container(path)
        self.kv = self.info["kv"]
        self.tensors = self.info["tensors"]
        self.alignment = self.info["alignment"]
        self._f = open(path, "rb")
        self.data_offset = self.info["data_offset"]

    def col_bytes(self, name):
        t = self.tensors[name]
        bb, bw = BLOCKS.get(t["type"], (None, None))
        if bb is None:
            raise SystemExit("oracle: tensor %s uses unsupported type %s"
                             % (name, GGML.get(t["type"], t["type"])))
        ne0 = t["dims"][0]
        if ne0 % bw:
            raise SystemExit("oracle: tensor %s has ne0=%d not a multiple of the "
                             "block weight count %d" % (name, ne0, bw))
        return ne0 // bw * bb

    def columns(self, name, idx, chunk=256):
        """Decode the given output columns of a [ne0, ne1] tensor.

        Reads only the byte ranges those columns occupy: token_embd is 715 MB of
        blocks and a three-token probe needs three of its 248 320 columns.
        """
        t = self.tensors[name]
        cb = self.col_bytes(name)
        ne0 = t["dims"][0]
        base = self.data_offset + t["offset"]
        out = np.empty((len(idx), ne0), dtype=np.float32)
        for i in range(0, len(idx), chunk):
            sel = np.asarray(idx[i:i + chunk], dtype=np.int64)
            # One seek per chunk when the columns are contiguous (they are, for
            # every caller here), a per-column seek otherwise.
            if len(sel) > 1 and np.all(np.diff(sel) == 1):
                self._f.seek(base + cb * int(sel[0]))
                raw = self._f.read(cb * len(sel))
                if len(raw) != cb * len(sel):
                    raise SystemExit("oracle: short read for %s" % name)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(len(sel), cb)
            else:
                arr = np.empty((len(sel), cb), dtype=np.uint8)
                for k, j in enumerate(sel):
                    self._f.seek(base + cb * int(j))
                    raw = self._f.read(cb)
                    if len(raw) != cb:
                        raise SystemExit("oracle: short read for %s column %d"
                                         % (name, j))
                    arr[k] = np.frombuffer(raw, dtype=np.uint8)
            out[i:i + len(sel)] = dequant_blocks(t["type"], arr, ne0)
        return out

    def row(self, name, j):
        """Decode element `j` along the slowest dimension (token_embd row j)."""
        return self.columns(name, np.array([j], dtype=np.int64))[0]

    def matvec(self, name, x, chunk=512):
        """y = W x for W stored [ne0, ne1] in ggml order."""
        t = self.tensors[name]
        if t["dims"][0] != len(x):
            raise SystemExit("oracle: %s expects %d inputs, got %d"
                             % (name, t["dims"][0], len(x)))
        ncol = t["dims"][1]
        y = np.empty(ncol, dtype=np.float32)
        for i in range(0, ncol, chunk):
            idx = np.arange(i, min(i + chunk, ncol), dtype=np.int64)
            w = self.columns(name, idx, chunk=len(idx))
            y[i:i + len(idx)] = w @ x
        return y

    def matrix(self, name):
        """The whole [ne0, ne1] tensor as fp32 [ne1, ne0] rows-out.

        Only for a matrix whose full fp32 form fits in RAM (the dense FFN). The
        head is 5 GB at fp32, so it goes through matvec() instead, which reads it
        in chunks.
        """
        t = self.tensors[name]
        ne0 = t["dims"][0]
        ne1 = t["dims"][1]
        if len(t["dims"]) != 2:
            raise SystemExit("oracle: matrix(%s) needs a 2-D tensor" % name)
        return self.columns(name, np.arange(ne1, dtype=np.int64), chunk=512)

    def f32(self, name):
        """An F32 tensor in full (norms and small SSM parameters)."""
        t = self.tensors[name]
        if t["type"] != 0:
            raise SystemExit("oracle: %s is %s, not F32"
                             % (name, GGML.get(t["type"], t["type"])))
        n = t["dims"][0]
        for d in t["dims"][1:]:
            n *= d
        self._f.seek(self.data_offset + t["offset"])
        raw = self._f.read(4 * n)
        if len(raw) != 4 * n:
            raise SystemExit("oracle: short read for %s" % name)
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def crosscheck(c, sample_blocks=8):
    """Verify this file's block decoders against gguf-py's, type by type.

    A reference implementation that is only self-consistent proves nothing, so
    the decoders used to build the vectors are checked against a third-party one
    first. Returns a list of dicts for the manifest; raises on disagreement.
    """
    try:
        from gguf import quants, GGMLQuantizationType
    except ImportError:
        return [{"status": "gguf-py not installed: decoders NOT cross-checked"}]
    seen = {}
    for name, t in c.tensors.items():
        if t["type"] in BLOCKS and t["type"] not in seen:
            seen[t["type"]] = name
    out = []
    for tt in sorted(seen):
        name = seen[tt]
        bb, bw = BLOCKS[tt]
        n = bb * sample_blocks
        c._f.seek(c.data_offset + c.tensors[name]["offset"])
        raw = np.frombuffer(c._f.read(n), dtype=np.uint8)
        mine = dequant_blocks(tt, raw, bw * sample_blocks).reshape(-1)
        try:
            theirs = quants.dequantize(raw.copy(),
                                       GGMLQuantizationType(tt)).astype(np.float32)
        except Exception as e:  # noqa: BLE001 -- report, never guess
            out.append({"type": GGML.get(tt, tt), "tensor": name,
                        "status": "gguf-py refused: %s" % e})
            continue
        theirs = np.asarray(theirs, dtype=np.float32).reshape(-1)
        if theirs.size != mine.size:
            out.append({"type": GGML.get(tt, tt), "tensor": name,
                        "status": "size mismatch %d vs %d" % (mine.size, theirs.size)})
            continue
        diff = float(np.max(np.abs(theirs - mine))) if mine.size else 0.0
        out.append({"type": GGML.get(tt, tt), "tensor": name,
                    "blocks": sample_blocks,
                    "max_abs_diff_vs_gguf_py": diff,
                    "status": "identical" if diff == 0.0 else "DIFFERS"})
        if diff != 0.0:
            raise SystemExit("oracle: decoder for %s disagrees with gguf-py by %g "
                             "on %s -- refusing to write a reference"
                             % (GGML.get(tt, tt), diff, name))
    return out


def l2norm(x, eps=1e-6):
    """Per-row L2 normalisation, the reference's formula: x / sqrt(sum(x^2) + eps)."""
    s = (x.astype(np.float32) ** 2).sum(axis=-1, keepdims=True, dtype=np.float32)
    return x * (np.float32(1.0) / np.sqrt(s + np.float32(eps)))


def sigmoid(x):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x.astype(np.float32)))


def softplus(x):
    x = x.astype(np.float32)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(np.float32)


def rmsnorm_gated(x, weight, z, eps):
    """Qwen3NextRMSNormGated: norm over the last dim, then weight, then silu(gate)."""
    var = (x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True, dtype=np.float32)
    y = x * (np.float32(1.0) / np.sqrt(var + np.float32(eps)))
    y = weight.astype(np.float32) * y
    z = z.astype(np.float32)
    return (y * (z / (np.float32(1.0) + np.exp(-z)))).astype(np.float32)


def recurrent_layer(c, layer, tokens, eps, num_k_heads, num_v_heads, head_k_dim,
                    head_v_dim, conv_kernel, emit, note, x_in=None):
    """One GatedDeltaNet layer, one token at a time, interleaved layout.

    Order and arithmetic follow the reference implementations read on this
    machine: transformers' Qwen3NextGatedDeltaNet (torch_recurrent_gated_delta_rule)
    and llama.cpp's src/models/qwen35.cpp + ggml-cpu gated_delta_net. Where the
    two agree there is no reading left to argue about, and where they disagree the
    disagreement is named in the manifest.
    """
    b = "blk.%d." % layer
    key_dim = num_k_heads * head_k_dim     # 2048
    value_dim = num_v_heads * head_v_dim   # 6144
    conv_dim = 2 * key_dim + value_dim     # 10240

    T = len(tokens)
    # 1. input: embeddings through attn_norm, or the caller's hidden state. The
    # per-layer probe embeds (that is what the layer_* vectors assume) and the trunk
    # passes the residual stream, which is a *different* input -- the same block,
    # called from two places.
    if x_in is None:
        emb = np.stack([c.row("token_embd.weight", t) for t in tokens])
        an = c.f32(b + "attn_norm.weight")
        x = np.stack([rmsnorm(emb[i], an, eps) for i in range(T)])
    else:
        x = x_in
    emit("layer_x", x.reshape(-1))

    # 2. fused qkv projection, then a causal depthwise conv1d + silu
    qkv = np.stack([c.matvec(b + "attn_qkv.weight", x[i]) for i in range(T)])   # [T, 10240]
    emit("layer_qkv", qkv.reshape(-1))
    w = c.f32(b + "ssm_conv1d.weight").reshape(conv_kernel, conv_dim)           # [4, 10240]
    conv_raw = np.zeros((T, conv_dim), dtype=np.float32)
    for t in range(T):
        for k in range(conv_kernel):
            src = t - (conv_kernel - 1) + k
            if src >= 0:
                conv_raw[t] += w[k] * qkv[src]
    emit("layer_conv_raw", conv_raw.reshape(-1))
    conv = (conv_raw / (np.float32(1.0) + np.exp(-conv_raw))).astype(np.float32)  # silu
    emit("layer_conv_silu", conv.reshape(-1))

    q = conv[:, 0:key_dim].reshape(T, num_k_heads, head_k_dim)
    k = conv[:, key_dim:2 * key_dim].reshape(T, num_k_heads, head_k_dim)
    v = conv[:, 2 * key_dim:].reshape(T, num_v_heads, head_v_dim)

    scale = np.float32(1.0) / np.sqrt(np.float32(head_k_dim))
    q = (l2norm(q) * scale).astype(np.float32)
    k = l2norm(k)
    emit("layer_q_l2", q.reshape(-1))
    emit("layer_k_l2", k.reshape(-1))

    # 3. gates. g = ssm_a * softplus(a + dt_bias); ssm_a stores -exp(A_log).
    beta = sigmoid(np.stack([c.matvec(b + "ssm_beta.weight", x[i]) for i in range(T)]))
    a = np.stack([c.matvec(b + "ssm_alpha.weight", x[i]) for i in range(T)])
    ssm_a = c.f32(b + "ssm_a")
    dt = c.f32(b + "ssm_dt.bias")
    g = (ssm_a[None, :] * softplus(a + dt[None, :])).astype(np.float32)
    emit("layer_beta", beta.reshape(-1))
    emit("layer_g", g.reshape(-1))

    # 4. the delta rule, one token at a time: state orientation S[k, v].
    rep = num_v_heads // num_k_heads
    q_rep = np.repeat(q, rep, axis=1)    # repeat_interleave over heads
    k_rep = np.repeat(k, rep, axis=1)
    state = np.zeros((num_v_heads, head_k_dim, head_v_dim), dtype=np.float32)
    o = np.zeros((T, num_v_heads, head_v_dim), dtype=np.float32)
    for t in range(T):
        decay = np.exp(g[t]).astype(np.float32)          # [num_v_heads]
        state *= decay[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", state, k_rep[t])          # [H, Dv]
        delta = ((v[t] - kv_mem) * beta[t][:, None]).astype(np.float32)
        state += k_rep[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", state, q_rep[t])
    emit("layer_o", o.reshape(-1))
    emit("layer_state", state.reshape(-1))

    # 5. gated normalisation, then the output projection
    z = np.stack([c.matvec(b + "attn_gate.weight", x[i]) for i in range(T)])
    z = z.reshape(T, num_v_heads, head_v_dim)
    nw = c.f32(b + "ssm_norm.weight")
    gated = rmsnorm_gated(o, nw, z, eps)
    emit("layer_gated", gated.reshape(-1))
    out = np.stack([c.matvec(b + "ssm_out.weight", gated[i].reshape(-1)) for i in range(T)])
    emit("layer_out", out.reshape(-1))
    note("delta-rule scale", "1/sqrt(head_k_dim) = %.6f" % float(scale))
    note("decay per token", "exp(g) in [%.6f, %.6f]" % (float(np.exp(g).min()), float(np.exp(g).max())))
    return out.astype(np.float32)


def imrope_axis(pair, sections):
    """Which position axis a rope *pair* index reads under ggml's IMROPE.

    Section lengths count cos/sin PAIRS, not dims (ggml.h: "sum of 4 sections are
    expected to be n_dims/2"), and IMROPE interleaves the axes rather than
    laying them out in blocks: `sector = pair % sum(sections)`, then axis t for
    `sector % 3 == 0`, h for 1, w for 2, with the extra axis as the fallback.
    Source: the ggml.h comment on GGML_ROPE_TYPE_IMROPE and ggml-cpu's
    `rope_yarn` (ops.cpp, the `is_imrope` branch). The axis BARELY matters for a
    text-only batch -- every position is the same token -- but it is computed
    here rather than assumed away, because "it reduces to plain rope" is exactly
    the kind of claim that hides a wrong section reading.
    """
    sect_dims = sections[0] + sections[1] + sections[2] + sections[3]
    if sect_dims <= 0:
        return 0
    sector = pair % sect_dims
    if sector % 3 == 1 and sector < 3 * sections[1]:
        return 1
    if sector % 3 == 2 and sector < 3 * sections[2]:
        return 2
    if sector % 3 == 0 and sector < 3 * sections[0]:
        return 0
    return 3


def attention_layer(c, layer, tokens, eps, n_head, n_head_kv, head_dim, rope_dims,
                    rope_base, sections, emit, note, x_in=None):
    """One gated full-attention layer, prefill, causal, one token at a time.

    Order from the two references read on this machine, which agree:
    llama.cpp's `build_layer_attn` (src/models/qwen35.cpp:260-330) and
    transformers' Qwen3NextAttention. Joint QG projection -> QG split, where the
    layout is [q_head(head_dim) | gate_head(head_dim)] interleaved per head ->
    Q RMS norm -> K/V projections -> K RMS norm -> partial RoPE (NEOX ordering
    over rope_dims of head_dim) -> causal attention -> multiply by
    sigmoid(gate) -> output projection.
    """
    b = "blk.%d." % layer
    q_dim = n_head * head_dim
    rep = n_head // n_head_kv
    T = len(tokens)
    positions = np.arange(T, dtype=np.float32)

    if x_in is None:
        emb = np.stack([c.row("token_embd.weight", t) for t in tokens])
        an = c.f32(b + "attn_norm.weight")
        x = np.stack([rmsnorm(emb[i], an, eps) for i in range(T)])
    else:
        x = x_in
    emit("layer_x", x.reshape(-1))

    qg = np.stack([c.matvec(b + "attn_q.weight", x[i]) for i in range(T)])
    emit("layer_qg", qg.reshape(-1))
    qg = qg.reshape(T, n_head, 2 * head_dim)
    q = np.ascontiguousarray(qg[:, :, :head_dim])
    gate = np.ascontiguousarray(qg[:, :, head_dim:])

    qn = c.f32(b + "attn_q_norm.weight")
    kn = c.f32(b + "attn_k_norm.weight")
    q = np.stack([rmsnorm(q[t, h], qn, eps) for t in range(T)
                  for h in range(n_head)]).reshape(T, n_head, head_dim)
    emit("layer_q_norm", q.reshape(-1))

    k = np.stack([c.matvec(b + "attn_k.weight", x[i]) for i in range(T)]
                 ).reshape(T, n_head_kv, head_dim)
    v = np.stack([c.matvec(b + "attn_v.weight", x[i]) for i in range(T)]
                 ).reshape(T, n_head_kv, head_dim)
    k = np.stack([rmsnorm(k[t, h], kn, eps) for t in range(T)
                  for h in range(n_head_kv)]).reshape(T, n_head_kv, head_dim)
    emit("layer_k_norm", k.reshape(-1))
    emit("layer_v", v.reshape(-1))

    # Partial RoPE over the first rope_dims of each head, in NEOX (rotate_half)
    # order: pair i is (dim i, dim i + rope_dims/2). ggml does the same in
    # `rotate_pairs` with n_offset = rope_dims/2, and HF's apply_rotary_pos_emb
    # splits the head at `rotary_dim` and applies rotate_half to the first part.
    half = rope_dims // 2
    axis = [imrope_axis(i, sections) for i in range(half)]
    axis_pos = [positions, positions, positions, positions]  # text-only: one token position
    th = np.empty((T, half), dtype=np.float32)
    for i in range(half):
        th[:, i] = axis_pos[axis[i]] * np.float32(rope_base) ** np.float32(-2.0 * i / rope_dims)
    cos = np.cos(th).astype(np.float32)
    sin = np.sin(th).astype(np.float32)

    def rope(a):
        out = a.copy()
        rot = a[:, :, :rope_dims]
        x1 = rot[:, :, :half]
        x2 = rot[:, :, half:]
        out[:, :, :half] = (x1 * cos[:, None, :] - x2 * sin[:, None, :]).astype(np.float32)
        out[:, :, half:rope_dims] = (x1 * sin[:, None, :] + x2 * cos[:, None, :]).astype(np.float32)
        return out

    q = rope(q)
    k = rope(k)
    emit("layer_q_rope", q.reshape(-1))
    emit("layer_k_rope", k.reshape(-1))

    scale = np.float32(1.0) / np.sqrt(np.float32(head_dim))
    scores = np.zeros((T, n_head, T), dtype=np.float32)
    attn = np.zeros((T, n_head, head_dim), dtype=np.float32)
    for t in range(T):
        for h in range(n_head):
            s = (k[:t + 1, h // rep, :] @ q[t, h, :]).astype(np.float32) * scale
            s = (s - s.max()).astype(np.float32)
            e = np.exp(s).astype(np.float32)
            w = (e / e.sum()).astype(np.float32)
            scores[t, h, :t + 1] = w
            attn[t, h] = w @ v[:t + 1, h // rep, :]
    emit("layer_scores", scores.reshape(-1))
    emit("layer_attn", attn.reshape(-1))

    gated = (attn * sigmoid(gate)).astype(np.float32)
    emit("layer_gated", gated.reshape(-1))
    out = np.stack([c.matvec(b + "attn_output.weight", gated[i].reshape(-1))
                    for i in range(T)])
    emit("layer_out", out.reshape(-1))

    axes = {}
    for a in axis:
        axes[a] = axes.get(a, 0) + 1
    note("attention scale", "1/sqrt(head_dim) = %.6f" % float(scale))
    note("rope", "partial NEOX over %d of %d dims, base %g" % (rope_dims, head_dim, rope_base))
    note("rope sections", "%s -> %s pairs per axis, of %d; text-only, so every axis "
         "carries the token position and this is plain NEOX rope"
         % (list(sections), axes, half))
    note("qg layout", "[q_head(%d) | gate_head(%d)] interleaved per head" % (head_dim, head_dim))
    return out.astype(np.float32)


def trunk_forward(c, kv, tokens, emit, note):
    """The whole causal stack, one layer at a time, with only that layer resident.

    Composition read from llama.cpp's qwen35 graph (src/models/qwen35.cpp, the
    layer loop and `build_layer_ffn`, on this machine):

        for il in [0, n_layer):
            inp = cur
            cur = block(rmsnorm(cur, attn_norm[il]))   # the block norms its own input
            cur = cur + inp                             # "attn_residual"
            res = cur
            cur = dense_ffn(rmsnorm(cur, post_attention_norm[il]))
            cur = cur + res                             # "post_ffn"
        logits = output.weight @ rmsnorm(cur, output_norm)

    `n_layer` is `block_count - nextn_predict_layers`: the MTP block is "loaded as an
    extra decoder block but not executed in the main pass", so blk.64 is not in the
    stack the logits come from. There is no logit scaling and no softcap.

    The FFN is dense SwiGLU (llama.cpp asserts `ffn_gate_inp == nullptr`: "Qwen3.5
    does not use MoE FFN"). Its three matrices are materialised in fp32 once per
    layer (~1.1 GB here) and dropped; every other tensor is read in chunks by
    `matvec`, so the whole forward never holds more than one layer.
    """
    p = "qwen35."
    eps = float(kv[p + "attention.layer_norm_rms_epsilon"])
    E = int(kv[p + "embedding_length"])
    n_all = int(kv[p + "block_count"])
    nextn = int(kv.get(p + "nextn_predict_layers", 0))
    n_layer = n_all - nextn
    n_ff = int(kv[p + "feed_forward_length"])
    n_vocab = int(c.tensors["token_embd.weight"]["dims"][1])
    rec = kv.get(p + "attention.recurrent_layers")
    interval = int(kv.get(p + "full_attention_interval", 0))
    if rec is None and not interval:
        raise SystemExit("oracle: neither recurrent_layers nor full_attention_interval")
    nk = int(kv[p + "ssm.group_count"])
    nv = int(kv[p + "ssm.time_step_rank"])
    hs = int(kv[p + "ssm.state_size"])
    ck = int(kv[p + "ssm.conv_kernel"])
    n_head = int(kv[p + "attention.head_count"])
    n_head_kv = int(kv[p + "attention.head_count_kv"])
    head_dim = int(kv[p + "attention.key_length"])
    rope_dims = int(kv[p + "rope.dimension_count"])
    rope_base = float(kv[p + "rope.freq_base"])
    sections = [int(v) for v in kv.get(p + "rope.dimension_sections", [0, 0, 0, 0])]

    T = len(tokens)
    h = np.stack([c.row("token_embd.weight", t) for t in tokens]).astype(np.float32)
    if h.shape != (T, E):
        raise SystemExit("oracle: token_embd row is %s, expected (%d, %d)" % (h.shape, T, E))
    emit("trunk_in", h.reshape(-1))

    n_rec = 0
    for L in range(n_layer):
        b = "blk.%d." % L
        is_rec = bool(rec[L]) if rec is not None else ((L + 1) % interval != 0)
        n_rec += 1 if is_rec else 0

        an = c.f32(b + "attn_norm.weight")
        x = np.stack([rmsnorm(h[i], an, eps) for i in range(T)])
        noop = lambda *a, **k: None
        if is_rec:
            out = recurrent_layer(c, L, tokens, eps, nk, nv, hs, hs, ck, noop, noop, x_in=x)
        else:
            out = attention_layer(c, L, tokens, eps, n_head, n_head_kv, head_dim,
                                  rope_dims, rope_base, sections, noop, noop, x_in=x)
        h = (h + out).astype(np.float32)
        # Two emission points per layer, because they are the two places a trunk
        # defect shows up differently: `trunk_mix<L>` is the stream after the mixer's
        # residual (isolating the block), `trunk_h<L>` after the FFN's (isolating the
        # dense FFN). The engine emits both from the same two buffers.
        emit("trunk_mix%d" % L, h.reshape(-1))

        pan = c.f32(b + "post_attention_norm.weight")
        fi = np.stack([rmsnorm(h[i], pan, eps) for i in range(T)])
        wg = c.matrix(b + "ffn_gate.weight")
        wu = c.matrix(b + "ffn_up.weight")
        gate = (fi @ wg.T).astype(np.float32)
        up = (fi @ wu.T).astype(np.float32)
        act = (gate / (np.float32(1.0) + np.exp(-gate)) * up).astype(np.float32)
        wd = c.matrix(b + "ffn_down.weight")
        h = (h + (act @ wd.T).astype(np.float32)).astype(np.float32)
        emit("trunk_h%d" % L, h.reshape(-1))

    on = c.f32("output_norm.weight")
    normed = np.stack([rmsnorm(h[i], on, eps) for i in range(T)])
    emit("trunk_norm", normed.reshape(-1))

    # The head is 248320 x 5120: 5 GB at fp32, so it goes through the chunked
    # matvec rather than matrix().
    logits = np.stack([c.matvec("output.weight", normed[t]) for t in range(T)])
    emit("trunk_logits", logits.reshape(-1))

    note("trunk", "%d causal layer(s) = block_count %d - nextn_predict_layers %d; "
                   "blk.%d carries the MTP tensors and is NOT in the main pass"
         % (n_layer, n_all, nextn, n_all - 1))
    note("trunk kinds", "%d recurrent (SSM) + %d full attention" % (n_rec, n_layer - n_rec))
    note("ffn", "dense SwiGLU, no router: down(silu(gate(x)) * up(x)), x = the "
                 "post_attention_norm output")
    note("head", "output_norm + output.weight [%d, %d] (untied), no scaling, no softcap"
         % (E, n_vocab))
    note("trunk argmax", "per position: %s" % [int(np.argmax(logits[t])) for t in range(T)])
    return T, logits


def rmsnorm(x, w, eps):
    ss = np.float32(0.0)
    for v in x:
        ss = np.float32(ss + np.float32(v * v))
    scale = np.float32(1.0) / np.float32(np.sqrt(np.float32(ss / np.float32(len(x)) + np.float32(eps))))
    return (x * scale) * w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", required=True,
                    help="comma-separated token ids the engine must use")
    ap.add_argument("--recurrent", action="store_true",
                    help="compute a recurrent (GatedDeltaNet) layer end to end "
                         "instead of only its input projections")
    ap.add_argument("--forward", action="store_true",
                    help="the whole trunk: every causal layer with its residuals, "
                         "the dense FFN, output_norm and the untied head, emitting "
                         "trunk_in, trunk_h<L> for every layer, trunk_norm and "
                         "trunk_logits")
    ap.add_argument("--attention", action="store_true",
                    help="compute a gated full-attention layer end to end "
                         "instead of only its input projections")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    c = Container(args.gguf)
    cross = crosscheck(c)
    for row in cross:
        print("oracle: dequant %-6s %-28s %s"
              % (row.get("type"), row.get("tensor", ""), row.get("status")))
    kv = c.kv
    arch = kv.get("general.architecture")
    if arch != "qwen35":
        raise SystemExit("oracle: architecture is '%s', not 'qwen35'" % arch)
    if kv.get("tokenizer.ggml.model") != "gpt2":
        raise SystemExit("oracle: tokenizer.ggml.model is '%s', expected 'gpt2'"
                         % kv.get("tokenizer.ggml.model"))

    p = "qwen35."
    eps = float(kv[p + "attention.layer_norm_rms_epsilon"])
    embd = int(kv[p + "embedding_length"])
    n_layer = int(kv[p + "block_count"])
    rec = kv.get(p + "attention.recurrent_layers")
    interval = int(kv.get(p + "full_attention_interval", 0))
    if rec is None and not interval:
        raise SystemExit("oracle: neither recurrent_layers nor full_attention_interval")
    kind = "recurrent" if (rec[args.layer] if rec is not None
                           else (args.layer + 1) % interval != 0) else "attention"

    tokens = [int(x) for x in args.tokens.split(",")]
    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "oracle": "tools/ggufmeta/qwen35_oracle.py",
        "file": os.path.abspath(args.gguf),
        "arch": arch,
        "layer": args.layer,
        "layer_kind": kind,
        "tokens": tokens,
        "token_strings": None,   # filled below if the vocab is present
        "reshape": "dims[0] is the fast axis in memory",
        "dequant_crosscheck_vs_gguf_py": cross,
        "vectors": [],
    }

    def emit(name, vec):
        path = os.path.join(args.out, name + ".f32")
        vec = np.ascontiguousarray(vec, dtype="<f4")
        with open(path, "wb") as fh:
            fh.write(vec.tobytes())
        manifest["vectors"].append({"name": name, "shape": list(vec.shape),
                                    "file": name + ".f32"})

    def call(name, x):
        return c.matvec(name, x)

    notes = {}

    def note(key, value):
        notes[key] = value

    if args.forward:
        nextn = int(kv.get(p + "nextn_predict_layers", 0))
        manifest["mode"] = "trunk"
        manifest["trunk"] = {
            "n_layer_all": n_layer, "nextn_predict_layers": nextn,
            "n_layer_main": n_layer - nextn, "n_embd": embd,
            "n_ff": int(kv[p + "feed_forward_length"]),
            "n_vocab": int(c.tensors["token_embd.weight"]["dims"][1]),
        }
        manifest["reading"] = {
            "layer order": "for il in [0, n_layer): inp = cur; cur = block("
                           "rmsnorm(cur, attn_norm[il])); cur = cur + inp; res = cur; "
                           "cur = ffn(rmsnorm(cur, post_attention_norm[il])); "
                           "cur = cur + res",
            "nextn": "the MTP block (blk.%d here) is loaded as an extra decoder "
                      "block and NOT executed in the main pass" % (n_layer - 1),
            "ffn": "dense SwiGLU, no router (llama.cpp asserts ffn_gate_inp is null)",
            "head": "output_norm then output.weight; no logit scaling, no softcap",
            "source": "llama.cpp src/models/qwen35.cpp (the layer loop, "
                      "build_layer_ffn, the output_norm + lm_head tail)",
        }
        T, _logits = trunk_forward(c, kv, tokens, emit, note)
        what = "trunk forward pass"
    elif args.recurrent:
        if kind != "recurrent":
            raise SystemExit("oracle: layer %d is %s, not recurrent; --recurrent "
                             "refuses to compute the other kind"
                             % (args.layer, kind))
        nk = int(kv[p + "ssm.group_count"])
        nv = int(kv[p + "ssm.time_step_rank"])
        hs = int(kv[p + "ssm.state_size"])
        ck = int(kv[p + "ssm.conv_kernel"])
        manifest["mode"] = "recurrent_layer"
        manifest["ssm"] = {"num_k_heads": nk, "num_v_heads": nv,
                           "head_dim": hs, "conv_kernel": ck,
                           "value_dim": nv * hs, "conv_dim": 2 * nk * hs + nv * hs}
        manifest["reading"] = {
            "ssm_a": "stored as -exp(A_log): g = ssm_a * softplus(alpha + dt_bias)",
            "conv": "causal depthwise conv1d over the fused qkv, then SiLU",
            "qkv_order": "[q_all | k_all | v_all]",
            "qk_norm": "l2norm per head (eps), q scaled by 1/sqrt(head_dim)",
            "source": "transformers Qwen3NextGatedDeltaNet + llama.cpp "
                      "src/models/qwen35.cpp and ggml-cpu gated_delta_net",
        }
        recurrent_layer(c, args.layer, tokens, eps, nk, nv, hs, hs, ck, emit, note)
        T = len(tokens)
        what = "recurrent layer"
    elif args.attention:
        if kind != "attention":
            raise SystemExit("oracle: layer %d is %s, not full attention; --attention "
                             "refuses to compute the other kind" % (args.layer, kind))
        n_head = int(kv[p + "attention.head_count"])
        n_head_kv = int(kv[p + "attention.head_count_kv"])
        head_dim = int(kv[p + "attention.key_length"])
        rope_dims = int(kv[p + "rope.dimension_count"])
        rope_base = float(kv[p + "rope.freq_base"])
        sections = [int(v) for v in kv.get(p + "rope.dimension_sections", [0, 0, 0, 0])]
        manifest["mode"] = "attention_layer"
        manifest["attention"] = {"n_head": n_head, "n_head_kv": n_head_kv,
                                 "head_dim": head_dim, "rope_dims": rope_dims,
                                 "rope_base": rope_base, "rope_sections": sections,
                                 "q_dim": n_head * head_dim,
                                 "kv_dim": n_head_kv * head_dim}
        manifest["reading"] = {
            "qg": "attn_q is [q_head | gate_head] interleaved per head; the gate "
                  "multiplies the attention output elementwise after sigmoid",
            "qk_norm": "RMS norm over head_dim, weight attn_q_norm / attn_k_norm",
            "rope": "partial: rope_dims of head_dim, NEOX (rotate_half) pairs, "
                    "positions are the token indices",
            "attention": "causal, GQA by head groups, scale 1/sqrt(head_dim)",
            "source": "llama.cpp src/models/qwen35.cpp build_layer_attn + "
                      "transformers Qwen3NextAttention",
        }
        attention_layer(c, args.layer, tokens, eps, n_head, n_head_kv, head_dim,
                            rope_dims, rope_base, sections, emit, note)
        T = len(tokens)
        what = "attention layer"
    if args.forward or args.recurrent or args.attention:
        manifest["tokens"] = tokens
        manifest["steps"] = T
        manifest["notes"] = notes
        with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print("oracle: wrote %d vector(s) for %s %d (%d token step(s)) "
              "into %s" % (len(manifest["vectors"]), what, args.layer, T, args.out))
        for k, v in notes.items():
            print("oracle: %-16s %s" % (k, v))
        return 0

    b = "blk.%d." % args.layer
    emb = [c.row("token_embd.weight", t) for t in tokens]
    for t, e in zip(tokens, emb):
        emit("embed_%d" % t, e)
    an = c.f32(b + "attn_norm.weight")
    xn = [rmsnorm(e, an, eps) for e in emb]
    for t, v in zip(tokens, xn):
        emit("xnorm_%d" % t, v)
    if kind == "recurrent":
        for t, v in zip(tokens, xn):
            emit("qkv_%d" % t, call(b + "attn_qkv.weight", v))
            emit("gate_%d" % t, call(b + "attn_gate.weight", v))
    else:
        for t, v in zip(tokens, xn):
            emit("q_%d" % t, call(b + "attn_q.weight", v))
            emit("k_%d" % t, call(b + "attn_k.weight", v))
            emit("v_%d" % t, call(b + "attn_v.weight", v))

    # gguf_meta's reader keeps only the first 12 strings of a long array (it has
    # to read them all to stay in sync, but does not retain them), so the token
    # text is only available when the ids happen to be small. Absent is fine:
    # the engine's own decode is cross-checked against the vocab by --dump-tokens.
    tokens_meta = kv.get("tokenizer.ggml.tokens")
    if isinstance(tokens_meta, list) and max(tokens) < len(tokens_meta) - 1:
        manifest["token_strings"] = [tokens_meta[t] for t in tokens]
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("oracle: wrote %d vector(s) for layer %d (%s) into %s"
          % (len(manifest["vectors"]), args.layer, kind, args.out))
    print("oracle: tokens %s -> %s" % (tokens, manifest["token_strings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
