#!/usr/bin/env python3
"""An independent numpy forward pass for Qwen3-MoE, on top of gguf-py.

Why this file exists
--------------------
The engine's forward pass and this one share no code: this one dequantises with
ggml's own python dequantisers (gguf-py) and multiplies with numpy, while the
engine decodes blocks by hand inside its inner product. When the two agree on an
intermediate, that intermediate is corroborated by a second implementation. When
they disagree, the disagreement is real and is localised to one tensor.

It is written to be *checked against llama.cpp*, not against itself: run it on
the prompt the reference binary was given and see whether the argmax tokens come
out the same. A reference that is only self-consistent proves nothing.

Usage
-----
  python tools/ref_qwen3moe.py <model.gguf> [--tokens 785,6722,...] [--dump DIR]

  --tokens   prompt ids. Default is "The capital of France is" in this model's
             tokenizer, which is verified against the engine's own tokenizer.
  --dump     write every named intermediate to DIR/<name>.f32, in the layout
             tools/hidden_diff.py expects, so the engine can be bisected.
"""

import argparse
import os
import sys
import time

import numpy as np
import gguf
from gguf.quants import dequantize


class Model:
    """Qwen3-MoE, exactly the graph llama.cpp's src/models/qwen3moe.cpp builds."""

    def __init__(self, path, dump_dir=None):
        t0 = time.time()
        self.reader = gguf.GGUFReader(path)
        self.T = {t.name: t for t in self.reader.tensors}
        self.kv = self.reader.fields
        self.dump_dir = dump_dir
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        # The greedy continuation re-runs the whole prompt, so a later forward
        # would overwrite the prefill's intermediates with a longer sequence
        # and the diff would compare two different shapes. The engine gates its
        # dumps on the first call (pos_ == 0); this gates on the same thing.
        self._forward_calls = 0

        arch = self.get_str("general.architecture")
        if arch != "qwen3moe":
            raise SystemExit(f"ref: architecture is '{arch}', expected 'qwen3moe'")
        p = arch + "."
        self.n_layer = self.get_int(p + "block_count")
        self.n_embd = self.get_int(p + "embedding_length")
        self.n_head = self.get_int(p + "attention.head_count")
        self.n_head_kv = self.get_int(p + "attention.head_count_kv")
        self.head_dim = self.get_int(p + "attention.key_length")
        self.n_ff = self.get_int(p + "expert_feed_forward_length")
        self.n_expert = self.get_int(p + "expert_count")
        self.top_k = self.get_int(p + "expert_used_count")
        self.eps = self.get_float(p + "attention.layer_norm_rms_epsilon")
        self.rope_base = self.get_float(p + "rope.freq_base")
        # vocab_size is optional in the spec; the embedding's slowest dim is
        # the authority when it is absent, exactly as the engine treats it.
        self.n_vocab = (
            self.get_int("vocab_size")
            if "vocab_size" in self.kv
            else int(self.T["token_embd.weight"].shape[1])
        )
        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_head_kv * self.head_dim

        # Precomputed rotary tables: (half,) each.
        half = self.head_dim // 2
        i = np.arange(half, dtype=np.float64)
        self.inv_freq = np.power(float(self.rope_base), -2.0 * i / self.head_dim)
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))
        self.group = self.n_head // self.n_head_kv

        self.describe()
        print(f"ref: reader ready in {time.time() - t0:.1f}s")

    # -- metadata helpers -------------------------------------------------
    def get_int(self, key):
        return int(self.kv[key].contents())

    def get_float(self, key):
        return float(self.kv[key].contents())

    def get_str(self, key):
        v = self.kv[key].contents()
        return v.decode() if isinstance(v, bytes) else str(v)

    def describe(self):
        print(
            f"ref geometry: layers={self.n_layer} hidden={self.n_embd} "
            f"heads={self.n_head}/{self.n_head_kv} head_dim={self.head_dim} "
            f"q_dim={self.q_dim} kv_dim={self.kv_dim}\n"
            f"  experts={self.n_expert} top_k={self.top_k} expert_ff={self.n_ff} "
            f"vocab={self.n_vocab} rms_eps={self.eps:.3g} rope_base={self.rope_base:.6g}\n"
            f"  rope type: NEOX (ggml pair (x[i], x[i+head_dim/2]))"
        )

    # -- weight access ----------------------------------------------------
    def flat(self, name):
        """Dequantised f32 payload in ggml order (fastest dim first)."""
        return self.flat3(name=name).reshape(-1)

    def flat3(self, name=None, t=None):
        t = t if t is not None else self.T[name]
        return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)

    def mat2(self, name, x):
        """y = W . x for a 2-D ggml tensor [ne0, ne1].

        ggml's fastest dimension is ne0, so the flat array reshaped to
        [ne1, ne0] has one *row* per output index.
        """
        t = self.T[name]
        ne0, ne1 = int(t.shape[0]), int(t.shape[1])
        return (self.flat3(t=t).reshape(ne1, ne0) @ x).astype(np.float32)

    # -- primitives -------------------------------------------------------
    def rmsnorm(self, x, name):
        """x is (..., width); the weight applies along the last axis."""
        w = self.flat(name)
        xf = x.astype(np.float64)
        ss = np.einsum("...i,...i->...", xf, xf)
        scale = (1.0 / np.sqrt(ss / x.shape[-1] + self.eps)).astype(np.float32)
        return (x * scale[..., None] * w).astype(np.float32)

    def rmsnorm_heads(self, v, w):
        """Per-head RMSNorm over head_dim, for (n_head, head_dim)."""
        vf = v.astype(np.float64)
        ss = np.einsum("hd,hd->h", vf, vf)
        scale = (1.0 / np.sqrt(ss / self.head_dim + self.eps)).astype(np.float32)
        return (v * scale[..., None] * w).astype(np.float32)

    def rope_neox(self, v, pos):
        """Rotate (n_head, head_dim) in place; NEOX pairs x[i] with x[i+d/2]."""
        d, half = self.head_dim, self.head_dim // 2
        theta = pos * self.inv_freq
        c = np.cos(theta).astype(np.float32)
        s = np.sin(theta).astype(np.float32)
        x0 = v[:, :half].copy()
        x1 = v[:, half:].copy()
        v[:, :half] = x0 * c - x1 * s
        v[:, half:] = x0 * s + x1 * c
        return v

    def softmax(self, v, axis=-1):
        m = v.max(axis=axis, keepdims=True)
        e = np.exp(v - m)
        return e / e.sum(axis=axis, keepdims=True)

    def attention(self, q, kv_k, kv_v, n):
        """Causal attention for one layer over n tokens."""
        qh = q.reshape(n, self.n_head, self.head_dim)
        K = kv_k.reshape(n, self.n_head_kv, self.head_dim)
        V = kv_v.reshape(n, self.n_head_kv, self.head_dim)
        g = self.group
        Ke = K[:, np.repeat(np.arange(self.n_head_kv), g), :]
        Ve = V[:, np.repeat(np.arange(self.n_head_kv), g), :]
        out = np.zeros_like(qh)
        for t in range(n):
            sc = (np.einsum("thd,hd->th", Ke[: t + 1], qh[t]) * self.scale).astype(np.float32)
            p = self.softmax(sc, axis=0)
            out[t] = np.einsum("th,thd->hd", p, Ve[: t + 1])
        return out.reshape(n, self.q_dim)

    # -- the forward pass -------------------------------------------------
    def forward(self, tokens):
        H = self.n_embd
        n = len(tokens)
        t0 = time.time()
        self._forward_calls += 1

        emb = self.flat("token_embd.weight").reshape(self.n_vocab, H)
        x = emb[np.asarray(tokens, dtype=np.int64)].astype(np.float32).copy()
        self.dump("00_embed", x)

        # KV cache for this prompt only: (n_layer, n, dim).
        kv_k = np.zeros((self.n_layer, n, self.kv_dim), dtype=np.float32)
        kv_v = np.zeros((self.n_layer, n, self.kv_dim), dtype=np.float32)

        for il in range(self.n_layer):
            b = f"blk.{il}."
            xn = self.rmsnorm(x, b + "attn_norm.weight")
            if il == 0:
                self.dump("01_attn_norm", xn)

            q = np.stack([self.mat2(b + "attn_q.weight", xn[t]) for t in range(n)])
            k = np.stack([self.mat2(b + "attn_k.weight", xn[t]) for t in range(n)])
            v = np.stack([self.mat2(b + "attn_v.weight", xn[t]) for t in range(n)])

            qw = self.flat(b + "attn_q_norm.weight")
            kw = self.flat(b + "attn_k_norm.weight")
            for t in range(n):
                qh = self.rmsnorm_heads(q[t].reshape(self.n_head, self.head_dim), qw)
                kh = self.rmsnorm_heads(k[t].reshape(self.n_head_kv, self.head_dim), kw)
                self.rope_neox(qh, t)
                self.rope_neox(kh, t)
                q[t] = qh.reshape(-1)
                k[t] = kh.reshape(-1)
            if il == 0:
                self.dump("02_q_rope", q)
                self.dump("03_k_rope", k)
                self.dump("04_v", v)

            kv_k[il] = k
            kv_v[il] = v

            ctx = self.attention(q, kv_k[il], kv_v[il], n)
            if il == 0:
                self.dump("05_ctx", ctx)

            x = x + np.stack([self.mat2(b + "attn_output.weight", ctx[t]) for t in range(n)])
            if il == 0:
                self.dump("06_after_attn", x)

            hx = self.rmsnorm(x, b + "ffn_norm.weight")
            if il == 0:
                self.dump("07_ffn_norm", hx)

            # One dequantisation of the three expert tensors per layer: doing
            # it per token would re-read 151 MB per token per layer.
            gate_e = self.flat3(t=self.T[b + "ffn_gate_exps.weight"]).reshape(
                self.n_expert, self.n_ff, H
            )
            up_e = self.flat3(t=self.T[b + "ffn_up_exps.weight"]).reshape(
                self.n_expert, self.n_ff, H
            )
            down_e = self.flat3(t=self.T[b + "ffn_down_exps.weight"]).reshape(
                self.n_expert, H, self.n_ff
            )
            gate_inp = self.flat3(t=self.T[b + "ffn_gate_inp.weight"]).reshape(
                self.n_expert, H
            )

            moe_out = np.zeros((n, H), dtype=np.float32)
            rlog_all = np.zeros((n, self.n_expert), dtype=np.float32)
            for t in range(n):
                logits = (gate_inp @ hx[t]).astype(np.float32)
                rlog_all[t] = logits
                probs = self.softmax(logits)
                top = np.argsort(-probs, kind="stable")[: self.top_k]
                wts = probs[top] / probs[top].sum()
                acc = np.zeros(H, dtype=np.float32)
                for e, wt in zip(top, wts):
                    e = int(e)
                    g = (gate_e[e] @ hx[t]).astype(np.float32)
                    u = (up_e[e] @ hx[t]).astype(np.float32)
                    act = (g / (1.0 + np.exp(-g))) * u
                    acc = acc + np.float32(wt) * (down_e[e] @ act).astype(np.float32)
                moe_out[t] = acc
            del gate_e, up_e, down_e, gate_inp
            if il == 0:
                # Dumped as whole batches, matching the engine's buffer layout:
                # the engine holds all n rows at once, so a per-token dump here
                # would overwrite itself and compare only the last token.
                self.dump("08_router_logits", rlog_all)
                self.dump("09b_moe_out", moe_out)

            x = x + moe_out
            self.dump(f"10_after_layer{il:02d}", x)
            print(f"  layer {il:3d} done  ({time.time() - t0:6.1f}s)", flush=True)

        xn = self.rmsnorm(x[n - 1], "output_norm.weight")
        self.dump("11_final_norm", xn)
        logits = self.mat2("output.weight", xn)
        self.dump("12_logits", logits)
        print(f"ref: forward complete in {time.time() - t0:.1f}s")
        return logits

    def dump(self, name, arr):
        if not self.dump_dir or self._forward_calls != 1:
            return
        path = os.path.join(self.dump_dir, name + ".f32")
        np.asarray(arr, dtype=np.float32).reshape(-1).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", default="785,6722,315,9625,374")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1, help="greedy continuation length")
    args = ap.parse_args()

    tokens = [int(s) for s in args.tokens.split(",") if s.strip()]
    m = Model(args.model, args.dump)
    print(f"ref prompt ids: {tokens}  ({len(tokens)} tokens)")

    logits = m.forward(tokens)
    order = np.argsort(-logits)[: args.top]
    print(f"\nref top-{args.top} logits for the last prompt position:")
    for r, i in enumerate(order):
        print(f"  {r:2d}. id={int(i):7d}  logit={logits[i]:+.6f}")
    print(f"\nref argmax = {int(order[0])}")

    # Greedy continuation, so the whole reference can be eyeballed against
    # llama.cpp's completion rather than only against itself.
    toks = list(tokens)
    out = []
    lg = logits
    for _ in range(args.steps):
        nxt = int(np.argmax(lg))
        toks.append(nxt)
        out.append(nxt)
        if nxt == 151645:
            break
        # Re-run the whole prompt each step. Slow, and deliberately so: this
        # is an oracle, and a KV cache here would be a second implementation
        # of the thing under test.
        lg = m.forward(toks)
    print(f"\nref greedy continuation ids: {out}")


if __name__ == "__main__":
    sys.exit(main())
