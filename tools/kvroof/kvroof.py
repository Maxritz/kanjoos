#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kvroof.py -- geometry resolution and traffic derivations for the KV engine.

Pure functions only. No printing, no exit codes, no filesystem access: the CLI
in `kv_roofline.py` owns those. Kept separate so the routing-locality probe
(`tools/route/route_locality.py`) can reuse the geometry resolver and so the
derivations can be unit-checked without a config on disk.

The one rule this module enforces everywhere: **a number is derived only from
a field that is actually present in the config.** If a field is missing and
cannot be derived from fields that are present, this raises KvError instead of
substituting a default. That is the whole point of the tool.

Provenance of every constant in MEASURED / below is cited to
docs/00-verified-facts.md section 8. Change them there first.
"""

from __future__ import annotations


class KvError(Exception):
    """Raised when a required field is missing or not derivable.

    `exit_code` is what the CLI returns. It is never 0.
    """

    def __init__(self, message, exit_code=2, hints=()):
        super().__init__(message)
        self.exit_code = exit_code
        self.hints = list(hints)


# ---------------------------------------------------------------- machine
# Measured on this host, 2026-10-05. Ranges across four runs, never point
# values: quoting the last digit of a bandwidth run is how this repo caught
# itself once already (00-verified-facts.md section 8.1).
VRAM_GBPS_LO, VRAM_GBPS_HI = 589.0, 598.0        # measured, section 8.1
PCIE_GBPS_LO, PCIE_GBPS_HI = 13.4, 14.7          # measured, section 8.1
                                               # (one 2.47 MB expert copy)
PEAK_TFLOPS = 39.3                              # derived, section 8.2

# Weight-pack constants: the single Python source for the 3 B/group rule and
# the bit-width ladder. Mirrors kernels/knj_pack.h (KNJ_META_BYTES_PER_GROUP,
# KNJ_GROUP_EXPERT, KNJ_W*_BYTES_PER_WEIGHT, knj_bytes_per_weight()).
# Every Python consumer of these numbers (kvroof, kv_roofline, route) imports
# from pack_common instead of restating them, so the 0.5+3/128 headline number
# that drifted to 0.5186 once can only be changed in one place.
from pack_common import (
    META_BYTES_PER_GROUP,
    GROUP_EXPERT,
    GROUP_KV,
    BITS_W8, BITS_W6, BITS_W4, BITS_W3, BITS_W2,
    W4_BYTES_PER_WEIGHT, W3_BYTES_PER_WEIGHT, W2_BYTES_PER_WEIGHT,
    W4G64_BYTES_PER_WEIGHT, W8_BYTES_PER_WEIGHT,
    bytes_per_weight as _bpw,
)

KV_GROUP_DEFAULT = GROUP_KV   # gemm_w4.hip header: "KV keeps group 64 because
                               # KV is read repeatedly and the metadata cost is
                               # amortised over that reuse."

DTYPE_BYTES = {
    "float16": 2, "fp16": 2, "half": 2, "bfloat16": 2, "bf16": 2,
    "float32": 4, "fp32": 4, "float64": 8, "fp64": 8,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "uint8": 1, "int4": 0.5,
}

# name -> (payload bytes per element, group size or None for no metadata)
# Built from the shared constants, not hard-coded, so a codec's metadata is
# always the same 3 B/group the packs use.
CODECS = (
    ("FP16",      2.0, None),
    ("FP8 E4M3",  1.0, None),
    ("INT8 g64",  1.0, GROUP_KV),
    ("INT8 g128", 1.0, GROUP_EXPERT),
    ("INT4 g64",  0.5, GROUP_KV),
    ("INT4 g128", 0.5, GROUP_EXPERT),
)


# name -> (bits, group_size)
# Built from the shared constants.
PACKS = (
    ("W4 g128", BITS_W4, GROUP_EXPERT),
    ("W3 g128", BITS_W3, GROUP_EXPERT),
    ("W2 g128", BITS_W2, GROUP_EXPERT),
    ("W4 g64",  BITS_W4, GROUP_KV),
    ("W8 g128", BITS_W8, GROUP_EXPERT),
)


def pack_bytes_per_weight(name):
    """Bytes per stored weight for a named pack, metadata included."""
    for pname, bits, group in PACKS:
        if pname == name:
            return _bpw(bits, group)
    raise KvError("unknown weight pack %r" % name, exit_code=6)


def resident_weight_bytes_for_pack(geom, pack):
    """Resident set for a given weight pack: the whole story in one number."""
    return resident_weight_bytes(geom, pack_bytes_per_weight(pack))


def codec_bytes_per_elem(name):
    """Bytes per cached element for a codec, metadata included."""
    for cname, payload, group in CODECS:
        if cname == name:
            if group is None:
                return payload
            return payload + META_BYTES_PER_GROUP / group
    raise KvError("unknown codec %r" % name, exit_code=6)


def machine_balance_flop_per_byte(vram_lo=VRAM_GBPS_LO, vram_hi=VRAM_GBPS_HI,
                                  peak_tflops=PEAK_TFLOPS):
    """FLOP/byte the machine needs to be compute-bound. Returns (lo, hi)."""
    if vram_lo <= 0 or vram_hi <= 0 or peak_tflops <= 0:
        raise KvError("non-positive bandwidth/peak argument", exit_code=6)
    peak = peak_tflops * 1e12
    return peak / (vram_hi * 1e9), peak / (vram_lo * 1e9)


# ---------------------------------------------------------------- config
def load_config(path):
    """Read a config.json. Raises KvError(exit 3) rather than returning junk."""
    import json
    import os
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise KvError("cannot read %s: %s" % (path, exc), exit_code=3)
    try:
        cfg = json.loads(raw)
    except ValueError as exc:
        raise KvError("%s is not valid JSON: %s" % (path, exc), exit_code=3)
    if not isinstance(cfg, dict):
        raise KvError("%s is a %s, not a JSON object" % (path, type(cfg)),
                      exit_code=3)
    return cfg, os.path.abspath(path)


def text_config(cfg, path="<root>"):
    """Return (text_cfg, path) descending into a multimodal nest if present.

    gemma4-* and similar nest the language model under `text_config`. Reading
    the root would silently give `num_attention_heads` of the wrong model.
    """
    for key in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict):
            inner, inner_path = text_config(sub, path + "." + key)
            merged = dict(cfg)
            merged.update(inner)
            return merged, inner_path
    return cfg, path


def _req(cfg, key, path, hints=()):
    val = cfg.get(key)
    if val is None:
        raise KvError("required field %s is absent from %s" % (key, path),
                      exit_code=2, hints=hints)
    return val


def _as_int(val, key):
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise KvError("field %s is %r, expected a number" % (key, val),
                      exit_code=2)
    return int(val)


def resolve_geometry(cfg, path="<root>"):
    """Pull every dimension the engine needs out of the config.

    Raises KvError(exit 2) for anything missing that cannot be derived.
    Returns a dict tagged `foreign` with the reason when the architecture is
    one this tool cannot compute exactly (MLA, hybrid linear/full attention).
    """
    tc, tpath = text_config(cfg, path)
    g = {}
    g["config_path"] = tpath
    g["model_type"] = cfg.get("model_type") or tc.get("model_type") or "?"
    g["architectures"] = cfg.get("architectures") or ["?"]

    g["dtype"] = (cfg.get("torch_dtype") or cfg.get("dtype")
                  or tc.get("torch_dtype") or tc.get("dtype"))
    if g["dtype"] is None:
        raise KvError("neither torch_dtype nor dtype present in %s" % tpath,
                      exit_code=2,
                      hints=["the weight budget in bytes/token needs it"])
    g["dtype_bytes"] = DTYPE_BYTES.get(str(g["dtype"]).lower())
    if g["dtype_bytes"] is None:
        raise KvError("unrecognised dtype %r in %s" % (g["dtype"], tpath),
                      exit_code=2)

    g["hidden_size"] = _as_int(_req(tc, "hidden_size", tpath), "hidden_size")
    g["num_layers"] = _as_int(_req(tc, "num_hidden_layers", tpath),
                              "num_hidden_layers")
    g["num_heads"] = _as_int(_req(tc, "num_attention_heads", tpath),
                             "num_attention_heads")
    g["vocab_size"] = _as_int(_req(tc, "vocab_size", tpath), "vocab_size")
    g["max_position_embeddings"] = _as_int(
        tc.get("max_position_embeddings", 0), "max_position_embeddings")

    # --- KV head geometry -------------------------------------------------
    # `head_dim` is optional: absent means hidden_size / num_attention_heads.
    # That derivation is arithmetic on present fields, not an assumption, so it
    # is allowed -- but it is recorded as `head_dim_source` so nobody can read a
    # number without seeing where the geometry came from.
    if tc.get("head_dim") is not None:
        g["head_dim"] = _as_int(tc["head_dim"], "head_dim")
        g["head_dim_source"] = "head_dim"
    elif tc.get("qk_head_dim") is not None:
        g["head_dim"] = _as_int(tc["qk_head_dim"], "qk_head_dim")
        g["head_dim_source"] = "qk_head_dim (partial-rotary head)"
    else:
        if g["hidden_size"] % g["num_heads"]:
            raise KvError("no head_dim and hidden_size %d is not divisible by "
                          "num_attention_heads %d in %s"
                          % (g["hidden_size"], g["num_heads"], tpath),
                          exit_code=2)
        g["head_dim"] = g["hidden_size"] // g["num_heads"]
        g["head_dim_source"] = "hidden_size / num_attention_heads"

    kv_key = None
    for k in ("num_key_value_heads", "num_kv_heads",
              "num_global_key_value_heads"):
        if tc.get(k) is not None:
            kv_key = k
            break
    if kv_key is None:
        # MHA is a legitimate architecture, not a missing field.
        g["num_kv_heads"] = g["num_heads"]
        g["kv_heads_source"] = "num_attention_heads (MHA: no GQA field)"
    else:
        g["num_kv_heads"] = _as_int(tc[kv_key], kv_key)
        g["kv_heads_source"] = kv_key
    g["gqa_ratio"] = g["num_heads"] // max(g["num_kv_heads"], 1)

    # --- windowing / layer kinds -----------------------------------------
    g["sliding_window"] = tc.get("sliding_window")
    g["layer_types"] = tc.get("layer_types")
    g["kv_shared_layers"] = _as_int(tc.get("num_kv_shared_layers", 0),
                                    "num_kv_shared_layers")
    g["full_attention_layers"] = g["num_layers"]
    g["windowed_attention_layers"] = 0
    g["window_limit_tokens"] = 0
    if isinstance(g["layer_types"], list) and g["layer_types"]:
        full = sum(1 for t in g["layer_types"]
                   if t == "full_attention" or t is None)
        win = len(g["layer_types"]) - full
        g["full_attention_layers"] = full
        g["windowed_attention_layers"] = win
        sw = tc.get("sliding_window")
        sw = _as_int(sw, "sliding_window") if sw else 0
        g["window_limit_tokens"] = sw

    # --- MoE --------------------------------------------------------------
    g["is_moe"] = tc.get("num_experts") is not None
    g["num_experts"] = 0
    g["top_k"] = 0
    g["moe_intermediate_size"] = 0
    g["shared_expert_intermediate_size"] = 0
    g["dense_mlp_layers"] = 0
    if g["is_moe"]:
        g["num_experts"] = _as_int(_req(tc, "num_experts", tpath),
                                   "num_experts")
        g["top_k"] = _as_int(_req(tc, "num_experts_per_tok", tpath),
                             "num_experts_per_tok")
        if tc.get("top_k_experts") is not None:
            g["top_k_field"] = "top_k_experts"
        g["moe_intermediate_size"] = _as_int(
            _req(tc, "moe_intermediate_size", tpath), "moe_intermediate_size")
        for k in ("moe_shared_expert_intermediate_size",
                  "shared_expert_intermediate_size"):
            if tc.get(k):
                g["shared_expert_intermediate_size"] = _as_int(tc[k], k)
                break
        fk = tc.get("first_k_dense_replace")
        if fk is not None:
            g["dense_mlp_layers"] = _as_int(fk, "first_k_dense_replace")
        elif isinstance(tc.get("mlp_only_layers"), list):
            g["dense_mlp_layers"] = len(tc["mlp_only_layers"])
        g["moe_layers"] = g["num_layers"] - g["dense_mlp_layers"]
    else:
        g["moe_intermediate_size"] = _as_int(tc.get("intermediate_size", 0),
                                             "intermediate_size")
        g["moe_layers"] = 0

    # --- foreign architectures -------------------------------------------
    foreign = []
    if tc.get("kv_lora_rank") is not None or tc.get("q_lora_rank") is not None:
        foreign.append("MLA: kv_lora_rank/q_lora_rank present, the KV cache is "
                       "a compressed latent, not K and V per head")
    if tc.get("num_kv_heads_for_linear_attn") is not None:
        foreign.append("hybrid linear (KDA) + full attention layers")
    if tc.get("layer_group_size") is not None:
        foreign.append("layer_group_size: KV is shared across a layer group")
    if tc.get("kda_lower_bound") is not None:
        foreign.append("KDA gate present: not all layers hold a K/V cache")
    if tc.get("use_mla_nope"):
        foreign.append("use_mla_nope: NoPE/RoPE split not modelled here")
    if tc.get("num_global_key_value_heads") is not None:
        foreign.append("num_global_key_value_heads: sliding layers use a "
                       "different KV head count than global layers")
    g["foreign"] = foreign
    g["tie_word_embeddings"] = bool(tc.get("tie_word_embeddings", False))
    g["exact"] = not foreign
    return g


# ---------------------------------------------------------------- traffic
def kv_bytes_per_token(geom, codec, tokens=None):
    """KV bytes read per generated token at `tokens` context.

    tokens=None means "full context, no window clipping" -- the number the
    crossover uses, because the crossover must not depend on a window that is
    itself derived from context.
    """
    bpe = codec_bytes_per_elem(codec)
    layers = geom["num_layers"]
    if geom["kv_shared_layers"]:
        # cross-layer KV sharing: the first N layers share one cache
        layers = 1 + (geom["num_layers"] - geom["kv_shared_layers"])
    per_layer_elems = geom["num_kv_heads"] * geom["head_dim"] * 2  # K and V
    full_tokens = 1 if tokens is None else tokens
    win_tokens = full_tokens
    if tokens is not None and geom["window_limit_tokens"]:
        win_tokens = min(tokens, geom["window_limit_tokens"])
    n_full = geom["full_attention_layers"]
    n_win = geom["windowed_attention_layers"]
    if geom["kv_shared_layers"] and n_full and n_win:
        # shared cache covers the windowed group; full layers keep their own
        shared_full = max(layers - n_full, 0)
        n_full = max(n_full - shared_full, 0)
    elems = n_full * full_tokens + n_win * win_tokens
    return elems * per_layer_elems * bpe


def expert_bytes_per_layer(geom, bytes_per_weight=W4_BYTES_PER_WEIGHT):
    """gate + up + down for one expert."""
    return 3 * geom["moe_intermediate_size"] * geom["hidden_size"] * bytes_per_weight


def shared_expert_bytes_per_layer(geom, bytes_per_weight=W4_BYTES_PER_WEIGHT):
    if not geom["shared_expert_intermediate_size"]:
        return 0.0
    return 3 * geom["shared_expert_intermediate_size"] * geom["hidden_size"] * bytes_per_weight


def attention_bytes_per_layer(geom, bytes_per_weight=W4_BYTES_PER_WEIGHT):
    """q, k, v, o projections, read in full every generated token."""
    h, hd, kvh = geom["hidden_size"], geom["head_dim"], geom["num_kv_heads"]
    nh = geom["num_heads"]
    return (h * nh * hd + 2 * (h * kvh * hd) + nh * hd * h) * bytes_per_weight


def weight_bytes_per_token(geom, bytes_per_weight=W4_BYTES_PER_WEIGHT):
    """Per-token weight traffic, broken out. Every row is bytes, not params.

    `routed_experts` is the number the docs have been quoting. `total` is what
    the decode step actually touches, and it is larger -- see lm_head.
    """
    w = {"bytes_per_weight": bytes_per_weight}
    w["routed_experts"] = (geom["top_k"] * expert_bytes_per_layer(geom,
                                                                  bytes_per_weight)
                           * geom["moe_layers"])
    w["shared_expert"] = (shared_expert_bytes_per_layer(geom, bytes_per_weight)
                          * geom["moe_layers"])
    w["attention"] = attention_bytes_per_layer(geom, bytes_per_weight) * geom["num_layers"]
    lm = 0.0
    if not geom.get("tie_word_embeddings"):
        lm = geom["vocab_size"] * geom["hidden_size"] * bytes_per_weight
    w["lm_head"] = lm
    w["expert_only"] = w["routed_experts"] + w["shared_expert"]
    w["total"] = w["expert_only"] + w["attention"] + w["lm_head"]
    return w


def resident_weight_bytes(geom, bytes_per_weight=W4_BYTES_PER_WEIGHT):
    """What must sit in VRAM to serve any token at all."""
    r = {"bytes_per_weight": bytes_per_weight}
    r["expert_bank"] = (geom["num_experts"] * expert_bytes_per_layer(geom,
                                                                    bytes_per_weight)
                        * geom["moe_layers"])
    r["shared_expert"] = (shared_expert_bytes_per_layer(geom, bytes_per_weight)
                          * geom["moe_layers"])
    r["attention"] = attention_bytes_per_layer(geom, bytes_per_weight) * geom["num_layers"]
    r["embeddings"] = geom["vocab_size"] * geom["hidden_size"] * geom["dtype_bytes"]
    lm = 0.0
    if not geom.get("tie_word_embeddings"):
        lm = geom["vocab_size"] * geom["hidden_size"] * bytes_per_weight
    r["lm_head"] = lm
    r["total"] = (r["expert_bank"] + r["shared_expert"] + r["attention"]
                  + r["embeddings"] + r["lm_head"])
    return r


def attention_flops_per_step(geom, tokens):
    """Decode attention FLOPs for one token at `tokens` of context.

    QK^T and PV both cost 2 * tokens * num_heads * head_dim per layer. Note
    that num_heads appears, not num_kv_heads: GQA broadcasts each KV head.
    """
    return (4.0 * tokens * geom["num_heads"] * geom["head_dim"]
            * geom["num_layers"])


def attention_intensity_flop_per_byte(geom, codec):
    """FLOP per KV byte at M=1, as a function of codec only.

    Cancels the context length entirely: 2 * gqa_ratio / bytes_per_elem.
    """
    bpe = codec_bytes_per_elem(codec)
    return 2.0 * geom["gqa_ratio"] / bpe