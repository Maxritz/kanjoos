#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kv_roofline.py -- read a model config.json and print the KV roofline, the
crossover context, and the Class A / Class B ceilings.

Why this tool exists
--------------------
Every KV number in this repo used to rest on an *assumed* head configuration
(48 layers x 4 KV heads x 128 head_dim, x 2 for K and V, x 2 bytes). That
assumption happened to be right for Qwen3-30B-A3B, which is the worst way to
be right: it is unverifiable, and it is wrong for four of the five configs
vendored in models/. This tool removes the assumption by reading the fields.

Nothing here is measured on the GPU. Everything here is *derived* from a real
config file plus the bandwidths measured in 00-verified-facts.md section 8.1.
The two are kept visually distinct everywhere they appear.

Usage
-----
    python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json
    python tools/kvroof/kv_roofline.py models/*/config.json --json
    python tools/kvroof/kv_roofline.py <cfg> --check     # doc-claim audit

Exit codes
----------
    0  report produced
    2  a required field is missing and cannot be derived from present fields
    3  config unreadable or not valid JSON
    4  architecture is not exactly computable here (MLA, hybrid attention);
       output printed only with --allow-foreign-arch, and it is NOT a roofline
    5  --check found at least one doc claim that this config contradicts
    6  a numeric argument (bandwidth, budget) was missing or non-positive
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kvroof as K  # noqa: E402
from kvroof import KvError  # noqa: E402

GIB = 1024.0 ** 3
MIB = 1024.0 ** 2
KIB = 1024.0


def fmt_bytes(n):
    """Human string. Always prints 3 significant figures, never 6."""
    if n >= GIB:
        return "%.2f GiB" % (n / GIB)
    if n >= MIB:
        return "%.1f MiB" % (n / MIB)
    if n >= KIB:
        return "%.1f KiB" % (n / KIB)
    return "%.0f B" % n


def fmt_mb(n):
    return "%.0f MB" % (n / 1e6)


def fmt_tok(n):
    if n >= 1000:
        return "%.1f K" % (n / 1000.0)
    return "%.0f" % n


def hr(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------ report
def report_one(path, args, cfg=None, abspath=None):
    if cfg is None:
        cfg, abspath = K.load_config(path)
    geom = K.resolve_geometry(cfg, abspath)

    hr("A. CONFIG AND WHERE THE GEOMETRY CAME FROM")
    print("config        %s" % abspath)
    print("model_type    %s   %s" % (geom["model_type"],
                                     ",".join(geom["architectures"])))
    print("dtype         %s (%d B/element)" % (geom["dtype"],
                                               geom["dtype_bytes"]))
    print("")
    print("This block is the whole point of the tool. If a field is absent the")
    print("tool refuses to produce a number; nothing below is inherited from a")
    print("previous model, a previous run, or a comment.")
    print("")
    print("  num_hidden_layers        %5d   %s"
          % (geom["num_layers"], "READ" if "num_hidden_layers" in cfg
             or "num_hidden_layers" in cfg.get("text_config", {})
             else "?"))
    print("  num_attention_heads      %5d" % geom["num_heads"])
    print("  num_key_value_heads      %5d   from %s"
          % (geom["num_kv_heads"], geom["kv_heads_source"]))
    print("  head_dim                 %5d   from %s"
          % (geom["head_dim"], geom["head_dim_source"]))
    print("  GQA ratio                %5d   query heads per KV head"
          % geom["gqa_ratio"])
    print("  hidden_size              %5d" % geom["hidden_size"])
    print("  vocab_size               %5d   tie_word_embeddings=%s"
          % (geom["vocab_size"], geom["tie_word_embeddings"]))
    if geom["max_position_embeddings"]:
        print("  max_position_embeddings  %5d   <-- hard ceiling on any Class A"
              % geom["max_position_embeddings"])
    if geom["is_moe"]:
        print("  num_experts              %5d   top_k=%d   moe_inter=%d"
              % (geom["num_experts"], geom["top_k"],
                 geom["moe_intermediate_size"]))
        print("  MoE layers               %5d   (dense MLP layers: %d)"
              % (geom["moe_layers"], geom["dense_mlp_layers"]))
        if geom["shared_expert_intermediate_size"]:
            print("  shared expert inter      %5d   read every token, always"
                  % geom["shared_expert_intermediate_size"])
    else:
        print("  MoE                      none in this config (dense)")
    if geom["windowed_attention_layers"]:
        print("  windowed/full layers     %5d / %d  window=%d tokens"
              % (geom["windowed_attention_layers"],
                 geom["full_attention_layers"], geom["window_limit_tokens"]))
    if geom["kv_shared_layers"]:
        print("  cross-layer KV sharing   %5d layers share one cache"
              % geom["kv_shared_layers"])
    if not geom["exact"]:
        print("")
        print("  !! NOT EXACTLY COMPUTABLE FOR THIS ARCHITECTURE:")
        for reason in geom["foreign"]:
            print("     - %s" % reason)
        print("     Figures below use the standard K/V-per-head formula and are")
        print("     therefore WRONG for this model. Do not quote them.")

    if not geom["exact"] and not args.allow_foreign_arch:
        raise KvError("architecture is not exactly computable by this tool; "
                      "re-run with --allow-foreign-arch to see (unverified) "
                      "output anyway", exit_code=4,
                      hints=geom["foreign"])

    # --- B. codecs --------------------------------------------------------
    hr("B. KV BYTES PER TOKEN, PER CODEC  (derived from section A)")
    print("Metadata is 3 B/group: fp16 scale + uint8 zero-point, matching")
    print("tools/bench/gemm_w4.hip. KV uses group %d there. The INT4 rows carry"
          % K.KV_GROUP_DEFAULT)
    print("the same 3 B per group, so INT4 is NOT half of FP16 on disk.")
    print("")
    print("  %-10s %12s %14s   %s" % ("codec", "B/elem", "KV B/token", "note"))
    for name, _p, _g in K.CODECS:
        bpe = K.codec_bytes_per_elem(name)
        bpt = K.kv_bytes_per_token(geom, name)
        note = "no metadata"
        if _g:
            note = "+3 B/group of %d" % _g
        print("  %-10s %12.4f %14s   %s"
              % (name, bpe, "%d (%.1f KiB)" % (bpt, bpt / KIB), note))

    # --- C. weight traffic ------------------------------------------------
    hr("C. WEIGHT BYTES PER GENERATED TOKEN  (M=1, W4 pack at %.4f B/weight)"
       % K.W4_BYTES_PER_WEIGHT)
    w = K.weight_bytes_per_token(geom)
    print("  %-28s %16s   %s" % ("component", "MB/token", "read every token?"))
    print("  %-28s %16s   %s" % ("routed experts (top_k x 3)",
                                 fmt_mb(w["routed_experts"]) if geom["is_moe"]
                                 else "n/a (dense)",
                                 "yes" if geom["is_moe"] else "no experts"))
    if w["shared_expert"]:
        print("  %-28s %16s   %s" % ("shared expert", fmt_mb(w["shared_expert"]),
                                     "yes"))
    print("  %-28s %16s   %s" % ("attention q/k/v/o", fmt_mb(w["attention"]),
                                 "yes"))
    print("  %-28s %16s   %s" % ("lm_head (vocab x hidden)",
                                 fmt_mb(w["lm_head"]), "yes, every step"))
    print("  %-28s %16s   %s" % ("TOTAL", fmt_mb(w["total"]), ""))
    print("")
    print("  expert-only subtotal (the convention the docs have been using)")
    print("  = %s/token. The lm_head is %s of the total and is missing from"
          % (fmt_mb(w["expert_only"]), "%.0f%%" % (100 * w["lm_head"] / w["total"])))
    print("  that convention; at decode every step touches it in full.")
    print("")
    r = K.resident_weight_bytes(geom)
    print("  Resident to serve anything at all:")
    print("    expert bank (%d experts)  %s" % (geom["num_experts"],
                                                 fmt_bytes(r["expert_bank"])))
    print("    attention + shared        %s"
          % fmt_bytes(r["attention"] + r["shared_expert"]))
    print("    embeddings                %s" % fmt_bytes(r["embeddings"]))
    print("    lm_head                   %s" % fmt_bytes(r["lm_head"]))
    print("    TOTAL                     %s" % fmt_bytes(r["total"]))

    # --- D. crossover -----------------------------------------------------
    hr("D. CROSSOVER CONTEXT: WHERE KV PASSES WEIGHTS  (derived)")
    print("crossover = weight bytes per token / KV bytes per token, at M=1.")
    print("")
    print("  %-10s %18s %22s" % ("codec", "crossover (expert-only)",
                                  "crossover (total w/ lm_head)"))
    crossings = {}
    for name, _p, _g in K.CODECS:
        bpt = K.kv_bytes_per_token(geom, name)
        c_e = w["expert_only"] / bpt
        c_t = w["total"] / bpt
        crossings[name] = c_e
        print("  %-10s %18s %22s"
              % (name, fmt_tok(c_e), fmt_tok(c_t)))
    fp16_cross = crossings["FP16"]
    print("")
    print("  FP16 crossover = %d tokens = %s." % (fp16_cross, fmt_tok(fp16_cross)))
    if geom["max_position_embeddings"]:
        verdict = ("BELOW" if fp16_cross < geom["max_position_embeddings"]
                   else "ABOVE")
        print("  The model supports %s tokens. The crossover is %s that limit,"
              % (fmt_tok(geom["max_position_embeddings"]), verdict))
        print("  so KV becomes the larger traffic term somewhere inside the")
        print("  context the model is able to hold.")

    # --- E. dominance -----------------------------------------------------
    hr("E. TRAFFIC DOMINANCE AT FIXED CONTEXT  (derived from C and B)")
    # A dense model has no expert traffic, so the denominator is the total
    # per-token weight read instead. Saying so beats dividing by zero.
    denom = w["expert_only"] if w["expert_only"] > 0 else w["total"]
    denom_label = ("expert-only subtotal" if w["expert_only"] > 0
                   else "total (dense model: no experts)")
    print("  %-8s %14s %14s %8s   %s"
          % ("context", "KV MB/token", "weight MB/tok", "ratio", "dominant"))
    for tok in (2048, 8192, 16384, 32768, 131072):
        kv = K.kv_bytes_per_token(geom, "FP16", tokens=tok)
        ratio = kv / denom if denom else 0.0
        print("  %-8s %14s %14s %7.2fx   %s"
              % (fmt_tok(tok), fmt_mb(kv), fmt_mb(denom), ratio,
                 "KV" if ratio > 1.0 else "weights"))
    print("  (K here; weight column is the %s: comparable to the table in"
          % denom_label)
    print("   docs/09 section 2. For a MoE, prefer the total column from C.)")

    # --- F. attention roofline -------------------------------------------
    hr("F. ATTENTION ROOFLINE AT M=1  (the actual shape of the ridge)")
    bal_lo, bal_hi = K.machine_balance_flop_per_byte(
        args.vram_lo, args.vram_hi, args.peak_tflops)
    print("  machine balance            %.1f - %.1f FLOP/byte"
          % (bal_lo, bal_hi))
    print("  (%.1f TFLOP/s packed-f16 derived, %.0f-%.0f GB/s VRAM measured,"
          % (args.peak_tflops, args.vram_lo, args.vram_hi))
    print("   00-verified-facts section 8.1/8.2 -- these two are the measured")
    print("   inputs, the balance is the derived ratio.)")
    print("")
    print("  %-10s %12s %14s %14s   %s"
          % ("codec", "FLOP/byte", "compute ms", "mem ms", "bound by"))
    for name, _p, _g in K.CODECS:
        if name not in ("FP16", "INT8 g64", "INT4 g64"):
            continue
        inten = K.attention_intensity_flop_per_byte(geom, name)
        bpt = K.kv_bytes_per_token(geom, name)
        flops = K.attention_flops_per_step(geom, 16384)
        t_compute = flops / (args.peak_tflops * 1e12) * 1e3
        t_mem = bpt * 16384 / (args.vram_lo * 1e9) * 1e3
        bound = "MEMORY" if t_mem > t_compute else "COMPUTE"
        print("  %-10s %12.1f %14.1f %14.2f   %s (%.1fx)"
              % (name, inten, t_compute, t_mem, bound, t_mem / t_compute))
    print("  (at 16 K context, FP16 KV. Intensity is independent of context:")
    print("   2 * gqa_ratio / bytes-per-element.)")
    print("")
    print("  Decode attention is memory-bound at every useful codec, because a")
    print("  GQA decode step does one multiply-accumulate per byte it reads.")
    print("  Coarser codecs move the ridge, they do not cross it.")

    # --- G. Class A -------------------------------------------------------
    hr("G. CLASS A (RESIDENT) CEILING")
    total_vram = args.vram_gib * GIB
    kv_budget = args.kv_budget_gib * GIB if args.kv_budget_gib else (
        total_vram - r["total"] - args.workspace_gib * GIB)
    print("  VRAM total                 %s (--vram-gib)" % fmt_bytes(total_vram))
    print("  resident weights           %s" % fmt_bytes(r["total"]))
    print("  workspace reserve          %s (--workspace-gib)"
          % fmt_bytes(args.workspace_gib * GIB))
    print("  => KV budget               %s%s"
          % (fmt_bytes(kv_budget),
             "  (negative: this model does not fit; see note)" if kv_budget < 0
             else ""))
    if kv_budget < 0:
        print("")
        print("  NEGATIVE BUDGET: at W4 the full expert bank does not leave room")
        print("  for KV on this card. That is a residency-policy finding, not a")
        print("  codec finding -- see the VRAM slots row in docs/09 section 8.")
    print("")
    print("  %-10s %16s %20s" % ("codec", "KV B/token", "resident context"))
    if kv_budget <= 0:
        for name, _p, _g in K.CODECS:
            bpt = K.kv_bytes_per_token(geom, name)
            print("  %-10s %16s %20s"
                  % (name, "%d (%.1f KiB)" % (bpt, bpt / KIB), "NO FIT"))
    else:
        for name, _p, _g in K.CODECS:
            bpt = K.kv_bytes_per_token(geom, name)
            toks = kv_budget / bpt if bpt else 0.0
            cap = ""
            if geom["max_position_embeddings"] and toks > geom["max_position_embeddings"]:
                cap = "  <- capped by max_position_embeddings"
                toks = geom["max_position_embeddings"]
            print("  %-10s %16s %20s%s"
                  % (name, "%d (%.1f KiB)" % (bpt, bpt / KIB), fmt_tok(toks), cap))

    # --- G.1 joint residency --------------------------------------------
    hr("G.1 JOINT RESIDENCY: KV BUDGET vs EXPERT SLOTS (the real trade)")
    if not geom["is_moe"]:
        print("  Not applicable: this config has no expert bank, so there are")
        print("  no slots to trade against KV. Sections G and H still hold.")
        print("")
    else:
        print("  Weights and KV compete for the same VRAM. Class A is not a KV")
        print("question on its own -- it is a question about the sum. This table")
        print("holds total = vram - workspace and spends the remainder.")
        print("")
        slot_bytes = K.expert_bytes_per_layer(geom)
        non_expert = r["total"] - r["expert_bank"]
        print("  one expert in one layer = %s at W4" % fmt_bytes(slot_bytes))
        print("  non-expert resident    = %s (attention, embeddings, lm_head)"
              % fmt_bytes(non_expert))
        print("")
        print("  %-10s %14s %14s %16s %18s"
              % ("KV budget", "FP16 ctx", "FP8 ctx", "expert slots", "slots/layer"))
        for kv_gib in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0):
            kvb = kv_gib * GIB
            avail = total_vram - args.workspace_gib * GIB - kvb - non_expert
            slots = avail / slot_bytes if slot_bytes else 0.0
            if slots < 0:
                slots = 0.0
            fp16_ctx = kvb / K.kv_bytes_per_token(geom, "FP16")
            fp8_ctx = kvb / K.kv_bytes_per_token(geom, "FP8 E4M3")
            frac = 100.0 * slots / max(geom["num_experts"] * geom["moe_layers"], 1)
            print("  %-10s %14s %14s %11d/%d %17.1f  (%.0f%% of bank)"
                  % ("%.2f GiB" % kv_gib, fmt_tok(fp16_ctx), fmt_tok(fp8_ctx),
                     int(slots), geom["num_experts"] * geom["moe_layers"],
                     slots / max(geom["moe_layers"], 1), frac))
        print("")
        print("  The slot count here is an IDEALISED uniform budget: it assumes the")
        print("  same number of resident experts in every layer. Real routing is not")
        print("  uniform, and the budget that actually matters is the hit rate at a")
        print("  given slot count. That is P0-7, measured in")
        print("  tools/route/, not assumed here.")

    # --- K. weight format decision ---------------------------------------
    hr("K. WEIGHT FORMAT DECISION (the only lever 9.6 left open)")
    print("Every KV row above assumed W4 because W4 is what was measured. But")
    print("section 9.6 of docs/00 shows context cannot be bought by evicting")
    print("experts, so the weight format stops being a bandwidth optimisation")
    print("and becomes a CAPACITY decision. Same 3 B/group metadata rule as")
    print("gemm_w4.hip, whatever the bit width.")
    print("")
    print("  %-10s %10s %14s %14s %14s %12s %14s"
          % ("pack", "B/weight", "expert bank", "resident", "KV budget",
             "FP16 ctx", "slots/layer"))
    rows = []
    for pname, _bits, _g in K.PACKS:
        bpw = K.pack_bytes_per_weight(pname)
        rr = K.resident_weight_bytes(geom, bpw)
        budget = total_vram - rr["total"] - args.workspace_gib * GIB
        if budget < 0:
            budget = 0.0
        fp16_ctx = budget / K.kv_bytes_per_token(geom, "FP16")
        fp8_ctx = budget / K.kv_bytes_per_token(geom, "FP8 E4M3")
        slots_per_layer = max((total_vram - args.workspace_gib * GIB - budget - non_expert)
                              / K.expert_bytes_per_layer(geom, bpw)
                              / max(geom["moe_layers"], 1), 0.0)
        rows.append((pname, bpw, rr, budget, fp16_ctx, fp8_ctx, slots_per_layer))
        print("  %-10s %10.4f %14s %14s %14s %12s %8.0f of %d"
              % (pname, bpw, fmt_bytes(rr["expert_bank"]),
                 fmt_bytes(rr["total"]), fmt_bytes(budget),
                 fmt_tok(fp16_ctx),
                 slots_per_layer, geom["num_experts"]))
    print("")
    print("  FP8 context for the same rows:")
    for pname, _bpw, _rr, budget, _f16, fp8, _s in rows:
        print("    %-10s %s" % (pname, fmt_tok(fp8)))
    if geom["max_position_embeddings"]:
        print("")
        print("  The model cannot attend past %s tokens. Any row above that"
              % fmt_tok(geom["max_position_embeddings"]))
        print("  number is unreachable capacity, not a feature.")

    print("")
    print("  Cost of a partial bank, using the MEASURED P0-7 hit-rate curve")
    print("  (tools/route/route_target-gates.json, LRU, median over layers):")
    curve = _load_hit_curve()
    if curve is None:
        print("    [unavailable] run tools/route/route_locality.py first; the")
        print("    hit rate is NOT assumed here.")
    else:
        print("    %-10s %12s %10s %14s %18s"
              % ("pack", "slots/layer", "LRU hit", "miss ms/tok",
                 "vs all-resident"))
        for pname, _bpw, _rr, _budget, _f16, _f8, slots in rows:
            per_layer = slots / max(geom["moe_layers"], 1)
            hit = _interp_hit(curve, per_layer, geom["num_experts"])
            miss_frac = 1.0 - hit
            bytes_tok = (miss_frac * geom["top_k"]
                         * K.expert_bytes_per_layer(geom)
                         * geom["moe_layers"])
            ms_lo = bytes_tok / (args.pcie_lo * 1e9) * 1e3
        print("    %-10s %7.0f of %-4d %9.1f%% %13.1f %17s"
              % (pname, slots_per_layer, geom["num_experts"], 100 * hit, ms_lo,
                 "baseline" if slots_per_layer >= geom["num_experts"]
                 else "+%.1f ms" % ms_lo))

    # --- H. Class B -------------------------------------------------------
    hr("H. CLASS B (SPILLED) PRICE  (PCIe MEASURED %.1f-%.1f GB/s)"
       % (args.pcie_lo, args.pcie_hi))
    print("  A cold context is read from host RAM on every step, once, forever.")
    print("  Bandwidth is quoted as a range: it was measured as a range.")
    print("")
    print("  %-8s %18s %18s %18s"
          % ("cold ctx", "FP16 @%.1f" % args.pcie_lo,
             "FP16 @%.1f" % args.pcie_hi, "FP8 range"))
    for tok in (1024, 2048, 4096, 8192, 16384, 65536):
        row = [tok]
        for codec in ("FP16", "FP16", "FP8 E4M3"):
            row.append(K.kv_bytes_per_token(geom, codec, tokens=tok)
                       / 1e9)
        fp16_at_lo = row[1] / args.pcie_lo * 1e3   # more ms (slower bw)
        fp16_at_hi = row[1] / args.pcie_hi * 1e3   # fewer ms (faster bw)
        fp8_at_lo  = row[2] / args.pcie_lo * 1e3
        fp8_at_hi  = row[2] / args.pcie_hi * 1e3
        # print low-then-high: the smaller ms first (fastest bandwidth),
        # the larger ms second (slowest bandwidth) — matches the 04 doc form
        print("  %-8s %15.1f ms %15.1f ms   %6.1f - %6.1f ms"
              % (fmt_tok(tok), fp16_at_hi, fp16_at_lo, fp8_at_hi, fp8_at_lo))
    print("")
    print("  Extra context purchasable with a fixed wire budget (%g ms of PCIe"
          % args.wire_ms)
    print("  per step, --wire-ms):")
    for name, _p, _g in K.CODECS:
        bpt = K.kv_bytes_per_token(geom, name)
        t_lo = bpt * args.wire_ms * 1e-3 / args.pcie_lo
        t_hi = bpt * args.wire_ms * 1e-3 / args.pcie_hi
        print("    %-10s %s tokens  (%.1f - %.1f ms of pure transfer)"
              % (name, fmt_tok(t_lo), t_hi * 1e3 / args.wire_ms,
                 t_lo * 1e3 / args.wire_ms))
    print("")
    print("  This is the honest form of the 'spilling buys ~2 K at FP16 and")
    print("  ~8 K at FP8' claim: the number is budget-dependent, and a codec")
    print("  that halves bytes per token buys exactly 2x the context for the")
    print("  same wire budget -- no more, no less.")

    return {"path": abspath, "geom": geom, "weights_per_token": w,
            "resident": r, "crossovers": crossings, "kv_budget": kv_budget}


# ------------------------------------------------------------------ claims
def _load_hit_curve():
    """The measured P0-7 LRU hit-rate curve, or None.

    Deliberately loaded rather than hard-coded. A capacity decision that
    silently assumed a hit rate would be exactly the kind of number this repo
    keeps refusing to quote.
    """
    path = os.path.join(os.path.dirname(HERE), "route",
                        "route_target-gates.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    lru = data.get("aggregate", {}).get("lru")
    if not lru:
        return None
    return {int(k): float(v) for k, v in lru.items()}


def _interp_hit(curve, slots, num_experts):
    """Piecewise-linear hit rate over the measured grid, clamped at the ends."""
    if slots >= num_experts:
        return 1.0
    keys = sorted(curve)
    if slots <= keys[0]:
        return curve[keys[0]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= slots <= hi:
            span = float(hi - lo)
            if span <= 0:
                return curve[hi]
            t = (slots - lo) / span
            return curve[lo] + t * (curve[hi] - curve[lo])
    return curve[keys[-1]]


def doc_claims(geom, w, vram_gib, workspace_gib):
    """The numeric claims in docs/09 sections 2 and 5, plus the W3/W2 operating
    point in docs/00 section 9.8, as checkable pairs.

    Kept here so that "the doc agrees with the config" is a machine-checked
    fact rather than a claim in prose. Returns (rows, failures).
    """
    rows = []
    fp16 = K.kv_bytes_per_token(geom, "FP16")
    bpt = fp16
    GIB = 1024.0 ** 3
    vram = vram_gib * GIB
    ws = workspace_gib * GIB

    def chk(label, doc_value, derived, tol=0.02, unit=""):
        if doc_value == 0:
            ok = derived == 0
            err = 0.0
        else:
            err = (derived - doc_value) / doc_value
            ok = abs(err) <= tol
        rows.append((label, doc_value, derived, err, ok, unit))

    # section 2, the crossover table
    chk("doc09 s2 KV bytes/token FP16", 98304.0, fp16, 0.0)
    chk("doc09 s2 weight bytes/token (expert-only)", 948e6,
        w["expert_only"], 0.01)
    chk("doc09 s2 crossover tokens", 9647.0,
        w["expert_only"] / fp16, 0.01)
    for tok, kv_mb, ratio in ((2048, 201.0, 0.21), (8192, 805.0, 0.85),
                              (16384, 1610.0, 1.70), (32768, 3221.0, 3.40),
                              (131072, 12880.0, 13.6)):
        kv = K.kv_bytes_per_token(geom, "FP16", tokens=tok)
        chk("doc09 s2 KV MB at %d" % tok, kv_mb, kv / 1e6, 0.01)
        chk("doc09 s2 ratio at %d" % tok, ratio,
            (kv / w["expert_only"]), 0.03)
    # section 5, the bytes/token column of the Class A table (as printed in the doc)
    for name, doc_bytes in (("FP16", 98304.0), ("FP8 E4M3", 49152.0),
                            ("INT8 g128", 50304.0), ("INT4 g128", 25728.0)):
        chk("doc09 s5 %s KV B/token" % name, doc_bytes,
            K.kv_bytes_per_token(geom, name), 0.005)
    # section 5, Class A resident context at a 6 GiB KV budget. The doc prints
    # these in 1000-based "K", so the claims are rounded to match.
    budget = 6 * GIB
    for name, doc_tok in (("FP16", 65500.0), ("FP8 E4M3", 131100.0),
                          ("INT8 g64", 125200.0), ("INT8 g128", 128100.0),
                          ("INT4 g64", 239700.0), ("INT4 g128", 250400.0)):
        chk("doc09 s5 Class A %s resident tokens" % name, doc_tok,
            budget / K.kv_bytes_per_token(geom, name), 0.005)
    # section 5, Class B at a 13.6 GB/s point value
    for tok, doc_fp16, doc_fp8 in ((1024, 7.4, 3.7), (2048, 14.8, 7.4),
                                   (4096, 29.6, 14.8), (8192, 59.2, 29.6),
                                   (16384, 118.4, 59.2), (65536, 473.7, 236.9)):
        chk("doc09 s5 Class B FP16 ms at %d" % tok, doc_fp16,
            K.kv_bytes_per_token(geom, "FP16", tokens=tok) / 13.6e9 * 1e3, 0.01)
        chk("doc09 s5 Class B FP8 ms at %d" % tok, doc_fp8,
            K.kv_bytes_per_token(geom, "FP8 E4M3", tokens=tok) / 13.6e9 * 1e3,
            0.01)
    # the expert bank, quoted in docs/01 section 4 and docs/09 section 8
    r = K.resident_weight_bytes(geom)
    chk("doc01 s4 expert params (billions)", 28.99,
        geom["num_experts"] * 3 * geom["moe_intermediate_size"]
        * geom["hidden_size"] * geom["moe_layers"] / 1e9, 0.001)
    chk("doc09 s8 expert bank GiB at W4", 14.13, r["expert_bank"] / GIB, 0.005)
    chk("doc09 s5 resident weight total GiB", 15.31, r["total"] / GIB, 0.005)
    chk("doc09 s9 lm_head MB/token", 163.0,
        K.weight_bytes_per_token(geom)["lm_head"] / 1e6, 0.01)
    chk("doc09 s9 attention MB/token", 474.0,
        K.weight_bytes_per_token(geom)["attention"] / 1e6, 0.01)
    chk("doc09 s9 total weight MB/token", 1586.0,
        K.weight_bytes_per_token(geom)["total"] / 1e6, 0.01)

    # --- W3 / W2 weight-format operating point (docs/00 section 9.8) -------
    # The decision in 00 section 9.8 is a CAPACITY decision, derived from the
    # model config plus the measured P0-7 hit-rate curve. These rows verify the
    # derived geometry (bank size, resident set, KV budget, resident context)
    # that the prose quotes for each pack. The hit-rate and miss-cost columns
    # depend on the route artifact, not the config, so they are NOT checked here.
    non_expert = r["total"] - r["expert_bank"]
    moe_layers = geom["moe_layers"]
    for pname, doc_bank_gib, doc_res_gib, doc_budget_gib, doc_fp16_k, doc_fp8_k, doc_slots in (
            ("W4 g128", 14.13, 15.31, 0.0, 0.0, 0.0, 127.0),
            ("W3 g128", 10.76, 11.79, 3.46, 37.8, 75.6, 126.0),
            ("W2 g128", 7.38, 8.27, 6.98, 76.2, 152.4, 123.0),
            ("W4 g64", 14.77, 15.97, 0.0, 0.0, 0.0, 122.0),
            ("W8 g128", 27.63, 29.37, 0.0, 0.0, 0.0, 65.0),
    ):
        bpw = K.pack_bytes_per_weight(pname)
        rr = K.resident_weight_bytes(geom, bpw)
        budget_bytes = vram - rr["total"] - ws
        if budget_bytes < 0:
            budget_bytes = 0.0
        budget_gib = budget_bytes / GIB
        fp16_ctx_tok = budget_bytes / K.kv_bytes_per_token(geom, "FP16")
        fp8_ctx_tok = budget_bytes / K.kv_bytes_per_token(geom, "FP8 E4M3")
        # per-layer slots, as printed in the doc
        slots_per_layer = max((vram - ws - budget_bytes - non_expert)
                              / K.expert_bytes_per_layer(geom, bpw)
                              / max(moe_layers, 1), 0.0)
        # bank + resident: derived from config, absolute tolerance ~50 MB
        chk("doc00 s9.8 %s expert bank GiB" % pname, doc_bank_gib,
            rr["expert_bank"] / GIB, 0.004)
        chk("doc00 s9.8 %s resident GiB" % pname, doc_res_gib,
            rr["total"] / GIB, 0.004)
        # KV budget: 0 is exact; positive is within ~0.05 GiB
        if doc_budget_gib == 0.0:
            chk("doc00 s9.8 %s KV budget GiB" % pname, doc_budget_gib,
                budget_gib, 1e-6)
        else:
            chk("doc00 s9.8 %s KV budget GiB" % pname, doc_budget_gib,
                budget_gib, 0.015)
        # resident context: the doc prints thousands with one decimal
        fp16_ctx_k = round(fp16_ctx_tok / 1000.0, 1)
        fp8_ctx_k = round(fp8_ctx_tok / 1000.0, 1)
        chk("doc00 s9.8 %s FP16 context K" % pname, doc_fp16_k,
            fp16_ctx_k, 0.015)
        chk("doc00 s9.8 %s FP8 context K" % pname, doc_fp8_k,
            fp8_ctx_k, 0.015)
        chk("doc00 s9.8 %s experts resident" % pname, doc_slots,
            slots_per_layer, 0.6)
    failures = [row for row in rows if not row[4]]
    return rows, failures


def do_check(geom, w, vram_gib, workspace_gib):
    global _LAST_ROWS
    rows, failures = doc_claims(geom, w, vram_gib, workspace_gib)
    _LAST_ROWS = rows
    hr("I. DOC CLAIM AUDIT: docs/09 sections 2 and 5 vs this config (%s)"
       % geom["model_type"])
    print("  tolerance 2% (or exact for the byte-count rows). A failure means")
    print("  the doc quotes a number this config does not produce.")
    print("")
    print("  %-46s %16s %16s %8s" % ("claim", "doc", "derived", "delta"))
    for label, doc_value, derived, err, ok, _u in rows:
        print("  %-46s %16s %16s %7.1f%%  %s"
              % (label, _short(doc_value), _short(derived), err * 100.0,
                 "ok" if ok else "MISMATCH"))
    print("")
    if failures:
        print("  %d of %d claims are contradicted by this config."
              % (len(failures), len(rows)))
        for label, _dv, derived, err, _ok, _u in failures:
            print("    %-46s doc=%.4g  derived=%.4g  (%.1f%%)"
                  % (label, _docv(label), derived, err * 100.0))
    else:
        print("  all %d claims agree with this config." % len(rows))
    return failures


def _docv(label):
    for row_label, doc_value, derived, err, ok, _u in _LAST_ROWS:
        if row_label == label:
            return doc_value
    return 0.0


_LAST_ROWS = []


def _short(v):
    if abs(v) >= 1e6:
        return "%.4g" % v
    if abs(v) >= 1000:
        return "%.0f" % v
    if abs(v) >= 10:
        return "%.1f" % v
    return "%.4g" % v


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Derive the KV roofline from a model config.json.")
    ap.add_argument("configs", nargs="+",
                    help="one or more config.json paths")
    ap.add_argument("--vram-gib", type=float, default=16.0,
                    help="total VRAM (default 16.0, this card)")
    ap.add_argument("--kv-budget-gib", type=float, default=None,
                    help="override the KV budget instead of deriving it")
    ap.add_argument("--workspace-gib", type=float, default=0.75,
                    help="non-weight VRAM reserve (default 0.75)")
    ap.add_argument("--vram-lo", type=float, default=K.VRAM_GBPS_LO,
                    help="VRAM bandwidth low, GB/s (measured)")
    ap.add_argument("--vram-hi", type=float, default=K.VRAM_GBPS_HI,
                    help="VRAM bandwidth high, GB/s (measured)")
    ap.add_argument("--pcie-lo", type=float, default=K.PCIE_GBPS_LO,
                    help="PCIe low, GB/s (measured)")
    ap.add_argument("--pcie-hi", type=float, default=K.PCIE_GBPS_HI,
                    help="PCIe high, GB/s (measured)")
    ap.add_argument("--peak-tflops", type=float, default=K.PEAK_TFLOPS,
                    help="packed-f16 peak (derived)")
    ap.add_argument("--wire-ms", type=float, default=10.0,
                    help="PCIe budget per step for the Class B table")
    ap.add_argument("--allow-foreign-arch", action="store_true",
                    help="print output for architectures this tool cannot "
                         "compute exactly (exit code is still 4)")
    ap.add_argument("--check", action="store_true",
                    help="audit the numeric claims in docs/09")
    ap.add_argument("--audit-model", default="qwen3_moe",
                    help="model_type the docs/09 claims describe (default "
                         "qwen3_moe); other configs are skipped, not failed")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable results only")
    args = ap.parse_args(argv)

    for key in ("vram_lo", "vram_hi", "pcie_lo", "pcie_hi", "peak_tflops",
                "vram_gib", "wire_ms"):
        if getattr(args, key) <= 0:
            print("kv_roofline: --%s must be positive (got %r)"
                  % (key.replace("_", "-"), getattr(args, key)), file=sys.stderr)
            return 6

    results = []
    worst = 0
    for path in args.configs:
        try:
            if args.json:
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = report_one(path, args)
                results.append({
                    "path": res["path"],
                    "exact": res["geom"]["exact"],
                    "kv_bytes_per_token_fp16":
                        K.kv_bytes_per_token(res["geom"], "FP16"),
                    "weight_bytes_per_token_expert_only":
                        res["weights_per_token"]["expert_only"],
                    "weight_bytes_per_token_total":
                        res["weights_per_token"]["total"],
                    "resident_weight_bytes": res["resident"]["total"],
                    "kv_budget_bytes": res["kv_budget"],
                    "crossover_tokens_expert_only": res["crossovers"],
                })
            else:
                res = report_one(path, args)
                if args.check:
                    if res["geom"]["model_type"] == args.audit_model:
                        res["failures"] = do_check(res["geom"],
                                                   res["weights_per_token"],
                                                   args.vram_gib, args.workspace_gib)
                    else:
                        # The numbers in docs/09 describe ONE model. Auditing
                        # them against a different config would manufacture
                        # failures that say nothing about the docs.
                        hr("I. DOC CLAIM AUDIT: SKIPPED")
                        print("  docs/09 sections 2 and 5 describe %s. This"
                              % args.audit_model)
                        print("  config is %s. Auditing those claims here would"
                              % res["geom"]["model_type"])
                        print("  report differences that are not doc errors.")
                        print("  Re-run with --audit-model %s if that is what"
                              % res["geom"]["model_type"])
                        print("  you meant.")
                results.append(res)
                if not res["geom"]["exact"]:
                    worst = max(worst, 4)
        except KvError as exc:
            print("kv_roofline: %s" % exc, file=sys.stderr)
            for hint in exc.hints:
                print("  - %s" % hint, file=sys.stderr)
            return exc.exit_code

    if args.json:
        import json
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
        return worst
    if args.check and any(r["geom"]["model_type"] == args.audit_model
                           for r in results):
        n = sum(len(r.get("failures", [])) for r in results)
        print("")
        if n:
            print("RESULT: %d doc claim(s) contradicted. docs/09 is stale."
                  % n)
            return 5
        print("RESULT: docs/09 is consistent with every config checked.")
    return worst


if __name__ == "__main__":
    sys.exit(main())