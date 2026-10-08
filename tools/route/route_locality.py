#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
route_locality.py -- P0-7. Measure MoE routing locality for real.

P0-7 was the last blocking unknown. It decides two things:

  (a) the EXPERT SLOT BUDGET -- how many of a layer's experts must be resident
      in VRAM before the miss rate stops mattering, and
  (b) whether CLASS B is viable -- a spilled KV tier is one question, but a
      spilled EXPERT tier is the other, and the second one has never been
      priced.

Two independent sources of routing are measured, because neither alone is
enough:

  SOURCE 1  local   A real MoE (Qwen1.5-MoE-A2.7B-Chat on this disk) run
                   end to end on real text with hooks on every gate. Its
                   gates see its own true hidden states, so this is a
                   self-consistent measurement of real routing. 60 experts,
                   top-4, 24 layers.

  SOURCE 2  target  The REAL gate matrices of Qwen3-30B-A3B itself (48 of
                   them, 25 MB pulled by HTTP range out of a 61 GB
                   checkpoint), applied to real hidden states from Qwen3-1.7B,
                   which is the same family and the SAME hidden width (2048).
                   This is NOT the target model's own forward pass and is
                   labelled as such everywhere it is printed. It is here
                   because it measures the routing FUNCTION the target will
                   actually run, at the layer count the target has.

Both sources answer the same question with different biases, and the answer
that matters -- the shape of the hit-rate-vs-slots curve -- is a property of
routing, not of one checkpoint.

Exit codes
----------
    0   trace collected and reported
    2   a required config field is absent (no guess is made)
    3   model files missing or unreadable
    5   no routing recorded -- the hooks found no gate. Treated as failure,
        because a silently empty trace is the classic way this goes wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kvroof"))

import pack_common as P

GIB = 1024.0 ** 3
MEASURED_PCIE_GBPS = (13.4, 14.7)     # 00-verified-facts.md section 8.1
W4_BYTES_PER_WEIGHT = P.W4_BYTES_PER_WEIGHT   # tools/kvroof/pack_common.py -> kernels/knj_pack.h
ARITH_US_AT_PEAK = 7.68                  # section 8.3, per expert, 100% of peak


# ------------------------------------------------------------------ traces
def trace_metrics(topk, num_experts, top_k):
    """topk: uint8 array [n_tokens, top_k] of expert ids, ascending order not
    assumed. Returns the locality statistics the engine actually needs."""
    n = topk.shape[0]
    flat = topk.reshape(-1)

    # --- popularity -------------------------------------------------------
    counts = np.bincount(flat, minlength=num_experts).astype(np.float64)
    total = counts.sum()
    order = np.argsort(-counts)
    csum = np.cumsum(counts[order]) / total
    used = int((counts > 0).sum())
    conc = {}
    for frac in (0.25, 0.5, 0.75):
        k = max(1, int(round(frac * num_experts)))
        conc["top%d%%" % int(frac * 100)] = int(round(csum[k - 1] * 100))
    p = counts / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    # --- temporal locality ------------------------------------------------
    if n > 1:
        prev = [set(topk[i].tolist()) for i in range(n - 1)]
        cur = [set(topk[i].tolist()) for i in range(1, n)]
        inter = sum(len(a & b) for a, b in zip(prev, cur)) / float((n - 1) * top_k)
        union = sum(len(a | b) for a, b in zip(prev, cur)) / float((n - 1) * top_k)
        jac = sum(len(a & b) / max(len(a | b), 1) for a, b in zip(prev, cur)) / (n - 1)
    else:
        inter = union = jac = float("nan")

    # --- slot budget curves ----------------------------------------------
    # policy "static": the S most popular experts are the resident set. This
    # is the ORACLE policy -- it assumes perfect foresight of frequency.
    # policy "lru":    realistic; on a miss the least-recently-used slot goes.
    curves = {}
    for policy in ("static", "lru"):
        curves[policy] = {}
        for S in _slot_grid(num_experts, top_k):
            curves[policy][S] = _hit_rate(topk, num_experts, S, policy)
    return {
        "n_tokens": int(n),
        "top_k": int(top_k),
        "num_experts": int(num_experts),
        "experts_ever_used": used,
        "concentration_pct_of_selections": conc,
        "entropy_bits": entropy,
        "entropy_max_bits": float(np.log2(num_experts)),
        "temporal_reuse": float(inter),
        "temporal_jaccard": float(jac),
        "temporal_new_expert_rate": float(1.0 - inter),
        "curves": curves,
    }


def _slot_grid(num_experts, top_k):
    grid = set()
    for frac in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        grid.add(max(top_k, int(round(frac * num_experts))))
    grid.add(num_experts)
    return sorted(grid)


def _hit_rate(topk, num_experts, slots, policy):
    if slots >= num_experts:
        return 1.0
    resident = set()
    lru = []          # most recent last
    counts = np.bincount(topk.reshape(-1), minlength=num_experts)
    if policy == "static":
        resident = set(np.argsort(-counts)[:slots].tolist())
    hits = 0
    lookups = topk.size
    for row in topk:
        miss = [e for e in row.tolist() if e not in resident]
        hits += len(row) - len(miss)
        if miss:
            if policy == "static":
                # a static oracle set cannot absorb a miss: it would have to
                # evict, and evicting means giving up a popular expert. The
                # honest oracle replaces the least popular resident expert.
                pop = {e: counts[e] for e in resident}
                for e in miss:
                    if len(resident) < slots:
                        resident.add(e)
                        continue
                    victim = min(resident, key=lambda x: pop.get(x, 0))
                    resident.discard(victim)
                    resident.add(e)
                    pop.pop(victim, None)
                    pop[e] = counts[e]
            else:
                for e in miss:
                    if e in lru:
                        lru.remove(e)
                    lru.append(e)
                    while len(lru) > slots:
                        resident.discard(lru.pop(0))
                    resident.add(e)
    return hits / float(lookups)


# ------------------------------------------------------- source 1: local
def collect_local(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mdir = args.model
    cfg_path = os.path.join(mdir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit3("no config.json in %s" % mdir)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", cfg)
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size",
                "num_hidden_layers", "hidden_size"):
        if tc.get(key) is None:
            raise SystemExit2("config lacks %s; refusing to assume it" % key)
    num_experts = tc["num_experts"]
    top_k = tc["num_experts_per_tok"]
    moe_inter = tc["moe_intermediate_size"]
    n_layers = tc["num_hidden_layers"]
    hidden = tc["hidden_size"]

    print("  model      %s" % mdir)
    print("  experts    %d, top-%d, moe_intermediate=%d, layers=%d, hidden=%d"
          % (num_experts, top_k, moe_inter, n_layers, hidden))
    print("  loading weights to CPU in bf16 (this is a %s-GB checkpoint)"
          % "%.1f" % (os.path.getsize(os.path.join(mdir, "model-00001-of-00008.safetensors")) / GIB
                      if os.path.exists(os.path.join(mdir, "model-00001-of-00008.safetensors"))
                      else 27))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    print("  loaded in %.1f s" % (time.time() - t0))

    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids[:, :args.max_tokens]
    print("  text       %d tokens (corpus %.1f KiB)"
          % (ids.shape[1], len(text) / 1024.0))

    traces = [[] for _ in range(n_layers)]

    def make_hook(li):
        def hook(_mod, _inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            flat = logits.reshape(-1, logits.shape[-1]).float()
            idx = torch.topk(flat, top_k, dim=-1).indices.cpu().numpy()
            traces[li].append(idx.astype(np.uint8))
        return hook

    handles = []
    layers = model.model.layers
    for li, layer in enumerate(layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:
            print("  layer %d has no mlp.gate -- hooks incomplete" % li)
            continue
        handles.append(gate.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        t0 = time.time()
        model(ids)
        print("  prefill    %.1f s" % (time.time() - t0))
    for h in handles:
        h.remove()

    got = [t for t in traces if t]
    if not got:
        raise SystemExit5("no routing recorded: the gate hooks never fired")
    per_layer = [np.concatenate(traces[li], axis=0)
                 for li in range(n_layers) if traces[li]]
    print("  routing    %d of %d layers hooked, %d tokens each"
          % (len(per_layer), n_layers, per_layer[0].shape[0]))

    return {"kind": "local", "name": os.path.basename(mdir.rstrip("/\\")),
            "num_experts": num_experts, "top_k": top_k,
            "moe_intermediate": moe_inter, "hidden": hidden,
            "layers": n_layers, "per_layer": per_layer,
            "caveat": "self-consistent: these gates saw this model's own hidden "
                      "states"}


# ------------------------------------------------------ source 2: target
def collect_target(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import hf_range

    donor_dir = args.donor
    with open(os.path.join(donor_dir, "config.json"), "r", encoding="utf-8") as fh:
        dcfg = json.load(fh)
    tok = AutoTokenizer.from_pretrained(donor_dir)
    model = AutoModelForCausalLM.from_pretrained(donor_dir, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids[:, :args.max_tokens]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states          # tuple [n_donor_layers+1] of [1, T, hidden]
    donor_layers = len(hs) - 1
    donor_hidden = hs[0].shape[-1]

    print("  donor      %s (%d layers, hidden=%d, %d tokens)"
          % (args.donor, donor_layers, donor_hidden, hs[0].shape[1]))

    cache = hf_range.ShardCache(args.repo)
    # The gate matrix is [num_experts, hidden_size]. Its second dimension is
    # the HIDDEN width, NOT moe_intermediate_size -- conflating the two
    # inflates every expert byte figure by hidden/moe_inter (2.67x on this
    # model). So the geometry comes from the repo's own config.json, parsed
    # by the same resolver the KV roofline uses.
    cfg_url = "%s/%s/resolve/main/config.json" % (hf_range.HF, args.repo)
    import urllib.request
    with urllib.request.urlopen(cfg_url, timeout=120) as resp:
        target_cfg = json.loads(resp.read().decode("utf-8"))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kvroof"))
    import kvroof as K
    tg = K.resolve_geometry(target_cfg, args.repo)
    print("  config     %s: %d experts, top-%d, moe_intermediate=%d, hidden=%d,"
          % (args.repo, tg["num_experts"], tg["top_k"],
             tg["moe_intermediate_size"], tg["hidden_size"]))
    print("             layers=%d (READ, not assumed)" % tg["num_layers"])
    gate_names = sorted(n for n in cache.wmap if "mlp.gate.weight" in n)
    if not gate_names:
        raise SystemExit5("no mlp.gate.weight in the %s index" % args.repo)
    print("  gates      %d found in %s" % (len(gate_names), args.repo))

    # Range-fetching 48 tensors costs ~100 s of pure round trips, so the
    # fetched gates are cached on disk next to this tool. The cache is keyed
    # by repo name; delete it to re-fetch.
    cache_path = os.path.join(HERE, "gate_cache_%s.npz"
                              % args.repo.replace("/", "_"))
    gates = None
    if os.path.exists(cache_path) and not args.refetch:
        z = np.load(cache_path)
        gates = [z[k] for k in sorted(z.files,
                                      key=lambda s: int(s.split("_")[1]))]
        print("  gates      loaded from cache %s (%d matrices)"
              % (os.path.basename(cache_path), len(gates)))
    if gates is None:
        print("  fetching   %d matrices over HTTP Range (%.1f MB total)"
              % (len(gate_names), len(gate_names) * 128 * 2048 * 2 / 1e6))
        t0 = time.time()
        gates = []
        for gi, name in enumerate(gate_names):
            w = cache.get(name)
            if w is None:
                continue
            gates.append(np.asarray(w, dtype=np.float32))
            if (gi + 1) % 12 == 0:
                print("    %2d/%d gates" % (gi + 1, len(gate_names)))
        print("  fetched    %.1f s" % (time.time() - t0))
        if gates:
            np.savez(cache_path,
                     **{"g_%03d" % i: g for i, g in enumerate(gates)})
            print("  cached to  %s" % os.path.basename(cache_path))

    traces = []
    num_experts = top_k = None
    for gi, w in enumerate(gates):
        if num_experts is None:
            num_experts = w.shape[0]
            top_k = tg["top_k"] if not args.top_k else args.top_k
        if w.shape[0] != tg["num_experts"]:
            raise SystemExit2("gate has %d experts, config says %d"
                              % (w.shape[0], tg["num_experts"]))
        if w.shape[1] != donor_hidden:
            raise SystemExit2("gate width %d != donor hidden %d: cannot "
                              "compare without a projection, and inventing "
                              "one would measure nothing" % (w.shape[1], donor_hidden))
        src = hs[min(gi, donor_layers)].reshape(-1, donor_hidden).float().numpy()
        logits = src @ w.T
        idx = np.argsort(-logits, axis=1)[:, :top_k]
        traces.append(idx.astype(np.uint8))

    if not traces:
        raise SystemExit5("no gates fetched")

    return {"kind": "target-gates", "name": args.repo,
            "num_experts": num_experts, "top_k": top_k,
            "moe_intermediate": tg["moe_intermediate_size"],
            "hidden": tg["hidden_size"],
            "layers": len(traces), "per_layer": traces,
            "caveat": "PROXY: the target's REAL gate matrices applied to REAL "
                      "hidden states from %s (same family, same hidden width "
                      "%d). Not the target model's own forward pass."
                      % (os.path.basename(donor_dir.rstrip("/\\")), donor_hidden)}


# ------------------------------------------------------------------ misc
class SystemExit2(Exception):
    pass


class SystemExit3(Exception):
    pass


class SystemExit5(Exception):
    pass


def load_text(args):
    if args.prompt:
        return args.prompt
    chunks = []
    for path in args.corpus:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            chunks.append(fh.read())
        chunks.append("\n\n")
    if not chunks:
        raise SystemExit3("no --corpus files and no --prompt")
    return "".join(chunks)


# ----------------------------------------------------------------- report
def report(src, args):
    per_layer = src["per_layer"]
    num_experts = src["num_experts"]
    top_k = src["top_k"]
    expert_bytes = 3 * src["moe_intermediate"] * src["hidden"] * W4_BYTES_PER_WEIGHT
    n_layers = src["layers"]

    print("")
    print("=" * 78)
    print("P0-7 ROUTING LOCALITY -- %s" % src["name"])
    print("=" * 78)
    print("  %d experts, top-%d, %d layers, %d tokens/layer"
          % (num_experts, top_k, n_layers, per_layer[0].shape[0]))
    print("  geometry read from the model's own config, not inferred: "
          "moe_intermediate=%d, hidden=%d"
          % (src["moe_intermediate"], src["hidden"]))
    print("  one expert in one layer at W4 = %.2f MiB"
          % (expert_bytes / 1024.0 / 1024.0))
    print("  CAVEAT: %s" % src["caveat"])

    per_layer_metrics = [trace_metrics(t, num_experts, top_k) for t in per_layer]
    agg = _aggregate(per_layer_metrics, num_experts, top_k)

    print("")
    print("1. HOW MANY EXPERTS DOES A LAYER ACTUALLY USE")
    print("  %-34s %s" % ("", "min / median / max over layers"))
    print("  %-34s %d / %.0f / %d"
          % ("experts ever selected", agg["used_min"], agg["used_med"],
             agg["used_max"]))
    print("  %-34s %.2f / %.2f / %.2f  (max is %.1f)"
          % ("entropy (bits)", agg["ent_min"], agg["ent_med"], agg["ent_max"],
             np.log2(num_experts)))
    print("  %-34s %.1f%% / %.1f%% / %.1f%%"
          % ("of selections in top 25% of experts",
             agg["c25_min"], agg["c25_med"], agg["c25_max"]))

    print("")
    print("2. TEMPORAL LOCALITY (consecutive tokens, same layer)")
    print("  %-34s %.3f / %.3f / %.3f"
          % ("top-k reused from t-1 (mean)", agg["reuse_min"], agg["reuse_med"],
             agg["reuse_max"]))
    print("  %-34s %.1f%% (median)"
          % ("NEW experts demanded per token", 100.0 * (1.0 - agg["reuse_med"])))

    print("")
    print("3. EXPERT SLOT BUDGET -- hit rate vs resident slots per layer")
    print("  (static = oracle on measured popularity; LRU = realistic)")
    print("")
    print("  %-8s %-22s %-22s %s"
          % ("slots", "static hit rate", "LRU hit rate", "slots/128"))
    for S in agg["slots_grid"]:
        st = agg["static"][S]
        lr = agg["lru"][S]
        bar = "#" * int(round(st * 40))
        print("  %-8d %5.1f%% %-16s %5.1f%% %-16s %s"
              % (S, 100 * st, bar, 100 * lr, "#" * int(round(lr * 40)),
                 "%.0f%%" % (100.0 * S / num_experts)))

    print("")
    print("4. SMALLEST SLOT SET PER LAYER FOR A HIT-RATE TARGET")
    for target in (0.90, 0.95, 0.99):
        s_static = agg["min_slots_static"][target]
        s_lru = agg["min_slots_lru"][target]
        print("  %2d%% hit rate:  static oracle %s slots   LRU %s slots   "
              "(of %d experts, %.0f%% of a layer)"
              % (int(target * 100),
                 "%.0f" % s_static if s_static else "never",
                 "%.0f" % s_lru if s_lru else "never",
                 num_experts, 100.0 * (s_lru or num_experts) / num_experts))

    print("")
    print("5. WHAT THE MISSES COST  (measured PCIe %.1f-%.1f GB/s)"
          % MEASURED_PCIE_GBPS)
    lo, hi = MEASURED_PCIE_GBPS
    print("  %-22s %14s %16s %16s"
          % ("LRU hit rate at...", "slots", "ms/token (lo)", "ms/token (hi)"))
    for S in agg["slots_grid"]:
        hit = agg["lru"][S]
        miss_frac = 1.0 - hit
        bytes_tok = miss_frac * top_k * expert_bytes * n_layers
        print("  %-22s %14d %16.1f %16.1f"
              % ("%.0f%% hit" % (100 * hit), S,
                 bytes_tok / (lo * 1e9) * 1e3, bytes_tok / (hi * 1e9) * 1e3))
    print("  (a miss costs one PCIe hop per layer, for every token; a hit costs")
    print("   nothing but VRAM. Arithmetic at 100%% of peak is %.2f us/expert."
          % ARITH_US_AT_PEAK)

    print("")
    print("6. VERDICT INPUTS")
    full = 1.0
    s95 = agg["min_slots_lru"][0.95]
    if s95:
        print("  - %.0f of %d experts per layer (%.0f%% of the bank) buys a"
              % (s95, num_experts, 100.0 * s95 / num_experts))
        print("    95%% hit rate under LRU, leaving %.0f%% of expert VRAM for KV."
              % (100.0 * (1.0 - s95 / num_experts)))
    print("  - Full residency (100%% hit) costs %.1f GiB of expert VRAM and"
          % (num_experts * expert_bytes * n_layers / GIB))
    print("    cannot be combined with a meaningful KV budget on this card.")
    print("  - Expert spilling is the SAME wire problem as KV spilling: at")
    print("    %.1f-%.1f GB/s a single expert costs %.2f-%.2f us per layer,"
          % (lo, hi, expert_bytes / (lo * 1e9) * 1e6, expert_bytes / (hi * 1e9) * 1e6))
    print("    so a %d-layer decode step pays %.2f-%.2f ms for ONE missed"
          % (n_layers, expert_bytes * n_layers / (lo * 1e9) * 1e3,
             expert_bytes * n_layers / (hi * 1e9) * 1e3))
    print("    expert per token before anything else happens.")

    return per_layer_metrics, agg


def _aggregate(per_layer, num_experts, top_k):
    def col(key):
        return np.array([m[key] for m in per_layer], dtype=np.float64)

    used = np.array([m["experts_ever_used"] for m in per_layer], dtype=np.float64)
    ent = col("entropy_bits")
    reuse = col("temporal_reuse")
    c25 = np.array([m["concentration_pct_of_selections"]["top25%"]
                    for m in per_layer], dtype=np.float64)
    grid = sorted(per_layer[0]["curves"]["static"].keys())
    static = {S: float(np.median([m["curves"]["static"][S] for m in per_layer]))
              for S in grid}
    lru = {S: float(np.median([m["curves"]["lru"][S] for m in per_layer]))
           for S in grid}
    min_static, min_lru = {}, {}
    for target in (0.90, 0.95, 0.99):
        s = next((S for S in grid if static[S] >= target), None)
        min_static[target] = s
        min_lru[target] = next((S for S in grid if lru[S] >= target), None)
    return {
        "used_min": int(used.min()), "used_med": float(np.median(used)),
        "used_max": int(used.max()),
        "ent_min": ent.min(), "ent_med": float(np.median(ent)),
        "ent_max": ent.max(),
        "reuse_min": reuse.min(), "reuse_med": float(np.median(reuse)),
        "reuse_max": reuse.max(),
        "c25_min": c25.min(), "c25_med": float(np.median(c25)),
        "c25_max": c25.max(),
        "slots_grid": grid, "static": static, "lru": lru,
        "min_slots_static": min_static, "min_slots_lru": min_lru,
    }


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="P0-7 routing locality")
    ap.add_argument("--source", choices=("local", "target-gates"), default="local")
    ap.add_argument("--model", help="local MoE directory (source=local)")
    ap.add_argument("--repo", default="Qwen/Qwen3-30B-A3B",
                    help="HF repo for the real gate matrices (source=target-gates)")
    ap.add_argument("--donor", help="local model providing real hidden states")
    ap.add_argument("--top-k", type=int, default=8,
                    help="top-k for target-gates source (default 8)")
    ap.add_argument("--corpus", nargs="*", default=[],
                    help="text files to route")
    ap.add_argument("--prompt", help="literal text instead of a corpus")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--refetch", action="store_true",
                    help="ignore the on-disk gate cache and range-fetch again")
    ap.add_argument("--tag", default="",
                    help="suffix for the JSON artifact, so per-corpus runs "
                         "do not overwrite each other (e.g. --tag ru)")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("P0-7 ROUTING LOCALITY PROBE")
    print("=" * 78)
    try:
        if args.source == "local":
            if not args.model:
                ap.error("--model is required for --source local")
            src = collect_local(args)
        else:
            if not args.donor:
                ap.error("--donor is required for --source target-gates")
            src = collect_target(args)
    except SystemExit2 as exc:
        print("route_locality: %s" % exc, file=sys.stderr)
        return 2
    except SystemExit3 as exc:
        print("route_locality: %s" % exc, file=sys.stderr)
        return 3
    except SystemExit5 as exc:
        print("route_locality: %s" % exc, file=sys.stderr)
        return 5

    per_layer_metrics, agg = report(src, args)
    out = {"source": src["name"], "kind": src["kind"],
           "caveat": src["caveat"], "num_experts": src["num_experts"],
           "top_k": src["top_k"], "layers": src["layers"],
           "aggregate": agg,
           "per_layer": [{k: v for k, v in m.items() if k != "curves"}
                         for m in per_layer_metrics]}
    path = os.path.join(HERE, "route_%s%s.json"
                        % (src["kind"], ("_" + args.tag) if args.tag else ""))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("")
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())