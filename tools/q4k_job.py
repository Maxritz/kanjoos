#!/usr/bin/env python3
"""tools/q4k_job.py -- emit a device prefill job descriptor from a real GGUF.

WHY THIS EXISTS
---------------
The gfx1201 grouped-MoE driver (tools/bench/rocwmma_moe.hip) proved the rocWMMA
A/B fragment layout on SYNTHETIC W4 g128 data packed by the repo's own packer.
That is a real correctness result for the pack, but it says nothing about the
model: the format the models on this machine actually carry is GGUF Q4_K / Q6_K,
and a GEMM measured on a synthetic tensor of roughly the right shape is the
"passes without measuring anything" class -- the shape looks right, the bytes are
not the bytes.

So this script reads the REAL model, finds the real ffn_gate_exps / ffn_up_exps
(Q4_K) and ffn_down_exps (Q6_K) blocks plus the real ffn_gate_inp (router), and
writes their FILE byte offsets and geometry into a small job file that the device
driver consumes. The driver never parses GGUF; it seeks to a known offset, which
keeps the device code to the one thing under test.

gguf-py facts this script depends on (checked, not assumed):
  * tensor names carry the `.weight` suffix: `blk.0.ffn_gate_exps.weight`
  * `ReaderTensor.shape` is already in ggml `ne` order (dim0 = fastest), NOT
    reversed -- ffn_gate_exps.weight is ne = [1024, 3072, 4] = [k, n, experts]
  * the offset field is `data_offset`, and it is ALREADY AN ABSOLUTE FILE
    OFFSET. gguf_reader.py computes `data_offs = int(start_offs +
    offset_tensor[0])`, where `start_offs` is the data section. Adding
    `GGUFReader.data_offset` on top of it double-counts the header and lands in
    the middle of an unrelated tensor -- bytes that decode to plausible but
    wrong floats. This script first shipped with exactly that bug; it was caught
    by tools/q4k_unpack_crosscheck.py, and `verify_offset()` below now catches it
    in-script instead of eleven million weights downstream.
  * quant type ids: 12 = Q4_K, 14 = Q6_K, 0 = F32

Usage:
  python tools/q4k_job.py <model.gguf> <layer> <out.job>
"""
import sys

import gguf

# ggml quant type -> (name, bytes per superblock, weights per superblock)
QTYPE = {
    gguf.GGMLQuantizationType.Q4_K: ("q4_k", 144, 256),
    gguf.GGMLQuantizationType.Q6_K: ("q6_k", 210, 256),
}


def verify_offset(path, name, off, nbytes, tensor):
    """The offset is only trustworthy if the file agrees with gguf-py.

    `tensor.data` is the exact byte range gguf-py will hand to `dequantize`, so
    if the file at `off` is not those bytes then every downstream comparison is
    comparing the oracle against some other tensor's bytes -- which decodes
    successfully and produces plausible floats.
    """
    with open(path, "rb") as f:
        f.seek(off)
        onfile = f.read(nbytes)
    if len(onfile) != nbytes:
        raise SystemExit("%s: short read at %d (%d of %d bytes)"
                         % (name, off, len(onfile), nbytes))
    if onfile != memoryview(tensor.data).cast("B").tobytes():
        raise SystemExit("%s: file bytes at offset %d are NOT this tensor -- "
                         "the offset is wrong, do not measure on it" % (name, off))


def emit_all_layers(path, out):
    """Emit a whole-stack job: one verified offset line per transformer block.

    The streaming driver walks every layer, so it needs every layer's real
    offsets. Shapes are asserted uniform across blocks (qwen3moe guarantees it
    structurally, but the assert is what turns a surprise model into an error
    instead of a wrong measurement).
    """
    r = gguf.GGUFReader(path)
    arch = r.get_field("general.architecture").contents()
    if arch != "qwen3moe":
        raise SystemExit("job generator is qwen3moe-only, model says %r" % arch)

    def field_int(k):
        return int(r.get_field("qwen3moe.%s" % k).contents())

    hidden = field_int("embedding_length")
    ff = field_int("expert_feed_forward_length")
    experts = field_int("expert_count")
    topk = field_int("expert_used_count")
    nlayers = field_int("block_count")

    by_name = {t.name: t for t in r.tensors}

    def layer_tensor(suffix, layer, expect3d=True):
        name = "blk.%d.%s.weight" % (layer, suffix)
        if name not in by_name:
            raise SystemExit("tensor not found: %s" % name)
        t = by_name[name]
        ne = [int(d) for d in t.shape]
        return name, t, ne

    ref_rows = None
    lines = []
    for layer in range(nlayers):
        offs = []
        rows = []
        for suffix, role in (("ffn_gate_exps", "gate"), ("ffn_up_exps", "up"),
                             ("ffn_down_exps", "down")):
            name, t, ne = layer_tensor(suffix, layer)
            if t.tensor_type not in QTYPE:
                raise SystemExit("%s has unsupported type %s" % (name, t.tensor_type))
            tname, bpb, wpb = QTYPE[t.tensor_type]
            if len(ne) != 3 or ne[2] != experts:
                raise SystemExit("%s is not a [k, n, experts] expert tensor: ne=%s"
                                 % (name, ne))
            if ne[0] % wpb:
                raise SystemExit("%s k=%d is not a multiple of %d" % (name, ne[0], wpb))
            nbytes = ne[0] * ne[1] * ne[2] // wpb * bpb
            off = int(t.data_offset)
            verify_offset(path, name, off, nbytes, t)
            offs.append(off)
            rows.append(dict(role=role, k=ne[0], n=ne[1], bpb=bpb, wpb=wpb, bytes=nbytes))
        rname, rt, rne = layer_tensor("ffn_gate_inp", layer, expect3d=False)
        if rt.tensor_type != gguf.GGMLQuantizationType.F32:
            raise SystemExit("%s is %s, expected F32" % (rname, rt.tensor_type))
        if rne[0] != hidden or rne[1] != experts:
            raise SystemExit("router is %s, expected [hidden=%d, experts=%d]"
                             % (rne, hidden, experts))
        roff = int(rt.data_offset)
        verify_offset(path, rname, roff, 4 * rne[0] * rne[1], rt)
        offs.append(roff)
        if ref_rows is None:
            ref_rows = rows
        else:
            # Shapes must be uniform across blocks; the QUANT FORMAT need not be
            # (Q4_K_M importance matrices mix Q6_K and Q4_K down tensors --
            # measured on this model: 14 layers each). The driver therefore
            # carries a per-layer bytes-per-block, and the down kernel switches
            # decode per layer.
            for a, b in zip(ref_rows, rows):
                if (a["k"], a["n"], a["wpb"]) != (b["k"], b["n"], b["wpb"]):
                    raise SystemExit("layer %d shape differs from layer 0: %s vs %s"
                                     % (layer, a, b))
        lines.append((layer, offs, rows))

    with open(out, "w") as f:
        f.write("# generated by tools/q4k_job.py --all-layers -- do not edit\n")
        f.write("model %s\n" % path)
        f.write("layers %d\n" % nlayers)
        f.write("hidden %d\n" % hidden)
        f.write("ff %d\n" % ff)
        f.write("experts %d\n" % experts)
        f.write("topk %d\n" % topk)
        for rw in ref_rows:
            f.write("%s_type %s\n" % (rw["role"], "q4_k" if rw["bpb"] == 144 else "q6_k"))
            for key in ("k", "n", "bpb", "wpb"):
                f.write("%s_%s %d\n" % (rw["role"], key, rw[key]))
        # One line per layer: layer, gate off+bpb, up off+bpb, down off+bpb,
        # router off. The per-layer bpb is load-bearing: 14 of this model's 28
        # down tensors are Q4_K, not Q6_K.
        for layer, offs, rows in lines:
            f.write("L %d %d %d %d %d %d %d %d\n"
                    % (layer, offs[0], rows[0]["bpb"], offs[1], rows[1]["bpb"],
                       offs[2], rows[2]["bpb"], offs[3]))

    print("model       %s" % path)
    print("layers      %d   arch %s" % (nlayers, arch))
    print("geometry    hidden=%d ff=%d experts=%d topk=%d" % (hidden, ff, experts, topk))
    from collections import Counter
    for i, rw in enumerate(ref_rows):
        bs = Counter(l[2][i]["bpb"] for l in lines)
        print("  %-5s per-expert slab bpb mix %s (bytes at bpb: %s)"
              % (rw["role"], dict(bs),
                 {b: rw["k"] * rw["n"] * experts // rw["wpb"] * b for b in bs}))
    print("wrote       %s" % out)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--all-layers":
        return emit_all_layers(sys.argv[2], sys.argv[3])
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    path, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    r = gguf.GGUFReader(path)
    arch = r.get_field("general.architecture").contents()
    if arch != "qwen3moe":
        raise SystemExit("job generator is qwen3moe-only, model says %r" % arch)

    def field_int(k):
        return int(r.get_field("qwen3moe.%s" % k).contents())

    hidden = field_int("embedding_length")
    ff = field_int("expert_feed_forward_length")
    experts = field_int("expert_count")
    topk = field_int("expert_used_count")

    by_name = {t.name: t for t in r.tensors}

    rows = []
    for suffix, role in (("ffn_gate_exps", "gate"), ("ffn_up_exps", "up"),
                         ("ffn_down_exps", "down")):
        name = "blk.%d.%s.weight" % (layer, suffix)
        if name not in by_name:
            raise SystemExit("tensor not found: %s" % name)
        t = by_name[name]
        if t.tensor_type not in QTYPE:
            raise SystemExit("%s has unsupported type %s" % (name, t.tensor_type))
        tname, bpb, wpb = QTYPE[t.tensor_type]
        ne = [int(d) for d in t.shape]          # ggml ne order: [k, n, e]
        if len(ne) != 3:
            raise SystemExit("%s is not a 3-D expert tensor: %s" % (name, ne))
        if ne[2] != experts:
            raise SystemExit("%s expert dim %d != expert_count %d"
                             % (name, ne[2], experts))
        if ne[0] % wpb:
            raise SystemExit("%s k=%d is not a multiple of %d" % (name, ne[0], wpb))
        nbytes = ne[0] * ne[1] * ne[2] // wpb * bpb
        off = int(t.data_offset)
        verify_offset(path, name, off, nbytes, t)
        rows.append(dict(role=role, name=name, qtype=tname, off=off,
                         k=ne[0], n=ne[1], e=ne[2], bpb=bpb, wpb=wpb))

    if rows[0]["k"] != hidden or rows[0]["n"] != ff:
        raise SystemExit("gate is [%d, %d, %d], expected [hidden=%d, ff=%d, e]"
                         % (rows[0]["k"], rows[0]["n"], rows[0]["e"], hidden, ff))
    if rows[2]["k"] != ff or rows[2]["n"] != hidden:
        raise SystemExit("down is [%d, %d, %d], expected [ff=%d, hidden=%d, e]"
                         % (rows[2]["k"], rows[2]["n"], rows[2]["e"], ff, hidden))

    # The router. Not an expert tensor and not quantised, but the driver needs
    # REAL router weights so the routing is data-dependent rather than synthetic:
    # an uneven, input-driven top-k is what exercises the gather and the padding.
    rname = "blk.%d.ffn_gate_inp.weight" % layer
    if rname not in by_name:
        raise SystemExit("tensor not found: %s" % rname)
    rt = by_name[rname]
    if rt.tensor_type != gguf.GGMLQuantizationType.F32:
        raise SystemExit("%s is %s, expected F32" % (rname, rt.tensor_type))
    rne = [int(d) for d in rt.shape]
    if rne[0] != hidden or rne[1] != experts:
        raise SystemExit("router is %s, expected [hidden=%d, experts=%d]"
                         % (rne, hidden, experts))
    rbytes = 4 * rne[0] * rne[1]
    roff = int(rt.data_offset)
    verify_offset(path, rname, roff, rbytes, rt)

    with open(out, "w") as f:
        f.write("# generated by tools/q4k_job.py -- do not edit\n")
        f.write("model %s\n" % path)
        f.write("layer %d\n" % layer)
        f.write("hidden %d\n" % hidden)
        f.write("ff %d\n" % ff)
        f.write("experts %d\n" % experts)
        f.write("topk %d\n" % topk)
        for rw in rows:
            f.write("%s_off %d\n" % (rw["role"], rw["off"]))
            f.write("%s_type %s\n" % (rw["role"], rw["qtype"]))
            for key in ("k", "n", "bpb", "wpb"):
                f.write("%s_%s %d\n" % (rw["role"], key, rw[key]))
        f.write("router_off %d\n" % roff)
        f.write("router_type f32\n")
        f.write("router_k %d\n" % rne[0])
        f.write("router_n %d\n" % rne[1])

    print("model       %s" % path)
    print("layer       %d   arch %s" % (layer, arch))
    print("geometry    hidden=%d ff=%d experts=%d topk=%d" % (hidden, ff, experts, topk))
    for rw in rows:
        nbytes = rw["k"] * rw["n"] * rw["e"] // rw["wpb"] * rw["bpb"]
        print("  %-5s %-28s %-5s off=%-10d %d x %d x %d  %d bytes"
              % (rw["role"], rw["name"], rw["qtype"], rw["off"],
                 rw["k"], rw["n"], rw["e"], nbytes))
    print("  %-5s %-28s %-5s off=%-10d %d x %d        %d bytes"
          % ("rtr", rname, "f32", roff, rne[0], rne[1], rbytes))
    print("wrote       %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
