#!/usr/bin/env python3
"""Read-only GGUF reconnaissance: header, metadata KV, tensor shapes/types.

Written to answer one question for a file the engine refuses: what architecture
does it actually carry, and what are its tensors called. Nothing here is part of
the engine; it is the equivalent of `strings` for a container.
"""
import os
import struct
import sys

TYPES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
         7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64"}

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
        9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
        15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
        20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
        26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
        35: "TQ2_0", 39: "MXFP4"}


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def s(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def val(self, t):
        if t == 0: return struct.unpack("<B", self.raw(1))[0]
        if t == 1: return struct.unpack("<b", self.raw(1))[0]
        if t == 2: return struct.unpack("<H", self.raw(2))[0]
        if t == 3: return struct.unpack("<h", self.raw(2))[0]
        if t == 4: return self.u32()
        if t == 5: return struct.unpack("<i", self.raw(4))[0]
        if t == 6: return struct.unpack("<f", self.raw(4))[0]
        if t == 7: return bool(struct.unpack("<B", self.raw(1))[0])
        if t == 8: return self.s()
        if t == 10: return self.u64()
        if t == 11: return struct.unpack("<q", self.raw(8))[0]
        if t == 12: return struct.unpack("<d", self.raw(8))[0]
        if t == 9:
            et = self.u32()
            n = self.u64()
            # Arrays of strings are the only ones worth printing whole.
            if et == 8:
                # Read EVERY element: the stream is sequential, so stopping early
                # desynchronises every later field. Only the display is truncated.
                vals = []
                for i in range(n):
                    s = self.s()
                    if i < 12:
                        vals.append(s)
                if n > 12:
                    vals.append(f"...(+{n - 12})")
                return vals
            if n > 4096:
                for _ in range(n):
                    self.val(et)
                return f"<array {TYPES.get(et, et)} x{n}>"
            return [self.val(et) for _ in range(n)]
        raise ValueError(f"unknown kv type {t}")


def main(path):
    with open(path, "rb") as f:
        r = R(f)
        magic = r.raw(4)
        if magic != b"GGUF":
            print("not a GGUF:", magic)
            return 1
        ver = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        print(f"FILE {path}")
        print(f"magic GGUF  version {ver}  tensors {n_tensors}  kv {n_kv}")
        kv = {}
        order = []
        for _ in range(n_kv):
            try:
                k = r.s()
                t = r.u32()
                if os.environ.get("GGUF_VERBOSE"):
                    print(f"  [kv {len(order):2d}] off={f.tell():9d} type={t:2d} {k}", flush=True)
                kv[k] = r.val(t)
                order.append(k)
            except Exception as e:
                print(f"\n!! KV parse stopped at entry {len(order)} "
                      f"(after {order[-3:]}) at offset {f.tell()}: {e}")
                break
        interesting = [k for k in kv if not k.startswith("tokenizer.ggml.tokens")
                       and not k.startswith("tokenizer.ggml.merges")
                       and not k.startswith("tokenizer.ggml.scores")]
        print(f"\n-- metadata ({len(interesting)} entries, tokenizer arrays summarised) --")
        for k in interesting:
            v = kv[k]
            s = repr(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  {k} = {s}")

        print(f"\n-- tensors ({n_tensors}) --")
        shapes = {}
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            tt = r.u32()
            r.u64()  # offset
            shapes[name] = (dims, tt)

        # Group by layer index and by role so the architecture is legible.
        import re
        roles = {}
        for name, (dims, tt) in shapes.items():
            m = re.match(r"^(.*?)\.(\d+)(\..*)?$", name)
            key = re.sub(r"\.\d+", ".N", name)
            roles.setdefault(key, []).append((name, dims, tt))
        for key in sorted(roles):
            entries = sorted(roles[key], key=lambda e: e[0])
            dims, tt = entries[0][1], entries[0][2]
            n = len(entries)
            extra = ""
            if n > 1:
                extra = f"  x{n}"
            print(f"  {key:44s} {str(dims):26s} {GGML.get(tt, tt)}{extra}")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
