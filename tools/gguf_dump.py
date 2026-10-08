#!/usr/bin/env python3
"""Minimal GGUF reader used to inspect a model before wiring a C++ reader.

Not part of the build; a development probe. Prints metadata KV pairs and the
tensor table (name, dims, ggml type, absolute offset) in file order.
"""
import struct
import sys

GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}
# GGUF metadata value type ids
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, \
    T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)

SIZES = {T_UINT8: 1, T_INT8: 1, T_UINT16: 2, T_INT16: 2, T_UINT32: 4,
         T_INT32: 4, T_FLOAT32: 4, T_BOOL: 1, T_UINT64: 8, T_INT64: 8,
         T_FLOAT64: 8}
FMT = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
       T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<B",
       T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        assert len(b) == n, "short read"
        return b

    def scalar(self, t):
        return struct.unpack(FMT[t], self.raw(SIZES[t]))[0]

    def string(self):
        n = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et = struct.unpack("<I", self.raw(4))[0]
            n = struct.unpack("<Q", self.raw(8))[0]
            return [self.value(et) for _ in range(n)]
        return self.scalar(t)


def main(path):
    with open(path, "rb") as f:
        r = Reader(f)
        magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", r.raw(24))
        print(f"magic={magic!r} version={ver} n_tensors={n_tensors} n_kv={n_kv}")
        assert magic == b"GGUF", "not a GGUF"
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            vt = struct.unpack("<I", r.raw(4))[0]
            val = r.value(vt)
            meta[key] = val
        # tensor table
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = struct.unpack("<I", r.raw(4))[0]
            dims = [struct.unpack("<Q", r.raw(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", r.raw(4))[0]
            off = struct.unpack("<Q", r.raw(8))[0]
            tensors.append((name, dims, tt, off))
        data_start = f.tell()
        # gguf pads the tensor data to a 32-byte boundary
        pad = (-data_start) % 32
        data_start += pad
        print(f"header ends at {data_start} (aligned), file size {__import__('os').path.getsize(path)}")
        print("\n--- metadata ---")
        for k in sorted(meta):
            v = meta[k]
            if isinstance(v, list) and len(v) > 12:
                v = v[:12] + [f"...(+{len(meta[k]) - 12} more, len={len(meta[k])})"]
            elif isinstance(v, bytes):
                v = f"<{len(v)} bytes>"
            print(f"{k} = {v}")
        print("\n--- tensors (file order) ---")
        # offset is relative to data_start; convert to absolute
        for name, dims, tt, off in tensors:
            print(f"{name:48s} dims={str(dims):22s} type={GGML_TYPE.get(tt, tt):8s} "
                  f"rel_off={off:12d} abs={data_start + off}")
        # group by type
        from collections import Counter
        c = Counter(GGML_TYPE.get(tt, tt) for _, _, tt, _ in tensors)
        print("\n--- quant mix ---")
        for k, v in c.most_common():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1])
