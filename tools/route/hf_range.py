#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf_range.py -- read individual tensors out of safetensors files, locally or
over HTTP with Range requests.

Local reads use the installed `safetensors` library. Remote reads use the
safetensors container's own header: 8 bytes of little-endian length, then that
many bytes of JSON naming every tensor's dtype, shape and byte offset. That is
enough to fetch ONE tensor from a 4 GB shard without downloading the shard.

Why this exists: the routing matrices of the target model are 25 MB inside a
61 GB checkpoint. Measuring P0-7 does not require 61 GB.
"""

from __future__ import annotations

import json
import struct
import urllib.request

HF = "https://huggingface.co"


class RemoteShard:
    """One safetensors file on a HuggingFace repo, readable tensor by tensor."""

    def __init__(self, repo_id, filename, revision="main"):
        self.url = "%s/%s/resolve/%s/%s" % (HF, repo_id, revision, filename)
        self._header = None

    def _get(self, start, end):
        req = urllib.request.Request(self.url)
        req.add_header("Range", "bytes=%d-%d" % (start, end))
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()

    def header(self):
        if self._header is None:
            raw = self._get(0, 7)
            (n,) = struct.unpack("<Q", raw)
            hdr = json.loads(self._get(8, 7 + n).decode("utf-8"))
            hdr.pop("__metadata__", None)
            self._header = hdr
        return self._header

    def names(self):
        return sorted(k for k in self.header() if k != "__metadata__")

    def get(self, name):
        import numpy as np
        entry = self.header()[name]
        start, end = entry["data_offsets"]
        buf = self._get(start, end - 1)
        if len(buf) != end - start:
            raise IOError("short read for %s: %d of %d bytes"
                          % (name, len(buf), end - start))
        dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
                 "F64": np.float64, "I64": np.int64, "I32": np.int32}[entry["dtype"]]
        arr = np.frombuffer(buf, dtype=dtype).reshape(entry["shape"])
        if entry["dtype"] == "BF16":
            # bf16 -> fp32 by widening; exact, no value is changed.
            bits = arr.astype(np.uint32) << 16
            arr = bits.view(np.float32).reshape(entry["shape"])
        return arr


def weight_map(repo_id, revision="main"):
    """{tensor_name: shard_filename} from the repo's safetensors index."""
    url = "%s/%s/resolve/%s/model.safetensors.index.json" % (HF, repo_id, revision)
    with urllib.request.urlopen(url, timeout=120) as resp:
        idx = json.loads(resp.read().decode("utf-8"))
    return idx["weight_map"]


class ShardCache:
    def __init__(self, repo_id, revision="main"):
        self.repo_id = repo_id
        self.revision = revision
        self.wmap = weight_map(repo_id, revision)
        self._shards = {}

    def get(self, name):
        shard = self.wmap.get(name)
        if shard is None:
            return None
        if shard not in self._shards:
            self._shards[shard] = RemoteShard(self.repo_id, shard, self.revision)
        return self._shards[shard].get(name)

    def has(self, name):
        return name in self.wmap