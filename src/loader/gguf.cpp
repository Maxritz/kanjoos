// src/loader/gguf.cpp -- GGUF container reader implementation.
#include "src/loader/gguf.h"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
// Guarded because the C++ library's own platform headers may already define
// these (MinGW's bits/os_defines.h does), and an unguarded redefine is a
// warning on every build -- noise that trains a reader to ignore the build log.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace knj {
namespace {

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error("gguf: " + msg);
}

uint64_t align_up(uint64_t v, uint64_t a) {
  if (a == 0) fail("alignment of 0");
  return (v + a - 1) / a * a;
}

// Cursor over the mapped bytes. Every read is bounds-checked against the
// mapping, so a truncated or hostile file is refused at the point of the bad
// read instead of producing a plausible-looking wrong tensor table.
struct Cursor {
  const uint8_t* p;
  uint64_t       n;
  uint64_t       at = 0;

  void need(uint64_t k) const {
    if (at + k > n) fail("truncated file (wanted " + std::to_string(k) +
                         " bytes at offset " + std::to_string(at) + " of " +
                         std::to_string(n) + ")");
  }
  const uint8_t* raw(uint64_t k) {
    need(k);
    const uint8_t* r = p + at;
    at += k;
    return r;
  }
  template <typename T>
  T pod() {
    T v;
    std::memcpy(&v, raw(sizeof(T)), sizeof(T));
    return v;
  }
  std::string str() {
    uint64_t len = pod<uint64_t>();
    const uint8_t* b = raw(len);
    return std::string(reinterpret_cast<const char*>(b), static_cast<size_t>(len));
  }
};

size_t meta_scalar_size(MetaType t) {
  switch (t) {
    case MetaType::UINT8:
    case MetaType::INT8:
    case MetaType::BOOL:    return 1;
    case MetaType::UINT16:
    case MetaType::INT16:   return 2;
    case MetaType::UINT32:
    case MetaType::INT32:
    case MetaType::FLOAT32: return 4;
    case MetaType::UINT64:
    case MetaType::INT64:
    case MetaType::FLOAT64: return 8;
    default: fail("value type " + std::to_string(static_cast<uint32_t>(t)) +
                  " is not a scalar");
  }
}

MetaValue read_value(Cursor& c, MetaType t) {
  MetaValue v;
  v.type = t;
  if (t == MetaType::STRING) {
    v.s = c.str();
    return v;
  }
  if (t == MetaType::ARRAY) {
    v.elem_type = static_cast<MetaType>(c.pod<uint32_t>());
    uint64_t n = c.pod<uint64_t>();
if (v.elem_type == MetaType::ARRAY) fail("nested arrays are not valid GGUF");
    // A declared count larger than the remaining bytes cannot be honest. The
    // check is cheap and turns a 2^63 element array into an error rather than
    // an allocation attempt.
    //
    // Reserve to the HONEST cap, not a fixed 1024: no element is smaller than
    // its own size (8 bytes for a string's length prefix, the scalar width for
    // the rest), so the file's remaining bytes bound the true count. A
    // 152K-entry vocab is then one allocation instead of ~9 geometric copies
    // of a 12 MB vector. A count past the cap is still refused exactly where it
    // always was -- inside the loop, at the read that runs off the file.
    const uint64_t per =
        (v.elem_type == MetaType::STRING) ? 8 : meta_scalar_size(v.elem_type);
    const uint64_t honest = (c.n - c.at) / per + 1;
    v.arr.reserve(static_cast<size_t>(n < honest ? n : honest));
    for (uint64_t i = 0; i < n; ++i) {
      if (c.at >= c.n) fail("array of " + std::to_string(n) + " ran off the file");
      v.arr.push_back(read_value(c, v.elem_type));
    }
    return v;
  }
  size_t sz = meta_scalar_size(t);
  const uint8_t* b = c.raw(sz);
  switch (t) {
    case MetaType::UINT8:   v.i = b[0]; break;
    case MetaType::INT8:    { int8_t x;   std::memcpy(&x, b, 1); v.i = x; break; }
    case MetaType::BOOL:    v.i = b[0] ? 1 : 0; break;
    case MetaType::UINT16:  { uint16_t x; std::memcpy(&x, b, 2); v.i = x; break; }
    case MetaType::INT16:   { int16_t x;  std::memcpy(&x, b, 2); v.i = x; break; }
    case MetaType::UINT32:  { uint32_t x; std::memcpy(&x, b, 4); v.i = x; break; }
    case MetaType::INT32:   { int32_t x;  std::memcpy(&x, b, 4); v.i = x; break; }
    case MetaType::FLOAT32: { float x;    std::memcpy(&x, b, 4); v.f = x; break; }
    case MetaType::UINT64:  { uint64_t x; std::memcpy(&x, b, 8); v.i = static_cast<int64_t>(x); break; }
    case MetaType::INT64:   { int64_t x;  std::memcpy(&x, b, 8); v.i = x; break; }
    case MetaType::FLOAT64: { double x;   std::memcpy(&x, b, 8); v.f = x; break; }
    default: fail("unhandled value type");
  }
  return v;
}

}  // namespace

// ---------------------------------------------------------------- type table

const char* ggml_type_name(GgmlType t) {
  switch (t) {
    case GgmlType::F32:  return "F32";
    case GgmlType::F16:  return "F16";
    case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_0: return "Q5_0";
    case GgmlType::Q5_1: return "Q5_1";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q8_1: return "Q8_1";
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::Q8_K: return "Q8_K";
    case GgmlType::IQ2_XXS: return "IQ2_XXS";
    case GgmlType::IQ2_XS:  return "IQ2_XS";
    case GgmlType::IQ3_XXS: return "IQ3_XXS";
    case GgmlType::IQ1_S:   return "IQ1_S";
    case GgmlType::IQ4_NL:  return "IQ4_NL";
    case GgmlType::IQ3_S:   return "IQ3_S";
    case GgmlType::IQ2_S:   return "IQ2_S";
    case GgmlType::IQ4_XS:  return "IQ4_XS";
    case GgmlType::IQ1_M:   return "IQ1_M";
    case GgmlType::MXFP4:   return "MXFP4";
    case GgmlType::TQ1_0:   return "TQ1_0";
    case GgmlType::TQ2_0:   return "TQ2_0";
    case GgmlType::NVFP4:   return "NVFP4";
    case GgmlType::Q1_0:    return "Q1_0";
    case GgmlType::I8:   return "I8";
    case GgmlType::I16:  return "I16";
    case GgmlType::I32:  return "I32";
    case GgmlType::I64:  return "I64";
    case GgmlType::F64:  return "F64";
    case GgmlType::BF16: return "BF16";
    case GgmlType::Q4_0_ROCMFP4:      return "Q4_0_ROCMFP4";
    case GgmlType::Q4_0_ROCMFP4_FAST: return "Q4_0_ROCMFP4_FAST";
    case GgmlType::Q6_0_ROCMFPX:      return "Q6_0_ROCMFPX";
    case GgmlType::Q8_0_ROCMFPX:      return "Q8_0_ROCMFPX";
    case GgmlType::Q3_0_ROCMFPX:      return "Q3_0_ROCMFPX";
    case GgmlType::Q2_0_ROCMFPX:      return "Q2_0_ROCMFPX";
    default:             return "UNKNOWN";
  }
}

size_t ggml_type_block_weights(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::I8:
    case GgmlType::I16:
    case GgmlType::I32:
    case GgmlType::I64:
    case GgmlType::F64:  return 1;
    // Legacy 32-weight quants.
    case GgmlType::Q4_0:
    case GgmlType::Q4_1:
    case GgmlType::Q5_0:
    case GgmlType::Q5_1:
    case GgmlType::Q8_0:
    case GgmlType::Q8_1: return 32;
    // IQ4_NL is the one IQ type with 32-weight blocks; MXFP4 is 32 too.
    // The ROCmFP family is all 32-weight blocks (QK_ROCMFP4/QK_ROCMFPX = 32).
    case GgmlType::IQ4_NL:
    case GgmlType::MXFP4:
    case GgmlType::Q4_0_ROCMFP4:
    case GgmlType::Q4_0_ROCMFP4_FAST:
    case GgmlType::Q6_0_ROCMFPX:
    case GgmlType::Q8_0_ROCMFPX:
    case GgmlType::Q3_0_ROCMFPX:
    case GgmlType::Q2_0_ROCMFPX: return 32;
    case GgmlType::NVFP4: return 64;
    case GgmlType::Q1_0:  return 128;
    // K-quants and the rest of the IQ family are 256-weight superblocks.
    case GgmlType::Q2_K:
    case GgmlType::Q3_K:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
    case GgmlType::Q8_K:
    case GgmlType::IQ2_XXS:
    case GgmlType::IQ2_XS:
    case GgmlType::IQ3_XXS:
    case GgmlType::IQ1_S:
    case GgmlType::IQ3_S:
    case GgmlType::IQ2_S:
    case GgmlType::IQ4_XS:
    case GgmlType::IQ1_M: return 256;
    default:             fail("unknown ggml type has no block size");
  }
}

// Block byte sizes, transcribed from ggml's type traits.
size_t ggml_type_block_bytes(GgmlType t) {
  switch (t) {
    case GgmlType::F32:  return 4;
    case GgmlType::F16:  return 2;
    case GgmlType::BF16: return 2;
    case GgmlType::I8:   return 1;
    case GgmlType::I16:  return 2;
    case GgmlType::I32:  return 4;
    case GgmlType::I64:  return 8;
    case GgmlType::F64:  return 8;
    case GgmlType::Q4_0: return 2 + 16;         // d + 16 packed nibbles
    case GgmlType::Q4_1: return 2 + 2 + 16;     // d, m, 16 packed nibbles
    case GgmlType::Q5_0: return 2 + 4 + 16;     // d, qh, 16 packed nibbles
    case GgmlType::Q5_1: return 2 + 2 + 4 + 16;
    case GgmlType::Q8_0: return 2 + 32;         // d + 32 int8
    case GgmlType::Q2_K: return 2 + 2 + 16 + 64;
    // Q3_K: hmask[32] + qs[64] + scales[12] + d. An older revision of this
    // table said 81; the C struct (ggml-common.h block_q3_K + static_assert)
    // and gguf-py's GGML_QUANT_SIZES agree on 110. Found 2026-10-10 while
    // adding the Q3_K decoder; no decoder ever ran on the wrong size.
    case GgmlType::Q3_K: return 32 + 64 + 12 + 2;
    case GgmlType::Q4_K: return 2 + 2 + 12 + 128;
    case GgmlType::Q5_K: return 2 + 2 + 12 + 32 + 128;
    case GgmlType::Q6_K: return 2 + 128 + 64 + 16;
    // Q8_K: float d + qs[256] + bsums[16 * int16]. An older revision said
    // 276 (16, not 32, trailing bytes); the C struct static_asserts
    // sizeof(float) + 256 + 16*sizeof(int16_t) = 292. Same finding as Q3_K.
    case GgmlType::Q8_K: return 4 + 256 + 32;
    case GgmlType::Q8_1: return 2 + 2 + 32;
    case GgmlType::TQ1_0: return 48 + 4 + 2;
    case GgmlType::TQ2_0: return 64 + 2;
    case GgmlType::NVFP4: return 4 + 32;
    case GgmlType::Q1_0:  return 2 + 16;
    // The IQ family and MXFP4: sizes taken from gguf-py's GGML_QUANT_SIZES
    // (`python -c "from gguf.constants import GGML_QUANT_SIZES as Q; print(Q)"`),
    // not from memory. Decoders for all of them live in src/loader/dequant.cpp
    // (dequant_iq*, dequant_mxfp4) and are validated bit-exact vs gguf-py by
    // tools/dequant_validate.py (21/21 PASS, 2026-10-10); the sizes exist so the
    // container can be read even where a decoder is missing, and any refusal
    // names the type per tensor (see ggml_type_is_dequantizable).
    case GgmlType::IQ2_XXS: return 66;
    case GgmlType::IQ2_XS:  return 74;
    case GgmlType::IQ3_XXS: return 98;
    case GgmlType::IQ1_S:   return 50;
    case GgmlType::IQ4_NL:  return 18;
    case GgmlType::IQ3_S:   return 110;
    case GgmlType::IQ2_S:   return 82;
    case GgmlType::IQ4_XS:  return 136;
    case GgmlType::IQ1_M:   return 56;
    case GgmlType::MXFP4:   return 17;
    // ROCmFP block bytes: qs payload + UE4M3 scale bytes. 100: 16+2,
    // 101: 16+1, 102: 24+2, 103: 32+1, 104: 12+2, 107: 8+2.
    case GgmlType::Q4_0_ROCMFP4:      return 18;
    case GgmlType::Q4_0_ROCMFP4_FAST: return 17;
    case GgmlType::Q6_0_ROCMFPX:      return 26;
    case GgmlType::Q8_0_ROCMFPX:      return 33;
    case GgmlType::Q3_0_ROCMFPX:      return 14;
    case GgmlType::Q2_0_ROCMFPX:      return 10;
    default:             fail("unknown ggml type has no block bytes");
  }
}

bool ggml_type_is_dequantizable(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::Q4_0:
    case GgmlType::Q4_1:
    case GgmlType::Q5_0:
    case GgmlType::Q5_1:
    case GgmlType::Q8_0:
    case GgmlType::Q8_1:
    case GgmlType::Q1_0:
    case GgmlType::Q2_K:
    case GgmlType::Q3_K:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
    case GgmlType::Q8_K:
    case GgmlType::IQ2_XXS:
    case GgmlType::IQ2_XS:
    case GgmlType::IQ3_XXS:
    case GgmlType::IQ1_S:
    case GgmlType::IQ4_NL:
    case GgmlType::IQ3_S:
    case GgmlType::IQ2_S:
    case GgmlType::IQ4_XS:
    case GgmlType::IQ1_M:
    case GgmlType::TQ1_0:
    case GgmlType::TQ2_0:
    case GgmlType::MXFP4:
    case GgmlType::NVFP4:
    case GgmlType::Q4_0_ROCMFP4:
    case GgmlType::Q4_0_ROCMFP4_FAST:
    case GgmlType::Q6_0_ROCMFPX:
    case GgmlType::Q8_0_ROCMFPX:
    case GgmlType::Q3_0_ROCMFPX:
    case GgmlType::Q2_0_ROCMFPX:
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------- TensorInfo

uint64_t TensorInfo::nelements() const {
  uint64_t n = 1;
  for (uint64_t d : dims) n *= d;
  return n;
}

uint64_t TensorInfo::nbytes() const {
  const size_t bw = ggml_type_block_weights(type);
  const uint64_t ne = nelements();
  if (ne % bw != 0) fail("tensor " + name + " has " + std::to_string(ne) +
                         " elements, not a multiple of its block size " +
                         std::to_string(bw));
  return ne / bw * ggml_type_block_bytes(type);
}

// ---------------------------------------------------------------- FileMap

FileMap::~FileMap() { close(); }

void FileMap::swap(FileMap& o) noexcept {
  std::swap(handle_, o.handle_);
  std::swap(mapping_, o.mapping_);
  std::swap(data_, o.data_);
  std::swap(size_, o.size_);
}

void FileMap::close() {
#ifdef _WIN32
  if (data_)    { UnmapViewOfFile(data_); data_ = nullptr; }
  if (mapping_) { CloseHandle(static_cast<HANDLE>(mapping_)); mapping_ = nullptr; }
  if (handle_)  { CloseHandle(static_cast<HANDLE>(handle_)); handle_ = nullptr; }
#endif
  size_ = 0;
}

FileMap FileMap::open_readonly(const std::string& path) {
  FileMap m;
#ifdef _WIN32
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    fail("cannot open '" + path + "' (error " + std::to_string(GetLastError()) + ")");
  LARGE_INTEGER li;
  if (!GetFileSizeEx(h, &li)) {
    CloseHandle(h);
    fail("cannot size '" + path + "'");
  }
  if (li.QuadPart == 0) {
    CloseHandle(h);
    fail("'" + path + "' is empty");
  }
  HANDLE sec = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!sec) {
    CloseHandle(h);
    fail("cannot create a mapping for '" + path + "'");
  }
  void* view = MapViewOfFile(sec, FILE_MAP_READ, 0, 0, 0);
  if (!view) {
    CloseHandle(sec);
    CloseHandle(h);
    fail("cannot map '" + path + "'");
  }
  m.handle_ = h;
  m.mapping_ = sec;
  m.data_ = static_cast<uint8_t*>(view);
  m.size_ = static_cast<uint64_t>(li.QuadPart);
  return m;
#else
  fail("only the Windows mapping path is implemented in this build");
#endif
}

// ---------------------------------------------------------------- GgufFile

GgufFile GgufFile::open(const std::string& path) {
  GgufFile g;
  g.path_ = path;
  g.map_ = FileMap::open_readonly(path);
  Cursor c{g.map_.data(), g.map_.size(), 0};

  const uint8_t* magic = c.raw(4);
  if (std::memcmp(magic, "GGUF", 4) != 0) fail("'" + path + "' is not a GGUF file");
  g.version_ = c.pod<uint32_t>();
  if (g.version_ < 2 || g.version_ > 3)
    fail("unsupported GGUF version " + std::to_string(g.version_));
  const uint64_t n_tensors = c.pod<uint64_t>();
  const uint64_t n_kv = c.pod<uint64_t>();

  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key = c.str();
    MetaType t = static_cast<MetaType>(c.pod<uint32_t>());
    g.meta_[key] = read_value(c, t);
  }

  g.tensors_.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo t;
    t.name = c.str();
    uint32_t nd = c.pod<uint32_t>();
    if (nd == 0 || nd > 4) fail("tensor '" + t.name + "' has " + std::to_string(nd) + " dims");
    t.dims.resize(nd);
    for (uint32_t d = 0; d < nd; ++d) t.dims[d] = c.pod<uint64_t>();
    t.type = static_cast<GgmlType>(c.pod<uint32_t>());
    t.offset = c.pod<uint64_t>();
    if (t.type != GgmlType::Unknown) {
      // Fail here, while we still know which tensor it was, rather than at the
      // first kernel that tries to read it.
      ggml_type_block_bytes(t.type);
      ggml_type_block_weights(t.type);
    }
    g.by_name_[t.name] = g.tensors_.size();
    g.tensors_.push_back(std::move(t));
  }

  // general.alignment decides where the data section starts. It is 32 in every
  // file seen so far, but it is a declared field and using a constant here
  // would silently misread any file that changes it.
  uint64_t align = 32;
  auto it = g.meta_.find("general.alignment");
  if (it != g.meta_.end() && it->second.i > 0) align = static_cast<uint64_t>(it->second.i);
  g.alignment_ = align;
  g.data_offset_ = align_up(c.at, align);

  // Every tensor must lie inside the file. This is the cheap integrity gate
  // that turns a corrupt or partial download into a refusal at open.
  for (const TensorInfo& t : g.tensors_) {
    uint64_t end = g.data_offset_ + t.offset + t.nbytes();
    if (t.offset > g.map_.size() || end > g.map_.size())
      fail("tensor '" + t.name + "' (" + std::to_string(t.nbytes()) + " bytes at " +
           std::to_string(g.data_offset_ + t.offset) + ") runs past the end of '" +
           path + "' (" + std::to_string(g.map_.size()) + " bytes)");
  }
  return g;
}

const TensorInfo* GgufFile::find(const std::string& name) const {
  auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& GgufFile::require(const std::string& name) const {
  const TensorInfo* t = find(name);
  if (!t) fail("required tensor '" + name + "' is missing");
  return *t;
}

const uint8_t* GgufFile::data_of(const TensorInfo& t) const {
  return map_.data() + data_offset_ + t.offset;
}

bool GgufFile::has(const std::string& key) const { return meta_.count(key) != 0; }

int64_t GgufFile::meta_int(const std::string& key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) fail("required metadata key '" + key + "' is missing");
  if (it->second.type == MetaType::FLOAT32 || it->second.type == MetaType::FLOAT64)
    return static_cast<int64_t>(it->second.f);
  if (it->second.type == MetaType::STRING) fail("metadata '" + key + "' is a string");
  return it->second.i;
}

int64_t GgufFile::meta_int_or(const std::string& key, int64_t fallback) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) return fallback;
  if (it->second.type == MetaType::FLOAT32 || it->second.type == MetaType::FLOAT64)
    return static_cast<int64_t>(it->second.f);
  return it->second.i;
}

double GgufFile::meta_float(const std::string& key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) fail("required metadata key '" + key + "' is missing");
  if (it->second.type == MetaType::FLOAT32 || it->second.type == MetaType::FLOAT64)
    return it->second.f;
  return static_cast<double>(it->second.i);
}

bool GgufFile::meta_bool(const std::string& key) const { return meta_int(key) != 0; }

std::string GgufFile::meta_string(const std::string& key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) fail("required metadata key '" + key + "' is missing");
  if (it->second.type != MetaType::STRING) fail("metadata '" + key + "' is not a string");
  return it->second.s;
}

uint64_t GgufFile::payload_span_bytes() const {
  if (tensors_.empty()) return 0;
  uint64_t lo = UINT64_MAX, hi = 0;
  for (const TensorInfo& t : tensors_) {
    lo = std::min(lo, t.offset);
    hi = std::max(hi, t.offset + t.nbytes());
  }
  return hi - lo;
}

}  // namespace knj
