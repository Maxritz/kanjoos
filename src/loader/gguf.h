// src/loader/gguf.h -- GGUF container reader (C4 / Phase 1 substrate).
//
// Reads the container only: header, metadata KV, tensor table, and a read-only
// mapping of the data section. It deliberately knows nothing about Qwen or MoE
// -- model semantics live in src/model/. That split is what lets the same
// reader open a 40 GiB streamed model without touching its expert payload.
//
// Why mmap and not fread: the engine's cold-start contract (BUILD-OUTLINE
// section 5) is "a GGUF opens without reading its expert payload". A read-all
// loader cannot honour that and would also double the model's RSS. The mapping
// is read-only and page-faulted on demand.
#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace knj {

// ggml tensor element types we may encounter in a GGUF tensor table.
// Values are the on-disk GGML_TYPE_* ids and must not be renumbered.
enum class GgmlType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
  Unknown = 0xFFFFFFFFu,
};

const char*  ggml_type_name(GgmlType t);
// Bytes occupied by one quantization block of this type.
size_t       ggml_type_block_bytes(GgmlType t);
// Weights per quantization block (1 for unquantized types).
size_t       ggml_type_block_weights(GgmlType t);
// True when dequant.cpp can turn this type into f32.
bool         ggml_type_is_dequantizable(GgmlType t);

// Metadata value. GGUF spells arrays as (element type, count); we keep the
// element type so a caller can tell token_type (int32 array) from tokens
// (string array) without a second pass over the file.
enum class MetaType : uint32_t {
  UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
  FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
  FLOAT64 = 12,
};

struct MetaValue {
  MetaType type = MetaType::UINT32;
  MetaType elem_type = MetaType::UINT32;  // meaningful only when type == ARRAY
  int64_t  i = 0;                          // all integer types + BOOL
  double   f = 0.0;                        // FLOAT32 / FLOAT64
  std::string s;                           // STRING
  std::vector<MetaValue> arr;              // ARRAY

  bool is_array() const { return type == MetaType::ARRAY; }
  size_t size() const { return arr.size(); }
};

// One entry of the GGUF tensor table.
//
// `dims` keeps ggml's "ne" order: dims[0] varies fastest in memory, which for
// a linear layer is the *input* dimension. Everything downstream depends on
// that convention, so it is stated here rather than re-derived per kernel.
struct TensorInfo {
  std::string           name;
  std::vector<uint64_t> dims;
  GgmlType              type = GgmlType::Unknown;
  uint64_t              offset = 0;  // relative to the data section start

  uint64_t nelements() const;
  // Byte length of the tensor payload, derived from dims + type. Validated
  // against the file length at load so a truncated file is refused, not read.
  uint64_t nbytes() const;
  size_t   ndims() const { return dims.size(); }
};

// Read-only file mapping. Owns the OS handles; movable, not copyable.
class FileMap {
 public:
  FileMap() = default;
  ~FileMap();
  FileMap(const FileMap&) = delete;
  FileMap& operator=(const FileMap&) = delete;
  FileMap(FileMap&& o) noexcept { swap(o); }
  FileMap& operator=(FileMap&& o) noexcept { swap(o); return *this; }

  // Throws std::runtime_error on failure.
  static FileMap open_readonly(const std::string& path);

  const uint8_t* data() const { return data_; }
  uint64_t size() const { return size_; }

 private:
  void swap(FileMap& o) noexcept;
  void close();

  void*    handle_ = nullptr;  // Windows: file handle
  void*    mapping_ = nullptr; // Windows: section handle
  uint8_t* data_ = nullptr;
  uint64_t size_ = 0;
};

class GgufFile {
 public:
  // An empty container, used when an owner wants to construct first and open
  // later (Model::open does). It is not a valid file: find() returns null and
  // the metadata helpers throw until open() has succeeded.
  GgufFile() = default;

  // Throws std::runtime_error with a readable reason on a malformed file.
  static GgufFile open(const std::string& path);

  const std::string& path() const { return path_; }
  uint32_t version() const { return version_; }
  uint64_t alignment() const { return alignment_; }
  uint64_t data_offset() const { return data_offset_; }
  uint64_t file_size() const { return map_.size(); }

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const std::map<std::string, MetaValue>& metadata() const { return meta_; }

  const TensorInfo* find(const std::string& name) const;
  const TensorInfo& require(const std::string& name) const;  // throws if absent

  // Base pointer of the tensor's data. No copy, no dequant.
  const uint8_t* data_of(const TensorInfo& t) const;

  // Metadata helpers. Throw if the key is missing or has the wrong shape, so a
  // model whose metadata disagrees with what the engine expects fails loudly at
  // load instead of silently computing with a default.
  int64_t     meta_int(const std::string& key) const;
  double      meta_float(const std::string& key) const;
  bool        meta_bool(const std::string& key) const;
  std::string meta_string(const std::string& key) const;
  int64_t     meta_int_or(const std::string& key, int64_t fallback) const;
  bool        has(const std::string& key) const;

  // Bytes that the tensor table's payloads occupy, and the sum of declared
  // tensor sizes. The two are compared at open: an out-of-range tensor is a
  // corrupt file, and this is where it is caught.
  uint64_t payload_span_bytes() const;

 private:
  std::string path_;
  FileMap     map_;
  uint32_t    version_ = 0;
  uint64_t    alignment_ = 32;
  uint64_t    data_offset_ = 0;
  std::map<std::string, MetaValue> meta_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, size_t> by_name_;
};

}  // namespace knj
