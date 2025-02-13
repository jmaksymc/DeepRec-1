#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_

#include <pthread.h>
#include <bitset>
#include <atomic>
#include <memory>

#include "tensorflow/core/framework/typed_allocator.h"
#include "leveldb/db.h"
#include "leveldb/comparator.h"

namespace tensorflow {

enum class LayoutType {
  LIGHT,
  NORMAL,
  LEVELDB,
};

namespace {
constexpr int COLUMN_BITSET_BYTES = 5;
constexpr int COLUMN_BITSET_SIZE = COLUMN_BITSET_BYTES * sizeof(char);

struct MetaHeader {
  unsigned char embed_num;
  unsigned char value_type;
  unsigned char header_size;
  unsigned char column_bitset[COLUMN_BITSET_BYTES];

  static const int kEmbeddingNumStartIndex = 0;
  static const int kValueTypeStartIndex =
      kEmbeddingNumStartIndex + sizeof(char);
  static const int kHeaderSizeStartIndex =
      kValueTypeStartIndex + sizeof(char);
  static const int kColumnBitsetIndex =
      kHeaderSizeStartIndex + sizeof(char);

  inline unsigned int GetEmbeddingNum() {
    return (unsigned int) embed_num;
  }

  inline void SetEmbeddingNum(size_t s) {
    embed_num = (unsigned char)s;
  }

  inline std::bitset<COLUMN_BITSET_SIZE> GetColumnBitset() {
    unsigned long meta = ((unsigned long*)this)[0];
    std::bitset<COLUMN_BITSET_SIZE> bs(meta >> (8 * kColumnBitsetIndex));
    return bs;
  }

  inline void SetColumnBitset(const std::bitset<COLUMN_BITSET_SIZE>& bs,
      unsigned int embnum) {
    ((unsigned long*)(this))[0] =
      (bs.to_ulong() << (8 * kColumnBitsetIndex)) |
      (header_size << (8 * kHeaderSizeStartIndex)) |
      (value_type << (8 * kValueTypeStartIndex)) |
      (embnum << (8 * kEmbeddingNumStartIndex));
  }

  inline unsigned int GetHeaderSize() {
    return (unsigned int) header_size;
  }

  inline void SetHeaderSize(size_t size) {
    header_size = (unsigned char)size;
  }

  inline void SetLayoutType(LayoutType vt) {
    value_type = (unsigned char)vt;
  }

  inline LayoutType GetLayoutType() {
    return (LayoutType)value_type;
  }
};

struct LightHeader {
/*__________________________________________________________________________________________
 |           |          |          |               |    embedding     |       slot       |
 | number of | valueptr |  header  | each bit a V* |        V*        |        V*        |
 | embedding | type     |   size   |    1 valid    | actually pointer | actually pointer |...
 |  columns  |          |          |   0 no-valid  |    by alloctor   |    by alloctor   |
 |  (8 bits) | (8 bits) | (8 bits) |   (40 bits)   |     (8 bytes)    |     (8 bytes)    |
 --------------------------------------------------------------------------------------------
*/
  MetaHeader meta;
  LightHeader() {
    memset(this, 0, sizeof(LightHeader));
    meta.SetLayoutType(LayoutType::LIGHT);
    meta.SetHeaderSize(sizeof(LightHeader) / sizeof(int64));
  }
};

struct NormalHeader {
/*_________________________________________________________________________________________________________________________
  |           |          |          |               |             |               |    embedding     |       slot       |
  | number of | valueptr |  header  | each bit a V* | global step | freq counter  |        V*        |        V*        |
  | embedding | type     |   size   |    1 valid    |             |               | actually pointer | actually pointer |...
  |  columns  |          |          |   0 no-valid  |    int64    |     int64     |    by alloctor   |    by alloctor   |
  |  (8 bits) | (8 bits) | (8 bits) |   (40 bits)   |  (8 bytes)  |   (8 bytes)   |     (8 bytes)    |     (8 bytes)    |
  --------------------------------------------------------------------------------------------------------------------------
 */
  MetaHeader meta;
  int64 global_step;
  int64 freq_counter;

  NormalHeader() {
    memset(this, 0, sizeof(NormalHeader));
    meta.SetLayoutType(LayoutType::NORMAL);
    meta.SetHeaderSize(sizeof(NormalHeader) / sizeof(int64));
  }

  inline int64 GetGlobalStep() {
    return global_step;
  }

  inline void SetGlobalStep(int64 gs) {
    global_step = gs;
  }

  inline int64 GetFreqCounter() {
    return freq_counter;
  }

  inline void SetFreqCounter(int64 fc) {
    freq_counter = fc;
  }

  inline void AddFreq() {
    __sync_bool_compare_and_swap(&freq_counter,
        freq_counter, freq_counter + 1);
  }

  inline void AddFreq(int64 count) {
    __sync_bool_compare_and_swap(&freq_counter,
        freq_counter, freq_counter + count);
  }
};
} // namespace

template <class V>
class ValuePtr {
 public:
  ~ValuePtr() {}

  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index) {
    MetaHeader* meta = (MetaHeader*)ptr_;
    unsigned int embnum = (unsigned int)meta->embed_num;
    auto metadata = meta->GetColumnBitset();

    if (!metadata.test(emb_index)) {
      while(flag_.test_and_set(std::memory_order_acquire));
      if (metadata.test(emb_index))
        return ((V**)((int64*)ptr_ + (unsigned int)meta->header_size))[emb_index];
      embnum++ ;
      int64 alloc_value_len = value_len;
      V* tensor_val = TypedAllocator::Allocate<V>(allocator, alloc_value_len, AllocationAttributes());
      memcpy(tensor_val, default_v, sizeof(V) * value_len);
      ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[emb_index]  = tensor_val;

      metadata.set(emb_index);
      // NOTE:if we use ((unsigned long*)((char*)ptr_ + 1))[0] = metadata.to_ulong();
      // the ptr_ will be occaionally  modified from 0x7f18700912a0 to 0x700912a0
      // must use  ((V**)ptr_ + 1 + 1)[emb_index] = tensor_val;  to avoid
      meta->SetColumnBitset(metadata, embnum);
      flag_.clear(std::memory_order_release);
      return tensor_val;
    } else {
      return ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[emb_index];
    }
  }

  // simple getter for V* and version
  virtual V* GetValue(int emb_index, int64 value_len) {
    MetaHeader* meta = (MetaHeader*)ptr_;
    auto metadata = meta->GetColumnBitset();
    if (metadata.test(emb_index)) {
      return ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[emb_index];
    } else {
      return nullptr;
    }
  }

  virtual void Commit(int64 value_len, const V* v, int emb_index) {}

  virtual void Free(const V* v) {}

  virtual void Destroy(Allocator* allocator, int64 value_len) {
    MetaHeader* meta = (MetaHeader*)ptr_;
    unsigned int embnum = (unsigned int)meta->embed_num;
    auto metadata = meta->GetColumnBitset();
    for (int i = 0; i< embnum; i++) {
      if (metadata.test(i)) {
        V* val = ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[i];
        if (val != nullptr) {
          TypedAllocator::Deallocate(allocator, val, value_len);
        }
      }
    }
  }

  // Global Step
  virtual int64 GetStep() {
    LOG(FATAL) << "Unsupport GlobalStep in subclass of ValuePtrBase";
  }

  virtual void SetStep(int64 gs) {
    LOG(FATAL) << "Unsupport GlobalStep in subclass of ValuePtrBase";
  }

  // Frequency Counter
  virtual int64 GetFreq() {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void SetFreq(int64 freq) {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void AddFreq() {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void AddFreq(int64 count) {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

 protected:
  void* ptr_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

template <class V>
class LightValuePtr : public ValuePtr<V> {
 public:
  LightValuePtr(size_t size) {
    this->ptr_ = (void*) malloc(sizeof(LightHeader) + sizeof(int64) * size);
    memset(this->ptr_ + sizeof(LightHeader), 0, sizeof(int64) * size);
    new ((char*)this->ptr_) LightHeader();
  }

  ~LightValuePtr() {
    free(this->ptr_);
  }
};

template <class V>
class NormalValuePtr : public ValuePtr<V> {
 public:
  NormalValuePtr(size_t size) {
    this->ptr_ = (void*) malloc(sizeof(NormalHeader) + sizeof(int64) * size);
    memset(this->ptr_ + sizeof(NormalHeader), 0, sizeof(int64) * size);
    new ((char*)this->ptr_) NormalHeader();
  }

  ~NormalValuePtr() {
    free(this->ptr_);
  }

  int64 GetStep() {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->GetGlobalStep();
  }

  void SetStep(int64 gs) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    ((NormalHeader*)this->ptr_)->SetGlobalStep(gs);
  }

  int64 GetFreq() {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->GetFreqCounter();
  }

  void SetFreq(int64 freq) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    ((NormalHeader*)this->ptr_)->SetFreqCounter(freq);
  }

  void AddFreq() {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->AddFreq();
  }

  void AddFreq(int64 count) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->AddFreq(count);
  }
};

template <class V>
class DBValuePtr : public NormalValuePtr<V> {
 public:
  DBValuePtr(size_t size, leveldb::DB* level_db) : NormalValuePtr<V>(size),
    level_db_(level_db) {
  }

  ~DBValuePtr() {
  }

  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    unsigned int embnum = (unsigned int)meta->embed_num;
    auto metadata = meta->GetColumnBitset();

    if (!metadata.test(emb_index)) {
      while(this->flag_.test_and_set(std::memory_order_acquire));
      if (metadata.test(emb_index)) {
        V* key = &((V*)((int64*)this->ptr_ + meta->GetHeaderSize()))[emb_index];
        leveldb::Slice db_key((char*)(&key), sizeof(void*));
        std::string value;
        leveldb::ReadOptions options;

        leveldb::Status st = level_db_->Get(options, db_key, &value);
        if (!st.ok()){
          LOG(FATAL) << "Fail to Get leveldb: " << level_db_ << ", key: " << &this->ptr_<< ", index: " << emb_index << ", msg: " << st.ToString();
        }
        V* tensor_val = (V*)malloc(value_len * sizeof(V));
        memcpy(tensor_val, value.data(), value_len*sizeof(V));
        return tensor_val;
      }
      embnum++ ;
      int64 alloc_value_len = value_len;
      V* tensor_val = (V*)malloc(value_len * sizeof(V));
      memcpy(tensor_val, default_v, sizeof(V) * value_len);

      V* key = &((V*)((int64*)this->ptr_ + meta->GetHeaderSize()))[emb_index];
      leveldb::Slice db_key((char*)(&key), sizeof(void*));
      leveldb::Slice db_value((char*)default_v, value_len*sizeof(V));

      leveldb::WriteOptions options;
      leveldb::Status st = level_db_->Put(options, db_key, db_value);
      if (!st.ok()) {
        LOG(FATAL) << "Fail to Put leveldb: "  <<&this->ptr_ << ", index: "<<emb_index <<", msg: "<< st.ToString();
      }

      metadata.set(emb_index);
      // NOTE:if we use ((unsigned long*)((char*)ptr_ + 1))[0] = metadata.to_ulong();
      // the ptr_ will be occaionally  modified from 0x7f18700912a0 to 0x700912a0
      // must use  ((V**)ptr_ + 1 + 1)[emb_index] = tensor_val;  to avoid
      meta->SetColumnBitset(metadata, embnum);
      this->flag_.clear(std::memory_order_release);
      return tensor_val;
    } else {
      V* key = &((V*)((int64*)this->ptr_ + meta->GetHeaderSize()))[emb_index];
      leveldb::Slice db_key((char*)(&key), sizeof(void*));
      std::string value;
      leveldb::ReadOptions options;

      leveldb::Status st = level_db_->Get(options, db_key, &value);
      if (!st.ok()){
        LOG(FATAL) << "Fail to Get leveldb: " << level_db_ << ", key: " << &this->ptr_<< ", index: " << emb_index << ", msg: " << st.ToString();
      }
      V* tensor_val = (V*)malloc(value_len * sizeof(V));
      memcpy(tensor_val, value.data(), value_len*sizeof(V));
      return tensor_val;
    }
  }

  virtual void Commit(int64 value_len, const V* v, int emb_index) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;

    V* key = &((V*)((int64*)this->ptr_ + meta->GetHeaderSize()))[emb_index];
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    leveldb::Slice db_value((char*)v, value_len * sizeof(V));

    leveldb::WriteOptions options;
    leveldb::Status st = level_db_->Put(options, db_key, db_value);
    if (!st.ok()) {
      LOG(FATAL) << "Fail to Put leveldb: "  << &this->ptr_<< ", index: "<<emb_index <<", msg: "<< st.ToString();
    }
    Free(v);
  }

  virtual void Free(const V* v) {
    free((void*)v);
  }

  virtual V* GetValue(int emb_index, int64 value_len) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    auto metadata = meta->GetColumnBitset();
    if (metadata.test(emb_index)) {
      V* key = &((V*)((int64*)this->ptr_ + meta->GetHeaderSize()))[emb_index];
      leveldb::Slice db_key((char*)(&key), sizeof(void*));
      std::string value;
      leveldb::ReadOptions options;

      leveldb::Status st = level_db_->Get(options, db_key, &value);
      if (!st.ok()){
        LOG(FATAL) << "Fail to Get leveldb: " << level_db_ << ", key: " << &this->ptr_<< ", index: " << emb_index << ", msg: " << st.ToString();
      }
      V* tensor_val = (V*)malloc(value_len * sizeof(V));
      memcpy(tensor_val, value.data() + value_len*sizeof(V) * emb_index, value_len*sizeof(V));
      return tensor_val;
    } else {
      return nullptr;
    }
  }

  virtual void Destroy(int64 value_len) {}

 private:
  leveldb::DB* level_db_;

};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_
