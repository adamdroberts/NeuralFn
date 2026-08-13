#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace neuralfn::native_train {

inline constexpr std::uint32_t kTokenShardV2Version = 2;
inline constexpr std::uint32_t kTokenShardV2HeaderBytes = 512;
inline constexpr std::uint32_t kTokenShardLittleEndianMarker = 0x01020304U;
inline constexpr std::uint32_t kStructuredSftV1Version = 1;
inline constexpr std::uint32_t kStructuredSftV1HeaderBytes = 512;
inline constexpr std::uint32_t kStructuredPreferenceV1Version = 1;
inline constexpr std::uint32_t kStructuredPreferenceV1HeaderBytes = 512;
inline constexpr std::uint32_t kStructuredPpoPromptV1Version = 1;
inline constexpr std::uint32_t kStructuredPpoPromptV1HeaderBytes = 512;

enum class TokenShardDType : std::uint32_t {
    legacy_uint16_le = 1,
    uint32_le = 2,
};

struct TokenShardFile {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    std::uintmax_t header_uint16 = 0;
    std::uintmax_t tokens = 0;
    std::uintmax_t header_bytes = 0;
    std::uint32_t element_bytes = 2;
    TokenShardDType dtype = TokenShardDType::legacy_uint16_le;
    std::uint32_t tokenizer_vocab_size = 0;
    std::uint32_t max_token_id = 0;
    std::string tokenizer_sha256;
    std::string tokenizer_revision;
    std::string tokenizer_name;
    std::string split;
    std::string objective;
    // Optional process-owned snapshot descriptor used by strict graph-bound
    // trainers.  The owner retains the descriptor; samplers only pread it and
    // verify the captured inode/timestamps around every batch read.
    int stable_fd = -1;
    std::uintmax_t stable_device = 0;
    std::uintmax_t stable_inode = 0;
    std::int64_t stable_mtime_seconds = 0;
    std::int64_t stable_mtime_nanoseconds = 0;
    std::int64_t stable_ctime_seconds = 0;
    std::int64_t stable_ctime_nanoseconds = 0;
};

struct TokenShardDataset {
    std::filesystem::path dataset_path;
    std::vector<TokenShardFile> train_shards;
    std::vector<TokenShardFile> val_shards;
    std::uintmax_t train_tokens = 0;
    std::uintmax_t val_tokens = 0;
};

struct BatchPlan {
    std::int64_t microbatch_tokens = 0;
    std::int64_t grad_accum_steps = 0;
    std::int64_t effective_train_batch_tokens = 0;
    std::int64_t train_sequences = 0;
    std::int64_t train_microbatches = 0;
    std::int64_t train_optimizer_steps_per_epoch = 0;
    std::int64_t val_sequences = 0;
    std::int64_t val_microbatches = 0;
};

struct TokenBatch {
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    std::vector<std::uint16_t> tokens;
    std::vector<std::uint16_t> targets;
};

// Wide native batches are the canonical boundary for vocabularies above
// uint16. The sampler accepts both legacy uint16 and versioned uint32 shards
// and widens the former without changing their on-disk interpretation.
struct TokenBatch32 {
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> targets;
};

// Fixed-width, independently addressable SFT records.  Each payload record is
// four little-endian arrays of `sequence_length` elements, in this order:
// uint32 input_ids, int32 targets, float32 loss_mask, int32 sequence_ids.
// Targets may be -100; all other token IDs are validated against the bound
// tokenizer vocabulary.  sequence_ids make packed-example attention
// boundaries explicit rather than relying on a loss mask to hide leakage.
struct StructuredSftFile {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    std::uint64_t records = 0;
    std::uint32_t sequence_length = 0;
    std::uint32_t tokenizer_vocab_size = 0;
    std::uint32_t pad_token_id = 0;
    std::string tokenizer_sha256;
    std::string chat_template_sha256;
    std::string tokenizer_revision;
    std::string split;
};

struct StructuredSftDataset {
    std::filesystem::path dataset_path;
    std::vector<StructuredSftFile> train_files;
    std::vector<StructuredSftFile> val_files;
    std::uint64_t train_records = 0;
    std::uint64_t val_records = 0;
};

struct StructuredSftBatch {
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    std::vector<std::uint32_t> input_ids;
    std::vector<std::int32_t> targets;
    std::vector<float> loss_mask;
    std::vector<std::int32_t> sequence_ids;
};

class SequentialStructuredSftBatchSampler {
public:
    SequentialStructuredSftBatchSampler(
        std::vector<StructuredSftFile> files,
        std::int64_t batch_size);

    bool next(StructuredSftBatch& out);
    bool seek_batch(std::int64_t batch_index);
    void reset();
    std::int64_t total_batches() const;

private:
    std::vector<StructuredSftFile> files_;
    std::int64_t batch_size_ = 0;
    std::uint32_t sequence_length_ = 0;
    std::size_t file_index_ = 0;
    std::uint64_t local_record_index_ = 0;
    std::vector<std::uint8_t> scratch_;
};

// Fixed-width preference records.  Each record contains a complete chosen SFT
// branch followed by a complete rejected SFT branch; each branch uses the
// exact four-array layout documented by StructuredSftFile.  Keeping masks and
// packed-example boundaries branch-local makes DPO and reward training
// deterministic and prevents prompt/padding tokens from entering sequence
// scores.
struct StructuredPreferenceFile {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    std::uint64_t records = 0;
    std::uint32_t sequence_length = 0;
    std::uint32_t tokenizer_vocab_size = 0;
    std::uint32_t pad_token_id = 0;
    std::string tokenizer_sha256;
    std::string chat_template_sha256;
    std::string tokenizer_revision;
    std::string split;
};

struct StructuredPreferenceDataset {
    std::filesystem::path dataset_path;
    std::vector<StructuredPreferenceFile> train_files;
    std::vector<StructuredPreferenceFile> val_files;
    std::uint64_t train_records = 0;
    std::uint64_t val_records = 0;
};

struct StructuredPreferenceBatch {
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    std::vector<std::uint32_t> chosen_input_ids;
    std::vector<std::int32_t> chosen_targets;
    std::vector<float> chosen_loss_mask;
    std::vector<std::int32_t> chosen_sequence_ids;
    std::vector<std::uint32_t> rejected_input_ids;
    std::vector<std::int32_t> rejected_targets;
    std::vector<float> rejected_loss_mask;
    std::vector<std::int32_t> rejected_sequence_ids;
};

class SequentialStructuredPreferenceBatchSampler {
public:
    SequentialStructuredPreferenceBatchSampler(
        std::vector<StructuredPreferenceFile> files,
        std::int64_t batch_size);

    bool next(StructuredPreferenceBatch& out);
    bool seek_batch(std::int64_t batch_index);
    void reset();
    std::int64_t total_batches() const;

private:
    std::vector<StructuredPreferenceFile> files_;
    std::int64_t batch_size_ = 0;
    std::uint32_t sequence_length_ = 0;
    std::size_t file_index_ = 0;
    std::uint64_t local_record_index_ = 0;
    std::vector<std::uint8_t> scratch_;
};

// Fixed-width online-PPO prompt records. Each record contains uint32 input_ids
// followed by float32 attention_mask, both of sequence_length elements. The
// mask is one non-empty contiguous prefix and trailing tokens are exact pads.
struct StructuredPpoPromptFile {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    std::uint64_t records = 0;
    std::uint32_t sequence_length = 0;
    std::uint32_t tokenizer_vocab_size = 0;
    std::uint32_t pad_token_id = 0;
    std::string tokenizer_sha256;
    std::string chat_template_sha256;
    std::string tokenizer_revision;
    std::string split;
};

struct StructuredPpoPromptDataset {
    std::filesystem::path dataset_path;
    std::vector<StructuredPpoPromptFile> train_files;
    std::vector<StructuredPpoPromptFile> val_files;
    std::uint64_t train_records = 0;
    std::uint64_t val_records = 0;
};

struct StructuredPpoPromptBatch {
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    std::vector<std::uint32_t> input_ids;
    std::vector<float> attention_mask;
};

class SequentialStructuredPpoPromptBatchSampler {
public:
    SequentialStructuredPpoPromptBatchSampler(
        std::vector<StructuredPpoPromptFile> files,
        std::int64_t batch_size);

    bool next(StructuredPpoPromptBatch& out);
    bool seek_batch(std::int64_t batch_index);
    void reset();
    std::int64_t total_batches() const;

private:
    std::vector<StructuredPpoPromptFile> files_;
    std::int64_t batch_size_ = 0;
    std::uint32_t sequence_length_ = 0;
    std::size_t file_index_ = 0;
    std::uint64_t local_record_index_ = 0;
    std::vector<std::uint8_t> scratch_;
};

struct ByteShardDataset {
    std::filesystem::path dataset_path;
    std::vector<TokenShardFile> train_shards;
    std::vector<TokenShardFile> val_shards;
    std::uintmax_t train_tokens = 0;
    std::uintmax_t val_tokens = 0;
};

struct ByteBatch {
    std::int64_t batch_size = 0;
    std::int64_t seq_len = 0;
    std::vector<std::uint8_t> tokens;
    std::vector<std::uint8_t> targets;
};

class SequentialTokenBatchSampler {
public:
    SequentialTokenBatchSampler(std::vector<TokenShardFile> shards, std::int64_t seq_len, std::int64_t batch_size);

    bool next(TokenBatch& out);
    bool next_into(std::uint16_t* tokens, std::uint16_t* targets, std::int64_t token_capacity);
    bool seek_batch(std::int64_t batch_index);
    void reset();
    std::int64_t total_batches() const;

private:
    std::vector<TokenShardFile> shards_;
    std::int64_t seq_len_ = 0;
    std::int64_t batch_size_ = 0;
    std::size_t shard_index_ = 0;
    std::uintmax_t local_chunk_index_ = 0;
    std::vector<std::uint16_t> scratch_;
};

class SequentialTokenBatchSampler32 {
public:
    SequentialTokenBatchSampler32(
        std::vector<TokenShardFile> shards,
        std::int64_t seq_len,
        std::int64_t batch_size);

    bool next(TokenBatch32& out);
    bool next_into(std::uint32_t* tokens, std::uint32_t* targets, std::int64_t token_capacity);
    bool seek_batch(std::int64_t batch_index);
    void reset();
    std::int64_t total_batches() const;

private:
    std::vector<TokenShardFile> shards_;
    std::int64_t seq_len_ = 0;
    std::int64_t batch_size_ = 0;
    std::size_t shard_index_ = 0;
    std::uintmax_t local_chunk_index_ = 0;
    std::vector<std::uint8_t> scratch_;
};

class SequentialByteBatchSampler {
public:
    SequentialByteBatchSampler(std::vector<TokenShardFile> shards, std::int64_t seq_len, std::int64_t batch_size);

    bool next(ByteBatch& out);
    bool next_into(std::uint8_t* tokens, std::uint8_t* targets, std::int64_t token_capacity);
    bool seek_batch(std::int64_t batch_index);
    void reset();
    std::int64_t total_batches() const;

private:
    std::vector<TokenShardFile> shards_;
    std::int64_t seq_len_ = 0;
    std::int64_t batch_size_ = 0;
    std::size_t shard_index_ = 0;
    std::uintmax_t local_chunk_index_ = 0;
    std::vector<std::uint8_t> scratch_;
};

std::filesystem::path native_datasets_dir();
std::filesystem::path resolve_dataset_path(const std::string& alias_or_path);
TokenShardDataset resolve_token_shards(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation = true);
StructuredSftDataset resolve_structured_sft_records(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation = true);
StructuredPreferenceDataset resolve_structured_preference_records(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation = true);
StructuredPpoPromptDataset resolve_structured_ppo_prompt_records(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation = true);
ByteShardDataset resolve_byte_shards(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation = true);
BatchPlan build_batch_plan(
    const TokenShardDataset& dataset,
    std::int64_t seq_len,
    std::int64_t batch_size,
    std::int64_t train_batch_tokens);
BatchPlan build_batch_plan(
    const ByteShardDataset& dataset,
    std::int64_t seq_len,
    std::int64_t batch_size,
    std::int64_t train_batch_tokens);
std::string token_shard_dataset_json(const TokenShardDataset& dataset, const BatchPlan* batch_plan = nullptr);
std::string token_batch_json(const TokenBatch& batch, std::size_t max_items = 16);
std::string token_batch_json(const TokenBatch32& batch, std::size_t max_items = 16);
std::string byte_shard_dataset_json(const ByteShardDataset& dataset, const BatchPlan* batch_plan = nullptr);
std::string byte_batch_json(const ByteBatch& batch, std::size_t max_items = 16);

}  // namespace neuralfn::native_train
