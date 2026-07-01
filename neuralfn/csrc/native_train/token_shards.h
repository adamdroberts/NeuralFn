#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace neuralfn::native_train {

struct TokenShardFile {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    std::uintmax_t header_uint16 = 0;
    std::uintmax_t tokens = 0;
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

class SequentialByteBatchSampler {
public:
    SequentialByteBatchSampler(std::vector<TokenShardFile> shards, std::int64_t seq_len, std::int64_t batch_size);

    bool next(ByteBatch& out);
    bool next_into(std::uint8_t* tokens, std::uint8_t* targets, std::int64_t token_capacity);
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
std::string byte_shard_dataset_json(const ByteShardDataset& dataset, const BatchPlan* batch_plan = nullptr);
std::string byte_batch_json(const ByteBatch& batch, std::size_t max_items = 16);

}  // namespace neuralfn::native_train
