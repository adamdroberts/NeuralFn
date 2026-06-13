#include "token_shards.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace neuralfn::native_train {
namespace {

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

fs::path home_dir() {
    std::string home = env_or_empty("HOME");
    if (home.empty()) {
        return fs::current_path();
    }
    return fs::path(home);
}

bool has_prefix_and_bin_extension(const fs::path& path, const std::string& prefix) {
    const std::string name = path.filename().string();
    return name.rfind(prefix, 0) == 0 && path.extension() == ".bin";
}

bool has_name_and_bin_extension(const fs::path& path, const std::string& stem) {
    return path.stem() == stem && path.extension() == ".bin";
}

std::uintmax_t shard_header_offset_uint16(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open token shard: " + path.string());
    }
    unsigned char magic[4] = {0, 0, 0, 0};
    input.read(reinterpret_cast<char*>(magic), 4);
    if (input.gcount() == 4 && magic[0] == 0x88 && magic[1] == 0xd8 && magic[2] == 0x34 && magic[3] == 0x01) {
        return 512;
    }
    return 0;
}

TokenShardFile read_shard_file(const fs::path& path) {
    const std::uintmax_t bytes = fs::file_size(path);
    if ((bytes % 2U) != 0U) {
        throw std::runtime_error("uint16 token shard has odd byte size: " + path.string());
    }
    const std::uintmax_t header_uint16 = shard_header_offset_uint16(path);
    const std::uintmax_t raw_tokens = bytes / 2U;
    if (header_uint16 > raw_tokens) {
        throw std::runtime_error("uint16 token shard header is larger than file: " + path.string());
    }
    return TokenShardFile{
        .path = path,
        .bytes = bytes,
        .header_uint16 = header_uint16,
        .tokens = raw_tokens - header_uint16,
    };
}

std::vector<TokenShardFile> sorted_shards(const fs::path& dataset_path, const std::vector<std::string>& prefixes) {
    std::vector<TokenShardFile> shards;
    if (!fs::is_directory(dataset_path)) {
        return shards;
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(dataset_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        bool matched = false;
        for (const std::string& prefix : prefixes) {
            if (has_prefix_and_bin_extension(entry.path(), prefix)) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            continue;
        }
        shards.push_back(read_shard_file(entry.path()));
    }
    std::sort(shards.begin(), shards.end(), [](const TokenShardFile& lhs, const TokenShardFile& rhs) {
        return lhs.path < rhs.path;
    });
    return shards;
}

std::vector<TokenShardFile> sorted_shards(const fs::path& dataset_path, const std::string& prefix) {
    return sorted_shards(dataset_path, std::vector<std::string>{prefix});
}

std::vector<TokenShardFile> named_shards(const fs::path& dataset_path, const std::vector<std::string>& stems) {
    std::vector<TokenShardFile> shards;
    if (!fs::is_directory(dataset_path)) {
        return shards;
    }
    for (const std::string& stem : stems) {
        const fs::path candidate = dataset_path / (stem + ".bin");
        if (fs::is_regular_file(candidate) && has_name_and_bin_extension(candidate, stem)) {
            shards.push_back(read_shard_file(candidate));
        }
    }
    return shards;
}

bool directory_has_matching_bin(const fs::path& dataset_path, const std::vector<std::string>& prefixes, const std::vector<std::string>& stems) {
    if (!fs::is_directory(dataset_path)) {
        return false;
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(dataset_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".bin") {
            continue;
        }
        for (const std::string& prefix : prefixes) {
            if (has_prefix_and_bin_extension(entry.path(), prefix)) {
                return true;
            }
        }
        for (const std::string& stem : stems) {
            if (has_name_and_bin_extension(entry.path(), stem)) {
                return true;
            }
        }
    }
    return false;
}

bool directory_has_native_token_bins(const fs::path& dataset_path) {
    return directory_has_matching_bin(
               dataset_path,
               {"fineweb_train_"},
               {"TinyStories_train", "TinyStoriesV2-GPT4_train"}) &&
           directory_has_matching_bin(
               dataset_path,
               {"fineweb_val_"},
               {"TinyStories_val", "TinyStories_valid", "TinyStoriesV2-GPT4_val", "TinyStoriesV2-GPT4_valid"});
}

fs::path inferred_validation_path(const fs::path& train_path) {
    const fs::path parent = train_path.parent_path();
    const std::string stem = train_path.stem().string();
    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"_train", "_val"},
        {"_train", "_valid"},
        {"-train", "-val"},
        {"-train", "-valid"},
        {"train", "val"},
        {"train", "valid"},
    };
    for (const auto& replacement : replacements) {
        const std::string& from = replacement.first;
        const std::string& to = replacement.second;
        const std::size_t pos = stem.rfind(from);
        if (pos == std::string::npos) {
            continue;
        }
        std::string val_stem = stem;
        val_stem.replace(pos, from.size(), to);
        const fs::path candidate = parent / (val_stem + train_path.extension().string());
        if (fs::is_regular_file(candidate)) {
            return candidate;
        }
    }
    return {};
}

bool is_tinystories_alias(const std::string& alias_or_path) {
    return alias_or_path == "tinystories" ||
           alias_or_path == "roneneldan__TinyStories__TinyStoriesV2-GPT4";
}

fs::path llm_kittens_tinystories_dir() {
    const std::string override = env_or_empty("NFN_LLM_KITTENS_TINYSTORIES_DIR");
    if (!override.empty()) {
        return fs::path(override);
    }
    return fs::path("/mnt/disk2/dev/open-source/llm.kittens/dev/data/tinystories");
}

std::uintmax_t sum_tokens(const std::vector<TokenShardFile>& shards) {
    std::uintmax_t total = 0;
    for (const TokenShardFile& shard : shards) {
        total += shard.tokens;
    }
    return total;
}

std::uintmax_t sum_sequences(const std::vector<TokenShardFile>& shards, std::int64_t seq_len) {
    std::uintmax_t total = 0;
    const std::uintmax_t seq = static_cast<std::uintmax_t>(seq_len);
    for (const TokenShardFile& shard : shards) {
        total += shard.tokens > 0 ? (shard.tokens - 1U) / seq : 0U;
    }
    return total;
}

std::int64_t checked_positive(std::int64_t value, const char* name) {
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return value;
}

std::int64_t ceil_div(std::int64_t lhs, std::int64_t rhs) {
    return (lhs + rhs - 1) / rhs;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
        }
    }
    return out.str();
}

void append_shards_json(std::ostringstream& out, const std::vector<TokenShardFile>& shards) {
    out << "[";
    for (std::size_t i = 0; i < shards.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "{\"path\": \"" << json_escape(shards[i].path.string()) << "\", "
            << "\"bytes\": " << shards[i].bytes << ", "
            << "\"header_uint16\": " << shards[i].header_uint16 << ", "
            << "\"tokens\": " << shards[i].tokens << "}";
    }
    out << "]";
}

void append_contiguous_chunks_into(
    const TokenShardFile& shard,
    std::uintmax_t local_chunk_index,
    std::uintmax_t chunk_count,
    std::int64_t seq_len,
    std::uint16_t* tokens,
    std::uint16_t* targets,
    std::size_t offset,
    std::vector<std::uint16_t>& scratch) {
    if (chunk_count == 0) {
        return;
    }
    const std::uintmax_t start = shard.header_uint16 + local_chunk_index * static_cast<std::uintmax_t>(seq_len);
    const std::uintmax_t values = chunk_count * static_cast<std::uintmax_t>(seq_len) + 1U;
    scratch.resize(static_cast<std::size_t>(values));
    std::ifstream input(shard.path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open token shard: " + shard.path.string());
    }
    input.seekg(static_cast<std::streamoff>(start * 2U), std::ios::beg);
    input.read(reinterpret_cast<char*>(scratch.data()), static_cast<std::streamsize>(values * 2U));
    if (input.gcount() != static_cast<std::streamsize>(values * 2U)) {
        throw std::runtime_error("short read from token shard: " + shard.path.string());
    }
    const std::size_t token_count = static_cast<std::size_t>(chunk_count * static_cast<std::uintmax_t>(seq_len));
    std::memcpy(tokens + offset, scratch.data(), token_count * sizeof(std::uint16_t));
    std::memcpy(targets + offset, scratch.data() + 1, token_count * sizeof(std::uint16_t));
}

}  // namespace

SequentialTokenBatchSampler::SequentialTokenBatchSampler(
    std::vector<TokenShardFile> shards,
    std::int64_t seq_len,
    std::int64_t batch_size)
    : shards_(std::move(shards)),
      seq_len_(checked_positive(seq_len, "seq_len")),
      batch_size_(checked_positive(batch_size, "batch_size")) {}

bool SequentialTokenBatchSampler::next(TokenBatch& out) {
    out.batch_size = batch_size_;
    out.seq_len = seq_len_;
    const std::int64_t total = batch_size_ * seq_len_;
    out.tokens.resize(static_cast<std::size_t>(total));
    out.targets.resize(static_cast<std::size_t>(total));
    if (!next_into(out.tokens.data(), out.targets.data(), total)) {
        out.tokens.clear();
        out.targets.clear();
        return false;
    }
    return true;
}

bool SequentialTokenBatchSampler::next_into(
    std::uint16_t* tokens,
    std::uint16_t* targets,
    std::int64_t token_capacity) {
    if (tokens == nullptr || targets == nullptr) {
        throw std::runtime_error("token batch destination pointers must be non-null");
    }
    const std::int64_t total = batch_size_ * seq_len_;
    if (token_capacity < total) {
        throw std::runtime_error("token batch destination capacity is smaller than batch_size * seq_len");
    }

    std::size_t produced = 0;
    while (static_cast<std::int64_t>(produced) < total) {
        if (shard_index_ >= shards_.size()) {
            break;
        }
        const TokenShardFile& shard = shards_[shard_index_];
        const std::uintmax_t chunk_count = shard.tokens > 0
            ? (shard.tokens - 1U) / static_cast<std::uintmax_t>(seq_len_)
            : 0U;
        if (local_chunk_index_ >= chunk_count) {
            shard_index_ += 1;
            local_chunk_index_ = 0;
            continue;
        }
        const std::uintmax_t remaining_batch_chunks = static_cast<std::uintmax_t>(
            (total - static_cast<std::int64_t>(produced)) / seq_len_);
        const std::uintmax_t remaining_shard_chunks = chunk_count - local_chunk_index_;
        const std::uintmax_t chunks_to_read = std::min(remaining_batch_chunks, remaining_shard_chunks);
        append_contiguous_chunks_into(
            shard,
            local_chunk_index_,
            chunks_to_read,
            seq_len_,
            tokens,
            targets,
            produced,
            scratch_);
        local_chunk_index_ += chunks_to_read;
        produced += static_cast<std::size_t>(chunks_to_read * static_cast<std::uintmax_t>(seq_len_));
    }

    return static_cast<std::int64_t>(produced) == total;
}

void SequentialTokenBatchSampler::reset() {
    shard_index_ = 0;
    local_chunk_index_ = 0;
}

std::int64_t SequentialTokenBatchSampler::total_batches() const {
    std::uintmax_t chunks = 0;
    for (const TokenShardFile& shard : shards_) {
        chunks += shard.tokens > 0 ? (shard.tokens - 1U) / static_cast<std::uintmax_t>(seq_len_) : 0U;
    }
    return static_cast<std::int64_t>(chunks / static_cast<std::uintmax_t>(batch_size_));
}

fs::path native_datasets_dir() {
    std::string override = env_or_empty("NFN_DATASETS_DIR");
    if (!override.empty()) {
        return fs::path(override);
    }
    return home_dir() / ".cache" / "nfn" / "datasets";
}

fs::path resolve_dataset_path(const std::string& alias_or_path) {
    fs::path candidate(alias_or_path);
    if (candidate.is_absolute()) {
        return candidate;
    }
    const fs::path cached_alias = native_datasets_dir() / alias_or_path;
    if (fs::is_regular_file(cached_alias) || directory_has_native_token_bins(cached_alias)) {
        return cached_alias;
    }
    if (is_tinystories_alias(alias_or_path)) {
        const fs::path llm_path = llm_kittens_tinystories_dir();
        if (fs::is_regular_file(llm_path / "TinyStories_train.bin") &&
            fs::is_regular_file(llm_path / "TinyStories_val.bin")) {
            return llm_path;
        }
    }
    return cached_alias;
}

TokenShardDataset resolve_token_shards(const std::string& alias_or_path, bool allow_train_as_val) {
    TokenShardDataset dataset;
    dataset.dataset_path = resolve_dataset_path(alias_or_path);
    if (fs::is_regular_file(dataset.dataset_path)) {
        dataset.train_shards = {read_shard_file(dataset.dataset_path)};
        const fs::path val_path = inferred_validation_path(dataset.dataset_path);
        if (!val_path.empty()) {
            dataset.val_shards = {read_shard_file(val_path)};
        }
    } else if (fs::is_directory(dataset.dataset_path)) {
        dataset.train_shards = sorted_shards(dataset.dataset_path, "fineweb_train_");
        dataset.val_shards = sorted_shards(dataset.dataset_path, "fineweb_val_");
        if (dataset.train_shards.empty()) {
            dataset.train_shards = named_shards(dataset.dataset_path, {"TinyStories_train", "TinyStoriesV2-GPT4_train"});
        }
        if (dataset.val_shards.empty()) {
            dataset.val_shards = named_shards(dataset.dataset_path, {"TinyStories_val", "TinyStories_valid", "TinyStoriesV2-GPT4_val", "TinyStoriesV2-GPT4_valid"});
        }
    } else {
        throw std::runtime_error("dataset directory not found: " + dataset.dataset_path.string());
    }
    if (dataset.train_shards.empty()) {
        throw std::runtime_error(
            "no native uint16 train token bin found under " + dataset.dataset_path.string() +
            " (expected fineweb_train_*.bin or TinyStories_train.bin)");
    }
    if (dataset.val_shards.empty()) {
        if (!allow_train_as_val) {
            throw std::runtime_error(
                "no native uint16 validation token bin found under " + dataset.dataset_path.string() +
                " (expected fineweb_val_*.bin, TinyStories_val.bin, or an inferred sibling for a direct train file)");
        }
        dataset.val_shards = dataset.train_shards;
    }
    dataset.train_tokens = sum_tokens(dataset.train_shards);
    dataset.val_tokens = sum_tokens(dataset.val_shards);
    return dataset;
}

BatchPlan build_batch_plan(
    const TokenShardDataset& dataset,
    std::int64_t seq_len,
    std::int64_t batch_size,
    std::int64_t train_batch_tokens) {
    seq_len = checked_positive(seq_len, "seq_len");
    batch_size = checked_positive(batch_size, "batch_size");
    train_batch_tokens = checked_positive(train_batch_tokens, "train_batch_tokens");
    BatchPlan plan;
    plan.microbatch_tokens = seq_len * batch_size;
    plan.grad_accum_steps = ceil_div(train_batch_tokens, plan.microbatch_tokens);
    plan.effective_train_batch_tokens = plan.grad_accum_steps * plan.microbatch_tokens;
    plan.train_sequences = static_cast<std::int64_t>(sum_sequences(dataset.train_shards, seq_len));
    plan.val_sequences = static_cast<std::int64_t>(sum_sequences(dataset.val_shards, seq_len));
    plan.train_microbatches = ceil_div(plan.train_sequences, batch_size);
    plan.train_optimizer_steps_per_epoch = ceil_div(plan.train_microbatches, plan.grad_accum_steps);
    plan.val_microbatches = ceil_div(plan.val_sequences, batch_size);
    return plan;
}

std::string token_shard_dataset_json(const TokenShardDataset& dataset, const BatchPlan* batch_plan) {
    std::ostringstream out;
    out << "{"
        << "\"dataset_path\": \"" << json_escape(dataset.dataset_path.string()) << "\", "
        << "\"batch_read_strategy\": \"contiguous_shard_segments\", "
        << "\"train_tokens\": " << dataset.train_tokens << ", "
        << "\"val_tokens\": " << dataset.val_tokens << ", "
        << "\"train_shards\": ";
    append_shards_json(out, dataset.train_shards);
    out << ", \"val_shards\": ";
    append_shards_json(out, dataset.val_shards);
    if (batch_plan != nullptr) {
        out << ", \"batch_plan\": {"
            << "\"microbatch_tokens\": " << batch_plan->microbatch_tokens << ", "
            << "\"grad_accum_steps\": " << batch_plan->grad_accum_steps << ", "
            << "\"effective_train_batch_tokens\": " << batch_plan->effective_train_batch_tokens << ", "
            << "\"train_sequences\": " << batch_plan->train_sequences << ", "
            << "\"train_microbatches\": " << batch_plan->train_microbatches << ", "
            << "\"train_optimizer_steps_per_epoch\": " << batch_plan->train_optimizer_steps_per_epoch << ", "
            << "\"val_sequences\": " << batch_plan->val_sequences << ", "
            << "\"val_microbatches\": " << batch_plan->val_microbatches << "}";
    }
    out << "}";
    return out.str();
}

std::string token_batch_json(const TokenBatch& batch, std::size_t max_items) {
    std::ostringstream out;
    const std::size_t total = batch.tokens.size();
    const std::size_t limit = std::min(total, max_items);
    out << "{"
        << "\"batch_size\": " << batch.batch_size << ", "
        << "\"seq_len\": " << batch.seq_len << ", "
        << "\"items\": " << total << ", "
        << "\"tokens\": [";
    for (std::size_t i = 0; i < limit; ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << batch.tokens[i];
    }
    out << "], \"targets\": [";
    for (std::size_t i = 0; i < limit; ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << batch.targets[i];
    }
    out << "]}";
    return out.str();
}

}  // namespace neuralfn::native_train
