#include "token_shards.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
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

std::uint32_t read_le32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t read_le64(const unsigned char* bytes) {
    return static_cast<std::uint64_t>(read_le32(bytes)) |
        (static_cast<std::uint64_t>(read_le32(bytes + 4)) << 32U);
}

std::int32_t read_le_i32(const unsigned char* bytes) {
    return std::bit_cast<std::int32_t>(read_le32(bytes));
}

float read_le_f32(const unsigned char* bytes) {
    return std::bit_cast<float>(read_le32(bytes));
}

std::string fixed_header_string(
    const std::array<unsigned char, kTokenShardV2HeaderBytes>& header,
    std::size_t offset,
    std::size_t width,
    const char* label) {
    if (offset + width > header.size()) {
        throw std::runtime_error(std::string("internal token shard header field overflow: ") + label);
    }
    const auto begin = header.begin() + static_cast<std::ptrdiff_t>(offset);
    const auto end = begin + static_cast<std::ptrdiff_t>(width);
    const auto nul = std::find(begin, end, static_cast<unsigned char>(0));
    if (nul == end) {
        throw std::runtime_error(std::string("token shard v2 field is not NUL-terminated: ") + label);
    }
    std::string value(begin, nul);
    for (unsigned char ch : value) {
        if (ch < 0x20U || ch > 0x7eU) {
            throw std::runtime_error(std::string("token shard v2 field contains non-ASCII bytes: ") + label);
        }
    }
    for (auto it = nul; it != end; ++it) {
        if (*it != 0U) {
            throw std::runtime_error(std::string("token shard v2 field has nonzero padding: ") + label);
        }
    }
    return value;
}

bool is_lower_hex_sha256(const std::string& value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0 || (ch >= 'a' && ch <= 'f');
    });
}

std::uint32_t validate_uint32_payload(
    const fs::path& path,
    std::uint64_t token_count,
    std::uint32_t tokenizer_vocab_size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open uint32 token shard: " + path.string());
    }
    input.seekg(static_cast<std::streamoff>(kTokenShardV2HeaderBytes), std::ios::beg);
    constexpr std::size_t kChunkTokens = 1U << 16U;
    std::vector<unsigned char> bytes(kChunkTokens * sizeof(std::uint32_t));
    std::uint64_t remaining = token_count;
    std::uint32_t max_token = 0;
    while (remaining > 0U) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(kChunkTokens)));
        const std::size_t byte_count = count * sizeof(std::uint32_t);
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byte_count));
        if (input.gcount() != static_cast<std::streamsize>(byte_count)) {
            throw std::runtime_error("short read while validating uint32 token shard: " + path.string());
        }
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t token = read_le32(bytes.data() + index * sizeof(std::uint32_t));
            if (token >= tokenizer_vocab_size) {
                throw std::runtime_error(
                    "uint32 token shard contains token id " + std::to_string(token) +
                    " outside tokenizer vocab " + std::to_string(tokenizer_vocab_size) +
                    ": " + path.string());
            }
            max_token = std::max(max_token, token);
        }
        remaining -= static_cast<std::uint64_t>(count);
    }
    return max_token;
}

TokenShardFile read_v2_uint32_shard_file(const fs::path& path, std::uintmax_t bytes) {
    if (bytes < kTokenShardV2HeaderBytes) {
        throw std::runtime_error("token shard v2 is smaller than its header: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    std::array<unsigned char, kTokenShardV2HeaderBytes> header{};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())) {
        throw std::runtime_error("short token shard v2 header: " + path.string());
    }
    constexpr std::array<unsigned char, 8> kMagic = {'N', 'F', 'N', 'T', 'S', 'H', '2', 0};
    if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
        throw std::runtime_error("invalid token shard v2 magic: " + path.string());
    }
    const std::uint32_t version = read_le32(header.data() + 8);
    const std::uint32_t header_bytes = read_le32(header.data() + 12);
    const std::uint32_t dtype_code = read_le32(header.data() + 16);
    const std::uint32_t endian = read_le32(header.data() + 20);
    const std::uint64_t token_count = read_le64(header.data() + 24);
    const std::uint32_t vocab_size = read_le32(header.data() + 32);
    const std::uint32_t flags = read_le32(header.data() + 36);
    if (version != kTokenShardV2Version || header_bytes != kTokenShardV2HeaderBytes) {
        throw std::runtime_error("unsupported token shard v2 version/header size: " + path.string());
    }
    if (dtype_code != static_cast<std::uint32_t>(TokenShardDType::uint32_le)) {
        throw std::runtime_error("unsupported token shard v2 dtype code " + std::to_string(dtype_code) + ": " + path.string());
    }
    if (endian != kTokenShardLittleEndianMarker) {
        throw std::runtime_error("token shard v2 endian marker mismatch: " + path.string());
    }
    if (token_count == 0U || vocab_size == 0U || flags != 0U) {
        throw std::runtime_error("invalid token shard v2 token count, vocab, or flags: " + path.string());
    }
    if (token_count > (std::numeric_limits<std::uintmax_t>::max() - header_bytes) / 4U ||
        bytes != static_cast<std::uintmax_t>(header_bytes) + static_cast<std::uintmax_t>(token_count) * 4U) {
        throw std::runtime_error("token shard v2 byte size does not match token_count: " + path.string());
    }
    const std::string tokenizer_sha256 = fixed_header_string(header, 40, 65, "tokenizer_sha256");
    const std::string tokenizer_revision = fixed_header_string(header, 105, 96, "tokenizer_revision");
    const std::string split = fixed_header_string(header, 201, 32, "split");
    const std::string objective = fixed_header_string(header, 233, 32, "objective");
    const std::string tokenizer_name = fixed_header_string(header, 265, 128, "tokenizer_name");
    if (!is_lower_hex_sha256(tokenizer_sha256) || tokenizer_revision.empty() ||
        tokenizer_name.empty() || (split != "train" && split != "validation" && split != "test") ||
        (objective != "ar" && objective != "pretrain")) {
        throw std::runtime_error("invalid token shard v2 tokenizer/split/objective metadata: " + path.string());
    }
    for (std::size_t index = 393; index < header.size(); ++index) {
        if (header[index] != 0U) {
            throw std::runtime_error("token shard v2 reserved header bytes must be zero: " + path.string());
        }
    }
    const std::uint32_t max_token_id = validate_uint32_payload(path, token_count, vocab_size);
    return TokenShardFile{
        .path = path,
        .bytes = bytes,
        .header_uint16 = 0,
        .tokens = static_cast<std::uintmax_t>(token_count),
        .header_bytes = header_bytes,
        .element_bytes = 4,
        .dtype = TokenShardDType::uint32_le,
        .tokenizer_vocab_size = vocab_size,
        .max_token_id = max_token_id,
        .tokenizer_sha256 = tokenizer_sha256,
        .tokenizer_revision = tokenizer_revision,
        .tokenizer_name = tokenizer_name,
        .split = split,
        .objective = objective,
    };
}

StructuredSftFile read_structured_sft_file(const fs::path& path) {
    const std::uintmax_t bytes = fs::file_size(path);
    if (bytes < kStructuredSftV1HeaderBytes) {
        throw std::runtime_error("structured SFT file is smaller than its header: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    std::array<unsigned char, kStructuredSftV1HeaderBytes> header{};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (input.gcount() != static_cast<std::streamsize>(header.size())) {
        throw std::runtime_error("short structured SFT header: " + path.string());
    }
    constexpr std::array<unsigned char, 8> kMagic = {'N', 'F', 'N', 'S', 'F', 'T', '1', 0};
    if (!std::equal(kMagic.begin(), kMagic.end(), header.begin()) ||
        read_le32(header.data() + 8) != kStructuredSftV1Version ||
        read_le32(header.data() + 12) != kStructuredSftV1HeaderBytes ||
        read_le32(header.data() + 16) != kTokenShardLittleEndianMarker ||
        read_le32(header.data() + 20) != 0U) {
        throw std::runtime_error("invalid structured SFT magic/version/endian/flags: " + path.string());
    }
    const std::uint64_t records = read_le64(header.data() + 24);
    const std::uint32_t sequence_length = read_le32(header.data() + 32);
    const std::uint32_t vocab_size = read_le32(header.data() + 36);
    const std::uint32_t pad_token_id = read_le32(header.data() + 40);
    if (records == 0 || sequence_length == 0 || vocab_size == 0 || pad_token_id >= vocab_size) {
        throw std::runtime_error("invalid structured SFT record geometry: " + path.string());
    }
    const std::uint64_t record_bytes = static_cast<std::uint64_t>(sequence_length) * 16U;
    if (records > (std::numeric_limits<std::uintmax_t>::max() - kStructuredSftV1HeaderBytes) /
            record_bytes ||
        bytes != kStructuredSftV1HeaderBytes + records * record_bytes) {
        throw std::runtime_error("structured SFT byte size does not match its record table: " + path.string());
    }
    const std::string tokenizer_sha256 = fixed_header_string(
        header, 48, 65, "tokenizer_sha256");
    const std::string chat_template_sha256 = fixed_header_string(
        header, 113, 65, "chat_template_sha256");
    const std::string tokenizer_revision = fixed_header_string(
        header, 178, 96, "tokenizer_revision");
    const std::string split = fixed_header_string(header, 274, 32, "split");
    const std::string objective = fixed_header_string(header, 306, 32, "objective");
    if (!is_lower_hex_sha256(tokenizer_sha256) ||
        !is_lower_hex_sha256(chat_template_sha256) || tokenizer_revision.empty() ||
        (split != "train" && split != "validation" && split != "test") ||
        objective != "sft") {
        throw std::runtime_error("invalid structured SFT lineage/split/objective metadata: " + path.string());
    }
    for (std::size_t index = 338; index < header.size(); ++index) {
        if (header[index] != 0U) {
            throw std::runtime_error("structured SFT reserved header bytes must be zero: " + path.string());
        }
    }

    constexpr std::uint64_t kChunkRecords = 64;
    std::vector<unsigned char> payload(static_cast<std::size_t>(
        std::min<std::uint64_t>(records, kChunkRecords) * record_bytes));
    std::uint64_t remaining = records;
    while (remaining > 0) {
        const std::uint64_t count = std::min<std::uint64_t>(remaining, kChunkRecords);
        const std::size_t count_bytes = static_cast<std::size_t>(count * record_bytes);
        input.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(count_bytes));
        if (input.gcount() != static_cast<std::streamsize>(count_bytes)) {
            throw std::runtime_error("short structured SFT payload: " + path.string());
        }
        for (std::uint64_t record = 0; record < count; ++record) {
            const unsigned char* base = payload.data() + record * record_bytes;
            const unsigned char* target_bytes = base + sequence_length * 4U;
            const unsigned char* mask_bytes = target_bytes + sequence_length * 4U;
            const unsigned char* segment_bytes = mask_bytes + sequence_length * 4U;
            double mask_sum = 0.0;
            std::int32_t previous_segment = 0;
            for (std::uint32_t token = 0; token < sequence_length; ++token) {
                const std::uint32_t input_id = read_le32(base + token * 4U);
                const std::int32_t target_id = read_le_i32(target_bytes + token * 4U);
                const float mask = read_le_f32(mask_bytes + token * 4U);
                const std::int32_t segment = read_le_i32(segment_bytes + token * 4U);
                if (input_id >= vocab_size ||
                    (target_id != -100 && (target_id < 0 ||
                        static_cast<std::uint32_t>(target_id) >= vocab_size)) ||
                    !std::isfinite(mask) || mask < 0.0f ||
                    (target_id == -100 && mask != 0.0f) || segment < 0 ||
                    (token == 0 && segment != 0) ||
                    (token > 0 && segment != previous_segment && segment != previous_segment + 1)) {
                    throw std::runtime_error("invalid structured SFT record payload: " + path.string());
                }
                previous_segment = segment;
                mask_sum += mask;
            }
            if (!(mask_sum > 0.0) || !std::isfinite(mask_sum)) {
                throw std::runtime_error("structured SFT record has an empty loss mask: " + path.string());
            }
        }
        remaining -= count;
    }
    return StructuredSftFile{
        .path = path,
        .bytes = bytes,
        .records = records,
        .sequence_length = sequence_length,
        .tokenizer_vocab_size = vocab_size,
        .pad_token_id = pad_token_id,
        .tokenizer_sha256 = tokenizer_sha256,
        .chat_template_sha256 = chat_template_sha256,
        .tokenizer_revision = tokenizer_revision,
        .split = split,
    };
}

TokenShardFile read_shard_file(const fs::path& path) {
    const std::uintmax_t bytes = fs::file_size(path);
    {
        std::ifstream input(path, std::ios::binary);
        std::array<unsigned char, 8> magic{};
        input.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
        constexpr std::array<unsigned char, 8> kV2Magic = {'N', 'F', 'N', 'T', 'S', 'H', '2', 0};
        if (input.gcount() == static_cast<std::streamsize>(magic.size()) && magic == kV2Magic) {
            return read_v2_uint32_shard_file(path, bytes);
        }
    }
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
        .header_bytes = header_uint16 * 2U,
        .element_bytes = 2,
        .dtype = TokenShardDType::legacy_uint16_le,
        .tokenizer_vocab_size = 65'536U,
        .max_token_id = 65'535U,
        .tokenizer_sha256 = {},
        .tokenizer_revision = {},
        .tokenizer_name = {},
        .split = {},
        .objective = {},
    };
}

TokenShardFile read_byte_shard_file(const fs::path& path) {
    const std::uintmax_t bytes = fs::file_size(path);
    return TokenShardFile{
        .path = path,
        .bytes = bytes,
        .header_uint16 = 0,
        .tokens = bytes,
        .header_bytes = 0,
        .element_bytes = 1,
        .dtype = TokenShardDType::legacy_uint16_le,
        .tokenizer_vocab_size = 0,
        .max_token_id = 0,
        .tokenizer_sha256 = {},
        .tokenizer_revision = {},
        .tokenizer_name = {},
        .split = {},
        .objective = {},
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

std::vector<TokenShardFile> sorted_byte_shards(const fs::path& dataset_path, const std::vector<std::string>& prefixes) {
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
        shards.push_back(read_byte_shard_file(entry.path()));
    }
    std::sort(shards.begin(), shards.end(), [](const TokenShardFile& lhs, const TokenShardFile& rhs) {
        return lhs.path < rhs.path;
    });
    return shards;
}

std::vector<StructuredSftFile> sorted_structured_sft_files(
    const fs::path& dataset_path,
    const std::string& prefix) {
    std::vector<StructuredSftFile> files;
    if (!fs::is_directory(dataset_path)) return files;
    for (const fs::directory_entry& entry : fs::directory_iterator(dataset_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".sft" ||
            entry.path().filename().string().rfind(prefix, 0) != 0) {
            continue;
        }
        files.push_back(read_structured_sft_file(entry.path()));
    }
    std::sort(files.begin(), files.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.path < rhs.path;
    });
    return files;
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

std::vector<TokenShardFile> named_byte_shards(const fs::path& dataset_path, const std::vector<std::string>& stems) {
    std::vector<TokenShardFile> shards;
    if (!fs::is_directory(dataset_path)) {
        return shards;
    }
    for (const std::string& stem : stems) {
        const fs::path candidate = dataset_path / (stem + ".bin");
        if (fs::is_regular_file(candidate) && has_name_and_bin_extension(candidate, stem)) {
            shards.push_back(read_byte_shard_file(candidate));
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

bool directory_has_native_byte_bins(const fs::path& dataset_path) {
    return directory_has_matching_bin(dataset_path, {"byte_train_", "hnet_train_"}, {"bytes_train", "hnet_train"}) &&
           directory_has_matching_bin(dataset_path, {"byte_val_", "hnet_val_"}, {"bytes_val", "bytes_valid", "hnet_val", "hnet_valid"});
}

bool directory_has_structured_sft_files(const fs::path& dataset_path) {
    return !sorted_structured_sft_files(dataset_path, "sft_train_").empty();
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
            << "\"header_bytes\": " << shards[i].header_bytes << ", "
            << "\"element_bytes\": " << shards[i].element_bytes << ", "
            << "\"dtype\": \""
            << (shards[i].element_bytes == 1U
                    ? "uint8"
                    : (shards[i].dtype == TokenShardDType::uint32_le ? "uint32_le" : "uint16_le"))
            << "\", "
            << "\"tokenizer_vocab_size\": " << shards[i].tokenizer_vocab_size << ", "
            << "\"max_token_id\": " << shards[i].max_token_id << ", "
            << "\"tokenizer_sha256\": \"" << json_escape(shards[i].tokenizer_sha256) << "\", "
            << "\"tokenizer_revision\": \"" << json_escape(shards[i].tokenizer_revision) << "\", "
            << "\"tokenizer_name\": \"" << json_escape(shards[i].tokenizer_name) << "\", "
            << "\"split\": \"" << json_escape(shards[i].split) << "\", "
            << "\"objective\": \"" << json_escape(shards[i].objective) << "\", "
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
    const std::uintmax_t byte_offset = start * 2U;
    const std::uintmax_t byte_count = values * 2U;
    if (shard.stable_fd >= 0) {
        const auto matches_snapshot = [&](const struct stat& status) {
            return S_ISREG(status.st_mode) && status.st_size >= 0 &&
                static_cast<std::uintmax_t>(status.st_size) == shard.bytes &&
                static_cast<std::uintmax_t>(status.st_dev) == shard.stable_device &&
                static_cast<std::uintmax_t>(status.st_ino) == shard.stable_inode &&
                static_cast<std::int64_t>(status.st_mtim.tv_sec) ==
                    shard.stable_mtime_seconds &&
                static_cast<std::int64_t>(status.st_mtim.tv_nsec) ==
                    shard.stable_mtime_nanoseconds &&
                static_cast<std::int64_t>(status.st_ctim.tv_sec) ==
                    shard.stable_ctime_seconds &&
                static_cast<std::int64_t>(status.st_ctim.tv_nsec) ==
                    shard.stable_ctime_nanoseconds;
        };
        struct stat before {};
        if (::fstat(shard.stable_fd, &before) != 0 ||
            !matches_snapshot(before) ||
            byte_offset > static_cast<std::uintmax_t>(
                std::numeric_limits<off_t>::max()) ||
            byte_count > static_cast<std::uintmax_t>(
                std::numeric_limits<ssize_t>::max())) {
            throw std::runtime_error(
                "stable token shard changed before batch read: " +
                shard.path.string());
        }
        std::size_t consumed = 0;
        while (consumed < static_cast<std::size_t>(byte_count)) {
            ssize_t count = -1;
            do {
                count = ::pread(
                    shard.stable_fd,
                    reinterpret_cast<char*>(scratch.data()) + consumed,
                    static_cast<std::size_t>(byte_count) - consumed,
                    static_cast<off_t>(byte_offset + consumed));
            } while (count < 0 && errno == EINTR);
            if (count <= 0) {
                throw std::runtime_error(
                    "stable token shard changed during batch read: " +
                    shard.path.string());
            }
            consumed += static_cast<std::size_t>(count);
        }
        struct stat after {};
        if (::fstat(shard.stable_fd, &after) != 0 ||
            !matches_snapshot(after)) {
            throw std::runtime_error(
                "stable token shard changed during batch read: " +
                shard.path.string());
        }
    } else {
        std::ifstream input(shard.path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open token shard: " + shard.path.string());
        }
        input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        input.read(
            reinterpret_cast<char*>(scratch.data()),
            static_cast<std::streamsize>(byte_count));
        if (input.gcount() != static_cast<std::streamsize>(byte_count)) {
            throw std::runtime_error("short read from token shard: " + shard.path.string());
        }
    }
    const std::size_t token_count = static_cast<std::size_t>(chunk_count * static_cast<std::uintmax_t>(seq_len));
    std::memcpy(tokens + offset, scratch.data(), token_count * sizeof(std::uint16_t));
    std::memcpy(targets + offset, scratch.data() + 1, token_count * sizeof(std::uint16_t));
}

void append_contiguous_chunks_into_wide(
    const TokenShardFile& shard,
    std::uintmax_t local_chunk_index,
    std::uintmax_t chunk_count,
    std::int64_t seq_len,
    std::uint32_t* tokens,
    std::uint32_t* targets,
    std::size_t offset,
    std::vector<std::uint8_t>& scratch) {
    if (chunk_count == 0) {
        return;
    }
    if (shard.element_bytes != 2U && shard.element_bytes != 4U) {
        throw std::runtime_error("unsupported token shard element width: " + shard.path.string());
    }
    const std::uintmax_t start = local_chunk_index * static_cast<std::uintmax_t>(seq_len);
    const std::uintmax_t values = chunk_count * static_cast<std::uintmax_t>(seq_len) + 1U;
    if (values > std::numeric_limits<std::size_t>::max() / shard.element_bytes) {
        throw std::runtime_error("wide token shard batch read is too large: " + shard.path.string());
    }
    const std::uintmax_t byte_offset = shard.header_bytes + start * shard.element_bytes;
    const std::uintmax_t byte_count = values * shard.element_bytes;
    scratch.resize(static_cast<std::size_t>(byte_count));
    if (shard.stable_fd >= 0) {
        const auto matches_snapshot = [&](const struct stat& status) {
            return S_ISREG(status.st_mode) && status.st_size >= 0 &&
                static_cast<std::uintmax_t>(status.st_size) == shard.bytes &&
                static_cast<std::uintmax_t>(status.st_dev) == shard.stable_device &&
                static_cast<std::uintmax_t>(status.st_ino) == shard.stable_inode &&
                static_cast<std::int64_t>(status.st_mtim.tv_sec) == shard.stable_mtime_seconds &&
                static_cast<std::int64_t>(status.st_mtim.tv_nsec) == shard.stable_mtime_nanoseconds &&
                static_cast<std::int64_t>(status.st_ctim.tv_sec) == shard.stable_ctime_seconds &&
                static_cast<std::int64_t>(status.st_ctim.tv_nsec) == shard.stable_ctime_nanoseconds;
        };
        struct stat before {};
        if (::fstat(shard.stable_fd, &before) != 0 || !matches_snapshot(before) ||
            byte_offset > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max()) ||
            byte_count > static_cast<std::uintmax_t>(std::numeric_limits<ssize_t>::max())) {
            throw std::runtime_error("stable wide token shard changed before batch read: " + shard.path.string());
        }
        std::size_t consumed = 0;
        while (consumed < scratch.size()) {
            ssize_t count = -1;
            do {
                count = ::pread(
                    shard.stable_fd,
                    reinterpret_cast<char*>(scratch.data()) + consumed,
                    scratch.size() - consumed,
                    static_cast<off_t>(byte_offset + consumed));
            } while (count < 0 && errno == EINTR);
            if (count <= 0) {
                throw std::runtime_error("stable wide token shard changed during batch read: " + shard.path.string());
            }
            consumed += static_cast<std::size_t>(count);
        }
        struct stat after {};
        if (::fstat(shard.stable_fd, &after) != 0 || !matches_snapshot(after)) {
            throw std::runtime_error("stable wide token shard changed during batch read: " + shard.path.string());
        }
    } else {
        std::ifstream input(shard.path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("failed to open wide token shard: " + shard.path.string());
        }
        input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        input.read(reinterpret_cast<char*>(scratch.data()), static_cast<std::streamsize>(byte_count));
        if (input.gcount() != static_cast<std::streamsize>(byte_count)) {
            throw std::runtime_error("short read from wide token shard: " + shard.path.string());
        }
    }
    const auto decode = [&](std::size_t index) -> std::uint32_t {
        const unsigned char* value = scratch.data() + index * shard.element_bytes;
        return shard.element_bytes == 2U
            ? static_cast<std::uint32_t>(value[0]) | (static_cast<std::uint32_t>(value[1]) << 8U)
            : read_le32(value);
    };
    const std::size_t token_count = static_cast<std::size_t>(
        chunk_count * static_cast<std::uintmax_t>(seq_len));
    for (std::size_t index = 0; index < token_count; ++index) {
        const std::uint32_t token = decode(index);
        const std::uint32_t target = decode(index + 1U);
        if (shard.tokenizer_vocab_size != 0U &&
            (token >= shard.tokenizer_vocab_size || target >= shard.tokenizer_vocab_size)) {
            throw std::runtime_error("token shard value changed outside declared tokenizer vocab: " + shard.path.string());
        }
        tokens[offset + index] = token;
        targets[offset + index] = target;
    }
}

void append_contiguous_byte_chunks_into(
    const TokenShardFile& shard,
    std::uintmax_t local_chunk_index,
    std::uintmax_t chunk_count,
    std::int64_t seq_len,
    std::uint8_t* tokens,
    std::uint8_t* targets,
    std::size_t offset,
    std::vector<std::uint8_t>& scratch) {
    if (chunk_count == 0) {
        return;
    }
    const std::uintmax_t start = local_chunk_index * static_cast<std::uintmax_t>(seq_len);
    const std::uintmax_t values = chunk_count * static_cast<std::uintmax_t>(seq_len) + 1U;
    scratch.resize(static_cast<std::size_t>(values));
    std::ifstream input(shard.path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open byte token shard: " + shard.path.string());
    }
    input.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    input.read(reinterpret_cast<char*>(scratch.data()), static_cast<std::streamsize>(values));
    if (input.gcount() != static_cast<std::streamsize>(values)) {
        throw std::runtime_error("short read from byte token shard: " + shard.path.string());
    }
    const std::size_t token_count = static_cast<std::size_t>(chunk_count * static_cast<std::uintmax_t>(seq_len));
    std::memcpy(tokens + offset, scratch.data(), token_count * sizeof(std::uint8_t));
    std::memcpy(targets + offset, scratch.data() + 1, token_count * sizeof(std::uint8_t));
}

}  // namespace

SequentialTokenBatchSampler::SequentialTokenBatchSampler(
    std::vector<TokenShardFile> shards,
    std::int64_t seq_len,
    std::int64_t batch_size)
    : shards_(std::move(shards)),
      seq_len_(checked_positive(seq_len, "seq_len")),
      batch_size_(checked_positive(batch_size, "batch_size")) {
    for (const TokenShardFile& shard : shards_) {
        if (shard.dtype != TokenShardDType::legacy_uint16_le || shard.element_bytes != 2U) {
            throw std::runtime_error(
                "legacy uint16 sampler cannot consume versioned uint32 token shard: " + shard.path.string());
        }
    }
}

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

bool SequentialTokenBatchSampler::seek_batch(std::int64_t batch_index) {
    if (batch_index < 0) {
        return false;
    }
    const std::int64_t batches = total_batches();
    if (batches <= 0) {
        return false;
    }
    std::uintmax_t chunks_to_skip =
        static_cast<std::uintmax_t>(batch_index % batches) * static_cast<std::uintmax_t>(batch_size_);
    shard_index_ = 0;
    local_chunk_index_ = 0;
    while (shard_index_ < shards_.size()) {
        const TokenShardFile& shard = shards_[shard_index_];
        const std::uintmax_t chunk_count = shard.tokens > 0
            ? (shard.tokens - 1U) / static_cast<std::uintmax_t>(seq_len_)
            : 0U;
        if (chunks_to_skip < chunk_count) {
            local_chunk_index_ = chunks_to_skip;
            return true;
        }
        chunks_to_skip -= chunk_count;
        shard_index_ += 1;
    }
    reset();
    return chunks_to_skip == 0;
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

SequentialTokenBatchSampler32::SequentialTokenBatchSampler32(
    std::vector<TokenShardFile> shards,
    std::int64_t seq_len,
    std::int64_t batch_size)
    : shards_(std::move(shards)),
      seq_len_(checked_positive(seq_len, "seq_len")),
      batch_size_(checked_positive(batch_size, "batch_size")) {
    for (const TokenShardFile& shard : shards_) {
        if ((shard.dtype != TokenShardDType::legacy_uint16_le &&
             shard.dtype != TokenShardDType::uint32_le) ||
            (shard.element_bytes != 2U && shard.element_bytes != 4U)) {
            throw std::runtime_error("wide sampler encountered unsupported token shard dtype: " + shard.path.string());
        }
    }
}

bool SequentialTokenBatchSampler32::next(TokenBatch32& out) {
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

bool SequentialTokenBatchSampler32::next_into(
    std::uint32_t* tokens,
    std::uint32_t* targets,
    std::int64_t token_capacity) {
    if (tokens == nullptr || targets == nullptr) {
        throw std::runtime_error("wide token batch destination pointers must be non-null");
    }
    const std::int64_t total = batch_size_ * seq_len_;
    if (token_capacity < total) {
        throw std::runtime_error("wide token batch destination capacity is smaller than batch_size * seq_len");
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
        append_contiguous_chunks_into_wide(
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

bool SequentialTokenBatchSampler32::seek_batch(std::int64_t batch_index) {
    if (batch_index < 0) {
        return false;
    }
    const std::int64_t batches = total_batches();
    if (batches <= 0) {
        return false;
    }
    std::uintmax_t chunks_to_skip =
        static_cast<std::uintmax_t>(batch_index % batches) * static_cast<std::uintmax_t>(batch_size_);
    shard_index_ = 0;
    local_chunk_index_ = 0;
    while (shard_index_ < shards_.size()) {
        const TokenShardFile& shard = shards_[shard_index_];
        const std::uintmax_t chunk_count = shard.tokens > 0
            ? (shard.tokens - 1U) / static_cast<std::uintmax_t>(seq_len_)
            : 0U;
        if (chunks_to_skip < chunk_count) {
            local_chunk_index_ = chunks_to_skip;
            return true;
        }
        chunks_to_skip -= chunk_count;
        shard_index_ += 1;
    }
    reset();
    return chunks_to_skip == 0;
}

void SequentialTokenBatchSampler32::reset() {
    shard_index_ = 0;
    local_chunk_index_ = 0;
}

std::int64_t SequentialTokenBatchSampler32::total_batches() const {
    std::uintmax_t chunks = 0;
    for (const TokenShardFile& shard : shards_) {
        chunks += shard.tokens > 0 ? (shard.tokens - 1U) / static_cast<std::uintmax_t>(seq_len_) : 0U;
    }
    return static_cast<std::int64_t>(chunks / static_cast<std::uintmax_t>(batch_size_));
}

SequentialStructuredSftBatchSampler::SequentialStructuredSftBatchSampler(
    std::vector<StructuredSftFile> files,
    std::int64_t batch_size)
    : files_(std::move(files)), batch_size_(checked_positive(batch_size, "batch_size")) {
    if (files_.empty()) {
        throw std::runtime_error("structured SFT sampler requires at least one file");
    }
    sequence_length_ = files_.front().sequence_length;
    for (const auto& file : files_) {
        if (file.sequence_length != sequence_length_ || file.records == 0) {
            throw std::runtime_error("structured SFT files must share one nonempty sequence length");
        }
    }
}

bool SequentialStructuredSftBatchSampler::next(StructuredSftBatch& out) {
    out.batch_size = batch_size_;
    out.seq_len = sequence_length_;
    const std::size_t rows = static_cast<std::size_t>(batch_size_) * sequence_length_;
    out.input_ids.resize(rows);
    out.targets.resize(rows);
    out.loss_mask.resize(rows);
    out.sequence_ids.resize(rows);
    std::int64_t produced = 0;
    const std::uint64_t record_bytes = static_cast<std::uint64_t>(sequence_length_) * 16U;
    while (produced < batch_size_ && file_index_ < files_.size()) {
        const StructuredSftFile& file = files_[file_index_];
        if (local_record_index_ >= file.records) {
            ++file_index_;
            local_record_index_ = 0;
            continue;
        }
        const std::uint64_t available = file.records - local_record_index_;
        const std::uint64_t wanted = static_cast<std::uint64_t>(batch_size_ - produced);
        const std::uint64_t count = std::min(available, wanted);
        const std::uint64_t byte_offset = kStructuredSftV1HeaderBytes +
            local_record_index_ * record_bytes;
        const std::uint64_t byte_count = count * record_bytes;
        if (byte_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
            byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("structured SFT batch offset exceeds stream limits");
        }
        scratch_.resize(static_cast<std::size_t>(byte_count));
        std::ifstream input(file.path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open structured SFT file: " + file.path.string());
        input.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        input.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(byte_count));
        if (input.gcount() != static_cast<std::streamsize>(byte_count)) {
            throw std::runtime_error("short structured SFT batch read: " + file.path.string());
        }
        for (std::uint64_t record = 0; record < count; ++record) {
            const std::uint8_t* base = scratch_.data() + record * record_bytes;
            const std::uint8_t* target = base + sequence_length_ * 4U;
            const std::uint8_t* mask = target + sequence_length_ * 4U;
            const std::uint8_t* segment = mask + sequence_length_ * 4U;
            const std::size_t destination =
                static_cast<std::size_t>(produced + static_cast<std::int64_t>(record)) * sequence_length_;
            for (std::uint32_t token = 0; token < sequence_length_; ++token) {
                out.input_ids[destination + token] = read_le32(base + token * 4U);
                out.targets[destination + token] = read_le_i32(target + token * 4U);
                out.loss_mask[destination + token] = read_le_f32(mask + token * 4U);
                out.sequence_ids[destination + token] = read_le_i32(segment + token * 4U);
            }
        }
        produced += static_cast<std::int64_t>(count);
        local_record_index_ += count;
    }
    if (produced != batch_size_) {
        out.input_ids.clear();
        out.targets.clear();
        out.loss_mask.clear();
        out.sequence_ids.clear();
        return false;
    }
    return true;
}

bool SequentialStructuredSftBatchSampler::seek_batch(std::int64_t batch_index) {
    if (batch_index < 0 || total_batches() <= 0) return false;
    std::uint64_t records = static_cast<std::uint64_t>(
        batch_index % total_batches()) * static_cast<std::uint64_t>(batch_size_);
    file_index_ = 0;
    local_record_index_ = 0;
    while (file_index_ < files_.size()) {
        if (records < files_[file_index_].records) {
            local_record_index_ = records;
            return true;
        }
        records -= files_[file_index_].records;
        ++file_index_;
    }
    reset();
    return records == 0;
}

void SequentialStructuredSftBatchSampler::reset() {
    file_index_ = 0;
    local_record_index_ = 0;
}

std::int64_t SequentialStructuredSftBatchSampler::total_batches() const {
    std::uint64_t records = 0;
    for (const auto& file : files_) records += file.records;
    return static_cast<std::int64_t>(records / static_cast<std::uint64_t>(batch_size_));
}

SequentialByteBatchSampler::SequentialByteBatchSampler(
    std::vector<TokenShardFile> shards,
    std::int64_t seq_len,
    std::int64_t batch_size)
    : shards_(std::move(shards)),
      seq_len_(checked_positive(seq_len, "seq_len")),
      batch_size_(checked_positive(batch_size, "batch_size")) {}

bool SequentialByteBatchSampler::next(ByteBatch& out) {
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

bool SequentialByteBatchSampler::next_into(
    std::uint8_t* tokens,
    std::uint8_t* targets,
    std::int64_t token_capacity) {
    if (tokens == nullptr || targets == nullptr) {
        throw std::runtime_error("byte token batch destination pointers must be non-null");
    }
    const std::int64_t total = batch_size_ * seq_len_;
    if (token_capacity < total) {
        throw std::runtime_error("byte token batch destination capacity is smaller than batch_size * seq_len");
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
        append_contiguous_byte_chunks_into(
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

bool SequentialByteBatchSampler::seek_batch(std::int64_t batch_index) {
    if (batch_index < 0) {
        return false;
    }
    const std::int64_t batches = total_batches();
    if (batches <= 0) {
        return false;
    }
    std::uintmax_t chunks_to_skip =
        static_cast<std::uintmax_t>(batch_index % batches) * static_cast<std::uintmax_t>(batch_size_);
    shard_index_ = 0;
    local_chunk_index_ = 0;
    while (shard_index_ < shards_.size()) {
        const TokenShardFile& shard = shards_[shard_index_];
        const std::uintmax_t chunk_count = shard.tokens > 0
            ? (shard.tokens - 1U) / static_cast<std::uintmax_t>(seq_len_)
            : 0U;
        if (chunks_to_skip < chunk_count) {
            local_chunk_index_ = chunks_to_skip;
            return true;
        }
        chunks_to_skip -= chunk_count;
        shard_index_ += 1;
    }
    reset();
    return chunks_to_skip == 0;
}

void SequentialByteBatchSampler::reset() {
    shard_index_ = 0;
    local_chunk_index_ = 0;
}

std::int64_t SequentialByteBatchSampler::total_batches() const {
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
    if (fs::is_regular_file(cached_alias) || directory_has_native_token_bins(cached_alias) ||
        directory_has_native_byte_bins(cached_alias) ||
        directory_has_structured_sft_files(cached_alias)) {
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

TokenShardDataset resolve_token_shards(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation) {
    TokenShardDataset dataset;
    dataset.dataset_path = resolve_dataset_path(alias_or_path);
    if (fs::is_regular_file(dataset.dataset_path)) {
        dataset.train_shards = {read_shard_file(dataset.dataset_path)};
        if (require_validation) {
            const fs::path val_path = inferred_validation_path(dataset.dataset_path);
            if (!val_path.empty()) {
                dataset.val_shards = {read_shard_file(val_path)};
            }
        }
    } else if (fs::is_directory(dataset.dataset_path)) {
        dataset.train_shards = sorted_shards(dataset.dataset_path, "fineweb_train_");
        if (require_validation) {
            dataset.val_shards = sorted_shards(dataset.dataset_path, "fineweb_val_");
        }
        if (dataset.train_shards.empty()) {
            dataset.train_shards = named_shards(dataset.dataset_path, {"TinyStories_train", "TinyStoriesV2-GPT4_train"});
        }
        if (require_validation && dataset.val_shards.empty()) {
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
    if (require_validation && dataset.val_shards.empty()) {
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

StructuredSftDataset resolve_structured_sft_records(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation) {
    StructuredSftDataset dataset;
    dataset.dataset_path = resolve_dataset_path(alias_or_path);
    if (fs::is_regular_file(dataset.dataset_path)) {
        StructuredSftFile file = read_structured_sft_file(dataset.dataset_path);
        if (file.split != "train") {
            throw std::runtime_error("direct structured SFT dataset file must declare split=train");
        }
        dataset.train_files.push_back(std::move(file));
    } else if (fs::is_directory(dataset.dataset_path)) {
        dataset.train_files = sorted_structured_sft_files(dataset.dataset_path, "sft_train_");
        if (require_validation) {
            dataset.val_files = sorted_structured_sft_files(dataset.dataset_path, "sft_val_");
            auto alternate = sorted_structured_sft_files(dataset.dataset_path, "sft_validation_");
            dataset.val_files.insert(
                dataset.val_files.end(),
                std::make_move_iterator(alternate.begin()),
                std::make_move_iterator(alternate.end()));
            std::sort(dataset.val_files.begin(), dataset.val_files.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.path < rhs.path;
            });
        }
    } else {
        throw std::runtime_error("structured SFT dataset path not found: " + dataset.dataset_path.string());
    }
    if (dataset.train_files.empty()) {
        throw std::runtime_error(
            "no native structured SFT records found (expected sft_train_*.sft)");
    }
    if (require_validation && dataset.val_files.empty()) {
        if (!allow_train_as_val) {
            throw std::runtime_error(
                "no native structured SFT validation records found (expected sft_val_*.sft)");
        }
        dataset.val_files = dataset.train_files;
    }
    const auto bind = [&](const StructuredSftFile& file) {
        const auto& first = dataset.train_files.front();
        if (file.sequence_length != first.sequence_length ||
            file.tokenizer_vocab_size != first.tokenizer_vocab_size ||
            file.pad_token_id != first.pad_token_id ||
            file.tokenizer_sha256 != first.tokenizer_sha256 ||
            file.chat_template_sha256 != first.chat_template_sha256 ||
            file.tokenizer_revision != first.tokenizer_revision) {
            throw std::runtime_error("structured SFT files do not share one exact tokenizer/template/geometry lineage");
        }
    };
    for (const auto& file : dataset.train_files) {
        if (file.split != "train") throw std::runtime_error("structured SFT train filename has non-train split metadata");
        bind(file);
        dataset.train_records += file.records;
    }
    for (const auto& file : dataset.val_files) {
        if (&dataset.val_files != &dataset.train_files && file.split != "validation" && file.split != "train") {
            throw std::runtime_error("structured SFT validation file has invalid split metadata");
        }
        bind(file);
        dataset.val_records += file.records;
    }
    return dataset;
}

ByteShardDataset resolve_byte_shards(
    const std::string& alias_or_path,
    bool allow_train_as_val,
    bool require_validation) {
    ByteShardDataset dataset;
    dataset.dataset_path = resolve_dataset_path(alias_or_path);
    if (fs::is_regular_file(dataset.dataset_path)) {
        dataset.train_shards = {read_byte_shard_file(dataset.dataset_path)};
        if (require_validation) {
            const fs::path val_path = inferred_validation_path(dataset.dataset_path);
            if (!val_path.empty()) {
                dataset.val_shards = {read_byte_shard_file(val_path)};
            }
        }
    } else if (fs::is_directory(dataset.dataset_path)) {
        dataset.train_shards = sorted_byte_shards(dataset.dataset_path, {"byte_train_", "hnet_train_"});
        if (require_validation) {
            dataset.val_shards = sorted_byte_shards(dataset.dataset_path, {"byte_val_", "hnet_val_"});
        }
        if (dataset.train_shards.empty()) {
            dataset.train_shards = named_byte_shards(dataset.dataset_path, {"bytes_train", "hnet_train"});
        }
        if (require_validation && dataset.val_shards.empty()) {
            dataset.val_shards = named_byte_shards(dataset.dataset_path, {"bytes_val", "bytes_valid", "hnet_val", "hnet_valid"});
        }
    } else {
        throw std::runtime_error("dataset directory not found: " + dataset.dataset_path.string());
    }
    if (dataset.train_shards.empty()) {
        throw std::runtime_error(
            "no native byte train token bin found under " + dataset.dataset_path.string() +
            " (expected byte_train_*.bin, hnet_train_*.bin, bytes_train.bin, or hnet_train.bin)");
    }
    if (require_validation && dataset.val_shards.empty()) {
        if (!allow_train_as_val) {
            throw std::runtime_error(
                "no native byte validation token bin found under " + dataset.dataset_path.string() +
                " (expected byte_val_*.bin, hnet_val_*.bin, bytes_val.bin, or an inferred sibling for a direct train file)");
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

BatchPlan build_batch_plan(
    const ByteShardDataset& dataset,
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

std::string byte_shard_dataset_json(const ByteShardDataset& dataset, const BatchPlan* batch_plan) {
    std::ostringstream out;
    out << "{"
        << "\"dataset_path\": \"" << json_escape(dataset.dataset_path.string()) << "\", "
        << "\"format\": \"uint8_byte_shards\", "
        << "\"batch_read_strategy\": \"contiguous_byte_shard_segments\", "
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

std::string token_batch_json(const TokenBatch32& batch, std::size_t max_items) {
    std::ostringstream out;
    const std::size_t total = batch.tokens.size();
    const std::size_t limit = std::min(total, max_items);
    out << "{"
        << "\"batch_size\": " << batch.batch_size << ", "
        << "\"seq_len\": " << batch.seq_len << ", "
        << "\"items\": " << total << ", "
        << "\"dtype\": \"uint32\", "
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

std::string byte_batch_json(const ByteBatch& batch, std::size_t max_items) {
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
        out << static_cast<unsigned int>(batch.tokens[i]);
    }
    out << "], \"targets\": [";
    for (std::size_t i = 0; i < limit; ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << static_cast<unsigned int>(batch.targets[i]);
    }
    out << "]}";
    return out.str();
}

}  // namespace neuralfn::native_train
