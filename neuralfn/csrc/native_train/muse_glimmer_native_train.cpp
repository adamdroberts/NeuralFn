#include "tile_ops.h"
#include "token_shards.h"
#include "../native_gpt2/resident_sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool valid_sha256(std::string_view value);

constexpr int kHostToDevice = 1;
constexpr int kDeviceToHost = 2;
constexpr int kDeviceToDevice = 3;

struct Geometry {
    std::int64_t max_sequence = 131072;
    std::int64_t vocab = 202048;
    std::int64_t layers = 52;
    std::int64_t dim = 6656;
    std::int64_t intermediate = 19968;
    std::int64_t query_heads = 32;
    std::int64_t kv_heads = 2;
    std::int64_t head_dim = 128;
    std::int64_t window = 2048;
    float rope_theta = 500000.0f;
    float norm_eps = 1.0e-5f;
    float post_norm_eps = 1.0e-8f;
    float q_scale = 3.87f;
    float output_multiplier = 0.19611613513818404f;
    float softcap = 20.0f;

    std::int64_t query_width() const { return query_heads * head_dim; }
    std::int64_t kv_width() const { return kv_heads * head_dim; }
    bool local(std::int64_t layer) const { return layer % 4 != 3; }

    void validate() const {
        if (max_sequence <= 0 || vocab <= 1 || layers <= 0 || layers % 4 != 0 ||
            dim <= 0 || intermediate <= 0 || query_heads <= 0 || kv_heads <= 0 ||
            query_heads % kv_heads != 0 || head_dim <= 0 || head_dim % 2 != 0 ||
            window <= 0 || window > max_sequence || !std::isfinite(rope_theta) ||
            !(rope_theta > 0.0f) || !std::isfinite(norm_eps) || !(norm_eps > 0.0f) ||
            !std::isfinite(post_norm_eps) || !(post_norm_eps > 0.0f) ||
            !std::isfinite(q_scale) || !(q_scale > 0.0f) ||
            !std::isfinite(output_multiplier) || !(output_multiplier > 0.0f) ||
            !std::isfinite(softcap) || !(softcap > 0.0f)) {
            throw std::runtime_error("invalid Muse Glimmer training geometry");
        }
    }
};

struct Options {
    Geometry geometry;
    std::string checkpoint;
    std::string checkpoint_sha256;
    std::string dataset;
    std::string output_dir = "artifacts/muse-glimmer-native";
    std::string resume;
    std::string graph_fingerprint;
    std::string objective = "ar";
    std::string adapter = "none";
    std::vector<std::string> lora_targets{
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj"};
    std::string chat_template_sha256;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::int64_t max_steps = 1;
    std::int64_t batch_size = 1;
    std::int64_t sequence_length = 128;
    std::int64_t activation_checkpoint_interval = 4;
    std::int64_t checkpoint_every_steps = 1000;
    float learning_rate = 1.0e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float adam_eps = 1.0e-8f;
    float weight_decay = 0.1f;
    float max_grad_norm = 1.0f;
    std::int64_t lora_rank = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;
    std::uint64_t lora_seed = 20260813ULL;
    std::int64_t qlora_group_size = 64;
    bool allow_train_as_validation = false;
    bool kernel_check = false;
    bool print_layout = false;
    bool tiny_geometry = false;
};

std::string require_value(int argc, char** argv, int* index, const char* flag) {
    if (*index + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
    return argv[++*index];
}

std::int64_t parse_i64(const std::string& value, const char* label) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error(std::string(label) + " is not an integer");
    return static_cast<std::int64_t>(parsed);
}

float parse_float(const std::string& value, const char* label) {
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed))
        throw std::runtime_error(std::string(label) + " is not finite");
    return parsed;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --checkpoint MODEL.bf16 --checkpoint-sha256 SHA256 --dataset PATH [options]\n"
        << "Exact no-Python Muse Glimmer native C++/CUDA pretraining/SFT core.\n\n"
        << "  --kernel-check                  Validate the complete Glimmer training ABI\n"
        << "  --print-parameter-layout        Print the ordered 627-tensor production table\n"
        << "  --tile-ops-lib PATH             Native Tile-CUDA shared library\n"
        << "  --cuda-runtime-lib PATH         CUDA runtime shared library\n"
        << "  --max-steps N --batch-size N --sequence-length N\n"
        << "  --objective {ar,sft}           Flat pretraining or structured masked SFT\n"
        << "  --adapter {none,lora,qlora}    Full update, LoRA, or frozen NF4-base LoRA\n"
        << "  --lora-targets LIST            Comma-separated exact Glimmer projection roles\n"
        << "  --lora-rank N --lora-alpha F --lora-dropout F --lora-seed N\n"
        << "  --qlora-group-size 64          Canonical self-contained NF4 group layout\n"
        << "  --chat-template-sha256 SHA256  Required exact ATEM hash for SFT\n"
        << "  --activation-checkpoint-interval N\n"
        << "  --learning-rate F --weight-decay F --max-grad-norm F\n"
        << "  --output-dir PATH --resume-from-checkpoint PATH\n"
        << "  --tiny-geometry D,F,QH,KVH,HD,L,V,W,MAX (test/oracle only)\n";
}

Options parse_options(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        if (arg == "--checkpoint") out.checkpoint = require_value(argc, argv, &i, "--checkpoint");
        else if (arg == "--checkpoint-sha256") out.checkpoint_sha256 = require_value(argc, argv, &i, "--checkpoint-sha256");
        else if (arg == "--dataset") out.dataset = require_value(argc, argv, &i, "--dataset");
        else if (arg == "--output-dir") out.output_dir = require_value(argc, argv, &i, "--output-dir");
        else if (arg == "--resume-from-checkpoint") out.resume = require_value(argc, argv, &i, "--resume-from-checkpoint");
        else if (arg == "--graph-fingerprint") out.graph_fingerprint = require_value(argc, argv, &i, "--graph-fingerprint");
        else if (arg == "--objective") out.objective = require_value(argc, argv, &i, "--objective");
        else if (arg == "--adapter") out.adapter = require_value(argc, argv, &i, "--adapter");
        else if (arg == "--lora-targets") {
            out.lora_targets.clear();
            const std::string raw = require_value(argc, argv, &i, "--lora-targets");
            std::size_t start = 0;
            while (start <= raw.size()) {
                const std::size_t comma = raw.find(',', start);
                const std::string value = raw.substr(start, comma - start);
                if (!value.empty()) out.lora_targets.push_back(value);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        else if (arg == "--lora-rank") out.lora_rank = parse_i64(require_value(argc, argv, &i, "--lora-rank"), "LoRA rank");
        else if (arg == "--lora-alpha") out.lora_alpha = parse_float(require_value(argc, argv, &i, "--lora-alpha"), "LoRA alpha");
        else if (arg == "--lora-dropout") out.lora_dropout = parse_float(require_value(argc, argv, &i, "--lora-dropout"), "LoRA dropout");
        else if (arg == "--lora-seed") {
            const std::string value = require_value(argc, argv, &i, "--lora-seed");
            std::size_t consumed = 0;
            out.lora_seed = std::stoull(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("LoRA seed is not an integer");
        }
        else if (arg == "--qlora-group-size") out.qlora_group_size = parse_i64(
            require_value(argc, argv, &i, "--qlora-group-size"), "QLoRA group size");
        else if (arg == "--chat-template-sha256") out.chat_template_sha256 = require_value(argc, argv, &i, "--chat-template-sha256");
        else if (arg == "--tile-ops-lib") out.tile_ops_lib = require_value(argc, argv, &i, "--tile-ops-lib");
        else if (arg == "--cuda-runtime-lib") out.cuda_runtime_lib = require_value(argc, argv, &i, "--cuda-runtime-lib");
        else if (arg == "--max-steps") out.max_steps = parse_i64(require_value(argc, argv, &i, "--max-steps"), "max steps");
        else if (arg == "--batch-size") out.batch_size = parse_i64(require_value(argc, argv, &i, "--batch-size"), "batch size");
        else if (arg == "--sequence-length") out.sequence_length = parse_i64(require_value(argc, argv, &i, "--sequence-length"), "sequence length");
        else if (arg == "--activation-checkpoint-interval") out.activation_checkpoint_interval = parse_i64(require_value(argc, argv, &i, "--activation-checkpoint-interval"), "activation checkpoint interval");
        else if (arg == "--checkpoint-every-steps") out.checkpoint_every_steps = parse_i64(require_value(argc, argv, &i, "--checkpoint-every-steps"), "checkpoint interval");
        else if (arg == "--learning-rate") out.learning_rate = parse_float(require_value(argc, argv, &i, "--learning-rate"), "learning rate");
        else if (arg == "--weight-decay") out.weight_decay = parse_float(require_value(argc, argv, &i, "--weight-decay"), "weight decay");
        else if (arg == "--max-grad-norm") out.max_grad_norm = parse_float(require_value(argc, argv, &i, "--max-grad-norm"), "max grad norm");
        else if (arg == "--allow-train-as-validation") out.allow_train_as_validation = true;
        else if (arg == "--kernel-check") out.kernel_check = true;
        else if (arg == "--print-parameter-layout") out.print_layout = true;
        else if (arg == "--tiny-geometry") {
            out.tiny_geometry = true;
            std::string raw = require_value(argc, argv, &i, "--tiny-geometry");
            std::vector<std::int64_t> values;
            std::size_t start = 0;
            while (start <= raw.size()) {
                const std::size_t comma = raw.find(',', start);
                values.push_back(parse_i64(raw.substr(start, comma - start), "tiny geometry"));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (values.size() != 9) throw std::runtime_error("--tiny-geometry needs D,F,QH,KVH,HD,L,V,W,MAX");
            out.geometry.dim = values[0]; out.geometry.intermediate = values[1];
            out.geometry.query_heads = values[2]; out.geometry.kv_heads = values[3];
            out.geometry.head_dim = values[4]; out.geometry.layers = values[5];
            out.geometry.vocab = values[6]; out.geometry.window = values[7];
            out.geometry.max_sequence = values[8];
        } else {
            throw std::runtime_error("unknown Muse Glimmer native trainer option: " + arg);
        }
    }
    out.geometry.validate();
    if (out.max_steps <= 0 || out.batch_size <= 0 || out.sequence_length <= 0 ||
        out.sequence_length > out.geometry.max_sequence ||
        out.activation_checkpoint_interval <= 0 || out.learning_rate < 0.0f ||
        out.checkpoint_every_steps <= 0 ||
        out.weight_decay < 0.0f || !(out.max_grad_norm > 0.0f)) {
        throw std::runtime_error("invalid Muse Glimmer training options");
    }
    if (out.objective != "ar" && out.objective != "sft") {
        throw std::runtime_error("--objective must be ar or sft");
    }
    if (out.adapter != "none" && out.adapter != "lora" && out.adapter != "qlora") {
        throw std::runtime_error("--adapter must be none, lora, or qlora");
    }
    const std::unordered_set<std::string> allowed_lora_targets{
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj"};
    std::unordered_set<std::string> unique_lora_targets;
    for (const auto& target : out.lora_targets) {
        if (!allowed_lora_targets.contains(target))
            throw std::runtime_error("unsupported Muse Glimmer LoRA target: " + target);
        if (!unique_lora_targets.insert(target).second)
            throw std::runtime_error("duplicate Muse Glimmer LoRA target: " + target);
    }
    if (out.adapter != "none" && (out.objective != "sft" || out.lora_targets.empty())) {
        throw std::runtime_error("native Muse Glimmer LoRA/QLoRA requires SFT and at least one target");
    }
    if (out.adapter != "none" &&
        (out.lora_rank <= 0 || !(out.lora_alpha > 0.0f) ||
         out.lora_dropout < 0.0f || out.lora_dropout >= 1.0f)) {
        throw std::runtime_error("invalid Muse Glimmer LoRA rank/alpha/dropout");
    }
    if (out.adapter == "qlora" && out.qlora_group_size != 64) {
        throw std::runtime_error("native Muse Glimmer QLoRA requires group size 64");
    }
    if (out.adapter == "none" &&
        (out.lora_rank != 8 || out.lora_alpha != 16.0f || out.lora_dropout != 0.0f ||
         out.lora_seed != 20260813ULL || out.qlora_group_size != 64)) {
        throw std::runtime_error("LoRA tuning parameters require --adapter lora or qlora");
    }
    if (out.objective == "sft" && !valid_sha256(out.chat_template_sha256)) {
        throw std::runtime_error("--chat-template-sha256 is required for SFT");
    }
    if (out.objective == "ar" && !out.chat_template_sha256.empty()) {
        throw std::runtime_error("--chat-template-sha256 is only valid for SFT");
    }
    return out;
}

template <typename Fn>
Fn symbol(void* handle, const char* name) {
    void* raw = handle == nullptr ? nullptr : dlsym(handle, name);
    if (raw == nullptr) throw std::runtime_error(std::string("missing required symbol: ") + name);
    return reinterpret_cast<Fn>(raw);
}

struct Runtime {
    using MallocFn = int (*)(void**, std::size_t);
    using FreeFn = int (*)(void*);
    using MemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using MemsetAsyncFn = int (*)(void*, int, std::size_t, void*);
    using StreamCreateFn = int (*)(void**);
    using StreamDestroyFn = int (*)(void*);
    using StreamSyncFn = int (*)(void*);
    using ErrorFn = const char* (*)(int);

    void* handle = nullptr;
    MallocFn malloc = nullptr;
    FreeFn free = nullptr;
    MemcpyFn memcpy = nullptr;
    MemsetAsyncFn memset_async = nullptr;
    StreamCreateFn stream_create = nullptr;
    StreamDestroyFn stream_destroy = nullptr;
    StreamSyncFn stream_sync = nullptr;
    ErrorFn error_string = nullptr;
    void* stream = nullptr;

    explicit Runtime(const std::string& path) {
        const std::vector<std::string> candidates = path.empty()
            ? std::vector<std::string>{"libcudart.so", "libcudart.so.13", "libcudart.so.12"}
            : std::vector<std::string>{path};
        for (const auto& candidate : candidates) {
            handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle != nullptr) break;
        }
        if (handle == nullptr) throw std::runtime_error("failed to load CUDA runtime");
        malloc = symbol<MallocFn>(handle, "cudaMalloc");
        free = symbol<FreeFn>(handle, "cudaFree");
        memcpy = symbol<MemcpyFn>(handle, "cudaMemcpy");
        memset_async = symbol<MemsetAsyncFn>(handle, "cudaMemsetAsync");
        stream_create = symbol<StreamCreateFn>(handle, "cudaStreamCreate");
        stream_destroy = symbol<StreamDestroyFn>(handle, "cudaStreamDestroy");
        stream_sync = symbol<StreamSyncFn>(handle, "cudaStreamSynchronize");
        error_string = symbol<ErrorFn>(handle, "cudaGetErrorString");
        check(stream_create(&stream), "cudaStreamCreate");
    }
    ~Runtime() {
        if (stream && stream_destroy) stream_destroy(stream);
        if (handle) dlclose(handle);
    }
    void check(int status, const char* action) const {
        if (status != 0) {
            const char* detail = error_string ? error_string(status) : nullptr;
            throw std::runtime_error(std::string(action) + " failed" +
                (detail ? std::string(": ") + detail : std::string()));
        }
    }
    void sync() const { check(stream_sync(stream), "cudaStreamSynchronize"); }
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(Runtime& runtime, std::int64_t count) { allocate(runtime, count); }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset(); runtime_ = other.runtime_; data_ = other.data_; count_ = other.count_;
            other.runtime_ = nullptr; other.data_ = nullptr; other.count_ = 0;
        }
        return *this;
    }
    void allocate(Runtime& runtime, std::int64_t count) {
        reset();
        if (count <= 0 || static_cast<std::uint64_t>(count) >
                std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::runtime_error("invalid CUDA buffer size");
        runtime_ = &runtime; count_ = count;
        runtime.check(runtime.malloc(reinterpret_cast<void**>(&data_), bytes()), "cudaMalloc");
    }
    void reset() {
        if (data_ && runtime_) runtime_->free(data_);
        runtime_ = nullptr; data_ = nullptr; count_ = 0;
    }
    T* get() { return data_; }
    const T* get() const { return data_; }
    std::int64_t count() const { return count_; }
    std::size_t bytes() const { return static_cast<std::size_t>(count_) * sizeof(T); }
    void zero() { runtime_->check(runtime_->memset_async(data_, 0, bytes(), runtime_->stream), "cudaMemsetAsync"); }
    void upload(const T* source, std::int64_t count) {
        if (count != count_) throw std::runtime_error("CUDA upload extent mismatch");
        runtime_->check(runtime_->memcpy(data_, source, bytes(), kHostToDevice), "cudaMemcpy H2D");
    }
    void upload_range(const T* source, std::int64_t count, std::int64_t offset) {
        if (source == nullptr || count < 0 || offset < 0 ||
            offset > count_ || count > count_ - offset)
            throw std::runtime_error("CUDA ranged upload extent mismatch");
        runtime_->check(runtime_->memcpy(
            data_ + offset, source, static_cast<std::size_t>(count) * sizeof(T),
            kHostToDevice), "cudaMemcpy ranged H2D");
    }
    void download(T* target, std::int64_t count) const {
        if (count != count_) throw std::runtime_error("CUDA download extent mismatch");
        runtime_->check(runtime_->memcpy(target, data_, bytes(), kDeviceToHost), "cudaMemcpy D2H");
    }
    void copy_from(const DeviceBuffer& source) {
        if (source.count_ != count_) throw std::runtime_error("CUDA D2D extent mismatch");
        runtime_->check(runtime_->memcpy(data_, source.data_, bytes(), kDeviceToDevice), "cudaMemcpy D2D");
    }
private:
    Runtime* runtime_ = nullptr;
    T* data_ = nullptr;
    std::int64_t count_ = 0;
};

struct ParameterSpec {
    std::string name;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    bool centered = false;
    std::int64_t elements() const { return rows * cols; }
};

std::vector<ParameterSpec> parameter_specs(const Geometry& g) {
    std::vector<ParameterSpec> out;
    out.push_back({"token_embedding.weight", g.vocab, g.dim, false});
    for (std::int64_t layer = 0; layer < g.layers; ++layer) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        out.push_back({p + "input_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "post_attention_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "pre_feedforward_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "post_feedforward_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "q_proj.weight", g.query_width(), g.dim, false});
        out.push_back({p + "k_proj.weight", g.kv_width(), g.dim, false});
        out.push_back({p + "v_proj.weight", g.kv_width(), g.dim, false});
        out.push_back({p + "attn_gate_proj.weight", g.query_width(), g.dim, false});
        out.push_back({p + "o_proj.weight", g.dim, g.query_width(), false});
        out.push_back({p + "gate_proj.weight", g.intermediate, g.dim, false});
        out.push_back({p + "up_proj.weight", g.intermediate, g.dim, false});
        out.push_back({p + "down_proj.weight", g.dim, g.intermediate, false});
    }
    out.push_back({"final_norm.weight", 1, g.dim, false});
    out.push_back({"lm_head.weight", g.vocab, g.dim, false});
    return out;
}

std::int64_t parameter_elements(const Geometry& g) {
    std::int64_t total = 0;
    for (const auto& spec : parameter_specs(g)) {
        if (spec.rows > std::numeric_limits<std::int64_t>::max() / spec.cols ||
            total > std::numeric_limits<std::int64_t>::max() - spec.elements())
            throw std::runtime_error("Muse Glimmer parameter extent overflow");
        total += spec.elements();
    }
    return total;
}

std::string sha256_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file for SHA-256: " + path.string());
    neuralfn::resident_support::Sha256 hash;
    std::vector<std::uint8_t> buffer(4 * 1024 * 1024);
    while (in) {
        in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = in.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!in.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
    return hash.finish_hex();
}

bool valid_sha256(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
}

struct TileOps {
    using AbiFn = int (*)();
    using PackedLinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, const float*, const float*, float*, std::int64_t, bool);
    using PackedLinearBackwardInputFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, const float*, float*, std::int64_t);
    using EmbeddingFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, const std::int32_t*, float*, std::int64_t);
    using NormFn = int (*)(const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, std::int64_t, std::int64_t, float, bool, void*);
    using NormBackwardFn = int (*)(const NfnNativeTileGlimmerRmsNormBackwardDescriptorV1*);
    using RopeBatchFn = int (*)(float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float, std::uint32_t, bool, void*);
    using AttentionFn = int (*)(const NfnNativeTileGlimmerAttentionTrainingDescriptorV1*);
    using GateFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using GateBackwardFn = int (*)(const float*, const float*, const float*, float*, float*, std::int64_t, void*);
    using SwiGluFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using SwiGluBackwardFn = int (*)(const float*, const float*, const float*, float*, float*, std::int64_t, void*);
    using AddFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using ScaleFn = int (*)(float*, std::int64_t, float, void*);
    using CopyFn = int (*)(const float*, float*, std::int64_t, void*);
    using DropoutFn = int (*)(const float*, float*, std::int64_t, float, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LogitFn = int (*)(float*, std::int64_t, float, float, void*);
    using LogitBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, float, float, void*);
    using CeFn = int (*)(const NfnNativeTileGlimmerMaskedCeDescriptorV1*);
    using EmbeddingBackwardFn = int (*)(const std::int32_t*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using AdamFn = int (*)(std::uint16_t*, const float*, float*, float*, std::int64_t, float, float, float, float, float, std::int64_t, float, void*);
    using ErrorFn = const char* (*)(int);

    void* handle = nullptr;
    PackedLinearFn linear = nullptr;
    PackedLinearBackwardInputFn linear_backward_input = nullptr;
    EmbeddingFn embedding = nullptr;
    NormFn norm = nullptr;
    NormBackwardFn norm_backward = nullptr;
    RopeBatchFn rope = nullptr;
    AttentionFn attention_forward = nullptr;
    AttentionFn attention_backward = nullptr;
    GateFn gate = nullptr;
    GateBackwardFn gate_backward = nullptr;
    SwiGluFn swiglu = nullptr;
    SwiGluBackwardFn swiglu_backward = nullptr;
    AddFn add = nullptr;
    ScaleFn scale = nullptr;
    CopyFn copy = nullptr;
    DropoutFn dropout_forward = nullptr;
    DropoutFn dropout_backward = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    LogitFn logit = nullptr;
    LogitBackwardFn logit_backward = nullptr;
    CeFn ce = nullptr;
    EmbeddingBackwardFn embedding_backward = nullptr;
    AdamFn adam = nullptr;
    ErrorFn error_string = nullptr;

    explicit TileOps(const std::string& path) {
        const std::string resolved = !path.empty() ? path :
            (std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB")
                ? std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB")
                : "libnfn_native_train_tile_ops.so");
        handle = dlopen(resolved.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) throw std::runtime_error("failed to load Tile-CUDA ops: " + resolved);
        const auto abi = symbol<AbiFn>(handle, "nfn_native_tile_glimmer_training_abi_version");
        if (abi() != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1)
            throw std::runtime_error("Tile-CUDA Glimmer training ABI mismatch");
        linear = symbol<PackedLinearFn>(handle, "nfn_native_tile_linear_packed_weight_float32_v1");
        linear_backward_input = symbol<PackedLinearBackwardInputFn>(handle, "nfn_native_tile_linear_backward_input_packed_weight_float32_v1");
        embedding = symbol<EmbeddingFn>(handle, "nfn_native_tile_glimmer_embedding_batch_i32_float32_v1");
        norm = symbol<NormFn>(handle, "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
        norm_backward = symbol<NormBackwardFn>(handle, "nfn_native_tile_glimmer_rms_norm_backward_float32_v1");
        rope = symbol<RopeBatchFn>(handle, "nfn_native_tile_glimmer_positioned_rope_batch_float32_v1");
        attention_forward = symbol<AttentionFn>(handle, "nfn_native_tile_glimmer_attention_forward_float32_v1");
        attention_backward = symbol<AttentionFn>(handle, "nfn_native_tile_glimmer_attention_backward_float32_v1");
        gate = symbol<GateFn>(handle, "nfn_native_tile_glimmer_sigmoid_gate_float32_v1");
        gate_backward = symbol<GateBackwardFn>(handle, "nfn_native_tile_glimmer_sigmoid_gate_backward_float32_v1");
        swiglu = symbol<SwiGluFn>(handle, "nfn_native_tile_swiglu_float32");
        swiglu_backward = symbol<SwiGluBackwardFn>(handle, "nfn_native_tile_swiglu_backward_float32");
        add = symbol<AddFn>(handle, "nfn_native_tile_add_float32");
        scale = symbol<ScaleFn>(handle, "nfn_native_tile_scale_inplace_float32");
        copy = symbol<CopyFn>(handle, "nfn_native_tile_copy_float32");
        dropout_forward = symbol<DropoutFn>(handle, "nfn_native_tile_dropout_forward_float32");
        dropout_backward = symbol<DropoutFn>(handle, "nfn_native_tile_dropout_backward_float32");
        linear_backward_weight = symbol<LinearBackwardWeightFn>(handle, "nfn_native_tile_linear_backward_weight_float32");
        logit = symbol<LogitFn>(handle, "nfn_native_tile_glimmer_logit_transform_float32_v1");
        logit_backward = symbol<LogitBackwardFn>(handle, "nfn_native_tile_glimmer_logit_transform_backward_float32_v1");
        ce = symbol<CeFn>(handle, "nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1");
        embedding_backward = symbol<EmbeddingBackwardFn>(handle, "nfn_native_tile_token_embedding_backward_weight_i32_float32");
        adam = symbol<AdamFn>(handle, "nfn_native_tile_glimmer_adamw_bf16_float32_v1");
        error_string = symbol<ErrorFn>(handle, "nfn_native_tile_ops_error_string");
    }
    ~TileOps() { if (handle) dlclose(handle); }
    void check(int status, const char* operation) const {
        if (status != 0) {
            const char* detail = error_string ? error_string(status) : nullptr;
            throw std::runtime_error(std::string(operation) + " failed" +
                (detail ? std::string(": ") + detail : std::string()));
        }
    }
};

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

constexpr std::array<float, 16> kNf4Codebook{
    -1.0f, -0.6961928009986877f, -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f,
    -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f,
    0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};

std::int64_t nf4_row_stride(std::int64_t columns) {
    if (columns <= 0 ||
        (columns + 63) / 64 > std::numeric_limits<std::int64_t>::max() / 36)
        throw std::runtime_error("invalid NF4 base width");
    return ((columns + 63) / 64) * 36;
}

void quantize_nf4_rows(
    const std::uint16_t* source, std::int64_t rows, std::int64_t columns,
    std::uint8_t* output) {
    if (source == nullptr || output == nullptr || rows <= 0)
        throw std::runtime_error("invalid NF4 quantization input");
    const std::int64_t row_stride = nf4_row_stride(columns);
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::uint16_t* source_row = source + row * columns;
        std::uint8_t* output_row = output + row * row_stride;
        const std::int64_t groups = (columns + 63) / 64;
        for (std::int64_t group_index = 0; group_index < groups; ++group_index) {
            const std::uint16_t* group = source_row + group_index * 64;
            std::uint8_t* packed = output_row + group_index * 36;
            const std::int64_t group_elements = std::min<std::int64_t>(
                64, columns - group_index * 64);
            float scale = 1.0e-8f;
            for (std::int64_t index = 0; index < group_elements; ++index)
                scale = std::max(scale, std::abs(bf16_to_float(group[index])));
            std::memcpy(packed, &scale, sizeof(scale));
            std::memset(packed + 4, 0x77, 32);
            for (std::int64_t index = 0; index < group_elements; ++index) {
                const float normalized = bf16_to_float(group[index]) / scale;
                std::uint8_t best = 0;
                float best_distance = std::numeric_limits<float>::infinity();
                for (std::uint8_t code = 0; code < kNf4Codebook.size(); ++code) {
                    const float distance = std::abs(normalized - kNf4Codebook[code]);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = code;
                    }
                }
                if ((index & 1) == 0) packed[4 + index / 2] = best;
                else packed[4 + index / 2] |= static_cast<std::uint8_t>(best << 4U);
            }
        }
    }
}

struct Parameter {
    ParameterSpec spec;
    bool trainable = true;
    bool nf4 = false;
    DeviceBuffer<std::uint16_t> value;
    DeviceBuffer<std::uint8_t> packed_value;
    DeviceBuffer<float> gradient;
    DeviceBuffer<float> exp_avg;
    DeviceBuffer<float> exp_avg_sq;

    Parameter(
        Runtime& runtime, ParameterSpec input, bool should_train = true,
        bool quantize_nf4 = false)
        : spec(std::move(input)), trainable(should_train), nf4(quantize_nf4) {
        if (nf4) {
            if (trainable)
                throw std::runtime_error("NF4 base parameters must be immutable");
            const std::int64_t stride = nf4_row_stride(spec.cols);
            if (spec.rows > std::numeric_limits<std::int64_t>::max() / stride)
                throw std::runtime_error("NF4 parameter extent overflow");
            packed_value.allocate(runtime, spec.rows * stride);
        } else {
            value.allocate(runtime, spec.elements());
        }
        if (trainable) {
            gradient.allocate(runtime, spec.elements());
            exp_avg.allocate(runtime, spec.elements());
            exp_avg_sq.allocate(runtime, spec.elements());
            gradient.zero(); exp_avg.zero(); exp_avg_sq.zero();
        }
    }

    NfnNativeTilePackedWeightDescriptorV1 descriptor(void* stream) const {
        const std::int64_t row_stride = nf4 ? nf4_row_stride(spec.cols) : spec.cols * 2;
        return NfnNativeTilePackedWeightDescriptorV1{
            .struct_size = sizeof(NfnNativeTilePackedWeightDescriptorV1),
            .version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1,
            .encoding = nf4 ? NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64
                            : NFN_NATIVE_TILE_PACKED_WEIGHT_BF16,
            .flags = 0,
            .data = nf4 ? packed_value.get()
                        : reinterpret_cast<const std::uint8_t*>(value.get()),
            .data_nbytes = nf4 ? static_cast<std::int64_t>(packed_value.bytes())
                               : static_cast<std::int64_t>(value.bytes()),
            .output_dim = spec.rows,
            .input_dim = spec.cols,
            .row_stride_bytes = row_stride,
            .reserved0 = 0,
            .reserved1 = 0,
            .cuda_stream = stream,
        };
    }
};

class Parameters {
public:
    Parameters(
        Runtime& runtime, const Geometry& geometry, bool trainable = true,
        bool quantize_nf4 = false)
        : runtime_(runtime), geometry_(geometry) {
        for (auto spec : parameter_specs(geometry)) {
            const bool packed = quantize_nf4 && spec.rows > 1;
            values_.push_back(std::make_unique<Parameter>(
                runtime, std::move(spec), trainable, packed));
        }
    }

    Parameter& embedding() { return *values_.at(0); }
    Parameter& layer(std::int64_t index, std::int64_t slot) {
        return *values_.at(static_cast<std::size_t>(1 + index * 12 + slot));
    }
    Parameter& final_norm() { return *values_.at(static_cast<std::size_t>(1 + geometry_.layers * 12)); }
    Parameter& lm_head() { return *values_.at(static_cast<std::size_t>(2 + geometry_.layers * 12)); }
    const std::vector<std::unique_ptr<Parameter>>& all() const { return values_; }

    void load(const fs::path& path, std::string_view expected_sha) {
        if (!valid_sha256(expected_sha)) throw std::runtime_error("checkpoint SHA-256 must be lowercase hexadecimal");
        const std::uintmax_t expected_bytes = static_cast<std::uintmax_t>(parameter_elements(geometry_)) * 2U;
        if (!fs::is_regular_file(path) || fs::file_size(path) != expected_bytes)
            throw std::runtime_error("Muse Glimmer BF16 checkpoint has the wrong byte extent");
        if (sha256_file(path) != expected_sha)
            throw std::runtime_error("Muse Glimmer BF16 checkpoint SHA-256 mismatch");
        std::ifstream in(path, std::ios::binary);
        constexpr std::int64_t kHostChunkElements = 2 * 1024 * 1024;
        std::vector<std::uint16_t> host;
        std::vector<std::uint8_t> packed;
        for (auto& parameter : values_) {
            const std::int64_t rows_per_chunk = std::max<std::int64_t>(
                1, kHostChunkElements / parameter->spec.cols);
            for (std::int64_t row = 0; row < parameter->spec.rows; row += rows_per_chunk) {
                const std::int64_t chunk_rows = std::min(
                    rows_per_chunk, parameter->spec.rows - row);
                const std::int64_t chunk_elements = chunk_rows * parameter->spec.cols;
                host.resize(static_cast<std::size_t>(chunk_elements));
                in.read(reinterpret_cast<char*>(host.data()),
                        static_cast<std::streamsize>(host.size() * 2));
                if (!in) throw std::runtime_error("short Muse Glimmer BF16 checkpoint read");
                if (parameter->nf4) {
                    const std::int64_t stride = nf4_row_stride(parameter->spec.cols);
                    packed.resize(static_cast<std::size_t>(chunk_rows * stride));
                    quantize_nf4_rows(
                        host.data(), chunk_rows, parameter->spec.cols, packed.data());
                    parameter->packed_value.upload_range(
                        packed.data(), static_cast<std::int64_t>(packed.size()), row * stride);
                } else {
                    parameter->value.upload_range(
                        host.data(), chunk_elements, row * parameter->spec.cols);
                }
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Muse Glimmer BF16 checkpoint has trailing bytes");
    }

    void zero_grad() {
        for (auto& parameter : values_)
            if (parameter->trainable) parameter->gradient.zero();
    }

    double gradient_norm() const {
        double sum = 0.0;
        std::vector<float> host;
        for (const auto& parameter : values_) {
            if (!parameter->trainable) continue;
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->gradient.download(host.data(), parameter->spec.elements());
            for (float value : host) sum += static_cast<double>(value) * value;
        }
        return std::sqrt(sum);
    }

    void step(TileOps& ops, std::int64_t step, const Options& options, float gradient_scale) {
        for (auto& parameter : values_) {
            if (!parameter->trainable) continue;
            ops.check(ops.adam(
                parameter->value.get(), parameter->gradient.get(),
                parameter->exp_avg.get(), parameter->exp_avg_sq.get(),
                parameter->spec.elements(), options.learning_rate, options.beta1,
                options.beta2, options.adam_eps, options.weight_decay, step,
                gradient_scale, runtime_.stream), "Glimmer AdamW");
        }
    }

    void save_model(const fs::path& path) const {
        if (std::any_of(values_.begin(), values_.end(), [](const auto& parameter) {
                return parameter->nf4;
            }))
            throw std::runtime_error("NF4 QLoRA base weights are immutable and adapter-only");
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer checkpoint");
        std::vector<std::uint16_t> host;
        for (const auto& parameter : values_) {
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->value.download(host.data(), parameter->spec.elements());
            out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * 2));
            if (!out) throw std::runtime_error("failed to write native Glimmer checkpoint");
        }
        out.close();
        fs::rename(temporary, path);
    }

    void save_optimizer(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer optimizer checkpoint");
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (const auto& parameter : values_) {
                if (!parameter->trainable) continue;
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                const DeviceBuffer<float>& source = moment == 0
                    ? parameter->exp_avg : parameter->exp_avg_sq;
                source.download(host.data(), parameter->spec.elements());
                out.write(reinterpret_cast<const char*>(host.data()),
                          static_cast<std::streamsize>(host.size() * sizeof(float)));
                if (!out) throw std::runtime_error("failed to write native Glimmer optimizer checkpoint");
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_optimizer(const fs::path& path, std::string_view expected_sha) {
        std::int64_t trainable_elements = 0;
        for (const auto& parameter : values_)
            if (parameter->trainable) trainable_elements += parameter->spec.elements();
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(trainable_elements) * 2U * sizeof(float);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha)
            throw std::runtime_error("native Glimmer optimizer checkpoint authentication failed");
        std::ifstream in(path, std::ios::binary);
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (auto& parameter : values_) {
                if (!parameter->trainable) continue;
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                in.read(reinterpret_cast<char*>(host.data()),
                        static_cast<std::streamsize>(host.size() * sizeof(float)));
                if (!in) throw std::runtime_error("short native Glimmer optimizer checkpoint read");
                DeviceBuffer<float>& target = moment == 0
                    ? parameter->exp_avg : parameter->exp_avg_sq;
                target.upload(host.data(), parameter->spec.elements());
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("native Glimmer optimizer checkpoint has trailing bytes");
    }

private:
    Runtime& runtime_;
    Geometry geometry_;
    std::vector<std::unique_ptr<Parameter>> values_;
};

constexpr std::array<std::string_view, 8> kLoraRoleOrder{
    "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
    "gate_proj", "up_proj", "down_proj"};

std::string parameter_role(std::string_view name) {
    constexpr std::string_view suffix = ".weight";
    if (!name.ends_with(suffix)) return {};
    name.remove_suffix(suffix.size());
    const std::size_t dot = name.rfind('.');
    return std::string(dot == std::string_view::npos ? name : name.substr(dot + 1));
}

std::uint32_t lora_target_mask(const std::vector<std::string>& targets) {
    std::uint32_t mask = 0;
    for (const auto& target : targets) {
        const auto found = std::find(kLoraRoleOrder.begin(), kLoraRoleOrder.end(), target);
        if (found == kLoraRoleOrder.end())
            throw std::runtime_error("unsupported Muse Glimmer LoRA target: " + target);
        mask |= 1U << static_cast<unsigned>(found - kLoraRoleOrder.begin());
    }
    return mask;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

struct LoraSite {
    std::string base_name;
    std::size_t index = 0;
    Parameter a;
    Parameter b;

    LoraSite(
        Runtime& runtime, const ParameterSpec& base, std::int64_t rank,
        std::size_t site_index, std::uint64_t seed)
        : base_name(base.name), index(site_index),
          a(runtime, ParameterSpec{base.name + ".lora_A", rank, base.cols, false}),
          b(runtime, ParameterSpec{base.name + ".lora_B", base.rows, rank, false}) {
        std::vector<std::uint16_t> values(static_cast<std::size_t>(a.spec.elements()));
        const float bound = 1.0f / std::sqrt(static_cast<float>(base.cols));
        for (std::size_t element = 0; element < values.size(); ++element) {
            const std::uint64_t random = splitmix64(seed ^
                (static_cast<std::uint64_t>(site_index) << 32U) ^ element);
            const double unit = static_cast<double>(random >> 11U) *
                (1.0 / 9007199254740992.0);
            values[element] = float_to_bf16(
                static_cast<float>((2.0 * unit - 1.0) * bound));
        }
        a.value.upload(values.data(), a.spec.elements());
        values.assign(static_cast<std::size_t>(b.spec.elements()), 0U);
        b.value.upload(values.data(), b.spec.elements());
    }
};

class LoraParameters {
public:
    LoraParameters(Runtime& runtime, const Geometry& geometry, const Options& options)
        : runtime_(runtime), rank_(options.lora_rank), alpha_(options.lora_alpha),
          dropout_(options.lora_dropout), seed_(options.lora_seed),
          target_mask_(lora_target_mask(options.lora_targets)) {
        const std::unordered_set<std::string> targets(
            options.lora_targets.begin(), options.lora_targets.end());
        for (const auto& base : parameter_specs(geometry)) {
            if (!targets.contains(parameter_role(base.name))) continue;
            const std::size_t index = sites_.size();
            sites_.push_back(std::make_unique<LoraSite>(
                runtime, base, rank_, index, seed_));
            lookup_.emplace(base.name, sites_.back().get());
        }
        if (sites_.empty()) throw std::runtime_error("Muse Glimmer LoRA resolved no projection sites");
    }

    LoraSite* find(const Parameter& base) const {
        const auto found = lookup_.find(base.spec.name);
        return found == lookup_.end() ? nullptr : found->second;
    }
    float scaling() const { return alpha_ / static_cast<float>(rank_); }
    float dropout() const { return dropout_; }
    std::int64_t rank() const { return rank_; }
    float alpha() const { return alpha_; }
    std::uint64_t seed() const { return seed_; }
    std::uint32_t target_mask() const { return target_mask_; }
    std::size_t site_count() const { return sites_.size(); }
    std::uint64_t dropout_seed(const LoraSite& site, std::int64_t step) const {
        return splitmix64(seed_ ^ (static_cast<std::uint64_t>(step) << 24U) ^
                          static_cast<std::uint64_t>(site.index));
    }

    void zero_grad() {
        for (auto& site : sites_) { site->a.gradient.zero(); site->b.gradient.zero(); }
    }

    double gradient_norm() const {
        double sum = 0.0;
        std::vector<float> host;
        for (const auto& site : sites_) {
            for (const Parameter* parameter : {&site->a, &site->b}) {
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                parameter->gradient.download(host.data(), parameter->spec.elements());
                for (float value : host) sum += static_cast<double>(value) * value;
            }
        }
        return std::sqrt(sum);
    }

    void step(TileOps& ops, std::int64_t step, const Options& options, float gradient_scale) {
        for (auto& site : sites_) {
            for (Parameter* parameter : {&site->a, &site->b}) {
                ops.check(ops.adam(
                    parameter->value.get(), parameter->gradient.get(),
                    parameter->exp_avg.get(), parameter->exp_avg_sq.get(),
                    parameter->spec.elements(), options.learning_rate, options.beta1,
                    options.beta2, options.adam_eps, 0.0f, step, gradient_scale,
                    runtime_.stream), "Glimmer LoRA AdamW");
            }
        }
    }

    std::int64_t elements() const {
        std::int64_t result = 0;
        for (const auto& site : sites_)
            result += site->a.spec.elements() + site->b.spec.elements();
        return result;
    }

    void save(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer LoRA artifact");
        std::vector<std::uint16_t> host;
        for (const auto& site : sites_) {
            for (const Parameter* parameter : {&site->a, &site->b}) {
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                parameter->value.download(host.data(), parameter->spec.elements());
                out.write(reinterpret_cast<const char*>(host.data()),
                          static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
                if (!out) throw std::runtime_error("failed to write native Glimmer LoRA artifact");
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load(const fs::path& path, std::string_view expected_sha) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(elements()) * sizeof(std::uint16_t);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha)
            throw std::runtime_error("native Glimmer LoRA artifact authentication failed");
        std::ifstream in(path, std::ios::binary);
        std::vector<std::uint16_t> host;
        for (auto& site : sites_) {
            for (Parameter* parameter : {&site->a, &site->b}) {
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                in.read(reinterpret_cast<char*>(host.data()),
                        static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
                if (!in) throw std::runtime_error("short native Glimmer LoRA artifact read");
                parameter->value.upload(host.data(), parameter->spec.elements());
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("native Glimmer LoRA artifact has trailing bytes");
    }

    void save_optimizer(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer LoRA optimizer");
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (const auto& site : sites_) {
                for (const Parameter* parameter : {&site->a, &site->b}) {
                    const DeviceBuffer<float>& source = moment == 0
                        ? parameter->exp_avg : parameter->exp_avg_sq;
                    host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                    source.download(host.data(), parameter->spec.elements());
                    out.write(reinterpret_cast<const char*>(host.data()),
                              static_cast<std::streamsize>(host.size() * sizeof(float)));
                    if (!out) throw std::runtime_error("failed to write native Glimmer LoRA optimizer");
                }
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_optimizer(const fs::path& path, std::string_view expected_sha) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(elements()) * 2U * sizeof(float);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha)
            throw std::runtime_error("native Glimmer LoRA optimizer authentication failed");
        std::ifstream in(path, std::ios::binary);
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (auto& site : sites_) {
                for (Parameter* parameter : {&site->a, &site->b}) {
                    DeviceBuffer<float>& target = moment == 0
                        ? parameter->exp_avg : parameter->exp_avg_sq;
                    host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                    in.read(reinterpret_cast<char*>(host.data()),
                            static_cast<std::streamsize>(host.size() * sizeof(float)));
                    if (!in) throw std::runtime_error("short native Glimmer LoRA optimizer read");
                    target.upload(host.data(), parameter->spec.elements());
                }
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("native Glimmer LoRA optimizer has trailing bytes");
    }

    const std::vector<std::unique_ptr<LoraSite>>& sites() const { return sites_; }

private:
    Runtime& runtime_;
    std::int64_t rank_;
    float alpha_;
    float dropout_;
    std::uint64_t seed_;
    std::uint32_t target_mask_;
    std::vector<std::unique_ptr<LoraSite>> sites_;
    std::unordered_map<std::string, LoraSite*> lookup_;
};

struct LayerActivations {
    DeviceBuffer<float> input_norm;
    DeviceBuffer<float> q_raw;
    DeviceBuffer<float> q;
    DeviceBuffer<float> k_raw;
    DeviceBuffer<float> k;
    DeviceBuffer<float> v;
    DeviceBuffer<float> gate;
    DeviceBuffer<float> attention;
    DeviceBuffer<float> logsumexp;
    DeviceBuffer<float> gated_attention;
    DeviceBuffer<float> attention_output;
    DeviceBuffer<float> post_attention;
    DeviceBuffer<float> attention_residual;
    DeviceBuffer<float> pre_feedforward;
    DeviceBuffer<float> mlp_gate;
    DeviceBuffer<float> mlp_up;
    DeviceBuffer<float> swiglu;
    DeviceBuffer<float> mlp_down;
    DeviceBuffer<float> post_feedforward;
    DeviceBuffer<float> output;

    LayerActivations(Runtime& runtime, const Geometry& g, std::int64_t rows)
        : input_norm(runtime, rows * g.dim),
          q_raw(runtime, rows * g.query_width()), q(runtime, rows * g.query_width()),
          k_raw(runtime, rows * g.kv_width()), k(runtime, rows * g.kv_width()),
          v(runtime, rows * g.kv_width()), gate(runtime, rows * g.query_width()),
          attention(runtime, rows * g.query_width()),
          logsumexp(runtime, rows * g.query_heads),
          gated_attention(runtime, rows * g.query_width()),
          attention_output(runtime, rows * g.dim), post_attention(runtime, rows * g.dim),
          attention_residual(runtime, rows * g.dim),
          pre_feedforward(runtime, rows * g.dim),
          mlp_gate(runtime, rows * g.intermediate), mlp_up(runtime, rows * g.intermediate),
          swiglu(runtime, rows * g.intermediate), mlp_down(runtime, rows * g.dim),
          post_feedforward(runtime, rows * g.dim), output(runtime, rows * g.dim) {}
};

struct LayerGradients {
    DeviceBuffer<float> post_feedforward_input;
    DeviceBuffer<float> swiglu;
    DeviceBuffer<float> mlp_gate;
    DeviceBuffer<float> mlp_up;
    DeviceBuffer<float> pre_feedforward_a;
    DeviceBuffer<float> pre_feedforward_b;
    DeviceBuffer<float> pre_feedforward;
    DeviceBuffer<float> attention_residual_from_ffn;
    DeviceBuffer<float> attention_residual;
    DeviceBuffer<float> attention_output;
    DeviceBuffer<float> gated_attention;
    DeviceBuffer<float> attention;
    DeviceBuffer<float> gate;
    DeviceBuffer<float> q;
    DeviceBuffer<float> k;
    DeviceBuffer<float> v;
    DeviceBuffer<float> q_raw;
    DeviceBuffer<float> k_raw;
    DeviceBuffer<float> norm_q;
    DeviceBuffer<float> norm_k;
    DeviceBuffer<float> norm_v;
    DeviceBuffer<float> norm_gate;
    DeviceBuffer<float> norm_qk;
    DeviceBuffer<float> norm_vg;
    DeviceBuffer<float> norm_all;
    DeviceBuffer<float> input_from_norm;
    DeviceBuffer<float> input;

    LayerGradients(Runtime& runtime, const Geometry& g, std::int64_t rows)
        : post_feedforward_input(runtime, rows * g.dim),
          swiglu(runtime, rows * g.intermediate), mlp_gate(runtime, rows * g.intermediate),
          mlp_up(runtime, rows * g.intermediate),
          pre_feedforward_a(runtime, rows * g.dim), pre_feedforward_b(runtime, rows * g.dim),
          pre_feedforward(runtime, rows * g.dim),
          attention_residual_from_ffn(runtime, rows * g.dim),
          attention_residual(runtime, rows * g.dim), attention_output(runtime, rows * g.dim),
          gated_attention(runtime, rows * g.query_width()),
          attention(runtime, rows * g.query_width()), gate(runtime, rows * g.query_width()),
          q(runtime, rows * g.query_width()), k(runtime, rows * g.kv_width()),
          v(runtime, rows * g.kv_width()), q_raw(runtime, rows * g.query_width()),
          k_raw(runtime, rows * g.kv_width()), norm_q(runtime, rows * g.dim),
          norm_k(runtime, rows * g.dim), norm_v(runtime, rows * g.dim),
          norm_gate(runtime, rows * g.dim), norm_qk(runtime, rows * g.dim),
          norm_vg(runtime, rows * g.dim), norm_all(runtime, rows * g.dim),
          input_from_norm(runtime, rows * g.dim),
          input(runtime, rows * g.dim) {}
};

class GlimmerTrainer {
public:
    GlimmerTrainer(
        Runtime& runtime, TileOps& ops, Parameters& parameters,
        LoraParameters* adapters, const Options& options)
        : runtime_(runtime), ops_(ops), parameters_(parameters), adapters_(adapters), options_(options),
          g_(options.geometry), rows_(options.batch_size * options.sequence_length),
          token_ids_(runtime, rows_), targets_(runtime, rows_), mask_(runtime, rows_),
          sequence_ids_(runtime, rows_),
          raw_embedding_(runtime, rows_ * g_.dim), initial_state_(runtime, rows_ * g_.dim),
          final_norm_(runtime, rows_ * g_.dim), logits_(runtime, rows_ * g_.vocab),
          loss_rows_(runtime, rows_), grad_logits_(runtime, rows_ * g_.vocab),
          grad_raw_logits_(runtime, rows_ * g_.vocab),
          grad_final_norm_(runtime, rows_ * g_.dim), grad_final_state_(runtime, rows_ * g_.dim),
          grad_raw_embedding_(runtime, rows_ * g_.dim),
          forward_scratch_(runtime, g_, rows_), backward_scratch_(runtime, g_, rows_) {
        const std::int64_t checkpoint_count =
            (g_.layers + options_.activation_checkpoint_interval - 1) /
                options_.activation_checkpoint_interval + 1;
        for (std::int64_t i = 0; i < checkpoint_count; ++i)
            checkpoints_.push_back(std::make_unique<DeviceBuffer<float>>(runtime_, rows_ * g_.dim));
        if (adapters_ != nullptr) {
            const std::int64_t width = std::max({g_.dim, g_.intermediate, g_.query_width()});
            lora_input_.allocate(runtime_, rows_ * width);
            lora_rank_.allocate(runtime_, rows_ * adapters_->rank());
            lora_delta_.allocate(runtime_, rows_ * width);
            lora_combined_.allocate(runtime_, rows_ * width);
            lora_grad_rank_.allocate(runtime_, rows_ * adapters_->rank());
            lora_input_grad_.allocate(runtime_, rows_ * width);
            lora_dropout_grad_.allocate(runtime_, rows_ * width);
        }
    }

    double train_step(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>* loss_mask,
        const std::vector<std::int32_t>* sequence_ids,
        std::int64_t optimizer_step) {
        if (tokens.size() != static_cast<std::size_t>(rows_) ||
            targets.size() != static_cast<std::size_t>(rows_))
            throw std::runtime_error("native Glimmer batch extent mismatch");
        std::vector<std::int32_t> host_tokens(tokens.size());
        std::vector<std::int32_t> host_targets(targets.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] >= static_cast<std::uint32_t>(g_.vocab) ||
                (targets[i] != -100 &&
                 (targets[i] < 0 || targets[i] >= g_.vocab)))
                throw std::runtime_error("native Glimmer batch token is outside vocabulary");
            host_tokens[i] = static_cast<std::int32_t>(tokens[i]);
            host_targets[i] = targets[i];
        }
        std::vector<float> host_mask(tokens.size(), 1.0f);
        if (loss_mask != nullptr) {
            if (loss_mask->size() != tokens.size())
                throw std::runtime_error("native Glimmer loss mask extent mismatch");
            host_mask = *loss_mask;
        }
        double mask_count = 0.0;
        for (float value : host_mask) {
            if (!std::isfinite(value) || value < 0.0f)
                throw std::runtime_error("native Glimmer loss mask must be finite and non-negative");
            mask_count += value;
        }
        if (!(mask_count > 0.0)) throw std::runtime_error("native Glimmer loss mask is empty");
        std::vector<std::int32_t> host_sequence_ids(tokens.size(), 0);
        if (sequence_ids != nullptr) {
            if (sequence_ids->size() != tokens.size())
                throw std::runtime_error("native Glimmer sequence-id extent mismatch");
            host_sequence_ids = *sequence_ids;
        }
        for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
            std::int32_t previous = 0;
            for (std::int64_t position = 0; position < options_.sequence_length; ++position) {
                const std::size_t index = static_cast<std::size_t>(
                    batch * options_.sequence_length + position);
                const std::int32_t value = host_sequence_ids[index];
                if (value < 0 || (position == 0 && value != 0) ||
                    (position > 0 && value != previous && value != previous + 1)) {
                    throw std::runtime_error("native Glimmer packed sequence IDs are invalid");
                }
                previous = value;
            }
        }
        token_ids_.upload(host_tokens.data(), rows_);
        targets_.upload(host_targets.data(), rows_);
        mask_.upload(host_mask.data(), rows_);
        sequence_ids_.upload(host_sequence_ids.data(), rows_);
        optimizer_step_ = optimizer_step;
        parameters_.zero_grad();
        if (adapters_ != nullptr) adapters_->zero_grad();

        embed_forward();
        forward_decoder();
        const double loss = loss_forward_backward(static_cast<float>(1.0 / mask_count));
        backward_decoder();
        embed_backward();
        runtime_.sync();

        const double base_norm = parameters_.gradient_norm();
        const double adapter_norm = adapters_ == nullptr ? 0.0 : adapters_->gradient_norm();
        const double norm = std::hypot(base_norm, adapter_norm);
        if (!std::isfinite(norm)) throw std::runtime_error("native Glimmer gradient norm is not finite");
        const float clip = norm > options_.max_grad_norm
            ? static_cast<float>(options_.max_grad_norm / norm) : 1.0f;
        parameters_.step(ops_, optimizer_step, options_, clip);
        if (adapters_ != nullptr) adapters_->step(ops_, optimizer_step, options_, clip);
        runtime_.sync();
        return loss;
    }

private:
    void check(int status, const char* operation) { ops_.check(status, operation); }

    NfnNativeTilePackedWeightDescriptorV1 descriptor(Parameter& parameter) {
        return parameter.descriptor(runtime_.stream);
    }

    void linear(Parameter& parameter, const float* input, float* output, std::int64_t rows) {
        auto weight = descriptor(parameter);
        check(ops_.linear(&weight, input, nullptr, output, rows, false), "Glimmer linear forward");
        LoraSite* site = adapters_ == nullptr ? nullptr : adapters_->find(parameter);
        if (site == nullptr) return;
        const std::int64_t input_elements = rows * parameter.spec.cols;
        const std::uint64_t seed = adapters_->dropout_seed(*site, optimizer_step_);
        if (adapters_->dropout() == 0.0f) {
            check(ops_.copy(input, lora_input_.get(), input_elements, runtime_.stream),
                  "Glimmer LoRA input copy");
        } else {
            check(ops_.dropout_forward(
                input, lora_input_.get(), input_elements, adapters_->dropout(),
                static_cast<std::int64_t>(seed),
                runtime_.stream), "Glimmer LoRA dropout forward");
        }
        auto a = descriptor(site->a);
        auto b = descriptor(site->b);
        check(ops_.linear(&a, lora_input_.get(), nullptr, lora_rank_.get(), rows, false),
              "Glimmer LoRA A forward");
        check(ops_.linear(&b, lora_rank_.get(), nullptr, lora_delta_.get(), rows, false),
              "Glimmer LoRA B forward");
        const std::int64_t output_elements = rows * parameter.spec.rows;
        check(ops_.scale(lora_delta_.get(), output_elements, adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA scaling");
        check(ops_.add(output, lora_delta_.get(), lora_combined_.get(), output_elements,
                       runtime_.stream), "Glimmer LoRA residual");
        check(ops_.copy(lora_combined_.get(), output, output_elements, runtime_.stream),
              "Glimmer LoRA output copy");
    }

    void linear_backward(
        Parameter& parameter, const float* input, const float* grad_output,
        float* grad_input, std::int64_t rows) {
        auto weight = descriptor(parameter);
        check(ops_.linear_backward_input(&weight, grad_output, grad_input, rows),
              "Glimmer linear backward input");
        if (parameter.trainable) {
            check(ops_.linear_backward_weight(
                input, grad_output, parameter.gradient.get(), rows,
                parameter.spec.cols, parameter.spec.rows, runtime_.stream),
                "Glimmer linear backward weight");
        }
        LoraSite* site = adapters_ == nullptr ? nullptr : adapters_->find(parameter);
        if (site == nullptr) return;
        const std::int64_t input_elements = rows * parameter.spec.cols;
        const std::int64_t output_elements = rows * parameter.spec.rows;
        const std::uint64_t seed = adapters_->dropout_seed(*site, optimizer_step_);
        if (adapters_->dropout() == 0.0f) {
            check(ops_.copy(input, lora_input_.get(), input_elements, runtime_.stream),
                  "Glimmer LoRA backward input copy");
        } else {
            check(ops_.dropout_forward(
                input, lora_input_.get(), input_elements, adapters_->dropout(),
                static_cast<std::int64_t>(seed),
                runtime_.stream), "Glimmer LoRA backward dropout replay");
        }
        auto a = descriptor(site->a);
        auto b = descriptor(site->b);
        check(ops_.linear(&a, lora_input_.get(), nullptr, lora_rank_.get(), rows, false),
              "Glimmer LoRA A replay");
        check(ops_.linear_backward_input(&b, grad_output, lora_grad_rank_.get(), rows),
              "Glimmer LoRA B backward input");
        check(ops_.linear_backward_weight(
            lora_rank_.get(), grad_output, site->b.gradient.get(), rows,
            site->b.spec.cols, site->b.spec.rows, runtime_.stream),
            "Glimmer LoRA B backward weight");
        check(ops_.scale(site->b.gradient.get(), site->b.spec.elements(),
                         adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA B gradient scale");
        check(ops_.linear_backward_weight(
            lora_input_.get(), lora_grad_rank_.get(), site->a.gradient.get(), rows,
            site->a.spec.cols, site->a.spec.rows, runtime_.stream),
            "Glimmer LoRA A backward weight");
        check(ops_.scale(site->a.gradient.get(), site->a.spec.elements(),
                         adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA A gradient scale");
        check(ops_.linear_backward_input(&a, lora_grad_rank_.get(), lora_input_grad_.get(), rows),
              "Glimmer LoRA A backward input");
        check(ops_.scale(lora_input_grad_.get(), input_elements,
                         adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA input gradient scale");
        const float* lora_input_gradient = lora_input_grad_.get();
        if (adapters_->dropout() != 0.0f) {
            check(ops_.dropout_backward(
                lora_input_grad_.get(), lora_dropout_grad_.get(), input_elements,
                adapters_->dropout(), static_cast<std::int64_t>(seed), runtime_.stream),
                "Glimmer LoRA dropout backward");
            lora_input_gradient = lora_dropout_grad_.get();
        }
        check(ops_.add(grad_input, lora_input_gradient, lora_combined_.get(),
                       input_elements, runtime_.stream), "Glimmer LoRA input gradient residual");
        check(ops_.copy(lora_combined_.get(), grad_input, input_elements, runtime_.stream),
              "Glimmer LoRA input gradient copy");
        (void)output_elements;
    }

    void norm_forward(
        const float* input, Parameter* parameter, float* output,
        std::int64_t rows, std::int64_t width, float eps, bool centered) {
        NfnNativeTilePackedWeightDescriptorV1 weight{};
        const NfnNativeTilePackedWeightDescriptorV1* pointer = nullptr;
        if (parameter != nullptr) { weight = descriptor(*parameter); pointer = &weight; }
        check(ops_.norm(input, pointer, output, rows, width, eps, centered, runtime_.stream),
              "Glimmer RMSNorm forward");
    }

    void norm_backward(
        const float* input, Parameter* parameter, const float* grad_output,
        float* grad_input, std::int64_t rows, std::int64_t width,
        float eps, bool centered) {
        NfnNativeTilePackedWeightDescriptorV1 weight{};
        const NfnNativeTilePackedWeightDescriptorV1* pointer = nullptr;
        if (parameter != nullptr) { weight = descriptor(*parameter); pointer = &weight; }
        NfnNativeTileGlimmerRmsNormBackwardDescriptorV1 operation{
            .struct_size = sizeof(NfnNativeTileGlimmerRmsNormBackwardDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0, .reserved0 = 0, .input = input, .weight = pointer,
            .grad_output = grad_output, .grad_input = grad_input,
            .grad_weight = parameter && parameter->trainable ? parameter->gradient.get() : nullptr,
            .rows = rows, .width = width, .eps = eps,
            .centered = centered ? 1U : 0U, .cuda_stream = runtime_.stream,
        };
        check(ops_.norm_backward(&operation), "Glimmer RMSNorm backward");
    }

    void embed_forward() {
        auto embedding = descriptor(parameters_.embedding());
        check(ops_.embedding(&embedding, token_ids_.get(), raw_embedding_.get(), rows_),
              "Glimmer embedding forward");
        norm_forward(raw_embedding_.get(), nullptr, initial_state_.get(), rows_, g_.dim,
                     g_.norm_eps, false);
        checkpoints_.front()->copy_from(initial_state_);
    }

    void forward_layer(std::int64_t layer_index, const float* input, LayerActivations& a) {
        Parameter& input_norm = parameters_.layer(layer_index, 0);
        Parameter& post_attention_norm = parameters_.layer(layer_index, 1);
        Parameter& pre_ffn_norm = parameters_.layer(layer_index, 2);
        Parameter& post_ffn_norm = parameters_.layer(layer_index, 3);
        norm_forward(input, &input_norm, a.input_norm.get(), rows_, g_.dim, g_.norm_eps, true);
        linear(parameters_.layer(layer_index, 4), a.input_norm.get(), a.q_raw.get(), rows_);
        linear(parameters_.layer(layer_index, 5), a.input_norm.get(), a.k_raw.get(), rows_);
        linear(parameters_.layer(layer_index, 6), a.input_norm.get(), a.v.get(), rows_);
        linear(parameters_.layer(layer_index, 7), a.input_norm.get(), a.gate.get(), rows_);
        norm_forward(a.q_raw.get(), nullptr, a.q.get(), rows_ * g_.query_heads,
                     g_.head_dim, g_.norm_eps, false);
        norm_forward(a.k_raw.get(), nullptr, a.k.get(), rows_ * g_.kv_heads,
                     g_.head_dim, g_.norm_eps, false);
        check(ops_.scale(a.q.get(), a.q.count(), g_.q_scale, runtime_.stream),
              "Glimmer query scale");
        if (g_.local(layer_index)) {
            for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
                check(ops_.rope(
                    a.q.get() + batch * options_.sequence_length * g_.query_width(),
                    a.k.get() + batch * options_.sequence_length * g_.kv_width(),
                    options_.sequence_length, g_.query_heads, g_.kv_heads, g_.head_dim,
                    0, g_.rope_theta, NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, false,
                    runtime_.stream), "Glimmer positioned RoPE forward");
            }
        }
        NfnNativeTileGlimmerAttentionTrainingDescriptorV1 attention{
            .struct_size = sizeof(NfnNativeTileGlimmerAttentionTrainingDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL, .reserved0 = 0,
            .query = a.q.get(), .key = a.k.get(), .value = a.v.get(),
            .output = a.attention.get(), .logsumexp = a.logsumexp.get(),
            .grad_output = nullptr, .grad_query = nullptr, .grad_key = nullptr,
            .grad_value = nullptr, .batch_size = options_.batch_size,
            .sequence_length = options_.sequence_length,
            .query_heads = g_.query_heads, .kv_heads = g_.kv_heads,
            .head_dim = g_.head_dim,
            .window = g_.local(layer_index) ? g_.window : 0,
            .scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim)),
            .reserved1 = 0, .cuda_stream = runtime_.stream,
            .sequence_ids = sequence_ids_.get(), .reserved2 = 0, .reserved3 = 0,
        };
        check(ops_.attention_forward(&attention), "Glimmer GQA forward");
        check(ops_.gate(a.attention.get(), a.gate.get(), a.gated_attention.get(),
                        a.gated_attention.count(), runtime_.stream),
              "Glimmer attention gate forward");
        linear(parameters_.layer(layer_index, 8), a.gated_attention.get(),
               a.attention_output.get(), rows_);
        norm_forward(a.attention_output.get(), &post_attention_norm,
                     a.post_attention.get(), rows_, g_.dim, g_.post_norm_eps, true);
        check(ops_.add(input, a.post_attention.get(), a.attention_residual.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer attention residual");
        norm_forward(a.attention_residual.get(), &pre_ffn_norm,
                     a.pre_feedforward.get(), rows_, g_.dim, g_.norm_eps, true);
        linear(parameters_.layer(layer_index, 9), a.pre_feedforward.get(), a.mlp_gate.get(), rows_);
        linear(parameters_.layer(layer_index, 10), a.pre_feedforward.get(), a.mlp_up.get(), rows_);
        check(ops_.swiglu(a.mlp_gate.get(), a.mlp_up.get(), a.swiglu.get(),
                          a.swiglu.count(), runtime_.stream), "Glimmer SwiGLU forward");
        linear(parameters_.layer(layer_index, 11), a.swiglu.get(), a.mlp_down.get(), rows_);
        norm_forward(a.mlp_down.get(), &post_ffn_norm, a.post_feedforward.get(),
                     rows_, g_.dim, g_.post_norm_eps, true);
        check(ops_.add(a.attention_residual.get(), a.post_feedforward.get(), a.output.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer feed-forward residual");
    }

    void forward_decoder() {
        DeviceBuffer<float> current(runtime_, rows_ * g_.dim);
        DeviceBuffer<float> next(runtime_, rows_ * g_.dim);
        current.copy_from(initial_state_);
        std::int64_t checkpoint_index = 1;
        for (std::int64_t layer = 0; layer < g_.layers; ++layer) {
            forward_layer(layer, current.get(), forward_scratch_);
            runtime_.check(runtime_.memcpy(next.get(), forward_scratch_.output.get(), next.bytes(),
                                           kDeviceToDevice), "cudaMemcpy decoder state");
            std::swap(current, next);
            if ((layer + 1) % options_.activation_checkpoint_interval == 0 || layer + 1 == g_.layers)
                checkpoints_.at(static_cast<std::size_t>(checkpoint_index++))->copy_from(current);
        }
        final_decoder_state_.allocate(runtime_, rows_ * g_.dim);
        final_decoder_state_.copy_from(current);
    }

    double loss_forward_backward(float loss_gradient_scale) {
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_forward(final_decoder_state_.get(), &final_norm_parameter, final_norm_.get(),
                     rows_, g_.dim, g_.norm_eps, false);
        linear(parameters_.lm_head(), final_norm_.get(), logits_.get(), rows_);
        check(ops_.logit(logits_.get(), logits_.count(), g_.output_multiplier,
                         g_.softcap, runtime_.stream), "Glimmer logit transform");
        NfnNativeTileGlimmerMaskedCeDescriptorV1 ce{
            .struct_size = sizeof(NfnNativeTileGlimmerMaskedCeDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0, .reserved0 = 0, .transformed_logits = logits_.get(),
            .targets = targets_.get(), .loss_mask = mask_.get(),
            .row_loss = loss_rows_.get(), .grad_transformed_logits = grad_logits_.get(),
            .rows = rows_, .vocab_size = g_.vocab, .ignore_index = -100,
            .reserved1 = 0, .grad_scale = loss_gradient_scale, .reserved2 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(ops_.ce(&ce), "Glimmer masked cross entropy");
        check(ops_.logit_backward(logits_.get(), grad_logits_.get(), grad_raw_logits_.get(),
                                  logits_.count(), g_.output_multiplier, g_.softcap,
                                  runtime_.stream), "Glimmer logit transform backward");
        linear_backward(parameters_.lm_head(), final_norm_.get(), grad_raw_logits_.get(),
                        grad_final_norm_.get(), rows_);
        norm_backward(final_decoder_state_.get(), &final_norm_parameter, grad_final_norm_.get(),
                      grad_final_state_.get(), rows_, g_.dim, g_.norm_eps, false);
        runtime_.sync();
        std::vector<float> host_loss(static_cast<std::size_t>(rows_));
        std::vector<float> host_mask(static_cast<std::size_t>(rows_));
        loss_rows_.download(host_loss.data(), rows_);
        mask_.download(host_mask.data(), rows_);
        double sum = 0.0, count = 0.0;
        for (std::int64_t row = 0; row < rows_; ++row) {
            sum += host_loss[static_cast<std::size_t>(row)];
            count += host_mask[static_cast<std::size_t>(row)];
        }
        if (!(count > 0.0) || !std::isfinite(sum)) throw std::runtime_error("invalid native Glimmer loss");
        return sum / count;
    }

    void backward_layer(
        std::int64_t layer_index, const float* input, const float* grad_output,
        LayerActivations& a, LayerGradients& b) {
        // x_out = attention_residual + post_ffn(down(swiglu(...)))
        norm_backward(a.mlp_down.get(), &parameters_.layer(layer_index, 3), grad_output,
                      b.post_feedforward_input.get(), rows_, g_.dim, g_.post_norm_eps, true);
        linear_backward(parameters_.layer(layer_index, 11), a.swiglu.get(),
                        b.post_feedforward_input.get(), b.swiglu.get(), rows_);
        check(ops_.swiglu_backward(a.mlp_gate.get(), a.mlp_up.get(), b.swiglu.get(),
                                   b.mlp_gate.get(), b.mlp_up.get(), b.swiglu.count(),
                                   runtime_.stream), "Glimmer SwiGLU backward");
        linear_backward(parameters_.layer(layer_index, 9), a.pre_feedforward.get(),
                        b.mlp_gate.get(), b.pre_feedforward_a.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 10), a.pre_feedforward.get(),
                        b.mlp_up.get(), b.pre_feedforward_b.get(), rows_);
        check(ops_.add(b.pre_feedforward_a.get(), b.pre_feedforward_b.get(),
                       b.pre_feedforward.get(), rows_ * g_.dim, runtime_.stream),
              "Glimmer MLP input gradient sum");
        norm_backward(a.attention_residual.get(), &parameters_.layer(layer_index, 2),
                      b.pre_feedforward.get(), b.attention_residual_from_ffn.get(),
                      rows_, g_.dim, g_.norm_eps, true);
        check(ops_.add(grad_output, b.attention_residual_from_ffn.get(),
                       b.attention_residual.get(), rows_ * g_.dim, runtime_.stream),
              "Glimmer feed-forward residual backward");
        norm_backward(a.attention_output.get(), &parameters_.layer(layer_index, 1),
                      b.attention_residual.get(), b.attention_output.get(),
                      rows_, g_.dim, g_.post_norm_eps, true);
        linear_backward(parameters_.layer(layer_index, 8), a.gated_attention.get(),
                        b.attention_output.get(), b.gated_attention.get(), rows_);
        check(ops_.gate_backward(a.attention.get(), a.gate.get(), b.gated_attention.get(),
                                 b.attention.get(), b.gate.get(), b.gated_attention.count(),
                                 runtime_.stream), "Glimmer attention gate backward");
        NfnNativeTileGlimmerAttentionTrainingDescriptorV1 attention{
            .struct_size = sizeof(NfnNativeTileGlimmerAttentionTrainingDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL, .reserved0 = 0,
            .query = a.q.get(), .key = a.k.get(), .value = a.v.get(),
            .output = a.attention.get(), .logsumexp = a.logsumexp.get(),
            .grad_output = b.attention.get(), .grad_query = b.q.get(),
            .grad_key = b.k.get(), .grad_value = b.v.get(),
            .batch_size = options_.batch_size, .sequence_length = options_.sequence_length,
            .query_heads = g_.query_heads, .kv_heads = g_.kv_heads,
            .head_dim = g_.head_dim, .window = g_.local(layer_index) ? g_.window : 0,
            .scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim)),
            .reserved1 = 0, .cuda_stream = runtime_.stream,
            .sequence_ids = sequence_ids_.get(), .reserved2 = 0, .reserved3 = 0,
        };
        check(ops_.attention_backward(&attention), "Glimmer GQA backward");
        if (g_.local(layer_index)) {
            for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
                check(ops_.rope(
                    b.q.get() + batch * options_.sequence_length * g_.query_width(),
                    b.k.get() + batch * options_.sequence_length * g_.kv_width(),
                    options_.sequence_length, g_.query_heads, g_.kv_heads, g_.head_dim,
                    0, g_.rope_theta, NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT, true,
                    runtime_.stream), "Glimmer positioned RoPE backward");
            }
        }
        check(ops_.scale(b.q.get(), b.q.count(), g_.q_scale, runtime_.stream),
              "Glimmer query scale backward");
        norm_backward(a.q_raw.get(), nullptr, b.q.get(), b.q_raw.get(),
                      rows_ * g_.query_heads, g_.head_dim, g_.norm_eps, false);
        norm_backward(a.k_raw.get(), nullptr, b.k.get(), b.k_raw.get(),
                      rows_ * g_.kv_heads, g_.head_dim, g_.norm_eps, false);
        linear_backward(parameters_.layer(layer_index, 4), a.input_norm.get(),
                        b.q_raw.get(), b.norm_q.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 5), a.input_norm.get(),
                        b.k_raw.get(), b.norm_k.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 6), a.input_norm.get(),
                        b.v.get(), b.norm_v.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 7), a.input_norm.get(),
                        b.gate.get(), b.norm_gate.get(), rows_);
        check(ops_.add(b.norm_q.get(), b.norm_k.get(), b.norm_qk.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer QK input gradient sum");
        check(ops_.add(b.norm_v.get(), b.norm_gate.get(), b.norm_vg.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer VG input gradient sum");
        check(ops_.add(b.norm_qk.get(), b.norm_vg.get(), b.norm_all.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer attention input gradient sum");
        norm_backward(input, &parameters_.layer(layer_index, 0), b.norm_all.get(),
                      b.input_from_norm.get(), rows_, g_.dim, g_.norm_eps, true);
        check(ops_.add(b.attention_residual.get(), b.input_from_norm.get(), b.input.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer block residual backward");
    }

    void backward_decoder() {
        DeviceBuffer<float> grad_current(runtime_, rows_ * g_.dim);
        grad_current.copy_from(grad_final_state_);
        const std::int64_t interval = options_.activation_checkpoint_interval;
        const std::int64_t segments = (g_.layers + interval - 1) / interval;
        for (std::int64_t segment = segments - 1; segment >= 0; --segment) {
            const std::int64_t start = segment * interval;
            const std::int64_t end = std::min(g_.layers, start + interval);
            std::vector<std::unique_ptr<DeviceBuffer<float>>> boundaries;
            for (std::int64_t i = start; i <= end; ++i)
                boundaries.push_back(std::make_unique<DeviceBuffer<float>>(runtime_, rows_ * g_.dim));
            boundaries.front()->copy_from(*checkpoints_.at(static_cast<std::size_t>(segment)));
            for (std::int64_t layer = start; layer < end; ++layer) {
                forward_layer(layer, boundaries[static_cast<std::size_t>(layer-start)]->get(),
                              forward_scratch_);
                runtime_.check(runtime_.memcpy(
                    boundaries[static_cast<std::size_t>(layer-start+1)]->get(),
                    forward_scratch_.output.get(), boundaries.front()->bytes(), kDeviceToDevice),
                    "cudaMemcpy recomputed activation");
            }
            for (std::int64_t layer = end - 1; layer >= start; --layer) {
                const float* input = boundaries[static_cast<std::size_t>(layer-start)]->get();
                forward_layer(layer, input, forward_scratch_);
                backward_layer(layer, input, grad_current.get(), forward_scratch_, backward_scratch_);
                runtime_.check(runtime_.memcpy(
                    grad_current.get(), backward_scratch_.input.get(), grad_current.bytes(),
                    kDeviceToDevice), "cudaMemcpy decoder gradient");
            }
        }
        grad_initial_state_.allocate(runtime_, rows_ * g_.dim);
        grad_initial_state_.copy_from(grad_current);
    }

    void embed_backward() {
        if (!parameters_.embedding().trainable) return;
        norm_backward(raw_embedding_.get(), nullptr, grad_initial_state_.get(),
                      grad_raw_embedding_.get(), rows_, g_.dim, g_.norm_eps, false);
        check(ops_.embedding_backward(
            token_ids_.get(), grad_raw_embedding_.get(), parameters_.embedding().gradient.get(),
            rows_, g_.vocab, g_.dim, runtime_.stream), "Glimmer embedding backward");
    }

    Runtime& runtime_;
    TileOps& ops_;
    Parameters& parameters_;
    LoraParameters* adapters_ = nullptr;
    Options options_;
    Geometry g_;
    std::int64_t rows_;
    DeviceBuffer<std::int32_t> token_ids_;
    DeviceBuffer<std::int32_t> targets_;
    DeviceBuffer<float> mask_;
    DeviceBuffer<std::int32_t> sequence_ids_;
    DeviceBuffer<float> raw_embedding_;
    DeviceBuffer<float> initial_state_;
    DeviceBuffer<float> final_decoder_state_;
    DeviceBuffer<float> final_norm_;
    DeviceBuffer<float> logits_;
    DeviceBuffer<float> loss_rows_;
    DeviceBuffer<float> grad_logits_;
    DeviceBuffer<float> grad_raw_logits_;
    DeviceBuffer<float> grad_final_norm_;
    DeviceBuffer<float> grad_final_state_;
    DeviceBuffer<float> grad_initial_state_;
    DeviceBuffer<float> grad_raw_embedding_;
    LayerActivations forward_scratch_;
    LayerGradients backward_scratch_;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> checkpoints_;
    std::int64_t optimizer_step_ = 0;
    DeviceBuffer<float> lora_input_;
    DeviceBuffer<float> lora_rank_;
    DeviceBuffer<float> lora_delta_;
    DeviceBuffer<float> lora_combined_;
    DeviceBuffer<float> lora_grad_rank_;
    DeviceBuffer<float> lora_input_grad_;
    DeviceBuffer<float> lora_dropout_grad_;
};

std::string topology_sha256(const Geometry& g) {
    std::ostringstream canonical;
    canonical << "neuralfn.muse_glimmer_native_train.topology.v1\n"
              << g.max_sequence << ',' << g.vocab << ',' << g.layers << ',' << g.dim << ','
              << g.intermediate << ',' << g.query_heads << ',' << g.kv_heads << ','
              << g.head_dim << ',' << g.window << '\n'
              << std::setprecision(17) << g.rope_theta << ',' << g.norm_eps << ','
              << g.post_norm_eps << ',' << g.q_scale << ',' << g.output_multiplier << ','
              << g.softcap << '\n'
              << "local,local,local,global;centered-sandwich-rms;weightless-qk-rms;"
                 "sigmoid-attention-gate;swiglu;untied-head;multiplier-before-softcap\n";
    const std::string value = canonical.str();
    return neuralfn::resident_support::sha256_hex(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
}

#pragma pack(push, 1)
struct TrainerStateV2 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    char model_sha256[65];
    char optimizer_sha256[65];
    char source_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
    char objective[16];
};

struct LoraTrainerStateV1 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    std::int64_t rank;
    float alpha;
    float dropout;
    std::uint64_t seed;
    std::uint32_t target_mask;
    std::uint32_t reserved;
    char adapter_sha256[65];
    char optimizer_sha256[65];
    char source_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
};
#pragma pack(pop)

void copy_sha(char (&target)[65], std::string_view source, const char* label) {
    if (!valid_sha256(source)) throw std::runtime_error(std::string(label) + " is not a SHA-256 digest");
    std::memcpy(target, source.data(), 64);
    target[64] = '\0';
}

std::string state_sha(const char (&value)[65], const char* label) {
    if (value[64] != '\0') throw std::runtime_error(std::string("corrupt ") + label + " field");
    const std::string out(value, 64);
    if (!valid_sha256(out)) throw std::runtime_error(std::string("corrupt ") + label + " digest");
    return out;
}

std::string state_objective(const TrainerStateV2& state) {
    const auto end = std::find(std::begin(state.objective), std::end(state.objective), '\0');
    if (end == std::end(state.objective)) throw std::runtime_error("corrupt objective field");
    return std::string(std::begin(state.objective), end);
}

void validate_state_geometry(
    const TrainerStateV2& state,
    const Geometry& g,
    const Options& options) {
    if (std::memcmp(state.magic, "NFNGLIMMERTRAIN2", 16) != 0 || state.version != 2 ||
        state.header_bytes != sizeof(TrainerStateV2) ||
        state.max_sequence != g.max_sequence || state.vocab != g.vocab ||
        state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate || state.query_heads != g.query_heads ||
        state.kv_heads != g.kv_heads || state.head_dim != g.head_dim ||
        state.window != g.window || state.completed_step < 0 || state.sampler_batch < 0 ||
        state_sha(state.topology_sha256, "topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint) ||
        state_objective(state) != options.objective ||
        (options.objective == "sft" &&
         state_sha(state.chat_template_sha256, "chat template") !=
             options.chat_template_sha256)) {
        throw std::runtime_error("native Glimmer resume state topology/geometry mismatch");
    }
}

TrainerStateV2 read_trainer_state(
    const fs::path& directory,
    const Geometry& g,
    const Options& options) {
    const fs::path path = directory / "trainer_state.v2";
    if (!fs::is_regular_file(path) || fs::file_size(path) != sizeof(TrainerStateV2))
        throw std::runtime_error("native Glimmer resume state is absent or has the wrong extent");
    TrainerStateV2 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    if (!in) throw std::runtime_error("failed to read native Glimmer resume state");
    validate_state_geometry(state, g, options);
    return state;
}

void write_trainer_state(
    const fs::path& directory, const Geometry& g, std::int64_t step,
    std::int64_t sampler_batch, std::string_view model_sha,
    std::string_view optimizer_sha, std::string_view source_sha,
    std::string_view tokenizer_sha, const Options& options) {
    TrainerStateV2 state{};
    std::memcpy(state.magic, "NFNGLIMMERTRAIN2", 16);
    state.version = 2; state.header_bytes = sizeof(state);
    state.max_sequence = g.max_sequence; state.vocab = g.vocab; state.layers = g.layers;
    state.dim = g.dim; state.intermediate = g.intermediate;
    state.query_heads = g.query_heads; state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim; state.window = g.window;
    state.completed_step = step; state.sampler_batch = sampler_batch;
    copy_sha(state.model_sha256, model_sha, "model digest");
    copy_sha(state.optimizer_sha256, optimizer_sha, "optimizer digest");
    copy_sha(state.source_sha256, source_sha, "source digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "topology digest");
    copy_sha(
        state.graph_fingerprint,
        options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint,
        "graph fingerprint");
    copy_sha(
        state.chat_template_sha256,
        options.objective == "sft" ? options.chat_template_sha256 : std::string(64, '0'),
        "chat template digest");
    if (options.objective.size() >= sizeof(state.objective))
        throw std::runtime_error("native Glimmer objective is too long");
    std::memcpy(state.objective, options.objective.data(), options.objective.size());
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.v2.tmp";
    const fs::path binary = directory / "trainer_state.v2";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer trainer state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_training.v2\",\n"
            << "  \"completed_step\": " << step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"model_sha256\": \"" << model_sha << "\",\n"
            << "  \"optimizer_sha256\": \"" << optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << source_sha << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \""
            << (options.objective == "sft" ? options.chat_template_sha256 : "") << "\",\n"
            << "  \"topology_sha256\": \"" << topology << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
            << "  \"activation_checkpointing\": true,\n"
            << "  \"objective\": \"" << options.objective << "\",\n"
            << "  \"loss\": \"masked_cross_entropy\",\n"
            << "  \"packed_sequence_boundaries\": "
            << (options.objective == "sft" ? "true" : "false") << "\n}\n";
        if (!out) throw std::runtime_error("failed to write native Glimmer JSON trainer state");
    }
    fs::rename(json_tmp, json);
}

void validate_lora_state(
    const LoraTrainerStateV1& state, const Geometry& g, const Options& options) {
    if (std::memcmp(state.magic, "NFNGLIMMERLORA1", 16) != 0 || state.version != 1 ||
        state.header_bytes != sizeof(LoraTrainerStateV1) ||
        state.max_sequence != g.max_sequence || state.vocab != g.vocab ||
        state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate || state.query_heads != g.query_heads ||
        state.kv_heads != g.kv_heads || state.head_dim != g.head_dim ||
        state.window != g.window || state.completed_step < 0 || state.sampler_batch < 0 ||
        state.rank != options.lora_rank ||
        std::bit_cast<std::uint32_t>(state.alpha) != std::bit_cast<std::uint32_t>(options.lora_alpha) ||
        std::bit_cast<std::uint32_t>(state.dropout) !=
            std::bit_cast<std::uint32_t>(options.lora_dropout) ||
        state.seed != options.lora_seed || state.target_mask != lora_target_mask(options.lora_targets) ||
        state.reserved != (options.adapter == "qlora" ? 1U : 0U) ||
        state_sha(state.topology_sha256, "topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint) ||
        state_sha(state.chat_template_sha256, "chat template") !=
            options.chat_template_sha256) {
        throw std::runtime_error("native Glimmer LoRA resume state topology/configuration mismatch");
    }
}

LoraTrainerStateV1 read_lora_trainer_state(
    const fs::path& directory, const Geometry& g, const Options& options) {
    const fs::path path = directory / "trainer_state.lora.v1";
    if (!fs::is_regular_file(path) || fs::file_size(path) != sizeof(LoraTrainerStateV1))
        throw std::runtime_error("native Glimmer LoRA resume state is absent or has the wrong extent");
    LoraTrainerStateV1 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    if (!in) throw std::runtime_error("failed to read native Glimmer LoRA resume state");
    validate_lora_state(state, g, options);
    return state;
}

std::string tensor_sha256(const std::vector<std::uint16_t>& values) {
    return neuralfn::resident_support::sha256_hex(
        reinterpret_cast<const std::uint8_t*>(values.data()),
        values.size() * sizeof(std::uint16_t));
}

void write_lora_artifact_manifest(
    const fs::path& directory, const LoraParameters& adapters,
    std::string_view adapter_sha, std::string_view source_sha,
    std::string_view tokenizer_sha, const Options& options) {
    const fs::path temporary = directory / "adapter_manifest.json.tmp";
    const fs::path path = directory / "adapter_manifest.json";
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create native Glimmer LoRA manifest");
    const std::string topology = topology_sha256(options.geometry);
    out << "{\n  \"format\": \"neuralfn.native_muse_glimmer_lora.bf16.v1\",\n"
        << "  \"architecture\": \"muse_glimmer\",\n"
        << "  \"base_weight_precision\": \"bf16\",\n"
        << "  \"training_base_precision\": \""
        << (options.adapter == "qlora" ? "nf4-group64-fp32-scale" : "bf16")
        << "\",\n"
        << "  \"training_adapter\": \"" << options.adapter << "\",\n"
        << "  \"layers\": " << options.geometry.layers << ",\n"
        << "  \"hidden_size\": " << options.geometry.dim << ",\n"
        << "  \"attention_size\": " << options.geometry.query_width() << ",\n"
        << "  \"kv_size\": " << options.geometry.kv_width() << ",\n"
        << "  \"intermediate_size\": " << options.geometry.intermediate << ",\n"
        << "  \"adapter_path\": \"adapter.bf16\",\n"
        << "  \"adapter_sha256\": \"" << adapter_sha << "\",\n"
        << "  \"base_sha256\": \"" << source_sha << "\",\n"
        << "  \"graph_topology_sha256\": \"" << topology << "\",\n"
        << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
        << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
        << "  \"chat_template_sha256\": \"" << options.chat_template_sha256 << "\",\n"
        << "  \"rank\": " << adapters.rank() << ",\n"
        << "  \"alpha\": " << std::setprecision(9) << adapters.alpha() << ",\n"
        << "  \"scaling\": " << adapters.scaling() << ",\n"
        << "  \"dropout\": " << adapters.dropout() << ",\n"
        << "  \"seed\": " << adapters.seed() << ",\n"
        << "  \"dtype\": \"bfloat16\",\n"
        << "  \"targets\": [";
    for (std::size_t i = 0; i < options.lora_targets.size(); ++i) {
        if (i) out << ',';
        out << "\"" << options.lora_targets[i] << "\"";
    }
    out << "],\n  \"tensors\": [\n";
    std::int64_t offset = 0;
    bool first = true;
    std::vector<std::uint16_t> host;
    for (const auto& site : adapters.sites()) {
        for (const Parameter* parameter : {&site->a, &site->b}) {
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->value.download(host.data(), parameter->spec.elements());
            if (!first) out << ",\n";
            first = false;
            out << "    {\"name\":\"" << parameter->spec.name << "\",\"rows\":"
                << parameter->spec.rows << ",\"cols\":" << parameter->spec.cols
                << ",\"byte_offset\":" << offset << ",\"nbytes\":"
                << parameter->spec.elements() * 2 << ",\"sha256\":\""
                << tensor_sha256(host) << "\"}";
            offset += parameter->spec.elements() * 2;
        }
    }
    out << "\n  ]\n}\n";
    if (!out) throw std::runtime_error("failed to write native Glimmer LoRA manifest");
    out.close();
    fs::rename(temporary, path);
}

void write_lora_trainer_state(
    const fs::path& directory, const Geometry& g, std::int64_t step,
    std::int64_t sampler_batch, std::string_view adapter_sha,
    std::string_view optimizer_sha, std::string_view source_sha,
    std::string_view tokenizer_sha, const LoraParameters& adapters,
    const Options& options) {
    LoraTrainerStateV1 state{};
    std::memcpy(state.magic, "NFNGLIMMERLORA1", 16);
    state.version = 1; state.header_bytes = sizeof(state);
    state.max_sequence = g.max_sequence; state.vocab = g.vocab; state.layers = g.layers;
    state.dim = g.dim; state.intermediate = g.intermediate;
    state.query_heads = g.query_heads; state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim; state.window = g.window;
    state.completed_step = step; state.sampler_batch = sampler_batch;
    state.rank = adapters.rank(); state.alpha = adapters.alpha();
    state.dropout = adapters.dropout(); state.seed = adapters.seed();
    state.target_mask = adapters.target_mask();
    state.reserved = options.adapter == "qlora" ? 1U : 0U;
    copy_sha(state.adapter_sha256, adapter_sha, "adapter digest");
    copy_sha(state.optimizer_sha256, optimizer_sha, "adapter optimizer digest");
    copy_sha(state.source_sha256, source_sha, "source digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "topology digest");
    copy_sha(state.graph_fingerprint,
             options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint,
             "graph fingerprint");
    copy_sha(state.chat_template_sha256, options.chat_template_sha256, "chat template digest");
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.lora.v1.tmp";
    const fs::path binary = directory / "trainer_state.lora.v1";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer LoRA trainer state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_lora_training.v1\",\n"
            << "  \"completed_step\": " << step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"adapter_sha256\": \"" << adapter_sha << "\",\n"
            << "  \"optimizer_sha256\": \"" << optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << source_sha << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \"" << options.chat_template_sha256 << "\",\n"
            << "  \"topology_sha256\": \"" << topology << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
            << "  \"objective\": \"sft\",\n  \"adapter\": \""
            << options.adapter << "\",\n"
            << "  \"rank\": " << adapters.rank() << ",\n"
            << "  \"alpha\": " << adapters.alpha() << ",\n"
            << "  \"scaling\": " << adapters.scaling() << ",\n"
            << "  \"dropout\": " << adapters.dropout() << ",\n"
            << "  \"seed\": " << adapters.seed() << ",\n"
            << "  \"base_frozen\": true,\n  \"adapter_only\": true\n}\n";
        if (!out) throw std::runtime_error("failed to write native Glimmer LoRA JSON state");
    }
    fs::rename(json_tmp, json);
    write_lora_artifact_manifest(
        directory, adapters, adapter_sha, source_sha, tokenizer_sha, options);
}

std::string dataset_tokenizer_sha(
    const neuralfn::native_train::TokenShardDataset& dataset,
    const Geometry& geometry) {
    std::string digest;
    auto inspect = [&](const neuralfn::native_train::TokenShardFile& shard) {
        if (geometry.vocab > 65536 &&
            shard.dtype != neuralfn::native_train::TokenShardDType::uint32_le)
            throw std::runtime_error("Muse Glimmer production training requires uint32 token shards");
        if (shard.tokenizer_vocab_size != 0 &&
            shard.tokenizer_vocab_size != static_cast<std::uint32_t>(geometry.vocab))
            throw std::runtime_error("Muse Glimmer shard tokenizer vocabulary mismatch");
        if (shard.tokenizer_sha256.empty())
            throw std::runtime_error("Muse Glimmer shard is missing tokenizer SHA-256 metadata");
        if (!valid_sha256(shard.tokenizer_sha256))
            throw std::runtime_error("Muse Glimmer shard tokenizer SHA-256 is malformed");
        if (digest.empty()) digest = shard.tokenizer_sha256;
        else if (digest != shard.tokenizer_sha256)
            throw std::runtime_error("Muse Glimmer shards use different tokenizers");
    };
    for (const auto& shard : dataset.train_shards) inspect(shard);
    for (const auto& shard : dataset.val_shards) inspect(shard);
    return digest;
}

std::string structured_sft_tokenizer_sha(
    const neuralfn::native_train::StructuredSftDataset& dataset,
    const Options& options) {
    std::string digest;
    auto inspect = [&](const neuralfn::native_train::StructuredSftFile& file) {
        if (file.tokenizer_vocab_size != static_cast<std::uint32_t>(options.geometry.vocab) ||
            file.sequence_length != static_cast<std::uint32_t>(options.sequence_length) ||
            file.chat_template_sha256 != options.chat_template_sha256 ||
            !valid_sha256(file.tokenizer_sha256)) {
            throw std::runtime_error(
                "structured SFT tokenizer/template/sequence geometry mismatch");
        }
        if (digest.empty()) digest = file.tokenizer_sha256;
        else if (digest != file.tokenizer_sha256)
            throw std::runtime_error("structured SFT files use different tokenizers");
    };
    for (const auto& file : dataset.train_files) inspect(file);
    for (const auto& file : dataset.val_files) inspect(file);
    return digest;
}

void save_training_checkpoint(
    Parameters& parameters, Runtime& runtime, const Options& options,
    std::int64_t step, std::int64_t sampler_batch,
    std::string_view source_sha, std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging))
        throw std::runtime_error("refusing to replace an existing Glimmer checkpoint staging directory");
    fs::create_directories(staging);
    runtime.sync();
    parameters.save_model(staging / "model.bf16");
    parameters.save_optimizer(staging / "optimizer.f32");
    const std::string model_sha = sha256_file(staging / "model.bf16");
    const std::string optimizer_sha = sha256_file(staging / "optimizer.f32");
    write_trainer_state(staging, options.geometry, step, sampler_batch,
                        model_sha, optimizer_sha, source_sha, tokenizer_sha, options);
    if (fs::exists(root)) throw std::runtime_error("refusing to replace an existing Glimmer checkpoint");
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
              << ",\"path\":\"" << root.string() << "\",\"model_sha256\":\""
              << model_sha << "\"}\n";
}

void save_lora_checkpoint(
    LoraParameters& adapters, Runtime& runtime, const Options& options,
    std::int64_t step, std::int64_t sampler_batch,
    std::string_view source_sha, std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging))
        throw std::runtime_error("refusing to replace an existing Glimmer LoRA staging directory");
    fs::create_directories(staging);
    runtime.sync();
    adapters.save(staging / "adapter.bf16");
    adapters.save_optimizer(staging / "adapter_optimizer.f32");
    const std::string adapter_sha = sha256_file(staging / "adapter.bf16");
    const std::string optimizer_sha = sha256_file(staging / "adapter_optimizer.f32");
    write_lora_trainer_state(
        staging, options.geometry, step, sampler_batch, adapter_sha,
        optimizer_sha, source_sha, tokenizer_sha, adapters, options);
    if (fs::exists(root))
        throw std::runtime_error("refusing to replace an existing Glimmer LoRA checkpoint");
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
              << ",\"path\":\"" << root.string() << "\",\"adapter_sha256\":\""
              << adapter_sha << "\",\"adapter_only\":true}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto layout = parameter_specs(options.geometry);
        if (options.print_layout) {
            std::int64_t offset = 0;
            std::cout << "{\"schema\":\"neuralfn.muse_glimmer_parameter_layout.v1\",\"tensor_count\":"
                      << layout.size() << ",\"parameter_count\":"
                      << parameter_elements(options.geometry) << ",\"tensors\":[";
            for (std::size_t index = 0; index < layout.size(); ++index) {
                if (index) std::cout << ',';
                std::cout << "{\"name\":\"" << layout[index].name << "\",\"rows\":"
                          << layout[index].rows << ",\"cols\":" << layout[index].cols
                          << ",\"element_offset\":" << offset << "}";
                offset += layout[index].elements();
            }
            std::cout << "]}\n";
            return 0;
        }
        const std::string topology = topology_sha256(options.geometry);
        if (!options.graph_fingerprint.empty() && !valid_sha256(options.graph_fingerprint))
            throw std::runtime_error("--graph-fingerprint must be a lowercase SHA-256 digest");
        if (!options.tiny_geometry && !options.kernel_check && !options.print_layout &&
            options.graph_fingerprint.empty())
            throw std::runtime_error("production Muse Glimmer training requires --graph-fingerprint");
        TileOps ops(options.tile_ops_lib);
        if (options.kernel_check) {
            std::cout
                << "{\"schema\":\"neuralfn.muse_glimmer_native_training_kernel_check.v1\","
                << "\"passed\":true,\"training_abi\":1,\"topology_sha256\":\""
                << topology << "\",\"required_symbols\":19,\"native_lora\":true,"
                << "\"native_qlora_nf4_group64\":true}\n";
            return 0;
        }
        if (options.dataset.empty()) throw std::runtime_error("--dataset is required");
        if ((options.resume.empty() || options.adapter != "none") &&
            (options.checkpoint.empty() || options.checkpoint_sha256.empty()))
            throw std::runtime_error("--checkpoint and --checkpoint-sha256 are required for a new run");

        std::unique_ptr<neuralfn::native_train::SequentialTokenBatchSampler32> ar_sampler;
        std::unique_ptr<neuralfn::native_train::SequentialStructuredSftBatchSampler> sft_sampler;
        std::string tokenizer_sha;
        if (options.objective == "sft") {
            auto dataset = neuralfn::native_train::resolve_structured_sft_records(
                options.dataset, options.allow_train_as_validation, true);
            tokenizer_sha = structured_sft_tokenizer_sha(dataset, options);
            sft_sampler = std::make_unique<
                neuralfn::native_train::SequentialStructuredSftBatchSampler>(
                    std::move(dataset.train_files), options.batch_size);
        } else {
            auto dataset = neuralfn::native_train::resolve_token_shards(
                options.dataset, options.allow_train_as_validation, true);
            tokenizer_sha = dataset_tokenizer_sha(dataset, options.geometry);
            ar_sampler = std::make_unique<
                neuralfn::native_train::SequentialTokenBatchSampler32>(
                    std::move(dataset.train_shards), options.sequence_length,
                    options.batch_size);
        }
        const auto total_batches = [&]() -> std::int64_t {
            return sft_sampler ? sft_sampler->total_batches() : ar_sampler->total_batches();
        };
        if (total_batches() <= 0)
            throw std::runtime_error("Muse Glimmer dataset has no complete training batch");

        Runtime runtime(options.cuda_runtime_lib);
        const bool use_lora = options.adapter != "none";
        const bool use_qlora = options.adapter == "qlora";
        Parameters parameters(runtime, options.geometry, !use_lora, use_qlora);
        std::unique_ptr<LoraParameters> adapters;
        if (use_lora)
            adapters = std::make_unique<LoraParameters>(runtime, options.geometry, options);
        std::int64_t completed_step = 0;
        std::int64_t sampler_batch = 0;
        std::string source_sha = options.checkpoint_sha256;
        if (!options.resume.empty()) {
            const fs::path resume_path = fs::path(options.resume);
            if (use_lora) {
                const LoraTrainerStateV1 state = read_lora_trainer_state(
                    resume_path, options.geometry, options);
                completed_step = state.completed_step;
                sampler_batch = state.sampler_batch;
                source_sha = state_sha(state.source_sha256, "source");
                if (options.checkpoint_sha256 != source_sha)
                    throw std::runtime_error("LoRA resume base checkpoint digest conflicts with CLI pin");
                if (state_sha(state.tokenizer_sha256, "tokenizer") != tokenizer_sha)
                    throw std::runtime_error("LoRA resume tokenizer digest does not match the dataset");
                parameters.load(options.checkpoint, options.checkpoint_sha256);
                adapters->load(
                    resume_path / "adapter.bf16", state_sha(state.adapter_sha256, "adapter"));
                adapters->load_optimizer(
                    resume_path / "adapter_optimizer.f32",
                    state_sha(state.optimizer_sha256, "adapter optimizer"));
            } else {
                const TrainerStateV2 state = read_trainer_state(
                    resume_path, options.geometry, options);
                completed_step = state.completed_step;
                sampler_batch = state.sampler_batch;
                source_sha = state_sha(state.source_sha256, "source");
                if (!options.checkpoint_sha256.empty() && options.checkpoint_sha256 != source_sha)
                    throw std::runtime_error("resume source checkpoint digest conflicts with CLI pin");
                if (state_sha(state.tokenizer_sha256, "tokenizer") != tokenizer_sha)
                    throw std::runtime_error("resume tokenizer digest does not match the dataset");
                parameters.load(
                    resume_path / "model.bf16", state_sha(state.model_sha256, "model"));
                parameters.load_optimizer(
                    resume_path / "optimizer.f32", state_sha(state.optimizer_sha256, "optimizer"));
            }
            const bool restored = sft_sampler
                ? sft_sampler->seek_batch(sampler_batch % total_batches())
                : ar_sampler->seek_batch(sampler_batch % total_batches());
            if (!restored)
                throw std::runtime_error("failed to restore Muse Glimmer sampler cursor");
        } else {
            parameters.load(options.checkpoint, options.checkpoint_sha256);
        }
        GlimmerTrainer trainer(runtime, ops, parameters, adapters.get(), options);
        neuralfn::native_train::TokenBatch32 ar_batch;
        neuralfn::native_train::StructuredSftBatch sft_batch;
        if (completed_step >= options.max_steps) {
            std::cout << "{\"event\":\"complete\",\"completed_step\":"
                      << completed_step << ",\"resumed\":true}\n";
            return 0;
        }
        for (std::int64_t step = completed_step + 1; step <= options.max_steps; ++step) {
            bool available = sft_sampler ? sft_sampler->next(sft_batch) : ar_sampler->next(ar_batch);
            if (!available) {
                if (sft_sampler) sft_sampler->reset();
                else ar_sampler->reset();
                sampler_batch = 0;
                available = sft_sampler ? sft_sampler->next(sft_batch) : ar_sampler->next(ar_batch);
                if (!available)
                    throw std::runtime_error("Muse Glimmer sampler could not produce a batch");
            }
            std::vector<std::int32_t> ar_targets;
            if (!sft_sampler) {
                ar_targets.reserve(ar_batch.targets.size());
                for (std::uint32_t target : ar_batch.targets) {
                    if (target > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
                        throw std::runtime_error("AR target does not fit the native i32 loss ABI");
                    ar_targets.push_back(static_cast<std::int32_t>(target));
                }
            }
            const double loss = sft_sampler
                ? trainer.train_step(
                      sft_batch.input_ids, sft_batch.targets, &sft_batch.loss_mask,
                      &sft_batch.sequence_ids, step)
                : trainer.train_step(ar_batch.tokens, ar_targets, nullptr, nullptr, step);
            ++sampler_batch;
            std::cout << std::setprecision(10)
                      << "{\"event\":\"train_step\",\"step\":" << step
                      << ",\"loss\":" << loss << ",\"batch_tokens\":"
                      << (sft_sampler ? sft_batch.input_ids.size() : ar_batch.tokens.size())
                      << ",\"dtype\":\"uint32\",\"objective\":\""
                      << options.objective << "\","
                      << "\"adapter\":\"" << options.adapter << "\","
                      << "\"topology_sha256\":\"" << topology << "\"}\n";
            if (step == options.max_steps || step % options.checkpoint_every_steps == 0) {
                if (use_lora) {
                    save_lora_checkpoint(
                        *adapters, runtime, options, step,
                        sampler_batch % total_batches(), source_sha, tokenizer_sha);
                } else {
                    save_training_checkpoint(
                        parameters, runtime, options, step,
                        sampler_batch % total_batches(), source_sha, tokenizer_sha);
                }
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 2;
    }
}
