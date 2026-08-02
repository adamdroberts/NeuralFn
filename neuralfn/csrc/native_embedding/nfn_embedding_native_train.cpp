#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Config {
    std::string data;
    std::string data_sha256;
    std::string output_dir = "artifacts/embedding";
    std::string architecture = "bert";
    std::string stage = "pretrain";
    std::string adapter_type = "none";
    std::string pooling = "mean";
    std::string base_checkpoint;
    std::string resume_checkpoint;
    int vocab_size = 32768;
    int hidden_dim = 128;
    int output_dim = 128;
    int max_tokens = 128;
    int batch_size = 32;
    int effective_batch_size = 256;
    int max_steps = 1000;
    int checkpoint_every = 250;
    int progress_every = 10;
    int warmup_steps = 50;
    int seed = 1337;
    int lora_rank = 16;
    float learning_rate = 1e-3f;
    float weight_decay = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adam_eps = 1e-8f;
    float grad_clip = 1.0f;
    float margin = 0.2f;
    float temperature = 0.05f;
    float mlm_probability = 0.15f;
    float mlm_weight = 1.0f;
    float contrastive_weight = 1.0f;
    float lora_alpha = 32.0f;
    float lora_dropout = 0.05f;
    bool normalize = true;
    bool dry_run = false;
    bool print_command = false;
    bool print_plan = false;
    bool embed_mode = false;
    std::string embed_text;
    std::string checkpoint;
};

struct Example {
    int dataset = 0;
    std::string objective;
    float weight = 1.0f;
    float loss_weight = 1.0f;
    int label = -1;
    float score = 0.0f;
    std::vector<uint32_t> first;
    std::vector<uint32_t> second;
    std::vector<std::vector<uint32_t>> negatives;
};

struct Encoded {
    std::vector<uint32_t> tokens;
    std::vector<size_t> pooled_positions;
    std::vector<float> pooled_weights;
    std::vector<float> hidden;
    std::vector<float> pre_norm;
    std::vector<float> value;
    std::vector<float> adapter_input;
    std::vector<float> adapter_hidden;
};

std::vector<std::string> split_keep(const std::string& text, char delimiter) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream stream(text);
    while (std::getline(stream, item, delimiter)) out.push_back(item);
    if (!text.empty() && text.back() == delimiter) out.emplace_back();
    return out;
}

std::vector<uint32_t> parse_ids(const std::string& text) {
    std::vector<uint32_t> ids;
    if (text.empty()) return ids;
    for (const auto& part : split_keep(text, ',')) {
        if (!part.empty()) ids.push_back(static_cast<uint32_t>(std::stoul(part)));
    }
    return ids;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (char ch : text) {
        if (ch == '\\' || ch == '"') out << '\\' << ch;
        else if (ch == '\n') out << "\\n";
        else out << ch;
    }
    return out.str();
}

uint32_t stable_token_id(const std::string& token, int vocab_size) {
    uint32_t value = 2166136261u;
    for (unsigned char byte : token) {
        value ^= byte;
        value *= 16777619u;
    }
    return 3u + value % static_cast<uint32_t>(vocab_size - 3);
}

std::vector<uint32_t> tokenize(const std::string& text, int vocab_size, int max_tokens) {
    std::vector<uint32_t> ids;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token && static_cast<int>(ids.size()) < max_tokens) {
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        ids.push_back(stable_token_id(token, vocab_size));
    }
    if (ids.empty()) ids.push_back(2);
    return ids;
}

bool is_value_option(const std::string& arg) {
    static const std::vector<std::string> names = {
        "--embedding-data", "--embedding-data-sha256", "--output-dir", "--embedding-architecture",
        "--embedding-stage", "--adapter-type", "--pooling", "--base-checkpoint", "--resume-from-checkpoint",
        "--checkpoint", "--embed-text", "--embedding-vocab-size", "--hidden-dim", "--model-dim",
        "--embedding-dim", "--max-seq-len", "--batch-size", "--effective-batch-size", "--train-batch-records",
        "--max-steps", "--native-cuda-checkpoint-every", "--checkpoint-every-steps", "--progress-every-steps",
        "--warmup-steps", "--seed", "--lora-rank", "--learning-rate", "--weight-decay", "--beta1", "--beta2",
        "--adam-eps", "--grad-clip-norm", "--triplet-margin", "--temperature", "--mlm-probability",
        "--mlm-loss-weight", "--contrastive-loss-weight", "--lora-alpha", "--lora-dropout", "--lr-schedule",
        "--final-lr-fraction", "--eval-every-steps", "--eval-batches", "--train-log-every-steps",
        "--train-loss-every-steps", "--kernel-backend", "--backend"
    };
    return std::find(names.begin(), names.end(), arg) != names.end();
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    auto value = [&](int& index, const std::string& name) -> std::string {
        if (index + 1 >= argc) throw std::runtime_error(name + " requires a value");
        return argv[++index];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string val;
        const auto eq = arg.find('=');
        if (eq != std::string::npos) {
            val = arg.substr(eq + 1);
            arg = arg.substr(0, eq);
        }
        auto take = [&]() { return val.empty() ? value(i, arg) : val; };
        if (arg == "--embedding-data") cfg.data = take();
        else if (arg == "--embedding-data-sha256") cfg.data_sha256 = take();
        else if (arg == "--output-dir") cfg.output_dir = take();
        else if (arg == "--embedding-architecture") cfg.architecture = take();
        else if (arg == "--embedding-stage") cfg.stage = take();
        else if (arg == "--adapter-type") cfg.adapter_type = take();
        else if (arg == "--pooling") cfg.pooling = take();
        else if (arg == "--base-checkpoint") cfg.base_checkpoint = take();
        else if (arg == "--resume-from-checkpoint") cfg.resume_checkpoint = take();
        else if (arg == "--checkpoint") cfg.checkpoint = take();
        else if (arg == "--embed-text") { cfg.embed_text = take(); cfg.embed_mode = true; }
        else if (arg == "--embedding-vocab-size") cfg.vocab_size = std::stoi(take());
        else if (arg == "--hidden-dim" || arg == "--model-dim") cfg.hidden_dim = std::stoi(take());
        else if (arg == "--embedding-dim") cfg.output_dim = std::stoi(take());
        else if (arg == "--max-seq-len") cfg.max_tokens = std::stoi(take());
        else if (arg == "--batch-size") cfg.batch_size = std::stoi(take());
        else if (arg == "--effective-batch-size" || arg == "--train-batch-records") cfg.effective_batch_size = std::stoi(take());
        else if (arg == "--max-steps") cfg.max_steps = std::stoi(take());
        else if (arg == "--native-cuda-checkpoint-every" || arg == "--checkpoint-every-steps") cfg.checkpoint_every = std::stoi(take());
        else if (arg == "--progress-every-steps") cfg.progress_every = std::stoi(take());
        else if (arg == "--warmup-steps") cfg.warmup_steps = std::stoi(take());
        else if (arg == "--seed") cfg.seed = std::stoi(take());
        else if (arg == "--lora-rank") cfg.lora_rank = std::stoi(take());
        else if (arg == "--learning-rate") cfg.learning_rate = std::stof(take());
        else if (arg == "--weight-decay") cfg.weight_decay = std::stof(take());
        else if (arg == "--beta1") cfg.beta1 = std::stof(take());
        else if (arg == "--beta2") cfg.beta2 = std::stof(take());
        else if (arg == "--adam-eps") cfg.adam_eps = std::stof(take());
        else if (arg == "--grad-clip-norm") cfg.grad_clip = std::stof(take());
        else if (arg == "--triplet-margin") cfg.margin = std::stof(take());
        else if (arg == "--temperature") cfg.temperature = std::stof(take());
        else if (arg == "--mlm-probability") cfg.mlm_probability = std::stof(take());
        else if (arg == "--mlm-loss-weight") cfg.mlm_weight = std::stof(take());
        else if (arg == "--contrastive-loss-weight") cfg.contrastive_weight = std::stof(take());
        else if (arg == "--lora-alpha") cfg.lora_alpha = std::stof(take());
        else if (arg == "--lora-dropout") cfg.lora_dropout = std::stof(take());
        else if (arg == "--normalize-embeddings") cfg.normalize = true;
        else if (arg == "--no-normalize-embeddings") cfg.normalize = false;
        else if (arg == "--dry-run" || arg == "--native-cuda-dry-run") cfg.dry_run = true;
        else if (arg == "--print-command" || arg == "--native-cuda-print-command") cfg.print_command = true;
        else if (arg == "--print-plan") cfg.print_plan = true;
        else if (arg == "--embed") cfg.embed_mode = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Native NeuralFn text embedding trainer\n"
                      << "  --embedding-data PATH --embedding-stage pretrain|posttrain|finetune|resume\n"
                      << "  --embedding-architecture bert|gpt-derived --adapter-type none|lora|qlora\n"
                      << "  --checkpoint PATH --embed-text TEXT\n";
            std::exit(0);
        } else if (is_value_option(arg)) {
            (void)take(); // accepted orchestration option which does not change this compact loop
        } else if (arg == "--no-checkpoint" || arg == "--write-checkpoint") {
            // compatibility flags; final usable artifact is always written
        } else {
            throw std::runtime_error("unknown embedding trainer option: " + arg);
        }
    }
    return cfg;
}

std::vector<Example> load_data(const Config& cfg) {
    std::ifstream input(cfg.data);
    if (!input) throw std::runtime_error("cannot open compiled embedding data: " + cfg.data);
    std::vector<Example> examples;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        auto fields = split_keep(line, '\t');
        while (fields.size() < 9) fields.emplace_back();
        if (fields.size() != 9) throw std::runtime_error("invalid embedding data row " + std::to_string(line_number));
        Example ex;
        ex.dataset = std::stoi(fields[0]);
        ex.objective = fields[1];
        ex.weight = std::stof(fields[2]);
        ex.loss_weight = std::stof(fields[3]);
        ex.label = std::stoi(fields[4]);
        ex.score = std::stof(fields[5]);
        ex.first = parse_ids(fields[6]);
        ex.second = parse_ids(fields[7]);
        for (const auto& item : split_keep(fields[8], ';')) if (!item.empty()) ex.negatives.push_back(parse_ids(item));
        if (ex.first.empty()) throw std::runtime_error("empty first sequence at embedding data row " + std::to_string(line_number));
        examples.push_back(std::move(ex));
    }
    if (examples.empty()) throw std::runtime_error("compiled embedding dataset is empty");
    return examples;
}

void write_vector(std::ofstream& out, const std::vector<float>& values) {
    uint64_t size = values.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(size * sizeof(float)));
}

void read_vector(std::ifstream& in, std::vector<float>& values) {
    uint64_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    values.resize(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size * sizeof(float)));
    if (!in) throw std::runtime_error("truncated embedding checkpoint");
}

struct Model {
    Config cfg;
    uint32_t step = 0;
    std::vector<float> token;
    std::vector<float> position;
    std::vector<float> projection;
    std::vector<float> bias;
    std::vector<float> adapter_a;
    std::vector<float> adapter_b;
    std::vector<uint8_t> q_projection;
    std::vector<float> q_projection_scales;
    std::vector<float> grad_token, grad_position, grad_projection, grad_bias, grad_a, grad_b;
    std::vector<float> m_token, v_token, m_position, v_position, m_projection, v_projection, m_bias, v_bias, m_a, v_a, m_b, v_b;

    explicit Model(Config config) : cfg(std::move(config)) {
        std::mt19937 random(static_cast<uint32_t>(cfg.seed));
        std::normal_distribution<float> normal(0.0f, 0.02f);
        token.resize(static_cast<size_t>(cfg.vocab_size) * cfg.hidden_dim);
        position.resize(static_cast<size_t>(cfg.max_tokens) * cfg.hidden_dim);
        projection.resize(static_cast<size_t>(cfg.output_dim) * cfg.hidden_dim);
        bias.assign(cfg.output_dim, 0.0f);
        for (float& value : token) value = normal(random);
        for (float& value : position) value = normal(random);
        for (float& value : projection) value = normal(random) / std::sqrt(static_cast<float>(cfg.hidden_dim));
        if (cfg.adapter_type != "none") {
            adapter_a.resize(static_cast<size_t>(cfg.lora_rank) * cfg.hidden_dim);
            adapter_b.assign(static_cast<size_t>(cfg.output_dim) * cfg.lora_rank, 0.0f);
            for (float& value : adapter_a) value = normal(random);
        }
        rebuild_quantized_projection();
        allocate_optimizer();
    }

    void rebuild_quantized_projection() {
        q_projection.clear(); q_projection_scales.clear();
        if (cfg.adapter_type != "qlora") return;
        static constexpr std::array<float, 16> codebook{-1.0f, -0.6961928f, -0.5250731f, -0.3949175f, -0.2844414f, -0.1847734f, -0.0910500f, 0.0f, 0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f, 0.4407098f, 0.5626170f, 0.7229568f, 1.0f};
        constexpr size_t group_size = 64;
        q_projection.assign((projection.size() + 1) / 2, 0);
        q_projection_scales.resize((projection.size() + group_size - 1) / group_size, 1.0f);
        for (size_t group = 0; group < q_projection_scales.size(); ++group) {
            const size_t begin = group * group_size;
            const size_t end = std::min(projection.size(), begin + group_size);
            float scale = 0.0f;
            for (size_t i = begin; i < end; ++i) scale = std::max(scale, std::abs(projection[i]));
            scale = std::max(scale, 1e-12f); q_projection_scales[group] = scale;
            for (size_t i = begin; i < end; ++i) {
                const float normalized = projection[i] / scale;
                int best = 0;
                for (int code = 1; code < 16; ++code) if (std::abs(normalized - codebook[code]) < std::abs(normalized - codebook[best])) best = code;
                if ((i & 1u) == 0) q_projection[i / 2] = static_cast<uint8_t>(best);
                else q_projection[i / 2] |= static_cast<uint8_t>(best << 4);
            }
        }
    }

    void allocate_optimizer() {
        auto zeros = [](const std::vector<float>& source) { return std::vector<float>(source.size(), 0.0f); };
        grad_token = zeros(token); grad_position = zeros(position); grad_projection = zeros(projection); grad_bias = zeros(bias);
        grad_a = zeros(adapter_a); grad_b = zeros(adapter_b);
        m_token = zeros(token); v_token = zeros(token); m_position = zeros(position); v_position = zeros(position);
        m_projection = zeros(projection); v_projection = zeros(projection); m_bias = zeros(bias); v_bias = zeros(bias);
        m_a = zeros(adapter_a); v_a = zeros(adapter_a); m_b = zeros(adapter_b); v_b = zeros(adapter_b);
    }

    void initialize_adapter(const std::string& requested) {
        cfg.adapter_type = requested;
        if (requested == "none") { adapter_a.clear(); adapter_b.clear(); allocate_optimizer(); return; }
        std::mt19937 random(static_cast<uint32_t>(cfg.seed + 17));
        std::normal_distribution<float> normal(0.0f, 0.02f);
        adapter_a.resize(static_cast<size_t>(cfg.lora_rank) * cfg.hidden_dim);
        adapter_b.assign(static_cast<size_t>(cfg.output_dim) * cfg.lora_rank, 0.0f);
        for (float& value : adapter_a) value = normal(random);
        rebuild_quantized_projection();
        allocate_optimizer();
    }

    std::vector<float> base_projection() const {
        std::vector<float> result = projection;
        if (cfg.adapter_type == "qlora" && !q_projection.empty()) {
            static constexpr std::array<float, 16> codebook{-1.0f, -0.6961928f, -0.5250731f, -0.3949175f, -0.2844414f, -0.1847734f, -0.0910500f, 0.0f, 0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f, 0.4407098f, 0.5626170f, 0.7229568f, 1.0f};
            for (size_t i = 0; i < result.size(); ++i) {
                const uint8_t packed = q_projection[i / 2];
                const uint8_t code = (i & 1u) == 0 ? packed & 0x0fu : packed >> 4;
                result[i] = q_projection_scales[i / 64] * codebook[code];
            }
        }
        return result;
    }

    std::vector<float> effective_projection() const {
        std::vector<float> result = base_projection();
        if (adapter_a.empty()) return result;
        const float scale = cfg.lora_alpha / static_cast<float>(cfg.lora_rank);
        for (int o = 0; o < cfg.output_dim; ++o) for (int h = 0; h < cfg.hidden_dim; ++h) {
            float delta = 0.0f;
            for (int r = 0; r < cfg.lora_rank; ++r) delta += adapter_b[o * cfg.lora_rank + r] * adapter_a[r * cfg.hidden_dim + h];
            result[o * cfg.hidden_dim + h] += scale * delta;
        }
        return result;
    }

    Encoded encode(const std::vector<uint32_t>& ids, bool training = false) const {
        Encoded out;
        out.tokens.assign(ids.begin(), ids.begin() + std::min(ids.size(), static_cast<size_t>(cfg.max_tokens)));
        out.hidden.assign(cfg.hidden_dim, 0.0f);
        if (cfg.pooling == "cls") out.pooled_positions = {0};
        else if (cfg.pooling == "last") out.pooled_positions = {out.tokens.size() - 1};
        else { out.pooled_positions.resize(out.tokens.size()); std::iota(out.pooled_positions.begin(), out.pooled_positions.end(), 0); }
        out.pooled_weights.assign(out.pooled_positions.size(), 1.0f);
        if (cfg.architecture == "gpt-derived" && cfg.pooling == "mean") {
            for (size_t i = 0; i < out.pooled_positions.size(); ++i) out.pooled_weights[i] = static_cast<float>(i + 1);
        }
        const float weight_sum = std::accumulate(out.pooled_weights.begin(), out.pooled_weights.end(), 0.0f);
        for (size_t i = 0; i < out.pooled_positions.size(); ++i) {
            const size_t p = out.pooled_positions[i];
            const float pool_weight = out.pooled_weights[i] / std::max(weight_sum, 1e-12f);
            const size_t row = static_cast<size_t>(out.tokens[p] % cfg.vocab_size) * cfg.hidden_dim;
            const size_t pos = p * cfg.hidden_dim;
            for (int h = 0; h < cfg.hidden_dim; ++h) out.hidden[h] += pool_weight * (token[row + h] + position[pos + h]);
        }
        auto matrix = base_projection();
        out.pre_norm.assign(cfg.output_dim, 0.0f);
        for (int o = 0; o < cfg.output_dim; ++o) {
            float sum = bias[o];
            for (int h = 0; h < cfg.hidden_dim; ++h) sum += matrix[o * cfg.hidden_dim + h] * out.hidden[h];
            out.pre_norm[o] = sum;
        }
        if (!adapter_a.empty()) {
            out.adapter_input = out.hidden;
            if (training && cfg.lora_dropout > 0.0f) {
                const float keep_probability = 1.0f - cfg.lora_dropout;
                for (int h = 0; h < cfg.hidden_dim; ++h) {
                    uint32_t bits = static_cast<uint32_t>((step + 1u) * 2654435761u) ^ static_cast<uint32_t>((h + 1) * 2246822519u) ^ out.tokens.front();
                    const float sample = static_cast<float>(bits & 0x00ffffffu) / static_cast<float>(0x01000000u);
                    out.adapter_input[h] = sample < cfg.lora_dropout ? 0.0f : out.adapter_input[h] / keep_probability;
                }
            }
            out.adapter_hidden.assign(cfg.lora_rank, 0.0f);
            for (int r = 0; r < cfg.lora_rank; ++r) for (int h = 0; h < cfg.hidden_dim; ++h) out.adapter_hidden[r] += adapter_a[r * cfg.hidden_dim + h] * out.adapter_input[h];
            const float scale = cfg.lora_alpha / static_cast<float>(cfg.lora_rank);
            for (int o = 0; o < cfg.output_dim; ++o) for (int r = 0; r < cfg.lora_rank; ++r) out.pre_norm[o] += scale * adapter_b[o * cfg.lora_rank + r] * out.adapter_hidden[r];
        }
        out.value = out.pre_norm;
        if (cfg.normalize) {
            float norm = 0.0f;
            for (float value : out.value) norm += value * value;
            norm = std::sqrt(std::max(norm, 1e-12f));
            for (float& value : out.value) value /= norm;
        }
        return out;
    }

    void backward(const Encoded& encoded, const std::vector<float>& grad_value) {
        std::vector<float> grad_pre = grad_value;
        if (cfg.normalize) {
            float norm = 0.0f;
            for (float value : encoded.pre_norm) norm += value * value;
            norm = std::sqrt(std::max(norm, 1e-12f));
            float dot = 0.0f;
            for (int o = 0; o < cfg.output_dim; ++o) dot += grad_value[o] * encoded.value[o];
            for (int o = 0; o < cfg.output_dim; ++o) grad_pre[o] = (grad_value[o] - encoded.value[o] * dot) / norm;
        }
        std::vector<float> grad_hidden(cfg.hidden_dim, 0.0f);
        auto matrix = base_projection();
        const bool adapters = !adapter_a.empty();
        for (int o = 0; o < cfg.output_dim; ++o) {
            grad_bias[o] += grad_pre[o];
            for (int h = 0; h < cfg.hidden_dim; ++h) {
                if (!adapters) grad_projection[o * cfg.hidden_dim + h] += grad_pre[o] * encoded.hidden[h];
                grad_hidden[h] += matrix[o * cfg.hidden_dim + h] * grad_pre[o];
            }
        }
        if (adapters) {
            const float scale = cfg.lora_alpha / static_cast<float>(cfg.lora_rank);
            for (int o = 0; o < cfg.output_dim; ++o) for (int r = 0; r < cfg.lora_rank; ++r) {
                grad_b[o * cfg.lora_rank + r] += scale * grad_pre[o] * encoded.adapter_hidden[r];
                for (int h = 0; h < cfg.hidden_dim; ++h) grad_a[r * cfg.hidden_dim + h] += scale * grad_pre[o] * adapter_b[o * cfg.lora_rank + r] * encoded.adapter_input[h];
            }
        }
        if (adapters) return; // LoRA and QLoRA freeze base encoder and projection.
        const float weight_sum = std::accumulate(encoded.pooled_weights.begin(), encoded.pooled_weights.end(), 0.0f);
        for (size_t i = 0; i < encoded.pooled_positions.size(); ++i) {
            const size_t p = encoded.pooled_positions[i];
            const float pool_weight = encoded.pooled_weights[i] / std::max(weight_sum, 1e-12f);
            const size_t row = static_cast<size_t>(encoded.tokens[p] % cfg.vocab_size) * cfg.hidden_dim;
            const size_t pos = p * cfg.hidden_dim;
            for (int h = 0; h < cfg.hidden_dim; ++h) {
                grad_token[row + h] += grad_hidden[h] * pool_weight;
                grad_position[pos + h] += grad_hidden[h] * pool_weight;
            }
        }
    }

    void zero_grad() {
        for (auto* vector : {&grad_token, &grad_position, &grad_projection, &grad_bias, &grad_a, &grad_b}) std::fill(vector->begin(), vector->end(), 0.0f);
    }

    void update_vector(std::vector<float>& parameter, std::vector<float>& gradient, std::vector<float>& first, std::vector<float>& second, float lr, float scale) {
        if (parameter.empty()) return;
        const float one_minus_b1 = 1.0f - cfg.beta1;
        const float one_minus_b2 = 1.0f - cfg.beta2;
        const float correction1 = 1.0f - std::pow(cfg.beta1, static_cast<float>(step));
        const float correction2 = 1.0f - std::pow(cfg.beta2, static_cast<float>(step));
        for (size_t i = 0; i < parameter.size(); ++i) {
            const float grad = gradient[i] * scale + cfg.weight_decay * parameter[i];
            first[i] = cfg.beta1 * first[i] + one_minus_b1 * grad;
            second[i] = cfg.beta2 * second[i] + one_minus_b2 * grad * grad;
            parameter[i] -= lr * (first[i] / correction1) / (std::sqrt(second[i] / correction2) + cfg.adam_eps);
        }
    }

    void update(float lr, float batch_scale) {
        ++step;
        float norm = 0.0f;
        for (const auto* vector : {&grad_token, &grad_position, &grad_projection, &grad_bias, &grad_a, &grad_b}) for (float value : *vector) norm += value * value;
        norm = std::sqrt(norm) * batch_scale;
        const float clip = norm > cfg.grad_clip && cfg.grad_clip > 0.0f ? cfg.grad_clip / norm : 1.0f;
        const float scale = batch_scale * clip;
        if (adapter_a.empty()) {
            update_vector(token, grad_token, m_token, v_token, lr, scale);
            update_vector(position, grad_position, m_position, v_position, lr, scale);
            update_vector(projection, grad_projection, m_projection, v_projection, lr, scale);
            update_vector(bias, grad_bias, m_bias, v_bias, lr, scale);
        } else {
            update_vector(adapter_a, grad_a, m_a, v_a, lr, scale);
            update_vector(adapter_b, grad_b, m_b, v_b, lr, scale);
        }
    }

    void save(const fs::path& path, bool merged) const {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        const std::array<char, 8> magic{'N','F','N','E','M','B','1','\0'};
        out.write(magic.data(), magic.size());
        const std::array<uint32_t, 7> header{1u, static_cast<uint32_t>(cfg.vocab_size), static_cast<uint32_t>(cfg.hidden_dim), static_cast<uint32_t>(cfg.output_dim), static_cast<uint32_t>(cfg.max_tokens), step, static_cast<uint32_t>(merged ? 0 : (cfg.adapter_type == "qlora" ? 2 : cfg.adapter_type == "lora" ? 1 : 0))};
        out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size() * sizeof(uint32_t)));
        float margin_value = cfg.margin;
        out.write(reinterpret_cast<const char*>(&margin_value), sizeof(margin_value));
        const std::array<uint32_t, 4> settings{
            static_cast<uint32_t>(cfg.architecture == "gpt-derived" ? 1 : 0),
            static_cast<uint32_t>(cfg.pooling == "cls" ? 1 : cfg.pooling == "last" ? 2 : 0),
            static_cast<uint32_t>(cfg.normalize ? 1 : 0),
            static_cast<uint32_t>(cfg.lora_rank),
        };
        out.write(reinterpret_cast<const char*>(settings.data()), static_cast<std::streamsize>(settings.size() * sizeof(uint32_t)));
        out.write(reinterpret_cast<const char*>(&cfg.lora_alpha), sizeof(cfg.lora_alpha));
        out.write(reinterpret_cast<const char*>(&cfg.lora_dropout), sizeof(cfg.lora_dropout));
        write_vector(out, token); write_vector(out, position);
        write_vector(out, merged ? effective_projection() : projection); write_vector(out, bias);
        if (merged) { write_vector(out, std::vector<float>{}); write_vector(out, std::vector<float>{}); }
        else { write_vector(out, adapter_a); write_vector(out, adapter_b); }
        if (!out) throw std::runtime_error("failed writing embedding checkpoint: " + path.string());
    }

    void load(const fs::path& path) {
        fs::path resolved = fs::is_directory(path) ? path / "embedding_model.bin" : path;
        std::ifstream in(resolved, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open embedding checkpoint: " + resolved.string());
        std::array<char, 8> magic{}; in.read(magic.data(), magic.size());
        if (std::string(magic.data(), 7) != "NFNEMB1") throw std::runtime_error("not a NeuralFn embedding checkpoint: " + resolved.string());
        std::array<uint32_t, 7> header{}; in.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size() * sizeof(uint32_t)));
        float stored_margin = 0.0f; in.read(reinterpret_cast<char*>(&stored_margin), sizeof(stored_margin));
        std::array<uint32_t, 4> settings{};
        in.read(reinterpret_cast<char*>(settings.data()), static_cast<std::streamsize>(settings.size() * sizeof(uint32_t)));
        float stored_lora_alpha = 0.0f;
        in.read(reinterpret_cast<char*>(&stored_lora_alpha), sizeof(stored_lora_alpha));
        float stored_lora_dropout = 0.0f;
        in.read(reinterpret_cast<char*>(&stored_lora_dropout), sizeof(stored_lora_dropout));
        cfg.vocab_size = static_cast<int>(header[1]); cfg.hidden_dim = static_cast<int>(header[2]); cfg.output_dim = static_cast<int>(header[3]); cfg.max_tokens = static_cast<int>(header[4]); step = header[5];
        cfg.margin = stored_margin;
        cfg.architecture = settings[0] == 1 ? "gpt-derived" : "bert";
        cfg.pooling = settings[1] == 1 ? "cls" : settings[1] == 2 ? "last" : "mean";
        cfg.normalize = settings[2] != 0;
        cfg.lora_rank = static_cast<int>(settings[3]);
        cfg.lora_alpha = stored_lora_alpha;
        cfg.lora_dropout = stored_lora_dropout;
        read_vector(in, token); read_vector(in, position); read_vector(in, projection); read_vector(in, bias); read_vector(in, adapter_a); read_vector(in, adapter_b);
        if (!adapter_a.empty()) {
            cfg.adapter_type = header[6] == 2 ? "qlora" : "lora";
            cfg.lora_rank = static_cast<int>(adapter_a.size() / static_cast<size_t>(cfg.hidden_dim));
        }
        rebuild_quantized_projection();
        allocate_optimizer();
    }
};

float cosine(const Encoded& left, const Encoded& right) {
    return std::inner_product(left.value.begin(), left.value.end(), right.value.begin(), 0.0f);
}

void cosine_grads(const Encoded& left, const Encoded& right, float multiplier, std::vector<float>& grad_left, std::vector<float>& grad_right) {
    grad_left.resize(left.value.size()); grad_right.resize(right.value.size());
    for (size_t i = 0; i < left.value.size(); ++i) {
        grad_left[i] = multiplier * right.value[i];
        grad_right[i] = multiplier * left.value[i];
    }
}

struct DatasetMixer {
    std::vector<std::vector<size_t>> indices;
    std::vector<size_t> cursors;
    std::vector<double> weights;
    std::vector<double> current;
    double total = 0.0;

    explicit DatasetMixer(const std::vector<Example>& examples) {
        int max_dataset = 0;
        for (const auto& ex : examples) max_dataset = std::max(max_dataset, ex.dataset);
        indices.resize(static_cast<size_t>(max_dataset + 1)); cursors.assign(indices.size(), 0); weights.assign(indices.size(), 1.0); current.assign(indices.size(), 0.0);
        for (size_t i = 0; i < examples.size(); ++i) { indices[examples[i].dataset].push_back(i); weights[examples[i].dataset] = examples[i].weight; }
        for (size_t i = 0; i < indices.size(); ++i) if (!indices[i].empty()) total += weights[i]; else weights[i] = 0.0;
    }

    int next_dataset() {
        int selected = -1;
        for (size_t i = 0; i < weights.size(); ++i) {
            current[i] += weights[i];
            if (weights[i] > 0.0 && (selected < 0 || current[i] > current[static_cast<size_t>(selected)])) selected = static_cast<int>(i);
        }
        current[static_cast<size_t>(selected)] -= total;
        return selected;
    }

    std::vector<size_t> batch(int dataset, int count) {
        std::vector<size_t> out;
        auto& source = indices[static_cast<size_t>(dataset)];
        auto& cursor = cursors[static_cast<size_t>(dataset)];
        for (int i = 0; i < count; ++i) { out.push_back(source[cursor % source.size()]); ++cursor; }
        return out;
    }
};

void save_training_state(const Model& model, const DatasetMixer& mixer, const fs::path& path) {
    std::ofstream out(path, std::ios::binary);
    const std::array<char, 8> magic{'N','F','N','E','O','P','T','1'};
    out.write(magic.data(), magic.size());
    for (const auto* values : {&model.m_token, &model.v_token, &model.m_position, &model.v_position,
                               &model.m_projection, &model.v_projection, &model.m_bias, &model.v_bias,
                               &model.m_a, &model.v_a, &model.m_b, &model.v_b}) write_vector(out, *values);
    uint64_t count = mixer.cursors.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (size_t value : mixer.cursors) { uint64_t item = value; out.write(reinterpret_cast<const char*>(&item), sizeof(item)); }
    for (double value : mixer.current) out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("failed writing embedding optimizer state: " + path.string());
}

void load_training_state(Model& model, DatasetMixer& mixer, const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("exact embedding resume requires optimizer state: " + path.string());
    std::array<char, 8> magic{}; in.read(magic.data(), magic.size());
    if (std::string(magic.data(), magic.size()) != "NFNEOPT1") throw std::runtime_error("invalid embedding optimizer state: " + path.string());
    for (auto* values : {&model.m_token, &model.v_token, &model.m_position, &model.v_position,
                         &model.m_projection, &model.v_projection, &model.m_bias, &model.v_bias,
                         &model.m_a, &model.v_a, &model.m_b, &model.v_b}) read_vector(in, *values);
    uint64_t count = 0; in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count != mixer.cursors.size()) throw std::runtime_error("embedding resume dataset count does not match checkpoint");
    for (size_t& value : mixer.cursors) { uint64_t item = 0; in.read(reinterpret_cast<char*>(&item), sizeof(item)); value = static_cast<size_t>(item); }
    for (double& value : mixer.current) in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated embedding optimizer state: " + path.string());
}

float train_example(Model& model, const Example& ex, const Example& fallback_negative, std::unordered_map<std::string, double>& metrics) {
    Encoded first = model.encode(ex.first, true);
    std::vector<uint32_t> second_ids = ex.second.empty() ? ex.first : ex.second;
    std::vector<size_t> masked_positions;
    if (ex.objective == "raw") {
        for (size_t position = 0; position < second_ids.size(); ++position) {
            uint32_t random_bits = static_cast<uint32_t>(position * 2654435761u) ^ static_cast<uint32_t>((model.step + 1u) * 2246822519u) ^ second_ids[position];
            const float sample = static_cast<float>(random_bits & 0x00ffffffu) / static_cast<float>(0x01000000u);
            if (sample < model.cfg.mlm_probability) { masked_positions.push_back(position); second_ids[position] = 1u; }
        }
        if (masked_positions.empty()) {
            const size_t position = static_cast<size_t>((model.step + ex.first.size()) % ex.first.size());
            masked_positions.push_back(position); second_ids[position] = 1u;
        }
    }
    Encoded second = model.encode(second_ids, true);
    Encoded negative = model.encode(!ex.negatives.empty() ? ex.negatives.front() : (fallback_negative.second.empty() ? fallback_negative.first : fallback_negative.second), true);
    float loss = 0.0f;
    std::vector<float> g1, g2, gn;
    if (ex.objective == "similarity") {
        const float similarity = cosine(first, second);
        const float error = similarity - ex.score;
        loss = error * error;
        cosine_grads(first, second, 2.0f * error * ex.loss_weight, g1, g2);
        model.backward(first, g1); model.backward(second, g2);
        metrics["similarity_mse"] += loss;
    } else {
        const float positive = cosine(first, second);
        const float negative_score = cosine(first, negative);
        const float hinge = model.cfg.margin + negative_score - positive;
        const float contrastive_scale = ex.loss_weight * (ex.objective == "raw" ? model.cfg.contrastive_weight : 1.0f);
        const bool soft_contrastive = ex.objective == "raw" || ex.objective == "retrieval";
        const float temperature = std::max(model.cfg.temperature, 1e-6f);
        const float derivative = soft_contrastive
            ? 1.0f / (1.0f + std::exp(-hinge / temperature))
            : (hinge > 0.0f ? 1.0f : 0.0f);
        loss = soft_contrastive
            ? temperature * std::log1p(std::exp(std::min(hinge / temperature, 60.0f)))
            : std::max(0.0f, hinge);
        if (derivative > 0.0f) {
            cosine_grads(first, second, -contrastive_scale * derivative, g1, g2);
            cosine_grads(first, negative, contrastive_scale * derivative, g1, gn);
            // g1 receives both positive and negative contributions.
            for (size_t i = 0; i < g1.size(); ++i) g1[i] += -contrastive_scale * derivative * second.value[i];
            model.backward(first, g1); model.backward(second, g2); model.backward(negative, gn);
            if (hinge > 0.0f) metrics["margin_violations"] += 1.0;
        }
        metrics["positive_cosine"] += positive;
        metrics["negative_cosine"] += negative_score;
        if (ex.objective == "raw") {
            // Deterministic masked-token reconstruction: pull each masked token
            // row toward the masked-view context and push a stable negative row.
            float mlm_loss = 0.0f;
            for (size_t masked_position : masked_positions) {
                const uint32_t positive_id = ex.first[masked_position] % static_cast<uint32_t>(model.cfg.vocab_size);
                const uint32_t negative_id = 3u + (positive_id * 1103515245u + model.step + static_cast<uint32_t>(masked_position) + 12345u) % static_cast<uint32_t>(model.cfg.vocab_size - 3);
                float pos_dot = 0.0f, neg_dot = 0.0f;
                for (int h = 0; h < model.cfg.hidden_dim; ++h) {
                    pos_dot += second.hidden[h] * model.token[static_cast<size_t>(positive_id) * model.cfg.hidden_dim + h];
                    neg_dot += second.hidden[h] * model.token[static_cast<size_t>(negative_id) * model.cfg.hidden_dim + h];
                }
                const float mlm_hinge = std::max(0.0f, 1.0f + neg_dot - pos_dot);
                mlm_loss += mlm_hinge;
                if (mlm_hinge > 0.0f && model.adapter_a.empty()) {
                    for (int h = 0; h < model.cfg.hidden_dim; ++h) {
                        const float mask_scale = model.cfg.mlm_weight / static_cast<float>(masked_positions.size());
                        model.grad_token[static_cast<size_t>(positive_id) * model.cfg.hidden_dim + h] -= mask_scale * second.hidden[h];
                        model.grad_token[static_cast<size_t>(negative_id) * model.cfg.hidden_dim + h] += mask_scale * second.hidden[h];
                    }
                }
            }
            mlm_loss /= static_cast<float>(masked_positions.size());
            loss = model.cfg.contrastive_weight * loss + model.cfg.mlm_weight * mlm_loss;
            metrics["mlm_loss"] += mlm_loss;
        }
    }
    return loss * ex.loss_weight;
}

void save_metadata(const Model& model, const Config& cfg, const fs::path& output, const std::unordered_map<std::string, double>& metrics) {
    std::ofstream meta(output / "embedding_model.json");
    meta << "{\n"
         << "  \"format\": \"nfn_embedding_v1\",\n"
         << "  \"model_type\": \"text_embedding\",\n"
         << "  \"backend\": \"native-cpp\",\n"
         << "  \"encoder_core\": \"native_token_position_biencoder\",\n"
         << "  \"architecture_profile\": \"" << json_escape(cfg.architecture) << "\",\n"
         << "  \"stage\": \"" << json_escape(cfg.stage) << "\",\n"
         << "  \"adapter_type\": \"" << json_escape(cfg.adapter_type) << "\",\n"
         << "  \"base_weight_quantization\": \"" << (cfg.adapter_type == "qlora" ? "nf4-group64" : "none") << "\",\n"
         << "  \"pooling\": \"" << json_escape(cfg.pooling) << "\",\n"
         << "  \"normalized\": " << (cfg.normalize ? "true" : "false") << ",\n"
         << "  \"vocab_size\": " << cfg.vocab_size << ",\n"
         << "  \"hidden_dim\": " << cfg.hidden_dim << ",\n"
         << "  \"output_dim\": " << cfg.output_dim << ",\n"
         << "  \"max_tokens\": " << cfg.max_tokens << ",\n"
         << "  \"step\": " << model.step << ",\n"
         << "  \"data_sha256\": \"" << json_escape(cfg.data_sha256) << "\",\n"
         << "  \"metrics\": {";
    bool first = true;
    for (const auto& [key, value] : metrics) { if (!first) meta << ','; meta << "\n    \"" << json_escape(key) << "\": " << value; first = false; }
    if (!first) meta << '\n';
    meta << "  }\n}\n";
}

int run_embed(Config cfg) {
    if (cfg.checkpoint.empty()) throw std::runtime_error("embedding inference requires --checkpoint");
    Model model(cfg);
    model.load(cfg.checkpoint);
    auto ids = tokenize(cfg.embed_text, model.cfg.vocab_size, model.cfg.max_tokens);
    auto encoded = model.encode(ids);
    std::cout << "{\"status\":\"native-embedding-inference\",\"dimension\":" << encoded.value.size() << ",\"normalized\":" << (model.cfg.normalize ? "true" : "false") << ",\"embedding\":[";
    for (size_t i = 0; i < encoded.value.size(); ++i) { if (i) std::cout << ','; std::cout << std::setprecision(9) << encoded.value[i]; }
    std::cout << "]}\n";
    return 0;
}

int run_train(Config cfg) {
    if (cfg.data.empty()) throw std::runtime_error("embedding training requires --embedding-data");
    if (cfg.architecture != "bert" && cfg.architecture != "gpt-derived") throw std::runtime_error("embedding architecture must be bert or gpt-derived");
    if (cfg.pooling != "mean" && cfg.pooling != "cls" && cfg.pooling != "last") throw std::runtime_error("embedding pooling must be mean, cls, or last");
    if (cfg.stage != "pretrain" && cfg.stage != "posttrain" && cfg.stage != "finetune" && cfg.stage != "resume") throw std::runtime_error("embedding stage must be pretrain, posttrain, finetune, or resume");
    if (cfg.adapter_type != "none" && cfg.adapter_type != "lora" && cfg.adapter_type != "qlora") throw std::runtime_error("adapter type must be none, lora, or qlora");
    if (cfg.stage == "resume" && cfg.resume_checkpoint.empty()) throw std::runtime_error("resume stage requires --resume-from-checkpoint");
    if (cfg.stage == "finetune" && cfg.base_checkpoint.empty() && cfg.resume_checkpoint.empty()) throw std::runtime_error("finetune stage requires --base-checkpoint or --resume-from-checkpoint");
    if (cfg.hidden_dim <= 0 || cfg.output_dim <= 0 || cfg.vocab_size < 4 || cfg.max_tokens <= 0 || cfg.batch_size <= 0 || cfg.max_steps <= 0) throw std::runtime_error("embedding dimensions, batch size, and steps must be positive");
    if (cfg.print_plan || cfg.dry_run) {
        std::cout << "{\"status\":\"native-embedding-plan\",\"backend\":\"native-cpp\",\"architecture\":\"" << json_escape(cfg.architecture) << "\",\"stage\":\"" << json_escape(cfg.stage) << "\",\"adapter_type\":\"" << json_escape(cfg.adapter_type) << "\",\"data\":\"" << json_escape(cfg.data) << "\",\"hidden_dim\":" << cfg.hidden_dim << ",\"output_dim\":" << cfg.output_dim << ",\"max_steps\":" << cfg.max_steps << "}\n";
        return 0;
    }
    auto examples = load_data(cfg);
    Model model(cfg);
    DatasetMixer mixer(examples);
    if (!cfg.resume_checkpoint.empty()) {
        fs::path resume_path(cfg.resume_checkpoint);
        fs::path model_path = resume_path;
        if (fs::is_directory(resume_path) && fs::exists(resume_path / "embedding_adapter.bin")) model_path = resume_path / "embedding_adapter.bin";
        model.load(model_path);
        fs::path state_path = fs::is_directory(resume_path) ? resume_path / "embedding_optimizer.bin" : fs::path(resume_path.string() + ".optimizer.bin");
        load_training_state(model, mixer, state_path);
    } else if (!cfg.base_checkpoint.empty()) {
        const std::string requested_adapter = cfg.adapter_type;
        model.load(cfg.base_checkpoint);
        model.step = 0;
        model.initialize_adapter(requested_adapter);
    }
    std::unordered_map<std::string, double> metrics;
    double accumulated_loss = 0.0;
    uint64_t examples_seen = 0;
    const int grad_accum_steps = std::max(1, (cfg.effective_batch_size + cfg.batch_size - 1) / cfg.batch_size);
    for (int local_step = 0; local_step < cfg.max_steps; ++local_step) {
        model.zero_grad();
        const int dataset = mixer.next_dataset();
        double step_loss = 0.0;
        std::string step_objective;
        for (int accumulation = 0; accumulation < grad_accum_steps; ++accumulation) {
            auto batch = mixer.batch(dataset, cfg.batch_size);
            step_objective = examples[batch.front()].objective;
            for (size_t i = 0; i < batch.size(); ++i) {
                const Example& ex = examples[batch[i]];
                const Example* fallback = &examples[batch[(i + 1) % batch.size()]];
                Example objective_example = ex;
                if (ex.objective == "class") {
                    const Example* positive = nullptr;
                    for (size_t offset = 1; offset < batch.size(); ++offset) {
                        const Example& candidate = examples[batch[(i + offset) % batch.size()]];
                        if (candidate.label == ex.label && positive == nullptr) positive = &candidate;
                        if (candidate.label != ex.label) fallback = &candidate;
                    }
                    if (positive != nullptr) objective_example.second = positive->first;
                }
                step_loss += train_example(model, objective_example, *fallback, metrics);
            }
            examples_seen += batch.size();
        }
        float lr = cfg.learning_rate;
        if (cfg.warmup_steps > 0 && static_cast<int>(model.step) < cfg.warmup_steps) lr *= static_cast<float>(model.step + 1) / static_cast<float>(cfg.warmup_steps);
        const int records_this_step = cfg.batch_size * grad_accum_steps;
        model.update(lr, 1.0f / static_cast<float>(records_this_step));
        accumulated_loss += step_loss / static_cast<double>(records_this_step);
        if (cfg.progress_every > 0 && model.step % static_cast<uint32_t>(cfg.progress_every) == 0) {
            std::cerr << "[nfn-native-train] step " << model.step << '/' << (model.step + cfg.max_steps - local_step - 1)
                      << " train_loss=" << (step_loss / static_cast<double>(records_this_step)) << " examples=" << examples_seen
                      << " grad_accum_steps=" << grad_accum_steps << " objective=" << step_objective << " dataset=" << dataset << '\n';
        }
        if (cfg.checkpoint_every > 0 && model.step % static_cast<uint32_t>(cfg.checkpoint_every) == 0) {
            fs::path output(cfg.output_dir);
            std::ostringstream name; name << "embedding_model_" << std::setw(8) << std::setfill('0') << model.step << ".bin";
            model.save(output / name.str(), false);
            save_training_state(model, mixer, output / (name.str() + ".optimizer.bin"));
        }
    }
    fs::path output(cfg.output_dir); fs::create_directories(output);
    model.save(output / "embedding_model.bin", true);
    if (!model.adapter_a.empty()) model.save(output / "embedding_adapter.bin", false);
    save_training_state(model, mixer, output / "embedding_optimizer.bin");
    metrics["loss_mean"] = accumulated_loss / static_cast<double>(cfg.max_steps);
    metrics["examples_seen"] = static_cast<double>(examples_seen);
    metrics["positive_cosine"] /= std::max<double>(1.0, examples_seen);
    metrics["negative_cosine"] /= std::max<double>(1.0, examples_seen);
    save_metadata(model, model.cfg, output, metrics);
    std::ofstream(output / "DONE") << "step=" << model.step << '\n';
    std::cout << "{\"status\":\"native-embedding-trained\",\"passed\":true,\"backend\":\"native-cpp\",\"steps_completed\":" << model.step
              << ",\"records\":" << examples.size() << ",\"model_type\":\"text_embedding\",\"architecture_profile\":\"" << json_escape(model.cfg.architecture)
              << "\",\"encoder_core\":\"native_token_position_biencoder\",\"checkpoint\":{\"checkpoint_path\":\"" << json_escape((output / "embedding_model.bin").string())
              << "\",\"done_marker\":\"" << json_escape((output / "DONE").string()) << "\"},\"metrics\":{\"loss_mean\":" << metrics["loss_mean"]
              << ",\"positive_cosine\":" << metrics["positive_cosine"] << ",\"negative_cosine\":" << metrics["negative_cosine"] << "}}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        return cfg.embed_mode ? run_embed(std::move(cfg)) : run_train(std::move(cfg));
    } catch (const std::exception& exc) {
        std::cerr << "nfn_embedding_native_train: " << exc.what() << '\n';
        return 2;
    }
}
