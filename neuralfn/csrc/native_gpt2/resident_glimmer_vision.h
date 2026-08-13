#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace neuralfn::resident_glimmer_vision {

struct Config {
    std::int64_t hidden_size = 1536;
    std::int64_t intermediate_size = 8960;
    std::int64_t num_layers = 50;
    std::int64_t num_heads = 16;
    std::int64_t patch_width = 1176;
    std::int64_t merge_size = 2;
    std::int64_t position_side = 32;
    std::int64_t adapter_size = 4096;
    std::int64_t output_size = 6656;
    float rope_theta = 10000.0f;
    float norm_eps = 1.0e-5f;
};

// Exact portable oracle for the canonical BF16 vision payload embedded after
// the target text tensors.  The owner of `payload` must outlive this object.
// `packed_patches` is [sum(t*h*w), patch_width]; `grid_thw` is flattened
// [media,3].  The returned rows are 2x2-merged and projected to the text width.
class Model final {
public:
    // Public only so allocation-free translation-unit helpers can consume the
    // typed views. These remain an internal C++ ABI, not a Python surface.
    struct Weight;
    struct Layer;
    class MappedFile;

    static std::shared_ptr<Model> load_bf16(
        const std::uint8_t* payload,
        std::int64_t payload_nbytes,
        Config config = {});
    static std::shared_ptr<Model> load_gguf(
        const std::string& checkpoint_path,
        Config config = {});

    std::vector<float> encode(
        const std::vector<float>& packed_patches,
        const std::vector<std::int64_t>& grid_thw,
        const std::atomic<bool>& cancelled) const;

    std::int64_t output_size() const noexcept { return config_.output_size; }
    std::int64_t weight_bytes() const noexcept { return payload_nbytes_; }

private:
    Model(const std::uint8_t* payload, std::int64_t payload_nbytes, Config config);
    Model(std::unique_ptr<MappedFile> mapped, Config config);
    void build_layout();
    void build_gguf_layout();

    const std::uint8_t* payload_ = nullptr;
    std::int64_t payload_nbytes_ = 0;
    std::unique_ptr<MappedFile> mapped_;
    Config config_;
    bool interleaved_rope_ = false;
    Weight* patch_ = nullptr;
    Weight* position_ = nullptr;
    Weight* pre_norm_weight_ = nullptr;
    Weight* pre_norm_bias_ = nullptr;
    Weight* post_norm_weight_ = nullptr;
    Weight* post_norm_bias_ = nullptr;
    Weight* adapter_fc1_ = nullptr;
    Weight* adapter_fc2_ = nullptr;
    Weight* projection_ = nullptr;
    std::vector<std::unique_ptr<Weight>> weights_;
    std::vector<Layer> layers_;
};

}  // namespace neuralfn::resident_glimmer_vision
