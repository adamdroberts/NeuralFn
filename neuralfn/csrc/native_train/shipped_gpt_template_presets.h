#pragma once

#include <string>
#include <vector>

// Generated from neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS by
// tools/generate_native_gpt_template_catalog.py.
namespace neuralfn_native {

inline const std::vector<std::string>& shipped_gpt_template_presets() {
    static const std::vector<std::string> presets = {
        "nanogpt",
        "nanogpt_megakernel",
        "gpt2",
        "gpt2_megakernel",
        "gpt2_moa",
        "llama",
        "modern_norms_llama",
        "mixllama",
        "moe",
        "llama_fast",
        "llama_fast_megakernel",
        "mixllama_fast",
        "mixllama_fast_megakernel",
        "jamba",
        "ternary_b158",
        "fp8_llama",
        "mxfp4_llama",
        "deepseek_v3",
        "deepseek_v4",
        "gemma3",
        "diff_transformer",
        "longctx_sparse_llama",
        "qwen3_longctx",
        "auxfree_moe_jepa_evo",
        "diff_semantic_moe_jepa_evo",
        "dyt_geglu_semantic_dense_jepa_evo",
        "llama_megakernel",
        "kv_pca_llama",
        "seq2seq",
        "diffusion",
        "ttt_llama",
        "llm_jepa",
        "dense_jepa_evo",
        "moe_jepa_evo",
        "jepa_semantic_hybrid",
        "jepa_semantic_hybrid_megakernel",
        "semantic_router_moe",
        "semantic_router_moe_megakernel",
        "semantic_moe_jepa_evo",
        "semantic_dense_jepa_evo",
        "hnet_lm",
        "universal_llama",
        "nanogpt_modern",
        "gpt2_modern",
        "llama_modern",
        "moe_modern",
        "jamba_modern",
        "ternary_b158_modern",
        "seq2seq_modern",
        "diffusion_modern",
        "ttt_llama_modern",
        "llm_jepa_modern",
        "dense_jepa_evo_modern",
        "moe_jepa_evo_modern",
        "hnet_lm_modern",
        "universal_llama_modern",
        "kv_pca_llama_modern",
        "jepa_semantic_hybrid_modern",
        "semantic_router_moe_modern",
        "semantic_dense_jepa_evo_modern",
        "semantic_moe_jepa_evo_modern",
    };
    return presets;
}

}  // namespace neuralfn_native
