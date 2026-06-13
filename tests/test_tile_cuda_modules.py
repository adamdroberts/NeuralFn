from __future__ import annotations

import pytest
import torch

from neuralfn.tile_cuda import (
    NVFP4Tensor,
    TileCudaConfig,
    build_tile_module,
    dequantize_nvfp4_reference,
    load_tile_cuda_extension,
    quantize_nvfp4_reference,
)
from neuralfn.semantic import ConversationalVocabulary, NUM_VOCAB_DIMS
from neuralfn.torch_backend import build_module


_SEMANTIC_TERM_COUNTS = tuple(len(ConversationalVocabulary().terms(dim_name)) for dim_name in ConversationalVocabulary().dim_names)
_SEMANTIC_MAX_TERMS = max(_SEMANTIC_TERM_COUNTS)


MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...]], ...] = (
    ("logit_softcap", {"softcap": 12.5}, (torch.linspace(-5.0, 5.0, 97),)),
    ("loss_scale", {"coef": 0.375}, (torch.linspace(-2.0, 2.0, 97),)),
    ("aux_loss_add", {"coef": 0.125}, (torch.linspace(-2.0, 2.0, 97), torch.linspace(1.0, -1.0, 97))),
    (
        "kl_penalty",
        {"kl_coef": 0.2},
        (torch.linspace(-2.0, 2.0, 97), torch.linspace(1.0, -1.0, 97), torch.linspace(-0.5, 0.5, 97)),
    ),
    (
        "residual_add",
        {"dim": 5, "init_scale": 0.75},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5), torch.linspace(1.5, -1.5, 2 * 3 * 5).reshape(2, 3, 5)),
    ),
    (
        "residual_mix",
        {"dim": 5, "primary_init": 0.8, "skip_init": 0.2},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5), torch.linspace(1.5, -1.5, 2 * 3 * 5).reshape(2, 3, 5)),
    ),
    (
        "manifold_hyper_connection",
        {"dim": 5, "beta_init": 0.15},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5), torch.linspace(1.5, -1.5, 2 * 3 * 5).reshape(2, 3, 5)),
    ),
    (
        "qk_gain",
        {"num_heads": 4, "qk_gain_init": 1.25},
        (torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 5).reshape(2, 4, 3, 5),),
    ),
    (
        "dyt",
        {"model_dim": 5, "alpha_init": 0.7},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "dropout",
        {"p": 0.0},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "mask_scheduler",
        {"vocab_size": 16, "mask_token_id": 0},
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
            torch.tensor([0.25, 0.75], dtype=torch.float32),
        ),
    ),
    (
        "rms_norm",
        {"eps": 1e-6},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "layer_norm",
        {"model_dim": 5, "eps": 1e-5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "group_norm",
        {"model_dim": 6, "num_groups": 3, "eps": 1e-5},
        (torch.linspace(-2.0, 2.0, 2 * 4 * 6).reshape(2, 4, 6),),
    ),
    (
        "qk_norm",
        {"eps": 1e-6},
        (
            torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 6).reshape(2, 4, 3, 6),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 6).reshape(2, 2, 3, 6),
        ),
    ),
    (
        "reshape_heads",
        {"num_heads": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 20).reshape(2, 3, 20),),
    ),
    (
        "merge_heads",
        {},
        (torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 5).reshape(2, 4, 3, 5),),
    ),
    (
        "repeat_kv",
        {"num_heads": 4, "num_kv_heads": 2},
        (torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),),
    ),
    (
        "scaled_dot_product_attention",
        {"is_causal": True, "backend": "math", "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "sliding_window_attention",
        {"window_size": 2, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "block_sparse_attention",
        {"sparse_block_size": 2, "num_sinks": 1, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "streaming_attention_sinks",
        {"window_size": 2, "num_sinks": 1, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "native_sparse_attention",
        {"window_size": 2, "num_sinks": 1, "compress_stride": 2, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "differential_attention",
        {"lambda_init": 0.35, "is_causal": True, "dropout_p": 0.0, "eps": 1e-5},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 6).reshape(2, 2, 4, 6),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 6).reshape(2, 2, 4, 6),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "causal_self_attention",
        {"model_dim": 8, "num_heads": 2, "num_kv_heads": 1, "rope_base": 10000.0, "qk_gain_init": 1.1},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 8).reshape(2, 4, 8),),
    ),
    (
        "fused_causal_attention",
        {"model_dim": 8, "num_heads": 2, "num_kv_heads": 1, "rope_base": 10000.0, "dropout_p": 0.0},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 8).reshape(2, 4, 8),),
    ),
    (
        "multi_latent_attention",
        {"model_dim": 8, "num_heads": 2, "kv_lora_rank": 4, "qk_rope_dim": 2, "rope_base": 10000.0, "dropout_p": 0.0},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 8).reshape(2, 4, 8),),
    ),
    (
        "routed_attention_experts",
        {
            "model_dim": 8,
            "num_heads": 2,
            "num_kv_heads": 1,
            "rope_base": 10000.0,
            "qk_gain_init": 1.1,
            "experts": 3,
            "top_k": 2,
            "is_causal": True,
        },
        (
            torch.linspace(-1.0, 1.0, 2 * 3 * 8).reshape(2, 3, 8),
            torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32),
            torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        ),
    ),
    (
        "mamba",
        {"model_dim": 5, "d_state": 4, "d_conv": 3, "expand": 2},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 5).reshape(2, 4, 5),),
    ),
    (
        "universal_transformer",
        {"model_dim": 8, "num_heads": 2, "mlp_mult": 1.5, "max_steps": 2, "halt_epsilon": 0.01},
        (torch.linspace(-1.0, 1.0, 2 * 3 * 8).reshape(2, 3, 8),),
    ),
    (
        "rotary_embedding",
        {"head_dim": 6, "rope_base": 10000.0},
        (
            torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 6).reshape(2, 4, 3, 6),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 6).reshape(2, 2, 3, 6),
        ),
    ),
    (
        "expert_combine",
        {},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "expert_dispatch",
        {"model_dim": 4, "experts": 3, "mlp_mult": 2},
        (
            torch.linspace(-1.0, 1.0, 2 * 3 * 4).reshape(2, 3, 4),
            torch.tensor(
                [
                    [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]],
                    [[0.2, 0.8], [0.9, 0.1], [0.35, 0.65]],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [[0, 1], [1, 2], [2, 0]],
                    [[2, 1], [0, 2], [1, 0]],
                ],
                dtype=torch.long,
            ),
        ),
    ),
    (
        "kv_cache_write",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
        ),
    ),
    (
        "kv_cache_read",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
            torch.linspace(-0.5, 0.5, 2 * 2 * 2 * 5).reshape(2, 2, 2, 5),
            torch.linspace(0.25, -0.25, 2 * 2 * 2 * 5).reshape(2, 2, 2, 5),
        ),
    ),
    (
        "kv_quant_pack",
        {},
        (
            (torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4) + 0.125) / 17.0,
            (torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4) * -0.75 + 0.375) / 19.0,
        ),
    ),
    (
        "kv_quant_unpack",
        {"head_dim": 4},
        (
            torch.cat(
                (
                    torch.linspace(-64.0, 63.0, 2 * 2 * 3 * 8).reshape(2, 2, 3, 8),
                    torch.full((2, 2, 3, 1), 0.03125, dtype=torch.float32),
                ),
                dim=-1,
            ),
        ),
    ),
    (
        "broadcast_expert_routes",
        {},
        (
            torch.linspace(-1.0, 1.0, 2 * 3 * 4).reshape(2, 3, 4),
            torch.tensor([[[0.7, 0.3]], [[0.2, 0.8]]], dtype=torch.float32),
            torch.tensor([[[1, 3]], [[2, 0]]], dtype=torch.long),
        ),
    ),
    (
        "broadcast_chunk_routes",
        {"chunk_size": 2},
        (
            torch.linspace(-1.0, 1.0, 2 * 5 * 4).reshape(2, 5, 4),
            torch.tensor([[[0.7, 0.3], [0.6, 0.4]], [[0.2, 0.8], [0.9, 0.1]]], dtype=torch.float32),
            torch.tensor([[[1, 3], [2, 0]], [[2, 0], [3, 1]]], dtype=torch.long),
        ),
    ),
    (
        "latent_mse_loss",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 3 * 5).reshape(2, 3, 5),
        ),
    ),
    (
        "latent_pool",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
    ),
    (
        "gelu",
        {},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "token_cross_entropy",
        {},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 7).reshape(2, 3, 7),
            torch.tensor([[0, 3, 6], [2, 4, 1]], dtype=torch.long),
        ),
    ),
    (
        "masked_token_cross_entropy",
        {"ignore_index": -100},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 7).reshape(2, 3, 7),
            torch.tensor([[0, 3, -100], [2, 4, 1]], dtype=torch.long),
            torch.tensor([[1.0, 0.5, 1.0], [0.0, 1.0, 0.25]], dtype=torch.float32),
        ),
    ),
    (
        "sequence_logp",
        {"ignore_index": -100},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 7).reshape(2, 3, 7),
            torch.tensor([[0, 3, -100], [2, 4, 1]], dtype=torch.long),
            torch.tensor([[1.0, 0.5, 1.0], [0.0, 1.0, 0.25]], dtype=torch.float32),
        ),
    ),
    (
        "byte_patch_merge",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.zeros((2, 7), dtype=torch.long),
        ),
    ),
    (
        "byte_patch_embed",
        {"model_dim": 4, "patch_size": 3, "stride": 2, "vocab_size": 16},
        (torch.tensor([[0, 1, 2, 3, 4], [15, 16, -1, 7, 8]], dtype=torch.long),),
    ),
    (
        "absolute_position_embedding",
        {"max_seq_len": 8, "model_dim": 5},
        (torch.zeros((2, 4, 5), dtype=torch.float32),),
    ),
    (
        "token_embedding",
        {"vocab_size": 11, "model_dim": 5},
        (torch.tensor([[0, 3, 4, 3], [10, 2, 1, 0]], dtype=torch.long),),
    ),
    (
        "linear",
        {"input_dim": 5, "output_dim": 7, "bias": True},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "lm_head",
        {"model_dim": 5, "vocab_size": 9},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "tied_lm_head",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(-1.0, 1.0, 9 * 5).reshape(9, 5),
        ),
    ),
    (
        "router_logits",
        {"model_dim": 5, "experts": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "value_head",
        {"model_dim": 5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "reward_head",
        {"model_dim": 5, "pool": "last"},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "denoise_head",
        {"model_dim": 5, "vocab_size": 9},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "kv_pca_encode",
        {"head_dim": 5, "compressed_dim": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
        ),
    ),
    (
        "kv_pca_decode",
        {"head_dim": 5, "compressed_dim": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3),
        ),
    ),
    (
        "jepa_projector",
        {"input_dim": 5, "latent_dim": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "jepa_predictor",
        {"latent_dim": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 8).reshape(2, 3, 8),),
    ),
    (
        "ttt_linear",
        {"input_dim": 5, "output_dim": 7, "hidden_dim": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "lora_linear",
        {"input_dim": 5, "output_dim": 7, "rank": 3, "alpha": 6.0, "bias": True, "dropout": 0.0},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "bitlinear_ternary",
        {"input_dim": 5, "output_dim": 7},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "fp8_linear",
        {"input_dim": 5, "output_dim": 7, "bias": True, "fp8_format": "e4m3", "amax_history_len": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "mx_linear",
        {"input_dim": 5, "output_dim": 7, "bias": True, "mx_format": "mxfp4", "mx_block_size": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "nf4_linear",
        {
            "input_dim": 5,
            "output_dim": 7,
            "rank": 3,
            "alpha": 6.0,
            "bias": True,
            "dropout": 0.0,
            "group_size": 4,
            "compute_dtype": "fp32",
        },
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "randmap_adapter",
        {"model_dim": 5, "adapter_dim": 3},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "causal_chunk_state",
        {"chunk_size": 2, "mode": "prefix"},
        (torch.linspace(-2.0, 2.0, 2 * 5 * 4).reshape(2, 5, 4),),
    ),
    (
        "causal_chunk_state",
        {"chunk_size": 2, "mode": "mean"},
        (torch.linspace(-2.0, 2.0, 2 * 5 * 4).reshape(2, 5, 4),),
    ),
    (
        "mlp_relu2",
        {"model_dim": 5, "mlp_mult": 3},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "swiglu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
    ),
    (
        "geglu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
    ),
    (
        "reglu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
    ),
    (
        "solu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
    ),
    (
        "act_weighted_sum",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 5).reshape(2, 4, 3, 5),
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
        ),
    ),
    (
        "act_halt_gate",
        {"model_dim": 5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "dpo_pairwise_loss",
        {"beta": 0.2, "label_smoothing": 0.1, "loss_type": "sigmoid"},
        (
            torch.linspace(-1.5, 1.5, 17),
            torch.linspace(1.0, -1.0, 17),
            torch.linspace(-0.5, 0.5, 17),
            torch.linspace(0.25, -0.25, 17),
        ),
    ),
    (
        "dpo_pairwise_loss",
        {"beta": 0.3, "loss_type": "hinge"},
        (
            torch.linspace(-0.5, 1.5, 17),
            torch.linspace(0.75, -1.25, 17),
            torch.linspace(-0.25, 0.25, 17),
            torch.linspace(0.5, -0.5, 17),
        ),
    ),
    (
        "dpo_pairwise_loss",
        {"beta": 0.4, "loss_type": "ipo"},
        (
            torch.linspace(-0.25, 1.75, 17),
            torch.linspace(1.25, -0.75, 17),
            torch.linspace(-0.1, 0.3, 17),
            torch.linspace(0.4, -0.2, 17),
        ),
    ),
    (
        "preference_bce_loss",
        {},
        (torch.linspace(-2.0, 2.0, 17), torch.linspace(1.5, -1.5, 17)),
    ),
    (
        "ppo_clipped_loss",
        {"clip_range": 0.2, "vf_coef": 0.5, "ent_coef": 0.0},
        (
            torch.linspace(-1.2, 1.1, 2 * 3).reshape(2, 3),
            torch.linspace(0.7, -0.8, 2 * 3).reshape(2, 3),
            torch.linspace(-0.5, 0.9, 2 * 3).reshape(2, 3),
            torch.linspace(-0.3, 0.8, 2 * 3).reshape(2, 3),
            torch.linspace(0.2, -0.4, 2 * 3).reshape(2, 3),
            torch.linspace(0.6, -0.2, 2 * 3).reshape(2, 3),
        ),
    ),
    (
        "gae_compute",
        {"gamma": 0.99, "lambda_": 0.95},
        (
            torch.linspace(-0.3, 1.2, 2 * 4).reshape(2, 4),
            torch.linspace(0.4, -0.2, 2 * 4).reshape(2, 4),
        ),
    ),
    (
        "route_balance_loss",
        {},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 4).reshape(2, 3, 4),),
    ),
    (
        "route_selection_loss",
        {"shared_experts": 2, "free_experts": 3, "ignore_index": -100},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * (2 + NUM_VOCAB_DIMS + 3)).reshape(2, 3, 2 + NUM_VOCAB_DIMS + 3),
            torch.tensor(
                [
                    [1 if i % 3 else -100 for i in range(NUM_VOCAB_DIMS)],
                    [2 if i % 4 else -100 for i in range(NUM_VOCAB_DIMS)],
                ],
                dtype=torch.long,
            ),
        ),
    ),
    (
        "route_distillation_loss",
        {"shared_experts": 2, "free_experts": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * (2 + NUM_VOCAB_DIMS + 3)).reshape(2, 3, 2 + NUM_VOCAB_DIMS + 3),
            torch.linspace(-1.5, 1.5, 2 * 3 * NUM_VOCAB_DIMS * _SEMANTIC_MAX_TERMS).reshape(
                2, 3, NUM_VOCAB_DIMS, _SEMANTIC_MAX_TERMS
            ),
        ),
    ),
    (
        "semantic_alignment_loss",
        {"ignore_index": -100},
        (
            torch.linspace(-2.0, 2.0, 2 * NUM_VOCAB_DIMS * _SEMANTIC_MAX_TERMS).reshape(
                2, NUM_VOCAB_DIMS, _SEMANTIC_MAX_TERMS
            ),
            torch.tensor(
                [
                    [0 if i % 5 else -100 for i in range(NUM_VOCAB_DIMS)],
                    [min(1, _SEMANTIC_TERM_COUNTS[i] - 1) if i % 4 else -100 for i in range(NUM_VOCAB_DIMS)],
                ],
                dtype=torch.long,
            ),
        ),
    ),
    (
        "semantic_hasher",
        {"dim": 5, "tables": 3, "planes": 4, "seed": 123},
        (torch.linspace(-1.0, 1.0, 2 * 5).reshape(2, 5),),
    ),
    (
        "semantic_chunk_hasher",
        {"dim": 5, "tables": 3, "planes": 4, "seed": 123},
        (torch.linspace(-1.0, 1.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "semantic_moe_router",
        {"n_experts": 5, "semantic_dim": 4, "top_k": 2},
        (torch.linspace(-1.0, 1.0, 2 * 3 * 4).reshape(2, 3, 4),),
    ),
    (
        "semantic_hash_router",
        {
            "n_experts": NUM_VOCAB_DIMS,
            "semantic_dim": NUM_VOCAB_DIMS,
            "top_k": 2,
            "tables": 3,
            "n_buckets": 16,
            "ignore_index": -100,
            "routing_source": "topic_logits",
        },
        (
            torch.linspace(-1.0, 1.0, 2 * NUM_VOCAB_DIMS).reshape(2, NUM_VOCAB_DIMS),
            torch.tensor([[0, 3, 7], [2, 4, 8]], dtype=torch.long),
            torch.linspace(-1.5, 1.5, 2 * NUM_VOCAB_DIMS * _SEMANTIC_MAX_TERMS).reshape(
                2, NUM_VOCAB_DIMS, _SEMANTIC_MAX_TERMS
            ),
            torch.full((2, NUM_VOCAB_DIMS), -100, dtype=torch.long),
        ),
    ),
    (
        "semantic_moe_jepa_evo_router",
        {
            "semantic_dim": NUM_VOCAB_DIMS,
            "top_k": 2,
            "shared_experts": 2,
            "free_experts": 3,
            "tables": 3,
            "n_buckets": 16,
            "ignore_index": -100,
        },
        (
            torch.linspace(-1.0, 1.0, 2 * 3 * NUM_VOCAB_DIMS).reshape(2, 3, NUM_VOCAB_DIMS),
            torch.tensor([[[0, 3, 7], [2, 4, 8], [1, 5, 9]], [[3, 6, 10], [4, 7, 11], [5, 8, 12]]], dtype=torch.long),
            torch.linspace(-1.5, 1.5, 2 * 3 * NUM_VOCAB_DIMS * _SEMANTIC_MAX_TERMS).reshape(
                2, 3, NUM_VOCAB_DIMS, _SEMANTIC_MAX_TERMS
            ),
            torch.full((2, NUM_VOCAB_DIMS), -100, dtype=torch.long),
        ),
    ),
    (
        "semantic_projector",
        {"input_dim": 5, "semantic_dim": 5, "residual_dim": 4, "n_sig_buckets": 8},
        (torch.linspace(-1.0, 1.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "semantic_chunk_projector",
        {"input_dim": 5, "semantic_dim": 5, "residual_dim": 4, "n_sig_buckets": 8},
        (torch.linspace(-1.0, 1.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "load_balance_loss",
        {"experts": 4},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 4).reshape(2, 3, 4),
            torch.tensor(
                [[[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], [[0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]],
                dtype=torch.float32,
            ),
            torch.tensor([[[1, 3], [2, 0], [1, 0]], [[2, 0], [3, 1], [2, 1]]], dtype=torch.long),
        ),
    ),
    (
        "topk_route",
        {"top_k": 2, "experts": 5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "attentionless_decoder",
        {"semantic_dim": 5, "residual_dim": 4, "vocab_size": 7, "n_buckets": 6},
        (
            torch.tensor([[0, 7, 2], [5, 6, 1]], dtype=torch.long),
            torch.linspace(-1.0, 1.0, 2 * 4).reshape(2, 4),
        ),
    ),
    (
        "auxfree_load_balancing",
        {"experts": 5, "top_k": 2, "bias_lr": 0.001},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "softmax_distillation_loss",
        {},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(1.25, -1.25, 2 * 3 * 5).reshape(2, 3, 5),
        ),
    ),
)

FP16_SIMPLE_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...]], ...] = MODULE_CASES[:9]
FP8_ELEMENTWISE_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...]], ...] = MODULE_CASES[:9]

FP16_AUXILIARY_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = (
    (
        "act_weighted_sum",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 5).reshape(2, 4, 3, 5),
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
        ),
        frozenset((1,)),
    ),
    (
        "latent_pool",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
        frozenset((1,)),
    ),
)

FP16_NORM_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...]], ...] = (
    (
        "rms_norm",
        {"eps": 1e-6},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "layer_norm",
        {"model_dim": 5, "eps": 1e-5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
    ),
    (
        "group_norm",
        {"model_dim": 6, "num_groups": 3, "eps": 1e-5},
        (torch.linspace(-2.0, 2.0, 2 * 4 * 6).reshape(2, 4, 6),),
    ),
    (
        "qk_norm",
        {"eps": 1e-6},
        (
            torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 6).reshape(2, 4, 3, 6),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 6).reshape(2, 2, 3, 6),
        ),
    ),
)

FP16_PROJECTION_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = (
    (
        "linear",
        {"input_dim": 5, "output_dim": 7, "bias": True},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "lm_head",
        {"model_dim": 5, "vocab_size": 9},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "tied_lm_head",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(-1.0, 1.0, 9 * 5).reshape(9, 5),
        ),
        frozenset((1,)),
    ),
    (
        "router_logits",
        {"model_dim": 5, "experts": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "value_head",
        {"model_dim": 5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "reward_head",
        {"model_dim": 5, "pool": "last"},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "denoise_head",
        {"model_dim": 5, "vocab_size": 9},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "kv_pca_encode",
        {"head_dim": 5, "compressed_dim": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
        ),
        frozenset(),
    ),
    (
        "kv_pca_decode",
        {"head_dim": 5, "compressed_dim": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3),
        ),
        frozenset(),
    ),
    (
        "jepa_projector",
        {"input_dim": 5, "latent_dim": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "jepa_predictor",
        {"latent_dim": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 8).reshape(2, 3, 8),),
        frozenset(),
    ),
    (
        "ttt_linear",
        {"input_dim": 5, "output_dim": 7, "hidden_dim": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "lora_linear",
        {"input_dim": 5, "output_dim": 7, "rank": 3, "alpha": 6.0, "bias": True, "dropout": 0.0},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "bitlinear_ternary",
        {"input_dim": 5, "output_dim": 7},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "fp8_linear",
        {"input_dim": 5, "output_dim": 7, "bias": True, "fp8_format": "e4m3", "amax_history_len": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "mx_linear",
        {"input_dim": 5, "output_dim": 7, "bias": True, "mx_format": "mxfp4", "mx_block_size": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "randmap_adapter",
        {"model_dim": 5, "adapter_dim": 3},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "mlp_relu2",
        {"model_dim": 5, "mlp_mult": 3},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "swiglu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
        frozenset(),
    ),
    (
        "geglu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
        frozenset(),
    ),
    (
        "reglu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
        frozenset(),
    ),
    (
        "solu",
        {"model_dim": 6, "mlp_mult": 4, "multiple_of": 8},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 6).reshape(2, 3, 6),),
        frozenset(),
    ),
    (
        "act_halt_gate",
        {"model_dim": 5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
)

FP8_PROJECTION_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = (
    (
        "linear",
        {"input_dim": 5, "output_dim": 7, "bias": True},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "lm_head",
        {"model_dim": 5, "vocab_size": 9},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "tied_lm_head",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(-1.0, 1.0, 9 * 5).reshape(9, 5),
        ),
        frozenset((1,)),
    ),
    (
        "router_logits",
        {"model_dim": 5, "experts": 4},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "value_head",
        {"model_dim": 5},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "reward_head",
        {"model_dim": 5, "pool": "last"},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "denoise_head",
        {"model_dim": 5, "vocab_size": 9},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),),
        frozenset(),
    ),
    (
        "kv_pca_encode",
        {"head_dim": 5, "compressed_dim": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 5).reshape(2, 2, 3, 5),
        ),
        frozenset(),
    ),
    (
        "kv_pca_decode",
        {"head_dim": 5, "compressed_dim": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3),
        ),
        frozenset(),
    ),
)
FP8_PROJECTION_MODULE_CASES = tuple(
    case for case in FP16_PROJECTION_MODULE_CASES if case[0] != "nf4_linear"
)
NVFP4_PROJECTION_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = tuple(
    case for case in FP16_PROJECTION_MODULE_CASES if case[0] != "nf4_linear"
)

FP16_ATTENTION_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...]], ...] = (
    (
        "rotary_embedding",
        {"head_dim": 6, "rope_base": 10000.0},
        (
            torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 6).reshape(2, 4, 3, 6),
            torch.linspace(1.5, -1.5, 2 * 2 * 3 * 6).reshape(2, 2, 3, 6),
        ),
    ),
    (
        "scaled_dot_product_attention",
        {"is_causal": True, "backend": "math", "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "sliding_window_attention",
        {"window_size": 2, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "block_sparse_attention",
        {"sparse_block_size": 2, "num_sinks": 1, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "streaming_attention_sinks",
        {"window_size": 2, "num_sinks": 1, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "native_sparse_attention",
        {"window_size": 2, "num_sinks": 1, "compress_stride": 2, "is_causal": True, "dropout_p": 0.0},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "differential_attention",
        {"lambda_init": 0.35, "is_causal": True, "dropout_p": 0.0, "eps": 1e-5},
        (
            torch.linspace(-0.75, 0.75, 2 * 2 * 4 * 6).reshape(2, 2, 4, 6),
            torch.linspace(0.5, -0.5, 2 * 2 * 4 * 6).reshape(2, 2, 4, 6),
            torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 3).reshape(2, 2, 4, 3),
        ),
    ),
    (
        "causal_self_attention",
        {"model_dim": 8, "num_heads": 2, "num_kv_heads": 1, "rope_base": 10000.0, "qk_gain_init": 1.1},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 8).reshape(2, 4, 8),),
    ),
    (
        "fused_causal_attention",
        {"model_dim": 8, "num_heads": 2, "num_kv_heads": 1, "rope_base": 10000.0, "dropout_p": 0.0},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 8).reshape(2, 4, 8),),
    ),
    (
        "multi_latent_attention",
        {"model_dim": 8, "num_heads": 2, "kv_lora_rank": 4, "qk_rope_dim": 2, "rope_base": 10000.0, "dropout_p": 0.0},
        (torch.linspace(-1.0, 1.0, 2 * 4 * 8).reshape(2, 4, 8),),
    ),
)

FP16_ROUTED_ATTENTION_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = (
    (
        "routed_attention_experts",
        {
            "model_dim": 8,
            "num_heads": 2,
            "num_kv_heads": 1,
            "rope_base": 10000.0,
            "qk_gain_init": 1.1,
            "experts": 3,
            "top_k": 2,
            "is_causal": True,
        },
        (
            torch.linspace(-1.0, 1.0, 2 * 3 * 8).reshape(2, 3, 8),
            torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32),
            torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        ),
        frozenset((1,)),
    ),
)
FP8_ATTENTION_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = tuple(
    (name, cfg, inputs, frozenset())
    for name, cfg, inputs in FP16_ATTENTION_MODULE_CASES
    if name != "rotary_embedding"
) + FP16_ROUTED_ATTENTION_MODULE_CASES
NVFP4_ATTENTION_MODULE_CASES = FP8_ATTENTION_MODULE_CASES

FP16_LOSS_MODULE_CASES: tuple[tuple[str, dict[str, object], tuple[torch.Tensor, ...], frozenset[int]], ...] = (
    (
        "latent_mse_loss",
        {},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(1.5, -1.5, 2 * 3 * 5).reshape(2, 3, 5),
        ),
        frozenset(),
    ),
    (
        "token_cross_entropy",
        {},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 7).reshape(2, 3, 7),
            torch.tensor([[0, 3, 6], [2, 4, 1]], dtype=torch.long),
        ),
        frozenset(),
    ),
    (
        "masked_token_cross_entropy",
        {"ignore_index": -100},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 7).reshape(2, 3, 7),
            torch.tensor([[0, 3, -100], [2, 4, 1]], dtype=torch.long),
            torch.tensor([[1.0, 0.5, 1.0], [0.0, 1.0, 0.25]], dtype=torch.float32),
        ),
        frozenset((2,)),
    ),
    (
        "sequence_logp",
        {"ignore_index": -100},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 7).reshape(2, 3, 7),
            torch.tensor([[0, 3, -100], [2, 4, 1]], dtype=torch.long),
            torch.tensor([[1.0, 0.5, 1.0], [0.0, 1.0, 0.25]], dtype=torch.float32),
        ),
        frozenset((2,)),
    ),
    (
        "dpo_pairwise_loss",
        {"beta": 0.2, "label_smoothing": 0.1, "loss_type": "sigmoid"},
        (
            torch.linspace(-1.5, 1.5, 17),
            torch.linspace(1.0, -1.0, 17),
            torch.linspace(-0.5, 0.5, 17),
            torch.linspace(0.25, -0.25, 17),
        ),
        frozenset(),
    ),
    (
        "preference_bce_loss",
        {},
        (torch.linspace(-2.0, 2.0, 17), torch.linspace(1.5, -1.5, 17)),
        frozenset(),
    ),
    (
        "ppo_clipped_loss",
        {"clip_range": 0.2, "vf_coef": 0.5, "ent_coef": 0.0},
        (
            torch.linspace(-1.2, 1.1, 2 * 3).reshape(2, 3),
            torch.linspace(0.7, -0.8, 2 * 3).reshape(2, 3),
            torch.linspace(-0.5, 0.9, 2 * 3).reshape(2, 3),
            torch.linspace(-0.3, 0.8, 2 * 3).reshape(2, 3),
            torch.linspace(0.2, -0.4, 2 * 3).reshape(2, 3),
            torch.linspace(0.6, -0.2, 2 * 3).reshape(2, 3),
        ),
        frozenset(),
    ),
    (
        "gae_compute",
        {"gamma": 0.99, "lambda_": 0.95},
        (
            torch.linspace(-0.3, 1.2, 2 * 4).reshape(2, 4),
            torch.linspace(0.4, -0.2, 2 * 4).reshape(2, 4),
        ),
        frozenset(),
    ),
    (
        "route_balance_loss",
        {},
        (torch.linspace(-2.0, 2.0, 2 * 3 * 4).reshape(2, 3, 4),),
        frozenset(),
    ),
    (
        "route_selection_loss",
        {"shared_experts": 2, "free_experts": 3, "ignore_index": -100},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * (2 + NUM_VOCAB_DIMS + 3)).reshape(2, 3, 2 + NUM_VOCAB_DIMS + 3),
            torch.tensor(
                [
                    [1 if i % 3 else -100 for i in range(NUM_VOCAB_DIMS)],
                    [2 if i % 4 else -100 for i in range(NUM_VOCAB_DIMS)],
                ],
                dtype=torch.long,
            ),
        ),
        frozenset(),
    ),
    (
        "route_distillation_loss",
        {"shared_experts": 2, "free_experts": 3},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * (2 + NUM_VOCAB_DIMS + 3)).reshape(2, 3, 2 + NUM_VOCAB_DIMS + 3),
            torch.linspace(-1.5, 1.5, 2 * 3 * NUM_VOCAB_DIMS * _SEMANTIC_MAX_TERMS).reshape(
                2, 3, NUM_VOCAB_DIMS, _SEMANTIC_MAX_TERMS
            ),
        ),
        frozenset(),
    ),
    (
        "semantic_alignment_loss",
        {"ignore_index": -100},
        (
            torch.linspace(-2.0, 2.0, 2 * NUM_VOCAB_DIMS * _SEMANTIC_MAX_TERMS).reshape(
                2, NUM_VOCAB_DIMS, _SEMANTIC_MAX_TERMS
            ),
            torch.tensor(
                [
                    [0 if i % 5 else -100 for i in range(NUM_VOCAB_DIMS)],
                    [min(1, _SEMANTIC_TERM_COUNTS[i] - 1) if i % 4 else -100 for i in range(NUM_VOCAB_DIMS)],
                ],
                dtype=torch.long,
            ),
        ),
        frozenset(),
    ),
    (
        "load_balance_loss",
        {"experts": 4},
        (
            torch.linspace(-2.0, 2.0, 2 * 3 * 4).reshape(2, 3, 4),
            torch.tensor(
                [[[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], [[0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]],
                dtype=torch.float32,
            ),
            torch.tensor([[[1, 3], [2, 0], [1, 0]], [[2, 0], [3, 1], [2, 1]]], dtype=torch.long),
        ),
        frozenset((1,)),
    ),
    (
        "softmax_distillation_loss",
        {},
        (
            torch.linspace(-1.5, 1.5, 2 * 3 * 5).reshape(2, 3, 5),
            torch.linspace(1.25, -1.25, 2 * 3 * 5).reshape(2, 3, 5),
        ),
        frozenset(),
    ),
)


def _clone_inputs(inputs: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
    cloned: list[torch.Tensor] = []
    for tensor in inputs:
        if torch.is_floating_point(tensor):
            cloned.append(tensor.detach().clone().to(device=device, dtype=torch.float32).contiguous().requires_grad_(True))
        else:
            cloned.append(tensor.detach().clone().to(device=device).contiguous())
    return tuple(cloned)


def _clone_fp16_inputs(
    inputs: tuple[torch.Tensor, ...],
    device: str = "cuda",
    fp32_indices: frozenset[int] = frozenset(),
) -> tuple[torch.Tensor, ...]:
    cloned: list[torch.Tensor] = []
    for index, tensor in enumerate(inputs):
        if torch.is_floating_point(tensor):
            dtype = torch.float32 if index in fp32_indices else torch.float16
            cloned.append(tensor.detach().clone().to(device=device, dtype=dtype).contiguous().requires_grad_(True))
        else:
            cloned.append(tensor.detach().clone().to(device=device).contiguous())
    return tuple(cloned)


def _clone_fp8_inputs(
    inputs: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
    device: str = "cuda",
    fp32_indices: frozenset[int] = frozenset(),
) -> tuple[torch.Tensor, ...]:
    cloned: list[torch.Tensor] = []
    for index, tensor in enumerate(inputs):
        if torch.is_floating_point(tensor):
            target_dtype = torch.float32 if index in fp32_indices else dtype
            cloned.append(tensor.detach().clone().to(device=device, dtype=target_dtype).contiguous().requires_grad_(True))
        else:
            cloned.append(tensor.detach().clone().to(device=device).contiguous())
    return tuple(cloned)


def _clone_nvfp4_inputs(
    inputs: tuple[torch.Tensor, ...],
    device: str = "cuda",
    fp32_indices: frozenset[int] = frozenset(),
) -> tuple[torch.Tensor | NVFP4Tensor, ...]:
    cloned: list[torch.Tensor | NVFP4Tensor] = []
    for index, tensor in enumerate(inputs):
        if torch.is_floating_point(tensor):
            source = tensor.detach().clone().to(device=device, dtype=torch.float32).contiguous().requires_grad_(True)
            if index in fp32_indices:
                cloned.append(source)
            else:
                cloned.append(quantize_nvfp4_reference(source, preserve_grad=True))
        else:
            cloned.append(tensor.detach().clone().to(device=device).contiguous())
    return tuple(cloned)


def _float32_view_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(dtype=torch.float32) if torch.is_floating_point(tensor) else tensor for tensor in inputs)


def _dequantized_nvfp4_view_inputs(inputs: tuple[torch.Tensor | NVFP4Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(dequantize_nvfp4_reference(item) if isinstance(item, NVFP4Tensor) else item for item in inputs)


def _input_grad_tensor(input_value: torch.Tensor | NVFP4Tensor) -> torch.Tensor | None:
    if isinstance(input_value, NVFP4Tensor):
        return None if input_value.source is None else input_value.source.grad
    return input_value.grad


def _iter_tensors(value: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def _cast_float_outputs(value: torch.Tensor | tuple[torch.Tensor, ...], dtype: torch.dtype) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return tuple(item.to(dtype=dtype) if torch.is_floating_point(item) else item for item in value)
    return value.to(dtype=dtype) if torch.is_floating_point(value) else value


def _loss_from_outputs(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor | None:
    losses = [tensor.float().square().mean() for tensor in _iter_tensors(value) if torch.is_floating_point(tensor)]
    if not losses:
        return None
    return sum(losses[1:], losses[0])


def _compare_stage(name: str, cfg: dict[str, object], inputs: tuple[torch.Tensor, ...], device: str, config: TileCudaConfig) -> None:
    ref = build_module(name, cfg).to(device)
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to(device)
    tile.load_state_dict(ref.state_dict())

    ref_inputs = _clone_inputs(inputs, device)
    tile_inputs = _clone_inputs(inputs, device)

    expected = ref(*ref_inputs)
    actual = tile(*tile_inputs)
    actual_items = _iter_tensors(actual)
    expected_items = _iter_tensors(expected)
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        torch.testing.assert_close(actual_item, expected_item, rtol=1e-5, atol=1e-6)

    expected_loss = _loss_from_outputs(expected)
    actual_loss = _loss_from_outputs(actual)
    if expected_loss is not None and actual_loss is not None:
        expected_loss.backward()
        actual_loss.backward()
    for actual_input, expected_input in zip(tile_inputs, ref_inputs):
        if not torch.is_floating_point(actual_input):
            continue
        if expected_input.grad is None or actual_input.grad is None:
            assert expected_input.grad is None
            assert actual_input.grad is None
            continue
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-6)
    for (actual_name, actual_param), (expected_name, expected_param) in zip(tile.named_parameters(), ref.named_parameters()):
        assert actual_name == expected_name
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-5, atol=1e-6)


def _compare_stage_fp16_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    config: TileCudaConfig,
    fp32_indices: frozenset[int] = frozenset(),
) -> None:
    ref = build_module(name, cfg).to("cuda")
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to("cuda")
    tile.load_state_dict(ref.state_dict())

    ref_inputs = _clone_fp16_inputs(inputs, fp32_indices=fp32_indices)
    tile_inputs = _clone_fp16_inputs(inputs, fp32_indices=fp32_indices)

    expected = _cast_float_outputs(ref(*_float32_view_inputs(ref_inputs)), torch.float16)
    actual = tile(*tile_inputs)
    actual_items = _iter_tensors(actual)
    expected_items = _iter_tensors(expected)
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        assert actual_item.dtype == torch.float16
        torch.testing.assert_close(actual_item, expected_item, rtol=3e-3, atol=3e-3)

    expected_loss = _loss_from_outputs(expected)
    actual_loss = _loss_from_outputs(actual)
    if expected_loss is not None and actual_loss is not None:
        expected_loss.backward()
        actual_loss.backward()
    for actual_input, expected_input in zip(tile_inputs, ref_inputs):
        if not torch.is_floating_point(actual_input):
            continue
        if expected_input.grad is None or actual_input.grad is None:
            assert expected_input.grad is None
            assert actual_input.grad is None
            continue
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=5e-2, atol=5e-2)
    for (actual_name, actual_param), (expected_name, expected_param) in zip(tile.named_parameters(), ref.named_parameters()):
        assert actual_name == expected_name
        assert actual_param.dtype == torch.float32
        if expected_param.grad is None or actual_param.grad is None:
            assert expected_param.grad is None
            assert actual_param.grad is None
            continue
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=5e-2, atol=5e-2)


def _compare_stage_fp16_mixed_output_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    config: TileCudaConfig,
    fp32_indices: frozenset[int] = frozenset(),
) -> None:
    ref = build_module(name, cfg).to("cuda")
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to("cuda")
    tile.load_state_dict(ref.state_dict())

    ref_inputs = _clone_fp16_inputs(inputs, fp32_indices=fp32_indices)
    tile_inputs = _clone_fp16_inputs(inputs, fp32_indices=fp32_indices)

    expected = ref(*_float32_view_inputs(ref_inputs))
    actual = tile(*tile_inputs)
    actual_items = _iter_tensors(actual)
    expected_items = _iter_tensors(expected)
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        if not torch.is_floating_point(actual_item):
            torch.testing.assert_close(actual_item, expected_item)
            continue
        assert actual_item.dtype in {torch.float16, torch.float32}
        torch.testing.assert_close(actual_item, expected_item.to(dtype=actual_item.dtype), rtol=5e-2, atol=5e-2)

    expected_loss = _loss_from_outputs(expected)
    actual_loss = _loss_from_outputs(actual)
    if expected_loss is not None and actual_loss is not None:
        expected_loss.backward()
        actual_loss.backward()
    for actual_input, expected_input in zip(tile_inputs, ref_inputs):
        if not torch.is_floating_point(actual_input):
            continue
        if expected_input.grad is None or actual_input.grad is None:
            assert expected_input.grad is None
            assert actual_input.grad is None
            continue
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=8e-2, atol=8e-2)
    for (actual_name, actual_param), (expected_name, expected_param) in zip(tile.named_parameters(), ref.named_parameters()):
        assert actual_name == expected_name
        if expected_param.grad is None or actual_param.grad is None:
            assert expected_param.grad is None
            assert actual_param.grad is None
            continue
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=8e-2, atol=8e-2)


def _compare_stage_fp8_projection_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    config: TileCudaConfig,
    dtype: torch.dtype,
    fp32_indices: frozenset[int] = frozenset(),
) -> None:
    ref = build_module(name, cfg).to("cuda")
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to("cuda")
    tile.load_state_dict(ref.state_dict())

    ref_inputs = _clone_fp8_inputs(inputs, dtype, fp32_indices=fp32_indices)
    tile_inputs = _clone_fp8_inputs(inputs, dtype, fp32_indices=fp32_indices)

    expected = ref(*_float32_view_inputs(ref_inputs))
    actual = tile(*tile_inputs)
    actual_items = _iter_tensors(actual)
    expected_items = _iter_tensors(expected)
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        assert actual_item.dtype == torch.float32
        torch.testing.assert_close(actual_item, expected_item, rtol=1e-4, atol=1e-4)

    expected_loss = _loss_from_outputs(expected)
    actual_loss = _loss_from_outputs(actual)
    if expected_loss is not None and actual_loss is not None:
        expected_loss.backward()
        actual_loss.backward()
    for actual_input, expected_input in zip(tile_inputs, ref_inputs):
        if not torch.is_floating_point(actual_input):
            continue
        if expected_input.grad is None or actual_input.grad is None:
            assert expected_input.grad is None
            assert actual_input.grad is None
            continue
        torch.testing.assert_close(
            actual_input.grad.to(dtype=torch.float32),
            expected_input.grad.to(dtype=torch.float32),
            rtol=8e-2,
            atol=8e-2,
        )
    for (actual_name, actual_param), (expected_name, expected_param) in zip(tile.named_parameters(), ref.named_parameters()):
        assert actual_name == expected_name
        assert actual_param.dtype == torch.float32
        if expected_param.grad is None or actual_param.grad is None:
            assert expected_param.grad is None
            assert actual_param.grad is None
            continue
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-4, atol=1e-4)


def _compare_stage_nvfp4_projection_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    config: TileCudaConfig,
    fp32_indices: frozenset[int] = frozenset(),
) -> None:
    ref = build_module(name, cfg).to("cuda")
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to("cuda")
    tile.load_state_dict(ref.state_dict())

    ref_inputs = _clone_nvfp4_inputs(inputs, fp32_indices=fp32_indices)
    tile_inputs = _clone_nvfp4_inputs(inputs, fp32_indices=fp32_indices)

    expected = ref(*_dequantized_nvfp4_view_inputs(ref_inputs))
    actual = tile(*tile_inputs)
    actual_items = _iter_tensors(actual)
    expected_items = _iter_tensors(expected)
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        if not torch.is_floating_point(actual_item):
            torch.testing.assert_close(actual_item, expected_item)
            continue
        assert actual_item.dtype == torch.float32
        torch.testing.assert_close(actual_item, expected_item, rtol=1e-4, atol=1e-4)

    expected_loss = _loss_from_outputs(expected)
    actual_loss = _loss_from_outputs(actual)
    if expected_loss is not None and actual_loss is not None:
        expected_loss.backward()
        actual_loss.backward()
    for actual_input, expected_input in zip(tile_inputs, ref_inputs):
        if isinstance(actual_input, NVFP4Tensor) or isinstance(expected_input, NVFP4Tensor):
            actual_grad = _input_grad_tensor(actual_input)
            expected_grad = _input_grad_tensor(expected_input)
        else:
            if not torch.is_floating_point(actual_input):
                continue
            actual_grad = actual_input.grad
            expected_grad = expected_input.grad
        if expected_grad is None or actual_grad is None:
            assert expected_grad is None
            assert actual_grad is None
            continue
        torch.testing.assert_close(
            actual_grad.to(dtype=torch.float32),
            expected_grad.to(dtype=torch.float32),
            rtol=8e-2,
            atol=8e-2,
        )
    for (actual_name, actual_param), (expected_name, expected_param) in zip(tile.named_parameters(), ref.named_parameters()):
        assert actual_name == expected_name
        assert actual_param.dtype == torch.float32
        if expected_param.grad is None or actual_param.grad is None:
            assert expected_param.grad is None
            assert actual_param.grad is None
            continue
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-4, atol=1e-4)


def _compare_stage_fp8_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    config: TileCudaConfig,
    dtype: torch.dtype,
) -> None:
    ref = build_module(name, cfg).to("cuda")
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to("cuda")
    tile.load_state_dict(ref.state_dict())

    ref_inputs = _clone_fp8_inputs(inputs, dtype)
    tile_inputs = _clone_fp8_inputs(inputs, dtype)

    expected = _cast_float_outputs(ref(*_float32_view_inputs(ref_inputs)), dtype)
    actual = tile(*tile_inputs)
    actual_items = _iter_tensors(actual)
    expected_items = _iter_tensors(expected)
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        assert actual_item.dtype == dtype
        torch.testing.assert_close(actual_item.to(torch.float32), expected_item.to(torch.float32), rtol=2e-1, atol=2e-1)

    expected_loss = _loss_from_outputs(expected)
    actual_loss = _loss_from_outputs(actual)
    if expected_loss is not None and actual_loss is not None:
        expected_loss.backward()
        actual_loss.backward()
    for actual_input, expected_input in zip(tile_inputs, ref_inputs):
        if not torch.is_floating_point(actual_input):
            continue
        if expected_input.grad is None or actual_input.grad is None:
            assert expected_input.grad is None
            assert actual_input.grad is None
            continue
        torch.testing.assert_close(
            actual_input.grad.to(dtype=torch.float32),
            expected_input.grad.to(dtype=torch.float32),
            rtol=3e-1,
            atol=3e-1,
        )
    for (actual_name, actual_param), (expected_name, expected_param) in zip(tile.named_parameters(), ref.named_parameters()):
        assert actual_name == expected_name
        assert actual_param.dtype == torch.float32
        if expected_param.grad is None or actual_param.grad is None:
            assert expected_param.grad is None
            assert actual_param.grad is None
            continue
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=3e-1, atol=3e-1)


@pytest.mark.parametrize(("name", "cfg", "inputs"), MODULE_CASES)
def test_tile_cuda_module_cpu_fallback_matches_torch_forward_and_backward(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    _compare_stage(name, cfg, inputs, "cpu", TileCudaConfig(backend="tile_cuda", strict=False))


def _require_cuda_tile_extension() -> TileCudaConfig:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    ext = load_tile_cuda_extension(config)
    if ext is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")
    return config


@pytest.mark.parametrize(("name", "cfg", "inputs"), MODULE_CASES)
def test_tile_cuda_module_gpu_kernel_matches_torch_forward_and_backward(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    _compare_stage(name, cfg, inputs, "cuda", _require_cuda_tile_extension())


@pytest.mark.parametrize(("name", "cfg", "inputs"), FP16_SIMPLE_MODULE_CASES)
def test_tile_cuda_simple_module_gpu_kernel_supports_fp16_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    _compare_stage_fp16_cast_contract(name, cfg, inputs, _require_cuda_tile_extension())


def test_tile_cuda_dropout_gpu_kernel_supports_fp16_training_mask_contract() -> None:
    config = _require_cuda_tile_extension()
    tile = build_tile_module("dropout", {"p": 0.25}, config)
    assert tile is not None
    tile = tile.to("cuda").train()
    x = (torch.arange(1, 49, device="cuda", dtype=torch.float32).reshape(2, 3, 8) / 17.0).to(torch.float16)
    x.requires_grad_(True)

    out = tile(x)

    assert out.dtype == torch.float16
    keep_prob = 0.75
    expected_kept = (x.detach().to(torch.float32) / keep_prob).to(torch.float16)
    kept = out != 0
    assert bool(kept.any())
    assert bool((~kept).any())
    torch.testing.assert_close(out[kept], expected_kept[kept], rtol=0.0, atol=0.0)
    torch.testing.assert_close(out[~kept], torch.zeros_like(out[~kept]), rtol=0.0, atol=0.0)

    out.float().sum().backward()

    assert x.grad is not None
    expected_grad = (kept.to(torch.float32) / keep_prob).to(torch.float16)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(("name", "cfg", "inputs"), FP8_ELEMENTWISE_MODULE_CASES)
@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_tile_cuda_elementwise_module_gpu_kernel_supports_fp8_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
) -> None:
    _compare_stage_fp8_cast_contract(name, cfg, inputs, _require_cuda_tile_extension(), dtype)


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), FP16_AUXILIARY_MODULE_CASES)
def test_tile_cuda_auxiliary_module_gpu_kernel_supports_fp16_activation_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
) -> None:
    _compare_stage_fp16_cast_contract(name, cfg, inputs, _require_cuda_tile_extension(), fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs"), FP16_NORM_MODULE_CASES)
def test_tile_cuda_norm_module_gpu_kernel_supports_fp16_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    _compare_stage_fp16_cast_contract(name, cfg, inputs, _require_cuda_tile_extension())


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), FP16_PROJECTION_MODULE_CASES)
def test_tile_cuda_projection_module_gpu_kernel_supports_fp16_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
) -> None:
    _compare_stage_fp16_cast_contract(name, cfg, inputs, _require_cuda_tile_extension(), fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), FP8_PROJECTION_MODULE_CASES)
@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_tile_cuda_projection_module_gpu_kernel_supports_fp8_activation_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
    dtype: torch.dtype,
) -> None:
    _compare_stage_fp8_projection_contract(name, cfg, inputs, _require_cuda_tile_extension(), dtype, fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), NVFP4_PROJECTION_MODULE_CASES)
def test_tile_cuda_projection_module_gpu_kernel_supports_nvfp4_activation_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
) -> None:
    _compare_stage_nvfp4_projection_contract(name, cfg, inputs, _require_cuda_tile_extension(), fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs"), FP16_ATTENTION_MODULE_CASES)
def test_tile_cuda_attention_module_gpu_kernel_supports_fp16_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    _compare_stage_fp16_cast_contract(name, cfg, inputs, _require_cuda_tile_extension())


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), FP8_ATTENTION_MODULE_CASES)
@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_tile_cuda_attention_module_gpu_kernel_supports_fp8_activation_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
    dtype: torch.dtype,
) -> None:
    _compare_stage_fp8_projection_contract(name, cfg, inputs, _require_cuda_tile_extension(), dtype, fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), NVFP4_ATTENTION_MODULE_CASES)
def test_tile_cuda_attention_module_gpu_kernel_supports_nvfp4_activation_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
) -> None:
    _compare_stage_nvfp4_projection_contract(name, cfg, inputs, _require_cuda_tile_extension(), fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), FP16_ROUTED_ATTENTION_MODULE_CASES)
def test_tile_cuda_routed_attention_module_gpu_kernel_supports_fp16_cast_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
) -> None:
    _compare_stage_fp16_cast_contract(name, cfg, inputs, _require_cuda_tile_extension(), fp32_indices)


@pytest.mark.parametrize(("name", "cfg", "inputs", "fp32_indices"), FP16_LOSS_MODULE_CASES)
def test_tile_cuda_loss_module_gpu_kernel_supports_fp16_mixed_output_contract(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    fp32_indices: frozenset[int],
) -> None:
    _compare_stage_fp16_mixed_output_contract(name, cfg, inputs, _require_cuda_tile_extension(), fp32_indices)


@pytest.mark.parametrize(
    ("name", "cfg", "inputs"),
    (
        ("random_timesteps", {}, (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),)),
        (
            "jepa_mask",
            {"mask_ratio": 0.5, "mask_token_id": 0, "mask_strategy": "random"},
            (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),),
        ),
        (
            "jepa_mask",
            {
                "mask_ratio": 0.5,
                "mask_token_id": 0,
                "mask_strategy": "block",
                "num_blocks": 2,
                "min_block_ratio": 0.25,
                "max_block_ratio": 0.5,
            },
            (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),),
        ),
    ),
)
def test_tile_cuda_stochastic_module_cpu_forward_matches_torch(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    ref = build_module(name, cfg)
    tile = build_tile_module(name, cfg, TileCudaConfig(backend="tile_cuda", strict=False))
    assert tile is not None
    tile.load_state_dict(ref.state_dict())
    expected = ref(*_clone_inputs(inputs, "cpu"))
    actual = tile(*_clone_inputs(inputs, "cpu"))
    for actual_item, expected_item in zip(_iter_tensors(actual), _iter_tensors(expected)):
        torch.testing.assert_close(actual_item, expected_item, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("name", "cfg", "inputs"),
    (
        ("random_timesteps", {}, (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),)),
        (
            "jepa_mask",
            {"mask_ratio": 0.5, "mask_token_id": 0, "mask_strategy": "random"},
            (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),),
        ),
        (
            "jepa_mask",
            {
                "mask_ratio": 0.5,
                "mask_token_id": 0,
                "mask_strategy": "block",
                "num_blocks": 2,
                "min_block_ratio": 0.25,
                "max_block_ratio": 0.5,
            },
            (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),),
        ),
    ),
)
def test_tile_cuda_stochastic_module_gpu_forward_matches_torch(
    name: str,
    cfg: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    config = _require_cuda_tile_extension()
    ref = build_module(name, cfg).to("cuda")
    tile = build_tile_module(name, cfg, config)
    assert tile is not None
    tile = tile.to("cuda")
    tile.load_state_dict(ref.state_dict())
    expected = ref(*_clone_inputs(inputs, "cuda"))
    actual = tile(*_clone_inputs(inputs, "cuda"))
    for actual_item, expected_item in zip(_iter_tensors(actual), _iter_tensors(expected)):
        torch.testing.assert_close(actual_item, expected_item, rtol=0.0, atol=0.0)
