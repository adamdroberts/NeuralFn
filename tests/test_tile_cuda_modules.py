from __future__ import annotations

import pytest
import torch

from neuralfn.tile_cuda import TileCudaConfig, build_tile_module, load_tile_cuda_extension
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


def _clone_inputs(inputs: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
    cloned: list[torch.Tensor] = []
    for tensor in inputs:
        if torch.is_floating_point(tensor):
            cloned.append(tensor.detach().clone().to(device=device, dtype=torch.float32).contiguous().requires_grad_(True))
        else:
            cloned.append(tensor.detach().clone().to(device=device).contiguous())
    return tuple(cloned)


def _iter_tensors(value: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


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
