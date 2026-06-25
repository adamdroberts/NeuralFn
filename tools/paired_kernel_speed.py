#!/usr/bin/env python3
"""Run paired kernel speed comparisons in one process.

This is intended for CUDA kernel experiments where external GPU load can change
over time. It alternates baseline and candidate command order across samples and
reports paired ratios instead of timing each variant in separate manual runs.
An optional reference command can be included when a candidate must be judged
against both the older native route and an external implementation such as
llm.kittens in one locked GPU run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Sequence


@dataclass(frozen=True)
class TimedCommand:
    name: str
    argv: list[str]
    env_overrides: dict[str, str]


@dataclass(frozen=True)
class MetricRatioLimit:
    metric: str
    stat: str = "mean"
    min_ratio: float | None = None
    max_ratio: float | None = None


NATIVE_METRIC_PATHS = (
    ("steps_completed", ("steps_completed",)),
    ("train_loop_wall_ms", ("timing", "train_loop_wall_ms")),
    (
        "train_loop_cuda_event_wall_ms",
        ("timing", "train_loop_cuda_event_wall_ms"),
    ),
    (
        "train_loop_cuda_event_wall_ms_per_step",
        ("timing", "train_loop_cuda_event_wall_ms_per_step"),
    ),
    (
        "train_loop_cuda_event_first_step_wall_ms",
        ("timing", "train_loop_cuda_event_first_step_wall_ms"),
    ),
    (
        "train_loop_cuda_event_first_step_wall_ms_per_step",
        ("timing", "train_loop_cuda_event_first_step_wall_ms_per_step"),
    ),
    (
        "train_loop_cuda_event_steady_state_wall_ms",
        ("timing", "train_loop_cuda_event_steady_state_wall_ms"),
    ),
    (
        "train_loop_cuda_event_steady_state_wall_ms_per_step",
        ("timing", "train_loop_cuda_event_steady_state_wall_ms_per_step"),
    ),
    (
        "train_loop_cuda_event_timing_enabled",
        ("timing", "train_loop_cuda_event_timing_enabled"),
    ),
    ("setup_wall_ms", ("timing", "setup_wall_ms")),
    (
        "setup_cuda_event_timing_requested",
        ("timing", "setup_cuda_event_timing_requested"),
    ),
    (
        "setup_cuda_event_timing_enabled",
        ("timing", "setup_cuda_event_timing_enabled"),
    ),
    (
        "setup_cuda_event_timing_sync_count",
        ("timing", "setup_cuda_event_timing_sync_count"),
    ),
    (
        "setup_cuda_event_timing_skipped_count",
        ("timing", "setup_cuda_event_timing_skipped_count"),
    ),
    (
        "stage_timing_prealloc_event_pairs_requested",
        ("timing", "stage_timing_prealloc_event_pairs_requested"),
    ),
    (
        "stage_timing_event_pair_create_count",
        ("timing", "stage_timing_event_pair_create_count"),
    ),
    (
        "stage_timing_event_pair_preallocated_count",
        ("timing", "stage_timing_event_pair_preallocated_count"),
    ),
    (
        "stage_timing_event_pair_hot_create_count",
        ("timing", "stage_timing_event_pair_hot_create_count"),
    ),
    (
        "stage_timing_event_pair_unused_destroy_count",
        ("timing", "stage_timing_event_pair_unused_destroy_count"),
    ),
    ("checkpoint_wall_ms", ("timing", "checkpoint_wall_ms")),
    ("total_wall_ms", ("timing", "total_wall_ms")),
    ("train_tokens_per_second", ("timing", "train_tokens_per_second")),
    ("train_loss_host_d2h_count", ("train_loss_host_d2h_count",)),
    (
        "train_loss_host_d2h_copies_per_logged_step",
        ("train_loss_host_d2h_copies_per_logged_step",),
    ),
    (
        "train_loss_microbatch_host_d2h_copies_elided_per_logged_step",
        ("train_loss_microbatch_host_d2h_copies_elided_per_logged_step",),
    ),
    ("linear_tk_gemm_count", ("linear_tk_gemm_count",)),
    ("linear_tk_dweight_gemm_count", ("linear_tk_dweight_gemm_count",)),
    ("linear_tk_dgelu_dinput_gemm_count", ("linear_tk_dgelu_dinput_gemm_count",)),
    ("linear_cublaslt_gemm_count", ("linear_cublaslt_gemm_count",)),
    ("linear_cublaslt_bgrad_gemm_count", ("linear_cublaslt_bgrad_gemm_count",)),
    (
        "linear_cublaslt_bgrad_direct_write_count",
        ("linear_cublaslt_bgrad_direct_write_count",),
    ),
    (
        "linear_cublaslt_bgrad_accumulate_count",
        ("linear_cublaslt_bgrad_accumulate_count",),
    ),
    (
        "linear_cublaslt_grouped_layout_probe_status",
        ("linear_cublaslt_grouped_layout_probe_status",),
    ),
    (
        "linear_cublaslt_grouped_matmul_probe_status",
        ("linear_cublaslt_grouped_matmul_probe_status",),
    ),
    (
        "linear_cublaslt_grouped_matmul_probe_requested",
        ("linear_cublaslt_grouped_matmul_probe_requested",),
    ),
    (
        "linear_cublas_grouped_bf16_gemm_probe_status",
        ("linear_cublas_grouped_bf16_gemm_probe_status",),
    ),
    (
        "linear_cublas_grouped_bf16_gemm_probe_requested",
        ("linear_cublas_grouped_bf16_gemm_probe_requested",),
    ),
    (
        "linear_cublas_handle_prewarm_enabled",
        ("linear_cublas_handle_prewarm_enabled",),
    ),
    (
        "linear_cublas_handle_prewarm_requested",
        ("linear_cublas_handle_prewarm_requested",),
    ),
    (
        "linear_cublas_handle_prewarm_success_count",
        ("linear_cublas_handle_prewarm_success_count",),
    ),
    (
        "linear_cublas_handle_prewarm_failure_count",
        ("linear_cublas_handle_prewarm_failure_count",),
    ),
    (
        "linear_bf16_workspace_prewarm_enabled",
        ("linear_bf16_workspace_prewarm_enabled",),
    ),
    (
        "linear_bf16_workspace_prewarm_requested",
        ("linear_bf16_workspace_prewarm_requested",),
    ),
    (
        "linear_bf16_workspace_prewarm_success_count",
        ("linear_bf16_workspace_prewarm_success_count",),
    ),
    (
        "linear_bf16_workspace_prewarm_failure_count",
        ("linear_bf16_workspace_prewarm_failure_count",),
    ),
    (
        "linear_tk_qkv_first_use_prewarm_requested",
        ("linear_tk_qkv_first_use_prewarm_requested",),
    ),
    (
        "linear_tk_qkv_first_use_prewarm_requested_count",
        ("linear_tk_qkv_first_use_prewarm_requested_count",),
    ),
    (
        "linear_tk_qkv_first_use_prewarm_enabled_count",
        ("linear_tk_qkv_first_use_prewarm_enabled_count",),
    ),
    (
        "linear_tk_qkv_first_use_prewarm_success_count",
        ("linear_tk_qkv_first_use_prewarm_success_count",),
    ),
    (
        "linear_tk_qkv_first_use_prewarm_failure_count",
        ("linear_tk_qkv_first_use_prewarm_failure_count",),
    ),
    ("linear_bf16_gemm_count", ("linear_bf16_gemm_count",)),
    (
        "linear_bf16_gemm_fast16bf_request_count",
        ("linear_bf16_gemm_fast16bf_request_count",),
    ),
    ("bf16_to_f32_vec4_count", ("bf16_to_f32_vec4_count",)),
    ("lm_head_logits_tk_gemm_count", ("lm_head_logits_tk_gemm_count",)),
    ("lm_head_logits_cublaslt_gemm_count", ("lm_head_logits_cublaslt_gemm_count",)),
    ("lm_head_logits_bf16_gemm_count", ("lm_head_logits_bf16_gemm_count",)),
    ("lm_head_dhidden_tk_gemm_count", ("lm_head_dhidden_tk_gemm_count",)),
    (
        "lm_head_dhidden_cublaslt_gemm_count",
        ("lm_head_dhidden_cublaslt_gemm_count",),
    ),
    ("lm_head_dhidden_bf16_gemm_count", ("lm_head_dhidden_bf16_gemm_count",)),
    (
        "lm_head_dhidden_strided_vocab_gemm_count",
        ("lm_head_dhidden_strided_vocab_gemm_count",),
    ),
    (
        "lm_head_dweight_strided_vocab_gemm_count",
        ("lm_head_dweight_strided_vocab_gemm_count",),
    ),
    (
        "lm_head_prob_only_corrections_chunk_count",
        ("lm_head_prob_only_corrections_chunk_count",),
    ),
    (
        "lm_head_prob_only_ce_target_correction_chunk_count",
        ("lm_head_prob_only_ce_target_correction_chunk_count",),
    ),
    (
        "lm_head_prob_only_combined_correction_launch_count",
        ("lm_head_prob_only_combined_correction_launch_count",),
    ),
    (
        "lm_head_prob_only_dhidden_correction_launch_count",
        ("lm_head_prob_only_dhidden_correction_launch_count",),
    ),
    (
        "lm_head_prob_only_dweight_correction_launch_count",
        ("lm_head_prob_only_dweight_correction_launch_count",),
    ),
    ("stored_mlp_activation_blocks", ("stored_mlp_activation_blocks",)),
    ("stored_mlp_activation_elements", ("stored_mlp_activation_elements",)),
    ("stored_mlp_activation_bytes", ("stored_mlp_activation_bytes",)),
    (
        "stored_packed_attention_activation_blocks",
        ("stored_packed_attention_activation_blocks",),
    ),
    (
        "stored_packed_attention_bf16_elements",
        ("stored_packed_attention_bf16_elements",),
    ),
    (
        "stored_packed_attention_bf16_bytes",
        ("stored_packed_attention_bf16_bytes",),
    ),
    (
        "stored_packed_attention_ln1_bf16_blocks",
        ("stored_packed_attention_ln1_bf16_blocks",),
    ),
    (
        "stored_packed_attention_ln1_bf16_elements",
        ("stored_packed_attention_ln1_bf16_elements",),
    ),
    (
        "stored_packed_attention_ln1_bf16_bytes",
        ("stored_packed_attention_ln1_bf16_bytes",),
    ),
    (
        "stored_packed_attention_lse_elements",
        ("stored_packed_attention_lse_elements",),
    ),
    (
        "stored_packed_attention_lse_bytes",
        ("stored_packed_attention_lse_bytes",),
    ),
    (
        "stored_residual1_activation_blocks",
        ("stored_residual1_activation_blocks",),
    ),
    (
        "stored_residual1_activation_elements",
        ("stored_residual1_activation_elements",),
    ),
    ("stored_residual1_activation_bytes", ("stored_residual1_activation_bytes",)),
    ("block_backward_dinput_tk_gemm_count", ("block_backward_dinput_tk_gemm_count",)),
    (
        "block_backward_dinput_cublaslt_gemm_count",
        ("block_backward_dinput_cublaslt_gemm_count",),
    ),
    ("block_backward_dinput_bf16_gemm_count", ("block_backward_dinput_bf16_gemm_count",)),
    (
        "bf16_persistent_block_input_ln1_backward_count",
        ("bf16_persistent_block_input_ln1_backward_count",),
    ),
    (
        "block_backward_mlp_proj_dinput_before_dweight_count",
        ("block_backward_mlp_proj_dinput_before_dweight_count",),
    ),
    (
        "block_backward_mlp_proj_concurrent_dinput_dweight_count",
        ("block_backward_mlp_proj_concurrent_dinput_dweight_count",),
    ),
    (
        "block_backward_mlp_fc_dinput_before_dweight_count",
        ("block_backward_mlp_fc_dinput_before_dweight_count",),
    ),
    (
        "block_backward_attn_proj_dinput_before_dweight_count",
        ("block_backward_attn_proj_dinput_before_dweight_count",),
    ),
    (
        "block_backward_attn_proj_first_step_concurrent_dinput_dweight_count",
        ("block_backward_attn_proj_first_step_concurrent_dinput_dweight_count",),
    ),
    (
        "block_backward_qkv_dinput_before_dweight_count",
        ("block_backward_qkv_dinput_before_dweight_count",),
    ),
    (
        "lm_head_overlap_last_dweight_queue_count",
        ("lm_head_overlap_last_dweight_queue_count",),
    ),
    (
        "lm_head_overlap_last_dweight_sync_count",
        ("lm_head_overlap_last_dweight_sync_count",),
    ),
    (
        "lm_head_classifier_chunk_kernel_available",
        ("lm_head_classifier_chunk_kernel_available",),
    ),
    (
        "lm_head_classifier_chunk_kernel_enabled",
        ("lm_head_classifier_chunk_kernel_enabled",),
    ),
    (
        "lm_head_classifier_ce_no_loss_requested",
        ("lm_head_classifier_ce_no_loss_requested",),
    ),
    (
        "lm_head_classifier_ce_no_loss_enabled",
        ("lm_head_classifier_ce_no_loss_enabled",),
    ),
    (
        "lm_head_ce_no_loss_default_specialized_requested",
        ("lm_head_ce_no_loss_default_specialized_requested",),
    ),
    (
        "lm_head_ce_no_loss_default_specialized_enabled",
        ("lm_head_ce_no_loss_default_specialized_enabled",),
    ),
    (
        "lm_head_ce_no_loss_llmk_style_specialized_requested",
        ("lm_head_ce_no_loss_llmk_style_specialized_requested",),
    ),
    (
        "lm_head_ce_no_loss_llmk_style_specialized_enabled",
        ("lm_head_ce_no_loss_llmk_style_specialized_enabled",),
    ),
    (
        "lm_head_ce_row_loss_reduction_requested",
        ("lm_head_ce_row_loss_reduction_requested",),
    ),
    (
        "lm_head_ce_row_loss_reduction_available",
        ("lm_head_ce_row_loss_reduction_available",),
    ),
    (
        "lm_head_ce_row_loss_reduction_enabled",
        ("lm_head_ce_row_loss_reduction_enabled",),
    ),
    (
        "lm_head_ce_row_loss_sum_accumulate_available",
        ("lm_head_ce_row_loss_sum_accumulate_available",),
    ),
    (
        "lm_head_ce_row_loss_sum_accumulate_requested",
        ("lm_head_ce_row_loss_sum_accumulate_requested",),
    ),
    (
        "lm_head_ce_row_loss_sum_accumulate_enabled",
        ("lm_head_ce_row_loss_sum_accumulate_enabled",),
    ),
    (
        "lm_head_ce_loss_bin_reduction_available",
        ("lm_head_ce_loss_bin_reduction_available",),
    ),
    (
        "lm_head_ce_loss_bin_reduction_requested",
        ("lm_head_ce_loss_bin_reduction_requested",),
    ),
    (
        "lm_head_ce_loss_bin_reduction_enabled",
        ("lm_head_ce_loss_bin_reduction_enabled",),
    ),
    ("lm_head_ce_loss_bin_count_requested", ("lm_head_ce_loss_bin_count_requested",)),
    ("lm_head_classifier_chunk_launch_count", ("lm_head_classifier_chunk_launch_count",)),
    (
        "lm_head_classifier_loss_bin_launch_count",
        ("lm_head_classifier_loss_bin_launch_count",),
    ),
    (
        "lm_head_classifier_no_loss_chunk_count",
        ("lm_head_classifier_no_loss_chunk_count",),
    ),
    ("lm_head_classifier_last_rows", ("lm_head_classifier_last_rows",)),
    ("lm_head_classifier_last_vocab", ("lm_head_classifier_last_vocab",)),
    ("lm_head_classifier_last_row_stride", ("lm_head_classifier_last_row_stride",)),
    ("linear_bf16_a_pack_count", ("linear_bf16_a_pack_count",)),
    ("linear_bf16_a_cache_hit_count", ("linear_bf16_a_cache_hit_count",)),
    ("attention_forward_tk_launch_count", ("attention_forward_tk_launch_count",)),
    ("attention_backward_tk_launch_count", ("attention_backward_tk_launch_count",)),
    ("attention_backward_tk_batch_cap", ("attention_backward_tk_batch_cap",)),
    (
        "attention_backward_tk_chunk_batch_total",
        ("attention_backward_tk_chunk_batch_total",),
    ),
    (
        "attention_backward_tk_chunk_batch_max",
        ("attention_backward_tk_chunk_batch_max",),
    ),
    (
        "attention_backward_tk_chunk_batch_min",
        ("attention_backward_tk_chunk_batch_min",),
    ),
    (
        "attention_backward_tk_chunk_batch_last",
        ("attention_backward_tk_chunk_batch_last",),
    ),
    ("attention_backward_tk_block_size", ("attention_backward_tk_block_size",)),
    (
        "attention_backward_tk_block_size_symbol_loaded",
        ("attention_backward_tk_block_size_symbol_loaded",),
    ),
    (
        "attention_backward_float_hd64_dprep_launch_count",
        ("attention_backward_float_hd64_dprep_launch_count",),
    ),
    ("attention_backward_dprep_timing_us", ("attention_backward_dprep_timing_us",)),
    ("attention_backward_dprep_timing_count", ("attention_backward_dprep_timing_count",)),
    ("attention_backward_tk_timing_us", ("attention_backward_tk_timing_us",)),
    ("attention_backward_tk_timing_count", ("attention_backward_tk_timing_count",)),
    ("lm_head_reuse_forward_logits_enabled", ("lm_head_reuse_forward_logits_enabled",)),
    (
        "lm_head_cooperative_backward_required",
        ("lm_head_cooperative_backward_required",),
    ),
    (
        "lm_head_cooperative_backward_abi_wrapper_available",
        ("lm_head_cooperative_backward_abi_wrapper_available",),
    ),
    (
        "lm_head_cooperative_backward_sequence_wrapper_available",
        ("lm_head_cooperative_backward_sequence_wrapper_available",),
    ),
    (
        "lm_head_cooperative_backward_kernel_available",
        ("lm_head_cooperative_backward_kernel_available",),
    ),
    (
        "lm_head_cooperative_backward_fused_kernel_available",
        ("lm_head_cooperative_backward_fused_kernel_available",),
    ),
    (
        "lm_head_cooperative_backward_fused_kernel_symbol_available",
        ("lm_head_cooperative_backward_fused_kernel_symbol_available",),
    ),
    (
        "lm_head_cooperative_backward_fused_kernel_capability_available",
        ("lm_head_cooperative_backward_fused_kernel_capability_available",),
    ),
    (
        "lm_head_cooperative_backward_route_integrated",
        ("lm_head_cooperative_backward_route_integrated",),
    ),
    (
        "lm_head_cooperative_backward_kernel_enabled",
        ("lm_head_cooperative_backward_kernel_enabled",),
    ),
    (
        "lm_head_cooperative_backward_cuda_graph_available",
        ("lm_head_cooperative_backward_cuda_graph_available",),
    ),
    (
        "lm_head_cooperative_backward_cuda_graph_enabled",
        ("lm_head_cooperative_backward_cuda_graph_enabled",),
    ),
    (
        "lm_head_cooperative_backward_graph_prewarm_enabled",
        ("lm_head_cooperative_backward_graph_prewarm_enabled",),
    ),
    (
        "lm_head_cooperative_backward_sequence_wrapper_enabled",
        ("lm_head_cooperative_backward_sequence_wrapper_enabled",),
    ),
    (
        "lm_head_cooperative_sequence_launch_count",
        ("lm_head_cooperative_sequence_launch_count",),
    ),
    (
        "lm_head_cooperative_sequence_ce_launch_count",
        ("lm_head_cooperative_sequence_ce_launch_count",),
    ),
    (
        "lm_head_cooperative_sequence_dhidden_launch_count",
        ("lm_head_cooperative_sequence_dhidden_launch_count",),
    ),
    (
        "lm_head_cooperative_sequence_dweight_launch_count",
        ("lm_head_cooperative_sequence_dweight_launch_count",),
    ),
    (
        "lm_head_cooperative_sequence_concurrent_count",
        ("lm_head_cooperative_sequence_concurrent_count",),
    ),
    (
        "lm_head_cooperative_sequence_legacy_count",
        ("lm_head_cooperative_sequence_legacy_count",),
    ),
    (
        "lm_head_cooperative_sequence_loss_bin_count",
        ("lm_head_cooperative_sequence_loss_bin_count",),
    ),
    (
        "lm_head_fused_graph_capture_attempt_count",
        ("lm_head_fused_graph_capture_attempt_count",),
    ),
    (
        "lm_head_fused_graph_capture_success_count",
        ("lm_head_fused_graph_capture_success_count",),
    ),
    (
        "lm_head_fused_graph_upload_success_count",
        ("lm_head_fused_graph_upload_success_count",),
    ),
    (
        "lm_head_fused_graph_upload_failure_count",
        ("lm_head_fused_graph_upload_failure_count",),
    ),
    (
        "lm_head_fused_graph_cache_hit_count",
        ("lm_head_fused_graph_cache_hit_count",),
    ),
    (
        "lm_head_fused_graph_thread_cache_hit_count",
        ("lm_head_fused_graph_thread_cache_hit_count",),
    ),
    (
        "lm_head_fused_graph_cache_entry_count",
        ("lm_head_fused_graph_cache_entry_count",),
    ),
    (
        "lm_head_fused_graph_replay_count",
        ("lm_head_fused_graph_replay_count",),
    ),
    (
        "lm_head_fused_graph_replay_success_count",
        ("lm_head_fused_graph_replay_success_count",),
    ),
    (
        "lm_head_fused_graph_body_node_count_per_replay",
        ("lm_head_fused_graph_body_node_count_per_replay",),
    ),
    (
        "lm_head_fused_graph_body_ce_node_count_per_replay",
        ("lm_head_fused_graph_body_ce_node_count_per_replay",),
    ),
    (
        "lm_head_fused_graph_body_dhidden_node_count_per_replay",
        ("lm_head_fused_graph_body_dhidden_node_count_per_replay",),
    ),
    (
        "lm_head_fused_graph_body_dweight_node_count_per_replay",
        ("lm_head_fused_graph_body_dweight_node_count_per_replay",),
    ),
    (
        "lm_head_fused_graph_body_node_replay_total",
        ("lm_head_fused_graph_body_node_replay_total",),
    ),
    (
        "lm_head_fused_graph_body_ce_node_replay_total",
        ("lm_head_fused_graph_body_ce_node_replay_total",),
    ),
    (
        "lm_head_fused_graph_body_dhidden_node_replay_total",
        ("lm_head_fused_graph_body_dhidden_node_replay_total",),
    ),
    (
        "lm_head_fused_graph_body_dweight_node_replay_total",
        ("lm_head_fused_graph_body_dweight_node_replay_total",),
    ),
    (
        "lm_head_fused_graph_fallback_count",
        ("lm_head_fused_graph_fallback_count",),
    ),
    (
        "lm_head_fused_graph_prewarm_success_count",
        ("lm_head_fused_graph_prewarm_success_count",),
    ),
    (
        "lm_head_fused_graph_prewarm_failure_count",
        ("lm_head_fused_graph_prewarm_failure_count",),
    ),
    (
        "lm_head_fused_graph_prewarm_cache_entry_count",
        ("lm_head_fused_graph_prewarm_cache_entry_count",),
    ),
    ("lm_head_bf16_logit_bytes", ("lm_head_bf16_logit_bytes",)),
    ("lm_head_full_logit_elements", ("lm_head_full_logit_elements",)),
    ("activation_tape_count", ("block_state_layout", "activation_tape_count")),
    (
        "full_activation_tape_enabled",
        ("block_state_layout", "full_activation_tape_enabled"),
    ),
    ("backward_recompute_blocks", ("block_state_layout", "backward_recompute_blocks")),
    (
        "final_block_backward_recompute_elided",
        ("block_state_layout", "final_block_backward_recompute_elided"),
    ),
    ("activation_tape_strategy", ("block_state_layout", "activation_tape_strategy")),
    (
        "block_state_layout.activation_tape_strategy",
        ("block_state_layout", "activation_tape_strategy"),
    ),
    ("stored_packed_attention_activation_blocks", ("stored_packed_attention_activation_blocks",)),
    ("transformer_device_arena_requested", ("transformer_device_arena_requested",)),
    ("transformer_device_arena_enabled", ("transformer_device_arena_enabled",)),
    (
        "transformer_device_arena_cuda_malloc_count",
        ("transformer_device_arena_cuda_malloc_count",),
    ),
    (
        "transformer_device_arena_requested_bytes",
        ("transformer_device_arena_requested_bytes",),
    ),
    (
        "transformer_device_arena_allocated_bytes",
        ("transformer_device_arena_allocated_bytes",),
    ),
    (
        "transformer_device_arena_uint16_byte_offset",
        ("transformer_device_arena_uint16_byte_offset",),
    ),
    (
        "lm_head_classifier.reference_full_bf16_logit_bytes",
        ("lm_head_classifier_strategy_contract", "reference_full_bf16_logit_bytes"),
    ),
    (
        "lm_head_classifier.native_chunk_bf16_logit_bytes",
        ("lm_head_classifier_strategy_contract", "native_chunk_bf16_logit_bytes"),
    ),
    (
        "lm_head_classifier.resident_logit_reduction_ratio",
        ("lm_head_classifier_strategy_contract", "resident_logit_reduction_ratio"),
    ),
    (
        "lm_head_classifier.native_logit_chunk_rows",
        ("lm_head_classifier_strategy_contract", "native_logit_chunk_rows"),
    ),
    (
        "lm_head_classifier.native_logit_chunk_count",
        ("lm_head_classifier_strategy_contract", "native_logit_chunk_count"),
    ),
    (
        "block_state_layout.layer_norm_backward_affine_row_chunk_size",
        ("block_state_layout", "layer_norm_backward_affine_row_chunk_size"),
    ),
    (
        "block_state_layout.linear_backward_bias_row_chunk_size",
        ("block_state_layout", "linear_backward_bias_row_chunk_size"),
    ),
    (
        "block_state_layout.linear_backward_bias_threads_per_block",
        ("block_state_layout", "linear_backward_bias_threads_per_block"),
    ),
    (
        "block_state_layout.optimizer_tile_size",
        ("block_state_layout", "optimizer_tile_size"),
    ),
    (
        "block_state_layout.optimizer_tile_strategy",
        ("block_state_layout", "optimizer_tile_strategy"),
    ),
    (
        "block_state_layout.attention_backward_tk_block_size",
        ("block_state_layout", "attention_backward_tk_block_size"),
    ),
    (
        "block_state_layout.attention_backward_tk_block_size_symbol_loaded",
        ("block_state_layout", "attention_backward_tk_block_size_symbol_loaded"),
    ),
    ("float_arena_cuda_malloc_wall_ms", ("float_arena_cuda_malloc_wall_ms",)),
    (
        "float_arena_pointer_assign_wall_ms",
        ("float_arena_pointer_assign_wall_ms",),
    ),
    ("uint16_arena_cuda_malloc_wall_ms", ("uint16_arena_cuda_malloc_wall_ms",)),
    (
        "uint16_arena_pointer_assign_wall_ms",
        ("uint16_arena_pointer_assign_wall_ms",),
    ),
    ("uint16_arena_first_requested", ("uint16_arena_first_requested",)),
    ("uint16_arena_first_enabled", ("uint16_arena_first_enabled",)),
    ("arena_materialize_order", ("arena_materialize_order",)),
    (
        "transformer_device_arena_cuda_malloc_wall_ms",
        ("transformer_device_arena_cuda_malloc_wall_ms",),
    ),
    (
        "transformer_device_arena_pointer_assign_wall_ms",
        ("transformer_device_arena_pointer_assign_wall_ms",),
    ),
    (
        "concurrent_arena_materialize_requested",
        ("concurrent_arena_materialize_requested",),
    ),
    (
        "concurrent_arena_materialize_enabled",
        ("concurrent_arena_materialize_enabled",),
    ),
    (
        "concurrent_arena_materialize_count",
        ("concurrent_arena_materialize_count",),
    ),
    (
        "block_state_layout.mlp_residual_next_ln1_fusion_enabled",
        ("block_state_layout", "mlp_residual_next_ln1_fusion_enabled"),
    ),
    (
        "block_state_layout.mlp_residual_next_ln1_fusion_count",
        ("block_state_layout", "mlp_residual_next_ln1_fusion_count"),
    ),
    (
        "block_state_layout.mlp_residual_next_ln1_strategy",
        ("block_state_layout", "mlp_residual_next_ln1_strategy"),
    ),
)
NATIVE_ROUTE_COUNTER_KEYS = (
    "linear_tk_gemm_count",
    "linear_tk_dweight_gemm_count",
    "linear_tk_dgelu_dinput_gemm_count",
    "linear_cublaslt_gemm_count",
    "linear_cublaslt_bgrad_gemm_count",
    "linear_cublaslt_bgrad_direct_write_count",
    "linear_cublaslt_bgrad_accumulate_count",
    "linear_cublaslt_grouped_layout_probe_status",
    "linear_cublaslt_grouped_matmul_probe_requested",
    "linear_cublaslt_grouped_matmul_probe_status",
    "linear_cublas_grouped_bf16_gemm_probe_requested",
    "linear_cublas_grouped_bf16_gemm_probe_status",
    "linear_cublas_handle_prewarm_enabled",
    "linear_cublas_handle_prewarm_requested",
    "linear_cublas_handle_prewarm_success_count",
    "linear_cublas_handle_prewarm_failure_count",
    "linear_bf16_workspace_prewarm_enabled",
    "linear_bf16_workspace_prewarm_requested",
    "linear_bf16_workspace_prewarm_success_count",
    "linear_bf16_workspace_prewarm_failure_count",
    "linear_tk_qkv_first_use_prewarm_requested",
    "linear_tk_qkv_first_use_prewarm_requested_count",
    "linear_tk_qkv_first_use_prewarm_enabled_count",
    "linear_tk_qkv_first_use_prewarm_success_count",
    "linear_tk_qkv_first_use_prewarm_failure_count",
    "linear_bf16_gemm_count",
    "linear_bf16_gemm_fast16bf_request_count",
    "bf16_to_f32_vec4_count",
    "lm_head_logits_tk_gemm_count",
    "lm_head_logits_cublaslt_gemm_count",
    "lm_head_logits_bf16_gemm_count",
    "lm_head_dhidden_tk_gemm_count",
    "lm_head_dhidden_cublaslt_gemm_count",
    "lm_head_dhidden_bf16_gemm_count",
    "lm_head_dhidden_strided_vocab_gemm_count",
    "lm_head_dweight_strided_vocab_gemm_count",
    "lm_head_prob_only_corrections_chunk_count",
    "lm_head_prob_only_ce_target_correction_chunk_count",
    "lm_head_prob_only_combined_correction_launch_count",
    "lm_head_prob_only_dhidden_correction_launch_count",
    "lm_head_prob_only_dweight_correction_launch_count",
    "block_backward_dinput_tk_gemm_count",
    "block_backward_dinput_cublaslt_gemm_count",
    "block_backward_dinput_bf16_gemm_count",
    "bf16_persistent_block_input_ln1_backward_count",
    "block_backward_mlp_proj_dinput_before_dweight_count",
    "block_backward_mlp_proj_concurrent_dinput_dweight_count",
    "block_backward_mlp_fc_dinput_before_dweight_count",
    "block_backward_attn_proj_dinput_before_dweight_count",
    "block_backward_attn_proj_first_step_concurrent_dinput_dweight_count",
    "block_backward_qkv_dinput_before_dweight_count",
    "lm_head_cooperative_sequence_launch_count",
    "lm_head_cooperative_sequence_ce_launch_count",
    "lm_head_cooperative_sequence_dhidden_launch_count",
    "lm_head_cooperative_sequence_dweight_launch_count",
    "lm_head_cooperative_sequence_concurrent_count",
    "lm_head_cooperative_sequence_legacy_count",
    "lm_head_cooperative_sequence_loss_bin_count",
    "lm_head_fused_graph_capture_attempt_count",
    "lm_head_fused_graph_capture_success_count",
    "lm_head_fused_graph_upload_success_count",
    "lm_head_fused_graph_upload_failure_count",
    "lm_head_fused_graph_cache_hit_count",
    "lm_head_fused_graph_thread_cache_hit_count",
    "lm_head_fused_graph_cache_entry_count",
    "lm_head_fused_graph_replay_count",
    "lm_head_fused_graph_replay_success_count",
    "lm_head_fused_graph_body_node_count_per_replay",
    "lm_head_fused_graph_body_ce_node_count_per_replay",
    "lm_head_fused_graph_body_dhidden_node_count_per_replay",
    "lm_head_fused_graph_body_dweight_node_count_per_replay",
    "lm_head_fused_graph_body_node_replay_total",
    "lm_head_fused_graph_body_ce_node_replay_total",
    "lm_head_fused_graph_body_dhidden_node_replay_total",
    "lm_head_fused_graph_body_dweight_node_replay_total",
    "lm_head_fused_graph_fallback_count",
    "lm_head_ce_row_loss_reduction_requested",
    "lm_head_ce_row_loss_reduction_available",
    "lm_head_ce_row_loss_reduction_enabled",
    "lm_head_ce_row_loss_sum_accumulate_available",
    "lm_head_ce_row_loss_sum_accumulate_requested",
    "lm_head_ce_row_loss_sum_accumulate_enabled",
    "lm_head_ce_loss_bin_reduction_available",
    "lm_head_ce_loss_bin_reduction_requested",
    "lm_head_ce_loss_bin_reduction_enabled",
    "lm_head_ce_loss_bin_count_requested",
    "lm_head_classifier_chunk_launch_count",
    "lm_head_classifier_loss_bin_launch_count",
    "lm_head_classifier_no_loss_chunk_count",
    "lm_head_classifier_last_rows",
    "lm_head_classifier_last_vocab",
    "lm_head_classifier_last_row_stride",
    "lm_head_prob_only_target_correction_threads",
    "stored_mlp_activation_blocks",
    "stored_mlp_activation_elements",
    "stored_mlp_activation_bytes",
    "stored_packed_attention_activation_blocks",
    "stored_packed_attention_bf16_elements",
    "stored_packed_attention_bf16_bytes",
    "stored_packed_attention_ln1_bf16_blocks",
    "stored_packed_attention_ln1_bf16_elements",
    "stored_packed_attention_ln1_bf16_bytes",
    "stored_packed_attention_lse_elements",
    "stored_packed_attention_lse_bytes",
    "stored_residual1_activation_blocks",
    "stored_residual1_activation_elements",
    "stored_residual1_activation_bytes",
    "block_state_layout.layer_norm_backward_affine_row_chunk_size",
    "block_state_layout.linear_backward_bias_row_chunk_size",
    "block_state_layout.linear_backward_bias_threads_per_block",
    "linear_bf16_a_pack_count",
    "linear_bf16_a_cache_hit_count",
    "attention_forward_tk_launch_count",
    "attention_backward_tk_launch_count",
    "attention_backward_tk_batch_cap",
    "attention_backward_tk_chunk_batch_total",
    "attention_backward_tk_chunk_batch_max",
    "attention_backward_tk_chunk_batch_min",
    "attention_backward_tk_chunk_batch_last",
    "attention_backward_tk_block_size",
    "attention_backward_float_hd64_dprep_launch_count",
    "block_state_layout.mlp_residual_next_ln1_fusion_enabled",
    "block_state_layout.mlp_residual_next_ln1_fusion_count",
    "transformer_device_arena_requested",
    "transformer_device_arena_enabled",
    "transformer_device_arena_cuda_malloc_count",
)
SETUP_ONLY_ROUTE_COUNTER_KEYS = frozenset(
    {
        "linear_cublas_handle_prewarm_enabled",
        "linear_cublas_handle_prewarm_requested",
        "linear_cublas_handle_prewarm_success_count",
        "linear_cublas_handle_prewarm_failure_count",
        "linear_bf16_workspace_prewarm_enabled",
        "linear_bf16_workspace_prewarm_requested",
        "linear_bf16_workspace_prewarm_success_count",
        "linear_bf16_workspace_prewarm_failure_count",
        "linear_tk_qkv_first_use_prewarm_requested",
        "linear_tk_qkv_first_use_prewarm_requested_count",
        "linear_tk_qkv_first_use_prewarm_enabled_count",
        "linear_tk_qkv_first_use_prewarm_success_count",
        "linear_tk_qkv_first_use_prewarm_failure_count",
        "transformer_device_arena_requested",
        "transformer_device_arena_enabled",
        "transformer_device_arena_cuda_malloc_count",
    }
)
NATIVE_STRATEGY_METRIC_KEYS = (
    "status",
    "error",
    "selected_graph_support_status",
    "train_loss_device_accumulation_strategy",
    "train_loss_host_copy_scope",
    "lm_head_training_logits_dtype",
    "lm_head_logits_linear_strategy",
    "lm_head_dhidden_linear_strategy",
    "lm_head_dweight_strategy",
    "lm_head_ce_loss_backward_strategy",
    "lm_head_ce_bf16_threads_per_row",
    "lm_head_prob_only_target_correction_threads",
    "lm_head_ce_bf16_vector_io_strategy",
    "lm_head_ce_bf16_vec_loads_enabled",
    "lm_head_ce_bf16_vec_stores_enabled",
    "lm_head_ce_bf16_vec_normal_stores_enabled",
    "lm_head_ce_reverse_rows_enabled",
    "lm_head_ce_row_order_strategy",
    "lm_head_ce_default_specialized_requested",
    "lm_head_ce_default_specialized_enabled",
    "lm_head_ce_no_loss_default_specialized_requested",
    "lm_head_ce_no_loss_default_specialized_enabled",
    "lm_head_ce_no_loss_llmk_style_specialized_requested",
    "lm_head_ce_no_loss_llmk_style_specialized_enabled",
    "lm_head_prob_only_corrections_requested",
    "lm_head_prob_only_corrections_available",
    "lm_head_prob_only_corrections_enabled",
    "lm_head_prob_only_combined_corrections_requested",
    "lm_head_prob_only_combined_corrections_available",
    "lm_head_prob_only_combined_corrections_enabled",
    "bf16_persistent_block_input_ln1_backward_requested",
    "bf16_persistent_block_input_ln1_backward_enabled",
    "lm_head_ce_llmk_style_specialized_requested",
    "lm_head_ce_llmk_style_specialized_enabled",
    "lm_head_ce_loss_bins_default_specialized_requested",
    "lm_head_ce_loss_bins_default_specialized_enabled",
    "lm_head_classifier_ce_no_loss_requested",
    "lm_head_classifier_ce_no_loss_enabled",
    "lm_head_classifier_fusion_scope",
    "lm_head_schedule_parity_status",
    "lm_head_ce_kernel_strategy",
    "lm_head_bf16_hidden_from_final_norm_requested",
    "lm_head_bf16_hidden_from_final_norm_enabled",
    "lm_head_cooperative_backward_strategy",
    "lm_head_cooperative_backward_required",
    "lm_head_cooperative_backward_abi_wrapper_available",
    "lm_head_cooperative_backward_sequence_wrapper_available",
    "lm_head_cooperative_backward_kernel_available",
    "lm_head_cooperative_backward_fused_kernel_available",
    "lm_head_cooperative_backward_fused_kernel_symbol_available",
    "lm_head_cooperative_backward_fused_kernel_capability_available",
    "lm_head_cooperative_backward_route_integrated",
    "lm_head_cooperative_backward_kernel_enabled",
    "lm_head_cooperative_backward_cuda_graph_available",
    "lm_head_cooperative_backward_cuda_graph_enabled",
    "lm_head_cooperative_backward_graph_prewarm_requested",
    "lm_head_cooperative_backward_graph_prewarm_enabled",
    "lm_head_cooperative_backward_sequence_wrapper_enabled",
    "lm_head_cooperative_backward_fused_kernel_abi_path_class",
    "lm_head_classifier_backward_path_class",
    "lm_head_ce_row_loss_reduction_enabled",
    "lm_head_ce_row_loss_sum_accumulate_enabled",
    "lm_head_dhidden_dweight_schedule_strategy",
    "lm_head_side_stream_count",
    "lm_head_dhidden_stream_enabled",
    "lm_head_dweight_stream_enabled",
    "lm_head_overlap_last_dweight_requested",
    "lm_head_overlap_last_dweight_available",
    "lm_head_overlap_last_dweight_enabled",
    "lm_head_overlap_last_dweight_queue_count",
    "lm_head_overlap_last_dweight_sync_count",
    "lm_head_pipeline_chunks_enabled",
    "lm_head_pipeline_logit_buffer_count",
    "lm_head_pipeline_slot_event_wait_count",
    "lm_head_pipeline_done_event_record_count",
    "activation_tape_count",
    "full_activation_tape_enabled",
    "backward_recompute_blocks",
    "final_block_backward_recompute_elided",
    "activation_tape_strategy",
    "block_state_layout.activation_tape_strategy",
    "linear_cublaslt_grouped_layout_supported",
    "linear_cublaslt_grouped_matmul_probe_requested",
    "linear_cublaslt_grouped_matmul_supported",
    "linear_cublas_grouped_bf16_gemm_probe_requested",
    "linear_cublas_grouped_bf16_gemm_supported",
    "block_forward_linear_strategy",
    "block_backward_input_linear_strategy",
    "block_backward_weight_linear_strategy",
    "block_backward_mlp_proj_concurrent_dinput_dweight_requested",
    "block_backward_mlp_proj_concurrent_dinput_dweight_enabled",
    "block_backward_mlp_fc_concurrent_dinput_dweight_requested",
    "block_backward_pair_streams_available",
    "block_backward_mlp_fc_concurrent_dinput_dweight_enabled",
    "block_backward_qkv_concurrent_dinput_dweight_requested",
    "block_backward_qkv_concurrent_dinput_dweight_enabled",
    "block_backward_attn_proj_concurrent_dinput_dweight_requested",
    "block_backward_attn_proj_concurrent_dinput_dweight_enabled",
    "reuse_packed_ln2_fc_gelu_enabled",
    "fused_ln2_bf16_out_enabled",
    "fused_ln2_bf16_norm_float_store_elision_enabled",
    "stored_mlp_ln2_bf16_prepack_strategy",
    "stored_mlp_forward_strategy",
    "attention_residual_ln2_strategy",
    "stored_packed_attention_lse_enabled",
    "block_state_layout.linear_backward_bias_row_chunk_size",
    "block_state_layout.linear_backward_bias_threads_per_block",
    "optimizer_tile_size",
    "optimizer_tile_strategy",
    "block_state_layout.optimizer_tile_size",
    "block_state_layout.optimizer_tile_strategy",
    "attention_backend_strategy",
    "attention_backward_strategy",
    "attention_backward_tk_block_size",
    "block_state_layout.attention_backward_tk_block_size",
    "block_state_layout.attention_backward_tk_block_size_symbol_loaded",
    "linear_tk_sm120_config_symbol_loaded",
    "linear_tk_sm120_k_tile",
    "linear_tk_sm120_grad_k_tile",
    "linear_tk_sm120_super_m",
    "linear_tk_sm120_dinput_super_m",
    "linear_tk_sm120_dweight_super_m",
    "linear_tk_sm120_huge_n_k_tile",
    "linear_tk_sm120_fast_dgelu_enabled",
    "linear_tk_sm120_approx_dgelu_tanh_enabled",
    "device_allocator_strategy",
    "device_cuda_malloc_async_requested",
    "device_cuda_malloc_async_enabled",
    "device_cuda_malloc_async_symbol_loaded",
    "device_cuda_free_async_symbol_loaded",
    "setup_cuda_event_timing_requested",
    "setup_cuda_event_timing_enabled",
    "float_allocation_strategy",
    "uint16_allocation_strategy",
    "transformer_device_arena_requested",
    "transformer_device_arena_enabled",
    "skip_exit_device_free_enabled",
    "token_weight_init_strategy",
    "token_weight_threaded_init_enabled",
    "token_weight_vector4_init_enabled",
    "token_weight_fast_int32_init_enabled",
    "token_weight_init_legacy_mod17_enabled",
    "token_weight_bf16_initial_refresh_fusion_enabled",
    "token_weight_bf16_initial_refresh_elided",
)
NATIVE_TEXT_METRIC_KEYS = (
    "train_loop_wall_ms_per_step",
    "train_loop_wall_ms",
    "train_loop_cuda_event_wall_ms_per_step",
    "train_loop_cuda_event_first_step_wall_ms_per_step",
    "train_loop_cuda_event_steady_state_wall_ms_per_step",
    "train_loop_cuda_event_wall_ms",
    "train_loop_cuda_event_first_step_wall_ms",
    "train_loop_cuda_event_steady_state_wall_ms",
    "steps_completed",
    "train_tokens_per_second",
    "llm_kittens_bf16_mfu_pct",
    "llm_kittens_last_step_wall_ms",
    "llm_kittens_last_step_tokens_per_second",
    "llm_kittens_last_step_bf16_mfu_pct",
    "llm_kittens_device_memory_used_mib",
    "train_loss_host_d2h_count",
    "train_loss_host_d2h_copies_per_logged_step",
    "train_loss_microbatch_host_d2h_copies_elided_per_logged_step",
    "linear_tk_gemm_count",
    "linear_tk_dweight_gemm_count",
    "linear_tk_dgelu_dinput_gemm_count",
    "linear_cublaslt_gemm_count",
    "linear_cublaslt_bgrad_gemm_count",
    "linear_cublaslt_bgrad_direct_write_count",
    "linear_cublaslt_bgrad_accumulate_count",
    "linear_cublaslt_grouped_layout_probe_status",
    "linear_cublaslt_grouped_matmul_probe_requested",
    "linear_cublaslt_grouped_matmul_probe_status",
    "linear_cublas_grouped_bf16_gemm_probe_requested",
    "linear_cublas_grouped_bf16_gemm_probe_status",
    "linear_cublas_handle_prewarm_enabled",
    "linear_cublas_handle_prewarm_requested",
    "linear_cublas_handle_prewarm_success_count",
    "linear_cublas_handle_prewarm_failure_count",
    "linear_bf16_workspace_prewarm_enabled",
    "linear_bf16_workspace_prewarm_requested",
    "linear_bf16_workspace_prewarm_success_count",
    "linear_bf16_workspace_prewarm_failure_count",
    "linear_tk_qkv_first_use_prewarm_requested",
    "linear_tk_qkv_first_use_prewarm_requested_count",
    "linear_tk_qkv_first_use_prewarm_enabled_count",
    "linear_tk_qkv_first_use_prewarm_success_count",
    "linear_tk_qkv_first_use_prewarm_failure_count",
    "linear_bf16_gemm_count",
    "linear_bf16_gemm_fast16bf_request_count",
    "bf16_to_f32_vec4_count",
    "lm_head_logits_tk_gemm_count",
    "lm_head_logits_cublaslt_gemm_count",
    "lm_head_logits_bf16_gemm_count",
    "lm_head_dhidden_tk_gemm_count",
    "lm_head_dhidden_cublaslt_gemm_count",
    "lm_head_dhidden_bf16_gemm_count",
    "lm_head_dhidden_strided_vocab_gemm_count",
    "lm_head_dweight_strided_vocab_gemm_count",
    "lm_head_prob_only_corrections_chunk_count",
    "lm_head_prob_only_ce_target_correction_chunk_count",
    "lm_head_prob_only_combined_correction_launch_count",
    "lm_head_prob_only_dhidden_correction_launch_count",
    "lm_head_prob_only_dweight_correction_launch_count",
    "block_backward_dinput_tk_gemm_count",
    "block_backward_dinput_cublaslt_gemm_count",
    "block_backward_dinput_bf16_gemm_count",
    "bf16_persistent_block_input_ln1_backward_count",
    "block_backward_mlp_proj_dinput_before_dweight_count",
    "block_backward_mlp_proj_concurrent_dinput_dweight_count",
    "block_backward_mlp_fc_dinput_before_dweight_count",
    "block_backward_attn_proj_dinput_before_dweight_count",
    "block_backward_attn_proj_first_step_concurrent_dinput_dweight_count",
    "block_backward_qkv_dinput_before_dweight_count",
    "lm_head_fused_loss_backward_enabled",
    "lm_head_ce_loss_backward_fused_available",
    "lm_head_ce_loss_backward_fused_enabled",
    "lm_head_ce_loss_backward_strategy",
    "lm_head_ce_reverse_rows_enabled",
    "lm_head_ce_row_order_strategy",
    "lm_head_ce_bf16_vector_io_strategy",
    "lm_head_ce_bf16_vec_loads_enabled",
    "lm_head_ce_bf16_vec_stores_enabled",
    "lm_head_ce_bf16_vec_normal_stores_enabled",
    "lm_head_ce_default_specialized_requested",
    "lm_head_ce_default_specialized_enabled",
    "lm_head_ce_llmk_style_specialized_requested",
    "lm_head_ce_llmk_style_specialized_enabled",
    "lm_head_ce_loss_bins_default_specialized_requested",
    "lm_head_ce_loss_bins_default_specialized_enabled",
    "lm_head_classifier_fusion_scope",
    "lm_head_schedule_parity_status",
    "lm_head_ce_kernel_strategy",
    "lm_head_ce_reverse_rows_enabled",
    "lm_head_ce_row_order_strategy",
    "lm_head_cooperative_backward_required",
    "lm_head_cooperative_backward_abi_wrapper_available",
    "lm_head_cooperative_backward_sequence_wrapper_available",
    "lm_head_cooperative_backward_kernel_available",
    "lm_head_cooperative_backward_fused_kernel_available",
    "lm_head_cooperative_backward_route_integrated",
    "lm_head_cooperative_backward_kernel_enabled",
    "lm_head_cooperative_backward_cuda_graph_available",
    "lm_head_cooperative_backward_cuda_graph_enabled",
    "lm_head_cooperative_backward_graph_prewarm_requested",
    "lm_head_cooperative_backward_graph_prewarm_enabled",
    "lm_head_cooperative_backward_sequence_wrapper_enabled",
    "lm_head_cooperative_backward_fused_kernel_abi_path_class",
    "lm_head_classifier_backward_path_class",
    "lm_head_classifier_chunk_kernel_available",
    "lm_head_classifier_chunk_kernel_enabled",
    "lm_head_classifier_ce_no_loss_requested",
    "lm_head_classifier_ce_no_loss_enabled",
    "lm_head_ce_row_loss_reduction_requested",
    "lm_head_ce_row_loss_reduction_available",
    "lm_head_ce_row_loss_reduction_enabled",
    "lm_head_ce_row_loss_sum_accumulate_available",
    "lm_head_ce_row_loss_sum_accumulate_requested",
    "lm_head_ce_row_loss_sum_accumulate_enabled",
    "lm_head_ce_loss_bin_reduction_available",
    "lm_head_ce_loss_bin_reduction_requested",
    "lm_head_ce_loss_bin_reduction_enabled",
    "lm_head_ce_loss_bin_count_requested",
    "lm_head_classifier_chunk_launch_count",
    "lm_head_classifier_loss_bin_launch_count",
    "lm_head_cooperative_sequence_launch_count",
    "lm_head_cooperative_sequence_ce_launch_count",
    "lm_head_cooperative_sequence_dhidden_launch_count",
    "lm_head_cooperative_sequence_dweight_launch_count",
    "lm_head_cooperative_sequence_concurrent_count",
    "lm_head_cooperative_sequence_legacy_count",
    "lm_head_cooperative_sequence_loss_bin_count",
    "lm_head_fused_graph_capture_attempt_count",
    "lm_head_fused_graph_capture_success_count",
    "lm_head_fused_graph_upload_success_count",
    "lm_head_fused_graph_upload_failure_count",
    "lm_head_fused_graph_cache_hit_count",
    "lm_head_fused_graph_thread_cache_hit_count",
    "lm_head_fused_graph_cache_entry_count",
    "lm_head_fused_graph_replay_count",
    "lm_head_fused_graph_replay_success_count",
    "lm_head_fused_graph_body_node_count_per_replay",
    "lm_head_fused_graph_body_ce_node_count_per_replay",
    "lm_head_fused_graph_body_dhidden_node_count_per_replay",
    "lm_head_fused_graph_body_dweight_node_count_per_replay",
    "lm_head_fused_graph_body_node_replay_total",
    "lm_head_fused_graph_body_ce_node_replay_total",
    "lm_head_fused_graph_body_dhidden_node_replay_total",
    "lm_head_fused_graph_body_dweight_node_replay_total",
    "lm_head_fused_graph_fallback_count",
    "lm_head_fused_graph_prewarm_attempt_count",
    "lm_head_fused_graph_prewarm_success_count",
    "lm_head_fused_graph_prewarm_failure_count",
    "lm_head_fused_graph_prewarm_last_error_code",
    "lm_head_fused_graph_prewarm_cache_hit_count",
    "lm_head_fused_graph_prewarm_cache_entry_count",
    "lm_head_classifier_last_rows",
    "lm_head_classifier_last_vocab",
    "lm_head_classifier_last_row_stride",
    "stored_mlp_activation_blocks",
    "stored_mlp_activation_elements",
    "stored_mlp_activation_bytes",
    "stored_packed_attention_activation_blocks",
    "stored_packed_attention_bf16_elements",
    "stored_packed_attention_bf16_bytes",
    "stored_packed_attention_ln1_bf16_blocks",
    "stored_packed_attention_ln1_bf16_elements",
    "stored_packed_attention_ln1_bf16_bytes",
    "stored_packed_attention_lse_elements",
    "stored_packed_attention_lse_bytes",
    "stored_residual1_activation_blocks",
    "stored_residual1_activation_elements",
    "stored_residual1_activation_bytes",
    "linear_bf16_a_pack_count",
    "linear_bf16_a_cache_hit_count",
    "attention_forward_tk_launch_count",
    "attention_backward_tk_launch_count",
    "attention_backward_tk_batch_cap",
    "attention_backward_tk_chunk_batch_total",
    "attention_backward_tk_chunk_batch_max",
    "attention_backward_tk_chunk_batch_min",
    "attention_backward_tk_chunk_batch_last",
    "attention_backward_tk_block_size",
    "concurrent_arena_materialize_count",
    "lm_head_classifier.reference_full_bf16_logit_bytes",
    "lm_head_classifier.native_chunk_bf16_logit_bytes",
    "lm_head_classifier.resident_logit_reduction_ratio",
    "lm_head_classifier.native_logit_chunk_rows",
    "lm_head_classifier.native_logit_chunk_count",
    "setup_wall_ms",
    "setup.float_arena_materialize.total_ms",
    "setup.uint16_arena_materialize.total_ms",
    "setup.token_weight_init.total_ms",
    "setup.zero_init.total_ms",
    "setup.block_weight_bf16_initial_refresh.total_ms",
    "setup.cuda_event.zero_init.total_ms",
    "setup.cuda_event.token_weight_init.total_ms",
    "setup.cuda_event.token_weight_bf16_initial_refresh.total_ms",
    "setup.cuda_event.nonzero_parameter_fill.total_ms",
    "setup.cuda_event.block_weight_bf16_initial_refresh.total_ms",
    "checkpoint_wall_ms",
    "total_wall_ms",
    "stage.train.model_forward.total_ms",
    "stage.train.lm_head_loss.total_ms",
    "stage.block_forward.total_ms",
    "stage.block_forward.attention.total_ms",
    "stage.block_forward.attention.sdpa.total_ms",
    "stage.block_forward.mlp_fc_gelu.total_ms",
    "stage.block_forward.mlp_proj.total_ms",
    "stage.block_recompute.total_ms",
    "stage.block_recompute.attention.total_ms",
    "stage.block_recompute.attention.sdpa.total_ms",
    "stage.block_recompute.mlp_fc_gelu.total_ms",
    "stage.block_recompute.mlp_proj.total_ms",
    "stage.lm_head_backward.total_ms",
    "stage.lm_head_backward.hidden_prepack.total_ms",
    "stage.lm_head_backward.logits.total_ms",
    "stage.lm_head_backward.loss_accumulate.total_ms",
    "stage.lm_head_backward.ce.total_ms",
    "stage.lm_head_backward.cooperative.total_ms",
    "stage.lm_head_backward.dhidden.total_ms",
    "stage.lm_head_backward.dweight.total_ms",
    "stage.lm_head_backward.dhidden_dweight_concurrent.total_ms",
    "stage.lm_head_backward.pipeline_queue.total_ms",
    "stage.lm_head_backward.pipeline_final_wait.total_ms",
    "stage.lm_head_backward.loss_copy.total_ms",
    "stage.final_norm_backward.total_ms",
    "stage.block_backward.total_ms",
    "stage.block_backward.mlp_fc.total_ms",
    "stage.block_backward.mlp_proj.total_ms",
    "stage.block_backward.mlp_proj.grad_out_bf16.total_ms",
    "stage.block_backward.mlp_proj.dweight_bias.total_ms",
    "stage.block_backward.mlp_proj.dinput.total_ms",
    "stage.block_backward.mlp_proj.gelu.total_ms",
    "stage.block_backward.mlp_fc.total_ms",
    "stage.block_backward.mlp_fc.dweight_bias.total_ms",
    "stage.block_backward.mlp_fc.dinput.total_ms",
    "stage.block_backward.ln2_residual.total_ms",
    "stage.block_backward.ln2_residual.fused_affine_dinput_add.total_ms",
    "stage.block_backward.ln2_residual.affine.total_ms",
    "stage.block_backward.ln2_residual.dinput.total_ms",
    "stage.block_backward.ln2_residual.add.total_ms",
    "stage.block_backward.attn_proj.total_ms",
    "stage.block_backward.attn_proj.dweight_bias.total_ms",
    "stage.block_backward.attn_proj.dinput.total_ms",
    "stage.block_backward.attn_proj.dinput_dweight_concurrent.total_ms",
    "stage.block_backward.attn_sdpa.total_ms",
    "stage.block_backward.attn_sdpa.grad_out_bf16.total_ms",
    "stage.block_backward.attn_sdpa.to_qkv.total_ms",
    "attention_backward_dprep_timing_us",
    "attention_backward_dprep_timing_count",
    "attention_backward_tk_timing_us",
    "attention_backward_tk_timing_count",
    "stage.block_backward.qkv.total_ms",
    "stage.block_backward.qkv.dweight_bias.total_ms",
    "stage.block_backward.qkv.dinput.total_ms",
    "stage.block_backward.qkv.dinput_dweight_concurrent.total_ms",
    "stage.block_backward.ln1_residual.total_ms",
    "stage.block_backward.ln1_residual.fused_affine_dinput_add.total_ms",
    "stage.block_backward.ln1_residual.affine.total_ms",
    "stage.block_backward.ln1_residual.dinput_add.total_ms",
    "stage.block_backward.ln1_residual.dinput.total_ms",
    "stage.block_backward.ln1_residual.add.total_ms",
    "stage.embedding_backward.total_ms",
    "stage.gradient_zero.total_ms",
    "stage.gradient_clip.total_ms",
    "stage.adamw_update.total_ms",
)
NATIVE_HOT_SUMMARY_METRIC_KEYS = (
    "train_loop_wall_ms_per_step",
    "train_loop_cuda_event_wall_ms_per_step",
    "train_loop_cuda_event_first_step_wall_ms_per_step",
    "train_loop_cuda_event_steady_state_wall_ms_per_step",
    "train_tokens_per_second",
    "setup_wall_ms",
    "setup.float_uint16_arena_materialize_concurrent.total_ms",
    "setup.float_arena_materialize.total_ms",
    "float_arena_cuda_malloc_wall_ms",
    "float_arena_pointer_assign_wall_ms",
    "setup.uint16_arena_materialize.total_ms",
    "uint16_arena_cuda_malloc_wall_ms",
    "uint16_arena_pointer_assign_wall_ms",
    "uint16_arena_first_enabled",
    "arena_materialize_order",
    "transformer_device_arena_cuda_malloc_wall_ms",
    "transformer_device_arena_pointer_assign_wall_ms",
    "setup.token_weight_init.total_ms",
    "setup.cublaslt_plan_prewarm.total_ms",
    "stage.train.model_forward.total_ms",
    "stage.train.model_forward.first_step_avg_ms",
    "stage.train.model_forward.steady_state_avg_ms",
    "stage.block_forward.total_ms",
    "stage.block_forward.first_step_avg_ms",
    "stage.block_forward.steady_state_avg_ms",
    "stage.block_forward.attention.qkv.first_step_avg_ms",
    "stage.block_forward.attention.qkv.steady_state_avg_ms",
    "stage.lm_head_backward.total_ms",
    "stage.lm_head_backward.logits.total_ms",
    "stage.lm_head_backward.cooperative.total_ms",
    "stage.block_backward.total_ms",
    "stage.block_backward.attn_sdpa.to_qkv.total_ms",
    "stage.block_backward.mlp_proj.total_ms",
    "stage.block_backward.mlp_proj.dweight_bias.total_ms",
    "stage.block_backward.mlp_proj.dinput.total_ms",
    "stage.block_backward.mlp_fc.total_ms",
    "stage.block_backward.attn_proj.total_ms",
    "stage.block_backward.qkv.total_ms",
    "stage.adamw_update.total_ms",
)
NATIVE_JSON_OUT_FLAGS = ("--json-out", "--profile-json", "--stage-profile-json")

LLM_KITTENS_STEP_RE = re.compile(
    r"^step\s+\d+/\d+\s+\|.*?\|\s+"
    r"(?P<step_ms>[0-9]+(?:\.[0-9]+)?)\s+ms\s+\|\s+"
    r"(?P<mfu>[0-9]+(?:\.[0-9]+)?)%\s+bf16\s+MFU\s+\|\s+"
    r"(?P<tok_s>[0-9]+(?:\.[0-9]+)?)\s+tok/s\s*$",
    re.MULTILINE,
)
LLM_KITTENS_MEMORY_RE = re.compile(
    r"^device memory usage:\s+"
    r"(?P<used>[0-9]+)\s+MiB\s+/\s+(?P<total>[0-9]+)\s+MiB\s*$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Older/baseline command, shell-quoted as one string.")
    parser.add_argument("--candidate", required=True, help="Candidate command, shell-quoted as one string.")
    parser.add_argument(
        "--reference",
        default="",
        help=(
            "Optional external reference command, shell-quoted as one string. When set, "
            "each sample runs baseline, candidate, and reference in rotated order and "
            "reports reference_over_baseline plus candidate_over_reference ratios."
        ),
    )
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied only to the baseline command. Repeat for multiple variables.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied only to the candidate command. Repeat for multiple variables.",
    )
    parser.add_argument(
        "--reference-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied only to the optional reference command.",
    )
    parser.add_argument("--samples", type=int, default=5, help="Paired samples to collect.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup command pairs before measurement.")
    parser.add_argument(
        "--cuda-visible-devices",
        default="dedicated",
        help=(
            "Set CUDA_VISIBLE_DEVICES for both commands. The default 'dedicated' requires "
            "an idle display-disabled NVIDIA GPU from nvidia-smi. Pass 'auto' to allow "
            "fallback to the lowest-utilization NVIDIA GPU, an explicit device id such as 0, "
            "or an empty string to leave the environment unchanged."
        ),
    )
    parser.add_argument(
        "--cuda-device-max-connections",
        default="1",
        help="Set CUDA_DEVICE_MAX_CONNECTIONS for both commands. Pass an empty string to leave it unchanged.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    parser.add_argument("--json-out", default="", help="Write the JSON payload to this file.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Attach arbitrary run metadata to the text and JSON output. Repeat for multiple "
            "fields; useful for wrapper-level candidate profile or build-flag details."
        ),
    )
    parser.add_argument(
        "--append-native-profile-json-dir",
        default="",
        help=(
            "Append a unique --profile-json PATH under this directory to native NeuralFn commands "
            "that do not already specify --json-out, --profile-json, or --stage-profile-json."
        ),
    )
    parser.add_argument(
        "--native-stage-timing",
        action="store_true",
        help=(
            "Set NFN_NATIVE_GPT_STAGE_TIMING=1 for native NeuralFn commands. "
            "Use this for attribution runs; leave it off for throughput comparisons."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed commands instead of stopping at the first nonzero exit.",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help=(
            "Resolve commands, environment, profile settings, and CUDA device selection "
            "without launching warmup or measured commands."
        ),
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Per-command timeout. The default 0 disables the timeout. With "
            "--continue-on-error, timed-out commands are recorded with timed_out=true; "
            "otherwise the run stops at the first timeout."
        ),
    )
    parser.add_argument(
        "--require-idle-selected-gpu",
        action="store_true",
        help=(
            "Abort before warmup or measured samples if nvidia-smi reports any compute "
            "process on the selected CUDA GPU."
        ),
    )
    parser.add_argument(
        "--max-selected-gpu-utilization-pct",
        type=float,
        default=-1.0,
        help=(
            "Abort before warmup or measured samples when the selected CUDA GPU's "
            "nvidia-smi utilization exceeds this percentage. Negative values disable "
            "the utilization guard."
        ),
    )
    parser.add_argument(
        "--selected-gpu-utilization-retries",
        type=int,
        default=3,
        help=(
            "Number of nvidia-smi utilization snapshots to try before rejecting an "
            "otherwise idle selected GPU. Retries only help when the selected GPU has "
            "no compute processes; compute processes still fail immediately."
        ),
    )
    parser.add_argument(
        "--selected-gpu-utilization-retry-interval-seconds",
        type=float,
        default=0.25,
        help=(
            "Seconds to wait between selected-GPU utilization guard retries. The "
            "default smooths transient WSL/NVML idle samples without hiding real "
            "compute load."
        ),
    )
    parser.add_argument(
        "--allow-stale-selected-gpu-utilization-without-compute-processes",
        action="store_true",
        help=(
            "After selected-GPU utilization retries are exhausted, allow the run "
            "to continue when nvidia-smi still reports high utilization but the "
            "selected GPU has no compute processes. This is intended for WSL/NVML "
            "stale-utilization samples on a display-disabled dedicated compute GPU; "
            "active compute processes still fail immediately."
        ),
    )
    parser.add_argument(
        "--no-gpu-benchmark-lock",
        action="store_true",
        help=(
            "Disable the per-selected-GPU benchmark lock. By default, GPU-visible "
            "runs lock /tmp/nfn_paired_kernel_speed_gpu_<device>.lock so overlapping "
            "paired benchmarks fail fast instead of contaminating candidate timings."
        ),
    )
    parser.add_argument(
        "--gpu-benchmark-lock-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait for the per-selected-GPU benchmark lock. The default 0 "
            "fails immediately when another paired benchmark already owns the lock."
        ),
    )
    parser.add_argument(
        "--max-candidate-ratio",
        action="append",
        default=[],
        metavar="[STAT:]METRIC=RATIO",
        help=(
            "Fail after measurement if the candidate-over-baseline ratio statistic for METRIC exceeds RATIO. "
            "STAT defaults to mean and may be mean, median, min, or max, e.g. "
            "median:train_loop_wall_ms_per_step=1.000. "
            "Use native metric names such as stage.lm_head_backward.total_ms or "
            "train_loop_wall_ms_per_step. Repeat for multiple hot-bucket gates."
        ),
    )
    parser.add_argument(
        "--min-candidate-ratio",
        action="append",
        default=[],
        metavar="[STAT:]METRIC=RATIO",
        help=(
            "Fail after measurement if the candidate-over-baseline ratio statistic for METRIC is below RATIO. "
            "STAT defaults to mean and may be mean, median, min, or max, e.g. "
            "mean:train_tokens_per_second=1.000. This is useful for speed-up gates and "
            "route counters that must not disappear."
        ),
    )
    parser.add_argument(
        "--max-candidate-reference-ratio",
        action="append",
        default=[],
        metavar="[STAT:]METRIC=RATIO",
        help=(
            "Fail after measurement if the candidate-over-reference ratio statistic for METRIC exceeds RATIO. "
            "This requires --reference and lets a candidate prove it beats both the old native route and "
            "the external reference in the same selected-GPU run."
        ),
    )
    parser.add_argument(
        "--min-candidate-reference-ratio",
        action="append",
        default=[],
        metavar="[STAT:]METRIC=RATIO",
        help=(
            "Fail after measurement if the candidate-over-reference ratio statistic for METRIC is below RATIO. "
            "This requires --reference and is useful for throughput metrics such as train_tokens_per_second."
        ),
    )
    parser.add_argument(
        "--require-native-route-change",
        action="store_true",
        help=(
            "Fail after measurement when candidate native metrics do not show any "
            "tracked route-counter, strategy-value, linear-shape, or cuBLASLt-plan "
            "change. Use this for kernel/profile promotion gates so timing-only "
            "noise cannot pass as an implementation change."
        ),
    )
    parser.add_argument(
        "--require-native-lm-head-true-fused",
        action="store_true",
        help=(
            "Fail after measurement when candidate native metrics show the LM-head "
            "backward path is still the diagnostic CUDA Graph wrapper or the strict "
            "true-fused Tile capability is unavailable."
        ),
    )
    return parser.parse_args()


def parse_env_overrides(values: Sequence[str], *, option_name: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"{option_name} expects KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not key:
            raise SystemExit(f"{option_name} expects a non-empty environment variable name")
        overrides[key] = value
    return overrides


def parse_metric_ratio_limits(
    values: Sequence[str],
    *,
    option_name: str,
    bound: str,
) -> list[MetricRatioLimit]:
    limits: list[MetricRatioLimit] = []
    if bound not in {"min", "max"}:
        raise ValueError(f"unsupported metric ratio bound: {bound}")
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"{option_name} expects [STAT:]METRIC=RATIO, got {raw!r}")
        metric_text, ratio_text = raw.split("=", 1)
        stat = "mean"
        metric = metric_text.strip()
        if ":" in metric:
            maybe_stat, maybe_metric = metric.split(":", 1)
            maybe_stat = maybe_stat.strip().lower()
            if maybe_stat in {"mean", "median", "min", "max"}:
                stat = maybe_stat
                metric = maybe_metric.strip()
        metric = metric.strip()
        if not metric:
            raise SystemExit(f"{option_name} expects a non-empty metric name")
        try:
            ratio = float(ratio_text)
        except ValueError as exc:
            raise SystemExit(
                f"{option_name} expects a numeric ratio for {metric!r}, got {ratio_text!r}"
            ) from exc
        if ratio <= 0.0:
            raise SystemExit(f"{option_name} for {metric!r} must be positive")
        if bound == "min":
            limits.append(MetricRatioLimit(metric=metric, stat=stat, min_ratio=ratio))
        else:
            limits.append(MetricRatioLimit(metric=metric, stat=stat, max_ratio=ratio))
    return limits


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def value_at_path(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def native_metrics_from_payload(payload: dict[str, Any]) -> dict[str, float | int | str | bool]:
    metrics: dict[str, float | int | str | bool] = {}
    for name, path in NATIVE_METRIC_PATHS:
        value = value_at_path(payload, path)
        if isinstance(value, (bool, int, float, str)):
            metrics[name] = value
    for key in NATIVE_STRATEGY_METRIC_KEYS:
        value = payload.get(key)
        if isinstance(value, (bool, int, float, str)):
            metrics[key] = value
    train_loop_ms = metrics.get("train_loop_wall_ms")
    steps_completed = metrics.get("steps_completed")
    if (
        isinstance(train_loop_ms, (int, float))
        and not isinstance(train_loop_ms, bool)
        and isinstance(steps_completed, (int, float))
        and not isinstance(steps_completed, bool)
        and float(steps_completed) > 0.0
    ):
        metrics["train_loop_wall_ms_per_step"] = float(train_loop_ms) / float(steps_completed)
    timing = payload.get("timing")
    if isinstance(timing, dict):
        setup_timing = timing.get("setup_timing")
        if isinstance(setup_timing, list):
            for stage in setup_timing:
                if not isinstance(stage, dict):
                    continue
                name = stage.get("name")
                if not isinstance(name, str) or not name:
                    continue
                metric_name = name if name.startswith("setup.") else "setup." + name
                for source_key, suffix in (
                    ("total_ms", "total_ms"),
                    ("avg_ms", "avg_ms"),
                    ("count", "count"),
                ):
                    value = stage.get(source_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"{metric_name}.{suffix}"] = value
        setup_cuda_timing = timing.get("setup_cuda_event_timing")
        if isinstance(setup_cuda_timing, list):
            for stage in setup_cuda_timing:
                if not isinstance(stage, dict):
                    continue
                name = stage.get("name")
                if not isinstance(name, str) or not name:
                    continue
                metric_name = name if name.startswith("setup.") else "setup." + name
                metric_name = metric_name.replace("setup.", "setup.cuda_event.", 1)
                for source_key, suffix in (
                    ("total_ms", "total_ms"),
                    ("avg_ms", "avg_ms"),
                    ("count", "count"),
                ):
                    value = stage.get(source_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"{metric_name}.{suffix}"] = value
        stage_timing = timing.get("stage_timing")
        if isinstance(stage_timing, list):
            for stage in stage_timing:
                if not isinstance(stage, dict):
                    continue
                name = stage.get("name")
                if not isinstance(name, str) or not name:
                    continue
                metric_name = "stage." + name
                for source_key, suffix in (
                    ("total_ms", "total_ms"),
                    ("avg_ms", "avg_ms"),
                    ("count", "count"),
                    ("first_step_total_ms", "first_step_total_ms"),
                    ("first_step_avg_ms", "first_step_avg_ms"),
                    ("first_step_count", "first_step_count"),
                    ("steady_state_total_ms", "steady_state_total_ms"),
                    ("steady_state_avg_ms", "steady_state_avg_ms"),
                    ("steady_state_count", "steady_state_count"),
                ):
                    value = stage.get(source_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"{metric_name}.{suffix}"] = value
    return metrics


def native_linear_shape_stats_from_payload(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows = payload.get("linear_shape_stats")
    if not isinstance(rows, list):
        return []
    stats: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path_name = row.get("path_name")
        m = row.get("m")
        n = row.get("n")
        k = row.get("k")
        op_a_name = row.get("op_a_name")
        op_b_name = row.get("op_b_name")
        if not (
            isinstance(path_name, str)
            and isinstance(m, int)
            and isinstance(n, int)
            and isinstance(k, int)
            and isinstance(op_a_name, str)
            and isinstance(op_b_name, str)
        ):
            continue
        item: dict[str, object] = {
            "path_name": path_name,
            "m": m,
            "n": n,
            "k": k,
            "op_a_name": op_a_name,
            "op_b_name": op_b_name,
            "calls": row.get("calls"),
            "total_us": row.get("total_us"),
            "avg_us": row.get("avg_us"),
        }
        for key in (
            "cublaslt_selected_heuristic",
            "cublaslt_returned_heuristics",
            "cublaslt_workspace_bytes",
        ):
            value = row.get(key)
            if isinstance(value, int):
                item[key] = value
        stats.append(item)
    return stats


def native_cublaslt_plan_cache_from_payload(payload: dict[str, Any]) -> list[dict[str, object]]:
    rows = payload.get("linear_cublaslt_plan_cache")
    if not isinstance(rows, list):
        return []
    plans: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        m = row.get("m")
        n = row.get("n")
        k = row.get("k")
        op_a_name = row.get("op_a_name")
        op_b_name = row.get("op_b_name")
        if not (
            isinstance(m, int)
            and isinstance(n, int)
            and isinstance(k, int)
            and isinstance(op_a_name, str)
            and isinstance(op_b_name, str)
        ):
            continue
        item: dict[str, object] = {
            "m": m,
            "n": n,
            "k": k,
            "op_a_name": op_a_name,
            "op_b_name": op_b_name,
        }
        for key in (
            "selected_heuristic",
            "returned_heuristics",
            "workspace_bytes",
            "epilogue",
        ):
            value = row.get(key)
            if isinstance(value, int):
                item[key] = value
        plans.append(item)
    return plans


def native_json_out_path_from_argv(argv: Sequence[str]) -> Path | None:
    for index, arg in enumerate(argv):
        if arg in NATIVE_JSON_OUT_FLAGS and index + 1 < len(argv):
            return Path(argv[index + 1])
        for flag in NATIVE_JSON_OUT_FLAGS:
            prefix = flag + "="
            if arg.startswith(prefix):
                return Path(arg[len(prefix) :])
    return None


def looks_like_neuralfn_native_command(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    executable = Path(argv[0]).name
    return executable in {
        "nfn_gpt_native_train_linked",
        "nfn_gpt_native_train",
        "nfn-native-train",
        "nfn-gpt-native",
        "nfn-gpt-native-train",
    }


def argv_with_auto_profile_json(
    argv: Sequence[str],
    *,
    command_name: str,
    profile_json_dir: Path | None,
) -> list[str]:
    next_argv = list(argv)
    if profile_json_dir is None:
        return next_argv
    if not looks_like_neuralfn_native_command(next_argv):
        return next_argv
    if native_json_out_path_from_argv(next_argv) is not None:
        return next_argv
    profile_json_dir.mkdir(parents=True, exist_ok=True)
    path = profile_json_dir / f"{command_name}_{time.time_ns()}.json"
    next_argv.extend(["--profile-json", str(path)])
    return next_argv


def native_payload_from_json_out(argv: Sequence[str]) -> dict[str, Any] | None:
    path = native_json_out_path_from_argv(argv)
    if path is None or not path.exists():
        return None
    return extract_json_object(path.read_text(encoding="utf-8", errors="replace"))


def native_payload_from_command_output(argv: Sequence[str], stdout: str) -> dict[str, Any] | None:
    payload = extract_json_object(stdout)
    if payload is not None:
        return payload
    return native_payload_from_json_out(argv)


def command_env_with_auto_stage_timing(
    command: TimedCommand,
    *,
    env: dict[str, str] | None,
    native_stage_timing: bool,
) -> dict[str, str] | None:
    command_env = env
    if command.env_overrides:
        command_env = dict(os.environ if env is None else env)
        command_env.update(command.env_overrides)
    if native_stage_timing and looks_like_neuralfn_native_command(command.argv):
        command_env = dict(os.environ if command_env is None else command_env)
        command_env.setdefault("NFN_NATIVE_GPT_STAGE_TIMING", "1")
    return command_env


def native_metrics_from_json_out(argv: Sequence[str]) -> dict[str, float | int | str | bool]:
    payload = native_payload_from_json_out(argv)
    return native_metrics_from_payload(payload) if payload is not None else {}


def native_metrics_from_command_output(argv: Sequence[str], stdout: str) -> dict[str, float | int | str | bool]:
    payload = extract_json_object(stdout)
    if payload is not None:
        return native_metrics_from_payload(payload)
    payload = native_payload_from_json_out(argv)
    if payload is not None:
        metrics = native_metrics_from_payload(payload)
        if metrics:
            metrics["native_metrics_source"] = "json-out"
        return metrics
    return llm_kittens_metrics_from_stdout(stdout)


def native_linear_shape_stats_from_command_output(argv: Sequence[str], stdout: str) -> list[dict[str, object]]:
    payload = native_payload_from_command_output(argv, stdout)
    return native_linear_shape_stats_from_payload(payload) if payload is not None else []


def native_cublaslt_plan_cache_from_command_output(argv: Sequence[str], stdout: str) -> list[dict[str, object]]:
    payload = native_payload_from_command_output(argv, stdout)
    return native_cublaslt_plan_cache_from_payload(payload) if payload is not None else []


def native_failure_summary(metrics: dict[str, float | int | str | bool]) -> str:
    fields: list[str] = []
    for key in ("status", "error"):
        value = metrics.get(key)
        if isinstance(value, (str, int, float, bool)):
            fields.append(f"{key}: {value}")
    return "\n".join(fields)


def native_metrics_from_stdout(stdout: str) -> dict[str, float | int | str | bool]:
    payload = extract_json_object(stdout)
    if payload is None:
        return llm_kittens_metrics_from_stdout(stdout)
    return native_metrics_from_payload(payload)


def llm_kittens_metrics_from_stdout(stdout: str) -> dict[str, float | int | str | bool]:
    metrics: dict[str, float | int | str | bool] = {}
    step_matches = list(LLM_KITTENS_STEP_RE.finditer(stdout))
    if step_matches:
        step_ms_values = [float(match.group("step_ms")) for match in step_matches]
        tok_s_values = [float(match.group("tok_s")) for match in step_matches]
        mfu_values = [float(match.group("mfu")) for match in step_matches]
        metrics["status"] = "llm-kittens-step-log"
        metrics["train_loop_wall_ms"] = sum(step_ms_values)
        metrics["train_loop_wall_ms_per_step"] = mean(step_ms_values)
        metrics["train_loop_cuda_event_wall_ms"] = sum(step_ms_values)
        metrics["train_loop_cuda_event_wall_ms_per_step"] = mean(step_ms_values)
        metrics["train_loop_cuda_event_first_step_wall_ms"] = step_ms_values[0]
        metrics["train_loop_cuda_event_first_step_wall_ms_per_step"] = step_ms_values[0]
        if len(step_ms_values) > 1:
            steady_state_values = step_ms_values[1:]
            metrics["train_loop_cuda_event_steady_state_wall_ms"] = sum(steady_state_values)
            metrics["train_loop_cuda_event_steady_state_wall_ms_per_step"] = mean(steady_state_values)
        metrics["train_tokens_per_second"] = mean(tok_s_values)
        metrics["llm_kittens_bf16_mfu_pct"] = mean(mfu_values)
        metrics["llm_kittens_last_step_wall_ms"] = step_ms_values[-1]
        metrics["llm_kittens_last_step_tokens_per_second"] = tok_s_values[-1]
        metrics["llm_kittens_last_step_bf16_mfu_pct"] = mfu_values[-1]
        metrics["llm_kittens_step_log_count"] = len(step_matches)
    memory_match = LLM_KITTENS_MEMORY_RE.search(stdout)
    if memory_match:
        metrics["llm_kittens_device_memory_used_mib"] = int(memory_match.group("used"))
        metrics["llm_kittens_device_memory_total_mib"] = int(memory_match.group("total"))
    return metrics


def timeout_output_to_text(value: str | bytes | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def terminate_process_group(
    proc: subprocess.Popen[str],
    *,
    first_signal: int,
    final_signal: int = signal.SIGKILL,
    wait_seconds: float = 5.0,
) -> int:
    """Terminate a command and its process group, returning its final code."""
    try:
        process_group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        process_group_id = proc.pid
    try:
        os.killpg(process_group_id, first_signal)
    except ProcessLookupError:
        pass
    except PermissionError:
        if first_signal == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    try:
        proc.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, final_signal)
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.kill()
        try:
            proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            return -1
    return proc.returncode if proc.returncode is not None else -1


def kill_timed_out_process(proc: subprocess.Popen[str]) -> int:
    """Kill a timed-out command and its process group, returning its final code."""
    return terminate_process_group(proc, first_signal=signal.SIGKILL)


def run_once(
    command: TimedCommand,
    *,
    continue_on_error: bool,
    env: dict[str, str] | None,
    timeout_seconds: float | None,
    profile_json_dir: Path | None,
    native_stage_timing: bool,
    gpu_before: dict[str, object] | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    run_argv = argv_with_auto_profile_json(
        command.argv,
        command_name=command.name,
        profile_json_dir=profile_json_dir,
    )
    command_env = command_env_with_auto_stage_timing(
        command,
        env=env,
        native_stage_timing=native_stage_timing,
    )
    try:
        proc = subprocess.Popen(
            run_argv,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=command_env,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - start
        stdout = timeout_output_to_text(exc.stdout)
        stderr = timeout_output_to_text(exc.stderr)
        process_returncode = -1
        if "proc" in locals():
            process_returncode = kill_timed_out_process(proc)
            try:
                killed_stdout, killed_stderr = proc.communicate(timeout=5.0)
                stdout += timeout_output_to_text(killed_stdout)
                stderr += timeout_output_to_text(killed_stderr)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not continue_on_error:
            raise SystemExit(
                f"{command.name} timed out after {timeout_seconds:.3f}s\n"
                f"command: {shlex.join(run_argv)}\n"
                f"stdout tail:\n{stdout[-2000:]}\n"
                f"stderr tail:\n{stderr[-2000:]}"
            ) from exc
        result = {
            "name": command.name,
            "argv": run_argv,
            "seconds": seconds,
            "returncode": -1,
            "process_returncode": process_returncode,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "native_metrics": native_metrics_from_command_output(run_argv, stdout),
            "native_linear_shape_stats": native_linear_shape_stats_from_command_output(run_argv, stdout),
            "native_cublaslt_plan_cache": native_cublaslt_plan_cache_from_command_output(run_argv, stdout),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        if gpu_before is not None:
            result["gpu_before"] = gpu_before
            result["gpu_after"] = gpu_snapshot()
        return result
    except KeyboardInterrupt:
        if "proc" in locals():
            terminate_process_group(proc, first_signal=signal.SIGTERM, wait_seconds=2.0)
        raise SystemExit(
            f"interrupted while running {command.name}; terminated child process group"
        ) from None
    except BaseException:
        if "proc" in locals():
            terminate_process_group(proc, first_signal=signal.SIGTERM, wait_seconds=2.0)
        raise
    seconds = time.perf_counter() - start
    returncode = proc.returncode if proc.returncode is not None else -1
    native_metrics = native_metrics_from_command_output(run_argv, stdout)
    native_linear_shape_stats = native_linear_shape_stats_from_command_output(run_argv, stdout)
    native_cublaslt_plan_cache = native_cublaslt_plan_cache_from_command_output(run_argv, stdout)
    if returncode != 0 and not continue_on_error:
        native_summary = native_failure_summary(native_metrics)
        native_section = f"\nnative JSON summary:\n{native_summary}\n" if native_summary else ""
        raise SystemExit(
            f"{command.name} failed with exit {returncode}\n"
            f"command: {shlex.join(run_argv)}\n"
            f"{native_section}"
            f"stdout tail:\n{stdout[-2000:]}\n"
            f"stderr tail:\n{stderr[-2000:]}"
        )
    result = {
        "name": command.name,
        "argv": run_argv,
        "seconds": seconds,
        "returncode": returncode,
        "timed_out": False,
        "native_metrics": native_metrics,
        "native_linear_shape_stats": native_linear_shape_stats,
        "native_cublaslt_plan_cache": native_cublaslt_plan_cache,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    if gpu_before is not None:
        result["gpu_before"] = gpu_before
        result["gpu_after"] = gpu_snapshot()
    return result


def ordered_pair(sample_index: int, baseline: TimedCommand, candidate: TimedCommand) -> list[TimedCommand]:
    if sample_index % 2 == 0:
        return [baseline, candidate]
    return [candidate, baseline]


def ordered_commands(sample_index: int, commands: Sequence[TimedCommand]) -> list[TimedCommand]:
    if len(commands) == 2:
        return ordered_pair(sample_index, commands[0], commands[1])
    if not commands:
        return []
    offset = sample_index % len(commands)
    return list(commands[offset:]) + list(commands[:offset])


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_metric_rows(rows: Sequence[dict[str, object]], command_name: str) -> dict[str, dict[str, float]]:
    values_by_metric: dict[str, list[float]] = {}
    for row in rows:
        command = row.get(command_name)
        if not isinstance(command, dict):
            continue
        metrics = command.get("native_metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_metric.setdefault(key, []).append(float(value))
    return {key: summarize(values) for key, values in values_by_metric.items() if values}


def summarize_categorical_metric_rows(rows: Sequence[dict[str, object]], command_name: str) -> dict[str, list[str]]:
    values_by_metric: dict[str, list[str]] = {}
    for row in rows:
        command = row.get(command_name)
        if not isinstance(command, dict):
            continue
        metrics = command.get("native_metrics")
        if not isinstance(metrics, dict):
            continue
        for key in NATIVE_STRATEGY_METRIC_KEYS:
            value = metrics.get(key)
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif isinstance(value, (int, float)):
                text = str(value)
            elif isinstance(value, str):
                text = value
            else:
                continue
            if text not in values_by_metric.setdefault(key, []):
                values_by_metric[key].append(text)
    return {key: values for key, values in values_by_metric.items() if values}


def summarize_metric_ratios(
    rows: Sequence[dict[str, object]],
    denominator_summary: dict[str, dict[str, float]],
    numerator_summary: dict[str, dict[str, float]],
    *,
    denominator_name: str = "baseline",
    numerator_name: str = "candidate",
) -> dict[str, dict[str, float]]:
    ratios_by_metric: dict[str, list[float]] = {}
    shared_metrics = set(denominator_summary).intersection(numerator_summary)
    for row in rows:
        denominator = row.get(denominator_name)
        numerator = row.get(numerator_name)
        if not isinstance(denominator, dict) or not isinstance(numerator, dict):
            continue
        denominator_metrics = denominator.get("native_metrics")
        numerator_metrics = numerator.get("native_metrics")
        if not isinstance(denominator_metrics, dict) or not isinstance(numerator_metrics, dict):
            continue
        for key in shared_metrics:
            denominator_value = denominator_metrics.get(key)
            numerator_value = numerator_metrics.get(key)
            if (
                isinstance(denominator_value, (int, float))
                and not isinstance(denominator_value, bool)
                and isinstance(numerator_value, (int, float))
                and not isinstance(numerator_value, bool)
                and float(denominator_value) != 0.0
            ):
                ratios_by_metric.setdefault(key, []).append(
                    float(numerator_value) / float(denominator_value)
                )
    return {key: summarize(values) for key, values in ratios_by_metric.items() if values}


def summarize_native_route_counter_changes(
    baseline_summary: dict[str, dict[str, float]],
    candidate_summary: dict[str, dict[str, float]],
) -> dict[str, object]:
    changed: dict[str, dict[str, float | None]] = {}
    hot_changed: dict[str, dict[str, float | None]] = {}
    setup_only_changed: dict[str, dict[str, float | None]] = {}
    unchanged: list[str] = []
    missing: list[str] = []
    for key in NATIVE_ROUTE_COUNTER_KEYS:
        baseline_stats = baseline_summary.get(key)
        candidate_stats = candidate_summary.get(key)
        if not isinstance(baseline_stats, dict) or not isinstance(candidate_stats, dict):
            missing.append(key)
            continue
        baseline_mean = baseline_stats.get("mean")
        candidate_mean = candidate_stats.get("mean")
        if not isinstance(baseline_mean, (int, float)) or not isinstance(candidate_mean, (int, float)):
            missing.append(key)
            continue
        delta = float(candidate_mean) - float(baseline_mean)
        if abs(delta) <= 1e-9:
            unchanged.append(key)
            continue
        change = {
            "baseline_mean": float(baseline_mean),
            "candidate_mean": float(candidate_mean),
            "delta": delta,
            "ratio": float(candidate_mean) / float(baseline_mean)
            if float(baseline_mean) != 0.0
            else None,
        }
        changed[key] = change
        if key in SETUP_ONLY_ROUTE_COUNTER_KEYS:
            setup_only_changed[key] = change
        else:
            hot_changed[key] = change
    return {
        "has_route_counter_change": bool(changed),
        "has_hot_route_counter_change": bool(hot_changed),
        "has_setup_only_route_counter_change": bool(setup_only_changed),
        "changed_count": len(changed),
        "hot_changed_count": len(hot_changed),
        "setup_only_changed_count": len(setup_only_changed),
        "tracked_count": len(changed) + len(unchanged),
        "changed": changed,
        "hot_changed": hot_changed,
        "setup_only_changed": setup_only_changed,
        "unchanged": unchanged,
        "missing": missing,
    }


def summarize_native_strategy_value_changes(
    baseline_values: dict[str, list[str]],
    candidate_values: dict[str, list[str]],
) -> dict[str, object]:
    changed: dict[str, dict[str, list[str]]] = {}
    unchanged: list[str] = []
    missing: list[str] = []
    for key in NATIVE_STRATEGY_METRIC_KEYS:
        baseline_observed = baseline_values.get(key)
        candidate_observed = candidate_values.get(key)
        if not baseline_observed or not candidate_observed:
            missing.append(key)
            continue
        if baseline_observed == candidate_observed:
            unchanged.append(key)
            continue
        changed[key] = {
            "baseline_values": list(baseline_observed),
            "candidate_values": list(candidate_observed),
        }
    return {
        "has_strategy_value_change": bool(changed),
        "changed_count": len(changed),
        "tracked_count": len(changed) + len(unchanged),
        "changed": changed,
        "unchanged": unchanged,
        "missing": missing,
    }


def evaluate_native_route_change_gate(
    *,
    required: bool,
    route_changes: dict[str, object],
    strategy_changes: dict[str, object],
    linear_shape_stats: dict[str, object],
    cublaslt_plan_cache: dict[str, object],
) -> dict[str, object]:
    has_route_counter_change = route_changes.get("has_route_counter_change") is True
    has_hot_route_counter_change = route_changes.get("has_hot_route_counter_change") is True
    has_strategy_value_change = strategy_changes.get("has_strategy_value_change") is True
    has_linear_shape_change = bool(linear_shape_stats.get("cublaslt_plan_changed"))
    has_plan_cache_change = bool(cublaslt_plan_cache.get("plan_cache_changed"))
    passed = (
        not required
        or has_hot_route_counter_change
        or has_strategy_value_change
        or has_linear_shape_change
        or has_plan_cache_change
    )
    return {
        "enabled": bool(required),
        "passed": bool(passed),
        "has_route_counter_change": bool(has_route_counter_change),
        "has_hot_route_counter_change": bool(has_hot_route_counter_change),
        "has_strategy_value_change": bool(has_strategy_value_change),
        "has_linear_shape_change": bool(has_linear_shape_change),
        "has_cublaslt_plan_cache_change": bool(has_plan_cache_change),
        "failure_reason": ""
        if passed
        else "candidate-native-metrics-did-not-change-route-strategy-or-plan",
    }


def linear_shape_key(row: dict[str, object]) -> str | None:
    path_name = row.get("path_name")
    m = row.get("m")
    n = row.get("n")
    k = row.get("k")
    op_a_name = row.get("op_a_name")
    op_b_name = row.get("op_b_name")
    if not (
        isinstance(path_name, str)
        and isinstance(m, int)
        and isinstance(n, int)
        and isinstance(k, int)
        and isinstance(op_a_name, str)
        and isinstance(op_b_name, str)
    ):
        return None
    return f"{path_name}:{m}x{n}x{k}:{op_a_name},{op_b_name}"


def cublaslt_plan_cache_key(row: dict[str, object]) -> str | None:
    m = row.get("m")
    n = row.get("n")
    k = row.get("k")
    op_a_name = row.get("op_a_name")
    op_b_name = row.get("op_b_name")
    if not (
        isinstance(m, int)
        and isinstance(n, int)
        and isinstance(k, int)
        and isinstance(op_a_name, str)
        and isinstance(op_b_name, str)
    ):
        return None
    return f"cublaslt:{m}x{n}x{k}:{op_a_name},{op_b_name}"


def linear_shape_numeric(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def summarize_cublaslt_plan_cache(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_command: dict[str, dict[str, list[dict[str, object]]]] = {
        "baseline": {},
        "candidate": {},
    }
    for row in rows:
        for command_name in ("baseline", "candidate"):
            command = row.get(command_name)
            if not isinstance(command, dict):
                continue
            plans = command.get("native_cublaslt_plan_cache")
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if not isinstance(plan, dict):
                    continue
                key = cublaslt_plan_cache_key(plan)
                if key is None:
                    continue
                by_command[command_name].setdefault(key, []).append(plan)

    def summarize_side(items: list[dict[str, object]]) -> dict[str, object]:
        selected = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("selected_heuristic")), int)
            }
        )
        returned = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("returned_heuristics")), int)
            }
        )
        workspace = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("workspace_bytes")), int)
            }
        )
        epilogues = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("epilogue")), int)
            }
        )
        return {
            "samples": len(items),
            "selected_heuristics": selected,
            "returned_heuristics": returned,
            "workspace_bytes": workspace,
            "epilogues": epilogues,
        }

    shared_keys = sorted(set(by_command["baseline"]).intersection(by_command["candidate"]))
    rows_out: list[dict[str, object]] = []
    plan_changes: list[dict[str, object]] = []
    for key in shared_keys:
        baseline = summarize_side(by_command["baseline"][key])
        candidate = summarize_side(by_command["candidate"][key])
        row_out = {"shape": key, "baseline": baseline, "candidate": candidate}
        rows_out.append(row_out)
        changed_fields = {
            field: {
                "baseline": baseline.get(field),
                "candidate": candidate.get(field),
            }
            for field in (
                "selected_heuristics",
                "returned_heuristics",
                "workspace_bytes",
                "epilogues",
            )
            if baseline.get(field) != candidate.get(field)
        }
        if changed_fields:
            plan_changes.append({"shape": key, "changed": changed_fields})
    return {
        "enabled": bool(rows_out),
        "shared_count": len(rows_out),
        "baseline_only": sorted(set(by_command["baseline"]) - set(by_command["candidate"])),
        "candidate_only": sorted(set(by_command["candidate"]) - set(by_command["baseline"])),
        "has_plan_cache_change": bool(plan_changes),
        "plan_cache_changed_count": len(plan_changes),
        "plan_cache_changed": plan_changes,
        "shared": rows_out,
    }


def summarize_linear_shape_stats(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_command: dict[str, dict[str, list[dict[str, object]]]] = {
        "baseline": {},
        "candidate": {},
    }
    for sample in rows:
        if not isinstance(sample, dict):
            continue
        for command_name in by_command:
            command = sample.get(command_name)
            if not isinstance(command, dict):
                continue
            stats = command.get("native_linear_shape_stats")
            if not isinstance(stats, list):
                continue
            for stat in stats:
                if not isinstance(stat, dict):
                    continue
                key = linear_shape_key(stat)
                if key is None:
                    continue
                by_command[command_name].setdefault(key, []).append(stat)

    def summarize_side(items: list[dict[str, object]]) -> dict[str, object]:
        avg_values = [
            value for item in items if (value := linear_shape_numeric(item, "avg_us")) is not None
        ]
        total_values = [
            value for item in items if (value := linear_shape_numeric(item, "total_us")) is not None
        ]
        calls_values = [
            value for item in items if (value := linear_shape_numeric(item, "calls")) is not None
        ]
        selected = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("cublaslt_selected_heuristic")), int)
            }
        )
        returned = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("cublaslt_returned_heuristics")), int)
            }
        )
        workspace = sorted(
            {
                int(value)
                for item in items
                if isinstance((value := item.get("cublaslt_workspace_bytes")), int)
            }
        )
        return {
            "samples": len(items),
            "avg_us": summarize(avg_values) if avg_values else None,
            "total_us": summarize(total_values) if total_values else None,
            "calls": summarize(calls_values) if calls_values else None,
            "cublaslt_selected_heuristics": selected,
            "cublaslt_returned_heuristics": returned,
            "cublaslt_workspace_bytes": workspace,
        }

    shared_keys = sorted(set(by_command["baseline"]).intersection(by_command["candidate"]))
    rows_out: list[dict[str, object]] = []
    for key in shared_keys:
        baseline = summarize_side(by_command["baseline"][key])
        candidate = summarize_side(by_command["candidate"][key])
        baseline_avg = baseline.get("avg_us")
        candidate_avg = candidate.get("avg_us")
        ratio = None
        if isinstance(baseline_avg, dict) and isinstance(candidate_avg, dict):
            base_mean = baseline_avg.get("mean")
            cand_mean = candidate_avg.get("mean")
            if isinstance(base_mean, (int, float)) and isinstance(cand_mean, (int, float)) and base_mean:
                ratio = float(cand_mean) / float(base_mean)
        rows_out.append(
            {
                "shape": key,
                "baseline": baseline,
                "candidate": candidate,
                "candidate_avg_us_over_baseline": ratio,
            }
        )
    def baseline_total_mean(item: dict[str, object]) -> float:
        baseline = item.get("baseline")
        if not isinstance(baseline, dict):
            return 0.0
        total_us = baseline.get("total_us")
        if not isinstance(total_us, dict):
            return 0.0
        value = total_us.get("mean")
        return float(value) if isinstance(value, (int, float)) else 0.0

    rows_out.sort(key=baseline_total_mean, reverse=True)
    cublaslt_plan_changes: list[dict[str, object]] = []
    for row in rows_out:
        shape = row.get("shape")
        baseline = row.get("baseline")
        candidate = row.get("candidate")
        if not isinstance(shape, str) or not shape.startswith("cublaslt:"):
            continue
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            continue
        baseline_selected = baseline.get("cublaslt_selected_heuristics")
        candidate_selected = candidate.get("cublaslt_selected_heuristics")
        baseline_workspace = baseline.get("cublaslt_workspace_bytes")
        candidate_workspace = candidate.get("cublaslt_workspace_bytes")
        if (
            baseline_selected != candidate_selected
            or baseline_workspace != candidate_workspace
        ):
            cublaslt_plan_changes.append(
                {
                    "shape": shape,
                    "baseline_selected_heuristics": (
                        baseline_selected if isinstance(baseline_selected, list) else []
                    ),
                    "candidate_selected_heuristics": (
                        candidate_selected if isinstance(candidate_selected, list) else []
                    ),
                    "baseline_workspace_bytes": (
                        baseline_workspace if isinstance(baseline_workspace, list) else []
                    ),
                    "candidate_workspace_bytes": (
                        candidate_workspace if isinstance(candidate_workspace, list) else []
                    ),
                    "candidate_avg_us_over_baseline": row.get(
                        "candidate_avg_us_over_baseline"
                    ),
                }
            )
    return {
        "enabled": bool(rows_out),
        "shared_count": len(rows_out),
        "baseline_only": sorted(set(by_command["baseline"]) - set(by_command["candidate"])),
        "candidate_only": sorted(set(by_command["candidate"]) - set(by_command["baseline"])),
        "has_cublaslt_plan_change": bool(cublaslt_plan_changes),
        "cublaslt_plan_changed_count": len(cublaslt_plan_changes),
        "cublaslt_plan_changed": cublaslt_plan_changes,
        "shared": rows_out,
    }


def evaluate_metric_ratio_limits(
    payload: dict[str, object],
    limits: Sequence[MetricRatioLimit],
    *,
    ratio_key: str = "candidate_over_baseline_native_metrics",
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    ratios = payload.get(ratio_key)
    ratio_metrics = ratios if isinstance(ratios, dict) else {}
    all_passed = True
    for limit in limits:
        stats = ratio_metrics.get(limit.metric)
        actual_ratio: float | None = None
        missing = True
        if isinstance(stats, dict):
            value = stats.get(limit.stat)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                actual_ratio = float(value)
                missing = False
        passed = not missing and actual_ratio is not None
        if passed and limit.min_ratio is not None:
            passed = actual_ratio >= limit.min_ratio
        if passed and limit.max_ratio is not None:
            passed = actual_ratio <= limit.max_ratio
        if not passed:
            all_passed = False
        result = {
            "metric": limit.metric,
            "stat": limit.stat,
            "actual_ratio": actual_ratio,
            "actual_mean_ratio": actual_ratio,
            "missing": missing,
            "passed": passed,
        }
        if limit.min_ratio is not None:
            result["min_ratio"] = limit.min_ratio
        if limit.max_ratio is not None:
            result["max_ratio"] = limit.max_ratio
        results.append(result)
    return {
        "enabled": bool(limits),
        "passed": all_passed,
        "results": results,
    }


def run_nvidia_smi(args: Sequence[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def parse_csv_rows(output: str, columns: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(columns):
            continue
        rows.append(dict(zip(columns, parts)))
    return rows


def gpu_snapshot() -> dict[str, object]:
    gpu_columns = [
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "display_active",
        "utilization.gpu_pct",
        "memory.used_mib",
        "memory.total_mib",
    ]
    process_columns = [
        "gpu_uuid",
        "pid",
        "process_name",
        "used_memory_mib",
    ]
    gpu_query = run_nvidia_smi(
            [
            "--query-gpu=index,name,uuid,pci.bus_id,display_active,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "gpus": parse_csv_rows(str(gpu_query.get("stdout", "")), gpu_columns)
        if gpu_query.get("returncode") == 0
        else [],
        "compute_processes": parse_csv_rows(str(process_query.get("stdout", "")), process_columns)
        if process_query.get("returncode") == 0
        else [],
        "gpu_query": gpu_query,
        "compute_process_query": process_query,
    }


def _csv_int(value: object, default: int) -> int:
    if not isinstance(value, str):
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _display_is_inactive(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in {"disabled", "no", "off", "false", "0"}


def _first_cuda_device_index(cuda_visible_devices: str) -> str:
    first = cuda_visible_devices.split(",", 1)[0].strip()
    return first


def _selected_gpu(snapshot: dict[str, object], cuda_visible_devices: str) -> dict[str, object] | None:
    selected_index = _first_cuda_device_index(cuda_visible_devices)
    if not selected_index:
        return None
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list):
        return None
    for gpu in gpus:
        if isinstance(gpu, dict) and str(gpu.get("index", "")).strip() == selected_index:
            return gpu
    return None


def _selected_gpu_uuid(snapshot: dict[str, object], cuda_visible_devices: str) -> str:
    gpu = _selected_gpu(snapshot, cuda_visible_devices)
    if not isinstance(gpu, dict):
        return ""
    return str(gpu.get("uuid", "")).strip()


def _compute_process_count(snapshot: dict[str, object], gpu_uuid: str = "") -> int:
    processes = snapshot.get("compute_processes")
    if not isinstance(processes, list):
        return 0
    if not gpu_uuid:
        return len(processes)
    count = 0
    for process in processes:
        if isinstance(process, dict) and str(process.get("gpu_uuid", "")).strip() == gpu_uuid:
            count += 1
    return count


def _compute_processes_for_gpu(snapshot: dict[str, object], gpu_uuid: str) -> list[dict[str, str]]:
    processes = snapshot.get("compute_processes")
    if not isinstance(processes, list) or not gpu_uuid:
        return []
    matched: list[dict[str, str]] = []
    for process in processes:
        if isinstance(process, dict) and str(process.get("gpu_uuid", "")).strip() == gpu_uuid:
            matched.append({str(key): str(value) for key, value in process.items()})
    return matched


def require_idle_selected_gpu(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    *,
    phase: str,
) -> None:
    if not cuda_visible_devices:
        return
    selected_gpu = _selected_gpu(snapshot, cuda_visible_devices)
    if not isinstance(selected_gpu, dict):
        raise SystemExit(
            f"--require-idle-selected-gpu could not identify CUDA device "
            f"{_first_cuda_device_index(cuda_visible_devices)!r} in nvidia-smi output during {phase}"
        )
    selected_uuid = str(selected_gpu.get("uuid", "")).strip()
    processes = _compute_processes_for_gpu(snapshot, selected_uuid)
    if not processes:
        return
    process_lines = "\n".join(
        "  "
        f"pid={process.get('pid', '')} name={process.get('process_name', '')} "
        f"used_memory_mib={process.get('used_memory_mib', '')}"
        for process in processes
    )
    raise SystemExit(
        f"--require-idle-selected-gpu found {len(processes)} compute process(es) on "
        f"CUDA device {_first_cuda_device_index(cuda_visible_devices)} "
        f"({selected_uuid}) during {phase}:\n{process_lines}"
    )


def require_selected_gpu_utilization_at_most(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    max_utilization_pct: float,
    *,
    phase: str,
) -> None:
    result = selected_gpu_utilization_guard_result(
        snapshot,
        cuda_visible_devices,
        max_utilization_pct,
        phase=phase,
    )
    if result is None:
        return
    raise SystemExit(result)


def selected_gpu_utilization_guard_result(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    max_utilization_pct: float,
    *,
    phase: str,
) -> str | None:
    if max_utilization_pct < 0.0 or not cuda_visible_devices:
        return None
    selected_gpu = _selected_gpu(snapshot, cuda_visible_devices)
    if not isinstance(selected_gpu, dict):
        return (
            f"--max-selected-gpu-utilization-pct could not identify CUDA device "
            f"{_first_cuda_device_index(cuda_visible_devices)!r} in nvidia-smi output during {phase}"
        )
    utilization_pct = float(_csv_int(selected_gpu.get("utilization.gpu_pct"), 100))
    if utilization_pct <= max_utilization_pct:
        return None
    return (
        f"--max-selected-gpu-utilization-pct={max_utilization_pct:g} rejected CUDA device "
        f"{_first_cuda_device_index(cuda_visible_devices)} during {phase}: "
        f"nvidia-smi utilization is {utilization_pct:g}%"
    )


def enforce_selected_gpu_guards(
    snapshot: dict[str, object],
    cuda_visible_devices: str,
    *,
    require_idle: bool,
    max_utilization_pct: float,
    phase: str,
    utilization_retries: int = 1,
    utilization_retry_interval_seconds: float = 0.0,
    allow_stale_utilization_without_compute_processes: bool = False,
    snapshot_supplier: Any | None = None,
) -> dict[str, object]:
    if require_idle:
        require_idle_selected_gpu(snapshot, cuda_visible_devices, phase=phase)
    result = selected_gpu_utilization_guard_result(
        snapshot,
        cuda_visible_devices,
        max_utilization_pct,
        phase=phase,
    )
    if result is None:
        return snapshot
    if (
        max_utilization_pct >= 0.0
        and snapshot_supplier is not None
        and _compute_process_count(snapshot, _selected_gpu_uuid(snapshot, cuda_visible_devices)) == 0
    ):
        attempts = max(1, int(utilization_retries))
        interval = max(0.0, float(utilization_retry_interval_seconds))
        for _attempt in range(1, attempts):
            if interval > 0.0:
                time.sleep(interval)
            retry_snapshot = snapshot_supplier()
            if require_idle:
                require_idle_selected_gpu(retry_snapshot, cuda_visible_devices, phase=phase)
            result = selected_gpu_utilization_guard_result(
                retry_snapshot,
                cuda_visible_devices,
                max_utilization_pct,
                phase=phase,
            )
            if result is None:
                return retry_snapshot
            if _compute_process_count(
                retry_snapshot,
                _selected_gpu_uuid(retry_snapshot, cuda_visible_devices),
            ) > 0:
                break
        selected_uuid = _selected_gpu_uuid(snapshot, cuda_visible_devices)
        if (
            allow_stale_utilization_without_compute_processes
            and _compute_process_count(snapshot, selected_uuid) == 0
        ):
            return snapshot
    raise SystemExit(result)


def gpu_benchmark_lock_path(cuda_visible_devices: str) -> Path | None:
    selected = _first_cuda_device_index(cuda_visible_devices)
    if not selected:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", selected)
    return Path("/tmp") / f"nfn_paired_kernel_speed_gpu_{safe}.lock"


class GpuBenchmarkLock:
    def __init__(self, path: Path | None, *, enabled: bool, timeout_seconds: float) -> None:
        self.path = path
        self.enabled = bool(enabled and path is not None)
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._fd: int | None = None
        self.acquired = False

    def __enter__(self) -> "GpuBenchmarkLock":
        if not self.enabled or self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = True
                os.ftruncate(self._fd, 0)
                os.write(
                    self._fd,
                    f"pid={os.getpid()} started_ns={time.time_ns()}\n".encode("utf-8"),
                )
                return self
            except BlockingIOError as exc:
                if self.timeout_seconds <= 0.0 or time.monotonic() >= deadline:
                    owner = ""
                    try:
                        os.lseek(self._fd, 0, os.SEEK_SET)
                        owner = os.read(self._fd, 512).decode("utf-8", errors="replace").strip()
                    except OSError:
                        owner = ""
                    detail = f" Current owner: {owner}" if owner else ""
                    raise SystemExit(
                        f"paired benchmark GPU lock is already held: {self.path}.{detail} "
                        "Rerun after the other benchmark exits, pass "
                        "--gpu-benchmark-lock-timeout-seconds N to wait, or pass "
                        "--no-gpu-benchmark-lock only for intentionally unmanaged runs."
                    ) from exc
                time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._fd is None:
            return
        try:
            if self.acquired:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            self.acquired = False


def snapshot_and_enforce_selected_gpu_guards(
    cuda_visible_devices: str,
    *,
    require_idle: bool,
    max_utilization_pct: float,
    utilization_retries: int,
    utilization_retry_interval_seconds: float,
    allow_stale_utilization_without_compute_processes: bool,
    phase: str,
) -> dict[str, object]:
    snapshot = gpu_snapshot()
    return enforce_selected_gpu_guards(
        snapshot,
        cuda_visible_devices,
        require_idle=require_idle,
        max_utilization_pct=max_utilization_pct,
        utilization_retries=utilization_retries,
        utilization_retry_interval_seconds=utilization_retry_interval_seconds,
        allow_stale_utilization_without_compute_processes=(
            allow_stale_utilization_without_compute_processes
        ),
        snapshot_supplier=gpu_snapshot,
        phase=phase,
    )


def summarize_gpu_sample_load(
    rows: Sequence[dict[str, object]],
    cuda_visible_devices: str,
) -> dict[str, object]:
    before_util: list[float] = []
    after_util: list[float] = []
    before_mem: list[float] = []
    after_mem: list[float] = []
    total_processes_before: list[float] = []
    total_processes_after: list[float] = []
    selected_processes_before: list[float] = []
    selected_processes_after: list[float] = []
    selected_index = _first_cuda_device_index(cuda_visible_devices)
    selected_uuid = ""

    for row in rows:
        before = row.get("gpu_before")
        after = row.get("gpu_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_gpu = _selected_gpu(before, cuda_visible_devices)
        after_gpu = _selected_gpu(after, cuda_visible_devices)
        if isinstance(before_gpu, dict):
            selected_uuid = selected_uuid or str(before_gpu.get("uuid", "")).strip()
            before_util.append(float(_csv_int(before_gpu.get("utilization.gpu_pct"), 0)))
            before_mem.append(float(_csv_int(before_gpu.get("memory.used_mib"), 0)))
        if isinstance(after_gpu, dict):
            selected_uuid = selected_uuid or str(after_gpu.get("uuid", "")).strip()
            after_util.append(float(_csv_int(after_gpu.get("utilization.gpu_pct"), 0)))
            after_mem.append(float(_csv_int(after_gpu.get("memory.used_mib"), 0)))
        before_uuid = selected_uuid or _selected_gpu_uuid(before, cuda_visible_devices)
        after_uuid = selected_uuid or _selected_gpu_uuid(after, cuda_visible_devices)
        total_processes_before.append(float(_compute_process_count(before)))
        total_processes_after.append(float(_compute_process_count(after)))
        if before_uuid:
            selected_processes_before.append(float(_compute_process_count(before, before_uuid)))
        if after_uuid:
            selected_processes_after.append(float(_compute_process_count(after, after_uuid)))

    summary: dict[str, object] = {
        "selected_cuda_visible_devices": cuda_visible_devices,
        "selected_gpu_index": selected_index,
        "selected_gpu_uuid": selected_uuid,
        "sample_count": len(rows),
    }
    metric_rows = (
        ("selected_gpu_utilization_before_pct", before_util),
        ("selected_gpu_utilization_after_pct", after_util),
        ("selected_gpu_memory_used_before_mib", before_mem),
        ("selected_gpu_memory_used_after_mib", after_mem),
        ("compute_process_count_before", total_processes_before),
        ("compute_process_count_after", total_processes_after),
        ("selected_gpu_compute_process_count_before", selected_processes_before),
        ("selected_gpu_compute_process_count_after", selected_processes_after),
    )
    for name, values in metric_rows:
        if values:
            summary[name] = summarize(values)
    return summary


def metric_summary_fragment(name: str, stats: object) -> str:
    if not isinstance(stats, dict):
        return ""
    required = ("mean", "median", "min", "max")
    if not all(isinstance(stats.get(item), (int, float)) for item in required):
        return ""
    return (
        f"{name}=mean={float(stats['mean']):.6f} "
        f"median={float(stats['median']):.6f} min={float(stats['min']):.6f} "
        f"max={float(stats['max']):.6f}"
    )


def metric_mean_fragment(name: str, stats: object) -> str:
    if not isinstance(stats, dict):
        return ""
    value = stats.get("mean")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    return f"{name}={float(value):.6f}"


def print_metric_summary_line(
    prefix: str,
    key: str,
    stats: object,
    metrics: dict[str, object] | None = None,
) -> None:
    main = metric_summary_fragment("", stats)
    if not main:
        return
    line = f"      {prefix}{key}: {main.removeprefix('=')}"
    if metrics is not None and key.endswith(".total_ms"):
        stem = key[: -len(".total_ms")]
        count = metric_mean_fragment("count_mean", metrics.get(stem + ".count"))
        avg = metric_mean_fragment("avg_ms_mean", metrics.get(stem + ".avg_ms"))
        fragments = [item for item in (count, avg) if item]
        if fragments:
            line += " (" + "; ".join(fragments) + ")"
    print(line)


def print_native_hot_summary(payload: dict[str, object]) -> None:
    sections = (
        ("baseline", "baseline_native_metrics", ""),
        ("candidate", "candidate_native_metrics", ""),
        ("candidate_over_baseline", "candidate_over_baseline_native_metrics", "ratio "),
        ("reference", "reference_native_metrics", ""),
        ("reference_over_baseline", "reference_over_baseline_native_metrics", "ratio "),
        ("candidate_over_reference", "candidate_over_reference_native_metrics", "ratio "),
    )
    printable_sections: list[tuple[str, dict[str, object], str]] = []
    for label, payload_key, metric_prefix in sections:
        metrics = payload.get(payload_key)
        if not isinstance(metrics, dict):
            continue
        if any(isinstance(metrics.get(key), dict) for key in NATIVE_HOT_SUMMARY_METRIC_KEYS):
            printable_sections.append((label, metrics, metric_prefix))
    if not printable_sections:
        return
    print("  native_hot_summary:")
    for label, metrics, metric_prefix in printable_sections:
        print(f"    {label}:")
        for key in NATIVE_HOT_SUMMARY_METRIC_KEYS:
            print_metric_summary_line(metric_prefix, key, metrics.get(key), metrics)


def _first_metric_value(
    values: dict[str, object],
    key: str,
) -> str | None:
    observed = values.get(key)
    if not isinstance(observed, list) or not observed:
        return None
    return str(observed[0])


def _observed_bool(values: dict[str, object], key: str) -> bool | None:
    value = _first_metric_value(values, key)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _metric_mean(metrics: dict[str, object], key: str) -> float | None:
    stats = metrics.get(key)
    if not isinstance(stats, dict):
        return None
    value = stats.get("mean")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def summarize_lm_head_true_fused_target(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload.get("candidate_native_metrics")
    values = payload.get("candidate_native_metric_values")
    if not isinstance(metrics, dict) or not isinstance(values, dict):
        return {"required": False, "status": "unobserved", "reason": "no candidate native metrics"}

    path_class = _first_metric_value(values, "lm_head_classifier_backward_path_class") or ""
    abi_path_class = (
        _first_metric_value(values, "lm_head_cooperative_backward_fused_kernel_abi_path_class")
        or ""
    )
    true_fused_capability = _observed_bool(
        values,
        "lm_head_cooperative_backward_fused_kernel_capability_available",
    )
    symbol_available = _observed_bool(
        values,
        "lm_head_cooperative_backward_fused_kernel_symbol_available",
    )
    graph_wrapper_active = path_class.startswith("diagnostic-cuda-graph-wrapper") or (
        not path_class and abi_path_class.startswith("diagnostic-cuda-graph-wrapper")
    )
    strict_true_fused = (
        true_fused_capability is True
        and path_class == "strict-true-fused-tile-kernel"
    )
    required = graph_wrapper_active or true_fused_capability is False
    status = (
        "strict-true-fused-tile-kernel"
        if strict_true_fused
        else path_class or abi_path_class or "unknown"
    )
    candidate_reference_gates = payload.get("candidate_reference_metric_ratio_gates")
    candidate_reference_gate_failed = (
        isinstance(candidate_reference_gates, dict)
        and candidate_reference_gates.get("enabled") is True
        and candidate_reference_gates.get("passed") is False
    )
    reason = (
        "candidate already reports strict true-fused Tile LM-head backward"
        if strict_true_fused
        else (
            "candidate is still the diagnostic CUDA Graph wrapper, so parity work must "
            "replace the wrapper body with a bounded true-fused Tile kernel"
            if graph_wrapper_active
            else (
                "candidate strict fused symbol is present but capability is false"
                if true_fused_capability is False
                else "candidate true-fused LM-head capability was not observed"
            )
        )
    )
    return {
        "required": required,
        "status": status,
        "path_class": path_class,
        "abi_path_class": abi_path_class,
        "true_fused_capability": true_fused_capability,
        "symbol_available": symbol_available,
        "graph_wrapper_active": graph_wrapper_active,
        "graph_replay_mean": _metric_mean(metrics, "lm_head_fused_graph_replay_count"),
        "graph_replay_success_mean": _metric_mean(
            metrics,
            "lm_head_fused_graph_replay_success_count",
        ),
        "graph_fallback_mean": _metric_mean(metrics, "lm_head_fused_graph_fallback_count"),
        "graph_body_nodes_per_replay_mean": _metric_mean(
            metrics,
            "lm_head_fused_graph_body_node_count_per_replay",
        ),
        "graph_body_ce_nodes_per_replay_mean": _metric_mean(
            metrics,
            "lm_head_fused_graph_body_ce_node_count_per_replay",
        ),
        "graph_body_dhidden_nodes_per_replay_mean": _metric_mean(
            metrics,
            "lm_head_fused_graph_body_dhidden_node_count_per_replay",
        ),
        "graph_body_dweight_nodes_per_replay_mean": _metric_mean(
            metrics,
            "lm_head_fused_graph_body_dweight_node_count_per_replay",
        ),
        "candidate_reference_gate_failed": candidate_reference_gate_failed,
        "next_symbol": "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16",
        "next_capability_symbol": (
            "nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused"
        ),
        "next_required_path_class": "strict-true-fused-tile-kernel",
        "reason": reason,
    }


def evaluate_lm_head_true_fused_gate(
    *,
    required: bool,
    target: dict[str, object],
) -> dict[str, object]:
    status = str(target.get("status", "unknown"))
    needs_work = target.get("required") is True
    passed = not required or not needs_work
    failure_reason = ""
    if required and needs_work:
        failure_reason = (
            "candidate native LM-head backward is not strict true-fused Tile "
            f"(status={status}, path_class={target.get('path_class', '')}, "
            f"true_fused_capability={target.get('true_fused_capability')})"
        )
    elif required and status == "unobserved":
        passed = False
        failure_reason = "candidate native LM-head true-fused metrics were not observed"
    return {
        "enabled": required,
        "passed": passed,
        "status": status,
        "failure_reason": failure_reason,
    }


def print_lm_head_true_fused_target(payload: dict[str, object]) -> None:
    target = payload.get("native_lm_head_true_fused_target")
    if not isinstance(target, dict):
        return
    if target.get("required") is not True and target.get("status") == "unobserved":
        return
    print("  native_lm_head_true_fused_target:")
    print(
        "    "
        f"required={str(target.get('required', False)).lower()} "
        f"status={target.get('status', '')} "
        f"path_class={target.get('path_class', '')} "
        f"abi_path_class={target.get('abi_path_class', '')}"
    )
    print(
        "    "
        f"symbol_available={target.get('symbol_available')} "
        f"true_fused_capability={target.get('true_fused_capability')} "
        f"graph_wrapper_active={str(target.get('graph_wrapper_active', False)).lower()}"
    )
    graph_fragments: list[str] = []
    for key in (
        "graph_replay_mean",
        "graph_replay_success_mean",
        "graph_fallback_mean",
        "graph_body_nodes_per_replay_mean",
        "graph_body_ce_nodes_per_replay_mean",
        "graph_body_dhidden_nodes_per_replay_mean",
        "graph_body_dweight_nodes_per_replay_mean",
    ):
        value = target.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            graph_fragments.append(f"{key}={float(value):.6f}")
    if graph_fragments:
        print("    " + " ".join(graph_fragments))
    print(
        "    next: "
        f"{target.get('next_symbol', '')} with "
        f"{target.get('next_capability_symbol', '')} returning true and "
        f"path_class={target.get('next_required_path_class', '')}"
    )
    print(f"    reason: {target.get('reason', '')}")


def print_lm_head_true_fused_gate(payload: dict[str, object]) -> None:
    gate = payload.get("native_lm_head_true_fused_gate")
    if not isinstance(gate, dict) or gate.get("enabled") is not True:
        return
    print(
        "  native_lm_head_true_fused_gate: "
        f"passed={str(gate.get('passed', False)).lower()} "
        f"status={gate.get('status', '')}"
    )
    failure_reason = gate.get("failure_reason")
    if failure_reason:
        print(f"    failure_reason: {failure_reason}")


def resolve_cuda_visible_devices(
    requested: str,
    snapshot: dict[str, object],
) -> dict[str, object]:
    requested = requested.strip()
    if requested not in {"auto", "dedicated", "dedicated-auto"}:
        return {
            "requested": requested,
            "resolved": requested,
            "mode": "explicit" if requested else "unchanged",
            "reason": "explicit CUDA_VISIBLE_DEVICES value" if requested else "explicit empty value",
        }

    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return {
            "requested": requested,
            "resolved": "",
            "mode": "auto-unresolved",
            "reason": "nvidia-smi did not return GPU rows",
        }

    processes = snapshot.get("compute_processes")
    busy_uuids: set[str] = set()
    if isinstance(processes, list):
        for process in processes:
            if not isinstance(process, dict):
                continue
            uuid = process.get("gpu_uuid")
            if isinstance(uuid, str) and uuid.strip():
                busy_uuids.add(uuid.strip())

    candidates: list[tuple[int, int, int, dict[str, object]]] = []
    fallback: list[tuple[int, int, int, dict[str, object]]] = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        index = _csv_int(gpu.get("index"), -1)
        if index < 0:
            continue
        util = _csv_int(gpu.get("utilization.gpu_pct"), 100)
        mem_used = _csv_int(gpu.get("memory.used_mib"), 1_000_000_000)
        uuid = str(gpu.get("uuid", "")).strip()
        row = (util, mem_used, index, gpu)
        fallback.append(row)
        if _display_is_inactive(gpu.get("display_active")) and uuid not in busy_uuids:
            candidates.append(row)

    if requested == "dedicated" and not candidates:
        return {
            "requested": requested,
            "resolved": "",
            "mode": "dedicated-unresolved",
            "reason": "no idle display-disabled GPU found for dedicated benchmark mode",
        }

    selected_pool = candidates if candidates else fallback
    if not selected_pool:
        return {
            "requested": requested,
            "resolved": "",
            "mode": "auto-unresolved",
            "reason": "no parseable nvidia-smi GPU index",
        }

    selected = sorted(selected_pool, key=lambda row: (row[0], row[1], row[2]))[0]
    mode = "auto-dedicated" if candidates else "auto-fallback"
    reason = (
        "selected lowest-utilization display-disabled GPU with no compute processes"
        if candidates
        else "no idle display-disabled GPU found; selected lowest-utilization GPU"
    )
    return {
        "requested": requested,
        "resolved": str(selected[2]),
        "mode": mode,
        "reason": reason,
        "selected_gpu": selected[3],
    }


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    baseline_env = parse_env_overrides(args.baseline_env, option_name="--baseline-env")
    candidate_env = parse_env_overrides(args.candidate_env, option_name="--candidate-env")
    reference_env = parse_env_overrides(args.reference_env, option_name="--reference-env")
    metadata = parse_env_overrides(args.metadata, option_name="--metadata")
    metric_ratio_limits = [
        *parse_metric_ratio_limits(
            args.max_candidate_ratio,
            option_name="--max-candidate-ratio",
            bound="max",
        ),
        *parse_metric_ratio_limits(
            args.min_candidate_ratio,
            option_name="--min-candidate-ratio",
            bound="min",
        ),
    ]
    candidate_reference_metric_ratio_limits = [
        *parse_metric_ratio_limits(
            args.max_candidate_reference_ratio,
            option_name="--max-candidate-reference-ratio",
            bound="max",
        ),
        *parse_metric_ratio_limits(
            args.min_candidate_reference_ratio,
            option_name="--min-candidate-reference-ratio",
            bound="min",
        ),
    ]
    baseline = TimedCommand("baseline", shlex.split(args.baseline), baseline_env)
    candidate = TimedCommand("candidate", shlex.split(args.candidate), candidate_env)
    reference_command_text = str(args.reference or "").strip()
    if candidate_reference_metric_ratio_limits and not reference_command_text:
        raise SystemExit(
            "--max-candidate-reference-ratio/--min-candidate-reference-ratio require --reference"
        )
    reference = (
        TimedCommand("reference", shlex.split(reference_command_text), reference_env)
        if reference_command_text
        else None
    )
    measured_commands = [baseline, candidate] + ([reference] if reference is not None else [])
    samples = max(1, args.samples)
    warmup = max(0, args.warmup)
    timeout_seconds = float(args.command_timeout_seconds or 0.0)
    command_timeout = timeout_seconds if timeout_seconds > 0.0 else None
    profile_json_dir = (
        Path(str(args.append_native_profile_json_dir)).expanduser()
        if str(args.append_native_profile_json_dir or "").strip()
        else None
    )
    gpu_before = gpu_snapshot()
    cuda_device_selection = resolve_cuda_visible_devices(str(args.cuda_visible_devices or ""), gpu_before)
    cuda_visible_devices = str(cuda_device_selection.get("resolved", "") or "").strip()
    if str(cuda_device_selection.get("mode", "")) == "dedicated-unresolved":
        raise SystemExit(str(cuda_device_selection.get("reason", "dedicated GPU selection failed")))
    cuda_device_max_connections = str(args.cuda_device_max_connections or "").strip()
    max_selected_gpu_utilization_pct = float(args.max_selected_gpu_utilization_pct)
    selected_gpu_utilization_retries = max(1, int(args.selected_gpu_utilization_retries))
    selected_gpu_utilization_retry_interval_seconds = max(
        0.0,
        float(args.selected_gpu_utilization_retry_interval_seconds),
    )
    allow_stale_utilization_without_compute_processes = bool(
        args.allow_stale_selected_gpu_utilization_without_compute_processes
    )
    gpu_lock_path = gpu_benchmark_lock_path(cuda_visible_devices)
    gpu_lock_enabled = bool(cuda_visible_devices and not args.no_gpu_benchmark_lock)
    gpu_lock_timeout_seconds = max(0.0, float(args.gpu_benchmark_lock_timeout_seconds or 0.0))
    run_env = None
    if cuda_visible_devices or cuda_device_max_connections:
        run_env = dict(os.environ)
        if cuda_visible_devices:
            run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if cuda_device_max_connections:
            run_env["CUDA_DEVICE_MAX_CONNECTIONS"] = cuda_device_max_connections

    if bool(args.dry_run_plan):
        order_plan = [
            {
                "sample": sample_index + 1,
                "order": [
                    command.name
                    for command in ordered_commands(sample_index, measured_commands)
                ],
            }
            for sample_index in range(samples)
        ]
        return {
            "measurement": "paired_interleaved_commands",
            "dry_run_plan": True,
            "samples": samples,
            "warmup": warmup,
            "cuda_visible_devices_requested": cuda_device_selection.get("requested", ""),
            "cuda_visible_devices": cuda_visible_devices,
            "cuda_device_selection": cuda_device_selection,
            "cuda_device_max_connections": cuda_device_max_connections,
            "require_idle_selected_gpu": bool(args.require_idle_selected_gpu),
            "max_selected_gpu_utilization_pct": max_selected_gpu_utilization_pct,
            "selected_gpu_utilization_retries": selected_gpu_utilization_retries,
            "selected_gpu_utilization_retry_interval_seconds": (
                selected_gpu_utilization_retry_interval_seconds
            ),
            "allow_stale_selected_gpu_utilization_without_compute_processes": (
                allow_stale_utilization_without_compute_processes
            ),
            "gpu_benchmark_lock_enabled": gpu_lock_enabled,
            "gpu_benchmark_lock_path": str(gpu_lock_path) if gpu_lock_path is not None else "",
            "gpu_benchmark_lock_timeout_seconds": gpu_lock_timeout_seconds,
            "gpu_benchmark_lock_acquired": False,
            "command_timeout_seconds": timeout_seconds,
            "metadata": metadata,
            "gpu_before": gpu_before,
            "baseline_command": baseline.argv,
            "candidate_command": candidate.argv,
            "reference_command": reference.argv if reference is not None else [],
            "baseline_env": baseline.env_overrides,
            "candidate_env": candidate.env_overrides,
            "reference_env": reference.env_overrides if reference is not None else {},
            "append_native_profile_json_dir": str(profile_json_dir) if profile_json_dir is not None else "",
            "native_stage_timing": bool(args.native_stage_timing),
            "metric_ratio_gates": {
                "enabled": bool(metric_ratio_limits),
                "passed": True,
                "results": [
                    ({
                        "metric": limit.metric,
                        "stat": limit.stat,
                        "actual_mean_ratio": None,
                        "missing": True,
                        "passed": True,
                    }
                    | ({"min_ratio": limit.min_ratio} if limit.min_ratio is not None else {})
                    | ({"max_ratio": limit.max_ratio} if limit.max_ratio is not None else {}))
                    for limit in metric_ratio_limits
                ],
            },
            "candidate_reference_metric_ratio_gates": {
                "enabled": bool(candidate_reference_metric_ratio_limits),
                "passed": True,
                "results": [
                    ({
                        "metric": limit.metric,
                        "stat": limit.stat,
                        "actual_mean_ratio": None,
                        "missing": True,
                        "passed": True,
                    }
                    | ({"min_ratio": limit.min_ratio} if limit.min_ratio is not None else {})
                    | ({"max_ratio": limit.max_ratio} if limit.max_ratio is not None else {}))
                    for limit in candidate_reference_metric_ratio_limits
                ],
            },
            "native_lm_head_true_fused_gate": {
                "enabled": bool(args.require_native_lm_head_true_fused),
                "passed": True,
                "status": "dry-run",
                "failure_reason": "",
            },
            "sample_order_plan": order_plan,
            "run_env_overrides": {
                key: value
                for key, value in (
                    ("CUDA_VISIBLE_DEVICES", cuda_visible_devices),
                    ("CUDA_DEVICE_MAX_CONNECTIONS", cuda_device_max_connections),
                )
                if value
            },
        }

    gpu_lock_acquired = False
    with GpuBenchmarkLock(
        gpu_lock_path,
        enabled=gpu_lock_enabled,
        timeout_seconds=gpu_lock_timeout_seconds,
    ) as gpu_lock:
        gpu_lock_acquired = bool(gpu_lock.acquired)
        enforce_selected_gpu_guards(
            gpu_before,
            cuda_visible_devices,
            require_idle=bool(args.require_idle_selected_gpu),
            max_utilization_pct=max_selected_gpu_utilization_pct,
            utilization_retries=selected_gpu_utilization_retries,
            utilization_retry_interval_seconds=selected_gpu_utilization_retry_interval_seconds,
            allow_stale_utilization_without_compute_processes=(
                allow_stale_utilization_without_compute_processes
            ),
            snapshot_supplier=gpu_snapshot,
            phase="initial snapshot",
        )

        for warmup_index in range(warmup):
            enforce_selected_gpu_guards(
                gpu_snapshot(),
                cuda_visible_devices,
                require_idle=bool(args.require_idle_selected_gpu),
                max_utilization_pct=max_selected_gpu_utilization_pct,
                utilization_retries=selected_gpu_utilization_retries,
                utilization_retry_interval_seconds=selected_gpu_utilization_retry_interval_seconds,
                allow_stale_utilization_without_compute_processes=(
                    allow_stale_utilization_without_compute_processes
                ),
                snapshot_supplier=gpu_snapshot,
                phase=f"warmup pair {warmup_index + 1}",
            )
            for command in ordered_commands(warmup_index, measured_commands):
                command_gpu_before = snapshot_and_enforce_selected_gpu_guards(
                    cuda_visible_devices,
                    require_idle=bool(args.require_idle_selected_gpu),
                    max_utilization_pct=max_selected_gpu_utilization_pct,
                    utilization_retries=selected_gpu_utilization_retries,
                    utilization_retry_interval_seconds=(
                        selected_gpu_utilization_retry_interval_seconds
                    ),
                    allow_stale_utilization_without_compute_processes=(
                        allow_stale_utilization_without_compute_processes
                    ),
                    phase=f"warmup pair {warmup_index + 1} {command.name}",
                )
                run_once(
                    command,
                    continue_on_error=args.continue_on_error,
                    env=run_env,
                    timeout_seconds=command_timeout,
                    profile_json_dir=profile_json_dir,
                    native_stage_timing=bool(args.native_stage_timing),
                    gpu_before=command_gpu_before,
                )

        sample_rows: list[dict[str, object]] = []
        baseline_seconds: list[float] = []
        candidate_seconds: list[float] = []
        reference_seconds: list[float] = []
        ratios: list[float] = []
        reference_over_baseline_ratios: list[float] = []
        candidate_over_reference_ratios: list[float] = []
        for sample_index in range(samples):
            sample_gpu_before = gpu_snapshot()
            sample_gpu_before = enforce_selected_gpu_guards(
                sample_gpu_before,
                cuda_visible_devices,
                require_idle=bool(args.require_idle_selected_gpu),
                max_utilization_pct=max_selected_gpu_utilization_pct,
                utilization_retries=selected_gpu_utilization_retries,
                utilization_retry_interval_seconds=selected_gpu_utilization_retry_interval_seconds,
                allow_stale_utilization_without_compute_processes=(
                    allow_stale_utilization_without_compute_processes
                ),
                snapshot_supplier=gpu_snapshot,
                phase=f"measured sample {sample_index + 1}",
            )
            order_names: list[str] = []
            row: dict[str, object] = {
                "sample": sample_index + 1,
                "order": order_names,
                "gpu_before": sample_gpu_before,
            }
            by_name: dict[str, dict[str, object]] = {}
            for command in ordered_commands(sample_index, measured_commands):
                order_names.append(command.name)
                command_gpu_before = snapshot_and_enforce_selected_gpu_guards(
                    cuda_visible_devices,
                    require_idle=bool(args.require_idle_selected_gpu),
                    max_utilization_pct=max_selected_gpu_utilization_pct,
                    utilization_retries=selected_gpu_utilization_retries,
                    utilization_retry_interval_seconds=(
                        selected_gpu_utilization_retry_interval_seconds
                    ),
                    allow_stale_utilization_without_compute_processes=(
                        allow_stale_utilization_without_compute_processes
                    ),
                    phase=f"measured sample {sample_index + 1} {command.name}",
                )
                result = run_once(
                    command,
                    continue_on_error=args.continue_on_error,
                    env=run_env,
                    timeout_seconds=command_timeout,
                    profile_json_dir=profile_json_dir,
                    native_stage_timing=bool(args.native_stage_timing),
                    gpu_before=command_gpu_before,
                )
                by_name[command.name] = result
            baseline_time = float(by_name["baseline"]["seconds"])
            candidate_time = float(by_name["candidate"]["seconds"])
            baseline_seconds.append(baseline_time)
            candidate_seconds.append(candidate_time)
            ratios.append(candidate_time / baseline_time if baseline_time else float("inf"))
            row["baseline"] = by_name["baseline"]
            row["candidate"] = by_name["candidate"]
            row["candidate_over_baseline"] = ratios[-1]
            if reference is not None:
                reference_time = float(by_name["reference"]["seconds"])
                reference_seconds.append(reference_time)
                reference_over_baseline = (
                    reference_time / baseline_time if baseline_time else float("inf")
                )
                candidate_over_reference = (
                    candidate_time / reference_time if reference_time else float("inf")
                )
                reference_over_baseline_ratios.append(reference_over_baseline)
                candidate_over_reference_ratios.append(candidate_over_reference)
                row["reference"] = by_name["reference"]
                row["reference_over_baseline"] = reference_over_baseline
                row["candidate_over_reference"] = candidate_over_reference
            row["gpu_after"] = gpu_snapshot()
            sample_rows.append(row)
        gpu_after = gpu_snapshot()
        baseline_native_metrics = summarize_metric_rows(sample_rows, "baseline")
        candidate_native_metrics = summarize_metric_rows(sample_rows, "candidate")
        reference_native_metrics = (
            summarize_metric_rows(sample_rows, "reference") if reference is not None else {}
        )
        baseline_native_metric_values = summarize_categorical_metric_rows(sample_rows, "baseline")
        candidate_native_metric_values = summarize_categorical_metric_rows(sample_rows, "candidate")
        reference_native_metric_values = (
            summarize_categorical_metric_rows(sample_rows, "reference")
            if reference is not None
            else {}
        )
        native_route_counter_changes = summarize_native_route_counter_changes(
            baseline_native_metrics,
            candidate_native_metrics,
        )
        native_strategy_value_changes = summarize_native_strategy_value_changes(
            baseline_native_metric_values,
            candidate_native_metric_values,
        )
        native_linear_shape_stats = summarize_linear_shape_stats(sample_rows)
        native_cublaslt_plan_cache = summarize_cublaslt_plan_cache(sample_rows)
        gpu_sample_summary = summarize_gpu_sample_load(sample_rows, cuda_visible_devices)

    payload = {
        "measurement": "paired_interleaved_commands",
        "samples": samples,
        "warmup": warmup,
        "cuda_visible_devices_requested": cuda_device_selection.get("requested", ""),
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_device_selection": cuda_device_selection,
        "cuda_device_max_connections": cuda_device_max_connections,
        "require_idle_selected_gpu": bool(args.require_idle_selected_gpu),
        "max_selected_gpu_utilization_pct": max_selected_gpu_utilization_pct,
        "selected_gpu_utilization_retries": selected_gpu_utilization_retries,
        "selected_gpu_utilization_retry_interval_seconds": (
            selected_gpu_utilization_retry_interval_seconds
        ),
        "allow_stale_selected_gpu_utilization_without_compute_processes": (
            allow_stale_utilization_without_compute_processes
        ),
        "gpu_benchmark_lock_enabled": gpu_lock_enabled,
        "gpu_benchmark_lock_path": str(gpu_lock_path) if gpu_lock_path is not None else "",
        "gpu_benchmark_lock_timeout_seconds": gpu_lock_timeout_seconds,
        "gpu_benchmark_lock_acquired": gpu_lock_acquired,
        "command_timeout_seconds": timeout_seconds,
        "metadata": metadata,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_sample_summary": gpu_sample_summary,
        "baseline_command": baseline.argv,
        "candidate_command": candidate.argv,
        "reference_command": reference.argv if reference is not None else [],
        "baseline_env": baseline.env_overrides,
        "candidate_env": candidate.env_overrides,
        "reference_env": reference.env_overrides if reference is not None else {},
        "append_native_profile_json_dir": str(profile_json_dir) if profile_json_dir is not None else "",
        "native_stage_timing": bool(args.native_stage_timing),
        "baseline_seconds": summarize(baseline_seconds),
        "candidate_seconds": summarize(candidate_seconds),
        "candidate_over_baseline": summarize(ratios),
        "baseline_native_metrics": baseline_native_metrics,
        "candidate_native_metrics": candidate_native_metrics,
        "reference_native_metrics": reference_native_metrics,
        "baseline_native_metric_values": baseline_native_metric_values,
        "candidate_native_metric_values": candidate_native_metric_values,
        "reference_native_metric_values": reference_native_metric_values,
        "native_route_counter_changes": native_route_counter_changes,
        "native_strategy_value_changes": native_strategy_value_changes,
        "native_linear_shape_stats": native_linear_shape_stats,
        "native_cublaslt_plan_cache": native_cublaslt_plan_cache,
        "candidate_over_baseline_native_metrics": summarize_metric_ratios(
            sample_rows,
            baseline_native_metrics,
            candidate_native_metrics,
        ),
        "paired_samples": sample_rows,
    }
    if reference is not None:
        payload["reference_seconds"] = summarize(reference_seconds)
        payload["reference_over_baseline"] = summarize(reference_over_baseline_ratios)
        payload["candidate_over_reference"] = summarize(candidate_over_reference_ratios)
        payload["reference_over_baseline_native_metrics"] = summarize_metric_ratios(
            sample_rows,
            baseline_native_metrics,
            reference_native_metrics,
            numerator_name="reference",
        )
        payload["candidate_over_reference_native_metrics"] = summarize_metric_ratios(
            sample_rows,
            reference_native_metrics,
            candidate_native_metrics,
            denominator_name="reference",
        )
    payload["metric_ratio_gates"] = evaluate_metric_ratio_limits(payload, metric_ratio_limits)
    payload["candidate_reference_metric_ratio_gates"] = evaluate_metric_ratio_limits(
        payload,
        candidate_reference_metric_ratio_limits,
        ratio_key="candidate_over_reference_native_metrics",
    )
    payload["native_lm_head_true_fused_target"] = summarize_lm_head_true_fused_target(payload)
    payload["native_lm_head_true_fused_gate"] = evaluate_lm_head_true_fused_gate(
        required=bool(args.require_native_lm_head_true_fused),
        target=payload["native_lm_head_true_fused_target"],
    )
    payload["native_route_change_gate"] = evaluate_native_route_change_gate(
        required=bool(args.require_native_route_change),
        route_changes=native_route_counter_changes,
        strategy_changes=native_strategy_value_changes,
        linear_shape_stats=native_linear_shape_stats,
        cublaslt_plan_cache=native_cublaslt_plan_cache,
    )
    return payload


def print_text(payload: dict[str, object]) -> None:
    print("Paired kernel speed comparison")
    print(f"  measurement: {payload['measurement']}")
    print(f"  samples: {payload['samples']}")
    print(f"  warmup: {payload['warmup']}")
    cuda_device_selection = payload.get("cuda_device_selection")
    if isinstance(cuda_device_selection, dict):
        print(
            "  cuda_visible_devices: "
            f"requested={cuda_device_selection.get('requested', '')} "
            f"resolved={cuda_device_selection.get('resolved', '')} "
            f"mode={cuda_device_selection.get('mode', '')}"
        )
    if payload.get("dry_run_plan") is True:
        print("  dry_run_plan: true")
        print(
            "  baseline: "
            f"{shlex.join([str(item) for item in payload.get('baseline_command', [])])}"
        )
        print(
            "  candidate: "
            f"{shlex.join([str(item) for item in payload.get('candidate_command', [])])}"
        )
        reference_command = payload.get("reference_command")
        if isinstance(reference_command, list) and reference_command:
            print(
                "  reference: "
                f"{shlex.join([str(item) for item in reference_command])}"
            )
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata:
            print(f"  metadata: {json.dumps(metadata, sort_keys=True)}")
        order_plan = payload.get("sample_order_plan")
        if isinstance(order_plan, list):
            rendered = [
                "/".join(str(item) for item in row.get("order", []))
                for row in order_plan
                if isinstance(row, dict)
            ]
            if rendered:
                print(f"  sample_order_plan: {', '.join(rendered)}")
        return
    print(f"  require_idle_selected_gpu: {payload.get('require_idle_selected_gpu', False)}")
    print(
        "  max_selected_gpu_utilization_pct: "
        f"{payload.get('max_selected_gpu_utilization_pct', -1.0)}"
    )
    print(
        "  selected_gpu_utilization_retries: "
        f"{payload.get('selected_gpu_utilization_retries', 1)} "
        "interval_seconds="
        f"{payload.get('selected_gpu_utilization_retry_interval_seconds', 0.0)}"
    )
    print(
        "  allow_stale_selected_gpu_utilization_without_compute_processes: "
        f"{payload.get('allow_stale_selected_gpu_utilization_without_compute_processes', False)}"
    )
    if payload.get("gpu_benchmark_lock_enabled") is not None:
        print(
            "  gpu_benchmark_lock: "
            f"enabled={payload.get('gpu_benchmark_lock_enabled', False)} "
            f"acquired={payload.get('gpu_benchmark_lock_acquired', False)} "
            f"path={payload.get('gpu_benchmark_lock_path', '')} "
            f"timeout={payload.get('gpu_benchmark_lock_timeout_seconds', 0.0)}"
        )
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata:
        print(f"  metadata: {json.dumps(metadata, sort_keys=True)}")
    baseline_env = payload.get("baseline_env")
    candidate_env = payload.get("candidate_env")
    reference_env = payload.get("reference_env")
    if isinstance(baseline_env, dict) and baseline_env:
        print(f"  baseline_env: {json.dumps(baseline_env, sort_keys=True)}")
    if isinstance(candidate_env, dict) and candidate_env:
        print(f"  candidate_env: {json.dumps(candidate_env, sort_keys=True)}")
    if isinstance(reference_env, dict) and reference_env:
        print(f"  reference_env: {json.dumps(reference_env, sort_keys=True)}")
    shape_stats = payload.get("native_linear_shape_stats")
    has_shape_plan_change = (
        isinstance(shape_stats, dict)
        and shape_stats.get("has_cublaslt_plan_change") is True
    )
    plan_cache = payload.get("native_cublaslt_plan_cache")
    has_plan_cache_change = (
        isinstance(plan_cache, dict)
        and plan_cache.get("has_plan_cache_change") is True
    )
    profile_json_dir = payload.get("append_native_profile_json_dir")
    if isinstance(profile_json_dir, str) and profile_json_dir:
        print(f"  append_native_profile_json_dir: {profile_json_dir}")
    print(f"  native_stage_timing: {payload.get('native_stage_timing', False)}")
    gpu_before = payload.get("gpu_before")
    if isinstance(gpu_before, dict):
        gpus = gpu_before.get("gpus")
        if isinstance(gpus, list) and gpus:
            print("  gpu_before:")
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                print(
                    "    "
                    f"index={gpu.get('index', '')} name={gpu.get('name', '')} "
                    f"display_active={gpu.get('display_active', '')} "
                    f"util={gpu.get('utilization.gpu_pct', '')}% "
                    f"mem={gpu.get('memory.used_mib', '')}/{gpu.get('memory.total_mib', '')} MiB"
                )
        processes = gpu_before.get("compute_processes")
        if isinstance(processes, list):
            print(f"  gpu_compute_processes_before: {len(processes)}")
    paired_samples = payload.get("paired_samples")
    if isinstance(paired_samples, list):
        timeout_counts = {"baseline": 0, "candidate": 0}
        if isinstance(payload.get("reference_command"), list) and payload.get("reference_command"):
            timeout_counts["reference"] = 0
        sample_process_counts: list[int] = []
        for row in paired_samples:
            if not isinstance(row, dict):
                continue
            for name in timeout_counts:
                command = row.get(name)
                if isinstance(command, dict) and command.get("timed_out") is True:
                    timeout_counts[name] += 1
            gpu_before_sample = row.get("gpu_before")
            if isinstance(gpu_before_sample, dict):
                processes = gpu_before_sample.get("compute_processes")
                if isinstance(processes, list):
                    sample_process_counts.append(len(processes))
        if any(timeout_counts.values()):
            print(
                "  command_timeouts: "
                + " ".join(f"{name}={count}" for name, count in timeout_counts.items())
            )
        if sample_process_counts:
            print(
                "  gpu_compute_processes_per_sample_before: "
                f"min={min(sample_process_counts)} max={max(sample_process_counts)}"
            )
    gpu_sample_summary = payload.get("gpu_sample_summary")
    if isinstance(gpu_sample_summary, dict):
        print(
            "  gpu_sample_summary: "
            f"selected_index={gpu_sample_summary.get('selected_gpu_index', '')} "
            f"selected_uuid={gpu_sample_summary.get('selected_gpu_uuid', '')}"
        )
        for key in (
            "selected_gpu_utilization_before_pct",
            "selected_gpu_utilization_after_pct",
            "selected_gpu_memory_used_before_mib",
            "selected_gpu_memory_used_after_mib",
            "selected_gpu_compute_process_count_before",
            "selected_gpu_compute_process_count_after",
            "compute_process_count_before",
            "compute_process_count_after",
        ):
            stats = gpu_sample_summary.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )
    timing_keys = ["baseline_seconds", "candidate_seconds", "candidate_over_baseline"]
    if isinstance(payload.get("reference_command"), list) and payload.get("reference_command"):
        timing_keys.extend(
            ["reference_seconds", "reference_over_baseline", "candidate_over_reference"]
        )
    for key in timing_keys:
        if key not in payload:
            continue
        stats = payload[key]
        assert isinstance(stats, dict)
        print(
            f"  {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )
    print_native_hot_summary(payload)
    print_lm_head_true_fused_target(payload)
    print_lm_head_true_fused_gate(payload)
    for section in ("baseline_native_metrics", "candidate_native_metrics", "reference_native_metrics"):
        metrics = payload.get(section)
        if not isinstance(metrics, dict) or not metrics:
            continue
        print(f"  {section}:")
        for key in NATIVE_TEXT_METRIC_KEYS:
            stats = metrics.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )
    for section in (
        "baseline_native_metric_values",
        "candidate_native_metric_values",
        "reference_native_metric_values",
    ):
        values = payload.get(section)
        if not isinstance(values, dict) or not values:
            continue
        print(f"  {section}:")
        for key in NATIVE_STRATEGY_METRIC_KEYS:
            observed = values.get(key)
            if isinstance(observed, list) and observed:
                print(f"    {key}: {', '.join(str(item) for item in observed)}")
    route_changes = payload.get("native_route_counter_changes")
    strategy_changes = payload.get("native_strategy_value_changes")
    has_strategy_change = (
        isinstance(strategy_changes, dict)
        and strategy_changes.get("has_strategy_value_change") is True
    )
    if isinstance(strategy_changes, dict) and int(strategy_changes.get("tracked_count", 0) or 0) > 0:
        changed = strategy_changes.get("changed")
        changed_count = strategy_changes.get("changed_count", 0)
        print(
            "  native_strategy_value_changes: "
            f"has_strategy_value_change={str(strategy_changes.get('has_strategy_value_change', False)).lower()} "
            f"changed_count={changed_count}"
        )
        if isinstance(changed, dict):
            for key in NATIVE_STRATEGY_METRIC_KEYS:
                stats = changed.get(key)
                if not isinstance(stats, dict):
                    continue
                baseline_values = stats.get("baseline_values")
                candidate_values = stats.get("candidate_values")
                if not isinstance(baseline_values, list) or not isinstance(candidate_values, list):
                    continue
                baseline_text = ", ".join(str(item) for item in baseline_values)
                candidate_text = ", ".join(str(item) for item in candidate_values)
                print(f"    {key}: baseline={baseline_text} candidate={candidate_text}")
    if isinstance(route_changes, dict) and int(route_changes.get("tracked_count", 0) or 0) > 0:
        changed = route_changes.get("changed")
        changed_count = route_changes.get("changed_count", 0)
        print(
            "  native_route_counter_changes: "
            f"has_route_counter_change={str(route_changes.get('has_route_counter_change', False)).lower()} "
            f"has_hot_route_counter_change={str(route_changes.get('has_hot_route_counter_change', False)).lower()} "
            f"changed_count={changed_count}"
        )
        if isinstance(changed, dict):
            for key in NATIVE_ROUTE_COUNTER_KEYS:
                stats = changed.get(key)
                if not isinstance(stats, dict):
                    continue
                ratio = stats.get("ratio")
                ratio_text = "none" if ratio is None else f"{float(ratio):.6f}"
                print(
                    f"    {key}: baseline_mean={float(stats['baseline_mean']):.6f} "
                    f"candidate_mean={float(stats['candidate_mean']):.6f} "
                    f"delta={float(stats['delta']):.6f} ratio={ratio_text}"
                )
        if (
            route_changes.get("has_route_counter_change") is False
            and isinstance(candidate_env, dict)
            and bool(candidate_env)
            and not has_strategy_change
            and not has_shape_plan_change
            and not has_plan_cache_change
        ):
            print(
                "    note: tracked route counters did not change; treat timing-only "
                "candidate improvements as noise until a route, strategy, or separate "
                "kernel-level attribution confirms the candidate."
            )
        if (
            route_changes.get("has_route_counter_change") is True
            and route_changes.get("has_hot_route_counter_change") is False
            and isinstance(candidate_env, dict)
            and bool(candidate_env)
            and not has_strategy_change
            and not has_shape_plan_change
            and not has_plan_cache_change
        ):
            print(
                "    note: only setup/prewarm route counters changed; treat "
                "throughput improvements as setup noise until a hot training "
                "route, strategy, shape, or plan-cache change confirms the candidate."
            )
    route_gate = payload.get("native_route_change_gate")
    if isinstance(route_gate, dict) and route_gate.get("enabled") is True:
        print(f"  native_route_change_gate: passed={str(route_gate.get('passed', False)).lower()}")
        print(
            "    changes: "
            f"route_counter={str(route_gate.get('has_route_counter_change', False)).lower()} "
            f"hot_route_counter={str(route_gate.get('has_hot_route_counter_change', False)).lower()} "
            f"strategy={str(route_gate.get('has_strategy_value_change', False)).lower()} "
            f"linear_shape={str(route_gate.get('has_linear_shape_change', False)).lower()} "
            f"cublaslt_plan_cache={str(route_gate.get('has_cublaslt_plan_cache_change', False)).lower()}"
        )
        failure_reason = route_gate.get("failure_reason")
        if failure_reason:
            print(f"    failure_reason: {failure_reason}")
    if isinstance(plan_cache, dict) and plan_cache.get("enabled") is True:
        print("  native_cublaslt_plan_cache:")
        plan_changes = plan_cache.get("plan_cache_changed")
        if isinstance(plan_changes, list) and plan_changes:
            print(
                "    plan_cache_changes: "
                f"changed_count={plan_cache.get('plan_cache_changed_count', len(plan_changes))}"
            )
            for change in plan_changes[:10]:
                if not isinstance(change, dict):
                    continue
                shape = change.get("shape")
                changed = change.get("changed")
                if not isinstance(changed, dict):
                    continue
                changed_fields = ", ".join(sorted(changed))
                print(f"      {shape}: changed={changed_fields}")
        print(
            f"    shared_count={plan_cache.get('shared_count', 0)} "
            f"baseline_only={len(plan_cache.get('baseline_only', []) or [])} "
            f"candidate_only={len(plan_cache.get('candidate_only', []) or [])}"
        )
    if isinstance(shape_stats, dict) and shape_stats.get("enabled") is True:
        print("  native_linear_shape_stats:")
        plan_changes = shape_stats.get("cublaslt_plan_changed")
        if isinstance(plan_changes, list) and plan_changes:
            print(
                "    cublaslt_plan_changes: "
                f"changed_count={shape_stats.get('cublaslt_plan_changed_count', len(plan_changes))}"
            )
            for change in plan_changes[:10]:
                if not isinstance(change, dict):
                    continue
                shape = change.get("shape")
                baseline_selected = change.get("baseline_selected_heuristics")
                candidate_selected = change.get("candidate_selected_heuristics")
                ratio = change.get("candidate_avg_us_over_baseline")
                ratio_text = (
                    "none" if not isinstance(ratio, (int, float)) else f"{float(ratio):.6f}"
                )
                print(
                    f"      {shape}: selected={baseline_selected}->{candidate_selected} "
                    f"candidate_avg_us_over_baseline={ratio_text}"
                )
        shared = shape_stats.get("shared")
        if isinstance(shared, list):
            for row in shared[:20]:
                if not isinstance(row, dict):
                    continue
                shape = row.get("shape")
                baseline = row.get("baseline")
                candidate = row.get("candidate")
                if not isinstance(shape, str) or not isinstance(baseline, dict) or not isinstance(candidate, dict):
                    continue
                baseline_avg = baseline.get("avg_us")
                candidate_avg = candidate.get("avg_us")
                baseline_total = baseline.get("total_us")
                ratio = row.get("candidate_avg_us_over_baseline")
                if not isinstance(baseline_avg, dict) or not isinstance(candidate_avg, dict):
                    continue
                base_mean = baseline_avg.get("mean")
                cand_mean = candidate_avg.get("mean")
                total_mean = baseline_total.get("mean") if isinstance(baseline_total, dict) else None
                if not isinstance(base_mean, (int, float)) or not isinstance(cand_mean, (int, float)):
                    continue
                ratio_text = "none" if not isinstance(ratio, (int, float)) else f"{float(ratio):.6f}"
                selected_base = baseline.get("cublaslt_selected_heuristics")
                selected_candidate = candidate.get("cublaslt_selected_heuristics")
                selected_text = ""
                if isinstance(selected_base, list) or isinstance(selected_candidate, list):
                    selected_text = (
                        " cublaslt_selected="
                        f"{selected_base if isinstance(selected_base, list) else []}->"
                        f"{selected_candidate if isinstance(selected_candidate, list) else []}"
                    )
                total_text = "" if not isinstance(total_mean, (int, float)) else f" baseline_total_us={float(total_mean):.3f}"
                print(
                    f"    {shape}: baseline_avg_us={float(base_mean):.3f} "
                    f"candidate_avg_us={float(cand_mean):.3f} ratio={ratio_text}"
                    f"{total_text}{selected_text}"
                )
        baseline_only = shape_stats.get("baseline_only")
        candidate_only = shape_stats.get("candidate_only")
        if isinstance(baseline_only, list) and baseline_only:
            print(f"    baseline_only: {', '.join(str(item) for item in baseline_only[:10])}")
        if isinstance(candidate_only, list) and candidate_only:
            print(f"    candidate_only: {', '.join(str(item) for item in candidate_only[:10])}")
    ratios = payload.get("candidate_over_baseline_native_metrics")
    if isinstance(ratios, dict) and ratios:
        print("  candidate_over_baseline_native_metrics:")
        for key in NATIVE_TEXT_METRIC_KEYS:
            stats = ratios.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )
    for section in ("reference_over_baseline_native_metrics", "candidate_over_reference_native_metrics"):
        ratios = payload.get(section)
        if not isinstance(ratios, dict) or not ratios:
            continue
        print(f"  {section}:")
        for key in NATIVE_TEXT_METRIC_KEYS:
            stats = ratios.get(key)
            if isinstance(stats, dict):
                print(
                    f"    {key}: mean={stats['mean']:.6f} median={stats['median']:.6f} "
                    f"min={stats['min']:.6f} max={stats['max']:.6f}"
                )
    for gate_key in ("metric_ratio_gates", "candidate_reference_metric_ratio_gates"):
        gates = payload.get(gate_key)
        if not isinstance(gates, dict) or gates.get("enabled") is not True:
            continue
        print(f"  {gate_key}: passed={str(gates.get('passed', False)).lower()}")
        results = gates.get("results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                actual = result.get("actual_ratio")
                if actual is None:
                    actual = result.get("actual_mean_ratio")
                actual_text = "missing" if actual is None else f"{float(actual):.6f}"
                stat = str(result.get("stat", "mean"))
                min_text = (
                    ""
                    if result.get("min_ratio") is None
                    else f" min_ratio={float(result.get('min_ratio', 0.0)):.6f}"
                )
                max_text = (
                    ""
                    if result.get("max_ratio") is None
                    else f" max_ratio={float(result.get('max_ratio', 0.0)):.6f}"
                )
                print(
                    f"    {stat}:{result.get('metric', '')}: actual_ratio={actual_text} "
                    f"{min_text}{max_text} passed={str(result.get('passed', False)).lower()}"
                )


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    if str(args.json_out or "").strip():
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(payload)
    for gate_key, label in (
        ("metric_ratio_gates", "metric ratio gate"),
        ("candidate_reference_metric_ratio_gates", "candidate reference metric ratio gate"),
    ):
        gates = payload.get(gate_key)
        if not (
            isinstance(gates, dict)
            and gates.get("enabled") is True
            and gates.get("passed") is False
        ):
            continue
        failed = [
            str(result.get("metric", ""))
            for result in gates.get("results", [])
            if isinstance(result, dict) and result.get("passed") is False
        ]
        print(
            f"{label} failed"
            + (": " + ", ".join(metric for metric in failed if metric) if failed else ""),
            file=sys.stderr,
        )
        return 1
    lm_head_gate = payload.get("native_lm_head_true_fused_gate")
    if (
        isinstance(lm_head_gate, dict)
        and lm_head_gate.get("enabled") is True
        and lm_head_gate.get("passed") is False
    ):
        reason = str(lm_head_gate.get("failure_reason", "")).strip()
        print(
            "native LM-head true-fused gate failed"
            + (": " + reason if reason else ""),
            file=sys.stderr,
        )
        return 1
    route_gate = payload.get("native_route_change_gate")
    if (
        isinstance(route_gate, dict)
        and route_gate.get("enabled") is True
        and route_gate.get("passed") is False
    ):
        print(
            "native route change gate failed: "
            + str(route_gate.get("failure_reason", "candidate-native-route-unchanged")),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
