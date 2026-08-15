# Pinned llama.cpp MMQ source subset

This directory contains an unmodified, dependency-closed subset of the
MIT-licensed `ggml` CUDA headers from llama.cpp commit
`62bf73d25c53b8161f8a22894d4f90c4aebbd7d0`. NeuralFn uses it only to compile
the Q4_K, Q5_K, and Q6_K small-batch MMQ device templates required by the
official Muse Glimmer GGUF artifacts. NeuralFn owns the surrounding C ABI,
validation, workspace, scheduling, and resident-model integration; no ggml
runtime, model object, allocator, or shared library is linked.

The upstream MIT license is preserved in `LICENSE`. Do not update individual
files in place. Any refresh must pin a new immutable upstream commit, copy the
complete compiler dependency closure, record the new provenance, and rerun
byte-level Q8/MMQ parity plus whole-model token-parity tests against both the
old implementation and the independently built pinned llama.cpp oracle.
