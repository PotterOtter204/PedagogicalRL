#!/usr/bin/env bash
./stop_vllm_server.sh

# Ensure PyTorch expandable_segments is NOT set; it's incompatible with vLLM CuMemAllocator
unset PYTORCH_CUDA_ALLOC_CONF

python vllm_server.py "$@"
