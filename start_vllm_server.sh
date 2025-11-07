#!/usr/bin/env bash
./stop_vllm_server.sh

# Help PyTorch avoid fragmentation when (re)allocating CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python vllm_server.py "$@"
