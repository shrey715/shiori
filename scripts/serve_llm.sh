#!/usr/bin/env bash
# Serve Qwen3-32B-AWQ (Qwen's official 4-bit release) with vLLM as an OpenAI-compatible API on localhost:8000.
# vLLM lives in its own venv because it pins its own torch.
# Usage: scripts/serve_llm.sh            (logs to outputs/vllm.log)
#
# Flags:
#   --tensor-parallel-size 4        split the model over all 4 GPUs (~5 GB of weights each)
#   --dtype float16                 Turing (2080 Ti) has no bf16; AWQ kernels run in fp16 anyway
#   --kv-cache-dtype auto           KV cache in fp16, no lossy fp8 cache
#   --gpu-memory-utilization 0.72   + CUDA graphs (~0.5 GB) leaves ~2.5 GB on GPU 0 for the notebook's retrieval models
#   --max-model-len 4096            prompts are < 1k tokens; a smaller cap means more KV blocks per request
#   --enable-prefix-caching         all prompts share a system prompt; its KV is computed once and reused
#   --max-num-batched-tokens 8192   chunked prefill: long prompts are split so decoding never stalls
#   --max-num-seqs 64               up to 64 requests decoded together (continuous batching); the notebook sends 64 at a time
#   --max-cudagraph-capture-size 64 CUDA graphs only up to batch 64 (default 256 costs ~1.5 GB per GPU we never use)
#   --generation-config vllm        ignore the model's default sampling settings; the notebook sends temperature 0
#   --seed 0                        reproducible
#   (no --enforce-eager: CUDA graphs stay on for faster decoding)
set -u
cd "$(dirname "$0")/.."
VENV=${VLLM_VENV:-/ssd_scratch/shreyas.deb/rag_assignment/vllm-venv}
export HF_HOME=$PWD/models/hf HF_HUB_OFFLINE=1
CACHE=/ssd_scratch/shreyas.deb/rag_assignment/cache          # compile caches on scratch (~/.triton is a dangling link)
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor VLLM_CACHE_ROOT=$CACHE/vllm
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT
exec "$VENV/bin/vllm" serve Qwen/Qwen3-32B-AWQ \
  --served-model-name qwen3-32b \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 64 \
  --generation-config vllm \
  --seed 0 \
  --port 8000 >> outputs/vllm.log 2>&1
