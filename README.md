# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunks of the **ICC Men's T20 World Cup 2026 Playing Conditions** (106 pages, Feb 2026), compared on 40 graded questions.

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| retrieve | bi-encoder, top 10 by cosine (FAISS) | same top 10 |
| re-rank | – | cross-encoder scores each (question, chunk) pair |
| answer | top 3 → LLM → **Answer 1** | new top 3 → LLM, same prompt → **Answer 2** |

A **No RAG** answer (question only) shows what the LLM knows without the document.

Everything is in one notebook: [`rag_reranking.ipynb`](rag_reranking.ipynb).

## Results

**Retrieval** (40 questions; gold chunk in the top 3, and MRR@10):

| setup | hit@3 baseline → re-ranked | MRR baseline → re-ranked |
|---|---|---|
| MiniLM, plain chunks | 0.65 → 0.70 | 0.53 → 0.65 |
| MiniLM, titled chunks | 0.62 → 0.68 | 0.50 → 0.65 |
| bge-small, plain chunks | 0.70 → 0.78 | 0.63 → 0.69 |
| bge-small, titled chunks | 0.75 → 0.75 | 0.60 → 0.69 |

**Answers** (mean score: correct 1, partial 0.5):

| | No RAG | Answer 1 | Answer 2 |
|---|---|---|---|
| tier 1: direct | 0.30 | 1.00 | 1.00 |
| tier 2: paraphrase | 0.50 | 0.71 | 0.75 |
| tier 3: exception / negation | 0.50 | 0.57 | 0.59 |
| tier 4: traps / cross-references | 0.10 | 0.32 | 0.40 |
| general cricket law | 0.53 | 0.67 | 0.63 |
| 2026 tournament-specific | 0.20 | 0.64 | 0.73 |

Without the document the LLM is wrong 55% of the time.

**Latency** (per query, GPU): Pipeline 1 **6.2 ms**, Pipeline 2 **19.8 ms** (3.2×). A cross-encoder over all 440 chunks takes **295 ms**, against 0.05 ms for the FAISS search.

## Setup

| part | model |
|---|---|
| bi-encoders | `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` |
| cross-encoders | `cross-encoder/ms-marco-MiniLM-L6-v2` (Answer 2), `BAAI/bge-reranker-base` (rankings only) |
| LLM + grader | `Qwen/Qwen3-32B-AWQ` via vLLM, tensor-parallel over 4× RTX 2080 Ti, temperature 0 |
| grader check | blind second grading by Claude Haiku: 79% exact agreement, 97% within one step |

## Repository

```
rag_reranking.ipynb   the whole project, with outputs
eval/                 questions.csv (40 questions, gold anchors, expected answers), grader check files
outputs/              retrieval runs, answers + grades, latency, LLM answer cache
scripts/              serve_llm.sh (vLLM server), run_notebook.py, PDF / chunk inspection helpers
report/               report.tex / report.pdf
docs/                 assignment brief
```

The ICC PDF is not committed (copyright): the notebook downloads it from the ICC site and checks its SHA-256.

## Reproduce

Python 3.12, an NVIDIA driver for CUDA 12.8+.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cu128
.venv/bin/python -m ipykernel install --user --name rag_assignment
.venv/bin/python scripts/run_notebook.py          # or open rag_reranking.ipynb in Jupyter
```

Models download from Hugging Face on first use. LLM answers and grades are cached in `outputs/llm_cache.json`, so the notebook runs without an LLM server.

To regenerate answers, the notebook starts the vLLM server itself (`scripts/serve_llm.sh`) when a prompt isn't cached. vLLM goes in its own venv because it pins its own torch:

```bash
uv venv --python 3.12 vllm-venv
uv pip install --python vllm-venv/bin/python \
  "https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --torch-backend cu128
VLLM_VENV=$PWD/vllm-venv scripts/serve_llm.sh
```
