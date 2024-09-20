# SLAM
## Abstract
Recent work on Large Language Models (LLMs) has largely overlooked the fault-resilience during inference. The intuitive recovery approach of restarting jobs from scratch when a failure occurs is inefficient especially when serving long-context requests to large models, such as streaming applications or multi-model tasks. As the context length increases, the memory for saving intermediate results such as key-value (KV) caches would quadratically increase, which poses a critical challenge for efficient recovery. In this work, we propose SLAM, a hybrid method that compresses KV caches to enable efficient long-context model inference, and simultaneously resolving large overheads in previous fault-resiliency challenges. We first observe that sparsity exists in attention scores across layers, which indicates only a small portion of tokens in the KV cache will affect model quality significantly. Inspired by this insight, we regulate three properties and design an approximation algorithm to estimate the criticality of tokens. SLAM will only compute attention with those selected critical tokens during decoding phases. To achieve better performace, SLAM keeps track of the middle critical tokens and perform query-aware replacement while keeping the initial and most recent tokens critical on the fly. By merely keeping critical tokens alive, SLAM can reduce memory footprints and theoretically improving throughput up to 9X while preserving model performance. Furthermore, we design a system that integrates SLAM with scheduling efforts on multi-GPU/node settings, enabling fault-resilient LLM inference with low memory footprints. The next step is to integrate SLAM with a standard inference engine and evaluate our system on LongBench, a long-context inference testbench for LLMs.

## Usage

### Environment Setup

```bash
conda create -yn slam python=3.8
conda activate slam

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```

### Run Streaming Llama Chatbot

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py  --enable_streaming
``

## TODOs
We will release the code and data in the following order, please stay tuned!

- [x] Release demo.
- [ ] Release perplexity evaluation code
- [ ] Release efficienct evaluation code
- [ ] Release fault-tolerance setups

