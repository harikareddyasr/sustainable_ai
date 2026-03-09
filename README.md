# Greener AI — LLM Inference Benchmarking

Empirical study comparing FP32, FP16, INT8, and INT4 quantization 
across 6 transformer models on 3 NLP tasks (summarization, QA, reasoning).

Measures accuracy, latency, and CO₂ emissions using CodeCarbon on Tesla T4 GPU.

**Key finding:** Low-bit quantization (INT8/INT4) frequently increased 
emissions due to dequantization overhead — FP16 was the practical sweet spot.

Accepted at IEEE ICECET 2026 — to be presented July 2026, Rome, Italy.

## Tech Stack
Python · PyTorch · Hugging Face Transformers · CodeCarbon · Azure GPU VM · Pandas
