# ACCORD-KV: Project Summary

**ACCORD-KV (arXiv:2606.08635)** presents a decentralized KV cache reorganization framework for prefill-decode disaggregated LLM serving, where the KV cache is the dominant communication payload between prefill and decode workers. Existing KV management approaches treat this as either binary token selection (keep/drop) or uniform quantization—ignoring a critical asymmetry: Key tensors retain 94.13% of their energy at SVD rank 8, while Value tensors retain only 59.95%. We call this the **Value Bottleneck**, and we show it explains why adaptive-rank strategies historically underperform.

ACCORD-KV introduces **Attention Contracts**: lightweight semantic descriptors on each KV block encoding minimum precision requirements for correct attention. The system dispatches blocks to five contract types—ExactLocal (full FP16), SketchLocal (SVD+INT4), RemoteExact, Rehydrate, and Drop—via a **Serial Cascade** scheduler achieving 128–255× speedup at 0.22% relative error. For clustered access patterns, we propose **Cluster-conditional SVD (Method D)**, which outperforms H2O, StreamingLLM, Scissorhands, and FastGen by 11.6–12.2×. The (m,ℓ,γ) wire format achieves 31,775× backward compatibility with FlashAttention-compatible tuples.

GPU experiments on Mistral-7B-Instruct-v0.3 and Gemma-2-9B-it validate all four claims.
