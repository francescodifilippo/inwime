# Project handoff — Out-of-core LLM inference, MSI Katana GF66

**Prepared:** 2026-11-04  
**Destination:** Codex Local / Codex CLI on the MSI Katana GF66, Linux x86-64 if available.  
**Status:** Design and partial proof-of-concept discussed in ChatGPT. This is NOT an end-to-end working LLM runtime.

## Goal

Build a reusable **local inference backend** for very large LLMs, including sparse Mixture-of-Experts models (GLM, DeepSeek, Qwen, Kimi), while keeping full model weights on SSD NVMe and processing them in bounded RAM / GPU VRAM buffers. Latency may be very high; correctness and extreme model size take priority. Use **DeepSeek Harness** as an agentic client/orchestrator once the backend provides a compatible HTTP API; do NOT recreate an agent orchestrator.

### Constraints, in priority order

1. Host is ordinary x86-64 hardware with commodity DDR RAM, 16 GB physically installed; 32 GB was an upper bound, not a target.
2. **No mandatory weight quantization.** Primary supported checkpoint formats should be original BF16/FP16, preserving file data; if an original checkpoint is FP8/MXFP4, distinguish its native representation from conversion. Quantized formats may be added as OPTIONAL execution kernels later.
3. **Dynamic streaming from NVMe via ordinary RAM to GPU VRAM**, with bounded memory footprint. CPU-only fallback required. Avoid relying entirely on OS swap. VRAM must not impose a requirement that the *entire model* fits, but individual operations must be tiled or have a CPU fallback when necessary.
4. The laptop GPU is optional for correctness; accelerate the actual model's numerical operations, not only a separate tiny assistant.
5. Multi-PC/distributed execution is optional, later; not needed for the first implementation.
6. Verify model-specific attention, MoE routing, tensor layout and reference logits. Don't silently replace an architecture with a spiking one or alter weights without making it an explicitly separate experiment.

## Confirmed target hardware (user supplied)

- **Machine:** MSI Katana GF66.
- **CPU:** Intel Core i7-12700H, 14 cores (6P+8E), 20 threads.
- **Memory:** 16 GB ordinary DDR RAM.
- **GPU:** NVIDIA GeForce RTX 3050 Laptop, **4096 MiB GDDR6 VRAM** (Compute Capability 8.6; validate driver/CUDA runtime on this host).
- **Internal SSD:** 2 TB NVMe, only approximately **200 GB currently free**. **The laptop has ONE internal NVMe slot** and it is occupied; installing a larger internal unit requires replacing the drive and migrating the existing installation.
- **External SSD:** 1 TB NVMe in USB 3 enclosure. Available free space not stated. Actual USB negotiated speed and sustained sequential throughput unknown.
- **Operating system:** Not definitively stated in this handoff; commands in prior discussion were Linux, so detect and confirm rather than assume.
- **Additional hardware:** User can obtain more storage, but prefers no purchase until benchmarks identify a bottleneck.

## Intended architecture

```
DeepSeek Harness / other client
      |
OpenAI-compatible inference API (/v1/models, /v1/chat/completions)
      |
model-specific adapter: tokenizer, attention, MoE, KV/recurrent state
      |
pluggable execution planner (CPU-only or CPU+RTX 3050)
      |
NVMe safetensors source -> bounded host staging -> CUDA async H2D tiles
      |                                    |                 |
      +-------------> CPU kernels         +--> GPU BF16 kernels, 2 buffers, selective cache
```

### Engineering requirements

- Exact, read-only safetensors tensor indexing: dtype, shape, file, byte offsets, shard mapping; avoid loading complete layers by default.
- 16 GB system: reserve at least 3-4 GB for Linux; initially aim ~10-12 GB **application RSS**, revisit with measurements. Page-cache accounting and pinned RAM matter. A `RLIMIT_AS` cap is not a replacement for measuring physical RSS.
- RTX 3050: do not pre-allocate all 4 GB. Measure available memory when it also drives the display. Initially test 2x smaller GPU staging buffers (e.g. 64-256 MiB each), async H2D transfers and kernels; enlarge based on measured throughput and OOM safety. Pin only bounded host memory.
- Use CUDA streams/events to overlap disk read, H2D, compute wherever dependencies and bus bandwidth permit. GPU kernel should consume tile directly rather than dequantize weights needlessly.
- Distinguish read bandwidth of INTERNAL NVMe versus EXTERNAL USB3 NVMe, cold-ish versus warm page cache; monitor USB throttling, thermal and power limits of the mobile GPU.
- Keep pure-CPU baseline and independent numerical reference for comparisons.
- Keep KV cache/recurrent state bounded and explicitly accounted; latency in prefill and decode both measured.
- Do not infer that any model is executable end-to-end just because its individual matrix multiplication can stream.
- Preserve exact tokenizer, template, routing and operator semantics. Validate layer intermediates and final logits before claiming a functional model.

## Existing sources and work worth reusing

- Existing user-owned CPU port: https://github.com/francescodifilippo/spikingbrain-cpu
  - Independent reference implementation of SpikingBrain2.0 in PyTorch; contains numerical validation reports, weight loader, CPU HTTP server and known limits.
  - Author documented that the released SpikingBrain2.0 implementation does NOT include the advertised INT8-Spiking event-driven path; do not assume neuromorphic compute is already implemented.
  - SpikingBrain 2.0 issue reported by user: https://github.com/BICLab/SpikingBrain2.0/issues/7 — `FLACache` rotary-position off-by-one on decode for layers 1+, reproducible; compare the upstream cache fix.
- SpikingBrain 1.0 code: https://github.com/BICLab/SpikingBrain-7B — W8ASpike pseudo-spiking path, GLA/SWA, 76B MoE example. Its pseudo-spiking is NOT real asynchronous hardware spiking. Study as **optional** research, independent from lossless BF16 inference.
- SpikingBrain 2.0 code: https://github.com/BICLab/SpikingBrain2.0 — SSE/Gated DeltaNet/MoBA. The user’s independent CPU validation repository is a good exact reference.
- Relevant existing disk-streaming engines to audit: https://github.com/JustVugg/colibri , https://github.com/sqliteai/warp , https://github.com/ggml-org/llama.cpp . Do not import a quantized representation unnoticed. Colibri/WARP claims need verification for ORIGINAL BF16 on <16 GB host.
- An earlier tiny **ooc-x86-poc** C++/Python prototype was created in a previous ChatGPT environment: header-only safetensors manifest, bounded `pread` BF16/F16/F32 matvec, 3D expert slices, generic top-k SwiGLU MoE toy tests and read-only host probe. **This handoff does not contain that earlier source bundle**; if it was downloaded separately (`ooc-x86-poc.zip`), unpack and audit it rather than treating it as a complete runtime.

## Model selection

- Initial correctness reference: a small original BF16 model / one real layer, or Qwen3.6-35B-A3B BF16 (approx ~70 GB) once architecture support exists; no premature download unless enough space and model checksum/revision known.
- Large model target: GLM-5.3 BF16 (~1.5 TB of weights), and later other large MoE. The external USB3 drive can hold smaller checkpoints; whether it is fast enough for repeated streaming is **unknown**.
- Native FP8/MXFP4 checkpoints are accepted only by explicitly implementing their native formats; do not claim they are BF16 without conversion.

## What the user's SpikingBrain comparison established

- SpikingBrain 1.0 has a W8ASpike pseudo-spiking implementation and 76B-class MoE variant that SpikingBrain2.0's released 5B checkpoint lacks. The code is research for a potential separate sparse architecture.
- The user’s SpikingBrain2.0 CPU port matches upstream numerical operations and records that its INT8-spiking component is not released. Kernel optimizations preserving exact math are separate from changing attention to GLA/SSE/spiking, which changes the model and may require retraining.
- BrainChip Akida AKD1500 PCIe (Akida 1) and edge NPUs such as Hailo/Coral/Axelera are not currently validated as generic BF16 streaming devices. **Do not buy them for the main inference project before proving an appropriate kernel/transfer path.** A standard NVIDIA GPU on this laptop is the immediate candidate for accelerated streamed GEMV/GEMM.

## Milestones with acceptance criteria

### M0: audit the host and repos — READ ONLY
- Print `uname -a`, OS, CPU capabilities, `free -h`, `nvidia-smi`, CUDA toolchain, `lsblk -d`, `lsusb -t`, `lspci`, `findmnt` and mount free space.
- Determine exactly which external USB speed is negotiated and where the test files should reside.
- Check whether `ooc-x86-poc.zip` was downloaded, and whether local repos already exist. Audit first, don't overwrite.
- Report issues and proposed file changes BEFORE editing if substantial.

### M1: isolated, reproducible GPU streaming benchmark
- Build a tiny BF16 numeric kernel (e.g. GEMV/GEMM on 256-MiB to 2-GiB synthetic matrices) and a reference CPU computation using identical values.
- Compare CPU-only, synchronous H2D+GPU, and 2-buffer async H2D+GPU, using pinned host memory bounded well below 16GB.
- Also measure the internal NVMe read stage and USB3 NVMe read stage separately. No giant files, no persistent test debris.
- Report: wall time, disk read MB/s, H2D GB/s, kernel ms, overlap efficiency, RAM RSS peak, VRAM peak, numerical error, repeatability, power/thermal caveats.
- Demonstrate working weights larger than 4GB VRAM by tiling from disk, not by silently shrinking/quantizing the weights. Begin smaller if needed; protect laptop thermals.

### M2: model-specific path
- Choose one model and implement exact forward graph, tokenizer, attention and MoE, starting with a tractable checkpoint; keep native weights unchanged.
- Cross-check intermediate activations/logits with official reference (document revision, dtype and tolerance).
- Maintain bounded RSS and GPU VRAM; measure cold/warm inference.

### M3: model API and orchestration
- Serve OpenAI-compatible chat completions, streamed SSE tokens and robust tool-call formatting as supported by model. Integrate with DeepSeek Harness rather than recreating agent management.

### M4: optimization and optional hardware
- Test expert prefetch, caching, async disk I/O, CPU/GPU scheduling, dynamic attention state; only then decide SSD replacement or additional device.
- Optional separate experiments with W8ASpike/SpikingBrain2.0, keeping baseline quality.

## First request to Codex Local

Read PROJECT_BRIEF.md and AGENTS.md fully. First **inspect the Katana in read-only mode**, establish GPU driver/CUDA viability, available storage and exact USB 3 link speed. Locate any existing out-of-core PoC or repositories. Produce a benchmark plan before large downloads or writes. Then implement the smallest correctness-verified BF16 NVMe→RAM→RTX3050 streaming GEMV/GEMM experiment, with CPU baseline, 4GB VRAM ceiling, ~10GB RSS target, repeatable timing/measurement and zero quantization. Do not claim end-to-end LLM support until verified against official checkpoint outputs. Do not modify the existing system disk layout or issue destructive commands.

## Important terminology

**Codex Local means Codex works with LOCAL files, shells and processes on the Katana**. It does not imply Codex itself uses the target LLM running locally. Our inference runtime is the software that must run locally without depending on a remote model provider. Keep these two concepts separate.
