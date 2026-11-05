# Codex repository instructions

Read `PROJECT_BRIEF.md` before any implementation. The user wants local CPU/GPU inference for extremely large LLMs via **dynamic streaming from NVMe**, using a 16 GB RAM MSI Katana GF66 (i7-12700H; RTX 3050 Laptop 4 GiB). Correctness first, speed second.

## Non-negotiable
- Never silently quantize original model weights. BF16/F16 primary; native lower precision only when original checkpoint uses that representation, documented separately. Optional quantized kernels may be added later.
- Keep checkpoint originals immutable. No giant model download until user has approved disk space consumption.
- Preserve CPU fallback. Any GPU kernel must tile weights so a tensor can exceed 4 GiB VRAM, with numerical comparisons.
- Report peak RSS / GPU memory and disk read volume; a successful short test does not prove a whole LLM is runnable.
- Only one internal NVMe slot, currently occupied. Do not repartition, overwrite, erase or migrate drives. Avoid uncontrolled multi-GB scratch writes.
- Do not git push, change remote repos, or install system-wide packages without approval. Audit local repos before edits.
- Separate benchmark performance claims from assumptions. Explain cache-warm vs cold read measurements and temperature/power state.
- Treat the independent user repository https://github.com/francescodifilippo/spikingbrain-cpu as a numerical reference; don't merge unrelated experiment branches into it.
- Keep DeepSeek Harness as optional client; don't build a replacement agentic UI.

## Preferred workflow
1. Inspect target environment and repo, report facts.
2. State plan, small scoped code change, and numerical test.
3. Run test and benchmark, report failures and measured evidence.
4. Commit only when the user wants versioned code; ask before pushes.

## First run
Request: `Read PROJECT_BRIEF.md. Audit hardware, drivers, storage, CUDA and any existing PoC. Build a small BF16 streamed-matrix benchmark with bounded RAM and VRAM; compare to CPU reference. No quantization.`
