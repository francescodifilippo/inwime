# Codex repository instructions

Read `docs/VISION_AND_ROADMAP.md` for the authoritative current mission and roadmap, then `PROJECT_BRIEF.md` for historical context and operational constraints, before any implementation. **North star (owner clarification 2026-11-23): progressively execute any AI model, of any size, using whatever hardware and storage are available, within objectively necessary per-model/per-host resource minima.** CPU, GPU, NPU/other accelerators and potentially distributed nodes can cooperate; hardware placement is adaptive. **CPU-first applies only to the initial implementation/verification stage.** The 16 GiB MSI Katana GF66 (i7-12700H; RTX 3050 4 GiB), WSL and NVMe streaming are the current development testbed, not universal product requirements. Fidelity, verified executability and bounded resources precede performance.

## Non-negotiable
- Never silently quantize original model weights. BF16/F16 primary; native lower precision only when original checkpoint uses that representation, documented separately. Optional quantized kernels may be added later.
- Keep checkpoint originals immutable. No giant model download until user has approved disk space consumption.
- Preserve the verified CPU reference path during initial development and a CPU fallback where actually supported; do not turn CPU availability into a global requirement for future architectures. GPU/accelerator kernels must respect their own memory budgets (including tiling when required), with numerical comparisons.
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

## Historical first run (already completed; not an active task)
Request: `Read PROJECT_BRIEF.md. Audit hardware, drivers, storage, CUDA and any existing PoC. Build a small BF16 streamed-matrix benchmark with bounded RAM and VRAM; compare to CPU reference. No quantization.`

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues of `francescodifilippo/inwime` (`gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` plus `docs/adr/`, created lazily. See `docs/agents/domain.md`.
