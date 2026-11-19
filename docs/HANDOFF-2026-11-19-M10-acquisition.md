# InWiMe — Handoff 2026-11-19: M10 acquisition, correcting a wrong memory policy

**Status:** recovery + minimal acquisition fix pending. **Do not download again yet; do not delete checkpoint partial files.**  
**Branch:** `feat/m10-preflight-tooling`  
**Last code SHA before this handoff:** `511617514a2b468959496729a542df36eff807ce`  
**PR:** https://github.com/francescodifilippo/inwime/pull/24 (draft; main remains `639de22ed7e30a796b831595cd0f9823d585de16`)  
**Decision / implementation tracker:** https://github.com/francescodifilippo/inwime/issues/25

## 1. Explicit architectural correction

The M10 checkpoint acquisition is a **normal streaming download of files from Hugging Face to an ext4 filesystem**, not model inference. Requiring a Windows RAM/paging `GO` profile or aborting on host `AvailableBytes < 300 MB` is **not justified for file acquisition**. The user expressly directed us to remove this coupling and finish the download.

Keep the host-memory preflight and detailed telemetry for later **inference / oracle / benchmarking**. For acquisition, keep integrity, disk space, network failures and resumability protections; low host RAM/paging alone must not trigger refusal or a host-pressure abort. Never suggest again that the whole checkpoint must fit in RAM to download.

The old policy is historical evidence (M10 amendments 001 and 002), **not a precedent to preserve as current acquisition policy**. Implement a narrowly scoped corrective amendment / version, not silent edits to frozen evidence.

## 2. Current repo and verified testing

- Base `main`: `639de22`, unchanged as of inspection.
- Feature branch: `5116175` immediately before this handoff.
- PR #24 contains C1–C5 plus amendments 001 (pre-download host gate and CI inventory) and 002 (in-run Windows host monitor).
- GitHub Actions run on `5116175`: https://github.com/francescodifilippo/inwime/actions/runs/44408356032 **SUCCESS**: **1169 passed, 11 skipped, 0 failed**; effective runner test inventory PASS, full inventory NOT PROVEN (11 M9 tests require local originals), archived local attestation 1180/1180 PASS.
- Tools: `tools/acquire10.py`, `tools/hostgate10.py`, `tools/hostmem10.py`, `tools/hostmonitor10.py`, `tools/protocol10.py`, `tools/inventory10.py`.
- Frozen spec `results/increment10/protocol.json`; amendment chain in `results/increment10/amendments/`; latest test inventory `results/increment10/test-inventory-a002.json`.
- Runtime files `src/` and `inwime_runtime/` were intentionally unchanged by preparation; do not modify as part of download fix.
- Source documentation: `docs/PREPARATION_INCREMENT_10.md`, `docs/PROTOCOL_INCREMENT_10.md`, `docs/AMENDMENT_INCREMENT_10_001.md`, `docs/AMENDMENT_INCREMENT_10_002.md`.

## 3. Concrete failed attempt (19 Nov 2026 conversation)

The user ran, on WSL Ubuntu in `~/GIT/francescodifilippo/ooc-inference`:

```sh
python3 -m tools.acquire10 checkpoints/qwen2.5-14b-97e1e76 \
  --backing-volume /mnt/c --execute \
  --authorization "$RUN_DIR/authorization.json" \
  --host-report "$RUN_DIR/download.json" \
  --host-baseline "$RUN_DIR/baseline.json" \
  --evidence "$RUN_DIR/acquisition.json"
```

After explicit `AUTORIZZO`, the download stopped with:

```
acquire10: refused: host stop condition: host_available_bytes=267665408
violates >= 300000000 (pause-and-annotate); .part retained,
resume only after a new hostgate10 GO
```

The immediate cause is **memory-monitor policy**, not a reported checksum, disk-capacity or HTTP transfer failure.

Before the attempt, a sampled host report was GO (Windows AvailableBytes ~4.50 GB; page-ins ~13.76/s, backing C free ~164.44 GB). **Caution:** the pasted JSON's `report_utc` is `2026-11-18T22:20:12Z`, while the message describing the failed attempt was sent 19 Nov; do not treat that report as current or assume whether the pasted timing was from the same execution without checking evidence.

The stored local log location reported by the user:
`/tmp/inwime-m10-download.KAX8RS/acquisition.json` (ephemeral /tmp; may disappear).
Checkpoint directory:
`checkpoints/qwen2.5-14b-97e1e76/` on WSL ext4, physical backing volume `/mnt/c`.

**Transferred byte count has NOT been established**. Existing `.part` and `.part.acquire10.json` ownership markers must be preserved; original `acquire10` is designed to hash and resume a marked partial with exact HTTP Range, so do not start over or rename them blindly.

Safe initial read-only local inventory:

```sh
du -sh checkpoints/qwen2.5-14b-97e1e76 2>/dev/null || true
find checkpoints/qwen2.5-14b-97e1e76 -maxdepth 1 -type f \
  \( -name '*.part' -o -name '*.safetensors' -o -name '*.json' \) \
  -printf '%f\t%s bytes\n' 2>/dev/null
test -f /tmp/inwime-m10-download.KAX8RS/acquisition.json && \
  python3 -m json.tool /tmp/inwime-m10-download.KAX8RS/acquisition.json | head -100
```

Avoid cleanup commands `--discard-rejected` or `--clean-temporaries` until ownership and intent are audited.

## 4. Narrow implementation requested (issue #25)

1. Use the existing pinned `QWEN25_14B` manifest/revision `97e1e76335b7017d8f67c08a19d103c0504298c9` (about **29.55 GB**) and current `acquire10` streaming/transfer/resume logic. **Remove the acquisition-time mandatory `hostgate10` GO and `HostMonitor` RAM aborts**; any host RAM data collected at acquisition time is advisory.
2. Preserve **HTTPS allowlist / redirect handling, manifest integrity and hash verification, exact-range resume, no overwrite, owner marker safety, lock, cancellation handling, conservative physical disk free-space stop, evidence/provenance**. Keep an independent explicit user authorization if desired; the user already authorized the interrupted attempt but a later execution should still be intentional.
3. Tests must demonstrate that a synthetic low-memory host does **not** stop acquisition, and that an interrupted real-format `.part` resumes without duplication, while disk/HTTP/hash errors still stop correctly. Do not force another full 29.55 GB download to test.
4. Maintain version/evidence integrity: preregistered protocol and previous amendments must remain unchanged; record the revised acquisition-only policy as an explicit **amendment 003 / correction decision**, update effective tool digests + inventory, revise tests that encoded the mistaken policy. Preserve historical evidence unchanged.
5. Run the full local suite and GitHub CI (the latter may legitimately skip 11 tests needing M9 originals). Keep PR #24 draft until ready; no merge.
6. Only after code review and a read-only inspection of existing `.part`, instruct the user how to **resume the acquisition** and validate SHA256 against the pinned manifest. Do not make extra memory tests a new prerequisite for a simple network download.

## 5. What remains after the download

M10's *real* scientific/technical work still remains: Qwen2.5-14B BF16 full checkpoint integrity; full independent 7B end-to-end oracle equivalence; 14B inference correctness, bounded RSS/cgroup/no swap, whole-operation performance and I/O attribution, preregistered gates A–H. Do **not** conflate download completion with M10 inference completion.

## 6. Human instructions for the next agent

Be concise. The user is frustrated by previous needless RAM gates and does not want further architectural overengineering. Prioritize **recovering/reusing the actual partial download** and **patching the acquisition path minimally**, with clear honesty about work not yet done. The new discussion should start with the status of issue #25 and PR #24, then execute that patch. Do not claim the issue is fixed or the checkpoint downloaded until proven by tests and SHA verification.
