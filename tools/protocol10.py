"""Build and check the preregistered increment-10 protocol (results/increment10/protocol.json).

build  Write protocol.json once ('x' mode): decisions (thresholds, limits, runs,
       verdict rules) are stated here; every derived number (sizes, active bytes,
       space, VmPeak estimates, coverage counts) is recomputed from the captured
       official metadata and from versioned increment-9 evidence, never typed in.
       Tool and inventory SHA256 are pinned.
check  Recompute everything and compare with the frozen file; verify the pinned
       tool/inventory digests, the protocol digest file and the amendment chain.
amend  Append the next amendment (results/increment10/amendments/NNN-<slug>.json,
       'x' mode) from a spec: 'from' digests are taken from the effective chain,
       'to' digests from the files on disk; frozen files are never edited. Only the
       sections in AMENDABLE may change (no threshold, criterion or verdict rule).

Classes used for every quantity: capacity (installed/configured size),
observation (measured state, with time), estimate (derived or assumed).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import acquire10

ROOT = Path(__file__).resolve().parents[1]
R10 = ROOT/'results'/'increment10'
METADATA = R10/'preflight-metadata'
MiB, GiB = 2**20, 2**30
BASE_COMMIT = '639de22ed7e30a796b831595cd0f9823d585de16'
PROMPT_IDS = [785, 6722, 315, 9625, 374]

PINNED_TOOLS = ['tools/acquire10.py', 'tools/hostmem10.py', 'tools/oracle10.py', 'tools/oracle10_plan.py',
                'tools/compare10.py', 'tools/equivalence10.py', 'tools/check_oracle10.py', 'tools/inventory10.py',
                'tools/fixture_qwen10.py', 'tools/protocol10.py',
                # increment-9/7 tools reused unchanged by the campaign
                'tools/cgroup9.py', 'tools/cgroup9_finalize.sh', 'tools/benchmark_increment9.py',
                'tools/oracle_precision9.py', 'tools/oracle_precision7.py', 'tools/integrity9.py',
                'src/cpu_kernel.c']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def headers():
    out = {}
    for path in sorted(METADATA.glob('header-*.safetensors.json')):
        shard = path.name.removeprefix('header-').removesuffix('.json')
        h = json.loads(path.read_bytes())
        h.pop('__metadata__', None)
        for name, d in h.items():
            out[name] = dict(shape=d['shape'], nbytes=d['data_offsets'][1]-d['data_offsets'][0], shard=shard)
    return out


def derived():
    """Every derived number, from the captured metadata and the increment-9 evidence."""
    t = headers()
    config = json.loads((METADATA/'file-config.json').read_text())
    width, layers, vocab = config['hidden_size'], config['num_hidden_layers'], config['vocab_size']
    heads, kvh, inter = config['num_attention_heads'], config['num_key_value_heads'], config['intermediate_size']
    kvw = kvh*width//heads
    payload = sum(d['nbytes'] for d in t.values())
    embedding = t['model.embed_tokens.weight']['nbytes']
    row = width*2
    per_token_logical = payload - embedding + row
    gemv = sum(d['nbytes']//2 for n, d in t.items() if len(d['shape']) == 2 and n != 'model.embed_tokens.weight')
    m = acquire10.QWEN25_14B
    sync, pipe = (json.loads((ROOT/'results'/'increment9'/f).read_text())
                  for f in ('phaseB-sync-runtime.json', 'phaseB-pipeline-runtime.json'))
    plan14 = dict(sync=80_543_216, pipeline=83_189_502)   # observation: unchanged Qwen2Runtime.minimum_plan
    est_sync = sync['process_after_startup']['VmPeak_bytes'] + plan14['sync'] - sync['arena_plan_bytes']
    est_pipe = pipe['process_after_startup']['VmPeak_bytes'] + plan14['pipeline'] - pipe['arena_plan_bytes']
    T = 8
    return dict(
        config=config, tensors=len(t), payload=payload, embedding=embedding, row=row, kv_width=kvw,
        active_without_embedding=payload-embedding,
        active_5_rows=payload-embedding+5*row, active_8_rows=payload-embedding+8*row,
        logical_bytes_per_token=per_token_logical, logical_bytes_8_positions=T*per_token_logical,
        gemv_coefficients_per_token=gemv, gemv_coefficients_8_positions=T*gemv,
        gemv_matrices_per_token=sum(1 for n, d in t.items() if len(d['shape']) == 2 and n != 'model.embed_tokens.weight'),
        parameters=payload//2, shard_bytes=sum(v[0] for v in m.shards.values()), download=m.total_bytes(),
        manifest=m.digest(), required_free=m.total_bytes()+acquire10.Reserves().held(),
        coverage=dict(logits=T*vocab, residuals=layers*T*width, kv_final=layers*2*T*kvw,
                      kv_per_phase_runtime=layers*2*(5+6+7+8)*kvw),
        layers_split=sorted({int(n.split('.')[2]) for n in t if n.startswith('model.layers.')
                             and len({d['shard'] for k, d in t.items() if k.startswith('.'.join(n.split('.')[:3])+'.')}) > 1}),
        plan14=plan14, m9=dict(sync_vmpeak=sync['process_after_startup']['VmPeak_bytes'],
                               pipeline_vmpeak=pipe['process_after_startup']['VmPeak_bytes'],
                               sync_plan=sync['arena_plan_bytes'], pipeline_plan=pipe['arena_plan_bytes']),
        est_sync=est_sync, est_pipe=est_pipe, inter=inter)


def criterion(quantity, op, threshold, required=True, **extra):
    return dict(quantity=quantity, op=op, threshold=threshold, required=required, **extra)


def build(inventory_path):
    d = derived()
    c = d['config']
    host_ram, dimm, memtotal = 16_900_739_072, 17_179_869_184, 11_483_848_704
    inventory = dict(path=str(inventory_path.relative_to(ROOT)), sha256=sha256(inventory_path),
                     baseline_node_ids_at_base=json.loads(inventory_path.read_text())['expected_counts']['baseline-m1-m9'],
                     total_node_ids=json.loads(inventory_path.read_text())['expected_counts']['total'])
    return {
        'schema': 'inwime-protocol-m10/1',
        'protocol_version': 1,
        'increment': 10,
        'status': 'preregistered before any increment-10 download, run or oracle on the 14B checkpoint',
        'base_commit': BASE_COMMIT,
        'separate_from': 'results/increment9/protocol.json (unchanged)',
        'classification': {
            'capacity': 'installed or configured size (RAM, DIMM, MemTotal, volume size)',
            'observation': 'measured state at a stated time; may change',
            'estimate': 'derived by declared arithmetic or assumed; never reported as a measurement'},
        'amendment_policy': {
            'rule': 'protocol.json and the test inventory are never edited after freezing; changes are new '
                    'files results/increment10/amendments/NNN-<slug>.json with reason, author, UTC time and '
                    'the SHA256 of protocol.json and of the previous amendment',
            'forbidden': ['widening a numerical threshold after any measurement',
                          'removing or renaming a required test', 'reclassifying a FAIL'],
            'protocol_sha256_file': 'results/increment10/protocol.sha256'},
        'checkpoint': {
            'model': acquire10.QWEN25_14B.model, 'variant': 'base (not Instruct: Instruct uses rms_norm_eps 1e-6)',
            'revision': acquire10.QWEN25_14B.revision, 'license': 'apache-2.0', 'gated': False,
            'acquisition_tool': 'tools/acquire10.py', 'acquisition_manifest_sha256': d['manifest'],
            'shards': {n: dict(nbytes=v[0], lfs_sha256=v[1], header_sha256=v[4])
                       for n, v in sorted(acquire10.QWEN25_14B.shards.items())},
            'shard_bytes': d['shard_bytes'], 'download_total_bytes': d['download'],
            'payload_bytes': d['payload'], 'tensor_count': d['tensors'], 'dtypes': ['BF16'],
            'parameters': d['parameters'],
            'layers_split_across_shards': d['layers_split'],
            'tokenizer': 'identical Git blobs to the verified local Qwen2.5-7B tokenizer (preflight 2.1)',
            'destination': 'checkpoints/qwen2.5-14b-97e1e76 (internal ext4 on the WSL VHDX, physical volume C:)',
            'integrity': 'SHA256 computed while streaming (acquire10) == official LFS digest; runtime full SHA at '
                         'start of R1-R3; final re-hash with tools/integrity9.py',
            'class': 'observation (official metadata captured 2026-11-18, results/increment10/preflight-metadata)'},
        'model_geometry': {
            k: c[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                              'num_key_value_heads', 'rms_norm_eps', 'rope_theta', 'vocab_size',
                              'tie_word_embeddings', 'use_sliding_window')} | {
            'gqa_ratio': c['num_attention_heads']//c['num_key_value_heads'], 'head_dim': c['hidden_size']//c['num_attention_heads'],
            'new_vs_7b': ['GQA 5:1', 'rms_norm_eps 1e-5', '48 layers'],
            'synthetic_coverage': 'tools/fixture_qwen10.py, tests/test_increment10_fixture.py'},
        'workload': {
            'prompt_text': 'The capital of France is', 'prompt_token_ids': PROMPT_IDS,
            'prefill_tokens': 5, 'decodes': 3,
            'decode_rule': 'greedy token from the runtime logits, fed back even if it is EOS (151643)',
            'chunks': [5, 1, 1, 1], 'capacity': 16, 'tile_elements': 262144, 'integrity': 'full',
            'kernel': 'c-fp64-ordered (build/libinwime_cpu.so from src/cpu_kernel.c)',
            'runtime_change': 'none: inwime_runtime/ and src/ must equal base_commit'},
        'active_weights': {
            'class': 'estimate (derived from the official headers; recomputed by tools/protocol10.py)',
            'unique_bytes_without_embedding': d['active_without_embedding'],
            'embedding_row_bytes': d['row'],
            'unique_bytes_5_distinct_rows': d['active_5_rows'], 'unique_bytes_8_distinct_rows': d['active_8_rows'],
            'logical_bytes_per_token': d['logical_bytes_per_token'],
            'logical_bytes_8_positions': d['logical_bytes_8_positions'],
            'gemv_matrices_per_token': d['gemv_matrices_per_token'],
            'gemv_coefficients_per_token': d['gemv_coefficients_per_token'],
            'gemv_coefficients_8_positions': d['gemv_coefficients_8_positions'],
            'versus_host_ram': dict(total_physical_memory=host_ram, installed_dimm=dimm, ratio_lower_bound=round(d['active_5_rows']/host_ram, 4)),
            'versus_wsl_memtotal': dict(memtotal=memtotal, ratio_lower_bound=round(d['active_5_rows']/memtotal, 4)),
            'not_claimed': 'larger than host RAM + pagefile (51.7 GB): irrelevant, weights are file pages'},
        'memory': {
            'host_total_physical_memory': dict(value=host_ram, class_='capacity', source='Win32_ComputerSystem, 2026-11-18'),
            'host_installed_dimm_bytes': dict(value=dimm, class_='capacity', source='Win32_PhysicalMemory, 2026-11-18'),
            'wsl_memtotal': dict(value=memtotal, class_='capacity', source='.wslconfig memory=11GB (read only, unchanged)'),
            'wsl_swap': dict(value=8*GiB, class_='capacity', source='.wslconfig swap=8GB (unchanged)'),
            'host_available_at_rest': dict(value=[648_921_088, 768_126_976], class_='observation',
                                           source='Win32_PerfRawData_PerfOS_Memory, 2026-11-18 07:00-08:40 CET'),
            'host_pagefile_in_use': dict(value_mib=[7002, 8639], class_='observation',
                                         source='Win32_PageFileUsage, 2026-11-18 (8639 after the 7B component checks)'),
            'no_changes': ['no .wslconfig, swap, pagefile, sysctl, drop_caches or Windows setting is modified']},
        'memory_limits': {
            'scope': dict(mechanism='systemd-run --user --scope (transient, delegated user manager)',
                          memory_max_bytes=2*GiB, swap_max_bytes=0, oom_policy='continue'),
            'plan_bytes': dict(d['plan14'], class_='observation: unchanged Qwen2Runtime.minimum_plan on sparse '
                               'copies of the official headers (preflight 3.1; re-run by test_increment10_fixture)'),
            'rlimit_as_bytes': dict(sync=256*MiB, pipeline=384*MiB),
            'anon_plus_kernel_max_bytes': 256*MiB, 'vmhwm_max_bytes': 256*MiB,
            'derivation': {
                'm9_7b_sync_vmpeak': d['m9']['sync_vmpeak'], 'm9_7b_pipeline_vmpeak': d['m9']['pipeline_vmpeak'],
                'm9_7b_sync_plan': d['m9']['sync_plan'], 'm9_7b_pipeline_plan': d['m9']['pipeline_plan'],
                'estimated_14b_sync_vmpeak': d['est_sync'], 'estimated_14b_pipeline_vmpeak': d['est_pipe'],
                'rule': 'VmPeak(14B) = VmPeak(7B, M9) + plan(14B) - plan(7B); same code path, so the remaining '
                        'address space (interpreter, ctypes kernel, thread stack and arena) is unchanged',
                'margin_sync_at_256MiB': 256*MiB-d['est_sync'], 'margin_pipeline_at_256MiB': 256*MiB-d['est_pipe'],
                'margin_pipeline_at_384MiB': 384*MiB-d['est_pipe'],
                'glibc_arena_alignment_transient_bytes': 64*MiB,
                'estimated_14b_pipeline_vmpeak_worst_case': d['est_pipe']+64*MiB,
                'margin_pipeline_worst_case_at_384MiB': 384*MiB-d['est_pipe']-64*MiB,
                'transient_observation': 'the I/O thread arena is created by mapping 2 x 64 MiB and trimming to an '
                                         'aligned 64 MiB: VmPeak may include a transient 64 MiB that VmSize never keeps. '
                                         'Structural pipeline overhead measured 72.9 MB on the 7B (M9) and 136 MiB on '
                                         'the fixture (2026-11-18, tests/test_increment10_limits.py)',
                'why_384': 'at 256 MiB the pipeline keeps < 14 MB of address space, and the worst-case arena '
                           'transient does not fit: a spurious MemoryError or a degraded arena, not a numerical '
                           'error, would decide the run. 384 MiB keeps > 140 MiB (> 77 MiB in the worst case). '
                           'RLIMIT_AS is a safety net, not the memory measurement: anon+kernel and VmHWM stay '
                           'bounded at 256 MiB and the 2 GiB scope is the real limit',
                'synthetic_tests': ['tests/test_increment10_limits.py::test_derivation_recomputed_from_m9_evidence',
                                    'tests/test_increment10_limits.py::test_pipeline_overhead_is_structural_and_runs_under_limits'],
                'class': 'estimate'}},
        'numerical_contract': {
            'thresholds': {'fp32': dict(atol=2e-3, rtol=2e-4), 'fp64': dict(atol=2e-5, rtol=2e-6),
                           'rule': 'abs(err) <= atol + rtol*abs(ref) for every element; never widened'},
            'contracts': {'stock_fp32': 'gating: upstream Transformers in FP32',
                          'full64': 'gating: upstream graph with RMSNorm, RoPE and softmax in FP64 (profile 7/9)',
                          'stock_fp64': 'diagnostic: reported raw; an expected FAIL is never rewritten'},
            'coverage_full': {
                'logits': d['coverage']['logits'], 'residuals': d['coverage']['residuals'],
                'kv_final': d['coverage']['kv_final'],
                'kv_per_phase_runtime': d['coverage']['kv_per_phase_runtime'],
                'also': 'argmax and top-10 of every position; upstream greedy equals the runtime continuation',
                'kv_rule': 'the oracle K/V after the last call are compared element-wise; the runtime per-phase '
                           'K/V (lengths 5,6,7,8) must be byte-identical prefixes of the final K/V'},
            'coverage_classes': {'FULL': 'every element above compared', 'PARTIAL': 'subset or missing kind',
                                 'STRUCTURAL': 'shapes, digests, argmax only', 'NOT PROVEN': 'nothing usable',
                                 'rule': 'F is PASS only with FULL coverage on both gating contracts (tools/compare10.py)'},
            'fp32_risk': 'if stock_fp32 fails while full64 passes, F is FAIL; per-layer threshold utilization reported',
            'oracle10': {
                'tool': 'tools/oracle10.py', 'independence': 'no InWiMe import (static test); only token IDs from the runtime',
                'versions': dict(torch='2.9.1+cpu', transformers='4.57.3', safetensors='0.7.0', numpy='2.3.5'),
                'contracts': {'stock_fp32': dict(dtype='float32', head_mode='resident', block_rows=16384),
                              'full64': dict(dtype='float64', head_mode='blocked', block_rows=16384),
                              'stock_fp64': dict(dtype='float64', head_mode='blocked', block_rows=16384)},
                'memory_budget_bytes': 5*GiB, 'scope_memory_max_bytes': 6*GiB, 'scope_swap_max_bytes': 0,
                'guest_mem_available_min_bytes': int(6.5*GiB),
                'head_policy': 'blocked F.linear is not bit-identical to a full one in general (BLAS tail kernels; '
                               'fixture: FP32 1.9e-6, FP64 4.4e-15): stock_fp32 keeps the full FP32 head (3.11 GB), '
                               'bit-identical by construction; the FP64 head is never materialized',
                'equivalence': {
                    'fixtures': 'bit identity with stock from_pretrained (residuals, K/V, FP32 logits); FP64 logits '
                                'max |delta| <= 1e-9 with identical argmax (tools/check_oracle10.py)',
                    '7b_components': 'embedding gather and final norm + head on the M9 captures (tools/equivalence10.py): '
                                     'executed 2026-11-18, PARTIAL coverage, see oracle10-7b-equivalence/',
                    '7b_end_to_end_prerequisite': 'before any 14B oracle: oracle10 on Qwen2.5-7B with the M9 sequence '
                                                  'and chunks against artifacts/increment9/oracle-*.npz. stock_fp32: '
                                                  'logits, residuals and K/V bit-identical; full64/stock_fp64: residuals '
                                                  'and K/V bit-identical, logits max |delta| <= 1e-9, identical argmax. '
                                                  'Failure makes the corresponding 14B contract NOT PROVEN',
                    'cost_estimate': 'about 7 min and 60 GB of reads per contract (M9 oracle wall time); estimate'}}},
        'fault_injection': [
            dict(id='N1', fault='I/O error', modes=['sync', 'pipeline'], target='14B real, integrity=metadata, RLIMIT_AS per mode'),
            dict(id='N2', fault='short read', modes=['sync', 'pipeline'], target='14B real'),
            dict(id='N3', fault='cancellation', modes=['sync', 'pipeline'], target='14B real'),
            dict(id='N4', fault='budget reduced by 1 byte', target='14B real: refused before payload'),
            dict(id='N5', fault='wrong SHA256 / file changed after open', target='fixture'),
            dict(id='N6', fault='OOM in a 32 MiB scope', target='SmolLM2-1.7B (local)'),
            dict(id='N7', fault='worker outside the scope', target='supervisor check'),
            dict(id='N8', fault='mutations of attribution (W as C) and of criterion D (7B bytes, missing RAM, MemTotal as RAM)', target='gate'),
            dict(id='N9', fault='perturbed weight / wrong epsilon', target='oracle sensitivity (check_oracle10)'),
            dict(id='N10', fault='space guard, interrupted transfer, ignored Range, wrong digest', target='acquire10 tests')],
        'rollback': {
            'repository': 'dedicated branch, additive files only; revert or abandon; no force push',
            'download': '.part retained for resume or removed only by acquire10 --clean-temporaries (owned files)',
            'checkpoint': 'deletion (29.55 GB) is the owner\'s decision; C: measured after any deletion (sparse VHDX reclaim not proven)',
            'runs': 'transient scopes with --collect; no residual units/processes; targeted eviction of experiment files only',
            'evidence': 'sanitized at write time; a FAIL is never fixed by changing thresholds or runtime inside M10'},
        'io_protocol': {
            'runs': [dict(id='R1', cache='C', mode='sync', captures='full', purpose='primary'),
                     dict(id='R2', cache='C', mode='pipeline', captures='digest', purpose='byte identity with R1'),
                     dict(id='R3', cache='C', mode='sync', captures='digest', purpose='repetition'),
                     dict(id='R4', cache='W', mode='sync', captures='digest', purpose='informative only')],
            'cold_definition': 'C = cold guest: experiment files evicted (POSIX_FADV_DONTNEED) and residency <= 1 % '
                               'per shard by fincore before start. Not cold NAND: Windows, controller and SSD caches '
                               'are not controlled; L4 shows whether the host could have served the reads',
            'warm_definition': 'W = as-is immediately after a C run; informative, never used for C/D',
            'counters': {'L0': 'logical bytes per tensor (runtime)', 'L1': '/proc/PID/io read_bytes',
                         'L2': 'scope memory.stat, events, PSI', 'L3': '/sys/block/<dev>/stat of the VHDX disk',
                         'L4': 'Windows physical disk raw read counter (global)',
                         'L4b': 'host AvailableBytes, PagesInput/Output, pagefile usage (tools/hostmem10.py)',
                         'io.stat': 'not available: io controller not delegated (declared)'},
            'attribution_rules': ['L1 >= 0.90 * L0 on inference and every decode of a C run',
                                  'delta L3 >= L1', 'delta L4 >= 0.90 * L1 when readable (global counter: never equality)',
                                  'host PagesInput sustained > 2048 pages/s during a run: I/O attribution of that run is '
                                  'invalid (condition C NOT PROVEN for it); numerical results kept'],
            'records': ['UTC times', 'AC power and power plan', 'process list', 'temperatures NOT MEASURED unless '
                        'readable without elevation']},
        'host_memory_preflight': {
            'tool': 'tools/hostmem10.py (read only)', 'interval_s': 30,
            'baseline': 'a hostmem10 report taken at the start of the campaign session, before the first run; '
                        'every later start check passes it as --baseline',
            'start_criteria': {
                'download': [
                    criterion('host_available_bytes', '>=', 512*MiB),
                    criterion('host_commit_headroom_bytes', '>=', 2*GiB),
                    criterion('host_pages_input_per_s', '<=', 1024),
                    criterion('host_pagefile_growth_since_baseline', '<=', GiB),
                    criterion('guest_mem_available', '>=', 2*GiB),
                    criterion('guest_swap_growth_since_baseline', '<=', 256*MiB),
                    criterion('backing_volume_free', '>=', d['required_free'])],
                'runtime': [
                    criterion('host_available_bytes', '>=', GiB),
                    criterion('host_commit_headroom_bytes', '>=', 3*GiB),
                    criterion('host_pages_input_per_s', '<=', 256),
                    criterion('host_pages_output_per_s', '<=', 256),
                    criterion('host_pagefile_growth_since_baseline', '<=', 512*MiB),
                    criterion('guest_mem_available', '>=', 3*GiB),
                    criterion('guest_swap_growth_since_baseline', '<=', 256*MiB),
                    criterion('guest_psi_memory_some_avg10', '<=', 1.0),
                    criterion('backing_volume_free', '>=', 40*GiB)],
                'oracle': [
                    criterion('host_available_bytes', '>=', GiB),
                    criterion('host_commit_headroom_bytes', '>=', 7*GiB),
                    criterion('host_pages_input_per_s', '<=', 256),
                    criterion('host_pagefile_growth_since_baseline', '<=', 512*MiB),
                    criterion('guest_mem_available', '>=', int(6.5*GiB)),
                    criterion('guest_swap_growth_since_baseline', '<=', 256*MiB),
                    criterion('guest_psi_memory_some_avg10', '<=', 1.0),
                    criterion('backing_volume_free', '>=', 40*GiB)]},
            'abort_criteria': [
                criterion('host_available_bytes', '>=', 300_000_000, sustain_s=10, action='pause-and-annotate'),
                criterion('guest_mem_available', '>=', 768*MiB, action='kill-scope-invalidate-run'),
                criterion('host_pagefile_growth_since_baseline', '<=', 4*GiB, action='pause-and-annotate'),
                criterion('host_commit_headroom_bytes', '>=', GiB, action='kill-scope-invalidate-run'),
                criterion('host_pages_input_per_s', '<=', 2048, sustain_s=30, action='annotate-io-attribution-invalid'),
                criterion('backing_volume_free', '>=', 40*GiB, action='stop-acquisition-retain-part')],
            'verdicts': {'GO': 'every start criterion measured and satisfied',
                         'DEFER': 'a measured value violates a start criterion: postpone, record blocked-host-memory-*.json',
                         'NOT MEASURED': 'a required value is unreadable: do not start (BLOCKED with cause)'},
            'observation_2026_11_18': 'host AvailableBytes 0.65-1.03 GB at rest: runtime and oracle profiles would be '
                                      'DEFER; the 7B component checks (peak RSS 3.8 GB in a 4.5 GiB scope) raised the '
                                      'pagefile from 7089 to 8639 MiB. A first hostmem10 revision launched one '
                                      'powershell.exe per sample and counted its start-up page-ins in the interval '
                                      '(8456 pages/s observed); the frozen revision takes both samples in one process '
                                      '(superseded reports kept in preflight-tooling/superseded-before-refreeze/)',
            'sampling': 'one powershell.exe process takes both samples (Start-Sleep inside), so its start-up is not '
                        'counted in the paging rates; guest values are read just before and just after it',
            'class': 'thresholds are policy decisions (estimates), not measured capacities'},
        'space': {
            'class': 'observation (C: free) + estimate (reserves)',
            'required_free_before_download': d['required_free'],
            'reserves': acquire10.Reserves().as_dict(),
            'stop_threshold_during_download': 40*GiB,
            'note': 'the ext4 VHDX free space (892 GB apparent) is never used for decisions; C: is the physical limit'},
        'verdicts': {
            'PASS': 'every required fact present, typed and recomputed from raw data',
            'FAIL': 'a fact contradicts the criterion, or invalid JSON / wrong field type',
            'NOT PROVEN': 'evidence or field missing, or partial coverage, after the attempt',
            'BLOCKED': 'not attempted because of an external precondition recorded in blocked-*.json with cause and UTC '
                       'time: download not authorized, space below threshold, host DEFER/NOT MEASURED, mount missing',
            'precedence': ['FAIL', 'BLOCKED', 'NOT PROVEN', 'PASS'],
            'mixed_rule': 'a missing field and a contradicted field in the same check give FAIL (Issue #22 residual)',
            'blocked_without_cause_file': 'NOT PROVEN'},
        'gate_conditions': {
            'A': 'integrity: acquisition SHA, runtime start SHA (R1-R3) and final re-hash equal the pinned LFS digests',
            'B': 'memory: 2 GiB scope, swap 0, memory.peak <= limit, oom = oom_kill = 0, anon+kernel <= 256 MiB, '
                 'VmSwap 0, RLIMIT_AS per mode recorded',
            'C': 'I/O attribution rules on R1-R3; residency <= 1 % after eviction',
            'D': 'unique bytes from counters > max(TotalPhysicalMemory, DIMM) measured in the same run, and > MemTotal, '
                 'equal to the header recomputation, on a C run that really read',
            'E': 'prefill 5 + 3 decodes, 48 layers, 579 tensor counters = shape x 8 (embedding rows exact), '
                 'gemv coefficients for 8 positions, KV 5/6/7/8, sync/pipeline digests identical',
            'F': 'stock_fp32 and full64 PASS on FULL coverage, oracle10 7B end-to-end equivalence PASS, stock_fp64 raw',
            'G': 'every run supervised, no unattributed OOM, guest swap growth <= 256 MiB, host recorded (L4b)',
            'H': 'pinned inventory verified on the archived JUnit (tools/inventory10.py verify), gates 7/8/9 PASS, '
                 'inwime_runtime/ and src/ unchanged since base_commit, no pre-existing file modified'},
        'stop_conditions': [
            'decode above 80 s/token: pause, investigate thermals and I/O',
            'guest MemAvailable < 768 MiB: cgroup.kill of the scope only; run invalidated and kept',
            'host AvailableBytes < 300 MB for more than 10 s: pause, annotated',
            'host pagefile growth > 4 GiB since baseline: pause, annotated',
            'unplanned oom_kill: FAIL recorded',
            'C: free < 40 GiB during acquisition: stop, .part retained',
            'no repetition until pass: every run, red or pilot, is kept'],
        'test_inventory': inventory,
        'pinned_tools_sha256': {p: sha256(ROOT/p) for p in PINNED_TOOLS},
    }


AMENDMENT_SCHEMA = 'inwime-protocol-m10-amendment/1'
# What an amendment may change. Start/abort criteria, numerical thresholds, workload, verdict rules
# and the frozen inventory are rebuilt from code and compared with the frozen protocol: they cannot move.
AMENDABLE = {'pinned_tools_sha256', 'test_inventory', 'acquisition_host_gate', 'ci_runner_verification',
             'acquisition_host_monitor', 'acquisition_host_policy'}
# Amendment 003 (Issue #25): the acquisition is not gated by host memory. The section must say so
# explicitly, name the issue and list what it supersedes; it can never re-enable a blocking host gate
# for the acquisition without a further amendment that replaces this one.
HOST_POLICY_KEYS = {'gated', 'issue', 'rule', 'supersedes', 'acquisition_stop_conditions', 'diagnostics',
                    'host_tools_kept_for'}
# Acquisition meaning of the frozen abort actions (amendment 002): only the I/O-attribution criterion
# may stay an annotation; every pause/kill/stop action must stop the transfer.
ANNOTATE_ONLY_ACTIONS = {'annotate-io-attribution-invalid'}


def amendment_files(directory=R10/'amendments'):
    return sorted(Path(directory).glob('*.json')) if Path(directory).exists() else []


def node_ids(inventory_path):
    return {n for nodes in json.loads(Path(inventory_path).read_text())['groups'].values() for n in nodes}


def effective(protocol_path=R10/'protocol.json', directory=None):
    """Protocol values after applying the amendment chain (no validation: see check())."""
    p = json.loads(Path(protocol_path).read_bytes())
    directory = Path(protocol_path).parent/'amendments' if directory is None else directory
    out = dict(pinned_tools_sha256=dict(p['pinned_tools_sha256']),
               test_inventory=dict(path=p['test_inventory']['path'], sha256=p['test_inventory']['sha256']),
               acquisition_host_gate=None, ci_runner_verification=None, acquisition_host_monitor=None,
               acquisition_host_policy=None, amendments=[])
    for path in amendment_files(directory):
        changes = json.loads(path.read_text()).get('changes') or {}
        for tool, change in (changes.get('pinned_tools_sha256') or {}).items():
            out['pinned_tools_sha256'][tool] = change['to']
        if 'test_inventory' in changes:
            out['test_inventory'] = dict(changes['test_inventory']['to'])
        for key in ('acquisition_host_gate', 'ci_runner_verification', 'acquisition_host_monitor',
                    'acquisition_host_policy'):
            if key in changes:
                out[key] = changes[key]
        out['amendments'].append(path.name)
    return out


def check_amendments(raw, p, directory, root=ROOT):
    """Validate the chain: digests, numbering, allowed keys, 'from' values, superset inventory."""
    problems = []
    protocol_digest = hashlib.sha256(raw).hexdigest()
    previous = protocol_digest
    tools = dict(p['pinned_tools_sha256'])
    inventory = dict(path=p['test_inventory']['path'], sha256=p['test_inventory']['sha256'])
    for index, path in enumerate(amendment_files(directory), 1):
        try:
            a = json.loads(path.read_text())
        except ValueError:
            problems.append('amendment is not valid JSON: '+path.name)
            break
        if a.get('protocol_sha256') != protocol_digest or a.get('previous_sha256') != previous:
            problems.append('broken amendment chain: '+path.name)
        if a.get('schema') != AMENDMENT_SCHEMA or a.get('number') != index or not path.name.startswith(f'{index:03d}-'):
            problems.append('amendment schema or numbering wrong: '+path.name)
        for k in ('author', 'utc', 'reason'):
            if not a.get(k):
                problems.append(f'amendment lacks {k}: '+path.name)
        changes = a.get('changes') or {}
        if not changes or set(changes) - AMENDABLE:
            problems.append(f'amendment changes non-amendable sections {sorted(set(changes) - AMENDABLE)}: '+path.name)
        for tool, change in (changes.get('pinned_tools_sha256') or {}).items():
            if not isinstance(change, dict) or change.get('from') != tools.get(tool) or not change.get('to'):
                problems.append(f'amendment {path.name}: pinned digest of {tool} does not start from the effective value')
                continue
            tools[tool] = change['to']
        if 'test_inventory' in changes:
            change = changes['test_inventory']
            new = change.get('to') or {}
            if change.get('from') != inventory:
                problems.append(f'amendment {path.name}: inventory does not start from the effective one')
            elif not (root/str(new.get('path'))).exists() or sha256(root/new['path']) != new.get('sha256'):
                problems.append(f'amendment {path.name}: amended inventory missing or SHA256 differs')
            else:
                old_nodes, new_nodes = node_ids(root/inventory['path']), node_ids(root/new['path'])
                if not old_nodes <= new_nodes:
                    problems.append(f'amendment {path.name}: required tests removed or renamed: '
                                    + ', '.join(sorted(old_nodes - new_nodes)[:5]))
                if new.get('total_node_ids') != len(new_nodes):
                    problems.append(f'amendment {path.name}: total_node_ids differs from the amended inventory')
                inventory = dict(new)
        gate = changes.get('acquisition_host_gate')
        if gate is not None:
            ok = (gate.get('profile') in p['host_memory_preflight']['start_criteria']
                  and all(type(gate.get(k)) is int and gate[k] > 0
                          for k in ('max_report_age_s', 'max_baseline_age_s', 'max_future_skew_s'))
                  and isinstance(gate.get('interval_tolerance'), list) and len(gate['interval_tolerance']) == 2
                  and 0 < gate['interval_tolerance'][0] <= 1 <= gate['interval_tolerance'][1])
            if not ok:
                problems.append(f'amendment {path.name}: acquisition_host_gate malformed')
        monitor = changes.get('acquisition_host_monitor')
        if monitor is not None:
            actions = {c['action'] for c in p['host_memory_preflight']['abort_criteria']}
            mapping = monitor.get('action_mapping') or {}
            ints = ('sample_interval_s', 'max_unmonitored_s', 'startup_timeout_s', 'max_line_bytes',
                    'max_samples_kept', 'max_errors_kept', 'max_annotations')
            ok = (all(type(monitor.get(k)) is int and monitor[k] > 0 for k in ints)
                  and monitor['sample_interval_s'] < monitor['max_unmonitored_s']
                  and set(mapping) == actions
                  and all(v == 'stop' or (v == 'annotate' and a in ANNOTATE_ONLY_ACTIONS) for a, v in mapping.items())
                  and monitor.get('criteria') == 'host_memory_preflight.abort_criteria')
            if not ok:
                problems.append(f'amendment {path.name}: acquisition_host_monitor malformed or weakens an abort action')
        policy = changes.get('acquisition_host_policy')
        if policy is not None:
            ok = (isinstance(policy, dict) and set(policy) == HOST_POLICY_KEYS and policy['gated'] is False
                  and policy['issue'] == 25 and isinstance(policy['supersedes'], list) and policy['supersedes']
                  and isinstance(policy['acquisition_stop_conditions'], list)
                  and {'checksum', 'physical space', 'network'} <= set(policy['acquisition_stop_conditions'])
                  and not any(w in ' '.join(policy['acquisition_stop_conditions']).lower()
                              for w in ('ram', 'paging', 'pagefile', 'memory')))
            if not ok:
                problems.append(f'amendment {path.name}: acquisition_host_policy malformed')
        ci = changes.get('ci_runner_verification')
        if ci is not None:
            unverifiable = ci.get('not_verifiable_on_runner') or {}
            nodes = unverifiable.get('node_ids')
            known = node_ids(root/inventory['path'])
            if not isinstance(nodes, list) or not nodes or not set(nodes) <= known or len(set(nodes)) != len(nodes) \
                    or not unverifiable.get('skip_message'):
                problems.append(f'amendment {path.name}: ci_runner_verification malformed or names unknown tests')
        previous = sha256(path)
    for tool, digest in tools.items():
        if not (root/tool).exists() or sha256(root/tool) != digest:
            problems.append('pinned tool differs from its effective digest: '+tool)
    return problems


def check(protocol_path=R10/'protocol.json'):
    problems = []
    raw = Path(protocol_path).read_bytes()
    p = json.loads(raw)
    digest_file = Path(protocol_path).parent/'protocol.sha256'
    if not digest_file.exists() or digest_file.read_text().split()[0] != hashlib.sha256(raw).hexdigest():
        problems.append('protocol.sha256 missing or different')
    inventory_path = ROOT/p['test_inventory']['path']
    rebuilt = build(inventory_path)
    for key in ('checkpoint', 'model_geometry', 'workload', 'active_weights', 'memory_limits', 'numerical_contract',
                'host_memory_preflight', 'space', 'verdicts', 'gate_conditions', 'test_inventory'):
        if json.loads(json.dumps(rebuilt[key])) != p[key]:
            problems.append('section differs from recomputation: '+key)
    # the frozen pin set is the one build() names; amended digests are verified through the chain
    if sorted(p['pinned_tools_sha256']) != sorted(PINNED_TOOLS):
        problems.append('frozen pinned tool set differs from PINNED_TOOLS')
    directory = Path(protocol_path).parent/'amendments'
    problems += check_amendments(raw, p, directory)
    return dict(passed=not problems, problems=problems, amendments=[x.name for x in amendment_files(directory)])


def amend(spec, protocol_path=R10/'protocol.json', utc=None):
    """Write the next amendment from spec = {slug, author, reason, tools, test_inventory?, sections?,
    frozen_evidence?}. Returns its path."""
    from datetime import datetime, timezone
    raw = Path(protocol_path).read_bytes()
    directory = Path(protocol_path).parent/'amendments'
    existing = amendment_files(directory)
    eff = effective(protocol_path)
    changes = {}
    tools = {}
    for tool in spec.get('tools', []):
        digest = sha256(ROOT/tool)
        if eff['pinned_tools_sha256'].get(tool) != digest:
            tools[tool] = {'from': eff['pinned_tools_sha256'].get(tool), 'to': digest}
    if tools:
        changes['pinned_tools_sha256'] = tools
    if spec.get('test_inventory'):
        path = spec['test_inventory']
        changes['test_inventory'] = {'from': eff['test_inventory'],
                                     'to': dict(path=path, sha256=sha256(ROOT/path), total_node_ids=len(node_ids(ROOT/path)))}
    for key, value in (spec.get('sections') or {}).items():
        changes[key] = value
    if set(changes) - AMENDABLE:
        raise SystemExit(f'non-amendable sections: {sorted(set(changes) - AMENDABLE)}')
    number = len(existing) + 1
    amendment = dict(schema=AMENDMENT_SCHEMA, number=number, slug=spec['slug'],
                     protocol_sha256=hashlib.sha256(raw).hexdigest(),
                     previous_sha256=sha256(existing[-1]) if existing else hashlib.sha256(raw).hexdigest(),
                     author=spec['author'], utc=utc or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                     reason=spec['reason'], invariants=spec.get('invariants', {}),
                     frozen_evidence_preserved={f: sha256(ROOT/f) for f in spec.get('frozen_evidence', [])},
                     changes=changes)
    directory.mkdir(exist_ok=True)
    out = directory/f"{number:03d}-{spec['slug']}.json"
    with out.open('x') as f:
        json.dump(amendment, f, indent=1); f.write('\n')
    return out


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--inventory', type=Path, default=R10/'test-inventory.json')
    b.add_argument('--frozen-utc', required=True)
    sub.add_parser('check')
    m = sub.add_parser('amend')
    m.add_argument('--spec', type=Path, required=True)
    args = a.parse_args(argv)
    if args.command == 'amend':
        out = amend(json.loads(args.spec.read_text()))
        result = check()
        print(json.dumps(dict(amendment=str(out.relative_to(ROOT)), sha256=sha256(out), check=result), indent=1))
        return 0 if result['passed'] else 1
    if args.command == 'build':
        protocol = build(args.inventory.absolute()) | {'frozen_utc': args.frozen_utc}
        out = R10/'protocol.json'
        with out.open('x') as f:
            json.dump(protocol, f, indent=1); f.write('\n')
        with (R10/'protocol.sha256').open('x') as f:
            f.write(f'{sha256(out)}  protocol.json\n')
        print(sha256(out))
        return 0
    result = check()
    print(json.dumps(result, indent=1))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
