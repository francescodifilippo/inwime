"""Increment-9 gate A-H with PASS / FAIL / NOT PROVEN, recomputed from raw evidence.

Fail-closed: a missing file or field is NOT PROVEN, a contradicted fact is FAIL,
never PASS. A present value of the wrong type (a bool, float, string or null where a
non-negative integer counter is required) is a contradicted fact, not a missing one.
Each condition's final verdict is the worst of its own verdict and the preconditions,
in the order FAIL > NOT PROVEN > PASS: a FAIL precondition makes every condition FAIL,
a NOT PROVEN one turns a PASS into NOT PROVEN, and a contradicted fact already
recorded stays FAIL. Gate 9 has no BLOCKED verdict. Derived booleans ('passed', 'all_match', ...) are not trusted: every
verdict is rebuilt from raw counters and bound to the approved protocol, the
official preflight, the supervisor of the same run, the increment-7/8 evidence
and, when present on this machine, the capture files and the unsanitized
originals. Exit code 0 only when every condition is PASS.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import sanitize9

PASS, FAIL, NOT_PROVEN = 'PASS', 'FAIL', 'NOT PROVEN'
SEVERITY = {PASS: 0, NOT_PROVEN: 1, FAIL: 2}
GIB2 = 2147483648
# Approved acceptance values, duplicated here on purpose: editing protocol.json or
# the amendment must not be able to move the bar.
PROTOCOL_SHA256 = 'b7da381cc83b08a712ba6e0f5b8fac4ca2a4f5fdf8f961e8e9ff7a0d630bf053'
AMENDMENT = 'protocol-amendment-remediation.json'
THRESHOLDS = {'fp32_comparison': (2e-3, 2e-4), 'fp64_comparison': (2e-5, 2e-6)}
CONTRACTS = {'stock_fp32': ('float32', 'stock', 'fp32_comparison'),
             'full64': ('float64', 'full64', 'fp64_comparison'),
             'stock_fp64': ('float64', 'stock', 'fp64_comparison')}
GATING, DIAGNOSTIC = ('stock_fp32', 'full64'), 'stock_fp64'
REVISION_A = 'd4c18ccb8a25e6e637e0609aac08a5d1a0436d1b'
SHA_A = '1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0'
REVISION_B = 'd149729398750b98c0af14eb82c78cfe92750796'
BASELINE = '68ba9f3adf3cffb89b18b19525644fa0d7c6d806'
QWEN, SMOLLM = 'qwen2.5-7b-d149729', 'smollm2-1.7b-d4c18cc'
SHARDS = tuple(f'model-0000{i}-of-00004.safetensors' for i in range(1, 5))
INDEX = 'model.safetensors.index.json'
# Post-measurement remediation parameters (tightenings only); the amendment file must
# carry exactly these values.
PARAMETERS = dict(
    anon_kernel_peak_max_bytes=268435456, worker_vmhwm_max_bytes=268435456,
    resident_after_eviction_max_fraction=0.01, phase_b_block_read_min_fraction=0.9,
    minimum_pytest_passed=613, minimum_unittest_ran={'oracle9-checks.txt': 3, 'check-oracle8.txt': 1,
                                                     'check-oracle-precision7.txt': 5},
    intentional_modifications=['README.md'])
QWEN_TENSORS = {'norm1': 'input_layernorm.weight', 'norm2': 'post_attention_layernorm.weight',
                'q': 'self_attn.q_proj.weight', 'q.bias': 'self_attn.q_proj.bias',
                'k': 'self_attn.k_proj.weight', 'k.bias': 'self_attn.k_proj.bias',
                'v': 'self_attn.v_proj.weight', 'v.bias': 'self_attn.v_proj.bias',
                'o': 'self_attn.o_proj.weight', 'ffn.gate': 'mlp.gate_proj.weight',
                'ffn.up': 'mlp.up_proj.weight', 'ffn.down': 'mlp.down_proj.weight'}
# The eleven sanitized evidence files: name -> (original SHA256, sanitized SHA256). Pinned
# here so that removing or rewriting manifest entries cannot shrink what is compared.
SANITIZED = {
    'cgroup-probe-128mib.json': ('87e34ab3ce2491b6e2796b24dd349932289dceb3571a0301989448c99172dd48',
                                'baca2c4717cb579dcd19b1f30b9171f84652096848b2ac8c86b486f93c001752'),
    'environment-after.json': ('fa3d66234f652db3e6b74b2a1a6daf82ada3b68a442dc0a440dc5a74cd23d090',
                              'ca99eb677bc757e9b3340d62fe1e6d35464c1972d0b28ebf5933bfa8422451f4'),
    'environment-before.json': ('c07f92ddac168c15483b434f044a7e8aa19e5abc5d5ef69e9303437f56c6ba84',
                               'e2386c8d00d75d548fc15da007c53bc996196558d2cd1bf2f12fc078de6cd662'),
    'negative-oom-policy-stop-supervisor.json': ('4fec6c79cad9aa1f95296b915b5848c1a87857212b1550e5d8eb5d39ba3e107a',
                                                'fcc17dff21c866fe84482a899f037f564582a16343aea053bca6a97dd82bfdb0'),
    'negative-oom-supervisor.json': ('41e732b129f5e9ddf79985848635c45aaf27ea5b7093233ddcc043294f0a6530',
                                    '6615bef3182d4b55a21cd0aafcf52ef821b830ba0d95ae8241375eb63b650a03'),
    'phaseA-P-supervisor.json': ('b88ae5f31ef12acf5b58938a0df47d8d212a2f30cf2b7a11036ced68586104a1',
                                '92511542247c9f11e2c94d1d652e2c68263c4b9bdd25e21983e1340eddb115e2'),
    'phaseA-W-supervisor.json': ('3cc703a7cabf75b75e91b2a9c8c6d6916b1c0b4d8048d173d2ad42c9f27761fa',
                                '998e9ea091ba1c40314a92ff644f07c5f127eba955b52b48c321abe141f8652f'),
    'phaseB-pipeline-supervisor.json': ('eaab1b286831616cb82af93d59873057ba79c25936b5da235d129bb5fb878ed6',
                                       '72e6bd103e7fa82cba8d6bfe3aec37914c3fa598c078d89c67d0f6655fc5c0e7'),
    'phaseB-sync-supervisor.json': ('147a13927e242e227b461326946a1a964166dfb11723b6089eea87352c910bc8',
                                   '3cebb02837f789c3b7bea69592a49de92d400d9498aea636c6c7b2889a0685e8'),
    'scope48mib-135m-supervisor.json': ('fe4b3252817c49b00391bc97cea47af3fca04766532bbd1528211e05eb7faf37',
                                       '9f8100a18751192a221f3dc1affdb862cfe31c4cc845c3b96b3e841e7fca2073'),
    'supervisor-faults.json': ('82b73acbb6df3c7fb087c0d9fa40ac8115c6c18108638587eb316cf74131d690',
                              '65c7beb7b79a8f14449967f72d2676f1982c38fca3333ef3b95361fa7baf730c'),
}
SANITIZED = {name: dict(original_sha256=o, sanitized_sha256=c) for name, (o, c) in SANITIZED.items()}
HEX64 = re.compile('[0-9a-f]{64}')
# Missing or malformed fields (NOT PROVEN) versus unreadable or contradicted evidence (FAIL).
INCOMPLETE = (KeyError, IndexError, TypeError, AttributeError)
CONTRADICTED = (ValueError, ArithmeticError)
PYTEST_PROGRESS = re.compile(r'\.+\s*\[\s*\d+%\]')
PYTEST_TAIL = re.compile(r'(\d+) passed in \d+(?:\.\d+)?s(?: \(\d+:\d\d:\d\d\))?')
UNITTEST_RAN = re.compile(r'Ran (\d+) tests? in \d+(?:\.\d+)?s')


class Missing(Exception):
    """An evidence file is absent: the condition is NOT PROVEN."""


class Evidence:
    def __init__(self, directory, root):
        self.directory, self.root, self.cache = Path(directory), Path(root), {}

    def read(self, path):
        if path not in self.cache:
            if not path.is_file():
                raise Missing(path.name)
            self.cache[path] = json.loads(path.read_text())
        return self.cache[path]

    def __getitem__(self, name):
        return self.read(self.directory/name)

    def optional(self, name):
        return self[name] if (self.directory/name).is_file() else None

    def text(self, name):
        if not (self.directory/name).is_file():
            raise Missing(name)
        return (self.directory/name).read_text()

    def increment8(self, name):
        return self.read(self.root/'results/increment8'/name)


def pytest_ok(text, minimum):
    """Strict: only dotted progress lines, then exactly 'N passed in Ts' with N >= minimum."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail = PYTEST_TAIL.fullmatch(lines[-1]) if lines else None
    return bool(tail and int(tail[1]) >= minimum and len(lines) > 1
                and all(PYTEST_PROGRESS.fullmatch(line) for line in lines[:-1]))


def unittest_ok(text, minimum):
    """Strict: last line exactly 'OK', 'Ran N tests' with N >= minimum, no failure/error/skip."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ran = UNITTEST_RAN.fullmatch(lines[-2]) if len(lines) >= 2 else None
    return bool(ran and lines[-1] == 'OK' and int(ran[1]) >= minimum
                and not re.search(r'FAIL|ERROR|skipped|expected failure|unexpected success', text))


def true(checks):
    return all(value is True for value in checks.values())


def settle(lazy):
    """Evaluate every check on its own, so that a missing field cannot hide a contradicted fact.

    With no contradicted check the first missing field is raised as before (NOT PROVEN);
    otherwise the missing checks are recorded as None and the result is not true (FAIL).
    """
    out, missing = {}, None
    for name, check in lazy.items():
        try:
            out[name] = check()
        except (Missing, *INCOMPLETE) as error:
            out[name], missing = None, missing or error
    if missing is not None and not any(value is False for value in out.values()):
        raise missing
    return out


def natural(*values):
    """Counters are non-negative JSON integers: bool (an int subclass), float, str and None are not."""
    return all(type(value) is int and value >= 0 for value in values)


def zero(*values):
    """Counters must be the integer 0: JSON false or 0.0 is not accepted as zero."""
    return all(type(value) is int and value == 0 for value in values)


def exactly(value, expected):
    """An integer counter equal to an integer: 2147483648.0 is not the approved limit."""
    return natural(value) and value == expected


def counters_equal(actual, expected):
    """Per-key counters equal to the expected ones, every value a non-negative integer."""
    return (isinstance(actual, dict) and isinstance(expected, dict) and natural(*actual.values())
            and natural(*expected.values()) and actual == expected)


def counter_sum(values):
    """Sum of non-negative integer counters, None when any of them is not one (never a TypeError)."""
    values = list(values)
    return sum(values) if natural(*values) else None


def combine(*verdicts):
    """The worst verdict, FAIL > NOT PROVEN > PASS; an unknown verdict counts as FAIL."""
    return max((verdict if verdict in SEVERITY else FAIL for verdict in verdicts), key=SEVERITY.__getitem__)


def output_argument(command, flag='--output'):
    return Path(command[command.index(flag)+1]).name if flag in command else None


def scope(sup, runtime, runtime_name, parameters):
    """Kernel limits and counters of one supervised run, bound to the worker's own report."""
    fin = sup['finalizer']
    if fin is None:
        return dict(finalizer_recorded=False), {}
    unit = sup['unit']
    events, local = lambda: fin['memory.events'], lambda: fin['memory.events.local']
    anon_kernel = lambda: sup['sampled_peaks']['stat.anon']+sup['sampled_peaks']['stat.kernel']
    lazy = dict(
        finalizer_recorded=lambda: True,
        exit_zero=lambda: zero(sup['exit_code'], fin['worker.exit']),
        no_watchdog=lambda: sup['watchdog_reason'] is None,
        limits_requested=lambda: (exactly(sup['memory_max'], GIB2) and zero(sup['swap_max'])
                                  and f'--unit={unit}' in sup['launcher'] and f'MemoryMax={GIB2}' in sup['launcher']
                                  and 'MemorySwapMax=0' in sup['launcher']),
        limits_observed_by_kernel=lambda: exactly(fin['memory.max'], GIB2) and zero(fin['memory.swap.max']),
        finalizer_in_same_scope=lambda: (fin['cgroup.path'].endswith(f'/{unit}.scope')
                                         and sup['scope'].endswith(f'/{unit}.scope')),
        peak_within_limit=lambda: natural(fin['memory.peak']) and 0 < fin['memory.peak'] <= GIB2,
        no_oom=lambda: zero(events()['oom'], events()['oom_kill'], local()['oom'], local()['oom_kill']),
        no_swap=lambda: zero(fin['memory.swap.peak'], fin['memory.swap.current'],
                             sup['sampled_peaks']['memory.swap.current']),
        global_oom_kill_unchanged=lambda: (natural(sup['global_oom_kill_after'], sup['global_oom_kill_before'])
                                           and sup['global_oom_kill_after'] == sup['global_oom_kill_before']),
        sampled=lambda: natural(sup['samples']) and sup['samples'] > 0,
        anon_kernel_below_threshold=lambda: (natural(sup['sampled_peaks']['stat.anon'], sup['sampled_peaks']['stat.kernel'])
                                             and anon_kernel() <= parameters['anon_kernel_peak_max_bytes']))
    if runtime is not None:
        inside, close = lambda: runtime['scope'], lambda: runtime['process_after_close']
        lazy.update(
            worker_in_supervisor_unit=lambda: (
                inside()['required'] is True and inside()['unit'] == unit and inside()['member'] is True
                and inside()['observed']['path'].endswith(f'/{unit}.scope')),
            worker_saw_limits_before_weights=lambda: (
                inside()['verified_before_opening_weights'] is True and exactly(inside()['observed']['memory.max'], GIB2)
                and zero(inside()['observed']['memory.swap.max']) and exactly(inside()['expected_memory_max'], GIB2)
                and zero(inside()['expected_swap_max'])),
            worker_report_is_this_file=lambda: output_argument(sup['command']) == runtime_name,
            worker_pid_sampled=lambda: any(key.startswith(f'pid{inside()["pid"]}.') for key in sup['sampled_peaks']),
            worker_vmhwm_below_threshold=lambda: (natural(close()['VmHWM_bytes'], runtime['peak_rss_bytes'])
                                                  and max(close()['VmHWM_bytes'], runtime['peak_rss_bytes'])
                                                  <= parameters['worker_vmhwm_max_bytes']),
            worker_no_swap=lambda: zero(close()['VmSwap_bytes'], *(p['process']['VmSwap_bytes'] for p in runtime['phases'])),
            worker_address_space_bounded=lambda: (
                natural(close()['VmPeak_bytes'], runtime['rlimit_as_bytes'][0], runtime['payload_bytes'])
                and close()['VmPeak_bytes'] <= runtime['rlimit_as_bytes'][0] < runtime['payload_bytes']))
    checks = settle(lazy)
    try:
        facts = dict(unit=unit, memory_peak=fin['memory.peak'], max_events=events()['max'], oom=events()['oom'],
                     oom_kill=events()['oom_kill'], swap_peak=fin['memory.swap.peak'],
                     sampled_anon_plus_kernel_peak=anon_kernel(), sampled_file_peak=sup['sampled_peaks']['stat.file'],
                     global_swap_used_before=sup['global_before']['SwapTotal']-sup['global_before']['SwapFree'],
                     global_swap_used_after=sup['global_after']['SwapTotal']-sup['global_after']['SwapFree'],
                     minimum_global_memavailable=sup['minimum_global_memavailable'])
        if runtime is not None:
            facts.update(worker_vmhwm=close()['VmHWM_bytes'], worker_vmpeak=close()['VmPeak_bytes'],
                         worker_block_read_bytes=runtime['inference_proc_io_delta']['read_bytes'],
                         startup_block_read_bytes=runtime['startup_proc_io_delta']['read_bytes'],
                         logical_inference_bytes=runtime['stats']['inference_read_bytes'],
                         host_disk_delta=sup['host_disk_delta'])
    except INCOMPLETE as error:
        if not any(value is False for value in checks.values()):
            raise
        facts = dict(incomplete_evidence=repr(error))  # the contradicted checks above stay FAIL
    return checks, facts


def eviction(sup, runtime, directory, parameters):
    """Targeted eviction verified before launch, for exactly the files the worker then hashed."""
    after = sup['residency_after_eviction']
    sizes = {name: entry['nbytes'] for name, entry in runtime['source_files'].items()}
    suffix = lambda key: next((n for n in sizes if key.endswith(f'/checkpoints/{directory}/{n}')), None)
    limit = parameters['resident_after_eviction_max_fraction']
    return dict(
        condition_evicted=sup['condition'] == 'evicted',
        evicted_files_are_the_checkpoint=(len(after) == len(sizes) and {suffix(k) for k in after} == set(sizes)
                                          and sorted(sup['files']) == sorted(after)),
        residency_sizes_equal_files=all(exactly(v['size_bytes'], sizes.get(suffix(k))) for k, v in after.items()),
        resident_after_eviction_within_limit=all(natural(v['resident_bytes'], v['size_bytes'])
                                                 and v['resident_bytes'] <= v['size_bytes']*limit for v in after.values())
    ), dict(resident_bytes_after_eviction=counter_sum(v['resident_bytes'] for v in after.values()),
            file_bytes=counter_sum(v['size_bytes'] for v in after.values()))


def forward(r, decodes):
    """A complete transactional prefill/decode sequence, from per-phase raw records."""
    phases, stats, prompt = r['phases'], r['stats'], r['input_ids']
    names = [f'decode.{i}' for i in range(len(phases))]
    expected = r['expected_tensor_read_bytes']
    return dict(
        prefill_is_the_prompt=phases[0]['ids'] == prompt and len(prompt) >= 2,
        enough_decodes=len(phases)-1 >= decodes,
        phase_names=[p['name'] for p in phases] == names and sorted(r['captures']) == names,
        kv_lengths=all(natural(p['length']) for p in phases) and [p['length'] for p in phases] == list(range(len(prompt), len(prompt)+len(phases))),
        kv_generations=all(natural(p['generation']) for p in phases) and [p['generation'] for p in phases] == list(range(1, len(phases)+1)),
        kv_prefix_preserved=all(p['kv_prefix_preserved'] is True for p in phases),
        decode_inputs_are_previous_greedy=all(len(p['greedy']) == len(p['ids']) for p in phases) and all(
            phase['ids'] == previous['greedy'][-1:] for previous, phase in zip(phases, phases[1:])),
        continuation_is_greedy=r['continuation'] == [p['greedy'][-1] for p in phases],
        capture_digests_are_phase_digests=all(
            r['captures'][p['name']][kind]['sha256'] == p['digests'][kind] and HEX64.fullmatch(p['digests'][kind])
            for p in phases for kind in ('logits', 'hidden', 'kv')),
        per_tensor_reads_equal_expected=bool(expected) and counters_equal(stats['tensor_read_bytes'], expected)
                                        and all(v > 0 for v in expected.values()),
        inference_bytes_equal_tensor_sum=exactly(stats['inference_read_bytes'], sum(expected.values())),
        requests_released=(zero(stats['failed'])
                           and natural(stats['requests'], stats['completed'], stats['released'], stats['session_length'])
                           and stats['requests'] == stats['completed'] == stats['released']
                           and stats['request_state'] == 'RELEASED' and stats['session_failed'] is False
                           and stats['session_length'] == len(prompt)+len(phases)-1),
        fds_and_arenas_released=r['released_fds'] is True and r['released_arenas'] is True,
        no_weight_cache_or_heavy_module=(r['weight_cache'] is False and zero(r['gpu_allocations_bytes'])
                                         and r['heavy_modules_loaded'] == []),
        budget_equals_plan=(natural(r['budget_bytes'], r['arena_plan_bytes'], stats['reserved_peak_bytes'])
                            and r['budget_bytes'] == r['arena_plan_bytes'] == stats['reserved_peak_bytes']),
        full_integrity_at_startup=(stats['integrity_policy'] == 'full'
                                   and natural(r['integrity_read_bytes'], stats['source_file_bytes'],
                                               *(f['nbytes'] for f in r['source_files'].values()))
                                   and r['integrity_read_bytes'] == sum(f['nbytes'] for f in r['source_files'].values())
                                   == stats['source_file_bytes']))


def qwen_reads(preflight, positions):
    """Per-tensor logical bytes of a dense Qwen2 forward, from the official header shapes only."""
    tensors, config = preflight['tensors'], preflight['config']
    out = {'embedding': positions*2*config['hidden_size'],
           'final_norm': positions*tensors['model.norm.weight']['nbytes'],
           'lm_head': positions*tensors['lm_head.weight']['nbytes']}
    for layer in range(config['num_hidden_layers']):
        for short, physical in QWEN_TENSORS.items():
            out[f'layers.{layer}.{short}'] = positions*tensors[f'model.layers.{layer}.{physical}']['nbytes']
    return out


def parity(report, contract, runtime, runtime_name, tensor_count):
    """Rebuild one Qwen parity verdict from its comparisons; 'passed' is only cross-checked."""
    dtype, profile, key = CONTRACTS[contract]
    meta, config, phases = report['oracle_metadata'], runtime['config'], runtime['phases']
    sequence = [t for p in phases for t in p['ids']]
    prompt, layers, vocab = len(runtime['input_ids']), config['num_hidden_layers'], config['vocab_size']
    kv_width = config['num_key_value_heads']*(config['hidden_size']//config['num_attention_heads'])
    comparisons = report['comparisons']
    structure = dict(
        contract=report['contract'] == contract == meta['contract'] and meta['dtype'] == dtype
                 and meta['precision_profile'] == profile and meta['attention_implementation'] == 'eager',
        revision=meta['revision'] == REVISION_B,
        approved_thresholds=(report['atol'], report['rtol']) == THRESHOLDS[key],
        oracle_digest=bool(HEX64.fullmatch(report['oracle_sha256'])) and report['oracle_sha256'] == meta['capture_sha256'],
        same_runtime_report=Path(report['runtime_report']).name == runtime_name,
        same_sequence=meta['sequence'] == sequence and meta['prompt_length'] == prompt,
        every_tensor_loaded_per_call=(meta['loader']['tensors'] == tensor_count
                                      and meta['loader']['loads_per_tensor'] == len(meta['chunks'])
                                      and sum(meta['chunks']) == len(sequence)),
        phases_compared=sorted(comparisons) == [p['name'] for p in phases],
        coverage_declared=report['coverage'] == dict(positions=len(sequence), logits_per_position=vocab, layers=layers,
                                                     kv_layers=layers, full_arrays=True))
    if contract != DIAGNOSTIC:
        structure['incremental_calls_like_runtime'] = meta['chunks'] == [len(p['ids']) for p in phases]
    total = dict(logits=0, hidden=0, kv=0)
    counts, failures, clean, argmax, greedy, upstream = True, dict(total), True, True, True, []
    for phase in phases:
        c, n = comparisons[phase['name']], len(phase['ids'])
        expected = dict(logits=n*vocab, hidden=n*layers*config['hidden_size'], kv=phase['length']*2*layers*kv_width)
        for kind, count in expected.items():
            per_layer = c[kind]['layers'] if kind != 'logits' else []
            counts &= exactly(c[kind]['count'], count) and natural(c[kind]['failures']) and (kind == 'logits' or (
                len(per_layer) == layers and natural(*(x['count'] for x in per_layer), *(x['failures'] for x in per_layer))
                and sum(x['count'] for x in per_layer) == count
                and sum(x['failures'] for x in per_layer) == c[kind]['failures']))
            # Facts only; a counter that is not a non-negative integer is already a FAIL in `counts`.
            total[kind] = counter_sum((total[kind], c[kind]['count'])) if total[kind] is not None else None
            failures[kind] = counter_sum((failures[kind], c[kind]['failures'])) if failures[kind] is not None else None
            clean &= zero(c[kind]['failures'], *(x['failures'] for x in per_layer))
        argmax &= c['argmax_equal'] is True and len(c['top10']) == n and all(
            row['actual'][0] == row['upstream'][0] == token for row, token in zip(c['top10'], phase['greedy']))
        greedy &= c['runtime_greedy_recorded'] is True
        upstream += [row['upstream'][0] for row in c['top10']]
    structure['element_counts_complete'] = bool(counts) and report['compared_elements'] == total
    upstream_greedy = (upstream == meta['argmax'] and upstream[prompt-1:len(sequence)-1] == sequence[prompt:]
                       and report['upstream_greedy_matches_sequence'] is True
                       and meta['upstream_greedy_matches_sequence'] is True)
    raw = bool(clean) and bool(argmax) and bool(greedy) and upstream_greedy
    structure['recorded_verdict_equals_recomputed'] = report['passed'] is raw
    return dict(structure_ok=true(structure), raw_passed=raw, facts=dict(
        role='diagnostic, raw verdict kept' if contract == DIAGNOSTIC else 'gate contract',
        raw_verdict=PASS if raw else FAIL, structure=structure, compared=total, failures=failures,
        max_abs={k: max(c[k]['max_abs'] for c in comparisons.values()) for k in total},
        argmax_equal=bool(argmax), runtime_greedy_recorded=bool(greedy), upstream_greedy=upstream_greedy,
        oracle_sha256=report['oracle_sha256']))


def phase_a_parity(report, runtime, runtime_name, evidence):
    """Increment-8 rule, rebuilt from the comparisons and bound to the increment-8 oracle."""
    from .report_increment8 import full_fp32_parity
    reference, protocol8 = evidence.increment8('parity-original-sync.json'), evidence.increment8('protocol.json')
    return dict(
        rebuilt_from_comparisons=full_fp32_parity(report, runtime) is True,
        same_runtime_report=Path(report['runtime_report']).name == runtime_name and report['scenario'] == 'decode',
        increment8_oracle=report['oracle_sha256'] == reference['oracle_sha256'],
        increment8_protocol=(protocol8['revision'] == REVISION_A == report['oracle_metadata']['revision']
                             and (protocol8['fp32_comparison']['atol'], protocol8['fp32_comparison']['rtol'])
                             == THRESHOLDS['fp32_comparison'] == (report['atol'], report['rtol'])))


@functools.lru_cache(maxsize=None)
def file_sha256(path, size, mtime_ns):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def capture_files(manifest, root):
    """Hash the local capture files when they exist; say so explicitly when they do not."""
    entries = manifest['files']
    present = [name for name in entries if (Path(root)/name).is_file()]
    if not present:
        return dict(state='absent: captures are local and outside Git; manifest digests not re-verified here',
                    files=len(entries))
    bad = [name for name in entries if name not in present]
    for name in present:
        path = Path(root)/name
        stat = path.stat()
        if stat.st_size != entries[name]['nbytes'] or file_sha256(str(path), stat.st_size, stat.st_mtime_ns) != entries[name]['sha256']:
            bad.append(name)
    if bad:
        return dict(state='mismatch: capture files missing or different from the manifest', files=len(entries),
                    different=sorted(bad))
    return dict(state=f'verified: all {len(entries)} capture files re-hashed and equal to the manifest', files=len(entries))


def captures_bound(manifest, runtimes, oracle_digests):
    """Manifest entries equal the digests recorded by each run and each parity report."""
    entries, ok = manifest['files'], True
    for label, runtime in runtimes.items():
        for phase, kinds in runtime['captures'].items():
            for kind, d in kinds.items():
                ok &= entries[f'artifacts/increment9/{label}/{d["file"]}'] == dict(nbytes=d['nbytes'], sha256=d['sha256'])
    for contract, digest in oracle_digests.items():
        ok &= entries[f'artifacts/increment9/oracle-{contract}.npz']['sha256'] == digest
    return bool(ok) and len(set(oracle_digests.values())) == len(oracle_digests)


def protocol_precondition(evidence):
    facts = {}
    try:
        protocol, amendment = lambda: evidence['protocol.json'], lambda: evidence[AMENDMENT]
        protocol()  # absent: NOT PROVEN; not valid JSON: FAIL
        digest = facts['protocol_sha256'] = hashlib.sha256((evidence.directory/'protocol.json').read_bytes()).hexdigest()
        cgroup = lambda: protocol()['phase_a']['cgroup']
        # Each check on its own: a missing amendment or field cannot hide an altered protocol.
        checks = facts['checks'] = settle(dict(
            protocol_file_is_the_approved_one=lambda: digest == PROTOCOL_SHA256,
            thresholds_approved=lambda: all((protocol()[k]['atol'], protocol()[k]['rtol']) == v
                                            for k, v in THRESHOLDS.items()),
            limits_approved=lambda: exactly(cgroup()['memory.max'], GIB2) and zero(cgroup()['memory.swap.max']),
            models_pinned=lambda: (protocol()['phase_a']['revision'] == REVISION_A
                                   and protocol()['phase_a']['file_sha256'] == SHA_A
                                   and protocol()['phase_b']['revision'] == REVISION_B
                                   and protocol()['baseline_head'] == BASELINE),
            remediation_parameters_are_the_gate_ones=lambda: amendment()['gate_parameters'] == PARAMETERS,
            amendment_declared_post_measurement=lambda: (amendment()['status'].startswith('post-measurement')
                                                         and amendment()['amends_protocol_sha256'] == PROTOCOL_SHA256)))
        return dict(verdict=PASS if true(checks) else FAIL, facts=facts)
    except Missing as error:
        return dict(verdict=NOT_PROVEN, facts=facts | {'missing_evidence': str(error)})
    except INCOMPLETE as error:
        return dict(verdict=NOT_PROVEN, facts=facts | {'incomplete_evidence': repr(error)})
    except CONTRADICTED as error:  # protocol or amendment present but not valid JSON
        return dict(verdict=FAIL, facts=facts | {'contradicted_evidence': repr(error)})


def condition_a(f, e):
    """Integrity of the originals: post-acquisition proofs against the official digests."""
    pre, acq, integ = e['qwen-preflight.json'], e['qwen-acquisition.json'], e['integrity-after.json']
    ra, rb, rp = e['phaseA-P-runtime.json'], e['phaseB-sync-runtime.json'], e['phaseB-pipeline-runtime.json']
    official = {n: pre['shards'][n]['sha256'] for n in SHARDS}
    sizes = {n: pre['shards'][n]['nbytes'] for n in SHARDS}
    c = f['checks'] = dict(
        preflight_official_bf16_revision=(
            pre['model'] == 'Qwen/Qwen2.5-7B' and pre['revision'] == REVISION_B and pre['dtypes'] == ['BF16']
            and sorted(pre['shards']) == sorted(SHARDS) and all(HEX64.fullmatch(v) for v in official.values())
            and all(pre['files'][n]['lfs_sha256'] == official[n] and pre['files'][n]['nbytes'] == sizes[n]
                    for n in SHARDS)
            and sum(sizes.values()) == sum(pre['shards'][n]['header_bytes'] for n in SHARDS)+pre['payload_bytes']),
        acquisition_used_the_same_official_digests=(
            acq['revision'] == REVISION_B and acq['execute'] is True
            and all(acq['files'][n]['lfs_sha256'] == official[n] and acq['downloaded'][n]['nbytes'] == sizes[n]
                    and acq['downloaded'][n]['sha256'] == official[n] for n in SHARDS)))
    computed = {n: acq['downloaded'][n].get('computed_sha256') for n in SHARDS}
    recorded = all(v is not None for v in computed.values())
    # A recorded computed digest is evidence only as a 64-hex value equal to the official one.
    c['acquisition_verified_every_file'] = (
        all(v is None or (isinstance(v, str) and bool(HEX64.fullmatch(v)) and v == official[n])
            for n, v in computed.items())
        and (recorded or acq['existing'] == []))
    f['acquisition_time_verification'] = dict(
        files_present_before_download=acq['existing'], computed_digest_recorded=recorded,
        statement=('computed digests recorded next to the official ones' if recorded else
                   'the download code raised on any streaming SHA256 mismatch before renaming, but the computed '
                   'digest was not recorded in this historical run: the sha256 field is the official value copied. '
                   'It is not used as proof; the proof is the post-acquisition integrity below.'))
    for label, r in (('sync', rb), ('pipeline', rp)):
        c[f'runtime_{label}_full_sha256_equals_official_lfs'] = (
            r['source_files'] == {n: dict(nbytes=sizes[n], sha256=official[n]) for n in SHARDS}
            and r['stats']['integrity_policy'] == 'full' and r['integrity_read_bytes'] == sum(sizes.values()))
    c['runtime_smollm_full_sha256_equals_protocol'] = (
        ra['source_files'] == {'model.safetensors': dict(nbytes=ra['integrity_read_bytes'], sha256=SHA_A)}
        and ra['stats']['integrity_policy'] == 'full')
    index_sha = acq['downloaded'][INDEX]['sha256']
    names = {f'{QWEN}/{n}': official[n] for n in SHARDS} | {f'{SMOLLM}/model.safetensors': SHA_A}

    def rehash(report, index_digest_required):
        files = report['files']
        index = files.get(f'{QWEN}/{INDEX}', {})
        # integrity-after.json is historical and recorded only the outcome for the index; any
        # later re-hash must carry the index digest, equal to the one the descriptor was built from.
        recorded = (index.get('sha256') == index.get('expected') == index_sha if index_digest_required
                    else index.get('sha256', index_sha) == index.get('expected', index_sha) == index_sha)
        return (sorted(files) == sorted([*names, f'{QWEN}/{INDEX}'])
                and all(files[n]['sha256'] == digest == files[n]['expected'] and files[n]['match'] is True
                        for n, digest in names.items())
                and index.get('match') is True and recorded and isinstance(index_sha, str)
                and bool(HEX64.fullmatch(index_sha)) and report['all_match'] is True)
    c['rehash_recomputed_per_file_equals_official'] = rehash(integ, index_digest_required=False)
    final = e.optional('integrity-final.json')
    if final is None:
        f['rehash_after_remediation'] = 'absent: integrity-after.json is the last recorded re-hash'
    else:
        c['rehash_after_remediation_equals_official'] = rehash(final, index_digest_required=True)
        f['rehash_after_remediation'] = 'present: integrity-final.json, a later full re-hash, also judged per file'
    f['integrity_read_bytes'] = dict(smollm=ra['integrity_read_bytes'], qwen=rb['integrity_read_bytes'])
    return true(c)


def phase_b_runs(e, parameters):
    """Both phase-B runs: scope, eviction before launch and real block reads under the limit."""
    out = {}
    for run in ('sync', 'pipeline'):
        sup, runtime = e[f'phaseB-{run}-supervisor.json'], e[f'phaseB-{run}-runtime.json']
        checks, facts = scope(sup, runtime, f'phaseB-{run}-runtime.json', parameters)
        if checks['finalizer_recorded']:
            evicted, residency = eviction(sup, runtime, QWEN, parameters)
            logical, block = runtime['stats']['inference_read_bytes'], runtime['inference_proc_io_delta']['read_bytes']
            streaming = evicted | dict(
                block_reads_cover_logical_bytes=(natural(block, logical)
                                                 and block >= logical*parameters['phase_b_block_read_min_fraction'] > 0),
                checkpoint_is_qwen=(runtime['family'] == 'qwen2' and runtime['mode'] == run
                                    and runtime['checkpoint'].endswith('checkpoints/'+QWEN)
                                    and sorted(runtime['source_files']) == sorted(SHARDS)))
            facts |= residency | dict(block_read_fraction=block/logical if natural(block, logical) and logical else None)
        else:
            streaming = dict(finalizer_recorded=False)
        out[run] = dict(scope=checks, streaming=streaming, facts=facts)
    return out


def condition_b(f, e, parameters):
    """Phase A: SmolLM2-1.7B evicted, then run under the verified 2 GiB scope."""
    sup, r, report = e['phaseA-P-supervisor.json'], e['phaseA-P-runtime.json'], e['phaseA-P-parity-inc8.json']
    inc8 = e.increment8('strict-sync-128.json')
    c, facts = scope(sup, r, 'phaseA-P-runtime.json', parameters)
    f['checks'] = c
    f.update(facts)
    if not c['finalizer_recorded']:
        return False
    evicted, residency = eviction(sup, r, SMOLLM, parameters)
    c.update(evicted)
    f.update(residency)
    c.update(forward(r, decodes=3))
    c.update(
        protocol_model_and_settings=(r['source_files']['model.safetensors']['sha256'] == SHA_A and r['family'] == 'llama'
                                     and r['mode'] == 'sync' and r['tile_elements'] == 262144 and r['capacity'] == 16),
        pages_read_under_the_limit=natural(r['inference_proc_io_delta']['read_bytes'])
                                   and r['inference_proc_io_delta']['read_bytes'] > 0,
        digests_equal_increment8=[p['digests'] for p in r['phases']] == [p['digests'] for p in inc8['phases']],
        per_tensor_reads_equal_increment8=counters_equal(r['stats']['tensor_read_bytes'], inc8['expected_tensor_read_bytes']))
    f['parity_increment8_fp32_oracle'] = c_parity = phase_a_parity(report, r, 'phaseA-P-runtime.json', e)
    c['parity_increment8_fp32_oracle_rebuilt'] = true(c_parity)
    f['prefill_tokens'], f['decodes'] = len(r['input_ids']), len(r['phases'])-1
    return true(c)


def condition_c(f, e, parameters):
    """I/O with correct attribution: warm pages charged elsewhere vs pages re-read under the limit."""
    sw, rw = e['phaseA-W-supervisor.json'], e['phaseA-W-runtime.json']
    sp, rp = e['phaseA-P-supervisor.json'], e['phaseA-P-runtime.json']
    wc, wf = scope(sw, rw, 'phaseA-W-runtime.json', parameters)
    pc, pf = scope(sp, rp, 'phaseA-P-runtime.json', parameters)
    b = phase_b_runs(e, parameters)
    warm = sw['residency_after_warm']
    resident = counter_sum(v['resident_bytes'] for v in warm.values())
    size = counter_sum(v['size_bytes'] for v in warm.values())
    f.update(warm=wf | dict(resident_before_launch=resident, file_bytes=size), evicted=pf,
             qwen={run: v['facts'] for run, v in b.items()},
             note='host disk counters are global to the Windows physical disk')
    c = f['checks'] = dict(
        warm_run_bound_to_its_scope=true(wc), evicted_run_bound_to_its_scope=true(pc),
        warm_condition=(sw['condition'] == 'warm' and len(warm) == 1 and natural(resident, size)
                        and exactly(rw['integrity_read_bytes'], size) and resident >= size*0.99),
        warm_pages_not_read_as_blocks=(natural(rw['inference_proc_io_delta']['read_bytes'], size)
                                       and rw['inference_proc_io_delta']['read_bytes'] < size//100),
        evicted_condition=true(eviction(sp, rp, SMOLLM, parameters)[0]) if pc['finalizer_recorded'] else False,
        evicted_pages_read_as_blocks=(natural(rp['inference_proc_io_delta']['read_bytes'], rp['payload_bytes'])
                                      and rp['inference_proc_io_delta']['read_bytes'] >= rp['payload_bytes'] > 0),
        host_disk_counters_recorded=all(x.get('host_disk_delta') is not None
                                        for x in (wf, pf, *(v['facts'] for v in b.values()))))
    for run, v in b.items():
        c[f'phase_b_{run}_bound_to_its_scope'] = true(v['scope'])
        c[f'phase_b_{run}_evicted_and_read_as_blocks'] = true(v['streaming'])
        f[f'phase_b_{run}_checks'] = v['streaming']
    return true(c)


def condition_d(f, e, parameters):
    """Unique active BF16 bytes above MemTotal, recomputed from the official headers."""
    pre, rb, sup = e['qwen-preflight.json'], e['phaseB-sync-runtime.json'], e['phaseB-sync-supervisor.json']
    sequence = [t for p in rb['phases'] for t in p['ids']]
    independent = pre['payload_bytes']-pre['embedding_bytes']+len(set(sequence))*pre['config']['hidden_size']*2
    memtotal = rb['meminfo_start']['MemTotal']
    b = phase_b_runs(e, parameters)['sync']
    f.update(memtotal_at_start=memtotal, runtime_unique_active_bf16_bytes=rb['unique_active_bf16_bytes'],
             independent_from_official_headers=independent, margin_bytes=independent-memtotal,
             streaming_checks=b['streaming'])
    c = f['checks'] = dict(
        runtime_equals_independent_recomputation=natural(independent) and exactly(rb['unique_active_bf16_bytes'], independent),
        above_memtotal=natural(memtotal) and independent > memtotal,
        memtotal_also_seen_by_supervisor=exactly(sup['global_at_launch']['MemTotal'], memtotal),
        counters_cover_every_official_tensor=counters_equal(rb['stats']['tensor_read_bytes'], qwen_reads(pre, len(sequence))),
        run_bound_to_its_scope=true(b['scope']),
        shards_evicted_and_read_as_blocks=true(b['streaming']))
    return true(c)


def condition_e(f, e):
    """Complete real forward: prefill and decodes over every official tensor."""
    pre, rb, rp = e['qwen-preflight.json'], e['phaseB-sync-runtime.json'], e['phaseB-pipeline-runtime.json']
    sequence = [t for p in rb['phases'] for t in p['ids']]
    expected = qwen_reads(pre, len(sequence))
    matrices = sum(t['nbytes']//2 for name, t in pre['tensors'].items()
                   if len(t['shape']) == 2 and name != 'model.embed_tokens.weight')
    c = f['checks'] = forward(rb, decodes=1)
    c.update(
        config_is_the_official_one=rb['config'] == pre['config'],
        per_tensor_reads_equal_official_shapes=(counters_equal(rb['stats']['tensor_read_bytes'], expected)
                                                and len(expected) == pre['tensor_count']),
        gemv_coefficients_equal_official_shapes=exactly(rb['stats']['gemv_weight_elements'], len(sequence)*matrices),
        pipeline_outputs_byte_identical=([p['digests'] for p in rp['phases']] == [p['digests'] for p in rb['phases']]
                                         and [p['ids'] for p in rp['phases']] == [p['ids'] for p in rb['phases']]),
        pipeline_forward_complete=true(forward(rp, decodes=1)))
    f.update(prefill_tokens=len(rb['input_ids']), decodes=len(rb['phases'])-1, layers=rb['config']['num_hidden_layers'],
             tensors=len(expected), continuation=rb['continuation'], generated_text=rb['generated_text'],
             lengths=[p['length'] for p in rb['phases']], generations=[p['generation'] for p in rb['phases']],
             gemv_coefficients_all_positions=rb['stats']['gemv_weight_elements'], positions=len(sequence))
    return true(c)


def condition_f(f, e, parameters):
    """Independent numerical verification, coverage rebuilt; the diagnostic FAIL stays reported."""
    rb, pre, manifest = e['phaseB-sync-runtime.json'], e['qwen-preflight.json'], e['capture-manifest.json']
    f['contracts'] = contracts = {}
    f['checks'] = c = {}
    # Every report is read before judging: a missing one, diagnostic included, is NOT PROVEN.
    reports = {contract: e[f'parity-{contract}.json'] for contract in CONTRACTS}
    for contract, report in reports.items():
        result = parity(report, contract, rb, 'phaseB-sync-runtime.json', pre['tensor_count'])
        contracts[contract] = result['facts']
        c[f'{contract}_report_complete_and_bound'] = result['structure_ok']
        if contract in GATING:
            c[f'{contract}_passes_recomputed'] = result['raw_passed']
    f['stock_fp64_diagnostic_raw_verdict'] = contracts[DIAGNOSTIC]['raw_verdict']
    f['small_case_checks'] = c['small_case_checks_ok'] = unittest_ok(
        e.text('oracle9-checks.txt'), parameters['minimum_unittest_ran']['oracle9-checks.txt'])
    for contract in CONTRACTS:
        sup = e[f'supervisor-oracle-{contract}.json']
        c[f'{contract}_oracle_run_clean'] = (
            zero(sup['exit_code']) and sup['watchdog_reason'] is None and 'tools.oracle9' in sup['command']
            and sup['command'][sup['command'].index('--contract')+1] == contract
            and sup['command'][sup['command'].index('--sequence')+1] == ','.join(map(str, reports[contract]['oracle_metadata']['sequence']))
            and output_argument(sup['command']) == f'oracle-{contract}.npz')
    ra = e['phaseA-P-runtime.json']
    f['phase_a_parity'] = a = phase_a_parity(e['phaseA-P-parity-inc8.json'], ra, 'phaseA-P-runtime.json', e)
    c['phase_a_parity_rebuilt'] = true(a)
    runtimes = {label: e[f'{label}-runtime.json'] for label in ('phaseA-P', 'phaseA-W', 'phaseB-sync', 'phaseB-pipeline')}
    c['manifest_binds_captures_and_oracles'] = captures_bound(
        manifest, runtimes, {contract: report['oracle_sha256'] for contract, report in reports.items()})
    f['capture_files'] = local = capture_files(manifest, e.root)
    c['capture_files_not_contradicted'] = not local['state'].startswith('mismatch')
    return true(c)


def condition_g(f, e, parameters):
    """Resources within the declared limits in every supervised run; no OOM, no swap."""
    runs = {'phaseA-W': ('phaseA-W-supervisor.json', 'phaseA-W-runtime.json'),
            'phaseA-P': ('phaseA-P-supervisor.json', 'phaseA-P-runtime.json'),
            'phaseB-sync': ('phaseB-sync-supervisor.json', 'phaseB-sync-runtime.json'),
            'phaseB-pipeline': ('phaseB-pipeline-supervisor.json', 'phaseB-pipeline-runtime.json'),
            'faults': ('supervisor-faults.json', None)}
    c = f['checks'] = {}
    f['all_supervised_runs'] = {}
    for label, (supervisor, runtime) in runs.items():
        sup = e[supervisor]
        checks, facts = scope(sup, e[runtime] if runtime else None, runtime, parameters)
        f['all_supervised_runs'][label] = facts
        c[f'{label}_within_limits'] = true(checks)
        if not true(checks):
            f[f'{label}_failed_checks'] = sorted(k for k, v in checks.items() if v is not True)
    for run, v in phase_b_runs(e, parameters).items():
        c[f'phaseB-{run}_evicted_and_read_as_blocks'] = true(v['streaming'])
    faults, sup = e['faults.json'], e['supervisor-faults.json']
    flags = ('budget_rejected_before_payload', 'old_kv_all_bytes_preserved', 'descriptor_preserved',
             'candidate_was_modified', 'fail_stop_no_resume', 'drained', 'released')
    c['faults_report_complete'] = (
        output_argument(sup['command']) == 'faults.json' and faults['passed'] is True
        and sorted((x['mode'], x['fault']) for x in faults['cases'])
        == sorted((m, k) for m in ('sync', 'pipeline') for k in ('io', 'short', 'cancel'))
        and all(x[flag] is True for x in faults['cases'] for flag in flags)
        and all(x['caught'] == {'io': 'OSError', 'short': 'ReadError', 'cancel': 'CancelledError'}[x['fault']]
                for x in faults['cases']))
    oracles = {}
    for contract in CONTRACTS:
        sup = e[f'supervisor-oracle-{contract}.json']
        oracles[contract] = dict(exit_code=sup['exit_code'], watchdog=sup['watchdog_reason'],
                                 sampled_vmswap=sup['sampled_peaks_bytes']['VmSwap'],
                                 sampled_vmhwm=sup['sampled_peaks_bytes']['VmHWM'],
                                 minimum_memavailable=sup['minimum_memavailable_bytes'])
    f['oracle_processes'] = oracles
    c['oracle_processes_no_swap_no_watchdog'] = all(
        zero(o['exit_code'], o['sampled_vmswap']) and o['watchdog'] is None for o in oracles.values())
    used = [(label, facts['global_swap_used_before'], facts['global_swap_used_after'])
            for label, facts in f['all_supervised_runs'].items() if 'global_swap_used_before' in facts]
    f['global_swap_used_bytes'] = dict(
        per_run_before_after={label: [before, after] for label, before, after in used},
        minimum=min(min(b, a) for _, b, a in used) if used else None,
        maximum=max(max(b, a) for _, b, a in used) if used else None,
        note='system-wide swap in use by processes outside the experiment cgroups; reported, not gated. '
             'Every scope has memory.swap.max=0 and swap peak 0; the increase between runs falls in the oracle window.')
    return true(c)


def git_preservation(baseline, intentional):
    from .integrity9 import preservation
    return preservation(baseline, intentional)


@functools.lru_cache(maxsize=None)
def rebuilt_gates(root):
    """Gate 7 and gate 8 recomputed by their own tools from their own evidence."""
    from .report_increment7 import resolved_report
    from .report_increment8 import report
    normal = lambda value: json.dumps(value, sort_keys=True)
    return (normal(resolved_report(Path(root)/'results/increment7/numerical-resolution')),
            normal(report(Path(root)/'results/increment8')))


def condition_h(f, e, parameters, preservation):
    """Increments 1-8 preserved: tests, gates 7/8 and git state recomputed, not trusted."""
    stored, gate7, gate8 = e['preservation.json'], e['gate7-rerun.json'], e['gate8-rerun.json']
    texts = {name: e.text(name) for name in ('pytest-final.txt', 'check-oracle8.txt', 'check-oracle-precision7.txt')}
    lines = texts['pytest-final.txt'].strip().splitlines()
    f['pytest'] = lines[-1] if lines else ''
    c = f['checks'] = dict(
        pytest_all_passed=pytest_ok(texts['pytest-final.txt'], parameters['minimum_pytest_passed']),
        oracle8_check_ok=unittest_ok(texts['check-oracle8.txt'], parameters['minimum_unittest_ran']['check-oracle8.txt']),
        oracle_precision7_check_ok=unittest_ok(texts['check-oracle-precision7.txt'],
                                               parameters['minimum_unittest_ran']['check-oracle-precision7.txt']))
    try:
        rebuilt7, rebuilt8 = (json.loads(x) for x in rebuilt_gates(str(e.root)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Missing('increment 7/8 evidence cannot be rebuilt: '+repr(error))
    normal = lambda value: json.loads(json.dumps(value, sort_keys=True))
    diagnostic = rebuilt7['contracts']['standard_fp64_diagnostic']
    c.update(
        gate7_rerun_equals_rebuilt=normal(gate7) == rebuilt7,
        gate7_passed_rebuilt=rebuilt7['passed'] is True and rebuilt7['status'] == 'passed',
        gate7_diagnostic_fail_preserved=diagnostic['raw_parity_passed'] is False and diagnostic['executed'] is True,
        gate8_rerun_equals_rebuilt=normal(gate8) == rebuilt8,
        gate8_all_checks_rebuilt=(rebuilt8['passed'] is True and len(rebuilt8['checks']) == 8
                                  and all(v is True for v in rebuilt8['checks'].values())))
    intentional = parameters['intentional_modifications']
    c['preservation_report_clean'] = (
        stored['baseline'] == BASELINE and set(stored['modified_tracked']) <= set(intentional)
        and stored['deleted'] == [] and stored['unexpected_modifications'] == [])
    try:
        live = preservation(BASELINE, intentional)
    except (OSError, subprocess.SubprocessError) as error:
        raise Missing('git state cannot be recomputed: '+type(error).__name__)
    c['git_state_recomputed_equals_report'] = (
        live['modified_tracked'] == stored['modified_tracked'] and live['deleted'] == stored['deleted'] == []
        and live['unexpected_modifications'] == [])
    f.update(modified_since_baseline=live['modified_tracked'], deleted_since_baseline=live['deleted'],
             gate7_failed_elements_diagnostic=diagnostic['failed_elements'])
    return true(c)


def gate(directory, root='.', originals=None, preservation=git_preservation):
    e = Evidence(directory, root)
    protocol = protocol_precondition(e)
    try:
        sanitization = sanitize9.verify(
            e.directory, Path(root)/sanitize9.ORIGINALS if originals is None else originals,
            e.optional(sanitize9.MANIFEST), expected=SANITIZED, present=(e.directory/sanitize9.MANIFEST).is_file())
    except CONTRADICTED as error:
        sanitization = dict(verdict=FAIL, files=0, originals='unknown',
                            problems=['sanitization manifest or evidence unreadable: '+repr(error)])
    preconditions = dict(protocol=protocol, sanitization=dict(verdict=sanitization['verdict'], facts=sanitization))
    blocked = [name for name, p in preconditions.items() if p['verdict'] != PASS]
    conditions = dict(A=(condition_a, e), B=(condition_b, e, PARAMETERS), C=(condition_c, e, PARAMETERS),
                      D=(condition_d, e, PARAMETERS), E=(condition_e, e), F=(condition_f, e, PARAMETERS),
                      G=(condition_g, e, PARAMETERS), H=(condition_h, e, PARAMETERS, preservation))
    out = {}
    for name, (function, *arguments) in conditions.items():
        facts = {}
        try:
            verdict = PASS if function(facts, *arguments) else FAIL
        except (Missing, *INCOMPLETE) as error:
            key = 'missing_evidence' if isinstance(error, Missing) else 'incomplete_evidence'
            facts[key] = str(error) if isinstance(error, Missing) else repr(error)
            # Missing evidence does not hide a fact already found contradicted.
            checks = facts.get('checks')
            contradicted = isinstance(checks, dict) and any(value is False for value in checks.values())
            verdict = FAIL if contradicted else NOT_PROVEN
        except CONTRADICTED as error:
            verdict, facts['contradicted_evidence'] = FAIL, repr(error)
        own = verdict
        if blocked:
            # Worst of the condition's own verdict and the preconditions: a FAIL precondition
            # (unapproved protocol, evidence differing from its sanitized pins) is a contradicted
            # fact for every condition, also one NOT PROVEN on its own evidence.
            verdict = combine(own, *(preconditions[b]['verdict'] for b in blocked))
            facts['precondition_not_met'] = blocked
        out[name] = dict(verdict=verdict, verdict_before_preconditions=own, facts=facts)
    return dict(protocol_sha256=protocol['facts'].get('protocol_sha256'), remediation_parameters=PARAMETERS,
                preconditions=preconditions, verdicts={k: v['verdict'] for k, v in out.items()}, conditions=out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = gate(a.directory)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(dict(preconditions={k: v['verdict'] for k, v in report['preconditions'].items()},
                          **report['verdicts']), indent=1))
    raise SystemExit(0 if all(v == PASS for v in report['verdicts'].values()) else 1)
