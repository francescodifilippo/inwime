"""Increment-10 gate A-H: PASS / FAIL / BLOCKED / NOT PROVEN, recomputed from raw evidence.

Verdicts (protocol.json "verdicts", precedence FAIL > BLOCKED > NOT PROVEN > PASS):
  PASS        every required fact present, typed and recomputed from raw data
  FAIL        a fact contradicts the criterion (also: a missing field next to a
              contradicted one in the same check; an unreadable file; a wrong type)
  BLOCKED     a required run was not attempted because of an external precondition
              recorded in blocked-<run>.json (cause and UTC time); a blocked file without
              cause or time is NOT PROVEN
  NOT PROVEN  evidence or field missing, partial coverage, or invalidated attribution

Nothing missing becomes PASS. Derived booleans in the evidence ('passed', 'match',
'verdict' of a sub-tool) are not trusted where the raw data are available: SHA256 are
compared with the pins, counters recomputed from the official headers, the numerical
parity recomputed by tools/parity10 from the capture files, the test inventory
re-verified from the archived JUnit, git state recomputed.

Evidence layout (results/increment10/campaign/): <run>-run.json (tools/campaign10),
<run>-scope.json (tools/cgroup9), <run>-forward.json (tools/benchmark_increment9),
<run>-host-start.json (tools/hostmem10), parity-14b-<contract>.json (tools/parity10),
equivalence-7b-e2e-<contract>.json (tools/equivalence10_e2e), F-<mode>-faults.json
(tools/fault10), integrity-final.json (tools/integrity9), acquisition/ (sanitized
acquisition reports), pytest-final.xml (archived JUnit). Large captures live in
artifacts/increment10/ (git-ignored) and are bound by SHA256.

Faults N1-N4 and the planned OOM N6 are evaluated inside G (supervision and fail-stop
behaviour), as in gate 9. Exit code 0 only when every condition is PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from . import protocol10
from .report_increment9 import (INCOMPLETE, CONTRADICTED, Missing, natural, zero, exactly, counters_equal,
                                counter_sum, forward as forward_checks, HEX64, qwen_reads, scope as scope_checks,
                                eviction as eviction_checks, true, output_argument)

PASS, FAIL, NOT_PROVEN, BLOCKED = 'PASS', 'FAIL', 'NOT PROVEN', 'BLOCKED'
SEVERITY = {PASS: 0, NOT_PROVEN: 1, BLOCKED: 2, FAIL: 3}
ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT/'results'/'increment10'/'campaign'
ARTIFACTS = ROOT/'artifacts'/'increment10'
CKDIR = 'qwen2.5-14b-97e1e76'
# Pinned here on purpose: editing protocol.json must not move the bar.
PROTOCOL_SHA256 = '8491cf5c69a2fb888740982c9b31b821b800a2dbe6f4437cce860c6efee19dd7'
BASE_COMMIT = '639de22ed7e30a796b831595cd0f9823d585de16'
GIB2, GIB6 = 2147483648, 6442450944
RLIMIT = {'sync': 268435456, 'pipeline': 402653184}
THRESHOLDS = {'fp32': (2e-3, 2e-4), 'fp64': (2e-5, 2e-6)}
PARAMETERS = dict(anon_kernel_peak_max_bytes=268435456, worker_vmhwm_max_bytes=268435456,
                  resident_after_eviction_max_fraction=0.01, l1_min_fraction=0.90, l4_min_fraction=0.90,
                  guest_swap_growth_max_bytes=268435456)
RUNS = {'R1': dict(cache='C', mode='sync', condition='evicted'),
        'R2': dict(cache='C', mode='pipeline', condition='evicted'),
        'R3': dict(cache='C', mode='sync', condition='evicted'),
        'R4': dict(cache='W', mode='sync', condition='as-is')}
COLD = ('R1', 'R2', 'R3')
GATING, DIAGNOSTIC = ('stock_fp32', 'full64'), 'stock_fp64'
SEQUENCE_PROMPT = [785, 6722, 315, 9625, 374]


class Blocked(Exception):
    """A required run was not attempted; the cause file is recorded."""


def combine(*verdicts):
    return max((v if v in SEVERITY else FAIL for v in verdicts), key=SEVERITY.__getitem__)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def settle(lazy, facts=None):
    """Every check evaluated on its own; a contradicted check is never hidden by a missing one.

    Without a contradicted check the most severe pending reason is raised (Blocked over
    Missing); with one, the pending checks are None and the condition is FAIL. The partial
    results are stored in facts['checks'] before raising, so a NOT PROVEN shows what passed."""
    out, pending = {}, None
    if facts is not None:
        facts['checks'] = out
    for name, check in lazy.items():
        try:
            out[name] = check()
        except Blocked as error:
            out[name], pending = None, error
        except (Missing, *INCOMPLETE) as error:
            out[name] = None
            if not isinstance(pending, Blocked):
                pending = pending or error
    if pending is not None and not any(v is False for v in out.values()):
        raise pending
    return out


class Evidence:
    def __init__(self, campaign=CAMPAIGN, artifacts=ARTIFACTS, root=ROOT):
        self.campaign, self.artifacts, self.root, self.cache = Path(campaign), Path(artifacts), Path(root), {}

    def json(self, path):
        path = Path(path)
        if path not in self.cache:
            if not path.is_file():
                raise Missing(path.name)
            self.cache[path] = json.loads(path.read_text())
        return self.cache[path]

    def file(self, name):
        return self.json(self.campaign/name)

    def run(self, run, kind='run'):
        """<run>-<kind>.json; if the run record is absent, a valid blocked file makes it BLOCKED."""
        record = self.campaign/f'{run}-run.json'
        if not record.is_file():
            blocked = sorted(self.campaign.glob(f'blocked-{run}*.json'))
            if blocked:
                b = json.loads(blocked[-1].read_text())
                if isinstance(b, dict) and b.get('cause') and b.get('utc'):
                    raise Blocked(f'{run}: {b["cause"]} at {b["utc"]} ({blocked[-1].name})')
            raise Missing(f'{run}-run.json')
        return self.file(f'{run}-{kind}.json')

    def attempts(self, prefix):
        """Run ids of a family with a run record, ordered by attempt number (prefix = attempt 1, -aN = N)."""
        out = []
        for p in self.campaign.glob(f'{prefix}*-run.json'):
            name = p.name[:-len('-run.json')]
            rest = name[len(prefix):]
            if rest == '':
                out.append((1, name))
            elif rest.startswith('-a') and rest[2:].isdigit():
                out.append((int(rest[2:]), name))
        return [n for _, n in sorted(out)]

    def latest(self, prefix):
        """Most recent attempt of a run id (numeric -aN order); blocked attempts are kept as evidence."""
        names = self.attempts(prefix)
        return names[-1] if names else prefix

    def sanitization(self):
        path = self.campaign/'sanitization-post-write.json'
        return json.loads(path.read_text()) if path.is_file() else {}

    def protocol(self):
        return self.json(self.root/'results/increment10/protocol.json')


# ---------------------------------------------------------------- shared facts

def host_values(report):
    return {k: v.get('value') for k, v in report['quantities'].items()}


def proc_read(r):
    return r['startup_proc_io_delta']['read_bytes'] + r['inference_proc_io_delta']['read_bytes']


def windows_disk(delta):
    """Read bytes of the physical disk holding C: (global counter); None when unreadable."""
    if not isinstance(delta, dict):
        return None
    c = [v for k, v in delta.items() if 'C:' in k]
    return c[0]['DiskReadBytesPersec'] if len(c) == 1 and natural(c[0].get('DiskReadBytesPersec')) else None


def io_invalidated(run_record, limit=2048, sustain_s=30):
    """Annotations of the in-run monitor, plus the same rule recomputed from the kept samples
    (PagesInput above the limit on consecutive samples spanning >= sustain_s). Either one invalidates."""
    s = run_record['observer']['summary']
    notes = [n for n in s['annotations'] if n.get('preregistered_action') == 'annotate-io-attribution-invalid']
    start = None
    for sample in s.get('last_samples', []):
        rate = sample.get('host_pages_input_per_s')
        if isinstance(rate, (int, float)) and rate > limit:
            start = sample['t'] if start is None else start
            if sample['t'] - start >= sustain_s:
                return notes + [dict(recomputed=True, from_t=start, to_t=sample['t'])]
        else:
            start = None
    return notes


def produced_bound(e, run, rec):
    """Every file listed in produced_sha256 still has that digest, or was sanitized after writing with
    the map entry original == recorded and sanitized == current (sanitization-post-write.json)."""
    log = e.sanitization()
    rows = {}
    for name, recorded in rec['produced_sha256'].items():
        path = (e.campaign/name) if '/' not in name else (e.root/name)
        if not path.is_file():
            raise Missing(f'{run}: produced file {name}')
        current = sha256(path)
        entry = log.get(name) if '/' not in name else log.get(str(Path(name).relative_to('results/increment10/campaign'))) \
            if name.startswith('results/increment10/campaign/') else None
        rows[name] = current == recorded or (isinstance(entry, dict) and entry.get('original_sha256') == recorded
                                             and entry.get('sanitized_sha256') == current)
    return rows


def observer_ok(run_record):
    s = run_record['observer']['summary']
    return (s['read_only'] is True and natural(s['counts']['valid']) and s['counts']['valid'] > 0
            and not any('not monitored' in r or 'sampler stopped' in r for r in s['stop_reasons'])
            and run_record['observer']['kills'] == [])


# ---------------------------------------------------------------- conditions

def condition_a(f, e):
    """Integrity: acquisition SHA, runtime start SHA (R1-R3) and final re-hash equal the pinned LFS digests."""
    pins = {n: s['lfs_sha256'] for n, s in e.protocol()['checkpoint']['shards'].items()}
    sizes = {n: s['nbytes'] for n, s in e.protocol()['checkpoint']['shards'].items()}
    acq = lambda: e.json(e.campaign/'acquisition'/'resume-acquisition-20261119T183903Z.json')
    final = lambda: e.file('integrity-final.json')
    lazy = dict(
        acquisition_manifest_pinned=lambda: acq()['manifest_sha256'] == e.protocol()['checkpoint']['acquisition_manifest_sha256']
        and acq()['revision'] == e.protocol()['checkpoint']['revision'],
        acquisition_sha_equals_pins=lambda: all(acq()['verification'][n]['computed_sha256'] == pins[n] for n in pins),
        acquisition_committed_equals_official=lambda: all(
            v['computed_sha256'] == v['official_lfs_sha256'] == pins[n] for n, v in acq()['committed'].items()),
        final_rehash_equals_pins=lambda: (
            all(final()['files'][f'{CKDIR}/{n}']['sha256'] == pins[n] for n in pins)),
    )
    for run in COLD:
        r = (lambda run: lambda: e.run(run, 'forward'))(run)
        lazy[f'{run}_startup_sha_equals_pins'] = (lambda r: lambda: (
            r()['stats']['integrity_policy'] == 'full'
            and {n: v['sha256'] for n, v in r()['source_files'].items()} == pins
            and {n: v['nbytes'] for n, v in r()['source_files'].items()} == sizes
            and exactly(r()['integrity_read_bytes'], sum(sizes.values()))))(r)
    c = f['checks'] = settle(lazy, f)
    return true(c)


def condition_b(f, e):
    """Memory: 2 GiB scope, swap 0, peak <= limit, no OOM, anon+kernel <= 256 MiB, VmSwap 0, RLIMIT_AS per mode."""
    c, facts = {}, {}
    pending = None
    for run, spec in RUNS.items():
        try:
            sup, r = e.run(run, 'scope'), e.run(run, 'forward')
            checks, fact = scope_checks(sup, r, f'{run}-forward.json', PARAMETERS)
            checks['mode'] = r['mode'] == spec['mode']
            checks['rlimit_as_per_mode'] = (isinstance(r['rlimit_as_bytes'], list)
                                            and r['rlimit_as_bytes'] == [RLIMIT[spec['mode']]]*2)
            fact['memory_peak_equals_max_is_not_oom'] = (fact.get('memory_peak') == GIB2 and fact.get('oom') == 0)
            c.update({f'{run}.{k}': v for k, v in checks.items()})
            facts[run] = fact
        except (Blocked, Missing, *INCOMPLETE) as error:
            c[f'{run}.evidence'] = None
            pending = error if isinstance(error, Blocked) or pending is None else pending
    f['checks'], f['runs'] = c, facts
    if any(v is False for v in c.values()):
        return False
    if pending is not None:
        raise pending
    return true(c)


def condition_c(f, e):
    """I/O attribution on R1-R3: eviction <= 1 %, L1 >= 0.90 L0 (inference and every decode), dL3 >= L1,
    dL4 >= 0.90 L1 when readable, no sustained host paging annotation (else NOT PROVEN for that run)."""
    c, facts, invalid, pending = {}, {}, [], None
    for run in COLD:
        try:
            rec, sup, r = e.run(run), e.run(run, 'scope'), e.run(run, 'forward')
            ev, ev_facts = eviction_checks(sup, r, CKDIR, PARAMETERS)
            l0, l1 = r['stats']['inference_read_bytes'], r['inference_proc_io_delta']['read_bytes']
            total = proc_read(r)
            l3 = (rec['l3_delta']['checkpoint'] or {}).get('read_bytes')
            l4 = windows_disk(sup['host_disk_delta'])
            checks = dict(ev, cache_declared_cold=rec['cache'] == 'C' and rec['condition'] == 'evicted',
                          l1_inference=natural(l0, l1) and l1 >= PARAMETERS['l1_min_fraction']*l0,
                          l1_every_phase=all(natural(p['counters_delta']['inference_read_bytes'], p['proc_io_delta']['read_bytes'])
                                             and p['proc_io_delta']['read_bytes'] >= PARAMETERS['l1_min_fraction']*p['counters_delta']['inference_read_bytes']
                                             for p in r['phases']),
                          l3_at_least_l1=natural(l3, total) and l3 >= total,
                          l4_when_readable=l4 is None or l4 >= PARAMETERS['l4_min_fraction']*total,
                          host_monitored=observer_ok(rec))
            c.update({f'{run}.{k}': v for k, v in checks.items()})
            bad = io_invalidated(rec)
            if bad:
                invalid.append(run)
            facts[run] = dict(ev_facts, l0_inference=l0, l1_inference=l1, l1_total=total, l3_read_bytes=l3,
                              l4_read_bytes_global=l4, l4_readable=l4 is not None, attribution_invalidated=bool(bad),
                              host_pages_input_max=rec['observer']['summary']['extremes'].get('host_pages_input_per_s'))
        except (Blocked, Missing, *INCOMPLETE) as error:
            c[f'{run}.evidence'] = None
            pending = error if isinstance(error, Blocked) or pending is None else pending
    f['checks'], f['runs'], f['attribution_invalidated'] = c, facts, invalid
    if any(v is False for v in c.values()):
        return False
    if pending is not None:
        raise pending
    if invalid:
        raise Missing('I/O attribution invalidated by sustained host paging (protocol rule) for '+', '.join(invalid))
    return true(c)


def runtime_forward(r, captured):
    """Gate-9 forward checks. Protocol io_protocol: R1 has full captures, R2-R4 digests only.

    For a digest-only run the two capture-file checks of gate 9 do not apply; they are replaced
    by: no capture declared, phase names exact, every phase digest a SHA256 (equality with R1 is
    checked separately in E). The placeholder captures below only neutralize the removed checks."""
    if captured:
        return forward_checks(r, 3)
    names = [f'decode.{i}' for i in range(len(r['phases']))]
    placeholder = {p['name']: {k: dict(sha256=p['digests'][k]) for k in ('logits', 'hidden', 'kv')} for p in r['phases']}
    checks = forward_checks(dict(r, captures=placeholder), 3)
    checks.pop('capture_digests_are_phase_digests')
    checks['phase_names'] = [p['name'] for p in r['phases']] == names and r['captures'] == {}
    checks['phase_digests_are_sha256'] = all(isinstance(p['digests'][k], str) and HEX64.fullmatch(p['digests'][k])
                                             for p in r['phases'] for k in ('logits', 'hidden', 'kv'))
    return checks


def recomputed_unique(protocol, sequence):
    aw = protocol['active_weights']
    return aw['unique_bytes_without_embedding'] + len(set(sequence))*aw['embedding_row_bytes']


def condition_d(f, e):
    """Unique bytes from counters > max(TotalPhysicalMemory, DIMM) measured in the same run, > MemTotal,
    equal to the header recomputation, on a C run that really read (R1)."""
    p = e.protocol()
    rec, r, host = (lambda: e.run('R1')), (lambda: e.run('R1', 'forward')), (lambda: e.run('R1', 'host-start'))
    seq = lambda: [t for ph in r()['phases'] for t in ph['ids']]
    headers = protocol10.headers()
    from_headers = lambda: (sum(d['nbytes'] for n, d in headers.items() if n != 'model.embed_tokens.weight')
                            + len(set(seq()))*2*p['model_geometry']['hidden_size'])
    ram = lambda: (host()['after']['windows']['TotalPhysicalMemory'], host()['after']['windows']['DimmBytes'])
    unique = lambda: r()['unique_active_bf16_bytes']
    c = f['checks'] = settle(facts=f, lazy=dict(
        cold_run=lambda: rec()['cache'] == 'C' and rec()['condition'] == 'evicted',
        really_read=lambda: natural(r()['inference_proc_io_delta']['read_bytes'], r()['stats']['inference_read_bytes'])
        and r()['inference_proc_io_delta']['read_bytes'] >= PARAMETERS['l1_min_fraction']*r()['stats']['inference_read_bytes'],
        unique_equals_protocol_recomputation=lambda: exactly(unique(), recomputed_unique(p, seq())),
        unique_equals_header_recomputation=lambda: exactly(unique(), from_headers()),
        ram_measured_is_host_capacity=lambda: (exactly(ram()[0], p['memory']['host_total_physical_memory']['value'])
                                               and exactly(ram()[1], p['memory']['host_installed_dimm_bytes']['value'])),
        exceeds_host_ram=lambda: natural(unique(), *ram()) and unique() > max(ram()),
        exceeds_memtotal=lambda: natural(r()['meminfo_start']['MemTotal']) and unique() > r()['meminfo_start']['MemTotal'],
        all_tensors_used=lambda: r()['all_tensors_used'] is True and len(r()['expected_tensor_read_bytes']) == p['checkpoint']['tensor_count'],
    ))
    try:
        f.update(unique_active_bf16_bytes=unique(), host_total_physical_memory=ram()[0], host_dimm_bytes=ram()[1],
                 memtotal=r()['meminfo_start']['MemTotal'], ratio_host_ram=unique()/max(ram()),
                 ratio_memtotal=unique()/r()['meminfo_start']['MemTotal'])
    except (Blocked, Missing, *INCOMPLETE, ZeroDivisionError):
        pass
    return true(c)


def condition_e(f, e):
    """Prefill 5 + 3 decodes, 48 layers, 579 counters = shape x 8, GEMV coefficients, KV 5/6/7/8,
    sync/pipeline/repeat digests identical."""
    p = e.protocol()
    preflight = dict(tensors=protocol10.headers(), config=p['model_geometry'])
    expected = lambda: qwen_reads(preflight, 8)
    lazy = {}
    for run in RUNS:
        r = (lambda run: lambda: e.run(run, 'forward'))(run)
        lazy[f'{run}.forward'] = (lambda r, run: lambda: true(runtime_forward(r(), run == 'R1')))(r, run)
        lazy[f'{run}.prompt'] = (lambda r: lambda: r()['input_ids'] == SEQUENCE_PROMPT and len(r()['phases']) == 4
                                 and [len(x['ids']) for x in r()['phases']] == [5, 1, 1, 1])(r)
        lazy[f'{run}.layers'] = (lambda r: lambda: r()['config']['num_hidden_layers'] == 48)(r)
        lazy[f'{run}.counters_from_headers'] = (lambda r: lambda: len(expected()) == p['checkpoint']['tensor_count']
                                                and counters_equal(r()['stats']['tensor_read_bytes'], expected()))(r)
        lazy[f'{run}.gemv_coefficients'] = (lambda r: lambda: exactly(
            r()['stats']['gemv_weight_elements'], p['active_weights']['gemv_coefficients_8_positions']))(r)
        lazy[f'{run}.kv_lengths'] = (lambda r: lambda: [x['length'] for x in r()['phases']] == [5, 6, 7, 8])(r)
    digests = lambda run: [x['digests'] for x in e.run(run, 'forward')['phases']]
    for run in ('R2', 'R3', 'R4'):
        lazy[f'{run}.digests_equal_R1'] = (lambda run: lambda: digests(run) == digests('R1'))(run)
    lazy['continuation_equal'] = lambda: len({tuple(e.run(run, 'forward')['continuation']) for run in RUNS}) == 1
    c = f['checks'] = settle(lazy, f)
    try:
        f['continuation'] = e.run('R1', 'forward')['continuation']
        f['generated_text'] = e.run('R1', 'forward')['generated_text']
    except (Blocked, Missing, *INCOMPLETE):
        pass
    return true(c)


def condition_f(f, e):
    """stock_fp32 and full64 PASS on FULL coverage (recomputed), oracle10 7B end-to-end equivalence PASS,
    stock_fp64 reported raw."""
    from . import parity10
    p = e.protocol()
    req = p['numerical_contract']['coverage_full']
    lazy = {}
    for contract in GATING:
        eq = (lambda c: lambda: e.file(f'equivalence-7b-e2e-{c}.json'))(contract)
        lazy[f'7b_e2e.{contract}.recomputed_and_bound'] = (lambda eq, c: lambda: e2e_recomputed(e, c, eq()))(eq, contract)
        lazy[f'7b_e2e.{contract}'] = (lambda eq, c: lambda: e.run(e.latest(f'O7-{c}')) is not None and (
            eq()['verdict'] == 'PASS' and eq()['problems'] == [] and eq()['comparison']['passed'] is True
            and eq()['comparison']['kinds']['hidden']['bit_identical'] is True
            and eq()['comparison']['kinds']['kv']['bit_identical'] is True
            and (eq()['comparison']['kinds']['logits']['bit_identical'] is True if c == 'stock_fp32'
                 else eq()['comparison']['kinds']['logits']['max_abs_difference'] <= 1e-9
                 and eq()['comparison']['logits_argmax_identical'] is True)
            and sha256(e.artifacts/f'oracle10-7b-{c}.npz') == eq()['inputs']['new_sha256']))(eq, contract)
    # Diagnostic 7B equivalence (amendment 007): judged only when its oracle run has a record; it never
    # makes F pass, it can only contradict it (recomputation != stored, unbound capture, unknown verdict).
    if e.attempts(f'O7-{DIAGNOSTIC}'):
        eq = lambda: e.file(f'equivalence-7b-e2e-{DIAGNOSTIC}.json')
        lazy[f'7b_e2e.{DIAGNOSTIC}.reported_raw'] = lambda: e2e_recomputed(e, DIAGNOSTIC, eq())
    for contract in (*GATING, DIAGNOSTIC):
        stored = (lambda c: lambda: e.file(f'parity-14b-{c}.json'))(contract)

        def recompute(c=contract, stored=stored):
            rec = e.run(e.latest(f'O14-{c}'))    # the oracle run record first: BLOCKED if only a cause file exists
            forward = e.run('R1', 'forward')
            npz = e.artifacts/f'oracle10-14b-{c}.npz'
            if rec['produced_sha256'].get(f'artifacts/increment10/oracle10-14b-{c}.npz') != sha256(npz):
                return {'verdict': FAIL}, False
            meta = e.json(npz.with_suffix('.json'))
            fresh = parity10.evaluate(forward, e.artifacts/'R1-capture', npz, meta)
            s = stored()
            same = (fresh['verdict'] == s['verdict'] and fresh['coverage'] == s['coverage']
                    and json.loads(json.dumps(fresh['comparison'])) == s['comparison']
                    and s['inputs']['forward_sha256'] == sha256(e.campaign/'R1-forward.json')
                    and s['inputs']['oracle_sha256'] == sha256(npz))
            return fresh, same
        if contract in GATING:
            lazy[f'14b.{contract}.recomputed_equals_stored'] = (lambda r: lambda: r()[1])(recompute)
            lazy[f'14b.{contract}.pass_full'] = (lambda r, c: lambda: (
                r()[0]['verdict'] == 'PASS' and r()[0]['coverage'] == 'FULL' and r()[0]['greedy_equal'] is True
                and r()[0]['comparison']['kinds']['logits']['elements'] == req['logits']
                and r()[0]['comparison']['kinds']['hidden']['elements'] == req['residuals']
                and r()[0]['comparison']['kinds']['kv']['elements'] == req['kv_final']
                and all(r()[0]['comparison']['kinds'][k]['failures'] == 0 for k in ('logits', 'hidden', 'kv'))
                and [x['length'] for x in r()[0]['kv_prefixes']] == [5, 6, 7, 8]
                and all(x['byte_identical_prefix'] for x in r()[0]['kv_prefixes'])
                and (r()[0]['comparison']['atol'], r()[0]['comparison']['rtol'])
                == THRESHOLDS['fp32' if c == 'stock_fp32' else 'fp64']))(recompute, contract)
        else:
            lazy[f'14b.{contract}.reported_raw'] = (lambda r: lambda: r()[1] and r()[0]['verdict'] in (PASS, FAIL, NOT_PROVEN))(recompute)
    c = f['checks'] = settle(lazy, f)
    raw = {}
    for contract in (*GATING, DIAGNOSTIC):
        try:
            s = e.file(f'parity-14b-{contract}.json')
            raw[contract] = dict(verdict=s['verdict'], coverage=s['coverage'],
                                 kinds={k: {x: v.get(x) for x in ('elements', 'failures', 'max_abs_error',
                                                                  'max_threshold_utilization')}
                                        for k, v in s['comparison']['kinds'].items()},
                                 top10_ordered_equal_positions=(s.get('top10') or {}).get('ordered_equal_positions'))
        except (Missing, *INCOMPLETE):
            raw[contract] = None
    f['parity_14b'] = raw
    e2e = {}
    for contract in (*GATING, DIAGNOSTIC):
        try:
            s = e.file(f'equivalence-7b-e2e-{contract}.json')
            e2e[contract] = dict(verdict=s['verdict'], run=e.latest(f'O7-{contract}'), elements=s.get('elements'),
                                 kinds={k: {x: v.get(x) for x in ('bit_identical', 'max_abs_difference')}
                                        for k, v in s['comparison']['kinds'].items()},
                                 logits_argmax_identical=s['comparison'].get('logits_argmax_identical'))
        except (Missing, *INCOMPLETE):
            e2e[contract] = None
    f['equivalence_7b'] = e2e
    return true(c)


def e2e_recomputed(e, contract, stored):
    """7B end-to-end equivalence recomputed from the capture files; both sides bound to their records
    (oracle10 capture = O7 run record produced_sha256; oracle9 capture = results/increment9/capture-manifest.json).
    Gating contracts must PASS; the diagnostic contract is reported raw (PASS or FAIL, recomputed == stored)."""
    from . import equivalence10_e2e as e2e
    new = e.artifacts/f'oracle10-7b-{contract}.npz'
    old = e.root/'artifacts'/'increment9'/f'oracle-{contract}.npz'
    rec = e.run(e.latest(f'O7-{contract}'))
    manifest = e.json(e.root/'results'/'increment9'/'capture-manifest.json')['files']
    if rec['produced_sha256'].get(f'artifacts/increment10/oracle10-7b-{contract}.npz') != sha256(new):
        return False
    if manifest[f'artifacts/increment9/oracle-{contract}.npz']['sha256'] != sha256(old):
        return False
    fresh = e2e.evaluate(new, e.json(new.with_suffix('.json')), old, e.json(old.with_suffix('.json')), e.protocol())
    allowed = (PASS,) if contract in GATING else (PASS, FAIL)
    return (fresh['verdict'] == stored['verdict'] and fresh['verdict'] in allowed
            and fresh['problems'] == stored['problems'] == []
            and stored['inputs']['new_sha256'] == sha256(new) and stored['inputs']['old_sha256'] == sha256(old)
            and json.loads(json.dumps(fresh['comparison'])) == stored['comparison'])


# Retries accepted by the gate: an earlier attempt with a run record is allowed only when an amendment
# names it (protocol stop condition "no repetition until pass"). Attempts never started (BLOCKED) are free.
JUSTIFIED_RETRIES = {'F-sync': ('004-campaign-and-gate-tooling.json', 'F-sync attempt 1')}


def retries_justified(e, f):
    families, ok = {}, True
    for path in e.campaign.glob('*-run.json'):
        name = path.name[:-len('-run.json')]
        base = name.rsplit('-a', 1)[0] if '-a' in name and name.rsplit('-a', 1)[1].isdigit() else name
        families.setdefault(base, set())
    for base in families:
        runs = e.attempts(base)
        families[base] = runs
        for earlier in runs[:-1]:
            amendment, needle = JUSTIFIED_RETRIES.get(earlier, (None, None))
            path = e.root/'results/increment10/amendments'/str(amendment)
            if amendment is None or not path.is_file() or needle not in path.read_text():
                ok = False
    f['attempts_with_records'] = {k: v for k, v in families.items() if len(v) > 1}
    return ok


def supervised(e, run, memory_max, planned_oom=False):
    sup, rec = e.run(run, 'scope'), e.run(run)
    fin = sup['finalizer']
    events = fin['memory.events']
    ok = dict(
        scope_limits=exactly(sup['memory_max'], memory_max) and zero(sup['swap_max'])
        and exactly(fin['memory.max'], memory_max) and zero(fin['memory.swap.max']),
        swap_zero=zero(fin['memory.swap.peak']),
        host_monitored=observer_ok(rec),
        host_start_go=rec['host_start']['verdict'] == 'GO',
        guest_swap_growth=host_values(e.run(run, 'host-start'))['guest_swap_growth_since_baseline']
        <= PARAMETERS['guest_swap_growth_max_bytes'],
        guest_swap_growth_at_end=(lambda b, a: natural(a['SwapTotal'], a['SwapFree'], b['SwapTotal'], b['SwapFree'])
                                  and (a['SwapTotal']-a['SwapFree']) - (b['SwapTotal']-b['SwapFree'])
                                  <= PARAMETERS['guest_swap_growth_max_bytes'])(
            e.file('hostmem-baseline.json')['after']['guest']['meminfo'], sup['global_after']),
        produced_files_bound=all(produced_bound(e, run, rec).values()))
    if planned_oom:
        ok.update(oom_attributed=natural(events['oom_kill']) and events['oom_kill'] >= 1
                  and sup['global_oom_kill_after'] - sup['global_oom_kill_before'] == events['oom_kill']
                  and fin['worker.exit'] == 137 and not (e.campaign/f'{run}-forward.json').exists())
    else:
        ok.update(no_oom=zero(events['oom'], events['oom_kill']) and zero(sup['exit_code'], fin['worker.exit'])
                  and sup['global_oom_kill_after'] == sup['global_oom_kill_before'],
                  peak_within_limit=natural(fin['memory.peak']) and fin['memory.peak'] <= memory_max)
    return ok


def condition_g(f, e):
    """Every run supervised, no unattributed OOM, guest swap growth <= 256 MiB, host recorded (L4b);
    faults N1-N4 fail-stop on the 14B in both modes; planned OOM N6 attributed."""
    lazy = {}
    for run in RUNS:
        lazy[run] = (lambda run: lambda: true(supervised(e, run, GIB2)))(run)
    for contract in GATING:
        o14 = (lambda c: lambda: e.latest(f'O14-{c}'))(contract)
        lazy[f'O14-{contract}'] = (lambda o: lambda: true(supervised(e, o(), GIB6)))(o14)
    for contract in GATING:
        o7 = (lambda c: lambda: e.latest(f'O7-{c}'))(contract)
        lazy[f'O7-{contract}'] = (lambda o: lambda: true(supervised(e, o(), GIB6)))(o7)
    for family in (f'O14-{DIAGNOSTIC}', f'O7-{DIAGNOSTIC}'):   # diagnostic oracles: only when recorded (007)
        if e.attempts(family):
            lazy[family] = (lambda run: lambda: true(supervised(e, run, GIB6)))(e.latest(family))
    flags = ('budget_rejected_before_payload', 'old_kv_all_bytes_preserved', 'descriptor_preserved',
             'candidate_was_modified', 'fail_stop_no_resume', 'drained', 'released')
    attempts = {}
    for mode in ('sync', 'pipeline'):
        run = e.latest(f'F-{mode}')
        attempts[mode] = sorted(p.name[:-len('-run.json')] for p in e.campaign.glob(f'F-{mode}*-run.json'))
        lazy[run] = (lambda run: lambda: true(supervised(e, run, GIB2)))(run)
        lazy[f'{run}.faults'] = (lambda run, mode: lambda: (
            (lambda r: r['mode'] == mode and r['rlimit_as_bytes'] == [RLIMIT[mode]]*2
             and sorted(x['fault'] for x in r['cases']) == ['cancel', 'io', 'short']
             and all(x[k] is True for x in r['cases'] for k in flags)
             and {x['fault']: x['caught'] for x in r['cases']} == {'io': 'OSError', 'short': 'ReadError',
                                                                 'cancel': 'CancelledError'}
             and output_argument(e.run(run, 'scope')['command']) == f'{run}-faults.json')(e.run(run, 'faults'))))(run, mode)
    lazy['N6.planned_oom'] = lambda: true(supervised(e, 'N6', 33554432, planned_oom=True))
    f['fault_attempts'] = attempts      # earlier failed attempts are kept and reported, never hidden

    def every_recorded_run():
        """Every run record of the campaign (also failed attempts and diagnostics): scope limits recorded by
        the kernel, swap 0, host monitored; OOM only in the planned N6. Exit codes are judged above."""
        rows = {}
        for path in sorted(e.campaign.glob('*-run.json')):
            run = path.name[:-len('-run.json')]
            sup, rec = e.run(run, 'scope'), e.run(run)
            fin = sup['finalizer']
            oom = fin['memory.events']['oom_kill']
            rows[run] = (natural(fin['memory.max']) and fin['memory.max'] == sup['memory_max'] and zero(fin['memory.swap.max'])
                         and zero(fin['memory.swap.peak']) and observer_ok(rec)
                         and (natural(oom) and oom >= 1 if run == 'N6' else zero(oom)))
        f['recorded_runs'] = rows
        if not rows:
            raise Missing('no run record in the campaign directory')
        return all(rows.values())
    lazy['every_recorded_run_supervised'] = every_recorded_run
    lazy['retries_justified'] = lambda: retries_justified(e, f)
    c = f['checks'] = settle(lazy, f)
    return true(c)


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, check=True, cwd=ROOT).stdout


def condition_h(f, e):
    """Inventory verified on the archived JUnit, protocol chain, gates 7/8/9 PASS, runtime and kernel
    unchanged since base_commit, no pre-existing file modified (beyond the amended workflow)."""
    from . import inventory10
    from .report_increment9 import rebuilt_gates, gate as gate9
    eff = protocol10.effective()
    junit = e.campaign/'pytest-final.xml'

    def inventory():
        if not junit.is_file():
            raise Missing('pytest-final.xml')
        inv_path = e.root/eff['test_inventory']['path']
        if sha256(inv_path) != eff['test_inventory']['sha256']:
            return False
        r = inventory10.verify(json.loads(inv_path.read_text()), junit, e.root)
        f['inventory'] = {k: r.get(k) for k in ('verdict', 'required_passed', 'required_total', 'fail_reasons',
                                                'not_proven_reasons')}
        if r['verdict'] == NOT_PROVEN:
            raise Missing('inventory verification NOT PROVEN: '+'; '.join(r['not_proven_reasons'][:3]))
        return r['verdict'] == PASS

    def gates789():
        g7, g8 = (json.loads(x) for x in rebuilt_gates(str(e.root)))
        g9 = gate9(e.root/'results/increment9', e.root)
        f['gate9'] = g9['verdicts']
        return (g7['passed'] is True and g8['passed'] is True and all(v == PASS for v in g9['verdicts'].values()))

    def preserved():
        changed = [n for n in git('diff', '--name-only', '--no-renames', '--diff-filter=MDT', BASE_COMMIT).split('\n') if n]
        allowed = {'.github/workflows/offline-cpu.yml'}   # amendment 001 (CI inventory step), recorded there
        f['modified_preexisting'] = changed
        return set(changed) <= allowed
    c = f['checks'] = settle(facts=f, lazy=dict(
        protocol_chain=lambda: protocol10.check()['passed'] is True and sha256(e.root/'results/increment10/protocol.json') == PROTOCOL_SHA256,
        inventory_on_archived_junit=inventory,
        gates_7_8_9=gates789,
        runtime_and_kernel_unchanged=lambda: git('diff', '--name-only', BASE_COMMIT, '--', 'inwime_runtime', 'src') == ''
        and git('status', '--porcelain', '--', 'inwime_runtime', 'src') == '',
        no_preexisting_file_modified=preserved))
    return true(c)


CONDITIONS = dict(A=condition_a, B=condition_b, C=condition_c, D=condition_d, E=condition_e, F=condition_f,
                  G=condition_g, H=condition_h)


def gate(e=None, only=None):
    e = e or Evidence()
    out = {}
    for name, function in CONDITIONS.items():
        if only and name not in only:
            continue
        facts = {}
        try:
            verdict = PASS if function(facts, e) else FAIL
        except Blocked as error:
            verdict, facts['blocked'] = BLOCKED, str(error)
        except (Missing, *INCOMPLETE) as error:
            verdict = NOT_PROVEN
            facts['missing_evidence' if isinstance(error, Missing) else 'incomplete_evidence'] = (
                str(error) if isinstance(error, Missing) else repr(error))
        except (*CONTRADICTED, json.JSONDecodeError, OSError) as error:
            verdict, facts['contradicted_evidence'] = FAIL, repr(error)
        checks = facts.get('checks')
        if isinstance(checks, dict) and any(v is False for v in checks.values()):
            verdict = FAIL
        out[name] = dict(verdict=verdict, facts=facts)
    verdicts = {k: v['verdict'] for k, v in out.items()}
    return dict(schema='inwime-gate10/1', protocol_sha256=PROTOCOL_SHA256, base_commit=BASE_COMMIT,
                parameters=PARAMETERS, verdicts=verdicts,
                overall=combine(*verdicts.values()) if verdicts else NOT_PROVEN, conditions=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', action='append')
    a = p.parse_args(argv)
    report = gate(only=a.only)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, default=str); f.write('\n')
    print(json.dumps(dict(overall=report['overall'], **report['verdicts']), indent=1))
    return 0 if report['overall'] == PASS else 1


if __name__ == '__main__':
    raise SystemExit(main())
