"""Increment 10, amendment 001 (gate) and amendment 003 (no longer wired into the acquisition).

tools/hostgate10.py is exercised with synthetic hostmem10 reports built by the real
hostmem10.make_report, against the real protocol (thresholds, interval, pinned
digest) and the amendment-001 policy: every verdict (GO, DEFER, NOT MEASURED,
REFUSED) is still computed and asserted, because the gate stays available.

Amendment 003 (Issue #25) removed the gate from tools/acquire10.py::execute: a
download is a streaming Internet -> ext4 transfer, not an inference run. Where these
tests used to assert "refused before any request", they now assert the opposite
half of the decision: the acquisition completes against the local fake HTTP server
whatever the gate verdict, and never calls the gate. Owner authorization is still
required and checked before any request. Node IDs are unchanged (inventory policy);
the bodies changed and are pinned by the amended inventory.
"""
from collections import namedtuple
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from tools import acquire10 as acq
from tools import hostgate10 as hg
from tools import hostmem10 as hm
from tools import protocol10
from tests.test_increment10_acquire import FakeHF, authorize, build_repo, small_reserves
from tests.support_increment10_monitor import healthy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT/'results'/'increment10'/'protocol.json'
PROTOCOL_RAW = PROTOCOL_PATH.read_bytes()
PROTOCOL = json.loads(PROTOCOL_RAW)
PROTOCOL_SHA = hashlib.sha256(PROTOCOL_RAW).hexdigest()
POLICY = protocol10.effective()['acquisition_host_gate']
GiB, MiB = 2**30, 2**20
Usage = namedtuple('Usage', 'total used free')
DEST_TOTAL, BACK_TOTAL = 1_081_101_176_832, 1_798_804_860_928
T0 = datetime(2026, 11, 18, 21, 0, 0, tzinfo=timezone.utc)


def utc(t):
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


def win(available=4*GiB, pf_mb=7000, pages_in=0, ts=0):
    return dict(TotalPhysicalMemory=16_900_739_072, DimmBytes=17_179_869_184, AvailableBytes=available,
                CommittedBytes=40*GiB, CommitLimit=50*GiB, PagesInput=pages_in, PagesOutput=0,
                PageReads=pages_in//4, Timestamp100ns=ts, Frequency100ns=10_000_000,
                PageFiles=[dict(Name='C:\\pagefile.sys', AllocatedBaseSize=34816, CurrentUsage=pf_mb,
                                PeakUsage=13799)], CFreeBytes=150*10**9)


def snap(t, dest, backing, ts=0, available=4*GiB, pages_in=0, pf_mb=7000, windows_error=False,
         back_free=150*10**9, dest_total=DEST_TOTAL, back_total=BACK_TOTAL):
    return dict(utc=utc(t), guest=dict(meminfo=dict(MemTotal=11_483_848_704, MemAvailable=7*GiB,
                                                    SwapTotal=8*GiB, SwapFree=8*GiB-500*MiB),
                                       psi=dict(some=dict(avg10=0.0))),
                windows={'error': 'powershell failed or unparsable output'} if windows_error else
                win(available, pf_mb, pages_in, ts),
                checkpoint_volume=dict(path_checked=str(dest), total=dest_total, free=10**12),
                backing_volume=dict(path_checked=str(backing), total=back_total, free=back_free))


def write_reports(tmp_path, dest, backing, *, profile='download', protocol_sha=PROTOCOL_SHA, interval=30.0,
                  start=T0, baseline_time=None, available=4*GiB, pages_in_rate=0, windows_error=False,
                  back_free=150*10**9, dt=30.6, criteria=None, report_dest=None, report_backing=None,
                  baseline_dest=None):
    """A baseline and a report exactly as hostmem10.main would write them (make_report)."""
    criteria = criteria or PROTOCOL['host_memory_preflight']['start_criteria'][profile]
    bt = baseline_time or start - timedelta(minutes=30)
    b_before = snap(bt, baseline_dest or dest, backing)
    b_after = snap(bt, baseline_dest or dest, backing, ts=int(dt*10**7))
    baseline = hm.make_report(b_before, b_after, None, criteria, profile, interval, protocol_sha)
    baseline_path = tmp_path/'hostmem-baseline.json'
    baseline_path.write_text(json.dumps(baseline, indent=2))
    rd, rb = report_dest or dest, report_backing or backing
    before = snap(start, rd, rb, available=available, windows_error=windows_error, back_free=back_free)
    after = snap(start + timedelta(seconds=round(dt)), rd, rb, ts=int(dt*10**7), available=available,
                 pages_in=int(pages_in_rate*dt), windows_error=windows_error, back_free=back_free)
    report = hm.make_report(before, after, baseline_path.read_bytes(), criteria, profile, interval, protocol_sha)
    report_path = tmp_path/'hostmem-download.json'
    report_path.write_text(json.dumps(report, indent=2))
    return report_path, baseline_path


def loader_for(manifest, protocol_check=None, policy=None):
    """Real protocol and amended policy; the synthetic manifest replaces the 14B digest."""
    protocol = json.loads(PROTOCOL_RAW)
    protocol['checkpoint']['acquisition_manifest_sha256'] = manifest.digest()
    return lambda: (protocol, PROTOCOL_SHA, POLICY if policy is None else policy,
                    protocol_check or dict(passed=True, problems=[]))


def gate(report, baseline, manifest, now=None, **kw):
    return hg.Gate(report, baseline, loader=kw.pop('loader', None) or loader_for(manifest),
                   clock=lambda: now or T0 + timedelta(seconds=120), **kw)


@pytest.fixture
def setup(tmp_path):
    manifest, files = build_repo()
    fake = FakeHF(manifest, files)
    dest, backing = tmp_path/'ckpt', tmp_path/'back'
    backing.mkdir()
    yield manifest, fake, dest, backing
    fake.close()


def usage_for(dest, backing, dest_total=DEST_TOTAL, back_total=BACK_TOTAL):
    def usage(p):
        return Usage(back_total if Path(p).absolute() == backing.absolute() else dest_total, 0, 10**15)
    return usage


class NeverCalled:
    """A gate that fails the test if the acquisition consults it (amendment 003)."""

    def __init__(self, gate=None):
        self.gate = gate

    def evaluate(self, *args):
        raise AssertionError('acquire10 consulted a host gate')


def run_execute(fake, manifest, dest, backing, auth, host_gate, usage=None):
    """Amendment 003: the gate is not an argument of execute any more; it is never consulted."""
    del host_gate
    return acq.execute(dest, backing, auth, manifest=manifest, endpoint=fake.endpoint,
                       transport=acq.Transport(allow_loopback_http=True), reserves=small_reserves(),
                       usage=usage or usage_for(dest, backing))


def assert_refused_before_network(fake, dest, call, match=None):
    with pytest.raises(acq.AcquireError, match=match):
        call()
    assert fake.requests == [] and not dest.exists()


def assert_not_gated(fake, manifest, dest, backing, auth, verdict):
    """Whatever the host gate says (verdict), a correct acquisition completes (amendment 003)."""
    out = run_execute(fake, manifest, dest, backing, auth, NeverCalled())
    assert out['payload_bytes_transferred'] == manifest.total_bytes()
    assert out['host_policy']['gated'] is False and 'host_readiness' not in out


# ---------------------------------------------------------------- policy and GO path

def test_amended_policy_is_preregistered():
    assert POLICY == dict(profile='download', max_report_age_s=900, max_baseline_age_s=43200,
                          max_future_skew_s=60, interval_tolerance=[0.95, 1.5],
                          require_recomputed_verdict='GO', rule=POLICY['rule'])
    # thresholds stay in protocol.json (frozen); the amendment adds no threshold of its own
    assert protocol10.check()['passed']


def test_make_report_names_the_baseline_file(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report_path, baseline_path = write_reports(tmp_path, dest, backing)
    report = json.loads(report_path.read_text())
    assert report['baseline_sha256'] == hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert report['baseline_utc'] == json.loads(baseline_path.read_text())['after']['utc']


def test_fresh_go_allows_execute_with_owner_authorization(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    g = gate(report, baseline, manifest)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'GO' and result['reasons'] == [], result
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    out = run_execute(fake, manifest, dest, backing, auth, g)
    assert out['payload_bytes_transferred'] == manifest.total_bytes()
    assert out['host_policy']['gated'] is False and 'host_readiness' not in out      # amendment 003
    assert set(out['authorization']) == {'approved_by', 'approved_utc', 'max_transfer_bytes'}


# ---------------------------------------------------------------- DEFER and NOT MEASURED

@pytest.mark.parametrize('kw,violated', [(dict(pages_in_rate=5000), 'host_pages_input_per_s'),
                                         (dict(available=400*MiB), 'host_available_bytes'),
                                         (dict(back_free=10**9), 'backing_volume_free')])
def test_defer_refuses_before_any_request(tmp_path, setup, kw, violated):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, **kw)
    g = gate(report, baseline, manifest)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'DEFER' and violated in result['violated']
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, result['verdict'])


def test_not_measured_when_windows_sampling_failed(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, windows_error=True)
    g = gate(report, baseline, manifest)
    assert g.evaluate(dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'NOT MEASURED'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, 'NOT MEASURED')


@pytest.mark.parametrize('missing', ['report', 'baseline'])
def test_missing_report_or_baseline_is_not_measured(tmp_path, setup, missing):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    (report if missing == 'report' else baseline).unlink()
    g = gate(report, baseline, manifest)
    assert g.evaluate(dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'NOT MEASURED'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, 'NOT MEASURED')


def test_report_without_baseline_is_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    data = json.loads(report.read_text())
    criteria = PROTOCOL['host_memory_preflight']['start_criteria']['download']
    plain = hm.make_report(data['before'], data['after'], None, criteria, 'download', 30.0, PROTOCOL_SHA)
    report.write_text(json.dumps(plain))
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED'
    assert any('without --baseline' in r for r in result['reasons'])


# ---------------------------------------------------------------- stale or out-of-session samples

@pytest.mark.parametrize('now,needle', [
    (T0 + timedelta(seconds=31 + 901), 'stale'),
    (T0 - timedelta(seconds=120), 'future'),
])
def test_stale_or_future_report_refused(tmp_path, setup, now, needle):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    g = gate(report, baseline, manifest, now=now)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any(needle in r for r in result['reasons']), result
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, result['verdict'])


@pytest.mark.parametrize('baseline_time,needle', [(T0 - timedelta(hours=13), 'another session'),
                                                  (T0 + timedelta(seconds=5), 'after the report')])
def test_baseline_out_of_session_refused(tmp_path, setup, baseline_time, needle):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, baseline_time=baseline_time)
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any(needle in r for r in result['reasons']), result


def test_report_goes_stale_between_network_check_and_payload(tmp_path, setup):
    """Gate level: a report ages out between two evaluations. Amendment 003: the acquisition does not
    evaluate the gate at all, so a stale report cannot stop it (and the gate is never called)."""
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    times = iter([T0 + timedelta(seconds=60), T0 + timedelta(seconds=2000)])
    g = hg.Gate(report, baseline, loader=loader_for(manifest), clock=lambda: next(times))
    assert g.evaluate(dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'GO'
    stale = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert stale['verdict'] == 'REFUSED' and any('stale' in r for r in stale['reasons'])
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, stale['verdict'])


def test_interval_measured_outside_tolerance_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, dt=5.0)
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('sampling interval' in r for r in result['reasons'])


# ---------------------------------------------------------------- wrong protocol

def test_report_from_another_protocol_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, protocol_sha='0'*64)
    g = gate(report, baseline, manifest)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('another protocol' in r for r in result['reasons'])
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, result['verdict'])


@pytest.mark.parametrize('profile', ['runtime', 'oracle'])
def test_other_profile_refused_even_if_go(tmp_path, setup, profile):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, profile=profile)
    assert json.loads(report.read_text())['verdict'] == 'GO'
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('profile' in r for r in result['reasons'])


def test_lowered_thresholds_in_report_refused(tmp_path, setup):
    """A report evaluated with relaxed criteria (e.g. a locally edited protocol) is not a GO."""
    manifest, fake, dest, backing = setup
    relaxed = [dict(c, threshold=10**9) if c['quantity'] == 'host_pages_input_per_s' else c
               for c in PROTOCOL['host_memory_preflight']['start_criteria']['download']]
    report, baseline = write_reports(tmp_path, dest, backing, criteria=relaxed, pages_in_rate=5000)
    assert json.loads(report.read_text())['verdict'] == 'GO'
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and result['recomputed_verdict'] == 'DEFER'


def test_unverified_protocol_chain_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    loader = loader_for(manifest, protocol_check=dict(passed=False, problems=['pinned tool differs']))
    g = gate(report, baseline, manifest, loader=loader)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('protocol chain' in r for r in result['reasons'])
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, result['verdict'])


def test_missing_policy_or_unreadable_protocol_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    no_policy = gate(report, baseline, manifest, loader=loader_for(manifest, policy={}))
    assert no_policy.evaluate(dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'REFUSED'

    def broken():
        raise FileNotFoundError('protocol.json')
    assert gate(report, baseline, manifest, loader=broken).evaluate(
        dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'REFUSED'


def test_real_protocol_binds_the_14b_manifest(tmp_path, setup):
    """Default loader (repository protocol + amendment chain): GO only for the pinned 14B manifest."""
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    real = hg.Gate(report, baseline, clock=lambda: T0 + timedelta(seconds=120))
    assert real.evaluate(dest, backing, acq.QWEN25_14B, usage_for(dest, backing))['verdict'] == 'GO'
    other = real.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert other['verdict'] == 'REFUSED' and any('manifest' in r for r in other['reasons'])


# ---------------------------------------------------------------- falsified reports

def test_falsified_quantity_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, pages_in_rate=5000)
    data = json.loads(report.read_text())
    data['quantities']['host_pages_input_per_s']['value'] = 10.0
    data.update(verdict='GO', violated=[])
    report.write_text(json.dumps(data))
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and result['recomputed_verdict'] == 'DEFER'
    assert any('recomputation' in r for r in result['reasons'])


def test_verdict_field_alone_edited_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, available=400*MiB)
    data = json.loads(report.read_text())
    data['verdict'] = 'GO'
    report.write_text(json.dumps(data))
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('stored verdict' in r for r in result['reasons'])


def test_swapped_baseline_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    data = json.loads(baseline.read_text())
    data['note'] = 'another baseline'
    baseline.write_text(json.dumps(data))
    result = gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('SHA256 of the supplied baseline' in r for r in result['reasons'])


@pytest.mark.parametrize('text', ['{not json', '[]'])
def test_malformed_report_refused(tmp_path, setup, text):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    report.write_text(text)
    assert gate(report, baseline, manifest).evaluate(dest, backing, manifest,
                                                     usage_for(dest, backing))['verdict'] == 'REFUSED'


# ---------------------------------------------------------------- destination and volume

@pytest.mark.parametrize('which', ['report_dest', 'report_backing', 'baseline_dest'])
def test_destination_or_backing_path_mismatch_refused(tmp_path, setup, which):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing, **{which: tmp_path/'elsewhere'})
    g = gate(report, baseline, manifest)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing))
    assert result['verdict'] == 'REFUSED' and any('path differs' in r for r in result['reasons']), result
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, result['verdict'])


@pytest.mark.parametrize('kw', [dict(dest_total=DEST_TOTAL + 4096), dict(back_total=BACK_TOTAL - 4096)])
def test_volume_size_mismatch_refused(tmp_path, setup, kw):
    """Same path text, different filesystem (e.g. another mount at the same place): refused."""
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    g = gate(report, baseline, manifest)
    result = g.evaluate(dest, backing, manifest, usage_for(dest, backing, **kw))
    assert result['verdict'] == 'REFUSED' and any('volume size' in r for r in result['reasons'])


def test_authorized_volume_smaller_than_plan_refused_even_with_go(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    auth = authorize(tmp_path/'auth.json', manifest, dest, max_bytes=manifest.total_bytes() - 1)
    assert_refused_before_network(fake, dest, lambda: run_execute(fake, manifest, dest, backing, auth,
                                                                  gate(report, baseline, manifest)), 'exceeds')


# ---------------------------------------------------------------- separation from owner authorization

def test_go_without_owner_authorization_refused(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    g = gate(report, baseline, manifest)
    assert g.evaluate(dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'GO'
    assert_refused_before_network(fake, dest, lambda: run_execute(fake, manifest, dest, backing,
                                                                  tmp_path/'absent-auth.json', g), 'authorization')
    for field, value in (('approved_by', ''), ('approved_utc', None)):
        auth = authorize(tmp_path/f'auth-{field}.json', manifest, dest, **{field: value})
        assert_refused_before_network(fake, dest, lambda: run_execute(fake, manifest, dest, backing, auth, g))


def test_owner_authorization_without_gate_refused(tmp_path, setup):
    """Inverted by amendment 003: owner authorization alone is sufficient; no host gate is needed."""
    manifest, fake, dest, backing = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, None)


@pytest.mark.parametrize('result', [None, {}, dict(schema='other', profile='download', verdict='GO', reasons=[]),
                                    dict(schema=hg.SCHEMA, profile='runtime', verdict='GO', reasons=[]),
                                    dict(schema=hg.SCHEMA, profile='download', verdict='GO', reasons=['x']),
                                    dict(schema=hg.SCHEMA, profile='download', verdict='go', reasons=[])])
def test_malformed_gate_result_refused(tmp_path, setup, result):
    manifest, fake, dest, backing = setup

    class Fake:
        def evaluate(self, *args):
            return result
    # gate level: hostgate10 consumers must still reject a malformed result as not GO
    r = Fake().evaluate()
    assert not (isinstance(r, dict) and r.get('schema') == hg.SCHEMA and r.get('profile') == 'download'
                and r.get('verdict') == 'GO' and not r.get('reasons'))
    # amendment 003: the acquisition never reads a gate result, malformed or not
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    assert_not_gated(fake, manifest, dest, backing, auth, None)


def test_cli_execute_requires_both_records(tmp_path):
    """Amendment 003: --execute requires the owner record only; --host-report/--host-baseline were removed
    from acquire10 and are rejected as unknown options (no silent acceptance of a superseded policy)."""
    dest = tmp_path/'ckpt'
    auth = tmp_path/'auth.json'
    auth.write_text('{}')
    for extra in ([], ['--host-report', 'r.json'], ['--host-baseline', 'b.json'],
                  ['--host-report', 'r.json', '--host-baseline', 'b.json']):
        argv = [str(dest), '--backing-volume', str(tmp_path), '--execute', *extra]
        if extra and len(extra) < 4:
            argv += ['--authorization', str(auth)]
        with pytest.raises(SystemExit):
            acq.main(argv)
    assert not dest.exists()


def test_dry_run_reports_host_readiness_separately(tmp_path, setup):
    manifest, fake, dest, backing = setup
    report, baseline = write_reports(tmp_path, dest, backing)
    kw = dict(manifest=manifest, endpoint=fake.endpoint, reserves=small_reserves(), usage=usage_for(dest, backing))
    # the gate verdicts stay computable on their own ...
    assert gate(report, baseline, manifest).evaluate(dest, backing, manifest, usage_for(dest, backing))['verdict'] == 'GO'
    report2, baseline2 = write_reports(tmp_path, dest, backing, pages_in_rate=5000)
    assert gate(report2, baseline2, manifest).evaluate(dest, backing, manifest,
                                                       usage_for(dest, backing))['verdict'] == 'DEFER'
    # ... and the dry-run reports storage readiness only (amendment 003), with the host policy stated
    d = acq.dry_run(dest, backing, transport=acq.Transport(allow_loopback_http=True), **kw)
    assert d['storage_ready'] and d['would_proceed_if_authorized'] and 'host_readiness' not in d
    assert d['host_policy']['gated'] is False and 'not evaluated' in d['owner_authorization']
    assert fake.payload_gets() == [] and not dest.exists()
