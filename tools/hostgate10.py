"""Pre-acquisition host-readiness gate for increment 10 (technical eligibility only).

Introduced by protocol amendment 001 (review of PR #24, observation 1). It decides
whether a hostmem10 report is a valid, *fresh* GO for the preregistered 'download'
profile, for this destination and this physical volume. It never decides whether
the owner allows the transfer: that is the separate authorization record checked by
tools/acquire10.py::check_authorization. Neither implies the other; --execute needs
both, and the report keeps them in separate sections.

Inputs: a hostmem10 report (profile 'download', taken with --baseline) and that
baseline report. Nothing in the report is trusted as a verdict:
  * the protocol must pass tools.protocol10 check (digest file, recomputation,
    pinned tool digests including the amendment chain);
  * the report must name the pinned protocol SHA256, the preregistered profile and
    sampling interval, and the SHA256 of the baseline file supplied;
  * both Windows samples must be present and error-free, the measured interval
    must match the preregistered one;
  * every quantity is recomputed from the raw snapshots and must equal the stored
    quantities; the verdict is recomputed with the protocol's start criteria
    (thresholds are read from protocol.json, never from the report);
  * the report must be fresh (max age from the amendment), the baseline must
    precede it and be from the same session (max baseline age);
  * the checkpoint and backing paths of report and baseline must resolve to the
    destination and backing volume of the acquisition, with matching volume sizes;
  * the protocol's acquisition manifest digest must equal the manifest acquired.

Verdicts: GO (the only one that allows a transfer), DEFER, NOT MEASURED (as in
hostmem10, recomputed) and REFUSED (report invalid, stale, for another protocol,
profile, destination or volume, or protocol chain not verified). Read only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

from . import hostmem10

SCHEMA = 'inwime-hostgate10/1'
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'results'/'increment10'/'protocol.json'
PROFILE = 'download'


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_utc(text):
    if not isinstance(text, str):
        raise ValueError('timestamp is not a string')
    return datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def existing_ancestor(path):
    p = Path(path).absolute()
    while not p.exists():
        p = p.parent
    return p


def same_path(recorded, actual, cwd):
    """hostmem10 records paths relative to its working directory; resolve them against cwd."""
    if not isinstance(recorded, str) or not recorded:
        return False
    return os.path.normpath(os.path.join(cwd, recorded)) == os.path.normpath(str(Path(actual).absolute()))


def normalised(value):
    return json.loads(json.dumps(value))


def evaluate(report_raw, baseline_raw, *, protocol, protocol_sha256, policy, manifest_digest, destination,
             backing, now, usage=shutil.disk_usage, cwd=None, protocol_check=None):
    """Pure evaluation. Returns a result dict; never raises for content problems."""
    cwd = os.getcwd() if cwd is None else cwd
    refused, notes = [], []
    result = dict(schema=SCHEMA, profile=PROFILE, scope='technical host readiness only; owner authorization '
                  'is a separate record and is not evaluated here', protocol_sha256=protocol_sha256)

    def done(verdict):
        result.update(verdict='REFUSED' if refused else verdict, reasons=refused or notes)
        return result

    if protocol_check is not None and not protocol_check.get('passed'):
        refused.append('protocol chain not verified: '+'; '.join(protocol_check.get('problems') or ['unknown']))
    if not isinstance(policy, dict) or policy.get('profile') != PROFILE:
        refused.append('no preregistered acquisition host gate policy for profile download (amendment 001)')
        return done('REFUSED')
    if (protocol.get('checkpoint') or {}).get('acquisition_manifest_sha256') != manifest_digest:
        refused.append('acquisition manifest differs from the protocol (different model, revision or files)')
    if report_raw is None:
        notes.append('host report missing')
        return done('NOT MEASURED')
    if baseline_raw is None:
        notes.append('baseline report missing')
        return done('NOT MEASURED')
    result.update(report_sha256=sha256_bytes(report_raw), baseline_sha256=sha256_bytes(baseline_raw))
    try:
        report, baseline = json.loads(report_raw), json.loads(baseline_raw)
    except ValueError:
        refused.append('host or baseline report is not valid JSON')
        return done('REFUSED')
    if not isinstance(report, dict) or not isinstance(baseline, dict):
        refused.append('host or baseline report is not a JSON object')
        return done('REFUSED')
    section = protocol.get('host_memory_preflight') or {}
    criteria = (section.get('start_criteria') or {}).get(PROFILE)
    interval = section.get('interval_s')
    if not isinstance(criteria, list) or not criteria or not isinstance(interval, (int, float)):
        refused.append('protocol lacks the download start criteria or the sampling interval')
        return done('REFUSED')

    # identity of the report
    for doc, label in ((report, 'report'), (baseline, 'baseline')):
        if doc.get('schema') != hostmem10.SCHEMA:
            refused.append(f'{label}: schema is not {hostmem10.SCHEMA}')
        if doc.get('read_only') is not True:
            refused.append(f'{label}: not marked read-only')
        if doc.get('protocol_sha256') != protocol_sha256:
            refused.append(f'{label}: taken with another protocol (SHA256 differs from the pinned one)')
    if report.get('profile') != PROFILE:
        refused.append(f"report profile is {report.get('profile')!r}, not 'download'")
    if report.get('interval_s') != interval:
        refused.append(f"report interval {report.get('interval_s')!r} s differs from the preregistered {interval} s")
    if report.get('baseline_used') is not True:
        refused.append('report was taken without --baseline')
    if report.get('baseline_sha256') != result['baseline_sha256']:
        refused.append('report does not name the SHA256 of the supplied baseline file')
    snaps = [report.get('before'), report.get('after'), baseline.get('after')]
    if not all(isinstance(s, dict) for s in snaps):
        refused.append('report or baseline lacks the raw snapshots')
        return done('REFUSED')
    before, after, base_after = snaps

    # freshness (all timestamps from the report's own snapshots)
    try:
        t_before, t_after, t_base = parse_utc(before.get('utc')), parse_utc(after.get('utc')), parse_utc(base_after.get('utc'))
    except ValueError:
        refused.append('missing or malformed sample timestamps')
        return done('REFUSED')
    age = (now - t_after).total_seconds()
    base_age = (now - t_base).total_seconds()
    result.update(report_utc=after.get('utc'), report_age_s=age, baseline_utc=base_after.get('utc'),
                  baseline_age_s=base_age)
    if age < -policy['max_future_skew_s'] or (now - t_before).total_seconds() < -policy['max_future_skew_s']:
        refused.append('report timestamps lie in the future')
    if age > policy['max_report_age_s']:
        refused.append(f"report is stale: {age:.0f} s old, limit {policy['max_report_age_s']} s")
    if t_after < t_before:
        refused.append('report snapshots out of order')
    if report.get('baseline_utc') != base_after.get('utc'):
        refused.append('report baseline_utc differs from the baseline file')
    if t_base > t_before:
        refused.append('baseline taken after the report')
    if base_age > policy['max_baseline_age_s']:
        refused.append(f"baseline is from another session: {base_age:.0f} s old, limit {policy['max_baseline_age_s']} s")

    # destination and physical volume
    for snap_, label in ((before, 'report.before'), (after, 'report.after'), (base_after, 'baseline')):
        cv, bv = snap_.get('checkpoint_volume') or {}, snap_.get('backing_volume') or {}
        if not same_path(cv.get('path_checked'), destination, cwd):
            refused.append(f'{label}: checkpoint volume path differs from the destination')
        if not same_path(bv.get('path_checked'), backing, cwd):
            refused.append(f'{label}: backing volume path differs from the acquisition backing volume')
    try:
        dest_total, back_total = usage(existing_ancestor(destination)).total, usage(backing).total
    except OSError as error:
        refused.append('cannot stat destination or backing volume: '+type(error).__name__)
    else:
        if (after.get('checkpoint_volume') or {}).get('total') != dest_total:
            refused.append('report checkpoint volume size differs from the destination volume')
        if (after.get('backing_volume') or {}).get('total') != back_total:
            refused.append('report backing volume size differs from the acquisition backing volume')

    # sampling: both Windows samples present, interval as preregistered
    w_b, w_a = before.get('windows'), after.get('windows')
    sampling_failed = [label for label, w in (('before', w_b), ('after', w_a)) if not isinstance(w, dict) or 'error' in w]
    if sampling_failed:
        notes.append('Windows sampling failed: '+', '.join(sampling_failed))
    else:
        try:
            dt = (w_a['Timestamp100ns'] - w_b['Timestamp100ns']) / w_a['Frequency100ns']
        except (KeyError, TypeError, ZeroDivisionError):
            dt = None
        result['measured_interval_s'] = dt
        low, high = interval*policy['interval_tolerance'][0], interval*policy['interval_tolerance'][1]
        if dt is None:
            notes.append('Windows sample timestamps unreadable')
        elif not low <= dt <= high:
            refused.append(f'measured sampling interval {dt:.2f} s outside [{low:.1f}, {high:.1f}] s')

    # recompute every quantity and the verdict from the raw snapshots and the protocol criteria
    quantities = hostmem10.measurements(before, after, base_after)
    if normalised(quantities) != report.get('quantities'):
        refused.append('stored quantities differ from the recomputation of the raw snapshots')
    def rules(checks):
        return sorted((str(c.get('quantity')), str(c.get('op')), repr(c.get('threshold')))
                      for c in checks if isinstance(c, dict))
    checks = report.get('checks')
    if not isinstance(checks, list) or rules(checks) != rules(criteria):
        refused.append('report criteria differ from the protocol download criteria')
    recomputed = hostmem10.evaluate(quantities, criteria)
    result.update(recomputed_verdict=recomputed['verdict'], violated=recomputed['violated'],
                  missing=recomputed['missing'],
                  values={c['quantity']: c['value'] for c in recomputed['checks']})
    if report.get('verdict') != recomputed['verdict']:
        refused.append(f"stored verdict {report.get('verdict')!r} differs from the recomputed {recomputed['verdict']!r}")
    if recomputed['verdict'] == 'DEFER':
        notes.append('violated: '+', '.join(recomputed['violated']))
    elif recomputed['verdict'] == 'NOT MEASURED':
        notes.append('not measured: '+', '.join(recomputed['missing']))
    if recomputed['verdict'] == 'DEFER':
        return done('DEFER')
    return done('NOT MEASURED' if sampling_failed or recomputed['verdict'] == 'NOT MEASURED' else 'GO')


def repository_protocol(protocol_path=PROTOCOL):
    """Load protocol.json, its pinned digest, the amendment chain check and the effective policy."""
    from . import protocol10
    raw = Path(protocol_path).read_bytes()
    check = protocol10.check(protocol_path)
    effective = protocol10.effective(protocol_path)
    return json.loads(raw), sha256_bytes(raw), effective.get('acquisition_host_gate'), check


class Gate:
    """Host-readiness gate bound to one report and one baseline file; re-read at every evaluation."""

    def __init__(self, report, baseline, *, loader=repository_protocol, clock=None, cwd=None):
        self.report, self.baseline = report, baseline
        self.loader = loader
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.cwd = cwd

    @staticmethod
    def _read(path):
        try:
            return Path(path).read_bytes() if path is not None else None
        except OSError:
            return None

    def evaluate(self, destination, backing, manifest, usage=shutil.disk_usage):
        try:
            protocol, digest, policy, check = self.loader()
        except Exception as error:          # an unreadable protocol is a refusal, never a pass
            return dict(schema=SCHEMA, profile=PROFILE, verdict='REFUSED',
                        reasons=['protocol unreadable: '+type(error).__name__])
        return evaluate(self._read(self.report), self._read(self.baseline), protocol=protocol, protocol_sha256=digest,
                        policy=policy, manifest_digest=manifest.digest(), destination=destination, backing=backing,
                        now=self.clock(), usage=usage, cwd=self.cwd, protocol_check=check)


def main(argv=None):
    from . import acquire10
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('destination', type=Path)
    p.add_argument('--backing-volume', type=Path, required=True)
    p.add_argument('--host-report', type=Path, required=True)
    p.add_argument('--host-baseline', type=Path, required=True)
    a = p.parse_args(argv)
    result = Gate(a.host_report, a.host_baseline).evaluate(a.destination, a.backing_volume, acquire10.QWEN25_14B)
    print(json.dumps(result, indent=1))
    return 0 if result['verdict'] == 'GO' else 3


if __name__ == '__main__':
    sys.exit(main())
