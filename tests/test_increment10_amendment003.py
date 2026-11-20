"""Increment 10, amendment 003 (Issue #25): the acquisition is not gated by host memory.

The amendment section acquisition_host_policy is validated by tools/protocol10.py:
it must state gated=false, name Issue #25, list what it supersedes, and its stop
conditions must be acquisition risks only (never RAM, paging, pagefile or memory).
Synthetic chains (pure, tmp root) prove that a weakened or re-gating section is
rejected; the repository chain proves the code (tools/acquire10.HOST_POLICY) and the
effective amendment agree. Historical amendments 001/002 stay in the chain unchanged.
"""
import copy
import hashlib
import json

import pytest

from tools import acquire10 as acq
from tools import protocol10
from tests.test_increment10_amendment import chain_root, sha256, write_amendment

POLICY = dict(
    gated=False, issue=25,
    rule='acquire10 never requires a host GO and never stops for host or guest memory pressure',
    supersedes=['001: hostgate10 GO required by acquire10 --execute',
                '002: in-run host monitor stopping the transfer'],
    acquisition_stop_conditions=['network', 'HTTP status', 'Range/Content-Range/Content-Length', 'checksum',
                                 'physical space', 'ENOSPC', 'I/O error', 'budget', 'lock', 'ownership'],
    diagnostics='optional and non-blocking: acquire10 --host-diagnostics BASELINE records hostmonitor10 samples',
    host_tools_kept_for=['inference', 'oracle', 'benchmark'])


def with_policy(tmp_path, policy):
    root, raw, protocol, amendment = chain_root(tmp_path)
    amendment['changes']['acquisition_host_policy'] = policy
    return root, raw, protocol, write_amendment(root, amendment)


def test_valid_host_policy_section_passes(tmp_path):
    root, raw, protocol, d = with_policy(tmp_path, copy.deepcopy(POLICY))
    assert protocol10.check_amendments(raw, protocol, d, root) == []


MUTATIONS = {
    're_gated': lambda p: p.update(gated=True),
    'gated_truthy_string': lambda p: p.update(gated='false'),
    'other_issue': lambda p: p.update(issue=24),
    'nothing_superseded': lambda p: p.update(supersedes=[]),
    'ram_stop_condition': lambda p: p['acquisition_stop_conditions'].append('host RAM below 300 MB'),
    'paging_stop_condition': lambda p: p['acquisition_stop_conditions'].append('host paging'),
    'checksum_dropped': lambda p: p['acquisition_stop_conditions'].remove('checksum'),
    'space_dropped': lambda p: p['acquisition_stop_conditions'].remove('physical space'),
    'missing_key': lambda p: p.pop('diagnostics'),
    'extra_key': lambda p: p.update(blocking_monitor=True),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_weakened_or_regating_host_policy_rejected(tmp_path, name):
    policy = copy.deepcopy(POLICY)
    MUTATIONS[name](policy)
    root, raw, protocol, d = with_policy(tmp_path, policy)
    problems = protocol10.check_amendments(raw, protocol, d, root)
    assert any('acquisition_host_policy malformed' in p for p in problems), problems


def test_repository_chain_has_amendment_003_and_code_agrees():
    result = protocol10.check()
    assert result['passed'], result['problems']
    assert result['amendments'][:3] == ['001-acquisition-host-gate-and-ci-inventory.json',
                                        '002-acquisition-in-run-host-monitor.json',
                                        '003-acquisition-without-host-memory-gate.json']
    eff = protocol10.effective()
    policy = eff['acquisition_host_policy']
    assert policy['gated'] is False and policy['issue'] == 25
    assert acq.HOST_POLICY['gated'] is False and acq.HOST_POLICY['issue'] == 25
    assert acq.HOST_POLICY['amendment'] == '003-acquisition-without-host-memory-gate'
    # historical policies stay in the chain (not rewritten): they remain readable for the run tooling
    assert eff['acquisition_host_gate']['profile'] == 'download'
    assert eff['acquisition_host_monitor']['criteria'] == 'host_memory_preflight.abort_criteria'


def test_historical_amendments_unchanged():
    """001 and 002 are byte-identical to their committed versions (digests recorded by 003)."""
    a3 = json.loads((protocol10.R10/'amendments'/'003-acquisition-without-host-memory-gate.json').read_text())
    for path, digest in a3['frozen_evidence_preserved'].items():
        assert sha256(protocol10.ROOT/path) == digest, path
    assert {'results/increment10/amendments/001-acquisition-host-gate-and-ci-inventory.json',
            'results/increment10/amendments/002-acquisition-in-run-host-monitor.json',
            'results/increment10/protocol.json'} <= set(a3['frozen_evidence_preserved'])
    assert hashlib.sha256((protocol10.R10/'protocol.json').read_bytes()).hexdigest() == a3['protocol_sha256']
