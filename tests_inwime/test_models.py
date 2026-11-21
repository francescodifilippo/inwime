"""Seam: `inwime models install/import/list/use` as a user runs them, against a fake Hub.

Same harness as the catalog tests (conftest.py): guarded subprocess, private HOME, fresh
Hugging Face cache and InWiMe registry. The physical free space is faked (fakestorage.py)
unless a test exercises the real host probe; the margin is always passed explicitly or left
at its 30 GiB default.
"""
import hashlib
import json
import stat

import hubmodels as hm

GIB = 1024**3
PLENTY = 500*GIB
REPO = 'acme/tiny-qwen2'


def cache(home):
    return home/'.cache/huggingface/hub'


def snapshot(home, repo_id, revision):
    return cache(home)/('models--'+repo_id.replace('/', '--'))/'snapshots'/revision


def manifest(home, repo_id, revision):
    path = home/'.local/share/inwime/installed'/('%s@%s.json' % (repo_id.replace('/', '--'), revision))
    return json.loads(path.read_text())


def weights_fetched(hub):
    """Weight contents that travelled. The summary itself reads the small safetensors index
    into memory to choose the files, so "nothing downloaded" is checked on the weights and on
    the cache."""
    return [r['path'].rsplit('/', 1)[1] for r in hub.downloads() if r['path'].endswith('.safetensors')]


def add_tiny(hub, repo_id=REPO, n=1, **kwargs):
    files = hm.checkpoint(hm.QWEN2, shards=2)
    hub.add(hm.repo(repo_id, files, revision=hm.sha(n), **kwargs))
    return files


def test_confirmed_install_downloads_verifies_and_records_the_checkpoint(hub, run, home):
    files = add_tiny(hub)
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 0, done.stderr
    # The summary names the resolved full SHA before asking.
    summary = done.stdout[:done.stdout.index("Type 'yes'")]
    assert hm.sha(1) in summary and 'main' in summary
    assert str(sum(map(len, files.values()))) in summary
    snap = snapshot(home, REPO, hm.sha(1))
    for name, data in files.items():
        assert (snap/name).read_bytes() == data
    m = manifest(home, REPO, hm.sha(1))
    assert m['schema'] == 'inwime-install-manifest/1'
    assert m['status'] == 'verified'
    assert (m['repo_id'], m['revision'], m['requested_ref']) == (REPO, hm.sha(1), 'main')
    assert m['endpoint'] == hub.endpoint
    recorded = {f['name']: f for f in m['files']}
    assert set(recorded) == set(files)
    for name, data in files.items():
        assert recorded[name]['outcome'] == 'verified'
        assert recorded[name]['size'] == len(data)
        assert recorded[name]['sha256'] == hashlib.sha256(data).hexdigest()
    assert recorded['model-00001-of-00002.safetensors']['kind'] == 'lfs'
    assert recorded['config.json']['kind'] == 'git'


def test_declined_or_missing_confirmation_downloads_nothing(hub, run, home):
    add_tiny(hub)
    for answer in ('no\n', '', 'yes please\n'):
        done = run('install', REPO, input=answer, free=PLENTY)
        assert done.returncode == 9, done.stderr
        assert 'not confirmed' in done.stderr and hm.sha(1) in done.stdout
    assert weights_fetched(hub) == [] and not cache(home).exists()
    assert not (home/'.local/share/inwime/installed').exists()
    assert not snapshot(home, REPO, hm.sha(1)).exists()


def test_yes_is_accepted_only_with_a_full_sha(hub, run, home):
    add_tiny(hub)
    done = run('install', REPO, '--yes', free=PLENTY)
    assert done.returncode == 2 and 'full 40-hex' in done.stderr
    done = run('install', REPO+'@main', '--yes', free=PLENTY)
    assert done.returncode == 2
    assert hub.requests == []
    done = run('install', REPO+'@'+hm.sha(1), '--yes', free=PLENTY)
    assert done.returncode == 0, done.stderr
    assert "Type 'yes'" not in done.stdout and hm.sha(1) in done.stdout
    assert manifest(home, REPO, hm.sha(1))['status'] == 'verified'


def test_download_refused_when_less_than_the_margin_would_remain(hub, run, home):
    files = add_tiny(hub)
    size = sum(map(len, files.values()))
    # Exactly the margin left after the download is accepted; one byte less is refused.
    done = run('install', REPO, '--host-margin-gib', '1', input='no\n', free=GIB+size)
    assert done.returncode == 9, done.stderr
    done = run('install', REPO, '--host-margin-gib', '1', input='yes\n', free=GIB+size-1)
    assert done.returncode == 10 and 'margin' in done.stderr
    assert "Type 'yes'" not in done.stdout             # refused before asking
    # Default margin: 30 GiB.
    done = run('install', REPO, input='yes\n', free=30*GIB+size-1)
    assert done.returncode == 10
    assert weights_fetched(hub) == [] and not cache(home).exists()
    assert not (home/'.local/share/inwime/installed').exists()


def test_unknown_physical_space_refuses_the_download(hub, run, home):
    add_tiny(hub)
    done = run('install', REPO, input='yes\n', free='unknown')
    assert done.returncode == 10 and 'unknown' in done.stderr and 'fail-closed' in done.stderr
    assert weights_fetched(hub) == [] and not cache(home).exists()


def test_space_is_checked_again_after_confirmation(hub, run, home, tmp_path):
    add_tiny(hub)
    log = tmp_path/'measures.log'
    done = run('install', REPO, input='yes\n', free=[PLENTY, 1*GIB],
               env={'INWIME_FAKE_FREE_LOG': str(log)})
    assert done.returncode == 10 and 'margin' in done.stderr
    assert "Type 'yes'" in done.stdout
    assert log.read_text().split() == [str(PLENTY), str(GIB)]
    assert weights_fetched(hub) == [] and not cache(home).exists()


def test_space_is_checked_before_every_file(hub, run, home):
    add_tiny(hub)
    # Summary, after confirmation (first file), second file; then the volume fills up.
    done = run('install', REPO, input='yes\n', free=[PLENTY, PLENTY, PLENTY, 1*GIB])
    assert done.returncode == 10, done.stderr
    assert sorted(p.name for p in snapshot(home, REPO, hm.sha(1)).iterdir()) == ['README.md', 'config.json']
    assert manifest(home, REPO, hm.sha(1))['status'] == 'failed'


def test_real_host_probe_refuses_an_impossible_margin(hub, run, home):
    add_tiny(hub)
    done = run('install', REPO, '--host-margin-gib', str(2**40), input='yes\n')
    assert done.returncode == 10, done.stderr       # insufficient, or unknown on this host
    assert weights_fetched(hub) == [] and not cache(home).exists()


WEIGHT = 'model-00001-of-00002.safetensors'


def blobs(home, repo_id=REPO):
    folder = cache(home)/('models--'+repo_id.replace('/', '--'))/'blobs'
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_sha256_mismatch_fails_the_install_and_leaves_no_valid_checkpoint(hub, run, home):
    files = add_tiny(hub)
    hub.corrupt.add(WEIGHT)
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 12, done.stderr
    assert WEIGHT in done.stderr and 'SHA256' in done.stderr
    m = manifest(home, REPO, hm.sha(1))
    assert m['status'] == 'failed'
    bad = next(f for f in m['files'] if f['name'] == WEIGHT)
    assert bad['outcome'] == 'mismatch'
    assert bad['expected']['sha256'] == hashlib.sha256(files[WEIGHT]).hexdigest() != bad['sha256']
    # The corrupt blob it downloaded is removed from the cache, so nothing can load it.
    assert hashlib.sha256(files[WEIGHT]).hexdigest() not in blobs(home)
    assert not (snapshot(home, REPO, hm.sha(1))/WEIGHT).exists()
    # Once the Hub serves the right bytes, the install completes.
    hub.corrupt.clear()
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 0, done.stderr
    assert manifest(home, REPO, hm.sha(1))['status'] == 'verified'
    assert (snapshot(home, REPO, hm.sha(1))/WEIGHT).read_bytes() == files[WEIGHT]


def test_wrong_git_blob_id_of_a_small_file_fails_the_install(hub, run, home):
    add_tiny(hub)
    hub.corrupt.add('config.json')
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 12 and 'config.json' in done.stderr and 'git blob id' in done.stderr
    assert manifest(home, REPO, hm.sha(1))['status'] == 'failed'


SECOND = 'model-00002-of-00002.safetensors'


def test_interrupted_install_resumes_without_downloading_completed_files(hub, run, home):
    files = add_tiny(hub)
    hub.fail[SECOND] = 100
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 7, done.stderr
    assert manifest(home, REPO, hm.sha(1))['status'] == 'failed'
    assert weights_fetched(hub)[0] == WEIGHT
    completed = {f['name'] for f in manifest(home, REPO, hm.sha(1))['files']}
    assert WEIGHT in completed and SECOND not in completed
    hub.fail.clear()
    hub.requests.clear()
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 0, done.stderr
    assert 'already cached:  %d files' % len(completed) in done.stdout
    # Only what was missing travels again; completed files are verified, not re-downloaded.
    assert weights_fetched(hub) == [SECOND]
    assert not any(r['path'].endswith('/'+name) for r in hub.downloads() for name in completed
                   if name != 'model.safetensors.index.json')
    m = manifest(home, REPO, hm.sha(1))
    assert m['status'] == 'verified'
    actions = {f['name']: f['action'] for f in m['files']}
    assert actions[WEIGHT] == 'reused' and actions[SECOND] == 'downloaded'
    assert all(f['outcome'] == 'verified' for f in m['files'])


def test_broken_transfer_resumes_with_a_byte_range(hub, run, home):
    # huggingface_hub writes in 10 MiB chunks: the file must exceed two of them for the cut,
    # halfway through, to land after a written chunk.
    shapes = dict(hm.tensors(hm.QWEN2), **{'model.extra.weight': (12*2**20,)})
    files = hm.checkpoint(hm.QWEN2, shapes=shapes)
    hub.add(hm.repo(REPO, files, revision=hm.sha(1)))
    hub.cut['model.safetensors'] = 1
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 0, done.stderr
    ranges = [r['range'] for r in hub.downloads('model.safetensors')]
    assert len(ranges) == 2 and ranges[0] is None
    offset = int(ranges[1][len('bytes='):-1])
    assert ranges[1].endswith('-') and 0 < offset < len(files['model.safetensors'])
    assert (snapshot(home, REPO, hm.sha(1))/'model.safetensors').read_bytes() == files['model.safetensors']


def test_concurrent_installs_are_refused_while_one_runs(hub, run, home):
    import threading
    import time
    add_tiny(hub)
    add_tiny(hub, 'acme/other', n=2)
    hub.hold[WEIGHT] = gate = threading.Event()
    first = run.start('install', REPO+'@'+hm.sha(1), '--yes', free=PLENTY)
    try:
        deadline = time.monotonic()+60
        while not hub.downloads(WEIGHT):
            assert first.poll() is None and time.monotonic() < deadline, first.communicate()
            time.sleep(0.05)
        for spec in (REPO+'@'+hm.sha(1), 'acme/other@'+hm.sha(2)):
            done = run('install', spec, '--yes', free=PLENTY)
            assert done.returncode == 11, done.stderr
            assert 'another install is running' in done.stderr and REPO in done.stderr
    finally:
        gate.set()
        out, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert manifest(home, REPO, hm.sha(1))['status'] == 'verified'
    assert not snapshot(home, 'acme/other', hm.sha(2)).exists()


SECRET = 'hf_secretTOKEN1234567890'


def registry_text(home):
    root = home/'.local/share/inwime'
    return ''.join(p.read_text() for p in root.rglob('*') if p.is_file())


def test_gated_access_problems_are_reported_before_confirmation(hub, run, home):
    add_tiny(hub, gated='manual', granted={SECRET})
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 4 and 'hf auth login' in done.stderr
    done = run('install', REPO, input='yes\n', free=PLENTY, token='hf_someoneelse')
    assert done.returncode == 5 and 'accept its licence' in done.stderr
    assert "Type 'yes'" not in done.stdout
    assert not snapshot(home, REPO, hm.sha(1)).exists()


def test_manifest_and_output_hold_no_credentials(hub, run, home):
    add_tiny(hub, gated='auto', granted={SECRET})
    add_tiny(hub, 'acme/open', n=2)
    done = run('install', REPO, input='yes\n', free=PLENTY, token=SECRET)
    assert done.returncode == 0, done.stderr
    assert any(r['token'] == SECRET for r in hub.downloads())       # authenticated download
    host = hub.endpoint[len('http://'):]
    opened = run('install', 'acme/open', input='yes\n', free=PLENTY,
                 endpoint='http://alice:pa55word@%s' % host)
    assert opened.returncode == 0, opened.stderr
    assert manifest(home, 'acme/open', hm.sha(2))['endpoint'] == hub.endpoint
    for text in (registry_text(home), done.stdout, done.stderr, opened.stdout, opened.stderr):
        assert SECRET not in text and 'pa55word' not in text and 'alice:' not in text


def test_installed_files_are_read_only_originals(hub, run, home):
    files = add_tiny(hub)
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 0
    snap = snapshot(home, REPO, hm.sha(1))
    for name, data in files.items():
        blob = (snap/name).resolve()
        assert blob.read_bytes() == data                            # byte for byte: nothing converted
        assert stat.S_IMODE(blob.stat().st_mode) & 0o222 == 0, name
    assert manifest(home, REPO, hm.sha(1))['read_only'] is True


def test_reinstalling_a_verified_checkpoint_downloads_nothing(hub, run, home):
    add_tiny(hub)
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 0
    hub.requests.clear()
    done = run('install', REPO, free=PLENTY)
    assert done.returncode == 0, done.stderr
    assert 'already installed and verified' in done.stdout and "Type 'yes'" not in done.stdout
    assert weights_fetched(hub) == []


def test_summary_reports_duplicates_already_in_the_cache(hub, run, home):
    files = add_tiny(hub)
    hub.add(hm.repo('mirror/tiny-qwen2', files, revision=hm.sha(2)))
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 0
    done = run('install', 'mirror/tiny-qwen2', input='no\n', free=PLENTY)
    duplicated = len(files[WEIGHT])+len(files[SECOND])
    assert 'duplicates:' in done.stdout and 'acme/tiny-qwen2' in done.stdout
    assert '%d bytes' % duplicated in done.stdout


def local_checkpoint(tmp_path, name='my-qwen2'):
    directory = tmp_path/'models'/name
    directory.mkdir(parents=True)
    files = hm.checkpoint(hm.QWEN2, shards=2)
    for filename, data in files.items():
        (directory/filename).write_bytes(data)
    return directory, files


def fingerprint(directory):
    return {p.name: (p.stat().st_mtime_ns, p.stat().st_mode, p.stat().st_ino, p.read_bytes())
            for p in sorted(directory.iterdir())}


def test_import_registers_a_reference_without_copying_or_modifying(hub, run, home, tmp_path):
    directory, files = local_checkpoint(tmp_path)
    before = fingerprint(directory)
    done = run('import', str(directory))
    assert done.returncode == 0, done.stderr
    assert str(directory) in done.stdout
    assert fingerprint(directory) == before
    stored = [p for p in (home/'.local/share/inwime').rglob('*') if p.is_file()]
    assert stored and all(p.stat().st_size < 64*1024 for p in stored)
    assert not cache(home).exists()
    assert hub.requests == []


def test_import_refuses_what_is_not_a_checkpoint_directory(hub, run, tmp_path):
    empty = tmp_path/'empty'
    empty.mkdir()
    done = run('import', str(empty))
    assert done.returncode == 2 and 'config.json' in done.stderr
    done = run('import', str(tmp_path/'absent'))
    assert done.returncode == 2 and 'not a directory' in done.stderr


def listing(run):
    result = run.json('list')
    assert result['schema'] == 'inwime-model-list/1'
    return {m['id']: m for m in result['models']}, result


def test_list_shows_installed_and_imported_with_states_and_space(hub, run, home, tmp_path):
    files = add_tiny(hub)
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 0
    directory, local = local_checkpoint(tmp_path)
    assert run('import', str(directory)).returncode == 0
    hub.requests.clear()
    models, result = listing(run)
    installed, imported = models[REPO+'@'+hm.sha(1)], models[str(directory)]
    assert installed['source'] == 'installed' and installed['state'] == 'verified'
    assert installed['bytes'] == sum(map(len, files.values()))
    assert installed['path'] == str(snapshot(home, REPO, hm.sha(1)))
    assert imported['source'] == 'imported' and imported['state'] == 'imported'
    assert imported['bytes'] == sum(map(len, local.values()))
    # Stati di capacità from the local metadata: a known architecture in BF16, not verified.
    for entry in (installed, imported):
        assert entry['support'] == 'partial', entry
        assert entry['reached'] == 'operators_resolved'
        assert entry['active'] is False
    assert result['bytes_total'] == installed['bytes'] + imported['bytes']
    assert result['active'] is None
    text = run('list')
    assert text.returncode == 0 and REPO+'@'+hm.sha(1) in text.stdout and str(directory) in text.stdout
    assert hub.requests == []                                       # the inventory is local


def test_list_reports_failed_incomplete_missing_and_changed_checkpoints(hub, run, home, tmp_path):
    add_tiny(hub)
    add_tiny(hub, 'acme/broken', n=2)
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 0
    hub.corrupt.add(WEIGHT)
    assert run('install', 'acme/broken', input='yes\n', free=PLENTY).returncode == 12
    directory, _ = local_checkpoint(tmp_path)
    assert run('import', str(directory)).returncode == 0
    # A blob removed from the cache behind InWiMe's back; a file of the import rewritten.
    (snapshot(home, REPO, hm.sha(1))/SECOND).resolve().unlink()
    (directory/'config.json').write_text('{}')
    models, _ = listing(run)
    assert models[REPO+'@'+hm.sha(1)]['state'] == 'missing'
    assert models['acme/broken@'+hm.sha(2)]['state'] == 'failed'
    assert models[str(directory)]['state'] == 'changed'
    for entry in models.values():
        assert entry['support'] is None                            # not assessed: files not intact


def test_use_sets_the_active_model_explicitly(hub, run, home, tmp_path):
    add_tiny(hub)
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 0
    directory, _ = local_checkpoint(tmp_path)
    assert run('import', str(directory)).returncode == 0
    hub.requests.clear()
    done = run('use', REPO+'@'+hm.sha(1))
    assert done.returncode == 0, done.stderr
    models, result = listing(run)
    assert result['active'] == REPO+'@'+hm.sha(1) and models[REPO+'@'+hm.sha(1)]['active'] is True
    # A repository with one installed revision may be named without it; a path selects an import.
    assert run('use', str(directory)).returncode == 0
    assert listing(run)[1]['active'] == str(directory)
    assert run('use', REPO).returncode == 0
    assert listing(run)[1]['active'] == REPO+'@'+hm.sha(1)
    assert '*' in run('list').stdout
    assert hub.requests == []                                       # use and list stay local


def test_use_refuses_unknown_or_unusable_models(hub, run, home):
    add_tiny(hub)
    hub.corrupt.add(WEIGHT)
    assert run('install', REPO, input='yes\n', free=PLENTY).returncode == 12
    done = run('use', REPO+'@'+hm.sha(1))
    assert done.returncode == 13 and 'failed' in done.stderr
    done = run('use', 'acme/nothing@'+hm.sha(9))
    assert done.returncode == 3 and 'models list' in done.stderr
    assert listing(run)[1]['active'] is None


def test_doctor_never_downloads_or_contacts_the_hub(hub, run, home):
    import subprocess
    import sys
    add_tiny(hub)
    env = run.environment(free=PLENTY)
    done = subprocess.run([sys.executable, '-m', 'inwime', 'doctor', '--json', '--no-host-query'],
                          capture_output=True, text=True, env=env, timeout=120, cwd=home)
    assert done.returncode in (0, 1), done.stderr
    run.check_guard()
    assert hub.requests == [] and not cache(home).exists()


def test_weights_in_formats_never_loaded_are_not_downloaded(hub, run, home):
    files = hm.checkpoint(hm.QWEN2, extra={'pytorch_model.bin': b'\x80'*4096})
    hub.add(hm.repo(REPO, files, revision=hm.sha(1)))
    done = run('install', REPO, input='yes\n', free=PLENTY)
    assert done.returncode == 0, done.stderr
    summary = done.stdout[:done.stdout.index("Type 'yes'")]
    assert 'pytorch_model.bin' in summary and str(sum(map(len, files.values()))-4096) in summary
    assert hub.downloads('pytorch_model.bin') == []
    m = manifest(home, REPO, hm.sha(1))
    assert m['excluded_files'] == ['pytorch_model.bin']
    assert {f['name'] for f in m['files']} == set(files) - {'pytorch_model.bin'}
