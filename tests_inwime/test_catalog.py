"""Seam: `inwime models search/info` as a user runs them, against a fake Hub on loopback.

Every command runs in a subprocess whose network is guarded (netguard/sitecustomize.py): a
name resolution or connection outside loopback fails and is logged, and each test asserts
that the log stays empty and that the fake Hub saw no request it does not implement.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import inwime
from fakehub import FakeHub
import hubmodels as hm


PACKAGE_PARENT = str(Path(inwime.__file__).resolve().parents[1])
GUARD = str(Path(__file__).resolve().parent/'netguard')
SCRUBBED = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
            'HF_TOKEN_PATH', 'HF_HUB_CACHE', 'XDG_CACHE_HOME', 'HTTP_PROXY', 'HTTPS_PROXY',
            'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')


@pytest.fixture
def hub():
    with FakeHub() as h:
        yield h
        assert h.unexpected == []


@pytest.fixture
def home(tmp_path):
    h = tmp_path/'home-of-alice'
    h.mkdir()
    return h


class Run:
    def __init__(self, hub, home, tmp_path):
        self.hub, self.home, self.guard_log = hub, home, tmp_path/'netguard.log'

    def __call__(self, *args, token=None, endpoint=None, env=None, expect_guard=False):
        full = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
        full.update(HOME=str(self.home), HF_HOME=str(self.home/'.cache/huggingface'),
                    HF_ENDPOINT=endpoint or self.hub.endpoint, HF_HUB_DISABLE_TELEMETRY='1',
                    NO_PROXY='*', INWIME_CPU_KERNEL='off', INWIME_NETGUARD_LOG=str(self.guard_log),
                    PYTHONPATH=os.pathsep.join((GUARD, PACKAGE_PARENT)))
        if token:
            full['HF_TOKEN'] = token
        full.update(env or {})
        result = subprocess.run([sys.executable, '-m', 'inwime', 'models', *args], capture_output=True,
                                text=True, env=full, timeout=120, cwd=self.home)
        violations = self.guard_log.read_text() if self.guard_log.exists() else ''
        if not expect_guard:
            assert violations == '', violations
        result.violations = violations
        return result

    def json(self, *args, **kwargs):
        result = self(*args, '--json', **kwargs)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)


@pytest.fixture
def run(hub, home, tmp_path):
    return Run(hub, home, tmp_path)


def test_info_resolves_the_revision_and_reports_the_architecture(hub, run):
    files = hm.checkpoint(hm.QWEN2, shards=2)
    hub.add(hm.repo('acme/tiny-qwen2', files, revision=hm.sha(1)))
    info = run.json('info', 'acme/tiny-qwen2')
    assert info['schema'] == 'inwime-model-info/1'
    assert info['repo_id'] == 'acme/tiny-qwen2'
    assert info['requested_ref'] == 'main'
    assert info['revision'] == hm.sha(1)
    assert info['architectures'] == ['Qwen2ForCausalLM'] and info['model_type'] == 'qwen2'
    assert info['hub_url'] == hub.endpoint+'/acme/tiny-qwen2/tree/'+hm.sha(1)


TINY_QWEN2_PARAMETERS = 1736     # worked by hand: 2*(32*8) head+embedding, 8 norm, 2*608 per layer


def test_info_reports_metadata_of_the_revision(hub, run):
    files = hm.checkpoint(hm.QWEN2, shards=2, template=hm.TOOL_TEMPLATE)
    hub.add(hm.repo('acme/tiny-qwen2', files, revision=hm.sha(1), license='apache-2.0'))
    info = run.json('info', 'acme/tiny-qwen2@'+hm.sha(1))
    weights = info['weights']
    assert weights['format'] == 'safetensors'
    assert weights['files'] == ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
    assert weights['sharded'] is True
    assert weights['bytes'] == len(files['model-00001-of-00002.safetensors']) \
        + len(files['model-00002-of-00002.safetensors'])
    assert weights['parameters'] == TINY_QWEN2_PARAMETERS
    assert weights['dtypes'] == {'BF16': TINY_QWEN2_PARAMETERS}
    assert info['declared_dtype'] == 'bfloat16'
    assert info['context_length'] == 32768
    assert info['license'] == 'apache-2.0'
    assert info['gated'] is False
    assert info['chat_template'] == {'present': True, 'tools': True}
    assert info['tokenizer']['files'] == ['tokenizer.json', 'tokenizer_config.json']


def test_weight_size_comes_from_the_files_not_from_the_parameter_count(hub, run):
    files = hm.checkpoint(hm.QWEN2, dtype='F32', template=hm.PLAIN_TEMPLATE)
    hub.add(hm.repo('acme/tiny-qwen2-f32', files))
    info = run.json('info', 'acme/tiny-qwen2-f32')
    assert info['weights']['bytes'] == len(files['model.safetensors'])
    assert info['weights']['bytes'] > 4*TINY_QWEN2_PARAMETERS > 2*TINY_QWEN2_PARAMETERS
    assert info['weights']['dtypes'] == {'F32': TINY_QWEN2_PARAMETERS}
    assert info['weights']['sharded'] is False
    assert info['chat_template'] == {'present': True, 'tools': False}


def test_info_never_downloads_weights_or_writes_the_cache(hub, run, home):
    files = hm.checkpoint(hm.QWEN2, shards=2)
    hub.add(hm.repo('acme/tiny-qwen2', files))
    run.json('info', 'acme/tiny-qwen2')
    # Only safetensors headers are read, through byte ranges.
    assert hub.served_bytes('.safetensors') <= 2*100_001
    assert all(r['range'] for r in hub.requests if r['path'].endswith('.safetensors'))
    assert not any(p.is_file() for p in home.rglob('*'))


ORDER = ('discovered', 'metadata_parsed', 'graph_resolved', 'operators_resolved',
         'resource_plan_feasible', 'end_to_end_executable', 'numerically_verified')


def states(info):
    assert [s['state'] for s in info['capability']['states']] == list(ORDER)
    return {s['state']: s['status'] for s in info['capability']['states']}


def diagnosis(info, category):
    return [d for d in info['capability']['diagnosis'] if d['category'] == category]


def reached_up_to(last):
    stop = ORDER.index(last)
    return {s: 'reached' if i <= stop else 'not_reached' for i, s in enumerate(ORDER)}


def test_known_architecture_in_bf16_is_partial_until_verified(hub, run):
    hub.add(hm.repo('acme/tiny-qwen2', hm.checkpoint(hm.QWEN2, shards=2)))
    hub.add(hm.repo('acme/tiny-llama', hm.checkpoint(hm.LLAMA)))
    for repo_id, adapter in (('acme/tiny-qwen2', 'Qwen2ForCausalLM'), ('acme/tiny-llama', 'LlamaForCausalLM')):
        info = run.json('info', repo_id)
        expected = reached_up_to('operators_resolved')
        expected['resource_plan_feasible'] = 'not_evaluated'
        assert states(info) == expected, repo_id
        assert info['capability']['support'] == 'partial'
        assert info['capability']['adapter'] == adapter
        assert diagnosis(info, 'verification')


def test_verified_revision_is_supported_and_only_that_revision(hub, run):
    smol = 'HuggingFaceTB/SmolLM2-135M'
    verified, other = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2', hm.sha(77)
    files = hm.checkpoint(hm.LLAMA)
    hub.add(hm.Repo(id=smol, revisions={verified: files, other: files}, branches={'main': other}))
    info = run.json('info', smol+'@'+verified)
    assert states(info) == reached_up_to('numerically_verified')
    assert info['capability']['support'] == 'supported'
    proof = {s['state']: s.get('evidence') for s in info['capability']['states']}
    assert proof['numerically_verified'] == 'docs/RUNTIME_CPU_INCREMENT_7.md'
    latest = run.json('info', smol)
    assert latest['revision'] == other
    assert latest['capability']['support'] == 'partial'
    assert states(latest)['numerically_verified'] == 'not_reached'


def test_evidence_is_not_granted_by_name_when_the_checkpoint_does_not_resolve(hub, run):
    impostor = dict(hm.QWEN2, architectures=['MixtralForCausalLM'], model_type='mixtral',
                    num_local_experts=8, num_experts_per_tok=2)
    revision = 'd149729398750b98c0af14eb82c78cfe92750796'          # verified Qwen2.5-7B revision
    hub.add(hm.repo('Qwen/Qwen2.5-7B', hm.checkpoint(impostor, shapes=hm.tensors(hm.QWEN2)), revision=revision))
    info = run.json('info', 'Qwen/Qwen2.5-7B@'+revision)
    assert states(info) == reached_up_to('metadata_parsed')
    assert info['capability']['support'] == 'unknown'


def test_unknown_architecture_is_diagnosed(hub, run):
    moe = dict(hm.QWEN2, architectures=['MixtralForCausalLM'], model_type='mixtral',
               num_local_experts=8, num_experts_per_tok=2)
    hub.add(hm.repo('acme/tiny-moe', hm.checkpoint(moe, shapes=hm.tensors(hm.QWEN2))))
    info = run.json('info', 'acme/tiny-moe')
    assert states(info) == reached_up_to('metadata_parsed')
    assert info['capability']['support'] == 'unknown'
    message = ' '.join(d['message'] for d in diagnosis(info, 'architecture'))
    assert 'MixtralForCausalLM' in message
    assert any('mixture-of-experts' in d['message'] for d in diagnosis(info, 'operators'))


def test_adapter_rejection_is_diagnosed_with_the_feature(hub, run):
    tied = dict(hm.QWEN2, tie_word_embeddings=True)
    hub.add(hm.repo('acme/tiny-qwen2-tied', hm.checkpoint(tied)))
    info = run.json('info', 'acme/tiny-qwen2-tied')
    assert states(info) == reached_up_to('metadata_parsed')
    assert info['capability']['support'] == 'unknown'
    assert any('tie_word_embeddings' in d['message'] for d in diagnosis(info, 'architecture'))


def test_non_bf16_dtype_stops_before_operators_resolved(hub, run):
    hub.add(hm.repo('acme/tiny-qwen2-f16', hm.checkpoint(hm.QWEN2, dtype='F16')))
    info = run.json('info', 'acme/tiny-qwen2-f16')
    assert states(info) == reached_up_to('graph_resolved')
    assert info['capability']['support'] == 'partial'
    assert any('F16' in d['message'] for d in diagnosis(info, 'dtype'))


def test_remote_code_stops_before_graph_resolved(hub, run):
    custom = dict(hm.QWEN2, auto_map={'AutoModelForCausalLM': 'modeling_custom.CustomForCausalLM'})
    hub.add(hm.repo('acme/remote-code', hm.checkpoint(custom, extra={'modeling_custom.py': b'#'})))
    info = run.json('info', 'acme/remote-code')
    assert states(info) == reached_up_to('metadata_parsed')
    assert any('trust_remote_code' in d['message'] for d in diagnosis(info, 'remote_code'))
    custom_tokenizer = {'auto_map': {'AutoTokenizer': ['tokenization_x.XTokenizer', None]}}
    hub.add(hm.repo('acme/remote-tokenizer', hm.checkpoint(hm.QWEN2, tokenizer_config=custom_tokenizer)))
    info = run.json('info', 'acme/remote-tokenizer')
    assert states(info) == reached_up_to('discovered')
    assert diagnosis(info, 'remote_code')


def test_missing_tokenizer_or_safetensors_stop_before_metadata_parsed(hub, run):
    hub.add(hm.repo('acme/no-tokenizer', hm.checkpoint(hm.QWEN2, tokenizer=False)))
    info = run.json('info', 'acme/no-tokenizer')
    assert states(info) == reached_up_to('discovered')
    assert diagnosis(info, 'tokenizer')
    assert info['chat_template'] == {'present': False, 'tools': False}
    pickle_only = hm.checkpoint(hm.QWEN2, drop=('model.safetensors',), extra={'pytorch_model.bin': b'\x80'*64})
    hub.add(hm.repo('acme/pytorch-only', pickle_only))
    info = run.json('info', 'acme/pytorch-only')
    assert states(info) == reached_up_to('discovered')
    assert info['weights']['other_weight_files'] == ['pytorch_model.bin']
    assert any('pytorch_model.bin' in d['message'] for d in diagnosis(info, 'weights'))


def test_certified_checkpoints_are_highlighted_without_limiting_selection(hub, run):
    certified = 'a09a35458c702b33eeacc393d103063234e8bc28'
    files = hm.checkpoint(hm.QWEN2)
    hub.add(hm.Repo(id='Qwen/Qwen2.5-7B-Instruct', revisions={certified: files, hm.sha(5): files},
                    branches={'main': hm.sha(5)}))
    info = run.json('info', 'Qwen/Qwen2.5-7B-Instruct@'+certified)
    assert info['certification']['revision'] == certified and info['certification']['this_revision']
    assert info['certification']['status'] == 'pending'
    other = run.json('info', 'Qwen/Qwen2.5-7B-Instruct')
    assert other['certification']['this_revision'] is False
    assert other['capability']['support'] == 'partial'       # still selectable and described
    hub.add(hm.repo('acme/uncertified', files))
    assert run.json('info', 'acme/uncertified')['certification'] is None


SECRET = 'hf_SECRETtokenVALUE0123456789abcdef'


def test_missing_repository_and_revision_are_distinct_errors(hub, run):
    hub.add(hm.repo('acme/tiny-qwen2', hm.checkpoint(hm.QWEN2)))
    missing = run('info', 'acme/does-not-exist')
    assert missing.returncode == 3 and 'not found' in missing.stderr and 'acme/does-not-exist' in missing.stderr
    revision = run('info', 'acme/tiny-qwen2@'+hm.sha(999))
    assert revision.returncode == 3 and 'revision' in revision.stderr.lower()
    assert 'Traceback' not in missing.stderr+revision.stderr
    bad = run('info', 'not a repo id!')
    assert bad.returncode == 2


def test_gated_repository_distinguishes_authentication_and_licence(hub, run):
    gated = hm.repo('acme/gated', hm.checkpoint(hm.QWEN2), gated='manual', granted={SECRET})
    hub.add(gated)
    anonymous = run('info', 'acme/gated')
    assert anonymous.returncode == 4
    assert 'hf auth login' in anonymous.stderr and 'HF_TOKEN' in anonymous.stderr
    refused = run('info', 'acme/gated', token='hf_someoneELSE0123456789')
    assert refused.returncode == 5
    assert hub.endpoint+'/acme/gated' in refused.stderr          # where to accept the licence
    allowed = run('info', 'acme/gated', '--json', token=SECRET)
    assert allowed.returncode == 0, allowed.stderr
    info = json.loads(allowed.stdout)
    assert info['gated'] == 'manual' and info['capability']['support'] == 'partial'
    # The token goes to the Hub, never to any output.
    assert any(r['token'] == SECRET for r in hub.requests)
    for result in (anonymous, refused, allowed):
        assert SECRET not in result.stdout+result.stderr
    text = run('info', 'acme/gated', token=SECRET)
    assert text.returncode == 0 and SECRET not in text.stdout+text.stderr


def test_offline_mode_is_reported_not_bypassed(hub, run):
    hub.add(hm.repo('acme/tiny-qwen2', hm.checkpoint(hm.QWEN2)))
    result = run('info', 'acme/tiny-qwen2', env={'HF_HUB_OFFLINE': '1'})
    assert result.returncode == 7 and 'HF_HUB_OFFLINE' in result.stderr
    assert hub.requests == []


def test_rate_limit_is_retried_after_the_announced_wait_then_reported(hub, run):
    hub.add(hm.repo('acme/tiny-qwen2', hm.checkpoint(hm.QWEN2)))
    info_path = '/api/models/acme/tiny-qwen2/revision/'
    hub.rate_limit(info_path, times=2, retry_after='0')
    assert run.json('info', 'acme/tiny-qwen2')['capability']['support'] == 'partial'
    hub.rate_limit('/acme/tiny-qwen2/resolve/', times=1, retry_after='0')
    assert run.json('info', 'acme/tiny-qwen2')['capability']['support'] == 'partial'
    hub.requests.clear()
    hub.rate_limit(info_path, times=100, retry_after='0')
    exhausted = run('info', 'acme/tiny-qwen2')
    assert exhausted.returncode == 6 and '429' in exhausted.stderr
    assert len([r for r in hub.requests if r['path'].startswith(info_path)]) == 4   # bounded attempts
    hub.requests.clear()
    hub.rate_limit(info_path, times=100, retry_after='3600')
    long_wait = run('info', 'acme/tiny-qwen2')
    assert long_wait.returncode == 6 and 'reset in 3600 s' in long_wait.stderr
    assert len(hub.requests) == 1                                 # an hour is reported, not slept


def catalog(hub):
    files = hm.checkpoint(hm.QWEN2)
    base = ['transformers', 'safetensors']
    for repo_id, tags, task, downloads, likes, params in (
            ('Qwen/Qwen2.5-7B-Instruct', ['qwen2', 'license:apache-2.0', 'conversational'],
             'text-generation', 900, 50, 7_615_616_512),
            ('Qwen/Qwen2.5-0.5B-Instruct', ['qwen2', 'license:apache-2.0'], 'text-generation', 800, 90,
             494_032_768),
            ('acme/llama-chat', ['llama', 'license:llama3.1'], 'text-generation', 700, 10, 8_030_261_248),
            ('acme/qwen2-finetune', ['qwen2', 'license:mit'], 'text-generation', 600, 5, 1_543_714_304),
            ('acme/classifier', ['bert', 'license:apache-2.0'], 'text-classification', 500, 70, 109_482_240),
            ('someone/obscure-experiment', ['mixtral'], 'text-generation', 1, 0, 46_702_792_704)):
        hub.add(hm.repo(repo_id, files, tags=base+tags, pipeline_tag=task, downloads=downloads, likes=likes,
                        parameters={'BF16': params}))


def ids(result):
    return [m['id'] for m in result['results']]


def test_search_lists_the_whole_catalog_without_a_whitelist(hub, run):
    catalog(hub)
    result = run.json('search', '--limit', '10')
    assert result['schema'] == 'inwime-model-search/1'
    assert ids(result) == ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', 'acme/llama-chat',
                           'acme/qwen2-finetune', 'acme/classifier', 'someone/obscure-experiment']
    first = result['results'][0]
    assert first['task'] == 'text-generation' and first['downloads'] == 900 and first['likes'] == 50
    assert first['parameters'] == 7_615_616_512 and first['license'] == 'apache-2.0'
    assert first['certification']['status'] == 'pending'
    assert result['results'][2]['certification'] is None


def test_search_filters_by_text_author_task_family_licence_and_size(hub, run):
    catalog(hub)
    assert ids(run.json('search', 'instruct')) == ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct']
    assert ids(run.json('search', '--author', 'acme')) == ['acme/llama-chat', 'acme/qwen2-finetune',
                                                            'acme/classifier']
    assert ids(run.json('search', '--task', 'text-classification')) == ['acme/classifier']
    assert ids(run.json('search', '--family', 'qwen2', '--license', 'apache-2.0')) == [
        'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct']
    assert ids(run.json('search', '--min-params', '1B', '--max-params', '8B')) == [
        'Qwen/Qwen2.5-7B-Instruct', 'acme/qwen2-finetune']
    assert ids(run.json('search', '--sort', 'likes', '--limit', '2')) == [
        'Qwen/Qwen2.5-0.5B-Instruct', 'acme/classifier']


def test_search_pages_through_results(hub, run):
    catalog(hub)
    pages = [run.json('search', '--limit', '4', '--page', str(n)) for n in (1, 2, 3)]
    assert ids(pages[0]) == ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', 'acme/llama-chat',
                             'acme/qwen2-finetune']
    assert pages[0]['has_more'] is True and pages[0]['page'] == 1
    assert ids(pages[1]) == ['acme/classifier', 'someone/obscure-experiment']
    assert pages[1]['has_more'] is False
    assert ids(pages[2]) == [] and pages[2]['has_more'] is False
    text = run('search', '--limit', '4')
    assert text.returncode == 0
    assert 'Qwen/Qwen2.5-7B-Instruct' in text.stdout and '--page 2' in text.stdout


def test_search_rejects_invalid_sizes_and_pages(hub, run):
    for args in (('--min-params', 'lots'), ('--page', '0'), ('--limit', '0'), ('--limit', '101')):
        result = run('search', *args)
        assert result.returncode == 2, args


def test_guard_fails_any_request_that_leaves_loopback(hub, run):
    result = run('info', 'Qwen/Qwen2.5-7B-Instruct', endpoint='https://huggingface.co', expect_guard=True)
    assert result.returncode != 0
    assert 'huggingface.co' in result.violations
    assert hub.requests == []


def test_client_version_is_pinned_and_reported(hub, run):
    import huggingface_hub
    from importlib import metadata
    try:
        requirements = metadata.requires('inwime') or []
    except metadata.PackageNotFoundError:          # source checkout: read the declaration itself
        import tomllib
        pyproject = Path(PACKAGE_PARENT)/'pyproject.toml'
        requirements = tomllib.loads(pyproject.read_text())['project']['dependencies']
    pins = [r for r in requirements if r.replace('-', '_').lower().startswith('huggingface_hub')]
    assert pins == ['huggingface_hub==%s' % huggingface_hub.__version__]
    hub.add(hm.repo('acme/tiny-qwen2', hm.checkpoint(hm.QWEN2)))
    assert run.json('info', 'acme/tiny-qwen2')['client'] == 'huggingface_hub ' + huggingface_hub.__version__


def test_diagnosis_lists_every_missing_piece_not_only_the_first(hub, run):
    custom = dict(hm.QWEN2, auto_map={'AutoModelForCausalLM': 'modeling_custom.CustomForCausalLM'})
    files = hm.checkpoint(custom, drop=('model.safetensors',), extra={'pytorch_model.bin': b'\x80'*64})
    hub.add(hm.repo('acme/remote-code-bin', files))
    info = run.json('info', 'acme/remote-code-bin')
    assert states(info) == reached_up_to('discovered')
    assert diagnosis(info, 'weights') and diagnosis(info, 'remote_code')
    mistral = dict(hm.LLAMA, architectures=['MistralForCausalLM'], model_type='mistral',
                   tie_word_embeddings=False)
    hub.add(hm.repo('acme/mistral-f16', hm.checkpoint(mistral, dtype='F16', tokenizer=False)))
    info = run.json('info', 'acme/mistral-f16')
    assert states(info) == reached_up_to('discovered')
    assert diagnosis(info, 'tokenizer') and diagnosis(info, 'architecture') and diagnosis(info, 'dtype')


def test_invalid_index_is_diagnosed_as_such(hub, run):
    files = hm.checkpoint(hm.QWEN2, shards=2)
    files['model.safetensors.index.json'] = b'{not json'
    hub.add(hm.repo('acme/broken-index', files))
    info = run.json('info', 'acme/broken-index')
    assert states(info) == reached_up_to('discovered')
    assert any('not valid JSON' in d['message'] for d in diagnosis(info, 'weights'))


def test_oversized_metadata_is_refused_before_download(hub, run):
    hub.add(hm.repo('acme/huge-tokenizer', hm.checkpoint(hm.QWEN2),
                    reported_sizes={'tokenizer_config.json': 64*2**20}))
    result = run('info', 'acme/huge-tokenizer')
    assert result.returncode == 8 and 'tokenizer_config.json' in result.stderr
    assert not any(r['path'].endswith('tokenizer_config.json') for r in hub.requests)


def test_search_marks_the_certified_revision_and_survives_a_rate_limit(hub, run):
    certified = 'a09a35458c702b33eeacc393d103063234e8bc28'
    hub.add(hm.repo('Qwen/Qwen2.5-7B-Instruct', hm.checkpoint(hm.QWEN2), revision=certified))
    hub.rate_limit('/api/models', times=1, retry_after='0')
    result = run.json('search')
    assert result['results'][0]['certification']['this_revision'] is True


def test_untrusted_metadata_cannot_drive_the_terminal(hub, run):
    hostile = dict(hm.QWEN2, model_type='qwen2\x1b]0;owned\x07\x1b[2J')
    hub.add(hm.repo('acme/hostile', hm.checkpoint(hostile), license='mit\x1b[31m\r\x9b'))
    for args in (('info', 'acme/hostile'), ('search',)):
        result = run(*args)
        assert result.returncode == 0, result.stderr
        assert not any(c in result.stdout for c in '\x1b\x07\r\x9b'), args


def test_absurd_dimensions_are_refused_without_allocating(hub, run):
    hub.add(hm.repo('acme/absurd', hm.checkpoint(dict(hm.QWEN2, num_hidden_layers=10**12),
                                                 shapes=hm.tensors(hm.QWEN2))))
    info = run.json('info', 'acme/absurd')
    assert states(info) == reached_up_to('metadata_parsed')
    assert any('num_hidden_layers' in d['message'] for d in diagnosis(info, 'architecture'))


def test_oversized_safetensors_header_is_diagnosed(hub, run):
    import struct
    files = hm.checkpoint(hm.QWEN2)
    files['model.safetensors'] = struct.pack('<Q', 30_000_000) + b'{}' + bytes(64)
    hub.add(hm.repo('acme/huge-header', files))
    info = run.json('info', 'acme/huge-header')
    assert states(info) == reached_up_to('discovered')
    assert any('header' in d['message'] for d in diagnosis(info, 'weights'))
    assert hub.served_bytes('.safetensors') <= 100_001


def test_endpoint_credentials_never_reach_the_output(hub, run):
    hub.add(hm.repo('acme/tiny-qwen2', hm.checkpoint(hm.QWEN2)))
    secret_endpoint = hub.endpoint.replace('http://', 'http://mirror-user:MIRRORsecret@')
    ok = run('info', 'acme/tiny-qwen2', '--json', endpoint=secret_endpoint)
    missing = run('info', 'acme/none', endpoint=secret_endpoint)
    dead = run('info', 'acme/tiny-qwen2', endpoint='http://u:DEADsecret@127.0.0.1:9')
    assert ok.returncode == 0 and missing.returncode == 3 and dead.returncode == 7
    assert json.loads(ok.stdout)['endpoint'] == hub.endpoint
    for result, secret in ((ok, 'MIRRORsecret'), (missing, 'MIRRORsecret'), (dead, 'DEADsecret')):
        assert secret not in result.stdout + result.stderr
