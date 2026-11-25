"""Stati di capacità and Vista di supporto of one Checkpoint, from its metadata only (#27, #29).

The states are milestones reached in order; a later state is never inferred from an earlier
one, and none from the repository name. What each one requires here:

- discovered: the Hub resolved the repository and the Revisione to a commit SHA.
- metadata_parsed: config, safetensors index and every shard header, and a tokenizer that
  needs no remote code, were read and are consistent.
- graph_resolved: no remote code is declared and an InWiMe adapter accepts the config and the
  exact tensor set. The decision is the runtime adapter's own; `PROFILES` only words hints.
- operators_resolved: every tensor is in a dtype the weight provider reads (original BF16
  only; nothing is converted) and the provider's metadata bounds hold.
- resource_plan_feasible: depends on a host; the catalog does not evaluate it, except where
  pinned evidence shows a run on the host of its report (scope stated in the output).
- end_to_end_executable, numerically_verified: only from pinned evidence for this exact
  Revisione (`evidence.VERIFIED`), on top of all the states above.

Every check runs even when an earlier state fails, so the diagnosis lists everything that is
missing; the states themselves still stop at the first failure.

Vista di supporto: `supported` when numerically_verified is reached, `partial` when
graph_resolved is reached but verification is not, `unknown` otherwise (not determinable, or
known to be missing something: the diagnosis says which).
"""
import inspect
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import evidence

ORDER = ('discovered', 'metadata_parsed', 'graph_resolved', 'operators_resolved',
         'resource_plan_feasible', 'end_to_end_executable', 'numerically_verified')
NATIVE_DTYPES = ('BF16',)
MAX_HEADER_FILES = 256                    # shard headers the catalog reads per Revisione
MAX_CATALOG_TENSORS = 200_000             # tensors the catalog keeps in memory per Revisione
TOKENIZER_SETS = (('tokenizer.json',), ('tokenizer.model',), ('spiece.model',), ('tiktoken.model',),
                  ('vocab.json', 'merges.txt'), ('vocab.txt',))

# Diagnostic hints only: which config values each adapter's profile expects. The acceptance
# decision always comes from the adapter itself (`resolve_graph`).
PROFILES = {
    'Qwen2ForCausalLM': {'hidden_act': 'silu', 'rope_scaling': None, 'use_sliding_window': False,
                         'use_mrope': False},
    'LlamaForCausalLM': {'tie_word_embeddings': True, 'hidden_act': 'silu', 'rope_scaling': None,
                         'attention_bias': False, 'mlp_bias': False, 'pretraining_tp': 1},
}
PROFILE_DESCRIPTION = {'Qwen2ForCausalLM': 'Qwen2ForCausalLM with a tied or untied head',
                       'LlamaForCausalLM': 'LlamaForCausalLM with a tied head'}


def adapters():
    from inwime_runtime.llama import LlamaCatalog
    from .qwen2_tied import qwen2_catalog
    return {'LlamaForCausalLM': LlamaCatalog, 'Qwen2ForCausalLM': qwen2_catalog}


def provider_limits():
    """Metadata bounds of the runtime weight provider, read from its signature."""
    from inwime_runtime.safetensors_source import SafetensorsSource
    p = inspect.signature(SafetensorsSource).parameters
    return {'max_shards': p['max_shards'].default, 'max_tensors': p['max_tensors'].default}


class _HeaderSource:
    """What an adapter reads from a weight source, built from remote headers only."""

    def __init__(self, directory, tensors):
        from inwime_runtime.core import Tensor
        self.directory = Path(directory)
        self.tensors = {name: Tensor(shape, '', None, math.prod(shape), (), ())
                        for name, (_, shape) in tensors.items()}
        self.retained_bytes = self.bootstrap_bytes = 0


def resolve_graph(architecture, config, config_raw, tensors):
    """Run the runtime adapter on the config bytes and tensor shapes; None or its refusal.

    The config comes from an untrusted Hub repository. Every adapter needs at least one tensor
    per layer, so a layer count above the tensor count is refused here, before the adapter
    builds per-layer structures from it.
    """
    from inwime_runtime.core import CheckpointError
    layers = config.get('num_hidden_layers')
    if type(layers) is not int or not 0 < layers <= len(tensors):
        return 'num_hidden_layers=%s is not a positive count within the %d tensors' % (
            json.dumps(layers)[:40], len(tensors))
    with tempfile.TemporaryDirectory(prefix='inwime-catalog-') as directory:
        (Path(directory)/'config.json').write_bytes(config_raw)
        try:
            adapters()[architecture](_HeaderSource(directory, tensors))
        except CheckpointError as error:
            return str(error)
        except (MemoryError, OverflowError, RecursionError) as error:
            return 'config dimensions out of range (%s)' % type(error).__name__
    return None


def feature_hints(config):
    """Config features that name missing operators or representations."""
    hints = []
    for key in ('num_local_experts', 'num_experts', 'n_routed_experts'):
        if isinstance(config.get(key), int) and config[key] > 1:
            hints.append(('operators', 'mixture-of-experts routing (%s=%d) has no InWiMe operator yet'
                          % (key, config[key])))
            break
    if config.get('kv_lora_rank') is not None:
        hints.append(('operators', 'multi-head latent attention (kv_lora_rank) has no InWiMe operator yet'))
    if config.get('use_sliding_window') is True:
        hints.append(('operators', 'sliding-window attention is not implemented'))
    if config.get('rope_scaling') is not None:
        scaling = config['rope_scaling']
        kind = scaling.get('rope_type', scaling.get('type')) if isinstance(scaling, dict) else scaling
        hints.append(('operators', 'RoPE scaling %r is not implemented' % (kind,)))
    if config.get('quantization_config') is not None:
        method = config['quantization_config']
        method = method.get('quant_method') if isinstance(method, dict) else method
        hints.append(('dtype', 'pre-quantized checkpoint (%s): only original BF16 weights are read' % method))
    return hints


def profile_hints(architecture, config):
    hints = []
    for key, expected in PROFILES.get(architecture, {}).items():
        # tie_word_embeddings has no default the adapters accept: absent counts as a mismatch.
        actual = config.get(key, None if key == 'tie_word_embeddings' else expected)
        if actual != expected:
            hints.append(('architecture', '%s=%s: the adapter supports %s only (%s=%s)'
                          % (key, json.dumps(actual), PROFILE_DESCRIPTION[architecture], key,
                             json.dumps(expected))))
    return hints


@dataclass(frozen=True)
class RevisionMetadata:
    """What the catalog read from one Revisione, without weights."""
    repo_id: str
    revision: str
    files: frozenset                # file names of the Revisione
    config: dict                    # parsed config.json ({} when absent or invalid)
    config_raw: bytes | None
    tokenizer_config: object        # parsed tokenizer_config.json, None when absent
    index: object                   # parsed safetensors index, None when absent
    weights: dict                   # catalog weight summary
    tensors: dict                   # name -> (dtype, shape), from the shard headers


INVALID_JSON = '__invalid_json__'


def assess(meta):
    """States, Vista di supporto and diagnosis of one Revisione."""
    diagnosis = []

    def blocked(state, category, message):
        diagnosis.append({'state': state, 'category': category, 'message': message})

    def failed(state):
        return any(d['state'] == state for d in diagnosis)

    config, weights = meta.config, meta.weights

    # metadata_parsed
    config_ok = bool(config) and not config.get(INVALID_JSON)
    if not config_ok:
        blocked('metadata_parsed', 'config', 'config.json is missing or is not a JSON object')
    elif not (isinstance(config.get('architectures'), list) and config['architectures']
              and isinstance(config.get('model_type'), str)):
        config_ok = False
        blocked('metadata_parsed', 'config', 'config.json does not declare architectures and model_type')
    if isinstance(meta.index, dict) and meta.index.get(INVALID_JSON):
        blocked('metadata_parsed', 'weights', 'model.safetensors.index.json is not valid JSON')
    elif weights['format'] != 'safetensors':
        found = ', '.join(weights['other_weight_files']) or 'none'
        blocked('metadata_parsed', 'weights', 'no safetensors weights (model.safetensors or its index); '
                'weight files found: %s. Pickle-based weights are never loaded' % found)
    elif weights['missing_files']:
        blocked('metadata_parsed', 'weights', 'the index names files absent from the Revisione: %s'
                % ', '.join(weights['missing_files']))
    elif weights.get('header_error'):
        blocked('metadata_parsed', 'weights', weights['header_error'])
    elif not weights['headers_read']:
        blocked('metadata_parsed', 'weights', '%d weight files: above the %d headers the catalog reads'
                % (len(weights['files']), MAX_HEADER_FILES))
    elif isinstance(meta.index, dict) and set(meta.index.get('weight_map') or ()) != set(meta.tensors):
        blocked('metadata_parsed', 'weights', 'the safetensors index and the shard headers disagree')
    if not any(all(name in meta.files for name in group) for group in TOKENIZER_SETS):
        blocked('metadata_parsed', 'tokenizer', 'no tokenizer files (tokenizer.json, tokenizer.model, '
                'vocab.json with merges.txt, ...)')
    if isinstance(meta.tokenizer_config, dict) and meta.tokenizer_config.get('auto_map'):
        blocked('metadata_parsed', 'remote_code', 'the tokenizer declares custom code (auto_map); '
                'trust_remote_code is not supported in v0.1')
    metadata = not diagnosis

    # graph_resolved (checked whatever the state above, for a complete diagnosis)
    adapter = None
    if config_ok:
        architectures = config['architectures']
        if config.get('auto_map'):
            blocked('graph_resolved', 'remote_code', 'config.json declares custom modeling code (auto_map); '
                    'trust_remote_code is not supported in v0.1')
        if len(architectures) != 1 or architectures[0] not in adapters():
            blocked('graph_resolved', 'architecture', 'no InWiMe adapter for %s (model_type %s); adapters: %s'
                    % (', '.join(map(str, architectures)), config['model_type'],
                       ', '.join(PROFILE_DESCRIPTION.values())))
        elif weights['headers_read']:
            refusal = resolve_graph(architectures[0], config, meta.config_raw, meta.tensors)
            if refusal is None:
                adapter = architectures[0]
            else:
                blocked('graph_resolved', 'architecture', 'the %s adapter refuses this checkpoint: %s'
                        % (architectures[0], refusal))
                for category, message in profile_hints(architectures[0], config):
                    blocked('graph_resolved', category, message)
        if adapter is None:
            for category, message in feature_hints(config):
                blocked('graph_resolved', category, message)
    graph = metadata and adapter is not None and not failed('graph_resolved')

    # operators_resolved (also checked for a complete diagnosis)
    if weights['headers_read']:
        foreign = {d: n for d, n in weights['dtypes'].items() if d not in NATIVE_DTYPES}
        if foreign:
            blocked('operators_resolved', 'dtype', 'tensors in %s: the weight provider reads original BF16 '
                    'only and converts nothing' % ', '.join('%s (%d parameters)' % kv for kv in foreign.items()))
        limits = provider_limits()
        if len(weights['files']) > limits['max_shards']:
            blocked('operators_resolved', 'weights', '%d shards: above the weight provider bound of %d'
                    % (len(weights['files']), limits['max_shards']))
        if len(meta.tensors) > limits['max_tensors']:
            blocked('operators_resolved', 'weights', '%d tensors: above the weight provider bound of %d'
                    % (len(meta.tensors), limits['max_tensors']))
    operators = graph and not failed('operators_resolved')

    proof = evidence.verified(meta.repo_id, meta.revision) if operators else None
    reached = {'discovered': True, 'metadata_parsed': metadata, 'graph_resolved': graph,
               'operators_resolved': operators}
    states = [{'state': s, 'status': 'reached' if reached[s] else 'not_reached'} for s in ORDER[:4]]
    for s in ORDER[4:]:
        if proof is not None:
            state = {'state': s, 'status': 'reached', 'evidence': proof.evidence, 'path': proof.path}
            if s == 'resource_plan_feasible':
                state['scope'] = 'the host of the report; other hosts are not evaluated by the catalog'
            states.append(state)
        elif s == 'resource_plan_feasible' and operators:
            states.append({'state': s, 'status': 'not_evaluated'})
        else:
            states.append({'state': s, 'status': 'not_reached'})
    if operators and proof is None:
        blocked('resource_plan_feasible', 'resources', 'depends on the host: evaluated by the resource '
                'planner when the model is installed or run, not by the catalog')
    if proof is None:
        blocked('numerically_verified', 'verification', 'no end-to-end or oracle evidence for %s@%s'
                % (meta.repo_id, meta.revision))
    support = 'supported' if proof is not None else 'partial' if graph else 'unknown'
    return {'states': states, 'support': support, 'adapter': adapter, 'diagnosis': diagnosis}
