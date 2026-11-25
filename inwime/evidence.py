"""Pinned evidence about exact Checkpoints (repository at a full Revisione).

Only these records can raise a Checkpoint beyond `operators_resolved`; a repository name alone
grants nothing. Each record is tied to a report of the repository that holds the measurements.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Verified:
    """A Checkpoint that produced tokens through the Motore and passed an independent oracle."""
    repo_id: str
    revision: str
    path: str            # Percorso computazionale of the evidence
    evidence: str        # report in this repository


@dataclass(frozen=True)
class CertificationTarget:
    """A Checkpoint the v0.1.0-alpha release certifies (spec #32); highlighted, never required."""
    repo_id: str
    revision: str
    role: str
    status: str          # 'pending' until the certification gates of the release pass


VERIFIED = (
    Verified('HuggingFaceTB/SmolLM2-135M', '93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
             'Percorso congelato, increment 7', 'docs/RUNTIME_CPU_INCREMENT_7.md'),
    Verified('HuggingFaceTB/SmolLM2-1.7B', 'd4c18ccb8a25e6e637e0609aac08a5d1a0436d1b',
             'Percorso congelato, increment 8', 'docs/RUNTIME_CPU_INCREMENT_8.md'),
    Verified('Qwen/Qwen2.5-7B', 'd149729398750b98c0af14eb82c78cfe92750796',
             'Percorso congelato, increment 9', 'docs/RUNTIME_CPU_INCREMENT_9.md'),
    Verified('Qwen/Qwen2.5-14B', '97e1e76335b7017d8f67c08a19d103c0504298c9',
             'Percorso congelato, increment 10', 'docs/RUNTIME_CPU_INCREMENT_10.md'),
    Verified('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775',
             'Percorso congelato with the tied-head catalog (#38)', 'results/issue38/README.md'),
)

CERTIFICATION_TARGETS = (
    CertificationTarget('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775',
                        'conversational integration with DeepSeek Harness', 'pending'),
    CertificationTarget('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28',
                        'out-of-core certification through the API', 'pending'),
)


def verified(repo_id, revision):
    return next((v for v in VERIFIED if (v.repo_id, v.revision) == (repo_id, revision)), None)


def certification(repo_id, revision=None):
    """The certification record of a repository, and whether it names this Revisione."""
    target = next((t for t in CERTIFICATION_TARGETS if t.repo_id == repo_id), None)
    if target is None:
        return None
    return {'revision': target.revision, 'role': target.role, 'status': target.status,
            'this_revision': revision == target.revision}


def highlights(repo_id, revision=None):
    """Marks for a search result: certification target and verified Revisioni of the repository."""
    return {'certification': certification(repo_id, revision),
            'verified_revisions': [v.revision for v in VERIFIED if v.repo_id == repo_id]}
