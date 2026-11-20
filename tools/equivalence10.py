"""Component-level equivalence of oracle10 against the increment-9 oracle captures (7B, local).

What is new in oracle10 relative to oracle9 is exactly (a) the row-gather embedding
and (b) the final-norm + head path (resident FP32 or blocked FP64). Decoder layers
use the same streamed-module code as oracle9 and are covered by the bit-identity
check against stock from_pretrained on the fixtures (tools/check_oracle10.py).

This tool verifies (a) and (b) on the real Qwen2.5-7B checkpoint at low cost:
  (a) every embedding row gathered by RowGatherEmbedding equals, bit for bit, the
      BF16 bytes read independently from the shard at the header offset (stdlib);
  (b) for every forward call recorded in the oracle9 capture (chunks 5,1,1,1), the
      captured last-layer residual (exact in the contract dtype) is passed through
      the upstream final RMSNorm and the oracle10 head; the logits are compared
      with the captured logits: bit identity for stock_fp32 (resident head), the
      preregistered rule (max |delta| <= 1e-9 and identical argmax) for FP64.

The result is classified PARTIAL (component parity): it does not re-run the 28
decoder layers. End-to-end equivalence on the 7B remains a preregistered
prerequisite of the 14B oracle runs (protocol.numerical_contract.oracle10).
Reads about 2.2 GB of BF16 (embedding rows, head, norm); no capture is modified.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np
import torch

from . import compare10, oracle10_plan
from .oracle10 import CONTRACTS, DEFAULT_HEAD_MODE, EMBED, Counters, build, LayerStreamer
from .oracle_precision9 import precision_profile


def raw_row(directory, wmap, token, width):
    """Independent stdlib read of one BF16 embedding row (header offset arithmetic)."""
    path = Path(directory)/wmap[EMBED]
    with path.open('rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
        begin = header[EMBED]['data_offsets'][0]
        f.seek(8+length+begin+token*width*2)
        return f.read(width*2)


def run(directory, capture, meta, contract, block_rows, budget):
    dtype, profile = CONTRACTS[contract]
    head_mode = DEFAULT_HEAD_MODE[contract]
    config = json.loads((Path(directory)/'config.json').read_text())
    plan = oracle10_plan.plan(directory, config, str(dtype).removeprefix('torch.'), len(meta['sequence']),
                              head_mode, block_rows)
    # Only norm + head + rows are materialized here: the layer component does not apply.
    plan['components']['layer'] = 0
    plan['planned_peak_bytes'] = plan['components']['overhead'] + plan['components']['head'] + sum(
        plan['components'][k] for k in ('embedding', 'outputs', 'activations'))
    plan['planned_rss_upper_bytes'] = plan['planned_peak_bytes'] + plan['mapped_source_bytes']
    oracle10_plan.check_budget(plan, budget)
    counters = Counters()
    model, cfg, wmap = build(directory, dtype, head_mode, block_rows, counters)
    streamer = LayerStreamer(model, directory, wmap, dtype, counters)
    arrays = np.load(capture)
    hidden, logits = arrays['full.hidden'], arrays['full.logits']
    width = cfg.hidden_size
    # (a) embedding rows
    tokens = meta['sequence']
    rows = model.model.embed_tokens(torch.tensor([tokens]))[0]
    gathered = model.model.embed_tokens.counters.embedding_rows
    with torch.inference_mode():
        bf16 = torch.cat([torch.frombuffer(bytearray(raw_row(directory, wmap, t, width)), dtype=torch.bfloat16)[None]
                          for t in tokens])
    embedding_ok = bool(torch.equal(rows, bf16.to(dtype)))
    # (b) final norm + head per recorded call
    calls, start = [], 0
    began = time.monotonic()
    with torch.inference_mode(), torch.autocast('cpu', enabled=False), precision_profile(model, profile):
        for size in meta['chunks']:
            last = torch.from_numpy(hidden[-1, start:start+size]).to(dtype)
            if not torch.equal(last.double(), torch.from_numpy(hidden[-1, start:start+size])):
                raise ValueError('captured residual is not exact in the contract dtype')
            out = model.lm_head(model.model.norm(last[None]))[0].double().numpy()
            ref = logits[start:start+size]
            calls.append(dict(tokens=tokens[start:start+size], bit_identical=compare10.identical(out, ref),
                              max_abs_difference=float(np.abs(out-ref).max()),
                              argmax_identical=bool((out.argmax(-1) == ref.argmax(-1)).all())))
            start += size
    for h in streamer.handles:
        h.remove()
    if contract == 'stock_fp32':
        passed = all(c['bit_identical'] for c in calls)
        rule = 'bit identity (resident FP32 head, same F.linear as nn.Linear)'
    else:
        passed = all(c['max_abs_difference'] <= 1e-9 and c['argmax_identical'] for c in calls)
        rule = 'max |delta| <= 1e-9 and identical argmax (blocked FP64 head)'
    return dict(contract=contract, head_mode=head_mode, block_rows=block_rows, rule=rule,
                coverage='PARTIAL', coverage_detail='embedding gather + final norm + head on every recorded call; '
                'decoder layers not re-run (covered on fixtures by check_oracle10)',
                embedding_rows_compared=gathered, embedding_rows_bit_identical=embedding_ok,
                calls=calls, logits_elements_compared=int(sum(len(c['tokens']) for c in calls)*cfg.vocab_size),
                passed=bool(passed and embedding_ok), wall_s=time.monotonic()-began,
                memory_plan=plan, memory_budget_bytes=budget,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                head_rows_read=counters.head_rows, source_bf16_bytes_read=counters.source_bytes,
                versions=dict(torch=torch.__version__, numpy=np.__version__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--contract', choices=sorted(CONTRACTS), required=True)
    p.add_argument('--block-rows', type=int, default=16384)
    p.add_argument('--memory-budget', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    meta = json.loads(a.capture.with_suffix('.json').read_text())
    if meta['contract'] != a.contract:
        raise ValueError('capture contract differs')
    import hashlib
    with a.capture.open('rb') as f:
        capture_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if capture_sha != meta['capture_sha256']:
        raise ValueError('capture digest differs from its metadata')
    result = run(a.checkpoint, a.capture, meta, a.contract, a.block_rows, a.memory_budget)
    result.update(capture=os.path.relpath(a.capture.absolute()), capture_sha256=capture_sha,
                  checkpoint=os.path.relpath(a.checkpoint.absolute()),
                  checkpoint_revision=json.loads((a.checkpoint/'provenance.json').read_text())['revision'],
                  sequence=meta['sequence'], chunks=meta['chunks'], threads=torch.get_num_threads())
    with a.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ('contract', 'passed', 'coverage', 'embedding_rows_bit_identical',
                                             'peak_rss_bytes', 'wall_s')}
                     | dict(calls=[(c['bit_identical'], c['max_abs_difference']) for c in result['calls']]), indent=1))


if __name__ == '__main__':
    main()
