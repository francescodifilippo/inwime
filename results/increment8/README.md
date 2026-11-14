# Incremento 8 — evidenze

**PASS A–F**, [summary.json](summary.json), ricostruibile con:

```bash
python3 -m tools.report_increment8 results/increment8 --output /tmp/inwime-gate8-nuovo.json
```

SmolLM2-1.7B originale, 3.422.752.768 B BF16 unici attivi, sotto RLIMIT_AS
**128 MiB** in sync e pipeline. Prefill 5 token più tre decode, circa 92 MB RSS
osservati. Parità completa standard Transformers FP32; nessuna parità realmente
FP64 dichiarata per 1.7B. Nessuna prova oltre RAM WSL/host. Misure cache OS calda,
`read_bytes=0` nei forward della serie principale: nessuna velocità NVMe cold.

Il [rapporto](../../docs/RUNTIME_CPU_INCREMENT_8.md) spiega contratti, limiti,
contabilità, tentativo fallito, difetto corretto nel loader oracle e riproduzione.

- `protocol.json`: soglie e budget fissati prima del forward 1.7B.
- `environment-before.json`, `resource-audit.json`, `cgroup-audit.json`,
  `system-memory-after-oracles.json`: RAM WSL/host, storage virtuale e backing
  Windows, inventario iniziale, limiti cgroup e swap globale distinto dal processo.
- `acquisition.json`, `provenance.json`, `integrity-after.json`: API ufficiale,
  revisione/digest/dimensioni, hash dei file e verifica finale anche del 135M.
- `smoke-sync-1024.json`: primo originale con cap 1 GiB e catture complete.
- `measured-{sync,pipeline}-{1,2,3}.json`: sei processi a 256 MiB, ordine alternato;
  tempi, copertura per tensore, byte, memoria e cleanup. `supervisor-*` conservano
  comandi esatti, exit code, picchi campionati e log. Il primo smoke è fuori mediane.
- `strict-{sync,pipeline}-128.json`: entrambi i modi completi a 128 MiB.
- `extended.json`, `chunks.json`: otto token completi e prefill 2+3.
- `parity-*.json`: ogni logit/residuo/KV confrontato con FP32, soglie invariate,
  metriche per layer e argmax, su tutti gli scenari e i due formati.
- `exact-invariants.json`: array FP64 identici nei byte fra modalità/formati e
  fra chunk/full/decode incrementale, inclusa tutta la cache KV.
- `repack.json`: tre shard standard, SHA file e per tensore, BF16 identico.
  `upstream-formats.json`: anche tutte le catture Transformers original/sharded
  sono identiche nei byte, con caricamento reale degli shard.
- `faults.json`: sei guasti reali su multi-shard (sync/pipeline × I/O/short/cancel),
  KV candidato già modificato, vecchio banco e descriptor preservati, drain e cleanup;
  budget insufficiente rifiutato prima del payload.
- `negative-as64-supervisor.*`: AS insufficiente, MemoryError/exit 1, senza OOM
  di sistema o risultato fittiziamente valido.
- `supervisor-oracle-fp32.*`: tentativo iniziale fallito per buffer RoPE `meta`.
  `oracle-loader-{red,green}.txt`: regressione prima/dopo correzione nel solo
  loader di verifica. `supervisor-oracle-fp32-fixed.*` e `supervisor-oracle-sharded-fp32.*`:
  catture riuscite, picco separato circa 7,25 GB, VmSwap osservato zero.
- `regression7-*.json/.log`, `regression7-gate.json`: tutti i contratti del 7
  rieseguiti contro le catture precedenti verificate. FP32 e FP64 effettivo PASS,
  standard FP64 diagnostico **FAIL invariato: 270 logits + 41 hidden**. Il suo
  exit 1 è atteso e conservato, senza reinterpretazione.
- `pytest-baseline.txt`: 382 PASS; `pytest-final.txt`, `regression-tests.json`:
  **395 PASS, zero skip** (382 precedenti + 13 nuovi). `oracle-precision7-tests.txt`:
  5 PASS aggiuntivi nel venv. `oracle-loader-green.txt`: 1 PASS dedicato.
- `preservation-{before,after}.json`: tutti i file consolidati invariati, salvo
  l'aggiunta esplicita al README; nessun risultato storico modificato.
- `capture-manifest.json`, `artifact-sha256.json`: forme/hash delle catture locali,
  fingerprint del codice, binario C e dei piccoli report finali.

I checkpoint sono in `checkpoints/`, le catture in `artifacts/increment8/`, entrambi
esclusi da Git. Le catture grezze non contengono pesi: soltanto output FP64 bounded;
gli NPZ upstream sono FP32. Per il nuovo modello gli oracoli eseguono Transformers
stock e non importano forward InWiMe. I comparatori offline leggono catture dopo
la chiusura del runtime. Nessuna installazione di sistema, commit, staging o push.
