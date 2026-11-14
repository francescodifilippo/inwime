# Revisione finale incremento 8

**Pronto per commit.** [Rapporto](../../../docs/REVIEW_INCREMENT_8.md).
Tre P2 del gate corretti: input storici mancanti, contratto oracle incoerente,
insieme di invarianti vuoto/incompleto. Nessun P0/P1 o rilievo residuo.

- `initial-audit.json`: tutti i 97 fingerprint originali e gli hash delle catture
  corrispondevano prima delle correzioni. Snapshot e report originali preservati.
- `gate-tests-red.txt`: 9 FAIL prima della correzione, 13 PASS preesistenti.
- `gate-tests-green.txt`: primo green 22 test; `gate-tests-final.txt`: 26 PASS con
  le quattro verifiche aggiuntive di completezza. `pytest-final.txt`: **408 PASS**.
- `gate-reviewed.json/.log`: gate A–F PASS, con verifica SHA degli input e gate7
  ricostruito dai cinque confronti, non dai soli booleani di riepilogo.
- `memory-{sync,pipeline}*.json/.log`, `worker-*.json`, `check_memory.py`: nuove
  esecuzioni complete a 128 MiB, PID/start ticks/limiti osservati dal padre e
  controlli delle aperture del checkpoint. Output identici all'acquisizione originale.
  Nessuna nuova misura prestazionale attribuita alle mediane storiche.
- `oracle-exact-values*.json/.log`, `check_oracle.py`: tutti i 218 parametri reali
  esatti dopo promozione BF16→FP32; RoPE e metodi upstream stock. Processo separato:
  picco RSS circa 7,79 GB e VmSwap osservato circa 27,8 MB, dichiarati nel rapporto.
- `oracle-loader.txt`, `oracle-precision7.txt`: rispettivamente 1 + 5 test PASS.
- `integrity-and-coverage.json`, `check_integrity.py`: SHA originale/shard e tutti
  i digest per tensore, contabilità derivata indipendentemente da config/header;
  digest del checkpoint 135M preservato.
- `parity-*.json`, `exact-invariants.json`: tutte le catture rilette, confronto
  completo e invarianti nei byte PASS, stesse metriche storiche.
- `regression7-gate.json`: FP32 e pure FP64 PASS, diagnostico stock FP64 FAIL
  storico conservato (270 logits + 41 hidden oltre soglia, zero KV).
- `faults*.json/.log`: sei nuove iniezioni reali su multi-shard, rollback KV,
  drain/cleanup e ammissione con budget insufficiente verificati.
- `negative-as64-supervisor.*`: nuova insufficienza AS, MemoryError/exit1, nessun
  artefatto di falso successo.
- `preservation-and-git.json`: sorgenti consolidati, esclusioni di pesi/catture e
  indice Git verificati; HEAD osservato durante la revisione `7afcd6b`.
- `summary.json`, `artifact-sha256-reviewed.json`: decisione finale e fingerprint
  della revisione. Il manifest originale rimane valido come snapshot precedente;
  i due sorgenti modificati sono identificati separatamente.

Il cap riguarda il processo InWiMe, non page cache o memoria di sistema. Nessuna
nuova prova oltre RAM WSL/host, cold NVMe, contesto lungo o oracle 1.7B pure FP64.
Nessun commit, push, cambio di sistema o incremento 9 eseguito dalla revisione.
