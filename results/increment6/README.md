# Evidenze incremento 6 — 14 novembre 2026

Base `f304ee9f19b6a89907d4ad10cbdce435640352fb`. Acquisizioni iniziali senza
commit/push; consolidamento locale autorizzato dopo la revisione.
Rapporto: [RUNTIME_CPU_INCREMENT_6.md](../../docs/RUNTIME_CPU_INCREMENT_6.md).

**Revisione e consolidamento successivi:** [review/REVIEW.md](review/REVIEW.md),
due correzioni sostanziali, **332 test superati**, nuove acquisizioni in `review/`.
Le voci `final` elencate sotto identificano la chiusura precedente alla revisione,
non il nuovo codice corretto. Un solo commit locale è ora autorizzato; nessun push.

- `pytest-baseline.txt`: 181 regressioni passate prima dell'implementazione.
- `pytest-source-first.txt`: prima verifica della sorgente, 113 test.
- `pytest-final.txt`, `pytest-resources.json`: checkpoint intermedio, 308 test.
- **`pytest-final-closed.txt`, `pytest-resources-final.json`**: codice finale,
  **310 passed in 33,59 s**, RSS campionato 74.682.368 byte, intervallo 10 ms,
  stop 1 GiB. OPENBLAS_NUM_THREADS=1 solo per il processo dei test; nessuna
  configurazione globale. Nessun processo benchmark eseguito in parallelo.
- **`abc-final/summary.json`** e JSON individuali: confronto conclusivo A/B/C,
  dense/MoE, sync/pipeline. Un warm-up per braccio (`*-0.json`) e cinque
  campioni misurati (`*-1.json` … `*-5.json`), processo nuovo per campione,
  ordine conservato nel riepilogo. Include file sorgente effettivamente importato,
  fingerprint del codice baseline esportato e della libreria C, output per fase,
  prefill/decode, parsing, startup/hash, contatori per tensore, RSS/HWM, swap,
  I/O di processo, metadata, arene e verifica cleanup.
- `abc/`: prima serie, conservata integralmente. MoE sync B/A +11,26% ha
  motivato `regression-moe-sync/`: altre cinque coppie A/B, +3,03%. Queste serie
  precedono gli ultimi controlli di total_size/FIFO. La serie conclusiva B/A
  riporta +3,89% dense sync, +2,48% dense pipeline, −10,41% MoE sync e −3,04%
  MoE pipeline. Range sovrapposti: nessuno speedup attribuito al contenitore.
- **`probe-final-sync.json`, `probe-final-pipeline.json`**: GEMV 513×4099,
  4.205.574 byte BF16 originali, tutte le righe confrontate indipendentemente con
  `math.fsum`, errore massimo 1,9984e-14, SHA output identica. I vecchi `probe-*`
  senza `final` conservano l'acquisizione precedente all'apertura non bloccante.
- `benchmark*.log` e log individuali: stdout/stderr dei driver. I log vuoti
  indicano conclusione senza output diagnostico; gli esiti sono nei JSON.
- `closing-audit.json`: ambiente e memoria visibile, conferma file originali
  identici a HEAD, kernel/wrapper/pipeline invariati, nuovo disco persistente
  4.406.063 byte, senza download. Il campo `src_cpu_kernel_unchanged` controlla
  anche `pipeline.py` e `math_cpu.py`.
- `artifact-sha256.json`: hash di sorgenti, test, documentazione, build CPU,
  descriptor fixture locali e risultati (escluso il manifest stesso).

Le fixture sono riproducibili localmente, senza scaricare pesi:

```bash
make cpu-kernel
python3 -m tools.benchmark_increment6 \
  --code-root /tmp/inwime-inc6-baseline-NUOVO \
  --fixture checkpoints/inc6-NUOVO --output results/inc6-NUOVO
python3 -m tools.probe_safetensors6 checkpoints/inc6-probe-NUOVO --generate
python3 -m tools.probe_safetensors6 checkpoints/inc6-probe-NUOVO \
  --mode sync --output results/probe-NUOVO.json
```

Comando suite completa nel JSON risorse finale. La prima verifica mirata è stata
eseguita chiamando lo stesso worker A/B su `checkpoints/inc6-small/moe-v2`, cinque
coppie in ordine AB/BA alternato; i singoli campioni conservano codice e opzioni
effettivi. I due driver non sovrascrivono fixture/output esistenti nei normali
percorsi CLI. I payload generati sono esclusi da Git in `checkpoints/` e gli
originali in `fixtures/reference_decoder/` non vengono modificati.

Tutte le letture d'inferenza misurate hanno `read_bytes=0` a livello blocco:
**cache-warm** dopo SHA completa, su ext4 WSL. Byte logici/hash/header e traffico
di blocco sono distinti. Nessuna prova cold NVMe, nessuna GPU inizializzata,
nessun modello pubblico end-to-end né prova nuova oltre RAM host. AC rilevata;
temperature, potenza e clock non disponibili nelle sonde interrogate. Non
generalizzare le latenze di fixture piccole a modelli reali.
