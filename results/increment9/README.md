# Evidenze dell'incremento 9

Rapporto completo: [docs/RUNTIME_CPU_INCREMENT_9.md](../../docs/RUNTIME_CPU_INCREMENT_9.md).
Revisione e remediation: [docs/REVIEW_INCREMENT_9.md](../../docs/REVIEW_INCREMENT_9.md).
Gate automatico: [summary.json](summary.json), ricalcolato dopo la remediation:
condizioni A–H tutte **PASS**, diagnostico FP64 standard **FAIL grezzo** conservato.
Hardening del gate successivo alla chiusura M9 (17 novembre 2026, D-1…D-6):
[post-closure/README.md](post-closure/README.md); con il gate aggiornato il
report sulle evidenze storiche resta identico a `summary.json`.
Nessun peso o cattura numerica è versionato: catture e NPZ restano in
`artifacts/increment9/` (136 MiB, esclusi da Git), con hash in
[capture-manifest.json](capture-manifest.json).

## Indice

| Area | File |
|---|---|
| Protocollo preregistrato e emendamento pre-misura (invariato, SHA fissato nel gate) | `protocol.json` |
| Emendamento della remediation, successivo alle misure (R1–R10; R10 non è un inasprimento) | `protocol-amendment-remediation.json` |
| Sanificazione dei percorsi personali | `sanitization-manifest.json` |
| Audit iniziale/finale memoria, swap, cgroup, storage, host | `environment-before.json`, `environment-after.json` |
| Prova del meccanismo cgroup/evizione su file da 300 MB | `cgroup-probe-128mib.json`, `.final.txt` |
| Preflight Qwen2.5-7B (API, header via HTTP Range) | `qwen-preflight.json` |
| Acquisizione Qwen2.5-7B, SHA in streaming | `qwen-acquisition.json` |
| Fase A, cache calda addebitata al supervisore (W) | `phaseA-W-supervisor.{json,log,final.txt}`, `phaseA-W-runtime.json`, `phaseA-W-parity-inc8.json` |
| Fase A, cache rimossa con DONTNEED e pressione sotto 2 GiB (P) | `phaseA-P-supervisor.{json,log,final.txt}`, `phaseA-P-runtime.json`, `phaseA-P-parity-inc8.json` |
| Fase B, forward Qwen2.5-7B sync / pipeline | `phaseB-{sync,pipeline}-supervisor.{json,log,final.txt}`, `phaseB-{sync,pipeline}-runtime.json` |
| Oracoli a blocchi (supervisione) | `supervisor-oracle-{stock_fp32,full64,stock_fp64}.{json,log}` |
| Parità completa | `parity-stock_fp32.json`, `parity-full64.json`, `parity-stock_fp64.json` (diagnostico) |
| Controlli piccoli oracolo/adapter contro upstream | `oracle9-checks.txt` |
| Guasti reali su Qwen2.5-7B | `faults.json`, `supervisor-faults.{json,log,final.txt}` |
| OOM cgroup controllato | `negative-oom-policy-stop-supervisor.{json,log}` (difetto), `negative-oom-supervisor.{json,log,final.txt}` (corretto) |
| SmolLM2-135M in scope da 48 MiB | `scope48mib-135m-*` |
| Re-hash dei checkpoint | `integrity-after.json` (16:45 del 15 novembre, prima di sonde di guasto e test OOM), `integrity-final.json` (ultima lettura dei checkpoint della remediation) |
| Conservazione e regressioni (rigenerati dalla remediation) | `preservation.json`, `pytest-final.txt`, `gate7-rerun.json`, `gate8-rerun.json`, `oracle9-checks.txt`, `check-oracle8.txt`, `check-oracle-precision7.txt` |
| Prove della remediation | `review/gate-mutations-{red,green}.txt`, `review/mutations_{red,green}.py`, `review/tool-tests-{red,green}.txt`, `review/parity-*-reverified.json` |
| Prove del follow-up della seconda revisione | `review/bypasses_followup.py`, `review/bypasses-followup-{red,green}.txt` |

`scope48mib-135m-*` era stato lanciato come test OOM negativo, ma il processo è
riuscito: i file sono stati rinominati per ciò che dimostrano. Il nome di unità
e i percorsi interni ai JSON conservano l'etichetta originale `negative-oom`.
`negative-oom-policy-stop-supervisor.*` documenta il difetto del supervisore poi corretto
(`OOMPolicy=continue`); le prove riuscite precedenti non hanno avuto OOM e il
loro finalizzatore ha scritto normalmente i contatori.

## Evidenze storiche, sanificate e derivate

Le evidenze di misura sono storiche e non vengono rigenerate: runtime,
supervisori, oracoli, parità, guasti, sonde, acquisizione, preflight, audit.
`qwen-acquisition.json` è lasciato com'è: il suo campo `sha256` è il digest
ufficiale copiato, non quello calcolato (limite dichiarato nel rapporto).

Undici file sono stati **sanificati** da `tools/sanitize9.py`: il percorso
assoluto del checkout diventa `<repo>`, la directory temporanea di sessione
`<scratch>`, l'identificativo della sessione di terminale `<session>`, e le voci
della cache Hugging Face estranee al progetto sono rimosse. Nessun numero è
cambiato. `sanitization-manifest.json` elenca, per ogni file, SHA256 originale e
sanificato e le regole applicate. Gli originali restano in locale in
`artifacts/increment9/original-evidence/` (esclusa da Git) e coincidono con i
file del commit `6031b52`. Verifica:

```bash
python3 -m tools.sanitize9 results/increment9 --verify
```

Il gate fissa nel codice (`SANITIZED` in `tools/report_increment9.py`) i nomi
degli undici file e i loro SHA256 originale e sanificato: un manifest vuoto,
una voce rimossa o un originale locale non elencato danno FAIL. Il comando qui
sopra, da solo, non conosce quell'insieme: senza originali non rileva una voce
rimossa. La sanificazione riguarda solo l'incremento 9: le evidenze degli
incrementi 4–8 e V2 contengono ancora percorsi locali (conteggi in
`docs/REVIEW_INCREMENT_9.md`).

I report derivati si rigenerano in quest'ordine, cancellando esplicitamente il
precedente: controlli oracle, `gate7-rerun.json`, `gate8-rerun.json`,
`preservation.json`, poi `pytest-final.txt` (i test del gate leggono i report
già presenti), infine `summary.json`. `review/mutations_red.py` documenta la
prova contro il gate originale del commit `6031b52`; `review/mutations_green.py`
è rieseguibile. `review/bypasses_followup.py` riproduce i 13 aggiramenti della
seconda revisione: senza argomenti contro il gate attuale, con una directory
che contiene i `tools/` di `eb669f2` contro il gate precedente.

## Riproduzione

Dalla radice, venv `.venv-oracle7` già presente (nessuna nuova installazione).
Usare sempre nomi di output nuovi: tutti gli strumenti rifiutano la sovrascrittura.

```bash
make cpu-kernel
python3 -m tools.audit_increment9 --output /tmp/ambiente9.json
# Preflight senza pesi; il download richiede --execute (circa 15,24 GB).
python3 -m tools.acquire_qwen9 checkpoints/qwen2.5-7b-d149729 --evidence /tmp/preflight9.json
S=checkpoints/smollm2-1.7b-d4c18cc
python3 -m tools.cgroup9 --label A-P --output /tmp/a-p.json --condition evicted \
  --file $PWD/$S/model.safetensors --host-disk -- \
  .venv-oracle7/bin/python -m tools.benchmark_increment9 $S --family llama --mode sync \
  --as-mib 256 --decodes 3 --output /tmp/a-p-runtime.json --capture artifacts/increment9/nuova-a \
  --cache-state evicted
# Condizione W: --condition warm. Fase B: --family qwen2 sul checkpoint Qwen, --file per ogni shard.
.venv-oracle7/bin/python -m tools.verify_increment8 --runtime /tmp/a-p-runtime.json \
  --capture artifacts/increment9/nuova-a --oracle artifacts/increment8/oracle-fp32.npz \
  --protocol results/increment8/protocol.json --output /tmp/a-parita.json
python3 -m tools.supervise8 --output /tmp/oracolo.json --rss-stop-mib 7168 --available-stop-mib 768 \
  .venv-oracle7/bin/python -m tools.oracle9 checkpoints/qwen2.5-7b-d149729 --contract full64 \
  --sequence 785,6722,315,9625,374,12095,13,1084 --prompt-length 5 --chunks 5,1,1,1 \
  --output artifacts/increment9/nuovo-full64.npz
.venv-oracle7/bin/python -m tools.verify_increment9 --runtime <runtime.json> --capture <cattura> \
  --oracle artifacts/increment9/nuovo-full64.npz --output /tmp/parita-full64.json
.venv-oracle7/bin/python -m unittest tools.check_oracle9
python3 -m tools.cgroup9 --label guasti --output /tmp/guasti-sup.json -- \
  python3 -m tools.fault_increment9 checkpoints/qwen2.5-7b-d149729 --output /tmp/guasti.json
.venv-oracle7/bin/python -m tools.check_oracle8
.venv-oracle7/bin/python -m tools.check_oracle_precision7
python3 -m tools.report_increment7 results/increment7/numerical-resolution \
  --resolve-numerical-contracts --output /tmp/gate7.json
python3 -m tools.report_increment8 results/increment8 --output /tmp/gate8.json
python3 -m tools.integrity9 --output /tmp/conservazione.json preservation \
  --baseline 68ba9f3adf3cffb89b18b19525644fa0d7c6d806 --intentional README.md
# Re-hash completo: legge circa 18,7 GB, da eseguire da solo.
python3 -m tools.integrity9 --output /tmp/integrita.json integrity \
  checkpoints/smollm2-1.7b-d4c18cc checkpoints/qwen2.5-7b-d149729
python3 -m pytest -q tests
python3 -m tools.report_increment9 results/increment9 --output /tmp/gate9.json
python3 results/increment9/review/mutations_green.py
python3 results/increment9/review/bypasses_followup.py
```

Le prove vanno eseguite una alla volta: i confronti di I/O non sono validi con
carichi concorrenti. Gli scope sono transitori (`--collect`) e scompaiono
all'uscita; non viene creata alcuna unità persistente.
