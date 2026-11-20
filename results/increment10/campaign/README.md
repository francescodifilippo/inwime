# Evidenze della campagna dell'incremento 10 (19–20 novembre 2026)

Rapporto: [docs/RUNTIME_CPU_INCREMENT_10.md](../../../docs/RUNTIME_CPU_INCREMENT_10.md).
Gate: `python -m tools.report_increment10 --output gate10.json` (eseguito nel venv di
verifica: F ricalcola la parità con NumPy dai file di cattura).

Ogni prova è stata eseguita una sola volta con il proprio identificatore. Le prove non
avviate per un DEFER dell'host hanno un file `blocked-<prova>.json` (causa, ora UTC,
rapporto host); un nuovo tentativo usa un nuovo identificatore (`-a2`, `-a3`). Nessun
risultato è stato sovrascritto o cancellato.

| File | Strumento | Contenuto |
|---|---|---|
| `hostmem-baseline.json` | `tools/hostmem10.py` | baseline della sessione (prima della prima prova) |
| `<prova>-host-start.json` | `tools/hostmem10.py` | controllo di avvio ufficiale (profilo runtime o oracle) |
| `<prova>-run.json` | `tools/campaign10.py` | record: controllo host, L3 prima/dopo, osservatore host in-run (L4b), SHA256 dei file prodotti |
| `<prova>-scope.json`, `.final.txt`, `.log` | `tools/cgroup9.py` (invariato) | scope transitorio, evizione e `fincore`, campioni, contatori del kernel, disco fisico Windows (L4) |
| `R1..R4-forward.json` | `tools/benchmark_increment9.py` (invariato) | forward 14B: fasi, contatori per tensore, `/proc/PID/io`, digest; R1 con catture complete in `artifacts/increment10/R1-capture/` |
| `F-sync*`, `F-pipeline*` | `tools/fault10.py` | guasti N1–N4 sul 14B reale; `F-sync` (tentativo 1) è il FAIL della sonda con guasto alla lettura 100, conservato |
| `N6-*` | `tools/benchmark_increment9.py` in scope da 32 MiB | OOM pianificato di SmolLM2-1.7B (FAIL atteso) |
| `O7-*`, `equivalence-7b-e2e-*.json` | `tools/oracle10.py`, `tools/equivalence10_e2e.py` | equivalenza 7B end-to-end di `oracle10` contro le catture `oracle9` di M9 |
| `O14-*`, `parity-14b-*.json` | `tools/oracle10.py`, `tools/parity10.py` | oracolo 14B e parità a copertura piena con R1 |
| `integrity-final.json` | `tools/integrity9.py` (invariato) | re-hash finale degli 8 shard |
| `check-oracle10.txt` | `tools/check_oracle10.py` | N9: sensibilità dell'oracolo (peso perturbato, epsilon errato) |
| `eviction-*.json` | evizione mirata | `POSIX_FADV_DONTNEED` sui soli file dei checkpoint dell'esperimento, con residenza prima/dopo |
| `host-wait.log`, `sequence.log` | `tools/campaign10_sequence.sh` | attese read-only del GO e passi della sequenza |
| `acquisition/` | copie sanificate | rapporti dell'acquisizione (incidente del gate RAM e ripresa), con SHA256 degli originali |
| `pilot/` | copie sanificate | pilota del 19/11 (informativo, non nel gate) |
| `sanitization-post-write.json`, `sanitization-reversal-local.json` | sanificazione dopo la scrittura | 13 file con percorsi assoluti sostituiti (`<repo>`, `<recovery>`), digest originale e sanificato; verifica locale che l'inversione riproduce i digest originali |
| `pytest-final.*`, `inventory-verify-final.json`, `gate10.json` | suite, `tools/inventory10.py`, `tools/report_increment10.py` | verifica finale (esecuzioni precedenti conservate con suffisso `-aN`) |
| `gate10-mutations*.json`, `mutations/mutate.py` | mutazioni delle evidenze reali | nessuna mutazione produce PASS |

Le catture grandi (`artifacts/increment10/`: catture del runtime e degli oracoli) non sono
in Git; i loro SHA256 sono nei record e nei confronti, e il gate li ricalcola.
