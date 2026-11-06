# Proposta di commit — consolidamento InWiMe, 2026-11-05

**Solo proposta: nessun file staged, nessun commit/push, nessun benchmark avviato.** Nessun file eliminato, resettato o sostituito. In questo ciclo sono aggiornati soltanto brief/handoff e creato questo documento; codice e risultati preesistenti restano invariati.

## Ambito del prossimo commit

Titolo proposto: `Consolida Benchmark V2 e confronto storage WSL2/USB`.

Descrizione proposta: registra il confronto mirato I/O/decode BF16 sui tre percorsi di storage, con dati grezzi, telemetria e contatori Windows; conserva i selettori della suite e i tool di setup/cleanup/report già implementati. Aggiorna brief e handoff con il Sabrent USB da 512 GB, la disponibilità del proprietario e la priorità della valutazione storage. Nessuna implementazione Briareus/MoE né predisposizione distruttiva del disco.

Il proprietario consente di dedicare l’intero disco a InWiMe, ma **non ha autorizzato una specifica cancellazione/formattazione**. Il prossimo ciclo deve prima valutare supporto e fattibilità di USB passthrough/ext4 nativo, VHDX e altre alternative supportate. Identificare fisicamente il dispositivo e ottenere autorizzazione all’esatta operazione prima di qualsiasi modifica distruttiva. D:/disk 2 sono alias storici, non identità sufficienti.

## Revisione dei cinque tracked iniziali

| File | Esito della revisione documentale/statica |
| --- | --- |
| `BENCHMARK_V2.md` | Addendum storage e correzione temporale dell’audit; valori dello sweep originario mantenuti. Includere il diff già presente, senza rigenerare il rapporto. |
| `README.md` | Documenta confronto mirato, setup controllato, ripresa e cleanup. Includere. |
| `tools/benchmark_v2.py` | Aggiunge filtri buffers/ns/modes mantenendo i default e il minimo di 10 ripetizioni. Includere. |
| `tools/report_v2.py` | Include addendum storage; non avviato durante il consolidamento. Includere. |
| `tools/summarize_v2.py` | Parametrizza input/output e gestisce il subset solo I/O. Includere. |

A questi si aggiunge il tracked `PROJECT_BRIEF.md`, aggiornato in questo ciclo. HEAD resta `44b393a209e38ea79cfc3f1170b32f54225634e0` su `main`. I sorgenti C++, Makefile e test sono già tracciati e puliti: non richiedono nuove modifiche nel commit proposto.

## Manifest preciso: 57 file da includere

Sei documenti, otto sorgenti/tool (compresa diagnostica opzionale) e 43 file di evidenza storage. Sono sei tracked modificati e 51 untracked, incluso questo documento. Il seguente elenco è un manifest, **non un comando eseguito**:

```text
BENCHMARK_V2.md
BENCHMARK_V2_STORAGE.md
PROJECT_BRIEF.md
README.md
docs/COMMIT_PROPOSAL-2026-11-05.md
docs/handoffs/HANDOFF-2026-11-05-benchmark-v2.md
results/v2/storage/cleanup.json
results/v2/storage/datasets.json
results/v2/storage/external-usb/counters-after-io.json
results/v2/storage/external-usb/counters-before-io.json
results/v2/storage/external-usb/physical-counters.json
results/v2/storage/external-usb/summary.csv
results/v2/storage/external-usb/summary.json
results/v2/storage/external-usb/sweep/gpu-1793906768.csv
results/v2/storage/external-usb/sweep/gpu-1793907319.csv
results/v2/storage/external-usb/sweep/n1-b64-c0-async.csv
results/v2/storage/external-usb/sweep/n1-b64-c0-async.json
results/v2/storage/external-usb/sweep/n1-b64-c0-async.log
results/v2/storage/external-usb/sweep/n1-b64-c0-async.memory.csv
results/v2/storage/external-usb/sweep/n1-b64-c0-io.csv
results/v2/storage/external-usb/sweep/n1-b64-c0-io.json
results/v2/storage/external-usb/sweep/n1-b64-c0-io.log
results/v2/storage/external-usb/sweep/n1-b64-c0-io.memory.csv
results/v2/storage/internal-drvfs/physical-counters.json
results/v2/storage/internal-drvfs/summary.csv
results/v2/storage/internal-drvfs/summary.json
results/v2/storage/internal-drvfs/sweep/gpu-1793907860.csv
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.csv
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.json
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.log
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.memory.csv
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.csv
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.json
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.log
results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.memory.csv
results/v2/storage/internal-ext4/physical-counters.json
results/v2/storage/internal-ext4/summary.csv
results/v2/storage/internal-ext4/summary.json
results/v2/storage/internal-ext4/sweep/gpu-1793908503.csv
results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.csv
results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.json
results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.log
results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.memory.csv
results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.csv
results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.json
results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.log
results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.memory.csv
results/v2/storage/usb-link.txt
results/v2/storage/usb-properties.json
tools/benchmark_v2.py
tools/diagnostics/windows_storage.ps1
tools/report_storage.py
tools/report_v2.py
tools/storage_cleanup.py
tools/storage_compare.py
tools/storage_setup.py
tools/summarize_v2.py
```

I CSV per-passata e i metadati permettono di ricostruire conteggi, mediane, minimi, massimi e dispersione. La telemetria conserva memoria/swap, clock, potenza e temperatura. I `.log` dei singoli casi contengono driver/runtime e setup cache: vanno inclusi, a differenza dei log di semplice avanzamento. `datasets.json` e `cleanup.json` documentano identità/hash del file e pulizia, senza incorporare pesi.

Il sorgente `tools/diagnostics/windows_storage.ps1` è proposto per provenienza della diagnostica fallita e coerenza con il ramo di verifica hash nel cleanup. **Non è validato, non è un tool produttivo, è stato bloccato dall’antivirus e non va eseguito né usato per aggirare la protezione.** Questo commit non lo attiva; il limite è documentato nei rapporti.

## Dieci file non tracciati da lasciare locali

Non sono ignorati o rimossi: resteranno visibili in `git status` anche dopo un eventuale commit selettivo. Nessun `git add .` o aggiunta ricorsiva indiscriminata di `results/v2/storage/`.

```text
results/v2/storage/comparison-progress.log
results/v2/storage/external-usb/buffer-comparison.svg
results/v2/storage/external-usb/comparison-progress.log
results/v2/storage/external-usb/io-progress.log
results/v2/storage/internal-drvfs/buffer-comparison.svg
results/v2/storage/internal-drvfs/comparison-progress.log
results/v2/storage/internal-ext4/buffer-comparison.svg
results/v2/storage/internal-ext4/comparison-progress.log
results/v2/storage/setup.log
results/v2/storage/windows-audit.txt
```

- Sei log di avanzamento/setup sono accessori: i risultati e i log dei singoli casi restano nel manifest di inclusione.
- Tre SVG di subset sono derivabili e non referenziate dal rapporto storage. Il grafico canonico dello sweep V2 è già tracciato e rimane intatto.
- L’inventario Windows completo include dispositivi estranei e identificatori di sistema: conservarlo localmente. Capacità/modello/percorso del disco sono documentati nei rapporti; includere le sole proprietà mirate del bridge e i contatori fisici. Un audit storico non sostituisce la futura verifica del dispositivo prima di operazioni distruttive.

## Inventario dei 60 untracked presenti all’inizio

Ogni file iniziale è classificato qui; il nuovo documento di proposta porta il totale finale a 61. Dimensioni in byte rilevate durante il consolidamento (l’handoff viene aggiornato in questo ciclo).

| Percorso | Byte | Categoria | Decisione |
| --- | ---: | --- | --- |
| `BENCHMARK_V2_STORAGE.md` | 8080 | documentazione | includere |
| `docs/handoffs/HANDOFF-2026-11-05-benchmark-v2.md` | 16819 | documentazione | includere |
| `results/v2/storage/cleanup.json` | 260 | evidenza pulizia copie temporanee | includere; non è il dataset |
| `results/v2/storage/comparison-progress.log` | 247 | log di avanzamento temporaneo | locale; riepilogo orchestrazione, non dato per-passata |
| `results/v2/storage/datasets.json` | 657 | provenienza/identità dataset | includere; percorsi storici rimossi |
| `results/v2/storage/external-usb/buffer-comparison.svg` | 4093 | grafico derivato accessorio | locale; non referenziato nel rapporto storage |
| `results/v2/storage/external-usb/comparison-progress.log` | 107 | log di avanzamento temporaneo | locale; riepilogo orchestrazione, non dato per-passata |
| `results/v2/storage/external-usb/counters-after-io.json` | 341 | contatori fisici Windows | includere |
| `results/v2/storage/external-usb/counters-before-io.json` | 340 | contatori fisici Windows | includere |
| `results/v2/storage/external-usb/io-progress.log` | 104 | log di avanzamento temporaneo | locale; riepilogo orchestrazione, non dato per-passata |
| `results/v2/storage/external-usb/physical-counters.json` | 1607 | contatori fisici Windows | includere |
| `results/v2/storage/external-usb/summary.csv` | 2172 | statistiche derivate CSV/JSON | includere per revisione/riproducibilità |
| `results/v2/storage/external-usb/summary.json` | 4402 | statistiche derivate CSV/JSON | includere per revisione/riproducibilità |
| `results/v2/storage/external-usb/sweep/gpu-1793906768.csv` | 146204 | telemetria GPU campionata | includere |
| `results/v2/storage/external-usb/sweep/gpu-1793907319.csv` | 148079 | telemetria GPU campionata | includere |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-async.csv` | 2402 | CSV grezzo delle 10 ripetizioni | includere |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-async.json` | 980 | comando/hash/metadati per caso | includere |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-async.log` | 263 | log caso (driver/cache/setup) | includere; distinto dai progress log |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-async.memory.csv` | 242745 | telemetria memoria/swap | includere |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-io.csv` | 1962 | CSV grezzo delle 10 ripetizioni | includere |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-io.json` | 975 | comando/hash/metadati per caso | includere |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-io.log` | 71 | log caso (driver/cache/setup) | includere; distinto dai progress log |
| `results/v2/storage/external-usb/sweep/n1-b64-c0-io.memory.csv` | 248762 | telemetria memoria/swap | includere |
| `results/v2/storage/internal-drvfs/buffer-comparison.svg` | 4090 | grafico derivato accessorio | locale; non referenziato nel rapporto storage |
| `results/v2/storage/internal-drvfs/comparison-progress.log` | 156 | log di avanzamento temporaneo | locale; riepilogo orchestrazione, non dato per-passata |
| `results/v2/storage/internal-drvfs/physical-counters.json` | 1640 | contatori fisici Windows | includere |
| `results/v2/storage/internal-drvfs/summary.csv` | 2188 | statistiche derivate CSV/JSON | includere per revisione/riproducibilità |
| `results/v2/storage/internal-drvfs/summary.json` | 4418 | statistiche derivate CSV/JSON | includere per revisione/riproducibilità |
| `results/v2/storage/internal-drvfs/sweep/gpu-1793907860.csv` | 173799 | telemetria GPU campionata | includere |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.csv` | 2402 | CSV grezzo delle 10 ripetizioni | includere |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.json` | 1011 | comando/hash/metadati per caso | includere |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.log` | 262 | log caso (driver/cache/setup) | includere; distinto dai progress log |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-async.memory.csv` | 151975 | telemetria memoria/swap | includere |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.csv` | 1962 | CSV grezzo delle 10 ripetizioni | includere |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.json` | 1009 | comando/hash/metadati per caso | includere |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.log` | 71 | log caso (driver/cache/setup) | includere; distinto dai progress log |
| `results/v2/storage/internal-drvfs/sweep/n1-b64-c0-io.memory.csv` | 139024 | telemetria memoria/swap | includere |
| `results/v2/storage/internal-ext4/buffer-comparison.svg` | 4081 | grafico derivato accessorio | locale; non referenziato nel rapporto storage |
| `results/v2/storage/internal-ext4/comparison-progress.log` | 156 | log di avanzamento temporaneo | locale; riepilogo orchestrazione, non dato per-passata |
| `results/v2/storage/internal-ext4/physical-counters.json` | 1634 | contatori fisici Windows | includere |
| `results/v2/storage/internal-ext4/summary.csv` | 2159 | statistiche derivate CSV/JSON | includere per revisione/riproducibilità |
| `results/v2/storage/internal-ext4/summary.json` | 4389 | statistiche derivate CSV/JSON | includere per revisione/riproducibilità |
| `results/v2/storage/internal-ext4/sweep/gpu-1793908503.csv` | 9323 | telemetria GPU campionata | includere |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.csv` | 2371 | CSV grezzo delle 10 ripetizioni | includere |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.json` | 996 | comando/hash/metadati per caso | includere |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.log` | 263 | log caso (driver/cache/setup) | includere; distinto dai progress log |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-async.memory.csv` | 7811 | telemetria memoria/swap | includere |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.csv` | 1942 | CSV grezzo delle 10 ripetizioni | includere |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.json` | 994 | comando/hash/metadati per caso | includere |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.log` | 71 | log caso (driver/cache/setup) | includere; distinto dai progress log |
| `results/v2/storage/internal-ext4/sweep/n1-b64-c0-io.memory.csv` | 7301 | telemetria memoria/swap | includere |
| `results/v2/storage/setup.log` | 205 | log di avanzamento temporaneo | locale; riepilogo orchestrazione, non dato per-passata |
| `results/v2/storage/usb-link.txt` | 271 | evidenza diagnostica fallita | includere; limite sperimentale |
| `results/v2/storage/usb-properties.json` | 1533 | identità bridge/driver USB | includere |
| `results/v2/storage/windows-audit.txt` | 80001 | inventario Windows completo | locale; contiene anche dispositivi estranei |
| `tools/diagnostics/windows_storage.ps1` | 5371 | sorgente diagnostico bloccato da AV | includere come provenienza; non eseguire |
| `tools/report_storage.py` | 11372 | codice suite/report/cleanup | includere |
| `tools/storage_cleanup.py` | 2353 | codice suite/report/cleanup | includere |
| `tools/storage_compare.py` | 4055 | codice suite/report/cleanup | includere |
| `tools/storage_setup.py` | 1805 | codice suite/report/cleanup | includere |

## File grandi e generati: esclusi, da mantenere locali

| File/gruppo | Dimensione osservata | Trattamento |
| --- | ---: | --- |
| `data/weights-5GiB.bf16` | 5.368.709.120 byte (5 GiB) | Dataset sintetico riproducibile, ignorato, non aggiungere |
| `data/weights.bf16` | 536.870.912 byte (512 MiB) | Dataset preliminare, ignorato, non aggiungere |
| `build/bf16_stream`, `build/bf16_stream_profile`, `build/bf16_stream_trace` | circa 206 KiB complessivi | Binari generati, ignorati, non aggiungere |
| `tools/__pycache__/`, `tests/__pycache__/` | 11 file `.pyc` al controllo | Cache generate, ignorate, non aggiungere |

Nessuno dei 60 untracked iniziali è un file grande: il maggiore è un CSV di memoria da 248.762 byte. Il loro totale iniziale era 1.464.181 byte. Nessun candidato proposto raggiunge 1 MiB per singolo file. Il maggiore file già tracciato è il trace storico `results/v2/profile-n1-async.qdstrm` (1.803.937 byte): non è una nuova aggiunta e non viene eliminato o deindicizzato.

Le due copie sperimentali da 5 GiB su C:/D: erano già state rimosse dal benchmark precedente (`cleanup.json`). Questo consolidamento non esegue cleanup o cancellazioni.

## Verifica .gitignore

Regole esistenti adeguate al layout attuale, **nessuna modifica necessaria**:

```gitignore
build/
data/
*.bf16
__pycache__/
```

Verificato con `git check-ignore --no-index`: binari/build, dataset BF16 e ipotetici checkpoint `.safetensors`/`.gguf` dentro `data/` sono protetti; sorgenti, test, documenti e tutti i risultati attuali non sono ignorati. `git ls-files --cached --ignored --exclude-standard` non riporta file. Non aggiungere ignore generici `results/`, `*.csv`, `*.json` o `*.log`, che nasconderebbero evidenze necessarie. Checkpoint futuri vanno nel percorso dati previsto e richiedono approvazione; piccoli fixture BF16 futuri indispensabili richiederebbero eccezioni mirate, non aggiunte forzate indiscriminate.

## Controlli e stato finale atteso

- Audit read-only dei 60 untracked e dei cinque diff iniziali; AST Python e JSON leggibili, senza eseguire i tool. Nessun nuovo test prestazionale o benchmark.
- Risultati benchmark, `COMPARISON.md`, sorgenti e tool precedenti protetti tramite checksum; nessuna rigenerazione o correzione cosmetica dei risultati.
- `git diff --check` segnala la riga vuota finale già presente in `BENCHMARK_V2.md:275`. Non corretta in questo consolidamento; valutarla consapevolmente prima del commit senza alterare dati o rigenerare report.
- Nuove modifiche autorizzate: `PROJECT_BRIEF.md`, handoff aggiornato e questo nuovo documento. `.gitignore`, codice, README e rapporti non sono modificati da questo ciclo.
- Stato finale: `main` / `44b393a`, sei tracked modificati, 61 untracked, index vuoto; nessun commit/push. Tra gli untracked, 51 sono proposti per inclusione e dieci restano locali. Nessuna eliminazione/reset/sovrascrittura di file estranei agli aggiornamenti documentali autorizzati.

Prima di creare il commit servirà l’istruzione dell’utente; la proposta non è autorizzazione a committare né a formattare l’unità esterna.
