# InWiMe — handoff tecnico dopo Benchmark V2

Created: 2026-11-05, Europe/Rome.
Source thread / Source host: identificatori Codex non disponibili; macchina locale Katana/WSL2. Nessun trasferimento automatico.

## Goal

Benchmark V2 concluso. Priorità del prossimo ciclo: valutare accesso efficiente allo storage esterno da WSL2, prima dello sviluppo dell'executor MoE BF16. **Questo consolidamento autorizza solo documentazione e proposta Git: niente Briareus/MoE, nuovi benchmark, commit o push; risultati esistenti immutabili.** Non esiste ancora un runtime LLM completo.

## Acceptance criteria

Preservare evidenze e modifiche non committate. Verificare le sentinelle, proporre il piano di valutazione storage e attendere autorizzazione all'esecuzione. Non rilanciare automaticamente sweep/setup/generatori o iniziare l'executor.

## Communication preferences

- Reply language: italiano, lingua prevalente osservata; mantenere identificatori e comandi.
- Locale or time zone: Europe/Rome, indicata dall'ambiente; riferimento 2026-11-05.

## Applicable constraints

Leggere `AGENTS.md`, `PROJECT_BRIEF.md`, `README.md`, `BENCHMARK_V2.md` e `BENCHMARK_V2_STORAGE.md`. Le istruzioni correnti dell'utente prevalgono.

- BF16/F16 originale, nessuna quantizzazione implicita; checkpoint immutabili. Formati nativi inferiori eventualmente separati e documentati.
- CPU fallback obbligatorio; operatori GPU a tile anche per tensori oltre 4 GiB, con confronti numerici. Niente swap/Unified Memory come streaming dei pesi.
- MSI Katana GF66, i7-12700H, 16 GB RAM fisica, RTX 3050 Laptop 4 GiB; RAM WSL disponibile inizialmente circa 9,5 GiB, da rimisurare. Toolkit CUDA 12.4, driver API compatibile 13.5, GCC13/OpenMP.
- Nessun checkpoint grande senza approvazione dello spazio, pacchetto di sistema, riconfigurazione, modifica partizioni, migrazione dischi, push o commit senza autorizzazione. Un solo slot NVMe interno, occupato.
- Il proprietario conferma che il **Sabrent SSD USB da 512 GB può essere interamente svuotato e dedicato a InWiMe**. È disponibilità all'uso, non autorizzazione operativa a cancellare o formattare. Prima di ogni operazione distruttiva servono verifica inequivocabile e attuale del dispositivo fisico, piano con impatto/perdita dati e autorizzazione specifica all'esatta operazione. Disk 2, D: e `/mnt/d` sono riferimenti storici, non identità durature; non bastano seriali bridge mancanti/non univoci. Non toccare il Samsung/Windows interno.
- `https://github.com/francescodifilippo/spikingbrain-cpu` è un riferimento indipendente: non modificarlo o importarvi branch estranei. Non sostituisce la specifica di altri modelli.
- DeepSeek Harness solo client opzionale, niente UI agentica sostitutiva. Correttezza prima della velocità; una GEMM sintetica non prova token/s o eseguibilità LLM.

## Workspace identity

- Path: `/home/francesco/GIT/francescodifilippo/ooc-inference`.
- Source workspace kind: checkout locale salvato; nessun nuovo worktree.
- Destination compatibility: selezionare questa stessa directory con accesso ordinario, senza attraversare workspace diverse o cambiare branch.
- Branch: `main`; HEAD `44b393a209e38ea79cfc3f1170b32f54225634e0` (`init again`), precedente `80f4231` (`init`).
- `HEAD...origin/main`: 0 avanti / 0 indietro rispetto alla ref locale; nessun fetch, stato remoto live non verificato.
- SHA256 del diff tracked dopo consolidamento (`git diff --binary`): `60bd93f24563564cc0598e1857cc8248019307472de8b7a2669207f5e3051664`.

### Stato Git non committato

VERIFIED — checkpoint consolidato: sei tracked modificati, index vuoto. C++/test/Makefile e `.gitignore` puliti rispetto a HEAD. Prima del consolidamento erano cinque modificati; si aggiunge `PROJECT_BRIEF.md`. Non assumere lo stato storico iniziale “non è un repository”: adesso `.git` esiste.

| Tracked modificato | Modifica già presente |
| --- | --- |
| `BENCHMARK_V2.md` | Addendum storage, mantenendo lo sweep originario |
| `README.md` | Riproduzione e limiti storage |
| `tools/benchmark_v2.py` | Filtri `--buffers`, `--ns`, `--modes` |
| `tools/summarize_v2.py` | `--input`/`--output`, subset solo I/O |
| `tools/report_v2.py` | Inclusione addendum storage |
| `PROJECT_BRIEF.md` | Sabrent 512 GB, disponibilità del proprietario, gate distruttivo e priorità storage |

Untracked al consolidamento finale: 61 file. Comprendono 53 file sotto `results/v2/storage/`, `BENCHMARK_V2_STORAGE.md`, `tools/report_storage.py`, `tools/storage_setup.py`, `tools/storage_compare.py`, `tools/storage_cleanup.py`, `tools/diagnostics/windows_storage.ps1`, questo handoff e `docs/COMMIT_PROPOSAL-2026-11-05.md`. La proposta elenca esattamente 57 file da includere (sei tracked + 51 untracked) e dieci risultati accessori da lasciare locali. Nessun file viene eliminato o nascosto per rendere pulito lo stato.

Il diff precedente al consolidamento aveva 146 inserimenti / 14 cancellazioni; ora comprende anche il brief. Nessun commit, add, stash, reset, cambio branch o push effettuato. `git diff --check` segnala una riga vuota finale preesistente in `BENCHMARK_V2.md:275`: non corretta per preservare il risultato. In questo ultimo ciclo vengono aggiornati solo brief/handoff e creata la proposta; nessun codice o risultato è modificato.

## Source lifecycle

- Goal status: none
- Source archival authorization: unspecified; non richiesta.
- Source closure status: solo consegna documentale, nessun nuovo thread o archivio automatico; nessun benchmark attivo osservato al checkpoint.

## Completed and verified

### Architettura e componenti presenti

VERIFIED — implementati, non una proposta:

```text
raw BF16 read-only → pread O_DIRECT → buffer allineato
  CPU: decode BF16 → dot FP32 OpenMP/SIMD
  GPU: 2 slot pinned/device + 2 stream non bloccanti
       H2D → cuBLAS GEMM BF16/accumulo FP32 → D2H FP32
       evento prima del riuso → drain → riferimento FP64
```

Letture host bloccanti; async accoda CUDA mentre l'host legge il blocco successivo, sync attende ogni blocco. Due slot in entrambe le modalità GPU; pinned con `posix_memalign(4096)` + `cudaHostRegister`. Nessun thread/prefetch disco asincrono.

Layout: A `[blocco][m][k]` row-major, B k×n e C m×n column-major. cuBLAS vede A k×m con `CUBLAS_OP_T`. BF16 sintetico esatto (multipli di 1/16), accumulo/output FP32, riferimento FP64 dalla posizione assoluta. Nessun kernel `.cu` custom.

Cache GPU: prefisso statico persistente, non LRU/cache esperti; setup fuori timer, hit/byte espliciti. Assenti loader safetensors/shard, router e proiezioni MoE, attenzione/KV, tokenizer, adapter e server.

| File | Responsabilità |
| --- | --- |
| `src/benchmark.cpp` | CLI/dati, letture, CPU/GPU, slot/cache/eventi, confronto e CSV |
| `src/activity_trace.hpp` | CUPTI opzionale: memcpy/kernel e timestamp letture |
| `Makefile` | `all`, `test`, `profile` NVTX, `trace` CUPTI, senza installazioni |
| `tests/smoke.py`, `tests/cache.py` | Correttezza completa piccola, errori, cache |
| `tools/benchmark_v2.py` | Sweep riprendibile e telemetria; stop su VmSwap processo >0 |
| `tools/summarize_v2.py`, `tools/report_v2.py` | Statistiche/SVG e rapporto |
| `tools/profile_v2.py`, `tools/analyze_trace.py` | Acquisizioni separate e intersezioni GPU |
| `tools/storage_*.py`, `tools/report_storage.py` | Copie esclusive, confronto, contatori Windows, cleanup/report |

### V2 e particolare attenzione a n=1

VERIFIED — `results/v2/summary.json/csv`: **620 passate, 62 casi × 10**. Stessi 5 GiB BF16, k=4096, n=1/4/16, buffer 4/8/16/32/64 MiB; variano m/blocchi, non pesi/volume. 45 casi n×buffer×modalità + 5 I/O + 12 GPU/cache 64/256 MiB a buffer 8 MiB; non cache×buffer completo.

Mediane totali, cache=0, buffer 64 MiB/slot:

| n | CPU s | GPU sync s | GPU async s | CPU/async |
| --- | --- | --- | --- | --- |
| 1 | 1,711 | 2,143 | 1,548 | 1,10× |
| 4 | 1,884 | 2,178 | 1,515 | 1,24× |
| 16 | 2,710 | 2,079 | 1,474 | 1,84× |

n=1: vantaggio modesto, non trasferire lo speedup n=16 al decode. A buffer 8 MiB: CPU 2,129 s, sync 2,808 s, async 1,864 s. 64 MiB è un candidato misurato, non optimum universale.

Errore campionato zero. Massimi: RSS 473,55 MiB, pinned 129 MiB, device esplicito 272,25 MiB; swap processo e delta pswpin/out zero. NVML globale campionato 401 MiB vs snapshot CUDA globale 1087,31 MiB: non sommabili né picchi per processo. GPU max 53,5 °C/18,28 W, SM 210–2062 MHz. Dispersione/raw/grafico: `results/v2/{summary.csv,sweep/,buffer-comparison.svg}`.

Il preliminare `results/COMPARISON.md` resta distinto: 3 ripetizioni, n=16/buffer 8 MiB, CPU 2,864 s e async 2,081 s. Non sostituirlo con il V2.

### Profiling CUDA: confine delle evidenze

VERIFIED — **30 acquisizioni diagnostiche**, una per n×buffer×sync/async, cache=0, separate dalle 620 passate. `results/v2/activities/` e `trace-summary.json`: attività GPU reali, copertura completa byte H2D e **overlap H2D–kernel osservato zero** in tutti i casi. cuBLAS può lanciare più kernel per blocco.

n=1 async/buffer 64 MiB: CUPTI circa 415,718 ms H2D e 28,901 ms kernel per 5 GiB. I bracket CUDA dei CSV includono possibili attese/scheduling, non sono kernel puri. Nsight Systems 2023.4.4 ha acquisito API/NVTX ma nessuna attività kernel/memcpy nel primo SQLite; risolto separatamente il percorso dell'importer host.

Timestamp host/GPU CUPTI con offset incompatibile: **lettura↔GPU non quantificabile**. `host_gpu_correlated=false`; intersezioni solo fra attività GPU, nessuna traslazione arbitraria; SVG solo GPU. Nessun nuovo activity profiling nel confronto USB.

### Storage: confronto mirato concluso

VERIFIED — `BENCHMARK_V2_STORAGE.md`, `results/v2/storage/`: 60 passate, 10 I/O + 10 async n=1 per percorso, buffer 64 MiB, cache=0, stesso SHA256/binario.

| Percorso | Solo I/O s | GPU async s |
| --- | --- | --- |
| Samsung 990 EVO 2TB, ext4/VHDX WSL | 1,597 | 1,515 |
| Stesso NVMe, C:/DrvFS | 30,516 | 31,495 |
| Sabrent SSD USB, D:/DrvFS | 54,092 | 51,810 |

Async: C:/DrvFS 20,80× ext4, USB/DrvFS 34,21×; non attribuire tutto all'USB. Sabrent 512 GB (non 1 TB del brief), Realtek 0BDA:9210, UASPStor; velocità negoziata/NVMe-SATA non certificati. Windows osserva circa 49,99 GiB letti da D: per gruppo da 50 GiB: contatori globali, non NAND/esclusivi.

Errori numerici e VmSwap Linux zero; pagefile Windows non monitorato. Copie temporanee da 10 GiB eliminate (`cleanup.json`: `all_clean=true`); `datasets.json` contiene percorsi storici rimossi. Restano locali e ignorati da Git `data/weights-5GiB.bf16` e il preliminare da 512 MiB.

### Test superati e riproduzione

VERIFIED — PASS prima della chiusura dei benchmark; non rieseguiti per questo documento:

```bash
cd /home/francesco/GIT/francescodifilippo/ooc-inference
make
make test
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/smoke.py --gpu
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/cache.py
```

Smoke: riferimento FP64 completo piccolo, blocchi singoli/dispari/riusati, code, file corrotti/troncati e CLI invalida. Cache: n=1/4/16, sync/async, miss/hit parziali/totali, due passate, byte/pinned. Grandi: fino a 32 indici deterministici/blocco; 2560 confronti/passata a 64 MiB, non output completi.

Comandi prestazionali in README, solo su richiesta: sweep circa 3 TiB/decine di minuti. Profiling, summarizer e report generator scrivono artefatti: non usarli per verificare dati immutabili. Dopo cleanup, nuovo storage setup richiede preservare i risultati: non fare resume sui percorsi rimossi.

## Unverified and open

- UNVERIFIED — overlap lettura↔GPU come spiegazione dello speedup: plausibile, non provato dalla timeline. n=1 è una GEMM tipo decode, non generazione autoregressiva.
- UNVERIFIED — hit rate/cache esperti e optimum generale. Il prefisso 64/256 MiB copre 1,25%/5%; uno scan LRU oltre cache può avere zero hit.
- UNVERIFIED — prestazioni cold/raw, USB negoziato, baseline Windows nativa, temperatura SSD e picco VRAM esatto. O_DIRECT esclude cache Linux, non Windows/controller; prove storage sequenziali, clock dinamici non controllati.
- Da sviluppare: loader/layout/shard read-only, tile parziali e riduzioni K, routing top-k, proiezioni/attivazioni MoE e memoria intermedi. Oggi raw sintetico, CLI limitata a file 1 TiB/blocco 256 MiB/cache 2048 MiB: non supporto GLM da 1,5 TB.
- Valori sintetici piccoli esatti, non dinamica numerica di checkpoint reali. CPU OpenMP/SIMD, non BLAS ottimizzata. HWM RSS/pinned cumulativi per processo: la suite isola le modalità in processi distinti.

## Failures and discarded approaches

- Sandbox iniziale senza `/dev/dxg`: root/chmod non risolvevano. Accesso poi riuscito in CLI WSL senza sandbox come francesco, non root; verificare nuovamente l'ambiente della destinazione.
- `nvidia-smi` funzionava ma CUDA non rilevava dispositivi: anteporre `/usr/lib/wsl/lib` al solo comando ha risolto nel terminale reale. Non installare driver Linux o cambiare configurazioni globali per riprodurre.
- `tools/diagnostics/windows_storage.ps1` bloccato dall'antivirus: nessuna protezione disabilitata/aggirata. Non ritentare o considerare un exit code apparentemente riuscito come prova; leggere il log.
- Non ricavare overlap H2D–kernel dallo speedup o “correggere” arbitrariamente timestamp CUPTI.

## Next action

Dopo le sentinelle, proporre prima una valutazione storage: USB passthrough con ext4 nativo, VHDX ospitato sull'unità esterna e altre alternative effettivamente supportate dall'installazione (eventuale reader Windows approvato). Verificare in sola lettura supporto kernel/UAS/tooling, ownership Windows/WSL, privilegi richiesti e reversibilità; non assumere che ogni metodo di mount/passthrough funzioni. Definire test confrontabili a byte BF16 fissi e contatori di lettura, ma non avviarli ora. Nessun attach/detach, creazione VHDX, formattazione o modifica distruttiva è autorizzato da questo documento. Per operazioni distruttive applicare il gate del brief, anche se il disco è disponibile per essere svuotato.

### Futuro executor MoE — non avviato

Solo dopo il piano storage e nuova autorizzazione, proporre **un blocco MoE sintetico BF16**. I seguenti sono sviluppi futuri, non componenti esistenti:

1. Contratto CPU FP64/FP32: router/top-k con tie-break, normalizzazione, SwiGLU gate/up/down e somma pesata; confrontare tutti gli intermedi GPU BF16.
2. Sorgente read-only dtype/shape/offset/stride; separare planner/cache/backend. Riutilizzare slot/eventi, senza riscrivere il benchmark o imporne il layout al modello.
3. Budget per staging/workspace/attivazioni/output; candidato 2×64 MiB su ext4 e cache zero, fallback CPU e tensor tiling oltre VRAM. Testare code e futura riduzione K.
4. Piccoli test completi, poi sintetici grandi/campioni, byte e memoria. Routing/riuso prima di cache/prefetch. Adapter/safetensors/checkpoint richiedono specifica e spazio approvato; niente UI/server.

## Destination sentinel

Controlli minimi read-only, senza benchmark o generatori:

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short
git diff --cached --stat
sha256sum results/COMPARISON.md results/v2/summary.json results/v2/trace-summary.json
```

Attesi main/HEAD sopra, index vuoto e modifiche elencate. SHA256:

- `results/COMPARISON.md`: `f3f455bddc6babbaf662df02efea32c4dfb58746dfc0bd7b4ce31426be0646eb`.
- `results/v2/summary.json`: `358953ecf3c24f796d7fd9e6496c22210fb2cb28b5aa6d6fe7a4427fbd720ebe`.
- `results/v2/trace-summary.json`: `ebc1ae00f62c553911ed9e3184dd2e79bc98e85c3eb2a50797ef592e1c1c202c`.

Verificare JSON: 62 configurazioni/count=10/errore=0; 30 trace verified=true, host_gpu_correlated=false, overlap H2D–kernel=0; tre summary storage con due modalità/count=10/errore=0; cleanup all_clean=true. Binario locale ignorato: SHA256 `3ce352807037731e33bf9ffa03fa02f0cf28f5f0eb2313ecf5301de7b79b2d86`.

In caso di differenza: `HANDOFF REGRESSION`, fermarsi prima di mutare stato, senza reset o nuovi privilegi. Se coincide: `HANDOFF VERIFIED`, riassumere limiti e proporre il piano storage, senza iniziare Briareus/MoE o nuove prove.

## Recovery

- Source remains intact: checkout, risultati e lavoro non committato sopra; nessun branch/processo trasferito automaticamente.
- Backup packet: `/tmp/inwime-final-handoff-20261106-fwmo85e5/packet.md`, copia locale sorgente del consolidamento; non richiesta alla destinazione. Copia persistente autorevole: `docs/handoffs/HANDOFF-2026-11-05-benchmark-v2.md`.
- Prompt nuova chat nella stessa directory: “Leggi AGENTS.md, PROJECT_BRIEF.md, l'handoff e la proposta Git; esegui solo sentinelle read-only e proponi il piano storage, senza implementare Briareus/MoE, formattare, committare o rigenerare risultati”. Nessuna archiviazione autorizzata.
