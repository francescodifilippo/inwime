# InWiMe — audit tecnico di Strata e proposta del prossimo incremento

Data: **13 novembre 2026**. Attività svolta: analisi statica del sorgente, lettura delle evidenze conservate e inventario leggero dell'ambiente. Nessuna modifica al runtime, compilazione, esecuzione di test o benchmark, acquisizione di pesi, installazione, commit o push. Questo documento è l'unico nuovo file prodotto nel repository.

**Raccomandazione:** non adottare Strata come sostituto immediato di InWiMe e non mantenere due runtime in produzione per ottenere il requisito CPU-only. Strata offre componenti e idee utili, ma il suo percorso completo esaminato è GPU-centrico e le prove con poca RAM riguardano soprattutto modelli quantizzati. Per il prossimo incremento propongo una sola estensione: **sorgente safetensors BF16 multi-shard, letta in posto, attraverso il contratto di lettura limitata già disponibile**. È un passo verso checkpoint originali di modelli reali, non una promessa di accelerazione né la prova finale oltre RAM fisica.

**Indicazione aggiornata del committente:** «se possiamo migliorare non consideriamo nulla di consolidato». I commit sotto identificano prove e baseline; non rendono intoccabili codice o architettura. Anche kernel, catalogo, scheduler e KV sono sostituibili se il confronto dimostra un miglioramento e preserva i requisiti. Conservare le evidenze significa poter verificare una sostituzione, non vietarla. La scelta proposta deriva dai divari funzionali verificati, non da una preferenza a priori per il runtime esistente.

**Estensione richiesta: GPU opzionale a bassa VRAM.** CPU x86-64 sufficiente per l'intero runtime; GPU usabile quando disponibile e conveniente, anche sotto 4 GiB, senza quantizzazione implicita. Il **P0 resta caricamento di checkpoint reali safetensors BF16 multi-shard e inferenza completa nel budget**: la sezione 6 ne descrive soltanto il primo passo sul contenitore. La GPU è un **P1 indipendente**, non un requisito né un sostituto del P0. La nuova sezione 8 verifica M1a/V2, valuta il riuso Strata e propone un solo esperimento GPU circoscritto. Nessuna delle due implementazioni è avviata da questo audit.

## 1. Stato verificato di InWiMe

### 1.1 Revisioni e forza delle evidenze

| Oggetto | Stato verificato |
|---|---|
| Directory locale | `/home/francesco/GIT/francescodifilippo/ooc-inference` |
| Repository configurato | `https://github.com/francescodifilippo/inwime.git` |
| Branch e HEAD | `main`, **`f304ee9f19b6a89907d4ad10cbdce435640352fb`** |
| Incremento 4 citato nella richiesta | `8419dcfffebeb86f0c76e07d6cbac7a3732c9765`, predecessore di HEAD |
| Incremento 5 già esistente | HEAD: «Completa incremento 5: inferenza CPU BF16 oltre RAM verificata» |
| Working tree iniziale | Pulito, verificato con `git status --short` / `--porcelain=v1` |
| Tracking remoto | Il riferimento **locale** `origin/main` coincide con HEAD. Non è stata verificata live la pubblicazione remota e non è stato eseguito fetch |
| Test | Log finale conservato: **181 passed in 41.53s**; incremento 4: 152. Nessun test rieseguito nell'audit |

Fonti: storia Git locale; [rapporto incremento 5](RUNTIME_CPU_INCREMENT_5.md), [log finale](../results/increment5/review/pytest-final.txt), [audit probatorio](../results/increment5/review/probatory-audit.json), [fingerprint di consolidamento](../results/increment5/artifact-sha256-consolidated.json). Il log riguarda i sei moduli di test nominati nel comando, non una verifica universale di ogni programma del repository.

Il [PROJECT_BRIEF](../PROJECT_BRIEF.md) e l'[handoff del 5 novembre](handoffs/HANDOFF-2026-11-05-benchmark-v2.md) sono stati letti: descrivono una fase precedente. Vanno confrontati con i rapporti successivi; non autorizzano a ripartire dal benchmark iniziale. La [decisione architetturale](INWIME_ARCHITECTURE_DECISION.md) e il [catalogo delle tecniche](INWIME_TECHNIQUE_WATCHLIST.md) sono contesto progettuale, non misurazioni comparative fra architetture.

Nel resto del documento:

- **V**: verificato ora tramite file, Git o inventario leggero.
- **M-I**: misura InWiMe conservata, con i limiti dell'acquisizione originale; non replicata ora.
- **M-S**: misura pubblicata nel repository Strata, non riprodotta su questo host.
- **S**: conclusione da analisi statica, non prova di esecuzione.
- **P**: proposta o previsione condizionata, da misurare.

### 1.2 Ambiente attuale e ambiente delle misure

Inventario **V**, senza modifiche di sistema: kernel `6.18.58.1-microsoft-standard-WSL2`, x86-64, i7-12700H, 20 CPU logiche; `MemTotal=11.483.848.704` byte e swap configurato `8.589.934.592` byte. Le feature esposte comprendono AVX2, FMA, F16C e AVX-VNNI; **AVX-512 non è esposto**. La topologia presentata dalla VM non va reinterpretata come nuova misura dei core P/E fisici.

Al campionamento erano disponibili circa 10,13 GB di memoria WSL e 2.232.320 byte di swap risultavano occupati globalmente: non sono consumi di un benchmark eseguito adesso. `df -B1 .` riportava 920.908.042.240 byte disponibili nel filesystem virtuale. Non è lo spazio effettivamente disponibile sul volume Windows che ospita il VHDX.

L'[audit risorse del 12 novembre](../results/increment5/resource-audit.json), **M-I**, registra RAM fisica host **16.900.739.072 byte** e spazio libero Windows pre-generazione **185.963.954.176 byte**. Non sono stati riacquisiti ora CIM, driver CUDA, temperature o memoria GPU. La configurazione della RTX 3050 da 4 GiB è contesto storico; questo audit non ne certifica la disponibilità corrente e non la usa come prerequisito della proposta.

### 1.3 Cosa è già dimostrato e cosa no

L'incremento 5 ha eseguito un **decoder denso sintetico completo**, non un semplice accesso al file: 24 layer, width 4096, 32 teste, FFN 16384, vocabolario 257, contesto 16. Pesi generati direttamente in BF16, senza checkpoint FP32 convertito. Prefill `[1,7]`, tre decode con feedback greedy; quattro output `[118,101,118,101]`; lunghezze KV pubblicate 2, 3, 4, 5. Non c'è tokenizer e non c'è prova di qualità linguistica. [M-I: rapporto](RUNTIME_CPU_INCREMENT_5.md)

| Quantità della prova grande | Valore verificato negli artefatti |
|---|---:|
| Payload BF16 del checkpoint | 12.889.645.056 byte |
| Matrici integralmente elaborate per ogni token | 12.887.007.232 byte |
| Unione dei pesi realmente usati nei cinque token elaborati | 12.887.482.368 byte |
| RAM totale WSL durante le acquisizioni | 11.483.848.704 byte |
| RAM fisica host storicamente rilevata | 16.900.739.072 byte |
| Letture logiche del workload, comprese riletture | 64.437.125.120 byte per modalità |
| SHA iniziale, separata dall'inferenza | 12.889.645.056 byte per modalità |

**Oltre RAM WSL: sì. Oltre RAM fisica host: no.** Già le matrici calcolate per intero superano MemTotal WSL; la dimostrazione non dipende dal solo budget interno o da `RLIMIT_AS`. Un modello sintetico con questo grafo non dimostra supporto end-to-end di Qwen/DeepSeek/GLM. Le righe di embedding non consultate non vengono contate nei pesi realmente usati.

| Misura grande, un campione per modalità | Sync | Pipeline |
|---|---:|---:|
| Prefill, 2 token | 193,713 s | 152,685 s |
| Decode, secondi per ciascuno dei 3 token | 128,200 / 117,564 / 120,129 | 109,924 / 88,160 / 71,604 |
| Throughput dei soli decode | 0,00820 token/s | 0,01112 token/s |
| Workload completo | 560,040 s | 423,019 s |
| Piano delle allocazioni controllate | 24.022.223 byte | 26.668.239 byte |
| `ru_maxrss` | 53.776 KiB | 57.440 KiB |
| `VmHWM` | 55.873.536 byte | 59.498.496 byte |
| `/proc/self/io/read_bytes`, workload | 63.978.094.592 byte | 64.100.728.832 byte |
| Intersezione lettura/calcolo osservata | 0 | 2,319 s |

Fonti **M-I**: [riepilogo](../results/increment5/large-summary.json), [sync](../results/increment5/large-sync-1.json), [pipeline](../results/increment5/large-pipeline-1.json). RSS e HWM sono indicatori distinti, non sommabili; non comprendono la page cache globale. Limite di spazio virtuale 512 MiB; GPU non inizializzata, allocazioni GPU del runtime zero. `VmSwap=0` ai confini delle fasi non esclude transitori fra snapshot. Alimentazione AC registrata; temperatura, clock e potenza non controllati.

Il wall diminuisce del 24,47%, ma il servizio letture cambia di circa 134 s: **non si può attribuire il vantaggio ai soli 2,319 s di overlap**. I/O buffered, DONTNEED consultivo, cache Windows/controller non controllate. `read_bytes` attribuisce traffico di blocco al processo, non letture NAND. La SHA iniziale può scaldare parte delle cache, senza rendere residente tutto il checkpoint.

Anche il **GEMV completo da 6 GiB è già stato eseguito**: 6.442.450.944 byte, 3.221.225.472 coefficienti; 9,473 s sync e 10,422 s pipeline. Matrice non materializzata; output necessario circa 1,5 GiB, incluso nei picchi RSS di 1.601.004/1.603.228 KiB. Hash dell'intero output identico, quattro righe controllate indipendentemente. È una GEMV, non il forward completo del checkpoint storico. [M-I: risultati](../results/increment5/gemv-6g/summary.json)

Regressioni conservate: pipeline circa **+10,0%** sul GEMV da 6 GiB, **+253,4%** sul GEMV da 128 MiB con tile 8191, **+37,6%** sulla mediana del decoder medio, tre campioni. La concorrenza non è perciò un default prestazionale dimostrato migliore. [M-I: rapporto](RUNTIME_CPU_INCREMENT_5.md)

Correttezza: confronto completo indipendente sulle fixture piccole dense/MoE, a ogni fase; sul grande, hash logits/hidden/KV uguali fra modalità e **507 dot product su 169 proiezioni** verificati indipendentemente usando però input intermedi del runtime. Questo ultimo controllo è parziale. La coppia grande precede le ultime protezioni resize/alias del wrapper; kernel e matematica sono identici, con rebuild verificato. Non si retrodata una misura del wrapper finale. [M-I: revisione](RUNTIME_CPU_INCREMENT_5.md)

### 1.4 Storage: il dato DrvFS non è l'ultimo risultato

| Sessione e percorso | Async su 5 GiB BF16, mediana di 10 |
|---|---:|
| V2, interno ext4 WSL | 1,515 s |
| V2, stesso interno via C:/DrvFS | 31,495 s |
| V2, esterno USB via D:/DrvFS | 51,810 s |
| M1a, 7 novembre, esterno ext4 diretto in WSL | **12,015 s** |
| M1a, controllo interno ext4/VHDX | **1,642 s** |

Fonti **M-I**: [V2 storage](../BENCHMARK_V2_STORAGE.md), [M1a ext4](../results/m1a-native-ext4-20261107T075250Z/REPORT.md), [contatori M1a](../results/m1a-native-ext4-20261107T075250Z/read-volumes.json). M1a documenta già un percorso esterno ext4, non DrvFS, non USB/IP e non un nuovo VHDX. Le identità `/dev/sdX` storiche non sono identificativi stabili e non autorizzano nuove operazioni sui dischi.

Il rapporto storico 51,810/12,015 = 4,31 non isola causalmente il solo filesystem: sessioni, layout e carico host differiscono. M1a usa O_DIRECT senza fallback; il decoder dell'incremento 5 usa buffered I/O. M1a registra anche paging Windows e memoria host disponibile bassa, pur senza swap Linux osservato; temperatura SSD e velocità USB negoziata non sono verificate. Non estrapolare i GB/s di scansione sequenziale a letture selettive MoE. **Multi-NVMe fisico concorrente non è dimostrato**: percorsi logici multipli o interno più USB non equivalgono a tale prova.

## 2. Strata: versione, architettura e sorgenti effettivi

### 2.1 Snapshot e perimetro

Snapshot del branch `main`: **`13eedf552417a6a791a7d13c6597e24300ee3a09`**, data committer **12 novembre 2026, 06:18:38 UTC**. [Commit esaminato][S0]

[AGENTS.md][S1] dichiara prodotto 0.2.13, engine 0.1.59. Il [manifest release][S2] dichiara tag `v0.2.13`, pubblicazione 9 novembre, Terminal `0.2.13.1`, Control `0.1.59.1`, engine `0.1.59`. Il [CMake dello snapshot][S3] riporta invece `project(strata VERSION 0.1.59.1)`. Pertanto questo audit riguarda **il commit**, successivo alla release dichiarata, non un binario che si presume identico alla release.

Letti l'albero Git ricorsivo non troncato, AGENTS, documenti tecnici, build, dipendenze, loader, sorgenti esperti, kernel CPU, router GPU, sessione/verificatore/MTP, percorsi KV, prefill, server e Harness. Per i file molto grandi la lettura analitica è per funzioni e blocchi pertinenti, non una revisione esaustiva di ogni riga. `generate.cpp` è stato acquisito tramite blob Git `6eea65a2217f05531b4fe5f6d08224d94fbd7349`, dopo che l'API contents non ne restituiva il contenuto. Nessun checkout, build o test Strata è stato eseguito localmente. Backend Metal/SYCL e DeepSeek sono considerati nei confini di build e nei componenti pertinenti, non certificati integralmente.

La documentazione di installazione contenuta in AGENTS è stata esaminata come fonte: nessuno dei suoi comandi di download o avvio è stato eseguito.

### 2.2 Il percorso di esecuzione

```text
Client / DeepSeek Harness
          |
serve/server.py: API, coda/sessioni, protocollo verso il processo engine
          |
src/program/generate.cpp: configurazione, cache, adattamento, generazione
          |
session / Verifier / prefill + MTP o altri proponenti opzionali
          |
GPU: dense, attenzione/stato ricorrente, routing, shared experts, head
      + esperti residenti e una quota dei miss
CPU: pool dei restanti esperti, anche mentre la GPU lavora
          |
ExpertSource: arena RAM oppure file/GGUF + cache/staging e letture SSD
```

**S:** il modello centrale è una gerarchia GPU/RAM/file con spartizione del lavoro, non un decoder universale CPU che usa una GPU soltanto facoltativamente. La presenza di sorgenti `.cpp`, worker CPU o di una configurazione CMake senza CUDA non cambia questo confine. [Build][S4], [sessione][S14], [driver][S5]

### 2.3 SSD/NVMe, granularità e concorrenza

**S — accesso ai pesi esperti.** `FileExpertSource::open`, `open_gguf`, `spans`, `copy_from_files` e `blob` in [expert_source.cpp][S6] supportano `experts.bin` oppure viste dei file GGUF originali del pack. Nel secondo caso gate/up/down possono essere tre intervalli fisici, con file e offset separati. `spans` produce tre span o un blob contiguo; `prefetch`/`fill_many` assemblano gli esperti richiesti. La granularità principale è **esperto o proiezione dell'esperto**, non una tile numerica arbitrariamente piccola di ogni tensore. Un numero di slot limitato non rende indipendente dalla dimensione dell'esperto il buffer necessario.

**S — buffered e page cache.** `set_io_prefetch` apre file read-only e applica `POSIX_FADV_RANDOM`; otto thread sono il valore di ripiego se non specificati. `io_worker` può leggere un intero span con `pread_full` in scratch per popolare la page cache, oppure riempire staging. Usa la residenza delle pagine per evitare alcune letture; tali controlli non sono prova di residenza futura o di coldness fisica. La coda semplice limita a 512 job ed elimina predizioni vecchie; `warm_scored` introduce, se abilitato, ammissione per confidenza, scadenza e budget di byte per finestra. `io_worker` dà precedenza alla domanda e scarta job scaduti. `io_budget_now` usa banda osservata e durata della finestra, con bootstrap di 256 MiB. È un budget di prefetch, **non un tetto RSS globale**. [S6, funzioni citate, righe 1025–1408]

**S — direct I/O.** `open_direct`, `set_unbuffered`, `read_direct`, `fill_many` hanno una via distinta. Linux usa O_DIRECT e Linux native AIO (`io_setup`, `io_submit`, `io_getevents`), **non io_uring**. Allinea finestre a 4096 byte, fonde intervalli adiacenti/sovrapposti fino a 32 MiB, con profondità massima 256 richieste per thread. Windows usa `ReadFile` overlapped/unbuffered e può fondere anche gap fino a 1 MiB entro richieste da 32 MiB: può quindi aumentare i byte fisici rispetto agli utili. `fill_many` lavora per gruppi di 16 esperti, fino a quattro thread per batch grandi. Gli scratch allineati crescono con il batch. [S6, righe 2395–2458 e 2754–3013]

Non confondere questo percorso con [platform/direct_file.cpp][S7]: `DirectFile::submit/wait` su Linux usa un pool di thread con `pread` su descrittore O_DIRECT; su Windows ha completamenti dedicati. Quest'astrazione non incorpora da sola un budget globale e il chiamante deve controllare anche la lunghezza realmente completata. Copiarla integralmente non aggiungerebbe automaticamente i contratti di errore di InWiMe.

**S — scelta del percorso.** `file_tier_probe`, `file_tier_unbuffered`, `set_unbuffered` e il blocco mmap del driver scelgono in funzione della RAM e del comportamento del percorso file. Il driver non onora l'opzione unbuffered in tutte le modalità: senza copia RAM esplicita la page cache può essere il tier utile. Non è corretto concludere «O_DIRECT è sempre migliore». [S6, righe 797–947, 2534 ss.; S5, righe 7549–7648]

**S — overlap e lifetime.** I worker possono leggere mentre CPU/GPU calcolano; il prefill e i cache refill hanno proprie copie asincrone. `claim_stage` riutilizza buffer secondo età/uso, evita buffer busy e attende l'evento CUDA dell'ultimo lettore prima di sovrascrivere staging pinned. In caso di submit AIO parziale, `read_direct` drena quanto già inviato prima di restituire un errore. Queste regole sono utili, ma concettualmente già presenti nel drain/lease di InWiMe. Non sono una prova di guadagno su questo NVMe.

**Limite statico rilevante:** `claim_stage` può allocare un altro blob quando nessuno è riusabile; a 512 buffer stampa un avviso, non applica lì un hard cap. Scratch per thread, staging, mapping, copie pinned e cache vanno contabilizzati insieme. Il commento su circa 257 buffer/0,4–0,6 GiB riguarda geometrie quantizzate tipiche, non è una garanzia per esperti BF16 grandi. [S6, righe 1865–1954]

### 2.4 MoE: routing, cache e predizione

**Routing effettivo.** [router_top10.cu][S10] implementa il percorso con softmax globale, top-k stabile, raccolta e rinormalizzazione con clamp. [native_router.cu][S11] ha kernel per geometrie specifiche, fra cui 512/10 e 256/8, con la propria aritmetica. Non è un router universale da sostituire senza controllare il modello. InWiMe usa la softmax sui selezionati e tie per indice minore nel proprio schema sintetico: uguaglianza matematica in casi ordinari non autorizza a ignorare clamp, ordine delle somme o precisione.

**Tre politiche diverse da non confondere.**

1. `ExpertCache::admit` assegna slot e **non espelle**; può distribuire ammissioni per layer. `replace` aggiorna la tabella di residenza: il completamento delle copie è una **precondizione a carico del chiamante**, non un'attesa eseguita dal metodo. Anche `fill_slot` accoda la copia e lascia l'ordinamento al chiamante. [expert_cache.hpp][S8]
2. Il driver implementa adattamento per frequenze di routing decadute: candidati mancanti contro residenti meno usati, soglia di vantaggio e numero di scambi limitato. Non è semplicemente LRU. Nell'opzione dichiarata: intervallo 4, decay 0,7, gain 1,5; i rami e i default per hardware possono differire. Il lambda `adapt` di generazione, righe 20704–20804, usa candidati con conteggio almeno 2 e una soglia letterale 1,5; il percorso serve ha gestione separata, anche asincrona. Non attribuire automaticamente un parametro a tutti i rami. [driver][S5]
3. Lo staging file usa età e recenza per riuso; il tier RAM complementare tenta di evitare copie degli esperti già sulla GPU (`make_cache_complement_plan`, `pin_cache_complement`, scambi). La deduplicazione RAM/VRAM non ha un beneficio equivalente in una macchina usata solo con CPU. [S6]

**Predizione.** `RouterLookahead::submit/run` applica router di layer successivi ad attivazioni disponibili in anticipo; calcola dot BF16 su CPU con AVX2/AVX/scalare, propone esperti e assegna confidenza dal rango, ridotta con la profondità. È una **stima per ordinare letture**, non il routing autorevole. Le dipendenze del forward impediscono di conoscere esattamente tutti gli esperti futuri senza calcolare prima i layer. `ForesightSwap::predict/take` può anticipare copie GPU; `warm_scored` limita traffico sprecato. [S6, righe 3083–3342 e 6495 ss.]

**Correttezza da proteggere:** cambiare sola disponibilità dei byte può mantenere i risultati; cambiare dove/come vengono calcolati può cambiare rounding e ordine. Il driver stesso annota che `--adapt-async` non è bit-exact fra esecuzioni. `native_route_resident` sostituisce alcuni esperti con altri residenti entro un margine: è una modifica dei risultati di routing, non una semplice ottimizzazione I/O. Anche tail-skip/speed projection/pruning sono fuori dalla linea lossless richiesta. [S5 righe 897–909; S11 righe 254 ss.; S6 `tail_skip_rank`]

### 2.5 CPU x86-64 e dipendenze GPU

| Componente Strata | Cosa fa realmente | Conseguenza per InWiMe |
|---|---|---|
| `cmake/backends/cpu.cmake`, `strata_kernels_cpu` | Pool esperti, dispatch ISA, libreria spec e test host senza toolkit GPU | Riutilizzabili in isolamento, non equivalgono al decoder |
| `native_expert.cpp`: `native_fmt`, `native_quant_act`, `native_gu_rows`, `native_down_rows` | Dot ggml/native dei formati supportati, attivazioni nel tipo richiesto dal dot | Quantizzazione delle attivazioni e layout da valutare; non kernel BF16 lossless intercambiabile |
| `iq_avx2/iq_avx512`, `kq_avx2`, `nvfp4_avx2/avx512`, `tq_*` | Percorsi SIMD specializzati per rappresentazioni compresse | Gran parte del beneficio richiede quei formati |
| `kq_avx2.cpp::bf16_rows_dot_multi` | Dot del router usato anche dal lookahead | Utile riferimento SIMD, ma somme FP32/ordine diverso dal kernel FP64 attuale |
| `cmake/strata_gpu_engine.cmake` | `strata_engine`, prefill e `strata` solo dentro CUDA/HIP; link runtime GPU | Nessun eseguibile completo CPU-only ottenuto semplicemente spegnendo CUDA |
| `generate.cpp`, `WeightTable::load`, `session`, `Verifier` | Arena GPU, kernel e graph/eventi, gestione KV e verifica sul backend device | Porting completo richiederebbe implementare interi operatori e stato host |
| `src/dsv41/ds4_gguf_cpu.cpp::mx_cpu_rows` | Corsia CPU per esperti MXFP4; benchmark host specifico | Non dimostra l'intero DeepSeek CPU-only |

Fonti **S**: [build CPU][S9], [build engine][S4], [native experts][S12], [kq AVX2][S13], [driver][S5], [corsia DeepSeek][S30]. La build C++ usa C++20/CMake ≥3.24. ggml-base/ggml-cpu e kernel MMQ sono vendorizzati dal pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`, con modifiche documentate. Server/tooling Python hanno dipendenze bloccate in `requirements.txt`; NumPy serve a tooling e server, non rende il nucleo un runtime PyTorch. GUI, Node/Tauri e bundle Harness non sono necessari a portare un algoritmo di lettura. [Dipendenze][S29], [vendoring][S28]

Su questo i7 non si può pianificare AVX-512. Un eventuale kernel BF16 AVX2 deve avere fallback e controlli numerici propri; il riuso del dispatch ISA è separabile dal riuso dei dot quantizzati. Inoltre una build CPU dei test prova componenti: non prova un server con modello reale senza GPU.

### 2.6 Memoria: cosa resta residente

**Dense.** `WeightTable::load` ha staging sorgente da 8 MiB e destinazione da 16 MiB, ma carica nella weight arena già dimensionata. `generate.cpp` alloca quell'arena con `cudaMalloc(pool_bytes)`. **Staging limitato durante il caricamento non significa dense out-of-core durante ogni token.** I percorsi native hanno ulteriori mapping/rappresentazioni, ma non costituiscono il GEMV CPU generico a tile richiesto da InWiMe. [S15, righe 176–258; S5, righe 5215–5259]

**Esperti.** Arena RAM residente/pinned, oppure file-backed e copie selettive. Cache GPU dimensionata dopo altri consumi. `auto_cache_slots` aggiunge una riserva per allocazioni tardive: 256 MiB come margine derivato da misure su alcune GPU, non stima universale. Oltre ai pesi vanno inclusi scratch, pool, indici, profili e copie di scambio. [S6; auto_cache.hpp nel sorgente]

**KV.** `KvHostPools` e `KvStreamMap` definiscono una copia autorevole in **RAM pinned** e un sottoinsieme di pagine in VRAM, risolte con CLOCK e copia dei miss. I formati includono FP16, INT8, Q4 e K8V4. Non è una KV cache genericamente paginata su NVMe. Ridurre VRAM può aumentare RAM host; INT8/Q4 modificano precisione dello stato, anche mantenendo invariati i file dei pesi. [KV contract][S16], [kernel][S17]

`Verifier::run/commit`, `capture_commit` e i percorsi batch gestiscono stato temporaneo e prefisso accettato, inclusa ricorrenza/indice del modello. `serve_stop_bound` limita il commit ai token emettibili entro EOS e `max_new`: il commit esaminato estende il guard oltre HIP. Questo è un esempio concreto di quanto sia delicata l'integrità conversazionale. Non è una dimostrazione che qualsiasi errore device sia atomico come il doppio banco di InWiMe. [Verifier][S18], [guard][S19]

**Page cache e RSS.** mmap non prenota automaticamente RAM per tutto il file, ma pagine residenti, copie RAM e scratch possono coesistere. Le decisioni in `host_available_memory` considerano anche cgroup e memoria recuperabile. Non emerge un unico tetto provato di 16 GiB che comprenda tutto il sistema e tutte le modalità; i budget di cache non sostituiscono RSS/HWM, locked memory, commit Windows e page cache.

### 2.7 MTP, speculative decoding e batching

**MTP.** `MtpDrafter::load/bind/draft` usa pesi e stato di un proponente specifico; non è una funzione aggiungibile a qualunque checkpoint senza MTP. `make_q4_dense` e `make_q4_head` sono percorsi espliciti di conversione del drafter. La verifica è nel modello principale, attraverso finestre `Verifier`; il commit mantiene il prefisso accettato. [MTP][S20], [Verifier][S18]

**Speculation distinta da MTP.** `src/spec/` contiene controller, draft policy e suffix drafter/prompt lookup. Questi possono proporre token senza un MTP addestrato. Il caso greedy richiede confronto con il target e gestione del primo rifiuto; sampling distribuzionalmente corretto richiede anche le probabilità e una regola di accettazione/correzione adeguata. La presenza di `spec_prob.hpp` e relativi test non prova automaticamente equivalenza di ogni combinazione di flag, batch e backend. Non equiparare «qualità simile», top-1 agreement e identità dei logits. [Spec probability][S21]

**Riuso dei pesi.** Nei kernel CPU multi-token l'iterazione sui token è interna al blocco di pesi: unpack e traffico dei pesi possono essere ammortizzati. Nel prefill si raggruppano token per esperto; nel batching più sessioni condividono passaggi mentre KV e posizioni rimangono distinti. `Verifier::run_slot_rows/commit_slot_prefixes` e il prefill realizzano percorsi diversi dal semplice decode di un token. [S13; S18; prefill][S22]

Modello di costo **P**, non previsione misurata: per una finestra speculativa,

`tempo/token accettato ≈ (tempo draft + tempo verify + I/O miss + commit) / token effettivamente emessi`.

Per un MoE la verifica legge l'unione degli esperti dei token della finestra. Se le selezioni divergono, tale unione e le letture speculative inutili possono cancellare il beneficio. Su CPU con pochi buffer servono operatori multi-token e una transazione KV capace di conservare esattamente il prefisso; non basta aggiungere un ciclo al decoder attuale. Batching può migliorare throughput aggregato e peggiorare latenza della singola richiesta e RAM. Sono candidati **P2** dopo un modello reale corretto.

`STRATA_KV_PREFETCH` anticipa upload della KV da RAM a GPU durante il **prefill**. `src/prefill/prefill.cpp`, `kv_prefetch_after` e il drain proteggono eventi e buffer. Il documento relativo dichiara vantaggi storici su una build combinata, nessun vantaggio decode e validazione pure-upstream ancora necessaria: non trasferisco quei numeri a questo snapshot o a CPU/NVMe. [S22; documento KV prefetch][S23]

### 2.8 Formati e pesi originali

Separare contenitore, precisione memorizzata e aritmetica:

- Letture a offset, code, contatori, backpressure e politiche cache sono **idee indipendenti dalla quantizzazione**; dimensione dei blob e limiti vanno ricalcolati per BF16/F16.
- GGUF non significa necessariamente quantizzato. Il reader riconosce anche BF16/F16/F32; questo **non dimostra** che tutti gli operatori e i pack del modello funzionino in full BF16. `GgufFile` e `check_architecture` distinguono metadati e compatibilità del modello. [reader][S24]
- `tools/iq_pack.py` può mantenere byte quantizzati già presenti nel GGUF e i float nei tipi attesi; `--compat-bf16` dequantizza/arrotonda alcune proiezioni e non ricostruisce il checkpoint originale. Il percorso F32→F16 e l'eventuale requantizzazione MTP hanno ulteriori implicazioni. Non adottarli come conversioni implicitamente lossless. [pack][S25], [loader][S15]
- `native_experts` in Strata significa lavorare sul tipo del pack/GGUF senza riconvertirlo in un altro pack canonico: **non garantisce che il pack coincida con i pesi pubblicati originalmente dall'autore del modello**.
- NVFP4/MXFP4 possono essere rappresentazioni native di uno specifico checkpoint, oppure una conversione: la provenienza va attestata per quel checkpoint. I loro kernel non soddisfano il requisito BF16 per il solo fatto di essere chiamati native.
- InWiMe CPU oggi supporta **BF16** nei manifest v1/v2 propri. F16 è un obiettivo del progetto, non una capacità già dimostrata del decoder corrente; safetensors e GGUF non sono ingressi già supportati.

### 2.9 API e `integrations/harness`

[serve/server.py][S26] contiene il protocollo `Engine`, `MockEngine`, `StrataEngine` e handler reali per `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/responses`, oltre a salute/status e gestione. Chat e Messages hanno percorsi streaming; Responses è implementata come adattamento stateless, non come intero servizio hosted OpenAI. Il server tratta tokenizer/template, tool-call, reasoning, coda, cancellazione e sessioni; copiare il file intero trascinerebbe semantica del prodotto e protocollo engine.

In [integrations/harness][S27]:

- `profile/cordis.patch.yml` configura `strata-local`, `openai-completions`, `baseURL` su loopback `/v1`; indica context 32768 e max output 2048. Sono valori del profilo, non limiti già supportati da InWiMe.
- `_bridge.py::start_worker` e `bridge_e2e.py` avviano `serve.server --engine mock`; `start_agent` usa il CLI DSH e la patch Cordis. Controllano provenienza provider/modello e sessione durevole, **non logits o inferenza reale**.
- `tool_flow_e2e.py` usa modello scripted/mock per verificare tool flow, decisioni e stop. Non è stato eseguito in questo audit.
- `profile/package.json` elenca i bundle DSH base/web-app senza fissare lì una versione delle dipendenze. Il registro `strata_cli/launch/harness.py` cita DSH `0.2.1-alpha.3` come versione testata e l'integrazione nell'app Strata; non certifica ogni futura installazione DSH.

**P:** riusare più avanti il contratto e piccoli test di compatibilità, puntando Harness all'API InWiMe. Non serve importare il desktop Strata, sostituire Harness o tenere il motore Strata accanto al motore InWiMe per servire la stessa richiesta. L'integrazione HTTP va dopo tokenizer/template e modello realmente supportato.

## 3. Matrice comparativa delle implementazioni

Tutte le possibilità di miglioramento restano aperte. «Già presente» significa che una riscrittura deve dimostrare un vantaggio rispetto a una baseline esistente, non che sia vietata.

| Tecnica / sorgente Strata | InWiMe attuale, funzione/modulo | Differenza e decisione proposta | Beneficio potenziale / complessità / rischio |
|---|---|---|---|
| Indice file e span GGUF: `GgufFile`, `FileExpertSource::open_gguf/spans` | `core.py::Catalog`, `Catalog.spans`; v1/v2, segmenti anche interni al tensore | Riusare l'idea di lettura in posto; aggiungere un adapter safetensors, non importare il parser GGUF per leggere un altro formato | P0 accesso a originali, evita copia completa su disco; nessun calo byte/token atteso; media; offset/hash/layout |
| Letture concorrenti: `fill_many/read_direct`, `DirectFile` | `FileBackend.readinto`, `ReadWorkers`, `PipelineCoordinator.submit/_load/wait/stream` | Esistono già slot bounded, backpressure e drain; un nuovo backend I/O va confrontato, non sovrapposto a un secondo scheduler | P1 meno attesa potenziale; media/alta per direct AIO; short read, alignment, lifetime, DrvFS |
| Lettura di interi blob/proiezioni | `gemv.py::streamed_linear`, tile multi-riga e riduzioni spezzate | Aggregazione righe già implementata. Esperti BF16 possono richiedere tile più piccole del blob Strata | P1 tuning granularità; bassa per sweep, alta per scheduler; amplificazione e RAM |
| Predizione `RouterLookahead`, `warm_scored/io_worker` | `_route` calcola selezione esatta; pipeline anticipa soltanto regioni già note | Portare in futuro algoritmo come hint separato, mai autorità del routing | P1 attesa inferiore con hit utili; byte e RAM possono aumentare; alta; starvation/domanda e stale buffer |
| Cache frequenze decadute, `ExpertCache`, lambda `adapt` | Runtime senza cache dei pesi; `_ffn` rilegge gate/up/down selezionati. Il benchmark V2 ha già una cache GPU di prefisso statico | Adattare policy RAM/VRAM dopo verifica del backend numerico; il prefisso V2 non è cache MoE. Evitare porting completo e modifiche del routing | P1 meno miss intertoken se località; media/alta; eviction durante lease, identità file, crescita |
| Copia RAM complementare a VRAM | Nessuna duplicazione GPU nel runtime CPU | Utile nel futuro percorso ibrido: una copia residente facoltativa, sorgente immutabile sempre rileggibile; beneficio CPU-only nullo | P1 risparmio RAM fino ai byte effettivamente deduplicati; alto accoppiamento e rischio fallback |
| `PcieModel::sample/solve/pick`, `expert_pool_dispatch_multi` | M1a ha GPU async a due slot, ma nessun planner CPU/GPU del runtime | Adattare il modello di costo dopo misure compatibili con la numerica InWiMe; non importare dispatch/quantizzazione Strata | P1 minore cammino critico potenziale; alta; costo PCIe, contesa RAM e ordine delle somme |
| SIMD e pool CPU: `native_expert`, `kq/iq/nvfp4/tq` | `math_cpu.py::NativeMath`, `src/cpu_kernel.c::inwime_expand/inwime_linear` | Rivalutare kernel e precisione delle arene; algoritmi SIMD adattabili, dot quantizzati non intercambiabili con BF16 originale | P1 compute/traffic RAM migliori; media/alta; FP32/FMA/riassociazione e tie routing |
| Verifica multi-token e prefill per esperto | `CPURuntime.prefill` causale, un token alla volta | Riuso pesi fra token nuovo, distinto dall'aggregazione delle righe già presente | P2 meno letture per token; alta; memoria attivazioni, ordine e causalità |
| KV RAM↔VRAM CLOCK, snapshot e commit parziale | Due banchi FP64, `decoder.py::prefill`, `_published=(bank,length,generation)` | Importare requisiti di visibilità/stop; valutare separatamente precisione e copy-on-write, non sostituire subito la transazione | P1 futura memoria, P2 spec; alta; vecchio KV alterato in errore |
| MTP / suffix draft / `Verifier::commit` | Assenti | Portare algoritmo solo dopo operatori e stato completi; niente import completo `Verifier` | P2 possibile ammortamento; molto alta; rifiuto draft, EOS, sampling |
| Telemetria `io_counters`, stage wait, cache stats | `telemetry.py::Timeline`, stats reader, RSS, blocchi e contabilità per tensore | Estendere solo contatori mancanti quando si aggiunge una tecnica; conservare denominatori | P1 attribuzione utile; bassa/media; confondere logico con fisico |
| API/DSH, `serve` e `integrations/harness` | Nessun server LLM pubblico del runtime CPU | Adattare contratto e test, non l'intera app | P2 per questa fase; media dopo modello; framing/tool-call/cancel |

Riferimenti locali: [core.py](../inwime_runtime/core.py), [pipeline.py](../inwime_runtime/pipeline.py), [decoder.py](../inwime_runtime/decoder.py), [gemv.py](../inwime_runtime/gemv.py), [math_cpu.py](../inwime_runtime/math_cpu.py), [kernel C](../src/cpu_kernel.c), [telemetria](../inwime_runtime/telemetry.py). Riferimenti Strata: sezioni 2.3–2.9 e link S4–S27.

### 3.1 Cosa potrebbe essere migliorato anche se già verificato

**Catalogo:** oggi unisce indice fisico e schema del decoder sintetico. Questa giunzione limita nuovi contenitori e modelli. Una separazione minima fra sorgente e validazione matematica è giustificata già nel prossimo passo; non occorre un framework esteso.

**Rappresentazione numerica:** BF16 viene espanso in FP64 per le tile; da 2 a 8 byte per peso in staging numerico, prima del traffico di lettura del kernel. È un costo reale di progetto, mantenuto per correttezza. Un kernel che consumi BF16 direttamente può ridurre RAM e traffico senza quantizzare i pesi, anche mantenendo accumulo FP64. Un percorso FP32/AVX2 può essere più veloce ma richiede un distinto contratto numerico. Non si assume che l'implementazione corrente sia ottima; i soli 17,876 s di GEMV contro 512,485 s di servizio letture del grande sync suggeriscono però di non promettere un grande guadagno wall dal solo compute. [M-I: grande sync; S: kernel e arene]

**KV:** il doppio banco e la copia integrale a ogni chiamata hanno costo proporzionale al contesto. Copy-on-write o journal potrebbero ridurlo, ma dovrebbero conservare byte e descriptor del prefisso pubblicato anche dopo errore al secondo shard, cancellazione e chiamata nativa fallita. È una sostituzione lecita, non scelta insieme al nuovo loader.

**Pipeline:** risultati piccoli/medi sfavorevoli e scarso overlap nel grande giustificano riesame del numero di slot, granularità e scelta sync/pipeline. Non giustificano né eliminarla né renderla default per tutti i carichi senza confronto controllato.

### 3.2 Beneficio atteso senza inventare speedup

Per dense, senza riuso intertoken, una passata deve leggere i pesi impiegati dal grafo. Per MoE il volume logico è dense più somma degli esperti **effettivamente selezionati e non già disponibili**. Idealmente il tempo è almeno il massimo fra servizio storage e compute, più dipendenze e overhead; con scarso overlap si avvicina alla somma. Il bandwidth utile deve essere misurato sullo stesso pattern, filesystem e stato cache.

Una cache di capacità `C` byte guadagna soltanto sui riusi che sopravvivono alla sostituzione; una scansione dense più grande di `C` può avere quasi zero hit. Il prefetch di `P` byte inutilizzati aggiunge traffico e pressione per quei byte. In una finestra di token, riusare un esperto comune può risparmiare letture; un'unione di esperti quasi disgiunta no. Queste sono relazioni di costo **P**, non misure sul Katana.

## 4. Opportunità, priorità e licenze

### 4.1 Ordine raccomandato

| Priorità | Candidato | Beneficio / rischio | Forma di riuso |
|---|---|---|---|
| **P0 — scelto** | Adapter sorgente safetensors BF16 multi-shard | Rimuove un ostacolo ai checkpoint originali senza cambiare matematica; rischio circoscritto agli offset, metadati e integrità | Adattare idea di sorgente indicizzata in posto; implementazione conforme alla specifica safetensors, non copia loader Strata |
| **P0 — prova ancora necessaria** | Forward con unione dei pesi usati > RAM fisica host | Chiude la prova C, non richiede una nuova cache o MTP | Riutilizzare/rivedere generatore e verifier InWiMe; nuova allocazione grande solo con preventivo approvato |
| **P0 — successivo traguardo funzionale** | Primo adapter esatto di un modello pubblico, tokenizer/template e logits upstream | Distingue prototipo numerico da LLM reale | Specifica dell'architettura originale; parti Strata solo se equivalenti e verificabili |
| **P1** | Backend direct I/O opzionale e confronto buffered, senza cambiare scheduler | Potenziale riduzione page-cache/copie; rischio medio/alto e dipendente dal filesystem | Portare regole di alignment/completion, misurare prima di adottare AIO |
| **P1** | BF16 consumato direttamente dal kernel, poi eventuale SIMD | Meno espansione/memoria temporanea; guadagno wall da verificare | Adattare idea e dispatch; non portare dot quantizzati come se fossero BF16 |
| **P1** | Cache RAM BF16 a capacità fissa | Risparmio I/O MoE se c'è riuso; nessun vantaggio universale dense | Portare una politica semplice, confrontare LFU-decay/LRU a parità budget, non entrambe nel primo esperimento |
| **P1 — filone indipendente** | Backend GPU opportunistico con budget riducibile e fallback CPU; poi eventuale cache VRAM MoE | M1a prova già streaming GPU bounded, ma non compatibilità con il runtime FP64. Vantaggio nullo o negativo possibile | Prima il singolo esperimento numerico della sezione 8.6; policy Strata adattabili, non secondo runtime |
| **P1** | Prefetch predittivo con domanda prioritaria, deadline e budget | Può nascondere miss; può saturare storage e peggiorare latenza | Adattare algoritmo, dopo misurazione della località su un modello vero |
| **P1** | Stato KV più economico con pubblicazione atomica | Possibile RAM/copie inferiori; alto rischio di regressione | Progetto proprio, verificato contro doppio banco |
| **P2** | Prefill multi-token, batching, MTP/speculation | Ammortamento pesi, ma più attivazioni, esperti e stato temporaneo | Porting algoritmico, non import runtime |
| **P2** | API/Harness e multi-PC | Fruibilità e distribuzione dopo core corretto | Piccolo adapter HTTP; distribuzione separata dal filone GPU P1 |

L'ordine non implica che tutte le tecniche saranno implementate. Un risultato negativo valido può portare a non intervenire. La prova C richiederà più di 16.900.739.072 byte **unici davvero elaborati**, non padding, pesi inutilizzati o riletture dello stesso payload; l'RSS piccolo da solo non la sostituisce.

### 4.2 Perché non incorporare un componente completo adesso

`FileExpertSource` accoppia formato esperti, CUDA, residenza GPU, scambi RAM, pool e geometrie del modello; `Verifier` dipende da graph/device state; il server incorpora semantica delle famiglie e del prodotto. Portarli completi avrebbe complessità maggiore dell'adapter delimitato e introdurrebbe requisiti contrari al CPU-only. Piccoli helper di policy o validazione possono essere estratti in futuro, con il loro test e un contratto esplicito.

Neppure conservare InWiMe è una conclusione irrevocabile: se un componente esterno offrirà decoder CPU completo, originali BF16/F16, memoria realmente limitata e parità migliore a costo inferiore, la sostituzione sarà da preferire. **Questo commit Strata non fornisce quella combinazione.** Un harness comune per confronti offline sarebbe utile; due runtime mantenuti come soluzione permanente non sono un requisito.

### 4.3 Licenze e provenienza

Strata espone [licenza MIT][S31]. Un eventuale riuso di codice sostanziale deve conservarne attribuzione e testo applicabile; l'estrazione non elimina la provenienza. Il [manifest vendorizzato ggml][S28] identifica il pin e le modifiche; alcuni kernel, come il router native, indicano derivazione da llama.cpp. Modelli, dati derivati, font e altre componenti hanno condizioni proprie: la licenza del motore non copre automaticamente pesi e artefatti. In questo audit non viene importato codice né viene scelto un checkpoint da distribuire.

La repository locale non contiene un file tracciato `LICENSE`, `COPYING` o `NOTICE`: prima di distribuire codice derivato va chiarito il regime del progetto e conservati i notice pertinenti. Non occorre introdurre ora una nuova licenza per realizzare questo documento. Per la proposta safetensors si usa la specifica del formato; nessun vendoring di Strata è necessario.

## 5. Verdetto sui vincoli CPU-only, 16 GiB e pesi originali

| Domanda | Verdetto | Evidenza / limite |
|---|---|---|
| Strata contiene kernel x86 utilizzabili senza GPU? | **Sì, S** | Build CPU, pool, dot e test host |
| Questo basta a eseguire il decoder completo senza GPU? | **No nel percorso completo esaminato, S** | Target engine condizionato al backend; arena, attenzione e verifier device |
| Esistono prove Strata con 16 GB RAM? | **Sì, M-S, con forti condizioni** | Caso V100 32 GB, Coder IQ1_M, quasi tutto in VRAM |
| Quelle prove dimostrano 16 GiB, sola CPU, originali BF16? | **No** | GPU grande e pesi quantizzati; assenza di letture modello durante quei decode |
| Il reader riconosce BF16/F16? | **Sì, S** | Tipi contenitore e alcuni tensori/loader |
| C'è una prova full-model originale BF16/F16, CPU-only, 16 GiB? | **Non individuata; requisito non soddisfatto dall'architettura completa esaminata** | Non trasformare supporto di un tipo in supporto del grafo |
| Si potrebbe portare Strata su CPU a qualunque velocità? | **Possibilità teorica, P** | Richiede operatori, dense streaming, stato, budget e verifica nuovi; non è un flag |
| InWiMe soddisfa già tutto? | **No** | CPU sintetico BF16 oltre WSL sì; oltre host, modello pubblico, F16 e contesti lunghi ancora no |

Il caso da 16 GB è particolarmente istruttivo: report community del 4 ottobre, engine 0.1.39/commit abbreviato `6f32ec0`, i7-13700KF, V100-PCIE-32GB, DDR5 16 GB, SATA SSD. Modello **Coder IQ1_M**, KV INT8, MTP `spec 4`; oltre 11.000 esperti in cache GPU. I JSON riportano, per prompt da 4096 token, **1232,73 token/s prefill**, **69,27 token/s decode**, **3,354 s TTFT**; tre campioni. Sono tre metriche diverse. Il rapporto riporta circa 8–9 GB RAM di sistema, 31,5–32,2 GB memoria V100 e 0 MB letti dai file modello durante le richieste. Non è un picco RSS processuale riprodotto qui, né streaming NVMe del modello BF16. [M-S: report][S32], [JSON][S33]

I 16/32 GiB possono essere teoricamente sufficienti a un'architettura CPU a tile se il **working set minimo** di attivazioni, output, KV, metadati e staging entra nel budget e il disco contiene il checkpoint. La dimensione totale dei pesi non impone di per sé la RAM residente, ma né la lentezza accettata né lo swap rendono automaticamente eseguibile un grafo privo degli operatori necessari. La prova oltre WSL di InWiMe dimostra questa possibilità per il suo grafo specifico; non una promessa per ogni LLM.

## 6. Una sola proposta per l'«incremento 5» richiesto

### 6.1 Numerazione e obiettivo

Il numero 5 è già associato a codice e risultati in HEAD. Per evitare equivoci, qui «incremento 5 proposto» significa **la revisione successiva a `f304ee9…` richiesta da questo audit**, non la ripetizione della prova già fatta. L'eventuale numerazione di versione potrà essere decisa all'avvio; nessuna dipendenza è considerata intoccabile per essere stata committata.

**Unico obiettivo verificabile:** eseguire lo stesso decoder piccolo dense/MoE e GEMV dalle stesse parole BF16 memorizzate in **safetensors multi-shard**, senza ricostruire file di pesi nel formato interno, ottenendo gli stessi byte letti, logits/hidden/KV e comportamento in errore dei formati v1/v2.

Questo è un incremento di compatibilità della sorgente, non l'aggiunta contemporanea di un nuovo modello, kernel, cache, direct I/O o protocollo di serving. La fonte d'ispirazione Strata è la lettura in posto di shard originali con indice di metadati; il codice GGUF non è la specifica safetensors. Il contenitore ha header JSON preceduto dalla lunghezza e offset relativi al payload: l'adapter deve tradurli in offset fisici, rispettare forma/tipo e preservare i byte. [Specifica ufficiale safetensors][F1]

### 6.2 Moduli e contratto progettato

| Modulo futuro | Modifica delimitata |
|---|---|
| Nuovo `inwime_runtime/safetensors_source.py` | Lettura metadata bounded; indice shard→tensor→offset; oggetto sorgente normalizzato con `region/spans`, identità e contabilità |
| `inwime_runtime/core.py` | Piccolo punto di selezione sorgente; separare validazione dello schema matematico dalla decodifica del contenitore; full integrity coerente con file contenenti header |
| `inwime_runtime/decoder.py` | Costruzione del catalogo tramite selettore esplicito; nessun nuovo operatore nel forward |
| `inwime_runtime/gemv.py` | Stesso selettore per GEMV e pianificazione; nessuna nuova riduzione |
| Nuovi test sorgente e fixture | Generatore test bounded per payload identico, oracle e guasti; lettore di riferimento separato |
| Piccolo driver di confronto | Usare il verifier esistente con sorgente esplicita; riportare header/hash/payload separati |

I nomi dei nuovi file sono proposti, non creati in questa fase. La selezione è esplicita e rifiuta combinazioni ambigue; il percorso dei manifest esistenti resta selezionabile e verificato contro la baseline. Non si cambia un default prestazionale per nascondere differenze.

Contratti necessari:

1. **Sorgente ≠ modello.** Il descriptor InWiMe conserva `Spec`, variante e binding dei nomi del decoder sintetico. Safetensors descrive tensori, non tokenizer, attenzione o architettura. Nessun «auto supporto Qwen» dedotto dai nomi.
2. **Byte originali e multi-shard.** Nessuna copia persistente, transpose, narrowing o repacking dei pesi di input. Un tensore safetensors ordinario è contiguo in un file; un checkpoint multi-shard distribuisce tensori. Non inventare split interni al tensore nel formato standard. Le segmentazioni arbitrarie v2 continuano a funzionare nel proprio adapter.
3. **Indice e offset.** Convalidare lunghezze, interi, dimensioni e overflow prima delle letture; coppie JSON duplicate rifiutate, index `weight_map` coerente con tutti i file/tensori, nomi file entro la sorgente, nessun intervallo sovrapposto o fuori file. Copertura del payload e header devono essere distinti: non disabilitare globalmente il controllo «nessun byte non indicizzato» per accomodare l'header.
4. **Precisione esplicita.** Prima versione BF16 soltanto. Rifiutare F16/F32/FP8/quantizzati con errore chiaro; non convertirli. Questo preserva i formati attualmente supportati, senza fingere di aver completato l'obiettivo F16. Non-finiti gestiti con lo stesso contratto runtime esistente.
5. **Integrità.** Safetensors non offre da solo SHA256 dei pesi. Il descriptor di prova deve contenere hash attesi esterni dei file, inclusi header, e identità di indice/binding. `full` confronta con tali hash; calcolare un hash e fidarsi dello stesso valore appena calcolato non verifica integrità contro una provenienza attesa. `metadata` resta rinuncia esplicita, mai fallback automatico. Mutazioni del file aperto dopo ammissione devono fallire secondo il controllo di identità esistente. Per una sostituzione atomica del pathname, il contratto proposto conserva fino alla chiusura il descrittore del file originale già verificato, senza riaprire per nome o mescolare versioni. `signature(fd)` da sola non rileva necessariamente che il pathname ora indica un altro inode.
6. **Memoria e ownership.** Il parser non legge l'intero payload; normalizza soltanto i metadati e produce span lazy. Stessi lease/slot, stesso thread proprietario, completamenti fuori ordine non cambiano ordine numerico. Metadata bootstrap/retained contati; esaurimento del budget rifiutato prima del forward.
7. **KV e fallimenti.** Errore a qualsiasi shard, short read o cancel lascia identici banco attivo, lunghezza, versione e byte pubblicati; le letture fisiche in corso sono drenate prima del rilascio. Nessun nuovo retry che pubblichi una transazione parziale.

### 6.3 Limiti espliciti della prova autorizzabile

Sono **limiti proposti**, non nuove allocazioni effettuate:

| Risorsa | Limite |
|---|---|
| Nuovo payload fixture complessivo, tutte le copie/formati inclusi | **≤64 MiB**; preferire riuso fixture esistenti |
| Report, log e metadata su disco | **≤16 MiB**; nuovo spazio totale ≤80 MiB |
| Download checkpoint / copia dei dataset da 6–12 GiB | **0** |
| Header/indice singolo accettato nel profilo di prova | ≤8 MiB; somma input metadata ≤16 MiB |
| Dimensione dell'indice di prova | ≤64 shard, ≤4096 tensori; errori espliciti oltre soglia, non promessa per checkpoint arbitrari |
| Memoria controllata runtime piccolo, metadata inclusi | **≤128 MiB**; dichiarare anche bootstrap e costo dell'oracle |
| RSS del singolo processo test/benchmark | Obiettivo **≤512 MiB**, stop di sicurezza **1 GiB**; un processo alla volta |
| GPU | Non inizializzata; zero allocazioni del runtime |
| Esecuzione | Nessun cambio WSL/cgroup/swap/mount; nessun benchmark grande |

Un limite RSS osservato non è un hard limit imposto dall'allocatore: il supervisore deve campionarlo e interrompere in caso di superamento, mentre il budget delle arene impedisce le allocazioni controllate eccedenti. `RLIMIT_AS`, se usato, è riportato separatamente. Metadata con espansione heap eccessiva devono fallire in modo controllato. Lasciare margine per OS e Windows; il fatto che il filesystem virtuale mostri molto spazio non autorizza nuove scritture grandi.

### 6.4 Test di regressione e correttezza

Preservare i sei moduli esistenti e le **181 regressioni della baseline**, senza cambiare tolleranze/oracle per fare passare l'adapter. Il conteggio nuovo deve essere registrato, non prefissato arbitrariamente.

- Per dense e MoE: stessi payload BF16 in v1/v2/safetensors; prefill più almeno tre decode; confronto completo a ogni fase di logits, hidden e KV con il riferimento indipendente, oltre all'identità fra sorgenti a backend matematico uguale.
- GEMV su dimensioni irregolari, tile che attraversano righe e tagliano riduzioni, tensori in shard differenti, ordine dei nomi diverso dall'ordine fisico. Confrontare tutti gli output piccoli con `math.fsum`; non usare il parser nuovo anche nell'oracle degli offset.
- Header troncato/cresciuto, lunghezza fuori budget, duplicate key, dimensioni/tipo invalidi, overflow, gap/overlap, dati finali non indicizzati, shard assente, index contraddittorio, nomi con traversal, hash errato e mutazione del file aperto. La sostituzione atomica del pathname deve conservare lo snapshot degli FD già verificati oppure produrre un errore conservativo, mai acquisire silenziosamente nuovi byte. Non assumere allineamento assoluto degli offset senza che la specifica lo richieda.
- Iniezione di short read, errore e cancellazione dopo un primo shard riuscito, sia sync sia pipeline; confrontare l'intero stato pubblicato precedente, verificare drain, chiusura FD e lease, sessione fallita secondo il contratto esistente.
- Limite metadata, pianificazione fuori budget e pulizia dopo ammissione fallita; alias input/output e lifetime del wrapper nativo restano coperti. File input identici prima/dopo tramite hash e nessuna scrittura di pesi fuori dalle fixture nuove.

### 6.5 Benchmark baseline/nuova versione

Non confondere regressione del codice con differenza del formato. Servono tre bracci piccoli:

| Braccio | Codice | Sorgente | Scopo |
|---|---|---|---|
| A | `f304ee9…` | v2 esistente | Baseline verificabile |
| B | Candidato | stessa v2, stessi file | Costo/regressione introdotto nel percorso storico |
| C | Candidato | safetensors con identiche parole BF16 | Costo del nuovo adapter/contenitore |

Stesso backend numerico C, tile/read, numero di slot/worker, prompt e decode, filesystem ext4 interno e limiti. Sync e pipeline in esperimenti separati; non cambiare backend fra A/B/C. Un warm-up dichiarato e **cinque ripetizioni misurate per braccio**, ordine controbilanciato; processo nuovo per campione. Nessun altro benchmark in parallelo.

Misurare separatamente startup/index/SHA, prefill, ciascun decode, totale, byte logici utili per tensore, header/hash, `read_bytes`, read calls, RSS/HWM, arene e metadata. Controllare integrità completa fuori dal timer d'inferenza, riportandone tempo e I/O. Per il corpus piccolo il caso principale sarà cache-warm; un braccio dopo DONTNEED può essere chiamato soltanto **cold-ish** e deve riportare i contatori di blocco. Nessun `drop_caches` globale e nessuna pretesa di cache hardware vuote.

Registrare AC, stato termico/potenza/frequenze se leggibili senza cambiare configurazione; indicare `non disponibile` quando manca una misura. Usare mediana, range e campioni individuali; non derivare speedup out-of-core da fixture di pochi MiB. Nessun numero prestazionale Strata entra come baseline di questo confronto.

### 6.6 Accettazione e rollback

**Accettazione funzionale:** payload letto esatto, nessuna conversione, risultati completi piccoli equivalenti, tutte le regressioni verdi, errori fail-closed, KV precedente intatto, sorgenti immutate, nessun payload intero residente, limiti rispettati. L'identità del checkpoint comprende formato/binding e digest; l'uguaglianza matematica fra due contenitori non richiede che il loro digest di file coincida.

**Accettazione prestazionale:** nessuna promessa di speedup. B deve mantenere byte logici e piano numerico di A; metadata aggiuntivi devono essere identificati. Una regressione di mediana B/A >10% richiede spiegazione e, se attribuibile al cambio, correzione prima dell'adozione. Il criterio non promuove rumore in misura: se range e condizioni impediscono attribuzione, il risultato è inconcludente. Per C un overhead diverso di apertura/hash è ammesso e separato; non è ammessa duplicazione del payload o crescita della memoria proporzionale alla dimensione totale dei pesi. Limite proposto sull'incremento di metadata/RSS per queste fixture: ≤32 MiB rispetto a B, dentro i tetti globali.

**Rollback:** adapter opzionale disattivabile; continuare con la sorgente v1/v2 di baseline. Non migrare né cancellare file e non riscrivere vecchi manifest. Qualunque mismatch numerico non spiegato, byte errati, violazione KV/ownership o consumo non limitato blocca l'adozione; il cambio viene corretto o rimosso tramite normale patch/revert revisionabile, senza reset distruttivi. Si mantengono i risultati negativi del confronto.

**Dimostrerebbe:** capacità del runtime di eseguire il grafo già verificato da un contenitore esterno BF16 multi-shard in posto, preservando budget e transazioni. È riuso dell'architettura di lettura con un nuovo confine di sorgente, aperto a ulteriori sostituzioni.

**Non dimostrerebbe:** modello pubblico completo, qualità linguistica, F16, compatibilità universale safetensors, inferenza oltre RAM fisica host, beneficio delle cache Strata, NVMe multipli, contesti lunghi, GPU, MTP, batching o integrazione Harness con modello reale. Nessun risultato piccolo annulla questi debiti di prova.

## 7. Questioni aperte e prove successive necessarie

1. **Prova C oltre host:** preflight aggiornato RAM fisica e spazio Windows/ext4; preventivo di un payload realmente usato superiore alla RAM host, non sparse e non gonfiato da tensori inutilizzati. Approvazione separata del consumo disco; misure host/WSL, hash e letture d'inferenza distinti. Non eseguita né autorizzata implicitamente da questo audit.
2. **Primo modello originale:** scegliere architettura e revisione esatte; tokenizer, template, RoPE/GQA o stato ricorrente, router e scaling, norme, weight tying e dtype. Fixture di operatori e logits upstream prima del modello grande. Safetensors risolve il contenitore, non questo contratto.
3. **Formati:** BF16 prima; F16 richiede espansione/overflow e test dedicati. FP8/MXFP4 nativi e formati derivati devono avere provenienza e kernel dichiarati separatamente. Nessuna equivalenza qualitativa presunta.
4. **Storage:** confrontare byte uguali, pattern uguale e percorso uguale; poi eventualmente interno ext4/esterno ext4/DrvFS. L'esistenza storica di un mount non certifica disponibilità odierna. Misurare latenza p50/p95, amplificazione e concorrenza su selezioni MoE, senza usare benchmark sequenziali come spiegazione causale universale.
5. **Memoria:** limite sul processo e su metadata, file descriptor aperti, scratch, page cache e paging Windows; picchi nel caricamento oltre che nel decode. KV lungo e multi-sessione richiedono nuove prove anche se i pesi sono streamabili.
6. **Predizione/cache:** prima tracce del routing reale e baseline senza cache; poi un solo algoritmo con utilità del prefetch, byte inutilizzati, hit/miss, evictions e attesa domanda. Nessuna variazione degli esperti autorevoli per favorire residenza.
7. **CPU:** BF16 diretto e AVX2 da confrontare con FP64 ordinato e reference indipendente, anche su tie e casi numericamente difficili. Nessuna necessità di perpetuare il kernel attuale, ma neppure di ridurre precisione implicitamente.
8. **Speculation/Harness:** separare test del protocollo, qualità del modello e correttezza della distribuzione/commit; verificare EOS, max token, rifiuti, cancellazione e ripresa. I mock di Harness non chiudono questi punti.
9. **Strata:** nessuna build/test riprodotta qui; resta da misurare ogni componente che si decidesse realmente di estrarre. L'assenza di una prova individuata non è una dimostrazione d'impossibilità teorica; per il percorso CPU-only completo esaminato esistono però dipendenze device concrete, non soltanto documentazione mancante.

Il prossimo traguardo P0 resta quello dei checkpoint reali e del forward completo nel budget; la sezione 6 copre il contenitore, non l'intero traguardo. La richiesta aggiuntiva sulla GPU introduce il solo esperimento P1 della sezione 8.6, separato da quel lavoro. Nessuna implementazione è avviata: il documento si ferma alla proposta.

## 8. GPU opzionale a bassa VRAM: verifica e progetto P1

### 8.1 M1a usa già tiling e trasferisce davvero l'output

**V/S:** riesaminati [benchmark.cpp](../src/benchmark.cpp), [Makefile](../Makefile), [activity_trace.hpp](../src/activity_trace.hpp), [profilatore V2](../tools/profile_v2.py), [analizzatore delle tracce](../tools/analyze_trace.py), [rapporto V2](../BENCHMARK_V2.md) e artefatti M1a. Il sorgente a HEAD è confrontato con le evidenze storiche; non è stato ricompilato né eseguito. M1a dichiara binario e strumenti V2 invariati. [started.json](../results/m1a-native-ext4-20261107T075250Z/started.json) registra SHA256 del binario `3ce352807037731e33bf9ffa03fa02f0cf28f5f0eb2313ecf5301de7b79b2d86`; gli artefatti hanno un [manifest hash](../results/m1a-native-ext4-20261107T075250Z/artifact-sha256.json). `git log -- src/benchmark.cpp` identifica come ultima modifica `80f423134d3dac9aeec752e854383544987cf820`; SHA256 del sorgente letto ora: `778cef7afa19a8c6d17c4c2b7c74c17638c7c05b408290175c0cdc2ea44e3cc4`. Non è una nuova verifica riproducibile della compilazione binaria.

| Aspetto | Implementazione effettiva in `src/benchmark.cpp` | Conseguenza |
|---|---|---|
| Tiling | `gpu`, righe 290–352: 80 blocchi da 64 MiB nel caso M1a, `m=8192`, `k=4096`, `n=1` | I 5 GiB non sono residenti in VRAM. Tile lungo righe/output; ogni riga conserva tutta la riduzione `k` |
| Buffer e concorrenza | `Slot::init`, righe 235–256: due slot, ciascuno con buffer pesi e output host pinned e device, uno stream non bloccante e un handle cuBLAS | Quattro buffer host e quattro device complessivi, oltre al vettore B. Memoria limitata già presente; niente Unified Memory |
| I/O | `read_block` esegue `pread`; il ciclo host legge serialmente dopo `finish` dello slot da riusare | La lettura host successiva può sovrapporsi al lavoro dell'altro slot; non sono due lettori NVMe concorrenti |
| Kernel chiamato | `submit`, righe 267–278: `cublasGemmEx`, A e B `CUDA_R_16BF`, `CUBLAS_COMPUTE_32F`, output `CUDA_R_32F` | Operandi BF16, accumulo/output FP32. `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` non impone FP64 né ordine scalare |
| Kernel fisico | Scelto da cuBLAS; le attività V2 contano kernel, senza registrarne il nome | Non si può certificare un particolare kernel Tensor Core dal solo nome GEMM. A seconda della geometria le tracce mostrano uno o due kernel per chiamata |
| Ritorno CPU | `submit` copia **tutto** l'output FP32, non soltanto i campioni di verifica; `finish` aspetta l'evento dopo D2H | La latenza delle copie output e il drain finale sono nel wall |
| Protezione buffer | `finish`, righe 280–287, attende prima di riusare host/device dello slot | Il modello di lifetime è riutilizzabile; non prova l'atomicità di una transazione KV |
| Cache | `WeightCache` e setup in `gpu`: prefisso statico persistente; M1a imposta cache zero | Cache GPU già sperimentata nel V2, ma non caching dinamico di esperti MoE |
| CPU di confronto | `cpu_gemm`, righe 207–214: OpenMP/SIMD, accumulo FP32 | La CPU M1a **non è** il backend FP64 ordinato dell'attuale runtime |

Il programma storico richiede compute capability ≥8 nel percorso GPU e collega CUDA/cuBLAS anche nel suo eseguibile con modalità CPU. Questo non va trasferito al pacchetto runtime: la libreria CPU attuale ha un target indipendente senza CUDA. «GPU piccola» riguarda capacità disponibile e supporto del backend, non garantisce che ogni vecchia GPU possa eseguire il kernel BF16 cuBLAS storico.

**Confine del timer.** `gpu::total` parte dopo allocazioni, creazione stream/handle, preparazione e H2D iniziale di B, due warm-up e riempimento dell'eventuale cache. Comprende letture dei pesi, submit host, H2D pesi, GEMM, D2H completo, attese eventi, raccolta campioni e drain. Termina prima della validazione FP64 e della distruzione/free delle risorse. Pertanto **comprende il ciclo di streaming misurato, non tutto il costo di una chiamata di inferenza con attivazioni nuove**. In un decoder, l'attivazione cambia fra proiezioni/token: la sua copia non può essere esclusa sistematicamente come B costante iniziale.

`read_ms` misura il wall host delle letture. `h2d_ms`, `compute_ms`, `d2h_ms` sono bracket fra eventi CUDA sullo stream, influenzabili da scheduling e contesa. L'attesa host in `cudaEventSynchronize` è dentro `total_ms` ma **non ha un timer separato**. Non si ricava sottraendo le somme dei bracket dal wall, perché gli intervalli possono sovrapporsi. Anche la somma delle mediane non ricostruisce una passata.

**Divario numerico verificato:** il runtime riceve attivazioni FP64 e usa `inwime_linear` con prodotti/somme FP64 in ordine crescente delle colonne, senza contraction (`-ffp-contract=off`) né fast-math. [Kernel](../src/cpu_kernel.c), [contratto matematico](../inwime_runtime/math_cpu.py), [GEMV](../inwime_runtime/gemv.py). Convertire quelle attivazioni in BF16 per riusare `submit` cambierebbe i risultati, pur lasciando intatti i file dei pesi. L'errore zero M1a sui campioni di numeri sintetici multipli di 1/16 non dimostra equivalenza per attivazioni FP64 generali. Manca inoltre nel benchmark storico il mantenimento delle riduzioni parziali attraverso tile, il catalogo multi-shard e l'intero forward/KV.

### 8.2 Collo di bottiglia: cosa dicono davvero le misure

**M-I:** mediane ricalcolate dai CSV originali, dieci campioni per riga; unità millisecondi. `Compute` GPU è un bracket di eventi, CPU è il timer host del kernel.

| M1a, 5 GiB, n=1, cache=0 | Read host | H2D eventi | Compute | D2H eventi | Wall |
|---|---:|---:|---:|---:|---:|
| Esterno ext4, async | 11.955,223 | 2.727,066 | 553,537 | 2,304 | 12.014,919 |
| Esterno ext4, CPU | 12.088,418 | 0 | 176,401 | 0 | 12.266,181 |
| Esterno ext4, solo I/O | 12.048,954 | 0 | 0 | 0 | 12.048,983 |
| Interno ext4/VHDX, async | 1.606,878 | 430,199 | 44,949 | 0,616 | 1.641,756 |
| Interno ext4/VHDX, CPU | 1.713,967 | 0 | 155,835 | 0 | 1.871,958 |
| Interno ext4/VHDX, solo I/O | 1.801,815 | 0 | 0 | 0 | 1.801,838 |

Fonti: [CSV esterno async](../results/m1a-native-ext4-20261107T075250Z/external-ext4/sweep/n1-b64-c0-async.csv), [CPU](../results/m1a-native-ext4-20261107T075250Z/external-ext4/sweep/n1-b64-c0-cpu.csv), [I/O](../results/m1a-native-ext4-20261107T075250Z/external-ext4/sweep/n1-b64-c0-io.csv); [CSV interno async](../results/m1a-native-ext4-20261107T075250Z/internal-ext4/sweep/n1-b64-c0-async.csv), [CPU](../results/m1a-native-ext4-20261107T075250Z/internal-ext4/sweep/n1-b64-c0-cpu.csv), [I/O](../results/m1a-native-ext4-20261107T075250Z/internal-ext4/sweep/n1-b64-c0-io.csv).

**Inferenza dai dati:** il servizio letture domina M1a su entrambi i percorsi, soprattutto sull'esterno. La GPU async riduce il wall rispetto alla CPU di circa **2,05% esterno** e **12,30% interno**: rispettivamente rapporti CPU/GPU 1,021× e 1,140×. Non sono speedup di un LLM né confronti a matematica uguale con il runtime attuale. Il solo-I/O interno più lento dell'async non è un paradosso da correggere sottraendo tempi: sono processi/campioni e condizioni diverse. I lunghi bracket GPU esterni non provano che USB rallenti fisicamente il kernel: mancano attività GPU della stessa acquisizione per separare scheduling, clock e lavoro effettivo.

**Profilo diagnostico V2, acquisizione distinta:** [trace-summary.json](../results/v2/trace-summary.json), voce `n1-b64-async`, registra 80 letture/H2D/kernel/D2H: read host 1423,074 ms, attività H2D **415,718 ms**, kernel **28,901 ms**, D2H **0,360 ms**. L'intersezione H2D–kernel è **0 ms**; nessuna delle 30 tracce V2 esaminate dimostra quell'overlap. I clock host/device non risultano correlabili (`host_gpu_correlated=false`), quindi le intersezioni read–GPU sono `null`, non zero. La riduzione del wall async è compatibile con overlap lettura host/lavoro GPU, ma non ne misura la durata. Nessuna nuova traccia CUPTI è stata raccolta in M1a. La banda derivabile da quel solo intervallo H2D V2 è circa 12,03 GiB/s; non è una misura della banda nominale PCIe, né un valore garantito sotto paging o su altra GPU.

Il decoder corrente ha un'altra contabilità: nel grande **sync**, 560,040 s wall comprendono 512,485 s di servizio letture, 23,347 s di espansione BF16 e 17,876 s di GEMV. In **pipeline**, 423,019 s wall, 378,522 s letture, 26,253 s espansione e 18,675 s GEMV, con overlap osservato 2,319 s. Fonti: JSON della sezione 1.3. GPU/H2D/D2H sono assenti in queste prove. Non attribuire alla sola CPU la lentezza del decoder, né confrontare direttamente questi accessi buffered da massimo 512 KiB con le letture M1a O_DIRECT da 64 MiB.

**Limite descrittivo di Amdahl, P:** azzerare soltanto i 17,876 s del GEMV sync, mantenendo tutto il resto invariato e senza costo GPU aggiuntivo, porterebbe a circa **1,033×**. Azzerare anche l'espansione porterebbe a circa **1,079×**. Sono controfattuali di quella singola acquisizione, non limiti universali: cambiano con cache, storage e modello. Mostrano perché il primo obiettivo P0 non deve dipendere dalla GPU. La sincronizzazione GPU M1a resta **non isolata**; il tempo lettura include lo stack storage/VM e non separa servizio NAND, USB, filesystem e paging host.

### 8.3 Meccanismi Strata concretamente riutilizzabili

Tutte le voci seguenti sono **S/P**, non misure di porting su RTX 3050. La tabella estende il confronto CPU della sezione 3.

| Meccanismo e sorgenti | Riuso sensato in InWiMe | Beneficio e limite | Rischio / decisione |
|---|---|---|---|
| `ExpertCache::slot_of/admit/fill_slot/verify_slot`, `replace`; [header][S8], [implementazione][S35] | Cache di byte originali con chiave checkpoint/tensore/intervallo; verifica byte, stati filling/ready/in-use | Hit VRAM evita rilettura e H2D pesi se il calcolo consuma la copia device. `admit` non decide una sostituzione quando piena | Medio/alto. Adattare gestione slot e test; non incorporare arena GPU/pack completo. `replace` non sincronizza da solo |
| Quota per layer e lambda `adapt` nel [driver][S5], righe 20704–20804 | Politica di frequenza decaduta, misurata sulla selezione esatta del router | Evita che il primo tratto del modello occupi tutta la cache; nessun hit rate BF16 garantito | Medio. Importare l'idea dopo una traccia reale; quota fissa può sprecare spazio su layer freddi |
| `PcieModel::sample/solve/pick`, [expert_source.hpp][S34], righe 407–494; `expert_pool_dispatch_multi`, [cpp][S6], righe 4885–5060 | Modello di costo per scegliere CPU o GPU per esperti già selezionati | Minimizza `max(g0+g*n_gpu, a+c*n_cpu+d*n_gpu)`, includendo contesa DRAM delle copie sul pool CPU; apprendimento con forgetting e sporadica esplorazione | Alto. Coefficienti non trasferibili da altre GPU o formati. InWiMe deve aggiungere costo miss NVMe, tile, budget e numerica; non copiare dispatch quantizzato |
| `make_cache_complement_plan`, `pin_cache_complement`, [S6] | Evitare copie RAM residenti ridondanti degli stessi byte già in VRAM | RAM liberabile pari ai duplicati effettivi; utile quando 16 GiB sono il vincolo | Alto. Fonte immutabile su disco sempre disponibile per CPU; deduplicazione non significa che il fallback sia gratis |
| `RouterLookahead::submit/run`, `ForesightSwap::predict/take`, [S6] | Hints di prefetch e upload, con domanda prioritaria e scadenza | Nasconde miss soltanto se previsione utile e conclusa prima della domanda | Alto. Hints non possono alterare esperti, coefficienti o ordine di somma; bytes sprecati vanno conteggiati |
| `FileExpertSource::claim_stage/fill_many`, `fill_slot_queued_parts/sync_queued` | Completamenti espliciti, copie da span multi-shard, fence prima del riuso | Possibile overlap I/O/copie/calcolo senza duplicare tutto il checkpoint | Alto se portato intero: `claim_stage` può far crescere lo staging, anche pinned; serve un tetto rigido e backpressure propri |
| `auto_cache_slots`, [auto_cache.hpp][S36] | Dimensionare la cache dopo workspace, attivazioni e altre riserve | Impedisce di assegnare tutta la VRAM ai pesi | Adattare la contabilità, non il margine fisso di 256 MiB o il minimo slot. InWiMe deve poter scegliere cache=0 e CPU |

La parte utile di `PcieModel` è riconoscere che spostare un esperto su GPU può **rallentare la CPU** leggendo la stessa RAM. La formula non rende lossless il kernel scelto e non modella automaticamente un miss da SSD. Nel dispatch Strata la quota GPU è pubblicata prima del lavoro CPU; è una strategia concreta di sovrapposizione, ma dipende dal suo grafo device e da sorgenti pinned. Non basta estrarre `pick` per ottenere un backend opportunistico completo.

Un esperto BF16 può essere più grande di tutta la cache disponibile: la cache a blob intero Strata non risolve quel caso. Per InWiMe la capacità di **calcolare a tile senza ammettere l'esperto in cache** è obbligatoria; cache per tile o proiezione è un adattamento possibile, con chiavi/offset/versioni rigorose e senza omettere parti. Nessun guadagno va dedotto dal numero di esperti quantizzati che entra in VRAM in un report Strata.

### 8.4 Architettura opportunistica e vincoli di correttezza

**P — percorso proposto:** checkpoint immutabile → sorgente/lease limitati → scelta backend per operazione → risultato CPU valido → normale forward e transazione KV. La GPU è un esecutore facoltativo dello stesso contratto, non un secondo runtime, e non possiede l'unica copia autorevole del KV. La prima integrazione può lasciare attenzione, routing, attivazioni non lineari e KV sulla CPU; i loro trasferimenti eventuali devono essere giustificati da misure successive.

Il backend GPU va caricato solo quando richiesto/disponibile: import e inferenza CPU devono riuscire senza driver, librerie CUDA o device. Disponibilità, capacità numerica e convenienza sono controlli diversi. Mancanza di VRAM o operazione non supportata deve poter scegliere CPU **prima** di iniziare lavoro che modifica lo stato; un errore GPU in volo deve invece drenare o invalidare il contesto e lasciare il KV precedente intatto. Non basta catturare un'eccezione e ripetere sulla CPU una riduzione già parzialmente pubblicata. Gli errori di checkpoint/integrità non diventano fallback silenziosi.

**Budget riducibile.** Siano `V_free` la memoria libera rilevata e `V_cap` il tetto applicativo: `V_use <= min(V_cap, V_free - riserva)`; in `V_use` entrano tile, attivazioni, risultati parziali, workspace esplicito e cache. Contesto/driver e altri processi richiedono contabilità separata e margine; i 4 GiB nominali non sono tutti disponibili. Una GPU più piccola deve ridurre tile/cache fino a zero cache o scegliere CPU, non quantizzare per farle entrare. Se neppure il working set minimo entra, nessuna promessa di esecuzione GPU: il runtime continua sulla CPU. Questo è valido anche con cache MoE da zero byte.

Per una futura lane che consumi BF16 direttamente, con `S` byte per tile, due tile richiedono circa `2*S` device per i pesi, oltre a vettori/risultati; il pinned RAM deve avere un tetto distinto e non duplicare inutilmente gli slot. L'adapter conservativo dell'esperimento 8.6 usa invece l'interfaccia FP64 già esistente: a parità di coefficienti, il trasferimento e lo staging device pesi sono **4*S** per tile. Questa penalità è dichiarata prima dell'esperimento. Non si nasconde espansione CPU o narrowing nell'interfaccia GPU.

**Ottimizzazioni Strata da distinguere esplicitamente:**

| Tecnica | Cambia potenzialmente i risultati? | Vincolo InWiMe |
|---|---|---|
| Copia/cache byte originali, prefetch solo come hint, eviction dopo l'ultimo lease | Non per la sola residenza, se bytes e ordine matematico restano uguali | Ammissibili con test byte, errori e lifetime |
| `native_route_resident`, [S11] | Sì: può sostituire esperti selezionati con residenti | Esclusa |
| `STRATA_ROUTE_TAIL_SKIP`, `expert_pool_dispatch_multi`, [S6] righe 4961–4982 e 5034–5037 | Sì: azzera il contributo di miss ai ranghi bassi senza rinormalizzare | Esclusa. **Non è sempre soltanto opt-in:** `generate.cpp` righe 12826–12867 può impostare rango 7 nel serve, a determinate condizioni di architettura/cache/batch; non importare quei default |
| `--adapt-async`, CPU/GPU split e kernel con riduzioni differenti | Sì: il driver avverte che l'adattamento async non è bit-exact fra esecuzioni | Riusare soltanto policy che non modificano il contratto numerico; conservare ordine di somma degli esperti |
| `--spec-split`, cambi di forma batch o kernel | Possibili differenze di rounding; verificatore e argmax possono risentirne | Non equivalenti per definizione alla baseline |
| Quantizzazione delle attivazioni nei dot native, MTP `make_q4_*`, KV INT8/Q4/K8V4, pack F32→F16 | Sì, anche se i file originali dei pesi rimangono intatti | Nessuna introduzione implicita; fuori dall'esperimento |
| BF16/FP32 cuBLAS M1a al posto di FP64 ordinato, TF32/fast-math/FMA o riassociazione non previste | Sì, anche senza quantizzazione dei pesi | Non autorizzate dalla sola scelta «GPU opzionale» |

Il contratto FP64 corrente non è intoccabile: una futura revisione numerica esplicita può essere valutata contro il modello e i suoi oracle. **Questa richiesta non autorizza a cambiarlo** per attribuire un vantaggio al backend GPU. Per una lane conservativa, parallelizzare righe indipendenti è diverso da riorganizzare la somma delle colonne della stessa riga. Errori, cancellazione, riduzioni spezzate e tie del router fanno parte dell'accettazione.

### 8.5 Benefici plausibili: cold, warm, byte e memoria

**M-I:** M1a ha letto 5 GiB per passata misurata e trasferito 5 GiB di pesi H2D senza cache. D2H completo: `80*8192*4 = 2.621.440` byte (2,5 MiB); B iniziale BF16: 8192 byte fuori timer, oltre ai due warm-up. I massimi registrati sono RSS circa **473,36 MiB**, pinned **128,06 MiB**, device esplicito **128,07 MiB**. NVML globale 226 MiB e snapshot CUDA 943,31 MiB sono strumenti/ambiti diversi, non sommabili e non prova che l'eseguibile entri in una GPU fisica da 256 MiB. M1a usa la RTX 3050; non verifica hardware con VRAM inferiore. [Rapporto M1a](../results/m1a-native-ext4-20261107T075250Z/REPORT.md)

Definire «warm» è necessario: cache filesystem/RAM, cache VRAM e contesto/kernel già inizializzati sono tre stati diversi. O_DIRECT non certifica cache Windows/controller fredde; nessuna delle mediane M1a va etichettata cold fisica.

| Scenario, previsione P | Traffico e memoria | Latenza / throughput plausibili |
|---|---|---|
| Primo passaggio, VRAM vuota, dati realmente da storage | Letture utili immutate; se ogni peso va sulla GPU, H2D almeno gli stessi byte nella lane BF16 diretta, più attivazioni/output. Setup e cache fill hanno un costo | Su percorso I/O-bound l'accelerazione del solo kernel può dare poco o peggiorare. M1a esterno è coerente con beneficio modesto, non ne misura il caso cold garantito |
| Pesi warm in RAM/page cache, VRAM vuota | Traffico fisico NVMe può ridursi; richieste logiche e H2D restano. Pinned/staging competono per RAM e banda | Compute/PCIe/submit/sync possono diventare dominanti. Nessun fattore misurato per il runtime FP64; GPU utile solo se costo totale inferiore alla CPU |
| Esperti o tile warm in VRAM, richiesti di nuovo | Ogni byte servito dalla cache evita un byte di lettura logica e di upload pesi rispetto al miss, salvo amplificazione e policy; attivazioni e risultati continuano a muoversi | Opportunità maggiore se il riuso sopravvive alle evictions. Costo del primo riempimento e delle sostituzioni va ammortizzato; nessuna percentuale di hit MoE misurata in InWiMe |
| Scansione dense molto più grande della cache | Una LRU ciclica può avere zero hit. Un prefisso riservato può risparmiare la propria dimensione per passata dopo il fill | V2 ha già provato prefissi 64/256 MiB su 5 GiB, cioè 1,25%/5% di byte: il risparmio di byte non ha prodotto un vantaggio wall universale |
| Più token davvero aggregabili | Riutilizzo pesi, ma più attivazioni/KV e unione di esperti potenzialmente maggiore | I risultati V2 n=4/16 non sono token/s di prefill, decode o speculation. Batching/prefill restano P2 e fuori dal singolo esperimento P1 |

Stima di costo per una lane GPU BF16 diretta senza hit: per le fasi idealmente sovrapponibili il servizio non può essere inferiore a `max(D/B_storage, D/B_PCIe, T_compute)`; si aggiungono dipendenze non sovrapponibili, attivazioni/output, gestione e startup. Se copie e kernel non si sovrappongono, il loro contributo è seriale. Le dipendenze fra layer impediscono di usare quel massimo come previsione esatta dell'intero token. Nel primo adapter FP64 il termine pesi PCIe diventa **`4*D/B_PCIe`**. `D` è byte BF16 effettivamente richiesti, non dimensione nominale del modello.

Esempio **soltanto condizionale**: riservare 512 MiB di VRAM a un sottoinsieme sempre riusato su una scansione di 5 GiB evita 0,5 GiB per passata dopo il primo fill. Usando le velocità end-to-end M1a solo come scala illustrativa, quel volume corrisponde a circa **0,16 s interno / 1,20 s esterno** di servizio equivalente; sul profilo PCIe V2 a circa **42 ms** di H2D BF16. Non sommare questi numeri per prevedere il wall, né presumere quella località in MoE. Un hit fraction utile `h=10%` darebbe al massimo circa 1,11× nel modello ideale puramente I/O-bound, prima di costi cache e trasferimenti. Con `h=0` non risparmia I/O. 512 MiB di cache non sono una prenotazione proposta adesso.

Per il runtime attuale la GPU di solo calcolo non riduce i 64,437 GB logici del workload grande né il KV in RAM. Una cache VRAM può ridurre quei byte soltanto attraverso hit effettivi; mantenere anche una copia host può consumare RAM senza altro vantaggio. Il beneficio più credibile della poca VRAM è dunque **selettivo**: esecuzione di operazioni convenienti e, in futuro, residenza dei pesi con riuso più utile. Non è l'obbligo di tenere un layer intero o tutti gli esperti sulla GPU.

### 8.6 Un solo esperimento incrementale P1: adapter GEMV GPU a numerica invariata

**Proposta, non implementazione.** Collegare un adapter GPU opzionale al contratto `NativeMath.linear` già consumato dalle lease, mantenendo lettore, espansione BF16→FP64, scheduler, tile, ordine matematico e transazione KV della baseline. Il primo adapter torna dalla chiamata soltanto quando H2D, calcolo e D2H sono conclusi. La scelta di un confine sincrono serve a isolare **la sostituzione del backend matematico**; non si introducono contemporaneamente cache MoE, nuovo overlap, BF16 diretto device, batching o scheduler adattivo.

**Perché questo esperimento:** M1a ha già verificato due slot, trasferimenti completi e GEMM BF16/FP32, mentre il divario ancora aperto è la compatibilità con attivazioni e riduzioni del runtime. Questa prova risponde «un backend GPU può rispettare lo stesso contratto, con quali costi complessivi e quando conviene scegliere CPU?». Non ripete lo sweep sintetico da 5 GiB. È lecito concludere che questo adapter non conviene: non si cambia precisione per ottenere uno speedup.

**Perimetro futuro dei file:**

| File/modulo | Cambiamento proposto |
|---|---|
| Nuovo `inwime_runtime/math_cuda.py` | Adapter opt-in con espansione delegata a `NativeMath`, arene bounded, controllo disponibilità, `linear` sincrona e fallback prima del lancio |
| Nuovo `src/gpu_ordered.cu` | GEMV FP64 a righe indipendenti; sequenza crescente delle colonne, prodotti e somme separati; nessuna allocazione per coefficiente |
| `Makefile` | Solo target GPU opzionale; il target CPU continua a compilare senza toolkit/driver CUDA |
| Nuovi test/driver piccoli | Confronto sullo stesso `StreamedGEMV` e su fixture dense/MoE del decoder, selezionando `math_backend`; fault injection e contatori di trasferimento |
| `core.py`, `pipeline.py`, `gemv.py`, `decoder.py` | Nessuna riscrittura prevista: l'interfaccia matematica esiste. Se servissero cambi di ownership o scheduler, ridurre/rivedere il perimetro prima di allargarlo |

I nomi nuovi sono soltanto progettati. Usare buffer device riutilizzabili e una sola lane ordinata; copiare attivazioni FP64 e accumulatori iniziali necessari **dentro il timer**. Attraverso tile e shard mantenere lo stesso accumulatore parziale. Il risultato di una tile resta privato finché copie e controllo finiti sono completati; niente scritture asincrone dopo la fine della lease. Non promettere rollback dell'intero output standalone di GEMV: tile precedenti possono essere già completate. Nel decoder il banco KV pubblicato deve invece restare invariato in caso di errore.

Un kernel candidato può usare `__dmul_rn` e `__dadd_rn`: la documentazione NVIDIA specifica round-to-nearest-even e impedisce la loro fusione in multiply-add. [CUDA Math API 12.8, fonte primaria][G1] È una base progettuale, **non una prova di equivalenza**: vanno verificati subnormali, signed zero, overflow e comportamento effettivo del compilatore/host. Niente Tensor Core, TF32, riduzioni ad albero o conversione FP64→BF16 in questo candidato.

Il costo è deliberatamente trasparente: ogni coefficiente letto su disco occupa 2 byte BF16 ma viene trasferito al device in 8 byte FP64. Anche ricopiare il vettore per più lease ha costo da contare; non dichiararlo residente gratis senza un nuovo contratto di lifetime. Questo adapter valuta un confine minimo corretto, non il massimo potenziale della GPU. Un risultato negativo non esclude un futuro backend che consumi direttamente BF16 mantenendo la stessa aritmetica; quel cambiamento non è parte dell'esperimento proposto.

**Limiti proposti:** nuovo payload fixture totale ≤32 MiB; log/metadata/build sperimentale ≤48 MiB, nuovo disco ≤80 MiB. Nessun peso scaricato o corpus da 5/6/12 GiB copiato. Budget controllato host, incluso staging CPU, pinned e workspace dell'adapter, ≤128 MiB; pinned ≤16 MiB. RSS obiettivo ≤512 MiB e stop a 1 GiB, un processo alla volta. Device esplicito dell'adapter ≤64 MiB, test con cap **8 MiB** e cap zero; tile FP64 ≤2 MiB nel profilo principale, riducibile ulteriormente. Contesto/driver e strumenti globali riportati separatamente, con almeno 256 MiB di margine sulla VRAM libera misurata prima delle allocazioni; se non disponibile si sceglie CPU. Nessuna prenotazione fissa di tutti i 4 GiB. Una matrice BF16 da 16 MiB, compresa nel tetto fixture, supera il cap device da 8 MiB e obbliga a più tile senza un file enorme. Ridurre un cap sulla RTX 3050 non prova l'esecuzione su un'altra GPU fisica.

**Test necessari, con oracle invariati:**

1. Output completo bitwise rispetto a `NativeMath` nelle fixture compatibili con lo stesso ordine, più confronto indipendente già previsto dal runtime. Attivazioni FP64 non rappresentabili in BF16/FP32, cancellazioni numeriche, ampia dinamica, subnormali/signed zero e non-finiti. Verificare i byte degli input prima/dopo; nessun cambio delle tolleranze per fare passare CUDA.
2. Dimensioni irregolari, tile che iniziano/finiscono dentro una riga, riduzioni tagliate, output con offset, più shard v1/v2 e forme dense/MoE già supportate. GPU/CPU devono leggere gli stessi byte nella stessa sorgente; safetensors sarà aggiunto quando l'adapter P0 esisterà, senza trasformarlo in dipendenza di questa prova.
3. Prefill breve e almeno tre decode sulle fixture piccole: tutti i logits, hidden, esperti selezionati e KV confrontati a ogni fase; tutti gli esperti top-k calcolati e combinati nello stesso ordine. Alias e lifetime del wrapper verificati durante chiamate che rilasciano il GIL.
4. Device assente, backend non installato, budget zero, allocazione negata prima del lancio: selezione CPU osservabile e risultato identico. Errore dopo lancio/H2D o cancellazione: drain/invalidazione, errore esplicito, nessun retry su output parziale, banco/length/generation KV precedente identici, nessun buffer riutilizzato in volo. I guasti device possono essere iniettati nel wrapper; non provocare reset reali della GPU.
5. Cap ridotto con matrice più grande del cap, riuso ripetuto dei buffer e cleanup: picchi entro limiti, nessuna crescita fra chiamate, nessuna dipendenza CUDA importata nel percorso CPU puro. Nelle prove GPU positive, contatori backend/kernel devono certificare **tutti i coefficienti elaborati sulla GPU, zero fallback**, H2D pesi pari a quattro volte i byte BF16 e copie di attivazioni/accumulatori/output conformi al piano. Cap zero/device assente sono prove CPU negative separate. Se la GPU non è disponibile, la prova GPU è non eseguita, non superata tramite fallback. Tutte le regressioni CPU della baseline restano necessarie.

**Confronto controllato:** A = `NativeMath` a HEAD; B = stesso codice e sorgenti con adapter GPU. Impostazione letture sync identica nel confronto principale; pipeline coperta come regressione di correttezza separata, senza promuoverla contestualmente a default. Un warm-up dichiarato e cinque coppie A/B controbilanciate, stessi file/tile/input e filesystem ext4 interno. Dataset piccolo: il risultato principale è **cache-warm**; eventuale serie dopo DONTNEED è soltanto cold-ish e richiede contatori di blocco. Nessun nuovo confronto DrvFS/USB, drop-caches globale, modifica a WSL o benchmark pesante. Porre un tetto di 2 GiB alle letture logiche dell'intera prova, warm-up/verifica inclusi; se insufficiente per il corpus scelto, ridurlo prima di avviare.

Registrare apertura/hash, startup CUDA/allocazioni, primo uso e regime separatamente. Il wall per GEMV deve andare dalla prima lettura all'output CPU disponibile; per decoder separare prefill, ciascun decode e totale. Misurare lettura, espansione CPU, submit, H2D **pesi/attivazioni/accumulatori**, eventi kernel e D2H, attesa host con timer dedicato, e byte per ciascuna copia. Il tempo di attesa host non si somma ai servizi device come se fosse lavoro disgiunto. RSS/HWM, pinned, device esplicito, snapshot CUDA/NVML e swap/paging osservabili restano distinti. Registrare driver/build, AC, temperatura/potenza/clock senza modificarli. Un'eventuale traccia diagnostica piccola è separata dai campioni prestazionali; se i clock non si correlano, dichiarare non misurabile l'overlap host/device.

**Accettazione funzionale:** byte e risultati conformi, nessun esperto omesso, regressioni verdi, KV intatto nei fallimenti, arene entro i tetti e CPU eseguibile senza CUDA. Qualunque narrowing, mismatch non spiegato, dipendenza GPU obbligatoria o violazione di lifetime blocca l'adozione. Lo stesso contratto è più importante dello speedup.

**Accettazione per l'uso opportunistico:** promuovere la lane soltanto per le geometrie nelle quali il wall completo mediano migliora di almeno 10% rispetto ad A e la variabilità non rende il confronto inconcludente; mostrare tutti i campioni e il costo del primo uso. Il percorso CPU selezionato deve restare entro 5% dalla sua baseline, salvo rumore documentato. Se CUDA è corretta ma più lenta, si conserva il risultato sperimentale e si lascia CPU selezionata: nessuna accelerazione dichiarata. Nessuna generalizzazione delle soglie a modelli o storage non misurati, né token/s ricavati dalla sola GEMV.

**Rollback e portata della prova:** backend opt-in, rimozione/disattivazione senza migrare checkpoint o KV. Un errore dopo avvio non commuta silenziosamente la sessione in corso; resta valido il prefisso KV già pubblicato secondo il contratto esistente. La prova dimostrerebbe il confine GPU matematicamente compatibile, il calcolo di un tensore oltre il cap device impostato e il suo costo completo sulle fixture. Non dimostrerebbe inferenza oltre RAM fisica, checkpoint pubblico completo, prestazioni su GPU fisiche più piccole, cache MoE efficace, cold NVMe garantito, Tensor Core compatibili o beneficio della GPU per un LLM reale. Questi limiti restano anche se tutti i test passano.

**Raccomandazione aggiornata:** procedere sul P0 dei checkpoint reali; mantenere aperto il P1 GPU con questo solo esperimento di compatibilità, prima di scegliere cache VRAM o scheduling Strata. La RTX 3050 può essere utile senza contenere il modello, come già indica M1a; non c'è però evidenza che acceleri molto il decoder attuale dominato dalle letture. Nessuna implementazione o nuova esecuzione segue automaticamente da questa proposta.

## Fonti primarie e riproducibilità della lettura

I link Strata seguenti sono fissati tutti al commit esaminato. I riferimenti locali sono relativi alla repository a HEAD indicato; le misure originali restano nei JSON/log citati. Sono stati usati comandi read-only (`git status/log/rev-parse/remote`, `rg`, lettura file, `uname`, `free`, `lscpu`, `df`) e API GitHub GET. Nessuna performance nuova è stata prodotta. La specifica safetensors è fissata a `9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4`, commit del 9 novembre 2026.

Test Strata esaminati, non eseguiti:

| Test | Contratto controllato nel sorgente | Limite della prova |
|---|---|---|
| `tests/core/file_expert_source_test.cpp` | Layout, memoria/cgroup, direct reads, prefetch, scambi | Non è un forward completo BF16 CPU |
| `tests/core/expert_cache_per_layer_test.cpp` | Ammissione per layer su slot uniformi/variabili | Richiede CUDA; senza device esce con codice 77 |
| `src/kernels/cpu/router_dot_parity.cpp` | Dot router BF16 AVX2/AVX contro riferimento double, 1–8 token | Soglia relativa aggregata `1e-5`, non identità bitwise del decoder |
| `src/kernels/cpu/native_arithmetic_parity.cpp` | Risultato degli esperti indipendente dalla dimensione del gruppo per gli entry point invariant | Blocchi sintetici compressi, confronto con dispatch singleton; non checkpoint BF16 originale |
| `tests/core/serve_stop_bound_test.cpp` | Prefisso pubblicabile entro EOS e budget output | Helper host, non transazione completa con guasto GPU |

Presenza del test e commenti su esiti precedenti non equivalgono a test passato in questa sessione. Anche i protocolli in `docs/BENCHMARK_PROTOCOL.md` e i documenti prestazionali sono stati trattati come prescrizioni/report, non come esecuzioni effettuate nell'audit.

[S0]: https://github.com/Niko1221/Strata/commit/13eedf552417a6a791a7d13c6597e24300ee3a09
[S1]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/AGENTS.md
[S2]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/data/release.json
[S3]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/CMakeLists.txt
[S4]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/cmake/strata_gpu_engine.cmake
[S5]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/program/generate.cpp
[S6]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/expert_source.cpp
[S7]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/platform/direct_file.cpp
[S8]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/expert_cache.hpp
[S9]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/cmake/backends/cpu.cmake
[S10]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/router_top10.cu
[S11]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/native_router.cu
[S12]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cpu/native_expert.cpp
[S13]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cpu/kq_avx2.cpp
[S14]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/session.cpp
[S15]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/weights.cpp
[S16]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/kernels/kv_stream.hpp
[S17]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/kv_stream.cu
[S18]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/verify.cpp
[S19]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/serve_stop_bound.hpp
[S20]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/mtp.cpp
[S21]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/spec_prob.hpp
[S22]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/prefill/prefill.cpp
[S23]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/KV_PREFETCH.md
[S24]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/artifact/gguf_reader.hpp
[S25]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/tools/iq_pack.py
[S26]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/serve/server.py
[S27]: https://github.com/Niko1221/Strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09/integrations/harness
[S28]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/third_party/strata-ggml/UPSTREAM.md
[S29]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/requirements.txt
[S30]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/dsv41/ds4_gguf_cpu.cpp
[S31]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/LICENSE
[S32]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/bench/results/2026-10-04-community-v100-16gb-ram/README.md
[S33]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/bench/results/2026-10-04-community-v100-16gb-ram/summary.json
[S34]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/expert_source.hpp
[S35]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/expert_cache.cpp
[S36]: https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/auto_cache.hpp
[G1]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
[F1]: https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/README.md#format
