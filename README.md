> **InWiMe oggi — visione e roadmap:** [scopo finale, requisiti irrinunciabili, prove concluse e traguardi futuri](docs/VISION_AND_ROADMAP.md). La documentazione storica dei benchmark qui sotto non rappresenta da sola lo stato attuale: gli incrementi 7–8 hanno già eseguito LLM pubblici addestrati e l'incremento 8 è stato revisionato. L'incremento 9 rimane da dimostrare.

# Benchmark BF16 con streaming esplicito

Benchmark C++17, CUDA Runtime e cuBLAS per `C_block = A_block × B`.
`A` è un file sintetico BF16 letto un blocco alla volta; `B` è BF16 residente;
accumulo e output sono FP32. Non richiede modelli, download o configurazioni di sistema.
La CPU decodifica BF16 in FP32 e usa OpenMP/SIMD: è una baseline implementata nel
progetto, non un confronto con una libreria BLAS CPU ottimizzata.

## Compilazione e verifica

Richiede CUDA Toolkit con header e librerie nel percorso del compilatore,
GCC 13, OpenMP e Linux con `O_DIRECT`. Non contiene kernel personalizzati:
il codice CUDA invoca `cublasGemmEx`, quindi si compila con il compilatore C++.

```bash
make
make test
# Richiede GPU Ampere o successiva realmente accessibile:
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/smoke.py --gpu
```

Il test confronta tutti gli output con un riferimento FP64 indipendente,
inclusi un solo blocco, numero dispari di blocchi, riuso dei due buffer e dimensioni
non multiple delle tile. Verifica anche corruzione, troncamento e parametri errati.
Le esecuzioni normali controllano fino a 32 elementi per blocco; `--full-check`
controlla ogni elemento. Gli input sono valori BF16 esatti, variabili per blocco,
con segni misti; la conversione non introduce arrotondamento né quantizzazione.
La validazione avviene fuori dalla misura e fallisce con exit code 1.

## Esecuzione

```bash
mkdir -p data results
# 640 blocchi da 8 MiB = 5 GiB, superiori alla VRAM della RTX 3050 da 4 GiB.
./build/bf16_stream --prepare --file data/weights-5GiB.bf16 --blocks 640
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/bf16_stream --file data/weights-5GiB.bf16 --blocks 640 --mode all \
  > results/all.csv 2> results/all.log
```

`--prepare` crea esclusivamente un file nuovo e chiama `fsync`. Non sovrascrive
file esistenti. Le dimensioni devono coincidere con quelle usate nella preparazione.
Il default è 64 blocchi (512 MiB), `m=1024`, `k=4096`, `n=16`, 8 thread CPU e
3 ripetizioni. Parametri modificabili: `--m`, `--k`, `--n`, `--blocks`,
`--threads`, `--repeats`, `--file`, `--full-check`.
Il file contiene solo `A`, in ordine `[blocco][riga][k]`, BF16 little endian;
`B` è generata deterministicamente in ordine column-major. I blocchi sono
moltiplicazioni indipendenti, non strati concatenati di una rete.

Il prefisso `LD_LIBRARY_PATH` nei comandi GPU seleziona la libreria driver WSL
per il singolo processo, senza modificare la configurazione del sistema.
È stato verificato dall'utente: il test GPU falliva senza questo prefisso
e passa con esso. Su Linux nativo omettere il prefisso WSL.

Modalità disponibili:

| Modalità | Percorso |
| --- | --- |
| `io` | letture sequenziali dirette, nessun calcolo |
| `cpu` | lettura e GEMM CPU, senza inizializzare CUDA |
| `sync` | lettura → H2D → GEMM → D2H → attesa, per ogni blocco |
| `async` | due slot pinned/device e due stream non bloccanti; attesa solo prima del riuso dello slot e al drenaggio finale |
| `all` | tutte le precedenti; ordine CPU/sync/async ruotato fra ripetizioni |

Le letture host sono esplicite e bloccanti (`pread`): nella modalità asincrona
possono sovrapporsi alle operazioni GPU del blocco precedente. H2D, GEMM e D2H
di uno slot sono ordinati nel suo stream. Un evento finale impedisce il riuso
prematuro di memoria host/device e output. Entrambe le modalità GPU allocano
due buffer identici, registrati con `cudaHostRegister`, e due stream.
Non si usa Unified Memory. Il numero di stream consente la sovrapposizione,
ma il guadagno effettivo dipende da dispositivo, carico e storage.

Con i default la memoria applicativa principale GPU è circa 16,3 MiB più
workspace e contesto cuBLAS; i buffer host pinned sono circa 16,1 MiB.
L'occupazione non cresce con la dimensione del file, salvo i piccoli campioni
di validazione. `--full-check` conserva tutti gli output e va usato su casi piccoli.

## Significato delle misure CSV

| Colonna | Misura |
| --- | --- |
| `read_ms`, `read_GiB_s` | somma delle durate host di `pread`, byte del file / durata |
| `h2d_ms`, `h2d_GiB_s` | somma degli intervalli fra eventi CUDA attorno alle copie di `A` |
| `compute_ms`, `compute_TFLOP_s` | GEMM CPU con clock monotono oppure GPU con eventi; operazioni = `2*m*k*n*blocks` |
| `d2h_ms` | somma degli intervalli CUDA per il ritorno di tutti gli output FP32 |
| `total_ms`, `effective_GiB_s` | tempo host dall'inizio della pipeline al drenaggio finale; byte del file / tempo totale |
| `checked`, `max_abs_error` | numero dei valori confrontati e massimo errore assoluto |

Le durate CUDA sono misurate sotto il carico della pipeline, possono includere
contesa e non sono benchmark isolati dei motori hardware. In asincrono le somme
dei tempi di fase **non vanno sommate** per ricostruire il totale. La misura totale
include letture, enqueue, attese e raccolta dei campioni, ma esclude allocazioni,
inizializzazione CUDA/cuBLAS, generazione di B, suo unico trasferimento iniziale,
warm-up e validazione FP64. Il warm-up esegue un blocco CPU oppure un GEMM per stream.
Per CPU e I/O i campi CUDA sono zero (non applicabili).
Per confrontare le modalità usare la mediana di `total_ms` delle ripetizioni;
speedup = mediana CPU / mediana GPU, oppure mediana sync / mediana async.

`O_DIRECT` evita la page cache Linux per i dati. Il buffer e gli offset sono
allineati a 4096 byte; nessun fallback silenzioso a letture buffered.
**In WSL2 il dispositivo visibile è un disco virtuale**: la misura descrive il
percorso filesystem/VHD/storage sottostante. Non prova la banda fisica del solo
NVMe e non elimina eventuali cache Windows, controller o unità. Posizionare il
file sul filesystem residente sull'NVMe desiderato; `/tmp` può essere tmpfs.
Non si svuotano cache globali e non si modificano driver o impostazioni.

Riferimenti tecnici: [cuBLAS BF16 GEMM](https://docs.nvidia.com/cuda/archive/12.0.1/cublas/index.html),
[memoria pinned e sovrapposizione CUDA](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/).

## Suite V2

La suite riusa lo stesso file da 5 GiB senza scrivere nuovi dataset. `k=4096`
rimane fisso; cambiano `m` e il numero di blocchi per ottenere buffer per slot
da 4, 8, 16, 32 e 64 MiB. Per ciascun buffer confronta CPU, GPU sync e async
con `n=1,4,16`; aggiunge una baseline I/O per buffer. Una seconda fase confronta
cache GPU da 64/256 MiB con la baseline 0, a buffer 8 MiB per tutti i valori di n.
Sono 62 configurazioni, ciascuna con almeno 10 ripetizioni; i tempi di profiling
diagnostico sono separati. Lo sweep completo legge logicamente circa 3 TiB,
oltre ai warm-up, senza creare file grandi aggiuntivi. Può richiedere decine di minuti.

```bash
make
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/cache.py
python3 tools/benchmark_v2.py
python3 tools/summarize_v2.py
```

`tools/benchmark_v2.py` è riprendibile e salva ogni configurazione appena completa.
Opzioni: `--stage buffers|cache|all`, `--file PATH`, `--output DIR`, `--threads 8`,
`--repeats 10` (minimo 10). Non sovrascrive risultati completi dello stesso binario;
se il binario cambia richiede una directory output differente. Un processo separato
per modalità mantiene interpretabile il picco RSS cumulativo di `getrusage`.
L'ordine delle configurazioni è mescolato con seed fisso 20261105.

Telemetria GPU globale ogni 200 ms tramite `nvidia-smi`; memoria e swap del processo
ogni 100 ms tramite `/proc/PID/status`. In caso di swap del processo lo sweep si
arresta senza cambiare la configurazione di swap. I byte pinned registrati e le
allocazioni device esplicite sono conteggiati nel C++; `cudaMemGetInfo` dà inoltre
uno snapshot globale dopo il warm-up. RSS HWM è misurato dal kernel a ogni passata.
I picchi della telemetria campionata possono perdere transitori molto brevi e
comprendono eventuali altri processi GPU. `VmPin`/`VmLck` possono non rappresentare
la memoria CUDA registrata in WSL; il conteggio applicativo è riportato separatamente.

`--cache-mib N` mantiene un prefisso di blocchi immutabili in VRAM fra le passate.
Il riempimento iniziale legge e trasferisce una volta il prefisso ed è riportato
in stderr (`cache_setup_bytes`, `cache_setup_ms`), fuori dal tempo di ogni passata.
Solo i miss passano per disco e H2D; `read_bytes`, `h2d_bytes` e `cache_hits`
sono espliciti nel CSV. Questa cache statica fornisce una baseline di riuso favorevole,
non simula routing MoE o LRU: uno scan ciclico completo con LRU più piccola del
working set avrebbe zero hit dopo il primo giro. I warm-up leggono due ulteriori
blocchi GPU (uno CPU) fuori timing per passata. La cache è limitata a 2048 MiB e
la dimensione effettiva è arrotondata a blocchi interi e limitata al dataset.

Profiling opzionale con strumenti già installati:

```bash
make profile trace
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=results/example-profile ./build/bf16_stream_profile \
  --file data/weights-5GiB.bf16 --blocks 640 --n 1 --mode async --repeats 1 --profile

BENCH_TRACE_FILE=results/example-activities.csv \
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/bf16_stream_trace --file data/weights-5GiB.bf16 \
  --blocks 640 --n 1 --mode async --repeats 1 --profile
python3 tools/analyze_trace.py results/example-activities.csv
# Acquisizioni diagnostiche per tutti i buffer e n, preservando trace esistenti:
python3 tools/profile_v2.py
python3 tools/report_v2.py
```

Il target normale non dipende da NVTX o CUPTI. Quelli opzionali usano le librerie
già presenti, senza installazioni. Le letture sono annotate NVTX; il tracer CUPTI
registra attività kernel/copie e letture host nella stessa base temporale.
La sovrapposizione si calcola per intersezione delle unioni degli intervalli,
senza dedurla dal solo speedup. Un trace senza attività reali GPU non è considerato
una verifica di sovrapposizione. Non eseguire il profiler contemporaneamente allo sweep.

## Confronto con disco USB

`BENCHMARK_V2_STORAGE.md` documenta il confronto mirato scelto dall'utente: I/O
e GPU async n=1, buffer 64 MiB, cache disabilitata, 10 ripetizioni, stesso file
da 5 GiB su ext4 WSL, C:/DrvFS e D:/USB/DrvFS. I dati sono separati in
`results/v2/storage/`; `COMPARISON.md` e i risultati dello sweep V2 restano invariati.

Lo script di setup locale crea due copie esclusive, verifica SHA256 e registra
le directory temporanee in `datasets.json`: massimo 10 GiB aggiuntivi, da rimuovere
alla fine. Non eseguirlo di nuovo se il manifest esiste. I nomi dei volumi del
setup sono specifici del Katana: verificare l'audit prima di riusarlo altrove.
Il confronto è riprendibile con i file temporanei ancora presenti e il medesimo
binario. Le opzioni `--buffers`, `--ns`, `--modes` della suite selezionano sottoinsiemi
senza cambiare il benchmark C++.

```bash
# Su una nuova directory risultati, dopo audit e setup:
# python3 tools/storage_setup.py
python3 tools/storage_compare.py
python3 tools/report_storage.py
python3 tools/report_v2.py
# Solo dopo tutte le 60 passate:
python3 tools/storage_cleanup.py
```

Le query standard WMI forniscono contatori globali di lettura dei dischi Windows.
La diagnostica nativa opzionale in `tools/diagnostics/windows_storage.ps1` è stata
bloccata dall'antivirus e non è usata per i risultati: non sono state modificate
le protezioni o installate nuove dipendenze. Velocità USB negoziata e baseline
Windows nativa senza cache restano non disponibili.

`storage_compare.py` avvia anche la calibrazione I/O iniziale se non esistono suoi
file; se è già in esecuzione la attende. Dopo la pulizia i dataset temporanei non
sono più disponibili: per una nuova acquisizione preservare prima l'intera cartella
`results/v2/storage` con un nome diverso, poi rifare audit e setup. Non sovrascrivere
risultati storici e non riutilizzare un manifest appartenente a file rimossi.

## Catalogo delle tecniche

Il [catalogo delle tecniche e dei riferimenti](docs/INWIME_TECHNIQUE_WATCHLIST.md)
è un inventario in evoluzione. Distingue gli argomenti attribuiti alle discussioni
storiche dalle nuove voci e dagli approfondimenti della ricerca successiva,
esplicitando le provenienze ancora da confermare. Non costituisce una roadmap
né un'autorizzazione a modificare il runtime o avviare nuove prove.

## Runtime CPU: checkpoint segmentati

L'[incremento 4](docs/RUNTIME_CPU_INCREMENT_4.md) aggiunge tensori multi-shard,
compatibilità manifest v1/v2, generazione BF16 scalabile e verifiche riproducibili.
Dimostra accesso a un checkpoint da 12 GiB oltre la RAM WSL e scansione progressiva
di un tensore da 6 GiB; il forward completo oltre RAM resta non dimostrato.
Strumenti: `python3 -m tools.checkpoint4 --help` e
`python3 -m tools.benchmark_increment4 --help`. Il generatore esegue soltanto
l'audit senza `--execute`; payload in `checkpoints/`, fuori da Git.

## Runtime CPU: inferenza completa oltre RAM

L'[incremento 5](docs/RUNTIME_CPU_INCREMENT_5.md) introduce un adapter CPU C opzionale
per espansione BF16 esatta e GEMV FP64 con tile che possono attraversare righe e
spezzare riduzioni. Il percorso Python resta disponibile; il target CPU non richiede CUDA.

```bash
make cpu-kernel
python3 -m pytest -q tests/test_gemv_cpu.py tests/test_checkpoint5.py
# Solo audit e preventivo; nessuna scrittura dei 12 GiB senza --execute:
python3 -m tools.checkpoint5 checkpoints/nuovo-decoder
# Il verificatore richiede un checkpoint già esistente e output nuovo:
python3 -m tools.verify_increment5 checkpoints/inc5-dense-12g \
  --mode sync --output results/nuova-inferenza.json
```

La prova usa un decoder denso sintetico completo e originale BF16, senza tokenizer
o pesi addestrati. Il rapporto distingue pesi realmente elaborati, limite del processo,
RAM WSL, RAM fisica host, verifiche numeriche complete e controlli parziali sul modello grande.

## Runtime CPU: safetensors BF16 in posto

L'[incremento 6](docs/RUNTIME_CPU_INCREMENT_6.md) aggiunge una sorgente safetensors
BF16, singola o multi-shard con indice, selezionata con `source='safetensors'` in
`CPURuntime`, `minimum_plan` e `StreamedGEMV`. Riusa letture limitate, lease, pipeline,
aritmetica FP64 e transazioni KV. I file originali restano immutati; nessuna conversione
o quantizzazione dei pesi. Il descriptor esterno `safetensors.json` contiene hash
attesi, riferimenti e, per il decoder sintetico, binding e configurazione espliciti.

```bash
make cpu-kernel
python3 -m pytest -q tests/test_safetensors_source.py
```

Le [evidenze](results/increment6/README.md) confrontano baseline, candidato v2 e
candidato safetensors, e misurano RSS/I/O su fixture controllate. Questo supporta
il contenitore reale, non ancora il grafo, tokenizer e logits di un modello pubblico.
Nessun nuovo backend GPU, cache VRAM o MTP è incluso.

## Primo modello pubblico: Llama / SmolLM2

L'[incremento 7](docs/RUNTIME_CPU_INCREMENT_7.md) aggiunge un adapter Llama con
RoPE, GQA, tokenizer ufficiale e greedy decoding su SmolLM2-135M BF16 originale.
I pesi vengono letti a tile; PyTorch/Transformers servono soltanto all'oracolo
isolato. **Gate numerico risolto:** parità completa Transformers standard FP32 e
riferimento PyTorch realmente FP64 superate alle tolleranze iniziali. Il
confronto standard FP64 rimane un diagnostico obbligatorio, con il precedente
fallimento conservato: RMSNorm, RoPE e softmax upstream usano internamente FP32.
Il runtime InWiMe non è stato modificato per imitarli. Non è una nuova prova oltre RAM fisica.

```bash
make cpu-kernel
python3 -m pytest -q tests
# Con tokenizers installato nel proprio ambiente e checkpoint già verificato:
.venv-oracle7/bin/python -m tools.generate7 checkpoints/smollm2-135m-93efa2f
```

## Modello reale sotto il limite del processo

L'[incremento 8](docs/RUNTIME_CPU_INCREMENT_8.md) esegue SmolLM2-1.7B originale
BF16 (3,423 GB di pesi unici attivi) con **RLIMIT_AS 128 MiB**, CPU FP64,
prefill di cinque token e tre decode. Parità completa Transformers standard FP32
su logits, residui di tutti i layer e KV; risultati identici fra sync/pipeline,
originale/multi-shard e forward completo/incrementale. **395 test PASS**;
gate numerico del 7 preservato, incluso il FAIL diagnostico FP64 standard.

Picco RSS osservato circa 92 MB, decode mediano circa 2,74 s/token, cache OS calda.
La prova supera il limite del processo, mentre i pesi restano sotto la RAM WSL
e quella fisica host. Nessuna quantizzazione o modifica all'adapter/kernel.
L'oracolo 1.7B è FP32 e risiede in un processo separato; nessuna parità FP64 1.7B
o prestazione NVMe cold dichiarata. [Evidenze e comandi](results/increment8/README.md).

## Oltre la RAM WSL con un LLM reale

L'[incremento 9](docs/RUNTIME_CPU_INCREMENT_9.md) aggiunge un adapter Qwen2
(bias Q/K/V, head non condivisa) ed esegue **Qwen/Qwen2.5-7B originale BF16**:
**14.141.295.616 byte unici di pesi attivi** contro `MemTotal` WSL di
11.483.848.704 byte, prefill di cinque token e tre decode, in un cgroup v2
transitorio con `memory.max` 2 GiB e `memory.swap.max` 0. Parità completa con
oracoli Transformers indipendenti a blocchi: standard FP32 e FP64 puro superati,
diagnostico FP64 standard FAIL grezzo conservato. Decode sync mediano circa
13,3 s/token con letture realmente servite dal disco (circa 14 GB per token).
SmolLM2-1.7B verificato sotto lo stesso limite di 2 GiB dopo evizione mirata della page cache.
Non dimostra il superamento della RAM fisica host né prestazioni pratiche.
La [revisione](docs/REVIEW_INCREMENT_9.md) ha reso il gate fail-closed sulle
mutazioni provate (80 più 13 aggiramenti trovati da una seconda revisione, con
limiti residui dichiarati) e ha sanificato i percorsi nelle evidenze
dell'incremento 9, senza ripetere misure. Le evidenze degli incrementi 4–8 e
V2 contengono ancora percorsi locali: sanificarle è una decisione in sospeso.
[Evidenze, gate A–H e comandi](results/increment9/README.md).
