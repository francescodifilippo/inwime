# InWiMe — incremento 5: inferenza CPU completa oltre RAM

Acquisizione iniziata il 12 novembre 2026 e conclusa il 13 novembre 2026.
Base verificata: `8419dcfffebeb86f0c76e07d6cbac7a3732c9765`, branch `main`,
repository remoto `francescodifilippo/inwime`; directory locale ancora denominata
`ooc-inference`. Nessun commit o push. Nessun intervento su partizioni, mount,
configurazione WSL, driver o pacchetti di sistema. Nessun download di modelli.

## Risultato e forza della prova

Il runtime esegue realmente il decoder denso con pesi BF16 letti progressivamente:
inizializzazione, prefill di `[1, 7]`, tre decode con feedback greedy dei logits,
verifica del prefisso KV pubblicato e chiusura delle risorse. I token generati dal
percorso sincrono sono `[118, 101, 118, 101]`; i primi tre vengono reinseriti nel
modello. Le lunghezze pubblicate sono 2, 3, 4, 5. Sono ID di un modello sintetico
non addestrato, senza tokenizer: non costituiscono una prova di qualità linguistica.

| Livello | Risultato | Evidenza |
|---|---|---|
| A: pesi oltre il budget interno | Dimostrato con forward completo | 12.887.482.368 byte unici utilizzati contro circa 24 MB di piano sincrono |
| B: pesi oltre memoria del processo | Dimostrato con forward completo | `RLIMIT_AS` effettivo di 536.870.912 byte, senza modificare il sistema |
| B: pesi oltre RAM totale WSL | Dimostrato con forward completo | 12.887.482.368 byte utilizzati contro `MemTotal=11.483.848.704` byte |
| C: pesi oltre RAM fisica host | Non dimostrato, non tentato | RAM fisica rilevata: 16.900.739.072 byte; il checkpoint è più piccolo |

Il limite di spazio virtuale è un vincolo del processo, distinto da un limite RSS
o da un cgroup. Non è l'unica base della prova B: le matrici effettivamente elaborate
superano anche tutta la RAM WSL, senza ridurne la configurazione. Swap WSL configurato:
8 GiB; `VmSwap` del processo è zero negli snapshot acquisiti ai confini delle fasi.
Gli snapshot non escludono transitori tra i campioni. I pesi vengono letti in staging,
non mappati o caricati integralmente per delegarne la residenza allo swap.

Fonti: `results/increment5/large-sync-1.json`, confronto concorrente e riepilogo nella
stessa directory. Questi risultati riguardano questa architettura sintetica, non un
modello pubblico o un LLM generico già supportato.

## Kernel CPU e acquisizione dei pesi

`src/cpu_kernel.c` contiene tre funzioni senza allocatore, file I/O o dipendenze GPU:
espansione BF16 finito in FP64, accumulo GEMV FP64 e generazione sintetica nativa BF16.
Il target separato `make cpu-kernel` usa il compilatore C disponibile, `-O3`,
`-fno-fast-math` e `-ffp-contract=off`; non richiede BLAS, NumPy, PyTorch o header Python.
Non richiede AVX512, AMX o istruzioni BF16 hardware. Non installa librerie nel sistema.

`inwime_runtime/math_cpu.py` fornisce gli adapter Python e C. Il kernel non conosce
checkpoint, shard, sessione o scheduler. Gli export dei buffer restano vivi durante
le chiamate C che rilasciano il GIL, impedendo resize che invaliderebbero i puntatori.
Il chiamante non deve mutare gli input mentre un'operazione li usa.

`inwime_runtime/gemv.py` emette regioni contigue con dimensione configurabile.
Una tile può contenere più righe oppure terminare dentro la dimensione di riduzione;
il risultato parziale resta nella destinazione FP64 già prenotata. Non occorrono copie
residenti della matrice o un piano per tutti i suoi elementi. Input e output che si
sovrappongono sono rifiutati prima delle letture, anche tramite exporter differenti.
Le arene GEMV sono buffer FP64 contigui e scrivibili. Il coordinatore risolve i segmenti
multi-shard e mantiene ownership fino al ritorno del kernel sincrono.

Il percorso standalone `StreamedGEMV` prenota soltanto pesi, staging, metadati e
richieste. Input e output del chiamante richiedono ulteriori `8*(colonne+righe)` byte,
riportati separatamente. Nel decoder tutte le arene numeriche restano invece nel piano
del motore. La memoria cresce con tile, input/output, attivazioni e KV necessari;
non con il numero totale dei coefficienti delle matrici.

Il default storico di `CPURuntime` resta il percorso scalare. Un `math_backend`
esplicito abilita l'adapter e l'aggregazione delle righe. L'espansione è esatta e
le somme procedono in ordine crescente di colonna, senza contrazione FMA o
riassociazione intenzionale. I confronti piccoli verificano uguaglianza bit per bit
con il vecchio percorso, oltre alle tolleranze del riferimento indipendente.

## Checkpoint denso completo

`tools/checkpoint5.py` genera il formato già supportato `inwime-segmented-bf16-v2`.
La configurazione è: 24 layer, width 4096, 32 teste, hidden FFN 16.384,
vocabolario 257, contesto 16. Matematica invariata: embedding token e posizione,
RMSNorm, attenzione causale, proiezioni Q/K/V/O, SwiGLU, residui, normalizzazione
finale e output non condiviso.

Sono presenti 220 tensori reali. Le 169 matrici lineari vengono elaborate
integralmente per ogni token: 7 per layer più output. Nessun esperto inutilizzato,
padding o tensore fittizio. Gli embedding sono consultati per le righe necessarie,
come richiesto dalla matematica; non vengono conteggiati integralmente nei pesi usati.

| Quantità | Byte |
|---|---:|
| Payload BF16 totale | 12.889.645.056 |
| Allocazione filesystem `st_blocks*512` | 12.889.653.248 |
| Matrici interamente calcolate per token | 12.887.007.232 |
| Unione dei pesi effettivamente usati nei cinque token | 12.887.482.368 |
| Letture BF16 d'inferenza, riletture incluse | 64.437.125.120 |
| Scansione SHA256 iniziale separata | 12.889.645.056 |

Le letture attese vengono ricostruite per ciascuno dei 220 tensori, confrontate
esattamente con `tensor_read_bytes` e con il numero di coefficienti GEMV.
La prova non si limita ad aprire, indicizzare, verificare hash o sommare i file.
Le uscite di ogni proiezione alimentano attenzione, FFN, residui e logits del forward.

Il generatore usa un finalizzatore SplitMix64 indicizzato da nome del tensore e
indice del coefficiente, con aritmetica intera definita modulo 2^64. Segno e
mantissa variano per coefficiente; l'esponente segue il fan-in per evitare
attivazioni degeneri o overflow. Le scale RMSNorm sono 1. Le parole BF16 vengono
prodotte direttamente: non esiste un checkpoint FP32 sorgente né una quantizzazione.
I metadati dichiarano forme, layout C, byte order, seed, esponente, segmenti e hash.
La sequenza è riproducibile e non dipende dalle dimensioni del chunk del writer.

I file sono nuovi e distinti, scritti realmente in chunk di massimo 1 MiB,
con `fsync` e controllo dell'allocazione di ogni shard. Non si usano sparse file,
compressione o deduplicazione applicativa. `st_blocks` misura allocazione ext4,
non byte NAND né politiche interne del VHDX Windows. La generazione ha richiesto
87,68 s; le dimensioni e il picco RSS del writer sono in `generation.json`.
Il writer ha prodotto 193 shard, con picco RSS di 25.992 KiB.

L'audit precedente alla scrittura ha rilevato circa 871 GiB liberi nell'ext4 virtuale,
ma soltanto 185.963.954.176 byte liberi sul volume Windows sottostante. Entrambe le
misure sono state considerate; 12 GiB aggiuntivi lasciano un ampio margine.
Il vecchio checkpoint `inc4-large` rimane immutato e viene riusato per GEMV: un suo
forward attraverserebbe circa 6 GiB di output, mentre quasi tutti gli embedding
resterebbero inutilizzati. Non sarebbe una prova oltre la RAM WSL con i pesi usati.

## Correttezza, errori e stato

Il riferimento NumPy residente, l'oracolo scalare indipendente, le fixture originali
e tutte le loro tolleranze rimangono invariati. I nuovi test verificano GEMV con
`math.fsum`, confini di riga/riduzione/segmento/shard, ogni codice BF16 finito,
segno dello zero, rifiuto dei non finiti e decoder completo dense/MoE nei due formati.
Un nuovo checkpoint piccolo con dimensioni non multiple delle tile è verificato
integralmente contro il riferimento indipendente.

Sul modello grande si conservano gli input e tre uscite selezionate della prima
invocazione di ogni proiezione. Dopo la misura di inferenza, il verificatore legge
indipendentemente le righe fisiche dal manifest, decodifica BF16 tramite IEEE FP32
e calcola i dot product con `math.fsum`. Non usa `Catalog.spans`, espansione o GEMV
del runtime. Controlla 507 righe complete su 169 matrici, circa 5,92 MB logici.
Le copie diagnostiche degli input sono limitate, esterne alle arene e incluse in RSS.

La soglia per ogni dot product è fissata prima della prova:
`gamma(2*n)*sum(abs(w_i*x_i)) + 4*u*abs(reference) + 1e-300`, con `u=2^-53`.
È un controllo dell'accumulo FP64 sequenziale; non sostituisce le tolleranze del
decoder storico. Nel primo sincrono l'errore massimo è `6,76e-15` e il rapporto
massimo errore/soglia è circa `0,000196`.

Questo controllo è parziale: usa input intermedi prodotti dal runtime e non verifica
integralmente e indipendentemente tutto il forward grande. Il confronto completo
con oracle resta sui modelli piccoli. Il confronto fra modalità aggiunge determinismo
di logits, hidden e KV, senza diventare a sua volta un oracle indipendente.

Il commit KV resta transazionale: un solo proprietario pubblica banco, lunghezza e
generazione soltanto al termine della chiamata. I test del backend C iniettano short
read, errore I/O e cancellazione in entrambi i percorsi; il vecchio KV rimane identico,
la sessione fallisce e gli slot vengono drenati. La cancellazione non può interrompere
una syscall che non ritorna. Nessun nuovo protocollo di retry o recovery viene introdotto.

## Memoria, I/O e tempi osservati

Prima acquisizione sincrona: piano controllato **24.022.223 byte**, slot pesi FP64
2.097.152 byte, staging 524.288 byte, due banchi KV complessivamente 15.728.640 byte.
`ru_maxrss=53.776 KiB`; `VmHWM=55.873.536 byte` nello snapshot. I due indicatori
sono osservazioni diverse e non si sommano. Comprendono interprete e diagnostica.
GPU non inizializzata, allocazioni GPU del runtime zero.

Bootstrap JSON, heap Python, allocator, libreria nativa, stack e page cache non sono
interamente intercettati dal budget delle arene. Il limite di processo è applicato
separatamente e verificato con `getrlimit`; RSS viene misurato dal kernel.
I limiti cgroup rilevati sono `max`, non un tetto più basso della RAM WSL.

Il primo sincrono impiega 193,713 s per il prefill di due token e
128,200 / 117,564 / 120,129 s nei tre decode, pari a circa **0,00820 token/s**
in decode. Il workload completo dura 560,040 s; avvio e SHA iniziale 107,690 s.
Le quattro uscite generate, includendo il prefill, sono distinte dai cinque token
effettivamente elaborati e dai tre decode temporizzati.

Il tempo di servizio delle letture è 512,485 s, l'espansione BF16 23,347 s e
le chiamate GEMV 17,876 s. Throughput del solo GEMV: circa 3,60 GFLOP/s;
banda logica end-to-end: circa 0,115 GB/s. Il kernel non è il collo di bottiglia
principale in questa acquisizione. Il tempo del lease include anche overhead;
`owner_phase_remainder_ns` è un residuo, non un timer puro dello scheduler.

`/proc/self/io` attribuisce al workload sincrono 63.978.094.592 byte `read_bytes`.
Questi sono byte del percorso di blocchi contabilizzato dal kernel, non letture NAND.
La SHA iniziale e le riletture del validatore hanno contatori e tempi separati.
Tutte le letture sono buffered: `POSIX_FADV_DONTNEED` è soltanto un suggerimento Linux.
Non si dichiara un supporto fisicamente cold; cache Windows, controller e dispositivo
rimangono incontrollate. Una SHA su 12 GiB non rende tutto il modello cache-warm.

La macchina espone i7-12700H, x86-64, 20 CPU logiche. Il kernel numerico usa un thread;
la pipeline usa due slot e un worker I/O. AC risulta online. Temperatura, governor,
frequenze e limiti di potenza non sono controllati: le differenze prestazionali non
vengono attribuite causalmente a uno stato termico specifico.

### Confronto del decoder grande

Una sola acquisizione per modalità, in processi nuovi e sequenziali. L'avvio/SHA
è escluso dal workload. I due modi usano lo stesso checkpoint, tile/read di
262.144 elementi e la stessa sequenza; la pipeline dispone di due slot contro uno.

| Misura | Sync | Pipeline |
|---|---:|---:|
| Workload, s | 560,040 | 423,019 |
| Prefill di due token, s | 193,713 | 152,685 |
| Decode per token, s | 128,200 / 117,564 / 120,129 | 109,924 / 88,160 / 71,604 |
| Decode token/s | 0,00820 | 0,01112 |
| Startup e SHA, s | 107,690 | 47,267 |
| Piano controllato, byte | 24.022.223 | 26.668.239 |
| Picco `ru_maxrss`, KiB | 53.776 | 57.440 |
| Byte logici d'inferenza | 64.437.125.120 | 64.437.125.120 |
| `/proc/self/io/read_bytes` workload | 63.978.094.592 | 64.100.728.832 |
| Servizio letture, s | 512,485 | 378,522 |
| Espansione BF16, s | 23,347 | 26,253 |
| Chiamate GEMV, s | 17,876 | 18,675 |
| Overlap lettura/calcolo osservato, s | 0 | 2,319 |

La variazione wall osservata è **−24,47%**, rapporto sync/pipeline 1,324.
Logits completi, hidden e KV hanno hash identici bit per bit a ogni fase;
le 507 verifiche numeriche indipendenti coincidono nei due modi.

Questo confronto non isola un guadagno causale della pipeline: il servizio dello
storage cambia di circa 134 s, mentre l'intersezione osservata è soltanto 2,319 s.
Anche gli startup sono molto diversi e i decode diventano progressivamente più
rapidi. Cache inferiori, carico dell'host e stato termico rimangono variabili non
controllate. Non si ricava un beneficio generale dal solo rapporto tra i due wall.

### GEMV completo sul tensore preesistente da 6 GiB

Il checkpoint storico `inc4-large` viene riusato senza modificarlo. La matrice
`embedding`, shape `[201326592, 16]`, partecipa integralmente a una GEMV autonoma:
**3.221.225.472 coefficienti** moltiplicati e accumulati, non una scansione SHA.
Il vettore ha valori deterministici a segno misto. Non si presenta questa GEMV
come inferenza del vecchio modello: la prova decoder completa è quella sopra.

| Misura, un campione per modo | Sync | Pipeline |
|---|---:|---:|
| Matrice BF16 elaborata, byte | 6.442.450.944 | 6.442.450.944 |
| Tempo GEMV incluso I/O, s | 9,473 | 10,422 |
| Banda end-to-end, GB/s | 0,680 | 0,618 |
| Throughput GEMV end-to-end, GFLOP/s | 0,680 | 0,618 |
| Throughput della sola chiamata matematica, GFLOP/s | 1,673 | 1,604 |
| Servizio letture, s | 2,859 | 4,422 |
| Espansione, s | 2,390 | 2,386 |
| Piano del reader, byte | 3.641.460 | 6.287.476 |
| Picco RSS, KiB | 1.601.004 | 1.603.228 |

Input/output del chiamante richiedono **1.610.612.864 byte**, soprattutto per
i 201 milioni di risultati. Sono misurati separatamente dalle arene del reader e
inclusi in RSS. `RLIMIT_AS=2.147.483.648` byte: la matrice supera il limite del
processo, ma non la RAM totale WSL. Quattro righe complete vengono confrontate
indipendentemente con `math.fsum` (errore massimo `2,08e-17`); l'hash di tutto
l'output è identico nei due modi. Non è un confronto indipendente di ogni risultato.
Il kernel legge progressivamente tutti i coefficienti, senza conservare la matrice.

La pipeline qui regredisce di circa il 10,0%, pur mostrando overlap; il risultato
non viene generalizzato al decoder. I contatori di blocco registrano circa 6 GiB
per workload anche dopo la SHA separata, con gli stessi caveat sulle cache inferiori.

### Riduzioni spezzate e fixture media

La matrice `layers.0.ffn.down` del decoder grande ha shape `[4096, 16384]`,
payload 134.217.728 byte. Tile/read di **8191 elementi** spezzano le righe e la
riduzione, attraversando anche segmenti e shard. Il risultato completo è identico
fra modalità; quattro dot product indipendenti hanno errore massimo `1,30e-15`.

| Carico | Sync | Pipeline | Variazione wall |
|---|---:|---:|---:|
| GEMV 128 MiB, tile 8191, un campione | 388,23 ms | 1372,06 ms | +253,4% |
| Decoder `inc4-dense`, 3 campioni, mediana | 51,42 ms | 70,76 ms | +37,6% |
| Decoder `inc4-dense`, minimo–massimo | 48,94–66,50 ms | 63,83–80,38 ms | — |

La GEMV con riduzioni spezzate riserva 1.450.133/1.556.619 byte sync/pipeline,
con picchi RSS 26.816/27.152 KiB; input/output richiedono altri 163.840 byte.
Il reader registra 136.298.496 byte di blocco per 134.217.728 byte logici.
La granularità di circa 16 KiB produce oltre 8.000 letture: trasporto e scheduling
restano costosi. La presenza di 52,6 ms di overlap non compensa l'overhead.

La fixture media esegue lo stesso prefill più tre decode, usa il checkpoint denso
preesistente da 3.152.992 byte e legge 15.755.200 byte logici per campione.
I contatori di blocco sono 3.166.208 byte: il primo passaggio dopo DONTNEED legge
da blocchi, mentre il resto sfrutta la cache Linux. I processi vengono avviati
sequenzialmente con ordine sync/pipeline alternato fra i tre campioni; nessun test
gira contemporaneamente alle misure. Le arene sono 2.958.558/5.604.574 byte e i
massimi RSS 27.488/30.236 KiB. Tutti gli output coincidono bit per bit.

Queste regressioni confermano che la concorrenza non è un'accelerazione automatica.
Non è stato introdotto autotuning: kernel compilato e raggruppamento delle righe
sono le sole ottimizzazioni necessarie a rendere ragionevole il forward grande.

## Test, revisione e provenienza

La baseline iniziale conferma **152 test superati**. La suite finale comprende
**181 test superati in 33,37 s**, senza test saltati o tolleranze allargate.
Log completo: `results/increment5/final-run.log`. I 29 casi nuovi comprendono
GEMV indipendente, manifest v1/v2, decoder dense/MoE, generazione scalabile,
codici BF16, cancellazione/errori e protezione delle arene native. Controlli di
sintassi Python e `git diff --check` non segnalano errori.
Ambiente verificato: CPython 3.14.4, NumPy 2.3.5, pytest 9.0.2,
compilatore `cc` GCC 15.2.0 già installato.

Le prime prove TDD fallivano perché mancavano il modulo GEMV, il parametro
`math_backend` e il nuovo generatore. La revisione AskMatt su due assi ha poi
identificato due difetti reali, corretti prima della suite finale:

- **Standards:** gli indirizzi C venivano ottenuti da export temporanei. Un altro
  thread avrebbe potuto ridimensionare un array del chiamante durante il kernel.
  Ora gli export restano vivi fino al ritorno C; una regressione verifica il
  rifiuto del resize durante la chiamata e il successivo rilascio.
- **Standards/Spec:** un GEMV con input/output sovrapposti sovrascriveva coefficienti
  del vettore ancora necessari alle righe successive. Il controllo ora confronta
  gli intervalli di indirizzi prima dell'I/O e copre array identici, memoryview e
  exporter distinti della stessa memoria.

La revisione finale non riporta rilievi importanti irrisolti. Il riferimento e le
evidenze storiche non sono stati modificati. Gli artefatti nuovi distinguono i
seguenti momenti: la coppia decoder da 12 GiB è stata acquisita prima delle ultime
protezioni del wrapper contro resize/alias; GEMV, fixture media e suite finale usano
le protezioni completate. Il kernel C, la matematica, il checkpoint e l'ordine delle
riduzioni sono identici in tutte le acquisizioni. Le prove grandi usano arene
disgiunte sul proprietario, senza resize concorrenti: i difetti non invalidano i
risultati numerici osservati. Non si attribuisce retroattivamente una nuova misura
del decoder grande al wrapper finale e non si ripete quel carico costoso per soli
controlli di ammissione/lifetime, verificati separatamente.

L'indice `results/increment5/artifact-sha256-final.json` identifica sorgenti finali,
binario C, manifest utilizzati ed evidenze. Ogni worker registra inoltre identità
del checkpoint e SHA256 del binario caricato. Le misure originali restano immutate;
`large-summary.json` e i riepiloghi delle sottodirectory derivano dai JSON acquisiti.
I checkpoint restano locali in `checkpoints/`, esclusi da Git.

## Riproduzione

Dalla radice del repository, con C compiler, Python, NumPy e pytest già disponibili
per i test. Il runtime non importa NumPy o il riferimento; non serve una GPU.
Usare sempre directory e file di output nuovi. Non rigenerare i checkpoint se
quelli già presenti sono utilizzabili.

```bash
make cpu-kernel
python3 -m pytest -q tests/test_reference_decoder.py tests/test_runtime_cpu.py \
  tests/test_pipeline_cpu.py tests/test_segmented_cpu.py \
  tests/test_gemv_cpu.py tests/test_checkpoint5.py

# Audit senza scritture; --execute crea realmente circa 12 GiB, solo se necessario.
python3 -m tools.checkpoint5 checkpoints/nuovo-decoder
python3 -m tools.checkpoint5 checkpoints/nuovo-decoder --execute

# Coppia completa: almeno diversi minuti e circa 155 GB di letture logiche,
# comprese le due SHA iniziali; 512 MiB di limite di spazio virtuale per worker.
python3 -m tools.benchmark_increment5 checkpoints/inc5-dense-12g \
  --large --output results/nuova-coppia-decoder

# Matrice preesistente da 6 GiB; output necessario di circa 1,5 GiB.
python3 -m tools.benchmark_increment5 checkpoints/inc4-large --large \
  --tensor embedding --address-limit-mib 2048 --output results/nuovo-gemv-6g

# Matrice del modello grande con tile inferiori alla riduzione.
python3 -m tools.benchmark_increment5 checkpoints/inc5-dense-12g --large \
  --tensor layers.0.ffn.down --tile 8191 --read 8191 \
  --output results/nuove-riduzioni

# Ripetizioni brevi, stesso decoder e checkpoint nei due modi.
python3 -m tools.benchmark_increment5 checkpoints/inc4-dense --samples 3 \
  --output results/nuova-fixture-media
```

Per eseguire un solo worker: `python3 -m tools.verify_increment5 --help`.
L'integrità `full` rimane il default. `metadata` resta una rinuncia esplicita alla
SHA, non una scorciatoia selezionata automaticamente in base alla dimensione.
Audit, picco RSS, I/O, arene e risorse dopo la chiusura sono salvati nei JSON.

## Architettura e limiti residui

L'alternativa A rimane il riferimento: il coordinatore mantiene stati e prenotazioni,
mentre accesso ai pesi e calcolo sono adapter separati. La pipeline può leggere
mentre il C rilascia il GIL durante il calcolo. Briareus, Mnemosyne, Ariadne e Iris
restano responsabilità logiche del runtime compatto, senza nuove autorità concorrenti.

Il risultato rimuove l'ostacolo specifico del forward completo oltre RAM con pesi
originali. Non dimostra supporto a safetensors o GGUF, tokenizer, template chat,
RoPE/GQA di un modello pubblico, logits upstream, contesti lunghi, GPU, distribuzione
o metadati di milioni di file. Un primo modello pubblico richiederà specifica esatta
del suo adapter e confronti con output ufficiali, oltre al solo riuso del GEMV.
Le latenze osservate non promettono un'esperienza interattiva utile.

Il lavoro termina all'incremento 5, lasciando codice ed evidenze per revisione.
Non viene avviato l'incremento 6.
