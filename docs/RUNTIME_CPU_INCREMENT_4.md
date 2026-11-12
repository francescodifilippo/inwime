# InWiMe — incremento 4: segmenti fisici e capacità out-of-core

12 novembre 2026. Base: `885a5d4f38ad9564d0b0b2400c090e1fa37c18cb`.
Durante l’implementazione: nessun commit, push, download, installazione o intervento
sul sistema/dischi. Il consolidamento successivo autorizza un solo commit locale,
senza push e senza avviare l’incremento 5.
Pesi originali BF16; operatori scalari FP64 invariati. GPU non inizializzata,
allocazioni GPU del runtime zero. Riferimento, tolleranze e risultati storici invariati.

## Risultato e forza delle evidenze

| Evidenza | Risultato | Limite |
|---|---|---|
| A: payload oltre il budget pesi | Dimostrata, fixture e checkpoint scalabili | Non equivale a superare RAM |
| B: payload oltre memoria del processo/ambiente | Dimostrata per apertura, verifica completa e accesso progressivo | Non per inferenza completa oltre RAM |
| C: payload oltre 16 GiB host | Non dimostrata | Checkpoint di circa 12 GiB |
| Tensore molto grande | Scansione e SHA256 indipendente di **6 GiB** | Hash dei byte, non GEMV completo |
| Espansione/calcolo su checkpoint grande | 98.304 byte campionati con oracle analitico | Non tutti i 6 GiB espansi/calcolati |
| Forward completo | Dense/MoE originali e segmentati, più decoder scalabili da 3/6 MB | Nessun modello reale né decoder oltre RAM |

Il checkpoint grande ha payload **12.884.911.776 byte**, allocazione filesystem
**12.884.926.464 byte**, 193 file. WSL espone **11.483.848.704 byte** di RAM totale:
anche tutta la RAM visibile sarebbe insufficiente a contenere il payload. Il limite
non è un budget artificiale InWiMe. Il runtime non rende residenti i pesi tramite swap. Non è stato campionato
`VmSwap`: non si deduce una misura di assenza di paging dal solo rispetto delle arene.

## Formato fisico v2 e contratti

`inwime-segmented-bf16-v2` conserva config, nomi, forme logiche, BF16 little-endian,
layout C, `files` con dimensione e SHA256 e `nbytes` totale per tensore. Sostituisce
`shard`/`offset` del tensore con `segments`, per esempio:

```json
{"dtype":"BF16", "shape":[4], "nbytes":8,
 "segments":[
   {"start":0,"shard":"part-000000.bf16.bin","offset":0,"nbytes":6},
   {"start":3,"shard":"part-000001.bf16.bin","offset":0,"nbytes":2}
 ]}
```

`start` è in **elementi logici**; offset e nbytes sono byte fisici. La lista deve
essere ordinata, contigua, completa, senza sovrapposizioni; tutti i segmenti hanno
lunghezza positiva e offset/lunghezza pari. Ogni shard deve essere interamente
indicizzato una sola volta: buchi, offset sovrapposti nello stesso shard e dati non indicizzati sono errori.
Nomi diversi collegati allo stesso inode tramite hardlink non vengono deduplicati
né respinti: il generatore della prova crea file distinti, senza hardlink.
Non si confonde una riga logica con un confine fisico. La lettura v1 rimane supportata
normalizzandola a un segmento. Il numero di shard non è più fisso a due.

`Catalog.spans(name,start,count,max_elements)` risolve pigramente un intervallo in
pezzi fisici, con ricerca binaria del primo segmento. Ogni pezzo indica sorgente,
offset, elementi e posizione nella destinazione. Non materializza il piano di tutto
il tensore o checkpoint. Il vecchio offset diretto di `region` è `None` quando un
tensore ha più segmenti: i nuovi consumatori devono usare `spans`, evitando un falso
offset contiguo. I campi shard/offset restano validi per tensori a segmento unico.

Sincrono e pipeline usano lo stesso risolutore. I worker scrivono soltanto lo staging
prenotato e pubblicano il completamento fisico; un unico proprietario espande i BF16,
concede lease e pubblica KV. Tile più piccole dei segmenti, confini di segmento/shard
e letture più piccole delle tile non cambiano ordine matematico o ownership.

Identità delle sorgenti aperte: device, inode, size, mtime e ctime prima/dopo ogni
lettura; apertura read-only, file regolari, niente symlink dove O_NOFOLLOW è disponibile.
Un file mancante fallisce esplicitamente. Short read/completamento invalido non
pubblicano una regione valida. Nessun retry o recupero parziale silenzioso.
La firma riguarda il file aperto; non è autenticazione del manifest né snapshot
filesystem contro modifiche ostili. Il backend conserva il contratto POSIX precedente.

## Integrità, memoria e limiti operativi

`CPURuntime(..., integrity='full')` resta il default: SHA256 completa in staging
limitato, byte e tempo separati dall'inferenza. `integrity='metadata'` è una rinuncia
**esplicita alla verifica del contenuto**: controlla layout, dimensioni e firme,
ma può accettare corruzione preesistente con dimensione invariata. La politica compare
nelle statistiche; nessuna selezione automatica in base alla dimensione. Per una
sorgente reale non fidata usare full. Il test di corruzione dimostra la differenza.

Il manifest ha un limite configurabile nel catalogo di 16 MiB, controllato prima
lettura/parse. I metadati crescono con tensori/segmenti/shard, non con i byte dei pesi.
Il generatore ammette al massimo 32.768 descrittori stimati, 10.000 esperti e una
stima conservativa di shard entro il limite dei file aperti, con tetto 4.096. Sono
politiche di questa implementazione, non vincoli dell'architettura. Tutti gli shard
sono attualmente aperti: cataloghi con milioni di file richiederanno handle limitati
con gestione della vita fisica; non sono dichiarati supportati.

Il JSON viene comunque parsato integralmente; i temporanei interni del parser e gli
stack Python non hanno un limite byte per byte imposto dalle arene. Bootstrap noto,
metadati trattenuti, arene, RSS e memoria OS restano quantità distinte. Un catalogo
grande non impone automaticamente arene decoder ammissibili: il verificatore tensoriale
usa un piano senza logits/KV proporzionali al vocabolario gigante, e **non** lo presenta
come forward. L'API decoder conserva ammissione e arene preallocate precedenti.

`logical_requested_bytes` conta domanda logica di regioni, anche se fallite;
`inference_read_bytes` conta byte effettivamente consegnati. Il tool registra inoltre
i byte visitati del workload. `backend_read_calls` include la scansione iniziale;
`integrity_read_bytes/ns` la separano dai byte di inferenza. I contatori Linux
`/proc/self/io/read_bytes` sono I/O attribuito al processo dal kernel, **non byte NAND**.
WSL/VHDX e cache Windows/controller rimangono sotto il filesystem osservato.

Errori, timeout restituiti dal backend e cancellazioni mantengono il KV pubblicato
precedente. Le syscall già avviate conservano il buffer fino al ritorno fisico.
Non si introduce un timeout capace di interrompere un backend che non torna mai.
Tracce e contatori restano limitati; `peak_concurrent_reads` distingue lettori reali
da `peak_occupied_slots` (richieste ammesse, anche READY/IN_USE).

## Generazione riproducibile e audit

`tools.checkpoint4` genera o ripartisce checkpoint, rifiutando directory esistenti.
Senza `--execute` stampa soltanto audit e stima. Il repacker verifica SHA e firme
sorgenti e copia i byte originali in chunk limitati, senza cambiare il riferimento.
Il generatore nativo usa parole BF16 esatte periodiche a 256 elementi, valori
`(-1)^floor((i mod 256)/128) * (1+(i mod 128)/128)/16`; normalizzazioni pari a 1.
Nessuna conversione da un checkpoint FP32 o quantizzazione. È un carico sintetico
regolare, non distribuzione statistica di pesi addestrati.

Audit prima delle scritture: MemAvailable/MemTotal, cgroup corrente e antenati,
RLIMIT_AS/NOFILE, spazio filesystem, stima descrittori/metadati e I/O. Qui RLIMIT_AS
e limiti cgroup memoria sono illimitati: il limite fisico visibile è WSL.
Circa 883 GiB liberi; riserva disco minima 2 GiB. Stima iniziale a 100 MiB/s:
circa 369 secondi per 3 payload di I/O; è un'ipotesi di pianificazione, non una
misura del dispositivo. Generazione osservata: 52,25 s, picco RSS 25.380 KiB,
chunk massimo 1 MiB. Nessun file sparse, compressione o deduplicazione applicativa;
scritture reali e fsync per shard. st_blocks prova l'allocazione ext4, non l'occupazione
NAND o un audit delle politiche del disco virtuale Windows.

Checkpoint persistenti in `checkpoints/inc4-*`, esclusi da Git; circa 12 GiB complessivi,
lasciati disponibili per ripetere le prove. Nessun download di modelli. Non si usa
`/tmp` (tmpfs) per i pesi. Non sono state aggiunte altre unità o configurazioni mount.
Non è dimostrato parallelismo su due dispositivi fisici; i nomi logici non contengono
un identificatore NVMe e l'adapter resta sostituibile.

## Benchmark misurati

Fonti: `results/increment4/summary.json`, file `final-medium-*`, `final-decoder-*`,
`large-progressive-verified.json`, `large-samples-final.json`. Tre processi freschi
per configurazione, ordine alternato, acquisizione finale sequenziale, niente test
contemporanei. Tile/read uguali, due slot/un worker in pipeline. Temperatura,
frequenza e potenza non fissate; sysfs registra AC online dove esposto, WSL nasconde
altri sensori. Non si attribuiscono le variazioni a un preciso stato termico.

| Carico | Sync mediana [min–max] ms | Pipeline mediana [min–max] ms | Overlap pipeline mediano ms |
|---|---:|---:|---:|
| Scan numerico 4 MiB, tile 16.384 | 971,90 [918,18–992,73] | 1143,27 [1124,70–1155,22] | 213,24 |
| Scan numerico 4 MiB, tile 65.536 | 954,73 [917,29–967,88] | 1140,78 [1114,13–1184,17] | 23,96 |
| Forward denso 1 token, hidden 16.384 | 1261,03 [1257,78–1265,13] | 6106,22 [5999,96–6115,64] | 22,45 |
| Forward MoE 1 token, hidden 4.096, 8 esperti | 644,64 [630,28–658,22] | 3006,86 [2956,83–3319,08] | 9,31 |

Lo scan numerico espande e verifica **ogni** elemento contro formula indipendente e
accumula una somma; non è GEMV né inferenza. Payload totale 8.398.496 byte, embedding
visitato 4.194.304 byte. Pesi riservati sync/pipeline: 131.072/262.144 byte per tile
16k, 524.288/1.048.576 byte per tile 64k; staging BF16 1/4 dei rispettivi pesi FP64.
Picco RSS massimo delle prove tabellate **24.896 KiB**. Allocazioni controllate nel
JSON: non equivalgono a RSS. L'incremento del tile non aumenta il payload visitato.

Per scan 16k: I/O service 1,28/310,31 ms, calcolo 518,05/688,95 ms,
attesa 2,96/10,89 ms, espansione 448,12/435,41 ms (sync/pipeline).
Per scan 64k: I/O 0,89/135,71 ms, calcolo 505,30/688,52 ms,
attesa 1,81/0,45 ms. I tempi worker wall includono scheduling/preemption/contesa;
non rappresentano latenza isolata del disco. Overlap è intersezione osservata,
non speedup dedotto. Il residuo owner è wall meno attesa, espansione e compute,
non un timer puro dello scheduler. Tutte le componenti per campione sono nei JSON;
le mediane delle componenti non devono sommarsi alla mediana del totale.

Il decoder denso legge 3.151.040 byte da un payload di 3.152.992;
MoE legge **1.578.688 byte** da **6.299.232**, scegliendo soltanto esperti 6 e 7
per entrambi i layer in questo input. La scansione iniziale legge invece tutti i
pesi, anche esperti inutilizzati. `tensor_read_bytes` rende verificabile la selezione.
Non si usa il catalogo di esperti inutilizzati come prova di calcolo enorme.
I forward scalabili coincidono bit per bit fra modi, ma non hanno un riferimento
residente indipendente completo: tale evidenza forte resta sulle due fixture originali.
Non si pubblicano token/s da un solo token sintetico.

I carichi medi sono post-SHA, Linux buffered, con `read_bytes=0` nelle passate:
cache calda Linux osservata, cache inferiori non controllate. La prova grande usa
POSIX_FADV_DONTNEED dopo SHA, un suggerimento Linux, non certificazione cold del
supporto. Gli accessi selettivi senza SHA sono etichettati **uncontrolled**.
Una SHA di un checkpoint oltre RAM non rende tutto il checkpoint cache-warm.

La prova grande legge progressivamente tutti i 6 GiB dell'embedding, calcola SHA256
in staging e confronta con un digest costruito indipendentemente mediante codifica
IEEE FP32/BF16 dei valori analitici. Il digest atteso è calcolato fuori dal tempo
workload. Nessun mapping o allocazione dell'intero tensore. Acquisizione finale: **10,675 s** di workload, **13,270 s** di startup/SHA
del checkpoint; picco `ru_maxrss` **28.100 KiB**, VmHWM **28.256 KiB**,
arene/riserva controllata **3.641.460 byte**. Il kernel attribuisce
**6.442.450.944 byte** `read_bytes` al workload: distinti dal dispositivo fisico
e dai 12 GiB letti per integrità. I tempi di read service e SHA del workload
sono rispettivamente 7,253 s e 3,260 s. RSS e VmHWM sono due osservazioni del
processo, non valori da sommare. Il campionamento numerico pipeline legge 12 tile da 4.096 elementi,
98.304 byte complessivi, distribuite ogni 512 MiB, verificando ogni valore letto.
Questo è accesso numerico selettivo al checkpoint oltre RAM, non elaborazione
numerica completa di quel tensore.

## Validazione, problemi e decisione

Test originali: 104 passavano già dopo l'estensione. Nuovi test coprono confronto
oracle/sync/pipeline e v1/v2 per dense/MoE, varie tile/read, intervalli arbitrari,
confini, determinismo, descrittori mancanti/sovrapposti/incompleti, alias fisici,
shard sconosciuti/mancanti, identità cambiata in uno shard successivo, corruzione,
short read, timeout backend e cancellazione senza pubblicazione KV parziale.
Una race del test storico fuori ordine è emersa nella prima suite completa:
la barriera segnalava una syscall, non il completamento di tutte le letture della
richiesta. Il test ora attende anche l’ack fisico del job, senza cambiare il runtime;
il fallimento è conservato in `pytest-first-full.txt`.
**144 test superati in 26,00 s**, inclusi i 104 precedenti e 40 nuovi.
Il log finale della suite è `results/increment4/pytest-final.txt`.
`compileall` e `git diff --check` completati senza errori.
`artifact-sha256.json` identifica codice e manifest usati per la riproduzione;
`preflight-final.json` registra il preventivo aggiornato senza scritture di pesi.

Revisione AskMatt con due assi indipendenti, Standards e Spec: corretti limite
preventivo dei descrittori (prima era verificato troppo tardi), lettura manifest
sorgente prima del limite, chiusura dei file dopo errore parziale di apertura,
audit del cgroup corrente e etichette cache troppo ottimistiche. Nessuna modifica
alla matematica per ottenere prestazioni. La verifica finale include regressioni
sul rifiuto preventivo e sulla memoria dei metadati.

Rispetto all'incremento 3, i dati letti passano da decine di KB/ritardi artificiali
a MB reali e GB progressivi. La pipeline resta più lenta nei decoder (circa 4,7–4,8x):
la larghezza logica 16 mantiene molte piccole richieste per riga anche con hidden
grande. Nei 4 MiB contigui è circa 18–20% più lenta, pur con overlap. Non esiste
un guadagno generale dalla sola concorrenza. Non si confondono i nuovi carichi con
lo speedup del workload ritardato dell'incremento 3.

L'alternativa A resta la baseline: nessuna nuova autorità o processo è servita
per segmentare i tensori, proteggere staging e mantenere commit KV. Il controllo
per molte righe minuscole è costoso, ma non prova superiorità di B/C. Il prossimo
passo motivato è un adapter di checkpoint reale con manifest/shape verificati e
operatori CPU a blocchi di righe/riduzioni, mantenendo oracle e ownership. Servono
validazione numerica indipendente e una prova di forward oltre RAM con arene output
ammissibili prima di dichiarare runnable un LLM enorme. Restano non dimostrati:
C, inferenza completa B, GEMV completo sul tensore da 6 GiB, multi-NVMe fisico,
checkpoint reali e metadati di milioni di file. Nessuno è anticipato in questo incremento.

## Riproduzione

Dalla radice, con le dipendenze di test già presenti. Usare directory nuove;
omettere `--execute` per ottenere soltanto il preventivo. I valori seguenti
ricreano i checkpoint, non sovrascrivono quelli presenti:

```bash
python3 -m tools.checkpoint4 checkpoints/nuovo/inc4-large --vocab 201326592
python3 -m tools.checkpoint4 checkpoints/nuovo/inc4-large --vocab 201326592 --execute
python3 -m tools.checkpoint4 checkpoints/nuovo/inc4-medium --vocab 131072 --shard-bytes 1000000 --segment-bytes 131074 --execute
python3 -m tools.checkpoint4 checkpoints/nuovo/inc4-dense --hidden 16384 --shard-bytes 1000000 --segment-bytes 131074 --execute
python3 -m tools.checkpoint4 checkpoints/nuovo/inc4-moe --variant moe --hidden 4096 --experts 8 --shard-bytes 1000000 --segment-bytes 131074 --execute
python3 -m tools.checkpoint4 checkpoints/nuovo/repacked --source fixtures/reference_decoder/moe --shard-bytes 510 --segment-bytes 14 --execute
python3 -m tools.benchmark_increment4 --checkpoint-root checkpoints/nuovo --output results/increment4-rerun --large
python3 -m pytest -q tests/test_segmented_cpu.py tests/test_runtime_cpu.py tests/test_pipeline_cpu.py tests/test_reference_decoder.py
```

La prova grande richiede spazio reale e tempo; ripetere audit/stima nel nuovo ambiente.
Il rapporto termina all'incremento 4; non propone commit o push automatici.


## Revisione mirata e consolidamento

La revisione finale ha riprodotto un difetto circoscritto del repacker: aggiungendo
byte in coda a uno shard sorgente, l’hash del solo intervallo dichiarato rimaneva
valido e il repacker accettava dati fisici non indicizzati. Ora verifica anche che
la sorgente sia un file regolare della dimensione esatta e rifiuta letture corte
nella scansione iniziale. Nessun cambiamento a layout, valori BF16, calcolo o KV.

Due regressioni (dense/MoE) fallivano prima della correzione e ora passano. Due
ulteriori test confrontano indipendentemente **tutti i byte logici di tutti i
tensori**, inclusi esperti non visitati, dopo repacking v1→v2 e v2→v2. Quattro
casi aggiuntivi verificano il rifiuto di offset/lunghezze BF16 dispari.
Suite completa confermata: **152 test superati in 26,21 s**, 144 precedenti più
8 casi di consolidamento; log `results/increment4/pytest-consolidation.txt`.

Le revisioni Standards e Spec non hanno rilevato altri difetti importanti.
Il riepilogo prestazionale è stato ricalcolato dai JSON già acquisiti e coincide
esattamente con `summary.json`: pipeline più lenta in tutti e quattro i carichi.
La prova da 12 GiB **non è stata ripetuta**. Sono stati verificati manifest,
dimensioni effettive, allocazioni st_blocks e 193 inode distinti mediante soli
metadati filesystem. Le evidenze di memoria/I/O restano quelle dell’acquisizione,
con i caveat page cache/WSL riportati sopra; nessuna ragione per invalidarle.

L’indice `artifact-sha256.json` dell’acquisizione resta immutato; il sorgente del
repacker corretto è identificato separatamente in `artifact-sha256-consolidated.json`.
I benchmark non vengono attribuiti retroattivamente al generatore rivisto.
Riferimento, fixture e benchmark storici sono invariati rispetto a HEAD iniziale.
Checkpoint, ambienti, segreti e temporanei restano esclusi da Git; staging esplicito
limitato a codice, test, documenti ed evidenze piccole dell’incremento 4.
Restano tutti i requisiti non dimostrati già elencati, in particolare inferenza
completa oltre RAM e payload superiore alla RAM fisica dell’host.
