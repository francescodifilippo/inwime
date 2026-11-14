# Incremento 8 — LLM pubblico BF16 sotto il payload dei pesi

14 novembre 2026. Base `main`, `ed166eddaa37ad5db6239d405bcbe8f7f8cdba49`,
repository `francescodifilippo/inwime` (directory locale `ooc-inference`).

**SmolLM2-1.7B esegue realmente prefill e tre decode con limite di spazio virtuale
128 MiB, contro 3.422.752.768 byte di pesi BF16 unici utilizzati.** Parità completa
Transformers standard FP32 alle soglie predefinite; KV transazionale verificato.
Il [gate riproducibile](../results/increment8/summary.json) distingue le condizioni A–F.
Questa prova riguarda il limite del processo: il checkpoint entra nella RAM WSL
e nella RAM fisica host. Non dimostra il superamento di nessuna delle due.

## Stato acquisito e provenienza

Working tree iniziale pulito, branch e commit corrispondenti alla richiesta.
Letti brief, README, decisione architetturale, rapporti 5–7, risoluzione numerica,
audit Strata e watchlist; esaminati adapter, sorgente, coordinatori, pipeline,
GEMV, wrapper/kernel C e strumenti oracle/gate/repack. Baseline **382 test PASS**.
Nessuna modifica ai sorgenti runtime, kernel o strumenti degli incrementi precedenti.

Modello originale [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/tree/d4c18ccb8a25e6e637e0609aac08a5d1a0436d1b),
revisione immutabile `d4c18ccb8a25e6e637e0609aac08a5d1a0436d1b`, Apache-2.0.
L'[API ufficiale con blobs](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-1.7B/revision/d4c18ccb8a25e6e637e0609aac08a5d1a0436d1b?blobs=true)
conferma file **3.422.777.952 byte** e SHA256
`1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0`.
Digest verificato dopo download, durante gli avvii e al termine. Payload
**3.422.752.768 byte**, 1.711.376.384 parametri in **218 tensori tutti BF16**;
25.184 byte complessivi per header e prefisso uint64. Nessun padding non indicizzato.

La [config originale](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/d4c18ccb8a25e6e637e0609aac08a5d1a0436d1b/config.json)
fornisce 24 layer, hidden 2048, intermediate 8192, Q/KV head 32/32, dimensione
head 64, theta 130000, epsilon RMS 1e-5, vocabolario 49152, contesto 8192,
SiLU, nessun bias ed embedding condiviso. **Adapter Llama preesistente sufficiente**:
questi valori rimangono dati della configurazione, senza nuove costanti nel runtime.
Tiling, ordine FP64, operatori, stato e semantica non sono cambiati. FP64 è il
backend attualmente verificato, senza imporlo ai backend futuri.

Prima dell'acquisizione controllati `checkpoints/`, `data/`, `artifacts/` e cache
Hugging Face: nessuna copia del modello individuata. Inventario: circa 25 GiB di
checkpoint precedenti, 5,6 GiB di dati, 189 MiB di artefatti, conservati. Spazio
iniziale circa 853 GiB nel filesystem ext4 virtuale e **161.486.647.296 byte** nel
volume Windows sottostante. Preventivo conservativo circa **7,12 GB** per originale,
un repack e fino a 256 MiB di evidenze/catture. Nessun download/cache duplicato:
`.part` rinominato in posto dopo verifica, buffer download ≤1 MiB. Scaricati solo
config, generation config, tokenizer JSON/config, special token map, README e
checkpoint. I piccoli file sono verificati anche contro gli ID Git blob ufficiali.

Originali in `checkpoints/smollm2-1.7b-d4c18cc/`; unica copia aggiuntiva in
`checkpoints/smollm2-1.7b-d4c18cc-sharded/`. Payload e catture sono esclusi da Git.
[Acquisizione](../results/increment8/acquisition.json),
[provenienza](../results/increment8/provenance.json),
[integrità finale](../results/increment8/integrity-after.json).
Riutilizzato il venv oracle7; nessuna nuova installazione. Nessuna modifica a
WSL, swap, cgroup, driver, filesystem, mount, protezioni o configurazione energetica.

## Inferenza, copertura e memoria

Tokenizzazione ufficiale dentro il processo limitato, confrontata con AutoTokenizer
anche su Unicode/newline e special token. `TOKENIZERS_PARALLELISM=false` evita
un pool Rayon superfluo. Nessun import torch/NumPy/Transformers nel processo InWiMe.

`The capital of France is` → `[504,3575,282,4649,314]`.
Prefill 5 token; tre decode reinseriscono `[7042],[30],[198]`; le quattro decisioni
next-token, includendo quella dopo l'ultimo decode, sono `[7042,30,198,198]`:
`The capital of France is Paris.\n\n`. Nessun EOS nella sequenza. Greedy deterministico
con ID minimo nei pareggi. Lunghezze KV pubblicate **5/6/7/8**, generazioni **1/2/3/4**.

Le **169 proiezioni** (7 per blocco più head) attraversano integralmente tutti i
coefficienti a ogni token: **13.690.208.256 coefficienti GEMV** per otto posizioni.
Norm, embedding, RoPE, attenzione causale, SiLU/SwiGLU e residui alimentano realmente
le uscite confrontate. Nessun layer saltato, peso fittizio, operatore sostituito,
quantizzazione aggiuntiva, caricamento anticipato delle matrici o calcolo su padding.
La contabilità per ogni tensore coincide esattamente col grafo richiesto.

L'head legge l'intero embedding condiviso: perciò l'unione dei pesi utilizzati
coincide con tutto il payload fisico. L'alias non aggiunge una seconda copia ai
byte unici. Le riletture, inclusa la riga embedding, sono invece conteggiate:
**3.422.756.864 byte/token**, **27.382.054.912 byte** per prefill più tre decode.
La sola SHA non viene conteggiata come utilizzo matematico.

| Quantità | Misura / limite |
|---|---:|
| Payload BF16 unico attivo | 3.422.752.768 B |
| RLIMIT_AS iniziale | 1.073.741.824 B |
| RLIMIT_AS serie principale, entrambi i modi | 268.435.456 B |
| RLIMIT_AS più stretto riuscito, entrambi i modi | **134.217.728 B** |
| RAM totale WSL | 11.483.848.704 B |
| RAM fisica host, nuova query CIM | 16.900.739.072 B |
| Limiti memory.max / memory.high, cgroup e antenato | `max` / `max` |
| Piano sync / pipeline, capacità 16 | 41.536.665 / 44.182.951 B |
| Picco RSS esterno osservato, prove a 128 MiB | ≤91.766.784 B |
| Picco virtuale esterno, sync / pipeline a 128 MiB | 102.735.872 / 111.149.056 B |
| VmSwap runtime, snapshot e campioni 20 ms | 0 B osservati |
| Memoria GPU/pinned del runtime | 0 B |

Fonti: `strict-*.json`, rispettivi `supervisor-*.json`, `resource-audit.json`,
`cgroup-audit.json`, `environment-before.json`. Il limite AS è applicato prima
della tokenizzazione e dell'apertura; distinto da RSS e dal budget delle arene.
Il cap impedisce la residenza/mappatura dell'intero checkpoint nel processo,
ma **non limita la page cache globale del sistema**.

Piano sync: pesi FP64 2.097.152 B, staging BF16 524.288 B, attivazioni 212.992 B,
due banche KV 25.165.824 B, accumulatori 520 B, output 12.582.912 B, metadata
952.977 B. Pipeline aggiunge uno slot pesi/staging e riserve richieste; nessuna
arena dell'ordine dei 3,42 GB. Metadati retained auditati circa 373 KB, entro riserva;
bootstrap parser distinto dal payload. Tokenizer, interprete, libreria C, catalogo
diagnostico e heap sono compresi nelle misure del processo, fuori dal solo piano.
Le dimensioni di tutte le arene sono controllate a ogni fase e restano costanti;
VmSize è costante tra i token delle due prove a 128 MiB, RSS cresce di circa una
pagina tra primo e ultimo snapshot, senza incremento proporzionale ai pesi.

`ru_maxrss`, letture self `/proc` e campioni esterni non coincidono sempre esattamente:
sono tutti conservati e non sommati. La tabella usa l'inviluppo esterno osservato.
Nella serie a 256 MiB la pipeline ha VmSize di regime circa 178 MB, con picco
virtuale campionato **245.350.400 B**, pur avendo arene di soli 44 MB. A 128 MiB
le prenotazioni virtuali ottenute dall'allocatore sono diverse: non è stato
necessario cambiare stack, variabili malloc o runtime. Nessuna estrapolazione a
interpreti/librerie diversi. Il campionamento non esclude transitori fra campioni.

Swap configurato 8 GiB invariato. Lo swap **globale** passa da circa 2 MiB iniziali
a circa 893 MiB nello snapshot dopo gli oracoli: non viene nascosto né attribuito
al runtime. Gli oracoli e i processi InWiMe hanno VmSwap osservato zero; ciò non
esclude attività di paging di altri processi o dell'host Windows.

## Riferimento indipendente e correttezza

[Protocollo](../results/increment8/protocol.json) scritto prima del primo forward
1.7B. Regola per **ogni elemento**: `abs(error) <= 0.002 + 0.0002*abs(reference)`.
Soglie FP32 dell'incremento 7 conservate per lo stesso grafo Llama: 24 blocchi invece
di 30, ma riduzioni più larghe. Criterio sperimentale, non un bound teorico globale;
nessun allargamento dopo le misure. Argmax e token greedy devono coincidere.

`tools/oracle8.py` non importa InWiMe e non riceve suoi intermedi o forward.
Transformers **4.57.3**, torch **2.9.1+cpu**, NumPy **2.3.5**, attenzione stock
`eager`, eval/inference mode, autocast disabilitato, seed 0, due thread CPU.
Tutti i parametri BF16 originali sono promossi esattamente a FP32 solo nell'oracolo.

Fattibilità valutata prima: soli parametri FP32 **6.845.505.536 B**; stima picco
circa 8,05 GB contro circa 8 GB disponibili inizialmente. Evitata la conversione
`.float()` di un modello BF16 residente: costruzione su `meta`, lettura safe_open
di **un tensore alla volta**, promozione, assegnazione e rilascio della mappatura
sorgente. Tensore massimo sorgente 201.326.592 B; tying applicato una sola volta.
Nessun checkpoint FP32 persistente e nessun modello BF16 completo residente.
Supervisore: stop RSS 8 GiB, stop MemAvailable 512 MiB, senza limiti di sistema nuovi.
Picco esterno circa **7,25 GB**; minimo MemAvailable >2,38 GB nei tentativi riusciti.
Oracoli terminati prima dell'inferenza InWiMe successiva; i loro consumi rimangono
separati e non entrano nelle tabelle runtime.

Il primo tentativo oracle ha esposto un difetto del **nuovo loader di verifica**:
il buffer RoPE non persistente era rimasto `meta`. Riproduzione piccola red/green,
poi inizializzazione tramite il costruttore upstream standard e controllo di tutti
i buffer; parametri, buffer, tying, logits e KV coincidono col caricamento standard
Transformers sulla fixture di regressione. Conservati log del tentativo fallito e
`oracle-loader-{red,green}.txt`. Nessun difetto runtime riscontrato o corretto.

| Confronto completo FP32 | Max assoluto | Elementi oltre soglia |
|---|---:|---:|
| Logits | 9,89073e-5 | 0 |
| Residui di tutti i blocchi | 1,32673e-2 | 0 |
| Tutti i K post-RoPE e V | 1,66954e-4 | 0 |

Il massimo assoluto hidden supera il solo `atol`; il criterio comprende il termine
relativo per ciascun elemento. Verificati **49152 logits per posizione**, residui
di tutti i 24 layer, K/V completi in layout compatibile, metriche per layer,
top-10 diagnostico e tutti gli argmax. Scenari: prefill 5 più tre decode,
forward completo 8 token, chunk 2+3. Il greedy upstream viene calcolato
indipendentemente con cache; InWiMe reinserisce i propri token e coincide.

Originale e shard, sync e pipeline, chunk/full e decode incrementale/full hanno
**tutti gli array FP64 identici nei byte**, oltre alla parità upstream FP32.
Catture InWiMe bounded FP64 grezze e catture upstream NPZ in `artifacts/increment8/`,
escluse da Git; forme, SHA, dimensioni e comandi nei report e nel manifest finale.
Il confronto offline avviene dopo la chiusura del processo limitato.

Un riferimento residente 1.7B realmente FP64 richiederebbe **13.691.011.072 B per i
soli parametri**, più del totale WSL: non tentato. **Il livello dichiarato per 1.7B
è parità end-to-end FP32 completa, non parità con un oracle realmente FP64 1.7B.**
L'intero gate dell'incremento 7 resta separato e obbligatorio: rieseguiti FP32 stock,
FP64 effettivo (sync, pipeline, shard) e FP64 stock diagnostico, usando le catture
precedenti verificate. Il diagnostico conserva esattamente **270 violazioni logits,
41 hidden, zero KV**, con argmax uguali e verdetto grezzo FAIL. Nessuna tolleranza,
cattura storica o interpretazione del FAIL modificata.

## Multi-shard, errori e regressioni

Prima eseguito l'originale. Poi `tools/repack7.py`, invariato, crea **tre shard
safetensors standard con tensori interi** e indice weight_map. SHA del sorgente,
firme dei file e digest per ogni tensore verificati; tutte le parole BF16 identiche,
nessuna conversione. Tre FD rientrano nel limite pubblico 64; ciclo di vita e cleanup
restano quelli esistenti. Output InWiMe identici all'originale. Anche l'oracolo
Transformers legge realmente questi shard: tutte le sue catture FP32 coincidono
nei byte con quelle dell'originale (`upstream-formats.json`).

Su questo checkpoint reale multi-shard, per ciascun modo, iniettati errore I/O,
lettura incompleta e cancellazione dopo 100 letture del decode, quando il KV
candidato è già stato modificato. Hash di **tutto** il vecchio banco pubblicato e
`(bank,length,generation)` invariati; sessione fail-stop, niente risultato parziale,
slot drenati, FD/thread/arene chiusi. Non si promette l'interruzione di una syscall
bloccata. Budget inferiore di un byte al piano rifiutato **prima del payload** in
ognuno dei sei casi. Prova AS 64 MiB: `MemoryError` controllato, exit 1, nessun
risultato falsamente riuscito; distinta dal `BudgetError` del piano.

**395 test PASS in 35,75 s, zero skip**: 382 precedenti e 13 nuovi. Separatamente,
5 test del profilo numerico oracle7 e 1 test del loader oracle8 PASS nel venv.
Le regressioni preesistenti
includono errori/cancellazioni durante prefill e decode; quelle nuove controllano
contabilità del tying, catture corrotte/troncate, rappresentazione, assenza oracle,
bitwise inclusi signed zero, scenari mancanti, confronti parziali e soglie alterate.
Il [manifest di conservazione](../results/increment8/preservation-after.json)
confronta i fingerprint di tutti i file consolidati prima/dopo, con la sola aggiunta
intenzionale al README distinta dai file preservati. Nessuna evidenza 1–7 riscritta.

## Prestazioni e I/O

Tre processi nuovi per modo, ordine `sync,pipeline / pipeline,sync / sync,pipeline`.
Stesso modello, prompt, decode, capacità 16, tile/read **262144 elementi**, un buffer
sync / due pipeline e un worker. Il primo smoke a 1 GiB resta fuori dalle mediane.
Nessun tuning o nuovo backend. Prima della serie gli altri carichi numerici sono
terminati; confronto offline e prove successive non sono inclusi nei timer.

| Misura, mediane salvo picchi | Sync | Pipeline |
|---|---:|---:|
| Apertura/indice, s | 0,00980 | 0,00913 |
| SHA standalone completa, s | 1,9570 | 1,9833 |
| Avvio con ulteriore integrità full, s | 2,0086 | 2,0538 |
| Prefill 5 token, s | 13,0709 | 13,3541 |
| Decode, s/token (9 osservazioni) | 2,7382 | 2,7693 |
| Decode, token/s | 0,3652 | 0,3611 |
| GEMV nel workload, s | 6,9180 | 7,2144 |
| Espansione BF16→FP64 nel workload, s | 9,1430 | 10,1707 |
| Servizio letture, s | 2,7611 | 2,8226 |
| Overlap lettura/compute, s | 0 | 0,0614 |
| Picco ru_maxrss, B | 91.234.304 | 91.533.312 |
| Picco VmHWM campionato esternamente, B | 91.500.544 | 91.779.072 |
| Byte logici inferenza / processo | 27.382.054.912 | 27.382.054.912 |
| Byte di blocco attribuiti al forward / processo | 0 | 0 |

Tempi di ciascun decode, attesa dati, proc I/O, memoria per fase e residuo aritmetico
sono nei sei JSON `measured-*`. Il residuo `wall - compute - expand - read_union +
overlap` comprende bookkeeping e coordinamento, non è un timer esclusivo del
coordinatore; in pipeline non corregge l'eventuale overlap lettura/espansione.
Servizi, attesa e wall non si sommano ingenuamente.

La SHA standalone temporizzata legge 3.422.777.952 B e si aggiunge alla SHA interna
della normale ammissione `integrity=full`, separata nei contatori startup. Questo
secondo avvio include anche parsing e prenotazioni: non è presentato come puro
tempo di hash. La doppia scansione serve a isolare una misura SHA senza cambiare
il default del runtime. Hash e startup non entrano nel forward né nei token/s.

**Cache OS calda**, favorita dalle scansioni SHA: `read_bytes=0` nel forward dei
sei campioni. I byte logici sono trasferiti dal filesystem ai buffer applicativi;
non sono byte effettivamente letti dal dispositivo. Page cache Linux, cache host
Windows e controller non controllati. Nessun drop-cache globale, DONTNEED, O_DIRECT
o prova cold controllata: **nessuna banda NVMe/NAND misurata** in questa serie.
La pipeline è leggermente più lenta sulle mediane; non emerge uno speedup e tre
coppie non sostengono una conclusione prestazionale generale. AC online; temperatura,
clock e potenza CPU non controllati o misurati continuamente. GPU mai inizializzata;
nessuna nuova misura globale VRAM dichiarata.

## Riproduzione e portata successiva

Usare nomi nuovi: strumenti ed evidenze rifiutano la sovrascrittura. Dalla radice,
checkpoint e venv già presenti; requisiti tokenizer/oracle rimangono quelli del 7.

```bash
make cpu-kernel
# Solo audit rete/spazio, senza pesi; download richiede --execute esplicito:
python3 -m tools.acquire_smollm8 checkpoints/nuovo-smollm2-1.7b --evidence /tmp/acquisizione8.json
python3 -m tools.supervise8 --output /tmp/supervisore8.json \
  .venv-oracle7/bin/python -m tools.benchmark_increment8 checkpoints/smollm2-1.7b-d4c18cc \
  --mode sync --as-mib 128 --capture artifacts/increment8/nuova-cattura \
  --output /tmp/runtime8.json
# Ripetere con --mode pipeline; per le mediane usare --as-mib 256, tre coppie alternate.
python3 -m tools.supervise8 --output /tmp/oracle8-supervisore.json \
  --rss-stop-mib 8192 --available-stop-mib 512 \
  .venv-oracle7/bin/python -m tools.oracle8 checkpoints/smollm2-1.7b-d4c18cc \
  --output artifacts/increment8/nuovo-oracle.npz
.venv-oracle7/bin/python -m tools.verify_increment8 --runtime /tmp/runtime8.json \
  --capture artifacts/increment8/nuova-cattura --oracle artifacts/increment8/nuovo-oracle.npz \
  --output /tmp/parita8.json
# Confronto esteso: --scenario extended --sequence-report /tmp/runtime8.json
# Chunk: --scenario chunks. Tutte le catture sono confrontabili con lo stesso oracle.
python3 -m tools.fault_increment8 checkpoints/smollm2-1.7b-d4c18cc-sharded --output /tmp/guasti8.json
.venv-oracle7/bin/python -m tools.check_oracle8
.venv-oracle7/bin/python -m tools.check_oracle_precision7
python3 -m pytest -q tests
python3 -m tools.report_increment8 results/increment8 --output /tmp/gate8.json
```

I comandi esatti di ogni processo limitato sono anche nei JSON dei supervisori.
Per repack e gate storico rimangono validi gli strumenti del 7, usando sempre
nuove destinazioni. Il report del gate 8 richiede tutte le sue evidenze; file o
scenari mancanti producono errore, senza skip o sostituzione dell'oracolo.

Per un **checkpoint pubblico con pesi attivi superiori alla RAM WSL** manca una
nuova acquisizione autorizzata con unione effettiva >11.483.848.704 B, architettura
compatibile verificata, spazio reale e budget per metadati/KV/attivazioni. Occorrono
forward completo, prefill/decode e misure di blocco nel nuovo regime di cache;
queste latenze calde non li predicono. Un riferimento FP32 completo potrebbe non
entrare: servirà un confronto indipendente organizzato per blocchi/streaming,
oppure dichiarare precisamente una verifica più limitata. Nessuna di queste
attività è iniziata qui. La prova sintetica del 5 oltre WSL resta separata.

Contesti lunghi, famiglie diverse, precisioni/backend futuri, GPU, distribuzione,
quantizzazione, speculation, MoE predittivo e Harness restano fuori dal cambio.
Nessun commit, staging, push o incremento 9.
