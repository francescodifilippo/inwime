# Incremento 9 — inferenza reale oltre il budget di memoria e oltre la RAM WSL

15 novembre 2026. Base `main`, `68ba9f3adf3cffb89b18b19525644fa0d7c6d806`,
repository `francescodifilippo/inwime` (directory locale `ooc-inference`).
Working tree iniziale pulito. Nessun commit, staging o push.

**Qwen/Qwen2.5-7B originale BF16 esegue realmente prefill di cinque token e tre
decode con 14.141.295.616 byte unici di pesi attivi, contro `MemTotal` WSL di
11.483.848.704 byte misurato all'avvio della stessa prova**, dentro un cgroup v2
transitorio con `memory.max` 2 GiB e `memory.swap.max` 0. Parità completa con un
oracolo Transformers indipendente a blocchi: standard FP32 e FP64 puro superati,
diagnostico FP64 standard FAIL grezzo conservato. SmolLM2-1.7B esegue
correttamente sotto lo stesso limite di 2 GiB dopo evizione mirata della cache.
[Gate A–H](../results/increment9/summary.json): tutte le condizioni **PASS**.
**Non dimostra** il superamento della RAM fisica host (16.900.739.072 byte) né
un'inferenza pratica: circa 13,3 s per token in decode.

## 1. Stato iniziale e audit

Letti README, brief, decisione architetturale, rapporti 5–8, revisione 8, audit
Strata e watchlist; esaminati adapter Llama, sorgente safetensors, coordinatori,
GEMV, kernel C e strumenti 7/8. Baseline **408 test PASS**.

[Audit iniziale](../results/increment9/environment-before.json), sola lettura:

| Grandezza | Valore |
|---|---:|
| `MemTotal` WSL (RAM della VM, non dell'host) | 11.483.848.704 B |
| `MemAvailable` WSL all'audit | 8.773.832.704 B |
| RAM fisica host, CIM | 16.900.739.072 B |
| RAM libera host all'audit | circa 0,5 GB |
| Swap WSL `/dev/sdc`, invariato | 8 GiB, circa 717 MiB usati da altri processi |
| ext4 virtuale `/dev/sdd`, libero dichiarato | 908.397.023.232 B |
| Volume Windows C: con il VHDX (BasePath su C:) | 151.969.673.216 B liberi |
| `.wslconfig` (solo letto) | `memory=11GB`, `swap=8GB`, `sparseVhd=true` |

Lo spazio libero dell'ext4 virtuale **non** coincide con quello reale: il VHDX
cresce su C:. Le decisioni hanno usato C:. Il disco fisico è un Samsung SSD 990
EVO 2TB; il Sabrent USB non è stato usato.

cgroup v2 montato; il root delega ai figli `cpu memory pids`, **non `io`**:
`io.stat` per cgroup non è disponibile. `user@1000.service` è delegato
all'utente con il controller `memory`. Nessun limite di memoria preesistente
nella catena (`max`). Nessun checkpoint Qwen o cache Hugging Face pertinente.

## 2. Protocollo preregistrato

[Il protocollo](../results/increment9/protocol.json) è stato scritto prima di
qualsiasi download, forward o oracolo. Contiene le soglie dell'incremento 7,
invariate: FP32 `2e-3 + 2e-4*|ref|`, FP64 `2e-5 + 2e-6*|ref|`, per ogni elemento.

**Un solo emendamento, prima di ogni misura di fase A/B.** Una prova del
meccanismo su un file casuale di 300 MB, con scope da 128 MiB
([cgroup-probe-128mib.json](../results/increment9/cgroup-probe-128mib.json)),
ha mostrato `memory.peak == memory.max` esattamente: la page cache riempie il
gruppo fino al limite e viene recuperata lì, senza OOM. Il criterio
`memory.peak < memory.max` avrebbe reso un FAIL spurio un comportamento corretto
del kernel. È stato sostituito da `memory.peak <= memory.max`, con `oom=0`,
`oom_kill=0`, swap 0 e la memoria anonima riportata separatamente.

## 3. Meccanismo cgroup e attribuzione

`tools/cgroup9.py` crea uno scope transitorio con
`systemd-run --user --scope --collect -p MemoryMax=2147483648 -p MemorySwapMax=0`.
Usa la sola delega utente esistente. Non usa `sudo`, non crea unità persistenti
e non tocca `.wslconfig`, swap, sysctl o altri cgroup. Lo scope scompare
all'uscita. Il supervisore resta nel proprio cgroup e campiona ogni 50 ms
`memory.current`, `memory.stat`, `memory.events`, swap, `/proc/PID/status`,
`/proc/PID/io` di ogni PID dello scope e `MemAvailable` globale. Se
`MemAvailable < 768 MiB`, esegue `cgroup.kill` del solo scope.

Un finalizzatore `/bin/sh` nello scope registra `memory.peak` e gli eventi del
kernel dopo l'uscita del worker. Il worker verifica da dentro, **prima** di
tokenizer, catalogo o SHA, l'appartenenza allo scope atteso (`cgroup.procs`) e i
limiti osservati: in tutte le prove `verified_before_opening_weights=true`.

Evizione: `POSIX_FADV_DONTNEED` sui soli file della prova, verificata con
`fincore` (mincore). Nessun `drop_caches`. La residenza cala da 3.422.781.440 a
**0 byte** per SmolLM2 e da 81.920 a 0 per ciascuno shard Qwen. Il contatore
globale `/proc/vmstat oom_kill` è registrato prima e dopo ogni prova.

## 4. Fase A — SmolLM2-1.7B sotto 2 GiB

Checkpoint già validato (`d4c18cc`, SHA ufficiale ricontrollato dal runtime e a
fine incremento). Stesso input dell'incremento 8: `The capital of France is`,
prefill di 5 token e 3 decode, sync, tile 262.144, `RLIMIT_AS` 256 MiB.

| Misura | W: cache calda altrove | P: cache rimossa, sotto pressione |
|---|---:|---:|
| Residenza del file al lancio | 3.422.781.440 B | 0 B |
| `memory.peak` dello scope | 112.832.512 B | **2.147.483.648 B** (= limite) |
| Eventi `max` (reclaim al limite) | 0 | 21.412 |
| `oom` / `oom_kill` / swap del gruppo | 0 / 0 / 0 | 0 / 0 / 0 |
| Picco campionato file / anon nel gruppo | 39,5 MB / 71,5 MB | 2.074 MB / 71,5 MB |
| Letture di blocco: avvio SHA / inferenza | 24.576 B / 0 B | 3.422.781.440 B / 18.567.114.752 B |
| Byte logici d'inferenza | 27.382.054.912 B | 27.382.054.912 B |
| Letture disco fisico host, C:, intervallo (globale) | 0,14 GB | 22,77 GB |
| Prefill 5 token / decode mediano | 12,97 s / 2,684 s | 17,18 s / 3,340 s |
| Picco RSS / VmPeak del worker | 91,2 MB / 102,8 MB | 91,0 MB / 102,8 MB |

**W mostra l'errore di attribuzione da evitare.** Le pagine già in cache e
addebitate al cgroup del supervisore non vengono addebitate allo scope. Il
picco resta 113 MB e non avviene alcuna lettura di blocco. Il file di 20–40 MB
nel gruppo sono le catture scritte dal worker. W non dimostra lo streaming.

**P è la prova.** Ogni pagina viene letta sotto lo scope. La cache del gruppo
sale fino al limite e il kernel la recupera 21.412 volte, senza OOM né swap. Il
decode rilegge circa 2,31 GB di blocchi per token su 3,42 GB logici: circa
1,1 GB per token viene servito dalla cache residua del gruppo. Il costo
osservato è circa 0,66 s per token rispetto a W. Dopo l'uscita restano
residenti circa 2,06 GB del file, riparentati al cgroup padre.

Il contatore fisico host conta 22,77 GB contro 21,99 GB di blocchi Linux
nell'intervallo. La cache Windows non ha servito queste letture: l'host aveva
circa 0,5 GB liberi. Il contatore è globale al disco e include altra attività host.

Numerica: logits, residui e KV di W e P hanno **SHA identici** a quelli
accettati nell'incremento 8. La parità completa con l'oracolo FP32
dell'incremento 8 passa
([P](../results/increment9/phaseA-P-parity-inc8.json),
[W](../results/increment9/phaseA-W-parity-inc8.json)). Generazione
`[7042,30,198,198]`: `The capital of France is Paris.\n\n`. FD, arene e scope
rilasciati.

## 5. Fase B — selezione e acquisizione

[Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/tree/d149729398750b98c0af14eb82c78cfe92750796),
revisione immutabile `d149729398750b98c0af14eb82c78cfe92750796`, Apache-2.0,
non gated. Il preflight ([qwen-preflight.json](../results/increment9/qwen-preflight.json))
ha letto API, config, index e i **quattro header safetensors via HTTP Range**,
senza scaricare pesi. Risultato: 339 tensori, **tutti BF16**,
7.615.616.512 parametri, payload 15.231.233.024 B uguale a `total_size`. La
`weight_map` coincide con gli header. Q/K/V hanno bias, O/MLP no, la head non è
condivisa, 28 layer, hidden 3584, FFN 18.944, 28/4 head, dim 128, theta 1e6,
eps 1e-6, vocabolario 152.064. Il limite inferiore dei byte attivi è
14.141.238.272 B, cioè il payload meno l'embedding. Il criterio era già
soddisfatto prima del download.

Download di 15.242.803.171 B con spazio reale controllato prima e ogni 256 MiB.
Margine di 8 GiB; minimo osservato su C: 136,78 GB liberi. SHA256 calcolato in
streaming e confrontato con il digest LFS ufficiale prima del rename in posto
dei `.part`. File piccoli verificati con l'ID Git blob ufficiale. **Una sola
copia**, nessuna cache HF, nessuna conversione, nessun repack: gli shard
originali sono usati direttamente. Descriptor `safetensors.json` esterno con i
digest. Re-hash completo a fine incremento:
[integrity-after.json](../results/increment9/integrity-after.json), tutti
corrispondenti.

## 6. Adapter Qwen2

`inwime_runtime/qwen2.py` (circa 150 righe) aggiunge `Qwen2Catalog` e
`Qwen2Runtime`. La sorgente safetensors resta indipendente dalla famiglia e non
è modificata. L'adapter Llama, i coordinatori, la GEMV e il kernel sono invariati.
Riusa dalla sessione Llama RoPE split-half, attenzione causale GQA, KV
transazionale e prefill fail-stop. Dalla base riusa embedding, RMSNorm, SwiGLU
e acquisizione a tile. Aggiunge soltanto:

- mapping esplicito dei 12 tensori per layer più embedding, norm e `lm_head`;
- bias Q/K/V letti a tile e sommati **prima** della RoPE, come upstream; K è
  conservato dopo la RoPE;
- `lm_head.weight` fisico distinto dall'embedding.

Rifiuta, invece di indovinare: altra architettura, embedding condivisi, sliding
window, mRoPE, `rope_scaling`, `layer_types` non full, geometria GQA/RoPE
incoerente, tensori mancanti o in più (inclusi un bias O e una head mancante),
forme errate e dtype non BF16. Nessun parametro fittizio, nessun layer saltato.
Il tokenizer è `tokenizer.json` ufficiale tramite `tokenizers`, identico ad
AutoTokenizer su quattro testi, inclusi Unicode e special token. BOS non viene
aggiunto (`add_bos_token=false`); EOS è 151643.

Piano di memoria sync, capacità 16: **43.658.344 B**. Pesi FP64 2.097.152,
staging 524.288, attivazioni 446.464, KV 7.340.032 (due banche), output
32.309.248, metadata 940.128, accumulatori 1.032. Pipeline: 46.304.630 B.

## 7. Forward oltre la RAM WSL

Scope da 2 GiB (fase A superata), shard rimossi dalla cache prima del lancio,
`RLIMIT_AS` 256 MiB, budget esplicito uguale al piano, integrità `full`.
`The capital of France is` → `[785,6722,315,9625,374]`. Prefill di 5 token, poi
tre decode reinseriscono `[12095],[13],[1084]`. Decisioni greedy
`[12095,13,1084,374]`: **`The capital of France is Paris. It is`**. Nessun EOS.
Lunghezze KV pubblicate 5/6/7/8, generazioni 1/2/3/4, prefisso KV preservato a
ogni passo.

| Quantità | Valore |
|---|---:|
| `MemTotal` all'avvio della prova | 11.483.848.704 B |
| **Byte BF16 unici attivamente usati** | **14.141.295.616 B** (+2.657.446.912 B) |
| Ricalcolo indipendente dagli header ufficiali | 14.141.295.616 B, identico |
| Byte logici per token (riletture incluse) | 14.141.245.440 B |
| Byte logici d'inferenza, 8 posizioni | 113.129.963.520 B |
| Letture di blocco d'inferenza del worker | 112.420.581.376 B (99,4%) |
| Letture di blocco all'avvio (SHA integrale 15,23 GB) | 15.231.524.864 B |
| Letture disco fisico host C:, intervallo (globale) | 130.314.880.512 B |
| `memory.peak` scope / eventi `max` / OOM / swap | 2.147.483.648 B / 63.980 / 0 / 0 |
| Picco anon / file campionato nel gruppo | 105,0 MB / 2.040,5 MB |
| Picco RSS / VmPeak / VmSwap del worker | 124.768.256 B / 142.110.720 B / 0 |
| `MemAvailable` globale minimo | 8.610.328.576 B |
| `oom_kill` globale prima/dopo | 0 / 0 |

L'unione dei byte attivi comprende tutte le 197 matrici (7 per layer più head),
lette integralmente a ogni token: **56.562.286.592 coefficienti GEMV**. Comprende
anche bias e norm, e le sole **8 righe** distinte di embedding effettivamente
consultate (57.344 B). L'embedding non letto (circa 1,09 GB) **non** è contato.
La contabilità di ogni tensore coincide esattamente con il grafo. Nessun layer
saltato, peso sostituito, quantizzazione, copia FP32/FP64 del modello o cache
applicativa dei pesi.

Il superamento della RAM WSL non dipende dal limite del processo o del cgroup:
è una proprietà dei coefficienti realmente consumati. Lo scope aggiunge un
vincolo più forte: al più 2 GiB di page cache attribuibile all'esperimento.

## 8. Verifica numerica indipendente

`tools/oracle9.py` non importa InWiMe e riceve dal runtime soltanto la sequenza
di token. Costruisce `Qwen2ForCausalLM` sul device `meta`. Hook di forward
materializzano **un modulo di primo livello alla volta** (embedding, un layer,
norm finale, head): leggono i tensori BF16 originali dagli shard, li promuovono
in RAM e li rilasciano dopo l'uso. Il forward è quello upstream non modificato:
maschera causale, rotary, `DynamicCache`, GQA, eager attention, head. Il
contratto FP32 e quello FP64 puro usano quattro chiamate con cache upstream
(5+1+1+1, come i decode del runtime). Ogni tensore viene caricato una volta per
chiamata (verificato). Il diagnostico FP64 standard usa una chiamata unica.
Transformers 4.57.3, torch 2.9.1+cpu, due thread, seed 0. Nessun checkpoint
convertito persistente.

Il greedy upstream è indipendente: per causalità, l'argmax alla posizione `i`
della sequenza completa è la scelta upstream dopo il prefisso `i`. Coincide con
tutta la continuazione. Il profilo FP64 puro (`tools/oracle_precision9.py`)
applica a Qwen2 la stessa politica dell'incremento 7: RMSNorm, RoPE e softmax in
FP64 nel solo processo oracolo.

Prima del modello grande, [controlli piccoli](../results/increment9/oracle9-checks.txt)
su una fixture Qwen2 sintetica: l'oracolo a blocchi coincide **bit per bit** con
`from_pretrained` standard (FP32 e FP64). Il profilo FP64 puro non emette output
floating a precisione inferiore e ripristina lo stato stock. InWiMe coincide con
upstream alle soglie FP64 e FP32. Un primo esito rosso di quest'ultimo controllo
era un difetto del **test**: le viste `np.frombuffer` sulle arene erano
sovrascritte dalla chiamata successiva. È stato corretto copiando gli array; il
runtime era corretto (errore singolo prefill circa 1e-15). Il log del tentativo
rosso non è stato conservato.

| Confronto completo, 8 posizioni | Max abs logits | Max abs residui | Max abs K/V | Esito |
|---|---:|---:|---:|---|
| Transformers standard FP32 | 1,657e-4 | 3,953e-3 | 9,13e-5 | **PASS**, 0 violazioni |
| PyTorch FP64 puro (profilo 7) | 2,549e-12 | 6,298e-11 | 5,76e-13 | **PASS**, 0 violazioni |
| Standard FP64, diagnostico | 1,040e-5 | 8,420e-4 | 1,33e-5 | **FAIL grezzo**: 31 residui, 0 logits, 0 KV |

Copertura: **tutti i 152.064 logits** di ognuna delle 8 posizioni (1.216.512
valori), residui di tutti i 28 layer (802.816), K post-RoPE e V di tutti i layer
dopo ogni fase (745.472), argmax, top-10 e greedy. È un confronto end-to-end
completo, non parziale. Le 31 violazioni diagnostiche sono tutte nell'ultimo
layer (indice 27), dove i residui hanno ampiezza grande. Gli argmax restano
uguali. Il comportamento è coerente con i cast FP32 interni di RMSNorm, RoPE e
softmax upstream dimostrati nell'incremento 7. Il verdetto non viene riscritto
e le soglie non sono state cambiate.

Risorse degli oracoli, in processi separati dopo le prove runtime:

| Oracolo | Picco RSS | Modulo massimo residente | Letture BF16 sorgente | MemAvailable min | Tempo |
|---|---:|---:|---:|---:|---:|
| FP32 | 3,64 GB | 2,18 GB (head/embedding) | 60,9 GB | 6,46 GB | 251 s |
| FP64 puro | 5,83 GB | 4,36 GB | 60,9 GB | 4,25 GB | 439 s |
| FP64 standard | 5,83 GB | 4,36 GB | 15,2 GB | 4,34 GB | 155 s |

I parametri FP32 completi richiederebbero 30,5 GB, quelli FP64 60,9 GB: oltre
la RAM WSL, per questo l'oracolo lavora a blocchi. VmSwap campionato degli
oracoli: 0.

## 9. Prestazioni e I/O

Una coppia sync/pipeline, processi nuovi, stesse condizioni: shard rimossi dalla
cache, scope 2 GiB, stesso input, tile 262.144, un worker. Gli output delle due
modalità sono **identici nei byte** (digest di logits, residui e KV).

| Mediane decode (3 osservazioni), salvo indicato | Sync | Pipeline |
|---|---:|---:|
| Avvio con SHA integrale 15,23 GB | 12,37 s | 12,35 s |
| di cui servizio letture integrità | 4,51 s | 4,25 s |
| Prefill 5 token | 66,91 s | 72,13 s |
| Decode, s/token | **13,27** | 13,83 |
| Espansione BF16→FP64 | 4,75 s | 5,41 s |
| GEMV | 3,65 s | 3,87 s |
| Regioni compute | 4,17 s | 4,46 s |
| Servizio letture | 3,76 s | 4,54 s |
| Attesa dati | 3,95 s | 2,90 s |
| Overlap lettura/compute | 0 | 0,32 s |
| Residuo aritmetico | 0,49 s | −0,31 s |

Ogni decode legge circa **14,04 GB di blocchi** in circa 3,76 s di servizio
letture: circa 3,7 GB/s apparenti tramite page cache, VHDX ext4 e NTFS. Il
contatore fisico host conferma che i byte provengono dal disco fisico. Questa
non è la banda isolata NAND/NVMe: comprende readahead, copia kernel e
virtualizzazione, e il contatore host è globale. Nessun `O_DIRECT`. La pipeline
qui non accelera: l'overlap è piccolo e il costo di espansione cresce. Una
coppia non sostiene una conclusione generale. Il residuo negativo in pipeline
deriva dall'overlap lettura/espansione non corretto, come documentato
nell'incremento 7. AC1 online=1 a fine prove. Temperatura, clock e potenza
non sono controllati né misurati. GPU mai inizializzata.

Nella fase A la stessa infrastruttura misura il costo dello streaming reale
rispetto alla cache calda (sezione 4). Nessun valore caldo è stato estrapolato
a banda NVMe.

## 10. Errori, test negativi e regressioni

Sul **checkpoint reale Qwen2.5-7B**, in uno scope da 2 GiB, sono stati iniettati
in sync e pipeline errore I/O, lettura incompleta e cancellazione alla lettura
100 di un decode, dopo un prefill di un token
([faults.json](../results/increment9/faults.json)). Il KV candidato era già
modificato. Il vecchio banco e il descriptor `(bank,length,generation)` restano
intatti. La sessione è fail-stop, senza ripresa. Slot drenati, FD e arene
chiusi. Il budget inferiore di un byte al piano viene rifiutato **prima** di
leggere il payload. Questa sonda usa `integrity='metadata'` per evitare sei SHA
da 15 GB; l'integrità è verificata dalle prove principali e dal re-hash finale.

**OOM controllato.** SmolLM2-1.7B in uno scope da 32 MiB viene ucciso dal kernel
(`oom 1`, `oom_kill 1`, exit 137). Non produce alcun rapporto di successo: è un
FAIL atteso. Il primo tentativo ha rivelato un **difetto del supervisore**: con
`OOMPolicy=stop`, default di systemd, l'OOM ferma l'intero scope e anche il
finalizzatore, perdendo i contatori finali. La correzione minima è
`-p OOMPolicy=continue`; la prova ripetuta registra i contatori. Le prove
riuscite precedenti non hanno avuto OOM e i loro finalizzatori hanno scritto
normalmente. Il contatore globale `oom_kill` è a 2 per questi due test
deliberati, entrambi attribuiti. SmolLM2-135M, lanciato prima come negativo in
48 MiB, è invece riuscito: l'evidenza è rinominata `scope48mib-135m-*`.

**484 test PASS in 36,39 s, zero skip**: 408 preesistenti e 76 nuovi. I 51 test
dell'adapter usano una fixture Qwen2 stdlib e un riferimento indipendente scritto
dalle equazioni (`math.fsum`, decodifica BF16 via `struct`). Coprono parità
incrementale e completa in sync/pipeline, scalare/C, chunk/shard, sensibilità a
bias e head, config/tensori/forme/dtype rifiutati, SHA errata, corruzione,
troncamento, token invalidi, transazioni con fault in prefill/decode, budget e
copertura delle letture. I 25 test degli strumenti coprono l'attribuzione dei
byte unici con head condivisa o separata, il parsing del finalizzatore, il gate
senza evidenze (NOT PROVEN) e i rifiuti del verificatore: soglie alterate,
contratto, profilo, revisione, sequenza, digest, un solo passo, valori e greedy
divergenti. Nove mutazioni di copie delle evidenze reali (OOM, parità mancante o
fallita, byte unici, attribuzione calda, evizione, picco, decode mancante, pytest)
portano il gate a FAIL o NOT PROVEN.

Il gate 8 rafforzato dalla revisione è stato rieseguito: PASS. Rieseguiti anche
il gate numerico 7 risolto (PASS, con il FAIL diagnostico storico conservato)
e i controlli oracle 7/8 (OK). La [conservazione](../results/increment9/preservation.json)
rispetto alla base mostra un solo file tracciato modificato, `README.md`, in
modo intenzionale; nessuna cancellazione. Runtime, kernel, adapter Llama e
sorgente non sono cambiati. Nessuna evidenza storica riscritta. Nessun peso o
cattura in Git.

## 11. Gate

| Condizione | Esito | Base |
|---|---|---|
| A — Integrità | PASS | SHA LFS in streaming, SHA integrale del runtime per entrambi i modelli, re-hash finale |
| B — Memoria cgroup | PASS | P: scope 2 GiB verificato dall'interno, picco ≤ limite, 0 OOM/swap, letture reali, SHA e parità uguali all'8 |
| C — I/O | PASS | W/P/B con contatori processo, cgroup e host; attribuzione W vs P coerente |
| D — Oltre RAM WSL | PASS | 14.141.295.616 > 11.483.848.704, ricalcolato dagli header |
| E — Forward reale | PASS | 5+3 token, 28 layer, copertura esatta, rilascio |
| F — Correttezza | PASS | FP32 e FP64 puro completi; diagnostico FP64 standard FAIL grezzo riportato |
| G — Risorse | PASS | tutte le cinque prove supervisionate: scope 2 GiB, 0 OOM/swap, `oom_kill` globale invariato; VmSwap 0 |
| H — Regressioni | PASS | 484 test, gate 7/8, conservazione |

`tools/report_increment9.py` ricalcola ogni verdetto dalle evidenze grezze. Una
evidenza mancante produce NOT PROVEN, mai PASS.

## 12. Limiti e passi successivi

- **RAM fisica host non superata.** 14,14 GB attivi restano sotto i 16,9 GB
  dell'host. Il caching Windows non ha servito le letture (contatore fisico),
  ma la prova riguarda RAM WSL e cgroup.
- **Non pratico per modelli enormi.** Il decode costa circa 13,3 s/token:
  espansione FP64 e regioni compute con GEMV a thread singolo (circa 8,9 s)
  più letture (circa 3,8 s), con riletture integrali a ogni token. Per un modello da 1,5 TB, a
  parità di banda e calcolo, ogni token denso richiederebbe ordini di grandezza
  in più. Servono kernel multi-thread o nativi BF16, batching del prefill (oggi
  seriale per token, quindi 5 letture complete), MoE con letture selettive e
  probabilmente la GPU.
- Contesto massimo provato 8 posizioni, capacità 16. Una sola famiglia nuova,
  senza RoPE scaling o sliding window.
- `io.stat` non disponibile senza delega del controller `io`. Le letture fisiche
  sono osservate solo con il contatore globale Windows. Nessun `O_DIRECT` e
  nessuna misura NAND isolata.
- Una sola coppia sync/pipeline; temperatura e potenza non controllate.
- I guasti reali Qwen usano `integrity='metadata'` per costo.
- Il precedente test OOM con `OOMPolicy=stop` ha perso i contatori finali
  (corretto).

Fuori perimetro e non avviati: GPU, quantizzazione, MoE, speculative decoding,
distribuzione, Harness, incremento 10. Nessun commit o push.
