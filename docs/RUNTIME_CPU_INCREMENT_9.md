# Incremento 9 — inferenza reale oltre il budget di memoria e oltre la RAM WSL

> **Stato attuale (17 novembre 2026).** Il follow-up della seconda revisione è
> pubblicato nel commit `5eb25da` («incremento 9 review»); la revisione
> conclusiva lo ha giudicato *READY WITH DOCUMENTED LIMITATIONS*. M9-A e M9-B
> sono chiusi con limitazioni documentate; le Issues #2 e #3 sono chiuse, le
> attività residue sono nelle Issues #17, #18 e #19. Le frasi che seguono su
> commit, staging e «non committato» descrivono lo stato **al momento della
> stesura** e restano come testo storico. L'hardening del gate successivo alla
> chiusura (D-1…D-6) è descritto in
> [results/increment9/post-closure/README.md](../results/increment9/post-closure/README.md).

15 novembre 2026. Base `main`, `68ba9f3adf3cffb89b18b19525644fa0d7c6d806`,
repository `francescodifilippo/inwime` (directory locale `ooc-inference`).
Working tree iniziale pulito. Nessun commit, staging o push durante l'acquisizione.
Rapporto aggiornato il 16 novembre 2026 dalla remediation della
[revisione indipendente](REVIEW_INCREMENT_9.md): numeri e formulazioni corretti,
gate reso fail-closed sulle mutazioni provate, nessuna misura ripetuta (sezione
13); aggiornato ancora dal follow-up della seconda revisione (sezione 13.4).

**Qwen/Qwen2.5-7B originale BF16 esegue realmente prefill di cinque token e tre
decode con 14.141.295.616 byte unici di pesi attivi, contro `MemTotal` WSL di
11.483.848.704 byte misurato all'avvio della stessa prova**, dentro un cgroup v2
transitorio con `memory.max` 2 GiB e `memory.swap.max` 0. Parità completa con un
oracolo Transformers indipendente a blocchi: standard FP32 e FP64 puro superati,
diagnostico FP64 standard FAIL grezzo conservato. SmolLM2-1.7B esegue
correttamente sotto lo stesso limite di 2 GiB dopo evizione mirata della cache.
[Gate A–H](../results/increment9/summary.json), ricalcolato dopo la remediation
con le regole più severe: tutte le condizioni **PASS**.
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
| `MemAvailable` WSL all'audit | 8.841.621.504 B |
| RAM fisica host, CIM | 16.900.739.072 B |
| RAM libera host all'audit | circa 0,5 GB |
| Swap WSL `/dev/sdc`, invariato | 8 GiB, circa 717 MiB usati da altri processi |
| ext4 virtuale `/dev/sdd`, libero dichiarato | 908.396.462.080 B |
| Volume Windows C: con il VHDX (BasePath su C:) | 151.969.796.096 B liberi |
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

Limiti di questa parte, rilevati dalla revisione. Il criterio emendato
`memory.peak <= memory.max` è vero per costruzione del kernel: da solo non
discrimina. La parte informativa dell'emendamento (anon+kernel «ben sotto il
limite») non era verificata dal gate. La
[remediation](../results/increment9/protocol-amendment-remediation.json) aggiunge
una soglia esplicita, scelta **dopo** le misure e dichiarata come inasprimento:
picco campionato anon+kernel ≤ 268.435.456 B e VmHWM del worker ≤ 268.435.456 B
per ogni prova supervisionata (massimi misurati: 110.895.104 B e 125.739.008 B).
Il contenuto di `protocol.json` precedente all'emendamento delle 14:56:31Z non è
verificabile: l'emendamento fu scritto nello stesso file e in Git esiste solo il
testo emendato. Da allora il file è invariato; il gate ne fissa lo SHA256.

Unità transitorie non riportate nel rapporto originale, ricostruite dal journal
utente locale (non versionato): `inwime9-probe` alle 15:49:01 (una shell che
stampa i propri limiti cgroup, prima di audit e protocollo), `inwime9-pidprobe`
alle 15:55:03 (`sleep 2`), un primo lancio della sonda da 128 MiB alle 15:56:03
e **un pilota `inwime9-smoke135` alle 15:57:32**, dopo l'emendamento e prima
della fase A. Secondo la revisione era SmolLM2-135M in uno scope da 256 MiB,
prova di funzionamento del worker sotto `cgroup9`; il journal conferma unità,
orario, durata di 4,82 s e picco di 256M. I suoi file stavano in una directory
temporanea di sessione che non esiste più: è uno smoke test **non conservato**
e non entra in alcuna condizione del gate.

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

Checkpoint già validato (`d4c18cc`, SHA ufficiale ricontrollato dal runtime
all'avvio di ogni prova e dal re-hash di `integrity-after.json`). Stesso input dell'incremento 8: `The capital of France is`,
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
Margine di 8 GiB; minimo osservato su C: 136,78 GB liberi. File piccoli
verificati con l'ID Git blob ufficiale. **Una sola copia**, nessuna cache HF,
nessuna conversione, nessun repack: gli shard originali sono usati direttamente.
Descriptor `safetensors.json` esterno con i digest.

L'integrità degli shard va letta su due piani distinti.

- **Verifica in acquisizione (limite dichiarato).** Il codice calcolava lo SHA256
  in streaming e sollevava un errore, senza rinominare il `.part`, se differiva
  dal digest LFS ufficiale. In questa esecuzione la destinazione era vuota
  (`existing: []`) e i quattro shard sono stati completati. Però il digest
  **calcolato non fu registrato**: il campo `sha256` di `qwen-acquisition.json` è
  il valore ufficiale copiato. Quel file quindi non è una prova indipendente e il
  gate non lo usa come tale. Il file storico non è stato modificato.
- **Integrità successiva (la prova).** SHA256 integrale di ogni shard calcolato
  dal runtime all'avvio di entrambe le prove di fase B (15.231.271.888 B letti) e
  re-hash completo registrato in
  [integrity-after.json](../results/increment9/integrity-after.json) alle 16:45,
  dopo forward e oracoli ma **prima** delle sonde di guasto e dei test OOM: tutti
  uguali ai digest LFS ufficiali del preflight. Un nuovo re-hash completo,
  ultima lettura dei checkpoint della remediation (16 novembre, 00:51–00:53 UTC,
  seguito solo dalla rigenerazione di conservazione, test e gate), è in
  [integrity-final.json](../results/increment9/integrity-final.json): tutti
  uguali, index incluso.

Per le acquisizioni future `tools/acquire_qwen9.py` registra il digest calcolato
accanto a quello ufficiale, ricalcola lo SHA di un file già presente prima di
accettarlo e verifica un `.part` completo senza richieste HTTP.

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
`RLIMIT_AS` 256 MiB, budget predefinito del runtime uguale al piano
(`budget_bytes` = `arena_plan_bytes` = 43.658.344 B; il benchmark non passa
`budget=`), integrità `full`.
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
lette integralmente a ogni token: **56.562.286.592 coefficienti GEMV in totale
sulle 8 posizioni**, cioè 7.070.285.824 per token. Comprende
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

Limite di indipendenza: le funzioni del profilo FP64 puro sono scritte dallo
stesso autore del runtime. L'indipendenza piena viene dai contratti upstream
**non modificati**: FP32 standard PASS e FP64 standard, il cui errore massimo
sui logits è 1,04e-5 con argmax uguali. Il confronto FP64 puro misura l'accordo
con il grafo upstream una volta rimossi i cast FP32 interni, non una seconda
implementazione indipendente di quelle tre operazioni.

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
oracoli: 0. Lo swap **globale** in uso è però salito da 717.316.096 B (684,1 MiB)
all'avvio della prova sync a 797.814.784 B (760,9 MiB) all'avvio della prova
pipeline: +76,8 MiB nella finestra dei tre oracoli, a carico di processi esterni
ai cgroup dell'esperimento. Gli scope hanno `memory.swap.max` 0 e picco swap 0.

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
nell'incremento 7. Lo stato di alimentazione non è stato registrato in alcun
file di evidenza: l'affermazione «AC1 online=1» del rapporto originale è stata
rimossa. Temperatura, clock e potenza non sono controllati né misurati. GPU mai
inizializzata.

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
da 15 GB; l'integrità è verificata dalle prove principali e dai re-hash
registrati (`integrity-after.json`, precedente a queste sonde, e
`integrity-final.json`, successivo).

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

All'acquisizione: **484 test PASS in 36,39 s, zero skip**: 408 preesistenti e 76
nuovi. Dopo la remediation: **613 test PASS, zero skip** (sezione 13). I 51 test
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
rispetto alla base mostra un solo file preesistente modificato, `README.md`, in
modo intenzionale; nessuna cancellazione. Runtime, kernel, adapter Llama e
sorgente non sono cambiati. Nessuna evidenza storica riscritta. Nessun peso o
cattura in Git.

## 11. Gate

| Condizione | Esito | Base |
|---|---|---|
| A — Integrità | PASS | SHA integrale del runtime (entrambe le prove B e fase A) e re-hash registrati, confrontati file per file con i digest LFS ufficiali del preflight; la verifica in acquisizione è dichiarata come limite |
| B — Memoria cgroup | PASS | P: evizione verificata, scope 2 GiB verificato dall'interno e legato al supervisore, picco ≤ limite, anon+kernel ≤ 256 MiB, 0 OOM/swap, letture reali, digest e parità ricostruita uguali all'8 |
| C — I/O | PASS | W/P/B con contatori processo, cgroup e host; attribuzione W vs P; fase B sync e pipeline: evizione ≤ 1% per shard e letture di blocco ≥ 90% dei byte logici |
| D — Oltre RAM WSL | PASS | 14.141.295.616 > 11.483.848.704, ricalcolato dagli header ufficiali, su una prova evitta e realmente letta |
| E — Forward reale | PASS | 5+3 token, 28 layer, contatori dei 339 tensori uguali alle forme ufficiali × 8 posizioni, prefisso KV e generazioni, rilascio, pipeline identica nei byte |
| F — Correttezza | PASS | FP32 e FP64 puro ricostruiti dai confronti e legati a runtime, revisione, sequenza e oracolo; diagnostico FP64 standard obbligatorio, **FAIL grezzo riportato** |
| G — Risorse | PASS | tutte le cinque prove supervisionate: scope 2 GiB, 0 OOM/swap, anon+kernel e VmHWM sotto soglia, `oom_kill` globale invariato; oracoli con VmSwap 0 |
| H — Regressioni | PASS | almeno 613 test (651 dopo il follow-up) con parsing rigoroso, gate 7/8 ricalcolati dai loro strumenti, conservazione ricalcolata da git |

`tools/report_increment9.py` ricalcola ogni verdetto dalle evidenze grezze. Un
file o un campo mancante produce NOT PROVEN, un fatto contraddetto FAIL. «Mai
PASS» **vale per le mutazioni provate**: le 80 della remediation e i 13
aggiramenti della seconda revisione (sezione 13). Non è una garanzia assoluta:
la seconda revisione ha trovato eccezioni, ora corrette o dichiarate come
limiti (sezione 13.4). Due precondizioni valgono per tutte le condizioni:
protocollo approvato e sanificazione verificata; se falliscono cambiano solo i
PASS, un FAIL resta FAIL.

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
distribuzione, Harness, incremento 10. Nessun commit o push durante
l'acquisizione né durante la remediation.

## 13. Remediation della revisione

La [revisione indipendente](REVIEW_INCREMENT_9.md) non ha trovato P0 né P1; ha
trovato quattro P2 nel gate e nell'acquisizione e dieci P3. La remediation del
16 novembre 2026 li corregge senza ripetere alcuna misura: nessun forward,
oracolo, benchmark o download. Le evidenze di misura restano quelle del 15
novembre; sono cambiati gate, verificatore, strumento di acquisizione, audit e
igiene dei percorsi. Ogni cambiamento ai criteri di accettazione è registrato in
[protocol-amendment-remediation.json](../results/increment9/protocol-amendment-remediation.json),
dichiarato successivo alle misure; `protocol.json` è invariato byte per byte e
le soglie numeriche approvate non sono cambiate.

Stato del repository trovato: l'incremento 9 era già nel commit `6031b52`
(«passo 9») e il riferimento locale `origin/main` puntava allo stesso commit. La
remediation non ha eseguito commit, staging, push, fetch o reset; è stata poi
committata dal proprietario come `eb669f2` («passo 9 fix»).

### 13.1 Risultati dimostrati

- **Gate A–H: PASS, exit 0**, ricalcolato dalle evidenze grezze con le regole
  inasprite; precondizioni protocollo e sanificazione PASS. Gate 7 e gate 8
  rieseguiti: PASS, exit 0.
- **Il gate è fail-closed sulle mutazioni provate.** Su copie delle evidenze
  reali, 80 mutazioni: con il gate originale 71 lasciavano PASS le condizioni
  interessate ([red](../results/increment9/review/gate-mutations-red.txt)); con
  il gate corretto nessuna
  ([green](../results/increment9/review/gate-mutations-green.txt)). La seconda
  revisione ha poi trovato 13 aggiramenti fuori da quell'insieme, tra cui un
  PASS con il manifest di sanificazione svuotato: corretti nel follow-up
  (sezione 13.4), con i limiti residui dichiarati.
- **Fase B realmente evitta e letta**, ora verificato dal gate per sync e
  pipeline: residenza dopo l'evizione 0 B su tutti e quattro gli shard, letture
  di blocco 112.420.581.376 B su 113.129.963.520 B logici (99,4%) in sync.
- **Parità confermata dal verificatore corretto** sulle catture esistenti: ogni
  campo dei tre report storici è identico; il greedy upstream ricalcolato
  dall'argmax degli NPZ coincide con la sequenza e ogni ingresso di decode è la
  scelta greedy precedente del runtime.
- **FAIL diagnostici conservati**: FP64 standard Qwen, 31 residui fuori soglia,
  0 logits, 0 KV (exit 1 del verificatore); FP64 standard dell'incremento 7,
  270 logits e 41 residui. Nessuno è stato convertito in PASS; un report
  diagnostico riscritto a PASS ora rende F FAIL.
- **Integrità successiva all'acquisizione**: SHA integrale del runtime e re-hash
  registrati uguali ai digest LFS ufficiali, sull'insieme completo dei file.
- **613 test PASS, zero skip** alla remediation: i 484 dell'acquisizione più
  129 regressioni di revisione; 651 dopo il follow-up. Controlli oracle 9/8/precisione 7: OK (3, 1 e 5 test).
- **Conservazione** rispetto a `68ba9f3`: unico file preesistente modificato
  `README.md`, nessuna cancellazione, niente in staging.
- **Sanificazione**: 11 file di evidenza, 72 sostituzioni dichiarate, nessun
  numero cambiato; ogni file sanificato è riottenuto esattamente dal proprio
  originale applicando le sole regole dichiarate.

### 13.2 Limiti sperimentali

- Restano tutti i limiti della sezione 12: RAM fisica host non superata, 13,3
  s/token, una sola coppia sync/pipeline, contesto di 8 posizioni, temperatura,
  potenza e stato di alimentazione non misurati.
- **Verifica SHA in acquisizione non documentata da un digest registrato** nella
  prova storica (sezione 5). La prova di integrità è quella successiva.
- **Soglie scelte dopo le misure, non preregistrate**: anon+kernel ≤ 256 MiB
  (massimo misurato 110.895.104 B, margine circa 2,4×; campioni ogni 50 ms, che
  possono perdere transitori), residenza dopo l'evizione ≤ 1%, letture di blocco
  ≥ 90% dei byte logici, minimi di test 613/3/1/5. La soglia su VmHWM è
  implicata da RLIMIT_AS = 256 MiB e non aggiunge informazione.
- **Profilo FP64 puro non pienamente indipendente** (sezione 8).
- **Swap globale cresciuto di 76,8 MiB** nella finestra degli oracoli, fuori dai
  cgroup dell'esperimento (sezione 8). Riportato dal gate, non vincolato.
- `integrity-after.json` precede le sonde di guasto e i test OOM e per l'index
  registra solo l'esito; `integrity-final.json` registra anche il digest.
- Smoke test pilota non conservato e contenuto di `protocol.json` prima
  dell'emendamento non verificabile (sezione 2).
- Catture numeriche e originali non sanificati sono locali, fuori da Git. Su
  un'altra macchina il gate lo dichiara e verifica soltanto i digest dei
  manifest; gli originali coincidono con i file del commit `6031b52`.
- `io.stat` non disponibile; contatori host globali; nessun `O_DIRECT`.

### 13.3 Criticità residue

- **I percorsi personali sono già nella storia Git.** Il commit `6031b52`
  contiene le evidenze non sanificate e, secondo il riferimento locale, è già su
  `origin/main`. La sanificazione vale per il working tree e per i commit
  futuri; rimuoverli dalla storia richiede una riscrittura con push forzato, che
  spetta al proprietario e non è stata eseguita.
- **Il percorso personale resta in 309 file tracciati, 548 occorrenze** (la
  remediation scriveva «tre righe in documenti precedenti»: sottostima). Tutti
  preesistenti alla base `68ba9f3`: `docs/` 2 file, `results/increment4` 35,
  `increment5` 15, `increment6` 153, `increment7` 4, `increment8` 8,
  `results/m1a-native-ext4-*` 14, `results/v2` 78. Il nome di un account
  Windows compare in altri 10 file (`BENCHMARK_V2.md`, tre strumenti V2 e
  `results/v2/`). Riscrivere la sola storia dell'incremento 9 non li
  rimuoverebbe. Sanificare le evidenze 4–8 e V2 è una decisione del
  proprietario, in sospeso, perché tocca la conservazione degli incrementi
  1–8: non sono state modificate.
- **La regola di conservazione è cambiata dopo le misure (emendamento R10).**
  Non è un inasprimento: è una correzione semantica. Con la regola originale
  l'albero committato fallirebbe con 75 file «inattesi» al commit `6031b52` e
  90 a `eb669f2`, cioè i file aggiunti dall'incremento. Lo strumento ora
  giudica i soli file preesistenti (modificati o cancellati) ed elenca a parte
  gli aggiunti, che quindi non sono protetti.
- **H dipende dallo stato di git al momento dell'esecuzione**: è ricalcolato dal
  vivo. In un incremento futuro che modifichi file preesistenti, il gate 9 non
  sarà più PASS finché l'elenco intenzionale non viene aggiornato.
- **I test del gate leggono i report derivati presenti** in
  `results/increment9`: vanno rigenerati nell'ordine indicato nel README delle
  evidenze, altrimenti la suite fallisce.
- Il follow-up della seconda revisione e i suoi test non sono ancora stati
  esaminati da una revisione indipendente (stato al momento della stesura; la
  revisione conclusiva del 17 novembre lo ha esaminato: *READY WITH
  DOCUMENTED LIMITATIONS*, sei P3 D-1…D-6, sezione 14).

### 13.4 Follow-up della seconda revisione

La seconda revisione indipendente ha esaminato il commit `eb669f2`. Il
follow-up, **non committato** al momento della stesura (poi pubblicato in
`5eb25da`), non ripete misure e non tocca runtime, kernel,
sorgente, evidenze di misura né `protocol.json`. Dettaglio per rilievo in
[REVIEW_INCREMENT_9.md](REVIEW_INCREMENT_9.md).

- **Sanificazione non più aggirabile dal manifest.** Il gate fissa nomi e
  digest (originale e sanificato) degli undici file; un manifest vuoto, una
  voce rimossa o un originale locale non elencato danno FAIL.
- **Un fatto contraddetto resta FAIL.** Precondizioni non soddisfatte ed
  evidenze mancanti cambiano solo un PASS. Ogni condizione riporta anche il
  verdetto prima delle precondizioni.
- **Nessun crash**: JSON non valido è FAIL, file mancante NOT PROVEN,
  `inference_read_bytes = 0` FAIL.
- **Index e digest calcolati**: `integrity-final.json` deve registrare il
  digest dell'index; un `computed_sha256` deve essere un digest esadecimale
  uguale a quello ufficiale. `resident_bytes` negativo e `oom_kill: false` non
  sono accettati.
- **Registro degli emendamenti**: R10 aggiunto, R5–R7 etichettati come scelti
  dopo le misure.
- Prova: 13 aggiramenti, tutti riusciti contro il gate di `eb669f2`
  ([red](../results/increment9/review/bypasses-followup-red.txt)), nessuno
  contro il gate attuale
  ([green](../results/increment9/review/bypasses-followup-green.txt)).
  **651 test PASS**, zero skip; gate 7, 8 e 9 PASS con exit 0; diagnostici FP64
  standard ancora FAIL grezzi.
- **Limiti dichiarati e non corretti**: digest LFS ancorati solo dal preflight;
  contatori `failures` di parità non ricalcolati dal gate; `generated_text` non
  legato ai token; evizione legata per suffisso; punti di pytest non contati;
  conservazione che non copre il codice aggiunto dal 9 né i flag
  `--assume-unchanged`/`--skip-worktree`; `repository_path` fuori dal checkout;
  test mancanti per interruzione a metà flusso e server senza `Range`;
  accoppiamento di `minimum_pytest_passed`; un solo file di emendamento, senza
  SHA fissato; nessuna opzione `--root`; index di `integrity-after.json`
  registrato solo come esito; combinazione campo mancante più fatto
  contraddetto ancora NOT PROVEN in C, D e parti di E e F.

## 14. Hardening successivo alla chiusura (17 novembre 2026)

Dopo la revisione conclusiva (commit `5eb25da`, *READY WITH DOCUMENTED
LIMITATIONS*) il gate è stato reso più severo su tre punti, senza ripetere
misure e senza modificare evidenze di misura, runtime, adapter, kernel,
`protocol.json` o contratti numerici degli incrementi 7 e 8:

- **D-2, precedenza dei verdetti**: verdetto finale = il peggiore tra il
  verdetto della condizione e quelli delle precondizioni, nell'ordine
  FAIL > NOT PROVEN > PASS. Il gate 9 non usa BLOCKED.
- **D-3, contatori malformati**: ogni contatore di runtime usato in una
  condizione deve essere un intero JSON non negativo (`type(v) is int`, quindi
  non `bool`); un valore presente di tipo errato è FAIL, un campo assente NOT
  PROVEN.
- **D-4, manifest malformato**: presente ma strutturalmente non valido è FAIL,
  assente NOT PROVEN.

Sulle evidenze storiche il report del gate aggiornato è identico a
`summary.json`: **A–H PASS**, precondizioni PASS, FP64 standard FAIL grezzo
conservato. Le differenze rispetto al gate di `5eb25da` riguardano solo
evidenze mutate e sono elencate, con le prove red/green, in
[results/increment9/post-closure/README.md](../results/increment9/post-closure/README.md).
