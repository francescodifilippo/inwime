# InWiMe — Revisione architetturale indipendente

**Data:** 16–17 novembre 2026 · **Ruolo:** Principal Systems Architect (analisi indipendente, critica) · **Modello:** Fable 5.2, effort XHIGH
**Repository:** `francescodifilippo/inwime` (directory locale `ooc-inference`), `HEAD = eb669f2` («passo 9 fix»), più un follow-up del gate 9 **non committato e non toccato**.
**Percorso di questo rapporto:** `[percorso locale originale omesso — rapporto esterno al repository]` (fuori dal working tree).

## Metodo, perimetro e forza delle affermazioni

**Cosa è stato fatto.**

- Lettura integrale del codice del runtime: `inwime_runtime/{core,pipeline,decoder,llama,qwen2,gemv,math_cpu,safetensors_source,telemetry,tokenizer}.py`, `src/cpu_kernel.c`, `Makefile`; lettura degli strumenti dell'incremento 9 (`tools/benchmark_increment9.py`, `cgroup9.py`, `oracle9.py`, `oracle_precision9.py`, `fault_increment9.py`, `integrity9.py`, `acquire_qwen9.py`).
- Lettura di `AGENTS.md`, `PROJECT_BRIEF.md`, `README.md`, `BENCHMARK_V2.md`, `docs/INWIME_ARCHITECTURE_DECISION.md`, `docs/BRIAREUS_ARCHITECTURE_PROPOSAL.md`, `docs/STRATA_TECH_AUDIT_2026-11-13.md`, `docs/RUNTIME_CPU_INCREMENT_9.md`, parti di `docs/INWIME_TECHNIQUE_WATCHLIST.md` e dei rapporti 5–8; lettura delle evidenze `results/increment9/*.json` (in particolare `phaseB-sync-runtime.json`, `phaseB-pipeline-runtime.json`, `parity-*.json`, `environment-before.json`, `protocol.json`).
- GitHub in sola lettura con `gh` (autenticato): Issues #2–#12 e PR #1 (`docs/VISION_AND_ROADMAP.md` sul branch `docs/vision-roadmap-2026-11-15`). Nessuna Issue, PR o label creata o modificata.
- Web, solo fatti pubblici: API e `config.json` Hugging Face di `Qwen/Qwen2.5-14B`, `Qwen2.5-32B`, `Qwen2.5-72B`, `Qwen/Qwen3-30B-A3B`, `deepseek-ai/DeepSeek-V3`, `moonshotai/Kimi-K2-Instruct`; README e `docs/fluxaeros.md` del fork `Betelgeuse-AI/Betelgeuse` citato nella Issue #11. Nessun contenuto del repository inviato all'esterno.
- **Micro-misure di questa revisione** (sintetiche, in memoria, pochi secondi, eseguite in una directory temporanea fuori dal repository, nessun checkpoint letto): costo per elemento dei cicli Python di copia KV e di attenzione; throughput di kernel GEMV/GEMM prototipo su una matrice BF16 sintetica da 135,8 MB; confronto bit per bit di un kernel prototipo con la libreria già compilata `build/libinwime_cpu.so`, caricata in sola lettura. Host non isolato (altri carichi presenti), nessun controllo termico: sono ordini di grandezza, **non benchmark del runtime**. Sorgenti e uscite grezze sono conservati accanto al rapporto (Appendice A).

**Cosa non è stato fatto.** Nessuna modifica a codice, evidenze, sistema o GitHub; nessun commit, stage, stash, checkout; nessun download; nessun forward su checkpoint reali; nessun `pytest`; il checkpoint da 15 GB non è stato letto.

**Classificazione delle raccomandazioni e delle affermazioni chiave.**

| Etichetta | Significato in questo rapporto |
|---|---|
| **PROVEN** | Misurato nelle evidenze del repository, oppure verificato qui leggendo il codice o con una micro-misura riportata in Appendice A (nel suo perimetro) |
| **HIGH CONFIDENCE** | Conseguenza aritmetica o estrapolazione corta di quantità misurate; non ancora osservata end-to-end |
| **HYPOTHESIS** | Plausibile e motivata, non misurata in questo ambiente |
| **REQUIRES EXPERIMENT** | Non decidibile senza una misura dedicata; il rapporto indica quale |

Unità: GB = 10⁹ byte, GiB = 2³⁰ byte. «Gw/s» = miliardi di pesi (coefficienti) elaborati al secondo; per un GEMV su BF16, 1 Gw/s = 2 GB/s di pesi = 2 GFLOP/s.

## 1. Executive technical assessment

**Giudizio.** InWiMe oggi è un **banco di correttezza out-of-core rigoroso**, non ancora un motore d'inferenza. Il disegno di memoria è sano e già scalabile; il costo per token è dominato da scelte rimovibili; sotto quelle scelte c'è un pavimento fisico che decide la strategia.

**Che cosa regge (PROVEN).**

- La memoria non cresce con il modello: 14,14 GB di pesi attivi con 124,8 MB di RSS. Un denso da 30B o 70B sarebbe eseguibile oggi, disco permettendo.
- La catena di verifica (oracolo indipendente a tre contratti, digest, guasti, gate fail-closed) è l'asset principale del progetto e va protetta.
- Il coordinatore unico ha garantito atomicità e riproducibilità dei guasti.

**Che cosa non regge alla scala.**

1. **Pavimento I/O (fondamentale).** Un token denso rilegge 2 byte per parametro. Alla banda misurata (3,0–3,6 GB/s): 7B ≥ 3,9–4,7 s; 70B ≥ 40–48 s; 400B ≥ 3,8–4,5 min; 1T denso ≥ 9–11 min. Nessun kernel lo supera.
2. **Il 63% del token attuale è calcolo evitabile.** Espansione FP64 (4,75 s) e GEMV scalare (3,65 s) in serie alla lettura (3,76 s), su 13,27 s.
3. **Prefill seriale:** N token = N scansioni complete (66,9 s per 5 token).
4. **Contesto limitato a decine di token:** attenzione in Python a 16,8 ms × T per token e copia integrale del KV a ogni passo (micro-misure).
5. **Assunzioni di taglia:** 4.096 tensori e 64 shard al massimo, solo BF16, SHA integrale a ogni avvio, 144,8 GB liberi su un unico NVMe.

**Tre risultati di questa revisione che cambiano le priorità.**

- **Un kernel fuso può restare bit-identico a quello attuale.** Accumulando una riga per corsia SIMD, in ordine di colonna e senza FMA, una micro-misura dà 4,9 Gw/s su un thread contro 0,90 Gw/s di oggi (5,5×) e 17 Gw/s su 12–14 thread, con uscita identica bit per bit a quella della libreria compilata del repository su dati sintetici. La validazione diventa un confronto di digest. **PROVEN a micro-scala.**
- **Il KV a doppio banco non serve.** In un decoder causale il prefisso pubblicato non viene mai riscritto: un banco append-only con pubblicazione atomica della lunghezza dà la stessa garanzia con metà memoria e nessuna copia. **PROVEN sul codice.**
- **Un MoE da 61 GB costerebbe meno di un denso da 14 GB.** Qwen3-30B-A3B ha 6,08 GB attivi per token (geometria verificata sul conteggio ufficiale): ≈ 5,7 s per token ai costi unitari di oggi, contro 13,3 s del 7B; con parte condivisa in RAM e kernel fuso, ordine di 1 token al secondo. **HIGH CONFIDENCE sui byte, HYPOTHESIS sulla latenza finale.**

**Fatto esterno rilevante (verificato oggi).** I MoE pubblici da 0,7–1 T (DeepSeek-V3, Kimi-K2) sono distribuiti in **FP8 E4M3 nativo**, non in BF16, e occupano 688 GB e 1,03 TB: per quella classe servono un dtype nativo in più e storage, non solo ottimizzazioni.

**Raccomandazioni, in ordine.**

| # | Raccomandazione | Classificazione |
|---|---|---|
| 1 | Incremento 10 con **Qwen2.5-14B** (27,98 GB attivi, 1,66× la RAM fisica) e **differenza zero nel runtime**; ≈ 1 h di prova, 29,6 GB di disco | HIGH CONFIDENCE che l'adapter attuale basti (verifica statica) |
| 2 | Subito dopo, motore a numerica invariata: kernel fuso bit-identico, operatori nativi, KV append-only, attraversamento layer-major. Validazione per digest | PROVEN a micro-scala; HIGH CONFIDENCE end-to-end |
| 3 | Poi M11 su **Qwen3-30B-A3B**, scritto una volta sola sulla nuova struttura, con protocollo dei margini del router | HIGH CONFIDENCE sulla fattibilità; il routing reale è REQUIRES EXPERIMENT |
| 4 | Cache e I/O esplicito solo dopo una traccia di routing reale e un A/B page cache contro cache propria | REQUIRES EXPERIMENT |
| 5 | GPU riservata al GEMM a batch (prefill, verifica); non al decode | PROVEN che il decode è limitato dall'I/O (V2/M1a); HYPOTHESIS il guadagno sul prefill |
| 6 | Coordinatore unico confermato, ma a grana di operazione; tre sole estrazioni modulari | HIGH CONFIDENCE |

**Che cosa non fare.** Non ottimizzare dentro l'incremento 10; non passare a FP32 per il GEMV; non aggiungere worker alla pipeline Python; non adottare lo streaming a moduli di PyTorch come motore; non convertire o reimpacchettare i checkpoint; non puntare sul denso ≥ 70B come obiettivo di usabilità.

## 2. Architettura attuale ricostruita dal codice

Ricostruzione dal sorgente a `eb669f2`. I riferimenti sono `file::funzione` con le righe dove utile.

### 2.1 Percorso dei dati NVMe → RAM → CPU

```text
shard safetensors originale (BF16, immutato)
   │  lettura buffered: os.preadv, 512 KiB per chiamata        core.py::FileBackend.readinto (330–336)
   ▼
page cache del kernel (readahead 8 MiB; unico «cache» del sistema; limitata solo dal cgroup esterno)
   │  copia kernel → utente
   ▼
arena «staging» (array 'B', read_elements × 2 byte)            core.py::Coordinator._read (447–469)
   │  espansione esatta BF16 → FP64, scalare                   cpu_kernel.c::inwime_expand (14–24)
   ▼
arena «weights» (array 'd', tile_elements × 8 byte = 2 MiB)    core.py::Coordinator.acquire (471–508)
   │  GEMV FP64 ordinato per colonna, scalare, senza FMA       cpu_kernel.c::inwime_linear (26–44)
   ▼
attivazioni / KV / logits (array 'd', FP64)                    decoder.py, llama.py, qwen2.py
```

- **Nessun percorso GPU nel runtime.** `src/benchmark.cpp` (cuBLAS, due slot pinned) è un benchmark sintetico separato: non è collegato a `inwime_runtime`.
- **Modalità pipeline** (`pipeline.py`): due slot (`weights`/`weights1`, `staging`/`staging1`), un worker di lettura (`ReadWorkers`, 53–117) che riempie lo staging dello slot; espansione e calcolo restano sul thread proprietario (`wait`, 274–313).
- Ogni byte di peso attraversa quindi tre passate seriali e viene espanso di 4 volte prima del calcolo.

### 2.2 Buffer, lease, cache e page cache

- **Arene preallocate** alla costruzione secondo un `Plan` per categoria (`weights`, `staging`, `activations`, `kv`, `accumulators`, `workspace`, `metadata`, `output`, `requests`); il budget è verificato **prima** di allocare o leggere (`Coordinator.__init__`, 393–399; `BudgetError`).
- **Lease** (`RegionLease`, 344–381; `pipeline.py::Lease`, 120–151): riferimento con generazione, valido sul solo thread proprietario; una regione in volo in sync, `io_slots` in pipeline; il rilascio rende lo slot riusabile; `stream()` garantisce il drenaggio anche in uscita anticipata.
- **Nessuna cache applicativa dei pesi.** L'unica cache è la page cache del kernel, che il runtime non governa: nessun `O_DIRECT`, nessun `fadvise`. Il tetto di 2 GiB dell'incremento 9 viene da `tools/cgroup9.py`.
- **Identità dei file:** firma `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` controllata prima e dopo ogni lettura (`signature`, 339–341); SHA256 integrale all'avvio in modalità `full` (432–439).

### 2.3 Scheduling e dipendenze

- **Non esiste uno scheduler.** L'ordine delle operazioni è il flusso di controllo Python di `_token`; le richieste sono generate pigramente da un generatore per ogni tensore (`gemv.py::streamed_linear`, 35–42).
- L'unica anticipazione è in `PipelineCoordinator.stream` (361–383): fino a `io_slots` richieste sottomesse, cioè una tile di anticipo **dentro lo stesso tensore**. Nessun prefetch fra tensori o fra layer.
- **Backpressure:** `BackpressureError` se non ci sono slot. **Cancellazione:** `cancel_event` controllato a ogni acquisizione e a ogni layer; in pipeline imposta uno stop e drena.
- Il coordinatore è l'unica autorità sugli stati di richiesta e slot; i worker restituiscono solo un `Completion`.

### 2.4 Calcolo matriciale e precisioni effettive

| Elemento | Precisione effettiva | Dove |
|---|---|---|
| Pesi su disco | BF16 originale, mai modificato | sorgente safetensors |
| Pesi in calcolo | FP64, espansione esatta; non-finiti rifiutati | `inwime_expand`, `core.py::expand_bf16` |
| Attivazioni, KV, logits | FP64 (`array('d')`) | tutte le arene |
| GEMV | FP64, somma in ordine crescente di colonna, prodotto e somma separati; le tile possono tagliare una riga (riporto in `y[riga]`) | `inwime_linear`; `Makefile` riga 31: `-fno-fast-math -ffp-contract=off`, nessun `-march` |
| RMSNorm, SiLU, softmax, RoPE | FP64 in Python (`math.sqrt`, `math.exp`, `math.cos`, `math.sin`) | `decoder.py::_norm`, `_ffn`; `llama.py::_rope`, `_attention` |

Non esistono percorsi FP32, BF16 nativo o GPU. Il contratto verificato al 9: parità con l'oracolo FP64 puro entro 2,5·10⁻¹² sui logits; parità FP32 standard entro soglia; diagnostico FP64 standard fallito su 31 residui per i cast FP32 interni di upstream.

### 2.5 KV, attenzione e contesti

- **KV:** due banchi FP64, layout `[layer][K|V][posizione][kv_width]`; K memorizzato dopo la RoPE. A ogni chiamata il candidato è copiato dall'attivo per intero, poi il passo scrive la nuova posizione; la pubblicazione è un'unica assegnazione `_published = (banco, lunghezza, generazione)` (`llama.py` riga 281). Qualunque eccezione rende la sessione fail-stop.
- **Attenzione:** causale per costruzione (il token alla posizione p vede 0…p); GQA con `kh = head // (heads // kv_heads)`; softmax a due passate con sottrazione del massimo; tutto in Python.
- **Contesto:** `capacity` fissata alla costruzione; massimo provato 8 posizioni con capacity 16. Niente sliding window, `rope_scaling`, paginazione o compressione del KV.
- **Output:** logits per **tutte** le posizioni del chunk e residui di tutti i layer restano in arena (`llama_plan`, 134–135); la head è calcolata per ogni token.

### 2.6 Adapter e semantica delle famiglie

| Classe | Semantica | Validata su |
|---|---|---|
| `decoder.py::CPURuntime` | Decoder sintetico: posizioni apprese, FFN densa o MoE con router; softmax sui soli top-k selezionati; a parità vince l'indice minore | fixture sintetiche, 12 GiB sintetici |
| `llama.py::LlamaRuntime` | Embedding condivisi, RoPE split-half, GQA, nessun bias | SmolLM2-135M, SmolLM2-1.7B |
| `qwen2.py::Qwen2Runtime` | Bias Q/K/V sommati prima della RoPE, head non condivisa | Qwen2.5-7B |

- I cataloghi (`LlamaCatalog`, `Qwen2Catalog`) validano il `config.json` in modo stretto e **rifiutano** ciò che non riconoscono; mappano nomi logici su nomi fisici e pretendono il numero esatto di tensori.
- La gerarchia è per ereditarietà: ogni famiglia riscrive `_token`; `prefill` e `__init__` sono duplicati.
- **Tokenizer:** backend `tokenizers` da `tokenizer.json`; BOS/EOS; solo greedy (`tokenizer.py::greedy`); nessun chat template.

### 2.7 Shard, FD, metadati ed errori

- **`SafetensorsSource`** (`safetensors_source.py`): descriptor esterno `safetensors.json` con SHA256 attesi di shard e index; JSON analizzato con limiti di profondità e nodi (`bounded_json`, 33–73); header validati (nessun buco o sovrapposizione, dimensioni coerenti, solo BF16); limiti: 64 shard, 4096 tensori, 8 MiB per input, 64 MiB di heap metadati.
- **FD:** uno per shard, aperti alla costruzione con `O_RDONLY | O_NOFOLLOW | O_NONBLOCK` e tenuti fino alla chiusura; le firme devono coincidere con quelle dell'ammissione dei metadati.
- **Errori:** `CheckpointError`, `ReadError` (lettura corta), `BudgetError`, `BackpressureError`, `CancelledError`. Comportamento fail-closed: nessun retry che pubblichi stato parziale; drenaggio delle letture fisiche prima del rilascio.
- **Telemetria** (`telemetry.py::Timeline`): contatori e traccia limitata; eventi sotto lock per ogni richiesta.

### 2.8 Implementato, documentato, ipotizzato

| Stato | Elementi |
|---|---|
| **Implementato e verificato** | Streaming CPU a tile con memoria limitata; safetensors BF16 multi-shard in posto; integrità `full`/`metadata`; sync e pipeline (2 slot, 1 worker); lease con generazioni; piano e budget; kernel FP64 ordinato (Python e C); adapter densi Llama e Qwen2; decoder MoE **sintetico**; KV transazionale a doppio banco; tokenizer e greedy; telemetria; supervisore cgroup; oracoli a blocchi; iniezione di guasti; gate A–H |
| **Documentato ma non implementato nel runtime** | Backend GPU a tile (esiste solo il benchmark sintetico; l'adapter `math_cuda` dell'audit Strata §8.6 è una proposta); O_DIRECT (solo nel benchmark); cache e eviction (Mnemosyne); prefetch e priorità (Ariadne); più sessioni, caricamenti condivisi, deduplicazione (descritti nella decisione architetturale §8); mappa di collocazione multi-NVMe; dtype F16; API OpenAI e Harness; chat template; MoE reale; KV paginato; prefill a batch |
| **Soltanto ipotizzato** (watchlist, Issue) | Speculative decoding e MTP; lookahead del router; relayout contiguo degli esperti; `io_uring`, SPDK; distribuzione; kernel FP8/MXFP4 nativi; acceleratori alternativi |

## 3. Rischi e limiti fondamentali (red team architetturale)

### 3.1 Quadro: che cosa limita davvero la scala

La memoria **non** è un limite di scala del disegno attuale: il piano delle arene dipende da `width`, `hidden`, `vocab`, `capacity` e dalla tile, non dal numero di parametri (`llama.py::llama_plan`, righe 115–145). Con 14,14 GB di pesi attivi il worker ha avuto un picco RSS di 124,8 MB (**PROVEN**, `phaseB-sync-runtime.json`). Un denso da 30B o 70B sarebbe già *eseguibile* oggi, disco permettendo. I limiti stanno altrove, in quattro famiglie:

1. **Fisica dell'I/O** (non rimovibile): un token denso richiede tutti i pesi attivi, cioè 2 byte per parametro BF16 a ogni token.
2. **Struttura dell'attraversamento** (rimovibile): prefill token per token, nessun riuso dei pesi.
3. **Piano dati in Python e kernel scalare** (rimovibile, in parte a costo numerico zero): espansione FP64, GEMV ordinato, attenzione, KV.
4. **Assunzioni di taglia** nei metadati, nell'integrità, nello storage fisico e nei dtype (rimovibili, ma bloccanti per MoE e per 400B–1T).

Sintesi dei rilievi. La gravità è riferita alla missione «LLM enormi su 16 GiB», non alla correttezza dell'incremento 9, che non viene contestata.

| ID | Limite | Gravità | Rimovibile? | Blocca a partire da |
|---|---|---|---|---|
| L1 | Pavimento I/O del decode denso: una scansione completa per token | **Fondamentale** | No (solo attenuabile) | 70B denso: ≥ 40–48 s/token alla banda misurata (≥ 29 s a quella di targa) |
| L2 | Prefill seriale token-major: N scansioni per N token | **Critica** | Sì | qualunque prompt reale, già a 7B |
| L3 | Kernel: tre passate seriali scalari su un thread (copia, espansione FP64, GEMV ordinato) | **Alta** | Sì, a numerica invariata | ogni taglia (fattore ≈ 3,4 sul token) |
| L4 | Piano di controllo Python per tile da 512 KiB | Media | Sì | cresce col volume: 2 M richieste per TB |
| L5 | Attenzione e operatori per-token in Python scalare | **Critica** oltre ~256 token di contesto | Sì, bit-identica | contesto, non parametri |
| L6 | KV a doppio banco FP64 con copia integrale O(capacity) per chiamata | **Alta** | Sì | capacity ≳ 1–4 k |
| L7 | Arene di output per tutte le posizioni (logits, hidden) e head calcolata a ogni token di prefill | Media | Sì | capacity ≳ 1 k |
| L8 | Limiti di catalogo: 64 shard, 4096 tensori, 65 536 nodi JSON, solo BF16 | Media (bloccante per MoE) | Sì | Qwen3-30B-A3B (18 867 tensori) |
| L9 | Integrità `full`: SHA256 integrale a ogni avvio | Media | Sì | 9 min a 688 GB, 27 min a 2 TB per avvio |
| L10 | Dipendenza dalla page cache; limite di memoria imposto solo dall'esterno (cgroup) | Media-alta | Sì | host da 16 GiB sotto pressione, prova M10 |
| L11 | Adapter per ereditarietà con forward duplicato; MoE solo sintetico | Media | Sì | M11 |
| L12 | Storage fisico: 145 GB liberi, un solo slot NVMe | **Alta** | Solo con hardware | 72B (145 GB), 684B (688 GB), 1T (1,03 TB) |
| L13 | I checkpoint MoE pubblici più grandi sono FP8 nativi, non BF16 | **Alta** per 400B–1T | Sì (kernel nativo esatto) | DeepSeek-V3, Kimi-K2 |
| L14 | Top-k del router MoE sensibile alla precisione: rischio per la verifica | **Alta** per M11 | Con protocollo | primo MoE reale |
| L15 | Banda sostenuta, temperatura SSD e throttling non misurati | Media | Con misure | ogni stima > minuti |

### 3.2 Dettaglio dei rilievi

**L1 — Pavimento I/O del decode denso.** *Classificazione: PROVEN (misura) + HIGH CONFIDENCE (estrapolazione).*

- *Causa tecnica.* Un decoder denso usa ogni matrice a ogni token. Senza residenza in RAM, ogni token rilegge `B = 2·P_attivi` byte.
- *Evidenza.* `gemv.py::streamed_linear` (righe 35–42) genera a ogni chiamata le richieste per l'intero tensore; `Qwen2Runtime._token` (righe 119–150) chiama `_linear` per le 7 matrici di ogni layer e per `lm_head`; `weight_cache=False` nel rapporto. Misura: 14.037.975.040 B di letture di blocco per un decode su 14.141.245.440 B logici.
- *Conseguenza.* Con la banda sequenziale misurata su questa macchina (3,0–3,6 GB/s, sezione 4.2): 7B ≥ 3,9–4,7 s/token; 72B ≥ 40–48 s; 405B ≥ 3,8–4,5 min; 1T denso ≥ 9–11 min. Nessuna ottimizzazione di calcolo scende sotto questo valore.
- *Alternative.* (a) Più token per scansione: prefill a batch (L2), verifica speculativa di k token per scansione, più sequenze insieme. (b) Meno byte per token: MoE con lettura selettiva; dtype nativi più compatti quando sono quelli del checkpoint (FP8). (c) Più banda: più NVMe in parallelo. (d) Cache RAM dell'insieme caldo: utile solo se la frazione cacheabile è rilevante (MoE sì, denso ≥ 70B no: 8 GB su 145 GB è il 5,5%).
- *Costo e rischi.* (a) è software ma richiede l'attraversamento layer-major; (b) è la scelta strategica della sezione 10; (c) è hardware.

**L2 — Prefill seriale token-major.** *PROVEN.*

- *Causa.* Il ciclo esterno è sui token, quello interno sui layer; i pesi non vengono riusati fra token dello stesso prefill.
- *Evidenza.* `decoder.py::CPURuntime.prefill` riga 328 e `llama.py::LlamaRuntime.prefill` riga 274: `for output_index, token in enumerate(tokens): self._token(...)`. Misura: prefill di 5 token 66,91 s contro decode mediano 13,27 s (rapporto 5,04). Inoltre `_token` calcola sempre `lm_head` (`llama.py` riga 250, `qwen2.py` riga 150): 1,09 GB letti e moltiplicati per ognuno degli N−1 token di prefill di cui i logits non servono (7,7% dei byte del token).
- *Conseguenza.* TTFT = N × costo di un decode. A 7B: 512 token ≈ 1,9 h; a 72B, con il costo unitario attuale, 512 token ≈ 19 h.
- *Alternativa.* Attraversamento **layer-major**: per ogni layer, ogni tile di pesi viene letta una volta e applicata a tutti i T token del chunk (GEMV multi-RHS). Scansioni = ⌈N/T_c⌉. Head solo per l'ultimo token.
- *Costo e rischi.* Medio-alto: riscrittura di `_token` in tutti gli adapter, arene di attivazione T×width e T×hidden, attenzione causale su batch, KV scritto per T posizioni. Rischio numerico **basso**: se ogni prodotto scalare (riga, token) conserva la stessa struttura di riduzione del GEMV, il risultato è bit-identico al percorso seriale e la validazione è un confronto di digest (sezione 5, O2). Prerequisito duro: attenzione nativa (L5), altrimenti i T²/2 accoppiamenti in Python rendono il batch inutilizzabile.

**L3 — Kernel numerico a tre passate seriali.** *PROVEN (misure del 9 e micro-misure).*

- *Causa.* Ogni tile percorre: copia dalla page cache nello staging (`FileBackend.readinto`), espansione BF16→FP64 in un'arena 4 volte più grande (`Coordinator.acquire` riga 495, `PipelineCoordinator.wait` righe 299–302), GEMV scalare ordinato (`src/cpu_kernel.c::inwime_linear`, righe 26–44). Tutto sul thread proprietario, in sequenza. Il `Makefile` (riga 31) compila senza `-march` e con `-ffp-contract=off`; la riduzione ordinata impedisce la vettorizzazione lungo la riga.
- *Evidenza.* Per token: espansione 4,75 s, GEMV 3,65 s, letture 3,76 s su 13,27 s. Calcolo ≈ 63% del token. Traffico DRAM stimato: ≈ 10 byte movimentati per byte BF16 (1 copia + 4 scritti + 4 riletti + 1 letto), contro 1 per un kernel fuso.
- *Conseguenza.* Moltiplica per ≈ 3,4 il pavimento I/O a ogni taglia.
- *Alternativa verificata a micro-scala.* Kernel **fuso** (BF16 → registro → accumulo) che resta **bit-identico** al kernel attuale: ogni corsia SIMD accumula una riga diversa in ordine crescente di colonna, prodotto e somma separati, niente FMA. Micro-misura (Appendice A): 4,9 Gw/s su un thread contro 0,90 Gw/s della sequenza espansione+GEMV attuale (5,5×); 12,2 Gw/s con 4 thread; 17 Gw/s con 12–14 thread (plateau DRAM). L'uscita del prototipo è **identica bit per bit a quella della libreria compilata del repository** (`build/libinwime_cpu.so`, chiamata come fa il runtime: tile da 262.144 elementi che tagliano le righe) su quattro geometrie sintetiche con zeri, zeri negativi, subnormali e attivazioni ad ampia dinamica.
- *Costo e rischi.* Basso-medio: un backend in più dietro l'interfaccia `math_backend` già sostituibile; `acquire`/`wait` devono poter consegnare lo staging BF16 senza espandere. Rischio numerico **nullo per costruzione** se la bit-identità è confermata sui digest delle catture 8 e 9. Va mantenuto il controllo sui non-finiti oggi fatto da `inwime_expand`.

**L4 — Piano di controllo Python a grana di tile.** *PROVEN.*

- *Causa.* Ogni tile da 262.144 elementi (512 KiB BF16) attraversa in Python: generatore delle richieste, `acquire`, 6 eventi `Timeline.event` sotto lock, 2 `fstat` (`signature`), lease, chiamata ctypes.
- *Evidenza.* 27.113 richieste per token; residuo host 0,51 s per token → ≈ 19 µs per richiesta. `benchmark_increment9.py` usa `read_elements = tile_elements = 262144`. Il benchmark V2 mostra che anche l'I/O soffre i blocchi piccoli: O_DIRECT a 4 MiB 2,0 GiB/s, a 64 MiB 3,36 GiB/s.
- *Conseguenza.* 0,036 s/GB fissi: 4% oggi, > 10% dopo L3; a 1 TB per token sarebbero 2 milioni di richieste. Rende anche inefficace la pipeline applicativa: il tempo di servizio di una tile (≈ 0,14 ms di lettura) è dello stesso ordine della latenza di passaggio fra thread Python (intervallo di switch del GIL: 5 ms). Misura: pipeline 13,83 s contro sync 13,27 s, overlap 0,32 s per token. Il readahead del kernel (8 MiB su `/dev/sdd`) fornisce già la sovrapposizione che la pipeline cercava.
- *Alternativa.* Coordinatore a grana di **operazione** (tensore × operatore, ≈ 10² per token), ciclo delle tile dentro un esecutore nativo; blocchi di lettura 8–32 MiB disaccoppiati dalla tile di calcolo.
- *Costo e rischi.* Basso-medio. Va conservata l'iniettabilità dei guasti a livello di backend (i test di `fault_increment9.py` contano le letture).

**L5 — Attenzione e operatori in Python scalare.** *PROVEN (micro-misura).*

- *Causa.* `LlamaRuntime._attention` (righe 199–221) è un doppio ciclo Python su teste e posizioni; `score()` viene valutata due volte per posizione (una per il massimo, una per la probabilità). Anche `_rope`, `_norm`, la SiLU di `_ffn`, `_bias` e le somme residue sono cicli Python.
- *Evidenza.* Micro-misura alla geometria di Qwen2.5-7B (28 teste, dim 128): 0,60 ms per (layer × posizione di contesto), cioè 16,8 ms × T per token. Coerente con i ≈ 0,5 s di «regioni compute» non-GEMV misurati a T ≤ 8.
- *Conseguenza.* T = 1024 → 17 s/token; T = 4096 → 69 s/token; T = 32 k → 9 min/token, solo di attenzione. Un prefill a batch di 512 token richiede 131 k accoppiamenti per layer: ≈ 37 min in Python. È il vero limite di contesto di oggi, mascherato dai test a 8 posizioni.
- *Alternativa.* Operatori nativi con **lo stesso ordine di operazioni** e la stessa libm (`exp`, `cos`, `sin`, `pow`): attesi bit-identici, validabili per digest. Calcolare lo score una volta sola.
- *Costo e rischi.* Basso-medio; rischio numerico nullo se la bit-identità è confermata.

**L6 — KV a doppio banco con copia integrale.** *PROVEN (codice e micro-misura).*

- *Causa.* A ogni chiamata il banco attivo è copiato elemento per elemento nel candidato: `for i in range(len(active)): candidate[i] = active[i]` (`decoder.py` righe 326–327, `llama.py` righe 272–273). `len(active)` è la **capacità**, non la lunghezza corrente. Due banchi FP64 (`llama_plan` riga 137).
- *Evidenza.* 65 ns per elemento. Geometria Qwen2.5-7B: 28.672 elementi per posizione → 1,86 ms × capacity per chiamata.
- *Conseguenza.* capacity 4096 → 7,6 s per token di sola copia; 32 k → 61 s. Memoria: 2 × 229 KB per posizione → 15 GB a 32 k per un 7B.
- *Osservazione decisiva.* In un decoder causale un passo scrive solo le posizioni ≥ `length`: `_token` scrive K e V a `position` e `_rope` agisce all'offset `ko` della sola nuova posizione. Il prefisso pubblicato non viene mai modificato. Quindi **un solo banco append-only con pubblicazione atomica della lunghezza** dà la stessa atomicità (un guasto lascia rifiuti oltre `length`, mai pubblicati), con metà memoria e zero copie.
- *Costo e rischi.* Basso. `state_digest` deve coprire il solo prefisso. I test di guasto esistenti verificano già l'invarianza del pubblicato. Più avanti: dtype del KV (FP64 diagnostico, FP32, BF16 nativo) e paginazione per layer.

**L7 — Arene di output di collaudo.** *PROVEN.* `llama_plan` righe 134–135 alloca `logits` = capacity × vocab e `hidden` = capacity × layers × width in FP64. A capacity 16 sono 32,3 MB dei 43,7 MB del piano; a capacity 4096 sarebbero 5,0 GB + 3,3 GB. Correzione banale: logits dell'ultima posizione; cattura di hidden solo in modalità diagnostica.

**L8 — Limiti di catalogo e dtype.** *PROVEN (codice) + fatti pubblici verificati.*

- *Evidenza.* `SafetensorsSource.__init__` (righe 77–79): `max_shards=64`, `max_tensors=4096`, `max_input_bytes=8 MiB`, heap metadati 64 MiB; `bounded_json` (riga 55): profondità ≤ 16, nodi ≤ 65.536; i runtime costruiscono `SafetensorsSource(directory)` senza poter passare limiti; solo BF16 (righe 207–208); i cataloghi pretendono il numero esatto di tensori (`qwen2.py` riga 45).
- *Fatti.* Qwen3-30B-A3B: 16 shard e, dal `config.json` verificato, 393 tensori per layer × 48 + 3 = **18.867 tensori**. Qwen2.5-72B: 37 shard. DeepSeek-V3: **163 shard**. Un indice MoE della classe DeepSeek supererebbe anche i 65.536 nodi (stima di ≈ 45 mila tensori esperto, non verificata sugli header).
- *Conseguenza.* Nessun MoE reale è apribile oggi. Correzione semplice (limiti parametrici con preventivo di heap), da esercitare su un caso reale, come chiede già la Issue #6.

**L9 — Integrità integrale a ogni avvio.** *PROVEN.* `Coordinator.__init__` righe 432–439 rilegge e fa l'hash di tutto il payload attraverso lo staging: 12,37 s per 15,23 GB (1,23 GB/s). Proiezione: 24 s a 29,5 GB; 2 min a 145 GB; 9,3 min a 688 GB; 27 min a 2 TB, a ogni processo. Già oggi le sonde di guasto usano `integrity='metadata'` per evitarlo. *Alternativa:* manifest «verificato una volta» legato a `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` più digest per chunk verificati alla prima lettura; `full` resta una modalità esplicita. *Rischio:* è un indebolimento se non è una politica dichiarata e coperta dal gate.

**L10 — Page cache e contenimento esterno.** *PROVEN (codice e misure) + HYPOTHESIS (effetti host).*

- *Evidenza.* Il runtime usa solo `os.preadv` buffered; nessun O_DIRECT, nessun `fadvise`. Il tetto di 2 GiB è imposto da `tools/cgroup9.py`, non dal runtime. In prova: `memory.peak` = limite, 49–64 k eventi `max` di reclaim per run.
- *Conseguenza.* Fuori dal cgroup la cache del guest cresce fino a `MemAvailable`; in WSL2 questo preme sulla RAM dell'host, che all'audit aveva ≈ 0,5 GB liberi (**HYPOTHESIS**: l'effetto sul paging di Windows non è stato misurato). La prova «oltre RAM host» resta valida per aritmetica, ma il motore non è auto-limitato.
- *Alternativa.* Backend esplicito O_DIRECT (o buffered con drop-behind `POSIX_FADV_DONTNEED`) con staging allineato; serve comunque per le letture multi-intervallo dei MoE. Costo medio; rischio numerico nullo (stessi byte).

**L11 — Struttura degli adapter.** *PROVEN.* `CPURuntime → LlamaRuntime → Qwen2Runtime`: `_token` è riscritto in ogni classe, il corpo di `prefill` è duplicato (`decoder.py` 307–349, `llama.py` 252–295), `__init__` è duplicato tre volte. Il MoE esiste solo nel decoder sintetico (`CPURuntime._route`). Per Qwen3-MoE mancano: `head_dim` esplicito (oggi `LlamaSpec.dim = width // heads` e un `head_dim` diverso viene rifiutato), norme su Q e K, router reale, raggruppamento per esperto. L'ereditarietà regge tre famiglie dense; con il layer-major e il MoE ogni famiglia andrebbe riscritta due volte. *Alternativa:* separare sessione/KV (generico) dal «programma del layer» (semantica della famiglia): l'adapter descrive le operazioni di un layer su un batch di posizioni. Non è un esecutore a grafo.

**L12 — Storage fisico.** *PROVEN (audit).* C: ha 144,8 GB liberi (misura odierna), il VHDX è sparse e cresce lì; un solo slot NVMe. Il percorso USB misurato in M1a è ≈ 8 volte più lento dell'interno. Qwen2.5-72B (145,4 GB) non entra con margine; DeepSeek-V3 (≈ 688 GB) e Kimi-K2 (≈ 1,03 TB) no. È una decisione hardware del proprietario, non un difetto software.

**L13 — dtype dei checkpoint enormi.** *PROVEN (fatto pubblico verificato oggi).* DeepSeek-V3: 680,57 miliardi di parametri `F8_E4M3` su 684,49; Kimi-K2-Instruct: 1.023,9 miliardi `F8_E4M3` su 1.026,4. I MoE pubblici da 0,7–1 T **non sono BF16**. Per la regola del progetto vanno letti nel loro formato nativo. FP8 E4M3 è rappresentabile esattamente in FP32, quindi l'espansione è esatta come per BF16 e dimezza i byte per token. La semantica delle eventuali scale per blocco **non è stata verificata** in questa revisione (**REQUIRES EXPERIMENT**: preflight degli header e della documentazione dei pesi).

**L14 — Verifica del routing MoE.** *HYPOTHESIS, rischio di progetto.* Il top-k è una decisione discreta. Runtime FP64 e oracolo FP32 differiscono di 10⁻⁴–10⁻³ sui residui (misurato al 9); se il distacco fra il k-esimo e il (k+1)-esimo logit del router è più piccolo, i due selezionano esperti diversi e divergono in modo visibile senza che nessuno dei due sia «sbagliato». Con 48 layer × posizioni l'evento non è trascurabile. Serve un protocollo preregistrato (sezione 7.1): confronto primario del routing contro l'oracolo FP64, e regola esplicita sui margini per il contratto FP32.

**L15 — Banda sostenuta e termica.** *REQUIRES EXPERIMENT.* Nessuna evidenza registra temperatura SSD, throttling o stato di alimentazione durante letture di minuti a > 3 GB/s; in WSL lo SMART non è visibile. Un kernel multi-thread aggiunge il throttling termico della CPU. Tutte le stime oltre il minuto della sezione 4 assumono banda costante.

### 3.3 Il coordinatore unico resta adeguato?

**Sì come autorità di stato, no alla grana attuale.** *HIGH CONFIDENCE.*

- Ciò per cui l'alternativa A fu scelta (ammissione atomica, transizioni riproducibili, pubblicazione del KV in un solo punto) è dimostrato dai test di guasto e **non dipende** dal fatto che il coordinatore tocchi ogni tile.
- Le ipotesi di falsificazione di `INWIME_ARCHITECTURE_DECISION.md` §8 vanno lette così: «il controllo resta indipendente dal modello» regge; «il lavoro di controllo per tile e completamento è sostenibile» è **falsificata in forma debole** da L4: è sostenibile a 14 GB, non scala, e ha reso inutile la pipeline.
- Raccomandazione: coordinatore unico a livello di **operazione** (acquisisci tensore/esperto, applica operatore a un batch, pubblica). Sotto quel confine, un esecutore nativo fa lettura, doppio buffer e kernel in streaming. L'autorità resta una; i completamenti scendono da ≈ 27.000 a ≈ 200 per token.
- Non serve passare a B (moduli con autorità distinte) né a C (grafo): nessuna delle condizioni di riapertura è presente. Non ci sono politiche sostituibili in concorrenza, né dipendenze dinamiche che un programma per layer non esprima.

### 3.4 Coerenza delle responsabilità Briareus / Mnemosyne / Ariadne / Iris

| Nome | Dove vive oggi nel codice | Giudizio | Modularizzare? |
|---|---|---|---|
| **Briareus** (accesso ai dati) | Indice in `IndexedSource`/`SafetensorsSource`; ma FD, stamp e integrità stanno in `Coordinator.__init__`/`_read`; lettura in `FileBackend` e `ReadWorkers` | Responsabilità **spezzata** fra sorgente (metadati) e coordinatore (FD, hash, letture) | **Sì, per primo**: uno *store di shard* che possiede FD, identità, stato di integrità, backend di lettura e, in futuro, la mappa shard→dispositivo. Serve a O_DIRECT, integrità lazy, multi-intervallo MoE, multi-NVMe |
| **Mnemosyne** (budget e residenze) | `memory_plan`/`llama_plan`, `Budget`, arene in `Coordinator.buffers`, lease | Coerente ma **statica**: nessuna cache, nessuna politica | **Non ancora**: diventa un modulo quando esiste la cache degli esperti (dopo M11) |
| **Ariadne** (ordine, dipendenze, prefetch) | Non esiste come componente: l'ordine è il flusso di controllo Python degli adapter; l'unica anticipazione è `stream()` con 2 slot | Nome senza implementazione | **Minimo**: il «programma del layer» emesso dall'adapter; il prefetch MoE arriverà come suggerimento, mai come autorità |
| **Iris** (osservabilità) | `telemetry.py::Timeline`, contatori in `stats()` | Coesa. Oggi è sul percorso caldo (lock per evento per tile) | No; con la grana a operazione il costo scompare |
| **Adapter** | `CPURuntime`/`LlamaRuntime`/`Qwen2Runtime` mescolano semantica, sessione, transazione KV e layout delle arene | **Incoerente**: tre responsabilità in una gerarchia | **Sì**: separare sessione+KV (generico) da semantica della famiglia |

Componenti che meritano vera modularizzazione, in ordine: (1) store di shard e backend di lettura; (2) esecutore nativo dei kernel (CPU oggi, GPU poi) dietro `math_backend`; (3) sessione e KV store append-only; (4) più avanti, cache bounded. **Non** meritano oggi un modulo: scheduler separato, bus di messaggi, framework di plugin, esecutore a grafo, servizi distinti.

## 4. Analisi quantitativa delle prestazioni

### 4.1 Che cosa è stato misurato nell'incremento 9

Decode di Qwen2.5-7B, modalità sync, scope da 2 GiB, shard rimossi dalla cache prima del lancio; mediane di 3 decode (`phaseB-sync-runtime.json`, tabella §9 del rapporto 9). Byte logici per token: 14.141.245.440 B; coefficienti GEMV per token: 7.070.285.824. **PROVEN.**

| Voce del token | Tempo | Quota | Costo unitario | Throughput equivalente |
|---|---:|---:|---:|---:|
| Servizio letture (`preadv` buffered, 512 KiB, include copia dalla page cache) | 3,76 s | 28% | 0,266 s/GB | 3,76 GB/s apparenti |
| Espansione BF16→FP64 (scalare, 1 thread) | 4,75 s | 36% | 0,336 s/GB | 1,49 Gw/s |
| GEMV FP64 ordinato (scalare, 1 thread) | 3,65 s | 28% | 0,258 s/GB | 1,94 Gw/s = 3,9 GFLOP/s |
| Altre regioni compute (attenzione, norme, SiLU, bias; T ≤ 8) | 0,52 s | 4% | — (dipende da layer, width e T) | — |
| Residuo host (controllo Python: 27.113 richieste) | 0,49 s | 4% | ≈ 19 µs/richiesta | — |
| **Totale** | **13,27 s** | | **0,94 s/GB** | 1,07 GB/s end-to-end |

Le voci sono mediane indipendenti: sommano a 13,17 s contro 13,27 s di totale (la differenza è l'attesa dati non coperta dal servizio letture).

Tre osservazioni che la tabella impone.

1. **Le voci si sommano invece di sovrapporsi.** Il token costa `B·(1/BW_app + 1/R_exp + 1/R_gemv)`: è la somma di tre passate seriali sugli stessi dati, non il massimo fra I/O e calcolo.
2. **Il disco non è saturo.** In media il token consuma 14,04 GB in 13,27 s, cioè 1,06 GB/s: un terzo della banda sequenziale misurata su questo stack. Il readahead del kernel (8 MiB) lavora mentre il thread calcola: per questo il «servizio letture» appare a 3,76 GB/s e per questo la pipeline applicativa non aggiunge nulla (13,83 s contro 13,27 s, overlap 0,32 s).
3. **Il prefill è 5,04 decode** (66,91 s per 5 token): nessun riuso dei pesi.

### 4.2 Banda NVMe sostenibile su questa macchina

| Misura (fonte) | Condizioni | Valore |
|---|---|---:|
| Solo I/O, V2 (`BENCHMARK_V2.md`) | O_DIRECT, QD1, blocchi 64 MiB, ext4 su VHDX interno | 3,36 GiB/s = 3,61 GB/s |
| Solo I/O, V2 | idem, blocchi 4 MiB | 2,00 GiB/s = 2,15 GB/s |
| Solo I/O, M1a (controllo interno) | O_DIRECT, 64 MiB, altra sessione | 5 GiB in 1,80 s = 2,98 GB/s |
| Esterno USB ext4, M1a | O_DIRECT, 64 MiB | 5 GiB in 12,05 s = 0,45 GB/s |
| SHA di avvio, incremento 9 | buffered, 512 KiB, hash sullo stesso thread | 15,23 GB in 12,37 s = 1,23 GB/s (limitato dall'hash) |
| Dato di targa Samsung 990 EVO 2 TB | lettura sequenziale, pubblico | fino a 5,0 GB/s |

Per il modello uso **BW = 3,0–3,6 GB/s** come intervallo misurato (**PROVEN** a QD1 su file da 5 GiB) e 5,0 GB/s come tetto di targa (**HYPOTHESIS** che lo stack WSL2/VHDX/NTFS lo raggiunga con code più profonde). Nessuna misura copre letture sostenute per minuti né la temperatura dell'SSD (L15). Il percorso misura il disco virtuale: non la NAND isolata.

### 4.3 Che cosa può fare la CPU: micro-misure di questa revisione

Matrice BF16 sintetica 18.944 × 3.584 (135,8 MB, più grande delle cache), in memoria, i7-12700H sotto WSL2, host non isolato. Appendice A per sorgente e output. **PROVEN nel perimetro della micro-misura; non è una misura del runtime.**

| Kernel GEMV | 1 thread | 4 thread | 12–14 thread | Numerica rispetto al kernel attuale |
|---|---:|---:|---:|---|
| A — semantica attuale: espansione FP64 + GEMV ordinato (due passate) | 0,90 Gw/s | — | — | riferimento (coerente con 0,84 Gw/s del 9) |
| E — fuso, **ordinato per riga, vettorizzato fra 8 righe** (mul e add separati) | 4,92 Gw/s | 12,2 Gw/s | 17,1–17,3 Gw/s | **bit-identica**: uguale a `build/libinwime_cpu.so` su 4 geometrie sintetiche |
| B — fuso, FP64, riduzione a corsie lungo la riga (FMA) | 5,66 Gw/s | 15,1 Gw/s | 20,4 Gw/s | scarto max 3,6·10⁻¹¹ su \|y\| ≤ 7,6·10³ |
| C — fuso, FP32, riduzione a corsie | 7,91 Gw/s | 20,0 Gw/s | 21,2–21,6 Gw/s | scarto max 1,5·10⁻³ (accumulo FP32) |

| Kernel GEMM FP32 ingenuo (T colonne per scansione) | 1 thread | 6 thread | 14–20 thread |
|---|---:|---:|---:|
| T = 16 | 20,6 GFLOP/s | 85,8 | 113–128 |
| T = 64 | 44,9 GFLOP/s | 156 | 197–211 |
| T = 256 (non bloccato per la cache) | 24,0 GFLOP/s | 116 | 157–165 |

Evidenze già nel repository, coerenti: il kernel CPU del benchmark V2 (OpenMP, 8 thread, accumulo FP32) elabora 5 GiB in 142,6 ms a n = 1 (37,6 GB/s di pesi) e raggiunge 94 GFLOP/s a n = 16; la GPU (cuBLAS BF16) elabora gli stessi 5 GiB in 36,3 ms a n = 16, cioè **2,37 TFLOP/s**, con H2D a 12,9 GB/s.

Conclusioni.

- Il plateau di banda della memoria è ≈ 17–21 Gw/s (34–42 GB/s di pesi BF16). A quel plateau FP64 e FP32 quasi coincidono: **la precisione FP64 non costa throughput quando il kernel è limitato dalla DRAM**.
- Il kernel bit-identico E su **un solo thread** elabora 9,8 GB/s di pesi, cioè 2,7–3,3 volte la banda dell'NVMe. **Per il decode denso da un solo NVMe basta un thread**; i thread servono al prefill (GEMM) e ai pesi serviti dalla RAM (parte condivisa dei MoE).
- Il GEMM ingenuo raggiunge 110–210 GFLOP/s. Una BLAS ottimizzata può fare meglio (**HYPOTHESIS**, non misurata).

### 4.4 Modello del costo per token

Simboli: `B` byte di pesi attivi per token; `B_miss` byte non serviti dalla RAM; `BW` banda NVMe; `R_k` velocità del kernel in byte di pesi al secondo; `P` coefficienti GEMV per token; `F` FLOP/s del GEMM; `T_c` token per scansione; `t_att(T)` attenzione e KV a contesto T; `t_ctl` controllo.

- **Oggi (seriale):** `t_dec ≈ B·(1/BW_app + 1/R_exp + 1/R_gemv) + t_att + t_ctl ≈ 0,94 s/GB · B`.
- **Dopo il kernel fuso, I/O invariato (non sovrapposto):** `t_dec ≈ B·(1/BW_app + 1/R_k) + t_att + t_ctl`. Con `R_k` = 9,8 GB/s: ≈ 0,41 s/GB.
- **Obiettivo (sovrapposto):** `t_dec ≈ max(B_miss/BW, B/R_k) + t_att + t_ctl`. Con `R_k ≫ BW` è il pavimento I/O `B_miss/BW`.
- **Prefill a batch:** `t_tok ≈ max(B_scan/(BW·T_c), 2P/F)`. Il punto di pareggio fra I/O e calcolo è `T* = F/BW` (F in FLOP/s, BW in byte/s).
- **Decode multi-token per scansione** (verifica speculativa, più sequenze): `t_tok ≈ max(B/(BW·k_eff), 2P/F)`.

**Roofline.** Un GEMV su BF16 ha intensità aritmetica 1 FLOP per byte di pesi. In streaming da NVMe il tetto è quindi `BW` espresso in FLOP: **3,0–3,6 GFLOP/s**. Il kernel scalare attuale (3,9 GFLOP/s nel solo GEMV) è già a quel livello: il problema non è la sua velocità di picco ma le passate in serie. Dalla DRAM il tetto è ≈ 34–42 GFLOP/s. Il batching alza l'intensità a T FLOP per byte: con `F` = 100–200 GFLOP/s (CPU) il pareggio è `T*` ≈ 28–67 token; con 2,37 TFLOP/s (GPU, misurato a n = 16) è `T*` ≈ 660–790 (**HYPOTHESIS** oltre n = 16).

Assunzioni esplicite del modello.

| # | Assunzione | Stato |
|---|---|---|
| A1 | BW sequenziale 3,0–3,6 GB/s; tetto 5,0 GB/s | PROVEN a QD1 su 5 GiB; tetto HYPOTHESIS |
| A2 | I costi unitari per byte non dipendono dalla taglia del modello a contesto corto | HIGH CONFIDENCE (tutte le voci sono lineari nei byte) |
| A3 | Letture selettive di estensioni ≥ 3 MiB a QD ≥ 8 rendono come le sequenziali | REQUIRES EXPERIMENT |
| A4 | Nessun degrado termico o di alimentazione su durate di minuti | REQUIRES EXPERIMENT (mai misurato) |
| A5 | Parametri attivi dei MoE diversi da Qwen3-30B-A3B: valori dichiarati dagli autori | non verificati qui |
| A6 | Le velocità dei kernel vengono da dati sintetici in memoria; end-to-end possono essere inferiori | HIGH CONFIDENCE come ordine di grandezza |
| A7 | Contesto corto (T ≲ 64) salvo dove indicato | — |

### 4.5 Modelli densi: decode per token

`B` = pesi attivi per token in BF16 (tutto il modello meno l'embedding). I valori di Qwen2.5 derivano dai conteggi ufficiali verificati (Appendice B).

| Modello | B (GB) | Runtime attuale, 0,94 s/GB | Kernel fuso 1 thread, I/O attuale, 0,41 s/GB | Limite I/O a 3,0–3,6 GB/s | Pavimento a 5 GB/s di targa |
|---|---:|---:|---:|---:|---:|
| 7B (Qwen2.5-7B) | 14,14 | **13,3 s (misurato)** | ≈ 5,8 s | 3,9–4,7 s | 2,8 s |
| 14B (Qwen2.5-14B) | 27,98 | ≈ 26 s | ≈ 11,4 s | 7,8–9,3 s | 5,6 s |
| 32B (Qwen2.5-32B) | 63,97 | ≈ 60 s | ≈ 26 s | 17,8–21,3 s | 12,8 s |
| 70B (Qwen2.5-72B) | 142,9 | ≈ 134 s | ≈ 58 s | 39,7–47,6 s | 28,6 s |
| 400B (classe 405B) | ≈ 810 | ≈ 12,7 min | ≈ 5,5 min | 3,8–4,5 min | 2,7 min |
| 1T denso | ≈ 2.000 | ≈ 31 min | ≈ 13,6 min | 9,3–11,1 min | 6,7 min |

Classificazione: prima colonna di tempi **PROVEN** per 7B e **HIGH CONFIDENCE** per gli altri (stesso codice, costi lineari); seconda **HIGH CONFIDENCE**; terza è un **limite inferiore** fisico alla banda misurata; quarta **HYPOTHESIS**. A contesto lungo va aggiunto `t_att` (sezione 4.8).

**Lettura.** L'ottimizzazione del calcolo vale al massimo un fattore ≈ 3 sul decode denso; poi comanda il disco. Un denso da 70B resta a decine di secondi per token e uno da 400B a minuti, qualunque kernel si scriva. Il denso oltre ~30B è una dimostrazione di eseguibilità, non un obiettivo di usabilità, salvo decode multi-token per scansione o più NVMe.

### 4.6 MoE: contano i byte attivi, non la dimensione totale

Geometria di **Qwen3-30B-A3B** ricostruita dal `config.json` ufficiale e verificata contro il conteggio pubblicato (30.532.122.624 parametri, coincidenza esatta, Appendice B):

| Quantità | Parametri | Byte BF16 |
|---|---:|---:|
| Totale del checkpoint | 30.532.122.624 | 61,06 GB |
| Parte condivisa usata a ogni token (attenzione, norme, router, head) | 1.229.928.448 | 2,46 GB |
| Esperti instradati per token (48 layer × 8 esperti × 4.718.592) | 1.811.939.328 | 3,62 GB |
| **Attivi per token** | **3.041.867.776** | **6,08 GB** |
| Un esperto (gate, up, down) | 4.718.592 | 9,44 MB (tre tensori da 3,0 MiB) |
| Insieme di tutti gli esperti | 28.991.029.248 | 57,98 GB |

| Scenario di decode, Qwen3-30B-A3B | Byte da NVMe per token | Stima |
|---|---:|---:|
| Costi unitari del runtime attuale, nessuna cache | 6,08 GB | ≈ 5,7 s |
| Kernel fuso 1 thread, I/O attuale | 6,08 GB | ≈ 2,5 s |
| Limite I/O, nessuna cache | 6,08 GB | 1,7–2,0 s |
| Parte condivisa residente in RAM (2,46 GB) | 3,62 GB | 1,0–1,2 s |
| Come sopra, più cache esperti con hit del 50% | 1,81 GB | 0,5–0,6 s |
| Pavimento di calcolo (3,04 Gw a 12–17 Gw/s, multi-thread) | 0 | 0,18–0,25 s |

Prima riga **HIGH CONFIDENCE** (stessi costi unitari, ma l'adapter non esiste); le altre **HYPOTHESIS**. Il tasso di hit della cache esperti è **REQUIRES EXPERIMENT**: nessuna traccia di routing reale è mai stata raccolta.

**Il punto strategico.** Già con il runtime di oggi, un MoE da 61 GB costerebbe circa 5,7 s per token: **meno della metà del denso da 14 GB (13,3 s) e un quinto del denso da 28 GB (26 s)**. Con cache della parte condivisa e kernel fuso si entra nell'ordine di 1 token al secondo su un modello 3,6 volte più grande della RAM fisica.

Altre classi MoE, solo limite I/O a 3,0–3,6 GB/s, senza cache:

| Classe (totale / attivi dichiarati) | dtype del checkpoint | Byte per token | Limite I/O | Note |
|---|---|---:|---:|---|
| ≈ 100B / 12B | BF16 | 24 GB | 6,7–8,0 s | attivi non verificati qui |
| ≈ 235B / 22B | BF16 | 44 GB | 12,2–14,7 s | attivi non verificati qui |
| DeepSeek-V3: 684B / ≈ 37B | **FP8 E4M3 nativo** (verificato) | ≈ 37 GB | 10,3–12,3 s | 688 GB su disco; in BF16 equivalente sarebbero 74 GB e 21–25 s |
| Kimi-K2: 1,03T / ≈ 32B | **FP8 E4M3 nativo** (verificato) | ≈ 32 GB | 8,9–10,7 s | 1,03 TB su disco |

Per queste classi la parte condivisa (decine di GB) non entra in 16 GiB: il guadagno della cache è limitato agli esperti caldi e ai tensori piccoli.

**Avvertenza sul prefill MoE.** In un chunk di T_c token ogni layer tocca fino a `min(n_esperti, 8·T_c)` esperti. Con T_c = 64 su Qwen3-30B-A3B sono quasi tutti i 128: il chunk legge circa l'intero layer. Il prefill MoE è limitato dall'**unione** degli esperti del chunk, non dai parametri attivi: ≈ 61 GB per chunk, cioè 17–20 s per 64 token (0,27–0,32 s/token) oppure per 512 token (0,03–0,04 s/token di I/O, poi comanda il calcolo). Serve il raggruppamento dei token per esperto e conviene un chunk grande. **HIGH CONFIDENCE** (aritmetica).

### 4.7 Prefill e batching

Tempo per token di prompt con attraversamento layer-major, `F` = 100–200 GFLOP/s (GEMM FP32 ingenuo misurato; una riduzione FP64 ordinata, bit-identica al seriale, renderà circa la metà: **HYPOTHESIS**):

| Modello | Coefficienti GEMV per token | Calcolo per token, CPU | Prompt da 512 token | Oggi (seriale) |
|---|---:|---:|---:|---:|
| 7B | 7,07·10⁹ | 0,07–0,14 s | 36–72 s | ≈ 6.800 s (1,9 h) |
| 14B | 13,99·10⁹ | 0,14–0,28 s | 72–143 s | ≈ 3,7 h |
| 70B | 71,5·10⁹ | 0,71–1,43 s | 6–12 min | ≈ 19 h |
| Qwen3-30B-A3B | 3,04·10⁹ | 0,03–0,06 s | ≈ 20–40 s (un chunk, limitato dall'I/O dell'unione) | — |

Guadagno atteso sul prefill: **50–190×** a 512 token (**HIGH CONFIDENCE** sull'ordine di grandezza). Il limite diventa il calcolo: un denso da 70B fa circa 1 token al secondo di prefill su questa CPU anche con tutti i pesi in RAM. Qui, e solo qui, la GPU cambia il quadro: a 2,37 TFLOP/s il 70B scende a ≈ 0,06 s per token di calcolo, e 4.096 token costano ≈ 4 min di calcolo più 8 scansioni da 40–48 s.

Decode multi-token per scansione, esempio 70B: con una scansione da 40–48 s e `k_eff` = 4 token accettati per scansione si avrebbero 10–12 s per token emesso, con uscita greedy identica al modello grande. **HYPOTHESIS**: il tasso di accettazione va misurato; richiede il forward a batch.

### 4.8 Attenzione e KV

Operazioni per token e per posizione di contesto, con lo score calcolato una sola volta: `layer × teste × head_dim × 2`.

| Modello | MAC per posizione | Costo oggi (Python) | A T = 32.768, nativo a 5–17 Gw/s | KV per posizione: FP64 / FP32 | KV a 32.768 posizioni |
|---|---:|---:|---:|---:|---:|
| 7B | 200.704 | 16,8 ms × T per token (misurato) | 0,4–1,3 s | 229 KB / 115 KB | 7,5 GB / 3,8 GB |
| 14B | 491.520 | ≈ 41 ms × T | 0,9–3,2 s | 786 KB / 393 KB | 25,8 GB / 12,9 GB |
| 70B | 1.310.720 | ≈ 110 ms × T | 2,5–8,6 s | 1.311 KB / 655 KB | 42,9 GB / 21,5 GB |

- Oggi l'attenzione Python supera il costo dei pesi a T ≈ 800 per il 7B; la copia del doppio banco aggiunge 1,86 ms × capacity per chiamata (**PROVEN**, micro-misure).
- Con operatori nativi l'attenzione resta sotto il pavimento I/O dei pesi fino a decine di migliaia di token (**HIGH CONFIDENCE**).
- Il KV in RAM è il vincolo di contesto per i modelli grandi: a 70B servono 21,5 GB in FP32 a 32 k. Oltre ≈ 8 k posizioni su 16 GiB occorre paginare il KV su NVMe per layer. L'accesso è sequenziale per layer, quindi è compatibile con lo stesso schema di streaming, ma comporta **scritture** su disco (≈ 0,66 MB per token a 70B): vanno autorizzate e limitate (AGENTS.md).

### 4.9 Overhead di scheduling

19 µs per richiesta × 27.113 richieste = 0,49 s per token a 7B (**PROVEN**). È lineare nei byte con tile da 512 KiB: 0,036 s/GB. Con blocchi da 8–32 MiB scende di 16–64 volte e diventa irrilevante; con la grana a operazione (≈ 200 per token) scompare dal profilo.

### 4.10 Valutazione critica delle tecniche

| Tecnica | Agisce su | Evidenza | Giudizio |
|---|---|---|---|
| Kernel multi-thread e SIMD | `R_k` | Micro-misure: 4,9 Gw/s (1 thread, bit-identico), 17 Gw/s (12–14 thread) | Fare. Per il decode denso basta 1 thread; i thread servono a GEMM e pesi in RAM. **PROVEN** a micro-scala, **HIGH CONFIDENCE** end-to-end |
| BF16 consumato direttamente | elimina `R_exp` e l'arena FP64 | L'espansione è il 36% del token; la conversione in registro costa due istruzioni intere | Fare, fuso nel kernel. **PROVEN** a micro-scala |
| FP32 al posto di FP64 nel GEMV | `R_k` | 1 thread: 7,9 contro 4,9 Gw/s; al plateau 21 contro 17 | **Non ora**: guadagno ≤ 1,6× su una voce che dopo la fusione non è più il collo; costa la bit-identità e la parità FP64 |
| BLAS CPU | `F` nel prefill | Non misurata; il GEMM ingenuo fa 110–210 GFLOP/s | Per il GEMV perde (richiede materializzare FP32). Per il GEMM a T grande è un candidato, con contratto FP32 e dipendenza esterna. **REQUIRES EXPERIMENT**, dopo il kernel multi-RHS proprio |
| Riuso dei pesi nel prefill, batching | scansioni per token | Prefill = 5,04 decode; l'oracolo a moduli processa 5 token con una sola passata | Fare: è la leva più grande sulla fruibilità. **HIGH CONFIDENCE** |
| Prefetch | sovrapposizione | Il readahead del kernel già copre il sequenziale; pipeline applicativa senza guadagno | Denso: nulla da aggiungere finché l'I/O è buffered. MoE: gli esperti sono noti solo dopo il router; un prefetch predittivo è un suggerimento con guadagno limitato dal tempo di calcolo. Rinviare dopo la traccia di routing |
| I/O asincrono e doppio buffer | `max` invece di somma; indipendenza dalla page cache | V2: O_DIRECT 64 MiB a 3,6 GB/s; a 4 MiB 2,15 GB/s | Fare in forma nativa con blocchi grandi, non con thread Python. Guadagno sul decode denso ≤ 1,2–1,3× (**REQUIRES EXPERIMENT**); valore principale: memoria auto-limitata e letture multi-intervallo MoE |
| Cache bounded | `B_miss` | Denso: head = 7,7% dei byte a 7B, 1,7% a 70B. MoE: parte condivisa = 40% dei byte per token a 30B-A3B | Denso: marginale. MoE: decisiva. Politica da scegliere sulla traccia reale. **HYPOTHESIS** |
| GPU tiled opzionale | `F` | V2/M1a: nel regime decode il guadagno wall è 2–12%; kernel a 2,37 TFLOP/s a n = 16 | Decode: no (limitato dall'I/O). Prefill e verifica a batch: sì. **PROVEN** il primo punto nel perimetro V2/M1a; **HYPOTHESIS** il secondo |
| Streaming selettivo degli esperti | `B` | Aritmetica della sezione 4.6 | La leva di scala. Richiede prima il routing esatto. **HIGH CONFIDENCE** sui byte, **REQUIRES EXPERIMENT** su località e hit |

## 5. Tre ottimizzazioni prioritarie

Criterio di ordinamento: beneficio atteso sulla missione, complessità, rischio numerico, costo di validazione. Le prime due hanno una proprietà rara: **possono essere validate per uguaglianza di digest** con le catture già accettate degli incrementi 8, 9 (e 10), invece che con nuove campagne di oracolo.

| # | Ottimizzazione | Beneficio atteso | Complessità | Rischio numerico | Costo di validazione | Classificazione |
|---|---|---|---|---|---|---|
| **O1** | Kernel GEMV BF16 **fuso e bit-identico**, blocchi grandi, coordinatore a grana di operazione | Decode 7B: 13,3 s → ≈ 5,8 s con 1 thread e I/O invariato; ≈ 4–5 s con sovrapposizione. Calcolo 8,4 s → 1,4 s | Bassa-media | **Nullo per costruzione** | Molto basso: digest | PROVEN a micro-scala; HIGH CONFIDENCE end-to-end |
| **O2** | Operatori nativi bit-identici e KV append-only (O2a); attraversamento **layer-major** con prefill multi-RHS (O2b) | Contesto: da decine a migliaia di token. Prefill: da N scansioni a ⌈N/T_c⌉ (50–190× a 512 token) | Media-alta | Nullo se la riduzione per (riga, token) resta ordinata | Basso-medio: digest batch = seriale | HIGH CONFIDENCE sul beneficio; bit-identità da confermare |
| **O3** | Backend di lettura esplicito (O_DIRECT, blocchi 16–64 MiB, lettore nativo a doppio buffer) e residenza RAM bounded dell'insieme caldo | Decode denso ≤ 1,2–1,3× dopo O1; soprattutto memoria auto-limitata, niente reclaim, letture multi-intervallo per i MoE | Media | Nullo (stessi byte) | Basso-medio | REQUIRES EXPERIMENT sul guadagno di velocità; HIGH CONFIDENCE sul contenimento della memoria |

### O1 — Kernel fuso bit-identico

**Che cosa.** Un backend matematico nuovo dietro l'interfaccia `math_backend` già esistente (`math_cpu.py::NativeMath`), che consuma lo staging BF16 senza l'arena FP64:

- gruppi di 8 righe; ogni corsia SIMD accumula **una riga** in ordine crescente di colonna; prodotto e somma separati, nessuna FMA: la stessa sequenza di arrotondamenti di `inwime_linear`;
- righe parziali ai bordi della tile gestite dal percorso scalare con il riporto in `y[riga]`, come oggi;
- controllo dei non-finiti conservato (confronto vettoriale sull'esponente BF16), oggi svolto da `inwime_expand`;
- thread facoltativi su gruppi di righe: le righe sono indipendenti, quindi il risultato non dipende dal numero di thread;
- blocchi di lettura da 8–32 MiB, distinti dalla tile di calcolo; ciclo delle tile dentro il codice nativo; il coordinatore vede un'operazione per tensore.

**Modifiche al runtime.** `Coordinator.acquire` e `PipelineCoordinator.wait` non espandono quando il backend dichiara di consumare BF16; `RegionLease.linear` consegna lo staging. Il kernel `c-fp64-ordered` attuale resta come riferimento.

**Evidenza.** Micro-misura (Appendice A): 4,92 Gw/s su un thread contro 0,90 Gw/s della sequenza attuale. Il prototipo è stato confrontato con la libreria compilata del repository (`build/libinwime_cpu.so`: `inwime_expand` più `inwime_linear`, con le stesse tile da 262.144 elementi che tagliano le righe): uscita **identica bit per bit** su quattro geometrie sintetiche (2.048 × 3.584, 1.027 × 5.120, 515 × 18.944, 64 × 13.827), con pesi BF16 casuali finiti, zeri, zeri negativi, subnormali e attivazioni fra 10⁻⁶ e 10⁶. Non è ancora una prova sui checkpoint reali.

**Esperimento.** A/B controbilanciato, 5 coppie, processi nuovi: SmolLM2-1.7B con cache calda (regime di calcolo) e Qwen2.5-7B cold-guest nello scope da 2 GiB (regime misto). Metriche: decode mediano, voci del profilo, digest.

**Accettazione.** Digest di logits, residui e KV **uguali** a quelli accettati per SmolLM2-135M, SmolLM2-1.7B e Qwen2.5-7B; invarianza rispetto a dimensione della tile, numero di thread e modalità sync/pipeline; test di guasto invariati; decode 7B cold almeno 2 volte più rapido.

**Criterio di abbandono.** Una differenza di digest non spiegata: il kernel non può essere adottato come «bit-identico». Resterebbe valutabile come contratto separato, con oracolo proprio.

### O2 — Operatori nativi, KV append-only, attraversamento layer-major

**O2a — prerequisito.** Attenzione (score calcolato una volta), RMSNorm, RoPE, SiLU, bias e somme residue in C, con **lo stesso ordine di operazioni** del Python attuale e la stessa libm; KV a banco unico append-only con pubblicazione atomica della lunghezza (sezione 3.2, L6); logits della sola ultima posizione, con cattura diagnostica facoltativa.

- Beneficio: elimina la crescita di 16,8 ms × T per token e la copia di 1,86 ms × capacity (geometria 7B); dimezza la memoria del KV.
- Validazione: digest uguali alle catture esistenti; nuovo test di contesto lungo (per esempio 1.024 posizioni su SmolLM2-1.7B) con parità contro l'oracolo.
- Rischio: l'identità con `math.exp`, `math.cos`, `math.sin` e l'elevamento a potenza di Python va verificata, non assunta (**REQUIRES EXPERIMENT**, costo minimo).

**O2b — layer-major.** Per ogni layer, ogni blocco di pesi viene letto una volta e applicato alle T posizioni del chunk (GEMV multi-RHS); attenzione causale sul batch; KV scritto per T posizioni e pubblicato in un solo punto; head solo per l'ultima posizione. T_c è limitato dal budget delle attivazioni (per il 14B: 512 × 13.824 × 8 B = 56,6 MB).

- Bit-identità con il prefill seriale: per ogni riga, l'accumulo sui token è vettorizzato **fra token** (ogni corsia è un token), in ordine crescente di colonna. Ogni prodotto scalare (riga, token) ha la stessa sequenza di operazioni del GEMV. **HYPOTHESIS** finché non è provata dai digest.
- Struttura: è il momento di separare sessione e KV (generici) dal «programma del layer» della famiglia, così Llama e Qwen2 condividono l'attraversamento e l'adapter MoE nascerà già sulla forma giusta.
- Validazione: digest batch = seriale sui tre checkpoint reali per più suddivisioni (1, 2, 5, N); atomicità di un passo da T posizioni sotto guasto; parità con gli oracoli invariata.
- Accettazione: prefill di 512 token su Qwen2.5-7B in meno di 5 minuti (oggi ≈ 1,9 h), digest invariati.

### O3 — Lettura esplicita e insieme caldo in RAM

**Che cosa.** Variante di `FileBackend` con O_DIRECT, staging allineato a 4096 byte, blocchi da 16–64 MiB, 2–3 buffer e un lettore nativo che sovrappone lettura e calcolo; offset dei tensori non allineati gestiti leggendo la finestra allineata che li contiene, con l'amplificazione contabilizzata; nessun ripiego silenzioso a buffered. In più, una piccola residenza RAM contabilizzata nel piano per i tensori usati a ogni token (per i densi `lm_head`; per i MoE la parte condivisa).

**Perché terza e non prima.** Sul decode denso il guadagno è limitato: dopo O1 il token è già vicino al pavimento I/O, e il readahead del kernel fornisce oggi la sovrapposizione. Il valore sta altrove:

- il runtime diventa **auto-limitato** in memoria, senza dipendere dal cgroup e senza far crescere la page cache del guest sotto WSL2;
- scompare il reclaim (49–64 mila eventi per run al 9);
- nasce l'interfaccia di lettura multi-intervallo che i MoE richiedono.

**Esperimento (ipotesi H01/H02 della watchlist).** Buffered contro diretto a pari dimensione di blocco, cold-guest, Qwen2.5-7B: decode, eventi di reclaim, PSI, memoria libera dell'host. Per i MoE la stessa prova decide se la page cache del kernel può fare da cache degli esperti (sezione 7.3).

**Accettazione.** Byte letti e digest identici; test di guasto su letture parziali, disallineamento e filesystem senza O_DIRECT (errore esplicito); nessuna regressione > 5% sul percorso buffered.

### Subito dopo le prime tre

| Candidato | Dipende da | Nota |
|---|---|---|
| Cache bounded della parte condivisa e degli esperti | M11, traccia di routing | La leva principale dei MoE dopo la lettura selettiva |
| GEMM su GPU a tile per prefill e verifica | O2b | Sposta `T*` da ≈ 30–70 a ≈ 660+ token per scansione |
| Decode multi-token per scansione (verifica greedy-esatta di k token proposti) | O2b | L'unico modo di scendere sotto una scansione per token nei densi |
| Backend BLAS FP32 per il GEMM | O2b | Solo come contratto FP32 dichiarato, con dipendenza esterna da approvare |

## 6. Specifica implementativa dell'Incremento 10

Corrisponde alla Issue #4 (M10). Questa sezione è una specifica: nulla è stato eseguito, scaricato o modificato.

### 6.1 Claim, non-claim e principio guida

**Claim da dimostrare.** Un LLM pubblico addestrato, in BF16 originale, esegue prefill e decode reali con **byte unici di pesi effettivamente usati superiori alla RAM fisica totale dell'host** misurata nella stessa prova (16.900.739.072 B all'ultimo audit), con parità numerica completa contro un oracolo indipendente.

**Non-claim.** Prestazioni pratiche; letture NAND fredde isolate; contesti lunghi; MoE; qualunque ottimizzazione.

**Principio guida: differenza zero nel runtime.** L'incremento 10 cambia una sola variabile rispetto al 9: la taglia del checkpoint. `inwime_runtime/` e `src/` restano identici byte per byte al commit di base; si aggiungono solo strumenti, test, documenti ed evidenze. Così un eventuale FAIL è attribuibile alla scala e non a codice nuovo, e le ottimizzazioni della sezione 5 partono da una baseline verificata su tre taglie. **HIGH CONFIDENCE** che sia possibile: sezione 6.4.

### 6.2 Selezione e verifica del checkpoint

Criteri, in ordine di priorità:

1. Pubblico, non gated, licenza permissiva, revisione immutabile fissata per SHA.
2. Tutti i tensori BF16 originali (`safetensors`), nessuna conversione.
3. Architettura già supportata dall'adapter Qwen2 **senza modifiche**: `Qwen2ForCausalLM`, head non condivisa, niente sliding window, niente `rope_scaling`.
4. Byte unici attivi > RAM fisica host con margine ampio; idealmente anche > RAM host + swap WSL, per togliere argomenti a un'obiezione sul paging.
5. Disco: download ≤ spazio libero reale su C: meno un margine di 32 GiB.
6. Tempo totale di prova compatibile con una sessione (ordine di un'ora).

Candidati, con fatti pubblici verificati oggi (API e `config.json` Hugging Face; dettagli in Appendice B):

| Candidato | Revisione | Parametri (tutti BF16) | Payload | Shard / tensori attesi | Byte unici attivi (limite inferiore) | Rapporto su RAM host | Esito |
|---|---|---:|---:|---|---:|---:|---|
| **Qwen/Qwen2.5-14B** | `97e1e763…298c9` | 14.770.033.664 | 29,54 GB | 8 / 579 | **27.982.931.968 B** | **1,66×** | **Scelta primaria** |
| Qwen/Qwen2.5-32B | `1818d358…f003` | 32.763.876.352 | 65,53 GB | 17 / 771 | 63.970.617.344 B | 3,79× | Estensione facoltativa (6.14) |
| Qwen/Qwen2.5-72B | `efba10c8…d4cb` | 72.706.203.648 | 145,4 GB | 37 / 963 | ≈ 142,9 GB | 8,5× | Escluso: non entra in 144,8 GB liberi; licenza «other» |
| Qwen/Qwen3-30B-A3B | `ad44e777…5d39` | 30.532.122.624 | 61,06 GB | 16 / 18.867 | dipende dal routing | — | Escluso: richiede un adapter nuovo (M11) |

Per Qwen2.5-14B (48 layer, hidden 5120, FFN 13.824, 40 teste, 8 teste KV, `head_dim` 128, vocabolario 152.064, `rms_norm_eps` 1e-5, `rope_theta` 1e6, Apache-2.0, non gated):

- Conteggio ricostruito dalla geometria: 48 × 275.268.608 + 2 × 778.567.680 + 5.120 = 14.770.033.664, **uguale** al valore pubblicato.
- Byte attivi = payload − embedding = 29.540.067.328 − 1.557.135.360 = **27.982.931.968 B**, più 10.240 B per ogni riga di embedding distinta.
- Supera la RAM fisica host di 11.082.192.896 B; supera `MemTotal` WSL di 2,44 volte; supera anche RAM host + swap WSL (25.490.673.664 B) di 2,49 GB.
- Byte logici per token: 27.982.942.208 B. Coefficienti GEMV per token: 13.990.625.280.

Questi numeri sono **previsioni da confermare nel preflight** sugli header reali; il criterio D usa solo i contatori della prova.

### 6.3 Preflight (sola lettura, prima di ogni download)

**Rete, senza pesi** (strumento parametrico nuovo, per esempio `tools/acquire10.py`; quello del 9 ha modello, revisione e 4 shard cablati):

1. API della revisione fissata: `sha` uguale, `gated=false`, licenza `apache-2.0`, elenco dei file.
2. `config.json`, `generation_config.json`, tokenizer e index, verificati con l'ID del blob Git.
3. Gli 8 header safetensors via HTTP Range: tutti i tensori BF16; 579 tensori; insieme dei nomi uguale alla mappa di `Qwen2Catalog`; forme attese; `total_size` dell'index uguale alla somma dei payload; digest LFS ufficiali registrati.
4. Ricalcolo indipendente del limite inferiore dei byte attivi e confronto con la RAM host misurata: il criterio deve risultare soddisfacibile **prima** del download.

**Storage:**

5. Spazio libero **reale su C:** (non quello dell'ext4 virtuale): richiesto ≥ dimensione del download + 32 GiB. Oggi: 144,8 GB liberi, download ≈ 29,6 GB, residuo ≈ 115 GB.
6. Una sola copia, nessuna cache Hugging Face, nessun repack; ricontrollo dello spazio ogni 256 MiB; `.part` rinominato solo dopo dimensione e SHA256 calcolato uguale a quello ufficiale; **digest calcolato registrato** accanto a quello ufficiale.

**Memoria e piano:**

7. RAM fisica host via CIM (`TotalPhysicalMemory`, `FreePhysicalMemory`) e uso del pagefile Windows, in sola lettura; `MemTotal`, `MemAvailable`, swap del guest; `.wslconfig` solo letto.
8. Piano delle arene calcolato sui soli metadati con `Qwen2Runtime.minimum_plan`. Stima aritmetica a capacity 16: ≈ 80,6 MB in sync (logits 19,5 MB, hidden 31,5 MB, due banchi KV 25,2 MB, pesi 2,1 MB, staging 0,5 MB, attivazioni 0,4 MB, metadati ≈ 1,5 MB), contro 43,7 MB del 7B.
9. **`RLIMIT_AS` = 384 MiB, non 256.** Al 9 la pipeline ha avuto `VmPeak` 217,6 MB con un piano da 46,3 MB; con ≈ 37 MB in più il picco stimato (≈ 255 MB) è a ridosso di 256 MiB (268,4 MB). Il valore va preregistrato con questa derivazione. Le soglie su anon+kernel e `VmHWM` restano 268.435.456 B (attesi ≈ 150–165 MB).
10. Delega cgroup v2 dell'utente con controller `memory`, come al 9.
11. `MemAvailable` ≥ 4 GiB prima dei run del runtime; ≥ 6,5 GiB prima dell'oracolo FP32.

**Autorizzazione:** il download di ≈ 29,6 GB richiede l'approvazione esplicita del proprietario (AGENTS.md) dopo la presentazione del preflight.

### 6.4 Adapter necessari e riuso del runtime

**Nessun adapter nuovo.** Verifica statica delle condizioni di `qwen2.py::Qwen2Catalog` contro il `config.json` di Qwen2.5-14B:

| Condizione nel codice | Valore 14B | Esito atteso |
|---|---|---|
| `model_type == 'qwen2'`, `architectures == ['Qwen2ForCausalLM']`, `hidden_act == 'silu'` | sì | accettato |
| `tie_word_embeddings is False` | false | accettato |
| `use_sliding_window` falso, niente `use_mrope`, `rope_scaling` assente, `layer_types` assente | sì | accettato |
| `width % heads == 0`, `heads % kv_heads == 0`, `dim` pari, `head_dim` assente o uguale | 5120/40 = 128; 40/8 = 5 | accettato |
| `layers*12+3 == len(tensors)` | 579 | da confermare sugli header |
| Limiti della sorgente: ≤ 64 shard, ≤ 4096 tensori, header ≤ 8 MiB | 8 shard, 579 tensori | entro i limiti |

Due valori sono **nuovi rispetto a tutto ciò che è stato provato**: rapporto GQA 5 (finora 7 e 3) e `rms_norm_eps` 1e-5 (finora 1e-6), oltre a 48 layer. Prima del modello grande: una fixture Qwen2 sintetica con rapporto GQA 5 ed eps 1e-5, confrontata con il riferimento indipendente e con Transformers (pochi MB, secondi).

File **aggiunti** (nessun file preesistente modificato, così il gate 9 resta valido):

- `tools/acquire10.py` (preflight e acquisizione parametrici);
- `tools/benchmark_increment10.py` (worker: come il 9, più RAM host, contatori del dispositivo a blocchi del guest, PSI);
- `tools/oracle10.py` (oracolo a blocchi con head a chunk e raccolta delle sole righe di embedding, sezione 6.8);
- `tools/fault_increment10.py`, `tools/report_increment10.py` (gate), `tools/audit_increment10.py`;
- `tests/test_increment10.py`, `docs/RUNTIME_CPU_INCREMENT_10.md`, `results/increment10/`.

Condizione del gate: `git diff --quiet <base> -- inwime_runtime src` deve avere esito zero.

### 6.5 cgroup, RAM host, page cache e swap

- **Scope transitorio** identico al 9: `systemd-run --user --scope --collect -p MemoryMax=2147483648 -p MemorySwapMax=0 -p OOMPolicy=continue`; il worker verifica appartenenza e limiti **prima** di aprire tokenizer o pesi; finalizzatore nello scope per `memory.peak` ed eventi.
- **Che cosa prova che cosa.** Il superamento della RAM host è una proprietà **aritmetica** dei byte unici usati (27,98 GB > 16,90 GB): in nessun istante possono essere tutti residenti, in nessuna cache di nessun livello. Lo scope aggiunge l'attribuzione: al più 2 GiB di page cache addebitabile alla prova.
- **Swap.** `memory.swap.max = 0` nello scope. I pesi non sono mai memoria anonima nel runtime (anon di picco ≈ 105 MB al 9), quindi lo swap non può contenerli. Si registrano comunque: swap del gruppo, `VmSwap` del worker, delta dello swap globale del guest, uso del pagefile Windows prima e dopo.
- **RAM host.** `TotalPhysicalMemory` misurata all'inizio di **ogni** run e legata al rapporto; il criterio D usa quel valore, non uno storico.
- **Nessuna modifica** a `.wslconfig`, swap, sysctl; nessun `drop_caches`; nessun `sudo`.
- **Guardia:** se `MemAvailable` < 768 MiB il supervisore esegue `cgroup.kill` del solo scope.

### 6.6 Protocollo warm/cold

Definizioni preregistrate:

- **Cold-guest (C):** tutti gli shard rimossi dalla page cache con `POSIX_FADV_DONTNEED` mirato e residenza verificata con `fincore` (≤ 1% per shard, atteso 0); worker nello scope da 2 GiB. Ogni byte arriva dal dispositivo a blocchi virtuale. Non è una dichiarazione di NAND fredda: cache di Windows, controller e SSD non sono controllate. Con 28 GB unici e ≈ 0,5–1 GB liberi sull'host, la cache dell'host non può comunque servire la prova.
- **As-is (W):** stesso comando subito dopo, senza evizione. La cache residua del guest serve una parte delle letture. Con 28 GB > `MemTotal` non può esistere un run «tutto caldo»: W misura la frazione residente a regime e il suo effetto, ed è **solo informativo**.

Sequenza (processi nuovi, uno alla volta, nessun altro carico):

| Run | Condizione | Modalità | Catture | Scopo |
|---|---|---|---|---|
| R1 | C | sync | sì | Prova principale: forward, contatori, parità |
| R2 | C | pipeline | digest | Identità byte per byte con R1 |
| R3 | C | sync | digest | Ripetizione: variabilità (al 9 c'era una sola coppia) |
| R4 | W | sync | digest | Attribuzione: mostra che senza evizione il criterio C non è soddisfatto |

Per ogni run: ordine e orari, stato di alimentazione letto da Windows (`BatteryStatus`), assenza di altri benchmark. Temperatura SSD e frequenze: dichiarate non disponibili se non leggibili.

### 6.7 Misure di I/O attribuibili

Gerarchia dei contatori, dal più attribuibile al più globale:

| Livello | Contatore | Attribuzione |
|---|---|---|
| L0 | Byte logici richiesti, per tensore (contatori del runtime) | esatta, per tensore e per fase |
| L1 | `/proc/PID/io` `read_bytes` del worker | processo |
| L2 | `memory.stat` dello scope (`file`, `workingset_refault_file`), eventi `max`, **PSI** `memory.pressure` e `io.pressure` dello scope | cgroup |
| L3 | **`/sys/block/<dev>/stat`** del disco virtuale che ospita il checkpoint: settori letti (nuovo rispetto al 9, leggibile senza privilegi) | disco del guest, globale al guest |
| L4 | Contatori Windows del disco fisico (`Win32_PerfRawData_PerfDisk_PhysicalDisk`) | host, globale |

Regole di coerenza preregistrate per i run C: L1 ≥ 90% di L0 sull'intera inferenza **e** su ciascun decode; ΔL3 ≥ L1 (a meno dell'arrotondamento a settore); ΔL4 ≥ 90% di L1 se il contatore è raggiungibile, altrimenti la sola voce host è NOT PROVEN senza invalidare C. `io.stat` per cgroup resta non disponibile (controller `io` non delegato): dichiararlo. Non presentare `read_bytes` come prova di letture NAND.

### 6.8 Oracolo indipendente e contratti numerici

**Soglie invariate e preregistrate:** FP32 `|err| ≤ 2e-3 + 2e-4·|ref|`; FP64 `|err| ≤ 2e-5 + 2e-6·|ref|`, per ogni elemento. Mai allargate dopo le misure.

**Contratti:**

| Contratto | Ruolo | Esito richiesto |
|---|---|---|
| `stock_fp32`: Transformers standard in FP32, forward upstream non modificato | Parità pratica indipendente | PASS obbligatorio |
| `full64`: grafo upstream con RMSNorm, RoPE e softmax in FP64 (profilo dell'incremento 7) | Parità matematica | PASS obbligatorio |
| `stock_fp64`: diagnostico | Trasparenza | Eseguito e riportato grezzo, mai riscritto |

**Copertura obbligatoria:** tutti i 152.064 logits di ogni posizione (8 × 152.064); residui di tutti i 48 layer (48 × 8 × 5.120); K post-RoPE e V di tutti i layer dopo ogni fase; argmax, top-10, greedy upstream uguale alla continuazione del runtime.

**Problema di memoria dell'oracolo e soluzione.** La tecnica del 9 materializza un modulo alla volta. Per il 14B head ed embedding in FP64 valgono 6,23 GB ciascuno, più 1,56 GB dell'originale BF16 durante la promozione: ≈ 8,2 GB di picco, oltre il `MemAvailable` tipico (7,7–8,8 GB). Soluzione:

- `stock_fp32`: tecnica del 9 invariata (picco stimato ≈ 5,1 GB).
- Contratti FP64: `tools/oracle10.py` sostituisce due sole operazioni lineari con equivalenti a memoria limitata: (a) embedding → raccolta delle sole righe dei token dal file; (b) `lm_head` → prodotto a chunk di righe (per esempio 16.384 righe, 671 MB per chunk). Tutto il resto del forward resta upstream. Picco stimato ≈ 3,2 GB.
- **Controllo preliminare di equivalenza, obbligatorio e preregistrato:** (i) sulla fixture sintetica, uscita bit per bit uguale a `from_pretrained`; (ii) su Qwen2.5-7B, `oracle10` in `full64` confrontato con la cattura dell'oracolo del 9 (legata per SHA): scarto massimo sui logits ≤ 1e-9 e argmax identici. Se il controllo fallisce, il contratto `full64` del 14B è NOT PROVEN.

**Indipendenza.** L'oracolo non importa InWiMe e riceve solo la sequenza di token; legge gli shard originali con la libreria `safetensors`; versioni fissate (Transformers 4.57.3, torch 2.9.1+cpu). Limite dichiarato, come al 9: le funzioni del profilo `full64` sono dello stesso autore del runtime; l'indipendenza piena viene da `stock_fp32`.

**Rischio numerico da dichiarare prima.** A 48 layer l'errore dell'oracolo FP32 si accumula più che a 28. Al 9 lo scarto massimo sui residui era 3,95·10⁻³ e passava grazie al termine relativo. Il margine reale non è registrato. Due regole preregistrate:

1. Registrare per ogni layer l'**utilizzo della soglia** `max |err| / (atol + rtol·|ref|)`. È un diagnostico, non un criterio.
2. Se `stock_fp32` fallisce mentre `full64` passa: F è **FAIL sul contratto FP32**, riportato con la curva di crescita dell'errore per layer. Nessuna soglia viene modificata nell'incremento; un nuovo contratto potrà nascere solo in un incremento successivo.

### 6.9 Forward, prefill, decode e KV

- Ingresso: `The capital of France is` (atteso `[785, 6722, 315, 9625, 374]`, stesso tokenizer del 7B: da verificare contro AutoTokenizer su quattro testi come al 9).
- Prefill di 5 token, poi 3 decode con reinserimento greedy; capacity 16; tile 262.144; integrità `full`; kernel `c-fp64-ordered`.
- Asserzioni per fase: lunghezze KV pubblicate 5/6/7/8; generazioni 1–4; prefisso KV invariato a ogni passo; dimensioni delle arene invariate; FD e arene rilasciati alla chiusura.
- Copertura esatta: per ciascuno dei 579 tensori, byte letti = forma ufficiale × 8 posizioni (per l'embedding le sole righe consultate); coefficienti GEMV = 8 × 13.990.625.280 = 111.925.002.240; nessun layer saltato.
- Byte unici attivi calcolati dai contatori per tensore (unione, alias contati una volta) **e** ricalcolati dagli header ufficiali: devono coincidere.
- Sync e pipeline: digest di logits, residui e KV identici.

### 6.10 Test negativi e guasti

| # | Prova | Dove | Esito atteso |
|---|---|---|---|
| N1 | Budget inferiore di 1 byte al piano | 14B reale | `BudgetError` prima di qualunque lettura del payload |
| N2–N4 | Errore I/O, lettura corta, cancellazione alla lettura 100 di un decode, sync e pipeline | 14B reale, nello scope, `integrity='metadata'` | Eccezione attesa; banco e descriptor pubblicati intatti; sessione fail-stop; slot drenati; FD e arene chiusi |
| N5 | SHA errata nel descriptor; file modificato dopo l'apertura | fixture | `CheckpointError`, nessun forward |
| N6 | OOM controllato in uno scope da 32 MiB | SmolLM2-1.7B | `oom_kill = 1`, exit 137, nessun rapporto di successo |
| N7 | Worker lanciato fuori dallo scope atteso | 14B, solo metadati | Rifiuto prima di aprire i pesi |
| N8 | Attribuzione: R4 (as-is) al posto di R1 | mutazione del gate | C non PASS |
| N9 | Falsificazione del claim: byte unici del 7B; RAM host mancante; RAM host sostituita da `MemTotal` | mutazioni del gate | D rispettivamente FAIL, NOT PROVEN, FAIL |
| N10 | Geometria nuova rifiutata o accettata correttamente (conteggio tensori, head condivisa, bias mancante) | fixture GQA 5, eps 1e-5 | Come da adapter |
| N11 | Sensibilità dell'oracolo: un peso perturbato in una copia della fixture | fixture | La parità deve fallire |
| N12 | Guardia di spazio dell'acquisizione | test unitario | Rifiuto sotto margine |

Il gate dell'incremento nasce **fail-closed** e con le sue mutazioni: file o campo mancante → NOT PROVEN; fatto contraddetto o JSON non valido → FAIL; nessun booleano derivato accettato senza ricalcolo dai dati grezzi; evidenze scritte già sanificate; emendamenti in file separati, in sola aggiunta, con SHA fissato. Sono le lezioni delle due revisioni del 9.

### 6.11 Criteri PASS, FAIL, NOT PROVEN

| Condizione | PASS se | FAIL se | NOT PROVEN se |
|---|---|---|---|
| **A — Integrità** | SHA256 calcolato in acquisizione, SHA integrale del runtime all'avvio di R1–R3 e re-hash finale uguali ai digest LFS ufficiali, index incluso | un digest diverso | digest calcolato non registrato; re-hash assente |
| **B — Memoria** | scope da 2 GiB verificato dall'interno prima dei pesi; `memory.peak` ≤ limite; `oom` = `oom_kill` = 0; swap del gruppo 0; anon+kernel campionato ≤ 256 MiB; `VmHWM` ≤ 256 MiB | OOM, swap nel gruppo, soglia superata | scope non verificato, finalizzatore assente |
| **C — I/O** | run C: residenza dopo evizione ≤ 1% per shard; L1 ≥ 90% di L0 su inferenza e su ogni decode; coerenza L3; R4 riportato come informativo | letture di blocco < 90% in un run dichiarato C; residenza > 1% | contatori mancanti; evizione non verificabile |
| **D — Oltre RAM host** | byte unici attivi > `TotalPhysicalMemory` misurata nello stesso run **e** > `MemTotal`; valore dai contatori uguale al ricalcolo dagli header; su un run C realmente letto | byte unici ≤ RAM host; ricalcolo discordante | RAM host non misurabile nel run |
| **E — Forward reale** | 5 + 3 posizioni, 48 layer, contatori dei 579 tensori esatti, KV e generazioni coerenti, rilascio, pipeline identica nei byte | copertura inesatta, digest sync/pipeline diversi | una fase mancante |
| **F — Correttezza** | `stock_fp32` e `full64` PASS su copertura completa, legati a revisione, sequenza e SHA delle catture; greedy upstream uguale; `stock_fp64` eseguito e riportato grezzo; controllo di equivalenza di `oracle10` superato | una violazione in un contratto obbligatorio; diagnostico riscritto | contratto non eseguibile entro la memoria; controllo di equivalenza fallito |
| **G — Risorse** | tutti i run supervisionati; `oom_kill` globale invariato salvo N6 attribuito; oracoli con `VmSwap` 0; nessun intervento della guardia; crescita dello swap globale nella finestra dei run del runtime ≤ 256 MiB | OOM di sistema non attribuito | guardia intervenuta; campioni mancanti |
| **H — Regressioni** | suite completa PASS senza skip; gate 7, 8 e 9 ricalcolati PASS; `inwime_runtime/` e `src/` invariati; nessun file preesistente modificato fuori dall'elenco intenzionale | regressione; runtime modificato | log di test assente |

Precondizioni per ogni PASS: protocollo preregistrato con SHA fissato prima di qualunque download o misura; evidenze sanificate alla scrittura. Un FAIL resta FAIL anche con precondizioni mancanti.

### 6.12 Costi stimati

Stime dai costi unitari misurati al 9 (0,94 s/GB per token; SHA a 1,23 GB/s; oracoli scalati per i byte). **HIGH CONFIDENCE** per il runtime, **HYPOTHESIS** per gli oracoli.

| Voce | Tempo | Letture | Scritture |
|---|---:|---:|---:|
| Download del checkpoint | dipende dalla rete (≈ 40 min a 100 Mbit/s) | — | 29,6 GB |
| Un run del runtime: SHA 24 s + prefill ≈ 131 s + 3 decode ≈ 79 s | ≈ 4 min | ≈ 253 GB | < 50 MB |
| R1–R4 | ≈ 16–17 min | ≈ 1,0 TB | < 0,2 GB |
| Oracolo `stock_fp32` (4 chiamate) | ≈ 8 min | 118 GB | ≈ 50 MB |
| Oracolo `full64` (4 chiamate) | ≈ 14 min | 118 GB | ≈ 50 MB |
| Diagnostico `stock_fp64` (1 chiamata) | ≈ 5 min | 30 GB | ≈ 50 MB |
| Controllo di equivalenza di `oracle10` su 7B | ≈ 7–8 min | 61 GB | trascurabile |
| Sonde di guasto N1–N4, OOM N6 | ≈ 4 min | ≈ 170 GB (sei prefill da 1 token) | trascurabile |
| Re-hash finale | ≈ 1 min | 30 GB | — |
| **Totale, escluso il download** | **≈ 1 h** | **≈ 1,5 TB** | **≈ 30 GB (di cui 29,6 il checkpoint)** |

Disco dopo la prova: ≈ 115 GB liberi su C:. Le letture non consumano in modo significativo la durata della NAND; le scritture sono il solo download. La latenza attesa in decode è ≈ 26 s per token.

### 6.13 Condizioni di arresto e rollback

- **Prima del download:** un tensore non BF16, insieme dei tensori diverso dalla mappa, `total_size` discordante, revisione o licenza cambiate, spazio sotto margine → stop, esito NOT PROVEN, nessun download.
- **Durante il download:** spazio sotto margine; digest diverso (il `.part` non viene rinominato); server senza `Range` → stop.
- **Durante i run:** guardia `MemAvailable` → `cgroup.kill`, run invalidato, nessuna ripetizione senza analisi; `oom_kill` → FAIL registrato; decode oltre 3 volte la previsione (≥ 80 s per token) → pausa e indagine su termica e I/O prima di proseguire.
- **Dopo:** un FAIL numerico non si corregge cambiando soglie o runtime nello stesso incremento; le evidenze si conservano, comprese quelle rosse e i run pilota.
- **Repository:** solo file aggiunti; il rollback è la loro rimozione. Nessun reset distruttivo, nessuna modifica a `.wslconfig`, swap o dischi. La cancellazione del checkpoint (29,6 GB) è una decisione del proprietario.

### 6.14 Estensione facoltativa: Qwen2.5-32B

Stessa specifica, stessi strumenti. 63,97 GB attivi (3,79× la RAM host); 17 shard, 771 tensori; decode ≈ 60 s per token; un run ≈ 9 min; oracoli ≈ 18 + 31 + 11 min; download 65,5 GB. Con il 14B già presente resterebbero ≈ 50 GB liberi su C:, sotto un margine prudente: conviene solo liberando prima spazio (decisione del proprietario) e solo se serve un titolo più forte. Non aggiunge informazione architetturale rispetto al 14B. **Raccomandazione: non includerla nell'incremento 10.**

## 7. Evoluzione M11–M12

Riferimenti: Issue #5 (M11), #6 (M12), #7 (M13), #8 (M14), #9 (M15+), #11 e #12 (ricerca). La proposta di ordinamento è nella sezione 9: una parte di M12 (motore a numerica invariata, O1 e O2) conviene **prima** di M11.

### 7.1 MoE reale con routing esatto (M11)

**Candidato: Qwen/Qwen3-30B-A3B**, revisione `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, Apache-2.0, non gated, 30.532.122.624 parametri tutti BF16, 16 shard (fatti verificati oggi). Motivi: è BF16 originale, ha la famiglia più vicina all'adapter Qwen2 esistente, e i suoi 61 GB superano di 3,6 volte la RAM fisica con soli 6,08 GB attivi per token. Spazio: dopo il 14B restano ≈ 115 GB su C:; con 61 GB in più ne resterebbero ≈ 54. È una decisione di storage del proprietario (per esempio liberare i 25,8 GB di checkpoint sintetici).

**Differenze rispetto a Qwen2 da implementare**, tutte da confermare sul sorgente Transformers della versione fissata prima di scrivere codice (**REQUIRES EXPERIMENT**: qui sono dedotte dal `config.json` e dalla conoscenza generale della famiglia, non lette nel sorgente):

| Aspetto | Qwen2 (oggi) | Qwen3-MoE | Impatto |
|---|---|---|---|
| `head_dim` | implicito `width / heads` | esplicito: 128 con hidden 2048 e 32 teste (Q ha 4096 uscite) | `LlamaSpec` va generalizzato; oggi un `head_dim` diverso è rifiutato |
| Bias di attenzione | Q/K/V con bias | `attention_bias: false` | mappa dei tensori |
| Norme su Q e K per testa | assenti | presenti: due vettori da 128 per layer, confermati dal conteggio esatto dei parametri (Appendice C) | nuovo operatore; posizione rispetto alla RoPE da leggere nel sorgente upstream |
| FFN | densa SwiGLU | 128 esperti SwiGLU (768 intermedi), top-8, `norm_topk_prob: true` | router, selezione, combinazione |
| Tensori | 12 per layer + 3 | 393 per layer + 3 = 18.867 | limiti di catalogo (L8) |

**Semantica del router da riprodurre esattamente:** logits = proiezione lineare del residuo normalizzato; softmax su **tutti** gli esperti; top-k; rinormalizzazione sui selezionati; somma pesata delle uscite degli esperti. Punti delicati:

- Il decoder sintetico attuale applica la softmax ai soli selezionati: è matematicamente equivalente alla sequenza softmax → top-k → rinormalizzazione, ma non identica negli arrotondamenti. Va riprodotta la sequenza upstream.
- L'ordine di somma degli esperti (per indice o per rango) va copiato da upstream.
- Se upstream calcola la softmax del router in FP32 anche in un modello FP64, il diagnostico `stock_fp64` fallirà come per l'attenzione; il profilo `full64` dovrà coprire anche il router.

**Protocollo di verifica del routing** (risposta a L14), da preregistrare:

1. **Arbitro primario:** runtime FP64 contro oracolo `full64`. Per ogni (layer, posizione): insieme degli esperti identico, pesi entro la soglia FP64. Con scarti di 10⁻¹¹ un'inversione è possibile solo su pareggi esatti.
2. **Margine registrato:** per ogni (layer, posizione) il distacco fra il k-esimo e il (k+1)-esimo logit del router.
3. **Contratto FP32:** soglia δ sul margine fissata prima della prova. Insieme diverso a margine ≥ δ → **FAIL**. Insieme diverso a margine < δ → «divergenza sensibile al margine», riportata, contratto FP32 **NOT PROVEN** per quel run, mai PASS. I prompt di prova vanno scelti in anticipo con margine minimo ≥ δ.
4. **Pareggi esatti:** regola dichiarata e provata su una fixture costruita con pareggi; se il comportamento di upstream non è documentato, i prompt non devono contenerne e lo si verifica.
5. Una continuazione greedy coincidente è necessaria, non sufficiente (come già scritto nella Issue #11).

**Oracolo.** La tecnica a moduli del 9 materializzerebbe un layer intero: 623 milioni di parametri, 5,0 GB in FP64 più 1,25 GB di originale. Va portata a **grana di esperto**: parte condivisa del layer più i soli esperti effettivamente chiamati dal forward upstream. Prima del modello grande: fixture Qwen3-MoE sintetica, uguale bit per bit a `from_pretrained`.

**Contabilità richiesta** (mai confondere): parametri totali; attivi per token; esperti distinti toccati nella prova; byte letti; riletture; byte non letti perché non selezionati (deve essere **zero byte** per ogni esperto non selezionato: è la prova della lettura selettiva).

**Accettazione M11:** preflight sugli header; adapter che rifiuta le varianti non supportate; routing identico all'arbitro; parità completa di logits, residui, KV; prefill e decode; contatori per esperto; guasti e cancellazione durante la lettura di un esperto con KV pubblicato intatto; prima versione **senza cache**. Costo atteso del decode con il runtime attuale: ≈ 5,7 s per token (**HIGH CONFIDENCE** sui costi unitari).

### 7.2 Streaming selettivo degli esperti

- **Unità di lettura:** il tensore. Un esperto di Qwen3-30B-A3B è fatto di tre tensori da 3,0 MiB; per layer servono 24 letture (≈ 28,3 MB), note solo dopo il router.
- **Nessun repack.** Betelgeuse impacchetta gli esperti in un file a parte con stride di 4 KiB; InWiMe deve restare sugli shard originali. Il preflight misura dagli header la **località**: distanza fisica fra gate, up e down dello stesso esperto. Se sono adiacenti basta una lettura per esperto; se non lo sono restano tre letture grandi, comunque efficienti. Un eventuale indice derivato è un file di soli metadati, mai una copia dei pesi.
- **Letture multi-intervallo** con coda profonda (24 intervalli per layer): è l'interfaccia che O3 introduce. Assunzione A3 da misurare.
- **Amplificazione:** con I/O buffered il readahead può leggere oltre l'esperto richiesto; va misurato il rapporto fra byte fisici e byte utili, e usato `POSIX_FADV_RANDOM` o O_DIRECT se supera una soglia dichiarata.
- **Prefill:** raggruppare i token del chunk per esperto, così ogni esperto è letto una volta per layer e applicato ai suoi token. I byte per chunk sono l'unione degli esperti (sezione 4.6): conviene un chunk grande.
- **Regola di correttezza invariabile:** la residenza non decide mai il routing. Nessuna sostituzione di esperti, nessun salto dei ranghi bassi, nessun contributo azzerato. Le varianti di Strata che lo fanno (`native_route_resident`, tail-skip) sono escluse.

### 7.3 Cache e scheduling bounded (M12)

**Prima la misura.** Nessuna politica va scelta senza una **traccia di routing reale**: (layer, posizione, esperti, margini) su prompt diversi; da essa si ricava la curva «byte di esperti distinti toccati in una finestra di N token». È un prodotto di M11 a costo quasi nullo.

**Disegno minimo, sotto l'unica autorità del coordinatore:**

- La cache conserva **byte BF16 originali**, con chiave (identità dello shard, offset, lunghezza); unità = tensore.
- Due classi: *fissata* (parte condivisa: per Qwen3-30B-A3B 2,46 GB, il 40% dei byte per token) e *sostituibile* (esperti).
- Budget fisso nel piano (categoria `weights`); una lease blocca la voce; si espelle solo ciò che non è né fissato né in uso; l'ammissione avviene dopo l'uso, dallo staging appena letto.
- Politiche da confrontare **a pari budget, una alla volta**: frequenza con decadimento, CLOCK. Metriche: hit e miss per byte, byte risparmiati, espulsioni, distribuzione per layer.
- Prefetch: solo suggerimento, mai autorità; con budget proprio; cede sempre alla domanda. Da valutare dopo la cache: il guadagno è limitato dal tempo di calcolo che può sovrapporre.

**Decisione aperta: cache esplicita o page cache.** Con I/O buffered sotto un cgroup da X GiB, la page cache del kernel **è già** una cache degli esperti a costo di codice zero; Betelgeuse dichiara di usare letture buffered proprio per questo. Contro: nessun controllo sulla politica, grana di 4 KiB, dipendenza dal cgroup, pressione sull'host WSL2. La scelta va fatta con un A/B sullo stesso checkpoint (ipotesi H01): buffered a cgroup 2/4/8 GiB contro O_DIRECT con cache esplicita a pari byte. **REQUIRES EXPERIMENT.** È la decisione D4 della sezione 10.

**Scheduling.** Con il coordinatore a grana di operazione, lo «scheduler» è il programma del layer più due regole: le letture obbligatorie precedono le speculative; nessuna prenotazione speculativa può impedire a un'operazione obbligatoria di completarsi. Non serve altro finché esiste una sola sessione.

### 7.4 KV per contesti lunghi

1. **Append-only a banco unico** (O2a): stessa atomicità, metà memoria, nessuna copia.
2. **Attenzione nativa a blocchi** sulle posizioni: nessuna matrice T × T materializzata; softmax in linea.
3. **dtype del KV come contratto esplicito:** FP64 (diagnostico, parità `full64`), FP32 (parità `stock_fp32`), BF16 nativo (semantica upstream in BF16). Ognuno con il proprio oracolo; nessuna conversione implicita.
4. **Budget:** posizioni che entrano in 4 GiB di KV FP32: ≈ 37 mila per il 7B, ≈ 22 mila per Qwen3-30B-A3B (197 KB per posizione), ≈ 11 mila per il 14B, ≈ 6,5 mila per il 72B.
5. **Oltre il budget: KV paginato per layer su NVMe.** L'accesso è sequenziale per layer, quindi rientra nello schema di streaming. Costa una rilettura del KV a ogni token (per il 72B a 32 mila posizioni: 21,5 GB, il 15% della scansione dei pesi) e **scritture** su disco limitate ma continue: richiede uno scratch esplicito, con tetto e autorizzazione (AGENTS.md).
6. **Prefill a chunk:** T_c limitato dal budget delle attivazioni; ogni chunk pubblica atomicamente le sue posizioni.

Accettazione: parità con l'oracolo a contesto ≥ 4.096 su un modello piccolo; curva misurata di costo e memoria in funzione di T; atomicità sotto guasto a metà chunk.

### 7.5 Shard numerosi e multi-NVMe

- **Store di shard** (prima estrazione modulare, sezione 3.4): possiede FD, firme, stato di integrità, backend di lettura e mappa shard → dispositivo.
- **Limiti parametrici** con preventivo di heap: tensori, shard, nodi JSON. Casi reali: 18.867 tensori (Qwen3-30B-A3B), 163 shard (DeepSeek-V3).
- **Integrità lazy:** manifest verificato una volta e legato alla firma del file, più digest per chunk verificati alla prima lettura; `full` resta disponibile ed è obbligatoria nei gate.
- **FD:** uno per shard regge fino a migliaia di shard (limite del processo qui: 1.048.576); altrove serve un tetto con apertura su richiesta, conservando l'identità verificata.
- **Multi-NVMe:** collocare shard diversi su dispositivi diversi, una coda per dispositivo, budget di byte in volo per dispositivo. La banda si somma solo se i dispositivi non condividono bus o controller. Niente striping, replica o migrazione. Sul Katana oggi non è esercitabile: un solo slot, e il percorso USB misurato vale ≈ 0,45 GB/s.

### 7.6 GPU facoltativa senza layer in VRAM (M13)

- **Dove serve.** Non nel decode: è limitato dall'I/O, e V2/M1a misurano un guadagno wall del 2–12%. Serve dove l'intensità aritmetica è alta: **prefill e verifica a batch** (GEMM con T colonne) e, più avanti, attenzione a contesto lungo.
- **Come.** Stesse tile del percorso CPU: il blocco BF16 letto dal disco va in VRAM (H2D misurato a 12,9 GB/s, più della banda NVMe), viene moltiplicato per le attivazioni del chunk, torna il risultato. VRAM necessaria: due tile (64–256 MiB), attivazioni (T × hidden) e workspace. Nessun layer residente.
- **Contratto numerico esplicito, due opzioni:** (a) cuBLAS BF16 × BF16 con accumulo FP32, già misurato in V2; le attivazioni vengono arrotondate a BF16, quindi il riferimento è l'inferenza upstream in BF16; (b) tile espansa a FP32 sul dispositivo e GEMM FP32, riferimento `stock_fp32`, doppia VRAM per tile e FLOP inferiori (**HYPOTHESIS**).
- **Regole.** Import e inferenza CPU funzionano senza driver; la scelta CPU/GPU avviene prima di modificare lo stato; un errore GPU in volo invalida il passo e lascia intatto il prefisso KV (naturale con il KV append-only); budget VRAM riducibile fino a zero.
- **Misura di accettazione:** `T*` reale sulla RTX 3050 per T = 16…1024; il backend si promuove solo per le geometrie in cui il wall completo migliora di almeno il 10%.

### 7.7 API locali e DeepSeek Harness (M14)

- Un solo esecutore, **una sessione alla volta**, coda di ammissione davanti: non servono coordinatori concorrenti.
- `/v1/models`, `/v1/chat/completions` con SSE. I token arrivano a secondi o decine di secondi: servono keep-alive, timeout lunghi lato client, `usage` con tempi di prefill e decode.
- Cancellazione: `cancel_event` esiste già; con la grana a operazione va controllato anche dentro l'esecutore nativo, a ogni blocco, per non aspettare un tensore intero.
- Chat template: quello del `tokenizer_config.json` della famiglia; l'esattezza va provata contro `apply_chat_template` di upstream. Tool call solo dove il modello le supporta.
- Harness resta un client: profilo provider «OpenAI-compatible» su loopback, come quello descritto nell'audit Strata. Nessuna interfaccia agentica propria.
- Prerequisito pratico: O2 (prefill a batch). Senza, un prompt di chat da poche centinaia di token costa ore.

### 7.8 Tecniche di SSD-streaming di altri progetti

Regola seguita: confronto solo su ciò che è stato verificato; nessuna capacità attribuita senza evidenza.

| Progetto | Verifica in questa revisione | Che cosa risulta | Trasferibile a InWiMe (CPU-first, BF16 originale) | Non trasferibile |
|---|---|---|---|---|
| **Betelgeuse** (fork di llama.cpp, Issue #11) | README e `docs/fluxaeros.md` letti oggi. **Codice del pager non esaminato**: la documentazione non indica i file che lo implementano | Dichiarato dagli autori: `experts.bin` con gate/up/down adiacenti e stride 4 KiB, indice ricavato dai metadati; slot VRAM (5.000), victim cache host da 8 GiB, NVMe; letture `io_uring` **buffered**; prefetch speculativo disattivato nella configurazione provata; errore di lettura fatale; «CPU-only non supportato»; prove su RTX 3060 12 GB, 16 GB RAM, contesto 512, un solo slot, pesi IQ2_XS: 13,9–14,3 tok/s senza victim cache e 16,3–22,0 con, «warm repeated-workload» | Contiguità per esperto (da misurare sugli header, senza repack); cache di byte originali; scelta buffered/diretto da misurare; fail-closed sulle letture | I tok/s: un esperto lì pesa ≈ 1,44 MB quantizzato, contro ≈ 9,44 MB in BF16 per Qwen3-30B-A3B; il livello VRAM; il percorso CUDA |
| **Strata** | Non riverificato qui: fonte è l'audit statico interno `STRATA_TECH_AUDIT_2026-11-13.md` (commit `13eedf5`) | Secondo l'audit: lettura in posto di span di esperti, direct I/O con Linux AIO e fusione di intervalli, cache a frequenze decadute, lookahead del router come suggerimento, percorso completo centrato sulla GPU, prove su pesi quantizzati | Principi: span in posto, politica a frequenza con decadimento, lookahead solo come hint, attesa prima del riuso dei buffer | Motore completo; kernel quantizzati; sostituzione di esperti residenti e tail-skip (alterano il routing) |
| **llama.cpp** (mmap e page cache) | Conoscenza generale, non verificata qui | Mapping del file e paging su domanda; nessun budget esplicito; richiede GGUF, cioè una copia convertita | Solo come **baseline di misura** (H01) | Come meccanismo primario (sezione 8) |
| TinyGiant, PR e RFC di llama.cpp citati nella watchlist, Guanaco, Colibri, WARP, MoE-Infinity, AirLLM | **Non verificati** in questa revisione | Solo le descrizioni della watchlist | Restano piste di studio | Nessuna capacità attribuita |

### 7.9 Le tecniche dell'oracolo possono andare in produzione? (Issue #12)

`tools/oracle9.py` contiene tre idee distinte, con risposte diverse.

| Idea dell'oracolo | In produzione? | Motivo |
|---|---|---|
| **Attraversamento layer-major a batch** (tutta la sequenza attraversa un layer mentre i suoi pesi sono disponibili) | **Sì, adottarla** dentro il motore a tile (è O2b) | È ciò che manca al runtime. Evidenza del 9: l'oracolo FP32 fa 4 passate complete in 251 s (≈ 63 s a passata, 2 thread) e una passata serve 5 token come ne servirebbe decine; il runtime spende 66,9 s per 5 token e crescerebbe linearmente |
| **Residenza a moduli** (un layer intero in RAM nel dtype di calcolo) | **No come meccanismo primario** | Il limite di memoria diventa il modulo più grande: head FP32 da 3,1 GB per il 14B e 5,0 GB per il 72B; layer MoE da 2,5 GB (30B-A3B) a decine di GB; picco misurato 3,64 GB (FP32) e 5,83 GB (FP64) contro 125 MB del runtime. Un singolo tensore non può superare la RAM. Utile come braccio di confronto B nella Issue #12 |
| **Kernel PyTorch/BLAS** | **Solo come backend opzionale per il GEMM**, più avanti | Dipendenza pesante (RSS di base > 1 GB), contratto FP32, ordine di riduzione non controllato. Per il GEMV perde contro il kernel fuso |

**Compromessi.**

- *Memoria:* tile = limite indipendente dalla taglia di qualunque tensore; moduli = limite pari al modulo più grande.
- *Velocità:* a pari attraversamento layer-major, il motore a tile con kernel fuso non ha motivo di essere più lento; oggi è più lento sul prefill solo perché è token-major.
- *Indipendenza della verifica:* è il vincolo più importante. Se la produzione adottasse codice o librerie dell'oracolo, gli errori diventerebbero comuni ai due. Va mantenuta la separazione attuale: produzione = parser proprio, kernel propri, semantica riscritta; oracolo = Transformers upstream, libreria `safetensors`, torch. Condividere **l'idea** dell'attraversamento non riduce l'indipendenza; condividere **codice** sì. Se un giorno un backend PyTorch entrasse in produzione, servirebbe un terzo riferimento.

**Raccomandazione per la Issue #12:** adottare l'attraversamento, non il backend; tenere `oracle9.py` e successori fuori dal pacchetto del runtime; eseguire il confronto A/B/C della Issue solo dopo O2b, quando il confronto è fra strategie di residenza e non fra attraversamenti diversi.

## 8. Alternative architetturali scartate, con motivazione

«Scartata» significa: non come percorso primario adesso. Dove un'alternativa resta utile come esperimento o baseline, è indicato.
| # | Alternativa | Perché scartata | Che cosa se ne conserva |
|---|---|---|---|
| S1 | **Streaming a moduli con PyTorch come motore di produzione** (stile oracolo, stile AirLLM) | Limite di memoria pari al modulo più grande (GB, non MB); un tensore non può superare la RAM; dipendenza pesante; azzera l'indipendenza dell'oracolo. Sezione 7.9 | L'attraversamento layer-major; il braccio B del confronto della Issue #12 |
| S2 | **`mmap` e page cache come meccanismo primario** (stile llama.cpp) | Nessun budget esplicito; grana di 4 KiB con readahead poco controllabile sulle letture selettive; pressione sull'host WSL2; richiederebbe GGUF, cioè una copia convertita | Baseline di misura (H01); la page cache come possibile cache degli esperti, da decidere con A/B (7.3) |
| S3 | **Layout derivato dei pesi** (repack contiguo per esperto, copie FP32, GGUF) | Viola «originali immutati, nessuna conversione persistente»; raddoppia il disco che già manca | Al più un indice derivato di soli metadati |
| S4 | **Quantizzazione aggiuntiva per far entrare il modello** | Fuori missione; cambia il modello | Percorso opzionale separato, mai implicito |
| S5 | **Esecutore a grafo incrementale** (alternativa C) adesso | Nessuna dipendenza che un programma lineare per layer non esprima; costo di verifica alto | Riaprire solo se il MoE a batch accumula dipendenze speciali |
| S6 | **Moduli con autorità distinte** (alternativa B) adesso | Non esistono politiche in concorrenza da sostituire; il protocollo fra moduli anticiperebbe decisioni | Tre estrazioni mirate: store di shard, esecutore dei kernel, sessione e KV |
| S7 | **Kernel FP32 a corsie come contratto di default del GEMV** | Al plateau DRAM FP64 e FP32 quasi coincidono (17 contro 21 Gw/s); su un thread il guadagno è 1,6× su una voce che dopo la fusione non è il collo; costa bit-identità e parità FP64 | FP32 solo dove raddoppia davvero i FLOP: GEMM del prefill e GPU, con contratto e oracolo propri |
| S8 | **Riduzione FP64 a corsie lungo la riga** (kernel B) al posto di quello ordinato per riga | +15% rispetto al kernel bit-identico; cambia i digest di tutte le catture (scarti ≈ 10⁻¹¹) | Nulla: il kernel bit-identico dà quasi lo stesso throughput a rischio zero |
| S9 | **GPU per accelerare il decode** | Decode limitato dall'I/O; guadagno misurato 2–12% in V2/M1a | GPU per GEMM a batch (M13) |
| S10 | **Più worker e più slot nella pipeline Python** | Il readahead del kernel sovrappone già; latenza di passaggio fra thread dell'ordine del servizio di una tile; misura: nessun guadagno | Lettore nativo a blocchi grandi (O3) |
| S11 | **Swap del sistema come deposito dei pesi** | Incontrollato; i pesi diventerebbero memoria anonima espansa 4 volte | Nulla |
| S12 | **Ottimizzare prima dell'incremento 10** | Mescolerebbe due variabili (taglia e codice nuovo) nella prova «oltre RAM host» | Differenza zero nel runtime per il 10; ottimizzazioni dopo, validate per digest |
| S13 | **Denso ≥ 70B come obiettivo di usabilità** | Pavimento fisico di decine di secondi per token su un NVMe | Dimostrazione di eseguibilità; usabilità tramite MoE o decode multi-token |
| S14 | **Speculative decoding subito** | Richiede il forward a batch (O2b), un modello di bozza e una regola di accettazione verificata | Ricerca dopo O2b |
| S15 | **Distribuzione multi-PC, RAID0 software, acceleratori alternativi** adesso | Fuori perimetro; nessuna evidenza che risolvano il limite attuale | M15+ |

## 9. Roadmap e proposte GitHub

### 9.1 Stato delle Issue esistenti (lette oggi con `gh`, sola lettura)

| Issue | Titolo (abbreviato) | Stato | Osservazione |
|---|---|---|---|
| #2 | [M9-A] SmolLM2-1.7B sotto cgroup con page cache | aperta | Evidenze presenti (gate B del 9). Chiudibile dal proprietario quando il follow-up del gate sarà revisionato e committato |
| #3 | [M9-B] Primo LLM pubblico oltre `MemTotal` WSL | aperta | Come sopra (gate D–F del 9) |
| #4 | [M10] Modello reale oltre RAM fisica host | aperta | Coperta dalla sezione 6 |
| #5 | [M11] MoE originale: routing esatto e streaming selettivo | aperta | Coperta dalla sezione 7.1–7.2 |
| #6 | [M12] Prestazioni: cache, scheduling, pipeline | aperta | Da dividere (9.3) |
| #7 | [M13] Backend GPU opzionale | aperta | Da restringere al GEMM a batch (7.6) |
| #8 | [M14] API locale e DeepSeek Harness | aperta | Dipende da O2 |
| #9 | [M15+] Grande scala | aperta | Aggiungere FP8 nativo e capacità di storage |
| #10 | [Roadmap] Scopo ultimo e milestone | aperta | Indice; rimanda alla PR #1 (`docs/VISION_AND_ROADMAP.md`) |
| #11 | [Research] Betelgeuse: expert demand paging | aperta | Sezione 7.8 |
| #12 | [Research] Tile streaming vs module streaming | aperta | Sezione 7.9 |

PR #1 (visione e roadmap) è aperta: il suo testo descrive l'incremento 9 come «da dimostrare» ed è quindi da aggiornare prima del merge.

### 9.2 Ordine ottimale degli incrementi

Proposta: **spostare la parte numericamente invariante di M12 prima di M11.** Motivi:

- O1 e O2 si validano per uguaglianza di digest sulle catture già accettate: il costo di validazione è minimo **ora**, con tre o quattro checkpoint reali e le loro catture disponibili.
- L'adapter MoE va scritto una volta sola, già sulla struttura layer-major, invece di nascere token-major ed essere riscritto.
- La campagna di oracolo di M11 su un checkpoint da 61 GB costa meno su un motore circa 3 volte più rapido e con prefill a batch.
- Non viola la gerarchia «correttezza → eseguibilità → prestazioni»: sono cambi che **non alterano un solo bit** degli output.

| Passo | Milestone | Contenuto | Criterio di accettazione sintetico | Dipende da |
|---|---|---|---|---|
| I10 | M10 (#4) | Qwen2.5-14B oltre la RAM fisica, runtime invariato | Gate A–H della sezione 6.11 | follow-up del gate 9 chiuso |
| I11 | M12a | O1: kernel fuso bit-identico, blocchi grandi, coordinatore a grana di operazione | Digest uguali a 8/9/10; decode 7B cold ≥ 2× più rapido | I10 |
| I12 | M12a | O2a: operatori nativi bit-identici, KV append-only, output dell'ultima posizione | Digest uguali; parità con l'oracolo a contesto ≥ 1.024 su un modello piccolo; curva di costo in T | I11 |
| I13 | M12a | O2b: attraversamento layer-major, prefill multi-RHS, separazione sessione/KV dal programma del layer | Digest batch = seriale; prefill di 512 token su 7B < 5 min | I12 |
| I14 | M11 (#5) | Adapter Qwen3-MoE, routing esatto, lettura selettiva, **senza cache**; limiti di catalogo parametrici; traccia di routing | Sezione 7.1; zero byte letti per esperti non selezionati | I13; decisione di storage |
| I15 | M12b (#6) | O3: backend di lettura esplicito; cache bounded (parte condivisa fissata, esperti sostituibili); A/B page cache contro cache esplicita | Byte e digest invariati; guadagno misurato a pari checkpoint; budget rispettato | I14 |
| I16 | M13 (#7) | GEMM a tile su GPU per prefill e verifica | `T*` misurato; fallback CPU; parità sul contratto dichiarato | I13 |
| I17 | M14 (#8) | API locale, coda a sessione singola, chat template, Harness come client | Test reale con Harness; cancellazione; isolamento | I13 (I15 consigliato) |
| — | M15+ (#9) | FP8 E4M3 nativo; integrità lazy; multi-NVMe; decode multi-token per scansione; storage | Per ciascuno: preflight, oracolo, misura | I14–I15; decisioni hardware |

Dipendenze tecniche in una riga: `I10 → I11 → I12 → I13 → {I14 → I15, I16, I17}`; FP8 e storage sono indipendenti dal resto ma bloccano i checkpoint da 0,7–1 T.

### 9.3 Milestone sperimentali

| Esperimento | Domanda | Misura decisiva | Quando |
|---|---|---|---|
| E1 | Il kernel fuso è bit-identico end-to-end e quanto rende? | Digest; decode A/B a 5 coppie | I11 |
| E2 | Come cresce il costo con il contesto dopo O2a? | s/token a T = 64…4.096; memoria KV | I12 |
| E3 | Il prefill a batch è bit-identico al seriale? Qual è `T*` sulla CPU? | Digest; tok/s di prefill per T_c = 16…512 | I13 |
| E4 | Quanta località ha il routing reale? | Curva dei byte di esperti distinti in finestre di N token; margini del top-k | I14 |
| E5 | Page cache o cache esplicita? Buffered o diretto? | A/B a pari budget: hit per byte, s/token, reclaim, PSI, RAM host | I15 |
| E6 | Le letture selettive rendono come le sequenziali? | GB/s e amplificazione su estensioni da 3 MiB a QD 1…32 | I14–I15 |
| E7 | Dove conviene la GPU? | `T*` sulla RTX 3050; wall completo | I16 |
| E8 | Quanti token per scansione con verifica speculativa? | Tasso di accettazione, uscita identica al greedy | dopo I13 |
| E9 | La banda regge per minuti? | GB/s nel tempo, temperatura SSD letta dall'host | prima di qualunque stima oltre 70B |

### 9.4 Issue da modificare (testo proposto, nessuna modifica eseguita)

| Issue | Modifica proposta |
|---|---|
| **#4** (M10) | Aggiungere: candidato `Qwen/Qwen2.5-14B` alla revisione `97e1e76335b7017d8f67c08a19d103c0504298c9`; criterio numerico (≈ 27,98 GB attivi contro 16.900.739.072 B); regola «differenza zero in `inwime_runtime/` e `src/`»; `RLIMIT_AS` 384 MiB con derivazione; contatore `/sys/block/<dev>/stat` e PSI; oracolo con head a chunk e controllo di equivalenza su 7B; registrazione dell'utilizzo della soglia per layer; run ripetuto per la variabilità |
| **#5** (M11) | Aggiungere: candidato `Qwen/Qwen3-30B-A3B` (`ad44e777…`), geometria verificata (18.867 tensori, 6,08 GB attivi per token, esperto da 9,44 MB); protocollo dei margini del top-k; oracolo a grana di esperto; prima versione senza cache; prova «zero byte per esperti non selezionati»; dipendenza da I13 |
| **#6** (M12) | Dividere in **M12a** (motore a numerica invariata: O1, O2a, O2b, prima di M11) e **M12b** (I/O esplicito, cache, prefetch, dopo M11). Aggiungere i numeri misurati: 63% del token in espansione e GEMV; 19 µs per richiesta; pipeline senza guadagno |
| **#7** (M13) | Restringere l'obiettivo: GEMM a batch per prefill e verifica; dichiarare che il decode non è l'obiettivo; accettazione su `T*` misurato; due contratti numerici possibili |
| **#8** (M14) | Aggiungere: coda a sessione singola; semantica SSE con token lenti; dipendenza da O2b |
| **#9** (M15+) | Aggiungere: i MoE pubblici da 0,7–1 T sono FP8 E4M3 nativi (DeepSeek-V3: 680,57 su 684,49 miliardi di parametri; Kimi-K2: 1.023,9 su 1.026,4); capacità richiesta 688 GB e 1,03 TB; 163 shard; integrità lazy |
| **#10** (indice) | Aggiornare l'elenco con M12a/M12b e l'ordine proposto; segnare #2 e #3 come «evidenze presenti, in attesa del follow-up del gate» |
| **#11** (Betelgeuse) | Annotare lo stato di verifica: README e `fluxaeros.md` letti il 16 novembre; codice del pager non esaminato; elenco dei principi trasferibili e non (sezione 7.8); la scelta buffered/diretto come esperimento E5 |
| **#12** (tile vs moduli) | Annotare: adottare l'attraversamento layer-major nel motore a tile; non adottare il backend PyTorch; rinviare il confronto A/B/C a dopo I13 |
| **PR #1** | Aggiornare lo stato dell'incremento 9 prima del merge |

### 9.5 Issue nuove da creare (nessun duplicato rispetto a #2–#12)

| Proposta | Titolo | Contenuto essenziale e accettazione |
|---|---|---|
| N1 | [M12a] Kernel GEMV BF16 fuso, bit-identico al kernel ordinato | Ordinato per riga, vettorizzato fra righe; nessuna arena FP64; digest uguali alle catture 8/9/10; invarianza a tile e thread |
| N2 | [M12a] Operatori nativi bit-identici e KV append-only | Attenzione, RMSNorm, RoPE, SiLU, bias; banco unico con pubblicazione della lunghezza; logits dell'ultima posizione; contesto ≥ 1.024 verificato |
| N3 | [M12a] Attraversamento layer-major e prefill a batch | Multi-RHS bit-identico al seriale; head solo ultima posizione; separazione sessione/KV dal programma del layer |
| N4 | [M12b] Backend di lettura esplicito e store di shard | O_DIRECT, blocchi grandi, multi-intervallo; FD, firme e integrità in un solo modulo; A/B contro buffered |
| N5 | [M11/M15] Scalabilità del catalogo e integrità lazy | Limiti parametrici (tensori, shard, nodi JSON); manifest verificato e digest per chunk; `full` obbligatoria nei gate |
| N6 | [M11] Traccia di routing e curva del working set degli esperti | Strumento di sola osservazione; prerequisito di qualunque politica di cache |
| N7 | [M15] dtype nativo FP8 E4M3 | Preflight della semantica (scale per blocco); espansione esatta; oracolo; nessuna conversione persistente |
| N8 | [Research] Decode multi-token per scansione con verifica greedy-esatta | Per i densi; dipende da N3; misura del tasso di accettazione |
| N9 | [Misure] Banda sostenuta, temperatura e alimentazione | Letture di minuti; temperatura SSD dall'host; stato di alimentazione registrato in ogni evidenza |

### 9.6 Rischi da tenere separati dalla roadmap

| Rischio | Natura | Nota |
|---|---|---|
| Capacità di storage: 144,8 GB liberi, un solo slot NVMe | Hardware, decisione del proprietario | Blocca 72B e ogni checkpoint da 0,7–1 T; non è un compito software |
| Licenze dei pesi | Legale | Qwen2.5-72B e Kimi-K2 riportano licenza «other»; DeepSeek-V3 ha un file di licenza del modello a parte. Verificare prima di ogni download |
| WSL2: crescita della cache del guest e paging dell'host | Ambiente | Host con ≈ 0,5 GB liberi all'audit; effetto non misurato |
| Termica e alimentazione | Misura | Mai registrate; incidono su tutte le stime oltre il minuto |
| Deriva delle versioni di Transformers e torch negli oracoli | Riproducibilità | Fissare le versioni per ogni incremento; il profilo `full64` dipende da funzioni interne di upstream |
| Costo crescente di gate e strumenti per incremento | Processo | `report_increment9.py` ha 788 righe e i suoi test 941; gli strumenti sono copiati a ogni incremento. Parametrizzare (modello e revisione come argomenti) senza costruire un framework |
| Percorsi personali già nella storia Git ed evidenze 4–8/V2 non sanificate | Privacy, decisione in sospeso | Già dichiarato nel README; indipendente dall'architettura |
| Indipendenza parziale del profilo `full64` | Verifica | Stesso autore del runtime; l'indipendenza piena viene da `stock_fp32` |
| Scritture di scratch per il KV paginato | Sicurezza operativa | Richiedono tetto e autorizzazione espliciti |
| Usura da download ripetuti | Hardware | Ogni checkpoint scaricato è una scrittura da decine di GB; pianificare quali conservare |

## 10. Le cinque decisioni più importanti per il futuro di InWiMe

### D1 — La scala passa dai MoE, non dai densi: adottare «byte attivi per token» come metrica di progetto

- **Perché.** Il pavimento del decode è `byte attivi / banda`. Su un NVMe da 3,0–3,6 GB/s un denso da 70B non scende sotto 40–48 s per token e uno da 400B sotto 3,8–4,5 min, con qualunque kernel. Un MoE da 61 GB con 6,08 GB attivi costa già oggi meno della metà del denso da 14 GB.
- **Decisione proposta.** Dopo M10, i densi oltre ~30B diventano prove di eseguibilità. L'obiettivo di usabilità è MoE: M11 subito dopo il motore a numerica invariata. Ogni rapporto riporta totale, attivi per token, byte letti per token e hit.
- **Classificazione.** HIGH CONFIDENCE (aritmetica su grandezze misurate). L'entità del beneficio della cache è REQUIRES EXPERIMENT.

### D2 — Contratto numerico: tenere FP64 ordinato bit-identico come riferimento del motore CPU

- **Perché.** La micro-misura mostra che un kernel fuso può restare **bit-identico** al kernel attuale ed essere 5,5 volte più rapido su un thread; al plateau della DRAM FP64 e FP32 quasi coincidono. Rinunciare a FP64 non compra banda nel decode e costa la validazione per digest, che è l'asset più prezioso del progetto.
- **Decisione proposta.** FP64 ordinato resta il contratto di riferimento e di regressione. Contratti FP32 o BF16 si introducono solo dove raddoppiano i FLOP utili (GEMM del prefill, GPU), ciascuno con nome, oracolo e soglie proprie, mai come sostituzione silenziosa.
- **Classificazione.** PROVEN a micro-scala; HIGH CONFIDENCE end-to-end.

### D3 — Coordinatore unico a grana di operazione, piano dati nativo, attraversamento layer-major

- **Perché.** L'autorità unica ha funzionato per correttezza e guasti. La grana a tile no: 27 mila passaggi Python per token, pipeline inutile, prefill N volte il decode, contesto limitato a decine di token.
- **Decisione proposta.** Mantenere l'alternativa A, spostando il ciclo delle tile, la lettura e i kernel in un esecutore nativo; rendere layer-major la struttura canonica degli adapter; estrarre tre soli moduli (store di shard, esecutore dei kernel, sessione e KV). Farlo **prima** di M11.
- **Classificazione.** HIGH CONFIDENCE.

### D4 — Chi governa la memoria: page cache più cgroup, oppure I/O diretto più cache esplicita

- **Perché.** Oggi il limite di memoria è imposto dall'esterno e la page cache è l'unica cache. Per i MoE questa scelta decide la forma della cache degli esperti, l'interazione con l'host WSL2 e il modo in cui si dimostra un budget.
- **Decisione proposta.** Non decidere a priori: eseguire l'A/B E5 sullo stesso checkpoint MoE a pari byte. Criteri: hit per byte, s/token, reclaim, PSI, RAM libera dell'host, complessità. In ogni caso il runtime deve poter funzionare auto-limitato senza cgroup.
- **Classificazione.** REQUIRES EXPERIMENT.

### D5 — Per 400B–1T servono tre cose che non sono ottimizzazioni: FP8 nativo, storage, integrità lazy

- **Perché.** I MoE pubblici da 0,7–1 T verificati oggi sono FP8 E4M3 nativi (non BF16), occupano 688 GB e 1,03 TB, in 61–163 shard. C: ha 144,8 GB liberi e la macchina ha un solo slot NVMe. La SHA integrale a ogni avvio costerebbe 9–27 minuti.
- **Decisione proposta.** (a) Decidere se e quando supportare FP8 E4M3 come dtype nativo esatto, con preflight della sua semantica. (b) Decidere l'investimento di storage **dopo** M11 e la traccia di routing, quando si conoscerà la latenza reale di un MoE sul percorso attuale. (c) Introdurre l'integrità lazy come politica dichiarata. Senza queste tre scelte la classe 400B+ resta fuori portata indipendentemente dal software.
- **Classificazione.** PROVEN i fatti (dtype, dimensioni, shard, spazio); HYPOTHESIS le latenze (sezione 4.6).

## Appendice A — Micro-misure di questa revisione

**Natura.** Prove sintetiche, in memoria, di pochi secondi, eseguite fuori dal repository. Nessun checkpoint letto, nessun file del repository scritto. Host non isolato, nessun controllo termico o di alimentazione. Servono a stabilire ordini di grandezza e una proprietà esatta (la bit-identità), **non** a caratterizzare il runtime.

**Materiali conservati** in `/home/francesco/inwime-reports/ARCHITECTURE_REVIEW_2026-11-16-appendix/`:

| File | Contenuto |
|---|---|
| `micro_python.py` | Costo per elemento dei cicli Python di copia KV e di attenzione, geometria Qwen2.5-7B |
| `kbench.c` | Kernel A (semantica attuale), B (FP64 a corsie), C (FP32 a corsie), D (GEMM FP32 ingenuo) |
| `kbench2.c` | Kernel E: ordinato per riga, vettorizzato fra 8 o 16 righe; confronto `memcmp` con il riferimento scalare |
| `e8lib.c`, `compare_with_repo_kernel.py` | Confronto bit per bit del kernel E con `build/libinwime_cpu.so` del repository |
| `outputs.txt` | Uscite grezze di tutte le prove |

**Ambiente.** i7-12700H sotto WSL2 (kernel 6.18.58.1), 20 CPU logiche, AVX2/FMA/F16C, niente AVX-512; gcc 15.2.0 con `-O3 -march=native -fopenmp -ffp-contract=off`; Python 3.14.4 (intervallo di switch del GIL 5 ms).

**Risultati principali.**

| Prova | Risultato |
|---|---|
| Copia KV in Python | 64,9 ns per elemento (458.752 elementi in 29,8 ms) |
| Attenzione in Python, un layer | 0,59–0,60 ms per posizione di contesto a T = 8–16 |
| A — espansione + GEMV ordinato, 1 thread | 1,59 e 2,08 Gw/s; combinati 0,90 Gw/s |
| E — ordinato per riga su 8 righe | 4,92 Gw/s (1 thread); 12,2 (4); 17,1–17,3 (12–14) |
| B — FP64 a corsie | 5,66 Gw/s (1); 15,1 (4); 20,4 (12–14); scarto max 3,64·10⁻¹¹ |
| C — FP32 a corsie | 7,91 Gw/s (1); 20,0 (4); 21,2–21,6 (12–14); scarto max 1,54·10⁻³ |
| D — GEMM FP32 ingenuo | 113–128 GFLOP/s (T = 16); 197–211 (T = 64); 157–165 (T = 256) |
| E contro `build/libinwime_cpu.so` | **identico bit per bit** su 2.048 × 3.584, 1.027 × 5.120, 515 × 18.944, 64 × 13.827 |

Il cuore del kernel E, per riferimento: ogni corsia è una riga, le colonne avanzano in ordine, prodotto e somma restano separati.

```c
/* 8 righe: sa accumula le righe 0–3, sb le righe 4–7; c0..c3 e d0..d3 sono le colonne j..j+3
   trasposte fra righe; x0..x3 = x[j..j+3] replicati. Nessuna FMA: stessa sequenza di
   arrotondamenti di  sum += w[j]*x[j]  compilato con -ffp-contract=off. */
sa = _mm256_add_pd(sa, _mm256_mul_pd(c0, x0));  sb = _mm256_add_pd(sb, _mm256_mul_pd(d0, x0));
sa = _mm256_add_pd(sa, _mm256_mul_pd(c1, x1));  sb = _mm256_add_pd(sb, _mm256_mul_pd(d1, x1));
sa = _mm256_add_pd(sa, _mm256_mul_pd(c2, x2));  sb = _mm256_add_pd(sb, _mm256_mul_pd(d2, x2));
sa = _mm256_add_pd(sa, _mm256_mul_pd(c3, x3));  sb = _mm256_add_pd(sb, _mm256_mul_pd(d3, x3));
```

**Limiti.** Dati sintetici; nessuna lettura da disco nel ciclo; con molti thread e matrici piccole i dati restano in cache (per questo le cifre del regime DRAM vengono solo dalla matrice da 135,8 MB); il prototipo non gestisce i non-finiti né il riporto di righe parziali fra tile (il confronto con la libreria del repository usa righe intere dal lato del prototipo e tile che tagliano le righe dal lato del repository).

## Appendice B — Fatti pubblici verificati il 16 novembre 2026

Fonte: API e file `config.json` di Hugging Face; pagine pubbliche GitHub. Nessun peso scaricato. Le pagine sono state lette con uno strumento che ne restituisce un estratto: i valori vanno ricontrollati sulla fonte dal preflight prima di entrare in un gate (per Qwen2.5-14B e Qwen3-30B-A3B il conteggio dei parametri è stato comunque ricostruito in modo indipendente dalla geometria e coincide, Appendice C).

| Oggetto | Revisione | Fatti |
|---|---|---|
| `Qwen/Qwen2.5-14B` | `97e1e76335b7017d8f67c08a19d103c0504298c9` | Non gated; Apache-2.0; 14.770.033.664 parametri, tutti BF16; 8 shard più index; `Qwen2ForCausalLM`; 48 layer; hidden 5120; FFN 13.824; 40 teste; 8 teste KV; vocabolario 152.064; `rms_norm_eps` 1e-5; `rope_theta` 1e6; sliding window disattivata; head non condivisa |
| `Qwen/Qwen2.5-32B` | `1818d35814b8319459f4bd55ed1ac8709630f003` | Non gated; Apache-2.0; 32.763.876.352 parametri BF16; 17 shard; 64 layer; hidden 5120; FFN 27.648; 40/8 teste; stesso vocabolario |
| `Qwen/Qwen2.5-72B` | `efba10c8e54e91e0d9570ab5f7b51a958474d4cb` | Non gated; licenza «other»; 72.706.203.648 parametri BF16; 37 shard |
| `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | Non gated; Apache-2.0; 30.532.122.624 parametri BF16; 16 shard; `Qwen3MoeForCausalLM`; 48 layer; hidden 2048; 32 teste; 4 teste KV; `head_dim` 128; 128 esperti; 8 per token; `moe_intermediate_size` 768; `norm_topk_prob` true; `attention_bias` false; vocabolario 151.936; head non condivisa |
| `deepseek-ai/DeepSeek-V3` | `e815299b0bcbac849fa540c768ef21845365c9eb` | 684.489.845.504 parametri: `F8_E4M3` 680.571.043.840, BF16 3.918.786.560, F32 15.104; 163 shard |
| `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | 1.026.408.235.864 parametri: `F8_E4M3` 1.023.893.241.856, BF16 2.514.970.968, F32 23.040; 61 shard; licenza «other» |
| `Betelgeuse-AI/Betelgeuse` | branch `main` alla data | Fork C++/CUDA di llama.cpp; README e `docs/fluxaeros.md` letti; contenuti riportati nella sezione 7.8; codice non esaminato |

Non verificati qui e quindi usati solo come ordini di grandezza: parametri attivi per token di DeepSeek-V3 e Kimi-K2 e delle classi 100B/12B e 235B/22B; semantica delle scale del formato FP8; sorgente Transformers di Qwen3-MoE; modelli citati nel `PROJECT_BRIEF.md` (Qwen3.6-35B-A3B, GLM-5.3).

## Appendice C — Derivazioni

**Qwen2.5-14B.** Per layer: Q 5120² + 5120; K 1024 × 5120 + 1024; V idem; O 5120²; MLP 3 × 5120 × 13.824; due norme da 5120. Totale per layer 275.268.608. Modello: 48 × 275.268.608 + embedding 778.567.680 + head 778.567.680 + norma finale 5.120 = **14.770.033.664** (uguale al valore pubblicato). Tensori: 48 × 12 + 3 = 579. Coefficienti GEMV per token: 48 × 275.251.200 + 778.567.680 = 13.990.625.280. Byte attivi: (14.770.033.664 − 778.567.680) × 2 = 27.982.931.968.

**Qwen3-30B-A3B.** Per layer: Q 4096 × 2048; K 512 × 2048; V 512 × 2048; O 2048 × 4096; norme su Q e K 128 + 128; due norme da 2048; router 128 × 2048; esperti 128 × 3 × 768 × 2048. Totale per layer 623.120.640. Modello: 48 × 623.120.640 + 2 × 311.164.928 + 2.048 = **30.532.122.624** (uguale al valore pubblicato; la coincidenza conferma la presenza delle due norme da 128 per layer). Tensori: 48 × 393 + 3 = 18.867. Parte condivisa per token: 48 × 19.140.864 + 311.164.928 + 2.048 = 1.229.928.448. Esperti per token: 48 × 8 × 4.718.592 = 1.811.939.328.

**Costi unitari del 9.** 13,27 s ÷ 14,141 GB = 0,94 s/GB. Letture 3,76 s → 0,266 s/GB; espansione 4,75 s → 0,336 s/GB; GEMV 3,65 s → 0,258 s/GB; residuo 0,49 s ÷ 27.113 richieste → ≈ 19 µs.

**Stime dell'incremento 10.** Decode: 27,98 GB × 0,94 s/GB ≈ 26 s. SHA di avvio: 29,54 GB ÷ 1,23 GB/s ≈ 24 s. Oracoli: tempi del 9 (251 s, 439 s, 155 s) × 29,54 ÷ 15,23.
