# InWiMe — Catalogo delle tecniche e dei sistemi da studiare

**Stato:** inventario in evoluzione; copertura non esaustiva e provenienza da completare.

**Revisione editoriale:** 1.4 — 12 novembre 2026. Base importata: versione 1.3, dichiarata del 7 novembre 2026.

**Scopo:** conservare tecniche, progetti, articoli e ipotesi senza trasformarli in una roadmap o in decisioni architetturali. Le priorità del corpo storico sono suggerimenti da riesaminare, non autorizzazioni per l'incremento 3.

**Provenienza del documento:** `INWIME_CATALOGO_TECNICHE_E_RIFERIMENTI_2026-11-07_v1.3.md`, fornito dal proprietario e trasferito nel percorso canonico indicato dalla sua intestazione. SHA256 dell'importazione originale: `61f80e57db27d770059a55b6276625a765e21e1f6fadd573afbe64d873f12541`.

Il consolidamento è documentale: non comprende nuove ricerche web, verifica delle release upstream o nuovi benchmark. Le 60 schede F01–F60, i principi T01–T39 e le ipotesi H01–H37 sono conservati. Il corpo delle sezioni 0–12 resta quello della versione importata, salvo la normalizzazione degli spazi finali Markdown. Le descrizioni tecniche, le misure di terzi e le espressioni «verificato» o «già discusso» riportano quanto dichiarava quel materiale; non rappresentano una nuova attestazione dello stato corrente delle fonti.

## Registri di provenienza

Le etichette originali hanno questi significati operativi:

- **[D] — Discussione storica attribuita:** il catalogo importato attribuisce l'argomento alle discussioni precedenti. Il registro A distingue le attestazioni nei documenti disponibili dalle attribuzioni ancora prive di thread/data o fonte primaria interna.
- **[R] — Ricerca o approfondimento successivo:** riferimento o dettaglio tecnico riportato dalla ricognizione del 7 novembre 2026. La data è quella dichiarata dal documento importato, non una nuova verifica editoriale.
- **[P] — Ipotesi per InWiMe:** applicazione possibile da dimostrare, senza implicare una funzione già implementata.

Provenienza, verifica tecnica e adozione sono assi diversi. `[D][R]` non significa che ogni dettaglio tecnico fosse già noto nella discussione originale. Una nuova scheda R può approfondire un progetto già storico: per esempio Colibri è attestato nel brief (F43), mentre F54 è registrato come approfondimento successivo su Vulkan e staging.

### Registro A — Argomenti attribuiti alle discussioni precedenti

Un riscontro nel brief prova che quel nome o principio era già documentato nel progetto; non certifica il supporto attuale, le prestazioni o tutte le affermazioni della scheda. Dove manca tale riscontro, conservare l'attribuzione senza inventare un thread o una data.

Fonti interne confrontate nel consolidamento: [PROJECT_BRIEF.md](../PROJECT_BRIEF.md) e [proposta architetturale](BRIAREUS_ARCHITECTURE_PROPOSAL.md). Il loro contenuto antecede questo consolidamento, ma non consente di ricostruire ogni conversazione originale.

| Scheda | Argomento | Attestazione disponibile / limite |
|---|---|---|
| F01 | llama.cpp / ggml | PROJECT_BRIEF.md: llama.cpp/ggml; dettagli di loader da verificare nelle fonti R. |
| F02 | AirLLM | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F03 | DeepSpeed ZeRO-Inference / DeepNVMe | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F04 | FlexGen | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F06 | Linux `io_uring` / liburing | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F07 | NVIDIA GPUDirect Storage, pinned host memory e CUDA streams | PROJECT_BRIEF.md: pinned memory e stream CUDA; GDS non attestato qui come discussione precedente. |
| F08 | Multi-NVMe / collocazione fisica degli shard | BRIAREUS_ARCHITECTURE_PROPOSAL.md, §9: collocazione multi-NVMe. |
| F09 | safetensors e GGUF | PROJECT_BRIEF.md: safetensors e shard; la provenienza storica di GGUF resta da precisare. |
| F11 | PowerInfer | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F13 | DwarfStar 4 (DS4) | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F16 | SpikingBrain 2.0, Gated DeltaNet, SSE, MoBA | PROJECT_BRIEF.md: famiglia SpikingBrain; non certifica tutte le sottotecniche della scheda. |
| F17 | Speculative decoding, EAGLE / Medusa / MTP / n-gram | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F18 | Memoria degli esperti | BRIAREUS_ARCHITECTURE_PROPOSAL.md, §8: cache e prefetch; fonti terze da distinguere. |
| F19 | vLLM | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F24 | DistServe e Splitwise | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F25 | Continuous batching, fairness, prefix-aware request routing | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F26 | Petals | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F27 | llama.cpp RPC | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F28 | Apple MLX distribuita | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F29 | vLLM, Ray Serve e llm-d | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F30 | NIXL, Mooncake Transfer Engine, RDMA, zero-copy | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F31 | Expert parallelism e distribuzione esperti MoE | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F32 | TinyGiant | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F33 | Expert-contiguous layout | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F37 | Qwen3.8-Flash-Next | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F38 | Text-calibrated pinning degli esperti vs prefetch predittivo | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F40 | SpikingBrain 1/2 | PROJECT_BRIEF.md: SpikingBrain 1.0/2.0 e operatori specifici. |
| F41 | DeepSeek Harness e API compatibile | PROJECT_BRIEF.md: DeepSeek Harness come client opzionale. |
| F43 | Colibri | PROJECT_BRIEF.md: Colibri fra i progetti da esaminare. |
| F44 | WARP (ex WASTE) | PROJECT_BRIEF.md: WARP fra i progetti da esaminare. |
| F50 | Porting indipendente SpikingBrain 2.0 CPU e regressione FLACache | PROJECT_BRIEF.md: porting spikingbrain-cpu e segnalazione FLACache. |
| F55 | Schede PCIe/M.2 e SBC esplorate nel progetto distribuito del settembre 2026 | PROJECT_BRIEF.md: famiglie Akida/Hailo/Coral/Axelera; altri modelli e SBC restano attribuzioni della v1.3. |
| F56 | Tenstorrent Blackhole p100a | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F57 | Taalas HC1 | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |
| F58 | Prototipo precedente `ooc-x86-poc` | PROJECT_BRIEF.md: descrizione del vecchio ooc-x86-poc, sorgente non recuperato. |
| F59 | SSD usato come «memoria estesa» | PROJECT_BRIEF.md: obiettivo di streaming SSD con memoria limitata. |
| F60 | CPU-only mini-nodi, accensione a richiesta e scaling energetico | Origine storica dichiarata nella v1.3; documento o thread originale da associare. |

### Registro B — Nuove voci o approfondimenti della ricerca successiva

Queste schede sono marcate R senza D nel materiale importato. Non vengono retroattribuite alle vecchie conversazioni. «Nuova» qualifica la voce o l'approfondimento nell'inventario disponibile, non dimostra che il progetto non fosse mai stato nominato altrove. In mancanza di documentazione aggiuntiva la provenienza rimane quella dichiarata dalla v1.3.

| Scheda | Voce o approfondimento | Origine registrata |
|---|---|---|
| F05 | Apple, *LLM in a Flash* | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F10 | MoE-Infinity | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F12 | PowerInfer-2 | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F14 | FlashAttention / FlashAttention-2/3 | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F15 | FlashInfer | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F20 | SGLang | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F21 | LMCache | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F22 | H2O / StreamingLLM | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F23 | Mooncake | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F34 | PoC Metal di expert paging con pool a slot e ID remapping | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F35 | Guanaco | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F36 | llama.cpp | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F39 | Fusion di caricamento, decodifica dtype e calcolo | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F42 | CPU eterogenea e affinità | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F45 | KTransformers / KT-Kernel | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F46 | ServerlessLLM | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F47 | SPDK e NVMe over Fabrics | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F48 | CUDA Unified Memory e mapped host memory | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F49 | Glinthawk | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F51 | oneDNN, ISA dispatch e differenti costi FP32/BF16 | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F52 | Stato del contesto riutilizzabile | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F53 | Storage distribuito vs compute distribuito | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |
| F54 | Vulkan, Resizable BAR, staging e scheduling misto CPU/GPU | Ricognizione del 7 novembre 2026 riportata nella v1.3; verifica upstream non aggiornata qui. |

### Aggiornamenti successivi

Per ogni aggiunta registrare separatamente: origine storica o nuova ricerca; documento/thread e data quando disponibili; URL primario; release o commit esaminato; ultima verifica tecnica; dtype e modello; evidenze locali o di terzi; eventuale decisione di adozione. Se un dato manca, indicarlo come non verificato. Un riferimento scoperto oggi non acquisisce automaticamente l'etichetta D; una vecchia menzione non equivale a una validazione tecnica attuale.

Il consolidamento non modifica il runtime dell'incremento 2 né i suoi 68 test. Le decisioni operative restano in [INWIME_ARCHITECTURE_DECISION.md](INWIME_ARCHITECTURE_DECISION.md) e il risultato verificato in [RUNTIME_CPU_INCREMENT_2.md](RUNTIME_CPU_INCREMENT_2.md). Nessuna voce del catalogo autorizza implementazioni, acquisti, download o benchmark.

---

## 0. Vincoli del progetto che la ricerca NON deve perdere

- Motore **out-of-core** per inferenza LLM con checkpoint in precisione originale (BF16, FP16 e altri dtype nativi esplicitamente supportati); **nessuna quantizzazione obbligatoria**. Gli studi che usano compressione sono validi come fonti di *altre* tecniche, non come autorizzazione a cambiare i pesi del modello.
- Funzionamento **CPU-only x86-64 completo** con 16 GB di RAM come scenario iniziale e 32 GB come possibile configurazione; GPU discreta con VRAM piccola come acceleratore **opzionale**, non come requisito universale.
- NVMe locale come storage primario scalabile; più NVMe e più nodi possibili. Il Sabrent USB temporaneo e il Katana sono **ambienti sperimentali**, non parametri architetturali.
- Distinguere **correttezza**, **memoria effettivamente controllata**, **RSS/page cache**, **byte letti**, **latenza del singolo token**, **token/s per sessione**, **token/s aggregati**, **TTFT** e **costo energetico/TCO**.
- Decisione attuale: partire con runtime locale compatto a coordinatore unico (**alternativa A**, falsificabile); i nomi Briareus, Mnemosyne, Ariadne e Iris indicano responsabilità, non quattro framework obbligatori. Non trasformare la watchlist in una nuova roadmap senza prove.
- Riferimenti interni: `docs/INWIME_ARCHITECTURE_DECISION.md`, `docs/BRIAREUS_ARCHITECTURE_PROPOSAL.md`, `docs/REFERENCE_DECODER_INCREMENT_1.md`, `docs/RUNTIME_CPU_INCREMENT_2.md`, `BENCHMARK_V2.md`, `BENCHMARK_V2_STORAGE.md` e rapporto M1a.

## Audit di completezza della versione 1.1

La versione 1.0 copriva le **famiglie principali**, ma **non tutte le implementazioni concrete già incontrate**. Questa revisione aggiunge esempi altamente pertinenti a SSD-first e MoE; distingue casi dimostrati, proposte, e benchmark di componenti. L'obiettivo non è garantire un inventario esaustivo di ogni conversazione storica. Una verifica futura può ancora trovare altri riferimenti.

**Omissioni recuperate nella versione 1.1:** TinyGiant e la discussione llama.cpp #27149, PoC Metal #23324, Guanaco, RFC/PR di cache e streaming esperti llama.cpp, disposizione *expert-contiguous* con amplificazione dei page fault, calibrazione degli esperti su testo, PLE/lookup lazy di Qwen3.8-Flash-Next, fusione fra decodifica dei pesi e GEMV, dettagli di SpikingBrain1/2 e integrazione con DeepSeek Harness. Le fonti con pesi quantizzati sono registrate per **principi trasferibili**, senza introdurre quantizzazione obbligatoria.

## 1. Indice rapido delle idee principali

| ID | Tecnica / principio | Perché può servire | Quando riesaminarla |
|---|---|---|---|
| T01 | `mmap`, **lazy tensor reads**, O_DIRECT | Scegliere fra page cache e letture controllate; evitare materializzazione completa | Backend file e primo benchmark I/O |
| T02 | I/O asincrono con `io_uring` / `libaio`, double/triple buffering | Nascondere latenza disco dietro lavoro CPU/GPU | Dopo runtime CPU corretto |
| T03 | **Layer-wise / tensor-wise streaming** | Processare modelli oltre RAM/VRAM senza caricarli interamente | Runtime out-of-core, GPU |
| T04 | **Tiling multidimensionale**, inclusa la dimensione di riduzione | Eseguire singoli tensori più grandi di RAM/VRAM | CPU ottimizzata e GPU |
| T05 | Gerarchia NVMe → RAM → pinned RAM → VRAM | Rendere espliciti costi, residenza, ownership e staging | Trasferimenti GPU |
| T06 | **Overlap reale** I/O↔H2D↔kernel | Recuperare tempo senza aumentare la memoria oltre budget | Pipeline GPU con tracing |
| T07 | Multi-NVMe, collocazione di shard, code per dominio fisico | Banda aggregata e maggiore capacità senza vincoli sul singolo disco | Dopo benchmark di un backend |
| T08 | **MoE expert streaming** just-in-time | Leggere solo esperti selezionati dal router originale | MoE end-to-end |
| T09 | Hot-expert / activation-aware cache, prefetch, eventuale replica | Ridurre letture ripetute quando c'è località | Dopo misurazione del routing |
| T10 | **Paged KV**, KV multi-tier, prefix reuse | Controllare memoria di sessioni e prefissi ripetuti | Context scaling / multi-sessione |
| T11 | **Continuous batching** e chunked prefill | Migliorare throughput aggregato e interattività sotto carico | Serving multi-sessione |
| T12 | FlashAttention / IO-aware attention | Ridurre intermedi e traffico di memoria, senza cambiare attenzione esatta | Attention CPU/GPU ottimizzata |
| T13 | Speculative decoding, EAGLE, MTP, n-gram | Ridurre il numero di costosi passaggi del modello principale | Solo con forward end-to-end stabile |
| T14 | Pipeline / tensor / expert parallelism | Distribuire modello e/o lavoro fra GPU/nodi | **Futuro** multi-server |
| T15 | Inference vicino ai pesi: comunicare **attivazioni**, non tutti i pesi | Risparmiare rete nei sistemi SSD-first multi-nodo | **Futuro** multi-server |
| T16 | Repliche indipendenti + request routing / batching | Baseline obbligatoria quando le sessioni sono multiple | **Futuro** multi-server |
| T17 | Disaggregazione prefill/decode + trasferimento KV | Specializzare risorse quando il carico lo giustifica | **Futuro** serving |
| T18 | Cost model, roofline e autotuning **esplicito** | Scegliere tile, pool, caching e collocazioni in base a misure | Dopo baseline riproducibili |
| T19 | Riferimenti numerici indipendenti e convalida di dtype | Evitare speedup che nascondono errori del modello | **Sempre**, prima di ogni ottimizzazione |
| T20 | Hardware eterogeneo e acceleratori specializzati | Studiare alternative future, senza vincolare il runtime | Dopo prova di scalabilità CPU/GPU |

| T21 | Contiguous expert layout e riduzione delle page fault | Rendere efficienti gli accessi selettivi reali e ridurre I/O amplification | Formati e benchmark MoE |
| T22 | Cache degli esperti con slot rimappati e *lease* | Usare GPU piccole e mantenere output corretti quando cambia la residenza | Backend GPU MoE |
| T23 | `madvise` applicato a pagine di esperti | Confrontare uso guidato della cache OS con cache esplicita del motore | Prima di fissare il backend SSD |
| T24 | Profilazione testuale per la cache hot-expert | Calibrare residenza e prefetch rispetto alle attivazioni osservate | Dopo routing MoE reale |
| T25 | Lookup per righe di grandi tabelle PLE / n-gram | Evitare il caricamento di interi tensori consultati selettivamente | Adattatori specifici del modello |
| T26 | Operatori fused e gestione registri senza grandi temporanei | Ridurre traffico RAM e materializzazioni intermedie | Kernel CPU/GPU ottimizzati |
| T27 | Risorsa CPU eterogenea, affinità P/E e topologia | Evitare contention e scheduling inefficiente | Profiling multi-core CPU |
| T28 | Interfaccia OpenAI-compatible per harness/agent | Consentire client esterni senza influenzare il core inferenza | API/serving dopo token reali |
| T29 | Tiering dei pesi densi vs esperti MoE e cache per ruolo | Limitare i byte SSD per token | MoE out-of-core |
| T30 | I/O raggruppato per esperto + CPU/GPU misto | Ridurre attese, mantenere più backend attivi | Dopo backend GPU funzionante |
| T31 | Native-BF16 CPU kernel dispatch AVX2/AVX512/AMX | Evitare conversioni e kernel inefficienti | Ottimizzazione CPU |
| T32 | Multi-tier startup e checkpoint locality-aware | Ridurre cold-start e traffico di caricamento | Multi-model serving |
| T33 | Userspace NVMe con SPDK | Valutare alternativa a `io_uring` solo se motivata | NVMe diretto futuro |
| T34 | Unified Memory + limiti OS/GPU espliciti | Non confondere memoria virtuale e capacità VRAM reale | Scelta backend GPU |
| T35 | Calcolo attention/KV separato dal tier dei pesi | Sfruttare hardware eterogeneo a batch | Cluster multi-sessione |
| T36 | KV riprendibile e stato ricorrente versione-specifico | Evitare re-prefill negli agenti | Session persistence |
| T37 | Storage remoto vs compute-near-weights | Scegliere in base a byte/latency reali | Multi-server |
| T38 | Vulkan/staging/Resizable BAR capability dispatch | Acceleratori opzionali eterogenei | Backend GPU secondario |

---

## 2. Catalogo delle fonti: offloading, SSD e gestione della memoria

### F01 — llama.cpp / ggml: `mmap`, modalità di caricamento e lazy tensor reads [D][R]
- **Sorgenti:** [llama.cpp](https://github.com/ggml-org/llama.cpp), [interfaccia del loader](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.h), [tipi di `load-mode` e `lazy-mode`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h), [documentazione server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- **Tecniche da studiare:** file mappati, `DIRECT_IO`, `TENSOR_READ_LAZY`, lettura di righe particolari, separazione storage vs allocazione backend, distribuzione di tensori CPU/GPU, cache MoE.
- **Per InWiMe:** confronto fra page cache del sistema operativo e scheduling esplicito dei byte; verificare quando la lettura lazy si applica soltanto a *specifici* tensori marcati dall'architettura (non presumere che ogni peso GGUF sia letto on-demand).
- **Attenzione:** llama.cpp non è automaticamente un motore per qualunque checkpoint BF16 nativo, né `mmap` equivale a un budget fisico garantito.

### F02 — AirLLM: streaming per layer su VRAM molto piccola [D][R]
- **Sorgente principale:** [AirLLM](https://github.com/lyogavin/airllm).
- **Tecniche:** caricare ed eseguire layer successivi invece di mantenere l'intero modello in GPU; anticipare il layer seguente; in casi particolari leggere solo esperti selezionati; file mapping di tabelle di lookup molto grandi.
- **Per InWiMe:** reference concreta per esecuzione di un modello più grande della VRAM, distinzione fra per-layer e per-expert, studio delle copie temporanee.
- **Attenzione:** pochissima VRAM non significa bassa latenza; il progetto può prevedere anche compressioni e formati diversi: studiare separatamente il percorso non quantizzato.

### F03 — DeepSpeed ZeRO-Inference / DeepNVMe [D][R]
- **Sorgenti:** [ZeRO-Inference esempi](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference), [DeepNVMe per ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/deepnvme/zero_inference/README.md), [tutorial DeepNVMe](https://www.deepspeed.ai/tutorials/deepnvme/).
- **Tecniche:** pesi su CPU/NVMe; ingresso per layer in GPU; `libaio`; buffer host intermedi oppure GPUDirect Storage (GDS); impiego di più NVMe in RAID0; batch grandi per ammortizzare la scansione dei pesi; possibile offload del KV.
- **Per InWiMe:** progetto da studiare direttamente per le code I/O, il ciclo dei buffer e il rapporto tra throughput e dimensione dei batch.
- **Attenzione:** i risultati pubblicati enfatizzano **throughput offline/batch**; non sono misure equivalenti ai token/s di una singola sessione.

### F04 — FlexGen: ottimizzazione con modello di costo [D][R]
- **Sorgenti:** [repository degli autori](https://github.com/FMInference/FlexGen), [articolo ICML 2023](https://proceedings.mlr.press/v202/sheng23a.html).
- **Tecniche:** gerarchia GPU/CPU/disk; scheduling a blocchi; riuso dei pesi fra più richieste; sovrapposizione calcolo/I/O; ricerca delle politiche di collocazione mediante programmazione lineare; trade-off latenza/throughput.
- **Per InWiMe:** ispirazione per cost model e policy di ammissione/tiling adattivi, dopo aver acquisito misure credibili.
- **Attenzione:** il progetto propone anche compressione a 4 bit; **non adottarla implicitamente**. I vantaggi dichiarati sono specifici di workload batch.

### F05 — Apple, *LLM in a Flash*: windowing e row-column bundling [R]
- **Sorgente:** [ricerca Apple (ACL 2024)](https://machinelearning.apple.com/research/efficient-large-language).
- **Tecniche:** ridurre la quantità di byte letti sfruttando località/attivazioni, combinare righe e colonne per letture flash più contigue, modello di costo basato sul dispositivo.
- **Per InWiMe:** esaminare **amplificazione I/O** e pattern selettivi; le ottimizzazioni dipendenti dalla sparsità vanno abilitate solo quando la semantica del modello le consente.
- **Attenzione:** può richiedere rappresentazione o organizzazione dei dati specifica. Non significa che la sparsità sia applicabile a qualsiasi Transformer denso.

### F06 — Linux `io_uring` / liburing [D][R]
- **Sorgente:** [liburing e documentazione](https://github.com/axboe/liburing), [manuale io_uring](https://github.com/axboe/liburing/blob/master/man/io_uring.7).
- **Tecniche:** submission/completion rings, più letture in volo, completamenti fuori ordine, cancellazioni logiche, code con limite.
- **Per InWiMe:** candidato per backend Briareus dopo correttezza del percorso sincrono; confrontarlo con `pread` su thread pool e con `libaio`.
- **Attenzione:** API di Linux; usare un contratto backend astratto, non costringere tutta InWiMe a Linux/io_uring.

### F07 — NVIDIA GPUDirect Storage, pinned host memory e CUDA streams [D][R]
- **Sorgenti:** [documentazione GDS/cuFile](https://docs.nvidia.com/gpudirect-storage/index.html), [driver nvidia-fs](https://github.com/NVIDIA/gds-nvidia-fs), [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
- **Tecniche:** I/O NVMe→GPU senza bounce buffer CPU **quando supportato**, doppio buffering, `cudaMemcpyAsync`, pinned host memory, streams, eventi, misurazione effettiva dell'overlap.
- **Per InWiMe:** percorso accelerato opzionale e profilo di capacità, non requisito per il funzionamento.
- **Attenzione:** GDS richiede compatibilità di GPU, driver, filesystem e piattaforma; non presumere che funzioni in WSL2 o sulla RTX 3050 Laptop senza verifica specifica. Memoria pinned sottratta al budget RAM host.

### F08 — Multi-NVMe / collocazione fisica degli shard [D][P]
- **Riferimenti:** [DeepNVMe / RAID e GDS](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/deepnvme/zero_inference/README.md), [Linux liburing](https://github.com/axboe/liburing).
- **Tecniche:** shard su dispositivi distinti, code per unità fisica, striping opzionale, mappa `tensor → shard → sorgente → dominio di banda`, limiti per controller/bus condivisi.
- **Per InWiMe:** consentire espansione NVMe interna senza riscrivere il modello. Misurare **banda aggregata reale** e contesa, non dedurle dal numero di filesystem.
- **Attenzione:** un USB SSD temporaneo e due mount dello stesso NVMe non sono due domini di banda indipendenti; RAID0 e striping sono ipotesi successive, non prerequisiti.

### F09 — safetensors e GGUF: cataloghi, shard e metadata [D][R]
- **Sorgenti:** [safetensors](https://github.com/huggingface/safetensors), [format GGUF / ggml](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [indicizzazione degli shard HF](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_safetensors.py).
- **Tecniche:** descrivere dtype, shape, offsets, lunghezze, ordine dei dati, checksum/revisione; leggere slice senza deserializzare tutti i pesi; adattare formati diversi al medesimo accesso logico.
- **Per InWiMe:** futuri adapter di checkpoint, preservando dtype e semantica *originali*.
- **Attenzione:** supportare GGUF non significa imporre quantizzazione né considerare equivalenti checkpoint GGUF e safetensors; i layout possono essere differenti.

---

## 3. Catalogo delle fonti: MoE, sparsità e calcolo

### F10 — MoE-Infinity: expert offload e activation-aware caching [R]
- **Sorgente:** [EfficientMoE/MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity), [documento d'architettura](https://github.com/EfficientMoE/MoE-Infinity/blob/main/ARCHITECTURE.md).
- **Tecniche:** RAM/SSD come archivio di esperti; leggere quelli selezionati just-in-time; hot-expert cache, tracing/prefetch, code I/O separate, distribuzione esperti su più GPU nello **stesso server**.
- **Per InWiMe:** riferimento prioritario per Ariadne/Mnemosyne quando attiveremo carichi MoE reali; misurare cache hit e read amplification per token.
- **Attenzione:** il README dichiara che il multi-node è ancora distinto/non supportato nell'implementazione descritta; i percorsi FP4 quantizzati non sono il baseline originale BF16.

### F11 — PowerInfer: neuroni hot/cold e inferenza ibrida CPU/GPU [D][R]
- **Sorgenti:** [PowerInfer, repository attuale](https://github.com/Tiiny-AI/PowerInfer), [documentazione di implementazione](https://github.com/Tiiny-AI/PowerInfer), [sito del gruppo](https://powerinfer.ai/).
- **Tecniche:** prevedere neuroni frequentemente attivi e mantenerli in GPU; usare CPU per neuroni freddi; kernel sparsi e località dell'attivazione.
- **Per InWiMe:** spunti per scheduling su CPU/GPU e cache condizionata ai pattern reali.
- **Attenzione:** guadagni riferiti a modelli con **sparsità prevedibile**; non assumere che ogni modello SwiGLU o MoE permetta di saltare pesi/neuroni senza cambiare risultato.

### F12 — PowerInfer-2: pipelining a piccoli cluster di neuroni [R]
- **Sorgente:** [PowerInfer-2, articolo tecnico](https://powerinfer.ai/v2/).
- **Tecniche:** suddivisione fine in cluster, pipeline di caricamento e calcolo, uso eterogeneo di risorse di smartphone, caching mirato.
- **Per InWiMe:** ispirazione per l'ottimizzazione di unità di lavoro **più piccole di un layer**.
- **Attenzione:** il progetto riconosce che sono necessari modelli resi prevedibilmente sparsi; non portare questo presupposto nel kernel generico.

### F13 — DwarfStar 4 (DS4): engine verticale per un modello MoE [D][R]
- **Sorgenti:** [sito DS4](https://dwarfstar.sh/), [repository](https://github.com/antirez/ds4), [relazione tecnica](https://pradeep-stellar.github.io/ds4).
- **Tecniche da studiare:** specializzazione rigorosa di operatori/layout, validazione dei logits di un modello specifico, gestione di esperti e KV persistente.
- **Per InWiMe:** esempio di quanto può guadagnare un runtime **specializzato**, da confrontare con il nostro obiettivo generalista.
- **Attenzione:** l'implementazione esposta pubblicamente include pesi esperti in bassa precisione (es. Q2); non trasferire le sue assunzioni a checkpoint BF16 senza esame.

### F14 — FlashAttention / FlashAttention-2/3: attenzione IO-aware esatta [R]
- **Sorgente:** [repository originale](https://github.com/Dao-AILab/flash-attention), [paper](https://arxiv.org/abs/2205.14135).
- **Tecniche:** tiling di Q/K/V, softmax online, evitare intermedi quadrati, minimizzare traffico SRAM/HBM/VRAM; soluzione di attenzione matematica esatta con differenze fisiologiche di arrotondamento.
- **Per InWiMe:** modello per attenzione senza materializzare l'intera matrice; utilissimo con contesti lunghi.
- **Attenzione:** gli specifici kernel e i vantaggi GPU dipendono dall'architettura; principi algoritmici applicabili anche altrove.

### F15 — FlashInfer: kernel BF16/FP16, paged attention, MoE e fused ops [R]
- **Sorgente:** [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [sito e blog](https://flashinfer.ai/).
- **Tecniche:** kernel per prefill/decode, paged/ragged KV, raggruppamento MoE, fused normalization/activation, selezione backend.
- **Per InWiMe:** repertorio di soluzioni e benchmark per futuro backend GPU; scegliere feature supportate dalla GPU effettiva.
- **Attenzione:** non tutti i kernel e dtype sono disponibili su tutte le GPU; la RTX 3050 non diventa requisito del progetto.

### F16 — SpikingBrain 2.0, Gated DeltaNet, SSE, MoBA [D][R]
- **Sorgenti:** [SpikingBrain2.0](https://github.com/BICLab/SpikingBrain2.0), [FlashMoBA](https://github.com/mit-han-lab/flash-moba).
- **Tecniche:** alternative alla full attention (Gated DeltaNet, block-sparse/MoBA), espansione sparsa dello stato; evoluzione della struttura di un modello, con semantica dedicata.
- **Per InWiMe:** catalogo di **famiglie architetturali** per futuri adapter; possibile lavoro su contesti lunghi, non una tecnica da applicare ai checkpoint generici arbitrariamente.
- **Attenzione:** 'spiking' e 'sparse' nel nome non implicano che l'attuale codice esegua un runtime Transformer BF16 con eventi neurali hardware in senso stretto.

### F17 — Speculative decoding, EAGLE / Medusa / MTP / n-gram [D][R]
- **Sorgenti:** [vLLM (feature disponibili)](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [panoramica EAGLE e algoritmi](https://github.com/sgl-project/SpecForge).
- **Tecniche:** un draft economico propone più token; il modello target **verifica**; il numero di passaggi del modello grande può ridursi senza cambiare la distribuzione quando il meccanismo di verifica è esatto. Varianti: n-gram reuse, alberi, predizione multi-token.
- **Per InWiMe:** particolarmente interessante se ogni passaggio del modello target comporta molti byte NVMe. Valutare costo del draft, capacità di batching del verifier, accettazione e memoria KV aggiuntiva.
- **Attenzione:** non è automaticamente più rapido; l'algoritmo deve conservare la semantica di campionamento scelta; servono pesi o teste draft compatibili nei metodi che li richiedono.

### F18 — Memoria degli esperti: LRU/LFU, hot-expert replication, prefetch [D][P]
- **Sorgenti:** [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity), [AirLLM](https://github.com/lyogavin/airllm), [PowerInfer](https://powerinfer.ai/).
- **Tecniche:** distinguere previsione speculativa da routing effettivo, cache con priorità del carico obbligatorio, repliche degli esperti caldi fra dispositivi/nodi, raggruppamento token per esperto e rilascio su fine utilizzo.
- **Per InWiMe:** misurare distribuzione degli esperti per layer/token, working set e predicibilità **prima** di introdurre un modello di previsione.
- **Attenzione:** cache/prefetch non devono mai sostituire o omettere un esperto matematicamente richiesto.

---

## 4. Catalogo delle fonti: KV, serving e parallelismo multi-sessione

### F19 — vLLM: PagedAttention, continuous batching, chunked prefill [D][R]
- **Sorgenti:** [vLLM](https://github.com/vllm-project/vllm), [documentazione di scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/).
- **Tecniche:** KV allocata a blocchi, batching continuo di richieste eterogenee, riutilizzo di prefissi, separazione di prefill e decode quando utile, TP/PP/EP.
- **Per InWiMe:** benchmark e soluzioni per *più sessioni*, una volta solido il single-session.
- **Attenzione:** il throughput aggregato non equivale a velocizzare un singolo utente; molti vantaggi dipendono da GPU di classe server e da memoria residente.

### F20 — SGLang: RadixAttention e prefix caching [R]
- **Sorgenti:** [SGLang](https://github.com/sgl-project/sglang), [documentazione](https://docs.sglang.io/).
- **Tecniche:** trie/radix del prefisso condiviso, KV reuse, gestione di richieste dinamiche, ottimizzazioni per serving MoE, prefill/decode e speculative decoding.
- **Per InWiMe:** ridurre ricalcolo di prompt ripetuti fra sessioni e agenti.
- **Attenzione:** lifecycle e versioning del KV devono essere compatibili con checkpoint, posizioni, modello e generazione della sessione.

### F21 — LMCache: KV-cache a più livelli e riuso fra istanze [R]
- **Sorgente:** [LMCache](https://github.com/LMCache/LMCache).
- **Tecniche:** KV cache fuori VRAM, tiering CPU/SSD/remoto, prefix caching, caching fra motori/repliche e trasferimento KV prefill→decode.
- **Per InWiMe:** fonte per distinguere la cache immutabile dei pesi dallo **stato KV mutabile**. Cache sharing utile solo dopo la definizione della consistenza.
- **Attenzione:** una cache KV salvata non è un semplice checkpoint immutabile di pesi.

### F22 — H2O / StreamingLLM: trattenere token importanti e attenzione a finestre [R]
- **Sorgenti:** [H2O: Heavy-Hitter Oracle](https://github.com/FMInference/H2O), [StreamingLLM](https://github.com/mit-han-lab/streaming-llm).
- **Tecniche:** cache KV selettiva o finestra con attention sinks per ridurre la memoria di sequenze molto lunghe.
- **Per InWiMe:** *eventuale* modalità distinta dedicata a context scaling.
- **Attenzione:** eliminare elementi KV può cambiare i risultati rispetto all'attenzione completa. **Non** farlo nel percorso baseline originale senza una scelta di semantica esplicita.

### F23 — Mooncake: KV disaggregata e Transfer Engine [R]
- **Sorgenti:** [Mooncake](https://github.com/kvcache-ai/Mooncake), [documentazione](https://kvcache-ai.github.io/Mooncake/).
- **Tecniche:** trasferimento di KV fra pool di prefill/decode, storage cache disaggregato, uso di DRAM/SSD inutilizzata in altri nodi, scheduling basato su SLO.
- **Per InWiMe:** applicabile quando avremo più nodi e sessioni; la rete è una risorsa da misurare e prenotare.
- **Attenzione:** non migliora automaticamente il decode di una singola sessione; il costo di trasferire KV può superare il beneficio.

### F24 — DistServe e Splitwise: separazione delle fasi [D][R]
- **Sorgenti:** [DistServe (articolo OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), [Splitwise (ricerca Microsoft)](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/), [NVIDIA Dynamo disaggregazione](https://docs.nvidia.com/dynamo/cli/disaggregated-serving/overview).
- **Tecniche:** prefill compute-heavy e decode spesso bandwidth/latency-sensitive su risorse separate, trasferendo KV fra fasi.
- **Per InWiMe:** valutazione solo con benchmark TTFT/TPOT/goodput multi-sessione, non sulla base di una singola metrica aggregata.
- **Attenzione:** richiede hardware/rete e workload adeguati, e spesso repliche dei pesi.

### F25 — Continuous batching, fairness, prefix-aware request routing [D][R]
- **Sorgenti:** [vLLM](https://github.com/vllm-project/vllm), [llm-d](https://github.com/llm-d/llm-d), [Ray Serve LLM](https://github.com/ray-project/ray/blob/master/doc/source/serve/llm/index.md).
- **Tecniche:** batch dinamici, controllo priorità e fairness, schedulazione per sessione, instradamento alla replica che possiede già prefisso/pesi/KV.
- **Per InWiMe:** misurare throughput complessivo con più utenti **contro** N copie indipendenti del runtime.
- **Attenzione:** maggior numero di richieste può aumentare latenza o memoria; non confondere goodput SLO con token/s teorici.

---

## 5. Catalogo delle fonti: distribuzione fra più server

### F26 — Petals: modello distribuito per blocchi su computer diversi [D][R]
- **Sorgenti:** [Petals](https://github.com/bigscience-workshop/petals), [paper su inferenza distribuita Internet](https://arxiv.org/abs/2312.08361).
- **Tecniche:** ogni nodo ospita e calcola un sottoinsieme dei layer; gli intermedi si spostano fra server invece di trasferire continuamente tutti i pesi.
- **Per InWiMe:** caso particolarmente vicino alla futura ipotesi **compute near weights** per modelli più grandi della RAM di ogni server.
- **Attenzione:** una pipeline senza microbatch non rende paralleli layer causalmente sequenziali per un singolo token; vantaggi derivano anche da memoria aggregata e minore offloading.

### F27 — llama.cpp RPC: dispositivi e pesi remoti [D][R]
- **Sorgente:** [RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md).
- **Tecniche:** far trattare risorse remote come backend di elaborazione, distribuzione pesi e KV, caching dei tensori remoti, trasporto TCP o RDMA dove supportato.
- **Per InWiMe:** interfacce per calcolo remoto e differenza fra rete per il **caricamento iniziale** e rete percorsa **a ogni token**.
- **Attenzione:** il costo e la granularità RPC vanno misurati; attenzione ai vincoli di sicurezza dell'esposizione di server RPC.

### F28 — Apple MLX distribuita: confronto pipeline vs tensor parallel [D][R]
- **Sorgente:** [Apple Research, 15 ottobre 2026](https://machinelearning.apple.com/research/mlx-distributed-inference).
- **Misure pubblicate (Qwen 3.8 27B BF16, batch 1, contesto 8.192):** 1 Mac **19,69 token/s**; 2 Mac pipeline **19,44**; 2 Mac tensor **33,27**; 4 Mac tensor **50,89**.
- **Lezione:** 4 repliche indipendenti alla velocità *per-macchina* del test implicherebbero **78,76 token/s aggregati** (4 × 19,69): è un **calcolo ipotetico**, non un benchmark Apple su quattro sessioni. Distribuire 4 server a un solo utente produce 50,89 token/s per quell'utente, con minor throughput aggregato rispetto all'ipotesi di 4 repliche.
- **Per InWiMe:** misurare **latency-per-session, throughput aggregato, risorse totali e memoria necessaria** nello stesso confronto; scegliere modalità in base al carico.
- **Attenzione:** Apple usa RDMA a latenza molto bassa su Mac Studio con memoria unificata; non trasferire automaticamente questi speedup a Ethernet ordinaria, PC commodity e SSD-first.

### F29 — vLLM, Ray Serve e llm-d: serving distribuito/repliche [D][R]
- **Sorgenti:** [vLLM](https://github.com/vllm-project/vllm), [Ray Serve](https://github.com/ray-project/ray), [llm-d](https://github.com/llm-d/llm-d).
- **Tecniche:** TP/PP/EP, replica indipendente, routing, bilanciamento per cache hit, partizionamento prefill/decode, autoscaling e priorità.
- **Per InWiMe:** architettura futura a due livelli: runtime locale con controllo risorse e pianificazione distribuita sovraordinata; *ipotesi*, non vincolo da implementare oggi.
- **Attenzione:** Kubernetes/orchestrazione non accelerano da soli una matrice né una singola generazione.

### F30 — NIXL, Mooncake Transfer Engine, RDMA, zero-copy [D][R]
- **Sorgenti:** [NIXL](https://github.com/ai-dynamo/nixl), [Mooncake](https://github.com/kvcache-ai/Mooncake), [NVIDIA Dynamo](https://developer.nvidia.com/dynamo).
- **Tecniche:** trasferire tensori/KV/attivazioni fra host, GPU e dispositivi tramite backend di trasporto intercambiabili, con domini di memoria e sincronizzazione espliciti.
- **Per InWiMe:** estensione futura di backend locale a remoto, distinguendo trasporto, checkpoint consistency, località del calcolo e fault handling.
- **Attenzione:** RDMA non esiste su ogni rete; QUIC e TCP sono possibili alternative con trade-off diversi. Non progettare un cluster solo intorno al percorso più veloce disponibile in un laboratorio.

### F31 — Expert parallelism e distribuzione esperti MoE [D][R]
- **Sorgenti:** [vLLM](https://github.com/vllm-project/vllm), [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity), [llm-d](https://github.com/llm-d/llm-d).
- **Tecniche:** eseguire esperti su dispositivi che ne conservano i pesi, combinare output secondo i coefficienti del router, replicare esperti hot quando la domanda lo giustifica.
- **Per InWiMe:** scambiare **attivazioni e risultati** anziché pesi interi a ogni decode, valutando dimensioni dei messaggi e sincronizzazioni.
- **Attenzione:** rete, gating, load imbalance e top-k riducono il guadagno; non cambiare la matematica del routing.

---

## 6. Hardware ed esecuzione eterogenea: piste da tenere visibili, non da imporre

| Tecnologia / famiglia | Fonte | Spunto | Limite rispetto al baseline InWiMe |
|---|---|---|---|
| GPU NVIDIA CUDA, cuBLAS, kernel BF16 | [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | Accelerare davvero GEMM/attenzione e overlappare I/O | VRAM, driver e capacità specifiche; sempre CPU fallback |
| Tenstorrent Blackhole PCIe | [specifiche ufficiali](https://docs.tenstorrent.com/aibs/blackhole/index.html) | BF16 e memoria locale; possibile backend futuro | Richiede infrastruttura/host ben maggiori delle 16 GB del test; piattaforma distinta |
| Hailo-10H / AI HAT+ 2 | [Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) | LLM piccoli su NPU, potenziale nodo eterogeneo | Formati/compilazione/memoria locali; non esegue automaticamente checkpoint BF16 enormi |
| Hailo-8/8L e NPU vision-oriented | [Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) | Possibili compiti ausiliari | Non backend generico per la matematica di grandi LLM BF16 |
| Akida/Metis/MemryX/Coral/DEEPX | [categoria discussa: verificare singole schede e SDK prima della scelta](https://www.hailo.ai/) | Acceleratori specializzati a basso consumo | Non assumere esecuzione generica Transformer originale senza supporto esplicito |
| SBC / mini-PC Orange Pi, Banana Pi, Radxa | [Orange Pi](http://www.orangepi.org/), [Banana Pi](https://www.banana-pi.org/), [Radxa](https://radxa.com/) | NVMe locale, rete, nodi accendibili al bisogno | ARM e NPU richiedono backend dedicati; fase di distribuzione futura, non requisito x86 attuale |

**Nota sull'idea originale di cluster SSD-first:** alimentazione e accensione dei nodi su domanda, RAM usata come cache del working set, NVMe come archivio locale dei pesi e rete usata preferibilmente per intermedi/attivazioni erano idee già discusse. Sono ipotesi di architettura futura: occorrono misure su consumi a riposo, avviamento, failure recovery, bandwidth, scheduling e costo totale prima di rivendicare un vantaggio su singolo host.

---


## 6 bis. Integrazioni recuperate dall'audit: MoE/SSD e progetti particolarmente vicini a InWiMe

### F32 — TinyGiant: expert-aware SSD streaming con modelli oltre la RAM [D][R]
- **Sorgenti:** [TinyGiant (codice)](https://github.com/jerryjokesalot/tinygiant), [RFC llama.cpp #27149](https://github.com/ggml-org/llama.cpp/discussions/27149).
- **Tecniche:** consultare i soli esperti selezionati; tenere stabilmente residenti router e parte condivisa; prefetch di miss I/O; buffer limitati; profiling di esperti reali; benchmark con BLAS e confronto cache/no cache.
- **Per InWiMe:** uno dei precedenti pubblici **più diretti** per motore SSD-first su 8–16 GiB di RAM. Studiare codice di mappatura expert→offset, misure di amplificazione, tracciamento hit/miss, overlap e differenza fra simulazione del calcolo ed esecuzione end-to-end.
- **Attenzione:** i risultati del prototipo sono ottenuti su configurazioni e versioni differenti; il progetto include ottimizzazioni Q4 e Apple/ARM. Non considerare i suoi tok/s una previsione per x86 BF16 originale. La stessa RFC distingue il PoC iniziale con routing/calcolo simulato dal successivo codice con BLAS e verifica dell'inferenza.

### F33 — Expert-contiguous layout: ridurre l'amplificazione dell'I/O selettivo [D][R]
- **Sorgenti:** [TinyGiant, README e tool `nws_expert_relayout.py`](https://github.com/jerryjokesalot/tinygiant), [commenti tecnici sulla disposizione degli esperti](https://github.com/ggml-org/llama.cpp/discussions/23324).
- **Tecniche:** riorganizzare eventualmente gate/up/down per esperto in intervalli fisici contigui, misurare page faults e byte effettivamente letti rispetto ai byte utili. Il rapporto tra layout di GGUF, dequantizzazione, mmap e `pread` cambia l'amplificazione delle letture.
- **Per InWiMe:** valutare il **layout fisico** come possibile profilo ottimizzato, senza imporre un formato checkpoint proprietario. Confrontare *zero-copy* su formato originale con un eventuale indice/sidecar o layout derivato ricostruibile.
- **Attenzione:** il progetto riporta un caso di riduzione dell'amplificazione dei page fault mediante relayout; **non è una legge per tutti i GGUF**. Lo stesso autore osserva che letture `pread` di esperti quantizzati già contigui possono non beneficiare del relayout. Mai trasformare silenziosamente precisione o rappresentazione originale.

### F34 — PoC Metal di expert paging con pool a slot e ID remapping [R]
- **Sorgente:** [llama.cpp discussione #23324](https://github.com/ggml-org/llama.cpp/discussions/23324).
- **Tecniche:** pool GPU/shared-memory di N slot, intercettazione dei router ID, miss gestiti dalla CPU via `pread`, rimappatura degli indici esperto, sincronizzazione attraverso eventi GPU, LRU e riutilizzo controllato dei buffer.
- **Per InWiMe:** analogo concreto per una futura GPU con poca VRAM, particolarmente utile per progettare la durata dei buffer e i completamenti asincroni senza alterare `top-k`.
- **Attenzione:** PoC Metal su Apple Silicon con checkpoint quantizzato; non è un backend CUDA/x86 già pronto. I risultati pubblicati sono specifici dei modelli e dell'hardware usati.

### F35 — Guanaco: utilizzo consapevole della page cache con `madvise` [R]
- **Sorgente:** [Atomic-Germ/Guanaco](https://github.com/Atomic-Germ/Guanaco).
- **Tecniche:** conservare la mappatura dei pesi con `mmap`, impartire suggerimenti per accessi random e promozione/dismissione pagine (`MADV_RANDOM`, `MADV_WILLNEED`, `MADV_DONTNEED`), cache dei soli esperti hot senza duplicare necessariamente tutti i dati in buffer indipendenti.
- **Per InWiMe:** alternativa tecnica alla gestione totalmente esplicita della cache; misurare quando l'OS è utile e quando crea letture involontarie o paging dannoso. Utile anche per confronti con cache disabilitata.
- **Attenzione:** `madvise` esprime suggerimenti al kernel; non è garanzia di un tetto rigido di RSS o di residenza. Il progetto è un'implementazione sperimentale legata a Linux/llama.cpp; validare i risultati in autonomia.

### F36 — llama.cpp: implementazioni e RFC di GPU expert cache / SSD streaming [R]
- **Sorgenti:** [issue #20757 (panoramica e collegamenti)](https://github.com/ggml-org/llama.cpp/issues/20757), [PR #25294 per streaming da disco](https://github.com/ggml-org/llama.cpp/pull/25294), [PR #27861 cache GPU LRU](https://github.com/ggml-org/llama.cpp/pull/27861), [codice `llama-moe-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-moe-cache.cpp), [RFC #28248 slot persistenti](https://github.com/ggml-org/llama.cpp/discussions/28248).
- **Tecniche:** cache GPU con slot residenti e rimappatura ID; upload degli esperti richiesti con throttling; politiche LRU, heatmap o aging; prefill wave-partitioned; streaming selettivo quando il modello eccede RAM.
- **Per InWiMe:** le API e il codice sono più istruttivi delle sole pagine generali di llama.cpp: esaminare il punto di integrazione tra router, trasferimento, rappresentazione dei pesi e calcolo con slot limitati.
- **Attenzione:** issue, PR, commit su `master` e fork sperimentali hanno **stati diversi** (aperti, chiusi o integrati). Verificarne lo stato alla data di utilizzo; non considerare ogni discussione una funzionalità già rilasciata.

### F37 — Qwen3.8-Flash-Next: PLE / tabella n-gram letta lazy per righe [D][R]
- **Sorgente:** [esempio riproducibile di configurazione `llama.cpp`](https://github.com/CostanzoPadovano/qwen3.8-flash-next-local/blob/main/README.md).
- **Tecniche:** un grande tensore per-layer embedding (PLE) può essere consultato selettivamente con `--tensor-read-lazy on` e `mmap`; collocazione esplicita su host/GPU; comparazione con il caricamento eager della tabella intera.
- **Per InWiMe:** dimostrazione del valore di un **contratto di accesso a slice/riga**, distinto dall'accesso per layer e per esperto. Potenziale caso ad alta località variabile.
- **Attenzione:** la tabella PLE (pesi appresi del modello) **non è** la cache di `ngram-mod`, che è invece un metodo separato di speculative decoding model-free. Il progetto documentato usa un checkpoint GGUF quantizzato e hardware specifico: è solo un riferimento di lettura selettiva, non una prova di supporto universale a BF16.

### F38 — Text-calibrated pinning degli esperti vs prefetch predittivo [D][R]
- **Sorgenti:** [TinyGiant](https://github.com/jerryjokesalot/tinygiant), [discussione #23324](https://github.com/ggml-org/llama.cpp/discussions/23324).
- **Tecniche:** profilare un breve tratto di testo per contare gli esperti realmente attivati, preriscaldare/pinnare quelli più frequenti per layer, combinare con LRU; misurare hit rate *per byte* e costo della calibrazione.
- **Per InWiMe:** strategia meno complessa di un predittore neurale per la prima versione delle politiche MoE; da confrontare con LRU, LFU e caching non calibrato.
- **Attenzione:** risultati su singoli carichi/implementazioni, spesso quantizzati. Non assumere che dieci token iniziali siano rappresentativi di ogni prompt, cambio di tema o architettura MoE.

### F39 — Fusion di caricamento, decodifica dtype e calcolo [R][P]
- **Sorgente:** [TinyGiant, implementazioni fused Q4](https://github.com/jerryjokesalot/tinygiant).
- **Tecniche:** consumare blocchi dei pesi e accumulare direttamente nei registri del kernel, evitando materializzazioni temporanee di grandi matrici in FP32/FP64 nella RAM host.
- **Per InWiMe:** studiare un **kernel BF16-originale** che converta la tile durante il calcolo e accumuli nel dtype dichiarato senza ricostruire l'intero tensore; utile anche per evitare overhead di Python.
- **Attenzione:** i numeri di TinyGiant derivano prevalentemente da percorsi **Q4+Q8 ARM**. La proposta BF16 qui è una **trasposizione ipotetica** del principio di fusione, non un benchmark BF16 né un'autorizzazione a quantizzare il checkpoint.

### F40 — SpikingBrain 1/2: sottotecniche distinte per nuovi adapter di modello [D][R]
- **Sorgenti:** [SpikingBrain 2.0 repo e architettura](https://github.com/BICLab/SpikingBrain2.0), [FlashMoBA](https://github.com/mit-han-lab/flash-moba).
- **Tecniche:** distinguere SpikingBrain 1.0 (ad esempio W8ASpike, architettura ibrida e attenzione a finestra) da SpikingBrain 2.0 (Dual-Space Sparse Attention, DSSA, SSE su Gated DeltaNet, MoBA e codifiche di attivazione alternative). Studiare requisiti di stato ricorrente e opportunità di parallelizzare il prefill.
- **Per InWiMe:** futuro supporto a *famiglie di operatori differenti* e relativi stati, mantenendo il decoder Transformer corrente solo come prima fixture.
- **Attenzione:** FP8, INT8/spiking e adattamenti dei modelli sono proprietà **di quei modelli**; non sono ottimizzazioni senza perdita da applicare arbitrariamente a checkpoint densi BF16.

### F41 — DeepSeek Harness e API compatibile: integrazione d'uso, non motore dei pesi [D][R]
- **Sorgenti:** [DeepSeek Harness sito](https://www.deepseek.com/harness/en/), [repository ufficiale](https://github.com/deepseek-ai/deepseek-harness).
- **Tecniche trasferibili:** disaccoppiare runtime dei modelli e client agentico; esporre dopo il core un endpoint OpenAI-compatible, streaming dei token, cancellazioni e telemetria.
- **Per InWiMe:** percorso futuro per verificare che il motore sia utilizzabile da un harness esterno senza costruire ora una nuova interfaccia agentica.
- **Attenzione:** DeepSeek Harness è un **agent harness**, non una soluzione di SSD swapping, multi-NVMe o calcolo BF16; il catalogo lo registra solo per interoperabilità.

### F42 — CPU eterogenea e affinità: nucleo P/E, NUMA, worker e I/O [R][P]
- **Sorgente:** [discussione llama.cpp su caching e benchmarking su CPU ibride](https://github.com/ggml-org/llama.cpp/discussions/24528).
- **Tecniche:** profilare l'affinità dei worker su core veloci e/o efficienti, isolare I/O da calcolo, considerare topologia NUMA e core contention; misurare performance/watt invece di assumere che più thread diano più token/s.
- **Per InWiMe:** quando ottimizzeremo CPU x86, distinguere limiti dell'algoritmo da scheduling sfavorevole dell'OS, senza specializzarsi sul Katana.
- **Attenzione:** esempi e numeri dei forum sono specifici della macchina e del modello. Non imporre una policy di affinità unica a tutti gli x86.

---

## 7. Tecniche da studiare con maggiore urgenza, senza cambiare il programma corrente

### Ora / immediatamente dopo la verifica dell'incremento 2

1. **Correctness + bounded memory:** mantenere oracle indipendente, conservazione del dtype originale, controllo dei budget e telemetria separata; baseline CPU sincrona.
2. **Confronto delle primitive di lettura:** `pread` / thread pool / `io_uring` e opzione `mmap`/direct I/O; misurare p50/p95/p99, byte richiesti vs byte effettivamente letti, page cache e RSS.
3. **Dimensionamento tile e memoria:** evitare che la piccola fixture o lo slot da 40 B diventino parametri permanenti; variazioni su tile di calcolo, intervalli I/O, unità di cache.
4. **Sovrapposizione di letture e computazione:** prima CPU, poi GPU; CUPTI e marcatori compatibili per dimostrare overlap reale, non dedotto da tempi aggregati.

### Dopo un primo runtime GPU verificato

5. **GPU optional e precisione:** BF16/FP16/accumulo dichiarato, tiled GEMM, staging pinned e fallback CPU; selezione dei kernel secondo la GPU reale.
6. **MoE locality:** trace dei router su modelli veri, hot-expert cache, prefetch non vincolante, caricamento solo degli esperti richiesti e misurazione miss per token.
7. **KV/contesti lunghi:** PagedAttention e cache tiered; distinguere tecniche lossless (recompute/paging) da eviction semantica approssimata.
8. **Batching/speculazione:** solo dopo stabilità del target model, con metriche per singola sessione e aggregate.

### Per la futura ricerca distribuita

9. **Repliche indipendenti come baseline obbligatoria** a parità di quattro server, rete, storage, modello e carico.
10. **Confronto fra pipeline/tensor/expert parallelism:** ottimizzare latency vs aggregated throughput separatamente.
11. **Compute-near-weights e multi-tier remoto:** indagare traffico delle attivazioni vs streaming dei pesi, collocazione esperti, caching cooperativo e rete.
12. **Distribuzione elastica:** accensione/spegnimento dei nodi quando conviene davvero sul TCO, mantenendo isolamento della sessione e failure recovery.

---

## 8. Registro delle ipotesi e dei benchmark da progettare

| ID | Ipotesi da falsificare | Misure minime | Sorgenti/tecniche correlate |
|---|---|---|---|
| H01 | `mmap` batte I/O esplicito nel carico selettivo reale | p50/p95/p99, RSS, page faults, byte fisici, token/s | F01, F06 |
| H02 | `io_uring` offre overlap utile su NVMe interno | completamenti, CPU overhead, bandwidth, p99 | F06, F03 |
| H03 | Buffer > 1 e tile regolabili migliorano token/s senza superare budget | token/s, overlap, picco RAM/pinned/VRAM, read amplification | F02, F07 |
| H04 | Un secondo NVMe aumenta banda effettivamente utilizzabile | throughput, bus/controller contention, p99 | F08 |
| H05 | Il routing MoE riduce i byte letti rispetto ai modelli densi comparabili | byte/token, hit/miss expert, attivi per token | F10, F18 |
| H06 | Cache hot-expert conviene rispetto a cache uniforme | hit-rate *per byte*, tempo di decode, eviction, skew | F10, F11 |
| H07 | Prefetch degli esperti nasconde latenza senza starvation | prefetch inutile, wait obbligatorio, timeout, budget | F10, F18 |
| H08 | Prefill/decode disaggregati battono l'aggregato in uno specifico carico | TTFT, TPOT, goodput, rete KV, p99 | F23, F24 |
| H09 | 4 repliche indipendenti producono più token/s aggregati di 1 modello TP su 4 nodi | tok/s totali, tok/s per sessione, Watt, GPU/CPU/SSD/RETE | F28, F29 |
| H10 | Il modello distribuito diventa preferibile quando RAM aggregate evitano più letture NVMe | bytes NVMe evitati, bytes rete introdotti, latenza/token | F26, F31 |
| H11 | Speculative decoding risparmia passaggi NVMe netti | rate accettazione, byte/token, overhead draft, output identico | F17 |
| H12 | Paged KV permette più sessioni senza crescita incontrollata | RAM/KV, fragmentation, TTFT/TPOT, cancel/failure | F19–F23 |
| H13 | FlashAttention è migliore di attenzione naive a parità di semantica | intermedi, errori vs oracle, tempo, peak memory | F14 |
| H14 | L'algoritmo del coordinatore locale A diventa il bottleneck | tempo controllo/completamento, worker idle, queue depth | documentazione InWiMe |

| H15 | Layout degli esperti e strategia `mmap`/`pread` cambiano l'amplificazione di I/O | bytes/logical vs physical, page faults, CPU, hit | F32–F35 |
| H16 | Slot/cache MoE rimappati consentono GPU acceleration utile entro VRAM piccola | slot, copie H2D, kernel stalls, tok/s e accuratezza | F34, F36 |
| H17 | `madvise` mirato compete con cache esplicita senza infrangere i budget | RSS, page cache, faults, read amplification, p99 | F35 |
| H18 | Text-calibrated pinning anticipa meglio gli esperti richiesti di LRU/LFU pure | hit per byte, warmup cost, workload shift, tok/s | F38 |
| H19 | Lettura lazy delle righe PLE elimina il caricamento completo in modelli adatti | bytes/lookup, latenza, fault, prefill/decode | F37 |
| H20 | Fusion BF16 tile → accumulo riduce i temporanei e aumenta throughput CPU | correttezza vs oracle, RSS, bytes memoria, tok/s | F39 |
| H21 | Core affinity/topologia migliora efficienza CPU su host eterogenei | tok/s, watt, latenza e contention | F42 |

### Regole per registrare una prova

Per ciascuna prova salvare: **commit, modello e dtype, checksum del checkpoint, hardware e topo storage, OS/filesystem, RAM/VRAM, frequenze e condizioni termiche se pertinenti, cache cold/warm, modalità di rete, thread e parallelismo, tile, numero di sessioni, prompt/output length, numero ripetizioni, distribuzioni di latenza, byte logici/fisici, potenza se disponibile, correttezza numerica**. Quando una misura manca, marcarla come `non misurata`, non dedurla.

**Metriche distinte**: TTFT (tempo al primo token), ITL/TPOT (inter-token/per-output-token latency), token/s per singola sequenza, token/s aggregati, richieste/s, goodput entro SLO, percentuale di hit KV e pesi, GB trasferiti per token, energia per token, errori e recuperi. Usare baseline **stesso hardware e stesso budget totale**.

---

## 9. Bibliografia compatta per area (link primari)

**Storage / out-of-core:** [llama.cpp](https://github.com/ggml-org/llama.cpp) · [AirLLM](https://github.com/lyogavin/airllm) · [ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference) · [DeepNVMe](https://www.deepspeed.ai/tutorials/deepnvme/) · [FlexGen](https://github.com/FMInference/FlexGen) · [Apple LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) · [liburing](https://github.com/axboe/liburing) · [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/index.html) · [safetensors](https://github.com/huggingface/safetensors).

**MoE / accelerazione:** [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) · [PowerInfer](https://powerinfer.ai/) · [PowerInfer-2](https://powerinfer.ai/v2/) · [DwarfStar](https://dwarfstar.sh/) · [FlashAttention](https://github.com/Dao-AILab/flash-attention) · [FlashInfer](https://github.com/flashinfer-ai/flashinfer) · [SpikingBrain 2.0](https://github.com/BICLab/SpikingBrain2.0) · [FlashMoBA](https://github.com/mit-han-lab/flash-moba).

**KV e serving:** [vLLM](https://github.com/vllm-project/vllm) · [SGLang](https://github.com/sgl-project/sglang) · [LMCache](https://github.com/LMCache/LMCache) · [Mooncake](https://github.com/kvcache-ai/Mooncake) · [H2O](https://github.com/FMInference/H2O) · [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) · [DistServe OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · [Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/).

**Integrazioni aggiuntive da recupero:** [TinyGiant](https://github.com/jerryjokesalot/tinygiant) · [llama.cpp SSD expert streaming #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) · [llama.cpp Metal paging #23324](https://github.com/ggml-org/llama.cpp/discussions/23324) · [Guanaco](https://github.com/Atomic-Germ/Guanaco) · [llama.cpp expert cache #20757](https://github.com/ggml-org/llama.cpp/issues/20757) · [Qwen3.8 lazy PLE](https://github.com/CostanzoPadovano/qwen3.8-flash-next-local) · [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

**Distribuzione:** [Petals](https://github.com/bigscience-workshop/petals) · [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) · [Apple MLX multi-Mac](https://machinelearning.apple.com/research/mlx-distributed-inference) · [Ray Serve](https://github.com/ray-project/ray) · [llm-d](https://github.com/llm-d/llm-d) · [NIXL](https://github.com/ai-dynamo/nixl) · [NVIDIA Dynamo](https://developer.nvidia.com/dynamo).

---

## 10. Politica di manutenzione del catalogo

Ogni nuova tecnica trovata durante sviluppo o benchmark va registrata con **nome, link primario, principio trasferibile, prerequisiti/hardware, compatibilità con dtype originale, metrica da migliorare, possibile regressione, fase e stato**. Se un test smentisce un'idea, conservarne la voce e aggiungere l'esito: una scelta scartata motivatamente è informazione utile.

Usare stati: **Da studiare** → **Letta/compresa** → **Ipotesi definita** → **Benchmarkata** → **Integrata** oppure **Scartata con motivazione**. Un riferimento di terzi non dimostra prestazioni su InWiMe. Non implementare una tecnica soltanto perché appare in questo catalogo.

**Nota di completezza:** anche la versione 1.2 è un inventario sottoposto ad audit, non una certificazione di completezza assoluta. Le conversazioni non integralmente accessibili potrebbero contenere altri riferimenti. Le misure di terzi sono attribuite ai rispettivi autori e non sostituiscono prove InWiMe.


---

## 11. Secondo audit di completezza: lacune effettive della versione 1.1

**Origine del secondo audit:** confronto testuale tra questo catalogo e `PROJECT_BRIEF.md`, `HANDOFF-2026-11-05-benchmark-v2.md`, l'architettura, i messaggi di progetto recuperabili e documentazione pubblica consultata il 7 novembre 2026. Non è stato possibile certificare la lettura integrale di *tutte* le conversazioni storiche. I riferimenti seguenti sono aggiunte, **non nuove decisioni architetturali**.

**Omissioni chiaramente documentate prima dell'audit**: [Colibri](https://github.com/JustVugg/colibri), [WARP](https://github.com/sqliteai/warp), il [porting CPU di SpikingBrain](https://github.com/francescodifilippo/spikingbrain-cpu) e il riferimento al [caso FLACache/decode SpikingBrain 2.0](https://github.com/BICLab/SpikingBrain2.0/issues/7). I primi due erano **esplicitamente citati in `PROJECT_BRIEF.md`**; pertanto la precedente dichiarazione di copertura ampia nascondeva una lacuna prioritaria.

**Nuovi riferimenti emersi nella verifica, non attestati come già discussi**: KTransformers/KT-Kernel, SPDK, ServerlessLLM, Glinthawk, oneDNN e la documentazione ufficiale sulle restrizioni di CUDA Unified Memory in WSL. L'inclusione di questi elementi è un ampliamento utile, non la prova che fossero già fra le nostre decisioni.

### F43 — Colibri: tre livelli di residenza, router-aware streaming, calcolo concorrente [D][R]
- **Fonte:** [Colibri / repository e documentazione](https://github.com/JustVugg/colibri).
- **Tecniche:** pesi condivisi/densi preferibilmente residenti; esperti *hot* su GPU, *warm* in RAM e *cold* su NVMe; router→unione esperti→collocazione→overlap→apprendimento dell'hot set. Letture `pread` raggruppate delle matrici di un esperto, pool di lettori, esperti CPU calcolati parallelamente a quelli GPU disponibili, cache per layer e preferenze misurate da workload, lookahead, multi-NVMe e profili NUMA.
- **Per InWiMe:** confrontare la separazione della residenza *per tipo di peso* con una cache generica; misurare sovrapposizione reale fra I/O, esperti CPU e GPU e costo dei dati letti per token. Studiare il coordinamento di esperti mancanti senza modificare gli ID selezionati dal router.
- **Attenzione:** molti risultati e container Colibri sono **quantizzati**; il fatto che il suo *principio* sia applicabile non dimostra che velocità, memoria o kernel si trasferiscano a checkpoint **BF16 originali**. I benchmark nel README sono risultati dichiarati dal progetto, da riprodurre, e il supporto varia per architettura.
- **Priorità:** **alta**, subito dopo il runtime CPU corretto; mantenere l'implementazione di Colibri come riferimento esterno, non come vincolo.

### F44 — WARP (ex WASTE): pesi condivisi residenti, esperti su NVMe e cache limitata [D][R]
- **Fonti:** [WARP / codice](https://github.com/sqliteai/warp), [misure e limiti](https://github.com/sqliteai/warp/blob/main/docs/TECHNICAL.md), [criteri sperimentali](https://github.com/sqliteai/warp/blob/main/docs/GATES.md).
- **Tecniche:** inferenza in C con modello MoE enorme, parti condivise residenti e lettura dal disco dei soli esperti instradati; cache sotto budget, analisi del rapporto hit-rate/byte letti/token; distribuzione dei banchi degli esperti su più NVMe indipendenti, con verifica di equivalenza dei logits fra collocazioni.
- **Per InWiMe:** benchmark di I/O costruiti sul pattern reale degli esperti e non soltanto su scansioni sequenziali; costo della residenza obbligatoria del tronco del modello; soglia oltre la quale un cache budget aggiuntivo non compra più velocità; distinzione tra uno e più dischi realmente indipendenti.
- **Attenzione:** il repository discute container convertiti/compressi e dimostra casi su hardware diversi; non attribuire a WARP una garanzia universale di **BF16 originale intatto** né generalizzare i suoi token/s al Katana. Le sue stime e misure sono risultati degli autori.
- **Priorità:** **alta** per MoE + NVMe e metodologia di falsificazione.

### F45 — KTransformers / KT-Kernel: CPU-GPU MoE, backend BF16 nativo e fusion [R]
- **Fonti:** [KTransformers](https://github.com/kvcache-ai/ktransformers), [KT-Kernel](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel), [backend AMX](https://ktransformers.net/en/docs/optimization-techniques/amx).
- **Tecniche:** partizionamento CPU/GPU degli esperti; kernel di inferenza su CPU AVX2, AVX512 e AMX dove supportati; percorsi BF16 nativi distinti dai percorsi INT8/INT4; fusione gate/up e aggregazione dei task MoE; task stealing per skew del routing.
- **Per InWiMe:** separare *scelta del dispositivo* da *scelta del kernel* e *precisione nativa dei dati*; valutare un adapter CPU BF16 ad alte prestazioni per hardware futuro e un fallback più generale per hardware ordinario.
- **Attenzione:** **AVX512/AMX non vanno dati per disponibili sull'i7-12700H**; verificare le capacità reali della CPU. Un percorso CPU non è automaticamente lossless se converte pesi in INT8/INT4. Le prestazioni dichiarate per Xeon non sono stime valide per laptop.
- **Priorità:** media, quando ottimizzeremo gli operatori CPU e la computazione MoE.

### F46 — ServerlessLLM: formato di caricamento ottimizzato e località dei checkpoint [R]
- **Fonti:** [ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM), [OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/fu), [store multi-tier](https://serverlessllm.github.io/docs/store/quickstart/).
- **Tecniche:** caricamento in parallelo fra SSD, DRAM e GPU; formato dei checkpoint ottimizzato per l'I/O; scheduling con consapevolezza della collocazione, avvio rapido di più modelli, migrazione live delle inferenze.
- **Per InWiMe:** distinzione fra *tempo di avvio* e *byte per token*; attenzione a posizionamento dei modelli nelle future repliche multi-nodo. Può essere utile anche per preparazione di indici/sidecar ricostruibili.
- **Attenzione:** ottimizzare il caricamento completo di un modello per un server GPU **non equivale** a streaming continuo dei pesi sotto la RAM. Il formato derivato non deve trasformare o quantizzare silenziosamente i dati.
- **Priorità:** bassa ora, più alta con molti modelli e sessioni.

### F47 — SPDK e NVMe over Fabrics: accesso userspace a code NVMe [R]
- **Fonti:** [SPDK](https://github.com/spdk/spdk), [introduzione ufficiale](https://spdk.io/doc/about.html), [driver NVMe](https://spdk.io/doc/nvme.html).
- **Tecniche:** I/O NVMe asincrono da userspace, queue pairs, polling, zero-copy e gestione esplicita dei domini di storage; NVMe-oF per trasporto remoto di dispositivi a blocchi.
- **Per InWiMe:** candidato **alternativo** a `pread`/`io_uring` qualora i profili reali evidenzino limiti di syscall, code e copie; possibile storage remoto che non richiede subito calcolo remoto.
- **Attenzione:** infrastruttura e permessi molto più invasivi di un normale file POSIX; può richiedere gestione esclusiva dei controller, hugepages e amministrazione specifica. **Non** è la prima implementazione da usare e NVMe-oF non è inferenza distribuita.
- **Priorità:** ricerca futura, dopo misure `pread`/`io_uring` su NVMe nativo.

### F48 — CUDA Unified Memory e mapped host memory: confrontare, non assumere SSD = VRAM [R]
- **Fonti:** [NVIDIA CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html), [tipi e limiti di memoria](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).
- **Tecniche:** managed memory, page migration, prefetch, memory advice, mapping di memoria host e capacità di oversubscription in configurazioni che la supportano.
- **Per InWiMe:** misurare copie esplicite e tile GPU contro eventuali modalità di memoria gestita, conservando CPU fallback e ownership corretti. La disponibilità della funzione dipende da OS, GPU, driver e modalità di esecuzione.
- **Avvertenza verificata, con distinzione importante:** il supporto limitato riguarda **CUDA sullo stack driver Windows, ereditato da WSL2**, e **non CUDA Unified Memory in generale**. NVIDIA documenta che, su Windows/WSL, non sono disponibili la GPU-memory oversubscription e l'accesso CPU/GPU concorrente nel paradigma di Unified Memory limitata. Su **Linux nativo con hardware/driver idonei**, la Unified Memory completa può supportare l'oversubscription tramite migrazione di pagine e HMM/ATS. Verificare via capability query, non dall'etichetta della GPU: `cudaDevAttrConcurrentManagedAccess`, `cudaDevAttrPageableMemoryAccess`. In ogni caso **Unified Memory non trasforma automaticamente SSD in VRAM**: il caricamento file, il paging host e i costi di migrazione restano problemi separati. Fonti: [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/), [Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html), [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
- **Priorità:** criterio negativo di progettazione subito; esperimenti su Linux nativo soltanto se motivati.

### F49 — Glinthawk: attenzione/KV su un tier CPU separato [R]
- **Fonti:** [paper Glinthawk](https://arxiv.org/abs/2501.11779), [codice](https://github.com/microsoft/glinthawk).
- **Tecniche:** separare le parti altamente GPU-intensive del forward e il calcolo dell'attenzione con grande KV cache; architettura a due tier CPU/GPU che fa scalare la memoria KV separatamente da quella dei pesi.
- **Per InWiMe:** riferimento per l'analisi *stato KV e attenzione vs pesi* e per gli acceleratori asimmetrici; ulteriori confronti utili in serving a batch e contesto lungo.
- **Attenzione:** progettato soprattutto per **throughput offline** e batch, con tolleranza alla latenza; non promette miglioramento per un solo utente in decode sincrono.
- **Priorità:** media nel futuro serving multi-sessione.

### F50 — Porting indipendente SpikingBrain 2.0 CPU e regressione FLACache [D]
- **Fonti:** [spikingbrain-cpu](https://github.com/francescodifilippo/spikingbrain-cpu), [SpikingBrain2.0 upstream](https://github.com/BICLab/SpikingBrain2.0), [issue #7 riportata nel brief](https://github.com/BICLab/SpikingBrain2.0/issues/7).
- **Tecniche:** confronto bit/numerico delle operazioni CPU contro specifica del modello, gestione dello stato ricorrente e correttezza delle posizioni durante il decode; nel brief precedente è riportata una possibile anomalia di indicizzazione rotary/FLACache, da verificare contro upstream prima di riusare codice.
- **Per InWiMe:** *test di regressione modello-specifici* quando verranno aggiunti adapter con DeltaNet, GLA, cache ricorrenti o attenzione mista; separare un bug matematico dai problemi dello storage.
- **Attenzione:** l'issue specifica è una segnalazione storica del progetto, **non una conferma qui del suo stato corrente**; l'esatta semantica di SpikingBrain non deve essere sostituita a quella di DeepSeek/Qwen/GLM. L'assenza di checkpoint ufficiali per alcune varianti spiking impedisce di dichiarare equivalenza con pesi inesistenti.
- **Priorità:** alta quando si aggiungono adapter non Transformer convenzionali.

### F51 — oneDNN, ISA dispatch e differenti costi FP32/BF16 [R]
- **Fonti:** [oneDNN](https://github.com/uxlfoundation/oneDNN), [tabella dtype/ISA](https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html), [rilascio 2026](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html).
- **Tecniche:** selezione runtime di kernel x86 in base all'ISA, GEMV/GEMM con caratteristiche diverse in prefill e decode, BF16 nativo solo con supporto hardware/implementativo adeguato, fallback FP32/FP64 corretto.
- **Per InWiMe:** evitare un unico kernel per `n=1` e batch grandi; misurare throughput, temporanei e correttezza per ogni percorso. Ottimizzare CPU generiche senza assumere AMX.
- **Attenzione:** le modalità di accumulo possono variare e il supporto alla dtype non implica identico comportamento numerico. Una cache di pesi BF16 non autorizza conversioni persistenti non dichiarate.
- **Priorità:** alta per prestazioni CPU dopo il primo forward corretto.

### F52 — Stato del contesto riutilizzabile: KV persistente, restart e compressioni native del modello [R]
- **Fonte:** [Colibri, sezioni attention/agent context](https://github.com/JustVugg/colibri).
- **Tecniche:** ripresa di sessioni mantenendo o recuperando lo stato KV, evitamento del ricalcolo di prompt già processati, utilizzo di MLA o strutture KV compatte *quando previste nativamente dal modello*; stato ricorrente distinto dal KV classico.
- **Per InWiMe:** stato su SSD solo con un contratto esplicito di versione, identità del checkpoint, lunghezza posizionale, commit di token e integrità. Un contesto sospeso è utile nei workflow agentici; evita re-prefill su ogni chiamata.
- **Attenzione:** ridurre la KV di un modello con una nuova approssimazione cambia la semantica. Una struttura MLA già richiesta dal modello **non** è una quantizzazione introdotta dal runtime.
- **Priorità:** dopo correttezza multi-sessione e state management.

### F53 — Storage distribuito vs compute distribuito: NVMe-oF come controesempio [R][P]
- **Fonti:** [SPDK NVMe-oF](https://spdk.io/doc/nvme.html), [NVIDIA GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage), [Petals](https://github.com/bigscience-workshop/petals).
- **Tecniche:** collocazione remota degli shard, lettura di intervalli su rete e, separatamente, esecuzione remota del calcolo *vicino ai pesi*.
- **Per InWiMe:** misurare due configurazioni diverse: *recuperare byte remoti e calcolare localmente* vs *spostare attivazioni e calcolare sul nodo con i pesi*. Ogni configurazione richiede metriche separate per rete e disco, senza assunzione che una debba essere migliore.
- **Attenzione:** un backend NAS/NVMe-oF **non** crea automaticamente pipeline/tensor/expert parallelism. Repliche indipendenti rimangono la baseline per più sessioni.
- **Priorità:** studio futuro per inferenza multi-server.

### F54 — Vulkan, Resizable BAR, staging e scheduling misto CPU/GPU [R]
- **Fonte:** [Colibri, documentazione CUDA/Vulkan](https://github.com/JustVugg/colibri).
- **Tecniche:** GPU opzionale non necessariamente CUDA; gestione del limite di mappatura PCIe in assenza di Resizable BAR; slot di staging espliciti; alcuni esperti in GPU e miss su CPU con elaborazione concomitante; verifiche di equivalenza numerica fra backend.
- **Per InWiMe:** possibile secondo backend GPU in futuro, soprattutto per GPU non NVIDIA; distinguere capacità di mappatura dall'effettiva VRAM, misurare costo PCIe e di dispatch e preservare un percorso CPU-only totale.
- **Attenzione:** far funzionare un kernel su Vulkan non dimostra speedup. Non consentire che vincoli del Katana determinino l'API universale di InWiMe.
- **Priorità:** dopo CUDA/BF16 end-to-end e analisi di costi/benefici.

### 11.1 Confronto finale con la versione 1.1: copertura reale

| Tema | v1.1 | v1.2 | Provenienza della correzione |
|---|---|---|---|
| Colibri, routing-aware three-tier memory, mix CPU/GPU | **Assente** | F43/F54 | **Già citato nel brief originale** |
| WARP, MoE SSD paginato, banco esperti multi-NVMe | **Assente** | F44 | **Già citato nel brief originale** |
| Porting CPU SpikingBrain e issue FLACache | **Assenti** | F50 | **Già citati nel brief originale** |
| KTransformers BF16/AVX/AMX | **Assente** | F45 | Nuova ricerca |
| ServerlessLLM, caricamento checkpoint multi-tier | **Assente** | F46 | Nuova ricerca |
| SPDK, queue pairs e NVMe over Fabrics | **Assente** | F47/F53 | Nuova ricerca |
| CUDA Unified Memory su Windows/WSL: limiti reali | **Non discusso esplicitamente** | F48 | Nuova verifica ufficiale |
| Glinthawk, attention/KV su tier CPU | **Assente** | F49 | Nuova ricerca |
| oneDNN e selezione dinamica dell'ISA | **Assente** | F51 | Nuova ricerca |
| KV persistente/riuso di sessione, MLA nativa | **Solo in termini generici** | F52 | Nuova specificazione |

### 11.2 Nuove ipotesi di verifica (senza autorizzazione a implementare)

| ID | Ipotesi | Misura discriminante | Fonti |
|---|---|---|---|
| H22 | Cache per tipo di peso (trunk/esperti) batte residenza uniforme | byte/token, latenza, RSS, hit utili | F43, F44 |
| H23 | Letture raggruppate per esperto e compute CPU/GPU concorrente convengono | overlap reale, wall-time, bytes I/O, correttezza | F43 |
| H24 | Cache MoE a budget aggiuntivo ha rendimenti decrescenti | curva hit-rate vs GB vs token/s | F44 |
| H25 | Uno schema nativo BF16 MoE beneficia di operator fusion e task stealing | errori, token/s e rapporto GEMV/GEMM | F45, F51 |
| H26 | Accorciare il model load migliora davvero TTFT in multi-model | cold/warm TTFT, I/O amplificato | F46 |
| H27 | SPDK supera `io_uring` in un workload reale abbastanza grande da giustificare costo | p99, CPU core occupati, ops/s, watt | F47 |
| H28 | Managed memory competerebbe con copie esplicite su piattaforme che supportano oversubscription | RSS/VRAM/errore/tempo per tile; supporto effettivo | F48 |
| H29 | Separare attenzione/KV dal calcolo pesi aiuta batch e contesti lunghi | throughput per GPU, rete, memoria KV, TPOT | F49 |
| H30 | KV ripristinato evita lavoro netto negli agenti multi-turno | tempo ripresa, bytes KV, TTFT, correttezza stato | F52 |
| H31 | Compute-near-weights supera la lettura remota dei pesi | rete+NVMe/step, token/s/sessione, throughput | F53 |
| H32 | Vulkan rende utile una GPU senza sacrificare correttezza e CPU fallback | speedup vs CPU, deviazioni numeriche, PCIe | F54 |

### 11.3 Regole di completezza e classificazione

1. **Presente nel brief/handoff ma assente dal catalogo = omissione certa.** Non si può dichiarare completo il catalogo finché l'incrocio dei documenti di progetto non è esaurito.
2. **Presente sul web ma non nelle vecchie discussioni = nuovo candidato.** Etichettare chiaramente la provenienza; non retroattribuirlo alla storia del progetto.
3. **Esistente ma non verificato nel codice upstream = da approfondire.** Un README può descrivere un risultato senza dimostrare che il ramo attuale lo riproduca.
4. **Quantizzazione richiesta dal progetto terzo = tecnica trasferibile con caveat.** Nel percorso base di InWiMe le rappresentazioni originali rimangono obbligatorie.
5. **Benchmark locale / dei terzi / simulazione = livelli diversi di evidenza.** Non mischiare velocità di GEMM con token/s end-to-end o un singolo utente con throughput aggregato.
6. **Mantenere sempre un registro dei riferimenti da confermare** con URL, versione/commit, stato, modello, dtype, benchmark e ultima data di controllo.

### F55 — Schede PCIe/M.2 e SBC esplorate nel progetto distribuito del settembre 2026 [D][R]
- **Contesto storico:** nelle discussioni sul cluster HPC/SSD-first furono valutati **acceleratori da aggiungere a PC**, distinti dagli SBC e dagli HAT per Raspberry Pi. Il catalogo v1.1 riportava alcune famiglie ma non identificava bene i modelli, i form factor né il limite rispetto ai grandi pesi BF16.
- **Moduli PC/M.2 identificati:** [DEEPX DX-M1 M.2](https://deepx.ai/products/dx-m1/) (25 TOPS INT8, non un backend generico BF16 per modelli enormi); [MemryX MX3 M.2](https://developer.memryx.com/2p1/specs/M.2_datasheet.html) (acceleratore edge AI su PCIe); [Axelera Metis PCIe/M.2](https://docs.axelera.ai/sdk/reference/system/hardware/) (accelerazione principalmente in INT8). Verificare SDK, memoria realmente accessibile, dtype, limiti del modello e interfaccia di trasferimento prima di scegliere un backend.
- **SBC discussi separatamente:** [Orange Pi 5 Plus](http://www.orangepi.org/) (RK3588, 32 GB possibili, NVMe e doppia 2.5GbE: candidato storico), [Banana Pi BPI-M7](https://docs.banana-pi.org/en/BPI-M7/BananaPi_BPI-M7), [Radxa ROCK 5C](https://radxa.com/), oltre a ROCK 5B/5B+, Orange Pi 5 Max e Raspberry Pi 5/4. Le differenze di bus NVMe e rete influiscono più del numero teorico di TOPS sullo streaming dei pesi.
- **HAT, non scheda PCIe generica per PC:** [Raspberry Pi AI HAT+ 2 / Hailo-10H](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html), con 8 GB dedicati e 40 TOPS INT4. Raspberry Pi lo presenta per LLM/VLM fino a circa 6B parametri: utile per modelli piccoli/compiti ausiliari, **non** per ospitare da solo checkpoint BF16 da centinaia di GB. Hailo-8/8L e Coral sono soprattutto per visione.
- **Per InWiMe:** preservare la distinzione fra *nodo che conserva ed elabora una parte dei pesi*, *acceleratore per una sotto-operazione* e *nodo che serve intere sessioni*. L'architettura locale x86-64 non deve dipendere da un SBC ARM né da una singola NPU.
- **Priorità:** bassa ora, da recuperare nella valutazione hardware/energia del cluster; confrontare anche l'alternativa di un NVMe interno più capiente e delle repliche indipendenti.

**Ulteriore principio T39 — Valutazione degli acceleratori per capacità effettiva, non TOPS di marketing.** Confrontare dtype BF16/FP16 nativi, capacità e banda della memoria, trasferimenti PCIe, supporto SDK/LLM/MoE e prezzo/energia per token. La capacità INT8/INT4 su un task vision non equivale a un backend capace di eseguire il checkpoint originale.

**Ulteriore ipotesi H33 — Acceleratori add-in e SBC contro GPU/CPU commodity:** verificare quale fra calcolo locale, servizi ausiliari, e distribuzione per layer/esperti dia un miglioramento effettivo di token/s o joule/token, usando lo stesso modello, correttezza e costo complessivo.


---

## 12. Terzo audit: **solo tecniche e progetti già discussi** prima della ricerca del 7 novembre

Questa sezione serve a rispondere a una domanda diversa da «quali altri progetti sono interessanti?»: **abbiamo conservato tutte le idee emerse nel nostro percorso?** Le voci seguenti erano già presenti nelle discussioni precedenti o in `PROJECT_BRIEF.md`, ma non disponevano ancora di una scheda adeguata nelle versioni 1.0–1.2. Le schede F43–F55 possono contenere sia recuperi storici sia ampliamenti della ricerca: perciò la matrice sottostante specifica la provenienza. *Nessuna delle voci di questa sezione autorizza acquisti, implementazioni o modifiche all'architettura attuale.*

### F56 — Tenstorrent Blackhole p100a: acceleratore PCIe general-purpose BF16 [D][R]
- **Fonti:** [specifiche ufficiali Tenstorrent](https://docs.tenstorrent.com/aibs/blackhole/index.html), [repository documentazione](https://github.com/tenstorrent/tenstorrent.github.io/blob/main/core/aibs/blackhole/index.md).
- **Tecniche da studiare:** GDDR6 locale, nodi Tensix con calcolo BF16/FP16 nativo, SRAM on-chip, data movement programmabile, separazione fra memoria on-device e RAM/NVMe host. La famiglia distingue la scheda **p100a** dalle **p150a/p150b**, che hanno anche collegamenti inter-scheda dedicati.
- **Per InWiMe:** possibile acceleratore **futuro** quando una soluzione x86 con scheda add-in sia economicamente e tecnicamente appropriata; confrontare throughput BF16 effettivo, memoria disponibile, SDK, trasferimenti e costo energetico. Una scheda con 28 GiB GDDR6 non è per definizione un motore SSD-first né risolve modelli da centinaia di GiB senza streaming/partizionamento.
- **Limiti:** il produttore indica alimentazione circa 300 W e requisiti da workstation (fra cui 64 GB di RAM host nelle specifiche consultate); **non è una scheda da aggiungere al Katana**. Non confrontare i valori teorici BLOCKFP8 con la velocità BF16 o con token/s misurati.
- **Provenienza:** discusso nella ricognizione di schede add-in del settembre/novembre 2026, ma omesso come modello specifico nel catalogo precedente.

### F57 — Taalas HC1: pesi e calcolo specializzati nello stesso silicio [D][R]
- **Fonti:** [Taalas — prodotti e HC1](https://taalas.com/products/), [Taalas — architettura](https://taalas.com/).
- **Principio:** incorporare i pesi di uno specifico modello nel silicio anziché movimentarli ripetutamente fra memoria esterna e unità di calcolo. È un esempio estremo di riduzione del *memory wall* e del costo dei dati per token.
- **Per InWiMe:** confronto di **principi architetturali**, in particolare «portare il calcolo vicino ai pesi» e ridurre movimenti superflui; utile come controesempio per esplicitare il compromesso fra throughput specializzato e generalità.
- **Limiti:** HC1 è un dimostratore vincolato a un modello specifico (Llama 3.1 8B): non costituisce un backend programmabile capace di caricare arbitrariamente enormi checkpoint BF16 originali. Le prestazioni pubblicizzate sono dichiarazioni del produttore, non benchmark trasferibili a InWiMe.
- **Provenienza:** scheda/chip discusso in precedenza; assente da v1.2 come voce nominativa.

### F58 — Prototipo precedente `ooc-x86-poc`: tecniche e risultati da non perdere [D]
- **Fonte storica interna:** `PROJECT_BRIEF.md`, sezione su «earlier tiny ooc-x86-poc C++/Python prototype»; eventuale archivio originale `ooc-x86-poc.zip` **non incluso nel materiale verificato**. Non inventare un URL GitHub: il prototipo potrebbe essere solo un artefatto locale.
- **Tecniche già sperimentate:** lettura `safetensors` per regione tramite indice/offset, `pread`, GEMV BF16/FP16/FP32 bounded su CPU x86-64, slice di esperti tridimensionali, prova MoE top-k con SwiGLU, verifiche numeriche e sondaggio delle risorse host.
- **Per InWiMe:** base da confrontare con il runtime dei nuovi incrementi, se il codice originale viene recuperato. Evitare di ricreare inconsapevolmente funzionalità già sviluppate; non dichiararlo un runtime completo senza audit del codice.
- **Limiti della provenienza:** la descrizione compare nel brief, ma **non è stato possibile rieseguire né ispezionare il sorgente del vecchio prototipo** durante questo audit.

### F59 — SSD usato come «memoria estesa»: distinguere tre meccanismi [D]
- **Fonti tecniche:** [Linux `mmap` / man-pages](https://man7.org/linux/man-pages/man2/mmap.2.html), [Linux `pread` / man-pages](https://man7.org/linux/man-pages/man2/pread.2.html), [NVIDIA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
- **Origine del problema:** le conversazioni iniziali chiedevano se si potesse usare **SSD al posto della RAM o della VRAM in eccesso**, eventualmente scaricando quasi tutto il modello su memoria di massa, senza quantizzarlo. È una *domanda progettuale* distinta dal classico offload CPU↔GPU.
- **Meccanismo A — swap/pagefile OS:** il sistema migra pagine generiche sotto pressione; è un fallback possibile ma non uno scheduler dei pesi consapevole di layer, esperti e dipendenze. Latenza, page fault e imprevedibilità possono essere proibitivi.
- **Meccanismo B — file-backed `mmap` / page cache:** il kernel materializza pagine file su domanda, con caching OS e hint `madvise`; possibile per accessi sparsi, ma la memoria e il pattern delle page fault non sono completamente controllati dal runtime.
- **Meccanismo C — streaming applicativo esplicito:** catalogo dei tensori→offset nei file→I/O bounded→host staging→CPU o GPU→rilascio. Permette backpressure, selettività MoE, contabilità e sovrapposizione, al costo di maggiore complessità.
- **Per InWiMe:** tenere queste tre alternative separate nei benchmark e nel ragionamento architetturale. **Lo streaming applicativo non vieta di confrontare `mmap` o altre strategie in futuro**; evita di fondare la correttezza del progetto sul paging automatico del sistema operativo. La GPU accelera il modello vero, non un processo ausiliario.

### F60 — CPU-only mini-nodi, accensione a richiesta e scaling energetico [D]
- **Origine storica:** discussioni su Orange Pi/Banana Pi/Radxa, piccoli host x86 e nodi da accendere per accelerare temporaneamente una generazione, lasciando una capacità di servizio minima sempre disponibile. Il cluster era un *plus*, non un requisito dell'inferenza locale.
- **Tecniche:** località fisica dei pesi per nodo; accensione e sospensione; pre-riscaldamento dei layer o degli esperti richiesti; vincoli di cold-start, I/O e rete; throughput vs latenza singola sessione e joule/token.
- **Per InWiMe:** preservare in futuro pianificazione compatibile con risorse dinamiche senza imporre ora cluster, framework distribuiti o nodo ARM nel runtime CPU x86-64. Una rete lenta può cancellare il guadagno: repliche indipendenti devono rimanere la baseline per più sessioni.
- **Nota:** la famiglia di SBC e schede add-in è già dettagliata in F55; questa voce ricorda l'**obiettivo energetico e di elasticità**, non propone nuovi hardware.

### 12.1 Matrice delle **tecniche storiche** confrontate con l'indice attuale

Questa tabella non prova che ogni vecchia conversazione sia stata recuperata; prova solo che i principali argomenti rintracciati nel brief, handoff e contesto conversazionale disponibile hanno ora una collocazione esplicita.

| Gruppo storicamente discusso | Dove compare nel catalogo v1.3 | Esito |
|---|---|---|
| SSD come estensione di RAM/VRAM; paging vs I/O del runtime | T01–T05, F01, F06, **F59** | Distinzione esplicitata |
| Pesatura originale BF16/FP16 e kernel CPU streaming | F01–F04, F09, F51, **F58** | Coperto come principio + vecchio PoC |
| Layer/expert streaming AirLLM, ZeRO, DwarfStar | F02, F03, F13 | Coperto |
| llama.cpp, lazy reads, mmap, Direct I/O | F01, F06, F34–F36 | Coperto |
| Colibri e WARP, MoE/SSD, cache e più NVMe | F43, F44 | Recuperati in v1.2 |
| Routing MoE, expert paging, hot/cold, cache adattiva | F10–F12, F18, F32–F38, F43–F45 | Coperto |
| `safetensors`, GGUF, shard, accesso per slice | F09, F37, **F58** | Coperto; formati da verificare per modello |
| SpikingBrain 1.0/2.0, porting CPU, issue FLACache | F16, F40, F50 | Recuperati in v1.2 |
| Acceleratori Akida, Hailo, DEEPX, MemryX, Axelera | §6, F55 | Coperti con caveat |
| **Taalas HC1; Tenstorrent Blackhole p100a** | **F56, F57** | Recuperati in v1.3 |
| Orange Pi, Banana Pi, Radxa e nodi on-demand | §6, F55, **F60** | Coperto |
| NVMe multipli, GDS, `io_uring`, O_DIRECT, overlap CUDA | F06–F08, F47, F53 | Coperto |
| Benchmark BF16 CPU/GPU, buffer, CPU-only, GPU facoltativa | §§0–1, §7–8, F39, F51 | Coperto; benchmark **non** token/s |
| KV, stato ricorrente, decode/prefill, gestione sessioni | F19–F25, F40, F49, F52 | Coperto |
| Speculative decoding, n-gram, PLE lazy, MTP | F17, F37 | Coperto |
| Distribuzione PP/TP/EP, compute-near-weights | F26–F31, F53 | Coperto |
| Repliche indipendenti vs inferenza distribuita | T14–T17, §5, H10, F53 | Baseline distinta da token/s sessione |
| DeepSeek Harness, API OpenAI-compatible | F41 | Coperto; client, non motore inferenza |
| Modelli target giganteschi e native FP8/MXFP4 | §0, F09; `PROJECT_BRIEF.md` | Specifiche del modello ancora da verificare |

### 12.2 Nuove ipotesi annotate, non nuove decisioni

- **H34:** usare Unified Memory in Linux nativo ove supportata (HMM/ATS) contro staging esplicito CPU/GPU, misurando page fault, banda, latenza/token e memoria allocata; WSL/Windows sono un caso distinto.
- **H35:** processore specializzato con pesi hardwired (Taalas) vs memoria programmabile (Tenstorrent/CPU/GPU): confrontare il compromesso *generalità/costo di movimentazione dei pesi*, senza proporre HC1 come acceleratore universale.
- **H36:** recuperare e ispezionare `ooc-x86-poc` **soltanto se l'archivio storico è disponibile**, prima di rifare componenti equivalenti.
- **H37:** cluster con nodi attivi su domanda contro host fissi: risparmio netto solo se il costo di avvio/preload e le comunicazioni non superano il beneficio energetico e prestazionale.

### 12.3 Limite esplicito dell'audit e regola per le future aggiunte

**Questo è un audit del materiale disponibile, non una verifica integrale di ogni vecchio messaggio.** Sono stati confrontati i documenti `PROJECT_BRIEF.md`, `HANDOFF-2026-11-05-benchmark-v2.md`, le architetture, i messaggi presenti nella conversazione e i riferimenti storici recuperabili. Non è giustificato affermare «abbiamo incluso tutto» senza un accesso verificabile a tutte le conversazioni originali o un elenco chiuso approvato dall'utente.

Per le prossime revisioni tenere **due registri separati**: (A) *riferimenti effettivamente discussi*, con data/thread o documento di origine quando noto; (B) *nuove scoperte dalla ricerca*, con data e fonte. Una nuova proposta interessante non deve mai essere etichettata retroattivamente come argomento discusso.
