# InWiMe — Visione, requisiti e roadmap

**Intelligence Without Memory Limits** · Documento canonico di orientamento  
**Stato:** roadmap di ricerca e sviluppo e piano delle release, riconciliata il 20 novembre 2026 dopo la chiusura di M10.  
**Baseline verificata:** `main` a `d3b73c7` (merge della PR #24, incremento 10 e gate 10 A–H PASS). La versione precedente di questa pagina (15 novembre, baseline `68ba9f3`) resta nella storia Git della PR #1.

> **Scopo ultimo.** Eseguire localmente, in modo progressivo, il maggior numero possibile di modelli AI open-weight — LLM densi e Mixture-of-Experts (MoE) estremamente grandi, poi architetture ibride e multimodali — su hardware eterogeneo e il più possibile limitato: PC x86-64 ordinari con **16 GiB di RAM preferiti (32 GiB come tetto di riferimento)**, con pesi originali residenti prevalentemente su SSD/NVMe e caricati/elaborati progressivamente in buffer di memoria limitati. L'intero modello non deve dover entrare in RAM o VRAM. CPU-only deve funzionare autonomamente; GPU e distribuzione sono accelerazioni facoltative. **Universalità prima delle prestazioni:** correttezza e fedeltà al checkpoint precedono la velocità; una latenza alta è accettabile, purché misurata.

“Tutti i modelli” è un obiettivo aperto e asintotico: nessun sistema garantisce il supporto automatico a operatori futuri sconosciuti, né a hardware privo della memoria di stato, dello storage o del calcolo minimi. La copertura si dichiara solo dove è stata verificata ([#29](https://github.com/francescodifilippo/inwime/issues/29)).

Questa è una visione progettuale, **non** una dichiarazione che modelli da centinaia di miliardi o trilioni di parametri siano già eseguibili o utilizzabili a velocità interattiva. “Without Memory Limits” indica indipendenza della *residenza dei pesi* dalla loro dimensione complessiva, **non** memoria letteralmente illimitata: KV cache, contesto, workspace, stato, metadata e throughput rimangono vincoli fisici.

## 1. Contratto di prodotto e priorità

1. **Hardware accessibile:** desktop/laptop x86-64, CPU ordinaria, RAM commodity. Baseline sperimentale MSI Katana GF66, i7-12700H, 16 GiB fisici; RAM disponibile in WSL osservata ~11,48 GB. Nessun Mac o hardware AI specialistico obbligatorio.
2. **Pesi originali preservati:** BF16/FP16 sono i primi formati; supportare altri dtype *nativi* dei checkpoint (p.es. FP8/MXFP4) richiederà lettori e kernel espliciti. **Nessuna quantizzazione aggiuntiva obbligatoria**, né conversione persistente nascosta. Quantizzazione opzionale soltanto come percorso distinto, misurato e dichiarato.
3. **Out-of-core reale:** letture a intervalli/tile su checkpoint immutabili anche multi-shard, buffer e cache con budget espliciti, senza materializzare l'intero modello. Evitare di usare lo swap del sistema come sostituto incontrollato del runtime.
4. **Fedeltà al modello:** tokenizer, layer, attenzione, RoPE, bias, routing MoE, tie/untie dei pesi, KV o stato ricorrente devono rispecchiare esattamente la famiglia selezionata. Nessun layer o esperto omesso per ottenere un PASS.
5. **Correttezza verificabile:** oracle indipendente e confronto di logits, intermedi, KV e token; contratti numerici separati per backend/precisione. La baseline attuale usa FP64 ordinato; **FP64 non è un requisito universale per acceleratori futuri**.
6. **CPU-first, GPU opzionale ma utile al modello reale:** acceleratori con memoria piccola (es. RTX 3050 4 GiB) devono poter operare su tile, senza richiedere che interi layer o tutto il modello entrino in VRAM. In WSL2 verificare il caricamento delle librerie NVIDIA effettive, anche da `/usr/lib/wsl/lib` quando necessario.
7. **Misure oneste:** distinguere budget del runtime, RLIMIT_AS, RSS, memoria cgroup, page cache Linux, cache host Windows, swap, letture logiche, I/O guest e letture fisiche del dispositivo. Tenere separate cache-warm e cold controllato.
8. **Estensibilità senza complessità anticipata:** storage multipli, più PC/HPC e acceleratori economici/neuromorfici rimangono **opzioni**, subordinate a evidenza di fattibilità. Non sono requisiti per il primo motore. Prestazioni, GPU, prefetch, spiking e distribuzione si anticipano solo quando sono indispensabili per **rendere eseguibile** un checkpoint sull'hardware target.
9. **Backend indipendente dal client:** API compatibile OpenAI dove appropriato e integrazione con **DeepSeek Harness** come client, a partire dalla v0.1.0-alpha ([#26](https://github.com/francescodifilippo/inwime/issues/26)); non ricreare un orchestratore agentico.
10. **Scelta aperta dei modelli:** catalogo Hugging Face navigabile, repo/revisione e checkpoint locali scelti liberamente, download volontario e verificato. Poter *scaricare* un modello non significa poterlo *eseguire*: un checkpoint non ancora supportato riceve una diagnosi precisa (operatore, dtype, architettura mancanti), mai output inventati, conversioni nascoste o quantizzazione silenziosa. Nessuna esecuzione automatica di codice remoto (`trust_remote_code` disabilitato per default).
11. **Riuso prima della reinvenzione:** importatori, librerie e operatori open source già esistenti (Transformers, PyTorch FX/`torch.export`, ONNX, componenti di llama.cpp, Colibri, WARP, AirLLM e altri, con licenze compatibili) vanno integrati quando tecnicamente appropriati ([#28](https://github.com/francescodifilippo/inwime/issues/28)). Il lavoro distintivo di InWiMe è l'esecuzione **out-of-core bounded e numericamente verificata**.

**Gerarchia dei trade-off:** correttezza e preservazione dei pesi → eseguibilità entro budget → robustezza/riproducibilità → velocità, consumi e comodità operativa. Una tecnica che cambia la rete neurale o gli output è un esperimento diverso, non un'ottimizzazione trasparente.

## 2. Architettura di riferimento

La scelta iniziale documentata in [INWIME_ARCHITECTURE_DECISION.md](INWIME_ARCHITECTURE_DECISION.md) è un **runtime compatto con un solo coordinatore autorevole sullo stato**, operazioni I/O concorrenti ma bounded, adapter sostituibili per modello, sorgente e calcolo. I nomi seguenti indicano responsabilità **concettuali, non processi o servizi obbligatori**:

| Responsabilità | Nome | Compito |
| --- | --- | --- |
| Accesso ai dati | **Briareus** | Risoluzione tensori/shard, letture NVMe→RAM, backend I/O, futuri trasferimenti RAM→VRAM, identità e integrità |
| Budget e residenze | **Mnemosyne** | Limiti RAM/VRAM, staging, lease, cache/eviction, ammissione e backpressure |
| Esecuzione | **Ariadne** | Ordine delle operazioni, dipendenze, prefetch, scheduling del routing MoE e dei dispositivi |
| Osservabilità | **Iris** | Byte logici/fisici, tempi, overlap, memoria, errori; telemetria bounded |
| Semantica del modello | **Adapter** | Tokenizer, forward, attenzione, routing, KV/stato, tensor mapping e contratti numerici |
| Sessione e controllo | **InWiMe** | Unica autorità per avanzamento e pubblicazione atomica del KV, cancellazione, chiusura e API future |

La struttura effettiva del codice può accorpare queste responsabilità. Evitare un'estrazione prematura in microservizi o un nuovo framework generale. La separazione fra **formato fisico dei pesi** e **semantica del decoder** è un contratto centrale.

### Decisione del 20 novembre 2026: un solo motore modulare ([#27](https://github.com/francescodifilippo/inwime/issues/27))

InWiMe resta **un solo motore d'inferenza con un solo coordinatore** autorevole per sessioni, stato KV, risorse e scheduling. La copertura cresce tramite **componenti modulari dello stesso motore**, non tramite una piattaforma che orchestra motori esterni indipendenti. Un esecutore esterno intero è ammesso solo come eccezione motivata e testata. Questa decisione supera le proposte multi-backend dei commenti precedenti di #10 e #26, senza riscrivere lo storico.

| Modulo logico | Responsabilità |
| --- | --- |
| Model discovery / metadata | Catalogo Hugging Face e checkpoint locali; `config.json`, tokenizer e template, `model_type`/`architectures`, indici safetensors, dtype e layout; repo e revisione esatti |
| Graph / model resolver | Adapter per famiglie note e grafi importati da strumenti esistenti (#28); operatori, ordine, forme, control flow, stato e KV espliciti |
| Operator registry | Operatori primitivi e specializzati (RoPE, GQA, attenzione, router MoE, KDA/DSA, Engram e altri) con semantica, shape, precisione, contratto di memoria e versione |
| Weight providers | Lettori safetensors (poi GGUF se utile) e dtype **nativi** BF16/FP16/FP8/MXFP4 con implementazioni esplicite; offset, shard e integrità (Briareus) |
| Device compute | Kernel CPU di base; GPU e acceleratori opzionali, con fallback CPU solo se implementato e verificato |
| Resource manager + scheduler | Unica autorità per residenza, cache, staging, lease, backpressure, limiti RAM/VRAM e prefetch (Mnemosyne, Ariadne) |
| State / session + observability | KV e stato ricorrente, isolamento sessioni, cancellazione atomica, telemetria RSS/cgroup/I/O (InWiMe, Iris) |
| Serving | API OpenAI-compatible, streaming SSE, DeepSeek Harness, comandi per scegliere, installare e usare i modelli |

Sono interfacce e responsabilità, non processi distinti. Il riconoscimento di un checkpoint attraversa **fasi distinte e osservabili**: `discovered` → `metadata_parsed` → `graph_resolved` → `operators_resolved` → `resource_plan_feasible` → `end_to_end_executable` → `numerically_verified`. Configurazione e header dei tensori non provano il grafo; un piano si compone solo se ogni operatore ha implementazione e budget di memoria verificabili.

## 3. Stato comprovato — non confondere le evidenze

| Fase | Stato al 20/11/2026 | Prova e limiti |
| --- | --- | --- |
| Benchmark iniziali / V2 / storage | **Completati nel loro perimetro** | BF16 tiled CPU/GPU e confronto di percorsi di storage; non erano inferenza LLM end-to-end |
| Incrementi 1–3 | **Completati** | Decoder sintetico denso/MoE, forward/KV, pipeline I/O limitata e test indipendenti |
| Incremento 4 | **Completato** | Manifest segmentato e letture BF16 oltre la RAM WSL; non ancora forward completo |
| Incremento 5 | **Completato** | Forward completo del decoder **sintetico** con pesi attivi superiori alla RAM WSL; non supera la RAM fisica host |
| Incremento 6 | **Completato, revisionato** | Sorgente safetensors BF16 originale multi-shard, bounded, preservando il decoder |
| Incremento 7 | **Completato** | SmolLM2-135M **addestrato**, tokenizer, RoPE/GQA e generazione; oracle FP32 stock e FP64 effettivo PASS. Il FAIL storico di Transformers standard double/mixed precision resta diagnostico |
| Incremento 8 | **Completato, revisionato e pubblicato** | SmolLM2-1.7B addestrato: **3.422.752.768 byte BF16 unici attivi**, forward sotto **RLIMIT_AS 128 MiB**, ~92 MB RSS; parità completa Transformers FP32, KV e multi-shard; 408 test generali + 6 oracle nella revisione |
| Incremento 9 (M9-A, M9-B) | **Completato con limitazioni documentate** | SmolLM2-1.7B corretto in un cgroup v2 da 2 GiB che include la page cache. Qwen2.5-7B BF16 originale: 14.141.295.616 B di pesi unici attivi contro `MemTotal` WSL di 11.483.848.704 B, prefill di 5 token e 3 decode nello stesso limite di 2 GiB senza swap; parità completa con oracolo Transformers indipendente (FP32 standard e FP64 puro PASS, FP64 standard diagnostico FAIL grezzo conservato). Non supera la RAM fisica host. Gate A–H PASS dopo remediation e hardening (PR #20) |
| Incremento 10 (M10) | **Completato, gate 10 A–H PASS** | Qwen2.5-14B BF16 originale: 27.983.013.888 B di pesi unici attivi, 1,656× la RAM fisica host. Prefill di 5 token e 3 decode in uno scope da 2 GiB senza swap né OOM, picco RSS 162 MB. Accordo elemento per elemento con oracolo indipendente bounded su logits, residui e K/V completi (`stock_fp32` e `full64` PASS a copertura piena; `stock_fp64` diagnostico FAIL grezzo). Inventario dei test pinnato e mutazioni del gate senza falsi PASS. PR #24, merge `d3b73c7` |

**Limiti tuttora aperti:** il cold misurato in M10 è cold *guest* (NAND, cache Windows e controller non controllati; temperatura e potenza non misurate); il contesto massimo provato è di 8 posizioni; il decode del 14B misura ~23 s/token (0,043 tok/s, calcolo FP64 a singolo thread, prefill non batchato); le famiglie reali eseguite sono solo dense BF16 (Llama/SmolLM2 e Qwen2); mancano un percorso MoE reale, un'API d'inferenza, un'installazione per utenti finali e un backend GPU integrato nel decoder. Non estendere a nuovi modelli la parità numerica dimostrata soltanto per quelli già testati.

**Fonti verificabili:** [incremento 5](RUNTIME_CPU_INCREMENT_5.md), [6](RUNTIME_CPU_INCREMENT_6.md), [7](RUNTIME_CPU_INCREMENT_7.md), [8](RUNTIME_CPU_INCREMENT_8.md), [review 8](REVIEW_INCREMENT_8.md), [9](RUNTIME_CPU_INCREMENT_9.md), [review 9](REVIEW_INCREMENT_9.md), [10](RUNTIME_CPU_INCREMENT_10.md), [evidenze M10](../results/increment10/campaign/README.md).

## 4. Roadmap verificabile

Queste sono **fasi logiche**, non date promesse né autorizzazioni implicite a scaricare modelli, cambiare il sistema o acquistare hardware. Le attività future possono procedere in parte in parallelo, ma ogni dichiarazione di completamento richiede il suo gate probatorio.

Dal 20 novembre 2026 la roadmap ha due piani collegati:
- il **piano delle release pubblicabili** (4.1, issue [#29](https://github.com/francescodifilippo/inwime/issues/29)), che alterna ricerca e sviluppo, test numerici e di memoria, e rilascio;
- la **roadmap di ricerca** M11–M16 (4.3), che fornisce capacità alle release e non è più una sequenza da completare prima di rilasciare.

### 4.1 Piano delle release (universality-first)

Ogni release deve essere installabile e interrogabile e aggiungere copertura *verificata* rispetto alla precedente. Versioni e ordine indicano obiettivi, non garanzie: le dipendenze tecniche si risolvono nell'ordine opportuno e sono ammesse patch o minor alpha intermedie.

| Release | Incremento di copertura / funzionalità | Evidenza di uscita |
| --- | --- | --- |
| **v0.1.0-alpha** (M10.5, [#26](https://github.com/francescodifilippo/inwime/issues/26)) | Alpha CPU-first installabile su x86-64 Linux/WSL2 dove testato; catalogo HF navigabile e scelta libera; download volontario con checksum e resume; import di checkpoint locali; API OpenAI-compatible e DeepSeek Harness; almeno un modello conversazionale reale verificato | Richiesta harness → InWiMe → modello → risposta, memoria bounded, errori e cancellazione, nessuna quantizzazione silenziosa; report hardware |
| **v0.2.0-alpha** ([#28](https://github.com/francescodifilippo/inwime/issues/28)) | Integrazione di importatori **esistenti** (Transformers `AutoConfig`/`AutoModel`, FX/`torch.export`, ONNX dove disponibile); risoluzione di grafo e operatori; adapter solo per gap reali | Almeno un modello denso importato → operatori InWiMe → forward bounded con parità; distinzione detected/imported/executable/verified |
| **v0.3.0-alpha** | Più lettori e dtype originali (BF16/FP16/FP32, FP8/MXFP4 nativi, GGUF se utile) e operatori di base comuni | Fixture e almeno un checkpoint reale per ogni nuovo formato; valori originali preservati; nessuna conversione nascosta |
| **v0.4.0-alpha** (M11) | MoE reale multi-famiglia: routing top-k, normalizzazione, layout degli esperti, caricamento selettivo bounded | Parità di router, logits, stato e token con oracolo indipendente su almeno una famiglia MoE; byte e risorse misurati |
| **v0.5.0-alpha** | Attenzione e stato non standard: MLA/KDA/DSA, Gated DeltaNet, Engram, state-space/ricorrenti/ibridi, KV custom e crescita del contesto | Una o più famiglie ibride reali validate; correttezza dei sottosistemi stateful entro i limiti di risorse |
| **v0.6.0-beta** | Multimodalità progressiva (encoder immagine/audio, cross-attention, preprocessing e output) quando il modello lo richiede | E2E reale per ogni modalità dichiarata, inclusi preprocessing e corrispondenza semantica |
| **v0.7.0-beta** | Long tail: grafi meno comuni, operatori custom, control flow dinamico, fallback CPU bounded, interfacce plugin per operatori; sicurezza del codice modello | Famiglie prima non supportate ora eseguibili; ogni caso non supportato produce diagnostica di operatore, shape e requisiti |
| **v0.8.0-beta** | Portabilità: altre CPU, OS e hardware dove possibile; storage locale o remoto facoltativo; molti shard e FD; contesti lunghi | Matrice reale su più sistemi e modelli senza superare i budget; cancellazione sicura; catalogo robusto |
| **v0.9.0-rc** | Coverage campaign su snapshot versionato del catalogo HF; risoluzione dei gap non dichiarati; packaging e documentazione | Test indipendenti su holdout (modelli non usati nello sviluppo); metriche coperto/non coperto con spiegazione dei gap |
| **v1.0.0** | Release stabile: installazione, API e catalogo; correttezza e memoria verificabili; **copertura dichiarata ed estensibile** | Gate stabili; matrice pubblica modello × architettura × dtype × operatori × hardware; regressioni CI; versioning e limiti documentati |

**Misura del progresso.** Una matrice pubblica di compatibilità registra per ogni modello identità (repo, revisione SHA, licenza, architettura, dtype, formato), la fase raggiunta fra quelle della sezione 2, l'hardware provato e l'esito dei test. Le percentuali di copertura si riportano sempre con denominatore e metodo di campionamento; il numero di modelli *riconosciuti* non si usa per vantare modelli *eseguibili*.

**Regole di release.** Prima di ogni release scegliere una classe architetturale o un problema di supporto non ancora coperto; integrare codice esistente dove possibile; aggiungere solo l'esecuzione out-of-core mancante; verificare e rilasciare. Una release pubblicata richiede codice installabile, documentazione, test di regressione, report di memoria e correttezza numerica, e una dimostrazione E2E verificabile. La challenge Kimi-K3.1-Pro (2,8T, MoE, MXFP4 nativo, ~1,56 TB) resta un test sperimentale opzionale e separato, non un gate della prima alpha.

### 4.2 Traguardi di ricerca completati

**M9-A** (budget che include la page cache), **M9-B** (primo modello pubblico oltre `MemTotal` WSL) e **M10** (oltre la RAM fisica host, con I/O cold guest attribuito) sono chiusi; esiti e limiti nella sezione 3 e nei rapporti degli incrementi 9 e 10. Le loro definizioni originali restano nella storia Git di questo file.

### 4.3 Roadmap di ricerca M11–M16

Questi filoni restano tracciati nelle rispettive issue e alimentano le release della sezione 4.1. Si anticipano solo quando sono indispensabili per rendere eseguibile un checkpoint sull'hardware target.

#### Capacità necessarie per modelli enormi

**M11 — Decoder MoE pubblico e semantica esatta.** Adapter per una famiglia MoE realmente pubblicata con routing top-k, normalizzazione, expert layout, attenzione e eventuale stato ibrido corretti; streaming **solo degli esperti effettivamente selezionati** secondo la semantica del modello, senza cambiarne il routing. Parità su router, esperti, logits, KV e generazione rispetto a un riferimento indipendente. Confluisce nella v0.4.0-alpha.

**M12 — Efficienza end-to-end e scheduling bounded.** Profilare letture NVMe, BF16 expansion, GEMV, tokenizer, KV, prefetch, cache degli esperti e pipeline. Scegliere ottimizzazioni una per volta con confronto A/B e compatibilità numerica. Provare scalabilità di metadata/FD oltre i limiti sperimentali attuali (64 shard) **solo quando necessaria**, con FD a capacità controllata.

**M13 — Accelerazione opzionale CPU/GPU.** Backend FP32/BF16/accumuli appropriati con contratti di accuratezza espliciti, inclusi tile su GPU con 4 GiB VRAM e fall-back CPU. Misurare H2D/D2H, kernel, overlap, RSS/VRAM e consumo energetico; niente obbligo di FP64 ordinato su hardware inadatto, ma baseline FP64 invariata per la diagnosi.

**M14 — Fruibilità e integrazione.** Sessioni robuste, streaming token, endpoint HTTP compatibili OpenAI ove possibile, tokenizer/chat template e tool call fedeli, integrazione **DeepSeek Harness** come client. Test su errori, cancellazione, interruzioni e isolamento delle sessioni. La v0.1.0-alpha ne anticipa il nucleo minimo senza chiudere la issue #8.

#### Traguardo di lungo periodo — grande scala

**M15 — Scalare senza cambiare il principio.** Supporto a checkpoint molto superiori a RAM/VRAM (decine, centinaia di miliardi e potenzialmente modelli con centinaia/migliaia di miliardi di parametri), inclusi formati originali nativi e shard numerosi. Prevedere **capacità SSD reale adeguata**: un checkpoint multi-TB non entra sul Sabrent da 512 GB. Calcolare throughput e tempi prima di dichiarare l'uso pratico; misurare la frazione attiva per token nei MoE, non confonderla con i parametri totali.

**M16 — Opzionali, solo se utili.** Multi-NVMe, nodo remoto/cluster Beowulf o mini-PC, acceleratori economici e neuromorfici, quantizzazione opzionale, decoding speculativo e tecniche ibride. Le alternative si valutano separatamente per benefici misurabili, costi, fedeltà numerica e compatibilità; **distribuzione e GPU non diventano prerequisiti**.

## 5. Principali rischi e criteri di scelta

- **Capacità di storage:** verificare disco fisico, spazio host Windows sottostante al VHDX e duplicazioni di download; non usare la capacità apparente di un filesystem virtuale come preventivo.
- **Throughput:** per un modello denso la lettura di tutti i pesi attivi a ogni token impone un limite di velocità; in generale `token/s <= throughput_sostenuto / byte_effettivamente_riletti_per_token` quando il carico è I/O-bound. Cache, riuso, routing MoE e sovrapposizione modificano il volume effettivo, ma non autorizzano a inventare prestazioni.
- **KV e contesti lunghi:** anche se i pesi sono out-of-core, KV e workspace crescono con architettura, batch e lunghezza della sequenza. Occorre ammissione/eviction/paging controllato e test specifici.
- **Compatibilità dei dtype:** BF16 originale e precisione di computazione sono concetti distinti; FP8 originale non va etichettato BF16. La modifica silenziosa della matematica invalida la parità.
- **WSL2, SSD esterno, NVIDIA:** il percorso USB/DrvFS storico è lento; non presume l'efficienza di ext4 nativo o VHDX esterno. Per eventuali prove GPU scegliere esplicitamente le librerie del driver WSL quando richiesto.
- **Qualità e prestazioni:** velocità inizialmente secondaria, ma dichiarare *praticabile* un grande modello richiederà misure di latenza, consumi e storage su workload reali.
- **Sicurezza operativa:** nessuna formattazione, modifica `.wslconfig`, swap globale, antivirus/driver o intervento distruttivo senza specifica autorizzazione. Download multi-GB solo dopo audit di capacità/provenienza e autorizzazione appropriata.

## 6. Come mantenere la roadmap aggiornata

- **Questo file** contiene visione, decisioni di prodotto, roadmap e distinzione tra dimostrato/previsto. È il punto di ingresso per chi sviluppa il motore.
- **`PROJECT_BRIEF.md`** conserva requisiti e contesto hardware del novembre 2026, ma le sue istruzioni temporanee e gli stati dei primi benchmark sono **storici**, non lo stato corrente.
- **`AGENTS.md`** contiene gli obblighi di sicurezza e sviluppo: leggerlo prima di intervenire sul codice.
- **`docs/INWIME_ARCHITECTURE_DECISION.md`** contiene le alternative e la scelta iniziale del coordinatore, senza trasformare nomi concettuali in microservizi.
- **`docs/INWIME_TECHNIQUE_WATCHLIST.md`** conserva le 60 schede e le ipotesi tecniche: **inventario, non backlog**, e non va riassorbito o riscritto per ogni incremento.
- **`docs/RUNTIME_CPU_INCREMENT_N.md` + `results/incrementN/`** sono le fonti di verità per verifiche, risultati e limiti di ciascun incremento.
- **GitHub Issues** rappresentano i filoni *aperti* e verificabili, con dipendenze e accettazione. Non aprire retroattivamente issue per incrementi già completati.
  - [Issue principale #10](https://github.com/francescodifilippo/inwime/issues/10): visione e storico degli incrementi, con collegamenti a [M9-A #2](https://github.com/francescodifilippo/inwime/issues/2), [M9-B #3](https://github.com/francescodifilippo/inwime/issues/3), [M10 #4](https://github.com/francescodifilippo/inwime/issues/4) (chiuse), [M11 #5](https://github.com/francescodifilippo/inwime/issues/5), [M12 #6](https://github.com/francescodifilippo/inwime/issues/6), [M13 #7](https://github.com/francescodifilippo/inwime/issues/7), [M14 #8](https://github.com/francescodifilippo/inwime/issues/8) e [M15+ #9](https://github.com/francescodifilippo/inwime/issues/9).
  - [Piano delle release #29](https://github.com/francescodifilippo/inwime/issues/29), con [v0.1.0-alpha #26](https://github.com/francescodifilippo/inwime/issues/26), [motore unico modulare #27](https://github.com/francescodifilippo/inwime/issues/27) e [integrazione degli importatori #28](https://github.com/francescodifilippo/inwime/issues/28).
- **GitHub Projects**, se configurato dal proprietario, è una vista organizzativa delle stesse Issues: **non una seconda roadmap concorrente**. Stati consigliati: Backlog, Ready, In progress, Blocked, Review, Done; campi: Area, Priorità, Milestone. La mancanza di un Project non impedisce di mantenere le issue.

**Regola di aggiornamento:** quando un milestone è davvero chiuso, aggiornare l'esito qui, collegare commit/report/gate e chiudere la relativa issue. Non trasformare in PASS un test omesso, un benchmark cache-warm o una capacità soltanto progettata.
