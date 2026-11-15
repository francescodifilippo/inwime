# InWiMe — Visione, requisiti e roadmap

**Intelligence Without Memory Limits** · Documento canonico di orientamento  
**Stato:** roadmap di ricerca e sviluppo, aggiornata al 15 novembre 2026.  
**Baseline verificata:** `main` a `68ba9f3adf3cffb89b18b19525644fa0d7c6d806`, incremento 8 e sua revisione. L'incremento 9 è pianificato; questa pagina non ne certifica l'esecuzione.

> **Scopo ultimo.** Realizzare un motore **locale** d'inferenza per LLM estremamente grandi, inclusi Mixture-of-Experts (MoE), eseguibile su PC x86-64 ordinari con **16 GiB di RAM preferiti (32 GiB come tetto di riferimento)**, con pesi originali residenti prevalentemente su SSD/NVMe e caricati/elaborati progressivamente in buffer di memoria limitati. L'intero modello non deve dover entrare in RAM o VRAM. CPU-only deve funzionare autonomamente; GPU e distribuzione sono accelerazioni facoltative. Correttezza dei risultati e fedeltà al checkpoint precedono la velocità; l'obiettivo successivo è renderne la latenza praticamente utile.

Questa è una visione progettuale, **non** una dichiarazione che modelli da centinaia di miliardi o trilioni di parametri siano già eseguibili o utilizzabili a velocità interattiva. “Without Memory Limits” indica indipendenza della *residenza dei pesi* dalla loro dimensione complessiva, **non** memoria letteralmente illimitata: KV cache, contesto, workspace, stato, metadata e throughput rimangono vincoli fisici.

## 1. Contratto di prodotto e priorità

1. **Hardware accessibile:** desktop/laptop x86-64, CPU ordinaria, RAM commodity. Baseline sperimentale MSI Katana GF66, i7-12700H, 16 GiB fisici; RAM disponibile in WSL osservata ~11,48 GB. Nessun Mac o hardware AI specialistico obbligatorio.
2. **Pesi originali preservati:** BF16/FP16 sono i primi formati; supportare altri dtype *nativi* dei checkpoint (p.es. FP8/MXFP4) richiederà lettori e kernel espliciti. **Nessuna quantizzazione aggiuntiva obbligatoria**, né conversione persistente nascosta. Quantizzazione opzionale soltanto come percorso distinto, misurato e dichiarato.
3. **Out-of-core reale:** letture a intervalli/tile su checkpoint immutabili anche multi-shard, buffer e cache con budget espliciti, senza materializzare l'intero modello. Evitare di usare lo swap del sistema come sostituto incontrollato del runtime.
4. **Fedeltà al modello:** tokenizer, layer, attenzione, RoPE, bias, routing MoE, tie/untie dei pesi, KV o stato ricorrente devono rispecchiare esattamente la famiglia selezionata. Nessun layer o esperto omesso per ottenere un PASS.
5. **Correttezza verificabile:** oracle indipendente e confronto di logits, intermedi, KV e token; contratti numerici separati per backend/precisione. La baseline attuale usa FP64 ordinato; **FP64 non è un requisito universale per acceleratori futuri**.
6. **CPU-first, GPU opzionale ma utile al modello reale:** acceleratori con memoria piccola (es. RTX 3050 4 GiB) devono poter operare su tile, senza richiedere che interi layer o tutto il modello entrino in VRAM. In WSL2 verificare il caricamento delle librerie NVIDIA effettive, anche da `/usr/lib/wsl/lib` quando necessario.
7. **Misure oneste:** distinguere budget del runtime, RLIMIT_AS, RSS, memoria cgroup, page cache Linux, cache host Windows, swap, letture logiche, I/O guest e letture fisiche del dispositivo. Tenere separate cache-warm e cold controllato.
8. **Estensibilità senza complessità anticipata:** storage multipli, più PC/HPC e acceleratori economici/neuromorfici rimangono **opzioni**, subordinate a evidenza di fattibilità. Non sono requisiti per il primo motore.
9. **Backend indipendente dal client:** in una fase successiva esporre un'API compatibile OpenAI dove appropriato e integrarsi con **DeepSeek Harness**; non ricreare un orchestratore agentico.

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

## 3. Stato comprovato — non confondere le evidenze

| Fase | Stato al 15/11/2026 | Prova e limiti |
| --- | --- | --- |
| Benchmark iniziali / V2 / storage | **Completati nel loro perimetro** | BF16 tiled CPU/GPU e confronto di percorsi di storage; non erano inferenza LLM end-to-end |
| Incrementi 1–3 | **Completati** | Decoder sintetico denso/MoE, forward/KV, pipeline I/O limitata e test indipendenti |
| Incremento 4 | **Completato** | Manifest segmentato e letture BF16 oltre la RAM WSL; non ancora forward completo |
| Incremento 5 | **Completato** | Forward completo del decoder **sintetico** con pesi attivi superiori alla RAM WSL; non supera la RAM fisica host |
| Incremento 6 | **Completato, revisionato** | Sorgente safetensors BF16 originale multi-shard, bounded, preservando il decoder |
| Incremento 7 | **Completato** | SmolLM2-135M **addestrato**, tokenizer, RoPE/GQA e generazione; oracle FP32 stock e FP64 effettivo PASS. Il FAIL storico di Transformers standard double/mixed precision resta diagnostico |
| Incremento 8 | **Completato, revisionato e pubblicato** | SmolLM2-1.7B addestrato: **3.422.752.768 byte BF16 unici attivi**, forward sotto **RLIMIT_AS 128 MiB**, ~92 MB RSS; parità completa Transformers FP32, KV e multi-shard; 408 test generali + 6 oracle nella revisione |
| Incremento 9 | **Da dimostrare / in pianificazione** | Prova con cache addebitata a cgroup ristretto; poi modello reale con pesi attivi maggiori di `MemTotal` WSL e forward verificato |

**Limiti tuttora aperti:** non è ancora pubblicata una prova con **modello reale addestrato** più grande della RAM totale WSL o della RAM fisica dell'host; non è dimostrato NVMe cold end-to-end; i contesti testati sono brevi; manca un percorso MoE reale, una API d'inferenza e un backend GPU integrato per il decoder. Non estendere a nuovi modelli la parità numerica dimostrata soltanto per quelli già testati.

**Fonti verificabili:** [incremento 5](RUNTIME_CPU_INCREMENT_5.md), [6](RUNTIME_CPU_INCREMENT_6.md), [7](RUNTIME_CPU_INCREMENT_7.md), [8](RUNTIME_CPU_INCREMENT_8.md), [review 8](REVIEW_INCREMENT_8.md), [evidenze](../results/increment8/review/README.md).

## 4. Roadmap verificabile

Queste sono **fasi logiche**, non date promesse né autorizzazioni implicite a scaricare modelli, cambiare il sistema o acquistare hardware. Le attività future possono procedere in parte in parallelo, ma ogni dichiarazione di completamento richiede il suo gate probatorio.

### Traguardo prossimo — memoria e I/O reali

**M9-A — Budget che include la cache.** Con SmolLM2-1.7B: processo completo entro un cgroup v2 isolato da ~2 GiB, se delega e permessi lo consentono; contabilizzare file cache, RSS, swap, eventi e I/O. Il PASS richiede forward corretto, budget realmente applicato, nessun OOM nascosto e attribuzione delle letture corretta. Se la delega manca, stato **BLOCKED/NOT PROVEN**, non falso PASS.

**M9-B — Primo modello pubblico oltre `MemTotal` WSL.** Candidato da verificare: Qwen2.5-7B BF16 (~15 GB), con adapter Qwen2 esatto e preflight storage; il criterio riguarda i **pesi unici effettivamente usati**, non la dimensione del download. Prefill e decode reali, stato KV coerente, budget espliciti e livello di parità numerica indipendente dichiarato. La prova supera RAM WSL (~11,48 GB), **non** necessariamente RAM fisica host (~16,9 GB).

**M10 — Oltre la RAM fisica e I/O cold credibile.** Checkpoint pubblico addestrato con pesi attivi maggiori della RAM fisica host, senza cambiare la RAM. Caratterizzare cache Linux/host e throughput, distinguere I/O guest da NAND. Validare forward/decoding e risorse; niente interpretazioni automatiche di `read_bytes`, DONTNEED o O_DIRECT.

### Capacità necessarie per modelli enormi

**M11 — Decoder MoE pubblico e semantica esatta.** Adapter per una famiglia MoE realmente pubblicata con routing top-k, normalizzazione, expert layout, attenzione e eventuale stato ibrido corretti; streaming **solo degli esperti effettivamente selezionati** secondo la semantica del modello, senza cambiarne il routing. Parità su router, esperti, logits, KV e generazione rispetto a un riferimento indipendente.

**M12 — Efficienza end-to-end e scheduling bounded.** Profilare letture NVMe, BF16 expansion, GEMV, tokenizer, KV, prefetch, cache degli esperti e pipeline. Scegliere ottimizzazioni una per volta con confronto A/B e compatibilità numerica. Provare scalabilità di metadata/FD oltre i limiti sperimentali attuali (64 shard) **solo quando necessaria**, con FD a capacità controllata.

**M13 — Accelerazione opzionale CPU/GPU.** Backend FP32/BF16/accumuli appropriati con contratti di accuratezza espliciti, inclusi tile su GPU con 4 GiB VRAM e fall-back CPU. Misurare H2D/D2H, kernel, overlap, RSS/VRAM e consumo energetico; niente obbligo di FP64 ordinato su hardware inadatto, ma baseline FP64 invariata per la diagnosi.

**M14 — Fruibilità e integrazione.** Sessioni robuste, streaming token, endpoint HTTP compatibili OpenAI ove possibile, tokenizer/chat template e tool call fedeli, integrazione **DeepSeek Harness** come client. Test su errori, cancellazione, interruzioni e isolamento delle sessioni.

### Traguardo di lungo periodo — grande scala

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
- **GitHub Projects**, se configurato dal proprietario, è una vista organizzativa delle stesse Issues: **non una seconda roadmap concorrente**. Stati consigliati: Backlog, Ready, In progress, Blocked, Review, Done; campi: Area, Priorità, Milestone. La mancanza di un Project non impedisce di mantenere le issue.

**Regola di aggiornamento:** quando un milestone è davvero chiuso, aggiornare l'esito qui, collegare commit/report/gate e chiudere la relativa issue. Non trasformare in PASS un test omesso, un benchmark cache-warm o una capacità soltanto progettata.
