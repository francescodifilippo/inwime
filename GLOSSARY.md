# InWiMe

InWiMe è un motore di inferenza per **qualunque modello e dimensione, come obiettivo progressivo**, che utilizza le risorse realmente disponibili (CPU, GPU, acceleratori e in prospettiva nodi distribuiti; RAM, VRAM, NVMe e altri storage) e ricorre a streaming/out-of-core quando i pesi non possono risiedere interamente in memoria. La fattibilità richiede i minimi fisici specifici per modello, contesto e hardware, da diagnosticare esplicitamente; nessuna promessa di eseguibilità universale già realizzata. **CPU-first e 16 GiB di RAM descrivono il percorso iniziale di sviluppo/test, non un vincolo finale.** Priorità: fedeltà e correttezza verificabile, copertura ed eseguibilità entro le risorse, poi prestazioni.

## Architettura

**Motore**:
L'unico sistema di inferenza InWiMe, composto da componenti modulari (modelli, operatori, formati, dispositivi, storage, risorse) e governato da un solo Coordinatore. Un unico Motore può pianificare calcolo e dati su più dispositivi o nodi, senza trasformarsi in una piattaforma di backend indipendenti.
_Avoid_: backend, engine multipli, runtime esterno

**Coordinatore**:
L'autorità unica del Motore su sessioni, stato KV, risorse e scheduling; nessun altro componente possiede lo stesso stato.
_Avoid_: orchestratore, scheduler (come sinonimo dell'intero ruolo)

**Gestore delle regioni**:
La componente storica (`core.Coordinator`) che gestisce prenotazioni di memoria, lease, accesso alle regioni dei pesi e I/O di un singolo runtime; è subordinata al Coordinatore.
_Avoid_: Coordinatore (per questo ruolo)

**Percorso congelato**:
Il percorso computazionale degli incrementi M7–M10 con i relativi protocolli ed evidenze, che non può essere modificato.
_Avoid_: legacy, vecchio runtime

**Percorso computazionale**:
La combinazione precisa di adapter, operatori, kernel, ordine di accumulo e configurazione di thread con cui un modello viene eseguito.

**Percorso additivo**:
Il percorso computazionale introdotto dopo M10 che affianca il Percorso congelato senza modificarlo e, dove entrambi accettano il Checkpoint, ne riproduce i risultati bit per bit.
_Avoid_: percorso ottimizzato, nuovo runtime

## Attestazioni numeriche

Tre attestazioni distinte, mai intercambiabili.

**Bit-identità**:
Uguaglianza esatta, bit per bit, di logits e stati tra due Percorsi computazionali del Motore sulla stessa configurazione certificata.

**Equivalenza con l'oracolo**:
Accordo entro tolleranze dichiarate tra il Motore e un oracolo indipendente (Transformers FP32, riferimento FP64).

**Parità dei token**:
Uguaglianza della sequenza di token generati in modo greedy tra il Motore e l'oracolo su prompt renderizzati.

## Modelli

**Checkpoint**:
L'insieme immutabile dei file originali di un modello (pesi, configurazione, tokenizer) identificato da repository e revisione.
_Avoid_: modello scaricato, snapshot

**Revisione**:
Il commit SHA che fissa in modo univoco il contenuto di un Checkpoint su Hugging Face.
_Avoid_: branch, tag, versione

**Registro dei modelli**:
L'inventario locale dei Checkpoint installati (con il manifest di verifica di ogni file) e importati (solo come riferimento a un percorso, mai come copia), e del modello attivo.
_Avoid_: catalogo (che è il Hub), cache (che è quella di Hugging Face)

**Contesto autorizzato**:
Il dominio di autorizzazione identificato da una API key del server, entro cui è ammesso il riuso del prefisso KV; non identifica una singola conversazione.
_Avoid_: sessione, conversazione, utente

**Quantizzazione silenziosa**:
Qualunque conversione dei pesi originali verso una rappresentazione diversa non richiesta esplicitamente dall'utente; vietata.

## Stati di capacità

Ogni Checkpoint avanza lungo questi stati, in ordine; nessuno stato si deduce dalla sola identificazione del Checkpoint.

**discovered**:
Il Checkpoint è stato individuato (Hub o percorso locale) e identificato per repository e Revisione.

**metadata_parsed**:
Configurazione, indice dei tensori, dtype e tokenizer del Checkpoint sono stati letti e validati.

**graph_resolved**:
La semantica completa del forward (operatori, ordine, forme, stato) è nota tramite un adapter o un grafo importato.

**operators_resolved**:
Ogni operatore del grafo ha un'implementazione nel Motore con semantica, dtype e contratto di memoria dichiarati.

**resource_plan_feasible**:
Esiste un piano di risorse che rispetta i limiti di memoria (RAM/VRAM/stato/workspace), storage, compatibilità degli operatori/dispositivi e altri vincoli oggettivi dell'host considerato. Non implica una velocità interattiva né un minimo RAM/NVMe uguale per tutti i modelli.

**end_to_end_executable**:
Il Checkpoint ha prodotto token reali attraverso il Motore entro i limiti di risorse dichiarati.

**numerically_verified**:
Un Checkpoint a una Revisione precisa, eseguito con un Percorso computazionale e una configurazione di test precisi, ha superato il confronto con un oracolo indipendente.

**Vista di supporto**:
Il riassunto per l'utente degli Stati di capacità nelle sole etichette `supported`, `partial`, `unknown`.
_Avoid_: certificato, compatibile (come stato)

## Esiti dei gate

**PASS / FAIL / BLOCKED / NOT PROVEN**:
I quattro esiti distinti di un gate: superato, fallito, impedito da una precondizione mancante, non dimostrato dalle evidenze disponibili.
