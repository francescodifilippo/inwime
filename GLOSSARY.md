# InWiMe

InWiMe è un motore di inferenza locale che esegue modelli i cui pesi originali non risiedono interamente in RAM/VRAM, leggendoli in streaming dallo storage con memoria limitata. Priorità: copertura di modelli e hardware, poi correttezza, poi prestazioni.

## Architettura

**Motore**:
L'unico sistema di inferenza InWiMe, composto da componenti modulari (modelli, operatori, formati, dispositivi, storage, risorse) e governato da un solo Coordinatore.
_Avoid_: backend, engine multipli, runtime esterno

**Coordinatore**:
L'autorità unica del Motore su sessioni, stato KV, risorse e scheduling; nessun altro componente possiede lo stesso stato.
_Avoid_: orchestratore, scheduler (come sinonimo dell'intero ruolo)

**Gestore delle regioni**:
La componente risorse storica (`core.Coordinator`) che amministra budget, lease e letture dei pesi dentro un singolo runtime; è subordinata al Coordinatore.
_Avoid_: Coordinatore (per questo ruolo)

**Percorso congelato**:
Il percorso computazionale degli incrementi M7–M10 con i relativi protocolli ed evidenze, che non può essere modificato.
_Avoid_: legacy, vecchio runtime

**Percorso computazionale**:
La combinazione precisa di adapter, operatori, kernel, ordine di accumulo e configurazione di thread con cui un modello viene eseguito.

## Modelli

**Checkpoint**:
L'insieme immutabile dei file originali di un modello (pesi, configurazione, tokenizer) identificato da repository e revisione.
_Avoid_: modello scaricato, snapshot

**Revisione**:
Il commit SHA che fissa in modo univoco il contenuto di un Checkpoint su Hugging Face.
_Avoid_: branch, tag, versione

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
Esiste un piano di risorse che rispetta i limiti di memoria e storage dell'hardware considerato.

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
