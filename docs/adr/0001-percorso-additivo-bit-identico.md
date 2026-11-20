---
status: accepted
date: 2026-11-20
---

# Percorso additivo bit-identico accanto al Percorso congelato

Per la v0.1.0-alpha (#26) il Motore deve servire prompt reali di DeepSeek Harness (circa 5.300 token per turno, misurati su `@deepseek-ai/dsh@0.2.2-rc.5`). Il Percorso congelato di M7–M10 non può farlo: il prefill elabora un token per volta rileggendo tutti i pesi, alloca `logits` e `hidden` proporzionali all'intera capacità in FP64 e usa un solo thread. Decidiamo di **affiancare** al Percorso congelato un Percorso additivo — prefill a blocchi, multithreading deterministico per righe di output, riuso del prefisso KV — che riproduce **bit per bit** i risultati del Percorso congelato dove entrambi accettano il Checkpoint, invece di modificare il codice e i protocolli che sostengono le evidenze M7–M10.

## Considered Options

- **Modificare il Percorso congelato.** Scartata: invaliderebbe la riproducibilità dei gate M7–M10 e mescolerebbe evidenze storiche con codice nuovo.
- **Accettare contesti minuscoli.** Scartata: l'harness diventa inutilizzabile (prefill di ore, memoria proporzionale al prompt).
- **Percorso additivo con sola equivalenza a un oracolo, senza bit-identità.** Scartata come requisito di default: l'ordine di accumulo FP64 per elemento è preservabile, quindi la bit-identità è ottenibile e dà un'attestazione più forte e più economica dell'oracolo.
- **Percorso additivo bit-identico.** Scelta.

## Decisione

- **Prefill a blocchi.** I pesi di un tile vengono espansi una volta e applicati a più posizioni. Ogni elemento di output conserva l'ordine di accumulo FP64 del Percorso congelato. Causal mask, RoPE, posizioni e KV sono gestiti esplicitamente. I `logits` si calcolano solo dove servono, di norma sull'ultima posizione. La memoria temporanea ha un limite esplicito, indipendente dalla lunghezza del prompt.
- **Multithreading deterministico.** I thread si partizionano per righe di output indipendenti. Nessuna riduzione concorrente sulla stessa riga. Modalità floating-point e FMA sono controllate da opzioni del compilatore. Il fallback Python puro resta.
- **Operatori nativi.** Attention, softmax, RoPE, RMSNorm, SiLU e residual passano in C. Si portano in C anche le primitive strettamente necessarie al prefill a blocchi. La semantica è quella del codice Python del Percorso congelato, che resta come fallback. Stessa libm e stesso ordine apparente delle operazioni **non** bastano a garantire la bit-identità. Il gate controlla nell'ambiente certificato:
  - ordine di valutazione;
  - arrotondamenti intermedi;
  - FMA e opzioni del compilatore;
  - funzioni matematiche;
  - valori limite.

  Un operatore che fallisce il gate non viene dichiarato equivalente. Nessun buffer di attention è quadratico nel contesto. I kernel nativi controllano la cancellazione con frequenza limitata e dichiarata.
- **KV append-only con troncamento esplicito.** Il secondo banco e la copia integrale della KV del Percorso congelato vengono eliminati.
  - Se la richiesta estende esattamente lo stato pubblicato, i token candidati si scrivono oltre la lunghezza pubblicata e il commit pubblica la nuova lunghezza.
  - Se il prefisso diverge, si applica la strategia 2: la KV è cache riutilizzabile, non stato da conservare. Esempio: KV pubblicata di 1.000 token, prefisso comune di 600. Prima di scrivere, il Coordinatore **pubblica il troncamento** a 600 e aggiorna la chiave di compatibilità ai 600 token. I 400 token di suffisso sono invalidati esplicitamente e il loro contenuto non viene dichiarato conservato. Il calcolo nuovo diventa così un'estensione pura.
  - Su cancellazione o errore lo stato visibile è l'ultimo pubblicato: 600 token validi, oppure 1.000 se non c'era divergenza. Nessuno stato parziale diventa visibile.
  - Copy-on-write e pagine versionate sono state scartate: costano memoria e complessità, e servono solo a conservare un suffisso che una cache con un solo slot non deve garantire.
  - I gate coprono sei scenari:
    - cancellazione durante un'estensione;
    - divergenza del prefisso;
    - troncamento;
    - errore I/O;
    - errore computazionale;
    - riuso successivo.
- **Riuso del prefisso KV.** Una cache con un solo slot si riusa solo se tutte queste condizioni valgono:
  - i token sono un prefisso esatto della richiesta precedente;
  - la chiave di compatibilità coincide: Checkpoint e Revisione, tokenizer, rendering del template, posizioni, configurazione, Percorso computazionale, parametri che influenzano la KV e Contesto autorizzato.

  Il Contesto autorizzato è la API key bearer. Delimita dove il riuso è ammesso, ma non identifica una conversazione. Due conversazioni con la stessa key condividono la KV solo per corrispondenza esatta di prefisso e chiave. Il Motore non deduce mai che appartengano alla stessa conversazione. Key diverse non condividono mai la cache. Il riuso si può disattivare. La memoria trattenuta viene contabilizzata, invalidata ed espulsa.
- **Capacità del contesto.** La capacità effettiva è calcolata dal piano di memoria, che somma:
  - KV, attivazioni e accumulatori;
  - workspace, metadati e cache;
  - limiti cgroup.

  I limiti superiori configurabili sono 16.384 token per il 0.5B e 8.192 per il 7B. Il 7B a 8.192 token richiede 1,88 GB di sola KV: non si assume fattibile sotto un cgroup da 2 GiB. Una richiesta oltre la capacità riceve HTTP 400 `context_length_exceeded` prima di iniziare l'inferenza.

## Quando la bit-identità è richiesta

- Il gate è obbligatorio quando il Percorso congelato accetta il Checkpoint: è il caso dei Qwen2 non tied, come Qwen2.5-7B-Instruct. Confronta `logits` e stati intermedi rilevanti (residual, KV) su casi deterministici comparabili. L'uso di GEMM non garantisce di per sé la bit-identità: va dimostrata.
- Per configurazioni che il Percorso congelato non accetta (Qwen2 con `tie_word_embeddings=true`) valgono due attestazioni sostitutive:
  - equivalenza con l'oracolo e parità dei token;
  - bit-identità tra testa legata e testa duplicata, su fixture controllate.
- Il riuso del prefisso è bit-identico al prefill completo da zero, sia per `logits` sia per KV.
- La garanzia vale solo per le configurazioni certificate: CPU, compilatore, libm e numero di thread. Non si estende a piattaforme non verificate.

## Memoria e isolamento

Il gate di memoria verifica separatamente la crescita della KV e quella delle attivazioni. Il prefill a blocchi non introduce allocazioni proporzionali all'intero prompt fuori budget. Se servono paging o streaming della KV, si implementano come capacità separata con un proprio gate. Cancellazione ed errori non pubblicano mai uno stato KV candidato.

## Rischi, fallback e rollback

- **Rischi:**
  - differenze di libm o compilatore che rompono la bit-identità;
  - data race nel multithreading;
  - errori di invalidazione della cache del prefisso.
- **Fallback:** una funzionalità del Percorso additivo diventa operativa solo dopo il superamento dei suoi gate. Fino ad allora resta disponibile il percorso precedente: Python puro, un thread, nessun riuso.
- **Rollback:** se un gate fallisce su una configurazione, la funzionalità si disattiva per quella configurazione e il Motore torna al percorso precedente, senza toccare il Percorso congelato.

## Consequences

- Nel codice coesistono due percorsi, e il Percorso congelato non si rimuove né si ottimizza.
- I risultati, i protocolli e le evidenze di M7–M10 restano invariati.
- I gate M7–M10 si rieseguono come regressione a ogni PR pertinente.
