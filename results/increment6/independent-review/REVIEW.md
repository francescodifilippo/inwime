# InWiMe — revisione architetturale indipendente dell'incremento 6

14 novembre 2026. Commit esaminato: `268486a310ae8997ebbecd56a4f4691f4919355b`.
Parent: `f304ee9f19b6a89907d4ad10cbdce435640352fb`.
Repository locale: `ooc-inference`; remote configurato: `francescodifilippo/inwime`.

## 1. Valutazione complessiva

**Pronto per il push dell'incremento 6. Nessun P0/P1 riproducibile individuato.**
Questo giudizio riguarda l'integrazione del contenitore BF16, entro il profilo
dichiarato e assumendo originali immutabili. Non certifica l'esecuzione di un
modello pubblico addestrato, uno snapshot transazionale del filesystem o migliaia
di shard. Nessun push è stato eseguito.

Esaminati ADR architetturale, rapporti incrementi 5/6, audit Strata (in particolare
il contratto della sezione 6), prima revisione High e relativo staging audit,
diff parent→commit, lettore, cataloghi, coordinatori, decoder, GEMV e test pertinenti.
La verifica è concentrata sulle interfacce dell'incremento 6. Due letture parallele
Standards/Spec secondo la skill code-review hanno affiancato l'ispezione e le prove
del revisore principale; non costituiscono ulteriori misure prestazionali.

Nel diff di produzione cambiano soltanto `core.py`, `decoder.py`, `gemv.py` e il
nuovo `safetensors_source.py`. Kernel C, wrapper matematico, pipeline, riferimento
e fixture originali sono identici al parent. Non si attribuiscono al contenitore
nuovo le prove grandi storiche dell'incremento 5.

## 2. Rilievi classificati

### P0

Nessuno dimostrato.

### P1

Nessuno dimostrato. Non occorrono correzioni di architettura prima del push.

### P2-01 — Il profilo corrente non copre centinaia/migliaia di shard

- **Posizione:** `inwime_runtime/safetensors_source.py:77` e `:139`;
  `inwime_runtime/core.py:421` e `:590`; `inwime_runtime/gemv.py:61`.
- **Condizione:** checkpoint con 65 shard validi; oppure aumento manuale di
  `max_shards` mentre il limite FD del processo è inferiore agli shard ammessi.
- **Conseguenza:** il percorso pubblico rifiuta il primo caso prima di aprire gli
  shard. Il coordinatore apre un FD per shard e può fallire con `EMFILE` nel secondo.
  I costruttori pubblici non espongono gli override del parser. Il costo di FD e
  indici cresce con la cardinalità, non con i byte dei pesi.
- **Evidenza nuova:** 65 shard da un solo coefficiente, **130 byte di payload**
  complessivo: `shard count bound exceeded`. Con sorgente esplicita
  `max_shards=65` e `RLIMIT_NOFILE=32`, errore 24 durante l'apertura; FD prima/dopo
  **5/5**, nessuna perdita. [Risultati](probes.json), [riproduzione](PROBES.md).
- **Attribuzione:** cap introdotto dall'incremento 6; apertura simultanea di tutti
  gli shard preesistente. Il cap 64/4096 è richiesto dal profilo sperimentale
  dell'audit Strata §6.3 e dichiarato nel rapporto: non è una promessa violata.
- **Minimo consigliato:** mantenere esplicito il profilo, come già documentato.
  Prima di estenderlo, esporre limiti e ammissione FD; se occorre un insieme di FD
  a capacità limitata, preservare identità verificata e lifetime delle letture in
  corso. Aumentare solo il numero massimo non dimostra scalabilità. Nessuna
  implementazione necessaria in questa revisione.

### P2-02 — Catalogazione di tensori vuoti e GEMV vuota hanno contratti diversi

- **Posizione:** `inwime_runtime/gemv.py:30` e `:31`;
  `inwime_runtime/safetensors_source.py:210`.
- **Condizione:** tensore BF16 valido `w`, shape `[1,0]`, offset `[0,0]`;
  `op.run('w', array('d'), array('d', [123.]))`.
- **Conseguenza:** il catalogo lo ammette; GEMV fallisce ottenendo l'indirizzo del
  buffer vuoto, con `ValueError: Buffer size too small (0 instead of at least 1
  bytes)`. Nessun risultato matematico viene restituito; nessun byte payload letto.
- **Evidenza nuova:** riprodotto sync/pipeline; output ancora `[123.0]`, FD e
  buffer rilasciati. [Risultati](empty-gemv.json), [riproduzione](PROBES.md).
- **Attribuzione:** assunzione preesistente del GEMV, resa raggiungibile dalla
  sorgente fisica generica. La documentazione promette tensori vuoti **nel catalogo**,
  non tutte le operazioni su di essi. Non ostacola un primo decoder ordinario.
- **Minimo consigliato:** documentare GEMV con dimensioni positive o aggiungere
  successivamente un rifiuto esplicito prima degli export. Non cambiare ora la
  semantica numerica per introdurre un nuovo caso d'uso.

## 3. Verifica delle interfacce e delle garanzie

### Storage e modello

`SafetensorsSource` interpreta file, digest, header, dtype BF16, forme e offset.
Non interpreta famiglia, nomi semantici, ordine degli operatori, routing, tokenizer
o forward. Conservare campi applicativi opachi nel descriptor non equivale a
interpretarli. Il test con config/variante sconosciute verifica questo confine.

`core.open_catalog` restituisce l'indice fisico; GEMV lo usa direttamente.
`decoder.DecoderCatalog` valida il binding dello **specifico decoder sintetico**
e crea un nuovo dizionario che condivide descrittori tensoriali immutabili, senza
rinominare la sorgente. I costi aggiuntivi sono inclusi nelle stime. La collocazione
di `decoder_schema` nel modulo core non introduce interpretazione dello schema
nel lettore safetensors.

La biunivocità/completamento del binding e la presenza obbligatoria di `position`
non sono un'interfaccia universale per tutti i modelli. Un adapter reale può usare
lo stesso indice fisico e costruire la propria vista, inclusi alias per weight tying
o sottointervalli di tensori impacchettati. Non occorre rendere permissivo il binding
sintetico né introdurre un framework generale in anticipo.

### Identità, SHA e TOCTOU

Il descriptor è una radice fiduciaria locale, non autenticazione del publisher.
L'identità include i byte di descriptor, indice e header; non è un hash diretto di
tutto il payload. I digest attesi degli shard sono nel descriptor e includono
prefisso/header. L'indice è verificato anche in modalità `metadata`; `full` legge
e confronta ogni shard con il digest atteso. Nessun fallback silenzioso.

Sequenze concrete controllate nel codice e nei test:

1. **Modifica tra indicizzazione e apertura:** il parser registra firma
   device/inode/size/mtime/ctime; il coordinatore riapre e confronta quella firma.
   Un'incoerenza fallisce prima dell'uso dei pesi, anche con `metadata`.
2. **Sostituzione del pathname dopo apertura:** il coordinatore conserva l'FD
   originale. Legge quel contenuto oppure rifiuta il cambiamento di ctime; non
   acquisisce automaticamente il nuovo inode. Le letture già inviate usano l'FD
   conservato fino al drain.
3. **Modifica/troncamento durante lettura:** controllo della firma prima e dopo
   `readinto`; short read rifiutata; errore worker trasmesso al proprietario senza
   pubblicare una lease parziale. Il test di fallimento sul secondo shard verifica
   anche tutto il banco KV precedente.
4. **Modifica dopo l'ultimo controllo di una tile:** i byte già copiati nello slot
   restano quelli letti. La successiva lettura del file modificato ne controlla la
   firma. Non esiste una promessa di notifica immediata per file non più consultati.

Non è stata dimostrata una sequenza TOCTOU che violi il contratto dichiarato.
Le firme non sono una protezione crittografica del filesystem: il presupposto di
immutabilità, la semantica dei timestamp e l'assenza di writer capaci di eluderli
restano necessari. `metadata` non rileva una corruzione preesistente con struttura
valida; è la rinuncia esplicita alla SHA, non un difetto nuovo.

Gli shard sono ammessi e verificati sequenzialmente, senza snapshot atomico
multi-file o lock su una revisione esterna. In `full` un descriptor fiduciario
fissa l'insieme dei contenuti attesi; la corretta associazione alla stessa revisione
pubblica va stabilita da chi prepara quel descriptor. Se descriptor e hash vengono
aggiornati insieme a pesi sbagliati, la SHA non dimostra autenticità o correttezza
del modello. L'ADR §6 ammette esplicitamente l'assunzione di sorgenti immutabili:
non serve imporre retroattivamente una garanzia più forte.

Offset relativi, estensioni, copertura senza gap/overlap/tail, interi non booleani,
rank/prodotto e limite di offset sono validati prima del payload. Il prodotto usa
interi Python senza overflow macchina; i limiti impediscono offset oltre il range
previsto. Allineamento BF16 relativo richiesto; quello assoluto può essere dispari.
Scalari/vuoti appartengono al catalogo, non automaticamente a ogni operatore.

### Memoria e risorse

Nessuna materializzazione implicita dei tensori: letture payload e SHA usano gli
slot prenotati; le span sono lazy. Le sole copie intere sono metadata limitati.
Lettura input ≤8 MiB, totale ≤16 MiB, complessità/profondità precontrollate,
stima heap ≤64 MiB, 64 shard/4096 tensori/rank 32. Il parser precede il budget del
coordinatore ed effettua parte della stima **dopo** il parsing: non è un allocatore
con hard cap RSS. Limiti input/complessità e RSS separato sono quindi essenziali,
come dichiarato. Non si interpreta `bootstrap_bytes` come misura esatta dell'heap.

Con i default, FD, richieste, staging e strutture operative sono limitati. Il
parser chiude ogni file; un fallimento parziale del coordinatore chiude quelli
già aperti. Le prove includono esaurimento FD, checksum, errore/short read e cancel.
Lettura cancellata significa cessazione logica seguita da drain: una syscall già
iniziata può dover terminare prima della liberazione; nessuna deadline di arresto
fisico è promessa. Una lista di richieste o un grafo dell'intero modello non viene
accumulata dal nuovo percorso.

Le arene del decoder dipendono anche da dimensioni, output e capacità KV; GEMV
esclude esplicitamente input/output del chiamante dal proprio piano. RSS comprende
inoltre interprete, allocator, librerie e stack; la page cache globale è un'altra
quantità. Non emergono nuove confusioni tra queste contabilità.

### Runtime, pipeline e numerica

Un solo proprietario cambia lo stato; i worker leggono in staging esclusivo per
slot. Espansione e consumo avvengono dopo completamento; la lease impedisce il
riuso. Il generatore di stream viene chiuso e drenato anche quando calcolo o letture
falliscono. Non sono state trovate nuove race/deadlock introdotte dal contenitore.

Il decoder lavora sul banco KV candidato e pubblica insieme bank/length/generation.
Il vecchio banco resta intatto per rollback. Il nuovo lettore attraversa questa
stessa strada, senza bypass o retry. Standalone GEMV mantiene il contratto
preesistente: tile già completate possono aver aggiornato l'output prima di un
errore successivo; non offre una transazione su tutto l'output.

La numerica FP64 e l'ordine delle riduzioni sono invariati. Le prove piccole
confrontano ogni fase di prefill/decode, logits, hidden, routing e KV con NumPy
indipendente; confrontano anche byte dei buffer tra formati e GEMV con `math.fsum`.
La nuova prova a due worker conferma l'oracle su dense/MoE in quattro fasi ciascuno.
Nessuna tolleranza è stata cambiata.

Queste prove non sono logits upstream di pesi addestrati. Parità tra contenitori
che condividono il forward non è da sola un oracle indipendente del forward;
qui il confronto NumPy separato copre le piccole fixture. I 507 dot product del
decoder grande e le misure oltre RAM WSL restano evidenze storiche parziali, in
un altro contenitore. Nessuna nuova prova oltre RAM o di prestazioni cold.

## 4. Audit della prima revisione ed esecuzioni nuove

Le due correzioni High sono presenti e pertinenti: binding nel solo decoder e
decodifica UTF-8 esplicita sui tre ingressi. Le regressioni corrispondenti sono
passate anche qui. Le conclusioni su lifetime, drain, SHA e budget sono supportate
entro il contratto sopra precisato; il solo numero **332** non prova proprietà
universali e la suite completa non è stata ripetuta.

Il contenuto di `staging-audit.json` coincide esattamente con i **498 file** del
diff parent→commit; dimensione massima 99.506 byte, sole estensioni testuali
dichiarate. Kernel, pipeline, wrapper, riferimento, oracle e fixture sono invariati.
Questo verifica il perimetro e la conservazione; non è una certificazione generale
di assenza di segreti. Le evidenze storiche non sono state riscritte.

| Nuova verifica | Esito |
|---|---|
| `python3 -m pytest -q tests/test_safetensors_source.py` | **151 passed in 6,46 s**, nessuno skip |
| Pipeline, 3 slot / 2 worker, ritardi diversi | Dense e MoE: oracle completo a ogni fase; concorrenza massima osservata 2; cleanup completo |
| 65 shard minuscoli / limite FD 32 | Rifiuto del profilo e `EMFILE` controllati; FD 5→5 |
| GEMV `[1,0]` | P2-02 riprodotto in entrambi i modi; zero payload letto |
| Diff e whitespace | Nessuna modifica ai file tracciati; `git diff --check` pulito |

Risorse della suite: picco RSS/HWM osservato **62.607.360 byte**, campionamento
10 ms; `ru_maxrss` figli 61.140 KiB. Ultimo snapshot `/proc/<pid>/io`:
`rchar=68.614.557`, `read_bytes=4.321.280`. Questi contatori includono test, parser,
SHA e oracle; non sono byte NAND né un benchmark d'inferenza. Sono snapshot prima
dell'uscita, non contatori finali garantiti. [Log](pytest-source.txt),
[risorse](pytest-resources.json).

Prove aggiuntive: `ru_maxrss=47.572 KiB`. Inferenza dense/MoE rispettivamente
59.520/88.320 byte logici; SHA 13.896/24.680 byte; piani 123.546/160.390 byte.
Le prove includono NumPy e copie diagnostiche esterne alle arene. Nessuna GPU
inizializzata dal runtime, zero allocazioni device del runtime; nessuna misura
globale GPU acquisita. Nessun claim prestazionale, termico o sullo stato cold/warm
derivato dai tempi dei test.

## 5. Prontezza al primo adapter reale

**Il confine storage è utilizzabile per iniziare l'adapter; il decoder pubblico
non è ancora implementato.** Non sono emersi difetti di categoria A che richiedano
correzioni prima del push. I seguenti lavori sono categoria B, incremento successivo:

| Aspetto | Lavoro necessario, senza cambiare il lettore fisico per la famiglia |
|---|---|
| Nomi/configurazione | Adapter esplicito con revisione del modello e schema config; nessun affidamento sui nomi sintetici |
| Layout | Validare orientamento `[out,in]`, proiezioni fuse e assi degli esperti; eventuali trasformazioni/operatori con workspace bounded, senza riscrivere originali |
| Weight tying | Alias logici corretti senza richiedere un duplicato fisico di embedding/lm_head |
| Norm/posizione/attenzione | Equazioni esatte, epsilon, eventuali bias, RoPE e sue varianti, GQA/MQA e geometria KV; l'attuale posizione appresa/MHA non le implementa |
| MoE | Routing, normalizzazione, esperti condivisi o layout impacchettati secondo la specifica scelta; il routing sintetico non basta |
| Precisione/contenitore | Scegliere un checkpoint nel profilo BF16; dtype misti/F16 oggi rifiutati esplicitamente, non convertiti |
| Tokenizer | Tokenizer, special token e template coerenti, separati dal reader |
| Oracle | Confrontare intermedi/logits con upstream fissato a revisione, configurazione e precisione note, mantenendo il riferimento FP64 e motivando le tolleranze |

Non esiste una prova diretta di interoperabilità con il pacchetto safetensors
upstream: le fixture sono scritte localmente. Va aggiunta al percorso del primo
checkpoint pubblico, insieme a provenienza/hash. Il profilo shard/tensori va
controllato prima della scelta/download; non serve costruire ora un'infrastruttura
per tutte le famiglie. Nessun adapter, tokenizer o operatore nuovo è stato avviato.

## 6. Stato finale e decisione

- HEAD resta `268486a310ae8997ebbecd56a4f4691f4919355b`, branch `main`, un commit
  avanti rispetto al riferimento locale `origin/main` (nessun fetch in questa revisione).
- Nessuna correzione di produzione o test esistenti; **nessun secondo commit**.
- Unica aggiunta al working tree: questa directory di rapporto/evidenze, non
  tracciata. Nessuno staging. Le ricette delle prove sono documentate in `PROBES.md`.
- Nessun push, download, installazione, scansione grande, modifica delle evidenze
  storiche o avvio dell'incremento 7.

**Giudizio finale: pronto per il push**, con i due P2 documentati e senza promuovere
il risultato a supporto completo di un LLM reale. Revisione conclusa.

### Esiti separati della skill di review

**Standards:** 0 P0/P1; 1 P2 sul profilo shard/FD. Nessuna violazione sostanziale
dimostrata di bounded I/O, ownership o conservazione delle evidenze.

**Spec:** 0 P0/P1; 2 P2 (profilo shard/FD e GEMV vuota). I contratti dell'audit §6
sono rispettati nel perimetro dichiarato; le funzionalità del modello pubblico
restano esplicitamente fuori dall'incremento. Il P2 shard/FD è condiviso fra i due
assi, non due difetti distinti.
