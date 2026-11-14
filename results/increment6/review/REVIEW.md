# Revisione mirata e consolidamento dell'incremento 6

14 novembre 2026; base `f304ee9f19b6a89907d4ad10cbdce435640352fb`.
Perimetro: solo modifiche incremento 6, comprese sorgente, binding, test e driver.
Due letture indipendenti secondo la skill code-review: Standards e Spec; nessuna
nuova revisione generale dell'architettura. Nessun checkpoint scaricato o push.

## Standards

Nessuna violazione sostanziale rilevata. Confermati contatori e mediane dei JSON
preesistenti, denominazione cache-warm, assenza di claim di speedup, limiti della
GEMV piccola, assenza di payload/credenziali nei risultati. Nessun refactoring
proposto sulla sola base di code smell. I pathname presenti nella provenienza
sono metadati tecnici dell'ambiente; non sono file locali aggiunti al commit.

## Spec

Due difetti sostanziali individuati e corretti, poi riesaminati senza ulteriori
rilievi sostanziali nel perimetro:

1. Il lettore interpretava ancora `config/variant/bindings` del decoder sintetico.
   Ora `SafetensorsSource` conserva esclusivamente indice fisico e metadati;
   `decoder.py::DecoderCatalog` valida lo schema e produce una vista semantica
   senza modificare la sorgente. Il descriptor conserva i campi applicativi
   opachi, la cui identità è già coperta dall'hash dei byte originali. L'indice
   aggiuntivo e lo schema sono inclusi nel budget. Nessun operatore nuovo.
2. `json.loads(bytes)` riconosceva anche UTF-16/32: un header UTF-16LE poteva
   iniziare con `{` e superare il controllo iniziale. Ora la decodifica UTF-8 è
   esplicita per descriptor, indice e header. Nuove regressioni riproducono
   UTF-16LE/UTF-32LE nei tre ingressi, con digest esterni coerenti ma rifiuto
   strutturale. Il tipo `bindings` è inoltre controllato esplicitamente.

## Esito delle otto verifiche richieste

| Verifica | Esito e limite |
|---|---|
| Separazione | Sorgente sempre fisica; mapping esclusivamente nell'adapter del decoder. Test di sorgente con metadati di un modello sconosciuto e vista che non ne cambia i nomi. |
| Bounded | Letture pesi e SHA negli slot del coordinatore. Nessun mmap o caricamento completo di shard/tensori. JSON metadata intero ma limitato a 8 MiB/input, 16 MiB aggregati, complessità/rank/numero shard-tensori limitati; stime heap e RSS distinti. SHA può scaldare la page cache globale, fuori dal budget RSS del processo. |
| Integrità | Offset/dimensioni/interi/allineamento relativo BF16, duplicati, overlap/gap/tail, riferimenti e completamenti validati. SHA full attesa da descriptor esterno, distinta da inferenza; metadata è rinuncia esplicita, senza fallback. |
| Mutabilità | Firme inode/device/size/mtime/ctime all'ammissione, all'apertura e attorno alle letture; mutazione/troncamento falliscono. Sostituzione pathname conserva l'FD originale o fallisce conservativamente. Aggiunti troncamenti dopo indice/apertura in entrambi i modi. Non è difesa contro writer privilegiati che eludano le firme. |
| Runtime | Stessi coordinatori, budget, lease, worker e drain. Fallimenti dopo primo shard preservano tutto il banco KV e il descriptor pubblicato. Nessun percorso speciale o retry. |
| Numerica | Oracle NumPy separato sul v1 originale, confronti completi ogni fase su logits/hidden/routing/KV, più parità byte dei buffer e GEMV indipendente fsum. Tolleranze e kernel invariati. |
| Evidenze | A/B/C cache-warm, startup/index/SHA separati. Nessun claim di accelerazione; GEMV da 4,2 MB dichiarata piccola. Misure prima/dopo correzioni identificate separatamente. |
| Futuro modello reale | `core.open_catalog` offre nomi/forme/span fisici a nuovi adapter; tokenizer, operatori e oracle upstream restano esterni. Nessuna famiglia LLM, checkpoint reale o incremento 7 implementato. |

## Validazione dopo le correzioni

- `pytest-source.txt`: **151 passed**, inclusi 22 casi aggiunti rispetto ai 310
  test complessivi precedenti. `pytest-final.txt`: **332 passed in 43,89 s**;
  `pytest-resources.json`: RSS campionato **74.649.600 byte**, intervallo 10 ms,
  stop 1 GiB mai raggiunto. Sono incluse tutte le 181 regressioni originali.
- `abc/summary.json`: nuova serie completa sul codice corretto, baseline A
  esportata dal commit fisso, B stesso v2, C safetensors. Un warm-up e cinque
  ripetizioni per braccio/modo/variante, nuovi processi. Output, byte per tensore
  e GEMV identici; piani A/B identici. I nuovi metadata del binding C sono
  contabilizzati: circa +19,9 KB dense e +26,2 KB MoE rispetto a B.
- `probe-sync.json`, `probe-pipeline.json`: stesso payload BF16 da 4.205.574 byte,
  tutte le 513 righe verificate, errore massimo **1,9984e-14**, output SHA uguale.
  Piani **140.927 / 261.753 byte**, RSS **25.024 / 25.120 KiB**, input/output
  chiamante 36.896 byte separati, massimo readinto 2.042 byte. Metadata bootstrap
  sorgente 7.821 byte incluso nei piani. I/O di blocco inferenza zero; cache-warm.
- `stdlib-only.json`: CPU scalare eseguita con `python3 -S`, senza NumPy,
  PyTorch, CUDA o pacchetto safetensors; nomi fisici preservati.
- `artifact-sha256-before-review.json`: fingerprint originali per identificare
  le vecchie acquisizioni; non devono essere verificati contro i file corretti.
  Il manifest finale nella directory padre copre invece il codice consolidato.

| Caso dopo revisione | A mediana [min–max], ms | B, ms | C, ms |
|---|---:|---:|---:|
| Dense sync | 22,392 [19,482–43,488] | 20,988 [18,565–22,437] | 20,370 [19,959–23,569] |
| Dense pipeline | 70,292 [69,114–144,300] | 71,143 [68,194–117,740] | 69,956 [63,284–72,955] |
| MoE sync | 29,988 [29,551–99,759] | 33,202 [29,686–40,649] | 30,764 [27,956–44,738] |
| MoE pipeline | 103,559 [101,106–113,516] | 98,709 [92,093–146,659] | 97,169 [96,268–106,026] |

MoE sync B/A è **+10,72%** in questa acquisizione; nelle serie precedenti il
segno e l'ampiezza variavano. I range sono sovrapposti e contengono campioni molto
più lenti delle mediane. La prestazione resta **inconclusiva**: non si attribuisce
la variazione al codice né si afferma di aver escluso ogni regressione entro il
10%. Nessuna nuova ripetizione viene selezionata per ottenere un risultato
favorevole. Contatori, piani A/B e controlli numerici sono invece deterministici.
AC osservata; potenza/clock/temperatura non disponibili nelle sonde interrogate.

## Consolidamento

Il commit comprende soltanto sorgenti, test, documentazione, audit di riferimento
e piccoli risultati testuali/JSON con provenienza. Esclusi `checkpoints/`,
`build/`, cache Python/pytest, ambienti, launcher locali, dati privati e segreti.
I descriptor dei payload locali restano identificati nel manifest senza includere
i pesi stessi. L'audit Strata già presente e non tracciato è incluso invariato
come specifica richiamata dalla documentazione, non come nuovo audit generale.
La verifica del contenuto staged e l'elenco file sono registrati in
`staging-audit.json` e nell'output Git del commit.

Resta da realizzare l'adapter esatto di un modello pubblico, tokenizer/template,
operatori e confronto logits upstream; nessuna prova nuova oltre RAM fisica host,
nessuna GPU/cache VRAM/MTP/quantizzazione/nuovi kernel. Arresto dopo il solo
commit locale dell'incremento 6, senza push.
