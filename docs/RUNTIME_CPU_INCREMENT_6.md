# InWiMe — incremento 6: sorgente safetensors BF16 multi-shard

**Stato consolidato dopo revisione mirata:** due difetti corretti (binding spostato
nell'adapter del decoder e UTF-8 imposto al parser), **332 test superati**.
Conclusioni, misure aggiornate e limiti in
[review/REVIEW.md](../results/increment6/review/REVIEW.md). Le sezioni delle misure
originarie sotto conservano esplicitamente i risultati precedenti alla revisione.

Data: 14 novembre 2026. Base locale `f304ee9f19b6a89907d4ad10cbdce435640352fb`,
branch `main`. Implementazione del solo P0 contenitore della sezione 6
dell'[audit Strata](STRATA_TECH_AUDIT_2026-11-13.md), letto integralmente, inclusa
la sezione 8. Nessun codice Strata importato, push, download, installazione o modifica di
dischi, WSL, swap, driver o impostazioni energetiche. Il consolidamento in un
unico commit locale è autorizzato dalla richiesta di revisione successiva. Le
acquisizioni originarie restano distinte da quelle dopo la revisione, in fondo al rapporto.

## Risultato e baseline verificata

Il decoder sintetico dense/MoE e GEMV leggono direttamente parole BF16 da file
safetensors standard, attraverso `region/spans` e i coordinatori già esistenti.
Gli indici distribuiscono **tensori interi** fra shard; non si introduce uno split
interno al tensore nel formato safetensors. La segmentazione arbitraria v2 resta
disponibile nel suo formato. Nessuna conversione persistente, transpose o
quantizzazione all'ingresso. La normale espansione esatta BF16→FP64 delle sole tile
rimane quella dell'incremento 5; kernel, ordine delle riduzioni e tolleranze invariati.

All'avvio il working tree conteneva soltanto l'audit non tracciato. Le sei suite
della baseline sono state rieseguite prima delle modifiche: **181 passed in 28,43 s**
([log](../results/increment6/pytest-baseline.txt)). Esaminati rapporto, README,
audit probatorio e risultati dell'incremento 5. Le sue prove storiche restano:

- forward denso sintetico su 12.887.482.368 byte unici realmente usati, oltre
  MemTotal WSL 11.483.848.704 byte ma sotto RAM fisica host 16.900.739.072 byte;
- confronto completo indipendente piccolo dense/MoE; sul grande 507 dot product
  su 169 proiezioni, controllo parziale che usa input intermedi del runtime;
- somme FP64 ordinate, lease che impediscono il riuso prematuro, drain delle
  letture e pubblicazione atomica di `(bank, length, generation)` KV;
- vantaggi pipeline dipendenti dal workload, con regressioni documentate.

Non sono stati rieseguiti i dataset da 6–12 GiB. Il nuovo codice non eredita
automaticamente una prova grande safetensors da queste misure storiche.

## Confine d'integrazione

`core.py` estrae in `IndexedSource` gli stessi `region/spans` e in `decoder_schema`
la validazione matematica già usata da `Catalog`. `open_catalog` seleziona
esplicitamente `fixture` (default storico) oppure `safetensors`; altri valori
falliscono. `gemv.py` seleziona soltanto la sorgente fisica; il decoder aggiunge il proprio
adapter di binding. `core.py::open_catalog` restituisce sempre la sorgente fisica.
Il binding semantico safetensors appartiene a `decoder.py::DecoderCatalog`,
che costruisce una vista senza cambiare i nomi fisici nella sorgente. Nessuna
riscrittura di pipeline, scheduler, KV, wrapper C o kernel; nessun nuovo
backend GPU o retry. L'apertura dei file usa anche `O_NONBLOCK` per poter rifiutare
una FIFO sostituita al pathname senza attendere un writer.

`safetensors_source.py` interpreta header, indice e i riferimenti fisici del
descriptor esterno, normalizza offset fisici e forme e conserva soltanto metadati.
I campi applicativi del descriptor sono opachi: nessuna conoscenza di `Spec`,
variante, nomi semantici o operatori del modello nel lettore. Non importa NumPy, PyTorch,
CUDA o la libreria safetensors. La sintassi del contenitore segue la
[specifica ufficiale](https://raw.githubusercontent.com/huggingface/safetensors/main/README.md#format):
lunghezza uint64 little endian, JSON UTF-8, offset relativi al payload, layout C.
Il link al pin riportato nell'audit non era acquisibile tramite web; è stata
consultata anche la documentazione primaria corrente. Non è dichiarata una prova
di interoperabilità con il pacchetto safetensors upstream, non installato localmente.

## Descriptor esterno e API

Una directory sorgente contiene gli originali e il file separato `safetensors.json`.
Esempio schematico, con digest attesi da sostituire con valori di provenienza nota:

```json
{
  "format": "inwime-safetensors-bf16-v1",
  "index": "model.safetensors.index.json",
  "index_sha256": "<SHA256 atteso dell'indice>",
  "files": {
    "model-00001-of-00002.safetensors": {"nbytes": 1234, "sha256": "<SHA256 atteso>"},
    "model-00002-of-00002.safetensors": {"nbytes": 5678, "sha256": "<SHA256 atteso>"}
  }
}
```

L'indice usa il consueto `weight_map: {nome_tensore: nome_shard}`. Deve concordare
esattamente con i tensori presenti e riferire tutti gli shard del descriptor.
Se presente, `metadata.total_size` deve coincidere con il payload complessivo.
Per un solo file è ammesso `"index": null`. I digest dei file coprono **anche
header e prefisso**. Safetensors non incorpora questi checksum: il descriptor
è un input fiduciario esterno, non una firma né una prova autonoma di autenticità.
Le fixture registrano hash al momento della scrittura controllata; il lettore
confronta i byte con quei valori, senza auto-accettare un nuovo hash calcolato.

Per il decoder sintetico si aggiungono `variant`, `config` dello schema già
esistente e `bindings: {nome_runtime: nome_safetensors}`. Il binding è biunivoco,
completo e con forme esatte; non interpreta automaticamente nomi Qwen/DeepSeek/GLM.
Le fixture usano nomi fisici `checkpoint.*` e un ordine diverso da quello matematico.

```python
from array import array
from inwime_runtime import CPURuntime
from inwime_runtime.gemv import StreamedGEMV

# Richiede descriptor con config/variant/bindings del decoder sintetico.
plan = CPURuntime.minimum_plan("percorso", source="safetensors",
                               tile_elements=73, read_elements=13)
with CPURuntime("percorso", source="safetensors", budget=plan.budget(),
                tile_elements=73, read_elements=13) as engine:
    with engine.prefill([1, 7]) as result:
        logits = list(result.logits)

# La sorgente GEMV è generica: non richiede config o binding del decoder.
with StreamedGEMV("percorso", source="safetensors",
                   tile_elements=8191, read_elements=1021) as op:
    op.run("nome_fisico_matrice", array("d", input_values), output_array)
```

`integrity='full'` è il default: scansione SHA256 limitata dallo staging,
fuori dal timer d'inferenza. `metadata` resta una rinuncia esplicita alla verifica
del payload, senza fallback automatico; struttura e hash dell'indice sono comunque
controllati. Le firme inode/device/size/mtime/ctime collegano la lettura degli
header agli FD aperti dal coordinatore e sono ricontrollate attorno alle letture.
Una mutazione dell'inode fallisce. Una sostituzione atomica del pathname mantiene
l'FD originale; il cambiamento ctime può provocare un errore conservativo, come
ammesso dall'audit. Non si riapre per nome durante il forward. Questo contratto
presuppone file immutabili; non è una garanzia contro uno scrittore privilegiato
capace di eludere le firme del filesystem dopo una verifica integrale.

## Validazione e limiti

Rifiutati: chiavi duplicate in tutti i JSON, UTF-8/JSON invalidi, lunghezze troncate
o fuori limite, shape/offset non interi o booleani, dimensioni negative/overflow,
dtype diversi da BF16, estensioni non coerenti, gap/overlap/tail non indicizzati,
tensori duplicati fra shard, riferimenti mancanti/contraddittori, traversal,
symlink finali, file non regolari, checksum errati e cambiamenti del file aperto.
Si ammettono tensori scalari e vuoti nel catalogo generico. Gli offset assoluti
possono essere dispari: non è imposto un allineamento aggiuntivo al formato.
NaN/Inf sono validi come byte del contenitore, ma l'espansione numerica li rifiuta
secondo il contratto preesistente, senza pubblicare una lease o il KV.

Profilo di questa versione: massimo 8 MiB per input JSON/header, 16 MiB di input
metadata complessivi, 64 shard, 4096 tensori, rank 32, profondità JSON 16 e 65.536
marcatori strutturali. I limiti di complessità sono indipendenti dal numero massimo
di tensori. Il parser precontrolla struttura e lunghezza, poi misura/riserva una
stima conservativa di bootstrap/retained con tetto 64 MiB; i dettagli dell'allocatore
JSON/Python restano osservati tramite RSS. Non è un parser streaming a heap
interamente intercettato. Superare un limite produce un errore; nessuna promessa
di accettare ogni checkpoint safetensors arbitrariamente grande o con dtype misti.

Il budget del coordinatore è controllato prima di arene e scansioni dei pesi;
il bootstrap del parser precede questo controllo e ha i limiti metadata propri.
Le arene restano proporzionali a tile, attivazioni/output e KV, non al payload.
Le span sono lazy; il parser non legge il payload, verificato intercettando le
letture. I FD del parser sono chiusi anche in errore; il coordinatore conserva
soltanto i propri FD read-only fino al cleanup.

## Correttezza e fallimenti

La suite prima della revisione comprendeva le 181 regressioni storiche e 129
test aggiunti della sorgente: **310 passed in 33,59 s**. RSS massimo campionato 74.682.368 byte
(71,22 MiB), con stop mai raggiunto. I log conclusivi e la misura RSS sono in
[results/increment6](../results/increment6/README.md).

Sono confrontati tutti i valori originali BF16 attraverso il nuovo percorso,
GEMV con `math.fsum`, prefill di tre token più tre decode in dense/MoE, backend
scalare/C e I/O sync/pipeline. I risultati v1/v2/safetensors coincidono; logits,
hidden e KV sono confrontati anche nei byte FP64, oltre all'oracle NumPy
indipendente con tolleranze storiche. Indici fisici e conversione IEEE dell'oracle
non usano il nuovo parser o `Catalog.spans`.

Short read, errore I/O, cancellazione e mutazione sono iniettati dopo letture
riuscite da un primo shard. Si verifica l'intero banco KV precedente byte per byte,
descriptor pubblicato identico, sessione fallita, slot drenati e FD/thread chiusi.
Coperti anche race fra ammissione e apertura, sostituzione atomica, FIFO, budget
insufficiente prima del payload e cleanup dopo checksum fallito. Nessuna nuova
semantica di rollback standalone GEMV: tile precedentemente completate possono
aver già scritto l'output del chiamante, come nella baseline.

## Confronto A/B/C e misure di memoria/I/O

[Serie prima della revisione](../results/increment6/abc-final/summary.json): A = runtime esportato
esattamente da `f304ee9`, B = candidato sui medesimi file v2, C = candidato su
safetensors con identiche parole BF16. Esportati solo sorgenti piccoli in `/tmp`,
senza worktree, commit o copie dei dataset grandi. Medesima libreria C e backend,
tile 73/read 13, due slot/un worker in pipeline. Ogni campione ha processo nuovo,
SHA completa prima dell'inferenza, un prefill `[1,7,4]` e decode `[7,2,11]`.
Un warm-up per braccio escluso, poi cinque ripetizioni in ordine controbilanciato.
Nessun altro benchmark concorrente. Output e byte per tensore identici in tutti
i campioni; piano numerico e budget A/B esattamente uguali.

| Caso | A mediana [min–max], ms | B, ms | C, ms |
|---|---:|---:|---:|
| Dense sync | 19,069 [18,677–21,775] | 19,810 [18,366–24,449] | 22,043 [19,523–22,668] |
| Dense pipeline | 66,765 [63,443–67,376] | 68,422 [65,979–70,126] | 65,829 [65,175–73,943] |
| MoE sync | 33,641 [32,221–35,757] | 30,138 [27,371–34,715] | 29,104 [26,385–35,634] |
| MoE pipeline | 102,726 [98,518–103,898] | 99,599 [97,804–102,151] | 99,458 [92,838–103,820] |

Gli scarti mediani B/A sono +3,89%, +2,48%, −10,41%, −3,04%: non emerge una
regressione attribuibile al cambio. I range sono ampi e sovrapposti; non si
attribuisce uno speedup al contenitore. Dense sync C/B è +11,27% in questa serie:
un costo piccolo osservato, non una promessa di parità temporale fra contenitori.
Startup C è circa 3,73–6,43 ms, B circa 2,54–4,25 ms; parsing e SHA sono separati
nei JSON. I payload piccoli sono 11.872/21.280 byte, le letture del workload
59.520/88.320 byte; gli header safetensors aggiungono 2.024/3.400 byte e sono
inclusi negli hash completi da 13.896/24.680 byte. Il parsing legge separatamente
5.336/8.806 byte metadata, compresi descriptor, indice e header.

La [prima serie](../results/increment6/abc/summary.json) aveva MoE sync B/A +11,26%;
la [verifica mirata](../results/increment6/regression-moe-sync/summary.json) ha dato
+3,03%, con range sovrapposti. La serie finale cambia anche il segno. Si conservano
tutti i risultati, senza selezionare una sola serie favorevole. Le prime acquisizioni
precedono l'ultimo controllo `total_size` e l'apertura non bloccante; non vengono
attribuite retroattivamente al codice finale. Il confronto conclusivo usa entrambi.

RSS massimo dei decoder A/B/C finali ≤25.988 KiB; i piani sono 61.232–117.159 byte.
L'incremento di metadata C/B è +1.879 byte dense e −292 byte MoE per queste fixture,
ampiamente entro 32 MiB. Sono stime del catalogo e contabilità delle arene,
distinte dal processo completo e dalla page cache globale.

Una [prova GEMV dedicata](../results/increment6/probe-final-sync.json), anche
[pipeline](../results/increment6/probe-final-pipeline.json), usa 513×4099 = 2.102.787
coefficienti originali BF16, **4.205.574 byte**, generati in chunk ≤64 KiB.
Tutte le 513 righe sono verificate indipendentemente leggendo una riga per volta;
errore massimo `1,9984e-14`, SHA output identica fra modalità.

| Misura GEMV | Sync | Pipeline |
|---|---:|---:|
| Piano controllato, byte | 140.083 | 260.909 |
| Slot pesi FP64 complessivi, byte | 65.528 | 131.056 |
| Massimo `readinto`, byte | 2.042 | 2.042 |
| Input/output del chiamante, byte aggiuntivi | 36.896 | 36.896 |
| Bootstrap catalogo stimato, incluso nel piano | 6.977 | 6.977 |
| RSS `ru_maxrss`, KiB | 24.988 | 25.048 |
| Hash iniziale completo, byte | 4.205.662 | 4.205.662 |
| Letture inferenza, byte | 4.205.574 | 4.205.574 |
| `/proc/self/io/read_bytes` inferenza | 0 | 0 |

Ulteriori letture separate: metadata 309 byte, riferimento 4.205.574 byte più
prefisso/buffering, hash finale 4.205.662 byte. Il riferimento usa al massimo una
riga raw di 8.198 byte più temporanei scalari; l'oracle residente NumPy delle fixture
decoder è confinato ai test e compreso nel loro RSS. Le misure non dimostrano
un file safetensors oltre la RAM del processo o dell'host.

Tutte le acquisizioni sono **cache-warm**, dopo verifica SHA, su ext4 interno WSL.
`read_bytes=0` indica assenza di letture di blocco attribuite ai processi nel workload,
non banda NVMe o NAND misurata. Nessun drop-caches o DONTNEED in questo incremento.
La serie misura compatibilità e overhead su fixture piccole, non prestazioni cold
out-of-core. AC1/online=1; temperatura, potenza e clock non disponibili tramite
le sonde sysfs interrogate e non controllati. GPU non inizializzata, pinned/device
del runtime zero; nessuna misura globale NVML/CUDA acquisita.

Il supervisore campiona RSS ogni 10 ms con stop a 1 GiB; RLIMIT_AS 512 MiB per
worker benchmark/probe, distinto da RSS. Tutti i processi osservati restano sotto
l'obiettivo 512 MiB; gli snapshot `VmSwap=0` non escludono transitori tra snapshot.
I file fixture nuovi persistenti, tutte le copie dei confronti e la prova GEMV,
occupano **4.406.063 byte**; report/log restano sotto 16 MiB. I payload temporanei dei
test sono piccoli e confinati al tmp pytest. Nessun checkpoint grande scaricato.

## Portata e revisione successiva

**Dimostrato:** sorgente BF16 safetensors effettiva multi-shard, letture in posto
bounded, parità completa sulle fixture, controllo di integrità e riferimenti,
regressioni e invarianti transazionali preservati nei casi eseguiti.

**Ancora necessario:** adapter esatto di un modello pubblico con architettura,
tokenizer/template, attenzione/RoPE/GQA/MoE e weight tying corretti, provenienza e
revisione dei pesi, confronto di intermedi/logits upstream e inferenza completa
nel budget. Non individuati piccoli checkpoint safetensors locali nelle directory
Git/cache Hugging Face controllate; nessun modello pubblico usato. Restano inoltre
la prova oltre RAM fisica host, contesti lunghi, F16 e formati nativi diversi da BF16.
GPU, cache VRAM, MTP, batching, nuovi kernel e Harness non fanno parte del cambio.

L'adapter è opt-in: tornare a `source='fixture'` non richiede migrazioni o riscritture.
La revisione mirata e il consolidamento sono descritti sotto; nessun incremento
ulteriore è avviato e nessun push è eseguito.

## Esito della revisione e consolidamento

La revisione ha confermato bounded I/O, SHA distinta dall'inferenza, firme dei file,
ownership, drain e transazione KV, oracle indipendente e tolleranze immutate. Ha
corretto due difetti: la sorgente ora conserva soltanto nomi/forme/span fisici e
metadati, con binding nel solo `decoder.py::DecoderCatalog`; i tre ingressi JSON
ora accettano esclusivamente UTF-8. Test nuovi coprono entrambi i difetti e il
troncamento dopo ammissione/apertura, senza allargare il perimetro a nuovi modelli.

Suite dopo correzioni: **332 passed in 43,89 s**, RSS campionato 74.649.600 byte.
Nuova GEMV completa: stessi 4.205.574 byte, stesso errore massimo 1,9984e-14,
piani 140.927/261.753 byte e RSS 25.024/25.120 KiB. Il nuovo A/B/C è in
`results/increment6/review/abc/`: dati numerici e I/O conformi; prestazioni
inconclusive per variabilità, incluso MoE sync B/A +10,72%. Nessuno speedup o
garanzia di regressione sotto il 10% viene dichiarato. I dettagli e tutti i
campioni sono nel [rapporto mirato](../results/increment6/review/REVIEW.md).

Le nuove fixture persistenti, inclusa la ripetizione dopo correzioni, occupano
4.506.153 byte; nessun payload entra nel commit. Il manifest prima della revisione
identifica i vecchi sorgenti; quello finale identifica codice e documentazione
consolidati. È autorizzato un solo commit locale con staging esplicito, comprendente
anche l'audit Strata già presente come riferimento della specifica. Nessun push,
download di checkpoint reali o avvio dell'incremento 7.
