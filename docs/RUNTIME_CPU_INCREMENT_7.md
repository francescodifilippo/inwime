# Incremento 7 — primo LLM pubblico CPU

> Aggiornamento — risoluzione numerica del 14 novembre 2026: **gate coerente
> superato**, con parità Transformers standard FP32 e riferimento PyTorch
> realmente FP64. Il confronto standard FP64 rimane obbligatorio come diagnostico
> mixed precision e conserva il FAIL. Le sezioni iniziali documentano l'esito
> storico parziale; la risoluzione e il nuovo criterio sono descritti in fondo.

14 novembre 2026. Base `88aea1a897e26efb61408c5ef68ccd40de90ae80`, `main`.
**Esito parziale: adapter funzionante e parità FP32 superata; criterio FP64
standard non superato. Incremento non dichiarato completato.** Nessun commit,
staging, push o avvio dell'incremento 8.

## Formato e provenienza

Modello [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2),
revisione immutabile `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0.
L'API ufficiale con `blobs=true` conferma revisione, dimensioni, licenza e digest
LFS proposto. Il download originale è stato verificato prima dell'uso e nuovamente
alla fine: SHA256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
File 269.060.552 byte, payload 269.030.016 byte, **272 tensori tutti BF16**.
Config confermata: 30 blocchi, hidden 576, FFN 1536, Q head 9, KV head 3,
dim 64, vocabolario 49152, contesto dichiarato dal modello 8192, RMS epsilon
1e-5, theta 100000, SiLU, nessun bias, embedding condiviso. Nessuna discrepanza.
Questi valori sono dati del checkpoint, non limiti generali dell'adapter.

Scaricati soltanto checkpoint, config, generation config, tokenizer JSON,
tokenizer config, special token map e README con licenza. URL, dimensioni e hash
per ciascun file in [provenance.json](../results/increment7/provenance.json).
Nessuna copia locale SmolLM2 trovata nell'inventario checkpoint iniziale.
Spazio iniziale ext4 WSL circa 855 GiB, RAM disponibile circa 8 GiB.
Originali conservati in `checkpoints/smollm2-135m-93efa2f/`, esclusi da Git.
Nessuna trasformazione, quantizzazione, transpose persistente o modifica dei pesi.
Venv CPU isolato circa 1002 MiB; nessuna installazione di sistema. Configurazione
WSL, dischi, swap e driver invariati.

Il repacker separato `tools/repack7.py` crea tre shard standard con tensori interi,
indice `weight_map` e SHA256. Payload copiato a blocchi di 64 KiB; controllo SHA
del sorgente e firme dei file, hash per tensore prima/dopo la copia. Nessuna
conversione di dtype. Unica copia aggiuntiva dei pesi, necessaria alla prova,
in `checkpoints/smollm2-135m-93efa2f-sharded/` (circa 259 MiB).
[repack.json](../results/increment7/repack.json) registra identità, indice, hash
degli shard e tutti i tensori. Anche **Transformers carica realmente gli shard**:
tutte le sue catture FP32 coincidono esattamente con il file ufficiale.
[Verifica dei formati e dei modi](../results/increment7/format-and-modes.json).

## Adapter e matematica

`SafetensorsSource`, `DecoderCatalog`, coordinatori, pipeline e kernel preesistenti
rimangono identici alla baseline; [conservazione](../results/increment7/baseline-preservation.json).
`inwime_runtime/llama.py` aggiunge una vista semantica `LlamaCatalog`, un piano
proprio e una sessione. Il runtime specializzato riusa da `CPURuntime` soltanto
primitive di acquisizione, embedding, norm, SwiGLU e lifecycle; implementa il
forward e l'attenzione Llama. Nessun allargamento dello schema sintetico.

Mapping esplicito delle matrici `[out,in]`, alias logico `lm_head` allo stesso
tensore embedding. Prefill seriale per token causale: embedding → per ogni blocco
RMSNorm → Q/K/V → RoPE split-half alle posizioni assolute → GQA con softmax
stabile e scaling `1/sqrt(head_dim)` → O → residuo → RMSNorm → SwiGLU → residuo;
poi norm finale e proiezione logits condivisa. Nessun embedding posizionale fittizio.
Il query head `h` usa KV head `h // (Q_heads/KV_heads)`. Il KV conserva soltanto
KV_heads*head_dim valori per posizione, senza espansione permanente al numero Q.

Pesi acquisiti esclusivamente tramite lease a tile bounded; espansione BF16
esatta in FP64 e GEMV con riduzioni FP64 ordinate. Nessun kernel modificato.
Percorso scalare autonomo disponibile; le misure reali usano `NativeMath` C
compilato con `make cpu-kernel`, senza CUDA/BLAS/torch.

Il profilo iniziale accetta Llama denso con embedding condiviso serializzato una
sola volta, SiLU, head dimension pari, RoPE standard split-half, GQA divisibile,
nessun bias e `pretraining_tp=1`. Rifiuta config/forme/tensori diversi, incluso
un `lm_head.weight` fisico aggiuntivo: non ne presume l'uguaglianza. Non promette
compatibilità con tutti i Llama, RoPE scaling, modelli untied o altre famiglie.

## Oracolo e criteri numerici

`tools/oracle7.py` **non importa InWiMe**. Usa Transformers 4.57.3, PyTorch
2.9.1+cpu, NumPy 2.3.5; tokenizers 0.22.1 e safetensors 0.7.0 nel venv di verifica.
Carica da disco localmente BF16 originale, verifica i parametri BF16 e converte
soltanto il riferimento residente in FP64/FP32. `trust_remote_code=False`,
attenzione `eager`, eval/inference mode, due thread CPU, seed 0.

Prima delle prove fissati `abs_error <= atol + rtol*abs(reference)` per ogni
valore: FP64 `2e-5 + 2e-6*abs(ref)`; FP32 `2e-3 + 2e-4*abs(ref)`.
Soglie sperimentali per 30 blocchi, riduzioni 576/1536, proiezione vocabolario e
operatori non lineari; il termine assoluto protegge gli zeri. Il margine FP64
prevedeva già gli operatori interni FP32 di Transformers, ma si è rivelato
insufficiente in alcuni elementi. Non è un bound teorico generale. **Soglie
mai allargate dopo il fallimento**, soglie del decoder sintetico invariate.

Catturati: tutti i logits di ogni nuovo token, output residuale di tutti i 30
blocchi (prima della norm finale), K dopo RoPE e V senza ripetizione GQA,
ID e argmax. La cache upstream `[batch,kv_heads,seq,dim]` viene solo riordinata
nel verificatore in `[seq,kv_width]`. Anche l'ultimo blocco ha un hook esplicito:
`hidden_states[-1]` di Transformers sarebbe già normalizzato e non comparabile.

Scenari: singolo ID 504; prefill di `[504,3575,282,4649,314]`; chunk 2+3;
lo stesso prefill più tre decode greedy `[260],[3575],[282]`; forward completo
di otto token; prefisso ripetuto. Nessuno stop EOS nelle prove di forward.
Il verificatore confronta tutti i **49152 logits per posizione**, errori assoluti,
relativi con floor 1e-8 e RMS, metriche di ciascun layer, top-10 e argmax.
La relativa vicino a zero è diagnostica, non un criterio separato.

| Riferimento | Max abs logits | Max abs residuali | Max abs K/V | Esito per elemento |
|---|---:|---:|---:|---|
| Transformers FP32 standard | 2.3401e-4 | 1.1015e-2 | 2.1648e-5 | PASS |
| Transformers FP64 standard | 2.7532e-5 | 1.6657e-3 | 2.7540e-6 | FAIL |
| Diagnostico FP64, RMSNorm senza cast FP32 | 2.7704e-6 | 4.2653e-5 | 2.0864e-6 | PASS |

Il massimo assoluto da solo non determina il risultato: ciascun elemento ha
anche il proprio termine relativo. FP64 standard: 270 occorrenze logits e 41
residuali oltre soglia, contando prefissi ripetuti fra scenari; nessun KV oltre
soglia. Prima violazione hidden al blocco indice **10** nel primo token.
Tutti gli argmax e la sequenza greedy coincidono anche nel caso fallito.

Diagnosi: `LlamaRMSNorm.forward` upstream forza input e riduzione in FP32 anche
quando parametri e input sono double. Già la prima `input_layernorm` sul token
504 differisce di `2.9156e-8` rispetto alla formula FP64 (576 elementi diversi):
[first-operator.json](../results/increment7/first-operator.json), riproduzione
`tools/diagnose7.py`. L'intervento isolato `--diagnostic-rms64` elimina soltanto
quel cast, lasciando il forward upstream e RoPE invariati: tutte le verifiche
passano alle soglie iniziali. RoPE continua a calcolare frequenze/trigonometria
in FP32 in upstream; il diagnostico non è un oracle interamente FP64.

Questo localizza una causa numerica dimostrata; non autorizza a sostituire il
riferimento standard col diagnostico. Rimane da risolvere il criterio FP64:
caratterizzare l'amplificazione degli arrotondamenti RMSNorm per layer e decidere
esplicitamente una politica numerica motivata oppure aggiungere un percorso
compatibile con quei cast, preservando il percorso FP64 attuale. Nessun aumento
a posteriori delle soglie in questo incremento. Il gate `report_increment7`
**esce con codice 1**. File o dipendenze mancanti producono errore, mai skip.

## Token, generazione e KV

`LlamaTokenizer` importa soltanto la libreria opzionale `tokenizers`, legge il
JSON ufficiale e interpreta BOS/EOS dalla configurazione della stessa revisione.
Per questo checkpoint BOS ed EOS sono ID 0; non si aggiungono automaticamente.
Verificata identità con AutoTokenizer su frasi inglesi, Unicode/newline e token
speciali, inclusa decodifica con rimozione degli speciali. L'interfaccia esclude
chat template: questo è il modello base, non un modello instruct.

Greedy deterministico: massimo logit, ID minore in caso di pareggio, arresto su
EOS o limite richiesto. La demo reinserisce i token scelti dal runtime:

- `The capital of France is` → ID `[504,3575,282,4649,314]`, generati
  `[260,3575,282]`, output `The capital of France is the capital of`.
- `Once upon a time` → ID `[6403,1980,253,655]`, generati
  `[28,665,436,253,1838]`, output `Once upon a time, there was a little`.

Nessuna delle due demo raggiunge EOS. La prima sequenza è confrontata col greedy
upstream per tre passi; la seconda è una demo riproducibile del runtime, non una
prova indipendente supplementare. Test dedicato verifica lo stop EOS.

Due banche KV prenotate: si copia lo stato pubblicato nella candidata, si calcola
l'intera chiamata e si pubblicano insieme bank/length/generation solo dopo successo.
Rollback conserva byte e descriptor precedenti; sessione in fail-stop dopo errore
o cancellazione, senza retry implicito. Le richieste già inviate vengono drenate;
nessuna deadline di arresto fisico promessa. Gli ID della chiamata sono copiati
in una tuple bounded prima delle letture che rilasciano il GIL.

Tutti gli array FP64 InWiMe coincidono esattamente fra originale e shard, sync e
pipeline. Completo/chunk/decode incrementale, ripetizione ed estensione causale
coincidono esattamente per logits/residuali/KV nel runtime. La parità col reference
permette invece le differenze numeriche sopra dichiarate.

## Memoria, I/O e prestazioni

Misure `measured-{sync,pipeline}-{1,2,3}.json`: **sei processi separati**, ordine
alternato, prefill 5 token e 3 decode, nessun import NumPy/torch/tokenizers.
Tetto di spazio virtuale `RLIMIT_AS=128 MiB`, accettazione RSS <128 MiB; il tetto
virtuale non viene presentato come un cap RSS. Capacità sessione **16**, tile
262144 elementi (512 KiB BF16, 2 MiB espansi), un buffer sync / due pipeline.
Metadati, output di tutte le posizioni e due banche KV sono nel piano. Bootstrap
fisico stimato 791309 byte; la vista logica aggiunge metadati conteggiati a parte.
Bootstrap precede il budget delle arene: RSS osservato comprende tutto il processo.

| Misura | Sync | Pipeline |
|---|---:|---:|
| Piano arene + riserve metadata/richieste | 15.242.224 B | 17.888.510 B |
| Picco RSS massimo, tre processi | 39.079.936 B | 41.766.912 B |
| Startup mediano, metadata + SHA + arene | 0,1855 s | 0,1879 s |
| Prefill 5 token mediano | 1,2488 s | 1,4671 s |
| Decode mediano, nove osservazioni | 0,2784 s/token | 0,3083 s/token |
| Residuo aritmetico mediano per chiamata | 0,0160 s | 0,0395 s |

Separati nei JSON: tempo SHA, GEMV, espansione BF16, regioni compute, attese,
letture, overlap e contatori per tensore. Residuo aritmetico = wall - compute - espansione - read_union + overlap
compute/lettura. Include coordinamento, bookkeeping e loop Python; in pipeline
non corregge eventuale overlap lettura/espansione. È una decomposizione
diagnostica, non un timer esclusivo del coordinatore o una stima esatta del suo
overhead. Le categorie temporali non vanno
sommate ingenuamente in pipeline. Il tempo SHA è nella startup, mai nel forward.

Letture logiche d'inferenza **2.152.249.344 byte** per otto token, cioè
269.031.168 byte/token con rilettura di tutte le proiezioni. Verifica SHA separata
269.060.552 byte/processo. Nessuna cache dei pesi applicativa. `/proc/self/io`
riporta `read_bytes=0` nei sei processi: **cache OS calda**, già favorita dalla SHA
iniziale. Non misura NAND, banda fisica NVMe o cold start. Nessun drop-cache.
La pipeline qui è più lenta; nessuna accelerazione attribuita al modello.
Tre ripetizioni sono un controllo limitato, non una campagna prestazionale.

CPU i7-12700H, 20 CPU logiche WSL, Linux 6.18.58.1-microsoft-standard-WSL2,
Python 3.14.4, ext4 virtuale `/dev/sdd`; alimentazione AC rilevata. Temperatura e
potenza CPU non misurate. NVML non accessibile nella query sandbox; il runtime
non inizializza GPU e non alloca memoria GPU. Nessuna misura globale VRAM valida.
L'oracolo FP64 residente usa circa **1,76 GB RSS**, in processo separato.
Il checkpoint entra nella RAM fisica: **nessuna prova oltre RAM fisica**, né
trasferimento automatico al modello pubblico delle proprietà grandi sintetiche
dell'incremento 5. Contesto 8192 è validato nella config, non provato end-to-end:
le prove reali arrivano a 8 posizioni con capacità prenotata 16.

## Test, revisione e riproduzione

**374 test superati in 35,97 s**, zero skip: 332 precedenti invariati e 42 nuovi.
[Log completo](../results/increment7/pytest-final.txt). I test nuovi coprono
config incompatibile, shape/tensore/tying, RoPE/GQA, ID e contesto invalidi,
budget, corruzione/troncamento, short read, I/O error e cancel durante prefill e
decode, byte di tutto il vecchio banco KV, nessuna pubblicazione parziale,
risultati scaduti, cleanup arene/FD, token input mutati da callback, EOS, import
senza dipendenze pesanti e fallimento esplicito se manca l'oracolo.
I test piccoli non sostituiscono il gate upstream separato, che resta fallito.

Revisione mirata eseguita sul codice nuovo, convenzioni GQA/RoPE, capture hook,
layout cache, SHA e semantica del gate. Corretti prima della chiusura: hook del
blocco finale compatibile con output tensor/tuple di Transformers; costruzione
del risultato prima del commit KV; limiti di cardinalità prima di costruire il
mapping; snapshot bounded degli ID; letture del repacker vincolate alle firme.
Un primo test negativo aveva capacità default 16 su fixture di contesto 8: errore
nel test, corretto senza cambiare il contratto runtime. Il log iniziale è conservato.
Nessun refactoring dei moduli consolidati; verifica whitespace e conservazione
file baseline riuscite. Nessuna revisione indipendente delegata dichiarata.

Dalla radice, con checkpoint già acquisito e venv come sopra:

```bash
make cpu-kernel
python3 -m pytest -q tests
# Per un nuovo ambiente isolato, senza installazione globale:
python3 -m venv --system-site-packages .venv-oracle7
.venv-oracle7/bin/python -m pip install -r requirements-oracle7.txt
# Download esplicito; prima senza --execute per audit, poi se autorizzato:
.venv-oracle7/bin/python -m tools.acquire_smollm7 checkpoints/smollm2-135m-93efa2f
# python ... -m tools.acquire_smollm7 checkpoints/smollm2-135m-93efa2f --execute
# Usare nuovi nomi per le catture/report: gli strumenti non li sovrascrivono.
OMP_NUM_THREADS=2 .venv-oracle7/bin/python -m tools.oracle7 \
  checkpoints/smollm2-135m-93efa2f --dtype float64 \
  --output artifacts/increment7/new-fp64.npz
.venv-oracle7/bin/python -m tools.verify_increment7 \
  checkpoints/smollm2-135m-93efa2f artifacts/increment7/new-fp64.npz \
  --output results/increment7/new-parity-fp64.json
# Ripetere oracle/verify con --dtype float32; verify accetta --mode pipeline.
# Diagnosi separata: oracle --dtype float64 --diagnostic-rms64.
python3 -m tools.repack7 checkpoints/smollm2-135m-93efa2f checkpoints/new-sharded --shards 3
python3 -m tools.benchmark_increment7 checkpoints/smollm2-135m-93efa2f \
  --mode sync --output results/increment7/new-benchmark-sync.json
.venv-oracle7/bin/python -m tools.generate7 checkpoints/smollm2-135m-93efa2f
python3 -m tools.report_increment7 results/increment7  # exit 1: esito parziale
```

Per il solo runtime a ID basta Python; per testo basta `requirements-tokenizer7.txt`.
I file completi NPZ con logits/hidden/KV sono locali in `artifacts/increment7/`,
esclusi da Git insieme ai pesi. Hash, dimensioni, versioni, input e schema sono
in [capture-manifest.json](../results/increment7/capture-manifest.json); le catture
si rigenerano col codice indipendente. Le evidenze pubblicabili contengono metriche
complete per layer, prompt pubblici e hash, senza pesi o dati privati.


## Risoluzione del gate numerico FP64 — 14 novembre 2026

**I criteri numericamente coerenti dell'incremento 7 sono soddisfatti.**
Il nuovo gate esce **0**, senza modifiche al forward di InWiMe, checkpoint,
librerie installate o tolleranze. Il gate precedente resta riproducibilmente
fallito. Le evidenze precedenti, incluso `results/increment7/summary.json`,
non sono state riscritte. Nuove prove in
[numerical-resolution](../results/increment7/numerical-resolution/README.md).

### Implementazione e dtype effettivi

Versioni effettive: Transformers **4.57.3**, PyTorch **2.9.1+cpu**. Percorso
`transformers.models.llama.modeling_llama`, attenzione **eager**, eval,
`inference_mode`, autocast CPU disabilitato. Nessun kernel RMSNorm alternativo
attivo: la replica per stadi coincide bit per bit col metodo effettivamente
eseguito. Sorgenti effettivi (RMSNorm, RoPE, inizializzazione frequenze e eager
attention), hash e versione in
[precision-audit.json](../results/increment7/numerical-resolution/precision-audit.json).

| Fase di RMSNorm standard con input e peso double | Dtype effettivo |
|---|---|
| Input checkpoint BF16 espanso, peso di scala | FP64 |
| `hidden_states.to(torch.float32)` | FP32 |
| Quadrato, media, aggiunta epsilon | FP32 |
| `rsqrt`, moltiplicazione di normalizzazione | FP32 |
| Ritorno a `input_dtype` | FP64, con arrotondamento precedente conservato |
| Moltiplicazione per il peso e output | FP64 |

Nel riferimento realmente FP64 tutte queste fasi sono FP64. InWiMe usa arene
`array('d')` con elementi da 8 byte e intermedi Python `float` con mantissa
53 bit. Una traccia delle variabili locali del metodo reale `_norm` conferma
`squared` e `denominator` di questo tipo; non è stata sostituita la sua formula.
Il valore è calcolato come `x / sqrt(mean(x*x)+eps) * weight`. L'oracolo usa
moltiplicazione per `rsqrt`: l'ordine aritmetico può differire senza cambiare
l'equazione. Nessuna uguaglianza bit per bit viene richiesta fra motori.

**Ulteriori politiche FP32 dimostrate:**

- RoPE inizializza `inv_freq` in FP32; il forward forza frequenze, posizioni,
  prodotto e trigonometria in FP32, poi restituisce cos/sin nel dtype dell'input.
  Promuovere il buffer a double non recupera le cifre perse all'inizializzazione.
- Eager attention esegue `softmax(..., dtype=torch.float32).to(query.dtype)`.
  Score e output possono essere double, mentre le probabilità sono state FP32.
- Embedding, proiezioni, residui, SiLU, KV e logits operano in double nel modello
  convertito; non sono stati trovati altri output floating di precisione ridotta
  nell'audit dei percorsi effettivamente eseguiti. Posizioni e maschere booleane
  restano correttamente intere/booleane. Non è una prova su tutti i backend HF.

L'audit dispatcher delle catture complete registra nello standard double **671**
quadrati/medie/rsqrt FP32, **330** softmax con output FP32 e **11** calcoli RoPE
FP32. Nel percorso `full64` non registra **alcun output tensoriale floating**
FP16/BF16/FP32, inclusi chunk e decode con cache. I dettagli sono nei metadata
[capture-manifest.json](../results/increment7/numerical-resolution/capture-manifest.json).

### Isolamento matematico e primo scarto

Sul vero embedding del token 504, con i pesi originali del primo RMSNorm:

| Confronto isolato | Max abs |
|---|---:|
| RMSNorm standard vs formula PyTorch FP64 | 2.91558e-8 |
| InWiMe vs formula PyTorch FP64 | 5.55112e-17 |
| InWiMe vs formula indipendente Decimal a 80 cifre | 2.77556e-17 |

La sequenza standard replicata per stadi, inclusi dtype e valori, coincide
esattamente col metodo installato. L'input di questo primo norm proviene da BF16
ed è già rappresentabile in FP32: qui l'effetto non è perdita nel solo cast
iniziale dell'input, bensì l'aritmetica FP32 che quel cast impone a quadrato,
riduzione, rsqrt e normalizzazione. Il replay con lo stesso input riproduce
esattamente l'output standard, separando la causa locale da errori propagati.

Caso matematico indipendente: `x=[3,4,-3,-4]`, scala `[1,2,3,4]`, epsilon `0.5`.
L'output è `[3,8,-9,-16]/sqrt(13)`, calcolato con Decimal 80 cifre. InWiMe e
PyTorch FP64 coincidono nel caso; differiscono dal valore Decimal arrotondato
al massimo di **4.4409e-16**. Lo standard FP32 interno differisce di **8.7446e-8**.
Il probe InWiMe richiama il vero metodo `_norm`, con soli pesi di scala forniti
da uno stream minimo; non introduce una seconda formula InWiMe.

Il tracciamento dei confini degli operatori, sul primo token del modello reale,
mostra embedding identico e primo output non esatto in **`layers.0.norm1`**.
Il primo superamento delle soglie storiche avviene nel residuo di
**`layers.10.residual`** (indice zero-based). Non confondere la prima differenza
aritmetica col primo superamento della soglia. Tutti i confini tracciati passano
contro `full64`. Il trace comprende norm, proiezioni e residui; non pretende
identità bit per bit o una decomposizione causale locale di ogni errore successivo.

Interventi progressivi, una politica per volta, tutti sullo stesso checkpoint:

| Oracle con parametri double | Max abs logits | Max abs hidden | Max abs KV |
|---|---:|---:|---:|
| Standard invariato | 2.75319e-5 | 1.66569e-3 | 2.75398e-6 |
| Solo RMSNorm FP64 | 2.77033e-6 | 4.26528e-5 | 2.08635e-6 |
| RMSNorm e RoPE FP64 | 1.93581e-6 | 3.67172e-5 | 8.46134e-7 |
| RMSNorm, RoPE e softmax FP64 | 3.37508e-13 | 3.63798e-11 | 1.20792e-13 |

Sul **singolo token**, rendere FP64 solo RMSNorm riduce il massimo errore logits
da **2.75319e-5 a 2.51299e-13**: posizione zero dà RoPE identità e softmax su una
sola chiave vale 1. Per più token gli scarti rimanenti non sono attribuiti a
RMSNorm: RoPE e softmax hanno contributi misurati separatamente. Un test diretto
su softmax con gli stessi Q/K/V double produce probabilità standard differenti
da quelle FP64. Le differenze di rete sono non lineari e non additive; i delta
fra interventi sono registrati in
[attribution.json](../results/increment7/numerical-resolution/attribution.json).

### Tre contratti distinti e modifica esplicita del criterio

1. **Transformers standard FP32:** criterio di parità pratica, superato.
2. **PyTorch realmente FP64:** medesimo grafo upstream con RMSNorm, RoPE e
   softmax FP64 nel solo processo oracolo, criterio di correttezza matematica,
   superato su tutti i logits, hidden, KV, prefill/chunk e tre decode.
3. **Transformers standard con parametri FP64:** confronto diagnostico obbligatorio
   col suo comportamento mixed precision. Il verdetto grezzo rimane **FAIL**,
   con le stesse 270 occorrenze logits e 41 hidden oltre soglia, zero KV oltre
   soglia e tutti gli argmax identici. Non è uno skip né una nuova tolleranza.

Le tolleranze sono esattamente quelle iniziali: FP64 `(2e-5,2e-6)`, FP32
`(2e-3,2e-4)` nella regola `abs <= atol + rtol*abs(reference)`.
Il nuovo gate richiede che tutti e tre i confronti siano realmente eseguiti,
che FP32 e FP64 effettivo passino, che il riferimento FP64 non produca tensor
floating a precisione inferiore e che scenari, causalità/KV e argmax siano
verificati. Rifiuta un diagnostico con solo RMSNorm modificato presentato come
`full64`, soglie cambiate o confronti mancanti. Se FP64 effettivo fallisce,
il gate rimane fallito. Il diagnostico standard viene sempre riportato con
il proprio verdetto grezzo, senza riscriverlo come PASS.

**Cambio del criterio:** da `stock FP32 AND stock FP64` a
`stock FP32 AND pure FP64`, con `stock FP64` mantenuto come diagnostico
obbligatorio. La precedente congiunzione richiedeva erroneamente parità alle
stesse soglie fra contratti aritmetici diversi. La correzione è supportata da
stadi isolati, riferimento indipendente Decimal, ablation e audit dei dtype;
non deriva semplicemente dal risultato finale. Il vecchio gate e le evidenze
rimangono disponibili. I nuovi stock FP32/FP64 riproducono **bit per bit tutti
gli array delle catture storiche**.

Implementazione dell'intervento: `tools/oracle_precision7.py`, nessun import
InWiMe. Il context manager associa metodi FP64 alle sole istanze RMSNorm/RoPE
del modello residente e sostituisce temporaneamente la funzione eager attention
nel processo oracolo. Ripristina tutti gli attributi anche in eccezione. Per RoPE
ricrea `inv_freq` dalla configurazione direttamente in double; nessun buffer
originale né file persistente viene cambiato. Pesi, maschere causali, GQA,
orientamento e ordine dei blocchi restano quelli upstream. I pesi sono caricati
prima in BF16 originale e promossi soltanto in RAM, come nel riferimento iniziale.

### Chiusura, verifiche e riproduzione

Nuova suite generale: **382 passed in 34,85 s**, zero skip, inclusi i 374 test
precedenti e 8 test del criterio numerico. Ulteriori **5 test dell'oracolo** nel
venv torch: precisione RMSNorm e softmax, forward/decode senza output floating
inferiori, ripristino dopo errore senza modifica dei pesi o del comportamento
stock e rifiuto del falso profilo FP64. Nessuna dipendenza torch aggiunta alla
suite CPU di base. Il comando oracolo dedicato deve essere eseguito esplicitamente
ed è stato eseguito, non saltato. Nuovi confronti completi anche `full64` in
pipeline e sul checkpoint multi-shard: medesimi errori massimi del sync originale.

Runtime, checkpoint, sorgenti delle librerie installate e tutte le evidenze
storiche sono verificati via SHA256 prima/dopo:
[preservation-after.json](../results/increment7/numerical-resolution/preservation-after.json).
Revisione mirata di modifiche, interventi, confronti, gate e prove eseguita.
Restano limiti del primo adapter: contesti brevi, profilo Llama limitato,
CPU-only, nessuna prova oltre RAM fisica, nessun test generalizzato su altri
backend/versioni/configurazioni. Le misure prestazionali restano quelle della
prima acquisizione; non sono state rieseguite né ottimizzate.

Nuovi comandi (usare nomi di output nuovi, nessuna sovrascrittura):

```bash
.venv-oracle7/bin/python -m tools.numerics7 checkpoints/smollm2-135m-93efa2f \
  --output /tmp/inwime-precision-audit-new.json
.venv-oracle7/bin/python -m tools.oracle7 checkpoints/smollm2-135m-93efa2f \
  --dtype float64 --precision-profile full64 --output artifacts/increment7/new-pure64.npz
.venv-oracle7/bin/python -m tools.verify_increment7 checkpoints/smollm2-135m-93efa2f \
  artifacts/increment7/new-pure64.npz --output /tmp/inwime-parity-pure64-new.json
# Contratti standard: --precision-profile stock e --dtype float32/float64.
# Ablation: --precision-profile rms64 oppure rms_rope64, sempre --dtype float64.
.venv-oracle7/bin/python -m tools.check_oracle_precision7
python3 -m pytest -q tests
python3 -m tools.report_increment7 results/increment7/numerical-resolution \
  --resolve-numerical-contracts --output /tmp/inwime-resolved-gate-new.json
# Riproduce la decisione storica fallita, scrivendo un nuovo report:
python3 -m tools.report_increment7 results/increment7 --output /tmp/inwime-old-gate-new.json
```

Il file conclusivo è
[final-gate.json](../results/increment7/numerical-resolution/final-gate.json):
**PASS / exit 0**, con FAIL diagnostico standard FP64 esplicitamente conservato.
Nessun commit, push o incremento 8.
