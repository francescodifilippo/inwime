# InWiMe — Incremento 1: riferimento numerico CPU

**Stato:** primo incremento del riferimento residente. L'alternativa A è approvata come ipotesi iniziale falsificabile; questo incremento definisce matematica, checkpoint e verifiche, senza implementare il controllo operativo del runtime.

## Esecuzione

Dalla radice del repository, usare un ambiente isolato con le dipendenze di test fissate in `requirements-reference.txt`:

```bash
python3 -m venv /tmp/inwime-reference-inc1-venv-new
/tmp/inwime-reference-inc1-venv-new/bin/python -m pip install -r requirements-reference.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/inwime-reference-inc1-venv-new/bin/python -m pytest -q -s tests/test_reference_decoder.py
```

In un ambiente che dispone già delle dipendenze:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -s tests/test_reference_decoder.py
```

Non serve compilare il benchmark C++ e non si usa `make test`, che appartiene ai benchmark precedenti. Le prove generano fixture piccole in directory temporanee gestite da pytest. I checkpoint conservati sono in `fixtures/reference_decoder/dense` e `fixtures/reference_decoder/moe`.

NumPy è scelto per il riferimento vettoriale residente e il confronto con un oracolo scalare basato sulla libreria standard. Questa scelta non prescrive linguaggio, librerie o precisione del runtime InWiMe. PyTorch non è necessario a questi confronti; non viene installato. Le dipendenze di test fissate non sono dipendenze architetturali del runtime.

Per rigenerare checkpoint in **nuove** directory:

```bash
python3 -m inwime_reference.checkpoint --variant dense --output /tmp/inwime-reference-dense-new
python3 -m inwime_reference.checkpoint --variant moe --output /tmp/inwime-reference-moe-new
```

Il generatore rifiuta directory preesistenti. Il seed predefinito è `20261107`. Non usa rete, CUDA, dati dei benchmark o percorsi dei dispositivi del Katana.

## Modello definito per la fixture

È un decoder sintetico non addestrato, con vocabolario di 31 ID interi, larghezza 16, due blocchi, due teste di attenzione da 8 elementi, hidden feed-forward 24 e contesto massimo 32. Non comprende tokenizer, sampling o compatibilità con checkpoint di modelli esterni.

Gli input sono la somma di embedding del token ed embedding posizionale appreso alla posizione assoluta. Ogni blocco usa pre-normalizzazione RMSNorm con epsilon `1e-5`, attenzione multi-head causale e due connessioni residuali. Q, K, V e proiezione d'uscita sono matrici distinte senza bias. Gli score sono `Q K^T / sqrt(8)`; la query alla posizione assoluta `t` vede esclusivamente chiavi alle posizioni `0..t`.

La parte feed-forward usa SwiGLU:

```text
FFN(x) = W_down [ SiLU(W_gate x) * (W_up x) ]
SiLU(z) = z / (1 + exp(-z))
```

Nella variante densa ogni blocco ha una FFN. Nella variante MoE ogni blocco ha tre FFN esperte e un router lineare senza bias. Si scelgono i due logits più grandi; in caso di pareggio precede l'indice esperto minore. I due pesi sono la softmax dei soli logits selezionati. L'output è la somma pesata delle due FFN selezionate: nessun limite di capacità, scarto di token, rumore o sostituzione di esperti.

Dopo il secondo blocco si applicano RMSNorm finale e una proiezione non condivisa verso 31 logits. Nessun dropout. Queste sono le specifiche del modello di test, non scelte imposte ai futuri adapter.

## Checkpoint BF16 originale

`checkpoint.py` genera direttamente parole BF16 finite con un PRNG intero xorshift32 specificato nel codice. Per le matrici vengono generati segno, esponente e mantissa BF16; i pesi di normalizzazione sono esattamente 1. Non esiste un checkpoint FP32 sorgente che venga quantizzato. Le fixture sono originali BF16 sintetiche, non pesi addestrati o convertiti da un modello pubblico.

La codifica ha segno al bit 15, esponente a 8 bit nei bit 14..7 con bias 127 e frazione a 7 bit nei bit 6..0. Per esponente `e` fra 1 e 254 il valore è `(-1)^s * (128 + f) * 2^(e - 134)`; per `e=0` è `(-1)^s * f * 2^-133`, inclusi gli zeri con segno. L'esponente 255 non è ammesso dal loader di questa fixture.

**Arrotondamento del checkpoint:** non applicabile alla generazione nativa per codici interi. Non si troncano valori FP32/FP64, né si applica un arrotondamento nearest-even: non esistono valori sorgente a precisione superiore da ridurre. L'espansione BF16 → FP32 → FP64 è esatta per tutti i codici finiti. La suite verifica tutti i 65.280 codici finiti tramite una formula scalare indipendente e ricostruisce esattamente i bit dopo l'espansione, compresi segno dello zero e subnormali. Questo non verifica un convertitore FP32 → BF16 con una politica di arrotondamento: tale convertitore non fa parte dell'incremento.

Ogni variante ha `manifest.json` e due file `weights-0000N-of-00002.bf16.bin`. Formato interno `inwime-reference-bf16-v1`, parole little-endian, array contigui in ordine C, matrici lineari memorizzate come `[output, input]`. Il manifest dichiara configurazione, variante, seed, generatore, dtype, shape, shard, offset, lunghezza e SHA256 degli shard. Nomi ordinati e assegnazione alternata dei tensori producono gli stessi byte a parità di input.

Il loader verifica lo schema della fixture, dimensioni e hash prima di interpretare i pesi. Carica integralmente i piccoli shard ed espande BF16 in FP64 senza perdita dei valori originali. È un loader residente del formato di test, non un reader out-of-core o un parser safetensors generale. Gli hash verificano integrità rispetto al manifest fornito, non l'autenticità del manifest o l'immutabilità amministrativa del filesystem.

## Precisione e indipendenza

**Politica aggiornata prima di eseguire la suite rivista**, in seguito alla precisazione numerica dell'utente. Sostituisce la soglia preliminare uniforme `1e-10`; le costanti effettive sono in `tests/test_reference_decoder.py`. Posto `u = 2^-52`, epsilon FP64:

| Quantità | Tolleranza assoluta | Tolleranza relativa |
|---|---:|---:|
| RMSNorm, softmax, SwiGLU su casi analitici | `16u` ≈ `3.55e-15` | `32u` ≈ `7.11e-15` |
| Probabilità del router | `32u` ≈ `7.11e-15` | `64u` ≈ `1.42e-14` |
| Equivalenza prefill/chunk/decode e causalità | `128u` ≈ `2.84e-14` | `256u` ≈ `5.68e-14` |
| KV contro oracolo scalare | `512u` ≈ `1.14e-13` | `512u` ≈ `1.14e-13` |
| Output dei blocchi contro oracolo scalare | `1024u` ≈ `2.27e-13` | `1024u` ≈ `2.27e-13` |
| Logits contro oracolo scalare | `2048u` ≈ `4.55e-13` | `2048u` ≈ `4.55e-13` |
| Decoder controllato con soluzione analitica | `512u` ≈ `1.14e-13` | `1024u` ≈ `2.27e-13` |
| Indici esperti, byte del checkpoint, lunghezze e vecchie regioni KV | Uguaglianza esatta | Uguaglianza esatta |

Le soglie degli operatori sono piccole perché riguardano test analitici con poche operazioni. Il confronto fra partizioni usa lo stesso algoritmo con ordini di riduzione eventualmente diversi. Le soglie dell'oracolo crescono da KV a intermedi e logits per l'accumulo attraverso proiezioni, normalizzazioni, non linearità e due blocchi residuali. Il termine assoluto protegge i confronti vicino a zero. Sono margini di accettazione prefissati per questa fixture piccola (riduzioni di larghezza 16/24 e contesto fino a 32), non limiti rigorosi dell'errore per ogni input o piattaforma. Non sono stati scelti per allargare una soglia dopo un test fallito.

Il riferimento NumPy usa pesi e operazioni FP64, con softmax stabilizzata. BF16 descrive i pesi persistiti; non si dichiara di eseguire gli operatori in aritmetica BF16. Questa verifica FP64 non dimostra correttezza FP32 o mixed precision. Un futuro percorso di quel tipo richiederà tolleranze motivate per accumulo, lunghezza delle riduzioni, operatore e output, prima delle sue prove.

`tests/reference_oracle.py` implementa il forward con liste Python, `math.fsum`, `math.exp` e `math.sqrt`. Non importa esecuzione, attenzione, FFN, routing o gestione KV dal riferimento. Calcola tutte le posizioni di ogni prefisso e non riusa il KV del decoder. Condivide soltanto configurazione e valori dei pesi; l'espansione BF16 è verificata anche mediante decodifica scalare con `struct`.

Per ridurre il rischio di replicare la stessa interpretazione sbagliata in due programmi, ulteriori test usano soluzioni analitiche: RMSNorm su valori a modulo costante, softmax con rapporti noti, SwiGLU con una sola componente attiva e un decoder controllato con `Q=K=0` e `V=O=I` nel primo blocco. In quel caso l'attenzione deve essere la media uniforme dei prefissi normalizzati, calcolata esplicitamente per tre posizioni, senza riusare l'algoritmo di mascheramento o attenzione. Anche prefill e decode sono confrontati con quella soluzione. Il routing ha test manuali di pareggi, ordinamento, probabilità e mancato contributo di un esperto escluso.

L'indipendenza non elimina ogni possibile errore condiviso nella specifica del modello sintetico. Questi confronti non sono una certificazione di un'architettura esterna; rendono verificabili separatamente le primitive, l'attenzione causale e la consistenza temporale della fixture.

La futura implementazione tiled/out-of-core non dovrà importare gli helper di esecuzione o aggiornamento dello stato da `inwime_reference.decoder`. I file del checkpoint e la specifica matematica possono essere input comuni; il forward deve restare indipendente. Non esiste ancora un confronto fra riferimento e runtime futuro.

## Stato KV e proprietà verificate

`prefill(tokens, state=None)` calcola un prefisso intero oppure un nuovo chunk dopo il KV fornito. `decode(token, state)` richiede un solo token e uno stato precedente. Ogni layer conserva K e V in forma `[heads, tokens, head_width]`; il KV è identificato dal checkpoint e dalla lunghezza già elaborata. Le posizioni dei nuovi token partono da quella lunghezza, anche nel chunked prefill.

Il riferimento restituisce un nuovo stato senza modificare gli array precedenti. Gli array restituiti sono protetti contro le normali scritture NumPy; non è una barriera di sicurezza contro chi modifica deliberatamente le strutture Python. Questo modello a valori serve come reference numerica, non implementa transazioni o recupero di uno stato in-place nel futuro runtime.

La suite copre entrambe le varianti e verifica:

- prefill e ciascun passo di decode contro l'oracolo scalare, inclusi logits, output dei due blocchi, KV e routing;
- equivalenza fra prefisso completo, chunked prefill e decode token per token;
- assenza di influenza dei token futuri sul prefisso causale;
- append esatto del vecchio KV, crescita su entrambi i blocchi e assenza di alias con lo stato precedente;
- scelta top-2, pareggi, normalizzazione e mancato contributo di un esperto non selezionato;
- contesto e token invalidi, incompatibilità del checkpoint o della forma KV, senza modifica dello stato ricevuto;
- riproducibilità byte per byte, rifiuto della sovrascrittura e rilevazione di corruzione, troncamento o offset alterati;
- import del riferimento senza moduli CUDA, PyTorch o runtime futuro.

## Limiti e requisiti futuri

Le fixture e tutti i pesi entrano comodamente nella RAM. Questo incremento non dimostra streaming, limiti di memoria del runtime, capacità oltre RAM/VRAM, cancellazioni asincrone, progresso sotto pressione o isolamento multi-sessione. Non contiene coordinatore, cache dei pesi, scheduler I/O, predittore di prefetch o distribuzione. Il KV di attenzione è distinto dalla futura cache di residenza dei pesi.

Nessun supporto a un LLM ufficiale, confronto con logits upstream, validazione GPU o prestazione token/s è dimostrato. La memoria del riferimento cresce con il contesto e il checkpoint residente; non costituisce un budget operativo per il motore out-of-core.

**Requisito futuro conservato:** confrontare inferenza distribuita e repliche indipendenti sui server. Ridurre la latenza di una singola sessione e aumentare il throughput aggregato sono obiettivi diversi; la valutazione dovrà riportare entrambi a parità di risorse e carico. Questo incremento non sceglie né implementa una strategia distribuita.

## Risultati acquisiti

Suite rivista eseguita il 7 novembre 2026 nell'ambiente isolato `/tmp/inwime-reference-inc1-venv`, con CPython 3.14.4, NumPy 2.3.5 e pytest 9.0.2 su Linux x86-64. Installate esclusivamente nel venv anche le dipendenze di pytest: iniconfig 2.3.1, packaging 26.3, pluggy 1.7.0 e Pygments 2.21.0. Il venv occupa circa 97 MiB ed è lasciato disponibile per ripetere le prove. `requirements-reference.txt` fissa tutte queste versioni; nessun pacchetto di sistema o driver modificato. Le versioni sono quelle verificate per il riferimento, senza prescrivere l'ambiente del runtime futuro.

Comando effettivo:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -v /tmp/inwime-reference-inc1-venv/bin/python -m pytest -q -s tests/test_reference_decoder.py
```

**24 test superati**, nessun errore. pytest ha riportato 0,30 s: durata della suite di correttezza, non misura di prestazione dell'inferenza. Sono incluse le due varianti e le verifiche analitiche e BF16 aggiunte dopo la precisazione numerica. Le tolleranze della politica aggiornata sono state fissate prima di questa esecuzione.

| Quantità osservata | Massimo errore assoluto nella suite |
|---|---:|
| Logits | `4.066191827689636e-15` |
| Output dei blocchi | `3.3306690738754696e-15` |
| KV | `1.519617764955683e-15` |
| Probabilità router | `3.3306690738754696e-16` |
| Softmax nei casi analitici | `2.220446049250313e-16` |
| RMSNorm e SwiGLU nei casi analitici | `0` |

I massimi aggregano i confronti della suite; ogni confronto applica la propria politica, non una soglia globale derivata dal massimo. Codici BF16 finiti, indici selezionati, byte riproducibili e prefissi KV conservati soddisfano i rispettivi confronti esatti. Non sono state allargate tolleranze per ottenere il passaggio.

| Fixture conservata | Payload BF16 dei due shard | Manifest | Totale file |
|---|---:|---:|---:|
| Densa | 11.872 byte | 4.872 byte | 16.744 byte |
| MoE | 21.280 byte | 7.736 byte | 29.016 byte |
| Totale | 33.152 byte | 12.608 byte | **45.760 byte** |

Queste dimensioni sono lunghezze logiche dei file, non blocchi fisicamente allocati. Ogni caricamento completo legge logicamente il payload e il manifest indicati per la propria variante; i test ripetono caricamenti e generazioni temporanee. I checksum degli shard sono nei rispettivi manifest e la suite confronta anche le fixture conservate con quelle rigenerate.

`/usr/bin/time -v` ha riportato picco RSS di 61.700 KiB (circa 60,25 MiB), zero swap e zero unità di filesystem input/output nel conteggio del processo. Questi ultimi valori riflettono la contabilizzazione del sistema in una prova piccola con cache e file temporanei: non significano assenza di letture logiche. Non sono stati acquisiti contatori dei dispositivi né misure cold; il volume fisico letto dal disco non è determinato. Il picco RSS comprende interprete, NumPy e pytest, non soltanto pesi e KV. Il riferimento non inizializza CUDA; la VRAM non è stata misurata.

Alla chiusura dell'implementazione dell'incremento 1, il controllo Git non mostrava modifiche a file già tracciati, inclusi i benchmark precedenti. Nel corso della sessione HEAD era stato aggiornato esternamente da `727e8be` a `c7c37a5` (`checkpoint`), che registra documenti e artefatti già presenti prima dell'incremento. In quella fase non erano stati eseguiti comandi di commit, stage o push e i file dell'incremento erano stati lasciati non tracciati per la revisione.

## Verifica prima del consolidamento Git

Su autorizzazione successiva dell'utente, la stessa suite è stata rieseguita nel medesimo venv isolato: **24 test superati in 0,35 s**, con gli stessi massimi errori assoluti sopra riportati. Le versioni installate coincidono con tutte le dipendenze fissate; `pip check` non segnala incompatibilità. I comandi documentati usano la radice del repository e l'eseguibile dell'ambiente isolato, senza richiedere build o dataset dei benchmark.

Il commit esterno `c7c37a5fd0d11365fd6a2ddd18ba08880f1d92c6` aggiunge 90 file: 77 artefatti M1a, i due documenti architetturali, 10 accessori V2 preesistenti e `codex.sh`. Non modifica file precedenti. M1a e architettura sono quindi già versionati e non richiedono duplicazioni o riscritture della cronologia. Sono stati verificati i 76 hash di `artifact-sha256.json` e i 450 hash di `preserved-files.json`, senza differenze.

Il consolidamento dell'incremento comprende solo i 13 file riconosciuti di riferimento, test, dipendenze, documentazione e fixture. Venv, cache, build, dataset e file temporanei restano esclusi. Il controllo dei nuovi contenuti non ha rilevato credenziali o dati personali; gli inventari di macchina preesistenti in M1a rimangono nel commit storico e non sono duplicati. Nessun push o avvio dell'incremento 2 fa parte del consolidamento.
