# Revisione finale dell'incremento 8

14 novembre 2026. **Pronto per il commit delle correzioni di revisione.**
Tre rilievi P2 riprodotti e corretti nel gate delle evidenze; nessun P0/P1 e nessun
rilievo concreto residuo. Runtime, adapter, kernel e matematica invariati.
[Gate A–F rieseguito](../results/increment8/review/gate-reviewed.json): PASS.

Perimetro: confronto con `ed166eddaa37ad5db6239d405bcbe8f7f8cdba49`, codice e
artefatti reali del working tree, non soltanto dichiarazioni del rapporto.
All'inizio HEAD era `42a1a95` e gli strumenti erano non tracciati; durante la
revisione è comparso il commit `7afcd6b` che li include. Non sono stati eseguiti
commit, staging, reset, fetch o push da questa revisione. Correzioni applicate
soltanto a `tools/report_increment8.py` e `tests/test_increment8.py`; nuove evidenze
in `results/increment8/review/`. Gli snapshot originali non sono riscritti.

## Rilievi concreti: asse Standards

**P2 — regressioni mancanti accettate dal gate aggregato.** Nel codice prima della
correzione (`7afcd6b:tools/report_increment8.py:78–84`), F leggeva i booleani di
`regression7-gate.json` e `preservation-after.json` senza richiedere i cinque report
numerici. Riproduzione su copia temporanea: eliminando i report individuali,
lasciando il riepilogo, A–F rimaneva PASS. Gli artefatti effettivi erano completi;
il difetto riguardava l'ammissione di un insieme di evidenze incompleto.

Correzione minima: verifica di SHA256/dimensioni degli input originali rispetto al
manifest e ricostruzione di `numerical_gate` dai cinque confronti, con uguaglianza
al riepilogo conservato. La stessa riproduzione ora fallisce esplicitamente.

## Rilievi concreti: asse Spec

**P2 — contratto oracle non verificato nell'aggregazione.**
`7afcd6b:tools/report_increment8.py:9–26,69–70` accettava report con contratto,
dtype, profilo o revisione alterati se conteggi e violazioni rimanevano favorevoli.
Il comparatore delle catture aveva già i controlli corretti; l'aggregatore li
perdeva. Correzione: richiedere contratto Transformers FP32 stock/eager, revisione
immutabile e coerenza del digest NPZ, oltre a conteggi completi, soglie e argmax.

**P2 — insieme vuoto di invarianti accettato.**
`7afcd6b:tools/report_increment8.py:71–72` usava `all()` senza controllare la
completezza: `{"passed":true,"checks":{}}` produceva ancora PASS. Correzione:
richiedere gli otto controlli previsti, sei causali logits/hidden/KV più pipeline e
shard, tutti esplicitamente veri. Vuoto, singolo controllo causale mancante e
confronto di formato mancante sono rifiutati.

Le prove red registrano **9 fallimenti e 13 test precedenti passati** prima della
correzione. Le nuove regressioni sono 13: sei contratti incoerenti, tre insiemi di
evidenze corrotti/incompleti, quattro omissioni degli invarianti. La revisione
indipendente sui due assi ha rieseguito le riproduzioni dopo le correzioni e non
ha trovato ulteriori rilievi concreti.

## Esito dei cinque controlli richiesti

**1. Limite e attribuzione della memoria: soddisfatti.**
`benchmark_increment8.run` imposta soft e hard RLIMIT_AS prima di tokenizer,
catalogo, SHA e costruzione del runtime. Gli import iniziali non aprono checkpoint
né caricano pesi. Il forward avviene nello stesso processo; le metriche interne
usano `/proc/self` e `RUSAGE_SELF`, il supervisore usa `/proc/<Popen.pid>`.

Due esecuzioni complete aggiuntive, sync e pipeline, hanno ripetuto prefill 5 e
3 decode a **134.217.728 B di AS**. Il verificatore di revisione registra PID del
worker e padre nello stesso namespace, start ticks e limiti osservati dal padre;
controlla il cap a ogni apertura Python del checkpoint. Le aperture Rust del
tokenizer seguono anch'esse `setrlimit` nel flusso verificato staticamente.
Il padre vede un breve avvio `unlimited` prima dell'applicazione del cap, senza
aperture di pesi: non viene descritto come processo limitato dall'istante di exec.

| Prova aggiuntiva | Picco RSS/VmHWM esterno | Picco VM esterno | VmSwap osservato |
|---|---:|---:|---:|
| Sync 128 MiB | 91.815.936 B | 103.882.752 B | 0 B |
| Pipeline 128 MiB | 92.053.504 B | 112.295.936 B | 0 B |

Tutti gli hash logits/residui/KV coincidono con la prima acquisizione. Queste prove
strumentate non entrano nelle mediane prestazionali originali. RLIMIT_AS controlla
lo spazio virtuale; RSS misura pagine residenti del processo; la page cache globale
rimane esterna a entrambi. I forward originali con `read_bytes=0` sono caldi e non
misurano banda del dispositivo NVMe.

**2. Copertura dei pesi: soddisfatta.** Parser indipendente stdlib dell'header e
hash delle regioni, senza import del runtime: **218 tensori BF16**, payload unico
**3.422.752.768 B**. Dalla configurazione e dalle forme derivano indipendentemente
**1.711.276.032 coefficienti GEMV/token** e **3.422.756.864 B logici/token**.
Ogni contatore per tensore di tutte le acquisizioni decoder complete, incluse le
due nuove, coincide esattamente. L'head usa interamente l'embedding condiviso:
l'alias non duplica i byte unici, la lettura della riga input è invece aggiuntiva.

Ispezionati loop su tutti i 24 blocchi, proiezioni, RoPE, attenzione, normalizzazioni,
SwiGLU, residui e kernel C: nessun salto o operatore fittizio. Il generatore
sintetico presente nella libreria non interviene nel percorso del checkpoint reale.
Le tile sono convertite esattamente BF16→FP64 e ogni coefficiente viene consumato.

**3. Oracolo e confronto completo: soddisfatti per il livello FP32 dichiarato.**
Audit aggiuntivo sul modello reale caricato dal nuovo loader: confronto di tutti i
218 parametri con i tensori originali promossi a FP32, uguaglianza esatta, head
fisicamente condivisa. Buffer, dtype FP32 e output RoPE per otto posizioni uguali
al costruttore upstream; forward rotary/RMSNorm/decoder e attenzione sono stock.
Nessun import o forward InWiMe nell'audit oracle.

Questo audit più oneroso del solo forward ha picco RSS esterno **7.789.563.904 B**
e VmSwap osservato massimo **27.799.552 B**; minimo MemAvailable **1.476.874.240 B**,
nessuna soglia del supervisore superata. È un processo separato, terminato dopo le
prove runtime: questo swap non viene attribuito al runtime e non sostituisce i
contatori zero dei precedenti oracoli acquisiti. Nessun cambiamento allo swap.

Riletti gli NPZ e tutte le catture raw con verifica SHA; ricalcolati tutti i logits,
i residui dei 24 blocchi e il KV completo per prefill, tre decode, extended e chunk,
in originale/pipeline/shard. Tutte le metriche coincidono esattamente con i report
originali, zero violazioni alle soglie predefinite. Ricalcolati anche gli invarianti
nei byte. L'oracolo 1.7B rimane FP32; nessuna pretesa di riferimento realmente FP64
1.7B, i cui soli parametri richiederebbero 13,69 GB.

**4. Regressioni: soddisfatte.** Hash dei **1.256 file preesistenti**, escluso il
README intenzionalmente aggiornato nell'implementazione, invariati; diff vuoto
rispetto a `ed166ed` per runtime/kernel/fixture e strumenti numerici precedenti.
Il gate 7 è ricostruito dai cinque report completi: stock FP32 e FP64 effettivo
(sync/pipeline/shard) PASS. Standard FP64 resta **FAIL**, 270 violazioni logits,
41 hidden, zero KV, argmax uguali. Nessuna tolleranza o evidenza storica modificata.

**408 test generali PASS**, zero skip, più **6 test oracle PASS** separati
(1 loader e 5 profilo numerico). I 395 test della precedente acquisizione sono
preservati, con 13 regressioni di revisione aggiunte. Le sei nuove iniezioni reali
I/O/short/cancel su multi-shard sono PASS in sync/pipeline: KV candidato già
modificato, vecchio banco e descriptor preservati, drain e cleanup confermati;
budget sotto il piano rifiutato prima del payload. Ripetuta anche la prova 64 MiB:
MemoryError/exit 1 controllato, senza artefatto di successo.

**5. Integrità e portata: soddisfatte.** Prima delle correzioni corrispondevano tutti
i **97 fingerprint** del manifest e tutti gli hash delle catture locali. Ricalcolati
SHA ufficiale del checkpoint, SHA di ogni shard e digest di ognuno dei 218 tensori
prima/dopo il repack: BF16 identico. Verificato anche il digest originale 135M.
Nessun checkpoint o file di cattura tracciato; esclusioni Git operative. L'unico
file tracciato >1 MiB è una traccia V2 preesistente da circa 1,8 MB, non un peso.

Il manifest originale resta lo snapshot dell'implementazione; contiene quindi i
fingerprint precedenti dei due sorgenti corretti. Il manifest della revisione lega
codice corretto, test e nuove evidenze, senza retroattribuire modifiche ai benchmark.
Nessun nuovo download o copia di checkpoint, nessuna installazione o modifica al
sistema. **Dimostrata inferenza sotto limite del processo; non oltre RAM WSL,
non oltre RAM fisica host e non un benchmark cold NVMe.** Contesto massimo provato
otto posizioni. Nessuna nuova funzionalità o incremento 9.

## Riproduzione

Dalla radice, con i checkpoint e il venv già disponibili, usare nomi nuovi per gli
output: tutti quelli di questa revisione sono conservati.

```bash
python3 -m pytest -q tests
.venv-oracle7/bin/python -m tools.check_oracle8
.venv-oracle7/bin/python -m tools.check_oracle_precision7
python3 -m tools.report_increment8 results/increment8 --output /tmp/gate8-rivisto-nuovo.json
```

Gli script `check_memory.py`, `check_oracle.py`, `check_integrity.py` nella directory
delle evidenze rendono ispezionabili le prove supplementari. I loro output sono
nomi esclusivi: per ripeterli conservarne una copia con nuove destinazioni, senza
sovrascrivere i risultati. Comandi e soglie oracle nei JSON dei supervisori.
[Evidenze e indice](../results/increment8/review/README.md).

Standards: **1 P2 corretto**, nessun rilievo residuo. Spec: **2 P2 corretti**,
nessun rilievo residuo. Nessun commit o push eseguito dalla revisione.
