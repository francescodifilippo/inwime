# Incremento 10 — protocollo preregistrato (Qwen2.5-14B, oltre la RAM fisica host)

**Fonte normativa:** [`results/increment10/protocol.json`](../results/increment10/protocol.json),
SHA256 in [`results/increment10/protocol.sha256`](../results/increment10/protocol.sha256).
Questo documento lo spiega; in caso di differenza vale il JSON. Il protocollo M9
(`results/increment9/protocol.json`) resta invariato e separato.

Il JSON è generato da `tools/protocol10.py build` e verificato da
`tools/protocol10.py check` (anche nel test `tests/test_increment10_protocol.py`):
le decisioni (soglie, limiti, run, regole dei verdetti) sono scritte nello
strumento; **ogni numero derivato** (dimensioni, byte attivi, spazio, stime di
`VmPeak`, conteggi di copertura) viene ricalcolato dai metadati ufficiali catturati
(`results/increment10/preflight-metadata/`) e dalle evidenze M9 versionate, mai
digitato a mano. Sono pinnati gli SHA256 degli strumenti e dell'inventario dei test.

Classi di ogni grandezza:

- **capacità**: dimensione installata o configurata (RAM, DIMM, `MemTotal`, volume);
- **osservazione**: stato misurato, con orario, che può cambiare;
- **stima**: derivata con aritmetica dichiarata o assunta; mai presentata come misura.

## 1. Checkpoint

| Campo | Valore | Classe |
|---|---|---|
| Modello | `Qwen/Qwen2.5-14B` base (non Instruct: Instruct ha `rms_norm_eps` 1e-6) | osservazione |
| Revisione | `97e1e76335b7017d8f67c08a19d103c0504298c9` | osservazione |
| Shard | 8, digest LFS e SHA256 dell'header pinnati in `tools/acquire10.py` | osservazione |
| Byte degli shard / download totale | 29.540.133.920 / 29.551.684.877 | osservazione |
| Payload / tensori | 29.540.067.328 B / 579, tutti BF16 | osservazione |
| Digest del manifesto di acquisizione | `69a3c92c…df84cd` | derivato |
| Destinazione | `checkpoints/qwen2.5-14b-97e1e76` (ext4 interno sul VHDX, volume fisico C:) | decisione |

Integrità: SHA256 calcolato in streaming da `acquire10` e registrato accanto al
digest LFS ufficiale; SHA integrale del runtime all'avvio di R1–R3; re-hash finale
con `tools/integrity9.py`.

## 2. Pesi unici matematicamente attivi (stima dagli header)

- Senza embedding: 27.982.931.968 B; riga di embedding: 10.240 B.
- Workload (5 + 3 posizioni): da **27.982.983.168 B** (5 righe distinte) a
  27.983.013.888 B (8 righe distinte).
- Byte logici per token 27.982.942.208; 337 matrici GEMV, 13.990.625.280
  coefficienti per token, 111.925.002.240 su 8 posizioni.
- Rapporto con la RAM fisica host (16.900.739.072 B): **≥ 1,6557×**; con i DIMM
  (17.179.869.184 B) ≥ 1,629×; con `MemTotal` WSL ≥ 2,437×.
- Non si afferma il superamento di RAM host + pagefile: irrilevante, i pesi sono
  pagine di file.

## 3. Workload

`The capital of France is` → `[785, 6722, 315, 9625, 374]`; prefill di 5 token e
**tre decode reali**: ogni token è il greedy dei logits del runtime e viene
reimmesso anche se è EOS (151643). Chunk `5,1,1,1`, capacità 16, tile 262.144,
integrità `full`, kernel `c-fp64-ordered`. Nessuna modifica a `inwime_runtime/` o
`src/` rispetto a `639de22` (condizione H, verificata dai test).

## 4. Memoria

| Grandezza | Valore | Classe |
|---|---:|---|
| RAM fisica host (`TotalPhysicalMemory`) | 16.900.739.072 B | capacità |
| DIMM installati | 17.179.869.184 B | capacità |
| `MemTotal` WSL (`memory=11GB`, invariato) | 11.483.848.704 B | capacità |
| Swap WSL (`swap=8GB`, invariato) | 8 GiB | capacità |
| Host `AvailableBytes` a riposo, 18/11 | 0,65–1,03 GB | osservazione |
| Pagefile in uso, 18/11 | 7.002–8.639 MiB | osservazione |

Scope del runtime: `systemd-run --user --scope`, `MemoryMax` 2 GiB,
`MemorySwapMax` 0. Soglie anon+kernel e `VmHWM` ≤ 256 MiB.

### 4.1 Limiti per sync e pipeline (`RLIMIT_AS`)

| Modalità | `RLIMIT_AS` | `VmPeak` 14B stimato | Margine |
|---|---:|---:|---:|
| sync | 256 MiB (come al 9) | 178.995.592 B | 89.439.864 B |
| pipeline, caso usuale | **384 MiB** | 254.509.448 B | 148.143.736 B |
| pipeline, caso peggiore | 384 MiB | 321.618.312 B | 81.034.872 B |

Derivazione (stima): `VmPeak(14B) = VmPeak(7B, M9) + piano(14B) − piano(7B)`, con
`VmPeak` M9 142.110.720 B (sync) e 217.624.576 B (pipeline) e piani 7B/14B
dall'unico `minimum_plan` invariato. A 256 MiB la pipeline avrebbe meno di 14 MB
di margine.

**Osservazione sintetica nuova (18/11):** l'arena glibc del thread I/O si crea
mappando 2 × 64 MiB e riducendo poi a 64 MiB allineati. `VmPeak` può quindi
includere un transitorio di 64 MiB che `VmSize` non conserva. Overhead strutturale
della pipeline: 72,9 MB sul 7B (M9), 136 MiB sulla fixture. Nel caso peggiore
256 MiB non basterebbe; 384 MiB lascia oltre 77 MiB. Test:
`tests/test_increment10_limits.py` (derivazione ricalcolata dalle evidenze M9;
esecuzione reale sync/pipeline sotto i limiti, byte identici, tre decode).
`RLIMIT_AS` è una rete di sicurezza, non la misura della memoria.

## 5. Contratto numerico

Soglie invariate da M7–M9, per ogni elemento: FP32 `|err| ≤ 2e-3 + 2e-4·|ref|`;
FP64 `|err| ≤ 2e-5 + 2e-6·|ref|`. Mai allargate dopo le misure.

| Contratto | Ruolo | Head in `oracle10` |
|---|---|---|
| `stock_fp32` | obbligatorio | FP32 **residente** (3,11 GB), bit-identica a `nn.Linear` per costruzione |
| `full64` | obbligatorio | FP64 **a blocchi** da 16.384 righe, mai materializzata |
| `stock_fp64` | diagnostico, riportato grezzo | FP64 a blocchi |

**Copertura completa** (tutta richiesta per un PASS di F): logits 1.216.512
valori (8 × 152.064), residui 1.966.080 (48 × 8 × 5.120), K/V finali 786.432
(48 × 2 × 8 × 1.024); K/V per fase del runtime (lunghezze 5, 6, 7, 8; 2.555.904
valori) prefissi byte-identici delle K/V finali; argmax e top-10 di ogni
posizione; greedy upstream uguale alla continuazione del runtime.

Classi di copertura (`tools/compare10.py`): **FULL**, **PARTIAL** (sottoinsieme o
tipo mancante), **STRUCTURAL** (solo forme, digest, argmax), **NOT PROVEN**. PASS
solo con FULL e zero violazioni; un valore non finito o una forma contraddetta è
FAIL; PARTIAL o STRUCTURAL senza violazioni restano NOT PROVEN.

### 5.1 Oracolo bounded `oracle10`

- Embedding: righe dei token lette dallo shard (`get_slice`), mai la matrice.
- Head: FP32 residente per `stock_fp32`; FP64 a blocchi per i contratti FP64.
  Una `F.linear` a blocchi **non** è bit-identica a quella intera in generale (kernel
  BLAS di coda): sulla fixture FP32 1,9e-6, FP64 4,4e-15. Da qui la politica per
  contratto, preregistrata in `DEFAULT_HEAD_MODE`.
- Layer e norma finale: un modulo alla volta, come `oracle9`.
- Memoria: piano esplicito da `tools/oracle10_plan.py` prima di leggere un tensore.
  Budget anonimo 5 GiB; scope `MemoryMax` 6 GiB, swap 0; `MemAvailable` guest
  ≥ 6,5 GiB. Piano 14B: `stock_fp32` 4,13 GB anonimi (5,69 GB con le pagine mmap
  del sorgente), FP64 3,21 GB (4,77 GB). Oltre il budget: rifiuto, nessun fallback.
- Indipendenza: nessun import InWiMe (test statico); dal runtime solo i token.

Equivalenza preregistrata:

1. fixture: bit-identità con `from_pretrained` (residui, K/V, logits FP32); logits
   FP64 a blocchi con `max |Δ| ≤ 1e-9` e argmax identici (`tools/check_oracle10.py`);
2. 7B, componenti nuovi (gather embedding, norma finale + head) sulle catture M9
   (`tools/equivalence10.py`): eseguita il 18/11, copertura PARTIAL, PASS;
3. **7B end-to-end, prerequisito di ogni oracolo 14B**: `stock_fp32` bit-identico su
   logits, residui e K/V; contratti FP64 con residui e K/V bit-identici e logits
   `≤ 1e-9` con argmax identici. Se fallisce, il contratto 14B corrispondente è
   NOT PROVEN. Costo stimato ≈ 7 min e 60 GB letti per contratto.

Rischio FP32: se `stock_fp32` fallisce e `full64` passa, F è FAIL; si riporta
l'utilizzo della soglia per layer.

## 6. Fault injection e rollback

N1–N3 errore I/O, lettura corta, cancellazione (sync e pipeline, 14B reale,
`integrity=metadata`); N4 budget ridotto di 1 B; N5 SHA errata e file cambiato
dopo l'apertura (fixture); N6 OOM in scope da 32 MiB (SmolLM2-1.7B); N7 worker
fuori dallo scope; N8 mutazioni di attribuzione (W per C) e del criterio D; N9 peso
perturbato ed epsilon errato (già coperto da `check_oracle10`); N10 guardia di
spazio, trasferimento interrotto, Range ignorato, digest errato (già coperto dai
test di `acquire10`).

Rollback: branch dedicato e solo file aggiunti; `.part` conservato per la ripresa
o rimosso solo da `acquire10 --clean-temporaries`; cancellazione del checkpoint
decisa dal proprietario, con misura di C: successiva; scope transitori `--collect`;
nessun `drop_caches`, `sysctl`, swap, `.wslconfig` o mount toccato.

## 7. I/O, warm e cold

| Run | Cache | Modalità | Catture | Scopo |
|---|---|---|---|---|
| R1 | C | sync | complete | prova principale |
| R2 | C | pipeline | digest | identità dei byte con R1 |
| R3 | C | sync | digest | ripetizione |
| R4 | W | sync | digest | solo informativo |

**C = cold guest**: file dell'esperimento evitti (`POSIX_FADV_DONTNEED`),
residenza ≤ 1% per shard con `fincore` prima dell'avvio. Non è NAND fredda: cache
di Windows, controller e SSD non sono controllate; L4 mostra se l'host avrebbe
potuto servire le letture. **W** = as-is subito dopo un run C, mai usato per C/D.

Contatori: L0 byte logici per tensore; L1 `/proc/PID/io`; L2 `memory.stat`, eventi,
PSI dello scope; L3 `/sys/block/<dev>/stat`; L4 disco fisico Windows (globale); L4b
`AvailableBytes`, `PagesInput/Output`, pagefile (`tools/hostmem10.py`). `io.stat`
non disponibile (controller `io` non delegato). Regole: L1 ≥ 0,90·L0 su inferenza e
ogni decode di un run C; ΔL3 ≥ L1; ΔL4 ≥ 0,90·L1 se leggibile (mai uguaglianza);
`PagesInput` host sostenuto > 2048 pagine/s durante un run invalida l'attribuzione
I/O di quel run (C NOT PROVEN per esso), i risultati numerici restano.

## 8. Pressione di memoria dell'host (preflight read-only)

`tools/hostmem10.py` legge, senza modificare nulla: classi CIM di Windows via
`powershell.exe` (RAM, `AvailableBytes`, commit, pagefile, contatori di paging
grezzi), `/proc/meminfo`, `/proc/vmstat`, PSI e cgroup del guest, `statvfs` del
volume del checkpoint e del volume fisico. Due istantanee a 30 s danno le
frequenze di paging; con `--baseline` (rapporto preso a inizio sessione) la
crescita di pagefile e swap.

Criteri di avvio (tutti misurati e soddisfatti → **GO**; uno violato → **DEFER**;
uno illeggibile → **NOT MEASURED**, la campagna non parte):

| Grandezza | download | runtime | oracle |
|---|---:|---:|---:|
| host `AvailableBytes` ≥ | 512 MiB | 1 GiB | 1 GiB |
| host commit headroom ≥ | 2 GiB | 3 GiB | 7 GiB |
| host `PagesInput` ≤ (pagine/s) | 1024 | 256 | 256 |
| host `PagesOutput` ≤ (pagine/s) | — | 256 | — |
| crescita pagefile dalla baseline ≤ | 1 GiB | 512 MiB | 512 MiB |
| guest `MemAvailable` ≥ | 2 GiB | 3 GiB | 6,5 GiB |
| crescita swap guest ≤ | 256 MiB | 256 MiB | 256 MiB |
| PSI memoria `some avg10` ≤ | — | 1,0 | 1,0 |
| volume fisico libero ≥ | 72.329.391.373 B | 40 GiB | 40 GiB |

Criteri di interruzione: host < 300 MB per > 10 s → pausa annotata; guest
`MemAvailable` < 768 MiB → `cgroup.kill` dello scope, run invalidato e conservato;
crescita del pagefile > 4 GiB → pausa; commit headroom < 1 GiB → kill dello scope;
`PagesInput` > 2048/s per 30 s → attribuzione I/O invalida; C: < 40 GiB durante
l'acquisizione → arresto con `.part` conservato.

Le soglie sono decisioni di politica (stime), non capacità misurate.
**Osservazione del 18/11:** host 0,65–1,03 GB disponibili a riposo, quindi i
profili runtime e oracle sarebbero DEFER; i controlli a componenti sul 7B (RSS di
picco 3,8 GB in uno scope da 4,5 GiB) hanno portato il pagefile da 7.089 a 8.639 MiB.

## 9. Verdetti, gate e arresto

PASS / FAIL / NOT PROVEN / **BLOCKED** (precondizione esterna registrata in
`blocked-*.json` con causa e orario: download non autorizzato, spazio sotto
soglia, host DEFER/NOT MEASURED, mount assente). Precedenza
**FAIL > BLOCKED > NOT PROVEN > PASS**. Un BLOCKED senza file di causa vale NOT
PROVEN. Un campo mancante e uno contraddetto nello stesso controllo danno FAIL
(residuo dell'Issue #22).

Gate A–H come nel preflight §8.4, con F legato a copertura FULL ed equivalenza 7B
end-to-end, e H legato all'inventario pinnato verificato sul JUnit archiviato
(`tools/inventory10.py verify`).

Condizioni di arresto: decode oltre 80 s/token; guest `MemAvailable` < 768 MiB;
host < 300 MB per > 10 s; crescita del pagefile > 4 GiB; `oom_kill` non pianificato
(FAIL); C: < 40 GiB durante l'acquisizione; nessuna ripetizione «finché passa».

## 10. Inventario dei test (Issue #22)

[`results/increment10/test-inventory.json`](../results/increment10/test-inventory.json),
SHA pinnato in `protocol.json`: 1.051 node ID (910 baseline M1–M9 a `639de22` e
141 di M10 in sette gruppi), SHA256 di ogni file in `tests/`, digest AST di ogni
funzione di test, versioni degli strumenti, politica degli esiti (skip, xfail,
errore, test mancante, duplicato o file cambiato → FAIL; test non inventariato →
NOT PROVEN; conteggio minimo mai sufficiente). Il gate ricostruisce ogni node ID
dal JUnit XML archiviato e ne confronta il digest. Emendamenti solo come nuovi file
in `results/increment10/amendments/`, con catena di SHA verificata da
`tools/protocol10.py check`.

## 11. Emendamenti

- [001](AMENDMENT_INCREMENT_10_001.md) (18/11, review della PR #24): gate host
  obbligatorio in `acquire10 --execute` (`tools/hostgate10.py`), identità del baseline
  in `hostmem10`, verifica CI dell'inventario con test non verificabili sul runner
  dichiarati (mai PASS) e attestazione locale separata; inventario emendato
  `test-inventory-a001.json` (1135 node ID). Soglie e criteri invariati.
- [002](AMENDMENT_INCREMENT_10_002.md) (18/11, review della PR #24 a `e8c588f`):
  monitor host read-only in-run (`tools/hostmonitor10.py`) che applica durante il
  trasferimento i criteri di arresto congelati (§8), con stop cooperativo, `.part`
  conservati e ripresa solo dopo un nuovo GO; inventario `test-inventory-a002.json`
  (1180 node ID). Soglie, sustain e azioni invariati.
