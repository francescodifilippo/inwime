# Incremento 10 — Qwen2.5-14B BF16 oltre la RAM fisica host

19 novembre 2026, 22:06–01:30 (20/11) CET. Branch `feat/m10-preflight-tooling` (PR #24), base
del protocollo `639de22`. Nessun push o merge. Runtime (`inwime_runtime/`, `src/`) e
kernel invariati. Pesi originali BF16, nessuna quantizzazione, nessun nuovo download.

Classi: **MISURATO** (osservato in questa sessione), **DERIVATO** (aritmetica dichiarata),
**IPOTESI**. Protocollo normativo: [`results/increment10/protocol.json`](../results/increment10/protocol.json)
con gli emendamenti 001–007. Evidenze: [`results/increment10/campaign/`](../results/increment10/campaign/README.md).

**Esito: gate 10 A–H PASS** (`results/increment10/campaign/gate10.json`, ricalcolato da
`tools/report_increment10.py`), con i limiti della sezione 11. Qwen2.5-14B BF16 originale:
27.983.013.888 B di pesi unici attivi, 1,629× i DIMM installati e 1,656× la RAM fisica host.
Prefill di 5 token e 3 decode reali, `The capital of France is Paris. The capital`, eseguiti in
uno scope da 2 GiB senza swap e senza OOM. Picco RSS 162 MB. Accordo elemento per elemento con
un oracolo indipendente bounded su logits, residui e K/V completi: `stock_fp32` e `full64` PASS
a copertura FULL; `stock_fp64` diagnostico FAIL grezzo, come in M9. Equivalenza 7B end-to-end di
`oracle10` con `oracle9` di M9 su tutti e tre i contratti (diagnostica `stock_fp64` eseguita il 20/11,
emendamento 007).

## 1. Audit iniziale

- Git: `5116175` pulito su `feat/m10-preflight-tooling`; emendamenti 001/002 già versionati.
- Il documento `docs/HANDOFF-2026-11-19-M10-acquisition.md` citato nella richiesta non era
  nel clone locale durante la campagna: era solo sul remoto (commit `43d3a3f`), integrato con
  un rebase il 20/11. I fatti dell'acquisizione vengono dai rapporti originali; l'handoff non
  li contraddice (causa del fermo: gate RAM, non checksum, spazio o rete).
- Checkpoint `checkpoints/qwen2.5-14b-97e1e76/`: 8 shard, SHA256 di acquisizione uguali ai
  digest LFS pinnati (rapporto di ripresa). Non modificato, non riscaricato.
- Acquisizione: primo tentativo fermato dal gate RAM dopo 19.449.945.445 B, ripresa
  10.101.739.432 B, totale **29.551.684.877 B = download pinnato**, nessuna doppia copia.
  I rapporti (uno era in `/tmp`, volatile) sono conservati sanificati in
  `campaign/acquisition/` con gli SHA256 degli originali.
- Pilota del 19/11 conservato in `campaign/pilot/` come riferimento informativo: niente
  preflight host, niente evizione, niente L3/L4, niente catture. Non entra nel gate.

Componenti che esistevano solo nel protocollo e sono stati costruiti ora: driver della
campagna (`tools/campaign10.py`, `tools/campaign10_sequence.sh`), equivalenza 7B
end-to-end (`tools/equivalence10_e2e.py`), parità 14B a copertura piena
(`tools/parity10.py`), sonde di guasto per modalità (`tools/fault10.py`), gate A–H con
BLOCKED (`tools/report_increment10.py`). Emendamento 004 li pinna.

## 2. Issue #25 — acquisizione senza gate di memoria (emendamento 003)

Commit separato `a03c807`, prima di ogni modifica di campagna. Dettagli in
[AMENDMENT_INCREMENT_10_003.md](AMENDMENT_INCREMENT_10_003.md). `acquire10 --execute`
richiede solo l'autorizzazione del proprietario e si ferma solo per rischi di
acquisizione. Suite 1199/1199, inventario a003 PASS, simulazione runner verde.

## 3. Campagna e host

Strumenti: `tools/campaign10_sequence.sh` → `tools/campaign10.py` → `tools/cgroup9.py`
(invariato) → `tools/benchmark_increment9.py` / `tools/oracle10.py` / `tools/fault10.py`. Per
ogni prova: controllo host ufficiale (`hostmem10`, profilo runtime o oracle, baseline della
sessione delle 22:10); con un esito diverso da GO si scrive `blocked-<prova>.json` e la prova
non parte. Poi L3 (`/sys/block/sdd/stat` e swap `sdc`), un osservatore host in-run che applica
i criteri di arresto congelati (nessuno è scattato in nessuna prova) e lo scope transitorio
con evizione mirata. Nessuna modifica a Windows o WSL.

**Blocco reale osservato (MISURATO).** Ogni oracolo in uno scope da 6 GiB mette l'host in
paging: disponibilità minima 327–710 MB, `PagesInput` fino a 7.098/s, pagefile fino a
+1,87 GB dal baseline. Il criterio di avvio «crescita del pagefile ≤ 512 MiB» è poi rimasto
violato per 15–40 minuti, anche con 5,9–6,9 GB disponibili: Windows non libera da solo le pagine
già scritte nel pagefile. Attese registrate in `host-wait.log`. Quattro tentativi non avviati,
ciascuno con il proprio `blocked-*.json`:

| Prova bloccata | Criterio violato | Esito successivo |
|---|---|---|
| O7-full64 | pagefile | eseguita come `O7-full64-a2` |
| O7-stock_fp64, -a2 | pagefile; host disponibile, paging | `-a3` fermata dall'operatore in attesa del GO (senza record); eseguita come `-a3` il 20/11 su richiesta del proprietario (emendamento 007) |
| O14-stock_fp64 | `PagesInput` | eseguita come `-a2` |

Il baseline non è mai stato rifatto per aggirare il criterio. Due evizioni mirate dei soli file
dei checkpoint (`eviction-*.json`, 3,4 e 5,3 GB di page cache del guest) hanno restituito
memoria all'host. Due cartelle temporanee di pytest create da questa sessione su tmpfs
(570 MB ciascuna) sono state rimosse; le suite successive usano `--basetemp` su disco.

## 4. Correttezza numerica

### 4.1 Equivalenza 7B end-to-end di `oracle10` (prerequisito preregistrato)

`oracle10` sul Qwen2.5-7B, stessa sequenza e stessi chunk di M9 (`5,1,1,1`; `8` per `stock_fp64`), contro le catture
`oracle9` di M9 (`tools/equivalence10_e2e.py`, identità di contratto, profilo, revisione,
sequenza e SHA verificata prima del confronto):

| Contratto | Logits | Residui | K/V | Argmax / greedy | Esito |
|---|---|---|---|---|---|
| `stock_fp32` | bit-identici | bit-identici | bit-identici | uguali | **PASS** (regola: tutto bit-identico) |
| `full64` | Δmax 9,77e-15 ≤ 1e-9 | bit-identici | bit-identici | uguali | **PASS** |
| `stock_fp64` (diagnostico, `O7-stock_fp64-a3`, 20/11) | bit-identici | bit-identici | bit-identici | uguali | **PASS** (riportato grezzo, non vincola F) |

Risorse: 42,8 s e 60,5 s, picco RSS 3,78 e 2,74 GB, 56,6 GB BF16 letti per contratto. Sono
4,4 GB meno di `oracle9`, perché l'embedding è letto per righe. `stock_fp64` (un solo chunk da 8):
25,4 s, picco RSS 2,72 GB, 14,1 GB BF16 letti; scope da 6 GiB con swap 0 e 0 OOM.

Il diagnostico 7B bit-identico mostra che `oracle10` riproduce `oracle9` anche nel profilo
`stock_fp64`. Il FAIL grezzo del 14B `stock_fp64` (sez. 4.2) resta quindi una proprietà del
confronto runtime/upstream con i cast FP32 interni, non un difetto del nuovo oracolo.

### 4.2 Qwen2.5-14B: runtime R1 contro `oracle10` (`tools/parity10.py`)

Sequenza `[785,6722,315,9625,374,12095,13,576]`, chunk `5,1,1,1`. Copertura FULL: logits
1.216.512 (8 × 152.064), residui 1.966.080 (48 × 8 × 5.120), K/V finali 786.432
(48 × 2 × 8 × 1.024). I K/V di ogni fase del runtime (lunghezze 5, 6, 7, 8) sono prefissi
byte-identici di quelli finali. I file di cattura sono legati per SHA256 al rapporto del runtime.

| Contratto | Logits Δmax | Residui Δmax (uso soglia) | K/V Δmax | Violazioni | Argmax/top-10 | Esito |
|---|---:|---:|---:|---:|---|---|
| `stock_fp32` (soglie FP32) | 3,89e-5 | 1,45e-2 (24,5%) | 1,17e-5 | 0 | 8/8 | **PASS, FULL** |
| `full64` (soglie FP64) | 2,93e-13 | 4,73e-11 (1,7e-7) | 1,95e-13 | 0 | 8/8 | **PASS, FULL** |
| `stock_fp64` (diagnostico) | 6,06e-6 | 2,09e-3 (283%) | 3,17e-6 | 51 residui | 8/8 | **FAIL grezzo** |

Le 51 violazioni diagnostiche cadono tutte nei layer 45 e 47 (posizioni 0 e 3), su residui
con |ref| fino a 773. Lo schema è quello di M9 (31 residui nell'ultimo layer): i cast FP32
interni di RMSNorm, RoPE e softmax di upstream. Il verdetto non è riscritto. Il greedy di
upstream coincide con la continuazione del runtime `[12095,13,576,6722]`. Soglie invariate.
Il gate ricalcola la parità dai file di cattura e la confronta con quella archiviata.

Oracoli 14B (scope 6 GiB, piano esplicito): `stock_fp32` 75 s, RSS 5,27 GB (piano 4,13 GB
anonimi, 5,69 GB con le pagine mmap); `full64` 114 s, RSS 3,15 GB; 111,9 GB BF16 letti per
contratto. N9: `check_oracle10` 4/4 OK (peso perturbato ed epsilon errato rilevati).

## 5. Memoria (MISURATO, R1–R4)

| Grandezza | R1 sync | R2 pipeline | R3 sync | R4 warm |
|---|---:|---:|---:|---:|
| `memory.max` / `memory.peak` scope | 2 GiB / 2 GiB | 2 GiB / 2 GiB | 2 GiB / 2 GiB | 2 GiB / 2 GiB |
| eventi `max` (reclaim, non OOM) | 122.530 | 122.772 | 121.538 | 115.283 |
| `oom` / `oom_kill` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| anon / kernel campionati (picco) | 142,0 / 7,8 MB | 142,5 / 7,2 MB | 142,0 / 7,1 MB | 142,0 / 2,8 MB |
| `VmHWM` worker | 161,6 MB | 162,2 MB | 161,6 MB | 161,5 MB |
| `VmPeak` / `RLIMIT_AS` | ≤ 256 MiB | **317,0 MB** / 384 MiB | ≤ 256 MiB | ≤ 256 MiB |
| swap scope / letture swap `sdc` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `MemAvailable` guest minimo | 8,36 GB | 8,31 GB | 8,33 GB | 8,32 GB |

`memory.peak == memory.max` non è un OOM: gli eventi `max` contano i reclaim della page cache
dentro lo scope; `oom` e `oom_kill` restano 0 e il worker termina con 0. Il `VmPeak` della
pipeline (317 MB) supera 256 MiB e conferma la stima preregistrata del caso peggiore (321,6 MB,
transitorio dell'arena glibc), quindi la `RLIMIT_AS` di 384 MiB. RSS e memoria anonima restano
sotto 256 MiB. RAM host, RAM WSL, page cache e RSS sono grandezze distinte. Il superamento della
RAM è una proprietà dei byte attivi (D), non del limite del cgroup.

## 6. Tempi e throughput (MISURATO)

| Mediana (3 decode), salvo prefill | R1 sync | R2 pipeline | R3 sync | R4 warm |
|---|---:|---:|---:|---:|
| Avvio con SHA integrale 29,54 GB | 19,8 s | 19,3 s | 19,1 s | 19,9 s |
| **Prefill 5 token** | 115,6 s | 113,9 s | 115,1 s | 116,5 s |
| **Decode, s/token** | 23,27 | 23,16 | 23,42 | 23,38 |
| Espansione BF16→FP64 | 8,92 | 9,96 | 8,91 | 8,89 |
| GEMV (dentro compute) | 6,82 | 7,16 | 6,86 | 6,85 |
| Regioni compute | 7,85 | 8,32 | 7,91 | 7,87 |
| Servizio letture | 5,62 | 5,86 | 5,70 | 5,71 |
| Overlap lettura/compute | 0 | 0,57 | 0 | 0 |
| Throughput decode | 0,043 tok/s | 0,043 | 0,043 | 0,043 |

**Prefill e decode.** Il prefill costa 4,97× un decode e legge **esattamente 5,00×** i byte
logici: 139,9 GB contro 27,98 GB. Ogni componente misurato scala di circa 5×: espansione 5,01×,
GEMV 5,0×, letture 4,82×. I contatori per tensore valgono forma × posizioni. Il runtime quindi
elabora le posizioni del prefill in serie e rilegge tutti i pesi per ciascuna. Il costo non
viene da un componente diverso dal decode: è lo stesso lavoro ripetuto per token. Non c'è
batching del prefill, già indicato come limite in M9.

**Decode.** Il decode è dominato dal calcolo: espansione più compute fanno circa 16,8 s
(72%), le letture circa 5,6 s (24%), il residuo dell'host 0,9 s. La pipeline sovrappone solo
0,57 s e aumenta il costo dell'espansione, per cui non accelera (come in M9). Temperature,
clock e potenza **NON MISURATI**; alimentazione non registrata.

**Warm/cold.** R4 (cache lasciata as-is subito dopo R3) non è più veloce. Lo scope trattiene al
più 2 GiB di page cache su 28 GB letti per token, e la residenza utile è trascurabile. R4 è solo
informativa e non entra in C o D.

## 7. I/O e attribuzione (MISURATO)

| Contatore | R1 | R2 | R3 | R4 (W) |
|---|---:|---:|---:|---:|
| L0 byte logici inferenza | 223.863.537.664 | uguale | uguale | uguale |
| L1 `/proc/PID/io` inferenza | 222.550.786.048 (99,41%) | 222.550.831.104 | 222.550.880.256 | — |
| L1 totale (avvio SHA + inferenza) | 252.091.678.720 | 252.091.723.776 | 252.091.637.760 | — |
| ΔL3 `/sys/block/sdd` | 252.195.885.056 | 252.351.389.696 | 252.138.557.440 | — |
| ΔL4 disco fisico Windows `0 C:` (globale) | 252.325.173.248 | 252.358.210.560 | 252.182.740.480 | 235.648.849.408 |
| Residenza dopo evizione (8 shard) | 0 B | 0 B | 0 B | non evitta |

Regole verificate su R1–R3: L1 ≥ 0,90·L0 sull'inferenza e su ogni decode; ΔL3 ≥ L1; ΔL4 ≥
0,90·L1; nessun `PagesInput` > 2048/s sostenuto per 30 s (picchi transitori fino a 3.322/s in
R1, nessuna annotazione). Limiti: L4 è un contatore globale del disco fisico, quindi conta
anche altre attività dell'host; non dimostra che la NAND fosse fredda. Cache di Windows, del
controller e dell'SSD non sono controllate. Il «cold» è cold guest. La banda apparente delle
letture di blocco (≈ 4,9 GB/s nel servizio letture) comprende readahead, copia kernel e
virtualizzazione: non è la banda isolata dell'NVMe. `io.stat` non è disponibile.

## 8. Guasti e rollback

| Id | Prova | Esito |
|---|---|---|
| N1–N3 sync | `F-sync-a2`: I/O error, lettura corta, cancellazione alla lettura 195 del decode, `RLIMIT_AS` 256 MiB | PASS: eccezione attesa, banco KV vecchio e descrittore intatti, banco candidato modificato, fail-stop, FD/arene rilasciati |
| N1–N3 pipeline | `F-pipeline`, 384 MiB | PASS, slot drenati |
| N4 | budget = piano − 1 B, in entrambe | rifiutato prima di leggere il payload |
| N5 | SHA errata / file cambiato (fixture) | test M9 invariati, PASS nella suite |
| N6 | SmolLM2-1.7B in scope da 32 MiB | OOM del kernel attribuito: `oom_kill` 1, contatore globale 2→3, exit 137, nessun rapporto di successo |
| N7 | worker fuori scope | rifiuto prima dei pesi (test M9 invariati) |
| N8 | mutazioni di attribuzione e del criterio D | 26 mutazioni delle evidenze reali, nessuna PASS (`gate10-mutations.json`) |
| N9 | oracolo con peso perturbato ed epsilon errato | `check_oracle10` 4/4 |
| N10 | acquisizione | test di `acquire10` (emendamento 003) |

**Errore trovato e conservato.** `F-sync` (tentativo 1) è fallito su un'asserzione della sonda
M9: «il banco candidato è già modificato». La sonda inietta il guasto alla lettura 100 del
decode. Sul 7B quella lettura cade in `o_proj` del layer 0, dopo `q` (49 letture), `k` e `v`. Sul
14B `q_proj` da sola richiede 100 letture da 262.144 elementi, quindi il guasto arrivava prima
di qualsiasi scrittura K/V. Le proprietà transazionali verificate prima di quell'asserzione
reggevano. La sonda però non dimostrava lo scarto di un banco già modificato. `tools/fault10.py`
inietta allo stesso punto strutturale di M9 (metà di `o_proj` del layer 0: lettura 92 sul 7B,
195 sul 14B) e mantiene invariata la verifica. F-pipeline è stata fermata prima dell'avvio,
senza record.

## 9. Gate 10

| Condizione | Esito | Base (ricalcolata) |
|---|---|---|
| A — Integrità | PASS | SHA dell'acquisizione (8/8), SHA integrale all'avvio di R1–R3 e re-hash finale uguali ai digest LFS pinnati |
| B — Memoria | PASS | R1–R4: scope 2 GiB e swap 0 visti dal kernel e dal worker prima dei pesi, picco ≤ limite, 0 OOM, anon+kernel e `VmHWM` ≤ 256 MiB, `RLIMIT_AS` per modalità |
| C — I/O | PASS | R1–R3: residenza 0, L1/L0 ≥ 0,90 per fase, ΔL3 ≥ L1, ΔL4 ≥ 0,90·L1, host monitorato, nessuna invalidazione |
| D — Oltre la RAM host | PASS | 27.983.013.888 B dai contatori = ricalcolo dagli header > max(16.900.739.072, 17.179.869.184) misurati nello stesso run > `MemTotal` 11.483.848.704, su R1 cold realmente letto |
| E — Forward | PASS | 5+3 token, 48 layer, 579 contatori = forma × 8 dagli header ufficiali, 111.925.002.240 coefficienti GEMV, KV 5/6/7/8, digest R1 = R2 = R3 = R4 |
| F — Correttezza | PASS | `stock_fp32` e `full64` PASS FULL ricalcolati, equivalenza 7B end-to-end PASS su entrambi; diagnostici riportati grezzi e ricalcolati: 14B `stock_fp64` FAIL, 7B `stock_fp64` PASS bit-identico |
| G — Risorse | PASS | 14 prove registrate supervisionate (anche gli oracoli diagnostici) (anche il tentativo fallito e N6), file prodotti legati ai digest dei record, 0 OOM non pianificati, crescita swap guest ≤ 256 MiB all'avvio e alla fine, host registrato; N1–N4 in entrambe le modalità, N6 attribuito; unica ripetizione con record (`F-sync`) giustificata dall'emendamento 004 |
| H — Regressioni | PASS | 1260/1260 test, inventario a007 verificato sul JUnit archiviato, catena 001→007, gate 7/8/9 ricalcolati PASS, runtime e kernel invariati, unico file preesistente modificato il workflow CI (emendamento 001) |

Precedenza FAIL > BLOCKED > NOT PROVEN > PASS. Un campo mancante accanto a uno contraddetto
nello stesso controllo dà FAIL. Un BLOCKED senza file di causa vale NOT PROVEN.

## 10. Test

| Esecuzione | Esito |
|---|---|
| Suite finale (`pytest-final.*`) | **1260 passed**, 0 skip |
| Esecuzioni precedenti conservate | `-a1`: 1245 passed, 1 failed (gate G su campagna vuota, emendamento 005); `-a2`: 1246 passed (prima dell'emendamento 006); `-a3`: 1014 errori di setup per una directory `--basetemp` rimossa dall'operatore (non fallimenti di test); `-a4`: 1252 passed (prima dell'emendamento 007) |
| `inventory10 verify` contro a007 | PASS 1260/1260 |
| `protocol10 check` | PASS, emendamenti 001–007 |
| Nuovi test M10 (`test_increment10_campaign.py`) | 61: driver, BLOCKED, osservatore, equivalenza, parità, algebra dei verdetti, mutazioni N8, posizione del guasto, regole degli emendamenti 006 e 007 |
| Mutazioni sulle evidenze reali | 26, nessuna PASS (serie precedenti: 20 in `gate10-mutations-a1.json`, 23 in `-a2`) |
| Revisione indipendente (subagente, sola lettura) | nessun difetto bloccante; cinque debolezze del gate, corrette dall'emendamento 006 (sez. 10.1) |
| Non eseguito | download reale con la correzione di #25 (per scelta: checkpoint già completo) |

### 10.1 Revisione indipendente

Un revisore separato, in sola lettura, ha ricalcolato A–G (PASS) e ha controllato parità, sonda
dei guasti, driver ed evidenze. Non ha trovato difetti bloccanti. Ha trovato cinque debolezze
del gate, tutte corrette nell'emendamento 006 e coperte da test e mutazioni:

1. `produced_sha256` dei record non verificato: ora ogni file prodotto è legato al suo digest,
   direttamente o tramite la mappa di sanificazione. La sanificazione ha toccato 13 file;
   `sanitization-reversal-local.json` mostra che, rimettendo i percorsi assoluti, tutti e 13
   riproducono il digest originale.
2. `latest()` permetteva di ripetere una prova finché passava: ordine numerico dei tentativi, e
   ogni tentativo superato con un record deve essere citato da un emendamento.
3. F: catture dell'oracolo 14B legate ai record O14; equivalenza 7B ricalcolata dai file, legata
   al record O7 e al manifesto delle catture di M9.
4. Crescita dello swap del guest verificata anche alla fine di ogni prova.
5. Regola del paging sostenuto ricalcolata dai campioni, non solo dalle annotazioni.

### 10.2 Diagnostica 7B `stock_fp64` (emendamento 007)

Il 20/11 il proprietario ha chiesto di completare la diagnostica rimasta. `O7-stock_fp64-a3` è
partita con il passo invariato della sequenza dopo un controllo di avvio GO (attesa 0 s) e ha
prodotto catture bit-identiche a `oracle9`. Il gate prima non leggeva questo diagnostico.
L'emendamento 007 aggiunge due controlli, attivi solo se la prova ha un record:

- F: l'equivalenza è ricalcolata dalle catture, legate al record O7 e al manifesto di M9; il
  verdetto grezzo (PASS o FAIL) deve coincidere con quello archiviato.
- G: gli oracoli diagnostici (O7 e O14 `stock_fp64`) sono supervisionati come quelli vincolanti.

Senza record (assente o BLOCKED) non si aggiunge alcun controllo: il diagnostico può solo
contraddire F o G. Tre nuove mutazioni (verdetto archiviato modificato, cattura non legata al
record, swap usato) danno FAIL. Suite, inventario, gate e mutazioni precedenti restano con suffisso
`-a4`/`-a2`.

## 11. Limiti residui

1. **Contratto diagnostico 14B `stock_fp64`**: FAIL grezzo riportato (51 residui, layer 45 e 47),
   come in M9; non vincola F. L'equivalenza 7B dello stesso profilo è ora PASS bit-identica
   (emendamento 007).
2. **Strumenti scritti durante la sessione** e pinnati dopo le prove (emendamenti 004–007). Le
   soglie e i criteri sono quelli congelati. Tre correzioni sono avvenute durante la campagna,
   ciascuna con l'evidenza del fallimento conservata (sonda dei guasti, controlli digest-only
   del gate, gate su campagna vuota).
3. **Indipendenza dell'oracolo**: Transformers upstream non modificato. Il profilo `full64` è
   dello stesso autore del runtime (come in M7–M9); l'indipendenza piena viene da `stock_fp32`.
4. **Cold = cold guest**: L4 è globale; NAND, cache Windows e controller non sono controllati.
   Temperature e potenza non misurate.
5. **Pressione host**: la campagna dipende dal paging di Windows. Il criterio del pagefile ha
   imposto attese lunghe; il proprietario deve liberare memoria per ripetere gli oracoli.
6. **Prestazioni**: 0,043 tok/s, dominate dal calcolo FP64 a singolo thread. Il prefill non è
   batchato. Il risultato non implica praticità per checkpoint multi-TB.
7. Il pilota e il download del 19/11 sono stati eseguiti prima di questa sessione. Il download
   ha usato una disattivazione locale del gate, ora resa permanente dall'emendamento 003.
8. Contesto massimo provato: 8 posizioni, capacità 16.

## 12. Decisione

**PASS** del gate 10 A–H, con i limiti dichiarati della sezione 11. È una dimostrazione
riproducibile, numericamente validata e strumentata di inferenza BF16 densa con pesi attivi
oltre la RAM fisica host. Non è una prova di prestazioni utili.

