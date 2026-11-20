# Incremento 10 — preparazione degli strumenti C1–C5

18 novembre 2026, 06:50–09:15 CET. Branch `feat/m10-preflight-tooling` da
`639de22` (`origin/main`, `main` locale allineato con fast-forward). Nessun commit,
push o merge. **Nessun byte di pesi Qwen2.5-14B scaricato**; acquisizione usata solo
in dry-run. Runtime di produzione, kernel, strumenti e test M1–M9 ed evidenze M7–M9
invariati: tutte le modifiche sono **file aggiunti**.

Origine: rapporto di preflight del 18/11 (CHANGES REQUIRED, C1–C5), Issue #4, Issue
#22. Protocollo spiegato in [PROTOCOL_INCREMENT_10.md](PROTOCOL_INCREMENT_10.md);
evidenze in [results/increment10/](../results/increment10/README.md).

Classi: **MISURATO** (osservato in questa sessione), **DERIVATO** (aritmetica
dichiarata), **IPOTESI**/stima.

## 1. File creati

| File | Ruolo |
|---|---|
| `tools/acquire10.py` | C1: acquisizione pinnata del 14B, dry-run predefinito, `--execute` solo con autorizzazione |
| `tools/hostmem10.py` | C2: preflight read-only della pressione di memoria host/guest |
| `tools/protocol10.py` | C2: costruzione e verifica del protocollo (numeri derivati ricalcolati) |
| `tools/inventory10.py` | C3: inventario dei test, verifica dell'identità dal JUnit |
| `tools/fixture_qwen10.py` | C4: fixture 14B-shaped (GQA 5:1, eps 1e-5, 3 shard) e copia strutturale sparse del 14B |
| `tools/oracle10.py`, `tools/oracle10_plan.py` | C5: oracolo bounded e piano di memoria esplicito |
| `tools/compare10.py` | C5: confronto con classi di copertura FULL/PARTIAL/STRUCTURAL/NOT PROVEN |
| `tools/check_oracle10.py` | C5: controlli nel venv di verifica (torch) |
| `tools/equivalence10.py` | C5: equivalenza a componenti sul 7B locale |
| `tests/test_increment10_{acquire,hostmem,limits,protocol,inventory,fixture,oracle}.py` | 143 test M10 |
| `docs/PROTOCOL_INCREMENT_10.md`, `docs/PREPARATION_INCREMENT_10.md` | documentazione |
| `results/increment10/` | protocollo, SHA, inventario, metadati ufficiali, evidenze, `SHA256SUMS` |

Nessun file modificato o cancellato (verificato da
`tests/test_increment10_protocol.py::test_no_preexisting_file_modified_or_deleted`
e `::test_runtime_and_kernel_unchanged_since_base`). Totale ≈ 4.800 righe di codice
e test, ≈ 1,1 MB di evidenze (di cui 532 KB l'inventario).

## 2. Risultati dei test (MISURATO)

| Esecuzione | Esito |
|---|---|
| Suite completa CI-equivalente (`make cpu-kernel`, `pytest -q tests -ra`, offline, Python 3.14.4, pytest 9.0.2) | **1053 passed, 0 skip, 0 xfail, 0 errori**, 118 s |
| Verifica dell'inventario pinnato sul JUnit archiviato | **PASS**, 1053/1053 node ID richiesti |
| `tools.sanitize9 results/increment9 --verify` (passo CI) | OK |
| `tools.protocol10 check` | PASS |
| `tools/check_oracle10.py` nel venv (torch 2.9.1+cpu, Transformers 4.57.3) | 4/4 OK |
| Test specifici M10 | acquire 55, hostmem 18, limits 2, protocol 15, inventory 15, fixture 17, oracle 21 |

Baseline M1–M9 invariata: 910 node ID raccolti a `639de22`, tutti PASS (nel
checkout reale, con storia Git e `artifacts/`, zero skip).

## 3. Diff rispetto a `origin/main`

Solo file nuovi non tracciati (nessuno staging): 10 strumenti, 7 file di test, 2
documenti, la cartella `results/increment10/`. `git diff origin/main` sui file
tracciati è vuoto.

## 4. Stato di C1–C5

### C1 — `tools/acquire10.py`: **COMPLETO**

- Manifesto pinnato alla revisione `97e1e763…298c9`: dimensioni esatte, ID blob,
  digest LFS degli 8 shard, SHA256 di ogni header grezzo. Digest del manifesto
  `69a3c92c…df84cd`. Indipendente dalle costanti del 7B.
- User-Agent generico (`Python-urllib/3`), nessuna credenziale, solo HTTPS e host
  in allowlist a ogni redirect, risposte di metadati limitate.
- Range accettato solo come 206 con `Content-Range` esatto, prima di leggere il
  corpo; `Content-Length` deve essere il resto atteso.
- `.part` di proprietà (marcatore `.part.acquire10.json`), ripresa con re-hash,
  rinomina finale con `link()+unlink()` (mai sovrascrittura), `.part` con digest
  errato marcato `rejected` e conservato, pulizia solo dei file propri, lock.
- Spazio sul volume fisico (C:) prima e ogni 256 MiB durante il trasferimento;
  margini configurabili (fisso 32 GiB, temporanei, verifica, rollback, evidenze
  1 GiB, crescita `swap.vhdx` 7.344.226.304 B, soglia di arresto 40 GiB); ENOSPC
  conserva il `.part`.
- `--execute` richiede un file di autorizzazione che nomina modello, revisione,
  digest del manifesto, destinazione e byte massimi; verificato prima di ogni
  richiesta di rete e di nuovo sotto lock; il tetto di byte è applicato al flusso.
- Test con server HTTP locale: risposte interrotte, Range ignorato su header e
  ripresa, `Content-Range` errato, `Content-Length` errato, checksum errati (shard
  e file piccoli), spazio insufficiente prima e durante, ENOSPC, `.part` estranei,
  troppo lunghi o completi, file esistenti mai sovrascritti, redirect verso host non
  ammesso, lock, autorizzazione non corrispondente, budget.
- **Usato solo in dry-run** (MISURATO): online 123.708 B di metadati (API, config,
  index, 16 Range 206 esatti); revisione, dimensioni e digest uguali al manifesto;
  nessuna directory creata; offline sui metadati catturati: uguale.

### C2 — protocollo preregistrato e pressione host: **COMPLETO**

`results/increment10/protocol.json` (SHA `8491cf5c…19dd7`), separato dal 9.
Contiene identità e checksum, pesi attivi, RAM host/WSL/cgroup, limiti sync e
pipeline, workload con tre decode reali, contratto numerico con soglie invariate e
copertura completa, fault injection, rollback, protocollo warm/cold e contatori,
criteri PASS/FAIL/NOT PROVEN/BLOCKED, criteri di avvio e arresto per la memoria
host, condizioni A–H, digest degli strumenti e dell'inventario. Dettagli nel
[documento del protocollo](PROTOCOL_INCREMENT_10.md).

- **Pipeline a 384 MiB** (DERIVATO dalle evidenze M9 + test sintetici): `VmPeak`
  14B stimato 254.509.448 B, margine a 256 MiB solo 13,3 MiB. Osservazione nuova:
  l'arena glibc del thread I/O può aggiungere a `VmPeak` un transitorio di 64 MiB
  (overhead strutturale 136 MiB sulla fixture, 72,9 MB sul 7B in M9). Caso peggiore
  ≈ 321,6 MB: non entra in 256 MiB, lascia 77 MiB in 384 MiB. Sync resta a 256 MiB.
- **Preflight host** (`tools/hostmem10.py`, read-only): Windows `AvailableBytes`,
  commit, pagefile e crescita dalla baseline, paging (`PagesInput/Output` da
  contatori grezzi), guest `MemAvailable`, swap e PSI, spazio di C:. Verdetti GO,
  DEFER, NOT MEASURED; ogni grandezza classificata capacità/osservazione/stima.
  La prima revisione lanciava un `powershell.exe` per campione e ne contava i
  page-in di avvio (8.456 pagine/s osservate); la revisione congelata prende i due
  campioni nello stesso processo. I rapporti superati sono conservati in
  `preflight-tooling/superseded-before-refreeze/`.

### C3 — inventario versionato: **COMPLETO**

`results/increment10/test-inventory.json` (SHA `12a67e37…c6553`, pinnato nel
protocollo): 1053 node ID in 8 gruppi (baseline M1–M9: 910; acquire 55, fixture
17, hostmem 18, inventory 15, limits 2, oracle 21, protocol 15), SHA256 di ogni
file in `tests/`, digest AST di ogni funzione, versioni (Python, pytest, NumPy, cc,
SHA di `src/cpu_kernel.c`, venv di verifica), conteggi attesi e politica. Verifica
dal JUnit: test rimosso, rinominato, sostituito a parità di conteggio, stesso nome
con corpo diverso, saltato, xfail, fallito, duplicato, XML falsificato (digest o
conteggi incoerenti), XML assente/malformato/vuoto/con DTD → FAIL o NOT PROVEN, mai
PASS (15 test di mutazione). Emendamenti solo come file aggiunti con catena di SHA.

### C4 — fixture sintetica: **COMPLETO**

`tools/fixture_qwen10.py`: 10/2 teste (GQA 5:1), `head_dim` 8, eps 1e-5, theta 1e6,
bias Q/K/V, head separata, larghezza 80, intermedia 54, vocabolario 37 (nessuna
dimensione multipla delle tile 7/5/3), 3 layer, 3 shard con un layer diviso fra due
shard e la head nell'ultimo. Un token con riga di embedding ≈ 2⁻¹⁰ rende l'epsilon
dominante (media dei quadrati ≈ 1e-6).

Il runtime invariato, sync e pipeline, kernel scalare e C, tile 7/5/16/262.144,
prefill di 5 e tre decode con KV incrementale coincide con un riferimento
indipendente (stdlib, `math.fsum`) entro 1e-11 relativo; contatori per tensore
esatti (embedding solo per righe); output byte-identici fra modalità. Sensibilità:
eps 1e-6, raggruppamento GQA modulo, RoPE interleaved o bias assenti spostano i
logits oltre 1e-3. Gli header reali del 14B, su copie sparse (nessun payload), sono
accettati dal `Qwen2Catalog` invariato con piano uguale al protocollo.

### C5 — `tools/oracle10.py`: **COMPLETO, con parità 7B parziale**

- Embedding per righe dallo shard, head FP64 mai materializzata (blocchi da 16.384
  righe), layer un modulo alla volta, nessun import InWiMe (test AST), logits completi
  (tutte le posizioni × vocabolario intero) verificati per forma.
- Piano di memoria esplicito prima di leggere un tensore; oltre il budget: rifiuto.
  Piano 14B (DERIVATO): `stock_fp32` 4,13 GB anonimi (5,69 GB con le pagine mmap),
  FP64 3,21 GB (4,77 GB); budget 5 GiB, scope 6 GiB. L'`oracle9` sul 14B richiederebbe
  ≈ 7,79 GB per il solo modulo head.
- **Scoperta:** una `F.linear` a blocchi non è bit-identica a quella intera (kernel
  BLAS di coda; fixture: FP32 fino a 1,9e-6, FP64 fino a 4,4e-15). Politica
  preregistrata: `stock_fp32` con head FP32 residente (bit-identica per costruzione,
  3,11 GB sul 14B); FP64 a blocchi con regola `max |Δ| ≤ 1e-9` e argmax identici.
- Fixture (MISURATO): residui e K/V sempre bit-identici a `from_pretrained`; logits
  FP32 bit-identici; logits FP64 entro la regola per blocchi 1, 5, 8, 16, 36,
  16.384; runtime invariato PASS su copertura FULL in `full64` (soglie FP64) e
  `stock_fp32` (soglie FP32); peso perturbato nella head o in una proiezione K ed
  eps errato → FAIL.
- **7B locale, componenti nuovi** (MISURATO, `tools/equivalence10.py`, catture M9
  `oracle-*.npz`): 8 righe di embedding bit-identiche alla lettura stdlib dello
  shard; norma finale + head su tutte e quattro le chiamate: `stock_fp32`
  **bit-identico**; `full64` bit-identico su 3 chiamate, Δmax 9,77e-15 sulla chiamata
  da 5 token, argmax identici. Copertura **PARTIAL** (i 28 layer non sono rieseguiti).
  Picco RSS 3,78 GB / 1,90 GB; ≈ 4,4 GB BF16 letti per contratto. Ha rivelato che il
  primo piano sottostimava l'RSS del 10–11% (pagine mmap): corretto.
- **Non eseguito: l'equivalenza 7B end-to-end** (≈ 7 min e 60 GB letti per
  contratto, +3–6 GB di memoria guest su un host già in paging). È preregistrata come
  prerequisito di ogni oracolo 14B: senza, F resta NOT PROVEN.
- Nessuna modifica al runtime è stata necessaria.

## 5. Spazio e memoria host (MISURATO, 18/11 09:05–09:10 CET)

| Grandezza | Valore | Classe |
|---|---:|---|
| C: libero (statvfs `/mnt/c` = CIM) | 153.345.945.600 B | osservazione |
| Richiesto prima del download (download + riserve) | 72.329.391.373 B | derivato |
| Eccedenza | ≈ 81,0 GB | derivato |
| ext4 VHDX libero (apparente, non usato) | 891,5 GB | osservazione |
| RAM fisica host / DIMM | 16.900.739.072 / 17.179.869.184 B | capacità |
| Host `AvailableBytes` | 0,40–0,45 GB (0,65–1,03 GB alle 07:00–08:40) | osservazione |
| Host `PagesInput` | 277–3.246 pagine/s | stima (intervallo 30 s) |
| Pagefile in uso | 6.936–7.265 MiB (8.639 MiB subito dopo i controlli 7B delle 08:45) | osservazione |
| Commit headroom host | 9,3–9,9 GB | stima |
| Guest `MemTotal` / `MemAvailable` | 11.483.848.704 B / 6,93 GB | capacità / osservazione |

Verdetti del preflight host: **DEFER per download, runtime e oracle**. Violati
`AvailableBytes` (tutti), `PagesInput` (runtime, oracle) e, per l'oracolo, guest
`MemAvailable` < 6,5 GiB. I controlli 7B (picco 3,8 GB in uno scope da 4,5 GiB)
hanno portato il pagefile da 7.089 a 8.639 MiB: la pressione host è reale e
misurabile, non un artefatto.

## 6. Limiti residui

1. **Host sotto pressione di memoria**: oggi ogni profilo è DEFER. Prima del
   download e della campagna il proprietario deve ridurre i carichi sull'host
   finché `tools/hostmem10.py` dà GO. Non si modificano WSL, swap o pagefile.
2. Equivalenza `oracle10` ↔ `oracle9` sul 7B end-to-end non eseguita (solo
   componenti): prerequisito preregistrato.
3. L'evidenza a componenti sul 7B è stata prodotta con la revisione precedente al
   freeze di `oracle10.py`/`oracle10_plan.py`; da allora sono cambiati solo piano
   di memoria e controllo dello scope, non il percorso numerico (gather, head). Non
   è stata ripetuta per non aggiungere pressione a un host in DEFER.
4. Head FP64 a blocchi non bit-identica: la regola `≤ 1e-9` è preregistrata, ma la
   bit-identità FP64 dei logits non è garantita.
5. Gate 10 A–H, wrapper di benchmark e driver dei guasti (piano del preflight, passi
   4–6) **non** sono implementati: servono dopo l'autorizzazione e non erano nel
   perimetro C1–C5. L'inventario e `compare10` ne sono i mattoni.
6. Banda interna attuale (2,0–2,5 GB/s O_DIRECT) e temperature restano NOT PROVEN /
   NOT MEASURED dal preflight.
7. I test che verificano runtime e file preesistenti invariati sono invarianti di
   M10: un incremento successivo che modifichi il runtime dovrà emendarli.
8. Fiducia nei digest: API `lfs.sha256` e `X-Linked-Etag` provengono dalla stessa
   radice HF (input di fiducia dichiarato).

## 7. Dimensione massima dell'acquisizione da autorizzare

**29.551.684.877 B** (8 shard 29.540.133.920 B + 9 file piccoli 11.550.957 B),
manifesto `69a3c92cb3f274f077b29838380974de21c51d77f31b0ba21c6aeceeb5df84cd`. Il
valore `max_transfer_bytes` dell'autorizzazione può essere esattamente questo:
l'acquisizione non scarica altro (nessuna cache HF, nessuna doppia copia,
temporanei in place). Spazio libero richiesto su C: prima dell'avvio:
72.329.391.373 B.

## 8. Destinazione raccomandata

`checkpoints/qwen2.5-14b-97e1e76/` sull'ext4 interno (VHDX sul Samsung 990 EVO,
volume fisico C:), già escluso da Git, accanto al 7B. Volume di riferimento per lo
spazio: `--backing-volume /mnt/c`. Il Sabrent resta escluso (0,43 GB/s, mount
effimero).

## 9. Condizioni di arresto

- C: libero < 40 GiB durante l'acquisizione → arresto, `.part` conservato.
- Digest o dimensione diversi → `.part` marcato `rejected`, nessuna rinomina.
- Host < 300 MB disponibili per > 10 s → pausa annotata; crescita del pagefile
  > 4 GiB dalla baseline → pausa; commit headroom < 1 GiB → kill dello scope.
- Guest `MemAvailable` < 768 MiB → `cgroup.kill` dello scope, run invalidato e
  conservato.
- `PagesInput` host > 2048/s per 30 s → attribuzione I/O del run invalida.
- Decode oltre 80 s/token → pausa e indagine; `oom_kill` non pianificato → FAIL.
- Nessuna ripetizione «finché passa».

## 10. Decisione finale

**READY WITH LIMITATIONS.**

C1–C5 sono realizzati e verificati senza modificare il motore di inferenza:
acquisizione pinnata e fail-closed (provata solo in dry-run), protocollo e
inventario congelati e verificabili, fixture GQA 5:1 / eps 1e-5 con riferimento
indipendente, oracolo bounded con piano di memoria esplicito e parità a componenti
sul 7B. Il limite che impedisce READY FOR DOWNLOAD APPROVAL è operativo e misurato:
l'host è in DEFER anche per il profilo di download (0,40–0,45 GB disponibili,
paging attivo). Restano inoltre da eseguire, dopo GO, l'equivalenza 7B end-to-end e
la costruzione del gate 10.

Il download non è richiesto né autorizzato da questo rapporto.
