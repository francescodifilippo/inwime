# Incremento 9 — hardening del gate dopo la chiusura M9

17 novembre 2026. Branch `fix/m9-post-closure-hardening`, partito da `main` a
`5eb25da955962496a4c258b9b7e6cba9d088c416` con working tree pulito. Riferimenti:
Issues #17 (robustezza del gate) e #18 (conservazione e documentazione). La
Issue #19 (percorsi personali storici) e la PR #16 non sono toccate.

Questa attività **non riapre l'esperimento M9**. Nessun forward, oracolo,
benchmark o download. Sono invariati: runtime, adapter Llama e Qwen2, kernel,
matematica FP64, pesi, `protocol.json`, misure, catture, report di parità, il
FAIL diagnostico `stock_fp64` e i contratti degli incrementi 7 e 8. Nessun
report derivato di `results/increment9/` è stato rigenerato: questa directory
contiene soltanto file nuovi.

## Aggiornamento dopo micro-correzione (commit `9417bb7`)

L'hardening iniziale `0cf37e2` ha prodotto **900 test PASS** (249 mirati),
registrati nel file storico [pytest-hardened.txt](pytest-hardened.txt).
Quel log viene **conservato senza riscritture** e le tabelle sottostanti
descrivono tale esecuzione.

Il successivo commit `9417bb7` modifica esclusivamente
`tools/report_increment9.py` e `tests/test_increment9_hardening.py`:
aggiunge verifiche di tipo per i contatori `completed`, `released`,
`session_length`, `arena_plan_bytes`, `reserved_peak_bytes`,
`source_file_bytes`, `source_files[*].nbytes`, nonché lunghezze e
generazioni KV. Corregge 10 falsi PASS dovuti all'uguaglianza numerica
in Python fra `int`, `float` e `bool` (esempio `True == 1`).

**Verifica locale riferita dall'operatore sul commit `9417bb7`:**
**910 test PASS** senza skip (259 nel modulo hardening), gate 7 e 8 PASS,
gate 9 A–H PASS con JSON invariato campo per campo rispetto a
`results/increment9/summary.json`, 80 mutazioni senza falsi PASS,
13 bypass respinti, `sanitize9 --verify` PASS e
`sanitize9 --against-commit 6031b52` PASS. I 10 nuovi test
eseguiti contro il gate precedente `0cf37e2` sono risultati tutti
rossi, con verdetto proprio E erroneamente PASS; sul nuovo gate passano.
La prova è stata eseguita in WSL dall'operatore; non è stata ancora
pubblicata una nuova trascrizione completa sotto `results/`.

**Nota di provenienza:** i numeri 900/249 nei log versionati indicano il
commit `0cf37e2`. I numeri 910/259 si riferiscono al successivo
`9417bb7`. Nessun output originario è stato modificato per sostituire
retroattivamente un risultato.

## Esito per intervento

| ID | Intervento | Classificazione |
|---|---|---|
| D-2 | Precedenza dei verdetti: FAIL > NOT PROVEN > PASS, uniforme tra precondizioni e condizioni | CORRECTED |
| D-3 | Contatori di runtime: solo interi JSON non negativi (niente `bool`, float, `null`, stringhe, negativi) | CORRECTED |
| D-4 | Manifest di sanificazione presente ma strutturalmente non valido: FAIL; assente: NOT PROVEN | CORRECTED |
| D-1 | Frasi superate in `docs/REVIEW_INCREMENT_9.md`, `docs/RUNTIME_CPU_INCREMENT_9.md`, `results/increment9/README.md` | DOCUMENTED |
| D-5 | Soglia `minimum_pytest_passed=613` | DOCUMENTED (meccanismo per M10 sotto; M9 invariato) |
| D-6 | Attestazione della sanificazione dagli originali del commit `6031b52`, senza copie locali | CORRECTED (strumento e test; il gate 9 non cambia) |
| BLOCKED | Il contratto del gate 9 non prevede BLOCKED | NOT APPLICABLE |

## Difetti riprodotti prima della correzione

Tutti riprodotti su copie temporanee delle evidenze reali, con il gate di
`5eb25da` ([hardening-tests-red.txt](hardening-tests-red.txt): 117 test su 249
falliscono; [green](hardening-tests-green.txt): 249 su 249 passano).

**D-3 — falsi PASS su valori malformati.** Con il gate di `5eb25da` lasciavano
A–H PASS, anche sul verdetto finale:

- `process_after_close.VmSwap_bytes` = `false` oppure `0.0`;
- `VmPeak_bytes` = `-1` oppure `1.5`;
- `VmHWM_bytes` = `-1` oppure `true` (bastava `peak_rss_bytes` valido nel `max`);
- `VmSwap` = `false` o `0.0` nel supervisore di un oracolo;
- `stats.failed` = `false` o `0.0`, `gpu_allocations_bytes` = `false` o `0.0`;
- `failures` = `false` o `0.0` in un confronto di parità, totale o per strato;
- `read_bytes` di blocco della fase B come float `1.2e11`.

Nei file sanificati (supervisori) la stessa classe di valori (`memory.peak:
true`, `memory.max: 2147483648.0`, `exit_code: false`, `samples: true`,
`stat.anon` negativo, `resident_bytes: true`) era respinta **solo** dal digest
fissato della precondizione di sanificazione: il verdetto proprio della
condizione restava PASS. Inoltre un valore presente di tipo errato che
provocava un `TypeError` (`samples: "0"`, `VmPeak_bytes: null`) era
classificato NOT PROVEN, come un campo assente.

**D-2 — precondizione FAIL più condizione NOT PROVEN.** Con il manifest
alterato (precondizione FAIL) e `faults.json` senza `cases` (G NOT PROVEN), G
finale era NOT PROVEN invece di FAIL. Con un `protocol.json` alterato e
l'emendamento assente la precondizione protocollo era NOT PROVEN: il campo
mancante mascherava il digest sbagliato.

**D-4 — manifest malformato.** `files: []`, `files: null`, `files` assente,
`files: "all"`, JSON `null` o `[]` davano precondizione NOT PROVEN, come un
manifest assente; ora FAIL, con e senza gli originali locali. Un manifest
valido con regole o digest contraddittori era già FAIL e resta FAIL.

## Correzioni

`tools/report_increment9.py`:

- `natural`, `exactly`, `counters_equal`, `counter_sum`: `type(v) is int` e
  `v >= 0`. Usati per ogni contatore di runtime in scope cgroup, evizione,
  letture di blocco, forward, parità, oracoli e condizioni B, C, D, E.
- `combine`: il verdetto finale è il peggiore tra il verdetto proprio e quelli
  delle precondizioni non PASS; un verdetto sconosciuto vale FAIL.
- `protocol_precondition` valuta ogni controllo da solo (`settle`): un digest
  del protocollo diverso da quello approvato resta FAIL anche se manca
  l'emendamento.
- `gate` passa a `sanitize9.verify` se il file del manifest esiste.

`tools/sanitize9.py`:

- `verify(..., present=None)`: assente NOT PROVEN, presente ma non un oggetto
  con `files` oggetto FAIL. La chiamata storica senza `present` è invariata.
- `verify_against_commit` e `--against-commit COMMIT` (D-6), sola lettura.

Le soglie e i parametri di M9 (`PARAMETERS`, inclusi 613/3/1/5) non cambiano.

## Gate originale e gate aggiornato

| Verifica | Gate di `5eb25da` | Gate aggiornato |
|---|---|---|
| Gate 9 sulle evidenze storiche, originali locali presenti | A–H PASS, exit 0 | A–H PASS, exit 0; report **identico** a `summary.json` in ogni campo ([gate9-hardened.json](gate9-hardened.json)) |
| Diagnostico FP64 standard | FAIL grezzo, 31 residui | FAIL grezzo, 31 residui |
| Gate 7 e gate 8 | PASS, uguali ai rerun conservati | PASS, uguali ai rerun conservati (strumenti non modificati) |
| 80 mutazioni (`review/mutations_green.py`) | 0 falsi PASS ([output](mutations-gate-5eb25da.txt)) | 0 falsi PASS ([output](mutations-gate-hardened.txt)) |
| 13 aggiramenti (`review/bypasses_followup.py`) | 13 chiusi | 13 chiusi, output identico ([output](bypasses-gate-hardened.txt)) |
| Nuovi casi avversariali (`tests/test_increment9_hardening.py`) | 117 falliti su 249 | 249 passati |
| Suite completa `python3 -m pytest -q tests` | 651 passati (`pytest-final.txt`, storico) | **900 passati**, zero skip, exit 0 ([pytest-hardened.txt](pytest-hardened.txt)) |
| Controlli oracle 9 / 8 / precisione 7 | OK: 3, 1, 5 test | OK: 3, 1, 5 test (non modificati) |

Le uniche differenze tra le due esecuzioni delle 80 mutazioni sono tre righe,
tutte casi D-2: `phaseA_residency_missing`, `phaseB_pipeline_residency_missing`
e `phaseB_sync_residency_missing`. Il campo rimosso da un file sanificato rompe
il digest fissato (precondizione FAIL); le condizioni che vedono il campo
mancante passano da NOT PROVEN a FAIL. Il verdetto proprio resta NOT PROVEN.

`review/gate-mutations-green.txt` è l'output storico del gate della
remediation (`eb669f2`). Il gate di `5eb25da` ne differiva già, per l'effetto
dei digest fissati nel follow-up. Resta invariato come evidenza storica.

Due asserzioni di test esistenti codificavano la vecchia precedenza e sono
aggiornate, con commento:

- `tests/test_increment9.py::test_gate_missing_evidence_is_not_proven`:
  directory vuota NOT PROVEN, come prima. Un `protocol.json` uguale a `{}` ora
  è FAIL (prima NOT PROVEN).
- `tests/test_increment9_review.py::test_missing_evidence_is_not_proven_not_fail`:
  B resta NOT PROVEN sulle proprie evidenze. Il verdetto finale è FAIL, per il
  digest fissato del file sanificato.

## D-1 e conservazione

I rapporti hanno ora una nota di stato al 17 novembre. Le frasi «non
committato» restano come testo storico, annotate con «stato al momento della
stesura; poi pubblicato in `5eb25da`».

`results/increment9/preservation.json` **non è modificato**: fotografa
`eb669f2` e i tre file allora non tracciati. L'attestazione successiva è in
[preservation-at-5eb25da.json](preservation-at-5eb25da.json). È stata prodotta
con `tools.integrity9 preservation` su `5eb25da`, a working tree pulito e prima
di qualunque modifica di questo branch. Risultato: unico file preesistente
modificato `README.md`, nessuna cancellazione, staging vuoto, nessun file non
tracciato. Rispetto al report storico risultano aggiunti e tracciati quattro
file: `preservation.json` stesso e i tre file `review/bypasses*`.

## D-6 — attestazione dagli originali storici

`python3 -m tools.sanitize9 results/increment9 --against-commit 6031b52` legge
con `git show` gli undici file del commit `6031b52`, senza modificare la
storia. Per ciascuno verifica tre cose: lo SHA256 dell'originale è quello
fissato nel gate; il file attuale ha lo SHA256 sanificato fissato; le sole
regole dichiarate applicate all'originale danno esattamente il file attuale.
Risultato: PASS, 11 file, commit
`6031b5263a24b350f97bab4e37874c6db61a8337`
([sanitization-against-6031b52.json](sanitization-against-6031b52.json)).
Prova quindi che nessun numero è cambiato anche su una macchina senza
`artifacts/increment9/original-evidence/`, purché il clone abbia la storia.
Il gate 9 non usa questo controllo: la sua semantica resta quella approvata.

## D-5 — proposta per il gate M10

Rischio: con `minimum_pytest_passed=613` e 651 test (oggi 900), fino a 38 test
delle regressioni del follow-up potevano sparire senza che H se ne accorgesse.
Un conteggio minimo non identifica quali test sono passati. Per M9 la soglia
resta 613 (R7, scelta dopo le misure, dichiarata): non viene cambiata a
posteriori.

Meccanismo proposto per M10, da fissare **prima** delle misure:

1. `results/increment10/test-inventory.json`, versionato. Contiene i node id
   pytest obbligatori raggruppati per regressione (per esempio `m9-review`,
   `m9-followup`, `m9-hardening`, `m10`). Per ogni gruppo il conteggio atteso;
   lo SHA256 di ogni file di test elencato; la versione di pytest.
2. La suite si esegue con `--junitxml`. L'XML è evidenza, accanto all'output `-q`.
3. Il gate fissa nel codice lo SHA256 dell'inventario, come fa per
   `protocol.json`. Ricostruisce l'esito per node id dall'XML. Richiede ogni id
   dell'inventario con esito passed, zero skip, xfail o error, e file di test
   con gli SHA dell'inventario.
4. Aggiungere test è libero. Rimuovere o modificare un test obbligatorio
   richiede un nuovo inventario, registrato come emendamento con la sua ragione.

## Limiti ancora presenti

- **Combinazione interna a una condizione.** Le condizioni C, D, parti di E e
  F calcolano più controlli in una sola espressione. Un campo mancante più un
  fatto contraddetto nella stessa condizione può dare NOT PROVEN invece di
  FAIL; mai PASS. Lo stesso vale quando manca un intero file letto all'inizio
  della condizione. La correzione D-2 riguarda la combinazione con le
  precondizioni; questo limite, già dichiarato, resta (DEFERRED).
- La tipizzazione stretta riguarda i contatori. Identificativi, token, liste
  di token e relative liste di ID sono confrontati per uguaglianza; alcuni
  altri campi strutturali non hanno ancora validazione stretta. Le lunghezze e
  generazioni KV scalari sono invece tipizzate dal commit `9417bb7`.
- Gli `integrity_read_bytes` letti dalla condizione A non sono tipizzati in A.
  Lo sono, per le stesse prove, nei controlli `forward` di B ed E.
- `tools.sanitize9 --verify` da solo continua a non conoscere l'insieme
  fissato; lo conoscono il gate e `--against-commit`.
- D-6 richiede un clone con la storia di `6031b52`; senza, il test relativo è
  saltato.
- Restano DEFERRED gli altri punti della Issue #18: la conservazione dei file
  aggiunti durante M9 (`inwime_runtime/qwen2.py`, tool, test) e la
  documentazione di SHA LFS, contatori `failures`, `generated_text`,
  `integrity-after.json`, SHA di R10, soglie R5–R7.

## Riproduzione

Dalla radice del repository:

```bash
python3 -m pytest -q tests
python3 -m pytest -q tests/test_increment9_hardening.py
python3 -m tools.report_increment9 results/increment9 --output /tmp/gate9-nuovo.json
python3 results/increment9/review/mutations_green.py
python3 results/increment9/review/bypasses_followup.py
python3 -m tools.sanitize9 results/increment9 --verify
python3 -m tools.sanitize9 results/increment9 --against-commit 6031b52
```

Red (gate di `5eb25da`): estrarre `tools/` e `tests/` del commit con
`git archive` in una directory vuota insieme alle evidenze. Copiarvi il nuovo
file di test ed eseguirlo, con `GIT_DIR` che punta al repository per D-6.
