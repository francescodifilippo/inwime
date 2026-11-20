# Incremento 10 — emendamento 001: gate host dell'acquisizione e verifica CI dell'inventario

18 novembre 2026, 22:05–22:35 CET. Branch `feat/m10-preflight-tooling` su `f9251f5`
(PR #24, bozza). Nessun commit, push o merge. **Nessun byte di pesi 14B scaricato.**
Runtime (`inwime_runtime/`, `src/`), strumenti/test M1–M9 ed evidenze M7–M9 invariati.

Origine: review del proprietario sulla PR #24 (commit `f9251f5`), osservazioni 1 e 2.
File normativo: [`results/increment10/amendments/001-acquisition-host-gate-and-ci-inventory.json`](../results/increment10/amendments/001-acquisition-host-gate-and-ci-inventory.json).
Evidenze: [`results/increment10/amendment-001/`](../results/increment10/amendment-001/README.md).

## 1. Review: esito

| Osservazione | Verifica sul codice di `f9251f5` | Esito |
|---|---|---|
| 1. `acquire10.execute()` non impone un GO di `hostmem10` | Confermata: `execute` controllava autorizzazione, metadati e spazio; `dry_run.would_proceed_if_authorized` solo stato file e spazio | **Corretta** (§2) |
| 2. La CI non verifica l'inventario M10 | Confermata: `offline-cpu.yml` eseguiva pytest e `sanitize9`, non `inventory10 verify`; 11 skip M9 (originali locali assenti) | **Corretta** (§3) |
| Ulteriore: rapporto `hostmem10` senza identità del baseline | Un baseline obsoleto o sostituito non era rilevabile | **Corretta** (§2) |

## 2. Gate host obbligatorio (`tools/hostgate10.py`)

`acquire10 --execute` ora richiede **due record indipendenti**: `--authorization`
(decisione del proprietario, invariata) e `--host-report` + `--host-baseline`
(idoneità tecnica). Entrambi sono verificati **prima di ogni richiesta di rete** e di
nuovo **sotto lock, prima della prima richiesta di payload**. Il GO non implica
l'autorizzazione e viceversa; nel rapporto restano in sezioni separate.

Il gate non si fida del verdetto scritto nel rapporto:

- `protocol10 check` deve passare (digest del protocollo, ricalcolo, pin degli strumenti con la catena di emendamenti);
- il rapporto deve nominare lo SHA256 pinnato del protocollo, il profilo `download`, l'intervallo preregistrato (30 s) e lo SHA256 del file di baseline fornito;
- entrambi i campioni Windows presenti e senza errore; intervallo misurato in [28,5; 45] s;
- ogni grandezza e il verdetto **ricalcolati** dalle istantanee grezze con i criteri congelati del protocollo (soglie lette da `protocol.json`, mai dal rapporto);
- rapporto ≤ 900 s, non nel futuro (tolleranza 60 s); baseline precedente e ≤ 12 h;
- percorsi checkpoint/backing di rapporto e baseline = destinazione e volume dell'acquisizione, dimensioni dei volumi uguali;
- digest del manifesto nel protocollo = manifesto acquisito.

Verdetti: GO (unico che consente il trasferimento), DEFER, NOT MEASURED, REFUSED.
Il dry-run riporta `storage_ready`, `host_readiness` e `would_proceed_if_authorized`
(vero solo con storage pronto **e** host GO); senza rapporto host il dry-run è
dichiarato *storage-only* e `would_proceed_if_authorized` è falso.

`tools/hostmem10.py` registra ora `baseline_sha256` e `baseline_utc` (funzione
`make_report`); misure e criteri invariati.

### Test avversariali (`tests/test_increment10_hostgate.py`, 44)

DEFER (page-in, `AvailableBytes`, spazio), NOT MEASURED (campionamento Windows
fallito, rapporto o baseline mancanti), rapporto senza baseline, rapporto stale o nel
futuro, baseline di altra sessione o successivo, rapporto che scade fra il controllo
pre-rete e quello pre-payload (nessun payload, nessun `.part`), intervallo fuori
tolleranza, protocollo diverso, profilo runtime/oracle anche se GO, soglie abbassate
nel rapporto, catena del protocollo non verificata, policy assente o protocollo
illeggibile, manifesto diverso dal 14B, quantità falsificata, verdetto modificato,
baseline sostituito, JSON malformato, destinazione/backing/baseline su percorso
diverso, volume di dimensione diversa, byte autorizzati < piano, GO senza
autorizzazione (assente o incompleta), autorizzazione senza gate, risultato del gate
malformato, CLI senza uno dei due record, dry-run GO/DEFER. Ogni rifiuto di
`execute` avviene **senza alcuna richiesta** al server di test e senza creare la
destinazione.

Mutazioni (gate neutralizzato / freschezza disattivata / verdetto del rapporto
creduto): 21 / 3 / 22 test falliscono.

## 3. CI e inventario

- `tools/inventory10.py ci`: verifica il JUnit del runner contro l'inventario
  effettivo. I test dichiarati nell'emendamento (11, tutti
  `tests/test_increment9_hardening.py::…[…real]`, messaggio esatto
  `local unsanitized originals are not on this machine`) se saltati con quel
  messaggio sono **NOT VERIFIABLE ON RUNNER**, mai PASS: il verdetto
  sull'inventario completo del runner resta **NOT PROVEN**. Altri skip, messaggio
  diverso, fallimento o errore → FAIL. Il *runner verdict* (tutto ciò che il runner
  può eseguire è PASS, nessun test mancante/extra/duplicato) è separato.
- L'**attestazione locale** (`amendment-001/local-attestation.json`, JUnit
  archiviato con SHA pinnato) deve essere un PASS completo dello stesso inventario;
  è etichettata come risultato locale, non del runner.
- Workflow: aggiunti `python -m tools.protocol10 check` e
  `python -m tools.inventory10 ci --junit ci-results/pytest.xml`; caricati entrambi
  i file di `ci-results/`. Nessuna modifica alle evidenze o ai test M9.
- Test: `tests/test_increment10_amendment.py` (38): classificazione runner,
  comando `ci` (attestazione assente, saltata, SHA o inventario diversi, catena
  rotta), 17 mutazioni della catena di emendamenti, inventario che perde un test,
  numerazione, evidenze congelate byte-identiche, insieme non verificabile = solo i
  casi M9 `real`.

## 4. Preregistrazione ed emendamento

Nessun file congelato modificato: `protocol.json` (`8491cf5c…`), `protocol.sha256`,
`test-inventory.json` (`12a67e37…`, 1053 node ID), `SHA256SUMS` e le evidenze di
`preflight-tooling/` sono byte-identici (registrati in
`frozen_evidence_preserved` e verificati da un test).

Procedura: `python -m tools.protocol10 amend --spec results/increment10/amendment-001/amendment-spec.json`
scrive `amendments/NNN-<slug>.json` in modalità `x`, con `from` dalla catena
effettiva e `to` dai file su disco. `protocol10 check` verifica: SHA del protocollo e
dell'emendamento precedente, numerazione, chiavi ammesse (`pinned_tools_sha256`,
`test_inventory`, `acquisition_host_gate`, `ci_runner_verification`; soglie,
criteri, workload e regole dei verdetti restano ricostruiti dal codice e confrontati
col JSON congelato), ogni `from` uguale al valore effettivo, inventario emendato
superset del precedente, digest effettivi uguali ai file.

| Pin | Da (`f9251f5`) | A |
|---|---|---|
| `tools/acquire10.py`, `tools/hostmem10.py`, `tools/protocol10.py`, `tools/inventory10.py` | digest congelati | nuovi digest (nell'emendamento) |
| `tools/hostgate10.py`, `.github/workflows/offline-cpu.yml` | non pinnati | aggiunti |
| Inventario | `test-inventory.json` 1053 | `test-inventory-a001.json` **1135** (+44 hostgate, +38 amendment; nessun test rimosso o rinominato) |

Modificati due test M10 esistenti (non M9): l'helper `run` di
`test_increment10_acquire.py` passa un gate GO esplicito e il dry-run storage-only
deve dare `would_proceed_if_authorized = False`; `test_pinned_tools_exist_and_match`
usa i pin effettivi. I digest dei file sono registrati nell'inventario emendato.

## 5. Suite (MISURATO)

| Esecuzione | Esito |
|---|---|
| Locale (`make cpu-kernel`, `pytest -q tests -ra`, Python 3.14.4, pytest 9.0.2) | **1135 passed, 0 skip, 0 fail**, 156 s |
| `inventory10 verify` locale vs `test-inventory-a001.json` pinnato | **PASS** 1135/1135 |
| Simulazione runner (clone senza originali M9, passi del workflow) | **1124 passed, 11 skipped, 0 fail**; `sanitize9 --verify` OK; `protocol10 check` PASS; `inventory10 ci` runner **PASS**, inventario completo sul runner **NOT PROVEN** (11 non verificabili), attestazione locale **PASS** 1135/1135, exit 0 |
| CI GitHub su `f9251f5` (precedente) | 1042 passed, 11 skipped: non confrontabile 1:1 col locale |

La CI reale sul nuovo codice non è stata eseguita (nessun push).

## 6. Memoria host Windows (MISURATO, sola lettura, 22:05–22:31 CET)

`hostmem10`, nuovo baseline, profili download/runtime/oracle, due volte (revisione
congelata 22:09–22:11; revisione emendata 22:29–22:31): **tutti DEFER**.

| Grandezza (revisione emendata) | download | runtime | oracle | soglia download |
|---|---:|---:|---:|---:|
| `AvailableBytes` | 364 MB | 655 MB | 765 MB | ≥ 537 MB |
| `PagesInput`/s | 4.326 | 5.220 | 19.730 | ≤ 1.024 |
| commit headroom | 7,67 GB | 8,01 GB | 9,40 GB | ≥ 2,15 GB |
| guest `MemAvailable` | 6,87 GB | 6,87 GB | 6,90 GB | ≥ 2,15 GB |
| pagefile in uso | 8.666 MiB | 8.643 MiB | 8.156 MiB | crescita ≤ 1 GiB |

Gate sul rapporto download reale: **DEFER** (non REFUSED: rapporto valido, fresco,
coerente), `acquire10` dry-run: `storage_ready` vero, `would_proceed_if_authorized`
**falso**. Nella prima serie il profilo download violava solo `PagesInput` (7.990/s).

Consumo per processo (Get-Process, 22:07; privata = commit, WS = working set):
commit totale 42,7 / 50,9 GB, disponibili 685 MB, standby 684 MB, pool non paginato
1,75 GB, page-in 4.468/s.

| Processo (gruppo) | N | Privata | WS | Indicazione |
|---|---:|---:|---:|---|
| Chrome | 48 | 10,1 GB | 4,2 GB | chiudere schede/finestre o Chrome intero: **principale leva** |
| `vmmemWSL` (VM WSL2) | 1 | 5,8 GB | 2,0 GB | è il guest stesso; non toccato |
| ChatGPT desktop | 13 | 1,6 GB | 0,9 GB | chiudibile |
| Bitdefender (`bdservicehost`) | 5 | 1,3 GB | 0,3 GB | antivirus: **non** chiudere |
| WebView2 (WhatsApp 0,59; Teams 0,34; Search 0,21; Google Drive 0,18 GB) | 24 | 1,3 GB | 0,4 GB | WhatsApp e Teams chiudibili |
| Claude desktop | 12 | 1,2 GB | 0,3 GB | chiudibile se non in uso |
| Launcher giochi: Steam, EA, GOG Galaxy, Epic, Ubisoft, Battle.net | 25 | ≈ 3,3 GB | ≈ 0,4 GB | tutti chiudibili (avvio automatico) |
| Razer (AppEngine, Central, Synapse) | 12 | ≈ 1,3 GB | ≈ 0,2 GB | chiudibili se periferiche non servono |
| Logi Options+ agent | 1 | 0,75 GB | 0,08 GB | chiudibile |
| `svchost` (50), `explorer`, `dwm` | — | ≈ 1,7 GB | ≈ 1,0 GB | sistema: non toccare |

Interpretazione (stima): la RAM fisica (16,9 GB) è satura dal working set; il
paging continuo è il criterio che fallisce in tutti i profili. Chiudendo Chrome,
launcher, ChatGPT/WhatsApp/Teams e le app Razer/Logitech si liberano ≈ 18 GB di
commit e, soprattutto, i loro working set (≈ 6 GB): verosimilmente sufficiente per
il profilo download (≥ 512 MiB disponibili, ≤ 1.024 page-in/s), da **verificare**
ripetendo `hostmem10`. Nessun processo terminato, nessuna modifica a WSL, pagefile,
swap o impostazioni.

## 7. Decisione

**READY WITH LIMITATIONS.**

Le due osservazioni della review sono corrette nel codice e coperte da test
avversariali; la preregistrazione è conservata con un emendamento tracciabile e
verificato; la suite locale è 1135/1135 e la simulazione del runner è coerente con
il contratto. Il limite che impedisce READY FOR DOWNLOAD APPROVAL è operativo e
misurato: l'host è **DEFER** anche per il profilo download. Inoltre la CI GitHub sul
nuovo codice non è ancora stata eseguita (richiede un push, non autorizzato) e
l'emendamento attende l'accettazione del proprietario nella review.

Passi successivi, nell'ordine:

1. il proprietario chiude manualmente le applicazioni del §6;
2. nuovo baseline e rapporto `hostmem10 --profile download --baseline …`;
3. `python -m tools.hostgate10 checkpoints/qwen2.5-14b-97e1e76 --backing-volume /mnt/c --host-report … --host-baseline …` → GO;
4. solo allora il proprietario decide se creare il file di autorizzazione (non creato).
