# Incremento 10 — emendamento 002: monitor host durante l'acquisizione

18 novembre 2026, 22:48–23:05 CET. Branch `feat/m10-preflight-tooling` su `e8c588f`
(PR #24, bozza). Nessun commit, push o merge. **Nessun byte di pesi 14B scaricato.**
Nessuna modifica a Windows/WSL. Runtime (`inwime_runtime/`, `src/`), test/strumenti
M1–M9, evidenze M7–M9, protocollo congelato ed emendamento 001 invariati.

File normativo: [`results/increment10/amendments/002-acquisition-in-run-host-monitor.json`](../results/increment10/amendments/002-acquisition-in-run-host-monitor.json).
Evidenze: [`results/increment10/amendment-002/`](../results/increment10/amendment-002/README.md).

## 1. Review

Osservazione della review su `e8c588f`, verificata nel codice: `execute` chiamava
`host_readiness` solo all'avvio. Durante `transfer()` girava soltanto `Guard.check()`,
cioè lo spazio libero. Le condizioni di arresto host preregistrate non erano
sorvegliate mentre i byte scorrevano, e il GO iniziale poteva diventare obsoleto.
**Confermata e corretta**: il monitor ora fa parte dell'acquisitore, che lo esige per
procedere. Non è un gate esterno.

## 2. Monitor (`tools/hostmonitor10.py`)

Criteri applicati: `protocol.json → host_memory_preflight.abort_criteria`, letti dal
file congelato, con soglie, sustain e azioni invariati:

| Criterio | Azione preregistrata | Significato per l'acquisizione (emendamento 002) |
|---|---|---|
| host `AvailableBytes` < 300 MB per ≥ 10 s | pause-and-annotate | stop |
| guest `MemAvailable` < 768 MiB | kill-scope-invalidate-run | stop |
| crescita pagefile dal baseline > 4 GiB | pause-and-annotate | stop |
| commit headroom host < 1 GiB | kill-scope-invalidate-run | stop |
| host `PagesInput` > 2048/s per ≥ 30 s | annotate-io-attribution-invalid | sola annotazione (come preregistrato) |
| volume fisico libero < 40 GiB | stop-acquisition-retain-part | stop |

`protocol10 check` impone che ogni azione pause/kill/stop sia mappata a `stop`; solo
il criterio di attribuzione I/O può restare annotazione.

**Funzionamento.**

- Avvio: il monitor parte sotto lock, dopo il secondo controllo `hostgate10` e prima
  della prima richiesta di payload.
- Se entro 60 s non arriva un campione valido, oppure il primo campione è già in
  condizione di stop, il trasferimento è rifiutato senza scaricare alcun byte.
- Durante il trasferimento, `Guard.poll_host()` è chiamato prima di ogni richiesta e
  dopo ogni chunk.
- In caso di stop viene sollevata `HostPressureStop` (sottoclasse di `TransferStopped`):
  - `.part` e marcatore `partial` restano conservati;
  - il lock viene rilasciato;
  - il processo di campionamento figlio viene terminato;
  - l'evidenza di stop (`--evidence`) contiene il riepilogo del monitor e i `.part`
    conservati.
- La ripresa avviene con un nuovo `--execute` e un nuovo GO fresco: il `.part` viene
  ri-hashato e si chiede solo il resto.

**Read-only.** Un solo `powershell.exe` figlio esegue in ciclo le query CIM di
`hostmem10` (`Get-CimInstance`, `ConvertTo-Json`, `Start-Sleep`). I valori del guest
vengono da `/proc` e da `statvfs`. L'unico processo che viene terminato è quel figlio.

**Bounded** (parametri dell'emendamento):

| Parametro | Valore |
|---|---:|
| intervallo di campionamento | 5 s |
| assenza massima di campioni validi | 20 s |
| attesa del primo campione | 60 s |
| lunghezza massima di una riga | 64 KiB |
| buffer di lettura | 256 righe |
| campioni conservati | 2048 |
| errori conservati | 64 |
| annotazioni conservate | 256 |

`poll()` fa lavoro solo quando arriva un campione.

**Fail-closed.**

- Un campione è valido solo se tutte le grandezze dei criteri sono leggibili. Il page
  rate può mancare solo sul primo campione; un contatore che si azzera rende il
  campione invalido.
- Righe d'errore, valori illeggibili o un campionatore silenzioso portano allo stop
  dopo 20 s senza campioni validi.
- Un campionatore che termina porta allo stop immediato.
- Se il campionatore non parte, il trasferimento è rifiutato prima del payload.

**Limiti dichiarati.**

- Latenza fino a un intervallo di campionamento più il sustain preregistrato.
- Il monitor ferma solo questa acquisizione: non allevia la pressione causata da altri
  processi.
- Il campionatore costa un `powershell.exe` (circa 100 MB) per tutta la durata del
  trasferimento.
- Il ciclo PowerShell reale non è stato eseguito: host in DEFER, nessun trasferimento.
  Sono stati testati il percorso subprocess e reader con un campionatore finto e il
  comando PowerShell in modo statico.

## 3. Test (`tests/test_increment10_hostmonitor.py`, 45)

- **Policy.** Valori preregistrati; criteri applicati uguali a quelli congelati;
  mappatura mancante rifiutata.
- **Passaggio GO → critico (unità).**
  - `AvailableBytes` < 300 MB: stop solo dopo 10 s di violazione continua; un calo
    transitorio non ferma.
  - Stop immediato per guest < 768 MiB, pagefile oltre 4 GiB, commit headroom < 1 GiB
    e spazio < 40 GiB.
  - Il paging sostenuto produce una sola annotazione per episodio e nessuno stop.
- **Errori di campionamento.**
  - Righe d'errore: stop esattamente quando si superano 20 s dall'ultimo campione
    valido.
  - Valori Windows o guest illeggibili, campionatore silenzioso, campionatore che
    termina.
  - Nessun primo campione entro 60 s; primo campione già critico; campionatore che non
    parte.
- **Limiti di memoria.** Storico ed errori limitati; dopo uno stop non si consumano
  altri campioni.
- **Integrazione con `execute`** (server HTTP locale, chunk da 512 B, transizione
  legata al file in scrittura):
  - pressione durante il primo shard, poi ripresa con host sano: `resumed .part
    re-hashed then streamed`, byte e SHA esatti;
  - pressione durante lo shard successivo: il primo shard è già verificato e
    committato, il secondo resta come `.part`;
  - pressione sul guest;
  - critico già prima del primo payload: nessuna richiesta di payload, nessun `.part`;
  - errori di campionamento e campionatore morto a metà trasferimento;
  - paging solo annotato: trasferimento completo;
  - monitor assente o falso: rifiuto prima della rete;
  - l'evidenza di stop viene scritta dalla CLI.
  In ogni stop: `.part` presente e minore del file, marcatore `partial` coerente,
  nessun file finale, lock rimosso, campionatore chiuso.
- **Subprocess reale** (campionatore Python finto): consegna dei campioni e
  terminazione del figlio; righe troppo lunghe, non JSON o non oggetto; uscita del
  figlio uguale a stop. Il comando PowerShell usa solo i verbi Get, Measure, Select,
  ConvertTo e Start-Sleep.
- **Regole di emendamento.** Rifiutati: kill/pause/stop declassati ad annotazione,
  azione mancante, valore sconosciuto, intervallo ≥ assenza massima, limiti nulli,
  criteri personalizzati.

**Mutazioni.** Ho disattivato a turno quattro parti del codice per verificare che i
test se ne accorgano:

| Parte disattivata | Test che falliscono |
|---|---:|
| `poll_host` nel trasferimento | 6 |
| controllo di staleness | 5 |
| sustain | 2 |
| tutte le azioni ridotte ad annotazione | 10 |

**Test M10 esistenti ritoccati** (non M9):

- gli helper di `test_increment10_acquire.py` e `test_increment10_hostgate.py`
  passano un monitor sano;
- `test_repository_chain_verifies` accetta che la catena continui dopo 001.

Supporto condiviso: `tests/support_increment10_monitor.py`, che non è un modulo di
test.

## 4. Preregistrazione

| Pin | Esito |
|---|---|
| `tools/acquire10.py`, `tools/protocol10.py` | nuovi digest nell'emendamento 002 (`from` = valori effettivi dopo 001) |
| `tools/hostmonitor10.py` | aggiunto |
| `hostgate10`, `hostmem10`, `inventory10`, workflow CI | invariati |
| Inventario | `test-inventory-a002.json`, **1180** (+45; nessun test rimosso o rinominato) |
| CI | `ci_runner_verification` riscritta solo per puntare all'attestazione `amendment-002/`; stessi 11 test non verificabili |

Byte-identici rispetto a `e8c588f` e registrati in `frozen_evidence_preserved`:

- `protocol.json`, `protocol.sha256`, `test-inventory.json`;
- `SHA256SUMS`;
- `test-inventory-a001.json`, emendamento 001;
- `amendment-001/SHA256SUMS` e l'attestazione 001.

## 5. Suite (MISURATO)

| Esecuzione | Esito |
|---|---|
| Locale (`make cpu-kernel`, `pytest -q tests -ra`, Python 3.14.4, pytest 9.0.2) | **1180 passed, 0 skip, 0 fail**, 186 s |
| `inventory10 verify` locale vs `test-inventory-a002.json` pinnato | **PASS** 1180/1180 |
| Simulazione del runner (clone senza originali M9, passi del workflow) | **1169 passed, 11 skipped, 0 fail**; `sanitize9` OK; `protocol10 check` PASS; `inventory10 ci`: runner **PASS**, inventario completo **NOT PROVEN** (11 non verificabili), attestazione locale **PASS** 1180/1180, exit 0 |

La CI GitHub sul nuovo codice non è stata eseguita (nessun push).

## 6. Decisione

**READY WITH LIMITATIONS.**

L'osservazione della review è risolta. Il monitor in-run applica le condizioni
preregistrate senza cambiarle. Gli arresti sono sicuri e i download parziali si
possono riprendere.

Il download resta non autorizzato per tre motivi:

- l'ultimo preflight Windows è DEFER;
- il ciclo PowerShell del monitor è verificato su host reale solo staticamente;
- la CI GitHub sul nuovo codice non è stata eseguita.
