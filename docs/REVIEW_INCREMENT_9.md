# Revisione dell'incremento 9 e remediation

16 novembre 2026. **Pronto per una nuova revisione indipendente; nessun commit
eseguito.** La revisione indipendente dell'incremento 9 non ha trovato P0 né P1.
Ha trovato quattro P2, tutti nel gate delle evidenze e nello strumento di
acquisizione, e dieci P3. Questa remediation li corregge. Runtime, adapter
Qwen2, kernel e matematica sono invariati; nessuna misura è stata ripetuta.
[Gate A–H ricalcolato](../results/increment9/summary.json): PASS.

Perimetro: base `68ba9f3adf3cffb89b18b19525644fa0d7c6d806`. All'inizio della
remediation l'incremento 9 era già nel commit `6031b52` («passo 9») e il
riferimento locale `origin/main` puntava allo stesso commit. Non sono stati
eseguiti forward, oracoli, benchmark, download, commit, staging, push, fetch o
reset. Rapporto aggiornato: [RUNTIME_CPU_INCREMENT_9.md](RUNTIME_CPU_INCREMENT_9.md),
sezione 13.

Ogni falso PASS indicato dalla revisione è stato prima **riprodotto** contro il
gate originale e poi corretto. Su copie temporanee delle evidenze reali, 80
mutazioni: [71 lasciavano PASS](../results/increment9/review/gate-mutations-red.txt)
le condizioni interessate, 9 erano già rilevate. Con il gate corretto
[nessuna passa](../results/increment9/review/gate-mutations-green.txt). Un test
di controllo verifica che la copia non mutata sia PASS in tutte le condizioni:
una mutazione non può quindi risultare respinta per un difetto preesistente.

## Rilievi P2

**P2-1 — il gate passava con evidenze mancanti o vuote. Corretto.**
Riprodotto: senza `parity-stock_fp64.json`, senza `residency_after_eviction` in
`phaseA-P-supervisor.json` (il controllo diventava `0 <= 0`) e con
`integrity-after.json` ridotto a `files: {}`, A–H restava PASS. Contraddiceva la
frase «evidenza mancante = NOT PROVEN, mai PASS».
Correzione: un file o un campo mancante produce NOT PROVEN, un fatto
contraddetto FAIL. Il diagnostico FP64 standard è evidenza obbligatoria: se
manca, F è NOT PROVEN; il suo FAIL grezzo resta riportato come FAIL e non fa
fallire F. La fase A richiede `condition == 'evicted'`, residenza non vuota,
riferita al file del checkpoint e con dimensione uguale al file verificato dal
runtime. `all_match` è ricalcolato dalle voci per file e l'insieme atteso è
obbligatorio: quattro shard Qwen, index e `model.safetensors` di SmolLM2.

**P2-2 — nessun controllo di evizione e letture reali in fase B. Corretto.**
Riprodotto: bastava `worker_read_bytes > 0`; una prova B con shard già in cache
altrove (l'errore W) avrebbe superato C, D e G.
Correzione, per `phaseB-sync` e `phaseB-pipeline`: `condition == 'evicted'`;
residenza dopo l'evizione ≤ 1% per ogni shard, con insieme uguale ai quattro
shard e dimensioni uguali ai file; letture di blocco d'inferenza ≥ 90% dei byte
logici. Valori reali: residenza 0 B; 112.420.581.376 / 113.129.963.520 B in
sync. La regola entra in C e G per entrambe le prove e in D per la prova sync.

**P2-3 — il gate si fidava di booleani derivati e non legava i report.
Corretto.** È la stessa classe di difetto corretta nella revisione 8. Tutte le
mutazioni elencate dalla revisione lasciavano A–H PASS; ora sono respinte:

| Mutazione | Ora | Controllo che la respinge |
|---|---|---|
| `parity-full64` con `failures=1000`, `argmax_equal=false`, `passed=true` | F FAIL | verdetto ricostruito dai confronti; `passed` deve coincidere |
| Copertura ridotta a 1 posizione e 10 logits, solo `decode.0` | F NOT PROVEN/FAIL | fasi uguali al runtime; 8×152.064 logits, 28×8×3.584 residui, KV per prefisso |
| `parity-full64` sostituito dal report FP32 | F FAIL | contratto, dtype, profilo, soglie, digest dell'oracolo nel manifest |
| Report di parità su altro runtime, revisione o sequenza | F FAIL | `runtime_report`, revisione del protocollo, sequenza del runtime |
| `phaseA-P-parity-inc8.json` ridotto a `{"passed":true}` | B, F NOT PROVEN | regola dell'incremento 8 ricalcolata; oracolo e protocollo dell'8 |
| `gate7-rerun.json` / `gate8-rerun.json` ridotti a `{"passed":true}` | H FAIL | gate 7 e 8 ricalcolati dai loro strumenti e confrontati |
| `scope.unit` del runtime diverso dal supervisore | B, C, D, G FAIL | unità, percorso cgroup, PID campionato, file di output del worker |
| SHA di `source_files` azzerato | A FAIL | uguaglianza con i digest LFS ufficiali del preflight |
| `integrity-after` con `match=false` e `all_match=true` | A FAIL | ricalcolo per file |
| `preservation.json` con `core.py` modificato e una cancellazione | H FAIL | contenuto del report e ricalcolo dal vivo con git |
| Contatori per tensore svuotati, `kv_prefix_preserved=false` | D, E FAIL | 339 contatori uguali alle forme ufficiali × 8 posizioni; prefisso, lunghezze, generazioni |
| Soglie o limite alterati in `protocol.json` | A–H FAIL | SHA256 del protocollo e valori approvati fissati nel gate |

Restano rilevate le mutazioni che il gate già respingeva: `oom_kill=1`,
finalizzatore mancante, digest dell'incremento 8 alterato, `scope.required=false`,
SHA di acquisizione alterato. Se presenti in locale, i 54 file di cattura sono
riletti e confrontati con `capture-manifest.json` (in questa esecuzione: tutti
uguali); se assenti, il verdetto lo dichiara.

**P2-4 — «SHA in streaming = LFS» era tautologico; un file esistente era
accettato senza verifica. Corretto per il futuro, dichiarato per il passato.**
Riprodotto: `download()` usciva subito se il file finale esisteva; `acquire()`
registrava come `sha256` il digest atteso copiato; il gate confrontava quel
valore con sé stesso. Nella prova storica la verifica è avvenuta (destinazione
vuota, errore su mismatch prima del rename), ma il digest calcolato non fu
registrato.
Correzione: `tools/acquire_qwen9.py` registra il digest calcolato accanto a
quello ufficiale; ricalcola lo SHA di un file già presente prima di accettarlo;
verifica un `.part` completo senza richiesta HTTP (un Range a fine file
riceverebbe 416); `provenance.json` conserva solo fatti stabili.
`qwen-acquisition.json` **non è stato modificato**. Il gate A non usa più quel
file come prova: la prova è l'integrità successiva, cioè lo SHA integrale del
runtime e i re-hash registrati, confrontati con i digest LFS del preflight. Il
limite della verifica in acquisizione è scritto nei fatti di A.

## Rilievi P3

| # | Rilievo | Esito |
|---|---|---|
| 1 | Tre numeri dell'audit diversi da `environment-before.json` | Corretto: 8.841.621.504, 908.396.462.080, 151.969.796.096 |
| 2 | Parsing fragile di pytest/unittest (`480 passed, 4 errors`, `1 passed`, `NOT OK`) | Corretto: espressioni rigorose, conteggio minimo, ultima riga esattamente `OK`; controlli oracle 8 e precisione 7 ora obbligatori |
| 3 | `memory.peak <= memory.max` tautologico; anon+kernel non vincolato | Corretto: soglia esplicita 268.435.456 B su anon+kernel campionato e VmHWM; emendamento dichiarato successivo alle misure e inasprimento |
| 4 | Swap globale da 684,1 a 760,9 MiB nella finestra degli oracoli | Riportato nel rapporto e nei fatti di G; non vincolato |
| 5 | «Re-hash a fine incremento» non era a fine incremento | Corretto: riformulato; nuovo `integrity-final.json`, ultima lettura dei checkpoint della remediation |
| 6 | «AC1 online=1» senza evidenza | Affermazione rimossa; l'audit registra `power_supply` per il futuro |
| 7 | Il verificatore si fidava di `upstream_greedy_matches_sequence` | Corretto: ricalcolo dall'argmax dell'NPZ e controllo che ogni ingresso di decode sia la scelta greedy precedente |
| 8 | Un `.part` completo bloccava la ripresa | Corretto (P2-4) |
| 9 | «Budget esplicito uguale al piano»; nome file `negative-oom-policy-stop` | Corretto: testo riformulato, il gate verifica `budget_bytes == arena_plan_bytes`; nome file corretto nel README delle evidenze |
| 10 | L'audit registrava tutta la cache Hugging Face dell'utente | Corretto: solo voci del progetto e un conteggio delle altre; voce estranea rimossa dalle evidenze |

Sul rilievo 7: il vecchio fixture del test del verificatore descriveva proprio
una sequenza non greedy accettata. È stato reso coerente; un nuovo test mostra
che una sequenza non greedy con dichiarazione favorevole dell'oracolo ora
fallisce. Il verificatore corretto, rieseguito sulle catture esistenti, produce
report [identici in ogni campo](../results/increment9/review/) a quelli storici,
più il blocco `upstream_greedy`.

## Altre note della revisione

- **Pilota non riportato.** Confermato dal journal utente locale:
  `inwime9-smoke135-1794754652`, 15:57:32–15:57:37, dopo l'emendamento. Con
  altre tre unità di sonda è ora descritto nel rapporto (sezione 2) e
  nell'emendamento come smoke test non conservato.
- **Indipendenza del profilo FP64 puro.** Limite dichiarato nella sezione 8.
- **«56.562.286.592 coefficienti GEMV».** Precisato: totale su 8 posizioni,
  7.070.285.824 per token; il gate lo ricalcola dalle forme ufficiali.
- **`protocol.json` pre-emendamento non verificabile.** Dichiarato nel rapporto
  e nell'emendamento.

## Cambiamenti al protocollo di accettazione

`protocol.json` non è stato toccato: il suo SHA256 è fissato nel gate. I
cambiamenti sono in un file separato,
[protocol-amendment-remediation.json](../results/increment9/protocol-amendment-remediation.json),
che il gate legge insieme al protocollo e i cui parametri devono coincidere con
quelli del codice. Sono nove, R1–R9, tutti inasprimenti o nuove precondizioni,
tutti successivi alle misure. Le soglie numeriche approvate restano FP32
`2e-3 + 2e-4*|ref|` e FP64 `2e-5 + 2e-6*|ref|`; `memory.max` 2.147.483.648 B e
swap 0. L'unica soglia nuova è quella su anon+kernel e VmHWM.

## Evidenze storiche, rigenerate e nuove

Le evidenze di misura non sono state alterate nei numeri né nel significato:
runtime, supervisori, oracoli, parità, guasti, sonde, acquisizione, preflight e
audit. Undici di esse sono state **sanificate** con regole dichiarate.

Report derivati rigenerati, con cancellazione esplicita del precedente:
`summary.json`, `preservation.json`, `pytest-final.txt`, `gate7-rerun.json`,
`gate8-rerun.json`, `oracle9-checks.txt`, `check-oracle8.txt`,
`check-oracle-precision7.txt`. I due rerun dei gate sono identici ai precedenti.
`capture-manifest.json` è stato verificato e non rigenerato.

File nuovi: `protocol-amendment-remediation.json`, `sanitization-manifest.json`,
`integrity-final.json` e la directory `review/` con le prove red/green e i tre
report di parità riverificati.

## Sanificazione

`tools/sanitize9.py` sostituisce soltanto: il percorso assoluto del checkout
con `<repo>`; la directory temporanea di sessione con `<scratch>`;
l'identificativo casuale della sessione di terminale; e rimuove le voci della
cache Hugging Face estranee al progetto. 11 file, 72 sostituzioni, nessun numero
cambiato: i file JSON sono riscritti strutturalmente e ogni valore non testuale
è riportato invariato. Gli originali sono in
`artifacts/increment9/original-evidence/`, esclusa da Git, e coincidono con i
file del commit `6031b52`.
[sanitization-manifest.json](../results/increment9/sanitization-manifest.json)
contiene SHA256 originale e sanificato di ogni file e le regole. Il gate ha una
precondizione: ogni file sanificato deve coincidere con il manifest e, se gli
originali sono presenti, essere esattamente l'originale trasformato dalle sole
regole dichiarate; se sono assenti lo dichiara. Per le prove future
`cgroup9.py`, `audit_increment9.py` e `benchmark_increment9.py` scrivono già
percorsi relativi o `<repo>`.

## Esito

| Verifica | Esito |
|---|---|
| `python3 -m pytest -q tests` | **613 passed**, zero skip, exit 0 (prima 484) |
| `tools.check_oracle9` / `check_oracle8` / `check_oracle_precision7` | OK: 3, 1 e 5 test |
| Gate 7 (`report_increment7 --resolve-numerical-contracts`) | PASS, exit 0; diagnostico FP64 standard FAIL conservato: 270 logits, 41 residui |
| Gate 8 (`report_increment8`) | PASS, exit 0; otto controlli veri |
| Gate 9 (`report_increment9`) | **A–H PASS, exit 0**; precondizioni PASS; diagnostico FP64 standard **FAIL grezzo**: 31 residui |
| Verificatore corretto sulle catture | FP32 PASS exit 0; FP64 puro PASS exit 0; FP64 standard FAIL exit 1 |
| Conservazione rispetto a `68ba9f3` | PASS: solo `README.md` modificato tra i file preesistenti; nessuna cancellazione; staging vuoto |
| Sanificazione | PASS: 11 file, originali presenti e coerenti |
| Re-hash finale dei checkpoint (18,65 GB letti, 2 min 02 s) | tutti i digest uguali a quelli ufficiali, index incluso (`integrity-final.json`) |

## Criticità residue

- **Storia Git.** Le evidenze non sanificate sono nel commit `6031b52` che,
  secondo il riferimento locale, è già su `origin/main`. Rimuoverle richiede una
  riscrittura della storia: decisione del proprietario, non eseguita.
- **Documenti precedenti.** Tre righe con il percorso locale restano in
  `docs/STRATA_TECH_AUDIT_2026-11-13.md` e
  `docs/handoffs/HANDOFF-2026-11-05-benchmark-v2.md`, preesistenti alla base:
  non modificati per non violare la conservazione.
- **Conservazione.** Con l'incremento già committato lo strumento giudica i
  file preesistenti ed elenca a parte gli aggiunti: verdetto invariato, regola
  precisata dopo la revisione.
- **Soglia anon+kernel** scelta dopo le misure; anon e kernel campionati a 50 ms.
- **Verifica SHA in acquisizione** della prova storica non documentata da un
  digest registrato.
- **Nessuna revisione indipendente** delle correzioni e dei 129 test nuovi.

## Riproduzione

Dalla radice, con il venv `.venv-oracle7` già presente. Tutti gli strumenti
rifiutano la sovrascrittura: usare nomi nuovi.

```bash
python3 -m pytest -q tests
.venv-oracle7/bin/python -m unittest -v tools.check_oracle9
.venv-oracle7/bin/python -m tools.check_oracle8
.venv-oracle7/bin/python -m tools.check_oracle_precision7
python3 -m tools.report_increment7 results/increment7/numerical-resolution \
  --resolve-numerical-contracts --output /tmp/gate7-nuovo.json
python3 -m tools.report_increment8 results/increment8 --output /tmp/gate8-nuovo.json
python3 -m tools.integrity9 --output /tmp/conservazione-nuova.json preservation \
  --baseline 68ba9f3adf3cffb89b18b19525644fa0d7c6d806 --intentional README.md
python3 -m tools.sanitize9 results/increment9 --verify
python3 -m tools.report_increment9 results/increment9 --output /tmp/gate9-nuovo.json
python3 results/increment9/review/mutations_green.py
```

P2: **quattro corretti**. P3: **dieci trattati**, nove corretti e uno (swap
globale) riportato senza vincolo. Nessun commit o push eseguito dalla remediation.
