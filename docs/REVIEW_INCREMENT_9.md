# Revisione dell'incremento 9 e remediation

> **Stato attuale (17 novembre 2026).** Il follow-up della seconda revisione è
> pubblicato nel commit `5eb25da` («incremento 9 review»); la revisione
> conclusiva lo ha giudicato *READY WITH DOCUMENTED LIMITATIONS*. M9-A e M9-B
> sono chiusi con limitazioni documentate; le Issues #2 e #3 sono chiuse, le
> attività residue sono nelle Issues #17, #18 e #19. Le frasi che seguono su
> commit, staging e «non committato» descrivono lo stato **al momento della
> stesura** e restano come testo storico. L'hardening del gate successivo alla
> chiusura (D-1…D-6) è descritto in
> [results/increment9/post-closure/README.md](../results/increment9/post-closure/README.md).

16 novembre 2026. **La remediation è nel commit `eb669f2` («passo 9 fix»), già
su `origin/main` secondo il riferimento locale. Il follow-up della seconda
revisione indipendente, descritto più sotto, non è committato** (stato del 16
novembre; poi pubblicato in `5eb25da`). La prima
revisione indipendente dell'incremento 9 non ha trovato P0 né P1.
Ha trovato quattro P2, tutti nel gate delle evidenze e nello strumento di
acquisizione, e dieci P3. Questa remediation li corregge. Runtime, adapter
Qwen2, kernel e matematica sono invariati; nessuna misura è stata ripetuta.
[Gate A–H ricalcolato](../results/increment9/summary.json): PASS.

Perimetro: base `68ba9f3adf3cffb89b18b19525644fa0d7c6d806`. All'inizio della
remediation l'incremento 9 era già nel commit `6031b52` («passo 9») e il
riferimento locale `origin/main` puntava allo stesso commit. Non sono stati
eseguiti forward, oracoli, benchmark, download, commit, staging, push, fetch o
reset da parte della remediation; il commit `eb669f2` è successivo. Rapporto aggiornato: [RUNTIME_CPU_INCREMENT_9.md](RUNTIME_CPU_INCREMENT_9.md),
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
quelli del codice. Sono dieci, R1–R10, tutti successivi alle misure. R1–R9
sono inasprimenti o nuove precondizioni. **R10 non è un inasprimento**: è la
correzione semantica della regola di conservazione, fatta dopo le misure. Con
la regola originale (`git diff --name-only` rispetto alla base, tutto giudicato)
l'albero committato fallirebbe: 75 file «inattesi» al commit `6031b52`, 90 al
commit `eb669f2`, cioè i file aggiunti dall'incremento stesso. La regola attuale
giudica solo i file preesistenti alla base ed elenca a parte gli aggiunti.

Le soglie numeriche approvate restano FP32 `2e-3 + 2e-4*|ref|` e FP64
`2e-5 + 2e-6*|ref|`; `memory.max` 2.147.483.648 B e swap 0. Le soglie nuove,
nessuna preregistrata e tutte scelte conoscendo già i valori misurati, sono:

- anon+kernel campionato ≤ 268.435.456 B (R6). Massimo misurato 110.895.104 B,
  margine circa 2,4×, su campioni ogni 50 ms. La parte su VmHWM non aggiunge
  nulla: è implicata da RLIMIT_AS = 256 MiB del worker;
- residenza dopo l'evizione ≤ 1% per shard e letture di blocco ≥ 90% dei byte
  logici in fase B (R5). Misurati: 0 B e 99,4%;
- minimi di test 613 (pytest) e 3, 1, 5 (unittest) (R7): sono i conteggi
  osservati quando la remediation fu scritta.

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
regole dichiarate; se sono assenti lo dichiara. Dopo la seconda revisione il
gate fissa nel codice i nomi e i digest degli undici file: il manifest non può
più restringere da solo ciò che viene confrontato. Per le prove future
`cgroup9.py`, `audit_increment9.py` e `benchmark_increment9.py` scrivono già
percorsi relativi o `<repo>`.

## Esito della remediation (commit `eb669f2`)

Valori della remediation, prima del follow-up; quelli attuali sono nella
sezione successiva.

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

## Seconda revisione indipendente e follow-up

La seconda revisione ha esaminato la remediation, ora nel commit `eb669f2`
(«passo 9 fix»). Ha trovato due P2 e alcuni P3. Il follow-up li tratta senza
ripetere misure, senza toccare runtime, kernel, sorgente, evidenze di misura,
`protocol.json` o evidenze e strumenti degli incrementi 1–8. **Il follow-up non
è committato** (stato al momento della stesura; poi pubblicato in `5eb25da`). I tredici aggiramenti sono stati riprodotti contro il gate di
`eb669f2` ([red](../results/increment9/review/bypasses-followup-red.txt): 13 su
13 passavano o mandavano in crash il gate) e sono chiusi dal gate attuale
([green](../results/increment9/review/bypasses-followup-green.txt): 0 su 13).

| Rilievo | Esito |
|---|---|
| P2-A — la precondizione di sanificazione si aggirava togliendo voci dal manifest (con `files: {}` il verdetto era PASS) | **Corretto.** Il gate fissa nel codice i nomi degli undici file sanificati e, per ciascuno, SHA256 originale e sanificato. Manifest vuoto: FAIL, mai «present». Ogni originale presente in locale deve essere elencato e coincidere. Voce rimossa, manifest vuoto o digest riscritto: A–H FAIL, con e senza originali |
| P2-B — il testo sottostimava i percorsi personali rimasti nel repository | **Testo corretto, file non toccati** (vedi «Criticità residue») |
| P3-1 — registro degli emendamenti | **Corretto.** Aggiunto R10 per il cambio della regola di conservazione, etichettato «non è un inasprimento»; R5, R6 e R7 etichettati come scelti dopo le misure; in R6 la parte VmHWM è dichiarata implicata da RLIMIT_AS |
| P3-2 — una precondizione fallita o un campo mancante declassava un FAIL a NOT PROVEN | **Corretto, con un limite dichiarato.** Precondizioni ed evidenze mancanti cambiano soltanto un PASS; `oom_kill=1` resta FAIL anche senza manifest o con un campo mancante nello stesso supervisore |
| P3-3 — crash su JSON non valido e su `inference_read_bytes = 0` | **Corretto.** JSON non valido in protocollo, emendamento, manifest o evidenze: FAIL; file mancante: NOT PROVEN; divisione per zero: FAIL |
| P3-4 — voce dell'index ridotta a `{"match": true}` | **Corretto per `integrity-final.json`**: `sha256` ed `expected` obbligatori e uguali al digest dell'index registrato in acquisizione, da cui deriva `index_sha256` del descrittore. I `computed_sha256`, se presenti, devono essere digest di 64 cifre esadecimali uguali a quelli ufficiali |
| P3-8 — testo non aggiornato | **Corretto**: stato dei commit e formulazioni assolute su «mai PASS» / fail-closed |
| In più, correzioni di una riga | `resident_bytes` negativo non vale come evizione; `oom_kill: false` non vale come 0 |

Ogni condizione riporta ora anche `verdict_before_preconditions`, il verdetto
sulle sole evidenze proprie: i test delle 80 mutazioni lo controllano, così una
mutazione di un file sanificato non risulta respinta solo perché cambia il
digest fissato.

**Dichiarato, non corretto:**

- i quattro digest LFS degli shard sono ancorati soltanto tramite
  `qwen-preflight.json`: il gate non li fissa nel codice;
- i contatori `failures` dei report di parità sono presi dai report: li protegge
  un test del verificatore, non un ricalcolo dalle catture nel gate;
- `generated_text` non è legato ai token della continuazione;
- l'evizione è legata ai file del checkpoint solo per suffisso del percorso;
- il parsing di pytest non conta i punti delle righe di avanzamento;
- la conservazione non copre il codice aggiunto dall'incremento 9 (per esempio
  `inwime_runtime/qwen2.py`) e non rileva i flag git `--assume-unchanged` e
  `--skip-worktree`;
- `repository_path` di `tools/benchmark_increment9.py` lascia assoluto un
  percorso esterno al checkout;
- mancano test dell'acquisizione per un'interruzione a metà flusso e per un
  server che ignora `Range`;
- `minimum_pytest_passed` (613) è accoppiato al numero di test del momento in
  cui fu scritto e non segue la suite (oggi 651);
- il gate legge un solo file di emendamento, con nome fissato nel codice, e non
  ne fissa lo SHA256: lo vincolano i parametri, non il testo;
- la CLI del gate non ha un'opzione `--root`: va eseguita dalla radice;
- `integrity-after.json`, storico, registra per l'index solo l'esito: il digest
  dell'index è richiesto soltanto in `integrity-final.json`;
- la regola «un fatto contraddetto resta FAIL» vale per i controlli già
  registrati quando si incontra il campo mancante e, dentro una prova
  supervisionata, per tutti i controlli dello scope; le condizioni che
  calcolano i controlli in un'unica espressione (C, D, parti di E e F) possono
  ancora dare NOT PROVEN per la combinazione campo mancante più fatto
  contraddetto. In nessun caso danno PASS;
- `tools.sanitize9 --verify` da solo non conosce l'insieme fissato: senza
  originali rileva un manifest vuoto ma non una voce rimossa. Il gate sì.

| Verifica del follow-up | Esito |
|---|---|
| `python3 -m pytest -q tests` | **651 passed**, zero skip, exit 0 (613 più 38 regressioni della seconda revisione) |
| `tools.check_oracle9` / `check_oracle8` / `check_oracle_precision7` | OK: 3, 1 e 5 test |
| Gate 7 e gate 8 | PASS, exit 0; output identici ai rerun conservati; diagnostico FP64 standard del 7 FAIL conservato |
| Gate 9 | **A–H PASS, exit 0**; precondizioni PASS, originali presenti; diagnostico FP64 standard **FAIL grezzo**: 31 residui |
| Conservazione rispetto a `68ba9f3` | PASS: solo `README.md` modificato tra i file preesistenti; nessuna cancellazione; staging vuoto |
| `mutations_green.py` (80 mutazioni) e `bypasses_followup.py` (13 aggiramenti) | 0 falsi PASS; 0 aggiramenti |

Report derivati rigenerati dal follow-up, con cancellazione esplicita del
precedente e nell'ordine del README delle evidenze: `preservation.json`,
`pytest-final.txt`, `summary.json`. Non rigenerati perché i loro ingressi non
sono cambiati: `gate7-rerun.json`, `gate8-rerun.json` (riesecuzione identica) e
i tre output dei controlli oracle (rieseguiti, OK). Modificato:
`protocol-amendment-remediation.json`. File nuovi: `review/bypasses_followup.py`
e i due output red/green. Nessun checkpoint è stato riletto.

## Criticità residue

- **Storia Git.** Le evidenze non sanificate dell'incremento 9 sono nel commit
  `6031b52` che, secondo il riferimento locale, è già su `origin/main`.
  Rimuoverle richiede una riscrittura della storia: decisione del proprietario,
  non eseguita.
- **Percorsi personali nel resto del repository.** La remediation scriveva «tre
  righe in documenti precedenti»: era una sottostima. Riconteggio su `HEAD`
  (`git ls-files` più `grep`): **309 file tracciati** contengono il percorso
  home locale, **548 occorrenze**, tutte in file che non sono cambiati dopo la
  base `68ba9f3`. Per area: `docs/` 2 file (3 occorrenze), `results/increment4`
  35, `results/increment5` 15, `results/increment6` 153, `results/increment7` 4,
  `results/increment8` 8, `results/m1a-native-ext4-*` 14, `results/v2` 78.
  Inoltre il nome di un account Windows compare in **10 file** (11 occorrenze):
  `BENCHMARK_V2.md`, `tools/report_v2.py`, `tools/storage_setup.py`,
  `tools/storage_cleanup.py` e sei file sotto `results/v2/`. I nomi non sono
  riportati qui. **Riscrivere la sola storia dell'incremento 9 non li
  rimuoverebbe**: sono nei commit precedenti e nel working tree. Sanificare le
  evidenze degli incrementi 4–8 e della V2 è una **decisione del proprietario,
  in sospeso**, perché tocca la regola di conservazione degli incrementi 1–8;
  questi file non sono stati modificati.
- **Conservazione.** La regola è stata cambiata dopo le misure (R10): non è un
  inasprimento. Non copre i file aggiunti dall'incremento 9.
- **Soglie scelte dopo le misure**: anon+kernel (campionato a 50 ms), 1% di
  residenza, 90% di letture di blocco, minimi di test.
- **Verifica SHA in acquisizione** della prova storica non documentata da un
  digest registrato.
- **Limiti del gate dichiarati dalla seconda revisione**: elenco nella sezione
  precedente.
- **Nessuna revisione indipendente** del follow-up e dei suoi 38 test (stato
  al momento della stesura: la revisione conclusiva del 17 novembre lo ha poi
  esaminato, con esito *READY WITH DOCUMENTED LIMITATIONS* e sei P3, D-1…D-6).

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
python3 results/increment9/review/bypasses_followup.py
```

Prima revisione — P2: **quattro corretti**; P3: **dieci trattati**, nove
corretti e uno (swap globale) riportato senza vincolo. Seconda revisione — P2-A
corretto, P2-B corretto nel testo, P3-1/2/3/4/8 corretti con i limiti
dichiarati sopra. La remediation è nel commit `eb669f2`; il follow-up non ha
eseguito commit, staging, push, fetch o reset ed è stato poi pubblicato dal
proprietario nel commit `5eb25da`.

## Hardening successivo alla chiusura (17 novembre 2026)

Correzioni mirate dei P3 D-2, D-3 e D-4 della revisione conclusiva (Issue #17),
riallineamento documentale D-1 e attestazione D-6 (Issue #18). Nessuna misura
ripetuta, nessuna evidenza di misura modificata, runtime e kernel invariati.
Dettaglio, confronto tra gate e prove red/green in
[results/increment9/post-closure/README.md](../results/increment9/post-closure/README.md).

- **D-2** — verdetto finale = il peggiore tra verdetto proprio e precondizioni
  (FAIL > NOT PROVEN > PASS). Una precondizione FAIL rende FAIL anche una
  condizione NOT PROVEN sulle proprie evidenze; prima restava NOT PROVEN. Nel
  protocollo, un `protocol.json` diverso dal file approvato è FAIL anche se
  manca l'emendamento.
- **D-3** — i contatori di runtime devono essere interi JSON non negativi:
  `false`, `0.0`, `-1`, float, `null` e stringhe numeriche non soddisfano più
  alcuna condizione (prima, per esempio, `VmSwap_bytes=false`,
  `VmSwap_bytes=0.0`, `VmPeak_bytes=-1` e `VmHWM_bytes=-1` lasciavano A–H PASS).
- **D-4** — un manifest di sanificazione presente ma strutturalmente non valido
  (`files: []`, `files: null`, `files` assente, JSON `null` o lista) è FAIL;
  il manifest assente resta NOT PROVEN.
- Sulle evidenze storiche il gate aggiornato dà lo stesso report di
  `summary.json`, campo per campo: A–H PASS, diagnostico FP64 standard FAIL
  grezzo con 31 residui.
