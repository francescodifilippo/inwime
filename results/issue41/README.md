# Gate di #41 — Percorso additivo A: prefill a blocchi e operatori nativi

Stato: **PASS** per bit-identità (fixture in CI e Checkpoint reale in locale) ed equivalenza con
l'oracolo (fixture in CI), sulla configurazione certificata qui sotto. Eseguibilità di un prompt
da 5.317 token: **dimostrata** sul 0.5B, a 4,06 token/s (non entro i 300 s dell'harness: vedi
«Limiti»).
Checkpoint reale: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`
(installato in #38, nessun nuovo download). Base: `main` a `e8a3cfa`. Data: 27 novembre 2026.

Gate **locale** per il Checkpoint reale: le misure vengono da un solo host e non si estendono ad
altre CPU, compilatori, librerie matematiche o numeri di thread. La CI ripete le attestazioni su
fixture.

## Che cosa è cambiato

- `inwime/additive.py`: `AdditiveQwen2Runtime`, il Percorso additivo per Qwen2 (testa tied o
  non tied). Il prompt è elaborato a blocchi di `block` posizioni (32 nel Motore): ogni tile di
  pesi è letta ed espansa una volta per blocco e applicata a tutte le sue posizioni. Norma finale e
  testa girano solo sull'ultima posizione della chiamata (`all_logits` le estende a ogni posizione,
  solo per gate e test). I residual non sono conservati: una traccia opzionale li riceve. La KV ha
  un solo banco: le posizioni si scrivono oltre la lunghezza pubblicata e il commit pubblica la
  nuova lunghezza; errore o cancellazione non pubblicano nulla e la sessione si ferma (fail-stop).
  Troncamento, divergenza del prefisso, i sei scenari e il piano di capacità restano a #42.
  Letture, lease e prenotazioni restano al Gestore delle regioni (`core.Coordinator`), invariato.
- `inwime/ops.py`: sette operatori (`linear`, `rms_denominators`, `rms_scale`, `add`, `rope`,
  `attention`, `silu_mul`) in due implementazioni con la stessa convenzione di chiamata: il
  riferimento Python, che riordina per blocchi i cicli del Percorso congelato senza cambiare
  l'ordine delle operazioni su alcun elemento, e i kernel C via ctypes.
- `src/inwime_ops.c` e l'header versionato `src/inwime_ops.h` (`INWIME_OPS_ABI_VERSION 1`):
  kernel a foglia secondo l'ADR 0002, senza allocazioni, I/O, thread o stato globale. Compilati
  con `-O3 -fno-fast-math -ffp-contract=off -fno-builtin`. Il kernel pinned `src/cpu_kernel.c`
  non cambia. L'attention calcola ogni score due volte, come il Percorso congelato: nessun buffer
  cresce con il contesto.
- `inwime/gate.py`: all'avvio del Motore ogni operatore nativo e il suo riferimento girano sugli
  stessi casi deterministici, e stato e byte di output devono coincidere. I casi coprono tile che
  tagliano righe, più posizioni con stride, GQA 5:1, posizioni fino a 10^6, due basi RoPE,
  underflow di `exp` e parità di score, rami e saturazione della SiLU, subnormali, zeri con segno,
  overflow verso lo stato non finito, flag di cancellazione impostato e geometrie invalide. Un
  operatore che fallisce è sostituito dal riferimento Python; gli altri restano nativi.
  `inwime doctor` riporta il gate (`operator-kernel`).
- `inwime/engine.py`: il Motore serve ogni richiesta con il Percorso additivo e gli operatori
  scelti dal gate; `INWIME_COMPUTE_PATH=frozen` riporta il Motore al Percorso congelato (rollback
  dell'ADR 0001). La cancellazione di un job raggiunge anche i kernel (`CancelSignal`).
- Packaging: seconda estensione opzionale `inwime._ops_kernel`; `make ops-kernel`; la CI controlla
  wheel, sdist, gate del doctor e installazione senza compilatore.

## Attestazione 1 — Bit-identità con il Percorso congelato (fixture, CI)

`tests_inwime/test_additive.py`, nella suite del prodotto e in CI. Fixture M10 (GQA 5:1, eps
osservabile, dimensioni non multiple delle tile), sequenza di 8 posizioni:

- testa non tied contro `Qwen2Runtime` congelato; testa tied contro il congelato sulla gemella
  `duplicated` (#38);
- byte di logits di ogni posizione, di ogni residual e dell'intera KV dopo ogni chiamata;
- chunk `[5,1,1,1]`, `[8]`, `[3,5]`, `[7,1]`; blocchi 2, 3, 4, 8, 32; tile da 7 elementi (taglia
  ogni riga di ogni tensore) e 262.144 (quella del Motore);
- configurazione servita dal Motore (nessuna traccia, logits solo sull'ultima posizione) e
  configurazione con tutti i logits;
- riferimento Python e kernel nativi.

Sensibilità, verificata a mano su questo host:
- gli stessi sorgenti compilati con `-ffp-contract=fast -march=native` (FMA) falliscono tutti i 10
  casi nativi del test, e il gate rifiuta `linear`, `rms_denominators`, `rope` e `attention`;
- con `-ffast-math` il gate rifiuta sei operatori su sette (`add` non ha operazioni riordinabili);
- senza `-fno-builtin` GCC fonde `sin`/`cos` in `sincos`: su questa glibc i bit coincidono e il
  gate passa. Il flag resta, perché la coincidenza non è garantita altrove;
- il gate inizialmente non vedeva l'FMA di `rms_denominators`: il ciclo principale è vettorizzato
  senza contrazione e solo il resto è contratto. È stato aggiunto un caso con larghezza dispari.

## Attestazione 2 — Bit-identità sul Checkpoint reale (locale)

`tools/additive41_motore.py identity` → `identity.json`. Prompt di chat renderizzato dal Motore
(43 token) e completamento greedy del Motore (6 token, `"Il ciclo dell'acqua"`). La stessa
sequenza (un prefill e 6 decode, 49 posizioni) gira in tre sessioni sul catalogo del Motore:

| Confronto | Esito |
|---|---|
| Logits di tutte le 49 posizioni × 151.936, additivo (blocchi da 8, `all_logits`) vs computazione congelata | identici |
| Residual di 24 layer × 49 posizioni | identici |
| KV intera dopo ogni chiamata | identica |
| Configurazione servita (blocchi da 32, solo ultima posizione): logits finali di ogni chiamata | identici |
| Configurazione servita: KV dopo ogni chiamata | identica |

Digest SHA256 in `identity.json`. La computazione congelata è `Qwen2Runtime` invariato sul
catalogo tied di #38 (`CatalogQwen2Runtime`): il Percorso congelato non accetta teste tied, e la
bit-identità tra testa tied e duplicata è l'attestazione di #38. Kernel: `c-fp64-ordered` per
l'espansione BF16 e `c-fp64-ordered-ops` per tutti gli operatori (gate `ok`).

## Attestazione 3 — Equivalenza con l'oracolo (fixture, CI)

`tools/additive41_oracle.py` (ambiente separato `.venv-oracle7`, nessun import di InWiMe) ha
calcolato offline con Transformers 4.57.3 e torch 2.9.1+cpu ogni logit, residual e voce K/V sulle
fixture tied e non tied, nei contratti `stock_fp32` e `full64`, al commit `7d8052d` con albero
tracciato pulito. `tests_inwime/reference/additive41_oracle.json` registra provenienza, SHA256 di
ogni file delle fixture e dei valori canonici. `tests_inwime/test_oracle_reference.py` controlla
i digest, rigenera le fixture e confronta in CI il Percorso additivo (Python e nativo) su ogni
elemento, alle soglie M10 (fp32 atol 2e-3 rtol 2e-4, fp64 atol 2e-5 rtol 2e-6), senza
Transformers né PyTorch. Utilizzo massimo della soglia: `stock_fp32` 0,8% (residual), `full64`
sotto 1e-6. Un confronto con i valori della fixture sbagliata fallisce.

## Memoria, I/O e tempi

| Misura | Identità (49 posizioni) | Prompt lungo (5.317 token) |
|---|---|---|
| RSS di picco (`/usr/bin/time`) | 408.732 kB | 485.000 kB |
| Piano riservato per sessione | additivo 16,8 MB; congelato 76,7 MB | additivo 270,0 MB, di cui KV 261,4 MB |
| Attivazioni del piano | — | 3.637.248 B (32 posizioni), indipendenti dal prompt |
| Logits nel piano | — | 1.215.488 B (una posizione) |
| Piano del Percorso congelato, **calcolato, non eseguito** | — | 7.906.171.281 B, di cui logits 7.378.831.360 B |
| Letture logiche di pesi | additivo 11,86 GB; congelato 48,42 GB | prefill 119,8 GB (~716 MB per blocco da 32) |
| Letture dal file system (blocchi da 512 B) | 0 | 56 |
| Wall time | congelato 44,5 s; additivo 17,2 s (`all_logits`) | prefill 1.309 s (4,06 token/s); decode 1,19 s |

Il congelato rilegge tutti i pesi a ogni posizione; l'additivo una volta per blocco, tranne la
testa, letta solo per l'ultima posizione. Con 5.317 token il Percorso congelato riserverebbe
7,9 GB, quasi tutti di logits FP64 per l'intera capacità, contro i ~10 GiB di RAM della VM WSL:
non è stato eseguito.

**Cache calda.** Le letture vengono dalla cache delle pagine: il Checkpoint era già stato letto
(56 e 0 blocchi letti dal file system). Non sono misure a freddo. Temperatura e alimentazione
**non misurate**. Un solo thread. Memoria prima dei run: `free-before.txt` e
`free-before-long.txt` (la VM aveva già ~4 GB di swap in uso da altri processi). Durante il run
lungo, per circa un minuto, sono girati anche alcuni file di test della suite del prodotto (load
average ~1,2 su 20 CPU logiche): il tempo del prefill è indicativo, non un benchmark. Le misure di
prefill e decode con stato termico dichiarato e cache fredda sono di #43.

## Configurazione certificata

- CPU: 12th Gen Intel Core i7-12700H, 20 CPU logiche; WSL2, kernel `6.18.58.1-microsoft-standard-WSL2`.
- Compilatore: GCC 15.2.0 (Ubuntu 15.2.0-16ubuntu1); glibc 2.43 (Ubuntu 2.43-2ubuntu2.7).
- Python 3.14.4; flag di `make ops-kernel` e `make cpu-kernel`; un thread.
- La CI ripete gate e attestazioni 1 e 3 su Ubuntu 24.04 con Python 3.11, 3.12 e 3.14, sulla
  libreria compilata da setuptools e la confronta con quella di `make ops-kernel`. Il confronto
  tra le due build non è stato ripetuto in locale.

## Limiti

- 4,06 token/s su un thread: un prompt di 5.317 token richiede ~22 minuti, oltre i 300 s dopo cui
  DeepSeek Harness abbandona la richiesta. Il multithreading per righe è #43.
- La cancellazione dentro i kernel è controllata da `linear` (a ogni segmento di riga di una tile)
  e `attention` (a ogni coppia posizione, head). Gli altri cinque operatori sono lineari nel
  blocco e non si interrompono a metà; il runtime controlla prima di ogni operatore, lettura e
  layer.
- L'attention ricalcola gli score invece di tenerli in un buffer lungo quanto il contesto: costa
  circa un terzo in più di prodotti, a vantaggio di una memoria temporanea indipendente dal
  contesto.
- KV a banco singolo con commit per lunghezza; troncamento e riuso del prefisso sono di #42.

## Riproduzione

    make cpu-kernel ops-kernel
    PYTHONPATH=. .venv-product/bin/pytest -q tests_inwime
    PYTHONPATH=. INWIME_CPU_KERNEL=build/libinwime_cpu.so INWIME_OPS_KERNEL=build/libinwime_ops.so \
        /usr/bin/time -v .venv-product/bin/python -m tools.additive41_motore identity --output results/issue41/identity.json
    PYTHONPATH=. INWIME_CPU_KERNEL=build/libinwime_cpu.so INWIME_OPS_KERNEL=build/libinwime_ops.so \
        /usr/bin/time -v .venv-product/bin/python -m tools.additive41_motore long --output results/issue41/long.json
    .venv-oracle7/bin/python -m tools.additive41_oracle --output tests_inwime/reference/additive41_oracle.json

Il run lungo viene dal codice di `d4a1316`; il run di identità da `7d8052d` con la correzione di
`tools/additive41_motore.py` inclusa nel commit di queste evidenze. Le modifiche di `7d8052d` agli
operatori aggiungono solo il rifiuto delle geometrie invalide, prima di ogni calcolo.
