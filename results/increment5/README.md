# Evidenze incremento 5

Acquisizioni nuove, 12–13 novembre 2026. Base `8419dcfffebeb86f0c76e07d6cbac7a3732c9765`.
Nessun artefatto degli incrementi precedenti viene modificato.

- `preflight.json`, `resource-audit.json`: memoria, limiti e spazio prima della generazione.
- `generation.json`: scrittura fisica del nuovo decoder denso originale BF16 da circa 12 GiB.
- `first-small-sync.json`: prima verifica del worker su fixture originale.
- `large-sync-1.json`, `large-pipeline-1.json`: prefill di due token e tre decode,
  con feedback dei logits. Le verifiche di righe indipendenti sono successive al timing.
- `large-summary.json`: confronto con uguaglianza bitwise degli output, un campione per modo.
- `gemv-6g/`: GEMV completa sul tensore storico da 6 GiB, non un decoder.
- `gemv-reduction/`: GEMV 128 MiB con tile 8191, inferiori alla riduzione di 16384.
- `small/`: tre processi freschi per modo sul decoder `inc4-dense`, ordine alternato.
- `final-run.log`: comandi delle ultime acquisizioni e suite finale, 181 test superati.
- `artifact-sha256-final.json`: identificazione di sorgenti finali, binario, manifest ed evidenze.

Il rapporto completo è `docs/RUNTIME_CPU_INCREMENT_5.md`. Il modello grande
supera sia il limite del processo sia la RAM WSL con i pesi realmente usati.
Non supera la RAM fisica dell'host. Il controllo numerico indipendente grande
riguarda 507 righe delle 169 matrici; non è un oracle integrale del decoder grande.

La coppia decoder grande precede le ultime protezioni del wrapper C contro
resize e alias; kernel matematico e checkpoint restano identici. Le acquisizioni
GEMV/media e i test finali usano le protezioni completate. I risultati non vengono
riscritti o attribuiti retroattivamente al wrapper finale.

I tempi provengono da letture buffered con DONTNEED consultivo. Cache Windows,
dispositivo, temperatura e potenza non sono controllati. Il guadagno wall osservato
del decoder grande non è attribuibile interamente alla pipeline: il servizio I/O
cambia molto più dell'overlap misurato. GEMV e fixture media mostrano regressioni.

Checkpoint e output numerici giganti non sono inclusi qui. Gli shard originali
rimangono locali e sono identificati dai manifest e dai relativi SHA256.
