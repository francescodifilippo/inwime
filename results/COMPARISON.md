# Confronto completo RTX 3050 — 5 novembre 2026

Risultati letti dai file `all.csv` e `all.log`, prodotti nel terminale WSL
dell'utente con la libreria `/usr/lib/wsl/lib`. Nessuna nuova esecuzione GPU
è stata effettuata dalla sandbox per questa analisi.

Dataset BF16: 5 GiB, 640 blocchi da 8 MiB; m=1024, k=4096, n=16.
CPU: 8 thread. GPU: RTX 3050 Laptop, compute capability 8.6, un async engine.
Driver API 13.5, runtime CUDA 12.4. Tre ripetizioni per modalità.

Mediane, in millisecondi; ogni colonna è aggregata indipendentemente:

| Modalità | Letture | H2D | Calcolo | D2H | Totale | GiB/s complessivi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solo I/O | 1787,09 | — | — | — | 1787,16 | 2,798 |
| CPU | 1945,92 | — | 900,83 | — | 2864,17 | 1,746 |
| GPU sincrona | 1751,13 | 728,29 | 254,11 | 10,13 | 2823,44 | 1,771 |
| GPU asincrona | 1919,88 | 588,53 | 149,45 | 10,12 | 2081,09 | 2,403 |

La GPU asincrona è 1,376× più veloce della CPU e 1,357× della GPU sincrona
sul tempo complessivo mediano. La GPU sincrona migliora la CPU di appena 1,014×.
H2D mediano: 6,865 GiB/s sincrono, 8,496 GiB/s asincrono; sono misure sotto
carico, non limiti isolati del collegamento. Il percorso I/O resta il contributo
principale. Le durate sovrapposte non sono additive.

Ogni esecuzione CPU/sync/async ha controllato 20.480 risultati con errore
massimo assoluto zero. Sono campioni per blocco, non tutti gli output del dataset.
I piccoli smoke test GPU completi erano già stati superati nel terminale utente.
Il file supera la VRAM fisica: l'esecuzione usa blocchi e non carica l'intero file.
Questo non dimostra l'eseguibilità end-to-end di un LLM.

Limiti della verifica:

- `O_DIRECT` evita la cache dati Linux; le cache Windows/VHD/controller non sono
  controllate. Il test misura lo storage virtuale WSL, non certifica un NVMe specifico.
- Non sono registrati picco RSS, picco VRAM, potenza e temperatura durante il
  benchmark. La VRAM libera iniziale era 3.462.660.096 byte: non è una misura di picco.
- Il volume logico letto per ogni passata misurata è 5.368.709.120 byte. Il warm-up
  legge un ulteriore blocco CPU o due blocchi GPU fuori timing. Il traffico fisico
  verso il dispositivo non è misurato.
- Il totale sincrono varia da 2,483 a 3,368 s; quello asincrono da 2,009 a 2,083 s.
  Senza telemetria non si può attribuire la variabilità a clock, temperatura,
  alimentazione o contesa. Non è stata acquisita una timeline hardware per
  quantificare separatamente la sovrapposizione di copie e kernel.

Per definizioni dei timer e comando riproducibile vedere `README.md`.
