# Verifica del benchmark — 5 novembre 2026

Implementazione compilata con GCC 13, `-O3 -march=native -fopenmp`, CUDA Runtime
e cuBLAS del Toolkit 12.4. CPU visibile: Intel Core i7-12700H; 8 thread usati.
Nessuna installazione, modifica di configurazione o download di modelli.

`make test`: PASS. Confronto di tutti gli output CPU con riferimento FP64 su
tre geometrie, più verifica dei percorsi di errore. I test GPU non sono stati
eseguiti: richiedono `python3 tests/smoke.py --gpu` su un dispositivo accessibile.

Dataset principale: 5.368.709.120 byte (5 GiB), 640 blocchi da 8 MiB,
`m=1024, k=4096, n=16`. Input BF16, output/accumulo FP32. Tre ripetizioni.
Le medie non sono usate: la tabella mostra le mediane delle singole colonne.

| Modalità | Letture ms | Letture GiB/s | Calcolo ms | Totale ms | Throughput totale GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Solo I/O diretto | 1954,655 | 2,558 | — | 1954,730 | 2,558 |
| CPU | 2296,333 | 2,177 | 973,607 | 3272,187 | 1,528 |
| GPU sincrona | non misurato | non misurato | non misurato | non misurato | non misurato |
| GPU asincrona | non misurato | non misurato | non misurato | non misurato | non misurato |

Ogni esecuzione CPU ha verificato 20.480 valori, errore massimo assoluto 0.
CSV completi: `cpu-5GiB.csv` e `io-5GiB.csv`; i file `cpu.csv` e `io.csv`
contengono la prova preliminare da 512 MiB.

Il filesystem è esposto come `/dev/sdd`, modello `Virtual Disk` in WSL2.
`O_DIRECT` evita la cache dati Linux; il risultato include il percorso VHD e
non certifica la banda fisica di un NVMe specifico né esclude cache lato Windows.
Il dataset supera la VRAM indicata dall'utente, mentre i buffer restano limitati
alla dimensione di due blocchi; non è stata verificata questa proprietà a runtime GPU.

La sessione non espone `/dev/dxg`; `nvidia-smi` fallisce con errore NVML e il
runtime restituisce `CUDA: no CUDA-capable device is detected` (vedere `gpu.log`).
La policy degli strumenti disabilita l'esecuzione esterna alla sandbox anche
dopo l'autorizzazione dell'utente. Non sono quindi disponibili misure H2D, D2H,
GEMM GPU o speedup, né una verifica effettiva della sovrapposizione hardware.
La compilazione della parte CUDA è riuscita, ma la correttezza dei due percorsi
GPU non era verificabile all'interno della sessione.

Aggiornamento dal terminale WSL dell'utente: `tests/smoke.py --gpu` è passato
con `LD_LIBRARY_PATH` che antepone `/usr/lib/wsl/lib`. Output comunicato:
`PASS: full reference checks, block reuse/tails, corruption, truncation, CLI validation`
e `GPU modes verified`. Entrambi i percorsi GPU hanno quindi superato i piccoli
test numerici sul dispositivo; le misure GPU su 5 GiB e la sovrapposizione
hardware rimangono da raccogliere. L'esito positivo è riferito dall'utente,
non da un'esecuzione nella sandbox.

Per completare la verifica in WSL2 con la RTX accessibile, dalla root del progetto:

```bash
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/smoke.py --gpu
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./build/bf16_stream --file data/weights-5GiB.bf16 --blocks 640 --mode all \
  > results/all.csv 2> results/all.log
```

Il README descrive i tempi inclusi/esclusi e le limitazioni delle misure.
