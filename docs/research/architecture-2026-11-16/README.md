# InWiMe — Audit architetturale Fable 5.2 (16–17 novembre 2026)

Questa directory archivia **ricerca e prototipi sperimentali**, non codice di produzione né evidenze di un nuovo gate PASS.

## Contenuto

- `ARCHITECTURE_REVIEW_2026-11-16.md`: rapporto di revisione indipendente, 10 sezioni e appendici, redatto su codice a `eb669f2` mentre il follow-up M9 era ancora locale; **i riferimenti allo stato del Git sono storici** e vanno letti alla data dell'audit.
- `appendix/e8lib.c`: prototipo AVX2 GEMV BF16→FP64 fuso, 8 righe in parallelo.
- `appendix/compare_with_repo_kernel.py`: verifica bit per bit contro `build/libinwime_cpu.so` su quattro geometrie sintetiche.
- `appendix/kbench.c`, `appendix/kbench2.c`: microbenchmark GEMV/GEMM in memoria.
- `appendix/micro_python.py`: micro-misure dei loop Python per copia KV e attenzione.
- `appendix/outputs.txt`: log **originario** delle misure Fable; nessun benchmark è stato ripetuto per questa importazione.

## Evidenze e limiti

- Dati sintetici in memoria, i7-12700H sotto WSL2, host non isolato e senza controllo termico: **non** sono prestazioni end-to-end del runtime.
- Il kernel E8 ha prodotto output bit-identici al kernel compilato nelle quattro geometrie documentate; **non** è una dimostrazione universale né una validazione su checkpoint addestrati.
- Il prototipo E8 non include ancora tutti i contratti del runtime (controllo BF16 non finiti, gestione delle righe interrotte fra tile, integrazione in budget/lease/fault handling). Non sostituisce il kernel corrente.
- `compare_with_repo_kernel.py` usa `os.urandom` per i pesi sintetici: i digest possono cambiare da un'esecuzione all'altra; l'uguaglianza dei digest va valutata **dentro la stessa esecuzione**.
- Le stime su throughput, prefill, MoE e cache nel rapporto sono classificate esplicitamente PROVEN / HIGH CONFIDENCE / HYPOTHESIS / REQUIRES EXPERIMENT.
- Il rapporto originale era fuori dal checkout; in questa copia è stato sostituito **un solo percorso locale personale** con un segnaposto. I risultati numerici e gli allegati sono preservati.
- I sorgenti `appendix/` sono scratch di ricerca, non vengono compilati dal Makefile e non devono entrare nel percorso di produzione senza review, test di parità e gate.

## Riproduzione locale opzionale

Consultare i comandi in cima a `compare_with_repo_kernel.py` e nel log `appendix/outputs.txt`. Compilare ed eseguire soltanto in un ambiente di test, senza cambiare la libreria di produzione e senza scaricare checkpoint.

**Filoni collegati**: [#6 — M12 prestazioni](https://github.com/francescodifilippo/inwime/issues/6), [#11 — ricerca Betelgeuse](https://github.com/francescodifilippo/inwime/issues/11), [#12 — studio dei backend](https://github.com/francescodifilippo/inwime/issues/12). La documentazione architetturale canonica e i gate M9 restano indipendenti da questa importazione.
