# Evidenze incremento 7

**Parziale.** `summary.json` mantiene il FAIL del confronto FP64 standard;
`python3 -m tools.report_increment7 results/increment7` termina con codice 1.
Il PASS FP32 e quello diagnostico non sostituiscono quel criterio.

- `provenance.json`, `repack.json`, `environment.json`: origine, integrità,
  formato e ambiente, senza pesi.
- `final-*.json`: confronti finali completi per layer, logits, KV, top-10,
  greedy, invarianti di sessione e digest degli array.
- `first-operator.json`, `diagnostic-rms64-final-parity.json`: isolamento del
  cast FP32 di RMSNorm e intervento diagnostico esplicito.
- `format-and-modes.json`: identità esatta originale/shard in Transformers e
  identità originale/shard/sync/pipeline in InWiMe.
- `measured-*.json`: sei misure finali, una per processo, cache OS calda.
- `pytest-final.txt`: 374 pass, zero skip. `baseline-preservation.json` verifica
  i moduli baseline byte per byte. Le prove upstream sono un gate separato.
- `capture-manifest.json`: hash e schema delle catture complete locali NPZ in
  `artifacts/increment7/`, non versionate. Ogni cattura si rigenera con
  `tools/oracle7.py`; non è una cache dei pesi.
- `generation-*.json`: demo testuali riproducibili, EOS non raggiunto.

I file senza prefisso `final`, `measured` o suffisso `reviewed` possono essere
prove preliminari. `adapter-tests.txt` conserva il primo errore del test di
capacità, poi corretto. `oracle-float64.log` appartiene alla prima cattura con hook
hidden errato; non viene usato come evidenza di parità. La cattura corretta è
`upstream-float64`. Il primo diagnostico RMS64 è superato dal diagnostico finale
con flag riproducibile. `benchmark-sync-1.json` è lo smoke preliminare e non entra
nelle mediane. Questi risultati non sono campagne cold o prove oltre RAM fisica.

Rapporto e comandi: [RUNTIME_CPU_INCREMENT_7.md](../../docs/RUNTIME_CPU_INCREMENT_7.md).
