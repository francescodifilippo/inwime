# Incremento 7 — risoluzione numerica FP64

**Gate coerente PASS**, `final-gate.json`, exit 0. Il gate originario resta FAIL;
`../summary.json` e ogni evidenza storica sono immutati.

Tre contratti, tutti eseguiti e tutti riportati:

1. Transformers standard FP32: parità completa PASS.
2. PyTorch realmente FP64: RMSNorm, inizializzazione/forward RoPE e softmax FP64
   nel solo processo oracolo; parità completa PASS.
3. Transformers standard con parametri FP64: diagnostico mixed precision,
   verdetto grezzo FAIL conservato, non saltato. Il cambiamento del criterio
   è esplicito nel rapporto e nel gate. Nessuna tolleranza cambiata.

- `precision-audit.json`: codice effettivo e versioni, dtype/valori di ogni stadio
  RMSNorm standard e FP64, locali reali di InWiMe, confronto Decimal 80 cifre,
  primo operatore e confronto ai confini di ogni layer. Le catture preliminari
  `operators*.json` precedono l'aggiunta dei locali/dtype dettagliati.
- `attribution.json`: ablation RMSNorm → RoPE → softmax, scarti singolo token e
  completi, controllo bit per bit delle nuove catture standard con quelle storiche.
- `parity-*.json`: tutti i logits, hidden, KV e decode confrontati per scenario;
  massimi, RMS, relativi, per-layer, top-10, argmax e invarianti. `initial` è il
  primo smoke FP64 puro, seguito dalle catture finali con audit su tutti gli step.
- `capture-manifest.json`: hash e metadata delle catture complete locali NPZ in
  `artifacts/increment7/numerical-resolution/`, escluse da Git. Nessun peso nelle
  evidenze pubblicabili. L'oracolo continua a non importare InWiMe.
- `preservation-before.json`, `preservation-after.json`: SHA di runtime, checkpoint,
  librerie e risultati storici, invariati.
- `pytest-final.txt`: **382 pass**, zero skip (374 precedenti + 8 del gate).
- `oracle-precision-tests.txt`: **5 pass** nel solo venv torch, incluso ripristino
  dell'oracolo dopo eccezione e audit dtype in prefill/decode.
- `gate-tests-red.txt`, `gate-tests-green.txt`: test del nuovo contratto prima/dopo
  l'implementazione. Il red iniziale documenta la funzione gate ancora assente;
  la divergenza numerica storica è riprodotta separatamente nel report standard.

Riproduzione del gate (output nuovo obbligatorio):

```bash
python3 -m tools.report_increment7 results/increment7/numerical-resolution \
  --resolve-numerical-contracts --output /tmp/inwime-numerical-gate-new.json
.venv-oracle7/bin/python -m tools.check_oracle_precision7
python3 -m pytest -q tests
```

Diagnosi e ricette oracle/verify:
[RUNTIME_CPU_INCREMENT_7.md](../../../docs/RUNTIME_CPU_INCREMENT_7.md).
Nessuna modifica al runtime, al checkpoint o alle librerie installate. Nessuna
misura prestazionale nuova, nessun commit/push o incremento 8.
