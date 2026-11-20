# Evidenze dell'emendamento 007 dell'incremento 10

Emendamento normativo: [../amendments/007-gate-diagnostic-7b-equivalence.json](../amendments/007-gate-diagnostic-7b-equivalence.json).
Diagnostica 7B `stock_fp64` eseguita il 20/11 (`campaign/O7-stock_fp64-a3-*`,
`campaign/equivalence-7b-e2e-stock_fp64.json`) e sua lettura nel gate
(docs/RUNTIME_CPU_INCREMENT_10.md, sezione 10.2).

| File | Contenuto |
|---|---|
| `amendment-spec.json` | input di `python -m tools.protocol10 amend` |
| `local-attestation.json` | attestazione locale per `inventory10 ci`: JUnit `campaign/pytest-final.xml`, 1260/1260 contro `test-inventory-a007.json` (non è un risultato del runner) |
