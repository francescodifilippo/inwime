# Evidenze dell'emendamento 001 dell'incremento 10

Rapporto: [docs/AMENDMENT_INCREMENT_10_001.md](../../../docs/AMENDMENT_INCREMENT_10_001.md).
Emendamento normativo: [../amendments/001-acquisition-host-gate-and-ci-inventory.json](../amendments/001-acquisition-host-gate-and-ci-inventory.json).
Le evidenze congelate di `results/increment10/` (protocollo, inventario `test-inventory.json`,
`SHA256SUMS`, `preflight-tooling/`) non sono modificate.

| File | Contenuto |
|---|---|
| `amendment-spec.json` | input di `python -m tools.protocol10 amend` |
| `hostmem-frozen-revision/` | baseline + profili download/runtime/oracle, revisione congelata di `hostmem10` (22:09–22:11 CET): tutti DEFER |
| `hostmem-amended-revision/` | idem con `hostmem10` emendato (baseline SHA/UTC registrati, 22:29–22:31 CET): tutti DEFER |
| `hostgate10-download.json` | gate sul rapporto download reale: DEFER |
| `acquire10-dry-run-with-host-gate.json` | dry-run offline con gate: `storage_ready` vero, `would_proceed_if_authorized` falso |
| `pytest-local.xml`, `pytest-local.txt` | suite locale completa: 1135 passed |
| `inventory-verify-local.json` | verifica vs `test-inventory-a001.json` pinnato: PASS 1135/1135 |
| `local-attestation.json` | attestazione locale usata da `inventory10 ci` (non è un risultato del runner) |
| `runner-simulation/` | clone senza originali M9, passi del workflow: 1124 passed, 11 skipped; `inventory10 ci` runner PASS, inventario completo NOT PROVEN, attestazione PASS |
| `SHA256SUMS` | digest dei file di questa cartella |

Nessun peso 14B scaricato; nessun file di autorizzazione creato.
