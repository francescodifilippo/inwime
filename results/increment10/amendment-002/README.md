# Evidenze dell'emendamento 002 dell'incremento 10

Rapporto: [docs/AMENDMENT_INCREMENT_10_002.md](../../../docs/AMENDMENT_INCREMENT_10_002.md).
Emendamento normativo: [../amendments/002-acquisition-in-run-host-monitor.json](../amendments/002-acquisition-in-run-host-monitor.json).
Protocollo congelato, emendamento 001 e le sue evidenze non sono modificati.

| File | Contenuto |
|---|---|
| `amendment-spec.json` | input di `python -m tools.protocol10 amend` |
| `pytest-local.xml`, `pytest-local.txt` | suite locale completa: 1180 passed |
| `inventory-verify-local.json` | verifica vs `test-inventory-a002.json` pinnato: PASS 1180/1180 |
| `local-attestation.json` | attestazione locale usata da `inventory10 ci` (non è un risultato del runner) |
| `runner-simulation/` | clone senza originali M9, passi del workflow: 1169 passed, 11 skipped; `inventory10 ci` runner PASS, inventario completo NOT PROVEN, attestazione PASS |
| `SHA256SUMS` | digest dei file di questa cartella |

Nessun peso 14B scaricato; nessun monitor eseguito su un trasferimento reale; nessun file di autorizzazione creato.
