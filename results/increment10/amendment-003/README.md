# Evidenze dell'emendamento 003 dell'incremento 10 (Issue #25)

Rapporto: [docs/AMENDMENT_INCREMENT_10_003.md](../../../docs/AMENDMENT_INCREMENT_10_003.md).
Emendamento normativo: [../amendments/003-acquisition-without-host-memory-gate.json](../amendments/003-acquisition-without-host-memory-gate.json).
Protocollo congelato, emendamenti 001/002 e le loro evidenze non sono modificati.

| File | Contenuto |
|---|---|
| `amendment-spec.json` | input di `python -m tools.protocol10 amend` |
| `pytest-local.xml`, `pytest-local.txt` | suite locale completa: 1199 passed, 0 skip |
| `inventory-verify-local.json` | verifica vs `test-inventory-a003.json` pinnato: PASS 1199/1199 |
| `local-attestation.json` | attestazione locale usata da `inventory10 ci` (non è un risultato del runner) |
| `runner-simulation/` | clone senza originali M9 né strumenti non ancora versionati, passi del workflow: 1188 passed, 11 skipped (messaggio dichiarato); `inventory10 ci`: runner PASS, inventario completo NOT PROVEN, attestazione PASS |
| `acquire10-dry-run-real-checkpoint.json` | dry-run offline sul checkpoint reale completo: 0 byte da trasferire, nessun `.part`, nessun blocco, politica host non vincolante |
| `SHA256SUMS` | digest dei file di questa cartella |

Nessun peso scaricato; nessun download reale eseguito per provare la correzione (fixture sintetiche).
