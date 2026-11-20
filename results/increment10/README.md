# Evidenze dell'incremento 10 — preparazione degli strumenti C1–C5

Stato: **strumenti pronti, download NON autorizzato, nessun peso 14B scaricato.**
Rapporto: [docs/PREPARATION_INCREMENT_10.md](../../docs/PREPARATION_INCREMENT_10.md).
Protocollo spiegato: [docs/PROTOCOL_INCREMENT_10.md](../../docs/PROTOCOL_INCREMENT_10.md).
Base: `639de22` (`origin/main`). Solo file aggiunti: runtime, kernel, test e
strumenti M1–M9 ed evidenze M7–M9 invariati (verificato da
`tests/test_increment10_protocol.py`).

## Indice

| Area | File |
|---|---|
| Protocollo preregistrato (normativo) e suo SHA256 | `protocol.json`, `protocol.sha256` |
| Inventario versionato dei test (Issue #22), SHA pinnato nel protocollo | `test-inventory.json` |
| Metadati ufficiali HF catturati nel preflight (nessun peso; 12 file, SHA in `SHA256SUMS`) | `preflight-metadata/` |
| Dry-run di `acquire10`: online (123.708 B di metadati) e offline | `preflight-tooling/acquire10-dry-run-*.json` |
| Suite completa archiviata (testo, JUnit) e verifica dell'inventario sul JUnit | `preflight-tooling/pytest-final.*`, `preflight-tooling/inventory-verify.json` |
| Controlli dell'oracolo nel venv di verifica | `preflight-tooling/check-oracle10.txt` |
| Preflight read-only della memoria host (baseline e profili download/runtime/oracle: tutti DEFER) | `preflight-tooling/hostmem-*.json` |
| Verifica della sanificazione M9 (passo CI) | `preflight-tooling/sanitize9-verify.txt` |
| Evidenze superate dal ri-freeze (prima revisione di `hostmem10` con effetto osservatore; suite e verifica sul vecchio inventario), conservate | `preflight-tooling/superseded-before-refreeze/` |
| Equivalenza a componenti di `oracle10` sul 7B locale, con istantanee host | `oracle10-7b-equivalence/` |
| Digest di tutte le evidenze di questa cartella | `SHA256SUMS` |

Emendamenti futuri: solo nuovi file `amendments/NNN-<slug>.json`, con catena di
SHA verificata da `python -m tools.protocol10 check`.

## Riproduzione (offline, senza pesi 14B)

```
make cpu-kernel
python -m pytest -q tests -ra --junitxml=pytest.xml
python -m tools.inventory10 verify --inventory results/increment10/test-inventory.json \
    --inventory-sha256 "$(python -c 'import json;print(json.load(open("results/increment10/protocol.json"))["test_inventory"]["sha256"])')" \
    --junit pytest.xml
python -m tools.protocol10 check
python -m tools.acquire10 checkpoints/qwen2.5-14b-97e1e76 --backing-volume /mnt/c \
    --metadata-dir results/increment10/preflight-metadata          # dry-run (default)
.venv-oracle7/bin/python -m unittest tools.check_oracle10           # venv di verifica
```
