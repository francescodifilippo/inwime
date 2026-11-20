# Evidenze dell'emendamento 005 dell'incremento 10

Emendamento normativo: [../amendments/005-gate-empty-campaign-not-proven.json](../amendments/005-gate-empty-campaign-not-proven.json).
La prima suite finale (`../campaign/pytest-final-a1.*`: 1245 passed, 1 failed) ha mostrato
che il gate G dava FAIL invece di NOT PROVEN su una campagna senza record. Corretto in
`tools/report_increment10.py`; test invariati; suite rieseguita 1246/1246 (`../campaign/pytest-final.*`).

| File | Contenuto |
|---|---|
| `amendment-spec.json` | input di `python -m tools.protocol10 amend` |
