# Incremento 10 — emendamento 003: acquisizione senza gate di memoria host (Issue #25)

19 novembre 2026, 22:15–22:45 CET. Branch `feat/m10-preflight-tooling` su `5116175`
(PR #24). Nessun push o merge. **Nessun byte di pesi scaricato di nuovo**: il
checkpoint `checkpoints/qwen2.5-14b-97e1e76/` è già completo e non è stato toccato.
Runtime (`inwime_runtime/`, `src/`), strumenti e test M1–M9, protocollo congelato ed
emendamenti 001/002 invariati.

File normativo: [`results/increment10/amendments/003-acquisition-without-host-memory-gate.json`](../results/increment10/amendments/003-acquisition-without-host-memory-gate.json).
Evidenze: [`results/increment10/amendment-003/`](../results/increment10/amendment-003/README.md).

## 1. Decisione e incidente

[Issue #25](https://github.com/francescodifilippo/inwime/issues/25), decisione del
proprietario del 19/11: il download è un trasferimento streaming Internet → ext4 di
pagine di file. Non dipende dai limiti di RAM e paging che proteggono inferenza,
oracolo e benchmark. Gli emendamenti 001 (GO di `hostgate10` obbligatorio per
`acquire10 --execute`) e 002 (monitor in-run che ferma il trasferimento) applicavano
a un trasferimento di file criteri pensati per le prove di inferenza.

Incidente: il download reale autorizzato è stato fermato da
`HostPressureStop: host_available_bytes=267665408 violates >= 300000000`, senza errori
di checksum, spazio o HTTP. Il download è stato poi completato disattivando il gate in
modo locale e temporaneo (rapporto `acquisition-resume-20261119T183903Z.json`, fuori dal
repository: 8/8 shard con SHA256 uguale al digest LFS ufficiale). Questo emendamento
rende la correzione permanente e versionata.

## 2. Correzione (`tools/acquire10.py`)

| Prima (001/002) | Dopo (003) |
|---|---|
| `--execute` richiede `--authorization`, `--host-report`, `--host-baseline` | richiede solo `--authorization` (decisione umana) |
| `hostgate10` valutato prima della rete e sotto lock | mai consultato (nessun import, test statico) |
| monitor in-run obbligatorio; uno stop solleva `HostPressureStop` | `HostPressureStop` rimossa; `--host-diagnostics BASELINE` facoltativo, non bloccante |
| dry-run: `would_proceed_if_authorized` richiede GO host | dry-run: dipende solo dallo stato di storage e destinazione; riporta `host_policy` e l'inventario dei `.part` |

Invariati: manifesto pinnato e digest del manifesto, verifica dei metadati (API,
config, index, header via Range), SHA256 in streaming confrontato con il digest LFS,
blob Git dei file piccoli, Range accettato solo come 206 con `Content-Range` esatto,
`Content-Length` uguale al resto atteso, `.part` di proprietà con marcatore, ripresa
con re-hash, rinomina `link()+unlink()` senza sovrascrittura, lock, guardia di spazio
sul volume fisico (prima e ogni 256 MiB) ed ENOSPC con `.part` conservato, tetto di
byte autorizzati, User-Agent generico.

Il trasferimento si ferma solo per rischi dell'acquisizione: rete, stato HTTP,
Range/`Content-Range`/`Content-Length`, checksum, spazio fisico, ENOSPC, errori di I/O,
budget, lock, proprietà dei `.part`. `protocol10 check` rifiuta una sezione
`acquisition_host_policy` che riattivi il gate o che elenchi RAM, paging, pagefile o
memoria fra le condizioni di arresto.

Le diagnostiche facoltative usano lo stesso `HostMonitor` dell'emendamento 002. Le
decisioni che quell'emendamento avrebbe applicato restano nel rapporto
(`would_have_stopped_under_amendment_002`, `reasons_ignored`) e non fermano nulla.
Se il monitor non parte, l'errore viene scritto su stderr e l'acquisizione prosegue.

`hostmem10`, `hostgate10` e `hostmonitor10` restano invariati e disponibili per
inferenza, oracolo e benchmark (`tools/campaign10.py` li usa per le prove M10).

## 3. Test

Nessun node ID rimosso o rinominato: l'inventario `test-inventory-a003.json` è un
sovrainsieme di a002 (1199 contro 1180 node ID). Cambiano i corpi di alcuni test:

- `test_increment10_hostgate.py`: ogni verdetto del gate (GO, DEFER, NOT MEASURED,
  REFUSED) è ancora calcolato e verificato. Dove il test affermava «rifiutato prima di
  ogni richiesta», ora afferma la metà opposta della decisione: l'acquisizione si
  completa qualunque sia il verdetto e non chiama mai il gate (`NeverCalled`).
- `test_increment10_hostmonitor.py`: i test unitari del monitor sono invariati. I test
  di integrazione con `execute` (host critico durante il primo o un file successivo,
  pressione del guest, errori di campionamento, campionatore morto, critico prima del
  primo payload) ora verificano che il trasferimento termini con ogni file verificato
  e che le ragioni di stop siano registrate e ignorate.
- `test_increment10_acquire.py`: sei test nuovi. RAM host ridotta sintetica (150 MB
  disponibili, commit quasi esaurito, pagefile +6.000 MiB, guest 400 MiB) senza stop;
  ripresa esatta dopo un'interruzione di rete sotto pressione; test statico sugli
  import; CLI con sola autorizzazione e opzioni host rimosse rifiutate; diagnostiche
  non disponibili che non bloccano; dry-run con inventario dei `.part` che non li
  modifica.
- `test_increment10_amendment003.py`: 13 test. Sezione valida, 10 mutazioni che la
  indeboliscono o riattivano il gate (tutte rifiutate), catena reale coerente con
  `acquire10.HOST_POLICY`, emendamenti 001/002 e protocollo invariati nei byte.

Nessun test scarica pesi: tutti usano il server HTTP locale e shard sintetici.

## 4. Risultati (MISURATO)

| Esecuzione | Esito |
|---|---|
| Suite locale completa (`make cpu-kernel`, `pytest -q tests -ra`) | **1199 passed**, 0 skip, 143,9 s |
| `inventory10 verify` contro `test-inventory-a003.json` pinnato | **PASS** 1199/1199 |
| `protocol10 check` | PASS, catena 001→002→003 |
| Dry-run offline sul checkpoint reale | `transfer_bytes` 0, nessun `.part`, nessun blocco, `would_proceed_if_authorized` true |
| Simulazione runner (clone senza originali M9) | vedi `amendment-003/README.md` |

## 5. Limiti

- La correzione non è stata provata su un nuovo download reale, per scelta: il
  checkpoint è completo e non va riscaricato. Le prove usano fixture sintetiche.
- Il download reale del 19/11 è stato completato con una disattivazione locale non
  versionata; il suo rapporto resta fuori dal repository ed è citato, non importato.
- Le soglie host del profilo `download` restano nel protocollo congelato (non si
  riscrive il passato), ma non vincolano più l'acquisizione.
