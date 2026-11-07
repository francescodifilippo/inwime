# M1a — Esclusioni dalla cronologia pubblicabile

Questa nota accompagna la correzione autorizzata dei due commit locali non ancora pubblicati. La base già pubblicata `727e8be7bc0e6a2aaeffc78c890273b435449c4d` resta invariata. I commit locali originali erano `c7c37a5fd0d11365fd6a2ddd18ba08880f1d92c6` e `f091e97fcbf7795803a2bb0c1734604470d3167b`: questi hash identificano la cronologia locale storica, non quella da pubblicare dopo la correzione.

## File conservati esclusivamente nell'archivio privato esterno

| Percorso storico | Dimensione originale | Motivo dell'esclusione |
|---|---:|---|
| `codex.sh` | 148 byte | Launcher personale escluso su richiesta del proprietario |
| `results/v2/storage/setup.log` | 205 byte | Percorsi personali e temporanei locali |
| `results/v2/storage/windows-audit.txt` | 80.001 byte | Inventario hardware grezzo con identificativi di istanza e metadati dei volumi |

Gli originali sono stati copiati e verificati byte per byte fuori dal repository, in un archivio con accesso ristretto. Una copia di recupero della cronologia locale originale è conservata nello stesso archivio privato. Nessun identificativo del contenuto degli audit è riprodotto in questa nota. I tre percorsi sono esclusi dal `.gitignore`; le misure e gli altri file sperimentali restano invariati.

## Significato dei manifest e dei rapporti storici

- `preserved-files.json` conserva **tutte le 450 associazioni originali fra percorso e SHA256**, incluse quelle dei tre file esclusi. Gli hash non sono ricalcolati, sostituiti o eliminati per simulare una verifica completa sul repository pubblico.
- `preservation-check.json` e le affermazioni di preservazione nei rapporti descrivono controlli effettuati quando i 450 file erano disponibili localmente. Non attestano che tutti i 450 file siano ora scaricabili dalla cronologia pubblica corretta.
- Nella cronologia pubblica sono verificabili **447 dei 450 file** del manifest storico: 446 corrispondono ai file dell'attuale checkout; l'hash storico di `.gitignore` va verificato sulla sua versione nella base pubblicata `727e8be`, poiché le nuove regole ne modificano intenzionalmente il contenuto corrente. Gli altri tre file richiedono accesso all'archivio privato: non vanno classificati come disponibili, verificati pubblicamente o rigenerabili da questo repository.
- Le **76 associazioni** di `artifact-sha256.json`, comprese quelle che attestano i manifest storici stessi, restano integralmente verificabili. Questa nota aggiunta per la pubblicazione non appartiene all'acquisizione originale e non viene inserita retroattivamente nel suo manifest.
- Il codice, i 13 file dell'incremento 1, i dati delle misure e i manifest non vengono alterati. Eventuali riferimenti ai vecchi hash nei documenti dell'incremento 1 restano registrazioni della cronologia locale precedente, da interpretare insieme a questa nota.

## Verifica riproducibile della parte pubblica

Dalla radice del repository, questo controllo usa soltanto la libreria standard Python. Distingue esplicitamente file pubblici verificati e file esclusi, anziché trattare l'assenza degli ultimi come un successo completo:

```bash
python3 - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

base = Path('results/m1a-native-ext4-20261107T075250Z')
excluded = {
    'codex.sh',
    'results/v2/storage/setup.log',
    'results/v2/storage/windows-audit.txt',
}
preserved = json.loads((base / 'preserved-files.json').read_text())
assert len(preserved) == 450 and excluded <= preserved.keys()
verified = 0
for name, expected in preserved.items():
    if name in excluded:
        assert not Path(name).exists(), 'private original inside checkout'
        continue
    if name == '.gitignore':
        data = subprocess.check_output([
            'git', 'show', '727e8be7bc0e6a2aaeffc78c890273b435449c4d:.gitignore'
        ])
    else:
        data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, name
    verified += 1
assert verified == 447
artifacts = json.loads((base / 'artifact-sha256.json').read_text())
assert len(artifacts) == 76
for name, expected in artifacts.items():
    assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, name
print('446 current files + original .gitignore from published history verified')
print('447/450 preserved files verified publicly; 3 excluded, not verified publicly')
print('76/76 original M1a artifact hashes verified')
PY
```

Il `.gitignore` protegge le convenzioni e i percorsi dichiarati, non analizza automaticamente contenuto o dimensioni e non rimuove file già tracciati. Le eccezioni per le fixture e le evidenze non autorizzano l'aggiunta indiscriminata di altri payload. La rimozione dei tre file dalla cronologia locale da pubblicare è una correzione separata dalle regole d'ignore. Non richiede modifiche alla cronologia già pubblicata né force push.
