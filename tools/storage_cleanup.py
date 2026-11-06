#!/usr/bin/env python3
"""Remove only the two exclusively created test files after all matched cases finish."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/v2/storage'


if __name__=='__main__':
    for name in ['external-usb','internal-drvfs','internal-ext4']:
        stats=json.loads((BASE/name/'summary.json').read_text())
        assert len(stats)==2 and all(row['count']==10 for row in stats), 'measurements incomplete'
        assert {r['mode'] for r in stats}=={'io','async'}
    manifest=json.loads((BASE/'datasets.json').read_text())
    result={'time':time.time(),'removed':[],'errors':[],'remaining_bytes':0}
    parents={'external-usb':Path('/mnt/d'),'internal-drvfs':Path('/mnt/c/Users/franc/AppData/Local/Temp')}
    for item in manifest['copies']:
        directory=Path(item['directory'])
        file=Path(item['file'])
        assert directory.parent==parents[item['name']] and directory.name.startswith('inwime-benchmark-')
        assert file.parent==directory and file.name=='weights-5GiB.bf16'
        assert not directory.is_symlink() and not file.is_symlink()
        try:
            if file.exists():
                assert file.stat().st_size==manifest['bytes']
                file.unlink()
                result['removed'].append(str(file))
            # This small helper was copied by the diagnosis, never a preexisting user file.
            helper=directory/'windows_storage.ps1'
            if helper.exists():
                import hashlib
                expected=(ROOT/'tools/diagnostics/windows_storage.ps1').read_bytes()
                assert hashlib.sha256(helper.read_bytes()).digest()==hashlib.sha256(expected).digest()
                helper.unlink()
                result['removed'].append(str(helper))
            if directory.exists():
                directory.rmdir()  # Refuses to remove any unexpected files.
        except (OSError,AssertionError) as error:
            result['errors'].append(str(error))
        if directory.exists():
            result['remaining_bytes']+=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    result['all_clean']=not result['errors'] and not result['remaining_bytes']
    (BASE/'cleanup.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
