#!/usr/bin/env python3
"""Report the matched storage experiment; never change original performance results."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/v2/storage'


def disk_map(items):
    return {r['Name']:int(r['DiskReadBytesPersec']) for r in items}


def main():
    paths=['internal-ext4','internal-drvfs','external-usb']
    data={}
    for name in paths:
        file=BASE/name/'summary.json'
        if file.exists():
            data[name]=json.loads(file.read_text())
    rows=[]
    for name,results in data.items():
        for r in results:
            rows.append([name,r['mode'],str(r['count']),*[f'{r["total_ms_"+k]/1000:.3f}' for k in ['median','min','max','iqr','stdev']],
                         f'{5000/r["total_ms_median"]:.3f}',
                         *[f'{r[k+"_ms_median"]:.2f}' for k in ['read','h2d','compute','d2h']]])
    text=['# Confronto storage — disco esterno collegato', '', 'Data: 5 novembre 2026.', '',
          '**Stato: '+('completo (60 passate)' if len(rows)==6 and all(r[2]=='10' for r in rows) else 'misure in corso')+'**.', '',
          '## Perimetro e dispositivo', '',
          'L’utente ha scelto il confronto mirato: solo I/O e GPU asincrona n=1, '
          '10 ripetizioni per percorso, buffer 64 MiB/slot, cache GPU disabilitata. '
          'La matrice completa CPU/sync/async e n=1/4/16 del precedente V2 resta separata; '
          'non si dichiarano misurate quelle combinazioni sul disco esterno.', '',
          'Windows identifica il disco 2 come **Sabrent SSD, BusType USB, 512.110.190.592 byte**, '
          'partizione D: NTFS, circa 152 GiB liberi all’audit. Non è l’unità da 1 TB '
          'menzionata nel brief. Il bridge è **Realtek VID 0BDA / PID 9210**, driver '
          '**UASPStor**, percorso PnP `USBROOT(0)#USB(15)` / ACPI `SS03`. '
          'Questi dati identificano il collegamento, ma non certificano il protocollo '
          'interno NVMe/SATA né la velocità negoziata in bit/s.', '',
          'La query WMI della velocità negoziata restituisce null. La diagnostica nativa '
          'USB/lettura Windows con P/Invoke è stata bloccata dall’antivirus '
          '(`ScriptContainedMaliciousContent`); non sono state cambiate né aggirate '
          'le protezioni. Pertanto non è disponibile una baseline di letture Windows '
          'native senza cache, né una misura diretta della velocità del link USB. '
          'Lo script diagnostico è separato in `tools/diagnostics/`, non fa parte della suite eseguita.', '',
          '## Metodologia e risultati', '',
          '- Stessi 5 GiB BF16 e stesso SHA256 per tutti i percorsi: '
          '`fbfd729651af3054f46600bd042c08f71d87f5a915479af9329bd6617a2c8c0b`.',
          '- NVMe interno Samsung 990 EVO 2TB: file originale in ext4/VHDX WSL e '
          'copia di controllo su C: via DrvFS. USB: copia su D: via DrvFS. '
          'Le copie sono state create esclusivamente in due directory temporanee di proprietà '
          'del test, con limite totale 10 GiB; nessun file preesistente sovrascritto.',
          '- Stesso binario C++ del V2, `k=4096`, `m=8192`, 80 blocchi da 64 MiB, '
          '`n=1`, 8 thread CPU configurati, due slot/stream nella modalità GPU. '
          'Volume logico letto: 5.368.709.120 byte per passata, oltre ai due blocchi '
          'di warm-up GPU fuori timer. Nessuna Unified Memory o streaming tramite swap.',
          '- `O_DIRECT` richiesto senza fallback. Esclude la cache dati Linux; '
          'le cache Windows/controller non sono state svuotate. Il costo DrvFS/9p '
          'è parte del risultato, che non equivale alla banda intrinseca del solo SSD.',
          '- RSS HWM dal kernel; memoria/swap processo campionati ogni 100 ms; '
          'VRAM globale, temperatura, clock e potenza GPU ogni 200 ms. '
          'Statistiche su 10 ripetizioni: mediana, minimo, massimo, IQR inclusivo e σ campionaria.', '',
          '| Percorso | Modalità | N | Mediana s | Min s | Max s | IQR s | σ s | GiB/s da mediana | Read ms | H2D ms | GEMM/eventi ms | D2H ms |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    text.extend('| '+' | '.join(r)+' |' for r in rows)
    text.extend(['', 'Il campo GEMM/eventi è il bracket di eventi CUDA della GEMM, non la durata '
                 'pura delle attività kernel. Non viene riutilizzato il profilo kernel del NVMe '
                 'come se fosse misurato anche su USB. I tempi H2D/D2H ed eventi completi sono '
                 'nei CSV e nelle statistiche per percorso; scheduling e clock dinamici possono '
                 'allargare i bracket. Le fasi sovrapposte non si sommano per ricostruire il totale.'])
    text.extend(['', '## Letture osservate nello stack Windows', '',
                 'I contatori cumulativi `Win32_PerfRawData_PerfDisk_PhysicalDisk.DiskReadBytesPersec` '
                 'sono acquisiti prima/dopo ogni gruppo. I delta sono byte letti dal disco secondo '
                 'lo stack Windows: globali al dispositivo, inclusi altri processi e metadati; '
                 'non sono byte NAND e non escludono cache del controller. Un delta vicino al '
                 'volume logico supporta l’interpretazione di letture effettive nello stack disco, '
                 'ma non attribuisce ogni byte esclusivamente al benchmark.', '',
                 '| Gruppo | Disco | Delta letture GiB |', '| --- | --- | --- |'])
    for name in paths:
        file=BASE/name/'physical-counters.json'
        if file.exists():
            record=json.loads(file.read_text())
            before,after=disk_map(record['before']),disk_map(record['after'])
            for disk in after:
                if disk!='_Total' and disk in before:
                    text.append(f'| {name} | {disk} | {(after[disk]-before[disk])/1024**3:.3f} |')
    before=BASE/'external-usb/counters-before-io.json'
    after=BASE/'external-usb/counters-after-io.json'
    if before.exists() and after.exists():
        a,b=disk_map(json.loads(before.read_text())),disk_map(json.loads(after.read_text()))
        for disk in b:
            if disk!='_Total' and disk in a:
                text.append(f'| external-usb I/O iniziale, 10 passate | {disk} | {(b[disk]-a[disk])/1024**3:.3f} |')
    text.extend(['', 'Il gruppo USB principale riusa la baseline I/O già completata: '
                 'il suo delta successivo riguarda le nuove passate GPU; il gruppo I/O iniziale '
                 'ha contatori separati. I gruppi di controllo nuovi comprendono I/O e GPU. '
                 'Sommare i byte nominali in base alle passate effettivamente eseguite, non alle '
                 'configurazioni riusate dalla funzione di resume.', '',
                 '## Correttezza, memoria e interpretazione', ''])
    for name,rs in data.items():
        gpu=[r for r in rs if r['mode']=='async']
        if gpu:
            r=gpu[0]
            text.append(f'- {name}: errore massimo {r["max_abs_error"]:.1f}, '
                        f'picco RSS {r["rss_peak_bytes"]/1024**2:.2f} MiB, pinned registrata '
                        f'{r["pinned_peak_bytes"]/1024**2:.2f} MiB, VRAM globale campionata '
                        f'{r["gpu_memory_MiB_max"]} MiB, snapshot CUDA globale '
                        f'{r["gpu_used_bytes"]/1024**2:.2f} MiB, VmSwap osservata '
                        f'{r["swap_peak_bytes"]:.0f} byte, temperatura massima '
                        f'{r["temperature_C_max"]} °C, clock SM '
                        f'{r["sm_clock_MHz_min"]}–{r["sm_clock_MHz_max"]} MHz.')
    text.extend(['', 'I casi grandi verificano campioni deterministici con riferimento FP64; '
                 'le verifiche complete dei casi piccoli sono quelle della suite V2 e vengono '
                 'ripetute prima della chiusura dell’esperimento. VRAM NVML e snapshot CUDA sono '
                 'misure diverse in WSL; non costituiscono un picco esatto per processo. '
                 'VmSwap riguarda il processo Linux; il pagefile dell’host Windows non è monitorato.', ''])
    if len(data)==3 and all(any(r['mode']=='async' for r in data[p]) for p in paths):
        asyncs={p:next(r for r in data[p] if r['mode']=='async') for p in paths}
        ext4=asyncs['internal-ext4']['total_ms_median']
        drvfs=asyncs['internal-drvfs']['total_ms_median']
        usb=asyncs['external-usb']['total_ms_median']
        text.extend([f'Nel decode misurato, C:/DrvFS impiega **{drvfs/ext4:.2f}×** il tempo '
                     f'NVMe/ext4 e D:/USB/DrvFS **{usb/ext4:.2f}×**. '
                     f'USB/DrvFS rispetto a C:/DrvFS: **{usb/drvfs:.2f}×**.', '',
                     'Per il primo executor MoE usare come riferimento il percorso realmente misurato '
                     'più rapido, mantenendo il planner di lettura separato dal kernel. '
                     'Non attribuire l’intera differenza al bus USB: il controllo sullo stesso '
                     'NVMe interno via DrvFS serve a distinguere il costo del percorso filesystem/virtualizzazione. '
                     'Prima di acquistare hardware o scegliere l’SSD come sede del modello, misurare '
                     'il percorso di lettura definitivo. Nessuna modifica a partizioni, VHDX o '
                     'configurazioni viene effettuata da questo esperimento.'])
    cleanup=BASE/'cleanup.json'
    text.extend(['', '## Dati, pulizia e limiti', '',
                 'Raw CSV/JSON/telemetria e contatori: `results/v2/storage/`. '
                 'Le statistiche per percorso sono nei rispettivi `summary.csv/json`. '
                 'Le prove non sono randomizzate fra percorsi né termicamente controllate; '
                 'sono misure sequenziali nello stesso ambiente, senza garanzie cold. '
                 'Una mediana async leggermente inferiore alla baseline solo I/O non dimostra '
                 'che la GPU acceleri il disco: ordine, warm-up e variabilità differiscono. '
                 'Nessun checkpoint scaricato, nessuna quantizzazione o nuova dipendenza installata.', '',
                 (('Le due copie temporanee sono state rimosse; vedi `cleanup.json`.' if json.loads(cleanup.read_text()).get('all_clean')
                   else f'Pulizia parziale: {json.loads(cleanup.read_text()).get("remaining_bytes")} byte residui; vedere errori in `cleanup.json`.') if cleanup.exists()
                  else 'Le due copie temporanee sono ancora presenti per completare le misure e saranno rimosse al termine.'), '',
                 'Hash del rapporto originale `COMPARISON.md`: `'+hashlib.sha256((ROOT/'results/COMPARISON.md').read_bytes()).hexdigest()+'`.', '',
                 'Riferimenti: [controllo della velocità USB](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/usbioctl/ns-usbioctl-_usb_node_connection_information_ex_v2_flags), '
                 '[letture Windows senza buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering), '
                 '[contatori raw PhysicalDisk](https://learn.microsoft.com/en-us/previous-versions/aa394308(v=vs.85)).'])
    (ROOT/'BENCHMARK_V2_STORAGE.md').write_text('\n'.join(text)+'\n')
    print('Storage report:',len(rows),'result rows')


if __name__=='__main__':
    main()
