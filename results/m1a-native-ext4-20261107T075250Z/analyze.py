"""Summarize this completed acquisition only; preserve historical reports."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

P = Path(__file__).resolve().parent
GIB = 1024**3
MIB = 1024**2


def load(name):
    return json.loads((P / name).read_text())


def sectors(snapshot, device):
    for line in snapshot['linux']['diskstats'].splitlines():
        fields = line.split()
        if fields[2] == device:
            return int(fields[5])
    raise ValueError(device)


def windows_reads(snapshot, number):
    rows = snapshot['windows']['Counters']
    return int(next(r['DiskReadBytesPersec'] for r in rows if r['Name'].split()[0] == str(number)))


assert load('completed.json')['passes'] == 60
assert not load('preservation-check.json')['changed']
summaries = {route: load(f'{route}/summary.json') for route in ['external-ext4', 'internal-ext4']}
rows_out = []
boundaries = []
lines = ['# M1a — Sabrent ext4 diretto in WSL2', '',
         'Acquisizione del 7 novembre 2026. 60 passate, nessuna quantizzazione; non è un runtime LLM.', '',
         '**Metodo e provenienza.** Binario e strumenti V2 invariati. Stesso SHA256 del dataset storico, '
         '5 GiB BF16, k=4096, m=8192, 80 blocchi, n=1, 8 thread CPU, cache GPU=0, '
         '64 MiB per slot, due slot GPU. CPU, solo I/O e GPU async, 10 ripetizioni per caso, '
         'processi separati. Modalità sync verificata soltanto sul fixture piccolo. '
         'O_DIRECT richiesto senza fallback. Validazione FP64 completa sui piccoli; 2560 campioni per passata grande di calcolo.', '',
         '**Percorsi.** Sabrent: `/mnt/wsl/Sabrent`, `/dev/sdd1`, ext4 UUID '
         '`5ee418f3-419e-4434-9337-ef5267c3a403`; disco fisico 512110190592 byte, '
         'Windows disk 2 offline. Controllo interno: dataset esistente nella distribuzione, '
         'ext4 su disco virtuale `/dev/sde`, Samsung 990 EVO tramite VHDX Windows. '
         'Il nuovo percorso esterno non è DrvFS né un nuovo VHDX né USB/IP. '
         'Velocità USB negoziata e protocollo interno SSD non verificati.', '',
         '| Percorso | Modalità | N | Mediana s | Min–max s | IQR s | σ s | GiB/s effettivi | Prima passata s | Mediana passate 2–10 s |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for route in summaries:
    assert len(summaries[route]) == 3
    for s in summaries[route]:
        mode = s['mode']
        raw = list(csv.DictReader((P / route / 'sweep' / f'n1-b64-c0-{mode}.csv').open()))
        assert len(raw) == 10 and all(float(r['max_abs_error']) == 0 and int(r['read_bytes']) == 5*GIB for r in raw)
        times = [float(r['total_ms'])/1000 for r in raw]
        assert s['swap_peak_bytes'] == 0
        before = load(f'{route}-{mode}-before.json')
        after = load(f'{route}-{mode}-after.json')
        boundaries.extend([before, after])
        disk = 'sdd' if route == 'external-ext4' else 'sde'
        expected = 50 + {'io': 0, 'cpu': .625, 'async': 1.25}[mode]
        item = {'route': route, 'mode': mode, 'logical_timed_GiB': 50, 'expected_including_warmup_GiB': expected,
                'linux_block_read_GiB': (sectors(after, disk)-sectors(before, disk))*512/GIB,
                'windows_USB_read_GiB': (windows_reads(after,2)-windows_reads(before,2))/GIB,
                'windows_internal_read_GiB': (windows_reads(after,0)-windows_reads(before,0))/GIB}
        rows_out.append(item)
        lines.append(f'| {route} | {mode} | 10 | {st.median(times):.3f} | {min(times):.3f}–{max(times):.3f} | '
                     f'{s["total_ms_iqr"]/1000:.3f} | {s["total_ms_stdev"]/1000:.3f} | {5/st.median(times):.3f} | '
                     f'{times[0]:.3f} | {st.median(times[1:]):.3f} |')
ext = {s['mode']: s for s in summaries['external-ext4']}
internal = {s['mode']: s for s in summaries['internal-ext4']}
historic = json.loads((P.parent / 'v2/storage/external-usb/summary.json').read_text())
old = {s['mode']: s for s in historic}
lines += ['', f'Confronto storico descrittivo: USB/DrvFS async {old["async"]["total_ms_median"]/1000:.3f} s → '
          f'USB/ext4 async {ext["async"]["total_ms_median"]/1000:.3f} s, '
          f'rapporto dei tempi {old["async"]["total_ms_median"]/ext["async"]["total_ms_median"]:.2f}×. '
          f'Solo I/O: {old["io"]["total_ms_median"]/ext["io"]["total_ms_median"]:.2f}×. '
          'Sono sessioni diverse, con filesystem/layout e carico host diversi: non è un isolamento causale del solo costo DrvFS.', '',
          f'Nel confronto corrente, async esterno richiede {ext["async"]["total_ms_median"]/internal["async"]["total_ms_median"]:.2f}× '
          f'il tempo interno. Sul Sabrent, CPU/async={ext["cpu"]["total_ms_median"]/ext["async"]["total_ms_median"]:.2f}×. '
          'Le durate H2D/GEMM dei CSV sono bracket di eventi sotto carico, non kernel isolati. '
          'Nessuna nuova acquisizione CUPTI; nessuna conclusione misurata sull’overlap letture/GPU.', '',
          '**Letture osservate.** 300 GiB logici nel timer + 3,75 GiB di warm-up. '
          'Preparazione, checksum e fixture piccoli sono separati. Contatori globali allo stack disco, '
          'non byte NAND né attribuzione esclusiva al processo.', '',
          '| Percorso | Modalità | Attesi con warm-up GiB | Linux block GiB | Windows USB GiB | Windows interno GiB |',
          '|---|---|---:|---:|---:|---:|']
for r in rows_out:
    lines.append(f'| {r["route"]} | {r["mode"]} | {r["expected_including_warmup_GiB"]:.3f} | '
                 f'{r["linux_block_read_GiB"]:.3f} | {r["windows_USB_read_GiB"]:.3f} | {r["windows_internal_read_GiB"]:.3f} |')
lines += ['', '**Memoria e condizioni.** Errore massimo registrato zero in tutte le passate di calcolo. '
          'Nessuno swap del processo Linux osservato. RSS è il massimo HWM per processo; '
          'VRAM NVML globale campionata e snapshot CUDA sono misure distinte e non sommabili.', '',
          '| Percorso | RSS max MiB | Pinned max MiB | Device esplicito max MiB | NVML globale max MiB | Snapshot CUDA max MiB | GPU max °C | GPU max W | pswpin/out pagine |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for route, ss in summaries.items():
    maximum = lambda key: max(s[key] for s in ss if s.get(key) is not None)
    lines.append(f'| {route} | {maximum("rss_peak_bytes")/MIB:.2f} | {maximum("pinned_peak_bytes")/MIB:.2f} | '
                 f'{maximum("device_alloc_bytes")/MIB:.2f} | {maximum("gpu_memory_MiB_max"):.1f} | '
                 f'{maximum("gpu_used_bytes")/MIB:.2f} | {maximum("temperature_C_max"):.1f} | '
                 f'{maximum("power_W_max"):.2f} | {sum(s["pswpin_delta"] for s in ss)}/{sum(s["pswpout_delta"] for s in ss)} |')
snapshots = [load('windows-before.json'), *[b['windows'] for b in boundaries], load('windows-after.json')]
available = [b['Memory']['AvailableMBytes'] for b in snapshots]
paging = [p['CurrentUsage'] for b in snapshots for p in b['Paging']]
lines += ['', f'Windows ai confini dei casi: memoria disponibile {min(available)}–{max(available)} MiB; '
          f'pagefile in uso {min(paging)}–{max(paging)} MiB. Prima della preparazione: '
          f'{snapshots[0]["Paging"][0]["CurrentUsage"]} MiB; alla fine: {snapshots[-1]["Paging"][0]["CurrentUsage"]} MiB. '
          'Paging host già presente e aumentato durante preparazione/prima serie: assenza di VmSwap Linux non significa assenza di paging Windows. '
          'Questi sono snapshot, non picchi continui né attribuzione al benchmark. '
          'Il carico host limita l’interpretazione delle prestazioni.', '',
          '**Cache e ripetibilità.** SHA256 e copia hanno letto il dataset prima delle prove. '
          'O_DIRECT evita la page cache dati Linux, non prova l’assenza di cache Windows/controller. '
          'Prima passata e ripetizioni sono riportate separatamente; nessuna passata è certificata cold. '
          'Ordine seed 20261107 salvato in order.json: per questo seed i tre casi esterni precedono i tre interni; '
          'non c’è alternanza fra percorsi. Alimentazione rete e piano Bilanciato registrati nei JSON; '
          'clock dinamici, nessun controllo termico e temperatura SSD non misurata. '
          'Servirebbe una futura sessione con host meno carico e ordine controbilanciato per una stima più pulita.', '',
          '**Artefatti e limiti.** run.py registra comandi, inventario e ordine; i JSON/CSV/log e la telemetria sono nelle '
          'sottocartelle external-ext4 e internal-ext4. run.py è una registrazione monouso e rifiuta artefatti già presenti; '
          'per ripetere occorre una nuova directory. Il dataset esterno da 5 GiB e il fixture piccolo sono lasciati intatti '
          'nel percorso di dataset-verified.json. Nessun nuovo VHDX, modifica a mount/driver/configurazioni o checkpoint scaricato. '
          'Preservazione di tutti i file preesistenti verificata tramite SHA256. '
          'Questo test sequenziale non misura accessi selettivi MoE, latenze p95 di tile casuali, token/s o eseguibilità di un LLM completo.']
with (P / 'read-volumes.json').open('x') as f:
    json.dump(rows_out, f, indent=2)
with (P / 'REPORT.md').open('x') as f:
    f.write('\n'.join(lines)+'\n')
print('\n'.join(lines))
