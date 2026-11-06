#!/usr/bin/env python3
"""Write BENCHMARK_V2.md from collected measurements, without editing COMPARISON.md."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def table(lines, header, rows):
    lines.extend(['', '| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |'])
    lines.extend('| '+' | '.join(map(str, row))+' |' for row in rows)
    lines.append('')


def main():
    s = json.loads((ROOT/'results/v2/summary.json').read_text())
    by = {(r['n'], r['buffer_mib'], r['cache_mib'], r['mode']): r for r in s}
    complete = len(s) == 62 and all(r['count'] >= 10 for r in s)
    lines = ['# InWiMe — benchmark V2', '', 'Data: 5 novembre 2026.', '',
             '**Stato: '+('sweep completo' if complete else f'sweep in corso ({len(s)}/62 configurazioni)')+'**.', '',
             f'Sono disponibili {sum(r["count"] for r in s)} passate misurate. '
             'I risultati precedenti in `results/COMPARISON.md` sono mantenuti invariati.', '',
             '## Metodologia', '',
             '- Stesso file BF16 da 5.368.709.120 byte per tutte le configurazioni; nessuna quantizzazione, '
             'checkpoint o nuova copia del dataset. `k=4096`, `n=1,4,16`; buffer per slot da 4/8/16/32/64 MiB. '
             '`m=buffer_bytes/(2*k)` e numero blocchi `5 GiB/buffer_bytes`: la sequenza dei pesi e il volume restano identici.',
             '- CPU OpenMP/SIMD a 8 thread; cuBLAS BF16 con accumulo/output FP32; due buffer host pinned, '
             'due stream non bloccanti, nessuna Unified Memory. `n` è il numero di vettori elaborati insieme; '
             '`n=1` rappresenta una singola operazione tipo decode, non una generazione autoregressiva completa.',
             '- Disegno a due fasi: 45 combinazioni n×buffer×modalità e 5 baseline I/O, tutte senza cache; '
             '12 combinazioni aggiuntive n×cache×modalità GPU a buffer 8 MiB, con cache da 64/256 MiB. '
             'Non è un prodotto cartesiano completo buffer×cache. Ordine configurazioni mescolato con seed 20261105; '
             '10 passate consecutive per configurazione, ciascuna validata.',
             '- Ogni modalità gira in un processo distinto. Inizializzazione e warm-up sono fuori dal timer della '
             'passata; include letture, copie, GEMM, ritorno completo FP32, raccolta campioni e attese. '
             'Validazione FP64 fuori dal timer. H2D e compute sono intervalli fra eventi CUDA, non necessariamente '
             'durate pure del motore: contesa e scheduling possono allargarli.',
             '- Mediana, minimo, massimo, IQR (Q3−Q1, quantili inclusivi) e deviazione standard campionaria: '
             '`results/v2/summary.csv` contiene questi valori per ogni tempo di fase e totale. '
             'I minimi non sono selezionati come risultati rappresentativi.',
             '- `O_DIRECT`: page cache Linux esclusa, cache Windows/VHD/controller non controllate. '
             'Sono misure ripetute dopo warm-up, non letture fisicamente cold. Non sono state svuotate cache globali.', '',
             '## Confronto n=1,4,16 (buffer 8 MiB, cache disabilitata)', '']
    rows = []
    for n in [1, 4, 16]:
        for mode in ['cpu', 'sync', 'async']:
            r = by.get((n, 8, 0, mode))
            if r:
                rows.append([n, mode, f'{r["total_ms_median"]:.2f}', f'{r["total_ms_min"]:.2f}',
                             f'{r["total_ms_max"]:.2f}', f'{r["total_ms_iqr"]:.2f}', f'{r["total_ms_stdev"]:.2f}',
                             f'{r["read_ms_median"]:.2f}', f'{r["h2d_ms_median"]:.2f}', f'{r["compute_ms_median"]:.2f}'])
    table(lines, ['n', 'Modalità', 'Totale med. ms', 'Min', 'Max', 'IQR', 'σ', 'Read med.', 'H2D med.', 'Compute med.'], rows)
    lines.extend(['## Sweep buffer', '', 'Tempi complessivi mediani in ms, con minimo–massimo fra parentesi. Cache=0.', ''])
    rows=[]
    for n in [1, 4, 16]:
        for mib in [4, 8, 16, 32, 64]:
            row=[n,mib]
            for mode in ['cpu','sync','async']:
                r=by.get((n,mib,0,mode))
                row.append(f'{r["total_ms_median"]:.1f} ({r["total_ms_min"]:.1f}–{r["total_ms_max"]:.1f})' if r else 'in attesa')
            rows.append(row)
    table(lines,['n','Buffer/slot MiB','CPU','GPU sync','GPU async'],rows)
    table(lines,['Buffer MiB','I/O mediana ms','Min ms','Max ms','IQR ms','GiB/s da mediana letture'],
          [[mib,f'{r["read_ms_median"]:.2f}',f'{r["read_ms_min"]:.2f}',f'{r["read_ms_max"]:.2f}',
            f'{r["read_ms_iqr"]:.2f}',f'{5000/r["read_ms_median"]:.3f}']
           for mib in [4,8,16,32,64] if (r:=by.get((1,mib,0,'io')))])
    lines.extend(['![Tempi mediani e range min–max](results/v2/buffer-comparison.svg)', '',
                  '## Cache GPU persistente', '',
                  'La cache trattiene un prefisso statico, valido fra passate, dopo riempimento iniziale '
                  'separato (`cache_setup_ms/bytes` nei log). È una baseline favorevole di riuso, non una LRU '
                  'né un routing MoE. Con uno scan circolare di 5 GiB, una LRU da 64/256 MiB può avere zero hit. '
                  'Non si può estrapolare un hit rate MoE da questo test.', ''])
    rows=[]
    for n in [1,4,16]:
        for cache in [0,64,256]:
            for mode in ['sync','async']:
                r=by.get((n,8,cache,mode))
                if r:
                    rows.append([n,cache,mode,f'{r["total_ms_median"]:.2f}',f'{r["total_ms_min"]:.2f}',
                                 f'{r["total_ms_max"]:.2f}',f'{r["total_ms_iqr"]:.2f}',
                                 f'{r["read_bytes_median"]/MIB:.0f}',f'{r["cache_hits_median"]:.0f}',f'{r["cache_setup_ms"]:.2f}'])
    table(lines,['n','Cache MiB','Modalità','Mediana ms','Min','Max','IQR','Letti/passata MiB','Hit/passata','Setup una tantum ms'],rows)
    lines.extend(['I soli hit statici rappresentano il 1,25%/5% dei pesi con cache da 64/256 MiB. '
                  'Confrontare differenze di tempo con IQR e deviazione standard prima di attribuirle alla cache: '
                  'routing reale e ordine di esecuzione non sono modellati. Anche sugli hit il bracket di eventi '
                  'H2D può includere un piccolo costo senza copia; il contatore `h2d_bytes` resta zero per gli hit.', ''])
    lines.extend(['## Memoria, swap e telemetria', '',
                  'RSS: HWM del kernel (`getrusage`) a ogni passata. Swap e memoria `/proc/PID/status` '
                  'campionati ogni 100 ms. Pinned: conteggio applicativo dei byte registrati con CUDA '
                  '(payload, senza arrotondamento a pagina). Device: allocazioni esplicite A/B/C/cache; '
                  'non includono contesto e workspace delle librerie. VRAM globale, temperatura, potenza e '
                  'clock da `nvidia-smi` ogni 200 ms, attribuiti per intervallo temporale della configurazione. '
                  'I picchi campionati possono perdere transitori; la VRAM globale include display e altri processi.', ''])
    gpu=[r for r in s if r['mode'] in ['sync','async']]
    for key,label,scale in [('rss_peak_bytes','Picco RSS massimo',MIB),('pinned_peak_bytes','Picco pinned registrato',MIB),
                            ('device_alloc_bytes','Massimo device esplicito',MIB),('swap_peak_bytes','Picco swap del processo',MIB)]:
        lines.append(f'- {label}: **{max(r[key] for r in s)/scale:.2f} MiB**.')
    lines.append(f'- Massimo snapshot CUDA globale usato (`total−free`, dopo warm-up/setup cache): '
                 f'**{max(r["gpu_used_bytes"] for r in s)/MIB:.2f} MiB**. È uno snapshot per passata, non un picco continuo.')
    for key,label in [('gpu_memory_MiB_max','VRAM globale campionata massima (MiB)'),('temperature_C_max','Temperatura GPU massima (°C)'),
                      ('power_W_max','Potenza GPU massima (W)')]:
        vals=[r[key] for r in gpu if r[key] is not None]
        lines.append(f'- {label}: **{max(vals):.2f}**.' if vals else f'- {label}: non disponibile.')
    clocks=[r for r in gpu if r['sm_clock_MHz_min'] is not None]
    if clocks:
        lines.append(f'- Range clock SM osservato: **{min(r["sm_clock_MHz_min"] for r in clocks):.0f}–{max(r["sm_clock_MHz_max"] for r in clocks):.0f} MHz**; '
                     f'memoria: **{min(r["memory_clock_MHz_min"] for r in clocks):.0f}–{max(r["memory_clock_MHz_max"] for r in clocks):.0f} MHz**. Include warm-up, attese e validazione.')
    lines.append(f'- Incrementi globali pswpin/pswpout durante le configurazioni: '
                 f'{sum(r["pswpin_delta"] for r in s)}/{sum(r["pswpout_delta"] for r in s)} pagine; '
                 'il file BF16 viene letto esplicitamente e lo swap non è usato per lo streaming.')
    lines.extend(['', 'Alimentazione Windows: BatteryStatus=2 (collegato alla rete), piano Bilanciato; '
                  'nessun clock/power limit/fan modificato. La temperatura non misura quella dell’SSD; '
                  'non è disponibile una misura NVMe SMART termica attraverso il disco virtuale. '
                  'Clock dinamici e carico desktop possono introdurre variabilità. `VmPin`/`VmLck` WSL '
                  'non sono considerati una misura affidabile dei byte pinned CUDA. I valori globali CUDA '
                  'e NVML possono differire in WSL: non sono sommabili e non certificano un picco VRAM '
                  'esatto attribuibile al solo processo.', '',
                  '## Profiling CUDA e sovrapposizione', ''])
    trace_file=ROOT/'results/v2/trace-summary.json'
    if trace_file.exists():
        traces=json.loads(trace_file.read_text())
        verified=[r for r in traces if r.get('verified')]
        if verified:
            fmt=lambda v:f'{v:.3f}' if v is not None else 'non correlabile'
            table(lines,['Trace','Read ms','H2D GPU ms','Kernel GPU ms','H2D ∩ kernel ms','Read ∩ kernel ms','Read ∩ H2D ms'],
                  [[Path(r['file']).stem,f'{r["pread_sum_ms"]:.3f}',f'{r["h2d_sum_ms"]:.3f}',f'{r["kernel_sum_ms"]:.3f}',
                    fmt(r['h2d_kernel_overlap_ms']),fmt(r['pread_kernel_overlap_ms']),fmt(r['pread_h2d_overlap_ms'])] for r in verified])
            if any(not r.get('host_gpu_correlated',False) for r in verified):
                lines.extend(['', '**Limite WSL/CUPTI:** i record host `cuptiGetTimestamp` e le attività GPU '
                              'hanno un offset incompatibile con la durata della pipeline. Non è stata applicata '
                              'una traslazione arbitraria: le intersezioni lettura↔GPU restano non misurabili. '
                              'Le intersezioni H2D↔kernel sono invece calcolabili fra attività GPU nella stessa '
                              'base temporale. Le timeline SVG in `results/v2/activities/` mostrano solo la parte GPU.'])
            if all(r['h2d_kernel_overlap_ms']==0 for r in verified):
                lines.extend(['', '**Risultato del tracing:** nessuna sovrapposizione H2D↔kernel osservata, '
                              'né in sync né in async, nelle acquisizioni diagnostiche riportate. '
                              'Il beneficio del tempo totale async è compatibile soprattutto con letture host '
                              'sovrapposte al lavoro GPU: è un’inferenza dalla struttura della pipeline e dai tempi, '
                              'non una misura diretta dell’overlap lettura↔GPU. La conclusione vale per '
                              'i buffer effettivamente profilati, con cache disabilitata.'])
        for r in traces:
            if not r.get('verified'):
                lines.append(f'- {Path(r["file"]).name}: {r.get("reason","tracing non disponibile")}.')
        lines.extend(['', 'Intersezioni GPU calcolate sulle unioni degli intervalli nella stessa base temporale CUPTI, '
                      'evitando doppio conteggio fra stream. Sono acquisizioni diagnostiche separate, non le '
                      '10 ripetizioni prestazionali; overhead del tracing non confrontabile con le mediane senza profiler. '
                      'I contatori in `trace-summary.json` distinguono copie e kernel: cuBLAS può lanciare due '
                      'kernel per GEMM (osservato a buffer 4 MiB e in alcuni casi 64 MiB); non si assume '
                      'un kernel per blocco. Gli intervalli vuoti di scheduling nei bracket CUDA non sono '
                      'conteggiati come tempo di esecuzione kernel nelle attività CUPTI.'])
    else:
        lines.append('Acquisizioni diagnostiche in attesa: nessuna sovrapposizione hardware ancora dichiarata.')
    lines.extend(['', 'Nsight Systems 2023.4.4 ha prodotto API CUDA e range NVTX `pread`/`pipeline`, '
                  'ma nessuna tabella attività kernel/memcpy nel primo SQLite. Il problema di percorso '
                  'dell’importer è stato risolto invocando il binario host già installato. È stato quindi '
                  'aggiunto un target opzionale CUPTI, senza installazioni e senza dipendenze nuove per il target normale. '
                  'La sola riduzione del wall time non dimostra sovrapposizione H2D–kernel.', '',
                  '## Storage fisico e percorsi disponibili', '',
                  '- Dataset in ext4 su `/dev/sdd`, disco virtuale WSL2. Il registro Windows identifica '
                  'il VHDX Ubuntu in `C:\\Users\\franc\\AppData\\Local\\wsl\\{62044ba3-1e2c-4b7f-a9f7-e009245169c7}\\ext4.vhdx`.',
                  '- `Get-Partition` mappa C: sul disco 0; `Get-Disk` identifica il disco 0 come '
                  '**Samsung SSD 990 EVO 2TB, BusType NVMe, 2.000.398.934.016 byte**. '
                  'Quindi è il percorso NVMe interno → NTFS/VHDX → ext4 WSL2.',
                  '- Nessun disco USB esterno era elencato da Windows al momento dell’audit iniziale. '
                  'Non è possibile misurare unità USB o velocità negoziata senza collegarla. '
                  'Nemmeno un confronto Linux nativo/raw NVMe è disponibile da questa sessione WSL. '
                  'Non sono stati smontati VHDX, aperti dischi raw o duplicati 5 GiB su DrvFS.',
                  '- Evidenze: `results/v2/windows-storage.txt`, `windows-power-vhd.txt`, `linux-environment.txt`. '
                  'Non interpretare queste misure come banda fisica cold del solo NVMe.', '',
                  '## Verifica numerica e limiti', '',
                  '- `tests/smoke.py --gpu`: output completi con riferimento FP64 su casi piccoli, '
                  'inclusi code/dimensioni non allineate alle tile, un blocco e riuso dei due buffer; '
                  'rifiuto di dataset corrotti/troncati e CLI invalida.',
                  '- `tests/cache.py`: output completi n=1/4/16, sync/async, miss/hit parziali/hit totali, '
                  'due passate e conteggio esatto di byte, hit e memoria pinned.',
                  f'- Nei casi grandi, fino a 32 indici deterministici per blocco, riferimento FP64 '
                  f'rigenerato dalla posizione assoluta nel file: errore assoluto massimo registrato '
                  f'**{max(r["max_abs_error"] for r in s):.1f}**. I pesi sintetici sono multipli BF16 esatti '
                  'di 1/16; non coprono l’intera gamma dinamica dei checkpoint reali.',
                  '- Lo sweep usa GEMM indipendenti con k=4096 e B residente. Non include routing, '
                  'tokenizer, attenzione, KV cache, attivazioni concatenate, conversioni safetensors '
                  'o verifica di logits reali. Una matrice da 5 GiB elaborata a blocchi non dimostra '
                  'che un intero LLM o MoE sia eseguibile.', '',
                  '## Raccomandazioni per il primo executor MoE', ''])
    for n in [1,4,16]:
        candidates=[r for r in s if r['n']==n and r['mode']=='async' and r['cache_mib']==0]
        if candidates:
            best=min(candidates,key=lambda r:r['total_ms_median'])
            cpu=by.get((n,best['buffer_mib'],0,'cpu'))
            text=f'- n={n}: candidato misurato con mediana async migliore **{best["buffer_mib"]} MiB/slot**, '
            text+=f'**{best["total_ms_median"]:.2f} ms** per 5 GiB.'
            if cpu:
                text+=f' Speedup sulla CPU allo stesso buffer: **{cpu["total_ms_median"]/best["total_ms_median"]:.2f}×**.'
            lines.append(text+' Usarlo come punto di partenza, non come optimum universale.')
    lines.extend(['- Partire da un executor BF16 read-only con offset safetensors verificati, '
                  'due staging buffer limitati e fallback CPU per ogni operatore. Separare planner '
                  'di letture, kernel e cache di esperti; riservare VRAM a workspace, attivazioni e KV.',
                  '- Per decode usare le misure n=1; batching n=4/16 va valutato solo quando esistono '
                  'richieste/token realmente aggregabili. Non convertire il throughput di questa GEMM in token/s.',
                  '- Aggiungere routing top-k deterministico e una traccia di riuso degli esperti '
                  'prima di scegliere politica/dimensione cache. Misurare hit rate, bytes miss e costo '
                  'del primo token: il prefisso statico di questo test non risolve lo scan thrashing.',
                  '- Validare un piccolo layer MoE contro il riferimento originale, inclusi '
                  'gate, up/down projection, SwiGLU, pesi di routing e riduzione degli esperti, '
                  'prima di scaricare o eseguire checkpoint grandi. Preservare BF16/F16 originale.', '',
                  '## Riproducibilità e dati', '',
                  '```bash', 'make',
                  'LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/smoke.py --gpu',
                  'LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 tests/cache.py',
                  'python3 tools/benchmark_v2.py', 'python3 tools/summarize_v2.py', 'python3 tools/report_v2.py', '```', '',
                  '`results/v2/sweep/`: CSV di ogni passata, log warm-up/cache, JSON con comando, '
                  'hash del binario, tempi e picchi; campioni memoria e GPU. `summary.csv/json`: '
                  'statistiche per configurazione. `buffer-comparison.svg`: grafico esportabile. '
                  'L’output è riprendibile solo con lo stesso binario e almeno 10 ripetizioni.', '',
                  'Hash SHA256 del rapporto originale `COMPARISON.md`: '
                  '`'+hashlib.sha256((ROOT/'results/COMPARISON.md').read_bytes()).hexdigest()+'`.', '',
                  'Riferimenti: [CUDA su WSL](https://docs.nvidia.com/cuda/archive/12.4.0/wsl-user-guide/index.html), '
                  '[CUPTI Toolkit 12.4](https://developer.nvidia.com/cupti-ctk12_4).'])
    storage=ROOT/'BENCHMARK_V2_STORAGE.md'
    if storage.exists():
        lines.extend(['','---','',storage.read_text().replace('# Confronto storage — disco esterno collegato',
                      '## Aggiornamento: disco esterno collegato').replace('\n## ','\n### ')])
    (ROOT/'BENCHMARK_V2.md').write_text('\n'.join(lines)+'\n')
    print('BENCHMARK_V2.md updated; complete:',complete)


if __name__ == '__main__':
    main()
