# Confronto storage — disco esterno collegato

Data: 5 novembre 2026.

**Stato: completo (60 passate)**.

## Perimetro e dispositivo

L’utente ha scelto il confronto mirato: solo I/O e GPU asincrona n=1, 10 ripetizioni per percorso, buffer 64 MiB/slot, cache GPU disabilitata. La matrice completa CPU/sync/async e n=1/4/16 del precedente V2 resta separata; non si dichiarano misurate quelle combinazioni sul disco esterno.

Windows identifica il disco 2 come **Sabrent SSD, BusType USB, 512.110.190.592 byte**, partizione D: NTFS, circa 152 GiB liberi all’audit. Non è l’unità da 1 TB menzionata nel brief. Il bridge è **Realtek VID 0BDA / PID 9210**, driver **UASPStor**, percorso PnP `USBROOT(0)#USB(15)` / ACPI `SS03`. Questi dati identificano il collegamento, ma non certificano il protocollo interno NVMe/SATA né la velocità negoziata in bit/s.

La query WMI della velocità negoziata restituisce null. La diagnostica nativa USB/lettura Windows con P/Invoke è stata bloccata dall’antivirus (`ScriptContainedMaliciousContent`); non sono state cambiate né aggirate le protezioni. Pertanto non è disponibile una baseline di letture Windows native senza cache, né una misura diretta della velocità del link USB. Lo script diagnostico è separato in `tools/diagnostics/`, non fa parte della suite eseguita.

## Metodologia e risultati

- Stessi 5 GiB BF16 e stesso SHA256 per tutti i percorsi: `fbfd729651af3054f46600bd042c08f71d87f5a915479af9329bd6617a2c8c0b`.
- NVMe interno Samsung 990 EVO 2TB: file originale in ext4/VHDX WSL e copia di controllo su C: via DrvFS. USB: copia su D: via DrvFS. Le copie sono state create esclusivamente in due directory temporanee di proprietà del test, con limite totale 10 GiB; nessun file preesistente sovrascritto.
- Stesso binario C++ del V2, `k=4096`, `m=8192`, 80 blocchi da 64 MiB, `n=1`, 8 thread CPU configurati, due slot/stream nella modalità GPU. Volume logico letto: 5.368.709.120 byte per passata, oltre ai due blocchi di warm-up GPU fuori timer. Nessuna Unified Memory o streaming tramite swap.
- `O_DIRECT` richiesto senza fallback. Esclude la cache dati Linux; le cache Windows/controller non sono state svuotate. Il costo DrvFS/9p è parte del risultato, che non equivale alla banda intrinseca del solo SSD.
- RSS HWM dal kernel; memoria/swap processo campionati ogni 100 ms; VRAM globale, temperatura, clock e potenza GPU ogni 200 ms. Statistiche su 10 ripetizioni: mediana, minimo, massimo, IQR inclusivo e σ campionaria.

| Percorso | Modalità | N | Mediana s | Min s | Max s | IQR s | σ s | GiB/s da mediana | Read ms | H2D ms | GEMM/eventi ms | D2H ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal-ext4 | async | 10 | 1.515 | 1.445 | 1.605 | 0.050 | 0.052 | 3.301 | 1489.05 | 424.14 | 45.64 | 0.59 |
| internal-ext4 | io | 10 | 1.597 | 1.523 | 1.667 | 0.034 | 0.042 | 3.131 | 1596.86 | 0.00 | 0.00 | 0.00 |
| internal-drvfs | async | 10 | 31.495 | 29.886 | 36.327 | 3.112 | 2.253 | 0.159 | 31432.04 | 3202.62 | 653.97 | 2.67 |
| internal-drvfs | io | 10 | 30.516 | 28.496 | 33.368 | 1.672 | 1.386 | 0.164 | 30515.98 | 0.00 | 0.00 | 0.00 |
| external-usb | async | 10 | 51.810 | 50.216 | 53.732 | 1.960 | 1.245 | 0.097 | 51746.07 | 3203.49 | 654.42 | 2.67 |
| external-usb | io | 10 | 54.092 | 52.274 | 58.232 | 1.806 | 1.793 | 0.092 | 54091.77 | 0.00 | 0.00 | 0.00 |

Il campo GEMM/eventi è il bracket di eventi CUDA della GEMM, non la durata pura delle attività kernel. Non viene riutilizzato il profilo kernel del NVMe come se fosse misurato anche su USB. I tempi H2D/D2H ed eventi completi sono nei CSV e nelle statistiche per percorso; scheduling e clock dinamici possono allargare i bracket. Le fasi sovrapposte non si sommano per ricostruire il totale.

## Letture osservate nello stack Windows

I contatori cumulativi `Win32_PerfRawData_PerfDisk_PhysicalDisk.DiskReadBytesPersec` sono acquisiti prima/dopo ogni gruppo. I delta sono byte letti dal disco secondo lo stack Windows: globali al dispositivo, inclusi altri processi e metadati; non sono byte NAND e non escludono cache del controller. Un delta vicino al volume logico supporta l’interpretazione di letture effettive nello stack disco, ma non attribuisce ogni byte esclusivamente al benchmark.

| Gruppo | Disco | Delta letture GiB |
| --- | --- | --- |
| internal-ext4 | 0 C: | 101.389 |
| internal-ext4 | 1 | 0.000 |
| internal-ext4 | 2 D: | 0.000 |
| internal-drvfs | 0 C: | 104.600 |
| internal-drvfs | 1 | 0.000 |
| internal-drvfs | 2 D: | 0.000 |
| external-usb | 0 C: | 5.051 |
| external-usb | 1 | 0.000 |
| external-usb | 2 D: | 49.991 |
| external-usb I/O iniziale, 10 passate | 0 C: | 5.643 |
| external-usb I/O iniziale, 10 passate | 1 | 0.000 |
| external-usb I/O iniziale, 10 passate | 2 D: | 49.991 |

Il gruppo USB principale riusa la baseline I/O già completata: il suo delta successivo riguarda le nuove passate GPU; il gruppo I/O iniziale ha contatori separati. I gruppi di controllo nuovi comprendono I/O e GPU. Sommare i byte nominali in base alle passate effettivamente eseguite, non alle configurazioni riusate dalla funzione di resume.

## Correttezza, memoria e interpretazione

- internal-ext4: errore massimo 0.0, picco RSS 473.32 MiB, pinned registrata 128.06 MiB, VRAM globale campionata 226.0 MiB, snapshot CUDA globale 943.31 MiB, VmSwap osservata 0 byte, temperatura massima 50.0 °C, clock SM 210.0–2040.0 MHz.
- internal-drvfs: errore massimo 0.0, picco RSS 473.24 MiB, pinned registrata 128.06 MiB, VRAM globale campionata 226.0 MiB, snapshot CUDA globale 943.31 MiB, VmSwap osservata 0 byte, temperatura massima 49.0 °C, clock SM 210.0–1500.0 MHz.
- external-usb: errore massimo 0.0, picco RSS 473.02 MiB, pinned registrata 128.06 MiB, VRAM globale campionata 226.0 MiB, snapshot CUDA globale 943.31 MiB, VmSwap osservata 0 byte, temperatura massima 49.1 °C, clock SM 210.0–1500.0 MHz.

I casi grandi verificano campioni deterministici con riferimento FP64; le verifiche complete dei casi piccoli sono quelle della suite V2 e vengono ripetute prima della chiusura dell’esperimento. VRAM NVML e snapshot CUDA sono misure diverse in WSL; non costituiscono un picco esatto per processo. VmSwap riguarda il processo Linux; il pagefile dell’host Windows non è monitorato.

Nel decode misurato, C:/DrvFS impiega **20.80×** il tempo NVMe/ext4 e D:/USB/DrvFS **34.21×**. USB/DrvFS rispetto a C:/DrvFS: **1.64×**.

Per il primo executor MoE usare come riferimento il percorso realmente misurato più rapido, mantenendo il planner di lettura separato dal kernel. Non attribuire l’intera differenza al bus USB: il controllo sullo stesso NVMe interno via DrvFS serve a distinguere il costo del percorso filesystem/virtualizzazione. Prima di acquistare hardware o scegliere l’SSD come sede del modello, misurare il percorso di lettura definitivo. Nessuna modifica a partizioni, VHDX o configurazioni viene effettuata da questo esperimento.

## Dati, pulizia e limiti

Raw CSV/JSON/telemetria e contatori: `results/v2/storage/`. Le statistiche per percorso sono nei rispettivi `summary.csv/json`. Le prove non sono randomizzate fra percorsi né termicamente controllate; sono misure sequenziali nello stesso ambiente, senza garanzie cold. Una mediana async leggermente inferiore alla baseline solo I/O non dimostra che la GPU acceleri il disco: ordine, warm-up e variabilità differiscono. Nessun checkpoint scaricato, nessuna quantizzazione o nuova dipendenza installata.

Le due copie temporanee sono state rimosse; vedi `cleanup.json`.

Hash del rapporto originale `COMPARISON.md`: `f3f455bddc6babbaf662df02efea32c4dfb58746dfc0bd7b4ce31426be0646eb`.

Riferimenti: [controllo della velocità USB](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/usbioctl/ns-usbioctl-_usb_node_connection_information_ex_v2_flags), [letture Windows senza buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering), [contatori raw PhysicalDisk](https://learn.microsoft.com/en-us/previous-versions/aa394308(v=vs.85)).
