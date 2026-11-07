# M1a — Sabrent ext4 diretto in WSL2

Acquisizione del 7 novembre 2026. 60 passate, nessuna quantizzazione; non è un runtime LLM.

**Metodo e provenienza.** Binario e strumenti V2 invariati. Stesso SHA256 del dataset storico, 5 GiB BF16, k=4096, m=8192, 80 blocchi, n=1, 8 thread CPU, cache GPU=0, 64 MiB per slot, due slot GPU. CPU, solo I/O e GPU async, 10 ripetizioni per caso, processi separati. Modalità sync verificata soltanto sul fixture piccolo. O_DIRECT richiesto senza fallback. Validazione FP64 completa sui piccoli; 2560 campioni per passata grande di calcolo.

**Percorsi.** Sabrent: `/mnt/wsl/Sabrent`, `/dev/sdd1`, ext4 UUID `5ee418f3-419e-4434-9337-ef5267c3a403`; disco fisico 512110190592 byte, Windows disk 2 offline. Controllo interno: dataset esistente nella distribuzione, ext4 su disco virtuale `/dev/sde`, Samsung 990 EVO tramite VHDX Windows. Il nuovo percorso esterno non è DrvFS né un nuovo VHDX né USB/IP. Velocità USB negoziata e protocollo interno SSD non verificati.

| Percorso | Modalità | N | Mediana s | Min–max s | IQR s | σ s | GiB/s effettivi | Prima passata s | Mediana passate 2–10 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| external-ext4 | async | 10 | 12.015 | 11.991–12.041 | 0.017 | 0.016 | 0.416 | 11.991 | 12.015 |
| external-ext4 | cpu | 10 | 12.266 | 12.147–12.308 | 0.050 | 0.053 | 0.408 | 12.218 | 12.267 |
| external-ext4 | io | 10 | 12.049 | 11.854–12.120 | 0.118 | 0.087 | 0.415 | 12.090 | 12.033 |
| internal-ext4 | async | 10 | 1.642 | 1.593–1.808 | 0.101 | 0.076 | 3.046 | 1.623 | 1.646 |
| internal-ext4 | cpu | 10 | 1.872 | 1.755–2.019 | 0.171 | 0.104 | 2.671 | 1.935 | 1.809 |
| internal-ext4 | io | 10 | 1.802 | 1.785–2.036 | 0.018 | 0.076 | 2.775 | 1.805 | 1.798 |

Confronto storico descrittivo: USB/DrvFS async 51.810 s → USB/ext4 async 12.015 s, rapporto dei tempi 4.31×. Solo I/O: 4.49×. Sono sessioni diverse, con filesystem/layout e carico host diversi: non è un isolamento causale del solo costo DrvFS.

Nel confronto corrente, async esterno richiede 7.32× il tempo interno. Sul Sabrent, CPU/async=1.02×. Le durate H2D/GEMM dei CSV sono bracket di eventi sotto carico, non kernel isolati. Nessuna nuova acquisizione CUPTI; nessuna conclusione misurata sull’overlap letture/GPU.

**Letture osservate.** 300 GiB logici nel timer + 3,75 GiB di warm-up. Preparazione, checksum e fixture piccoli sono separati. Contatori globali allo stack disco, non byte NAND né attribuzione esclusiva al processo.

| Percorso | Modalità | Attesi con warm-up GiB | Linux block GiB | Windows USB GiB | Windows interno GiB |
|---|---|---:|---:|---:|---:|
| external-ext4 | async | 51.250 | 51.250 | 51.250 | 6.380 |
| external-ext4 | cpu | 50.625 | 50.625 | 50.625 | 11.232 |
| external-ext4 | io | 50.000 | 50.000 | 50.000 | 5.419 |
| internal-ext4 | async | 51.250 | 51.274 | 0.000 | 51.567 |
| internal-ext4 | cpu | 50.625 | 50.628 | 0.000 | 50.713 |
| internal-ext4 | io | 50.000 | 50.001 | 0.000 | 50.236 |

**Memoria e condizioni.** Errore massimo registrato zero in tutte le passate di calcolo. Nessuno swap del processo Linux osservato. RSS è il massimo HWM per processo; VRAM NVML globale campionata e snapshot CUDA sono misure distinte e non sommabili.

| Percorso | RSS max MiB | Pinned max MiB | Device esplicito max MiB | NVML globale max MiB | Snapshot CUDA max MiB | GPU max °C | GPU max W | pswpin/out pagine |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| external-ext4 | 473.31 | 128.06 | 128.07 | 226.0 | 943.31 | 51.5 | 8.95 | 0/0 |
| internal-ext4 | 473.36 | 128.06 | 128.07 | 226.0 | 943.31 | 51.7 | 17.60 | 0/0 |

Windows ai confini dei casi: memoria disponibile 377–2671 MiB; pagefile in uso 6618–15610 MiB. Prima della preparazione: 6618 MiB; alla fine: 7019 MiB. Paging host già presente e aumentato durante preparazione/prima serie: assenza di VmSwap Linux non significa assenza di paging Windows. Questi sono snapshot, non picchi continui né attribuzione al benchmark. Il carico host limita l’interpretazione delle prestazioni.

**Cache e ripetibilità.** SHA256 e copia hanno letto il dataset prima delle prove. O_DIRECT evita la page cache dati Linux, non prova l’assenza di cache Windows/controller. Prima passata e ripetizioni sono riportate separatamente; nessuna passata è certificata cold. Ordine seed 20261107 salvato in order.json: per questo seed i tre casi esterni precedono i tre interni; non c’è alternanza fra percorsi. Alimentazione rete e piano Bilanciato registrati nei JSON; clock dinamici, nessun controllo termico e temperatura SSD non misurata. Servirebbe una futura sessione con host meno carico e ordine controbilanciato per una stima più pulita.

**Artefatti e limiti.** run.py registra comandi, inventario e ordine; i JSON/CSV/log e la telemetria sono nelle sottocartelle external-ext4 e internal-ext4. run.py è una registrazione monouso e rifiuta artefatti già presenti; per ripetere occorre una nuova directory. Il dataset esterno da 5 GiB e il fixture piccolo sono lasciati intatti nel percorso di dataset-verified.json. Nessun nuovo VHDX, modifica a mount/driver/configurazioni o checkpoint scaricato. Preservazione di tutti i file preesistenti verificata tramite SHA256. Questo test sequenziale non misura accessi selettivi MoE, latenze p95 di tile casuali, token/s o eseguibilità di un LLM completo.
