---
status: accepted
date: 2026-11-20
---

# Coordinatore in Python e kernel nativi C a foglia, interfaccia C stabile

Prima del Percorso additivo (#41–#43) abbiamo valutato due strade. La prima era un core nativo in Rust per Coordinatore, KV, scheduling e I/O, riusando i kernel C. La seconda era tenere il Coordinatore in Python e spostare in C solo le operazioni critiche. **Decidiamo la seconda.** Il Coordinatore e il Gestore delle regioni restano in Python. Le nuove operazioni critiche sono kernel C a foglia dietro un'interfaccia C stabile, così un'implementazione futura in un altro linguaggio, Rust compreso, può sostituirle o chiamarle senza buttare codice. CLI, Hugging Face e integrazioni restano in Python.

## Motivazione

- **Un solo Coordinatore (#27).** Il Gestore delle regioni del Percorso congelato (`core.Coordinator`) è Python, irrobustito in M7–M10 e non modificabile. Un core Rust per il Percorso additivo avrebbe due strade, entrambe scartate:
  - una seconda autorità su prenotazioni, lease e I/O nello stesso processo;
  - un Coordinatore Rust che pilota codice Python congelato.
- **Python non è il costo dominante, secondo le misure.** Nel decode del 14B (M10, R1) il residuo fuori da espansione, GEMV e letture è circa 0,9 s su 23,3 s (circa 4%). Il prefill lento è algoritmico (un token per volta, rilettura completa dei pesi) e si risolve con il prefill a blocchi in qualunque linguaggio.
- **Il GIL non blocca il parallelismo per righe.** `tools/adr0002_row_parallel.py` chiama il kernel pinned, invariato, da un pool di thread Python, uno per gruppo di righe disgiunto. Ripete la misura per 1, 2, 4, 6, 8 e 12 thread. La forma è quella di gate/up del Qwen2.5-0.5B (4864×896), con 32 posizioni per tile espanso. Risultati sull'host quieto:
  - linear ×32 scende da 77,9 ms a 1 thread a 16,0 ms a 8 thread (4,9×);
  - a 12 thread sale a 20,8 ms;
  - pesi e output sono bit-identici a 1 thread per ogni numero di thread;
  - una chiamata di `NativeMath.linear` costa circa 2,1 µs;
  - l'espansione BF16→FP64 satura a circa 2× già con 4 thread, perché il limite è la banda della memoria.

  Una seconda esecuzione, con l'host carico da altri processi (load average 12), ha tempi circa 3× più alti, ma la stessa accelerazione (4,9× a 8 thread) e la stessa bit-identità. Condizioni: WSL2, i7-12700H, pesi sintetici, memoria calda. Temperatura e alimentazione **NON MISURATE**.
- **Stima per v0.1.** Un prompt di 5.317 token sul 0.5B, anche chiamando il kernel una volta per posizione, costa circa 4 s di overhead Python. Il calcolo a 8 thread costa circa 300 s, quindi l'overhead è circa l'1%. È una stima, non una misura: la sostituiscono le misure di #43.
- **Costi di Rust oggi.**
  - Toolchain assente sull'host.
  - Packaging con PyO3/maturin.
  - Le installazioni da sorgente richiedono cargo, oppure ricadono sul fallback Python.
  - Due linguaggi compilati da mantenere, perché il kernel C pinned resta.
  - Il riuso più probabile (ggml/llama.cpp, kernel CUDA) è C/C++.
- **Bit-identità e portabilità sono equivalenti.** C usa `-ffp-contract=off` e Rust non contrae per default. Entrambi usano la libm della piattaforma. C compila con qualunque `cc`, e il fallback senza compilatore è già verificato in CI.

## Contratto dei kernel nativi

- Funzioni pure su arene possedute dal chiamante: niente allocazioni, niente I/O, niente thread, niente stato globale.
- Simboli `inwime_*` dichiarati in un header versionato. Sorgenti separati da `src/cpu_kernel.c`, che resta pinned.
- Le forme vengono validate in Python prima della chiamata e ricontrollate nel kernel, perché i checkpoint non sono fidati.
- La cancellazione usa un flag condiviso, controllato con una frequenza dichiarata.
- La partizione per righe la decide il Coordinatore in Python: ogni thread scrive righe di output disgiunte, senza riduzioni concorrenti. Un pool di thread nativo diventa una capacità separata, con un proprio gate, solo se le misure di #43 mostrano un costo di distribuzione del lavoro eccessivo.
- Il fallback Python puro resta.

## Considered Options

- **Core nativo Rust (Coordinatore, KV, scheduling, I/O) con riuso dei kernel C.** Rinviato. Vincerebbe su memory safety e thread sicuri se il codice nativo diventasse complesso e con stato condiviso. Con kernel a foglia questo vantaggio si riduce molto, mentre il costo e il conflitto con #27 restano.
- **Kernel a foglia scritti in Rust invece che in C.** Rinviato per gli stessi costi di toolchain e packaging. L'interfaccia C lo permette in futuro, kernel per kernel.
- **Coordinatore in Python e kernel C a foglia.** Scelto.

## Quando riaprire la decisione

Basta una di queste condizioni:

1. In un run certificato il tempo di Python fuori dalle chiamate native e dall'I/O supera il 10% del tempo totale, dopo il prefill a blocchi.
2. Una capacità richiede codice nativo con stato condiviso tra thread: io_uring/O_DIRECT, stream GPU, paginazione degli expert MoE (#11, #15).
3. Serve un binario autonomo senza Python, per esempio l'appliance di #13.
4. I gate o il fuzzing trovano più di un difetto di memory safety nel codice C.

In quel caso Rust entra per i nuovi sottosistemi nativi, dietro la stessa interfaccia C. Il Coordinatore resta quello Python, salvo un nuovo ADR.
