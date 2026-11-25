# Gate locale di #38 — Qwen2 con testa legata e certificazione di base di Qwen2.5-0.5B-Instruct

Stato: **PASS** per le tre attestazioni, sulla configurazione certificata qui sotto.
Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
Base: `main` a `e974f87`. Data: 25 novembre 2026.

Questo è un gate **locale**: le misure vengono da un solo host e non si estendono ad altre CPU,
compilatori, librerie matematiche o numeri di thread. La CI ripete solo le attestazioni su
fixture (bit-identità), non quelle sul Checkpoint reale.

## Che cosa è cambiato

- `inwime/qwen2_tied.py`: `TiedQwen2Catalog` è l'adapter additivo per i Qwen2ForCausalLM con
  `tie_word_embeddings=true` che serializzano la testa una sola volta, come
  `model.embed_tokens.weight`. Lega il tensore semantico `lm_head` al descrittore
  dell'embedding, come già fa `LlamaCatalog`. Rifiuta un config tied che serializza anche
  `lm_head.weight`. Sessione, piano e calcolo sono quelli di `Qwen2Runtime`, ereditati invariati.
  `inwime_runtime/qwen2.py` (Percorso congelato) non cambia e continua a rifiutare i tied.
- Il Motore (`inwime/engine.py`) e il catalogo (`inwime/capability.py`) scelgono l'adapter
  dal config (`qwen2_catalog`).
- Il generatore delle fixture `tools/fixture_qwen10.py` (pinned) non cambia.
  `tests_inwime/tinyqwen.generator(head)` lo estende in modo additivo su un'istanza privata del
  modulo, con le varianti `tied` e `duplicated` (testa con gli stessi valori dell'embedding).
- `inwime/snapshot.py`: correzione emersa dal Checkpoint reale, vedi «Acquisizione».

## Acquisizione

`inwime models install --yes Qwen/Qwen2.5-0.5B-Instruct@7ae55760…` dopo il ricontrollo, senza
`--yes`, del piano (`install.log` riporta il piano e l'esito del run con `--yes`):

- identità: Revisione completa a 40 hex;
- manifest: 10 file, 999.604.126 byte;
- destinazione: snapshot nel cache HF;
- spazio: 62,64 GiB liberi sul volume fisico, 61,71 GiB dopo il download, margine 30 GiB.

Il download ha richiesto 98 s ed è terminato `installed and verified`, con SHA256 di ogni file.
RSS di picco del processo: 919.368 kB. Lo shard dei pesi ha 290 tensori (24 layer × 12 + 2) e
non contiene `lm_head.weight`.

**Difetto trovato e corretto.** `huggingface_hub` 2.4.0 scarica i file Xet in uno store
condiviso del cache: il file sta in `<cache>/blobs/<2 hex>/<xet hash>` e il blob del repository
`models--…/blobs/<sha256>` è un link relativo a quel file. Il loader congelato apre i blob con
`O_NOFOLLOW` e riceveva `ELOOP`, quindi il Motore rifiutava il modello installato. Il FakeHub dei
test non usa Xet, perciò #36/#37 non lo avevano visto. Ora `snapshot.blob_link` accetta il
secondo link solo nella forma esatta dello store condiviso e lega lo shard al file regolare
finale. Il contenuto resta verificato dal passaggio SHA256 completo contro il manifest
all'avvio. Due test in `tests_inwime/test_serve.py` ricreano il layout: un Checkpoint nello
store condiviso viene servito, mentre un link che esce dallo store viene rifiutato.

## Attestazione 1 — Bit-identità tra testa legata e testa duplicata (fixture controllate)

`tests_inwime/test_tied_head.py`, nella suite del prodotto e in CI:

- La fixture tied passa per `TiedQwen2Catalog`. La gemella `duplicated` (`tie_word_embeddings`
  false, `lm_head.weight` con i valori dell'embedding) passa per il `Qwen2Runtime` congelato.
  Su prefill di 5 token e 3 decode, i byte di logits, di tutti i residual e dell'intera KV sono
  identici. Tile da 7 elementi (nessun tensore ne è multiplo, quindi la testa scorre l'embedding
  in molti lease parziali) e tile da 262.144 (quello del Motore), sia con il riferimento Python
  sia con il kernel nativo `c-fp64-ordered` della configurazione certificata.
- Nessuna seconda copia persistente:
  - in memoria `tensors['lm_head'] is tensors['embedding']`;
  - ogni categoria del piano è uguale a quella della gemella, tranne i metadati, più piccoli di
    un header di tensore;
  - su disco il payload della fixture tied è 5.920 byte (37 × 80 BF16) più piccolo.
- Il Percorso congelato rifiuta ancora la fixture tied.
- Il generatore storico produce gli stessi file di prima, confrontati con gli SHA256 letterali
  di `config.json` e `safetensors.json`.
- Il Motore via HTTP (directory importata e Checkpoint installato da Hub) serve la fixture tied
  con gli stessi token greedy che il Percorso congelato dà sulla gemella duplicata.

## Attestazione 2 — Equivalenza con l'oracolo (Checkpoint reale)

Strumenti: `tools/tied38_motore.py`, lato Motore, e `tools/tied38_oracle.py`, lato oracolo, che
non importa InWiMe. Per ciascuno dei 3 casi, il Motore renderizza il prompt con il template del
Checkpoint e genera 8 token. Una sessione nuova sullo stesso Percorso computazionale cattura in
FP64 l'intera sequenza (prompt + completamento, prefill del prompt e un decode per token):

- tutti i logits (posizioni × 151.936);
- tutti i residual (24 layer);
- l'intera KV.

L'oracolo è `tools.check_oracle10.stock`, pinned e invariato: `from_pretrained` stock con
attention eager, sulla stessa sequenza e con gli stessi chunk. Il confronto è
`tools.compare10.compare`, con copertura FULL e le soglie preregistrate in M10:

| Contratto | Pesi e aritmetica | atol | rtol |
|---|---|---|---|
| `stock_fp32` (oracolo Transformers) | FP32 | 2e-3 | 2e-4 |
| `full64` (riferimento FP64) | FP64, profilo di precisione inc. 7/9 | 2e-5 | 2e-6 |

| Caso | Token (prompt + gen.) | Contratto | Esito | max \|Δ\| logits | max \|Δ\| residual | max \|Δ\| KV | utilizzo max soglia |
|---|---|---|---|---|---|---|---|
| 0 `What is the capital of Italy?` | 36 + 8 | stock_fp32 | FULL / PASS | 1,32e-4 | 5,36e-4 | 3,72e-5 | 0,053 |
| 0 | | full64 | FULL / PASS | 4,04e-13 | 1,14e-12 | 1,94e-13 | 1,9e-8 |
| 1 `Scrivi una frase sul mare.` | 36 + 8 | stock_fp32 | FULL / PASS | 1,19e-4 | 5,36e-4 | 3,75e-5 | 0,055 |
| 1 | | full64 | FULL / PASS | 4,04e-13 | 1,14e-12 | 1,94e-13 | 1,9e-8 |
| 2 system + `Name three primary colors.` | 24 + 8 | stock_fp32 | FULL / PASS | 3,46e-4 | 5,36e-4 | 8,27e-5 | 0,125 |
| 2 | | full64 | FULL / PASS | 6,47e-13 | 1,14e-12 | 1,94e-13 | 2,3e-8 |

In ogni caso e per entrambi i contratti l'argmax dei logits coincide in tutte le posizioni.
Il rendering del template di Transformers dà gli stessi id di prompt del Motore in tutti e tre
i casi. Valori completi: `oracle.json`. Digest degli array catturati: `motore.json`
(`arrays_sha256`; gli array, 172 MB, non sono versionati).

## Attestazione 3 — Parità dei token greedy

Ogni contratto genera fino a 8 token dal proprio rendering, per argmax: vince l'id più basso in
caso di parità, come fanno `torch.argmax` e la regola M10. Si ferma a un EOS di
`generation_config.json`. I token coincidono con quelli del Motore in tutti i casi e in entrambi
i contratti (PASS / PASS):

| Caso | Token del Motore | Testo | Fine |
|---|---|---|---|
| 0 | 785 6722 315 15344 374 21718 13 151645 | `The capital of Italy is Rome.` | stop (EOS) |
| 1 | 1 8747 326 8565 11 2525 650 1079 | `"La luna, come un am` | length |
| 2 | 19641 6028 7987 525 2518 11 6303 11 | `Three primary colors are red, blue,` | length |

**Prima esecuzione, non valida come parità.** Il primo run chiedeva al Motore solo
`temperature: 0`. Il `generation_config.json` di questo Checkpoint ha `repetition_penalty: 1.1`,
che il Motore applica anche al greedy (semantica Transformers, scelta in #39). Nel caso 1 il
Motore ha quindi scelto 4403 dove l'argmax puro dà 11, e la parità è risultata FAIL in entrambi
i contratti, mentre l'equivalenza numerica era PASS con argmax identico in ogni posizione. Il run
valido chiede `temperature: 0, repetition_penalty: 1` (greedy puro): la sampling effettiva di
ogni richiesta è registrata in `motore.json`.

## Memoria, I/O e tempi

| Misura | Motore (`tied38_motore`) | Oracolo (`tied38_oracle`) |
|---|---|---|
| RSS di picco (`/usr/bin/time`) | 404.732 kB | 5.295.116 kB (FP64) |
| Memoria riservata dal piano, per sessione | 69,3 MB (44 posizioni), 51,4 MB (32) | — |
| Lettura SHA256 completa dei pesi | 988.097.824 B all'avvio del Motore e a ogni sessione di cattura | — |
| Letture logiche di pesi per caso di 44 posizioni | 43,47 GB (~988 MB per posizione) | — |
| Letture effettive dal file system (blocchi da 512 B) | 225.816 (~116 MB) | 2.132.640 (~1,09 GB) |
| Wall time | 3 min 40 s (3 casi: servizio + cattura) | 56 s |

Le letture del Motore vengono quasi tutte dalla cache delle pagine, calda perché il Checkpoint
era appena stato scaricato e letto: i tempi **non** sono misure a freddo. Il Percorso congelato
rilegge tutti i pesi a ogni posizione e il prefill elabora un token per volta: circa 0,9 s per
posizione, cioè 39 s per 36 + 8 token. Il prefill a blocchi arriva con #41/#43. Temperatura e
stato di alimentazione non sono stati misurati. Memoria disponibile in WSL prima del run:
`free-before.txt`. Non sono benchmark: servono a documentare il costo del gate.

## Configurazione certificata

- CPU 12th Gen Intel Core i7-12700H (20 thread logici), WSL2 kernel
  6.18.58.1-microsoft-standard-WSL2.
- Kernel nativo `c-fp64-ordered` (`build/libinwime_cpu.so`, `make cpu-kernel`, gcc 15.2.0),
  un thread di calcolo.
- Python 3.14.4.
- Oracolo: torch 2.9.1+cpu, transformers 4.57.3, numpy 2.3.5, 4 thread torch.
- Prompt brevi (24–36 token) e 8 token generati. Questo gate non dimostra il comportamento su
  prompt lunghi come quelli di DeepSeek Harness (~5.300 token).

## Riproduzione

```
PYTHONPATH=. INWIME_CPU_KERNEL=build/libinwime_cpu.so .venv-product/bin/python \
    -m tools.tied38_motore --output $GATE/motore
HF_HUB_OFFLINE=1 .venv-oracle7/bin/python -m tools.tied38_oracle \
    --motore $GATE/motore --output $GATE/oracle.json
```

`$GATE` è una directory di lavoro fuori dal repository (circa 172 MB di array). Nei log
versionati la directory di lavoro è scritta `$GATE` e la home `~`; `tied38_motore` scrive da sé
`~` in `motore.json` e `tied38_oracle` lo espande. Le evidenze vengono dal commit `3ccb279`: lì
la diagnostica di `computation` aveva il campo `head: tied`, poi sostituito da `adapter`
(`TiedQwen2Catalog (additive, tied head)`).
