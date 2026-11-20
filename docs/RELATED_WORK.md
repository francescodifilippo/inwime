# InWiMe — Fonti di ispirazione, precedenti tecnici e related work

**Aggiornamento:** 20 novembre 2026 (prima stesura 18 novembre). **Stato:** documentazione dei riferimenti studiati; non costituisce una rivendicazione di novità, riutilizzo di codice, compatibilità, prestazioni o autorizzazione all'implementazione.

InWiMe (*Intelligence Without Memory Limits*) persegue inferenza LLM **CPU-first**, su x86-64 con RAM limitata e pesi di precisione originale (in particolare BF16) conservati su NVMe. La GPU, la quantizzazione e la distribuzione sono percorsi opzionali. Questi requisiti consentono di usare come **related work** anche progetti con GPU obbligatoria o checkpoint quantizzati, pur senza ereditarne implicitamente il comportamento.

## Registro completo e provenienza

Il [**catalogo canonico delle fonti e tecniche**](INWIME_TECHNIQUE_WATCHLIST.md) contiene le schede **F01–F60** dell'inventario consolidato il 12 novembre 2026 e ora l'addendum **F61–F62**. Per ciascuna voce documenta sorgente, principio tecnico, possibile relazione con InWiMe e limiti. Non va sostituito con una lista abbreviata.

Nel catalogo, **[D]** è un'attribuzione a discussioni precedenti, **[R]** un approfondimento di ricerca e **[P]** un'ipotesi da verificare. Le etichette attestano provenienza documentale, **non** implementazione. Quando non sono disponibili il thread originale, la revisione upstream o una verifica indipendente, ciò rimane dichiarato. Un elenco completo di ogni conversazione passata **non è certificabile** a partire dal solo materiale disponibile.

Questo documento è il **punto di accesso pubblico** alle fonti. I collegamenti a paper, autori, implementazioni e discussion thread sono nel catalogo F01–F62 e nei documenti specifici; conservarli anche quando una tecnica viene scartata.

## Principali progetti e pubblicazioni presi in esame

| Famiglia / fonti primarie | Relazione con InWiMe | Registro / fase | Stato dell'attribuzione |
| --- | --- | --- | --- |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), [safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | File mapping, lettura selettiva, tensor metadata, shard, contratti di formato | F01, F09, F27, F34–F36; M10–M15 | Precedenti studiati; integrazione non implicita |
| [AirLLM](https://github.com/lyogavin/airllm), [ZeRO-Inference/DeepNVMe](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference), [FlexGen](https://github.com/FMInference/FlexGen), [LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) | Layer streaming, offload, cost model, locality e I/O flash | F02–F05; M10/M12 | Related work, condizioni hardware/workload differenti |
| [Colibri](https://github.com/JustVugg/colibri), [WARP](https://github.com/sqliteai/warp) | Gerarchie di residenza VRAM/RAM/NVMe, esperti MoE, caching e letture selettive | F43–F44; M11/M12 | **Già nel brief storico**; non riscoperti a novembre |
| [Strata — snapshot esaminato](https://github.com/Niko1221/Strata/commit/13eedf552417a6a791a7d13c6597e24300ee3a09) | Storage dei pesi, cache degli esperti, CPU/GPU, I/O, scheduling, costo numerico | F61; [audit indipendente](STRATA_TECH_AUDIT_2026-11-13.md); M11/M12 | Codice e documenti analizzati; nessun benchmark Strata eseguito da InWiMe |
| [Betelgeuse](https://github.com/Betelgeuse-AI/Betelgeuse), [documentazione tecnica](https://github.com/Betelgeuse-AI/Betelgeuse/blob/main/docs/fluxaeros.md) | Demand paging degli esperti MoE, pool VRAM, cache RAM, NVMe e prefetch | F62; [Issue #11](https://github.com/francescodifilippo/inwime/issues/11); M11/M12 | Fonte studiata; risultati dichiarati dagli autori, non replicati |
| [TinyGiant](https://github.com/jerryjokesalot/tinygiant), [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity), [PowerInfer](https://powerinfer.ai/), [KTransformers](https://github.com/kvcache-ai/ktransformers), [Guanaco](https://github.com/Atomic-Germ/Guanaco) | Hot/cold experts, layout, routing, trasferimenti e kernel CPU/GPU | F10–F13, F32–F39, F45; M11/M12 | Alternative e antecedenti, da confrontare per precisione e workload |
| [SpikingBrain 2.0](https://github.com/BICLab/SpikingBrain2.0), [porting CPU indipendente](https://github.com/francescodifilippo/spikingbrain-cpu) | Nuove famiglie di operazioni, attenzione e stato ricorrente, esperienze di verifica numerica | F16, F40, F50; adapter futuri | Ricerca e esperienza precedente, separata dal runtime attuale |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [oneDNN](https://github.com/uxlfoundation/oneDNN) | Metodi IO-aware, kernel e ottimizzazione numerica | F14–F15, F51; M12/M13 | Riferimenti per studio e benchmark |
| [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), DistServe e Splitwise | KV, batching, prefissi, split prefill/decode e serving | F19–F25, F29–F30; M12/M14 | Related work, non requisiti del core CPU |
| [Petals](https://github.com/bigscience-workshop/petals), [llama.cpp RPC](https://github.com/ggml-org/llama.cpp), [Ray Serve](https://docs.ray.io/en/latest/serve/), [NIXL](https://github.com/ai-dynamo/nixl) | Layer/expert parallelism, compute-near-data, memorie remote e costo di rete | F26–F31, F53; [#14](https://github.com/francescodifilippo/inwime/issues/14), [#15](https://github.com/francescodifilippo/inwime/issues/15); M15+ | Soluzioni future da comparare, non implementate |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) (`AutoConfig`/`AutoModel`), [PyTorch FX](https://docs.pytorch.org/docs/stable/fx.html) e [`torch.export`](https://docs.pytorch.org/docs/stable/export.html), [ONNX](https://onnx.ai/onnx/intro/concepts.html), [TVM](https://tvm.apache.org/) | Riconoscimento dell'architettura e importazione di grafi e operatori già esistenti; InWiMe aggiunge lowering e pianificazione out-of-core | [#28](https://github.com/francescodifilippo/inwime/issues/28); v0.2.0-alpha | **Riuso deciso il 20/11/2026**, non ancora integrato; TVM solo se il lowering lo richiede. Da registrare nel catalogo F |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) ([ricerca nel catalogo](https://huggingface.co/docs/huggingface_hub/guides/search)) | Catalogo, metadata per revisione, download e cache dei checkpoint | [#26](https://github.com/francescodifilippo/inwime/issues/26); v0.1.0-alpha | Riuso pianificato; da registrare nel catalogo F |
| [Ollama](https://github.com/ollama/ollama), [llama.cpp server](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm) | Scelta e download dei modelli, API OpenAI-compatible e streaming sono già funzionalità diffuse: InWiMe **non** le presenta come innovazioni | [#26](https://github.com/francescodifilippo/inwime/issues/26); v0.1.0-alpha | Related work per il posizionamento; confronto minimo richiesto dal gate della v0.1.0-alpha |
| [Hugging Face Accelerate — big model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) | Offload di pesi originali su CPU e disco con `device_map` | [#26](https://github.com/francescodifilippo/inwime/issues/26); M10.5/M12 | Related work per l'offloading su disco; nessun benchmark InWiMe ancora eseguito. Da registrare nel catalogo F |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | Client e interfaccia agentica esterna, distinta dall'inferenza dei pesi | F41; M14 | Integrazione pianificata, non motore out-of-core |

**Altri riferimenti conservati nel catalogo:** DwarfStar, FlashMoBA, speculative decoding (EAGLE/Medusa/MTP), H2O, StreamingLLM, ServerlessLLM, SPDK, Glinthawk, GPUDirect Storage, liburing, Vulkan, Tenstorrent, Taalas e acceleratori/SBC esplorativi. L'elenco in questa pagina è un **indice**, non sostituisce le voci F01–F62.

## Ricerca interna e contributi da attribuire separatamente

- [Audit Fable del 16–17 novembre](https://github.com/francescodifilippo/inwime/pull/16): review architetturale e prototipi sperimentali GEMV/FP64 **prodotti per InWiMe**, custoditi in una PR documentale distinta. Sono ricerca interna di confronto, non un motore esterno da cui sarebbe stato importato codice. Le prove sintetiche non dimostrano uno speedup end-to-end.
- [Oracolo indipendente M9](../tools/oracle9.py): riferimento di correttezza progettato all'interno del progetto usando upstream PyTorch/Transformers, distinto dal runtime di produzione. [Issue #12](https://github.com/francescodifilippo/inwime/issues/12) studia il module streaming come possibile backend, senza assumere che sia già stato adottato.
- [Strata — audit e limiti](STRATA_TECH_AUDIT_2026-11-13.md): la documentazione cita revisioni e file sorgente effettivamente letti, distinguendo proprietà del codice, benchmark degli autori e ipotesi per InWiMe.

## Come decidere se adottare una tecnica

1. **Citata / discussa:** fonte identificata; possibile applicazione e limiti.
2. **Analizzata:** codice, paper o architettura esaminati a revisione nota.
3. **Benchmarkata:** prova riprodotta su hardware, dtype e workload dichiarati.
4. **Adattata / integrata:** esiste una PR o un commit InWiMe che documenta l'uso reale, con test di correttezza, provenienza e licenze.
5. **Scartata / sospesa:** motivi, dati e alternative conservati.

Qualsiasi futura importazione di **codice** di terzi richiede verifica separata di licenza, copyright, avvisi e obblighi di attribuzione; una citazione bibliografica non è da sola una verifica di conformità. Le licenze dei **pesi** e dei dati sono distinte da quelle dei motori.

InWiMe deve evitare di chiamare «ispirazione implementata» una semplice lettura di un repository, di usare benchmark di terzi come misure proprie o di attribuire innovazioni tecniche generali a un unico autore. Questo registro non certifica un audit completo delle dipendenze o della storia Git.

## Manutenzione

Aggiungere ogni nuova fonte al [catalogo F01–F62](INWIME_TECHNIQUE_WATCHLIST.md) con identificativo stabile, **autore/progetto e URL primario**, data e revisione esaminata, origine storica o nuova ricerca, tecnica/idea, differenze rispetto a InWiMe, dtype/hardware, stato, eventuali benchmark locali e licensing **se verificato**. Aggiornare questo indice quando una nuova fonte diventa strutturalmente importante.

Riferimenti di roadmap: [piano delle release #29](https://github.com/francescodifilippo/inwime/issues/29), [v0.1.0-alpha #26](https://github.com/francescodifilippo/inwime/issues/26), [importatori #28](https://github.com/francescodifilippo/inwime/issues/28), [M11 #5](https://github.com/francescodifilippo/inwime/issues/5) e [M12 #6](https://github.com/francescodifilippo/inwime/issues/6). Le fonti rendono trasparente il lavoro precedente e i confronti futuri; non modificano da sole la roadmap.
