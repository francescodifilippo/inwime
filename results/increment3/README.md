# Evidenza dell'incremento 3

Acquisizione canonica: **`verification-final-clean.json`**. Codice finale, cinque
campioni/processi nuovi per coppia modalità/caso, ordine alternato, suite di test
terminata prima dell'avvio. Calcolo del carico controllato reale; solo il ritardo
di lettura è iniettato. `timeline.svg` mostra gli intervalli esportati del primo
campione controllato di ciascuna modalità, sulla stessa scala temporale.

Le acquisizioni precedenti sono conservate per trasparenza e non vanno aggregate:

- `verification.json`: prima revisione, consumatore controllato simulato con pausa;
  antecedente alle correzioni di dispatch e rilascio traccia.
- `verification-reviewed.json`: dispatch rivisto; consumatore ancora simulato con
  pausa; antecedente alle ultime correzioni della contabilità.
- `verification-final.json`: calcolo scalare reale e correzioni finali; acquisizione
  avviata insieme alla suite di test, quindi esclusa dal confronto prestazionale finale.

Tutte le prove leggono fixture piccole già scaldate dalla scansione SHA256.
`read_bytes` Linux fisici di inferenza pari a zero: non sono misure NVMe cold.
Nessun file di pesi aggiuntivo. Nessun risultato storico precedente all'incremento
è stato modificato. Metodo, limiti, test e valutazione A:
[rapporto](../../docs/RUNTIME_CPU_INCREMENT_3.md).
