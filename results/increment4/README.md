# Evidenze incremento 4

Canonici: `summary.json`, `final-medium-*`, `final-decoder-*`,
`large-progressive-verified.json`, `large-samples-final.json`, `pytest-final.txt`.
Il rapporto completo e i comandi sono in `docs/RUNTIME_CPU_INCREMENT_4.md`.

`generation-*.json` documenta le scritture effettive. La prima generazione grande
precede il rafforzamento dell'audit cgroup: l'audit corretto con foglia e antenati
è nei risultati canonici della scansione. Entrambi i limiti sono `max`.

`medium-*` senza `final-`, `large-progressive.json`,
`large-progressive-final.json`, `large-samples.json` sono acquisizioni preliminari,
conservate per tracciabilità. Alcune furono concorrenti: non usarle per il confronto
prestazionale finale. Il primo `large-progressive.json` include anche il calcolo
del digest atteso nel wall; le versioni successive lo escludono.
Le etichette `warm` preliminari non certificano il calore delle cache, specialmente
quando `integrity=metadata`; le etichette corrette sono nei risultati canonici.

La prova grande è un hash progressivo di 6 GiB e campioni numerici da 98.304 byte.
Dimostra accesso a payload fisicamente allocato oltre RAM WSL, non un forward
completo oltre RAM né capacità oltre la RAM dell'host Windows.


Consolidamento: `pytest-consolidation.txt` conferma 152 test dopo la correzione
circoscritta del repacker e 8 casi aggiuntivi. `artifact-sha256-consolidated.json`
identifica i sorgenti consolidati; l’indice originale dell’acquisizione rimane
immutato. La prova grande non è stata ripetuta: soltanto verifiche dei metadati
filesystem e ricalcolo del riepilogo dai campioni esistenti.

Nel log del primo fallimento sono stati rimossi soltanto spazi finali delle righe;
il testo diagnostico e il risultato della prova sono conservati.
