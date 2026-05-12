from __future__ import annotations

from typing import Dict, Tuple


# Keyed by:
#   (Messages.key, fragment_index)
#
# The Messages.key value is the pipe-joined technique key from the localization master,
# for example:
#   singles-2|singles-3
#   x-wings|x-wings-3|x-wings-4
#
# Every translation must preserve the exact same placeholder set as English.
MESSAGE_TRANSLATIONS: Dict[Tuple[str, int], Dict[str, str]] = {
    ("singles-1", 1): {
        "fr": "Lorsqu'on observe {dim}, {char} est le seul chiffre manquant dans la case {cell}. ",
        "de": "Betrachten wir {dim}, ist {char} der einzige fehlende Wert in Zelle {cell}. ",
        "it": "Osservando {dim}, {char} è l'unico valore mancante nella cella {cell}. ",
        "es": "Al observar {dim}, {char} es el único valor faltante en la celda {cell}. ",
    },
    ("singles-2|singles-3", 1): {
        "fr": "Il n'existe qu'une seule position pour {char} dans {dim}. Les autres cases de {dim} ne peuvent pas contenir {char}, car leurs {dims} contiennent déjà {char}. ",
        "de": "Es gibt nur eine einzige Position für {char} in {dim}. Die anderen Zellen in {dim} können {char} nicht enthalten, weil ihre {dims} bereits {char} enthalten. ",
        "it": "C'è una sola posizione per {char} in {dim}. Le altre celle in {dim} non possono contenere {char}, perché le loro {dims} contengono già {char}. ",
        "es": "Hay una única posición para {char} en {dim}. Las otras celdas en {dim} no pueden contener {char}, porque sus {dims} ya contienen {char}. ",
    },
    ("singles-naked-2|singles-naked-3", 1): {
        "fr": "Lorsqu'on observe {dims}, {char} est le seul chiffre manquant dans la case {cell}. ",
        "de": "Betrachten wir {dims}, ist {char} der einzige fehlende Wert in Zelle {cell}. ",
        "it": "Osservando {dims}, {char} è l'unico valore mancante nella cella {cell}. ",
        "es": "Al observar {dims}, {char} es el único valor faltante en la celda {cell}. ",
    },
    ("doubles|triplets|quads", 1): {
        "fr": "Les cases {cells} sont les seules positions possibles pour les candidats {chars_and} dans {dim}, car les autres positions ne peuvent pas être {chars_or}. ",
        "de": "Die Zellen {cells} sind die einzigen Positionen für die Kandidaten {chars_and} in {dim}, weil die anderen Positionen nicht {chars_or} sein können. ",
        "it": "Le celle {cells} sono le uniche posizioni per i candidati {chars_and} in {dim}, perché le altre posizioni non possono essere {chars_or}. ",
        "es": "Las celdas {cells} son las únicas posiciones posibles para los candidatos {chars_and} en {dim}, porque las demás posiciones no pueden ser {chars_or}. ",
    },
    ("singles-pointing", 1): {
        "fr": "Lorsqu'on observe {dim}, {char} est limité à {size} cases de la région {box} alignées {dir}; on peut donc retirer {num} {_position_s_} pour {char} en {cells}. ",
        "de": "Betrachten wir {dim}, liegt {char} in einer von {size} Zellen in Block {box} und zeigt {dir}; dadurch werden {num} {_position_s_} für {char} in {cells} entfernt. ",
        "it": "Osservando {dim}, {char} si trova in una delle {size} celle della regione {box} e punta {dir}; quindi si possono rimuovere {num} {_position_s_} per {char} in {cells}. ",
        "es": "Al observar {dim}, {char} queda limitado a una de {size} celdas de la caja {box} y apunta {dir}; por tanto se eliminan {num} {_position_s_} para {char} en {cells}. ",
    },
    ("singles-boxed", 1): {
        "fr": "Lorsqu'on observe {dim}, {char} n'est possible que dans {size} cases de la région {box}, car les autres cases de {dim} ne peuvent pas contenir {char}. ",
        "de": "Betrachten wir {dim}, kann {char} nur in einer von {size} Zellen in Block {box} stehen, weil die anderen Zellen von {dim} {char} nicht enthalten können. ",
        "it": "Osservando {dim}, {char} può trovarsi solo in una delle {size} celle della regione {box}, perché le altre celle di {dim} non possono contenere {char}. ",
        "es": "Al observar {dim}, {char} solo puede estar en una de {size} celdas de la caja {box}, porque las demás celdas de {dim} no pueden contener {char}. ",
    },
    ("x-wings|x-wings-3|x-wings-4", 1): {
        "fr": "Lorsqu'on observe {dims} {vals}, la valeur {char} est limitée exactement à {size} {dims_help} {vals_help}. Chacun de ces {size} {dims_help} contient donc un {char} qui est limité uniquement à {dims} {vals}. Par conséquent, {char} peut être retiré des autres cases de {dims_help} {vals_help}. ",
        "de": "Betrachten wir {dims} {vals}, ist der Wert {char} genau auf {size} {dims_help} {vals_help} beschränkt. Daher enthält jede dieser {size} {dims_help} ein {char}, das nur auf {dims} {vals} beschränkt ist. Deshalb kann {char} aus den anderen Zellen von {dims_help} {vals_help} entfernt werden. ",
        "it": "Osservando {dims} {vals}, il valore {char} è limitato esattamente a {size} {dims_help} {vals_help}. Quindi ciascuna di queste {size} {dims_help} contiene un {char} limitato soltanto a {dims} {vals}. Pertanto {char} può essere rimosso dalle altre celle di {dims_help} {vals_help}. ",
        "es": "Al observar {dims} {vals}, el valor {char} queda restringido exactamente a {size} {dims_help} {vals_help}. Por eso, cada una de esas {size} {dims_help} contiene un {char} restringido únicamente a {dims} {vals}. Por tanto, {char} puede eliminarse de las demás celdas de {dims_help} {vals_help}. ",
    },
    ("ab-chains", 1): {
        "fr": "Lorsqu'on observe {cells}, si la valeur dans l'une des deux cases n'est pas {char}, alors la valeur dans l'autre case doit être {char}. Par conséquent, la case {cell} ne peut pas contenir {char}. ",
        "de": "Betrachten wir {cells}: Wenn der Wert in einer der beiden Zellen nicht {char} ist, dann muss der Wert in der anderen Zelle {char} sein. Daher kann Zelle {cell} {char} nicht enthalten. ",
        "it": "Osservando {cells}, se il valore in una delle due celle non è {char}, allora il valore nell'altra cella deve essere {char}. Pertanto la cella {cell} non può contenere {char}. ",
        "es": "Al observar {cells}, si el valor en una de las dos celdas no es {char}, entonces el valor en la otra celda debe ser {char}. Por tanto, la celda {cell} no puede contener {char}. ",
    },
    ("remote-pairs", 1): {
        "fr": "Lorsqu'on observe {cells}, si la valeur dans l'une des deux cases n'est pas {char_1}, alors la valeur dans l'autre case doit être {char_1}; et si la valeur dans l'une des deux cases n'est pas {char_2}, alors la valeur dans l'autre case doit être {char_2}. Par conséquent, la case {cell} ne peut contenir ni {char_1} ni {char_2}. ",
        "de": "Betrachten wir {cells}: Wenn der Wert in einer der beiden Zellen nicht {char_1} ist, dann muss der Wert in der anderen Zelle {char_1} sein; und wenn der Wert in einer der beiden Zellen nicht {char_2} ist, dann muss der Wert in der anderen Zelle {char_2} sein. Daher kann Zelle {cell} weder {char_1} noch {char_2} enthalten. ",
        "it": "Osservando {cells}, se il valore in una delle due celle non è {char_1}, allora il valore nell'altra cella deve essere {char_1}; e se il valore in una delle due celle non è {char_2}, allora il valore nell'altra cella deve essere {char_2}. Pertanto la cella {cell} non può contenere né {char_1} né {char_2}. ",
        "es": "Al observar {cells}, si el valor en una de las dos celdas no es {char_1}, entonces el valor en la otra celda debe ser {char_1}; y si el valor en una de las dos celdas no es {char_2}, entonces el valor en la otra celda debe ser {char_2}. Por tanto, la celda {cell} no puede contener ni {char_1} ni {char_2}. ",
    },
    ("y-wings", 1): {
        "fr": "Lorsqu'on observe les cases {cells}, quelle que soit la valeur de la case {cell} parmi {chars}, l'une des deux autres cases {cells_wings} doit contenir {char}; ainsi, {char} peut être retiré de {_cell_s_} {cells_removed}. ",
        "de": "Betrachten wir die Zellen {cells}: Unabhängig davon, ob der Wert in Zelle {cell} {chars} ist, muss eine der beiden anderen Zellen {cells_wings} den Wert {char} enthalten. Daher kann {char} aus {_cell_s_} {cells_removed} entfernt werden. ",
        "it": "Osservando le celle {cells}, qualunque sia il valore della cella {cell} tra {chars}, una delle altre due celle {cells_wings} deve contenere {char}; quindi {char} può essere rimosso da {_cell_s_} {cells_removed}. ",
        "es": "Al observar las celdas {cells}, sea cual sea el valor de la celda {cell} entre {chars}, una de las otras dos celdas {cells_wings} debe contener {char}; por tanto, {char} puede eliminarse de {_cell_s_} {cells_removed}. ",
    },
    ("doubles-naked", 1): {
        "fr": "Les cases {cells} contiennent exactement les deux mêmes candidats {chars} dans {dim}. ",
        "de": "Die Zellen {cells} enthalten in {dim} genau dieselben zwei Kandidaten {chars}. ",
        "it": "Le celle {cells} contengono esattamente gli stessi due candidati {chars} in {dim}. ",
        "es": "Las celdas {cells} contienen exactamente los mismos dos candidatos {chars} en {dim}. ",
    },
    ("triplets-naked|quads-naked", 1): {
        "fr": "Les cases {cells} contiennent collectivement exactement {size} candidats {chars} dans {dim}. ",
        "de": "Die Zellen {cells} enthalten zusammen genau {size} Kandidaten {chars} in {dim}. ",
        "it": "Le celle {cells} contengono complessivamente esattamente {size} candidati {chars} in {dim}. ",
        "es": "Las celdas {cells} contienen en conjunto exactamente {size} candidatos {chars} en {dim}. ",
    },
    ("boxed-doubles", 1): {
        "fr": "Lorsqu'on observe l'interaction entre la case {cell} et la région {box}, si la valeur de la case {cell} est {char_1}, alors la valeur de la case {cell_target} doit être {char_1}; et si la valeur de la case {cell} est {char_2}, alors la valeur de la case {cell_target} doit être {char_2}. Par conséquent, la case {cell_target} ne peut contenir que les valeurs {chars_target}. ",
        "de": "Betrachten wir die Wechselwirkung zwischen Zelle {cell} und Block {box}: Wenn der Wert in Zelle {cell} {char_1} ist, dann muss der Wert in Zelle {cell_target} {char_1} sein; und wenn der Wert in Zelle {cell} {char_2} ist, dann muss der Wert in Zelle {cell_target} {char_2} sein. Daher kann Zelle {cell_target} nur die Werte {chars_target} enthalten. ",
        "it": "Osservando l'interazione tra la cella {cell} e la regione {box}, se il valore nella cella {cell} è {char_1}, allora il valore nella cella {cell_target} deve essere {char_1}; e se il valore nella cella {cell} è {char_2}, allora il valore nella cella {cell_target} deve essere {char_2}. Pertanto la cella {cell_target} può contenere solo i valori {chars_target}. ",
        "es": "Al observar la interacción entre la celda {cell} y la caja {box}, si el valor de la celda {cell} es {char_1}, entonces el valor de la celda {cell_target} debe ser {char_1}; y si el valor de la celda {cell} es {char_2}, entonces el valor de la celda {cell_target} debe ser {char_2}. Por tanto, la celda {cell_target} solo puede contener los valores {chars_target}. ",
    },
    ("boxed-triplets|boxed-quads", 1): {
        "fr": "Lorsqu'on observe l'interaction entre les {num} cases {cells} et la région {box}, si les cases {cells} sont soit {chars}, alors la case {cell_target} doit être {chars_target}. Par conséquent, la case {cell_target} ne peut contenir que les valeurs {chars_target}. ",
        "de": "Betrachten wir die Wechselwirkung zwischen den {num} Zellen {cells} und Block {box}: Wenn die Zellen {cells} entweder {chars} sind, dann muss Zelle {cell_target} {chars_target} sein. Daher kann Zelle {cell_target} nur die Werte {chars_target} enthalten. ",
        "it": "Osservando l'interazione tra le {num} celle {cells} e la regione {box}, se le celle {cells} sono {chars}, allora la cella {cell_target} deve essere {chars_target}. Pertanto la cella {cell_target} può contenere solo i valori {chars_target}. ",
        "es": "Al observar la interacción entre las {num} celdas {cells} y la caja {box}, si las celdas {cells} son {chars}, entonces la celda {cell_target} debe ser {chars_target}. Por tanto, la celda {cell_target} solo puede contener los valores {chars_target}. ",
    },
    ("boxed-wings", 1): {
        "fr": "Lorsqu'on observe la case {cell_row} dans la ligne {row} et la case {cell_col} dans la colonne {col}, ainsi que leur interaction avec la région {box}, si la valeur de la case {cell_row} n'est pas {char}, alors la valeur de la case {cell_col} doit être {char}; et si la valeur de la case {cell_col} n'est pas {char}, alors la valeur de la case {cell_row} doit être {char}. Par conséquent, {char} peut être retiré de la case {cell}. ",
        "de": "Betrachten wir Zelle {cell_row} in Zeile {row} und Zelle {cell_col} in Spalte {col} sowie ihre Wechselwirkung mit Block {box}: Wenn der Wert in Zelle {cell_row} nicht {char} ist, dann muss der Wert in Zelle {cell_col} {char} sein; und wenn der Wert in Zelle {cell_col} nicht {char} ist, dann muss der Wert in Zelle {cell_row} {char} sein. Daher kann {char} aus Zelle {cell} entfernt werden. ",
        "it": "Osservando la cella {cell_row} nella riga {row} e la cella {cell_col} nella colonna {col}, insieme alla loro interazione con la regione {box}, se il valore nella cella {cell_row} non è {char}, allora il valore nella cella {cell_col} deve essere {char}; e se il valore nella cella {cell_col} non è {char}, allora il valore nella cella {cell_row} deve essere {char}. Pertanto {char} può essere rimosso dalla cella {cell}. ",
        "es": "Al observar la celda {cell_row} en la fila {row} y la celda {cell_col} en la columna {col}, junto con su interacción con la caja {box}, si el valor de la celda {cell_row} no es {char}, entonces el valor de la celda {cell_col} debe ser {char}; y si el valor de la celda {cell_col} no es {char}, entonces el valor de la celda {cell_row} debe ser {char}. Por tanto, {char} puede eliminarse de la celda {cell}. ",
    },
    ("boxed-rays", 1): {
        "fr": "La valeur {char} dans la région {box_ray} se trouve soit dans le groupe horizontal de cases {cells_hor}, soit dans le groupe vertical de cases {cells_ver}. Si {char} se trouve dans le groupe horizontal de cases {cells_hor}, alors {char} doit aussi se trouver dans le groupe vertical de cases {cells_ver_target} de la région {box_target}. Mais si {char} se trouve plutôt dans le groupe vertical de cases {cells_ver}, alors {char} doit aussi se trouver dans le groupe horizontal de cases {cells_hor_target} de la région {box_target}. Dans tous les cas, {char} ne peut être que dans 1 de {num} {_cell_s_} {cells} dans la région {box_target}. Par conséquent, {char} peut être retiré de {_cell_s_remove_} {cells_remove}. ",
        "de": "Der Wert {char} in Block {box_ray} liegt entweder in der horizontalen Zellgruppe {cells_hor} oder in der vertikalen Zellgruppe {cells_ver}. Wenn {char} in der horizontalen Zellgruppe {cells_hor} liegt, dann muss {char} auch in der vertikalen Zellgruppe {cells_ver_target} in Block {box_target} liegen. Liegt {char} dagegen in der vertikalen Zellgruppe {cells_ver}, dann muss {char} auch in der horizontalen Zellgruppe {cells_hor_target} in Block {box_target} liegen. In jedem Fall kann {char} in Block {box_target} nur in 1 von {num} {_cell_s_} {cells} stehen. Daher kann {char} aus {_cell_s_remove_} {cells_remove} entfernt werden. ",
        "it": "Il valore {char} nella regione {box_ray} si trova o nel gruppo orizzontale di celle {cells_hor} oppure nel gruppo verticale di celle {cells_ver}. Se {char} si trova nel gruppo orizzontale di celle {cells_hor}, allora {char} deve trovarsi anche nel gruppo verticale di celle {cells_ver_target} nella regione {box_target}. Se invece {char} si trova nel gruppo verticale di celle {cells_ver}, allora {char} deve trovarsi anche nel gruppo orizzontale di celle {cells_hor_target} nella regione {box_target}. In ogni caso, {char} può trovarsi solo in 1 di {num} {_cell_s_} {cells} nella regione {box_target}. Pertanto {char} può essere rimosso da {_cell_s_remove_} {cells_remove}. ",
        "es": "El valor {char} en la caja {box_ray} está contenido o bien en el grupo horizontal de celdas {cells_hor}, o bien en el grupo vertical de celdas {cells_ver}. Si {char} está en el grupo horizontal de celdas {cells_hor}, entonces {char} también debe estar en el grupo vertical de celdas {cells_ver_target} de la caja {box_target}. Pero si {char} está en cambio en el grupo vertical de celdas {cells_ver}, entonces {char} también debe estar en el grupo horizontal de celdas {cells_hor_target} de la caja {box_target}. En cualquier caso, {char} solo puede estar en 1 de {num} {_cell_s_} {cells} en la caja {box_target}. Por tanto, {char} puede eliminarse de {_cell_s_remove_} {cells_remove}. ",
    },
    ("ab-rings", 1): {
        "fr": "Les cases {cells} contiennent chacune exactement deux valeurs et, ensemble, exactement quatre candidats {chars}. ",
        "de": "Die Zellen {cells} enthalten jeweils genau zwei Werte und zusammen genau vier Kandidaten {chars}. ",
        "it": "Le celle {cells} contengono ciascuna esattamente due valori e complessivamente esattamente quattro candidati {chars}. ",
        "es": "Las celdas {cells} contienen exactamente dos valores cada una y, en conjunto, exactamente cuatro candidatos {chars}. ",
    },
    ("ab-rings", 2): {
        "fr": "Si la valeur de l'une des deux cases {cells} dans {dim} n'est pas {char}, alors la valeur de l'autre case doit être {char}. Par conséquent, {char} peut être retiré de toutes les autres cases de {dim}. ",
        "de": "Wenn der Wert in einer der beiden Zellen {cells} in {dim} nicht {char} ist, dann muss der Wert in der anderen Zelle {char} sein. Daher kann {char} aus allen anderen Zellen in {dim} entfernt werden. ",
        "it": "Se il valore in una delle due celle {cells} in {dim} non è {char}, allora il valore nell'altra cella deve essere {char}. Pertanto {char} può essere rimosso da tutte le altre celle in {dim}. ",
        "es": "Si el valor en una de las dos celdas {cells} en {dim} no es {char}, entonces el valor en la otra celda debe ser {char}. Por tanto, {char} puede eliminarse de todas las demás celdas en {dim}. ",
    },
    ("leftovers-1|leftovers-2|leftovers-3|leftovers-4|leftovers-5|leftovers-6|leftovers-7|leftovers-8|leftovers-9", 1): {
        "fr": "Lorsqu'on observe les maisons constituées de {dims}, on peut identifier {num} {_in_s_} {cells_in} et {num} {_out_s_} {cells_out}. Les {num} {_in_s_} {_contain_s_} {_candidate_s_in_} {chars_in}, et les {num} {_out_s_} {_contain_s_} {_candidate_s_out_} {chars_out}. ",
        "de": "Betrachten wir die Bereiche, die aus {dims} bestehen, können wir {num} {_in_s_} {cells_in} und {num} {_out_s_} {cells_out} erkennen. Die {num} {_in_s_} {_contain_s_} {_candidate_s_in_} {chars_in}, und die {num} {_out_s_} {_contain_s_} {_candidate_s_out_} {chars_out}. ",
        "it": "Osservando le case composte da {dims}, possiamo identificare {num} {_in_s_} {cells_in} e {num} {_out_s_} {cells_out}. Le {num} {_in_s_} {_contain_s_} {_candidate_s_in_} {chars_in}, e le {num} {_out_s_} {_contain_s_} {_candidate_s_out_} {chars_out}. ",
        "es": "Al observar las casas formadas por {dims}, podemos identificar {num} {_in_s_} {cells_in} y {num} {_out_s_} {cells_out}. Las {num} {_in_s_} {_contain_s_} {_candidate_s_in_} {chars_in}, y las {num} {_out_s_} {_contain_s_} {_candidate_s_out_} {chars_out}. ",
    },
    ("leftovers-1|leftovers-2|leftovers-3|leftovers-4|leftovers-5|leftovers-6|leftovers-7|leftovers-8|leftovers-9", 2): {
        "fr": "Donc {}. ",
        "de": "Also {}. ",
        "it": "Quindi {}. ",
        "es": "Así que {}. ",
    },
    ("leftovers-1|leftovers-2|leftovers-3|leftovers-4|leftovers-5|leftovers-6|leftovers-7|leftovers-8|leftovers-9", 3): {
        "fr": "{chars} peut être retiré des {num} {_in_s_out_s_}",
        "de": "{chars} kann aus den {num} {_in_s_out_s_} entfernt werden",
        "it": "{chars} può essere rimosso dalle {num} {_in_s_out_s_}",
        "es": "{chars} puede eliminarse de las {num} {_in_s_out_s_}",
    },
}


RATING_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "Easy": {"fr": "Facile", "de": "Leicht", "it": "Facile", "es": "Fácil"},
    "Medium": {"fr": "Moyen", "de": "Mittel", "it": "Medio", "es": "Medio"},
    "Hard": {"fr": "Difficile", "de": "Schwer", "it": "Difficile", "es": "Difícil"},
}


TECHNIQUE_NAME_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "singles-1": {"fr": "Placement direct", "de": "Offener Single", "it": "Singolo diretto", "es": "Candidato directo"},
    "singles-2": {"fr": "Candidat caché, 2 directions", "de": "Versteckter Single, 2 Richtungen", "it": "Singolo nascosto, 2 direzioni", "es": "Candidato oculto, 2 direcciones"},
    "singles-3": {"fr": "Candidat caché, 3 directions", "de": "Versteckter Single, 3 Richtungen", "it": "Singolo nascosto, 3 direzioni", "es": "Candidato oculto, 3 direcciones"},
    "singles-naked-2": {"fr": "Candidat nu, 2 directions", "de": "Nackter Single, 2 Richtungen", "it": "Singolo nudo, 2 direzioni", "es": "Candidato desnudo, 2 direcciones"},
    "singles-naked-3": {"fr": "Candidat nu, 3 directions", "de": "Nackter Single, 3 Richtungen", "it": "Singolo nudo, 3 direzioni", "es": "Candidato desnudo, 3 direcciones"},
    "doubles": {"fr": "Paire cachée", "de": "Verstecktes Paar", "it": "Coppia nascosta", "es": "Par oculto"},
    "triplets": {"fr": "Triplet caché", "de": "Verstecktes Tripel", "it": "Tripla nascosta", "es": "Trío oculto"},
    "quads": {"fr": "Quadruplet caché", "de": "Verstecktes Quartett", "it": "Quadrupla nascosta", "es": "Cuarteto oculto"},
    "singles-pointing": {"fr": "Pointage {PLACEHOLDER}", "de": "Pointing {PLACEHOLDER}", "it": "Puntamento {PLACEHOLDER}", "es": "Apuntado {PLACEHOLDER}"},
    "singles-boxed": {"fr": "Réclamation {PLACEHOLDER}", "de": "Claiming {PLACEHOLDER}", "it": "Claiming {PLACEHOLDER}", "es": "Reclamación {PLACEHOLDER}"},
    "x-wings": {"fr": "X-Wing", "de": "X-Wing", "it": "X-Wing", "es": "X-Wing"},
    "x-wings-3": {"fr": "Swordfish", "de": "Swordfish", "it": "Swordfish", "es": "Swordfish"},
    "x-wings-4": {"fr": "Jellyfish", "de": "Jellyfish", "it": "Jellyfish", "es": "Jellyfish"},
    "ab-chains": {"fr": "Chaîne XY", "de": "XY-Kette", "it": "Catena XY", "es": "Cadena XY"},
    "remote-pairs": {"fr": "Paires distantes", "de": "Entfernte Paare", "it": "Coppie remote", "es": "Pares remotos"},
    "y-wings": {"fr": "Y-Wing", "de": "Y-Wing", "it": "Y-Wing", "es": "Y-Wing"},
    "doubles-naked": {"fr": "Paire nue", "de": "Nacktes Paar", "it": "Coppia nuda", "es": "Par desnudo"},
    "triplets-naked": {"fr": "Triplet nu", "de": "Nacktes Tripel", "it": "Tripla nuda", "es": "Trío desnudo"},
    "quads-naked": {"fr": "Quadruplet nu", "de": "Nacktes Quartett", "it": "Quadrupla nuda", "es": "Cuarteto desnudo"},
    "boxed-doubles": {"fr": "Paires miroir", "de": "Spiegelpaare", "it": "Coppie speculari", "es": "Pares espejo"},
    "boxed-triplets": {"fr": "Triplets miroir", "de": "Spiegeltripel", "it": "Triple speculari", "es": "Tríos espejo"},
    "boxed-quads": {"fr": "Quadruplets miroir", "de": "Spiegelquartette", "it": "Quadruple speculari", "es": "Cuartetos espejo"},
    "boxed-wings": {"fr": "Pivot de région et pinces", "de": "Block-Pivot und Zangen", "it": "Perno di regione e pinze", "es": "Pivote de caja y pinzas"},
    "boxed-rays": {"fr": "Réclamation multidirectionnelle", "de": "Mehrdirektionales Claiming", "it": "Claiming multidirezionale", "es": "Reclamación multidireccional"},
    "ab-rings": {"fr": "Anneau XY", "de": "XY-Ring", "it": "Anello XY", "es": "Anillo XY"},
    "leftovers-1": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-2": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-3": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-4": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-5": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-6": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-7": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-8": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
    "leftovers-9": {"fr": "Restes {PLACEHOLDER}", "de": "Reste {PLACEHOLDER}", "it": "Rimanenze {PLACEHOLDER}", "es": "Sobrantes {PLACEHOLDER}"},
}


HEADER_TRANSLATIONS_BY_EN: Dict[str, Dict[str, str]] = {
    "{instance_id} STEP-BY-STEP SOLUTION": {
        "fr": "{instance_id} SOLUTION ÉTAPE PAR ÉTAPE",
        "de": "{instance_id} SCHRITT-FÜR-SCHRITT-LÖSUNG",
        "it": "{instance_id} SOLUZIONE PASSO DOPO PASSO",
        "es": "{instance_id} SOLUCIÓN PASO A PASO",
    },
    '"{instance_id} STEP-BY-STEP SOLUTION"': {
        "fr": '"{instance_id} SOLUTION ÉTAPE PAR ÉTAPE"',
        "de": '"{instance_id} SCHRITT-FÜR-SCHRITT-LÖSUNG"',
        "it": '"{instance_id} SOLUZIONE PASSO DOPO PASSO"',
        "es": '"{instance_id} SOLUCIÓN PASO A PASO"',
    },
    "STEP {step_no}": {
        "fr": "ÉTAPE {step_no}",
        "de": "SCHRITT {step_no}",
        "it": "PASSO {step_no}",
        "es": "PASO {step_no}",
    },
    '"STEP {step_no}"': {
        "fr": '"ÉTAPE {step_no}"',
        "de": '"SCHRITT {step_no}"',
        "it": '"PASSO {step_no}"',
        "es": '"PASO {step_no}"',
    },
}


KEYWORD_TRANSLATIONS_BY_EN: Dict[str, Dict[str, str]] = {
    "row": {"fr": "la ligne", "de": "Zeile", "it": "riga", "es": "la fila"},
    "rows": {"fr": "les lignes", "de": "Zeilen", "it": "righe", "es": "las filas"},
    "column": {"fr": "la colonne", "de": "Spalte", "it": "colonna", "es": "la columna"},
    "columns": {"fr": "les colonnes", "de": "Spalten", "it": "colonne", "es": "las columnas"},
    "col": {"fr": "colonne", "de": "Spalte", "it": "colonna", "es": "columna"},
    "box": {"fr": "la région", "de": "Block", "it": "regione", "es": "la caja"},
    "boxes": {"fr": "les régions", "de": "Blöcke", "it": "regioni", "es": "las cajas"},
    "horizontal": {"fr": "horizontal", "de": "horizontal", "it": "orizzontale", "es": "horizontal"},
    "vertical": {"fr": "vertical", "de": "vertikal", "it": "verticale", "es": "vertical"},
    "horizontally": {"fr": "horizontalement", "de": "horizontal", "it": "orizzontalmente", "es": "horizontalmente"},
    "vertically": {"fr": "verticalement", "de": "vertikal", "it": "verticalmente", "es": "verticalmente"},
    "Easy": {"fr": "Facile", "de": "Leicht", "it": "Facile", "es": "Fácil"},
    "Medium": {"fr": "Moyen", "de": "Mittel", "it": "Medio", "es": "Medio"},
    "Hard": {"fr": "Difficile", "de": "Schwer", "it": "Difficile", "es": "Difícil"},
    "one": {"fr": "un", "de": "eins", "it": "uno", "es": "uno"},
    "two": {"fr": "deux", "de": "zwei", "it": "due", "es": "dos"},
    "three": {"fr": "trois", "de": "drei", "it": "tre", "es": "tres"},
    "four": {"fr": "quatre", "de": "vier", "it": "quattro", "es": "cuatro"},
    "five": {"fr": "cinq", "de": "fünf", "it": "cinque", "es": "cinco"},
    "six": {"fr": "six", "de": "sechs", "it": "sei", "es": "seis"},
    "seven": {"fr": "sept", "de": "sieben", "it": "sette", "es": "siete"},
    "eight": {"fr": "huit", "de": "acht", "it": "otto", "es": "ocho"},
    "nine": {"fr": "neuf", "de": "neun", "it": "nove", "es": "nueve"},
    "single": {"fr": "unique", "de": "einzeln", "it": "singolo", "es": "único"},
    "pair": {"fr": "paire", "de": "Paar", "it": "coppia", "es": "par"},
    "triple": {"fr": "triplet", "de": "Tripel", "it": "tripla", "es": "trío"},
    "quad": {"fr": "quadruplet", "de": "Quartett", "it": "quadrupla", "es": "cuarteto"},
    "and": {"fr": "et", "de": "und", "it": "e", "es": "y"},
    "or": {"fr": "ou", "de": "oder", "it": "o", "es": "o"},
    "cell": {"fr": "case", "de": "Zelle", "it": "cella", "es": "celda"},
    "cells": {"fr": "cases", "de": "Zellen", "it": "celle", "es": "celdas"},
    "position": {"fr": "position", "de": "Position", "it": "posizione", "es": "posición"},
    "positions": {"fr": "positions", "de": "Positionen", "it": "posizioni", "es": "posiciones"},
    "candidate": {"fr": "candidat", "de": "Kandidat", "it": "candidato", "es": "candidato"},
    "candidates": {"fr": "candidats", "de": "Kandidaten", "it": "candidati", "es": "candidatos"},
    "contain": {"fr": "contiennent", "de": "enthalten", "it": "contengono", "es": "contienen"},
    "contains": {"fr": "contient", "de": "enthält", "it": "contiene", "es": "contiene"},
    "innie": {"fr": "case intérieure", "de": "Innenzelle", "it": "cella interna", "es": "celda interior"},
    "innies": {"fr": "cases intérieures", "de": "Innenzellen", "it": "celle interne", "es": "celdas interiores"},
    "outie": {"fr": "case extérieure", "de": "Außenzelle", "it": "cella esterna", "es": "celda exterior"},
    "outies": {"fr": "cases extérieures", "de": "Außenzellen", "it": "celle esterne", "es": "celdas exteriores"},
}