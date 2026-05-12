package com.contextionary.sudoku.conductor.solving

import org.json.JSONArray
import org.json.JSONObject
import com.contextionary.sudoku.conductor.policy.FactBundleV1

/**
 * Phase 5 — Technique knowledge base (TEACHING_CARD).
 *
 * IMPORTANT:
 * - This is NOT scripted speech.
 * - It's structured knowledge for Tick2 grounding.
 * - Keep it stable, small, and evidence-oriented.
 */
object TechniqueCards {

    data class Card(
        val id: String,
        val nameEn: String,
        val nameFr: String? = null,
        val difficulty: String, // "easy" | "medium" | "hard" | "expert"
        val appliesTo: List<String>, // "ROW" | "COL" | "BOX"
        val definitionEn: String,
        val definitionFr: String? = null,
        val howToSpotEn: List<String>,
        val howToSpotFr: List<String>? = null,
        val evidenceShapesEn: List<String>,
        val evidenceShapesFr: List<String>? = null,
        val commonConfusionsEn: List<String> = emptyList(),
        val commonConfusionsFr: List<String> = emptyList()
    ) {
        fun toJson(preferredLang: String?): JSONObject {
            val lang = (preferredLang ?: "en").trim().lowercase()
            val useFr = (lang == "fr")

            fun pick(en: String, fr: String?): String = if (useFr && !fr.isNullOrBlank()) fr else en
            fun pickList(en: List<String>, fr: List<String>?): JSONArray =
                JSONArray().apply {
                    val xs = if (useFr && !fr.isNullOrEmpty()) fr else en
                    xs.forEach { put(it) }
                }

            return JSONObject().apply {
                put("schema", "teaching_card_v1")
                put("technique_id", id)
                put("technique_name", pick(nameEn, nameFr))
                put("difficulty", difficulty)
                put("applies_to", JSONArray().apply { appliesTo.forEach { put(it) } })
                put("definition", pick(definitionEn, definitionFr))
                put("how_to_spot", pickList(howToSpotEn, howToSpotFr))
                put("evidence_shapes", pickList(evidenceShapesEn, evidenceShapesFr))
                put("common_confusions", pickList(commonConfusionsEn, commonConfusionsFr))
            }
        }
    }

    // -------------------------
    // Minimal-but-real KB
    // -------------------------
    private val cards: Map<String, Card> = listOf(

        // ---- Singles ----
        Card(
            id = "singles-1",
            nameEn = "Naked Single",
            nameFr = "Single nu",
            difficulty = "easy",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "A cell has only one possible digit left, so you can place it.",
            definitionFr = "Une case n’a plus qu’un seul candidat possible : on peut placer ce chiffre.",
            howToSpotEn = listOf(
                "Look at one empty cell and list its candidates (digits not used in its row/col/box).",
                "If only one candidate remains, that digit is forced."
            ),
            howToSpotFr = listOf(
                "Regarde une case vide et liste ses candidats (chiffres absents de sa ligne/colonne/boîte).",
                "S’il ne reste qu’un candidat, ce chiffre est forcé."
            ),
            evidenceShapesEn = listOf(
                "Target cell has candidate set of size 1.",
                "Row/Col/Box already contains the other 8 digits (directly or effectively)."
            ),
            evidenceShapesFr = listOf(
                "La case cible a un ensemble de candidats de taille 1.",
                "La ligne/colonne/boîte contient déjà les 8 autres chiffres."
            ),
            commonConfusionsEn = listOf(
                "Confusing 'only one empty cell in a house' with a naked single (that’s a hidden single)."
            ),
            commonConfusionsFr = listOf(
                "Confondre 'une seule case vide dans une maison' avec un single nu (c’est plutôt un single caché)."
            )
        ),
        Card(
            id = "singles-2",
            nameEn = "Hidden Single",
            nameFr = "Single caché",
            difficulty = "easy",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "In a row/column/box, a digit can go in only one cell—even if that cell has multiple candidates.",
            definitionFr = "Dans une ligne/colonne/boîte, un chiffre ne peut aller que dans une seule case, même si la case a plusieurs candidats.",
            howToSpotEn = listOf(
                "Pick a house (row/col/box) and choose a missing digit.",
                "Check candidate positions for that digit; if only one cell can take it, place it."
            ),
            howToSpotFr = listOf(
                "Choisis une maison (ligne/colonne/boîte) et un chiffre manquant.",
                "Regarde où ce chiffre peut aller ; si une seule case possible, on le place."
            ),
            evidenceShapesEn = listOf(
                "House missing digit D.",
                "Only one cell in the house allows candidate D (others blocked by peers)."
            ),
            evidenceShapesFr = listOf(
                "La maison manque le chiffre D.",
                "Une seule case de la maison autorise le candidat D."
            ),
            commonConfusionsEn = listOf(
                "Thinking 'hidden single' means the cell has one candidate. It doesn't; the house has one spot for the digit."
            ),
            commonConfusionsFr = listOf(
                "Penser que 'single caché' signifie 'une case avec un seul candidat'. Non : c’est la maison qui n’a qu’un seul emplacement pour le chiffre."
            )
        ),
        Card(
            id = "singles-3",
            nameEn = "Claiming / Last Remaining in House",
            nameFr = "Dernière place dans une maison",
            difficulty = "easy",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "A house has only one empty cell left, so that cell must take the missing digit.",
            definitionFr = "Une maison n’a plus qu’une case vide : elle prend forcément le chiffre manquant.",
            howToSpotEn = listOf(
                "Find a row/col/box with 8 digits placed.",
                "The last empty cell is forced to be the missing digit."
            ),
            howToSpotFr = listOf(
                "Trouve une ligne/colonne/boîte avec 8 chiffres déjà placés.",
                "La dernière case vide est forcée : elle prend le chiffre manquant."
            ),
            evidenceShapesEn = listOf(
                "House has exactly one empty cell.",
                "Missing digit is determined uniquely by present digits."
            ),
            evidenceShapesFr = listOf(
                "La maison n’a qu’une case vide.",
                "Le chiffre manquant est unique."
            )
        ),

        // ---- Intersections ----
        Card(
            id = "pointing-1",
            nameEn = "Pointing Pair/Triple",
            nameFr = "Paire/Triplet pointant",
            difficulty = "medium",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "In a box, all candidates of a digit lie in one row or one column, so you can remove that digit from the rest of that row/column outside the box.",
            definitionFr = "Dans une boîte, tous les candidats d’un chiffre sont sur une seule ligne ou colonne : on peut éliminer ce chiffre du reste de cette ligne/colonne hors de la boîte.",
            howToSpotEn = listOf(
                "Pick a digit D inside a 3×3 box.",
                "If all D candidates in that box are confined to one row (or one column), eliminate D from that row/col outside the box."
            ),
            howToSpotFr = listOf(
                "Choisis un chiffre D dans une boîte 3×3.",
                "Si tous les candidats D de la boîte sont sur une seule ligne (ou colonne), élimine D du reste de cette ligne/colonne hors de la boîte."
            ),
            evidenceShapesEn = listOf(
                "Box B: candidate D appears only in row R (within the box).",
                "Therefore, in row R outside box B, D cannot appear."
            ),
            evidenceShapesFr = listOf(
                "Boîte B : le candidat D n’apparaît que sur la ligne R (dans la boîte).",
                "Donc sur la ligne R, hors de la boîte B, on enlève D."
            )
        ),
        Card(
            id = "boxed-1",
            nameEn = "Box/Line Reduction (Claiming)",
            nameFr = "Réduction boîte/ligne (claiming)",
            difficulty = "medium",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "In a row/column, all candidates of a digit lie inside one box, so you can remove that digit from the rest of the box.",
            definitionFr = "Dans une ligne/colonne, tous les candidats d’un chiffre sont dans une seule boîte : on peut éliminer ce chiffre du reste de la boîte.",
            howToSpotEn = listOf(
                "Pick a digit D in a row/column.",
                "If every D candidate in that row/col sits inside the same box, remove D from other cells of that box."
            ),
            howToSpotFr = listOf(
                "Choisis un chiffre D dans une ligne/colonne.",
                "Si tous les candidats D de cette ligne/colonne sont dans la même boîte, enlève D des autres cases de la boîte."
            ),
            evidenceShapesEn = listOf(
                "Row/col confines candidate D to cells inside one box.",
                "So other cells of that box cannot take D."
            ),
            evidenceShapesFr = listOf(
                "La ligne/colonne confine le candidat D à une seule boîte.",
                "Donc les autres cases de la boîte ne peuvent pas prendre D."
            )
        ),

        // ---- Subsets ----
        Card(
            id = "naked-2",
            nameEn = "Naked Pair",
            nameFr = "Paire nue",
            difficulty = "medium",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "Two cells in a house share the same two candidates; those candidates can be removed from other cells in the house.",
            definitionFr = "Deux cases d’une maison partagent exactement les mêmes deux candidats ; on retire ces candidats des autres cases de la maison.",
            howToSpotEn = listOf(
                "In a house, find two cells with identical candidate set {a,b}.",
                "Eliminate a and b from all other cells in that house."
            ),
            howToSpotFr = listOf(
                "Dans une maison, trouve deux cases avec le même ensemble {a,b}.",
                "Enlève a et b des autres cases de la maison."
            ),
            evidenceShapesEn = listOf(
                "House has exactly two cells containing candidates {a,b} and no others.",
                "a and b cannot appear elsewhere in the house."
            ),
            evidenceShapesFr = listOf(
                "La maison a exactement deux cases portant {a,b} (et rien d’autre).",
                "a et b ne peuvent pas être ailleurs dans la maison."
            )
        ),
        Card(
            id = "naked-3",
            nameEn = "Naked Triple",
            nameFr = "Triplet nu",
            difficulty = "hard",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "Three cells in a house contain only three digits among them; remove those digits from other cells in the house.",
            definitionFr = "Trois cases d’une maison contiennent seulement trois chiffres au total ; on retire ces chiffres des autres cases de la maison.",
            howToSpotEn = listOf(
                "Find 3 cells whose union of candidates has size 3.",
                "Eliminate those 3 digits from other cells in the house."
            ),
            evidenceShapesEn = listOf(
                "Union of candidates across the 3 cells is exactly {a,b,c}."
            )
        ),
        Card(
            id = "naked-4",
            nameEn = "Naked Quad",
            nameFr = "Quadruplet nu",
            difficulty = "expert",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "Four cells in a house contain only four digits among them; remove those digits from other cells in the house.",
            definitionFr = "Quatre cases d’une maison contiennent seulement quatre chiffres au total ; on retire ces chiffres des autres cases de la maison.",
            howToSpotEn = listOf(
                "Find 4 cells whose union of candidates has size 4.",
                "Eliminate those 4 digits from other cells in the house."
            ),
            evidenceShapesEn = listOf(
                "Union of candidates across the 4 cells is exactly {a,b,c,d}."
            )
        ),

        // ---- Fish / Wings / Chains ----
        Card(
            id = "xwing-1",
            nameEn = "X-Wing",
            nameFr = "X-Wing",
            difficulty = "hard",
            appliesTo = listOf("ROW", "COL"),
            definitionEn = "A digit forms a rectangle across two rows and two columns; eliminate that digit from other cells in those columns/rows.",
            definitionFr = "Un chiffre forme un rectangle sur deux lignes et deux colonnes ; on élimine ce chiffre des autres cases des colonnes/lignes concernées.",
            howToSpotEn = listOf(
                "Pick digit D. Find two rows where D appears in exactly the same two columns (or vice versa).",
                "Eliminate D from other cells in those two columns (if row-based)."
            ),
            evidenceShapesEn = listOf(
                "Two rows each have candidate D in exactly two matching columns.",
                "So other candidates D in those columns are eliminated."
            )
        ),
        Card(
            id = "ywing-1",
            nameEn = "Y-Wing",
            nameFr = "Y-Wing",
            difficulty = "hard",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "A pivot bi-value cell links two bi-value pincers; a shared digit can be eliminated from their intersection.",
            definitionFr = "Une case pivot bivaleur relie deux 'pincers' bivaleurs ; on peut éliminer un chiffre commun à l’intersection de leurs influences.",
            howToSpotEn = listOf(
                "Find a pivot with candidates (A,B).",
                "Find two pincers (A,C) and (B,C) each seeing the pivot.",
                "Eliminate C from cells that see both pincers."
            ),
            evidenceShapesEn = listOf(
                "Pivot (A,B) + pincers (A,C) and (B,C).",
                "C is forced in one pincer, eliminating C in their common peers."
            )
        ),
        Card(
            id = "remote-pairs-1",
            nameEn = "Remote Pairs",
            nameFr = "Paires à distance",
            difficulty = "expert",
            appliesTo = listOf("ROW", "COL", "BOX"),
            definitionEn = "A chain of bi-value cells with the same pair forces parity; eliminate digits from peers that see both parities.",
            definitionFr = "Une chaîne de cases bivaleurs avec la même paire force une parité ; on élimine des chiffres chez les pairs qui voient les deux parités.",
            howToSpotEn = listOf(
                "Find a chain of cells all with the same two candidates (A,B).",
                "Use alternating parity to eliminate A or B from cells seeing both parities."
            ),
            evidenceShapesEn = listOf(
                "Alternating chain on pair (A,B).",
                "Peer sees both colors → eliminate that digit."
            )
        )

        // You can keep extending as your engine grows; unknown techniques fall back safely.
    ).associateBy { it.id }

    fun hasCard(techniqueId: String?): Boolean =
        !techniqueId.isNullOrBlank() && cards.containsKey(techniqueId.trim())

    fun bundleFor(techniqueId: String?, preferredLang: String? = null): FactBundleV1? {
        val id = techniqueId?.trim().takeUnless { it.isNullOrBlank() } ?: return null
        val c = cards[id] ?: return null
        return FactBundleV1(
            type = FactBundleV1.Type.TEACHING_CARD,
            payload = c.toJson(preferredLang)
        )
    }

    /**
     * Safe fallback card if engine outputs an unknown technique id.
     */
    fun fallbackBundleFor(techniqueId: String?, preferredLang: String? = null): FactBundleV1? {
        val id = techniqueId?.trim().takeUnless { it.isNullOrBlank() } ?: return null
        val lang = (preferredLang ?: "en").trim().lowercase()
        val useFr = (lang == "fr")
        val payload = JSONObject().apply {
            put("schema", "teaching_card_v1")
            put("technique_id", id)
            put("technique_name", if (useFr) "Technique (non documentée)" else "Technique (undocumented)")
            put("difficulty", "unknown")
            put("applies_to", JSONArray().apply { put("ROW"); put("COL"); put("BOX") })
            put("definition", if (useFr) "Je n’ai pas encore une fiche pédagogique pour cette technique." else "I don’t have a teaching card for this technique yet.")
            put("how_to_spot", JSONArray().apply {
                put(if (useFr) "Demande-moi de décrire la technique, et je te dirai ce que je peux à partir des preuves du pas actuel." else "Ask me to describe the technique; I’ll use the current step’s evidence as far as possible.")
            })
            put("evidence_shapes", JSONArray().apply {
                put(if (useFr) "Preuves issues du pas de résolution (mais pas de fiche KB)." else "Evidence from the solve-step packet (but no KB card).")
            })
            put("common_confusions", JSONArray().apply { })
        }
        return FactBundleV1(type = FactBundleV1.Type.TEACHING_CARD, payload = payload)
    }
}