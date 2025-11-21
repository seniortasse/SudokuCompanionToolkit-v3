package com.contextionary.sudoku.logic

/**
 * Conversation-related models for the Sudoku Companion LLM flow.
 * This file contains NO networking. Your SudokuLLMClient implementation
 * performs the actual HTTP/LLM calls.
 */

// -------------------------
// Severity of the grid state (for the LLM)
// -------------------------
enum class GridSeverity {
    OK,
    MILD,
    SERIOUS,
    RETAKE_NEEDED
}

// -------------------------
// LLM-facing GridState
// -------------------------
data class LLMGridState(
    val correctedGrid: IntArray,        // 81 digits, row-major
    val unresolvedCells: List<Int>,     // 0..80
    val changedCells: List<Int>,        // indices auto-corrected
    val conflictCells: List<Int>,       // indices involved in Sudoku conflicts
    val lowConfidenceCells: List<Int>,  // NEW – 0..80
    val uniqueSolvable: Boolean,
    val unresolvedCount: Int,
    val severity: String                 // "ok" | "mild" | "serious" | "retake_needed"
)

// Optional human-friendly summary
data class GridHumanSummary(val message: String)


// -------------------------
// Raw LLM response structures
// -------------------------
data class LLMRawAction(
    val type: String,
    val cell: String? = null,
    val digit: Int? = null,
    val options: List<Int>? = null
)

data class LLMRawResponse(
    val assistant_message: String,
    val action: LLMRawAction
)


// -------------------------
// Domain action types
// -------------------------
sealed class LLMAction {
    data class ChangeCell(val row: Int, val col: Int, val digit: Int) : LLMAction()
    data class AskUserConfirmation(val row: Int, val col: Int, val options: List<Int>) : LLMAction()
    object RetakePhoto : LLMAction()
    object NoAction : LLMAction()
    object ValidateGrid : LLMAction()
    data class Unknown(val raw: LLMRawAction) : LLMAction()
}


// -------------------------
// Cell label parser ("r4c7")
// -------------------------
private fun parseCellLabel(label: String?): Pair<Int, Int>? {
    if (label == null) return null
    val regex = Regex("""r(\d+)c(\d+)""")
    val m = regex.matchEntire(label.trim()) ?: return null
    val row = m.groupValues[1].toIntOrNull() ?: return null
    val col = m.groupValues[2].toIntOrNull() ?: return null
    if (row !in 1..9 || col !in 1..9) return null
    return (row - 1) to (col - 1)
}

fun LLMRawAction.toDomain(): LLMAction {
    return when (type) {
        "change_cell" -> {
            val (r, c) = parseCellLabel(cell) ?: return LLMAction.Unknown(this)
            val d = digit ?: return LLMAction.Unknown(this)
            LLMAction.ChangeCell(r, c, d)
        }
        "ask_user_confirmation" -> {
            val (r, c) = parseCellLabel(cell) ?: return LLMAction.Unknown(this)
            val opts = options ?: emptyList()
            if (opts.isEmpty()) LLMAction.Unknown(this)
            else LLMAction.AskUserConfirmation(r, c, opts)
        }
        "retake_photo" -> LLMAction.RetakePhoto
        "no_action" -> LLMAction.NoAction
        "validate_grid" -> LLMAction.ValidateGrid
        else -> LLMAction.Unknown(this)
    }
}


// -------------------------
// LLM Client Interface (implemented by RealSudokuLLMClient)
// -------------------------
interface SudokuLLMClient {
    suspend fun sendGridUpdate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): LLMRawResponse
}


// -------------------------
// Conversation Coordinator
// -------------------------
class SudokuLLMConversationCoordinator(
    private val solver: SudokuSolver,
    private val llmClient: SudokuLLMClient
) {

    companion object {
        private const val MAX_UNRESOLVED_FOR_CONV = 6
    }

    /**
     * Severity logic strictly aligned with auto-corrector.
     */
    fun computeSeverity(
        uniqueSolvable: Boolean,
        unresolvedCount: Int,
        conflictCount: Int,
        changedCount: Int,
        lowConfCount: Int
    ): GridSeverity {

        return when {
            unresolvedCount > MAX_UNRESOLVED_FOR_CONV ||
                    conflictCount > MAX_UNRESOLVED_FOR_CONV ->
                GridSeverity.RETAKE_NEEDED

            conflictCount == 0 &&
                    unresolvedCount == 0 &&
                    uniqueSolvable &&
                    changedCount == 0 &&
                    lowConfCount == 0 ->
                GridSeverity.OK

            conflictCount == 0 &&
                    unresolvedCount == 0 &&
                    uniqueSolvable &&
                    (changedCount > 0 || lowConfCount > 0) ->
                GridSeverity.MILD

            unresolvedCount in 1..3 && conflictCount == 0 ->
                GridSeverity.MILD

            unresolvedCount in 4..MAX_UNRESOLVED_FOR_CONV ||
                    conflictCount in 1..MAX_UNRESOLVED_FOR_CONV ->
                GridSeverity.SERIOUS

            else -> GridSeverity.SERIOUS
        }
    }

    /**
     * Build consistent LLMGridState
     */
    fun buildLLMGridState(
        auto: AutoCorrectionResult,
        conflicts: List<Int>,
        lowConfidenceCells: List<Int>
    ): LLMGridState {

        val digits = auto.correctedGrid.digits
        val unresolved = auto.unresolvedIndices
        val changed = auto.changedIndices
        val unique = auto.wasSolvable
        val lowCount = lowConfidenceCells.size

        val sev = computeSeverity(
            uniqueSolvable = unique,
            unresolvedCount = unresolved.size,
            conflictCount = conflicts.size,
            changedCount = changed.size,
            lowConfCount = lowCount
        )

        val sevString = when (sev) {
            GridSeverity.OK -> "ok"
            GridSeverity.MILD -> "mild"
            GridSeverity.SERIOUS -> "serious"
            GridSeverity.RETAKE_NEEDED -> "retake_needed"
        }

        return LLMGridState(
            correctedGrid = digits,
            unresolvedCells = unresolved,
            changedCells = changed,
            conflictCells = conflicts,
            lowConfidenceCells = lowConfidenceCells,
            uniqueSolvable = unique,
            unresolvedCount = unresolved.size,
            severity = sevString
        )
    }

    /**
     * Produce a short human summary
     */
    fun buildHumanSummary(g: LLMGridState): GridHumanSummary {
        val sb = StringBuilder()

        if (g.uniqueSolvable) sb.append("Grid currently has a unique solution. ")
        else sb.append("Grid does not have a unique solution. ")

        if (g.unresolvedCount == 0) sb.append("There are no unresolved cells. ")
        else sb.append("There are ${g.unresolvedCount} unresolved cells. ")

        if (g.conflictCells.isNotEmpty())
            sb.append("${g.conflictCells.size} cells have conflicts. ")

        if (g.changedCells.isNotEmpty())
            sb.append("${g.changedCells.size} cells were auto-corrected. ")

        if (g.lowConfidenceCells.isNotEmpty())
            sb.append("${g.lowConfidenceCells.size} cells had low confidence. ")

        sb.append("Severity is ${g.severity}.")

        return GridHumanSummary(sb.toString())
    }

    /**
     * Build developer prompt for LLM.
     */
    fun buildDeveloperPrompt(
        g: LLMGridState,
        summary: GridHumanSummary,
        userMessage: String
    ): String {

        val safeMsg = if (userMessage.isBlank())
            "(no user message; system-initiated)"
        else userMessage

        return """
            You are Sudo, the Sudoku Companion.

            GRID SNAPSHOT:
            - uniqueSolvable: ${g.uniqueSolvable}
            - severity: ${g.severity}
            - unresolvedCount: ${g.unresolvedCells.size}
            - changedCount: ${g.changedCells.size}
            - conflictCount: ${g.conflictCells.size}
            - lowConfidenceCount: ${g.lowConfidenceCells.size}

            Indices (0-based):
            - unresolved: ${g.unresolvedCells}
            - changed: ${g.changedCells}
            - conflicts: ${g.conflictCells}
            - lowConfidence: ${g.lowConfidenceCells}

            HUMAN SUMMARY:
            "${summary.message}"

            USER MESSAGE:
            "$safeMsg"

            Use the above to speak warmly and choose exactly ONE action in JSON.
        """.trimIndent()
    }

    /**
     * Send to LLM
     */

    /**
     * Send to LLM with post-parse guardrails that enforce product truth.
     */
    suspend fun sendToLLM(
        systemPrompt: String,
        gridState: LLMGridState,
        userMessage: String
    ): Pair<String, LLMAction> {

        val summary = buildHumanSummary(gridState)
        val devPrompt = buildDeveloperPrompt(gridState, summary, userMessage)

        val raw = llmClient.sendGridUpdate(
            systemPrompt = systemPrompt,
            developerPrompt = devPrompt,
            userMessage = userMessage
        )

        // Parse to domain first
        val domain = raw.action.toDomain()
        val sev = gridState.severity
        val changedCount = gridState.changedCells.size
        val lowConfCount = gridState.lowConfidenceCells.size
        val unresolvedCount = gridState.unresolvedCells.size
        val conflictCount = gridState.conflictCells.size

        // Helper: pick a “most relevant” cell index to confirm
        fun pickIndex(): Int? =
            (gridState.changedCells.firstOrNull()
                ?: gridState.lowConfidenceCells.firstOrNull()
                ?: gridState.unresolvedCells.firstOrNull())

        // Guard 1: retake_needed must force RetakePhoto
        if (sev == "retake_needed") {
            val msg = "This photo is quite hard to read. Let’s retake it with good lighting and the page flat."
            return msg to LLMAction.RetakePhoto
        }

        // Guard 2: serious must NOT validate grid
        if (sev == "serious" && domain is LLMAction.ValidateGrid) {
            val idx = pickIndex()
            return if (idx != null) {
                val row = idx / 9
                val col = idx % 9
                val msg = "I see several doubtful squares. Let’s double-check one before we start."
                msg to LLMAction.AskUserConfirmation(row, col, (1..9).toList())
            } else {
                val msg = "Some parts of the grid look uncertain. Would you mind checking a couple of squares?"
                msg to LLMAction.NoAction
            }
        }

        // Guard 3: if mild OR there were auto changes OR low-confidence cells,
        // we must ask for confirmation (not validate grid)
        if (sev == "mild" || changedCount > 0 || lowConfCount > 0) {
            val alreadyConfirm = domain is LLMAction.AskUserConfirmation
            if (!alreadyConfirm) {
                val idx = pickIndex()
                if (idx != null) {
                    val row = idx / 9
                    val col = idx % 9
                    val msg = "I adjusted a couple of squares I wasn’t fully sure about. Could you confirm this one?"
                    return msg to LLMAction.AskUserConfirmation(row, col, (1..9).toList())
                } else {
                    val msg = "Everything looks close, but I’d like a quick double-check on a square before we begin."
                    return msg to LLMAction.NoAction
                }
            }
        }

        // Guard 4: ok severity is allowed to validate_grid; otherwise keep model’s choice.
        return raw.assistant_message to domain
    }
}