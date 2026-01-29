package com.contextionary.sudoku.conductor

/**
 * Clarification type requested by the policy.
 *
 * Note: the policy JSON may use aliases ("COLUMN" vs "COL").
 * Use [fromWire] in your JSON->ToolCall mapper.
 */
enum class ClarifyKind {
    ROW,
    COL,
    DIGIT,
    POSITION;

    companion object {
        /**
         * Accepts common aliases coming from LLM/tool JSON.
         * Returns null if unknown.
         */
        fun fromWire(raw: String?): ClarifyKind? {
            val s = raw
                ?.trim()
                ?.uppercase()
                ?.replace("-", "_")
                ?.replace(" ", "_")
                ?: return null

            return when (s) {
                "ROW", "R" -> ROW
                "COL", "COLUMN", "C" -> COL
                "DIGIT", "NUMBER", "VALUE" -> DIGIT
                "POSITION", "POS", "CELL", "CELL_POSITION" -> POSITION
                else -> null
            }
        }

        /**
         * Canonical wire name (if you want to serialize back to JSON).
         */
        fun toWire(kind: ClarifyKind): String = when (kind) {
            ROW -> "ROW"
            COL -> "COL"
            DIGIT -> "DIGIT"
            POSITION -> "POSITION"
        }
    }
}

enum class CellClass {
    GIVEN,
    SOLUTION,
    EMPTY;

    companion object {
        fun fromWire(raw: String?): CellClass? {
            val s = raw?.trim()?.uppercase()?.replace("-", "_") ?: return null
            return when (s) {
                "GIVEN", "GIVENS" -> GIVEN
                "SOLUTION", "SOLUTIONS", "USER_SOLUTION" -> SOLUTION
                "EMPTY", "BLANK" -> EMPTY
                else -> null
            }
        }

        fun toWire(v: CellClass): String = when (v) {
            GIVEN -> "GIVEN"
            SOLUTION -> "SOLUTION"
            EMPTY -> "EMPTY"
        }
    }
}

/**
 * Policy tool calls emitted by the LLM (grid mode).
 *
 * Keep this file aligned with:
 * - CompanionConversation.kt ToolNames (string tool names)
 * - JSON->ToolCall decoder/mapper
 * - SudoConductor.applyPolicyTools(...)
 */
sealed class ToolCall {

    // ----------------------------
    // Core tools
    // ----------------------------

    data class Reply(val text: String) : ToolCall()

    /**
     * LEGACY / BACKWARD COMPAT ONLY.
     *
     * ⚠ In GRID mode this tool should not be used because it is the
     * main source of "spoken cell ≠ pending cell" mismatch.
     *
     * Prefer [AskConfirmCellRC] in GRID_SESSION.
     */
    data class AskConfirmCell(
        val cellIndex: Int,
        val prompt: String
    ) : ToolCall()

    /**
     * ✅ Patch 1:
     * Row/Col variant to eliminate cellIndex mismatch in confirmations.
     *
     * row/col are 1..9.
     */
    data class AskConfirmCellRC(
        val row: Int,    // 1..9
        val col: Int,    // 1..9
        val prompt: String
    ) : ToolCall()

    data class ProposeEdit(
        val cellIndex: Int,
        val digit: Int,
        val reason: String,
        val confidence: Float
    ) : ToolCall()

    /**
     * A user-requested edit that should be applied immediately.
     * (cellIndex is 0..80, digit 0..9, where 0 = blank)
     */
    data class ApplyUserEdit(
        val cellIndex: Int,
        val digit: Int,
        val source: String = "user_request"
    ) : ToolCall()

    /**
     * ✅ Row/Col variant to reduce cellIndex mapping mistakes.
     * (row/col are 1..9, digit 0..9, where 0 = blank)
     *
     * Your adapter/mapper should convert this to cellIndex=(row-1)*9+(col-1)
     * or map directly in your conductor.
     */
    data class ApplyUserEditRC(
        val row: Int,   // 1..9
        val col: Int,   // 1..9
        val digit: Int, // 0..9
        val source: String = "user_request"
    ) : ToolCall()



    /**
     * ✅ NEW (Option B): explicit confirmation (NOT an edit).
     *
     * Index-based variant (0..80). Used when the policy already knows the concrete cellIndex
     * (e.g., from pending AskCellValue).
     *
     * digit allows 0..9 (0 = blank).
     *
     * NOTE: schema currently provides only (cell_index, digit). `source` is defaulted locally.
     */
    data class ConfirmCellValue(
        val cellIndex: Int,      // 0..80
        val digit: Int,          // 0..9 (0=blank)
        val source: String = "user_confirm"
    ) : ToolCall()

    /**
     * ✅ NEW (Option B): explicit confirmation (NOT an edit).
     *
     * Row/Col variant (preferred in GRID_SESSION).
     *
     * digit allows 0..9 (0 = blank).
     *
     * NOTE: schema currently provides only (row, col, digit). `source` is defaulted locally.
     */
    data class ConfirmCellValueRC(
        val row: Int,        // 1..9
        val col: Int,        // 1..9
        val digit: Int,      // 0..9 (0=blank)
        val source: String = "user_confirm"
    ) : ToolCall()




    // ----------------------------
    // Truth/provenance tools (givens vs user solutions)
    // ----------------------------

    /**
     * Reclassify a cell's provenance group.
     * - kind="given"     => DNA / facts
     * - kind="solution"  => user's placed digit (opinion)
     * - kind="neither"   => neither group (digit may still exist; provenance cleared)
     */
    data class ReclassifyCell(
        val cellIndex: Int,           // 0..80
        val kind: String,             // "given" | "solution" | "neither"
        val source: String = "user_request"
    ) : ToolCall()

    /**
     * Batch reclassification (preferred when user reclassifies multiple cells).
     */
    data class ReclassifyCells(
        val cells: List<ReclassifyCell>,
        val source: String = "user_request"
    ) : ToolCall()

    /**
     * Row/Col variant (optional but useful).
     */
    data class ReclassifyCellRC(
        val row: Int,                 // 1..9
        val col: Int,                 // 1..9
        val kind: String,             // "given" | "solution" | "neither"
        val source: String = "user_request"
    ) : ToolCall()

    data class ApplyUserClassify(
        val cellIndex: Int,          // 0..80
        val cellClass: CellClass,    // GIVEN|SOLUTION|EMPTY
        val source: String = "user_reclass"
    ) : ToolCall()

    data class ApplyUserClassifyRC(
        val row: Int,                // 1..9
        val col: Int,                // 1..9
        val cellClass: CellClass,
        val source: String = "user_reclass"
    ) : ToolCall()

    // ----------------------------
    // Candidate tools (thought process layer)
    // ----------------------------

    /**
     * Set the full candidate bitmask for a cell.
     * mask uses bit (d-1) for digit d (1..9). mask=0 clears all candidates.
     */
    data class SetCandidates(
        val cellIndex: Int,           // 0..80
        val mask: Int,                // 0..(2^9-1)
        val source: String = "user_request"
    ) : ToolCall()

    data class ClearCandidates(
        val cellIndex: Int,           // 0..80
        val source: String = "user_request"
    ) : ToolCall()

    /**
     * Toggle a single candidate digit.
     */
    data class ToggleCandidate(
        val cellIndex: Int,           // 0..80
        val digit: Int,               // 1..9
        val source: String = "user_request"
    ) : ToolCall()

    data class RecommendRetake(
        val strength: String, // "soft" | "strong"
        val reason: String
    ) : ToolCall()

    object RecommendValidate : ToolCall()

    // ----------------------------
    // Gate 4: conversational repair moves (NOT edits)
    // ----------------------------

    data class ConfirmInterpretation(
        val row: Int? = null,     // 1..9
        val col: Int? = null,     // 1..9
        val digit: Int? = null,   // 0..9
        val prompt: String,
        val confidence: Float = 0.6f
    ) : ToolCall()

    data class AskClarifyingQuestion(
        val kind: ClarifyKind,
        val prompt: String
    ) : ToolCall()

    data class SwitchToTap(
        val prompt: String
    ) : ToolCall()

    object Noop : ToolCall()

    /**
     * Centralize tool-name strings to keep JSON decoding consistent.
     */
    companion object WireNames {
        const val REPLY = "reply"

        // confirmations
        const val ASK_CONFIRM_CELL = "ask_confirm_cell"           // legacy
        const val ASK_CONFIRM_CELL_RC = "ask_confirm_cell_rc"     // ✅ Patch 1

        // ✅ NEW: explicit confirmation tools (NOT an edit)
        const val CONFIRM_CELL_VALUE = "confirm_cell_value"
        const val CONFIRM_CELL_VALUE_RC = "confirm_cell_value_rc"

        const val PROPOSE_EDIT = "propose_edit"

        const val APPLY_USER_EDIT = "apply_user_edit"
        const val APPLY_USER_EDIT_RC = "apply_user_edit_rc"

        const val RECOMMEND_RETAKE = "recommend_retake"
        const val RECOMMEND_VALIDATE = "recommend_validate"

        // Gate 4
        const val CONFIRM_INTERPRETATION = "confirm_interpretation"
        const val ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
        const val SWITCH_TO_TAP = "switch_to_tap"

        const val NOOP = "noop"

        const val RECLASSIFY_CELL = "reclassify_cell"
        const val RECLASSIFY_CELLS = "reclassify_cells"
        const val RECLASSIFY_CELL_RC = "reclassify_cell_rc"

        const val SET_CANDIDATES = "set_candidates"
        const val CLEAR_CANDIDATES = "clear_candidates"
        const val TOGGLE_CANDIDATE = "toggle_candidate"

        const val APPLY_USER_CLASSIFY = "apply_user_classify"
        const val APPLY_USER_CLASSIFY_RC = "apply_user_classify_rc"
    }
}