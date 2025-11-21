package com.contextionary.sudoku.logic

/**
 * Per-cell view of the board at a given moment.
 */
data class GridCellState(
    val index: Int,          // 0..80
    val row: Int,            // 0..8
    val col: Int,            // 0..8
    val digit: Int,          // 0..9 (0 = empty)
    val confidence: Float,   // 0.0 .. 1.0
    val isConflict: Boolean, // participates in any row/col/box conflict
    val wasChangedByLogic: Boolean, // in AutoCorrectionResult.changedIndices
    val isUnresolved: Boolean,      // in AutoCorrectionResult.unresolvedIndices or overlayUnresolved
    val isLowConfidence: Boolean    // driven by overlay / THR_HIGH from SudokuResultView
)

/**
 * Snapshot of the entire grid: digits, confidence, conflicts, solvability, etc.
 */
data class GridState(
    val cells: List<GridCellState>,
    val digits: IntArray,            // 81-length copy
    val confidences: FloatArray,     // 81-length copy
    val conflictIndices: List<Int>,
    val changedByLogic: List<Int>,
    val unresolvedIndices: List<Int>,
    val isStructurallyValid: Boolean,
    val hasUniqueSolution: Boolean,
    val hasMultipleSolutions: Boolean
)

/**
 * Description of a single user edit in the overlay.
 */
data class GridEditEvent(
    val seq: Int,            // 1, 2, 3, ...
    val cellIndex: Int,      // 0..80
    val row: Int,
    val col: Int,
    val oldDigit: Int,
    val newDigit: Int,
    val timestampMs: Long
)

/**
 * Thin coordination layer between:
 *  - auto-correction / overlay editing
 *  - the future LLM-based "Sudoku companion" brain.
 *
 * Right now it:
 *  - logs state transitions
 *  - synthesizes a candidate "companion message" string
 *
 * Later, you'll:
 *  - feed these messages into a chat / TTS UI
 *  - or replace them with real LLM calls while keeping this shape.
 */
class GridConversationCoordinator {

    /**
     * Called once after auto-correction finishes and the result grid is shown.
     */
    fun onAutoCorrectionCompleted(state: GridState) {
        val companionMessage = buildInitialCompanionMessage(state)

        // ---- Confidence stats for debugging ----
        val minCell = state.cells.minByOrNull { it.confidence }
        val maxCell = state.cells.maxByOrNull { it.confidence }
        val lowCount = state.cells.count { it.isLowConfidence }

        // Build a compact stats string like:
        // confMin=0.612(r3c8=8) confMax=0.997(r5c5=4) lowCells=3
        val stats = if (minCell != null && maxCell != null) {
            "confMin=%.3f(r%dc%d=%d) confMax=%.3f(r%dc%d=%d) lowCells=%d".format(
                java.util.Locale.US,
                minCell.confidence,
                minCell.row + 1,
                minCell.col + 1,
                minCell.digit,
                maxCell.confidence,
                maxCell.row + 1,
                maxCell.col + 1,
                maxCell.digit,
                lowCount
            )
        } else {
            "noCells"
        }

        android.util.Log.i(
            "GridConversation",
            "autoCorrectionCompleted: " +
                    "unique=${state.hasUniqueSolution} " +
                    "multi=${state.hasMultipleSolutions} " +
                    "structurallyValid=${state.isStructurallyValid} " +
                    "conflicts=${state.conflictIndices.size} " +
                    "unresolved=${state.unresolvedIndices.size} " +
                    "$stats " +
                    "message=\"$companionMessage\""
        )

        // Later:
        //  - push `companionMessage` into a UI strip above the grid
        //  - send it to TTS so Sudo "speaks"
        //  - or let an LLM generate this instead, keeping this call site intact.
    }

    /**
     * Called after each manual overlay edit (digit picker).
     */
    fun onManualEditApplied(state: GridState, edit: GridEditEvent) {
        val companionMessage = buildMessageAfterEdit(state, edit)

        android.util.Log.i(
            "GridConversation",
            "manualEdit: seq=${edit.seq} " +
                    "cell=r${edit.row + 1}c${edit.col + 1} " +
                    "from=${edit.oldDigit} to=${edit.newDigit} " +
                    "unique=${state.hasUniqueSolution} " +
                    "multi=${state.hasMultipleSolutions} " +
                    "conflicts=${state.conflictIndices.size} " +
                    "unresolved=${state.unresolvedIndices.size} " +
                    "message=\"$companionMessage\""
        )

        // Later:
        //  - use `companionMessage` to update the "Sudo says" area and TTS
        //  - potentially adapt what Sudo says if the grid just became unique,
        //    or if conflicts disappeared after this edit.
    }

    /**
     * Build the first thing Sudo would say after auto-correction completes.
     *
     * This now:
     *  - sends the user to a RETAKE path if there are "too many" unresolved/conflict cells,
     *  - distinguishes between:
     *      * all high-confidence, no auto-changes  → strong "I copied correctly"
     *      * auto-changes and/or low-confidence    → cautious confirmation
     */
    fun buildInitialCompanionMessage(state: GridState): String {
        val unresolvedCount = state.unresolvedIndices.size
        val conflictCount = state.conflictIndices.size
        val hasLowConf = hasLowConfidenceCells(state)
        val changedCount = state.changedByLogic.size
        val hasAutoChanges = changedCount > 0

        // 1) Too messy: many conflicts / unresolved cells → suggest retake
        if (unresolvedCount > MAX_UNRESOLVED_FOR_CONVERSATION ||
            conflictCount > MAX_UNRESOLVED_FOR_CONVERSATION
        ) {
            return "This photo is quite hard to read and there are many doubtful cells. It might be easier to retake the picture with better lighting and framing."
        }

        // 2) Structurally invalid but not "hopelessly messy"
        if (!state.isStructurallyValid) {
            return "I see some conflicts in your grid. Let's double-check a few cells together."
        }

        // 3) Clean, unique solution, no conflicts, no unresolved cells.
        //    We now distinguish:
        //      - no low-conf, no auto-changes → strong "copied correctly"
        //      - auto-changes and/or low-conf → confirmation messages.
        if (conflictCount == 0 &&
            unresolvedCount == 0 &&
            state.hasUniqueSolution
        ) {
            return when {
                // All digits high confidence, no auto-changes at all
                !hasLowConf && !hasAutoChanges -> {
                    "Great! I copied your puzzle correctly and it has a unique solution. Ready to play?"
                }

                // Auto-changes happened, but no remaining low-confidence digits
                hasAutoChanges && !hasLowConf -> {
                    "I adjusted a few doubtful cells so your puzzle now has a unique solution. Please double-check the highlighted cells to confirm they match your book."
                }

                // Low-confidence digits, but no auto-changes
                !hasAutoChanges && hasLowConf -> {
                    "Good news: the puzzle I see has a unique solution. A few digits were hard to read, so before we start, please confirm that what you see on screen matches your book."
                }

                // Both auto-changes and low-confidence digits
                else -> {
                    "I adjusted some hard-to-read cells so the puzzle has a unique solution. Before we start, please carefully compare the highlighted cells with your book."
                }
            }
        }

        // 4) A small number of unresolved cells → conversational fixing is reasonable.
        if (unresolvedCount in 1..MAX_UNRESOLVED_FOR_CONVERSATION) {
            return "Your grid is almost ready. We just need to confirm ${unresolvedCount} cell(s)."
        }

        // 5) Multiple solutions but not too messy otherwise.
        if (state.hasMultipleSolutions) {
            return "This grid has more than one solution. Let's check that the digits match your book exactly."
        }

        // 6) Fallback generic message.
        return "I’ve analyzed your grid. Let's take a closer look at a few cells."
    }

    /**
     * Build what Sudo would say right after a single manual edit.
     *
     * Also respects the "too many unresolved" threshold: if, after edits, we still
     * have a very large number of doubtful cells, we gently propose a retake.
     */
    fun buildMessageAfterEdit(state: GridState, edit: GridEditEvent): String {
        val unresolvedCount = state.unresolvedIndices.size
        val conflictCount = state.conflictIndices.size

        // If after edits the grid is still very messy, suggest retake.
        if (unresolvedCount > MAX_UNRESOLVED_FOR_CONVERSATION ||
            conflictCount > MAX_UNRESOLVED_FOR_CONVERSATION
        ) {
            return "Thanks for the update. There are still quite a lot of uncertain cells, so it may be easier to retake the picture rather than fixing everything by hand."
        }

        return when {
            !state.isStructurallyValid ->
                "After that change, the grid still has some conflicts. Let's double-check row ${edit.row + 1}, column ${edit.col + 1}."

            state.conflictIndices.isEmpty() &&
                    state.unresolvedIndices.isEmpty() &&
                    state.hasUniqueSolution ->
                "Nice! Your latest change made the puzzle a clean, uniquely solvable grid."

            state.unresolvedIndices.isNotEmpty() ->
                "Good update. We still have ${state.unresolvedIndices.size} cell(s) to confirm."

            else ->
                "Change noted in row ${edit.row + 1}, column ${edit.col + 1}. We’re getting closer."
        }
    }

    // -----------------------
    // Internal helpers / tuning
    // -----------------------

    /**
     * Decide whether the grid has any "low-confidence" cells.
     *
     * We trust the overlay / SudokuResultView to have already decided
     * what counts as "low confidence" based on THR_HIGH.
     */
    private fun hasLowConfidenceCells(state: GridState): Boolean {
        return state.cells.any { it.isLowConfidence }
    }

    companion object {
        /**
         * Above this number of unresolved / conflict cells, we stop trying to have
         * a detailed "let's fix them one by one" conversation and instead suggest
         * retaking the photo.
         *
         * Tune this as you gain more experience; starting at 6 as you suggested.
         */
        private const val MAX_UNRESOLVED_FOR_CONVERSATION: Int = 6
    }
}