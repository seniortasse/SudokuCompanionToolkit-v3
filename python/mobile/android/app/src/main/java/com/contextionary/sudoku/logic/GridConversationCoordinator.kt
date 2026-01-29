package com.contextionary.sudoku.logic

/**
 * Per-cell view of the board at a given moment.
 *
 * NOTE:
 * - New provenance/history fields are OPTIONAL (defaulted) to preserve back-compat.
 * - MainActivity can populate them when available so the LLM sees full history.
 */
data class GridCellState(
    val index: Int,          // 0..80
    val row: Int,            // 0..8
    val col: Int,            // 0..8

    // Current (what user sees NOW)
    val digit: Int,          // 0..9 (0 = empty)
    val confidence: Float,   // 0.0 .. 1.0

    // Validation flags (current)
    val isConflict: Boolean, // participates in any row/col/box conflict
    val wasChangedByLogic: Boolean, // in AutoCorrectionResult.changedIndices
    val isUnresolved: Boolean,      // in AutoCorrectionResult.unresolvedIndices or overlayUnresolved
    val isLowConfidence: Boolean,   // driven by overlay / THR_HIGH

    // ✅ NEW: manual correction flag (distinct from auto)
    val wasManuallyCorrected: Boolean = false,

    // ✅ NEW: scanned baseline (pre-autocorrect, from decidePick / lastGridPrediction)
    val scannedDigit: Int? = null,
    val scannedConfidence: Float? = null,

    // ✅ NEW: autocorrect provenance
    // (autoFromDigit is typically scannedDigit; autoToDigit is digit after autocorrect stage)
    val autoFromDigit: Int? = null,
    val autoToDigit: Int? = null,

    // ✅ NEW: last manual edit provenance (most recent manual correction)
    val manualFromDigit: Int? = null,
    val manualToDigit: Int? = null
)

/**
 * Snapshot of the entire grid: digits, confidence, conflicts, solvability, etc.
 *
 * NOTE:
 * - NEW: manuallyCorrected indices
 * - Back-compat constructors preserved for older call sites.
 */
data class GridState(
    val cells: List<GridCellState>,
    val digits: IntArray,            // 81-length copy
    val confidences: FloatArray,     // 81-length copy
    val conflictIndices: List<Int>,
    val changedByLogic: List<Int>,   // auto-correct changed indices
    val unresolvedIndices: List<Int>,
    val isStructurallyValid: Boolean,
    val hasUniqueSolution: Boolean,
    val hasMultipleSolutions: Boolean,

    // ✅ Existing fields (kept)
    val hasNoSolution: Boolean = false,
    val solutionCountCapped: Int = 0,

    // ✅ NEW: manual-corrected indices (defaulted for back-compat)
    val manuallyCorrected: List<Int> = emptyList()
) {
    /**
     * ✅ Back-compat: allow call sites that still pass IntArray for changedByLogic.
     */
    constructor(
        cells: List<GridCellState>,
        digits: IntArray,
        confidences: FloatArray,
        conflictIndices: List<Int>,
        changedByLogic: IntArray,
        unresolvedIndices: List<Int>,
        isStructurallyValid: Boolean,
        hasUniqueSolution: Boolean,
        hasMultipleSolutions: Boolean,
        hasNoSolution: Boolean = false,
        solutionCountCapped: Int = 0,
        manuallyCorrected: List<Int> = emptyList()
    ) : this(
        cells = cells,
        digits = digits,
        confidences = confidences,
        conflictIndices = conflictIndices,
        changedByLogic = changedByLogic.toList(),
        unresolvedIndices = unresolvedIndices,
        isStructurallyValid = isStructurallyValid,
        hasUniqueSolution = hasUniqueSolution,
        hasMultipleSolutions = hasMultipleSolutions,
        hasNoSolution = hasNoSolution,
        solutionCountCapped = solutionCountCapped,
        manuallyCorrected = manuallyCorrected
    )

    /**
     * ✅ Optional back-compat convenience: allow IntArray manuallyCorrected too.
     * (Use this only if you have call sites that naturally keep these as IntArray.)
     */
    constructor(
        cells: List<GridCellState>,
        digits: IntArray,
        confidences: FloatArray,
        conflictIndices: List<Int>,
        changedByLogic: IntArray,
        unresolvedIndices: List<Int>,
        isStructurallyValid: Boolean,
        hasUniqueSolution: Boolean,
        hasMultipleSolutions: Boolean,
        hasNoSolution: Boolean = false,
        solutionCountCapped: Int = 0,
        manuallyCorrected: IntArray
    ) : this(
        cells = cells,
        digits = digits,
        confidences = confidences,
        conflictIndices = conflictIndices,
        changedByLogic = changedByLogic.toList(),
        unresolvedIndices = unresolvedIndices,
        isStructurallyValid = isStructurallyValid,
        hasUniqueSolution = hasUniqueSolution,
        hasMultipleSolutions = hasMultipleSolutions,
        hasNoSolution = hasNoSolution,
        solutionCountCapped = solutionCountCapped,
        manuallyCorrected = manuallyCorrected.toList()
    )
}

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
 */
class GridConversationCoordinator {

    fun onAutoCorrectionCompleted(state: GridState) {
        val companionMessage = buildInitialCompanionMessage(state)

        val minCell = state.cells.minByOrNull { it.confidence }
        val maxCell = state.cells.maxByOrNull { it.confidence }
        val lowCount = state.cells.count { it.isLowConfidence }

        val autoCount = state.changedByLogic.size
        val manualCount = state.manuallyCorrected.size

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
                    "noSol=${state.hasNoSolution} " +
                    "solCountCapped=${state.solutionCountCapped} " +
                    "structurallyValid=${state.isStructurallyValid} " +
                    "conflicts=${state.conflictIndices.size} " +
                    "unresolved=${state.unresolvedIndices.size} " +
                    "autoChanged=$autoCount " +
                    "manualChanged=$manualCount " +
                    "$stats " +
                    "message=\"$companionMessage\""
        )
    }

    fun onManualEditApplied(state: GridState, edit: GridEditEvent) {
        val companionMessage = buildMessageAfterEdit(state, edit)

        val autoCount = state.changedByLogic.size
        val manualCount = state.manuallyCorrected.size

        android.util.Log.i(
            "GridConversation",
            "manualEdit: seq=${edit.seq} " +
                    "cell=r${edit.row + 1}c${edit.col + 1} " +
                    "from=${edit.oldDigit} to=${edit.newDigit} " +
                    "unique=${state.hasUniqueSolution} " +
                    "multi=${state.hasMultipleSolutions} " +
                    "noSol=${state.hasNoSolution} " +
                    "solCountCapped=${state.solutionCountCapped} " +
                    "conflicts=${state.conflictIndices.size} " +
                    "unresolved=${state.unresolvedIndices.size} " +
                    "autoChanged=$autoCount " +
                    "manualChanged=$manualCount " +
                    "message=\"$companionMessage\""
        )
    }

    fun buildInitialCompanionMessage(state: GridState): String {
        val unresolvedCount = state.unresolvedIndices.size
        val conflictCount = state.conflictIndices.size
        val hasLowConf = hasLowConfidenceCells(state)
        val changedCount = state.changedByLogic.size
        val manualCount = state.manuallyCorrected.size

        val hasAutoChanges = changedCount > 0
        val hasManualChanges = manualCount > 0

        if (state.hasNoSolution) {
            return "Something seems inconsistent: this grid doesn’t appear to have a valid solution. Let’s double-check the digits or consider retaking the photo."
        }

        if (unresolvedCount > MAX_UNRESOLVED_FOR_CONVERSATION ||
            conflictCount > MAX_UNRESOLVED_FOR_CONVERSATION
        ) {
            return "This photo is quite hard to read and there are many doubtful cells. It might be easier to retake the picture with better lighting and framing."
        }

        if (!state.isStructurallyValid) {
            return "I see some conflicts in your grid. Let's double-check a few cells together."
        }

        if (conflictCount == 0 &&
            unresolvedCount == 0 &&
            state.hasUniqueSolution
        ) {
            return when {
                !hasLowConf && !hasAutoChanges && !hasManualChanges ->
                    "Great! I copied your puzzle correctly and it has a unique solution. Ready to play?"

                (hasAutoChanges || hasManualChanges) && !hasLowConf ->
                    "Your grid looks consistent and uniquely solvable. Please double-check the highlighted corrected cells to confirm they match your book."

                !hasAutoChanges && hasLowConf ->
                    "Good news: the puzzle I see has a unique solution. A few digits were hard to read, so before we start, please confirm that what you see on screen matches your book."

                else ->
                    "I adjusted some hard-to-read cells so the puzzle has a unique solution. Before we start, please carefully compare the corrected cells with your book."
            }
        }

        if (unresolvedCount in 1..MAX_UNRESOLVED_FOR_CONVERSATION) {
            return "Your grid is almost ready. We just need to confirm ${unresolvedCount} cell(s)."
        }

        if (state.hasMultipleSolutions) {
            return "This grid has more than one solution. Let's check that the digits match your book exactly."
        }

        return "I’ve analyzed your grid. Let's take a closer look at a few cells."
    }

    fun buildMessageAfterEdit(state: GridState, edit: GridEditEvent): String {
        val unresolvedCount = state.unresolvedIndices.size
        val conflictCount = state.conflictIndices.size

        if (state.hasNoSolution) {
            return "Thanks — but the grid still looks inconsistent (no valid solution). Let’s verify the scanned digits or retake the photo."
        }

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

    private fun hasLowConfidenceCells(state: GridState): Boolean {
        return state.cells.any { it.isLowConfidence }
    }

    companion object {
        private const val MAX_UNRESOLVED_FOR_CONVERSATION: Int = 6
    }
}