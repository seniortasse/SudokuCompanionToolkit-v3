package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.logic.LLMGridState

enum class SudoMode { GRID_SESSION, FREE_TALK }

sealed class Pending {

    // Existing
    data class ConfirmEdit(
        val cellIndex: Int,          // 0..80
        val proposedDigit: Int,       // 1..9
        val source: String,           // "llm" | "user_tap" | etc
        val prompt: String            // what we asked the user
    ) : Pending()

    data class AskCellValue(
        val cellIndex: Int,
        val prompt: String
    ) : Pending()

    data class ConfirmRetake(
        val strength: String,         // "soft"|"strong"
        val prompt: String
    ) : Pending()

    object ConfirmValidate : Pending()

    // ----------------------------
    // Gate 4: repair conversation states
    // ----------------------------

    /**
     * Model asked us to echo-confirm an interpretation.
     * After a "yes", we either apply (if we have position+digit) or ask the missing part.
     */
    data class ConfirmInterpretation(
        val row: Int? = null,         // 1..9
        val col: Int? = null,         // 1..9
        val digit: Int? = null,       // 1..9
        val confidence: Float = 0.6f,
        val prompt: String
    ) : Pending()

    /**
     * One constrained clarifying step (row-only / col-only / digit-only / position).
     * We keep the last hints we knew so we can continue the repair turn coherently.
     */
    data class AskClarification(
        val kind: ClarifyKind,
        val rowHint: Int? = null,
        val colHint: Int? = null,
        val digitHint: Int? = null,
        val prompt: String
    ) : Pending()

    /**
     * Stop voice parsing position; wait for user tap.
     * If digitHint is present we can confirm/apply after the tap.
     */
    data class WaitForTap(
        val prompt: String,
        val digitHint: Int? = null,
        val confidence: Float = 0.6f
    ) : Pending()
}

data class GridSnapshot(
    val llm: LLMGridState,           // your factual grid context object
    val epochMs: Long = System.currentTimeMillis()
)

data class PendingTick2(
    val toolResults: List<String>,
    val listenAfter: Boolean,
    val fallbackText: String
)

/**
 * ✅ NEW: store-level memory of the last explicit cell confirmation.
 * This lets Sudo acknowledge "no change" deterministically even if the grid did not change.
 */
data class LastCellConfirmation(
    val cellIndex: Int,              // 0..80
    val digit: Int,                  // 0..9 (0=blank)
    val changed: Boolean,            // whether digit differed from what was displayed before confirmation
    val source: String,              // user_voice/user_text/tap/etc
    val seq: Long,                   // monotonic within session (can reuse turnSeq or a separate counter)
    val epochMs: Long = System.currentTimeMillis()
)

data class SudoState(
    val sessionId: String,
    val mode: SudoMode = SudoMode.FREE_TALK,

    val grid: GridSnapshot? = null,
    val pending: Pending? = null,

    val lastUserText: String? = null,
    val lastAssistantText: String? = null,

    // ✅ NEW: last explicit confirmation (including "confirmed same digit")
    val lastConfirmation: LastCellConfirmation? = null,

    /**
     * ✅ NEW: UI-only focus cell (pulsing highlight in SudokuResultView).
     * Not grid truth; just an affordance to guide attention.
     */
    val focusCellIndex: Int? = null,

    // Gate 4: counts consecutive repair turns (used to switch to tap deterministically)
    val repairAttempt: Int = 0,

    val turnSeq: Long = 0L,

    val pendingTick2: PendingTick2? = null
)