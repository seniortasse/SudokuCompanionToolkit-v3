package com.contextionary.sudoku.conductor

/**
 * Phase 6 (clean): ToolCall is INTERNAL-ONLY.
 *
 * - No LLM “tool schema” surface here.
 * - No wire names / JSON decoding helpers.
 * - These are produced only by app-side planning/engine and executed locally.
 */

// ----------------------------
// Shared internal enums
// ----------------------------

/** Clarification types the app can ask for (internal). */
enum class ClarifyKind {
    ROW,
    COL,
    DIGIT,
    POSITION,
    YESNO,
    // Generic non-grid clarifications (mode, verbosity, preferences, etc.)
    WORKFLOW
}

/** Provenance/classification for a cell (internal). */
enum class CellClass {
    GIVEN,
    SOLUTION,
    EMPTY
}

/**
 * SOLVING: canonical North Star CTA families.
 *
 * These are the only normal stage-level CTA semantics the solving rail should expose:
 * - SHOW_PROOF : setup -> walkthrough/proof
 * - LOCK_IT_IN : confrontation -> commit/apply
 * - NEXT_STEP  : resolution -> continue the solve loop
 *
 * Legacy multi-branch tutoring options are intentionally removed from the canonical enum.
 */
enum class SolvePreferenceOption {
    SHOW_PROOF,
    LOCK_IT_IN,
    NEXT_STEP;

    companion object {
        fun parse(raw: String): SolvePreferenceOption? {
            val key = raw.trim().uppercase().replace("-", "_").replace(" ", "_")
            return when (key) {
                "SHOW_PROOF",
                "GUIDE_ME",
                "MORE_HINT" -> SHOW_PROOF

                "LOCK_IT_IN",
                "READY_FOR_ANSWER",
                "APPLY_NOW",
                "REVEAL_ONLY",
                "REVEAL_WITH_EXPLANATION" -> LOCK_IT_IN

                "NEXT_STEP" -> NEXT_STEP
                else -> null
            }
        }
    }
}

/**
 * Post-resolution canonical CTA family.
 *
 * North Star resolution should reopen the solve loop with NEXT_STEP.
 * Deep-dive / technique-talk is no longer a default sibling CTA in the normal rail.
 */
enum class SolveCtaOption {
    NEXT_STEP;

    companion object {
        fun parse(raw: String): SolveCtaOption? {
            val key = raw.trim().uppercase().replace("-", "_").replace(" ", "_")
            return when (key) {
                "NEXT_STEP" -> NEXT_STEP
                "DEEP_DIVE",
                "ASK_TECHNIQUE" -> null
                else -> null
            }
        }
    }
}

/**
 * INTERNAL app actions/commands (deterministic).
 *
 * Phase 6:
 * - ToolCall is no longer an LLM “tool schema” surface.
 * - Nothing here is decoded from model JSON.
 * - These are produced only by app-side planning/engine and executed locally.
 */
sealed class ToolCall {

    // ----------------------------
    // Core tools (internal)
    // ----------------------------

    /** User-facing assistant text to speak/show (produced by Tick2 renderer or app templates). */
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
     * Row/Col variant to reduce cellIndex mapping mistakes.
     * (row/col are 1..9, digit 0..9, where 0 = blank)
     */
    data class ApplyUserEditRC(
        val row: Int,   // 1..9
        val col: Int,   // 1..9
        val digit: Int, // 0..9
        val source: String = "user_request"
    ) : ToolCall()

    /**
     * Explicit confirmation (NOT an edit).
     *
     * Index-based variant (0..80). Used when the app already knows the concrete cellIndex
     * (e.g., from pending AskCellValue).
     */
    data class ConfirmCellValue(
        val cellIndex: Int,      // 0..80
        val digit: Int,          // 0..9 (0=blank)
        val source: String = "user_confirm"
    ) : ToolCall()

    /**
     * Explicit confirmation (NOT an edit).
     *
     * Row/Col variant (preferred in GRID_SESSION).
     */
    data class ConfirmCellValueRC(
        val row: Int,        // 1..9
        val col: Int,        // 1..9
        val digit: Int,      // 0..9 (0=blank)
        val source: String = "user_confirm"
    ) : ToolCall()

    // ----------------------------
    // Wrap-up / solving transition
    // ----------------------------

    /**
     * Operational:
     * Seal the grid as fully validated and remove all WIP visual artifacts.
     * This does NOT change digits; it changes presentation state.
     */
    data class FinalizeValidationPresentation(
        val reason: String = "validated"
    ) : ToolCall()

    /** Control tool used in CONFIRMING/SEALING flows (kept for backward compat). */
    object AskUserToConfirmValidation : ToolCall()

    /**
     * Control:
     * Switch conversational intent to SOLVING (hints/techniques/next move).
     * No args needed; the user-facing text is still in Reply.
     */
    object StartSolving : ToolCall()

    // ----------------------------
    // Grid validation clarification (NOT an edit)
    // ----------------------------

    enum class ValidationClarifyReason {
        ASR_GARBLED,
        MIXED_SIGNAL,
        OFF_TOPIC,
        PARTIAL_CONFIRM
    }

    enum class ValidationClarifyStyle {
        YES_NO,
        SPOT_CHECK_3,
        ASK_WHICH_CELL_MISMATCHES
    }

    data class ClarifyValidation(
        val reason: ValidationClarifyReason,
        val style: ValidationClarifyStyle,
        val prompt: String
    ) : ToolCall()

    // ----------------------------
    // SOLVING: Evidence/engine + overlay controls (pure UI)
    // ----------------------------

    data class ShowSolveOverlay(
        val stepId: String,
        val frameId: String,
        val style: String,      // "mini"|"full"
        val gridHash12: String
    ) : ToolCall()

    data class HideSolveOverlay(
        val stepId: String,
        val frameId: String,
        val gridHash12: String
    ) : ToolCall()

    data class RefreshSolveStep(
        val gridHash12: String,
        val reason: String
    ) : ToolCall()

    // ----------------------------
    // SOLVING: End-of-turn CTA control tools (REQUIRED in SOLVING)
    // ----------------------------

    /**
     * REQUIRED SOLVING CTA:
     * Ask for the one normal next action for the CURRENT solve-step stage.
     *
     * North Star canonical families:
     * - SHOW_PROOF : walkthrough the logic
     * - LOCK_IT_IN : commit/apply the move
     * - NEXT_STEP  : continue to the next step
     *
     * stepId should reference the cached engine step / CoachPlan id.
     *
     * options remain List<String> for transitional compatibility, but they are
     * expected to normalize into SolvePreferenceOption via the parser below.
     */
    data class AskSolvePreference(
        val stepId: String,
        val options: List<String>,
        val prompt: String,
        val hintIndex: Int,
        val isLastHint: Boolean,
        val gridHash12: String
    ) : ToolCall() {

        /** Best-effort normalization into canonical North Star CTA families. */
        val normalizedOptions: List<SolvePreferenceOption> =
            options.mapNotNull { raw -> SolvePreferenceOption.parse(raw) }
                .distinct()
    }

    /**
     * REQUIRED SOLVING CTA after a completed step:
     * North Star resolution should reopen the solve loop with NEXT_STEP.
     *
     * This tool name is kept temporarily for compatibility, but the normal
     * resolution rail should no longer advertise deep-dive as a default sibling CTA.
     *
     * Note: options are expected to normalize into SolveCtaOption via the parser below.
     */
    data class AskNextStepOrDeepDive(
        val stepId: String,
        val prompt: String,
        val options: List<String> = listOf(
            SolveCtaOption.NEXT_STEP.name
        ),
        val gridHash12: String
    ) : ToolCall() {

        val normalizedOptions: List<SolveCtaOption> =
            options.mapNotNull { raw -> SolveCtaOption.parse(raw) }
                .distinct()
    }

    /** Optional explicit “try it now” CTA before giving more hints. */
    data class AskUserToApplyHint(
        val stepId: String,
        val prompt: String,
        val gridHash12: String
    ) : ToolCall()

    // ----------------------------
    // Truth/provenance tools (givens vs user solutions)
    // ----------------------------

    data class ReclassifyCell(
        val cellIndex: Int,           // 0..80
        val kind: String,             // internal canonical: "given" | "solution" | "neither"
        val source: String = "user_request"
    ) : ToolCall()

    data class ReclassifyCells(
        val cells: List<ReclassifyCell>,
        val source: String = "user_request"
    ) : ToolCall()

    data class ReclassifyCellRC(
        val row: Int,                 // 1..9
        val col: Int,                 // 1..9
        val kind: String,             // internal canonical: "given" | "solution" | "neither"
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

    data class SetCandidates(
        val cellIndex: Int,           // 0..80
        val mask: Int,                // 0..(2^9-1)
        val source: String = "user_request"
    ) : ToolCall()

    data class ClearCandidates(
        val cellIndex: Int,           // 0..80
        val source: String = "user_request"
    ) : ToolCall()

    data class ToggleCandidate(
        val cellIndex: Int,           // 0..80
        val digit: Int,               // 1..9
        val source: String = "user_request"
    ) : ToolCall()

    // ----------------------------
    // Existing control tools (kept)
    // ----------------------------

    data class RecommendRetake(
        val strength: String, // "soft" | "strong"
        val reason: String
    ) : ToolCall()

    /**
     * META / NON-EXECUTING.
     * Explanatory rationale for internal planning (logging/audit only).
     *
     * Must never be treated as operational or control.
     * Must never affect sanitizer decisions.
     */
    data class ToolplanRationale(
        val summary: String,
        val factsUsed: List<String>,
        val rulesUsed: List<String>,
        val chosenControl: String?,
        val chosenOps: List<String>,
        val stateHeaderSha12: String? = null,
        val gridHash12: String? = null
    ) : ToolCall()

    object RecommendValidate : ToolCall()

    // ----------------------------
    // Conversational repair moves (NOT edits)
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
}