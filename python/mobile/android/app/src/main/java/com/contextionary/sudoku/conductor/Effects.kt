package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.logic.LLMGridState

sealed class Eff {

    // UI + voice
    data class Speak(val text: String, val listenAfter: Boolean) : Eff()
    data class UpdateUiMessage(val text: String) : Eff()

    /**
     * ✅ NEW: UI focus highlight for the cell Sudo is currently talking about.
     *
     * - cellIndex != null -> pulse that cell (yellow inset border)
     * - cellIndex == null -> clear pulse
     *
     * This is a pure UI affordance: it must NOT change grid data.
     */
    data class SetFocusCell(
        val cellIndex: Int?,          // 0..80, or null to clear
        val reason: String? = null    // for telemetry/debug
    ) : Eff()

    // Audio machine commands (TurnController/AudioOrchestrator should interpret these)
    data class StopAsr(val reason: String) : Eff()
    data class RequestListen(val reason: String) : Eff()

    // Grid writes must be effects (store applies them)
    data class ApplyCellEdit(val cellIndex: Int, val digit: Int, val source: String) : Eff()

    /**
     * ✅ NEW (Option B): confirmation is NOT an edit.
     *
     * Use this when the user confirms a value for a cell (including "blank"/0),
     * regardless of whether it changes the digit.
     *
     * - changed=false => confirmed same digit that was already displayed; UI SHOULD NOT restyle/rewrite.
     * - changed=true  => confirmation accompanied an actual edit (often emitted alongside ApplyCellEdit).
     */
    data class ConfirmCellValue(
        val cellIndex: Int,      // 0..80
        val digit: Int,          // 0..9 (0=blank)
        val source: String,      // "user_voice" | "user_text" | "tap" | etc
        val changed: Boolean     // whether the digit differed from what was displayed
    ) : Eff()

    // Policy call to LLM layer
    data class CallPolicy(
        val userText: String,
        val stateHeader: String,     // must include mode + pending + repairAttempt + turnSeq
        val gridContext: LLMGridState?,

        // Diagnostics / evidence bundle hooks
        val mode: SudoMode? = null,
        val reason: String? = null,
        val gridHash: String? = null,
        val turnId: Long? = null
    ) : Eff()

    data class ReclassifyCell(
        val cellIndex: Int,       // 0..80
        val kind: String,         // "given" | "solution" | "neither"
        val source: String
    ) : Eff()

    data class SetCandidates(
        val cellIndex: Int,       // 0..80
        val mask: Int,            // 0..0x1FF
        val source: String
    ) : Eff()

    data class ToggleCandidate(
        val cellIndex: Int,       // 0..80
        val digit: Int,           // 1..9
        val source: String
    ) : Eff()

    data class ApplyCellClassify(
        val cellIndex: Int,                 // 0..80
        val cellClass: CellClass,           // GIVEN|SOLUTION|EMPTY
        val source: String
    ) : Eff()

    data class ApplyCellCandidatesMask(
        val cellIndex: Int,
        val candidateMask: Int,             // bits 0..8 => digits 1..9
        val source: String
    ) : Eff()


    data class CallPolicyContinuationTick2(
        val toolResults: List<String>,
        val stateHeader: String,
        val mode: SudoMode,
        val reason: String,
        val turnId: Long
    ) : Eff()

    /**
     * Optional convenience: some apps prefer a single effect that "speaks + updates UI".
     * Keep disabled if you don't use it anywhere.
     */
    data class SpeakAndShow(val text: String, val listenAfter: Boolean) : Eff()
}