package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.logic.LLMGridState
import com.contextionary.sudoku.conductor.policy.*

sealed class Eff {

    // UI + voice
    data class Speak(val text: String, val listenAfter: Boolean) : Eff()
    data class UpdateUiMessage(val text: String) : Eff()

    /**
     * ✅ UI focus highlight for the cell Sudo is currently talking about.
     *
     * - cellIndex != null -> pulse that cell (yellow inset border)
     * - cellIndex == null -> clear pulse
     *
     * Pure UI affordance: must NOT change grid data.
     */
    data class SetFocusCell(
        val cellIndex: Int?,          // 0..80, or null to clear
        val reason: String? = null    // for telemetry/debug
    ) : Eff()

    // ----------------------------
    // SOLVING: engine + overlay effects (pure UI + compute)
    // ----------------------------

    data class ComputeSolveStep(
        val grid81: String,
        val gridHash12: String,
        val reason: String,
        val force: Boolean = false
    ) : Eff()

    data class RunDetourSolverQuery(
        val queryId: String,
        val op: String,
        val payloadJson: String,
        val reason: String
    ) : Eff()

    data class RenderSolveOverlay(
        // ✅ Canonical name used everywhere (MainActivity + SudoConductor)
        val frameJson: String,
        val style: String,        // "mini"|"full"
        val reason: String
    ) : Eff() {
        // ✅ Back-compat alias if you have older code referencing stepJson
        val stepJson: String get() = frameJson
    }

    data class ClearSolveOverlay(
        val reason: String
    ) : Eff()

    // Audio machine commands (TurnController/AudioOrchestrator should interpret these)
    data class StopAsr(val reason: String) : Eff()
    data class RequestListen(val reason: String) : Eff()

    // ---------------------------------------------------------------------
    // ✅ Operational effects MUST carry toolCallId (deterministic correlation)
    // ---------------------------------------------------------------------

    // Grid writes must be effects (store applies them)
    data class ApplyCellEdit(
        val toolCallId: String,
        val cellIndex: Int,
        val digit: Int,
        val source: String
    ) : Eff()

    /**
     * ✅ Phase 4: UNDO / REDO (explicit for telemetry clarity)
     *
     * If undo infra isn't ready yet, Store may treat as no-op but MUST still
     * produce a toolResult describing "unsupported" deterministically.
     */
    data class Undo(
        val toolCallId: String,
        val source: String
    ) : Eff()

    data class Redo(
        val toolCallId: String,
        val source: String
    ) : Eff()

    /**
     * ✅ Phase 4: confirmation is NOT an edit.
     *
     * Use this when the user confirms a value for a cell (including "blank"/0),
     * regardless of whether it changes the digit.
     */
    data class ConfirmCellValue(
        val toolCallId: String,
        val cellIndex: Int,      // 0..80
        val digit: Int,          // 0..9 (0=blank)
        val source: String,      // "user_voice" | "user_text" | "tap" | etc
        val changed: Boolean     // whether the digit differed from what was displayed
    ) : Eff()

    data class ReclassifyCell(
        val toolCallId: String,
        val cellIndex: Int,       // 0..80
        val kind: String,         // "given" | "solution" | "neither"
        val source: String
    ) : Eff()

    /**
     * ✅ Candidates: set mask (idempotent)
     */
    data class SetCandidates(
        val toolCallId: String,
        val cellIndex: Int,       // 0..80
        val mask: Int,            // 0..0x1FF
        val source: String
    ) : Eff()

    /**
     * Legacy / UI tap semantics (NOT idempotent).
     * Conductor should prefer AddCandidate/RemoveCandidate for tool-driven ops.
     */
    data class ToggleCandidate(
        val toolCallId: String,
        val cellIndex: Int,       // 0..80
        val digit: Int,           // 1..9
        val source: String
    ) : Eff()

    /**
     * ✅ Phase 4: candidates add/remove (idempotent)
     * Better for retries/audits than ToggleCandidate.
     */
    data class AddCandidate(
        val toolCallId: String,
        val cellIndex: Int,       // 0..80
        val digit: Int,           // 1..9
        val source: String
    ) : Eff()

    data class RemoveCandidate(
        val toolCallId: String,
        val cellIndex: Int,       // 0..80
        val digit: Int,           // 1..9
        val source: String
    ) : Eff()

    data class ApplyCellClassify(
        val toolCallId: String,
        val cellIndex: Int,                 // 0..80
        val cellClass: CellClass,           // GIVEN|SOLUTION|EMPTY
        val source: String
    ) : Eff()

    data class ApplyCellCandidatesMask(
        val toolCallId: String,
        val cellIndex: Int,
        val candidateMask: Int,             // bits 0..8 => digits 1..9
        val source: String
    ) : Eff()

    /**
     * ✅ Atomic UI+overlay cleanup after user confirms the grid matches.
     */
    data class FinalizeValidationPresentation(
        val toolCallId: String,
        val reason: String
    ) : Eff()

    // --------------------------------
    // Policy call to LLM layer (control)
    // --------------------------------
    data class CallPolicy(
        val ctx: PolicyCallCtx,
        val userText: String,
        val stateHeader: String,     // must include mode + pending + repairAttempt + turnSeq
        val gridContext: LLMGridState?,

        // Diagnostics / evidence bundle hooks
        val mode: SudoMode? = null,
        val reason: String? = null,
        val gridHash: String? = null,
        val turnId: Long? = null,

        // ✅ Patch 5.5
        val phase: GridPhase = GridPhase.CONFIRMING,
        val engineStepSummary: String? = null
    ) : Eff()

    // --------------------------------
    // Tick1 Meaning Extract (Intent/Meaning V1)
    // --------------------------------
    data class CallMeaningExtractV1(
        val ctx: PolicyCallCtx,
        val userText: String,
        val stateHeader: String,     // must include mode + pending + repairAttempt + turnSeq

        val mode: SudoMode,
        val reason: String,

        val turnId: Long? = null,

        // keep phase so the prompt can be phase-aware
        val phase: GridPhase = GridPhase.CONFIRMING
    ) : Eff()

    data class CallPolicyContinuationTick2(
        val ctx: PolicyCallCtx,
        val sessionId: String,
        val systemPrompt: String,

        // ✅ MUST be the grid state AFTER the operational tools were applied
        val gridStateAfterTools: LLMGridState?,

        // ✅ LLM1 reply that the user already heard (ack)
        val llm1ReplyText: String,

        // ✅ serialized tool outcomes (apply/confirm results, etc.)
        val toolResults: List<String>,
        val toolResultIds: List<String>,

        // updated header (pending/turnSeq/repairAttempt) after tools
        val stateHeader: String,

        val continuationUserMessage: String = "Continue.",

        val mode: SudoMode,
        val reason: String,

        // ✅ SAME turn id as tick1
        val turnId: Long
    ) : Eff()

    data class CallReplyGenerateV1(
        val ctx: PolicyCallCtx,
        val replyRequest: ReplyRequestV1,
        val planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null,
        val reason: String,
        val turnId: Long
    ) : Eff()

    /**
     * Optional convenience: some apps prefer a single effect that "speaks + updates UI".
     * Keep disabled if you don't use it anywhere.
     */
    data class SpeakAndShow(val text: String, val listenAfter: Boolean) : Eff()
}