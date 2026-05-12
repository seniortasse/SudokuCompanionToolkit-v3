package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.conductor.policy.ToolplanDiagnostics

import com.contextionary.sudoku.conductor.policy.DecisionKindV1


sealed class Evt {

    data class AppStarted(val epochMs: Long = System.currentTimeMillis()) : Evt()

    // dispatch(Evt.CameraActive) with no parentheses
    object CameraActive : Evt()

    data class GridCaptured(val grid: GridSnapshot) : Evt()
    object GridCleared : Evt()
    data class GridSnapshotUpdated(val grid: GridSnapshot) : Evt()

    /**
     * Tick2 continuation reply (spoken text) with optional diagnostics.
     * Defaults keep existing callsites compatible.
     */
    data class PolicyContinuationReply(
        val ctx: PolicyCallCtx,
        val text: String,
        val diag: ToolplanDiagnostics? = null,

        // Optional correlation metadata (match your other events style)
        val turnId: Long = -1L,
        val tickId: Int = -1,
        val policyReqSeq: Long = -1L,
        val modelCallId: String? = null,
        val toolplanId: String? = null,
        val correlationId: String = ""
    ) : Evt()

    //object PolicyContinuationFailed : Evt()

    data class PolicyContinuationFailed(
        val ctx: PolicyCallCtx,
        val errorCode: String?,
        val errorMsg: String?
    ) : Evt()

    // ASR
    data class AsrFinal(
        val rowId: Int? = null,
        val text: String,
        val confidence: Float? = null
    ) : Evt()

    data class AsrError(val code: Int, val name: String) : Evt()

    // TTS lifecycle (if your TurnController emits these)
    data class TtsStarted(val reason: String) : Evt()
    object TtsFinished : Evt()


    // ----------------------------
    // DecisionTrace reducer events (app-only)
    // ----------------------------

    data class IntentEnvelopeReceived(
        val ctx: PolicyCallCtx,
        val userText: String,
        val env: com.contextionary.sudoku.conductor.policy.IntentEnvelopeV1
    ) : Evt()

    data class DecisionApplied(
        val decisionKind: DecisionKindV1,
        val factBundleTypes: List<String> = emptyList()
    ) : Evt()

    data class ReplyReceived(
        val text: String,
        val source: String = "tick2"
    ) : Evt()

    data class PolicyTools(

        val ctx: PolicyCallCtx,
        val tools: List<ToolCall>,

        // ✅ Deterministic per-tool IDs (same length/order as tools)
        val toolCallIds: List<String>,

        val turnId: Long,
        val tickId: Int,
        val policyReqSeq: Long,
        val modelCallId: String?,
        val toolplanId: String?,
        val correlationId: String,
        val diag: ToolplanDiagnostics?
    ) : Evt()

    data class ToolExecuted(
        val toolCallId: String,
        val toolName: String,

        // stable ID you attach into tick2
        val toolResultId: String,

        // what you attach into tick2 (text or compact json string)
        val toolResultText: String,

        val ok: Boolean = true,
        val err: String? = null,

        val turnId: Long = -1L,
        val tickId: Int = -1,
        val policyReqSeq: Long = -1L,
        val modelCallId: String? = null,
        val toolplanId: String? = null,
        val correlationId: String = ""
    ) : Evt()


    // ----------------------------
    // SOLVING: engine solve step results
    // ----------------------------
    data class SolveStepUpdated(
        val gridHash12: String,
        val stepJson: String,
        val reason: String
    ) : Evt()

    data class SolveStepFailed(
        val gridHash12: String,
        val error: String,
        val reason: String
    ) : Evt()

    data class DetourSolverQuerySucceeded(
        val queryId: String,
        val op: String,
        val resultJson: String,
        val reason: String
    ) : Evt()

    data class DetourSolverQueryFailed(
        val queryId: String,
        val op: String,
        val error: String,
        val reason: String
    ) : Evt()

    // UI taps (grid-mode helpers)
    data class CellTapped(val cellIndex: Int) : Evt()
    data class DigitPicked(val cellIndex: Int, val digit: Int) : Evt()

    // Optional: if you ever emit “manual listen requested” or “cancel speaking”
    object CancelTts : Evt()
}