package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.logic.LLMGridState

/**
 * LlmPolicy = language + judgment module.
 *
 * Conductor remains the boss:
 * - It decides *when* to call the policy.
 * - Policy returns ToolCalls only (no direct UI writes).
 *
 * Gate 0/1 acceptance:
 * - Policy must return a non-empty tool plan (or the caller must emit a safe fallback tool plan).
 */
interface LlmPolicy {

    // -----------------------------
    // ✅ V2 (canonical) decide
    // -----------------------------
    suspend fun decide(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        correlationId: String,
        policyReqSeq: Long,
        modelCallId: String?,
        toolplanId: String?,
        userText: String,
        stateHeader: String,   // always includes pending info
        grid: LLMGridState?
    ): List<ToolCall>

    // -----------------------------
    // ✅ V1 (legacy) decide overload — keeps old call sites compiling
    // -----------------------------
    suspend fun decide(
        sessionId: String,
        userText: String,
        stateHeader: String,
        grid: LLMGridState?
    ): List<ToolCall> {
        return decide(
            sessionId = sessionId,
            turnId = -1L,
            tickId = 1,
            correlationId = "turn--1",
            policyReqSeq = 0L,
            modelCallId = null,
            toolplanId = null,
            userText = userText,
            stateHeader = stateHeader,
            grid = grid
        )
    }

    /**
     * Tick2 continuation: same contract as decide(): return ToolCalls ONLY.
     * Must include Reply + (in GRID mode) exactly one control tool.
     */

    // -----------------------------
    // ✅ V2 (canonical) continueTick2
    // -----------------------------
    suspend fun continueTick2(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        correlationId: String,
        policyReqSeq: Long,
        modelCallId: String?,
        toolplanId: String?,

        systemPrompt: String,
        gridStateAfterTools: LLMGridState?, // nullable matches Eff + conductor
        stateHeader: String,
        toolResults: List<String>,
        toolResultIds: List<String>,
        llm1ReplyText: String,              // the ack the user already heard
        grid: LLMGridState?,                // keep for compatibility (you can pass same as gridStateAfterTools)
        mode: SudoMode,
        reason: String
    ): List<ToolCall>

    // -----------------------------
    // ✅ V1 (legacy) continueTick2 overload — keeps old call sites compiling
    // -----------------------------
    suspend fun continueTick2(
        sessionId: String,
        systemPrompt: String,
        gridStateAfterTools: LLMGridState?,
        stateHeader: String,
        toolResults: List<String>,
        toolResultIds: List<String>,
        llm1ReplyText: String,
        grid: LLMGridState?,
        mode: SudoMode,
        reason: String,
        turnId: Long
    ): List<ToolCall> {
        return continueTick2(
            sessionId = sessionId,
            turnId = turnId,
            tickId = 2,
            correlationId = "turn-$turnId",
            policyReqSeq = 0L,
            modelCallId = null,
            toolplanId = null,

            systemPrompt = systemPrompt,
            gridStateAfterTools = gridStateAfterTools,
            stateHeader = stateHeader,
            toolResults = toolResults,
            toolResultIds = toolResultIds,
            llm1ReplyText = llm1ReplyText,
            grid = grid,
            mode = mode,
            reason = reason
        )
    }
}