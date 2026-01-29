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
    suspend fun decide(
        sessionId: String,
        userText: String,
        stateHeader: String,   // always includes pending info
        grid: LLMGridState?
    ): List<ToolCall>
}