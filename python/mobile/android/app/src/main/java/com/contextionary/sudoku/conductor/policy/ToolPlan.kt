package com.contextionary.sudoku.conductor.policy

import com.contextionary.sudoku.conductor.ToolCall

/**
 * Phase-6 compliant toolplan structures:
 * - MUST NOT depend on any LLM tool schema types (e.g., LlmToolCall, Tool JSON schema, etc.)
 * - ToolCall here means INTERNAL deterministic ops only.
 *
 * NOTE: If nothing references these anymore, you may delete this file later.
 * Keeping it as a compatibility layer is safe for now.
 */

data class ToolplanDiagnostics(
    val promptHash: String = "",
    val personaHash: String = "",
    val toolplanStatus: String = "accepted",   // accepted / rewritten / recovered
    val rewriteReasons: List<String> = emptyList(),
    val inTools: List<String> = emptyList(),
    val outTools: List<String> = emptyList(),
    val replyLen: Int = 0,
    val chosenControl: String? = null,

    // logging-only rationale (optional)
    val rationaleSummary: String = "",
    val rationaleFacts: List<String> = emptyList(),
    val rationaleRules: List<String> = emptyList()
)

/**
 * ToolplanResult now holds INTERNAL ToolCall only (deterministic ops).
 * If your Phase-6 architecture no longer returns toolplans from the LLM,
 * this type may become unused; keeping it avoids cascading refactors.
 */
data class ToolplanResult(
    val tools: List<ToolCall> = emptyList(),
    val diag: ToolplanDiagnostics = ToolplanDiagnostics()
)