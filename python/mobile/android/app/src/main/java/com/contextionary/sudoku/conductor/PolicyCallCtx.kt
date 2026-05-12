package com.contextionary.sudoku.conductor

data class PolicyCallCtx(
    val sessionId: String,
    val turnId: Long,          // your long turn id (you use turnSeq as turn id)
    val tickId: Int,           // 1 or 2
    val policyReqSeq: Long,    // monotonically increasing per session
    val correlationId: String, // e.g. "turn-123"
    val modelCallId: String,   // stable id for this model call span
    val toolplanId: String,    // stable toolplan id for this call
    val mode: String,
    val reason: String,
    val stateHeaderSha12: String?
)