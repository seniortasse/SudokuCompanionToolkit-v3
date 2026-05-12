package com.contextionary.sudoku.telemetry

import android.util.Log
import com.contextionary.sudoku.conductor.PolicyCallCtx

/**
 * CHECKPOINT trace helper.
 *
 * Always logs to:
 *  - Logcat
 *  - ConversationTelemetry (type="CHECKPOINT")
 *
 * Always includes:
 *  sessionId, turnSeq, turnId, tickId, correlationId, policyReqSeq, toolplanId, modelCallId, tag
 */
object Checkpoint {

    private const val LOG_TAG = "CP"

    @JvmStatic
    fun cp(
        tag: String,
        sessionId: String?,
        turnSeq: Long?,
        turnId: Long?,
        tickId: Int?,
        correlationId: String?,
        policyReqSeq: Long?,
        toolplanId: String?,
        modelCallId: String?,
        kv: Map<String, Any?> = emptyMap()
    ) {
        // Build payload (stable order helps when eyeballing JSON)
        val payload = linkedMapOf<String, Any?>(
            "type" to "CHECKPOINT",
            "tag" to tag,

            "session_id" to sessionId,
            "turn_seq" to turnSeq,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "correlation_id" to correlationId,
            "policy_req_seq" to policyReqSeq,
            "toolplan_id" to toolplanId,
            "model_call_id" to modelCallId
        )

        // Attach user KV (but don't allow overriding core keys)
        for ((k, v) in kv) {
            if (!payload.containsKey(k)) payload[k] = v
        }

        // Logcat (small + readable)
        runCatching {
            val small = buildString {
                append(tag)
                append(" sid=").append(sessionId ?: "null")
                append(" turnSeq=").append(turnSeq ?: "null")
                append(" turnId=").append(turnId ?: "null")
                append(" tick=").append(tickId ?: "null")
                append(" req=").append(policyReqSeq ?: "null")
                append(" tp=").append(toolplanId ?: "null")
                append(" mc=").append(modelCallId ?: "null")
            }
            Log.i(LOG_TAG, small)
        }

        // Telemetry JSON
        runCatching {
            ConversationTelemetry.emit(payload)
        }
    }

    /**
     * Convenience overload when you already have a PolicyCallCtx.
     * (Note: turnSeq is not in ctx, so pass it explicitly if you want it.)
     */
    @JvmStatic
    fun cp(
        tag: String,
        ctx: PolicyCallCtx?,
        turnSeq: Long? = null,
        kv: Map<String, Any?> = emptyMap()
    ) {
        cp(
            tag = tag,
            sessionId = ctx?.sessionId,
            turnSeq = turnSeq,
            turnId = ctx?.turnId,
            tickId = ctx?.tickId,
            correlationId = ctx?.correlationId,
            policyReqSeq = ctx?.policyReqSeq,
            toolplanId = ctx?.toolplanId,
            modelCallId = ctx?.modelCallId,
            kv = kv
        )
    }
}