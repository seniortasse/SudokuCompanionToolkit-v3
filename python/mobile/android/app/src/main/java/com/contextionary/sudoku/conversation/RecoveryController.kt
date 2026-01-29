package com.contextionary.sudoku.conversation

import com.contextionary.sudoku.telemetry.ConversationTelemetry

class RecoveryController(
    private val store: TurnStore,
    private val lifecycle: TurnLifecycleManager
) {
    enum class Decision {
        OK_RESUME,
        DISCARD_INFLIGHT_TURN,
        NEW_SESSION_REQUIRED
    }

    data class Result(
        val decision: Decision,
        val affectedTurnId: TurnId? = null,
        val note: String? = null
    )

    private fun emit(type: String, sessionId: SessionId, extras: Map<String, Any?> = emptyMap()) {
        val m = mutableMapOf<String, Any?>(
            "type" to type,
            "session_id" to sessionId
        )
        for ((k, v) in extras) m[k] = v
        ConversationTelemetry.emit(m)
    }

    /**
     * Run on app start (or when conversation module initializes),
     * BEFORE you allow ASR to start again.
     */
    fun recover(sessionId: SessionId, persona: PersonaDescriptor): Result {
        emit("RECOVERY_BEGIN", sessionId, mapOf("reason" to "process_restart"))

        val last = store.loadLast(sessionId)
        if (last == null) {
            val r = Result(Decision.OK_RESUME, note = "no_previous_turns")
            emit("RECOVERY_END", sessionId, mapOf("result" to "ok", "note" to r.note))
            return r
        }

        // Persona continuity: if persona hash differs, we require new session
        if (last.personaHash != persona.hash) {
            val r = Result(
                decision = Decision.NEW_SESSION_REQUIRED,
                affectedTurnId = last.turnId,
                note = "persona_hash_mismatch"
            )
            emit("RECOVERY_RESOLVE", sessionId, mapOf("turn_id" to last.turnId, "resolution" to "new_session_required"))
            emit("RECOVERY_END", sessionId, mapOf("result" to "ok", "note" to r.note))
            return r
        }

        val r = when (last.status) {
            TurnStatus.CREATED -> {
                // An allocated-but-empty turn can be discarded safely
                lifecycle.discardTurn(sessionId, last.turnId, "recovery_discard_created")
                Result(Decision.DISCARD_INFLIGHT_TURN, last.turnId, "discard_created")
            }

            TurnStatus.USER_COMMITTED -> {
                // User committed but assistant never started: resume normally (you'll rebuild prompt)
                Result(Decision.OK_RESUME, last.turnId, "resume_user_committed")
            }

            TurnStatus.ASSISTANT_INFLIGHT -> {
                // In-flight must be resolved deterministically
                lifecycle.discardTurn(sessionId, last.turnId, "recovery_discard_inflight")
                Result(Decision.DISCARD_INFLIGHT_TURN, last.turnId, "discard_inflight")
            }

            TurnStatus.ASSISTANT_COMMITTED -> {
                // Assistant committed but not DONE: finalize or treat as OK resume
                // We choose OK_RESUME; your app can finalize if desired.
                Result(Decision.OK_RESUME, last.turnId, "resume_assistant_committed")
            }

            TurnStatus.DONE -> {
                Result(Decision.OK_RESUME, last.turnId, "resume_done")
            }

            TurnStatus.DISCARDED -> {
                Result(Decision.OK_RESUME, last.turnId, "resume_after_discard")
            }
        }

        emit(
            "RECOVERY_RESOLVE",
            sessionId,
            mapOf(
                "turn_id" to (r.affectedTurnId ?: -1L),
                "resolution" to r.decision.name,
                "note" to r.note
            )
        )
        emit("RECOVERY_END", sessionId, mapOf("result" to "ok", "note" to (r.note ?: "")))
        return r
    }
}