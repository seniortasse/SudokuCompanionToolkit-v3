package com.contextionary.sudoku.conversation

import com.contextionary.sudoku.telemetry.ConversationTelemetry
import java.util.UUID
import kotlin.math.max

class TurnLifecycleManager(
    private val store: TurnStore
) {
    private fun now() = System.currentTimeMillis()

    private fun emit(type: String, sessionId: SessionId, extras: Map<String, Any?> = emptyMap()) {
        val m = mutableMapOf<String, Any?>(
            "type" to type,
            "session_id" to sessionId
        )
        for ((k, v) in extras) m[k] = v
        ConversationTelemetry.emit(m)
    }

    private fun newMsgId(): MessageId = UUID.randomUUID().toString()

    fun createTurn(sessionId: SessionId, persona: PersonaDescriptor): TurnRecord {
        val last = store.loadLast(sessionId)
        val nextId = max(1L, (last?.turnId ?: 0L) + 1L)

        val t = TurnRecord(
            sessionId = sessionId,
            turnId = nextId,
            createdAtMs = now(),
            updatedAtMs = now(),
            status = TurnStatus.CREATED,
            userMessage = null,
            assistantMessage = null,
            personaVersion = persona.version,
            personaHash = persona.hash
        )

        store.upsert(t)
        emit("TURN_CREATE", sessionId, mapOf("turn_id" to nextId, "persona_hash" to persona.hash))
        return t
    }


    /**
     * Persist a system-initiated assistant message as its own completed turn.
     *
     * This is used for assistant preludes / greetings that the user hears BEFORE
     * any user utterance in the session, so they become recallable history.
     *
     * Emits the standard lifecycle events so telemetry + scoring remain consistent:
     * TURN_CREATE -> TURN_ASSISTANT_INFLIGHT -> TURN_COMMIT_ASSISTANT -> TURN_DONE
     */
    fun commitAssistantPrelude(
        sessionId: SessionId,
        persona: PersonaDescriptor,
        text: String,
        source: String = "PRELUDE"
    ): TurnRecord {
        // 1) Create a new turn
        val created = createTurn(sessionId = sessionId, persona = persona)

        // 2) Move it to ASSISTANT_INFLIGHT (system-initiated path; no user message)
        val t0 = requireNotNull(store.loadLast(sessionId)) { "No turn exists right after createTurn()" }
        require(t0.turnId == created.turnId) { "Prelude must apply to latest turn" }
        require(t0.status == TurnStatus.CREATED) { "Illegal prelude base state ${t0.status}" }

        val inflight = t0.copy(updatedAtMs = now(), status = TurnStatus.ASSISTANT_INFLIGHT)
        store.upsert(inflight)
        emit(
            "TURN_ASSISTANT_INFLIGHT",
            sessionId,
            mapOf(
                "turn_id" to created.turnId,
                "system_initiated" to true,
                "source" to source
            )
        )

        // 3) Commit assistant + finalize
        val committed = commitAssistant(sessionId = sessionId, turnId = created.turnId, text = text)
        val done = finalizeTurn(sessionId = sessionId, turnId = created.turnId)

        // Optional extra breadcrumb (helps Bucket S diagnosis quickly)
        emit(
            "TURN_PRELUDE_PERSISTED",
            sessionId,
            mapOf(
                "turn_id" to created.turnId,
                "source" to source,
                "text_len" to text.length
            )
        )

        return done
    }

    fun commitUser(sessionId: SessionId, turnId: TurnId, persona: PersonaDescriptor, text: String): TurnRecord {
        val t0 = requireNotNull(store.loadLast(sessionId)) { "No turn exists; call createTurn() first" }
        require(t0.turnId == turnId) { "commitUser must apply to latest turn. expected=${t0.turnId} got=$turnId" }
        require(t0.status == TurnStatus.CREATED) { "Illegal transition ${t0.status} -> USER_COMMITTED" }
        require(t0.personaHash == persona.hash) { "Persona hash changed mid-session (forbidden). old=${t0.personaHash} new=${persona.hash}" }

        val t1 = t0.copy(
            updatedAtMs = now(),
            status = TurnStatus.USER_COMMITTED,
            userMessage = MessageRecord(
                messageId = newMsgId(),
                role = MessageRole.USER,
                text = text,
                createdAtMs = now()
            )
        )

        store.upsert(t1)
        emit("TURN_COMMIT_USER", sessionId, mapOf("turn_id" to turnId, "text_len" to text.length))
        return t1
    }

    fun markAssistantInflight(sessionId: SessionId, turnId: TurnId): TurnRecord {
        val t0 = requireNotNull(store.loadLast(sessionId)) { "No turn exists" }
        require(t0.turnId == turnId) { "markAssistantInflight must apply to latest turn" }
        require(t0.status == TurnStatus.USER_COMMITTED) { "Illegal transition ${t0.status} -> ASSISTANT_INFLIGHT" }

        val t1 = t0.copy(updatedAtMs = now(), status = TurnStatus.ASSISTANT_INFLIGHT)
        store.upsert(t1)
        emit("TURN_ASSISTANT_INFLIGHT", sessionId, mapOf("turn_id" to turnId))
        return t1
    }

    fun commitAssistant(sessionId: SessionId, turnId: TurnId, text: String): TurnRecord {
        val t0 = requireNotNull(store.loadLast(sessionId)) { "No turn exists" }
        require(t0.turnId == turnId) { "commitAssistant must apply to latest turn" }
        require(t0.status == TurnStatus.ASSISTANT_INFLIGHT) { "Illegal transition ${t0.status} -> ASSISTANT_COMMITTED" }

        val t1 = t0.copy(
            updatedAtMs = now(),
            status = TurnStatus.ASSISTANT_COMMITTED,
            assistantMessage = MessageRecord(
                messageId = newMsgId(),
                role = MessageRole.ASSISTANT,
                text = text,
                createdAtMs = now()
            )
        )

        store.upsert(t1)
        emit("TURN_COMMIT_ASSISTANT", sessionId, mapOf("turn_id" to turnId, "text_len" to text.length))
        return t1
    }

    fun finalizeTurn(sessionId: SessionId, turnId: TurnId): TurnRecord {
        val t0 = requireNotNull(store.loadLast(sessionId)) { "No turn exists" }
        require(t0.turnId == turnId) { "finalizeTurn must apply to latest turn" }
        require(t0.status == TurnStatus.ASSISTANT_COMMITTED) { "Illegal transition ${t0.status} -> DONE" }

        val t1 = t0.copy(updatedAtMs = now(), status = TurnStatus.DONE)
        store.upsert(t1)
        emit("TURN_DONE", sessionId, mapOf("turn_id" to turnId))
        return t1
    }

    // ✅ This is what your RecoveryController was trying to call
    fun discardTurn(sessionId: SessionId, turnId: TurnId, note: String): TurnRecord {
        val t0 = store.loadLast(sessionId) ?: error("No turn exists")
        require(t0.turnId == turnId) { "discardTurn must apply to latest turn" }

        val t1 = t0.copy(
            updatedAtMs = now(),
            status = TurnStatus.DISCARDED,
            recoveryNote = note
        )
        store.upsert(t1)
        emit("TURN_DISCARDED", sessionId, mapOf("turn_id" to turnId, "note" to note))
        return t1
    }
}