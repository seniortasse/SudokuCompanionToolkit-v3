package com.contextionary.sudoku.conversation

/**
 * Core ids.
 */
typealias SessionId = String
typealias TurnId = Long
typealias MessageId = String

/**
 * Roles for prompt building / chat history.
 */
enum class MessageRole {
    SYSTEM,
    USER,
    ASSISTANT
}

/**
 * Message record stored inside a turn.
 */
data class MessageRecord(
    val messageId: MessageId,
    val role: MessageRole,
    val text: String,
    val createdAtMs: Long
)

/**
 * Turn lifecycle states used by TurnLifecycleManager + RecoveryController.
 */
enum class TurnStatus {
    CREATED,
    USER_COMMITTED,
    ASSISTANT_INFLIGHT,
    ASSISTANT_COMMITTED,
    DONE,
    DISCARDED
}

/**
 * Deterministic persona identity.
 */
data class PersonaDescriptor(
    val id: String,
    val version: Int,
    val hash: String
)

/**
 * One conversation turn persisted in the store.
 *
 * NOTE: This matches your TurnLifecycleManager.kt usage exactly:
 *  - createdAtMs, updatedAtMs
 *  - status values above
 *  - userMessage / assistantMessage (MessageRecord?)
 *  - personaVersion / personaHash
 *  - recoveryNote
 */
data class TurnRecord(
    val sessionId: SessionId,
    val turnId: TurnId,

    val createdAtMs: Long,
    val updatedAtMs: Long,

    val status: TurnStatus,

    val userMessage: MessageRecord?,
    val assistantMessage: MessageRecord?,

    val personaVersion: Int,
    val personaHash: String,

    val recoveryNote: String? = null
)
