package com.contextionary.sudoku.conversation

/**
 * Stable serializable schema for storage/JSON/Room, etc.
 */
data class TurnRecordSchema(
    val sessionId: String,
    val turnId: Long,
    val createdAtMs: Long,
    val updatedAtMs: Long,
    val status: String,

    val userMessage: MessageRecordSchema?,
    val assistantMessage: MessageRecordSchema?,

    val personaVersion: Int,
    val personaHash: String,
    val recoveryNote: String?
)

data class MessageRecordSchema(
    val messageId: String,
    val role: String,
    val text: String,
    val createdAtMs: Long
)

fun TurnRecord.toSchema(): TurnRecordSchema =
    TurnRecordSchema(
        sessionId = sessionId,
        turnId = turnId,
        createdAtMs = createdAtMs,
        updatedAtMs = updatedAtMs,
        status = status.name,
        userMessage = userMessage?.toSchema(),
        assistantMessage = assistantMessage?.toSchema(),
        personaVersion = personaVersion,
        personaHash = personaHash,
        recoveryNote = recoveryNote
    )

fun MessageRecord.toSchema(): MessageRecordSchema =
    MessageRecordSchema(
        messageId = messageId,
        role = role.name,
        text = text,
        createdAtMs = createdAtMs
    )

fun TurnRecordSchema.toTurnRecord(): TurnRecord =
    TurnRecord(
        sessionId = sessionId,
        turnId = turnId,
        createdAtMs = createdAtMs,
        updatedAtMs = updatedAtMs,
        status = runCatching { TurnStatus.valueOf(status) }.getOrElse { TurnStatus.DONE },
        userMessage = userMessage?.toMessageRecord(),
        assistantMessage = assistantMessage?.toMessageRecord(),
        personaVersion = personaVersion,
        personaHash = personaHash,
        recoveryNote = recoveryNote
    )

fun MessageRecordSchema.toMessageRecord(): MessageRecord =
    MessageRecord(
        messageId = messageId,
        role = runCatching { MessageRole.valueOf(role) }.getOrElse { MessageRole.USER },
        text = text,
        createdAtMs = createdAtMs
    )