package com.contextionary.sudoku.conversation

import com.contextionary.sudoku.telemetry.ConversationTelemetry
import java.security.MessageDigest

class PromptBuilder(
    private val store: TurnStore
) {
    data class PromptMessage(val role: MessageRole, val content: String)

    data class BuildResult(
        val messages: List<PromptMessage>,
        val personaHash: String,
        val promptHash: String,
        val msgCount: Int
    )

    private fun emit(sessionId: SessionId, turnId: TurnId?, extras: Map<String, Any?>) {
        val m = mutableMapOf<String, Any?>(
            "type" to "PROMPT_BUILD",
            "session_id" to sessionId
        )
        if (turnId != null) m["turn_id"] = turnId
        for ((k, v) in extras) m[k] = v
        ConversationTelemetry.emit(m)
    }

    fun build(
        sessionId: SessionId,
        persona: PersonaDescriptor,
        systemPrompt: String,
        maxTurns: Int = 8,
        maxHistoryChars: Int = 6000 // ~ rough token budget; tune later
    ): BuildResult {
        val turns = store.loadRecent(sessionId, limit = maxTurns)

        // system message first
        val msgs = ArrayList<PromptMessage>(1 + turns.size * 2)
        msgs.add(PromptMessage(MessageRole.SYSTEM, systemPrompt.trim()))

        for (t in turns) {
            t.userMessage?.let { um ->
                val txt = um.text.trim()
                if (txt.isNotEmpty()) msgs.add(PromptMessage(MessageRole.USER, txt))
            }
            t.assistantMessage?.let { am ->
                val txt = am.text.trim()
                if (txt.isNotEmpty()) msgs.add(PromptMessage(MessageRole.ASSISTANT, txt))
            }
        }

        // ---- Truncation: prefer keeping USER messages; drop oldest ASSISTANT first ----
        fun historyChars(): Int =
            msgs.drop(1).sumOf { it.content.length + 16 } // + role/format overhead

        var droppedAssistant = 0
        var droppedUser = 0
        var truncated = false

        while (msgs.size > 1 && historyChars() > maxHistoryChars) {
            truncated = true

            // Find the oldest assistant message after system
            val idxAssistant = msgs.indexOfFirst { it.role == MessageRole.ASSISTANT }
            if (idxAssistant >= 1) {
                msgs.removeAt(idxAssistant)
                droppedAssistant++
                continue
            }

            // If no assistant msgs left, drop oldest user msg
            val idxUser = msgs.indexOfFirst { it.role == MessageRole.USER }
            if (idxUser >= 1) {
                msgs.removeAt(idxUser)
                droppedUser++
                continue
            }

            // Nothing left to drop (shouldn’t happen)
            break
        }

        val canonical = buildCanonicalString(persona.hash, msgs)
        val promptHash = sha256Hex(canonical)

        emit(
            sessionId = sessionId,
            turnId = turns.lastOrNull()?.turnId,
            extras = mapOf(
                "msg_count" to msgs.size,
                "persona_hash" to persona.hash,
                "prompt_hash" to promptHash,
                "max_turns" to maxTurns,
                "max_history_chars" to maxHistoryChars,
                "history_chars" to historyChars(),
                "truncated" to truncated,
                "dropped_assistant_msgs" to droppedAssistant,
                "dropped_user_msgs" to droppedUser
            )
        )

        return BuildResult(
            messages = msgs,
            personaHash = persona.hash,
            promptHash = promptHash,
            msgCount = msgs.size
        )
    }

    private fun buildCanonicalString(personaHash: String, msgs: List<PromptMessage>): String {
        val sb = StringBuilder()
        sb.append("persona_hash=").append(personaHash).append('\n')
        for (m in msgs) {
            sb.append(m.role.name).append(":")
            sb.append(m.content.replace("\r\n", "\n").trimEnd())
            sb.append('\n')
        }
        return sb.toString()
    }

    private fun sha256Hex(s: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(s.toByteArray(Charsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        for (b in digest) hex.append(String.format("%02x", b))
        return hex.toString()
    }
}