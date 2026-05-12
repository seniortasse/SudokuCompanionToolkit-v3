package com.contextionary.sudoku.logic

import com.contextionary.sudoku.conductor.policy.IntentEnvelopeV1
import com.contextionary.sudoku.conductor.policy.IntentPayloadV1
import com.contextionary.sudoku.conductor.policy.IntentTargetV1
import com.contextionary.sudoku.conductor.policy.IntentTypeV1
import com.contextionary.sudoku.conductor.policy.IntentV1

/**
 * FakeSudokuLLMClient — deterministic stub for dev/testing.
 *
 * - Tick1: returns a conservative IntentEnvelopeV1 with a single UNKNOWN intent
 *         and a "missing" slot to force clarification paths.
 * - Tick2: returns valid JSON string {"text":"..."} to keep downstream parsers happy.
 * - Free talk / clues: simple stubs.
 */
class FakeSudokuLLMClient : SudokuLLMClient {

    // Tick-1: NLU only
    override suspend fun sendIntentEnvelope(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx?
    ): IntentEnvelopeV1 {
        val raw = userMessage.trim()

        // Conservative: always produce UNKNOWN + missing to trigger clarification/recovery paths.
        val intent = IntentV1(
            id = "t1_i0",
            type = IntentTypeV1.UNKNOWN,
            confidence = 0.0,
            targets = emptyList<IntentTargetV1>(),
            payload = IntentPayloadV1(
                digit = null,
                rawText = raw.take(120).ifBlank { null },
                digits = null,
                regionDigits = null
            ),
            missing = listOf("intent"),
            evidenceText = raw.take(120).ifBlank { "Fake client: empty user text." },
            addressesUserAgendaId = null
        )

        return IntentEnvelopeV1(
            version = "intent_envelope_v1",
            intents = listOf(intent),
            freeTalkTopic = if (raw.isBlank()) "silence" else null,
            freeTalkConfidence = if (raw.isBlank()) 0.6 else 0.0,
            repairSignal = null,
            contextSpanHint = null,
            referencesPriorTurns = null,
            rawUserText = raw.ifBlank { null },
            language = null,
            asrQuality = null
        )
    }

    // Tick-2: Spoken reply JSON -> extract "text"
    override suspend fun sendReplyGenerate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx?
    ): String {
        // Keep it valid JSON so upstream reply parsing never crashes.
        return """{"text":"Fake client: reply-generate stub reply."}"""
    }

    // Free talk (non-grid)
    override suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): FreeTalkRawResponse {
        return FreeTalkRawResponse("Fake client: free talk stub reply.")
    }

    // Clue extraction
    override suspend fun extractClues(
        systemPrompt: String,
        developerPrompt: String,
        transcript: String
    ): ClueExtractionRawResponse {
        return ClueExtractionRawResponse(emptyList())
    }
}