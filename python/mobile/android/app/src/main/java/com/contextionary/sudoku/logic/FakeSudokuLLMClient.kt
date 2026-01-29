package com.contextionary.sudoku.logic

/**
 * Fake client for offline / tests.
 * Tool-calls-only.
 */
class FakeSudokuLLMClient : SudokuLLMClient {

    override suspend fun sendGridUpdate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        history: List<Pair<String, String>>
    ): PolicyRawResponse {
        return PolicyRawResponse(
            tool_calls = listOf(
                ToolCallRaw(
                    name = "reply",
                    args = mapOf("text" to "Fake: got it. What cell do you want to confirm?")
                ),
                ToolCallRaw(
                    name = "ask_confirm_cell",
                    args = mapOf("cellIndex" to 0, "prompt" to "What digit is at r1c1?")
                )
            )
        )
    }

    override suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): FreeTalkRawResponse {
        return FreeTalkRawResponse("Fake: hello! Tell me more.")
    }

    override suspend fun extractClues(
        systemPrompt: String,
        developerPrompt: String,
        transcript: String
    ): ClueExtractionRawResponse {
        return ClueExtractionRawResponse(emptyList())
    }
}