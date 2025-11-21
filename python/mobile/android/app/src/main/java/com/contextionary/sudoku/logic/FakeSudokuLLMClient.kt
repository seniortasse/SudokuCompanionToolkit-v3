package com.contextionary.sudoku.logic

import kotlinx.coroutines.delay

/**
 * A very simple fake client you can use to wire the UI and test the flow
 * BEFORE connecting a real LLM backend.
 */
class FakeSudokuLLMClient : SudokuLLMClient {

    override suspend fun sendGridUpdate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): LLMRawResponse {
        // Simulate a tiny bit of latency so you see it in logs / UI if needed.
        delay(100)

        // Extremely naive behavior based on substrings in the developer prompt.
        // This is just to prove the plumbing works end-to-end.
        val severity = when {
            developerPrompt.contains("Severity is retake_needed") -> "retake_needed"
            developerPrompt.contains("Severity is serious") -> "serious"
            developerPrompt.contains("Severity is mild") -> "mild"
            else -> "ok"
        }

        val (assistantMessage, action) = when (severity) {
            "retake_needed" -> {
                "This photo is quite hard to read. Let's retake it with better lighting and a flatter angle." to
                        LLMRawAction(
                            type = "retake_photo"
                        )
            }

            "serious" -> {
                "I see several doubtful cells. Let's review a couple of them together." to
                        LLMRawAction(
                            type = "no_action"
                        )
            }

            "mild" -> {
                "Your puzzle looks almost ready. Please double-check the highlighted cells before we start." to
                        LLMRawAction(
                            type = "no_action"
                        )
            }

            else -> { // "ok"
                "Great, your puzzle looks good. Ready to start solving?" to
                        LLMRawAction(
                            type = "validate_grid"
                        )
            }
        }

        return LLMRawResponse(
            assistant_message = assistantMessage,
            action = action
        )
    }
}
