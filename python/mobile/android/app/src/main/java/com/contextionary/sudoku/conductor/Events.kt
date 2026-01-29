package com.contextionary.sudoku.conductor

sealed class Evt {

    data class AppStarted(val epochMs: Long = System.currentTimeMillis()) : Evt()

    // dispatch(Evt.CameraActive) with no parentheses
    object CameraActive : Evt()

    data class GridCaptured(val grid: GridSnapshot) : Evt()
    object GridCleared : Evt()
    data class GridSnapshotUpdated(val grid: GridSnapshot) : Evt()

    data class PolicyContinuationReply(val text: String) : Evt()
    object PolicyContinuationFailed : Evt()

    // ASR
    data class AsrFinal(
        val rowId: Int? = null,
        val text: String,
        val confidence: Float? = null
    ) : Evt()

    data class AsrError(val code: Int, val name: String) : Evt()

    // TTS lifecycle (if your TurnController emits these)
    data class TtsStarted(val reason: String) : Evt()
    object TtsFinished : Evt()

    // Policy results: tool-calls-only (no raw assistant text here)
    data class PolicyTools(val tools: List<ToolCall>) : Evt()

    // UI taps (grid-mode helpers)
    data class CellTapped(val cellIndex: Int) : Evt()
    data class DigitPicked(val cellIndex: Int, val digit: Int) : Evt()

    // Optional: if you ever emit “manual listen requested” or “cancel speaking”
    object CancelTts : Evt()
}