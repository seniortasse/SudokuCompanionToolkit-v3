package com.contextionary.sudoku

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.contextionary.sudoku.telemetry.ConversationTelemetry
import java.util.concurrent.atomic.AtomicInteger

/**
 * ConversationTurnController — Contract-compliant Conductor
 *
 * Single authority for:
 *  - turn state
 *  - starting/stopping ASR
 *  - gating transitions
 *  - enforcing post-speech pause before listening resumes
 *
 * Workers (ASR, TTS, UI) must NEVER:
 *  - start ASR
 *  - stop ASR
 *  - resume ASR after TTS
 *  - infer state transitions
 *
 * Workers may ONLY notify events:
 *  - onSystemSpeaking(...)
 *  - onTtsFinished()
 *  - onAsrFinal(...)
 *  - onCameraActive()
 *  - onConversationEnded()
 */
class ConversationTurnController {

    enum class TurnState {
        CAMERA,
        SYSTEM_SPEAKING,
        COOLDOWN,
        USER_LISTENING,
        ENDING
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var state: TurnState = TurnState.CAMERA

    fun currentState(): TurnState = state

    // Stable ID for this "conversation session" if you want to correlate events.
    private val sessionId: String = java.util.UUID.randomUUID().toString().take(8)

    // Used to cancel stale delayed ASR starts when state changes quickly.
    private val listenToken = AtomicInteger(0)

    // --- Authority: Conductor-owned ASR commands ---
    private var cmdStartAsr: (() -> Unit)? = null
    private var cmdStopAsr: (() -> Unit)? = null

    /**
     * Attach worker commands ONCE (e.g. from MainActivity).
     * After this, only the Conductor calls these.
     */
    fun attachWorkers(
        startAsr: () -> Unit,
        stopAsr: () -> Unit
    ) {
        cmdStartAsr = startAsr
        cmdStopAsr = stopAsr

        val asrIsNull = (cmdStartAsr == null)
        val stopIsNull = (cmdStopAsr == null)

        // Logcat visibility (super helpful in early bring-up)
        logI("Workers attached (asr_is_null=$asrIsNull stop_is_null=$stopIsNull session_id=$sessionId)")

        // Telemetry visibility
        emit(
            "ASR_WORKERS_ATTACHED",
            mapOf(
                "asr_is_null" to asrIsNull,
                "stop_is_null" to stopIsNull
            )
        )
    }


    // ----------------------- ASR error recovery (contract-compliant) -----------------------

    private var asrRetryCount: Int = 0
    private var asrLastErrorCode: Int? = null

    // Correlation for offline scoring: last accepted ASR row_id and its timestamp
    private var lastAcceptedRowId: Int? = null
    private var lastAcceptedFinalMs: Long? = null

    private fun asrRetryDelayMsFor(error: Int): Long {
        return when (error) {
            // Very common “soft failures”
            android.speech.SpeechRecognizer.ERROR_NO_MATCH,
            android.speech.SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> 250L

            // Network hiccup: backoff
            android.speech.SpeechRecognizer.ERROR_NETWORK,
            android.speech.SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> {
                // simple stepped backoff: 600, 1200, 2000, 2000...
                when (asrRetryCount.coerceAtMost(3)) {
                    0 -> 600L
                    1 -> 1200L
                    else -> 2000L
                }
            }

            // Client-side: usually cancel/stop races; small delay
            android.speech.SpeechRecognizer.ERROR_CLIENT -> 400L

            else -> 800L
        }
    }

    private fun scheduleAsrRetry(reason: String, delayMs: Long) {
        // Only makes sense if we want to be listening.
        if (state != TurnState.USER_LISTENING && state != TurnState.COOLDOWN) {
            emit("ASR_RETRY_SKIPPED", mapOf("reason" to reason, "state" to state.name))
            return
        }

        val token = listenToken.get()
        emit("ASR_RETRY_SCHEDULED", mapOf("reason" to reason, "delay_ms" to delayMs, "retry_count" to asrRetryCount))

        postMainDelayed(delayMs) {
            // Abort if something else happened since scheduling
            if (token != listenToken.get()) {
                emit("ASR_RETRY_ABORTED", mapOf("reason" to reason, "token_stale" to true, "state" to state.name))
                return@postMainDelayed
            }

            // Ensure we are in USER_LISTENING before starting ASR
            if (state == TurnState.COOLDOWN) {
                transitionTo(TurnState.USER_LISTENING, "asr_retry_from_cooldown:$reason")
            }

            dispatchStartAsr("retry:$reason")
        }
    }




    /**
     * Worker should call this when ASR ends with onError(code).
     * Contract: Conductor decides whether/how to restart ASR.
     */
    fun onAsrError(code: Int, name: String? = null) {
        emit(
            "ASR_ERROR",
            mapOf(
                "code" to code,
                "name" to (name ?: "unknown"),
                "retry_count_before" to asrRetryCount,
                "state_at_error" to state.name
            )
        )

        // Stop ASR defensively (some devices keep it half-alive after errors)
        dispatchStopAsr("asr_error:$code")

        asrLastErrorCode = code

        // If we are ending or camera, don’t retry.
        if (state == TurnState.ENDING || state == TurnState.CAMERA) {
            emit("ASR_ERROR_NO_RETRY", mapOf("code" to code, "state" to state.name))
            return
        }

        // Retry policy
        val soft = when (code) {
            android.speech.SpeechRecognizer.ERROR_NO_MATCH,
            android.speech.SpeechRecognizer.ERROR_SPEECH_TIMEOUT,
            android.speech.SpeechRecognizer.ERROR_NETWORK,
            android.speech.SpeechRecognizer.ERROR_NETWORK_TIMEOUT,
            android.speech.SpeechRecognizer.ERROR_CLIENT -> true
            else -> false
        }

        if (!soft) {
            emit("ASR_ERROR_HARD_RECOVER", mapOf("code" to code))
            safeRecover("asr_error_hard:$code")
            return
        }

        // ✅ IMPORTANT SAFETY:
        // If we’re not actively in USER_LISTENING, this error is likely "late" and should NOT
        // pull the system into listening or schedule retries while speaking/cooldown.
        if (state != TurnState.USER_LISTENING) {
            emit(
                "ASR_ERROR_SOFT_IGNORED",
                mapOf(
                    "code" to code,
                    "note" to "soft_error_received_outside_user_listening",
                    "state" to state.name
                )
            )
            return
        }

        // Soft errors: retry with backoff
        val delay = asrRetryDelayMsFor(code)
        asrRetryCount = (asrRetryCount + 1).coerceAtMost(6)

        scheduleAsrRetry("asr_error:$code", delay)
    }



    // ----------------------- Logging / Telemetry -----------------------

    private fun emit(type: String, extras: Map<String, Any?> = emptyMap()) {
        val base = mutableMapOf<String, Any?>(
            "type" to type,
            "session_id" to sessionId,
            "state" to state.name
        )
        for ((k, v) in extras) base[k] = v
        ConversationTelemetry.emit(base)
    }

    private fun logI(msg: String) {
        Log.i("TurnController", msg)
    }

    private fun logW(msg: String) {
        Log.w("TurnController", msg)
    }

    private fun setStateInternal(newState: TurnState, reason: String) {
        if (state == newState) return

        val old = state
        state = newState

        // Any state transition invalidates pending delayed ASR starts.
        listenToken.incrementAndGet()

        logI("STATE $old → $newState ($reason)")
        emit(
            "TURN_STATE",
            mapOf(
                "from" to old.name,
                "to" to newState.name,
                "reason" to reason
            )
        )
    }

    /**
     * Legal transitions (updated to support real conversation loop):
     *
     * CAMERA -> SYSTEM_SPEAKING
     * SYSTEM_SPEAKING -> COOLDOWN
     * COOLDOWN -> USER_LISTENING
     *
     * USER_LISTENING -> SYSTEM_SPEAKING   (SYSTEM responds while listening)
     * COOLDOWN -> SYSTEM_SPEAKING         (fast re-speak / follow-up)
     *
     * USER_LISTENING -> CAMERA            (user cancels / new capture)
     *
     * (any) -> ENDING
     */
    private fun transitionTo(newState: TurnState, reason: String) {
        val old = state

        // ✅ No-op transitions are allowed (prevents CAMERA→CAMERA “invalid transition” recovery spam)
        if (newState == old) {
            emit(
                "TURN_NOOP_TRANSITION",
                mapOf(
                    "state" to old.name,
                    "reason" to reason
                )
            )
            return
        }

        val legal = when (old) {
            TurnState.CAMERA -> (newState == TurnState.SYSTEM_SPEAKING || newState == TurnState.ENDING)
            TurnState.SYSTEM_SPEAKING -> (newState == TurnState.COOLDOWN || newState == TurnState.ENDING)
            TurnState.COOLDOWN -> (
                    newState == TurnState.USER_LISTENING ||
                            newState == TurnState.SYSTEM_SPEAKING || // allow immediate re-speak
                            newState == TurnState.ENDING
                    )
            TurnState.USER_LISTENING -> (
                    newState == TurnState.CAMERA ||
                            newState == TurnState.SYSTEM_SPEAKING || // ✅ critical fix
                            newState == TurnState.ENDING
                    )
            TurnState.ENDING -> false
        }

        if (!legal) {
            emit(
                "TURN_INVALID_TRANSITION",
                mapOf(
                    "from" to old.name,
                    "to" to newState.name,
                    "reason" to reason
                )
            )
            // Generic recovery: stop ASR and go to CAMERA baseline.
            safeRecover("invalid_transition:$old->$newState")
            return
        }

        setStateInternal(newState, reason)
    }

    /**
     * Force state for "high authority" events where recovery-to-camera is worse than
     * accepting the event (e.g., TTS actually started).
     */
    private fun forceState(newState: TurnState, reason: String) {
        val old = state
        if (old == newState) return
        logW("FORCE_STATE $old → $newState ($reason)")
        emit(
            "TURN_FORCE_STATE",
            mapOf(
                "from" to old.name,
                "to" to newState.name,
                "reason" to reason
            )
        )
        setStateInternal(newState, reason)
    }

    // ----------------------- Main-thread helpers -----------------------

    private fun postMain(block: () -> Unit) {
        mainHandler.post { block() }
    }

    private fun postMainDelayed(ms: Long, block: () -> Unit) {
        if (ms <= 0L) postMain(block) else mainHandler.postDelayed({ block() }, ms)
    }

    private fun safeRecover(reason: String) {
        // Always stop ASR during recovery to respect invariant "no overlap"
        try { cmdStopAsr?.invoke() } catch (_: Throwable) {}
        emit("RECOVERY", mapOf("reason" to reason))
        setStateInternal(TurnState.CAMERA, "recovery_to_camera")
    }

    // ----------------------- Contract-owned ASR control -----------------------

    /**
     * Conductor-only ASR start.
     * Must only be called when state == USER_LISTENING.
     */
    private fun dispatchStartAsr(reason: String) {
        if (state != TurnState.USER_LISTENING) {
            emit("ASR_START_BLOCKED", mapOf("reason" to reason, "blocked_state" to state.name))
            return
        }

        val hasCmd = (cmdStartAsr != null)
        emit("ASR_START_DISPATCH", mapOf("reason" to reason, "has_cmd" to hasCmd))

        if (!hasCmd) {
            logW("ASR_START_NO_CMD (did you forget attachWorkers?) reason=$reason")
            emit("ASR_START_NO_CMD", mapOf("reason" to reason))
            return
        }

        try {
            cmdStartAsr?.invoke()
            emit("ASR_START_INVOKED", mapOf("reason" to reason))
        } catch (t: Throwable) {
            emit("ASR_START_ERROR", mapOf("reason" to reason, "message" to (t.message ?: t.toString())))
            safeRecover("asr_start_error")
        }
    }

    /**
     * Conductor-only ASR stop.
     * Can be called from any state when needed (e.g., before speaking).
     */
    private fun dispatchStopAsr(reason: String) {
        emit("ASR_STOP_DISPATCH", mapOf("reason" to reason))
        try {
            cmdStopAsr?.invoke()
        } catch (_: Throwable) {}
    }

    // ----------------------- Public event API (workers call these) -----------------------

    /**
     * Called when camera becomes active (e.g. user returned to capture mode).
     * This is also a safe place to stop ASR if camera should own the turn.
     */
    fun onCameraActive() {
        // Camera owns the turn. No ASR running here.
        dispatchStopAsr("camera_active")
        transitionTo(TurnState.CAMERA, "camera_active")
    }

    /**
     * Called when system begins speaking (TTS start callback).
     * Must guarantee ASR is stopped (no overlap).
     *
     * IMPORTANT: If TTS is actually starting, we must not "recover_to_camera" because
     * that poisons subsequent tts_finished handling. We accept/force SYSTEM_SPEAKING.
     */
    fun onSystemSpeaking(reason: String) {
        // Invariant: speaking => ASR fully stopped.
        dispatchStopAsr("enter_system_speaking:$reason")

        val old = state
        // Prefer legal transition; if illegal due to a race, FORCE speaking state.
        val legal = when (old) {
            TurnState.CAMERA,
            TurnState.USER_LISTENING,
            TurnState.COOLDOWN -> true
            TurnState.SYSTEM_SPEAKING -> true
            TurnState.ENDING -> false
        }

        if (!legal) {
            // If ENDING, ignore.
            emit("SYSTEM_SPEAKING_IGNORED", mapOf("reason" to reason, "note" to "state_ending"))
            return
        }

        // Try normal transition first (captures intent in logs); if it would be illegal,
        // we still force to speaking (because TTS truly started).
        try {
            transitionTo(TurnState.SYSTEM_SPEAKING, reason)
        } catch (_: Throwable) {
            forceState(TurnState.SYSTEM_SPEAKING, "force_system_speaking:$reason")
        }
    }

    /**
     * Called when TTS has finished playback (TTS_DONE/ended).
     *
     * Contract: small pause (200–300ms) before listening resumes.
     *
     * FIX: If we're not currently SYSTEM_SPEAKING, ignore this event instead of
     * trying to transition (prevents CAMERA->COOLDOWN cascades).
     */
    fun onTtsFinished() {
        if (state != TurnState.SYSTEM_SPEAKING) {
            emit(
                "TTS_FINISH_IGNORED",
                mapOf(
                    "note" to "tts_finished_received_outside_system_speaking",
                    "current_state" to state.name
                )
            )
            return
        }

        transitionTo(TurnState.COOLDOWN, "tts_finished")

        // Fixed contract pause.
        val delayMs = 250L

        val token = listenToken.get()
        emit("ASR_REQUEST_ACCEPTED", mapOf("reason" to "post_tts_pause", "delay_ms" to delayMs))

        postMainDelayed(delayMs) {
            // Abort if state changed since scheduling.
            if (token != listenToken.get() || state != TurnState.COOLDOWN) {
                emit(
                    "ASR_START_ABORTED",
                    mapOf(
                        "reason" to "post_tts_pause",
                        "state" to state.name,
                        "token_stale" to (token != listenToken.get())
                    )
                )
                return@postMainDelayed
            }

            transitionTo(TurnState.USER_LISTENING, "asr_start:post_tts_pause")
            dispatchStartAsr("post_tts_pause")
        }
    }

    /**
     * Optional: if your ASR produces a final result, the worker should call this.
     * Conductor can stop ASR immediately; next steps (LLM/TTS) happen elsewhere.
     */
    fun onAsrFinal(text: String, rowId: Int? = null, confidence: Float? = null) {
        // Ensure conductor logic runs on main thread (avoids racey state reads/writes).
        postMain {
            val trimmed = text.trim()
            val preview = trimmed.replace("\n", " ").take(80)

            // Always log what the Conductor received (this is the log line I suggested).
            logI("USER_HEARD state=${state.name} text='$preview'")

            // Telemetry for correlation/debug
            emit(
                "USER_HEARD",
                mapOf(
                    "text_len" to trimmed.length,
                    "preview" to preview,
                    "row_id" to rowId,
                    "confidence" to confidence
                )
            )

            // Drop empty finals on the floor.
            if (trimmed.isBlank()) {
                emit("ASR_FINAL_EMPTY_DROPPED", emptyMap())
                // Still stop ASR defensively.
                dispatchStopAsr("asr_final_empty")
                return@postMain
            }

            // Late final protection: do NOT let a late final disturb the turn contract.
            if (state != TurnState.USER_LISTENING) {
                emit(
                    "ASR_FINAL_IGNORED_LATE",
                    mapOf("note" to "final_received_outside_user_listening")
                )
                // Stop ASR defensively; some devices keep it half-alive.
                dispatchStopAsr("asr_final_late")
                return@postMain
            }

            // Reset retry counters on a successful final.
            asrRetryCount = 0
            asrLastErrorCode = null

            // Stop ASR immediately (end-of-utterance).
            dispatchStopAsr("asr_final")

            // Persist last accepted user row for pairing with the next assistant speak
            lastAcceptedRowId = rowId
            lastAcceptedFinalMs = System.currentTimeMillis()

            // Emit a normalized USER_SAY for transcript reconstruction
            ConversationTelemetry.emitUserSay(
                trimmed,
                source = "turn_controller",
                rowId = rowId,
                confidence = confidence
            )

            // IMPORTANT: We do NOT transition state here.
            // We remain USER_LISTENING until TTS actually starts (onSystemSpeaking()).
            emit("ASR_FINAL_ACCEPTED", mapOf("text_len" to trimmed.length, "row_id" to rowId, "confidence" to confidence))
        }
    }

    /**
     * Optional helper for the orchestrator/UI: pair the next assistant speak with the last user ASR row.
     * consume() clears it so multiple speaks don't accidentally attach to the same user row.
     */
    fun peekLastAcceptedRowId(): Int? = lastAcceptedRowId

    fun consumeLastAcceptedRowId(): Int? {
        val v = lastAcceptedRowId
        lastAcceptedRowId = null
        return v
    }

    fun peekLastAcceptedFinalMs(): Long? = lastAcceptedFinalMs



    fun requestStartAsr(reason: String) {
        emit("ASR_REQUEST", mapOf("reason" to reason))

        when (state) {
            TurnState.COOLDOWN -> {
                // onTtsFinished already schedules it.
                emit("ASR_REQUEST_IGNORED", mapOf("reason" to reason, "note" to "already in cooldown"))
            }

            TurnState.USER_LISTENING -> {
                dispatchStartAsr("request:$reason")
            }

            else -> {
                emit("ASR_REQUEST_BLOCKED", mapOf("reason" to reason, "blocked_state" to state.name))
            }
        }
    }

    /**
     * End of conversation session.
     * Stop ASR, stop any pending delays.
     */
    fun onConversationEnded() {
        dispatchStopAsr("conversation_end")
        setStateInternal(TurnState.ENDING, "conversation_end")
    }
}