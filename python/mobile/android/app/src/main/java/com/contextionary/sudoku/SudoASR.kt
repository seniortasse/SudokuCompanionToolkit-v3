package com.contextionary.sudoku

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

import com.contextionary.sudoku.telemetry.ConversationTelemetry

/**
 * SudoASR — Robust SpeechRecognizer wrapper for conversational apps.
 *
 * Goals:
 * - Deterministic ownership: the conductor decides when to start/stop/cancel.
 * - Strong session isolation: late callbacks from old sessions are ignored.
 * - Main-thread correctness: SpeechRecognizer is created/used/destroyed on main thread.
 * - Clear semantics: stop() vs cancel() are distinct.
 * - Defensive OEM behavior handling: empty results, NO_MATCH storms, BUSY, ERROR_CLIENT races.
 * - Telemetry-first: every ASR "row" begins and ends exactly once.
 *
 * Contract:
 * - Call startListening() only when you truly want to listen (user turn).
 * - Call cancelListening() for turn transitions (TTS start, camera, recovery).
 * - Call stopListening() only when you want to request "finalization" (rare; usually the recognizer
 *   finalizes itself after END).
 */
class SudoASR(
    private val context: Context,
    private val config: Config = Config()
) {

    data class Config(
        // Recognition
        val partialResults: Boolean = true,
        val maxResults: Int = 3,
        val preferOffline: Boolean = true,
        val preferredLocaleTag: String = "",
        val defaultLocaleTag: String = "en-US",

        // Silence tuning (OEM-dependent). We auto-disable after repeated suspicious failures.
        val useSilenceTuningExtras: Boolean = true,
        val completeSilenceMs: Long = 6000L,
        val possiblyCompleteSilenceMs: Long = 3500L,
        val minInputLengthMs: Long = 1500L,

        // Start throttling & recovery
        val minGapBetweenStartsMs: Long = 300L,
        val busyBackoffMs: Long = 450L,
        val clientErrorBackoffMs: Long = 450L,

        // Empty final handling (some OEMs emit empty onResults)
        val emptyFinalGraceMs: Long = 250L,

        // Auto-disable silence extras after N consecutive suspicious failures
        val silenceExtrasDisableThreshold: Int = 2,

        // Audio focus (optional; helps some devices route mic correctly)
        val requestAudioFocus: Boolean = true
    )

    interface Listener {
        fun onReady(localeTag: String) {}
        fun onBegin() {}
        fun onRmsChanged(rmsDb: Float) {}

        fun onPartial(text: String) {}
        fun onFinal(rowId: Int, text: String, confidence: Float?, reason: String) {}

        fun onEnd() {}
        fun onError(code: Int) {}
        fun onDebugEvent(type: String, extras: Map<String, Any?> = emptyMap()) {}
    }

    var listener: Listener? = null

    // ---------- Public gates ----------
    @Volatile private var suppressed: Boolean = false
    @Volatile private var isSpeaking: Boolean = false

    /**
     * speaking=true: immediately cancel listening and suppress callbacks.
     * speaking=false: lift suppression (does NOT auto-start).
     */
    fun setSpeaking(speaking: Boolean) {
        isSpeaking = speaking
        if (speaking) {
            suppressed = true
            cancelListening("tts_speaking")
            emit("ASR_SUPPRESS_FOR_TTS", mapOf("speaking" to true))
        } else {
            suppressed = false
            emit("ASR_SUPPRESS_LIFTED", mapOf("speaking" to false))
        }
    }

    fun setSuppressed(s: Boolean, reason: String) {
        suppressed = s
        if (s) cancelListening("suppressed:$reason")
        emit("ASR_SUPPRESSED_SET", mapOf("suppressed" to s, "reason" to reason))
    }

    fun setConversationLocaleTag(tag: String?) {
        conversationLocaleTag = tag?.trim()?.ifEmpty { null }
        emit("ASR_CONVERSATION_LOCALE_SET", mapOf("tag" to (conversationLocaleTag ?: "")))
    }

    // ---------- Lifecycle ----------
    fun prewarm() {
        // Only validates locale parsing. Does not create the recognizer.
        val tag = effectiveLocale().toLanguageTag()
        emit("ASR_PREWARM_OK", mapOf("locale" to tag))
    }

    fun release() {
        runOnMain {
            cancelListening("release")
            destroyRecognizerLocked("release")
            state = State.IDLE
            currentRowId = -1
            currentSessionToken = 0L
            currentLocaleTag = ""
            lastStartAtMs = 0L
        }
    }

    // ---------- Control API ----------
    fun isActive(): Boolean = (state == State.LISTENING || state == State.STARTING)

    /**
     * Start a listening session.
     * The "row" is started here and will always be ended exactly once.
     */
    fun startListening() {
        runOnMain {
            if (!canStartNow()) return@runOnMain

            val now = System.currentTimeMillis()
            if (now - lastStartAtMs < config.minGapBetweenStartsMs) {
                emit("ASR_START_SKIPPED", mapOf("reason" to "debounce", "delta_ms" to (now - lastStartAtMs)))
                return@runOnMain
            }

            if (!hasMicPermission()) {
                emit("ASR_START_DENIED", mapOf("reason" to "missing_record_audio_permission"))
                safeToast("Enable microphone permission to talk to Sudo.")
                return@runOnMain
            }

            if (!SpeechRecognizer.isRecognitionAvailable(context)) {
                emit("ASR_START_DENIED", mapOf("reason" to "recognition_not_available"))
                safeToast("Speech recognition not available on this device.")
                return@runOnMain
            }

            lastStartAtMs = now
            state = State.STARTING

            // New session token; all callbacks must match this token.
            currentSessionToken = tokenGen.incrementAndGet()

            // Choose locale and build intent
            val chosenLocaleTag = chooseLocaleTag()
            currentLocaleTag = chosenLocaleTag

            // Determine silence extras mode (with auto-disable latch)
            val silenceExtrasEnabled = config.useSilenceTuningExtras && !silenceExtrasDisabled
            emit(
                "ASR_SILENCE_EXTRAS_MODE",
                mapOf(
                    "enabled" to silenceExtrasEnabled,
                    "disabled_latch" to silenceExtrasDisabled,
                    "disabled_reason" to (silenceExtrasDisabledReason ?: ""),
                    "complete_ms" to config.completeSilenceMs,
                    "possibly_ms" to config.possiblyCompleteSilenceMs,
                    "min_len_ms" to config.minInputLengthMs
                )
            )

            // Start telemetry row
            currentRowId = ConversationTelemetry.asrRowStart(
                "locale" to chosenLocaleTag,
                "session_token" to currentSessionToken
            )
            emitRow("ASR_ROW_START", mapOf("row_id" to currentRowId, "locale" to chosenLocaleTag))

            // Always recreate recognizer per session (OEM robustness)
            destroyRecognizerLocked("start_new_session")
            recognizer = SpeechRecognizer.createSpeechRecognizer(context).also { r ->
                r.setRecognitionListener(makeListener(currentSessionToken, chosenLocaleTag, silenceExtrasEnabled))
            }
            intent = buildIntent(chosenLocaleTag, silenceExtrasEnabled)

            // Optional audio focus
            acquireAudioFocusIfNeeded("startListening")

            emitRow("ASR_LISTEN_START", mapOf("locale" to chosenLocaleTag, "token" to currentSessionToken))
            listener?.onDebugEvent("ASR_LISTEN_START", mapOf("locale" to chosenLocaleTag, "token" to currentSessionToken))

            try {
                recognizer?.startListening(intent)
                state = State.LISTENING
            } catch (t: Throwable) {
                emitRow("ASR_START_THROW", mapOf("message" to (t.message ?: t.toString())))
                endRowOnceLocked("ASR_ERROR", mapOf("reason" to "start_throw", "message" to (t.message ?: t.toString())))
                releaseAudioFocusIfNeeded("start_throw")
                state = State.IDLE
                destroyRecognizerLocked("start_throw")
            }
        }
    }

    /**
     * stopListening():
     * A polite request to end and deliver final results. Use sparingly.
     * For turn transitions, prefer cancelListening().
     */
    fun stopListening(reason: String = "stop") {
        runOnMain {
            if (state == State.IDLE) return@runOnMain
            if (state == State.DESTROYED) return@runOnMain

            emitRow("ASR_STOP_REQUESTED", mapOf("reason" to reason))
            try { recognizer?.stopListening() } catch (_: Throwable) {}

            // Do NOT end row here. We end on onResults/onError or onTimeoutPath.
            // But we do move to STOPPING so we don't accept "new starts" accidentally.
            state = State.STOPPING
        }
    }

    /**
     * cancelListening():
     * Hard cancel; drops results. Use for TTS start, camera, recovery, etc.
     * Cancelling ends the row deterministically.
     */
    fun cancelListening(reason: String = "cancel") {
        runOnMain {
            if (state == State.IDLE) return@runOnMain
            if (state == State.DESTROYED) return@runOnMain

            emitRow("ASR_CANCEL_REQUESTED", mapOf("reason" to reason))
            try { recognizer?.cancel() } catch (_: Throwable) {}

            // End row immediately; ignore late callbacks (session token invalidation)
            endRowOnceLocked("ASR_STOP", mapOf("reason" to "cancel:$reason"))
            releaseAudioFocusIfNeeded("cancel:$reason")

            // Invalidate token so late callbacks are rejected.
            currentSessionToken = 0L
            state = State.IDLE
            destroyRecognizerLocked("cancel:$reason")
        }
    }

    // ---------- Internals ----------
    private enum class State { IDLE, STARTING, LISTENING, STOPPING, DESTROYED }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val logTag = "SudoASR"

    private var recognizer: SpeechRecognizer? = null
    private var intent: Intent? = null

    @Volatile private var state: State = State.IDLE
    @Volatile private var currentSessionToken: Long = 0L
    @Volatile private var currentLocaleTag: String = ""
    @Volatile private var currentRowId: Int = -1

    private var lastStartAtMs: Long = 0L
    private var lastPartial: String = ""

    @Volatile private var conversationLocaleTag: String? = null

    // Locale failure latches (reactive)
    @Volatile private var lastFailedLocaleTag: String? = null
    @Volatile private var lastLanguageNotSupported: Boolean = false // 12
    @Volatile private var lastLanguageUnavailable: Boolean = false  // 13

    // Silence extras auto-disable latch
    @Volatile private var silenceExtrasDisabled: Boolean = false
    @Volatile private var silenceExtrasDisabledReason: String? = null
    private var consecutiveSuspiciousNoHypothesis: Int = 0

    // Token generator for session isolation
    private val tokenGen = AtomicLong(1000L)

    // Audio focus
    private val audioManager: AudioManager? =
        context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
    private var hasAudioFocus: Boolean = false

    private fun runOnMain(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else mainHandler.post { block() }
    }

    private fun canStartNow(): Boolean {
        if (suppressed) {
            emit("ASR_START_BLOCKED", mapOf("reason" to "suppressed"))
            return false
        }
        if (isSpeaking) {
            emit("ASR_START_BLOCKED", mapOf("reason" to "is_speaking"))
            return false
        }
        if (state == State.LISTENING || state == State.STARTING || state == State.STOPPING) {
            emit("ASR_START_BLOCKED", mapOf("reason" to "already_active", "state" to state.name))
            return false
        }
        return true
    }

    private fun hasMicPermission(): Boolean {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
    }

    private fun safeToast(msg: String) {
        // Avoid bringing in Toast here if you prefer pure engine class; keep it silent.
        // If you want to show toast, wire it from UI layer. We only log/telemetry here.
        Log.w(logTag, msg)
    }

    private fun buildIntent(localeTag: String, silenceExtrasEnabled: Boolean): Intent {
        return Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, localeTag)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, localeTag)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, config.partialResults)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, max(1, config.maxResults))

            if (config.preferOffline) {
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            }

            if (silenceExtrasEnabled) {
                putExtra(
                    RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS,
                    config.completeSilenceMs
                )
                putExtra(
                    RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS,
                    config.possiblyCompleteSilenceMs
                )
                putExtra(
                    RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS,
                    config.minInputLengthMs
                )
            }

            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
        }
    }

    private fun effectiveLocale(): Locale {
        val conv = conversationLocaleTag?.trim().orEmpty()
        if (conv.isNotEmpty()) {
            runCatching { Locale.forLanguageTag(conv).takeIf { it.language.isNotBlank() } }
                .getOrNull()?.let { return it }
        }

        val cfg = config.preferredLocaleTag.trim()
        if (cfg.isNotEmpty()) {
            runCatching { Locale.forLanguageTag(cfg).takeIf { it.language.isNotBlank() } }
                .getOrNull()?.let { return it }
        }

        runCatching { Locale.forLanguageTag(config.defaultLocaleTag).takeIf { it.language.isNotBlank() } }
            .getOrNull()?.let { return it }

        return Locale.getDefault()
    }

    private fun chooseLocaleTag(): String {
        val requested = effectiveLocale().toLanguageTag()
        val deviceDefault = Locale.getDefault().toLanguageTag()

        // Only run fallback ladder if we latched a locale failure last time
        if (!lastLanguageUnavailable && !lastLanguageNotSupported) return requested

        val candidates = LinkedHashSet<String>()
        candidates.add(deviceDefault)

        runCatching { Locale.forLanguageTag(deviceDefault).language }
            .getOrNull()?.takeIf { it.isNotBlank() }?.let { candidates.add(it) }

        candidates.add("en-US")
        candidates.add("en-GB")

        runCatching { Locale.forLanguageTag(requested).language }
            .getOrNull()?.takeIf { it.isNotBlank() }?.let { candidates.add(it) }

        candidates.add(requested)

        val chosen = candidates.firstOrNull { it.isNotBlank() && it != lastFailedLocaleTag } ?: requested

        emit(
            "ASR_LOCALE_CHOSEN",
            mapOf(
                "requested" to requested,
                "chosen" to chosen,
                "device_default" to deviceDefault,
                "last_failed" to (lastFailedLocaleTag ?: ""),
                "prev_lang_unavailable" to lastLanguageUnavailable,
                "prev_lang_not_supported" to lastLanguageNotSupported
            )
        )
        return chosen
    }

    private fun destroyRecognizerLocked(reason: String) {
        try { recognizer?.setRecognitionListener(null) } catch (_: Throwable) {}
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
        intent = null
        emit("ASR_RECOGNIZER_DESTROYED", mapOf("reason" to reason))
    }

    private fun callbacksAllowed(token: Long): Boolean {
        if (suppressed || isSpeaking) return false
        // Session isolation: reject late callbacks
        if (token == 0L || token != currentSessionToken) return false
        return true
    }

    private fun makeListener(
        token: Long,
        localeTag: String,
        silenceExtrasEnabled: Boolean
    ): RecognitionListener {

        // row-end idempotency
        var ended = false

        // hypothesis tracking
        var sawHypothesis = false

        // empty final grace runnable
        var pendingEmptyFinalize: Runnable? = null

        fun cancelEmptyFinalize(reason: String) {
            pendingEmptyFinalize?.let { mainHandler.removeCallbacks(it) }
            pendingEmptyFinalize = null
            if (reason.isNotBlank()) emitRow("ASR_EMPTY_FINAL_CANCELLED", mapOf("reason" to reason))
        }

        fun endRowOnce(kind: String, extras: Map<String, Any?> = emptyMap()) {
            if (ended) return
            ended = true
            endRowOnceLocked(kind, extras)
        }

        fun clearLocaleFailureLatches() {
            lastLanguageNotSupported = false
            lastLanguageUnavailable = false
            lastFailedLocaleTag = null
        }

        fun maybeAutoDisableSilenceExtrasOnSuspiciousEnd(errorCode: Int, errorName: String) {
            val isNoMatchish =
                (errorCode == SpeechRecognizer.ERROR_NO_MATCH || errorCode == SpeechRecognizer.ERROR_SPEECH_TIMEOUT)

            if (!silenceExtrasEnabled || !isNoMatchish || sawHypothesis) {
                consecutiveSuspiciousNoHypothesis = 0
                return
            }

            consecutiveSuspiciousNoHypothesis += 1
            emit(
                "ASR_SILENCE_EXTRAS_SUSPICIOUS_FAILURE",
                mapOf(
                    "count" to consecutiveSuspiciousNoHypothesis,
                    "threshold" to config.silenceExtrasDisableThreshold,
                    "code" to errorCode,
                    "name" to errorName,
                    "token" to token
                )
            )

            if (consecutiveSuspiciousNoHypothesis >= config.silenceExtrasDisableThreshold) {
                silenceExtrasDisabled = true
                silenceExtrasDisabledReason = "auto_disabled_after_${consecutiveSuspiciousNoHypothesis}_no_hypothesis_failures"
                emit(
                    "ASR_SILENCE_EXTRAS_AUTO_DISABLED",
                    mapOf("reason" to (silenceExtrasDisabledReason ?: ""), "token" to token)
                )
            }
        }

        return object : RecognitionListener {

            override fun onReadyForSpeech(params: Bundle?) {
                if (!callbacksAllowed(token) || ended) return
                listener?.onReady(localeTag)
                emitRow("ASR_READY", mapOf("locale" to localeTag, "token" to token))
            }

            override fun onBeginningOfSpeech() {
                if (!callbacksAllowed(token) || ended) return
                clearLocaleFailureLatches()
                listener?.onBegin()
                emitRow("ASR_BEGIN", mapOf("token" to token))
            }

            override fun onRmsChanged(rmsdB: Float) {
                if (!callbacksAllowed(token) || ended) return
                listener?.onRmsChanged(rmsdB)
            }

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                if (!callbacksAllowed(token) || ended) return
                listener?.onEnd()
                emitRow("ASR_END", mapOf("token" to token))
            }

            override fun onPartialResults(partialResults: Bundle?) {
                if (!callbacksAllowed(token) || ended) return

                val parts = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = parts?.firstOrNull()?.orEmpty() ?: return
                if (text.isBlank()) return

                if (text != lastPartial) {
                    sawHypothesis = true
                    lastPartial = text
                    listener?.onPartial(text)
                    emitRow("ASR_PARTIAL", mapOf("text" to text, "token" to token))
                }
            }

            override fun onResults(results: Bundle?) {
                if (!callbacksAllowed(token) || ended) return

                clearLocaleFailureLatches()
                cancelEmptyFinalize("onResults")

                // Prefer final; salvage from partial if empty
                val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val scores = results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)

                val finalCandidate = list?.firstOrNull()?.trim().orEmpty()
                val partialCandidate = lastPartial.trim()
                val conf = scores?.firstOrNull()

                val chosenText = when {
                    finalCandidate.isNotBlank() -> finalCandidate
                    partialCandidate.isNotBlank() -> partialCandidate
                    else -> ""
                }

                // Some OEMs emit empty onResults then real onResults shortly after.
                if (chosenText.isBlank()) {
                    emitRow("ASR_FINAL_EMPTY", mapOf("token" to token))

                    val runnable = Runnable {
                        // Only finalize if still same session and not ended
                        if (!callbacksAllowed(token) || ended) return@Runnable
                        endRowOnce("ASR_STOP", mapOf("reason" to "final_empty_grace_expired", "token" to token))
                        releaseAudioFocusIfNeeded("final_empty_grace_expired")
                        // Invalidate session token (reject late callbacks)
                        currentSessionToken = 0L
                        state = State.IDLE
                        destroyRecognizerLocked("final_empty_grace_expired")
                    }

                    pendingEmptyFinalize = runnable
                    mainHandler.postDelayed(runnable, config.emptyFinalGraceMs)
                    return
                }

                sawHypothesis = true
                lastPartial = ""
                state = State.IDLE

                val reason = if (finalCandidate.isNotBlank()) "final_results" else "final_from_partial"
                emitRow("ASR_FINAL", mapOf("text" to chosenText, "confidence" to conf, "reason" to reason, "token" to token))
                listener?.onFinal(currentRowId, chosenText, conf, reason)

                endRowOnce("ASR_STOP", mapOf("reason" to reason, "final_text" to chosenText, "token" to token))
                releaseAudioFocusIfNeeded("final:$reason")

                // Invalidate token so late callbacks are rejected.
                currentSessionToken = 0L
                destroyRecognizerLocked("final:$reason")
            }

            override fun onError(error: Int) {
                val name = errorName(error)

                if (ended) return
                if (!callbacksAllowed(token)) return

                cancelEmptyFinalize("onError:$name")

                // OEM: salvage partial on NO_MATCHish error if we have a good partial
                val salvageCandidate = lastPartial.trim()
                val isNoMatchish =
                    (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT)

                maybeAutoDisableSilenceExtrasOnSuspiciousEnd(error, name)

                if (isNoMatchish && salvageCandidate.isNotBlank()) {
                    sawHypothesis = true
                    lastPartial = ""
                    state = State.IDLE

                    emitRow(
                        "ASR_FINAL_SALVAGED_ON_ERROR",
                        mapOf("text" to salvageCandidate, "code" to error, "name" to name, "token" to token)
                    )
                    listener?.onFinal(currentRowId, salvageCandidate, null, "error_${name}_salvaged_partial")

                    endRowOnce("ASR_STOP", mapOf("reason" to "final_salvaged_on_error", "final_text" to salvageCandidate, "token" to token))
                    releaseAudioFocusIfNeeded("salvage_on_error")

                    currentSessionToken = 0L
                    destroyRecognizerLocked("salvage_on_error")
                    return
                }

                // Latch locale failures for next start
                when (error) {
                    12 -> {
                        lastLanguageNotSupported = true
                        lastFailedLocaleTag = localeTag
                        emit("ASR_LANG_NOT_SUPPORTED_LATCHED", mapOf("locale" to localeTag, "token" to token))
                    }
                    13 -> {
                        lastLanguageUnavailable = true
                        lastFailedLocaleTag = localeTag
                        emit("ASR_LANG_UNAVAILABLE_LATCHED", mapOf("locale" to localeTag, "fallback" to Locale.getDefault().toLanguageTag(), "token" to token))
                    }
                }

                // Handle BUSY/CLIENT with optional backoff (caller might retry)
                state = State.IDLE
                lastPartial = ""

                emitRow("ASR_ERROR", mapOf("code" to error, "name" to name, "token" to token))
                listener?.onError(error)

                endRowOnce("ASR_ERROR", mapOf("code" to error, "name" to name, "token" to token))
                releaseAudioFocusIfNeeded("error:$name")

                currentSessionToken = 0L
                destroyRecognizerLocked("error:$name")
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
    }

    private fun endRowOnceLocked(kind: String, extras: Map<String, Any?> = emptyMap()) {
        // Always end exactly once for the current rowId if valid.
        val rowId = currentRowId
        if (rowId < 0) return

        val payload = mutableMapOf<String, Any?>(
            "kind" to kind,
            "row_id" to rowId,
            "locale" to currentLocaleTag,
            "state" to state.name
        )
        for ((k, v) in extras) payload[k] = v

        // Use your existing telemetry API shape:
        ConversationTelemetry.asrRowEnd(kind, *payload.entries.map { it.key to it.value }.toTypedArray())

        currentRowId = -1
    }

    private fun emit(type: String, extras: Map<String, Any?> = emptyMap()) {
        val base = mutableMapOf<String, Any?>(
            "type" to type,
            "asr_state" to state.name,
            "locale" to currentLocaleTag,
            "session_token" to currentSessionToken
        )
        for ((k, v) in extras) base[k] = v
        ConversationTelemetry.emit(base)
        listener?.onDebugEvent(type, extras)
        Log.i(logTag, "$type $extras")
    }

    private fun emitRow(type: String, extras: Map<String, Any?> = emptyMap()) {
        val base = mutableMapOf<String, Any?>(
            "type" to type,
            "row_id" to currentRowId,
            "asr_state" to state.name,
            "locale" to currentLocaleTag,
            "session_token" to currentSessionToken
        )
        for ((k, v) in extras) base[k] = v
        ConversationTelemetry.asrRowEvent(type, *base.entries.map { it.key to it.value }.toTypedArray())
        listener?.onDebugEvent(type, extras)
        Log.i(logTag, "$type $extras")
    }

    companion object {
        @JvmStatic
        fun errorName(code: Int): String = when (code) {
            SpeechRecognizer.ERROR_AUDIO -> "ERROR_AUDIO"
            SpeechRecognizer.ERROR_CLIENT -> "ERROR_CLIENT"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "ERROR_INSUFFICIENT_PERMISSIONS"
            SpeechRecognizer.ERROR_NETWORK -> "ERROR_NETWORK"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "ERROR_NETWORK_TIMEOUT"
            SpeechRecognizer.ERROR_NO_MATCH -> "ERROR_NO_MATCH"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "ERROR_RECOGNIZER_BUSY"
            SpeechRecognizer.ERROR_SERVER -> "ERROR_SERVER"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "ERROR_SPEECH_TIMEOUT"
            12 -> "ERROR_LANGUAGE_NOT_SUPPORTED"
            13 -> "ERROR_LANGUAGE_UNAVAILABLE"
            else -> "ERROR_$code"
        }
    }

    // ---------- Audio focus helpers ----------
    private fun acquireAudioFocusIfNeeded(reason: String) {
        if (!config.requestAudioFocus) return
        if (hasAudioFocus) return
        val am = audioManager ?: return

        // This is the legacy API; if you want AudioFocusRequest (API 26+),
        // you can swap it in. Keeping this minimal and widely compatible.
        @Suppress("DEPRECATION")
        val result = am.requestAudioFocus(
            { /* ignored */ },
            AudioManager.STREAM_MUSIC,
            AudioManager.AUDIOFOCUS_GAIN_TRANSIENT
        )
        hasAudioFocus = (result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED)
        emit("ASR_AUDIO_FOCUS", mapOf("acquired" to hasAudioFocus, "reason" to reason))
    }

    private fun releaseAudioFocusIfNeeded(reason: String) {
        if (!config.requestAudioFocus) return
        if (!hasAudioFocus) return
        val am = audioManager ?: return

        @Suppress("DEPRECATION")
        am.abandonAudioFocus(null)
        hasAudioFocus = false
        emit("ASR_AUDIO_FOCUS", mapOf("released" to true, "reason" to reason))
    }
}