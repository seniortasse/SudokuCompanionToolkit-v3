package com.contextionary.sudoku

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat
import java.util.Locale

import com.contextionary.sudoku.telemetry.ConversationTelemetry

/**
 * SudoASR — SpeechRecognizer wrapper.
 *
 * Key design (robust on flaky devices):
 * - Every start() creates a fresh SpeechRecognizer + fresh RecognitionListener capturing a serial.
 * - We do NOT block onResults based on "stopRequested" (many devices deliver results after stop()).
 * - Phase 0 invariant: never forward callbacks while suppressed/isSpeaking.
 * - TurnController owns when to call start(); SudoASR does not auto-restart.
 */
class SudoASR(
    private val context: Context,
    private val config: Config = Config()
) {

    data class Config(
        val partialResults: Boolean = true,
        val maxResults: Int = 1,
        val completeSilenceMs: Long = 900L,
        val possiblyCompleteSilenceMs: Long = 700L,
        val minInputLengthMs: Long = 400L,

        val preferredLocaleTag: String = "",
        val restartDelayMs: Long = 300L,
        val minGapBetweenSessionsMs: Long = 250L
    )

    interface Listener {
        fun onReady(localeTag: String) {}
        fun onBegin() {}
        fun onRmsChanged(rmsDb: Float) {}
        fun onPartial(text: String) {}
        fun onHeard(text: String, confidence: Float?) {}
        fun onFinal(rowId: Int, text: String, confidence: Float?, reason: String) {}
        fun onEnd() {}
        fun onError(code: Int) {}
    }

    var listener: Listener? = null

    // Recognizer + intent rebuilt per start() to avoid stale callbacks breaking sessions.
    private var recognizer: SpeechRecognizer? = null
    private var intent: Intent? = null

    // External gates/state
    private var isSpeaking: Boolean = false
    private var suppressed: Boolean = false
    private var asrActive: Boolean = false

    // Telemetry correlation: row_id allocated at startListening()
    private var activeRowId: Int = -1

    private var lastStartMs: Long = 0L
    private var lastPartial: String = ""

    @Volatile private var intentLocaleTag: String = ""

    // Session serial (mainly for logs/telemetry)
    @Volatile private var startSerial: Int = 0

    @Volatile private var conversationLocaleTag: String? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private var pendingRestart: Runnable? = null

    // ---- Robust locale fallback state ----
    @Volatile private var lastLanguageUnavailable: Boolean = false   // code 13
    @Volatile private var lastLanguageNotSupported: Boolean = false  // code 12
    @Volatile private var lastFailedLocaleTag: String? = null

    // TurnController should implement this (or you can just emit telemetry and let it decide)
    var onLocaleFallbackSuggested: ((fallbackLocaleTag: String) -> Unit)? = null

    private var langUnavailableRetryArmed = false

    private val logTag = "SudoASR"

    companion object {
        private const val DEFAULT_LOCALE_TAG = "en-US"

        // ✅ Make this callable from MainActivity: SudoASR.errorName(code)
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

    fun isActive(): Boolean = asrActive

    fun prewarm() {
        // Prewarm just validates availability + locale; does not create recognizer yet
        val tag = effectiveLocale().toLanguageTag()
        intentLocaleTag = tag
        Log.i(logTag, "Prewarm OK (intentLocaleTag=$intentLocaleTag)")
    }

    fun release() = destroy()

    fun setConversationLocaleTag(tag: String?) {
        conversationLocaleTag = tag?.trim()?.ifEmpty { null }
        // next start() will rebuild with new locale
        intentLocaleTag = ""
    }

    fun setSpeaking(speaking: Boolean) {
        // speaking=true  => hard suppress + cancel listening immediately
        // speaking=false => lift suppress immediately; Conductor decides when to start()
        isSpeaking = speaking

        if (speaking) {
            suppressed = true

            pendingRestart?.let { mainHandler.removeCallbacks(it) }
            pendingRestart = null

            // While TTS speaks, we must not be listening.
            cancel()

            Log.i(logTag, "SUPPRESS_ON (tts speaking)")
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "ASR_SUPPRESS_FOR_TTS",
                    "reason" to "tts_speaking"
                )
            )
        } else {
            // IMPORTANT: lift suppression here.
            // We do NOT start ASR here — Turn Conductor will call start() when legal.
            suppressed = false

            pendingRestart?.let { mainHandler.removeCallbacks(it) }
            pendingRestart = null

            Log.i(logTag, "SUPPRESS_OFF (tts finished; Conductor may start ASR)")
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "ASR_SUPPRESS_LIFTED",
                    "source" to "setSpeaking(false)"
                )
            )
        }
    }

    @Deprecated(
        message = "Contract violation: workers (TTS) must not control ASR. Do not use.",
        level = DeprecationLevel.ERROR
    )
    fun asAsrController(): Any {
        // Intentionally impossible to use. Remove all call sites (MainActivity, AzureCloudTtsEngine wiring).
        error("Do not use asAsrController(): Turn Conductor is the only ASR authority.")
    }

    fun start() {
        if (suppressed) {
            Log.i(logTag, "LISTEN_SKIP reason=suppressed_for_tts")
            ConversationTelemetry.emit(mapOf("type" to "ASR_LISTEN_SKIP", "reason" to "suppressed_for_tts"))
            return
        }
        if (isSpeaking) {
            Log.i(logTag, "LISTEN_SKIP reason=still_speaking")
            ConversationTelemetry.emit(mapOf("type" to "ASR_LISTEN_SKIP", "reason" to "still_speaking"))
            return
        }
        if (asrActive) {
            Log.i(logTag, "LISTEN_SKIP reason=already_active")
            ConversationTelemetry.emit(mapOf("type" to "ASR_LISTEN_SKIP", "reason" to "already_active"))
            return
        }

        val now = System.currentTimeMillis()
        if (now - lastStartMs < config.minGapBetweenSessionsMs) {
            Log.i(logTag, "LISTEN_SKIP reason=debounce (${now - lastStartMs}ms since last)")
            ConversationTelemetry.emit(mapOf("type" to "ASR_LISTEN_SKIP", "reason" to "debounce"))
            return
        }

        val ok = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        if (!ok) {
            Log.w(logTag, "LISTEN_DENY reason=missing_record_audio_permission")
            Toast.makeText(context, "Enable microphone permission to talk to Sudo.", Toast.LENGTH_SHORT).show()
            ConversationTelemetry.emit(mapOf("type" to "ASR_LISTEN_DENY", "reason" to "missing_record_audio_permission"))
            return
        }

        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            Log.w(logTag, "LISTEN_DENY reason=recognition_not_available")
            Toast.makeText(context, "Speech recognition not available on this device.", Toast.LENGTH_SHORT).show()
            ConversationTelemetry.emit(mapOf("type" to "ASR_LISTEN_DENY", "reason" to "recognition_not_available"))
            return
        }

        // Fresh recognizer per start() (prevents stale callbacks / ERROR_CLIENT weirdness from poisoning next sessions)
        destroyRecognizerOnly()

        // ---- Locale selection with robust fallback ladder ----
        val requestedLocale: Locale = effectiveLocale()
        val deviceDefaultLocale: Locale = Locale.getDefault()

        val requestedTag = requestedLocale.toLanguageTag()
        val deviceDefaultTag = deviceDefaultLocale.toLanguageTag()

        val prevUnavailable = lastLanguageUnavailable
        val prevNotSupported = lastLanguageNotSupported

        val chosenTag = pickSupportedLocaleTag(
            requestedTag = requestedTag,
            deviceDefaultTag = deviceDefaultTag,
            lastFailed = lastFailedLocaleTag
        )

        intentLocaleTag = chosenTag

        ConversationTelemetry.emit(
            mapOf(
                "type" to "ASR_LOCALE_CHOSEN",
                "requested" to requestedTag,
                "chosen" to chosenTag,
                "device_default" to deviceDefaultTag,
                "prev_lang_unavailable" to prevUnavailable,
                "prev_lang_not_supported" to prevNotSupported,
                "last_failed" to (lastFailedLocaleTag ?: "")
            )
        )

        // Clear "retry suggestion" arm per new row
        langUnavailableRetryArmed = false
        // -----------------------------------------------------

        val serial = ++startSerial
        lastStartMs = now
        lastPartial = ""
        asrActive = true

        recognizer = SpeechRecognizer.createSpeechRecognizer(context)
        intent = buildIntent(chosenTag)

        recognizer!!.setRecognitionListener(makeListener(serial, chosenTag))

        Log.i(logTag, "LISTEN_START locale=$chosenTag serial=$serial")
        activeRowId = ConversationTelemetry.asrRowStart("locale" to chosenTag, "serial" to serial)

        try {
            recognizer?.startListening(intent)
        } catch (t: Throwable) {
            asrActive = false
            Log.e(logTag, "LISTEN_ERROR ${t.message}", t)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "ASR_START_THROW",
                    "message" to (t.message ?: t.toString()),
                    "serial" to serial
                )
            )
        }
    }

    fun stop() {
        // Important: do NOT create a latch that blocks onResults (some devices still deliver results after stop()).
        try { recognizer?.stopListening() } catch (_: Throwable) {}

        if (asrActive) {
            Log.i(logTag, "LISTEN_STOP_REQUESTED")
            ConversationTelemetry.asrRowEvent("ASR_STOP_REQUESTED", "source" to "manual_stop")
        }

        asrActive = false
        lastPartial = ""
    }

    fun cancel() {
        // Cancel means: drop everything.
        try { recognizer?.cancel() } catch (_: Throwable) {}

        if (asrActive) {
            Log.i(logTag, "LISTEN_CANCEL_REQUESTED")
            ConversationTelemetry.asrRowEvent("ASR_CANCEL_REQUESTED", "source" to "manual_cancel")
        }

        asrActive = false
        lastPartial = ""
    }

    fun destroy() {
        cancel()
        destroyRecognizerOnly()

        intent = null
        intentLocaleTag = ""
        suppressed = false

        pendingRestart?.let { mainHandler.removeCallbacks(it) }
        pendingRestart = null
    }

    // ---------------- internals ----------------

    private fun callbacksAllowed(): Boolean {
        if (suppressed) return false
        if (isSpeaking) return false
        return true
    }

    private fun buildIntent(localeTag: String): Intent =
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, localeTag)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, localeTag)

            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, config.partialResults)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, config.maxResults)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, config.completeSilenceMs)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, config.possiblyCompleteSilenceMs)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, config.minInputLengthMs)
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
        }

    private fun makeListener(serial: Int, localeTag: String): RecognitionListener =
        object : RecognitionListener {

            // After an ASR row ends (onError or accepted onResults), ignore any late callbacks.
            @Volatile private var ended: Boolean = false

            // Some OEM recognizers can deliver an empty onResults() then a real one shortly after.
            // We wait a short grace window before treating empty as NO_MATCH.
            private var pendingEmptyFinalize: Runnable? = null
            private var pendingEmptyAtMs: Long = 0L
            private val emptyFinalGraceMs: Long = 250L

            private fun allowed(): Boolean {
                if (ended) return false
                return callbacksAllowed()
            }

            private fun endRowOnce(kind: String) {
                if (!ended) {
                    ended = true
                }
            }

            private fun cancelPendingEmptyFinalize(reason: String) {
                pendingEmptyFinalize?.let { mainHandler.removeCallbacks(it) }
                pendingEmptyFinalize = null
                pendingEmptyAtMs = 0L

                // Optional breadcrumb: only emit if we were actually waiting
                if (reason.isNotBlank()) {
                    ConversationTelemetry.asrRowEvent(
                        "ASR_EMPTY_FINAL_CANCELLED",
                        "serial" to serial,
                        "reason" to reason
                    )
                }
            }

            private fun clearLocaleFailureLatches() {
                lastLanguageUnavailable = false
                lastLanguageNotSupported = false
                lastFailedLocaleTag = null
            }

            override fun onReadyForSpeech(params: Bundle?) {
                if (!allowed()) {
                    Log.i(logTag, "READY_IGNORED serial=$serial ended=$ended suppressed=$suppressed isSpeaking=$isSpeaking")
                    return
                }
                asrActive = true
                listener?.onReady(localeTag)
                Log.i(logTag, "READY locale=$localeTag serial=$serial")
                ConversationTelemetry.asrRowEvent("ASR_READY", "locale" to localeTag, "serial" to serial)
            }

            override fun onBeginningOfSpeech() {
                if (!allowed()) {
                    Log.i(logTag, "BEGIN_IGNORED serial=$serial ended=$ended suppressed=$suppressed isSpeaking=$isSpeaking")
                    return
                }
                // If we got to actual speech, recognizer is functioning — clear latches.
                clearLocaleFailureLatches()

                asrActive = true
                listener?.onBegin()
                Log.i(logTag, "BEGIN serial=$serial")
                ConversationTelemetry.asrRowEvent("ASR_BEGIN", "serial" to serial)
            }

            override fun onRmsChanged(rmsdB: Float) {
                if (!allowed()) return
                listener?.onRmsChanged(rmsdB)
                ConversationTelemetry.asrRowEvent("ASR_RMS", "rms_db" to rmsdB, "serial" to serial)
            }

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                if (!allowed()) {
                    Log.i(logTag, "END_IGNORED serial=$serial ended=$ended suppressed=$suppressed isSpeaking=$isSpeaking")
                    return
                }
                listener?.onEnd()
                Log.i(logTag, "END serial=$serial")
                ConversationTelemetry.asrRowEvent("ASR_END", "serial" to serial)
            }

            override fun onError(error: Int) {
                val name = SudoASR.errorName(error)

                if (ended) {
                    Log.i(logTag, "ERROR_IGNORED late code=$error ($name) serial=$serial")
                    return
                }
                if (!callbacksAllowed()) {
                    Log.i(
                        logTag,
                        "ERROR_IGNORED gate code=$error ($name) serial=$serial suppressed=$suppressed isSpeaking=$isSpeaking"
                    )
                    return
                }

                // If we were waiting on an empty final, cancel that path.
                cancelPendingEmptyFinalize(reason = "onError:$name")

                // Latch locale failures for *next* start()
                when (error) {
                    SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE, 13 -> {
                        lastLanguageUnavailable = true
                        lastFailedLocaleTag = localeTag

                        val fallback = Locale.getDefault().toLanguageTag()
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "ASR_LANG_UNAVAILABLE_LATCHED",
                                "serial" to serial,
                                "locale" to localeTag,
                                "fallback" to fallback
                            )
                        )

                        if (!langUnavailableRetryArmed) {
                            langUnavailableRetryArmed = true
                            ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "ASR_LOCALE_FALLBACK_SUGGESTED",
                                    "from" to localeTag,
                                    "to" to fallback,
                                    "serial" to serial
                                )
                            )
                            onLocaleFallbackSuggested?.invoke(fallback)
                        }
                    }

                    12 -> {
                        lastLanguageNotSupported = true
                        lastFailedLocaleTag = localeTag
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "ASR_LANG_NOT_SUPPORTED_LATCHED",
                                "serial" to serial,
                                "locale" to localeTag
                            )
                        )
                    }
                }

                endRowOnce("error")

                asrActive = false
                lastPartial = ""

                Log.w(logTag, "ERROR code=$error ($name) serial=$serial")
                listener?.onError(error)

                ConversationTelemetry.asrRowEnd(
                    "ASR_ERROR",
                    "code" to error,
                    "name" to name,
                    "serial" to serial
                )
            }

            override fun onPartialResults(partialResults: Bundle?) {
                if (!allowed()) return

                val parts = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!parts.isNullOrEmpty()) {
                    val text = parts.first().orEmpty()
                    if (text != lastPartial) {
                        lastPartial = text
                        logHeard("PARTIAL", text, null)
                        listener?.onPartial(text)
                        ConversationTelemetry.asrRowEvent("ASR_PARTIAL", "text" to text, "serial" to serial)
                    }
                }
            }

            override fun onResults(results: Bundle?) {
                if (ended) {
                    Log.i(logTag, "RESULTS_IGNORED late serial=$serial")
                    return
                }
                if (!callbacksAllowed()) {
                    Log.i(
                        logTag,
                        "RESULTS_IGNORED gate serial=$serial suppressed=$suppressed isSpeaking=$isSpeaking"
                    )
                    return
                }

                // If we got results, recognizer is functioning — clear latches.
                lastLanguageUnavailable = false
                lastLanguageNotSupported = false
                lastFailedLocaleTag = null

                // End the active ASR row (idempotent).
                endRowOnce("results")

                // Capture partial BEFORE clearing; we may need it if final is empty.
                val partialCandidate = lastPartial.trim()

                asrActive = false
                lastPartial = ""

                val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val scores = results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)

                val finalCandidate = list?.firstOrNull()?.trim().orEmpty()
                val conf = scores?.firstOrNull()

                // Prefer final, but salvage from partial if final is empty (common on some devices).
                val text = when {
                    finalCandidate.isNotBlank() -> finalCandidate
                    partialCandidate.isNotBlank() -> partialCandidate
                    else -> ""
                }

                if (text.isBlank()) {
                    // ✅ Drop empty finals on the floor (no NO_MATCH escalation).
                    Log.i(logTag, "HEARD: (final empty) serial=$serial → dropped")
                    ConversationTelemetry.asrRowEvent("ASR_FINAL_EMPTY", "serial" to serial)

                    ConversationTelemetry.asrRowEnd(
                        "ASR_STOP",
                        "reason" to "final_empty_dropped",
                        "serial" to serial
                    )
                    return
                }

                // Normal final (or salvaged from partial)
                val reason = if (finalCandidate.isNotBlank()) "final_results" else "final_from_partial"

                logHeard("HEARD", text, conf)
                listener?.onHeard(text, conf)
                listener?.onFinal(activeRowId, text, conf, reason)

                ConversationTelemetry.asrRowEvent(
                    "ASR_FINAL",
                    "text" to text,
                    "confidence" to (conf ?: null),
                    "serial" to serial,
                    "final_reason" to reason
                )

                ConversationTelemetry.asrRowEnd(
                    "ASR_STOP",
                    "reason" to reason,
                    "final_text" to text,
                    "final_confidence" to (conf ?: null),
                    "serial" to serial
                )
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        }




    private fun destroyRecognizerOnly() {
        try { recognizer?.setRecognitionListener(null) } catch (_: Throwable) {}
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
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

        runCatching { Locale.forLanguageTag(DEFAULT_LOCALE_TAG).takeIf { it.language.isNotBlank() } }
            .getOrNull()?.let { return it }

        return Locale.getDefault()
    }

    /**
     * Pick a locale tag for the next start() with a robust fallback ladder.
     *
     * We cannot reliably query “supported locales” across OEM recognizers,
     * so we:
     * - react to real failures (12/13) by trying different tags next time,
     * - avoid repeating the exact same failing tag,
     * - provide a sane English fallback set.
     */
    private fun pickSupportedLocaleTag(
        requestedTag: String,
        deviceDefaultTag: String,
        lastFailed: String?
    ): String {
        // If no previous locale failure, keep requested
        if (!lastLanguageUnavailable && !lastLanguageNotSupported) return requestedTag

        val candidates = LinkedHashSet<String>()

        // 1) Device default (full)
        candidates.add(deviceDefaultTag)

        // 2) Device default language-only (e.g., "en")
        runCatching { Locale.forLanguageTag(deviceDefaultTag).language }
            .getOrNull()?.takeIf { it.isNotBlank() }?.let { candidates.add(it) }

        // 3) Safe English fallbacks
        candidates.add("en-US")
        candidates.add("en-GB")

        // 4) Requested language-only (if requested had region)
        runCatching { Locale.forLanguageTag(requestedTag).language }
            .getOrNull()?.takeIf { it.isNotBlank() }?.let { candidates.add(it) }

        // 5) Finally, requestedTag itself (in case the failure was transient)
        candidates.add(requestedTag)

        val cleaned = candidates.filter { it.isNotBlank() }

        // avoid repeating the last failing tag if possible
        val chosen = cleaned.firstOrNull { it != lastFailed } ?: requestedTag
        return chosen
    }

    private fun logHeard(kind: String, text: String, confidence: Float?) {
        val base = "$kind: \"${text.replace("\n", " ")}\""
        if (confidence != null) {
            Log.i(logTag, "$base (conf=${"%.2f".format(confidence)})")
        } else {
            Log.i(logTag, base)
        }
    }
}