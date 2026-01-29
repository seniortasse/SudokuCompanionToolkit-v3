package com.contextionary.sudoku

import android.content.Context
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.net.toUri
import com.google.android.exoplayer2.C
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.android.exoplayer2.audio.AudioAttributes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

// telemetry
import com.contextionary.sudoku.telemetry.ConversationTelemetry

/**
 * AzureCloudTtsEngine (patched):
 *
 * Goal: make it impossible for the app to "go silent" after an abnormal stop (e.g. AUDIO_FOCUS_LOST).
 *
 * Root issue:
 * - Previously, focus-loss triggered stopInternal() which STOPPED playback but did NOT call the
 *   per-utterance terminal callback (onDone/onError) that MainActivity relies on to clear
 *   isSpeaking/asrSuppressedByTts and to resume listening.
 *
 * Patch strategy:
 * 1) Store the per-utterance callbacks (onStart/onDone/onError) as active callbacks.
 * 2) When stopInternal() happens for "terminal stop" reasons (audio_focus_lost, manual_stop, etc),
 *    invoke the active onDone() exactly once on the MAIN thread.
 *    - We intentionally invoke onDone (not onError) to avoid replay/fallback loops.
 *    - We DO NOT do this for "pre_new_tts" preemption or for error sources where onError is already invoked.
 * 3) Always clear active ids + callbacks on stop, just like DONE cleanup, to prevent stale state.
 */
class AzureCloudTtsEngine(
    private val context: Context,
    private val subscriptionKey: String,
    private val region: String
) {
    private val logTag = "SudokuTTS"

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private var player: ExoPlayer? = null

    // ---- Phase 1: explicit audio focus ownership ----
    private val audioManager: AudioManager =
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    private val ttsIdCounter = AtomicInteger(0)

    @Volatile private var hasAudioFocus: Boolean = false

    @Volatile private var activeTtsId: Int? = null
    @Volatile private var activeStopTtsId: Int? = null
    var onStopped: ((ttsId: Int, source: String) -> Unit)? = null

    // ✅ NEW: capture pairing fields for telemetry correlation
    @Volatile private var activeSpeakReqId: Int? = null
    @Volatile private var activeReplyToRowId: Int? = null
    @Volatile private var activeConvTurn: Int? = null
    @Volatile private var activeVoiceName: String? = null
    @Volatile private var activeLocaleTag: String? = null

    // ✅ NEW: hold per-utterance callbacks so stopInternal can finish the speech contract
    @Volatile private var activeOnStart: (() -> Unit)? = null
    @Volatile private var activeOnDone: (() -> Unit)? = null
    @Volatile private var activeOnError: ((Throwable) -> Unit)? = null

    // prevents double-callbacks if stopInternal gets called multiple times
    private val stopOnce = java.util.concurrent.atomic.AtomicBoolean(false)

    private val mainHandler = Handler(Looper.getMainLooper())

    private val focusChangeListener = AudioManager.OnAudioFocusChangeListener { change ->
        when (change) {
            AudioManager.AUDIOFOCUS_LOSS,
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "AUDIO_FOCUS_LOST",
                        "engine" to "Azure",
                        "tts_id" to activeTtsId,
                        "speak_req_id" to activeSpeakReqId,
                        "reply_to_row_id" to activeReplyToRowId,
                        "conv_turn" to activeConvTurn,
                        "change" to change
                    )
                )
                stopInternal(source = "audio_focus_lost", ttsId = activeTtsId ?: -1)
            }
            else -> Unit
        }
    }

    private var focusRequest: AudioFocusRequest? = null

    fun isReady(): Boolean = subscriptionKey.isNotBlank() && region.isNotBlank()

    suspend fun speakSsml(
        ssml: String,
        voiceName: String,
        localeTag: String,
        speakReqId: Int? = null,
        replyToRowId: Int? = null,
        convTurn: Int? = null,
        onStart: (() -> Unit)?,
        onDone: (() -> Unit)?,
        onError: ((Throwable) -> Unit)?
    ): Int {

        // 1) Preempt previous utterance ONLY if playback pipeline exists.
        val prevId: Int? = activeTtsId
        if (prevId != null) {
            val hasLivePlayer = (player != null)
            if (hasLivePlayer) {
                // IMPORTANT: preemption is not a "terminal end" from the app POV, so do NOT
                // trigger onDone/onError here. Just stop playback quietly.
                stopInternal(source = "pre_new_tts", ttsId = prevId)
            } else {
                // ✅ Nothing playing anymore; forget stale ids
                clearActiveState()
                synchronized(this) { activeStopTtsId = prevId }
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TTS_PREEMPT_NOOP",
                        "engine" to "Azure",
                        "tts_id" to prevId
                    )
                )
            }
        }

        // 2) New utterance id + reset idempotence guard
        stopOnce.set(false)

        val ttsId = ttsIdCounter.incrementAndGet()
        activeTtsId = ttsId
        activeSpeakReqId = speakReqId
        activeReplyToRowId = replyToRowId
        activeConvTurn = convTurn
        activeVoiceName = voiceName
        activeLocaleTag = localeTag

        // store callbacks for stopInternal
        activeOnStart = onStart
        activeOnDone = onDone
        activeOnError = onError

        synchronized(this) { activeStopTtsId = null }

        try {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_FETCH_BEGIN",
                    "engine" to "Azure",
                    "tts_id" to ttsId,
                    "speak_req_id" to speakReqId,
                    "reply_to_row_id" to replyToRowId,
                    "conv_turn" to convTurn,
                    "voice" to voiceName,
                    "locale" to localeTag
                )
            )

            val outFile = cachedFileFor(ssml, voiceName, localeTag)
            Log.i(logTag, "Azure TTS: target file=${outFile.absolutePath}")

            if (!outFile.exists()) {
                Log.i(logTag, "Azure TTS: cache miss, fetching from API…")
                withContext(Dispatchers.IO) { synthToFile(ssml, voiceName, localeTag, outFile) }

                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TTS_FETCH_OK",
                        "engine" to "Azure",
                        "tts_id" to ttsId,
                        "speak_req_id" to speakReqId,
                        "reply_to_row_id" to replyToRowId,
                        "conv_turn" to convTurn,
                        "bytes" to outFile.length()
                    )
                )
            } else {
                Log.i(logTag, "Azure TTS: cache hit, reusing ${outFile.name}")
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TTS_FETCH_CACHE_HIT",
                        "engine" to "Azure",
                        "tts_id" to ttsId,
                        "speak_req_id" to speakReqId,
                        "reply_to_row_id" to replyToRowId,
                        "conv_turn" to convTurn,
                        "bytes" to outFile.length()
                    )
                )
            }

            withContext(Dispatchers.Main) {
                playFile(
                    uri = outFile.toUri(),
                    onStart = onStart,
                    onDone = onDone,
                    onError = onError,
                    voiceName = voiceName,
                    localeTag = localeTag,
                    ttsId = ttsId,
                    speakReqId = speakReqId,
                    replyToRowId = replyToRowId,
                    convTurn = convTurn
                )
            }

        } catch (t: Throwable) {
            Log.w(logTag, "Azure speak error", t)

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_ERROR",
                    "engine" to "Azure",
                    "tts_id" to ttsId,
                    "speak_req_id" to speakReqId,
                    "reply_to_row_id" to replyToRowId,
                    "conv_turn" to convTurn,
                    "message" to (t.message ?: t.toString())
                )
            )

            // stop playback, but do NOT trigger onDone here; caller will handle onError
            stopInternal(source = "speakssml_exception", ttsId = ttsId)
            onError?.invoke(t)
        }

        return ttsId
    }

    fun stop() {
        stopInternal(source = "manual_stop", ttsId = activeTtsId ?: -1)
    }

    // ----------------------- Internals -----------------------

    private fun cachedFileFor(ssml: String, voice: String, locale: String): File {
        val hash = sha1("$voice|$locale|$ssml")
        val dir = File(context.cacheDir, "sudo_tts")
        if (!dir.exists()) dir.mkdirs()
        return File(dir, "$hash.mp3")
    }

    private suspend fun synthToFile(
        ssml: String,
        voiceName: String,
        localeTag: String,
        outFile: File
    ) {
        val url = "https://$region.tts.speech.microsoft.com/cognitiveservices/v1"
        val mediaType = "application/ssml+xml".toMediaType()

        val request = Request.Builder()
            .url(url)
            .addHeader("Ocp-Apim-Subscription-Key", subscriptionKey)
            .addHeader("Ocp-Apim-Subscription-Region", region)
            .addHeader("X-Microsoft-OutputFormat", "audio-16khz-128kbitrate-mono-mp3")
            .addHeader("User-Agent", "SudokuCompanion")
            .post(ssml.toRequestBody(mediaType))
            .build()

        client.newCall(request).execute().use { r ->
            if (!r.isSuccessful) {
                val bodyText = try { r.body?.string().orEmpty() } catch (_: Throwable) { "" }
                Log.w(logTag, "Azure TTS HTTP ${r.code} — $bodyText")
                throw IllegalStateException("Azure TTS HTTP ${r.code} — $bodyText")
            }
            val bytes = r.body?.bytes() ?: throw IllegalStateException("Azure TTS empty body")
            Log.i(logTag, "Azure TTS: HTTP OK, received ${bytes.size} bytes")
            outFile.outputStream().use { it.write(bytes) }
        }
    }

    private fun requestAudioFocusOrLog(
        ttsId: Int,
        speakReqId: Int?,
        replyToRowId: Int?,
        convTurn: Int?
    ): Boolean {
        ConversationTelemetry.emit(
            mapOf(
                "type" to "AUDIO_FOCUS_REQUEST",
                "engine" to "Azure",
                "tts_id" to ttsId,
                "speak_req_id" to speakReqId,
                "reply_to_row_id" to replyToRowId,
                "conv_turn" to convTurn,
                "usage" to "USAGE_ASSISTANT",
                "content_type" to "CONTENT_TYPE_SPEECH"
            )
        )

        val granted = if (Build.VERSION.SDK_INT >= 26) {
            val attrs = android.media.AudioAttributes.Builder()
                .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
                .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()

            val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
                .setAudioAttributes(attrs)
                .setOnAudioFocusChangeListener(focusChangeListener)
                .setAcceptsDelayedFocusGain(false)
                .setWillPauseWhenDucked(false)
                .build()

            focusRequest = req
            audioManager.requestAudioFocus(req) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        } else {
            @Suppress("DEPRECATION")
            audioManager.requestAudioFocus(
                focusChangeListener,
                AudioManager.STREAM_MUSIC,
                AudioManager.AUDIOFOCUS_GAIN_TRANSIENT
            ) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        }

        hasAudioFocus = granted

        ConversationTelemetry.emit(
            mapOf(
                "type" to (if (granted) "AUDIO_FOCUS_GRANTED" else "AUDIO_FOCUS_DENIED"),
                "engine" to "Azure",
                "tts_id" to ttsId,
                "speak_req_id" to speakReqId,
                "reply_to_row_id" to replyToRowId,
                "conv_turn" to convTurn
            )
        )

        return granted
    }

    private fun abandonAudioFocusIfHeld(ttsId: Int) {
        if (!hasAudioFocus) return
        hasAudioFocus = false

        if (Build.VERSION.SDK_INT >= 26) {
            focusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
            focusRequest = null
        } else {
            @Suppress("DEPRECATION")
            audioManager.abandonAudioFocus(focusChangeListener)
        }

        ConversationTelemetry.emit(
            mapOf(
                "type" to "AUDIO_FOCUS_ABANDON",
                "engine" to "Azure",
                "tts_id" to ttsId,
                "speak_req_id" to activeSpeakReqId,
                "reply_to_row_id" to activeReplyToRowId,
                "conv_turn" to activeConvTurn
            )
        )
    }

    private fun clearActiveState() {
        activeTtsId = null
        activeSpeakReqId = null
        activeReplyToRowId = null
        activeConvTurn = null
        activeVoiceName = null
        activeLocaleTag = null

        activeOnStart = null
        activeOnDone = null
        activeOnError = null
    }

    private fun shouldInvokeTerminalCallbackOnStop(source: String): Boolean {
        // We should NOT invoke onDone for preemption, otherwise you might start listening
        // while a new TTS is about to begin.
        if (source == "pre_new_tts") return false

        // For these sources, the caller already invokes onError explicitly after stopInternal.
        if (source == "player_error") return false
        if (source == "playfile_exception") return false
        if (source == "speakssml_exception") return false

        // Otherwise, treat it as a terminal stop and "finish" the utterance so the app can resume ASR.
        return true
    }

    private fun playFile(
        uri: Uri,
        onStart: (() -> Unit)?,
        onDone: (() -> Unit)?,
        onError: ((Throwable) -> Unit)?,
        voiceName: String,
        localeTag: String,
        ttsId: Int,
        speakReqId: Int?,
        replyToRowId: Int?,
        convTurn: Int?
    ) {
        try {
            requestAudioFocusOrLog(ttsId, speakReqId, replyToRowId, convTurn)

            val p = ExoPlayer.Builder(context).build()
            player = p

            val exoAttrs = AudioAttributes.Builder()
                .setUsage(C.USAGE_ASSISTANT)
                .setContentType(C.CONTENT_TYPE_SPEECH)
                .build()

            p.setAudioAttributes(exoAttrs, /* handleAudioFocus = */ false)

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_AUDIO_ATTR",
                    "engine" to "Azure",
                    "tts_id" to ttsId,
                    "speak_req_id" to speakReqId,
                    "reply_to_row_id" to replyToRowId,
                    "conv_turn" to convTurn,
                    "usage" to "USAGE_ASSISTANT",
                    "content_type" to "CONTENT_TYPE_SPEECH",
                    "handle_focus" to false
                )
            )

            p.volume = 1f
            p.setMediaItem(MediaItem.fromUri(uri))

            var started = false
            var terminalEmitted = false
            var playbackStartMs: Long = 0L

            fun isCurrentPlayer(): Boolean = (player === p)

            fun cleanupAfterDone() {
                try { p.stop() } catch (_: Throwable) {}
                try { p.release() } catch (_: Throwable) {}

                if (isCurrentPlayer()) {
                    player = null
                    clearActiveState()
                }
                abandonAudioFocusIfHeld(ttsId)
            }

            p.addListener(object : com.google.android.exoplayer2.Player.Listener {

                override fun onIsPlayingChanged(isPlaying: Boolean) {
                    if (!isCurrentPlayer()) return
                    if (terminalEmitted) return

                    if (isPlaying && !started) {
                        started = true
                        playbackStartMs = System.currentTimeMillis()

                        Log.i(logTag, "Azure TTS: playback started (tts_id=$ttsId)")
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TTS_START",
                                "engine" to "Azure",
                                "tts_id" to ttsId,
                                "speak_req_id" to speakReqId,
                                "reply_to_row_id" to replyToRowId,
                                "conv_turn" to convTurn,
                                "voice" to voiceName,
                                "locale" to localeTag
                            )
                        )
                        onStart?.invoke()
                    }
                }

                override fun onPlaybackStateChanged(state: Int) {
                    if (!isCurrentPlayer()) return
                    if (terminalEmitted) return

                    if (state == com.google.android.exoplayer2.Player.STATE_ENDED) {
                        terminalEmitted = true

                        val durationMs =
                            if (playbackStartMs > 0L) System.currentTimeMillis() - playbackStartMs
                            else null

                        Log.i(logTag, "Azure TTS: playback ended (tts_id=$ttsId, durationMs=$durationMs)")
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TTS_DONE",
                                "engine" to "Azure",
                                "tts_id" to ttsId,
                                "speak_req_id" to speakReqId,
                                "reply_to_row_id" to replyToRowId,
                                "conv_turn" to convTurn,
                                "duration_ms" to (durationMs ?: -1L),
                                "voice" to voiceName,
                                "locale" to localeTag
                            )
                        )

                        cleanupAfterDone()
                        onDone?.invoke()
                    }
                }

                override fun onPlayerError(error: com.google.android.exoplayer2.PlaybackException) {
                    if (!isCurrentPlayer()) return
                    if (terminalEmitted) return
                    terminalEmitted = true

                    Log.w(logTag, "Azure TTS: player error (tts_id=$ttsId)", error)

                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "TTS_ERROR",
                            "engine" to "Azure",
                            "tts_id" to ttsId,
                            "speak_req_id" to speakReqId,
                            "reply_to_row_id" to replyToRowId,
                            "conv_turn" to convTurn,
                            "message" to (error.message ?: error.toString())
                        )
                    )

                    // Stop and clean. Caller will invoke onError.
                    stopInternal(source = "player_error", ttsId = ttsId)
                    onError?.invoke(error)
                }
            })

            p.prepare()
            p.playWhenReady = true

        } catch (t: Throwable) {
            Log.w(logTag, "Azure TTS: playFile threw (tts_id=$ttsId)", t)

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_ERROR",
                    "engine" to "Azure",
                    "tts_id" to ttsId,
                    "speak_req_id" to speakReqId,
                    "reply_to_row_id" to replyToRowId,
                    "conv_turn" to convTurn,
                    "message" to (t.message ?: t.toString())
                )
            )

            // Stop playback, but caller handles onError.
            stopInternal(source = "playfile_exception", ttsId = ttsId)
            onError?.invoke(t)
        }
    }

    private fun stopInternal(
        source: String,
        ttsId: Int?
    ) {
        val id: Int = ttsId ?: (activeTtsId ?: -1)

        if (!stopOnce.compareAndSet(false, true)) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_STOP_DUPLICATE_IGNORED",
                    "engine" to "Azure",
                    "tts_id" to id,
                    "speak_req_id" to activeSpeakReqId,
                    "reply_to_row_id" to activeReplyToRowId,
                    "conv_turn" to activeConvTurn,
                    "source" to source
                )
            )
            return
        }

        synchronized(this) { activeStopTtsId = id }

        // Stop/release player (safe)
        try { player?.stop() } catch (_: Throwable) {}
        try { player?.release() } catch (_: Throwable) {}
        player = null

        ConversationTelemetry.emit(
            mapOf(
                "type" to "TTS_STOP",
                "engine" to "Azure",
                "tts_id" to id,
                "speak_req_id" to activeSpeakReqId,
                "reply_to_row_id" to activeReplyToRowId,
                "conv_turn" to activeConvTurn,
                "source" to source
            )
        )

        abandonAudioFocusIfHeld(id)

        // Capture callbacks before clearing state
        val cbDone = activeOnDone
        val cbError = activeOnError
        val shouldInvokeTerminal = shouldInvokeTerminalCallbackOnStop(source)

        // Always clear state so we don't appear "stuck speaking"
        clearActiveState()

        // Legacy notification hook (if MainActivity uses it)
        try {
            onStopped?.invoke(id, source)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_STOP_NOTIFIED",
                    "engine" to "Azure",
                    "tts_id" to id,
                    "source" to source,
                    "has_onStopped" to true
                )
            )
        } catch (t: Throwable) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_STOP_NOTIFY_ERROR",
                    "engine" to "Azure",
                    "tts_id" to id,
                    "source" to source,
                    "message" to (t.message ?: t.toString())
                )
            )
        }

        // 🔥 CRITICAL PATCH: If this stop is terminal (focus loss, manual stop, etc),
        // invoke a terminal callback so the app runs its finish logic and resumes listening.
        if (shouldInvokeTerminal) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_STOP_TERMINAL_CALLBACK",
                    "engine" to "Azure",
                    "tts_id" to id,
                    "source" to source,
                    "invokes" to (if (cbDone != null) "onDone" else if (cbError != null) "onError" else "none")
                )
            )

            // Prefer onDone to avoid "fallback & repeat speech" behavior.
            if (cbDone != null) {
                mainHandler.post { cbDone.invoke() }
            } else if (cbError != null) {
                // Worst-case fallback: surface as error so caller can still unstick.
                val ex = IllegalStateException("Azure TTS stopped: $source")
                mainHandler.post { cbError.invoke(ex) }
            }
        }
    }

    private fun sha1(s: String): String {
        val md = MessageDigest.getInstance("SHA-1")
        val b = md.digest(s.toByteArray())
        return b.joinToString("") { "%02x".format(it) }
    }
}