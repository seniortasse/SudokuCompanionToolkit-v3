package com.contextionary.sudoku

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.core.net.toUri
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * Cloud TTS for Azure: downloads MP3 to cache (off main thread), then plays with ExoPlayer.
 * Fires onStart exactly when playback begins (no early pulsing), and onDone when playback ends.
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

    fun isReady(): Boolean = subscriptionKey.isNotBlank() && region.isNotBlank()

    /**
     * Prepare (download if needed) OFF-MAIN, then play ON-MAIN.
     * onStart fires when audio is actually playing; onDone when ended or stopped.
     */
    suspend fun speakSsml(
        ssml: String,
        voiceName: String,
        localeTag: String,
        onStart: (() -> Unit)?,
        onDone: (() -> Unit)?,
        onError: ((Throwable) -> Unit)?
    ) {
        try {
            val outFile = cachedFileFor(ssml, voiceName, localeTag)
            Log.i(logTag, "Azure TTS: target file=${outFile.absolutePath}")

            // 1) Synthesis OFF-MAIN
            if (!outFile.exists()) {
                Log.i(logTag, "Azure TTS: cache miss, fetching from API…")
                withContext(Dispatchers.IO) {
                    synthToFile(ssml, voiceName, localeTag, outFile)
                }
            } else {
                Log.i(logTag, "Azure TTS: cache hit, reusing ${outFile.name}")
            }

            // 2) Playback ON-MAIN
            withContext(Dispatchers.Main) {
                playFile(outFile.toUri(), onStart, onDone, onError)
            }
        } catch (t: Throwable) {
            Log.w(logTag, "Azure speak error — will bubble to fallback", t)
            onError?.invoke(t)
        }
    }

    fun stop() {
        try {
            player?.stop()
        } catch (_: Throwable) { }
        try {
            player?.release()
        } catch (_: Throwable) { }
        player = null
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

        // NOTE: We send only SSML here (caller builds the <speak>…<voice>… block).
        val request = Request.Builder()
            .url(url)
            .addHeader("Ocp-Apim-Subscription-Key", subscriptionKey)
            // This header helps prevent 400s in some setups:
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

    private fun playFile(
        uri: Uri,
        onStart: (() -> Unit)?,
        onDone: (() -> Unit)?,
        onError: ((Throwable) -> Unit)?
    ) {
        try {
            // Clean previous
            stop()

            val p = ExoPlayer.Builder(context).build()
            player = p

            val item = MediaItem.fromUri(uri)
            p.setMediaItem(item)

            var started = false

            p.addListener(object : com.google.android.exoplayer2.Player.Listener {
                override fun onPlaybackStateChanged(state: Int) {
                    when (state) {
                        com.google.android.exoplayer2.Player.STATE_READY -> {
                            // Don’t fire onStart here; wait for isPlaying==true to avoid early bars.
                        }
                        com.google.android.exoplayer2.Player.STATE_ENDED -> {
                            Log.i(logTag, "Azure TTS: playback ended")
                            onDone?.invoke()
                        }
                        else -> Unit
                    }
                }

                override fun onIsPlayingChanged(isPlaying: Boolean) {
                    if (isPlaying && !started) {
                        started = true
                        Log.i(logTag, "Azure TTS: playback started")
                        onStart?.invoke()
                    }
                }

                override fun onPlayerError(error: com.google.android.exoplayer2.PlaybackException) {
                    Log.w(logTag, "Azure TTS: player error", error)
                    onError?.invoke(error)
                }
            })

            p.prepare()
            p.playWhenReady = true
        } catch (t: Throwable) {
            Log.w(logTag, "Azure TTS: playFile threw", t)
            onError?.invoke(t)
        }
    }

    private fun sha1(s: String): String {
        val md = MessageDigest.getInstance("SHA-1")
        val b = md.digest(s.toByteArray())
        return b.joinToString("") { "%02x".format(it) }
    }
}