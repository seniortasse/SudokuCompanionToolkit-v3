package com.contextionary.sudoku.telemetry

import android.content.Context
import android.os.Build
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import android.os.Process
import android.os.SystemClock
import java.util.concurrent.atomic.AtomicLong
import java.security.MessageDigest

/**
 * ConversationTelemetry — tiny, low-overhead JSONL event sink.
 *
 * - Call ConversationTelemetry.init(context) once at app start.
 * - Use emit(mapOf(...)) or emitKv("TYPE", "key" to value, ...)
 * - Writes JSON lines into app-private storage: files/telemetry/<day>/events_<session>.jsonl
 * - Thread-safe, non-blocking (producer queue + background writer).
 * - Optional Logcat echo for quick inspection.
 *
 * Extended with:
 * - ASR "row" tracking helpers (asrRowStart / asrRowEvent / asrRowEnd)
 *   so each listening attempt gets a row_id and a synthesized
 *   CONVTEL_ASR_ROW_EVAL summary event.
 *
 * Additional:
 * - ASSISTANT_SAY capture (capped preview + original length)
 *   so telemetry becomes a single source of truth for voice I/O analysis.
 *
 * Patch 0 additions:
 * - emitPolicyTrace(tag, data): standard POLICY_TRACE event.
 * - Multi-line string safety: sanitizeStringsForJsonl() ensures JSONL lines stay 1-line even
 *   if payload values contain newlines (e.g., prompts, long context blocks).
 */
object ConversationTelemetry {

    private const val LOG_TAG = "ConvTel"
    private const val MAX_BYTES_PER_FILE = 2_000_000L // ~2 MB per file

    /**
     * Option 2 (your choice): store a capped assistant text preview + full length.
     * This keeps JSONL small and avoids giant lines.
     */
    private const val ASSISTANT_TEXT_CAP_CHARS = 500
    private const val USER_TEXT_CAP_CHARS = 500

    /**
     * Minimum duration (ms) for an ASR row to be considered "normal".
     * Shorter bursts are classified as abnormal micro-bursts.
     */
    private const val MIN_ROW_DURATION_MS = 400L

    // Writer thread state
    private val started = AtomicBoolean(false)
    private val queue = LinkedBlockingQueue<String>(4096)
    private var writerThread: Thread? = null

    // App / session
    @Volatile private var appContext: Context? = null
    @Volatile private var sessionId: String = UUID.randomUUID().toString().substring(0, 8)
    @Volatile private var userId: String? = null
    @Volatile private var logcatEcho: Boolean = true

    // File rolling
    private var currentDayDir: File? = null
    private var fileWriter: FileWriter? = null
    private var byteCount: Long = 0L

    // Useful static bits
    private val dayFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)
    private val isoFmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US)

    private val telemetryId: String = UUID.randomUUID().toString().substring(0, 8)
    private val seq = AtomicLong(0L)

    private val policySeq = java.util.concurrent.atomic.AtomicLong(0)

    fun nextPolicyReqSeq(sessionId: String, turnId: Long): Long {
        // If you want per-session sequencing, key by sessionId; but global is fine for audit.
        return policySeq.incrementAndGet()
    }

    // ---------------------- Assistant speech capture ----------------------

    private data class CappedText(
        val preview: String,
        val fullLen: Int,
        val truncated: Boolean
    )

    /**
     * Normalize whitespace (including newlines) and cap to maxChars.
     * Keeps logs compact and stable.
     */
    private fun capTextStable(text: String, maxChars: Int): CappedText {
        val normalized = text
            .replace('\n', ' ')
            .replace('\r', ' ')
            .replace(Regex("\\s+"), " ")
            .trim()

        val fullLen = normalized.length
        if (fullLen <= maxChars) {
            return CappedText(preview = normalized, fullLen = fullLen, truncated = false)
        }
        val preview = normalized.substring(0, maxChars).trimEnd() + "…"
        return CappedText(preview = preview, fullLen = fullLen, truncated = true)
    }

    /**
     * Ensure a string cannot break JSONL by embedding real newlines.
     * We keep content readable by converting CR/LF to spaces and collapsing whitespace.
     *
     * NOTE: JSONObject will JSON-escape strings by itself; this extra step prevents accidental
     * raw newlines from custom toString() payload values or unexpected paths.
     */
    private fun sanitizeStringForJsonl(s: String): String {
        if (!s.contains('\n') && !s.contains('\r')) return s
        return s
            .replace('\r', ' ')
            .replace('\n', ' ')
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    /**
     * Emit what Sudo is about to say (before any TTS engine starts).
     *
     * Intended call site: MainActivity.logSudoSay(...) or speakAssistant(...)
     *
     * Fields:
     * - text_len: full normalized length
     * - text: capped preview (ASSISTANT_TEXT_CAP_CHARS)
     * - truncated: whether preview was capped
     * - source: e.g. "logSudoSay", "speakAssistant"
     * - engine/locale/turn_id (optional) if you have them
     */
    fun emitAssistantSay(
        text: String,
        source: String = "logSudoSay",
        engine: String? = null,
        locale: String? = null,
        turnId: Int? = null,
        convTurn: Int? = null,
        speakReqId: Int? = null,
        ttsId: Int? = null,
        utteranceId: String? = null,
        replyToRowId: Int? = null
    ) {
        val c = capTextStable(text, ASSISTANT_TEXT_CAP_CHARS)
        emitKv(
            "ASSISTANT_SAY",
            "source" to source,
            "engine" to engine,
            "locale" to locale,
            "turn_id" to turnId,
            "conv_turn" to convTurn,

            // ✅ critical for pairing
            "speak_req_id" to speakReqId,
            "tts_id" to ttsId,
            "utterance_id" to utteranceId,
            "reply_to_row_id" to replyToRowId,

            "text_len" to c.fullLen,
            "text" to c.preview,
            "truncated" to c.truncated
        )
    }

    /**
     * Emit what the user said (ASR final that your conductor accepted).
     *
     * Intended call site: ConversationTurnController.onAsrFinal(...)
     *
     * Fields mirror ASSISTANT_SAY:
     * - text_len: full normalized length
     * - text: capped preview (USER_TEXT_CAP_CHARS)
     * - truncated: whether preview was capped
     * - source: e.g. "turn_controller", "manual_ui"
     * - row_id / confidence / turn_id / conv_turn (optional)
     */
    fun emitUserSay(
        text: String,
        source: String = "turn_controller",
        rowId: Int? = null,
        confidence: Float? = null,
        turnId: Int? = null,
        convTurn: Int? = null
    ) {
        val c = capTextStable(text, USER_TEXT_CAP_CHARS)
        emitKv(
            "USER_SAY",
            "source" to source,
            "row_id" to rowId,
            "confidence" to confidence,
            "turn_id" to turnId,
            "text_len" to c.fullLen,
            "text" to c.preview,
            "truncated" to c.truncated
        )
    }

    /**
     * Optional: explicit pairing between a user ASR row and the next assistant speak request.
     * This removes ambiguity during offline scoring when multiple events interleave.
     */
    fun emitTurnPair(
        rowId: Int,
        speakReqId: Int,
        convTurn: Int? = null
    ) {
        emitKv(
            "TURN_PAIR",
            "row_id" to rowId,
            "speak_req_id" to speakReqId,
            "conv_turn" to convTurn
        )
    }

    // ---------------------- Grid hash helpers ----------------------

    /**
     * Stable hash of the current 81-cell grid. This is used to prove the LLM call
     * was made for a specific grid snapshot.
     *
     * Accepts:
     * - IntArray of size 81 (0..9) or null
     * Returns:
     * - sha256 hex of "d0,d1,...,d80" or null if unavailable
     */
    fun gridHashFromDigits(digits: IntArray?): String? {
        if (digits == null || digits.size != 81) return null
        val s = buildString(81 * 2) {
            for (i in 0 until 81) {
                if (i > 0) append(',')
                append(digits[i])
            }
        }
        // reuse existing private SHA helper from inside the object
        return sha256HexUtf8(s)
    }

    /** Public wrapper so other files can hash without duplicating MessageDigest code. */
    fun sha256Hex(text: String): String = sha256HexUtf8(text)

    // ---------------------- POLICY TRACE helpers ----------------------

    /**
     * Standardized policy trace event.
     *
     * Use this for every policy call boundary, so offline analysis can quickly answer:
     * - what pending state existed?
     * - was repair_mode on?
     * - what prompt hash / digest was used?
     *
     * This function additionally sanitizes multiline strings to keep JSONL 1-line safe.
     *
     * Suggested keys (caller supplies):
     * - pending_type, pending_idx, pending_row, pending_col, pending_digit
     * - repair_mode, repair_reason
     * - prompt_hash (or sys_sha256/dev_sha256 if you prefer)
     */
    fun emitPolicyTrace(tag: String, data: Map<String, Any?>) {
        val payload = linkedMapOf<String, Any?>(
            "type" to "POLICY_TRACE",
            "tag" to tag
        )

        // Sanitize string values defensively.
        data.forEach { (k, v) ->
            payload[k] = when (v) {
                is String -> sanitizeStringForJsonl(v)
                else -> v
            }
        }

        emit(payload)
    }

    // ---------------------- LLM prompt digest helpers ----------------------

    private const val PROMPT_TEXT_CAP_CHARS = 420

    private fun sha256HexUtf8(s: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(s.toByteArray(Charsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        for (b in digest) hex.append(String.format("%02x", b))
        return hex.toString()
    }

    private fun textDigestPairs(prefix: String, text: String?, cap: Int = PROMPT_TEXT_CAP_CHARS): List<Pair<String, Any?>> {
        if (text == null) {
            return listOf(
                "${prefix}_len" to 0,
                "${prefix}_sha256" to null,
                "${prefix}_preview" to null,
                "${prefix}_truncated" to false
            )
        }
        // capTextStable already flattens newlines -> spaces
        val c = capTextStable(text, cap)
        val sha = sha256HexUtf8(text)
        return listOf(
            "${prefix}_len" to c.fullLen,
            "${prefix}_sha256" to sha,
            "${prefix}_preview" to c.preview,
            "${prefix}_truncated" to c.truncated
        )
    }

    /**
     * Emit a compact, stable digest of what we're about to send to the LLM.
     * This is the fastest way to prove whether history/context was actually present.
     *
     * NOTE: All previews are newline-safe (capTextStable) so JSONL stays valid.
     */
    fun emitLlmRequestDigest(
        mode: String, // "FREE_TALK" / "GRID"
        model: String?,
        convoSessionId: String,
        turnId: Long?,
        promptHash: String?,
        personaHash: String?,
        systemPrompt: String?,
        developerPrompt: String?,
        userMessage: String?
    ) {
        val kv = mutableListOf<Pair<String, Any?>>()
        kv.add("mode" to mode)
        kv.add("model" to model)
        kv.add("convo_session_id" to convoSessionId)
        kv.add("turn_id" to turnId)
        kv.add("prompt_hash" to promptHash)
        kv.add("persona_hash" to personaHash)

        kv.addAll(textDigestPairs("sys", systemPrompt))
        kv.addAll(textDigestPairs("dev", developerPrompt))
        kv.addAll(textDigestPairs("usr", userMessage, cap = 220))

        emitKv("LLM_REQUEST_DIGEST", *kv.toTypedArray())
    }

    fun emitLlmResponseDigest(
        mode: String,
        model: String?,
        convoSessionId: String,
        turnId: Long?,
        assistantText: String?
    ) {
        val kv = mutableListOf<Pair<String, Any?>>()
        kv.add("mode" to mode)
        kv.add("model" to model)
        kv.add("convo_session_id" to convoSessionId)
        kv.add("turn_id" to turnId)

        kv.addAll(textDigestPairs("asst", assistantText))

        emitKv("LLM_RESPONSE_DIGEST", *kv.toTypedArray())
    }

    // ---------------------- ASR row tracking helpers ----------------------

    /**
     * Internal per-row accumulator for ASR "turn-taking" stats.
     * We only ever have at most one active ASR row at a time.
     */
    private data class AsrRowStats(
        val rowId: Int,
        val startMs: Long,
        var endMs: Long = startMs,
        var hasReady: Boolean = false,
        var hasRms: Boolean = false,
        var hasStop: Boolean = false,
        var hasNoMatch: Boolean = false,
        var hasBusy: Boolean = false,
        var hasClient: Boolean = false,
        var barsStartCount: Int = 0,
        var barsStopCount: Int = 0,
        var hasTtsOverlap: Boolean = false
    )

    /**
     * Public, immutable summary that we log as CONVTEL_ASR_ROW_EVAL.
     */
    data class AsrRowEval(
        val rowId: Int,
        val durationMs: Long,
        val hasReady: Boolean,
        val hasRms: Boolean,
        val hasStopOrNoMatch: Boolean,
        val hasBusy: Boolean,
        val hasClient: Boolean,
        val hasBarsStart: Boolean,
        val hasBarsStop: Boolean,
        val hasTtsOverlap: Boolean,
        val isNormal: Boolean
    )

    private val asrLock = Any()
    private var nextAsrRowId: Int = 1
    private var activeAsrRow: AsrRowStats? = null

    /**
     * Start a new ASR "row" (one listening attempt).
     *
     * - Assigns a fresh row_id (1, 2, 3, ...)
     * - Stores internal stats for that row
     * - Emits an ASR_START event with row_id attached.
     *
     * Returns the allocated row_id, or -1 if telemetry isn't initialized.
     */
    fun asrRowStart(vararg kv: Pair<String, Any?>): Int {
        val ctx = appContext ?: return -1
        val now = System.currentTimeMillis()

        val rowId: Int
        synchronized(asrLock) {
            rowId = nextAsrRowId++
            activeAsrRow = AsrRowStats(rowId = rowId, startMs = now)
        }

        val payload = linkedMapOf<String, Any?>(
            "type" to "ASR_START",
            "row_id" to rowId
        )
        for ((k, v) in kv) payload[k] = v

        emitInternal(ctx, payload, now)
        return rowId
    }

    /**
     * Attach an event to the currently active ASR row, if any, and log it.
     */
    fun asrRowEvent(type: String, vararg kv: Pair<String, Any?>) {
        val ctx = appContext ?: return
        val now = System.currentTimeMillis()

        val rowId: Int?
        synchronized(asrLock) {
            val row = activeAsrRow
            rowId = row?.rowId
            if (row != null) {
                row.endMs = now
                when (type) {
                    "ASR_READY" -> row.hasReady = true
                    "ASR_RMS" -> row.hasRms = true
                    "ASR_STOP" -> row.hasStop = true
                    "ASR_ERROR" -> {
                        val code = kv.firstOrNull { it.first == "code" }?.second as? Int
                        when (code) {
                            7 -> row.hasNoMatch = true
                            8 -> row.hasBusy = true
                            5 -> row.hasClient = true
                        }
                    }
                    "UI_BARS_START" -> row.barsStartCount += 1
                    "UI_BARS_STOP" -> row.barsStopCount += 1
                    "TTS_START" -> row.hasTtsOverlap = true
                }
            }
        }

        val payload = linkedMapOf<String, Any?>("type" to type)
        if (rowId != null) payload["row_id"] = rowId
        for ((k, v) in kv) payload[k] = v

        emitInternal(ctx, payload, now)
    }

    /**
     * End the current ASR row and emit:
     *  - The terminal ASR event itself
     *  - A synthesized CONVTEL_ASR_ROW_EVAL summary event for that row.
     */
    fun asrRowEnd(type: String, vararg kv: Pair<String, Any?>) {
        val ctx = appContext ?: return
        val now = System.currentTimeMillis()

        val eval: AsrRowEval?
        val rowId: Int?

        synchronized(asrLock) {
            val row = activeAsrRow
            rowId = row?.rowId
            if (row != null) {
                row.endMs = now
                when (type) {
                    "ASR_READY" -> row.hasReady = true
                    "ASR_RMS" -> row.hasRms = true
                    "ASR_STOP" -> row.hasStop = true
                    "ASR_ERROR" -> {
                        val code = kv.firstOrNull { it.first == "code" }?.second as? Int
                        when (code) {
                            7 -> row.hasNoMatch = true
                            8 -> row.hasBusy = true
                            5 -> row.hasClient = true
                        }
                    }
                    "UI_BARS_START" -> row.barsStartCount += 1
                    "UI_BARS_STOP" -> row.barsStopCount += 1
                    "TTS_START" -> row.hasTtsOverlap = true
                }
                eval = buildAsrEval(row)
                activeAsrRow = null
            } else {
                eval = null
            }
        }

        // Emit the terminal ASR event itself.
        val payload = linkedMapOf<String, Any?>("type" to type)
        if (rowId != null) payload["row_id"] = rowId
        for ((k, v) in kv) payload[k] = v
        emitInternal(ctx, payload, now)

        // Emit the synthesized per-row summary if we had an active row.
        if (eval != null) {
            val evalPayload = linkedMapOf<String, Any?>(
                "type" to "CONVTEL_ASR_ROW_EVAL",
                "row_id" to eval.rowId,
                "duration_ms" to eval.durationMs,
                "has_ready" to eval.hasReady,
                "has_rms" to eval.hasRms,
                "has_stop_or_no_match" to eval.hasStopOrNoMatch,
                "has_busy" to eval.hasBusy,
                "has_client" to eval.hasClient,
                "has_bars_start" to eval.hasBarsStart,
                "has_bars_stop" to eval.hasBarsStop,
                "has_tts_overlap" to eval.hasTtsOverlap,
                "is_normal" to eval.isNormal
            )
            emitInternal(ctx, evalPayload, now)
        }
    }

    private fun buildAsrEval(row: AsrRowStats): AsrRowEval {
        val duration = (row.endMs - row.startMs).coerceAtLeast(0L)
        val hasStopOrNoMatch = row.hasStop || row.hasNoMatch

        // Phase 0 truth:
        // - ASR row health must NOT require UI bars (bars are for TTS/Sudo voice only).
        // - ERROR_CLIENT is often a benign side-effect around cancel/tts suppression;
        //   do not fail "normal" solely on hasClient.
        val isNormal =
            (row.hasReady || row.hasRms) &&
                    hasStopOrNoMatch &&
                    !row.hasBusy &&
                    !row.hasTtsOverlap &&
                    duration >= MIN_ROW_DURATION_MS

        return AsrRowEval(
            rowId = row.rowId,
            durationMs = duration,
            hasReady = row.hasReady,
            hasRms = row.hasRms,
            hasStopOrNoMatch = hasStopOrNoMatch,
            hasBusy = row.hasBusy,
            hasClient = row.hasClient,
            hasBarsStart = row.barsStartCount > 0,
            hasBarsStop = row.barsStopCount > 0,
            hasTtsOverlap = row.hasTtsOverlap,
            isNormal = isNormal
        )
    }

    // ---------------------- Public generic API ----------------------

    /** Call once in Application.onCreate() (preferred) or very early in MainActivity. */
    fun init(context: Context, userId: String? = null, sessionId: String? = null, logcatEcho: Boolean = true) {
        if (started.get()) return
        this.appContext = context.applicationContext
        userId?.let { this.userId = it }
        sessionId?.let { this.sessionId = it }
        this.logcatEcho = logcatEcho

        startWriterThread()
        emitKv(
            "APP_START",
            "sdk_int" to Build.VERSION.SDK_INT,
            "device" to "${Build.MANUFACTURER} ${Build.MODEL}".trim(),
            "user_id" to (this.userId ?: "(none)"),
            "session_id" to this.sessionId
        )
    }

    fun emitConversationModeChosen(
        convoSessionId: String,
        mode: String,      // "FREE_TALK" or "GRID"
        reason: String,    // e.g. "no_grid", "grid_present"
        printed: Int? = null,
        handwritten: Int? = null,
        blanks: Int? = null,
        nonZero: Int? = null
    ) {
        emitKv(
            "CONV_MODE_CHOSEN",
            "convo_session_id" to convoSessionId,
            "mode" to mode,
            "reason" to reason,
            "printed" to printed,
            "handwritten" to handwritten,
            "blanks" to blanks,
            "nonZero" to nonZero
        )
    }

    /** Update just the session id (e.g., per capture flow). */
    fun setSessionId(id: String) {
        val prev = this.sessionId
        this.sessionId = id
        emitKv(
            "SESSION_SET",
            "prev_session_id" to prev,
            "session_id" to id
        )
    }

    /** Update/attach a user id for context. */
    fun setUserId(id: String) {
        this.userId = id
        emitKv("USER_SET", "user_id" to id)
    }

    /** Enable/disable Logcat echo. */
    fun setLogcatEcho(enabled: Boolean) {
        this.logcatEcho = enabled
    }

    fun emit(payload: Map<String, Any?>) {
        val ctx = appContext ?: return
        val now = System.currentTimeMillis()
        emitInternal(ctx, payload, now)
    }

    fun emitKv(type: String, vararg kv: Pair<String, Any?>) {
        val map = linkedMapOf<String, Any?>("type" to type)
        for ((k, v) in kv) map[k] = v
        emit(map)
    }

    /** Flush and stop (usually not needed; process exit closes). */
    fun shutdown() {
        try {
            started.set(false)
            writerThread?.interrupt()
            writerThread = null
            fileWriter?.flush()
            fileWriter?.close()
        } catch (_: Throwable) {
        }
        fileWriter = null
    }

    // ---------------------- Internals ----------------------

    private fun startWriterThread() {
        if (!started.compareAndSet(false, true)) return
        writerThread = thread(name = "ConvTelWriter", start = true, isDaemon = true) {
            while (started.get()) {
                try {
                    val line = queue.poll(1500, TimeUnit.MILLISECONDS) ?: continue
                    writeLine(line)
                } catch (_: InterruptedException) {
                    break
                } catch (t: Throwable) {
                    Log.w(LOG_TAG, "writer loop error", t)
                }
            }
            // drain on exit
            var leftover: String? = queue.poll()
            while (leftover != null) {
                try {
                    writeLine(leftover)
                } catch (_: Throwable) {
                }
                leftover = queue.poll()
            }
            try { fileWriter?.flush() } catch (_: Throwable) {}
            try { fileWriter?.close() } catch (_: Throwable) {}
        }
    }

    private fun offer(line: String) {
        if (!queue.offer(line)) {
            queue.poll()
            queue.offer(line)
        }
    }

    private fun emitInternal(ctx: Context, payload: Map<String, Any?>, now: Long) {
        val obj = JSONObject()

        // ---- extra always-on diagnostics ----
        val pid = Process.myPid()
        val tid = Process.myTid()
        val threadName = Thread.currentThread().name
        val uptimeMs = SystemClock.uptimeMillis()
        val elapsedMs = SystemClock.elapsedRealtime()

        // Standard fields
        obj.put("ts_epoch_ms", now)
        obj.put("ts_iso", isoFmt.format(Date(now)))
        obj.put("type", payload["type"] ?: "(unknown)")
        obj.put("session_id", sessionId)
        obj.put("telemetry_id", telemetryId)
        obj.put("seq", seq.incrementAndGet())
        obj.put("pid", pid)
        obj.put("tid", tid)
        obj.put("thread", threadName)
        obj.put("ts_uptime_ms", uptimeMs)
        obj.put("ts_elapsed_ms", elapsedMs)

        userId?.let { obj.put("user_id", it) }

        // Merge user payload
        payload.forEach { (k, v) ->
            if (k == "type") return@forEach
            putSafe(obj, k, v)
        }

        val line = obj.toString()

        if (logcatEcho) Log.i(LOG_TAG, line)

        // IMPORTANT: open file BEFORE enqueueing (prevents rare "drop" race)
        ensureOpenForToday(ctx)
        offer(line)
    }

    private fun ensureOpenForToday(ctx: Context) {
        val dayDir = File(ctx.filesDir, "telemetry/${dayFmt.format(Date())}")
        if (currentDayDir?.absolutePath != dayDir.absolutePath) {
            try { fileWriter?.flush() } catch (_: Throwable) {}
            try { fileWriter?.close() } catch (_: Throwable) {}
            currentDayDir = dayDir
            if (!dayDir.exists()) dayDir.mkdirs()
            byteCount = 0L
            openNewFile(dayDir)
        } else if (fileWriter == null || byteCount >= MAX_BYTES_PER_FILE) {
            openNewFile(dayDir)
        }
    }

    private fun openNewFile(dir: File) {
        val ts = SimpleDateFormat("HHmmss", Locale.US).format(Date())
        val fname = "events_${sessionId}_$ts.jsonl"
        val f = File(dir, fname)
        fileWriter = FileWriter(f, /*append*/ true)
        byteCount = 0L
        try {
            Log.i(LOG_TAG, "telemetry file: ${f.absolutePath}")
        } catch (_: Throwable) {
        }
    }

    private fun writeLine(line: String) {
        val fw = fileWriter ?: return
        fw.write(line)
        fw.write("\n")
        byteCount += (line.length + 1)
        if (byteCount >= MAX_BYTES_PER_FILE) {
            try { fw.flush() } catch (_: Throwable) {}
            fileWriter = null
        }
    }

    private fun putSafe(obj: JSONObject, key: String, value: Any?) {
        when (value) {
            null -> obj.put(key, JSONObject.NULL)
            is Number, is Boolean -> obj.put(key, value)
            is String -> obj.put(key, sanitizeStringForJsonl(value))
            is Iterable<*> -> {
                val arr = org.json.JSONArray()
                for (v in value) {
                    val vv = when (v) {
                        is String -> sanitizeStringForJsonl(v)
                        else -> v
                    }
                    arr.put(vv)
                }
                obj.put(key, arr)
            }
            is IntArray -> {
                val arr = org.json.JSONArray()
                value.forEach { arr.put(it) }
                obj.put(key, arr)
            }
            is FloatArray -> {
                val arr = org.json.JSONArray()
                value.forEach { arr.put(it) }
                obj.put(key, arr)
            }
            is LongArray -> {
                val arr = org.json.JSONArray()
                value.forEach { arr.put(it) }
                obj.put(key, arr)
            }
            is Array<*> -> {
                val arr = org.json.JSONArray()
                value.forEach { v ->
                    val vv = when (v) {
                        is String -> sanitizeStringForJsonl(v)
                        else -> v
                    }
                    arr.put(vv)
                }
                obj.put(key, arr)
            }
            else -> obj.put(key, sanitizeStringForJsonl(value.toString()))
        }
    }
}