package com.contextionary.sudoku.telemetry

import android.content.Context
import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread


/**
 * ConversationTelemetry — tiny, low-overhead JSONL event sink.
 *
 * ✅ Goals (audit-friendly, deterministic reconstruction):
 * - Provide canonical IDs and lifecycle events so an offline audit tool can reconstruct:
 * - what the user said (final accepted input)
 * - what the assistant displayed and spoke (and whether UI text == TTS text)
 * - which model call happened when (tick_id + model_call_id + model_call_name)
 * - which toolplan was accepted/rewritten/rejected and why
 * - which tool results were produced and consumed by continuation (tick2)
 * - state machine transitions (pending/phase) without guesswork
 *
 * ✅ This file is intentionally “dumb”: it does not compute product logic.
 * It just standardizes events, IDs, and structure so downstream tooling can be precise.
 *
 * ---------------------------------------------------------------------------
 * FILE OUTPUT
 * - Writes JSON lines into app-private storage: files/telemetry/<day>/events_<app_session>.jsonl
 * - Thread-safe, non-blocking (producer queue + background writer).
 * - Optional Logcat echo for quick inspection.
 * ---------------------------------------------------------------------------
 *
 * ---------------------------------------------------------------------------
 * IMPORTANT CONCEPTS (the canonical join keys)
 * - app_session_id: stable for this telemetry file stream (previously "session_id")
 * - conversation_id: stable across the whole user conversation (set by the app)
 * - turn_id / turn_seq: stable identifiers for a single assistant turn
 * - row_id: stable identifier for a single ASR listening attempt
 * - tick_id: model call tick number in a turn (e.g., 1=policy, 2=continuation)
 * - model_call_id: stable ID for one model call (LLM request/response)
 * - toolplan_id: stable ID for one toolplan output by the model
 * - tool_result_id: stable ID for one tool execution result
 * - correlation_id: stable “span” ID that ties prompt build + HTTP + parsing + apply + tick2
 * ---------------------------------------------------------------------------
 */
object ConversationTelemetry {

    // =========================================================================
    // Versioning / schema
    // =========================================================================

    /**
     * Increase when you add/remove/rename event fields in a way that breaks an audit tool.
     * Audit tooling should read this and adapt parsing accordingly.
     */
    private const val TELEMETRY_SCHEMA_VERSION = "v2.0"

    // =========================================================================
    // Tuning
    // =========================================================================

    private const val LOG_TAG = "ConvTel"
    private const val MAX_BYTES_PER_FILE = 2_000_000L // ~2 MB per file
    private const val ASSISTANT_TEXT_CAP_CHARS = 2000
    private const val USER_TEXT_CAP_CHARS = 2000
    private const val PROMPT_TEXT_CAP_CHARS = 2000
    private const val MIN_ROW_DURATION_MS = 400L

    // =========================================================================
    // Writer thread state
    // =========================================================================

    private val started = AtomicBoolean(false)
    private val queue = LinkedBlockingQueue<String>(4096)
    private var writerThread: Thread? = null

    // =========================================================================
    // Identity / context (stable)
    // =========================================================================

    @Volatile
    private var appContext: Context? = null

    /**
     * app_session_id is the telemetry stream/session for this file.
     * Historically you used "session_id". We keep that as "app_session_id" (canonical),
     * and we still emit "session_id" for backward compatibility, but avoid using it in audits.
     */
    @Volatile
    private var appSessionId: String = UUID.randomUUID().toString().substring(0, 8)

    /** Optional user identity, if you have it (not required). */
    @Volatile
    private var userId: String? = null

    /** Whether to echo each JSONL line into Logcat. */
    @Volatile
    private var logcatEcho: Boolean = true

    /**
     * conversation_id: stable across the entire conversation.
     * Set this once you start a conversation (e.g., app launch -> new conversation).
     *
     * Audit benefit:
     * - Lets the audit tool group multiple app sessions (ASR/TTS subsystems) into one conversation,
     *   even if process restarts or session ids differ.
     */
    @Volatile
    private var conversationId: String? = null

    /**
     * Optional subsystem session IDs (for your multi-session telemetry reality).
     * These are NOT required, but incredibly useful when events interleave.
     */
    @Volatile
    private var deviceSessionId: String? = null

    @Volatile
    private var voiceSessionId: String? = null

    @Volatile
    private var policySessionId: String? = null

    /**
     * App build metadata (set once).
     * Audit benefit:
     * - Correlate behavior changes to builds; prevent mixing schemas.
     */
    @Volatile
    private var appVersionName: String? = null

    @Volatile
    private var appBuildCode: Long? = null

    @Volatile
    private var gitSha: String? = null

    // =========================================================================
    // File rolling
    // =========================================================================

    private var currentDayDir: File? = null
    private var fileWriter: FileWriter? = null
    private var byteCount: Long = 0L

    // =========================================================================
    // Common timestamps
    // =========================================================================

    private val dayFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)
    private val isoFmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US)

    // =========================================================================
    // Sequencing / ID generators
    // =========================================================================

    private val telemetryId: String = UUID.randomUUID().toString().substring(0, 8)
    private val seq = AtomicLong(0L)
    private val policyReqSeq = AtomicLong(0L)
    private val modelCallSeq = AtomicLong(0L)
    private val toolplanSeq = AtomicLong(0L)
    private val toolResultSeq = AtomicLong(0L)
    private val assistantMessageSeq = AtomicLong(0L)
    private val uiBarsSeq = AtomicLong(0L)
    private val suppressionSeq = AtomicLong(0L)
    private val correlationSeq = AtomicLong(0L)

    // =========================================================================
    // 6.1/6.2 — Pending “speak” registry (join ASSISTANT_SAY / ENQUEUED -> TTS DONE)
    // =========================================================================

    private data class PendingSpeak(
        val engine: String,
        val key: String,                      // stable key used for lookup at completion
        val assistantMessageId: String? = null,
        val turnId: Long? = null,
        val tickId: Int? = null,
        val speakReqId: Int? = null,
        val replyToRowId: Int? = null,
        val ttsId: Int? = null,
        val utteranceId: String? = null,
        val voice: String? = null,
        val locale: String? = null,
        val textHash: String? = null,
        val textLen: Int? = null,
        val enqueuedTsEpochMs: Long = 0L,

        // ✅ new fields (defaults keep old constructors compiling)
        val replyTextSha256: String? = null,
        val uiTextSha256: String? = null,
        val replyId: String? = null,
        val replySha12: String? = null
    )


    private val pendingSpeak = ConcurrentHashMap<String, PendingSpeak>()

    /*
    private fun speakKey(
        engine: String,
        ttsId: Int?,
        speakReqId: Int?,
        utteranceId: String?
    ): String {
        // Prefer strongest identifiers first
        if (ttsId != null) return "$engine:tts:$ttsId"
        if (speakReqId != null) return "$engine:sr:$speakReqId"
        if (!utteranceId.isNullOrBlank()) return "$engine:utt:$utteranceId"
        // last resort (not great, but deterministic)
        return "$engine:anon"
    }
    */

    private val pendingSpeakByKey = java.util.concurrent.ConcurrentHashMap<String, PendingSpeak>()
    private val pendingSpeakBySpeakReqId = java.util.concurrent.ConcurrentHashMap<Int, PendingSpeak>()
    private val pendingSpeakByReplyId = java.util.concurrent.ConcurrentHashMap<String, PendingSpeak>()

    private fun speakKey(engine: String, ttsId: Int?, speakReqId: Int?, utteranceId: String?): String {
        // IMPORTANT: utteranceId can be null for Azure; speakReqId is the most stable.
        return buildString {
            append(engine)
            append("|tts=").append(ttsId ?: -1)
            append("|sr=").append(speakReqId ?: -1)
            append("|utt=").append(utteranceId ?: "null")
        }
    }


    private fun speakKeyV2(
        engine: String,
        replyId: String?,
        ttsId: Int?,
        speakReqId: Int?,
        utteranceId: String?
    ): String {
        // replyId is strongest — stable, explicit, and cross-engine.
        if (!replyId.isNullOrBlank()) return "rid:$engine:$replyId"

        // fallback key (still deterministic)
        val t = ttsId?.toString() ?: "na"
        val s = speakReqId?.toString() ?: "na"
        val u = utteranceId ?: "na"
        return "k:$engine:t$t:s$s:u$u"
    }

    /**
     * Register a speak attempt at ENQUEUE time.
     * This lets TTS_DONE/TTS_ERROR resolve what was actually spoken later.
     */



    private fun registerPendingSpeak(
        engine: String,
        assistantMessageId: String?,
        turnId: Long?,
        tickId: Int?,
        ttsId: Int?,
        speakReqId: Int?,
        replyToRowId: Int?,
        utteranceId: String?,
        voice: String?,
        locale: String?,
        replyTextSha256: String?,
        uiTextSha256: String?,
        text: String?,
        // ✅ new binding inputs (optional; we derive if missing)
        replyId: String? = null,
        replySha12: String? = null
    ) {
        val now = System.currentTimeMillis()

        val replySha = replyTextSha256 ?: text?.let { sha256HexUtf8(it) }
        val rid = replyId ?: run {
            val t = turnId?.toString() ?: "na"
            val sr = speakReqId?.toString() ?: "na"
            "turn-$t-sr-$sr"
        }
        val sha12 = replySha12 ?: replySha?.take(12)

        val k = speakKeyV2(engine, rid, ttsId, speakReqId, utteranceId)

        val textHash = text?.let { sha256HexUtf8(it) }
        val textLen = text?.length

        pendingSpeak[k] = PendingSpeak(
            engine = engine,
            key = k,
            assistantMessageId = assistantMessageId,
            turnId = turnId,
            tickId = tickId,
            speakReqId = speakReqId,
            replyToRowId = replyToRowId,
            ttsId = ttsId,
            utteranceId = utteranceId,
            voice = voice,
            locale = locale,
            textHash = textHash,
            textLen = textLen,
            enqueuedTsEpochMs = now,
            replyTextSha256 = replySha,
            uiTextSha256 = uiTextSha256,
            replyId = rid,
            replySha12 = sha12
        )
    }

    /**
     * Resolve & emit ASSISTANT_REPLY_SPOKEN.
     *
     * 6.2: This must be called ONLY from real TTS completion (success path).
     * Do NOT call it from enqueue/start.
     */
    fun resolveAndEmitAssistantReplySpoken(
        engine: String,
        ttsId: Int? = null,
        speakReqId: Int? = null,
        utteranceId: String? = null,
        durationMs: Long? = null,
        cacheHit: Boolean? = null,
        ok: Boolean,
        errorCode: String? = null,
        reason: String? = null,
        // Optional: if engine can provide it at completion time, you can pass it and we’ll cross-check
        replyTextSha256Observed: String? = null,

        // ✅ 6.4 — NEW (keep defaults so old call sites compile)
        replyId: String? = null,
        replySha12: String? = null
    ) {
        // Prefer replyId as the join key if present (makes unmatched basically impossible)
        val k = speakKeyV2(engine, replyId, ttsId, speakReqId, utteranceId)
        val p = pendingSpeak.remove(k)

        if (!ok) {
            emitKv(
                "ASSISTANT_REPLY_NOT_SPOKEN",
                "conversation_id" to conversationId,
                "engine" to engine,
                "reply_id" to replyId,
                "reply_sha12" to replySha12,
                "tts_id" to ttsId,
                "speak_req_id" to speakReqId,
                "utterance_id" to utteranceId,
                "error_code" to errorCode,
                "reason" to reason,
                "matched_pending" to (p != null),
                "turn_id" to p?.turnId,
                "tick_id" to p?.tickId,
                "reply_text_sha256" to p?.replyTextSha256,
                "ui_text_sha256" to p?.uiTextSha256
            )
            return
        }

        if (p == null) {
            emitKv(
                "ASSISTANT_REPLY_SPOKEN",
                "conversation_id" to conversationId,
                "engine" to engine,
                "assistant_message_id" to null,
                "turn_id" to null,
                "tick_id" to null,
                "reply_id" to replyId,
                "reply_sha12" to replySha12,
                "tts_id" to ttsId,
                "speak_req_id" to speakReqId,
                "reply_to_row_id" to null,
                "utterance_id" to utteranceId,
                "voice" to null,
                "locale" to null,
                "text_hash" to null,
                "text_len" to null,
                "duration_ms" to durationMs,
                "cache_hit" to cacheHit,
                "ok" to true,
                "unmatched" to true,
                "reason" to reason,
                "reply_text_sha256" to replyTextSha256Observed
            )
            return
        }

        val mismatch =
            (replyTextSha256Observed != null && p.replyTextSha256 != null && replyTextSha256Observed != p.replyTextSha256)

        emitKv(
            "ASSISTANT_REPLY_SPOKEN",
            "conversation_id" to conversationId,
            "engine" to engine,
            "assistant_message_id" to p.assistantMessageId,
            "turn_id" to p.turnId,
            "tick_id" to p.tickId,

            // ✅ 6.4 binding fields (authoritative from pending)
            "reply_id" to p.replyId,
            "reply_sha12" to p.replySha12,

            "tts_id" to (ttsId ?: p.ttsId),
            "speak_req_id" to (speakReqId ?: p.speakReqId),
            "reply_to_row_id" to p.replyToRowId,
            "utterance_id" to (utteranceId ?: p.utteranceId),
            "voice" to p.voice,
            "locale" to p.locale,
            "text_hash" to p.textHash,
            "text_len" to p.textLen,
            "duration_ms" to durationMs,
            "cache_hit" to cacheHit,
            "ok" to true,
            "unmatched" to false,
            "reason" to reason,
            "reply_text_sha256" to p.replyTextSha256,
            "ui_text_sha256" to p.uiTextSha256,
            "reply_text_sha256_observed" to replyTextSha256Observed,
            "hash_mismatch" to mismatch
        )
    }

    fun nextPolicyReqSeq(sessionId: String, turnId: Long): Long =
        policyReqSeq.incrementAndGet()

    /** Stable short correlation id for joining many low-level events within a single span. */
    fun nextCorrelationId(prefix: String = "corr"): String {
        val n = correlationSeq.incrementAndGet()
        return "$prefix-$n"
    }

    private val perTurnTickReqSeq = java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.atomic.AtomicLong>()

    fun nextPolicyReqSeq(sessionId: String, turnId: Long, tickId: Int): Long {
        val key = "$sessionId:$turnId:$tickId"
        val c = perTurnTickReqSeq.getOrPut(key) { java.util.concurrent.atomic.AtomicLong(0L) }
        return c.incrementAndGet()
    }

    /** Stable model call id (one LLM request/response). */
    fun nextModelCallId(prefix: String = "mc"): String {
        val n = modelCallSeq.incrementAndGet()
        return "$prefix-$n"
    }

    /** Stable toolplan id (one parsed model tool_calls output). */
    fun nextToolplanId(prefix: String = "tp"): String {
        val n = toolplanSeq.incrementAndGet()
        return "$prefix-$n"
    }

    /** Stable tool result id (one tool execution result, used by tick2). */
    fun nextToolResultId(prefix: String = "tr"): String {
        val n = toolResultSeq.incrementAndGet()
        return "$prefix-$n"
    }

    /** Stable assistant message id (UI display / speak events). */
    fun nextAssistantMessageId(prefix: String = "am"): String {
        val n = assistantMessageSeq.incrementAndGet()
        return "$prefix-$n"
    }

    /** Stable UI bars instance id (one start/stop window). */
    fun nextUiBarsInstanceId(prefix: String = "bars"): String {
        val n = uiBarsSeq.incrementAndGet()
        return "$prefix-$n"
    }

    /** Stable ASR suppression window id. */
    fun nextAsrSuppressionWindowId(prefix: String = "sup"): String {
        val n = suppressionSeq.incrementAndGet()
        return "$prefix-$n"
    }

    // =========================================================================
    // Text helpers
    // =========================================================================

    private data class CappedText(
        val preview: String,
        val fullLen: Int,
        val truncated: Boolean
    )

    /**
     * Normalize whitespace and cap to maxChars.
     * - Prevents multi-line JSONL
     * - Stabilizes whitespace for comparisons
     */
    private fun capTextStable(text: String, maxChars: Int): CappedText {
        val normalized = text
            .replace('\n', ' ')
            .replace('\r', ' ')
            .replace(Regex("\\s+"), " ")
            .trim()

        val fullLen = normalized.length
        if (fullLen <= maxChars) return CappedText(preview = normalized, fullLen = fullLen, truncated = false)

        val preview = normalized.substring(0, maxChars).trimEnd() + "…"
        return CappedText(preview = preview, fullLen = fullLen, truncated = true)
    }

    /**
     * Additional safety to ensure no raw newline sneaks into JSONL due to non-String payload toString().
     */
    private fun sanitizeStringForJsonl(s: String): String {
        if (!s.contains('\n') && !s.contains('\r')) return s
        return s
            .replace('\r', ' ')
            .replace('\n', ' ')
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun sha256HexUtf8(s: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(s.toByteArray(Charsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        for (b in digest) hex.append(String.format("%02x", b))
        return hex.toString()
    }

    /** Public wrapper for hashing text consistently across the app. */
    fun sha256Hex(text: String): String = sha256HexUtf8(text)

    /**
     * Stable hash of a grid snapshot.
     * - Accepts 81 digits (0..9)
     * - Returns sha256 of "d0,d1,...,d80"
     *
     * Audit benefit:
     * - Proves what grid snapshot a model call or toolplan refers to.
     */
    fun gridHashFromDigits(digits: IntArray?): String? {
        if (digits == null || digits.size != 81) return null
        val s = buildString(81 * 2) {
            for (i in 0 until 81) {
                if (i > 0) append(',')
                append(digits[i])
            }
        }
        return sha256HexUtf8(s)
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
        val c = capTextStable(text, cap)
        val sha = sha256HexUtf8(text)
        return listOf(
            "${prefix}_len" to c.fullLen,
            "${prefix}_sha256" to sha,
            "${prefix}_preview" to c.preview,
            "${prefix}_truncated" to c.truncated
        )
    }

    // =========================================================================
    // Canonical “context” events
    // =========================================================================

    /**
     * CONVERSATION_CONTEXT
     * -----------------------------------------------------------------------
     * Meaning:
     * - Announces the stable IDs that define the “join space” for all telemetry.
     *
     * When to emit:
     * - On init()
     * - Whenever conversation_id or subsystem session ids change
     *
     * Audit benefit:
     * - Lets offline tooling correctly correlate interleaved sessions (ASR/TTS/policy).
     */
    fun emitConversationContext(reason: String = "init_or_update") {
        emitKv(
            "CONVERSATION_CONTEXT",
            "schema_version" to TELEMETRY_SCHEMA_VERSION,
            "reason" to reason,
            "conversation_id" to conversationId,
            "app_session_id" to appSessionId,
            "device_session_id" to deviceSessionId,
            "voice_session_id" to voiceSessionId,
            "policy_session_id" to policySessionId,
            "app_version_name" to appVersionName,
            "app_build_code" to appBuildCode,
            "git_sha" to gitSha
        )
    }

    /**
     * TURN_CONTEXT_BIND
     * -----------------------------------------------------------------------
     * Meaning:
     * - Explicitly ties row_id (user input) to turn_id/turn_seq and correlation_id.
     *
     * When to emit:
     * - Immediately when a new turn is created (TURN_CREATE)
     * - Or when row pairing is resolved
     *
     * Audit benefit:
     * - Removes guesswork (timestamps) when multiple events interleave.
     */
    fun emitTurnContextBind(
        turnId: Long,
        turnSeq: Int?,
        rowId: Int?,
        correlationId: String,
        note: String? = null
    ) {
        emitKv(
            "TURN_CONTEXT_BIND",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "turn_seq" to turnSeq,
            "row_id" to rowId,
            "correlation_id" to correlationId,
            "note" to note
        )
    }

    // =========================================================================
    // Canonical “model call” events (LLM1/LLM2; tick_id)
    // =========================================================================

    /**
     * MODEL_SELECTED
     * -----------------------------------------------------------------------
     * Meaning:
     * - Records which model/provider is being used and why (fallback, config, etc.)
     *
     * When to emit:
     * - Right before making the HTTP call (or when model is chosen)
     *
     * Audit benefit:
     * - Prevents “model=null” ambiguity in digests; ties behavior to exact model.
     */
    fun emitModelSelected(
        modelCallId: String,
        tickId: Int,
        modelName: String,
        provider: String? = null,
        reason: String? = null
    ) {
        emitKv(
            "MODEL_SELECTED",
            "conversation_id" to conversationId,
            "model_call_id" to modelCallId,
            "tick_id" to tickId,
            "model" to modelName,
            "provider" to provider,
            "reason" to reason
        )
    }

    /**
     * MODEL_CALL_BEGIN / MODEL_CALL_END
     * -----------------------------------------------------------------------
     * Meaning:
     * - A named model call within a turn (e.g., LLM1_ACK, LLM2_CONTINUE).
     * - This is the single best way to make “Design A” verifiable.
     *
     * Use case examples:
     * - tick_id=1, model_call_name=LLM1_ACK, includes tool planning
     * - tick_id=2, model_call_name=LLM2_CONTINUE, sees tool results, must not repeat ack
     *
     * Audit benefit:
     * - Offline tool can measure “guard compliance” and detect duplicated restatements.
     */
    fun emitModelCallBegin(
        modelCallId: String,
        modelCallName: String,
        tickId: Int,
        turnId: Long?,
        correlationId: String,
        policyReqSeq: Long? = null
    ) {
        emitKv(
            "MODEL_CALL_BEGIN",
            "conversation_id" to conversationId,
            "model_call_id" to modelCallId,
            "model_call_name" to modelCallName,
            "tick_id" to tickId,
            "turn_id" to turnId,
            "correlation_id" to correlationId,
            "policy_req_seq" to policyReqSeq
        )
    }

    fun emitModelCallEnd(
        modelCallId: String,
        tickId: Int,
        turnId: Long?,
        correlationId: String,
        ok: Boolean,
        latencyMs: Long? = null,
        httpCode: Int? = null,
        parseOk: Boolean? = null,
        errorCode: String? = null,
        errorMsgShort: String? = null
    ) {
        emitKv(
            "MODEL_CALL_END",
            "conversation_id" to conversationId,
            "model_call_id" to modelCallId,
            "tick_id" to tickId,
            "turn_id" to turnId,
            "correlation_id" to correlationId,
            "ok" to ok,
            "latency_ms" to latencyMs,
            "http_code" to httpCode,
            "parse_ok" to parseOk,
            "error_code" to errorCode,
            "error_msg" to errorMsgShort
        )
    }

    // =========================================================================
    // Prompt / grid context lineage events (Design A proof)
    // =========================================================================

    /**
     * PROMPT_ATTACHMENTS
     * -----------------------------------------------------------------------
     * Meaning:
     * - Declares which components were included in the prompt for a model call.
     *
     * Examples (Design A):
     * - history_includes_llm1_reply=true
     * - history_includes_tool_results=true
     * - history_includes_updated_grid_context=true
     *
     * Audit benefit:
     * - Verifies “LLM2 saw LLM1 + tool results” without reading full prompt dumps.
     */
    fun emitPromptAttachments(
        modelCallId: String,
        tickId: Int,
        turnId: Long?,
        correlationId: String,
        promptHash: String?,
        msgCount: Int?,
        historyMsgsIn: Int?,
        historyMsgsUsed: Int?,
        historyIncludesLlm1Reply: Boolean?,
        historyIncludesToolResults: Boolean?,
        historyIncludesUpdatedGridContext: Boolean?,
        notes: String? = null
    ) {
        emitKv(
            "PROMPT_ATTACHMENTS",
            "conversation_id" to conversationId,
            "model_call_id" to modelCallId,
            "tick_id" to tickId,
            "turn_id" to turnId,
            "correlation_id" to correlationId,
            "prompt_hash" to promptHash,
            "msg_count" to msgCount,
            "history_msgs_in" to historyMsgsIn,
            "history_msgs_used" to historyMsgsUsed,
            "history_includes_llm1_reply" to historyIncludesLlm1Reply,
            "history_includes_tool_results" to historyIncludesToolResults,
            "history_includes_updated_grid_context" to historyIncludesUpdatedGridContext,
            "notes" to notes
        )
    }

    /**
     * GRID_CONTEXT_EMITTED
     * -----------------------------------------------------------------------
     * Meaning:
     * - Logs which grid context (hashes & counts) was used for a model call.
     *
     * Audit benefit:
     * - Confirms tick2 used the updated grid after tools applied.
     */
    fun emitGridContextEmitted(
        modelCallId: String,
        tickId: Int,
        turnId: Long?,
        correlationId: String,
        gridHashBefore: String?,
        gridHashAfter: String?,
        gridContextVersion: String?,
        gridContextHash: String?,
        solvability: String?,
        mismatchCount: Int?,
        unresolvedCount: Int?
    ) {
        emitKv(
            "GRID_CONTEXT_EMITTED",
            "conversation_id" to conversationId,
            "model_call_id" to modelCallId,
            "tick_id" to tickId,
            "turn_id" to turnId,
            "correlation_id" to correlationId,
            "grid_hash_before" to gridHashBefore,
            "grid_hash_after" to gridHashAfter,
            "grid_context_version" to gridContextVersion,
            "grid_context_hash" to gridContextHash,
            "solvability" to solvability,
            "mismatch_count" to mismatchCount,
            "unresolved_count" to unresolvedCount
        )
    }


    fun emitAssistantReplyComposed(
        turnId: Long?,
        tickId: Int?,
        speakReqId: Int?,
        ttsId: Int?,
        utteranceId: String?,
        engine: String?,
        source: String,
        replyText: String,
        uiText: String? = null,
        replyToRowId: Int? = null,
        replyId: String? = null,
        replySha12: String? = null
    ) {
        val replySha = sha256HexUtf8(replyText)
        val uiSha = uiText?.let { sha256HexUtf8(it) }
        val c = capTextStable(replyText, ASSISTANT_TEXT_CAP_CHARS)

        val eng = engine ?: "unknown"

        // ✅ Register pending speak BEFORE we emit so completion can always bind
        runCatching {
            registerPendingSpeak(
                engine = eng,
                assistantMessageId = null,
                turnId = turnId,
                tickId = tickId,
                ttsId = ttsId,
                speakReqId = speakReqId,
                replyToRowId = replyToRowId,
                utteranceId = utteranceId,
                voice = null,
                locale = null,
                replyTextSha256 = replySha,
                uiTextSha256 = uiSha,
                text = replyText,
                replyId = replyId,
                replySha12 = replySha12
            )
        }

        val rid = replyId ?: run {
            val t = turnId?.toString() ?: "na"
            val sr = speakReqId?.toString() ?: "na"
            "turn-$t-sr-$sr"
        }
        val sha12 = replySha12 ?: replySha.take(12)

        emitKv(
            "ASSISTANT_REPLY_COMPOSED",
            "conversation_id" to conversationId,
            "app_session_id" to appSessionId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "speak_req_id" to speakReqId,
            "tts_id" to ttsId,
            "utterance_id" to utteranceId,
            "engine" to engine,
            "source" to source,
            "reply_to_row_id" to replyToRowId,

            // ✅ 6.4 binding fields
            "reply_id" to rid,
            "reply_sha12" to sha12,

            "reply_text_len" to c.fullLen,
            "reply_text_sha256" to replySha,
            "ui_text_sha256" to uiSha,
            "reply_text_preview" to c.preview,
            "truncated" to c.truncated
        )
    }


    fun emitAssistantReplyComposed(
        turnId: Int?,
        tickId: Int?,
        speakReqId: Int?,
        ttsId: Int?,
        utteranceId: String?,
        engine: String?,
        source: String,
        replyText: String,
        uiText: String? = null,
        replyToRowId: Int? = null,
        replyId: String? = null,
        replySha12: String? = null
    ) {
        emitAssistantReplyComposed(
            turnId = turnId?.toLong(),
            tickId = tickId,
            speakReqId = speakReqId,
            ttsId = ttsId,
            utteranceId = utteranceId,
            engine = engine,
            source = source,
            replyText = replyText,
            uiText = uiText,
            replyToRowId = replyToRowId,
            replyId = replyId,
            replySha12 = replySha12
        )
    }




    /**
     * GRID_FACTS_SNAPSHOT — compact, parseable truth layers + facts lists.
     *
     * stage: "PRE_POLICY" | "POST_APPLY" (string, stable)
     *
     * IMPORTANT sizing rule:
     * - Always include the three 81-digit strings + index lists
     * - Cap big lists (e.g. mismatch_details)
     * - Do NOT include full candidates grid in this phase (out of scope)
     */
    fun emitGridFactsSnapshot(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        policyReqSeq: Long,
        modelCallId: String,
        correlationId: String,
        toolplanId: String,
        stage: String,
        payload: Map<String, Any?>
    ) {
        emitKv(
            type = "GRID_FACTS_SNAPSHOT",
            "session_id" to sessionId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "policy_req_seq" to policyReqSeq,
            "model_call_id" to modelCallId,
            "correlation_id" to correlationId,
            "toolplan_id" to toolplanId,
            "stage" to stage,
            "payload" to payload
        )
    }




    // =========================================================================
    // User input canonicalization (source classification)
    // =========================================================================

    /**
     * USER_INPUT_CLASSIFIED
     * -----------------------------------------------------------------------
     * Meaning:
     * - Declares what kind of input caused the policy call:
     *   voice / text / ui_event / synthetic, plus normalized text.
     *
     * Audit benefit:
     * - Removes confusion around "[EVENT] pending_* raw='...'" wrappers.
     */
    fun emitUserInputClassified(
        inputId: String,
        turnId: Long?,
        rowId: Int?,
        source: String, // "voice", "text", "ui_event", "synthetic"
        rawText: String?,
        normalizedText: String?,
        pendingType: String?,
        pendingIdx: Int?,
        stateHeaderPreview: String? = null
    ) {
        val raw = rawText?.let { capTextStable(it, USER_TEXT_CAP_CHARS) }
        val norm = normalizedText?.let { capTextStable(it, USER_TEXT_CAP_CHARS) }
        val sh = stateHeaderPreview?.let { capTextStable(it, 260) }

        emitKv(
            "USER_INPUT_CLASSIFIED",
            "conversation_id" to conversationId,
            "input_id" to inputId,
            "turn_id" to turnId,
            "row_id" to rowId,
            "source" to source,
            "raw_text_len" to raw?.fullLen,
            "raw_text" to raw?.preview,
            "raw_text_truncated" to raw?.truncated,
            "normalized_text_len" to norm?.fullLen,
            "normalized_text" to norm?.preview,
            "normalized_text_truncated" to norm?.truncated,
            "pending_type" to pendingType,
            "pending_idx" to pendingIdx,
            "state_header_preview_len" to sh?.fullLen,
            "state_header_preview" to sh?.preview,
            "state_header_preview_truncated" to sh?.truncated
        )
    }

    // =========================================================================
    // Assistant output canonical events: displayed vs spoken
    // =========================================================================

    /**
     * EVT_ASSISTANT_DISPLAYED
     * -----------------------------------------------------------------------
     * Meaning:
     * - The UI has displayed the assistant message (not just committed).
     *
     * Audit benefit:
     * - “Committed” != “shown”. This event makes UI-visible truth explicit.
     */
    fun emitAssistantDisplayed(
        turnId: Long?,
        assistantMessageId: String,
        uiMessageId: String? = null,
        text: String,
        tickId: Int? = null,
        modelCallId: String? = null
    ) {
        val c = capTextStable(text, ASSISTANT_TEXT_CAP_CHARS)
        emitKv(
            "EVT_ASSISTANT_DISPLAYED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "ui_message_id" to uiMessageId,
            "tick_id" to tickId,
            "model_call_id" to modelCallId,
            "text_len" to c.fullLen,
            "text_hash" to sha256HexUtf8(text),
            "text" to c.preview,
            "truncated" to c.truncated
        )
    }

    /**
     * EVT_ASSISTANT_SPEAK_ENQUEUED / STARTED / DONE
     * -----------------------------------------------------------------------
     * Meaning:
     * - Enqueued: your app decided to speak some text (before TTS fetch/start)
     * - Started: audio playback actually began
     * - Done: playback finished
     *
     * Audit benefit:
     * - Distinguishes intent-to-speak from actual audio playback.
     */
    fun emitAssistantSpeakEnqueued(
        turnId: Long?,
        assistantMessageId: String,
        ttsId: Int?,
        speakReqId: Int?,
        text: String,
        listenAfter: Boolean?,
        engine: String? = null,
        locale: String? = null
    ) {
        val c = capTextStable(text, ASSISTANT_TEXT_CAP_CHARS)
        emitKv(
            "EVT_ASSISTANT_SPEAK_ENQUEUED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "speak_req_id" to speakReqId,
            "engine" to engine,
            "locale" to locale,
            "listen_after" to listenAfter,
            "text_len" to c.fullLen,
            "text_hash" to sha256HexUtf8(text),
            "text" to c.preview,
            "truncated" to c.truncated
        )
    }

    fun emitAssistantSpeakStarted(
        turnId: Long?,
        assistantMessageId: String,
        ttsId: Int?,
        speakReqId: Int?,
        engine: String? = null,
        locale: String? = null
    ) {
        emitKv(
            "EVT_ASSISTANT_SPEAK_STARTED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "speak_req_id" to speakReqId,
            "engine" to engine,
            "locale" to locale
        )
    }

    fun emitAssistantSpeakDone(
        turnId: Long?,
        assistantMessageId: String,
        ttsId: Int?,
        speakReqId: Int?,
        durationMs: Long?,
        ok: Boolean = true,
        errorCode: String? = null
    ) {
        emitKv(
            "EVT_ASSISTANT_SPEAK_DONE",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "speak_req_id" to speakReqId,
            "duration_ms" to durationMs,
            "ok" to ok,
            "error_code" to errorCode
        )
    }

    /**
     * EVT_TTS_SPOKEN
     * -----------------------------------------------------------------------
     * Meaning:
     * - A “canonical” summary that the TTS utterance was successfully spoken.
     * - Use this even if you already emit engine-specific TTS_* events.
     *
     * Audit benefit:
     * - One event that all engines can produce uniformly.
     */
    fun emitTtsSpoken(
        turnId: Long?,
        assistantMessageId: String,
        ttsId: Int?,
        engine: String?,
        voice: String?,
        locale: String?,
        textHash: String?,
        durationMs: Long?,
        cacheHit: Boolean?
    ) {
        emitKv(
            "EVT_TTS_SPOKEN",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "engine" to engine,
            "voice" to voice,
            "locale" to locale,
            "text_hash" to textHash,
            "duration_ms" to durationMs,
            "cache_hit" to cacheHit
        )
    }

    /**
     * ASSISTANT_OUTPUT_CANONICAL
     * -----------------------------------------------------------------------
     * Meaning:
     * - Records whether UI-displayed text == TTS-spoken text (hash compare).
     *
     * Audit benefit:
     * - Instantly detects “UI said X, voice said Y”.
     */
    fun emitAssistantOutputCanonical(
        turnId: Long?,
        assistantMessageId: String,
        uiTextHash: String?,
        ttsTextHash: String?,
        uiLen: Int?,
        ttsLen: Int?
    ) {
        emitKv(
            "ASSISTANT_OUTPUT_CANONICAL",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "ui_text_hash" to uiTextHash,
            "tts_text_hash" to ttsTextHash,
            "same_text" to (uiTextHash != null && uiTextHash == ttsTextHash),
            "ui_len" to uiLen,
            "tts_len" to ttsLen
        )
    }

    /**
     * ASSISTANT_TEXT_REWRITE
     * -----------------------------------------------------------------------
     * Meaning:
     * - Captures when you rewrite/sanitize the assistant’s reply before user sees it.
     *
     * Examples:
     * - unique-stop rewrite: ask_confirm -> recommend_validate prompt
     * - validator rewrite: replace control tool target
     *
     * Audit benefit:
     * - Offline tool can display “model output” vs “final output” and explain why.
     */
    fun emitAssistantTextRewrite(
        turnId: Long?,
        tickId: Int?,
        modelCallId: String?,
        originalText: String?,
        finalText: String,
        rewriteReason: String,
        rewriteRuleId: String? = null
    ) {
        val origHash = originalText?.let { sha256HexUtf8(it) }
        val finHash = sha256HexUtf8(finalText)
        emitKv(
            "ASSISTANT_TEXT_REWRITE",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "model_call_id" to modelCallId,
            "rewrite_reason" to rewriteReason,
            "rewrite_rule_id" to rewriteRuleId,
            "original_text_hash" to origHash,
            "original_len" to (originalText?.length ?: 0),
            "final_text_hash" to finHash,
            "final_len" to finalText.length
        )
    }

    // =========================================================================
    // Toolplan / tool results canonical events
    // =========================================================================

    /**
     * TOOLPLAN_DECISION
     * -----------------------------------------------------------------------
     * Meaning:
     * - Single canonical record: accepted / rewritten / rejected toolplan.
     *
     * Audit benefit:
     * - Offline tool doesn’t need to infer from scattered “validator rewrite” events.
     */
    fun emitToolplanDecision(
        toolplanId: String,
        turnId: Long?,
        tickId: Int,
        modelCallId: String?,
        correlationId: String,
        status: String, // "accepted" | "rewritten" | "rejected"
        reasons: List<String>?,
        inTools: List<String>?,
        outTools: List<String>?
    ) {
        emitKv(
            "TOOLPLAN_DECISION",
            "conversation_id" to conversationId,
            "toolplan_id" to toolplanId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "model_call_id" to modelCallId,
            "correlation_id" to correlationId,
            "status" to status,
            "reasons" to reasons,
            "in_tools" to inTools,
            "out_tools" to outTools
        )
    }

    /**
     * TOOL_RESULT
     * -----------------------------------------------------------------------
     * Meaning:
     * - A stable, engine-agnostic record of one tool execution result.
     *
     * Audit benefit:
     * - tick2 can reference tool_result_id[] explicitly; no preview-string parsing.
     */
    fun emitToolResult(
        toolResultId: String,
        turnId: Long?,
        tickId: Int?,
        toolName: String,
        argsHash: String? = null,
        success: Boolean,
        resultSummary: String? = null,
        errorCode: String? = null
    ) {
        val c = resultSummary?.let { capTextStable(it, 220) }
        emitKv(
            "TOOL_RESULT",
            "conversation_id" to conversationId,
            "tool_result_id" to toolResultId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "tool_name" to toolName,
            "args_hash" to argsHash,
            "success" to success,
            "result_summary_len" to c?.fullLen,
            "result_summary" to c?.preview,
            "result_summary_truncated" to c?.truncated,
            "error_code" to errorCode
        )
    }

    /**
     * TOOL_HANDLER_RESULT (JOINABLE)
     * -----------------------------------------------------------------------
     * Minimum join keys:
     * - session_id, turn_id, tick_id, policy_req_seq, toolplan_id
     * - tool_call_id, tool_result_id
     */
    fun emitToolHandlerResult(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        policyReqSeq: Long,
        toolplanId: String,
        toolCallId: String,
        toolResultId: String,
        toolName: String,
        status: String, // "ok" | "noop" | "rejected" | "error"
        errorCode: String? = null,
        errorMsgShort: String? = null
    ) {
        emitKv(
            "TOOL_HANDLER_RESULT",
            "conversation_id" to conversationId,
            "session_id" to sessionId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "policy_req_seq" to policyReqSeq,
            "toolplan_id" to toolplanId,
            "tool_call_id" to toolCallId,
            "tool_result_id" to toolResultId,
            "tool_name" to toolName,
            "status" to status,
            "error_code" to errorCode,
            "error_msg" to errorMsgShort
        )
    }

    /**
     * TOOLS_EXECUTION_SUMMARY
     * -----------------------------------------------------------------------
     * Meaning:
     * - Summarizes what was planned vs executed vs skipped.
     *
     * Audit benefit:
     * - Makes “validator removed X” explicit.
     */
    fun emitToolsExecutionSummary(
        turnId: Long?,
        tickId: Int?,
        plannedTools: List<String>?,
        executedTools: List<String>?,
        skippedTools: List<String>?,
        skipReasons: List<String>?
    ) {
        emitKv(
            "TOOLS_EXECUTION_SUMMARY",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "planned_tools" to plannedTools,
            "executed_tools" to executedTools,
            "skipped_tools" to skippedTools,
            "skip_reasons" to skipReasons
        )
    }

    /**
     * CONTROL_TARGET_CHANGED
     * -----------------------------------------------------------------------
     * Meaning:
     * - Canonical record that control tool target changed (e.g., r1c1 -> r9c8).
     *
     * Audit benefit:
     * - Explains “why did you ask that cell again?”.
     */
    fun emitControlTargetChanged(
        turnId: Long?,
        tickId: Int?,
        requestedTarget: String?,
        finalTarget: String,
        ruleId: String,
        allowedTargets: List<String>?
    ) {
        emitKv(
            "CONTROL_TARGET_CHANGED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "requested_target" to requestedTarget,
            "final_target" to finalTarget,
            "rule_id" to ruleId,
            "allowed_targets" to allowedTargets
        )
    }

    /**
     * UNIQUE_STOP_APPLIED
     * -----------------------------------------------------------------------
     * Meaning:
     * - Unique-stop (validate gating) was applied and what it replaced.
     *
     * Audit benefit:
     * - Makes this policy decision auditable and defensible (not “random behavior”).
     */
    fun emitUniqueStopApplied(
        turnId: Long?,
        tickId: Int?,
        solvability: String?,
        mismatchCount: Int?,
        unresolvedCount: Int?,
        controlIn: String?,
        controlOut: String,
        ruleId: String
    ) {
        emitKv(
            "UNIQUE_STOP_APPLIED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "solvability" to solvability,
            "mismatch_count" to mismatchCount,
            "unresolved_count" to unresolvedCount,
            "control_in" to controlIn,
            "control_out" to controlOut,
            "rule_id" to ruleId
        )
    }

    // =========================================================================
    // State machine transitions (pending / phase)
    // =========================================================================

    /**
     * PENDING_TRANSITION
     * -----------------------------------------------------------------------
     * Meaning:
     * - Records before/after pending state transitions.
     *
     * Audit benefit:
     * - Allows clean state machine visualization without reading state headers.
     */
    fun emitPendingTransition(
        turnId: Long?,
        tickId: Int?,
        pendingBefore: String?,
        pendingAfter: String?,
        trigger: String, // "tool", "validator", "policy", etc.
        note: String? = null
    ) {
        emitKv(
            "PENDING_TRANSITION",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "pending_before" to pendingBefore,
            "pending_after" to pendingAfter,
            "trigger" to trigger,
            "note" to note
        )
    }

    /**
     * PHASE_TRANSITION
     * -----------------------------------------------------------------------
     * Meaning:
     * - Records before/after phase transitions (even if unchanged, for clarity).
     *
     * Audit benefit:
     * - Prevents silent transitions across code paths.
     */
    fun emitPhaseTransition(
        turnId: Long?,
        tickId: Int?,
        phaseBefore: String?,
        phaseAfter: String?,
        reason: String
    ) {
        emitKv(
            "PHASE_TRANSITION",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "phase_before" to phaseBefore,
            "phase_after" to phaseAfter,
            "reason" to reason
        )
    }

    // =========================================================================
    // Continuation guard evaluation (Design A “don’t repeat”)
    // =========================================================================

    /**
     * CONTINUATION_GUARD_EVAL
     * -----------------------------------------------------------------------
     * Meaning:
     * - Evaluates a guard rule for continuation tick (e.g., "do not restate applied edit").
     *
     * Audit benefit:
     * - Enables automatic scoring of “Design A helped” over many sessions.
     */
    fun emitContinuationGuardEval(
        turnId: Long?,
        tickId: Int,
        guardName: String,
        passed: Boolean,
        violationType: String? = null,
        evidence: String? = null
    ) {
        val e = evidence?.let { capTextStable(it, 260) }
        emitKv(
            "CONTINUATION_GUARD_EVAL",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "guard_name" to guardName,
            "passed" to passed,
            "violation_type" to violationType,
            "evidence_len" to e?.fullLen,
            "evidence" to e?.preview,
            "evidence_truncated" to e?.truncated
        )
    }

    // =========================================================================
    // Listen-after semantics (turn-taking determinism)
    // =========================================================================

    /**
     * LISTEN_AFTER_DECISION / OUTCOME
     * -----------------------------------------------------------------------
     * Meaning:
     * - Decision: requested listen-after, scheduling parameters, why
     * - Outcome: whether listening actually started, or was blocked
     *
     * Audit benefit:
     * - Makes deterministic turn-taking measurable.
     */
    fun emitListenAfterDecision(
        turnId: Long?,
        assistantMessageId: String?,
        requested: Boolean,
        delayMs: Long?,
        reason: String
    ) {
        emitKv(
            "LISTEN_AFTER_DECISION",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "requested" to requested,
            "delay_ms" to delayMs,
            "reason" to reason
        )
    }

    fun emitListenAfterOutcome(
        turnId: Long?,
        assistantMessageId: String?,
        started: Boolean,
        asrRowId: Int?,
        blockedReason: String? = null
    ) {
        emitKv(
            "LISTEN_AFTER_OUTCOME",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "started" to started,
            "asr_row_id" to asrRowId,
            "blocked_reason" to blockedReason
        )
    }

    // =========================================================================
    // ASR suppression window (overlap debugging)
    // =========================================================================

    /**
     * ASR_SUPPRESSION_WINDOW
     * -----------------------------------------------------------------------
     * Meaning:
     * - A single record for suppression start/end (rather than scattered logs).
     *
     * Audit benefit:
     * - Lets tooling compute overlap windows precisely without scanning many events.
     */
    fun emitAsrSuppressionWindow(
        suppressionWindowId: String,
        turnId: Long?,
        reason: String,
        startTsEpochMs: Long,
        endTsEpochMs: Long,
        ttsId: Int? = null
    ) {
        emitKv(
            "ASR_SUPPRESSION_WINDOW",
            "conversation_id" to conversationId,
            "suppression_window_id" to suppressionWindowId,
            "turn_id" to turnId,
            "reason" to reason,
            "start_ts_epoch_ms" to startTsEpochMs,
            "end_ts_epoch_ms" to endTsEpochMs,
            "tts_id" to ttsId
        )
    }

    // =========================================================================
    // UI bars binding to utterance (no guessing)
    // =========================================================================

    /**
     * UI_BARS_BOUND_TO_UTTERANCE
     * -----------------------------------------------------------------------
     * Meaning:
     * - Binds UI bars instance to a specific utterance.
     *
     * Audit benefit:
     * - Prevents “bars started but which TTS was that?” confusion.
     */
    fun emitUiBarsBoundToUtterance(
        uiBarsInstanceId: String,
        turnId: Long?,
        assistantMessageId: String?,
        ttsId: Int?,
        startTsEpochMs: Long,
        stopTsEpochMs: Long?
    ) {
        emitKv(
            "UI_BARS_BOUND_TO_UTTERANCE",
            "conversation_id" to conversationId,
            "ui_bars_instance_id" to uiBarsInstanceId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "start_ts_epoch_ms" to startTsEpochMs,
            "stop_ts_epoch_ms" to stopTsEpochMs
        )
    }

    // =========================================================================
    // Audio focus context (ties to TTS)
    // =========================================================================

    /**
     * AUDIO_FOCUS_CONTEXT
     * -----------------------------------------------------------------------
     * Meaning:
     * - Captures the audio focus lifecycle in a uniform record, tied to utterance.
     *
     * Audit benefit:
     * - Diagnoses playback failures without reading engine-specific logs.
     */
    fun emitAudioFocusContext(
        turnId: Long?,
        assistantMessageId: String?,
        ttsId: Int?,
        usage: String?,
        granted: Boolean?,
        latencyMs: Long?,
        abandonReason: String? = null
    ) {
        emitKv(
            "AUDIO_FOCUS_CONTEXT",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "usage" to usage,
            "granted" to granted,
            "latency_ms" to latencyMs,
            "abandon_reason" to abandonReason
        )
    }

    // =========================================================================
    // Turn completion (logic vs UI vs audio)
    // =========================================================================

    /**
     * TURN_DONE_LOGIC / TURN_DONE_UI / TURN_DONE_AUDIO
     * -----------------------------------------------------------------------
     * Meaning:
     * - Logic: toolplan resolved + state updated
     * - UI: message displayed (or UI “done”)
     * - Audio: TTS playback done (if any)
     *
     * Audit benefit:
     * - Eliminates the #1 voice app ambiguity: “turn done” means what, exactly?
     */
    fun emitTurnDoneLogic(turnId: Long, note: String? = null) {
        emitKv(
            "TURN_DONE_LOGIC",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "note" to note
        )
    }

    fun emitTurnDoneUi(turnId: Long, assistantMessageId: String?, note: String? = null) {
        emitKv(
            "TURN_DONE_UI",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "note" to note
        )
    }

    fun emitTurnDoneAudio(turnId: Long, assistantMessageId: String?, ttsId: Int?, note: String? = null) {
        emitKv(
            "TURN_DONE_AUDIO",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "note" to note
        )
    }

    // =========================================================================
    // Backward-compatible events you already had (enhanced with hashes/ids)
    // =========================================================================

    /**
     * ASSISTANT_SAY (backward compatible)
     * -----------------------------------------------------------------------
     * Meaning:
     * - What the app is about to say (pre TTS start).
     *
     * Added fields:
     * - assistant_message_id (optional)
     * - text_hash
     * - tick_id / model_call_id (optional) for perfect traceability
     *
     * Audit benefit:
     * - Provides a stable text hash for matching with UI/TTS events.
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
        replyToRowId: Int? = null,
        // New optional canonical fields:
        assistantMessageId: String? = null,
        tickId: Int? = null,
        modelCallId: String? = null
    ) {
        val c = capTextStable(text, ASSISTANT_TEXT_CAP_CHARS)

        // 6.2: Register pending speak so completion (TTS_DONE) can emit ASSISTANT_REPLY_SPOKEN.
        // NOTE: We only register here; we do NOT emit SPOKEN here.
        runCatching {
            val eng = engine ?: "Unknown"
            val replySha = sha256HexUtf8(text)

            registerPendingSpeak(
                engine = eng,
                assistantMessageId = assistantMessageId,
                turnId = turnId?.toLong(),
                tickId = tickId,                 // ✅ now supported
                ttsId = ttsId,
                speakReqId = speakReqId,
                replyToRowId = replyToRowId,
                utteranceId = utteranceId,
                voice = null,
                locale = locale,
                replyTextSha256 = replySha,      // ✅ now supported
                uiTextSha256 = null,             // fill if you have UI-specific string
                text = text
            )
        }

        emitKv(
            "ASSISTANT_SAY",
            "conversation_id" to conversationId,
            "app_session_id" to appSessionId,
            "source" to source,
            "engine" to engine,
            "locale" to locale,
            "turn_id" to turnId,
            "conv_turn" to convTurn,
            "tick_id" to tickId,
            "model_call_id" to modelCallId,
            // pairing
            "speak_req_id" to speakReqId,
            "tts_id" to ttsId,
            "utterance_id" to utteranceId,
            "reply_to_row_id" to replyToRowId,
            // canonical msg id
            "assistant_message_id" to assistantMessageId,
            "text_len" to c.fullLen,
            "text_hash" to sha256HexUtf8(text),
            "text" to c.preview,
            "truncated" to c.truncated
        )
    }

    /**
     * USER_SAY (backward compatible)
     * -----------------------------------------------------------------------
     * Meaning:
     * - The ASR final text that your turn controller accepted.
     *
     * Added fields:
     * - text_hash
     *
     * Audit benefit:
     * - Stable matching with pending events / policy calls without guessing.
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
            "conversation_id" to conversationId,
            "source" to source,
            "row_id" to rowId,
            "confidence" to confidence,
            "turn_id" to turnId,
            "conv_turn" to convTurn,
            "text_len" to c.fullLen,
            "text_hash" to sha256HexUtf8(text),
            "text" to c.preview,
            "truncated" to c.truncated
        )
    }

    /**
     * TURN_PAIR (existing)
     * -----------------------------------------------------------------------
     * Note:
     * - Keep this, but prefer TURN_PAIR_RESOLVED when you know turn_id and message ids.
     */
    fun emitTurnPair(rowId: Int, speakReqId: Int, convTurn: Int? = null) {
        emitKv(
            "TURN_PAIR",
            "conversation_id" to conversationId,
            "row_id" to rowId,
            "speak_req_id" to speakReqId,
            "conv_turn" to convTurn
        )
    }

    /**
     * TURN_PAIR_RESOLVED
     * -----------------------------------------------------------------------
     * Meaning:
     * - Canonical pairing record (no “conv_turn=null” ambiguity).
     *
     * Audit benefit:
     * - Perfect joining between user row and assistant message/tts.
     */
    fun emitTurnPairResolved(
        rowId: Int,
        turnId: Long,
        assistantMessageId: String?,
        ttsId: Int?,
        pairMethod: String, // "direct", "heuristic", "controller"
        pairConfidence: Float? = null
    ) {
        emitKv(
            "TURN_PAIR_RESOLVED",
            "conversation_id" to conversationId,
            "row_id" to rowId,
            "turn_id" to turnId,
            "assistant_message_id" to assistantMessageId,
            "tts_id" to ttsId,
            "pair_method" to pairMethod,
            "pair_confidence" to pairConfidence
        )
    }

    // --- Phase 0: JSONL safety -------------------------------------------------

    private fun safeJsonValue(v: Any?): Any? = when (v) {
        null -> JSONObject.NULL
        is JSONObject -> v
        is JSONArray -> v
        is Boolean, is Int, is Long, is Float, is Double -> v
        is Number -> v.toDouble()
        is String -> sanitizeStringForJsonl(v)
        is Map<*, *> -> {
            val obj = JSONObject()
            for ((k, value) in v) {
                val key = sanitizeStringForJsonl(k?.toString() ?: "null")
                obj.put(key, safeJsonValue(value))
            }
            obj
        }
        is List<*> -> {
            val arr = JSONArray()
            for (x in v) arr.put(safeJsonValue(x))
            arr
        }
        is Array<*> -> {
            val arr = JSONArray()
            for (x in v) arr.put(safeJsonValue(x))
            arr
        }
        is IntArray -> {
            val arr = JSONArray()
            for (x in v) arr.put(x)
            arr
        }
        is LongArray -> {
            val arr = JSONArray()
            for (x in v) arr.put(x)
            arr
        }
        else -> sanitizeStringForJsonl(v.toString())
    }

    private fun buildEventJson(type: String, vararg kv: Pair<String, Any?>): String {
        val obj = JSONObject()
        obj.put("type", type)
        obj.put("schema_version", TELEMETRY_SCHEMA_VERSION)

        // Always include these two; they’re invaluable for forensic ordering.
        obj.put("telemetry_id", telemetryId)
        obj.put("seq", seq.incrementAndGet())
        obj.put("ts_epoch_ms", System.currentTimeMillis())
        obj.put("ts_monotonic_ms", SystemClock.elapsedRealtime())

        // Context keys (stable join space)
        obj.put("app_session_id", appSessionId)
        obj.put("conversation_id", conversationId)

        for ((k, v) in kv) {
            obj.put(sanitizeStringForJsonl(k), safeJsonValue(v))
        }

        // IMPORTANT: JSONObject.toString() will escape control chars,
        // but we *also* guarantee the final line has no literal \n or \r.
        return sanitizeStringForJsonl(obj.toString())
    }



    // =========================================================================
    // PATCH 5 — Agenda / CTA / Ops audit events
    // =========================================================================

    /**
     * AGENDA_SELECTED
     * - Emitted when the planner selects the next app agenda head to drive the turn.
     */
    fun emitAgendaSelected(
        turnId: Long,
        tickId: Int,
        agendaType: String,
        agendaReason: String? = null,
        appQueueSize: Int? = null,
        userQueueSize: Int? = null,
        phase: String? = null
    ) {
        emitKv(
            "AGENDA_SELECTED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "agenda_type" to agendaType,
            "agenda_reason" to agendaReason,
            "app_queue_size" to appQueueSize,
            "user_queue_size" to userQueueSize,
            "phase" to phase
        )
    }

    /**
     * CTA_SELECTED
     * - Emitted when a Pending CTA is set for Tick2.
     *
     * North Star add-ons:
     * - primary_cta_id: canonical CTA family expected for this pending
     * - options: raw pending options emitted to Tick2/UI
     * - options_count: option count for quick purity checks
     * - is_north_star_single: whether the pending exposes exactly one option
     */
    fun emitCtaSelected(
        turnId: Long,
        tickId: Int,
        pendingType: String,
        ctaName: String,
        expectedAnswerKind: String? = null,
        primaryCtaId: String? = null,
        options: List<String> = emptyList(),
        isNorthStarSingle: Boolean? = null
    ) {
        emitKv(
            "CTA_SELECTED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "pending_type" to pendingType,
            "cta_name" to ctaName,
            "expected_answer_kind" to expectedAnswerKind,
            "primary_cta_id" to primaryCtaId,
            "options" to options,
            "options_count" to options.size,
            "is_north_star_single" to isNorthStarSingle
        )
    }

    /**
     * AGENDA_TRANSITION
     * - Emitted when planner consumes/pushes agenda items during Tick1 planning.
     */
    fun emitAgendaTransition(
        turnId: Long,
        tickId: Int,
        consumedCount: Int,
        pushedCount: Int,
        consumedTypes: List<String> = emptyList(),
        pushedTypes: List<String> = emptyList(),
        appQueueSizeAfter: Int? = null
    ) {
        emitKv(
            "AGENDA_TRANSITION",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "consumed_count" to consumedCount,
            "pushed_count" to pushedCount,
            "consumed_types" to consumedTypes,
            "pushed_types" to pushedTypes,
            "app_queue_size_after" to appQueueSizeAfter
        )
    }

    /**
     * OPS_PLANNED
     * - Emitted with a compact summary of planned ops for the turn.
     */
    fun emitOpsPlanned(
        turnId: Long,
        tickId: Int,
        ops: List<String>
    ) {
        emitKv(
            "OPS_PLANNED",
            "conversation_id" to conversationId,
            "turn_id" to turnId,
            "tick_id" to tickId,
            "ops" to ops,
            "ops_count" to ops.size
        )
    }



    // =========================================================================
    // POLICY TRACE helpers (keep, but now add canonical fields when possible)
    // =========================================================================

    /**
     * POLICY_TRACE
     * -----------------------------------------------------------------------
     * Meaning:
     * - Structured debug trace for policy logic.
     *
     * Audit benefit:
     * - Gives you deep breadcrumbs, but canonical events are still needed
     *   because traces are not guaranteed to exist in every path.
     */
    fun emitPolicyTrace(tag: String, data: Map<String, Any?>) {
        val payload = linkedMapOf<String, Any?>(
            "type" to "POLICY_TRACE",
            "tag" to tag,
            "conversation_id" to conversationId
        )
        data.forEach { (k, v) ->
            payload[k] = when (v) {
                is String -> sanitizeStringForJsonl(v)
                else -> v
            }
        }
        emit(payload)
    }

    fun emitCtaPolicyTrace(
        turnId: Long?,
        tickId: Int? = 2,
        ownerKind: String?,
        phase: String?,
        storyStage: String?,
        pendingAfter: String?,
        ctaFamily: String?,
        ctaRouteMoment: String?,
        ctaExpectedResponseType: String?,
        ctaAskMode: String?,
        ctaFocusCellRef: String?,
        ctaFocusHouseRef: String?,
        ctaFocusDigit: Int?,
        ctaTechniqueName: String?,
        ctaAllowFollowUp: Boolean?,
        ctaAllowReturnToRoute: Boolean?,
        ctaAllowRouteMutation: Boolean?,
        ctaMustOfferFollowUpChoice: Boolean?,
        ctaMustOfferReturnChoice: Boolean?,
        ctaMustNotAdvanceStage: Boolean?,
        ctaMustReferenceFocusScope: Boolean?,
        ctaAllowInternalJargon: Boolean?,
        ctaBannedPhrases: List<String> = emptyList()
    ) {
        emit(
            linkedMapOf(
                "type" to "CTA_POLICY_TRACE",
                "conversation_id" to conversationId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "owner_kind" to ownerKind,
                "phase" to phase,
                "story_stage" to storyStage,
                "pending_after" to pendingAfter,
                "cta_family" to ctaFamily,
                "cta_route_moment" to ctaRouteMoment,
                "cta_expected_response_type" to ctaExpectedResponseType,
                "cta_ask_mode" to ctaAskMode,
                "cta_focus_cell_ref" to ctaFocusCellRef,
                "cta_focus_house_ref" to ctaFocusHouseRef,
                "cta_focus_digit" to ctaFocusDigit,
                "cta_technique_name" to ctaTechniqueName,
                "cta_allow_followup" to ctaAllowFollowUp,
                "cta_allow_return_to_route" to ctaAllowReturnToRoute,
                "cta_allow_route_mutation" to ctaAllowRouteMutation,
                "cta_must_offer_followup_choice" to ctaMustOfferFollowUpChoice,
                "cta_must_offer_return_choice" to ctaMustOfferReturnChoice,
                "cta_must_not_advance_stage" to ctaMustNotAdvanceStage,
                "cta_must_reference_focus_scope" to ctaMustReferenceFocusScope,
                "cta_allow_internal_jargon" to ctaAllowInternalJargon,
                "cta_banned_phrases" to ctaBannedPhrases
            )
        )
    }

    fun emitCtaRenderAudit(
        turnId: Long?,
        tickId: Int? = 2,
        ownerKind: String?,
        phase: String?,
        storyStage: String?,
        assistantText: String,
        bannedPhrases: List<String> = emptyList()
    ) {
        val preview = capTextStable(assistantText, ASSISTANT_TEXT_CAP_CHARS)
        val lower = assistantText.lowercase(Locale.ROOT)

        val bannedFound = bannedPhrases.filter { phrase ->
            phrase.isNotBlank() && lower.contains(phrase.lowercase(Locale.ROOT))
        }.distinct()

        val hasBinaryChoiceSignal =
            Regex("""\b(or|either)\b""", RegexOption.IGNORE_CASE).containsMatchIn(assistantText) ||
                    assistantText.contains(" or ")

        val hasQuestionMark = assistantText.contains("?")

        val hasReturnChoiceSignal =
            Regex("""\b(return|go back|pick up where we left off|continue with (the )?(move|step)|continue there)\b""", RegexOption.IGNORE_CASE)
                .containsMatchIn(assistantText)

        val hasFollowUpChoiceSignal =
            Regex("""\b(one more question|another question|ask more|ask one more|stay on this|keep exploring)\b""", RegexOption.IGNORE_CASE)
                .containsMatchIn(assistantText)

        val hasCommitLanguage =
            Regex("""\b(place|fill|put|enter|write)\b""", RegexOption.IGNORE_CASE)
                .containsMatchIn(assistantText)

        val hasDiscoveryLanguage =
            Regex("""\b(which digit|which cell|what do you notice|what remains|what is left)\b""", RegexOption.IGNORE_CASE)
                .containsMatchIn(assistantText)

        val stageFitGuess =
            when (storyStage?.uppercase(Locale.ROOT)) {
                "SETUP" ->
                    when {
                        hasCommitLanguage -> "LIKELY_STAGE_MISMATCH_SETUP_SOUNDS_LIKE_COMMIT"
                        hasDiscoveryLanguage -> "LIKELY_STAGE_FIT_SETUP_DISCOVERY"
                        else -> "UNKNOWN_STAGE_FIT_SETUP"
                    }

                "CONFRONTATION" ->
                    when {
                        hasCommitLanguage -> "POSSIBLE_STAGE_DRIFT_CONFRONTATION_TO_COMMIT"
                        else -> "UNKNOWN_STAGE_FIT_CONFRONTATION"
                    }

                "RESOLUTION" ->
                    when {
                        hasCommitLanguage -> "LIKELY_STAGE_FIT_RESOLUTION_COMMIT"
                        else -> "UNKNOWN_STAGE_FIT_RESOLUTION"
                    }

                else -> "UNKNOWN_STAGE"
            }

        emit(
            linkedMapOf(
                "type" to "CTA_RENDER_AUDIT",
                "conversation_id" to conversationId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "owner_kind" to ownerKind,
                "phase" to phase,
                "story_stage" to storyStage,
                "assistant_text_len" to preview.fullLen,
                "assistant_text_sha256" to sha256Hex(assistantText),
                "assistant_text_preview" to preview.preview,
                "cta_rendered_contains_banned_phrase" to bannedFound.isNotEmpty(),
                "cta_rendered_banned_phrases_found" to bannedFound,
                "cta_rendered_has_binary_choice_signal" to hasBinaryChoiceSignal,
                "cta_rendered_has_question_mark" to hasQuestionMark,
                "cta_rendered_return_choice_guess" to hasReturnChoiceSignal,
                "cta_rendered_followup_choice_guess" to hasFollowUpChoiceSignal,
                "cta_rendered_commit_language_guess" to hasCommitLanguage,
                "cta_rendered_discovery_language_guess" to hasDiscoveryLanguage,
                "cta_rendered_owner_stage_fit_guess" to stageFitGuess
            )
        )
    }

    /**
     * Phase 1 — canonical reply-demand routing breadcrumb.
     *
     * This is intentionally lightweight and non-invasive:
     * it does not change behavior, only records which demand category the
     * conductor believes Tick2 is serving on this turn.
     */
    fun emitReplyDemandResolved(
        turnId: Long?,
        tickId: Int? = 2,
        category: String,
        reason: String,
        phase: String? = null,
        pendingKind: String? = null,
        storyStage: String? = null,
        openingTurn: Boolean? = null
    ) {
        emit(
            mapOf(
                "type" to "REPLY_DEMAND_RESOLVED",
                "turn_id" to turnId,
                "tick_id" to tickId,
                "category" to category,
                "reason" to reason,
                "phase" to phase,
                "pending_kind" to pendingKind,
                "story_stage" to storyStage,
                "opening_turn" to openingTurn
            )
        )
    }


    /**
     * Phase 8 — budget / waste audit for a Tick2 reply request.
     *
     * This answers:
     * - what was needed
     * - what was selected
     * - what was excluded
     * - how big each supplied channel was
     * - whether the request exceeded its budget
     * - whether any guardrails fired
     */
    fun emitReplyWasteAudit(
        turnId: Long?,
        tickId: Int? = 2,
        demandCategory: String,
        rolloutMode: String? = null,
        selectedPromptModules: List<String>,
        selectedChannels: List<String>,
        forbiddenChannels: List<String>,
        excludedChannels: List<String>,
        channelChars: Map<String, Int>,
        channelTokens: Map<String, Int>,
        totalDynamicChars: Int,
        totalDynamicTokens: Int,
        softCharBudget: Int? = null,
        softTokenBudget: Int? = null,
        charOverrun: Int = 0,
        tokenOverrun: Int = 0,
        warnings: List<String> = emptyList()
    ) {
        emit(
            mapOf(
                "type" to "REPLY_WASTE_AUDIT",
                "turn_id" to turnId,
                "tick_id" to tickId,
                "demand_category" to demandCategory,
                "rollout_mode" to rolloutMode,
                "selected_prompt_modules" to selectedPromptModules,
                "selected_channels" to selectedChannels,
                "forbidden_channels" to forbiddenChannels,
                "excluded_channels" to excludedChannels,
                "channel_chars" to channelChars,
                "channel_tokens" to channelTokens,
                "total_dynamic_chars" to totalDynamicChars,
                "total_dynamic_tokens" to totalDynamicTokens,
                "soft_char_budget" to softCharBudget,
                "soft_token_budget" to softTokenBudget,
                "char_overrun" to charOverrun,
                "token_overrun" to tokenOverrun,
                "warnings" to warnings
            )
        )
    }

    fun emitConfrontationReplyCoverage(
        turnId: Long?,
        tickId: Int? = 2,
        demandCategory: String,
        rolloutMode: String? = null,
        proofProfile: String? = null,
        packetLen: Int = 0,
        packetSha12: String? = null,
        packetSelected: Boolean = false,
        packetProjected: Boolean = false,
        hasTarget: Boolean = false,
        hasTriggerReference: Boolean = false,
        hasTriggerEffect: Boolean = false,
        hasCollapse: Boolean = false,
        hasPreCommitLine: Boolean = false,
        hasCta: Boolean = false,
        rawProofRowCount: Int = 0,
        boundedProofRowCount: Int = 0,
        proofRowLimit: Int = 0,
        proofRowsTruncated: Boolean = false,
        supportLen: Int = 0,
        actorStructure: String? = null,
        ordinaryWitnessFirstRequired: Boolean = false,
        techniqueFinishingCutRequired: Boolean = false,
        hasTargetSpotlightLine: Boolean = false,
        hasSurvivorRevealLine: Boolean = false,
        ladderFirstStepKind: String? = null,
        ladderSecondStepKind: String? = null,
        overlayVariant: String? = null,
        fallbackReason: String? = null
    ) {
        emit(
            mapOf(
                "type" to "CONFRONTATION_REPLY_COVERAGE",
                "turn_id" to turnId,
                "tick_id" to tickId,
                "demand_category" to demandCategory,
                "rollout_mode" to rolloutMode,
                "confrontation_proof_profile" to proofProfile,
                "confrontation_packet_len" to packetLen,
                "confrontation_packet_sha12" to packetSha12,
                "confrontation_packet_selected" to packetSelected,
                "confrontation_packet_projected" to packetProjected,
                "confrontation_packet_has_target" to hasTarget,
                "confrontation_packet_has_trigger_reference" to hasTriggerReference,
                "confrontation_packet_has_trigger_effect" to hasTriggerEffect,
                "confrontation_packet_has_collapse" to hasCollapse,
                "confrontation_packet_has_pre_commit_line" to hasPreCommitLine,
                "confrontation_packet_has_cta" to hasCta,
                "confrontation_packet_raw_proof_row_count" to rawProofRowCount,
                "confrontation_packet_bounded_proof_row_count" to boundedProofRowCount,
                "confrontation_packet_proof_row_limit" to proofRowLimit,
                "confrontation_packet_proof_rows_truncated" to proofRowsTruncated,
                "confrontation_support_total_len" to supportLen,
                "confrontation_actor_structure" to actorStructure,
                "confrontation_ordinary_witness_first_required" to ordinaryWitnessFirstRequired,
                "confrontation_technique_finishing_cut_required" to techniqueFinishingCutRequired,
                "confrontation_has_target_spotlight_line" to hasTargetSpotlightLine,
                "confrontation_has_survivor_reveal_line" to hasSurvivorRevealLine,
                "confrontation_ladder_first_step_kind" to ladderFirstStepKind,
                "confrontation_ladder_second_step_kind" to ladderSecondStepKind,
                "confrontation_overlay_variant" to overlayVariant,
                "confrontation_fallback_reason" to fallbackReason
            )
        )
    }

    fun emitResolutionReplyCoverage(
        turnId: Long?,
        tickId: Int? = 2,
        demandCategory: String,
        rolloutMode: String? = null,
        resolutionProfile: String? = null,
        packetLen: Int = 0,
        packetSha12: String? = null,
        packetSelected: Boolean = false,
        packetProjected: Boolean = false,
        hasCommit: Boolean = false,
        hasRecap: Boolean = false,
        hasTechniqueContribution: Boolean = false,
        hasFinalForcing: Boolean = false,
        hasHonesty: Boolean = false,
        hasPresentStateLine: Boolean = false,
        hasPostCommitSummary: Boolean = false,
        hasCta: Boolean = false,
        recapMaxBeats: Int = 0,
        compactMode: String? = null,
        supportLen: Int = 0,
        overlayVariant: String? = null,
        fallbackReason: String? = null
    ) {
        emit(
            mapOf(
                "type" to "RESOLUTION_REPLY_COVERAGE",
                "turn_id" to turnId,
                "tick_id" to tickId,
                "demand_category" to demandCategory,
                "rollout_mode" to rolloutMode,
                "resolution_profile" to resolutionProfile,
                "resolution_packet_len" to packetLen,
                "resolution_packet_sha12" to packetSha12,
                "resolution_packet_selected" to packetSelected,
                "resolution_packet_projected" to packetProjected,
                "resolution_packet_has_commit" to hasCommit,
                "resolution_packet_has_recap" to hasRecap,
                "resolution_packet_has_technique_contribution" to hasTechniqueContribution,
                "resolution_packet_has_final_forcing" to hasFinalForcing,
                "resolution_packet_has_honesty" to hasHonesty,
                "resolution_packet_has_present_state_line" to hasPresentStateLine,
                "resolution_packet_has_post_commit_summary" to hasPostCommitSummary,
                "resolution_packet_has_cta" to hasCta,
                "resolution_packet_recap_max_beats" to recapMaxBeats,
                "resolution_packet_compact_mode" to compactMode,
                "resolution_support_total_len" to supportLen,
                "resolution_overlay_variant" to overlayVariant,
                "resolution_fallback_reason" to fallbackReason
            )
        )
    }


    /**
     * Phase 5 — emits the projected channel slices selected by the current
     * contract, purely for visibility. This does not alter payload assembly yet.
     */
    fun emitReplyProjectedChannels(
        turnId: Long?,
        tickId: Int? = 2,
        demandCategory: String,
        channels: Map<String, String>
    ) {
        emit(
            mapOf(
                "type" to "REPLY_PROJECTED_CHANNELS",
                "turn_id" to turnId,
                "tick_id" to tickId,
                "demand_category" to demandCategory,
                "channels" to channels
            )
        )
    }

    /**
     * Phase 2 — emit the contract or placeholder plan selected for a reply.
     * This is still non-invasive: it just lets us inspect the new vocabulary
     * before Phase 6 starts using it for actual request shaping.
     */
    fun emitReplyAssemblyPlan(
        turnId: Long?,
        tickId: Int? = 2,
        demandCategory: String,
        requiredPromptModules: List<String>,
        requiredChannels: List<String>,
        optionalChannels: List<String>,
        forbiddenChannels: List<String>,
        selectedPromptModules: List<String> = emptyList(),
        selectedChannels: List<String> = emptyList(),
        rolloutMode: String? = null,
        softCharBudget: Int? = null,
        softTokenBudget: Int? = null,
        notes: String? = null
    ) {
        emit(
            mapOf(
                "type" to "REPLY_ASSEMBLY_PLAN",
                "turn_id" to turnId,
                "tick_id" to tickId,
                "demand_category" to demandCategory,
                "required_prompt_modules" to requiredPromptModules,
                "required_channels" to requiredChannels,
                "optional_channels" to optionalChannels,
                "forbidden_channels" to forbiddenChannels,
                "selected_prompt_modules" to selectedPromptModules,
                "selected_channels" to selectedChannels,
                "rollout_mode" to rolloutMode,
                "soft_char_budget" to softCharBudget,
                "soft_token_budget" to softTokenBudget,
                "notes" to notes
            )
        )
    }

    // =========================================================================
    // LLM request/response digests (kept; recommend pairing with MODEL_CALL_* events)
    // =========================================================================

    fun emitLlmRequestDigest(
        mode: String,
        model: String?,
        convoSessionId: String,
        turnId: Long?,
        promptHash: String?,
        personaHash: String?,
        systemPrompt: String?,
        developerPrompt: String?,
        userMessage: String?,
        // New optional canonical fields:
        tickId: Int? = null,
        modelCallId: String? = null,
        modelCallName: String? = null,
        correlationId: String? = null
    ) {
        val kv = mutableListOf<Pair<String, Any?>>()
        kv.add("conversation_id" to conversationId)
        kv.add("mode" to mode)
        kv.add("model" to model)
        kv.add("convo_session_id" to convoSessionId)
        kv.add("turn_id" to turnId)
        kv.add("prompt_hash" to promptHash)
        kv.add("persona_hash" to personaHash)
        kv.add("tick_id" to tickId)
        kv.add("model_call_id" to modelCallId)
        kv.add("model_call_name" to modelCallName)
        kv.add("correlation_id" to correlationId)
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
        assistantText: String?,
        // New optional canonical fields:
        tickId: Int? = null,
        modelCallId: String? = null,
        modelCallName: String? = null,
        correlationId: String? = null
    ) {
        val kv = mutableListOf<Pair<String, Any?>>()
        kv.add("conversation_id" to conversationId)
        kv.add("mode" to mode)
        kv.add("model" to model)
        kv.add("convo_session_id" to convoSessionId)
        kv.add("turn_id" to turnId)
        kv.add("tick_id" to tickId)
        kv.add("model_call_id" to modelCallId)
        kv.add("model_call_name" to modelCallName)
        kv.add("correlation_id" to correlationId)
        kv.addAll(textDigestPairs("asst", assistantText))
        emitKv("LLM_RESPONSE_DIGEST", *kv.toTypedArray())
    }

    // =========================================================================
    // ASR row tracking helpers (kept as-is, with conversation_id included)
    // =========================================================================

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
            "conversation_id" to conversationId,
            "row_id" to rowId
        )
        for ((k, v) in kv) payload[k] = v
        emitInternal(ctx, payload, now)
        return rowId
    }

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

        val payload = linkedMapOf<String, Any?>(
            "type" to type,
            "conversation_id" to conversationId
        )
        if (rowId != null) payload["row_id"] = rowId
        for ((k, v) in kv) payload[k] = v
        emitInternal(ctx, payload, now)
    }

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

        val payload = linkedMapOf<String, Any?>(
            "type" to type,
            "conversation_id" to conversationId
        )
        if (rowId != null) payload["row_id"] = rowId
        for ((k, v) in kv) payload[k] = v
        emitInternal(ctx, payload, now)

        if (eval != null) {
            val evalPayload = linkedMapOf<String, Any?>(
                "type" to "CONVTEL_ASR_ROW_EVAL",
                "conversation_id" to conversationId,
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
        val isNormal = (row.hasReady || row.hasRms) &&
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

    // =========================================================================
    // Public generic API
    // =========================================================================

    /**
     * init(context)
     * -----------------------------------------------------------------------
     * - Call once in Application.onCreate() (preferred).
     * - Emits APP_START and CONVERSATION_CONTEXT.
     */
    fun init(
        context: Context,
        userId: String? = null,
        sessionId: String? = null,
        logcatEcho: Boolean = true,
        conversationId: String? = null,
        deviceSessionId: String? = null,
        voiceSessionId: String? = null,
        policySessionId: String? = null,
        appVersionName: String? = null,
        appBuildCode: Long? = null,
        gitSha: String? = null
    ) {
        if (started.get()) return

        this.appContext = context.applicationContext
        userId?.let { this.userId = it }
        sessionId?.let { this.appSessionId = it }
        this.logcatEcho = logcatEcho
        conversationId?.let { this.conversationId = it }
        deviceSessionId?.let { this.deviceSessionId = it }
        voiceSessionId?.let { this.voiceSessionId = it }
        policySessionId?.let { this.policySessionId = it }
        appVersionName?.let { this.appVersionName = it }
        appBuildCode?.let { this.appBuildCode = it }
        gitSha?.let { this.gitSha = it }

        startWriterThread()

        emitKv(
            "APP_START",
            "schema_version" to TELEMETRY_SCHEMA_VERSION,
            "sdk_int" to Build.VERSION.SDK_INT,
            "device" to "${Build.MANUFACTURER} ${Build.MODEL}".trim(),
            "user_id" to (this.userId ?: "(none)"),
            "conversation_id" to this.conversationId,
            "app_session_id" to this.appSessionId,
            "device_session_id" to this.deviceSessionId,
            "voice_session_id" to this.voiceSessionId,
            "policy_session_id" to this.policySessionId
        )

        emitConversationContext(reason = "init")
    }

    /**
     * Update conversation and subsystem session IDs any time they change.
     * Emit CONVERSATION_CONTEXT so audits can re-bind streams deterministically.
     */
    fun setConversationIdentity(
        conversationId: String? = null,
        deviceSessionId: String? = null,
        voiceSessionId: String? = null,
        policySessionId: String? = null,
        reason: String = "update"
    ) {
        conversationId?.let { this.conversationId = it }
        deviceSessionId?.let { this.deviceSessionId = it }
        voiceSessionId?.let { this.voiceSessionId = it }
        policySessionId?.let { this.policySessionId = it }
        emitConversationContext(reason = reason)
    }

    /** Optional build info update (safe to call anytime). */
    fun setAppBuildInfo(versionName: String?, buildCode: Long?, gitSha: String?) {
        this.appVersionName = versionName
        this.appBuildCode = buildCode
        this.gitSha = gitSha
        emitConversationContext(reason = "build_info_update")
    }

    fun emitConversationModeChosen(
        convoSessionId: String,
        mode: String,
        reason: String,
        printed: Int? = null,
        handwritten: Int? = null,
        blanks: Int? = null,
        nonZero: Int? = null
    ) {
        emitKv(
            "CONV_MODE_CHOSEN",
            "conversation_id" to conversationId,
            "convo_session_id" to convoSessionId,
            "mode" to mode,
            "reason" to reason,
            "printed" to printed,
            "handwritten" to handwritten,
            "blanks" to blanks,
            "nonZero" to nonZero
        )
    }

    /** Update just the app_session_id (telemetry stream id). */
    fun setSessionId(id: String) {
        val prev = this.appSessionId
        this.appSessionId = id
        emitKv(
            "SESSION_SET",
            "conversation_id" to conversationId,
            "prev_app_session_id" to prev,
            "app_session_id" to id
        )
        emitConversationContext(reason = "app_session_changed")
    }

    fun setUserId(id: String) {
        this.userId = id
        emitKv("USER_SET", "conversation_id" to conversationId, "user_id" to id)
    }

    fun setLogcatEcho(enabled: Boolean) {
        this.logcatEcho = enabled
    }

    fun emit(payload: Map<String, Any?>) {
        val ctx = appContext ?: return
        val now = System.currentTimeMillis()
        emitInternal(ctx, payload, now)
    }

    internal fun emitKv(type: String, vararg kv: Pair<String, Any?>) {
        // Ensure writer thread started
        ensureStarted()

        val line = try {
            buildEventJson(type, *kv)
        } catch (t: Throwable) {
            // Last resort: never break JSONL. Emit minimal error line.
            val fallback = JSONObject()
                .put("type", "TELEMETRY_ENCODE_ERROR")
                .put("schema_version", TELEMETRY_SCHEMA_VERSION)
                .put("telemetry_id", telemetryId)
                .put("seq", seq.incrementAndGet())
                .put("ts_epoch_ms", System.currentTimeMillis())
                .put("app_session_id", appSessionId)
                .put("conversation_id", conversationId)
                .put("error", sanitizeStringForJsonl((t.message ?: t.javaClass.simpleName).take(240)))
                .toString()
            sanitizeStringForJsonl(fallback)
        }

        // Exactly one line per event:
        val ok = queue.offer(line, 20, TimeUnit.MILLISECONDS)
        if (!ok) {
            // Queue is full -> drop event but keep a counter (and optionally logcat)
            Log.w(LOG_TAG, "Telemetry queue full; dropping event type=$type")
        }
    }

    private fun ensureStarted() {
        if (started.get()) return
        synchronized(this) {
            if (started.get()) return
            started.set(true)
            startWriterThread()
            emitConversationContext(reason = "telemetry_started")
        }
    }

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

    // =========================================================================
    // Internals
    // =========================================================================

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

            var leftover: String? = queue.poll()
            while (leftover != null) {
                try {
                    writeLine(leftover)
                } catch (_: Throwable) {
                }
                leftover = queue.poll()
            }

            try {
                fileWriter?.flush()
            } catch (_: Throwable) {
            }
            try {
                fileWriter?.close()
            } catch (_: Throwable) {
            }
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

        // ---- always-on diagnostics ----
        val pid = Process.myPid()
        val tid = Process.myTid()
        val threadName = Thread.currentThread().name
        val uptimeMs = SystemClock.uptimeMillis()
        val elapsedMs = SystemClock.elapsedRealtime()

        // Standard fields
        obj.put("ts_epoch_ms", now)
        obj.put("ts_iso", isoFmt.format(Date(now)))
        obj.put("type", payload["type"] ?: "(unknown)")

        // Backward compatibility: you historically used "session_id"
        // We keep it, but canonical is app_session_id.
        obj.put("session_id", appSessionId)
        obj.put("app_session_id", appSessionId)

        // Canonical conversation join key
        obj.put("conversation_id", conversationId ?: JSONObject.NULL)

        // Optional subsystem session ids
        obj.put("device_session_id", deviceSessionId ?: JSONObject.NULL)
        obj.put("voice_session_id", voiceSessionId ?: JSONObject.NULL)
        obj.put("policy_session_id", policySessionId ?: JSONObject.NULL)

        // Schema + build info
        obj.put("schema_version", TELEMETRY_SCHEMA_VERSION)
        obj.put("app_version_name", appVersionName ?: JSONObject.NULL)
        obj.put("app_build_code", appBuildCode ?: JSONObject.NULL)
        obj.put("git_sha", gitSha ?: JSONObject.NULL)

        // Telemetry stream fields
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

        val line = sanitizeStringForJsonl(obj.toString())
        if (logcatEcho) Log.i(LOG_TAG, line)
        ensureOpenForToday(ctx)
        offer(line)
    }

    private fun ensureOpenForToday(ctx: Context) {
        val dayDir = File(ctx.filesDir, "telemetry/${dayFmt.format(Date())}")
        if (currentDayDir?.absolutePath != dayDir.absolutePath) {
            try {
                fileWriter?.flush()
            } catch (_: Throwable) {
            }
            try {
                fileWriter?.close()
            } catch (_: Throwable) {
            }

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
        val fname = "events_${appSessionId}_$ts.jsonl"
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
        // ✅ Patch C: write "line + newline" atomically (single write call)
        val out = if (line.endsWith('\n')) line else (line + "\n")
        fw.write(out)
        byteCount += out.length.toLong()
        if (byteCount >= MAX_BYTES_PER_FILE) {
            try {
                fw.flush()
            } catch (_: Throwable) {
            }
            fileWriter = null
        }
    }

    /**
     * JSON safety: supports primitives, arrays, iterables, maps, and nested JSON objects.
     * Keep it strict because audit tools hate inconsistent types.
     */
    private fun putSafe(obj: JSONObject, key: String, value: Any?) {
        when (value) {
            null -> obj.put(key, JSONObject.NULL)
            is Number, is Boolean -> obj.put(key, value)
            is String -> obj.put(key, sanitizeStringForJsonl(value))
            is JSONObject -> obj.put(key, value)
            is JSONArray -> obj.put(key, value)
            is Map<*, *> -> {
                val o = JSONObject()
                value.forEach { (k, v) ->
                    if (k is String) putSafe(o, k, v)
                }
                obj.put(key, o)
            }
            is Iterable<*> -> {
                val arr = JSONArray()
                for (v in value) {
                    when (v) {
                        null -> arr.put(JSONObject.NULL)
                        is String -> arr.put(sanitizeStringForJsonl(v))
                        is Number, is Boolean -> arr.put(v)
                        is Map<*, *> -> {
                            val o = JSONObject()
                            v.forEach { (k, vv) ->
                                if (k is String) putSafe(o, k, vv)
                            }
                            arr.put(o)
                        }
                        else -> arr.put(sanitizeStringForJsonl(v.toString()))
                    }
                }
                obj.put(key, arr)
            }
            is IntArray -> {
                val arr = JSONArray()
                value.forEach { arr.put(it) }
                obj.put(key, arr)
            }
            is FloatArray -> {
                val arr = JSONArray()
                value.forEach { arr.put(it) }
                obj.put(key, arr)
            }
            is LongArray -> {
                val arr = JSONArray()
                value.forEach { arr.put(it) }
                obj.put(key, arr)
            }
            is Array<*> -> {
                val arr = JSONArray()
                value.forEach { v ->
                    when (v) {
                        null -> arr.put(JSONObject.NULL)
                        is String -> arr.put(sanitizeStringForJsonl(v))
                        is Number, is Boolean -> arr.put(v)
                        else -> arr.put(sanitizeStringForJsonl(v.toString()))
                    }
                }
                obj.put(key, arr)
            }
            else -> obj.put(key, sanitizeStringForJsonl(value.toString()))
        }
    }
}