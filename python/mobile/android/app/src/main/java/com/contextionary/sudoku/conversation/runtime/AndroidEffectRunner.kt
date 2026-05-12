package com.contextionary.sudoku.conversation.runtime

import android.os.SystemClock
import android.util.Log
import com.chaquo.python.Python
import com.contextionary.sudoku.MainActivity
import com.contextionary.sudoku.SudokuResultView
import com.contextionary.sudoku.conductor.CellClass
import com.contextionary.sudoku.conductor.Eff
import com.contextionary.sudoku.conductor.EffectRunner
import com.contextionary.sudoku.conductor.Evt
import com.contextionary.sudoku.conductor.LlmPolicy
import com.contextionary.sudoku.conductor.PolicyCallCtx
import com.contextionary.sudoku.conductor.SudoMode
import com.contextionary.sudoku.conductor.ToolCall
import com.contextionary.sudoku.conductor.policy.IntentEnvelopeV1

import com.contextionary.sudoku.telemetry.Checkpoint
import com.contextionary.sudoku.telemetry.ConversationTelemetry
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * Phase 1.5:
 * - Restores missing effect branches that used to live in MainActivity.ensureSudoStore() runner.
 * - Restores ORIGINAL runner behavior for policy calls:
 *   - policy call guarded: runCatching { decide }.getOrElse { emptyList() }
 *   - retry bounded until meaningful toolplan (with backoff delay)
 *   - FM-02 Reply-only fallback if still not meaningful
 *   - never emit empty / reply-less toolplans into the Store
 *
 * Fix 1 (2026-02):
 * - Single-flight policy calls + stale-response drop:
 *   - Keep activePolicyJob + activePolicyReqSeq
 *   - Cancel previous job when a new policy effect starts (CallPolicy or Tick2)
 *   - Drop any response whose policyReqSeq != activePolicyReqSeq (don’t emit tools)
 *
 * Phase 4 (second half):
 * - Adds true Undo/Redo stacks
 * - Pushes snapshot BEFORE any grid mutation (and clears redo on new branch)
 * - Implements Eff.Undo / Eff.Redo with snapshot restore + GridSnapshotUpdated emit
 */
class AndroidEffectRunner(
    private val sid: String,
    private val scope: CoroutineScope,
    private val policy: LlmPolicy,
    private val host: Host
) : EffectRunner {

    /**
     * Host = the minimum surface AndroidEffectRunner needs from MainActivity.
     */
    interface Host {
        // --- UI ---
        fun runOnUiThread(block: () -> Unit)
        fun updateSudoMessage(text: String)
        fun speakAssistant(message: String, listenAfter: Boolean, turnId: Long?, tickId: Int?, source: String)

        // --- ASR / TurnController ---
        fun requestListenCompat(reason: String)
        fun stopAsrCompat(reason: String)

        // --- Sudoku view / overlay hooks used by effects ---
        fun onSetFocusCellPulse(cellIndex: Int?)
        fun renderSolveOverlayV0(frameJson: String)
        fun getResultsSudokuViewOrNull(): Any?
        fun stopConfirmationPulseIfAny()

        // --- Snapshot builder hooks (used by tool handlers) ---
        fun buildGridStateFromOverlayOrNull(): Any?
        fun buildLLMGridStateFromOverlayOrNull(gridState: Any): Any?
        fun emitGridSnapshotUpdated(llmGridState: Any, reason: String, editSeq: Long)

        // --- Canonical UI arrays used by tool handlers ---
        val uiDigits: IntArray
        val uiConfs: FloatArray
        val uiGiven: BooleanArray
        val uiSol: BooleanArray
        val uiCand: IntArray
        val uiAuto: BooleanArray
        val uiManual: BooleanArray

        // --- Phase 1.5 additions used by removed runner ---
        val uiConfirmed: BooleanArray?
        val resultsDigitsOrNull: IntArray?
        val resultsConfidencesOrNull: FloatArray?

        fun rerenderFromCanonical()
        fun persistConfirmedIfPossible(cellIndex: Int, confirmed: Boolean)
    }

    // ============================================================
    // Phase 4 (second half): Undo/Redo stacks
    // ============================================================

    private data class GridUndoSnapshot(
        val digits: IntArray,
        val confs: FloatArray,
        val given: BooleanArray,
        val sol: BooleanArray,
        val cand: IntArray,
        val auto: BooleanArray,
        val manual: BooleanArray,
        val confirmed: BooleanArray?
    )

    private val undoStack: ArrayDeque<GridUndoSnapshot> = ArrayDeque()
    private val redoStack: ArrayDeque<GridUndoSnapshot> = ArrayDeque()


    private fun IntentEnvelopeV1.normalizeCompat(rawUserText: String): IntentEnvelopeV1 = this

    private fun <T> ArrayDeque<T>.removeLastOrNullCompat(): T? {
        return if (this.isEmpty()) null else this.removeLast()
    }

    private fun captureSnapshot(h: Host): GridUndoSnapshot {
        return GridUndoSnapshot(
            digits = h.uiDigits.copyOf(),
            confs = h.uiConfs.copyOf(),
            given = h.uiGiven.copyOf(),
            sol = h.uiSol.copyOf(),
            cand = h.uiCand.copyOf(),
            auto = h.uiAuto.copyOf(),
            manual = h.uiManual.copyOf(),
            confirmed = h.uiConfirmed?.copyOf()
        )
    }

    private fun restoreSnapshot(h: Host, snap: GridUndoSnapshot) {
        System.arraycopy(snap.digits, 0, h.uiDigits, 0, 81)
        System.arraycopy(snap.confs, 0, h.uiConfs, 0, 81)
        System.arraycopy(snap.given, 0, h.uiGiven, 0, 81)
        System.arraycopy(snap.sol, 0, h.uiSol, 0, 81)
        System.arraycopy(snap.cand, 0, h.uiCand, 0, 81)
        System.arraycopy(snap.auto, 0, h.uiAuto, 0, 81)
        System.arraycopy(snap.manual, 0, h.uiManual, 0, 81)

        snap.confirmed?.let { src ->
            val dst = h.uiConfirmed
            if (dst != null && dst.size == 81 && src.size == 81) {
                System.arraycopy(src, 0, dst, 0, 81)
            }
        }

        h.resultsDigitsOrNull?.let { System.arraycopy(h.uiDigits, 0, it, 0, 81) }
        h.resultsConfidencesOrNull?.let { System.arraycopy(h.uiConfs, 0, it, 0, 81) }

        h.rerenderFromCanonical()
    }

    private fun pushUndoSnapshot(h: Host) {
        undoStack.addLast(captureSnapshot(h))
        redoStack.clear()
    }

    // -----------------------------
    // State previously captured in ensureSudoStore()
    // -----------------------------

    private var applyEditSeq = 0L
    private var lastAssistantSpoken: String = ""

    // Solve overlay dedupe memory (from removed runner)
    private var lastSolveOverlayFrameSha12: String? = null

    // ============================================================
    // Fix 1: Single-flight policy calls + stale-response drop
    // ============================================================

    private val policyFlightLock = Any()


    // Phase 0: detect duplicate Tick2 per turn (symptom of loops / double calls)
    private val tick2CallsByTurn = mutableMapOf<Long, Int>()
    private fun bumpTick2Calls(turnId: Long): Int {
        val n = (tick2CallsByTurn[turnId] ?: 0) + 1
        tick2CallsByTurn[turnId] = n
        // Keep map bounded to avoid leaks in long sessions
        if (tick2CallsByTurn.size > 40) {
            val oldest = tick2CallsByTurn.keys.sorted().take(10)
            oldest.forEach { tick2CallsByTurn.remove(it) }
        }
        return n
    }


    private var activePolicyJob: Job? = null
    private val activePolicyReqSeq = AtomicLong(0L)

    private fun beginPolicyFlight(reqSeq: Long, label: String) {
        synchronized(policyFlightLock) {
            val prev = activePolicyJob
            if (prev != null && prev.isActive) {
                runCatching {
                    prev.cancel(CancellationException("superseded_by=$label reqSeq=$reqSeq"))
                }
            }
            activePolicyReqSeq.set(reqSeq)
        }
    }

    private fun setActivePolicyJob(job: Job) {
        synchronized(policyFlightLock) {
            activePolicyJob = job
        }
    }

    private fun isStale(reqSeq: Long): Boolean = reqSeq != activePolicyReqSeq.get()

    // ---- joinable parse of toolCallId ----
    // expected: tc:<turnId>:<tickId>:<policyReqSeq>:<toolplanId>:<idxOrTag>
    private data class ToolCallMeta(
        val turnId: Long,
        val tickId: Int,
        val policyReqSeq: Long,
        val toolplanId: String
    )

    private fun parseToolCallId(toolCallId: String): ToolCallMeta? {
        val parts = toolCallId.split(":")
        if (parts.size < 6) return null
        if (parts[0] != "tc") return null

        val turnId = parts[1].toLongOrNull() ?: return null
        val tickId = parts[2].toIntOrNull() ?: return null
        val policyReqSeq = parts[3].toLongOrNull() ?: return null
        val toolplanId = parts[4]
        return ToolCallMeta(turnId, tickId, policyReqSeq, toolplanId)
    }

    private fun sha12(s: String): String =
        runCatching { ConversationTelemetry.sha256Hex(s).take(12) }.getOrElse { "sha12_err" }

    private fun invokeNoArgIfExists(target: Any?, methodName: String): Boolean {
        if (target == null) return false
        return runCatching {
            val m = target::class.java.methods.firstOrNull { it.name == methodName && it.parameterTypes.isEmpty() }
            if (m != null) {
                m.isAccessible = true
                m.invoke(target)
                true
            } else false
        }.getOrDefault(false)
    }

    // ============================================================
    // ORIGINAL runner behavior (match MainActivity_ORIGINAL):
    // - isMeaningfulToolplan: non-blank reply AND not all Reply with blank text
    // - retryBounded: delay between attempts (baseDelayMs = 250)
    // - policy call guarded BEFORE retry logic sees output (exception -> emptyList)
    // ============================================================

    private fun isMeaningfulToolplan(tools: List<ToolCall>?): Boolean {
        if (tools.isNullOrEmpty()) return false

        val hasReply = tools.any { it is ToolCall.Reply && it.text.isNotBlank() }
        val allNoop = tools.all { it is ToolCall.Reply } &&
                tools.filterIsInstance<ToolCall.Reply>().all { it.text.isBlank() }

        return hasReply && !allNoop
    }

    private fun fm02FallbackReplyText(): String {
        // Short, deterministic fallback (same intent as original FM-02).
        return "Sorry — I couldn’t reach the assistant reliably. Please try again."
    }

    private data class RetryOutcome(
        val tools: List<ToolCall>?,
        val shouldRetry: Boolean
    )

    private suspend fun retryBounded(
        maxAttempts: Int,
        baseDelayMs: Long = 250L,
        block: suspend (attempt: Int) -> RetryOutcome
    ): List<ToolCall>? {
        var attempt = 1
        while (attempt <= maxAttempts) {
            val outcome = runCatching { block(attempt) }.getOrElse {
                // If the retry wrapper itself crashes, stop immediately.
                RetryOutcome(tools = emptyList(), shouldRetry = false)
            }

            val out = outcome.tools
            if (isMeaningfulToolplan(out)) return out

            // ✅ Fail-fast path (e.g., timeout): stop retry loop now.
            if (!outcome.shouldRetry) return null

            attempt++
            if (attempt <= maxAttempts) {
                delay(baseDelayMs)
            }
        }
        return null
    }

    private fun emitPolicyTrace(tag: String, kv: Map<String, Any?> = emptyMap()) {
        runCatching {
            val base = mutableMapOf<String, Any?>(
                "type" to "POLICY_TRACE",
                "session_id" to sid,
                "tag" to tag
            )
            for ((k, v) in kv) base[k] = v
            ConversationTelemetry.emit(base)
        }
    }

    // -----------------------------
    // Stage timing + payload sizing (Phase F)
    // -----------------------------

    private fun bytesOf(s: String?): Int =
        if (s == null) 0 else runCatching { s.toByteArray().size }.getOrDefault(s.length)

    private fun emitStage(
        stage: String,
        phase: String, // "START" | "END"
        ctx: PolicyCallCtx,
        elapsedMs: Long? = null,
        extra: Map<String, Any?> = emptyMap()
    ) {
        val kv = mutableMapOf<String, Any?>(
            "where" to "AndroidEffectRunner",
            "stage" to stage,
            "phase" to phase,
            "turnId" to ctx.turnId,
            "tickId" to ctx.tickId,
            "policyReqSeq" to ctx.policyReqSeq,
            "modelCallId" to ctx.modelCallId,
            "toolplanId" to ctx.toolplanId,
            "correlationId" to ctx.correlationId,
            "mode" to ctx.mode,
            "reason" to ctx.reason
        )
        if (elapsedMs != null) kv["elapsed_ms"] = elapsedMs
        for ((k, v) in extra) kv[k] = v

        // Use the same POLICY_TRACE channel so your audit pipeline stays simple.
        emitPolicyTrace(
            tag = "STAGE_${stage}_${phase}",
            kv = kv
        )
    }

    private fun emitTick1IntentEnvelopeV1(
        ctx: PolicyCallCtx,
        rawUserText: String,
        env: IntentEnvelopeV1
    ) {
        val intentsSorted = env.intents.sortedByDescending { it.confidence }
        val top = intentsSorted.firstOrNull()

        fun idxFromCell(cell: String?): Int? {
            if (cell.isNullOrBlank()) return null
            val m = Regex("""r([1-9])c([1-9])""", RegexOption.IGNORE_CASE).find(cell.trim()) ?: return null
            val r = m.groupValues[1].toIntOrNull() ?: return null
            val c = m.groupValues[2].toIntOrNull() ?: return null
            return ((r - 1) * 9 + (c - 1)).takeIf { it in 0..80 }
        }

        val intentsPayload = intentsSorted.map { i ->
            val cell = i.targets.firstOrNull { !it.cell.isNullOrBlank() }?.cell
            mapOf(
                "id" to i.id,
                "type" to i.type.name,
                "confidence" to i.confidence,
                "targets" to mapOf(
                    "cell" to cell,
                    "cell_index" to idxFromCell(cell),
                    "region" to (i.targets.firstOrNull()?.region?.let { mapOf("kind" to it.kind.name, "index" to it.index) })
                ),
                "values" to mapOf(
                    "digit" to i.payload.digit,
                    "digits" to (i.payload.digits ?: emptyList<Int>()),
                    "region_digits" to (i.payload.regionDigits ?: "")
                ),
                "uncertainty" to mapOf(
                    "missing_fields" to (i.missing ?: emptyList<String>())
                ),
                "evidence" to mapOf(
                    "notes" to (i.evidenceText ?: "")
                ),
                "addresses_user_agenda_id" to runCatching {
                    // Keep joinability if the field exists on this build; otherwise null.
                    val m = i::class.java.methods.firstOrNull { it.name == "addressesUserAgendaIdCompat" && it.parameterTypes.isEmpty() }
                    m?.invoke(i) as? String
                }.getOrNull(),
                "addresses_app_agenda_id" to null
            )
        }

        val ambiguityHigh = runCatching {
            // Prefer env.isUnclearCompat() if present; fallback to top-missing.
            val m = env::class.java.methods.firstOrNull { it.name == "isUnclearCompat" && it.parameterTypes.isEmpty() }
            val v = (m?.invoke(env) as? Boolean)
            v ?: false
        }.getOrElse { false } || (top?.missing?.isNotEmpty() == true)

        val summary = buildString {
            append("top="); append(top?.type?.name ?: "none")
            val cell = top?.targets?.firstOrNull { !it.cell.isNullOrBlank() }?.cell
            val digit = top?.payload?.digit
            if (!cell.isNullOrBlank()) append(" $cell")
            if (digit != null) append("=$digit")
            append(" conf="); append(String.format("%.2f", (top?.confidence ?: 0.0)))
            append(" ambiguity="); append(if (ambiguityHigh) "HIGH" else "none")
            append(" intents="); append(env.intents.size)
        }

        ConversationTelemetry.emit(
            mapOf(
                // ✅ This event is now “the Tick1 truth”
                "type" to "TICK1_INTENT_ENVELOPE",
                "schema_version" to "v1.0",
                "payload_kind" to "intent_envelope_v1",

                // Common header (ids)
                "session_id" to ctx.sessionId,
                "turn_id" to ctx.turnId,
                "tick_id" to ctx.tickId,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,

                // ✅ No “Meaning” naming anywhere
                "tag" to "IntentEnvelopeV1",

                "summary" to summary,
                "payload" to mapOf(
                    "raw_user_text" to rawUserText,
                    "raw_user_text_preview" to rawUserText.take(160),
                    "top_intent_id" to (top?.id ?: ""),
                    "top_intent_type" to (top?.type?.name ?: ""),
                    "top_intent_confidence" to (top?.confidence ?: 0.0),
                    "intents_count" to env.intents.size,
                    "intents" to intentsPayload,
                    "is_ambiguous" to ambiguityHigh,
                    "free_talk_topic" to env.freeTalkTopic,
                    "free_talk_confidence" to env.freeTalkConfidence
                )
            )
        )
    }

    // ✅ helper: emit deterministic ToolExecuted + joinable TOOL_HANDLER_RESULT
    private fun emitToolExecuted(
        emit: (Evt) -> Unit,
        toolCallId: String,
        toolName: String,
        statusRaw: String,
        details: String
    ) {
        val toolResultId = "tr:$toolCallId"
        val toolResultText = "name=$toolName status=$statusRaw $details"

        val status = when (statusRaw.lowercase()) {
            "ok" -> "ok"
            "noop" -> "noop"
            "rejected" -> "rejected"
            else -> "error"
        }

        // 1) Emit ToolExecuted event (for tick2 attachments)
        runCatching {
            emit(
                Evt.ToolExecuted(
                    toolCallId = toolCallId,
                    toolName = toolName,
                    toolResultId = toolResultId,
                    toolResultText = toolResultText
                )
            )
        }

        // 2) Emit joinable TOOL_HANDLER_RESULT
        val meta = parseToolCallId(toolCallId)
        if (meta != null) {
            runCatching {
                ConversationTelemetry.emitToolHandlerResult(
                    sessionId = sid,
                    turnId = meta.turnId,
                    tickId = meta.tickId,
                    policyReqSeq = meta.policyReqSeq,
                    toolplanId = meta.toolplanId,
                    toolCallId = toolCallId,
                    toolResultId = toolResultId,
                    toolName = toolName,
                    status = status,
                    errorCode = if (status == "error") "handler_error" else null,
                    errorMsgShort = if (status == "error") details.take(180) else null
                )
            }
        } else {
            runCatching {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TOOL_HANDLER_RESULT_UNJOINABLE",
                        "session_id" to sid,
                        "toolCallId" to toolCallId,
                        "toolName" to toolName,
                        "status" to status,
                        "details" to details.take(220)
                    )
                )
            }
        }
    }

    // Helper: build & emit exactly one GridSnapshotUpdated
    private fun emitSnapshotOnce(
        emit: (Evt) -> Unit,
        reason: String,
        seq: Long
    ) {
        val gs = host.buildGridStateFromOverlayOrNull()
        if (gs != null) {
            val llmGrid = host.buildLLMGridStateFromOverlayOrNull(gs)
            if (llmGrid != null) {
                host.emitGridSnapshotUpdated(llmGrid, reason, seq)
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "GRID_SNAPSHOT_UPDATED_EMIT",
                        "edit_seq" to seq,
                        "reason" to reason
                    )
                )
            }
        } else {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "GRID_SNAPSHOT_UPDATED_MISSED",
                    "edit_seq" to seq,
                    "reason" to "buildGridStateFromOverlay returned null ($reason)"
                )
            )
            Log.e("AndroidEffectRunner", "Gate1 FAIL: buildGridStateFromOverlay() returned null ($reason)")
        }
    }

    // ------------------------------------
    // EffectRunner interface
    // ------------------------------------

    override fun run(effect: Eff, emit: (Evt) -> Unit) {
        try {
            when (effect) {

                // ----------------------------
                // UI + voice
                // ----------------------------

                is Eff.UpdateUiMessage -> {
                    host.runOnUiThread {
                        runCatching { host.updateSudoMessage(effect.text) }
                            .onFailure { Log.w("AndroidEffectRunner", "UpdateUiMessage failed", it) }
                    }
                }

                is Eff.Speak -> {
                    // CP7-SPEAK checkpoint (preserve behavior)
                    runCatching {
                        Checkpoint.cp(
                            tag = "CP7-SPEAK",
                            sessionId = sid,
                            turnSeq = null,
                            turnId = null,
                            tickId = null,
                            correlationId = null,
                            policyReqSeq = null,
                            toolplanId = null,
                            modelCallId = null,
                            kv = mapOf(
                                "where" to "AndroidEffectRunner.run",
                                "text_len" to effect.text.length,
                                "listen_after" to effect.listenAfter
                            )
                        )
                    }

                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "EFFECT_RUN",
                            "effect" to "Speak",
                            "text_len" to effect.text.length,
                            "listen_after" to effect.listenAfter
                        )
                    )

                    lastAssistantSpoken = effect.text

                    host.runOnUiThread {
                        runCatching {
                            host.speakAssistant(
                                message = effect.text,
                                listenAfter = effect.listenAfter,
                                turnId = null,
                                tickId = null,
                                source = "effect_speak"
                            )
                        }.onFailure { Log.w("AndroidEffectRunner", "Speak failed", it) }
                    }
                }

                is Eff.SpeakAndShow -> {
                    host.runOnUiThread {
                        runCatching { host.updateSudoMessage(effect.text) }
                        runCatching {
                            host.speakAssistant(
                                message = effect.text,
                                listenAfter = effect.listenAfter,
                                turnId = null,
                                tickId = null,
                                source = "effect_speak_and_show"
                            )
                        }
                    }
                }

                is Eff.SetFocusCell -> {
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "EFFECT_RUN",
                            "effect" to "SetFocusCell",
                            "cellIndex" to (effect.cellIndex ?: -1),
                            "reason" to (effect.reason ?: "")
                        )
                    )
                    host.runOnUiThread {
                        runCatching { host.onSetFocusCellPulse(effect.cellIndex) }
                            .onFailure { t -> Log.w("AndroidEffectRunner", "SetFocusCell failed", t) }
                    }
                }

                is Eff.RequestListen -> {
                    ConversationTelemetry.emit(
                        mapOf("type" to "EFFECT_RUN", "effect" to "RequestListen", "reason" to effect.reason)
                    )
                    Log.i("AndroidEffectRunner", "Eff.RequestListen(reason=${effect.reason})")
                    host.requestListenCompat(reason = "eff:${effect.reason}")
                }

                is Eff.StopAsr -> {
                    Log.i("AndroidEffectRunner", "Eff.StopAsr(reason=${effect.reason})")
                    ConversationTelemetry.emit(mapOf("type" to "EFF_STOP_ASR", "reason" to effect.reason))
                    host.stopAsrCompat(reason = "eff:${effect.reason}")
                }

                // ============================================================
                // SOLVING effects + solver query bridge
                // ============================================================

                is Eff.ComputeSolveStep -> {
                    thread(name = "SolveStepEngine", isDaemon = true) {
                        val payload = JSONObject()
                            .put("grid81", effect.grid81)
                            .put(
                                "options",
                                JSONObject()
                                    .put("use_cleanup_method", true)
                                    .put("include_magic_technique", false)
                                    .put("step_style", "full")
                            )
                            .toString()

                        try {
                            val py = Python.getInstance()
                            val mod = py.getModule("step_by_step_bridge")
                            val out = mod.callAttr("next_step", payload).toString()

                            val obj = JSONObject(out)
                            val ok = obj.optBoolean("ok", false)
                            if (ok) {
                                val h12 = effect.gridHash12.take(12)
                                emit(
                                    Evt.SolveStepUpdated(
                                        gridHash12 = h12,
                                        stepJson = obj.toString(),
                                        reason = effect.reason
                                    )
                                )
                            } else {
                                emit(
                                    Evt.SolveStepFailed(
                                        gridHash12 = effect.gridHash12.take(12),
                                        error = obj.optString("error", "unknown_error"),
                                        reason = effect.reason
                                    )
                                )
                            }
                        } catch (t: Throwable) {
                            emit(
                                Evt.SolveStepFailed(
                                    gridHash12 = effect.gridHash12.take(12),
                                    error = (t.message ?: t.toString()),
                                    reason = effect.reason
                                )
                            )
                        }
                    }
                }

                is Eff.RunDetourSolverQuery -> {
                    thread(name = "DetourSolverQuery", isDaemon = true) {
                        try {
                            val py = Python.getInstance()
                            val mod = py.getModule("step_by_step_bridge")
                            val out = mod.callAttr("detour_query", effect.payloadJson).toString()

                            val obj = JSONObject(out)
                            val ok = obj.optBoolean("ok", false)
                            if (ok) {
                                emit(
                                    Evt.DetourSolverQuerySucceeded(
                                        queryId = effect.queryId,
                                        op = effect.op,
                                        resultJson = obj.toString(),
                                        reason = effect.reason
                                    )
                                )
                            } else {
                                val errObj = obj.optJSONObject("error")
                                val errMsg =
                                    errObj?.optString("msg")?.takeIf { it.isNotBlank() }
                                        ?: obj.optString("status", "detour_query_failed")

                                emit(
                                    Evt.DetourSolverQueryFailed(
                                        queryId = effect.queryId,
                                        op = effect.op,
                                        error = errMsg,
                                        reason = effect.reason
                                    )
                                )
                            }
                        } catch (t: Throwable) {
                            emit(
                                Evt.DetourSolverQueryFailed(
                                    queryId = effect.queryId,
                                    op = effect.op,
                                    error = (t.message ?: t.toString()),
                                    reason = effect.reason
                                )
                            )
                        }
                    }
                }

                is Eff.RenderSolveOverlay -> {
                    host.runOnUiThread {
                        val frame = effect.frameJson
                        if (frame.isBlank()) return@runOnUiThread

                        val sha = runCatching { sha12(frame) }.getOrNull()
                        if (sha != null && sha == lastSolveOverlayFrameSha12) return@runOnUiThread
                        lastSolveOverlayFrameSha12 = sha

                        runCatching {
                            val obj = JSONObject(frame)
                            val v = obj.optInt("v", 0)

                            when (v) {
                                0 -> host.renderSolveOverlayV0(frame)
                                1 -> {
                                    val focusIdx = obj.optJSONObject("focus")
                                        ?.let { f ->
                                            if (f.isNull("cellIndex")) null
                                            else f.optInt("cellIndex", -1).takeIf { it in 0..80 }
                                        }

                                    // Keep pulse semantics (optional UI affordance)
                                    host.onSetFocusCellPulse(focusIdx)

                                    val viewAny = host.getResultsSudokuViewOrNull()
                                    val view = viewAny as? SudokuResultView

                                    if (view != null) {
                                        runCatching {
                                            // Let the view render highlights + reveal deterministically from the same JSON.
                                            view.setSolveOverlayFromJson(frame)
                                            view.invalidate()
                                        }.onFailure {
                                            // fallback: do nothing special
                                        }
                                    } else {
                                        // Legacy fallback (if host returns a different view type)
                                        runCatching {
                                            val inv = viewAny?.javaClass?.methods?.firstOrNull { it.name == "invalidate" && it.parameterTypes.isEmpty() }
                                            inv?.invoke(viewAny)
                                        }
                                    }
                                }

                                else -> host.renderSolveOverlayV0(frame)
                            }
                        }.onFailure {
                            host.renderSolveOverlayV0(frame)
                        }
                    }
                }

                is Eff.ClearSolveOverlay -> {
                    host.runOnUiThread {
                        lastSolveOverlayFrameSha12 = null
                        host.onSetFocusCellPulse(null)
                        host.stopConfirmationPulseIfAny()

                        runCatching {
                            val viewAny = host.getResultsSudokuViewOrNull()
                            val view = viewAny as? SudokuResultView
                            if (view != null) {
                                view.clearSolveOverlay()
                            } else {
                                val inv = viewAny?.javaClass?.methods?.firstOrNull { it.name == "invalidate" && it.parameterTypes.isEmpty() }
                                inv?.invoke(viewAny)
                            }
                        }
                    }
                }

                // ============================================================
                // Operational grid effects (from removed runner)
                // ============================================================

                is Eff.ApplyCellEdit -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val d = effect.digit

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80 || d !in 0..9) {
                                emitToolExecuted(emit, effect.toolCallId, "apply_cell_edit", "rejected", "bad_args idx=$idx digit=$d")
                                return@runCatching
                            }

                            // ✅ Phase 4: snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            val old = host.uiDigits[idx]
                            host.uiDigits[idx] = d
                            host.uiConfs[idx] = 1.0f

                            if (d == 0) {
                                host.uiGiven[idx] = false
                                host.uiSol[idx] = false
                            }

                            host.uiCand[idx] = 0
                            host.uiAuto[idx] = false
                            host.uiManual[idx] = true

                            host.resultsDigitsOrNull?.let { System.arraycopy(host.uiDigits, 0, it, 0, 81) }
                            host.resultsConfidencesOrNull?.let { System.arraycopy(host.uiConfs, 0, it, 0, 81) }

                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "apply_cell_edit idx=$idx", seq = seq)

                            val status = if (old == d) "noop" else "ok"
                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "apply_cell_edit",
                                status,
                                "idx=$idx from=$old to=$d source=${effect.source}"
                            )
                        }.onFailure { t ->
                            Log.w("AndroidEffectRunner", "ApplyCellEdit failed", t)
                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "apply_cell_edit",
                                "error",
                                "crash=${t.javaClass.simpleName}:${t.message}"
                            )
                        }
                    }
                }

                is Eff.ConfirmCellValue -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val d = effect.digit

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80 || d !in 0..9) {
                                emitToolExecuted(emit, effect.toolCallId, "confirm_cell_value", "rejected", "bad_args idx=$idx digit=$d")
                                return@runCatching
                            }

                            // ✅ Phase 4: snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            host.uiConfirmed?.let { confirmedArr ->
                                if (idx in confirmedArr.indices) confirmedArr[idx] = true
                            }
                            host.persistConfirmedIfPossible(idx, true)

                            emitSnapshotOnce(
                                emit,
                                reason = "confirm_cell_value idx=$idx changed=${effect.changed}",
                                seq = seq
                            )

                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "confirm_cell_value",
                                "ok",
                                "idx=$idx digit=$d changed=${effect.changed} source=${effect.source}"
                            )
                        }.onFailure { t ->
                            Log.w("AndroidEffectRunner", "ConfirmCellValue failed", t)
                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "confirm_cell_value",
                                "error",
                                "crash=${t.javaClass.simpleName}:${t.message}"
                            )
                        }
                    }
                }

                is Eff.Undo -> {
                    val seq = ++applyEditSeq
                    host.runOnUiThread {
                        runCatching {
                            val snap = undoStack.removeLastOrNullCompat()
                            if (snap == null) {
                                emitToolExecuted(emit, effect.toolCallId, "undo", "noop", "empty_undo_stack source=${effect.source}")
                                return@runCatching
                            }

                            // push current to redo, restore prior
                            redoStack.addLast(captureSnapshot(host))
                            restoreSnapshot(host, snap)

                            emitSnapshotOnce(emit, reason = "undo source=${effect.source}", seq = seq)
                            emitToolExecuted(emit, effect.toolCallId, "undo", "ok", "restored_previous source=${effect.source}")
                        }.onFailure { t ->
                            emitToolExecuted(emit, effect.toolCallId, "undo", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.Redo -> {
                    val seq = ++applyEditSeq
                    host.runOnUiThread {
                        runCatching {
                            val snap = redoStack.removeLastOrNullCompat()
                            if (snap == null) {
                                emitToolExecuted(emit, effect.toolCallId, "redo", "noop", "empty_redo_stack source=${effect.source}")
                                return@runCatching
                            }

                            undoStack.addLast(captureSnapshot(host))
                            restoreSnapshot(host, snap)

                            emitSnapshotOnce(emit, reason = "redo source=${effect.source}", seq = seq)
                            emitToolExecuted(emit, effect.toolCallId, "redo", "ok", "restored_next source=${effect.source}")
                        }.onFailure { t ->
                            emitToolExecuted(emit, effect.toolCallId, "redo", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.ApplyCellClassify -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80) {
                                emitToolExecuted(emit, effect.toolCallId, "apply_cell_classify", "rejected", "bad_idx=$idx")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            when (effect.cellClass) {
                                CellClass.GIVEN -> {
                                    host.uiGiven[idx] = true
                                    host.uiSol[idx] = false
                                }

                                CellClass.SOLUTION -> {
                                    host.uiGiven[idx] = false
                                    host.uiSol[idx] = true
                                }

                                CellClass.EMPTY -> {
                                    host.uiGiven[idx] = false
                                    host.uiSol[idx] = false
                                    host.uiDigits[idx] = 0
                                    host.uiConfs[idx] = 1.0f
                                    host.uiCand[idx] = 0
                                    host.uiAuto[idx] = false
                                    host.uiManual[idx] = true
                                }
                            }

                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "apply_cell_classify idx=$idx", seq = seq)

                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "apply_cell_classify",
                                "ok",
                                "idx=$idx class=${effect.cellClass.name} source=${effect.source}"
                            )
                        }.onFailure { t ->
                            Log.w("AndroidEffectRunner", "ApplyCellClassify failed", t)
                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "apply_cell_classify",
                                "error",
                                "crash=${t.javaClass.simpleName}:${t.message}"
                            )
                        }
                    }
                }

                is Eff.ApplyCellCandidatesMask -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val mask = effect.candidateMask

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80) {
                                emitToolExecuted(emit, effect.toolCallId, "apply_candidates_mask", "rejected", "bad_idx=$idx")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            host.uiCand[idx] = mask.coerceIn(0, (1 shl 9) - 1)
                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "apply_candidates idx=$idx", seq = seq)

                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "apply_candidates_mask",
                                "ok",
                                "idx=$idx mask=${host.uiCand[idx]} source=${effect.source}"
                            )
                        }.onFailure { t ->
                            Log.w("AndroidEffectRunner", "ApplyCellCandidatesMask failed", t)
                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "apply_candidates_mask",
                                "error",
                                "crash=${t.javaClass.simpleName}:${t.message}"
                            )
                        }
                    }
                }

                is Eff.ReclassifyCell -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val kind = effect.kind.lowercase()

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80) {
                                emitToolExecuted(emit, effect.toolCallId, "reclassify_cell", "rejected", "bad_idx=$idx")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            when (kind) {
                                "given" -> {
                                    host.uiGiven[idx] = true; host.uiSol[idx] = false
                                }

                                "solution" -> {
                                    host.uiGiven[idx] = false; host.uiSol[idx] = true
                                }

                                "neither", "empty" -> {
                                    host.uiGiven[idx] = false; host.uiSol[idx] = false
                                }

                                else -> {
                                    emitToolExecuted(emit, effect.toolCallId, "reclassify_cell", "rejected", "bad_kind=$kind")
                                    return@runCatching
                                }
                            }

                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "reclassify_cell idx=$idx kind=$kind", seq = seq)

                            emitToolExecuted(
                                emit,
                                effect.toolCallId,
                                "reclassify_cell",
                                "ok",
                                "idx=$idx kind=$kind source=${effect.source}"
                            )
                        }.onFailure { t ->
                            Log.w("AndroidEffectRunner", "ReclassifyCell failed", t)
                            emitToolExecuted(emit, effect.toolCallId, "reclassify_cell", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.SetCandidates -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val mask = effect.mask

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80) {
                                emitToolExecuted(emit, effect.toolCallId, "set_candidates", "rejected", "bad_idx=$idx")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            host.uiCand[idx] = mask.coerceIn(0, (1 shl 9) - 1)
                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "set_candidates idx=$idx", seq = seq)
                            emitToolExecuted(emit, effect.toolCallId, "set_candidates", "ok", "idx=$idx mask=${host.uiCand[idx]} source=${effect.source}")
                        }.onFailure { t ->
                            emitToolExecuted(emit, effect.toolCallId, "set_candidates", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.ToggleCandidate -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val digit = effect.digit

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80 || digit !in 1..9) {
                                emitToolExecuted(emit, effect.toolCallId, "toggle_candidate", "rejected", "bad_args idx=$idx digit=$digit")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            val bit = 1 shl (digit - 1)
                            host.uiCand[idx] = host.uiCand[idx] xor bit
                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "toggle_candidate idx=$idx digit=$digit", seq = seq)
                            emitToolExecuted(emit, effect.toolCallId, "toggle_candidate", "ok", "idx=$idx digit=$digit source=${effect.source}")
                        }.onFailure { t ->
                            emitToolExecuted(emit, effect.toolCallId, "toggle_candidate", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.AddCandidate -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val digit = effect.digit

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80 || digit !in 1..9) {
                                emitToolExecuted(emit, effect.toolCallId, "add_candidate", "rejected", "bad_args idx=$idx digit=$digit")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            val bit = 1 shl (digit - 1)
                            val old = host.uiCand[idx]
                            val neu = old or bit
                            host.uiCand[idx] = neu
                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "add_candidate idx=$idx digit=$digit", seq = seq)
                            val status = if (old == neu) "noop" else "ok"
                            emitToolExecuted(emit, effect.toolCallId, "add_candidate", status, "idx=$idx digit=$digit mask=$neu source=${effect.source}")
                        }.onFailure { t ->
                            emitToolExecuted(emit, effect.toolCallId, "add_candidate", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.RemoveCandidate -> {
                    val seq = ++applyEditSeq
                    val idx = effect.cellIndex
                    val digit = effect.digit

                    host.runOnUiThread {
                        runCatching {
                            if (idx !in 0..80 || digit !in 1..9) {
                                emitToolExecuted(emit, effect.toolCallId, "remove_candidate", "rejected", "bad_args idx=$idx digit=$digit")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            val bit = 1 shl (digit - 1)
                            val old = host.uiCand[idx]
                            val neu = old and bit.inv()
                            host.uiCand[idx] = neu
                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "remove_candidate idx=$idx digit=$digit", seq = seq)
                            val status = if (old == neu) "noop" else "ok"
                            emitToolExecuted(emit, effect.toolCallId, "remove_candidate", status, "idx=$idx digit=$digit mask=$neu source=${effect.source}")
                        }.onFailure { t ->
                            emitToolExecuted(emit, effect.toolCallId, "remove_candidate", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                is Eff.FinalizeValidationPresentation -> {
                    val seq = ++applyEditSeq

                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "EFFECT_RUN",
                            "effect" to "FinalizeValidationPresentation",
                            "reason" to effect.reason
                        )
                    )

                    host.runOnUiThread {
                        runCatching {
                            val view = host.getResultsSudokuViewOrNull()
                            if (view == null) {
                                emitToolExecuted(emit, effect.toolCallId, "finalize_validation_presentation", "noop", "ui_not_ready")
                                return@runCatching
                            }

                            // ✅ snapshot BEFORE mutation
                            pushUndoSnapshot(host)

                            for (i in 0 until 81) {
                                host.uiConfs[i] = 1.0f
                                host.uiCand[i] = 0
                                host.uiAuto[i] = false
                                host.uiManual[i] = false
                            }

                            host.stopConfirmationPulseIfAny()
                            host.onSetFocusCellPulse(null)

                            invokeNoArgIfExists(view, "clearLogicAnnotations")
                            invokeNoArgIfExists(view, "clearAnnotations")
                            invokeNoArgIfExists(view, "clearHighlights")
                            invokeNoArgIfExists(view, "clearUnresolved")
                            invokeNoArgIfExists(view, "clearChanged")
                            invokeNoArgIfExists(view, "clearConflictBorders")

                            host.resultsConfidencesOrNull?.let { System.arraycopy(host.uiConfs, 0, it, 0, 81) }

                            host.rerenderFromCanonical()
                            emitSnapshotOnce(emit, reason = "finalize_validation_presentation reason=${effect.reason}", seq = seq)

                            emitToolExecuted(emit, effect.toolCallId, "finalize_validation_presentation", "ok", "reason=${effect.reason}")
                        }.onFailure { t ->
                            Log.w("AndroidEffectRunner", "FinalizeValidationPresentation failed", t)
                            emitToolExecuted(emit, effect.toolCallId, "finalize_validation_presentation", "error", "crash=${t.javaClass.simpleName}:${t.message}")
                        }
                    }
                }

                // ============================================================
                // Policy calls (Fix 1: single-flight + stale drop)
                // ============================================================

                is Eff.CallPolicy -> {
                    val reqSeq = effect.ctx.policyReqSeq
                    beginPolicyFlight(reqSeq, label = "CallPolicy")

                    // Voice-loop consistency: do not listen while we are waiting for LLM.
                    runCatching { host.stopAsrCompat(reason = "policy_wait_tick1 turn=${effect.ctx.turnId}") }

                    val job = scope.launch {
                        val t0 = SystemClock.elapsedRealtime()
                        try {
                            if (isStale(reqSeq)) return@launch

                            // ---- Tick1 START telemetry + payload sizes ----
                            emitStage(
                                stage = "TICK1",
                                phase = "START",
                                ctx = effect.ctx,
                                extra = mapOf(
                                    "tick1_user_text_chars" to effect.userText.length,
                                    "tick1_user_text_bytes" to bytesOf(effect.userText),
                                    "tick1_state_header_chars" to effect.stateHeader.length,
                                    "tick1_state_header_bytes" to bytesOf(effect.stateHeader),
                                    "tick1_grid_chars" to (effect.gridContext?.toString()?.length ?: 0),
                                    "tick1_grid_bytes" to bytesOf(effect.gridContext?.toString())
                                )
                            )

                            val isGridSession = (effect.mode == SudoMode.GRID_SESSION)

                            val toolsOrNull: List<ToolCall>? = if (!isGridSession) {
                                // FREE_TALK: tool-schema is decommissioned; deterministic fallback
                                emitPolicyTrace(
                                    tag = "PHASE6_CALLPOLICY_FREE_TALK_DECOMMISSIONED",
                                    kv = mapOf(
                                        "where" to "AndroidEffectRunner.CallPolicy.free_talk",
                                        "turnId" to effect.ctx.turnId,
                                        "tickId" to effect.ctx.tickId,
                                        "policyReqSeq" to effect.ctx.policyReqSeq,
                                        "mode" to (effect.mode?.name ?: effect.ctx.mode)
                                    )
                                )
                                listOf(ToolCall.Reply(fm02FallbackReplyText()))
                            } else {
                                // ✅ GRID_SESSION Tick1 = IntentEnvelopeV1 ONLY
                                val adapter = policy as? com.contextionary.sudoku.conductor.policy.CoordinatorPolicyAdapter
                                if (adapter == null) {
                                    emitPolicyTrace(
                                        tag = "TICK1_ENV_V1_MISSING_ADAPTER",
                                        kv = mapOf(
                                            "where" to "AndroidEffectRunner.CallPolicy.grid_session",
                                            "turnId" to effect.ctx.turnId,
                                            "tickId" to effect.ctx.tickId,
                                            "policyReqSeq" to effect.ctx.policyReqSeq
                                        )
                                    )
                                    null
                                } else {
                                    val env: IntentEnvelopeV1? = runCatching {
                                        adapter.decideIntentEnvelopeV1(
                                            sessionId = effect.ctx.sessionId,
                                            turnId = effect.ctx.turnId,
                                            tickId = effect.ctx.tickId,
                                            correlationId = effect.ctx.correlationId,
                                            policyReqSeq = effect.ctx.policyReqSeq,
                                            modelCallId = effect.ctx.modelCallId,
                                            toolplanId = effect.ctx.toolplanId,
                                            userText = effect.userText,
                                            stateHeader = effect.stateHeader
                                        )
                                    }.getOrElse { t ->
                                        emitPolicyTrace(
                                            tag = "TICK1_ENV_V1_EXCEPTION",
                                            kv = mapOf(
                                                "where" to "AndroidEffectRunner.CallPolicy.grid_session",
                                                "turnId" to effect.ctx.turnId,
                                                "tickId" to effect.ctx.tickId,
                                                "policyReqSeq" to effect.ctx.policyReqSeq,
                                                "ex" to (t.javaClass.simpleName ?: "Throwable"),
                                                "msg" to (t.message ?: "")
                                            )
                                        )
                                        null
                                    }

                                    if (env != null) {
                                        val env2 = env.normalizeCompat(effect.userText)

                                        // Contract telemetry/event
                                        emitTick1IntentEnvelopeV1(ctx = effect.ctx, rawUserText = effect.userText, env = env2)

                                        // ✅ Single contract event for downstream reducer
                                        emit(Evt.IntentEnvelopeReceived(effect.ctx, effect.userText, env2))
                                    } else {
                                        emitPolicyTrace(
                                            tag = "TICK1_ENV_V1_NULL",
                                            kv = mapOf(
                                                "where" to "AndroidEffectRunner.CallPolicy.grid_session",
                                                "turnId" to effect.ctx.turnId,
                                                "tickId" to effect.ctx.tickId,
                                                "policyReqSeq" to effect.ctx.policyReqSeq
                                            )
                                        )
                                    }

                                    // Return null so the rest won’t emit PolicyTools in GRID_SESSION.
                                    null
                                }
                            }

                            // ---- Tick1 END telemetry ----
                            emitStage(
                                stage = "TICK1",
                                phase = "END",
                                ctx = effect.ctx,
                                elapsedMs = (SystemClock.elapsedRealtime() - t0),
                                extra = mapOf(
                                    "tick1_tools_null" to (toolsOrNull == null),
                                    "tick1_tools_count" to (toolsOrNull?.size ?: 0)
                                )
                            )

                            if (isStale(reqSeq)) return@launch

                            if (!isGridSession) {
                                val finalTools = if (!toolsOrNull.isNullOrEmpty()) {
                                    toolsOrNull
                                } else {
                                    emitPolicyTrace(
                                        tag = "PHASE6_UNEXPECTED_EMPTY_TOOLPLAN",
                                        kv = mapOf(
                                            "where" to "AndroidEffectRunner.CallPolicy",
                                            "turnId" to effect.ctx.turnId,
                                            "tickId" to effect.ctx.tickId,
                                            "policyReqSeq" to effect.ctx.policyReqSeq,
                                            "toolplanId" to effect.ctx.toolplanId
                                        )
                                    )
                                    listOf(ToolCall.Reply(fm02FallbackReplyText()))
                                }

                                if (isStale(reqSeq)) return@launch

                                val toolCallIds = finalTools.indices.map { i ->
                                    "tc:${effect.ctx.turnId}:${effect.ctx.tickId}:${effect.ctx.policyReqSeq}:${effect.ctx.toolplanId}:$i"
                                }

                                emit(
                                    Evt.PolicyTools(
                                        ctx = effect.ctx,
                                        tools = finalTools,
                                        toolCallIds = toolCallIds,
                                        turnId = effect.ctx.turnId,
                                        tickId = effect.ctx.tickId,
                                        policyReqSeq = effect.ctx.policyReqSeq,
                                        modelCallId = effect.ctx.modelCallId,
                                        toolplanId = effect.ctx.toolplanId,
                                        correlationId = effect.ctx.correlationId,
                                        diag = null
                                    )
                                )
                            }
                        } catch (ce: CancellationException) {
                            emitPolicyTrace(
                                tag = "POLICY_JOB_CANCELLED",
                                kv = mapOf(
                                    "where" to "AndroidEffectRunner.CallPolicy",
                                    "turnId" to effect.ctx.turnId,
                                    "tickId" to effect.ctx.tickId,
                                    "policyReqSeq" to reqSeq,
                                    "msg" to (ce.message ?: "cancelled")
                                )
                            )
                        } catch (t: Throwable) {
                            emitPolicyTrace(
                                tag = "POLICY_JOB_CRASH",
                                kv = mapOf(
                                    "where" to "AndroidEffectRunner.CallPolicy",
                                    "turnId" to effect.ctx.turnId,
                                    "tickId" to effect.ctx.tickId,
                                    "policyReqSeq" to reqSeq,
                                    "ex" to (t.javaClass.simpleName ?: "Throwable"),
                                    "msg" to (t.message ?: "")
                                )
                            )
                        }
                    }

                    setActivePolicyJob(job)
                }

                is Eff.CallPolicyContinuationTick2 -> {
                    val reqSeq = effect.ctx.policyReqSeq
                    beginPolicyFlight(reqSeq, label = "CallPolicyContinuationTick2_DEPRECATED")

                    runCatching { host.stopAsrCompat(reason = "phase5_blocked_tick2_deprecated turn=${effect.ctx.turnId}") }

                    runCatching {
                        emitPolicyTrace(
                            tag = "PHASE5_BLOCKED_TICK2_CONTINUATION_EFFECT",
                            kv = mapOf(
                                "where" to "AndroidEffectRunner.run",
                                "turnId" to effect.ctx.turnId,
                                "tickId" to effect.ctx.tickId,
                                "policyReqSeq" to effect.ctx.policyReqSeq,
                                "toolplanId" to effect.ctx.toolplanId,
                                "modelCallId" to effect.ctx.modelCallId,
                                "correlationId" to effect.ctx.correlationId,
                                "mode" to effect.ctx.mode,
                                "reason" to effect.reason
                            )
                        )
                    }

                    emit(Evt.ReplyReceived(text = fm02FallbackReplyText(), source = "phase5_blocked_tick2"))
                }

                // ============================================================
                // ✅ B) Retire MeaningExtract shim path:
                // Eff.CallMeaningExtractV1 now performs the SAME Tick1 envelope call
                // ============================================================
                is Eff.CallMeaningExtractV1 -> {
                    val reqSeq = effect.ctx.policyReqSeq
                    beginPolicyFlight(reqSeq, label = "CallIntentEnvelopeV1") // renamed label for clarity

                    runCatching { host.stopAsrCompat(reason = "policy_wait_tick1_env_v1 turn=${effect.ctx.turnId}") }

                    val job = scope.launch {
                        val t0 = SystemClock.elapsedRealtime()
                        try {
                            if (isStale(reqSeq)) return@launch

                            emitStage(
                                stage = "TICK1_INTENT_ENVELOPE_V1",
                                phase = "START",
                                ctx = effect.ctx,
                                extra = mapOf("user_text_len" to effect.userText.length)
                            )

                            val adapter = policy as? com.contextionary.sudoku.conductor.policy.CoordinatorPolicyAdapter
                            if (adapter == null) {
                                emitPolicyTrace(
                                    tag = "TICK1_ENV_V1_NO_ADAPTER",
                                    kv = mapOf(
                                        "turnId" to effect.ctx.turnId,
                                        "tickId" to 1,
                                        "policyReqSeq" to effect.ctx.policyReqSeq
                                    )
                                )
                                return@launch
                            }

                            val envRaw: IntentEnvelopeV1 = runCatching {
                                adapter.decideIntentEnvelopeV1(
                                    sessionId = effect.ctx.sessionId,
                                    turnId = effect.ctx.turnId,
                                    tickId = 1,
                                    correlationId = effect.ctx.correlationId,
                                    policyReqSeq = effect.ctx.policyReqSeq,
                                    modelCallId = effect.ctx.modelCallId,
                                    toolplanId = effect.ctx.toolplanId,
                                    userText = effect.userText,
                                    stateHeader = effect.stateHeader
                                )
                            }.getOrElse { t ->
                                emitPolicyTrace(
                                    tag = "TICK1_ENV_V1_EXCEPTION",
                                    kv = mapOf(
                                        "turnId" to effect.ctx.turnId,
                                        "tickId" to 1,
                                        "policyReqSeq" to effect.ctx.policyReqSeq,
                                        "ex" to (t.javaClass.simpleName ?: "Throwable"),
                                        "msg" to (t.message ?: "unknown")
                                    )
                                )
                                null
                            } ?: return@launch

                            emitStage(
                                stage = "TICK1_INTENT_ENVELOPE_V1",
                                phase = "END",
                                ctx = effect.ctx,
                                elapsedMs = (SystemClock.elapsedRealtime() - t0),
                                extra = mapOf(
                                    "env_intents_n" to envRaw.intents.size,
                                    "env_top_type" to (envRaw.intents.firstOrNull()?.type?.name ?: "NONE"),
                                    "env_top_conf" to (envRaw.intents.firstOrNull()?.confidence ?: 0.0),
                                    "env_top_missing_n" to (envRaw.intents.firstOrNull()?.missing?.size ?: 0)
                                )
                            )

                            if (isStale(reqSeq)) return@launch

                            val env2 = envRaw.normalizeCompat(effect.userText)

                            emitTick1IntentEnvelopeV1(
                                ctx = effect.ctx,
                                rawUserText = effect.userText,
                                env = env2
                            )

                            emit(Evt.IntentEnvelopeReceived(effect.ctx, effect.userText, env2))

                        } catch (t: Throwable) {
                            emitPolicyTrace(
                                tag = "TICK1_ENV_V1_CRASH",
                                kv = mapOf(
                                    "turnId" to effect.ctx.turnId,
                                    "policyReqSeq" to effect.ctx.policyReqSeq,
                                    "ex" to (t.javaClass.simpleName ?: "Throwable"),
                                    "msg" to (t.message ?: "unknown")
                                )
                            )
                        }
                    }

                    setActivePolicyJob(job)
                }

                is Eff.CallReplyGenerateV1 -> {
                    val reqSeq = effect.ctx.policyReqSeq
                    beginPolicyFlight(reqSeq, label = "CallReplyGenerateV1")


                    // ✅ Phase 0: duplicate Tick2 detection
                    val tick2CallsNow = bumpTick2Calls(effect.ctx.turnId)
                    runCatching {
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TICK2_CALL_OBSERVED",
                                "session_id" to effect.ctx.sessionId,
                                "turn_id" to effect.ctx.turnId,
                                "tick_id" to effect.ctx.tickId,
                                "policy_req_seq" to effect.ctx.policyReqSeq,
                                "correlation_id" to effect.ctx.correlationId,
                                "model_call_id" to effect.ctx.modelCallId,
                                "toolplan_id" to effect.ctx.toolplanId,
                                "tick2_calls_for_turn" to tick2CallsNow,
                                "reason" to effect.reason
                            )
                        )

// ✅ Phase 0: Joinable counter event (merge with TURN_CONTRACT_SNAPSHOT via turn_id)
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TURN_TICK2_COUNTER",
                                "session_id" to effect.ctx.sessionId,
                                "turn_id" to effect.ctx.turnId,
                                "tick_id" to effect.ctx.tickId,
                                "policy_req_seq" to effect.ctx.policyReqSeq,
                                "correlation_id" to effect.ctx.correlationId,
                                "model_call_id" to effect.ctx.modelCallId,
                                "toolplan_id" to effect.ctx.toolplanId,
                                "tick2_calls_for_turn" to tick2CallsNow,
                                "reason" to "CallReplyGenerateV1"
                            )
                        )
                    }
                    if (tick2CallsNow > 1) {
                        runCatching {
                            ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "TICK2_DUPLICATE_DETECTED",
                                    "session_id" to effect.ctx.sessionId,
                                    "turn_id" to effect.ctx.turnId,
                                    "tick_id" to effect.ctx.tickId,
                                    "policy_req_seq" to effect.ctx.policyReqSeq,
                                    "tick2_calls_for_turn" to tick2CallsNow,
                                    "label" to "duplicate_tick2_same_turn"
                                )
                            )
                        }

                        // Phase 0 invariant: exactly one Tick2 call per turn.
                        // Hard-block duplicates to prevent double-speaking / double-CTA.
                        emit(Evt.ReplyReceived(text = fm02FallbackReplyText(), source = "tick2_duplicate_blocked"))
                        return
                    }

                    runCatching { host.stopAsrCompat(reason = "policy_wait_tick2_v1 turn=${effect.turnId}") }

                    val job = scope.launch {
                        val t0 = SystemClock.elapsedRealtime()
                        try {
                            if (isStale(reqSeq)) return@launch

                            val rrJson = runCatching { effect.replyRequest.toJson().toString() }.getOrElse { "{}" }
                            val rrSha12 = runCatching { sha12(rrJson) }.getOrElse { "sha12_err" }

                            runCatching {
                                ConversationTelemetry.emit(
                                    mapOf(
                                        "type" to "REPLY_REQUEST_V1_OUT",
                                        "session_id" to effect.ctx.sessionId,
                                        "turn_id" to effect.ctx.turnId,
                                        "tick_id" to effect.ctx.tickId,
                                        "policy_req_seq" to effect.ctx.policyReqSeq,
                                        "correlation_id" to effect.ctx.correlationId,
                                        "model_call_id" to effect.ctx.modelCallId,
                                        "toolplan_id" to effect.ctx.toolplanId,
                                        "rr_len" to rrJson.length,
                                        "rr_sha12" to rrSha12,
                                        "rr_json" to (if (false) rrJson else null)
                                    )
                                )
                            }

// ✅ Phase 0: Tick2 Evidence Manifest (joinable, compact, audit-friendly)
                            runCatching {
                                val root = org.json.JSONObject(rrJson)
                                val turn = root.optJSONObject("turn") ?: org.json.JSONObject()
                                val factsArr = root.optJSONArray("facts") ?: org.json.JSONArray()

                                // Find STORY_CONTEXT_V1 payload (if present)
                                var storyStage: String? = null
                                var focusAtomIndex: Int? = null
                                val discussedAtoms = mutableListOf<Int>()
                                val requiredOverlayIds = mutableListOf<String>()

                                // Find SOLVING_STEP_PACKET_V1 overlays.applied_frame_ids (if present)
                                val appliedOverlayIds = mutableListOf<String>()

                                // Build per-fact manifest
                                val factManifest = org.json.JSONArray()

                                for (i in 0 until factsArr.length()) {
                                    val f = factsArr.optJSONObject(i) ?: continue
                                    val t = f.optString("type", "")
                                    val p = f.optJSONObject("payload")

                                    val pStr = p?.toString() ?: ""
                                    val len = pStr.length
                                    val sha = if (p != null) sha12(pStr) else ""

                                    val row = org.json.JSONObject()
                                        .put("type", t)
                                        .put("payload_len", len)
                                        .put("payload_sha12", sha)

                                    val schema = p?.optString("schema_version", "") ?: ""
                                    if (schema.isNotBlank()) row.put("schema_version", schema)

                                    // Heuristic: pick common step id fields if present
                                    val stepId =
                                        p?.optString("step_id", "")?.takeIf { it.isNotBlank() }
                                            ?: p?.optJSONObject("step")?.optString("step_id", "")?.takeIf { it.isNotBlank() }
                                    if (!stepId.isNullOrBlank()) row.put("step_id", stepId)

                                    factManifest.put(row)

                                    // Parse story context
                                    if (t == "STORY_CONTEXT_V1" && p != null) {
                                        storyStage = p.optString("story_stage", null)
                                        focusAtomIndex = if (p.has("focus_atom_index") && !p.isNull("focus_atom_index")) p.optInt("focus_atom_index") else null

                                        val disc = p.optJSONArray("discussed_atom_indices")
                                        if (disc != null) {
                                            for (di in 0 until disc.length()) {
                                                val v = disc.optInt(di, -1)
                                                if (v >= 0) discussedAtoms.add(v)
                                            }
                                        }

                                        val req = p.optJSONArray("required_overlay_frame_ids")
                                        if (req != null) {
                                            for (ri in 0 until req.length()) {
                                                val s = req.optString(ri, "")
                                                if (s.isNotBlank()) requiredOverlayIds.add(s)
                                            }
                                        }
                                    }

                                    // Parse applied overlay ids from step packet v1
                                    if (t == "SOLVING_STEP_PACKET_V1" && p != null) {
                                        val overlays = p.optJSONObject("overlays")
                                        val applied = overlays?.optJSONArray("applied_frame_ids")
                                        if (applied != null) {
                                            for (ai in 0 until applied.length()) {
                                                val s = applied.optString(ai, "")
                                                if (s.isNotBlank()) appliedOverlayIds.add(s)
                                            }
                                        }
                                    }
                                }

                                ConversationTelemetry.emit(
                                    mapOf(
                                        "type" to "TICK2_EVIDENCE_MANIFEST",
                                        "session_id" to effect.ctx.sessionId,
                                        "turn_id" to effect.ctx.turnId,
                                        "tick_id" to effect.ctx.tickId,
                                        "policy_req_seq" to effect.ctx.policyReqSeq,
                                        "correlation_id" to effect.ctx.correlationId,
                                        "model_call_id" to effect.ctx.modelCallId,
                                        "toolplan_id" to effect.ctx.toolplanId,

                                        "mode" to turn.optString("mode", ""),
                                        "phase" to turn.optString("phase", ""),
                                        "pending_before" to turn.optString("pending_before", ""),
                                        "pending_after" to turn.optString("pending_after", ""),

                                        "story_stage" to (storyStage ?: ""),
                                        "focus_atom_index" to (focusAtomIndex ?: -1),
                                        "discussed_atom_indices" to discussedAtoms.joinToString(","),
                                        "required_overlay_frame_ids" to requiredOverlayIds.joinToString(","),
                                        "applied_overlay_frame_ids" to appliedOverlayIds.joinToString(","),

                                        "facts_count" to factsArr.length(),
                                        "facts_manifest_json" to factManifest.toString()
                                    )
                                )
                            }.onFailure { t ->
                                runCatching {
                                    ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "TICK2_EVIDENCE_MANIFEST_FAILED",
                                            "session_id" to effect.ctx.sessionId,
                                            "turn_id" to effect.ctx.turnId,
                                            "tick_id" to effect.ctx.tickId,
                                            "policy_req_seq" to effect.ctx.policyReqSeq,
                                            "ex" to (t.javaClass.simpleName ?: "Throwable"),
                                            "msg" to (t.message ?: "unknown")
                                        )
                                    )
                                }
                            }

                            emitStage(
                                stage = "TICK2_V1",
                                phase = "START",
                                ctx = effect.ctx,
                                extra = mapOf(
                                    "tick2_req_chars" to rrJson.length,
                                    "tick2_req_bytes" to bytesOf(rrJson),
                                    "tick2_req_sha12" to rrSha12
                                )
                            )

                            val adapter = policy as? com.contextionary.sudoku.conductor.policy.CoordinatorPolicyAdapter
                            val text = if (adapter != null) {
                                adapter.generateReplyV1(
                                    ctx = effect.ctx,
                                    replyRequest = effect.replyRequest,
                                    planResult = effect.planResult
                                )
                            } else {
                                fm02FallbackReplyText()
                            }

                            emitStage(
                                stage = "TICK2_V1",
                                phase = "END",
                                ctx = effect.ctx,
                                elapsedMs = (SystemClock.elapsedRealtime() - t0),
                                extra = mapOf("reply_len" to text.length)
                            )

                            if (isStale(reqSeq)) return@launch

                            emit(Evt.ReplyReceived(text = text, source = "tick2_v1"))

                        } catch (t: Throwable) {
                            emitPolicyTrace(
                                tag = "TICK2_V1_CRASH",
                                kv = mapOf(
                                    "turnId" to effect.ctx.turnId,
                                    "policyReqSeq" to effect.ctx.policyReqSeq,
                                    "ex" to (t.javaClass.simpleName ?: "Throwable"),
                                    "msg" to (t.message ?: "")
                                )
                            )
                            emit(Evt.ReplyReceived(text = fm02FallbackReplyText(), source = "tick2_v1_fallback"))
                        }
                    }

                    setActivePolicyJob(job)
                }

                else -> {
                    Log.w("AndroidEffectRunner", "Unhandled effect: ${effect::class.java.simpleName} sid=$sid")
                    runCatching {
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "EFFECT_UNHANDLED",
                                "session_id" to sid,
                                "effect" to effect::class.java.simpleName
                            )
                        )
                    }
                }
            }
        } catch (t: Throwable) {
            Log.e("AndroidEffectRunner", "EffectRunner.run crashed", t)
        }
    }

    override fun applyTools(tools: List<ToolCall>, emit: (Evt) -> Unit) {
        // Preserve MainActivity behavior for "manual apply"
        val sessionId = sid
        val turnId = -1L
        val tickId = 0
        val modeStr = "APPLY"
        val reasonStr = "applyTools"
        val stateHeader: String? = null

        val ctx = runCatching {
            val policyReqSeq = ConversationTelemetry.nextPolicyReqSeq(sessionId, turnId, tickId)
            val modelCallId = ConversationTelemetry.nextModelCallId("apply")
            val toolplanId = ConversationTelemetry.nextToolplanId("tp")
            val correlationId = "turn-$turnId"

            val headerSha12: String? = stateHeader?.let {
                runCatching { ConversationTelemetry.sha256Hex(it).take(12) }.getOrNull()
            }

            PolicyCallCtx(
                sessionId = sessionId,
                turnId = turnId,
                tickId = tickId,
                policyReqSeq = policyReqSeq,
                correlationId = correlationId,
                modelCallId = modelCallId,
                toolplanId = toolplanId,
                mode = modeStr,
                reason = reasonStr,
                stateHeaderSha12 = headerSha12
            )
        }.getOrElse { t ->
            PolicyCallCtx(
                sessionId = sessionId,
                turnId = turnId,
                tickId = tickId,
                policyReqSeq = 0L,
                correlationId = "turn-$turnId",
                modelCallId = "mc_apply_fallback",
                toolplanId = "tp_apply_fallback",
                mode = modeStr,
                reason = "applyTools_ctx_failed:${t.javaClass.simpleName}",
                stateHeaderSha12 = null
            )
        }

        val finalTools = if (isMeaningfulToolplan(tools)) tools else listOf(ToolCall.Reply(fm02FallbackReplyText()))
        val toolCallIds = finalTools.indices.map { i ->
            "tc:${ctx.turnId}:${ctx.tickId}:${ctx.policyReqSeq}:${ctx.toolplanId}:$i"
        }

        runCatching {
            emit(
                Evt.PolicyTools(
                    ctx = ctx,
                    tools = finalTools,
                    toolCallIds = toolCallIds,
                    turnId = ctx.turnId,
                    tickId = ctx.tickId,
                    policyReqSeq = ctx.policyReqSeq,
                    modelCallId = ctx.modelCallId,
                    toolplanId = ctx.toolplanId,
                    correlationId = ctx.correlationId,
                    diag = null
                )
            )
        }.onFailure {
            Log.e("AndroidEffectRunner", "applyTools crashed", it)
        }
    }
}