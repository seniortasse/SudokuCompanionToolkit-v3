package com.contextionary.sudoku.conductor

import android.os.SystemClock
import android.util.Log
import com.contextionary.sudoku.logic.LLMGridState
import com.contextionary.sudoku.telemetry.ConversationTelemetry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * FM-02 hard rule:
 * - Policy failures MUST NOT create ToolCall.Reply fallback text.
 * - No local user-facing chat bubbles or TTS from store.
 * - On error/empty-tools: emit telemetry + silently retry (bounded) OR do nothing.
 */
class SudoStore(
    initial: SudoState,
    private val conductor: SudoConductor,
    private val policy: LlmPolicy,
    private val effects: EffectRunner,
    private val scope: CoroutineScope
) {
    private val _state = MutableStateFlow(initial)
    val state: StateFlow<SudoState> = _state

    private val evtCh = Channel<Evt>(capacity = Channel.UNLIMITED)

    private val tag = "SudoStore"

    // Sequence numbers to correlate “dispatch → process → effects → policy → tools”
    private val dispatchSeq = AtomicLong(0)
    private val processSeq = AtomicLong(0)
    private val policyReqSeq = AtomicLong(0)

    // ✅ Gate: last-write-wins for concurrent policy calls (drop stale responses)
    private val latestPolicyIssuedSeq = AtomicLong(0)

    // ✅ FM-02: bounded silent retries for policy failures / empty tools
    private val policyRetryCount = ConcurrentHashMap<Long, Int>()
    private val MAX_POLICY_RETRIES = 2

    init {
        scope.launch {
            Log.i(tag, "START sessionId=${initial.sessionId} mode=${initial.mode} turnSeq=${initial.turnSeq}")

            for (e in evtCh) {
                val procId = processSeq.incrementAndGet()

                val s0 = _state.value
                val before = summarizeState(s0)

                Log.i(tag, "EVT#$procId ${evtSummary(e)}  BEFORE: $before")

                val t0 = SystemClock.elapsedRealtime()

                val next: SudoConductor.Next = try {
                    conductor.reduce(s0, e)
                } catch (t: Throwable) {
                    Log.e(tag, "REDUCE_CRASH EVT#$procId ${evtSummary(e)}", t)
                    // Keep running: no state change, no effects
                    SudoConductor.Next(s0, emptyList())
                }

                // ----------------------------
                // ✅ NEW: derive UI focus cell from pending and store it in state
                // ----------------------------
                val desiredFocus: Int? = focusFromPending(next.state.pending)

                val stateWithFocus =
                    if (next.state.focusCellIndex != desiredFocus) next.state.copy(focusCellIndex = desiredFocus)
                    else next.state

                _state.value = stateWithFocus

                // ----------------------------
                // ✅ NEW: auto-inject SetFocusCell effect (unless conductor already emitted one)
                // Put it FIRST so highlight is visible before Speak/Ask.
                // ----------------------------
                val effs: MutableList<Eff> = next.effects.toMutableList()
                val hasExplicitFocusEff = effs.any { it is Eff.SetFocusCell }

                // Compare against previous state's focus (s0) to decide whether to emit.
                val focusChanged = (s0.focusCellIndex != desiredFocus)

                if (!hasExplicitFocusEff && focusChanged) {
                    effs.add(
                        0,
                        Eff.SetFocusCell(
                            cellIndex = desiredFocus,
                            reason = "auto_from_pending"
                        )
                    )
                }

                val after = summarizeState(_state.value)
                val dt = SystemClock.elapsedRealtime() - t0

                Log.i(tag, "EVT#$procId REDUCED in ${dt}ms  AFTER: $after")

                if (effs.isEmpty()) {
                    Log.i(tag, "EVT#$procId EFFECTS: (none)")
                    continue
                }

                Log.i(tag, "EVT#$procId EFFECTS: ${effectsSummary(effs)}")

                effs.forEachIndexed { idx, eff ->
                    when (eff) {
                        is Eff.CallPolicy -> {
                            val reqSeq = policyReqSeq.incrementAndGet()
                            latestPolicyIssuedSeq.set(reqSeq)
                            policyRetryCount.remove(reqSeq) // fresh call resets retry count

                            // Capture a "live" snapshot of state at the boundary
                            val sNow = _state.value
                            val userPrev = eff.userText.replace("\n", " ").trim().take(220)
                            val headerPrev = eff.stateHeader.replace("\n", " ").trim().take(220)

                            val effGrid: LLMGridState? = eff.gridContext
                            val liveGrid: LLMGridState? = sNow.grid?.llm

                            val effGridHash = computeGridHashSafe(effGrid)
                            val liveGridHash = computeGridHashSafe(liveGrid)

                            // Chosen grid is ALWAYS the live store grid when present (Gate-2 robustness).
                            val gridToUse: LLMGridState? = liveGrid ?: effGrid
                            val chosenGridHash = computeGridHashSafe(gridToUse)
                            val gridPresent = (gridToUse != null)

                            // Gate-2 alarm: Effect carried a different snapshot than the store currently has.
                            if (effGridHash != null && liveGridHash != null && effGridHash != liveGridHash) {
                                ConversationTelemetry.emitKv(
                                    "LLM_CALLPOLICY_GRID_MISMATCH",
                                    "policy_req_seq" to reqSeq,
                                    "evt_proc_id" to procId,
                                    "eff_index" to idx,
                                    "session_id" to sNow.sessionId,
                                    "turn_id" to sNow.turnSeq,
                                    "eff_grid_hash" to effGridHash,
                                    "live_grid_hash" to liveGridHash,
                                    "chosen_grid_hash" to chosenGridHash
                                )
                                Log.w(tag, "POLICY#$reqSeq GRID_MISMATCH effHash=$effGridHash liveHash=$liveGridHash (using live)")
                            }

                            // Patch-0 evidence: CallPolicy boundary
                            ConversationTelemetry.emitKv(
                                "LLM_CALLPOLICY_BEGIN",
                                "policy_req_seq" to reqSeq,
                                "evt_proc_id" to procId,
                                "eff_index" to idx,
                                "session_id" to sNow.sessionId,
                                "turn_id" to sNow.turnSeq,
                                "mode" to sNow.mode.name,
                                "user_text" to userPrev,
                                "state_header_preview" to headerPrev,
                                "grid_present" to gridPresent,
                                "eff_grid_hash" to (effGridHash ?: ""),
                                "live_grid_hash" to (liveGridHash ?: ""),
                                "chosen_grid_hash" to (chosenGridHash ?: "")
                            )

                            // Patch-0 requirement: EFFECT_RUN(CallPolicy)
                            ConversationTelemetry.emitKv(
                                "EFFECT_RUN",
                                "effect" to "CallPolicy",
                                "policy_req_seq" to reqSeq,
                                "evt_proc_id" to procId,
                                "eff_index" to idx,
                                "session_id" to sNow.sessionId,
                                "turn_id" to sNow.turnSeq,
                                "mode" to sNow.mode.name,
                                "grid_present" to gridPresent,
                                "chosen_grid_hash" to (chosenGridHash ?: ""),
                                "user_preview" to userPrev.take(120)
                            )

                            Log.i(
                                tag,
                                "POLICY#$reqSeq BEGIN (from EVT#$procId eff[$idx]) " +
                                        "sessionId=${sNow.sessionId} turnSeq=${sNow.turnSeq} " +
                                        "grid=$gridPresent chosenHash=${chosenGridHash ?: "null"} " +
                                        "user='${userPrev.take(80)}' header='${headerPrev.take(120)}'"
                            )

                            fun emitEffectDoneOk(ms: Long, tools: List<ToolCall>) {
                                val hasReply = tools.any { it is ToolCall.Reply }
                                ConversationTelemetry.emitKv(
                                    "EFFECT_DONE",
                                    "effect" to "CallPolicy",
                                    "policy_req_seq" to reqSeq,
                                    "evt_proc_id" to procId,
                                    "eff_index" to idx,
                                    "session_id" to _state.value.sessionId,
                                    "turn_id" to _state.value.turnSeq,
                                    "ms" to ms,
                                    "tools_n" to tools.size,
                                    "has_reply" to hasReply,
                                    "chosen_grid_hash" to (chosenGridHash ?: "")
                                )
                            }

                            fun emitEffectDoneErr(ms: Long, err: String) {
                                ConversationTelemetry.emitKv(
                                    "EFFECT_DONE_ERR",
                                    "effect" to "CallPolicy",
                                    "policy_req_seq" to reqSeq,
                                    "evt_proc_id" to procId,
                                    "eff_index" to idx,
                                    "session_id" to _state.value.sessionId,
                                    "turn_id" to _state.value.turnSeq,
                                    "ms" to ms,
                                    "err" to err.take(220),
                                    "chosen_grid_hash" to (chosenGridHash ?: "")
                                )
                            }

                            fun scheduleRetry(reason: String, errMsg: String? = null) {
                                val attempt = (policyRetryCount[reqSeq] ?: 0) + 1
                                policyRetryCount[reqSeq] = attempt

                                ConversationTelemetry.emitKv(
                                    "LLM_CALLPOLICY_RETRY_SCHEDULED",
                                    "policy_req_seq" to reqSeq,
                                    "evt_proc_id" to procId,
                                    "session_id" to _state.value.sessionId,
                                    "turn_id" to _state.value.turnSeq,
                                    "attempt" to attempt,
                                    "reason" to reason,
                                    "err" to (errMsg ?: "")
                                )

                                if (attempt > MAX_POLICY_RETRIES) {
                                    Log.e(tag, "POLICY#$reqSeq GIVE_UP after $attempt attempts reason=$reason")
                                    ConversationTelemetry.emitKv(
                                        "LLM_CALLPOLICY_GIVE_UP",
                                        "policy_req_seq" to reqSeq,
                                        "evt_proc_id" to procId,
                                        "session_id" to _state.value.sessionId,
                                        "turn_id" to _state.value.turnSeq,
                                        "attempt" to attempt,
                                        "reason" to reason
                                    )
                                    return
                                }

                                val backoffMs = 250L * attempt // simple linear backoff
                                scope.launch {
                                    delay(backoffMs)

                                    // Drop stale retries too
                                    if (latestPolicyIssuedSeq.get() != reqSeq) {
                                        Log.w(tag, "POLICY#$reqSeq RETRY_STALE (dropping) reason=$reason")
                                        return@launch
                                    }

                                    Log.w(tag, "POLICY#$reqSeq RETRY attempt=$attempt after=${backoffMs}ms reason=$reason")

                                    val p0 = SystemClock.elapsedRealtime()
                                    ConversationTelemetry.emitKv(
                                        "EFFECT_RUN",
                                        "effect" to "CallPolicyRetry",
                                        "policy_req_seq" to reqSeq,
                                        "evt_proc_id" to procId,
                                        "eff_index" to idx,
                                        "attempt" to attempt,
                                        "reason" to reason
                                    )

                                    try {
                                        val tools = policy.decide(
                                            sessionId = sNow.sessionId,
                                            userText = eff.userText,
                                            stateHeader = eff.stateHeader + "\npolicy_retry=$attempt\npolicy_retry_reason=$reason",
                                            grid = gridToUse
                                        )

                                        val pdt = SystemClock.elapsedRealtime() - p0

                                        // Drop stale results
                                        if (latestPolicyIssuedSeq.get() != reqSeq) {
                                            Log.w(tag, "POLICY#$reqSeq RETRY_RESULT_STALE (dropping) tools=${tools.size}")
                                            ConversationTelemetry.emitKv(
                                                "LLM_CALLPOLICY_STALE_DROP",
                                                "policy_req_seq" to reqSeq,
                                                "evt_proc_id" to procId,
                                                "session_id" to _state.value.sessionId,
                                                "turn_id" to _state.value.turnSeq,
                                                "ms" to pdt,
                                                "tools_n" to tools.size
                                            )
                                            return@launch
                                        }

                                        emitEffectDoneOk(pdt, tools)

                                        val meaningful = tools.any { it !is ToolCall.Noop }
                                        if (!meaningful) {
                                            scheduleRetry(reason = "empty_tools_on_retry_$attempt", errMsg = null)
                                            return@launch
                                        }

                                        val hasReply = tools.any { it is ToolCall.Reply }
                                        if (!hasReply) {
                                            ConversationTelemetry.emitKv(
                                                "POLICY_MISSING_REPLY",
                                                "policy_req_seq" to reqSeq,
                                                "evt_proc_id" to procId,
                                                "session_id" to _state.value.sessionId,
                                                "turn_id" to _state.value.turnSeq,
                                                "attempt" to attempt,
                                                "tools_n" to tools.size,
                                                "tools" to toolsSummary(tools)
                                            )
                                            scheduleRetry(reason = "missing_reply_on_retry_$attempt", errMsg = null)
                                            return@launch
                                        }

                                        Log.i(tag, "POLICY#$reqSeq RETRY_OK tools=${tools.size} ${toolsSummary(tools)}")
                                        effects.applyTools(tools) { followUpEvt ->
                                            Log.i(tag, "POLICY#$reqSeq APPLYTOOLS_EMIT(RETRY) -> ${evtSummary(followUpEvt)}")
                                            dispatch(followUpEvt)
                                        }
                                    } catch (t: Throwable) {
                                        val pdt = SystemClock.elapsedRealtime() - p0
                                        Log.e(tag, "POLICY#$reqSeq RETRY_ERR attempt=$attempt reason=$reason", t)
                                        emitEffectDoneErr(pdt, t.message ?: t.toString())
                                        scheduleRetry(reason = "exception_on_retry_$attempt", errMsg = t.message ?: t.toString())
                                    }
                                }
                            }

                            scope.launch {
                                val p0 = SystemClock.elapsedRealtime()
                                try {
                                    val tools = policy.decide(
                                        sessionId = sNow.sessionId,
                                        userText = eff.userText,
                                        stateHeader = eff.stateHeader,
                                        grid = gridToUse
                                    )
                                    val pdt = SystemClock.elapsedRealtime() - p0

                                    // ✅ Drop stale policy results (last-write-wins)
                                    if (latestPolicyIssuedSeq.get() != reqSeq) {
                                        Log.w(tag, "POLICY#$reqSeq STALE (dropping) ms=$pdt tools=${tools.size}")
                                        ConversationTelemetry.emitKv(
                                            "LLM_CALLPOLICY_STALE_DROP",
                                            "policy_req_seq" to reqSeq,
                                            "evt_proc_id" to procId,
                                            "session_id" to _state.value.sessionId,
                                            "turn_id" to _state.value.turnSeq,
                                            "ms" to pdt,
                                            "tools_n" to tools.size
                                        )
                                        return@launch
                                    }

                                    Log.i(tag, "POLICY#$reqSeq DONE in ${pdt}ms tools=${tools.size} ${toolsSummary(tools)}")

                                    ConversationTelemetry.emitKv(
                                        "LLM_CALLPOLICY_END_OK",
                                        "policy_req_seq" to reqSeq,
                                        "evt_proc_id" to procId,
                                        "eff_index" to idx,
                                        "session_id" to _state.value.sessionId,
                                        "turn_id" to _state.value.turnSeq,
                                        "ms" to pdt,
                                        "tools_n" to tools.size,
                                        "chosen_grid_hash" to (chosenGridHash ?: "")
                                    )

                                    emitEffectDoneOk(pdt, tools)

                                    val meaningful = tools.any { it !is ToolCall.Noop }
                                    if (!meaningful) {
                                        Log.e(tag, "POLICY#$reqSeq EMPTY_TOOLS (scheduling silent retry)")
                                        scheduleRetry(reason = "empty_tools", errMsg = null)
                                        return@launch
                                    }

                                    val hasReply = tools.any { it is ToolCall.Reply }
                                    if (!hasReply) {
                                        ConversationTelemetry.emitKv(
                                            "POLICY_MISSING_REPLY",
                                            "policy_req_seq" to reqSeq,
                                            "evt_proc_id" to procId,
                                            "session_id" to _state.value.sessionId,
                                            "turn_id" to _state.value.turnSeq,
                                            "attempt" to 0,
                                            "tools_n" to tools.size,
                                            "tools" to toolsSummary(tools)
                                        )
                                        scheduleRetry(reason = "missing_reply", errMsg = null)
                                        return@launch
                                    }

                                    effects.applyTools(tools) { followUpEvt ->
                                        Log.i(tag, "POLICY#$reqSeq APPLYTOOLS_EMIT -> ${evtSummary(followUpEvt)}")
                                        dispatch(followUpEvt)
                                    }
                                } catch (t: Throwable) {
                                    val pdt = SystemClock.elapsedRealtime() - p0
                                    Log.e(tag, "POLICY#$reqSeq ERROR in ${pdt}ms (NO local speech; scheduling retry)", t)

                                    if (latestPolicyIssuedSeq.get() != reqSeq) {
                                        Log.w(tag, "POLICY#$reqSeq STALE_ERR (dropping) ms=$pdt")
                                        ConversationTelemetry.emitKv(
                                            "LLM_CALLPOLICY_STALE_DROP_ERR",
                                            "policy_req_seq" to reqSeq,
                                            "evt_proc_id" to procId,
                                            "session_id" to _state.value.sessionId,
                                            "turn_id" to _state.value.turnSeq,
                                            "ms" to pdt
                                        )
                                        return@launch
                                    }

                                    ConversationTelemetry.emitKv(
                                        "LLM_CALLPOLICY_END_ERR",
                                        "policy_req_seq" to reqSeq,
                                        "evt_proc_id" to procId,
                                        "eff_index" to idx,
                                        "session_id" to _state.value.sessionId,
                                        "turn_id" to _state.value.turnSeq,
                                        "ms" to pdt,
                                        "err" to (t.message ?: t.toString()),
                                        "chosen_grid_hash" to (chosenGridHash ?: "")
                                    )

                                    emitEffectDoneErr(pdt, t.message ?: t.toString())
                                    scheduleRetry(reason = "exception", errMsg = t.message ?: t.toString())
                                }
                            }
                        }

                        else -> {
                            val effName = eff::class.java.simpleName
                            Log.i(tag, "EVT#$procId RUN_EFFECT[$idx] $effName ${effectDetails(eff)}")

                            ConversationTelemetry.emitKv(
                                "EFFECT_RUN",
                                "effect" to effName,
                                "evt_proc_id" to procId,
                                "eff_index" to idx,
                                "session_id" to _state.value.sessionId,
                                "turn_id" to _state.value.turnSeq
                            )

                            try {
                                effects.run(eff) { followUpEvt ->
                                    Log.i(tag, "EVT#$procId EFFECT_EMIT from $effName -> ${evtSummary(followUpEvt)}")
                                    dispatch(followUpEvt)
                                }
                                ConversationTelemetry.emitKv(
                                    "EFFECT_DONE",
                                    "effect" to effName,
                                    "evt_proc_id" to procId,
                                    "eff_index" to idx,
                                    "session_id" to _state.value.sessionId,
                                    "turn_id" to _state.value.turnSeq
                                )
                            } catch (t: Throwable) {
                                Log.e(tag, "EFFECT_CRASH EVT#$procId eff[$idx] $effName", t)
                                ConversationTelemetry.emitKv(
                                    "EFFECT_DONE_ERR",
                                    "effect" to effName,
                                    "evt_proc_id" to procId,
                                    "eff_index" to idx,
                                    "session_id" to _state.value.sessionId,
                                    "turn_id" to _state.value.turnSeq,
                                    "err" to (t.message ?: t.toString()).take(220)
                                )
                            }
                        }
                    }
                }
            }

            Log.w(tag, "EVENT_LOOP_ENDED (channel closed?)")
        }
    }

    fun dispatch(e: Evt) {
        val id = dispatchSeq.incrementAndGet()
        val res = evtCh.trySend(e)

        if (res.isSuccess) {
            Log.i(tag, "DISPATCH#$id OK ${evtSummary(e)}")
        } else {
            Log.w(tag, "DISPATCH#$id FAILED ${evtSummary(e)} reason=${res.exceptionOrNull()?.message ?: "unknown"}")
        }
    }

    // -------------------- Focus derivation --------------------

    private fun focusFromPending(p: Pending?): Int? {
        return when (p) {
            null -> null
            is Pending.ConfirmEdit -> p.cellIndex.coerceIn(0, 80)
            is Pending.AskCellValue -> p.cellIndex.coerceIn(0, 80)

            // If we have row+col we can focus a concrete cell
            is Pending.ConfirmInterpretation -> {
                val r = p.row
                val c = p.col
                if (r != null && c != null && r in 1..9 && c in 1..9) ((r - 1) * 9 + (c - 1)) else null
            }

            // Clarification might include both hints; if so, focus it. Otherwise don't.
            is Pending.AskClarification -> {
                val r = p.rowHint
                val c = p.colHint
                if (r != null && c != null && r in 1..9 && c in 1..9) ((r - 1) * 9 + (c - 1)) else null
            }

            // We don't know which cell until user taps
            is Pending.WaitForTap -> null

            Pending.ConfirmValidate -> null
            is Pending.ConfirmRetake -> null
        }
    }

    // -------------------- Gate-2 helper: stable grid hash --------------------

    private fun computeGridHashSafe(g: LLMGridState?): String? {
        if (g == null) return null
        return runCatching {
            val digits = g.correctedGrid.joinToString(separator = "") { d -> d.coerceIn(0, 9).toString() }

            val givenBits = g.truthIsGiven.joinToString(separator = "") { if (it) "1" else "0" }
            val solBits = g.truthIsSolution.joinToString(separator = "") { if (it) "1" else "0" }

            val cand = g.candidateMask81.joinToString(separator = ",") { (it and 0x1FF).toString() }

            val deduced = g.deducedSolutionGrid?.joinToString(separator = "") { d -> d.coerceIn(0, 9).toString() } ?: ""

            val key = buildString {
                append("digits="); append(digits)
                append("|given="); append(givenBits)
                append("|sol="); append(solBits)
                append("|cand="); append(cand)
                append("|deduced="); append(deduced)
                append("|deducedCount="); append(g.deducedSolutionCountCapped)
                append("|mm="); append(g.mismatchCells.size)
                append("|solv="); append(g.solvability)
                append("|struct="); append(g.isStructurallyValid)
                append("|unr="); append(g.unresolvedCells.size)
                append("|chg="); append(g.changedCells.size)
                append("|conf="); append(g.conflictCells.size)
                append("|man="); append(g.manuallyCorrectedCells.size)
                append("|edits="); append(g.manualEdits.size)
            }

            ConversationTelemetry.sha256Hex(key)
        }.getOrNull()
    }

    // -------------------- Summaries (high signal, low noise) --------------------

    private fun summarizeState(s: SudoState): String {
        val pending = summarizePending(s.pending)
        val grid = if (s.grid != null) "yes" else "no"
        val focus = s.focusCellIndex?.toString() ?: "none"
        val lastU = s.lastUserText?.replace("\n", " ")?.take(40) ?: ""
        val lastA = s.lastAssistantText?.replace("\n", " ")?.take(40) ?: ""
        return "mode=${s.mode} turnSeq=${s.turnSeq} pending=$pending focus=$focus grid=$grid lastU='${lastU}' lastA='${lastA}'"
    }

    private fun summarizePending(p: Pending?): String {
        return when (p) {
            null -> "none"
            is Pending.ConfirmEdit -> "confirm_edit idx=${p.cellIndex} digit=${p.proposedDigit} src=${p.source}"
            is Pending.AskCellValue -> "ask_cell_value idx=${p.cellIndex}"
            is Pending.ConfirmRetake -> "confirm_retake strength=${p.strength}"
            Pending.ConfirmValidate -> "confirm_validate"
            is Pending.ConfirmInterpretation -> {
                val r = p.row?.toString() ?: "?"
                val c = p.col?.toString() ?: "?"
                val d = p.digit?.toString() ?: "?"
                "confirm_interpretation r=$r c=$c d=$d conf=${p.confidence}"
            }
            is Pending.AskClarification -> {
                val r = p.rowHint?.toString() ?: "?"
                val c = p.colHint?.toString() ?: "?"
                val d = p.digitHint?.toString() ?: "?"
                "ask_clarification kind=${p.kind} hints(r=$r c=$c d=$d)"
            }
            is Pending.WaitForTap -> {
                val d = p.digitHint?.toString() ?: "?"
                "wait_for_tap digitHint=$d conf=${p.confidence}"
            }
        }
    }

    private fun evtSummary(e: Evt): String {
        return when (e) {
            is Evt.AsrFinal -> "AsrFinal(len=${e.text.trim().length} conf=${e.confidence})"
            is Evt.AsrError -> "AsrError(code=${e.code} name=${e.name})"
            is Evt.GridCaptured -> "GridCaptured(nonNullGrid=true)"
            is Evt.GridSnapshotUpdated -> "GridSnapshotUpdated(nonNullGrid=true)"
            is Evt.PolicyTools -> "PolicyTools(n=${e.tools.size})"
            is Evt.TtsStarted -> "TtsStarted(reason=${e.reason})"
            is Evt.AppStarted -> "AppStarted(epochMs=${e.epochMs})"
            Evt.CameraActive -> "CameraActive"
            Evt.GridCleared -> "GridCleared"
            Evt.TtsFinished -> "TtsFinished"
            is Evt.CellTapped -> "CellTapped(idx=${e.cellIndex})"
            is Evt.DigitPicked -> "DigitPicked(idx=${e.cellIndex} digit=${e.digit})"
            Evt.CancelTts -> "CancelTts"

            is Evt.PolicyContinuationReply -> "PolicyContinuationReply(len=${e.text.trim().length})"
            Evt.PolicyContinuationFailed -> "PolicyContinuationFailed"
        }
    }

    private fun effectsSummary(effs: List<Eff>): String {
        return effs.joinToString(prefix = "[", postfix = "]") { it::class.java.simpleName }
    }

    private fun effectDetails(e: Eff): String {
        return when (e) {
            is Eff.Speak ->
                "len=${e.text.length} listenAfter=${e.listenAfter} prev='${e.text.replace("\n", " ").take(50)}'"

            is Eff.UpdateUiMessage ->
                "len=${e.text.length} prev='${e.text.replace("\n", " ").take(50)}'"

            is Eff.SpeakAndShow ->
                "len=${e.text.length} listenAfter=${e.listenAfter} prev='${e.text.replace("\n", " ").take(50)}'"

            is Eff.SetFocusCell ->
                "idx=${e.cellIndex?.toString() ?: "null"} reason=${e.reason ?: ""}"

            is Eff.StopAsr -> "reason=${e.reason}"
            is Eff.RequestListen -> "reason=${e.reason}"

            is Eff.ApplyCellEdit -> "idx=${e.cellIndex} digit=${e.digit} source=${e.source}"

            is Eff.ReclassifyCell -> "idx=${e.cellIndex} kind=${e.kind} source=${e.source}"
            is Eff.SetCandidates -> "idx=${e.cellIndex} mask=${e.mask} source=${e.source}"
            is Eff.ToggleCandidate -> "idx=${e.cellIndex} digit=${e.digit} source=${e.source}"
            is Eff.ApplyCellClassify -> "idx=${e.cellIndex} class=${e.cellClass} source=${e.source}"
            is Eff.ApplyCellCandidatesMask -> "idx=${e.cellIndex} mask=${e.candidateMask} source=${e.source}"

            is Eff.ConfirmCellValue ->
                "idx=${e.cellIndex} digit=${e.digit} source=${e.source} changed=${e.changed}"

            is Eff.CallPolicy -> "userLen=${e.userText.length} grid=${e.gridContext != null}"

            is Eff.CallPolicyContinuationTick2 ->
                "toolResults=${e.toolResults.size} mode=${e.mode.name} reason=${e.reason} turnId=${e.turnId}"
        }
    }

    private fun toolsSummary(tools: List<ToolCall>): String {
        if (tools.isEmpty()) return "(empty)"
        return tools.joinToString(prefix = "[", postfix = "]") { t ->
            when (t) {
                is ToolCall.Reply -> "Reply(len=${t.text.length})"
                is ToolCall.AskConfirmCell -> "AskConfirmCell(idx=${t.cellIndex})"
                is ToolCall.ProposeEdit -> "ProposeEdit(idx=${t.cellIndex} d=${t.digit} conf=${t.confidence})"
                is ToolCall.ApplyUserEdit -> "ApplyUserEdit(idx=${t.cellIndex} d=${t.digit} src=${t.source})"
                is ToolCall.RecommendRetake -> "RecommendRetake(str=${t.strength})"
                ToolCall.RecommendValidate -> "RecommendValidate"
                is ToolCall.ConfirmInterpretation -> {
                    val r = t.row?.toString() ?: "?"
                    val c = t.col?.toString() ?: "?"
                    val d = t.digit?.toString() ?: "?"
                    "ConfirmInterpretation(r=$r c=$c d=$d conf=${t.confidence})"
                }
                is ToolCall.AskClarifyingQuestion -> "AskClarifyingQuestion(kind=${t.kind})"
                is ToolCall.SwitchToTap -> "SwitchToTap"
                ToolCall.Noop -> "Noop"
                else -> "Tool(${t::class.java.simpleName})"
            }
        }
    }
}

interface EffectRunner {
    fun run(effect: Eff, emit: (Evt) -> Unit)
    fun applyTools(tools: List<ToolCall>, emit: (Evt) -> Unit)
}