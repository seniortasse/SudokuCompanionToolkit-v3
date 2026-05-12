package com.contextionary.sudoku.conductor

import android.os.SystemClock
import android.util.Log
import com.contextionary.sudoku.telemetry.ConversationTelemetry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong
import com.contextionary.sudoku.conductor.policy.*

/**
 * Option A (canonical):
 * - SudoStore MUST NOT call policy at all.
 * - Store only: reducer + effects runner.
 * - Policy calls happen only inside EffectRunner (MainActivity), which emits Evt.PolicyTools back into the store.
 */
class SudoStore(
    initial: SudoState,
    private val conductor: SudoConductor,
    private val effects: EffectRunner,
    private val scope: CoroutineScope
) {
    private val _state = MutableStateFlow(initial)
    val state: StateFlow<SudoState> = _state

    private val evtCh = Channel<Evt>(capacity = Channel.UNLIMITED)
    private val tag = "SudoStore"

    // Sequence numbers to correlate “dispatch → process → effects”
    private val dispatchSeq = AtomicLong(0)
    private val processSeq = AtomicLong(0)

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
                    SudoConductor.Next(s0, emptyList())
                }

                // ----------------------------
                // ✅ derive UI focus cell from pending and store it in state
                // ----------------------------
                val desiredFocus: Int? = focusFromActiveOwnerOrQueues(next.state)

                val stateWithFocus =
                    if (next.state.focusCellIndex != desiredFocus) next.state.copy(focusCellIndex = desiredFocus)
                    else next.state

                val traced = applyDecisionTraceEvent(e, stateWithFocus)
                _state.value = traced

                // ----------------------------
                // ✅ auto-inject SetFocusCell effect (unless conductor already emitted one)
                // Put it FIRST so highlight is visible before Speak/Ask.
                // ----------------------------
                val effs: MutableList<Eff> = next.effects.toMutableList()
                val hasExplicitFocusEff = effs.any { it is Eff.SetFocusCell }
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

                // ----------------------------
                // ✅ Option A: Store never runs policy.
                // Everything routes through EffectRunner.run(...).
                // ----------------------------
                effs.forEachIndexed { idx, rawEff ->
                    val eff = maybeInjectPhaseIntoPolicyEffects(rawEff, _state.value)

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

    private fun deriveFocus(s: SudoState): Int? {
        // 1) Pending wins
        focusFromPending(s.pending)?.let { return it }

        // 2) Fallback: if the next app agenda item is a cell-targeted CTA, focus it.
        val a0 = runCatching { s.appAgendaQueue.firstOrNull() }.getOrNull() ?: return null
        return when (a0) {
            is AppAgendaItem.AskConfirmCell -> a0.cellIndex.coerceIn(0, 80)
            is AppAgendaItem.ProposeEdit -> a0.cellIndex.coerceIn(0, 80)
            else -> null
        }
    }

    private fun focusFromPending(p: Pending?): Int? {
        return when (p) {
            null -> null

            // Existing
            is Pending.ConfirmEdit -> p.cellIndex.coerceIn(0, 80)
            is Pending.AskCellValue -> p.cellIndex.coerceIn(0, 80)

            // Gate 4 / repair
            is Pending.ConfirmInterpretation -> {
                val r = p.row
                val c = p.col
                if (r != null && c != null && r in 1..9 && c in 1..9) ((r - 1) * 9 + (c - 1)) else null
            }

            is Pending.AskClarification -> {
                val r = p.rowHint
                val c = p.colHint
                if (r != null && c != null && r in 1..9 && c in 1..9) ((r - 1) * 9 + (c - 1)) else null
            }

            is Pending.WaitForTap -> null
            is Pending.ConfirmValidate -> null
            is Pending.ConfirmRetake -> null

            // SOLVING: CTA pendings — focus is not a specific cell
            is Pending.SolveIntroAction -> null
            is Pending.AfterResolution -> null
            is Pending.ApplyHintNow -> null

            else -> null // ✅ stability: do not break compilation if new pending is added
        }
    }

    private fun userAgendaFocusCellIndex(s: SudoState): Int? {
        val head = s.userAgendaQueue.firstOrNull() ?: return null

        fun parseCell(cellRef: String?): Int? {
            val t = cellRef?.trim()?.lowercase() ?: return null
            val m = Regex("""r\s*([1-9])\s*c\s*([1-9])""").find(t) ?: return null
            val r = m.groupValues.getOrNull(1)?.toIntOrNull() ?: return null
            val c = m.groupValues.getOrNull(2)?.toIntOrNull() ?: return null
            return ((r - 1) * 9 + (c - 1)).takeIf { it in 0..80 }
        }

        return when (head) {
            is UserAgendaItem.Clarification -> null
            is UserAgendaItem.ProofChallenge -> parseCell(head.cellRef)
            is UserAgendaItem.CandidateStateQuery -> parseCell(head.cellRefs.firstOrNull())
            is UserAgendaItem.TargetCellQuery -> parseCell(head.targetCellRef)
            is UserAgendaItem.NeighborCellQuery -> parseCell(head.neighborCellRef ?: head.anchorCellRef)
            is UserAgendaItem.UserReasoningCheck -> parseCell(head.cellRef)
            is UserAgendaItem.AlternativeTechniqueQuery -> parseCell(head.cellRef)
            is UserAgendaItem.RouteComparisonQuery -> null
            is UserAgendaItem.RouteControl -> null
            is UserAgendaItem.OverlayControl -> parseCell(head.focusCellRef)
            is UserAgendaItem.GeneralQuestion -> null
        }
    }

    private fun focusFromActiveOwnerOrQueues(s: SudoState): Int? {
        val authorityOwner = s.activeTurnAuthorityDecisionV1.owner

        if (
            authorityOwner == TurnAuthorityOwnerV1.USER_DETOUR_OWNER ||
            authorityOwner == TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER ||
            authorityOwner == TurnAuthorityOwnerV1.REPAIR_OWNER
        ) {
            userAgendaFocusCellIndex(s)?.let { return it }
        }

        // Pending is projection-only, but still a useful focus bridge for UI compatibility.
        focusFromPending(s.pending)?.let { return it }

        if (
            authorityOwner == TurnAuthorityOwnerV1.APP_ROUTE_OWNER ||
            s.activeRouteBoundaryDecisionV1.status == RouteBoundaryStatusV1.RELEASED_TO_NEXT_STEP
        ) {
            val first = s.appAgendaQueue.firstOrNull()
            return when (first) {
                is AppAgendaItem.AskConfirmCell -> first.cellIndex.coerceIn(0, 80)
                is AppAgendaItem.ProposeEdit -> first.cellIndex.coerceIn(0, 80)
                else -> s.focusCellIndex
            }
        }

        return s.focusCellIndex ?: userAgendaFocusCellIndex(s)
    }

    // -------------------- Phase-in-header injection --------------------

    /**
     * Keeps previous behavior: ensure policy-bound effects carry `phase=...` in stateHeader.
     * This does NOT call policy; it only mutates the effect payload.
     */
    private fun maybeInjectPhaseIntoPolicyEffects(e: Eff, s: SudoState): Eff {
        return when (e) {

            is Eff.CallPolicy -> {
                val hdr = injectPhaseIntoStateHeader(e.stateHeader, s)
                e.copy(stateHeader = hdr)
            }

            // ✅ Tick1 Meaning Extract has stateHeader too
            is Eff.CallMeaningExtractV1 -> {
                val hdr = injectPhaseIntoStateHeader(e.stateHeader, s)
                e.copy(stateHeader = hdr)
            }

            is Eff.CallPolicyContinuationTick2 -> {
                val hdr = injectPhaseIntoStateHeader(e.stateHeader, s)
                e.copy(stateHeader = hdr)
            }

            // ✅ v1 Tick2 has no stateHeader, nothing to inject
            is Eff.CallReplyGenerateV1 -> e

            else -> e
        }
    }

    private fun injectPhaseIntoStateHeader(stateHeader: String, s: SudoState): String {
        val phaseVal = runCatching { s.phase }.getOrNull() ?: return stateHeader
        val phaseStr = phaseVal.toString()

        if (stateHeader.contains("phase=")) return stateHeader

        val modeIdx = stateHeader.indexOf("mode=")
        if (modeIdx >= 0) {
            val endTok = stateHeader.indexOfAny(charArrayOf(' ', '\n'), startIndex = modeIdx)
                .let { if (it < 0) stateHeader.length else it }
            val head = stateHeader.substring(0, endTok)
            val tail = stateHeader.substring(endTok)
            return "$head phase=$phaseStr$tail"
        }

        return "phase=$phaseStr $stateHeader"
    }

    // -------------------- Summaries --------------------

    private fun summarizeState(s: SudoState): String {
        val pending = summarizePending(s.pending)
        val grid = if (s.grid != null) "yes" else "no"
        val focus = s.focusCellIndex?.toString() ?: "none"
        val phasePart = runCatching { " phase=${s.phase}" }.getOrNull() ?: ""

        val lastU = s.lastUserText?.replace("\n", " ")?.take(40) ?: ""
        val lastA = s.lastAssistantText?.replace("\n", " ")?.take(40) ?: ""

        return "mode=${s.mode}$phasePart turnSeq=${s.turnSeq} pending=$pending focus=$focus grid=$grid lastU='${lastU}' lastA='${lastA}'"
    }

    private fun summarizePending(p: Pending?): String {
        return when (p) {
            null -> "none"

            is Pending.ConfirmEdit -> "confirm_edit idx=${p.cellIndex} digit=${p.proposedDigit} src=${p.source}"
            is Pending.AskCellValue -> "ask_cell_value idx=${p.cellIndex}"
            is Pending.ConfirmRetake -> "confirm_retake strength=${p.strength}"
            is Pending.ConfirmValidate -> "confirm_validate"

            // SOLVING (North Star CTA rail)
            is Pending.SolveIntroAction ->
                "solve_intro_action stepId=${p.stepId} atomIndex=${p.atomIndex}/${p.atomsCount} last=${p.isLastHint} options=${p.options}"
            is Pending.AfterResolution ->
                "after_resolution stepId=${p.stepId} options=${p.options}"
            is Pending.ApplyHintNow ->
                "apply_hint_now stepId=${p.stepId} options=${p.options}"

            // Repair
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

            else -> "other:${p::class.java.simpleName}" // ✅ stability
        }
    }

    private data class IntentTrace(
        val intentType: IntentTypeV1,
        val confidence: Double,
        val needsClarification: Boolean
    )

    private fun intentToTrace(i: IntentV1): IntentTrace {
        // Display-only trace for DecisionTrace fields.
        // Do not drive behavior from this.
        val needsClar = i.missing.isNotEmpty()
        return IntentTrace(
            intentType = i.type,
            confidence = i.confidence,
            needsClarification = needsClar
        )
    }


    private fun applyDecisionTraceEvent(e: Evt, s: SudoState): SudoState {
        val now = SystemClock.elapsedRealtime()

        return when (e) {

            // ✅ NEW MODEL: Tick1 emits IntentEnvelopeV1
            is Evt.IntentEnvelopeReceived -> {
                val env = e.env
                val top = env.intents.maxByOrNull { it.confidence }

                if (top != null) {
                    val trace = intentToTrace(top)
                    s.copy(
                        lastMeaningIntentType = trace.intentType,
                        lastMeaningConfidence = trace.confidence,
                        lastMeaningNeedsClarification = trace.needsClarification
                    )
                } else {
                    // No intents => treat as unknown/unclear
                    s.copy(
                        lastMeaningIntentType = IntentTypeV1.UNKNOWN,
                        lastMeaningConfidence = 0.0,
                        lastMeaningNeedsClarification = true
                    )
                }
            }

            is Evt.DecisionApplied -> s.copy(
                lastDecisionKind = e.decisionKind,
                lastFactBundleTypes = e.factBundleTypes,
                lastDecisionEndMs = now
            )

            is Evt.ReplyReceived -> s.copy(
                lastAssistantText = e.text,
                lastReplyLen = e.text.length,
                lastTick2EndMs = now
            )

            // Back-compat: if you still emit this somewhere
            is Evt.PolicyContinuationReply -> s.copy(
                lastAssistantText = e.text,
                lastReplyLen = e.text.length,
                lastTick2EndMs = now
            )

            else -> s
        }
    }

    private fun evtSummary(e: Evt): String {
        return when (e) {
            is Evt.AsrFinal -> "AsrFinal(len=${e.text.trim().length} conf=${e.confidence})"
            is Evt.AsrError -> "AsrError(code=${e.code} name=${e.name})"

            is Evt.GridCaptured -> "GridCaptured(nonNullGrid=true)"
            is Evt.GridSnapshotUpdated -> "GridSnapshotUpdated(nonNullGrid=true)"
            Evt.GridCleared -> "GridCleared"

            is Evt.PolicyTools -> "PolicyTools(n=${e.tools.size})"

            is Evt.TtsStarted -> "TtsStarted(reason=${e.reason})"
            Evt.TtsFinished -> "TtsFinished"
            Evt.CameraActive -> "CameraActive"
            is Evt.AppStarted -> "AppStarted(epochMs=${e.epochMs})"

            is Evt.CellTapped -> "CellTapped(idx=${e.cellIndex})"
            is Evt.DigitPicked -> "DigitPicked(idx=${e.cellIndex} digit=${e.digit})"
            Evt.CancelTts -> "CancelTts"

            is Evt.PolicyContinuationReply -> "PolicyContinuationReply(len=${e.text.trim().length})"
            is Evt.PolicyContinuationFailed -> "PolicyContinuationFailed(code=${e.errorCode} msg=${e.errorMsg})"

            is Evt.DetourSolverQuerySucceeded ->
                "DetourSolverQuerySucceeded(id=${e.queryId} op=${e.op} len=${e.resultJson.length})"

            is Evt.DetourSolverQueryFailed ->
                "DetourSolverQueryFailed(id=${e.queryId} op=${e.op} err=${e.error.take(80)})"

            // ✅ NEW MODEL: Intent envelope event (Tick1 result)
            is Evt.IntentEnvelopeReceived -> {
                val env = e.env
                val top = env.intents.maxByOrNull { it.confidence }
                val topStr = if (top == null) "none" else "${top.type}:${"%.2f".format(top.confidence)}"
                "IntentEnvelopeReceived(intents=${env.intents.size} top=$topStr)"
            }

            is Evt.DecisionApplied -> "DecisionApplied(kind=${e.decisionKind.name} facts=${e.factBundleTypes.size})"
            is Evt.ReplyReceived -> "ReplyReceived(len=${e.text.trim().length} src=${e.source})"

            // ✅ UPDATED: ToolExecuted signature changed in Events.kt
            is Evt.ToolExecuted -> {
                val okStr = if (e.ok) "ok" else "err"
                "ToolExecuted(name=${e.toolName} $okStr tcid=${e.toolCallId} rid=${e.toolResultId})"
            }

            else -> e::class.java.simpleName ?: "Evt"
        }
    }

    private fun effectsSummary(effs: List<Eff>): String {
        return effs.joinToString(prefix = "[", postfix = "]") { it::class.java.simpleName }
    }

    private fun effectDetails(e: Eff): String {
        fun preview(s: String): String = s.replace("\n", " ").take(50)

        return when (e) {

            // ----------------------------
            // UI + voice
            // ----------------------------
            is Eff.Speak ->
                "len=${e.text.length} listenAfter=${e.listenAfter} prev='${preview(e.text)}'"

            is Eff.UpdateUiMessage ->
                "len=${e.text.length} prev='${preview(e.text)}'"

            is Eff.SpeakAndShow ->
                "len=${e.text.length} listenAfter=${e.listenAfter} prev='${preview(e.text)}'"

            is Eff.SetFocusCell ->
                "idx=${e.cellIndex?.toString() ?: "null"} reason=${e.reason.orEmpty()}"

            // ----------------------------
            // SOLVING: compute + overlay (pure UI + compute)
            // ----------------------------
            is Eff.ComputeSolveStep ->
                "gridHash12=${e.gridHash12} gridLen=${e.grid81.length} force=${e.force} reason=${e.reason}"

            is Eff.RunDetourSolverQuery ->
                "id=${e.queryId} op=${e.op} payloadLen=${e.payloadJson.length} reason=${e.reason}"

            is Eff.RenderSolveOverlay ->
                "style=${e.style} stepLen=${e.stepJson.length} reason=${e.reason}"

            is Eff.ClearSolveOverlay ->
                "reason=${e.reason}"

            // ----------------------------
            // Audio machine commands
            // ----------------------------
            is Eff.StopAsr ->
                "reason=${e.reason}"

            is Eff.RequestListen ->
                "reason=${e.reason}"

            // ----------------------------
            // Operational grid effects (toolCallId required)
            // ----------------------------
            is Eff.ApplyCellEdit ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} digit=${e.digit} source=${e.source}"

            is Eff.ReclassifyCell ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} kind=${e.kind} source=${e.source}"

            is Eff.SetCandidates ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} mask=${e.mask} source=${e.source}"

            is Eff.ToggleCandidate ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} digit=${e.digit} source=${e.source}"

            is Eff.ApplyCellClassify ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} class=${e.cellClass} source=${e.source}"

            is Eff.ApplyCellCandidatesMask ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} mask=${e.candidateMask} source=${e.source}"

            is Eff.ConfirmCellValue ->
                "tcid=${e.toolCallId} idx=${e.cellIndex} digit=${e.digit} source=${e.source} changed=${e.changed}"

            is Eff.FinalizeValidationPresentation ->
                "tcid=${e.toolCallId} reason=${e.reason}"

            // ----------------------------
            // Policy calls
            // ----------------------------
            is Eff.CallPolicy ->
                "userLen=${e.userText.length} grid=${e.gridContext != null} turnId=${e.turnId ?: -1} reason=${e.reason.orEmpty()}"

            is Eff.CallMeaningExtractV1 ->
                "turnId=${e.ctx.turnId} reason=${e.ctx.reason} userLen=${e.userText.length}"

            is Eff.CallReplyGenerateV1 ->
                "turnId=${e.turnId} reason=${e.reason} replyReqChars=${e.replyRequest.toJson().toString().length}"

            is Eff.CallPolicyContinuationTick2 ->
                "toolResults=${e.toolResults.size} mode=${e.mode.name} reason=${e.reason} turnId=${e.turnId}"

            // ✅ IMPORTANT: avoid non-exhaustive when() if Eff grows
            else -> ""
        }
    }
}

interface EffectRunner {
    fun run(effect: Eff, emit: (Evt) -> Unit)
    fun applyTools(tools: List<ToolCall>, emit: (Evt) -> Unit)
}