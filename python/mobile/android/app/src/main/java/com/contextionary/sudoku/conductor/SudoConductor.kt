package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.logic.LLMGridState
import com.contextionary.sudoku.telemetry.ConversationTelemetry

class SudoConductor {

    data class Next(val state: SudoState, val effects: List<Eff>)

    companion object {
        // Used only for bounded replans on protocol violations (missing reply / noop alone)
        private const val MAX_MISSING_REPLY_REPLANS = 2

        // Deterministic “never silent” heartbeat (A)
        private const val SURVIVAL_REPLY = "Sorry — I didn’t catch that clearly; could you please repeat?"
    }

    // ------------------------------------------------------------------------
    // Patch 0: Telemetry helpers (POLICY_CALL + TOOLS_APPLIED)
    // ------------------------------------------------------------------------

    private data class RepairInfo(val repairMode: Boolean, val repairReason: String?)

    private fun parseRepairInfo(stateHeader: String): RepairInfo {
        // We keep this for telemetry continuity; B stops producing repair headers locally.
        val repairMode = Regex("(?m)^repair_mode\\s*=\\s*true\\s*$").containsMatchIn(stateHeader)
        val repairReason = Regex("(?m)^repair_reason\\s*=\\s*([^\\n\\r]+)\\s*$")
            .find(stateHeader)
            ?.groupValues
            ?.getOrNull(1)
            ?.trim()
        return RepairInfo(repairMode = repairMode, repairReason = repairReason)
    }

    private fun eventTypeFromUserText(userText: String): String {
        val m = Regex("^\\[EVENT\\]\\s+([a-zA-Z0-9_\\-]+)").find(userText.trim())
        return m?.groupValues?.getOrNull(1) ?: "user_turn"
    }

    private fun pendingType(p: Pending?): String? = when (p) {
        null -> null
        is Pending.ConfirmEdit -> "confirm_edit"
        is Pending.AskCellValue -> "ask_cell_value"
        is Pending.ConfirmRetake -> "confirm_retake"
        Pending.ConfirmValidate -> "confirm_validate"
        is Pending.ConfirmInterpretation -> "confirm_interpretation"
        is Pending.AskClarification -> "ask_clarification"
        is Pending.WaitForTap -> "wait_for_tap"
    }

    private fun pendingIdx(p: Pending?): Int? = when (p) {
        is Pending.ConfirmEdit -> p.cellIndex
        is Pending.AskCellValue -> p.cellIndex
        else -> null
    }

    private fun pendingRow(p: Pending?): Int? = when (p) {
        is Pending.ConfirmInterpretation -> p.row
        is Pending.AskClarification -> p.rowHint
        else -> null
    }

    private fun pendingCol(p: Pending?): Int? = when (p) {
        is Pending.ConfirmInterpretation -> p.col
        is Pending.AskClarification -> p.colHint
        else -> null
    }

    private fun pendingDigit(p: Pending?): Int? = when (p) {
        is Pending.ConfirmEdit -> p.proposedDigit
        is Pending.ConfirmInterpretation -> p.digit
        is Pending.AskClarification -> p.digitHint
        is Pending.WaitForTap -> p.digitHint
        else -> null
    }

    private fun emitPolicyCallTrace(
        traceState: SudoState,
        userText: String,
        stateHeader: String,
        reason: String,
        turnId: Long
    ) {
        val repair = parseRepairInfo(stateHeader)
        val headerSha12 = try {
            ConversationTelemetry.sha256Hex(stateHeader).take(12)
        } catch (_: Throwable) {
            null
        }

        ConversationTelemetry.emitPolicyTrace(
            tag = "POLICY_CALL",
            data = mapOf(
                "event_type" to eventTypeFromUserText(userText),
                "reason" to reason,
                "mode" to traceState.mode.name,
                "turn_seq" to traceState.turnSeq,
                "turn_id" to turnId,

                "pending_type" to pendingType(traceState.pending),
                "pending_idx" to pendingIdx(traceState.pending),
                "pending_row" to pendingRow(traceState.pending),
                "pending_col" to pendingCol(traceState.pending),
                "pending_digit" to pendingDigit(traceState.pending),

                "repair_mode" to repair.repairMode,
                "repair_reason" to repair.repairReason,

                "state_header_sha12" to headerSha12
            )
        )
    }


    private fun rcFromIdx(idx: Int): Pair<Int, Int> {
        val r = (idx / 9) + 1
        val c = (idx % 9) + 1
        return r to c
    }

    private fun buildToolResultsForTick2(
        sanitized: List<ToolCall>,
        synthesizedConfirm: LastCellConfirmation?, // optional signal
    ): List<String> {
        val out = mutableListOf<String>()

        fun add(s: String) { if (s.isNotBlank()) out += s }

        sanitized.forEach { t ->
            when (t) {
                is ToolCall.ConfirmCellValue -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("confirm_cell_value r${r}c${c} digit=${t.digit} source=${t.source}")
                }
                is ToolCall.ConfirmCellValueRC -> {
                    add("confirm_cell_value r${t.row}c${t.col} digit=${t.digit} source=${t.source}")
                }

                is ToolCall.ApplyUserEdit -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("apply_user_edit r${r}c${c} digit=${t.digit} source=${t.source}")
                }
                is ToolCall.ApplyUserEditRC -> {
                    add("apply_user_edit r${t.row}c${t.col} digit=${t.digit} source=${t.source}")
                }

                is ToolCall.ReclassifyCell -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("reclassify_cell r${r}c${c} kind=${t.kind} source=${t.source}")
                }
                is ToolCall.ReclassifyCells -> {
                    t.cells.forEach { c2 ->
                        val (r, c) = rcFromIdx(c2.cellIndex)
                        val src = c2.source.ifBlank { t.source }
                        add("reclassify_cell r${r}c${c} kind=${c2.kind} source=${src}")
                    }
                }
                is ToolCall.ReclassifyCellRC -> {
                    add("reclassify_cell r${t.row}c${t.col} kind=${t.kind} source=${t.source}")
                }

                is ToolCall.ApplyUserClassify -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("apply_user_classify r${r}c${c} class=${t.cellClass.name} source=${t.source}")
                }
                is ToolCall.ApplyUserClassifyRC -> {
                    add("apply_user_classify r${t.row}c${t.col} class=${t.cellClass.name} source=${t.source}")
                }

                is ToolCall.SetCandidates -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("set_candidates r${r}c${c} mask=${t.mask} source=${t.source}")
                }
                is ToolCall.ClearCandidates -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("clear_candidates r${r}c${c} source=${t.source}")
                }
                is ToolCall.ToggleCandidate -> {
                    val (r, c) = rcFromIdx(t.cellIndex)
                    add("toggle_candidate r${r}c${c} digit=${t.digit} source=${t.source}")
                }

                else -> Unit
            }
        }

        // optional: if you want a single explicit marker when we synthesized
        synthesizedConfirm?.let {
            add("synth_confirm idx=${it.cellIndex} digit=${it.digit} changed=${it.changed} source=${it.source}")
        }

        return out
    }

    private fun llmGrid(s: SudoState): LLMGridState? = s.grid?.llm as? LLMGridState

    private fun callPolicyWithTrace(
        traceState: SudoState,
        userText: String,
        stateHeader: String,
        mode: SudoMode,
        reason: String,
        turnId: Long
    ): Eff.CallPolicy {
        emitPolicyCallTrace(
            traceState = traceState,
            userText = userText,
            stateHeader = stateHeader,
            reason = reason,
            turnId = turnId
        )
        return Eff.CallPolicy(
            userText = userText,
            stateHeader = stateHeader,
            gridContext = llmGrid(traceState),
            mode = mode,
            reason = reason,
            turnId = turnId
        )
    }

    private fun toolWireName(t: ToolCall): String = when (t) {
        is ToolCall.Reply -> ToolCall.REPLY

        is ToolCall.AskConfirmCellRC -> ToolCall.ASK_CONFIRM_CELL_RC
        is ToolCall.AskConfirmCell -> ToolCall.ASK_CONFIRM_CELL
        is ToolCall.ProposeEdit -> ToolCall.PROPOSE_EDIT

        // ✅ NEW
        is ToolCall.ConfirmCellValue -> ToolCall.CONFIRM_CELL_VALUE
        is ToolCall.ConfirmCellValueRC -> ToolCall.CONFIRM_CELL_VALUE_RC

        is ToolCall.ApplyUserEdit -> ToolCall.APPLY_USER_EDIT
        is ToolCall.ApplyUserEditRC -> ToolCall.APPLY_USER_EDIT_RC

        is ToolCall.RecommendRetake -> ToolCall.RECOMMEND_RETAKE
        ToolCall.RecommendValidate -> ToolCall.RECOMMEND_VALIDATE
        is ToolCall.ConfirmInterpretation -> ToolCall.CONFIRM_INTERPRETATION
        is ToolCall.AskClarifyingQuestion -> ToolCall.ASK_CLARIFYING_QUESTION
        is ToolCall.SwitchToTap -> ToolCall.SWITCH_TO_TAP
        ToolCall.Noop -> ToolCall.NOOP

        is ToolCall.ReclassifyCell -> ToolCall.RECLASSIFY_CELL
        is ToolCall.ReclassifyCells -> ToolCall.RECLASSIFY_CELLS
        is ToolCall.ReclassifyCellRC -> ToolCall.RECLASSIFY_CELL_RC

        is ToolCall.SetCandidates -> ToolCall.SET_CANDIDATES
        is ToolCall.ClearCandidates -> ToolCall.CLEAR_CANDIDATES
        is ToolCall.ToggleCandidate -> ToolCall.TOGGLE_CANDIDATE

        is ToolCall.ApplyUserClassify -> ToolCall.APPLY_USER_CLASSIFY
        is ToolCall.ApplyUserClassifyRC -> ToolCall.APPLY_USER_CLASSIFY_RC

        else -> t.javaClass.simpleName ?: "unknown_tool"
    }

    private fun pendingSummary(p: Pending?): String = when (p) {
        null -> "none"
        is Pending.ConfirmEdit -> "confirm_edit(idx=${p.cellIndex},digit=${p.proposedDigit})"
        is Pending.AskCellValue -> "ask_cell_value(idx=${p.cellIndex})"
        is Pending.ConfirmRetake -> "confirm_retake(strength=${p.strength})"
        Pending.ConfirmValidate -> "confirm_validate"
        is Pending.ConfirmInterpretation -> "confirm_interpretation(r=${p.row},c=${p.col},d=${p.digit},conf=${p.confidence})"
        is Pending.AskClarification -> "ask_clarification(kind=${p.kind},r=${p.rowHint},c=${p.colHint},d=${p.digitHint})"
        is Pending.WaitForTap -> "wait_for_tap(digitHint=${p.digitHint},conf=${p.confidence})"
    }

    fun reduce(s: SudoState, e: Evt): Next {
        return when (e) {

            is Evt.AppStarted -> {
                Next(
                    s.copy(
                        mode = SudoMode.FREE_TALK,
                        pending = null,
                        repairAttempt = 0
                    ),
                    emptyList()
                )
            }

            is Evt.CameraActive -> {
                Next(
                    s.copy(
                        mode = SudoMode.FREE_TALK,
                        grid = null,
                        pending = null,
                        repairAttempt = 0
                    ),
                    listOf(Eff.StopAsr("camera_active"))
                )
            }

            // Policy owns first line on capture
            is Evt.GridCaptured -> {
                val s1 = s.copy(
                    mode = SudoMode.GRID_SESSION,
                    grid = e.grid,
                    pending = null,
                    repairAttempt = 0,
                    turnSeq = s.turnSeq + 1
                )

                val header = buildStateHeader(s1)
                Next(
                    s1,
                    listOf(
                        callPolicyWithTrace(
                            traceState = s1,
                            userText = "[EVENT] grid_captured",
                            stateHeader = header,
                            mode = s1.mode,
                            reason = "grid_captured_first_turn",
                            turnId = s1.turnSeq
                        )
                    )
                )
            }

            is Evt.GridCleared -> {
                val s1 = s.copy(
                    mode = SudoMode.FREE_TALK,
                    grid = null,
                    pending = null,
                    repairAttempt = 0,
                    turnSeq = s.turnSeq + 1
                )
                val header = buildStateHeader(s1)
                Next(
                    s1,
                    listOf(
                        callPolicyWithTrace(
                            traceState = s1,
                            userText = "[EVENT] grid_cleared",
                            stateHeader = header,
                            mode = s1.mode,
                            reason = "grid_cleared",
                            turnId = s1.turnSeq
                        )
                    )
                )
            }

            is Evt.GridSnapshotUpdated -> {
                Next(s.copy(grid = e.grid), emptyList())
            }


            is Evt.PolicyContinuationReply -> {
                val t = e.text.trim()
                val pendingTick2 = s.pendingTick2

                val finalText =
                    if (t.isNotBlank()) t
                    else pendingTick2?.fallbackText ?: SURVIVAL_REPLY

                val listenAfter = pendingTick2?.listenAfter ?: true

                Next(
                    s.copy(
                        lastAssistantText = finalText,
                        pendingTick2 = null,
                        repairAttempt = 0
                    ),
                    listOf(
                        Eff.UpdateUiMessage(finalText),
                        Eff.Speak(finalText, listenAfter = listenAfter)
                    )
                )
            }

            Evt.PolicyContinuationFailed -> {
                val pendingTick2 = s.pendingTick2
                val finalText = pendingTick2?.fallbackText ?: SURVIVAL_REPLY
                val listenAfter = pendingTick2?.listenAfter ?: true

                runCatching {
                    ConversationTelemetry.emitKv(
                        "TICK2_FAILED_FALLBACK_USED",
                        "turn_seq" to s.turnSeq,
                        "mode" to s.mode.name
                    )
                }

                Next(
                    s.copy(
                        lastAssistantText = finalText,
                        pendingTick2 = null
                    ),
                    listOf(
                        Eff.UpdateUiMessage(finalText),
                        Eff.Speak(finalText, listenAfter = listenAfter)
                    )
                )
            }




            is Evt.AsrError -> {
                // Deterministic: re-arm listening. No user-facing text here.
                Next(s, listOf(Eff.RequestListen("asr_error_${e.code}_${e.name}")))
            }

            is Evt.TtsStarted -> Next(s, emptyList())
            is Evt.TtsFinished -> Next(s, emptyList())

            Evt.CancelTts -> {
                Next(s, listOf(Eff.StopAsr("cancel_tts_evt")))
            }

            // Tap flows are signals; do not locally interpret intent
            is Evt.CellTapped -> {
                if (s.mode != SudoMode.GRID_SESSION) return Next(s, emptyList())

                val idx = e.cellIndex.coerceIn(0, 80)

                val s1 = s.copy(
                    pending = Pending.AskCellValue(cellIndex = idx, prompt = ""),
                    repairAttempt = 0,
                    turnSeq = s.turnSeq + 1
                )

                val header = buildStateHeader(s1)
                Next(
                    s1,
                    listOf(
                        callPolicyWithTrace(
                            traceState = s1,
                            userText = "[EVENT] cell_tapped idx=$idx",
                            stateHeader = header,
                            mode = s1.mode,
                            reason = "cell_tapped",
                            turnId = s1.turnSeq
                        )
                    )
                )
            }

            is Evt.DigitPicked -> {
                if (s.mode != SudoMode.GRID_SESSION) return Next(s, emptyList())

                val idx = e.cellIndex.coerceIn(0, 80)
                val d = e.digit.coerceIn(0, 9)

                val s1 = s.copy(
                    pending = null,
                    repairAttempt = 0,
                    turnSeq = s.turnSeq + 1
                )

                val header = buildStateHeader(s1)
                Next(
                    s1,
                    listOf(
                        Eff.ApplyCellEdit(idx, d, "digit_picked"),
                        callPolicyWithTrace(
                            traceState = s1,
                            userText = "[EVENT] digit_picked idx=$idx digit=$d",
                            stateHeader = header,
                            mode = s1.mode,
                            reason = "digit_picked",
                            turnId = s1.turnSeq
                        )
                    )
                )
            }

            // ------------------------------------------------------------
            // Main user input (B): NO local meaning/parsing; policy decides.
            // ------------------------------------------------------------
            is Evt.AsrFinal -> {
                val text = e.text.trim()
                if (text.isBlank()) {
                    return Next(s, listOf(Eff.RequestListen("asr_final_blank")))
                }

                val s1 = s.copy(
                    lastUserText = text,
                    repairAttempt = 0, // reserve repairAttempt for protocol replans only
                    turnSeq = s.turnSeq + 1
                )

                val pending = s1.pending
                if (pending != null) {
                    return handlePendingB(s1, text, pending)
                }

                val header = buildStateHeader(s1)
                Next(
                    s1,
                    listOf(
                        callPolicyWithTrace(
                            traceState = s1,
                            userText = text,
                            stateHeader = header,
                            mode = s1.mode,
                            reason = if (s1.mode == SudoMode.GRID_SESSION) "user_turn_grid_session" else "user_turn_free_talk",
                            turnId = s1.turnSeq
                        )
                    )
                )
            }

            is Evt.PolicyTools -> applyPolicyTools(s, e.tools)

            else -> Next(s, emptyList())
        }
    }

    // ------------------------------------------------------------------------
    // Tool-plan sanitizer (B)
    // ------------------------------------------------------------------------

    private fun isOperationalTool(t: ToolCall): Boolean = when (t) {
        // ✅ confirmations are operational: they produce Eff.ConfirmCellValue and may produce an edit
        is ToolCall.ConfirmCellValue,
        is ToolCall.ConfirmCellValueRC,

        is ToolCall.ApplyUserEdit,
        is ToolCall.ApplyUserEditRC,

        is ToolCall.ApplyUserClassify,
        is ToolCall.ApplyUserClassifyRC,

        is ToolCall.ReclassifyCell,
        is ToolCall.ReclassifyCells,
        is ToolCall.ReclassifyCellRC,

        is ToolCall.SetCandidates,
        is ToolCall.ClearCandidates,
        is ToolCall.ToggleCandidate -> true

        else -> false
    }

    private fun isControlTool(t: ToolCall): Boolean = when (t) {
        is ToolCall.AskConfirmCellRC,
        is ToolCall.AskConfirmCell,
        is ToolCall.ProposeEdit,
        is ToolCall.RecommendRetake,
        ToolCall.RecommendValidate,
        is ToolCall.ConfirmInterpretation,
        is ToolCall.AskClarifyingQuestion,
        is ToolCall.SwitchToTap,
        ToolCall.Noop -> true
        else -> false
    }

    private fun sanitizeToolPlan(s: SudoState, toolsIn: List<ToolCall>): List<ToolCall> {
        if (toolsIn.isEmpty()) return toolsIn

        val replyIn = toolsIn.filterIsInstance<ToolCall.Reply>().firstOrNull()
        val replyTextIn = replyIn?.text?.trim().orEmpty()

        // If no reply, do not try to "fix" here; applyPolicyTools already replans.
        if (replyTextIn.isEmpty()) return toolsIn

        // We'll allow ToolPlanValidator to rewrite the reply text if it rewrites the asked cell.
        var replyTextOut = replyTextIn

        fun idxFromRC(row: Int, col: Int): Int = (row - 1) * 9 + (col - 1)

        fun idxToRC(idx: Int): Pair<Int, Int> {
            val r = (idx / 9) + 1
            val c = (idx % 9) + 1
            return r to c
        }

        fun parseRCFromPrompt(prompt: String): Pair<Int, Int>? {
            val p = prompt.trim()

            Regex("""\br\s*([1-9])\s*c\s*([1-9])\b""", RegexOption.IGNORE_CASE).find(p)?.let {
                val r = it.groupValues[1].toInt()
                val c = it.groupValues[2].toInt()
                return r to c
            }

            Regex("""\brow\s*([1-9])\b.*\bcol(?:umn)?\s*([1-9])\b""", RegexOption.IGNORE_CASE).find(p)?.let {
                val r = it.groupValues[1].toInt()
                val c = it.groupValues[2].toInt()
                return r to c
            }

            return null
        }

        // ✅ Unique-stop state available at conductor level (last line of defense)
        val g = llmGrid(s)
        val uniqueStop = (s.mode == SudoMode.GRID_SESSION &&
                g != null &&
                g.solvability == "unique" &&
                g.mismatchCells.isEmpty()
                )

        // Helper: does a “next step” exist in this state?
        fun nextStepExists(g: com.contextionary.sudoku.logic.LLMGridState?): Boolean {
            if (g == null) return false
            if (g.mismatchCells.isNotEmpty()) return true
            if (g.solvability == "none" && g.unresolvedCells.isNotEmpty()) return true
            if (g.solvability == "unique") return true
            if (g.solvability == "multiple") return true
            if (g.retakeRecommendation != "none") return true
            return false
        }

        // --------------------------------------------------------------------
        // ✅ Reply text patcher (when we rewrite the asked cell)
        // --------------------------------------------------------------------
        fun patchReplyToRC(text: String, newRow: Int, newCol: Int): String {
            var out = text

            // Replace first "row X, column Y" occurrence if present
            val rcRegex = Regex(
                pattern = "\\brow\\s*(\\d)\\s*(?:,\\s*)?(?:col(?:umn)?)\\s*(\\d)\\b",
                option = RegexOption.IGNORE_CASE
            )
            val m1 = rcRegex.find(out)
            if (m1 != null) {
                val oldR = m1.groupValues[1].toIntOrNull()
                val oldC = m1.groupValues[2].toIntOrNull()
                if (oldR != null && oldC != null && (oldR != newRow || oldC != newCol)) {
                    out = out.replaceRange(m1.range, "row $newRow, column $newCol")
                    return out
                }
            }

            // Replace first "rNcM" token (e.g., r6c2)
            val compactRegex = Regex("\\br(\\d)c(\\d)\\b", RegexOption.IGNORE_CASE)
            val m2 = compactRegex.find(out)
            if (m2 != null) {
                val oldR = m2.groupValues[1].toIntOrNull()
                val oldC = m2.groupValues[2].toIntOrNull()
                if (oldR != null && oldC != null && (oldR != newRow || oldC != newCol)) {
                    out = out.replaceRange(m2.range, "r${newRow}c${newCol}")
                }
            }

            return out
        }

        // 1) Dedupe operational tools that are semantically idempotent.
        val ops = mutableListOf<ToolCall>()
        val seenApplyEdits = mutableSetOf<String>()
        val seenClassify = mutableSetOf<String>()
        val seenConfirms = mutableSetOf<String>()

        val operationalIn = toolsIn.filter { isOperationalTool(it) }.map { toolWireName(it) }

        toolsIn.forEach { t ->
            if (!isOperationalTool(t)) return@forEach
            when (t) {
                is ToolCall.ApplyUserEdit -> {
                    val idx = t.cellIndex
                    val d = t.digit
                    if (idx in 0..80 && d in 0..9) {
                        val key = "idx=$idx|d=$d"
                        if (seenApplyEdits.add(key)) ops += t
                    }
                }

                is ToolCall.ApplyUserEditRC -> {
                    val valid = (t.row in 1..9 && t.col in 1..9 && t.digit in 0..9)
                    if (valid) {
                        val idx = idxFromRC(t.row, t.col)
                        val key = "idx=$idx|d=${t.digit}"
                        if (seenApplyEdits.add(key)) ops += t
                    }
                }

                is ToolCall.ApplyUserClassify -> {
                    val idx = t.cellIndex
                    if (idx in 0..80) {
                        val key = "idx=$idx|cls=${t.cellClass.name}"
                        if (seenClassify.add(key)) ops += t
                    }
                }

                is ToolCall.ApplyUserClassifyRC -> {
                    val valid = (t.row in 1..9 && t.col in 1..9)
                    if (valid) {
                        val idx = idxFromRC(t.row, t.col)
                        val key = "idx=$idx|cls=${t.cellClass.name}"
                        if (seenClassify.add(key)) ops += t
                    }
                }

                is ToolCall.ConfirmCellValue -> {
                    val idx = t.cellIndex
                    val d = t.digit
                    if (idx in 0..80 && d in 0..9) {
                        val key = "idx=$idx|d=$d"
                        if (seenConfirms.add(key)) ops += t
                    }
                }

                is ToolCall.ConfirmCellValueRC -> {
                    val valid = (t.row in 1..9 && t.col in 1..9 && t.digit in 0..9)
                    if (valid) {
                        val idx = idxFromRC(t.row, t.col)
                        val key = "idx=$idx|d=${t.digit}"
                        if (seenConfirms.add(key)) ops += t
                    }
                }

                else -> ops += t
            }
        }

        // ✅ Tripwire: if operational tools came in but none survived, log it loudly
        if (operationalIn.isNotEmpty() && ops.isEmpty()) {
            runCatching {
                ConversationTelemetry.emitKv(
                    "TOOLPLAN_SANITIZE_DROPPED_OPERATIONAL",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "operational_in" to operationalIn,
                    "tools_in" to toolsIn.map { toolWireName(it) }
                )
            }
        }

        // --------------------------------------------------------------------
        // ✅ Handshake insertion at the TOOL PLAN level (log-visible)
        //
        // If pending is ask_cell_value(idx) AND the plan includes an APPLY for idx
        // BUT does not include any CONFIRM for idx, insert a synthetic confirm tool
        // (prefer RC form so it matches your DNA and logs cleanly).
        // --------------------------------------------------------------------
        runCatching {
            val pendingIdx = (s.pending as? Pending.AskCellValue)?.cellIndex?.coerceIn(0, 80)
            if (pendingIdx != null) {

                // Find any apply targeting the pending idx (prefer first match)
                var applyDigit: Int? = null
                var applySource: String? = null

                ops.forEach { op ->
                    when (op) {
                        is ToolCall.ApplyUserEdit -> {
                            if (op.cellIndex == pendingIdx) {
                                applyDigit = op.digit
                                applySource = op.source
                                return@forEach
                            }
                        }
                        is ToolCall.ApplyUserEditRC -> {
                            val idx = if (op.row in 1..9 && op.col in 1..9) idxFromRC(op.row, op.col) else -999
                            if (idx == pendingIdx) {
                                applyDigit = op.digit
                                applySource = op.source
                                return@forEach
                            }
                        }
                        else -> Unit
                    }
                }

                if (applyDigit != null && applySource != null) {
                    val hasAnyConfirmForIdx = ops.any { op ->
                        when (op) {
                            is ToolCall.ConfirmCellValue -> op.cellIndex == pendingIdx
                            is ToolCall.ConfirmCellValueRC -> {
                                val idx = if (op.row in 1..9 && op.col in 1..9) idxFromRC(op.row, op.col) else -999
                                idx == pendingIdx
                            }
                            else -> false
                        }
                    }

                    if (!hasAnyConfirmForIdx) {
                        val (r, c) = idxToRC(pendingIdx)
                        val synthSource = "synth_confirm_from_apply:${applySource}"

                        val key = "idx=$pendingIdx|d=$applyDigit"
                        if (seenConfirms.add(key)) {
                            // Put confirm BEFORE apply for readability and deterministic ordering
                            ops.add(
                                0,
                                ToolCall.ConfirmCellValueRC(
                                    row = r,
                                    col = c,
                                    digit = applyDigit!!,
                                    source = synthSource
                                )
                            )

                            ConversationTelemetry.emitKv(
                                "TOOLPLAN_SANITIZER_INSERTED_CONFIRM",
                                "turn_seq" to s.turnSeq,
                                "mode" to s.mode.name,
                                "pending_idx" to pendingIdx,
                                "row" to r,
                                "col" to c,
                                "digit" to applyDigit,
                                "apply_source" to applySource
                            )
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // ✅ NEW: Confirm/Apply consistency enforcement (DNA: confirm digit is TRUE digit)
        //
        // For ANY idx in the plan:
        // - If we have confirm(idx, d_true) AND apply(idx, d_apply) and d_apply != d_true,
        //   rewrite apply digit to d_true (keep RC vs idx form) and emit telemetry.
        //
        // This makes the tool plan internally coherent and guarantees we only mutate
        // the grid to the user-confirmed TRUE digit.
        // --------------------------------------------------------------------
        runCatching {
            data class ConfirmInfo(val digit: Int, val tool: ToolCall)
            data class ApplyInfo(val digit: Int, val tool: ToolCall, val pos: Int)

            val confirmByIdx = mutableMapOf<Int, ConfirmInfo>()
            val applyInfos = mutableListOf<ApplyInfo>()

            // collect confirms + applies (first confirm wins; conflicts logged)
            ops.forEachIndexed { i, op ->
                when (op) {
                    is ToolCall.ConfirmCellValue -> {
                        val idx = op.cellIndex
                        if (idx in 0..80 && op.digit in 0..9) {
                            val prev = confirmByIdx[idx]
                            if (prev == null) {
                                confirmByIdx[idx] = ConfirmInfo(op.digit, op)
                            } else if (prev.digit != op.digit) {
                                ConversationTelemetry.emitKv(
                                    "TOOLPLAN_SANITIZER_MULTIPLE_CONFIRMS_CONFLICT",
                                    "turn_seq" to s.turnSeq,
                                    "mode" to s.mode.name,
                                    "idx" to idx,
                                    "digit_a" to prev.digit,
                                    "digit_b" to op.digit
                                )
                            }
                        }
                    }

                    is ToolCall.ConfirmCellValueRC -> {
                        if (op.row in 1..9 && op.col in 1..9 && op.digit in 0..9) {
                            val idx = idxFromRC(op.row, op.col)
                            val prev = confirmByIdx[idx]
                            if (prev == null) {
                                confirmByIdx[idx] = ConfirmInfo(op.digit, op)
                            } else if (prev.digit != op.digit) {
                                ConversationTelemetry.emitKv(
                                    "TOOLPLAN_SANITIZER_MULTIPLE_CONFIRMS_CONFLICT",
                                    "turn_seq" to s.turnSeq,
                                    "mode" to s.mode.name,
                                    "idx" to idx,
                                    "row" to op.row,
                                    "col" to op.col,
                                    "digit_a" to prev.digit,
                                    "digit_b" to op.digit
                                )
                            }
                        }
                    }

                    is ToolCall.ApplyUserEdit -> {
                        val idx = op.cellIndex
                        if (idx in 0..80 && op.digit in 0..9) {
                            applyInfos += ApplyInfo(op.digit, op, i)
                        }
                    }

                    is ToolCall.ApplyUserEditRC -> {
                        if (op.row in 1..9 && op.col in 1..9 && op.digit in 0..9) {
                            applyInfos += ApplyInfo(op.digit, op, i)
                        }
                    }

                    else -> Unit
                }
            }

            // rewrite applies to match confirm digit
            var rewroteAny = false

            applyInfos.forEach { a ->
                val idx: Int = when (val t = a.tool) {
                    is ToolCall.ApplyUserEdit -> t.cellIndex
                    is ToolCall.ApplyUserEditRC -> idxFromRC(t.row, t.col)
                    else -> -999
                }

                val c = confirmByIdx[idx] ?: return@forEach
                val dTrue = c.digit
                val dApply = a.digit

                if (dApply != dTrue) {
                    rewroteAny = true

                    val newTool: ToolCall = when (val t = a.tool) {
                        is ToolCall.ApplyUserEdit -> t.copy(digit = dTrue)
                        is ToolCall.ApplyUserEditRC -> t.copy(digit = dTrue)
                        else -> a.tool
                    }

                    ops[a.pos] = newTool

                    ConversationTelemetry.emitKv(
                        "TOOLPLAN_SANITIZER_REWROTE_APPLY_TO_CONFIRM_DIGIT",
                        "turn_seq" to s.turnSeq,
                        "mode" to s.mode.name,
                        "idx" to idx,
                        "confirm_digit" to dTrue,
                        "apply_digit_in" to dApply,
                        "apply_wire" to toolWireName(a.tool),
                        "confirm_wire" to toolWireName(c.tool)
                    )
                }
            }

            // second-pass dedupe for applies in case rewrite created duplicates
            if (rewroteAny) {
                val newOps = mutableListOf<ToolCall>()
                val seenApply2 = mutableSetOf<String>()

                ops.forEach { op ->
                    when (op) {
                        is ToolCall.ApplyUserEdit -> {
                            val idx = op.cellIndex
                            val d = op.digit
                            val key = "idx=$idx|d=$d"
                            if (idx in 0..80 && d in 0..9) {
                                if (seenApply2.add(key)) newOps += op else {
                                    ConversationTelemetry.emitKv(
                                        "TOOLPLAN_SANITIZER_DROPPED_DUP_APPLY_AFTER_REWRITE",
                                        "turn_seq" to s.turnSeq,
                                        "mode" to s.mode.name,
                                        "idx" to idx,
                                        "digit" to d
                                    )
                                }
                            } else newOps += op
                        }

                        is ToolCall.ApplyUserEditRC -> {
                            if (op.row in 1..9 && op.col in 1..9 && op.digit in 0..9) {
                                val idx = idxFromRC(op.row, op.col)
                                val key = "idx=$idx|d=${op.digit}"
                                if (seenApply2.add(key)) newOps += op else {
                                    ConversationTelemetry.emitKv(
                                        "TOOLPLAN_SANITIZER_DROPPED_DUP_APPLY_AFTER_REWRITE",
                                        "turn_seq" to s.turnSeq,
                                        "mode" to s.mode.name,
                                        "idx" to idx,
                                        "digit" to op.digit
                                    )
                                }
                            } else newOps += op
                        }

                        else -> newOps += op
                    }
                }

                ops.clear()
                ops.addAll(newOps)
            }
        }

        // 2) Pick exactly ONE control tool deterministically (priority order).
        val controls = toolsIn.filter { it !is ToolCall.Reply && isControlTool(it) && !isOperationalTool(it) }

        fun controlPriority(t: ToolCall): Int = when (t) {
            is ToolCall.SwitchToTap -> 0
            is ToolCall.ConfirmInterpretation -> 1
            is ToolCall.AskClarifyingQuestion -> 2
            is ToolCall.AskConfirmCellRC -> 3
            is ToolCall.AskConfirmCell -> 4
            is ToolCall.ProposeEdit -> 5
            ToolCall.RecommendValidate -> 6
            is ToolCall.RecommendRetake -> 7
            ToolCall.Noop -> 99
            else -> 50
        }

        var control: ToolCall? = controls.minByOrNull { controlPriority(it) }

        // 3) In GRID_SESSION, prefer RC; and upgrade legacy AskConfirmCell -> RC if prompt contains rc.
        val isGrid = (s.mode == SudoMode.GRID_SESSION)
        if (isGrid && control is ToolCall.AskConfirmCell) {
            val rc = parseRCFromPrompt(control.prompt)
            if (rc != null) {
                val (r, c) = rc
                control = ToolCall.AskConfirmCellRC(row = r, col = c, prompt = control.prompt)
            }
        }

        // ✅ (4) ENFORCE CASE 3 HARD (unique + no mismatch)
        if (uniqueStop) {
            if (control !is ToolCall.RecommendValidate) {
                runCatching {
                    ConversationTelemetry.emitKv(
                        "TOOLPLAN_VALIDATOR_REWRITE",
                        "turn_seq" to s.turnSeq,
                        "mode" to s.mode.name,
                        "reason" to "case3_unique_no_mismatch_enforce_only_validate",
                        "control_in" to (control?.let { toolWireName(it) } ?: "none"),
                        "control_out" to ToolCall.RECOMMEND_VALIDATE,
                        "solvability" to g?.solvability,
                        "mismatch_sz" to (g?.mismatchCells?.size ?: 0),
                        "unresolved_sz" to (g?.unresolvedCells?.size ?: 0)
                    )
                }
            }
            control = ToolCall.RecommendValidate
        }

        // --------------------------------------------------------------------
        // ✅ ToolPlanValidator — enforce NEXT CHECK policy + patch reply if rewritten
        // --------------------------------------------------------------------
        fun enforceNextCheckPolicy(
            g: LLMGridState?,
            controlIn: ToolCall?,
            replyTextIn: String
        ): Pair<ToolCall?, String> {
            if (!isGrid || g == null) return controlIn to replyTextIn

            val mismatch = g.mismatchCells.filter { it in 0..80 }
            val unresolved = g.unresolvedCells.filter { it in 0..80 }

            val case1 = mismatch.isNotEmpty()
            val case2 = (!case1 && g.solvability == "none" && unresolved.isNotEmpty())
            val case3 = (!case1 && g.solvability == "unique" && mismatch.isEmpty())
            val case4 = (!case1 && g.solvability == "multiple" && mismatch.isEmpty())

            val allowed: List<Int> = when {
                case1 -> mismatch.sorted()
                case2 -> unresolved.sorted()
                else -> emptyList()
            }

            fun controlAskedIdx(t: ToolCall): Int? = when (t) {
                is ToolCall.AskConfirmCellRC ->
                    if (t.row in 1..9 && t.col in 1..9) idxFromRC(t.row, t.col) else null
                is ToolCall.AskConfirmCell ->
                    if (t.cellIndex in 0..80) t.cellIndex else null
                is ToolCall.ProposeEdit ->
                    if (t.cellIndex in 0..80) t.cellIndex else null
                else -> null
            }

            fun mkAskConfirmForIdx(idx: Int): ToolCall {
                val (r, c) = idxToRC(idx)
                return ToolCall.AskConfirmCellRC(
                    row = r,
                    col = c,
                    prompt = "Can you confirm what’s in row $r, column $c on your puzzle?"
                )
            }

            fun mkClarify(): ToolCall = ToolCall.AskClarifyingQuestion(
                kind = ClarifyKind.POSITION,
                prompt = "Which exact cell should we check next? (Say: row 7 column 8.)"
            )

            if (case3 || case4) {
                if (controlIn !is ToolCall.RecommendValidate) {
                    runCatching {
                        ConversationTelemetry.emitKv(
                            "TOOLPLAN_VALIDATOR_REWRITE",
                            "turn_seq" to s.turnSeq,
                            "mode" to s.mode.name,
                            "reason" to "case_${if (case3) "3_unique_no_mismatch" else "4_multiple_no_mismatch"}_enforce_only_validate",
                            "control_in" to (controlIn?.let { toolWireName(it) } ?: "none"),
                            "control_out" to ToolCall.RECOMMEND_VALIDATE,
                            "solvability" to g.solvability,
                            "mismatch_sz" to mismatch.size,
                            "unresolved_sz" to unresolved.size
                        )
                    }
                }
                return ToolCall.RecommendValidate to replyTextIn
            }

            if (allowed.isNotEmpty()) {
                if (controlIn is ToolCall.AskClarifyingQuestion || controlIn is ToolCall.ConfirmInterpretation) {
                    return controlIn to replyTextIn
                }

                val askedIdx = controlIn?.let { controlAskedIdx(it) }
                val ok = (askedIdx != null && askedIdx in allowed)

                if (ok) {
                    if (controlIn is ToolCall.AskConfirmCell) {
                        val (r, c) = idxToRC(controlIn.cellIndex)
                        return ToolCall.AskConfirmCellRC(row = r, col = c, prompt = controlIn.prompt) to replyTextIn
                    }
                    return controlIn to replyTextIn
                }

                val replacement: ToolCall =
                    if (allowed.isNotEmpty()) mkAskConfirmForIdx(allowed.first()) else mkClarify()

                val repRc = replacement as? ToolCall.AskConfirmCellRC
                val repIdx: Int? = repRc?.let { idxFromRC(it.row, it.col) }

                runCatching {
                    ConversationTelemetry.emitKv(
                        "TOOLPLAN_VALIDATOR_REWRITE",
                        "turn_seq" to s.turnSeq,
                        "mode" to s.mode.name,
                        "reason" to (if (case1) "case1_mismatch_enforce_allowed" else "case2_unresolved_enforce_allowed"),
                        "control_in" to (controlIn?.let { toolWireName(it) } ?: "none"),
                        "control_out" to toolWireName(replacement),
                        "asked_idx" to askedIdx,
                        "replacement_idx" to repIdx,
                        "allowed_first" to allowed.firstOrNull(),
                        "allowed_sz" to allowed.size,
                        "solvability" to g.solvability
                    )
                }

                // ✅ If we forced a specific RC ask, make the reply text match it.
                if (repRc != null) {
                    val patched = patchReplyToRC(replyTextIn, repRc.row, repRc.col)
                    return replacement to patched
                }

                return replacement to replyTextIn
            }

            return controlIn to replyTextIn
        }

        val enforced = enforceNextCheckPolicy(g, control, replyTextOut)
        control = enforced.first
        replyTextOut = enforced.second

        if (control == null && ops.isNotEmpty() && nextStepExists(g)) {
            control = ToolCall.AskClarifyingQuestion(
                kind = ClarifyKind.POSITION,
                prompt = "Quick one — which cell are we poking next? (Say: row 7 column 8.)"
            )
        }

        val out = mutableListOf<ToolCall>()

        val replyOut: ToolCall.Reply? =
            if (replyIn != null && replyTextOut != replyTextIn) {
                ToolCall.Reply(text = replyTextOut)
            } else {
                replyIn
            }

        replyOut?.let { out += it }
        out += ops
        control?.let { out += it }

        runCatching {
            val inNames = toolsIn.map { toolWireName(it) }
            val outNames = out.map { toolWireName(it) }

            val replyRewritten = (replyTextOut != replyTextIn)

            if (inNames != outNames || replyRewritten) {
                ConversationTelemetry.emitKv(
                    "TOOLPLAN_SANITIZED",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "in_wire_names" to inNames,
                    "out_wire_names" to outNames,
                    "unique_stop" to uniqueStop,
                    "reply_rewritten" to replyRewritten
                )
            }
        }

        return out
    }


    // ------------------------------------------------------------------------
// Policy tool application (A preserved) + (B sanitizer)
// ------------------------------------------------------------------------
    private fun applyPolicyTools(s: SudoState, tools: List<ToolCall>): Next {

        val sanitized = sanitizeToolPlan(s, tools)

        val firstReply = sanitized.filterIsInstance<ToolCall.Reply>().firstOrNull()
        val replyText0 = firstReply?.text?.trim().orEmpty()
        val hasReply = replyText0.isNotEmpty()

        val firstNonReply = sanitized.firstOrNull { it !is ToolCall.Reply }
        val noopAlone = sanitized.size == 1 && sanitized.firstOrNull() is ToolCall.Noop

        if (!hasReply || noopAlone) {
            runCatching {
                ConversationTelemetry.emitKv(
                    "TOOLPLAN_INVALID",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "tool_count" to sanitized.size,
                    "tool_names" to sanitized.map { it.javaClass.simpleName },
                    "reason" to (if (!hasReply) "missing_reply" else "noop_alone"),
                    "main_tool" to (firstNonReply?.javaClass?.simpleName ?: "none")
                )
            }

            val nextMissing = (s.repairAttempt + 1).coerceAtMost(MAX_MISSING_REPLY_REPLANS)
            val s2 = s.copy(repairAttempt = nextMissing)

            if (nextMissing < MAX_MISSING_REPLY_REPLANS) {
                val header = buildString {
                    append(buildStateHeader(s2))
                    append("\nprotocol_violation=")
                    append(if (!hasReply) "missing_reply" else "noop_alone")
                    append("\nprotocol_rule=Every response MUST include reply(text=non_empty). noop cannot be alone.")
                }

                return Next(
                    s2,
                    listOf(
                        callPolicyWithTrace(
                            traceState = s2,
                            userText = "[EVENT] protocol_violation_${if (!hasReply) "missing_reply" else "noop_alone"}",
                            stateHeader = header,
                            mode = s2.mode,
                            reason = "protocol_guard_replan",
                            turnId = s2.turnSeq
                        )
                    )
                )
            }

            val effects = mutableListOf<Eff>()
            effects += Eff.UpdateUiMessage(SURVIVAL_REPLY)
            effects += Eff.Speak(SURVIVAL_REPLY, listenAfter = true)

            runCatching {
                ConversationTelemetry.emitKv(
                    "TOOLPLAN_FALLBACK_REPLY_USED",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "reason" to (if (!hasReply) "missing_reply_terminal" else "noop_alone_terminal")
                )
            }

            return Next(
                s.copy(lastAssistantText = SURVIVAL_REPLY),
                effects
            )
        }

        fun isInternalToolPlanErrorText(text: String): Boolean =
            text.lowercase().contains("something went wrong while preparing the tool plan")

        var replyText = replyText0

        val operationalMain: ToolCall? = firstNonReply
        val controlTool: ToolCall? = sanitized.firstOrNull { t ->
            t !is ToolCall.Reply && isControlTool(t) && !isOperationalTool(t)
        }

        if (isInternalToolPlanErrorText(replyText)) {
            ConversationTelemetry.emitKv(
                "INTERNAL_TOOLPLAN_TEXT_STRIPPED_CONDUCTOR",
                "turn_seq" to s.turnSeq,
                "mode" to s.mode.name
            )
            replyText = SURVIVAL_REPLY
        }

        val hasApplyEditTool =
            sanitized.any { it is ToolCall.ApplyUserEdit } ||
                    sanitized.any { it is ToolCall.ApplyUserEditRC }

        val hasValidNextStepControlTool =
            sanitized.any { t ->
                (t !is ToolCall.Reply) &&
                        isControlTool(t) &&
                        !isOperationalTool(t) &&
                        t !is ToolCall.Noop
            }

        if (hasApplyEditTool && !hasValidNextStepControlTool) {
            runCatching {
                ConversationTelemetry.emitKv(
                    "TOOLPLAN_INVALID",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "tool_count" to sanitized.size,
                    "tool_names" to sanitized.map { it.javaClass.simpleName },
                    "reason" to "apply_missing_next_step",
                    "main_tool" to (operationalMain?.javaClass?.simpleName ?: "none"),
                    "reply_preview" to replyText.take(180)
                )
            }

            val nextMissing = (s.repairAttempt + 1).coerceAtMost(MAX_MISSING_REPLY_REPLANS)
            val s2 = s.copy(repairAttempt = nextMissing)

            if (nextMissing < MAX_MISSING_REPLY_REPLANS) {
                val header = buildString {
                    append(buildStateHeader(s2))
                    append("\nprotocol_violation=apply_missing_next_step")
                    append("\nprotocol_rule=If apply_user_edit(_rc) is emitted, you MUST also emit ONE next-step control tool (ask_confirm_cell_rc / ask_confirm_cell / ask_clarifying_question / confirm_interpretation / recommend_validate / recommend_retake / switch_to_tap) in the same response.")
                }

                return Next(
                    s2,
                    listOf(
                        callPolicyWithTrace(
                            traceState = s2,
                            userText = "[EVENT] protocol_violation_apply_missing_next_step",
                            stateHeader = header,
                            mode = s2.mode,
                            reason = "protocol_guard_replan",
                            turnId = s2.turnSeq
                        )
                    )
                )
            }

            val effects = mutableListOf<Eff>()
            effects += Eff.UpdateUiMessage(SURVIVAL_REPLY)
            effects += Eff.Speak(SURVIVAL_REPLY, listenAfter = true)

            runCatching {
                ConversationTelemetry.emitKv(
                    "TOOLPLAN_FALLBACK_REPLY_USED",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "reason" to "apply_missing_next_step_terminal"
                )
            }

            return Next(
                s.copy(lastAssistantText = SURVIVAL_REPLY),
                effects
            )
        }

        val hasOperationalTool =
            sanitized.any { it is ToolCall.ConfirmCellValue } ||
                    sanitized.any { it is ToolCall.ConfirmCellValueRC } ||
                    sanitized.any { it is ToolCall.ApplyUserEdit } ||
                    sanitized.any { it is ToolCall.ApplyUserEditRC } ||
                    sanitized.any { it is ToolCall.ApplyUserClassify } ||
                    sanitized.any { it is ToolCall.ApplyUserClassifyRC } ||
                    sanitized.any { it is ToolCall.ReclassifyCell } ||
                    sanitized.any { it is ToolCall.ReclassifyCells } ||
                    sanitized.any { it is ToolCall.ReclassifyCellRC } ||
                    sanitized.any { it is ToolCall.SetCandidates } ||
                    sanitized.any { it is ToolCall.ClearCandidates } ||
                    sanitized.any { it is ToolCall.ToggleCandidate }

        fun claimsApplied(text: String): Boolean {
            val t = text.lowercase()

            if (Regex("\\b(all set|we're all set|you('re| are) all set)\\b").containsMatchIn(t)) return false

            val hasEditVerb =
                Regex("\\b(applied|updated|changed|cleared|reclassified|toggled|placed|removed|fixed)\\b")
                    .containsMatchIn(t)

            if (!hasEditVerb) return false

            val hasLocation =
                Regex("\\b(r\\s*[1-9]\\s*c\\s*[1-9]|row\\s*[1-9]|col(?:umn)?\\s*[1-9])\\b")
                    .containsMatchIn(t)

            return hasLocation
        }

        var newPending: Pending? = s.pending
        var listenAfter = true
        val extraEffects = mutableListOf<Eff>()
        var lastConfirmation: LastCellConfirmation? = s.lastConfirmation

        fun idxFromRC(row: Int, col: Int): Int = (row - 1) * 9 + (col - 1)

        fun parseRCFromPrompt(prompt: String): Pair<Int, Int>? {
            val p = prompt.trim()

            Regex("""\br\s*([1-9])\s*c\s*([1-9])\b""", RegexOption.IGNORE_CASE).find(p)?.let {
                val r = it.groupValues[1].toInt()
                val c = it.groupValues[2].toInt()
                return r to c
            }

            Regex("""\brow\s*([1-9])\b.*\bcol(?:umn)?\s*([1-9])\b""", RegexOption.IGNORE_CASE).find(p)?.let {
                val r = it.groupValues[1].toInt()
                val c = it.groupValues[2].toInt()
                return r to c
            }

            return null
        }

        // --------------------------------------------------------------------
        // ✅ NEW: Build an "apply map" so confirm can compute `changed` without
        // needing a UI digit snapshot.
        //
        // changed=true  ⇔ the same toolplan also contains an apply for this idx
        // with the same digit (i.e., mismatch path).
        // --------------------------------------------------------------------
        val applyByIdx: MutableMap<Int, Pair<Int, String>> = mutableMapOf() // idx -> (digit, source)

        sanitized.forEach { t ->
            when (t) {
                is ToolCall.ApplyUserEdit -> {
                    val idx = t.cellIndex
                    val d = t.digit
                    if (idx in 0..80 && d in 0..9) {
                        if (!applyByIdx.containsKey(idx)) applyByIdx[idx] = d to t.source
                    }
                }

                is ToolCall.ApplyUserEditRC -> {
                    val r = t.row
                    val c = t.col
                    val d = t.digit
                    if (r in 1..9 && c in 1..9 && d in 0..9) {
                        val idx = idxFromRC(r, c)
                        if (!applyByIdx.containsKey(idx)) applyByIdx[idx] = d to t.source
                    }
                }

                else -> Unit
            }
        }

        // --------------------------------------------------------------------
        // ✅ NEW: Handshake guard (pending ask_cell_value):
        // If the model emitted an APPLY for the pending cell but forgot to emit
        // the CONFIRM tool, synthesize the confirm bookkeeping effect.
        // --------------------------------------------------------------------
        val pendingAskIdx: Int? = (s.pending as? Pending.AskCellValue)?.cellIndex?.coerceIn(0, 80)
        if (pendingAskIdx != null) {
            val hasConfirmForPending = sanitized.any { t ->
                when (t) {
                    is ToolCall.ConfirmCellValue -> t.cellIndex == pendingAskIdx
                    is ToolCall.ConfirmCellValueRC -> {
                        val idx = if (t.row in 1..9 && t.col in 1..9) idxFromRC(t.row, t.col) else -999
                        idx == pendingAskIdx
                    }
                    else -> false
                }
            }

            val applyPending = applyByIdx[pendingAskIdx]
            if (!hasConfirmForPending && applyPending != null) {
                val (d, src) = applyPending
                newPending = null

                extraEffects += Eff.ConfirmCellValue(
                    cellIndex = pendingAskIdx,
                    digit = d,
                    source = "synth_confirm_from_apply:$src",
                    changed = true
                )

                lastConfirmation = LastCellConfirmation(
                    cellIndex = pendingAskIdx,
                    digit = d,
                    changed = true,
                    source = "synth_confirm_from_apply:$src",
                    seq = s.turnSeq
                )

                runCatching {
                    ConversationTelemetry.emitKv(
                        "CONFIRM_SYNTHESIZED_FROM_APPLY",
                        "turn_seq" to s.turnSeq,
                        "mode" to s.mode.name,
                        "idx" to pendingAskIdx,
                        "digit" to d,
                        "apply_source" to src
                    )
                }
            }
        }

        // --------------------------------------------------------------------
        // ✅ Core application loop
        // - confirm tools: bookkeeping only (NO ApplyCellEdit)
        // - apply tools: mutation only
        // --------------------------------------------------------------------
        sanitized.forEach { t ->
            when (t) {

                is ToolCall.ConfirmCellValue -> {
                    val idx = t.cellIndex
                    val d = t.digit
                    if (idx in 0..80 && d in 0..9) {

                        // changed means: "this confirm is paired with an apply in the same plan"
                        val apply = applyByIdx[idx]
                        val changed = (apply != null && apply.first == d)

                        newPending = null

                        extraEffects += Eff.ConfirmCellValue(
                            cellIndex = idx,
                            digit = d,
                            source = t.source,
                            changed = changed
                        )

                        lastConfirmation = LastCellConfirmation(
                            cellIndex = idx,
                            digit = d,
                            changed = changed,
                            source = t.source,
                            seq = s.turnSeq
                        )

                        runCatching {
                            ConversationTelemetry.emitKv(
                                "CONFIRM_CELL_VALUE_HANDLED",
                                "turn_seq" to s.turnSeq,
                                "mode" to s.mode.name,
                                "idx" to idx,
                                "digit" to d,
                                "old_digit" to -1, // unknown by design
                                "changed" to changed,
                                "source" to t.source
                            )
                        }
                    }
                }

                is ToolCall.ConfirmCellValueRC -> {
                    val r = t.row
                    val c = t.col
                    val d = t.digit
                    if (r in 1..9 && c in 1..9 && d in 0..9) {
                        val idx = idxFromRC(r, c)

                        val apply = applyByIdx[idx]
                        val changed = (apply != null && apply.first == d)

                        newPending = null

                        extraEffects += Eff.ConfirmCellValue(
                            cellIndex = idx,
                            digit = d,
                            source = t.source,
                            changed = changed
                        )

                        lastConfirmation = LastCellConfirmation(
                            cellIndex = idx,
                            digit = d,
                            changed = changed,
                            source = t.source,
                            seq = s.turnSeq
                        )

                        runCatching {
                            ConversationTelemetry.emitKv(
                                "CONFIRM_CELL_VALUE_RC_HANDLED",
                                "turn_seq" to s.turnSeq,
                                "mode" to s.mode.name,
                                "row" to r,
                                "col" to c,
                                "idx" to idx,
                                "digit" to d,
                                "old_digit" to -1, // unknown by design
                                "changed" to changed,
                                "source" to t.source
                            )
                        }
                    }
                }

                is ToolCall.ApplyUserEdit -> {
                    val idx = t.cellIndex
                    val digit = t.digit
                    if (idx in 0..80 && digit in 0..9) {
                        newPending = null
                        extraEffects += Eff.ApplyCellEdit(idx, digit, t.source)
                    }
                }

                is ToolCall.ApplyUserEditRC -> {
                    val r = t.row
                    val c = t.col
                    val d = t.digit
                    if (r in 1..9 && c in 1..9 && d in 0..9) {
                        val idx = idxFromRC(r, c)
                        newPending = null
                        extraEffects += Eff.ApplyCellEdit(idx, d, t.source)
                    }
                }

                is ToolCall.ReclassifyCell -> {
                    val idx = t.cellIndex
                    val kind = t.kind.trim().lowercase()
                    if (idx in 0..80 && (kind == "given" || kind == "solution" || kind == "neither")) {
                        extraEffects += Eff.ReclassifyCell(idx, kind, t.source)
                    }
                }

                is ToolCall.ReclassifyCells -> {
                    t.cells.forEach { c2 ->
                        val idx = c2.cellIndex
                        val kind = c2.kind.trim().lowercase()
                        val src = c2.source.ifBlank { t.source }
                        if (idx in 0..80 && (kind == "given" || kind == "solution" || kind == "neither")) {
                            extraEffects += Eff.ReclassifyCell(idx, kind, src)
                        }
                    }
                }

                is ToolCall.ReclassifyCellRC -> {
                    val r = t.row
                    val c = t.col
                    val kind = t.kind.trim().lowercase()
                    if (r in 1..9 && c in 1..9 && (kind == "given" || kind == "solution" || kind == "neither")) {
                        val idx = idxFromRC(r, c)
                        extraEffects += Eff.ReclassifyCell(idx, kind, t.source)
                    }
                }

                is ToolCall.ApplyUserClassify -> {
                    val idx = t.cellIndex
                    if (idx in 0..80) {
                        newPending = null
                        extraEffects += Eff.ApplyCellClassify(
                            cellIndex = idx,
                            cellClass = t.cellClass,
                            source = t.source
                        )
                    }
                }

                is ToolCall.SetCandidates -> {
                    val idx = t.cellIndex
                    val mask = t.mask
                    if (idx in 0..80 && mask in 0..0x1FF) {
                        extraEffects += Eff.SetCandidates(idx, mask, t.source)
                    }
                }

                is ToolCall.ClearCandidates -> {
                    val idx = t.cellIndex
                    if (idx in 0..80) {
                        extraEffects += Eff.SetCandidates(idx, 0, t.source)
                    }
                }

                is ToolCall.ToggleCandidate -> {
                    val idx = t.cellIndex
                    val d = t.digit
                    if (idx in 0..80 && d in 1..9) {
                        extraEffects += Eff.ToggleCandidate(idx, d, t.source)
                    }
                }

                else -> Unit
            }
        }

        if (controlTool != null) {
            when (controlTool) {
                is ToolCall.ProposeEdit -> {
                    newPending = Pending.ConfirmEdit(
                        cellIndex = controlTool.cellIndex,
                        proposedDigit = controlTool.digit,
                        source = controlTool.reason,
                        prompt = ""
                    )
                }

                is ToolCall.AskConfirmCellRC -> {
                    val r = controlTool.row
                    val c = controlTool.col
                    if (r in 1..9 && c in 1..9) {
                        val idx = idxFromRC(r, c)
                        newPending = Pending.AskCellValue(
                            cellIndex = idx,
                            prompt = controlTool.prompt
                        )

                        runCatching {
                            ConversationTelemetry.emitKv(
                                "ASK_CONFIRM_CELL_RC_MAPPED",
                                "turn_seq" to s.turnSeq,
                                "mode" to s.mode.name,
                                "row" to r,
                                "col" to c,
                                "idx" to idx,
                                "prompt_preview" to controlTool.prompt.take(180)
                            )
                        }
                    }
                }

                is ToolCall.AskConfirmCell -> {
                    newPending = Pending.AskCellValue(
                        cellIndex = controlTool.cellIndex,
                        prompt = controlTool.prompt
                    )

                    val rc = parseRCFromPrompt(controlTool.prompt)
                    if (rc != null) {
                        val (r, c) = rc
                        val expectedIdx = idxFromRC(r, c)
                        if (expectedIdx != controlTool.cellIndex) {
                            runCatching {
                                ConversationTelemetry.emitKv(
                                    "ASK_CONFIRM_CELL_LEGACY_MISMATCH",
                                    "turn_seq" to s.turnSeq,
                                    "mode" to s.mode.name,
                                    "tool_idx" to controlTool.cellIndex,
                                    "prompt_row" to r,
                                    "prompt_col" to c,
                                    "prompt_idx" to expectedIdx,
                                    "prompt_preview" to controlTool.prompt.take(220)
                                )
                            }
                        }
                    }
                }

                is ToolCall.RecommendRetake -> {
                    newPending = Pending.ConfirmRetake(
                        strength = controlTool.strength,
                        prompt = ""
                    )
                }

                ToolCall.RecommendValidate -> {
                    newPending = Pending.ConfirmValidate
                }

                is ToolCall.ConfirmInterpretation -> {
                    newPending = Pending.ConfirmInterpretation(
                        row = controlTool.row,
                        col = controlTool.col,
                        digit = controlTool.digit,
                        confidence = controlTool.confidence,
                        prompt = controlTool.prompt
                    )
                }

                is ToolCall.AskClarifyingQuestion -> {
                    newPending = Pending.AskClarification(
                        kind = controlTool.kind,
                        rowHint = null,
                        colHint = null,
                        digitHint = null,
                        prompt = controlTool.prompt
                    )
                }

                is ToolCall.SwitchToTap -> {
                    newPending = Pending.WaitForTap(
                        prompt = controlTool.prompt,
                        digitHint = null,
                        confidence = 0.6f
                    )
                    listenAfter = false
                    extraEffects += Eff.StopAsr("switch_to_tap")
                }

                else -> Unit
            }
        }

        runCatching {
            val wireNames = sanitized.map { toolWireName(it) }
            ConversationTelemetry.emitKv(
                "TOOLS_APPLIED",
                "turn_seq" to s.turnSeq,
                "mode" to s.mode.name,
                "wire_names" to wireNames,
                "main_tool" to (operationalMain?.let { toolWireName(it) } ?: null),
                "control_tool" to (controlTool?.let { toolWireName(it) } ?: null),
                "new_pending" to pendingSummary(newPending),
                "new_pending_type" to pendingType(newPending),
                "new_pending_idx" to pendingIdx(newPending),
                "new_pending_row" to pendingRow(newPending),
                "new_pending_col" to pendingCol(newPending),
                "new_pending_digit" to pendingDigit(newPending),
                "reply_len" to replyText.length,
                "has_operational_tool" to hasOperationalTool,
                "has_last_confirmation" to (lastConfirmation != null)
            )
        }

        val effects = mutableListOf<Eff>()
        effects += extraEffects

        if (claimsApplied(replyText) && !hasOperationalTool) {
            runCatching {
                ConversationTelemetry.emitKv(
                    "LLM_CLAIMED_APPLY_WITHOUT_TOOL",
                    "turn" to s.turnSeq,
                    "mode" to s.mode.name,
                    "main_tool" to (operationalMain?.javaClass?.simpleName ?: "none"),
                    "reply_preview" to replyText.take(180),
                    "last_user_preview" to (s.lastUserText.orEmpty().take(180))
                )
            }

            val nextMissing = (s.repairAttempt + 1).coerceAtMost(MAX_MISSING_REPLY_REPLANS)
            val s2 = s.copy(
                pending = newPending,
                repairAttempt = nextMissing,
                lastConfirmation = lastConfirmation
            )

            if (nextMissing < MAX_MISSING_REPLY_REPLANS) {
                val header = buildStateHeader(s2) + "\nprotocol_violation=claimed_apply_without_tool"
                effects += callPolicyWithTrace(
                    traceState = s2,
                    userText = "[EVENT] claimed_apply_without_tool",
                    stateHeader = header,
                    mode = s2.mode,
                    reason = "claimed_apply_without_tool_replan",
                    turnId = s2.turnSeq
                )
                return Next(s2, effects)
            }

            effects += Eff.UpdateUiMessage(SURVIVAL_REPLY)
            effects += Eff.Speak(SURVIVAL_REPLY, listenAfter = true)
            return Next(s2.copy(lastAssistantText = SURVIVAL_REPLY), effects)
        }

        // --------------------------------------------------------------------
// ✅ Design A: Post-tools continuation (Tick2)
// - If we applied any operational tool, we ask LLM2 to produce the final
//   user-visible reply using toolResults.
// - Otherwise (no operational changes), we keep the LLM1 reply as-is.
// --------------------------------------------------------------------

        val headerForTick2 = buildStateHeader(
            s.copy(
                pending = newPending,
                lastConfirmation = lastConfirmation
            )
        )

// Build toolResults from the *sanitized plan* (matches logs deterministically)
        val toolResults = buildToolResultsForTick2(
            sanitized = sanitized,
            synthesizedConfirm = null // (optional) you can pass something if you want
        )

        val shouldRunTick2 = hasOperationalTool || toolResults.isNotEmpty()

        return if (shouldRunTick2) {

            // We do NOT speak yet. We wait for tick2 to return the final text.
            runCatching {
                ConversationTelemetry.emitKv(
                    "TICK2_REQUESTED",
                    "turn_seq" to s.turnSeq,
                    "mode" to s.mode.name,
                    "tool_results_sz" to toolResults.size,
                    "listen_after" to listenAfter
                )
            }

            val s2 = s.copy(
                pending = newPending,
                // store LLM1 reply as a safe fallback; tick2 may fail
                lastAssistantText = replyText,
                repairAttempt = 0,
                lastConfirmation = lastConfirmation,
                pendingTick2 = PendingTick2(
                    toolResults = toolResults,
                    listenAfter = listenAfter,
                    fallbackText = replyText.ifBlank { SURVIVAL_REPLY }
                )
            )

            Next(
                s2,
                effects + Eff.CallPolicyContinuationTick2(
                    toolResults = toolResults,
                    stateHeader = headerForTick2,
                    mode = s2.mode,
                    reason = "post_tools_tick2",
                    turnId = s2.turnSeq
                )
            )

        } else {

            // No operational tool: keep current behavior (LLM1 reply is already fine)
            effects += Eff.UpdateUiMessage(replyText)
            effects += Eff.Speak(replyText, listenAfter = listenAfter)

            Next(
                s.copy(
                    pending = newPending,
                    lastAssistantText = replyText,
                    repairAttempt = 0,
                    lastConfirmation = lastConfirmation,
                    pendingTick2 = null
                ),
                effects
            )
        }
    }


    // ------------------------------------------------------------------------
    // Pending handling (B): no local parsing/meaning inference.
    // We provide only factual/structural pending context + raw user text to the LLM.
    // ------------------------------------------------------------------------
    private fun handlePendingB(s: SudoState, userText: String, pending: Pending): Next {

        fun idxToRC(idx: Int): Pair<Int, Int> {
            val r = (idx / 9) + 1
            val c = (idx % 9) + 1
            return r to c
        }

        val header = buildString {
            append(buildStateHeader(s))
            append("\n")

            when (pending) {
                is Pending.AskCellValue -> {
                    val idx = pending.cellIndex.coerceIn(0, 80)
                    val (r, c) = idxToRC(idx)
                    append("pending_ctx=ask_cell_value idx=$idx row=$r col=$c")
                }

                is Pending.ConfirmEdit -> {
                    val idx = pending.cellIndex.coerceIn(0, 80)
                    val (r, c) = idxToRC(idx)
                    append("pending_ctx=confirm_edit idx=$idx row=$r col=$c proposedDigit=${pending.proposedDigit} source=${pending.source}")
                }

                is Pending.ConfirmInterpretation -> {
                    append("pending_ctx=confirm_interpretation r=${pending.row ?: "null"} c=${pending.col ?: "null"} d=${pending.digit ?: "null"} conf=${pending.confidence}")
                }

                is Pending.ConfirmRetake -> {
                    append("pending_ctx=confirm_retake strength=${pending.strength}")
                }

                Pending.ConfirmValidate -> {
                    append("pending_ctx=confirm_validate")
                }

                is Pending.AskClarification -> {
                    append("pending_ctx=ask_clarification kind=${pending.kind}")
                }

                is Pending.WaitForTap -> {
                    append("pending_ctx=wait_for_tap conf=${pending.confidence} digitHint=${pending.digitHint ?: "null"}")
                }
            }
        }

        val event = when (pending) {
            is Pending.AskCellValue -> "[EVENT] pending_ask_cell_value raw='$userText'"
            is Pending.ConfirmEdit -> "[EVENT] pending_confirm_edit raw='$userText'"
            is Pending.ConfirmInterpretation -> "[EVENT] pending_confirm_interpretation raw='$userText'"
            is Pending.ConfirmRetake -> "[EVENT] pending_confirm_retake raw='$userText'"
            Pending.ConfirmValidate -> "[EVENT] pending_confirm_validate raw='$userText'"
            is Pending.AskClarification -> "[EVENT] pending_ask_clarification raw='$userText'"
            is Pending.WaitForTap -> "[EVENT] pending_wait_for_tap raw='$userText'"
        }

        val reason = when (pending) {
            is Pending.AskCellValue -> "pending_ask_cell_value_utterance"
            is Pending.ConfirmEdit -> "pending_confirm_edit_utterance"
            is Pending.ConfirmInterpretation -> "pending_confirm_interpretation_utterance"
            is Pending.ConfirmRetake -> "pending_confirm_retake_utterance"
            Pending.ConfirmValidate -> "pending_confirm_validate_utterance"
            is Pending.AskClarification -> "pending_ask_clarification_utterance"
            is Pending.WaitForTap -> "pending_wait_for_tap_user_spoke"
        }

        // B: Keep pending; policy/tools decide when it changes.
        return Next(
            s,
            listOf(
                callPolicyWithTrace(
                    traceState = s,
                    userText = event,
                    stateHeader = header,
                    mode = s.mode,
                    reason = reason,
                    turnId = s.turnSeq
                )
            )
        )
    }

    private fun buildStateHeader(s: SudoState): String {
        val p = when (val pending = s.pending) {
            null -> "pending:none"
            is Pending.ConfirmEdit -> "pending:confirm_edit idx=${pending.cellIndex} digit=${pending.proposedDigit}"
            is Pending.AskCellValue -> "pending:ask_cell_value idx=${pending.cellIndex}"
            is Pending.ConfirmRetake -> "pending:confirm_retake strength=${pending.strength}"
            Pending.ConfirmValidate -> "pending:confirm_validate"
            is Pending.ConfirmInterpretation -> "pending:confirm_interpretation row=${pending.row} col=${pending.col} digit=${pending.digit}"
            is Pending.AskClarification -> "pending:ask_clarification kind=${pending.kind} row=${pending.rowHint} col=${pending.colHint} digit=${pending.digitHint}"
            is Pending.WaitForTap -> "pending:wait_for_tap digitHint=${pending.digitHint}"
        }
        return "mode=${s.mode} $p repairAttempt=${s.repairAttempt} turnSeq=${s.turnSeq}"
    }
}