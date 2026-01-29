package com.contextionary.sudoku.conductor.policy

import android.util.Log
import com.contextionary.sudoku.conductor.ClarifyKind
import com.contextionary.sudoku.conductor.LlmPolicy
import com.contextionary.sudoku.conductor.ToolCall
import com.contextionary.sudoku.logic.LLMGridState
import com.contextionary.sudoku.logic.SudokuLLMConversationCoordinator

/**
 * FM-01 / FM-02 hard rule:
 * - This adapter MUST NOT author any user-facing text.
 * - No fallback prompts, no default Reply strings, no "snag" lines.
 * - If the LLM response is malformed or empty, return emptyList() and let caller retry.
 *
 * Patch 0.5:
 * - If we got actionable (non-Reply) tools but NO Reply, treat as malformed and return emptyList()
 *   so SudoStore's silent retry path can kick in.
 *
 * Patch 1:
 * - Parse ask_confirm_cell_rc(row,col,prompt) into ToolCall.AskConfirmCellRC.
 * - Keep ask_confirm_cell(cell_index,prompt) as legacy.
 *
 * Patch (telemetry):
 * - Emit structured traces on EVERY early-return so "conversation died" causes are obvious.
 */
class CoordinatorPolicyAdapter(
    private val coord: SudokuLLMConversationCoordinator,
    private val systemPrompt: String
) : LlmPolicy {

    override suspend fun decide(
        sessionId: String,
        userText: String,
        stateHeader: String,
        grid: LLMGridState?
    ): List<ToolCall> {

        val isGridSession = stateHeader.contains("mode=GRID_SESSION")

        fun emitEarlyReturn(reason: String, extra: Map<String, Any?> = emptyMap()) {
            runCatching {
                com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                    tag = "LLM_TOOLS_EMPTY_RETURNED",
                    data = mapOf(
                        "session_id" to sessionId,
                        "is_grid_session" to isGridSession,
                        "reason" to reason,
                        "user_preview" to userText.take(160),
                        "state_header_preview" to stateHeader.take(220)
                    ) + extra
                )
            }
        }

        if (grid == null) {
            Log.w("CoordinatorPolicyAdapter", "decide(): grid=null (isGridSession=$isGridSession); returning empty tools")
            emitEarlyReturn(reason = "grid_null")
            return emptyList()
        }

        val llmTools = try {
            coord.sendToLLMTools(
                sessionId = sessionId,
                systemPrompt = systemPrompt,
                gridState = grid,
                userMessage = userText,
                stateHeader = stateHeader
            )
        } catch (t: Throwable) {
            Log.e("CoordinatorPolicyAdapter", "LLM tools call failed (returning empty tools)", t)
            emitEarlyReturn(
                reason = "llm_call_failed",
                extra = mapOf(
                    "throwable" to (t.javaClass.simpleName ?: "Throwable"),
                    "message" to (t.message?.take(180) ?: "")
                )
            )
            return emptyList()
        }

        // -------- Patch 0: raw tools trace (name only) --------
        runCatching {
            com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                tag = "TOOLS_RAW",
                data = mapOf(
                    "session_id" to sessionId,
                    "is_grid_session" to isGridSession,
                    "llm_tools_n" to llmTools.size,
                    "raw_tool_names" to llmTools.map { runCatching { it.name }.getOrNull()?.trim().orEmpty() }
                )
            )
        }

        if (llmTools.isEmpty()) {
            emitEarlyReturn(reason = "llm_returned_empty_tool_list")
            return emptyList()
        }

        val out = mutableListOf<ToolCall>()

        fun Map<String, Any?>.anyToIntAnyOf(vararg keys: String): Int? {
            for (k in keys) {
                val v = this[k] ?: continue
                val n = when (v) {
                    is Int -> v
                    is Long -> v.toInt()
                    is Double -> v.toInt()
                    is Float -> v.toInt()
                    is Number -> v.toInt()
                    is String -> v.trim().toIntOrNull()
                    else -> null
                }
                if (n != null) return n
            }
            return null
        }

        fun Map<String, Any?>.anyToFloatAnyOf(vararg keys: String): Float? {
            for (k in keys) {
                val v = this[k] ?: continue
                val f = when (v) {
                    is Float -> v
                    is Double -> v.toFloat()
                    is Int -> v.toFloat()
                    is Long -> v.toFloat()
                    is Number -> v.toFloat()
                    is String -> v.trim().toFloatOrNull()
                    else -> null
                }
                if (f != null) return f
            }
            return null
        }

        fun Map<String, Any?>.anyToStringAnyOf(vararg keys: String): String? {
            for (k in keys) {
                val v = this[k] ?: continue
                val s = v as? String
                if (!s.isNullOrBlank()) return s
            }
            return null
        }

        fun parseClarifyKind(raw: String?): ClarifyKind? = ClarifyKind.fromWire(raw)

        // -------- Patch 0: TOOLS_PARSED trace data helpers --------
        val rawToolNames = llmTools.map { runCatching { it.name }.getOrNull()?.trim().orEmpty() }

        fun stableArgsJson(args: Map<String, Any?>): String {
            return runCatching { org.json.JSONObject(args).toString() }.getOrElse { args.toString() }
        }

        val rawToolArgsJson = llmTools.map { t ->
            runCatching { stableArgsJson(t.args) }.getOrElse { "<args_unavailable>" }
        }

        fun wireNameOf(tc: ToolCall): String = when (tc) {
            is ToolCall.Reply -> ToolCall.WireNames.REPLY
            is ToolCall.AskConfirmCellRC -> ToolCall.WireNames.ASK_CONFIRM_CELL_RC
            is ToolCall.AskConfirmCell -> ToolCall.WireNames.ASK_CONFIRM_CELL
            is ToolCall.ProposeEdit -> ToolCall.WireNames.PROPOSE_EDIT
            is ToolCall.ApplyUserEdit -> ToolCall.WireNames.APPLY_USER_EDIT
            is ToolCall.ApplyUserEditRC -> ToolCall.WireNames.APPLY_USER_EDIT_RC
            is ToolCall.ConfirmCellValueRC -> ToolCall.WireNames.CONFIRM_CELL_VALUE_RC
            is ToolCall.ReclassifyCell -> ToolCall.WireNames.RECLASSIFY_CELL
            is ToolCall.ReclassifyCells -> ToolCall.WireNames.RECLASSIFY_CELLS
            is ToolCall.ReclassifyCellRC -> ToolCall.WireNames.RECLASSIFY_CELL_RC
            is ToolCall.ApplyUserClassify -> ToolCall.WireNames.APPLY_USER_CLASSIFY
            is ToolCall.ApplyUserClassifyRC -> ToolCall.WireNames.APPLY_USER_CLASSIFY_RC
            is ToolCall.SetCandidates -> ToolCall.WireNames.SET_CANDIDATES
            is ToolCall.ClearCandidates -> ToolCall.WireNames.CLEAR_CANDIDATES
            is ToolCall.ToggleCandidate -> ToolCall.WireNames.TOGGLE_CANDIDATE
            is ToolCall.RecommendRetake -> ToolCall.WireNames.RECOMMEND_RETAKE
            ToolCall.RecommendValidate -> ToolCall.WireNames.RECOMMEND_VALIDATE
            is ToolCall.ConfirmInterpretation -> ToolCall.WireNames.CONFIRM_INTERPRETATION
            is ToolCall.AskClarifyingQuestion -> ToolCall.WireNames.ASK_CLARIFYING_QUESTION
            is ToolCall.SwitchToTap -> ToolCall.WireNames.SWITCH_TO_TAP
            ToolCall.Noop -> ToolCall.WireNames.NOOP
            else -> tc.javaClass.simpleName ?: "unknown"
        }

        fun parsedSummary(tc: ToolCall): String = when (tc) {
            is ToolCall.Reply -> "reply(len=${tc.text.length})"
            is ToolCall.AskConfirmCellRC -> "ask_confirm_cell_rc(r=${tc.row},c=${tc.col})"
            is ToolCall.AskConfirmCell -> "ask_confirm_cell(idx=${tc.cellIndex})"
            is ToolCall.ProposeEdit -> "propose_edit(idx=${tc.cellIndex},d=${tc.digit},conf=${"%.2f".format(tc.confidence)})"
            is ToolCall.ApplyUserEdit -> "apply_user_edit(idx=${tc.cellIndex},d=${tc.digit},src=${tc.source})"
            is ToolCall.ApplyUserEditRC -> "apply_user_edit_rc(r=${tc.row},c=${tc.col},d=${tc.digit},src=${tc.source})"
            is ToolCall.ConfirmCellValueRC -> "confirm_cell_value_rc(r=${tc.row},c=${tc.col},d=${tc.digit},src=${tc.source})"
            is ToolCall.ReclassifyCell -> "reclassify_cell(idx=${tc.cellIndex},kind=${tc.kind})"
            is ToolCall.ReclassifyCells -> "reclassify_cells(n=${tc.cells.size})"
            is ToolCall.ReclassifyCellRC -> "reclassify_cell_rc(r=${tc.row},c=${tc.col},kind=${tc.kind})"
            is ToolCall.ApplyUserClassify -> "apply_user_classify(idx=${tc.cellIndex},class=${tc.cellClass.name})"
            is ToolCall.ApplyUserClassifyRC -> "apply_user_classify_rc(r=${tc.row},c=${tc.col},class=${tc.cellClass.name})"
            is ToolCall.SetCandidates -> "set_candidates(idx=${tc.cellIndex},mask=${tc.mask})"
            is ToolCall.ClearCandidates -> "clear_candidates(idx=${tc.cellIndex})"
            is ToolCall.ToggleCandidate -> "toggle_candidate(idx=${tc.cellIndex},d=${tc.digit})"
            is ToolCall.RecommendRetake -> "recommend_retake(str=${tc.strength})"
            ToolCall.RecommendValidate -> "recommend_validate"
            is ToolCall.ConfirmInterpretation -> "confirm_interpretation(r=${tc.row},c=${tc.col},d=${tc.digit},conf=${"%.2f".format(tc.confidence)})"
            is ToolCall.AskClarifyingQuestion -> "ask_clarifying_question(kind=${ClarifyKind.toWire(tc.kind)})"
            is ToolCall.SwitchToTap -> "switch_to_tap"
            ToolCall.Noop -> "noop"
            else -> tc.javaClass.simpleName ?: "tool"
        }

        // -----------------------------
        // Parse loop
        // -----------------------------
        toolsLoop@ for (t in llmTools) {
            val name = t.name.trim().lowercase()
            val args = t.args

            when (name) {

                ToolCall.WireNames.REPLY -> {
                    val text = args.anyToStringAnyOf("text")?.trim().orEmpty()
                    if (text.isNotEmpty()) out += ToolCall.Reply(text)
                }

                ToolCall.WireNames.ASK_CONFIRM_CELL_RC -> {
                    val row = args.anyToIntAnyOf("row")?.takeIf { it in 1..9 }
                    val col = args.anyToIntAnyOf("col", "column")?.takeIf { it in 1..9 }
                    val prompt = args.anyToStringAnyOf("prompt")?.trim()
                    if (row != null && col != null && !prompt.isNullOrEmpty()) {
                        out += ToolCall.AskConfirmCellRC(row = row, col = col, prompt = prompt)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping ask_confirm_cell_rc (row/col/prompt invalid)")
                    }
                }

                ToolCall.WireNames.ASK_CONFIRM_CELL -> {
                    val idx = args.anyToIntAnyOf("cell_index", "cellIndex")
                    val prompt = args.anyToStringAnyOf("prompt")?.trim()
                    if (idx != null && idx in 0..80 && !prompt.isNullOrEmpty()) {
                        out += ToolCall.AskConfirmCell(cellIndex = idx, prompt = prompt)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping ask_confirm_cell (idx/prompt invalid)")
                    }
                }

                ToolCall.WireNames.PROPOSE_EDIT -> {
                    val idx = args.anyToIntAnyOf("cell_index", "cellIndex")
                    val digit = args.anyToIntAnyOf("digit", "value")
                    val reason = args.anyToStringAnyOf("reason")
                    val conf = args.anyToFloatAnyOf("confidence")?.coerceIn(0f, 1f)

                    if (idx != null && idx in 0..80 &&
                        digit != null && digit in 1..9 &&
                        !reason.isNullOrBlank() &&
                        conf != null
                    ) {
                        out += ToolCall.ProposeEdit(cellIndex = idx, digit = digit, reason = reason, confidence = conf)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping propose_edit (missing/invalid required args)")
                    }
                }

                ToolCall.WireNames.APPLY_USER_EDIT, "apply_edit", "commit_user_edit" -> {
                    val idx = args.anyToIntAnyOf("cell_index", "cellIndex")
                    val digit = args.anyToIntAnyOf("digit", "value")
                    val source = args.anyToStringAnyOf("source")

                    if (idx != null && idx in 0..80 && digit != null && digit in 0..9 && !source.isNullOrBlank()) {
                        out += ToolCall.ApplyUserEdit(cellIndex = idx, digit = digit, source = source)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping apply_user_edit (missing/invalid required args)")
                    }
                }

                ToolCall.WireNames.APPLY_USER_EDIT_RC -> {
                    val row = args.anyToIntAnyOf("row")?.takeIf { it in 1..9 }
                    val col = args.anyToIntAnyOf("col", "column")?.takeIf { it in 1..9 }
                    val digit = args.anyToIntAnyOf("digit", "value")?.takeIf { it in 0..9 }
                    val source = args.anyToStringAnyOf("source")

                    if (row != null && col != null && digit != null && !source.isNullOrBlank()) {
                        out += ToolCall.ApplyUserEditRC(row = row, col = col, digit = digit, source = source)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping apply_user_edit_rc (missing/invalid required args)")
                    }
                }

                // ✅ NEW: explicit confirmation tool (NOT an edit)
                ToolCall.WireNames.CONFIRM_CELL_VALUE_RC -> {
                    val row = args.anyToIntAnyOf("row")?.takeIf { it in 1..9 }
                    val col = args.anyToIntAnyOf("col", "column")?.takeIf { it in 1..9 }
                    val digit = args.anyToIntAnyOf("digit", "value")?.takeIf { it in 0..9 }
                    val source = args.anyToStringAnyOf("source") ?: "user_confirm"

                    if (row != null && col != null && digit != null) {
                        out += ToolCall.ConfirmCellValueRC(row = row, col = col, digit = digit, source = source)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping confirm_cell_value_rc (missing/invalid required args)")
                    }
                }

                ToolCall.WireNames.APPLY_USER_CLASSIFY -> {
                    val idx = args.anyToIntAnyOf("cell_index", "cellIndex")
                    val rawClass = args.anyToStringAnyOf("cell_class", "cellClass", "class")
                    val cellClass = com.contextionary.sudoku.conductor.CellClass.fromWire(rawClass)
                    val source = args.anyToStringAnyOf("source") ?: "user_reclass"

                    if (idx != null && idx in 0..80 && cellClass != null) {
                        out += ToolCall.ApplyUserClassify(cellIndex = idx, cellClass = cellClass, source = source)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping apply_user_classify (missing/invalid args)")
                    }
                }

                ToolCall.WireNames.APPLY_USER_CLASSIFY_RC -> {
                    val row = args.anyToIntAnyOf("row")?.takeIf { it in 1..9 }
                    val col = args.anyToIntAnyOf("col", "column")?.takeIf { it in 1..9 }
                    val rawClass = args.anyToStringAnyOf("cell_class", "cellClass", "class")
                    val cellClass = com.contextionary.sudoku.conductor.CellClass.fromWire(rawClass)
                    val source = args.anyToStringAnyOf("source") ?: "user_reclass"

                    if (row != null && col != null && cellClass != null) {
                        out += ToolCall.ApplyUserClassifyRC(row = row, col = col, cellClass = cellClass, source = source)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping apply_user_classify_rc (missing/invalid args)")
                    }
                }

                ToolCall.WireNames.RECOMMEND_RETAKE -> {
                    val strength = args.anyToStringAnyOf("strength")?.trim()
                    val reason = args.anyToStringAnyOf("reason")?.trim()

                    if (!strength.isNullOrEmpty() && !reason.isNullOrEmpty()) {
                        out += ToolCall.RecommendRetake(strength = strength, reason = reason)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping recommend_retake (missing strength/reason)")
                    }
                }

                ToolCall.WireNames.RECOMMEND_VALIDATE -> out += ToolCall.RecommendValidate

                ToolCall.WireNames.CONFIRM_INTERPRETATION -> {
                    val row = args.anyToIntAnyOf("row")?.takeIf { it in 1..9 }
                    val col = args.anyToIntAnyOf("col", "column")?.takeIf { it in 1..9 }
                    val digit = args.anyToIntAnyOf("digit", "value")?.takeIf { it in 1..9 }
                    val prompt = args.anyToStringAnyOf("prompt")?.trim()
                    val conf = args.anyToFloatAnyOf("confidence")?.coerceIn(0f, 1f)

                    if (!prompt.isNullOrEmpty() && conf != null) {
                        out += ToolCall.ConfirmInterpretation(row = row, col = col, digit = digit, prompt = prompt, confidence = conf)
                    } else {
                        Log.w("CoordinatorPolicyAdapter", "Dropping confirm_interpretation (missing prompt/conf)")
                    }
                }

                ToolCall.WireNames.ASK_CLARIFYING_QUESTION -> {
                    val kind = parseClarifyKind(args.anyToStringAnyOf("kind")) ?: continue@toolsLoop
                    val prompt = args.anyToStringAnyOf("prompt")?.trim()
                    if (!prompt.isNullOrEmpty()) out += ToolCall.AskClarifyingQuestion(kind = kind, prompt = prompt)
                    else Log.w("CoordinatorPolicyAdapter", "Dropping ask_clarifying_question (missing prompt)")
                }

                ToolCall.WireNames.SWITCH_TO_TAP -> {
                    val prompt = args.anyToStringAnyOf("prompt")?.trim()
                    if (!prompt.isNullOrEmpty()) out += ToolCall.SwitchToTap(prompt = prompt)
                    else Log.w("CoordinatorPolicyAdapter", "Dropping switch_to_tap (missing prompt)")
                }

                ToolCall.WireNames.NOOP -> out += ToolCall.Noop

                else -> Unit
            }
        }

        // --------------------------------------------
        // ✅ FINAL ASSEMBLY (guarantee): Reply + ALL ops + EXACTLY ONE control
        // --------------------------------------------
        val replyTool = out.filterIsInstance<ToolCall.Reply>().firstOrNull()

        // Classify operational vs control at the Conductor ToolCall layer
        fun isOperational(tc: ToolCall): Boolean = when (tc) {
            is ToolCall.ApplyUserEdit,
            is ToolCall.ApplyUserEditRC,
            is ToolCall.ConfirmCellValueRC,
            is ToolCall.ApplyUserClassify,
            is ToolCall.ApplyUserClassifyRC,
            is ToolCall.SetCandidates,
            is ToolCall.ClearCandidates,
            is ToolCall.ToggleCandidate,
            is ToolCall.ReclassifyCell,
            is ToolCall.ReclassifyCells,
            is ToolCall.ReclassifyCellRC -> true
            else -> false
        }

        fun isControl(tc: ToolCall): Boolean = when (tc) {
            is ToolCall.AskConfirmCellRC,
            is ToolCall.AskConfirmCell,
            is ToolCall.ProposeEdit,
            is ToolCall.RecommendRetake,
            ToolCall.RecommendValidate,
            is ToolCall.ConfirmInterpretation,
            is ToolCall.AskClarifyingQuestion,
            is ToolCall.SwitchToTap -> true
            else -> false
        }

        fun controlPriority(tc: ToolCall): Int = when (tc) {
            is ToolCall.SwitchToTap -> 0
            is ToolCall.ConfirmInterpretation -> 1
            is ToolCall.AskClarifyingQuestion -> 2
            is ToolCall.AskConfirmCellRC -> 3
            is ToolCall.AskConfirmCell -> 4
            is ToolCall.ProposeEdit -> 5
            ToolCall.RecommendValidate -> 6
            is ToolCall.RecommendRetake -> 7
            else -> 50
        }

        fun opPriority(tc: ToolCall): Int = when (tc) {
            // keep confirmations and edits first if we must cap
            is ToolCall.ConfirmCellValueRC -> 0
            is ToolCall.ApplyUserEditRC -> 1
            is ToolCall.ApplyUserEdit -> 2
            is ToolCall.ApplyUserClassifyRC -> 3
            is ToolCall.ApplyUserClassify -> 4
            // candidates + reclassify later
            is ToolCall.SetCandidates -> 10
            is ToolCall.ClearCandidates -> 11
            is ToolCall.ToggleCandidate -> 12
            is ToolCall.ReclassifyCellRC -> 20
            is ToolCall.ReclassifyCell -> 21
            is ToolCall.ReclassifyCells -> 22
            else -> 99
        }

        val opsAll = out.filter { isOperational(it) }.sortedBy { opPriority(it) }
        val controlsAll = out.filter { isControl(it) }
        val chosenControl = controlsAll.minByOrNull { controlPriority(it) }

        // Trace what we parsed before enforcing the contract
        runCatching {
            com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                tag = "TOOLS_PARSED_PRE_CONTRACT",
                data = mapOf(
                    "session_id" to sessionId,
                    "is_grid_session" to isGridSession,
                    "parsed_total" to out.size,
                    "parsed_wire_names" to out.map { wireNameOf(it) },
                    "parsed_tools" to out.map { parsedSummary(it) },
                    "ops_n" to opsAll.size,
                    "controls_n" to controlsAll.size,
                    "chosen_control" to (chosenControl?.let { wireNameOf(it) } ?: "none")
                )
            )
        }

        // Hard requirements
        if (replyTool == null) {
            runCatching {
                com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                    tag = "TOOLS_MISSING_REPLY",
                    data = mapOf(
                        "session_id" to sessionId,
                        "is_grid_session" to isGridSession,
                        "raw_tool_names" to rawToolNames,
                        "raw_tool_args_json" to rawToolArgsJson,
                        "parsed_wire_names" to out.map { wireNameOf(it) },
                        "parsed_tools" to out.map { parsedSummary(it) }
                    )
                )
            }
            emitEarlyReturn(reason = "missing_reply")
            return emptyList()
        }

        // If we have ops, we MUST have one control (your contract).
        if (opsAll.isNotEmpty() && chosenControl == null) {
            emitEarlyReturn(
                reason = "missing_control_with_ops",
                extra = mapOf(
                    "ops_wire_names" to opsAll.map { wireNameOf(it) },
                    "parsed_wire_names" to out.map { wireNameOf(it) }
                )
            )
            return emptyList()
        }

        // Safety cap (but prioritize confirm/apply tools)
        val maxOps = 12
        val opsCapped = if (opsAll.size <= maxOps) opsAll else opsAll.take(maxOps)

        val finalTools = buildList {
            add(replyTool)
            addAll(opsCapped)
            chosenControl?.let { add(it) }
        }

        // -------- Patch: TOOLS_PARSED trace (post-contract) --------
        runCatching {
            com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                tag = "TOOLS_PARSED",
                data = mapOf(
                    "session_id" to sessionId,
                    "is_grid_session" to isGridSession,
                    "raw_tool_names" to rawToolNames,
                    "raw_tool_args_json" to rawToolArgsJson,
                    "parsed_tools" to finalTools.map { parsedSummary(it) },
                    "parsed_wire_names" to finalTools.map { wireNameOf(it) },
                    "ops_capped" to (opsAll.size != opsCapped.size),
                    "ops_kept_n" to opsCapped.size
                )
            )
        }

        if (finalTools.isEmpty()) {
            emitEarlyReturn(reason = "all_tools_dropped_during_parse_or_contract")
            return emptyList()
        }

        return finalTools
    }
}