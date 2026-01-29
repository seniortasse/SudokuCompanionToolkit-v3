package com.contextionary.sudoku.logic

import com.contextionary.sudoku.profile.UserProfile
import com.contextionary.sudoku.profile.PlayerProfileSnapshot
import com.contextionary.sudoku.profile.toSnapshot
import com.contextionary.sudoku.telemetry.ConversationTelemetry

import com.contextionary.sudoku.conversation.PersonaDescriptor
import com.contextionary.sudoku.conversation.PromptBuilder
import com.contextionary.sudoku.conversation.RecoveryController
import com.contextionary.sudoku.conversation.TurnLifecycleManager
import com.contextionary.sudoku.conversation.TurnStore
import com.contextionary.sudoku.conversation.DeveloperPromptComposer

import com.contextionary.sudoku.conductor.ToolCall as ConductorToolCall

private fun sha256Hex(s: String): String {
    val md = java.security.MessageDigest.getInstance("SHA-256")
    val digest = md.digest(s.toByteArray(Charsets.UTF_8))
    val hex = StringBuilder(digest.size * 2)
    for (b in digest) hex.append(String.format("%02x", b))
    return hex.toString()
}

enum class GridSeverity { OK, MILD, SERIOUS, RETAKE_NEEDED }

fun computeSeverity(
    uniqueSolvable: Boolean,
    unresolvedCount: Int,
    conflictCount: Int,
    changedCount: Int,
    lowConfCount: Int
): GridSeverity = when {
    unresolvedCount > 6 || conflictCount > 6 ->
        GridSeverity.RETAKE_NEEDED

    conflictCount == 0 && unresolvedCount == 0 && uniqueSolvable &&
            changedCount == 0 && lowConfCount == 0 ->
        GridSeverity.OK

    conflictCount == 0 && unresolvedCount == 0 && uniqueSolvable &&
            (changedCount > 0 || lowConfCount > 0) ->
        GridSeverity.MILD

    unresolvedCount in 1..3 && conflictCount == 0 ->
        GridSeverity.MILD

    unresolvedCount in 4..6 || conflictCount in 1..6 ->
        GridSeverity.SERIOUS

    else -> GridSeverity.SERIOUS
}

data class LLMCellEditEvent(
    val seq: Int,
    val index: Int,
    val cellLabel: String,
    val fromDigit: Int,
    val toDigit: Int,
    val whenEpochMs: Long,
    val source: String = "manual"
)

data class LLMGridState(
    val correctedGrid: IntArray,

    val truthIsGiven: BooleanArray = BooleanArray(81),
    val truthIsSolution: BooleanArray = BooleanArray(81),

    val candidateMask81: IntArray = IntArray(81),

    val unresolvedCells: List<Int> = emptyList(),
    val changedCells: List<Int> = emptyList(),
    val conflictCells: List<Int> = emptyList(),

    val lowConfidenceCells: List<Int> = emptyList(),

    val manuallyCorrectedCells: List<Int> = emptyList(),
    val manualEdits: List<LLMCellEditEvent> = emptyList(),

    val deducedSolutionGrid: IntArray? = null,
    val deducedSolutionCountCapped: Int = 0,
    val mismatchCells: List<Int> = emptyList(),
    val mismatchDetails: List<String> = emptyList(),

    // ✅ NEW: cells that were explicitly confirmed by the user (handled already)
    // 0-based indices, stable facts for the LLM
    val confirmedCells: List<Int> = emptyList(),

    val solvability: String,
    val isStructurallyValid: Boolean,

    val unresolvedCount: Int,
    val severity: String,
    val retakeRecommendation: String
) {
    @get:Deprecated("Use solvability instead (\"unique\"|\"multiple\"|\"none\")")
    val uniqueSolvable: Boolean get() = (solvability == "unique")

    @Deprecated("Use the primary constructor with solvability/isStructurallyValid/retakeRecommendation")
    constructor(
        correctedGrid: IntArray,
        unresolvedCells: List<Int>,
        changedCells: List<Int>,
        conflictCells: List<Int>,
        lowConfidenceCells: List<Int>,
        uniqueSolvable: Boolean,
        unresolvedCount: Int,
        severity: String
    ) : this(
        correctedGrid = correctedGrid,

        truthIsGiven = BooleanArray(81),
        truthIsSolution = BooleanArray(81),
        candidateMask81 = IntArray(81),

        unresolvedCells = unresolvedCells,
        changedCells = changedCells,
        conflictCells = conflictCells,
        lowConfidenceCells = lowConfidenceCells,

        manuallyCorrectedCells = emptyList(),
        manualEdits = emptyList(),

        deducedSolutionGrid = null,
        deducedSolutionCountCapped = 0,
        mismatchCells = emptyList(),
        mismatchDetails = emptyList(),

        // ✅ NEW: legacy path -> none
        confirmedCells = emptyList(),

        solvability = if (uniqueSolvable) "unique" else "none",
        isStructurallyValid = conflictCells.isEmpty(),
        unresolvedCount = unresolvedCount,
        severity = when (severity) {
            "retake_needed" -> "serious"
            else -> severity
        },
        retakeRecommendation = when {
            conflictCells.size >= 8 -> "strong"
            conflictCells.size in 4..7 -> "soft"
            else -> "none"
        }
    )
}

data class GridHumanSummary(val message: String)

data class ChatTurn(
    val role: String,
    val text: String
)

data class ToolCallRaw(
    val name: String,
    val args: Map<String, Any?> = emptyMap()
)

data class PolicyRawResponse(
    val tool_calls: List<ToolCallRaw> = emptyList()
)

data class LlmToolCall(
    val name: String,
    val args: Map<String, Any?> = emptyMap()
)

private object ToolNames {
    const val REPLY = ConductorToolCall.WireNames.REPLY

    const val ASK_CONFIRM_CELL = ConductorToolCall.WireNames.ASK_CONFIRM_CELL
    const val ASK_CONFIRM_CELL_RC = ConductorToolCall.WireNames.ASK_CONFIRM_CELL_RC

    const val PROPOSE_EDIT = ConductorToolCall.WireNames.PROPOSE_EDIT

    const val APPLY_USER_EDIT = ConductorToolCall.WireNames.APPLY_USER_EDIT
    const val APPLY_USER_EDIT_RC = ConductorToolCall.WireNames.APPLY_USER_EDIT_RC

    // ✅ NEW: confirmation tools (NOT “apply”; may conditionally apply in Conductor if changed=true)
    const val CONFIRM_CELL_VALUE = ConductorToolCall.WireNames.CONFIRM_CELL_VALUE
    const val CONFIRM_CELL_VALUE_RC = ConductorToolCall.WireNames.CONFIRM_CELL_VALUE_RC

    const val RECOMMEND_RETAKE = ConductorToolCall.WireNames.RECOMMEND_RETAKE
    const val RECOMMEND_VALIDATE = ConductorToolCall.WireNames.RECOMMEND_VALIDATE

    const val CONFIRM_INTERPRETATION = ConductorToolCall.WireNames.CONFIRM_INTERPRETATION
    const val ASK_CLARIFYING_QUESTION = ConductorToolCall.WireNames.ASK_CLARIFYING_QUESTION
    const val SWITCH_TO_TAP = ConductorToolCall.WireNames.SWITCH_TO_TAP

    const val APPLY_USER_CLASSIFY = ConductorToolCall.WireNames.APPLY_USER_CLASSIFY
    const val APPLY_USER_CLASSIFY_RC = ConductorToolCall.WireNames.APPLY_USER_CLASSIFY_RC

    const val SET_CANDIDATES = ConductorToolCall.WireNames.SET_CANDIDATES
    const val CLEAR_CANDIDATES = ConductorToolCall.WireNames.CLEAR_CANDIDATES
    const val TOGGLE_CANDIDATE = ConductorToolCall.WireNames.TOGGLE_CANDIDATE

    const val NOOP = ConductorToolCall.WireNames.NOOP
}

data class FreeTalkRawResponse(val assistant_message: String)

data class UserClue(
    val key: String,
    val value: String,
    val confidence: Float = 0.7f,
    val source: String = "conversation",
    val whenEpochMs: Long = System.currentTimeMillis()
)

data class ClueExtractionRawResponse(val clues: List<UserClue> = emptyList())

interface SudokuLLMClient {
    suspend fun sendGridUpdate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        history: List<Pair<String, String>> // role must be "user" or "assistant"
    ): PolicyRawResponse

    suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): FreeTalkRawResponse

    suspend fun extractClues(
        systemPrompt: String,
        developerPrompt: String,
        transcript: String
    ): ClueExtractionRawResponse
}

class SudokuLLMConversationCoordinator(
    private val solver: SudokuSolver,
    private val llmClient: SudokuLLMClient,

    private val turnStore: TurnStore,
    private val lifecycle: TurnLifecycleManager = TurnLifecycleManager(turnStore),
    private val promptBuilder: PromptBuilder = PromptBuilder(turnStore),
    private val recovery: RecoveryController = RecoveryController(turnStore, lifecycle),

    private val personaSystemPrompt: String = "You are Sudo, a warm, encouraging Sudoku companion.",

    private val persona: PersonaDescriptor = PersonaDescriptor(
        id = "sudo",
        version = 1,
        hash = sha256Hex(personaSystemPrompt)
    )
) {

    companion object {
        private const val DEFAULT_SESSION_ID = "default"
    }

    fun buildHumanSummary(g: LLMGridState): GridHumanSummary {
        val u = g.unresolvedCells.size
        val c = g.conflictCells.size
        val lc = g.lowConfidenceCells.size
        val ch = g.changedCells.size

        val msg = when {
            g.retakeRecommendation == "strong" ->
                "Many conflicts detected. Retake is strongly recommended."
            u > 0 ->
                "Needs confirmation: $u unresolved cell(s)."
            c > 0 ->
                "Conflicts detected: $c cell(s) involved."
            lc > 0 || ch > 0 ->
                "Looks mostly OK. A few cells may need a quick check."
            else ->
                "Looks clean."
        }

        val full =
            "severity=${g.severity} solvability=${g.solvability} structValid=${g.isStructurallyValid} " +
                    "unresolved=$u conflicts=$c lowConf=$lc changed=$ch retake=${g.retakeRecommendation} | $msg"

        ConversationTelemetry.emit(mapOf("type" to "GRID_SUMMARY", "len" to full.length))
        return GridHumanSummary(full)
    }

    suspend fun sendToLLMTools(
        systemPrompt: String,
        gridState: LLMGridState,
        userMessage: String,
        stateHeader: String = ""
    ): List<LlmToolCall> {
        return sendToLLMTools(
            sessionId = DEFAULT_SESSION_ID,
            systemPrompt = systemPrompt,
            gridState = gridState,
            userMessage = userMessage,
            stateHeader = stateHeader
        )
    }

    suspend fun sendToLLMTools(
        sessionId: String,
        systemPrompt: String,
        gridState: LLMGridState,
        userMessage: String,
        stateHeader: String = ""
    ): List<LlmToolCall> {

        fun toolReply(text: String) = LlmToolCall(
            name = ToolNames.REPLY,
            args = mapOf("text" to text)
        )

        fun toolAskConfirmCell(cellIndex: Int, prompt: String) = LlmToolCall(
            name = ToolNames.ASK_CONFIRM_CELL,
            args = mapOf("cell_index" to cellIndex, "prompt" to prompt)
        )

        fun toolAskConfirmCellRC(row: Int, col: Int, prompt: String) = LlmToolCall(
            name = ToolNames.ASK_CONFIRM_CELL_RC,
            args = mapOf("row" to row, "col" to col, "prompt" to prompt)
        )

        // ✅ NEW: confirmation tools
        fun toolConfirmCellValue(cellIndex: Int, digit: Int, source: String) = LlmToolCall(
            name = ToolNames.CONFIRM_CELL_VALUE,
            args = mapOf("cell_index" to cellIndex, "digit" to digit, "source" to source)
        )

        fun toolConfirmCellValueRC(row: Int, col: Int, digit: Int, source: String) = LlmToolCall(
            name = ToolNames.CONFIRM_CELL_VALUE_RC,
            args = mapOf("row" to row, "col" to col, "digit" to digit, "source" to source)
        )

        fun toolRecommendRetake(strength: String, reason: String) = LlmToolCall(
            name = ToolNames.RECOMMEND_RETAKE,
            args = mapOf("strength" to strength, "reason" to reason)
        )

        fun toolRecommendValidate() = LlmToolCall(
            name = ToolNames.RECOMMEND_VALIDATE,
            args = emptyMap()
        )

        fun toolConfirmInterpretation(
            row: Int?,
            col: Int?,
            digit: Int?,
            prompt: String,
            confidence: Float
        ) = LlmToolCall(
            name = ToolNames.CONFIRM_INTERPRETATION,
            args = mapOf(
                "row" to row,
                "col" to col,
                "digit" to digit,
                "prompt" to prompt,
                "confidence" to confidence
            )
        )

        fun toolAskClarifyingQuestion(kind: String, prompt: String) = LlmToolCall(
            name = ToolNames.ASK_CLARIFYING_QUESTION,
            args = mapOf("kind" to kind, "prompt" to prompt)
        )

        fun toolSwitchToTap(prompt: String) = LlmToolCall(
            name = ToolNames.SWITCH_TO_TAP,
            args = mapOf("prompt" to prompt)
        )

        fun userSaysDontSuggestCorrections(msg: String): Boolean {
            val s = msg.lowercase()
            return s.contains("don't suggest") || s.contains("do not suggest") ||
                    s.contains("no suggestion") || s.contains("no suggestions") ||
                    s.contains("don't propose") || s.contains("do not propose") ||
                    s.contains("no correction") || s.contains("no corrections")
        }

        fun userRefusesTap(msg: String): Boolean {
            val s = msg.lowercase()
            return s.contains("don't tap") ||
                    s.contains("do not tap") ||
                    s.contains("no tap") ||
                    s.contains("not tap") ||
                    s.contains("we don't need to tap") ||
                    s.contains("we do not need to tap") ||
                    s.contains("stop asking me to tap") ||
                    s.contains("no tapping")
        }

        fun isToolPlanInternalErrorText(t: String): Boolean {
            val s = t.lowercase()
            return s.contains("something went wrong while preparing the tool plan")
        }

        fun nextStepExists(g: LLMGridState): Boolean {
            if (g.mismatchCells.isNotEmpty()) return true
            if (g.solvability == "none" && g.unresolvedCells.isNotEmpty()) return true
            if (g.solvability == "unique") return true
            if (g.solvability == "multiple") return true
            if (g.retakeRecommendation != "none") return true
            return false
        }

        fun toolIsExtraCheckName(name: String): Boolean {
            return when (name) {
                ToolNames.ASK_CONFIRM_CELL,
                ToolNames.ASK_CONFIRM_CELL_RC,
                ToolNames.PROPOSE_EDIT,
                ToolNames.RECOMMEND_RETAKE,
                ToolNames.SWITCH_TO_TAP -> true
                else -> false
            }
        }

        fun uniqueStopApplies(g: LLMGridState): Boolean =
            (g.solvability == "unique" && g.mismatchCells.isEmpty())

        // ✅ Operational vs Control
        fun isOperationalToolName(name: String): Boolean = when (name) {
            ToolNames.APPLY_USER_EDIT,
            ToolNames.APPLY_USER_EDIT_RC,
            ToolNames.CONFIRM_CELL_VALUE,
            ToolNames.CONFIRM_CELL_VALUE_RC,
            ToolNames.APPLY_USER_CLASSIFY,
            ToolNames.APPLY_USER_CLASSIFY_RC,
            ToolNames.SET_CANDIDATES,
            ToolNames.CLEAR_CANDIDATES,
            ToolNames.TOGGLE_CANDIDATE -> true
            else -> false
        }

        fun isControlToolName(name: String): Boolean = when (name) {
            ToolNames.ASK_CONFIRM_CELL,
            ToolNames.ASK_CONFIRM_CELL_RC,
            ToolNames.PROPOSE_EDIT,
            ToolNames.RECOMMEND_RETAKE,
            ToolNames.RECOMMEND_VALIDATE,
            ToolNames.CONFIRM_INTERPRETATION,
            ToolNames.ASK_CLARIFYING_QUESTION,
            ToolNames.SWITCH_TO_TAP -> true
            else -> false
        }

        // Deterministic fallback control tool if model forgot one
        fun pickDeterministicControl(g: LLMGridState): LlmToolCall? {
            if (g.mismatchCells.isNotEmpty()) {
                val idx = g.mismatchCells.first()
                val r = idx / 9 + 1
                val c = idx % 9 + 1
                return toolAskConfirmCellRC(r, c, "Quick check: what digit is in row $r, column $c on your paper?")
            }
            if (g.solvability == "none" && g.unresolvedCells.isNotEmpty()) {
                val idx = g.unresolvedCells.sorted().first()
                val r = idx / 9 + 1
                val c = idx % 9 + 1
                return toolAskConfirmCellRC(r, c, "Quick check: what digit is in row $r, column $c on your paper (or is it blank)?")
            }
            if (g.solvability == "unique" || g.solvability == "multiple") return toolRecommendValidate()
            if (g.retakeRecommendation != "none") {
                val strength = if (g.retakeRecommendation == "strong") "strong" else "soft"
                return toolRecommendRetake(strength, "The scan quality looks shaky; a quick retake may be faster than correcting many cells.")
            }
            return null
        }

        // ---------------------------
        // Recovery + Turn lifecycle
        // ---------------------------
        val rec = runRecovery(sessionId)
        if (rec.decision == RecoveryController.Decision.NEW_SESSION_REQUIRED) {
            ConversationTelemetry.emit(mapOf("type" to "NEW_SESSION_REQUIRED", "session_id" to sessionId))
        }

        val created = lifecycle.createTurn(sessionId = sessionId, persona = persona)
        val committedUser = lifecycle.commitUser(
            sessionId = sessionId,
            turnId = created.turnId,
            persona = persona,
            text = userMessage
        )

        val built = buildTurnHistoryPrompt(sessionId)

        val summary = buildHumanSummary(gridState)
        val gridContext = buildGridContextV1(gridState)

        val stateHeaderBlock = stateHeader.trim().takeIf { it.isNotEmpty() }?.let {
            "STATE_HEADER (from Conductor):\n$it\n"
        }.orEmpty()

        val extraNotes = buildString {
            appendLine(stateHeaderBlock)
            appendLine("=== OUTPUT CONTRACT (GRID MODE) ===")
            appendLine("Return ONLY JSON with tool calls (no markdown, no extra text).")
            appendLine("HARD RULES:")
            appendLine("- MUST include reply(text=non_empty).")
            appendLine("- You MAY include multiple operational tools.")
            appendLine("- You MUST include exactly ONE control/progress tool in GRID_MODE:")
            appendLine("  ask_confirm_cell_rc OR ask_confirm_cell OR recommend_validate OR recommend_retake")
            appendLine("  OR (if needed) confirm_interpretation / ask_clarifying_question / switch_to_tap / propose_edit.")
            appendLine("- If you include apply_user_edit(_rc), you MUST ALSO include one control tool in the same response.")
            appendLine()
            appendLine("CONFIRMATION RULE (IMPORTANT):")
            appendLine("- If the user is answering a \"what digit is in row/col\" check, emit confirm_cell_value_rc for THAT asked cell.")
            appendLine("- Do NOT use apply_user_edit_rc just to record a confirmation; apply tools are for explicit edits/changes.")
            appendLine()
            appendLine("HUMAN_SUMMARY_FOR_DEBUG:")
            appendLine(summary.message)
        }.trim()

        val developerPrompt = DeveloperPromptComposer.composeForGridMode(
            gridContext = gridContext,
            captureOrigin = "",
            extraDeveloperNotes = extraNotes
        )

        lifecycle.markAssistantInflight(sessionId = sessionId, turnId = committedUser.turnId)

        // ✅ NEW: policy sequencing for audits
        val policyReqSeq = ConversationTelemetry.nextPolicyReqSeq(sessionId = sessionId, turnId = committedUser.turnId)
        ConversationTelemetry.emit(
            mapOf(
                "type" to "POLICY_REQ_BEGIN",
                "tick" to 1,
                "policy_req_seq" to policyReqSeq,
                "session_id" to sessionId,
                "turn_id" to committedUser.turnId
            )
        )

        ConversationTelemetry.emitLlmRequestDigest(
            mode = "GRID_MODE",
            model = null,
            convoSessionId = sessionId,
            turnId = committedUser.turnId,
            promptHash = built.promptHash,
            personaHash = built.personaHash,
            systemPrompt = systemPrompt,
            developerPrompt = developerPrompt,
            userMessage = userMessage
        )

        // ---------------------------
        // ✅ LLM call (history passed in)
        // ---------------------------
        val history = buildHistoryFromPromptBuilderMessages(
            builtMessages = built.messages,
            currentUserMessage = userMessage,
            maxTurns = 12
        )

        val raw = try {
            llmClient.sendGridUpdate(
                systemPrompt = systemPrompt,
                developerPrompt = developerPrompt,
                userMessage = userMessage,
                history = history
            )
        } catch (t: Throwable) {
            val errMsg = (t.message ?: t.javaClass.simpleName).take(240)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "LLM_GRID_UPDATE_THROW",
                    "session_id" to sessionId,
                    "turn_id" to committedUser.turnId,
                    "policy_req_seq" to policyReqSeq,
                    "error" to errMsg
                )
            )

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "POLICY_REQ_END",
                    "tick" to 1,
                    "policy_req_seq" to policyReqSeq,
                    "session_id" to sessionId,
                    "turn_id" to committedUser.turnId,
                    "ok" to false
                )
            )

            val fallbackText =
                "I hit a snag and couldn’t generate the next step. " +
                        "Please tell me ONE cell like “row 1 column 1 is 5” (or tap a cell and say the digit)."

            lifecycle.commitAssistant(
                sessionId = sessionId,
                turnId = committedUser.turnId,
                text = fallbackText
            )
            lifecycle.finalizeTurn(sessionId = sessionId, turnId = committedUser.turnId)
            return emptyList()
        }

        fun anyToInt(x: Any?): Int? = when (x) {
            is Number -> x.toInt()
            is String -> x.trim().toIntOrNull()
            else -> null
        }

        fun anyToFloat(x: Any?): Float? = when (x) {
            is Number -> x.toFloat()
            is String -> x.trim().toFloatOrNull()
            else -> null
        }

        fun anyToString(x: Any?): String? = x as? String

        val allowed = setOf(
            ToolNames.REPLY,
            ToolNames.ASK_CONFIRM_CELL,
            ToolNames.ASK_CONFIRM_CELL_RC,
            ToolNames.PROPOSE_EDIT,
            ToolNames.APPLY_USER_EDIT,
            ToolNames.APPLY_USER_EDIT_RC,
            ToolNames.CONFIRM_CELL_VALUE,
            ToolNames.CONFIRM_CELL_VALUE_RC,
            ToolNames.APPLY_USER_CLASSIFY,
            ToolNames.APPLY_USER_CLASSIFY_RC,
            ToolNames.SET_CANDIDATES,
            ToolNames.CLEAR_CANDIDATES,
            ToolNames.TOGGLE_CANDIDATE,
            ToolNames.RECOMMEND_RETAKE,
            ToolNames.RECOMMEND_VALIDATE,
            ToolNames.CONFIRM_INTERPRETATION,
            ToolNames.ASK_CLARIFYING_QUESTION,
            ToolNames.SWITCH_TO_TAP,
            ToolNames.NOOP
        )

        fun cleanToolCalls(rawResp: PolicyRawResponse): List<LlmToolCall> {
            val allowedNorm: Set<String> = allowed.map { it.trim().lowercase() }.toSet()
            return rawResp.tool_calls
                .asSequence()
                .map { tc -> ToolCallRaw(name = tc.name.trim().lowercase(), args = tc.args) }
                .filter { it.name in allowedNorm }
                .take(12)
                .map { LlmToolCall(it.name, it.args) }
                .toList()
        }

        var cleaned = cleanToolCalls(raw)

        var modelReplyText: String? = cleaned
            .firstOrNull { it.name == ToolNames.REPLY }
            ?.let { anyToString(it.args["text"]) }
            ?.trim()
            ?.takeIf { it.isNotEmpty() }

        val userSaysNoSuggestions = userSaysDontSuggestCorrections(userMessage)
        val refusesTap = userRefusesTap(userMessage)

        fun coerceModelToolOrNull(t: LlmToolCall?): LlmToolCall? {
            if (t == null) return null
            return when (t.name) {
                ToolNames.ASK_CONFIRM_CELL_RC -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (row == null || col == null || prompt.isNullOrBlank()) null
                    else toolAskConfirmCellRC(row, col, prompt)
                }

                ToolNames.ASK_CONFIRM_CELL -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (idx == null || idx !in 0..80 || prompt.isNullOrBlank()) null
                    else toolAskConfirmCell(idx, prompt)
                }

                ToolNames.CONFIRM_CELL_VALUE_RC -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 0..9 }
                    val source = anyToSourceOrDefault(t.args["source"], default = "user_voice")
                    if (row == null || col == null || digit == null) null
                    else toolConfirmCellValueRC(row, col, digit, source)
                }

                ToolNames.CONFIRM_CELL_VALUE -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 0..9 }
                    val source = anyToSourceOrDefault(t.args["source"], default = "user_voice")
                    if (idx == null || idx !in 0..80 || digit == null) null
                    else toolConfirmCellValue(idx, digit, source)
                }

                ToolNames.PROPOSE_EDIT -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])
                    val reason = anyToString(t.args["reason"])?.trim()
                    val conf = anyToFloat(t.args["confidence"])?.coerceIn(0f, 1f)
                    if (idx == null || idx !in 0..80) null
                    else if (digit == null || digit !in 1..9) null
                    else if (reason.isNullOrBlank()) null
                    else if (conf == null) null
                    else LlmToolCall(
                        name = ToolNames.PROPOSE_EDIT,
                        args = mapOf("cell_index" to idx, "digit" to digit, "reason" to reason, "confidence" to conf)
                    )
                }

                ToolNames.APPLY_USER_EDIT -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])
                    val source = anyToString(t.args["source"])?.trim()
                    if (idx == null || idx !in 0..80) null
                    else if (digit == null || digit !in 0..9) null
                    else if (source.isNullOrBlank()) null
                    else LlmToolCall(
                        name = ToolNames.APPLY_USER_EDIT,
                        args = mapOf("cell_index" to idx, "digit" to digit, "source" to source)
                    )
                }

                ToolNames.APPLY_USER_EDIT_RC -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 0..9 }
                    val source = anyToString(t.args["source"])?.trim()
                    if (row == null || col == null || digit == null || source.isNullOrBlank()) null
                    else LlmToolCall(
                        name = ToolNames.APPLY_USER_EDIT_RC,
                        args = mapOf("row" to row, "col" to col, "digit" to digit, "source" to source)
                    )
                }

                ToolNames.APPLY_USER_CLASSIFY -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val cls = anyToString(t.args["cellClass"] ?: t.args["cell_class"])?.trim()
                    val source = anyToString(t.args["source"])?.trim()
                    if (idx == null || idx !in 0..80 || cls.isNullOrBlank() || source.isNullOrBlank()) null
                    else LlmToolCall(
                        name = ToolNames.APPLY_USER_CLASSIFY,
                        args = mapOf("cell_index" to idx, "cellClass" to cls, "source" to source)
                    )
                }

                ToolNames.APPLY_USER_CLASSIFY_RC -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val cls = anyToString(t.args["cellClass"] ?: t.args["cell_class"])?.trim()
                    val source = anyToString(t.args["source"])?.trim()
                    if (row == null || col == null || cls.isNullOrBlank() || source.isNullOrBlank()) null
                    else LlmToolCall(
                        name = ToolNames.APPLY_USER_CLASSIFY_RC,
                        args = mapOf("row" to row, "col" to col, "cellClass" to cls, "source" to source)
                    )
                }

                ToolNames.SET_CANDIDATES -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val mask = anyToInt(t.args["mask"])
                    val source = anyToString(t.args["source"])?.trim()
                    if (idx == null || idx !in 0..80 || mask == null || mask !in 0..0x1FF || source.isNullOrBlank()) null
                    else LlmToolCall(name = ToolNames.SET_CANDIDATES, args = mapOf("cell_index" to idx, "mask" to mask, "source" to source))
                }

                ToolNames.CLEAR_CANDIDATES -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val source = anyToString(t.args["source"])?.trim()
                    if (idx == null || idx !in 0..80 || source.isNullOrBlank()) null
                    else LlmToolCall(name = ToolNames.CLEAR_CANDIDATES, args = mapOf("cell_index" to idx, "source" to source))
                }

                ToolNames.TOGGLE_CANDIDATE -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val d = anyToInt(t.args["digit"] ?: t.args["value"])
                    val source = anyToString(t.args["source"])?.trim()
                    if (idx == null || idx !in 0..80 || d == null || d !in 1..9 || source.isNullOrBlank()) null
                    else LlmToolCall(name = ToolNames.TOGGLE_CANDIDATE, args = mapOf("cell_index" to idx, "digit" to d, "source" to source))
                }

                ToolNames.RECOMMEND_RETAKE -> {
                    val strength = anyToString(t.args["strength"])?.trim()
                    val reason = anyToString(t.args["reason"])?.trim()
                    if (strength.isNullOrBlank() || reason.isNullOrBlank()) null
                    else toolRecommendRetake(strength, reason)
                }

                ToolNames.RECOMMEND_VALIDATE -> toolRecommendValidate()

                ToolNames.CONFIRM_INTERPRETATION -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 1..9 }
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    val conf = anyToFloat(t.args["confidence"])?.coerceIn(0f, 1f)
                    if (prompt.isNullOrBlank() || conf == null) null
                    else toolConfirmInterpretation(row, col, digit, prompt, conf)
                }

                ToolNames.ASK_CLARIFYING_QUESTION -> {
                    val kind = anyToString(t.args["kind"])?.trim()?.uppercase()
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (kind.isNullOrBlank() || prompt.isNullOrBlank()) null
                    else toolAskClarifyingQuestion(kind, prompt)
                }

                ToolNames.SWITCH_TO_TAP -> {
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (prompt.isNullOrBlank()) null
                    else toolSwitchToTap(prompt)
                }

                else -> null
            }
        }

        val coercedNonReply = cleaned
            .filter { it.name != ToolNames.REPLY && it.name != ToolNames.NOOP }
            .mapNotNull { coerceModelToolOrNull(it) }

        val coercedOps = coercedNonReply.filter { isOperationalToolName(it.name) }

        var controlCandidates = coercedNonReply.filter { isControlToolName(it.name) }

        if (userSaysNoSuggestions) {
            controlCandidates = controlCandidates.filterNot { it.name == ToolNames.PROPOSE_EDIT }
        }
        if (refusesTap) {
            controlCandidates = controlCandidates.filterNot { it.name == ToolNames.SWITCH_TO_TAP }
        }

        // solvability-first override
        val forceValidateBecauseUnique = (gridState.solvability == "unique")
        if (forceValidateBecauseUnique) {
            controlCandidates = controlCandidates.filter { it.name == ToolNames.RECOMMEND_VALIDATE }
            val t = (modelReplyText ?: "").lowercase()
            val alreadyAsksMatch = t.contains("match") && (t.contains("paper") || t.contains("book") || t.contains("screen"))
            if (!alreadyAsksMatch) {
                modelReplyText =
                    "Looks good — I’m seeing a uniquely solvable grid now. Does the on-screen grid match your paper exactly?"
            }
        }

        fun controlPriorityName(name: String): Int = when (name) {
            ToolNames.SWITCH_TO_TAP -> 0
            ToolNames.CONFIRM_INTERPRETATION -> 1
            ToolNames.ASK_CLARIFYING_QUESTION -> 2
            ToolNames.ASK_CONFIRM_CELL_RC -> 3
            ToolNames.ASK_CONFIRM_CELL -> 4
            ToolNames.PROPOSE_EDIT -> 5
            ToolNames.RECOMMEND_VALIDATE -> 6
            ToolNames.RECOMMEND_RETAKE -> 7
            else -> 50
        }

        var chosenControl: LlmToolCall? = controlCandidates.minByOrNull { controlPriorityName(it.name) }

        val needsControl = nextStepExists(gridState) || coercedOps.isNotEmpty()
        if (needsControl && chosenControl == null) {
            chosenControl = pickDeterministicControl(gridState)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "LLM_TOOLPLAN_CONTROL_APPENDED_DETERMINISTICALLY",
                    "session_id" to sessionId,
                    "turn_id" to committedUser.turnId,
                    "policy_req_seq" to policyReqSeq,
                    "ops_count" to coercedOps.size,
                    "solvability" to gridState.solvability,
                    "mismatch_count" to gridState.mismatchCells.size,
                    "unresolved_count" to gridState.unresolvedCells.size,
                    "retake" to gridState.retakeRecommendation,
                    "control" to (chosenControl?.name ?: "none")
                )
            )
        }

        if (modelReplyText.isNullOrBlank()) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "LLM_TOOLPLAN_MISSING_REPLY",
                    "session_id" to sessionId,
                    "turn_id" to committedUser.turnId,
                    "policy_req_seq" to policyReqSeq
                )
            )

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "POLICY_REQ_END",
                    "tick" to 1,
                    "policy_req_seq" to policyReqSeq,
                    "session_id" to sessionId,
                    "turn_id" to committedUser.turnId,
                    "ok" to true,
                    "missing_reply" to true
                )
            )

            lifecycle.commitAssistant(sessionId = sessionId, turnId = committedUser.turnId, text = "")
            lifecycle.finalizeTurn(sessionId = sessionId, turnId = committedUser.turnId)
            return emptyList()
        }

        var patchedReplyText = modelReplyText!!

        if (isToolPlanInternalErrorText(patchedReplyText)) {
            ConversationTelemetry.emit(mapOf("type" to "INTERNAL_TOOLPLAN_TEXT_STRIPPED", "turn_id" to committedUser.turnId, "policy_req_seq" to policyReqSeq))
            patchedReplyText = "Sorry — something glitched on my side. Could you repeat that last part?"
            chosenControl = toolAskClarifyingQuestion("POSITION", "I missed a detail — which row and column are you referring to?")
        }

        if (uniqueStopApplies(gridState) && chosenControl != null && toolIsExtraCheckName(chosenControl!!.name)) {
            chosenControl = toolRecommendValidate()
            val t = patchedReplyText.lowercase()
            if (!(t.contains("match") && (t.contains("paper") || t.contains("book") || t.contains("screen")))) {
                patchedReplyText =
                    "Looks good — I’m seeing a uniquely solvable grid now. Does the on-screen grid match your paper exactly?"
            }
            ConversationTelemetry.emit(mapOf("type" to "UNIQUE_STOP_GATE_APPLIED", "turn_id" to committedUser.turnId, "policy_req_seq" to policyReqSeq))
        }

        lifecycle.commitAssistant(sessionId = sessionId, turnId = committedUser.turnId, text = patchedReplyText)
        lifecycle.finalizeTurn(sessionId = sessionId, turnId = committedUser.turnId)

        ConversationTelemetry.emitLlmResponseDigest(
            mode = "GRID_MODE",
            model = null,
            convoSessionId = sessionId,
            turnId = committedUser.turnId,
            assistantText = patchedReplyText
        )

        val out = mutableListOf<LlmToolCall>()
        out += toolReply(patchedReplyText)
        out += coercedOps
        if (chosenControl != null && chosenControl!!.name != ToolNames.NOOP) out += chosenControl!!

        ConversationTelemetry.emit(
            mapOf(
                "type" to "LLM_TOOLPLAN_FINALIZED",
                "turn_id" to committedUser.turnId,
                "policy_req_seq" to policyReqSeq,
                "reply_len" to patchedReplyText.length,
                "ops" to coercedOps.map { it.name },
                "control" to (chosenControl?.name ?: "none"),
                "out_names" to out.map { it.name }
            )
        )

        ConversationTelemetry.emit(
            mapOf(
                "type" to "POLICY_REQ_END",
                "tick" to 1,
                "policy_req_seq" to policyReqSeq,
                "session_id" to sessionId,
                "turn_id" to committedUser.turnId,
                "ok" to true
            )
        )

        return out
    }




    suspend fun sendToLLMToolsContinuationTick2(
        sessionId: String,
        systemPrompt: String,
        gridStateAfterTools: LLMGridState,
        // LLM1 output (the ack text that the user heard)
        llm1ReplyText: String,
        // Tool results (serialize whatever you executed in Conductor)
        toolResults: List<String>,
        // Updated state header after tools (optional)
        stateHeader: String = "",
        // Optional; defaults to "Continue."
        continuationUserMessage: String = "Continue.",
        // ✅ NEW: for policy sequencing + audit correlation
        turnId: Long = -1L
    ): List<LlmToolCall> {

        fun toolReply(text: String) = LlmToolCall(
            name = ToolNames.REPLY,
            args = mapOf("text" to text)
        )

        fun toolAskConfirmCell(cellIndex: Int, prompt: String) = LlmToolCall(
            name = ToolNames.ASK_CONFIRM_CELL,
            args = mapOf("cell_index" to cellIndex, "prompt" to prompt)
        )

        fun toolAskConfirmCellRC(row: Int, col: Int, prompt: String) = LlmToolCall(
            name = ToolNames.ASK_CONFIRM_CELL_RC,
            args = mapOf("row" to row, "col" to col, "prompt" to prompt)
        )

        fun toolRecommendRetake(strength: String, reason: String) = LlmToolCall(
            name = ToolNames.RECOMMEND_RETAKE,
            args = mapOf("strength" to strength, "reason" to reason)
        )

        fun toolRecommendValidate() = LlmToolCall(
            name = ToolNames.RECOMMEND_VALIDATE,
            args = emptyMap()
        )

        fun toolAskClarifyingQuestion(kind: String, prompt: String) = LlmToolCall(
            name = ToolNames.ASK_CLARIFYING_QUESTION,
            args = mapOf("kind" to kind, "prompt" to prompt)
        )

        fun toolSwitchToTap(prompt: String) = LlmToolCall(
            name = ToolNames.SWITCH_TO_TAP,
            args = mapOf("prompt" to prompt)
        )

        fun toolConfirmInterpretation(
            row: Int?,
            col: Int?,
            digit: Int?,
            prompt: String,
            confidence: Float
        ) = LlmToolCall(
            name = ToolNames.CONFIRM_INTERPRETATION,
            args = mapOf(
                "row" to row,
                "col" to col,
                "digit" to digit,
                "prompt" to prompt,
                "confidence" to confidence
            )
        )

        fun toolConfirmCellValue(cellIndex: Int, digit: Int, source: String) = LlmToolCall(
            name = ToolNames.CONFIRM_CELL_VALUE,
            args = mapOf("cell_index" to cellIndex, "digit" to digit, "source" to source)
        )

        fun toolConfirmCellValueRC(row: Int, col: Int, digit: Int, source: String) = LlmToolCall(
            name = ToolNames.CONFIRM_CELL_VALUE_RC,
            args = mapOf("row" to row, "col" to col, "digit" to digit, "source" to source)
        )

        fun nextStepExists(g: LLMGridState): Boolean {
            if (g.mismatchCells.isNotEmpty()) return true
            if (g.solvability == "none" && g.unresolvedCells.isNotEmpty()) return true
            if (g.solvability == "unique") return true
            if (g.solvability == "multiple") return true
            if (g.retakeRecommendation != "none") return true
            return false
        }

        fun isOperationalToolName(name: String): Boolean = when (name) {
            ToolNames.APPLY_USER_EDIT,
            ToolNames.APPLY_USER_EDIT_RC,
            ToolNames.CONFIRM_CELL_VALUE,
            ToolNames.CONFIRM_CELL_VALUE_RC,
            ToolNames.APPLY_USER_CLASSIFY,
            ToolNames.APPLY_USER_CLASSIFY_RC,
            ToolNames.SET_CANDIDATES,
            ToolNames.CLEAR_CANDIDATES,
            ToolNames.TOGGLE_CANDIDATE -> true
            else -> false
        }

        fun isControlToolName(name: String): Boolean = when (name) {
            ToolNames.ASK_CONFIRM_CELL,
            ToolNames.ASK_CONFIRM_CELL_RC,
            ToolNames.PROPOSE_EDIT,
            ToolNames.RECOMMEND_RETAKE,
            ToolNames.RECOMMEND_VALIDATE,
            ToolNames.CONFIRM_INTERPRETATION,
            ToolNames.ASK_CLARIFYING_QUESTION,
            ToolNames.SWITCH_TO_TAP -> true
            else -> false
        }

        fun pickDeterministicControl(g: LLMGridState): LlmToolCall? {
            if (g.mismatchCells.isNotEmpty()) {
                val idx = g.mismatchCells.first()
                val r = idx / 9 + 1
                val c = idx % 9 + 1
                return toolAskConfirmCellRC(r, c, "Quick check: what digit is in row $r, column $c on your paper?")
            }
            if (g.solvability == "none" && g.unresolvedCells.isNotEmpty()) {
                val idx = g.unresolvedCells.sorted().first()
                val r = idx / 9 + 1
                val c = idx % 9 + 1
                return toolAskConfirmCellRC(r, c, "Quick check: what digit is in row $r, column $c on your paper (or is it blank)?")
            }
            if (g.solvability == "unique" || g.solvability == "multiple") return toolRecommendValidate()
            if (g.retakeRecommendation != "none") {
                val strength = if (g.retakeRecommendation == "strong") "strong" else "soft"
                return toolRecommendRetake(strength, "The scan quality looks shaky; a quick retake may be faster than correcting many cells.")
            }
            return null
        }

        fun uniqueStopApplies(g: LLMGridState): Boolean =
            (g.solvability == "unique" && g.mismatchCells.isEmpty())

        fun toolIsExtraCheckName(name: String): Boolean {
            return when (name) {
                ToolNames.ASK_CONFIRM_CELL,
                ToolNames.ASK_CONFIRM_CELL_RC,
                ToolNames.PROPOSE_EDIT,
                ToolNames.RECOMMEND_RETAKE,
                ToolNames.SWITCH_TO_TAP -> true
                else -> false
            }
        }

        fun anyToInt(x: Any?): Int? = when (x) {
            is Number -> x.toInt()
            is String -> x.trim().toIntOrNull()
            else -> null
        }

        fun anyToFloat(x: Any?): Float? = when (x) {
            is Number -> x.toFloat()
            is String -> x.trim().toFloatOrNull()
            else -> null
        }

        fun anyToString(x: Any?): String? = x as? String

        // Build prompt history from store
        val built = buildTurnHistoryPrompt(sessionId)

        // Base history (user/assistant only, no duplication)
        val baseHistory = buildHistoryFromPromptBuilderMessages(
            builtMessages = built.messages,
            currentUserMessage = continuationUserMessage,
            maxTurns = 12
        )

        // ✅ Augment history so LLM2 sees: (LLM1 reply) + (tool results)
        fun cap(s: String, n: Int) = s.replace("\r\n", "\n").trimEnd().let { if (it.length <= n) it else it.take(n).trimEnd() + "…" }

        val augmentedHistory = buildList<Pair<String, String>> {
            addAll(baseHistory)

            val ack = llm1ReplyText.trim()
            if (ack.isNotEmpty()) add("assistant" to cap(ack, 900))

            for (tr in toolResults) {
                val line = tr.trim()
                if (line.isNotEmpty()) add("assistant" to cap("[TOOL_RESULT] $line", 900))
            }
        }

        val summary = buildHumanSummary(gridStateAfterTools)
        val gridContext = buildGridContextV1(gridStateAfterTools)

        val stateHeaderBlock = stateHeader.trim().takeIf { it.isNotEmpty() }?.let {
            "STATE_HEADER (from Conductor):\n$it\n"
        }.orEmpty()

        val extraNotes = buildString {
            appendLine(stateHeaderBlock)
            appendLine("=== CONTINUATION TICK (TICK #2) ===")
            appendLine("You are continuing the same assistant turn after tools were executed.")
            appendLine("Do NOT restate the applied edit/confirmation. Proceed to the next best action.")
            appendLine()
            appendLine("=== OUTPUT CONTRACT (GRID MODE) ===")
            appendLine("Return ONLY JSON with tool calls (no markdown, no extra text).")
            appendLine("- MUST include reply(text=non_empty).")
            appendLine("- You MUST include exactly ONE control/progress tool in GRID_MODE.")
            appendLine()
            appendLine("HUMAN_SUMMARY_FOR_DEBUG:")
            appendLine(summary.message)
        }.trim()

        val developerPrompt = DeveloperPromptComposer.composeForGridMode(
            gridContext = gridContext,
            captureOrigin = "",
            extraDeveloperNotes = extraNotes
        )

        // ✅ NEW: policy sequencing for audits (tick2)
        val seqTurnId = if (turnId > 0L) turnId else 0L
        val policyReqSeq = ConversationTelemetry.nextPolicyReqSeq(sessionId = sessionId, turnId = seqTurnId)
        ConversationTelemetry.emit(
            mapOf(
                "type" to "POLICY_REQ_BEGIN",
                "tick" to 2,
                "policy_req_seq" to policyReqSeq,
                "session_id" to sessionId,
                "turn_id" to seqTurnId
            )
        )

        val raw = try {
            llmClient.sendGridUpdate(
                systemPrompt = systemPrompt,
                developerPrompt = developerPrompt,
                userMessage = continuationUserMessage,
                history = augmentedHistory
            )
        } catch (t: Throwable) {
            val errMsg = (t.message ?: t.javaClass.simpleName).take(240)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "LLM_GRID_UPDATE_TICK2_THROW",
                    "session_id" to sessionId,
                    "turn_id" to seqTurnId,
                    "policy_req_seq" to policyReqSeq,
                    "error" to errMsg
                )
            )
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "POLICY_REQ_END",
                    "tick" to 2,
                    "policy_req_seq" to policyReqSeq,
                    "session_id" to sessionId,
                    "turn_id" to seqTurnId,
                    "ok" to false
                )
            )
            return listOf(
                toolReply("I hit a snag continuing. Tell me ONE cell like “row 1 column 1 is 5”."),
                toolAskClarifyingQuestion("POSITION", "Which row and column are you referring to?")
            )
        }

        val allowed = setOf(
            ToolNames.REPLY,
            ToolNames.ASK_CONFIRM_CELL,
            ToolNames.ASK_CONFIRM_CELL_RC,
            ToolNames.PROPOSE_EDIT,
            ToolNames.APPLY_USER_EDIT,
            ToolNames.APPLY_USER_EDIT_RC,
            ToolNames.CONFIRM_CELL_VALUE,
            ToolNames.CONFIRM_CELL_VALUE_RC,
            ToolNames.APPLY_USER_CLASSIFY,
            ToolNames.APPLY_USER_CLASSIFY_RC,
            ToolNames.SET_CANDIDATES,
            ToolNames.CLEAR_CANDIDATES,
            ToolNames.TOGGLE_CANDIDATE,
            ToolNames.RECOMMEND_RETAKE,
            ToolNames.RECOMMEND_VALIDATE,
            ToolNames.CONFIRM_INTERPRETATION,
            ToolNames.ASK_CLARIFYING_QUESTION,
            ToolNames.SWITCH_TO_TAP,
            ToolNames.NOOP
        ).map { it.lowercase().trim() }.toSet()

        val cleaned = raw.tool_calls
            .asSequence()
            .map { ToolCallRaw(it.name.trim().lowercase(), it.args) }
            .filter { it.name in allowed }
            .take(12)
            .map { LlmToolCall(it.name, it.args) }
            .toList()

        var modelReplyText: String? = cleaned
            .firstOrNull { it.name == ToolNames.REPLY }
            ?.let { anyToString(it.args["text"]) }
            ?.trim()
            ?.takeIf { it.isNotEmpty() }

        fun coerceModelToolOrNull(t: LlmToolCall?): LlmToolCall? {
            if (t == null) return null
            return when (t.name) {
                ToolNames.ASK_CONFIRM_CELL_RC -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (row == null || col == null || prompt.isNullOrBlank()) null
                    else toolAskConfirmCellRC(row, col, prompt)
                }

                ToolNames.ASK_CONFIRM_CELL -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (idx == null || idx !in 0..80 || prompt.isNullOrBlank()) null
                    else toolAskConfirmCell(idx, prompt)
                }

                ToolNames.CONFIRM_CELL_VALUE_RC -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 0..9 }
                    val source = anyToSourceOrDefault(t.args["source"], default = "user_voice")
                    if (row == null || col == null || digit == null) null
                    else toolConfirmCellValueRC(row, col, digit, source)
                }

                ToolNames.CONFIRM_CELL_VALUE -> {
                    val idx = anyToInt(t.args["cell_index"] ?: t.args["cellIndex"])
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 0..9 }
                    val source = anyToSourceOrDefault(t.args["source"], default = "user_voice")
                    if (idx == null || idx !in 0..80 || digit == null) null
                    else toolConfirmCellValue(idx, digit, source)
                }

                ToolNames.RECOMMEND_RETAKE -> {
                    val strength = anyToString(t.args["strength"])?.trim()
                    val reason = anyToString(t.args["reason"])?.trim()
                    if (strength.isNullOrBlank() || reason.isNullOrBlank()) null
                    else toolRecommendRetake(strength, reason)
                }

                ToolNames.RECOMMEND_VALIDATE -> toolRecommendValidate()

                ToolNames.CONFIRM_INTERPRETATION -> {
                    val row = anyToInt(t.args["row"])?.takeIf { it in 1..9 }
                    val col = anyToInt(t.args["col"] ?: t.args["column"])?.takeIf { it in 1..9 }
                    val digit = anyToInt(t.args["digit"] ?: t.args["value"])?.takeIf { it in 1..9 }
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    val conf = anyToFloat(t.args["confidence"])?.coerceIn(0f, 1f)
                    if (prompt.isNullOrBlank() || conf == null) null
                    else toolConfirmInterpretation(row, col, digit, prompt, conf)
                }

                ToolNames.ASK_CLARIFYING_QUESTION -> {
                    val kind = anyToString(t.args["kind"])?.trim()?.uppercase()
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (kind.isNullOrBlank() || prompt.isNullOrBlank()) null
                    else toolAskClarifyingQuestion(kind, prompt)
                }

                ToolNames.SWITCH_TO_TAP -> {
                    val prompt = anyToString(t.args["prompt"])?.trim()
                    if (prompt.isNullOrBlank()) null
                    else toolSwitchToTap(prompt)
                }

                // Allow ops to pass through (they were already schema-validated upstream)
                ToolNames.APPLY_USER_EDIT,
                ToolNames.APPLY_USER_EDIT_RC,
                ToolNames.APPLY_USER_CLASSIFY,
                ToolNames.APPLY_USER_CLASSIFY_RC,
                ToolNames.SET_CANDIDATES,
                ToolNames.CLEAR_CANDIDATES,
                ToolNames.TOGGLE_CANDIDATE,
                ToolNames.PROPOSE_EDIT -> LlmToolCall(t.name, t.args)

                else -> null
            }
        }

        val coercedNonReply = cleaned
            .filter { it.name != ToolNames.REPLY && it.name != ToolNames.NOOP }
            .mapNotNull { coerceModelToolOrNull(it) }

        val coercedOps = coercedNonReply.filter { isOperationalToolName(it.name) }
        val controlCandidates = coercedNonReply.filter { isControlToolName(it.name) }

        fun controlPriorityName(name: String): Int = when (name) {
            ToolNames.SWITCH_TO_TAP -> 0
            ToolNames.CONFIRM_INTERPRETATION -> 1
            ToolNames.ASK_CLARIFYING_QUESTION -> 2
            ToolNames.ASK_CONFIRM_CELL_RC -> 3
            ToolNames.ASK_CONFIRM_CELL -> 4
            ToolNames.PROPOSE_EDIT -> 5
            ToolNames.RECOMMEND_VALIDATE -> 6
            ToolNames.RECOMMEND_RETAKE -> 7
            else -> 50
        }

        var chosenControl: LlmToolCall? = controlCandidates.minByOrNull { controlPriorityName(it.name) }

        val needsControl = nextStepExists(gridStateAfterTools) || coercedOps.isNotEmpty()
        if (needsControl && chosenControl == null) {
            chosenControl = pickDeterministicControl(gridStateAfterTools)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "LLM_TOOLPLAN_TICK2_CONTROL_APPENDED_DETERMINISTICALLY",
                    "session_id" to sessionId,
                    "turn_id" to seqTurnId,
                    "policy_req_seq" to policyReqSeq,
                    "ops_count" to coercedOps.size,
                    "solvability" to gridStateAfterTools.solvability,
                    "mismatch_count" to gridStateAfterTools.mismatchCells.size,
                    "unresolved_count" to gridStateAfterTools.unresolvedCells.size,
                    "retake" to gridStateAfterTools.retakeRecommendation,
                    "control" to (chosenControl?.name ?: "none")
                )
            )
        }

        if (modelReplyText.isNullOrBlank()) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "LLM_TOOLPLAN_TICK2_MISSING_REPLY",
                    "session_id" to sessionId,
                    "turn_id" to seqTurnId,
                    "policy_req_seq" to policyReqSeq
                )
            )
            modelReplyText = "Okay — let’s keep going. Tell me ONE cell like “row 1 column 1 is 5”."
            chosenControl = chosenControl ?: toolAskClarifyingQuestion("POSITION", "Which row and column are you referring to?")
        }

        var patchedReplyText = modelReplyText!!

        if (uniqueStopApplies(gridStateAfterTools) && chosenControl != null && toolIsExtraCheckName(chosenControl!!.name)) {
            chosenControl = toolRecommendValidate()
            val t = patchedReplyText.lowercase()
            if (!(t.contains("match") && (t.contains("paper") || t.contains("book") || t.contains("screen")))) {
                patchedReplyText =
                    "Looks good — I’m seeing a uniquely solvable grid now. Does the on-screen grid match your paper exactly?"
            }
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "UNIQUE_STOP_GATE_APPLIED_TICK2",
                    "session_id" to sessionId,
                    "turn_id" to seqTurnId,
                    "policy_req_seq" to policyReqSeq
                )
            )
        }

        val out = mutableListOf<LlmToolCall>()
        out += toolReply(patchedReplyText)
        out += coercedOps
        if (chosenControl != null && chosenControl!!.name != ToolNames.NOOP) out += chosenControl!!

        ConversationTelemetry.emit(
            mapOf(
                "type" to "LLM_TOOLPLAN_TICK2_FINALIZED",
                "session_id" to sessionId,
                "turn_id" to seqTurnId,
                "policy_req_seq" to policyReqSeq,
                "reply_len" to patchedReplyText.length,
                "ops" to coercedOps.map { it.name },
                "control" to (chosenControl?.name ?: "none"),
                "out_names" to out.map { it.name }
            )
        )

        ConversationTelemetry.emit(
            mapOf(
                "type" to "POLICY_REQ_END",
                "tick" to 2,
                "policy_req_seq" to policyReqSeq,
                "session_id" to sessionId,
                "turn_id" to seqTurnId,
                "ok" to true
            )
        )

        return out
    }

    // -------------------------
    // FREE TALK (unchanged)
    // -------------------------

    private fun buildFreeTalkSystemPrompt(profile: PlayerProfileSnapshot): String {
        val name = profile.name ?: "friend"
        val locale = profile.locale ?: "en"
        return """
            You are Sudo, a warm, encouraging Sudoku companion.
            Speak concisely, friendly, and curious. Avoid over-explaining unless asked.
            Player name: $name. Locale: $locale.
        """.trimIndent()
    }



    private fun buildHistoryFromPromptBuilderMessages(
        builtMessages: List<PromptBuilder.PromptMessage>,
        currentUserMessage: String,
        maxTurns: Int
    ): List<Pair<String, String>> {
        // builtMessages typically: [system, developer, ...history..., currentUser]
        val histMsgs = builtMessages.drop(2)

        // Remove the last USER msg if it is the current user message (avoid duplication)
        fun norm(s: String) = s.trim().replace(Regex("\\s+"), " ")
        val trimmed = histMsgs.toMutableList()
        if (trimmed.isNotEmpty()) {
            val last = trimmed.last()
            val isUser = last.role.name.equals("USER", ignoreCase = true)
            if (isUser && norm(last.content) == norm(currentUserMessage)) {
                trimmed.removeAt(trimmed.size - 1)
            }
        }

        // Convert to (role,text) pairs
        val pairs = trimmed.mapNotNull { m ->
            val r = m.role.name.lowercase()
            if (r != "user" && r != "assistant") null
            else {
                val c = m.content.replace("\r\n", "\n").trimEnd()
                if (c.isBlank()) null else (r to c)
            }
        }

        // Truncate by turns: keep last N turns where a “turn” is USER + ASSISTANT (up to 2 msgs)
        if (pairs.isEmpty()) return emptyList()
        val maxMsgs = maxTurns * 2
        return if (pairs.size > maxMsgs) pairs.takeLast(maxMsgs) else pairs
    }

    private fun buildFreeTalkDeveloperPrompt(
        profile: PlayerProfileSnapshot,
        recentTurns: List<ChatTurn>,
        historyPrompt: String
    ): String {
        val turns = recentTurns.takeLast(6).joinToString("\n") { t -> "${t.role}: ${t.text}" }
        val fav = profile.favoriteDifficulty ?: "(none)"
        val interests = if (profile.interests.isEmpty()) "(none)" else profile.interests.joinToString(", ")

        val prompt = """
        === TURN HISTORY (deterministic) ===
        $historyPrompt

        === PROFILE CONTEXT ===
        - Player: ${profile.name ?: "friend"}
        - Favorite difficulty: $fav
        - Interests: $interests

        === RECENT TURNS (ephemeral, optional) ===
        $turns

        INSTRUCTIONS
        - Reply naturally as Sudo (friendly, short, and curious).
        - If the player shares personal preferences or Sudoku habits, acknowledge them.
        - Ask one small follow-up question when appropriate.
        - Do NOT output JSON; just the assistant message.
        """.trimIndent()

        ConversationTelemetry.emit(mapOf("type" to "DEV_PROMPT_FREE_TALK", "chars" to prompt.length))
        return prompt
    }

    suspend fun freeTalk(
        sessionId: String,
        profile: PlayerProfileSnapshot,
        recentTurns: List<ChatTurn>,
        userMessage: String
    ): String {
        runRecovery(sessionId)

        val created = lifecycle.createTurn(sessionId = sessionId, persona = persona)
        val committedUser = lifecycle.commitUser(
            sessionId = sessionId,
            turnId = created.turnId,
            persona = persona,
            text = userMessage
        )

        val built = buildTurnHistoryPrompt(sessionId)
        lifecycle.markAssistantInflight(sessionId = sessionId, turnId = committedUser.turnId)

        val sys = buildFreeTalkSystemPrompt(profile)

        val histMsgs = built.messages.drop(1) // drop system

        fun isAssistant(m: PromptBuilder.PromptMessage): Boolean =
            m.role.name.equals("ASSISTANT", ignoreCase = true)

        fun isUser(m: PromptBuilder.PromptMessage): Boolean =
            m.role.name.equals("USER", ignoreCase = true)

// Find last and second-last assistant messages
        val lastAssistantIdx = histMsgs.indexOfLast { isAssistant(it) }
        val secondLastAssistantIdx =
            if (lastAssistantIdx > 0) {
                histMsgs.subList(0, lastAssistantIdx).indexOfLast { isAssistant(it) }
            } else {
                -1
            }

// Keep from second-last assistant if possible; else keep from last assistant; else no assistant in history
        val keepFrom = when {
            secondLastAssistantIdx >= 0 -> secondLastAssistantIdx
            lastAssistantIdx >= 0 -> lastAssistantIdx
            else -> histMsgs.size
        }

// Head: keep only USER messages before keepFrom, but cap to avoid prompt bloat
        val HEAD_USER_LIMIT = 8
        val headUsersAll = histMsgs.take(keepFrom).filter { isUser(it) }
        val head = if (headUsersAll.size > HEAD_USER_LIMIT) headUsersAll.takeLast(HEAD_USER_LIMIT) else headUsersAll

// Tail: keep the conversation intact from keepFrom onward
        val tail = if (keepFrom < histMsgs.size) histMsgs.drop(keepFrom) else emptyList()

        val selected = head + tail

        val canonicalHistoryBody = selected.joinToString("\n") { m ->
            "${m.role.name.lowercase()}: ${m.content.replace("\r\n", "\n").trimEnd()}"
        }

        val historyText = buildString {
            appendLine("BEGIN_CANONICAL_HISTORY")
            appendLine(canonicalHistoryBody)
            appendLine("END_CANONICAL_HISTORY")
        }

        val dev = buildFreeTalkDeveloperPrompt(profile, recentTurns, historyText)

        ConversationTelemetry.emitLlmRequestDigest(
            mode = "FREE_TALK",
            model = null,
            convoSessionId = sessionId,
            turnId = committedUser.turnId,
            promptHash = built.promptHash,
            personaHash = built.personaHash,
            systemPrompt = sys,
            developerPrompt = dev,
            userMessage = userMessage
        )

        val resp = llmClient.chatFreeTalk(sys, dev, userMessage)

        lifecycle.commitAssistant(sessionId = sessionId, turnId = committedUser.turnId, text = resp.assistant_message)
        lifecycle.finalizeTurn(sessionId = sessionId, turnId = committedUser.turnId)

        ConversationTelemetry.emit(
            mapOf(
                "type" to "PROMPT_HASH_USED",
                "session_id" to sessionId,
                "turn_id" to committedUser.turnId,
                "persona_hash" to built.personaHash,
                "prompt_hash" to built.promptHash
            )
        )

        ConversationTelemetry.emitLlmResponseDigest(
            mode = "FREE_TALK",
            model = null,
            convoSessionId = sessionId,
            turnId = committedUser.turnId,
            assistantText = resp.assistant_message
        )

        return resp.assistant_message
    }

    suspend fun freeTalk(
        @Suppress("UNUSED_PARAMETER") systemPrompt: String,
        profile: UserProfile,
        userMessage: String
    ): String {
        val snap = profile.toSnapshot()
        return freeTalk(DEFAULT_SESSION_ID, snap, emptyList(), userMessage)
    }

    private fun buildClueExtractionSystemPrompt(): String =
        """
        You extract compact, privacy-conscious "clues" about a player from short text.
        Each clue is (key, value, confidence). Be conservative; only include clear facts.
        """.trimIndent()

    private fun buildClueExtractionDeveloperPrompt(): String =
        """
        TASK
        - Read the short transcript (user + assistant).
        - Return 0..5 compact clues about the user.
        - Prefer stable facts that help future replies (e.g., preferred_name/nickname,
          preferred difficulty, language, typical play times, city-level locale, etc.)
        - Avoid sensitive attributes (health, politics, religion) and precise addresses.
        - Use concise keys like: preferred_name, nickname, favorite_difficulty, locale_city, prefers_voice, etc.

        OUTPUT
        - Return a JSON array of clues with keys: key, value, confidence (0..1), source="conversation".
        """.trimIndent()

    suspend fun extractCluesFromExchange(transcript: String): List<UserClue> {
        ConversationTelemetry.emit(mapOf("type" to "CLUES_BEGIN", "transcript_chars" to transcript.length))
        val sys = buildClueExtractionSystemPrompt()
        val dev = buildClueExtractionDeveloperPrompt()
        val resp = llmClient.extractClues(sys, dev, transcript)
        ConversationTelemetry.emit(mapOf("type" to "CLUES_OK", "count" to resp.clues.size))
        return resp.clues
    }

    suspend fun extractProfileClues(@Suppress("UNUSED_PARAMETER") userText: String): UserProfile {
        throw UnsupportedOperationException(
            "extractProfileClues: Unknown UserProfile shape. " +
                    "Implement in UserProfileStore or share the data class so I can return a proper delta."
        )
    }

    private fun runRecovery(sessionId: String): RecoveryController.Result {
        val r = recovery.recover(sessionId = sessionId, persona = persona)

        ConversationTelemetry.emit(
            mapOf(
                "type" to "RECOVERY_RESULT",
                "session_id" to sessionId,
                "decision" to r.decision.name,
                "affected_turn_id" to (r.affectedTurnId ?: -1L),
                "note" to (r.note ?: "")
            )
        )
        return r
    }

    private fun buildTurnHistoryPrompt(sessionId: String): PromptBuilder.BuildResult {
        return promptBuilder.build(
            sessionId = sessionId,
            persona = persona,
            systemPrompt = personaSystemPrompt,
            maxTurns = 16
        )
    }

    fun anyToSourceOrDefault(x: Any?, default: String): String =
        (x as? String)?.trim().takeUnless { it.isNullOrBlank() } ?: default

    private fun formatGridRows81(digits: IntArray): String {
        require(digits.size == 81) { "correctedGrid must be 81 digits" }
        return buildString {
            for (r in 0 until 9) {
                append("r${r + 1}: ")
                for (c in 0 until 9) {
                    val d = digits[r * 9 + c]
                    append(d.coerceIn(0, 9))
                    if (c != 8) append(' ')
                }
                append('\n')
            }
        }.trimEnd()
    }


    // ----------------------------------------------------------------
    // buildGridContextV1(...) unchanged (your existing v4.1 block)
    // ----------------------------------------------------------------
    private fun buildGridContextV1(gridState: LLMGridState): String {
        val digits = gridState.correctedGrid
        val rows = formatGridRows81(digits)

        fun maskToList(mask: Int): List<Int> {
            if (mask == 0) return emptyList()
            val out = mutableListOf<Int>()
            for (d in 1..9) if ((mask and (1 shl (d - 1))) != 0) out += d
            return out
        }

        fun countBits(mask: Int): Int = Integer.bitCount(mask and 0x1FF)
        fun rc(idx: Int): String = "r${idx / 9 + 1}c${idx % 9 + 1}"

        // ✅ NEW: confirmation facts from LLMGridState (not UI globals)
        val confirmed = gridState.confirmedCells.distinct().sorted()
        fun isConfirmed(idx: Int): Boolean = confirmed.contains(idx)

        val givenCount = (0 until 81).count { gridState.truthIsGiven[it] }
        val solCount = (0 until 81).count { gridState.truthIsSolution[it] }
        val totalCandidateMarks = (0 until 81).sumOf { countBits(gridState.candidateMask81[it]) }

        val candByRow = IntArray(9)
        val candByCol = IntArray(9)
        for (idx in 0 until 81) {
            val r = idx / 9
            val c = idx % 9
            val n = countBits(gridState.candidateMask81[idx])
            candByRow[r] += n
            candByCol[c] += n
        }

        // Truth grids (0 where not in group)
        val givensGrid = IntArray(81) { idx -> if (gridState.truthIsGiven[idx]) digits[idx] else 0 }
        val userSolGrid = IntArray(81) { idx -> if (gridState.truthIsSolution[idx]) digits[idx] else 0 }
        val givensRows = formatGridRows81(givensGrid)
        val userSolRows = formatGridRows81(userSolGrid)

        val deducedRows = gridState.deducedSolutionGrid?.let { formatGridRows81(it) }

        // Candidate detail lines
        val candLines = buildString {
            var any = false
            for (idx in 0 until 81) {
                val mask = gridState.candidateMask81[idx]
                if (mask == 0) continue
                any = true
                appendLine("- ${rc(idx)} idx=$idx: ${maskToList(mask)}")
            }
            if (!any) appendLine("- (none)")
        }.trimEnd()

        // ------------------------------------------------------------
        // ✅ CONFLICTS_DETAILS (authoritative; computed from CURRENT_DISPLAY)
        // ------------------------------------------------------------
        data class HouseConflict(val houseType: String, val houseId: Int, val digit: Int, val indices: List<Int>)

        fun computeConflictsFromDisplay(): List<HouseConflict> {
            val out = mutableListOf<HouseConflict>()

            // rows
            for (r in 0 until 9) {
                val byDigit = mutableMapOf<Int, MutableList<Int>>()
                for (c in 0 until 9) {
                    val idx = r * 9 + c
                    val d = digits[idx].coerceIn(0, 9)
                    if (d == 0) continue
                    byDigit.getOrPut(d) { mutableListOf() }.add(idx)
                }
                for ((d, idxs) in byDigit) if (idxs.size >= 2) out += HouseConflict("row", r + 1, d, idxs)
            }

            // cols
            for (c in 0 until 9) {
                val byDigit = mutableMapOf<Int, MutableList<Int>>()
                for (r in 0 until 9) {
                    val idx = r * 9 + c
                    val d = digits[idx].coerceIn(0, 9)
                    if (d == 0) continue
                    byDigit.getOrPut(d) { mutableListOf() }.add(idx)
                }
                for ((d, idxs) in byDigit) if (idxs.size >= 2) out += HouseConflict("col", c + 1, d, idxs)
            }

            // boxes
            for (br in 0 until 3) for (bc in 0 until 3) {
                val boxId = br * 3 + bc + 1
                val byDigit = mutableMapOf<Int, MutableList<Int>>()
                for (rr in 0 until 3) for (cc in 0 until 3) {
                    val r = br * 3 + rr
                    val c = bc * 3 + cc
                    val idx = r * 9 + c
                    val d = digits[idx].coerceIn(0, 9)
                    if (d == 0) continue
                    byDigit.getOrPut(d) { mutableListOf() }.add(idx)
                }
                for ((d, idxs) in byDigit) if (idxs.size >= 2) out += HouseConflict("box", boxId, d, idxs)
            }

            return out
        }

        val conflictsDisplay = computeConflictsFromDisplay()

        // conflict participation count per cell
        val conflictCountByIdx = IntArray(81)
        for (hc in conflictsDisplay) {
            for (idx in hc.indices) {
                if (idx in 0..80) conflictCountByIdx[idx] += 1
            }
        }

        fun conflictReasonForIdx(idx: Int): String? {
            val hc = conflictsDisplay.firstOrNull { it.indices.contains(idx) } ?: return null
            val where = when (hc.houseType) {
                "row" -> "row r${hc.houseId}"
                "col" -> "col c${hc.houseId}"
                "box" -> "box b${hc.houseId}"
                else -> "${hc.houseType} ${hc.houseId}"
            }
            return "$where contains duplicate ${hc.digit}"
        }

        val conflictDetailLines = buildString {
            if (conflictsDisplay.isEmpty()) {
                appendLine("- (none)")
            } else {
                for (hc in conflictsDisplay) {
                    val where = when (hc.houseType) {
                        "row" -> "row r${hc.houseId}"
                        "col" -> "col c${hc.houseId}"
                        "box" -> "box b${hc.houseId}"
                        else -> "${hc.houseType} ${hc.houseId}"
                    }
                    val cells = hc.indices.joinToString(", ") { "${rc(it)}(idx=$it)" }
                    appendLine("- $where has duplicate digit ${hc.digit} at $cells")
                }
            }
        }.trimEnd()

        // ------------------------------------------------------------
        // ✅ RECOMMENDED_NEXT_CHECK — STRICT 4-case policy (confirmation-aware)
        // ------------------------------------------------------------
        data class NextCheck(val idx: Int, val priority: String, val reason: String)

        fun findMismatchDetailFor(idx: Int): String? {
            val target = rc(idx)
            return gridState.mismatchDetails.firstOrNull { it.startsWith(target) }
        }

        fun pickRecommendedNextCheck(): NextCheck? {
            // Case 1 — mismatch (highest priority), but NEVER re-ask confirmed cells
            val mismatch = gridState.mismatchCells.firstOrNull { !isConfirmed(it) }
            if (mismatch != null) {
                val detail = findMismatchDetailFor(mismatch)
                val reason = if (detail != null) {
                    "mismatch_vs_deduced: $detail (givens-only deduction forces a different value)"
                } else {
                    "mismatch_vs_deduced: givens-only deduction forces a different value here"
                }
                return NextCheck(mismatch, "mismatch", reason)
            }

            // Case 2 — solvability == none: pick from unresolved only, excluding confirmed
            if (gridState.solvability == "none") {
                val unresolved = gridState.unresolvedCells.filterNot { isConfirmed(it) }
                if (unresolved.isNotEmpty()) {

                    fun isSolution(idx: Int) = gridState.truthIsSolution[idx]
                    fun isGiven(idx: Int) = gridState.truthIsGiven[idx]

                    val ordered = unresolved.sortedWith(
                        compareByDescending<Int> { conflictCountByIdx[it] }
                            .thenByDescending { if (isSolution(it)) 1 else 0 }
                            .thenByDescending { if (isGiven(it)) 1 else 0 }
                            .thenBy { it }
                    )

                    val idx = ordered.first()
                    val cc = conflictCountByIdx[idx]
                    val cr = conflictReasonForIdx(idx)

                    val reason = when {
                        cc > 0 && cr != null -> "unresolved + conflict($cc): $cr"
                        cc > 0 -> "unresolved + conflict($cc): involved in contradictions"
                        else -> "unresolved: needs verification on paper / scan uncertainty"
                    }

                    val pri = if (cc > 0) "conflict" else "unresolved"
                    return NextCheck(idx, pri, reason)
                }
                return null
            }

            // Case 3 — unique and no mismatch: no next check
            if (gridState.solvability == "unique") return null

            // Case 4 — multiple: no next check (user-driven reconcile / retake)
            return null
        }

        val next = pickRecommendedNextCheck()

        val nextCheckBlock = buildString {
            appendLine("RECOMMENDED_NEXT_CHECK:")
            if (next == null) {
                appendLine("- none")
            } else {
                appendLine("- cell: ${rc(next.idx)} idx=${next.idx}")
                appendLine("  priority: ${next.priority}")
                appendLine("  reason: ${next.reason}")
            }
        }.trimEnd()

        val nextActionBlock = buildString {
            appendLine("RECOMMENDED_NEXT_ACTION:")
            when {
                next != null -> {
                    val r = next.idx / 9 + 1
                    val c = next.idx % 9 + 1
                    appendLine("- ask_confirm_cell_rc(row=$r, col=$c)")
                }

                gridState.solvability == "unique" && gridState.mismatchCells.isEmpty() -> {
                    appendLine("- recommend_validate  # ask user if on-screen grid matches paper; do not propose corrections")
                }

                gridState.solvability == "multiple" -> {
                    appendLine("- recommend_validate  # ask user for 100% match; if match and still multiple -> recommend_retake")
                }

                gridState.solvability == "none" -> {
                    appendLine("- recommend_retake_soft  # no unresolved targets provided; ask for retake or user-driven edit")
                }

                else -> appendLine("- noop")
            }
        }.trimEnd()

        val multipleGuidanceBlock = buildString {
            appendLine("MULTIPLE_SOLUTIONS_GUIDANCE:")
            if (gridState.solvability == "multiple") {
                appendLine("- The current on-screen grid admits multiple solutions.")
                appendLine("- Do NOT ask cell-by-cell checks.")
                appendLine("- Ask user if the on-screen grid is a 100% match with the paper.")
                appendLine("- If user says YES and solvability remains multiple -> recommend retake (Sudo solves only unique grids).")
                appendLine("- If user says NO -> user will identify mismatching cells; apply_user_edit_rc on request.")
            } else {
                appendLine("- (n/a)")
            }
        }.trimEnd()

        return buildString {
            appendLine("GRID_CONTEXT_VERSION=v4.2")
            appendLine("GRID_DIM=9x9 CELLS=81")
            appendLine()

            appendLine("CURRENT_DISPLAY_DIGITS_0_MEANS_BLANK:")
            appendLine(rows)
            appendLine()

            appendLine("TRUTH_LAYER_GIVENS (FACTS / DNA):")
            appendLine(givensRows)
            appendLine()

            appendLine("TRUTH_LAYER_USER_SOLUTIONS (USER CLAIMS / OPINIONS):")
            appendLine(userSolRows)
            appendLine()

            appendLine("CANDIDATES (USER THOUGHT PROCESS; MAY BE NOISY):")
            appendLine(candLines)
            appendLine()

            appendLine("CONFIRMATION_FACTS (USER CONFIRMED THESE CELLS; DO NOT RE-ASK):")
            appendLine("- confirmed_indices: $confirmed")
            appendLine("- confirmed_count: ${confirmed.size}")
            appendLine()

            appendLine("SETS_0BASED_INDICES:")
            appendLine("- unresolved_indices: ${gridState.unresolvedCells}")
            appendLine("- auto_changed_indices: ${gridState.changedCells}")
            appendLine("- conflict_indices: ${gridState.conflictCells}")
            appendLine("- low_confidence_indices: ${gridState.lowConfidenceCells}")
            appendLine("- manual_corrected_indices: ${gridState.manuallyCorrectedCells}")
            appendLine("- mismatch_indices_vs_deduced (only if unique): ${gridState.mismatchCells}")
            appendLine()

            appendLine("CONFLICTS_DETAILS (authoritative explanations from CURRENT_DISPLAY):")
            appendLine(conflictDetailLines)
            appendLine()

            appendLine(nextCheckBlock)
            appendLine(nextActionBlock)
            appendLine(multipleGuidanceBlock)
            appendLine()

            appendLine("MANUAL_EDIT_HISTORY (most recent last):")
            if (gridState.manualEdits.isEmpty()) {
                appendLine("- (none)")
            } else {
                gridState.manualEdits.sortedBy { it.seq }.forEach { e ->
                    appendLine("- seq=${e.seq} cell=${e.cellLabel} idx=${e.index} from=${e.fromDigit} to=${e.toDigit} t=${e.whenEpochMs} source=${e.source}")
                }
            }
            appendLine()

            appendLine("COUNTS:")
            appendLine("- givens_count: $givenCount")
            appendLine("- user_solutions_count: $solCount")
            appendLine("- unresolved_count: ${gridState.unresolvedCells.size}")
            appendLine("- auto_changed_count: ${gridState.changedCells.size}")
            appendLine("- conflict_count: ${gridState.conflictCells.size}")
            appendLine("- low_confidence_count: ${gridState.lowConfidenceCells.size}")
            appendLine("- manual_corrected_count: ${gridState.manuallyCorrectedCells.size}")
            appendLine("- candidate_marks_total: $totalCandidateMarks")
            appendLine("- candidate_marks_by_row (r1..r9): ${candByRow.toList()}")
            appendLine("- candidate_marks_by_col (c1..c9): ${candByCol.toList()}")
            appendLine()

            appendLine("SOLVER_FACTS_FROM_GIVENS_ONLY:")
            appendLine("- deduced_solution_count_capped: ${gridState.deducedSolutionCountCapped}  # 0,1,2 (2 means 2+)")
            appendLine("- deduced_is_unique: ${gridState.deducedSolutionGrid != null}")
            if (gridState.deducedSolutionGrid != null && deducedRows != null) {
                appendLine("DEDUCED_UNIQUE_SOLUTION_GRID:")
                appendLine(deducedRows)
                if (gridState.mismatchDetails.isNotEmpty()) {
                    appendLine("MISMATCH_DETAILS (expected vs user_solution):")
                    gridState.mismatchDetails.take(40).forEach { appendLine("- $it") }
                }
            }
            appendLine()

            appendLine("STATUS:")
            appendLine("- solvability_of_current_display: ${gridState.solvability}      # unique|multiple|none")
            appendLine("- is_structurally_valid: ${gridState.isStructurallyValid}")
            appendLine("- severity: ${gridState.severity}            # ok|mild|serious")
            appendLine("- retake_recommendation: ${gridState.retakeRecommendation} # none|soft|strong")
            appendLine()

            appendLine("STRICT FACTUALITY RULES (HARD):")
            appendLine("- CONFLICTS: You may ONLY claim a contradiction if it appears in CONFLICTS_DETAILS above.")
            appendLine("- If CONFLICTS_DETAILS says (none), you MUST NOT claim any duplicate/contradiction.")
            appendLine("- NEXT CHECK POLICY: You may ONLY ask a next-check cell if it is in mismatch_indices_vs_deduced OR unresolved_indices AND it is NOT in confirmed_indices.")
            appendLine("- For solvability==multiple: do NOT drive cell-by-cell; ask for 100% match; if match and still multiple => retake.")
            appendLine("- For solvability==unique and mismatch empty: stop corrections; offer grid as-is; ask if it matches paper.")
        }.trim()
    }
}