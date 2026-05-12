package com.contextionary.sudoku.conductor.policy

import com.contextionary.sudoku.SessionStore
import com.contextionary.sudoku.conductor.GridPhase
import com.contextionary.sudoku.conductor.LlmPolicy
import com.contextionary.sudoku.conductor.PolicyCallCtx
import com.contextionary.sudoku.conductor.SudoMode
import com.contextionary.sudoku.conductor.ToolCall
import com.contextionary.sudoku.logic.LLMGridState
import com.contextionary.sudoku.logic.ModelCallTelemetryCtx
import com.contextionary.sudoku.logic.SudokuLLMConversationCoordinator
import com.contextionary.sudoku.telemetry.ConversationTelemetry
import org.json.JSONObject

import com.contextionary.sudoku.conductor.policy.ReplyDemandContractsV1

import com.contextionary.sudoku.conductor.policy.ReplySupplyProjectorsV1

/**
 * FM-01 / FM-02 hard rule:
 * - This adapter MUST NOT author any user-facing text.
 * - No fallback prompts, no default Reply strings, no "snag" lines.
 * - If the LLM response is malformed or empty, return null/emptyList() and let caller retry.
 *
 * ID OWNERSHIP (HARD):
 * - ALL authoritative chain IDs are ctx-owned by the caller (Activity/EffectRunner).
 * - This adapter must NOT invent modelCallId/toolplanId/policyReqSeq/etc.
 * - If required IDs are missing, return null/emptyList() (caller may retry/bound/fallback).
 *
 * Phase 6 (canonical):
 * - Tick1 NLU output is ONLY: IntentEnvelopeV1 (intent_envelope_v1).
 * - MeaningExtractV1 is retired at the adapter boundary (no shim).
 * - Tick2 NLG output is: ReplyGenerateV1 (spoken reply always LLM-generated).
 *
 * Legacy tool-schema planning/continuation is decommissioned:
 * - decide()/continueTick2() MUST return emptyList() and emit traces.
 */
class CoordinatorPolicyAdapter(
    private val coord: SudokuLLMConversationCoordinator,
    private val systemPrompt: String
) : LlmPolicy {

    // -------------------------------------------------
    // Diagnostics side-channel (Option 2)
    // -------------------------------------------------

    @Volatile private var lastDiag: ToolplanDiagnostics? = null
    @Volatile private var lastDiagTickId: Int? = null

    // Tick-1 output side-channel (IntentEnvelopeV1)
    @Volatile private var lastIntentEnvelope: IntentEnvelopeV1? = null
    @Volatile private var lastIntentEnvelopeTickId: Int? = null

    /** Consume the last diagnostics (one-shot). */
    fun consumeLastDiag(): ToolplanDiagnostics? {
        val d = lastDiag
        lastDiag = null
        lastDiagTickId = null
        return d
    }

    /** Consume the last IntentEnvelopeV1 (one-shot). */
    fun consumeLastIntentEnvelope(): IntentEnvelopeV1? {
        val e = lastIntentEnvelope
        lastIntentEnvelope = null
        lastIntentEnvelopeTickId = null
        return e
    }

    /** Non-destructive peek (useful for debugging). */
    fun peekLastIntentEnvelope(): IntentEnvelopeV1? = lastIntentEnvelope

    private fun stashIntentEnvelope(tickId: Int, env: IntentEnvelopeV1?) {
        if (env == null) return
        lastIntentEnvelope = env
        lastIntentEnvelopeTickId = tickId
    }

    /** Non-destructive peek (useful for debugging). */
    fun peekLastDiag(): ToolplanDiagnostics? = lastDiag

    private fun stashDiag(tickId: Int, diag: ToolplanDiagnostics?) {
        if (diag == null) return
        lastDiag = diag
        lastDiagTickId = tickId
    }

    // -----------------------------
    // Shared helpers
    // -----------------------------


    internal fun replyDemandContractFor(
        demand: ReplyDemandResolutionV1
    ): ReplyDemandContractV1 {
        return ReplyDemandContractsV1.forDemand(demand)
    }

    /**
     * Phase 5 — projector access shim.
     *
     * This keeps the projector layer discoverable from the policy adapter side
     * before the full assembly planner exists.
     */
    internal fun projectReplySupplyChannel(
        replyRequest: ReplyRequestV1,
        channel: ReplySupplyChannelV1
    ): JSONObject {
        return ReplySupplyProjectorsV1.projectChannel(replyRequest, channel)
    }

    /**
     * Phase 6 — central planner shim for later rollout.
     */
    internal fun planReplyAssembly(
        demand: ReplyDemandResolutionV1,
        replyRequest: ReplyRequestV1
    ): ReplyAssemblyPlannerV1.PlanResultV1 {
        return ReplyAssemblyPlannerV1.plan(
            demand = demand,
            replyRequest = replyRequest
        )
    }

    private fun countQuestions(text: String): Int =
        Regex("""\?""").findAll(text).count()

    private fun capWords(text: String, maxWords: Int): String {
        if (maxWords <= 0) return text.trim()
        val words = text.trim().split(Regex("""\s+""")).filter { it.isNotBlank() }
        if (words.size <= maxWords) return text.trim()
        return words.take(maxWords).joinToString(" ").trimEnd() + "…"
    }

    private fun stripForbiddenMeta(text: String): String {
        val forbidden = listOf(
            "tick", "schema", "tool", "toolplan", "id=", "model", "json", "telemetry",
            "timeout", "retry", "network", "llm", "as an ai"
        )
        var t = text
        forbidden.forEach { w ->
            t = t.replace(Regex("""\b${Regex.escape(w)}\b""", RegexOption.IGNORE_CASE), "")
        }
        t = t.replace(Regex("""\s{2,}"""), " ").trim()
        return t
    }

    private fun enforceOneQuestion(text: String): String {
        return text.trim()
    }

    /**
     * Phase-5 reply contract enforcement:
     * - <= 1 question
     * - <= maxWords (if provided)
     * - remove internal/meta/tool/tick mentions
     * - optional: SOLVING mode ban on “validation talk”
     */
    private fun enforceReplyContract(
        textIn: String,
        maxWords: Int?,
        phase: String?
    ): String {
        var t = textIn.trim()
        t = stripForbiddenMeta(t)

        if (phase != null && phase.equals("SOLVING", ignoreCase = true)) {
            val banned = listOf("validate", "validation", "match your paper", "matches your paper", "confirm the grid")
            banned.forEach { b ->
                t = t.replace(Regex(Regex.escape(b), RegexOption.IGNORE_CASE), "")
            }
            t = t.replace(Regex("""\s{2,}"""), " ").trim()
        }

        if (maxWords != null && maxWords > 0) {
            t = capWords(t, maxWords)
        }

        return t
    }

    private fun emitTrace(tag: String, data: Map<String, Any?>) {
        runCatching { ConversationTelemetry.emitPolicyTrace(tag = tag, data = data) }
    }

    private fun sha12(s: String): String =
        runCatching { ConversationTelemetry.sha256Hex(s).take(12) }.getOrElse { "sha_err" }

    private fun preview(s: String, max: Int = 900): String =
        if (s.length <= max) s else s.take(max)

    private fun parseLineRawValue(stateHeader: String, key: String): String? {
        return stateHeader
            .lineSequence()
            .map { it.trim() }
            .firstOrNull { it.startsWith("$key=", ignoreCase = true) }
            ?.substringAfter("=", "")
            ?.trim()
    }

    private fun parseLineValue(stateHeader: String, key: String): String? {
        return parseLineRawValue(stateHeader, key)
            ?.takeIf { it.isNotBlank() && !it.equals("none", ignoreCase = true) }
    }

    private fun parseModeFromStateHeader(stateHeader: String, isGridSession: Boolean): String {
        return parseLineValue(stateHeader, "mode")
            ?: if (isGridSession) "GRID_SESSION" else "UNKNOWN"
    }

    private fun parsePendingFromHeader(stateHeader: String): String? {
        return runCatching {
            // 1) Preferred machine line: pending=OTHER_SolvePreference
            parseLineValue(stateHeader, "pending")?.let { token ->
                return@runCatching token
            }

            // 2) Fallback: human line at end, e.g. "pending:other type=SolvePreference"
            val human = stateHeader.lineSequence()
                .map { it.trim() }
                .firstOrNull { it.startsWith("pending:", ignoreCase = true) }
                ?: return@runCatching null

            val kindToken = human.substringAfter("pending:", "").substringBefore(' ').trim()
            if (kindToken.isBlank()) return@runCatching null

            if (!kindToken.equals("other", ignoreCase = true)) {
                return@runCatching kindToken.uppercase()
            }

            val typeName = human.substringAfter("type=", "").substringBefore(' ').trim()
            if (typeName.isBlank()) return@runCatching "OTHER_UNKNOWN"
            "OTHER_$typeName"
        }.getOrNull()
    }

    private fun parseFocusIdxFromHeader(stateHeader: String): Int? {
        return parseLineValue(stateHeader, "focus")?.toIntOrNull()
    }

    private fun parsePendingExpectedKindFromHeader(stateHeader: String): String? =
        parseLineValue(stateHeader, "pending_expected_answer_kind")

    private fun parsePendingTargetCellFromHeader(stateHeader: String): String? =
        parseLineValue(stateHeader, "pending_target_cell")

    private fun parseFocusCellFromHeader(stateHeader: String): String? =
        parseLineValue(stateHeader, "focus_cell")


    private fun parseCanonicalSolvingPositionKindFromHeader(stateHeader: String): String? =
        parseLineValue(stateHeader, "canonical_solving_position_kind")
            ?: parseLineValue(stateHeader, "canonical_position_kind")

    private fun parseLastQuestionKeyFromHeader(stateHeader: String): String? =
        parseLineValue(stateHeader, "last_assistant_question_key")

    private fun computeHeaderSha12(stateHeader: String): String? {
        return runCatching {
            ConversationTelemetry.sha256Hex(stateHeader).take(12)
        }.getOrNull()
    }

    private fun parsePhaseFromStateHeader(stateHeader: String): GridPhase {
        val v = stateHeader.lineSequence()
            .map { it.trim() }
            .firstOrNull { it.startsWith("grid_phase=", ignoreCase = true) || it.startsWith("phase=", ignoreCase = true) }
            ?.substringAfter("=", "")
            ?.trim()
            ?.uppercase()
            ?: return GridPhase.CONFIRMING

        return runCatching { GridPhase.valueOf(v) }.getOrElse { GridPhase.CONFIRMING }
    }

    private fun requireCtxOwnedIdsOrEmpty(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        correlationId: String,
        policyReqSeq: Long,
        modelCallId: String?,
        toolplanId: String?,
        isGridSession: Boolean,
        stateHeader: String,
        userText: String? = null,
        reason: String
    ): PolicyCallCtx? {
        val mc = modelCallId?.trim().takeUnless { it.isNullOrEmpty() }
        val tp = toolplanId?.trim().takeUnless { it.isNullOrEmpty() }

        if (mc == null || tp == null) {
            emitTrace(
                tag = "CTX_IDS_MISSING",
                data = buildMap {
                    put("session_id", sessionId)
                    put("is_grid_session", isGridSession)
                    put("reason", reason)
                    put("turn_id", turnId)
                    put("tick_id", tickId)
                    put("policy_req_seq", policyReqSeq)
                    put("correlation_id", correlationId)
                    put("missing_model_call_id", mc == null)
                    put("missing_toolplan_id", tp == null)
                    put("state_header_preview", stateHeader.take(220))
                    if (userText != null) put("user_preview", userText.take(160))
                }
            )
            return null
        }

        val headerSha12 = computeHeaderSha12(stateHeader)
        val modeStr = parseModeFromStateHeader(stateHeader, isGridSession)

        return PolicyCallCtx(
            sessionId = sessionId,
            turnId = turnId,
            tickId = tickId,
            policyReqSeq = policyReqSeq,
            correlationId = correlationId,
            modelCallId = mc,
            toolplanId = tp,
            mode = modeStr,
            reason = reason,
            stateHeaderSha12 = headerSha12
        )
    }

    // ---------------------------------------------
    // ✅ Phase 6: Tick 1 NLU (canonical) — IntentEnvelopeV1 only
    // ---------------------------------------------
    suspend fun decideIntentEnvelopeV1(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        correlationId: String,
        policyReqSeq: Long,
        modelCallId: String?,
        toolplanId: String?,
        userText: String,
        stateHeader: String
    ): IntentEnvelopeV1? {

        val isGridSession = stateHeader.contains("mode=GRID_SESSION")

        fun early(reason: String, extra: Map<String, Any?> = emptyMap()): IntentEnvelopeV1? {
            emitTrace(
                tag = "INTENT_ENVELOPE_V1_EMPTY_RETURNED",
                data = mapOf(
                    "session_id" to sessionId,
                    "is_grid_session" to isGridSession,
                    "reason" to reason,
                    "turn_id" to turnId,
                    "tick_id" to tickId,
                    "policy_req_seq" to policyReqSeq,
                    "correlation_id" to correlationId,
                    "model_call_id_in" to (modelCallId ?: ""),
                    "toolplan_id_in" to (toolplanId ?: ""),
                    "user_preview" to userText.take(160),
                    "state_header_preview" to stateHeader.take(220)
                ) + extra
            )
            return null
        }

        val ctx = requireCtxOwnedIdsOrEmpty(
            sessionId = sessionId,
            turnId = turnId,
            tickId = tickId,
            correlationId = correlationId,
            policyReqSeq = policyReqSeq,
            modelCallId = modelCallId,
            toolplanId = toolplanId,
            isGridSession = isGridSession,
            stateHeader = stateHeader,
            userText = userText,
            reason = "tick1_intent_envelope_v1"
        ) ?: return early("ctx_ids_missing")

        val phase = parsePhaseFromStateHeader(stateHeader)
        val modeStr = parseModeFromStateHeader(stateHeader, isGridSession)
        val pendingBefore = parsePendingFromHeader(stateHeader)
        val focusIdx = parseFocusIdxFromHeader(stateHeader)

        val pendingExpectedKind: String? = parsePendingExpectedKindFromHeader(stateHeader)
        val pendingTargetCell: String? = parsePendingTargetCellFromHeader(stateHeader)
        val focusCell: String? = parseFocusCellFromHeader(stateHeader)
        val lastAssistantQuestionKey: String? = parseLastQuestionKeyFromHeader(stateHeader)
        val canonicalSolvingPositionKind: String? =
            parseCanonicalSolvingPositionKindFromHeader(stateHeader)

        emitTrace(
            tag = "TURN_HEADER_PARSED_TICK1",
            data = mapOf(
                "session_id" to sessionId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "policy_req_seq" to policyReqSeq,
                "correlation_id" to correlationId,
                "model_call_id_in" to (modelCallId ?: ""),
                "toolplan_id_in" to (toolplanId ?: ""),
                "mode_parsed" to modeStr,
                "phase_parsed" to phase.name,
                "pending_before_parsed" to (pendingBefore ?: ""),
                "pending_expected_answer_kind_parsed" to (pendingExpectedKind ?: ""),

                "pending_target_cell_parsed" to (pendingTargetCell ?: ""),
                "focus_cell_parsed" to (focusCell ?: ""),
                "last_assistant_question_key_parsed" to (lastAssistantQuestionKey ?: ""),
                "canonical_solving_position_kind_parsed" to (canonicalSolvingPositionKind ?: ""),
                "state_header_sha12" to (computeHeaderSha12(stateHeader) ?: ""),
                "state_header_preview" to stateHeader.take(260)

            )
        )

        val telemetryCtx = ModelCallTelemetryCtx(
            modelCallId = ctx.modelCallId,
            turnId = ctx.turnId,
            tickId = ctx.tickId,
            policyReqSeq = ctx.policyReqSeq,
            toolplanId = ctx.toolplanId,
            correlationId = ctx.correlationId
        )

        // ---- Turn context (TO-BE) -> LLM ----
        val baseTurnCtx = SessionStore.snapshotTurnContextV1(
            turnId = turnId,
            mode = modeStr,
            phase = phase.name,
            userText = userText,
            pendingBefore = pendingBefore,
            pendingExpectedAnswerKind = pendingExpectedKind,
            pendingTargetCell = pendingTargetCell,
            focusCell = focusCell,
            lastAssistantQuestionKey = lastAssistantQuestionKey,
            canonicalSolvingPositionKind = canonicalSolvingPositionKind
        )

        // Series 6: adaptive transcript window for Tick-1 driven by structured pre-hints.
        val transcriptHints =
            SessionStore.derivePreTick1TranscriptContextHintsV1(
                pendingBefore = pendingBefore,
                lastAssistantQuestionKey = lastAssistantQuestionKey
            )

        val adaptiveTurns = SessionStore.getRecentTurnsAdaptive(
            hints = transcriptHints
        )
        val adaptiveTurnsJson = TranscriptTurnV1.jsonArray(adaptiveTurns).toString()

        // Override only recentTurnsJson for Tick-1 payload (keep tallies identical)
        val turnCtx = baseTurnCtx.copy(recentTurnsJson = adaptiveTurnsJson)

        val turnCtxJson = turnCtx.toJsonString()

        // (compat) keep these vars for existing coordinator signature + telemetry previews
        val userTallyJson = turnCtx.userTallyJson
        val assistantTallyJson = turnCtx.assistantTallyJson
        val recentTurnsJson = turnCtx.recentTurnsJson

        // ---- Request payload trace (audit visibility) ----
        val envReqJson = runCatching {
            JSONObject().apply {
                put("kind", "intent_envelope_v1")
                put("mode", modeStr)
                put("phase", phase.name)
                put("pending_before", pendingBefore ?: "")
                put("pending_expected_answer_kind", pendingExpectedKind ?: "")

                put("pending_target_cell", pendingTargetCell ?: "")
                put("focus_cell", focusCell ?: "")
                put("last_assistant_question_key", lastAssistantQuestionKey ?: "")
                put("canonical_solving_position_kind", canonicalSolvingPositionKind ?: "")

                // TO-BE: Tick1 user message is a single TurnContextV1 JSON payload
                put("user_message_kind", "TurnContextV1")

                put("turn_ctx_json_preview", preview(turnCtxJson, 700))
                put("raw_user_text_preview", userText.take(220))

                put("user_tally_json_preview", preview(userTallyJson, 300))
                put("assistant_tally_json_preview", preview(assistantTallyJson, 300))
                put("recent_turns_json_preview", preview(recentTurnsJson, 300))

                put("state_header_sha12", computeHeaderSha12(stateHeader) ?: "")
                put("state_header_preview", stateHeader.take(400))
            }.toString()
        }.getOrElse { "intent_env_req_json_error:${it.javaClass.simpleName}" }




        emitTrace(
            tag = "POLICY_MODEL_REQUEST_PREVIEW_OUT",
            data = mapOf(
                "session_id" to sessionId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "policy_req_seq" to policyReqSeq,
                "correlation_id" to correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "payload_kind" to "intent_envelope_v1",
                "payload_sha12" to sha12(envReqJson),
                "payload_len" to envReqJson.length,
                "payload_preview" to preview(envReqJson, 900)
            )
        )

        // ---- Call LLM ----
        val env: IntentEnvelopeV1 = try {
            coord.sendIntentEnvelopeV1(
                mode = modeStr,
                phase = phase.name,
                pendingBefore = pendingBefore,
                pendingExpectedAnswerKind = pendingExpectedKind,
                pendingTargetCell = pendingTargetCell,
                focusCell = focusCell,
                lastAssistantQuestionKey = lastAssistantQuestionKey,
                userText = turnCtxJson,
                userTallyJson = userTallyJson,
                assistantTallyJson = assistantTallyJson,
                recentTurnsJson = recentTurnsJson,
                discourseStateJson = SessionStore.snapshotDiscourseStateJson(),
                telemetryCtx = telemetryCtx
            )
        } catch (t: Throwable) {
            emitTrace(
                tag = "POLICY_MODEL_RESPONSE_PREVIEW_IN",
                data = mapOf(
                    "session_id" to sessionId,
                    "turn_id" to turnId,
                    "tick_id" to tickId,
                    "policy_req_seq" to policyReqSeq,
                    "correlation_id" to correlationId,
                    "model_call_id" to ctx.modelCallId,
                    "toolplan_id" to ctx.toolplanId,
                    "payload_kind" to "intent_envelope_v1",
                    "parse_ok" to false,
                    "error_type" to (t.javaClass.simpleName ?: "Throwable"),
                    "error_msg" to (t.message?.take(220) ?: "")
                )
            )
            return early(
                "llm_call_failed",
                extra = mapOf(
                    "throwable" to (t.javaClass.simpleName ?: "Throwable"),
                    "message" to (t.message?.take(180) ?: "")
                )
            )
        }

        // Always emit MODEL_PAYLOAD_IN on success too

        // Always emit MODEL_PAYLOAD_IN on success too
        val envJson = runCatching {
            org.json.JSONObject().apply {
                put("version", env.version)

                // intents[]
                put("intents", org.json.JSONArray().apply {
                    env.intents.forEach { i ->
                        put(org.json.JSONObject().apply {
                            put("id", i.id)
                            put("type", i.type.name)
                            put("confidence", i.confidence)

                            // targets[]
                            put("targets", org.json.JSONArray().apply {
                                i.targets.forEach { t ->
                                    put(t.toJson())
                                }
                            })

                            // payload
                            put("payload", i.payload.toJson())

                            // missing[]
                            put("missing", org.json.JSONArray().apply {
                                i.missing.forEach { put(it) }
                            })

                            // optional
                            put("evidence_text", i.evidenceText ?: org.json.JSONObject.NULL)
                            put("addresses_user_agenda_id", i.addressesUserAgendaId ?: org.json.JSONObject.NULL)
                            put("reference_resolution_mode", i.referenceResolutionMode?.name ?: org.json.JSONObject.NULL)
                        })
                    }
                })

                // free_talk (keep stable shape for telemetry readability)
                put("free_talk", org.json.JSONObject().apply {
                    put("topic", env.freeTalkTopic ?: org.json.JSONObject.NULL)
                    put("confidence", env.freeTalkConfidence)
                })

                put("repair_signal", env.repairSignal?.name ?: org.json.JSONObject.NULL)
                put("context_span_hint", env.contextSpanHint?.name ?: org.json.JSONObject.NULL)
                put("references_prior_turns", env.referencesPriorTurns ?: org.json.JSONObject.NULL)

                // notes (keep stable shape)
                put("notes", org.json.JSONObject().apply {
                    put("raw_user_text", env.rawUserText ?: org.json.JSONObject.NULL)
                    put("language", env.language ?: org.json.JSONObject.NULL)
                    put("asr_quality", env.asrQuality ?: org.json.JSONObject.NULL)
                })
            }.toString()
        }.getOrElse { "" }


        emitTrace(
            tag = "POLICY_MODEL_RESPONSE_PREVIEW_IN",
            data = mapOf(
                "session_id" to sessionId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "policy_req_seq" to policyReqSeq,
                "correlation_id" to correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "payload_kind" to "intent_envelope_v1",
                "parse_ok" to true,
                "response_sha12" to sha12(envJson),
                "response_len" to envJson.length,
                "payload_preview" to preview(envJson, 900)
            )
        )

        val intentTypes = env.intents.map { it.type.name }.distinct().sorted()
        val hasAnyCellTarget = env.intents.any { i -> i.targets.any { !it.cell.isNullOrBlank() } }
        val hasAnyRegionTarget = env.intents.any { i -> i.targets.any { it.region != null } }
        val hasAnyDigit = env.intents.any { i ->
            (i.payload.digit != null) ||
                    ((i.payload.digits ?: emptyList()).isNotEmpty()) ||
                    ((i.payload.regionDigits ?: "").isNotBlank())
        }
        val missingTotal = env.intents.sumOf { it.missing.size }

        val topIntentType = env.intents.firstOrNull()?.type
        val topIntentBucket =
            topIntentType?.let { agendaIntentConstitutionBucketV1(it).name } ?: "NONE"

        val directAppOwnedCount =
            env.intents.count { agendaIntentConstitutionBucketV1(it.type) == AgendaIntentConstitutionBucketV1.DIRECT_APP_OWNED }
        val userDetourCount =
            env.intents.count { agendaIntentConstitutionBucketV1(it.type) == AgendaIntentConstitutionBucketV1.USER_DETOUR }
        val userRouteJumpCount =
            env.intents.count { agendaIntentConstitutionBucketV1(it.type) == AgendaIntentConstitutionBucketV1.USER_ROUTE_JUMP }
        val repairCandidateCount =
            env.intents.count { agendaIntentConstitutionBucketV1(it.type) == AgendaIntentConstitutionBucketV1.REPAIR_CANDIDATE }

        emitTrace(
            tag = "INTENT_ENVELOPE_V1_OK",
            data = mapOf(
                "session_id" to sessionId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "policy_req_seq" to policyReqSeq,
                "correlation_id" to correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "intents_n" to env.intents.size,
                "intent_types" to intentTypes.joinToString(","),
                "missing_total" to missingTotal,
                "has_any_cell_target" to hasAnyCellTarget,
                "has_any_region_target" to hasAnyRegionTarget,
                "has_any_digit" to hasAnyDigit,

                // Series 8: constitutional telemetry
                "top_intent_type" to (topIntentType?.name ?: "UNKNOWN"),
                "top_intent_constitution_bucket" to topIntentBucket,
                "direct_app_owned_intents_n" to directAppOwnedCount,
                "user_detour_intents_n" to userDetourCount,
                "user_route_jump_intents_n" to userRouteJumpCount,
                "repair_candidate_intents_n" to repairCandidateCount,

                "free_talk_conf" to env.freeTalkConfidence,
                "focus_idx" to focusIdx,
                "user_preview" to userText.take(160)
            )
        )

        // Keep the (existing) “first speech” rule: adapter ensures SessionStore has it.
        SessionStore.ensureFirstUserSpeech(userText)

        stashIntentEnvelope(tickId, env)
        return env
    }

    suspend fun generateReplyV1(
        ctx: PolicyCallCtx,
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null
    ): String {

        emitTrace(
            tag = "REPLY_V1_START",
            data = mapOf(
                "session_id" to ctx.sessionId,
                "turn_id" to ctx.turnId,
                "tick_id" to 2,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "mode" to ctx.mode,
                "reason" to (ctx.reason ?: "")
            )
        )

        val telemetryCtx = ModelCallTelemetryCtx(
            modelCallId = ctx.modelCallId,
            turnId = ctx.turnId,
            tickId = 2,
            policyReqSeq = ctx.policyReqSeq,
            toolplanId = ctx.toolplanId,
            correlationId = ctx.correlationId
        )

        val replyReqJson: String = runCatching { replyRequest.toJsonString() }
            .getOrElse { replyRequest.toJson().toString() }



        // ✅ Phase 0: Turn header/pending summary as seen by Tick2 (derived from ReplyRequestV1 fields)
        runCatching {
            val t = replyRequest.turn
            val d = replyRequest.decision

            emitTrace(
                tag = "TURN_HEADER_PARSED_TICK2",
                data = mapOf(
                    "session_id" to ctx.sessionId,
                    "turn_id" to ctx.turnId,
                    "tick_id" to 2,
                    "policy_req_seq" to ctx.policyReqSeq,
                    "correlation_id" to ctx.correlationId,
                    "model_call_id" to ctx.modelCallId,
                    "toolplan_id" to ctx.toolplanId,

                    "mode" to (t.mode ?: ""),
                    "phase" to (t.phase ?: ""),

                    "pending_before" to (t.pendingBefore ?: ""),
                    "pending_after" to (t.pendingAfter ?: ""),
                    "focus_before" to (t.focusBefore?.toString() ?: ""),
                    "focus_after" to (t.focusAfter?.toString() ?: ""),

                    // ✅ Story header visibility (Phase 1)
                    "story_present" to (t.story?.present?.toString() ?: ""),
                    "story_stage" to (t.story?.stage ?: ""),
                    "story_step_id" to (t.story?.stepId ?: ""),
                    "story_grid_hash12" to (t.story?.gridHash12 ?: ""),
                    "story_atoms_count" to (t.story?.atomsCount?.toString() ?: ""),
                    "story_focus_atom_index" to (t.story?.focusAtomIndex?.toString() ?: ""),
                    "story_discussed_atoms" to (t.story?.discussedAtomIndices?.joinToString(",") ?: ""),
                    "story_ready_for_commit" to (t.story?.readyForCommit?.toString() ?: ""),

                    "decision_kind" to (d.decisionKind ?: ""),
                    "decision_summary" to (d.summary ?: ""),

                    // Helpful correlation with your existing OUT payload
                    "reply_request_sha12" to sha12(replyReqJson),
                    "reply_request_len" to replyReqJson.length
                )
            )
        }



        // Phase 0: trace presence of frozen Roadmap v1 contracts in Tick2 input (debug-only).
        runCatching {
            val root = JSONObject(replyReqJson)

            // Different versions may name this array differently; check both.
            val factsArr = root.optJSONArray("facts")
                ?: root.optJSONArray("fact_bundles")
                ?: root.optJSONArray("factBundles")
                ?: root.optJSONArray("factBundlesV1")

            fun hasType(typeName: String): Boolean {
                if (factsArr == null) return false
                for (i in 0 until factsArr.length()) {
                    val o = factsArr.optJSONObject(i) ?: continue
                    // Most of your bundles are { "type": "...", "payload": {...} }
                    val t = o.optString("type")
                    if (t.equals(typeName, ignoreCase = true)) return true
                }
                return false
            }


            val hasCta = hasType("CTA_PACKET_V1")
            val hasRecovery = hasType("RECOVERY_PACKET_V1")
            val ctaCount = (if (hasCta) 1 else 0) + (if (hasRecovery) 1 else 0)

            if (ctaCount > 1) {
                emitTrace(
                    "CTA_CONTRACT_VIOLATION_TICK2_INPUT",
                    mapOf(
                        "session_id" to ctx.sessionId,
                        "turn_id" to ctx.turnId,
                        "tick_id" to 2,
                        "policy_req_seq" to ctx.policyReqSeq,
                        "correlation_id" to ctx.correlationId,
                        "model_call_id" to ctx.modelCallId,
                        "toolplan_id" to ctx.toolplanId,
                        "cta_count" to ctaCount,
                        "has_cta_packet_v1" to hasCta,
                        "has_recovery_packet_v1" to hasRecovery
                    )
                )
            }

            emitTrace(
                "FACT_BUNDLES_V1_PRESENT",
                mapOf(
                    "session_id" to ctx.sessionId,
                    "turn_id" to ctx.turnId,
                    "tick_id" to 2,
                    "policy_req_seq" to ctx.policyReqSeq,
                    "correlation_id" to ctx.correlationId,
                    "model_call_id" to ctx.modelCallId,
                    "toolplan_id" to ctx.toolplanId,
                    "SOLVING_STEP_PACKET_V1" to hasType("SOLVING_STEP_PACKET_V1"),
                    "TEACHING_CARD_V1" to hasType("TEACHING_CARD_V1"),
                    "CTA_PACKET_V1" to hasType("CTA_PACKET_V1"),
                    "RECOVERY_PACKET_V1" to hasType("RECOVERY_PACKET_V1")
                )
            )
        }


        emitTrace(
            tag = "REPLY_REQUEST_V1_OUT",
            data = mapOf(
                "session_id" to ctx.sessionId,
                "turn_id" to ctx.turnId,
                "tick_id" to 2,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "payload_sha12" to sha12(replyReqJson),
                "payload_len" to replyReqJson.length,
                "payload_preview" to preview(replyReqJson, 1000)
            )
        )

        emitTrace(
            tag = "POLICY_MODEL_REQUEST_PREVIEW_OUT",
            data = mapOf(
                "session_id" to ctx.sessionId,
                "turn_id" to ctx.turnId,
                "tick_id" to 2,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "payload_kind" to "reply_generate_v1",
                "payload_sha12" to sha12(replyReqJson),
                "payload_len" to replyReqJson.length,
                "payload_preview" to preview(replyReqJson, 900)
            )
        )

        val replyText: String = try {
            emitTrace(
                tag = "REPLY_V1_PROMPT_LOCK",
                data = mapOf(
                    "session_id" to ctx.sessionId,
                    "turn_id" to ctx.turnId,
                    "tick_id" to 2,
                    "policy_req_seq" to ctx.policyReqSeq,
                    "correlation_id" to ctx.correlationId,
                    "model_call_id" to ctx.modelCallId,
                    "toolplan_id" to ctx.toolplanId,
                    "tick2_sys_sha12" to "locked_in_companion_conversation",
                    "tick2_dev_sha12" to "locked_in_companion_conversation"
                )
            )

            coord.sendReplyGenerateV1(
                replyRequest = replyRequest,
                planResult = planResult,
                telemetryCtx = telemetryCtx
            )
        } catch (t: Throwable) {
            emitTrace(
                tag = "POLICY_MODEL_RESPONSE_PREVIEW_IN",
                data = mapOf(
                    "session_id" to ctx.sessionId,
                    "turn_id" to ctx.turnId,
                    "tick_id" to 2,
                    "policy_req_seq" to ctx.policyReqSeq,
                    "correlation_id" to ctx.correlationId,
                    "model_call_id" to ctx.modelCallId,
                    "toolplan_id" to ctx.toolplanId,
                    "payload_kind" to "reply_generate_v1",
                    "parse_ok" to false,
                    "error_type" to (t.javaClass.simpleName ?: "Throwable"),
                    "error_msg" to (t.message?.take(220) ?: "")
                )
            )
            throw t
        }

        emitTrace(
            tag = "POLICY_MODEL_RESPONSE_PREVIEW_IN",
            data = mapOf(
                "session_id" to ctx.sessionId,
                "turn_id" to ctx.turnId,
                "tick_id" to 2,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "payload_kind" to "reply_generate_v1",
                "parse_ok" to true,
                "payload_sha12" to sha12(replyText),
                "payload_len" to replyText.length,
                "payload_preview" to preview(replyText, 600)
            )
        )

        emitTrace(
            tag = "REPLY_V1_OK",
            data = mapOf(
                "session_id" to ctx.sessionId,
                "turn_id" to ctx.turnId,
                "tick_id" to 2,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,
                "reply_len" to replyText.length,
                "reply_sha12" to sha12(replyText)
            )
        )

        val maxWords: Int? = runCatching { replyRequest.style.maxWords }.getOrNull()
        val phase: String? = runCatching { replyRequest.turn.phase }.getOrNull()

        val enforced = enforceReplyContract(
            textIn = replyText,
            maxWords = maxWords,
            phase = phase
        )

        if (enforced != replyText) {
            emitTrace(
                tag = "REPLY_V1_ENFORCED",
                data = mapOf(
                    "session_id" to ctx.sessionId,
                    "turn_id" to ctx.turnId,
                    "tick_id" to 2,
                    "policy_req_seq" to ctx.policyReqSeq,
                    "correlation_id" to ctx.correlationId,
                    "model_call_id" to ctx.modelCallId,
                    "toolplan_id" to ctx.toolplanId,
                    "orig_len" to replyText.length,
                    "new_len" to enforced.length,
                    "orig_sha12" to sha12(replyText),
                    "new_sha12" to sha12(enforced)
                )
            )
        }

        return enforced
    }

    // ---------------------------------------------
    // Phase 6: Legacy tool-schema API (hard-disabled)
    // ---------------------------------------------

    @Deprecated("Phase 6: tool-schema planning is decommissioned. Use decideIntentEnvelopeV1.")
    override suspend fun decide(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        correlationId: String,
        policyReqSeq: Long,
        modelCallId: String?,
        toolplanId: String?,
        userText: String,
        stateHeader: String,
        grid: LLMGridState?
    ): List<ToolCall> {

        emitTrace(
            tag = "LEGACY_TOOLSCHEMA_DECIDE_CALLED",
            data = mapOf(
                "session_id" to sessionId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "policy_req_seq" to policyReqSeq,
                "correlation_id" to correlationId,
                "model_call_id_in" to (modelCallId ?: ""),
                "toolplan_id_in" to (toolplanId ?: ""),
                "state_header_preview" to stateHeader.take(220),
                "user_preview" to userText.take(160)
            )
        )

        // Phase 6: never call LLM tool schema; return empty so caller must use V1 bridge.
        return emptyList()
    }

    @Deprecated("Phase 6: tool-schema Tick2 is decommissioned. Use generateReplyV1.")
    override suspend fun continueTick2(
        sessionId: String,
        turnId: Long,
        tickId: Int,
        correlationId: String,
        policyReqSeq: Long,
        modelCallId: String?,
        toolplanId: String?,
        systemPrompt: String,
        gridStateAfterTools: LLMGridState?,
        stateHeader: String,
        toolResults: List<String>,
        toolResultIds: List<String>,
        llm1ReplyText: String,
        grid: LLMGridState?,
        mode: SudoMode,
        reason: String
    ): List<ToolCall> {

        emitTrace(
            tag = "LEGACY_TOOLSCHEMA_TICK2_CALLED",
            data = mapOf(
                "session_id" to sessionId,
                "turn_id" to turnId,
                "tick_id" to tickId,
                "policy_req_seq" to policyReqSeq,
                "correlation_id" to correlationId,
                "model_call_id_in" to (modelCallId ?: ""),
                "toolplan_id_in" to (toolplanId ?: ""),
                "state_header_preview" to stateHeader.take(220),
                "tool_results_n" to toolResults.size
            )
        )

        // Phase 6: never call tool continuation; return empty.
        return emptyList()
    }
}